import io
import json
import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="CalorieAI Backend")

# Промпт для строгого JSON-ответа от Gemini
GEMINI_FOOD_PROMPT = """
Проанализируй это фото еды и выдай ответ СТРОГО в формате JSON без какого-либо дополнительного текста или разметки markdown (без ```json ... ```), используя следующие ключи:
{
  "dish_name": "Название блюда на русском языке",
  "calories": 000,
  "protein_g": 00.0,
  "fat_g": 00.0,
  "carbs_g": 00.0,
  "confidence_score": 0.95
}
Если на фото не еда, верни примерные значения для блюда, похожего на то, что изображено, или укажи nutrition равным 0.
"""

HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CalorieAI - Тестирование бэкенда</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex items-center justify-center p-4">
    <div class="max-w-md w-full bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl space-y-5">
        <div>
            <span class="bg-emerald-500/20 text-emerald-400 text-xs font-bold px-3 py-1 rounded-full border border-emerald-500/30">
                AI Тестер
            </span>
            <h1 class="text-2xl font-black text-white mt-3">CalorieAI Бэкенд</h1>
            <p class="text-xs text-slate-400 mt-1">Проверка распознавания еды через Gemini API</p>
        </div>

        <form id="food-form" class="space-y-4">
            <div>
                <label class="block text-xs font-semibold text-slate-300 mb-1">
                    Gemini API Key <span class="text-slate-500 font-normal">(запомнится в браузере или считается с Render)</span>
                </label>
                <input type="password" id="api-key" placeholder="Оставьте пустым, если задан GEMINI_API_KEY на Render" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-white focus:outline-none focus:border-emerald-500">
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-300 mb-1">Фото блюда</label>
                <input type="file" id="food-image" accept="image/*" required class="w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-emerald-500 file:text-slate-950 hover:file:bg-emerald-400 cursor-pointer">
            </div>

            <button type="submit" id="submit-btn" class="w-full bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-extrabold text-xs py-3 rounded-xl transition shadow-lg shadow-emerald-500/20">
                Распознать калории
            </button>
        </form>

        <div id="loading" class="hidden text-center py-4 text-xs text-emerald-400 animate-pulse">
            ⏳ Нейросеть Gemini анализирует тарелку...
        </div>

        <div id="result-box" class="hidden bg-slate-950 border border-slate-800 rounded-2xl p-4 space-y-2 text-xs">
            <h3 class="font-bold text-emerald-400 border-b border-slate-800 pb-2">Результат анализа:</h3>
            <div id="result-content" class="font-mono text-slate-200 overflow-x-auto"></div>
        </div>
    </div>

    <script>
        // Подтягиваем ключ из памяти браузера при загрузке
        const savedKey = localStorage.getItem('calorie_ai_gemini_key');
        if (savedKey) {
            document.getElementById('api-key').value = savedKey;
        }

        document.getElementById('food-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const apiKey = document.getElementById('api-key').value.trim();
            const imageFile = document.getElementById('food-image').files[0];
            const loading = document.getElementById('loading');
            const resultBox = document.getElementById('result-box');
            const resultContent = document.getElementById('result-content');

            if (apiKey) {
                localStorage.setItem('calorie_ai_gemini_key', apiKey);
            }

            const formData = new FormData();
            if (apiKey) {
                formData.append('api_key', apiKey);
            }
            formData.append('image', imageFile);

            loading.classList.remove('hidden');
            resultBox.classList.add('hidden');

            try {
                const response = await fetch('/api/analyze-food', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                loading.classList.add('hidden');
                resultBox.classList.remove('hidden');

                if (response.ok) {
                    resultContent.innerHTML = `<pre class="text-emerald-300">${JSON.stringify(data, null, 2)}</pre>`;
                } else {
                    resultContent.innerHTML = `<span class="text-red-400">Ошибка: ${data.detail || JSON.stringify(data)}</span>`;
                }
            } catch (err) {
                loading.classList.add('hidden');
                resultBox.classList.remove('hidden');
                resultContent.innerHTML = `<span class="text-red-400">Ошибка соединения: ${err.message}</span>`;
            }
        });
    </script>
</body>
</html>
"""

class NutritionData(BaseModel):
    dish_name: str
    calories: int
    protein_g: float
    fat_g: float
    carbs_g: float
    confidence_score: float

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_INTERFACE

@app.post("/api/analyze-food", response_model=NutritionData)
async def analyze_food(api_key: str = Form(None), image: UploadFile = File(...)):
    try:
        # Приоритет: Ключ из формы -> Переменная GEMINI_API_KEY из окружения Render
        final_api_key = (api_key.strip() if api_key and api_key.strip() else os.environ.get("GEMINI_API_KEY", "")).strip()

        if not final_api_key:
            raise HTTPException(
                status_code=400,
                detail="API-ключ не найден! Задайте переменную GEMINI_API_KEY в настройках Render или введите ключ в форме."
            )

        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Инициализируем клиента с полученным ключом
        client = genai.Client(api_key=final_api_key)

        candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        last_error = None

        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[pil_image, GEMINI_FOOD_PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]

                result_json = json.loads(raw_text.strip())
                return NutritionData(**result_json)
            except Exception as model_err:
                last_error = model_err
                if "404" in str(model_err) or "NOT_FOUND" in str(model_err):
                    continue
                raise model_err

        raise HTTPException(
            status_code=500,
            detail=f"Не удалось подобрать доступную модель Gemini. Ошибка: {str(last_error)}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)