import os
import json
import re
import io
import base64
import urllib.request
import urllib.error
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai

app = FastAPI(
    title="CalorieAI API",
    description="API для распознавания калорий и БЖУ по фотографии еды",
    version="1.2.0"
)

class NutritionData(BaseModel):
    dish_name: str
    calories: int
    protein_g: float
    fat_g: float
    carbs_g: float
    confidence_score: float

HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CalorieAI - Multi-AI Тестер</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex items-center justify-center p-4">

    <div class="max-w-xl w-full bg-slate-900 border border-slate-800 rounded-3xl p-6 md:p-8 space-y-6 shadow-2xl">
        
        <!-- Header -->
        <div class="text-center space-y-2">
            <div class="inline-flex items-center gap-2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1 rounded-full text-xs font-semibold">
                <i class="fa-solid fa-bolt"></i> CalorieAI Backend Online
            </div>
            <h1 class="text-2xl font-extrabold text-white">Тестирование распознавания еды</h1>
            <p class="text-xs text-slate-400">Выберите провайдера AI или используйте Демо-режим</p>
        </div>

        <!-- Form -->
        <form id="analyze-form" class="space-y-4">
            
            <!-- AI Provider Select -->
            <div class="space-y-1.5">
                <label class="text-xs font-semibold text-slate-300">Выберите AI Движок:</label>
                <select id="ai-provider" onchange="toggleProviderInputs()"
                    class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 font-semibold transition">
                    <option value="groq" selected>🚀 Groq AI (Рекомендуется — Без блокировок VPN)</option>
                    <option value="gemini">✨ Google Gemini API</option>
                    <option value="demo">🎮 Демо-режим (Тест UI без нейросетей)</option>
                </select>
            </div>

            <!-- Groq Key Input -->
            <div id="groq-key-box" class="space-y-1.5">
                <label class="text-xs font-semibold text-slate-300 flex justify-between">
                    <span>Groq API Key (Бесплатно):</span>
                    <a href="https://console.groq.com/keys" target="_blank" class="text-cyan-400 hover:underline">Получить ключ за 20 сек →</a>
                </label>
                <input type="password" id="groq-key" placeholder="gsk_..."
                    class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 transition">
                <p class="text-[10px] text-slate-500">Groq не блокирует VPN и бесплатно даёт 14 400 запросов в день.</p>
            </div>

            <!-- Gemini Key Input -->
            <div id="gemini-key-box" class="space-y-1.5 hidden">
                <label class="text-xs font-semibold text-slate-300 flex justify-between">
                    <span>Gemini API Key:</span>
                    <a href="https://aistudio.google.com/app/apikey" target="_blank" class="text-emerald-400 hover:underline">Получить ключ →</a>
                </label>
                <input type="password" id="api-key" placeholder="AIzaSy..."
                    class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 transition">
            </div>

            <!-- Image Upload Box -->
            <div class="space-y-1.5">
                <label class="text-xs font-semibold text-slate-300">Фотография блюда:</label>
                <div onclick="document.getElementById('file-input').click()" 
                    class="border-2 border-dashed border-slate-800 hover:border-emerald-500/50 bg-slate-950 rounded-2xl p-6 text-center cursor-pointer transition group">
                    <input type="file" id="file-input" accept="image/*" class="hidden" onchange="previewImage(event)">
                    
                    <div id="upload-placeholder" class="space-y-2">
                        <i class="fa-solid fa-cloud-arrow-up text-3xl text-slate-600 group-hover:text-emerald-400 transition"></i>
                        <p class="text-xs text-slate-400 font-medium">Нажмите для выбора фото</p>
                        <p class="text-[10px] text-slate-600">JPG, PNG, WEBP до 10MB</p>
                    </div>

                    <div id="image-preview-box" class="hidden relative">
                        <img id="image-preview" class="max-h-48 mx-auto rounded-xl object-contain shadow-md" src="" alt="Превью">
                        <span class="text-[10px] text-emerald-400 block mt-2">Нажмите, чтобы заменить фото</span>
                    </div>
                </div>
            </div>

            <!-- Submit Button -->
            <button type="submit" id="submit-btn" 
                class="w-full bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-sm py-3 rounded-xl transition shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2">
                <i class="fa-solid fa-wand-magic-sparkles"></i> Распознать блюдо
            </button>
        </form>

        <!-- Loader -->
        <div id="loader" class="hidden text-center py-6 space-y-3">
            <div class="inline-block w-8 h-8 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
            <p class="text-xs text-slate-400 font-medium">Нейросеть анализирует изображение...</p>
        </div>

        <!-- Result Card -->
        <div id="result-card" class="hidden bg-slate-950 border border-slate-800 rounded-2xl p-5 space-y-4">
            <div class="flex justify-between items-start border-b border-slate-800 pb-3">
                <div>
                    <span id="res-confidence" class="text-[9px] bg-emerald-500/20 text-emerald-300 font-bold px-2 py-0.5 rounded">Точность 95%</span>
                    <h3 id="res-name" class="text-base font-bold text-white mt-1">Название блюда</h3>
                </div>
                <div class="text-right">
                    <div id="res-calories" class="text-xl font-extrabold text-emerald-400">0 ккал</div>
                </div>
            </div>

            <!-- Macros Grid -->
            <div class="grid grid-cols-3 gap-2 text-center text-xs">
                <div class="bg-slate-900 p-2.5 rounded-xl border border-slate-800/80">
                    <div class="text-[10px] text-slate-400">Белки</div>
                    <div id="res-protein" class="font-bold text-blue-400 mt-0.5">0g</div>
                </div>
                <div class="bg-slate-900 p-2.5 rounded-xl border border-slate-800/80">
                    <div class="text-[10px] text-slate-400">Жиры</div>
                    <div id="res-fat" class="font-bold text-amber-400 mt-0.5">0g</div>
                </div>
                <div class="bg-slate-900 p-2.5 rounded-xl border border-slate-800/80">
                    <div class="text-[10px] text-slate-400">Углеводы</div>
                    <div id="res-carbs" class="font-bold text-purple-400 mt-0.5">0g</div>
                </div>
            </div>
        </div>

        <!-- Error Card -->
        <div id="error-card" class="hidden bg-rose-950/40 border border-rose-500/40 rounded-2xl p-4 text-xs text-rose-300 space-y-2">
            <div class="font-bold flex items-center gap-2">
                <i class="fa-solid fa-triangle-exclamation"></i> Ошибка анализа:
            </div>
            <p id="error-text" class="leading-relaxed text-slate-300">Описание ошибки...</p>
        </div>

    </div>

    <script>
        function toggleProviderInputs() {
            const provider = document.getElementById('ai-provider').value;
            document.getElementById('groq-key-box').classList.toggle('hidden', provider !== 'groq');
            document.getElementById('gemini-key-box').classList.toggle('hidden', provider !== 'gemini');
        }

        function previewImage(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('image-preview').src = e.target.result;
                    document.getElementById('upload-placeholder').classList.add('hidden');
                    document.getElementById('image-preview-box').classList.remove('hidden');
                }
                reader.readAsDataURL(file);
            }
        }

        document.getElementById('analyze-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            const fileInput = document.getElementById('file-input');
            const provider = document.getElementById('ai-provider').value;
            const apiKeyInput = document.getElementById('api-key').value.trim();
            const groqKeyInput = document.getElementById('groq-key').value.trim();

            if (!fileInput.files[0]) {
                alert('Пожалуйста, выберите фото блюда!');
                return;
            }

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            formData.append('provider', provider);
            if (apiKeyInput) formData.append('api_key', apiKeyInput);
            if (groqKeyInput) formData.append('groq_api_key', groqKeyInput);

            document.getElementById('loader').classList.remove('hidden');
            document.getElementById('result-card').classList.add('hidden');
            document.getElementById('error-card').classList.add('hidden');
            document.getElementById('submit-btn').disabled = true;

            try {
                const response = await fetch('/api/analyze-food', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || 'Ошибка сервера при запросе.');
                }

                document.getElementById('res-name').textContent = data.dish_name;
                document.getElementById('res-calories').textContent = `${data.calories} ккал`;
                document.getElementById('res-protein').textContent = `${data.protein_g}g`;
                document.getElementById('res-fat').textContent = `${data.fat_g}g`;
                document.getElementById('res-carbs').textContent = `${data.carbs_g}g`;
                document.getElementById('res-confidence').textContent = `Точность ${Math.round(data.confidence_score * 100)}%`;

                document.getElementById('result-card').classList.remove('hidden');
            } catch (err) {
                document.getElementById('error-text').innerHTML = err.message;
                document.getElementById('error-card').classList.remove('hidden');
            } finally {
                document.getElementById('loader').classList.add('hidden');
                document.getElementById('submit-btn').disabled = false;
            }
        });
    </script>
</body>
</html>
"""

def analyze_with_groq(groq_key: str, image_bytes: bytes) -> NutritionData:
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """
    Проанализируй фото еды. Определи блюдо, его калорийность и БЖУ (белки, жиры, углеводы).
    Верни результат ИСКЛЮЧИТЕЛЬНО в формате JSON без разметки markdown:
    {
      "dish_name": "Название блюда на русском",
      "calories": 450,
      "protein_g": 25.0,
      "fat_g": 12.5,
      "carbs_g": 55.0,
      "confidence_score": 0.95
    }
    """

    payload = {
        "model": "llama-3.2-11b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }

    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode('utf-8'),
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            res_data = json.loads(resp.read().decode('utf-8'))
            content = res_data['choices'][0]['message']['content']
            data = json.loads(content)

            return NutritionData(
                dish_name=data.get("dish_name", "Неизвестное блюдо"),
                calories=int(data.get("calories", 0)),
                protein_g=float(data.get("protein_g", 0.0)),
                fat_g=float(data.get("fat_g", 0.0)),
                carbs_g=float(data.get("carbs_g", 0.0)),
                confidence_score=float(data.get("confidence_score", 0.95))
            )
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8')
        raise HTTPException(status_code=e.code, detail=f"Ошибка Groq API ({e.code}): {err_body}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка Groq: {str(e)}")

def analyze_with_gemini(gemini_key: str, pil_image: Image.Image) -> NutritionData:
    genai.configure(api_key=gemini_key)

    prompt = """
    Проанализируй фото еды. Определи блюдо, его калорийность и БЖУ (белки, жиры, углеводы).
    Верни результат ИСКЛЮЧИТЕЛЬНО в формате JSON со следующей структурой:
    {
      "dish_name": "Название блюда на русском",
      "calories": 450,
      "protein_g": 25.0,
      "fat_g": 12.5,
      "carbs_g": 55.0,
      "confidence_score": 0.95
    }
    """

    candidate_models = [
        "gemini-2.0-flash", 
        "gemini-1.5-flash", 
        "gemini-1.5-flash-8b", 
        "gemini-2.0-flash-lite-preview-02-05"
    ]
    last_error = ""

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content([prompt, pil_image])
            text = response.text.strip()
            data = json.loads(text)

            return NutritionData(
                dish_name=data.get("dish_name", "Неизвестное блюдо"),
                calories=int(data.get("calories", 0)),
                protein_g=float(data.get("protein_g", 0.0)),
                fat_g=float(data.get("fat_g", 0.0)),
                carbs_g=float(data.get("carbs_g", 0.0)),
                confidence_score=float(data.get("confidence_score", 0.90))
            )
        except Exception as e:
            last_error = str(e)
            if "RESOURCE_EXHAUSTED" not in last_error and "quota" not in last_error.lower() and "429" not in last_error:
                break

    if "RESOURCE_EXHAUSTED" in last_error or "429" in last_error or "limit" in last_error.lower():
        raise HTTPException(
            status_code=429,
            detail=(
                "<b>Google заблокировал бесплатную квоту (Limit 0) для этого VPN IP:</b><br>"
                "Google AI Studio определяет публичные VPN сервера и отключает бесплатные запросы.<br><br>"
                "<b>Быстрое решение без VPN:</b><br>"
                "1. Выберите вверху AI Движок: <b>«Groq AI (Рекомендуется)»</b>.<br>"
                "2. Зарегистрируйтесь на <a href='https://console.groq.com/keys' target='_blank' class='underline font-bold text-cyan-400'>console.groq.com</a> (занимает 20 секунд).<br>"
                "3. Создайте бесплатный API ключ и вставьте его сюда."
            )
        )

    raise HTTPException(status_code=500, detail=f"Ошибка Gemini: {last_error}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTML_INTERFACE

@app.post("/api/analyze-food", response_model=NutritionData)
async def analyze_food(
    provider: Optional[str] = Form("groq"),
    api_key: Optional[str] = Form(None), 
    groq_api_key: Optional[str] = Form(None), 
    image: UploadFile = File(...)
):
    if provider == "demo":
        return NutritionData(
            dish_name="Филе лосося с запеченной спаржей (Демо)",
            calories=420,
            protein_g=34.5,
            fat_g=22.0,
            carbs_g=8.0,
            confidence_score=0.98
        )

    try:
        contents = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать изображение: {str(e)}")

    if provider == "groq":
        final_groq_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not final_groq_key:
            raise HTTPException(
                status_code=400, 
                detail="Укажите Groq API Key или перейдите на <a href='https://console.groq.com/keys' target='_blank' class='underline font-bold'>console.groq.com</a> для его бесплатного получения."
            )
        return analyze_with_groq(final_groq_key, contents)

    # Gemini provider fallback
    final_gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not final_gemini_key:
        raise HTTPException(
            status_code=400, 
            detail="API ключ Gemini не найден."
        )

    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Невалидный файл изображения: {str(e)}")

    return analyze_with_gemini(final_gemini_key, pil_image)