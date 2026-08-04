"""Показывает ВСЕ модели, доступные вашему ключу NVIDIA прямо сейчас."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # читает .env из ТЕКУЩЕЙ папки

base = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
key = os.getenv("NVIDIA_API_KEY")

if not key or "ВСТАВЬТЕ" in key:
    print("❌ Сначала впишите настоящий ключ в файл .env!")
    exit(1)

r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
models = sorted(m["id"] for m in r.json()["data"])

print(f"📋 Доступно моделей: {len(models)}\n")

print("── СО ЗРЕНИЕМ (для анализа кадров) ──")
for m in models:
    if "vision" in m or "vl" in m:
        print("  👁️ ", m)

print("\n── ТЯЖЕЛЫЕ ДЛЯ КОДА ──")
for m in models:
    if any(k in m for k in ["glm", "kimi", "qwen", "70b", "405b", "nemotron", "deepseek"]):
        print("  🧠 ", m)

print("\n── ВСЕ ОСТАЛЬНЫЕ ──")
for m in models:
    print("  ·", m)