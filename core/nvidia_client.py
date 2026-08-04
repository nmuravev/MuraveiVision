"""Слой 4: Облачная проверка подозрительных кропов через NVIDIA API (OpenAI-совместимый).

NvidiaVisionClient:
  - вырезает кроп с паддингом и клиппингом по границам кадра;
  - кодирует в base64 JPEG;
  - шлёт POST на {NVIDIA_API_BASE}/chat/completions (OpenAI-совместимый формат);
  - между запросами делает time.sleep(NVIDIA_API_DELAY) — лимит 40 запросов/мин;
  - если нет ключа или нет сети — НЕ падает, возвращает None.
"""
import os
import time
import base64

import cv2
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Настройки из .env ──
_API_BASE = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
_API_KEY = os.getenv("NVIDIA_API_KEY", "")
_VISION_MODEL = os.getenv("NVIDIA_VISION_MODEL", "meta/llama-3.2-90b-vision-instruct")
_API_DELAY = float(os.getenv("NVIDIA_API_DELAY", "2.0"))

# Вопрос по умолчанию для военного аналитика
_DEFAULT_PROMPT = (
    "Ты военный аналитик. На этом кадре есть признаки замаскированной "
    "военной позиции? Перечисли демаскирующие признаки."
)

# Время последнего запроса — для соблюдения задержки между запросами
_last_request_time = 0.0


def _crop_with_padding(frame, bbox, padding=0.15):
    """Вырезает кроп с паддингом и клиппингом по границам кадра.

    Аргументы:
      frame   — numpy array (H, W, 3), BGR;
      bbox    — [x1, y1, x2, y2] в пикселях;
      padding — доля расширения кропа (15% по умолчанию).

    Возвращает:
      numpy array — кроп кадра.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Расширяем кроп на padding%
    bw = x2 - x1
    bh = y2 - y1
    dx = bw * padding
    dy = bh * padding

    x1 = int(max(0, x1 - dx))
    y1 = int(max(0, y1 - dy))
    x2 = int(min(w, x2 + dx))
    y2 = int(min(h, y2 + dy))

    return frame[y1:y2, x1:x2]


def _encode_crop_base64(crop, quality=85):
    """Кодирует кроп в base64 JPEG."""
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _wait_rate_limit():
    """Соблюдает задержку между запросами (лимит 40 запросов/мин)."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _API_DELAY:
        time.sleep(_API_DELAY - elapsed)
    _last_request_time = time.time()


def is_available():
    """Проверяет, доступен ли слой 4 (есть ключ и модель)."""
    if not _API_KEY or "ВСТАВЬТЕ" in _API_KEY:
        return False
    return True


def analyze_crop(frame, bbox, prompt=None):
    """Отправляет кроп в NVIDIA Vision API и возвращает текстовый ответ.

    Аргументы:
      frame  — numpy array (H, W, 3), BGR;
      bbox   — [x1, y1, x2, y2] в пикселях;
      prompt — текст вопроса (если None — используется _DEFAULT_PROMPT).

    Возвращает:
      str — ответ модели, или None если слой недоступен/ошибка.
    """
    # Проверка доступности
    if not is_available():
        return None

    # Вырезаем кроп
    crop = _crop_with_padding(frame, bbox)
    if crop is None or crop.size == 0:
        return None

    # Кодируем в base64
    b64 = _encode_crop_base64(crop)
    if b64 is None:
        return None

    # Соблюдаем лимит запросов
    _wait_rate_limit()

    # Формируем запрос в OpenAI-совместимом формате
    question = prompt or _DEFAULT_PROMPT
    data_url = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model": _VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(
            f"{_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # Извлекаем текст ответа (OpenAI-совместимый формат)
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        # Нет сети — работаем офлайн
        return None
    except Exception:
        # Любая другая ошибка — не падаем
        return None
