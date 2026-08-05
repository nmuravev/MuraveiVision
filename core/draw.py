"""Хелпер для рисования bbox с кириллическими подписями через PIL.

cv2.putText не поддерживает кириллицу (выводит "??????").
Используем PIL.ImageDraw + ImageFont (arial.ttf / DejaVuSans).
"""
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Кэш шрифта (загружаем один раз)
_font_cache = None


def _load_font(size=16):
    """Загружает шрифт с поддержкой кириллицы.

    Приоритет:
      1. C:\\Windows\\Fonts\\arial.ttf
      2. Ultralytics Arial.ttf в AppData/Roaming/Ultralytics/
      3. DejaVuSans (встроен в matplotlib/PIL, если доступен)
      4. Дефолтный PIL-шрифт (fallback, кириллица может не работать)
    """
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        os.path.join(os.environ.get("APPDATA", ""),
                     "Roaming", "Ultralytics", "Arial.ttf"),
        os.path.join(os.environ.get("APPDATA", ""),
                     "Ultralytics", "Arial.ttf"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass

    # DejaVuSans — часто доступен через matplotlib
    try:
        import matplotlib
        dejavu = os.path.join(os.path.dirname(matplotlib.__file__),
                              "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
        if os.path.isfile(dejavu):
            return ImageFont.truetype(dejavu, size)
    except Exception:
        pass

    # Fallback: дефолтный PIL-шрифт
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_boxes_pil(frame, objects):
    """Рисует рамки bbox и кириллические подписи через PIL.

    Аргументы:
      frame   — BGR-кадр (numpy array, как в cv2);
      objects — список объектов [{class, confidence, bbox:[x1,y1,x2,y2]}].

    Возвращает BGR-кадр (numpy array) с нарисованными рамками и подписями.
    Подпись: "танк 0.87" (класс на русском + confidence).
    """
    global _font_cache

    h, w = frame.shape[:2]
    thickness = max(1, int(min(h, w) / 300))

    # Цвета для разных классов (BGR → RGB для PIL)
    colors_bgr = [
        (0x44, 0x44, 0xff), (0x44, 0xff, 0x44), (0xff, 0x44, 0x44),
        (0x44, 0xff, 0xff), (0xff, 0x44, 0xff), (0xff, 0xff, 0x44),
    ]

    # BGR → RGB для PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    # Размер шрифта зависит от размера кадра
    font_size = max(12, int(min(h, w) / 50))
    if _font_cache is None or getattr(_font_cache, "size", 0) != font_size:
        _font_cache = _load_font(font_size)

    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        b, g, r = colors_bgr[i % len(colors_bgr)]
        # PIL использует RGB
        rgb_color = (r, g, b)

        # Рамка
        draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=thickness)

        # Подпись: "танк 0.87"
        label = f"{obj['class']} {obj['confidence']:.2f}"

        # Фон подписи
        try:
            bbox = draw.textbbox((x1, y1), label, font=_font_cache)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label) * font_size // 2, font_size

        # Прямоугольник-фон над рамкой
        bg_y1 = max(0, y1 - th - 4)
        bg_y2 = y1
        draw.rectangle([x1, bg_y1, x1 + tw + 4, bg_y2], fill=rgb_color)

        # Текст (чёрный на цветном фоне)
        text_color = (0, 0, 0)
        draw.text((x1 + 2, bg_y1 + 1), label, fill=text_color, font=_font_cache)

    # RGB → BGR обратно
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result