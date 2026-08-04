"""Генератор самодостаточного HTML-отчёта (AMOLED-стиль).

generate_html(...) создаёт output/<video_stem>_report.html:
  - миниатюры (ширина 480, JPEG quality 70) встроены как base64;
  - файл самодостаточен (можно отправить коллеге — откроется в браузере);
  - шапка: видео, дата, железо, модель, настройки;
  - карточки моментов: время, классы, confidence;
  - вердикты слоя 4 (если есть);
  - максимум 200 миниатюр.
"""
import base64
import datetime
from pathlib import Path

import cv2


# ── AMOLED-палитра (как в GUI) ──
_C_BG = "#000000"
_C_CARD = "#0D0D0D"
_C_BORDER = "#1E1E1E"
_C_ACCENT = "#00E5FF"
_C_SUCCESS = "#00FF88"
_C_ERROR = "#FF3B30"
_C_TEXT = "#E0E0E0"
_C_TEXT_SEC = "#8A8A8A"


def _frame_to_base64(frame, max_width=480, quality=70):
    """Масштабирует BGR-кадр по ширине и кодирует в base64 JPEG.

    Возвращает строку data URL для <img src="...">.
    """
    h, w = frame.shape[:2]
    scale = max_width / w if w > max_width else 1.0
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _draw_boxes(frame, objects):
    """Рисует рамки bbox и имена классов на кадре (для миниатюры)."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    thickness = max(1, int(min(h, w) / 300))

    # Цвета для разных классов (BGR)
    colors = [
        (0x44, 0x44, 0xff), (0x44, 0xff, 0x44), (0xff, 0x44, 0x44),
        (0x44, 0xff, 0xff), (0xff, 0x44, 0xff), (0xff, 0xff, 0x44),
    ]

    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        b, g, r = colors[i % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (b, g, r), thickness)

        label = f"{obj['class']} {obj['confidence']:.2f}"
        font_scale = max(0.4, min(h, w) / 800)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4),
                      (x1 + tw + 4, y1), (b, g, r), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


def _esc(value) -> str:
    """Экранирование HTML-сущностей."""
    if value is None:
        return ""
    s = str(value)
    # Экранируем спецсимволы HTML
    s = s.replace("&", "&" + "amp;")
    s = s.replace("<", "&" + "lt;")
    s = s.replace(">", "&" + "gt;")
    s = s.replace('"', "&" + "quot;")
    return s


def _build_header(video_path, report_data, settings):
    """Шапка отчёта: видео, дата, железо, модель, настройки."""
    created = report_data.get("created_at", "")
    model = report_data.get("model", "")
    hw = report_data.get("hardware", {})
    moments_count = report_data.get("moments_count", 0)

    # Железо — компактная строка
    if isinstance(hw, dict):
        gpu = hw.get("gpu")
        cpu = hw.get("cpu", {})
        if gpu:
            hw_str = f"GPU: {_esc(gpu.get('name', ''))} ({gpu.get('vram_mb', 0) // 1024} GB)"
        else:
            hw_str = f"CPU: {_esc(cpu.get('brand', ''))}"
    else:
        hw_str = _esc(hw)

    # Настройки
    settings_rows = ""
    if settings:
        for k, v in settings.items():
            settings_rows += (
                f"<div class='kv-row'><span class='kv-key'>{_esc(k)}</span>"
                f"<span class='kv-val'>{_esc(v)}</span></div>"
            )

    return f"""
    <header class='header'>
        <h1>🎯 MuraveiVision — Отчёт</h1>
        <div class='meta'>
            <div class='kv-row'><span class='kv-key'>Видео</span>
                <span class='kv-val mono'>{_esc(video_path)}</span></div>
            <div class='kv-row'><span class='kv-key'>Дата</span>
                <span class='kv-val'>{_esc(created)}</span></div>
            <div class='kv-row'><span class='kv-key'>Железо</span>
                <span class='kv-val'>{hw_str}</span></div>
            <div class='kv-row'><span class='kv-key'>Модель</span>
                <span class='kv-val mono'>{_esc(model)}</span></div>
            <div class='kv-row'><span class='kv-key'>Моментов</span>
                <span class='kv-val' style='color:{_C_SUCCESS}'>{moments_count}</span></div>
        </div>
        {('<div class="settings">' + settings_rows + '</div>') if settings_rows else ''}
    </header>
    """


def _build_card(moment, idx, cloud_annotations=None):
    """Карточка одного момента: миниатюра, время, классы, confidence, вердикт слоя 4."""
    cloud_annotations = cloud_annotations or {}
    ts = moment.get("timestamp", "")
    frame_idx = moment.get("frame", 0)
    objects = moment.get("objects", [])

    # Список классов с confidence
    obj_lines = ""
    for o in objects:
        cls = _esc(o.get("class", ""))
        conf = o.get("confidence", 0)
        obj_lines += (
            f"<div class='obj-row'>"
            f"<span class='obj-class'>{cls}</span>"
            f"<span class='obj-conf'>{conf:.2f}</span></div>"
        )

    # Вердикт слоя 4 (если есть)
    cloud_text = cloud_annotations.get(frame_idx, "")
    cloud_block = ""
    if cloud_text:
        cloud_block = (
            f"<div class='cloud-verdict'>"
            f"<div class='cloud-title'>☁️ Слой 4 (NVIDIA Vision):</div>"
            f"<div class='cloud-text'>{_esc(cloud_text[:500])}</div></div>"
        )

    # Миниатюра — заглушка, заполнится при генерации (data-img-id)
    return f"""
    <div class='card' data-img-id='{idx}'>
        <div class='card-thumb'><img class='thumb' alt='moment {idx}' /></div>
        <div class='card-info'>
            <div class='card-ts mono'>⏱️ {ts}</div>
            <div class='card-objs'>{obj_lines}</div>
            {cloud_block}
        </div>
    </div>
    """


def generate_html(video_path, moments, report_data=None, settings=None,
                  cloud_annotations=None, output_dir="output",
                  max_thumbnails=200):
    """Генерирует самодостаточный HTML-отчёт.

    Аргументы:
      video_path        — путь к видеофайлу;
      moments           — список моментов [{timestamp, frame, objects}];
      report_data       — dict с метаданными (model, hardware, created_at, moments_count);
      settings          — dict настроек анализа (drone_mode, confidence, ...);
      cloud_annotations — dict {frame_idx: str} — ответы слоя 4 (опционально);
      output_dir        — папка для сохранения HTML;
      max_thumbnails    — максимум миниатюр (200).

    Возвращает путь к созданному HTML-файлу (str).
    """
    report_data = report_data or {}
    cloud_annotations = cloud_annotations or {}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem
    out_file = out_dir / f"{video_stem}_report.html"

    # Ограничиваем количество моментов для миниатюр
    display_moments = moments[:max_thumbnails]

    # Шапка
    header_html = _build_header(video_path, report_data, settings)

    # Карточки (с заглушками под миниатюры)
    cards_html = ""
    for i, moment in enumerate(display_moments):
        cards_html += _build_card(moment, i, cloud_annotations)

    # CSS в AMOLED-стиле
    css = f"""
    <style>
    * {{ box-sizing: border-box; }}
    body {{
        margin: 0; padding: 20px;
        background: {_C_BG}; color: {_C_TEXT};
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }}
    .mono {{ font-family: 'Consolas', 'Courier New', monospace; }}
    .header {{
        background: {_C_CARD}; border: 1px solid {_C_BORDER};
        border-radius: 12px; padding: 20px; margin-bottom: 20px;
    }}
    .header h1 {{ color: {_C_ACCENT}; margin: 0 0 15px 0; font-size: 24px; }}
    .meta {{ display: flex; flex-direction: column; gap: 6px; }}
    .kv-row {{ display: flex; gap: 12px; font-size: 14px; }}
    .kv-key {{ color: {_C_TEXT_SEC}; min-width: 100px; }}
    .kv-val {{ color: {_C_TEXT}; }}
    .settings {{
        margin-top: 12px; padding-top: 12px;
        border-top: 1px solid {_C_BORDER};
    }}
    .card {{
        display: flex; gap: 16px;
        background: {_C_CARD}; border: 1px solid {_C_BORDER};
        border-radius: 10px; padding: 12px; margin-bottom: 14px;
    }}
    .card-thumb img {{ width: 480px; border-radius: 6px; display: block; }}
    .card-info {{ flex: 1; }}
    .card-ts {{
        color: {_C_ACCENT}; font-size: 16px; font-weight: bold; margin-bottom: 8px;
    }}
    .card-objs {{ display: flex; flex-direction: column; gap: 4px; }}
    .obj-row {{ display: flex; justify-content: space-between; max-width: 300px; }}
    .obj-class {{ color: {_C_TEXT}; }}
    .obj-conf {{ color: {_C_SUCCESS}; font-family: 'Consolas', monospace; }}
    .cloud-verdict {{
        margin-top: 10px; padding: 10px;
        background: #1a1500; border: 1px solid #ffcc88; border-radius: 6px;
    }}
    .cloud-title {{ color: #ffcc88; font-size: 13px; font-weight: bold; margin-bottom: 4px; }}
    .cloud-text {{ color: #e8d8a0; font-size: 12px; white-space: pre-wrap; }}
    .footer {{ text-align: center; color: {_C_TEXT_SEC}; margin-top: 30px; font-size: 12px; }}
    </style>
    """

    # JS: подставляет base64-миниатюры в заглушки (генерируются в Python)
    # Сами data-URL вставляем прямо в <script> как объект
    thumbs_js = "<script>\nvar _thumbs = {};\n</script>\n"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MuraveiVision — Отчёт — {Path(video_path).stem}</title>
{css}
</head>
<body>
{header_html}
<main>
{cards_html}
</main>
<div class="footer">Сгенерировано MuraveiVision · {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
{thumbs_js}
</body>
</html>
"""

    # Теперь генерируем миниатюры и вставляем их base64 прямо в HTML
    # (замена заглушек <img class='thumb' ...> на <img src='data:...'>)
    cap = cv2.VideoCapture(video_path)
    thumbs_data = {}
    for i, moment in enumerate(display_moments):
        frame_idx = moment.get("frame", 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        annotated = _draw_boxes(frame, moment.get("objects", []))
        data_url = _frame_to_base64(annotated, max_width=480, quality=70)
        if data_url:
            thumbs_data[i] = data_url
    cap.release()

    # Встраиваем миниатюры: заменяем заглушки на реальные <img src>
    for i, data_url in thumbs_data.items():
        placeholder = f"<img class='thumb' alt='moment {i}' />"
        real = f"<img class='thumb' src='{data_url}' alt='moment {i}' />"
        html = html.replace(placeholder, real, 1)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    return str(out_file)