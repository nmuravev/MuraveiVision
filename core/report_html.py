"""Генератор самодостаточного HTML-отчёта v2 (AMOLED-стиль).

generate_html(...) создаёт output/<video_stem>_report.html:
  - раскладка карточки: фото сверху (520px), снизу мета-строка
    [🕐 ЧЧ-ММ-СС] [класс1] [класс2] [0.87] [verdict слоя 4];
  - поиск по классу, сортировка (время/confidence), фильтр "только с вердиктом";
  - чекбоксы выбора у карточек;
  - плавающая панель: [💾 Сохранить выбранные] (ZIP через JSZip),
    [📋 Скачать JSON];
  - НЕТ лимита 200 — все объекты;
  - миниатюры (520px, JPEG quality 70) встроены как base64;
  - классы на русском (через ru()).
"""
import base64
import datetime
import json
import zipfile
import io
from pathlib import Path

import cv2

from core.classes import ru


# ── AMOLED-палитра (все используемые константы объявлены) ──
_C_BG = "#000000"
_C_PANEL = "#0D0D0D"      # панели/карточки (alias _C_CARD)
_C_CARD = "#0D0D0D"       # карточки
_C_BORDER = "#1E1E1E"
_C_ACCENT = "#00E5FF"
_C_ACCENT_HOVER = "#00B8D4"  # hover-цвет акцента
_C_OK = "#00FF88"         # успех (alias _C_SUCCESS)
_C_SUCCESS = "#00FF88"
_C_ERR = "#FF3B30"        # ошибка (alias _C_ERROR)
_C_ERROR = "#FF3B30"
_C_TEXT = "#E0E0E0"
_C_DIM = "#8A8A8A"        # вторичный текст (alias _C_TEXT_SEC)
_C_TEXT_SEC = "#8A8A8A"


def _frame_to_base64(frame, max_width=520, quality=70):
    """Масштабирует BGR-кадр по ширине и кодирует в base64 JPEG."""
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
    """Рисует рамки bbox и имена классов (на русском) на кадре.

    П.4: кириллица через PIL (cv2.putText не поддерживает русские буквы).
    Делегирует в core.draw.draw_boxes_pil.
    """
    from core.draw import draw_boxes_pil
    return draw_boxes_pil(frame, objects)


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
        <h1>🎯 MuraveiVision — Отчёт v2</h1>
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


def generate_html(video_path, moments, report_data=None, settings=None,
                  cloud_annotations=None, output_dir="output"):
    """Генерирует самодостаточный HTML-отчёт v2.

    Аргументы:
      video_path        — путь к видеофайлу;
      moments           — список моментов [{timestamp, frame, objects}];
      report_data       — dict с метаданными (model, hardware, created_at, moments_count);
      settings          — dict настроек анализа;
      cloud_annotations — dict {frame_idx: str} — ответы слоя 4 (опционально);
      output_dir        — папка для сохранения HTML.

    Возвращает путь к созданному HTML-файлу (str).
    Лимита 200 НЕТ — все объекты.
    """
    report_data = report_data or {}
    cloud_annotations = cloud_annotations or {}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem
    out_file = out_dir / f"{video_stem}_report.html"

    # Шапка
    header_html = _build_header(video_path, report_data, settings)

    # CSS в AMOLED-стиле v2
    css = f"""
    <style>
    * {{ box-sizing: border-box; }}
    body {{
        margin: 0; padding: 20px; padding-bottom: 80px;
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
    .settings {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid {_C_BORDER}; }}

    /* ── Панель управления сверху ── */
    .toolbar {{
        position: sticky; top: 0; z-index: 100;
        background: {_C_CARD}; border: 1px solid {_C_BORDER};
        border-radius: 10px; padding: 12px; margin-bottom: 20px;
        display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
    }}
    .toolbar input[type="text"] {{
        background: #000; border: 1px solid {_C_BORDER}; color: {_C_TEXT};
        padding: 8px 12px; border-radius: 6px; font-size: 14px; width: 200px;
    }}
    .toolbar select {{
        background: #000; border: 1px solid {_C_BORDER}; color: {_C_TEXT};
        padding: 8px 12px; border-radius: 6px; font-size: 14px;
    }}
    .toolbar label {{ color: {_C_TEXT_SEC}; font-size: 13px; cursor: pointer; }}

    /* ── Карточки v2: фото сверху, мета снизу ── */
    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
        gap: 16px;
    }}
    .card {{
        background: {_C_CARD}; border: 1px solid {_C_BORDER};
        border-radius: 10px; overflow: hidden;
        transition: box-shadow 0.2s, border-color 0.2s;
    }}
    .card:hover {{
        border-color: {_C_ACCENT};
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.3);
    }}
    .card.hidden {{ display: none; }}
    .card-thumb {{ position: relative; }}
    .card-thumb img {{ width: 100%; display: block; }}
    .card-check {{
        position: absolute; top: 8px; right: 8px;
        width: 24px; height: 24px; cursor: pointer;
        accent-color: {_C_ACCENT};
    }}
    .card-meta {{
        padding: 10px 12px; display: flex; flex-wrap: wrap;
        gap: 8px; align-items: center; font-size: 13px;
    }}
    .meta-ts {{ color: {_C_ACCENT}; font-weight: bold; font-family: 'Consolas', monospace; }}
    .meta-class {{
        background: #1a1a1a; border: 1px solid {_C_BORDER};
        padding: 2px 8px; border-radius: 4px; color: {_C_TEXT};
    }}
    .meta-conf {{ color: {_C_SUCCESS}; font-family: 'Consolas', monospace; }}
    .meta-verdict {{
        color: #ffcc88; font-size: 12px; font-style: italic;
        background: #1a1500; padding: 2px 8px; border-radius: 4px;
        max-width: 300px; overflow: hidden; text-overflow: ellipsis;
        white-space: nowrap;
    }}

    /* ── Плавающая панель снизу ── */
    .float-panel {{
        position: fixed; bottom: 0; left: 0; right: 0;
        background: {_C_CARD}; border-top: 1px solid {_C_BORDER};
        padding: 12px 20px; display: flex; gap: 12px;
        justify-content: center; z-index: 200;
    }}
    .float-btn {{
        background: {_C_ACCENT}; color: #000; border: none;
        padding: 10px 20px; border-radius: 6px; font-size: 14px;
        font-weight: bold; cursor: pointer;
    }}
    .float-btn:hover {{ background: {_C_ACCENT_HOVER}; }}
    .float-btn.secondary {{ background: #333; color: {_C_TEXT}; }}

    .footer {{ text-align: center; color: {_C_TEXT_SEC}; margin-top: 30px; font-size: 12px; }}
    </style>
    """

    # JS: поиск, сортировка, фильтр, чекбоксы, сохранение выбранных (ZIP), JSON
    # Данные моментов встраиваем как JSON для JS-обработки
    moments_js_data = []
    for i, m in enumerate(moments):
        objs = []
        for o in m.get("objects", []):
            cls_display = ru(o.get("class_en", o.get("class", "")))
            objs.append({
                "class": cls_display,
                "confidence": o.get("confidence", 0),
            })
        moments_js_data.append({
            "idx": i,
            "timestamp": m.get("timestamp", ""),
            "frame": m.get("frame", 0),
            "objects": objs,
            "has_verdict": str(m.get("frame", 0)) in [str(k) for k in cloud_annotations.keys()],
        })

    # Встраиваем base64-миниатюры в JS-данные (для ZIP)
    cap = cv2.VideoCapture(video_path)
    thumbs_data = {}
    for i, moment in enumerate(moments):
        frame_idx = moment.get("frame", 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        annotated = _draw_boxes(frame, moment.get("objects", []))
        data_url = _frame_to_base64(annotated, max_width=520, quality=70)
        if data_url:
            thumbs_data[i] = data_url
    cap.release()

    # JSON-отчёт для скачивания (встраиваем как base64)
    report_json = json.dumps({
        "video": video_path,
        "model": report_data.get("model", ""),
        "hardware": report_data.get("hardware", {}),
        "created_at": report_data.get("created_at", ""),
        "moments_count": len(moments),
        "moments": moments,
    }, ensure_ascii=False, indent=2)
    json_b64 = base64.b64encode(report_json.encode("utf-8")).decode("utf-8")

    js = f"""
    <script>
    // Данные моментов для JS
    var _moments = {json.dumps(moments_js_data, ensure_ascii=False)};
    // base64-миниатюры
    var _thumbs = {json.dumps(thumbs_data, ensure_ascii=False)};
    // JSON-отчёт (base64) для скачивания
    var _json_b64 = "{json_b64}";
    // Вердикты слоя 4
    var _verdicts = {json.dumps({str(k): v[:200] for k, v in cloud_annotations.items()}, ensure_ascii=False)};

    // ── Рендер карточек ──
    function renderCards() {{
        var grid = document.getElementById('grid');
        grid.innerHTML = '';
        _moments.forEach(function(m) {{
            var thumb = _thumbs[m.idx] || '';
            var verdict = _verdicts[String(m.frame)] || '';
            var classesHtml = m.objects.map(function(o) {{
                return '<span class="meta-class">' + escapeHtml(o.class) + '</span>';
            }}).join('');
            var maxConf = m.objects.length ? Math.max.apply(null, m.objects.map(function(o) {{ return o.confidence; }})) : 0;
            var verdictHtml = verdict ? '<span class="meta-verdict" title="' + escapeHtml(verdict) + '">☁️ ' + escapeHtml(verdict.substring(0, 60)) + '…</span>' : '';
            var card = document.createElement('div');
            card.className = 'card';
            card.dataset.idx = m.idx;
            card.dataset.ts = m.timestamp;
            card.dataset.conf = maxConf;
            card.dataset.hasVerdict = verdict ? '1' : '0';
            card.innerHTML =
                '<div class="card-thumb">' +
                (thumb ? '<img src="' + thumb + '" alt="moment ' + m.idx + '">' : '<div style="height:200px;display:flex;align-items:center;justify-content:center;color:#555">нет кадра</div>') +
                '<input type="checkbox" class="card-check" data-idx="' + m.idx + '">' +
                '</div>' +
                '<div class="card-meta">' +
                '<span class="meta-ts">🕐 ' + escapeHtml(m.timestamp) + '</span>' +
                classesHtml +
                '<span class="meta-conf">' + maxConf.toFixed(2) + '</span>' +
                verdictHtml +
                '</div>';
            grid.appendChild(card);
        }});
        applyFilters();
    }}

    function escapeHtml(s) {{
        if (s === null || s === undefined) return '';
        return String(s).replace(/&/g, '&' + 'amp;').replace(/</g, '&' + 'lt;').replace(/>/g, '&' + 'gt;');
    }}

    // ── Фильтры/сортировка/поиск ──
    function applyFilters() {{
        var search = document.getElementById('search').value.toLowerCase();
        var sort = document.getElementById('sort').value;
        var onlyVerdict = document.getElementById('onlyVerdict').checked;
        var cards = Array.from(document.querySelectorAll('.card'));

        // Фильтр
        cards.forEach(function(c) {{
            var idx = c.dataset.idx;
            var m = _moments[idx];
            var classesStr = m.objects.map(function(o) {{ return o.class; }}).join(' ').toLowerCase();
            var matchSearch = !search || classesStr.includes(search);
            var matchVerdict = !onlyVerdict || c.dataset.hasVerdict === '1';
            c.classList.toggle('hidden', !(matchSearch && matchVerdict));
        }});

        // Сортировка
        var visible = cards.filter(function(c) {{ return !c.classList.contains('hidden'); }});
        var grid = document.getElementById('grid');
        visible.sort(function(a, b) {{
            if (sort === 'time') {{
                return a.dataset.ts.localeCompare(b.dataset.ts);
            }} else if (sort === 'conf') {{
                return parseFloat(b.dataset.conf) - parseFloat(a.dataset.conf);
            }}
            return 0;
        }});
        visible.forEach(function(c) {{ grid.appendChild(c); }});
    }}

    // ── Сохранить выбранные (ZIP через JSZip) ──
    function saveSelected() {{
        var checked = Array.from(document.querySelectorAll('.card-check:checked')).map(function(cb) {{ return cb.dataset.idx; }});
        if (checked.length === 0) {{
            alert('Выберите хотя бы одну карточку');
            return;
        }}
        if (typeof JSZip === 'undefined') {{
            alert('JSZip не загружен. Проверьте интернет-соединение.');
            return;
        }}
        var zip = new JSZip();
        var folder = zip.folder('selected');
        checked.forEach(function(idx, i) {{
            var thumb = _thumbs[idx];
            if (thumb) {{
                var b64 = thumb.split(',')[1];
                folder.file('moment_' + (i + 1) + '.jpg', b64, {{base64: true}});
            }}
        }});
        // gallery.html внутри ZIP
        var html = '<!DOCTYPE html><html><head><meta charset="UTF-8"><style>' +
            'body{{background:#000;color:#E0E0E0;font-family:sans-serif;padding:20px}}' +
            '.card{{background:#0D0D0D;border:1px solid #1E1E1E;border-radius:10px;margin-bottom:16px;overflow:hidden}}' +
            'img{{width:100%;display:block}}.meta{{padding:10px;color:#00E5FF;font-family:monospace}}' +
            '</style></head><body><h1 style="color:#00E5FF">Выбранные моменты</h1>';
        checked.forEach(function(idx, i) {{
            var m = _moments[idx];
            var thumb = _thumbs[idx] || '';
            var classes = m.objects.map(function(o) {{ return o.class + ' ' + o.confidence.toFixed(2); }}).join(', ');
            html += '<div class="card">' +
                (thumb ? '<img src="' + thumb + '">' : '') +
                '<div class="meta">🕐 ' + m.timestamp + ' — ' + escapeHtml(classes) + '</div>' +
                '</div>';
        }});
        html += '</body></html>';
        folder.file('gallery.html', html);
        zip.generateAsync({{type: 'blob'}}).then(function(content) {{
            var a = document.createElement('a');
            a.href = URL.createObjectURL(content);
            a.download = 'selected_moments.zip';
            a.click();
        }});
    }}

    // ── Скачать JSON ──
    function downloadJson() {{
        var json = atob(_json_b64);
        var blob = new Blob([json], {{type: 'application/json'}});
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'report.json';
        a.click();
    }}

    // ── Клик по карточке → сигнал (для PRO: прыжок плеера) ──
    document.addEventListener('click', function(e) {{
        if (e.target.classList.contains('card-check')) return;
        var card = e.target.closest('.card');
        if (card) {{
            var idx = card.dataset.idx;
            var m = _moments[idx];
            // Для PRO: вызываем Python через QWebChannel (если доступен)
            if (typeof qt !== 'undefined' && window.pyBridge) {{
                window.pyBridge.seekToMoment(m.frame, m.timestamp);
            }}
            // Для Мини: просто подсвечиваем
            card.style.borderColor = '#00FF88';
            setTimeout(function() {{ card.style.borderColor = ''; }}, 500);
        }}
    }});

    // Инициализация
    document.addEventListener('DOMContentLoaded', function() {{
        renderCards();
        document.getElementById('search').addEventListener('input', applyFilters);
        document.getElementById('sort').addEventListener('change', applyFilters);
        document.getElementById('onlyVerdict').addEventListener('change', applyFilters);
        document.getElementById('btnSave').addEventListener('click', saveSelected);
        document.getElementById('btnJson').addEventListener('click', downloadJson);
    }});
    </script>
    """

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MuraveiVision — Отчёт v2 — {Path(video_path).stem}</title>
{css}
<!-- JSZip для создания ZIP выбранных моментов -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
</head>
<body>
{header_html}

<!-- Панель управления -->
<div class="toolbar">
    <input type="text" id="search" placeholder="🔍 Поиск по классу...">
    <select id="sort">
        <option value="time">Сортировка: по времени</option>
        <option value="conf">Сортировка: по confidence</option>
    </select>
    <label><input type="checkbox" id="onlyVerdict"> Только с вердиктом слоя 4</label>
</div>

<!-- Сетка карточек -->
<main>
<div class="grid" id="grid"></div>
</main>

<!-- Плавающая панель -->
<div class="float-panel">
    <button class="float-btn" id="btnSave">💾 Сохранить выбранные (ZIP)</button>
    <button class="float-btn secondary" id="btnJson">📋 Скачать JSON</button>
</div>

<div class="footer">Сгенерировано MuraveiVision v2 · {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
{js}
</body>
</html>
"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    return str(out_file)