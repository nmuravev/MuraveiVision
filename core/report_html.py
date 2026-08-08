"""Генератор самодостаточного HTML-отчёта v2 (AMOLED-стиль + боковой фильтр).

generate_html(...) создаёт output/<video_stem>_report.html:
  - боковое меню со всеми классами, сгруппированными по категориям;
  - клик по имени класса на карточке = фильтр по этому классу;
  - облако активных тегов с кнопками сброса;
  - чекбоксы выбора карточек + сохранение выбранных в ZIP;
  - поиск по классу, сортировка (время/уверенность), фильтр "только с вердиктом";
  - НЕТ лимита 200 — все объекты;
  - миниатюры (520px, JPEG quality 70) встроены как base64;
  - классы на русском (через ru());
  - ЗАЩИТА: любая кривая структура объектов не роняет генерацию.
"""
import base64
import datetime
import json
from pathlib import Path

import cv2

from core.classes import ru
from core.paths import output_dir as _output_dir


# ── AMOLED-палитра ──
_C_BG = "#000000"
_C_PANEL = "#0D0D0D"
_C_CARD = "#0D0D0D"
_C_BORDER = "#1E1E1E"
_C_ACCENT = "#00E5FF"
_C_ACCENT_HOVER = "#00B8D4"
_C_OK = "#00FF88"
_C_SUCCESS = "#00FF88"
_C_ERR = "#FF3B30"
_C_ERROR = "#FF3B30"
_C_TEXT = "#E0E0E0"
_C_DIM = "#8A8A8A"
_C_TEXT_SEC = "#8A8A8A"

# ── Категории классов (ключевые слова для автоматической группировки) ──
_CATEGORY_KEYWORDS = [
    ("🚗 Техника", ["tank", "vehicle", "truck", "jeep", "artillery", "howitzer",
                    "missile launcher", "self-propelled", "armored", "motorcycle",
                    "atv", "snowmobile", "pickup", "bowser", "ambulance",
                    "evacuation", "train", "flatbed", "tanker"]),
    ("🔫 Оружие", ["rifle", "machine gun", "carbine", "submachine",
                   "anti-materiel", "suppressed", "grenade launcher",
                   "underbarrel", "cannon"]),
    ("💣 Взрывчатка", ["mine", "grenade", "rpg", "law", "at4", "shmel",
                       "missile", "atgm", "manpads", "explosive", "booby"]),
    ("👤 Люди", ["soldier", "person", "medic", "wounded", "fallen", "body", "ghillie"]),
    ("🎒 Снаряжение", ["helmet", "vest", "backpack", "tourniquet", "bandage",
                       "ifak", "stretcher", "magazine", "belt", "ammunition",
                       "shell", "cartridge"]),
    ("🛡️ Укрепления", ["trench", "bunker", "dugout", "foxhole", "barbed wire",
                       "obstacle", "hedgehog", "gabion", "sandbag", "checkpoint",
                       "watchtower", "berm", "ditch", "block"]),
    ("📡 Связь/РЭБ", ["antenna", "radio", "radar", "satellite",
                      "electronic warfare", "mast", "dish", "controller"]),
    ("🏠 Укрытия", ["tent", "net", "cage", "grille", "camouflage",
                    "camouflaged", "covered"]),
    ("🚁 Воздух", ["helicopter", "aircraft", "drone", "quadcopter", "fpv",
                   "fixed-wing", "reconnaissance", "propeller", "wreckage",
                   "fiber optic"]),
    ("🗑️ Следы/Мусор", ["bottle", "can", "wrapper", "paper", "box", "bag",
                        "garbage", "litter", "campfire", "smoke", "dust", "tire",
                        "footpath", "trampled", "disturbed", "fresh dug",
                        "discarded", "crater", "shrapnel", "branches", "foliage",
                        "ash", "soot", "muzzle blast", "snow", "glint", "sheet",
                        "cement", "sand", "gravel", "rebar", "mesh", "felt",
                        "debris", "sleepers", "track", "pallet", "generator",
                        "kitchen", "barrel", "crate", "jerry"]),
    ("🎯 Макеты", ["inflatable", "decoy", "dummy", "reflector"]),
]


def _categorize(class_en: str) -> str:
    """Определяет категорию класса по ключевым словам."""
    low = (class_en or "").lower()
    for cat, kws in _CATEGORY_KEYWORDS:
        if any(k in low for k in kws):
            return cat
    return "❓ Прочее"


def _safe_class_en(o) -> str:
    """Достаёт EN-имя класса из объекта детекции. Всегда возвращает str."""
    if isinstance(o, dict):
        v = o.get("class_en") or o.get("class") or o.get("name") or ""
    else:
        v = o
    if isinstance(v, dict):
        v = v.get("en") or v.get("name") or v.get("class") or ""
    if v is None:
        v = ""
    if not isinstance(v, str):
        v = str(v)
    return v


def _safe_class_ru(o) -> str:
    """Русское имя класса, гарантированно str (защита от unhashable dict)."""
    try:
        return str(ru(_safe_class_en(o)))
    except Exception:
        return ""


def _safe_conf(o) -> float:
    """Уверенность детекции, гарантированно число."""
    try:
        if isinstance(o, dict):
            return float(o.get("confidence", 0) or 0)
        return 0.0
    except Exception:
        return 0.0


def _frame_to_base64(frame, max_width=520, quality=70):
    h, w = frame.shape[:2]
    scale = max_width / w if w > max_width else 1.0
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def _draw_boxes(frame, objects):
    from core.draw import draw_boxes_pil
    return draw_boxes_pil(frame, objects)


def _esc(value) -> str:
    if value is None:
        return ""
    s = str(value)
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _build_header(video_path, report_data, settings):
    created = report_data.get("created_at", "")
    model = report_data.get("model", "")
    hw = report_data.get("hardware", {})
    moments_count = report_data.get("moments_count", 0)

    if isinstance(hw, dict):
        gpu = hw.get("gpu")
        cpu = hw.get("cpu", {})
        if gpu:
            hw_str = f"GPU: {_esc(gpu.get('name', ''))} ({gpu.get('vram_mb', 0) // 1024} GB)"
        else:
            hw_str = f"CPU: {_esc(cpu.get('brand', ''))}"
    else:
        hw_str = _esc(hw)

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
                  cloud_annotations=None, output_dir=None):
    report_data = report_data or {}
    cloud_annotations = cloud_annotations or {}

    out_dir = Path(output_dir) if output_dir else _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    video_stem = Path(video_path).stem
    out_file = out_dir / f"{video_stem}_report.html"

    header_html = _build_header(video_path, report_data, settings)

    # Собираем статистику классов по всем моментам (защита: только str в set)
    class_counts = {}
    for m in moments:
        seen = set()
        for o in m.get("objects", []):
            cls_ru = _safe_class_ru(o)
            if cls_ru and cls_ru not in seen:
                class_counts[cls_ru] = class_counts.get(cls_ru, 0) + 1
                seen.add(cls_ru)

    # en->ru словарь для обратной трассировки категорий (защита: только str)
    en_to_ru = {}
    for m in moments:
        for o in m.get("objects", []):
            en = _safe_class_en(o)
            if en:
                en_to_ru[str(ru(en))] = en

    categories = {}
    for cls_ru, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        cls_en = en_to_ru.get(cls_ru, cls_ru)
        cat = _categorize(cls_en)
        categories.setdefault(cat, []).append((cls_ru, cnt))

    # ── CSS ──
    css = f"""
    <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
        background: {_C_BG}; color: {_C_TEXT};
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        display: flex; min-height: 100vh;
    }}
    .mono {{ font-family: 'Consolas', 'Courier New', monospace; }}

    /* ── Боковое меню ── */
    .sidebar {{
        width: 320px; min-width: 320px;
        background: {_C_CARD};
        border-right: 1px solid {_C_BORDER};
        padding: 20px; overflow-y: auto;
        position: sticky; top: 0; height: 100vh;
    }}
    .sidebar-header {{
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 16px; padding-bottom: 12px;
        border-bottom: 1px solid {_C_BORDER};
    }}
    .sidebar-header h2 {{
        margin: 0; font-size: 18px; color: {_C_ACCENT};
    }}
    .sidebar-search {{
        width: 100%; padding: 10px 12px;
        background: #000; border: 1px solid {_C_BORDER};
        color: {_C_TEXT}; border-radius: 6px;
        font-size: 13px; margin-bottom: 16px;
    }}
    .sidebar-search:focus {{
        outline: none; border-color: {_C_ACCENT};
    }}
    .category {{
        margin-bottom: 8px;
        background: #0a0a0a;
        border: 1px solid {_C_BORDER};
        border-radius: 8px;
        overflow: hidden;
    }}
    .category-head {{
        padding: 10px 12px;
        background: linear-gradient(90deg, #111 0%, #0a0a0a 100%);
        display: flex; justify-content: space-between; align-items: center;
        cursor: pointer; user-select: none;
        transition: background 0.2s;
    }}
    .category-head:hover {{ background: #161616; }}
    .category-head .title {{
        font-weight: 600; font-size: 14px; color: {_C_TEXT};
        display: flex; align-items: center; gap: 8px;
    }}
    .category-head .count {{
        background: {_C_ACCENT}; color: #000;
        padding: 2px 8px; border-radius: 10px;
        font-size: 11px; font-weight: bold;
    }}
    .category-head .arrow {{
        color: {_C_DIM}; transition: transform 0.2s;
        font-size: 10px;
    }}
    .category.collapsed .arrow {{ transform: rotate(-90deg); }}
    .category.collapsed .category-body {{ display: none; }}
    .category-body {{ padding: 6px 0; }}
    .category-actions {{
        display: flex; gap: 6px;
        padding: 4px 12px 8px;
        border-bottom: 1px solid #1a1a1a;
    }}
    .category-actions button {{
        flex: 1; padding: 4px 8px; font-size: 11px;
        background: transparent; color: {_C_DIM};
        border: 1px solid #2a2a2a; border-radius: 4px;
        cursor: pointer; transition: all 0.15s;
    }}
    .category-actions button:hover {{
        background: {_C_ACCENT}; color: #000;
        border-color: {_C_ACCENT};
    }}
    .class-label {{
        display: flex; align-items: center; gap: 8px;
        padding: 6px 12px; cursor: pointer;
        transition: background 0.15s;
        font-size: 13px;
    }}
    .class-label:hover {{ background: #151515; }}
    .class-label input {{ accent-color: {_C_ACCENT}; cursor: pointer; }}
    .class-label .name {{ flex: 1; }}
    .class-label .count {{
        color: {_C_DIM}; font-size: 11px;
        font-family: 'Consolas', monospace;
    }}
    .class-label.empty {{ display: none; }}

    /* ── Основной контент ── */
    .content {{
        flex: 1; padding: 20px 30px 100px;
        min-width: 0;
    }}

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

    /* ── Панель инструментов ── */
    .toolbar {{
        position: sticky; top: 0; z-index: 100;
        background: {_C_CARD}; border: 1px solid {_C_BORDER};
        border-radius: 10px; padding: 12px; margin-bottom: 12px;
        display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
    }}
    .toolbar input[type="text"] {{
        background: #000; border: 1px solid {_C_BORDER}; color: {_C_TEXT};
        padding: 8px 12px; border-radius: 6px; font-size: 14px; width: 200px;
    }}
    .toolbar input[type="text"]:focus {{
        outline: none; border-color: {_C_ACCENT};
    }}
    .toolbar select {{
        background: #000; border: 1px solid {_C_BORDER}; color: {_C_TEXT};
        padding: 8px 12px; border-radius: 6px; font-size: 14px;
    }}
    .toolbar label {{ color: {_C_TEXT_SEC}; font-size: 13px; cursor: pointer; }}

    /* ── Активные фильтры ── */
    .active-filters {{
        background: {_C_CARD}; border: 1px solid {_C_BORDER};
        border-radius: 10px; padding: 10px 12px; margin-bottom: 12px;
        display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
        min-height: 44px;
    }}
    .active-filters.empty {{
        justify-content: center;
        color: {_C_DIM}; font-size: 13px; font-style: italic;
    }}
    .af-label {{
        color: {_C_TEXT_SEC}; font-size: 13px; font-weight: 500;
        margin-right: 4px;
    }}
    .active-tag {{
        background: linear-gradient(135deg, #003a47 0%, #001f28 100%);
        border: 1px solid {_C_ACCENT};
        color: {_C_ACCENT}; padding: 4px 10px;
        border-radius: 14px; font-size: 12px;
        display: inline-flex; align-items: center; gap: 6px;
        cursor: pointer; transition: all 0.2s;
    }}
    .active-tag:hover {{
        background: {_C_ACCENT}; color: #000;
    }}
    .active-tag .close {{
        font-weight: bold; font-size: 14px; line-height: 1;
    }}
    .reset-btn {{
        background: transparent; color: {_C_ERR};
        border: 1px solid {_C_ERR}; padding: 4px 12px;
        border-radius: 14px; font-size: 12px;
        cursor: pointer; transition: all 0.2s;
        margin-left: auto;
    }}
    .reset-btn:hover {{
        background: {_C_ERR}; color: #fff;
    }}

    /* ── Счётчик ── */
    .counter {{
        color: {_C_DIM}; font-size: 13px;
        margin-bottom: 16px; padding: 0 4px;
    }}
    .counter .highlight {{ color: {_C_ACCENT}; font-weight: 600; }}

    /* ── Сетка карточек ── */
    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
        gap: 16px;
    }}
    .card {{
        background: {_C_CARD}; border: 1px solid {_C_BORDER};
        border-radius: 10px; overflow: hidden;
        transition: all 0.25s;
        animation: fadeIn 0.25s ease-out;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .card:hover {{
        border-color: {_C_ACCENT};
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.25);
        transform: translateY(-2px);
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
        cursor: pointer; transition: all 0.15s;
        user-select: none;
    }}
    .meta-class:hover {{
        background: {_C_ACCENT}; color: #000;
        border-color: {_C_ACCENT};
    }}
    .meta-class.active {{
        background: {_C_ACCENT}; color: #000;
        border-color: {_C_ACCENT};
    }}
    .meta-conf {{ color: {_C_SUCCESS}; font-family: 'Consolas', monospace; }}
    .meta-verdict {{
        color: #ffcc88; font-size: 12px; font-style: italic;
        background: #1a1500; padding: 2px 8px; border-radius: 4px;
        max-width: 300px; overflow: hidden; text-overflow: ellipsis;
        white-space: nowrap;
    }}

    /* ── Плавающая панель ── */
    .float-panel {{
        position: fixed; bottom: 0; left: 320px; right: 0;
        background: {_C_CARD}; border-top: 1px solid {_C_BORDER};
        padding: 12px 20px; display: flex; gap: 12px;
        justify-content: center; z-index: 200;
    }}
    .float-btn {{
        background: {_C_ACCENT}; color: #000; border: none;
        padding: 10px 20px; border-radius: 6px; font-size: 14px;
        font-weight: bold; cursor: pointer;
        transition: background 0.2s;
    }}
    .float-btn:hover {{ background: {_C_ACCENT_HOVER}; }}
    .float-btn.secondary {{ background: #333; color: {_C_TEXT}; }}
    .float-btn:disabled {{ background: #333; color: #666; cursor: not-allowed; }}

    .footer {{ text-align: center; color: {_C_TEXT_SEC}; margin-top: 30px; font-size: 12px; }}

    /* ── Пустое состояние ── */
    .empty-state {{
        text-align: center; padding: 60px 20px;
        color: {_C_DIM}; font-size: 16px;
    }}
    .empty-state .icon {{ font-size: 48px; margin-bottom: 12px; }}

    @media (max-width: 1200px) {{
        body {{ flex-direction: column; }}
        .sidebar {{ width: 100%; min-width: 100%; height: auto; position: relative; }}
        .float-panel {{ left: 0; }}
    }}
    </style>
    """

    # ── Данные для JS (защита: только str/числа) ──
    moments_js_data = []
    for i, m in enumerate(moments):
        objs = []
        for o in m.get("objects", []):
            cls_en = _safe_class_en(o)
            cls_ru = _safe_class_ru(o)
            objs.append({
                "class_en": cls_en,
                "class": cls_ru,
                "category": _categorize(cls_en),
                "confidence": _safe_conf(o),
            })
        moments_js_data.append({
            "idx": i,
            "timestamp": m.get("timestamp", ""),
            "frame": m.get("frame", 0),
            "objects": objs,
            "has_verdict": str(m.get("frame", 0)) in [str(k) for k in cloud_annotations.keys()],
        })

    # Миниатюры (защита: битый кадр не роняет весь отчёт)
    cap = cv2.VideoCapture(video_path)
    thumbs_data = {}
    for i, moment in enumerate(moments):
        try:
            frame_idx = moment.get("frame", 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            annotated = _draw_boxes(frame, moment.get("objects", []))
            data_url = _frame_to_base64(annotated, max_width=520, quality=70)
            if data_url:
                thumbs_data[i] = data_url
        except Exception:
            continue
    cap.release()

    # JSON-отчёт (base64)
    report_json = json.dumps({
        "video": video_path,
        "model": report_data.get("model", ""),
        "hardware": report_data.get("hardware", {}),
        "created_at": report_data.get("created_at", ""),
        "moments_count": len(moments),
        "moments": moments,
    }, ensure_ascii=False, indent=2)
    json_b64 = base64.b64encode(report_json.encode("utf-8")).decode("utf-8")

    # ── JS ──
    js = f"""
    <script>
    var _moments = {json.dumps(moments_js_data, ensure_ascii=False)};
    var _thumbs = {json.dumps(thumbs_data, ensure_ascii=False)};
    var _json_b64 = "{json_b64}";
    var _verdicts = {json.dumps({{str(k): v[:200] for k, v in cloud_annotations.items()}}, ensure_ascii=False)};

    // Состояние фильтров
    var _activeClasses = new Set(); // классы в боковом меню
    var _activeTags = new Set();    // теги (быстрые фильтры)
    var LS_KEY = 'mv_filters_' + location.pathname;

    // ── Рендер карточек ──
    function renderCards() {{
        var grid = document.getElementById('grid');
        grid.innerHTML = '';
        _moments.forEach(function(m) {{
            var thumb = _thumbs[m.idx] || '';
            var verdict = _verdicts[String(m.frame)] || '';
            var classesHtml = m.objects.map(function(o) {{
                var isActive = _activeTags.has(o.class);
                var activeClass = isActive ? ' active' : '';
                return '<span class="meta-class' + activeClass + '" data-class="' + escapeHtml(o.class) + '">' + escapeHtml(o.class) + '</span>';
            }}).join('');
            var maxConf = m.objects.length ? Math.max.apply(null, m.objects.map(function(o) {{ return o.confidence; }})) : 0;
            var verdictHtml = verdict ? '<span class="meta-verdict" title="' + escapeHtml(verdict) + '">☁️ ' + escapeHtml(verdict.substring(0, 60)) + (verdict.length > 60 ? '…' : '') + '</span>' : '';
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
                '<span class="meta-conf">увер. ' + maxConf.toFixed(2) + '</span>' +
                verdictHtml +
                '</div>';
            grid.appendChild(card);
        }});
        applyFilters();
    }}

    function escapeHtml(s) {{
        if (s === null || s === undefined) return '';
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }}

    // ── Фильтры ──
    function applyFilters() {{
        var search = document.getElementById('search').value.toLowerCase();
        var sort = document.getElementById('sort').value;
        var onlyVerdict = document.getElementById('onlyVerdict').checked;
        var cards = Array.from(document.querySelectorAll('.card'));
        var visibleCount = 0;
        var totalCount = cards.length;

        cards.forEach(function(c) {{
            var idx = c.dataset.idx;
            var m = _moments[idx];
            var cardClasses = m.objects.map(function(o) {{ return o.class; }});
            var cardClassesLower = cardClasses.map(function(s) {{ return s.toLowerCase(); }});

            // Поиск
            var matchSearch = !search || cardClassesLower.some(function(s) {{ return s.includes(search); }});

            // Только с вердиктом
            var matchVerdict = !onlyVerdict || c.dataset.hasVerdict === '1';

            // Фильтр по классам (из sidebar + теги = объединение)
            var allFilterClasses = new Set([..._activeClasses, ..._activeTags]);
            var matchClass;
            if (allFilterClasses.size === 0) {{
                matchClass = true;
            }} else {{
                matchClass = cardClasses.some(function(cls) {{ return allFilterClasses.has(cls); }});
            }}

            var visible = matchSearch && matchVerdict && matchClass;
            c.classList.toggle('hidden', !visible);
            if (visible) visibleCount++;
        }});

        // Сортировка видимых
        var visible = cards.filter(function(c) {{ return !c.classList.contains('hidden'); }});
        var grid = document.getElementById('grid');
        visible.sort(function(a, b) {{
            if (sort === 'time') return a.dataset.ts.localeCompare(b.dataset.ts);
            if (sort === 'conf') return parseFloat(b.dataset.conf) - parseFloat(a.dataset.conf);
            return 0;
        }});
        visible.forEach(function(c) {{ grid.appendChild(c); }});

        // Обновляем счётчик
        var counterEl = document.getElementById('counter');
        counterEl.innerHTML = 'Показано <span class="highlight">' + visibleCount + '</span> из <span class="highlight">' + totalCount + '</span> карточек';

        // Пустое состояние
        var emptyEl = document.getElementById('emptyState');
        if (emptyEl) emptyEl.style.display = visibleCount === 0 ? 'block' : 'none';

        renderActiveTags();
        updateSidebarCounts();
        saveState();
    }}

    // ── Активные теги ──
    function renderActiveTags() {{
        var container = document.getElementById('activeFilters');
        container.innerHTML = '';
        var allFilters = new Set([..._activeClasses, ..._activeTags]);

        if (allFilters.size === 0) {{
            container.classList.add('empty');
            container.innerHTML = '<span>💡 Фильтры не активны. Кликните на класс в боковом меню или на название класса в карточке</span>';
            return;
        }}
        container.classList.remove('empty');

        var label = document.createElement('span');
        label.className = 'af-label';
        label.textContent = '🎯 Активные фильтры:';
        container.appendChild(label);

        _activeTags.forEach(function(cls) {{
            var tag = document.createElement('span');
            tag.className = 'active-tag';
            tag.innerHTML = escapeHtml(cls) + ' <span class="close">×</span>';
            tag.onclick = function(e) {{
                e.stopPropagation();
                _activeTags.delete(cls);
                applyFilters();
            }};
            container.appendChild(tag);
        }});

        _activeClasses.forEach(function(cls) {{
            var tag = document.createElement('span');
            tag.className = 'active-tag';
            tag.innerHTML = '☑ ' + escapeHtml(cls) + ' <span class="close">×</span>';
            tag.onclick = function(e) {{
                e.stopPropagation();
                _activeClasses.delete(cls);
                updateSidebarCheckbox(cls, false);
                applyFilters();
            }};
            container.appendChild(tag);
        }});

        var resetAllBtn = document.createElement('button');
        resetAllBtn.className = 'reset-btn';
        resetAllBtn.textContent = '🔄 Сбросить всё';
        resetAllBtn.onclick = resetAll;
        container.appendChild(resetAllBtn);
    }}

    // ── Сброс ──
    function resetAll() {{
        _activeClasses.clear();
        _activeTags.clear();
        document.querySelectorAll('.class-label input[type="checkbox"]').forEach(function(cb) {{ cb.checked = false; }});
        document.getElementById('search').value = '';
        document.getElementById('onlyVerdict').checked = false;
        applyFilters();
    }}

    function updateSidebarCheckbox(cls, checked) {{
        var cb = document.querySelector('.class-label input[data-class="' + CSS.escape(cls) + '"]');
        if (cb) cb.checked = checked;
    }}

    function updateSidebarCounts() {{
        // Подсвечиваем категории, где есть активные
        document.querySelectorAll('.category').forEach(function(cat) {{
            var hasActive = false;
            cat.querySelectorAll('.class-label input').forEach(function(cb) {{
                if (_activeClasses.has(cb.dataset.class)) hasActive = true;
            }});
            cat.classList.toggle('has-active', hasActive);
        }});
    }}

    // ── Рендер бокового меню ──
    function renderSidebar() {{
        var container = document.getElementById('sidebarCategories');
        container.innerHTML = '';

        // Группируем по категориям
        var cats = {{}};
        _moments.forEach(function(m) {{
            m.objects.forEach(function(o) {{
                if (!cats[o.category]) cats[o.category] = {{}};
                if (!cats[o.category][o.class]) cats[o.category][o.class] = 0;
                cats[o.category][o.class]++;
            }});
        }});

        // Сортируем категории по количеству
        var catOrder = Object.keys(cats).sort(function(a, b) {{
            var countA = Object.keys(cats[a]).length;
            var countB = Object.keys(cats[b]).length;
            return countB - countA;
        }});

        catOrder.forEach(function(catName) {{
            var classes = cats[catName];
            var classEntries = Object.entries(classes).sort(function(a, b) {{ return b[1] - a[1]; }});

            var catEl = document.createElement('div');
            catEl.className = 'category';

            var head = document.createElement('div');
            head.className = 'category-head';
            head.innerHTML = '<span class="title">' + escapeHtml(catName) + ' <span class="count">' + classEntries.length + '</span></span><span class="arrow">▼</span>';
            head.onclick = function() {{ catEl.classList.toggle('collapsed'); }};
            catEl.appendChild(head);

            var body = document.createElement('div');
            body.className = 'category-body';

            // Кнопки "Выбрать все / Снять"
            var actions = document.createElement('div');
            actions.className = 'category-actions';
            var selectAllBtn = document.createElement('button');
            selectAllBtn.textContent = '✓ Выбрать все';
            selectAllBtn.onclick = function(e) {{
                e.stopPropagation();
                classEntries.forEach(function(entry) {{
                    _activeClasses.add(entry[0]);
                }});
                catEl.querySelectorAll('.class-label input').forEach(function(cb) {{ cb.checked = true; }});
                applyFilters();
            }};
            var clearAllBtn = document.createElement('button');
            clearAllBtn.textContent = '✕ Снять';
            clearAllBtn.onclick = function(e) {{
                e.stopPropagation();
                classEntries.forEach(function(entry) {{
                    _activeClasses.delete(entry[0]);
                }});
                catEl.querySelectorAll('.class-label input').forEach(function(cb) {{ cb.checked = false; }});
                applyFilters();
            }};
            actions.appendChild(selectAllBtn);
            actions.appendChild(clearAllBtn);
            body.appendChild(actions);

            classEntries.forEach(function(entry) {{
                var cls = entry[0];
                var count = entry[1];
                var label = document.createElement('label');
                label.className = 'class-label';
                label.dataset.search = cls.toLowerCase();

                var cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.dataset.class = cls;
                cb.checked = _activeClasses.has(cls);
                cb.onchange = function() {{
                    if (cb.checked) _activeClasses.add(cls);
                    else _activeClasses.delete(cls);
                    applyFilters();
                }};

                var nameSpan = document.createElement('span');
                nameSpan.className = 'name';
                nameSpan.textContent = cls;

                var countSpan = document.createElement('span');
                countSpan.className = 'count';
                countSpan.textContent = count;

                label.appendChild(cb);
                label.appendChild(nameSpan);
                label.appendChild(countSpan);
                body.appendChild(label);
            }});

            catEl.appendChild(body);
            container.appendChild(catEl);
        }});
    }}

    // ── Поиск внутри бокового меню ──
    function filterSidebar() {{
        var q = document.getElementById('sidebarSearch').value.toLowerCase();
        document.querySelectorAll('.class-label').forEach(function(label) {{
            var match = !q || label.dataset.search.includes(q);
            label.classList.toggle('empty', !match);
        }});
        // Скрываем пустые категории
        document.querySelectorAll('.category').forEach(function(cat) {{
            var visible = Array.from(cat.querySelectorAll('.class-label')).some(function(l) {{ return !l.classList.contains('empty'); }});
            cat.style.display = visible ? '' : 'none';
        }});
    }}

    // ── Сохранение/загрузка состояния ──
    function saveState() {{
        try {{
            var state = {{
                activeClasses: Array.from(_activeClasses),
                activeTags: Array.from(_activeTags),
                search: document.getElementById('search').value,
                sort: document.getElementById('sort').value,
                onlyVerdict: document.getElementById('onlyVerdict').checked,
            }};
            localStorage.setItem(LS_KEY, JSON.stringify(state));
        }} catch (e) {{}}
    }}

    function loadState() {{
        try {{
            var state = JSON.parse(localStorage.getItem(LS_KEY));
            if (!state) return;
            (state.activeClasses || []).forEach(function(c) {{ _activeClasses.add(c); }});
            (state.activeTags || []).forEach(function(c) {{ _activeTags.add(c); }});
            document.querySelectorAll('.class-label input').forEach(function(cb) {{
                if (_activeClasses.has(cb.dataset.class)) cb.checked = true;
            }});
            if (state.search) document.getElementById('search').value = state.search;
            if (state.sort) document.getElementById('sort').value = state.sort;
            if (state.onlyVerdict) document.getElementById('onlyVerdict').checked = state.onlyVerdict;
        }} catch (e) {{}}
    }}

    // ── Сохранить выбранные (ZIP) ──
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

    function downloadJson() {{
        var json = atob(_json_b64);
        var blob = new Blob([json], {{type: 'application/json'}});
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'report.json';
        a.click();
    }}

    // ── Клик по классу в карточке = быстрый фильтр (тег) ──
    document.addEventListener('click', function(e) {{
        // Клик по метке класса
        if (e.target.classList.contains('meta-class')) {{
            var cls = e.target.dataset.class;
            if (_activeTags.has(cls)) _activeTags.delete(cls);
            else _activeTags.add(cls);
            applyFilters();
            return;
        }}

        // Клик по чекбоксу карточки — игнорируем
        if (e.target.classList.contains('card-check')) return;

        // Клик по самой карточке — сигнал (PRO)
        var card = e.target.closest('.card');
        if (card) {{
            var idx = card.dataset.idx;
            var m = _moments[idx];
            if (typeof qt !== 'undefined' && window.pyBridge) {{
                window.pyBridge.seekToMoment(m.frame, m.timestamp);
            }}
            card.style.borderColor = '{_C_SUCCESS}';
            setTimeout(function() {{ card.style.borderColor = ''; }}, 500);
        }}
    }});

    // ── Горячие клавиши ──
    document.addEventListener('keydown', function(e) {{
        if (e.key === 'Escape') resetAll();
    }});

    // ── Инициализация ──
    document.addEventListener('DOMContentLoaded', function() {{
        renderSidebar();
        renderCards();
        loadState();
        applyFilters();

        document.getElementById('search').addEventListener('input', applyFilters);
        document.getElementById('sort').addEventListener('change', applyFilters);
        document.getElementById('onlyVerdict').addEventListener('change', applyFilters);
        document.getElementById('sidebarSearch').addEventListener('input', filterSidebar);
        document.getElementById('btnSave').addEventListener('click', saveSelected);
        document.getElementById('btnJson').addEventListener('click', downloadJson);
        document.getElementById('resetAllBtn').addEventListener('click', resetAll);
    }});
    </script>
    """

    # ── HTML-каркас ──
    sidebar_html = f"""
    <aside class="sidebar">
        <div class="sidebar-header">
            <h2>🏷️ Фильтр по классам</h2>
        </div>
        <input type="text" class="sidebar-search" id="sidebarSearch" placeholder="🔍 Найти класс...">
        <div style="margin-bottom: 12px;">
            <button class="reset-btn" id="resetAllBtn" style="width: 100%; margin-left: 0;">🔄 Сбросить все фильтры</button>
        </div>
        <div id="sidebarCategories"></div>
    </aside>
    """

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MuraveiVision — Отчёт v2 — {Path(video_path).stem}</title>
{css}
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
</head>
<body>
{sidebar_html}

<main class="content">
{header_html}

<div class="toolbar">
    <input type="text" id="search" placeholder="🔍 Поиск по классу...">
    <select id="sort">
        <option value="time">⏱ По времени</option>
        <option value="conf">📊 По уверенности</option>
    </select>
    <label><input type="checkbox" id="onlyVerdict"> ☁️ Только с вердиктом</label>
</div>

<div class="active-filters empty" id="activeFilters">
    <span>💡 Фильтры не активны. Кликните на класс в боковом меню или на название класса в карточке</span>
</div>

<div class="counter" id="counter">Показано <span class="highlight">0</span> из <span class="highlight">0</span> карточек</div>

<main>
<div class="grid" id="grid"></div>
<div class="empty-state" id="emptyState" style="display:none;">
    <div class="icon">🔍</div>
    <div>По текущим фильтрам ничего не найдено</div>
</div>
</main>

<div class="footer">Сгенерировано MuraveiVision v2 · {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Нажмите <kbd>Esc</kbd> для сброса фильтров</div>
</main>

<div class="float-panel">
    <button class="float-btn" id="btnSave">💾 Сохранить выбранные (ZIP)</button>
    <button class="float-btn secondary" id="btnJson">📋 Скачать JSON</button>
</div>

{js}
</body>
</html>
"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    return str(out_file)