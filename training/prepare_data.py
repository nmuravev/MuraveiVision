"""Скачивает РЕАЛЬНЫЕ военные датасеты (UAV/спутник/фото) и сливает в единый
YOLO-датасет с классами MuraveiVision. СИНТЕТИКА ИЗ ИГР ИСКЛЮЧЕНА.
Терминал: PowerShell."""
import os, shutil, re, yaml, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(file).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from core.classes import MILITARY_CLASSES

ROOT = Path(file).parent.parent / "datasets" / "military_merged"
MANUAL = Path(file).parent.parent / "datasets" / "manual"  # сюда кладёте zip вручную

# ── ЧЁРНЫЙ СПИСОК: синтетика/игры — исключаем ──
BLACKLIST_PATTERNS = [
    "warno", "arma", "squad", "battlefield", "game", "synthetic",
    "virtual", "simulation", "render", "3d_model", "unity", "unreal"
]

def is_real_data(img_path: Path) -> bool:
    """Проверяет, что изображение из реального источника, а не из игры."""
    name_lower = img_path.name.lower()
    for pattern in BLACKLIST_PATTERNS:
        if pattern in name_lower:
            return False
    # Проверка метаданных (если есть exif)
    try:
        from PIL import Image
        img = Image.open(img_path)
        # Игры часто имеют специфичные метаданные
        if hasattr(img, "_getexif") and img._getexif():
            exif_str = str(img._getexif()).lower()
            if any(p in exif_str for p in ["unity", "unreal", "game"]):
                return False
    except Exception:
        pass
    return True

# ── ТОЧНЫЙ МАППИНГ: класс датасета -> наш класс ──
EXACT = {
    "tank": "tank", "Tank": "tank", "T-72": "tank", "T-90": "tank",
    "person": "soldier", "soldier": "soldier", "infantry": "soldier",
    "ah64": "military helicopter", "apache": "military helicopter",
    "2S1": "self-propelled artillery", "2S3": "self-propelled artillery",
    "puma": "armored personnel carrier", "turtle": "vehicle with cage armor",
    "vehicle": "armored vehicle", "truck": "military truck",
    "warship": "military boat", "ship": "military boat",
    "drone": "quadcopter drone", "uav": "quadcopter drone",
}

# ── FUZZY-ПРАВИЛА для неизвестных имён ──
FUZZY = [
    ("heli", "military helicopter"), ("drone", "quadcopter drone"),
    ("uav", "quadcopter drone"), ("jet", "military aircraft"),
    ("plane", "military aircraft"), ("aircraft", "military aircraft"),
    ("fighter", "military aircraft"), ("bomber", "military aircraft"),
    ("artiller", "self-propelled artillery"), ("howitzer", "howitzer"),
    ("tank", "tank"), ("apc", "armored personnel carrier"),
    ("bmp", "armored personnel carrier"), ("btr", "armored personnel carrier"),
    ("jeep", "military jeep"), ("truck", "military truck"),
    ("soldier", "soldier"), ("person", "soldier"),
    ("ship", "military boat"), ("boat", "military boat"),
    ("patrol", "military boat"),
]

def our_class(name):
    if name in EXACT: return EXACT[name]
    low = name.lower()
    for frag, cls in FUZZY:
        if frag in low: return cls
    return None

# ── СКАЧИВАНИЕ ИСТОЧНИКОВ (ТОЛЬКО РЕАЛЬНЫЕ) ──
def dl_roboflow(ws, proj, ver, dest):
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    rf.workspace(ws).project(proj).version(ver).download("yolov8", location=str(dest))

def dl_kaggle(slug, dest):
    os.system(f'kaggle datasets download -d {slug} -p "{dest}" --unzip')

# Источники: только реальные данные (UAV, спутник, фото)
SOURCES = [
    # Roboflow: реальные фото/видео
    ("roboflow", "uce03211-gmail-com", "yolo-military", 1),  # 2.9k real
    ("roboflow", "mfatih", "yolo-military-y9ufx-hlxef", 1),  # 2.2k real, 2S1/2S3
    ("roboflow", "capstone2025-mifho", "military-base-object-detection", 1),  # 12k real
    
    # Kaggle: реальные датасеты
    ("kaggle", "rookieengg/military-aircraft-detection-dataset-yolo-format", None),  # 73 типа авиации
    ("kaggle", "rawsi18/military-assets-dataset-12-classes-yolo8-format", None),  # 12 классов real
    ("kaggle", "killa92/military-object-detection-recognition-yolov11", None),  # KIIT-MiTA UAV
    
    # Mendeley (качать вручную, класть в datasets/manual/KIIT-MiTA/)
    # https://data.mendeley.com/datasets/9z7yrcrpjk — 3000 танков, 1359 дронов, 2644 человека
    # https://data.mendeley.com/datasets/... — MVRSD спутниковые снимки
]

# ── ПЕРЕМАППИНГ ОДНОЙ ПАПКИ YOLO ──
def ingest(src: Path, tag: str):
    """src — папка с images/ и labels/ (или train/val внутри)."""
    names_file = src / "classes.txt"
    if not names_file.exists():
        for cand in src.rglob("data.yaml"):
            names = yaml.safe_load(cand.read_text(encoding="utf-8")).get("names")
            if names:
                src_names = list(names.values()) if isinstance(names, dict) else names
            break
        else:
            print(f"⚠️ {src}: нет classes.txt/data.yaml — пропуск")
            return 0
    else:
        src_names = names_file.read_text(encoding="utf-8").split()

    idx_map = {}
    for i, n in enumerate(src_names):
        oc = our_class(n)
        if oc and oc in MILITARY_CLASSES:
            idx_map[i] = MILITARY_CLASSES.index(oc)

    moved = 0
    for split in (["train", "val"] if (src / "train").exists() else [None]):
        base = src / split if split else src
        out_split = "val" if split in ("val", "valid", "test") else "train"
        imgs = base / "images"
        labs = base / "labels"
        if not imgs.exists():
            continue
        for img in list(imgs.glob("*.jpg")) + list(imgs.glob("*.png")) + list(imgs.glob("*.jpeg")):
            # ПРОВЕРКА: только реальные данные
            if not is_real_data(img):
                continue
            lab = (labs / img.stem).with_suffix(".txt")
            if not lab.exists():
                continue
            lines = []
            for ln in lab.read_text().splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                ni = idx_map.get(int(p[0]))
                if ni is None:
                    continue
                lines.append(f"{ni} {p[1]} {p[2]} {p[3]} {p[4]}")
            if not lines:
                continue
            # 10% уходит в val
            if out_split == "train" and random.random() < 0.1:
                out_split = "val"
            (ROOT / "images" / out_split).mkdir(parents=True, exist_ok=True)
            (ROOT / "labels" / out_split).mkdir(parents=True, exist_ok=True)
            shutil.copy(img, ROOT / "images" / out_split / f"{tag}_{img.name}")
            (ROOT / "labels" / out_split / f"{tag}_{img.stem}.txt").write_text("\n".join(lines))
            moved += 1
    print(f"✅ {tag}: {moved} РЕАЛЬНЫХ кадров")
    return moved

def main():
    random.seed(42)
    total = 0
    for kind, a, b, ver in SOURCES:
        dest = ROOT.parent / "raw" / a.replace("/", "_")
        try:
            if kind == "roboflow":
                dl_roboflow(a, b, ver, dest)
            else:
                dl_kaggle(a, dest)
            total += ingest(dest, a.replace("/", "_").replace(".", ""))
        except Exception as e:
            print(f"⚠️ источник {a} не взят: {e}")
    
    # Ручные датасеты (Mendeley KIIT-MiTA и др.) — распакуйте в datasets/manual/*/
    if MANUAL.exists():
        for d in MANUAL.iterdir():
            if d.is_dir():
                total += ingest(d, d.name)
    
    # data.yaml с подмножеством классов, которые реально есть
    present = set()
    for lab in (ROOT / "labels").rglob("*.txt"):
        for ln in lab.read_text().splitlines():
            if ln.strip():
                present.add(int(ln.split()[0]))
    names = [MILITARY_CLASSES[i] for i in sorted(present)]
    (ROOT / "data.yaml").write_text(yaml.safe_dump({
        "path": str(ROOT), "train": "images/train", "val": "images/val", "names": names
    }, allow_unicode=True), encoding="utf-8")
    
    print(f"\n🎉 Итого: {total} РЕАЛЬНЫХ кадров")
    print(f"Классов в обучении: {len(names)}")
    print("Классы:", names)

if name == "main":
    main()