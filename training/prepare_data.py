"""Скачивает РЕАЛЬНЫЕ военные датасеты (UAV/спутник/фото) и сливает в единый
YOLO-датасет с классами MuraveiVision. СИНТЕТИКА ИЗ ИГР ИСКЛЮЧЕНА.
Терминал: PowerShell."""
import os, shutil, random, sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from core.classes import MILITARY_CLASSES

ROOT = Path(__file__).parent.parent / "datasets" / "military_merged"
MANUAL = Path(__file__).parent.parent / "datasets" / "manual"

# ── ЧЁРНЫЙ СПИСОК: синтетика/игры — исключаем ──
BLACKLIST_PATTERNS = [
    "warno", "arma", "squad", "battlefield", "game", "synthetic",
    "virtual", "simulation", "render", "3d_model", "unity", "unreal"
]

def is_real_data(img_path: Path) -> bool:
    name_lower = img_path.name.lower()
    for pattern in BLACKLIST_PATTERNS:
        if pattern in name_lower:
            return False
    try:
        from PIL import Image
        img = Image.open(img_path)
        if hasattr(img, "_getexif") and img._getexif():
            exif_str = str(img._getexif()).lower()
            if any(p in exif_str for p in ["unity", "unreal", "game"]):
                return False
    except Exception:
        pass
    return True

# ── ТОЧНЫЙ МАППИНГ (расширенный под реальные датасеты) ──
EXACT = {
    # Танки
    "tank": "tank", "Tank": "tank", "T-72": "tank", "T-90": "tank",
    "t72": "tank", "t90": "tank", "MBT": "tank", "main battle tank": "tank",
    "tank turret in ground": "tank turret in ground",
    "dug-in tank": "tank turret in ground",
    
    # БМП/БТР/БМД
    "apc": "armored personnel carrier", "APC": "armored personnel carrier",
    "bmp": "armored personnel carrier", "BMP": "armored personnel carrier",
    "bmp1_2": "armored personnel carrier", "bmp3": "armored personnel carrier",
    "btr": "armored personnel carrier", "BTR": "armored personnel carrier",
    "bmd1_2": "armored personnel carrier", "bmd3_4": "armored personnel carrier",
    "puma": "armored personnel carrier", "bradley": "armored personnel carrier",
    "ifv": "armored personnel carrier", "IFV": "armored personnel carrier",
    "brdm2": "armored personnel carrier", "cv90": "armored personnel carrier",
    "mtlb": "armored personnel carrier",
    
    # Артиллерия/РСЗО
    "2S1": "self-propelled artillery", "2S3": "self-propelled artillery",
    "howitzer": "self-propelled artillery", "artillery": "self-propelled artillery",
    "SPG": "self-propelled artillery", "mortar": "self-propelled artillery",
    "grad": "self-propelled artillery", "s60": "self-propelled artillery",
    
    # Солдаты
    "person": "soldier", "soldier": "soldier", "infantry": "soldier",
    "fighter": "soldier", "troop": "soldier", "troops": "soldier",
    "paratrooper": "soldier", "sniper": "soldier",
    
    # Авиация
    "ah64": "military helicopter", "apache": "military helicopter",
    "helicopter": "military helicopter", "heli": "military helicopter",
    "mi8": "military helicopter",
    "jet": "military aircraft", "fighter jet": "military aircraft",
    "su25": "military aircraft", "su34": "military aircraft",
    "mig": "military aircraft", "aircraft": "military aircraft",
    "Aircraft": "military aircraft",
    "plane": "military aircraft", "bomber": "military aircraft",
    
    # Дроны
    "drone": "quadcopter drone", "uav": "quadcopter drone", "UAV": "quadcopter drone",
    "fpv": "quadcopter drone", "FPV": "quadcopter drone",
    "mavic": "quadcopter drone", "dji": "quadcopter drone",
    
    # Техника
    "vehicle": "armored vehicle", "armored vehicle": "armored vehicle",
    "military-vehicle": "armored vehicle",
    "truck": "military truck", "ural": "military truck", "kamaz": "military truck",
    "tigr": "military truck",
    "jeep": "military jeep", "humvee": "military jeep", "hummer": "military jeep",
    "fuel": "military truck",
    
    # Флот
    "ship": "military boat", "boat": "military boat", "warship": "military boat",
    "patrol boat": "military boat",
    
    # Укрытия
    "bunker": "concrete bunker", "shelter": "concrete bunker",
    "trench": "trench", "foxhole": "trench",
    
    # Мусор/следы/эффекты
    "wreck": "destroyed vehicle", "destroyed": "destroyed vehicle",
    "debris": "destroyed vehicle",
    "fire": "fire", "smoke": "smoke", "Missile": "missile",
    
    # Специфика
    "turtle": "vehicle with cage armor", "cage armor": "vehicle with cage armor",
    "cope cage": "vehicle with cage armor",
}

FUZZY = [
    ("tank", "tank"), ("t72", "tank"), ("t90", "tank"),
    ("bmp", "armored personnel carrier"), ("btr", "armored personnel carrier"),
    ("bmd", "armored personnel carrier"), ("apc", "armored personnel carrier"),
    ("ifv", "armored personnel carrier"), ("brdm", "armored personnel carrier"),
    ("cv90", "armored personnel carrier"), ("mtlb", "armored personnel carrier"),
    ("artiller", "self-propelled artillery"), ("howitzer", "self-propelled artillery"),
    ("grad", "self-propelled artillery"), ("s60", "self-propelled artillery"),
    ("soldier", "soldier"), ("person", "soldier"), ("infantry", "soldier"),
    ("troop", "soldier"), ("fighter", "soldier"),
    ("heli", "military helicopter"), ("apache", "military helicopter"),
    ("ah64", "military helicopter"), ("mi8", "military helicopter"),
    ("jet", "military aircraft"), ("plane", "military aircraft"),
    ("aircraft", "military aircraft"), ("su25", "military aircraft"),
    ("drone", "quadcopter drone"), ("uav", "quadcopter drone"),
    ("fpv", "quadcopter drone"), ("mavic", "quadcopter drone"),
    ("vehicle", "armored vehicle"), ("military-vehicle", "armored vehicle"),
    ("truck", "military truck"), ("ural", "military truck"), ("kamaz", "military truck"),
    ("tigr", "military truck"), ("fuel", "military truck"),
    ("jeep", "military jeep"), ("humvee", "military jeep"),
    ("ship", "military boat"), ("boat", "military boat"),
    ("bunker", "concrete bunker"), ("trench", "trench"),
    ("wreck", "destroyed vehicle"), ("destroyed", "destroyed vehicle"),
    ("fire", "fire"), ("smoke", "smoke"), ("missile", "missile"),
]

def our_class(name):
    if name in EXACT:
        return EXACT[name]
    low = name.lower()
    for frag, cls in FUZZY:
        if frag in low:
            return cls
    return None

def dl_roboflow(ws, proj, ver, dest):
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    rf.workspace(ws).project(proj).version(ver).download(
        "yolov8", location=str(dest))

def dl_kaggle(slug, dest):
    os.system(f'kaggle datasets download -d {slug} -p "{dest}" --unzip')

SOURCES = [
    ("roboflow", "uce03211-gmail-com", "yolo-military", 1),
    ("roboflow", "mfatih", "yolo-military-y9ufx-hlxef", 1),
    ("roboflow", "capstone2025-mifho", "military-base-object-detection", 1),
    ("kaggle", "rookieengg/military-aircraft-detection-dataset-yolo-format", None, None),
    ("kaggle", "rawsi18/military-assets-dataset-12-classes-yolo8-format", None, None),
    ("kaggle", "killa92/military-object-detection-recognition-yolov11", None, None),
]

def ingest(src: Path, tag: str):
    print(f"\n🔍 Анализирую: {tag}")
    
    # Создаём ROOT заранее
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (ROOT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (ROOT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ROOT / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    names_file = src / "classes.txt"
    if not names_file.exists():
        src_names = None
        for cand in src.rglob("data.yaml"):
            names = yaml.safe_load(cand.read_text(encoding="utf-8")).get("names")
            if names:
                src_names = (list(names.values())
                             if isinstance(names, dict) else names)
                break
        if src_names is None:
            print(f"⚠️ {src}: нет classes.txt/data.yaml — пропуск")
            return 0
    else:
        src_names = [ln.strip() for ln in
                     names_file.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]

    print(f"   Классы в датасете ({len(src_names)}): {src_names}")
    idx_map = {}
    for i, n in enumerate(src_names):
        oc = our_class(n)
        if oc and oc in MILITARY_CLASSES:
            idx_map[i] = MILITARY_CLASSES.index(oc)
    print(f"   idx_map: {idx_map}")

    moved = 0
    skipped_no_img = 0
    skipped_no_lab = 0
    skipped_no_lines = 0
    
    # Ищем все картинки рекурсивно
    all_imgs = list(src.rglob("*.jpg")) + list(src.rglob("*.png")) + list(src.rglob("*.jpeg"))
    print(f"   Всего картинок найдено: {len(all_imgs)}")
    
    for img in all_imgs:
        if not is_real_data(img):
            continue
        
        # Ищем разметку: 1) рядом в labels/, 2) тот же путь но .txt, 3) рекурсивно
        lab = None
        # Вариант 1: рядом в labels/
        lab_candidate = img.parent.parent / "labels" / (img.stem + ".txt")
        if lab_candidate.exists():
            lab = lab_candidate
        else:
            # Вариант 2: тот же путь но .txt
            lab_candidate = img.with_suffix(".txt")
            if lab_candidate.exists():
                lab = lab_candidate
            else:
                # Вариант 3: рекурсивно
                lab_candidates = list(src.rglob(f"{img.stem}.txt"))
                if lab_candidates:
                    lab = lab_candidates[0]
        
        if not lab:
            skipped_no_lab += 1
            continue
        
        # Читаем разметку
        lines = []
        for ln in lab.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            try:
                ni = idx_map.get(int(p[0]))
                if ni is None:
                    continue
                lines.append(f"{ni} {p[1]} {p[2]} {p[3]} {p[4]}")
            except (ValueError, IndexError):
                continue
        
        if not lines:
            skipped_no_lines += 1
            continue
        
        # Определяем split
        # Если в пути есть val/valid/test — val, иначе train
        split_i = "val" if any(s in str(img).lower() for s in ["val", "valid", "test"]) else "train"
        # 10% train уходит в val
        if split_i == "train" and random.random() < 0.1:
            split_i = "val"
        
        # Копируем
        try:
            shutil.copy(img, ROOT / "images" / split_i / f"{tag}_{img.name}")
            (ROOT / "labels" / split_i / f"{tag}_{img.stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8")
            moved += 1
        except Exception as e:
            print(f"   ❌ Ошибка копирования {img.name}: {e}")
    
    print(f"   ✅ Скопировано: {moved}")
    print(f"   ❌ Пропущено: нет разметки={skipped_no_lab}, "
          f"пустые метки={skipped_no_lines}")
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

    if MANUAL.exists():
        for d in MANUAL.iterdir():
            if d.is_dir():
                total += ingest(d, d.name)

    # Создаём ROOT если ещё не создан
    ROOT.mkdir(parents=True, exist_ok=True)
    
    present = set()
    for lab in (ROOT / "labels").rglob("*.txt"):
        for ln in lab.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    present.add(int(ln.split()[0]))
                except ValueError:
                    pass
    
    (ROOT / "data.yaml").write_text(yaml.safe_dump({
        "path": str(ROOT), "train": "images/train", "val": "images/val",
        "nc": len(MILITARY_CLASSES), "names": MILITARY_CLASSES,
    }, allow_unicode=True), encoding="utf-8")

    print(f"\n🎉 Итого: {total} РЕАЛЬНЫХ кадров")
    print(f"Классов с данными: {len(present)} из {len(MILITARY_CLASSES)}")
    print("Классы:", [MILITARY_CLASSES[i] for i in sorted(present)])

if __name__ == "__main__":
    main()