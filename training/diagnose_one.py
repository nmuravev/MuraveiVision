"""Диагностика: почему 0 кадров на первом датасете."""
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.classes import MILITARY_CLASSES

SRC = Path(__file__).parent.parent / "datasets" / "raw" / "uce03211-gmail-com"

# Ищем train (может быть где угодно)
trains = list(SRC.rglob("train")) + list(SRC.rglob("train/"))
print(f"Папки train: {trains}")
if not trains:
    print("⚠️ Папки train не найдено. Структура:")
    for p in SRC.rglob("*")[:20]:
        print(f"   {p.relative_to(SRC)}")
    sys.exit(1)

train = trains[0]
imgs = train / "images"
labs = train / "labels"
print(f"images: {imgs} (exists={imgs.exists()})")
print(f"labels: {labs} (exists={labs.exists()})")

if not imgs.exists():
    # может, картинки лежат прямо в train?
    imgs = train
    print(f"Картинки прямо в train/: {len(list(train.glob('*.jpg')))}")

# Читаем классы датасета
names = None
for cand in SRC.rglob("*.yaml"):
    try:
        data = yaml.safe_load(cand.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("names"):
            names = data["names"]
            print(f"Классы: {names} (из {cand})")
            break
    except Exception:
        pass

if not names:
    for cand in SRC.rglob("classes.txt"):
        names = [l.strip() for l in cand.read_text().splitlines() if l.strip()]
        print(f"Классы из classes.txt: {names}")
        break

if not names:
    print("⚠️ Классы датасета не найдены")

all_imgs = list(imgs.glob("*.jpg")) + list(imgs.glob("*.png")) + list(imgs.glob("*.jpeg"))
print(f"\nВсего картинок: {len(all_imgs)}")
print(f"Примеры: {[f.name for f in all_imgs[:5]]}")

# Разметка — где она?
all_labs = list(SRC.rglob("*.txt"))
print(f"Всего .txt файлов: {len(all_labs)}")
print(f"Примеры разметки: {[f.name for f in all_labs[:5]]}")

# Проверяем первые 5 картинок по шагам
print("\n=== ДИАГНОСТИКА ПЕРВЫХ 5 КАРТИНОК ===")
for img in all_imgs[:5]:
    print(f"\n📷 {img.name}")
    
    # Шаг 1: is_real_data
    try:
        from PIL import Image
        name_lower = img.name.lower()
        blacklist = ["warno", "arma", "squad", "battlefield", "game", 
                     "synthetic", "virtual", "simulation", "render", 
                     "3d_model", "unity", "unreal"]
        blacklisted = any(p in name_lower for p in blacklist)
        print(f"   Чёрный список: {'❌ ЗАБЛОКИРОВАНА' if blacklisted else '✅ OK'}")
        if blacklisted:
            matched = [p for p in blacklist if p in name_lower]
            print(f"      совпало с: {matched}")
            continue
    except Exception as e:
        print(f"   Ошибка проверки: {e}")
    
    # Шаг 2: поиск разметки
    # Вариант A: рядом с картинкой в labels/
    lab_a = (labs / img.stem).with_suffix(".txt")
    # Вариант B: тот же путь что у картинки, но .txt
    lab_b = img.with_suffix(".txt")
    # Вариант C: где-то в датасете с таким же именем
    lab_c = list(SRC.rglob(f"{img.stem}.txt"))
    
    lab = None
    if lab_a.exists():
        lab = lab_a
        print(f"   Разметка найдена (вариант A): {lab_a}")
    elif lab_b.exists():
        lab = lab_b
        print(f"   Разметка найдена (вариант B): {lab_b}")
    elif lab_c:
        lab = lab_c[0]
        print(f"   Разметка найдена (вариант C): {lab_c[0]}")
    else:
        print(f"   ❌ РАЗМЕТКА НЕ НАЙДЕНА!")
        print(f"      искал: {lab_a}")
        print(f"      искал: {lab_b}")
        print(f"      искал: {img.stem}.txt рекурсивно")
        continue
    
    # Шаг 3: читаем разметку
    lines = lab.read_text(encoding="utf-8").splitlines()
    print(f"   Строк в разметке: {len(lines)}")
    print(f"   Примеры: {lines[:3]}")
    
    # Шаг 4: маппим классы
    mapped = 0
    for ln in lines[:3]:
        p = ln.split()
        if len(p) < 5:
            continue
        class_id = int(p[0])
        if names and class_id < len(names):
            class_name = names[class_id]
            print(f"      класс {class_id}: {class_name}")
        mapped += 1
    print(f"   ✅ Маппится {mapped} строк")