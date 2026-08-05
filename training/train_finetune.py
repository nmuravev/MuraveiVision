"""Файнтюн YOLO-World v2 на РЕАЛЬНЫХ данных + вентиль качества + экспорт.
RTX 5060 8GB: s=batch 8, m=batch 4, l=batch 2."""
import os, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(file).parent.parent))
from dotenv import load_dotenv
load_dotenv()
from ultralytics import YOLO
from core.classes import MILITARY_CLASSES

SIZE = os.getenv("FINETUNE_SIZE", "s")  # s / m / l
BATCH = {"s": 8, "m": 4, "l": 2}[SIZE]
SRC = Path("source_weights") / f"yolov8{SIZE}-worldv2.pt"
DATA = Path("datasets/military_merged/data.yaml")

print("=" * 60)
print("🎓 ФАЙНТЮН YOLO-WORLD V2 НА РЕАЛЬНЫХ ДАННЫХ")
print("=" * 60)
print(f"Размер модели: {SIZE}")
print(f"Batch size: {BATCH}")
print(f"Датасет: {DATA}")
print()

print("📊 Фаза 1: обучение на реальных данных...")
model = YOLO(str(SRC))
model.train(data=str(DATA), epochs=30, imgsz=640, batch=BATCH,
            device=0, workers=4, patience=8,
            project="runs/mv_finetune", name=f"worldv2_{SIZE}_real",
            exist_ok=True, verbose=False)
best = Path("runs/mv_finetune") / f"worldv2_{SIZE}_real" / "weights" / "best.pt"

print("\n📏 Фаза 2: вентиль качества (zero-shot vs finetune)...")
names = import("yaml").safe_load(DATA.read_text(encoding="utf-8"))["names"]
base = YOLO(str(SRC))
base.set_classes(names)
m_base = base.val(data=str(DATA), device=0, verbose=False)
ft = YOLO(str(best))
m_ft = ft.val(data=str(DATA), device=0, verbose=False)

print(f"   zero-shot mAP50: {m_base.box.map50:.3f}")
print(f"   finetune  mAP50: {m_ft.box.map50:.3f}")
improvement = ((m_ft.box.map50 - m_base.box.map50) / m_base.box.map50) * 100 if m_base.box.map50 > 0 else 0
print(f"   улучшение: {improvement:+.1f}%")

if m_ft.box.map50 <= m_base.box.map50:
    print("\n❌ Финтюн не лучше базовой — боевая модель НЕ тронута.")
    sys.exit(0)

print("\n📦 Фаза 3: экспорт с полным словарём 86 классов...")
ft.set_classes(MILITARY_CLASSES)
ft.export(format="onnx", simplify=True, opset=17, dynamic=False)
out = Path(f"models/yolov8{SIZE}-worldv2-finetuned.onnx")
shutil.move(f"yolov8{SIZE}-worldv2-finetuned.onnx", out)
print(f"\n✅ Готово: {out}")
print(f"🚀 Теперь MuraveiVision автоматически подхватит финтюнутую модель")