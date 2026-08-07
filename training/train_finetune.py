"""Переобучение YOLOv8s-worldv2 на военном датасете.
Запуск из корня проекта: python training/train_finetune.py"""
import sys
from pathlib import Path

# Корень проекта в пути поиска (иначе "No module named 'core'")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from core.classes import MILITARY_CLASSES

# Замок от автообновления ultralytics
import ultralytics.utils.checks as _ul_checks
_ul_checks.check_requirements = lambda *args, **kwargs: None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "datasets" / "military_merged" / "data.yaml"
SOURCE_MODEL = PROJECT_ROOT / "source_weights" / "yolov8s-worldv2.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / "finetune"

def main():
    if not DATASET_YAML.exists():
        print(f"❌ Нет датасета: {DATASET_YAML}")
        sys.exit(1)

    print(f"📋 Словарь программы: {len(MILITARY_CLASSES)} классов")
    print(f"📦 Датасет: {DATASET_YAML} (86 размеченных классов)")
    print(f"🎯 Источник: {SOURCE_MODEL.name}")
    print("💡 Эмбеддинги классов тренер построит сам из data.yaml\n")

    model = YOLO(str(SOURCE_MODEL))

    print("🚀 Старт обучения...")
    model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        project=str(RUNS_DIR),
        name="v2_dict206",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        amp=True,
        cache=False,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        degrees=5.0, translate=0.1, scale=0.3,
        fliplr=0.5, mosaic=0.5, mixup=0.1,
    )

    best = RUNS_DIR / "v2_dict206" / "weights" / "best.pt"
    print(f"\n✅ Обучение завершено!")
    print(f"🏆 Лучший чекпоинт: {best}")

if __name__ == "__main__":
    main()