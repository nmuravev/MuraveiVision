"""Экспорт обученной модели в ONNX + бэкап старой.
Запуск из корня проекта: python training/export_finetune.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

import ultralytics.utils.checks as _ul_checks
_ul_checks.check_requirements = lambda *args, **kwargs: None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PT = PROJECT_ROOT / "runs" / "finetune" / "v2_dict206" / "weights" / "best.pt"
OUTPUT_ONNX = PROJECT_ROOT / "models" / "yolov8s-worldv2-finetuned.onnx"
BACKUP_ONNX = PROJECT_ROOT / "models" / "yolov8s-worldv2-finetuned-backup.onnx"

def main():
    if not BEST_PT.exists():
        print(f"❌ Нет обученной модели: {BEST_PT}")
        print("   Сначала: python training/train_finetune.py")
        sys.exit(1)

    # Бэкап старой модели (один раз)
    if OUTPUT_ONNX.exists() and not BACKUP_ONNX.exists():
        print(f"💾 Бэкап старой модели: {BACKUP_ONNX.name}")
        OUTPUT_ONNX.replace(BACKUP_ONNX)

    model = YOLO(str(BEST_PT))

    print("🚀 Экспорт в ONNX...")
    export_path = model.export(format="onnx", opset=17, simplify=True, imgsz=640)

    if os.path.exists(export_path):
        os.replace(export_path, str(OUTPUT_ONNX))
    else:
        print(f"❌ Файл не найден: {export_path}")
        sys.exit(1)

    # Проверка: сколько классов в итоге
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUTPUT_ONNX))
    n = sess.get_outputs()[0].shape[1]
    print(f"✅ Экспортировано: {OUTPUT_ONNX.name}")
    print(f"📊 Классов в ONNX: {n} (ожидается 86 — размеченные; индексы 0-85 = первые 86 словаря)")

if __name__ == "__main__":
    main()