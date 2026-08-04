"""Экспорт YOLO-World V2 в ONNX с вшитым словарём из 88 классов."""
import sys
sys.path.insert(0, ".")
from core.classes import MILITARY_CLASSES
from ultralytics import YOLO

print(f"📋 Классов в словаре: {len(MILITARY_CLASSES)}")

for size in ["s", "m", "l"]:
    print(f"\n=== Экспорт yolov8{size}-worldv2 ===")
    model = YOLO(f"yolov8{size}-worldv2.pt")   # именно v2!
    model.set_classes(MILITARY_CLASSES)
    model.export(format="onnx", simplify=True, opset=17, dynamic=False)
    print(f"✅ Готово: yolov8{size}-worldv2.onnx")

print("\n🎉 Экспорт завершен!")