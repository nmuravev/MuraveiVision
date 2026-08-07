"""Экспорт YOLO-World моделей в ONNX.
ВАЖНО: YOLO-World v2 при экспорте сохраняет 210 классов (ограничение архитектуры).
При инференсе используем только первые 206 из нашего словаря."""
import os
import sys
from ultralytics import YOLO
from core.classes import MILITARY_CLASSES

# Отключаем автообновление зависимостей ultralytics
import ultralytics.utils.checks as _ul_checks
_ul_checks.check_requirements = lambda *args, **kwargs: None

def export_model(model_name: str):
    """Экспортирует одну модель (s/m/l)."""
    print(f"\n=== Экспорт {model_name} ===")
    
    # Загружаем модель
    model = YOLO(f"source_weights/{model_name}.pt")
    
    # Создаём папку models/ если её нет
    os.makedirs("models", exist_ok=True)
    
    # Экспорт в ONNX (YOLO-World v2 экспортирует 210 классов — это нормально)
    output_path = model.export(format="onnx", opset=17, simplify=True)
    
    # Перемещаем из source_weights/ в models/
    final_path = f"models/{model_name}.onnx"
    if os.path.exists(output_path):
        os.replace(output_path, final_path)
    
    print(f"✅ Готово: {final_path}")
    return final_path

def main():
    print(f"📋 Классов в словаре: {len(MILITARY_CLASSES)}")
    print("⚠️  YOLO-World v2 экспортирует 210 классов (ограничение архитектуры)")
    print("   При инференсе используем только первые 206 из нашего словаря\n")
    
    models = ["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2"]
    
    for model_name in models:
        try:
            export_model(model_name)
        except Exception as e:
            print(f"❌ Ошибка при экспорте {model_name}: {e}")
            sys.exit(1)
    
    print("\n🎉 Экспорт завершен!")
    print("💡 Базовые модели готовы. Для лучшего качества используйте finetuned модель.")

if __name__ == "__main__":
    main()