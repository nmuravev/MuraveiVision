"""Вентиль качества + экспорт УЖЕ ОБУЧЕННОЙ модели (без переобучения).
Windows-фикс: весь код внутри __name__ == '__main__'."""
import os, shutil, sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()
from ultralytics import YOLO
from core.classes import MILITARY_CLASSES


def find_best():
    cands = sorted(Path("runs").rglob("best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main():
    SIZE = os.getenv("FINETUNE_SIZE", "s")
    DATA = Path("datasets/military_merged/data.yaml")

    best = find_best()
    if best is None:
        print("❌ best.pt не найден в runs/. Сначала обучение.")
        sys.exit(1)
    print(f"📦 Модель найдена: {best}")

    names = yaml.safe_load(DATA.read_text(encoding="utf-8"))["names"]

    print("📏 Вентиль качества (zero-shot vs finetune)...")
    base = YOLO(str(Path("source_weights") / f"yolov8{SIZE}-worldv2.pt"))
    base.set_classes(names)
    m_base = base.val(data=str(DATA), device=0, verbose=False, workers=0)
    ft = YOLO(str(best))
    m_ft = ft.val(data=str(DATA), device=0, verbose=False, workers=0)

    print(f"   zero-shot mAP50: {m_base.box.map50:.3f}")
    print(f"   finetune  mAP50: {m_ft.box.map50:.3f}")
    if m_base.box.map50 > 0:
        imp = (m_ft.box.map50 - m_base.box.map50) / m_base.box.map50 * 100
        print(f"   улучшение: {imp:+.1f}%")

    if m_ft.box.map50 <= m_base.box.map50:
        print("❌ Финтюн не лучше базовой — боевая модель НЕ тронута.")
        sys.exit(0)

    print("📦 Экспорт с полным словарём 86 классов...")
    ft.set_classes(MILITARY_CLASSES)
    ft.export(format="onnx", simplify=True, opset=17, dynamic=False)
    src_onnx = best.with_suffix(".onnx")
    out = Path(f"models/yolov8{SIZE}-worldv2-finetuned.onnx")
    shutil.move(str(src_onnx), str(out))
    print(f"\n✅ Готово: {out}")


if __name__ == "__main__":
    main()