"""Файнтюн YOLO-World v2 на РЕАЛЬНЫХ данных + вентиль + экспорт.
RTX 5060: s=batch 8 (OOM → 4), m=4, l=2. Windows-фикс: код в __main__."""
import os, shutil, sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()
from ultralytics import YOLO
from core.classes import MILITARY_CLASSES


def find_best():
    """Ищем свежий best.pt ГДЕ УГОДНО в runs/ — ultralytics
    может добавить свои папки (runs/detect/runs/...)."""
    cands = sorted(Path("runs").rglob("best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main():
    SIZE = os.getenv("FINETUNE_SIZE", "s")
    BATCH = {"s": 8, "m": 4, "l": 2}[SIZE]
    SRC = Path("source_weights") / f"yolov8{SIZE}-worldv2.pt"
    DATA = Path("datasets/military_merged/data.yaml")

    print("=" * 60)
    print("🎓 ФАЙНТЮН YOLO-WORLD V2 НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)

    if not DATA.exists():
        print("❌ Датасет не найден. Сначала: python training/prepare_data.py")
        sys.exit(1)

    print("\n📊 Фаза 1: обучение...")
    model = YOLO(str(SRC))
    try:
        model.train(data=str(DATA), epochs=30, imgsz=640, batch=BATCH,
                    device=0, workers=4, patience=8,
                    project="runs/mv_finetune", name=f"worldv2_{SIZE}_real",
                    exist_ok=True, verbose=False)
    except Exception as e:
        print(f"⚠️ Ошибка с workers=4, повторяю с workers=0: {e}")
        model = YOLO(str(SRC))
        model.train(data=str(DATA), epochs=30, imgsz=640, batch=BATCH,
                    device=0, workers=0, patience=8,
                    project="runs/mv_finetune", name=f"worldv2_{SIZE}_real",
                    exist_ok=True, verbose=False)

    best = find_best()
    if best is None:
        print("❌ best.pt не создан. См. логи выше.")
        sys.exit(1)
    print(f"📦 Модель найдена: {best}")

    print("\n📏 Фаза 2: вентиль качества...")
    names = yaml.safe_load(DATA.read_text(encoding="utf-8"))["names"]
    base = YOLO(str(SRC))
    base.set_classes(names)
    m_base = base.val(data=str(DATA), device=0, verbose=False)
    ft = YOLO(str(best))
    m_ft = ft.val(data=str(DATA), device=0, verbose=False)

    print(f"   zero-shot mAP50: {m_base.box.map50:.3f}")
    print(f"   finetune  mAP50: {m_ft.box.map50:.3f}")
    if m_base.box.map50 > 0:
        imp = (m_ft.box.map50 - m_base.box.map50) / m_base.box.map50 * 100
        print(f"   улучшение: {imp:+.1f}%")

    if m_ft.box.map50 <= m_base.box.map50:
        print("\n❌ Финтюн не лучше базовой — боевая модель НЕ тронута.")
        sys.exit(0)

    print("\n📦 Фаза 3: экспорт с полным словарём 86 классов...")
    ft.set_classes(MILITARY_CLASSES)
    ft.export(format="onnx", simplify=True, opset=17, dynamic=False)
    src_onnx = best.with_suffix(".onnx")
    out = Path(f"models/yolov8{SIZE}-worldv2-finetuned.onnx")
    shutil.move(str(src_onnx), str(out))
    print(f"\n✅ Готово: {out}")


if __name__ == "__main__":
    main()