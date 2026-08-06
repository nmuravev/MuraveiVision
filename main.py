"""Точка входа MuraveiVision — лаунчер выбора версии (Мини / PRO).

Важно: PySide6 НЕ импортируется на верхнем уровне — иначе он загружается
в процесс Мини-версии и вызывает фатальный краш GIL при webbrowser.open().
Проверка PRO идёт через importlib.util.find_spec (без импорта).
"""
import sys
import os
import importlib.util
import traceback
import datetime
from pathlib import Path

# Корень проекта — в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Portable-совместимая загрузка .env
from core.paths import app_root, env_path

load_dotenv(env_path())
# Fallback: пробуем обычную загрузку (текущая папка)
load_dotenv(override=False)


def _check_qt_available():
    """Проверяет доступность PySide6 и PySide6-WebEngine БЕЗ импорта.

    Использует importlib.util.find_spec — не загружает PySide6 в процесс,
    чтобы Мини-версия не крашилась (GIL) при открытии браузера.

    Возвращает список отсутствующих пакетов с точными именами для pip install.
    """
    missing = []
    if importlib.util.find_spec("PySide6") is None:
        missing.append("PySide6")
    if importlib.util.find_spec("PySide6.QtWebEngineWidgets") is None:
        missing.append("PySide6-WebEngine")
    return missing


def _save_diagnostics(exc_info):
    """Сохраняет traceback + sysinfo в error_log/muravevision_diagnostics_<ts>.tgz."""
    import tarfile
    import platform
    import io

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    err_dir = Path("error_log")
    err_dir.mkdir(exist_ok=True)
    out_path = err_dir / f"muravevision_diagnostics_{ts}.tgz"

    # Собираем диагностическую информацию
    info = (
        f"MuraveiVision Diagnostics\n"
        f"=========================\n"
        f"Timestamp: {datetime.datetime.now().isoformat()}\n"
        f"Python: {sys.version}\n"
        f"Platform: {platform.platform()}\n"
        f"Executable: {sys.executable}\n\n"
        f"Traceback:\n{''.join(traceback.format_exception(*exc_info))}\n"
    )

    # Пишем info.txt во временный файл, затем в tar.gz
    info_path = err_dir / f"_info_{ts}.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(info)

    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(info_path, arcname="diagnostics.txt")

    # Удаляем временный файл
    try:
        info_path.unlink()
    except Exception:
        pass

    return str(out_path)


def run_selftest():
    """Режим самопроверки (без GUI): импорт, hardware, модели, загрузка YOLO.

    Использование: python main.py --selftest
    Возвращает exit 0 при успехе, exit 1 при ошибке (с traceback).
    """
    print("🧪 MuraveiVision Self-Test")
    print("=" * 50)

    # 1. Импорт ключевых модулей
    print("1. Импорт модулей...")
    try:
        from core import hardware, backend
        from core.paths import models_dir
        print("   ✅ core.hardware, core.backend, core.paths")
    except Exception:
        print("   ❌ Ошибка импорта core.*")
        traceback.print_exc()
        sys.exit(1)

    # 2. Определение железа
    print("2. Определение железа...")
    try:
        hw = hardware.detect_all()
        gpu = hw.get("gpu")
        if gpu:
            print(f"   ✅ GPU: {gpu.get('name', '?')} ({gpu.get('vram_mb', 0)} MB)")
        else:
            cpu = hw.get("cpu", {})
            print(f"   ✅ CPU-only: {cpu.get('brand', '?')}")
    except Exception:
        print("   ❌ Ошибка hardware.detect_all()")
        traceback.print_exc()
        sys.exit(1)

    # 3. Список моделей
    print("3. Доступные модели...")
    try:
        mdir = models_dir()
        onnx_files = sorted([f.name for f in mdir.glob("*.onnx")])
        if not onnx_files:
            print(f"   ⚠️ Нет ONNX-моделей в {mdir}")
        else:
            for f in onnx_files:
                print(f"   📦 {f}")
    except Exception:
        print("   ❌ Ошибка чтения папки models/")
        traceback.print_exc()
        sys.exit(1)

    # 4. Загрузка выбранной модели
    print("4. Загрузка модели YOLO...")
    try:
        from core.detector import MilitaryDetector
        det = MilitaryDetector()
        print(f"   ✅ Модель: {det.model_name}")
        print(f"   ✅ Провайдеры: {', '.join(det.providers)}")
    except Exception:
        print("   ❌ Ошибка загрузки модели")
        traceback.print_exc()
        sys.exit(1)

    print("=" * 50)
    print("✅ SELFTEST OK")
    sys.exit(0)


def launch_mini():
    """Запускает Мини-версию (CustomTkinter)."""
    from ui.app import launch
    launch()


def launch_pro():
    """Запускает PRO-версию (PySide6).

    Импорт ui.ide_app — ТОЛЬКО здесь (не на верхнем уровне),
    чтобы PySide6 не загружался в процесс Мини-версии.
    """
    from ui.ide_app import launch as launch_ide
    launch_ide()


def main():
    """Лаунчер: окно выбора [🟢 Мини][🟣 PRO]."""
    # Режим самопроверки
    if "--selftest" in sys.argv:
        run_selftest()
        return

    # Проверяем Qt
    qt_missing = _check_qt_available()

    # Portable-сборка: PRO недоступна (PySide6 исключён из spec)
    is_frozen = getattr(sys, "frozen", False)

    # Если нет PySide6 — показываем простой диалог выбора через tkinter
    # (чтобы не зависеть от Qt для лаунчера)
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        # Нет tkinter — запускаем Мини по умолчанию
        launch_mini()
        return

    root = tk.Tk()
    root.title("🎯 MuraveiVision — Выбор версии")
    root.geometry("420x340")
    root.configure(bg="#000000")

    # Заголовок
    tk.Label(root, text="🎯 MuraveiVision", font=("Segoe UI", 18, "bold"),
             bg="#000000", fg="#00E5FF").pack(pady=15)
    tk.Label(root, text="Выберите версию:", font=("Segoe UI", 12),
             bg="#000000", fg="#8A8A8A").pack(pady=5)

    # Кнопка Мини
    mini_btn = tk.Button(
        root, text="🟢 Мини (CustomTkinter)", font=("Segoe UI", 12, "bold"),
        bg="#00FF88", fg="#000000", activebackground="#00CC6E",
        relief="flat", padx=20, pady=10, cursor="hand2",
        command=lambda: _on_mini(),
    )
    mini_btn.pack(pady=10)

    # Кнопка PRO
    # В portable-сборке (frozen) PRO недоступна
    if is_frozen:
        pro_state = "disabled"
        pro_reason = "PRO недоступна в портативной версии"
    elif qt_missing:
        pro_state = "disabled"
        pro_reason = "PRO недоступна: не установлены " + ", ".join(qt_missing)
    else:
        pro_state = "normal"
        pro_reason = None

    pro_btn = tk.Button(
        root, text="🟣 PRO (PySide6 + WebEngine)", font=("Segoe UI", 12, "bold"),
        bg="#9D4EDD", fg="#FFFFFF", activebackground="#7B2CBF",
        relief="flat", padx=20, pady=10, cursor="hand2",
        state=pro_state,
        command=lambda: _on_pro(),
    )
    pro_btn.pack(pady=10)

    # Причина блокировки PRO
    if pro_reason:
        tk.Label(root, text=pro_reason, font=("Segoe UI", 10),
                 bg="#000000", fg="#FF3B30", wraplength=380).pack(pady=2)
        if qt_missing and not is_frozen:
            pip_hint = "pip install " + " ".join(qt_missing)
            tk.Label(root, text=f"Установите: {pip_hint}", font=("Consolas", 10),
                     bg="#000000", fg="#00E5FF", wraplength=380).pack(pady=2)

    def _on_mini():
        root.destroy()
        try:
            launch_mini()
        except Exception:
            exc_info = sys.exc_info()
            diag = _save_diagnostics(exc_info)
            messagebox.showerror(
                "Ошибка Мини",
                f"Произошла ошибка:\n\n{''.join(traceback.format_exception(*exc_info))}\n\n"
                f"Диагностика сохранена: {diag}\nОтправьте нам архив.",
            )

    def _on_pro():
        root.destroy()
        try:
            launch_pro()
        except Exception:
            exc_info = sys.exc_info()
            diag = _save_diagnostics(exc_info)
            messagebox.showerror(
                "Ошибка PRO",
                f"Произошла ошибка:\n\n{''.join(traceback.format_exception(*exc_info))}\n\n"
                f"Диагностика сохранена: {diag}\nОтправьте нам архив.",
            )

    root.mainloop()


if __name__ == "__main__":
    main()
