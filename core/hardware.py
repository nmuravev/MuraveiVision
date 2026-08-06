"""Детекция железа: любая NVIDIA GPU + любой CPU (Intel/AMD)."""
import os
import platform
import subprocess


def _hidden_run(cmd, timeout=5):
    """Запуск внешней утилиты без чёрного консольного окна (Windows).

    CREATE_NO_WINDOW (0x08000000) + STARTUPINFO(wShowWindow=SW_HIDE=0)
    предотвращают мелькание консоли при вызове nvidia-smi и др.
    Возвращает CompletedProcess-подобный объект с .stdout.
    """
    kwargs = dict(capture_output=True, text=True, timeout=timeout)
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = si
    return subprocess.run(cmd, **kwargs)


def get_cpu_info() -> dict:
    """Определяет производителя CPU: Intel, AMD или другой."""
    brand = platform.processor() or "Неизвестный CPU"
    if "Intel" in brand:
        vendor = "Intel"
    elif "AMD" in brand:
        vendor = "AMD"
    else:
        vendor = "Другой"
    return {"vendor": vendor, "brand": brand, "cores": os.cpu_count()}


def get_nvidia_gpu() -> dict | None:
    """Спрашивает nvidia-smi: есть ли NVIDIA и сколько VRAM.
    Работает с ЛЮБОЙ картой NVIDIA, если установлены драйверы."""
    try:
        out = _hidden_run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).stdout.strip()
        if not out:
            return None
        name, vram = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "vram_mb": int(vram)}
    except Exception:
        return None  # NVIDIA нет вообще


def detect_all() -> dict:
    return {"cpu": get_cpu_info(), "gpu": get_nvidia_gpu()}