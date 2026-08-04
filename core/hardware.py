"""Детекция железа: любая NVIDIA GPU + любой CPU (Intel/AMD)."""
import os
import platform
import subprocess


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
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not out:
            return None
        name, vram = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "vram_mb": int(vram)}
    except Exception:
        return None  # NVIDIA нет вообще


def detect_all() -> dict:
    return {"cpu": get_cpu_info(), "gpu": get_nvidia_gpu()}