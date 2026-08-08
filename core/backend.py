"""Выбор движка и размера модели под обнаруженное железо."""
import glob
import os
import sys

import onnxruntime as ort


def _ensure_cuda_dlls():
    """Подключает CUDA DLL перед созданием ort.InferenceSession.

    Сканирует:
      - torch\\lib и nvidia\\*\\bin (в site-packages);
      - C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin;
      - CUDA_PATH.
    Приоритет — версии torch (т.к. onnxruntime собран под CUDA из torch).
    Дополнительно вызывает os.add_dll_directory (Python 3.8+).
    Если torch не установлен — тихо пропускает.
    """
    found = []

    def _add_dir(d):
        try:
            d = os.path.abspath(d)
            if os.path.isdir(d):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(d)
                except (AttributeError, OSError):
                    pass
                found.append(d)
        except Exception:
            pass

    # 1) torch и nvidia-* из site-packages (приоритет версии torch)
    try:
        import torch
        torch_dir = os.path.dirname(torch.__file__)
        # torch\\lib
        _add_dir(os.path.join(torch_dir, "lib"))
        # nvidia\\*\\bin (рядом с site-packages)
        sp_dir = os.path.dirname(torch_dir)
        for ndir in sorted(glob.glob(os.path.join(sp_dir, "nvidia", "*", "bin"))):
            _add_dir(ndir)
    except ImportError:
        pass
    except Exception:
        pass

    # 2) CUDA Toolkit: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin
    for cdir in sorted(glob.glob(
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                     "NVIDIA GPU Computing Toolkit", "CUDA", "v*", "bin")
    )):
        _add_dir(cdir)

    # 3) CUDA_PATH
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        _add_dir(os.path.join(cuda_path, "bin"))

    if found:
        print(f"[CUDA-bridge] подключено: {found}")
    return found


def choose_providers(gpu: dict | None) -> list:
    """Любая NVIDIA -> CUDA; иначе -> CPU (Intel/AMD/что угодно).

    Перед выбором вызывает _ensure_cuda_dlls() чтобы ONNX Runtime нашёл CUDA.
    Возвращает пересечение желаемых ["CUDAExecutionProvider",
    "CPUExecutionProvider"] с ort.get_available_providers().
    """
    _ensure_cuda_dlls()
    available = ort.get_available_providers()
    wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in wanted if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in providers:
        print("[backend] CUDA недоступен — работаем на CPU")
    return providers


def choose_model(gpu: dict | None) -> str:
    """Подбираем размер модели под объем VRAM (или CPU).

    Возвращает имя ONNX-файла, который реально лежит в models/.
    Доступные размеры: s, m, l (nano нет — понижаем до small).
    """
    if gpu is None:
        return "yolov8s-worldv2.onnx"          # CPU — лёгкая small
    if gpu["vram_mb"] >= 8000:
        return "yolov8l-worldv2.onnx"          # мощная NVIDIA — large
    if gpu["vram_mb"] >= 4000:
        return "yolov8m-worldv2.onnx"          # средняя NVIDIA — medium
    return "yolov8s-worldv2.onnx"              # слабая NVIDIA — small


def describe_backend(gpu: dict | None, cpu: dict) -> str:
    """Красивая строка для GUI."""
    if gpu is not None and "CUDAExecutionProvider" in ort.get_available_providers():
        return f"GPU: {gpu['name']} ({gpu['vram_mb'] // 1024} GB)"
    return f"CPU: {cpu['brand']}"