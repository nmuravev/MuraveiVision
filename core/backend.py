"""Выбор движка и размера модели под обнаруженное железо."""
import onnxruntime as ort


def choose_providers(gpu: dict | None) -> list:
    """Любая NVIDIA -> CUDA; иначе -> CPU (Intel/AMD/что угодно)."""
    available = ort.get_available_providers()
    if gpu is not None and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


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