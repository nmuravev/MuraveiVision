"""Выбор движка и размера модели под обнаруженное железо."""
import onnxruntime as ort


def choose_providers(gpu: dict | None) -> list:
    """Любая NVIDIA -> CUDA; иначе -> CPU (Intel/AMD/что угодно)."""
    available = ort.get_available_providers()
    if gpu is not None and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def choose_model(gpu: dict | None) -> str:
    """Подбираем размер модели под объем VRAM (или CPU)."""
    if gpu is None:
        return "yolov8n-world.onnx"          # только CPU
    if gpu["vram_mb"] >= 8000:
        return "yolov8m-world.onnx"          # мощная NVIDIA
    if gpu["vram_mb"] >= 4000:
        return "yolov8s-world.onnx"          # средняя NVIDIA
    return "yolov8n-world.onnx"              # слабая NVIDIA


def describe_backend(gpu: dict | None, cpu: dict) -> str:
    """Красивая строка для GUI."""
    if gpu is not None and "CUDAExecutionProvider" in ort.get_available_providers():
        return f"GPU: {gpu['name']} ({gpu['vram_mb'] // 1024} GB)"
    return f"CPU: {cpu['brand']}"