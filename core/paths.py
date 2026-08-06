"""Портативно-совместимые пути к ресурсам приложения.

app_root() возвращает корневую папку приложения:
  - в замороженной (PyInstaller) сборке — папка рядом с exe;
  - в обычном режиме — корень проекта (на два уровня выше этого файла).
"""
import sys
from pathlib import Path


def app_root() -> Path:
    """Корневая папка приложения (portable-совместимая)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent   # папка exe
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    """Папка с ONNX-моделями."""
    return app_root() / "models"


def output_dir() -> Path:
    """Папка для отчётов и стоп-кадров."""
    return app_root() / "output"


def env_path() -> Path:
    """Путь к файлу .env."""
    return app_root() / ".env"