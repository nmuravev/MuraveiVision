"""Точка входа MuraveiVision."""
import sys
from pathlib import Path

# Корень проекта — в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from ui.app import launch

if __name__ == "__main__":
    launch()