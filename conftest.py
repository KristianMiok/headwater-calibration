"""Pytest configuration: make src/ importable without requiring `pip install -e .`."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
