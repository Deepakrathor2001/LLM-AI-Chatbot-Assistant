"""Run this to launch Vision AI from any terminal location."""
import os, sys, subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)
subprocess.run([sys.executable, "-m", "streamlit", "run",
                str(PROJECT_DIR / "app.py")], cwd=str(PROJECT_DIR))
