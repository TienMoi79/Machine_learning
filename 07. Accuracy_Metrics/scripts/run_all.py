"""Chạy toàn bộ chương trình Python của Lab 07."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["PYTHONIOENCODING"] = "utf-8"
files = sorted((ROOT / "code").glob("h*.py")) + sorted((ROOT / "baitap07").glob("bai*.py"))
for path in files:
    print(f"\n===== {path.relative_to(ROOT)} =====", flush=True)
    result = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)
print(f"\nHoan tat {len(files)} chuong trinh.")
