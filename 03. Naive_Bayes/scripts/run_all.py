"""Chạy toàn bộ chương trình Python của Lab 03."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    files = sorted((ROOT / "code").glob("d*.py"))
    files += sorted((ROOT / "baitap03").glob("bai*.py"))
    for path in files:
        print(f"\n===== {path.relative_to(ROOT)} =====", flush=True)
        result = subprocess.run([sys.executable, str(path)], cwd=ROOT)
        if result.returncode:
            return result.returncode
    print(f"\nHoàn tất {len(files)} chương trình.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
