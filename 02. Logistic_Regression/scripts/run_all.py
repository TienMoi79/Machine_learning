"""Chạy hai notebook của Lab 02."""

from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    notebooks = [
        root / "code" / "Lab2_LeQuyetTien.ipynb",
        root / "code" / "Baitap.ipynb",
    ]
    for notebook in notebooks:
        print(f"Dang chay: {notebook.name}", flush=True)
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=300",
            str(notebook),
        ]
        result = subprocess.run(command, cwd=root)
        if result.returncode:
            return result.returncode
    print("Hoan tat 2 notebook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
