"""Chạy toàn bộ notebook của Bài 01 và lưu kết quả."""

from pathlib import Path
import subprocess
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    notebooks = sorted((project / "code").glob("*.ipynb"), key=lambda p: (not p.name.startswith("Lab"), p.name))
    if not notebooks:
        print("Không tìm thấy notebook trong code/.", file=sys.stderr)
        return 1

    for notebook in notebooks:
        print(f"Đang chạy: {notebook.name}", flush=True)
        command = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=300", str(notebook),
        ]
        result = subprocess.run(command, cwd=project)
        if result.returncode != 0:
            return result.returncode

    print(f"Hoàn tất {len(notebooks)} notebook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
