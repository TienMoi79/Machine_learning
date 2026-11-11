from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["PYTHONIOENCODING"] = "utf-8"
for ten in ["Lab8_LeQuyetTien.ipynb", "Baitap.ipynb"]:
    print("Dang chay", ten)
    path = ROOT / "code" / ten
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, path)
print("Da chay xong ca hai notebook.")
