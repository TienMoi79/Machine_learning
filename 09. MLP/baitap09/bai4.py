# -*- coding: utf-8 -*-
"""Bài 4: ảnh hưởng của tốc độ học trên bộ chữ số."""

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

fig, ax = plt.subplots(figsize=(8, 5))
ket_qua = []
for toc_do in [0.0001, 0.001, 0.01, 0.1]:
    mh = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100,), learning_rate_init=toc_do,
            max_iter=1000, random_state=0
        ),
    ).fit(X_tr, y_tr)
    mlp = mh[-1]
    acc = accuracy_score(y_te, mh.predict(X_te))
    ket_qua.append((toc_do, mlp.n_iter_, mlp.loss_curve_[-1], acc))
    ax.plot(mlp.loss_curve_, label=f"learning_rate={toc_do:g}")

ax.set_xlabel("Vong hoc")
ax.set_ylabel("Loss")
ax.set_title("Anh huong cua learning_rate_init")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/toc_do_hoc.png", dpi=160, bbox_inches="tight")
if "agg" not in plt.get_backend().lower():
    plt.show()
else:
    plt.close(fig)

print("toc do hoc   so vong   loss cuoi   accuracy")
for toc_do, so_vong, loss, acc in ket_qua:
    print(f"{toc_do:10g}   {so_vong:7d}   {loss:9.5f}   {acc:.4f}")
print("\nNhan xet:")
print("- Toc do qua nho lam loss giam cham, can nhieu vong va co the chua den diem tot.")
print("- Toc do qua lon co the vuot qua diem toi uu, loss dao dong hoac hoi tu den nghiem kem.")
print("- Toc do vua phai giup loss giam nhanh, on dinh va accuracy kiem tra cao.")
print("- Da luu hinh tai figures/toc_do_hoc.png.")
