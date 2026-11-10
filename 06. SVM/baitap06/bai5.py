# -*- coding: utf-8 -*-
"""Bài 5: bảng nhầm lẫn và trực quan các ảnh SVM đoán sai."""

from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
svm = SVC(kernel="rbf").fit(X_tr, y_tr)
du_doan = svm.predict(X_te)
cm = confusion_matrix(y_te, du_doan)
print("Bang nham lan (hang=chu so that, cot=chu so doan):")
print(cm)

sai = np.flatnonzero(du_doan != y_te)
cap_sai = Counter(zip(y_te[sai], du_doan[sai]))
print("\nCac cap bi nham:")
for (that, doan), so_lan in cap_sai.most_common():
    print(f"  {that} -> {doan}: {so_lan} lan")

so_anh = len(sai)
fig, axes = plt.subplots(1, so_anh, figsize=(3 * so_anh, 3))
axes = np.atleast_1d(axes)
for ax, i in zip(axes, sai):
    ax.imshow(X_te[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"That {y_te[i]}, doan {du_doan[i]}")
    ax.axis("off")
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/bai5_chu_so_doan_sai.png", dpi=160, bbox_inches="tight")
plt.show()

print("\nNhan xet:")
print("- Mo hinh chi sai o mot vai chu so co net viet mo, dut hoac hinh dang gan voi lop khac.")
print("- Vi du 9 co the giong 3 khi net vong ben trai mo; 8 co the giong 1 neu hai vong bi dut;")
print("  4 co the giong 8 khi cac net khep kin; 7 co the giong 9 khi co them net cong.")
print("- Da luu hinh cac anh sai tai figures/bai5_chu_so_doan_sai.png.")
