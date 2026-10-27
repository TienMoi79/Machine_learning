"""Bước 1: tự tính chỉ số Gini và tự chọn câu hỏi tốt nhất."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


nho = pd.DataFrame(
    {
        "thu_nhap": [6, 8, 9, 11, 13, 15, 18, 20, 25, 30],
        "co_viec_on_dinh": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        "tra_dung_han": [0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    }
)
y = nho["tra_dung_han"].to_numpy()


def gini(labels):
    """Chỉ số Gini của một nút có hai lớp."""
    if len(labels) == 0:
        return 0.0
    p1 = labels.mean()
    return 1 - p1**2 - (1 - p1) ** 2


def gini_sau_chia(di_trai):
    """Gini trung bình có trọng số của hai nhánh."""
    trai, phai = y[di_trai], y[~di_trai]
    n = len(y)
    return len(trai) / n * gini(trai) + len(phai) / n * gini(phai)


gini_goc = gini(y)
print(f"Gini của nút gốc: {gini_goc:.4f}\n")

thu_nhap = nho["thu_nhap"].to_numpy()
muc = np.sort(np.unique(thu_nhap))
print("Câu hỏi 'thu_nhap <= ngưỡng?'")
print(f"{'ngưỡng':>8} {'Gini sau chia':>15} {'Gini giảm':>12}")
for a, b in zip(muc[:-1], muc[1:]):
    nguong = (a + b) / 2
    gini_chia = gini_sau_chia(thu_nhap <= nguong)
    print(f"{nguong:8.1f} {gini_chia:15.4f} {gini_goc - gini_chia:12.4f}")

viec_lam = nho["co_viec_on_dinh"].to_numpy()
gini_viec = gini_sau_chia(viec_lam == 0)
print("\nCâu hỏi 'có việc ổn định?'")
print(f"Gini sau chia {gini_viec:.4f}, Gini giảm {gini_goc - gini_viec:.4f}")

features = ["thu_nhap", "co_viec_on_dinh"]
tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(nho[features], y)
print("\nCây một tầng do scikit-learn chọn:")
print(export_text(tree, feature_names=features, show_weights=True))
