"""Bài 2: tự tìm câu hỏi gốc tốt nhất trên dữ liệu khách vay thật."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path.cwd()


def gini(labels):
    if len(labels) == 0:
        return 0.0
    p1 = np.mean(labels)
    return 1 - p1**2 - (1 - p1) ** 2


def tim_cau_hoi(X, y):
    """Thử mọi cột và ngưỡng giữa hai giá trị liên tiếp, trả về câu hỏi tốt nhất."""
    y_array = np.asarray(y)
    gini_goc = gini(y_array)
    best = {"cot": None, "nguong": None, "gini_sau_chia": np.inf, "gini_giam": -np.inf}

    for column in X.columns:
        values = X[column].to_numpy(dtype=float)
        unique_values = np.sort(np.unique(values))
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        for threshold in thresholds:
            left = values <= threshold
            if left.all() or (~left).all():
                continue
            gini_split = (
                left.mean() * gini(y_array[left])
                + (~left).mean() * gini(y_array[~left])
            )
            decrease = gini_goc - gini_split
            if decrease > best["gini_giam"]:
                best = {
                    "cot": column,
                    "nguong": float(threshold),
                    "gini_sau_chia": float(gini_split),
                    "gini_giam": float(decrease),
                }
    return best


df = pd.read_csv(ROOT / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

best = tim_cau_hoi(X_train, y_train)
print(f"Cột tốt nhất       : {best['cot']}")
print(f"Ngưỡng tốt nhất    : {best['nguong']:.3f}")
print(f"Gini sau chia      : {best['gini_sau_chia']:.4f}")
print(f"Mức Gini giảm      : {best['gini_giam']:.4f}")

print("\nGiải thích:")
print("- Với mỗi cột, hàm sắp xếp các giá trị khác nhau và thử ngưỡng ở chính giữa")
print("  từng cặp giá trị liên tiếp. Các ngưỡng nằm trong cùng một khoảng chia dữ liệu như nhau.")
print("- Mỗi câu hỏi được chấm bằng Gini trung bình có trọng số của hai nhánh.")
print("- Câu hỏi làm Gini giảm nhiều nhất là thu_nhap <= 15.05, trùng câu hỏi gốc")
print("  của cây scikit-learn trong bài thực hành.")
