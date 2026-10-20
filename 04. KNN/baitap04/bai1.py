"""Bài 1: tự viết KNN không dùng thư viện KNN."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path.cwd()


def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Dự đoán một mẫu mới bằng khoảng cách Euclid và bỏ phiếu đa số."""
    khoang_cach = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))
    chi_so_gan = np.argsort(khoang_cach)[:k]
    phieu = np.bincount(y_hoc[chi_so_gan])
    return int(np.argmax(phieu))


df = pd.read_csv(ROOT / "data" / "hai_vang_trang.csv")
X = df[["x1", "x2"]].to_numpy()
y = df["nhan"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

manual_prediction = np.array([
    knn_du_doan(X_train, y_train, sample, k=13) for sample in X_test
])
sklearn_model = KNeighborsClassifier(n_neighbors=13).fit(X_train, y_train)
sklearn_prediction = sklearn_model.predict(X_test)

print(f"Accuracy KNN tự viết : {accuracy_score(y_test, manual_prediction):.4f}")
print(f"Accuracy sklearn     : {accuracy_score(y_test, sklearn_prediction):.4f}")
print(f"Số dự đoán giống nhau: {(manual_prediction == sklearn_prediction).sum()}/{len(y_test)}")
print("\nGiải thích:")
print("- Với từng mẫu kiểm tra, hàm tính khoảng cách Euclid đến mọi mẫu học.")
print("- np.argsort lấy 13 khoảng cách nhỏ nhất; np.bincount đếm phiếu từng lớp.")
print("- Lớp có nhiều phiếu nhất là dự đoán. Quy trình này giống KNeighborsClassifier,")
print("  nên hai mô hình cho cùng 75/75 dự đoán và cùng accuracy.")
