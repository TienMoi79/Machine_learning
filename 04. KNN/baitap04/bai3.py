"""Bài 3: đổi đơn vị chi tiêu và kiểm tra tác động lên KNN."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "sinh_vien_mo_rong.csv")
y = df["qua_mon"]
columns = ["gio_on", "diem_giua_ky", "chi_tieu"]

X_nghin = df[columns].copy()
X_trieu = X_nghin.copy()
X_trieu["chi_tieu"] = X_trieu["chi_tieu"] / 1000

train_index, test_index = train_test_split(
    df.index, test_size=0.25, random_state=17, stratify=y
)
y_train, y_test = y.loc[train_index], y.loc[test_index]


def accuracy(X, standardized):
    model = (
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
        if standardized
        else KNeighborsClassifier(n_neighbors=7)
    )
    model.fit(X.loc[train_index], y_train)
    return accuracy_score(y_test, model.predict(X.loc[test_index]))


print("Không chuẩn hóa:")
print(f"  Chi tiêu tính bằng nghìn đồng: {accuracy(X_nghin, False):.4f}")
print(f"  Chi tiêu tính bằng triệu đồng: {accuracy(X_trieu, False):.4f}")
print("Có chuẩn hóa:")
print(f"  Chi tiêu tính bằng nghìn đồng: {accuracy(X_nghin, True):.4f}")
print(f"  Chi tiêu tính bằng triệu đồng: {accuracy(X_trieu, True):.4f}")

print("\nGiải thích: khi chưa chuẩn hóa, đổi đơn vị làm thay đổi độ lớn của cột chi tiêu")
print("và vì thế làm thay đổi khoảng cách. Sau chuẩn hóa, hai cách ghi đơn vị tương đương.")
