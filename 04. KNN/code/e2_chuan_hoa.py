"""Bước 2: đo hậu quả của việc quên chuẩn hóa trên tập kiểm tra."""

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

X3 = df[["gio_on", "diem_giua_ky", "chi_tieu"]]
X_train, X_test, y_train, y_test = train_test_split(
    X3, y, test_size=0.25, random_state=17, stratify=y
)


def danh_gia(model, columns):
    model.fit(X_train[columns], y_train)
    return accuracy_score(y_test, model.predict(X_test[columns]))


def knn_chuan_hoa():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))


ba_cot = ["gio_on", "diem_giua_ky", "chi_tieu"]
hai_cot = ["gio_on", "diem_giua_ky"]

khong_chuan = danh_gia(KNeighborsClassifier(n_neighbors=7), ba_cot)
co_chuan = danh_gia(knn_chuan_hoa(), ba_cot)
bo_chi_tieu = danh_gia(knn_chuan_hoa(), hai_cot)

print(f"Đoán bừa cả lớp qua              : {y_test.mean():.4f}")
print(f"Ba cột, không chuẩn hóa          : {khong_chuan:.4f}")
print(f"Ba cột, có chuẩn hóa             : {co_chuan:.4f}")
print(f"Bỏ cột chi_tieu, có chuẩn hóa    : {bo_chi_tieu:.4f}")
