# -*- coding: utf-8 -*-
"""Bài 3: dò C cho kernel đa thức bậc 2, 3 và 4."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

ket_qua = []
for bac in [2, 3, 4]:
    pipe = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=bac))
    tim = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10, 100]}, cv=5)
    tim.fit(X_tr, y_tr)
    ket_qua.append((bac, tim.best_params_["svc__C"], tim.best_score_, tim.score(X_te, y_te)))

print("bac  C tot nhat  diem CV  accuracy kiem tra")
for bac, C, cv, test in ket_qua:
    print(f" {bac:1d}      {C:6g}    {cv:.4f}       {test:.4f}")

tot_nhat = max(ket_qua, key=lambda dong: dong[2])
print("\nNhan xet:")
print(f"- Theo diem kiem dinh cheo, kernel da thuc tot nhat la bac {tot_nhat[0]} "
      f"voi C={tot_nhat[1]:g}, diem CV={tot_nhat[2]:.4f}.")
print(f"- Accuracy kiem tra cua cau hinh nay la {tot_nhat[3]:.4f}.")
print("- Kernel RBF o muc 5 dat 0.9200 tren tap kiem tra; doi chieu bang tren de ket luan")
print("  kernel da thuc co vuot RBF hay khong. Chon mo hinh dua tren CV, khong dua tren tap kiem tra.")
