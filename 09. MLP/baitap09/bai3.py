# -*- coding: utf-8 -*-
"""Bài 3: so sánh các hàm kích hoạt trên bộ hai vầng trăng."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

ket_qua = []
for kich_hoat in ["logistic", "tanh", "relu"]:
    mh = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100, 100), activation=kich_hoat, alpha=1.0,
            solver="lbfgs", max_iter=20000, random_state=0
        ),
    ).fit(X_tr, y_tr)
    ket_qua.append((
        kich_hoat,
        accuracy_score(y_tr, mh.predict(X_tr)),
        accuracy_score(y_te, mh.predict(X_te)),
    ))

print("ham kich hoat   tap hoc   tap kiem tra")
for ten, hoc, kt in ket_qua:
    print(f"{ten:13s}   {hoc:.4f}       {kt:.4f}")
diem_tot = max(dong[2] for dong in ket_qua)
ten_tot = [dong[0] for dong in ket_qua if abs(dong[2] - diem_tot) < 1e-12]
print("\nNhan xet:")
print(f"- Tren lan chia du lieu nay, {' va '.join(ten_tot)} cung dat accuracy kiem tra cao nhat: {diem_tot:.4f}.")
print("- Ham kich hoat tao dang phi tuyen khac nhau; ket qua con phu thuoc du lieu va dieu chuan.")
print("- Nen chon bang kiem dinh cheo thay vi ket luan mot ham luon tot nhat.")
