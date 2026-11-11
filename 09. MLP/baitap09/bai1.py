# -*- coding: utf-8 -*-
"""Bài 1: tự dự đoán bằng trọng số MLP đã học."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


df = pd.read_csv("data/xor.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mlp = MLPClassifier(
    hidden_layer_sizes=(8,), solver="lbfgs", max_iter=5000, random_state=0
).fit(X_tr, y_tr)

X5 = X_te.iloc[:5].to_numpy()
W1, W2 = mlp.coefs_
b1, b2 = mlp.intercepts_
H = relu(X5 @ W1 + b1)
xac_suat_tu_tinh = sigmoid(H @ W2 + b2).ravel()
xac_suat_thu_vien = mlp.predict_proba(X_te.iloc[:5])[:, 1]

print(" i   tu tinh        predict_proba   chenh lech")
for i, (a, b) in enumerate(zip(xac_suat_tu_tinh, xac_suat_thu_vien), 1):
    print(f" {i}   {a:.10f}    {b:.10f}      {abs(a-b):.2e}")
print("\nSai khac lon nhat:", np.max(np.abs(xac_suat_tu_tinh - xac_suat_thu_vien)))
print("\nNhan xet:")
print("- Lop an tinh X @ W1 + b1 roi dua qua ReLU; lop ra tinh H @ W2 + b2 roi qua sigmoid.")
print("- Ket qua tu tinh trung voi predict_proba den sai so may, chung to lan truyen tien da dung.")
print("- coefs_ chua ma tran trong so giua cac lop; intercepts_ chua he so chan cua tung lop.")
