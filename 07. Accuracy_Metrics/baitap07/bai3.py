# -*- coding: utf-8 -*-
"""Bài 3: chọn ngưỡng lớn nhất để recall kiểm định chéo đạt ít nhất 0.8."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
lr = make_pipeline(StandardScaler(), LogisticRegression())
xs_hoc = cross_val_predict(lr, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
cac_nguong = np.round(np.arange(0.01, 1.00, 0.01), 2)
hop_le = []
for t in cac_nguong:
    du_hoc = (xs_hoc >= t).astype(int)
    if recall_score(y_tr, du_hoc) >= 0.8:
        hop_le.append(t)
nguong = max(hop_le)
recall_cv = recall_score(y_tr, (xs_hoc >= nguong).astype(int))

lr.fit(X_tr, y_tr)
xs_te = lr.predict_proba(X_te)[:, 1]
du_te = (xs_te >= nguong).astype(int)
print(f"Nguong lon nhat co recall CV >= 0.8: {nguong:.2f}")
print(f"Recall tren du doan cheo tap hoc: {recall_cv:.4f}")
print(f"Tap kiem tra - recall:    {recall_score(y_te, du_te):.4f}")
print(f"Tap kiem tra - precision: {precision_score(y_te, du_te):.4f}")
print(f"So ban bi bao: {du_te.sum()} tren {len(du_te)}")

print("\nNhan xet va giai thich:")
print("- Nguong duoc chon chi tu xac suat cross_val_predict tren tap hoc. Moi xac suat den tu")
print("  mot mo hinh chua tung thay mau do, nen khong nhin dap an cua tap kiem tra.")
print("- Chon nguong lon nhat van dat recall >=0.8 giup han che bao dong nham trong cac nguong")
print("  thoa yeu cau. Tap kiem tra chi dung mot lan sau khi nguong da duoc khoa.")
print("- Recall tap kiem tra co the khac 0.8 do sai khac lay mau; yeu cau tren tap hoc khong bao")
print("  dam chinh xac cung mot con so tren du lieu moi.")
