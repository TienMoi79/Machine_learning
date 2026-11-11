# -*- coding: utf-8 -*-
"""Bài 6: MAE, RMSE và R2 trước/sau một dự đoán hỏng."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich", "so_phong", "tuoi_nha"]]
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17
)
mh = LinearRegression().fit(X_tr, y_tr)
du_goc = mh.predict(X_te)


def tinh_chi_so(y_that, y_doan):
    return (
        mean_absolute_error(y_that, y_doan),
        np.sqrt(mean_squared_error(y_that, y_doan)),
        r2_score(y_that, y_doan),
    )


goc = tinh_chi_so(y_te, du_goc)
du_hong = du_goc.copy()
du_hong[0] += 5.0
hong = tinh_chi_so(y_te, du_hong)
print("                 MAE      RMSE       R2")
print(f"Du doan goc     {goc[0]:.4f}   {goc[1]:.4f}   {goc[2]:.4f}")
print(f"Mot du doan +5  {hong[0]:.4f}   {hong[1]:.4f}   {hong[2]:.4f}")
print("\nMuc thay doi tuyet doi:")
print(f" MAE:  {abs(hong[0]-goc[0]):.4f}")
print(f" RMSE: {abs(hong[1]-goc[1]):.4f}")
print(f" R2:   {abs(hong[2]-goc[2]):.4f}")
print("\nNhan xet:")
print("- RMSE tang manh hon MAE theo don vi gia, vi RMSE binh phuong tung sai so truoc khi lay")
print("  trung binh. Sai so lon 5 ty dong tao thanh 25 trong tong binh phuong sai so.")
print("- MAE chi cong do lon sai so nen moi diem anh huong tuyen tinh. R2 giam vi tong sai so")
print("  binh phuong tang, cho thay mo hinh giai thich du lieu kem hon sau lan doan hong.")
