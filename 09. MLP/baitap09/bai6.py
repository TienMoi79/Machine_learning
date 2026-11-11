# -*- coding: utf-8 -*-
"""Bài 6: mạng nơ-ron cho hồi quy giá nhà."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17
)

tuyen_tinh = LinearRegression().fit(X_tr, y_tr)
mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0),
).fit(X_tr, y_tr)

mae_lr = mean_absolute_error(y_te, tuyen_tinh.predict(X_te))
mae_mlp = mean_absolute_error(y_te, mlp.predict(X_te))
moi = pd.DataFrame({"dien_tich": [250.0]})
gia_lr = tuyen_tinh.predict(moi)[0]
gia_mlp = mlp.predict(moi)[0]
print(f"MAE hoi quy tuyen tinh: {mae_lr:.4f}")
print(f"MAE MLPRegressor:       {mae_mlp:.4f}")
print(f"Gia du doan can 250 m2 - tuyen tinh: {gia_lr:.4f}")
print(f"Gia du doan can 250 m2 - MLP:        {gia_mlp:.4f}")
print(f"Khoang dien tich du lieu hoc: {X_tr['dien_tich'].min():.1f} den {X_tr['dien_tich'].max():.1f} m2")

print("\nNhan xet:")
print("- Can 250 m2 nam ngoai khoang dien tich hoc, nen day la bai toan ngoai suy.")
print("- Hoi quy tuyen tinh tiep tuc duong thang ra ngoai mien hoc.")
print("- MLP hoc quan he phi tuyen trong mien da thay, nhung khong bi rang buoc phai tiep tuc theo")
print("  xu huong duong thang; vi vay ngoai suy thuong kem on dinh hon hoi quy tuyen tinh.")
