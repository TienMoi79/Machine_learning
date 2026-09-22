# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")
nho = df.iloc[[0, 14, 29, 44, 59]]
x = nho["dien_tich"].to_numpy()
y = nho["gia"].to_numpy()

def mse(w, b):
    return ((y - (w * x + b)) ** 2).mean()

print("Nam can duoc chon:")
for i in range(5):
    print(f"  {x[i]:6.1f} m2 -> {y[i]:5.2f} ty")

for w, b in [(0.05, 1.5), (0.08, 0.5)]:
    print(f"\nDuong thang y = {w} * x + {b}")
    for i in range(5):
        du_doan = w * x[i] + b
        print(f"  x = {x[i]:6.1f}  that = {y[i]:5.2f}  du doan = {du_doan:5.2f}  sai so = {y[i] - du_doan:+6.2f}")
    print(f"  MSE = {mse(w, b):.4f}")
