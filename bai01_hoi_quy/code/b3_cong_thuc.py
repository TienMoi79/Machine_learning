# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")
x = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()
x_tb, y_tb = x.mean(), y.mean()
w = ((x - x_tb) * (y - y_tb)).sum() / ((x - x_tb) ** 2).sum()
b = y_tb - w * x_tb
y_du_doan = w * x + b

print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print(f"MSE tren toan bo 60 can: {((y - y_du_doan) ** 2).mean():.4f}")
print(f"Can 80 m2 duoc du doan: {w * 80 + b:.3f} ty dong")
