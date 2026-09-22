# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")
x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()
x_tb, x_do_lech = x_goc.mean(), x_goc.std()
x = (x_goc - x_tb) / x_do_lech
w, b = 0.0, 0.0
toc_do_hoc, so_vong, n = 0.1, 200, len(x)

for vong in range(1, so_vong + 1):
    chenh_lech = (w * x + b) - y
    w -= toc_do_hoc * (2 / n) * (chenh_lech * x).sum()
    b -= toc_do_hoc * (2 / n) * chenh_lech.sum()
    if vong in (1, 2, 5, 10, 25, 50, 100, 200):
        mse = (((w * x + b) - y) ** 2).mean()
        print(f"{vong:4d}  {w:7.4f}  {b:7.4f}  {mse:8.4f}")

print(f"w = {w / x_do_lech:.6f}")
print(f"b = {b - w * x_tb / x_do_lech:.6f}")
