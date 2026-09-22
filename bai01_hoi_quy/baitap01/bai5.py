# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")
x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()
x = (x_goc - x_goc.mean()) / x_goc.std()
n = len(x)

def chay_gradient_descent(toc_do_hoc, so_vong=200):
    w, b = 0.0, 0.0
    for _ in range(so_vong):
        chenh_lech = (w * x + b) - y
        grad_w = (2 / n) * (chenh_lech * x).sum()
        grad_b = (2 / n) * chenh_lech.sum()
        w -= toc_do_hoc * grad_w
        b -= toc_do_hoc * grad_b
    mse = (((w * x + b) - y) ** 2).mean()
    return mse

for toc_do_hoc in (0.001, 1.02):
    mse = chay_gradient_descent(toc_do_hoc)
    print(f"Toc do hoc = {toc_do_hoc}: MSE tai vong 200 = {mse:.6e}")

print("Giai thich: 0.001 qua nho nen hoi tu cham; 1.02 qua lon nen nhay qua day va lam MSE tang manh.")
