# -*- coding: utf-8 -*-
"""Bài 1: tính tay khoảng cách tới ranh giới."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(
    n_samples=24, centers=[[2, 2], [6, 6]], cluster_std=0.9, random_state=0
)
y = 1 - y
svm = SVC(kernel="linear", C=1000).fit(X, y)
w = svm.coef_[0]
b = svm.intercept_[0]
do_dai_w = np.sqrt(w[0] ** 2 + w[1] ** 2)

print(f"w1 = {w[0]:.6f}, w2 = {w[1]:.6f}, b = {b:.6f}")
print(f"||w|| = {do_dai_w:.6f}")
print("\nHai diem can tinh:")
for diem in np.array([[3.0, 3.0], [5.0, 4.5]]):
    f = w @ diem + b
    khoang_cach = abs(f) / do_dai_w
    print(f"  {tuple(diem)}: f(x) = {f:+.6f}, khoang cach = {khoang_cach:.6f}")

print("\nBa vecto ho tro:")
for i in svm.support_:
    f = w @ X[i] + b
    khoang_cach = abs(f) / do_dai_w
    print(f"  diem {i:2d} {tuple(X[i].round(4))}: f(x) = {f:+.6f}, khoang cach = {khoang_cach:.6f}")

print("\nNhan xet:")
print(f"- Moi vecto ho tro cach ranh gioi xap xi 1/||w|| = {1/do_dai_w:.6f}.")
print("- Ly do: vecto ho tro nam tren hai mep le f(x)=+1 hoac f(x)=-1,")
print("  nen |f(x)|/||w|| = 1/||w||. Hai mep le cach nhau 2/||w||.")
