# -*- coding: utf-8 -*-
"""Bước 1: tìm đường có lề rộng nhất và các vectơ hỗ trợ."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(
    n_samples=24, centers=[[2, 2], [6, 6]], cluster_std=0.9, random_state=0
)
y = 1 - y

svm = SVC(kernel="linear", C=1000)
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]
print(f"w1 = {w[0]:.3f}, w2 = {w[1]:.3f}, b = {b:.3f}")
do_dai_w = np.linalg.norm(w)
print(f"Do rong cua le = 2 / {do_dai_w:.4f} = {2 / do_dai_w:.4f}")
print()

print("So vecto ho tro:", len(svm.support_))
for i in svm.support_:
    f = svm.decision_function(X[i:i + 1])[0]
    print(
        f" diem {i:2d}: x1 = {X[i, 0]:.2f}, x2 = {X[i, 1]:.2f},"
        f" lop {y[i]}, f(x) = {f:+.3f}"
    )
print()

f_tat_ca = svm.decision_function(X)
ngoai = np.setdiff1d(np.arange(len(X)), svm.support_)
print(f"|f(x)| nho nhat cua cac diem khac: {np.abs(f_tat_ca[ngoai]).min():.3f}")
print()

chi_sv = SVC(kernel="linear", C=1000)
chi_sv.fit(X[svm.support_], y[svm.support_])
w2 = chi_sv.coef_[0]
b2 = chi_sv.intercept_[0]
print("Hoc lai chi voi vecto ho tro:")
print(f" w1 = {w2[0]:.3f}, w2 = {w2[1]:.3f}, b = {b2:.3f}")
print()

moi = np.array([[3.0, 3.0], [5.0, 4.5]])
for diem, f, lop in zip(moi, svm.decision_function(moi), svm.predict(moi)):
    print(f"Diem ({diem[0]}, {diem[1]}): f(x) = {f:+.3f}, doan lop {lop}")
