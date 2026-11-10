# -*- coding: utf-8 -*-
"""Bài 2: thêm một điểm và giải thích bằng khái niệm vectơ hỗ trợ."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(
    n_samples=24, centers=[[2, 2], [6, 6]], cluster_std=0.9, random_state=0
)
y = 1 - y

def hoc_va_in(ten, X_hoc, y_hoc):
    mh = SVC(kernel="linear", C=1000).fit(X_hoc, y_hoc)
    w = mh.coef_[0]
    b = mh.intercept_[0]
    print(f"{ten}")
    print(f"  w = ({w[0]:+.6f}, {w[1]:+.6f}), b = {b:+.6f}")
    print(f"  chi so vecto ho tro = {mh.support_.tolist()}")
    return mh

goc = hoc_va_in("Mo hinh goc", X, y)

X_xa = np.vstack([X, [-1.0, -1.0]])
y_xa = np.append(y, 1)
them_xa = hoc_va_in("\nThem diem lop 1 tai (-1, -1)", X_xa, y_xa)
f_xa = them_xa.decision_function([[-1.0, -1.0]])[0]
print(f"  f(diem moi) = {f_xa:+.6f}; diem moi la vecto ho tro: {24 in them_xa.support_}")

X_gan = np.vstack([X, [4.5, 4.0]])
y_gan = np.append(y, 1)
them_gan = hoc_va_in("\nThay bang diem lop 1 tai (4.5, 4)", X_gan, y_gan)
f_gan = them_gan.decision_function([[4.5, 4.0]])[0]
print(f"  f(diem moi) = {f_gan:+.6f}; diem moi la vecto ho tro: {24 in them_gan.support_}")

print("\nMuc thay doi so voi mo hinh goc:")
for ten, mh in [("(-1, -1)", them_xa), ("(4.5, 4)", them_gan)]:
    print(f"  {ten:10s}: ||delta w|| = {np.linalg.norm(mh.coef_[0]-goc.coef_[0]):.6f}, "
          f"|delta b| = {abs(mh.intercept_[0]-goc.intercept_[0]):.6f}")

print("\nNhan xet:")
print("- Diem (-1,-1) nam xa ranh gioi ve dung phia lop 1, khong thanh vecto ho tro;")
print("  vi vay duong ranh gioi gan nhu khong thay doi.")
print("- Diem (4.5,4) nam sat/vuong vao le, tro thanh vecto ho tro va lam thay doi ranh gioi.")
print("- SVM duoc quyet dinh boi cac diem gan le, khong phai tat ca diem hoc.")
