# -*- coding: utf-8 -*-
"""Bài 6: đo thời gian SVC và LinearSVC khi số mẫu tăng."""

from time import perf_counter
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, SVC

dong = []
for n in [2000, 5000, 10000, 20000]:
    X, y = make_classification(
        n_samples=n, n_features=20, random_state=0
    )
    for ten, mh in [
        ("SVC", SVC()),
        ("LinearSVC", LinearSVC(dual="auto", max_iter=10000, random_state=0)),
    ]:
        bat_dau = perf_counter()
        mh.fit(X, y)
        thoi_gian = perf_counter() - bat_dau
        dong.append({"n": n, "mo_hinh": ten, "giay": thoi_gian})

bang = pd.DataFrame(dong)
pivot = bang.pivot(index="n", columns="mo_hinh", values="giay")
print("Thoi gian fit (giay):")
print(pivot.round(4).to_string())

print("\nHe so tang thoi gian khi n tang:")
for ten in ["SVC", "LinearSVC"]:
    ds = pivot[ten].to_numpy()
    print(ten, [round(ds[i] / ds[i-1], 2) for i in range(1, len(ds))])

print("\nNhan xet:")
print("- LinearSVC tang gan tuyen tinh theo so mau va phu hop hon cho du lieu lon.")
print("- SVC kernel RBF phai tinh quan he giua nhieu cap diem, nen thoi gian thuong tang nhanh")
print("  hon tuyen tinh; khi n tang gap doi, thoi gian co the tang nhieu hon gap doi.")
print("- So lieu thuc te phu thuoc cau hinh may, nhung xu huong ve do phuc tap van giu nguyen.")
