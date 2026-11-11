# -*- coding: utf-8 -*-
"""Bài 1: tự viết hàm đánh giá chỉ bằng numpy."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def chia_an_toan(tu_so, mau_so):
    return tu_so / mau_so if mau_so != 0 else 0.0


def danh_gia(y_that, y_doan):
    y_that = np.asarray(y_that)
    y_doan = np.asarray(y_doan)
    tp = np.sum((y_that == 1) & (y_doan == 1))
    fp = np.sum((y_that == 0) & (y_doan == 1))
    fn = np.sum((y_that == 1) & (y_doan == 0))
    tn = np.sum((y_that == 0) & (y_doan == 0))
    accuracy = chia_an_toan(tp + tn, tp + fp + fn + tn)
    precision = chia_an_toan(tp, tp + fp)
    recall = chia_an_toan(tp, tp + fn)
    f1 = chia_an_toan(2 * precision * recall, precision + recall)
    return accuracy, precision, recall, f1


df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = {
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Mo hinh luoi": DummyClassifier(strategy="most_frequent"),
}
for ten, mh in mo_hinh.items():
    du = mh.fit(X_tr, y_tr).predict(X_te)
    tu_tinh = danh_gia(y_te, du)
    thu_vien = (
        accuracy_score(y_te, du),
        precision_score(y_te, du, zero_division=0),
        recall_score(y_te, du, zero_division=0),
        f1_score(y_te, du, zero_division=0),
    )
    print(ten)
    print("  Tu tinh :", tuple(round(float(x), 4) for x in tu_tinh))
    print("  Thu vien:", tuple(round(x, 4) for x in thu_vien))

print("\nNhan xet va giai thich:")
print("- Mo hinh luoi khong bao mot mau duong nao, nen TP+FP=0 va precision co phep chia cho 0.")
print("- Ve mat toan hoc precision khi do khong xac dinh. Ham nay quy uoc tra 0.0, giong")
print("  scikit-learn khi dat zero_division=0, de chuong trinh khong loi va y nghia ro rang.")
print("- Accuracy cua mo hinh luoi cao do lop 0 chiem da so, nhung recall lop bo hoc bang 0.")
