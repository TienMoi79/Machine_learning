# -*- coding: utf-8 -*-
"""Bài 5: cân bằng lớp cho Random Forest và SVC."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = {
    "RF binh thuong": RandomForestClassifier(n_estimators=200, random_state=0),
    "RF can bang": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0),
    "SVC binh thuong": make_pipeline(StandardScaler(), SVC()),
    "SVC can bang": make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
}
ket_qua = {}
print("Mo hinh          precision recall F1")
for ten, mh in mo_hinh.items():
    du = mh.fit(X_tr, y_tr).predict(X_te)
    p = precision_score(y_te, du)
    r = recall_score(y_te, du)
    f1 = f1_score(y_te, du)
    ket_qua[ten] = (p, r, f1)
    print(f"{ten:16s} {p:.4f}    {r:.4f} {f1:.4f}")

doi_rf = sum(abs(a-b) for a, b in zip(ket_qua["RF binh thuong"], ket_qua["RF can bang"]))
doi_svc = sum(abs(a-b) for a, b in zip(ket_qua["SVC binh thuong"], ket_qua["SVC can bang"]))
print("\nNhan xet:")
print(f"- Tong thay doi ba thuoc do: Random Forest={doi_rf:.4f}, SVC={doi_svc:.4f}.")
print(f"- {'SVC' if doi_svc > doi_rf else 'Random Forest'} thay doi nhieu hon theo tong chenh lech nay.")
print("- class_weight='balanced' phat nang loi o lop hiem, thuong lam recall tang nhung precision")
print("  co the giam vi mo hinh bao nhieu mau duong hon.")
