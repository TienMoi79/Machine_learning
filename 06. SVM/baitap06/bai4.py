# -*- coding: utf-8 -*-
"""Bài 4: dò lưới mịn quanh C=10, gamma=1."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

luoi = {"C": [3, 10, 30], "gamma": [0.3, 1, 3]}
tim = GridSearchCV(SVC(kernel="rbf"), luoi, cv=5)
tim.fit(X_tr, y_tr)
bang = pd.DataFrame(tim.cv_results_["params"])
bang["diem"] = tim.cv_results_["mean_test_score"]
print("Diem kiem dinh cheo cua luoi min:")
print(bang.pivot(index="C", columns="gamma", values="diem").round(4).to_string())
print("\nCap tot nhat:", tim.best_params_)
print(f"Diem CV tot nhat: {tim.best_score_:.4f}")
print(f"Accuracy tap kiem tra: {tim.score(X_te, y_te):.4f}")

print("\nNhan xet:")
print(f"- Diem CV tot nhat {'cao hon' if tim.best_score_ > 0.8711 else 'khong cao hon'} 0.8711 cua luoi tho.")
print(f"- Accuracy kiem tra {'doi' if abs(tim.score(X_te,y_te)-0.9200)>1e-12 else 'khong doi'}, "
      f"so voi 0.9200 cua luoi tho.")
print("- Luoi min giup kiem tra ky vung tham so tot, nhung khong bao dam tap kiem tra se tang.")
