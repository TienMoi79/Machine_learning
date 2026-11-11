# -*- coding: utf-8 -*-
"""Bài 2: classification_report cho SVM nhận dạng 10 chữ số."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
svm = SVC(kernel="rbf").fit(X_tr, y_tr)
du = svm.predict(X_te)
print(classification_report(y_te, du, digits=4))
bao_cao = classification_report(y_te, du, output_dict=True, zero_division=0)
recall = {int(k): v["recall"] for k, v in bao_cao.items() if k.isdigit()}
precision = {int(k): v["precision"] for k, v in bao_cao.items() if k.isdigit()}
r_min = min(recall.values())
p_min = min(precision.values())
so_recall_thap = [k for k, v in recall.items() if np.isclose(v, r_min)]
so_precision_thap = [k for k, v in precision.items() if np.isclose(v, p_min)]
print("Chu so co recall thap nhat:", so_recall_thap, f"({r_min:.4f})")
print("Chu so co precision thap nhat:", so_precision_thap, f"({p_min:.4f})")
print("\nNhan xet:")
print("- Cac chu so nay co net viet gan voi lop khac, nen de bi bo sot hoac nhan nham.")
print("- Moi lop co khoang 44-46 anh trong tap kiem tra. Vi support gan bang nhau, macro avg")
print("  va weighted avg gan bang nhau; khong co lop dong nao chi phoi trung binh co trong so.")
