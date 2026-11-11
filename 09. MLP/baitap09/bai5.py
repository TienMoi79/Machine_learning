# -*- coding: utf-8 -*-
"""Bài 5: dò lưới cấu trúc lớp ẩn và alpha cho MLP."""

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

pipe = make_pipeline(
    StandardScaler(),
    MLPClassifier(max_iter=500, random_state=0),
)
luoi = {
    "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
    "mlpclassifier__alpha": [0.0001, 0.01, 1.0],
}
tim = GridSearchCV(pipe, luoi, cv=3, n_jobs=-1)
tim.fit(X_tr, y_tr)
acc_mlp = accuracy_score(y_te, tim.predict(X_te))
svm = SVC().fit(X_tr, y_tr)
acc_svm = accuracy_score(y_te, svm.predict(X_te))

print("Cap tham so tot nhat:", tim.best_params_)
print(f"Diem kiem dinh cheo tot nhat: {tim.best_score_:.4f}")
print(f"Accuracy MLP tren tap kiem tra: {acc_mlp:.4f}")
print(f"Accuracy SVM kernel RBF:        {acc_svm:.4f}")
print("\nNhan xet:")
print(f"- Cau hinh tot nhat theo kiem dinh cheo la {tim.best_params_}.")
print(f"- MLP {'vuot' if acc_mlp > acc_svm else 'khong vuot'} SVM tren tap kiem tra nay.")
print("- Tap kiem tra chi duoc dung sau khi chon tham so; khong dung no de chon mo hinh.")
