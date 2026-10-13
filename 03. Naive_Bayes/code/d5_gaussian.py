"""Bước 5: Gaussian Naive Bayes cho dữ liệu sinh viên liên tục."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

gaussian = GaussianNB()
logistic = LogisticRegression(max_iter=1000)
gaussian.fit(X_train, y_train)
logistic.fit(X_train, y_train)

print("=== So sánh độ chính xác ===")
print(f"GaussianNB         : {accuracy_score(y_test, gaussian.predict(X_test)):.4f}")
print(f"LogisticRegression : {accuracy_score(y_test, logistic.predict(X_test)):.4f}")

print("\n=== Tham số GaussianNB học được ===")
for index, label in enumerate(gaussian.classes_):
    means = gaussian.theta_[index]
    stds = np.sqrt(gaussian.var_[index])
    print(
        f"Lớp {label}: mean(gio_on, diem_giua_ky) = ({means[0]:.2f}, {means[1]:.2f}), "
        f"std = ({stds[0]:.2f}, {stds[1]:.2f}), prior = {gaussian.class_prior_[index]:.4f}"
    )

new_students = pd.DataFrame(
    [[5, 4], [13, 6], [20, 7], [28, 9]], columns=["gio_on", "diem_giua_ky"]
)
pass_probabilities = gaussian.predict_proba(new_students)[:, 1]
print("\n=== Dự đoán sinh viên mới ===")
for row, probability in zip(new_students.itertuples(index=False), pass_probabilities):
    print(f"({row.gio_on:>2}, {row.diem_giua_ky}) -> P(qua môn) = {probability:.4f}")
