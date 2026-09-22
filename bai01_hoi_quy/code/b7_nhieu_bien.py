# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")
y = df["gia"]
X_mot = df[["dien_tich"]]
X_ba = df[["dien_tich", "so_phong", "tuoi_nha"]]

for ten, X in [("Mot bien", X_mot), ("Ba bien", X_ba)]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mo_hinh = LinearRegression()
    mo_hinh.fit(X_train, y_train)
    print(f"{ten}: R2 tren tap kiem tra = {r2_score(y_test, mo_hinh.predict(X_test)):.4f}")

X_train, _, y_train, _ = train_test_split(X_ba, y, test_size=0.2, random_state=42)
mo_hinh = LinearRegression()
mo_hinh.fit(X_train, y_train)
print("\nHe so cua tung bien:")
for ten_cot, he_so in zip(X_ba.columns, mo_hinh.coef_):
    print(f"  {ten_cot:12s} {he_so:+.4f}")
print(f"  {'he so chan':12s} {mo_hinh.intercept_:+.4f}")
