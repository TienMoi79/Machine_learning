# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mo_hinh = LinearRegression()
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("So can de hoc:", len(X_train))
print("So can de kiem tra:", len(X_test))
print(f"MAE = {mean_absolute_error(y_test, y_pred):.4f} ty dong")
print(f"MSE = {mse:.4f}")
print(f"RMSE = {np.sqrt(mse):.4f} ty dong")
print(f"R2 = {r2_score(y_test, y_pred):.4f}")
