# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")
y = df["gia"]

def tinh_r2(cac_cot):
    X = df[cac_cot]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mo_hinh = LinearRegression()
    mo_hinh.fit(X_train, y_train)
    return r2_score(y_test, mo_hinh.predict(X_test))

r2_mot_bien = tinh_r2(["dien_tich"])
r2_hai_bien = tinh_r2(["dien_tich", "so_phong"])

print("BAI 4: THEM SO PHONG VAO MO HINH")
print(f"R2 cua mo hinh mot bien: {r2_mot_bien:.4f}")
print(f"R2 cua mo hinh hai bien: {r2_hai_bien:.4f}")
print(f"Muc thay doi R2: {r2_hai_bien - r2_mot_bien:+.4f}")

if r2_hai_bien > r2_mot_bien:
    print("Nhan xet: Them so phong lam R2 tang, nhung can dua vao muc tang de danh gia muc cai thien.")
else:
    print("Nhan xet: Them so phong khong giup mo hinh du doan tot hon tren tap kiem tra nay.")
