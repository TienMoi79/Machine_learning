# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")
x = df["tuoi_nha"].to_numpy()
y = df["gia"].to_numpy()

x_tb = x.mean()
y_tb = y.mean()
w = ((x - x_tb) * (y - y_tb)).sum() / ((x - x_tb) ** 2).sum()
b = y_tb - w * x_tb

print("BAI 3: DUNG TUOI NHA DE DU DOAN GIA")
print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print("Nhan xet: w mang dau am, nghia la khi tuoi nha tang them mot nam thi gia du doan co xu huong giam.")
print("Giai thich: Can ho cu hon thuong xuong cap va co thoi gian su dung con lai ngan hon, nen gia ban thuong thap hon.")
print("Luu y: Day la quan he thong ke trong bo du lieu, khong phai ket luan nhan qua.")
