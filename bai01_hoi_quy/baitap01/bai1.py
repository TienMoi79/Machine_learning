# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")
nhom_lon = df[df["dien_tich"] > 100]

print("BAI 1: LOC NHOM CAN HO LON")
print(f"So can co dien tich lon hon 100 m2: {len(nhom_lon)}")
print(f"Gia trung binh cua nhom: {nhom_lon['gia'].mean():.4f} ty dong")
