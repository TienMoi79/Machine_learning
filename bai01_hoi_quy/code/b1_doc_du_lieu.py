# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")
print("Kich thuoc bang (so dong, so cot):", df.shape)
print("\nNam dong dau tien:")
print(df.head())
print("\nTen cac cot:", list(df.columns))
print("\nThong ke nhanh cot dien_tich va cot gia:")
print(df[["dien_tich", "gia"]].describe().round(2))
print("\nSo o bi thieu trong tung cot:")
print(df.isna().sum())
