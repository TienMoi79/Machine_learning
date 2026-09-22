# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/gia_nha.csv")

plt.figure(figsize=(8, 5))
plt.scatter(df["so_phong"], df["gia"], color="royalblue", alpha=0.75, edgecolors="black")
plt.xlabel("Số phòng ngủ")
plt.ylabel("Giá bán (tỷ đồng)")
plt.title("Giá căn hộ theo số phòng ngủ")
plt.xticks(sorted(df["so_phong"].unique()))
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("baitap01/bai2.png", dpi=150)

print("Nhan xet: Nhin chung, can ho co nhieu phong ngu hon thuong co gia cao hon. Tuy nhien, cac muc gia van phan tan nen so phong khong phai la yeu to duy nhat quyet dinh gia.")
plt.show()
