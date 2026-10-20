"""Bước 1: tự tính khoảng cách và tìm láng giềng, chưa dùng KNN thư viện."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

# A và C học gần giống nhau, nhưng C có mức chi tiêu khác A nhiều.
ba_ban = {
    "A": np.array([5.0, 3.0, 4000.0]),
    "B": np.array([25.0, 9.0, 4100.0]),
    "C": np.array([6.0, 3.5, 7000.0]),
}


def kc(u, v):
    """Khoảng cách Euclid giữa hai điểm."""
    return np.sqrt(((u - v) ** 2).sum())


print("Khoảng cách KHI CHƯA chuẩn hóa")
print(f"  d(A, B) = {kc(ba_ban['A'], ba_ban['B']):8.1f}")
print(f"  d(A, C) = {kc(ba_ban['A'], ba_ban['C']):8.1f}")

df = pd.read_csv(ROOT / "data" / "sinh_vien_mo_rong.csv")
X = df[["gio_on", "diem_giua_ky", "chi_tieu"]].to_numpy(dtype=float)
trung_binh = X.mean(axis=0)
do_lech = X.std(axis=0)
print("\nĐộ lệch chuẩn của ba cột:")
print(
    f"  gio_on {do_lech[0]:.2f}  diem_giua_ky {do_lech[1]:.2f}"
    f"  chi_tieu {do_lech[2]:.0f}"
)

chuan = {ten: (gia_tri - trung_binh) / do_lech for ten, gia_tri in ba_ban.items()}
print("\nKhoảng cách SAU KHI chuẩn hóa")
print(f"  d(A, B) = {kc(chuan['A'], chuan['B']):.2f}")
print(f"  d(A, C) = {kc(chuan['A'], chuan['C']):.2f}")

# Tìm 5 láng giềng gần nhất cho một bạn mới bằng hai cột học tập.
X2 = df[["gio_on", "diem_giua_ky"]].to_numpy(dtype=float)
y = df["qua_mon"].to_numpy()
tb2, dl2 = X2.mean(axis=0), X2.std(axis=0)
ban_moi = np.array([12.0, 6.5])
X2_chuan = (X2 - tb2) / dl2
ban_moi_chuan = (ban_moi - tb2) / dl2
khoang_cach = np.sqrt(((X2_chuan - ban_moi_chuan) ** 2).sum(axis=1))
gan_nhat = np.argsort(khoang_cach)[:5]

print("\nBạn mới: ôn 12 giờ, điểm giữa kỳ 6.5")
print("Năm láng giềng gần nhất:")
for i in gan_nhat:
    ket_qua = "qua" if y[i] == 1 else "rớt"
    print(
        f"  giờ {X2[i, 0]:5.1f}  điểm {X2[i, 1]:4.1f}  "
        f"khoảng cách {khoang_cach[i]:.3f} -> {ket_qua}"
    )
so_qua = int(y[gan_nhat].sum())
print(f"Bỏ phiếu: {so_qua} qua, {5 - so_qua} rớt")
print("Dự đoán:", "qua môn" if so_qua >= 3 else "rớt môn")
