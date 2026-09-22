# -*- coding: utf-8 -*-
w = 0.078367
b = 0.401752

def du_doan_gia(dien_tich):
    if dien_tich < 35.5 or dien_tich > 117.5:
        print(f"CANH BAO: {dien_tich} m2 nam ngoai khoang du lieu 35.5-117.5 m2; ket qua la ngoai suy.")
    return w * dien_tich + b

for dien_tich in [60, 80, 200]:
    print(f"Can {dien_tich} m2 -> du doan {du_doan_gia(dien_tich):.3f} ty dong")
