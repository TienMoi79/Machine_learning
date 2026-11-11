# -*- coding: utf-8 -*-
"""Bài 2: kiểm chứng một mạng khác cũng tính đúng XOR."""

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
h1 = np.maximum(0, X[:, 0] - X[:, 1])
h2 = np.maximum(0, X[:, 1] - X[:, 0])
y = h1 + h2

print("x1 x2  h1  h2  y")
for x, a, b, c in zip(X, h1, h2, y):
    print(f" {x[0]:.0f}  {x[1]:.0f}   {a:.0f}   {b:.0f}  {c:.0f}")

print("\nNhan xet va giai thich:")
print("- Neu x1=x2 thi ca x1-x2 va x2-x1 deu bang 0, nen h1=h2=0 va y=0.")
print("- Neu x1 khac x2, mot trong hai hieu bang 1 con hieu kia bang -1.")
print("  ReLU giu so 1 va bien so -1 thanh 0, nen h1+h2=1.")
print("- Vi vay y bang 1 khi hai dau vao khac nhau va bang 0 khi chung giong nhau: dung la XOR.")
