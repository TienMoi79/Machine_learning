"""Bước 3: chọn số láng giềng k bằng kiểm định chéo."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("Accuracy trên tập học và tập kiểm tra theo k")
print(f"{'k':>4} {'tập học':>10} {'tập kiểm tra':>14}")
for k in [1, 5, 13, 31, 101]:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    print(
        f"{k:4d} {accuracy_score(y_train, model.predict(X_train)):10.4f} "
        f"{accuracy_score(y_test, model.predict(X_test)):14.4f}"
    )

print("\nKiểm định chéo 5 phần trên tập học:")
scores = {
    k: cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5).mean()
    for k in range(1, 52, 2)
}
k_tot = max(scores, key=scores.get)
for k in [1, 5, 9, 13, 17, 25, 51]:
    note = " <- tốt nhất" if k == k_tot else ""
    print(f"k = {k:2d}  điểm trung bình {scores[k]:.4f}{note}")

model = KNeighborsClassifier(n_neighbors=k_tot).fit(X_train, y_train)
print(f"\nk tốt nhất = {k_tot}")
print(f"Accuracy trên tập kiểm tra = {accuracy_score(y_test, model.predict(X_test)):.4f}")
