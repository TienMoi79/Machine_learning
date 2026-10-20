"""Bài 4: so sánh bỏ phiếu đều và trọng số theo khoảng cách."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print(f"{'k':>4} {'uniform':>10} {'distance':>10}")
for k in [1, 13, 101]:
    results = []
    for weights in ["uniform", "distance"]:
        model = KNeighborsClassifier(n_neighbors=k, weights=weights).fit(X_train, y_train)
        results.append(accuracy_score(y_test, model.predict(X_test)))
    print(f"{k:4d} {results[0]:10.4f} {results[1]:10.4f}")

print("\nVới k lớn, trọng số khoảng cách giúp các điểm gần có tiếng nói mạnh hơn,")
print("nên giảm tác động làm trơn quá mức của nhiều láng giềng ở xa.")
