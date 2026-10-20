"""Bài 2: so sánh khoảng cách Manhattan và Euclid."""

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


def chon_va_cham(metric):
    scores = {
        k: cross_val_score(
            KNeighborsClassifier(n_neighbors=k, metric=metric),
            X_train,
            y_train,
            cv=5,
        ).mean()
        for k in range(1, 52, 2)
    }
    best_k = max(scores, key=scores.get)
    model = KNeighborsClassifier(n_neighbors=best_k, metric=metric).fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    return best_k, scores[best_k], test_accuracy


euclid = chon_va_cham("euclidean")
manhattan = chon_va_cham("manhattan")

print(f"{'Khoảng cách':<12} {'k tốt nhất':>12} {'CV':>8} {'test':>8}")
print(f"{'Euclid':<12} {euclid[0]:12d} {euclid[1]:8.4f} {euclid[2]:8.4f}")
print(f"{'Manhattan':<12} {manhattan[0]:12d} {manhattan[1]:8.4f} {manhattan[2]:8.4f}")
chenh_lech = manhattan[2] - euclid[2]
print(
    f"\nNhận xét: Manhattan chọn k = {manhattan[0]} thay vì {euclid[0]} và accuracy kiểm tra "
    f"thay đổi {chenh_lech:+.4f}. Mức chênh lệch nhỏ, nên đổi khoảng cách không làm kết quả khác nhiều."
)
