"""Bài 3: so sánh tiêu chí entropy với Gini."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def chon_do_sau(criterion):
    scores = {
        depth: cross_val_score(
            DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=0),
            X_train,
            y_train,
            cv=5,
        ).mean()
        for depth in range(1, 16)
    }
    best_depth = max(scores, key=scores.get)
    model = DecisionTreeClassifier(
        max_depth=best_depth, criterion=criterion, random_state=0
    ).fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    return best_depth, scores[best_depth], test_accuracy


gini_result = chon_do_sau("gini")
entropy_result = chon_do_sau("entropy")

print(f"{'Tiêu chí':<10} {'độ sâu':>8} {'CV':>10} {'test':>10}")
print(f"{'Gini':<10} {gini_result[0]:8d} {gini_result[1]:10.4f} {gini_result[2]:10.4f}")
print(f"{'Entropy':<10} {entropy_result[0]:8d} {entropy_result[1]:10.4f} {entropy_result[2]:10.4f}")
print(
    f"\nNhận xét: đổi sang entropy làm accuracy kiểm tra thay đổi "
    f"{entropy_result[2] - gini_result[2]:+.4f}; mức thay đổi này không lớn."
)
