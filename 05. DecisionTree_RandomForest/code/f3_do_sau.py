"""Bước 3: để cây mọc tự do, rồi chọn độ sâu bằng kiểm định chéo."""

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

free_tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print("Cây mọc tự do")
print(f"  độ sâu {free_tree.get_depth()}, số lá {free_tree.get_n_leaves()}")
print(f"  tập học {accuracy_score(y_train, free_tree.predict(X_train)):.4f}")
print(f"  tập kiểm tra {accuracy_score(y_test, free_tree.predict(X_test)):.4f}\n")

print(f"{'độ sâu':>8} {'tập học':>10} {'kiểm định chéo':>15}")
scores = {}
for depth in range(1, 16):
    model = DecisionTreeClassifier(max_depth=depth, random_state=0)
    scores[depth] = cross_val_score(model, X_train, y_train, cv=5).mean()
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    print(f"{depth:8d} {train_accuracy:10.4f} {scores[depth]:15.4f}")

best_depth = max(scores, key=scores.get)
best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=0).fit(X_train, y_train)
print(f"\nĐộ sâu tốt nhất = {best_depth}, cây có {best_tree.get_n_leaves()} lá")
print(f"Accuracy trên tập kiểm tra = {accuracy_score(y_test, best_tree.predict(X_test)):.4f}")
