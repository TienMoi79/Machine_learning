"""Bài 4: cắt tỉa bằng số khách tối thiểu ở lá."""

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

leaf_sizes = [1, 2, 5, 10, 20, 40]
scores = {}
print(f"{'min_samples_leaf':>18} {'CV trung bình':>14}")
for leaf_size in leaf_sizes:
    model = DecisionTreeClassifier(min_samples_leaf=leaf_size, random_state=0)
    scores[leaf_size] = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"{leaf_size:18d} {scores[leaf_size]:14.4f}")

best_leaf_size = max(scores, key=scores.get)
best_score = scores[best_leaf_size]
ties = [size for size in leaf_sizes if abs(scores[size] - best_score) < 1e-12]
best_model = DecisionTreeClassifier(
    min_samples_leaf=best_leaf_size, random_state=0
).fit(X_train, y_train)
leaf_test_accuracy = accuracy_score(y_test, best_model.predict(X_test))

depth_5 = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
depth_5_accuracy = accuracy_score(y_test, depth_5.predict(X_test))

print(f"\nGiá trị có điểm CV cao nhất: {', '.join(map(str, ties))}")
print(f"Chọn min_samples_leaf = {best_leaf_size} (giá trị nhỏ hơn khi hòa điểm)")
print(f"Accuracy cách cắt theo số khách ở lá: {leaf_test_accuracy:.4f}")
print(f"Accuracy cây sâu tối đa 5          : {depth_5_accuracy:.4f}")
if leaf_test_accuracy > depth_5_accuracy:
    print("Cắt theo min_samples_leaf tốt hơn trên tập kiểm tra của bộ dữ liệu này.")
elif leaf_test_accuracy < depth_5_accuracy:
    print("Giới hạn max_depth=5 tốt hơn trên tập kiểm tra của bộ dữ liệu này.")
else:
    print("Hai cách cắt tỉa cho cùng accuracy trên tập kiểm tra.")
