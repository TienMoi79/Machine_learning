"""Bước 5: trồng rừng ngẫu nhiên và so sánh với một cây."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

rng = np.random.default_rng(0)
n = len(X_train)
indices = rng.integers(0, n, n)
different = len(np.unique(indices))
print(f"Bốc {n} lần, có {different} khách khác nhau ({different / n:.0%})\n")

forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
print("Rừng 200 cây")
print(f"  tập học     : {accuracy_score(y_train, forest.predict(X_train)):.4f}")
print(f"  tập kiểm tra: {accuracy_score(y_test, forest.predict(X_test)):.4f}")

print("\nSố cây và accuracy tập kiểm tra")
for number in [1, 10, 50, 200, 400]:
    model = RandomForestClassifier(n_estimators=number, random_state=0).fit(X_train, y_train)
    print(f"  {number:4d}: {accuracy_score(y_test, model.predict(X_test)):.4f}")

models = {
    "Cây mọc tự do": DecisionTreeClassifier(random_state=0),
    "Cây sâu 5": DecisionTreeClassifier(max_depth=5, random_state=0),
    "Rừng 200 cây": forest,
}
print("\nMô hình          tập học  tập kiểm tra")
for name, model in models.items():
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<16} {train_accuracy:.4f}    {test_accuracy:.4f}")
