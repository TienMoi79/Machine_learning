"""Bước 6: đọc độ quan trọng của đặc trưng và thấy giới hạn của nó."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
importance = pd.Series(forest.feature_importances_, index=X.columns)
print("Độ quan trọng, bốn cột thật")
print(importance.sort_values(ascending=False).round(3).to_string())
print("Tổng:", round(importance.sum(), 3))

rng = np.random.default_rng(0)
X_train_random = X_train.copy()
X_train_random["so_ngau_nhien"] = rng.random(len(X_train))
forest_random = RandomForestClassifier(n_estimators=200, random_state=0).fit(
    X_train_random, y_train
)
importance_random = pd.Series(
    forest_random.feature_importances_, index=X_train_random.columns
)
print("\nĐộ quan trọng khi thêm một cột ngẫu nhiên")
print(importance_random.sort_values(ascending=False).round(3).to_string())
