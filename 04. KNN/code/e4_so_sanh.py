"""Bước 4: so sánh KNN với hai mô hình đã học trên dữ liệu có biên cong."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

models = [
    ("Hồi quy logistic", LogisticRegression()),
    ("Naive Bayes", GaussianNB()),
    ("KNN, k = 13", KNeighborsClassifier(n_neighbors=13)),
]

print("Accuracy trên tập kiểm tra, bộ hai vầng trăng")
for name, model in models:
    model.fit(X_train, y_train)
    print(f"  {name:<18}: {accuracy_score(y_test, model.predict(X_test)):.4f}")

print("\nKNN phù hợp hơn vì có thể tạo biên quyết định cong theo hình dạng dữ liệu.")
