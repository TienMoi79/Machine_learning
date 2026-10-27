"""Bước 4: kiểm chứng cây không cần chuẩn hóa dữ liệu."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def cay_sau_5():
    return DecisionTreeClassifier(max_depth=5, random_state=0)


original = cay_sau_5().fit(X_train, y_train)
pred_original = original.predict(X_test)

scaler = StandardScaler().fit(X_train)
standardized = cay_sau_5().fit(scaler.transform(X_train), y_train)
pred_standardized = standardized.predict(scaler.transform(X_test))

X_train_thousand, X_test_thousand = X_train.copy(), X_test.copy()
X_train_thousand["thu_nhap"] *= 1000
X_test_thousand["thu_nhap"] *= 1000
thousand = cay_sau_5().fit(X_train_thousand, y_train)
pred_thousand = thousand.predict(X_test_thousand)

print("Accuracy trên tập kiểm tra")
print(f"  dữ liệu gốc          : {accuracy_score(y_test, pred_original):.4f}")
print(f"  đã chuẩn hóa         : {accuracy_score(y_test, pred_standardized):.4f}")
print(f"  thu nhập nghìn đồng  : {accuracy_score(y_test, pred_thousand):.4f}")
print("\nSố khách bị đoán khác cây gốc")
print(f"  đã chuẩn hóa         : {(pred_standardized != pred_original).sum()}")
print(f"  thu nhập nghìn đồng  : {(pred_thousand != pred_original).sum()}")
print("\nNgưỡng của câu hỏi gốc")
print(f"  dữ liệu gốc          : {original.tree_.threshold[0]:.2f}")
print(f"  thu nhập nghìn đồng  : {thousand.tree_.threshold[0]:.2f}")
