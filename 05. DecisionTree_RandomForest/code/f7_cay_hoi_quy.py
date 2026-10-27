"""Bước 7: cây quyết định cho bài toán hồi quy trên dữ liệu giá nhà."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)

print(f"Cây hồi quy sâu 3 có {tree.get_n_leaves()} lá")
print(export_text(tree, feature_names=["dien_tich"], decimals=1))
mae_tree = mean_absolute_error(y_test, tree.predict(X_test))
mae_linear = mean_absolute_error(y_test, linear.predict(X_test))
print("MAE trên tập kiểm tra (tỷ đồng)")
print(f"  Cây sâu 3   : {mae_tree:.4f}")
print(f"  Đường thẳng : {mae_linear:.4f}")

new_houses = pd.DataFrame({"dien_tich": [60.0, 100.0, 150.0, 250.0]})
print("\nDự đoán cho bốn căn mới (tỷ đồng)")
print(f"{'diện tích':>10} {'cây':>8} {'đường thẳng':>12}")
for area, tree_price, linear_price in zip(
    new_houses["dien_tich"], tree.predict(new_houses), linear.predict(new_houses)
):
    print(f"{area:7.0f} m2 {tree_price:8.2f} {linear_price:12.2f}")

print("\nCây không ngoại suy: các diện tích vượt ngưỡng cuối cùng đều rơi vào cùng một lá.")
