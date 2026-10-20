"""Bước 5: KNN cho bài toán hồi quy trên dữ liệu giá nhà."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)

print("MAE trên tập kiểm tra (tỷ đồng)")
mae_knn = mean_absolute_error(y_test, knn.predict(X_test))
mae_linear = mean_absolute_error(y_test, linear.predict(X_test))
print(f"  KNN, k = 5 : {mae_knn:.4f}")
print(f"  Đường thẳng : {mae_linear:.4f}")

new_houses = pd.DataFrame({"dien_tich": [60.0, 100.0, 150.0, 250.0]})
print("\nDự đoán cho bốn căn mới (tỷ đồng)")
print(f"{'diện tích':>10} {'KNN':>8} {'đường thẳng':>12}")
for area, knn_price, linear_price in zip(
    new_houses["dien_tich"], knn.predict(new_houses), linear.predict(new_houses)
):
    print(f"{area:7.0f} m2 {knn_price:8.2f} {linear_price:12.2f}")

_, indices = knn.kneighbors(pd.DataFrame({"dien_tich": [250.0]}))
neighbors = X_train.iloc[indices[0]]["dien_tich"].tolist()
print("\nNăm láng giềng của căn 250 m2:", ", ".join(f"{value:.1f}" for value in neighbors))
print("KNN không ngoại suy: ngoài vùng dữ liệu, nó vẫn dùng các căn lớn nhất đã thấy.")
