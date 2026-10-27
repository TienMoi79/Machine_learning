"""Bài 6: rừng ngẫu nhiên cho bài toán hồi quy."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

forest = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_train, y_train)
tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)

mae_forest = mean_absolute_error(y_test, forest.predict(X_test))
mae_tree = mean_absolute_error(y_test, tree.predict(X_test))
mae_linear = mean_absolute_error(y_test, linear.predict(X_test))

print("MAE trên tập kiểm tra (tỷ đồng)")
print(f"  Rừng 200 cây: {mae_forest:.4f}")
print(f"  Cây sâu 3   : {mae_tree:.4f}")
print(f"  Đường thẳng : {mae_linear:.4f}")

house_250 = pd.DataFrame({"dien_tich": [250.0]})
prediction_250 = forest.predict(house_250)[0]
print(f"\nDự đoán của rừng cho căn 250 m2: {prediction_250:.4f} tỷ đồng")
print("Rừng không ngoại suy được. Mỗi cây đưa căn 250 m2 tới một lá cuối và dự đoán bằng")
print("trung bình giá các căn tập học trong lá đó; rừng chỉ lấy trung bình các dự đoán của 200 cây.")
print("Vì vậy dự đoán vẫn bị giới hạn bởi các mức giá đã xuất hiện trong tập học.")
