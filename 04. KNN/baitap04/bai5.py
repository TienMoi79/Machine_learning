"""Bài 5: chọn k cho KNN hồi quy bằng kiểm định chéo."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv_mae = {}
for k in range(1, 21):
    scores = cross_val_score(
        KNeighborsRegressor(n_neighbors=k),
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    cv_mae[k] = -scores.mean()

best_k = min(cv_mae, key=cv_mae.get)
knn = KNeighborsRegressor(n_neighbors=best_k).fit(X_train, y_train)
linear = LinearRegression().fit(X_train, y_train)
mae_knn = mean_absolute_error(y_test, knn.predict(X_test))
mae_linear = mean_absolute_error(y_test, linear.predict(X_test))

print("MAE kiểm định chéo theo k:")
for k in range(1, 21):
    note = " <- tốt nhất" if k == best_k else ""
    print(f"  k = {k:2d}: {cv_mae[k]:.4f}{note}")

print(f"\nk tốt nhất = {best_k}")
print(f"MAE KNN trên tập kiểm tra       : {mae_knn:.4f}")
print(f"MAE đường thẳng trên tập kiểm tra: {mae_linear:.4f}")
if mae_knn < mae_linear:
    print("KNN vượt đường thẳng trên tập kiểm tra.")
else:
    print("KNN chưa vượt đường thẳng vì quan hệ giá và diện tích gần tuyến tính.")
