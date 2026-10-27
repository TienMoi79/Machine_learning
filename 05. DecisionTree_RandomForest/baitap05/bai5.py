"""Bài 5: đo độ quan trọng bằng cách xáo trộn."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

rng = np.random.default_rng(0)
X_train_random = X_train.copy()
X_test_random = X_test.copy()
X_train_random["so_ngau_nhien"] = rng.random(len(X_train_random))
X_test_random["so_ngau_nhien"] = rng.random(len(X_test_random))

forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(
    X_train_random, y_train
)

impurity_importance = pd.Series(
    forest.feature_importances_, index=X_train_random.columns
).sort_values(ascending=False)

permutation = permutation_importance(
    forest,
    X_test_random,
    y_test,
    n_repeats=20,
    random_state=0,
)
permutation_mean = pd.Series(
    permutation.importances_mean, index=X_test_random.columns
).sort_values(ascending=False)

print("Độ quan trọng do giảm Gini (feature_importances_):")
print(impurity_importance.round(4).to_string())
print("\nĐộ quan trọng xáo trộn trung bình trên tập kiểm tra:")
print(permutation_mean.round(4).to_string())

rank_impurity = impurity_importance.index.get_loc("so_ngau_nhien") + 1
rank_permutation = permutation_mean.index.get_loc("so_ngau_nhien") + 1
print(f"\nCột ngẫu nhiên xếp thứ {rank_impurity} theo giảm Gini.")
print(f"Cột ngẫu nhiên xếp thứ {rank_permutation} theo permutation importance.")
print("Giải thích: feature_importances_ thiên vị cột có nhiều giá trị vì cây có nhiều ngưỡng")
print("để thử và dễ tìm được phép chia tình cờ tốt trên tập học. Permutation importance đo trên")
print("tập kiểm tra: xáo trộn cột vô dụng hầu như không làm accuracy giảm, nên thứ hạng thấp hơn.")
print("Độ quan trọng chỉ cho biết mô hình sử dụng cột, không chứng minh quan hệ nhân quả.")
