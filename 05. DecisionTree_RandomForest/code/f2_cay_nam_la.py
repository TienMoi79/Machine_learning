"""Bước 2: học một cây nhỏ năm lá trên dữ liệu khách vay rồi đọc nó."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "khach_vay.csv")
print("Kích thước bảng:", df.shape)
print(df.head())
print(f"\nTỷ lệ khách trả đúng hạn: {df['tra_dung_han'].mean():.2f}\n")

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

tree = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X_train, y_train)
print(f"Độ sâu {tree.get_depth()}, số lá {tree.get_n_leaves()}")
print(export_text(tree, feature_names=list(X.columns), show_weights=True, decimals=3))
print(f"Accuracy tập học     : {accuracy_score(y_train, tree.predict(X_train)):.4f}")
print(f"Accuracy tập kiểm tra: {accuracy_score(y_test, tree.predict(X_test)):.4f}")

new_customers = pd.DataFrame(
    {
        "thu_nhap": [20.0, 10.0],
        "ty_le_no": [0.4, 0.3],
        "co_viec_on_dinh": [1, 0],
        "so_nguoi_phu_thuoc": [2, 1],
    }
)
predictions = tree.predict(new_customers)
probabilities = tree.predict_proba(new_customers)[:, 1]
print()
for i, (prediction, probability) in enumerate(zip(predictions, probabilities), start=1):
    result = "đúng hạn" if prediction == 1 else "trễ hạn"
    print(f"Khách {i}: {result}, xác suất đúng hạn {probability:.2f}")
