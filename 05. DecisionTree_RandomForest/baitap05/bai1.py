"""Bài 1: vẽ cây năm lá và cây sâu tối đa 2."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


ROOT = Path.cwd()
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

tree_five_leaves = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(
    X_train, y_train
)
plt.figure(figsize=(18, 9))
plot_tree(
    tree_five_leaves,
    feature_names=list(X.columns),
    class_names=["trễ hạn", "đúng hạn"],
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Cây quyết định có tối đa 5 lá")
plt.tight_layout()
plt.savefig(FIGURES / "cay_nam_la.png", dpi=160, bbox_inches="tight")
plt.close()

tree_depth_2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
plt.figure(figsize=(16, 8))
plot_tree(
    tree_depth_2,
    feature_names=list(X.columns),
    class_names=["trễ hạn", "đúng hạn"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Cây quyết định sâu tối đa 2")
plt.tight_layout()
plt.savefig(FIGURES / "cay_sau_2.png", dpi=160, bbox_inches="tight")
plt.close()

print("Đã lưu: figures/cay_nam_la.png")
print("Đã lưu: figures/cay_sau_2.png")
print("\nCây sâu 2:")
print(export_text(tree_depth_2, feature_names=list(X.columns), show_weights=True, decimals=3))
print("Nhận xét: hai lá bên trái đều dự đoán trễ hạn (lớp 0).")
print("Cây vẫn chia vì hai nhóm có tỷ lệ đúng/trễ hạn và độ thuần khác nhau;")
print("phép chia làm giảm Gini dù lớp chiếm đa số của cả hai lá vẫn giống nhau.")
