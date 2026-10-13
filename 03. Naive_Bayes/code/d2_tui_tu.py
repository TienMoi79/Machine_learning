"""Bước 2: biến văn bản thành mô hình túi từ."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "tin_nhan.csv")

print("=== Thông tin dữ liệu ===")
print(f"Kích thước: {df.shape}")
print("Số lượng từng lớp:")
print(df["la_rac"].value_counts().sort_index().to_string())
print(f"Tỷ lệ thư rác: {df['la_rac'].mean():.4f}")

vectorizer = CountVectorizer(token_pattern=r"(?u)\S+")
X = vectorizer.fit_transform(df["noi_dung"])
words = vectorizer.get_feature_names_out()

print("\n=== Túi từ ===")
print(f"Kích thước ma trận X: {X.shape}")
print(f"Số từ trong từ điển: {len(words)}")

for label, name in [(0, "thường"), (1, "rác")]:
    counts = X[df["la_rac"].to_numpy() == label].sum(axis=0).A1
    top = counts.argsort()[::-1][:10]
    print(f"\n10 từ xuất hiện nhiều nhất ở lớp {name}:")
    for index in top:
        print(f"  {words[index]:<16} {int(counts[index])}")
