"""Bước 3: huấn luyện bộ lọc tin nhắn rác bằng MultinomialNB."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "tin_nhan.csv")

vectorizer = CountVectorizer(token_pattern=r"(?u)\S+")
X = vectorizer.fit_transform(df["noi_dung"])
y = df["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== Kết quả trên tập kiểm tra ===")
print(f"Tập huấn luyện: {X_train.shape[0]} mẫu")
print(f"Tập kiểm tra:   {X_test.shape[0]} mẫu")
print("Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")

messages = [
    "trung thuong khuyen mai nhanh tay click ngay",
    "hop nhom bao cao chieu mai phong 305",
    "ban gui giup em tai lieu bai tap nhe",
]
probabilities = model.predict_proba(vectorizer.transform(messages))[:, 1]
print("\n=== Dự đoán tin nhắn mới ===")
for message, probability in zip(messages, probabilities):
    label = "rác" if probability >= 0.5 else "thường"
    print(f"{message!r}: P(rác) = {probability:.4f} -> {label}")
