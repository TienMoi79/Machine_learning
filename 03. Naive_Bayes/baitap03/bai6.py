"""Bài 6: đánh giá mô hình khi dữ liệu bị mất cân bằng."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "tin_nhan.csv")
normal = df[df["la_rac"] == 0]
spam = df[df["la_rac"] == 1].head(20)
imbalanced = pd.concat([normal, spam], ignore_index=True)

texts_train, texts_test, y_train, y_test = train_test_split(
    imbalanced["noi_dung"], imbalanced["la_rac"], test_size=0.25,
    random_state=17, stratify=imbalanced["la_rac"]
)
vectorizer = CountVectorizer(token_pattern=r"(?u)\S+")
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)
model = MultinomialNB(alpha=1.0).fit(X_train, y_train)
prediction = model.predict(X_test)

print("Phân bố dữ liệu mất cân bằng:")
print(imbalanced["la_rac"].value_counts().sort_index().to_string())
print("\nMa trận nhầm lẫn:")
print(confusion_matrix(y_test, prediction))
print(f"Accuracy      : {accuracy_score(y_test, prediction):.4f}")
print(f"Spam precision: {precision_score(y_test, prediction, zero_division=0):.4f}")
print(f"Spam recall   : {recall_score(y_test, prediction, zero_division=0):.4f}")
print("\nAccuracy có thể vẫn cao vì lớp thường chiếm đa số.")
print("Spam recall quan trọng hơn để biết mô hình đã bỏ sót bao nhiêu tin rác.")
