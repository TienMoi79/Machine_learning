"""Bài 4: tự cài đặt Multinomial Naive Bayes và so sánh sklearn."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "tin_nhan.csv")
texts_train, texts_test, y_train, y_test = train_test_split(
    df["noi_dung"], df["la_rac"], test_size=0.25, random_state=17,
    stratify=df["la_rac"]
)

# Chỉ học từ điển trên tập huấn luyện để tránh rò rỉ dữ liệu.
vectorizer = CountVectorizer(token_pattern=r"(?u)\S+")
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)
vocabulary = vectorizer.get_feature_names_out()
word_to_index = {word: index for index, word in enumerate(vocabulary)}


class ManualMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, texts, labels):
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        for label in self.classes_:
            class_texts = texts[labels == label]
            counts = Counter(word for text in class_texts for word in text.split())
            total = sum(counts[word] for word in vocabulary)
            denominator = total + self.alpha * len(vocabulary)
            self.class_log_prior_[label] = np.log(len(class_texts) / len(texts))
            self.feature_log_prob_[label] = np.array([
                np.log((counts[word] + self.alpha) / denominator)
                for word in vocabulary
            ])
        return self

    def predict(self, texts):
        predictions = []
        for text in texts:
            counts = Counter(text.split())
            x = np.zeros(len(vocabulary))
            for word, count in counts.items():
                if word in word_to_index:
                    x[word_to_index[word]] = count
            scores = {
                label: self.class_log_prior_[label] + x @ self.feature_log_prob_[label]
                for label in self.classes_
            }
            predictions.append(max(scores, key=scores.get))
        return np.asarray(predictions)


manual = ManualMultinomialNB(alpha=1.0).fit(
    texts_train.to_numpy(), y_train.to_numpy()
)
manual_prediction = manual.predict(texts_test.to_numpy())

sklearn_model = MultinomialNB(alpha=1.0).fit(X_train, y_train)
sklearn_prediction = sklearn_model.predict(X_test)

print(f"Số từ học từ tập train: {len(vocabulary)}")
print(f"Accuracy tự cài đặt: {accuracy_score(y_test, manual_prediction):.4f}")
print(f"Accuracy sklearn   : {accuracy_score(y_test, sklearn_prediction):.4f}")
print(f"Số dự đoán giống nhau: {(manual_prediction == sklearn_prediction).sum()}/{len(y_test)}")
print("\nDùng log giúp biến phép nhân nhiều xác suất nhỏ thành phép cộng,")
print("tránh hiện tượng số quá nhỏ bị làm tròn về 0 (underflow).")
