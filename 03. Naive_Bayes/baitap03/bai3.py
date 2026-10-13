"""Bài 3: khảo sát ảnh hưởng của hệ số làm trơn alpha."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "tin_nhan.csv")
texts_train, texts_test, y_train, y_test = train_test_split(
    df["noi_dung"], df["la_rac"], test_size=0.25, random_state=17,
    stratify=df["la_rac"]
)
vectorizer = CountVectorizer(token_pattern=r"(?u)\S+")
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

print(f"{'alpha':>8} {'accuracy':>10} {'precision':>10} {'recall':>10}")
for alpha in [0.001, 0.1, 1, 10, 100]:
    model = MultinomialNB(alpha=alpha).fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(
        f"{alpha:>8g} {accuracy_score(y_test, prediction):>10.4f} "
        f"{precision_score(y_test, prediction):>10.4f} "
        f"{recall_score(y_test, prediction):>10.4f}"
    )

print("\nAlpha quá lớn kéo xác suất của các từ về gần nhau, làm giảm sức phân biệt của mô hình.")
