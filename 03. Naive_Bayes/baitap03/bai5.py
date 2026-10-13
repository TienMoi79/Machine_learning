"""Bài 5: so sánh MultinomialNB và GaussianNB trên dữ liệu liên tục."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

multinomial = MultinomialNB(alpha=1.0).fit(X_train, y_train)
gaussian = GaussianNB().fit(X_train, y_train)

print(f"Accuracy MultinomialNB: {accuracy_score(y_test, multinomial.predict(X_test)):.4f}")
print(f"Accuracy GaussianNB   : {accuracy_score(y_test, gaussian.predict(X_test)):.4f}")
print("\nGaussianNB phù hợp hơn vì giờ ôn và điểm giữa kỳ là biến liên tục.")
print("MultinomialNB được thiết kế cho số đếm/rời rạc, nên giả định của nó không phù hợp ở đây.")
