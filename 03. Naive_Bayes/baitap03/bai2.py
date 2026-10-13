"""Bài 2: tìm các từ đặc trưng nhất cho hai lớp."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "tin_nhan.csv")
vectorizer = CountVectorizer(token_pattern=r"(?u)\S+")
X = vectorizer.fit_transform(df["noi_dung"])
y = df["la_rac"]

model = MultinomialNB(alpha=1.0).fit(X, y)
words = vectorizer.get_feature_names_out()
log_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]

spam_indices = log_ratio.argsort()[::-1][:5]
normal_indices = log_ratio.argsort()[:5]

print("5 từ gợi ý thư rác mạnh nhất:")
for index in spam_indices:
    print(f"  {words[index]:<16} log-ratio = {log_ratio[index]:.4f}")

print("\n5 từ gợi ý tin thường mạnh nhất:")
for index in normal_indices:
    print(f"  {words[index]:<16} log-ratio = {log_ratio[index]:.4f}")

print("\nLog-ratio dương lớn nghiêng về lớp rác; âm lớn về độ lớn nghiêng về lớp thường.")
