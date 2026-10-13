"""Bước 4: minh họa làm trơn Laplace cho Naive Bayes."""

from collections import Counter
from math import prod


corpus = [
    ("trung thuong ngay", 1),
    ("khuyen mai giam gia", 1),
    ("trung thuong click", 1),
    ("hop nhom chieu mai", 0),
    ("bai tap nhom nop mai", 0),
    ("thay gui tai lieu", 0),
]

vocabulary = sorted({word for text, _ in corpus for word in text.split()})
test_words = "trung thuong bai tap".split()
print(f"Số từ trong từ điển: {len(vocabulary)}")
print(f"Tin nhắn cần phân loại: {' '.join(test_words)!r}")


def class_score(label, alpha):
    documents = [text for text, target in corpus if target == label]
    counts = Counter(word for text in documents for word in text.split())
    total_words = sum(counts.values())
    prior = len(documents) / len(corpus)
    probabilities = [
        (counts[word] + alpha) / (total_words + alpha * len(vocabulary))
        for word in test_words
    ]
    return prior * prod(probabilities)


for alpha in [0, 1]:
    spam_score = class_score(1, alpha)
    normal_score = class_score(0, alpha)
    prediction = "rác" if spam_score > normal_score else "thường"
    print(f"\nalpha = {alpha}")
    print(f"Điểm lớp rác   : {spam_score:.10f}")
    print(f"Điểm lớp thường: {normal_score:.10f}")
    print(f"Dự đoán: {prediction}")

print("\nKhi alpha = 0, một từ chưa từng xuất hiện làm cả tích xác suất bằng 0.")
print("Làm trơn Laplace (alpha = 1) giúp mọi từ có xác suất khác 0.")
