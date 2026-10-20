"""Bài 6: đo thời gian fit và predict của KNN khi dữ liệu tăng."""

from time import perf_counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


rng = np.random.default_rng(0)
X_new = rng.random((1000, 2))

print(f"{'Số mẫu học':>12} {'fit (giây)':>12} {'predict 1000 TB (giây)':>25}")
for n_samples in [1000, 10000, 100000]:
    X_train = rng.random((n_samples, 2))
    y_train = rng.integers(0, 2, size=n_samples)
    # Dùng tìm kiếm vét cạn để quan sát rõ chi phí so sánh với toàn bộ tập học.
    model = KNeighborsClassifier(n_neighbors=5, algorithm="brute", n_jobs=-1)

    start = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - start

    # Khởi động thư viện tính toán trước khi đo để lần đầu không tính thời gian nạp hệ thống.
    model.predict(X_new[:10])
    start = perf_counter()
    for _ in range(5):
        model.predict(X_new)
    predict_time = (perf_counter() - start) / 5

    print(f"{n_samples:12d} {fit_time:12.6f} {predict_time:25.6f}")

print("\nKNN được gọi là học lười vì fit chủ yếu chỉ lưu dữ liệu nên rất nhanh.")
print("Khi predict, mô hình phải tìm hàng xóm trong tập học; dữ liệu càng lớn thì dự đoán càng lâu.")
print("Hai cỡ nhỏ có thể gần nhau do chi phí khởi động; cỡ 100000 cho thấy chi phí dự đoán tăng rõ.")
print("Thời gian cụ thể có thể khác nhau giữa các máy, cần so sánh xu hướng tương đối.")
