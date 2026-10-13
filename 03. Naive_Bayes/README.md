# Bài 03 - Naive Bayes

Lab 03 trình bày định lý Bayes, biểu diễn túi từ, bộ lọc tin nhắn rác bằng
Multinomial Naive Bayes, làm trơn Laplace và Gaussian Naive Bayes.

## Hai phần chính

- `code/Lab3_LeQuyetTien.ipynb`: năm nội dung thực hành trước phần bài tập.
- `code/Baitap.ipynb`: lời giải đầy đủ sáu bài tập.

Các chương trình `.py` riêng được giữ trong `code/` và `baitap03/` theo cấu trúc
của tài liệu. Hai tệp dữ liệu nằm trong `data/`:

- `tin_nhan.csv`: 300 tin nhắn, dùng cho bài toán lọc tin nhắn rác.
- `sinh_vien.csv`: 120 sinh viên, dùng cho Gaussian Naive Bayes.

## Chạy trên Colab

Mount Google Drive, dùng `%cd` để đi đến thư mục `03. Naive_Bayes`, sau đó mở
notebook và chọn **Runtime > Run all**. Notebook đã chứa sẵn kết quả mẫu.

## Chạy bằng Python

```bash
python code/d1_bayes_tay.py
python baitap03/bai1.py
python scripts/run_all.py
```

Mọi lệnh phải chạy từ thư mục gốc của bài để các đường dẫn `data/...` hoạt động.
