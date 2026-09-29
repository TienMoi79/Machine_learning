# Bài 02 - Hồi quy logistic

Thực hành phân loại nhị phân trên dữ liệu sinh viên, trong đó nhãn `1` là qua môn và nhãn `0` là rớt môn.

## Nội dung

- Đọc và phân tích dữ liệu sinh viên.
- Tự cài đặt hàm sigmoid.
- Khớp mô hình `LogisticRegression`.
- Đánh giá bằng confusion matrix, accuracy, precision, recall và F1.
- Khảo sát ảnh hưởng của ngưỡng quyết định.
- Mở rộng từ một biến sang hai biến đầu vào.

## Cấu trúc

- `code/Lab2_LeQuyetTien.ipynb`: sáu bước thực hành.
- `code/Baitap.ipynb`: lời giải sáu bài tập.
- `data/sinh_vien.csv`: dữ liệu gồm 120 sinh viên.
- `figures/sigmoid.png`: đồ thị sigmoid của Bài 2.
- `outputs/`: kết quả chạy dạng văn bản.
- `scripts/run_all.py`: chạy lại hai notebook.

## Chạy bài

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
```

Khi dùng Google Colab, hãy giữ nguyên cấu trúc thư mục và chuyển thư mục làm việc đến `02. Logistic_Regression` trước khi chọn **Run all**.
