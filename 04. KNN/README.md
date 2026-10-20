# Bài 04 - K láng giềng gần nhất

Lab 04 trình bày khoảng cách Euclid, chuẩn hóa dữ liệu, lựa chọn số láng giềng,
so sánh mô hình và áp dụng KNN cho cả phân loại lẫn hồi quy.

## Hai phần chính

- `code/Lab4_LeQuyetTien.ipynb`: năm nội dung thực hành trước phần bài tập.
- `code/Baitap.ipynb`: lời giải đầy đủ sáu bài tập.

Các chương trình `.py` riêng nằm trong `code/` và `baitap04/`. Dữ liệu nằm trong
`data/` gồm:

- `sinh_vien_mo_rong.csv`: 120 sinh viên và một cột chi tiêu.
- `hai_vang_trang.csv`: 300 điểm thuộc hai lớp có biên cong.
- `gia_nha.csv`: 60 căn nhà dùng cho KNN hồi quy.

## Chạy trên Colab

Mount Google Drive, dùng `%cd` để đi đến thư mục `04. KNN`, mở notebook rồi chọn
**Runtime > Run all**. Giữ nguyên cấu trúc thư mục để các đường dẫn `data/...` hoạt động.

## Chạy bằng Python

```bash
python code/e1_khoang_cach.py
python baitap04/bai1.py
python scripts/run_all.py
```
