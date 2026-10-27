# Bài 05 - Cây quyết định và Rừng ngẫu nhiên

Lab 05 trình bày chỉ số Gini, cây quyết định phân loại, kiểm soát quá khớp,
Random Forest, độ quan trọng của đặc trưng và cây/rừng cho bài toán hồi quy.

## Hai phần chính

- `code/Lab5_LeQuyetTien.ipynb`: bảy nội dung thực hành trước phần bài tập.
- `code/Baitap.ipynb`: lời giải đầy đủ sáu bài tập.

Các chương trình `.py` riêng nằm trong `code/` và `baitap05/`. Dữ liệu nằm trong
`data/` gồm:

- `khach_vay.csv`: 600 hồ sơ khách vay dùng cho bài toán phân loại.
- `gia_nha.csv`: 60 căn nhà dùng cho bài toán hồi quy.

Hai hình của Bài tập 1 được lưu trong `figures/`.

## Chạy trên Colab

Mount Google Drive, dùng `%cd` để đi đến thư mục `05. DecisionTree_RandomForest`,
mở notebook rồi chọn **Runtime > Run all**. Giữ nguyên cấu trúc thư mục để các đường
dẫn `data/...` hoạt động.

## Chạy bằng Python

```bash
python code/f1_gini.py
python baitap05/bai1.py
python scripts/run_all.py
```
