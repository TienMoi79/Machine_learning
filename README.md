# Machine Learning — Học Máy Và Ứng Dụng

Repository lưu mã nguồn, dữ liệu, notebook, biểu đồ và kết quả các bài thực hành môn **Học Máy Và Ứng Dụng** tại Trường Đại học Văn Lang.

## Thông tin sinh viên

- **Họ và tên:** Lê Quyết Tiến
- **MSSV:** 2474802010386
- **Môn:** Học Máy Và Ứng Dụng
- **LHP:** 261_71ITAI41203_0101
- **GVHD:** ThS. Nguyễn Thái Anh ([GitHub](https://github.com/AnhNguyenVLU))

## Danh sách bài thực hành

| Bài | Nội dung | Tài liệu và mã nguồn |
|:---:|---|---|
| 01 | Hồi quy tuyến tính: đọc và phân tích dữ liệu giá nhà, bình phương tối thiểu, đánh giá mô hình, gradient descent và hồi quy nhiều biến | [01. Linear_Regression](01.%20Linear_Regression/) |
| 02 | Hồi quy logistic: sigmoid, phân loại nhị phân, ma trận nhầm lẫn, các thước đo đánh giá, ngưỡng quyết định và mô hình nhiều biến | [02. Logistic_Regression](02.%20Logistic_Regression/) |
| 03 | Naive Bayes: định lý Bayes, mô hình túi từ, lọc tin nhắn rác, làm trơn Laplace và Gaussian Naive Bayes | [03. Naive_Bayes](03.%20Naive_Bayes/) |
| 04 | K láng giềng gần nhất: khoảng cách, chuẩn hóa, chọn k bằng kiểm định chéo, phân loại và hồi quy KNN | [04. KNN](04.%20KNN/) |
| 05 | Cây quyết định và Rừng ngẫu nhiên: Gini, cắt tỉa cây, Random Forest, độ quan trọng đặc trưng và hồi quy | [05. DecisionTree_RandomForest](05.%20DecisionTree_RandomForest/) |

Các bài tiếp theo sẽ được bổ sung lần lượt theo tiến độ môn học.

## Cấu trúc repository

```text
Machine_learning/
├── 01. Linear_Regression/
│   ├── code/                 # Notebook thực hành và bài tập
│   ├── data/                 # Dữ liệu đầu vào
│   ├── figures/              # Biểu đồ được tạo khi chạy bài
│   ├── outputs/              # Kết quả chạy dạng văn bản
│   ├── scripts/              # Script chạy toàn bộ notebook
│   ├── requirements.txt      # Danh sách thư viện Python
│   └── README.md             # Hướng dẫn riêng của bài 01
├── 02. Logistic_Regression/
│   ├── code/                 # Notebook thực hành và bài tập
│   ├── data/                 # Dữ liệu sinh viên
│   ├── figures/              # Biểu đồ sigmoid
│   ├── outputs/              # Kết quả chạy dạng văn bản
│   ├── scripts/              # Script chạy toàn bộ notebook
│   ├── requirements.txt      # Danh sách thư viện Python
│   └── README.md             # Hướng dẫn riêng của bài 02
├── 03. Naive_Bayes/
│   ├── baitap03/             # Sáu bài tập Python độc lập
│   ├── code/                 # Notebook và năm phần thực hành
│   ├── data/                 # Dữ liệu tin nhắn và sinh viên
│   ├── figures/              # Hình ảnh phát sinh khi chạy bài
│   ├── outputs/              # Kết quả chạy dạng văn bản
│   ├── scripts/              # Script chạy toàn bộ chương trình
│   ├── requirements.txt      # Danh sách thư viện Python
│   └── README.md             # Hướng dẫn riêng của bài 03
├── 04. KNN/
│   ├── baitap04/             # Sáu bài tập Python độc lập
│   ├── code/                 # Notebook và năm phần thực hành
│   ├── data/                 # Ba bộ dữ liệu của bài KNN
│   ├── figures/              # Hình ảnh phát sinh khi chạy bài
│   ├── outputs/              # Kết quả chạy dạng văn bản
│   ├── scripts/              # Script chạy toàn bộ chương trình
│   ├── requirements.txt      # Danh sách thư viện Python
│   └── README.md             # Hướng dẫn riêng của bài 04
├── 05. DecisionTree_RandomForest/
│   ├── baitap05/             # Sáu bài tập Python độc lập
│   ├── code/                 # Notebook và bảy phần thực hành
│   ├── data/                 # Dữ liệu khách vay và giá nhà
│   ├── figures/              # Hình cây quyết định được tạo khi chạy bài
│   ├── outputs/              # Kết quả chạy dạng văn bản
│   ├── scripts/              # Script chạy toàn bộ chương trình
│   ├── requirements.txt      # Danh sách thư viện Python
│   └── README.md             # Hướng dẫn riêng của bài 05
├── .gitignore
└── README.md
```

Mỗi bài thực hành được đặt trong một thư mục riêng theo định dạng `STT. Ten_Bai`. Các thư mục bài mới sẽ được thêm vào repository khi môn học triển khai đến nội dung tương ứng.

## Tải và chạy bài thực hành

Clone repository:

```bash
git clone https://github.com/TienMoi79/Machine_learning.git
cd Machine_learning
```

Ví dụ với Bài 01:

```bash
cd "01. Linear_Regression"
python -m pip install -r requirements.txt
python scripts/run_all.py
```

Bạn cũng có thể mở các tệp `.ipynb` trong thư mục `code/` bằng **Google Colab**, **Jupyter Notebook** hoặc **Visual Studio Code** rồi chọn **Run all**.

> Khi tải hoặc di chuyển bài thực hành, hãy giữ nguyên cấu trúc các thư mục `code/`, `data/`, `figures/`, `outputs/` và `scripts/` để đường dẫn dữ liệu hoạt động chính xác.

## Công nghệ sử dụng

- Python 3
- NumPy và pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook / Google Colab
