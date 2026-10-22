# Kết quả thực nghiệm

Kết quả đánh giá 49 điều kiện trên 550 ảnh Test với một checkpoint MobileNetV2 cố định:

- `condition_metrics.csv`: 49 dòng, gồm metric phân loại và chất lượng ảnh.
- `predictions.csv`: 26.950 dự đoán theo ảnh và điều kiện.
- `per_class_metrics.csv`: 245 dòng metric từng lớp.
- `statistical_tests.csv`: 66 dòng so sánh cặp, gồm bootstrap và McNemar hiệu chỉnh Holm.
- `confusion_matrices/`: 49 ma trận CSV và các hình tiêu biểu.
- `error_analysis.csv`, `top_confusion_pairs.csv`: phân tích lỗi và cặp lớp nhầm.
- `manifest.json`: checksum đối chiếu model, split và bảng kết quả.

Tái chạy bằng `python scripts/run_full_pipeline.py --skip-train` sau khi chuẩn bị raw dataset. Tham số phục hồi được chọn trên Validation, không chọn trên Test.
