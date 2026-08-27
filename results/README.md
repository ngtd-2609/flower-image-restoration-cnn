# Quy ước kết quả

- `image_quality_results_49_conditions.csv`: metric ảnh đã có; không phải Accuracy/Macro F1.
- `validation_tuning_all.csv`, `validation_tuning_best.csv`: chỉ sinh từ Validation sau model thật.
- `final/condition_metrics.csv`: 49 dòng metric ảnh + metric phân loại.
- `final/predictions.csv`: 49 × 550 dòng dự đoán theo ảnh ở `FULL_RUN`.
- `final/per_class_metrics.csv`: 49 × 5 dòng.
- `final/statistical_tests.csv`: bootstrap CI và McNemar theo cặp degraded/enhanced.
- `final/confusion_matrices/`: CSV cho 49 điều kiện, PNG cho điều kiện tiêu biểu.
- `final/manifest.json`: checksum và môi trường.

Không tạo các tệp `final/` bằng dữ liệu giả hoặc surrogate. Validator cuối phải thất bại khi thiếu chúng.
