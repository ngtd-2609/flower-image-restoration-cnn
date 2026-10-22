# Kiểm tra dự án

Các lệnh kiểm tra được chạy từ thư mục gốc sau khi cài `requirements-dev.txt`:

```bash
python -m unittest discover -s tests -v
ruff check src app_components streamlit_app.py scripts tests
python scripts/check_notebook.py --require-executed --require-full-run
python scripts/validate_project.py --require-full-run
```

Validator đối chiếu split, ma trận 49 điều kiện, số dòng dự đoán, checksum kết quả và khả năng nạp checkpoint trong process mới. Kết quả thực nghiệm được lưu tại `results/final/`; cấu hình phục hồi tại `configs/locked_enhancement_params.json`.

`--require-final` còn kiểm tra hồ sơ nộp và bằng chứng kiểm thử ứng dụng trên cloud. Hồ sơ HTTP tại `artifacts/deployment_verification.json` chỉ ghi nhận truy cập trang ngày 05/10/2026; chưa xác nhận suy luận một ảnh và batch trên cloud.
