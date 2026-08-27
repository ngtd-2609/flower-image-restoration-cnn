# Evidence artifacts

Thư mục này chỉ chứa bằng chứng sinh từ một lần chạy hoặc một lần nghiệm thu thật. Không tạo file rỗng
để vượt validator.

- `data_audit.json`: audit raw data.
- `environment.json`: Python, TensorFlow, thiết bị và seed của FULL_RUN.
- `training/history.csv`: history hai giai đoạn.
- `training/learning_curves.png`: biểu đồ sinh từ history.
- `app_smoke_test.json`: trạng thái local/AppTest với checksum model cuối.
- `deployment_verification.json`: URL, commit, checksum, timestamp và screenshot deploy thật.

Nếu chưa có quyền deploy, giữ `NOT_SUBMISSION_READY` và không tạo URL mẫu.
