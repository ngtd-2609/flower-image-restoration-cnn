# Chạy lại thực nghiệm và ứng dụng

## Chuẩn bị

Dùng Python 3.11 và cài dependency theo README. Đặt TensorFlow Flowers trong `data/flower_photos/`, gồm năm thư mục `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`. Đối chiếu ảnh với `data/inventory.csv` và giữ nguyên ba split tại `splits/`.

Checkpoint đã được lưu trong `models/mobilenetv2_flowers.keras`. Các kết quả đầy đủ hiện có trong `results/final/`.

## Chạy thực nghiệm

```bash
# Dùng checkpoint hiện tại, chạy tuning và đánh giá:
python scripts/run_full_pipeline.py --skip-train

# Hoặc huấn luyện lại từ đầu:
python scripts/run_full_pipeline.py --retrain
```

Nên chạy trên GPU cho thực nghiệm đầy đủ. Lệnh có thể ghi đè kết quả và cấu hình khóa; sao lưu lần chạy hiện tại trước khi thực hiện. Không dùng `--quick-run` để lấy số liệu báo cáo.

## Đối chiếu kết quả

```bash
python scripts/validate_project.py --require-full-run
python scripts/check_notebook.py --require-executed --require-full-run
python -m unittest discover -s tests -v
```

Một lần chạy đầy đủ có 49 điều kiện, 26.950 dự đoán, 245 dòng metric từng lớp và 66 dòng kiểm định. Manifest lưu checksum để phát hiện dữ liệu hoặc checkpoint không khớp.

## Ứng dụng

```bash
streamlit run streamlit_app.py
```

Mở địa chỉ do Streamlit in ra. Thử một ảnh và batch JPEG/PNG, đối chiếu nhãn, xác suất và ảnh phục hồi. Checkpoint, metadata và 33 cấu hình khóa phải khớp để ứng dụng cho phép suy luận.

Demo hiện tại: [flower-image-restoration-cnn-td.streamlit.app](https://flower-image-restoration-cnn-td.streamlit.app/).

Để cấu hình triển khai từ repo, xem mục ứng dụng trong README và sử dụng `streamlit_app.py` làm entrypoint. `requirements.txt`, `.python-version`, checkpoint và cấu hình đã được lưu cùng mã nguồn. Hồ sơ kiểm tra truy cập demo nằm tại `artifacts/deployment_verification.json`; hồ sơ này chưa xác nhận suy luận một ảnh hoặc batch trên cloud.
