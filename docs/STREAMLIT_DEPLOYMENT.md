# Triển khai Streamlit standalone

## Kiến trúc chính thức

`streamlit_app.py` gọi trực tiếp `src/` theo luồng: đọc ảnh trong bộ nhớ → tạo suy giảm → áp dụng tham
số enhancement đã khóa trên Validation → chuẩn hóa thống nhất → một lần batch inference cho ảnh gốc,
suy giảm và sau xử lý. Ứng dụng không cần dịch vụ HTTP trung gian hoặc cấu hình liên dịch vụ.

## Chạy cục bộ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Trước khi chạy, phải có:

- `models/mobilenetv2_flowers.keras` từ notebook `FULL_RUN`;
- `configs/locked_enhancement_params.json` từ Validation tuning;
- `models/class_names.json` và `models/model_metadata.json` khớp checkpoint.

Nếu thiếu model/config, giao diện chỉ báo trạng thái và không tạo dự đoán thay thế.

## Docker một dịch vụ

```bash
docker build -t flower-restoration-streamlit .
docker run --rm -p 8501:8501 flower-restoration-streamlit
```

Health endpoint nội bộ của Streamlit: `http://localhost:8501/_stcore/health`.

## Streamlit Community Cloud

1. Push repository đã nghiệm thu lên GitHub của nhóm.
2. Chọn **Create app**, repository/branch tương ứng và entrypoint `streamlit_app.py`.
3. Dùng Python 3.11 và dependency tại `requirements.txt`.
4. Phân phối checkpoint bằng Git LFS hoặc release/storage có URL phiên bản cố định và checksum; không
   dùng URL tạm hoặc công khai secret.
5. Kiểm tra ở phiên ẩn danh: tải trang, model ready, upload hợp lệ, file lỗi, pipeline và reload.
6. Chỉ sau deploy thật mới ghi URL/screenshot vào `artifacts/deployment_verification.json` và
   `figures/deployment/`. Bản hiện tại không bịa các bằng chứng này.

## Checklist local smoke

```bash
python -m compileall -q src app_components streamlit_app.py
python -m unittest discover -s tests -v
python scripts/validate_project.py
streamlit run streamlit_app.py --server.headless=true
```
