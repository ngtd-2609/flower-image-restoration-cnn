# Ứng dụng Streamlit

Demo: [flower-image-restoration-cnn-td.streamlit.app](https://flower-image-restoration-cnn-td.streamlit.app/).

Chạy cục bộ từ thư mục gốc bằng `streamlit run streamlit_app.py` sau khi cài dependency. Ứng dụng nạp checkpoint trực tiếp, hỗ trợ một ảnh hoặc batch tối đa 20 ảnh JPEG/PNG.

Repo chứa các tệp cần cho triển khai: `streamlit_app.py`, `requirements.txt`, `.python-version`, `models/`, `configs/locked_enhancement_params.json` và `results/final/manifest.json`. Raw dataset không cần cho suy luận.

Hồ sơ tại `artifacts/deployment_verification.json` ghi nhận trang trả HTTP 200 ngày 05/10/2026. Hồ sơ chưa xác nhận commit đang chạy, checksum model hoặc suy luận một ảnh và batch trên cloud.

Khi kiểm thử bản triển khai, đối chiếu kết quả một ảnh và batch với bản cục bộ dùng cùng checkpoint; kiểm tra nhãn, xác suất và ảnh sau xử lý.
