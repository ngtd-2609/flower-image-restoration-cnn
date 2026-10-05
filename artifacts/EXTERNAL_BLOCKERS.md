# Phần còn cần xác minh trên bản deploy

Đã có URL do chủ dự án xác nhận: [Streamlit demo](https://flower-image-restoration-cnn-td.streamlit.app/). Ngày 05/10/2026, kiểm tra HTTP qua phiên có cookie nhận 200 và HTML Streamlit. Không còn chặn vì thiếu URL hoặc chưa có hosting.

Chưa xác minh: commit thực sự đang chạy, hash checkpoint trên cloud, ảnh chụp phiên ẩn danh, upload và dự đoán ảnh đơn/lô. Các mục này là giới hạn của bằng chứng kiểm tra, không có nghĩa ứng dụng chưa deploy.

Xem deployment_verification.json. Metadata thực nghiệm tháng 08 và báo cáo trước khi URL được cung cấp giữ trạng thái lịch sử; không dùng chúng để kết luận trạng thái deploy hiện tại.
