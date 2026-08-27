# Quality Report — checkpoint-ready package

Ngày kiểm tra: 2026-08-27

Trạng thái: `TRAINED_CHECKPOINT_READY — FULL_49_PENDING_USER_GPU — DEPLOY_PENDING_USER_ACTION`.

## Các gate đã đạt

- 25/25 unit và smoke tests PASS.
- Notebook 29 cell, 12 code cell; Restart/Run All PASS, không có error output.
- Notebook tải checkpoint TensorFlow thật và suy luận ảnh demo khi môi trường có TensorFlow.
- Core validator PASS: đủ file lõi, split 2.571/549/550, không overlap, 49 condition ID.
- Cross-artifact coherence PASS ở phạm vi checkpoint-ready.
- Checkpoint SHA-256 khớp metadata; model khoảng 21,8 MB.
- Office gồm DOCX/PDF/PPTX/XLSX byte-for-byte từ phiên bản B được chọn.
- PDF đọc được 43 trang, không mã hóa; PPTX/XLSX đã có structural inspection và spreadsheet formula-error scan không phát hiện lỗi.
- Smoke results được tách khỏi `results/final` để không bị dùng nhầm.

## Gate cố ý chưa chạy

- Full 49 trên toàn bộ 550 ảnh Test.
- `validate_project.py --require-full-run`.
- URL Streamlit public và deployment verification.

Ba gate này không được tuyên bố PASS. Hướng dẫn thao tác nằm trong `docs/RUN_FULL_49_AND_DEPLOY.md`.

## Lưu ý Office QA

Không thể render lại DOCX trong phiên đóng gói vì máy thiếu LibreOffice/`soffice`; file PDF gốc tương ứng đã được kiểm tra cấu trúc. Do yêu cầu là giữ nguyên bản tốt nhất, không thực hiện sửa nội dung Office.
