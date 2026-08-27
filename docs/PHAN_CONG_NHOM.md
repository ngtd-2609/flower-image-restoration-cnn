# Phân công Nhóm 7

| MSSV | Thành viên | Vai trò chính | Tỷ trọng |
|---|---|---|---:|
| 24100358 | Nguyễn Tùng Dương | Trưởng nhóm; kiến trúc CNN; training/evaluation pipeline | 34% |
| 24100065 | Trịnh Ngọc Nga | Raw audit; split chống leakage; degradation/enhancement; thống kê | 33% |
| 24106898 | Trương Việt Thành | Streamlit single/batch; Office artifacts; QA và đóng gói | 33% |

Tổng tỷ trọng: **100%**. Workbook `ASSIGNMENT_FINAL.xlsx` dùng công thức `COUNTA(A10:A12)` và `SUM(D10:D12)` để kiểm soát hai chỉ tiêu này.

## Deliverables và evidence

| ID | Deliverable | Chủ trì | Review | Trạng thái/evidence |
|---|---|---|---|---|
| D01 | Inventory, duplicate audit, split | Nga | Dương | 3.670 ảnh hợp lệ; 0 ảnh lỗi; 0 path/hash overlap |
| D02 | MobileNetV2 hai giai đoạn | Dương | Nga | `.keras`, model metadata, history, learning curves |
| D03 | Validation tuning | Dương + Nga | Thành | 33 cấu hình khóa; Macro F1 → SSIM → latency |
| D04 | Full 49-condition Test | Dương | Nga | 49 rows; 26.950 predictions; 245 per-class; 66 statistics |
| D05 | Error analysis | Nga | Dương | Trace theo condition/path/SHA-256/confidence delta |
| D06 | Streamlit standalone | Thành | Dương | Single + batch 20 ảnh; readiness/checksum gate |
| D07 | Word/PDF/PPTX/XLSX | Thành | Cả nhóm | Sinh từ canonical facts và final CSV; render QA |
| D08 | Test, validator, checksum, ZIP | Cả nhóm | Cả nhóm | Clean-extract verification trước bàn giao |
| D09 | Public deployment | Thành | Dương | `DEPLOY_READY_BUT_NOT_DEPLOYED`; cần quyền tài khoản/URL |

## Quy tắc phối hợp

1. Không đổi split hoặc class order sau khi bắt đầu huấn luyện.
2. Không dùng Test để chọn model hay enhancement.
3. Không nhập metric thủ công vào tài liệu; mọi số liệu đọc từ `results/final`.
4. Không tạo URL, screenshot, checkpoint hoặc trạng thái deploy giả.
5. Mọi thay đổi preprocessing/model contract phải cập nhật checksum và chạy lại validator.
