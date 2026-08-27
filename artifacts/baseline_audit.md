# Baseline audit và kế hoạch nâng cấp

## Phạm vi

Audit thực hiện trên bản sao của gói Tùng Dương; bản giải nén gốc được giữ nguyên. Gói không chứa thư mục
ảnh `data/flower_photos`, checkpoint Keras, kết quả phân loại cuối hoặc quyền GitHub/Community Cloud.

## Baseline trước migration

- 13 MB, không có Git metadata.
- Compile đạt; 8 test lõi đạt, 1 test ứng dụng skip.
- Notebook: 54 cell, 30 code cell, 17 phần, chưa thực thi.
- Split: Train 2.568, Validation 550, Test 550; 0 overlap path/hash.
- Metric chất lượng ảnh: 49 hàng; metric CNN chưa tồn tại.
- Kiến trúc triển khai cũ gồm hai dịch vụ và nhiều dependency/contract trùng lặp.
- Validator cũ trả mã thành công dù thiếu model và kết quả cuối.
- Word/PDF/PowerPoint/Excel còn mục chờ kết quả, placeholder và dấu vết kiến trúc cũ.

## Kiểm tra dữ liệu đã ghi nhận

- 3.668 ảnh hợp lệ; 2 ảnh truncated.
- Class count: 633 / 897 / 640 / 699 / 799.
- 3 nhóm exact duplicate; 1 nhóm cross-label.
- Audit phải được chạy lại từ ảnh gốc trước khi khóa bản nộp; không coi CSV bàn giao là thay thế dataset.

## Checksum baseline chính

| Artifact | SHA-256 |
|---|---|
| `splits/train.csv` | `0c54b4ad94470f1875fac16d06901cb04fa1f2aa3a2fde99d566811616184510` |
| `splits/validation.csv` | `d7d14c89b2fb0f63efed0aa15eb0b6e0135bc2d6f355679b1ad59193827d89e5` |
| `splits/test.csv` | `1ca5ed168c0c5ad5c10fbc5b7d239f61b92bd1baccac60d99d1569623c735247` |
| `results/data_validation.json` | `441ce3725da31b46d6f64b0fe8f0d3245c2f909087c02012034746cd31ce344d` |
| `results/image_quality_results_49_conditions.csv` | `ff789b186a3d39f34c59da071ba97677a8a9c4fa220312c70b6395559e2ce607` |

## P0/P1/P2

| Ưu tiên | Việc | Tiêu chí hoàn thành |
|---|---|---|
| P0 | Preprocessing, batch inference, locked params, 49-condition artifacts, Streamlit standalone, strict validator | Compile/test/core validator đạt; strict validator chỉ thiếu bằng chứng thực nghiệm/tài liệu cuối |
| P1 | Restore dataset, chạy `FULL_RUN`, sinh model/results/statistics/manifest | Checkpoint thật; 26.950 prediction rows; validator cuối đạt phần kỹ thuật |
| P2 | Điền thông tin nhóm, đồng bộ Word/PDF/PPTX/XLSX, deploy và chụp bằng chứng thật | Không placeholder/dấu vết cũ; render QA; URL incognito pass |

## Quy tắc chấm

Không chấm model, metric phân loại, deploy hoặc tính hoàn chỉnh ở mức tối đa khi thiếu bằng chứng thật.
