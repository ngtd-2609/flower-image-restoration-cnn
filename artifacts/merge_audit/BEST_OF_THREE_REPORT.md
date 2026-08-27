# Best-of-three merge report

## Quyết định

Version B được chọn cho kiến trúc `src/`, test, validator và Streamlit standalone. Version C cung cấp ngôn ngữ trình bày, cấu trúc phân công và bố cục Office tốt nhất. Version A chỉ cung cấp taxonomy phân tích lỗi và một số ý tưởng trực quan; toàn bộ metric/result/model surrogate bị loại.

## Xung đột đã loại

- Không giữ backend/frontend/FastAPI và `docker-compose.yml` của Version C.
- Không sử dụng `surrogate_classifier.joblib`, metric hoặc kết luận sinh từ surrogate của Version A.
- Không chọn metric cao nhất giữa archive; final results chỉ được sinh bởi checkpoint Keras, split và protocol canonical.
- Thông tin thành viên cũ/placeholder bị thay bằng ba thành viên canonical.

## Nguồn final theo component

| Component | A | B | C | Final source | Improvement |
|---|---:|---:|---:|---|---|
| Scientific design | 6 | 9 | 8 | B + Final | Khóa protocol và validator |
| Data | 5 | 8 | 7 | Final | Audit lại raw dataset thực tế |
| Code | 4 | 9 | 7 | B + Final | Sửa loader, metadata, latency tie-break |
| Model | 0 | 5 | 4 | Final run | MobileNetV2 `.keras` thật; surrogate bị loại |
| Results | 2 | 4 | 3 | Final run | 49 conditions + paired statistics |
| Notebook | 5 | 7 | 6 | B + Final | Luồng canonical, execution evidence |
| Tests | 2 | 8 | 5 | B + Final | Thêm semantic/coherence gates |
| Streamlit | 3 | 9 | 6 | B | Standalone, batch inference |
| Word | 7 | 8 | 9 | C + B + Final | Bố cục C, nội dung B, số liệu canonical |
| PPT | 7 | 8 | 9 | C + Final | Bố cục C, nguồn/metric canonical |
| Excel | 3 | 7 | 8 | C + Final | Sửa member count và allocation 100% |
| README | 4 | 9 | 9 | B + C + Final | Tree/status/result đồng bộ |
| Coherence | 3 | 8 | 5 | Final | Registry + visible-text audit |

Chi tiết định lượng nằm trong `COMPONENT_SCORECARD.csv`, `VERSION_SCORING.md` và `MERGE_DECISION_MATRIX.csv`.
