# Model Card — MobileNetV2 nhận diện năm lớp hoa

## Tổng quan

Model mục tiêu là một MobileNetV2 duy nhất dùng cho clean, degraded và enhanced. Không fine-tune theo
từng điều kiện, không ensemble và không có surrogate fallback.

## Trạng thái

**FULL_RUN_TRAINED — EVALUATION_PENDING.** Checkpoint `models/mobilenetv2_flowers.keras` đã huấn luyện
đủ 15 epoch head + 10 epoch fine-tune, có SHA-256 và đã load/suy luận trong process mới. Các metric
full 49, latency và thống kê vẫn để trống tới khi người dùng chạy GPU và validator full PASS.

## Kiến trúc

- Backbone: MobileNetV2 pretrained ImageNet, `include_top=False`.
- Pooling: Global Average Pooling.
- Regularization: Dropout.
- Output: Dense softmax 5 lớp.
- Class order: daisy, dandelion, roses, sunflowers, tulips.
- Input shape: 224×224×3.

## Input contract

```text
EXIF transpose → RGB uint8 → letterbox LANCZOS 224×224
→ float32 0..255 → mobilenet_v2.preprocess_input trong graph
```

Đưa `preprocess_input` vào graph ngăn train/notebook/Streamlit dùng sai scale. `ModelService` phải load
đúng một checkpoint, xác minh class order và cung cấp `predict_batch`.

## Huấn luyện

### Giai đoạn 1 — classifier head

- Backbone frozen.
- Learning rate 1e-3.
- Tối đa 15 epoch ở FULL_RUN.
- Theo dõi `val_loss`.

### Giai đoạn 2 — fine-tune

- Mở 30 lớp cuối.
- BatchNormalization vẫn frozen.
- Learning rate 1e-5.
- Tối đa 10 epoch ở FULL_RUN.

EarlyStopping, ReduceLROnPlateau và ModelCheckpoint cùng giám sát Validation. Checkpoint tốt nhất được
giữ; không chọn epoch dựa trên Test.

## Augmentation

Chỉ áp dụng cho Train và chỉ gồm biến đổi hình học phù hợp: lật, xoay, zoom và tịnh tiến. Degradation
nghiên cứu không được trộn vào huấn luyện clean baseline vì sẽ thay đổi câu hỏi nghiên cứu.

## Khóa enhancement

Sau khi model clean cố định, grid tuning chạy trên Validation. Macro F1 là tiêu chí chính, SSIM phá hòa.
33 lựa chọn được ghi vào `configs/locked_enhancement_params.json` kèm split checksum, seed, thời gian và
cờ run mode. Test chỉ chạy sau khi khóa.

## Evaluation contract

- Test size: 550.
- Conditions: 49.
- Prediction rows: 26.950.
- Per-class rows: 245.
- Metrics: Accuracy, Macro Precision/Recall/F1, Weighted F1.
- Image metrics: PSNR, SSIM, Delta E 2000, brightness, contrast, edge preservation, histogram distance.
- Statistics: paired bootstrap CI cho Accuracy/Macro F1; exact McNemar; Holm correction trên 33 cặp.
- Latency: mean, median, p95 qua nhiều batch runs.

## Kết quả

Chưa có kết quả full 49 hợp lệ trong gói hiện tại. Không dùng số smoke. Sau full evaluation, phần này phải
được sinh từ `results/final/condition_metrics.csv`, `per_class_metrics.csv` và
`statistical_tests.csv`, đồng thời đồng bộ với Word/PPTX.

## Kiểm tra trước khi phát hành

- Model load được trong process Python mới.
- `model_sha256` trong metadata khớp tệp.
- Class order khớp `models/class_names.json`.
- Output có đúng năm xác suất hữu hạn, tổng xấp xỉ 1.
- Predictions có 550 dòng cho mỗi condition ID.
- Manifest checksum bao phủ model, split, bảng và confusion matrices.
- Streamlit AppTest/local smoke dùng chính checksum model cuối.
- Docker smoke PASS nếu Docker khả dụng.

## Mục đích sử dụng

- Nghiên cứu robustness có kiểm soát trên năm lớp hoa.
- Minh họa tác động của xử lý ảnh cổ điển tới classifier cố định.
- Demo học thuật qua Streamlit sau khi có artifact thật.

## Ngoài phạm vi

- Open-set recognition.
- Nhận diện loài hoa ngoài năm lớp.
- Ảnh đa nhãn, detection hoặc segmentation.
- Dự đoán có rủi ro cao.
- Khẳng định khả năng tổng quát sang suy giảm đời thực chưa kiểm thử.

## Giới hạn và rủi ro

Dataset nhỏ, mất cân bằng vừa, có label conflict, suy giảm tổng hợp và padding letterbox. Softmax
confidence chưa calibration. Enhancement có thể cải thiện metric ảnh nhưng làm xấu phân loại, hoặc
ngược lại. Mọi kết luận phải kèm CI/kiểm định và giới hạn family-wise inference.

## Giám sát và tái lập

Mỗi lần huấn luyện phải lưu environment, history, learning curves, split checksum, model checksum và
run ID. Thay đổi dữ liệu, split, preprocessing, class order hoặc checkpoint tạo ra một phiên bản model
mới; không được ghép metric từ nhiều phiên chạy.
