# Thiết kế thực nghiệm và hợp đồng dữ liệu

## Câu hỏi nghiên cứu

- RQ1: Mỗi loại/mức suy giảm làm giảm chất lượng ảnh và Macro F1 bao nhiêu so với clean?
- RQ2: Phương pháp tiền xử lý nào phục hồi Macro F1 tốt nhất cho từng suy giảm?
- RQ3: PSNR, SSIM và ΔE2000 liên hệ thế nào với Macro F1?
- RQ4: Lớp hoa nào nhạy cảm nhất và dấu hiệu thị giác nào bị phá hủy?
- RQ5: Enhancement có trường hợp làm hại dự đoán dù metric ảnh tăng không?

## Biến

- Độc lập: loại suy giảm, mức độ, phương pháp và tham số enhancement.
- Phụ thuộc: PSNR, SSIM, ΔE2000, brightness, contrast, edge preservation, Accuracy, Macro F1,
  per-class metric, confidence, thời gian.
- Kiểm soát: split, seed, checkpoint, class order, resize, preprocess, Test set.

## Quy tắc chống leakage

1. Quét dữ liệu và SHA-256 trước chia.
2. Group duplicate theo SHA-256.
3. Chia 14/3/3 fold xấp xỉ 70/15/15, seed 42.
4. Train chỉ ảnh sạch và augmentation hình học.
5. Tuning enhancement chỉ trên Validation.
6. Khóa JSON tham số rồi mới mở Test.
7. Cùng degraded array là input cho mọi method đối chứng trong điều kiện.
8. Model không được huấn luyện lại sau khi xem Test.

## Khóa điều kiện 49 dòng

Khóa duy nhất là `(image_type, degradation, level, enhancement_method)`. Clean dùng `none`; degraded
method `none`; enhanced dùng một trong mapping. Script assert đúng 49 dòng và không trùng khóa.

## Metric

PSNR/SSIM/ΔE đo khôi phục ảnh so với clean. Macro F1 là metric nhận diện chính; weighted F1 chỉ bổ sung.
Khoảng tin cậy bootstrap dùng resampling theo ảnh. McNemar dùng cặp đúng/sai trên cùng ảnh cho những cặp
degraded–enhanced quan trọng. Tương quan Pearson/Spearman không chứng minh nhân quả.

## Trạng thái run

`image_quality_results_49_conditions.csv` đã được chạy bằng pipeline ảnh trên mẫu Test. Các cột nhận diện
không được thêm giả. Sau `FULL_RUN`, `results/final/condition_metrics.csv` phải có 49 dòng cùng
`predictions.csv`, metric từng lớp, kiểm định thống kê và confusion matrices; validator sẽ fail nếu thiếu.

## Tái lập

Lưu model, history, class order, metadata, locked params, kết quả, figures và `run_manifest.json` trong
cùng một run. Khi cập nhật dữ liệu hay code, tạo run mới; không ghi đè số cũ mà không đổi manifest.
