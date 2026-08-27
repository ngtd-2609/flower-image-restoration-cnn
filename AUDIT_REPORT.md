# Audit Report — best-of-three merge

## Quyết định hợp nhất

- Version B làm backbone kỹ thuật và cung cấp Office nguyên bản cuối vì nội dung/mục lục phù hợp Streamlit standalone.
- Version C đóng góp tiêu chí bố cục và kiểm tra chéo.
- Version A đóng góp taxonomy/ý tưởng trình bày; loại bỏ surrogate/fake-result.
- Bằng chứng inventory, scorecard, decision matrix và post-merge error matrix nằm tại `artifacts/merge_audit/`.

## Sự thật dữ liệu chuẩn

- 3.670 ảnh hợp lệ, 0 ảnh hỏng trong dữ liệu được cung cấp.
- Lớp: daisy 633, dandelion 898, roses 641, sunflowers 699, tulips 799.
- Split: train 2.571, validation 549, test 550.
- Không giao nhau theo relative path hoặc SHA-256.
- Có 3 nhóm duplicate và 1 cross-label duplicate được ghi nhận trong audit; split hash vẫn disjoint.

## CNN và thí nghiệm

- MobileNetV2 ImageNet, 224×224, 5 lớp.
- 15 epoch head + 10 epoch fine-tune đã hoàn tất.
- Trạng thái metadata: `FULL_RUN_TRAINED — EVALUATION_PENDING`.
- Tuning validation-only; Test chỉ dùng cho đánh giá cuối.
- Full matrix gồm đúng 49 condition ID.

## Trạng thái bàn giao

Checkpoint, source, notebook, tests, Streamlit và packaging đã hoàn tất. Full 49 và deploy là bước người dùng chủ động chạy sau trên GPU; không có metric smoke nào được trình bày như kết quả cuối.
