# Dataset placement

Không commit ảnh hoặc `flower_photos.rar` lên GitHub. Sau khi tải/giải nén bộ TensorFlow Flowers, cấu trúc cần có:

```text
data/flower_photos/
├── daisy/
├── dandelion/
├── roses/
├── sunflowers/
└── tulips/
```

Notebook kiểm tra toàn bộ kho ảnh, ghi 3.668 ảnh hợp lệ cùng 2 ảnh lỗi, SHA-256 duplicate và phân bố lớp trước khi huấn luyện.
