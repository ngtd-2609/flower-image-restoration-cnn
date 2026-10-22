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

Inventory ghi nhận 3.670 ảnh hợp lệ và 0 ảnh hỏng. Xem `data/inventory.csv`, `results/data_validation.json` và `artifacts/data_audit.json` để đối chiếu kích thước, nhãn và SHA-256.
