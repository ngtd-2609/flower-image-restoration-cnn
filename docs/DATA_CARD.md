# Data Card — Flower Photos

## Mục đích

Bộ dữ liệu phục vụ nghiên cứu học thuật về phân loại năm lớp hoa dưới các suy giảm tổng hợp. Dataset
không được thiết kế cho nhận diện loài hoa ngoài năm lớp, ảnh đa nhãn, object detection hoặc quyết định
có tác động tới con người.

## Trạng thái bằng chứng

Raw dataset không được đưa vào ZIP để giảm dung lượng, nhưng đã được audit lại trong phiên nâng cấp.
Khi chạy full 49, phải khôi phục `data/flower_photos/` đúng bộ gốc và chạy lại audit tự động.

Nguồn inventory duy nhất: `data/inventory.csv`.

## Thành phần

| Lớp | Số ảnh hợp lệ | Tỷ lệ |
|---|---:|---:|
| daisy | 633 | 17,26% |
| dandelion | 898 | 24,47% |
| roses | 641 | 17,47% |
| sunflowers | 699 | 19,06% |
| tulips | 799 | 21,78% |
| **Tổng** | **3.670** | **100%** |

Hồ sơ ghi nhận 3.670 tệp nguồn, 3.670 ảnh giải mã hợp lệ và 0 ảnh hỏng.

## Schema inventory

| Trường | Ý nghĩa |
|---|---|
| `relative_path` | đường dẫn tương đối theo repository |
| `label` | một trong năm class name khóa |
| `width`, `height` | kích thước sau khi đọc metadata |
| `aspect_ratio` | width / height |
| `mode` | mode ảnh sau xác minh |
| `sha256` | checksum bytes, khóa duplicate group |
| `bytes` | dung lượng tệp |
| `decode_status` | trạng thái giải mã ghi nhận |
| `duplicate_group` | ID nhóm exact duplicate hoặc rỗng |

## Duplicate và xung đột nhãn

- Exact duplicate groups ghi nhận: 3.
- Cross-label duplicate groups ghi nhận: 1.
- Không tự động xóa duplicate vì có thể làm thay đổi phân bố.
- SHA-256 được dùng làm group khi split, bảo đảm mọi bản sao byte ở cùng một split.
- Cross-label duplicate là rủi ro label noise và phải được nêu khi diễn giải trần hiệu năng.

## Split khóa

| Split | Số ảnh | Tỷ lệ gần đúng | Mục đích |
|---|---:|---:|---|
| Train | 2.571 | 70% | học tham số model |
| Validation | 549 | 15% | early stopping và khóa enhancement |
| Test | 550 | 15% | đánh giá cuối một lần |

Split dùng grouped stratification, seed 42. Các cặp Train/Validation/Test phải có giao path = 0 và
giao SHA-256 = 0. Không tạo degradation hay augmentation trước split.

## Tiền xử lý

Ảnh được EXIF transpose, chuyển RGB uint8 và letterbox LANCZOS về 224×224. Quy tắc giữ tỷ lệ nhưng thêm
padding đen; đây là một biến có thể ảnh hưởng ảnh quá dẹt và được áp dụng giống nhau cho mọi điều kiện.

## EDA bắt buộc khi có raw data

- số ảnh theo lớp;
- width, height và aspect ratio;
- độ sáng, RMS contrast và edge density;
- RGB/HSV/CIELAB distribution;
- ảnh mẫu theo lớp;
- bad-file report;
- duplicate và cross-label duplicate report;
- checksum inventory và split.

## Sử dụng phù hợp

- So sánh có kiểm soát giữa clean, degraded và enhanced.
- Phân tích robustness của một classifier cố định.
- Thực hành pipeline tái lập, leakage prevention và paired statistics.

## Sử dụng không phù hợp

- Suy diễn hiệu năng cho mọi loài hoa hoặc ảnh ngoài phân phối.
- Dùng confidence softmax như xác suất đã calibration.
- Khẳng định hiệu quả trên suy giảm thực từ kết quả suy giảm tổng hợp.
- Chọn tham số trên Test.
- Công bố metric khi raw data/checkpoint/manifest chưa xác minh.

## Rủi ro và thiên lệch

Dữ liệu có mất cân bằng vừa, nguồn ảnh không đồng nhất, một nhóm duplicate khác nhãn và không có mô tả
đầy đủ về thiết bị chụp/điều kiện địa lý. Năm lớp có dấu hiệu màu và cấu trúc dễ trùng; nền ảnh có thể
trở thành shortcut. Kết quả chỉ có giá trị trong phạm vi dataset và protocol đã khóa.

## Quản trị phiên bản

Mỗi `FULL_RUN` phải ghi checksum `data/inventory.csv`, ba split CSV, model và các bảng kết quả vào
manifest. Nếu raw dataset thay đổi, phải tạo run mới; không được giữ metric cũ với inventory mới.

## Liên hệ và giấy phép

Thông tin người phụ trách và quyền sử dụng dataset phải được nhóm bổ sung từ nguồn phân phối thực tế.
Không suy diễn quyền tái phân phối raw images từ giấy phép mã nguồn của repository.
