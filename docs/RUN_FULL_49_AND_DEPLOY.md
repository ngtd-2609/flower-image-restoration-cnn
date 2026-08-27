# Hướng dẫn chi tiết: chạy full 49 bằng GPU và deploy Streamlit

Tài liệu này dành cho người thao tác lần đầu. Làm lần lượt từ Bước 0 đến Bước 15. Không cần huấn luyện lại CNN vì checkpoint MobileNetV2 đã có trong `models/mobilenetv2_flowers.keras`.

## Bước 0 — Hiểu trạng thái hiện tại

Bản dự án đang ở trạng thái:

```text
TRAINED_CHECKPOINT_READY
FULL_49_PENDING_USER_GPU
DEPLOY_PENDING_USER_ACTION
```

Đã hoàn thành:

- audit 3.670 ảnh và split 2.571/549/550;
- huấn luyện MobileNetV2 15 epoch head + 10 epoch fine-tune;
- checkpoint khoảng 21,8 MB và checksum;
- code tạo suy giảm, phục hồi, tuning và đánh giá 49 điều kiện;
- notebook, test, Streamlit và packaging.

Bạn còn thực hiện hai việc:

1. Chạy full 49 trên GPU bằng checkpoint sẵn có.
2. Đưa dự án đã có kết quả full lên GitHub và deploy Streamlit.

Full 49 gồm khoảng 63.184 lượt ảnh biến đổi/suy luận. Không dùng `--quick-run` vì kết quả quick chỉ để thử code và không hợp lệ cho báo cáo.

## Bước 1 — Xác nhận vị trí dự án và dataset trên Windows

Các vị trí hiện có trên máy:

```text
D:\Nawm 3_ Ky 1\Processing_image\TUNGDUONG_flower-image-restoration-cnn_95plus_merged
D:\Nawm 3_ Ky 1\Processing_image\Data\flower_photos
```

Dataset phải có đúng năm thư mục:

```text
flower_photos/
├── daisy/        633 ảnh
├── dandelion/    898 ảnh
├── roses/        641 ảnh
├── sunflowers/   699 ảnh
└── tulips/       799 ảnh
```

Tổng mong đợi là 3.670 ảnh. Tên thư mục trong code là `roses` và `sunflowers` ở dạng số nhiều; không đổi thành `rose` hoặc `sunflower`.

## Bước 2 — Kiểm tra NVIDIA GPU trong Windows

Mở PowerShell hoặc Windows Terminal và chạy:

```powershell
nvidia-smi
```

Kết quả đúng phải hiển thị tên GPU, phiên bản driver và dung lượng VRAM. Nếu lệnh không tồn tại hoặc báo lỗi:

1. Xác định model GPU trong Device Manager.
2. Cài/cập nhật NVIDIA driver dành cho GPU đó.
3. Khởi động lại Windows.
4. Chạy lại `nvidia-smi`.

Không chạy TensorFlow 2.21 GPU trực tiếp bằng Python Windows. TensorFlow chỉ hỗ trợ GPU Windows native tới 2.10; dự án dùng TensorFlow 2.21 nên cần WSL2. Tham khảo [TensorFlow pip install chính thức](https://www.tensorflow.org/install/pip).

## Bước 3 — Cài hoặc cập nhật WSL2

Mở **PowerShell bằng quyền Administrator**, rồi chạy từng lệnh:

```powershell
wsl --status
wsl --update
wsl --set-default-version 2
wsl --install -d Ubuntu
```

Nếu Windows yêu cầu khởi động lại, hãy khởi động lại. Sau đó mở ứng dụng **Ubuntu** từ Start Menu và tạo username/password Linux.

Quay lại PowerShell để kiểm tra:

```powershell
wsl -l -v
```

Dòng Ubuntu phải có cột `VERSION` bằng `2`. Nếu là `1`, chạy:

```powershell
wsl --set-version Ubuntu 2
```

Trong cửa sổ Ubuntu, kiểm tra GPU được chuyển vào WSL:

```bash
nvidia-smi
```

Nếu Windows thấy GPU nhưng Ubuntu không thấy, chạy `wsl --update`, sau đó trong PowerShell chạy `wsl --shutdown`, mở lại Ubuntu và thử lại. Không cài Linux NVIDIA display driver riêng bên trong WSL.

## Bước 4 — Chép dự án và dữ liệu vào ổ Linux của WSL

Chạy trong Ubuntu:

```bash
RUN_DIR="$HOME/flower-project-$(date +%Y%m%d-%H%M)"
mkdir -p "$RUN_DIR"
cp -a "/mnt/d/Nawm 3_ Ky 1/Processing_image/TUNGDUONG_flower-image-restoration-cnn_95plus_merged/." "$RUN_DIR/"
mkdir -p "$RUN_DIR/data"
cp -a "/mnt/d/Nawm 3_ Ky 1/Processing_image/Data/flower_photos" "$RUN_DIR/data/flower_photos"
cd "$RUN_DIR"
printf "%s" "$RUN_DIR" > "$HOME/.flower_project_path"
pwd
```

`pwd` phải cho ra đường dẫn bắt đầu bằng `/home/`, không phải `/mnt/d/`. Chạy trong ổ Linux thường nhanh và ổn định hơn khi xử lý hàng chục nghìn ảnh.

Kiểm tra năm lớp và số file:

```bash
for c in daisy dandelion roses sunflowers tulips; do printf "%s: " "$c"; find "data/flower_photos/$c" -type f | wc -l; done
find data/flower_photos -type f | wc -l
```

Kết quả mong đợi lần lượt là `633`, `898`, `641`, `699`, `799`, tổng `3670`. Nếu khác, dừng lại và kiểm tra bước sao chép.

## Bước 5 — Tạo môi trường Python

Trong Ubuntu, tại thư mục dự án:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git tmux
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install "tensorflow[and-cuda]==2.21.0"
```

Dấu nhắc lệnh nên có `(.venv)` ở đầu. Mỗi lần đóng và mở Ubuntu, quay lại thư mục dự án rồi kích hoạt lại:

```bash
cd "$RUN_DIR"
source .venv/bin/activate
```

Nếu mở terminal mới và biến `RUN_DIR` không còn, khôi phục bằng:

```bash
RUN_DIR="$(cat "$HOME/.flower_project_path")"
cd "$RUN_DIR"
source .venv/bin/activate
```

Dự án chấp nhận Python từ 3.10 đến 3.13; Python 3.11 là lựa chọn ưu tiên. Cài `tensorflow[and-cuda]` theo hướng dẫn chính thức để thêm các thư viện CUDA cần thiết trong môi trường pip.

## Bước 6 — Xác minh TensorFlow thực sự dùng GPU

Chạy:

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU')); print('Build:', tf.sysconfig.get_build_info())"
```

Kết quả đạt yêu cầu:

```text
TensorFlow: 2.21.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', ...)]
```

Nếu `GPU: []`, không chạy full 49 vội. Thử:

```bash
deactivate
source .venv/bin/activate
pip install --upgrade "tensorflow[and-cuda]==2.21.0"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Nếu vẫn rỗng, kiểm tra lại `nvidia-smi`, phiên bản WSL2 và NVIDIA driver. Bạn vẫn có thể chạy CPU nhưng sẽ mất nhiều thời gian hơn.

## Bước 7 — Kiểm tra checkpoint và audit dữ liệu trước khi chạy dài

Kiểm tra checkpoint tồn tại:

```bash
ls -lh models/mobilenetv2_flowers.keras
sha256sum models/mobilenetv2_flowers.keras
```

SHA-256 mong đợi của checkpoint đóng gói:

```text
54e76a3464d73672dabf823b4040ef67607c49861e2a7286ca233c138dbbc44d
```

Chạy audit-only; lệnh này không huấn luyện và không chạy full 49:

```bash
python scripts/run_full_pipeline.py --audit-only
```

Kiểm tra output phải thể hiện 3.670 ảnh hợp lệ và 0 ảnh hỏng. Sau đó chạy core validator:

```bash
python scripts/validate_project.py
```

`errors` phải là danh sách rỗng. Trạng thái `submission_ready: false` ở thời điểm này là bình thường vì full 49 và deploy chưa hoàn tất.

## Bước 8 — Chạy full 49 trong tmux

`tmux` giúp tiến trình tiếp tục chạy nếu bạn đóng cửa sổ Terminal. Tạo phiên:

```bash
tmux new -s flower49
```

Trong phiên tmux:

```bash
RUN_DIR="$(cat "$HOME/.flower_project_path")"
cd "$RUN_DIR"
source .venv/bin/activate
set -o pipefail
python scripts/run_full_pipeline.py --skip-train 2>&1 | tee full49_console.log
```

Ý nghĩa tham số:

- `--skip-train`: dùng checkpoint đã huấn luyện, không huấn luyện lại.
- Không có `--quick-run`: chạy toàn bộ validation và Test.
- Không có `--regenerate-splits`: giữ nguyên split đã khóa.

Để rời tmux mà không dừng chương trình, nhấn `Ctrl+B`, thả ra, rồi nhấn `D`. Để quay lại:

```bash
tmux attach -t flower49
```

Mở cửa sổ Ubuntu khác để theo dõi GPU:

```bash
watch -n 2 nvidia-smi
```

Bạn nên thấy Python sử dụng GPU memory và GPU utilization thay đổi. Pipeline cũng sử dụng CPU cho xử lý ảnh nên GPU có thể không luôn ở 100%.

Không tắt máy, sleep hoặc đóng WSL bằng `wsl --shutdown` trong lúc chạy. Nếu chương trình bị ngắt, chạy lại cùng lệnh; pipeline chưa có cơ chế resume giữa chừng nên sẽ thực hiện lại tuning/evaluation.

## Bước 9 — Nhận biết full 49 đã thành công

Cuối terminal phải có thông báo hoàn tất. Sau đó chạy:

```bash
python -c "import json,pandas as pd; from pathlib import Path; r=Path('results/final'); m=json.loads((r/'manifest.json').read_text()); print('conditions=',len(pd.read_csv(r/'condition_metrics.csv'))); print('predictions=',len(pd.read_csv(r/'predictions.csv'))); print('per_class=',len(pd.read_csv(r/'per_class_metrics.csv'))); print('quick_run=',m['metadata']['quick_run'])"
```

Kết quả tối thiểu phải là:

```text
conditions= 49
predictions= 26950
per_class= 245
quick_run= False
```

Kiểm tra metadata:

```bash
python -c "import json; m=json.load(open('models/model_metadata.json',encoding='utf-8')); print(m['status']); print(m['model_sha256'])"
```

Trạng thái phải là:

```text
FULL_RUN_COMPLETE
```

Những file quan trọng được tạo:

```text
results/final/condition_metrics.csv
results/final/predictions.csv
results/final/per_class_metrics.csv
results/final/statistical_tests.csv
results/final/error_analysis.csv
results/final/top_confusion_pairs.csv
results/final/confusion_matrices/
results/final/manifest.json
configs/locked_enhancement_params.json
artifacts/full_run_metadata.json
```

Không dùng kết quả trong `results/quick_smoke_not_for_report/` cho báo cáo.

## Bước 10 — Chạy các gate sau full 49

Thực hiện lần lượt:

```bash
python scripts/validate_project.py --require-full-run
python scripts/check_consistency.py
python scripts/build_evidence_notebook.py
python scripts/execute_notebook.py
python scripts/check_notebook.py --require-executed --require-full-run
python -m pytest tests -q -p no:cacheprovider
```

Chỉ tiếp tục deploy khi:

- validator không có lỗi;
- coherence có `pass: true`;
- notebook không có error cell và có marker `FULL_RUN_COMPLETE`;
- toàn bộ tests PASS.

Nếu muốn xem notebook bằng giao diện:

```bash
pip install jupyterlab
jupyter lab --no-browser
```

Mở URL có token mà Jupyter in ra trong trình duyệt Windows.

## Bước 11 — Chạy thử Streamlit cục bộ

Trong Ubuntu:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

Mở trình duyệt Windows tại:

```text
http://localhost:8501
```

Kiểm tra:

1. Trang không còn báo readiness blocked.
2. Upload một ảnh JPG/PNG và chạy nhận diện.
3. Kiểm tra nhãn, confidence và ảnh sau xử lý.
4. Upload batch 2–3 ảnh.
5. Thử đủ các lựa chọn suy giảm/phục hồi quan trọng.

Dừng Streamlit bằng `Ctrl+C`.

Nếu app báo checksum hoặc `FULL_RUN_COMPLETE` chưa đúng, chạy lại Bước 9 và Bước 10; không sửa tay readiness gate.

## Bước 12 — Đưa dự án lên GitHub

Raw dataset không được đưa lên GitHub. File `.gitignore` đã bỏ qua `data/flower_photos/`, `.venv/`, cache và secrets. Checkpoint `.keras` được phép theo dõi vì app deploy cần model này.

Kiểm tra trước:

```bash
git check-ignore data/flower_photos/daisy/* | head
git check-ignore models/mobilenetv2_flowers.keras
```

Lệnh thứ nhất nên in đường dẫn dataset bị ignore. Lệnh thứ hai phải không in gì, nghĩa là model không bị ignore.

Tạo repository mới trên GitHub nhưng không chọn tạo README/License tự động. Sau đó, trong Ubuntu:

```bash
git init
git branch -M main
git config user.name "TEN_CUA_BAN"
git config user.email "EMAIL_GITHUB_CUA_BAN"
git add .
git status --short
```

Trong `git status`, bảo đảm:

- có `models/mobilenetv2_flowers.keras`;
- không có `.venv/`;
- không có `data/flower_photos/`;
- không có secret/token.

Commit và push, thay URL bằng repository của bạn:

```bash
git commit -m "Final flower restoration CNN project"
git remote add origin https://github.com/TEN_GITHUB/TEN_REPOSITORY.git
git push -u origin main
```

Không ghi GitHub password/token trực tiếp vào file dự án. Dùng luồng đăng nhập an toàn mà Git/GitHub hiển thị.

## Bước 13 — Deploy trên Streamlit Community Cloud

Streamlit Community Cloud lấy source từ GitHub, đọc dependency trong `requirements.txt` và chạy entrypoint từ root repository. Hướng dẫn chính thức: [Deploy app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) và [quản lý dependencies](https://docs.streamlit.io/deploy/concepts/dependencies).

Thao tác:

1. Mở [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng GitHub.
2. Cho phép Streamlit truy cập repository vừa tạo.
3. Chọn **Create app**.
4. Chọn **Yup, I have an app** nếu được hỏi.
5. Repository: repository dự án của bạn.
6. Branch: `main`.
7. Main file path: `streamlit_app.py`.
8. Vào **Advanced settings**.
9. Chọn Python `3.11` để khớp `.python-version` của dự án.
10. Không cần nhập Secrets cho dự án này.
11. Nhấn **Deploy**.

Sau khi build xong, mở URL dạng `https://<ten-app>.streamlit.app` và kiểm tra lại một ảnh cùng một batch 2–3 ảnh.

Nếu deploy lỗi, mở **Manage app → Logs** và đối chiếu:

- `ModuleNotFoundError`: package thiếu trong `requirements.txt`;
- model missing: kiểm tra checkpoint có trên GitHub;
- checksum/readiness blocked: kiểm tra metadata, locked params và full-run manifest đã commit;
- out of memory: reboot app, giảm batch upload, hoặc cân nhắc nền tảng có RAM cao hơn;
- Python không tương thích: xóa app và deploy lại với Python 3.11; Streamlit yêu cầu redeploy để đổi Python version.

## Bước 14 — Ghi bằng chứng deploy

Lấy commit SHA:

```bash
git rev-parse HEAD
```

Lấy model SHA-256:

```bash
sha256sum models/mobilenetv2_flowers.keras
```

Tạo hoặc cập nhật `artifacts/deployment_verification.json` theo mẫu:

```json
{
  "status": "PASS",
  "public_url": "https://TEN-APP.streamlit.app",
  "git_commit": "COMMIT_SHA",
  "model_sha256": "MODEL_SHA256",
  "verified_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "checks": {
    "startup": "PASS",
    "single_image": "PASS",
    "batch_upload": "PASS"
  }
}
```

Chụp một ảnh màn hình URL public, phần dự đoán một ảnh và kết quả batch để làm bằng chứng nộp bài.

## Bước 15 — Đóng gói lại ZIP sau full 49

Trong thư mục dự án WSL:

```bash
python scripts/build_submission_manifest.py
python scripts/package_submission.py
python scripts/verify_submission_package.py ../TUNGDUONG_flower-image-restoration-cnn_95plus_merged.zip
```

Khi thấy `PACKAGE_VERIFY_PASS`, sao chép ZIP mới về Windows:

```bash
cp ../TUNGDUONG_flower-image-restoration-cnn_95plus_merged.zip "/mnt/d/Nawm 3_ Ky 1/Processing_image/TUNGDUONG_flower-image-restoration-cnn_95plus_merged_FULL49.zip"
```

Giữ hai bản:

- ZIP hiện tại: checkpoint-ready, dùng để phục hồi nếu chạy lỗi.
- ZIP `_FULL49.zip`: có kết quả full 49, dùng để nộp sau khi validator PASS.

## Xử lý lỗi thường gặp

### `Dataset missing`

Kiểm tra:

```bash
pwd
ls data/flower_photos
```

Phải thấy đúng năm thư mục lớp. Nếu dataset ở vị trí khác, chép lại vào `data/flower_photos` hoặc truyền `--data-dir` với đường dẫn tuyệt đối.

### Số ảnh audit không phải 3.670

Không chạy tiếp. So sánh số file từng lớp với Bước 1. Không tự tái tạo split với dataset thiếu.

### `No module named tensorflow` hoặc package khác

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install "tensorflow[and-cuda]==2.21.0"
```

### TensorFlow không nhận GPU

Kiểm tra theo thứ tự:

1. `nvidia-smi` trên Windows.
2. `wsl -l -v` phải là WSL version 2.
3. `nvidia-smi` trong Ubuntu.
4. `.venv` đã activate.
5. `tensorflow[and-cuda]` đã cài.

### CUDA out of memory

Đóng ứng dụng đang dùng GPU, kiểm tra `nvidia-smi`, rồi chạy lại. Nếu GPU VRAM nhỏ, có thể giảm `batch_size` mặc định từ 32 xuống 16 trong `src/inference.py` và `src/evaluate.py`; sau thay đổi phải chạy lại test và ghi rõ cấu hình latency mới.

### Máy tắt hoặc tiến trình bị ngắt

Mở Ubuntu, vào đúng `RUN_DIR`, activate `.venv`, rồi chạy lại lệnh full. Không dùng các file `results/final` dở dang để báo cáo trước khi validator full PASS.

### Streamlit local chạy nhưng cloud lỗi

Kiểm tra repository thực sự chứa:

```text
streamlit_app.py
requirements.txt
models/mobilenetv2_flowers.keras
models/model_metadata.json
models/class_names.json
configs/locked_enhancement_params.json
results/final/manifest.json
```

Xem log cloud trước khi thay đổi code. Sau mỗi sửa đổi, commit và push; Streamlit sẽ tự redeploy.

## Phương án Google Colab nếu WSL2 không hoạt động

1. Upload ZIP và dataset lên Google Drive.
2. Mở Colab, chọn **Runtime → Change runtime type → GPU**.
3. Giải nén dự án vào `/content/flower-project`, không chạy trực tiếp trên Drive nếu có thể.
4. Cài `requirements.txt`.
5. Kiểm tra `tf.config.list_physical_devices('GPU')`.
6. Chạy `python scripts/run_full_pipeline.py --skip-train`.
7. Tải về ít nhất `results/final/`, `configs/locked_enhancement_params.json`, `models/model_metadata.json` và `artifacts/full_run_metadata.json`.
8. Chép các file đó vào dự án gốc, chạy validator và đóng gói lại.

Không dùng Colab session tạm làm nơi lưu duy nhất; tải kết quả về trước khi session hết hạn.
