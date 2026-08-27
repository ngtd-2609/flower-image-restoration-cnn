"""Build a clear, executable notebook for the trained-checkpoint handoff.

The notebook runs without the raw dataset. Set RUN_FULL_49=True only after
placing data/flower_photos in the project and using a GPU environment.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "BTL_XuLyAnh_NhanDienHoa.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.strip().splitlines(True),
    }


cells = [
    md("""# Phần 0 — Tổng quan\n\nPhục hồi ảnh hoa và phân loại bằng MobileNetV2. Notebook này chứng minh checkpoint, dữ liệu audit, split và quy trình; full 49 được để thành bước chủ động chạy bằng GPU."""),
    code("""from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
ROOT = Path.cwd()
assert (ROOT / 'src').exists(), 'Hãy chạy notebook từ thư mục gốc dự án'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print({'python': sys.version.split()[0], 'root': str(ROOT)})"""),
    md("""## Phần 1 — Trạng thái bàn giao\n\nCNN đã huấn luyện; full 49 và deploy chờ người dùng chạy theo hướng dẫn."""),
    code("""meta = json.loads((ROOT/'models/model_metadata.json').read_text(encoding='utf-8'))
print({k: meta[k] for k in ['architecture','status','model_size_bytes','training_duration_seconds']})
print('FULL_49_PENDING_USER_GPU' if meta['status'] != 'FULL_RUN_COMPLETE' else 'FULL_RUN_COMPLETE')"""),
    md("""## Phần 2 — Audit dữ liệu\n\nInventory là bằng chứng audit đã lưu, nên notebook vẫn chạy khi ZIP không kèm raw dataset."""),
    code("""inventory = pd.read_csv(ROOT/'data/inventory.csv')
print({'valid_images': len(inventory), 'classes': inventory['label'].nunique()})
print(inventory['label'].value_counts().sort_index().to_dict())"""),
    md("""## Phần 3 — Split cố định\n\nTrain/validation/test được kiểm tra không giao nhau theo cả đường dẫn và SHA-256."""),
    code("""splits = {n: pd.read_csv(ROOT/f'splits/{n}.csv') for n in ['train','validation','test']}
print({n: len(df) for n, df in splits.items()})
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert set(splits[a].relative_path).isdisjoint(set(splits[b].relative_path))
    assert set(splits[a].sha256).isdisjoint(set(splits[b].sha256))
print('SPLIT_DISJOINT_PASS')"""),
    md("""## Phần 4 — Tiền xử lý\n\nEXIF transpose → RGB → letterbox LANCZOS 224×224 → MobileNetV2 preprocess_input trong graph."""),
    md("""## Phần 5 — Kiến trúc CNN\n\nMobileNetV2 dùng trọng số ImageNet, head 5 lớp; huấn luyện 15 epoch head và 10 epoch fine-tune."""),
    code("""history = pd.read_csv(ROOT/'artifacts/training/history.csv')
print({'epochs_recorded': len(history), 'outputs': meta['output_shape'], 'classes': meta['class_names']})"""),
    md("""## Phần 6 — Checksum checkpoint\n\nChecksum được tính lại trực tiếp, không tin giá trị chép tay."""),
    code("""model_path = ROOT/'models/mobilenetv2_flowers.keras'
actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
print({'bytes': model_path.stat().st_size, 'sha256': actual_sha, 'matches_metadata': actual_sha == meta['model_sha256']})
assert actual_sha == meta['model_sha256']"""),
    md("""## Phần 7 — Suy luận mẫu bằng model thật\n\nTải checkpoint trong tiến trình mới và dự đoán ảnh demo đóng gói kèm dự án."""),
    code("""try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None
if tf is None:
    print('TENSORFLOW_NOT_INSTALLED: checkpoint/checksum hợp lệ; cài requirements để chạy suy luận')
else:
    from PIL import Image, ImageOps
    model = tf.keras.models.load_model(model_path, compile=False)
    img = ImageOps.exif_transpose(Image.open(ROOT/'assets/demo_flower.jpg')).convert('RGB')
    img = ImageOps.pad(img, (224,224), method=Image.Resampling.LANCZOS, color=(0,0,0))
    x = np.asarray(img, dtype=np.float32)[None,...]
    probs = model.predict(x, verbose=0)[0]
    print({'predicted_class': meta['class_names'][int(np.argmax(probs))], 'confidence': float(np.max(probs)), 'probability_sum': float(probs.sum())})"""),
    md("""## Phần 8 — Năm loại suy giảm\n\nLow light, Gaussian noise, salt-and-pepper, Gaussian blur và color cast; mỗi loại có ba mức độ."""),
    code("""from src.experiment_matrix import build_experiment_matrix
matrix = build_experiment_matrix()
print({'condition_count': len(matrix), 'unique_ids': len({c.condition_id for c in matrix})})
assert len(matrix) == 49"""),
    md("""## Phần 9 — Các phép phục hồi\n\nGamma/CLAHE, Gaussian/bilateral/median filter, sharpening/unsharp mask và cân bằng màu RGB/HSV/LAB."""),
    md("""## Phần 10 — Quy tắc tuning\n\nChỉ validation được dùng để chọn tham số: Macro F1 giảm dần → SSIM giảm dần → latency tăng dần. Test không tham gia lựa chọn."""),
    code("""locked = json.loads((ROOT/'configs/locked_enhancement_params.json').read_text(encoding='utf-8'))
print({'selection_split': locked.get('_metadata',{}).get('selection_split'), 'quick_run': locked.get('_metadata',{}).get('quick_run'), 'parameter_groups': len(locked.get('parameters',{}))})"""),
    md("""## Phần 11 — Giao thức full 49\n\nMỗi condition chạy trên toàn bộ 550 ảnh Test; predictions được lưu một lần rồi tái sử dụng cho metrics, thống kê và error analysis."""),
    md("""## Phần 12 — Cờ chạy GPU\n\nĐể mặc định False khi trình bày. Chỉ bật sau khi dataset đúng cấu trúc và TensorFlow đã nhận GPU."""),
    code("""RUN_FULL_49 = False
if RUN_FULL_49:
    import subprocess
    subprocess.run([sys.executable, 'scripts/run_full_pipeline.py', '--skip-train'], check=True)
else:
    print('SKIPPED_BY_DESIGN: xem docs/RUN_FULL_49_AND_DEPLOY.md')"""),
    md("""## Phần 13 — Đọc kết quả cuối an toàn\n\nNotebook không nhầm smoke test với kết quả báo cáo."""),
    code("""manifest_path = ROOT/'results/final/manifest.json'
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest.get('metadata',{}).get('quick_run') is False
    print({'status':'FULL_RUN_COMPLETE','conditions':manifest['condition_count'],'predictions':manifest['prediction_rows']})
else:
    print('FULL_EVALUATION_PENDING')"""),
    md("""## Phần 14 — Streamlit readiness\n\nỨng dụng khóa dự đoán nếu full params/checksum/metadata chưa đồng bộ; đây là hành vi trung thực có chủ đích."""),
    code("""from app_components.readiness import inspect_artifact_readiness
r = inspect_artifact_readiness(model_path, ROOT/'models/model_metadata.json', ROOT/'models/class_names.json', ROOT/'configs/locked_enhancement_params.json')
print({'app_ready': r['ready'], 'reasons': r['errors']})"""),
    md("""## Phần 15 — Test và coherence\n\nChạy `pytest`, core validator và `scripts/check_consistency.py`; full gate chỉ dùng sau khi full 49 hoàn tất."""),
    md("""## Phần 16 — Kết luận và bước tiếp theo\n\nCheckpoint CNN, pipeline, notebook và app đã đóng gói. Tiếp theo: chạy full 49 trên WSL2 GPU, validate, rồi deploy Streamlit theo `docs/RUN_FULL_49_AND_DEPLOY.md`."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells")
