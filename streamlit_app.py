from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app_components.io import content_identifier, decode_uploaded_image
from app_components.pipeline import run_pipeline
from app_components.readiness import inspect_artifact_readiness
from src.config import CLASS_NAMES, DEGRADATION_PARAMS, LEVELS, METHODS
from src.inference import ModelService

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "mobilenetv2_flowers.keras"
LOCKED_PATH = ROOT / "configs" / "locked_enhancement_params.json"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"
LEVEL_LABELS = {"light": "Nhẹ", "medium": "Vừa", "strong": "Mạnh"}
DEGRADATION_LABELS = {
    "low_light": "Thiếu sáng",
    "gaussian_noise": "Nhiễu Gaussian",
    "salt_pepper": "Nhiễu salt-and-pepper",
    "gaussian_blur": "Làm mờ Gaussian",
    "color_cast": "Ám màu",
}


@st.cache_resource(show_spinner="Đang nạp MobileNetV2...")
def load_model_service(model_path: str) -> ModelService:
    return ModelService(Path(model_path))


@st.cache_data
def load_readiness(model_path: str, metadata_path: str, class_names_path: str, locked_path: str) -> dict:
    return inspect_artifact_readiness(
        Path(model_path), Path(metadata_path), Path(class_names_path), Path(locked_path)
    )


def probability_frame(predictions: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            title: [predictions[key]["probabilities"][name] for name in CLASS_NAMES]
            for key, title in (("clean", "Ảnh gốc"), ("degraded", "Suy giảm"), ("enhanced", "Sau xử lý"))
        },
        index=CLASS_NAMES,
    )


st.set_page_config(page_title="Nhận diện hoa dưới suy giảm", page_icon="🌼", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background:#F7FAFC;}
    h1,h2,h3 {color:#17365D;}
    [data-testid='stMetric'] {background:white;border:1px solid #DCE6F1;padding:12px;border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Nhận diện hoa trong điều kiện ảnh suy giảm")
st.caption("Streamlit standalone · một MobileNetV2 cố định · xử lý ảnh và suy luận trong cùng process")

readiness = load_readiness(
    str(MODEL_PATH), str(METADATA_PATH), str(CLASS_NAMES_PATH), str(LOCKED_PATH)
)
model_ready = readiness["ready"]
locked_ready = bool(readiness["locked"])
with st.sidebar:
    st.header("Cấu hình thí nghiệm")
    run_mode = st.radio("Chế độ", ["Một ảnh", "Batch (tối đa 20 ảnh)"])
    degradation = st.selectbox(
        "Dạng suy giảm",
        list(METHODS),
        format_func=lambda value: DEGRADATION_LABELS[value],
    )
    level = st.select_slider("Mức độ", options=LEVELS, format_func=lambda value: LEVEL_LABELS[value])
    method = st.selectbox(
        "Phương pháp xử lý",
        METHODS[degradation],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    st.divider()
    st.write(f"Model: **{'sẵn sàng' if model_ready else 'chưa có'}**")
    st.write(f"Tham số Validation: **{'đã khóa' if locked_ready else 'chưa khóa'}**")
    short_hash = (readiness["model_sha256"] or "missing")[:12]
    st.caption(f"mobilenetv2_flowers.keras · SHA256 `{short_hash}`")

for readiness_error in readiness["errors"]:
    st.error(f"{readiness_error}. Pipeline chính thức đang bị vô hiệu hóa; không có kết quả thay thế.")

uploaded = st.file_uploader(
    "Tải ảnh JPEG/PNG (tối đa 10 MB mỗi ảnh)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=run_mode.startswith("Batch"),
    disabled=not (model_ready and locked_ready),
)
has_upload = bool(uploaded) if isinstance(uploaded, list) else uploaded is not None
if not has_upload:
    st.info("Sau khi model và tham số khóa sẵn sàng, tải một ảnh để so sánh ảnh gốc, ảnh suy giảm và ảnh sau xử lý.")
else:
    try:
        metadata, locked = readiness["metadata"], readiness["locked"]
        service = load_model_service(str(MODEL_PATH.resolve()))
        files = uploaded[:20] if isinstance(uploaded, list) else [uploaded]
        batch_results = []
        with st.spinner(f"Đang xử lý {len(files)} ảnh..."):
            for file in files:
                content = file.getvalue()
                result = run_pipeline(
                    decode_uploaded_image(content), identifier=content_identifier(content),
                    degradation=degradation, level=level, method=method,
                    model_service=service, locked_params=locked,
                    degradation_params=DEGRADATION_PARAMS[degradation][level],
                )
                batch_results.append((file.name, result))

        if len(batch_results) == 1:
            _, result = batch_results[0]
            columns = st.columns(3)
            for column, picture, title in zip(
                columns, (result.clean, result.degraded, result.enhanced),
                ("Ảnh gốc", "Ảnh suy giảm", "Ảnh sau xử lý"),
            ):
                column.image(picture, caption=title, use_container_width=True)
            metric_columns = st.columns(3)
            for column, key, title in zip(metric_columns, ("clean", "degraded", "enhanced"), ("Gốc", "Suy giảm", "Sau xử lý")):
                prediction = result.predictions[key]
                column.metric(title, prediction["label"], f'{prediction["confidence"]:.1%}')
            st.bar_chart(probability_frame(result.predictions))
            st.caption(f"Thời gian pipeline đo thật: {result.processing_time_ms:.1f} ms · Model: {metadata.get('architecture', 'MobileNetV2')}")
            st.json({"degradation": result.degradation_params, "enhancement_locked": result.enhancement_params})
        else:
            rows = []
            for filename, result in batch_results:
                rows.append({
                    "file": filename,
                    "clean_label": result.predictions["clean"]["label"],
                    "degraded_label": result.predictions["degraded"]["label"],
                    "enhanced_label": result.predictions["enhanced"]["label"],
                    "enhanced_confidence": result.predictions["enhanced"]["confidence"],
                    "processing_ms": result.processing_time_ms,
                })
            frame = pd.DataFrame(rows)
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.download_button("Tải kết quả batch CSV", frame.to_csv(index=False).encode("utf-8-sig"), "batch_predictions.csv", "text/csv")
            st.caption("Mỗi ảnh dùng cùng degradation/enhancement đã chọn; confidence không phải Macro F1.")
    except ValueError as exc:
        st.error(str(exc))
    except Exception:  # noqa: BLE001
        st.error("Pipeline gặp lỗi. Hãy kiểm tra checkpoint, class mapping và cấu hình đã khóa; chi tiết kỹ thuật không được hiển thị trên giao diện.")

with st.expander("Giới hạn của hệ thống"):
    st.write(
        "Mô hình chỉ nhận diện năm lớp hoa; các suy giảm là mô phỏng; confidence không phải độ chắc chắn tuyệt đối. "
        "Ứng dụng phục vụ mục đích học thuật, không phải hệ thống nhận diện chuyên nghiệp."
    )
