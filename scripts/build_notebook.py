from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


cells = [
    md("""
    # BÀI TẬP LỚN XỬ LÝ ẢNH
    ## Đánh giá và nâng cao độ chính xác nhận diện 5 loại hoa trong điều kiện ảnh suy giảm

    Notebook này là **kịch bản thực nghiệm chính thức** của dự án. Toàn bộ kết quả nhận diện chỉ được
    xem là hợp lệ sau khi chạy `Runtime → Restart session and run all` trên Google Colab có GPU.
    Môi trường bàn giao hiện chưa chạy TensorFlow nên các ô kết quả MobileNetV2 cố ý để trống; không có
    số liệu surrogate nào được dùng thay kết quả CNN.
    """),
    md("""
    # Phần 0 — Hướng dẫn sử dụng

    **Mục tiêu.** Kiểm tra dữ liệu, EDA, chia tập chống rò rỉ, sinh năm dạng suy giảm, chọn tham số
    tiền xử lý trên Validation, huấn luyện đúng một MobileNetV2 và đánh giá 49 điều kiện trên Test.

    **Input.** Thư mục `data/flower_photos/` gồm năm lớp `daisy`, `dandelion`, `roses`, `sunflowers`,
    `tulips`. **Output.** Model `.keras`, split CSV, bảng 49 điều kiện, metric theo lớp, phân tích lỗi,
    biểu đồ và manifest tái lập.

    Trên Colab, tải repository và dataset rồi bật GPU T4. Thời gian dự kiến 45–120 phút tùy GPU.
    Không đổi seed, class order hoặc split giữa các lần chạy. Nếu phiên làm việc bị ngắt, tải lại model
    tốt nhất và tiếp tục từ phần đánh giá; không tái chia dữ liệu.
    """),
    code("""
    # Chỉ chạy trên Colab mới: cài đúng dependency của dự án.
    import sys
    if "google.colab" in sys.modules:
        !pip -q install -r requirements.txt
    """),
    md("""
    ## Cấu trúc chạy

    1. Đặt notebook ở thư mục gốc repository.
    2. Giải nén dữ liệu vào `data/flower_photos/`.
    3. Chọn `Runtime > Change runtime type > T4 GPU`.
    4. Chọn `Runtime > Restart session and run all`.
    5. Kiểm tra checklist cuối: model tồn tại, split không giao nhau và bảng kết quả có 49 dòng.
    """),
    md("""
    # Phần 1 — Thiết lập

    Phần này import thư viện đúng một lần, khóa seed và tạo thư mục đầu ra. Mọi đường dẫn đều tương đối
    theo repository để notebook chạy được cả local và Colab.
    """),
    code("""
    from pathlib import Path
    import json, os, platform, random, time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    import yaml

    import tensorflow as tf
    from sklearn.metrics import confusion_matrix, classification_report

    from src.config import CLASS_NAMES, DEGRADATION_PARAMS, LEVELS, METHODS, SEED
    from src.data_validation import scan_dataset, save_validation
    from src.data_split import grouped_stratified_split, save_splits, assert_no_leakage
    from src.data_loader import build_tf_dataset, load_rgb
    from src.degradations import apply_degradation, stable_seed
    from src.enhancements import apply_enhancement
    from src.image_metrics import full_reference_metrics
    from src.train import train_two_stage
    from src.inference import ModelService
    from src.tuning import tune_on_validation, save_tuning
    from src.evaluate import evaluate_full_experiment, save_evaluation_artifacts
    from src.error_analysis import build_error_analysis
    """),
    code("""
    ROOT = Path.cwd()
    DATA_DIR = ROOT / "data" / "flower_photos"
    SPLIT_DIR, MODEL_DIR = ROOT / "splits", ROOT / "models"
    RESULT_DIR, FIGURE_DIR = ROOT / "results", ROOT / "figures"
    for folder in (SPLIT_DIR, MODEL_DIR, RESULT_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    RUN_MODE = os.getenv("RUN_MODE", "FULL_RUN").upper()
    assert RUN_MODE in {"QUICK_RUN", "FULL_RUN"}
    print({"python": platform.python_version(), "tensorflow": tf.__version__,
           "gpu": tf.config.list_physical_devices("GPU"), "root": str(ROOT), "run_mode": RUN_MODE})
    assert DATA_DIR.exists(), "Thiếu data/flower_photos"
    environment = {
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "seed": SEED, "run_mode": RUN_MODE, "created_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    """),
    md("""
    # Phần 2 — Kiểm tra dữ liệu

    Kiểm kê đọc từng ảnh, chuẩn hóa EXIF, xác thực khả năng giải mã và tính SHA-256. SHA-256 được dùng
    làm khóa nhóm khi chia tập; vì vậy ảnh trùng byte và mọi biến thể về sau không thể xuất hiện ở hai
    split. Trùng khác nhãn được báo cáo riêng vì đây là dấu hiệu xung đột nhãn.
    """),
    code("""
    inventory, validation_report = scan_dataset(DATA_DIR)
    save_validation(inventory, validation_report, RESULT_DIR)
    display(pd.DataFrame({
        "chỉ tiêu": ["Ảnh hợp lệ", "Ảnh hỏng", "Nhóm trùng", "Nhóm trùng khác nhãn"],
        "giá trị": [validation_report["valid_images"], len(validation_report["bad_images"]),
                    validation_report["duplicate_group_count"],
                    validation_report["cross_label_duplicate_group_count"]],
    }))
    display(inventory.groupby("label").agg(images=("path", "size"),
            width_mean=("width", "mean"), height_mean=("height", "mean")))
    """),
    code("""
    assert inventory["label"].nunique() == 5
    assert set(inventory["label"]) == set(CLASS_NAMES)
    assert inventory["sha256"].str.len().eq(64).all()
    if validation_report["bad_images"]:
        display(pd.DataFrame(validation_report["bad_images"]))
    if validation_report["duplicate_groups"]:
        display(pd.DataFrame(validation_report["duplicate_groups"][0]))
    """),
    md("""
    **Nhận xét cần ghi sau khi chạy.** So sánh số tệp trong kho với số ảnh giải mã hợp lệ; nêu rõ tệp
    lỗi và quyết định loại khỏi thí nghiệm. Không tự xóa trùng: ảnh trùng được gom cùng split để vừa
    giữ dấu vết dữ liệu vừa ngăn rò rỉ.
    """),
    md("""
    # Phần 3 — EDA

    EDA xem xét mất cân bằng lớp, kích thước, tỷ lệ khung hình, độ sáng, tương phản, phân bố màu và biên.
    Các thống kê này quyết định cách resize, metric phụ và rủi ro khi một kỹ thuật xử lý làm biến dạng
    dấu hiệu màu hoặc cấu trúc cánh hoa.
    """),
    code("""
    class_counts = inventory["label"].value_counts().reindex(CLASS_NAMES)
    ax = class_counts.plot.bar(color="#4F81BD", figsize=(8, 4), title="Phân bố năm lớp hoa")
    ax.set(xlabel="Lớp", ylabel="Số ảnh")
    for container in ax.containers: ax.bar_label(container)
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "eda" / "class_distribution.png", dpi=200)
    plt.show()
    display(class_counts.rename("images").to_frame())
    """),
    code("""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, label in zip(axes, CLASS_NAMES):
        row = inventory[inventory.label.eq(label)].sample(1, random_state=SEED).iloc[0]
        ax.imshow(load_rgb(ROOT / row.relative_path)); ax.set_title(label); ax.axis("off")
    fig.suptitle("Ảnh mẫu theo lớp"); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda" / "sample_classes.png", dpi=200); plt.show()
    """),
    code("""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(inventory, x="width", hue="label", element="step", ax=axes[0])
    sns.histplot(inventory, x="height", hue="label", element="step", legend=False, ax=axes[1])
    sns.histplot(inventory, x="aspect_ratio", hue="label", element="step", legend=False, ax=axes[2])
    axes[0].set_title("Chiều rộng"); axes[1].set_title("Chiều cao")
    axes[2].set_title("Tỷ lệ rộng/cao"); plt.tight_layout(); plt.show()
    """),
    code("""
    eda_metrics = pd.read_csv(RESULT_DIR / "eda_sample_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, column, title in zip(axes, ["brightness", "contrast", "edge_density"],
            ["Độ sáng", "Tương phản RMS", "Mật độ biên"]):
        sns.boxplot(eda_metrics, x="label", y=column, ax=ax, color="#9DC3E6")
        ax.set_title(title); ax.tick_params(axis="x", rotation=25)
    plt.tight_layout(); plt.show()
    """),
    md("""
    **Nhận xét EDA.** Sau khi chạy, mô tả lớp nhiều/ít ảnh nhất, mức lệch tỷ lệ, sự khác biệt độ sáng
    và cấu trúc. Resize vuông 224×224 giúp batching nhưng có thể làm biến dạng ảnh quá dẹt; dự án dùng
    cùng quy tắc cho mọi điều kiện để biến kiểm soát không thay đổi.
    """),
    md("""
    # Phần 4 — Split

    Dữ liệu được chia xấp xỉ 70/15/15 bằng `StratifiedGroupKFold`: nhãn dùng để duy trì tỷ lệ lớp,
    SHA-256 dùng làm group. Split diễn ra trước augmentation/degradation. Validation dành cho lựa chọn
    tham số; Test chỉ mở đúng một lần sau khi khóa cấu hình.
    """),
    code("""
    splits = grouped_stratified_split(inventory)
    save_splits(splits, SPLIT_DIR)
    split_summary = pd.concat([
        frame.groupby("label").size().rename(name) for name, frame in splits.items()
    ], axis=1).fillna(0).astype(int)
    split_summary.loc["Tổng"] = split_summary.sum()
    display(split_summary)
    """),
    code("""
    assert_no_leakage(splits)
    all_paths = set().union(*(set(frame.relative_path) for frame in splits.values()))
    assert len(all_paths) == len(inventory)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(splits[left].sha256) & set(splits[right].sha256)
    print("PASS: không giao đường dẫn, không giao SHA-256, không bỏ sót ảnh hợp lệ.")
    """),
    md("""
    # Phần 5 — Degradation

    Năm phép suy giảm được định nghĩa trong `src/degradations.py`. Noise dùng seed băm từ đường dẫn,
    loại và mức độ để tái lập tuyệt đối. Ba mức giúp quan sát xu hướng liều–đáp ứng; tuy nhiên PSNR/SSIM
    không bắt buộc đơn điệu trên mọi ảnh riêng lẻ, nên kết luận dựa trên trung bình toàn tập.
    """),
    code("""
    sample_row = splits["validation"].iloc[0]
    clean = load_rgb(ROOT / sample_row.relative_path, 224)
    for kind in METHODS:
        outputs = [apply_degradation(clean, kind, level,
                   stable_seed(sample_row.relative_path, kind, level)) for level in LEVELS]
        for output in outputs:
            assert output.shape == clean.shape and output.dtype == np.uint8
            assert 0 <= output.min() <= output.max() <= 255
        print(kind, [round(full_reference_metrics(clean, x)["ssim"], 3) for x in outputs])
    """),
    code("""
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    for row_idx, kind in enumerate(METHODS):
        axes[row_idx, 0].imshow(clean); axes[row_idx, 0].set_title(f"{kind}: clean")
        for col_idx, level in enumerate(LEVELS, 1):
            x = apply_degradation(clean, kind, level,
                stable_seed(sample_row.relative_path, kind, level))
            axes[row_idx, col_idx].imshow(x); axes[row_idx, col_idx].set_title(level)
        for ax in axes[row_idx]: ax.axis("off")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "degradation" / "degradation_grid.png", dpi=200)
    plt.show()
    """),
    md("""
    # Phần 6 — Enhancement

    Mỗi suy giảm chỉ được ghép với các phương pháp có cơ sở: gamma/CLAHE cho thiếu sáng, lọc trơn cho
    Gaussian noise, median cho impulse noise, làm sắc nét cho blur và ba không gian màu cho color cast.
    Clean chỉ dùng tính metric, không được truyền vào thuật toán hiệu chỉnh lúc inference.
    """),
    code("""
    for kind, methods in METHODS.items():
        degraded = apply_degradation(clean, kind, "strong",
            stable_seed(sample_row.relative_path, kind, "strong"))
        for method in methods:
            enhanced = apply_enhancement(degraded, kind, "strong", method)
            assert enhanced.shape == clean.shape and enhanced.dtype == np.uint8
            assert np.isfinite(enhanced).all() and 0 <= enhanced.min() <= enhanced.max() <= 255
        print(f"PASS {kind}: {', '.join(methods)}")
    """),
    code("""
    examples = []
    for kind, methods in METHODS.items():
        degraded = apply_degradation(clean, kind, "strong",
            stable_seed(sample_row.relative_path, kind, "strong"))
        examples.append((kind, clean, degraded,
                         apply_enhancement(degraded, kind, "strong", methods[0])))
    fig, axes = plt.subplots(5, 3, figsize=(10, 15))
    for row_idx, (kind, a, b, c) in enumerate(examples):
        for ax, image, title in zip(axes[row_idx], (a, b, c), ("Clean", "Degraded", METHODS[kind][0])):
            ax.imshow(image); ax.set_title(f"{kind} — {title}"); ax.axis("off")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "enhancement" / "enhancement_grid.png", dpi=200)
    plt.show()
    """),
    md("""
    **Kiểm soát chất lượng.** Cảnh báo cháy sáng khi tỷ lệ pixel ≥250 tăng mạnh; cảnh báo mất chi tiết
    khi năng lượng biên giảm sâu. Enhancement có thể làm PSNR thấp hơn nhưng Macro F1 cao hơn vì metric
    nhận diện và metric cảm nhận đo hai khía cạnh khác nhau.
    """),
    md("""
    # Phần 7 — Chọn tham số bằng Validation

    Tham số được chọn bằng Macro F1 trên Validation, SSIM chỉ phá hòa. Không một ảnh Test nào tham gia.
    Sau khi chọn, cấu hình được khóa ở `configs/locked_enhancement_params.json` và dùng nguyên trạng cho
    toàn bộ 49 điều kiện Test.
    """),
    code("""
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    validation_df = pd.read_csv(SPLIT_DIR / "validation.csv")
    test_df = pd.read_csv(SPLIT_DIR / "test.csv")
    train_ds = build_tf_dataset(train_df, ROOT, CLASS_NAMES, training=True)
    validation_ds = build_tf_dataset(validation_df, ROOT, CLASS_NAMES)
    print({"train": len(train_df), "validation": len(validation_df), "test": len(test_df)})
    """),
    md("""
    Việc tuning được thực hiện **sau khi có model clean cố định** ở Phần 10. Ô dưới được chạy tại Phần 10
    theo thứ tự notebook bằng biến `model_service`; để giữ mạch đọc, grid và nguyên tắc được khai báo ở đây.
    """),
    code("""
    LOCKED_CONFIG = ROOT / "configs" / "locked_enhancement_params.json"
    print("Grid tuning nằm trong src/tuning.py; đầu ra khóa:", LOCKED_CONFIG)
    """),
    md("""
    # Phần 8 — MobileNetV2

    Kiến trúc dùng ImageNet weights, `include_top=False`, Global Average Pooling, Dropout 0,3 và Dense
    softmax năm lớp. `preprocess_input` được đặt trong model để notebook và Streamlit dùng cùng quy tắc.
    MobileNetV2 có inverted residual và linear bottleneck, phù hợp mục tiêu deploy nhẹ.
    """),
    code("""
    parameter_table = pd.DataFrame({
        "Kiến trúc": ["MobileNetV2 + GAP + Dropout + Dense"],
        "Khởi tạo": ["ImageNet"],
        "Input contract": ["EXIF → RGB → letterbox LANCZOS 224×224"],
        "Output": [len(CLASS_NAMES)]})
    display(parameter_table)
    """),
    md("""
    # Phần 9 — Huấn luyện giai đoạn 1

    Backbone bị đóng băng; chỉ đầu phân loại học với learning rate 1e-3. Augmentation chỉ gồm lật,
    xoay, zoom và tịnh tiến hình học trên Train. EarlyStopping, ReduceLROnPlateau và ModelCheckpoint
    cùng giám sát `val_loss` để giữ checkpoint tốt nhất.
    """),
    code("""
    MODEL_PATH = MODEL_DIR / "mobilenetv2_flowers.keras"
    HISTORY_PATH = ROOT / "artifacts" / "training" / "history.csv"
    trained_model, history = train_two_stage(
        train_ds, validation_ds, MODEL_PATH,
        head_epochs=2 if RUN_MODE == "QUICK_RUN" else 15,
        fine_tune_epochs=1 if RUN_MODE == "QUICK_RUN" else 10,
        history_path=HISTORY_PATH)
    assert MODEL_PATH.exists()
    display(history.tail())
    """),
    md("""
    # Phần 10 — Fine-tuning

    Trong giai đoạn hai, chỉ 30 lớp cuối được mở; BatchNormalization vẫn đóng băng và learning rate giảm
    xuống 1e-5. Đồ thị loss/accuracy được dùng phát hiện overfitting: nếu train tiếp tục tốt lên trong khi
    validation xấu đi, checkpoint trước đó được giữ thay vì chọn epoch cuối.
    """),
    code("""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    history.plot(x="epoch", y=["loss", "val_loss"], ax=axes[0], title="Loss")
    history.plot(x="epoch", y=["accuracy", "val_accuracy"], ax=axes[1], title="Accuracy")
    for ax in axes: ax.axvline(15.5, ls="--", color="#F79646", label="Fine-tune")
    curves_path = ROOT / "artifacts" / "training" / "learning_curves.png"
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(curves_path, dpi=200)
    plt.show()
    """),
    code("""
    model_service = ModelService(MODEL_PATH)
    validation_images = [load_rgb(ROOT / path) for path in validation_df.relative_path]
    tuning_all, tuning_best, locked_params = tune_on_validation(
        model_service, validation_images, validation_df.relative_path.tolist(),
        validation_df.label.tolist())
    save_tuning(tuning_all, tuning_best, locked_params, RESULT_DIR, LOCKED_CONFIG,
        metadata={"selection_split": "validation", "selection_metric": "macro_f1",
                  "tie_break_metric": "ssim", "seed": SEED, "run_mode": RUN_MODE})
    display(tuning_best[["degradation", "level", "method", "params", "macro_f1", "ssim"]])
    """),
    md("""
    # Phần 11 — Clean baseline

    Clean baseline đo trần tham chiếu của model trên Test. Báo cáo gồm metric tổng hợp, metric từng lớp,
    confusion matrix và ví dụ đúng/sai. Đây là cùng một checkpoint sẽ được dùng cho mọi biến thể degraded
    và enhanced; không fine-tune lại sau khi xem Test.
    """),
    code("""
    test_images = [load_rgb(ROOT / path) for path in test_df.relative_path]
    clean_outputs = model_service.predict_batch(test_images)
    y_true = np.array([CLASS_NAMES.index(x) for x in test_df.label])
    y_pred = np.array([CLASS_NAMES.index(x["label"]) for x in clean_outputs])
    clean_report = pd.DataFrame(classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)).T
    clean_report.to_csv(RESULT_DIR / "clean_classification_report.csv")
    display(clean_report)
    """),
    code("""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Dự đoán"); plt.ylabel("Nhãn thật"); plt.title("Confusion matrix — clean Test")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "classification" / "confusion_matrix_clean.png", dpi=200)
    plt.show()
    """),
    md("""
    # Phần 12 — Đánh giá 49 điều kiện

    Test gồm 1 clean, 15 degraded và 33 enhanced. Noise của mỗi ảnh tái lập bằng seed ổn định. Hàm đánh
    giá assert cả số dòng và khóa duy nhất; cùng `model_service` chứng minh model chỉ được load một lần.
    """),
    code("""
    evaluation = evaluate_full_experiment(
        model_service, test_images, test_df.relative_path.tolist(), test_df.label.tolist(),
        hashes=test_df.sha256.tolist(), locked_params=locked_params,
        bootstrap_samples=200 if RUN_MODE == "QUICK_RUN" else 2000,
        latency_runs=1 if RUN_MODE == "QUICK_RUN" else 5)
    results_49 = evaluation.condition_metrics
    expected_columns = {"accuracy", "macro_precision", "macro_recall", "macro_f1",
                        "weighted_f1", "psnr", "ssim", "delta_e_2000"}
    assert len(results_49) == 49 and expected_columns.issubset(results_49.columns)
    assert not results_49.duplicated(
        ["image_type", "degradation", "level", "enhancement_method"]).any()
    final_manifest = save_evaluation_artifacts(
        evaluation, RESULT_DIR / "final", model_path=MODEL_PATH,
        split_path=SPLIT_DIR / "test.csv",
        metadata={"seed": SEED, "run_mode": RUN_MODE, "selection_split": "validation"})
    display(results_49.head())
    """),
    md("""
    # Phần 13 — Phân tích kết quả

    Macro F1 là metric chính vì phân bố lớp không hoàn toàn cân bằng. Phân tích báo cáo chênh lệch tuyệt
    đối, tỷ lệ phục hồi so với clean, xu hướng theo mức và tương quan Pearson/Spearman. Tương quan chỉ là
    liên hệ thống kê, không được diễn giải thành quan hệ nhân quả.
    """),
    code("""
    degraded_rows = results_49[results_49.image_type.eq("degraded")]
    enhanced_rows = results_49[results_49.image_type.eq("enhanced")]
    clean_f1 = float(results_49.loc[results_49.image_type.eq("clean"), "macro_f1"].iloc[0])
    merged = enhanced_rows.merge(degraded_rows,
        on=["degradation", "level"], suffixes=("_enh", "_deg"))
    merged["macro_f1_gain"] = merged.macro_f1_enh - merged.macro_f1_deg
    merged["recovery_ratio"] = merged.macro_f1_gain / (clean_f1 - merged.macro_f1_deg + 1e-12)
    display(merged.sort_values("macro_f1_gain", ascending=False)[
        ["degradation", "level", "enhancement_method_enh", "macro_f1_deg",
         "macro_f1_enh", "macro_f1_gain", "recovery_ratio"]])
    """),
    code("""
    numeric = results_49[["psnr", "ssim", "delta_e_2000", "accuracy", "macro_f1"]]
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(pearson, annot=True, vmin=-1, vmax=1, cmap="vlag", ax=axes[0]); axes[0].set_title("Pearson")
    sns.heatmap(spearman, annot=True, vmin=-1, vmax=1, cmap="vlag", ax=axes[1]); axes[1].set_title("Spearman")
    plt.tight_layout(); plt.savefig(FIGURE_DIR / "classification" / "correlation_heatmap.png", dpi=200)
    plt.show()
    """),
    code("""
    summary_plot = results_49[results_49.image_type.ne("clean")].copy()
    summary_plot["condition"] = summary_plot["degradation"] + "/" + summary_plot["level"]
    plt.figure(figsize=(13, 5)); sns.pointplot(summary_plot, x="condition", y="macro_f1",
        hue="image_type", errorbar=None); plt.xticks(rotation=45, ha="right")
    plt.title("Macro F1 theo điều kiện"); plt.tight_layout()
    plt.savefig(FIGURE_DIR / "classification" / "macro_f1_comparison.png", dpi=200); plt.show()
    """),
    md("""
    # Phần 14 — Phân tích lỗi

    Bốn nhóm được truy vết theo cùng ảnh: clean đúng–degraded sai; degraded sai–enhanced đúng; enhanced
    vẫn sai; enhancement làm xấu dự đoán. Cần mô tả texture, biên, màu, nền, góc chụp và confidence,
    tránh chỉ đếm lỗi mà không giải thích dấu hiệu thị giác.
    """),
    code("""
    # Ví dụ cặp quan trọng: suy giảm mạnh và phương pháp đã khóa tốt nhất.
    error_records = build_error_analysis(
        model_service=model_service, clean_images=test_images,
        relative_paths=test_df.relative_path.tolist(), labels=test_df.label.tolist(),
        locked_params=locked_params, max_examples_per_group=20)
    error_records.to_csv(RESULT_DIR / "final" / "error_analysis.csv", index=False)
    display(error_records.groupby(["degradation", "level", "error_group"]).size()
            .rename("count").reset_index().head(20))
    """),
    md("""
    # Phần 15 — Xuất sản phẩm

    Model, thứ tự lớp, metadata, tuning, bảng kết quả và manifest được lưu cùng nhau. Streamlit đọc đúng
    các artifact này; nếu thiếu model/config, ứng dụng dừng dự đoán thay vì dùng mô hình thay thế.
    """),
    code("""
    metadata = {
        "architecture": "MobileNetV2", "input_size": [224, 224, 3],
        "class_names": CLASS_NAMES,
        "preprocess": "EXIF transpose + RGB + letterbox LANCZOS 224 + in-graph MobileNetV2 preprocess_input",
        "seed": SEED, "tensorflow": tf.__version__, "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
        "split_counts": {name: len(frame) for name, frame in splits.items()},
        "validation_config": str(LOCKED_CONFIG.relative_to(ROOT)),
        "status": "FULL_RUN_COMPLETE" if RUN_MODE == "FULL_RUN" else "QUICK_RUN_ONLY_NOT_FOR_REPORT",
        "model_sha256": final_manifest["model_sha256"],
    }
    (MODEL_DIR / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_manifest, indent=2))
    """),
    md("""
    # Phần 16 — Kiểm tra cuối

    Notebook chỉ hoàn tất khi mọi assert bên dưới đều qua. Sau Run All, cập nhật báo cáo/slide từ đúng
    CSV sinh ra; không chép số thủ công từ một lần chạy khác.
    """),
    code("""
    assert MODEL_PATH.exists(), "Thiếu model Keras"
    assert RUN_MODE == "FULL_RUN", "QUICK_RUN chỉ dùng smoke test, không dùng cho bản nộp"
    assert_no_leakage(splits)
    assert len(results_49) == 49
    assert np.isfinite(results_49[["accuracy", "macro_f1", "weighted_f1", "ssim"]]).all().all()
    assert json.loads((MODEL_DIR / "class_names.json").read_text()) == CLASS_NAMES
    assert metadata["class_names"] == CLASS_NAMES
    checklist = {
        "model_keras": True, "split_no_leakage": True, "conditions_49": True,
        "metrics_finite": True, "class_order_consistent": True,
        "model_load_count": model_service.load_count, "full_run": True,
        "status": "FULL_RUN_COMPLETE", "submission_gate": "SUBMISSION_READY",
    }
    print("HOÀN TẤT RUN ALL\\n", json.dumps(checklist, ensure_ascii=False, indent=2))
    """),
    md("""
    ## Kết luận sau Run All

    Điền kết luận bằng số từ `results/final/condition_metrics.csv`: suy giảm làm giảm Macro F1
    bao nhiêu; phương pháp nào phục hồi tốt cho từng dạng; trường hợp nào enhancement gây hại; lớp nào nhạy
    nhất. Không kết luận vượt quá năm lớp, bộ dữ liệu và các suy giảm tổng hợp đã khảo sát.
    """),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "BTL_XuLyAnh_NhanDienHoa.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


if __name__ == "__main__":
    output = ROOT / "BTL_XuLyAnh_NhanDienHoa.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {output} with {len(cells)} cells")
