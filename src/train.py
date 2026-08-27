from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .model import build_mobilenetv2, unfreeze_last_layers


def train_two_stage(
    train_ds,
    val_ds,
    model_path: Path,
    head_epochs=15,
    fine_tune_epochs=10,
    history_path: Path | None = None,
):
    import tensorflow as tf

    class EpochEvidence(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.started = 0.0
            self.durations: list[float] = []
            self.learning_rates: list[float] = []

        def on_epoch_begin(self, epoch, logs=None):
            self.started = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.durations.append(time.perf_counter() - self.started)
            self.learning_rates.append(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model, base = build_mobilenetv2()
    head_evidence = EpochEvidence()
    head_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        head_evidence,
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    first = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=head_callbacks)
    best_head_loss = min(first.history.get("val_loss", [float("inf")]))
    unfreeze_last_layers(base, 30)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    fine_evidence = EpochEvidence()
    fine_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            initial_value_threshold=best_head_loss,
        ),
        fine_evidence,
    ]
    second = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=fine_callbacks)
    head = pd.DataFrame(first.history)
    head["learning_rate"] = head_evidence.learning_rates
    head["duration_seconds"] = head_evidence.durations
    head.insert(0, "stage", "head")
    fine = pd.DataFrame(second.history)
    fine["learning_rate"] = fine_evidence.learning_rates
    fine["duration_seconds"] = fine_evidence.durations
    fine.insert(0, "stage", "fine_tune")
    history = pd.concat([head, fine], ignore_index=True)
    history.index += 1
    history.index.name = "epoch"
    history = history.reset_index()
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(history_path, index=False)
    return tf.keras.models.load_model(model_path), history
