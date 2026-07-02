from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.losses import SparseCategoricalFocalLoss
from src.model import AttentionGate, SliceLayer
from src.utils import ensure_dir, load_config


def run_export() -> None:
    config = load_config()

    export_dir = Path("outputs/exported")
    ensure_dir(export_dir)

    model = tf.keras.models.load_model(
        "outputs/checkpoints/best_model.keras",
        compile=False,
        safe_mode=False,
        custom_objects={
            "AttentionGate": AttentionGate,
            "SliceLayer": SliceLayer,
            "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
        },
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    with open(export_dir / "model.tflite", "wb") as f:
        f.write(tflite_model)

    with open(export_dir / "labels.txt", "w", encoding="utf-8") as f:
        for class_name in config["data"]["classes"]:
            f.write(f"{class_name}\n")

    print("TFLite model saved.")
    print("Export complete.")


if __name__ == "__main__":
    run_export()