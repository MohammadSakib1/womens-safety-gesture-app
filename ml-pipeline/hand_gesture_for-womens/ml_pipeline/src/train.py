# from __future__ import annotations

# from pathlib import Path

# import numpy as np
# import tensorflow as tf
# from sklearn.utils.class_weight import compute_class_weight

# from src.losses import SparseCategoricalFocalLoss
# from src.model import build_resmlp_feature_fusion_model
# from src.utils import ensure_dir, load_config, set_seed


# def run_training() -> None:
#     config = load_config()
#     set_seed(int(config["training"]["random_seed"]))

#     processed_dir = Path(config["data"]["processed_dir"])
#     checkpoints_dir = Path("outputs/checkpoints")
#     metrics_dir = Path("outputs/metrics")

#     ensure_dir(checkpoints_dir)
#     ensure_dir(metrics_dir)

#     X_train = np.load(processed_dir / "X_train.npy")
#     y_train = np.load(processed_dir / "y_train.npy")
#     X_val = np.load(processed_dir / "X_val.npy")
#     y_val = np.load(processed_dir / "y_val.npy")

#     input_dim = int(X_train.shape[1])

#     model = build_resmlp_feature_fusion_model(
#         input_dim=input_dim,
#         raw_dim=63,
#         num_classes=len(config["data"]["classes"]),
#         dropout_rate=float(config["training"]["dropout"]),
#         l2_weight=float(config["training"]["l2"]),
#     )

#     if config["training"].get("use_focal_loss", False):
#         loss_fn = SparseCategoricalFocalLoss()
#     else:
#         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

#     optimizer = tf.keras.optimizers.AdamW(
#         learning_rate=float(config["training"]["learning_rate"]),
#         weight_decay=float(config["training"]["weight_decay"]),
#     )

#     model.compile(
#         optimizer=optimizer,
#         loss=loss_fn,
#         metrics=["accuracy"],
#     )

#     class_weights = None
#     if config["training"].get("class_weight", True):
#         classes = np.unique(y_train)
#         weights = compute_class_weight(
#             class_weight="balanced",
#             classes=classes,
#             y=y_train,
#         )
#         class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(
#             monitor="val_loss",
#             patience=10,
#             restore_best_weights=True,
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor="val_loss",
#             factor=0.5,
#             patience=4,
#             min_lr=1e-6,
#         ),
#         tf.keras.callbacks.ModelCheckpoint(
#             filepath=str(checkpoints_dir / "best_model.keras"),
#             monitor="val_accuracy",
#             save_best_only=True,
#             verbose=1,
#         ),
#         tf.keras.callbacks.CSVLogger(str(metrics_dir / "training_log.csv")),
#     ]

#     model.fit(
#         X_train,
#         y_train,
#         validation_data=(X_val, y_val),
#         epochs=int(config["training"]["epochs"]),
#         batch_size=int(config["training"]["batch_size"]),
#         class_weight=class_weights,
#         callbacks=callbacks,
#         verbose=1,
#     )

#     model.save(checkpoints_dir / "final_model.keras")
#     print("Training complete.")


# if __name__ == "__main__":
#     run_training()


from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.losses import SparseCategoricalFocalLoss
from src.model import build_resmlp_feature_fusion_model
from src.utils import ensure_dir, load_config, set_seed


def run_training() -> None:
    config = load_config()
    set_seed(int(config["training"]["random_seed"]))

    processed_dir = Path(config["data"]["processed_dir"])
    checkpoints_dir = Path("outputs/checkpoints")
    metrics_dir = Path("outputs/metrics")

    ensure_dir(checkpoints_dir)
    ensure_dir(metrics_dir)

    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")

    input_dim = int(X_train.shape[1])

    model = build_resmlp_feature_fusion_model(
        input_dim=input_dim,
        raw_dim=63,
        num_classes=len(config["data"]["classes"]),
        dropout_rate=float(config["training"]["dropout"]),
        l2_weight=float(config["training"]["l2"]),
    )

    if config["training"].get("use_focal_loss", False):
        loss_fn = SparseCategoricalFocalLoss()
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            label_smoothing=float(config["training"]["label_smoothing"])
        )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    class_weights = None
    if config["training"].get("class_weight", True):
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoints_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(metrics_dir / "training_log.csv")),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(checkpoints_dir / "final_model.keras")
    print("Training complete.")


if __name__ == "__main__":
    run_training()