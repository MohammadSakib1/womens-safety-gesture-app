# from __future__ import annotations

# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import tensorflow as tf
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
# )

# from src.model import AttentionGate
# from src.utils import ensure_dir, load_config, save_json


# def run_evaluation() -> None:
#     config = load_config()

#     processed_dir = Path(config["data"]["processed_dir"])
#     metrics_dir = Path("outputs/metrics")
#     figures_dir = Path("outputs/figures")

#     ensure_dir(metrics_dir)
#     ensure_dir(figures_dir)

#     X_test = np.load(processed_dir / "X_test.npy")
#     y_test = np.load(processed_dir / "y_test.npy")

#     model = tf.keras.models.load_model(
#         "outputs/checkpoints/best_model.keras",
#         compile=False,
#         safe_mode=False,
#         custom_objects={"AttentionGate": AttentionGate},
#     )

#     y_prob = model.predict(X_test, verbose=0)
#     y_pred = np.argmax(y_prob, axis=1)

#     metrics = {
#         "accuracy": float(accuracy_score(y_test, y_pred)),
#         "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
#         "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
#         "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
#     }

#     save_json(metrics, metrics_dir / "summary_metrics.json")

#     report = classification_report(
#         y_test,
#         y_pred,
#         target_names=config["data"]["classes"],
#         output_dict=True,
#         zero_division=0,
#     )
#     pd.DataFrame(report).transpose().to_csv(metrics_dir / "classification_report.csv")

#     cm = confusion_matrix(y_test, y_pred)

#     plt.figure(figsize=(7, 6))
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt="d",
#         cmap="Blues",
#         xticklabels=config["data"]["classes"],
#         yticklabels=config["data"]["classes"],
#     )
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.savefig(figures_dir / "confusion_matrix.png", dpi=200)
#     plt.close()

#     print(metrics)
#     print("Evaluation complete.")


# if __name__ == "__main__":
#     run_evaluation()


from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.model import AttentionGate, SliceLayer
from src.utils import ensure_dir, load_config, save_json


def run_evaluation() -> None:
    config = load_config()

    processed_dir = Path(config["data"]["processed_dir"])
    metrics_dir = Path("outputs/metrics")
    figures_dir = Path("outputs/figures")

    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    model = tf.keras.models.load_model(
        "outputs/checkpoints/best_model.keras",
        compile=False,
        custom_objects={
            "AttentionGate": AttentionGate,
            "SliceLayer": SliceLayer,
        },
    )

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }

    save_json(metrics, metrics_dir / "summary_metrics.json")

    report = classification_report(
        y_test,
        y_pred,
        target_names=config["data"]["classes"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(metrics_dir / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config["data"]["classes"],
        yticklabels=config["data"]["classes"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print(metrics)
    print("Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()