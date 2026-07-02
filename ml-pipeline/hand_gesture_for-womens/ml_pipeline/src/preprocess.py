from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_engineering import build_feature_vector, normalize_landmarks
from src.utils import ensure_dir, load_config, save_json, set_seed

mp_hands = mp.solutions.hands


def iter_images(class_dir: Path) -> Iterator[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        yield from class_dir.glob(ext)


def extract_landmarks_from_image(
    image_bgr: np.ndarray,
    hands: mp.solutions.hands.Hands,
) -> np.ndarray | None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None

    hand_landmarks = result.multi_hand_landmarks[0]
    features: list[float] = []

    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])

    return np.asarray(features, dtype=np.float32)


def run_preprocessing() -> None:
    config = load_config()
    set_seed(int(config["training"]["random_seed"]))

    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    reports_dir = Path(config["data"]["reports_dir"])

    ensure_dir(processed_dir)
    ensure_dir(reports_dir)

    classes: list[str] = config["data"]["classes"]
    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    records: list[dict[str, str]] = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=float(config["data"].get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(config["data"].get("min_tracking_confidence", 0.5)),
    ) as hands:
        for label_index, class_name in enumerate(classes):
            class_dir = raw_dir / class_name

            if not class_dir.exists():
                records.append(
                    {
                        "file": str(class_dir),
                        "class": class_name,
                        "status": "missing_class_directory",
                    }
                )
                continue

            for image_path in iter_images(class_dir):
                image = cv2.imread(str(image_path))

                if image is None:
                    records.append(
                        {
                            "file": str(image_path),
                            "class": class_name,
                            "status": "unreadable_image",
                        }
                    )
                    continue

                raw_landmarks = extract_landmarks_from_image(image, hands)

                if raw_landmarks is None:
                    records.append(
                        {
                            "file": str(image_path),
                            "class": class_name,
                            "status": "failed_detection",
                        }
                    )
                    continue

                normalized = normalize_landmarks(raw_landmarks)
                feature_vector = build_feature_vector(normalized)

                all_features.append(feature_vector)
                all_labels.append(label_index)

                records.append(
                    {
                        "file": str(image_path),
                        "class": class_name,
                        "status": "ok",
                    }
                )

    if not all_features:
        raise RuntimeError(
            "No valid landmarks extracted. Check dataset path, class folder names, and MediaPipe detection."
        )

    X = np.asarray(all_features, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.int32)

    train_ratio = float(config["data"]["train_ratio"])
    val_ratio = float(config["data"]["val_ratio"])
    test_ratio = float(config["data"]["test_ratio"])

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - train_ratio),
        random_state=int(config["training"]["random_seed"]),
        stratify=y,
    )

    val_portion = val_ratio / (val_ratio + test_ratio)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_portion),
        random_state=int(config["training"]["random_seed"]),
        stratify=y_temp,
    )

    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_val.npy", X_val)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_test.npy", y_test)

    pd.DataFrame(records).to_csv(reports_dir / "preprocessing_log.csv", index=False)

    save_json(
        {
            "num_train": int(len(X_train)),
            "num_val": int(len(X_val)),
            "num_test": int(len(X_test)),
            "num_classes": int(len(classes)),
            "feature_dim": int(X.shape[1]),
            "raw_landmark_dim": 63,
        },
        reports_dir / "preprocessing_summary.json",
    )

    print("Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()