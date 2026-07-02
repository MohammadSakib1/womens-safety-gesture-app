# from __future__ import annotations

# import math
# import numpy as np

# LANDMARK_COUNT = 21

# WRIST = 0
# THUMB_CMC = 1
# THUMB_MCP = 2
# THUMB_IP = 3
# THUMB_TIP = 4

# INDEX_MCP = 5
# INDEX_PIP = 6
# INDEX_DIP = 7
# INDEX_TIP = 8

# MIDDLE_MCP = 9
# MIDDLE_PIP = 10
# MIDDLE_DIP = 11
# MIDDLE_TIP = 12

# RING_MCP = 13
# RING_PIP = 14
# RING_DIP = 15
# RING_TIP = 16

# PINKY_MCP = 17
# PINKY_PIP = 18
# PINKY_DIP = 19
# PINKY_TIP = 20


# DISTANCE_PAIRS = [
#     (WRIST, THUMB_TIP),
#     (WRIST, INDEX_TIP),
#     (WRIST, MIDDLE_TIP),
#     (WRIST, RING_TIP),
#     (WRIST, PINKY_TIP),
#     (THUMB_TIP, INDEX_TIP),
#     (INDEX_TIP, MIDDLE_TIP),
#     (MIDDLE_TIP, RING_TIP),
#     (RING_TIP, PINKY_TIP),
#     (THUMB_TIP, PINKY_TIP),
# ]


# ANGLE_TRIPLETS = [
#     (THUMB_MCP, THUMB_IP, THUMB_TIP),
#     (INDEX_MCP, INDEX_PIP, INDEX_TIP),
#     (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
#     (RING_MCP, RING_PIP, RING_TIP),
#     (PINKY_MCP, PINKY_PIP, PINKY_TIP),
# ]


# def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
#     coords = landmarks.reshape(LANDMARK_COUNT, 3).astype(np.float32)

#     wrist = coords[WRIST].copy()
#     coords -= wrist

#     max_val = np.max(np.abs(coords))
#     if max_val > 1e-6:
#         coords /= max_val

#     return coords.reshape(-1)


# def _distance(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.linalg.norm(a - b))


# def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
#     ba = a - b
#     bc = c - b

#     dot = np.dot(ba, bc)
#     norm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8

#     cos = dot / norm
#     cos = np.clip(cos, -1.0, 1.0)

#     return float(math.acos(cos) / math.pi)


# def extract_features(normalized_landmarks: np.ndarray) -> np.ndarray:
#     coords = normalized_landmarks.reshape(LANDMARK_COUNT, 3)

#     distances = [
#         _distance(coords[i], coords[j])
#         for i, j in DISTANCE_PAIRS
#     ]

#     angles = [
#         _angle(coords[i], coords[j], coords[k])
#         for i, j, k in ANGLE_TRIPLETS
#     ]

#     return np.array(distances + angles, dtype=np.float32)


# def build_feature_vector(normalized_landmarks: np.ndarray) -> np.ndarray:
#     geo = extract_features(normalized_landmarks)
#     return np.concatenate([normalized_landmarks, geo])




from __future__ import annotations

import math
import numpy as np

LANDMARK_COUNT = 21

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


DISTANCE_PAIRS = [
    (WRIST, THUMB_TIP),
    (WRIST, INDEX_TIP),
    (WRIST, MIDDLE_TIP),
    (WRIST, RING_TIP),
    (WRIST, PINKY_TIP),
    (THUMB_TIP, INDEX_TIP),
    (INDEX_TIP, MIDDLE_TIP),
    (MIDDLE_TIP, RING_TIP),
    (RING_TIP, PINKY_TIP),
    (THUMB_TIP, PINKY_TIP),
    (INDEX_MCP, PINKY_MCP),
    (THUMB_CMC, PINKY_MCP),
]

ANGLE_TRIPLETS = [
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
    (WRIST, INDEX_MCP, INDEX_TIP),
    (WRIST, MIDDLE_MCP, MIDDLE_TIP),
    (WRIST, RING_MCP, RING_TIP),
    (WRIST, PINKY_MCP, PINKY_TIP),
]


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    coords = landmarks.reshape(LANDMARK_COUNT, 3).astype(np.float32).copy()

    wrist = coords[WRIST].copy()
    coords -= wrist

    palm_size = np.linalg.norm(coords[MIDDLE_MCP] - coords[WRIST])
    if palm_size > 1e-6:
        coords /= palm_size

    return coords.reshape(-1).astype(np.float32)


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v) + 1e-8)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = _safe_norm(ba) * _safe_norm(bc)
    cosine = float(np.dot(ba, bc) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return float(math.acos(cosine) / math.pi)


def build_feature_vector(normalized_landmarks: np.ndarray) -> np.ndarray:
    coords = normalized_landmarks.reshape(LANDMARK_COUNT, 3)

    distances = [_distance(coords[i], coords[j]) for i, j in DISTANCE_PAIRS]
    angles = [_angle(coords[i], coords[j], coords[k]) for i, j, k in ANGLE_TRIPLETS]

    palm_width = _distance(coords[INDEX_MCP], coords[PINKY_MCP])
    palm_height = _distance(coords[WRIST], coords[MIDDLE_MCP])
    thumb_index_gap = _distance(coords[THUMB_TIP], coords[INDEX_TIP])
    index_pinky_gap = _distance(coords[INDEX_TIP], coords[PINKY_TIP])

    extra_features = [
        palm_width,
        palm_height,
        thumb_index_gap,
        index_pinky_gap,
        palm_width / (palm_height + 1e-8),
        thumb_index_gap / (palm_width + 1e-8),
    ]

    feature_vector = np.concatenate(
        [
            normalized_landmarks.astype(np.float32),
            np.asarray(distances, dtype=np.float32),
            np.asarray(angles, dtype=np.float32),
            np.asarray(extra_features, dtype=np.float32),
        ],
        axis=0,
    )

    return feature_vector.astype(np.float32)