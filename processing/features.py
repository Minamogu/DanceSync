from dataclasses import dataclass
import math

import numpy as np

from processing.pose_estimation import Landmark, PoseFrame


FEATURE_GROUPS: tuple[tuple[str, int, float], ...] = (
    ("angles", 4, 1.0),
    ("arm_directions", 6, 1.0),
    ("leg_directions", 6, 1.0),
    ("shoulder_hip_relation", 3, 1.0),
)


@dataclass(frozen=True)
class PoseFeature:
    """DTWの距離計算に使う正規化済み骨格特徴量。"""

    values: tuple[float, ...]
    weights: tuple[float, ...]


def _landmark_map(pose_frame: PoseFrame) -> dict[str, Landmark] | None:
    if pose_frame.landmarks is None:
        return None
    landmarks = {landmark.name: landmark for landmark in pose_frame.landmarks}
    required_names = {
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
    }
    if not required_names.issubset(landmarks):
        return None
    return landmarks


def _point(landmark: Landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y, landmark.z], dtype=float)


def _unit_vector(start: Landmark, end: Landmark) -> np.ndarray | None:
    vector = _point(end) - _point(start)
    length = float(np.linalg.norm(vector))
    if length == 0:
        return None
    return vector / length


def _angle(first: Landmark, vertex: Landmark, last: Landmark) -> float | None:
    first_vector = _point(first) - _point(vertex)
    last_vector = _point(last) - _point(vertex)
    first_length = float(np.linalg.norm(first_vector))
    last_length = float(np.linalg.norm(last_vector))
    if first_length == 0 or last_length == 0:
        return None
    cosine = float(
        np.dot(first_vector, last_vector) / (first_length * last_length)
    )
    cosine = max(-1.0, min(1.0, cosine))
    return math.acos(cosine) / math.pi


def extract_pose_feature(
    pose_frame: PoseFrame,
    min_visibility: float = 0.5,
) -> PoseFeature | None:
    """1フレームから撮影距離と体格差を抑えた特徴量を抽出する。"""
    if not 0 <= min_visibility <= 1:
        raise ValueError("min_visibilityは0から1の範囲で指定してください。")

    landmarks = _landmark_map(pose_frame)
    if landmarks is None:
        return None
    required_landmarks = [
        landmarks[name]
        for name in (
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_ELBOW",
            "RIGHT_ELBOW",
            "LEFT_WRIST",
            "RIGHT_WRIST",
            "LEFT_HIP",
            "RIGHT_HIP",
            "LEFT_KNEE",
            "RIGHT_KNEE",
            "LEFT_ANKLE",
            "RIGHT_ANKLE",
        )
    ]
    if any(
        landmark.visibility < min_visibility
        for landmark in required_landmarks
    ):
        return None

    left_shoulder = landmarks["LEFT_SHOULDER"]
    right_shoulder = landmarks["RIGHT_SHOULDER"]
    left_hip = landmarks["LEFT_HIP"]
    right_hip = landmarks["RIGHT_HIP"]
    shoulder_center = (_point(left_shoulder) + _point(right_shoulder)) / 2
    hip_center = (_point(left_hip) + _point(right_hip)) / 2
    shoulder_width = float(np.linalg.norm(_point(left_shoulder) - _point(right_shoulder)))
    hip_width = float(np.linalg.norm(_point(left_hip) - _point(right_hip)))
    torso_scale = max((shoulder_width + hip_width) / 2, 1e-6)

    angles = (
        _angle(left_shoulder, landmarks["LEFT_ELBOW"], landmarks["LEFT_WRIST"]),
        _angle(right_shoulder, landmarks["RIGHT_ELBOW"], landmarks["RIGHT_WRIST"]),
        _angle(left_hip, landmarks["LEFT_KNEE"], landmarks["LEFT_ANKLE"]),
        _angle(right_hip, landmarks["RIGHT_KNEE"], landmarks["RIGHT_ANKLE"]),
    )
    arm_directions = (
        _unit_vector(left_shoulder, landmarks["LEFT_WRIST"]),
        _unit_vector(right_shoulder, landmarks["RIGHT_WRIST"]),
    )
    leg_directions = (
        _unit_vector(left_hip, landmarks["LEFT_ANKLE"]),
        _unit_vector(right_hip, landmarks["RIGHT_ANKLE"]),
    )
    shoulder_hip_relation = (hip_center - shoulder_center) / torso_scale

    if any(value is None for value in angles + arm_directions + leg_directions):
        return None

    values_array = np.concatenate(
        [
            np.asarray(angles, dtype=float),
            np.concatenate([value for value in arm_directions if value is not None]),
            np.concatenate([value for value in leg_directions if value is not None]),
            shoulder_hip_relation,
        ]
    )
    weights: list[float] = []
    for _, size, group_weight in FEATURE_GROUPS:
        weights.extend([group_weight / size] * size)

    return PoseFeature(
        values=tuple(float(value) for value in values_array),
        weights=tuple(weights),
    )


def weighted_feature_distance(
    reference: PoseFeature,
    practice: PoseFeature,
) -> float:
    """グループ内の次元数を正規化した重み付きユークリッド距離。"""
    if len(reference.values) != len(practice.values):
        raise ValueError("特徴量の次元数が一致していません。")
    differences = np.asarray(reference.values) - np.asarray(practice.values)
    weights = np.asarray(reference.weights)
    return float(np.sqrt(np.sum(weights * differences * differences)))
