from dataclasses import dataclass
import math
from statistics import median

import numpy as np

from processing.angles import calculate_frame_angles
from processing.pose_estimation import Landmark, PoseFrame


@dataclass(frozen=True)
class MetricValue:
    """対応フレームにおける1つの評価指標。"""

    name: str
    reference: float
    practice: float
    difference: float
    absolute_difference: float


@dataclass(frozen=True)
class FrameMetrics:
    """対応フレームの評価指標一式。"""

    values: tuple[MetricValue, ...]


def _landmarks(pose_frame: PoseFrame) -> dict[str, Landmark] | None:
    if pose_frame.landmarks is None:
        return None
    return {landmark.name: landmark for landmark in pose_frame.landmarks}


def _point(landmark: Landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y, landmark.z], dtype=float)


def _distance(first: Landmark, second: Landmark) -> float:
    return float(np.linalg.norm(_point(first) - _point(second)))


def _required_visible(
    landmarks: dict[str, Landmark],
    names: tuple[str, ...],
    min_visibility: float,
) -> bool:
    return all(
        name in landmarks and landmarks[name].visibility >= min_visibility
        for name in names
    )


def _leg_length(landmarks: dict[str, Landmark], side: str) -> float:
    return (
        _distance(landmarks[f"{side}_HIP"], landmarks[f"{side}_KNEE"])
        + _distance(landmarks[f"{side}_KNEE"], landmarks[f"{side}_ANKLE"])
    )


def representative_leg_length(
    frames: tuple[PoseFrame, ...],
    min_visibility: float = 0.25,
) -> float | None:
    """動画内の有効フレームから代表脚長を求める。"""
    lengths: list[float] = []
    seen_frame_indices: set[int] = set()
    required = (
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
    )
    for frame in frames:
        if frame.frame_index in seen_frame_indices:
            continue
        seen_frame_indices.add(frame.frame_index)
        landmarks = _landmarks(frame)
        if landmarks is None or not _required_visible(
            landmarks,
            required,
            min_visibility,
        ):
            continue
        left_length = _leg_length(landmarks, "LEFT")
        right_length = _leg_length(landmarks, "RIGHT")
        if left_length > 0 and right_length > 0:
            lengths.append((left_length + right_length) / 2)
    return median(lengths) if lengths else None


def _metric(
    name: str,
    reference: float | None,
    practice: float | None,
) -> MetricValue | None:
    if reference is None or practice is None:
        return None
    difference = practice - reference
    return MetricValue(
        name=name,
        reference=reference,
        practice=practice,
        difference=difference,
        absolute_difference=abs(difference),
    )


def calculate_frame_metrics(
    reference_frame: PoseFrame,
    practice_frame: PoseFrame,
    reference_leg_length: float | None,
    practice_leg_length: float | None,
    min_visibility: float = 0.25,
) -> FrameMetrics:
    """DTW対応フレームの角度・スタンス・腰高さを計算する。"""
    reference_angles = {
        angle.name: angle.degrees
        for angle in calculate_frame_angles(
            reference_frame,
            min_visibility=min_visibility,
        )
    }
    practice_angles = {
        angle.name: angle.degrees
        for angle in calculate_frame_angles(
            practice_frame,
            min_visibility=min_visibility,
        )
    }
    values: list[MetricValue] = []
    for name in (
        "左肩",
        "右肩",
        "左肘",
        "右肘",
        "左股関節",
        "右股関節",
        "左膝",
        "右膝",
    ):
        value = _metric(name, reference_angles.get(name), practice_angles.get(name))
        if value is not None:
            values.append(value)

    reference_landmarks = _landmarks(reference_frame)
    practice_landmarks = _landmarks(practice_frame)
    required = (
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
    )
    if (
        reference_landmarks is not None
        and practice_landmarks is not None
        and _required_visible(reference_landmarks, required, min_visibility)
        and _required_visible(practice_landmarks, required, min_visibility)
    ):
        reference_hip_width = abs(
            reference_landmarks["LEFT_HIP"].x
            - reference_landmarks["RIGHT_HIP"].x
        )
        practice_hip_width = abs(
            practice_landmarks["LEFT_HIP"].x
            - practice_landmarks["RIGHT_HIP"].x
        )
        if reference_hip_width > 1e-4 and practice_hip_width > 1e-4:
            reference_stance = abs(
                reference_landmarks["LEFT_ANKLE"].x
                - reference_landmarks["RIGHT_ANKLE"].x
            ) / reference_hip_width
            practice_stance = abs(
                practice_landmarks["LEFT_ANKLE"].x
                - practice_landmarks["RIGHT_ANKLE"].x
            ) / practice_hip_width
            stance = _metric(
                "スタンスの広さ",
                reference_stance,
                practice_stance,
            )
            if stance is not None:
                values.append(stance)

    if (
        reference_landmarks is not None
        and practice_landmarks is not None
        and reference_leg_length
        and practice_leg_length
        and _required_visible(
            reference_landmarks,
            required + ("LEFT_KNEE", "RIGHT_KNEE"),
            min_visibility,
        )
        and _required_visible(
            practice_landmarks,
            required + ("LEFT_KNEE", "RIGHT_KNEE"),
            min_visibility,
        )
    ):
        def waist_height(
            landmarks: dict[str, Landmark],
            leg_length: float,
        ) -> float:
            hip_center = (
                _point(landmarks["LEFT_HIP"])
                + _point(landmarks["RIGHT_HIP"])
            ) / 2
            foot_center = (
                _point(landmarks["LEFT_ANKLE"])
                + _point(landmarks["RIGHT_ANKLE"])
            ) / 2
            return abs(float(foot_center[1] - hip_center[1])) / leg_length

        height = _metric(
            "腰の高さ",
            waist_height(reference_landmarks, reference_leg_length),
            waist_height(practice_landmarks, practice_leg_length),
        )
        if height is not None and math.isfinite(height.difference):
            values.append(height)

    return FrameMetrics(values=tuple(values))
