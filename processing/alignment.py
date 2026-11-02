from dataclasses import dataclass

import numpy as np

from processing.features import (
    PoseFeature,
    extract_pose_feature,
    weighted_feature_distance,
)
from processing.pose_estimation import PoseFrame


@dataclass(frozen=True)
class AlignedPoseFrame:
    """DTWで対応付けられた2つの骨格フレーム。"""

    reference: PoseFrame
    practice: PoseFrame
    normalized_position: float
    feature_distance: float
    path_index: int


def _valid_feature_frames(
    frames: tuple[PoseFrame, ...],
    min_visibility: float,
) -> tuple[tuple[PoseFrame, PoseFeature], ...]:
    valid_frames: list[tuple[PoseFrame, PoseFeature]] = []
    for frame in frames:
        feature = extract_pose_feature(
            frame,
            min_visibility=min_visibility,
        )
        if feature is not None:
            valid_frames.append((frame, feature))
    return tuple(valid_frames)


def _band_allows(
    reference_index: int,
    practice_index: int,
    reference_count: int,
    practice_count: int,
    band_ratio: float,
) -> bool:
    if reference_count == 1 or practice_count == 1:
        return True
    reference_position = reference_index / (reference_count - 1)
    practice_position = practice_index / (practice_count - 1)
    return abs(reference_position - practice_position) <= band_ratio


def _dtw_path(
    reference_features: tuple[tuple[PoseFrame, PoseFeature], ...],
    practice_features: tuple[tuple[PoseFrame, PoseFeature], ...],
    band_ratio: float,
) -> tuple[tuple[int, int, float], ...]:
    reference_count = len(reference_features)
    practice_count = len(practice_features)
    costs = np.full((reference_count, practice_count), np.inf)
    previous = np.full((reference_count, practice_count, 2), -1, dtype=int)

    for reference_index in range(reference_count):
        for practice_index in range(practice_count):
            if not _band_allows(
                reference_index,
                practice_index,
                reference_count,
                practice_count,
                band_ratio,
            ):
                continue

            distance = weighted_feature_distance(
                reference_features[reference_index][1],
                practice_features[practice_index][1],
            )
            if reference_index == 0 and practice_index == 0:
                costs[reference_index, practice_index] = distance
                continue

            candidates = []
            if reference_index > 0:
                candidates.append(
                    (
                        costs[reference_index - 1, practice_index],
                        reference_index - 1,
                        practice_index,
                    )
                )
            if practice_index > 0:
                candidates.append(
                    (
                        costs[reference_index, practice_index - 1],
                        reference_index,
                        practice_index - 1,
                    )
                )
            if reference_index > 0 and practice_index > 0:
                candidates.append(
                    (
                        costs[reference_index - 1, practice_index - 1],
                        reference_index - 1,
                        practice_index - 1,
                    )
                )

            best_cost, previous_reference, previous_practice = min(
                candidates,
                key=lambda candidate: candidate[0],
            )
            if np.isfinite(best_cost):
                costs[reference_index, practice_index] = best_cost + distance
                previous[reference_index, practice_index] = (
                    previous_reference,
                    previous_practice,
                )

    if not np.isfinite(costs[-1, -1]):
        raise ValueError(
            "DTWの対応経路を作成できませんでした。band_ratioを大きくしてください。"
        )

    path: list[tuple[int, int, float]] = []
    reference_index = reference_count - 1
    practice_index = practice_count - 1
    while True:
        distance = weighted_feature_distance(
            reference_features[reference_index][1],
            practice_features[practice_index][1],
        )
        path.append((reference_index, practice_index, distance))
        previous_reference, previous_practice = previous[
            reference_index,
            practice_index,
        ]
        if previous_reference < 0:
            break
        reference_index = previous_reference
        practice_index = previous_practice

    path.reverse()
    return tuple(path)


def align_pose_frames(
    reference_frames: tuple[PoseFrame, ...],
    practice_frames: tuple[PoseFrame, ...],
    pair_count: int | None = None,
    band_ratio: float = 0.25,
    min_visibility: float = 0.5,
) -> tuple[AlignedPoseFrame, ...]:
    """正規化骨格特徴量のDTWでフレームを対応付ける。

    pair_countは既存API互換のために受け付けるが、DTW経路上の自然な
    繰り返し対応を優先するため、指定時も経路を間引かない。
    """
    del pair_count
    if not reference_frames or not practice_frames:
        raise ValueError("対応付けには両方の骨格フレームが必要です。")
    if not 0 < band_ratio <= 1:
        raise ValueError("band_ratioは0より大きく1以下で指定してください。")

    reference_features = _valid_feature_frames(
        reference_frames,
        min_visibility=min_visibility,
    )
    practice_features = _valid_feature_frames(
        practice_frames,
        min_visibility=min_visibility,
    )
    if not reference_features or not practice_features:
        raise ValueError(
            "DTWに必要な有効な骨格特徴量がありません。欠損または信頼度を確認してください。"
        )

    path = _dtw_path(
        reference_features,
        practice_features,
        band_ratio=band_ratio,
    )
    reference_duration = reference_frames[-1].timestamp_seconds
    practice_duration = practice_frames[-1].timestamp_seconds
    aligned_frames = []
    for path_index, (reference_index, practice_index, distance) in enumerate(path):
        reference_frame = reference_features[reference_index][0]
        practice_frame = practice_features[practice_index][0]
        normalized_position = (
            reference_frame.timestamp_seconds / reference_duration
            if reference_duration > 0
            else 0.0
        )
        aligned_frames.append(
            AlignedPoseFrame(
                reference=reference_frame,
                practice=practice_frame,
                normalized_position=normalized_position,
                feature_distance=distance,
                path_index=path_index,
            )
        )
    return tuple(aligned_frames)
