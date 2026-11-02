from dataclasses import dataclass, field
from pathlib import Path

import mediapipe as mp
import numpy as np

from processing.video import VideoFrame, VideoProcessingError, extract_frames


class PoseEstimationError(Exception):
    """骨格推定に失敗した場合に発生する例外。"""


class PoseNotDetectedError(PoseEstimationError):
    """人物の骨格を検出できなかった場合に発生する例外。"""


@dataclass(frozen=True)
class Landmark:
    """1つの関節ランドマーク。座標は画像サイズに対して正規化される。"""

    name: str
    x: float
    y: float
    z: float
    visibility: float


@dataclass(frozen=True)
class PoseFrame:
    """1フレーム分の骨格推定結果。"""

    frame_index: int
    landmarks: tuple[Landmark, ...] | None
    timestamp_seconds: float = 0.0
    source_frame: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8)
    )
    interpolated: bool = False


@dataclass(frozen=True)
class PoseEstimationResult:
    """動画から取得した骨格推定結果。"""

    frames: tuple[PoseFrame, ...]
    annotated_frames: tuple[np.ndarray, ...]

    @property
    def detected_frame_count(self) -> int:
        """元動画で骨格が検出されたフレーム数を返す。"""
        return sum(
            frame.landmarks is not None and not frame.interpolated
            for frame in self.frames
        )

    @property
    def interpolated_frame_count(self) -> int:
        """補間されたフレーム数を返す。"""
        return sum(frame.interpolated for frame in self.frames)

    @property
    def missing_frame_count(self) -> int:
        """補間できず欠損として残ったフレーム数を返す。"""
        return sum(frame.landmarks is None for frame in self.frames)


def estimate_pose_frames(
    frames: list[VideoFrame | np.ndarray],
    start_frame_index: int = 0,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> PoseEstimationResult:
    """RGBフレーム列から骨格を推定する。"""
    if not frames:
        raise PoseEstimationError("骨格推定するフレームがありません。")
    if not 0 <= min_detection_confidence <= 1:
        raise ValueError("min_detection_confidenceは0から1の範囲で指定してください。")
    if not 0 <= min_tracking_confidence <= 1:
        raise ValueError("min_tracking_confidenceは0から1の範囲で指定してください。")

    video_frames = [
        frame
        if isinstance(frame, VideoFrame)
        else VideoFrame(
            image=frame,
            frame_index=start_frame_index + offset,
            timestamp_seconds=float(offset),
        )
        for offset, frame in enumerate(frames)
    ]

    pose_module = mp.solutions.pose
    drawing_utils = mp.solutions.drawing_utils
    results: list[PoseFrame] = []
    annotated_frames: list[np.ndarray] = []

    with pose_module.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        for frame_data in video_frames:
            frame = frame_data.image
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise PoseEstimationError("フレームは3チャンネルのRGB画像で指定してください。")

            pose_result = pose.process(frame)
            annotated_frame = frame.copy()

            if pose_result.pose_landmarks is None:
                results.append(
                    PoseFrame(
                        frame_index=frame_data.frame_index,
                        timestamp_seconds=frame_data.timestamp_seconds,
                        landmarks=None,
                        source_frame=frame,
                    )
                )
                annotated_frames.append(annotated_frame)
                continue

            landmarks = tuple(
                Landmark(
                    name=pose_module.PoseLandmark(index).name,
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility,
                )
                for index, landmark in enumerate(pose_result.pose_landmarks.landmark)
            )
            results.append(
                PoseFrame(
                    frame_index=frame_data.frame_index,
                    timestamp_seconds=frame_data.timestamp_seconds,
                    landmarks=landmarks,
                    source_frame=frame,
                )
            )

            drawing_utils.draw_landmarks(
                annotated_frame,
                pose_result.pose_landmarks,
                pose_module.POSE_CONNECTIONS,
            )
            annotated_frames.append(annotated_frame)

    results = _interpolate_short_gaps(results)

    if not any(frame.landmarks is not None for frame in results):
        raise PoseNotDetectedError(
            "体を検出できませんでした。全身を写した動画をアップロードしてください。"
        )

    return PoseEstimationResult(
        frames=tuple(results),
        annotated_frames=tuple(annotated_frames),
    )


def _interpolate_short_gaps(
    frames: list[PoseFrame],
    max_gap: int = 2,
) -> list[PoseFrame]:
    """短い検出欠損を前後の骨格から線形補間する。"""
    interpolated_frames = list(frames)
    index = 0
    while index < len(interpolated_frames):
        if interpolated_frames[index].landmarks is not None:
            index += 1
            continue

        gap_start = index
        while (
            index < len(interpolated_frames)
            and interpolated_frames[index].landmarks is None
        ):
            index += 1
        gap_end = index
        gap_length = gap_end - gap_start
        before_index = gap_start - 1
        after_index = gap_end

        if (
            gap_length > max_gap
            or before_index < 0
            or after_index >= len(interpolated_frames)
            or interpolated_frames[before_index].landmarks is None
            or interpolated_frames[after_index].landmarks is None
        ):
            continue

        before = interpolated_frames[before_index]
        after = interpolated_frames[after_index]
        if len(before.landmarks) != len(after.landmarks):
            continue

        for offset, frame_index in enumerate(range(gap_start, gap_end), start=1):
            ratio = offset / (gap_length + 1)
            landmarks = tuple(
                Landmark(
                    name=before_landmark.name,
                    x=before_landmark.x
                    + (after_landmark.x - before_landmark.x) * ratio,
                    y=before_landmark.y
                    + (after_landmark.y - before_landmark.y) * ratio,
                    z=before_landmark.z
                    + (after_landmark.z - before_landmark.z) * ratio,
                    visibility=before_landmark.visibility
                    + (after_landmark.visibility - before_landmark.visibility) * ratio,
                )
                for before_landmark, after_landmark in zip(
                    before.landmarks,
                    after.landmarks,
                )
            )
            original = interpolated_frames[frame_index]
            interpolated_frames[frame_index] = PoseFrame(
                frame_index=original.frame_index,
                timestamp_seconds=original.timestamp_seconds,
                landmarks=landmarks,
                source_frame=original.source_frame,
                interpolated=True,
            )

    return interpolated_frames


def estimate_pose_video(
    video_path: str | Path,
    sample_interval_seconds: float = 0.1,
    max_frames: int | None = None,
) -> PoseEstimationResult:
    """動画から一定時間間隔でフレームを抽出し、骨格を推定する。"""
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_secondsは正の値で指定してください。")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_framesは1以上で指定してください。")

    try:
        frames = extract_frames(
            video_path,
            sample_interval_seconds=sample_interval_seconds,
            max_frames=max_frames,
        )
        return estimate_pose_frames(frames)
    except VideoProcessingError as error:
        raise PoseEstimationError(str(error)) from error
