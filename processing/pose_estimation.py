from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np

from processing.video import VideoProcessingError, extract_frames


class PoseEstimationError(Exception):
    """骨格推定に失敗した場合に発生する例外。"""


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
    landmarks: tuple[Landmark, ...]


@dataclass(frozen=True)
class PoseEstimationResult:
    """動画から取得した骨格推定結果。"""

    frames: tuple[PoseFrame, ...]
    annotated_frames: tuple[np.ndarray, ...]

    @property
    def detected_frame_count(self) -> int:
        """骨格が検出されたフレーム数を返す。"""
        return len(self.frames)


def estimate_pose_frames(
    frames: list[np.ndarray],
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
        for offset, frame in enumerate(frames):
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise PoseEstimationError("フレームは3チャンネルのRGB画像で指定してください。")

            pose_result = pose.process(frame)
            annotated_frame = frame.copy()

            if pose_result.pose_landmarks is None:
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
                    frame_index=start_frame_index + offset,
                    landmarks=landmarks,
                )
            )

            drawing_utils.draw_landmarks(
                annotated_frame,
                pose_result.pose_landmarks,
                pose_module.POSE_CONNECTIONS,
            )
            annotated_frames.append(annotated_frame)

    if not results:
        raise PoseEstimationError("人物の骨格を検出できませんでした。")

    return PoseEstimationResult(
        frames=tuple(results),
        annotated_frames=tuple(annotated_frames),
    )


def estimate_pose_video(
    video_path: str | Path,
    sample_interval: int = 10,
    max_frames: int = 30,
) -> PoseEstimationResult:
    """動画から一定間隔でフレームを抽出し、骨格を推定する。"""
    if sample_interval <= 0:
        raise ValueError("sample_intervalは1以上で指定してください。")
    if max_frames <= 0:
        raise ValueError("max_framesは1以上で指定してください。")

    try:
        frames = extract_frames(
            video_path,
            sample_interval=sample_interval,
            max_frames=max_frames,
        )
        return estimate_pose_frames(
            frames,
            start_frame_index=0,
        )
    except VideoProcessingError as error:
        raise PoseEstimationError(str(error)) from error
