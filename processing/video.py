from dataclasses import dataclass
from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

import cv2
import numpy as np


class VideoProcessingError(Exception):
    """動画を読み込めない場合に発生する例外。"""


@dataclass(frozen=True)
class VideoMetadata:
    """動画の基本情報。"""

    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        """動画の長さを秒で返す。"""
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


@dataclass(frozen=True)
class VideoFrame:
    """抽出された動画フレームと元動画上の位置。"""

    image: np.ndarray
    frame_index: int
    timestamp_seconds: float


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    """動画ファイルからFPS、フレーム数、解像度を取得する。"""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoProcessingError("動画を開けませんでした。")

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or frame_count < 0 or width <= 0 or height <= 0:
            raise VideoProcessingError("動画の基本情報を取得できませんでした。")

        return VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
    finally:
        capture.release()


def extract_frames(
    video_path: str | Path,
    sample_interval_seconds: float = 0.1,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
) -> list[VideoFrame]:
    """動画から一定時間間隔でフレームを抽出する。

    抽出した画像はMediaPipeで扱いやすいRGB形式で返す。
    resizeを指定する場合は(width, height)の順で指定する。
    """
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_secondsは正の値で指定してください。")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_framesは1以上で指定してください。")
    if resize is not None and (resize[0] <= 0 or resize[1] <= 0):
        raise ValueError("resizeには正の幅と高さを指定してください。")

    capture = cv2.VideoCapture(str(video_path))
    frames: list[VideoFrame] = []
    frame_index = 0
    next_sample_time = 0.0

    try:
        if not capture.isOpened():
            raise VideoProcessingError("動画を開けませんでした。")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise VideoProcessingError("動画のFPSを取得できませんでした。")

        while max_frames is None or len(frames) < max_frames:
            success, frame = capture.read()
            if not success:
                break

            timestamp_seconds = frame_index / fps
            if timestamp_seconds + (0.5 / fps) >= next_sample_time:
                if resize is not None:
                    frame = cv2.resize(frame, resize)
                frames.append(
                    VideoFrame(
                        image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        frame_index=frame_index,
                        timestamp_seconds=timestamp_seconds,
                    )
                )
                next_sample_time += sample_interval_seconds
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise VideoProcessingError("動画からフレームを抽出できませんでした。")
    return frames


@contextmanager
def save_uploaded_video(video_file) -> Iterator[Path]:
    """Streamlitのアップロードファイルを一時ファイルとして保存する。"""
    if video_file is None:
        raise ValueError("動画ファイルが指定されていません。")

    temporary_file = NamedTemporaryFile(suffix=".mp4", delete=False)
    temporary_path = Path(temporary_file.name)
    try:
        temporary_file.write(video_file.getvalue())
        temporary_file.flush()
        temporary_file.close()
        yield temporary_path
    finally:
        if not temporary_file.closed:
            temporary_file.close()
        temporary_path.unlink(missing_ok=True)
