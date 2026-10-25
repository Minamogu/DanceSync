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
    sample_interval: int = 1,
    max_frames: int | None = None,
    resize: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """動画から一定間隔でフレームを抽出する。

    抽出した画像はMediaPipeで扱いやすいRGB形式で返す。
    resizeを指定する場合は(width, height)の順で指定する。
    """
    if sample_interval <= 0:
        raise ValueError("sample_intervalは1以上で指定してください。")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_framesは1以上で指定してください。")
    if resize is not None and (resize[0] <= 0 or resize[1] <= 0):
        raise ValueError("resizeには正の幅と高さを指定してください。")

    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    frame_index = 0

    try:
        if not capture.isOpened():
            raise VideoProcessingError("動画を開けませんでした。")

        while max_frames is None or len(frames) < max_frames:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % sample_interval == 0:
                if resize is not None:
                    frame = cv2.resize(frame, resize)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
