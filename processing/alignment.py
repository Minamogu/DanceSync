from dataclasses import dataclass

from processing.pose_estimation import PoseFrame


@dataclass(frozen=True)
class AlignedPoseFrame:
    """時間割合で対応付けられた2つの骨格フレーム。"""

    reference: PoseFrame
    practice: PoseFrame
    normalized_position: float


def _nearest_frame(
    frames: tuple[PoseFrame, ...],
    normalized_position: float,
) -> PoseFrame:
    """指定した時間割合に最も近い骨格フレームを返す。"""
    if len(frames) == 1:
        return frames[0]

    position = normalized_position * (len(frames) - 1)
    nearest_index = min(
        range(len(frames)),
        key=lambda index: abs(index - position),
    )
    return frames[nearest_index]


def align_pose_frames(
    reference_frames: tuple[PoseFrame, ...],
    practice_frames: tuple[PoseFrame, ...],
    pair_count: int | None = None,
) -> tuple[AlignedPoseFrame, ...]:
    """2つの動画の骨格フレームを時間割合で対応付ける。

    動画のFPSやフレーム数が異なっていても、各動画の先頭から末尾までの
    同じ時間割合にあるフレームを対応させる。
    """
    if not reference_frames or not practice_frames:
        raise ValueError("対応付けには両方の骨格フレームが必要です。")
    if pair_count is None:
        pair_count = min(len(reference_frames), len(practice_frames))
    if pair_count <= 0:
        raise ValueError("pair_countは1以上で指定してください。")

    aligned_frames: list[AlignedPoseFrame] = []
    for pair_index in range(pair_count):
        if pair_count == 1:
            normalized_position = 0.0
        else:
            normalized_position = pair_index / (pair_count - 1)

        aligned_frames.append(
            AlignedPoseFrame(
                reference=_nearest_frame(reference_frames, normalized_position),
                practice=_nearest_frame(practice_frames, normalized_position),
                normalized_position=normalized_position,
            )
        )

    return tuple(aligned_frames)
