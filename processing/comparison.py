from dataclasses import dataclass

from processing.alignment import AlignedPoseFrame
from processing.angles import calculate_frame_angles


@dataclass(frozen=True)
class JointComparison:
    """部位ごとの関節角度比較結果。"""

    name: str
    sample_count: int
    mean_difference: float
    max_difference: float
    mean_signed_difference: float


def compare_aligned_frames(
    aligned_frames: tuple[AlignedPoseFrame, ...],
    min_visibility: float = 0.5,
) -> tuple[JointComparison, ...]:
    """対応付け済みフレームの関節角度を部位ごとに比較する。

    differenceは「自分の動画 - お手本動画」の符号付き差分として計算し、
    mean_differenceとmax_differenceは絶対値で集計する。
    """
    if not aligned_frames:
        raise ValueError("比較対象の対応フレームがありません。")

    differences_by_joint: dict[str, list[float]] = {}
    for aligned_frame in aligned_frames:
        if (
            aligned_frame.reference.landmarks is None
            or aligned_frame.practice.landmarks is None
        ):
            continue

        reference_angles = {
            angle.name: angle.degrees
            for angle in calculate_frame_angles(
                aligned_frame.reference,
                min_visibility=min_visibility,
            )
        }
        practice_angles = {
            angle.name: angle.degrees
            for angle in calculate_frame_angles(
                aligned_frame.practice,
                min_visibility=min_visibility,
            )
        }

        for joint_name in reference_angles.keys() & practice_angles.keys():
            differences_by_joint.setdefault(joint_name, []).append(
                practice_angles[joint_name] - reference_angles[joint_name]
            )

    comparisons: list[JointComparison] = []
    for joint_name, differences in differences_by_joint.items():
        if not differences:
            continue
        absolute_differences = [abs(difference) for difference in differences]
        comparisons.append(
            JointComparison(
                name=joint_name,
                sample_count=len(differences),
                mean_difference=sum(absolute_differences) / len(differences),
                max_difference=max(absolute_differences),
                mean_signed_difference=sum(differences) / len(differences),
            )
        )

    return tuple(
        sorted(
            comparisons,
            key=lambda comparison: comparison.mean_difference,
            reverse=True,
        )
    )
