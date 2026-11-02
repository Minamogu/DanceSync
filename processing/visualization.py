from typing import Optional

from processing.alignment import AlignedPoseFrame
from processing.angles import calculate_frame_angles


def calculate_angle_difference_series(
    aligned_frames: tuple[AlignedPoseFrame, ...],
    min_visibility: float = 0.25,
) -> dict[str, list[Optional[float]]]:
    """対応フレームごとの関節角度差を、グラフ用データとして返す。"""
    if not aligned_frames:
        raise ValueError("可視化する対応フレームがありません。")

    joint_names = {
        angle.name
        for aligned_frame in aligned_frames
        for angle in calculate_frame_angles(
            aligned_frame.reference,
            min_visibility=min_visibility,
        )
    }
    differences_by_joint = {
        joint_name: [] for joint_name in sorted(joint_names)
    }
    for aligned_frame in aligned_frames:
        if (
            aligned_frame.reference.landmarks is None
            or aligned_frame.practice.landmarks is None
        ):
            for joint_name in differences_by_joint:
                differences_by_joint[joint_name].append(None)
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

        for joint_name in differences_by_joint:
            if joint_name in reference_angles and joint_name in practice_angles:
                difference = abs(
                    practice_angles[joint_name] - reference_angles[joint_name]
                )
                differences_by_joint[joint_name].append(difference)
            else:
                differences_by_joint[joint_name].append(None)

    return {
        joint_name: values
        for joint_name, values in differences_by_joint.items()
        if any(value is not None for value in values)
    }
