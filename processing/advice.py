from dataclasses import dataclass

from processing.alignment import AlignedPoseFrame
from processing.angles import calculate_frame_angles


@dataclass(frozen=True)
class Advice:
    """対応フレーム単位で生成した改善アドバイス。"""

    joint_name: str
    difference: float
    message: str
    aligned_frame: AlignedPoseFrame
    reference_angle: float
    practice_angle: float
    persistence_count: int


@dataclass(frozen=True)
class _Candidate:
    joint_name: str
    difference: float
    reference_angle: float
    practice_angle: float
    aligned_frame: AlignedPoseFrame
    path_index: int


def _message(
    joint_name: str,
    reference_angle: float,
    practice_angle: float,
) -> str:
    if practice_angle > reference_angle:
        return f"{joint_name}をもう少し曲げましょう。"
    if practice_angle < reference_angle:
        return f"{joint_name}をもう少し伸ばしましょう。"
    return f"{joint_name}をお手本に近づけましょう。"


def generate_advice(
    aligned_frames: tuple[AlignedPoseFrame, ...],
    difference_threshold: float = 15.0,
    max_advice_count: int = 6,
    min_persistence: int = 2,
    min_visibility: float = 0.25,
) -> tuple[Advice, ...]:
    """対応フレームの角度差と継続性から改善アドバイスを生成する。"""
    if difference_threshold < 0:
        raise ValueError("difference_thresholdは0以上で指定してください。")
    if max_advice_count <= 0:
        raise ValueError("max_advice_countは1以上で指定してください。")
    if min_persistence <= 0:
        raise ValueError("min_persistenceは1以上で指定してください。")

    candidates_by_joint: dict[str, list[_Candidate]] = {}
    seen_pairs: set[tuple[str, int, int]] = set()
    for aligned_frame in aligned_frames:
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
            reference_angle = reference_angles[joint_name]
            practice_angle = practice_angles[joint_name]
            difference = abs(practice_angle - reference_angle)
            pair_key = (
                joint_name,
                aligned_frame.reference.frame_index,
                aligned_frame.practice.frame_index,
            )
            if difference < difference_threshold or pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            candidates_by_joint.setdefault(joint_name, []).append(
                _Candidate(
                    joint_name=joint_name,
                    difference=difference,
                    reference_angle=reference_angle,
                    practice_angle=practice_angle,
                    aligned_frame=aligned_frame,
                    path_index=aligned_frame.path_index,
                )
            )

    selected: list[Advice] = []
    for joint_name, candidates in candidates_by_joint.items():
        candidates.sort(key=lambda candidate: candidate.path_index)
        episodes: list[list[_Candidate]] = []
        for candidate in candidates:
            if (
                not episodes
                or candidate.path_index != episodes[-1][-1].path_index + 1
            ):
                episodes.append([candidate])
            else:
                episodes[-1].append(candidate)

        for episode in episodes:
            if len(episode) < min_persistence:
                continue
            representative = max(
                episode,
                key=lambda candidate: candidate.difference,
            )
            selected.append(
                Advice(
                    joint_name=joint_name,
                    difference=representative.difference,
                    message=_message(
                        joint_name,
                        representative.reference_angle,
                        representative.practice_angle,
                    ),
                    aligned_frame=representative.aligned_frame,
                    reference_angle=representative.reference_angle,
                    practice_angle=representative.practice_angle,
                    persistence_count=len(episode),
                )
            )

    selected.sort(
        key=lambda advice: (
            advice.difference * advice.persistence_count,
            advice.difference,
        ),
        reverse=True,
    )
    return tuple(selected[:max_advice_count])
