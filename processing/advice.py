from dataclasses import dataclass

from processing.alignment import AlignedPoseFrame
from processing.metrics import (
    FrameMetrics,
    MetricValue,
    calculate_frame_metrics,
    representative_leg_length,
)


ANGLE_METRICS = {
    "左肩",
    "右肩",
    "左肘",
    "右肘",
    "左股関節",
    "右股関節",
    "左膝",
    "右膝",
}
LEG_METRICS = {"左股関節", "右股関節", "左膝", "右膝"}


@dataclass(frozen=True)
class Advice:
    """対応フレーム単位で生成した改善アドバイス。"""

    joint_name: str
    difference: float
    message: str
    aligned_frame: AlignedPoseFrame
    reference_angle: float | None
    practice_angle: float | None
    persistence_count: int
    evidence_metrics: tuple[str, ...] = ()
    normalized_severity: float = 0.0


@dataclass(frozen=True)
class _Candidate:
    metric: MetricValue
    aligned_frame: AlignedPoseFrame
    path_index: int
    normalized_severity: float


def _thresholds(
    angle_threshold_degrees: float,
    stance_threshold: float,
    height_threshold: float,
) -> dict[str, float]:
    return {
        **{name: angle_threshold_degrees for name in ANGLE_METRICS},
        "スタンスの広さ": stance_threshold,
        "腰の高さ": height_threshold,
    }


def _message(metric: MetricValue) -> str:
    if metric.name == "スタンスの広さ":
        if metric.difference < 0:
            return "お手本より足の開きが狭いため、もう少し足を広げましょう。"
        return "お手本より足の開きが広いため、少し足を閉じましょう。"
    if metric.name == "腰の高さ":
        if metric.difference > 0:
            return "お手本より腰の位置が高いため、もう少し腰を落としましょう。"
        return "お手本より腰の位置が低いため、少し腰を上げましょう。"
    if metric.difference < 0:
        return f"{metric.name}をもう少し曲げましょう。"
    return f"{metric.name}をもう少し伸ばしましょう。"


def _episodes(
    candidates: list[_Candidate],
    min_duration_seconds: float,
    max_gap_seconds: float,
) -> list[list[_Candidate]]:
    candidates.sort(
        key=lambda candidate: (
            candidate.aligned_frame.reference.timestamp_seconds,
            candidate.aligned_frame.practice.timestamp_seconds,
        )
    )
    episodes: list[list[_Candidate]] = []
    for candidate in candidates:
        if not episodes:
            episodes.append([candidate])
            continue
        previous = episodes[-1][-1]
        reference_gap = (
            candidate.aligned_frame.reference.timestamp_seconds
            - previous.aligned_frame.reference.timestamp_seconds
        )
        practice_gap = (
            candidate.aligned_frame.practice.timestamp_seconds
            - previous.aligned_frame.practice.timestamp_seconds
        )
        if (
            0 <= reference_gap <= max_gap_seconds
            and 0 <= practice_gap <= max_gap_seconds
        ):
            episodes[-1].append(candidate)
        else:
            episodes.append([candidate])

    valid_episodes = []
    for episode in episodes:
        reference_duration = (
            episode[-1].aligned_frame.reference.timestamp_seconds
            - episode[0].aligned_frame.reference.timestamp_seconds
        )
        practice_duration = (
            episode[-1].aligned_frame.practice.timestamp_seconds
            - episode[0].aligned_frame.practice.timestamp_seconds
        )
        if min(reference_duration, practice_duration) >= min_duration_seconds:
            valid_episodes.append(episode)
    return valid_episodes


def _best_candidate(episode: list[_Candidate]) -> _Candidate:
    return max(
        episode,
        key=lambda candidate: candidate.normalized_severity,
    )


def _integrate_waist_advice(
    candidates: dict[str, list[_Candidate]],
    min_duration_seconds: float,
    max_gap_seconds: float,
) -> tuple[list[Advice], set[int]]:
    """腰高さと脚角度が同じ方向のときだけ1件に統合する。"""
    height_episodes = _episodes(
        candidates.get("腰の高さ", []),
        min_duration_seconds,
        max_gap_seconds,
    )
    leg_candidates = [
        candidate
        for name in LEG_METRICS
        for candidate in candidates.get(name, [])
    ]
    leg_episodes = _episodes(
        leg_candidates,
        min_duration_seconds,
        max_gap_seconds,
    )
    integrated: list[Advice] = []
    used_candidates: set[int] = set()
    for height_episode in height_episodes:
        height = _best_candidate(height_episode)
        matching_legs = []
        for index, episode in enumerate(leg_episodes):
            if any(
                id(candidate) in used_candidates
                for candidate in episode
            ):
                continue
            leg = _best_candidate(episode)
            reference_time = height.aligned_frame.reference.timestamp_seconds
            leg_time = leg.aligned_frame.reference.timestamp_seconds
            if abs(reference_time - leg_time) > max_gap_seconds:
                continue
            lowering = height.metric.difference > 0
            leg_is_less_bent = leg.metric.difference > 0
            if lowering == leg_is_less_bent:
                matching_legs.append((index, leg))

        if not matching_legs:
            continue
        used_candidates.update(id(candidate) for candidate in height_episode)
        for index, _ in matching_legs:
            used_candidates.update(
                id(candidate) for candidate in leg_episodes[index]
            )
        evidence = ("腰の高さ",) + tuple(
            leg.metric.name for _, leg in matching_legs
        )
        message = _message(height.metric)
        if height.metric.difference > 0:
            message = (
                "お手本より腰の位置が高いため、"
                "膝・股関節を曲げてもう少し腰を落としましょう。"
            )
        integrated.append(
            Advice(
                joint_name="腰の落とし方",
                difference=height.metric.absolute_difference,
                message=message,
                aligned_frame=height.aligned_frame,
                reference_angle=None,
                practice_angle=None,
                persistence_count=max(
                    len(height_episode),
                    *(len(leg_episodes[index]) for index, _ in matching_legs),
                ),
                evidence_metrics=evidence,
                normalized_severity=max(
                    height.normalized_severity,
                    *(leg.normalized_severity for _, leg in matching_legs),
                ),
            )
        )
    return integrated, used_candidates


def generate_advice(
    aligned_frames: tuple[AlignedPoseFrame, ...],
    angle_threshold_degrees: float = 15.0,
    stance_threshold: float = 0.15,
    height_threshold: float = 0.10,
    max_advice_count: int = 6,
    min_duration_seconds: float = 0.15,
    max_gap_seconds: float = 0.25,
    min_visibility: float = 0.25,
) -> tuple[Advice, ...]:
    """指標別の継続候補を抽出し、関連する脚の指摘を統合する。"""
    thresholds = _thresholds(
        angle_threshold_degrees,
        stance_threshold,
        height_threshold,
    )
    if any(value <= 0 for value in thresholds.values()):
        raise ValueError("各閾値は0より大きく指定してください。")
    if max_advice_count <= 0:
        raise ValueError("max_advice_countは1以上で指定してください。")

    reference_frames = tuple(frame.reference for frame in aligned_frames)
    practice_frames = tuple(frame.practice for frame in aligned_frames)
    reference_leg_length = representative_leg_length(
        reference_frames,
        min_visibility=min_visibility,
    )
    practice_leg_length = representative_leg_length(
        practice_frames,
        min_visibility=min_visibility,
    )
    candidates: dict[str, list[_Candidate]] = {}
    seen_pairs: set[tuple[str, int, int]] = set()
    for aligned_frame in aligned_frames:
        metrics: FrameMetrics = calculate_frame_metrics(
            aligned_frame.reference,
            aligned_frame.practice,
            reference_leg_length,
            practice_leg_length,
            min_visibility=min_visibility,
        )
        for metric in metrics.values:
            threshold = thresholds[metric.name]
            pair_key = (
                metric.name,
                aligned_frame.reference.frame_index,
                aligned_frame.practice.frame_index,
            )
            if (
                metric.absolute_difference < threshold
                or pair_key in seen_pairs
            ):
                continue
            seen_pairs.add(pair_key)
            candidates.setdefault(metric.name, []).append(
                _Candidate(
                    metric=metric,
                    aligned_frame=aligned_frame,
                    path_index=aligned_frame.path_index,
                    normalized_severity=metric.absolute_difference / threshold,
                )
            )

    advice, used_candidates = _integrate_waist_advice(
        candidates,
        min_duration_seconds,
        max_gap_seconds,
    )
    for metric_name, metric_candidates in candidates.items():
        remaining_candidates = [
            candidate
            for candidate in metric_candidates
            if id(candidate) not in used_candidates
        ]
        for episode in _episodes(
            remaining_candidates,
            min_duration_seconds,
            max_gap_seconds,
        ):
            representative = _best_candidate(episode)
            metric = representative.metric
            advice.append(
                Advice(
                    joint_name=metric.name,
                    difference=metric.absolute_difference,
                    message=_message(metric),
                    aligned_frame=representative.aligned_frame,
                    reference_angle=(
                        metric.reference if metric.name in ANGLE_METRICS else None
                    ),
                    practice_angle=(
                        metric.practice if metric.name in ANGLE_METRICS else None
                    ),
                    persistence_count=len(episode),
                    evidence_metrics=(metric.name,),
                    normalized_severity=representative.normalized_severity,
                )
            )

    advice.sort(
        key=lambda item: (
            item.normalized_severity * item.persistence_count,
            item.normalized_severity,
        ),
        reverse=True,
    )
    return tuple(advice[:max_advice_count])
