from dataclasses import dataclass

from processing.comparison import JointComparison


@dataclass(frozen=True)
class Advice:
    """関節角度の比較結果から生成した改善アドバイス。"""

    joint_name: str
    difference: float
    message: str


def generate_advice(
    comparisons: tuple[JointComparison, ...],
    difference_threshold: float = 15.0,
    max_advice_count: int = 3,
) -> tuple[Advice, ...]:
    """角度差が閾値以上の部位について改善アドバイスを生成する。"""
    if difference_threshold < 0:
        raise ValueError("difference_thresholdは0以上で指定してください。")
    if max_advice_count <= 0:
        raise ValueError("max_advice_countは1以上で指定してください。")

    advice: list[Advice] = []
    for comparison in comparisons:
        if comparison.mean_difference < difference_threshold:
            continue

        if comparison.mean_signed_difference > 0:
            direction = "曲げ"
        elif comparison.mean_signed_difference < 0:
            direction = "伸ばし"
        else:
            direction = "調整"

        if direction == "調整":
            message = (
                f"{comparison.name}の角度をお手本に近づけるように調整してください。"
            )
        else:
            message = f"{comparison.name}をもう少し{direction}ましょう。"

        advice.append(
            Advice(
                joint_name=comparison.name,
                difference=comparison.mean_difference,
                message=message,
            )
        )
        if len(advice) >= max_advice_count:
            break

    return tuple(advice)
