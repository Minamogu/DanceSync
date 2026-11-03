import streamlit as st

from processing.comparison import JointComparison
from processing.visualization import calculate_angle_difference_series
from processing.alignment import AlignedPoseFrame


def render_comparison_charts(
    comparisons: tuple[JointComparison, ...],
    aligned_frames: tuple[AlignedPoseFrame, ...],
    min_visibility: float = 0.25,
) -> None:
    """対応フレームごとの関節角度差をグラフ表示する。"""
    if not comparisons or not aligned_frames:
        return

    st.subheader("角度差の可視化")
    st.caption("対応フレームごとの角度差を表示しています。")

    try:
        difference_series = calculate_angle_difference_series(
            aligned_frames,
            min_visibility=min_visibility,
        )
    except ValueError as error:
        st.error(f"角度差を可視化できませんでした: {error}")
        return

    if not difference_series:
        st.info("フレームごとの角度差を可視化できるデータがありません。")
        return

    st.markdown("#### フレームごとの角度差")
    chart_data = {
        joint_name: values
        for joint_name, values in difference_series.items()
    }
    st.line_chart(chart_data, x_label="対応フレーム", y_label="角度差（度）")
