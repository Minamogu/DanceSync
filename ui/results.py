import streamlit as st

from processing.advice import generate_advice
from processing.alignment import AlignedPoseFrame, align_pose_frames
from processing.angles import calculate_frame_angles
from processing.comparison import compare_aligned_frames
from processing.pose_estimation import (
    PoseEstimationError,
    PoseNotDetectedError,
    estimate_pose_video,
)
from processing.video import save_uploaded_video
from ui.visualization import render_comparison_charts


def _estimate_pose(video_file, label: str):
    """アップロード動画から骨格を推定する。"""
    try:
        with save_uploaded_video(video_file) as video_path:
            return estimate_pose_video(video_path)
    except PoseNotDetectedError as error:
        st.warning(f"{label}: {error}")
        return None
    except PoseEstimationError as error:
        st.error(f"{label}の骨格を推定できませんでした: {error}")
        return None


def _render_pose_result(result, label: str) -> None:
    """1本分の骨格推定結果を表示する。"""
    st.caption(f"骨格を検出したフレーム数: {result.detected_frame_count}")
    st.image(result.annotated_frames[0], caption=f"{label}（骨格描画）")

    angles = calculate_frame_angles(result.frames[0])
    if angles:
        st.caption("代表フレームの関節角度")
        st.dataframe(
            [
                {"関節": angle.name, "角度（度）": round(angle.degrees, 1)}
                for angle in angles
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("代表フレームでは信頼できる関節角度を計算できませんでした。")


def _render_alignment_info(
    reference_result,
    practice_result,
) -> tuple[AlignedPoseFrame, ...]:
    """2つの骨格推定結果の対応付け情報を表示する。"""
    try:
        aligned_frames = align_pose_frames(
            reference_result.frames,
            practice_result.frames,
        )
    except ValueError as error:
        st.error(f"フレームを対応付けられませんでした: {error}")
        return ()

    st.subheader("フレーム対応付け")
    st.caption(
        f"時間割合に基づいて、{len(aligned_frames)}組のフレームを対応付けました。"
    )
    st.dataframe(
        [
            {
                "対応位置": f"{frame.normalized_position:.0%}",
                "お手本フレーム": frame.reference.frame_index,
                "自分フレーム": frame.practice.frame_index,
            }
            for frame in aligned_frames
        ],
        hide_index=True,
        use_container_width=True,
    )
    return aligned_frames


def _render_comparison_info(
    aligned_frames: tuple[AlignedPoseFrame, ...],
) -> None:
    """対応フレームの比較結果と改善アドバイスを表示する。"""
    if not aligned_frames:
        return

    try:
        comparisons = compare_aligned_frames(aligned_frames)
    except ValueError as error:
        st.error(f"関節角度を比較できませんでした: {error}")
        return

    st.subheader("関節角度の比較")
    if not comparisons:
        st.info("両方の動画で共通して計算できる関節角度がありませんでした。")
        return

    st.caption(
        "角度差は「自分の動画 - お手本動画」で計算しています。"
        "平均差と最大差は絶対値です。"
    )
    st.dataframe(
        [
            {
                "関節": comparison.name,
                "比較フレーム数": comparison.sample_count,
                "平均角度差（度）": round(comparison.mean_difference, 1),
                "最大角度差（度）": round(comparison.max_difference, 1),
                "符号付き平均差（度）": round(
                    comparison.mean_signed_difference,
                    1,
                ),
            }
            for comparison in comparisons
        ],
        hide_index=True,
        use_container_width=True,
    )
    render_comparison_charts(comparisons, aligned_frames)

    st.subheader("改善アドバイス")
    advice = generate_advice(comparisons)
    if not advice:
        st.success("大きな角度差はありませんでした。")
        return
    for item in advice:
        st.warning(
            f"**{item.joint_name}**（平均差 {item.difference:.1f}度）: "
            f"{item.message}"
        )


def render_analysis_results(reference_video, practice_video) -> None:
    """2本の動画を解析し、進捗と結果を表示する。"""
    progress = st.progress(0, text="解析を準備しています...")
    with st.status("解析中", expanded=True) as status:
        status.write("お手本の動画を解析しています...")
        reference_result = _estimate_pose(reference_video, "お手本の動画")
        progress.progress(40, text="お手本の動画の解析が完了しました")
        if reference_result is None:
            status.update(label="解析を中止しました", state="error")
            return

        status.write("自分の動画を解析しています...")
        practice_result = _estimate_pose(practice_video, "自分の動画")
        progress.progress(80, text="自分の動画の解析が完了しました")
        if practice_result is None:
            status.update(label="解析を中止しました", state="error")
            return

        status.write("フレームを対応付けています...")
        aligned_frames = _render_alignment_info(
            reference_result,
            practice_result,
        )
        progress.progress(100, text="解析が完了しました")
        status.update(label="解析が完了しました", state="complete")

    st.subheader("骨格推定結果")
    result_columns = st.columns(2)
    with result_columns[0]:
        _render_pose_result(reference_result, "お手本の動画")
    with result_columns[1]:
        _render_pose_result(practice_result, "自分の動画")
    _render_comparison_info(aligned_frames)
