import streamlit as st

from processing.angles import calculate_frame_angles
from processing.alignment import AlignedPoseFrame, align_pose_frames
from processing.advice import generate_advice
from processing.comparison import compare_aligned_frames
from processing.video import (
    VideoProcessingError,
    read_video_metadata,
    save_uploaded_video,
)
from processing.pose_estimation import PoseEstimationError, estimate_pose_video


def render_video_uploader(label: str, key: str):
    """動画をアップロードし、アップロードされたファイルを返す。"""
    return st.file_uploader(
        label,
        type=["mp4"],
        key=key,
        help="MP4形式の動画を選択してください。",
    )


def format_file_size(size_in_bytes: int) -> str:
    """ファイルサイズを読みやすい単位に変換する。"""
    if size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    return f"{size_in_bytes / (1024 * 1024):.1f} MB"


def render_video_info(video_file) -> None:
    """選択された動画の基本情報を表示する。"""
    if video_file is None:
        return

    st.caption(
        f"ファイル名: {video_file.name}  |  "
        f"形式: {video_file.type or '不明'}  |  "
        f"サイズ: {format_file_size(video_file.size)}"
    )

    try:
        with save_uploaded_video(video_file) as video_path:
            metadata = read_video_metadata(video_path)
    except VideoProcessingError as error:
        st.error(f"動画情報を読み込めませんでした: {error}")
        return

    st.caption(
        f"解像度: {metadata.width} x {metadata.height}  |  "
        f"FPS: {metadata.fps:.2f}  |  "
        f"フレーム数: {metadata.frame_count}  |  "
        f"長さ: {metadata.duration_seconds:.2f} 秒"
    )


def render_uploaded_video(video_file) -> None:
    """アップロードされた動画をプレビューする。"""
    if video_file is not None:
        st.video(video_file)


def render_pose_estimation(video_file, label: str):
    """動画から骨格を推定し、代表フレームを表示する。"""
    if video_file is None:
        return None

    try:
        with save_uploaded_video(video_file) as video_path:
            result = estimate_pose_video(video_path)
    except PoseEstimationError as error:
        st.error(f"{label}の骨格を推定できませんでした: {error}")
        return None

    st.caption(f"骨格を検出したフレーム数: {result.detected_frame_count}")
    preview_frame = result.annotated_frames[0]
    st.image(preview_frame, caption=f"{label}（骨格描画）")

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

    return result


def render_alignment_info(
    reference_result,
    practice_result,
) -> tuple[AlignedPoseFrame, ...]:
    """2つの骨格推定結果の対応付け情報を表示する。"""
    if reference_result is None or practice_result is None:
        return ()

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
                "対応位置": f"{aligned_frame.normalized_position:.0%}",
                "お手本フレーム": aligned_frame.reference.frame_index,
                "自分フレーム": aligned_frame.practice.frame_index,
            }
            for aligned_frame in aligned_frames
        ],
        hide_index=True,
        use_container_width=True,
    )
    return aligned_frames


def render_comparison_info(
    aligned_frames: tuple[AlignedPoseFrame, ...],
) -> None:
    """対応フレームの関節角度比較結果を表示する。"""
    if not aligned_frames:
        return

    try:
        comparisons = compare_aligned_frames(aligned_frames)
    except ValueError as error:
        st.error(f"関節角度を比較できませんでした: {error}")
        return

    st.subheader("関節角度の比較")
    if not comparisons:
        st.info(
            "両方の動画で共通して計算できる関節角度がありませんでした。"
        )
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

    advice = generate_advice(comparisons)
    st.subheader("改善アドバイス")
    if not advice:
        st.success("大きな角度差はありませんでした。")
        return

    for item in advice:
        st.warning(
            f"**{item.joint_name}**（平均差 {item.difference:.1f}度）: "
            f"{item.message}"
        )


def main() -> None:
    st.set_page_config(page_title="DanceSync", page_icon="💃")
    st.title("DanceSync")
    st.write("お手本のダンス動画と自分のダンス動画を比較するアプリです。")

    reference_video = render_video_uploader(
        "お手本のダンス動画",
        key="reference_video",
    )
    practice_video = render_video_uploader(
        "自分のダンス動画",
        key="practice_video",
    )

    if reference_video is None and practice_video is None:
        st.info("2つの動画をアップロードしてください。")
    elif reference_video is None or practice_video is None:
        st.warning("お手本と自分の動画を両方アップロードしてください。")
    else:
        st.success("2つの動画が選択されました。")

    st.subheader("動画プレビュー")
    preview_columns = st.columns(2)

    with preview_columns[0]:
        st.caption("お手本のダンス動画")
        render_video_info(reference_video)
        render_uploaded_video(reference_video)

    with preview_columns[1]:
        st.caption("自分のダンス動画")
        render_video_info(practice_video)
        render_uploaded_video(practice_video)

    if reference_video is not None and practice_video is not None:
        if st.button("骨格推定を実行", type="primary"):
            st.subheader("骨格推定結果")
            result_columns = st.columns(2)

            with result_columns[0]:
                reference_result = render_pose_estimation(
                    reference_video,
                    "お手本の動画",
                )

            with result_columns[1]:
                practice_result = render_pose_estimation(
                    practice_video,
                    "自分の動画",
                )

            aligned_frames = render_alignment_info(
                reference_result,
                practice_result,
            )
            render_comparison_info(aligned_frames)


if __name__ == "__main__":
    main()
