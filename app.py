import streamlit as st


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


def render_uploaded_video(video_file) -> None:
    """アップロードされた動画をプレビューする。"""
    if video_file is not None:
        st.video(video_file)


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


if __name__ == "__main__":
    main()
