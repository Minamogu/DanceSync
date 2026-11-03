# 深層学習アプリ開発課題 開発記録

- 学籍番号：2432015
- 氏名：葛木美奈
- 提出file名：`DEV_LOG.md`（repository直下に置き、commitして提出する）
- 締切：**11/3（火）23:59**

> AIとのやり取りのうち、**重要な修正3件**を選んで書く。大量の会話ログは不要である。
> 「AIが正しく書いた例」ではなく、「問題があり、自分で確認して判断した例」を選ぶ。

---

## 使用したAIツール

- ツール名・利用した場面：copilot。アプリの実装、修正、デバックに利用した。

---

## 修正1：フレーム抽出の方法を変更

- 何を依頼したか（promptの要点）：

```text
現在は動画から10フレームごとに画像を抽出しているが、一定の時間間隔で姿勢を比較できるように0.1秒ごとにフレームを抽出する方式に変更してください。
また、最大30フレームの制限をなくし、抽出したフレームの元動画でのフレーム番号とタイムスタンプを保存してください。
```

- 何が問題だったか（症状・期待と実際）：フレームを切り取るタイミングが10フレーム感覚だったため、動画のfpsによって切り取り方が変わってしまう。また、最大で30フレームという制限があり、動画の途中で抽出が終了してしまう可能性があった。
- どう確認したか（実行したcommand、表示した値、test）：

```text
@'
from pathlib import Path
from processing.video import extract_frames, read_video_metadata

for name in ("TestExample.mp4", "TestPractice.mp4", "TestBendElbow.mp4"):
    path = Path(name)
    if not path.exists():
        continue

    metadata = read_video_metadata(path)
    frames = extract_frames(path, sample_interval_seconds=0.1)
    timestamps = [frame.timestamp_seconds for frame in frames]
    intervals = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]

    print(f"{name}: fps={metadata.fps:.1f}, extracted={len(frames)}")
    print("  frame_index, timestamp_seconds")
    for frame in frames[:8]:
        print(f"  {frame.frame_index:>3}, {frame.timestamp_seconds:.6f}")
    print("  intervals_first_7:", [f"{x:.6f}" for x in intervals[:7]])
    print(
        "  max_abs_error_from_0.1:",
        f"{max((abs(x - 0.1) for x in intervals), default=0.0):.6f}",
    )
'@ | python -X utf8 -

TestExample.mp4: fps=30.0, extracted=22
  0, 0.000000
  3, 0.100000
  6, 0.200000
  9, 0.300000
 12, 0.400000
 15, 0.500000
 18, 0.600000
 21, 0.700000
  intervals_first_7: ['0.100000', '0.100000', '0.100000',
                      '0.100000', '0.100000', '0.100000', '0.100000']
  max_abs_error_from_0.1: 0.000000


TestPractice.mp4: fps=30.0, extracted=19
  0, 0.000000
  3, 0.100000
  6, 0.200000
  9, 0.300000
 12, 0.400000
 15, 0.500000
 18, 0.600000
 21, 0.700000
  intervals_first_7: ['0.100000', '0.100000', '0.100000',
                      '0.100000', '0.100000', '0.100000', '0.100000']
  max_abs_error_from_0.1: 0.000000

TestBendElbow.mp4: fps=30.0, extracted=23
  0, 0.000000
  3, 0.100000
  6, 0.200000
  9, 0.300000
 12, 0.400000
 15, 0.500000
 18, 0.600000
 21, 0.700000
  intervals_first_7: ['0.100000', '0.100000', '0.100000',
                      '0.100000', '0.100000', '0.100000', '0.100000']
  max_abs_error_from_0.1: 0.000000
```

- どう修正したか（diffの要点、AIの再提案を採用したか）：

```diff
- FPSを考慮し、デフォルトで0.1秒間隔にフレーム抽出
- 最大30フレーム制限を撤廃し、デフォルトで動画全体を処理
-  各フレームに以下を保持
  - 元動画のフレーム番号
  - タイムスタンプ
  - RGB画像
```

- この修正から分かったこと：動画処理ではFPSや実際の経過時間を考慮する必要があるとわかった。異なる動画を比較する場合は、時間間隔を統一することで比較条件をそろえられる。

---

## 修正2：フレームの対応付けをDTWに変更

- 何を依頼したか：骨格特徴量を用いたDTW（Dynamic Time Warping）を導入し、動作速度の違いを考慮して姿勢を対応付けるよう依頼した。
- 何が問題だったか：動画内の相対的な時刻を基準に姿勢を比較していたが、同じ振付でも踊る速度が異なる可能性が高いため、別の動作段階を比較してしまうことがあった。
- どう確認したか：
```
$env:PYTHONPATH = 'C:\DanceSyncRun;' + (Get-Location).Path
@'
from pathlib import Path
from processing.video import extract_frames
from processing.pose_estimation import estimate_pose_frames
from processing.alignment import align_pose_frames, summarize_alignment

def load(path):
    frames = extract_frames(path, sample_interval_seconds=0.1)
    return estimate_pose_frames(frames).frames

reference = load(Path("TestExample.mp4"))
practice = load(Path("TestPractice.mp4"))

aligned = align_pose_frames(
    reference,
    practice,
    band_ratio=0.25,
    max_repeat_count=4,
    min_visibility=0.25,
)
diagnostics = summarize_alignment(aligned)

ref = [x.reference.frame_index for x in aligned]
prac = [x.practice.frame_index for x in aligned]

print("DTW validation: TestExample.mp4 -> TestPractice.mp4")
print(f"pair_count={diagnostics.pair_count}")
print(f"unique_reference_frames={diagnostics.unique_reference_frame_count}")
print(f"unique_practice_frames={diagnostics.unique_practice_frame_count}")
print(f"reference_monotonic={all(a <= b for a, b in zip(ref, ref[1:]))}")
print(f"practice_monotonic={all(a <= b for a, b in zip(prac, prac[1:]))}")
print(f"start=({ref[0]}, {prac[0]})")
print(f"end=({ref[-1]}, {prac[-1]})")
print(f"max_reference_repeat={diagnostics.max_reference_repeat_count}")
print(f"max_practice_repeat={diagnostics.max_practice_repeat_count}")
print(f"mean_feature_distance={diagnostics.mean_feature_distance:.4f}")
print(f"max_feature_distance={diagnostics.max_feature_distance:.4f}")

print("path(frame_index):")
print(" -> ".join(f"({a.reference.frame_index},{a.practice.frame_index})" for a in aligned))

print("path(timestamp_seconds):")
print(" -> ".join(
    f"({a.reference.timestamp_seconds:.1f},{a.practice.timestamp_seconds:.1f})"
    for a in aligned
))
'@ | python -X utf8 -


出力
DTW validation: TestExample.mp4 -> TestPractice.mp4
pair_count=24
unique_reference_frames=22
unique_practice_frames=19
reference_monotonic=True
practice_monotonic=True
start=(0, 0)
end=(63, 54)
max_reference_repeat=3
max_practice_repeat=4
mean_feature_distance=0.4542
max_feature_distance=1.2299

path(frame_index):
(0,0) -> (3,0) -> (6,0) -> (9,3) -> (12,3) -> (15,3) -> (18,3) -> (21,6) -> (24,9) -> (27,12) -> (30,15) -> (33,18) -> (36,21) -> (39,24) -> (42,27) -> (45,30) -> (48,33) -> (51,36) -> (54,39) -> (54,42) -> (54,45) -> (57,48) -> (60,51) -> (63,54)

```

- どう修正したか：時刻ベースの対応付けから、骨格特徴量の類似度に基づくDTWへ変更した。これにより、動画間の動作速度の違いを考慮できるようにした。
- この修正から分かったこと：DTWは動画間の動作速度を合わせることができる一方、完全に人間が同じ動作を再現できるわけではないため必ずしも正しく対応付けられるわけではないと考えられる。

---

## 修正3：改善アドバイスの表示UIを変更

- 何を依頼したか：アドバイスの文章を「腕をもう少し曲げましょう」といった直感的にわかりやすい文章に変更した。また、改善アドバイスの部分の時間と画像を表示するようにした。
- 何が問題だったか：はじめはアドバイスの文章「肘の角度を少し小さくしてください」等の角度の大小でアドバイスを行っており、さらに文章のみなのでアドバイスが伝わりにくかった。
- どう確認したか：お手本動画と練習動画を入力し、アドバイスの表示方法を目視で確認した。
- どう修正したか：文章の変更及び、画像と時間を表示した。
- この修正から分かったこと：AIが作ったUIが人間にとって使いやすいUIであるとは限らない。実際に使うユーザーの視点に立ってどう修正するかは人間がやる必要があると感じた。

---

## 提出前の確認

- [ok] 3件それぞれに、依頼・問題・確認・修正を書いた
- [ok] 確認の証拠（出力・test・値）を貼った
- [ok] AIの提案を採用しなかった理由、または修正した理由を書いた
- [ok] API keyや個人情報を書いていない
