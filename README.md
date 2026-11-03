# DanceSync

お手本動画と練習動画の姿勢を比較し、DTWで対応付けたフレームをもとに改善アドバイスを表示するStreamlitアプリです。

## 必要な環境

- Windows
- Python 3.10以上を推奨
- PowerShell

## 起動手順

PowerShellでプロジェクトのルートディレクトリに移動します。
仮想環境を作成して有効化します。すでに`venv`がある場合は作成を省略してください。

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

依存パッケージをインストールします。

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

アプリを起動します。

```powershell
streamlit run app.py
```

ブラウザで表示されたURL（通常は`http://localhost:8501`）を開きます。

## testディレクトリの動画で検証する方法

アプリは`test`ディレクトリを自動では読み込まないため、起動後に画面上のファイル選択欄から動画をアップロードします。

```text
test/
├─ TestExample.mp4
├─ TestPractice.mp4
└─ TestBendElbow.mp4
```

各動画の用途は次のとおりです。

- `TestExample.mp4`: お手本動画
- `TestPractice.mp4`: お手本と同じ振付を、意図的な姿勢変更を加えずに踊った練習動画
- `TestBendElbow.mp4`: お手本を基準に、腕を意図的に曲げて踊った練習動画

### 通常の比較

1. 「お手本のダンス動画」に`test\TestExample.mp4`を選択します。
2. 「自分のダンス動画」に`test\TestPractice.mp4`を選択します。
3. 「骨格推定を実行」をクリックします。
4. 同じ動きの動画同士で、DTWによるフレーム対応や改善アドバイスが適切か確認します。

### 肘を曲げた動画の比較

1. お手本として`test\TestExample.mp4`を選択します。
2. 練習動画として`test\TestBendElbow.mp4`を選択します。
3. 「骨格推定を実行」をクリックします。
4. 意図的に腕を曲げたことによる肘の角度差と改善アドバイスを確認します。

## 注意事項

- アップロードできる動画形式はMP4です。
- できるだけ全身が映り、正面から撮影された動画を使用してください。
- 動画を2本とも選択してから「骨格推定を実行」をクリックしてください。
- MediaPipeの初回実行時はモデルの読み込みに時間がかかる場合があります。
- PowerShellで仮想環境を有効化できない場合は、次のコマンドを一時的に実行してから再試行してください。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 終了方法

Streamlitを起動したPowerShellで`Ctrl+C`を押します。
