# MLZero (AutoGluon Assistant) Web UI インストールガイド

## 環境確認

- **OS**: Windows 10/11
- **WSL**: Ubuntu 22.04 ✅ インストール済み
- **Python**: 3.10.12 (WSL内) ✅
- **pip**: 22.0.2 (WSL内) ✅
- **Conda**: 未インストール（必要に応じて後でインストール）

## インストール手順

### 方法1: uvを使用したインストール（推奨）

MLZeroは`uv`パッケージマネージャーを使用したインストールを推奨しています。

#### Step 1: uvのインストール

WSLターミナルで実行：

```bash
# WSLに入る
wsl

# uvをインストール
pip3 install uv

# または、curlを使用して直接インストール
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Step 2: MLZeroのインストール

```bash
# AutoGluon Assistantをインストール
uv pip install autogluon.assistant>=1.0

# または、GitHubから直接インストール
uv pip install git+https://github.com/autogluon/autogluon-assistant.git
```

### 方法2: Condaを使用したインストール

Condaが必要な場合は、まずMinicondaまたはAnacondaをインストール：

```bash
# WSL内でMinicondaをインストール
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Conda環境を作成
conda create -n mlzero python=3.10
conda activate mlzero

# MLZeroをインストール
pip install autogluon.assistant>=1.0
```

## Web UIの起動

### Step 1: LLMプロバイダーの設定

デフォルトではAWS Bedrockが使用されます。OpenAIを使用する場合：

```bash
# WSL内で環境変数を設定
export OPENAI_API_KEY="your-api-key-here"
```

または、`.bashrc`に追加：

```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: バックエンドサーバーの起動

**重要**: PATHに`$HOME/.local/bin`を追加する必要があります。

```bash
# WSL内で実行
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key-here"
mlzero-backend
```

デフォルトでは、バックエンドは `http://localhost:8509` で起動します。

### Step 3: フロントエンドの起動（別ターミナル）

新しいWSLターミナルを開いて：

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key-here"
mlzero-frontend
```

**注意**: `mlzero-backend`と`mlzero-frontend`コマンドは`~/.local/bin/`にインストールされます。
PATHに追加されていない場合は、上記の`export PATH`コマンドを実行してください。

### Step 4: Web UIへのアクセス

Windowsのブラウザから以下のURLにアクセス：

```
http://localhost:8509
```

**注意**: WSL内のポートをWindowsからアクセスできるようにするには、ポートフォワーディングの設定が必要な場合があります。

## Cursorからの利用方法

### 1. WSLターミナルの統合

CursorはWSLターミナルを統合できます：

1. Cursorの設定でターミナルをWSLに設定
2. 統合ターミナルから直接WSLコマンドを実行可能

### 2. ポートフォワーディング設定

WSL内で起動したWeb UIにWindowsからアクセスするには、ポートフォワーディングが必要です。

PowerShell（管理者権限）で実行：

```powershell
# WSLのIPアドレスを取得
wsl hostname -I

# ポートフォワーディング設定（例：WSL IPが172.x.x.xの場合）
netsh interface portproxy add v4tov4 listenport=8509 listenaddress=0.0.0.0 connectport=8509 connectaddress=172.x.x.x
```

### 3. 自動起動スクリプトの作成

`start_mlzero.sh`を作成：

```bash
#!/bin/bash
# WSL内で実行

# 環境変数の読み込み
source ~/.bashrc

# バックエンドをバックグラウンドで起動
mlzero-backend &

# フロントエンドを起動
mlzero-frontend
```

実行権限を付与：

```bash
chmod +x start_mlzero.sh
```

## トラブルシューティング

### 問題1: ポート8509にアクセスできない

**解決策**:
1. WSL内でファイアウォール設定を確認
2. Windowsファイアウォールでポート8509を許可
3. ポートフォワーディングを設定

### 問題2: Condaが見つからない

**解決策**:
- `uv`を使用したインストール方法を試す（Conda不要）
- または、Minicondaをインストール

### 問題3: 依存関係のエラー

**解決策**:
```bash
# Pythonのバージョンを確認（3.8-3.11が必要）
python3 --version

# pipをアップグレード
pip3 install --upgrade pip

# 依存関係を再インストール
uv pip install --upgrade autogluon.assistant
```

## Python APIの使用（Web UIなし）

Web UIが不要な場合は、Python APIを直接使用できます：

```python
from autogluon.assistant import run_agent

# 基本的な使用法
run_agent(
    input_data_folder="./data",
    output_folder="./output",
    initial_user_input="Create a classification model"
)
```

この方法なら、Cursor内で直接Pythonコードとして実行可能です。

## 参考リンク

- [AutoGluon Assistant GitHub](https://github.com/autogluon/autogluon-assistant)
- [MLZero Documentation](https://mlzero.ai/)
- [WSL Documentation](https://docs.microsoft.com/ja-jp/windows/wsl/)

