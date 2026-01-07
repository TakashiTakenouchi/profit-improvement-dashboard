# MLZero (AutoGluon Assistant) 完全セットアップガイド

## 目次

1. [概要](#概要)
2. [前提条件](#前提条件)
3. [インストール手順](#インストール手順)
4. [APIキーの設定](#apiキーの設定)
5. [Web UIの起動](#web-uiの起動)
6. [ブラウザアクセス問題解決方法](#ブラウザアクセス問題解決方法)
7. [トラブルシューティング](#トラブルシューティング)
8. [よくある問題と解決方法](#よくある問題と解決方法)

---

## 概要

MLZero（別名：AutoGluon Assistant）は、複数のLLMエージェントを統合してデータ分析からモデル構築までを自動化するシステムです。Web UIとCLIの両方のインターフェースを提供しています。

### 主な機能

- データ分析の自動化
- モデル構築の自動化
- 時系列予測
- 分類・回帰タスク
- Web UIによるインタラクティブな操作

---

## 前提条件

### システム要件

- **OS**: Linux（WSL経由でWindowsでも利用可能）
- **Python**: 3.8 - 3.11
- **パッケージマネージャー**: pip または uv（推奨）

### Windows環境での利用

Windows環境では、**WSL（Windows Subsystem for Linux）**が必要です。

**WSLの確認:**
```bash
wsl --status
```

WSLがインストールされていない場合:
```bash
wsl --install
```

---

## インストール手順

### ステップ1: WSL環境の準備

Windows環境の場合、WSLターミナルを開きます：

```bash
wsl
```

### ステップ2: pipのアップグレード

```bash
pip3 install --upgrade pip
```

### ステップ3: uvのインストール（推奨）

高速なパッケージマネージャー`uv`を使用します：

```bash
pip3 install uv
```

または、curlを使用して直接インストール：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**PATHにuvを追加:**
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### ステップ4: MLZeroのインストール

**方法1: uvを使用（推奨）**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv pip install autogluon.assistant>=1.0
```

**方法2: GitHubから直接インストール**

```bash
uv pip install git+https://github.com/autogluon/autogluon-assistant.git
```

**方法3: pipで直接インストール**

```bash
pip3 install autogluon.assistant>=1.0
```

### ステップ5: インストール確認

```bash
# コマンドの確認
export PATH="$HOME/.local/bin:$PATH"
which mlzero-backend
which mlzero-frontend

# Pythonモジュールの確認
python3 -c "import autogluon.assistant; print('✓ AutoGluon Assistant インストール成功')"
```

**期待される出力:**
```
/home/takenouchiy/.local/bin/mlzero-backend
/home/takenouchiy/.local/bin/mlzero-frontend
✓ AutoGluon Assistant インストール成功
```

---

## APIキーの設定

### OpenAI APIキーの取得

1. [OpenAIの公式サイト](https://openai.com/)でアカウントを作成
2. APIキーを取得

### APIキーの設定

**一時的な設定（現在のセッションのみ）:**

```bash
export OPENAI_API_KEY="your-api-key-here"
```

**永続的な設定（推奨）:**

```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**設定確認:**
```bash
echo $OPENAI_API_KEY
```

---

## Web UIの起動

### 公式の起動手順

MLZeroのWeb UIは、**バックエンド**と**フロントエンド**の2つのプロセスで構成されています。

### ステップ1: バックエンドサーバーの起動

**ターミナル1**で実行：

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key-here"
mlzero-backend
```

**期待される出力:**
```
INFO:autogluon.assistant.webui.backend.queue.manager: QueueManager initialized
INFO:autogluon.assistant.webui.backend.queue.manager: Executor loop started
* Serving Flask app 'autogluon.assistant.webui.backend.app'
* Running on http://127.0.0.1:5000
```

### ステップ2: フロントエンドの起動

**ターミナル2**（別のターミナル）で実行：

**Windowsからアクセス可能にする場合（推奨）:**

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key-here"
streamlit run /home/takenouchiy/.local/lib/python3.10/site-packages/autogluon/assistant/webui/Launch_MLZero.py --server.port=8509 --server.address=0.0.0.0
```

**WSL内からのみアクセスする場合:**

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key-here"
mlzero-frontend
```

**期待される出力:**
```
Starting AutoGluon Assistant Frontend on http://localhost:8509
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8509
```

### ステップ3: Web UIへのアクセス

ブラウザで以下のURLにアクセス：

```
http://localhost:8509
```

### 起動スクリプトの使用

**Windowsバッチファイル:**

```bash
scripts\start_mlzero_separate.bat
```

**WSLシェルスクリプト:**

```bash
# バックエンド
wsl bash scripts/start_backend.sh

# フロントエンド（別ターミナル）
wsl bash scripts/start_frontend.sh
```

**再起動スクリプト:**

```bash
wsl bash scripts/restart_mlzero.sh
```

---

## ブラウザアクセス問題解決方法

### 問題

フロントエンドは起動しているが、ブラウザで `http://localhost:8509` にアクセスできない。

### 確認事項

#### 1. バックエンドが起動しているか

```bash
wsl bash -c "ps aux | grep mlzero-backend | grep -v grep"
```

バックエンドが起動していない場合、起動してください：

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key"
mlzero-backend
```

#### 2. ポートが正しくリスニングしているか

```bash
# WSL内で確認
wsl bash -c "ss -tuln | grep 8509"
wsl bash -c "curl http://localhost:8509"
```

#### 3. WSLからWindowsへのポートフォワーディング

WSL内で起動したアプリケーションにWindowsのブラウザからアクセスするには、ポートフォワーディングが必要な場合があります。

**確認方法:**
- WSL内で `http://localhost:8509` にアクセスできるか確認
- Windowsのブラウザで `http://localhost:8509` にアクセスできない場合、ポートフォワーディングが必要

**ポートフォワーディング設定:**

PowerShell（管理者権限）で実行：

```powershell
# WSLのIPアドレスを取得
$wslIP = (wsl hostname -I).Trim()

# ポートフォワーディング設定
netsh interface portproxy add v4tov4 listenport=8509 listenaddress=0.0.0.0 connectport=8509 connectaddress=$wslIP

# 設定確認
netsh interface portproxy show v4tov4
```

または、スクリプトを使用：

```powershell
.\scripts\setup_wsl_port_forwarding.ps1
```

#### 4. ファイアウォール設定

Windowsファイアウォールでポート8509を許可：

```powershell
New-NetFirewallRule -DisplayName "MLZero Web UI" -Direction Inbound -LocalPort 8509 -Protocol TCP -Action Allow
```

#### 5. ブラウザのキャッシュをクリア

ブラウザのキャッシュをクリアして、再度アクセスを試みてください。

### 解決手順

#### ステップ1: バックエンドとフロントエンドの状態確認

```bash
wsl bash scripts/check_mlzero.sh
```

#### ステップ2: WSL内でアクセステスト

```bash
wsl
curl http://localhost:8509
```

WSL内でアクセスできる場合、ポートフォワーディングの問題です。

#### ステップ3: ポートフォワーディング設定

```powershell
# PowerShell（管理者権限）
.\scripts\setup_wsl_port_forwarding.ps1
```

#### ステップ4: ブラウザでアクセス

```
http://localhost:8509
```

### 代替アクセス方法

#### 方法1: WSLのIPアドレスに直接アクセス

```bash
# WSLのIPアドレスを取得
wsl hostname -I
```

ブラウザで `http://[WSLのIPアドレス]:8509` にアクセス

#### 方法2: Windowsのlocalhostに直接アクセス

ポートフォワーディングを設定した後、`http://localhost:8509` にアクセス

### トラブルシューティング

#### エラー: "This site can't be reached"

1. バックエンドが起動しているか確認
2. ポートフォワーディングが設定されているか確認
3. ファイアウォールでポート8509が許可されているか確認

#### エラー: "Connection refused"

1. フロントエンドが正しく起動しているか確認
2. ポート8509が使用されているか確認
3. 別のプロセスがポート8509を使用していないか確認

---

## トラブルシューティング

### 問題1: コマンドが見つからない

**症状:**
```
bash: mlzero-backend: command not found
```

**解決方法:**

```bash
# PATHを確認
echo $PATH

# PATHに追加
export PATH="$HOME/.local/bin:$PATH"

# コマンドの場所を確認
which mlzero-backend
ls -la ~/.local/bin/ | grep mlzero
```

### 問題2: ポート8509が既に使用中

**症状:**
```
Port 8509 is already in use
```

**解決方法:**

```bash
# 既存プロセスを停止
wsl bash scripts/stop_mlzero.sh

# または手動で停止
pkill -f "mlzero-frontend"
pkill -f "streamlit.*8509"
lsof -ti :8509 | xargs kill -9
```

### 問題3: Windowsブラウザからアクセスできない

**症状:**
フロントエンドは起動しているが、Windowsのブラウザで `http://localhost:8509` にアクセスできない。

**解決方法:**

フロントエンドを`0.0.0.0`で起動：

```bash
streamlit run /home/takenouchiy/.local/lib/python3.10/site-packages/autogluon/assistant/webui/Launch_MLZero.py --server.port=8509 --server.address=0.0.0.0
```

### 問題4: 無効なモデル名エラー

**症状:**
```
ValueError: Invalid model: ChatGPT5 for provider openai
```

**解決方法:**

設定ファイルのモデル名を修正：

```bash
# すべての設定ファイルでChatGPT5をgpt-4oに変更
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: ChatGPT5/model: gpt-4o/g' {} \;
```

### 問題5: コンテキスト長エラー

**症状:**
```
BadRequestError: This model's maximum context length is 8192 tokens. However, you requested 16690 tokens
```

**解決方法:**

`gpt-4`を`gpt-4-turbo`または`gpt-4o`に変更：

```bash
# gpt-4をgpt-4-turboに変更
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: gpt-4$/model: gpt-4-turbo/g' {} \;
```

---

## よくある問題と解決方法

### 設定ファイルの修正

**モデル名の一括修正:**

```bash
# ChatGPT5 → gpt-4o
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: ChatGPT5/model: gpt-4o/g' {} \;

# gpt-4 → gpt-4-turbo
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: gpt-4$/model: gpt-4-turbo/g' {} \;
```

**修正スクリプトの使用:**

```bash
# モデル名修正
wsl bash scripts/fix_mlzero_config.sh gpt-4o

# コンテキスト長エラー修正
wsl bash scripts/fix_mlzero_context_length.sh gpt-4-turbo
```

### プロセスの管理

**状態確認:**

```bash
wsl bash scripts/check_mlzero.sh
```

**停止:**

```bash
wsl bash scripts/stop_mlzero.sh
```

**再起動:**

```bash
wsl bash scripts/restart_mlzero.sh
```

### ポートフォワーディング（必要に応じて）

WSL内のポートにWindowsからアクセスできない場合：

**PowerShell（管理者権限）で実行:**

```powershell
# WSLのIPアドレスを取得
$wslIP = (wsl hostname -I).Trim()

# ポートフォワーディング設定
netsh interface portproxy add v4tov4 listenport=8509 listenaddress=0.0.0.0 connectport=8509 connectaddress=$wslIP

# 設定確認
netsh interface portproxy show v4tov4
```

**スクリプトを使用:**

```powershell
.\scripts\setup_wsl_port_forwarding.ps1
```

---

## 使用可能なモデル名

### OpenAIプロバイダー

| モデル名 | 最大コンテキスト長 | 推奨max_tokens | 備考 |
|---------|-------------------|---------------|------|
| `gpt-4o` | 128,000 | 16,384 | 最新・推奨 |
| `gpt-4-turbo` | 128,000 | 16,384 | 高性能 |
| `gpt-4` | 8,192 | 7,000 | コンテキスト長が小さい |
| `gpt-3.5-turbo` | 16,385 | 4,096 | コスト効率が良い |

**注意:** `ChatGPT5`は無効なモデル名です。使用しないでください。

---

## クイックリファレンス

### インストール

```bash
wsl
pip3 install uv
export PATH="$HOME/.local/bin:$PATH"
uv pip install autogluon.assistant>=1.0
```

### APIキー設定

```bash
echo 'export OPENAI_API_KEY="your-api-key"' >> ~/.bashrc
source ~/.bashrc
```

### 起動

**ターミナル1（バックエンド）:**
```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
mlzero-backend
```

**ターミナル2（フロントエンド）:**
```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
streamlit run /home/takenouchiy/.local/lib/python3.10/site-packages/autogluon/assistant/webui/Launch_MLZero.py --server.port=8509 --server.address=0.0.0.0
```

### 停止

```bash
wsl bash scripts/stop_mlzero.sh
```

### 再起動

```bash
wsl bash scripts/restart_mlzero.sh
```

---

## 参考リンク

- [AutoGluon Assistant GitHub](https://github.com/autogluon/autogluon-assistant)
- [MLZero Documentation](https://mlzero.ai/)
- [WSL Documentation](https://docs.microsoft.com/ja-jp/windows/wsl/)

---

## 付録：作成されたスクリプト一覧

### インストール関連

- `scripts/install_mlzero_wsl.sh` - MLZeroインストールスクリプト
- `scripts/set_env.sh` - 環境変数設定スクリプト

### 起動関連

- `scripts/start_backend.sh` - バックエンド起動スクリプト
- `scripts/start_frontend.sh` - フロントエンド起動スクリプト（0.0.0.0対応）
- `scripts/start_mlzero_separate.bat` - Windowsバッチファイル
- `scripts/restart_mlzero.sh` - 再起動スクリプト

### 管理関連

- `scripts/stop_mlzero.sh` - プロセス停止スクリプト
- `scripts/check_mlzero.sh` - 状態確認スクリプト
- `scripts/verify_mlzero.sh` - 起動確認スクリプト

### 設定修正関連

- `scripts/fix_mlzero_config.sh` - モデル名修正スクリプト
- `scripts/fix_mlzero_context_length.sh` - コンテキスト長エラー修正スクリプト

### ポート関連

- `scripts/setup_wsl_port_forwarding.ps1` - ポートフォワーディング設定スクリプト

---

## 更新履歴

- 2025-12-30: 初版作成
  - インストール手順
  - Web UI起動手順
  - トラブルシューティング
  - よくある問題と解決方法
  - ブラウザアクセス問題解決方法を追加
