# MLZero (AutoGluon Assistant) セットアップ完全ガイド

## 📋 目次

1. [はじめに](#はじめに)
2. [インストール](#インストール)
3. [APIキー設定](#apiキー設定)
4. [Web UI起動](#web-ui起動)
5. [トラブルシューティング](#トラブルシューティング)

---

## はじめに

MLZero（AutoGluon Assistant）は、LLMエージェントを統合してデータ分析からモデル構築までを自動化するシステムです。

### システム要件

- **OS**: Linux（WindowsはWSL経由）
- **Python**: 3.8 - 3.11
- **WSL**: Windows環境では必須

---

## インストール

### 1. WSL環境の確認

```bash
wsl --status
```

WSLがインストールされていない場合：
```bash
wsl --install
```

### 2. WSLターミナルを開く

```bash
wsl
```

### 3. pipのアップグレード

```bash
pip3 install --upgrade pip
```

### 4. uvのインストール（推奨）

```bash
pip3 install uv
export PATH="$HOME/.local/bin:$PATH"
```

### 5. MLZeroのインストール

```bash
export PATH="$HOME/.local/bin:$PATH"
uv pip install autogluon.assistant>=1.0
```

### 6. インストール確認

```bash
which mlzero-backend
which mlzero-frontend
python3 -c "import autogluon.assistant; print('✓ インストール成功')"
```

---

## APIキー設定

### OpenAI APIキーの取得

1. [OpenAI公式サイト](https://openai.com/)でアカウント作成
2. APIキーを取得

### APIキーの設定

**永続的な設定（推奨）:**

```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**確認:**
```bash
echo $OPENAI_API_KEY
```

---

## Web UI起動

### 重要：2つのターミナルが必要

MLZeroのWeb UIは、**バックエンド**と**フロントエンド**の2つのプロセスで構成されています。

### ステップ1: バックエンドの起動

**ターミナル1**で実行：

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key-here"
mlzero-backend
```

**成功の確認:**
```
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

**成功の確認:**
```
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8509
```

### ステップ3: ブラウザでアクセス

```
http://localhost:8509
```

---

## トラブルシューティング

### ❌ コマンドが見つからない

```bash
export PATH="$HOME/.local/bin:$PATH"
which mlzero-backend
```

### ❌ ポート8509が使用中

```bash
wsl bash scripts/stop_mlzero.sh
```

### ❌ Windowsブラウザからアクセスできない

フロントエンドを`0.0.0.0`で起動：

```bash
streamlit run /home/takenouchiy/.local/lib/python3.10/site-packages/autogluon/assistant/webui/Launch_MLZero.py --server.port=8509 --server.address=0.0.0.0
```

### ❌ 無効なモデル名エラー

```bash
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: ChatGPT5/model: gpt-4o/g' {} \;
```

### ❌ コンテキスト長エラー

```bash
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: gpt-4$/model: gpt-4-turbo/g' {} \;
```

---

## 便利なスクリプト

### 状態確認

```bash
wsl bash scripts/check_mlzero.sh
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

## クイックスタート

```bash
# 1. インストール
wsl
pip3 install uv
export PATH="$HOME/.local/bin:$PATH"
uv pip install autogluon.assistant>=1.0

# 2. APIキー設定
echo 'export OPENAI_API_KEY="your-api-key"' >> ~/.bashrc
source ~/.bashrc

# 3. バックエンド起動（ターミナル1）
mlzero-backend

# 4. フロントエンド起動（ターミナル2）
streamlit run /home/takenouchiy/.local/lib/python3.10/site-packages/autogluon/assistant/webui/Launch_MLZero.py --server.port=8509 --server.address=0.0.0.0

# 5. ブラウザで http://localhost:8509 にアクセス
```

---

## 参考資料

詳細なドキュメント:
- `docs/MLZero_Complete_Setup_Guide.md` - 完全ガイド
- `docs/MLZero_Installation_Guide.md` - インストールガイド
- `docs/MLZero_Troubleshooting.md` - トラブルシューティング



