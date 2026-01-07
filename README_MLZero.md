# MLZero (AutoGluon Assistant) Web UI クイックスタート

## 前提条件

- ✅ WSL (Ubuntu 22.04) インストール済み
- ✅ AutoGluon Assistant インストール済み
- ✅ OpenAI APIキー設定済み

## 起動方法

### 方法1: 公式ドキュメントに基づく起動（推奨）

**ターミナル1（バックエンド）:**
```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key-here"
mlzero-backend
```

**ターミナル2（フロントエンド）:**
```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key-here"
mlzero-frontend
```

### 方法2: 起動スクリプトを使用

**Windowsバッチファイル:**
```bash
scripts\start_mlzero_separate.bat
```

**WSLシェルスクリプト:**
```bash
wsl bash scripts/start_mlzero_official.sh
```

## Web UIへのアクセス

起動後、ブラウザで以下にアクセス：
```
http://localhost:8509
```

## ポートフォワーディング（必要に応じて）

WSL内のポートにWindowsからアクセスできない場合：

```powershell
# PowerShell（管理者権限）で実行
.\scripts\setup_wsl_port_forwarding.ps1
```

## トラブルシューティング

### コマンドが見つからない

```bash
# PATHを確認
echo $PATH

# PATHに追加
export PATH="$HOME/.local/bin:$PATH"

# コマンドの場所を確認
which mlzero-backend
which mlzero-frontend
```

### APIキーが設定されていない

```bash
# 一時的に設定
export OPENAI_API_KEY="your-api-key-here"

# 永続的に設定
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### ポート8509にアクセスできない

1. WSL内でポートがリスニングしているか確認
2. Windowsファイアウォールでポート8509を許可
3. ポートフォワーディングを設定

## 参考

- [インストールガイド](docs/MLZero_Installation_Guide.md)
- [トラブルシューティング](docs/MLZero_Troubleshooting.md)



