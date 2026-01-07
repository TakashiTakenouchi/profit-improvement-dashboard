# MLZero ブラウザアクセス問題解決方法

> **注意**: より詳細な情報については、[MLZero_Complete_Setup_Guide.md](./MLZero_Complete_Setup_Guide.md) を参照してください。

## 問題

フロントエンドは起動しているが、ブラウザで `http://localhost:8509` にアクセスできない。

## 確認事項

### 1. バックエンドが起動しているか

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

### 2. ポートが正しくリスニングしているか

```bash
# WSL内で確認
wsl bash -c "ss -tuln | grep 8509"
wsl bash -c "curl http://localhost:8509"
```

### 3. WSLからWindowsへのポートフォワーディング

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

### 4. ファイアウォール設定

Windowsファイアウォールでポート8509を許可：

```powershell
New-NetFirewallRule -DisplayName "MLZero Web UI" -Direction Inbound -LocalPort 8509 -Protocol TCP -Action Allow
```

### 5. ブラウザのキャッシュをクリア

ブラウザのキャッシュをクリアして、再度アクセスを試みてください。

## 解決手順

### ステップ1: バックエンドとフロントエンドの状態確認

```bash
wsl bash scripts/check_mlzero.sh
```

### ステップ2: WSL内でアクセステスト

```bash
wsl
curl http://localhost:8509
```

WSL内でアクセスできる場合、ポートフォワーディングの問題です。

### ステップ3: ポートフォワーディング設定

```powershell
# PowerShell（管理者権限）
.\scripts\setup_wsl_port_forwarding.ps1
```

### ステップ4: ブラウザでアクセス

```
http://localhost:8509
```

## 代替アクセス方法

### 方法1: WSLのIPアドレスに直接アクセス

```bash
# WSLのIPアドレスを取得
wsl hostname -I
```

ブラウザで `http://[WSLのIPアドレス]:8509` にアクセス

### 方法2: Windowsのlocalhostに直接アクセス

ポートフォワーディングを設定した後、`http://localhost:8509` にアクセス

## トラブルシューティング

### エラー: "This site can't be reached"

1. バックエンドが起動しているか確認
2. ポートフォワーディングが設定されているか確認
3. ファイアウォールでポート8509が許可されているか確認

### エラー: "Connection refused"

1. フロントエンドが正しく起動しているか確認
2. ポート8509が使用されているか確認
3. 別のプロセスがポート8509を使用していないか確認

## 関連ドキュメント

- [MLZero_Complete_Setup_Guide.md](./MLZero_Complete_Setup_Guide.md) - 完全セットアップガイド
- [MLZero_Troubleshooting.md](./MLZero_Troubleshooting.md) - トラブルシューティングガイド
