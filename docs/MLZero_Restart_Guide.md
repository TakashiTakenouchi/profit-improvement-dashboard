# MLZero 再起動ガイド

## 再起動が必要な場合

設定ファイルを変更した後は、MLZeroを再起動する必要があります。

## 再起動方法

### 方法1: 既存プロセスを停止してから再起動（推奨）

**ステップ1: 既存のプロセスを停止**

実行中のMLZeroプロセスを確認・停止：

```bash
wsl
# プロセスを確認
ps aux | grep mlzero | grep -v grep

# プロセスを停止
pkill -f "mlzero-backend"
pkill -f "mlzero-frontend"
```

**ステップ2: 新規ターミナルで再起動**

**ターミナル1（バックエンド）:**
```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key"
mlzero-backend
```

**ターミナル2（フロントエンド）:**
```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key"
mlzero-frontend
```

### 方法2: 再起動スクリプトを使用

```bash
wsl bash scripts/restart_mlzero.sh
```

このスクリプトは：
1. 既存のプロセスを自動的に停止
2. バックエンドを起動
3. フロントエンドを起動

### 方法3: 既存プロセスをそのままにして新規起動（非推奨）

**注意**: この方法は推奨されません。複数のプロセスが競合する可能性があります。

## プロセス確認コマンド

```bash
# 実行中のMLZeroプロセスを確認
wsl bash -c "ps aux | grep -E 'mlzero|python.*(backend|frontend)' | grep -v grep"

# ポート8509を使用しているプロセスを確認
wsl bash -c "lsof -i :8509 2>/dev/null || ss -tuln | grep 8509"
```

## トラブルシューティング

### プロセスが停止しない場合

```bash
# 強制終了
pkill -9 -f "mlzero-backend"
pkill -9 -f "mlzero-frontend"

# ポートを確認
lsof -i :8509
# 必要に応じてプロセスを強制終了
kill -9 [PID]
```

### ポートが使用中の場合

```bash
# ポート8509を使用しているプロセスを確認
wsl bash -c "lsof -i :8509 2>/dev/null || ss -tuln | grep 8509"

# プロセスを停止
kill [PID]
```

## 推奨手順

1. **既存プロセスを停止** - 設定変更後は必ず停止
2. **新規ターミナルで起動** - クリーンな状態で起動
3. **設定を確認** - Web UIで新しい設定が反映されているか確認

## 注意事項

- 設定ファイルを変更した後は**必ず再起動**してください
- 既存プロセスを停止せずに新規起動すると、ポート競合が発生する可能性があります
- バックエンドとフロントエンドは**別々のターミナル**で実行する必要があります



