# MLZero ポート競合エラー解決方法

## エラー内容

```
Port 8509 is already in use
```

## 問題の原因

既存のMLZeroフロントエンドプロセス（Streamlit）がポート8509を使用しているため、新しいフロントエンドを起動できません。

## 解決方法

### 方法1: 停止スクリプトを使用（推奨）

```bash
wsl bash scripts/stop_mlzero.sh
```

このスクリプトは：
1. 実行中のすべてのMLZeroプロセスを確認
2. バックエンド、フロントエンド、Streamlitプロセスを停止
3. ポート8509を解放

### 方法2: 手動で停止

```bash
wsl
# プロセスを確認
ps aux | grep mlzero | grep -v grep
ps aux | grep streamlit | grep -v grep

# プロセスを停止
pkill -f "mlzero-backend"
pkill -f "mlzero-frontend"
pkill -f "streamlit.*8509"

# ポート8509を使用しているプロセスを確認・停止
lsof -ti :8509 | xargs kill -9
```

### 方法3: ポートを確認して停止

```bash
wsl
# ポート8509を使用しているプロセスを確認
lsof -i :8509
# または
ss -tuln | grep 8509

# プロセスIDを確認して停止
kill [PID]
# または強制終了
kill -9 [PID]
```

## 再起動手順

プロセスを停止した後、再起動してください：

```bash
# 再起動スクリプトを使用
wsl bash scripts/restart_mlzero.sh

# または手動で起動
# ターミナル1（バックエンド）
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key"
mlzero-backend

# ターミナル2（フロントエンド）
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key"
mlzero-frontend
```

## 確認方法

```bash
# プロセスが起動しているか確認
wsl bash -c "ps aux | grep -E 'mlzero|streamlit.*8509' | grep -v grep"

# ポート8509が使用されているか確認
wsl bash -c "lsof -i :8509 || ss -tuln | grep 8509"

# Web UIにアクセス
# http://localhost:8509
```

## トラブルシューティング

### プロセスが停止しない場合

```bash
# 強制終了
pkill -9 -f "mlzero-backend"
pkill -9 -f "mlzero-frontend"
pkill -9 -f "streamlit"

# ポートを強制的に解放
lsof -ti :8509 | xargs kill -9
```

### 複数のプロセスが実行中の場合

```bash
# すべてのMLZero関連プロセスを確認
ps aux | grep -E 'mlzero|autogluon.*assistant' | grep -v grep

# すべてを停止
pkill -9 -f "mlzero"
pkill -9 -f "autogluon.*assistant"
```

## 予防策

- 新しいMLZeroを起動する前に、既存プロセスを停止する習慣をつける
- 設定ファイルを変更した後は、必ず再起動する
- ポート競合が発生した場合は、`stop_mlzero.sh`スクリプトを使用する



