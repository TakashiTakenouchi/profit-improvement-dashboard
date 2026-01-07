# MLZero Windowsブラウザアクセス問題の解決

## 問題

フロントエンドは起動しているが、Windowsのブラウザで `http://localhost:8509` にアクセスできない。

## 原因

フロントエンドが `localhost` (127.0.0.1) のみでリスニングしているため、WSL内からのみアクセス可能で、Windowsのブラウザからはアクセスできません。

## 解決方法

### 方法1: フロントエンドを0.0.0.0で起動（推奨）

**現在のフロントエンドを停止:**

```bash
wsl
pkill -f "mlzero-frontend"
pkill -f "streamlit.*8509"
```

**0.0.0.0で再起動:**

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="your-api-key"
streamlit run /home/takenouchiy/.local/lib/python3.10/site-packages/autogluon/assistant/webui/Launch_MLZero.py --server.port=8509 --server.address=0.0.0.0
```

または、修正済みスクリプトを使用：

```bash
wsl bash scripts/start_frontend.sh
```

### 方法2: ポートフォワーディング設定

PowerShell（管理者権限）で実行：

```powershell
# WSLのIPアドレスを取得
$wslIP = (wsl hostname -I).Trim()

# ポートフォワーディング設定
netsh interface portproxy add v4tov4 listenport=8509 listenaddress=0.0.0.0 connectport=8509 connectaddress=$wslIP

# 設定確認
netsh interface portproxy show v4tov4
```

### 方法3: WSLのIPアドレスに直接アクセス

```bash
# WSLのIPアドレスを取得
wsl hostname -I
```

ブラウザで `http://[WSLのIPアドレス]:8509` にアクセス

例: `http://172.18.90.232:8509`

## 確認方法

### WSL内でアクセステスト

```bash
wsl
curl http://localhost:8509
```

HTTP 200が返れば、WSL内では正常に動作しています。

### Windowsブラウザでアクセス

方法1を使用した場合: `http://localhost:8509`
方法2を使用した場合: `http://localhost:8509`
方法3を使用した場合: `http://[WSLのIPアドレス]:8509`

## 推奨解決方法

**方法1（0.0.0.0で起動）を推奨**します。理由：
- ポートフォワーディング設定が不要
- Windowsから直接アクセス可能
- 設定が簡単

## 注意事項

- `0.0.0.0`でバインドすると、すべてのネットワークインターフェースからアクセス可能になります
- ファイアウォールでポート8509が許可されていることを確認してください
- セキュリティが重要な環境では、適切なファイアウォール設定を行ってください



