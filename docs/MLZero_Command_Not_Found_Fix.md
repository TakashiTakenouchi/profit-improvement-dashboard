# MLZero コマンドが見つからないエラー解決方法

## エラー内容

```
mlzero-backend: command not found
```

または

```
mlzero-frontend: command not found
```

## 原因

`mlzero-backend`と`mlzero-frontend`コマンドは`~/.local/bin`にインストールされますが、このディレクトリがPATH環境変数に含まれていないため、コマンドが見つかりません。

## 解決方法

### 方法1: PATHを一時的に追加（現在のセッションのみ）

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
mlzero-backend
```

### 方法2: PATHを永続的に追加（推奨）

```bash
wsl
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
mlzero-backend
```

### 方法3: スクリプトを使用

提供されているスクリプトは自動的にPATHを設定します：

```bash
# バックエンド起動
wsl bash scripts/start_backend.sh

# フロントエンド起動
wsl bash scripts/start_frontend.sh
```

## 確認方法

### コマンドの場所を確認

```bash
wsl bash -c "ls -la ~/.local/bin/ | grep mlzero"
```

### PATHを確認

```bash
wsl bash -c "echo \$PATH"
```

### コマンドが使用可能か確認

```bash
wsl bash -c "export PATH=\"\$HOME/.local/bin:\$PATH\" && which mlzero-backend"
```

## トラブルシューティング

### コマンドがインストールされていない場合

```bash
wsl
# インストール確認
python3 -c "import autogluon.assistant; print('インストール済み')"

# コマンドの場所を確認
find ~/.local -name "mlzero-backend" 2>/dev/null
find ~/.local -name "mlzero-frontend" 2>/dev/null
```

### 再インストールが必要な場合

```bash
wsl
pip3 install --upgrade autogluon.assistant
```

## 推奨手順

1. **PATHを永続的に追加**（一度だけ実行）
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **MLZeroを起動**
   ```bash
   # ターミナル1（バックエンド）
   mlzero-backend
   
   # ターミナル2（フロントエンド）
   mlzero-frontend
   ```

または、スクリプトを使用：

```bash
wsl bash scripts/start_backend.sh
# 別ターミナルで
wsl bash scripts/start_frontend.sh
```



