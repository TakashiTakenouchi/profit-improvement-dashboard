# MLZero Windowsコマンドエラー解決方法

## エラー内容

Windowsのコマンドプロンプトから直接Linuxコマンドを実行しようとして、以下のようなエラーが発生：

```
ファイルが見つかりません
```

## 原因

Windowsのコマンドプロンプトから直接`find`などのLinuxコマンドを実行すると、引用符や特殊文字が正しく解釈されません。

### 間違った実行方法

```cmd
find ~/.autogluon_assistant -name "autogluon_config.yaml" ...
```

これはWindowsの`find`コマンドとして解釈され、Linuxの`find`コマンドとして実行されません。

## 解決方法

### ✅ 正しい実行方法

**方法1: WSL環境で実行（推奨）**

```bash
wsl bash -c "find ~/.autogluon_assistant -name 'autogluon_config.yaml' -type f -exec grep -H 'model:' {} \;"
```

**方法2: WSLターミナルで直接実行**

```bash
wsl
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec grep -H "model:" {} \;
```

**方法3: スクリプトを使用（最も簡単）**

```bash
wsl bash scripts/fix_model_name.sh
```

## よくある間違い

### ❌ 間違い1: Windowsコマンドプロンプトから直接実行

```cmd
find ~/.autogluon_assistant -name "autogluon_config.yaml" ...
```

### ❌ 間違い2: 無効なモデル名

```bash
# gpt-40は無効なモデル名です
model: gpt-40  # ❌ 間違い
```

### ✅ 正しいモデル名

```bash
model: gpt-4o        # ✅ 推奨
model: gpt-4-turbo   # ✅ 有効
model: gpt-4         # ✅ 有効（ただしコンテキスト長が小さい）
```

## 推奨手順

1. **WSL環境でコマンドを実行**
   ```bash
   wsl bash -c "コマンド"
   ```

2. **または、WSLターミナルを開く**
   ```bash
   wsl
   # その後、Linuxコマンドを実行
   ```

3. **スクリプトを使用**
   ```bash
   wsl bash scripts/fix_model_name.sh
   ```

## 注意事項

- Windowsのコマンドプロンプトから直接Linuxコマンドを実行しないでください
- `wsl bash -c "..."`を使用するか、WSLターミナルで実行してください
- モデル名は正確に入力してください（`gpt-40`ではなく`gpt-4o`）



