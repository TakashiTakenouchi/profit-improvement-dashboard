# MLZero エラー修正ガイド

## エラー: Invalid model: ChatGPT5

### 問題の原因

設定ファイル（`autogluon_config.yaml`）で`model: ChatGPT5`が指定されていますが、これは無効なモデル名です。

### 解決方法

#### 方法1: 設定ファイルを直接編集

設定ファイルの場所:
```
/home/takenouchiy/.autogluon_assistant/[UUID]/upload_[ID]/autogluon_config.yaml
```

設定ファイル内のすべての`model: ChatGPT5`を有効なモデル名に変更してください。

**推奨モデル名:**
- `gpt-4o` - 最新のGPT-4oモデル（推奨）
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-4` - GPT-4
- `gpt-3.5-turbo` - GPT-3.5 Turbo（コスト効率が良い）

#### 方法2: Web UIで設定を変更

1. Web UIの「モデル構成」セクションでモデルを選択
2. `ChatGPT5`から有効なモデル名（例：`gpt-4o`）に変更
3. 設定を保存

### 修正例

**修正前:**
```yaml
llm:
  provider: openai
  model: ChatGPT5  # ❌ 無効
  max_tokens: 16384
  temperature: 1.0
```

**修正後:**
```yaml
llm:
  provider: openai
  model: gpt-4o  # ✅ 有効
  max_tokens: 16384
  temperature: 1.0
```

### 有効なモデル名一覧

OpenAIプロバイダーで使用可能なモデル:
- `gpt-4o` - 最新のGPT-4o（推奨）
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-4` - GPT-4
- `gpt-3.5-turbo` - GPT-3.5 Turbo
- `gpt-4o-mini` - GPT-4o Mini（コスト効率が良い）
- `o1` - O1モデル
- `o1-pro` - O1 Proモデル

その他のモデル名はエラーログに表示されています。

### 設定ファイルの場所を確認

```bash
# WSL内で実行
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f
```

### 一括置換スクリプト

複数の設定ファイルがある場合、一括で置換できます：

```bash
# WSL内で実行
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: ChatGPT5/model: gpt-4o/g' {} \;
```

### 注意事項

- 設定ファイルを変更した後、MLZeroを再起動してください
- モデル名は大文字小文字を区別します
- `gpt-4o`は最新のモデルで、性能とコストのバランスが良いです



