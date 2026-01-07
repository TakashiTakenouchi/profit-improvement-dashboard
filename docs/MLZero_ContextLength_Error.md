# MLZero コンテキスト長エラー解決方法

## エラー内容

```
BadRequestError: This model's maximum context length is 8192 tokens. 
However, you requested 16690 tokens (306 in the messages, 16384 in the completion).
```

## 問題の原因

### 1. モデルとコンテキスト長の不一致

- **使用モデル**: `gpt-4`
- **モデルの最大コンテキスト長**: 8192トークン
- **設定されているmax_tokens**: 16384トークン
- **実際の要求**: 入力306トークン + 出力16384トークン = 16690トークン

### 2. 設定ファイルの問題

すべてのエージェント（`llm`, `coder`, `reader`, `executer`など）で`max_tokens: 16384`が設定されていますが、`gpt-4`モデルではこれは不可能です。

## 解決方法

### 方法1: モデルを変更（推奨）

`gpt-4`を`gpt-4-turbo`または`gpt-4o`に変更します。これらのモデルはより大きなコンテキスト長をサポートしています。

**推奨モデル:**
- `gpt-4-turbo` - 最大128Kトークン
- `gpt-4o` - 最大128Kトークン（最新・推奨）

**修正コマンド:**
```bash
# gpt-4をgpt-4-turboに変更
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: gpt-4$/model: gpt-4-turbo/g' {} \;

# またはgpt-4oに変更（推奨）
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: gpt-4$/model: gpt-4o/g' {} \;
```

### 方法2: max_tokensを減らす

`gpt-4`を使い続ける場合は、`max_tokens`を8192以下に減らす必要があります。

**注意**: 入力メッセージも考慮する必要があるため、実際には`max_tokens`は約7000以下に設定する必要があります。

**修正コマンド:**
```bash
# max_tokensを7000に変更
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/max_tokens: 16384/max_tokens: 7000/g' {} \;
```

## モデル別のコンテキスト長

| モデル | 最大コンテキスト長 | 推奨max_tokens |
|--------|-------------------|---------------|
| `gpt-4` | 8,192トークン | 7,000 |
| `gpt-4-turbo` | 128,000トークン | 16,384 |
| `gpt-4o` | 128,000トークン | 16,384 |
| `gpt-3.5-turbo` | 16,385トークン | 4,096 |

## 修正後の確認

設定ファイルを修正した後、以下で確認してください：

```bash
# モデル名を確認
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec grep "model:" {} \; | head -10

# max_tokensを確認
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec grep "max_tokens:" {} \; | head -10
```

## 次のステップ

1. 設定ファイルを修正
2. MLZeroを再起動
3. タスクを再実行

## 注意事項

- `gpt-4-turbo`や`gpt-4o`は`gpt-4`よりコストが高い場合があります
- ただし、より大きなコンテキスト長により、より複雑なタスクを処理できます
- `gpt-4o`は最新のモデルで、性能とコストのバランスが良いです



