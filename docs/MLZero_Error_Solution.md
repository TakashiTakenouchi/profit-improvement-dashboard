# MLZero エラー解決方法

## ✅ 問題解決済み

### エラー内容
```
ValueError: Invalid model: ChatGPT5 for provider openai
```

### 原因
設定ファイル（`autogluon_config.yaml`）で無効なモデル名`ChatGPT5`が使用されていました。

### 解決策
すべての設定ファイルで`ChatGPT5`を`gpt-4o`に一括置換しました。

### 修正コマンド（実行済み）
```bash
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: ChatGPT5/model: gpt-4o/g' {} \;
```

## 次のステップ

1. **MLZeroを再起動**
   - バックエンドとフロントエンドを再起動してください

2. **Web UIでタスクを再実行**
   - 同じタスクを再度実行してください
   - 今度は正常に動作するはずです

3. **モデル名の確認**
   - Web UIの「モデル構成」セクションで`gpt-4o`が選択されていることを確認

## 使用可能なモデル名

- `gpt-4o` - 最新のGPT-4o（推奨・現在設定済み）
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-4` - GPT-4
- `gpt-3.5-turbo` - GPT-3.5 Turbo（コスト効率が良い）
- `gpt-4o-mini` - GPT-4o Mini（コスト効率が良い）

## 今後の注意事項

- Web UIでモデルを選択する際は、有効なモデル名を選択してください
- `ChatGPT5`のような無効なモデル名は使用しないでください
- 設定ファイルを手動で編集する場合は、モデル名のスペルを確認してください



