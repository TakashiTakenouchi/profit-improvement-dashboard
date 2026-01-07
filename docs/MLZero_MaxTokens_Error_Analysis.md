# MLZero max_tokens エラー分析

## エラー内容

```
BadRequestError: Error code: 400 - {'error': {'message': 'max_tokens is too large: 16384. This model supports at most 4096 completion tokens, whereas you provided 16384.', 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': 'invalid_value'}}
```

## 問題の原因

### ❌ Excelファイルのサイズは関係ありません

**重要なポイント:**
- エラーは**Excelファイルのサイズ**が原因ではありません
- エラーは**設定ファイルの`max_tokens`値**が原因です

### 実際の問題

1. **モデル**: `gpt-4-turbo` を使用
2. **設定**: `max_tokens: 16384` に設定されている
3. **制限**: `gpt-4-turbo`は最大**4096 completion tokens**までしかサポートしていない

### エラーの発生タイミング

エラーは、MLZeroがファイルを読み込む**初期段階**（`DataPerceptionAgent`）で発生しています。この時点では、まだExcelファイルの内容を読み込んでいません。

```
DataPerceptionAgent: beginning to scan data folder and group similar files.
...
Reading file: /home/takenouchiy/.autogluon_assistant/.../autogluon_config.yaml
...
BadRequestError: max_tokens is too large: 16384
```

## 解決方法

### 方法1: 自動修正スクリプトを使用（推奨）

```bash
wsl bash scripts/fix_max_tokens_gpt4_turbo.sh
```

このスクリプトは、すべての設定ファイルで`max_tokens: 16384`を`max_tokens: 4096`に変更します。

### 方法2: 手動で修正

```bash
# すべての設定ファイルでmax_tokensを修正
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/max_tokens: 16384/max_tokens: 4096/g' {} \;
```

### 方法3: より大きなコンテキスト長が必要な場合

`gpt-4-turbo`の代わりに、より大きなコンテキスト長をサポートするモデルを使用：

```bash
# gpt-4oに変更（最大128,000トークン）
find ~/.autogluon_assistant -name "autogluon_config.yaml" -type f -exec sed -i 's/model: gpt-4-turbo/model: gpt-4o/g' {} \;
```

その後、`max_tokens: 16384`の設定が有効になります。

## モデル別の制限

| モデル | 最大コンテキスト長 | 最大completion tokens | 推奨max_tokens |
|--------|-------------------|----------------------|---------------|
| `gpt-4-turbo` | 128,000 | **4,096** | 4,000 |
| `gpt-4o` | 128,000 | 16,384 | 16,000 |
| `gpt-4` | 8,192 | 8,192 | 7,000 |

## よくある質問

### Q: Excelファイルを小さくすれば解決しますか？

**A: いいえ、解決しません。** エラーは設定ファイルの`max_tokens`値が原因で、Excelファイルのサイズとは無関係です。

### Q: 小規模なExcelサンプルならOKになりますか？

**A: いいえ、なりません。** エラーはファイルを読み込む前に発生しているため、Excelファイルのサイズは関係ありません。

### Q: なぜ`gpt-4-turbo`で`max_tokens: 16384`が設定されているのですか？

**A:** これは設定ファイルのデフォルト値または以前の設定の残りです。`gpt-4-turbo`は最大4096 completion tokensまでしかサポートしていないため、この値を超える設定はエラーになります。

## 修正後の確認

修正後、MLZeroを再起動して、再度タスクを実行してください：

```bash
# MLZeroを再起動
wsl bash scripts/restart_mlzero.sh

# または手動で再起動
# ターミナル1: mlzero-backend
# ターミナル2: mlzero-frontend
```

## まとめ

- ❌ **Excelファイルのサイズは関係ない**
- ✅ **設定ファイルの`max_tokens`値を修正する必要がある**
- ✅ **`gpt-4-turbo`の場合は`max_tokens: 4096`以下に設定**
- ✅ **より大きな値が必要な場合は`gpt-4o`に変更**



