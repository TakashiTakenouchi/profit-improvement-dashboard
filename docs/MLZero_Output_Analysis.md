# MLZero 出力分析ガイド

## 状況

MLZeroタスクが失敗したが、出力ディレクトリにファイルが生成されている場合の対処方法。

## 出力ディレクトリの確認

### 出力ディレクトリの場所

```
/home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/
```

### ディレクトリ構造の確認

```bash
wsl bash -c "ls -la /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/"
```

### 生成されたファイルの検索

```bash
# Pythonファイル
find /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/ -name "*.py"

# ログファイル
find /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/ -name "*.log"

# テキストファイル
find /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/ -name "*.txt"

# JSONファイル
find /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/ -name "*.json"
```

## よくある出力ディレクトリ構造

```
mlzero-[日時]-[UUID]/
├── initialization/
│   ├── states/
│   └── prompts/
├── iteration_1/
│   ├── code/
│   ├── execution/
│   └── states/
├── iteration_2/
│   └── ...
└── final/
    └── ...
```

## 出力ファイルの確認方法

### 1. 生成されたコードの確認

```bash
# すべてのPythonファイルを検索
find /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/ -name "*.py" -type f
```

### 2. 実行ログの確認

```bash
# ログファイルを検索
find /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/ -name "*.log" -type f
```

### 3. エラーメッセージの確認

```bash
# エラーログを検索
find /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/ -name "*error*" -o -name "*stderr*"
```

## トークン使用量の分析

### トークン使用量の見方

- **Total tokens**: 合計トークン数
- **Input tokens**: 入力トークン数（プロンプトなど）
- **Output tokens**: 出力トークン数（LLMの応答）

### 主要なセッション

- `multi_turn_python_coder`: Pythonコード生成（最も多くのトークンを使用）
- `multi_turn_bash_coder`: Bashスクリプト生成
- `single_turn_error_analyzer`: エラー分析
- `single_turn_python_executer`: Pythonコード実行

## タスクが失敗した場合の対処

### 1. エラーログを確認

```bash
# 最新のエラーログを確認
find /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/ -name "*error*" -o -name "*stderr*" | xargs tail -50
```

### 2. 生成されたコードを確認

```bash
# 最新のコードファイルを確認
find /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/ -name "*.py" -type f | sort | tail -5 | xargs cat
```

### 3. 実行結果を確認

```bash
# 実行結果を確認
find /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/ -name "*stdout*" -o -name "*result*" | xargs tail -50
```

## 出力ファイルのダウンロード

### Windowsからアクセス

WSLのファイルは以下のパスからアクセス可能：

```
\\wsl$\Ubuntu\home\takenouchiy\.local\lib\python3.10\runs\mlzero-[日時]-[UUID]\
```

または：

```
C:\Users\竹之内隆\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu*\LocalState\rootfs\home\takenouchiy\.local\lib\python3.10\runs\mlzero-[日時]-[UUID]\
```

## スクリプト: 出力ディレクトリの分析

```bash
#!/bin/bash
# MLZero出力ディレクトリ分析スクリプト

OUTPUT_DIR="/home/takenouchiy/.local/lib/python3.10/runs/mlzero-20251230_082831-7f693fcc-0116-4dbe-9948-b33a0e73b6da"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "出力ディレクトリが見つかりません: $OUTPUT_DIR"
    exit 1
fi

echo "=========================================="
echo "MLZero 出力ディレクトリ分析"
echo "=========================================="
echo ""
echo "ディレクトリ: $OUTPUT_DIR"
echo ""

# ディレクトリ構造
echo "1. ディレクトリ構造:"
find "$OUTPUT_DIR" -type d | head -20
echo ""

# ファイル一覧
echo "2. ファイル一覧:"
find "$OUTPUT_DIR" -type f | head -20
echo ""

# Pythonファイル
echo "3. Pythonファイル:"
find "$OUTPUT_DIR" -name "*.py" -type f
echo ""

# ログファイル
echo "4. ログファイル:"
find "$OUTPUT_DIR" -name "*.log" -type f
echo ""

# エラーファイル
echo "5. エラーファイル:"
find "$OUTPUT_DIR" -name "*error*" -o -name "*stderr*" | head -5
echo ""

# サイズ
echo "6. ディレクトリサイズ:"
du -sh "$OUTPUT_DIR"
```



