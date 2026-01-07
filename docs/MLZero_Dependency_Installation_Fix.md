# MLZero 依存関係インストール問題の解決

## 問題

MLZeroが生成したBashスクリプトに依存関係のインストールが含まれておらず、コード実行時に`ModuleNotFoundError`が発生します。

## 原因

MLZeroが生成したBashスクリプトは、単にPythonコードを実行するだけで、必要な依存関係のインストールを行いません。

### 現在のBashスクリプト

```bash
#!/bin/bash
PYTHON_SCRIPT="..."
python3 "$PYTHON_SCRIPT"
```

## 解決方法

### 方法1: Bashスクリプトを修正（推奨）

修正スクリプトを使用：

```bash
wsl bash scripts/fix_mlzero_bash_script.sh
```

または、手動で修正：

```bash
# Bashスクリプトを編集
wsl bash -c "cat > /path/to/extracted_bash_script.sh << 'EOF'
#!/bin/bash

# Install required dependencies
pip install autogluon.timeseries pandas openpyxl

# Execute Python script
python3 /path/to/generated_code.py
EOF"
```

### 方法2: 生成されたPythonコードを直接実行

システム環境で依存関係がインストールされている場合：

```bash
wsl bash -c "python3 /home/takenouchiy/.local/lib/python3.10/runs/mlzero-20251230_090730-04671264-874b-4d85-965b-94324438c445/generation_iter_5/generated_code.py"
```

### 方法3: タスク説明を改善

次回のタスク説明に、依存関係のインストールを明示的に指示：

```
添付ExcelのDailyForecastDataシートを見て90日間の予測を作成してください。

重要: 
- Bashスクリプトの最初に「pip install autogluon.timeseries pandas openpyxl」を追加してください
- 依存関係をインストールしてからPythonコードを実行してください
```

## 修正後のBashスクリプト例

```bash
#!/bin/bash

# Install required dependencies
pip install autogluon.timeseries pandas openpyxl

# Define the path to the Python script
PYTHON_SCRIPT="/path/to/generated_code.py"

# Execute the Python script
python3 "$PYTHON_SCRIPT"
```

## よくある依存関係

### AutoGluon関連

```bash
pip install autogluon.timeseries  # 時系列予測
pip install autogluon.tabular     # 表形式データ
```

### データ処理関連

```bash
pip install pandas openpyxl       # Excelファイル読み込み
pip install numpy matplotlib      # データ分析・可視化
```

## 推奨ワークフロー

1. **MLZeroでタスクを実行**
2. **生成されたBashスクリプトを確認**
3. **依存関係のインストールが含まれているか確認**
4. **含まれていない場合、修正スクリプトを実行**
5. **修正したスクリプトを実行**

## まとめ

- MLZeroが生成したBashスクリプトに依存関係のインストールが含まれていない場合がある
- スクリプトを修正して依存関係のインストールを追加する
- または、生成されたPythonコードを直接実行する
- 次回のタスク説明で依存関係のインストールを明示的に指示する



