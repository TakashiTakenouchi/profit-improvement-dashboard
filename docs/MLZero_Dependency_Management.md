# MLZero 依存関係管理ガイド

## MLZeroの動作方法

### 自動コード生成

MLZeroは、**ユーザーが提供したデータとタスクの説明から、自動的にコードを生成**します。

- ✅ **事前にコードを提供する必要はありません**
- ✅ MLZeroが自動的に必要なコードを生成します
- ❌ ただし、生成されたコードが依存するライブラリがインストールされている必要があります

## 問題の原因

MLZeroが生成したコードが`autogluon.timeseries`をインポートしようとしていますが、このモジュールがインストールされていないため、エラーが発生しています。

## 解決方法

### 方法1: 依存関係を事前にインストール（推奨）

MLZeroを実行する前に、必要な依存関係をインストールします：

```bash
wsl
pip3 install autogluon.timeseries
```

または、AutoGluonの完全なパッケージをインストール：

```bash
wsl
pip3 install autogluon[timeseries]
```

### 方法2: MLZeroに依存関係のインストールを指示

タスクの説明に、必要な依存関係のインストールを明示的に含めます：

```
添付ExcelのDailyForecastDataシートを見て90日間の予測を作成してください。
必要なライブラリ（autogluon.timeseries）がインストールされていない場合は、インストールしてください。
```

### 方法3: 生成されたコードを修正して実行

MLZeroが生成したコードを確認し、必要な依存関係をインストールしてから手動で実行：

```bash
# 1. 生成されたコードを確認
wsl bash -c "cat /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/generation_iter_5/generated_code.py"

# 2. 依存関係をインストール
pip3 install autogluon.timeseries

# 3. コードを実行
python3 /home/takenouchiy/.local/lib/python3.10/runs/mlzero-[日時]-[UUID]/generation_iter_5/generated_code.py
```

## MLZeroのタスク説明の改善

### 良いタスク説明の例

```
添付ExcelのDailyForecastDataシートを見て90日間の予測を作成してください。

要件:
- AutoGluon TimeSeriesPredictorを使用
- 必要なライブラリがインストールされていない場合は、インストールしてください
- 結果をCSVファイルとして保存してください
```

### 依存関係を明示的に指定

```
タスク: 時系列予測
データ: 添付ExcelファイルのDailyForecastDataシート
予測期間: 90日間
使用ライブラリ: autogluon.timeseries
出力形式: CSV

注意: autogluon.timeseriesがインストールされていない場合は、pip install autogluon.timeseriesを実行してください。
```

## よくある依存関係

### AutoGluon関連

```bash
pip3 install autogluon.timeseries      # 時系列予測
pip3 install autogluon.tabular         # 表形式データ
pip3 install autogluon.multimodal      # マルチモーダル
pip3 install autogluon[timeseries]     # 時系列を含む完全版
```

### その他の一般的な依存関係

```bash
pip3 install pandas numpy matplotlib seaborn
pip3 install scikit-learn
pip3 install openpyxl  # Excelファイル読み込み
```

## 推奨ワークフロー

### ステップ1: 必要な依存関係を事前にインストール

```bash
wsl
pip3 install autogluon.timeseries pandas openpyxl
```

### ステップ2: MLZeroでタスクを実行

タスクの説明を提供し、MLZeroにコード生成を依頼します。

### ステップ3: 生成されたコードを確認

```bash
wsl bash scripts/analyze_mlzero_output.sh
```

### ステップ4: 必要に応じて修正

生成されたコードを確認し、必要に応じて修正して実行します。

## まとめ

- ❌ **事前にコードを提供する必要はありません**
- ✅ **必要な依存関係を事前にインストールするか、MLZeroにインストールを指示してください**
- ✅ **生成されたコードを確認し、必要に応じて修正して実行できます**

## 次のステップ

1. `autogluon.timeseries`をインストール
2. MLZeroでタスクを再実行
3. または、生成されたコードを手動で実行



