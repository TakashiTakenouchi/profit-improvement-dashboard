# AutoGluon TimeSeries 時系列予測ナレッジドキュメント

## 1. プロジェクト概要

### 1.1 目的
恵比寿店・横浜元町店の商品カテゴリー別売上を、AutoGluon TimeSeriesを用いて2026年1月1日から90日間予測する。

### 1.2 処理フロー
```
[入力データ] → [データ整合性修正] → [AutoGluon学習] → [予測実行] → [結果出力]
```

---

## 2. フォルダ構造とPythonコード所在

### 2.1 フォルダ構造
```
Ensuring consistency between tabular data and time series forecast data/
├── input/                              # 入力データ
│   ├── fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx
│   └── time_series_forecast_data_2024.xlsx
├── Process/                            # 処理スクリプト
│   ├── fix_timeseries_data.py          # データ整合性修正
│   ├── autogluon_timeseries_forecast.py # 予測（初版）
│   └── timeseries_forecast_run_v2.py    # 予測（日本語パス対応版）★推奨
├── output/                             # 出力データ
│   ├── time_series_forecast_data_2024_fixed.xlsx
│   └── forecast_results_2026_90days.xlsx
├── docs/                               # ドキュメント
│   ├── 時系列予測データ修正要件.md
│   └── AutoGluon_TimeSeries_Knowledge.md
└── autogluon_ts_models/                # 学習済みモデル（AutoGluon生成）
```

### 2.2 Pythonコード所在一覧

| ファイル名 | パス | 用途 |
|-----------|-----|-----|
| **fix_timeseries_data.py** | `Process/fix_timeseries_data.py` | データ整合性修正 |
| **timeseries_forecast_run_v2.py** | `Process/timeseries_forecast_run_v2.py` | 時系列予測（推奨版） |
| autogluon_timeseries_forecast.py | `Process/autogluon_timeseries_forecast.py` | 時系列予測（初版） |

### 2.3 学習済みモデル
```
C:\PyCharm\AutoGluon AssistantTEST\autogluon_ts_models\
```
※ AutoGluonが自動生成。`TimeSeriesPredictor.load()`で再利用可能。

---

## 3. データ整合性修正

### 3.1 問題点
時系列データ（time_series_forecast_data_2024.xlsx）の日付が**1年ズレていた**。

| 項目 | 修正前 | 修正後 |
|-----|-------|-------|
| 日付範囲 | 2019/04/01 ～ 2024/12/31 | 2020/04/30 ～ 2025/12/31 |
| 売上整合性 | 不一致（最大97%差異） | 完全一致（0%差異） |

### 3.2 修正手順
1. **日付の+1年シフト**: 全シートの日付を+1年
2. **OriginalMonthlyData上書き**: 正データで上書き
3. **ForecastQuantity調整**: 月次売上合計が正データと一致するよう調整
4. **集計シート再計算**: MonthlySummary, ValidationResults

### 3.3 修正結果
```
整合性チェック: 1,104件中1,104件 PASS（差異率0%）
```

---

## 4. AutoGluon TimeSeries予測

### 4.1 入力データ形式
```python
# 必須列
| item_id | timestamp | target | weekend |
|---------|-----------|--------|---------|
| EBISU_Mens_JACKETS&OUTER2_001 | 2020-04-30 | 103.72 | 0 |
```

- **item_id**: 店舗コード + ItemCode（例: EBISU_Mens_KNIT_001）
- **timestamp**: 日付
- **target**: 予測対象（ForecastQuantity）
- **weekend**: 共変量（0:平日、1:土日）

### 4.2 Static Features
```python
| item_id | Shop | Category |
|---------|------|----------|
| EBISU_Mens_JACKETS&OUTER2_001 | EBISU | Mens_JACKETS&OUTER2 |
```

### 4.3 予測設定
```python
predictor = TimeSeriesPredictor(
    prediction_length=90,        # 90日間予測
    eval_metric="WQL",           # Weighted Quantile Loss
    known_covariates_names=["weekend"],  # 時間変動共変量
    freq="D",                    # 日次頻度
)

predictor.fit(
    ts_df,
    presets="medium_quality",    # 品質プリセット
    time_limit=1800,             # 30分制限
)
```

### 4.4 学習モデル結果

| モデル | Validation Score (-WQL) | 学習時間 | 備考 |
|--------|------------------------|---------|------|
| **WeightedEnsemble** | **-0.3298** | 1.05s | **最良モデル** |
| Chronos2 | -0.3380 | 18.48s | 基盤モデル |
| TemporalFusionTransformer | -0.3461 | 642.59s | 深層学習 |
| DirectTabular | -0.3774 | 20.38s | LightGBM |
| Theta | -0.4945 | 0.04s | 統計 |
| ETS | -0.6297 | 0.05s | 指数平滑法 |
| SeasonalNaive | -0.7695 | 0.04s | ベースライン |
| RecursiveTabular | -1.4450 | 1.93s | LightGBM |

### 4.5 アンサンブル重み
```
Chronos2: 57%
TemporalFusionTransformer: 39%
DirectTabular: 4%
```

---

## 5. 予測結果サマリー

### 5.1 店舗別予測数量（90日間合計）
| 店舗 | 予測数量 |
|-----|---------|
| 恵比寿 | 7,225 個 |
| 横浜元町 | 16,979 個 |
| **合計** | **24,204 個** |

### 5.2 カテゴリー別予測数量（全店舗90日間合計）
| カテゴリー | 予測数量 |
|-----------|---------|
| レディース ジャケット | 5,835 個 |
| メンズ ジャケット・アウター | 5,494 個 |
| レディース ボトムス | 3,944 個 |
| レディース ワンピース | 2,648 個 |
| レディース トップス | 2,086 個 |
| メンズ パンツ | 1,876 個 |
| メンズ ニット | 1,593 個 |
| レディース スカーフ・ストール | 729 個 |

---

## 6. 実装上の注意点

### 6.1 日本語パス問題
Windowsでユーザー名が日本語の場合、joblibが一時ファイル作成でエラーを起こす。

**解決策**:
```python
# 環境変数で一時フォルダを英語パスに設定
temp_dir = r'C:\PyCharm\AutoGluon AssistantTEST\temp'
os.environ['TMP'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
```

### 6.2 店舗名のエンコーディング
item_idに日本語を含めるとエラーになる場合がある。

**解決策**:
```python
shop_code_map = {'恵比寿': 'EBISU', '横浜元町': 'YOKOHAMA'}
df_daily['ShopCode'] = df_daily['Shop'].map(shop_code_map)
df_daily['item_id'] = df_daily['ShopCode'] + '_' + df_daily['ItemCode']
```

### 6.3 頻度の自動推定
データに欠損日がある場合、頻度が'IRREG'と推定される。

**解決策**:
```python
predictor = TimeSeriesPredictor(freq="D")  # 明示的に日次を指定
```

---

## 7. 出力ファイル形式

### 7.1 forecast_results_2026_90days.xlsx

**DailyForecastsシート**:
| 列名 | 説明 |
|-----|-----|
| item_id | アイテムID |
| timestamp | 予測日付 |
| predicted_quantity | 予測数量（mean） |
| 0.1 | 10パーセンタイル |
| 0.5 | 50パーセンタイル（中央値） |
| 0.9 | 90パーセンタイル |
| Shop | 店舗名 |
| ItemCode | 商品コード |
| CategoryCode | カテゴリーコード |
| Category | カテゴリー名 |

**MonthlySummaryシート**:
| 列名 | 説明 |
|-----|-----|
| Shop | 店舗名 |
| YearMonth | 年月 |
| Category | カテゴリー名 |
| Predicted_Qty | 予測数量合計 |
| Lower_10pct | 10%下限合計 |
| Upper_90pct | 90%上限合計 |

**CategorySummaryシート**:
店舗・カテゴリー別の90日間予測数量合計

---

## 8. Q&A ナレッジ

### Q1: Excelで信頼区間付きグラフを作成する方法

**手順:**

1. **データ準備**: `forecast_results_2026_90days.xlsx`の`DailyForecasts`シートで新しい列を追加
   ```
   G列: Upper_Error = 0.9列 - predicted_quantity列
   H列: Lower_Error = predicted_quantity列 - 0.1列
   ```

2. **グラフ作成**:
   - データ範囲を選択（timestamp, predicted_quantity）
   - **挿入** → **グラフ** → **折れ線グラフ**

3. **誤差範囲追加**:
   - predicted_quantityの系列を選択
   - **グラフデザイン** → **グラフ要素を追加** → **誤差範囲** → **その他の誤差範囲オプション**
   - **ユーザー設定**を選択:
     - 正の誤差値: G列（Upper_Error）を指定
     - 負の誤差値: H列（Lower_Error）を指定

4. **ピボットグラフの場合**:
   - ピボットテーブルで集計後、グラフを作成
   - 誤差範囲は手動で追加

---

### Q2: Chronos2が57%で選ばれた理由とTheoreticalBasisの関係

#### 入力データの分布型別統計特性

| 分布型 | 件数 | 平均 | CV(変動係数) | 歪度 | 尖度 | ゼロ率 |
|-------|-----|-----|-------------|-----|-----|-------|
| **ポアソン分布** | 12,438 | 1.64 | **2.72** | **19.0** | 507 | **37.6%** |
| 正規分布 | 12,438 | 8.19 | 1.63 | 13.1 | 434 | 13.0% |
| 負の二項分布 | 24,876 | 4.19 | 1.90 | 10.7 | 243 | 20.7% |

#### 予測難易度スコア

| 分布型 | 難易度スコア | 予測の困難さ |
|-------|------------|------------|
| **ポアソン分布** | **2.14** | 最も困難 |
| 負の二項分布 | 1.31 | 中程度 |
| 正規分布 | 1.27 | 中程度 |

#### Chronos2が57%で選ばれた理由

**1. 非正規分布への対応力**
- 全ての分布型が高い歪度（10〜19）と尖度（200〜500）
- 正規分布を仮定するETS/ARIMAは不利（WQL: -0.63）
- Chronos2は**分布を仮定しない**基盤モデル

**2. ゼロ値・間欠的需要への対応**
- ポアソン分布データの**37.6%がゼロ値**
- 従来モデルはゼロが多いデータで精度低下
- Chronos2は**大規模事前学習**で希少パターンにも対応

**3. 高変動データへの対応**
- ポアソン分布のCV（変動係数）= **2.72**（平均の2.7倍の標準偏差）
- RecursiveTabularは高変動に弱い（WQL: -1.44、最悪）
- Chronos2は確率的予測で不確実性を適切にモデル化

#### 分布型と各モデルの適合性マトリックス

| モデル | ポアソン分布 | 正規分布 | 負の二項分布 | WQL Score | 採用率 |
|-------|------------|---------|------------|-----------|-------|
| **Chronos2** | ◎ | ◎ | ◎ | -0.338 | **57%** |
| TFT | ○ | ◎ | ○ | -0.346 | 39% |
| DirectTabular | △ | ○ | ○ | -0.377 | 4% |
| Theta | △ | ○ | △ | -0.494 | 0% |
| ETS | × | ○ | △ | -0.630 | 0% |
| SeasonalNaive | △ | △ | △ | -0.770 | 0% |
| RecursiveTabular | × | △ | × | -1.445 | 0% |

凡例: ◎=非常に適合, ○=適合, △=やや不適合, ×=不適合

#### TheoreticalBasisの意味

| 価格帯 | カテゴリー分類 | 理論分布 | マージン | 需要特性 |
|-------|-------------|---------|---------|---------|
| 高価格帯 | 高級カテゴリ | 正規分布 | 高マージン | 安定商品 |
| 高価格帯 | 季節カテゴリ | ポアソン分布 | 高マージン | 安定商品 |
| 高価格帯 | ベーシックカテゴリ | 負の二項分布 | 高マージン | 安定商品 |
| 中価格帯 | 高級カテゴリ | 正規分布 | 中マージン | 季節商品 |
| 中価格帯 | 季節カテゴリ | ポアソン分布 | 中マージン | 季節商品 |
| 中価格帯 | ベーシックカテゴリ | 負の二項分布 | 中マージン | 季節商品 |
| 低価格帯 | 高級カテゴリ | 正規分布 | 中マージン | 季節商品 |
| 低価格帯 | 季節カテゴリ | ポアソン分布 | 中マージン | 季節商品 |
| 低価格帯 | ベーシックカテゴリ | 負の二項分布 | 中マージン | 季節商品 |

#### 結論

**Chronos2が最も選ばれた理由:**

1. **ポアソン分布データ（37.6%ゼロ値、CV=2.72）に強い**
   - 間欠的需要パターンを事前学習で獲得済み

2. **非正規分布（歪度10〜19）に対応**
   - 分布仮定なしで任意の時系列に適用可能

3. **3種類の異なる分布特性を統合的に学習**
   - ポアソン、正規、負の二項分布の混在データに対応

---

## 9. 今後の改善点

1. **GPU利用**: CPUのみの場合、Chronos2やTFTの学習に時間がかかる
2. **追加共変量**: 祝日、プロモーション、気温などの追加
3. **予測期間延長**: 90日以上の予測
4. **自動再学習**: 定期的なモデル更新パイプラインの構築

---

## 10. 参考資料

- [AutoGluon TimeSeries Documentation](https://auto.gluon.ai/stable/tutorials/timeseries/index.html)
- [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- AutoGluon Version: 1.5.0
- Python Version: 3.11.4

---

**作成日**: 2024-12-28
**更新日**: 2024-12-28
**作成者**: Claude Code
