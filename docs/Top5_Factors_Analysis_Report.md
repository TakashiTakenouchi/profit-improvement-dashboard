# 黒字要因TOP5 統計分析・時系列予測モデル選定レポート

---

## 目次

1. [分析概要](#1-分析概要)
2. [TASK1: データ生成Pythonコードの特定](#2-task1-データ生成pythonコードの特定)
3. [TASK2: TheoreticalBasisのShop/Category単位再分類](#3-task2-theoreticalbasisのshopcategory単位再分類)
4. [TASK3: 黒字要因TOP5の詳細分析](#4-task3-黒字要因top5の詳細分析)
5. [確率分布の理論的解説](#5-確率分布の理論的解説)
6. [AutoGluon-TimeSeriesでのモデル検証](#6-autogluon-timeseriesでのモデル検証)
7. [推奨アクションサマリー](#7-推奨アクションサマリー)
8. [GitHub/Render更新手順](#8-githubrender更新手順)
9. [最終仕様サマリー](#9-最終仕様サマリー)

---

## 1. 分析概要

### 1.1 目的
ロジスティック回帰分析で特定された黒字要因TOP5について、確率分布特性を分析し、数理最適化および時系列予測モデル選定の考慮事項を明文化する。

### 1.2 対象データ
- **損益データ**: `fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx`
- **時系列データ**: `time_series_forecast_data_2024_fixed.xlsx`

### 1.3 黒字要因TOP5（オッズ比順）
| 順位 | フィールド | 列番号 | オッズ比 |
|------|-----------|--------|---------|
| 1位 | WOMEN'S_JACKETS2 | O列 | 最高 |
| 2位 | Number_of_guests | AA列 | 高 |
| 3位 | WOMEN'S_ONEPIECE | T列 | 高 |
| 4位 | Mens_KNIT | P列 | 中 |
| 5位 | Mens_PANTS | Q列 | 中 |

---

## 2. TASK1: データ生成Pythonコードの特定

### 2.1 fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx

| ファイル | パス | 役割 |
|---------|------|------|
| `04_complete_improvement_vol5_20251227.py` | src/analysis/ | vol5.xlsx生成（ロジスティック回帰＋黒字化改善） |
| `05_ebisu_optimization_vol6_20251227.py` | src/analysis/ | vol6.xlsx生成（PuLP最適化＋±30%バラつき適用） |

**処理フロー:**
```
vol3.xlsx → vol5.xlsx（黒字化改善） → vol6.xlsx（最適化） → vol6_new.xlsx（最終版）
```

### 2.2 time_series_forecast_data_2024_fixed.xlsx

| ファイル | パス | 役割 |
|---------|------|------|
| `fix_timeseries_data.py` | Process/ | 日付+1年シフト、ForecastQuantity調整、整合性修正 |

---

## 3. TASK2: TheoreticalBasisのShop/Category単位再分類

### 3.1 現状の課題
TheoreticalBasisは現在**ItemCode単位**（高価格帯/中価格帯/低価格帯）で分類されているが、黒字要因TOP5はCategory単位の集計値であるため、そのままでは使用不可。

### 3.2 Shop/Category単位の代表分類

| CategoryCode | 代表的TheoreticalBasis | 確率分布 | 需要特性 |
|--------------|----------------------|---------|---------|
| **WOMEN'S_JACKETS2** | 高級カテゴリ→正規分布; 中マージン→季節商品 | 正規分布（名目） | イベント的需要 |
| **WOMEN'S_ONEPIECE** | ベーシックカテゴリ→負の二項分布; 中マージン→季節商品 | 負の二項分布 | 集中的需要 |
| **Mens_KNIT** | 季節カテゴリ→ポアソン分布; 中マージン→季節商品 | ポアソン分布 | 季節的需要 |
| **Mens_PANTS** | ベーシックカテゴリ→負の二項分布; 中マージン→季節商品 | 負の二項分布 | 安定的需要 |
| **Number_of_guests** | N/A（日次集計なし） | 負の二項分布 | 外部要因依存 |

---

## 4. TASK3: 黒字要因TOP5の詳細分析

### 4.1 月次統計分析（損益データ）

![月次ヒストグラム](output/top5_factors_histogram.png)

#### 4.1.1 WOMEN'S_JACKETS2（1位）

| 統計量 | 値 | 解釈 |
|--------|-----|------|
| 平均 | 6,746,196円 | 高額売上カテゴリ |
| 標準偏差 | 2,698,722円 | 変動幅大（CV=40%） |
| 分散 | 7.28×10¹²円² | 極めて大きい |
| 歪度 | -0.46 | 左に偏り（高売上月が多い） |
| 尖度 | 0.09 | ほぼ正規分布 |
| 正規性検定 | p=0.0006 | **非正規分布** |

#### 4.1.2 Number_of_guests（2位）

| 統計量 | 値 | 解釈 |
|--------|-----|------|
| 平均 | 975人 | 店舗平均客数 |
| 標準偏差 | 469人 | 変動幅大（CV=48%） |
| 分散 | 220,398人² | 過分散 |
| 歪度 | 0.05 | ほぼ対称 |
| 尖度 | -0.09 | 平坦な分布 |
| 分散/平均比 | **225.95** | **極度の過分散** |
| 正規性検定 | p=0.003 | **非正規分布** |

**店舗別特性:**
- 恵比寿: 平均727人、CV=47.4%
- 横浜元町: 平均1,224人、CV=36.6%

**季節性（月別平均）:**
- 最低: 5月（706人）
- 最高: 12月（1,316人）
- 傾向: 12月→1月→2月と逓減

#### 4.1.3 WOMEN'S_ONEPIECE（3位）

| 統計量 | 値 | 解釈 |
|--------|-----|------|
| 平均 | 3,290,808円 | 中規模売上 |
| 標準偏差 | 1,451,587円 | 変動幅中 |
| 歪度 | -0.11 | ほぼ対称 |
| 正規性検定 | p=0.009 | **非正規分布** |

#### 4.1.4 Mens_KNIT（4位）

| 統計量 | 値 | 解釈 |
|--------|-----|------|
| 平均 | 1,592,326円 | 中規模売上 |
| 標準偏差 | 702,381円 | 変動幅中 |
| 歪度 | -0.11 | ほぼ対称 |
| 正規性検定 | p=0.009 | **非正規分布** |

#### 4.1.5 Mens_PANTS（5位）

| 統計量 | 値 | 解釈 |
|--------|-----|------|
| 平均 | 2,441,567円 | 中規模売上 |
| 標準偏差 | 1,076,984円 | 変動幅中 |
| 歪度 | -0.11 | ほぼ対称 |
| 正規性検定 | p=0.009 | **非正規分布** |

### 4.2 日次統計分析（時系列データ）

![日次ヒストグラム](output/daily_quantity_histogram.png)

#### 日次販売数量の分布特性

| CategoryCode | 平均 | 標準偏差 | 歪度 | 尖度 | ゼロ率 | 分散/平均比 |
|--------------|------|---------|------|------|--------|-------------|
| WOMEN'S_JACKETS2 | 11.04 | 15.02 | 8.82 | 216.74 | 8.8% | **20.44** |
| WOMEN'S_ONEPIECE | 5.40 | 8.62 | 17.37 | 814.26 | 13.4% | **13.77** |
| Mens_KNIT | 2.62 | 3.56 | 16.93 | 649.06 | 16.2% | **4.85** |
| Mens_PANTS | 4.00 | 6.70 | 13.68 | 428.40 | 16.0% | **11.20** |

**重要な発見:**
- 全カテゴリで**分散/平均比 > 1**（過分散）
- 全カテゴリで**ゼロ率がポアソン期待値を大幅に超過**（ゼロ過剰）
- 高い歪度・尖度 → 右裾が重い分布

---

## 5. 確率分布の理論的解説

### 5.1 負の二項分布（Negative Binomial Distribution）

**適用対象**: Number_of_guests, WOMEN'S_ONEPIECE, Mens_PANTS

#### 定義
負の二項分布は、成功確率pのベルヌーイ試行をr回成功するまでに必要な失敗回数の分布。

#### 数式
$$P(X=k) = \binom{k+r-1}{k} p^r (1-p)^k$$

#### 特徴
| 特性 | 説明 |
|------|------|
| 平均 | μ = r(1-p)/p |
| 分散 | σ² = r(1-p)/p² |
| **過分散対応** | σ² > μ（分散が平均より大きい場合に適切） |
| ポアソン分布との関係 | r→∞でポアソン分布に収束 |

#### Number_of_guestsへの適用理由
- 分散/平均比 = 225.95（極度の過分散）
- 客数は「来店イベント」の集積であり、日によってばらつきが大きい
- 天候・曜日・イベントなどの外部要因により分散が増大

```
Number_of_guests ~ NegBin(r, p)
  r（成功回数パラメータ）: 店舗の基本集客力
  p（成功確率）: 来店確率
```

### 5.2 ゼロ過剰ポアソン分布（Zero-Inflated Poisson, ZIP）

**適用対象**: WOMEN'S_ONEPIECE, Mens_KNIT, Mens_PANTS

#### 定義
ゼロ過剰ポアソン分布は、ゼロが通常のポアソン分布で期待されるよりも多く発生するデータに適用。

#### 数式
$$P(X=0) = \pi + (1-\pi)e^{-\lambda}$$
$$P(X=k) = (1-\pi)\frac{\lambda^k e^{-\lambda}}{k!}, \quad k > 0$$

- π: ゼロ過剰パラメータ（構造的ゼロの確率）
- λ: ポアソン分布の平均パラメータ

#### 特徴
| 特性 | 説明 |
|------|------|
| 構造的ゼロ | 「売れない日」が存在（店休日、在庫切れ等） |
| サンプリングゼロ | たまたま売れなかった日 |
| **二重のゼロ生成メカニズム** | 両方を同時にモデル化 |

#### WOMEN'S_ONEPIECEのゼロ過剰検証結果
| 指標 | 値 |
|------|-----|
| 実際のゼロ率 | 13.4% |
| ポアソン期待ゼロ率 | 0.5% |
| **ゼロ過剰度** | **13.0%pt** |

→ ポアソン分布の26倍のゼロが発生 → **ZIPが適切**

### 5.3 なぜChronos2 + TFTが推奨されるのか

#### Chronos2の特徴
| 強み | 説明 |
|------|------|
| 事前学習モデル | 大量の時系列データで事前学習済み |
| 分布フリー | 特定の分布仮定なしで予測可能 |
| ゼロ値対応 | 間欠的需要（ゼロが多いデータ）に対応 |
| 高い歪度対応 | 非正規分布でも安定した予測 |

#### TFT（Temporal Fusion Transformer）の特徴
| 強み | 説明 |
|------|------|
| 注意機構 | 重要な時点に注目可能 |
| 共変量対応 | 曜日・祝日・季節性を明示的にモデル化 |
| 確率分布出力 | 信頼区間を直接出力 |
| 長期依存性 | 長期の季節パターンを捕捉 |

#### アンサンブルの効果
```
WeightedEnsemble = Chronos2(57%) + TFT(39%) + DirectTabular(4%)
```

| 状況 | Chronos2 | TFT | アンサンブル効果 |
|------|----------|-----|-----------------|
| 通常の季節パターン | ◎ | ◎ | 安定予測 |
| 異常なゼロ発生 | ◎ | △ | Chronos2が補完 |
| イベント的需要 | △ | ◎ | TFTが補完 |
| 急激な変動 | △ | ◎ | TFTが補完 |

**結論**: 過分散・ゼロ過剰・季節性が複合するデータには、単一モデルでは対応困難。**Chronos2 + TFTのアンサンブルが最適解**。

---

## 6. AutoGluon-TimeSeriesでのモデル検証

### 6.1 NegBin（負の二項分布）モデルの検証方法

AutoGluon-TimeSeriesでは、以下のアプローチでNegBinモデルを検証できる：

#### 方法1: DeepARモデルのdistr設定
```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    prediction_length=90,
    eval_metric="WQL",
    freq="D",
)

# DeepARでnegative_binomial分布を指定
predictor.fit(
    train_data,
    hyperparameters={
        "DeepAR": {
            "distr_output": "negative_binomial",  # NegBin分布
            "num_layers": 2,
            "hidden_size": 40,
        }
    },
    time_limit=1800,
)
```

#### 方法2: AutoGluonのauto_hyperparameters
```python
predictor.fit(
    train_data,
    presets="high_quality",
    hyperparameters={
        "DeepAR": {"distr_output": "negative_binomial"},
        "SimpleFeedForward": {"distr_output": "negative_binomial"},
    },
    time_limit=3600,
)
```

### 6.2 Mens_PANTSでの具体的検証手順

```python
# -*- coding: utf-8 -*-
"""
Mens_PANTS カテゴリの NegBin モデル検証
"""
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# データ読み込み
df = pd.read_excel('time_series_forecast_data_2024_fixed.xlsx',
                   sheet_name='DailyForecastData')

# Mens_PANTSのみ抽出
df_pants = df[df['CategoryCode'] == 'Mens_PANTS'].copy()
df_pants = df_pants[['Date', 'ItemCode', 'ForecastQuantity']]
df_pants['Date'] = pd.to_datetime(df_pants['Date'])

# TimeSeriesDataFrame作成
ts_df = TimeSeriesDataFrame.from_data_frame(
    df_pants,
    id_column='ItemCode',
    timestamp_column='Date'
)

# NegBinモデルで学習
predictor = TimeSeriesPredictor(
    prediction_length=90,
    eval_metric="WQL",
    freq="D",
    path="negbin_pants_model"
)

predictor.fit(
    ts_df,
    hyperparameters={
        "DeepAR": {"distr_output": "negative_binomial"},
    },
    time_limit=600,  # 10分
)

# リーダーボード確認
print(predictor.leaderboard())
```

### 6.3 期待される検証結果

| モデル | 分布仮定 | Mens_PANTS適合度 |
|--------|---------|-----------------|
| DeepAR (NegBin) | 負の二項分布 | ◎ 過分散対応 |
| DeepAR (Gaussian) | 正規分布 | △ 過分散非対応 |
| Chronos2 | 分布フリー | ○ 汎用的 |
| TFT | 正規分布ベース | △ 共変量効果で補完 |

---

## 7. 推奨アクションサマリー

### 7.1 WOMEN'S_JACKETS2（1位）

| 項目 | 内容 |
|------|------|
| **分布特性** | 高級カテゴリ、正規分布ベース、季節商品 |
| **ヒストグラム所見** | 左偏り（高売上月が多い）、高分散 |
| **数理最適化** | 販促イベントが必要条件、在庫リスク考慮 |
| **時系列予測** | Chronos2 + TFT推奨（季節性対応） |
| **推奨アクション** | **1月中に集中販促実施**、2月以降は逓減見込み |

### 7.2 Number_of_guests（2位）

| 項目 | 内容 |
|------|------|
| **分布特性** | **負の二項分布**（分散/平均比=226、極度の過分散） |
| **ヒストグラム所見** | 対称分布、高変動係数（48%） |
| **季節性** | 12月最高（1,316人）→5月最低（706人） |
| **数理最適化** | 店舗別パラメータ必要、外部要因考慮 |
| **時系列予測** | **NegBinモデル必須**、共変量（曜日・天候）追加推奨 |
| **推奨アクション** | 12月の高客数活用、5月にイベント企画 |

### 7.3 WOMEN'S_ONEPIECE（3位）

| 項目 | 内容 |
|------|------|
| **分布特性** | **ゼロ過剰ポアソン分布**（ゼロ率13.4% vs 期待0.5%） |
| **ヒストグラム所見** | 極めて高い尖度（814）、右裾重い |
| **数理最適化** | 季節商品としての在庫リスク管理 |
| **時系列予測** | **Chronos2 + TFT推奨**（ゼロインフレ対応） |
| **推奨アクション** | 春夏シーズン（3-6月）集中販促、秋冬在庫縮小 |

### 7.4 Mens_KNIT（4位）

| 項目 | 内容 |
|------|------|
| **分布特性** | 季節カテゴリ、ポアソン分布ベース |
| **ヒストグラム所見** | 高いゼロ率（16.2%）、分散/平均比=4.85 |
| **数理最適化** | 秋冬限定商品としての特性 |
| **時系列予測** | **Chronos2単独で対応可能**（ポアソンに強い） |
| **推奨アクション** | 10月-12月集中販促、1月以降在庫処分 |

### 7.5 Mens_PANTS（5位）

| 項目 | 内容 |
|------|------|
| **分布特性** | ベーシックカテゴリ、**負の二項分布**（分散/平均比=11.2） |
| **ヒストグラム所見** | 高いゼロ率（16.0%）、高尖度（428） |
| **数理最適化** | 年間安定販売戦略 |
| **時系列予測** | **DeepAR (NegBin) + TFTアンサンブル推奨** |
| **推奨アクション** | 通年販売、セール時期に集中投入 |

#### 7.5.1 AutoGluon-TimeSeries実行結果（2026-01-02実施）

Mens_PANTSデータに対して、DeepAR-NegBinとChronos-Boltの比較検証を実施した。

**データ特性分析結果:**
| 項目 | 値 | 解釈 |
|------|-----|------|
| データ行数 | 12,438行 | 日次販売データ |
| ItemCode数 | 3 | 商品SKU |
| 平均販売数量 | 4.00個 | 低頻度販売 |
| 分散 | 44.83 | 高分散 |
| **分散/平均比** | **11.20** | **極度の過分散** |

**モデル比較結果（WQLスコア）:**
| モデル | WQL（低いほど良い） | 訓練時間 | 備考 |
|--------|-------------------|----------|------|
| **DeepAR-NegBin** | **0.2200** ✓ | 150.5秒 | **Best Model** |
| Chronos-Bolt | 0.2337 | 1.8秒 | Zero-shot |

**検証結果の考察:**
1. **DeepAR-NegBinがChronos-Boltより6.2%優れた精度**を達成
2. 過分散データ（分散/平均比=11.2）に対して、負の二項分布を明示的にモデル化することで予測精度が向上
3. Chronos-Boltは訓練時間が短く（1.8秒 vs 150秒）、ゼロショット性能として優秀
4. 精度重視の場合はDeepAR-NegBin、速度重視の場合はChronos-Boltを選択

**実行環境:**
- AutoGluon Version: 1.5.0
- Python: 3.11.4
- CPU: AMD64 32コア
- GPU: 未使用（CPU推論）

**結論:** Mens_PANTSのような過分散カウントデータには、**DeepAR with NegativeBinomialOutput**が最適。Chronos-Boltとのアンサンブルでさらなる精度向上が期待される。

### 7.6 モデル選定サマリー

| カテゴリ | 推奨モデル | 理由 |
|---------|-----------|------|
| WOMEN'S_JACKETS2 | Chronos2 + TFT | 季節性＋イベント需要 |
| Number_of_guests | DeepAR (NegBin) + 共変量 | 極度の過分散 |
| WOMEN'S_ONEPIECE | Chronos2 + TFT | ゼロ過剰＋季節性 |
| Mens_KNIT | Chronos2 | ポアソン分布適合 |
| Mens_PANTS | DeepAR (NegBin) + TFT | 過分散＋ゼロ過剰 |

---

## 8. GitHub/Render更新手順

### 8.1 GitHubへのプッシュ手順

```bash
# 1. 変更をステージング
cd streamlit_app
git add .

# 2. コミット
git commit -m "Add Top5 Factors Analysis Report and Histograms

- Add statistical analysis for top 5 profitability factors
- Add histogram visualizations (monthly and daily)
- Document theoretical basis for NegBin and ZIP distributions
- Add AutoGluon-TimeSeries NegBin model verification guide
- Update recommended actions for each category

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# 3. リモートにプッシュ
git push origin main
```

### 8.2 Renderの更新手順

#### 自動デプロイの場合（推奨）
Renderは`render.yaml`の設定によりGitHubのmainブランチにプッシュすると**自動でデプロイ**される。

1. GitHubにプッシュ後、Renderダッシュボードを確認
2. デプロイが自動開始されていることを確認
3. デプロイ完了後、URLにアクセスして動作確認

#### 手動デプロイの場合
1. [Render Dashboard](https://dashboard.render.com/) にログイン
2. 対象サービス（store-profit-optimizer）を選択
3. 「Manual Deploy」→「Deploy latest commit」をクリック
4. ビルドログを確認
5. デプロイ完了後、公開URLで動作確認

### 8.3 デプロイ確認チェックリスト

- [ ] GitHubにプッシュ完了
- [ ] Renderでビルド開始を確認
- [ ] ビルドエラーがないことを確認
- [ ] デプロイ完了（通常2-5分）
- [ ] 公開URLでアプリが動作することを確認
- [ ] 新しいレポートファイルがアクセス可能

---

## 9. 最終仕様サマリー

### 9.1 時系列予測モデル仕様

| カテゴリ | 分布特性 | 推奨モデル | 評価指標 | 備考 |
|---------|---------|-----------|---------|------|
| WOMEN'S_JACKETS2 | 正規分布ベース | Chronos2 + TFT | WQL | 季節性＋イベント需要対応 |
| Number_of_guests | NegBin（分散/平均=226） | DeepAR (NegBin) + 共変量 | WQL | 極度の過分散、曜日・天候共変量必須 |
| WOMEN'S_ONEPIECE | ZIP（ゼロ率13.4%） | Chronos2 + TFT | WQL | ゼロ過剰＋季節性対応 |
| Mens_KNIT | ポアソン分布 | Chronos2 | WQL | 季節カテゴリ |
| Mens_PANTS | NegBin（分散/平均=11.2） | DeepAR (NegBin) + TFT | WQL | **実証済み**: WQL=0.2200 |

### 9.2 AutoGluon-TimeSeries実装仕様

```python
# 推奨設定（過分散カウントデータ向け）
from autogluon.timeseries import TimeSeriesPredictor
from gluonts.torch.distributions import NegativeBinomialOutput

predictor = TimeSeriesPredictor(
    prediction_length=30,
    eval_metric='WQL',           # 確率予測評価に最適
    quantile_levels=[0.1, 0.5, 0.9],
    freq='D'
)

predictor.fit(
    train_data,
    hyperparameters={
        'DeepAR': {
            'distr_output': NegativeBinomialOutput(),  # NegBin分布
            'context_length': 60,
            'num_layers': 2,
            'hidden_size': 40,
            'max_epochs': 50,
        },
        'Chronos': {'model_path': 'bolt_small'},  # Zero-shot
    },
    time_limit=600,
)
```

### 9.3 Streamlitアプリ統合仕様

| ページ | ファイル | 機能 |
|--------|---------|------|
| 7_統計分析レポート | `pages/7_統計分析レポート.py` | インタラクティブレポート表示 |

**表示セクション:**
1. 概要 - TOP5要因一覧
2. ヒストグラム分析 - PNG画像表示（月次/日次）
3. 確率分布理論 - NegBin/ZIP解説（タブ切替）
4. モデル比較結果 - WQLチャート表示
5. 推奨アクション - カテゴリ別推奨
6. フルレポート - Markdown表示＋ダウンロード

### 9.4 デプロイ仕様

| 項目 | 設定値 |
|------|--------|
| プラットフォーム | Render (Web Service) |
| リポジトリ | https://github.com/TakashiTakenouchi/profit-improvement-dashboard |
| ブランチ | main |
| 自動デプロイ | 有効 (`autoDeploy: true`) |
| Python | 3.11 |
| フレームワーク | Streamlit |

### 9.5 技術的注意点

1. **Windows日本語パス問題**
   - joblib TEMPフォルダーに日本語ユーザー名が含まれる場合、環境変数で回避
   ```python
   os.environ['TEMP'] = r"C:\path\to\ascii\temp"
   os.environ['JOBLIB_TEMP_FOLDER'] = r"C:\path\to\ascii\temp"
   ```

2. **NegBin分布入力形式**
   - `distr_output`は文字列ではなく`NegativeBinomialOutput()`オブジェクトを指定

3. **整数データ要件**
   - NegBin分布はカウントデータ（整数）を想定
   ```python
   data['target'] = data['target'].round().astype(int).clip(lower=0)
   ```

---

## バージョン履歴

| バージョン | 日付 | 更新者 | 変更内容 |
|-----------|------|--------|---------|
| 1.0.0 | 2026-01-02 | Takashi.Takenouchi | 初版作成 |
| 1.1.0 | 2026-01-02 | Takashi.Takenouchi | Mens_PANTS NegBin vs Chronos-Bolt実行結果追加（セクション7.5.1） |
| 1.2.0 | 2026-01-02 | Takashi.Takenouchi | 最終仕様サマリー追加（セクション9）、Streamlitアプリ統合 |

---

**作成日**: 2026-01-02
**最終更新日**: 2026-01-02
**作成者**: Takashi.Takenouchi
**更新者**: Takashi.Takenouchi
