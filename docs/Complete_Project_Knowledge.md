# 店舗別損益分析・時系列予測プロジェクト 完全ナレッジドキュメント

---

## 目次

1. [プロジェクト全体概要](#1-プロジェクト全体概要)
2. [フォルダ構造とファイル一覧](#2-フォルダ構造とファイル一覧)
3. [Phase 1: ロジスティック回帰分析](#3-phase-1-ロジスティック回帰分析)
4. [Phase 2: EDAダッシュボード](#4-phase-2-edaダッシュボード)
5. [Phase 3: データ整合性修正](#5-phase-3-データ整合性修正)
6. [Phase 4: AutoGluon時系列予測](#6-phase-4-autogluon時系列予測)
7. [Phase 5: Streamlit営業利益改善ダッシュボード](#7-phase-5-streamlit営業利益改善ダッシュボード)
8. [継承・再利用ガイド](#8-継承再利用ガイド)
9. [Q&A ナレッジ](#9-qa-ナレッジ)

---

## 1. プロジェクト全体概要

### 1.1 背景
恵比寿店・横浜元町店の2店舗における営業利益改善と販売予測を目的としたデータ分析プロジェクト。

### 1.2 処理フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         プロジェクト全体フロー                               │
└─────────────────────────────────────────────────────────────────────────────┘

[Phase 1] ロジスティック回帰分析
    │
    ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  入力データ       │ ──▶ │  オッズ比算出     │ ──▶ │  黒字化要因特定   │
│  店舗別損益       │     │  L1正則化         │     │  TOP5抽出         │
└───────────────────┘     └───────────────────┘     └───────────────────┘
    │
    ▼
[Phase 2] EDAダッシュボード
    │
    ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  単変量分析       │ ──▶ │  二変量分析       │ ──▶ │  多変量チェック   │
│  ヒストグラム     │     │  箱ひげ図         │     │  相関行列         │
└───────────────────┘     └───────────────────┘     └───────────────────┘
    │
    ▼
[Phase 3] データ整合性修正
    │
    ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  日付+1年シフト   │ ──▶ │  ForecastQty調整  │ ──▶ │  整合性検証       │
│                   │     │  月次売上一致     │     │  1,104件PASS      │
└───────────────────┘     └───────────────────┘     └───────────────────┘
    │
    ▼
[Phase 4] AutoGluon時系列予測
    │
    ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  TimeSeriesDF     │ ──▶ │  モデル学習       │ ──▶ │  90日間予測       │
│  作成             │     │  Chronos2+TFT     │     │  信頼区間付き     │
└───────────────────┘     └───────────────────┘     └───────────────────┘
    │
    ▼
[Phase 5] Streamlit営業利益改善ダッシュボード（6ページ構成）
    │
    ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  現状把握・要因   │ ──▶ │  目標設定・最適化 │ ──▶ │  時系列予測・     │
│  分析（EDA）      │     │  （PuLP）         │     │  レポート出力     │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

---

## 2. フォルダ構造とファイル一覧

### 2.1 完全フォルダ構造

```
Ensuring consistency between tabular data and time series forecast data/
│
├── input/                                    # 入力データ
│   ├── fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx  # 正データ
│   └── time_series_forecast_data_2024.xlsx   # 時系列データ（修正前）
│
├── src/analysis/                             # 分析スクリプト
│   ├── 01_logistic_regression_analysis_20251227.py  # ロジスティック回帰
│   ├── 02_ebisu_store_improvement_20251227.py       # 恵比寿店改善分析
│   ├── 03_ebisu_profit_variance_20251227.py         # 利益変動分析
│   ├── 04_complete_improvement_vol5_20251227.py     # 総合改善分析
│   ├── 05_ebisu_optimization_vol6_20251227.py       # 最適化分析
│   └── task2_eda_dashboard.py                       # EDAダッシュボード
│
├── Process/                                  # データ処理スクリプト
│   ├── fix_timeseries_data.py                # データ整合性修正
│   ├── autogluon_timeseries_forecast.py      # 予測（初版）
│   └── timeseries_forecast_run_v2.py         # 予測（推奨版）★
│
├── streamlit_app/                            # Streamlitマルチページアプリ ★
│   ├── Home.py                               # エントリーポイント（認証・ホーム画面）
│   ├── pages/
│   │   ├── 1_現状把握.py                     # EDA（VIF、正規性検定）
│   │   ├── 2_要因分析.py                     # オッズ比TOP5抽出
│   │   ├── 3_目標設定.py                     # 赤字月数・変動幅設定
│   │   ├── 4_最適化実行.py                   # Before/After比較
│   │   ├── 5_時系列予測.py                   # 信頼区間付きグラフ
│   │   └── 6_レポート出力.py                 # Markdown/Excelエクスポート
│   ├── components/
│   │   ├── __init__.py
│   │   ├── auth.py                           # 簡易認証（SHA256ハッシュ）
│   │   ├── data_loader.py                    # Excelアップロード・キャッシュ
│   │   └── charts.py                         # Plotlyグラフ生成関数
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logistic.py                       # ロジスティック回帰ロジック
│   │   └── optimization.py                   # PuLP利用の手動最適化アルゴリズム
│   ├── data/                                 # サンプルデータ
│   │   └── (Excelファイル)
│   ├── .streamlit/
│   │   ├── config.toml                       # テーマ設定
│   │   └── secrets.toml.example              # 認証テンプレート
│   ├── requirements.txt                      # 14パッケージ
│   ├── render.yaml                           # Renderデプロイ設定
│   ├── .gitignore
│   └── README.md
│
├── output/                                   # 出力データ
│   ├── time_series_forecast_data_2024_fixed.xlsx  # 修正済み時系列データ
│   ├── forecast_results_2026_90days.xlsx     # 予測結果
│   └── eda/                                  # EDA出力
│       ├── 01_histograms.png
│       ├── 02_boxplots_by_judge.png
│       ├── 03_correlation_heatmap.png
│       ├── eda_dashboard.html
│       └── eda_summary.xlsx
│
├── docs/                                     # ドキュメント
│   ├── AutoGluon_TimeSeries_Knowledge.md     # 時系列予測ナレッジ
│   ├── Colab_Dashboard_Knowledge.md          # Colabダッシュボードナレッジ
│   ├── Complete_Project_Knowledge.md         # 本ドキュメント★
│   └── 時系列予測データ修正要件.md
│
└── autogluon_ts_models/                      # 学習済みモデル
    └── (AutoGluon自動生成)
```

### 2.2 Pythonコード所在一覧

| Phase | ファイル名 | パス | 用途 |
|-------|-----------|------|------|
| Phase1 | 01_logistic_regression_analysis_20251227.py | src/analysis/ | ロジスティック回帰 |
| Phase2 | task2_eda_dashboard.py | src/analysis/ | EDAダッシュボード |
| Phase3 | fix_timeseries_data.py | Process/ | データ整合性修正 |
| Phase4 | timeseries_forecast_run_v2.py | Process/ | 時系列予測（推奨）★ |
| Phase5 | streamlit_app/ | streamlit_app/ | マルチページStreamlitアプリ★ |

---

## 3. Phase 1: ロジスティック回帰分析

### 3.1 目的
営業利益の増減要因（黒字化要因TOP5）を明確化

### 3.2 入力データ
```
fixed_extended_store_data_2024-FIX_kaizen_monthlyvol3.xlsx
```

### 3.3 処理フロー

```python
# 1. judge列の作成（目的変数）
mean_profit = df['Operating_profit'].mean()
df['judge'] = (df['Operating_profit'] > mean_profit).astype(int)

# 2. 説明変数の選定（除外変数）
exclude_cols = ['shop', 'shop_code', 'Date', 'Operating_profit', 'judge',
                'gross_profit', 'operating_cost']

# 3. L1正則化ロジスティック回帰
model = LogisticRegression(
    penalty='l1',
    solver='saga',
    max_iter=2000,
    C=0.5,
    random_state=42
)

# 4. オッズ比算出
odds_ratios = np.exp(model.coef_[0])
```

### 3.4 出力
- 黒字化要因TOP5（オッズ比 > 1）
- 赤字要因（オッズ比 < 1）

### 3.5 コード全文
**ファイル:** `src/analysis/01_logistic_regression_analysis_20251227.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
店舗別損益計算書 ロジスティック回帰分析
営業利益の増減要因（黒字化要因TOP5）を明確化
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 設定
BASE_DIR = r"C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\..."
INPUT_FILE = os.path.join(BASE_DIR, "fixed_extended_store_data_2024-FIX_kaizen_monthlyvol3.xlsx")

# judge列の作成
def create_judge_column(df):
    mean_profit = df['Operating_profit'].mean()
    df['judge'] = (df['Operating_profit'] > mean_profit).astype(int)
    return df, mean_profit

# ロジスティック回帰
def run_logistic_regression(df):
    exclude_cols = ['shop', 'shop_code', 'Date', 'Operating_profit', 'judge',
                    'gross_profit', 'operating_cost']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['judge']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(penalty='l1', solver='saga', max_iter=2000, C=0.5)
    model.fit(X_scaled, y)

    odds_ratios = np.exp(model.coef_[0])
    return pd.DataFrame({'変数': feature_cols, 'オッズ比': odds_ratios})
```

---

## 4. Phase 2: EDAダッシュボード

### 4.1 目的
ロジスティック回帰の事前分析（探索的データ解析）

### 4.2 分析内容

| 分析種別 | 内容 | 出力 |
|---------|------|------|
| 単変量分析 | ヒストグラム、基本統計量 | 01_histograms.png |
| 二変量分析 | judge別分布比較（箱ひげ図） | 02_boxplots_by_judge.png |
| 多変量チェック | 相関行列（ヒートマップ） | 03_correlation_heatmap.png |
| VIF計算 | 分散拡大係数（多重共線性） | eda_summary.xlsx |
| 正規性検定 | Shapiro-Wilk検定 | eda_summary.xlsx |

### 4.3 入力データ
```
fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx
```

### 4.4 出力ファイル
```
output/eda/
├── 01_histograms.png           # ヒストグラム
├── 02_boxplots_by_judge.png    # 箱ひげ図
├── 03_correlation_heatmap.png  # 相関行列
├── eda_dashboard.html          # HTMLダッシュボード
└── eda_summary.xlsx            # 統計サマリー
```

### 4.5 コード全文
**ファイル:** `src/analysis/task2_eda_dashboard.py`

```python
# -*- coding: utf-8 -*-
"""
Task2: ロジスティック回帰の事前分析（EDA）ダッシュボード
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

# ファイルパス
BASE_PATH = r"C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\..."
TARGET_FILE = f"{BASE_PATH}\\fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx"
OUTPUT_DIR = f"{BASE_PATH}\\output\\eda"

# 1. 基本統計量
stats_df = pd.DataFrame({
    '変数名': feature_cols,
    '平均': [df[col].mean() for col in feature_cols],
    '標準偏差': [df[col].std() for col in feature_cols],
    '歪度': [df[col].skew() for col in feature_cols],
    '尖度': [df[col].kurtosis() for col in feature_cols],
})

# 2. 外れ値検出（IQR法）
for col in feature_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR

# 3. judge別t検定
for col in feature_cols:
    t_stat, p_value = stats.ttest_ind(
        df[df['judge']==0][col],
        df[df['judge']==1][col]
    )

# 4. 相関行列・VIF計算
corr_matrix = df[feature_cols].corr()

# 5. 正規性検定（Shapiro-Wilk）
for col in feature_cols:
    stat, p_value = stats.shapiro(df[col].dropna())
```

---

## 5. Phase 3: データ整合性修正

### 5.1 問題点
時系列データの日付が**1年ズレていた**（2019年→2020年）

### 5.2 修正手順

```python
# 1. 日付の+1年シフト
df_daily['Date'] = df_daily['Date'] + pd.DateOffset(years=1)

# 2. OriginalMonthlyDataを正データで上書き
for idx, row in df_orig.iterrows():
    fixed_row = df_fixed[(df_fixed['shop']==row['shop']) &
                          (df_fixed['YearMonth']==row['YearMonth'])]
    if len(fixed_row) > 0:
        for cat_en in categories_en:
            df_orig.loc[idx, cat_en] = fixed_row[cat_en].values[0]

# 3. ForecastQuantity調整
for (shop, ym, cat_code), group_idx in df_daily.groupby(['Shop','YearMonth','CategoryCode']).groups.items():
    target_sales = monthly_sales_dict.get((shop, ym, cat_code), None)
    if target_sales and target_sales > 0:
        current_sales = (df_daily.loc[group_idx, 'UnitPrice'] *
                        df_daily.loc[group_idx, 'ForecastQuantity']).sum()
        if current_sales > 0:
            adjustment_ratio = target_sales / current_sales
            df_daily.loc[group_idx, 'ForecastQuantity'] *= adjustment_ratio
```

### 5.3 検証結果
```
整合性チェック: 1,104件中1,104件 PASS（差異率0%）
```

### 5.4 入出力

| 種別 | ファイル |
|------|---------|
| 入力（正データ） | fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx |
| 入力（修正前） | time_series_forecast_data_2024.xlsx |
| 出力（修正後） | output/time_series_forecast_data_2024_fixed.xlsx |

---

## 6. Phase 4: AutoGluon時系列予測

### 6.1 入力データ形式

```python
# TimeSeriesDataFrame必須列
| item_id | timestamp | target | weekend |
|---------|-----------|--------|---------|
| EBISU_Mens_JACKETS&OUTER2_001 | 2020-04-30 | 103.72 | 0 |

# Static Features
| item_id | Shop | Category |
|---------|------|----------|
| EBISU_Mens_JACKETS&OUTER2_001 | EBISU | Mens_JACKETS&OUTER2 |
```

### 6.2 予測設定

```python
predictor = TimeSeriesPredictor(
    prediction_length=90,           # 90日間予測
    eval_metric="WQL",              # Weighted Quantile Loss
    known_covariates_names=["weekend"],  # 共変量
    freq="D",                       # 日次
    path="autogluon_ts_models",
)

predictor.fit(
    ts_df,
    presets="medium_quality",
    time_limit=1800,  # 30分
)
```

### 6.3 学習結果

| モデル | WQL Score | 採用率 |
|--------|-----------|--------|
| **WeightedEnsemble** | **-0.3298** | 100% |
| Chronos2 | -0.3380 | 57% |
| TFT | -0.3461 | 39% |
| DirectTabular | -0.3774 | 4% |

### 6.4 日本語パス問題の解決

```python
# 環境変数で一時フォルダを英語パスに設定
temp_dir = r'C:\PyCharm\AutoGluon AssistantTEST\temp'
os.environ['TMP'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir

# 店舗名を英語コードに変換
shop_code_map = {'恵比寿': 'EBISU', '横浜元町': 'YOKOHAMA'}
df['item_id'] = df['ShopCode'] + '_' + df['ItemCode']
```

### 6.5 出力ファイル

```
output/forecast_results_2026_90days.xlsx
├── DailyForecasts      # 日次予測結果（4,320行）
├── MonthlySummary      # 月別集計
└── CategorySummary     # カテゴリ別集計
```

---

## 7. Phase 5: Streamlit営業利益改善ダッシュボード

### 7.1 概要

QCストーリー手法に基づく6ページ構成のマルチページStreamlitアプリケーション。
GitHub: https://github.com/TakashiTakenouchi/profit-improvement-dashboard

### 7.2 ディレクトリ構造

```
streamlit_app/
├── Home.py                        # エントリーポイント（認証・ホーム画面）
├── pages/
│   ├── 1_現状把握.py               # EDA（VIF、正規性検定）
│   ├── 2_要因分析.py               # オッズ比TOP5抽出
│   ├── 3_目標設定.py               # 赤字月数・変動幅設定
│   ├── 4_最適化実行.py             # Before/After比較
│   ├── 5_時系列予測.py             # 信頼区間付きグラフ
│   └── 6_レポート出力.py           # Markdown/Excelエクスポート
├── components/
│   ├── __init__.py
│   ├── auth.py                    # 簡易認証（SHA256ハッシュ）
│   ├── data_loader.py             # Excelアップロード・キャッシュ
│   └── charts.py                  # Plotlyグラフ生成関数（7種類）
├── utils/
│   ├── __init__.py
│   ├── logistic.py                # ロジスティック回帰ロジック
│   └── optimization.py            # PuLP利用の手動最適化アルゴリズム
├── data/                          # サンプルデータ
│   └── (Excelファイル)
├── .streamlit/
│   ├── config.toml                # テーマ設定
│   └── secrets.toml.example       # 認証テンプレート
├── requirements.txt               # 14パッケージ
├── render.yaml                    # Renderデプロイ設定
├── .gitignore
└── README.md
```

### 7.3 6ページ構成の詳細

| ページ | ファイル名 | 機能 | 主要コンポーネント |
|--------|-----------|------|-------------------|
| Home | Home.py | 認証・ホーム画面 | auth.py（ログイン/ログアウト） |
| 1. 現状把握 | 1_現状把握.py | EDA（探索的データ解析） | VIF計算、正規性検定（Shapiro-Wilk）、ヒストグラム、箱ひげ図、相関行列 |
| 2. 要因分析 | 2_要因分析.py | ロジスティック回帰 | L1正則化、オッズ比TOP5/Bottom5、正規化パラメータ(C)調整 |
| 3. 目標設定 | 3_目標設定.py | 最適化パラメータ設定 | 赤字月数目標、変動幅(±%)、制約条件設定 |
| 4. 最適化実行 | 4_最適化実行.py | 利益最適化 | PuLP利用の手動最適化、Before/After比較チャート |
| 5. 時系列予測 | 5_時系列予測.py | 予測可視化 | 90日予測、90%信頼区間、WeightedEnsemble情報 |
| 6. レポート出力 | 6_レポート出力.py | エクスポート | Excel(.xlsx)、Markdown(.md)、CSV出力 |

### 7.4 主要コンポーネント

#### 7.4.1 認証機能（auth.py）

```python
# SHA256ハッシュによる簡易認証
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_authentication():
    return st.session_state.get('authenticated', False)

def show_login_form():
    # ログインフォーム表示
    # デモ認証: admin/admin123, user/user123
```

#### 7.4.2 データローダー（data_loader.py）

```python
@st.cache_data
def load_excel_file(uploaded_file):
    """アップロードされたExcelファイルを読み込み"""
    return pd.read_excel(uploaded_file)

def load_sample_data():
    """サンプルデータを読み込み（data/フォルダから）"""
    pass

def show_file_uploader():
    """ファイルアップロードUIまたはサンプルデータ選択"""
    pass
```

#### 7.4.3 グラフ生成（charts.py）

| 関数名 | 用途 |
|--------|------|
| create_histogram() | ヒストグラム（複数サブプロット） |
| create_boxplot() | 箱ひげ図（グループ別） |
| create_correlation_heatmap() | 相関行列ヒートマップ |
| create_odds_ratio_chart() | オッズ比横棒グラフ |
| create_before_after_chart() | 改善前後比較チャート |
| create_time_series_chart() | 時系列予測グラフ（信頼区間付き） |
| create_profit_variance_chart() | 営業利益変動チャート |

#### 7.4.4 最適化ロジック（optimization.py）

**注意**: PuLPライブラリをインポートするが、実際の最適化は**手動アルゴリズム**で実行。

```python
def run_pulp_optimization(df, target_indices, deficit_target, variance_ratio):
    """
    PuLP利用の手動最適化アルゴリズム

    処理内容:
    1. 最低パフォーマンス月を特定（赤字月候補）
    2. 赤字月の利益を削減し、黒字月に再配分
    3. 年間合計は維持しつつ月次配分を最適化
    4. judge列を再計算（平均営業利益基準）
    """
    pass

def calculate_improvement_metrics(df_before, df_after):
    """改善前後のメトリクス比較"""
    # 赤字月数、黒字月数、合計利益、平均利益、最小/最大
    pass

def get_monthly_comparison(df_before, df_after):
    """月次比較テーブル生成"""
    pass
```

### 7.5 依存パッケージ（requirements.txt）

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
openpyxl>=3.1.0
xlsxwriter>=3.1.0
japanize-matplotlib>=1.1.3
PuLP>=2.7.0
streamlit-authenticator>=0.3.2
python-dotenv>=1.0.0
scipy>=1.11.0
statsmodels>=0.14.0
seaborn>=0.12.0
```

### 7.6 起動方法

```bash
# 方法1: コマンドライン
cd streamlit_app
streamlit run Home.py

# 方法2: Python経由
python -m streamlit run Home.py

# アクセスURL
http://localhost:8501

# デモ認証情報
Username: admin
Password: admin123
```

### 7.7 デプロイ設定（render.yaml）

```yaml
services:
  - type: web
    name: store-profit-optimizer
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run Home.py --server.port $PORT --server.headless true
```

---

## 8. 継承・再利用ガイド

### 8.1 新しいデータでの再利用手順

```
Step 1: 入力データを input/ フォルダに配置
        └── 月次損益データ（Excel形式）

Step 2: 01_logistic_regression_analysis.py を実行
        └── INPUT_FILE パスを変更

Step 3: task2_eda_dashboard.py を実行
        └── TARGET_FILE パスを変更

Step 4: fix_timeseries_data.py を実行（データ整合性が必要な場合）
        └── fixed_file, ts_file パスを変更

Step 5: timeseries_forecast_run_v2.py を実行
        └── input_file, data_dir パスを変更

Step 6: streamlit_app/Home.py を起動
        └── data/ フォルダにサンプルデータを配置
```

### 8.2 カスタマイズポイント

| 項目 | ファイル | 変更箇所 |
|------|---------|---------|
| 店舗名 | timeseries_forecast_run_v2.py | shop_code_map |
| カテゴリ | timeseries_forecast_run_v2.py | category_jp_map |
| 予測期間 | timeseries_forecast_run_v2.py | prediction_length |
| 共変量 | timeseries_forecast_run_v2.py | known_covariates_names |
| モデル品質 | timeseries_forecast_run_v2.py | presets |
| 学習時間 | timeseries_forecast_run_v2.py | time_limit |
| 認証設定 | streamlit_app/.streamlit/secrets.toml | credentials |
| テーマ | streamlit_app/.streamlit/config.toml | [theme] |

### 8.3 テンプレートコード

#### ロジスティック回帰テンプレート
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# データ準備
X = df[feature_cols].fillna(df[feature_cols].mean())
y = df['target']

# 標準化 + L1正則化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(penalty='l1', solver='saga', C=0.5)
model.fit(X_scaled, y)

# オッズ比
odds_ratios = np.exp(model.coef_[0])
```

#### 時系列予測テンプレート
```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# TimeSeriesDataFrame作成
ts_df = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column='item_id',
    timestamp_column='timestamp'
)
ts_df.static_features = static_features

# 予測器設定
predictor = TimeSeriesPredictor(
    prediction_length=90,
    eval_metric="WQL",
    known_covariates_names=["weekend"],
    freq="D",
)

# 学習
predictor.fit(ts_df, presets="medium_quality", time_limit=1800)

# 予測
predictions = predictor.predict(ts_df, known_covariates=future_covariates_df)
```

#### Streamlitマルチページテンプレート
```python
# Home.py
import streamlit as st
from components.auth import check_authentication, show_login_form

def main():
    st.set_page_config(page_title="ダッシュボード", layout="wide")

    if not check_authentication():
        show_login_form()
        return

    st.title("ホーム画面")
    # ページコンテンツ

if __name__ == "__main__":
    main()
```

---

## 9. Q&A ナレッジ

### Q1: Excelで信頼区間付きグラフを作成する方法

1. データ準備（新しい列を追加）
   ```
   Upper_Error = 0.9列 - predicted_quantity列
   Lower_Error = predicted_quantity列 - 0.1列
   ```

2. 折れ線グラフを作成

3. 誤差範囲を追加
   - グラフデザイン → グラフ要素を追加 → 誤差範囲
   - ユーザー設定で Upper_Error, Lower_Error を指定

### Q2: Chronos2が57%で選ばれた理由

| 分布型 | 歪度 | ゼロ率 | Chronos2適合度 |
|-------|-----|-------|---------------|
| ポアソン分布 | 19.0 | 37.6% | ◎ |
| 正規分布 | 13.1 | 13.0% | ◎ |
| 負の二項分布 | 10.7 | 20.7% | ◎ |

**理由:**
1. 非正規分布（高い歪度・尖度）に対応
2. ゼロインフレーションに強い
3. 分布仮定なしの基盤モデル

### Q3: 日本語パスでエラーが出る場合

```python
# 一時フォルダを英語パスに設定
temp_dir = r'C:\temp\autogluon'
os.makedirs(temp_dir, exist_ok=True)
os.environ['TMP'] = temp_dir
os.environ['TEMP'] = temp_dir
os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
```

### Q4: Streamlitでグラフが表示されない場合

```python
# Plotly 6.0互換のためScatterで縦線を描画
fig.add_trace(go.Scatter(
    x=[forecast_start, forecast_start],
    y=[y_min, y_max],
    mode='lines',
    line=dict(color='red', dash='dash')
))
```

### Q5: optimization.pyの最適化方式について

**重要**: `optimization.py`はPuLPライブラリをインポートするが、実際の最適化処理は**手動アルゴリズム**で実行される。

処理内容:
1. 営業利益が最も低い月を特定（赤字月候補）
2. 赤字月の利益を削減し、黒字月に再配分
3. 年間合計営業利益は維持（制約条件）
4. judge列（黒字/赤字フラグ）を再計算

---

## 10. 参考資料

- [AutoGluon TimeSeries Documentation](https://auto.gluon.ai/stable/tutorials/timeseries/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [PuLP Documentation](https://coin-or.github.io/pulp/)
- [GitHub Repository](https://github.com/TakashiTakenouchi/profit-improvement-dashboard)

---

## バージョン履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2025-12-28 | 初版作成 |
| 2.0.0 | 2026-01-01 | Phase 5をstreamlit_app/マルチページ構成に更新、data/フォルダ追加、optimization.py説明補足 |

---

**作成日**: 2025-12-28
**更新日**: 2026-01-01
**作成者**: Takashi.Takenouchi
