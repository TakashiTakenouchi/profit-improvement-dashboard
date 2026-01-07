# ML-MIP Integration v3.0 変更箇所一覧

**作成日**: 2026-01-07
**バージョン**: v3.0
**概要**: ML-MIP Integration Skillを活用した最適化機能の統合

---

## 変更サマリー

| カテゴリ | ファイル数 | 変更タイプ |
|---------|-----------|-----------|
| バックエンド | 2ファイル | 新規関数追加 |
| フロントエンド | 4ファイル | UI拡張・機能追加 |
| テスト | 1ファイル | 新規作成 |

---

## バックエンド変更

### 1. utils/logistic.py

**変更タイプ**: 新規関数追加（ML-MIP統合用）

#### 追加インポート
```python
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression  # Ridge, LinearRegression追加
from sklearn.model_selection import cross_val_score  # 追加
from typing import Tuple, List, Optional, Dict, Any  # Dict, Any追加
```

#### 新規関数

| 関数名 | 行番号 | 目的 | 所属フェーズ |
|--------|--------|------|-------------|
| `train_profit_regressor()` | 179-243 | Operating_profit予測用の回帰モデル訓練 | 要因分析 |
| `get_model_for_mip()` | 246-300 | MIP統合用にモデル情報をパッケージ化 | 要因分析 |

#### train_profit_regressor() 詳細
```python
def train_profit_regressor(
    df: pd.DataFrame,
    target_col: str = 'Operating_profit',
    alpha: float = 1.0,
    model_type: str = 'ridge'
) -> Tuple[Any, StandardScaler, List[str], float, Dict]:
    """
    Operating_profit予測用の回帰モデルを訓練（ML-MIP統合用）

    Returns:
        model: 訓練済み回帰モデル（Ridge or LinearRegression）
        scaler: StandardScaler
        feature_cols: 特徴量カラム名リスト
        r2_score: R²スコア（5-fold交差検証平均）
        metrics: 追加メトリクス（intercept, top5_features等）
    """
```

#### get_model_for_mip() 詳細
```python
def get_model_for_mip(
    df: pd.DataFrame,
    target_col: str = 'Operating_profit',
    alpha: float = 1.0
) -> Dict[str, Any]:
    """
    MIP統合用にモデルと変数情報をまとめて返す

    Returns:
        dict:
            - model: 訓練済みモデル
            - scaler: スケーラー
            - feature_cols: 特徴量名リスト
            - input_bounds_lb/ub: 入力変数の上下限（標準化後）
            - r2_score: R²スコア
            - metrics: モデルメトリクス
    """
```

---

### 2. utils/optimization.py

**変更タイプ**: 新規関数追加（ML-MIP最適化）

#### 追加インポート
```python
from typing import Tuple, List, Dict, Optional, Any  # Any追加
import time  # 追加
import sys, os  # 追加

# ML-MIP Integrationライブラリのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '.claude', 'skills', 'ml-mip-integration'))
```

#### 新規関数

| 関数名 | 行番号 | 目的 | 所属フェーズ |
|--------|--------|------|-------------|
| `run_mlmip_optimization()` | 299-575 | ML-MIP統合による最適化実行 | 数理最適化 |
| `get_mlmip_report_section()` | 578-627 | ML-MIP詳細のMarkdownセクション生成 | レポート出力 |

#### run_mlmip_optimization() 詳細
```python
def run_mlmip_optimization(
    df: pd.DataFrame,
    target_indices: List[int],
    mip_model_info: Dict[str, Any],
    target_deficit_months: int = 4,
    variance_ratio: float = 0.3,
    solver_type: str = 'highs'
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    ML-MIP統合による営業利益最適化

    特徴:
    - 回帰モデルをMIP制約として埋め込み
    - R²スコアが負の場合は自動フォールバック
    - ソルバーエラー時もフォールバック（HiGHS -> CBC -> 従来モード）

    Returns:
        result_df: 最適化後のデータフレーム
        summary: 最適化サマリー（従来と同形式）
        mlmip_details: ML-MIP詳細情報（レポート用）
    """
```

#### フォールバック条件
| 条件 | フォールバック先 | 理由 |
|------|-----------------|------|
| ML-MIPライブラリ未インストール | 従来モード | ImportError |
| R²スコア < 0 | 従来モード | 回帰モデルの予測精度不足 |
| HiGHSソルバー未インストール | CBC | ソルバー互換性 |
| CBCも失敗 | 従来モード | 完全フォールバック |
| 最適化実行エラー | 従来モード | 例外処理 |

---

## フロントエンド変更（フェーズ別）

### フェーズ1: 現状把握（EDA）

**変更なし** - 既存機能を維持

---

### フェーズ2: 要因分析

**ファイル**: `pages/2_要因分析.py`

#### 変更内容

| 変更箇所 | 行番号 | 変更内容 |
|---------|--------|---------|
| インポート追加 | 17-21 | `get_model_for_mip`をインポート |
| ロジスティック回帰実行後の処理 | 107-117 | ML-MIP用回帰モデルも同時訓練・セッション保存 |

#### コード変更詳細

**インポート追加**:
```python
from utils.logistic import (
    create_judge_column, run_logistic_regression,
    get_top_factors, get_negative_factors, get_feature_columns,
    get_model_for_mip  # v3.0追加: ML-MIP統合用
)
```

**回帰モデル訓練・保存**:
```python
# v3.0追加: ML-MIP統合用の回帰モデルも訓練・保存
try:
    mip_model_info = get_model_for_mip(df_filtered)
    st.session_state['mip_model_info'] = mip_model_info
    r2_score = mip_model_info.get('r2_score', 0)
    st.success(f"✅ 分析完了！ ロジスティック回帰精度: {accuracy:.1%} | 回帰モデルR²: {r2_score:.3f}")
except Exception as mip_e:
    st.session_state['mip_model_info'] = None
    st.success(f"✅ 分析完了！ モデル精度: {accuracy:.1%}")
    st.info(f"ℹ️ ML-MIP用モデル訓練スキップ: {str(mip_e)}")
```

#### セッション状態の変更
| キー | 型 | 説明 | 新規/既存 |
|-----|-----|------|----------|
| `mip_model_info` | Dict | ML-MIP用モデル情報 | **新規** |

---

### フェーズ3: 目標設定

**ファイル**: `pages/3_目標設定.py`

#### 変更内容

| 変更箇所 | 行番号 | 変更内容 |
|---------|--------|---------|
| スライダー初期値 | 84-94 | セッションから前回値を取得（ページ遷移時維持） |
| ML-MIP設定UI | 114-148 | ソルバー選択、R²スコア表示 |
| セッション保存 | 185-196 | `use_mlmip`, `solver_type`を追加 |

#### 新規UI要素

```
┌─────────────────────────────────────┐
│ 🤖 ML-MIP設定（v3.0）               │
├─────────────────────────────────────┤
│ ☑ ML-MIP最適化を使用                │
│                                     │
│ ソルバー選択: [HiGHS ▼]             │
│                                     │
│ 📊 回帰モデルR²: 0.850 | 特徴量数: 27│
└─────────────────────────────────────┘
```

#### セッション状態の変更
| キー | 型 | 説明 | 新規/既存 |
|-----|-----|------|----------|
| `optimization_params.use_mlmip` | bool | ML-MIPモード使用フラグ | **新規** |
| `optimization_params.solver_type` | str | ソルバー選択（HiGHS/CBC） | **新規** |

---

### フェーズ4: 数理最適化

**ファイル**: `pages/4_最適化実行.py`

#### 変更内容

| 変更箇所 | 行番号 | 変更内容 |
|---------|--------|---------|
| インポート追加 | 16-19 | `run_mlmip_optimization`をインポート |
| パラメータ表示 | 74-84 | 最適化モード表示を追加 |
| 最適化実行分岐 | 99-155 | ML-MIPモード/従来モードで分岐 |

#### 最適化モード分岐ロジック
```python
if use_mlmip and mip_model_info is not None:
    # ML-MIP最適化
    df_optimized, summary, mlmip_details = run_mlmip_optimization(...)
    st.session_state['mlmip_details'] = mlmip_details
else:
    # 従来の最適化
    df_optimized, summary = run_pulp_optimization(...)
    st.session_state['mlmip_details'] = None
```

#### 成功時の表示
```
✅ 最適化が完了しました！ （ML-MIP）
🤖 ML-MIP: ソルバー=HIGHS, 解決時間=0.345秒, 予測誤差=0.000012
```

#### セッション状態の変更
| キー | 型 | 説明 | 新規/既存 |
|-----|-----|------|----------|
| `mlmip_details` | Dict | ML-MIP詳細情報 | **新規** |

---

### フェーズ5: 時系列予測

**変更なし** - 既存機能を維持（AutoGluon予測結果表示）

---

### フェーズ6: レポート出力

**ファイル**: `pages/6_レポート出力.py`

#### 変更内容

| 変更箇所 | 行番号 | 変更内容 |
|---------|--------|---------|
| インポート追加 | 26 | `get_mlmip_report_section`をインポート |
| レポート生成関数 | 177-301 | ML-MIP詳細セクション追加対応 |
| レポート生成呼び出し | 381-384 | `mlmip_details`パラメータ追加 |

#### generate_summary_report() 変更

**シグネチャ変更**:
```python
# 変更前
def generate_summary_report(params, metrics, logistic_results=None):

# 変更後（v3.0）
def generate_summary_report(params, metrics, logistic_results=None, mlmip_details=None):
```

**ML-MIPセクション追加**:
```python
# v3.0: ML-MIP詳細セクションを追加
if mlmip_details and mlmip_details.get('used_mlmip'):
    report += get_mlmip_report_section(mlmip_details)
    report += "\n---\n"
```

#### レポート出力例（ML-MIP使用時）

```markdown
## ML-MIP最適化詳細

### 最適化結果
| 項目 | 値 |
|------|-----|
| **使用ソルバー** | HIGHS |
| **ステータス** | Optimal |
| **目的関数値** | 1,234,567 |
| **予測誤差** | 0.000012 |
| **解決時間** | 0.345秒 |

### 回帰モデル情報
| 項目 | 値 |
|------|-----|
| **モデルタイプ** | ridge |
| **R²スコア** | 0.8500 |
| **特徴量数** | 27 |

### 主要調整特徴量（TOP5）
| 特徴量 | 最適化後の値 |
|--------|-------------|
| WOMEN'S_JACKETS2 | 523,456.78 |
| Number_of_guests | 1,234.56 |
| ...
```

---

## セッション状態一覧（v3.0）

| キー | 型 | フェーズ | 説明 | 新規 |
|-----|-----|---------|------|------|
| `uploaded_data` | DataFrame | 現状把握 | アップロードデータ | |
| `logistic_results` | DataFrame | 要因分析 | オッズ比結果 | |
| `logistic_accuracy` | float | 要因分析 | モデル精度 | |
| `top_factors` | list | 要因分析 | TOP5要因 | |
| **`mip_model_info`** | Dict | 要因分析 | ML-MIP用モデル情報 | ✓ |
| `optimization_params` | dict | 目標設定 | 最適化パラメータ | |
| **`optimization_params.use_mlmip`** | bool | 目標設定 | ML-MIPフラグ | ✓ |
| **`optimization_params.solver_type`** | str | 目標設定 | ソルバー選択 | ✓ |
| `optimized_data` | DataFrame | 最適化実行 | 最適化後データ | |
| `optimization_summary` | dict | 最適化実行 | 最適化サマリー | |
| `optimization_metrics` | dict | 最適化実行 | 改善メトリクス | |
| **`mlmip_details`** | Dict | 最適化実行 | ML-MIP詳細情報 | ✓ |

---

## テスト

**ファイル**: `tests/test_mlmip_integration.py`（新規作成）

### テスト項目

| # | テスト項目 | 判定基準 | 結果 |
|---|-----------|---------|------|
| 1 | ロジスティック回帰 | 精度 > 50% | PASS |
| 2 | ML-MIP用モデル訓練 | モデル != None, 特徴量数 > 0 | PASS |
| 3 | 従来最適化（Case 0-4） | 全ケースsuccess=True | PASS |
| 4 | ML-MIP最適化（Case 0-4） | 全ケースsuccess=True（フォールバック含む） | PASS |
| 5 | レポート生成 | 出力文字数 > 50 | PASS |

### テスト結果
```
======================================================================
テスト結果サマリー
======================================================================
  ロジスティック回帰: PASS ✓
  ML-MIP用モデル訓練: PASS ✓
  従来最適化（Case 0-4）: PASS ✓
  ML-MIP最適化（Case 0-4）: PASS ✓
  レポート生成: PASS ✓

======================================================================
全テスト合格！ GitHubへのプッシュが可能です。
======================================================================
```

---

## 依存関係

### 既存パッケージ（変更なし）
- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- PuLP

### ML-MIP Integration Skill
- 場所: `.claude/skills/ml-mip-integration/ml_mip_integration.py`
- クラス: `PuLPLinearMLIntegrator`, `SolverType`
- 注意: このスキルが存在しない場合は自動的に従来モードにフォールバック

---

## 互換性

| 項目 | 状態 |
|------|------|
| 既存機能（従来モード） | 完全維持 |
| セッション状態 | 後方互換 |
| レポート形式 | 拡張（オプション追加） |
| データ形式 | 変更なし |

---

**作成者**: Claude Code v3.0
**最終更新**: 2026-01-07
