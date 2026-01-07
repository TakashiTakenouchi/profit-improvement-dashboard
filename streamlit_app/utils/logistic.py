# -*- coding: utf-8 -*-
"""
ロジスティック回帰分析ユーティリティ
営業利益の黒字化要因を分析

【v3.0追加】ML-MIP統合用の回帰モデル訓練機能
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Tuple, List, Optional, Dict, Any


# 分析から除外するカラム
EXCLUDE_COLUMNS = [
    'shop', 'shop_code', 'Date', 'year', 'month',
    'Operating_profit', 'gross_profit', 'operating_cost',
    'judge', 'Total_Sales', 'discount', 'purchasing'
]


def create_judge_column(df: pd.DataFrame, target_col: str = 'Operating_profit') -> Tuple[pd.DataFrame, float]:
    """
    judge列を作成（平均値基準で黒字/赤字を分類）

    Args:
        df: データフレーム
        target_col: 目標変数のカラム名

    Returns:
        df: judge列が追加されたデータフレーム
        mean_value: 平均値
    """
    df = df.copy()
    mean_value = df[target_col].mean()
    df['judge'] = (df[target_col] > mean_value).astype(int)
    return df, mean_value


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """分析に使用する特徴量カラムを取得"""
    return [col for col in df.columns if col not in EXCLUDE_COLUMNS]


def run_logistic_regression(df: pd.DataFrame, C: float = 0.5,
                            max_iter: int = 2000) -> Tuple[pd.DataFrame, float]:
    """
    L1正則化ロジスティック回帰を実行

    Args:
        df: データフレーム（judge列必須）
        C: 正則化パラメータ（小さいほど強い正則化）
        max_iter: 最大イテレーション数

    Returns:
        results_df: 特徴量、係数、オッズ比のデータフレーム
        accuracy: モデルの精度
    """
    feature_cols = get_feature_columns(df)

    # 特徴量と目的変数の準備
    X = df[feature_cols].fillna(0)
    y = df['judge']

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ロジスティック回帰
    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=max_iter,
        C=C,
        random_state=42
    )
    model.fit(X_scaled, y)

    # オッズ比計算
    odds_ratios = np.exp(model.coef_[0])

    # 結果をデータフレームに
    results_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0],
        'odds_ratio': odds_ratios
    })

    # オッズ比でソート
    results_df = results_df.sort_values('odds_ratio', ascending=False)

    # 精度
    accuracy = model.score(X_scaled, y)

    return results_df, accuracy


def calculate_odds_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """オッズ比を計算して返す"""
    results_df, _ = run_logistic_regression(df)
    return results_df


def get_top_factors(results_df: pd.DataFrame, n: int = 5,
                    positive_only: bool = True) -> pd.DataFrame:
    """
    上位n個の要因を取得

    Args:
        results_df: ロジスティック回帰結果
        n: 取得する要因数
        positive_only: True の場合、オッズ比 > 1 のみ

    Returns:
        上位要因のデータフレーム
    """
    if positive_only:
        filtered = results_df[results_df['odds_ratio'] > 1]
    else:
        filtered = results_df

    return filtered.head(n)


def get_negative_factors(results_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    赤字要因（オッズ比 < 1）を取得

    Args:
        results_df: ロジスティック回帰結果
        n: 取得する要因数

    Returns:
        赤字要因のデータフレーム
    """
    negative = results_df[results_df['odds_ratio'] < 1]
    return negative.sort_values('odds_ratio', ascending=True).head(n)


def analyze_store(df: pd.DataFrame, store_name: str,
                  store_col: str = 'shop') -> Tuple[pd.DataFrame, dict]:
    """
    店舗別のロジスティック回帰分析

    Args:
        df: 全データ
        store_name: 店舗名
        store_col: 店舗カラム名

    Returns:
        results_df: 分析結果
        summary: サマリー情報
    """
    # 店舗でフィルタ
    store_df = df[df[store_col] == store_name].copy()

    # judge列作成
    store_df, mean_profit = create_judge_column(store_df)

    # 分析実行
    results_df, accuracy = run_logistic_regression(store_df)

    # サマリー
    summary = {
        'store_name': store_name,
        'total_months': len(store_df),
        'profit_months': (store_df['judge'] == 1).sum(),
        'deficit_months': (store_df['judge'] == 0).sum(),
        'mean_profit': mean_profit,
        'accuracy': accuracy
    }

    return results_df, summary


# =============================================================================
# ML-MIP統合用関数（v3.0追加）
# =============================================================================

def train_profit_regressor(
    df: pd.DataFrame,
    target_col: str = 'Operating_profit',
    alpha: float = 1.0,
    model_type: str = 'ridge'
) -> Tuple[Any, StandardScaler, List[str], float, Dict]:
    """
    Operating_profit予測用の回帰モデルを訓練（ML-MIP統合用）

    Args:
        df: データフレーム
        target_col: 目標変数のカラム名
        alpha: 正則化パラメータ（Ridge用）
        model_type: 'ridge' または 'linear'

    Returns:
        model: 訓練済み回帰モデル
        scaler: 特徴量スケーラー
        feature_cols: 使用した特徴量カラム名リスト
        r2_score: R²スコア（交差検証平均）
        metrics: 追加メトリクス
    """
    feature_cols = get_feature_columns(df)

    # 特徴量と目的変数の準備
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # モデル選択
    if model_type == 'ridge':
        model = Ridge(alpha=alpha, random_state=42)
    else:
        model = LinearRegression()

    # 交差検証でスコア計算
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    r2_mean = cv_scores.mean()

    # 全データで訓練
    model.fit(X_scaled, y)

    # 係数の重要度
    coef_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)

    # メトリクス
    metrics = {
        'r2_cv_mean': r2_mean,
        'r2_cv_std': cv_scores.std(),
        'intercept': model.intercept_,
        'n_features': len(feature_cols),
        'n_samples': len(df),
        'top5_features': coef_importance.head(5)['feature'].tolist(),
        'model_type': model_type,
        'alpha': alpha if model_type == 'ridge' else None
    }

    return model, scaler, feature_cols, r2_mean, metrics


def get_model_for_mip(
    df: pd.DataFrame,
    target_col: str = 'Operating_profit',
    alpha: float = 1.0
) -> Dict[str, Any]:
    """
    MIP統合用にモデルと変数情報をまとめて返す

    Args:
        df: データフレーム
        target_col: 目標変数のカラム名
        alpha: 正則化パラメータ

    Returns:
        dict: MIP統合に必要な全情報
            - model: 訓練済みモデル
            - scaler: スケーラー
            - feature_cols: 特徴量名リスト
            - input_bounds: 入力変数の上下限
            - metrics: モデルメトリクス
    """
    # モデル訓練
    model, scaler, feature_cols, r2_score, metrics = train_profit_regressor(
        df, target_col, alpha
    )

    # 特徴量データ
    X = df[feature_cols].fillna(0)

    # 入力変数の上下限を計算（標準化後の値）
    X_scaled = scaler.transform(X)
    margin = 0.2  # 20%のマージン

    lb = X_scaled.min(axis=0) * (1 + margin)
    ub = X_scaled.max(axis=0) * (1 + margin)

    # 上下限を調整（lb < ub を保証）
    for i in range(len(lb)):
        if lb[i] > ub[i]:
            lb[i], ub[i] = ub[i], lb[i]
        if lb[i] == ub[i]:
            lb[i] -= 0.1
            ub[i] += 0.1

    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'input_bounds_lb': lb.tolist(),
        'input_bounds_ub': ub.tolist(),
        'r2_score': r2_score,
        'metrics': metrics,
        'X_original': X,
        'y_original': df[target_col].values
    }
