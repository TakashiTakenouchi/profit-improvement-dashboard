# -*- coding: utf-8 -*-
"""
ロジスティック回帰分析ユーティリティ
営業利益の黒字化要因を分析
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


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
