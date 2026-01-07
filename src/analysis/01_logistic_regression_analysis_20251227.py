#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ファイル名: 01_logistic_regression_analysis_20251227.py
作成日: 2025年12月27日
目的: 店舗別損益計算書データに対してロジスティック回帰分析を実行し、
      営業利益の増減要因（黒字化要因TOP5）を明確化

入力:
    - fixed_extended_store_data_2024-FIX_kaizen_monthlyvol3.xlsx: 店舗別損益データ
    - データ項目定義.xlsx: データ項目の定義

出力:
    - コンソール出力: ロジスティック回帰結果（オッズ比）
    - judge列が追加されたデータフレーム
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# 設定
# =============================================================================
BASE_DIR = r"C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\Ensuring consistency between tabular data and time series forecast data"
INPUT_FILE = os.path.join(BASE_DIR, "fixed_extended_store_data_2024-FIX_kaizen_monthlyvol3.xlsx")
DEFINITION_FILE = os.path.join(BASE_DIR, "データ項目定義.xlsx")

# =============================================================================
# データ読み込み
# =============================================================================
def load_data(file_path):
    """Excelファイルを読み込み、不要な列を削除"""
    df = pd.read_excel(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


# =============================================================================
# judge列の作成
# =============================================================================
def create_judge_column(df):
    """
    営業利益が平均値より大きければ1、小さければ0を設定
    """
    mean_profit = df['Operating_profit'].mean()
    df['judge'] = (df['Operating_profit'] > mean_profit).astype(int)

    print('=== judge列の作成 ===')
    print(f'営業利益の平均値: {mean_profit:,.0f}')
    print(f'judge=1（黒字傾向）: {(df["judge"] == 1).sum()}件')
    print(f'judge=0（赤字傾向）: {(df["judge"] == 0).sum()}件')
    print()

    return df, mean_profit


# =============================================================================
# ロジスティック回帰分析
# =============================================================================
def run_logistic_regression(df):
    """
    ロジスティック回帰分析を実行し、オッズ比を算出

    除外変数:
    - shop, shop_code, Date: カテゴリ/時間変数
    - Operating_profit, judge: 目的変数関連
    - gross_profit: 結果変数（Total_Sales, discount, purchasingから算出）
    - operating_cost: 合計値
    """
    # 説明変数の選定
    exclude_cols = ['shop', 'shop_code', 'Date', 'Operating_profit', 'judge',
                    'gross_profit', 'operating_cost']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print('=== 説明変数（操作可能なパラメータ） ===')
    for i, col in enumerate(feature_cols, 1):
        print(f'{i}. {col}')
    print()

    # データ準備
    X = df[feature_cols].copy()
    X = X.fillna(X.mean())
    y = df['judge']

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # L1正則化ロジスティック回帰
    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=2000,
        C=0.5,
        random_state=42
    )
    model.fit(X_scaled, y)

    # オッズ比を計算
    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)

    # 結果をDataFrameにまとめる
    results = pd.DataFrame({
        '変数': feature_cols,
        '係数': coefficients,
        'オッズ比': odds_ratios
    })

    # 係数が0でない変数のみ抽出
    results_nonzero = results[results['係数'] != 0].sort_values('オッズ比', ascending=False)

    return results_nonzero, model, scaler, feature_cols


# =============================================================================
# 結果表示
# =============================================================================
def display_results(results):
    """分析結果を表示"""
    print('=== ロジスティック回帰結果（オッズ比降順） ===')
    print(results.to_string(index=False))
    print()

    # 黒字化要因（オッズ比>1）
    print('=== 黒字化への貢献度 上位5項目（オッズ比>1） ===')
    positive_factors = results[results['オッズ比'] > 1].head(5)
    for i, row in enumerate(positive_factors.itertuples(), 1):
        print(f'{i}. {row.変数}: オッズ比 {row.オッズ比:.4f}')
    print()

    # 赤字要因（オッズ比<1）
    print('=== 赤字要因（オッズ比<1、抑制すべき項目） ===')
    negative_factors = results[results['オッズ比'] < 1]
    for i, row in enumerate(negative_factors.itertuples(), 1):
        print(f'{i}. {row.変数}: オッズ比 {row.オッズ比:.4f}')
    print()

    return positive_factors, negative_factors


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print('=' * 60)
    print('店舗別損益計算書 ロジスティック回帰分析')
    print('=' * 60)
    print()

    # データ読み込み
    print('データを読み込み中...')
    df = load_data(INPUT_FILE)
    print(f'データ形状: {df.shape[0]}行 x {df.shape[1]}列')
    print()

    # judge列の作成
    df, mean_profit = create_judge_column(df)

    # ロジスティック回帰分析
    print('ロジスティック回帰分析を実行中...')
    results, model, scaler, feature_cols = run_logistic_regression(df)

    # 結果表示
    positive_factors, negative_factors = display_results(results)

    print('=' * 60)
    print('分析完了')
    print('=' * 60)

    return df, results, positive_factors, negative_factors


if __name__ == "__main__":
    df, results, positive_factors, negative_factors = main()
