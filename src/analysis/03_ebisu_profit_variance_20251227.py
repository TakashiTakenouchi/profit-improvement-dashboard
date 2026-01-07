#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ファイル名: 03_ebisu_profit_variance_20251227.py
作成日: 2025年12月27日
目的: 恵比寿店の2025年1月〜12月の営業利益に±30%のランダム分散を適用
      年間合計は保存（約23,513,593円）
      売上、粗利、営業利益のデータ整合性を維持

入力:
    - fixed_extended_store_data_2024-FIX_kaizen_monthlyvol4.xlsx: 改善後データ

出力:
    - fixed_extended_store_data_2024-FIX_kaizen_monthlyvol4.xlsx: 分散適用後データ（上書き）

アルゴリズム:
    1. 係数の生成: 0.7〜1.3の範囲でランダムな数値を12個生成
    2. 正規化: 生成した12個の数値の合計が「12」になるように調整
       （各数値を「12 / 合計」で乗算）
    3. 適用: 元の固定値（1,959,466円）に正規化した各係数を掛け合わせる
    4. 整合性維持: 営業利益の変動に合わせて売上・粗利を逆算調整

    数式: Σ(新利益) = 固定値 × Σ(係数) = 固定値 × 12 = 年間合計

データ整合性:
    - Operating_profit = gross_profit - operating_cost
    - gross_profit = Total_Sales - purchasing - discount
    - 営業利益が変動する場合、gross_profitとTotal_Salesを連動調整
"""

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# 設定
# =============================================================================
BASE_DIR = r"C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\Ensuring consistency between tabular data and time series forecast data"
INPUT_FILE = os.path.join(BASE_DIR, "fixed_extended_store_data_2024-FIX_kaizen_monthlyvol4.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "fixed_extended_store_data_2024-FIX_kaizen_monthlyvol4.xlsx")

# 対象店舗・期間
TARGET_SHOP_CODE = 11  # 恵比寿店
TARGET_YEAR = 2025
VARIANCE_MIN = 0.7  # 最小係数（-30%）
VARIANCE_MAX = 1.3  # 最大係数（+30%）

# 乱数シード（再現性のため）
RANDOM_SEED = 42

# カテゴリ売上列
CATEGORY_COLS = [
    'Mens_JACKETS&OUTER2', 'Mens_KNIT', 'Mens_PANTS',
    "WOMEN'S_JACKETS2", "WOMEN'S_TOPS", "WOMEN'S_ONEPIECE",
    "WOMEN'S_bottoms", "WOMEN'S_SCARF & STOLES"
]


# =============================================================================
# データ読み込み
# =============================================================================
def load_data(file_path):
    """Excelファイルを読み込み"""
    df = pd.read_excel(file_path)
    return df


# =============================================================================
# ランダム係数の生成と正規化
# =============================================================================
def generate_normalized_coefficients(n_months, min_val, max_val, seed=None):
    """
    正規化されたランダム係数を生成

    Parameters:
    -----------
    n_months : int
        生成する係数の数（12）
    min_val : float
        最小係数（0.7）
    max_val : float
        最大係数（1.3）
    seed : int, optional
        乱数シード

    Returns:
    --------
    np.array
        正規化された係数（合計 = n_months）
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: 0.7〜1.3の範囲でランダムな数値を12個生成
    raw_coefficients = np.random.uniform(min_val, max_val, n_months)

    # Step 2: 合計が12になるように正規化
    # 正規化係数 = 12 / Σ(raw_coefficients)
    normalization_factor = n_months / raw_coefficients.sum()
    normalized_coefficients = raw_coefficients * normalization_factor

    return normalized_coefficients


# =============================================================================
# 分散適用（データ整合性維持）
# =============================================================================
def apply_variance_with_consistency(df, shop_code, year, coefficients):
    """
    営業利益に分散を適用し、売上・粗利のデータ整合性を維持

    計算式:
    - Operating_profit = gross_profit - operating_cost
    - gross_profit = Total_Sales - purchasing - discount

    営業利益が変動する場合:
    - 新gross_profit = 新Operating_profit + operating_cost
    - 新Total_Sales = 新gross_profit + purchasing + discount
    - カテゴリ売上も比例配分で調整

    Parameters:
    -----------
    df : pd.DataFrame
        元データ
    shop_code : int
        店舗コード
    year : int
        対象年
    coefficients : np.array
        正規化された係数

    Returns:
    --------
    pd.DataFrame
        分散適用後のデータ
    """
    df_modified = df.copy()

    # 対象レコードのマスク（恵比寿店 & 2025年）
    mask = (df_modified['shop_code'] == shop_code) & (df_modified['Date'].dt.year == year)
    target_indices = df_modified[mask].index

    if len(target_indices) != len(coefficients):
        raise ValueError(f"対象レコード数({len(target_indices)})と係数数({len(coefficients)})が一致しません")

    # 元の固定値を取得
    original_profit = df_modified.loc[target_indices[0], 'Operating_profit']
    original_total = original_profit * len(target_indices)

    print(f'=== 分散適用前 ===')
    print(f'元の営業利益（固定値）: {original_profit:,.0f}円')
    print(f'年間合計: {original_total:,.0f}円')
    print()

    # 各月に分散を適用
    for i, idx in enumerate(target_indices):
        # 元の値を取得
        old_profit = df_modified.loc[idx, 'Operating_profit']
        old_gross = df_modified.loc[idx, 'gross_profit']
        old_total_sales = df_modified.loc[idx, 'Total_Sales']
        old_operating_cost = df_modified.loc[idx, 'operating_cost']
        old_purchasing = df_modified.loc[idx, 'purchasing']
        old_discount = df_modified.loc[idx, 'discount']

        # Step 1: 新しい営業利益を計算
        new_profit = original_profit * coefficients[i]

        # Step 2: 営業利益の変動額を計算
        profit_delta = new_profit - old_profit

        # Step 3: 粗利益を同額増減（operating_costは固定）
        # Operating_profit = gross_profit - operating_cost
        # 営業利益がΔ増えた場合、粗利益もΔ増える
        new_gross = old_gross + profit_delta

        # Step 4: 売上を調整（粗利益率を維持）
        # gross_profit = Total_Sales - purchasing - discount
        # 粗利益率 = gross_profit / Total_Sales
        if old_total_sales > 0:
            old_gross_margin = old_gross / old_total_sales
            # 新売上 = 新粗利益 / 粗利益率
            new_total_sales = new_gross / old_gross_margin
            sales_ratio = new_total_sales / old_total_sales
        else:
            new_total_sales = old_total_sales
            sales_ratio = 1.0

        # Step 5: カテゴリ売上を比例配分で調整
        for col in CATEGORY_COLS:
            df_modified.loc[idx, col] = df_modified.loc[idx, col] * sales_ratio

        # Step 6: purchasing, discountも売上に比例して調整（比率維持）
        new_purchasing = old_purchasing * sales_ratio
        new_discount = old_discount * sales_ratio

        # 値を更新
        df_modified.loc[idx, 'Total_Sales'] = new_total_sales
        df_modified.loc[idx, 'purchasing'] = new_purchasing
        df_modified.loc[idx, 'discount'] = new_discount

        # gross_profitを正しく再計算（整合性確保）
        # gross_profit = Total_Sales - purchasing - discount
        df_modified.loc[idx, 'gross_profit'] = new_total_sales - new_purchasing - new_discount

        # Operating_profitを正しく再計算（整合性確保）
        # Operating_profit = gross_profit - operating_cost
        df_modified.loc[idx, 'Operating_profit'] = df_modified.loc[idx, 'gross_profit'] - old_operating_cost

    return df_modified


# =============================================================================
# 結果検証
# =============================================================================
def verify_results(df, shop_code, year, original_profit):
    """結果を検証（データ整合性含む）"""
    mask = (df['shop_code'] == shop_code) & (df['Date'].dt.year == year)
    target_data = df[mask].copy()

    profit_values = target_data['Operating_profit'].values
    profit_total = profit_values.sum()
    original_total = original_profit * len(profit_values)

    print(f'=== 分散適用後 ===')
    print(f'年間営業利益合計: {profit_total:,.0f}円')
    print(f'差異: {profit_total - original_total:+,.0f}円')
    print()

    print(f'各月の詳細:')
    print('-' * 100)
    print(f'{"日付":10s} {"営業利益":>15s} {"粗利益":>15s} {"売上高":>15s} {"係数":>8s} {"整合性":>10s}')
    print('-' * 100)

    for idx, row in target_data.iterrows():
        date_str = row['Date'].strftime("%Y/%m")
        profit = row['Operating_profit']
        gross = row['gross_profit']
        sales = row['Total_Sales']
        ratio = profit / original_profit

        # 整合性チェック
        # gross_profit = Total_Sales - purchasing - discount
        calc_gross = sales - row['purchasing'] - row['discount']
        # Operating_profit = gross_profit - operating_cost
        calc_profit = gross - row['operating_cost']

        consistency_ok = (abs(calc_gross - gross) < 1) and (abs(calc_profit - profit) < 1)
        consistency_str = "OK" if consistency_ok else "NG"

        print(f'{date_str:10s} {profit:>15,.0f} {gross:>15,.0f} {sales:>15,.0f} {ratio:>8.3f} {consistency_str:>10s}')

    print('-' * 100)
    print()
    print(f'営業利益 最小値: {profit_values.min():,.0f}円 ({profit_values.min()/original_profit:.1%})')
    print(f'営業利益 最大値: {profit_values.max():,.0f}円 ({profit_values.max()/original_profit:.1%})')
    print(f'営業利益 標準偏差: {profit_values.std():,.0f}円')


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print('=' * 60)
    print('恵比寿店 営業利益 ランダム分散適用')
    print('（売上・粗利・営業利益の整合性維持）')
    print('=' * 60)
    print()

    # データ読み込み
    print('データを読み込み中...')
    df = load_data(INPUT_FILE)
    print(f'データ形状: {df.shape[0]}行 x {df.shape[1]}列')
    print()

    # 元の固定値を取得
    mask = (df['shop_code'] == TARGET_SHOP_CODE) & (df['Date'].dt.year == TARGET_YEAR)
    original_profit = df[mask]['Operating_profit'].iloc[0]

    # 係数生成
    print('ランダム係数を生成中...')
    coefficients = generate_normalized_coefficients(
        n_months=12,
        min_val=VARIANCE_MIN,
        max_val=VARIANCE_MAX,
        seed=RANDOM_SEED
    )
    print(f'生成した係数: {np.round(coefficients, 4)}')
    print(f'係数の合計: {coefficients.sum():.6f}（目標: 12.0）')
    print()

    # 分散適用（整合性維持）
    df_modified = apply_variance_with_consistency(
        df, TARGET_SHOP_CODE, TARGET_YEAR, coefficients
    )

    # 結果検証
    verify_results(df_modified, TARGET_SHOP_CODE, TARGET_YEAR, original_profit)

    # Excel出力
    df_modified.to_excel(OUTPUT_FILE, index=False)
    print(f'出力完了: {OUTPUT_FILE}')

    print()
    print('=' * 60)
    print('処理完了')
    print('=' * 60)

    return df_modified


if __name__ == "__main__":
    df_modified = main()
