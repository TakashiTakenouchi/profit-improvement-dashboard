#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ファイル名: 04_complete_improvement_vol5_20251227.py
作成日: 2025年12月27日
目的:
    1. judge列作成（営業利益 > 平均なら1、それ以外は0）
    2. ロジスティック回帰でオッズ比を算出
    3. 黒字化要因TOP5をリストアップ
    4. 恵比寿店（2020/4/30-2025/12/31）を60%黒字化
    5. 営業利益に±30%のバラつきを適用
    6. vol5.xlsxとして出力

入力:
    - fixed_extended_store_data_2024-FIX_kaizen_monthlyvol3.xlsx

出力:
    - fixed_extended_store_data_2024-FIX_kaizen_monthlyvol5.xlsx
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
OUTPUT_FILE = os.path.join(BASE_DIR, "fixed_extended_store_data_2024-FIX_kaizen_monthlyvol5.xlsx")

TARGET_SHOP_CODE = 11  # 恵比寿店
TARGET_SURPLUS_RATE = 0.60  # 目標黒字率60%
VARIANCE_MIN = 0.7  # 最小係数（-30%）
VARIANCE_MAX = 1.3  # 最大係数（+30%）
RANDOM_SEED = 42

CATEGORY_COLS = [
    'Mens_JACKETS&OUTER2', 'Mens_KNIT', 'Mens_PANTS',
    "WOMEN'S_JACKETS2", "WOMEN'S_TOPS", "WOMEN'S_ONEPIECE",
    "WOMEN'S_bottoms", "WOMEN'S_SCARF & STOLES"
]


# =============================================================================
# Step 1: データ読み込みとjudge列作成
# =============================================================================
def load_and_create_judge(file_path):
    """データ読み込みとjudge列の作成"""
    df = pd.read_excel(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # judge列作成（営業利益 > 平均なら1、それ以外は0）
    mean_profit = df['Operating_profit'].mean()
    df['judge'] = (df['Operating_profit'] > mean_profit).astype(int)

    print('=' * 60)
    print('Step 1: judge列の作成')
    print('=' * 60)
    print(f'営業利益の全体平均: {mean_profit:,.0f}円')
    print(f'judge=1（営業利益 > 平均）: {(df["judge"] == 1).sum()}件')
    print(f'judge=0（営業利益 <= 平均）: {(df["judge"] == 0).sum()}件')
    print()

    return df, mean_profit


# =============================================================================
# Step 2: ロジスティック回帰分析
# =============================================================================
def run_logistic_regression(df):
    """ロジスティック回帰でオッズ比を算出"""
    # 説明変数の選定（除外: shop, shop_code, Date, Operating_profit, judge, gross_profit, operating_cost）
    exclude_cols = ['shop', 'shop_code', 'Date', 'Operating_profit', 'judge',
                    'gross_profit', 'operating_cost']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    X = X.fillna(X.mean())
    y = df['judge']

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # L1正則化ロジスティック回帰
    model = LogisticRegression(penalty='l1', solver='saga', max_iter=2000, C=0.5, random_state=42)
    model.fit(X_scaled, y)

    # オッズ比を計算
    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)

    results = pd.DataFrame({
        'variable': feature_cols,
        'coefficient': coefficients,
        'odds_ratio': odds_ratios
    })

    results_nonzero = results[results['coefficient'] != 0].sort_values('odds_ratio', ascending=False)

    print('=' * 60)
    print('Step 2: ロジスティック回帰結果')
    print('=' * 60)
    print(results_nonzero.to_string(index=False))
    print()

    return results_nonzero, feature_cols


# =============================================================================
# Step 3: 黒字化要因TOP5のリストアップ
# =============================================================================
def get_top5_factors(results):
    """黒字化要因TOP5を抽出"""
    positive_factors = results[results['odds_ratio'] > 1].head(5)
    negative_factors = results[results['odds_ratio'] < 1]

    print('=' * 60)
    print('Step 3: 黒字化要因TOP5（オッズ比 > 1）')
    print('=' * 60)
    for i, row in enumerate(positive_factors.itertuples(), 1):
        print(f'{i}. {row.variable}: オッズ比 {row.odds_ratio:.4f}')
    print()

    print('赤字要因（オッズ比 < 1、抑制すべき項目）:')
    for i, row in enumerate(negative_factors.itertuples(), 1):
        print(f'{i}. {row.variable}: オッズ比 {row.odds_ratio:.4f}')
    print()

    return positive_factors, negative_factors


# =============================================================================
# Step 4: 恵比寿店の黒字化改善
# =============================================================================
def improve_ebisu_store(df, target_surplus_rate):
    """恵比寿店の黒字化改善"""
    ebisu_mask = df['shop_code'] == TARGET_SHOP_CODE

    # 現状確認
    ebisu_data = df[ebisu_mask].copy()
    current_surplus = (ebisu_data['Operating_profit'] >= 0).sum()
    current_rate = current_surplus / len(ebisu_data) * 100

    print('=' * 60)
    print('Step 4: 恵比寿店の黒字化改善')
    print('=' * 60)
    print(f'改善前:')
    print(f'  総月数: {len(ebisu_data)}')
    print(f'  黒字月: {current_surplus}')
    print(f'  赤字月: {(ebisu_data["Operating_profit"] < 0).sum()}')
    print(f'  黒字率: {current_rate:.1f}%')
    print()

    # 目標黒字月数
    target_surplus = int(np.ceil(len(ebisu_data) * target_surplus_rate))

    # 黒字月の特徴値を取得
    ebisu_surplus = df[ebisu_mask & (df['Operating_profit'] >= 0)]
    target_total_sales = ebisu_surplus['Total_Sales'].mean()
    target_personnel = ebisu_surplus['personnel_expenses'].mean()
    target_guests = ebisu_surplus['Number_of_guests'].mean()
    avg_discount_rate = (ebisu_surplus['discount'] / ebisu_surplus['Total_Sales']).mean()
    avg_purchasing_rate = (ebisu_surplus['purchasing'] / ebisu_surplus['Total_Sales']).mean()
    gross_margin = 1 - avg_discount_rate - avg_purchasing_rate

    print(f'黒字月の特徴:')
    print(f'  Total_Sales平均: {target_total_sales:,.0f}')
    print(f'  人件費平均: {target_personnel:,.0f}')
    print(f'  客数平均: {target_guests:.0f}')
    print(f'  粗利益率: {gross_margin:.1%}')
    print()

    # 赤字月を改善
    df_improved = df.copy()
    ebisu_deficit = df[ebisu_mask & (df['Operating_profit'] < 0)].copy()
    ebisu_deficit_sorted = ebisu_deficit.sort_values('Operating_profit', ascending=False)

    for idx in ebisu_deficit_sorted.index:
        original = df.loc[idx].copy()
        current_operating_cost = original['operating_cost']

        # 黒字化に必要な最低売上
        min_sales_for_surplus = current_operating_cost / gross_margin
        target_sales = max(min_sales_for_surplus, target_total_sales * 0.9)

        # 売上が足りない場合は引き上げ
        if df_improved.loc[idx, 'Total_Sales'] < target_sales:
            scale = target_sales / df_improved.loc[idx, 'Total_Sales']
            for col in CATEGORY_COLS:
                df_improved.loc[idx, col] = df_improved.loc[idx, col] * scale
            df_improved.loc[idx, 'Total_Sales'] = target_sales

        # discount, purchasingを黒字月の比率で再計算
        df_improved.loc[idx, 'discount'] = df_improved.loc[idx, 'Total_Sales'] * avg_discount_rate
        df_improved.loc[idx, 'purchasing'] = df_improved.loc[idx, 'Total_Sales'] * avg_purchasing_rate

        # gross_profitを再計算
        df_improved.loc[idx, 'gross_profit'] = (
            df_improved.loc[idx, 'Total_Sales'] -
            df_improved.loc[idx, 'purchasing'] -
            df_improved.loc[idx, 'discount']
        )

        # 人件費を黒字月水準に削減
        if df_improved.loc[idx, 'personnel_expenses'] > target_personnel:
            df_improved.loc[idx, 'personnel_expenses'] = target_personnel

        # 客数を更新
        if df_improved.loc[idx, 'Number_of_guests'] < target_guests:
            df_improved.loc[idx, 'Number_of_guests'] = target_guests

        # operating_costを再計算
        df_improved.loc[idx, 'operating_cost'] = (
            df_improved.loc[idx, 'rent'] +
            df_improved.loc[idx, 'personnel_expenses'] +
            df_improved.loc[idx, 'depreciation'] +
            df_improved.loc[idx, 'sales_promotion'] +
            df_improved.loc[idx, 'head_office_expenses']
        )

        # 営業利益を再計算
        df_improved.loc[idx, 'Operating_profit'] = (
            df_improved.loc[idx, 'gross_profit'] -
            df_improved.loc[idx, 'operating_cost']
        )

        # 目標達成チェック
        current_surplus_count = (df_improved[ebisu_mask]['Operating_profit'] >= 0).sum()
        if current_surplus_count >= target_surplus:
            break

    # 改善後確認
    ebisu_after = df_improved[ebisu_mask].copy()
    final_surplus = (ebisu_after['Operating_profit'] >= 0).sum()
    final_rate = final_surplus / len(ebisu_after) * 100

    print(f'改善後:')
    print(f'  黒字月: {final_surplus}')
    print(f'  黒字率: {final_rate:.1f}%')
    print()

    return df_improved


# =============================================================================
# Step 5: 営業利益に±30%のバラつきを適用
# =============================================================================
def apply_variance(df, mean_profit):
    """恵比寿店の2025年データに±30%のバラつきを適用"""
    df_final = df.copy()
    ebisu_mask = df_final['shop_code'] == TARGET_SHOP_CODE

    # 2025年のデータを対象
    mask_2025 = ebisu_mask & (df_final['Date'].dt.year == 2025)
    target_indices = df_final[mask_2025].index

    if len(target_indices) == 0:
        print('2025年のデータがありません')
        return df_final

    # 元の営業利益の平均値（2025年の恵比寿店）
    original_profit = df_final.loc[target_indices[0], 'Operating_profit']

    print('=' * 60)
    print('Step 5: 営業利益に±30%のバラつきを適用')
    print('=' * 60)
    print(f'対象: 恵比寿店 2025年（{len(target_indices)}ヶ月）')
    print(f'元の営業利益: {original_profit:,.0f}円')
    print()

    # ランダム係数を生成（合計が12になるように正規化）
    np.random.seed(RANDOM_SEED)
    raw_coefficients = np.random.uniform(VARIANCE_MIN, VARIANCE_MAX, len(target_indices))
    normalization_factor = len(target_indices) / raw_coefficients.sum()
    coefficients = raw_coefficients * normalization_factor

    print(f'生成した係数: {np.round(coefficients, 4)}')
    print(f'係数の合計: {coefficients.sum():.6f}')
    print()

    # 各月にバラつきを適用
    for i, idx in enumerate(target_indices):
        old_profit = df_final.loc[idx, 'Operating_profit']
        old_gross = df_final.loc[idx, 'gross_profit']
        old_total_sales = df_final.loc[idx, 'Total_Sales']
        old_operating_cost = df_final.loc[idx, 'operating_cost']
        old_purchasing = df_final.loc[idx, 'purchasing']
        old_discount = df_final.loc[idx, 'discount']

        # 新しい営業利益
        new_profit = original_profit * coefficients[i]
        profit_delta = new_profit - old_profit

        # 粗利益を同額増減
        new_gross = old_gross + profit_delta

        # 売上を調整（粗利益率を維持）
        if old_total_sales > 0 and old_gross > 0:
            old_gross_margin = old_gross / old_total_sales
            new_total_sales = new_gross / old_gross_margin
            sales_ratio = new_total_sales / old_total_sales
        else:
            new_total_sales = old_total_sales
            sales_ratio = 1.0

        # カテゴリ売上を比例配分で調整
        for col in CATEGORY_COLS:
            df_final.loc[idx, col] = df_final.loc[idx, col] * sales_ratio

        # purchasing, discountも比例調整
        new_purchasing = old_purchasing * sales_ratio
        new_discount = old_discount * sales_ratio

        # 値を更新
        df_final.loc[idx, 'Total_Sales'] = new_total_sales
        df_final.loc[idx, 'purchasing'] = new_purchasing
        df_final.loc[idx, 'discount'] = new_discount
        df_final.loc[idx, 'gross_profit'] = new_total_sales - new_purchasing - new_discount
        df_final.loc[idx, 'Operating_profit'] = df_final.loc[idx, 'gross_profit'] - old_operating_cost

    # 結果表示
    print('各月の詳細:')
    print('-' * 80)
    for idx in target_indices:
        row = df_final.loc[idx]
        date_str = row['Date'].strftime("%Y/%m")
        profit = row['Operating_profit']
        ratio = profit / original_profit
        print(f'{date_str}: 営業利益 {profit:>12,.0f}円 (係数: {ratio:.3f})')
    print('-' * 80)
    print()

    # judge列を再計算
    df_final['judge'] = (df_final['Operating_profit'] > mean_profit).astype(int)

    return df_final


# =============================================================================
# Step 6: 最終確認と出力
# =============================================================================
def verify_and_save(df, mean_profit, output_path):
    """最終確認とExcel出力"""
    ebisu_mask = df['shop_code'] == TARGET_SHOP_CODE
    ebisu_data = df[ebisu_mask].copy()

    print('=' * 60)
    print('Step 6: 最終確認')
    print('=' * 60)

    # judge=1の割合
    judge_1_count = (ebisu_data['judge'] == 1).sum()
    judge_1_rate = judge_1_count / len(ebisu_data) * 100

    # 実際の黒字月
    surplus_count = (ebisu_data['Operating_profit'] >= 0).sum()
    surplus_rate = surplus_count / len(ebisu_data) * 100

    print(f'恵比寿店（2020/4/30 - 2025/12/31）:')
    print(f'  総月数: {len(ebisu_data)}')
    print(f'  judge=1: {judge_1_count}ヶ月 ({judge_1_rate:.1f}%)')
    print(f'  黒字月（営業利益>=0）: {surplus_count}ヶ月 ({surplus_rate:.1f}%)')
    print()

    # 年度別内訳
    print('年度別内訳:')
    ebisu_data['year'] = ebisu_data['Date'].dt.year
    for year in sorted(ebisu_data['year'].unique()):
        year_data = ebisu_data[ebisu_data['year'] == year]
        j1 = (year_data['judge'] == 1).sum()
        total = len(year_data)
        rate = j1 / total * 100 if total > 0 else 0
        print(f'  {year}年: judge=1 {j1}/{total}ヶ月 ({rate:.1f}%)')
    print()

    # Excel出力
    df.to_excel(output_path, index=False)
    print(f'出力完了: {output_path}')
    print()

    return judge_1_rate >= 60


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print('=' * 60)
    print('店舗別損益計算書 - 完全改善プロセス')
    print('=' * 60)
    print()

    # Step 1: データ読み込みとjudge列作成
    df, mean_profit = load_and_create_judge(INPUT_FILE)

    # Step 2: ロジスティック回帰分析
    results, feature_cols = run_logistic_regression(df)

    # Step 3: 黒字化要因TOP5のリストアップ
    positive_factors, negative_factors = get_top5_factors(results)

    # Step 4: 恵比寿店の黒字化改善
    df_improved = improve_ebisu_store(df, TARGET_SURPLUS_RATE)

    # Step 5: 営業利益に±30%のバラつきを適用
    df_final = apply_variance(df_improved, mean_profit)

    # Step 6: 最終確認と出力
    success = verify_and_save(df_final, mean_profit, OUTPUT_FILE)

    print('=' * 60)
    if success:
        print('処理完了: 目標達成!')
    else:
        print('処理完了: 目標未達成')
    print('=' * 60)

    return df_final


if __name__ == "__main__":
    df_final = main()
