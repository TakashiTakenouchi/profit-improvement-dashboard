#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ファイル名: 02_ebisu_store_improvement_20251227.py
作成日: 2025年12月27日
目的: 恵比寿店（shop_code:11）の黒字化率を14.5%から60%以上に改善
      ロジスティック回帰で特定した黒字化要因TOP5をパラメータとして使用

入力:
    - fixed_extended_store_data_2024-FIX_kaizen_monthlyvol3.xlsx: 店舗別損益データ

出力:
    - fixed_extended_store_data_2024-FIX_kaizen_monthlyvol4.xlsx: 改善後データ

黒字化要因TOP5（操作可能パラメータ）:
    1. WOMEN'S_TOPS: オッズ比 4.75
    2. Number_of_guests: オッズ比 1.62
    3. Mens_KNIT: オッズ比 1.42
    4. WOMEN'S_SCARF & STOLES: オッズ比 1.42
    5. Mens_PANTS: オッズ比 1.42
    + personnel_expenses削減: オッズ比 0.25
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
INPUT_FILE = os.path.join(BASE_DIR, "fixed_extended_store_data_2024-FIX_kaizen_monthlyvol3.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "fixed_extended_store_data_2024-FIX_kaizen_monthlyvol4.xlsx")

# 対象店舗
TARGET_SHOP_CODE = 11  # 恵比寿店
TARGET_SURPLUS_RATE = 0.60  # 目標黒字率60%

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
    """Excelファイルを読み込み、不要な列を削除"""
    df = pd.read_excel(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


# =============================================================================
# 恵比寿店の現状分析
# =============================================================================
def analyze_ebisu_store(df, shop_code):
    """恵比寿店の現状を分析"""
    ebisu_mask = df['shop_code'] == shop_code
    ebisu_data = df[ebisu_mask].copy()

    total_months = len(ebisu_data)
    surplus_months = (ebisu_data['Operating_profit'] >= 0).sum()
    deficit_months = (ebisu_data['Operating_profit'] < 0).sum()
    surplus_rate = surplus_months / total_months * 100

    print('=== 恵比寿店の現状 ===')
    print(f'期間: {ebisu_data["Date"].min()} - {ebisu_data["Date"].max()}')
    print(f'総月数: {total_months}')
    print(f'黒字月: {surplus_months}')
    print(f'赤字月: {deficit_months}')
    print(f'黒字率: {surplus_rate:.1f}%')
    print()

    return ebisu_mask, total_months, surplus_rate


# =============================================================================
# 黒字月の特徴値を取得
# =============================================================================
def get_surplus_characteristics(df, ebisu_mask):
    """黒字月の特徴値（平均値・比率）を取得"""
    ebisu_surplus = df[ebisu_mask & (df['Operating_profit'] >= 0)]

    # 黒字月の平均値
    target_total_sales = ebisu_surplus['Total_Sales'].mean()
    target_personnel = ebisu_surplus['personnel_expenses'].mean()
    target_guests = ebisu_surplus['Number_of_guests'].mean()

    # 黒字月の値引き率と仕入れ率
    avg_discount_rate = (ebisu_surplus['discount'] / ebisu_surplus['Total_Sales']).mean()
    avg_purchasing_rate = (ebisu_surplus['purchasing'] / ebisu_surplus['Total_Sales']).mean()

    # 粗利益率
    gross_margin = 1 - avg_discount_rate - avg_purchasing_rate

    print('=== 黒字月の特徴 ===')
    print(f'Total_Sales平均: {target_total_sales:,.0f}')
    print(f'人件費平均: {target_personnel:,.0f}')
    print(f'客数平均: {target_guests:.0f}')
    print(f'値引き率: {avg_discount_rate:.1%}')
    print(f'仕入れ率: {avg_purchasing_rate:.1%}')
    print(f'粗利益率: {gross_margin:.1%}')
    print()

    return {
        'target_total_sales': target_total_sales,
        'target_personnel': target_personnel,
        'target_guests': target_guests,
        'avg_discount_rate': avg_discount_rate,
        'avg_purchasing_rate': avg_purchasing_rate,
        'gross_margin': gross_margin
    }


# =============================================================================
# 改善適用
# =============================================================================
def apply_improvements(df, ebisu_mask, characteristics, target_surplus_rate):
    """
    赤字月に改善を適用して黒字化

    改善戦略:
    1. 売上カテゴリを黒字月水準に引き上げ
    2. 値引き率を黒字月水準に改善
    3. 仕入れ率を黒字月水準に改善
    4. 人件費を黒字月水準に削減
    5. 客数を黒字月水準に増加
    """
    df_improved = df.copy()

    # 全体平均
    mean_profit = df['Operating_profit'].mean()

    # 目標黒字月数
    total_months = ebisu_mask.sum()
    target_surplus = int(np.ceil(total_months * target_surplus_rate))

    # 赤字月を抽出（赤字額が少ない順にソート）
    ebisu_deficit = df[ebisu_mask & (df['Operating_profit'] < 0)].copy()
    ebisu_deficit_sorted = ebisu_deficit.sort_values('Operating_profit', ascending=False)

    print(f'赤字月数: {len(ebisu_deficit_sorted)}')
    print(f'目標黒字月: {target_surplus}')
    print()

    # 各赤字月に改善を適用
    for idx in ebisu_deficit_sorted.index:
        original = df.loc[idx].copy()

        # 現在のoperating_cost
        current_operating_cost = original['operating_cost']

        # 黒字化に必要な最低売上
        min_sales_for_surplus = current_operating_cost / characteristics['gross_margin']

        # 目標売上を設定
        target_sales = max(min_sales_for_surplus, characteristics['target_total_sales'] * 0.9)

        # 売上が足りない場合は引き上げ
        if df_improved.loc[idx, 'Total_Sales'] < target_sales:
            scale = target_sales / df_improved.loc[idx, 'Total_Sales']
            for col in CATEGORY_COLS:
                df_improved.loc[idx, col] = df_improved.loc[idx, col] * scale
            df_improved.loc[idx, 'Total_Sales'] = target_sales

        # discount, purchasingを黒字月の比率で再計算
        df_improved.loc[idx, 'discount'] = (
            df_improved.loc[idx, 'Total_Sales'] * characteristics['avg_discount_rate']
        )
        df_improved.loc[idx, 'purchasing'] = (
            df_improved.loc[idx, 'Total_Sales'] * characteristics['avg_purchasing_rate']
        )

        # gross_profitを再計算
        df_improved.loc[idx, 'gross_profit'] = (
            df_improved.loc[idx, 'Total_Sales'] -
            df_improved.loc[idx, 'purchasing'] -
            df_improved.loc[idx, 'discount']
        )

        # 人件費を黒字月水準に削減
        if df_improved.loc[idx, 'personnel_expenses'] > characteristics['target_personnel']:
            df_improved.loc[idx, 'personnel_expenses'] = characteristics['target_personnel']

        # 客数を更新
        if df_improved.loc[idx, 'Number_of_guests'] < characteristics['target_guests']:
            df_improved.loc[idx, 'Number_of_guests'] = characteristics['target_guests']

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
        current_surplus = (df_improved[ebisu_mask]['Operating_profit'] >= 0).sum()
        if current_surplus >= target_surplus:
            print(f'目標達成! 黒字化月数: {current_surplus}')
            break

    # judge列を再計算
    df_improved['judge'] = (df_improved['Operating_profit'] > mean_profit).astype(int)

    return df_improved


# =============================================================================
# 結果確認
# =============================================================================
def verify_results(df_improved, ebisu_mask):
    """改善結果を確認"""
    ebisu_after = df_improved[ebisu_mask].copy()
    final_surplus = (ebisu_after['Operating_profit'] >= 0).sum()
    final_rate = final_surplus / len(ebisu_after) * 100

    print()
    print('=== 改善後の恵比寿店 ===')
    print(f'黒字月: {final_surplus}/{len(ebisu_after)}')
    print(f'黒字率: {final_rate:.1f}%')
    print()

    if final_rate >= 60:
        print('*** 目標達成! ***')
    else:
        print('目標未達成')

    return final_surplus, final_rate


# =============================================================================
# Excelファイル出力
# =============================================================================
def save_to_excel(df, output_path):
    """結果をExcelファイルに保存"""
    df.to_excel(output_path, index=False)
    print(f'出力完了: {output_path}')


# =============================================================================
# メイン処理
# =============================================================================
def main():
    print('=' * 60)
    print('恵比寿店 黒字化改善プロセス')
    print('=' * 60)
    print()

    # データ読み込み
    print('データを読み込み中...')
    df = load_data(INPUT_FILE)
    print(f'データ形状: {df.shape[0]}行 x {df.shape[1]}列')
    print()

    # 恵比寿店の現状分析
    ebisu_mask = df['shop_code'] == TARGET_SHOP_CODE
    analyze_ebisu_store(df, TARGET_SHOP_CODE)

    # 黒字月の特徴値を取得
    characteristics = get_surplus_characteristics(df, ebisu_mask)

    # 改善適用
    print('改善を適用中...')
    df_improved = apply_improvements(df, ebisu_mask, characteristics, TARGET_SURPLUS_RATE)

    # 結果確認
    final_surplus, final_rate = verify_results(df_improved, ebisu_mask)

    # Excelファイル出力
    save_to_excel(df_improved, OUTPUT_FILE)

    print()
    print('=== 適用した改善戦略（黒字化要因TOP5） ===')
    print("1. WOMEN'S_TOPS（レディーストップス）売上増加 - オッズ比4.75")
    print('2. Number_of_guests（客数）増加 - オッズ比1.62')
    print('3. Mens_KNIT（メンズニット）売上増加 - オッズ比1.42')
    print("4. WOMEN'S_SCARF & STOLES（スカーフ・ストール）売上増加 - オッズ比1.42")
    print('5. Mens_PANTS（メンズパンツ）売上増加 - オッズ比1.42')
    print('+ personnel_expenses（人件費）削減 - オッズ比0.25')

    print()
    print('=' * 60)
    print('処理完了')
    print('=' * 60)

    return df_improved


if __name__ == "__main__":
    df_improved = main()
