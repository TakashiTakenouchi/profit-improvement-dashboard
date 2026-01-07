# -*- coding: utf-8 -*-
"""
時系列予測データ修正スクリプト
正データに合わせてtime_series_forecast_data_2024.xlsxを修正
"""
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

print('='*60)
print('時系列予測データ修正スクリプト')
print('='*60)

# ファイルパス
fixed_file = 'fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx'
ts_file = 'time_series_forecast_data_2024.xlsx'
output_file = 'time_series_forecast_data_2024_fixed.xlsx'

# ===========================================
# Step 1: データ読み込み
# ===========================================
print('\n[Step 1] データ読み込み')

df_fixed = pd.read_excel(fixed_file)
df_fixed['Date'] = pd.to_datetime(df_fixed['Date'])
df_fixed['YearMonth'] = df_fixed['Date'].dt.strftime('%Y-%m')
print(f'  正データ: {len(df_fixed)}行')

sheets = {}
xl_ts = pd.ExcelFile(ts_file)
for sheet in xl_ts.sheet_names:
    sheets[sheet] = pd.read_excel(ts_file, sheet_name=sheet)
print(f'  時系列データ: {len(sheets)}シート読み込み完了')

# カテゴリーマッピング
category_mapping = {
    'メンズ ジャケット・アウター': 'Mens_JACKETS&OUTER2',
    'メンズ ニット': 'Mens_KNIT',
    'メンズ パンツ': 'Mens_PANTS',
    'レディース ジャケット': "WOMEN'S_JACKETS2",
    'レディース トップス': "WOMEN'S_TOPS",
    'レディース ワンピース': "WOMEN'S_ONEPIECE",
    'レディース ボトムス': "WOMEN'S_bottoms",
    'レディース スカーフ・ストール': "WOMEN'S_SCARF & STOLES"
}
category_mapping_reverse = {v: k for k, v in category_mapping.items()}
categories_en = list(category_mapping.values())

# ===========================================
# Step 2: 日付の+1年シフト
# ===========================================
print('\n[Step 2] 日付の+1年シフト')

# DailyForecastData
df_daily = sheets['DailyForecastData'].copy()
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_daily['Date'] = df_daily['Date'] + pd.DateOffset(years=1)
df_daily['YearMonth'] = df_daily['Date'].dt.strftime('%Y-%m')
df_daily = df_daily[(df_daily['Date'] >= '2020-04-30') & (df_daily['Date'] <= '2025-12-31')]
print(f'  DailyForecastData: {len(df_daily)}行')

# MonthlySummary
df_monthly = sheets['MonthlySummary'].copy()
df_monthly['YearMonth'] = pd.to_datetime(df_monthly['YearMonth'] + '-01') + pd.DateOffset(years=1)
df_monthly['YearMonth'] = df_monthly['YearMonth'].dt.strftime('%Y-%m')
df_monthly = df_monthly[(df_monthly['YearMonth'] >= '2020-04') & (df_monthly['YearMonth'] <= '2025-12')]

# OriginalMonthlyData
df_orig = sheets['OriginalMonthlyData'].copy()
df_orig['YearMonth'] = pd.to_datetime(df_orig['YearMonth'] + '-01') + pd.DateOffset(years=1)
df_orig['YearMonth'] = df_orig['YearMonth'].dt.strftime('%Y-%m')
df_orig = df_orig[(df_orig['YearMonth'] >= '2020-04') & (df_orig['YearMonth'] <= '2025-12')]

# ValidationResults
df_val = sheets['ValidationResults'].copy()
df_val['YearMonth'] = pd.to_datetime(df_val['YearMonth'] + '-01') + pd.DateOffset(years=1)
df_val['YearMonth'] = df_val['YearMonth'].dt.strftime('%Y-%m')
df_val = df_val[(df_val['YearMonth'] >= '2020-04') & (df_val['YearMonth'] <= '2025-12')]

print('  日付シフト完了')

# ===========================================
# Step 3: OriginalMonthlyDataを正データで上書き
# ===========================================
print('\n[Step 3] OriginalMonthlyDataを正データで上書き')

for idx, row in df_orig.iterrows():
    shop = row['shop']
    ym = row['YearMonth']
    fixed_row = df_fixed[(df_fixed['shop'] == shop) & (df_fixed['YearMonth'] == ym)]

    if len(fixed_row) > 0:
        for cat_en in categories_en:
            if cat_en in df_orig.columns:
                df_orig.loc[idx, cat_en] = fixed_row[cat_en].values[0]
        for col in ['Total_Sales', 'gross_profit', 'discount', 'Number_of_guests', 'Price_per_customer']:
            if col in df_orig.columns and col in fixed_row.columns:
                df_orig.loc[idx, col] = fixed_row[col].values[0]

print('  OriginalMonthlyData更新完了')

# ===========================================
# Step 4: DailyForecastDataのForecastQuantity調整
# ===========================================
print('\n[Step 4] DailyForecastDataのForecastQuantity調整')

# 正データから月次売上辞書を作成
monthly_sales_dict = {}
for _, row in df_fixed.iterrows():
    shop = row['shop']
    ym = row['YearMonth']
    for cat_en in categories_en:
        key = (shop, ym, cat_en)
        monthly_sales_dict[key] = row[cat_en]

# 調整係数を計算して適用
adjustment_count = 0
for (shop, ym, cat_code), group_idx in df_daily.groupby(['Shop', 'YearMonth', 'CategoryCode']).groups.items():
    # 正データの月次売上を取得
    target_sales = monthly_sales_dict.get((shop, ym, cat_code), None)

    if target_sales is not None and target_sales > 0:
        # 現在の月次売上（UnitPrice × ForecastQuantity）を計算
        current_calc_sales = (df_daily.loc[group_idx, 'UnitPrice'] * df_daily.loc[group_idx, 'ForecastQuantity']).sum()

        if current_calc_sales > 0:
            # 調整係数
            adjustment_ratio = target_sales / current_calc_sales

            # ForecastQuantityを調整
            df_daily.loc[group_idx, 'ForecastQuantity'] = df_daily.loc[group_idx, 'ForecastQuantity'] * adjustment_ratio

            # ForecastSalesを再計算
            df_daily.loc[group_idx, 'ForecastSales'] = df_daily.loc[group_idx, 'UnitPrice'] * df_daily.loc[group_idx, 'ForecastQuantity']

            # ForecastGrossProfitも再計算
            df_daily.loc[group_idx, 'ForecastGrossProfit'] = df_daily.loc[group_idx, 'ForecastSales'] * df_daily.loc[group_idx, 'GrossMarginRatio']

            adjustment_count += 1

print(f'  {adjustment_count}グループを調整')

# 検証
check_mask = (df_daily['Shop'] == '恵比寿') & (df_daily['YearMonth'] == '2025-12') & (df_daily['CategoryCode'] == 'Mens_JACKETS&OUTER2')
check_sales = df_daily.loc[check_mask, 'ForecastSales'].sum()
print(f'  検証: 恵比寿 2025-12 Mens_JACKETS&OUTER2 ForecastSales合計 = {check_sales:,.2f}')

# ===========================================
# Step 5: 集計シートの再計算
# ===========================================
print('\n[Step 5] 集計シートの再計算')

# MonthlySummaryを再計算
df_monthly_new = df_daily.groupby(['YearMonth', 'Shop', 'Category']).agg({
    'ForecastSales': 'sum',
    'ForecastQuantity': 'sum',
    'ForecastGrossProfit': 'sum'
}).reset_index()
print(f'  MonthlySummary: {len(df_monthly_new)}行')

# ValidationResultsを再計算
df_val_new = []
for shop in ['恵比寿', '横浜元町']:
    for ym in df_daily['YearMonth'].unique():
        for cat_jp, cat_en in category_mapping.items():
            # 正データの売上
            orig_sales = monthly_sales_dict.get((shop, ym, cat_en), 0)

            # 予測売上合計
            mask = (df_daily['Shop'] == shop) & (df_daily['YearMonth'] == ym) & (df_daily['CategoryCode'] == cat_en)
            forecast_sum = df_daily.loc[mask, 'ForecastSales'].sum()

            discrepancy = forecast_sum - orig_sales
            discrepancy_rate = (discrepancy / orig_sales * 100) if orig_sales > 0 else 0

            df_val_new.append({
                'Shop': shop,
                'YearMonth': ym,
                'Category': cat_jp,
                'OriginalSales': orig_sales,
                'ForecastSalesSum': forecast_sum,
                'Discrepancy': discrepancy,
                'DiscrepancyRate': discrepancy_rate
            })

df_val_new = pd.DataFrame(df_val_new)
print(f'  ValidationResults: {len(df_val_new)}行')

# ===========================================
# Step 6: Excelファイル出力
# ===========================================
print('\n[Step 6] Excelファイル出力')

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_daily.to_excel(writer, sheet_name='DailyForecastData', index=False)
    df_monthly_new.to_excel(writer, sheet_name='MonthlySummary', index=False)
    sheets['ItemSummary'].to_excel(writer, sheet_name='ItemSummary', index=False)
    sheets['DistributionAnalysis'].to_excel(writer, sheet_name='DistributionAnalysis', index=False)
    df_orig.to_excel(writer, sheet_name='OriginalMonthlyData', index=False)
    sheets['ProductMaster'].to_excel(writer, sheet_name='ProductMaster', index=False)
    df_val_new.to_excel(writer, sheet_name='ValidationResults', index=False)

print(f'  出力完了: {output_file}')

# ===========================================
# Step 7: 整合性検証
# ===========================================
print('\n[Step 7] 整合性検証')
print('='*60)

# サンプル検証
samples = [
    ('恵比寿', '2025-12', 'Mens_JACKETS&OUTER2'),
    ('恵比寿', '2025-01', 'Mens_JACKETS&OUTER2'),
    ('恵比寿', '2024-12', 'Mens_JACKETS&OUTER2'),
    ('恵比寿', '2025-12', 'Mens_KNIT'),
    ('横浜元町', '2025-12', 'Mens_JACKETS&OUTER2'),
]

print('\n整合性チェック結果:')
all_pass = True
for shop, ym, cat in samples:
    orig = monthly_sales_dict.get((shop, ym, cat), 0)
    mask = (df_daily['Shop'] == shop) & (df_daily['YearMonth'] == ym) & (df_daily['CategoryCode'] == cat)
    forecast = df_daily.loc[mask, 'ForecastSales'].sum()
    diff_pct = abs(forecast - orig) / orig * 100 if orig > 0 else 0
    status = 'PASS' if diff_pct < 0.01 else 'FAIL'
    if diff_pct >= 0.01:
        all_pass = False
    print(f'  [{status}] {shop} {ym} {cat}')
    print(f'         正データ: {orig:,.2f}')
    print(f'         予測合計: {forecast:,.2f}')
    print(f'         差異率: {diff_pct:.4f}%')

print('\n' + '='*60)
if all_pass:
    print('全てのチェックがPASSしました！')
else:
    print('一部チェックがFAILしました')
print('='*60)
print(f'\n修正ファイル: {output_file}')
