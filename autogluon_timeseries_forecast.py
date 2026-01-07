# -*- coding: utf-8 -*-
"""
AutoGluon TimeSeries 時系列予測スクリプト
恵比寿店・横浜元町店のItemCode毎に2026/01/01から90日間の予測を実行
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print('='*60)
print('AutoGluon TimeSeries 時系列予測')
print('='*60)

# ===========================================
# Step 1: データ読み込みと前処理
# ===========================================
print('\n[Step 1] データ読み込みと前処理')

input_file = 'time_series_forecast_data_2024_fixed.xlsx'
df_daily = pd.read_excel(input_file, sheet_name='DailyForecastData')
df_daily['Date'] = pd.to_datetime(df_daily['Date'])

print(f'  読み込み完了: {len(df_daily)}行')
print(f'  日付範囲: {df_daily["Date"].min().date()} ～ {df_daily["Date"].max().date()}')
print(f'  Shop: {df_daily["Shop"].unique()}')
print(f'  ItemCode数: {df_daily["ItemCode"].nunique()}')

# ===========================================
# Step 2: 予測用データフレーム作成
# ===========================================
print('\n[Step 2] 予測用データフレーム作成')

# item_idを作成（Shop + ItemCode）
df_daily['item_id'] = df_daily['Shop'] + '_' + df_daily['ItemCode']

# weekend列を作成（0:平日、1:土日）
df_daily['weekend'] = df_daily['Date'].dt.dayofweek.isin([5, 6]).astype(int)

# 必要な列のみ抽出
df_ts = df_daily[['item_id', 'Date', 'ForecastQuantity', 'weekend']].copy()
df_ts = df_ts.rename(columns={'Date': 'timestamp', 'ForecastQuantity': 'target'})

# 重複を集約（同じitem_id, timestampで複数行ある場合はtargetを合計）
df_ts = df_ts.groupby(['item_id', 'timestamp', 'weekend']).agg({'target': 'sum'}).reset_index()

print(f'  データフレーム行数: {len(df_ts)}')
print(f'  item_id数: {df_ts["item_id"].nunique()}')
print(f'\nサンプルデータ:')
print(df_ts.head(10).to_string())

# ===========================================
# Step 3: Static Features作成
# ===========================================
print('\n[Step 3] Static Features作成')

# item_idとCategoryのマッピング
static_features = df_daily[['item_id', 'Shop', 'Category', 'CategoryCode']].drop_duplicates()
static_features = static_features.set_index('item_id')

print(f'  Static Features: {len(static_features)}件')
print(f'\nサンプル:')
print(static_features.head(10).to_string())

# ===========================================
# Step 4: 将来の共変量（weekend）を生成
# ===========================================
print('\n[Step 4] 将来の共変量生成（2026/01/01～90日間）')

# 予測期間
forecast_start = pd.Timestamp('2026-01-01')
forecast_end = forecast_start + timedelta(days=89)
forecast_dates = pd.date_range(forecast_start, forecast_end, freq='D')

# 各item_idに対してweekend情報を生成
future_covariates_list = []
for item_id in df_ts['item_id'].unique():
    for date in forecast_dates:
        future_covariates_list.append({
            'item_id': item_id,
            'timestamp': date,
            'weekend': 1 if date.dayofweek in [5, 6] else 0
        })

future_known_covariates = pd.DataFrame(future_covariates_list)
print(f'  将来共変量行数: {len(future_known_covariates)}')
print(f'  予測期間: {forecast_start.date()} ～ {forecast_end.date()}')

# ===========================================
# Step 5: AutoGluon TimeSeries用に変換
# ===========================================
print('\n[Step 5] AutoGluon TimeSeriesDataFrame作成')

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# TimeSeriesDataFrame作成
ts_df = TimeSeriesDataFrame.from_data_frame(
    df_ts,
    id_column='item_id',
    timestamp_column='timestamp'
)

# Static Featuresを設定
ts_df.static_features = static_features

print(f'  TimeSeriesDataFrame作成完了')
print(f'  アイテム数: {ts_df.num_items}')
print(f'  頻度: {ts_df.freq}')

# 将来共変量もTimeSeriesDataFrame形式に
future_covariates_df = TimeSeriesDataFrame.from_data_frame(
    future_known_covariates,
    id_column='item_id',
    timestamp_column='timestamp'
)

print(f'  将来共変量DataFrame作成完了')

# ===========================================
# Step 6: TimeSeriesPredictor学習
# ===========================================
print('\n[Step 6] TimeSeriesPredictor学習開始')
print('  prediction_length=90, eval_metric=WQL, presets=medium_quality')
print('  time_limit=1800秒（30分）')
print('  known_covariates: weekend')
print('-'*60)

predictor = TimeSeriesPredictor(
    prediction_length=90,
    eval_metric="WQL",
    known_covariates_names=["weekend"],
    freq="D",
    path="autogluon_ts_models",
    verbosity=2,
)

predictor.fit(
    ts_df,
    presets="medium_quality",
    time_limit=1800,
)

print('-'*60)
print('学習完了')

# リーダーボード表示
print('\n[モデルリーダーボード]')
leaderboard = predictor.leaderboard()
print(leaderboard.to_string())

# ===========================================
# Step 7: 予測実行
# ===========================================
print('\n[Step 7] 90日間の予測実行')

predictions = predictor.predict(
    ts_df,
    known_covariates=future_covariates_df
)

print(f'  予測完了: {len(predictions)}行')

# ===========================================
# Step 8: 予測結果の出力
# ===========================================
print('\n[Step 8] 予測結果の出力')

# 予測結果をDataFrameに変換
predictions_df = predictions.reset_index()
predictions_df = predictions_df.rename(columns={'mean': 'predicted_quantity'})

# Shop, ItemCode, Categoryを復元
predictions_df['Shop'] = predictions_df['item_id'].str.split('_').str[0]
predictions_df['ItemCode'] = predictions_df['item_id'].str.replace(r'^(恵比寿|横浜元町)_', '', regex=True)

# Categoryを追加
category_map = df_daily[['ItemCode', 'Category', 'CategoryCode']].drop_duplicates().set_index('ItemCode')
predictions_df = predictions_df.merge(
    category_map.reset_index(),
    on='ItemCode',
    how='left'
)

# 出力ファイルに保存
output_file = 'forecast_results_2026_90days.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 詳細予測結果
    predictions_df.to_excel(writer, sheet_name='DailyForecasts', index=False)

    # 月別集計
    predictions_df['YearMonth'] = predictions_df['timestamp'].dt.strftime('%Y-%m')
    monthly_summary = predictions_df.groupby(['Shop', 'YearMonth', 'Category']).agg({
        'predicted_quantity': 'sum',
        '0.1': 'sum',
        '0.9': 'sum'
    }).reset_index()
    monthly_summary.columns = ['Shop', 'YearMonth', 'Category', 'Predicted_Qty', 'Lower_10pct', 'Upper_90pct']
    monthly_summary.to_excel(writer, sheet_name='MonthlySummary', index=False)

    # 店舗・カテゴリー別集計
    category_summary = predictions_df.groupby(['Shop', 'Category']).agg({
        'predicted_quantity': 'sum'
    }).reset_index()
    category_summary.to_excel(writer, sheet_name='CategorySummary', index=False)

print(f'  出力完了: {output_file}')

# ===========================================
# Step 9: 結果サマリー表示
# ===========================================
print('\n[Step 9] 予測結果サマリー')
print('='*60)

# 店舗別サマリー
print('\n【店舗別予測数量（90日間合計）】')
shop_summary = predictions_df.groupby('Shop')['predicted_quantity'].sum()
for shop, qty in shop_summary.items():
    print(f'  {shop}: {qty:,.0f} 個')

# カテゴリー別サマリー
print('\n【カテゴリー別予測数量（全店舗90日間合計）】')
cat_summary = predictions_df.groupby('Category')['predicted_quantity'].sum().sort_values(ascending=False)
for cat, qty in cat_summary.items():
    print(f'  {cat}: {qty:,.0f} 個')

print('\n' + '='*60)
print('予測完了')
print('='*60)
print(f'\n出力ファイル:')
print(f'  - {output_file}')
print(f'  - autogluon_ts_models/ (学習済みモデル)')
