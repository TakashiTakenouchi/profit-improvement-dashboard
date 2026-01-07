# -*- coding: utf-8 -*-
"""
AutoGluon TimeSeries 予測結果ダッシュボード
店舗別・アイテム別の予測グラフを信頼区間付きで表示

[更新履歴]
- 2025: 初版作成
- リアルタイム編集確認用のコメント追加
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォント対応
import os
from datetime import datetime

# ページ設定
st.set_page_config(
    page_title="時系列予測ダッシュボード",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# データパス設定
@st.cache_data
def get_data_paths():
    """データファイルパスを取得"""
    # 環境に応じてパスを設定
    base_paths = [
        r"C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\Ensuring consistency between tabular data and time series forecast data",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "."
    ]

    for base in base_paths:
        forecast_path = os.path.join(base, "output", "forecast_results_2026_90days.xlsx")
        training_path = os.path.join(base, "output", "time_series_forecast_data_2024_fixed.xlsx")
        if os.path.exists(forecast_path):
            return forecast_path, training_path

    return None, None

@st.cache_data
def load_forecast_data(filepath):
    """予測データを読み込み"""
    df = pd.read_excel(filepath, sheet_name='DailyForecasts')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 日本語の店舗名・カテゴリ名を修正（文字化け対策）
    shop_map = {'EBISU': '恵比寿', 'YOKOHAMA': '横浜元町'}
    df['ShopCode'] = df['item_id'].str.split('_').str[0]
    df['Shop'] = df['ShopCode'].map(shop_map)

    category_jp_map = {
        'Mens_JACKETS&OUTER2': 'メンズ ジャケット・アウター',
        'Mens_KNIT': 'メンズ ニット',
        'Mens_PANTS': 'メンズ パンツ',
        "WOMEN'S_JACKETS2": 'レディース ジャケット',
        "WOMEN'S_TOPS": 'レディース トップス',
        "WOMEN'S_ONEPIECE": 'レディース ワンピース',
        "WOMEN'S_bottoms": 'レディース ボトムス',
        "WOMEN'S_SCARF & STOLES": 'レディース スカーフ・ストール'
    }
    df['Category'] = df['CategoryCode'].map(category_jp_map)

    return df

@st.cache_data
def load_training_data(filepath):
    """学習データを読み込み"""
    df = pd.read_excel(filepath, sheet_name='DailyForecastData')
    df['Date'] = pd.to_datetime(df['Date'])

    # item_idを作成
    shop_code_map = {'恵比寿': 'EBISU', '横浜元町': 'YOKOHAMA'}
    df['ShopCode'] = df['Shop'].map(shop_code_map)
    df['item_id'] = df['ShopCode'] + '_' + df['ItemCode']

    # 日次集計
    df_daily = df.groupby(['item_id', 'Date', 'Shop', 'CategoryCode']).agg({
        'ForecastQuantity': 'sum'
    }).reset_index()
    df_daily = df_daily.rename(columns={'Date': 'timestamp', 'ForecastQuantity': 'actual_quantity'})

    return df_daily

def create_forecast_chart(df_forecast, df_training, item_id, show_all_training=False):
    """予測グラフを作成"""
    # 選択したitem_idのデータをフィルタ
    forecast_item = df_forecast[df_forecast['item_id'] == item_id].sort_values('timestamp')
    training_item = df_training[df_training['item_id'] == item_id].sort_values('timestamp')

    if len(forecast_item) == 0:
        return None

    # 表示期間を決定
    if show_all_training:
        # 全期間表示
        training_display = training_item
    else:
        # 最後の180日間のみ表示
        training_end = training_item['timestamp'].max()
        training_start = training_end - pd.Timedelta(days=180)
        training_display = training_item[training_item['timestamp'] >= training_start]

    fig = go.Figure()

    # 信頼区間（90%）を塗りつぶし領域で表示
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_item['timestamp'], forecast_item['timestamp'][::-1]]),
        y=pd.concat([forecast_item['0.9'], forecast_item['0.1'][::-1]]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='90%信頼区間 (10%-90%)'
    ))

    # 学習データ（実績）
    fig.add_trace(go.Scatter(
        x=training_display['timestamp'],
        y=training_display['actual_quantity'],
        mode='lines',
        name='実績データ',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='日付: %{x|%Y-%m-%d}<br>実績: %{y:.2f}<extra></extra>'
    ))

    # 予測値（中央値）
    fig.add_trace(go.Scatter(
        x=forecast_item['timestamp'],
        y=forecast_item['predicted_quantity'],
        mode='lines',
        name='予測値 (mean)',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='日付: %{x|%Y-%m-%d}<br>予測: %{y:.2f}<extra></extra>'
    ))

    # 上限・下限の点線
    fig.add_trace(go.Scatter(
        x=forecast_item['timestamp'],
        y=forecast_item['0.9'],
        mode='lines',
        name='上限 (90%)',
        line=dict(color='#d62728', width=1, dash='dash'),
        hovertemplate='日付: %{x|%Y-%m-%d}<br>上限90%: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_item['timestamp'],
        y=forecast_item['0.1'],
        mode='lines',
        name='下限 (10%)',
        line=dict(color='#9467bd', width=1, dash='dash'),
        hovertemplate='日付: %{x|%Y-%m-%d}<br>下限10%: %{y:.2f}<extra></extra>'
    ))

    # 予測開始日の縦線（Scatterで描画してPlotly 6.0互換に）
    forecast_start = forecast_item['timestamp'].min()
    y_max = max(forecast_item['0.9'].max(), training_display['actual_quantity'].max() if len(training_display) > 0 else 0)
    y_min = min(forecast_item['0.1'].min(), training_display['actual_quantity'].min() if len(training_display) > 0 else 0)

    fig.add_trace(go.Scatter(
        x=[forecast_start, forecast_start],
        y=[y_min * 0.9, y_max * 1.1],
        mode='lines',
        name='予測開始',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=True,
        hoverinfo='skip'
    ))

    # レイアウト設定
    shop = forecast_item['Shop'].iloc[0]
    category = forecast_item['Category'].iloc[0]
    item_code = forecast_item['ItemCode'].iloc[0]

    fig.update_layout(
        title=dict(
            text=f"<b>{shop} - {category}</b><br><sub>{item_code}</sub>",
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_title="日付",
        yaxis_title="予測数量",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        height=500,
        template='plotly_white',
        margin=dict(t=100)
    )

    return fig

def create_shop_summary_chart(df_forecast, shop):
    """店舗別サマリーグラフ"""
    shop_data = df_forecast[df_forecast['Shop'] == shop]

    # カテゴリ別日次集計
    daily_by_cat = shop_data.groupby(['timestamp', 'Category']).agg({
        'predicted_quantity': 'sum',
        '0.1': 'sum',
        '0.9': 'sum'
    }).reset_index()

    fig = go.Figure()

    categories = daily_by_cat['Category'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i, cat in enumerate(categories):
        cat_data = daily_by_cat[daily_by_cat['Category'] == cat].sort_values('timestamp')
        fig.add_trace(go.Scatter(
            x=cat_data['timestamp'],
            y=cat_data['predicted_quantity'],
            mode='lines',
            name=cat,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f'{cat}<br>日付: %{{x|%Y-%m-%d}}<br>予測: %{{y:.1f}}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{shop} - カテゴリ別予測推移</b>",
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_title="日付",
        yaxis_title="予測数量合計",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )

    return fig

# メイン処理
def main():
    # ヘッダー
    st.markdown('<div class="main-header">📈 時系列予測ダッシュボード</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AutoGluon TimeSeries による90日間販売予測</div>', unsafe_allow_html=True)

    # モデル情報
    st.markdown("""
    <div class="model-info">
        <h4>🤖 使用モデル情報</h4>
        <table style="width:100%">
            <tr><td><b>モデル:</b></td><td>WeightedEnsemble</td></tr>
            <tr><td><b>構成:</b></td><td>Chronos2 (57%) + TemporalFusionTransformer (39%) + DirectTabular (4%)</td></tr>
            <tr><td><b>評価指標:</b></td><td>WQL (Weighted Quantile Loss) = -0.3298</td></tr>
            <tr><td><b>予測期間:</b></td><td>2026/01/01 - 2026/03/31 (90日間)</td></tr>
            <tr><td><b>共変量:</b></td><td>weekend (土日フラグ)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # データ読み込み
    forecast_path, training_path = get_data_paths()

    if forecast_path is None:
        st.error("データファイルが見つかりません。パスを確認してください。")
        st.info("以下のパスにファイルを配置してください:")
        st.code("output/forecast_results_2026_90days.xlsx\noutput/time_series_forecast_data_2024_fixed.xlsx")
        return

    with st.spinner("データを読み込み中..."):
        df_forecast = load_forecast_data(forecast_path)
        df_training = load_training_data(training_path)

    # サイドバー
    st.sidebar.header("🔧 フィルター設定")

    # 店舗選択
    shops = df_forecast['Shop'].dropna().unique().tolist()
    selected_shop = st.sidebar.selectbox("店舗を選択", shops, index=0)

    # カテゴリ選択
    categories = df_forecast[df_forecast['Shop'] == selected_shop]['Category'].dropna().unique().tolist()
    selected_category = st.sidebar.selectbox("カテゴリを選択", categories, index=0)

    # アイテム選択
    items = df_forecast[
        (df_forecast['Shop'] == selected_shop) &
        (df_forecast['Category'] == selected_category)
    ]['item_id'].unique().tolist()
    selected_item = st.sidebar.selectbox("アイテムを選択", items, index=0)

    # 表示オプション
    st.sidebar.header("📊 表示オプション")
    show_all_training = st.sidebar.checkbox("全学習期間を表示", value=False)
    show_summary = st.sidebar.checkbox("店舗サマリーを表示", value=True)

    # メインコンテンツ
    tab1, tab2, tab3 = st.tabs(["📈 個別予測グラフ", "🏪 店舗サマリー", "📋 データテーブル"])

    with tab1:
        st.subheader(f"選択アイテム: {selected_item}")

        # 予測グラフ
        fig = create_forecast_chart(df_forecast, df_training, selected_item, show_all_training)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("選択したアイテムのデータが見つかりません。")

        # 選択アイテムの統計
        item_forecast = df_forecast[df_forecast['item_id'] == selected_item]
        if len(item_forecast) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("90日間予測合計", f"{item_forecast['predicted_quantity'].sum():.1f}")
            with col2:
                st.metric("日平均予測", f"{item_forecast['predicted_quantity'].mean():.2f}")
            with col3:
                st.metric("最大値 (90%上限)", f"{item_forecast['0.9'].max():.2f}")
            with col4:
                st.metric("最小値 (10%下限)", f"{item_forecast['0.1'].min():.2f}")

    with tab2:
        if show_summary:
            st.subheader(f"{selected_shop} - カテゴリ別予測サマリー")

            # 店舗サマリーグラフ
            fig_summary = create_shop_summary_chart(df_forecast, selected_shop)
            st.plotly_chart(fig_summary, use_container_width=True)

            # カテゴリ別集計テーブル
            shop_summary = df_forecast[df_forecast['Shop'] == selected_shop].groupby('Category').agg({
                'predicted_quantity': ['sum', 'mean'],
                '0.1': 'sum',
                '0.9': 'sum'
            }).round(2)
            shop_summary.columns = ['予測合計', '日平均', '下限合計(10%)', '上限合計(90%)']
            shop_summary = shop_summary.sort_values('予測合計', ascending=False)

            st.dataframe(shop_summary, use_container_width=True)

    with tab3:
        st.subheader("予測データテーブル")

        # フィルタリングされたデータを表示
        filtered_df = df_forecast[
            (df_forecast['Shop'] == selected_shop) &
            (df_forecast['Category'] == selected_category)
        ][['item_id', 'timestamp', 'predicted_quantity', '0.1', '0.5', '0.9', 'Shop', 'Category']].copy()

        filtered_df = filtered_df.rename(columns={
            'predicted_quantity': '予測数量',
            '0.1': '下限(10%)',
            '0.5': '中央値(50%)',
            '0.9': '上限(90%)',
            'Shop': '店舗',
            'Category': 'カテゴリ'
        })

        st.dataframe(filtered_df, use_container_width=True, height=400)

        # CSVダウンロード
        csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSVダウンロード",
            data=csv,
            file_name=f"forecast_{selected_shop}_{selected_category}.csv",
            mime="text/csv"
        )

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>AutoGluon TimeSeries 1.5.0 | WeightedEnsemble Model</p>
        <p>データ期間: 2020/04/30 - 2025/12/31 (学習) | 2026/01/01 - 2026/03/31 (予測)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
