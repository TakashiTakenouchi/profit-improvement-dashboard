# -*- coding: utf-8 -*-
"""
時系列予測ページ
AutoGluon TimeSeries 予測結果の可視化
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.auth import check_authentication, show_login_form, show_logout_button
from components.data_loader import get_forecast_data_path, get_timeseries_data_path

# ページ設定
st.set_page_config(
    page_title="時系列予測",
    page_icon="📈",
    layout="wide"
)

# カテゴリマッピング
CATEGORY_JP_MAP = {
    'Mens_JACKETS&OUTER2': 'メンズ ジャケット・アウター',
    'Mens_KNIT': 'メンズ ニット',
    'Mens_PANTS': 'メンズ パンツ',
    "WOMEN'S_JACKETS2": 'レディース ジャケット',
    "WOMEN'S_TOPS": 'レディース トップス',
    "WOMEN'S_ONEPIECE": 'レディース ワンピース',
    "WOMEN'S_bottoms": 'レディース ボトムス',
    "WOMEN'S_SCARF & STOLES": 'レディース スカーフ・ストール'
}

SHOP_MAP = {'EBISU': '恵比寿', 'YOKOHAMA': '横浜元町'}


@st.cache_data
def load_forecast_data():
    """予測データを読み込み"""
    forecast_path = get_forecast_data_path()
    if os.path.exists(forecast_path):
        df = pd.read_excel(forecast_path, sheet_name='DailyForecasts')
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # マッピング
        df['ShopCode'] = df['item_id'].str.split('_').str[0]
        df['Shop'] = df['ShopCode'].map(SHOP_MAP)
        df['Category'] = df['CategoryCode'].map(CATEGORY_JP_MAP)

        return df
    return None


@st.cache_data
def load_training_data():
    """学習データを読み込み"""
    training_path = get_timeseries_data_path()
    if os.path.exists(training_path):
        df = pd.read_excel(training_path, sheet_name='DailyForecastData')
        df['Date'] = pd.to_datetime(df['Date'])

        # item_id作成
        shop_code_map = {'恵比寿': 'EBISU', '横浜元町': 'YOKOHAMA'}
        df['ShopCode'] = df['Shop'].map(shop_code_map)
        df['item_id'] = df['ShopCode'] + '_' + df['ItemCode']

        # 日次集計
        df_daily = df.groupby(['item_id', 'Date', 'Shop', 'CategoryCode']).agg({
            'ForecastQuantity': 'sum'
        }).reset_index()
        df_daily = df_daily.rename(columns={'Date': 'timestamp', 'ForecastQuantity': 'actual_quantity'})

        return df_daily
    return None


def create_forecast_chart(df_forecast, df_training, item_id, show_all_training=False):
    """予測グラフを作成"""
    forecast_item = df_forecast[df_forecast['item_id'] == item_id].sort_values('timestamp')
    training_item = df_training[df_training['item_id'] == item_id].sort_values('timestamp') if df_training is not None else None

    if len(forecast_item) == 0:
        return None

    # 表示期間
    if training_item is not None and len(training_item) > 0:
        if show_all_training:
            training_display = training_item
        else:
            training_end = training_item['timestamp'].max()
            training_start = training_end - pd.Timedelta(days=180)
            training_display = training_item[training_item['timestamp'] >= training_start]
    else:
        training_display = None

    fig = go.Figure()

    # 信頼区間
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_item['timestamp'], forecast_item['timestamp'][::-1]]),
        y=pd.concat([forecast_item['0.9'], forecast_item['0.1'][::-1]]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='90%信頼区間'
    ))

    # 実績データ
    if training_display is not None and len(training_display) > 0:
        fig.add_trace(go.Scatter(
            x=training_display['timestamp'],
            y=training_display['actual_quantity'],
            mode='lines',
            name='実績データ',
            line=dict(color='#2ca02c', width=2)
        ))

    # 予測値
    fig.add_trace(go.Scatter(
        x=forecast_item['timestamp'],
        y=forecast_item['predicted_quantity'],
        mode='lines',
        name='予測値 (mean)',
        line=dict(color='#1f77b4', width=3)
    ))

    # 上限・下限
    fig.add_trace(go.Scatter(
        x=forecast_item['timestamp'],
        y=forecast_item['0.9'],
        mode='lines',
        name='上限 (90%)',
        line=dict(color='#d62728', width=1, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=forecast_item['timestamp'],
        y=forecast_item['0.1'],
        mode='lines',
        name='下限 (10%)',
        line=dict(color='#9467bd', width=1, dash='dash')
    ))

    # 予測開始線
    forecast_start = forecast_item['timestamp'].min()
    y_max = forecast_item['0.9'].max()
    y_min = forecast_item['0.1'].min()

    if training_display is not None and len(training_display) > 0:
        y_max = max(y_max, training_display['actual_quantity'].max())
        y_min = min(y_min, training_display['actual_quantity'].min())

    fig.add_trace(go.Scatter(
        x=[forecast_start, forecast_start],
        y=[y_min * 0.9, y_max * 1.1],
        mode='lines',
        name='予測開始',
        line=dict(color='red', width=2, dash='dash')
    ))

    # レイアウト
    shop = forecast_item['Shop'].iloc[0]
    category = forecast_item['Category'].iloc[0]

    fig.update_layout(
        title=dict(text=f"<b>{shop} - {category}</b>", x=0.5),
        xaxis_title="日付",
        yaxis_title="予測数量",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )

    return fig


def main():
    """メイン処理"""
    if not check_authentication():
        show_login_form()
        return

    # ヘッダー
    st.markdown("# 📈 時系列予測ダッシュボード")
    st.markdown("AutoGluon TimeSeries による90日間販売予測を可視化します。")

    # サイドバー
    show_logout_button()

    # モデル情報
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 1rem;">
        <h4>🤖 使用モデル情報</h4>
        <table style="width:100%">
            <tr><td><b>モデル:</b></td><td>WeightedEnsemble</td></tr>
            <tr><td><b>構成:</b></td><td>Chronos2 (57%) + TemporalFusionTransformer (39%) + DirectTabular (4%)</td></tr>
            <tr><td><b>評価指標:</b></td><td>WQL (Weighted Quantile Loss) = -0.3298</td></tr>
            <tr><td><b>予測期間:</b></td><td>2026/01/01 - 2026/03/31 (90日間)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # データ読み込み
    with st.spinner("データを読み込み中..."):
        df_forecast = load_forecast_data()
        df_training = load_training_data()

    if df_forecast is None:
        st.error("予測データが見つかりません。")
        st.info("output/forecast_results_2026_90days.xlsx が必要です。")
        return

    st.success(f"✅ 予測データを読み込みました: {len(df_forecast)}件")

    st.markdown("---")

    # フィルター
    st.sidebar.header("🔧 フィルター設定")

    # 店舗選択
    shops = df_forecast['Shop'].dropna().unique().tolist()
    selected_shop = st.sidebar.selectbox("店舗を選択", shops)

    # カテゴリ選択
    categories = df_forecast[df_forecast['Shop'] == selected_shop]['Category'].dropna().unique().tolist()
    selected_category = st.sidebar.selectbox("カテゴリを選択", categories)

    # アイテム選択
    items = df_forecast[
        (df_forecast['Shop'] == selected_shop) &
        (df_forecast['Category'] == selected_category)
    ]['item_id'].unique().tolist()
    selected_item = st.sidebar.selectbox("アイテムを選択", items)

    # 表示オプション
    st.sidebar.header("📊 表示オプション")
    show_all_training = st.sidebar.checkbox("全学習期間を表示", value=False)

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

        # 統計
        item_forecast = df_forecast[df_forecast['item_id'] == selected_item]
        if len(item_forecast) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("90日間予測合計", f"{item_forecast['predicted_quantity'].sum():.1f}")
            with col2:
                st.metric("日平均予測", f"{item_forecast['predicted_quantity'].mean():.2f}")
            with col3:
                st.metric("最大値 (90%)", f"{item_forecast['0.9'].max():.2f}")
            with col4:
                st.metric("最小値 (10%)", f"{item_forecast['0.1'].min():.2f}")

    with tab2:
        st.subheader(f"{selected_shop} - カテゴリ別予測サマリー")

        # カテゴリ別集計
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

        # フィルタリングされたデータ
        filtered_df = df_forecast[
            (df_forecast['Shop'] == selected_shop) &
            (df_forecast['Category'] == selected_category)
        ][['item_id', 'timestamp', 'predicted_quantity', '0.1', '0.5', '0.9', 'Shop', 'Category']].copy()

        filtered_df.columns = ['item_id', '日付', '予測数量', '下限(10%)', '中央値(50%)', '上限(90%)', '店舗', 'カテゴリ']

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
    <div style="text-align: center; color: #666;">
        <p>AutoGluon TimeSeries 1.5.0 | WeightedEnsemble Model</p>
        <p>次のステップ: 左サイドバーから「6_レポート出力」へ進んでください</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
