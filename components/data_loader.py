# -*- coding: utf-8 -*-
"""
データローダーコンポーネント
Excelファイルのアップロードと読み込みを担当
"""
import streamlit as st
import pandas as pd
import os
from typing import Optional, Tuple


def get_sample_data_path() -> str:
    """サンプルデータのパスを取得"""
    # 相対パスで親ディレクトリのデータを参照
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx")


def get_forecast_data_path() -> str:
    """予測結果データのパスを取得"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "output", "forecast_results_2026_90days.xlsx")


def get_timeseries_data_path() -> str:
    """時系列データのパスを取得"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "output", "time_series_forecast_data_2024_fixed.xlsx")


@st.cache_data
def load_excel_file(uploaded_file) -> Optional[pd.DataFrame]:
    """アップロードされたExcelファイルを読み込み"""
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {str(e)}")
        return None


@st.cache_data
def load_sample_data() -> Optional[pd.DataFrame]:
    """サンプルデータを読み込み"""
    sample_path = get_sample_data_path()
    if os.path.exists(sample_path):
        try:
            df = pd.read_excel(sample_path)
            return df
        except Exception as e:
            st.error(f"サンプルデータ読み込みエラー: {str(e)}")
            return None
    return None


@st.cache_data
def load_forecast_data() -> Optional[pd.DataFrame]:
    """予測結果データを読み込み"""
    forecast_path = get_forecast_data_path()
    if os.path.exists(forecast_path):
        try:
            df = pd.read_excel(forecast_path, sheet_name='DailyForecasts')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            st.error(f"予測データ読み込みエラー: {str(e)}")
            return None
    return None


def show_file_uploader() -> Tuple[Optional[pd.DataFrame], str]:
    """ファイルアップローダーを表示"""
    st.markdown("### 📂 データ読み込み")

    tab1, tab2 = st.tabs(["📤 ファイルアップロード", "📁 サンプルデータ使用"])

    with tab1:
        uploaded_file = st.file_uploader(
            "店舗別損益データ（Excel）をアップロード",
            type=["xlsx", "xls"],
            help="fixed_extended_store_data形式のExcelファイルをアップロードしてください"
        )

        if uploaded_file is not None:
            df = load_excel_file(uploaded_file)
            if df is not None:
                st.success(f"✅ ファイルを読み込みました: {len(df)}行 × {len(df.columns)}列")
                return df, "uploaded"

    with tab2:
        if st.button("📊 サンプルデータを使用", use_container_width=True):
            df = load_sample_data()
            if df is not None:
                st.success(f"✅ サンプルデータを読み込みました: {len(df)}行 × {len(df.columns)}列")
                return df, "sample"
            else:
                st.warning("⚠️ サンプルデータが見つかりません")

    return None, ""


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, list]:
    """データフレームのバリデーション"""
    required_columns = [
        'shop', 'shop_code', 'Date', 'Operating_profit',
        'gross_profit', 'operating_cost', 'Total_Sales'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return False, missing_columns

    return True, []


def get_store_options(df: pd.DataFrame) -> list:
    """店舗選択肢を取得"""
    if 'shop' in df.columns:
        return df['shop'].unique().tolist()
    elif 'shop_code' in df.columns:
        shop_map = {11: '恵比寿', 12: '横浜元町'}
        codes = df['shop_code'].unique().tolist()
        return [shop_map.get(code, f"店舗{code}") for code in codes]
    return []


def filter_by_store(df: pd.DataFrame, store_name: str) -> pd.DataFrame:
    """店舗でデータをフィルタリング"""
    if 'shop' in df.columns:
        return df[df['shop'] == store_name].copy()
    elif 'shop_code' in df.columns:
        shop_code_map = {'恵比寿': 11, '横浜元町': 12}
        code = shop_code_map.get(store_name)
        if code:
            return df[df['shop_code'] == code].copy()
    return df.copy()


def filter_by_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """日付範囲でデータをフィルタリング"""
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        return df[mask].copy()
    return df.copy()
