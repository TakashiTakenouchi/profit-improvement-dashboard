# -*- coding: utf-8 -*-
"""
目標設定ページ
営業利益改善目標の設定
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.auth import check_authentication, show_login_form, show_logout_button
from components.data_loader import get_store_options, filter_by_store
from components.charts import create_profit_variance_chart

# ページ設定
st.set_page_config(
    page_title="目標設定",
    page_icon="🎯",
    layout="wide"
)


def main():
    """メイン処理"""
    if not check_authentication():
        show_login_form()
        return

    # ヘッダー
    st.markdown("# 🎯 目標設定")
    st.markdown("営業利益改善の目標パラメータを設定します。")

    # サイドバー
    show_logout_button()

    # データ確認
    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ まず「1_現状把握」ページでデータを読み込んでください。")
        return

    df = st.session_state['uploaded_data'].copy()

    # 日付処理
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month

    st.markdown("---")

    # 設定セクション
    st.markdown("### ⚙️ 最適化パラメータ設定")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 店舗・期間選択")

        # 店舗選択
        stores = get_store_options(df)
        selected_store = st.selectbox("対象店舗", stores, key="target_store")

        # 年選択
        if 'year' in df.columns:
            available_years = sorted(df['year'].unique())
            selected_year = st.selectbox("対象年", available_years, index=len(available_years)-1)
        else:
            selected_year = 2025

        # 月範囲選択
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            start_month = st.selectbox("開始月", list(range(1, 13)), index=3)  # 4月デフォルト
        with col_m2:
            end_month = st.selectbox("終了月", list(range(1, 13)), index=11)  # 12月デフォルト

    with col2:
        st.markdown("#### 最適化目標")

        # セッションから前回の値を取得（ページ遷移時の維持）
        prev_params = st.session_state.get('optimization_params', {})
        default_deficit = prev_params.get('target_deficit_months', 4)
        default_variance = int(prev_params.get('variance_ratio', 0.3) * 100)

        # 赤字月数目標
        target_deficit_months = st.slider(
            "目標赤字月数",
            min_value=0,
            max_value=end_month - start_month + 1,
            value=min(default_deficit, end_month - start_month + 1),
            help="最適化後に許容する赤字月の数"
        )

        # 変動幅
        variance_ratio = st.slider(
            "変動幅 (±%)",
            min_value=10,
            max_value=50,
            value=default_variance,
            step=5,
            help="営業利益の変動許容範囲"
        ) / 100

        # 制約条件
        st.markdown("#### 制約条件")
        st.checkbox("gross_profit（粗利）を固定", value=True, disabled=True)
        st.checkbox("Total_Sales（売上）を固定", value=True, disabled=True)
        st.checkbox("年間operating_cost合計を維持", value=True, disabled=True)

        # v3.0追加: ML-MIP設定
        st.markdown("---")
        st.markdown("#### 🤖 ML-MIP設定（v3.0）")

        # ML-MIPモデルが利用可能かチェック
        mip_model_available = st.session_state.get('mip_model_info') is not None

        if mip_model_available:
            use_mlmip = st.checkbox(
                "ML-MIP最適化を使用",
                value=prev_params.get('use_mlmip', False),
                help="回帰モデルをMIP制約として埋め込み、数学的に整合性のある最適化を実行"
            )

            if use_mlmip:
                solver_options = ["HiGHS", "CBC"]
                default_solver = prev_params.get('solver_type', 'HiGHS')
                solver_idx = solver_options.index(default_solver) if default_solver in solver_options else 0

                solver_type = st.selectbox(
                    "ソルバー選択",
                    solver_options,
                    index=solver_idx,
                    help="HiGHS: 高速（推奨）、CBC: 互換性重視"
                )

                # 回帰モデル情報表示
                mip_info = st.session_state['mip_model_info']
                st.info(f"📊 回帰モデルR²: {mip_info.get('r2_score', 0):.3f} | 特徴量数: {len(mip_info.get('feature_cols', []))}")
            else:
                solver_type = "HiGHS"
        else:
            use_mlmip = False
            solver_type = "HiGHS"
            st.warning("⚠️ ML-MIPを使用するには、先に「2_要因分析」でロジスティック回帰を実行してください。")

    st.markdown("---")

    # 対象データの抽出とプレビュー
    st.markdown("### 📊 対象データプレビュー")

    # フィルタリング
    if 'shop' in df.columns:
        df_store = df[df['shop'] == selected_store].copy()
    else:
        shop_code_map = {'恵比寿': 11, '横浜元町': 12}
        df_store = df[df['shop_code'] == shop_code_map.get(selected_store)].copy()

    # 年月フィルタ
    df_target = df_store[
        (df_store['year'] == selected_year) &
        (df_store['month'] >= start_month) &
        (df_store['month'] <= end_month)
    ].copy()

    if len(df_target) == 0:
        st.warning("⚠️ 選択した条件に該当するデータがありません。")
        return

    # 現状サマリー
    col1, col2, col3, col4 = st.columns(4)

    current_deficit = (df_target['Operating_profit'] < 0).sum()
    current_surplus = (df_target['Operating_profit'] >= 0).sum()

    with col1:
        st.metric("対象月数", f"{len(df_target)}ヶ月")
    with col2:
        st.metric("現在の赤字月", f"{current_deficit}ヶ月",
                  delta=f"{target_deficit_months - current_deficit:+d}" if current_deficit != target_deficit_months else None)
    with col3:
        st.metric("現在の黒字月", f"{current_surplus}ヶ月")
    with col4:
        total_profit = df_target['Operating_profit'].sum()
        st.metric("合計営業利益", f"¥{total_profit:,.0f}")

    # 月別営業利益チャート
    st.markdown("#### 月別営業利益推移")
    fig = create_profit_variance_chart(df_target, f"{selected_store} {selected_year}年 営業利益推移")
    st.plotly_chart(fig, use_container_width=True)

    # データテーブル
    with st.expander("📋 詳細データ", expanded=False):
        display_cols = ['Date', 'month', 'Total_Sales', 'gross_profit', 'operating_cost', 'Operating_profit']
        available_cols = [c for c in display_cols if c in df_target.columns]
        st.dataframe(df_target[available_cols], use_container_width=True)

    st.markdown("---")

    # 設定サマリー
    st.markdown("### 📝 設定サマリー")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.info(f"""
        **対象店舗:** {selected_store}

        **対象期間:** {selected_year}年 {start_month}月 〜 {end_month}月

        **対象月数:** {len(df_target)}ヶ月
        """)

    with summary_col2:
        st.success(f"""
        **目標赤字月数:** {target_deficit_months}ヶ月

        **変動幅:** ±{variance_ratio*100:.0f}%

        **現状 → 目標:** 赤字 {current_deficit}ヶ月 → {target_deficit_months}ヶ月
        """)

    # セッションに保存（v3.0: ML-MIP設定追加）
    st.session_state['optimization_params'] = {
        'store': selected_store,
        'year': selected_year,
        'start_month': start_month,
        'end_month': end_month,
        'target_deficit_months': target_deficit_months,
        'variance_ratio': variance_ratio,
        'target_indices': df_target.index.tolist(),
        'use_mlmip': use_mlmip,
        'solver_type': solver_type
    }

    # 次へボタン
    st.markdown("---")

    if st.button("✅ この設定で最適化を実行する", use_container_width=True, type="primary"):
        st.session_state['ready_for_optimization'] = True
        st.success("✅ 設定を保存しました。「4_最適化実行」ページへ進んでください。")

    # フッター
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>次のステップ: 左サイドバーから「4_最適化実行」へ進んでください</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
