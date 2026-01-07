# -*- coding: utf-8 -*-
"""
要因分析ページ
ロジスティック回帰による黒字化要因分析
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.auth import check_authentication, show_login_form, show_logout_button
from components.data_loader import show_file_uploader, validate_dataframe, get_store_options
from components.charts import create_odds_ratio_chart
from utils.logistic import (
    create_judge_column, run_logistic_regression,
    get_top_factors, get_negative_factors, get_feature_columns,
    get_model_for_mip  # v3.0追加: ML-MIP統合用
)

# ページ設定
st.set_page_config(
    page_title="要因分析",
    page_icon="🔍",
    layout="wide"
)


def main():
    """メイン処理"""
    if not check_authentication():
        show_login_form()
        return

    # ヘッダー
    st.markdown("# 🔍 要因分析（ロジスティック回帰）")
    st.markdown("L1正則化ロジスティック回帰による営業利益の黒字化要因を分析します。")

    # サイドバー
    show_logout_button()

    # データ取得
    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ まず「1_現状把握」ページでデータを読み込んでください。")

        # データ読み込み
        df, source = show_file_uploader()
        if df is not None:
            st.session_state['uploaded_data'] = df
            st.session_state['data_source'] = source
        else:
            return

    df = st.session_state['uploaded_data'].copy()

    st.success(f"✅ データを読み込みました: {len(df)}行")

    st.markdown("---")

    # 店舗選択
    stores = get_store_options(df)
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        selected_store = st.selectbox("店舗を選択", ["全店舗"] + stores)

    with col2:
        regularization_c = st.slider("正則化パラメータ (C)", 0.1, 2.0, 0.5, 0.1,
                                      help="小さいほど強い正則化")

    # データフィルタ
    if selected_store != "全店舗":
        if 'shop' in df.columns:
            df_filtered = df[df['shop'] == selected_store].copy()
        else:
            shop_code_map = {'恵比寿': 11, '横浜元町': 12}
            df_filtered = df[df['shop_code'] == shop_code_map.get(selected_store)].copy()
    else:
        df_filtered = df.copy()

    st.markdown(f"**分析対象:** {len(df_filtered)}件 ({selected_store})")

    # judge列の作成
    df_filtered, mean_profit = create_judge_column(df_filtered)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("平均営業利益", f"¥{mean_profit:,.0f}")
    with col2:
        judge_1_count = (df_filtered['judge'] == 1).sum()
        st.metric("黒字月数 (judge=1)", f"{judge_1_count}件")
    with col3:
        judge_0_count = (df_filtered['judge'] == 0).sum()
        st.metric("赤字月数 (judge=0)", f"{judge_0_count}件")

    st.markdown("---")

    # ロジスティック回帰実行
    if st.button("🚀 ロジスティック回帰を実行", use_container_width=True):
        with st.spinner("分析を実行中..."):
            try:
                results_df, accuracy = run_logistic_regression(df_filtered, C=regularization_c)
                st.session_state['logistic_results'] = results_df
                st.session_state['logistic_accuracy'] = accuracy

                # v3.0追加: ML-MIP統合用の回帰モデルも訓練・保存
                try:
                    mip_model_info = get_model_for_mip(df_filtered)
                    st.session_state['mip_model_info'] = mip_model_info
                    st.session_state['mip_model_error'] = None
                    r2_score = mip_model_info.get('r2_score', 0)
                    st.session_state['analysis_success_msg'] = f"✅ 分析完了！ ロジスティック回帰精度: {accuracy:.1%} | 回帰モデルR²: {r2_score:.3f}"
                except Exception as mip_e:
                    st.session_state['mip_model_info'] = None
                    st.session_state['mip_model_error'] = str(mip_e)
                    st.session_state['analysis_success_msg'] = f"✅ 分析完了！ モデル精度: {accuracy:.1%}"

            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
                return

    # 結果表示
    if 'logistic_results' in st.session_state:
        results_df = st.session_state['logistic_results']
        accuracy = st.session_state['logistic_accuracy']

        # 分析結果メッセージを表示（セッションから）
        if 'analysis_success_msg' in st.session_state:
            st.success(st.session_state['analysis_success_msg'])
        if st.session_state.get('mip_model_error'):
            st.info(f"ℹ️ ML-MIP用モデル訓練スキップ: {st.session_state['mip_model_error']}")

        st.markdown("### 📊 分析結果")

        tab1, tab2, tab3 = st.tabs(["📈 オッズ比チャート", "✅ 黒字化要因", "❌ 赤字要因"])

        with tab1:
            st.markdown("#### オッズ比チャート")
            st.markdown("オッズ比 > 1（緑）は黒字化に貢献、オッズ比 < 1（赤）は赤字要因")

            # 上位・下位を抽出して表示
            top_n = st.slider("表示する変数数", 5, 30, 15)
            display_df = pd.concat([
                results_df.head(top_n // 2),
                results_df.tail(top_n // 2)
            ]).drop_duplicates()

            fig = create_odds_ratio_chart(display_df, "オッズ比分析結果")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("#### ✅ 黒字化要因 TOP5（オッズ比 > 1）")
            st.markdown("これらの変数が増加すると、黒字確率が上昇します。")

            top_factors = get_top_factors(results_df, n=5)
            if len(top_factors) > 0:
                for i, (_, row) in enumerate(top_factors.iterrows(), 1):
                    odds = row['odds_ratio']
                    improvement = (odds - 1) * 100
                    st.markdown(f"""
                    **{i}位: {row['feature']}**
                    - オッズ比: **{odds:.2f}**
                    - 解釈: 1標準偏差増加で黒字確率が **{improvement:+.0f}%** 変化
                    """)
                    st.progress(min(odds / 5, 1.0))

                # セッションに保存
                st.session_state['top_factors'] = top_factors['feature'].tolist()
            else:
                st.info("オッズ比 > 1 の変数が見つかりませんでした")

        with tab3:
            st.markdown("#### ❌ 赤字要因（オッズ比 < 1）")
            st.markdown("これらの変数が増加すると、赤字リスクが上昇します。")

            negative_factors = get_negative_factors(results_df, n=5)
            if len(negative_factors) > 0:
                for i, (_, row) in enumerate(negative_factors.iterrows(), 1):
                    odds = row['odds_ratio']
                    risk = (1 - odds) * 100
                    st.markdown(f"""
                    **{i}位: {row['feature']}**
                    - オッズ比: **{odds:.2f}**
                    - 解釈: 1標準偏差増加で黒字確率が **{-risk:.0f}%** 低下
                    """)
                    st.progress(1 - odds if odds < 1 else 0)
            else:
                st.info("オッズ比 < 1 の変数が見つかりませんでした")

        st.markdown("---")

        # 全結果表示
        with st.expander("📋 全変数のオッズ比一覧", expanded=False):
            st.dataframe(
                results_df.style.background_gradient(
                    subset=['odds_ratio'],
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=2
                ),
                use_container_width=True
            )

            # CSVダウンロード
            csv = results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 結果をCSVダウンロード",
                data=csv,
                file_name=f"logistic_regression_results_{selected_store}.csv",
                mime="text/csv"
            )

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>次のステップ: 左サイドバーから「3_目標設定」へ進んでください</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
