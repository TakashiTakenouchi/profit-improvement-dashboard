# -*- coding: utf-8 -*-
"""
最適化実行ページ
PuLPによる営業利益最適化
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.auth import check_authentication, show_login_form, show_logout_button
from components.charts import create_before_after_chart
from utils.optimization import (
    run_pulp_optimization, calculate_improvement_metrics, get_monthly_comparison,
    run_mlmip_optimization  # v3.0追加: ML-MIP最適化
)

# ページ設定
st.set_page_config(
    page_title="最適化実行",
    page_icon="⚡",
    layout="wide"
)


def main():
    """メイン処理"""
    if not check_authentication():
        show_login_form()
        return

    # ヘッダー
    st.markdown("# ⚡ 最適化実行")
    st.markdown("PuLPによる数理最適化を実行し、営業利益を改善します。")

    # サイドバー
    show_logout_button()

    # データ・パラメータ確認
    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ まず「1_現状把握」ページでデータを読み込んでください。")
        return

    if 'optimization_params' not in st.session_state:
        st.warning("⚠️ まず「3_目標設定」ページで最適化パラメータを設定してください。")
        return

    df = st.session_state['uploaded_data'].copy()
    params = st.session_state['optimization_params']

    st.markdown("---")

    # パラメータ確認
    st.markdown("### ⚙️ 最適化パラメータ確認")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"""
        **対象店舗:** {params['store']}

        **対象期間:** {params['year']}年 {params['start_month']}月〜{params['end_month']}月
        """)

    with col2:
        st.info(f"""
        **対象月数:** {len(params['target_indices'])}ヶ月

        **目標赤字月数:** {params['target_deficit_months']}ヶ月
        """)

    with col3:
        # v3.0: ML-MIPモード表示
        use_mlmip = params.get('use_mlmip', False)
        solver_type = params.get('solver_type', 'HiGHS')
        mode_str = f"ML-MIP ({solver_type})" if use_mlmip else "従来モード"

        st.info(f"""
        **変動幅:** ±{params['variance_ratio']*100:.0f}%

        **最適化モード:** {mode_str}
        """)

    st.markdown("---")

    # 最適化実行
    st.markdown("### 🚀 最適化実行")

    if st.button("⚡ 最適化を実行", use_container_width=True, type="primary"):
        with st.spinner("最適化を実行中..."):
            progress_bar = st.progress(0)

            try:
                # 最適化実行
                progress_bar.progress(30)

                # v3.0: ML-MIPモードか従来モードかで分岐
                use_mlmip = params.get('use_mlmip', False)
                mip_model_info = st.session_state.get('mip_model_info')

                if use_mlmip and mip_model_info is not None:
                    # ML-MIP最適化
                    df_optimized, summary, mlmip_details = run_mlmip_optimization(
                        df,
                        params['target_indices'],
                        mip_model_info,
                        params['target_deficit_months'],
                        params['variance_ratio'],
                        params.get('solver_type', 'highs').lower()
                    )
                    st.session_state['mlmip_details'] = mlmip_details
                else:
                    # 従来の最適化
                    df_optimized, summary = run_pulp_optimization(
                        df,
                        params['target_indices'],
                        params['target_deficit_months'],
                        params['variance_ratio']
                    )
                    st.session_state['mlmip_details'] = None

                progress_bar.progress(70)

                # メトリクス計算
                metrics = calculate_improvement_metrics(
                    df, df_optimized, params['target_indices']
                )

                progress_bar.progress(100)

                # 結果保存
                st.session_state['optimized_data'] = df_optimized
                st.session_state['optimization_summary'] = summary
                st.session_state['optimization_metrics'] = metrics

                if summary['success']:
                    mode_name = summary.get('optimization_mode', '従来モード')
                    st.success(f"✅ 最適化が完了しました！ （{mode_name}）")

                    # ML-MIP詳細情報表示
                    mlmip_details = st.session_state.get('mlmip_details')
                    if mlmip_details and mlmip_details.get('used_mlmip'):
                        st.info(f"🤖 ML-MIP: ソルバー={mlmip_details['solver']}, "
                               f"解決時間={mlmip_details['solve_time']:.3f}秒, "
                               f"予測誤差={mlmip_details['prediction_error']:.6f}")
                else:
                    st.warning("⚠️ 最適化は完了しましたが、目標に完全には到達できませんでした。")

            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return

    # 結果表示
    if 'optimized_data' in st.session_state:
        df_optimized = st.session_state['optimized_data']
        summary = st.session_state['optimization_summary']
        metrics = st.session_state['optimization_metrics']

        st.markdown("---")
        st.markdown("### 📊 最適化結果")

        # サマリーメトリクス
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "赤字月数",
                f"{metrics['after']['deficit_months']}ヶ月",
                delta=f"{metrics['improvement']['deficit_change']:+d}"
            )

        with col2:
            st.metric(
                "黒字月数",
                f"{metrics['after']['surplus_months']}ヶ月",
                delta=f"{-metrics['improvement']['deficit_change']:+d}"
            )

        with col3:
            st.metric(
                "合計営業利益",
                f"¥{metrics['after']['total_profit']:,.0f}",
                delta=f"¥{metrics['improvement']['profit_change']:,.0f}"
            )

        with col4:
            success_rate = metrics['after']['surplus_months'] / metrics['months_count'] * 100
            st.metric("黒字化率", f"{success_rate:.1f}%")

        # 月別比較チャート
        st.markdown("#### 月別 Operating Profit 比較")

        comparison_df = get_monthly_comparison(df, df_optimized, params['target_indices'])

        before_data = comparison_df['Operating_profit_before'].tolist()
        after_data = comparison_df['Operating_profit_after'].tolist()
        labels = comparison_df['Date'].tolist()

        fig = create_before_after_chart(before_data, after_data, labels, "改善前後の営業利益比較")
        st.plotly_chart(fig, use_container_width=True)

        # 詳細テーブル
        st.markdown("#### 月別詳細")

        comparison_display = comparison_df.copy()
        comparison_display['Operating_profit_before'] = comparison_display['Operating_profit_before'].apply(lambda x: f"¥{x:,.0f}")
        comparison_display['Operating_profit_after'] = comparison_display['Operating_profit_after'].apply(lambda x: f"¥{x:,.0f}")
        comparison_display['change_percent'] = comparison_display['change_percent'].apply(lambda x: f"{x:+.1f}%")

        comparison_display.columns = ['月', '改善前', '改善後', '変化率', 'op_cost前', 'op_cost後', '状態(前)', '状態(後)']

        st.dataframe(comparison_display[['月', '改善前', '改善後', '変化率', '状態(前)', '状態(後)']], use_container_width=True)

        # 統計サマリー
        st.markdown("#### 統計サマリー")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**改善前**")
            st.markdown(f"""
            - 赤字月数: {metrics['before']['deficit_months']}ヶ月
            - 平均営業利益: ¥{metrics['before']['avg_profit']:,.0f}
            - 最小営業利益: ¥{metrics['before']['min_profit']:,.0f}
            - 最大営業利益: ¥{metrics['before']['max_profit']:,.0f}
            """)

        with col2:
            st.markdown("**改善後**")
            st.markdown(f"""
            - 赤字月数: {metrics['after']['deficit_months']}ヶ月
            - 平均営業利益: ¥{metrics['after']['avg_profit']:,.0f}
            - 最小営業利益: ¥{metrics['after']['min_profit']:,.0f}
            - 最大営業利益: ¥{metrics['after']['max_profit']:,.0f}
            """)

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>次のステップ: 左サイドバーから「5_時系列予測」または「6_レポート出力」へ進んでください</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
