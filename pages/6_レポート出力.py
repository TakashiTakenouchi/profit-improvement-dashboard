# -*- coding: utf-8 -*-
"""
レポート出力ページ
改善結果のエクスポート
"""
import streamlit as st
import pandas as pd
import io
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.auth import check_authentication, show_login_form, show_logout_button
from utils.optimization import get_monthly_comparison

# ページ設定
st.set_page_config(
    page_title="レポート出力",
    page_icon="📥",
    layout="wide"
)


def generate_summary_report(params, metrics, logistic_results=None):
    """分析サマリーレポートを生成"""
    report = f"""
# 営業利益改善分析レポート

**生成日時:** {datetime.now().strftime('%Y年%m月%d日 %H:%M')}

---

## 1. 分析概要

- **対象店舗:** {params.get('store', 'N/A')}
- **対象期間:** {params.get('year', 'N/A')}年 {params.get('start_month', 'N/A')}月 〜 {params.get('end_month', 'N/A')}月
- **対象月数:** {metrics.get('months_count', 'N/A')}ヶ月

---

## 2. 最適化結果

### 2.1 改善前後比較

| 項目 | 改善前 | 改善後 | 変化 |
|------|--------|--------|------|
| 赤字月数 | {metrics['before']['deficit_months']}ヶ月 | {metrics['after']['deficit_months']}ヶ月 | {metrics['improvement']['deficit_change']:+d}ヶ月 |
| 黒字月数 | {metrics['before']['surplus_months']}ヶ月 | {metrics['after']['surplus_months']}ヶ月 | {-metrics['improvement']['deficit_change']:+d}ヶ月 |
| 合計営業利益 | ¥{metrics['before']['total_profit']:,.0f} | ¥{metrics['after']['total_profit']:,.0f} | ¥{metrics['improvement']['profit_change']:,.0f} |
| 平均営業利益 | ¥{metrics['before']['avg_profit']:,.0f} | ¥{metrics['after']['avg_profit']:,.0f} | - |

### 2.2 最適化パラメータ

- **目標赤字月数:** {params.get('target_deficit_months', 'N/A')}ヶ月
- **変動幅:** ±{params.get('variance_ratio', 0.3)*100:.0f}%
- **制約条件:** gross_profit固定、年間operating_cost維持

---

## 3. 黒字化要因（ロジスティック回帰）

"""

    if logistic_results is not None:
        top_factors = logistic_results[logistic_results['odds_ratio'] > 1].head(5)
        report += "### TOP5 黒字化要因（オッズ比 > 1）\n\n"
        report += "| 順位 | 変数 | オッズ比 |\n"
        report += "|------|------|----------|\n"
        for i, (_, row) in enumerate(top_factors.iterrows(), 1):
            report += f"| {i} | {row['feature']} | {row['odds_ratio']:.2f} |\n"
    else:
        report += "*ロジスティック回帰結果がありません*\n"

    report += """

---

## 4. 推奨アクション

1. **売上カテゴリの強化:** オッズ比上位の変数を重点的に改善
2. **コスト管理:** 人件費等のコスト項目を適正化
3. **継続モニタリング:** 時系列予測を活用した先行指標管理

---

*このレポートはAI Agents営業利益改善ダッシュボードにより自動生成されました*
"""

    return report


def main():
    """メイン処理"""
    if not check_authentication():
        show_login_form()
        return

    # ヘッダー
    st.markdown("# 📥 レポート出力")
    st.markdown("分析結果と改善データをエクスポートします。")

    # サイドバー
    show_logout_button()

    st.markdown("---")

    # データ確認
    has_optimized = 'optimized_data' in st.session_state
    has_metrics = 'optimization_metrics' in st.session_state
    has_params = 'optimization_params' in st.session_state
    has_logistic = 'logistic_results' in st.session_state

    # ステータス表示
    st.markdown("### 📋 エクスポート可能データ")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if has_optimized:
            st.success("✅ 最適化データ")
        else:
            st.warning("⚠️ 最適化データなし")

    with col2:
        if has_metrics:
            st.success("✅ 改善メトリクス")
        else:
            st.warning("⚠️ メトリクスなし")

    with col3:
        if has_logistic:
            st.success("✅ ロジスティック回帰結果")
        else:
            st.warning("⚠️ 回帰結果なし")

    with col4:
        if 'uploaded_data' in st.session_state:
            st.success("✅ 元データ")
        else:
            st.warning("⚠️ 元データなし")

    if not has_optimized:
        st.info("💡 最適化を実行すると、改善後のデータをダウンロードできます。")
        st.markdown("「3_目標設定」→「4_最適化実行」の順に進んでください。")
        return

    st.markdown("---")

    # エクスポートオプション
    st.markdown("### 📦 エクスポート")

    tab1, tab2, tab3 = st.tabs(["📊 改善P/L", "📄 分析レポート", "📈 詳細データ"])

    with tab1:
        st.markdown("#### 改善後 損益計算書データ")

        df_optimized = st.session_state['optimized_data']
        params = st.session_state['optimization_params']

        # 対象データの抽出
        target_df = df_optimized.loc[params['target_indices']].copy()

        st.dataframe(target_df.head(20), use_container_width=True)
        st.markdown(f"**データサイズ:** {len(target_df)}行")

        # Excel出力
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_optimized.to_excel(writer, sheet_name='全データ', index=False)
            target_df.to_excel(writer, sheet_name='対象期間', index=False)

            # 比較データ
            if has_metrics:
                comparison_df = get_monthly_comparison(
                    st.session_state['uploaded_data'],
                    df_optimized,
                    params['target_indices']
                )
                comparison_df.to_excel(writer, sheet_name='改善前後比較', index=False)

        output.seek(0)

        st.download_button(
            label="📥 改善P/Lをダウンロード (Excel)",
            data=output,
            file_name=f"improved_pl_{params['store']}_{params['year']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with tab2:
        st.markdown("#### 分析サマリーレポート")

        params = st.session_state['optimization_params']
        metrics = st.session_state['optimization_metrics']
        logistic_results = st.session_state.get('logistic_results')

        report = generate_summary_report(params, metrics, logistic_results)

        st.markdown(report)

        st.download_button(
            label="📥 レポートをダウンロード (Markdown)",
            data=report,
            file_name=f"analysis_report_{params['store']}_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )

    with tab3:
        st.markdown("#### 詳細データエクスポート")

        col1, col2 = st.columns(2)

        with col1:
            if has_logistic:
                st.markdown("**ロジスティック回帰結果**")
                logistic_df = st.session_state['logistic_results']
                csv = logistic_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 オッズ比データ (CSV)",
                    data=csv,
                    file_name="logistic_regression_results.csv",
                    mime="text/csv"
                )

        with col2:
            if has_metrics:
                st.markdown("**改善前後比較**")
                params = st.session_state['optimization_params']
                comparison_df = get_monthly_comparison(
                    st.session_state['uploaded_data'],
                    st.session_state['optimized_data'],
                    params['target_indices']
                )
                csv = comparison_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 月別比較データ (CSV)",
                    data=csv,
                    file_name="monthly_comparison.csv",
                    mime="text/csv"
                )

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>営業利益改善AI Agents v1.0</p>
        <p>分析ワークフローが完了しました。お疲れ様でした！</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
