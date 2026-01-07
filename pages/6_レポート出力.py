# -*- coding: utf-8 -*-
"""
レポート出力ページ
改善結果のエクスポート

【更新履歴】
- 2026-01-07: Case 1-5のケース別コメント機能を追加
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


# ============================================
# Case別コメント定義（赤字月数に応じた分析コメント）
# ============================================
CASE_COMMENTS = {
    0: {
        "title": "Case 1: 全月黒字化シナリオ（赤字0ヶ月）",
        "summary": "全ての月で黒字を達成する最も積極的な改善シナリオです。",
        "analysis": """
**シナリオ概要:**
- 対象期間の全月で営業利益がプラスになるよう最適化
- 売上カテゴリ（WOMEN'S_JACKETS2、WOMEN'S_ONEPIECE、Mens_KNIT、Mens_PANTS）を増加させることで粗利を確保

**達成のための施策:**
1. **売上強化策**: オッズ比上位カテゴリの販促強化、特にWOMEN'S_JACKETS2の重点販売
2. **客数増加策**: Number_of_guests増加のためのイベント企画、SNSマーケティング強化
3. **コスト最適化**: 変動費の抑制、効率的な人員配置

**リスク要因:**
- 季節要因による売上変動（5-7月の閑散期対策が必要）
- 過度な販促によるマージン低下の可能性
- 現実的には達成難易度が高いため、段階的なアプローチを推奨
""",
        "recommendation": "このシナリオは理想的ですが、現実的には段階的に赤字月を減らすアプローチを推奨します。"
    },
    1: {
        "title": "Case 2: 赤字1ヶ月シナリオ",
        "summary": "年間で1ヶ月のみ赤字を許容する現実的な改善シナリオです。",
        "analysis": """
**シナリオ概要:**
- 9ヶ月中8ヶ月で黒字、1ヶ月で赤字を許容
- 最も営業利益が低い月（通常6月）を赤字月として設定

**達成のための施策:**
1. **重点月の特定**: 6月を戦略的撤退月として位置づけ、コスト最小化
2. **黒字月の強化**: 4月、8月、12月の繁忙期を最大限活用
3. **在庫戦略**: 赤字月の在庫を最小化し、キャッシュフロー改善

**リスク要因:**
- 赤字月の設定が適切でない場合、連鎖的な影響の可能性
- 従業員モチベーションへの配慮が必要

**年間収支への影響:**
- 年間Operating_profit合計は維持（約1,700万円）
- 月次のバラつきを戦略的にコントロール
""",
        "recommendation": "現状の4ヶ月赤字から大幅な改善となり、達成可能性の高い目標です。"
    },
    2: {
        "title": "Case 3: 赤字2ヶ月シナリオ",
        "summary": "年間で2ヶ月の赤字を許容するバランス型シナリオです。",
        "analysis": """
**シナリオ概要:**
- 9ヶ月中7ヶ月で黒字、2ヶ月で赤字を許容
- 通常、5月・6月または6月・7月の連続月を赤字月として設定

**達成のための施策:**
1. **閑散期対策**: 5-7月の売上低下を前提としたコスト構造の見直し
2. **繁忙期最大化**: 春（4月）と秋冬（10-12月）の売上最大化
3. **セール戦略**: 赤字月のセール実施で在庫回転率向上

**リスク要因:**
- 連続赤字月の場合、キャッシュフロー管理が重要
- スタッフのシフト調整による人件費最適化

**年間収支への影響:**
- 年間Operating_profit合計は維持
- 赤字月の損失を黒字月で十分にカバー可能
""",
        "recommendation": "現実的かつ達成可能性の高い目標設定です。段階的な改善の中間目標として適切。"
    },
    3: {
        "title": "Case 4: 赤字3ヶ月シナリオ",
        "summary": "年間で3ヶ月の赤字を許容する安定志向シナリオです。",
        "analysis": """
**シナリオ概要:**
- 9ヶ月中6ヶ月で黒字、3ヶ月で赤字を許容
- 現状（4ヶ月赤字）から1ヶ月改善の段階的アプローチ

**達成のための施策:**
1. **段階的改善**: 最も改善しやすい月から着手（例：7月の黒字化）
2. **コスト構造の見直し**: 固定費の削減余地を検討
3. **売上カテゴリの集中**: オッズ比上位カテゴリへのリソース集中

**リスク要因:**
- 改善幅が小さいため、外部要因の影響を受けやすい
- 継続的なモニタリングが必要

**年間収支への影響:**
- 年間Operating_profit合計は維持
- 黒字化率: 66.7%（6/9ヶ月）
""",
        "recommendation": "現状からの第一歩として最適な目標設定。確実に達成し、次の段階へ進むことを推奨。"
    },
    4: {
        "title": "Case 5: 赤字4ヶ月シナリオ（現状維持）",
        "summary": "現状の赤字月数を維持しながら、月次バラつきを最適化するシナリオです。",
        "analysis": """
**シナリオ概要:**
- 9ヶ月中5ヶ月で黒字、4ヶ月で赤字（現状同等）
- 月次の営業利益バラつきを±30%の範囲で最適化

**このシナリオの意義:**
1. **ベースライン確立**: 現状の収益構造を理解し、改善の基準点を設定
2. **変動幅の把握**: 各月のOperating_profitの変動可能範囲を確認
3. **感度分析**: 売上カテゴリの変化が利益に与える影響を分析

**現状の課題:**
- 赤字月（5月、6月、7月、11月）の特徴分析が必要
- 閑散期と繁忙期の格差が大きい

**年間収支への影響:**
- 年間Operating_profit合計: 約1,700万円（維持）
- 黒字化率: 55.6%（5/9ヶ月）
""",
        "recommendation": "まずは現状を正確に把握し、データに基づいた改善計画を立案することを推奨します。"
    }
}


def get_case_comment(target_deficit_months: int) -> dict:
    """
    赤字月数に応じたケースコメントを取得

    Args:
        target_deficit_months: 目標赤字月数（0-4）

    Returns:
        ケースコメント辞書
    """
    if target_deficit_months in CASE_COMMENTS:
        return CASE_COMMENTS[target_deficit_months]
    else:
        # 5以上の場合は汎用コメント
        return {
            "title": f"Case: 赤字{target_deficit_months}ヶ月シナリオ",
            "summary": f"年間で{target_deficit_months}ヶ月の赤字を許容するシナリオです。",
            "analysis": "詳細な分析コメントは個別に検討が必要です。",
            "recommendation": "目標赤字月数に応じた施策を検討してください。"
        }


def generate_summary_report(params, metrics, logistic_results=None):
    """分析サマリーレポートを生成（ケース別コメント付き）"""

    # ケース別コメントを取得
    target_deficit = params.get('target_deficit_months', 4)
    case_comment = get_case_comment(target_deficit)

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
- **制約条件:** gross_profit調整、年間Operating_profit合計維持

---

## 3. シナリオ分析

### {case_comment['title']}

**概要:** {case_comment['summary']}

{case_comment['analysis']}

**推奨事項:** {case_comment['recommendation']}

---

## 4. 黒字化要因（ロジスティック回帰）

"""

    if logistic_results is not None:
        top_factors = logistic_results[logistic_results['odds_ratio'] > 1].head(5)
        report += "### TOP5 黒字化要因（オッズ比 > 1）\n\n"
        report += "| 順位 | 変数 | オッズ比 | 解釈 |\n"
        report += "|------|------|----------|------|\n"

        interpretations = {
            "WOMEN'S_JACKETS2": "最重要カテゴリ。販促強化で黒字化に大きく寄与",
            "Number_of_guests": "客数増加が黒字化の鍵。集客施策を強化",
            "WOMEN'S_ONEPIECE": "季節商品。春夏シーズンの販売強化",
            "Mens_KNIT": "秋冬商品。10-12月の販売強化",
            "Mens_PANTS": "定番商品。通年で安定した販売が可能"
        }

        for i, (_, row) in enumerate(top_factors.iterrows(), 1):
            interp = interpretations.get(row['feature'], "黒字化に寄与する要因")
            report += f"| {i} | {row['feature']} | {row['odds_ratio']:.2f} | {interp} |\n"
    else:
        report += "*ロジスティック回帰結果がありません*\n"

    report += """

---

## 5. 推奨アクション

### 短期施策（1-3ヶ月）
1. **売上カテゴリの強化:** オッズ比1位のWOMEN'S_JACKETS2を重点販売
2. **客数増加施策:** イベント企画、SNSマーケティング強化
3. **コスト見直し:** 赤字月の変動費抑制

### 中期施策（3-6ヶ月）
1. **季節戦略の最適化:** 閑散期（5-7月）の対策強化
2. **在庫管理改善:** 赤字月の在庫最小化
3. **人員配置最適化:** 繁忙期・閑散期のシフト調整

### 長期施策（6-12ヶ月）
1. **継続モニタリング:** 時系列予測を活用した先行指標管理
2. **段階的目標設定:** 赤字月数を段階的に削減
3. **データ駆動経営:** 定期的な分析とPDCAサイクルの確立

---

## 6. ケース別シナリオ比較

| ケース | 赤字月数 | 黒字化率 | 難易度 | 推奨度 |
|--------|---------|---------|--------|--------|
| Case 1 | 0ヶ月 | 100% | 高 | 理想的だが段階的アプローチ推奨 |
| Case 2 | 1ヶ月 | 89% | 中〜高 | 達成可能な挑戦目標 |
| Case 3 | 2ヶ月 | 78% | 中 | バランス型の現実的目標 |
| Case 4 | 3ヶ月 | 67% | 低〜中 | 段階的改善の第一歩 |
| Case 5 | 4ヶ月 | 56% | 低 | 現状維持・ベースライン |

---

*このレポートはAI Agents営業利益改善ダッシュボードにより自動生成されました*

*最適化アルゴリズム: 要因ベース最適化（v2.0）*
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
