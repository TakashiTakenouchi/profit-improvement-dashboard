# -*- coding: utf-8 -*-
"""
レポート出力ページ
改善結果のエクスポート

【更新履歴】
- 2026-01-07: シナリオ1〜5と推奨アクションの画面表示・ダウンロード機能を追加
- 2026-01-07: Case 1-5のケース別コメント機能を追加
"""
import streamlit as st
import pandas as pd
import io
from datetime import datetime
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# パス設定
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_DIR = BASE_DIR / "output"

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

    st.markdown("---")

    # =====================================================
    # レポート種類の選択
    # =====================================================
    st.markdown("### 📦 レポート選択")

    report_type = st.radio(
        "表示するレポートを選択してください",
        [
            "📄 最適化結果レポート",
            "🎯 シナリオ1〜5と推奨アクション",
            "📊 統計分析サマリーレポート",
            "📈 統計分析詳細レポート（TOP5要因）",
            "📥 データエクスポート"
        ],
        horizontal=False
    )

    st.markdown("---")

    # =====================================================
    # 1. 最適化結果レポート
    # =====================================================
    if report_type == "📄 最適化結果レポート":
        st.markdown("## 📄 最適化結果レポート")

        if not has_optimized:
            st.warning("⚠️ 最適化がまだ実行されていません。")
            st.info("「3_目標設定」→「4_最適化実行」の順に進んでください。")
        else:
            params = st.session_state['optimization_params']
            metrics = st.session_state['optimization_metrics']
            logistic_results = st.session_state.get('logistic_results')

            # レポート生成
            report = generate_summary_report(params, metrics, logistic_results)

            # 表示モード選択
            view_mode = st.radio(
                "表示モード",
                ["📖 画面表示", "📥 ダウンロード"],
                horizontal=True,
                key="opt_view_mode"
            )

            if view_mode == "📖 画面表示":
                st.markdown("---")
                st.markdown(report)
            else:
                st.markdown("---")
                st.markdown("#### ダウンロード")
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📥 レポートをダウンロード (Markdown)",
                        data=report,
                        file_name=f"optimization_report_{params['store']}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

                with col2:
                    # Excel出力
                    df_optimized = st.session_state['optimized_data']
                    target_df = df_optimized.loc[params['target_indices']].copy()

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_optimized.to_excel(writer, sheet_name='全データ', index=False)
                        target_df.to_excel(writer, sheet_name='対象期間', index=False)
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
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

    # =====================================================
    # 2. シナリオ1〜5と推奨アクション
    # =====================================================
    elif report_type == "🎯 シナリオ1〜5と推奨アクション":
        st.markdown("## 🎯 シナリオ1〜5と推奨アクション")
        st.markdown("""
        最適化における5つのシナリオ（Case 1〜5）と各シナリオの推奨アクションを一覧表示します。
        赤字月数に応じた戦略的アプローチを確認できます。
        """)

        # 表示モード選択
        view_mode = st.radio(
            "表示モード",
            ["📖 画面表示", "📥 ダウンロード"],
            horizontal=True,
            key="scenario_view_mode"
        )

        if view_mode == "📖 画面表示":
            st.markdown("---")

            # シナリオ比較表
            st.markdown("### 📋 シナリオ比較一覧")
            scenario_df = pd.DataFrame({
                "シナリオ": ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"],
                "赤字月数": ["0ヶ月", "1ヶ月", "2ヶ月", "3ヶ月", "4ヶ月"],
                "黒字化率": ["100%", "89%", "78%", "67%", "56%"],
                "難易度": ["高", "中〜高", "中", "低〜中", "低"],
                "推奨度": ["理想的だが段階的アプローチ推奨", "達成可能な挑戦目標", "バランス型の現実的目標", "段階的改善の第一歩", "現状維持・ベースライン"]
            })
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # 各シナリオの詳細
            st.markdown("### 📖 シナリオ詳細")

            for deficit_months in range(5):
                case_info = CASE_COMMENTS[deficit_months]
                with st.expander(f"📌 {case_info['title']}", expanded=(deficit_months == 0)):
                    st.markdown(f"**概要:** {case_info['summary']}")
                    st.markdown("---")
                    st.markdown("**詳細分析:**")
                    st.markdown(case_info['analysis'])
                    st.markdown("---")
                    st.info(f"**推奨事項:** {case_info['recommendation']}")

            st.markdown("---")

            # 推奨アクションサマリー
            st.markdown("### 🎯 推奨アクション総合サマリー")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 短期施策（1-3ヶ月）")
                st.markdown("""
                1. **売上カテゴリの強化**
                   - WOMEN'S_JACKETS2の重点販売
                   - 季節に応じた商品展開
                2. **客数増加施策**
                   - イベント企画
                   - SNSマーケティング強化
                3. **コスト見直し**
                   - 赤字月の変動費抑制
                """)

            with col2:
                st.markdown("#### 中長期施策（3-12ヶ月）")
                st.markdown("""
                1. **季節戦略の最適化**
                   - 閑散期（5-7月）の対策強化
                   - 繁忙期（10-12月）の売上最大化
                2. **在庫・人員管理**
                   - 赤字月の在庫最小化
                   - 効率的な人員配置
                3. **継続的改善**
                   - 時系列予測の活用
                   - PDCAサイクルの確立
                """)

        else:
            st.markdown("---")
            st.markdown("#### ダウンロード")

            # シナリオレポートのMarkdown生成
            scenario_report = f"""# シナリオ1〜5と推奨アクション

**生成日時:** {datetime.now().strftime('%Y年%m月%d日 %H:%M')}

---

## シナリオ比較一覧

| シナリオ | 赤字月数 | 黒字化率 | 難易度 | 推奨度 |
|---------|---------|---------|--------|--------|
| Case 1 | 0ヶ月 | 100% | 高 | 理想的だが段階的アプローチ推奨 |
| Case 2 | 1ヶ月 | 89% | 中〜高 | 達成可能な挑戦目標 |
| Case 3 | 2ヶ月 | 78% | 中 | バランス型の現実的目標 |
| Case 4 | 3ヶ月 | 67% | 低〜中 | 段階的改善の第一歩 |
| Case 5 | 4ヶ月 | 56% | 低 | 現状維持・ベースライン |

---

## シナリオ詳細

"""
            # 各シナリオの詳細を追加
            for deficit_months in range(5):
                case_info = CASE_COMMENTS[deficit_months]
                scenario_report += f"""
### {case_info['title']}

**概要:** {case_info['summary']}

{case_info['analysis']}

**推奨事項:** {case_info['recommendation']}

---
"""

            scenario_report += """
## 推奨アクション総合サマリー

### 短期施策（1-3ヶ月）

1. **売上カテゴリの強化**
   - WOMEN'S_JACKETS2の重点販売
   - 季節に応じた商品展開

2. **客数増加施策**
   - イベント企画
   - SNSマーケティング強化

3. **コスト見直し**
   - 赤字月の変動費抑制

### 中長期施策（3-12ヶ月）

1. **季節戦略の最適化**
   - 閑散期（5-7月）の対策強化
   - 繁忙期（10-12月）の売上最大化

2. **在庫・人員管理**
   - 赤字月の在庫最小化
   - 効率的な人員配置

3. **継続的改善**
   - 時系列予測の活用
   - PDCAサイクルの確立

---

*このレポートはAI Agents営業利益改善ダッシュボードにより自動生成されました*
"""

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="📥 シナリオレポートをダウンロード (Markdown)",
                    data=scenario_report,
                    file_name=f"scenario_report_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            with col2:
                # CSVでもダウンロード可能に
                scenario_csv_df = pd.DataFrame({
                    "シナリオ": ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"],
                    "赤字月数": [0, 1, 2, 3, 4],
                    "黒字化率": ["100%", "89%", "78%", "67%", "56%"],
                    "難易度": ["高", "中〜高", "中", "低〜中", "低"],
                    "概要": [CASE_COMMENTS[i]["summary"] for i in range(5)],
                    "推奨事項": [CASE_COMMENTS[i]["recommendation"] for i in range(5)]
                })
                csv_data = scenario_csv_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 シナリオ比較表をダウンロード (CSV)",
                    data=csv_data,
                    file_name=f"scenario_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # =====================================================
    # 3. 統計分析サマリーレポート
    # =====================================================
    elif report_type == "📊 統計分析サマリーレポート":
        st.markdown("## 📊 統計分析サマリーレポート")
        st.markdown("""
        黒字要因TOP5の確率分布特性を分析し、時系列予測モデル選定の考慮事項をまとめたレポートです。
        """)

        # 表示モード選択
        view_mode = st.radio(
            "表示モード",
            ["📖 画面表示", "📥 ダウンロード"],
            horizontal=True,
            key="stat_view_mode"
        )

        report_path = DOCS_DIR / "Top5_Factors_Analysis_Report.md"

        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                statistical_report = f.read()

            if view_mode == "📖 画面表示":
                st.markdown("---")
                # サマリー表示（最初のセクションのみ）
                st.markdown("### 📋 分析概要")

                st.markdown("""
                | 順位 | フィールド | 列番号 | オッズ比 |
                |------|------------|--------|----------|
                | 1位 | WOMEN'S_JACKETS2 | O列 | 最高 |
                | 2位 | Number_of_guests | AA列 | 高 |
                | 3位 | WOMEN'S_ONEPIECE | T列 | 高 |
                | 4位 | Mens_KNIT | P列 | 中 |
                | 5位 | Mens_PANTS | Q列 | 中 |
                """)

                st.markdown("### 📈 分布特性サマリー")
                col1, col2 = st.columns(2)

                with col1:
                    st.info("""
                    **負の二項分布 適用対象:**
                    - Number_of_guests（分散/平均比=226）
                    - Mens_PANTS（分散/平均比=11.2）
                    """)

                with col2:
                    st.info("""
                    **ゼロ過剰ポアソン分布 適用対象:**
                    - WOMEN'S_ONEPIECE（ゼロ率13.4%）
                    - Mens_KNIT
                    - Mens_PANTS
                    """)

                st.markdown("### 🔬 モデル比較結果")
                st.success("""
                **DeepAR-NegBin vs Chronos-Bolt 比較（Mens_PANTS）:**
                - DeepAR-NegBin: WQL = 0.2200 ✓ Best
                - Chronos-Bolt: WQL = 0.2337
                - **結論**: 過分散データには負の二項分布モデルが6.2%優れた精度
                """)

                st.markdown("### 🎯 推奨モデル")
                recommendations = pd.DataFrame({
                    "カテゴリ": ["WOMEN'S_JACKETS2", "Number_of_guests", "WOMEN'S_ONEPIECE", "Mens_KNIT", "Mens_PANTS"],
                    "推奨モデル": ["Chronos2 + TFT", "DeepAR (NegBin)", "Chronos2 + TFT", "Chronos2", "DeepAR (NegBin)"],
                    "理由": ["季節性＋イベント需要", "極度の過分散", "ゼロ過剰＋季節性", "ポアソン分布適合", "過分散＋ゼロ過剰"]
                })
                st.dataframe(recommendations, use_container_width=True, hide_index=True)

            else:
                st.markdown("---")
                st.download_button(
                    label="📥 統計分析レポートをダウンロード (Markdown)",
                    data=statistical_report,
                    file_name="Top5_Factors_Analysis_Report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        else:
            st.warning(f"統計分析レポートが見つかりません: {report_path}")

    # =====================================================
    # 4. 統計分析詳細レポート
    # =====================================================
    elif report_type == "📈 統計分析詳細レポート（TOP5要因）":
        st.markdown("## 📈 統計分析詳細レポート")
        st.markdown("""
        TOP5要因の詳細な確率分布分析、ヒストグラム、モデル比較結果を含む完全版レポートです。
        """)

        # 表示モード選択
        view_mode = st.radio(
            "表示モード",
            ["📖 画面表示", "📥 ダウンロード"],
            horizontal=True,
            key="detail_view_mode"
        )

        report_path = DOCS_DIR / "Top5_Factors_Analysis_Report.md"

        if view_mode == "📖 画面表示":
            st.markdown("---")

            # セクション選択
            section = st.selectbox(
                "表示セクション",
                ["全文表示", "ヒストグラム分析", "確率分布理論", "モデル比較結果"]
            )

            if section == "全文表示":
                if report_path.exists():
                    with open(report_path, "r", encoding="utf-8") as f:
                        full_report = f.read()
                    st.markdown(full_report)
                else:
                    st.warning("レポートファイルが見つかりません")

            elif section == "ヒストグラム分析":
                st.markdown("### 📈 ヒストグラム分析")
                col1, col2 = st.columns(2)

                with col1:
                    monthly_hist = OUTPUT_DIR / "top5_factors_histogram.png"
                    if monthly_hist.exists():
                        st.markdown("**月次データ分布**")
                        st.image(str(monthly_hist), use_container_width=True)
                    else:
                        st.info("月次ヒストグラム画像がありません")

                with col2:
                    daily_hist = OUTPUT_DIR / "daily_quantity_histogram.png"
                    if daily_hist.exists():
                        st.markdown("**日次販売数量分布**")
                        st.image(str(daily_hist), use_container_width=True)
                    else:
                        st.info("日次ヒストグラム画像がありません")

                st.markdown("""
                **分析所見:**
                - 全カテゴリで**分散/平均比 > 1**（過分散）
                - 全カテゴリで**ゼロ率がポアソン期待値を大幅に超過**（ゼロ過剰）
                - 高い歪度・尖度 → 右裾が重い分布
                """)

            elif section == "確率分布理論":
                st.markdown("### 📐 確率分布理論")

                tab_nb, tab_zip = st.tabs(["負の二項分布", "ゼロ過剰ポアソン分布"])

                with tab_nb:
                    st.markdown("""
                    ### 負の二項分布（Negative Binomial Distribution）

                    **適用対象**: Number_of_guests, WOMEN'S_ONEPIECE, Mens_PANTS

                    #### 特徴
                    | 特性 | 説明 |
                    |------|------|
                    | 平均 | μ = r(1-p)/p |
                    | 分散 | σ² = r(1-p)/p² |
                    | **過分散対応** | σ² > μ（分散が平均より大きい場合に適切） |

                    #### Number_of_guestsへの適用理由
                    - 分散/平均比 = **225.95**（極度の過分散）
                    - 客数は「来店イベント」の集積であり、日によってばらつきが大きい
                    """)

                with tab_zip:
                    st.markdown("""
                    ### ゼロ過剰ポアソン分布（Zero-Inflated Poisson, ZIP）

                    **適用対象**: WOMEN'S_ONEPIECE, Mens_KNIT, Mens_PANTS

                    #### 特徴
                    | 特性 | 説明 |
                    |------|------|
                    | 構造的ゼロ | 「売れない日」が存在（店休日、在庫切れ等） |
                    | サンプリングゼロ | たまたま売れなかった日 |

                    #### WOMEN'S_ONEPIECEのゼロ過剰検証結果
                    | 指標 | 値 |
                    |------|-----|
                    | 実際のゼロ率 | 13.4% |
                    | ポアソン期待ゼロ率 | 0.5% |
                    | **ゼロ過剰度** | **13.0%pt** |

                    → ポアソン分布の26倍のゼロが発生 → **ZIPが適切**
                    """)

            elif section == "モデル比較結果":
                st.markdown("### 🔬 AutoGluon-TimeSeries モデル比較結果")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### データ特性分析結果")
                    data_df = pd.DataFrame({
                        "項目": ["データ行数", "ItemCode数", "平均販売数量", "分散", "分散/平均比"],
                        "値": ["12,438行", "3", "4.00個", "44.83", "11.20"],
                        "解釈": ["日次販売データ", "商品SKU", "低頻度販売", "高分散", "極度の過分散"]
                    })
                    st.dataframe(data_df, use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("#### モデル比較結果（WQLスコア）")
                    model_df = pd.DataFrame({
                        "モデル": ["DeepAR-NegBin", "Chronos-Bolt"],
                        "WQL": [0.2200, 0.2337],
                        "訓練時間": ["150.5秒", "1.8秒"],
                        "備考": ["Best Model ✓", "Zero-shot"]
                    })
                    st.dataframe(model_df, use_container_width=True, hide_index=True)

                st.bar_chart(pd.DataFrame({
                    "モデル": ["DeepAR-NegBin", "Chronos-Bolt"],
                    "WQL": [0.2200, 0.2337]
                }).set_index("モデル"))

        else:
            st.markdown("---")
            st.markdown("#### ダウンロード")

            col1, col2, col3 = st.columns(3)

            with col1:
                if report_path.exists():
                    with open(report_path, "r", encoding="utf-8") as f:
                        full_report = f.read()
                    st.download_button(
                        label="📥 フルレポート (MD)",
                        data=full_report,
                        file_name="Top5_Factors_Analysis_Report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

            with col2:
                monthly_hist = OUTPUT_DIR / "top5_factors_histogram.png"
                if monthly_hist.exists():
                    with open(monthly_hist, "rb") as f:
                        st.download_button(
                            label="📥 月次ヒストグラム (PNG)",
                            data=f.read(),
                            file_name="top5_factors_histogram.png",
                            mime="image/png",
                            use_container_width=True
                        )

            with col3:
                daily_hist = OUTPUT_DIR / "daily_quantity_histogram.png"
                if daily_hist.exists():
                    with open(daily_hist, "rb") as f:
                        st.download_button(
                            label="📥 日次ヒストグラム (PNG)",
                            data=f.read(),
                            file_name="daily_quantity_histogram.png",
                            mime="image/png",
                            use_container_width=True
                        )

    # =====================================================
    # 5. データエクスポート
    # =====================================================
    elif report_type == "📥 データエクスポート":
        st.markdown("## 📥 データエクスポート")
        st.markdown("各種データをCSV/Excel形式でダウンロードできます。")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ロジスティック回帰結果")
            if has_logistic:
                logistic_df = st.session_state['logistic_results']
                st.dataframe(logistic_df.head(10), use_container_width=True)
                csv = logistic_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 オッズ比データ (CSV)",
                    data=csv,
                    file_name="logistic_regression_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("ロジスティック回帰を実行してください")

        with col2:
            st.markdown("#### 改善前後比較")
            if has_optimized and has_metrics:
                params = st.session_state['optimization_params']
                comparison_df = get_monthly_comparison(
                    st.session_state['uploaded_data'],
                    st.session_state['optimized_data'],
                    params['target_indices']
                )
                st.dataframe(comparison_df, use_container_width=True)
                csv = comparison_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 月別比較データ (CSV)",
                    data=csv,
                    file_name="monthly_comparison.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("最適化を実行してください")

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
