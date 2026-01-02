# -*- coding: utf-8 -*-
"""
7_統計分析レポート.py
黒字要因TOP5の統計分析・時系列予測モデル選定レポート表示ページ

Author: Takashi.Takenouchi
"""

import streamlit as st
import os
from pathlib import Path

# ページ設定
st.set_page_config(
    page_title="統計分析レポート",
    page_icon="📊",
    layout="wide"
)

st.title("📊 黒字要因TOP5 統計分析レポート")
st.markdown("---")

# パス設定
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_DIR = BASE_DIR / "output"

# サイドバー
st.sidebar.header("レポートセクション")
section = st.sidebar.radio(
    "表示セクション",
    [
        "概要",
        "ヒストグラム分析",
        "確率分布理論",
        "モデル比較結果",
        "推奨アクション",
        "フルレポート"
    ]
)

# ヒストグラム表示関数
def show_histograms():
    st.subheader("📈 ヒストグラム分析")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 月次データ分布")
        monthly_hist = OUTPUT_DIR / "top5_factors_histogram.png"
        if monthly_hist.exists():
            st.image(str(monthly_hist), use_container_width=True)
        else:
            st.warning(f"画像が見つかりません: {monthly_hist}")

    with col2:
        st.markdown("### 日次販売数量分布")
        daily_hist = OUTPUT_DIR / "daily_quantity_histogram.png"
        if daily_hist.exists():
            st.image(str(daily_hist), use_container_width=True)
        else:
            st.warning(f"画像が見つかりません: {daily_hist}")

    st.markdown("""
    **分析所見:**
    - 全カテゴリで**分散/平均比 > 1**（過分散）
    - 全カテゴリで**ゼロ率がポアソン期待値を大幅に超過**（ゼロ過剰）
    - 高い歪度・尖度 → 右裾が重い分布
    """)

# 概要セクション
def show_overview():
    st.subheader("📋 分析概要")

    st.markdown("""
    ### 目的
    ロジスティック回帰分析で特定された黒字要因TOP5について、確率分布特性を分析し、
    数理最適化および時系列予測モデル選定の考慮事項を明文化する。

    ### 対象データ
    - **損益データ**: `fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx`
    - **時系列データ**: `time_series_forecast_data_2024_fixed.xlsx`
    """)

    st.markdown("### 黒字要因TOP5（オッズ比順）")

    import pandas as pd
    top5_df = pd.DataFrame({
        "順位": ["1位", "2位", "3位", "4位", "5位"],
        "フィールド": ["WOMEN'S_JACKETS2", "Number_of_guests", "WOMEN'S_ONEPIECE", "Mens_KNIT", "Mens_PANTS"],
        "列番号": ["O列", "AA列", "T列", "P列", "Q列"],
        "オッズ比": ["最高", "高", "高", "中", "中"]
    })
    st.dataframe(top5_df, use_container_width=True, hide_index=True)

# 確率分布理論セクション
def show_distribution_theory():
    st.subheader("📐 確率分布の理論的解説")

    tab1, tab2 = st.tabs(["負の二項分布", "ゼロ過剰ポアソン分布"])

    with tab1:
        st.markdown("""
        ### 負の二項分布（Negative Binomial Distribution）

        **適用対象**: Number_of_guests, WOMEN'S_ONEPIECE, Mens_PANTS

        #### 定義
        負の二項分布は、成功確率pのベルヌーイ試行をr回成功するまでに必要な失敗回数の分布。

        #### 特徴
        | 特性 | 説明 |
        |------|------|
        | 平均 | μ = r(1-p)/p |
        | 分散 | σ² = r(1-p)/p² |
        | **過分散対応** | σ² > μ（分散が平均より大きい場合に適切） |
        | ポアソン分布との関係 | r→∞でポアソン分布に収束 |

        #### Number_of_guestsへの適用理由
        - 分散/平均比 = **225.95**（極度の過分散）
        - 客数は「来店イベント」の集積であり、日によってばらつきが大きい
        - 天候・曜日・イベントなどの外部要因により分散が増大
        """)

    with tab2:
        st.markdown("""
        ### ゼロ過剰ポアソン分布（Zero-Inflated Poisson, ZIP）

        **適用対象**: WOMEN'S_ONEPIECE, Mens_KNIT, Mens_PANTS

        #### 定義
        ゼロ過剰ポアソン分布は、ゼロが通常のポアソン分布で期待されるよりも多く発生するデータに適用。

        #### 特徴
        | 特性 | 説明 |
        |------|------|
        | 構造的ゼロ | 「売れない日」が存在（店休日、在庫切れ等） |
        | サンプリングゼロ | たまたま売れなかった日 |
        | **二重のゼロ生成メカニズム** | 両方を同時にモデル化 |

        #### WOMEN'S_ONEPIECEのゼロ過剰検証結果
        | 指標 | 値 |
        |------|-----|
        | 実際のゼロ率 | 13.4% |
        | ポアソン期待ゼロ率 | 0.5% |
        | **ゼロ過剰度** | **13.0%pt** |

        → ポアソン分布の26倍のゼロが発生 → **ZIPが適切**
        """)

# モデル比較結果セクション
def show_model_comparison():
    st.subheader("🔬 AutoGluon-TimeSeries モデル比較結果")

    st.markdown("""
    ### Mens_PANTS NegBin vs Chronos-Bolt 実行結果（2026-01-02実施）

    Mens_PANTSデータに対して、DeepAR-NegBinとChronos-Boltの比較検証を実施。
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### データ特性分析結果")
        import pandas as pd
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

    # 比較チャート
    import pandas as pd
    chart_data = pd.DataFrame({
        "モデル": ["DeepAR-NegBin", "Chronos-Bolt"],
        "WQL": [0.2200, 0.2337]
    })
    st.bar_chart(chart_data.set_index("モデル"))

    st.success("""
    **結論**: DeepAR-NegBinがChronos-Boltより**6.2%優れた精度**を達成。
    過分散カウントデータには負の二項分布モデルが最適。
    """)

    st.markdown("""
    #### 検証結果の考察
    1. **DeepAR-NegBinがChronos-Boltより6.2%優れた精度**を達成
    2. 過分散データ（分散/平均比=11.2）に対して、負の二項分布を明示的にモデル化することで予測精度が向上
    3. Chronos-Boltは訓練時間が短く（1.8秒 vs 150秒）、ゼロショット性能として優秀
    4. 精度重視の場合はDeepAR-NegBin、速度重視の場合はChronos-Boltを選択

    #### 実行環境
    - AutoGluon Version: 1.5.0
    - Python: 3.11.4
    - CPU: AMD64 32コア
    - GPU: 未使用（CPU推論）
    """)

# 推奨アクションセクション
def show_recommendations():
    st.subheader("🎯 推奨アクションサマリー")

    recommendations = [
        {
            "カテゴリ": "WOMEN'S_JACKETS2（1位）",
            "分布特性": "高級カテゴリ、正規分布ベース、季節商品",
            "推奨モデル": "Chronos2 + TFT",
            "推奨アクション": "1月中に集中販促実施、2月以降は逓減見込み"
        },
        {
            "カテゴリ": "Number_of_guests（2位）",
            "分布特性": "負の二項分布（分散/平均比=226、極度の過分散）",
            "推奨モデル": "DeepAR (NegBin) + 共変量",
            "推奨アクション": "12月の高客数活用、5月にイベント企画"
        },
        {
            "カテゴリ": "WOMEN'S_ONEPIECE（3位）",
            "分布特性": "ゼロ過剰ポアソン分布（ゼロ率13.4% vs 期待0.5%）",
            "推奨モデル": "Chronos2 + TFT",
            "推奨アクション": "春夏シーズン（3-6月）集中販促、秋冬在庫縮小"
        },
        {
            "カテゴリ": "Mens_KNIT（4位）",
            "分布特性": "季節カテゴリ、ポアソン分布ベース",
            "推奨モデル": "Chronos2",
            "推奨アクション": "10月-12月集中販促、1月以降在庫処分"
        },
        {
            "カテゴリ": "Mens_PANTS（5位）",
            "分布特性": "負の二項分布（分散/平均比=11.2）【実証済み】",
            "推奨モデル": "DeepAR (NegBin) + TFT",
            "推奨アクション": "通年販売、セール時期に集中投入"
        }
    ]

    for rec in recommendations:
        with st.expander(f"📌 {rec['カテゴリ']}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**分布特性**: {rec['分布特性']}")
                st.markdown(f"**推奨モデル**: `{rec['推奨モデル']}`")
            with col2:
                st.info(f"**推奨アクション**: {rec['推奨アクション']}")

    st.markdown("---")
    st.markdown("### モデル選定サマリー")

    import pandas as pd
    summary_df = pd.DataFrame({
        "カテゴリ": ["WOMEN'S_JACKETS2", "Number_of_guests", "WOMEN'S_ONEPIECE", "Mens_KNIT", "Mens_PANTS"],
        "推奨モデル": ["Chronos2 + TFT", "DeepAR (NegBin) + 共変量", "Chronos2 + TFT", "Chronos2", "DeepAR (NegBin) + TFT"],
        "理由": ["季節性＋イベント需要", "極度の過分散", "ゼロ過剰＋季節性", "ポアソン分布適合", "過分散＋ゼロ過剰"]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# フルレポート表示
def show_full_report():
    st.subheader("📄 フルレポート")

    report_path = DOCS_DIR / "Top5_Factors_Analysis_Report.md"

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()

        # 画像パスを修正（相対パスからStreamlit用に変更）
        # ../output/ → output/ に変換して表示
        st.markdown(report_content)

        # ダウンロードボタン
        st.download_button(
            label="📥 レポートをダウンロード (Markdown)",
            data=report_content,
            file_name="Top5_Factors_Analysis_Report.md",
            mime="text/markdown"
        )
    else:
        st.error(f"レポートファイルが見つかりません: {report_path}")

# メイン処理
if section == "概要":
    show_overview()
elif section == "ヒストグラム分析":
    show_histograms()
elif section == "確率分布理論":
    show_distribution_theory()
elif section == "モデル比較結果":
    show_model_comparison()
elif section == "推奨アクション":
    show_recommendations()
elif section == "フルレポート":
    show_full_report()

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <small>作成者: Takashi.Takenouchi | 最終更新: 2026-01-02 | Version 1.1.0</small>
</div>
""", unsafe_allow_html=True)
