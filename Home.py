# -*- coding: utf-8 -*-
"""
営業利益改善AI Agentsダッシュボード
メインエントリーポイント（ホームページ）
"""
import streamlit as st
import sys
import os

# パス設定
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from components.auth import check_authentication, show_login_form, show_logout_button

# ページ設定
st.set_page_config(
    page_title="営業利益改善ダッシュボード",
    page_icon="📊",
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
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
        height: 100%;
    }
    .feature-card h4 {
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .workflow-step {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .workflow-arrow {
        font-size: 2rem;
        color: #1f77b4;
        text-align: center;
    }
    .stats-card {
        background: linear-gradient(135deg, #1f77b4, #2ca02c);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def show_home_page():
    """ホームページを表示"""
    # ヘッダー
    st.markdown('<div class="main-header">📊 営業利益改善AI Agentsダッシュボード</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">QCストーリーに基づく店舗損益分析・改善システム</div>', unsafe_allow_html=True)

    # ユーザー情報表示
    username = st.session_state.get("username", "ゲスト")
    st.sidebar.markdown(f"**ログインユーザー:** {username}")
    show_logout_button()

    st.markdown("---")

    # ワークフロー説明
    st.markdown("### 🔄 分析ワークフロー（QCストーリー）")

    cols = st.columns(6)
    workflow_items = [
        ("1️⃣", "現状把握", "EDA分析"),
        ("2️⃣", "要因分析", "ロジスティック回帰"),
        ("3️⃣", "目標設定", "改善目標入力"),
        ("4️⃣", "最適化", "PuLP実行"),
        ("5️⃣", "予測", "時系列分析"),
        ("6️⃣", "出力", "レポート生成")
    ]

    for i, (icon, title, desc) in enumerate(workflow_items):
        with cols[i]:
            st.markdown(f"""
            <div class="workflow-step">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-weight: bold;">{title}</div>
                <div style="font-size: 0.8rem; color: #666;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 機能カード
    st.markdown("### 📋 機能一覧")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 現状把握（EDA）</h4>
            <p>店舗別損益データの探索的データ分析</p>
            <ul>
                <li>ヒストグラム（分布確認）</li>
                <li>箱ひげ図（グループ比較）</li>
                <li>相関行列（変数間関係）</li>
                <li>VIF・正規性検定</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>🔍 要因分析</h4>
            <p>ロジスティック回帰による黒字化要因特定</p>
            <ul>
                <li>L1正則化モデル</li>
                <li>オッズ比分析</li>
                <li>TOP5黒字化要因</li>
                <li>赤字要因の特定</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 目標設定</h4>
            <p>営業利益改善目標の設定</p>
            <ul>
                <li>店舗選択</li>
                <li>対象期間設定</li>
                <li>赤字月数目標</li>
                <li>変動幅設定（±30%）</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>⚡ 最適化実行</h4>
            <p>PuLPによる数理最適化</p>
            <ul>
                <li>制約条件の設定</li>
                <li>最適化実行</li>
                <li>Before/After比較</li>
                <li>改善シミュレーション</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📈 時系列予測</h4>
            <p>90日間販売予測の可視化</p>
            <ul>
                <li>Chronos2 + TFT モデル</li>
                <li>信頼区間付きグラフ</li>
                <li>カテゴリ別集計</li>
                <li>CSVダウンロード</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>📥 レポート出力</h4>
            <p>改善結果のエクスポート</p>
            <ul>
                <li>改善前後比較表</li>
                <li>Excel出力</li>
                <li>分析サマリー</li>
                <li>改善P/Lダウンロード</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # クイックスタート
    st.markdown("### 🚀 クイックスタート")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **はじめての方へ**

        1. 左サイドバーから **「1_現状把握」** を選択
        2. Excelファイルをアップロード（またはサンプルデータを使用）
        3. 各ページを順番に進めて分析を実施
        """)

    with col2:
        st.success("""
        **使用データ形式**

        - Excel形式（.xlsx）
        - 必須カラム: shop, Date, Operating_profit, gross_profit, operating_cost
        - 推奨: 月次データ（69ヶ月分）
        """)

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>営業利益改善AI Agents v1.0 | Powered by Streamlit & AutoGluon</p>
        <p>QCストーリー: 現状把握 → 要因分析 → 目標設定 → 対策立案 → 効果確認</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """メイン処理"""
    if not check_authentication():
        show_login_form()
    else:
        show_home_page()


if __name__ == "__main__":
    main()
