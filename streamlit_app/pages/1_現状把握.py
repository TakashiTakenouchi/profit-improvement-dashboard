# -*- coding: utf-8 -*-
"""
現状把握ページ（EDAダッシュボード）
店舗別損益データの探索的データ分析
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.auth import check_authentication, show_login_form, show_logout_button
from components.data_loader import show_file_uploader, validate_dataframe, get_store_options
from components.charts import create_histogram, create_boxplot, create_correlation_heatmap

# ページ設定
st.set_page_config(
    page_title="現状把握 - EDA",
    page_icon="📊",
    layout="wide"
)


def calculate_vif(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """VIF（分散膨張係数）を計算"""
    from sklearn.linear_model import LinearRegression

    vif_data = []
    for col in columns:
        other_cols = [c for c in columns if c != col]
        if len(other_cols) == 0:
            continue

        X = df[other_cols].fillna(0)
        y = df[col].fillna(0)

        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)

        vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
        vif_data.append({'変数': col, 'VIF': round(vif, 2)})

    return pd.DataFrame(vif_data)


def run_normality_test(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """正規性検定（Shapiro-Wilk）を実行"""
    results = []
    for col in columns:
        data = df[col].dropna()
        if len(data) >= 3:
            stat, p_value = stats.shapiro(data[:5000])  # 最大5000件
            results.append({
                '変数': col,
                '統計量': round(stat, 4),
                'p値': round(p_value, 4),
                '正規性': '正規分布' if p_value > 0.05 else '非正規分布'
            })

    return pd.DataFrame(results)


def main():
    """メイン処理"""
    if not check_authentication():
        show_login_form()
        return

    # ヘッダー
    st.markdown("# 📊 現状把握（EDAダッシュボード）")
    st.markdown("店舗別損益データの探索的データ分析を行います。")

    # サイドバー
    show_logout_button()

    # データ読み込み
    df, source = show_file_uploader()

    if df is None:
        st.info("👆 上記からデータをアップロードまたはサンプルデータを選択してください。")
        return

    # セッションに保存
    st.session_state['uploaded_data'] = df
    st.session_state['data_source'] = source

    # バリデーション
    is_valid, missing_cols = validate_dataframe(df)
    if not is_valid:
        st.error(f"⚠️ 必須カラムが不足しています: {missing_cols}")
        return

    st.success("✅ データの読み込みが完了しました")

    # データプレビュー
    with st.expander("📋 データプレビュー", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown(f"**データサイズ:** {len(df)}行 × {len(df.columns)}列")

    st.markdown("---")

    # 店舗フィルター
    stores = get_store_options(df)
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_store = st.selectbox("店舗を選択", ["全店舗"] + stores)

    if selected_store != "全店舗":
        if 'shop' in df.columns:
            df_filtered = df[df['shop'] == selected_store]
        else:
            shop_code_map = {'恵比寿': 11, '横浜元町': 12}
            df_filtered = df[df['shop_code'] == shop_code_map.get(selected_store)]
    else:
        df_filtered = df

    st.markdown(f"**分析対象:** {len(df_filtered)}件")

    # 数値カラムの抽出
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['shop_code', 'year', 'month']
    analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

    # タブ
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 基本統計量",
        "📈 ヒストグラム",
        "📦 箱ひげ図",
        "🔥 相関行列",
        "🔬 統計検定"
    ])

    with tab1:
        st.markdown("### 基本統計量")
        stats_df = df_filtered[analysis_cols].describe().T
        stats_df['歪度'] = df_filtered[analysis_cols].skew()
        stats_df['尖度'] = df_filtered[analysis_cols].kurtosis()
        st.dataframe(stats_df.round(2), use_container_width=True)

    with tab2:
        st.markdown("### ヒストグラム（分布確認）")
        selected_hist_cols = st.multiselect(
            "表示する変数を選択",
            analysis_cols,
            default=analysis_cols[:6] if len(analysis_cols) >= 6 else analysis_cols
        )
        if selected_hist_cols:
            fig = create_histogram(df_filtered, selected_hist_cols, "変数の分布")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 箱ひげ図（グループ別比較）")

        # judge列の作成
        if 'judge' not in df_filtered.columns:
            mean_profit = df_filtered['Operating_profit'].mean()
            df_filtered['judge'] = (df_filtered['Operating_profit'] > mean_profit).astype(int)
            st.info(f"📌 judge列を作成しました（平均営業利益 {mean_profit:,.0f}円 を基準）")

        selected_box_cols = st.multiselect(
            "表示する変数を選択（箱ひげ図）",
            analysis_cols,
            default=analysis_cols[:6] if len(analysis_cols) >= 6 else analysis_cols,
            key="boxplot_cols"
        )
        if selected_box_cols:
            fig = create_boxplot(df_filtered, selected_box_cols, 'judge', "judge別分布比較")
            st.plotly_chart(fig, use_container_width=True)

            # 凡例
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🔴 **judge=0**: 営業利益 ≤ 平均（赤字傾向）")
            with col2:
                st.markdown("🟢 **judge=1**: 営業利益 > 平均（黒字傾向）")

    with tab4:
        st.markdown("### 相関行列ヒートマップ")
        selected_corr_cols = st.multiselect(
            "表示する変数を選択（相関行列）",
            analysis_cols,
            default=analysis_cols[:10] if len(analysis_cols) >= 10 else analysis_cols,
            key="corr_cols"
        )
        if selected_corr_cols and len(selected_corr_cols) >= 2:
            fig = create_correlation_heatmap(df_filtered, selected_corr_cols, "変数間の相関")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("### 統計検定")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### VIF（多重共線性）")
            selected_vif_cols = st.multiselect(
                "VIF計算対象",
                analysis_cols,
                default=analysis_cols[:8] if len(analysis_cols) >= 8 else analysis_cols,
                key="vif_cols"
            )
            if selected_vif_cols and len(selected_vif_cols) >= 2:
                vif_df = calculate_vif(df_filtered, selected_vif_cols)
                st.dataframe(vif_df, use_container_width=True)
                st.markdown("⚠️ **VIF > 10** は多重共線性の疑いあり")

        with col2:
            st.markdown("#### 正規性検定（Shapiro-Wilk）")
            selected_norm_cols = st.multiselect(
                "正規性検定対象",
                analysis_cols,
                default=analysis_cols[:8] if len(analysis_cols) >= 8 else analysis_cols,
                key="norm_cols"
            )
            if selected_norm_cols:
                norm_df = run_normality_test(df_filtered, selected_norm_cols)
                st.dataframe(norm_df, use_container_width=True)
                st.markdown("📌 **p値 > 0.05** で正規分布と判定")

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>次のステップ: 左サイドバーから「2_要因分析」へ進んでください</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
