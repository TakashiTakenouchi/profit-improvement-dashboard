# -*- coding: utf-8 -*-
"""
チャートコンポーネント
Plotlyを使用したグラフ生成
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional


def create_histogram(df: pd.DataFrame, columns: List[str], title: str = "分布") -> go.Figure:
    """ヒストグラムを作成"""
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=columns[:n_rows*n_cols]
    )

    for i, col in enumerate(columns[:n_rows*n_cols]):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_idx
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=300 * n_rows,
        template='plotly_white'
    )

    return fig


def create_boxplot(df: pd.DataFrame, columns: List[str], group_col: str = 'judge',
                   title: str = "グループ別分布") -> go.Figure:
    """箱ひげ図を作成"""
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=columns[:n_rows*n_cols]
    )

    colors = {'0': '#d62728', '1': '#2ca02c'}

    for i, col in enumerate(columns[:n_rows*n_cols]):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        for group in df[group_col].unique():
            group_data = df[df[group_col] == group][col]
            color = colors.get(str(group), '#1f77b4')
            fig.add_trace(
                go.Box(
                    y=group_data,
                    name=f"{group_col}={group}",
                    marker_color=color,
                    showlegend=(i == 0)
                ),
                row=row, col=col_idx
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=350 * n_rows,
        template='plotly_white',
        boxmode='group'
    )

    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: List[str],
                                title: str = "相関行列") -> go.Figure:
    """相関行列ヒートマップを作成"""
    corr_matrix = df[columns].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=600,
        width=800,
        template='plotly_white'
    )

    return fig


def create_odds_ratio_chart(odds_df: pd.DataFrame, title: str = "オッズ比分析") -> go.Figure:
    """オッズ比チャートを作成"""
    # オッズ比でソート
    odds_df = odds_df.sort_values('odds_ratio', ascending=True)

    # 色分け（オッズ比 > 1: 緑、< 1: 赤）
    colors = ['#2ca02c' if x > 1 else '#d62728' for x in odds_df['odds_ratio']]

    fig = go.Figure()

    # 基準線（オッズ比 = 1）
    fig.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="基準線")

    # 水平バーチャート
    fig.add_trace(go.Bar(
        x=odds_df['odds_ratio'],
        y=odds_df['feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:.2f}" for x in odds_df['odds_ratio']],
        textposition='outside'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="オッズ比",
        yaxis_title="変数",
        height=max(400, len(odds_df) * 25),
        template='plotly_white',
        showlegend=False
    )

    return fig


def create_before_after_chart(before_data: List[float], after_data: List[float],
                               labels: List[str], title: str = "改善前後比較") -> go.Figure:
    """改善前後の比較チャートを作成"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='改善前',
        x=labels,
        y=before_data,
        marker_color='#d62728',
        opacity=0.7
    ))

    fig.add_trace(go.Bar(
        name='改善後',
        x=labels,
        y=after_data,
        marker_color='#2ca02c',
        opacity=0.7
    ))

    # ゼロライン
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="月",
        yaxis_title="Operating Profit",
        barmode='group',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_time_series_chart(df: pd.DataFrame, actual_col: str, forecast_col: str,
                              lower_col: str = '0.1', upper_col: str = '0.9',
                              title: str = "時系列予測") -> go.Figure:
    """時系列予測チャートを作成"""
    fig = go.Figure()

    # 信頼区間
    fig.add_trace(go.Scatter(
        x=pd.concat([df['timestamp'], df['timestamp'][::-1]]),
        y=pd.concat([df[upper_col], df[lower_col][::-1]]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='90%信頼区間'
    ))

    # 実績
    if actual_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[actual_col],
            mode='lines',
            name='実績',
            line=dict(color='#2ca02c', width=2)
        ))

    # 予測
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[forecast_col],
        mode='lines',
        name='予測',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="日付",
        yaxis_title="数量",
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def create_profit_variance_chart(df: pd.DataFrame, title: str = "営業利益変動") -> go.Figure:
    """営業利益変動チャートを作成"""
    df = df.sort_values('Date')

    # 赤字/黒字で色分け
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in df['Operating_profit']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Operating_profit'],
        marker_color=colors,
        name='Operating Profit'
    ))

    # 平均線
    avg = df['Operating_profit'].mean()
    fig.add_hline(y=avg, line_dash="dash", line_color="blue",
                  annotation_text=f"平均: {avg:,.0f}")
    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="日付",
        yaxis_title="営業利益",
        height=400,
        template='plotly_white'
    )

    return fig
