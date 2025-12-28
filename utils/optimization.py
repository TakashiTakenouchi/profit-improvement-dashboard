# -*- coding: utf-8 -*-
"""
PuLP最適化ユーティリティ
営業利益の最適化を実行
"""
import pandas as pd
import numpy as np
from pulp import *
from typing import Tuple, List, Dict, Optional


# コスト項目
COST_FIELDS = ['rent', 'personnel_expenses', 'depreciation',
               'sales_promotion', 'head_office_expenses']


def run_pulp_optimization(df: pd.DataFrame, target_indices: List[int],
                          target_deficit_months: int = 4,
                          variance_ratio: float = 0.3) -> Tuple[pd.DataFrame, Dict]:
    """
    PuLPによる営業利益最適化

    Args:
        df: 元データフレーム
        target_indices: 最適化対象のインデックス
        target_deficit_months: 目標赤字月数
        variance_ratio: 変動幅（±30%なら0.3）

    Returns:
        result_df: 最適化後のデータフレーム
        summary: 最適化サマリー
    """
    df_result = df.copy()
    n_months = len(target_indices)

    # 元データの取得
    gross_profits = [df.loc[idx, 'gross_profit'] for idx in target_indices]
    original_op_costs = [df.loc[idx, 'operating_cost'] for idx in target_indices]
    original_op_profits = [df.loc[idx, 'Operating_profit'] for idx in target_indices]

    # 手動最適化アルゴリズム
    # Operating_profitが低い順に赤字月を選択
    np.random.seed(42)

    sorted_month_indices = sorted(
        range(n_months),
        key=lambda i: original_op_profits[i]
    )
    deficit_month_indices = sorted_month_indices[:target_deficit_months]
    surplus_month_indices = sorted_month_indices[target_deficit_months:]

    # 赤字月のOperating_profit（負の値）
    deficit_target_profits = []
    for i in deficit_month_indices:
        deficit_amount = abs(original_op_profits[i]) * np.random.uniform(0.1, 0.5)
        deficit_target_profits.append(-deficit_amount)

    # 年間合計を維持
    total_original_op_profit = sum(original_op_profits)
    total_deficit = sum(deficit_target_profits)
    total_required_surplus = total_original_op_profit - total_deficit

    # 黒字月のOperating_profitを配分
    surplus_coefficients = np.random.uniform(1 - variance_ratio, 1 + variance_ratio, len(surplus_month_indices))
    surplus_base = [original_op_profits[i] for i in surplus_month_indices]
    surplus_adjusted = [surplus_base[j] * surplus_coefficients[j] for j in range(len(surplus_month_indices))]
    surplus_adjusted_total = sum(surplus_adjusted)

    scale_factor = total_required_surplus / surplus_adjusted_total if surplus_adjusted_total != 0 else 1
    surplus_final = [v * scale_factor for v in surplus_adjusted]

    # 結果の構築
    new_op_profits = [0.0] * n_months
    for j, i in enumerate(deficit_month_indices):
        new_op_profits[i] = deficit_target_profits[j]
    for j, i in enumerate(surplus_month_indices):
        new_op_profits[i] = surplus_final[j]

    # operating_costを逆算
    new_op_costs = [gross_profits[i] - new_op_profits[i] for i in range(n_months)]

    # DataFrameに適用
    for i, idx in enumerate(target_indices):
        old_op_cost = original_op_costs[i]
        new_op_cost = new_op_costs[i]

        df_result.loc[idx, 'Operating_profit'] = new_op_profits[i]
        df_result.loc[idx, 'operating_cost'] = new_op_cost

        # コスト内訳を按分
        if old_op_cost > 0:
            ratio = new_op_cost / old_op_cost
            for field in COST_FIELDS:
                if field in df_result.columns:
                    df_result.loc[idx, field] = df.loc[idx, field] * ratio

    # judge列の再計算
    avg_op_profit = df_result['Operating_profit'].mean()
    df_result['judge'] = (df_result['Operating_profit'] > avg_op_profit).astype(int)

    # サマリー作成
    deficit_count = sum(1 for op in new_op_profits if op < 0)
    summary = {
        'target_months': n_months,
        'deficit_months_before': sum(1 for op in original_op_profits if op < 0),
        'deficit_months_after': deficit_count,
        'target_deficit_months': target_deficit_months,
        'total_op_profit_before': sum(original_op_profits),
        'total_op_profit_after': sum(new_op_profits),
        'variance_ratio': variance_ratio,
        'success': abs(deficit_count - target_deficit_months) <= 1
    }

    return df_result, summary


def calculate_improvement_metrics(df_before: pd.DataFrame, df_after: pd.DataFrame,
                                   target_indices: List[int]) -> Dict:
    """
    改善メトリクスを計算

    Args:
        df_before: 最適化前のデータ
        df_after: 最適化後のデータ
        target_indices: 対象インデックス

    Returns:
        metrics: 改善メトリクス
    """
    before_profits = [df_before.loc[idx, 'Operating_profit'] for idx in target_indices]
    after_profits = [df_after.loc[idx, 'Operating_profit'] for idx in target_indices]

    before_deficit = sum(1 for p in before_profits if p < 0)
    after_deficit = sum(1 for p in after_profits if p < 0)

    metrics = {
        'months_count': len(target_indices),
        'before': {
            'deficit_months': before_deficit,
            'surplus_months': len(target_indices) - before_deficit,
            'total_profit': sum(before_profits),
            'avg_profit': np.mean(before_profits),
            'min_profit': min(before_profits),
            'max_profit': max(before_profits)
        },
        'after': {
            'deficit_months': after_deficit,
            'surplus_months': len(target_indices) - after_deficit,
            'total_profit': sum(after_profits),
            'avg_profit': np.mean(after_profits),
            'min_profit': min(after_profits),
            'max_profit': max(after_profits)
        },
        'improvement': {
            'deficit_change': before_deficit - after_deficit,
            'profit_change': sum(after_profits) - sum(before_profits)
        }
    }

    return metrics


def get_monthly_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame,
                           target_indices: List[int]) -> pd.DataFrame:
    """
    月別の比較データを取得

    Args:
        df_before: 最適化前のデータ
        df_after: 最適化後のデータ
        target_indices: 対象インデックス

    Returns:
        comparison_df: 月別比較データフレーム
    """
    comparison_data = []

    for idx in target_indices:
        before_op = df_before.loc[idx, 'Operating_profit']
        after_op = df_after.loc[idx, 'Operating_profit']
        before_oc = df_before.loc[idx, 'operating_cost']
        after_oc = df_after.loc[idx, 'operating_cost']

        date = df_before.loc[idx, 'Date']
        if hasattr(date, 'strftime'):
            month_str = date.strftime('%Y-%m')
        else:
            month_str = str(date)

        change_pct = ((after_op - before_op) / abs(before_op) * 100) if before_op != 0 else 0

        comparison_data.append({
            'Date': month_str,
            'Operating_profit_before': before_op,
            'Operating_profit_after': after_op,
            'change_percent': change_pct,
            'operating_cost_before': before_oc,
            'operating_cost_after': after_oc,
            'status_before': '黒字' if before_op >= 0 else '赤字',
            'status_after': '黒字' if after_op >= 0 else '赤字'
        })

    return pd.DataFrame(comparison_data)
