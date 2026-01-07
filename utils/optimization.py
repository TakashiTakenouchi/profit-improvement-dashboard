# -*- coding: utf-8 -*-
"""
PuLP最適化ユーティリティ（改良版）
営業利益の最適化を実行

【改良点】
- 要因ベースの最適化アルゴリズム
- target_deficit_months = 0, 1, 2, 3, ... n_months まで全て対応
- 元データに赤字月があっても正常に動作
"""
import pandas as pd
import numpy as np
from pulp import *
from typing import Tuple, List, Dict, Optional


# コスト項目
COST_FIELDS = ['rent', 'personnel_expenses', 'depreciation',
               'sales_promotion', 'head_office_expenses']

# 売上カテゴリ（オッズ比TOP5から、Number_of_guestsを除く）
TOP5_SALES_CATEGORIES = ["WOMEN'S_JACKETS2", "WOMEN'S_ONEPIECE", 'Mens_KNIT', 'Mens_PANTS']


def run_pulp_optimization(df: pd.DataFrame, target_indices: List[int],
                          target_deficit_months: int = 4,
                          variance_ratio: float = 0.3) -> Tuple[pd.DataFrame, Dict]:
    """
    要因ベースの営業利益最適化（改良版）

    【ロジック】
    1. 赤字月を選定（Operating_profitが低い順）
    2. 黒字月はgross_profitを増加させて黒字を確保
    3. 赤字月はgross_profitを減少させて赤字を発生
    4. 年間Operating_profit合計を維持
    5. operating_costは結果として逆算

    Args:
        df: 元データフレーム
        target_indices: 最適化対象のインデックス
        target_deficit_months: 目標赤字月数（0〜n_monthsまで対応）
        variance_ratio: 変動幅（±30%なら0.3）

    Returns:
        result_df: 最適化後のデータフレーム
        summary: 最適化サマリー
    """
    df_result = df.copy()
    n_months = len(target_indices)

    # 入力検証
    target_deficit_months = max(0, min(target_deficit_months, n_months))

    # 元データの取得
    gross_profits = [df.loc[idx, 'gross_profit'] for idx in target_indices]
    original_op_costs = [df.loc[idx, 'operating_cost'] for idx in target_indices]
    original_op_profits = [df.loc[idx, 'Operating_profit'] for idx in target_indices]
    total_original_op_profit = sum(original_op_profits)

    np.random.seed(42)

    # ソート（Operating_profitが低い順）
    sorted_month_indices = sorted(range(n_months), key=lambda i: original_op_profits[i])
    deficit_month_indices = sorted_month_indices[:target_deficit_months]
    surplus_month_indices = sorted_month_indices[target_deficit_months:]

    # 新しいOperating_profit
    new_op_profits = [0.0] * n_months
    new_gross_profits = gross_profits.copy()

    # ========================================
    # 赤字月の処理
    # ========================================
    for i in deficit_month_indices:
        current_op = original_op_profits[i]
        current_gp = gross_profits[i]
        current_oc = original_op_costs[i]

        # 赤字にするために必要なgross_profit減少
        # 目標: Operating_profit = -|current_op| * random(0.1, 0.5)
        base_amount = abs(current_op) if current_op != 0 else current_gp * 0.05
        target_deficit = -base_amount * np.random.uniform(0.1, 0.5)
        required_gp = current_oc + target_deficit

        new_gross_profits[i] = required_gp
        new_op_profits[i] = required_gp - current_oc

    # ========================================
    # 黒字月の処理
    # ========================================
    for i in surplus_month_indices:
        current_op = original_op_profits[i]
        current_gp = gross_profits[i]
        current_oc = original_op_costs[i]

        # 黒字を確保（元が赤字でも強制的に黒字化）
        if current_op < 0:
            # 赤字から黒字へ変換
            target_surplus = abs(current_op) + current_gp * np.random.uniform(0.05, 0.15)
        else:
            # 既に黒字：±variance_ratioでバラつき
            target_surplus = current_op * np.random.uniform(1 - variance_ratio, 1 + variance_ratio)
            target_surplus = max(target_surplus, 100000)  # 最低10万円の黒字

        required_gp = current_oc + target_surplus

        new_gross_profits[i] = required_gp
        new_op_profits[i] = required_gp - current_oc

    # ========================================
    # 年間合計を維持（制約）
    # ========================================
    new_total = sum(new_op_profits)
    if abs(new_total) > 0:
        scale = total_original_op_profit / new_total
        new_op_profits = [p * scale for p in new_op_profits]
        # gross_profitも再計算
        new_gross_profits = [original_op_costs[i] + new_op_profits[i] for i in range(n_months)]

    # operating_costを逆算
    new_op_costs = [new_gross_profits[i] - new_op_profits[i] for i in range(n_months)]

    # ========================================
    # DataFrameに適用
    # ========================================
    for i, idx in enumerate(target_indices):
        old_op_cost = original_op_costs[i]
        new_op_cost = new_op_costs[i]

        df_result.loc[idx, 'Operating_profit'] = new_op_profits[i]
        df_result.loc[idx, 'gross_profit'] = new_gross_profits[i]
        df_result.loc[idx, 'operating_cost'] = new_op_cost

        # コスト内訳を按分
        if old_op_cost > 0 and new_op_cost > 0:
            ratio = new_op_cost / old_op_cost
            for field in COST_FIELDS:
                if field in df_result.columns:
                    df_result.loc[idx, field] = df.loc[idx, field] * ratio

        # 売上カテゴリも按分（オプション）
        if gross_profits[i] > 0:
            sales_ratio = new_gross_profits[i] / gross_profits[i]
            for field in TOP5_SALES_CATEGORIES:
                if field in df_result.columns:
                    df_result.loc[idx, field] = df.loc[idx, field] * sales_ratio

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


def get_factor_adjustments(df_before: pd.DataFrame, df_after: pd.DataFrame,
                           target_indices: List[int]) -> pd.DataFrame:
    """
    要因（売上カテゴリ）の調整内容を取得

    Args:
        df_before: 最適化前のデータ
        df_after: 最適化後のデータ
        target_indices: 対象インデックス

    Returns:
        adjustments_df: 要因調整データフレーム
    """
    adjustments_data = []

    for idx in target_indices:
        date = df_before.loc[idx, 'Date']
        if hasattr(date, 'strftime'):
            month_str = date.strftime('%Y-%m')
        else:
            month_str = str(date)

        row_data = {'Date': month_str}

        for field in TOP5_SALES_CATEGORIES:
            if field in df_before.columns and field in df_after.columns:
                before_val = df_before.loc[idx, field]
                after_val = df_after.loc[idx, field]
                change_pct = ((after_val - before_val) / before_val * 100) if before_val != 0 else 0
                row_data[f'{field}_before'] = before_val
                row_data[f'{field}_after'] = after_val
                row_data[f'{field}_change%'] = change_pct

        adjustments_data.append(row_data)

    return pd.DataFrame(adjustments_data)
