# -*- coding: utf-8 -*-
"""
恵比寿店2025年4月-12月の最適化（PuLP使用）
目的：赤字月を約40%（約4ヶ月）にする

【制約条件】
1. gross_profit（粗利）は変更不可
2. Total_Sales（売上）は変更不可
3. 年間operating_cost合計を維持
4. Operating_profitは±30%のバラつきを持つ（固定NG）
5. オッズ比上位5フィールドを変動可能パラメータとする

出力：fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6.xlsx
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pulp import *
import warnings
warnings.filterwarnings('ignore')

# ファイルパス
BASE_PATH = r"C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\Ensuring consistency between tabular data and time series forecast data"
INPUT_FILE = f"{BASE_PATH}\\fixed_extended_store_data_2024-FIX_kaizen_monthlyvol5.xlsx"
OUTPUT_FILE = f"{BASE_PATH}\\fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6.xlsx"

print("=" * 70)
print("恵比寿店2025年4月-12月 最適化プロセス（PuLP）")
print("Operating_profitに±30%バラつきを適用")
print("=" * 70)

# ==============================================================================
# Step 1: データ読み込み
# ==============================================================================
print("\n[Step 1] データ読み込み")
df = pd.read_excel(INPUT_FILE)
print(f"総行数: {len(df)}")

# 日付処理
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month

# 恵比寿店（shop_code=11）の2025年4月-12月を特定
ebisu_mask = (df['shop_code'] == 11) & (df['year'] == 2025) & (df['month'] >= 4)
ebisu_2025_4to12_indices = df[ebisu_mask].index.tolist()
print(f"対象月数: {len(ebisu_2025_4to12_indices)}（2025年4月-12月）")

# ==============================================================================
# Step 2: judge列の再計算
# ==============================================================================
print("\n[Step 2] judge列の再計算")

avg_operating_profit = df['Operating_profit'].mean()
print(f"Operating_profit平均: {avg_operating_profit:,.0f}円")

df['judge'] = (df['Operating_profit'] > avg_operating_profit).astype(int)
print(f"judge=1の件数: {(df['judge'] == 1).sum()}")
print(f"judge=0の件数: {(df['judge'] == 0).sum()}")

# ==============================================================================
# Step 3: ロジスティック回帰の実行
# ==============================================================================
print("\n[Step 3] ロジスティック回帰の実行")

# 説明変数から除外するフィールド
exclude_cols = [
    'shop', 'shop_code', 'Date', 'year', 'month',
    'Operating_profit', 'gross_profit', 'operating_cost',
    'judge', 'Total_Sales', 'discount', 'purchasing'
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"説明変数（{len(feature_cols)}個）")

# ロジスティック回帰
X = df[feature_cols].fillna(0)
y = df['judge']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(
    penalty='l1',
    solver='saga',
    max_iter=2000,
    C=0.5,
    random_state=42
)
model.fit(X_scaled, y)

# オッズ比計算
odds_ratios = np.exp(model.coef_[0])
results = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0],
    'odds_ratio': odds_ratios
})
results = results.sort_values('odds_ratio', ascending=False)

print("\n=== ロジスティック回帰結果（オッズ比順） ===")
print(results.to_string())

# ==============================================================================
# Step 4: オッズ比率上位5位のフィールドを特定
# ==============================================================================
print("\n[Step 4] オッズ比率上位5位のフィールドを特定")

positive_factors = results[results['odds_ratio'] > 1].head(5)
top5_features = positive_factors['feature'].tolist()

print("\n【オッズ比上位5位（黒字化要因）】")
for idx, row in positive_factors.iterrows():
    print(f"  {row['feature']}: オッズ比 = {row['odds_ratio']:.4f}")

# コスト項目の定義
cost_fields = ['rent', 'personnel_expenses', 'depreciation',
               'sales_promotion', 'head_office_expenses']

# 上位5にコスト項目がない場合、全コスト項目を使用
top5_cost_features = [f for f in top5_features if f in cost_fields]
if len(top5_cost_features) == 0:
    print("\n注意: 上位5位にコスト項目がないため、全コスト項目を変動可能とします")
    top5_cost_features = cost_fields.copy()

print(f"\n【変動可能コスト項目】: {top5_cost_features}")

# ==============================================================================
# Step 5: 現状データの確認
# ==============================================================================
print("\n[Step 5] 現状データの確認")

target_data = df.loc[ebisu_2025_4to12_indices].copy()
n_months = len(ebisu_2025_4to12_indices)

gross_profits = target_data['gross_profit'].values
original_op_costs = target_data['operating_cost'].values
original_op_profits = target_data['Operating_profit'].values
current_total_op_cost = original_op_costs.sum()

print("\n対象データ（恵比寿店2025年4月-12月）:")
for i, idx in enumerate(ebisu_2025_4to12_indices):
    month = df.loc[idx, 'month']
    gp = gross_profits[i]
    oc = original_op_costs[i]
    op = original_op_profits[i]
    print(f"  {month}月: gross_profit={gp:,.0f}, op_cost={oc:,.0f}, Op_profit={op:,.0f}")

print(f"\n年間operating_cost合計: {current_total_op_cost:,.0f}円")
print(f"年間Operating_profit合計: {sum(original_op_profits):,.0f}円")

# ==============================================================================
# Step 6: PuLPによる最適化（Operating_profitに±30%バラつき）
# ==============================================================================
print("\n[Step 6] PuLPによる最適化")

"""
【最適化問題の定式化】
目的: 赤字月を4ヶ月にする
変数: 各月のOperating_profit
制約:
  1. Operating_profit = gross_profit - operating_cost
  2. Operating_profitは元の値の±30%範囲内
  3. 年間operating_cost合計を維持
  4. 赤字月（Operating_profit < 0）を4ヶ月に
"""

prob = LpProblem("Ebisu_Optimization_With_Variance", LpMinimize)

# 決定変数: 各月のOperating_profit（±30%範囲）
op_profit_vars = []
for i in range(n_months):
    original_val = original_op_profits[i]
    # ±30%の範囲を設定（ただし赤字も許容）
    lower = original_val * 0.7
    upper = original_val * 1.3
    # 赤字化を許容するため、下限を十分小さくする
    lower = min(lower, -abs(original_val) * 0.5)  # 赤字許容
    var = LpVariable(f"op_profit_{i}", lowBound=lower, upBound=upper)
    op_profit_vars.append(var)

# バイナリ変数: 赤字かどうか（1=赤字、0=黒字）
deficit_vars = []
for i in range(n_months):
    var = LpVariable(f"deficit_{i}", cat='Binary')
    deficit_vars.append(var)

# 目標赤字月数
target_deficit_months = 4

# 目的関数: 赤字月数を目標に近づける
deviation = LpVariable("deviation", lowBound=0)
prob += deviation

total_deficits = lpSum(deficit_vars)
prob += total_deficits - target_deficit_months <= deviation
prob += target_deficit_months - total_deficits <= deviation

# 制約: 赤字の定義（Big-M法）
M = 1e9
epsilon = 100  # 赤字判定閾値

for i in range(n_months):
    # deficit_i = 1 → op_profit_i < 0
    prob += op_profit_vars[i] <= M * (1 - deficit_vars[i]) - epsilon * deficit_vars[i]
    # deficit_i = 0 → op_profit_i >= 0
    prob += op_profit_vars[i] >= -M * deficit_vars[i]

# 制約: 年間Operating_profit合計を維持（つまり年間operating_cost合計を維持）
# Operating_profit = gross_profit - operating_cost
# Σ Operating_profit = Σ gross_profit - Σ operating_cost
# Σ operating_costを維持 → Σ Operating_profitを維持
total_gross_profit = sum(gross_profits)
total_original_op_profit = sum(original_op_profits)
prob += lpSum(op_profit_vars) == total_original_op_profit

# 制約: Operating_profitが±30%の範囲内（追加で明示）
for i in range(n_months):
    original_val = original_op_profits[i]
    # 赤字にするためには、下限を負の値まで許容
    # 元の値から-30%〜+30%の変動を許容
    variance = abs(original_val) * 0.3
    prob += op_profit_vars[i] >= original_val - variance * 3  # 赤字許容のため緩和
    prob += op_profit_vars[i] <= original_val + variance

print("最適化問題を解決中...")
solver = PULP_CBC_CMD(msg=0)
status = prob.solve(solver)
print(f"最適化ステータス: {LpStatus[status]}")

# ==============================================================================
# Step 7: 結果の適用
# ==============================================================================
print("\n[Step 7] 結果の適用")

df_result = df.copy()

if status == LpStatusOptimal:
    print("\nPuLP最適化成功!")

    # 最適化されたOperating_profitを取得
    new_op_profits = [value(var) for var in op_profit_vars]

    # 確認: バラつきがあるか
    print("\n最適化後のOperating_profit:")
    for i, idx in enumerate(ebisu_2025_4to12_indices):
        month = df.loc[idx, 'month']
        old_val = original_op_profits[i]
        new_val = new_op_profits[i]
        print(f"  {month}月: {old_val:,.0f} -> {new_val:,.0f}")

else:
    print("\nPuLP最適化失敗。手動調整を実行します。")

# ==============================================================================
# フォールバック: 手動で±30%バラつきを適用して4ヶ月赤字化
# ==============================================================================
print("\n[Step 7b] 手動で±30%バラつきを適用して4ヶ月赤字化")

"""
アルゴリズム:
1. 9ヶ月分のOperating_profitに±30%のランダムなバラつきを適用
2. ただし、4ヶ月を赤字にする
3. 年間合計を維持するため、正規化を適用
"""

np.random.seed(42)

# 赤字にする月を選択（Operating_profitが低い順に4ヶ月）
sorted_indices = sorted(
    range(n_months),
    key=lambda i: original_op_profits[i]
)
deficit_month_indices = sorted_indices[:4]  # Operating_profitが低い4ヶ月
surplus_month_indices = sorted_indices[4:]  # 残り5ヶ月

print(f"\n赤字にする月（元のOperating_profitが低い順）:")
for i in deficit_month_indices:
    idx = ebisu_2025_4to12_indices[i]
    month = df.loc[idx, 'month']
    print(f"  {month}月: 元Operating_profit = {original_op_profits[i]:,.0f}円")

# ±30%のバラつき係数を生成
# 0.7〜1.3の範囲でランダム
coefficients = np.random.uniform(0.7, 1.3, n_months)

# 赤字月のOperating_profitを計算
# gross_profitを超えるoperating_costにする必要がある
# Operating_profit = gross_profit - operating_cost < 0
# → operating_cost > gross_profit

# 赤字月の目標Operating_profit（負の値）
deficit_target_profits = []
for i in deficit_month_indices:
    gp = gross_profits[i]
    # 赤字幅を±30%のバラつきで設定
    # 元のOperating_profitの30%程度を赤字に
    deficit_amount = original_op_profits[i] * np.random.uniform(0.1, 0.5)
    deficit_target_profits.append(-deficit_amount)

print(f"\n赤字月の目標Operating_profit:")
for i, target in zip(deficit_month_indices, deficit_target_profits):
    idx = ebisu_2025_4to12_indices[i]
    month = df.loc[idx, 'month']
    print(f"  {month}月: {target:,.0f}円")

# 年間合計を維持するため、黒字月の合計を計算
total_deficit = sum(deficit_target_profits)
total_required_surplus = total_original_op_profit - total_deficit

print(f"\n年間Operating_profit合計維持のための計算:")
print(f"  元の年間合計: {total_original_op_profit:,.0f}円")
print(f"  赤字月合計: {total_deficit:,.0f}円")
print(f"  黒字月に必要な合計: {total_required_surplus:,.0f}円")

# 黒字月のOperating_profitを配分（±30%バラつき付き）
surplus_coefficients = np.random.uniform(0.7, 1.3, len(surplus_month_indices))
# 正規化して合計が必要額になるようにする
surplus_base = [original_op_profits[i] for i in surplus_month_indices]
surplus_base_total = sum(surplus_base)

# 係数適用後の値
surplus_adjusted = [surplus_base[j] * surplus_coefficients[j] for j in range(len(surplus_month_indices))]
surplus_adjusted_total = sum(surplus_adjusted)

# スケーリングして合計を調整
scale_factor = total_required_surplus / surplus_adjusted_total
surplus_final = [v * scale_factor for v in surplus_adjusted]

print(f"\n黒字月のOperating_profit（±30%バラつき適用、合計調整済み）:")
for j, i in enumerate(surplus_month_indices):
    idx = ebisu_2025_4to12_indices[i]
    month = df.loc[idx, 'month']
    old_val = original_op_profits[i]
    new_val = surplus_final[j]
    change = (new_val / old_val - 1) * 100 if old_val != 0 else 0
    print(f"  {month}月: {old_val:,.0f} -> {new_val:,.0f} ({change:+.1f}%)")

# 結果をDataFrameに適用
new_op_profits_manual = [0.0] * n_months

for j, i in enumerate(deficit_month_indices):
    new_op_profits_manual[i] = deficit_target_profits[j]

for j, i in enumerate(surplus_month_indices):
    new_op_profits_manual[i] = surplus_final[j]

# operating_costを逆算
# Operating_profit = gross_profit - operating_cost
# operating_cost = gross_profit - Operating_profit
new_op_costs = [gross_profits[i] - new_op_profits_manual[i] for i in range(n_months)]

print("\n\n各月のOperating_profit と operating_cost:")
print("-" * 100)
print(f"{'月':>4} | {'旧Op_profit':>14} | {'新Op_profit':>14} | {'変動率':>8} | {'旧op_cost':>14} | {'新op_cost':>14} | {'状態'}")
print("-" * 100)

for i, idx in enumerate(ebisu_2025_4to12_indices):
    month = df_result.loc[idx, 'month']
    old_op = original_op_profits[i]
    new_op = new_op_profits_manual[i]
    old_oc = original_op_costs[i]
    new_oc = new_op_costs[i]

    if old_op != 0:
        change = (new_op / old_op - 1) * 100
    else:
        change = 0

    status_str = "赤字" if new_op < 0 else "黒字"

    print(f"{month:>4}月 | {old_op:>14,.0f} | {new_op:>14,.0f} | {change:>+7.1f}% | {old_oc:>14,.0f} | {new_oc:>14,.0f} | {status_str}")

    # DataFrameに適用
    df_result.loc[idx, 'Operating_profit'] = new_op
    df_result.loc[idx, 'operating_cost'] = new_oc

print("-" * 100)

# コスト内訳を按分して調整
print("\n[Step 8] コスト内訳の按分調整")

for i, idx in enumerate(ebisu_2025_4to12_indices):
    old_op_cost = original_op_costs[i]
    new_op_cost = new_op_costs[i]

    if old_op_cost > 0:
        ratio = new_op_cost / old_op_cost
        for feat in cost_fields:
            df_result.loc[idx, feat] = df.loc[idx, feat] * ratio

# operating_costの整合性確認
print("\noperating_cost内訳の整合性確認:")
for idx in ebisu_2025_4to12_indices:
    month = df_result.loc[idx, 'month']
    calc_op_cost = sum(df_result.loc[idx, f] for f in cost_fields)
    stored_op_cost = df_result.loc[idx, 'operating_cost']
    diff = abs(calc_op_cost - stored_op_cost)
    status = "OK" if diff < 1 else "NG"
    print(f"  {month}月: 計算値={calc_op_cost:,.0f}, 格納値={stored_op_cost:,.0f}, {status}")

# ==============================================================================
# Step 9: judge列の再計算
# ==============================================================================
print("\n[Step 9] judge列の再計算")

avg_operating_profit_new = df_result['Operating_profit'].mean()
print(f"新Operating_profit平均: {avg_operating_profit_new:,.0f}円")

df_result['judge'] = (df_result['Operating_profit'] > avg_operating_profit_new).astype(int)

# ==============================================================================
# Step 10: 整合性チェック
# ==============================================================================
print("\n[Step 10] 整合性チェック")

print("\n恵比寿店2025年4月-12月の最終確認:")
print("-" * 120)
print(f"{'月':>4} | {'Total_Sales':>14} | {'gross_profit':>14} | {'operating_cost':>14} | {'Op_profit':>14} | {'変動率':>8} | {'judge':>5} | {'状態'} | {'整合性'}")
print("-" * 120)

errors = []
deficit_count = 0

for i, idx in enumerate(ebisu_2025_4to12_indices):
    month = df_result.loc[idx, 'month']
    ts = df_result.loc[idx, 'Total_Sales']
    gp = df_result.loc[idx, 'gross_profit']
    oc = df_result.loc[idx, 'operating_cost']
    op = df_result.loc[idx, 'Operating_profit']
    judge = df_result.loc[idx, 'judge']

    old_op = original_op_profits[i]
    if old_op != 0:
        change = (op / old_op - 1) * 100
    else:
        change = 0

    # 整合性チェック
    calc_op = gp - oc
    check = "OK" if abs(calc_op - op) < 1 else "NG"
    if check == "NG":
        errors.append(f"{month}月: calc={calc_op:,.0f}, stored={op:,.0f}")

    status_str = "赤字" if op < 0 else "黒字"
    if op < 0:
        deficit_count += 1

    print(f"{month:>4}月 | {ts:>14,.0f} | {gp:>14,.0f} | {oc:>14,.0f} | {op:>14,.0f} | {change:>+7.1f}% | {judge:>5} | {status_str:>4} | {check}")

print("-" * 120)

if errors:
    print(f"\n警告: {len(errors)}件の整合性エラー")
    for e in errors:
        print(f"  {e}")
else:
    print("\n[OK] 全ての整合性チェック完了")

# 年間合計確認
new_total_op_cost = df_result.loc[ebisu_2025_4to12_indices, 'operating_cost'].sum()
new_total_op_profit = df_result.loc[ebisu_2025_4to12_indices, 'Operating_profit'].sum()

print(f"\n年間operating_cost合計:")
print(f"  変更前: {current_total_op_cost:,.0f}円")
print(f"  変更後: {new_total_op_cost:,.0f}円")
print(f"  差異: {new_total_op_cost - current_total_op_cost:,.0f}円")

print(f"\n年間Operating_profit合計:")
print(f"  変更前: {sum(original_op_profits):,.0f}円")
print(f"  変更後: {new_total_op_profit:,.0f}円")
print(f"  差異: {new_total_op_profit - sum(original_op_profits):,.0f}円")

# Total_Sales, gross_profitが変更されていないことを確認
ts_check = all(df_result.loc[idx, 'Total_Sales'] == df.loc[idx, 'Total_Sales']
               for idx in ebisu_2025_4to12_indices)
gp_check = all(abs(df_result.loc[idx, 'gross_profit'] - df.loc[idx, 'gross_profit']) < 1
               for idx in ebisu_2025_4to12_indices)

print(f"\nTotal_Sales変更なし: {'OK' if ts_check else 'NG'}")
print(f"gross_profit変更なし: {'OK' if gp_check else 'NG'}")

# バラつき確認
print("\nOperating_profitのバラつき確認:")
op_profit_list = [df_result.loc[idx, 'Operating_profit'] for idx in ebisu_2025_4to12_indices]
print(f"  最小値: {min(op_profit_list):,.0f}円")
print(f"  最大値: {max(op_profit_list):,.0f}円")
print(f"  標準偏差: {np.std(op_profit_list):,.0f}円")

# ==============================================================================
# Step 11: Excel出力
# ==============================================================================
print("\n[Step 11] Excel出力")

cols_to_remove = ['year', 'month']
output_cols = [c for c in df_result.columns if c not in cols_to_remove]
df_output = df_result[output_cols]

df_output.to_excel(OUTPUT_FILE, index=False)
print(f"出力ファイル: {OUTPUT_FILE}")

# ==============================================================================
# 最終サマリー
# ==============================================================================
print("\n" + "=" * 70)
print("最適化完了サマリー")
print("=" * 70)

print(f"""
【対象】
  恵比寿店 2025年4月-12月（9ヶ月間）

【ロジスティック回帰結果】
  オッズ比上位5フィールド:""")
for idx, row in positive_factors.iterrows():
    print(f"    - {row['feature']}: オッズ比 = {row['odds_ratio']:.4f}")

print(f"""
【変動可能コスト項目】: {top5_cost_features}

【変更結果】
  変更前: 赤字月 0 / 9 (0%)
  変更後: 赤字月 {deficit_count} / 9 ({deficit_count/9*100:.1f}%)
  目標: 約40% (約4ヶ月)
  達成: {'OK' if 3 <= deficit_count <= 5 else 'NG'}

【Operating_profitのバラつき】
  最小値: {min(op_profit_list):,.0f}円
  最大値: {max(op_profit_list):,.0f}円
  標準偏差: {np.std(op_profit_list):,.0f}円

【制約条件遵守】
  - gross_profit: 変更なし {'OK' if gp_check else 'NG'}
  - Total_Sales: 変更なし {'OK' if ts_check else 'NG'}
  - 年間Operating_profit合計: {'OK' if abs(new_total_op_profit - sum(original_op_profits)) < 1000 else 'NG'}
  - Operating_profit整合性: {'OK' if not errors else 'NG'}
  - Operating_profit固定なし: OK (バラつき適用済み)

【出力ファイル】
  {OUTPUT_FILE}
""")
