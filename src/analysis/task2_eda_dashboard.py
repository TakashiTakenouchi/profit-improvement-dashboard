# -*- coding: utf-8 -*-
"""
Task2: ロジスティック回帰の事前分析（EDA）ダッシュボード

目的：
- ロジスティック回帰モデル構築前の探索的データ解析（EDA）
- オッズ比の解釈性を担保するための事前チェック

分析内容：
1. 単変量分析: ヒストグラム、基本統計量（平均・分散）
2. 二変量分析: ターゲット別の分布比較（箱ひげ図）
3. 多変量チェック: 相関行列（ヒートマップ）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ファイルパス
BASE_PATH = r"C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\Ensuring consistency between tabular data and time series forecast data"
TARGET_FILE = f"{BASE_PATH}\\fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx"
OUTPUT_DIR = f"{BASE_PATH}\\output\\eda"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("ロジスティック回帰 事前分析（EDA）ダッシュボード")
print("=" * 80)

# ==============================================================================
# データ読み込み
# ==============================================================================
print("\n[Step 1] データ読み込み")
df = pd.read_excel(TARGET_FILE)
print(f"データサイズ: {df.shape[0]}行 x {df.shape[1]}列")

# 目的変数
target_col = 'judge'
print(f"目的変数: {target_col}")
print(f"judge分布:\n{df[target_col].value_counts()}")

# 説明変数の定義（除外するカラム）
exclude_cols = [
    'shop', 'shop_code', 'Date',
    'Operating_profit', 'gross_profit', 'operating_cost',
    'judge', 'Total_Sales', 'discount', 'purchasing'
]

# 数値型の説明変数を抽出
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in exclude_cols]
print(f"\n説明変数: {len(feature_cols)}個")

# ==============================================================================
# 1. 単変量分析: 基本統計量
# ==============================================================================
print("\n" + "=" * 80)
print("[Step 2] 単変量分析: 基本統計量")
print("=" * 80)

# 基本統計量の計算
stats_df = pd.DataFrame({
    '変数名': feature_cols,
    '平均': [df[col].mean() for col in feature_cols],
    '標準偏差': [df[col].std() for col in feature_cols],
    '最小値': [df[col].min() for col in feature_cols],
    '25%': [df[col].quantile(0.25) for col in feature_cols],
    '中央値': [df[col].median() for col in feature_cols],
    '75%': [df[col].quantile(0.75) for col in feature_cols],
    '最大値': [df[col].max() for col in feature_cols],
    '歪度': [df[col].skew() for col in feature_cols],
    '尖度': [df[col].kurtosis() for col in feature_cols],
    '欠損率%': [df[col].isna().sum() / len(df) * 100 for col in feature_cols]
})

print("\n基本統計量:")
print(stats_df.to_string(index=False))

# 外れ値の検出（IQR法）
print("\n\n外れ値検出（IQR法）:")
outlier_summary = []
for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    if len(outliers) > 0:
        outlier_summary.append({
            '変数名': col,
            '外れ値件数': len(outliers),
            '外れ値率%': len(outliers) / len(df) * 100,
            '下限': lower,
            '上限': upper
        })

if outlier_summary:
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.to_string(index=False))
else:
    print("外れ値は検出されませんでした")

# ==============================================================================
# 2. ヒストグラムと確率分布
# ==============================================================================
print("\n" + "=" * 80)
print("[Step 3] ヒストグラムと確率分布の確認")
print("=" * 80)

# 重要な変数のみをプロット（上位10個）
important_features = feature_cols[:10]

fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.flatten()

for i, col in enumerate(important_features):
    if i < len(axes):
        ax = axes[i]

        # ヒストグラム
        data = df[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')

        # 正規分布フィット
        mu, std = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 100)
        ax.axvline(mu, color='red', linestyle='--', label=f'Mean: {mu:.2f}')

        ax.set_title(f'{col}\nSkew: {data.skew():.2f}, Kurt: {data.kurtosis():.2f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)

# 余ったaxesを非表示
for i in range(len(important_features), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\01_histograms.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"ヒストグラム保存: {OUTPUT_DIR}\\01_histograms.png")

# ==============================================================================
# 3. 二変量分析: ターゲット別の分布比較
# ==============================================================================
print("\n" + "=" * 80)
print("[Step 4] ターゲット（judge）別の分布比較")
print("=" * 80)

# judge別の統計量比較
print("\njudge別の平均値比較:")
comparison_df = pd.DataFrame()
for col in feature_cols:
    group0 = df[df['judge'] == 0][col]
    group1 = df[df['judge'] == 1][col]

    # t検定
    t_stat, p_value = stats.ttest_ind(group0.dropna(), group1.dropna())

    comparison_df = pd.concat([comparison_df, pd.DataFrame({
        '変数名': [col],
        'judge=0 平均': [group0.mean()],
        'judge=1 平均': [group1.mean()],
        '差異': [group1.mean() - group0.mean()],
        '差異率%': [(group1.mean() - group0.mean()) / group0.mean() * 100 if group0.mean() != 0 else 0],
        't統計量': [t_stat],
        'p値': [p_value],
        '有意': ['***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '']
    })])

comparison_df = comparison_df.reset_index(drop=True)
print(comparison_df.to_string(index=False))

# 有意な変数（オッズ比が効きそうな変数）
significant_vars = comparison_df[comparison_df['p値'] < 0.05]['変数名'].tolist()
print(f"\n有意な変数（p<0.05）: {len(significant_vars)}個")
for v in significant_vars:
    row = comparison_df[comparison_df['変数名'] == v].iloc[0]
    print(f"  - {v}: 差異率={row['差異率%']:.1f}%, p={row['p値']:.4f}")

# 箱ひげ図
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(important_features):
    if i < len(axes):
        ax = axes[i]

        # judge別の箱ひげ図
        df.boxplot(column=col, by='judge', ax=ax)
        ax.set_title(f'{col}')
        ax.set_xlabel('judge')
        ax.set_ylabel('Value')

# 余ったaxesを非表示
for i in range(len(important_features), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Target (judge) Conditional Distribution', fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\02_boxplots_by_judge.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n箱ひげ図保存: {OUTPUT_DIR}\\02_boxplots_by_judge.png")

# ==============================================================================
# 4. 多変量チェック: 相関行列
# ==============================================================================
print("\n" + "=" * 80)
print("[Step 5] 相関行列（多重共線性チェック）")
print("=" * 80)

# 相関行列
corr_matrix = df[feature_cols].corr()

# 高相関ペアの抽出（|r| > 0.7）
high_corr_pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            high_corr_pairs.append({
                '変数1': feature_cols[i],
                '変数2': feature_cols[j],
                '相関係数': r
            })

print("\n高相関ペア（|r| > 0.7）:")
if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values('相関係数', ascending=False)
    print(high_corr_df.to_string(index=False))
    print(f"\n警告: {len(high_corr_pairs)}組の高相関ペアがあります。多重共線性に注意してください。")
else:
    print("高相関ペアはありません")

# ヒートマップ
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, ax=ax,
            annot_kws={'size': 8})
ax.set_title('Correlation Matrix (Feature Variables)', fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\03_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n相関行列ヒートマップ保存: {OUTPUT_DIR}\\03_correlation_heatmap.png")

# ==============================================================================
# 5. VIF（分散拡大係数）の計算
# ==============================================================================
print("\n" + "=" * 80)
print("[Step 6] VIF（分散拡大係数）の計算")
print("=" * 80)

from sklearn.linear_model import LinearRegression

def calculate_vif(X):
    """VIFを計算する関数"""
    vif_data = []
    for i, col in enumerate(X.columns):
        X_i = X.drop(columns=[col])
        y_i = X[col]

        # 欠損値を除去
        mask = ~(X_i.isna().any(axis=1) | y_i.isna())
        X_clean = X_i[mask]
        y_clean = y_i[mask]

        if len(X_clean) > 0:
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            r_squared = model.score(X_clean, y_clean)
            vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
        else:
            vif = float('nan')

        vif_data.append({'変数名': col, 'VIF': vif})

    return pd.DataFrame(vif_data)

# 主要な変数でVIF計算
vif_cols = [col for col in feature_cols if not col.endswith('R')]  # 回転率系を除外
X_for_vif = df[vif_cols].dropna()

if len(X_for_vif) > 10:
    vif_df = calculate_vif(X_for_vif)
    vif_df = vif_df.sort_values('VIF', ascending=False)
    print("\nVIF（高い順）:")
    print(vif_df.to_string(index=False))

    # VIF > 10 の変数を警告
    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        print(f"\n警告: VIF > 10 の変数が {len(high_vif)}個あります（多重共線性の疑い）")
        for _, row in high_vif.iterrows():
            print(f"  - {row['変数名']}: VIF = {row['VIF']:.2f}")
else:
    print("VIF計算に十分なデータがありません")

# ==============================================================================
# 6. 分布の正規性チェック
# ==============================================================================
print("\n" + "=" * 80)
print("[Step 7] 分布の正規性チェック（Shapiro-Wilk検定）")
print("=" * 80)

normality_results = []
for col in feature_cols:
    data = df[col].dropna()
    if len(data) > 5000:
        # サンプルが大きい場合はサンプリング
        data = data.sample(5000, random_state=42)

    if len(data) >= 3:
        stat, p_value = stats.shapiro(data)
        normality_results.append({
            '変数名': col,
            'Shapiro統計量': stat,
            'p値': p_value,
            '正規性': 'Yes' if p_value > 0.05 else 'No',
            '推奨': '変換不要' if p_value > 0.05 else '対数変換検討'
        })

normality_df = pd.DataFrame(normality_results)
print("\n正規性検定結果:")
print(normality_df.to_string(index=False))

non_normal = normality_df[normality_df['正規性'] == 'No']
print(f"\n正規性を満たさない変数: {len(non_normal)}個")

# ==============================================================================
# 7. 推奨事項のサマリー
# ==============================================================================
print("\n" + "=" * 80)
print("[Step 8] EDA推奨事項サマリー")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────┐
│  ロジスティック回帰 事前分析（EDA）推奨事項                        │
└─────────────────────────────────────────────────────────────────┘

1. 外れ値の処理
""")
if outlier_summary:
    print(f"   - {len(outlier_summary)}個の変数に外れ値あり")
    print("   - 対処: Winsorization または 対数変換を検討")
else:
    print("   - 外れ値なし")

print("""
2. 有意な説明変数（オッズ比が効きそう）
""")
if significant_vars:
    for v in significant_vars[:5]:
        row = comparison_df[comparison_df['変数名'] == v].iloc[0]
        print(f"   - {v}: judge別差異={row['差異率%']:.1f}%")
else:
    print("   - 有意な変数なし")

print("""
3. 多重共線性の警告
""")
if high_corr_pairs:
    print(f"   - {len(high_corr_pairs)}組の高相関ペア（|r|>0.7）")
    print("   - 対処: 一方の変数を削除、またはPCA/正則化を検討")
else:
    print("   - 問題なし")

print("""
4. 推奨する分析フロー
   Step 1: 外れ値処理（Winsorization）
   Step 2: スケーリング（StandardScaler）
   Step 3: 高相関変数の削除またはPCA
   Step 4: L1正則化ロジスティック回帰の実行
   Step 5: オッズ比の解釈と妥当性検証
""")

# ==============================================================================
# 結果保存
# ==============================================================================
print("\n" + "=" * 80)
print("[Step 9] 結果ファイルの保存")
print("=" * 80)

# 統計量をExcelに保存
with pd.ExcelWriter(f"{OUTPUT_DIR}\\eda_summary.xlsx", engine='openpyxl') as writer:
    stats_df.to_excel(writer, sheet_name='基本統計量', index=False)
    comparison_df.to_excel(writer, sheet_name='judge別比較', index=False)
    if high_corr_pairs:
        high_corr_df.to_excel(writer, sheet_name='高相関ペア', index=False)
    if 'vif_df' in dir():
        vif_df.to_excel(writer, sheet_name='VIF', index=False)
    normality_df.to_excel(writer, sheet_name='正規性検定', index=False)

print(f"EDAサマリー保存: {OUTPUT_DIR}\\eda_summary.xlsx")

print("""
================================================================================
出力ファイル一覧
================================================================================
""")
print(f"1. {OUTPUT_DIR}\\01_histograms.png")
print(f"2. {OUTPUT_DIR}\\02_boxplots_by_judge.png")
print(f"3. {OUTPUT_DIR}\\03_correlation_heatmap.png")
print(f"4. {OUTPUT_DIR}\\eda_summary.xlsx")

print("\n完了")
