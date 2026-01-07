# 店舗別損益計算書データ分析 - ロジスティック回帰による営業利益改善

## 概要
店舗別損益計算書データに対してロジスティック回帰分析を実行し、営業利益の増減要因を明確化。恵比寿店の黒字化率を14.5%から60.9%に改善。

---

## 1. データ準備

### 使用データ
- **入力ファイル**: `fixed_extended_store_data_2024-FIX_kaizen_monthlyvol3.xlsx`
- **出力ファイル**: `fixed_extended_store_data_2024-FIX_kaizen_monthlyvol4.xlsx`
- **データ項目定義**: `データ項目定義.xlsx`

### 重要な計算式
```
gross_profit = Total_Sales - purchasing - discount
operating_cost = rent + personnel_expenses + depreciation + sales_promotion + head_office_expenses
Operating_profit = gross_profit - operating_cost
```

### judge列の定義
- 営業利益（Operating_profit）の全体平均値: **-1,675,956円**
- `judge = 1`: 営業利益 > 平均値（黒字傾向）
- `judge = 0`: 営業利益 <= 平均値（赤字傾向）

---

## 2. ロジスティック回帰分析

### 説明変数の選定ルール
以下を説明変数から**除外**:
- `shop`: 店舗名称（カテゴリ変数）
- `shop_code`: 店舗コード（カテゴリ変数）
- `Date`: 日付（時間変数）
- `Operating_profit`: 目的変数（自己矛盾回避）
- `gross_profit`: 結果変数（Total_Sales, discount, purchasingから算出される）
- `operating_cost`: 合計値（rent等の合計）

### モデル設定
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# L1正則化ロジスティック回帰
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
```

---

## 3. 分析結果

### 黒字化要因TOP5（オッズ比 > 1）

| 順位 | 変数 | オッズ比 | 解釈 |
|------|------|---------|------|
| 1 | WOMEN'S_TOPS（レディーストップス売上） | 4.75 | 1標準偏差増加で黒字確率が約4.75倍 |
| 2 | Number_of_guests（客数） | 1.62 | 客数増加が黒字化に大きく貢献 |
| 3 | Mens_KNIT（メンズニット売上） | 1.42 | ニット売上の増加が効果的 |
| 4 | WOMEN'S_SCARF & STOLES（スカーフ・ストール売上） | 1.42 | アクセサリー売上の増加 |
| 5 | Mens_PANTS（メンズパンツ売上） | 1.42 | パンツ売上の増加 |

### 赤字要因（オッズ比 < 1、抑制すべき項目）

| 変数 | オッズ比 | 解釈 |
|------|---------|------|
| personnel_expenses（人件費） | 0.25 | 1標準偏差増加で黒字確率が75%低下 |
| WOMEN'S_JACKETSR | 0.76 | 回転率の問題 |
| WOMEN'S_bottomsR | 0.80 | 回転率の問題 |
| Price_per_customer（客単価） | 0.80 | 高すぎる客単価は逆効果 |

---

## 4. 恵比寿店改善プロセス

### 改善前の状況
- 店舗コード: 11
- 期間: 2020/4/30 - 2025/12/31（69ヶ月）
- 黒字月: 10ヶ月
- 赤字月: 59ヶ月
- **黒字率: 14.5%**

### 黒字月 vs 赤字月の比較

| 変数 | 黒字月平均 | 赤字月平均 | 差異 |
|------|-----------|-----------|------|
| Total_Sales | 31,210,998 | 11,253,713 | +19,957,285 |
| WOMEN'S_TOPS | 2,932,029 | 1,130,303 | +1,801,726 |
| Number_of_guests | 983 | 329 | +654 |
| Mens_KNIT | 1,780,040 | 636,146 | +1,143,894 |
| personnel_expenses | 2,778,000 | 2,856,579 | -78,578 |

### 改善戦略
1. **売上カテゴリの引き上げ**: 黒字月水準まで増加
2. **値引き率の改善**: 8.9%（黒字月水準）
3. **仕入れ率の改善**: 43.8%（黒字月水準）
4. **人件費の削減**: 黒字月水準まで削減
5. **客数の増加**: 黒字月水準まで引き上げ

### 黒字化に必要な売上計算
```python
# 粗利益率
gross_margin = 1 - avg_discount_rate - avg_purchasing_rate  # 約47.3%

# 黒字化に必要な最低売上
min_sales_for_surplus = operating_cost / gross_margin
```

### 改善後の結果
- 黒字月: **42ヶ月**
- 赤字月: 27ヶ月
- **黒字率: 60.9%**（目標60%達成）

---

## 5. 実装コード（改善適用部分）

```python
# 赤字月を赤字額が少ない順にソート
ebisu_deficit_sorted = ebisu_deficit.sort_values('Operating_profit', ascending=False)

for idx in ebisu_deficit_sorted.index:
    original = df.loc[idx].copy()
    current_operating_cost = original['operating_cost']

    # 黒字化に必要な最低売上
    min_sales_for_surplus = current_operating_cost / gross_margin
    target_sales = max(min_sales_for_surplus, target_total_sales * 0.9)

    # 売上が足りない場合は引き上げ
    if df_improved.loc[idx, 'Total_Sales'] < target_sales:
        scale = target_sales / df_improved.loc[idx, 'Total_Sales']
        for col in category_cols:
            df_improved.loc[idx, col] = df_improved.loc[idx, col] * scale
        df_improved.loc[idx, 'Total_Sales'] = target_sales

    # 値引き・仕入れを黒字月の比率で再計算
    df_improved.loc[idx, 'discount'] = df_improved.loc[idx, 'Total_Sales'] * avg_discount_rate
    df_improved.loc[idx, 'purchasing'] = df_improved.loc[idx, 'Total_Sales'] * avg_purchasing_rate

    # gross_profitを再計算
    df_improved.loc[idx, 'gross_profit'] = (
        df_improved.loc[idx, 'Total_Sales'] -
        df_improved.loc[idx, 'purchasing'] -
        df_improved.loc[idx, 'discount']
    )

    # 人件費を黒字月水準に削減
    if df_improved.loc[idx, 'personnel_expenses'] > target_personnel:
        df_improved.loc[idx, 'personnel_expenses'] = target_personnel

    # operating_cost、Operating_profitを再計算
    # ...
```

---

## 6. 次のステップ

### 6.1 時系列予測データとの整合性確保
- 改善後のデータ（vol4）と時系列予測モデルの整合性を検証
- 予測精度への影響を評価
- 季節性・トレンドの再検証

### 6.2 Q-Storm改善AI Agentsの開発
- ロジスティック回帰結果を活用した自動改善提案エンジン
- 店舗別・月別の最適改善パラメータ算出
- What-Ifシミュレーション機能
- リアルタイムモニタリングダッシュボード

---

## 7. 注意事項

1. **gross_profitは結果変数**: Total_Sales、discount、purchasingから算出されるため、直接操作するパラメータではない
2. **多重共線性の回避**: operating_cost（合計値）は説明変数から除外
3. **L1正則化の活用**: 変数選択を自動化し、重要な変数のみを抽出
4. **改善の現実性**: 売上増加率は黒字月水準を参考に設定

---

## 8. ファイル構成

```
Ensuring consistency between tabular data and time series forecast data/
├── .claude/                                                      # Claude Codeナレッジ
│   ├── knowledge_project_structure.md                            # 共通プロジェクト構成ルール
│   └── （プロジェクト固有ナレッジ）
├── src/                                                          # ソースコード
│   └── analysis/
│       ├── 01_logistic_regression_analysis_20251227.py           # ロジスティック回帰分析
│       ├── 02_ebisu_store_improvement_20251227.py                # 恵比寿店改善プロセス
│       ├── 03_ebisu_profit_variance_20251227.py                  # 営業利益バラつき適用
│       └── 04_complete_improvement_vol5_20251227.py              # 完全改善プロセス（vol5出力）
├── output/
│   └── data/                                                     # 出力データ格納先
├── logs/                                                         # 実行ログ
├── fixed_extended_store_data_2024-FIX_kaizen_monthlyvol3.xlsx    # 入力データ
├── fixed_extended_store_data_2024-FIX_kaizen_monthlyvol4.xlsx    # 改善後データ（出力）
├── fixed_extended_store_data_2024-FIX_kaizen_monthlyvol5.xlsx    # 最終版（バラつき適用済み）
├── データ項目定義.xlsx                                            # データ定義
└── knowledge_logistic_regression_improvement.md                   # 本ドキュメント
```

---

## 9. Pythonコード

### 9.1 ロジスティック回帰分析
**ファイル**: `src/analysis/01_logistic_regression_analysis_20251227.py`

```python
# 主要機能
- load_data(): Excelファイル読み込み
- create_judge_column(): judge列の作成（営業利益が平均以上なら1、未満なら0）
- run_logistic_regression(): L1正則化ロジスティック回帰の実行
- display_results(): オッズ比の表示
```

### 9.2 恵比寿店改善プロセス
**ファイル**: `src/analysis/02_ebisu_store_improvement_20251227.py`

```python
# 主要機能
- load_data(): Excelファイル読み込み
- analyze_ebisu_store(): 恵比寿店の現状分析
- get_surplus_characteristics(): 黒字月の特徴値取得
- apply_improvements(): 改善適用（黒字化要因TOP5を使用）
- verify_results(): 結果検証
- save_to_excel(): Excel出力
```

### 9.3 営業利益バラつき適用
**ファイル**: `src/analysis/03_ebisu_profit_variance_20251227.py`

```python
# 主要機能
- generate_normalized_coefficients(): ±30%範囲で正規化されたランダム係数を生成
- apply_variance_with_consistency(): データ整合性を維持しながらバラつきを適用
- verify_results(): 整合性チェック（OK/NG表示）

# アルゴリズム
1. 係数の生成: 0.7〜1.3の範囲でランダムな数値を12個生成
2. 正規化: 生成した12個の数値の合計が「12」になるように調整
3. 適用: 元の固定値に正規化した各係数を掛け合わせる
4. 整合性維持: gross_profit, Total_Sales, purchasing, discountを連動調整
```

### 9.4 完全改善プロセス（推奨）
**ファイル**: `src/analysis/04_complete_improvement_vol5_20251227.py`

```python
# 主要機能（1ファイルで全処理を実行）
- load_and_create_judge(): データ読み込み + judge列作成
- run_logistic_regression(): ロジスティック回帰分析
- get_top5_factors(): 黒字化要因TOP5を抽出
- improve_ebisu_store(): 恵比寿店の黒字化改善
- apply_variance(): ±30%バラつき適用
- verify_and_save(): 最終確認とExcel出力

# 出力ファイル
fixed_extended_store_data_2024-FIX_kaizen_monthlyvol5.xlsx
```

### 9.5 実行方法
```bash
cd "C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\Ensuring consistency between tabular data and time series forecast data"

# 個別実行
python src/analysis/01_logistic_regression_analysis_20251227.py
python src/analysis/02_ebisu_store_improvement_20251227.py
python src/analysis/03_ebisu_profit_variance_20251227.py

# 一括実行（推奨）
python src/analysis/04_complete_improvement_vol5_20251227.py
```

---

## 10. 営業利益バラつき適用の詳細

### 10.1 問題点
改善後の恵比寿店2025年データは、営業利益が毎月 **1,959,466円** の固定値となり非現実的。

### 10.2 解決策
±30%のランダムな分散を加え、12ヶ月間の合計金額は変更しない。

### 10.3 アルゴリズム（数学的表現）

```
Step 1: 係数生成
    c_i ∈ [0.7, 1.3]  (i = 1, 2, ..., 12)

Step 2: 正規化
    c'_i = c_i × (12 / Σc_i)
    ※ Σc'_i = 12 を保証

Step 3: 適用
    新営業利益_i = 固定値 × c'_i
    ※ Σ(新営業利益) = 固定値 × 12 = 年間合計（保存）
```

### 10.4 データ整合性維持

営業利益を変更する際、以下の計算式を維持する必要がある：

```
Operating_profit = gross_profit - operating_cost
gross_profit = Total_Sales - purchasing - discount
```

**整合性維持の手順:**

1. 営業利益の変動額を計算: `profit_delta = new_profit - old_profit`
2. 粗利益を同額増減: `new_gross = old_gross + profit_delta`
3. 粗利益率を維持して売上を調整: `new_total_sales = new_gross / old_gross_margin`
4. 売上比率を計算: `sales_ratio = new_total_sales / old_total_sales`
5. カテゴリ売上、purchasing、discountを比例調整
6. gross_profitとOperating_profitを再計算

### 10.5 適用結果（2025年恵比寿店）

| 月 | 営業利益 | 係数 | 整合性 |
|----|----------|------|--------|
| 1月 | 1,794,748円 | 0.916 | OK |
| 2月 | 2,465,708円 | 1.258 | OK |
| 3月 | 2,211,006円 | 1.128 | OK |
| 4月 | 2,055,736円 | 1.049 | OK |
| 5月 | 1,540,278円 | 0.786 | OK |
| 6月 | 1,540,250円 | 0.786 | OK |
| 7月 | 1,426,232円 | 0.728 | OK |
| 8月 | 2,367,263円 | 1.208 | OK |
| 9月 | 2,058,597円 | 1.051 | OK |
| 10月 | 2,183,150円 | 1.114 | OK |
| 11月 | 1,382,564円 | 0.706 | OK |
| 12月 | 2,488,061円 | 1.270 | OK |

- **年間合計**: 23,513,593円（変更なし）
- **最小値**: 1,382,564円（70.6%）
- **最大値**: 2,488,061円（127.0%）
- **標準偏差**: 390,527円

---

## 11. 最終結果（vol5）

### 11.1 恵比寿店の改善結果

| 項目 | 改善前 | 改善後 |
|------|--------|--------|
| 総月数 | 69 | 69 |
| 黒字月（Operating_profit >= 0） | 10 | 42 |
| 赤字月 | 59 | 27 |
| 黒字率 | 14.5% | **60.9%** |
| judge=1（営業利益 > 平均） | - | 42 (60.9%) |

### 11.2 年度別内訳

| 年 | judge=1 | 割合 |
|----|---------|------|
| 2020年 | 9/9ヶ月 | 100.0% |
| 2021年 | 5/12ヶ月 | 41.7% |
| 2022年 | 11/12ヶ月 | 91.7% |
| 2023年 | 0/12ヶ月 | 0.0% |
| 2024年 | 5/12ヶ月 | 41.7% |
| 2025年 | 12/12ヶ月 | 100.0% |

---

## 12. Claude Code プロジェクト構成ルール

### 12.1 必須プロセス（毎回実行）

1. **フォルダ構成の事前定義**
   - input/: 入力データ
   - output/: 出力データ
   - src/: ソースコード
   - .claude/: ナレッジファイル

2. **Pythonコードのローカル保存**
   - Bash内での直接実行（`python -c`）は禁止
   - ファイル命名規則: `{番号}_{処理内容}_{日付}.py`

3. **ナレッジファイルの更新**
   - プロジェクト完了時に結果をMDファイルに記録

### 12.2 共通ナレッジファイル
**パス**: `.claude/knowledge_project_structure.md`

詳細なプロジェクト構成ルール、テンプレート、チェックリストを記載。

---

## 13. 処理経緯のログ

### 2025年12月27日

1. **初期分析**: vol3からロジスティック回帰を実行、黒字化要因TOP5を特定
2. **恵比寿店改善**: 黒字率14.5% → 60.9%達成（vol4出力）
3. **問題発覚**: 2025年の営業利益が固定値（1,959,466円）で非現実的
4. **バラつき適用**: ±30%のランダム分散を適用
5. **整合性問題**: 最初はOperating_profitのみ変更し、整合性NGが発生
6. **整合性解決**: gross_profit, Total_Sales, purchasing, discountを連動調整
7. **最終版出力**: vol5として整合性OKのデータを出力

---

**作成日**: 2025年12月27日
**最終更新**: 2025年12月27日
**分析手法**: ロジスティック回帰（L1正則化）
**目的**: 営業利益の増減要因分析と恵比寿店の黒字化改善
**出力ファイル**: fixed_extended_store_data_2024-FIX_kaizen_monthlyvol5.xlsx
