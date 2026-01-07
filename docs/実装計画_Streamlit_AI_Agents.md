# Streamlit 営業利益改善AI Agentsアプリ 実装計画

## 概要
恵比寿店・横浜元町店の営業利益改善を目的としたStreamlitアプリを開発し、GitHub連携・Renderデプロイを行う。

## プロジェクトパス
- **元データ**: `C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\Ensuring consistency between tabular data and time series forecast data`
- **新規Streamlitアプリ**: 上記ディレクトリ内 `streamlit_app/` に作成

---

## 1. アプリ構造（QCストーリーベース）

```
streamlit_app/
├── Home.py                       # ログイン + 概要ダッシュボード
├── pages/
│   ├── 1_現状把握.py              # EDAダッシュボード
│   ├── 2_要因分析.py              # ロジスティック回帰結果
│   ├── 3_目標設定.py              # 営業利益改善目標入力
│   ├── 4_最適化実行.py            # PuLP最適化実行・結果表示
│   ├── 5_時系列予測.py            # 90日間予測可視化
│   └── 6_レポート出力.py          # 改善P/Lダウンロード
├── components/
│   ├── auth.py                   # 簡易認証
│   ├── data_loader.py            # ファイルアップロード
│   └── charts.py                 # Plotlyグラフ生成
├── utils/
│   ├── logistic.py               # ロジスティック回帰ロジック
│   └── optimization.py           # PuLP最適化ロジック
├── .streamlit/
│   ├── config.toml               # Streamlit設定
│   └── secrets.toml.example      # 認証情報テンプレート
├── requirements.txt
├── render.yaml
└── README.md
```

---

## 2. ページ別機能詳細

### 2.1 Home.py（ログイン + 概要）
- streamlit-authenticatorによる簡易ログイン
- プロジェクト概要表示
- 各ページへのナビゲーション

### 2.2 現状把握（EDA）
- **入力**: Excelファイルアップロード
- **出力**:
  - ヒストグラム（単変量分析）
  - 箱ひげ図（judge別分布）
  - 相関行列ヒートマップ
  - VIF計算結果
- **ベース**: `src/analysis/task2_eda_dashboard.py`

### 2.3 要因分析（ロジスティック回帰）
- judge列作成（Operating_profit > 平均 → 1）
- L1正則化ロジスティック回帰（C=0.5）
- オッズ比表示（TOP5黒字化要因）
- 赤字要因表示（オッズ比 < 1）
- **ベース**: `src/analysis/01_logistic_regression_analysis_20251227.py`

### 2.4 目標設定
- 店舗選択（恵比寿/横浜元町/両方）
- 対象期間選択（年月）
- 赤字月数目標（スライダー: 0-9）
- 変動幅設定（±30%デフォルト）
- 制約条件プレビュー

### 2.5 最適化実行（PuLP）
- パラメータ確認表示
- 最適化実行ボタン（プログレスバー付き）
- Before/After比較表
- 月別Operating_profit変化グラフ
- **ベース**: `src/analysis/05_ebisu_optimization_vol6_20251227.py`

### 2.6 時系列予測
- 事前計算結果の読み込み（forecast_results_2026_90days.xlsx）
- 店舗・カテゴリ・アイテム選択
- 信頼区間付き予測グラフ
- CSVダウンロード
- **ベース**: `app/forecast_dashboard.py`（既存）

### 2.7 レポート出力
- 改善前後比較テーブル
- 改善P/L Excel出力
- 分析サマリーレポート生成

---

## 3. 技術仕様

### 3.1 依存パッケージ（requirements.txt）
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
openpyxl>=3.1.0
japanize-matplotlib>=1.1.3
PuLP>=2.7.0
streamlit-authenticator>=0.3.2
python-dotenv>=1.0.0
scipy>=1.11.0
statsmodels>=0.14.0
```

### 3.2 認証設定（secrets.toml）
```toml
[passwords]
admin = "admin123"
user = "user123"
```

### 3.3 Render設定（render.yaml）
```yaml
services:
  - type: web
    name: store-profit-optimizer
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run streamlit_app/Home.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.4
      - key: STREAMLIT_SERVER_HEADLESS
        value: true
    autoDeploy: true
```

---

## 4. 既存スクリプト再利用

| 既存ファイル | 新規ファイル | 変更内容 |
|-------------|-------------|---------|
| `01_logistic_regression_analysis_20251227.py` | `utils/logistic.py` | 関数化、パラメータ化 |
| `05_ebisu_optimization_vol6_20251227.py` | `utils/optimization.py` | 店舗選択対応、パラメータ化 |
| `task2_eda_dashboard.py` | `components/charts.py` | Plotlyグラフ化 |
| `forecast_dashboard.py` | `pages/5_時系列予測.py` | 統合、スタイル調整 |

---

## 5. 実装フェーズ

### Phase 1: 基盤構築
- [ ] GitHubリポジトリ作成
- [ ] Streamlitマルチページ構造作成
- [ ] 認証機能実装
- [ ] 共通コンポーネント作成

### Phase 2: 分析機能
- [ ] ファイルアップロード機能
- [ ] EDAダッシュボード
- [ ] ロジスティック回帰モジュール
- [ ] オッズ比可視化

### Phase 3: 最適化機能
- [ ] 目標設定UI
- [ ] PuLP最適化ロジック
- [ ] Before/After比較
- [ ] 結果ダウンロード

### Phase 4: 予測可視化
- [ ] 既存dashboard統合
- [ ] スタイル統一

### Phase 5: デプロイ
- [ ] requirements.txt確定
- [ ] render.yaml設定
- [ ] GitHub → Render連携
- [ ] 動作確認

---

## 6. 制約・注意事項

1. **AutoGluon除外**: Render無料枠（512MB）では動作不可 → 予測は事前計算結果を表示
2. **ファイルサイズ**: 最大200MBまで
3. **日本語パス**: ファイルアップロード方式で回避
4. **認証**: シンプルなパスワード認証（全ユーザー同一権限）

---

## 7. 成功基準

- 単純ログイン認証が機能する
- Excelアップロード → EDA表示完了
- ロジスティック回帰 → オッズ比TOP5表示
- 目標設定 → PuLP最適化 → 改善P/L生成
- 時系列予測結果可視化（既存データ）
- Render上で安定稼働
- GitHub自動デプロイ動作

---

## 8. 修正対象ファイル一覧

### 新規作成
- `streamlit_app/Home.py`
- `streamlit_app/pages/1_現状把握.py`
- `streamlit_app/pages/2_要因分析.py`
- `streamlit_app/pages/3_目標設定.py`
- `streamlit_app/pages/4_最適化実行.py`
- `streamlit_app/pages/5_時系列予測.py`
- `streamlit_app/pages/6_レポート出力.py`
- `streamlit_app/components/auth.py`
- `streamlit_app/components/data_loader.py`
- `streamlit_app/components/charts.py`
- `streamlit_app/utils/logistic.py`
- `streamlit_app/utils/optimization.py`
- `streamlit_app/.streamlit/config.toml`
- `streamlit_app/.streamlit/secrets.toml.example`
- `streamlit_app/requirements.txt`
- `streamlit_app/render.yaml`
- `streamlit_app/README.md`
- `.gitignore`

### 参照（読み取りのみ）
- `src/analysis/01_logistic_regression_analysis_20251227.py`
- `src/analysis/05_ebisu_optimization_vol6_20251227.py`
- `src/analysis/task2_eda_dashboard.py`
- `app/forecast_dashboard.py`
- `fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx`
- `output/forecast_results_2026_90days.xlsx`

---

**作成日**: 2025-12-28
**作成者**: Claude Code
