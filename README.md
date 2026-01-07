# 営業利益改善AI Agentsダッシュボード

QCストーリーに基づく店舗損益分析・改善システム（Streamlit + PuLP最適化）

## 概要

恵比寿店・横浜元町店の営業利益改善を目的としたStreamlitアプリケーションです。
ロジスティック回帰による要因分析、PuLPによる数理最適化、AutoGluon TimeSeries予測結果の可視化を統合しています。

## 主要機能

| ページ | 機能 | 説明 |
|--------|------|------|
| 1_現状把握 | EDAダッシュボード | ヒストグラム、箱ひげ図、相関行列、VIF・正規性検定 |
| 2_要因分析 | ロジスティック回帰 | L1正則化によるオッズ比分析、黒字化要因TOP5抽出 |
| 3_目標設定 | 改善目標設定 | 店舗・期間選択、赤字月数目標、変動幅設定 |
| 4_最適化実行 | PuLP最適化 | 数理最適化実行、Before/After比較 |
| 5_時系列予測 | 予測可視化 | Chronos2+TFTモデルによる90日間予測、信頼区間表示 |
| 6_レポート出力 | エクスポート | 改善P/L Excel出力、分析レポート生成 |

## インストール

### 前提条件

- Python 3.11以上
- pip

### セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/YOUR_USERNAME/profit-improvement-dashboard.git
cd profit-improvement-dashboard

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 実行方法

```bash
streamlit run Home.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

### デモ認証情報

- ユーザー名: `admin`
- パスワード: `admin123`

## プロジェクト構造

```
streamlit_app/
├── Home.py                 # エントリーポイント（ログイン画面）
├── pages/                  # マルチページ
│   ├── 1_現状把握.py        # EDAダッシュボード
│   ├── 2_要因分析.py        # ロジスティック回帰
│   ├── 3_目標設定.py        # 改善目標設定
│   ├── 4_最適化実行.py      # PuLP最適化
│   ├── 5_時系列予測.py      # 予測可視化
│   └── 6_レポート出力.py    # レポートエクスポート
├── components/             # 共通コンポーネント
│   ├── auth.py             # 認証機能
│   ├── data_loader.py      # データ読み込み
│   └── charts.py           # Plotlyグラフ生成
├── utils/                  # ビジネスロジック
│   ├── logistic.py         # ロジスティック回帰
│   └── optimization.py     # PuLP最適化
├── .streamlit/             # Streamlit設定
│   ├── config.toml
│   └── secrets.toml.example
├── requirements.txt        # 依存パッケージ
├── render.yaml             # Renderデプロイ設定
└── README.md
```

## 技術スタック

- **フロントエンド**: Streamlit 1.28+
- **可視化**: Plotly, Seaborn
- **機械学習**: scikit-learn (ロジスティック回帰)
- **最適化**: PuLP (線形計画法)
- **統計**: SciPy, Statsmodels
- **時系列予測**: AutoGluon TimeSeries (事前計算結果を表示)

## デプロイ

### Renderへのデプロイ

1. GitHubリポジトリをRenderに接続
2. `render.yaml` に基づいて自動デプロイ

### 本番環境の認証設定

`.streamlit/secrets.toml` を作成:

```toml
[credentials]
usernames = ["admin", "user"]

[passwords]
admin = "your_secure_password"
user = "your_secure_password"
```

## 分析ワークフロー（QCストーリー）

```
1. テーマ選定 → 店舗・期間選択
2. 現状把握   → EDA分析（ヒストグラム、箱ひげ図、相関行列）
3. 要因分析   → ロジスティック回帰（オッズ比TOP5）
4. 目標設定   → 赤字月数目標、変動幅設定
5. 対策立案   → PuLP最適化実行
6. 効果確認   → Before/After比較、時系列予測
7. 標準化     → 改善P/Lダウンロード
```

## ライセンス

MIT License

## 作成者

Takshi.Takenouchi

---

**最終更新**: 2025年12月
