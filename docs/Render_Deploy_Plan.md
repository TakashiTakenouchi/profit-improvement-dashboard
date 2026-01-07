# Streamlit アプリ Render デプロイ計画書

**作成日**: 2025-12-31
**対象リポジトリ**: https://github.com/TakashiTakenouchi/profit-improvement-dashboard

---

## 1. 現状確認

### 1.1 完了済み項目

| 項目 | 状態 | 備考 |
|------|------|------|
| Streamlitアプリ実装 | ✅ 完了 | `streamlit_app/` |
| Gitリポジトリ初期化 | ✅ 完了 | `.git/` 存在 |
| GitHub push | ✅ 完了 | `origin/main` |
| render.yaml | ✅ 存在 | 設定確認必要 |
| requirements.txt | ✅ 存在 | 14パッケージ |
| README.md | ✅ 存在 | |

### 1.2 GitHubリポジトリ情報

```
リポジトリ: TakashiTakenouchi/profit-improvement-dashboard
ブランチ: main
最新コミット: 8600654 "Initial commit: 営業利益改善AI Agentsダッシュボード"
```

### 1.3 ファイル構成

```
streamlit_app/
├── Home.py                     # エントリーポイント
├── pages/
│   ├── 1_現状把握.py
│   ├── 2_要因分析.py
│   ├── 3_目標設定.py
│   ├── 4_最適化実行.py
│   ├── 5_時系列予測.py
│   └── 6_レポート出力.py
├── components/
│   ├── auth.py                 # 認証
│   ├── data_loader.py          # データ読み込み
│   └── charts.py               # Plotlyグラフ
├── utils/
│   ├── logistic.py             # ロジスティック回帰
│   └── optimization.py         # PuLP最適化
├── .streamlit/
│   ├── config.toml             # テーマ設定
│   └── secrets.toml.example    # 認証テンプレート
├── render.yaml                 # Renderデプロイ設定
├── requirements.txt            # 依存パッケージ
├── .gitignore
└── README.md
```

---

## 2. デプロイTASK一覧

### TASK 1: render.yaml 設定確認・修正
**優先度**: 高
**所要時間**: 5分

**確認項目**:
- startCommand が正しいか
- 環境変数設定
- プラン設定（free）

**期待する設定**:
```yaml
services:
  - type: web
    name: profit-improvement-dashboard
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run Home.py --server.port $PORT --server.address 0.0.0.0
    autoDeploy: true
    envVars:
      - key: PYTHON_VERSION
        value: "3.10"
```

---

### TASK 2: サンプルデータ追加
**優先度**: 高
**所要時間**: 10分

**問題**:
Render上ではユーザーのローカルファイルにアクセスできない。
デモ用のサンプルデータをリポジトリに含める必要がある。

**対応**:
1. `data/` フォルダを作成
2. サンプルExcelファイルを配置
   - `sample_store_data.xlsx`（損益計算書サンプル）
   - `sample_forecast_results.xlsx`（予測結果サンプル）
3. `.gitignore` でユーザーアップロードデータは除外

**ファイルサイズ注意**:
- GitHubの推奨: 50MB以下/ファイル
- 大きい場合はデータを間引く

---

### TASK 3: data_loader.py 修正
**優先度**: 高
**所要時間**: 10分

**修正内容**:
- サンプルデータのパスを追加
- Render環境でのパス解決

```python
# 修正例
import os

def get_sample_data_path():
    """サンプルデータのパスを取得"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_dir, 'data', 'sample_store_data.xlsx')
```

---

### TASK 4: 環境変数設定（secrets対応）
**優先度**: 中
**所要時間**: 5分

**Renderでの設定**:
1. Render Dashboard → Environment
2. 以下を追加:
   - `ADMIN_PASSWORD`: 本番用パスワード

**コード修正**（auth.py）:
```python
import os

def get_credentials():
    # Render環境変数から取得
    admin_pw = os.environ.get('ADMIN_PASSWORD', 'admin123')
    return {"admin": admin_pw}
```

---

### TASK 5: GitHubにpush
**優先度**: 高
**所要時間**: 5分

**コマンド**:
```bash
cd streamlit_app
git add .
git commit -m "Add sample data and Render deployment config"
git push origin main
```

---

### TASK 6: Renderでデプロイ
**優先度**: 高
**所要時間**: 10分

**手順**:
1. https://render.com にログイン
2. "New" → "Web Service"
3. GitHubリポジトリを接続
   - `TakashiTakenouchi/profit-improvement-dashboard`
4. 設定確認:
   - Name: `profit-improvement-dashboard`
   - Region: Singapore (近い方)
   - Branch: `main`
   - Root Directory: （空白 or `streamlit_app`）
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run Home.py --server.port $PORT --server.address 0.0.0.0`
5. "Create Web Service" クリック
6. デプロイ完了を待つ（5-10分）

---

### TASK 7: 動作確認
**優先度**: 高
**所要時間**: 5分

**確認項目**:
- [ ] アプリにアクセスできる
- [ ] ログインできる
- [ ] サンプルデータが読み込まれる
- [ ] 各ページが表示される
- [ ] グラフが表示される

---

## 3. 重要な考慮事項

### 3.1 メモリ制限
| プラン | メモリ | CPU |
|--------|--------|-----|
| Free | 512MB | 0.1 |
| Starter | 512MB | 0.5 |

**対策**:
- AutoGluonは除外済み（予測は事前計算結果を表示）
- 大きなDataFrameはキャッシュ活用

### 3.2 スリープ問題（無料プラン）
- 15分間アクセスがないとスリープ
- 再起動に30-60秒かかる

### 3.3 日本語ファイル名
- `pages/` の日本語ファイル名はGitで正常に処理される
- Render上でも問題なし

---

## 4. デプロイ後のURL

```
https://profit-improvement-dashboard.onrender.com
```
（実際のURLはRenderが自動生成）

---

## 5. トラブルシューティング

### 5.1 デプロイ失敗
- Render Dashboardの "Logs" を確認
- `requirements.txt` のパッケージバージョン確認

### 5.2 アプリが起動しない
- `startCommand` の確認
- `Home.py` のパスが正しいか

### 5.3 データが読み込めない
- `data/` フォルダがコミットされているか
- パスの指定が相対パスになっているか

---

**次のアクション**: TASK 1 から順に実行

**作成者**: Claude Code
**関連ドキュメント**:
- `docs/Streamlit_App_Knowledge.md`
- `docs/Colab_Dashboard_Knowledge.md`
