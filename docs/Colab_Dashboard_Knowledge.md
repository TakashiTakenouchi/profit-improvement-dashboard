# Google Colaboratory ダッシュボードアプリ開発ナレッジ

**作成日**: 2025-12-31
**最終更新**: 2025-12-31
**対象ファイル**: `colab/profit_improvement_dashboard_v1～v4.ipynb`

---

## 1. プロジェクト概要

### 1.1 目的
Streamlitアプリ（`streamlit_app/`）をGoogle Colaboratory上で動作する`ipywidgets`ベースのダッシュボードに移植。恵比寿店・横浜元町店の営業利益改善分析を提供。

### 1.2 技術スタック
| 技術 | 用途 |
|------|------|
| ipywidgets | インタラクティブUI（ドロップダウン、ボタン、タブ） |
| Plotly | グラフ可視化 |
| pandas | データ処理 |
| scikit-learn | ロジスティック回帰（要因分析） |
| Google Drive | データファイル格納 |

### 1.3 データファイル（Google Drive）
```
/content/drive/My Drive/Colab Notebooks/TitanicDisaster/
├── fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx  # 損益計算書（138行×37列）
├── time_series_forecast_data_2024_fixed.xlsx                       # 過去実績（99,504行）
└── forecast_results_2026_90days.xlsx                               # AutoGluon予測（4,320行）
```

---

## 2. バージョン履歴

### v1: 初版 (`profit_improvement_dashboard.ipynb`)
**作成日**: 2025-12-30

**機能**:
- 基本的なダッシュボード構造
- 4タブ構成（現状把握、要因分析、時系列予測、レポート）
- `ipywidgets.ToggleButtons`によるタブ切り替え
- 店舗・年度フィルター

**課題**:
- 時系列予測タブが何も表示されない（エラーなし）

---

### v2: AutoGluon予測表示追加 (`profit_improvement_dashboard_v2.ipynb`)
**作成日**: 2025-12-30

**変更点**:
- `forecast_results_2026_90days.xlsx`からAutoGluon予測結果を読み込み
- 予測グラフ表示機能追加
- モデル情報（WeightedEnsemble構成）を表示

**課題**:
- 時系列予測タブでグラフが表示されない

---

### v3: Plotly表示問題の修正 (`profit_improvement_dashboard_v3.ipynb`)
**作成日**: 2025-12-31

**問題の原因**:
`ipywidgets.Output()`ウィジェット内では、Plotlyの標準的な表示方法（`fig.show()`、`display(fig)`）が動作しない。

**試行錯誤の経緯**:

1. **試行1**: `fig.show()` → 表示されず
2. **試行2**: `pio.renderers.default = 'colab'` 追加 → 表示されず
3. **試行3**: `display(fig)` に変更 → 表示されず
4. **試行4**: `display(HTML(fig.to_html(include_plotlyjs='cdn', full_html=False)))` → 表示されず
5. **試行5（成功）**: **iframe + base64エンコード方式**

**最終解決策**:
```python
def show_plotly_chart(fig, height=450):
    """Google Colab + ipywidgets.Output()内でPlotlyグラフを確実に表示"""
    import base64

    # HTMLに変換（完全なHTML）
    html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)

    # base64エンコード
    b64 = base64.b64encode(html_str.encode()).decode()

    # iframeで表示
    iframe_html = f'''
    <iframe
        src="data:text/html;base64,{b64}"
        width="100%"
        height="{height}px"
        style="border:none;">
    </iframe>
    '''
    display(HTML(iframe_html))
```

**ポイント**:
- `full_html=True`で完全なHTMLドキュメントを生成
- base64エンコードしてdata URIとして埋め込み
- iframeで表示することでサンドボックス内でPlotly.jsが動作

**追加機能**:
- 過去実績データ（`time_series_forecast_data_2024_fixed.xlsx`）との連携
- `item_id`による過去実績とAutoGluon予測のリンク
- 信頼区間（10%-90%）の表示
- 上位3アイテムの予測グラフ自動表示

---

### v4: アイテム選択機能追加 (`profit_improvement_dashboard_v4.ipynb`)
**作成日**: 2025-12-31

**新機能**:

1. **アイテム選択UI**（連動ドロップダウン）
   ```python
   # 店舗選択 → カテゴリ更新 → アイテム更新
   forecast_shop_dropdown.observe(update_categories, names='value')
   forecast_category_dropdown.observe(update_items, names='value')
   ```

2. **選択ロジック**
   ```python
   items = df_forecast[
       (df_forecast['Shop'] == selected_shop) &
       (df_forecast['Category'] == selected_category)
   ]['item_id'].unique().tolist()
   ```

3. **予測統計カード**
   - 90日間予測合計
   - 日平均予測
   - 最大値 (90%)
   - 最小値 (10%)

4. **表示オプション**
   - 「全学習期間を表示」チェックボックス

**UI構成**:
```
┌─────────────────────────────────────────────────────┐
│ 🔧 アイテム選択                                      │
│ ┌─────────┐ ┌──────────────┐ ┌────────────────────┐ │
│ │店舗: ▼  │ │カテゴリ: ▼   │ │アイテム: ▼         │ │
│ └─────────┘ └──────────────┘ └────────────────────┘ │
│ ☐ 全学習期間を表示    [予測グラフ表示]              │
└─────────────────────────────────────────────────────┘
```

---

## 3. 技術的知見

### 3.1 ipywidgets + Plotly の組み合わせ

**問題**: `widgets.Output()`内でPlotlyグラフが表示されない

**原因**:
- Google ColabのOutput widgetはiframe内で動作
- Plotlyの標準レンダラーがこの環境で正しく動作しない

**解決策**: iframe + base64エンコード方式（上記参照）

### 3.2 連動ドロップダウンの実装

```python
def update_categories(change):
    """店舗変更時にカテゴリを更新"""
    selected_shop = change['new']
    categories = df[df['Shop'] == selected_shop]['Category'].unique().tolist()
    category_dropdown.options = categories
    if categories:
        category_dropdown.value = categories[0]

shop_dropdown.observe(update_categories, names='value')
```

### 3.3 item_idの構造

```
{ShopCode}_{CategoryCode}_{ItemNumber}
例: EBISU_Mens_JACKETS&OUTER2_001
```

**店舗コード変換**:
```python
shop_code_map = {'恵比寿': 'EBISU', '横浜元町': 'YOKOHAMA'}
```

### 3.4 データの紐付け

| データソース | キー | 内容 |
|-------------|------|------|
| `time_series_forecast_data_2024_fixed.xlsx` | item_id | 過去実績（ForecastQuantity） |
| `forecast_results_2026_90days.xlsx` | item_id | AutoGluon予測（predicted_quantity, 0.1, 0.9） |

---

## 4. ファイル構成

```
colab/
├── profit_improvement_dashboard.ipynb      # v1: 初版
├── profit_improvement_dashboard_v2.ipynb   # v2: AutoGluon追加
├── profit_improvement_dashboard_v3.ipynb   # v3: Plotly表示修正
└── profit_improvement_dashboard_v4.ipynb   # v4: アイテム選択機能
```

---

## 5. 使用方法

### 5.1 Google Colabでの実行

1. Google Driveに必要なExcelファイルを配置
2. ノートブックを開く
3. 「ランタイム → すべてのセルを実行」
4. Google Driveマウントの許可

### 5.2 タブ操作

| タブ | 機能 |
|------|------|
| 現状把握 | EDA（基本統計量、営業利益推移、分布） |
| 要因分析 | L1正則化ロジスティック回帰、オッズ比TOP5 |
| 時系列予測 | アイテム選択、予測グラフ、統計情報 |
| レポート | 店舗別パフォーマンスサマリー |

---

## 6. 今後の拡張ポイント

1. **CSVダウンロード機能**: 予測データのエクスポート
2. **複数アイテム比較**: 2-3アイテムを同時にグラフ表示
3. **月次集計表示**: 日次→月次の切り替え
4. **アラート機能**: 予測値が閾値を超えた場合のハイライト

---

## 7. トラブルシューティング

### 7.1 グラフが表示されない
- `show_plotly_chart()`関数を使用しているか確認
- iframe方式でない場合は表示されない

### 7.2 ドロップダウンが更新されない
- `observe()`が正しく登録されているか確認
- 初期化時に`update_categories()`を呼び出しているか確認

### 7.3 データが読み込まれない
- Google Driveのパスが正しいか確認
- ファイル名のスペルミスがないか確認

---

**作成者**: Claude Code
**参考**: `docs/Streamlit_App_Knowledge.md`
