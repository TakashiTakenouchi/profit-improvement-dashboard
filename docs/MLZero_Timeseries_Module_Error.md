# MLZero autogluon.timeseries モジュールエラー解決方法

## エラー内容

MLZeroを実行した際に以下のエラーが発生：

```
ModuleNotFoundError: No module named 'autogluon.timeseries'
```

## エラーの詳細

- **発生箇所**: MLZeroが生成したPythonコードの実行時
- **原因**: `autogluon.timeseries`パッケージがMLZeroの実行環境にインストールされていない
- **影響**: 時系列予測タスクが実行できない

## 解決方法

### ステップ1: パッケージのインストール確認

WSL環境で以下を実行して、パッケージがインストールされているか確認：

```bash
wsl
python3 -c "import autogluon.timeseries; print('✓ autogluon.timeseries インストール済み')"
```

### ステップ2: パッケージのインストール

パッケージがインストールされていない場合、以下を実行：

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
pip3 install autogluon.timeseries
```

### ステップ3: インストール確認

```bash
wsl
python3 -c "import autogluon.timeseries; print('OK')"
```

`OK`と表示されればインストール成功です。

### ステップ4: MLZeroの再実行

パッケージをインストールした後、MLZeroでタスクを再実行してください。

## トラブルシューティング

### 問題1: インストールしてもエラーが続く

**原因**: MLZeroが別のPython環境を使用している可能性があります。

**解決方法**:

```bash
# MLZeroが使用しているPythonのパスを確認
wsl bash -c "which python3"

# そのPython環境に直接インストール
wsl bash -c "python3 -m pip install autogluon.timeseries"
```

### 問題2: 複数のPython環境がある

**解決方法**: システム全体にインストール：

```bash
wsl
pip3 install --user autogluon.timeseries
```

### 問題3: バージョンの競合

**解決方法**: 特定のバージョンを指定：

```bash
wsl
pip3 install autogluon.timeseries==1.0.0
```

## 予防策

### 事前インストール

MLZeroを使用する前に、必要なパッケージを事前にインストール：

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
pip3 install autogluon.timeseries autogluon.tabular autogluon.multimodal
```

### requirements.txtの作成

プロジェクトの依存関係を管理：

```bash
# requirements.txtを作成
cat > requirements.txt << EOF
autogluon.timeseries>=1.0.0
autogluon.tabular>=1.0.0
autogluon.assistant>=1.0.0
EOF

# インストール
pip3 install -r requirements.txt
```

## 関連ドキュメント

- [MLZero_Complete_Setup_Guide.md](./MLZero_Complete_Setup_Guide.md) - 完全セットアップガイド
- [MLZero_Troubleshooting.md](./MLZero_Troubleshooting.md) - トラブルシューティングガイド
- [MLZero_Dependency_Installation_Fix.md](./MLZero_Dependency_Installation_Fix.md) - 依存関係インストール修正

## 更新履歴

- 2025-12-30: 初版作成
  - autogluon.timeseriesモジュールエラーの解決方法を追加



