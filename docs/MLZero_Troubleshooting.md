# MLZero Web UI トラブルシューティング

## 現在の状況

- ✅ AutoGluon Assistantはインストール済み
- ❌ `mlzero-backend`と`mlzero-frontend`コマンドが見つからない

## 確認事項

### 1. インストールの確認

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
pip3 list | grep autogluon
python3 -c "import autogluon.assistant; print('OK')"
```

### 2. コマンドの場所を確認

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
which mlzero-backend
which mlzero-frontend
ls -la ~/.local/bin/ | grep mlzero
```

### 3. Pythonモジュールから直接起動

AutoGluon AssistantがCLIコマンドを提供していない場合、Python APIから直接起動する必要があるかもしれません。

```python
from autogluon.assistant import run_agent

# Web UIモードで起動する方法を確認
```

## 代替案: Python APIの使用

Web UIが利用できない場合、Python APIを直接使用できます：

```python
from autogluon.assistant import run_agent

run_agent(
    input_data_folder="./input",
    output_folder="./output",
    initial_user_input="Create a classification model"
)
```

## よくあるエラーと解決方法

### エラー: ModuleNotFoundError: No module named 'autogluon.timeseries'

**症状**: MLZeroが時系列予測タスクを実行しようとした際に発生

**解決方法**:

```bash
wsl
export PATH="$HOME/.local/bin:$PATH"
pip3 install autogluon.timeseries
```

**確認**:

```bash
wsl
python3 -c "import autogluon.timeseries; print('OK')"
```

**詳細**: [MLZero_Timeseries_Module_Error.md](./MLZero_Timeseries_Module_Error.md) を参照

### エラー: Task failed after maximum iterations

**症状**: MLZeroが最大イテレーション数に達してもタスクを完了できない

**原因**: 
- 必要なパッケージがインストールされていない
- データ形式が正しくない
- コード生成が失敗している

**解決方法**:
1. ログファイルを確認（`~/.local/lib/python3.10/runs/mlzero-*/logs.txt`）
2. エラーメッセージを確認
3. 必要なパッケージをインストール
4. タスクを再実行

## 次のステップ

1. GitHubリポジトリの最新ドキュメントを確認
2. インストール方法を再確認
3. Python APIから直接使用する方法を検討
4. [MLZero_Complete_Setup_Guide.md](./MLZero_Complete_Setup_Guide.md) を参照

