# ML-MIP Integration トラブルシューティング

## インストール関連

### HiGHSが見つからない

```
Error: HiGHS solver not found
```

**解決策:**
```bash
pip install highspy
# または
pip install pulp[highs]
```

### SCIPが見つからない

```
ImportError: cannot import name 'Model' from 'pyscipopt'
```

**解決策:**
```bash
# conda推奨（バイナリが含まれる）
conda install -c conda-forge pyscipopt
pip install pyscipopt-ml

# pipの場合（ビルド環境が必要）
pip install pyscipopt pyscipopt-ml
```

### OMLT変換エラー

```
ImportError: No module named 'skl2onnx'
```

**解決策:**
```bash
pip install skl2onnx onnx onnxruntime
```

### Pyomoソルバーが見つからない

```
ApplicationError: No executable found for solver 'highs'
```

**解決策:**
```bash
# HiGHSをシステムにインストール
pip install highspy

# 確認
python -c "import highspy; print(highspy.__version__)"
```

## 最適化実行時エラー

### "Model is infeasible"

**原因:** 制約が矛盾している、またはバウンドが厳しすぎる

**解決策:**
```python
# 1. バウンドを緩める
lb = X_train.min(axis=0) * 0.8  # 20%マージン
ub = X_train.max(axis=0) * 1.2

# 2. 制約を確認
# 例: x[0] <= 10 と x[0] >= 20 が同時にあると不可能

# 3. 個別に制約を追加して原因特定
integrator.add_constraint(...)
result = integrator.optimize()  # ここで失敗するか確認
```

### "Time limit reached" / 最適解が見つからない

**原因:** モデルが大きすぎる、またはバウンドが緩すぎる

**解決策:**
```python
# 1. モデルサイズを縮小
rf = RandomForestRegressor(
    n_estimators=5,    # 減らす
    max_depth=3,       # 減らす
)

# 2. バウンドを厳しく
lb = X_train.min(axis=0) * 0.95
ub = X_train.max(axis=0) * 1.05

# 3. タイムリミットを延長
result = integrator.optimize(time_limit=600)  # 10分

# 4. MIPギャップを許容
# SCIPの場合
integrator.model.setParam("limits/gap", 0.05)  # 5%ギャップ許容
```

### 予測誤差が大きい

**原因:** 最適化の数値精度、またはバウンド外の外挿

**確認方法:**
```python
result = integrator.optimize(sense="maximize")

# 予測誤差を確認
print(f"予測誤差: {result.prediction_error}")

# ML直接予測と比較
ml_pred = model.predict(result.input_values.reshape(1, -1))[0]
print(f"最適化出力: {result.objective_value}")
print(f"ML予測: {ml_pred}")
```

**解決策:**
```python
# バウンドを訓練データ範囲内に制限
lb = X_train.min(axis=0)  # マージンなし
ub = X_train.max(axis=0)
```

## モデル固有の問題

### 線形モデル以外でPuLPを使用しようとした

```
ValueError: PuLPLinearMLIntegratorは線形モデルのみサポートします
```

**解決策:**
```python
# ファクトリーを使用して自動選択
from ml_mip_integration import MLMIPFactory

integrator = MLMIPFactory.create(model)  # モデルに応じて自動選択
```

### ReLU以外の活性化関数

```
Only ReLU activation is supported
```

**解決策:**
```python
# MLPRegressorでReLUを指定
mlp = MLPRegressor(
    activation='relu',  # 'tanh', 'logistic'は非対応
    ...
)
```

### MinMaxScalerの使用

MinMaxScalerはサポートされていません。

**解決策:**
```python
from sklearn.preprocessing import StandardScaler

# MinMaxScalerの代わりにStandardScalerを使用
pipe = make_pipeline(StandardScaler(), Ridge())
```

## パフォーマンス問題

### 最適化が遅い

**診断:**
```python
import time

start = time.time()
result = integrator.optimize()
print(f"解決時間: {time.time() - start:.2f}秒")

# モデル統計
print(f"変数数: {len(integrator.input_vars)}")
```

**解決策:**

1. **バウンドを厳しくする**
   ```python
   lb = X_train.min(axis=0) * 0.95
   ub = X_train.max(axis=0) * 1.05
   ```

2. **モデルを簡素化**
   ```python
   # NNの場合
   mlp = MLPRegressor(hidden_layer_sizes=(20, 20))  # 小さく

   # RFの場合
   rf = RandomForestRegressor(n_estimators=5, max_depth=3)
   ```

3. **ソルバーパラメータ調整（SCIP）**
   ```python
   integrator.model.setParam("limits/time", 60)
   integrator.model.setParam("limits/gap", 0.01)
   integrator.model.setParam("presolving/maxrounds", 0)  # 前処理スキップ
   ```

### メモリ不足

**原因:** 大きなランダムフォレストやディープNNのMIP変換

**解決策:**
```python
# モデルサイズを制限
rf = RandomForestRegressor(
    n_estimators=10,
    max_depth=4,
    max_leaf_nodes=50  # リーフ数を制限
)
```

## デバッグ方法

### 最適化モデルの確認（PuLP）

```python
# LP/MIPファイルとして出力
integrator.model.writeLP("debug.lp")

# 変数と制約の確認
for v in integrator.model.variables():
    print(f"{v.name}: lb={v.lowBound}, ub={v.upBound}, val={v.varValue}")
```

### 最適化モデルの確認（SCIP）

```python
# 問題をファイルに出力
integrator.model.writeProblem("debug.cip")

# ログを有効化
integrator.model.hideOutput(False)
```

### 最適化モデルの確認（Pyomo/OMLT）

```python
# モデル構造を表示
integrator.model.pprint()

# LP形式で出力
integrator.model.write("debug.lp")
```

## よくある質問

### Q: Gurobiと比較してどの程度遅い？

A: 問題サイズによりますが、一般的に：
- 線形モデル: ほぼ同等（HiGHSは非常に高速）
- 決定木/RF: 2-5倍遅い（SCIPは汎用ソルバー）
- NN: 5-10倍遅い可能性（バウンド推定がGurobiより弱い）

### Q: Pipeline全体を直接渡せる？

A: 現在は非対応。最終段階のモデルを取り出して使用：
```python
final_model = pipe.steps[-1][1]
integrator.add_predictor_constraint(final_model, x, y)
```

スケーリングは手動で適用が必要です。

### Q: 複数の出力変数は対応？

A: 基本的に単一出力を想定。複数出力の場合：
```python
# 各出力に対して個別にモデルを作成
for i, model in enumerate(models):
    y = integrator.add_output_var(name=f"y_{i}")
    integrator.add_predictor_constraint(model, x, y)
```

### Q: 整数変数との組み合わせは？

A: PuLP/SCIP/Pyomoの機能を直接使用：
```python
# PuLPの場合
z = integrator._pulp.LpVariable("z", cat="Integer", lowBound=0, upBound=10)
integrator.model += x[0] <= z * 100, "integer_constraint"
```
