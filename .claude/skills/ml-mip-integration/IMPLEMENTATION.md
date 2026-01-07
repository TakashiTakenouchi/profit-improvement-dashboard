# ML-MIP Integration 詳細実装ガイド

## アーキテクチャ詳細

### MLモデルがMIP制約に変換される仕組み

機械学習モデルの予測関数 `y = g(x)` を**混合整数計画法（MIP）**の制約として定式化します。

#### 線形回帰

最も単純。`y = β₀ + Σβᵢxᵢ` を線形制約として直接表現。近似誤差ゼロ。

```python
# 内部的な変換
# y = intercept + coef[0]*x[0] + coef[1]*x[1] + ...
linear_expr = model.intercept_
for i, c in enumerate(model.coef_):
    linear_expr += c * x[i]
problem += y == linear_expr
```

#### 決定木

各リーフノードへのパスをバイナリ変数で選択：

```
リーフ1: x[0] <= 5 AND x[1] > 3  → z[1] = 1
リーフ2: x[0] <= 5 AND x[1] <= 3 → z[2] = 1
...
y = Σ (予測値[i] * z[i])
Σ z[i] = 1  # 1つのリーフのみ選択
```

#### ニューラルネットワーク（ReLU）

Big-M法による定式化：

```
各ニューロン j について:
y[j] >= 0
y[j] >= w'x + b
y[j] <= M * z[j]           # z=0 なら y=0
y[j] <= w'x + b + M*(1-z[j]) # z=1 なら y=w'x+b
z[j] ∈ {0, 1}
```

## Option 1: PuLP + HiGHS（線形モデル専用）

### 完全な使用例

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from ml_mip_integration import PuLPLinearMLIntegrator, SolverType

# データ準備
np.random.seed(42)
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + 10 + np.random.randn(100) * 0.1

# Pipelineでスケーリング込みで訓練
# 注意: MinMaxScalerは非対応、StandardScalerを使用
pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipe.fit(X, y)

# 最適化モデル構築
integrator = PuLPLinearMLIntegrator(solver=SolverType.HIGHS)
integrator.create_model("OptimizationWithScaling")

# バウンド設定（重要！）
lb = X.min(axis=0) - 1
ub = X.max(axis=0) + 1

# 変数設定
x = integrator.add_input_vars(5, lb=lb.tolist(), ub=ub.tolist(),
                              names=['cost', 'labor', 'material', 'overhead', 'risk'])
y_var = integrator.add_output_var(name='profit')

# Pipelineの最終段階（Ridge）を取り出して制約追加
# 注意: Pipelineをそのまま渡すことはできない
ridge_model = pipe.named_steps['ridge']

# スケーリングを手動適用
scaler = pipe.named_steps['standardscaler']
x_scaled = [(x[i] - scaler.mean_[i]) / scaler.scale_[i] for i in range(5)]

# 線形制約を手動構築
linear_expr = ridge_model.intercept_
for i, c in enumerate(ridge_model.coef_):
    linear_expr += c * x_scaled[i]
integrator.model += y_var == linear_expr, "ml_prediction"

# 追加制約
integrator.add_constraint(x[0] + x[1] + x[2] <= 5, "budget")
integrator.add_constraint(x[3] >= 0.5, "min_overhead")

# 最適化
result = integrator.optimize(sense="maximize")

# 結果検証
if result.is_optimal():
    # Pipelineで予測して検証
    predicted = pipe.predict(result.input_values.reshape(1, -1))[0]
    print(f"最適化結果: {result.objective_value:.4f}")
    print(f"Pipeline予測: {predicted:.4f}")
    print(f"誤差: {abs(result.objective_value - predicted):.6f}")
```

### 利用可能なソルバー

```python
# HiGHS（推奨）
integrator = PuLPLinearMLIntegrator(solver=SolverType.HIGHS)

# CBC
integrator = PuLPLinearMLIntegrator(solver=SolverType.CBC)

# GLPK
integrator = PuLPLinearMLIntegrator(solver=SolverType.GLPK)
```

## Option 2: OMLT + Pyomo（ニューラルネットワーク対応）

### sklearn MLPRegressorの使用

```python
from sklearn.neural_network import MLPRegressor
import numpy as np
from ml_mip_integration import OMLTIntegrator, VariableBounds, SolverType

# MLPモデル訓練
X = np.random.randn(500, 5)
y = np.sin(X).sum(axis=1) + X[:, 0] ** 2

mlp = MLPRegressor(
    hidden_layer_sizes=(20, 20),
    activation='relu',  # 必須: ReLUのみサポート
    max_iter=1000,
    random_state=42
)
mlp.fit(X, y)

# OMLT最適化
integrator = OMLTIntegrator(solver=SolverType.HIGHS)
integrator.create_model()

# バウンド設定（NNでは特に重要）
input_bounds = VariableBounds(
    lower=X.min(axis=0) - 0.5,
    upper=X.max(axis=0) + 0.5
)

# MLPを制約として追加（内部でONNX変換）
x, y_var = integrator.add_sklearn_mlp(mlp, input_bounds, n_inputs=5)

# 追加制約（Pyomo式で記述）
pyo = integrator._pyo
integrator.add_constraint(x[0] + x[1] <= 2, "sum_constraint")

# 最適化
result = integrator.optimize(sense="minimize")

if result.is_optimal():
    print(f"最適値: {result.objective_value:.4f}")
    print(f"予測誤差: {result.prediction_error:.6f}")
```

### Kerasモデルの使用

```python
from ml_mip_integration import OMLTIntegrator, VariableBounds

# Kerasモデル（事前に訓練済み）
# model = keras.Sequential([
#     keras.layers.Dense(32, activation='relu', input_shape=(5,)),
#     keras.layers.Dense(16, activation='relu'),
#     keras.layers.Dense(1)
# ])

integrator = OMLTIntegrator()
integrator.create_model()

input_bounds = VariableBounds(-2, 2)
x, y = integrator.add_keras_model(keras_model, input_bounds, n_inputs=5)

result = integrator.optimize(sense="maximize")
```

## Option 3: PySCIPOpt-ML（全モデル対応）

### ランダムフォレストの使用

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from ml_mip_integration import SCIPMLIntegrator

# ランダムフォレスト訓練（サイズに注意）
X = np.random.randn(500, 5)
y = X[:, 0] * 2 + X[:, 1] ** 2 + np.random.randn(500) * 0.1

rf = RandomForestRegressor(
    n_estimators=10,    # 少なめに
    max_depth=5,        # 浅めに
    min_samples_leaf=20,
    random_state=42
)
rf.fit(X, y)

# SCIP最適化
integrator = SCIPMLIntegrator()
integrator.create_model()

lb = X.min(axis=0) - 1
ub = X.max(axis=0) + 1
x = integrator.add_input_vars(5, lb=lb.tolist(), ub=ub.tolist())
y_var = integrator.add_output_var()

# ランダムフォレストを制約として追加
integrator.add_predictor_constraint(rf, x, y_var, epsilon=0.001)

# 制約追加（SCIP形式）
integrator.add_constraint(y_var, ">=", 5, "min_profit")  # 利益下限

# 総入力最小化
total = sum(x[i] for i in range(5))
integrator.set_objective(total, "minimize")

result = integrator.optimize(time_limit=120)
```

### Gradient Boostingの使用

```python
from sklearn.ensemble import GradientBoostingRegressor
from ml_mip_integration import SCIPMLIntegrator

# GBT訓練
gbt = GradientBoostingRegressor(
    n_estimators=20,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
gbt.fit(X, y)

# SCIP最適化
integrator = SCIPMLIntegrator()
integrator.create_model()

x = integrator.add_input_vars(5, lb=-5, ub=5)
y_var = integrator.add_output_var()

integrator.add_predictor_constraint(gbt, x, y_var)
result = integrator.optimize(sense="maximize", time_limit=300)
```

## 損益計算書最適化の完全例

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
from ml_mip_integration import PuLPLinearMLIntegrator, SolverType

# 損益データ生成
np.random.seed(42)
n_samples = 500

# 費用項目（千円単位）
variable_cost = np.random.uniform(20000, 40000, n_samples)
fixed_cost = np.random.uniform(10000, 20000, n_samples)
depreciation = np.random.uniform(5000, 15000, n_samples)
selling_expense = np.random.uniform(5000, 15000, n_samples)
admin_expense = np.random.uniform(3000, 10000, n_samples)

X = np.column_stack([variable_cost, fixed_cost, depreciation,
                     selling_expense, admin_expense])

# 営業利益 = 売上(1億) - 総費用 + 非線形効果 + ノイズ
total_cost = X.sum(axis=1)
y = 100000 - total_cost - 0.0001 * variable_cost**2 + np.random.randn(n_samples) * 500

# モデル訓練
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 最適化設定
integrator = PuLPLinearMLIntegrator(solver=SolverType.HIGHS)
integrator.create_model("PLOptimization")

feature_names = ['variable_cost', 'fixed_cost', 'depreciation',
                 'selling_expense', 'admin_expense']

# バウンド設定
lb = X_train.min(axis=0) * 0.9
ub = X_train.max(axis=0) * 1.1

x = integrator.add_input_vars(5, lb=lb.tolist(), ub=ub.tolist(), names=feature_names)
profit = integrator.add_output_var(name='operating_income')

# ML予測を制約に
integrator.add_predictor_constraint(model, x, profit)

# 業務制約
integrator.add_constraint(x[0] + x[1] + x[2] + x[3] + x[4] <= 80000, "total_budget")
integrator.add_constraint(x[0] <= 2 * x[1], "var_fixed_ratio")
integrator.add_constraint(x[3] >= x[4], "selling_admin_ratio")

# 利益最大化
result = integrator.optimize(sense="maximize")

print(f"最適営業利益: {result.objective_value:,.0f} 千円")
for name, val in zip(feature_names, result.input_values):
    print(f"  {name}: {val:,.0f} 千円")
```

## LangGraph統合

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
import numpy as np

class OptState(TypedDict):
    model: any
    X_train: np.ndarray
    objective: str
    result: any

def optimize_node(state: OptState) -> OptState:
    from ml_mip_integration import MLMIPFactory, VariableBounds, SolverType

    model = state["model"]
    X = state["X_train"]

    bounds = VariableBounds(X.min(axis=0) * 0.9, X.max(axis=0) * 1.1)
    integrator = MLMIPFactory.create(model, SolverType.HIGHS)
    integrator.create_model()

    n = X.shape[1]
    lb = bounds.lower.tolist() if hasattr(bounds.lower, 'tolist') else [bounds.lower]*n
    ub = bounds.upper.tolist() if hasattr(bounds.upper, 'tolist') else [bounds.upper]*n

    x = integrator.add_input_vars(n, lb=lb, ub=ub)
    y = integrator.add_output_var()
    integrator.add_predictor_constraint(model, x, y)

    result = integrator.optimize(sense=state.get("objective", "maximize"))
    return {**state, "result": result}

# ワークフロー構築
workflow = StateGraph(OptState)
workflow.add_node("optimize", optimize_node)
workflow.set_entry_point("optimize")
workflow.add_edge("optimize", END)
app = workflow.compile()
```

## パフォーマンス最適化のヒント

### 1. バウンドは必ず指定

```python
# 悪い例（バウンドなし）
x = integrator.add_input_vars(5)  # Big-M制約が非常に弱くなる

# 良い例
lb = X_train.min(axis=0) * 0.9
ub = X_train.max(axis=0) * 1.1
x = integrator.add_input_vars(5, lb=lb, ub=ub)
```

### 2. モデルサイズを制限

```python
# 推奨設定
rf = RandomForestRegressor(
    n_estimators=10,      # 20以下
    max_depth=5,          # 5以下
    min_samples_leaf=20
)

mlp = MLPRegressor(
    hidden_layer_sizes=(50, 50),  # 2-3層、各50-200ニューロン
    activation='relu'
)
```

### 3. タイムリミットを設定

```python
# SCIP
result = integrator.optimize(time_limit=120)  # 秒

# 解の品質チェック
if result.gap and result.gap > 0.01:
    print(f"警告: MIPギャップ {result.gap*100:.1f}%")
```

### 4. 予測誤差を検証

```python
result = integrator.optimize(sense="maximize")

# 最適化結果とML予測を比較
ml_pred = model.predict(result.input_values.reshape(1, -1))[0]
print(f"最適化出力: {result.objective_value}")
print(f"ML直接予測: {ml_pred}")
print(f"誤差: {abs(result.objective_value - ml_pred)}")
```
