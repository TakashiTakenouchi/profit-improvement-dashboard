---
name: ml-mip-integration
description: Machine Learning models as MIP constraints using open-source solvers (HiGHS, SCIP, CBC). Use when optimizing with ML predictions, embedding sklearn models in optimization, profit maximization, cost minimization, or prescriptive analytics. Japanese/English supported.
---

# ML-MIP Integration Library

Gurobi Machine Learningの機能をオープンソースソルバー（HiGHS、SCIP、CBC）で実現する統合ライブラリ

## Overview

このSkillは、訓練済み機械学習モデルを数理最適化の制約として埋め込むことを可能にします。

### 主な用途

- **利益最大化**: MLモデルで予測した利益を最大化する入力を求める
- **コスト最小化**: 目標性能を達成しつつ総コストを最小化
- **処方的分析（Prescriptive Analytics）**: 「何が起こるか」の予測と「何をすべきか」の最適化を統合
- **損益計算書最適化**: 費用配分の最適化

### 3つの実装オプション

| Option | ソルバー | サポートモデル | 特徴 |
|--------|---------|--------------|------|
| **PuLP+HiGHS** | HiGHS, CBC, GLPK | 線形モデルのみ | 最軽量・高速 |
| **OMLT+Pyomo** | HiGHS, CBC, SCIP | NN, GBT | ONNX変換必要 |
| **PySCIPOpt-ML** | SCIP | 全モデル対応 | 最も広範 |

## Quick Start

### 最小構成（線形モデル）

```python
from sklearn.linear_model import Ridge
from ml_mip_integration import PuLPLinearMLIntegrator, SolverType

# 1. モデル訓練
model = Ridge().fit(X_train, y_train)

# 2. 最適化設定
integrator = PuLPLinearMLIntegrator(solver=SolverType.HIGHS)
integrator.create_model("ProfitMax")

# 3. 変数設定（バウンド必須）
x = integrator.add_input_vars(5, lb=X_train.min(axis=0), ub=X_train.max(axis=0))
y = integrator.add_output_var()

# 4. ML制約追加
integrator.add_predictor_constraint(model, x, y)

# 5. 最適化実行
result = integrator.optimize(sense="maximize")
print(f"最適値: {result.objective_value}")
```

### ファクトリーによる自動選択

```python
from ml_mip_integration import MLMIPFactory, SolverType

# モデルタイプに応じて最適なインテグレーターを自動選択
integrator = MLMIPFactory.create(model, solver=SolverType.HIGHS)
```

### クイック最適化（1行）

```python
from ml_mip_integration import quick_optimize

result = quick_optimize(model, X_train, objective="maximize")
```

## Supported Models

### 線形モデル（PuLP対応）
- LinearRegression, Ridge, Lasso, ElasticNet, PLSRegression

### ツリーモデル（SCIP対応）
- DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor
- XGBRegressor, LGBMRegressor

### ニューラルネットワーク（OMLT/SCIP対応）
- MLPRegressor（ReLU活性化のみ）
- Keras Sequential

## Files in This Skill

- [IMPLEMENTATION.md](IMPLEMENTATION.md) - 詳細実装ガイド
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - トラブルシューティング
- [ml_mip_integration.py](ml_mip_integration.py) - コアライブラリ
- [financial_optimization.py](financial_optimization.py) - 損益計算書最適化例
- [requirements.txt](requirements.txt) - 依存関係

## Installation

```bash
# 最小構成（線形モデルのみ）
pip install pulp[highs] numpy scikit-learn

# SCIP構成（全モデル対応）
pip install pyscipopt pyscipopt-ml

# OMLT構成（ニューラルネットワーク）
pip install pyomo omlt skl2onnx onnx
```

## Key Concepts

### 変数バウンドの重要性

Big-M定式化の効率はバウンドの質に強く依存します。必ず訓練データの範囲に基づいてバウンドを設定してください：

```python
lb = X_train.min(axis=0) * 0.9  # 10%マージン
ub = X_train.max(axis=0) * 1.1
```

### モデルサイズの目安

| モデルタイプ | 推奨サイズ | 計算時間目安 |
|------------|----------|-------------|
| 線形モデル | 制限なし | ミリ秒 |
| 決定木 | max_depth ≤ 10 | 数秒 |
| ランダムフォレスト | n_estimators ≤ 20, max_depth ≤ 5 | 数十秒〜数分 |
| MLP | 2-3層, 各層50-200ニューロン | 数分 |

## References

- [Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/)
- [OMLT](https://github.com/cog-imperial/OMLT)
- [PySCIPOpt-ML](https://github.com/Opt-Mucca/PySCIPOpt-ML)
- [HiGHS](https://highs.dev/)
