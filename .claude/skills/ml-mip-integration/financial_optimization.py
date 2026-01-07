"""
Q-Storm 損益計算書最適化サンプル
================================
損益勘定、経費、変動費、固定費、償却費の最適化例

このサンプルでは、機械学習モデルを使用して利益を予測し、
数理最適化で最適な経費配分を求めます。
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ML-MIP統合ライブラリ
from ml_mip_integration import (
    PuLPLinearMLIntegrator,
    SCIPMLIntegrator,
    OMLTIntegrator,
    MLMIPFactory,
    VariableBounds,
    OptimizationResult,
    SolverType,
    quick_optimize
)


# =============================================================================
# 損益計算書データモデル
# =============================================================================

@dataclass
class PLStatement:
    """損益計算書データ構造"""
    revenue: float                    # 売上高
    variable_cost: float              # 変動費
    fixed_cost: float                 # 固定費
    depreciation: float               # 償却費
    selling_expense: float            # 販売費
    admin_expense: float              # 管理費
    
    @property
    def gross_profit(self) -> float:
        """粗利益"""
        return self.revenue - self.variable_cost
    
    @property
    def operating_income(self) -> float:
        """営業利益"""
        return (self.gross_profit - self.fixed_cost - self.depreciation 
                - self.selling_expense - self.admin_expense)
    
    def to_feature_vector(self) -> np.ndarray:
        """特徴量ベクトルに変換"""
        return np.array([
            self.variable_cost,
            self.fixed_cost,
            self.depreciation,
            self.selling_expense,
            self.admin_expense
        ])


def generate_pl_dataset(n_samples: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    損益計算書の合成データセットを生成
    
    特徴量:
    - variable_cost: 変動費
    - fixed_cost: 固定費
    - depreciation: 償却費
    - selling_expense: 販売費
    - admin_expense: 管理費
    
    目的変数:
    - operating_income: 営業利益（売上から費用を引いた値に非線形効果を加えたもの）
    """
    np.random.seed(seed)
    
    # 基本的な売上高（固定）
    base_revenue = 100_000_000  # 1億円
    
    # 費用の範囲設定（千円単位）
    variable_cost = np.random.uniform(20_000, 40_000, n_samples)      # 2000万〜4000万
    fixed_cost = np.random.uniform(10_000, 20_000, n_samples)         # 1000万〜2000万
    depreciation = np.random.uniform(5_000, 15_000, n_samples)        # 500万〜1500万
    selling_expense = np.random.uniform(5_000, 15_000, n_samples)     # 500万〜1500万
    admin_expense = np.random.uniform(3_000, 10_000, n_samples)       # 300万〜1000万
    
    # 特徴量行列
    X = np.column_stack([
        variable_cost,
        fixed_cost,
        depreciation,
        selling_expense,
        admin_expense
    ])
    
    # 営業利益の計算（基本式 + 非線形効果 + ノイズ）
    # 基本: 売上 - (変動費 + 固定費 + 償却費 + 販売費 + 管理費)
    total_cost = variable_cost + fixed_cost + depreciation + selling_expense + admin_expense
    basic_profit = (base_revenue / 1000) - total_cost  # 千円単位
    
    # 非線形効果: 変動費の二次効果、固定費と販売費の交互作用
    nonlinear_effect = (
        - 0.0001 * variable_cost ** 2  # 変動費が高すぎると効率低下
        + 0.001 * fixed_cost * selling_expense / 1000  # 固定費と販売費のシナジー
        - 0.0002 * admin_expense ** 2  # 管理費の過剰は非効率
    )
    
    # ノイズ
    noise = np.random.normal(0, 500, n_samples)
    
    # 最終的な営業利益
    y = basic_profit + nonlinear_effect + noise
    
    # DataFrameも作成
    df = pd.DataFrame({
        'variable_cost': variable_cost,
        'fixed_cost': fixed_cost,
        'depreciation': depreciation,
        'selling_expense': selling_expense,
        'admin_expense': admin_expense,
        'operating_income': y
    })
    
    return X, y, df


# =============================================================================
# 例1: 線形モデルによる利益最大化（PuLP + HiGHS）
# =============================================================================

def example_linear_profit_maximization():
    """
    線形回帰モデルを使用した利益最大化
    
    最も軽量なアプローチ。
    線形モデルは制約として直接表現可能。
    """
    print("=" * 60)
    print("例1: 線形モデルによる利益最大化（PuLP + HiGHS）")
    print("=" * 60)
    
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    # データ生成
    X, y, df = generate_pl_dataset(500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # モデル訓練
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    print(f"\nモデル精度 (R²): {r2_score(y_test, model.predict(X_test)):.4f}")
    print(f"係数: {model.coef_}")
    print(f"切片: {model.intercept_:.2f}")
    
    # 最適化設定
    integrator = PuLPLinearMLIntegrator(solver=SolverType.HIGHS)
    integrator.create_model("ProfitMaximization")
    
    # 決定変数（費用項目）
    feature_names = ['variable_cost', 'fixed_cost', 'depreciation', 'selling_expense', 'admin_expense']
    
    # バウンド設定（訓練データの範囲）
    lb = X_train.min(axis=0) * 0.9
    ub = X_train.max(axis=0) * 1.1
    
    x = integrator.add_input_vars(5, lb=lb.tolist(), ub=ub.tolist(), names=feature_names)
    y_var = integrator.add_output_var(name='operating_income')
    
    # ML予測を制約として追加
    integrator.add_predictor_constraint(model, x, y_var)
    
    # 追加の業務制約
    pulp = integrator._pulp
    
    # 制約1: 総費用は8000万円以下
    integrator.add_constraint(
        x[0] + x[1] + x[2] + x[3] + x[4] <= 80_000,
        "total_cost_limit"
    )
    
    # 制約2: 変動費は固定費の2倍以下
    integrator.add_constraint(
        x[0] <= 2 * x[1],
        "variable_fixed_ratio"
    )
    
    # 制約3: 販売費は管理費より大きい
    integrator.add_constraint(
        x[3] >= x[4],
        "selling_admin_ratio"
    )
    
    # 最適化実行（利益最大化）
    result = integrator.optimize(sense="maximize")
    
    print(f"\n最適化結果:")
    print(f"  ステータス: {result.status}")
    print(f"  最適営業利益: {result.objective_value:,.0f} 千円")
    print(f"  解決時間: {result.solve_time:.4f} 秒")
    print(f"  予測誤差: {result.prediction_error:.4f}")
    
    print(f"\n最適費用配分:")
    for name, value in zip(feature_names, result.input_values):
        print(f"  {name}: {value:,.0f} 千円")
    
    # 検証: モデルの予測値と比較
    predicted = model.predict(result.input_values.reshape(1, -1))[0]
    print(f"\n検証:")
    print(f"  最適化結果の予測値: {result.objective_value:,.2f}")
    print(f"  モデル直接予測値: {predicted:,.2f}")
    print(f"  差分: {abs(result.objective_value - predicted):.4f}")
    
    return result


# =============================================================================
# 例2: 決定木モデルによるコスト最適化（SCIP）
# =============================================================================

def example_decision_tree_cost_optimization():
    """
    決定木回帰モデルを使用したコスト最適化
    
    PySCIPOpt-MLを使用。
    非線形関係をキャプチャ可能。
    """
    print("\n" + "=" * 60)
    print("例2: 決定木モデルによるコスト最適化（PySCIPOpt-ML + SCIP）")
    print("=" * 60)
    
    try:
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
    except ImportError as e:
        print(f"必要なライブラリがありません: {e}")
        return None
    
    # データ生成
    X, y, df = generate_pl_dataset(500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 決定木モデル訓練
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\nモデル精度 (R²): {r2_score(y_test, model.predict(X_test)):.4f}")
    print(f"木の深さ: {model.get_depth()}")
    print(f"リーフ数: {model.get_n_leaves()}")
    
    try:
        # SCIP最適化
        integrator = SCIPMLIntegrator()
        integrator.create_model("CostOptimization")
        
        # バウンド設定
        lb = X_train.min(axis=0) * 0.9
        ub = X_train.max(axis=0) * 1.1
        
        x = integrator.add_input_vars(5, lb=lb.tolist(), ub=ub.tolist())
        y_var = integrator.add_output_var()
        
        # 決定木を制約として追加
        integrator.add_predictor_constraint(model, x, y_var, epsilon=0.001)
        
        # 目標利益制約: 営業利益は25000千円以上
        integrator.add_constraint(y_var, ">=", 25000, "profit_target")
        
        # 最適化実行（総費用最小化 = 全費用変数の合計最小化）
        total_cost = sum(x[i] for i in range(5))
        integrator.set_objective(total_cost, "minimize")
        
        result = integrator.optimize(time_limit=60)
        
        print(f"\n最適化結果:")
        print(f"  ステータス: {result.status}")
        if result.is_optimal():
            print(f"  最小総費用: {sum(result.input_values):,.0f} 千円")
            print(f"  達成営業利益: {result.output_values[0]:,.0f} 千円")
            print(f"  解決時間: {result.solve_time:.4f} 秒")
            
            feature_names = ['variable_cost', 'fixed_cost', 'depreciation', 'selling_expense', 'admin_expense']
            print(f"\n最適費用配分:")
            for name, value in zip(feature_names, result.input_values):
                print(f"  {name}: {value:,.0f} 千円")
        
        return result
        
    except ImportError as e:
        print(f"\nSCIPが利用できません: {e}")
        print("pip install pyscipopt pyscipopt-ml でインストールしてください。")
        return None


# =============================================================================
# 例3: ランダムフォレストによるロバスト最適化
# =============================================================================

def example_random_forest_robust_optimization():
    """
    ランダムフォレストモデルを使用したロバスト最適化
    
    アンサンブルモデルにより予測の安定性を向上。
    """
    print("\n" + "=" * 60)
    print("例3: ランダムフォレストによるロバスト最適化")
    print("=" * 60)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
    except ImportError as e:
        print(f"必要なライブラリがありません: {e}")
        return None
    
    # データ生成
    X, y, df = generate_pl_dataset(500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ランダムフォレストモデル（小規模に抑える）
    model = RandomForestRegressor(
        n_estimators=5,      # 木の数を少なく
        max_depth=4,         # 深さを制限
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print(f"\nモデル精度 (R²): {r2_score(y_test, model.predict(X_test)):.4f}")
    print(f"木の数: {model.n_estimators}")
    
    try:
        integrator = SCIPMLIntegrator()
        integrator.create_model("RobustOptimization")
        
        lb = X_train.min(axis=0) * 0.9
        ub = X_train.max(axis=0) * 1.1
        
        x = integrator.add_input_vars(5, lb=lb.tolist(), ub=ub.tolist())
        y_var = integrator.add_output_var()
        
        # ランダムフォレストを制約として追加
        print("\nランダムフォレストをMIP制約に変換中...")
        integrator.add_predictor_constraint(model, x, y_var)
        
        # 最適化（利益最大化）
        result = integrator.optimize(sense="maximize", time_limit=120)
        
        print(f"\n最適化結果:")
        print(f"  ステータス: {result.status}")
        if result.is_optimal():
            print(f"  最適営業利益: {result.output_values[0]:,.0f} 千円")
            print(f"  解決時間: {result.solve_time:.4f} 秒")
        
        return result
        
    except Exception as e:
        print(f"\n最適化エラー: {e}")
        return None


# =============================================================================
# 例4: クイック最適化API
# =============================================================================

def example_quick_optimize():
    """
    quick_optimize関数を使用した簡易最適化
    
    最小限のコードで最適化を実行。
    """
    print("\n" + "=" * 60)
    print("例4: クイック最適化API")
    print("=" * 60)
    
    from sklearn.linear_model import LinearRegression
    
    # データ生成と訓練
    X, y, df = generate_pl_dataset(500)
    model = LinearRegression().fit(X, y)
    
    print(f"\nモデル精度 (R²): {model.score(X, y):.4f}")
    
    # クイック最適化（1行で実行）
    result = quick_optimize(
        model=model,
        X_train=X,
        objective="maximize",
        solver=SolverType.HIGHS
    )
    
    print(f"\n最適化結果:")
    print(f"  ステータス: {result.status}")
    print(f"  最適営業利益: {result.objective_value:,.0f} 千円")
    print(f"  解決時間: {result.solve_time:.4f} 秒")
    
    feature_names = ['variable_cost', 'fixed_cost', 'depreciation', 'selling_expense', 'admin_expense']
    print(f"\n最適費用配分:")
    for name, value in zip(feature_names, result.input_values):
        print(f"  {name}: {value:,.0f} 千円")
    
    return result


# =============================================================================
# 例5: LangGraph統合用ノード
# =============================================================================

def create_langgraph_optimization_node():
    """
    LangGraph統合用の最適化ノードを作成
    
    Q-Storm-ML-WorkFlow-LangGraphへの統合例
    """
    print("\n" + "=" * 60)
    print("例5: LangGraph統合用ノードテンプレート")
    print("=" * 60)
    
    # LangGraph用のノード関数テンプレート
    node_code = '''
# LangGraph統合用ノード関数
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import numpy as np

class OptimizationState(TypedDict):
    """最適化ワークフローの状態"""
    model: any                          # 訓練済みMLモデル
    X_train: np.ndarray                 # 訓練データ
    constraints: dict                    # 追加制約
    objective: str                       # "maximize" or "minimize"
    result: OptimizationResult | None   # 最適化結果

def ml_optimization_node(state: OptimizationState) -> OptimizationState:
    """
    ML-MIP最適化を実行するLangGraphノード
    
    - 入力: 訓練済みモデル、訓練データ、制約条件
    - 出力: 最適化結果を含む更新された状態
    """
    from ml_mip_integration import (
        MLMIPFactory, 
        VariableBounds, 
        quick_optimize,
        SolverType
    )
    
    model = state["model"]
    X_train = state["X_train"]
    objective = state.get("objective", "maximize")
    
    # バウンドを訓練データから自動推定
    margin = 0.1
    input_bounds = VariableBounds(
        lower=X_train.min(axis=0) * (1 - margin),
        upper=X_train.max(axis=0) * (1 + margin)
    )
    
    # ファクトリーで最適なインテグレーターを選択
    integrator = MLMIPFactory.create(model, solver=SolverType.HIGHS)
    integrator.create_model()
    
    n_features = X_train.shape[1]
    x = integrator.add_input_vars(
        n_features,
        lb=input_bounds.lower.tolist() if hasattr(input_bounds.lower, 'tolist') else input_bounds.lower,
        ub=input_bounds.upper.tolist() if hasattr(input_bounds.upper, 'tolist') else input_bounds.upper
    )
    y = integrator.add_output_var()
    
    integrator.add_predictor_constraint(model, x, y)
    
    # 追加制約を適用
    constraints = state.get("constraints", {})
    for name, constraint_def in constraints.items():
        # constraint_def: {"type": "<=", "coeffs": [...], "rhs": value}
        pass  # カスタム制約ロジック
    
    result = integrator.optimize(sense=objective)
    
    return {**state, "result": result}

def result_formatting_node(state: OptimizationState) -> OptimizationState:
    """
    最適化結果をフォーマットするノード
    """
    result = state["result"]
    
    if result and result.is_optimal():
        formatted = {
            "status": "成功",
            "objective_value": f"{result.objective_value:,.2f}",
            "optimal_inputs": result.input_values.tolist(),
            "solve_time": f"{result.solve_time:.4f}秒"
        }
    else:
        formatted = {
            "status": "失敗",
            "error": result.status if result else "結果なし"
        }
    
    return {**state, "formatted_result": formatted}

# LangGraphワークフロー定義
def create_optimization_workflow():
    """最適化ワークフローを作成"""
    workflow = StateGraph(OptimizationState)
    
    workflow.add_node("optimize", ml_optimization_node)
    workflow.add_node("format", result_formatting_node)
    
    workflow.set_entry_point("optimize")
    workflow.add_edge("optimize", "format")
    workflow.add_edge("format", END)
    
    return workflow.compile()
'''
    
    print(node_code)
    
    return node_code


# =============================================================================
# メイン実行
# =============================================================================

def main():
    """すべての例を実行"""
    print("\n" + "=" * 70)
    print("Q-Storm 損益計算書最適化サンプル")
    print("=" * 70)
    
    # 例1: 線形モデル（必ず動作）
    result1 = example_linear_profit_maximization()
    
    # 例4: クイック最適化（線形モデルで動作）
    result4 = example_quick_optimize()
    
    # 例5: LangGraph統合テンプレート
    create_langgraph_optimization_node()
    
    # SCIP依存の例（オプション）
    print("\n" + "-" * 60)
    print("以下の例はPySCIPOpt-MLが必要です")
    print("-" * 60)
    
    try:
        result2 = example_decision_tree_cost_optimization()
        result3 = example_random_forest_robust_optimization()
    except Exception as e:
        print(f"\nSCIP関連の例はスキップされました: {e}")
        print("pip install pyscipopt pyscipopt-ml でインストールしてください。")
    
    print("\n" + "=" * 70)
    print("すべての例が完了しました")
    print("=" * 70)


if __name__ == "__main__":
    main()
