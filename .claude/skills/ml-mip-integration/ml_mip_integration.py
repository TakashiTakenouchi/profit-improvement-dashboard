"""
Q-Storm ML-MIP Integration Library
==================================
Gurobi MLの機能をオープンソースソルバー（HiGHS、SCIP、CBC）で実現する統合ライブラリ

サポートするアプローチ:
1. OMLT + Pyomo + HiGHS: ニューラルネットワーク・GBT対応
2. PySCIPOpt-ML + SCIP: scikit-learn直接サポート
3. Manual PuLP/HiGHS: 線形モデル専用（最軽量）

Author: Q-Storm Platform
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
import warnings

# =============================================================================
# 基本型定義
# =============================================================================

class SolverType(Enum):
    """利用可能なソルバータイプ"""
    HIGHS = "highs"
    SCIP = "scip"
    CBC = "cbc"
    GLPK = "glpk"


class MLModelType(Enum):
    """サポートするMLモデルタイプ"""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    MLP_REGRESSOR = "mlp_regressor"


@dataclass
class OptimizationResult:
    """最適化結果のデータクラス"""
    status: str
    objective_value: Optional[float] = None
    input_values: Optional[np.ndarray] = None
    output_values: Optional[np.ndarray] = None
    solve_time: Optional[float] = None
    gap: Optional[float] = None
    prediction_error: Optional[float] = None
    
    def is_optimal(self) -> bool:
        return self.status.lower() in ["optimal", "ok"]


@dataclass
class VariableBounds:
    """変数の上下限を定義"""
    lower: Union[float, np.ndarray]
    upper: Union[float, np.ndarray]
    
    def to_dict(self, n_vars: int) -> Dict[int, Tuple[float, float]]:
        """OMLTフォーマットに変換"""
        lb = np.full(n_vars, self.lower) if np.isscalar(self.lower) else self.lower
        ub = np.full(n_vars, self.upper) if np.isscalar(self.upper) else self.upper
        return {i: (lb[i], ub[i]) for i in range(n_vars)}


# =============================================================================
# 抽象基底クラス
# =============================================================================

class MLMIPIntegrator(ABC):
    """ML-MIP統合の抽象基底クラス"""
    
    @abstractmethod
    def add_predictor_constraint(
        self,
        predictor: Any,
        input_vars: Any,
        output_vars: Any,
        **kwargs
    ) -> Any:
        """MLモデルを制約として追加"""
        pass
    
    @abstractmethod
    def optimize(self, sense: str = "minimize") -> OptimizationResult:
        """最適化を実行"""
        pass
    
    @abstractmethod
    def get_prediction_error(self) -> float:
        """予測誤差を取得"""
        pass


# =============================================================================
# Option 1: PuLP + HiGHS による線形モデル実装（最軽量）
# =============================================================================

class PuLPLinearMLIntegrator(MLMIPIntegrator):
    """
    PuLP + HiGHSによる線形MLモデルの最適化
    
    サポートモデル:
    - LinearRegression
    - Ridge
    - Lasso
    - PLSRegression
    
    使用例:
    ```python
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # モデル訓練
    X = np.random.randn(100, 5)
    y = X @ np.array([1, 2, 3, 4, 5]) + 10
    model = LinearRegression().fit(X, y)
    
    # 最適化
    integrator = PuLPLinearMLIntegrator(solver=SolverType.HIGHS)
    integrator.create_model("profit_optimization")
    
    # 入出力変数
    x = integrator.add_input_vars(5, lb=-10, ub=10, names=["cost", "labor", "material", "overhead", "risk"])
    y = integrator.add_output_var(name="profit")
    
    # ML制約追加
    integrator.add_predictor_constraint(model, x, y)
    
    # 追加制約（予算制約など）
    integrator.add_constraint(x[0] + x[1] + x[2] <= 100, "budget")
    
    # 最適化
    result = integrator.optimize(sense="maximize")
    ```
    """
    
    def __init__(self, solver: SolverType = SolverType.HIGHS):
        self.solver = solver
        self.model = None
        self.input_vars = []
        self.output_vars = []
        self.predictor = None
        self._pulp = None
        self._import_pulp()
    
    def _import_pulp(self):
        """PuLPをインポート"""
        try:
            import pulp
            self._pulp = pulp
        except ImportError:
            raise ImportError(
                "PuLPがインストールされていません。\n"
                "pip install pulp[highs] でインストールしてください。"
            )
    
    def _get_solver(self):
        """ソルバーインスタンスを取得"""
        pulp = self._pulp
        
        solver_map = {
            SolverType.HIGHS: pulp.HiGHS_CMD(msg=0),
            SolverType.CBC: pulp.PULP_CBC_CMD(msg=0),
            SolverType.GLPK: pulp.GLPK_CMD(msg=0),
        }
        
        if self.solver not in solver_map:
            raise ValueError(f"PuLPでサポートされないソルバー: {self.solver}")
        
        return solver_map[self.solver]
    
    def create_model(self, name: str = "MLOptimization"):
        """最適化モデルを作成"""
        self.model = self._pulp.LpProblem(name, self._pulp.LpMinimize)
        self.input_vars = []
        self.output_vars = []
        return self
    
    def add_input_vars(
        self,
        n_vars: int,
        lb: Union[float, List[float]] = None,
        ub: Union[float, List[float]] = None,
        names: Optional[List[str]] = None
    ) -> List:
        """入力変数を追加"""
        pulp = self._pulp
        
        if names is None:
            names = [f"x_{i}" for i in range(n_vars)]
        
        lb_list = [lb] * n_vars if np.isscalar(lb) or lb is None else lb
        ub_list = [ub] * n_vars if np.isscalar(ub) or ub is None else ub
        
        vars_list = []
        for i, name in enumerate(names):
            var = pulp.LpVariable(
                name,
                lowBound=lb_list[i],
                upBound=ub_list[i],
                cat=pulp.LpContinuous
            )
            vars_list.append(var)
        
        self.input_vars.extend(vars_list)
        return vars_list
    
    def add_output_var(
        self,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
        name: str = "y"
    ):
        """出力変数を追加"""
        pulp = self._pulp
        var = pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=pulp.LpContinuous)
        self.output_vars.append(var)
        return var
    
    def add_predictor_constraint(
        self,
        predictor: Any,
        input_vars: List,
        output_vars: Any,
        **kwargs
    ):
        """
        線形回帰モデルを制約として追加
        
        y = β₀ + Σβᵢxᵢ を線形制約として直接表現
        """
        # モデルタイプ検証
        model_type = type(predictor).__name__
        supported_types = ['LinearRegression', 'Ridge', 'Lasso', 'PLSRegression', 'ElasticNet']
        
        if model_type not in supported_types:
            raise ValueError(
                f"PuLPLinearMLIntegratorは線形モデルのみサポートします。\n"
                f"サポート: {supported_types}\n"
                f"受け取った: {model_type}\n"
                f"非線形モデルにはOMLTIntegratorまたはSCIPMLIntegratorを使用してください。"
            )
        
        self.predictor = predictor
        
        # 係数と切片を抽出
        coef = predictor.coef_.flatten()
        intercept = predictor.intercept_ if hasattr(predictor, 'intercept_') else 0
        if hasattr(intercept, '__len__'):
            intercept = intercept[0]
        
        # 線形制約を構築: y = intercept + sum(coef[i] * x[i])
        linear_expr = intercept
        for i, (c, x) in enumerate(zip(coef, input_vars)):
            linear_expr += c * x
        
        # 等式制約として追加
        output_var = output_vars if not isinstance(output_vars, list) else output_vars[0]
        self.model += output_var == linear_expr, "ml_predictor_constraint"
        
        return self
    
    def add_constraint(self, constraint, name: str = None):
        """追加制約を設定"""
        self.model += constraint, name
        return self
    
    def set_objective(self, expr, sense: str = "minimize"):
        """目的関数を設定"""
        pulp = self._pulp
        
        if sense.lower() == "minimize":
            self.model.sense = pulp.LpMinimize
        else:
            self.model.sense = pulp.LpMaximize
        
        self.model.setObjective(expr)
        return self
    
    def optimize(self, sense: str = "minimize") -> OptimizationResult:
        """最適化を実行"""
        import time
        
        pulp = self._pulp
        
        # 目的関数が未設定の場合、出力変数を目的関数に
        if self.model.objective is None or len(self.model.objective) == 0:
            if self.output_vars:
                obj = self.output_vars[0]
                self.set_objective(obj, sense)
        
        # センスを設定
        if sense.lower() == "maximize":
            self.model.sense = pulp.LpMaximize
        else:
            self.model.sense = pulp.LpMinimize
        
        # 解決
        start_time = time.time()
        solver = self._get_solver()
        status = self.model.solve(solver)
        solve_time = time.time() - start_time
        
        # 結果を収集
        status_str = pulp.LpStatus[status]
        
        if status == pulp.LpStatusOptimal:
            input_values = np.array([v.varValue for v in self.input_vars])
            output_values = np.array([v.varValue for v in self.output_vars])
            objective_value = pulp.value(self.model.objective)
            
            # 予測誤差を計算
            if self.predictor is not None:
                predicted = self.predictor.predict(input_values.reshape(1, -1))[0]
                prediction_error = abs(predicted - output_values[0])
            else:
                prediction_error = 0.0
            
            return OptimizationResult(
                status=status_str,
                objective_value=objective_value,
                input_values=input_values,
                output_values=output_values,
                solve_time=solve_time,
                prediction_error=prediction_error
            )
        else:
            return OptimizationResult(
                status=status_str,
                solve_time=solve_time
            )
    
    def get_prediction_error(self) -> float:
        """最後の最適化の予測誤差を取得"""
        if self.predictor is None:
            return 0.0
        
        input_values = np.array([v.varValue for v in self.input_vars])
        output_values = np.array([v.varValue for v in self.output_vars])
        
        predicted = self.predictor.predict(input_values.reshape(1, -1))[0]
        return abs(predicted - output_values[0])


# =============================================================================
# Option 2: OMLT + Pyomo + HiGHS による高度な実装
# =============================================================================

class OMLTIntegrator(MLMIPIntegrator):
    """
    OMLT + Pyomo + HiGHSによるNN/GBT最適化
    
    サポートモデル:
    - Neural Networks (Keras, PyTorch, ONNX)
    - Gradient Boosted Trees (ONNX)
    - Linear Models (via ONNX)
    
    使用例:
    ```python
    from sklearn.neural_network import MLPRegressor
    import numpy as np
    
    # モデル訓練
    X = np.random.randn(100, 5)
    y = np.sin(X).sum(axis=1)
    model = MLPRegressor([20, 20], activation='relu', max_iter=1000).fit(X, y)
    
    # ONNXに変換
    integrator = OMLTIntegrator(solver=SolverType.HIGHS)
    integrator.create_model()
    
    # 入出力変数（バウンド必須）
    input_bounds = VariableBounds(-2, 2)
    x, y = integrator.add_sklearn_mlp(model, input_bounds, n_inputs=5)
    
    # 最適化
    result = integrator.optimize(sense="minimize")
    ```
    """
    
    def __init__(self, solver: SolverType = SolverType.HIGHS):
        self.solver = solver
        self.model = None
        self.input_vars = None
        self.output_vars = None
        self.predictor = None
        self.omlt_block = None
        self._pyo = None
        self._omlt = None
        self._import_dependencies()
    
    def _import_dependencies(self):
        """必要なライブラリをインポート"""
        try:
            import pyomo.environ as pyo
            self._pyo = pyo
        except ImportError:
            raise ImportError(
                "Pyomoがインストールされていません。\n"
                "pip install pyomo でインストールしてください。"
            )
        
        try:
            import omlt
            from omlt import OmltBlock
            from omlt.neuralnet import ReluBigMFormulation
            self._omlt = omlt
        except ImportError:
            raise ImportError(
                "OMLTがインストールされていません。\n"
                "pip install omlt でインストールしてください。"
            )
    
    def _get_solver(self):
        """Pyomoソルバーを取得"""
        pyo = self._pyo
        
        solver_map = {
            SolverType.HIGHS: "highs",
            SolverType.CBC: "cbc",
            SolverType.GLPK: "glpk",
            SolverType.SCIP: "scip",
        }
        
        solver_name = solver_map.get(self.solver)
        if solver_name is None:
            raise ValueError(f"サポートされないソルバー: {self.solver}")
        
        return pyo.SolverFactory(solver_name)
    
    def create_model(self, name: str = "OMMLTOptimization"):
        """Pyomoモデルを作成"""
        pyo = self._pyo
        self.model = pyo.ConcreteModel(name)
        return self
    
    def add_sklearn_mlp(
        self,
        sklearn_model,
        input_bounds: VariableBounds,
        n_inputs: int,
        n_outputs: int = 1
    ) -> Tuple:
        """
        scikit-learn MLPRegressorをONNX経由で追加
        
        注意: ReLU活性化関数のみサポート
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            from omlt.io import load_onnx_neural_network
            from omlt.neuralnet import ReluBigMFormulation
            from omlt import OmltBlock
        except ImportError as e:
            raise ImportError(
                f"ONNX変換に必要なライブラリがありません: {e}\n"
                "pip install skl2onnx onnx onnxruntime でインストールしてください。"
            )
        
        pyo = self._pyo
        
        # sklearn -> ONNX変換
        initial_type = [('float_input', FloatTensorType([None, n_inputs]))]
        onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type)
        
        # ONNX -> OMLT NetworkDefinition
        bounds_dict = input_bounds.to_dict(n_inputs)
        network_definition = load_onnx_neural_network(onnx_model, scaling_object=None, input_bounds=bounds_dict)
        
        # OmltBlockを作成
        self.model.nn = OmltBlock()
        
        # 定式化を構築
        formulation = ReluBigMFormulation(network_definition)
        self.model.nn.build_formulation(formulation)
        
        # 入出力変数への参照を保存
        self.input_vars = self.model.nn.inputs
        self.output_vars = self.model.nn.outputs
        self.predictor = sklearn_model
        self.omlt_block = self.model.nn
        
        return self.input_vars, self.output_vars
    
    def add_keras_model(
        self,
        keras_model,
        input_bounds: VariableBounds,
        n_inputs: int,
        scaling: Optional[Tuple] = None
    ) -> Tuple:
        """
        Kerasモデルを追加
        
        Parameters:
        -----------
        keras_model: 訓練済みKerasモデル
        input_bounds: 入力変数の上下限
        n_inputs: 入力次元数
        scaling: (offset_inputs, factor_inputs, offset_outputs, factor_outputs)
        """
        try:
            from omlt.io import load_keras_sequential
            from omlt.neuralnet import ReluBigMFormulation
            from omlt import OmltBlock, OffsetScaling
        except ImportError as e:
            raise ImportError(f"Keras統合に必要なライブラリがありません: {e}")
        
        pyo = self._pyo
        
        # スケーリング設定
        if scaling:
            scaler = OffsetScaling(
                offset_inputs=scaling[0],
                factor_inputs=scaling[1],
                offset_outputs=scaling[2],
                factor_outputs=scaling[3]
            )
        else:
            scaler = None
        
        # バウンド辞書
        bounds_dict = input_bounds.to_dict(n_inputs)
        
        # NetworkDefinitionを読み込み
        network_definition = load_keras_sequential(keras_model, scaler, bounds_dict)
        
        # OmltBlockを構築
        self.model.nn = OmltBlock()
        formulation = ReluBigMFormulation(network_definition)
        self.model.nn.build_formulation(formulation)
        
        self.input_vars = self.model.nn.inputs
        self.output_vars = self.model.nn.outputs
        self.omlt_block = self.model.nn
        
        return self.input_vars, self.output_vars
    
    def add_predictor_constraint(
        self,
        predictor: Any,
        input_vars: Any,
        output_vars: Any,
        input_bounds: VariableBounds = None,
        **kwargs
    ):
        """汎用的な予測モデル制約追加"""
        model_type = type(predictor).__name__
        
        if 'MLP' in model_type:
            n_inputs = predictor.n_features_in_
            if input_bounds is None:
                input_bounds = VariableBounds(-10, 10)
                warnings.warn("input_boundsが指定されていません。デフォルト[-10, 10]を使用します。")
            return self.add_sklearn_mlp(predictor, input_bounds, n_inputs)
        
        elif 'Sequential' in model_type or 'Model' in model_type:
            n_inputs = predictor.input_shape[1]
            if input_bounds is None:
                input_bounds = VariableBounds(-10, 10)
            return self.add_keras_model(predictor, input_bounds, n_inputs)
        
        else:
            raise ValueError(
                f"OMLTIntegratorでサポートされないモデル: {model_type}\n"
                f"線形モデルにはPuLPLinearMLIntegratorを使用してください。"
            )
    
    def add_constraint(self, constraint_expr, name: str = None):
        """Pyomo制約を追加"""
        pyo = self._pyo
        
        if name is None:
            name = f"constraint_{len(list(self.model.component_objects(pyo.Constraint)))}"
        
        self.model.add_component(name, pyo.Constraint(expr=constraint_expr))
        return self
    
    def set_objective(self, expr, sense: str = "minimize"):
        """目的関数を設定"""
        pyo = self._pyo
        
        sense_map = {
            "minimize": pyo.minimize,
            "maximize": pyo.maximize
        }
        
        self.model.obj = pyo.Objective(expr=expr, sense=sense_map[sense.lower()])
        return self
    
    def optimize(self, sense: str = "minimize") -> OptimizationResult:
        """最適化を実行"""
        import time
        
        pyo = self._pyo
        
        # 目的関数が未設定の場合
        if not hasattr(self.model, 'obj'):
            if self.output_vars is not None:
                # 出力変数の合計を目的関数に
                output_expr = sum(self.output_vars[i] for i in self.output_vars)
                self.set_objective(output_expr, sense)
        
        # ソルバー取得と実行
        solver = self._get_solver()
        start_time = time.time()
        
        try:
            results = solver.solve(self.model, tee=False)
            solve_time = time.time() - start_time
            
            status = str(results.solver.status)
            termination = str(results.solver.termination_condition)
            
            if termination.lower() == 'optimal':
                # 値を抽出
                input_values = np.array([pyo.value(self.input_vars[i]) for i in self.input_vars])
                output_values = np.array([pyo.value(self.output_vars[i]) for i in self.output_vars])
                objective_value = pyo.value(self.model.obj)
                
                # 予測誤差
                if self.predictor is not None:
                    predicted = self.predictor.predict(input_values.reshape(1, -1))[0]
                    prediction_error = abs(predicted - output_values[0])
                else:
                    prediction_error = 0.0
                
                return OptimizationResult(
                    status=f"{status}/{termination}",
                    objective_value=objective_value,
                    input_values=input_values,
                    output_values=output_values,
                    solve_time=solve_time,
                    prediction_error=prediction_error
                )
            else:
                return OptimizationResult(
                    status=f"{status}/{termination}",
                    solve_time=solve_time
                )
        
        except Exception as e:
            return OptimizationResult(
                status=f"error: {str(e)}",
                solve_time=time.time() - start_time
            )
    
    def get_prediction_error(self) -> float:
        """予測誤差を取得"""
        if self.predictor is None or self.input_vars is None:
            return 0.0
        
        pyo = self._pyo
        input_values = np.array([pyo.value(self.input_vars[i]) for i in self.input_vars])
        output_values = np.array([pyo.value(self.output_vars[i]) for i in self.output_vars])
        
        predicted = self.predictor.predict(input_values.reshape(1, -1))[0]
        return abs(predicted - output_values[0])


# =============================================================================
# Option 3: PySCIPOpt-ML による実装
# =============================================================================

class SCIPMLIntegrator(MLMIPIntegrator):
    """
    PySCIPOpt-ML によるscikit-learn直接統合
    
    サポートモデル:
    - LinearRegression, Ridge, Lasso
    - DecisionTreeRegressor
    - RandomForestRegressor
    - GradientBoostingRegressor
    - MLPRegressor
    - XGBoost, LightGBM
    
    使用例:
    ```python
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # モデル訓練
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + X[:, 1] ** 2
    model = RandomForestRegressor(n_estimators=10, max_depth=5).fit(X, y)
    
    # 最適化
    integrator = SCIPMLIntegrator()
    integrator.create_model()
    
    x = integrator.add_input_vars(5, lb=-5, ub=5)
    y = integrator.add_output_var()
    
    integrator.add_predictor_constraint(model, x, y)
    
    result = integrator.optimize(sense="minimize")
    ```
    """
    
    def __init__(self):
        self.model = None
        self.input_vars = []
        self.output_vars = []
        self.predictor = None
        self.pred_constr = None
        self._scip = None
        self._pyscipopt_ml = None
        self._import_dependencies()
    
    def _import_dependencies(self):
        """PySCIPOpt-MLをインポート"""
        try:
            from pyscipopt import Model
            self._scip = Model
        except ImportError:
            raise ImportError(
                "PySCIPOptがインストールされていません。\n"
                "pip install pyscipopt でインストールしてください。"
            )
        
        try:
            import pyscipopt_ml
            self._pyscipopt_ml = pyscipopt_ml
        except ImportError:
            raise ImportError(
                "PySCIPOpt-MLがインストールされていません。\n"
                "pip install pyscipopt-ml でインストールしてください。"
            )
    
    def create_model(self, name: str = "SCIPMLOptimization"):
        """SCIPモデルを作成"""
        self.model = self._scip(name)
        self.model.hideOutput()
        self.input_vars = []
        self.output_vars = []
        return self
    
    def add_input_vars(
        self,
        n_vars: int,
        lb: Union[float, List[float]] = None,
        ub: Union[float, List[float]] = None,
        names: Optional[List[str]] = None
    ) -> np.ndarray:
        """入力変数を追加"""
        if names is None:
            names = [f"x_{i}" for i in range(n_vars)]
        
        lb_list = [lb] * n_vars if np.isscalar(lb) or lb is None else lb
        ub_list = [ub] * n_vars if np.isscalar(ub) or ub is None else ub
        
        vars_array = np.empty(n_vars, dtype=object)
        for i, name in enumerate(names):
            var = self.model.addVar(
                name=name,
                vtype="C",
                lb=lb_list[i],
                ub=ub_list[i]
            )
            vars_array[i] = var
        
        self.input_vars = vars_array
        return vars_array
    
    def add_output_var(
        self,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
        name: str = "y"
    ):
        """出力変数を追加"""
        var = self.model.addVar(name=name, vtype="C", lb=lb, ub=ub)
        self.output_vars.append(var)
        return var
    
    def add_predictor_constraint(
        self,
        predictor: Any,
        input_vars: Any,
        output_vars: Any,
        epsilon: float = 0.0,
        **kwargs
    ):
        """
        MLモデルを制約として追加
        
        Parameters:
        -----------
        predictor: scikit-learnモデル
        input_vars: 入力変数配列
        output_vars: 出力変数（単一またはリスト）
        epsilon: 決定木の分岐閾値処理パラメータ
        """
        from pyscipopt_ml import add_predictor_constr
        
        self.predictor = predictor
        
        # 出力変数の形式を調整
        if not isinstance(output_vars, (list, np.ndarray)):
            output_vars = np.array([output_vars])
        
        # 入力変数をreshape
        input_array = np.array(input_vars).reshape(1, -1)
        output_array = np.array(output_vars).reshape(1, -1)
        
        # 制約追加
        self.pred_constr = add_predictor_constr(
            self.model,
            predictor,
            input_array,
            output_array,
            epsilon=epsilon
        )
        
        return self.pred_constr
    
    def add_constraint(self, lhs, sense: str, rhs, name: str = None):
        """制約を追加"""
        self.model.addCons(lhs <= rhs if sense == "<=" else (lhs >= rhs if sense == ">=" else lhs == rhs), name=name)
        return self
    
    def set_objective(self, expr, sense: str = "minimize"):
        """目的関数を設定"""
        self.model.setObjective(expr, sense=sense)
        return self
    
    def optimize(self, sense: str = "minimize", time_limit: float = None) -> OptimizationResult:
        """最適化を実行"""
        import time
        
        # 目的関数が未設定の場合
        if self.output_vars:
            self.set_objective(self.output_vars[0], sense)
        
        if time_limit:
            self.model.setParam("limits/time", time_limit)
        
        start_time = time.time()
        self.model.optimize()
        solve_time = time.time() - start_time
        
        status = self.model.getStatus()
        
        if status == "optimal":
            input_values = np.array([self.model.getVal(v) for v in self.input_vars])
            output_values = np.array([self.model.getVal(v) for v in self.output_vars])
            objective_value = self.model.getObjVal()
            
            # 予測誤差
            if self.pred_constr is not None:
                try:
                    prediction_error = np.max(self.pred_constr.get_error())
                except:
                    if self.predictor is not None:
                        predicted = self.predictor.predict(input_values.reshape(1, -1))[0]
                        prediction_error = abs(predicted - output_values[0])
                    else:
                        prediction_error = 0.0
            else:
                prediction_error = 0.0
            
            return OptimizationResult(
                status=status,
                objective_value=objective_value,
                input_values=input_values,
                output_values=output_values,
                solve_time=solve_time,
                prediction_error=prediction_error
            )
        else:
            return OptimizationResult(
                status=status,
                solve_time=solve_time
            )
    
    def get_prediction_error(self) -> float:
        """予測誤差を取得"""
        if self.pred_constr is not None:
            try:
                return np.max(self.pred_constr.get_error())
            except:
                pass
        
        if self.predictor is None:
            return 0.0
        
        input_values = np.array([self.model.getVal(v) for v in self.input_vars])
        output_values = np.array([self.model.getVal(v) for v in self.output_vars])
        
        predicted = self.predictor.predict(input_values.reshape(1, -1))[0]
        return abs(predicted - output_values[0])


# =============================================================================
# ファクトリークラス
# =============================================================================

class MLMIPFactory:
    """
    MLモデルタイプとソルバーに応じた最適なインテグレーターを選択するファクトリー
    
    使用例:
    ```python
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    # 線形モデル -> PuLP + HiGHS
    linear_model = Ridge().fit(X, y)
    integrator = MLMIPFactory.create(linear_model, solver=SolverType.HIGHS)
    
    # ツリーモデル -> SCIP
    tree_model = RandomForestRegressor().fit(X, y)
    integrator = MLMIPFactory.create(tree_model)  # 自動でSCIPを選択
    ```
    """
    
    # モデルタイプとインテグレーターのマッピング
    LINEAR_MODELS = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'PLSRegression']
    TREE_MODELS = ['DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor',
                   'XGBRegressor', 'LGBMRegressor']
    NN_MODELS = ['MLPRegressor', 'Sequential', 'Model']
    
    @classmethod
    def create(
        cls,
        model: Any = None,
        solver: SolverType = None,
        prefer_lightweight: bool = True
    ) -> MLMIPIntegrator:
        """
        最適なインテグレーターを作成
        
        Parameters:
        -----------
        model: 訓練済みMLモデル（オプション）
        solver: 使用するソルバー（オプション）
        prefer_lightweight: 軽量実装を優先するか
        
        Returns:
        --------
        MLMIPIntegrator: 適切なインテグレーター
        """
        if model is None:
            # モデル未指定の場合はデフォルト
            if solver == SolverType.SCIP:
                return SCIPMLIntegrator()
            else:
                return PuLPLinearMLIntegrator(solver or SolverType.HIGHS)
        
        model_type = type(model).__name__
        
        # 線形モデル
        if model_type in cls.LINEAR_MODELS:
            if prefer_lightweight:
                return PuLPLinearMLIntegrator(solver or SolverType.HIGHS)
            else:
                return SCIPMLIntegrator()
        
        # ツリーモデル
        elif model_type in cls.TREE_MODELS:
            # PySCIPOpt-MLが最も直接的なサポート
            return SCIPMLIntegrator()
        
        # ニューラルネットワーク
        elif any(nn in model_type for nn in cls.NN_MODELS):
            if solver == SolverType.SCIP:
                return SCIPMLIntegrator()
            else:
                return OMLTIntegrator(solver or SolverType.HIGHS)
        
        else:
            # 不明なモデルタイプ
            warnings.warn(f"不明なモデルタイプ: {model_type}。SCIPMLIntegratorを使用します。")
            return SCIPMLIntegrator()
    
    @classmethod
    def get_supported_models(cls) -> Dict[str, List[str]]:
        """サポートされるモデルタイプを取得"""
        return {
            "linear": cls.LINEAR_MODELS,
            "tree": cls.TREE_MODELS,
            "neural_network": cls.NN_MODELS
        }


# =============================================================================
# 便利なユーティリティ関数
# =============================================================================

def quick_optimize(
    model: Any,
    X_train: np.ndarray,
    objective: str = "minimize",
    input_bounds: Optional[VariableBounds] = None,
    additional_constraints: Optional[List] = None,
    solver: SolverType = SolverType.HIGHS
) -> OptimizationResult:
    """
    クイック最適化関数
    
    訓練データから自動的にバウンドを推定し、最適化を実行
    
    Parameters:
    -----------
    model: 訓練済みMLモデル
    X_train: 訓練データ（バウンド推定用）
    objective: "minimize" or "maximize"
    input_bounds: 手動で指定するバウンド（オプション）
    additional_constraints: 追加制約のリスト（オプション）
    solver: 使用するソルバー
    
    Returns:
    --------
    OptimizationResult: 最適化結果
    
    使用例:
    ```python
    from sklearn.linear_model import LinearRegression
    
    X = np.random.randn(100, 5)
    y = X @ np.array([1, 2, 3, 4, 5])
    model = LinearRegression().fit(X, y)
    
    result = quick_optimize(model, X, objective="maximize")
    print(f"最適値: {result.objective_value}")
    print(f"最適入力: {result.input_values}")
    ```
    """
    n_features = X_train.shape[1]
    
    # バウンドを自動推定
    if input_bounds is None:
        margin = 0.1
        lb = X_train.min(axis=0) * (1 + margin)
        ub = X_train.max(axis=0) * (1 + margin)
        input_bounds = VariableBounds(lb, ub)
    
    # インテグレーターを作成
    integrator = MLMIPFactory.create(model, solver)
    integrator.create_model()
    
    # モデルタイプに応じた処理
    model_type = type(model).__name__
    
    if model_type in MLMIPFactory.LINEAR_MODELS:
        # PuLP用
        x = integrator.add_input_vars(
            n_features,
            lb=input_bounds.lower if np.isscalar(input_bounds.lower) else input_bounds.lower.tolist(),
            ub=input_bounds.upper if np.isscalar(input_bounds.upper) else input_bounds.upper.tolist()
        )
        y = integrator.add_output_var()
        integrator.add_predictor_constraint(model, x, y)
        
    elif isinstance(integrator, SCIPMLIntegrator):
        # SCIP用
        lb = input_bounds.lower if np.isscalar(input_bounds.lower) else input_bounds.lower.tolist()
        ub = input_bounds.upper if np.isscalar(input_bounds.upper) else input_bounds.upper.tolist()
        x = integrator.add_input_vars(n_features, lb=lb, ub=ub)
        y = integrator.add_output_var()
        integrator.add_predictor_constraint(model, x, y)
        
    elif isinstance(integrator, OMLTIntegrator):
        # OMLT用
        x, y = integrator.add_predictor_constraint(model, None, None, input_bounds=input_bounds)
    
    # 最適化実行
    result = integrator.optimize(sense=objective)
    
    return result


# =============================================================================
# バージョン情報
# =============================================================================

__version__ = "1.0.0"
__author__ = "Q-Storm Platform"


if __name__ == "__main__":
    # 簡単なテスト
    print("Q-Storm ML-MIP Integration Library")
    print(f"Version: {__version__}")
    print("\nサポートされるモデル:")
    for category, models in MLMIPFactory.get_supported_models().items():
        print(f"  {category}: {', '.join(models)}")
