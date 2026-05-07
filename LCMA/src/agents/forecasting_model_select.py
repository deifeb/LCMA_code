#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测模型选择智能体: 为不同类别的备件推荐最优预测算法和参数
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
from itertools import product

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsforecast import StatsForecast
from statsforecast.models import CrostonClassic, ADIDA, IMAPA
from xgboost import XGBRegressor
from src.model_interface import OllamaInterface

try:
    from statsforecast.models import CrostonSBA, TSB
except ImportError:
    CrostonSBA = None
    TSB = None


class ForecastingModelSelectAgent:
    """预测模型选择器: 为不同类别的备件推荐最优预测算法和参数"""

    def __init__(self, model_interface: OllamaInterface):
        self.model = model_interface
        self.system_prompt = """你是一名专业的预测算法专家。
你的任务是分析不同类别备件的特征模式，并为每类备件推荐最优的预测算法和参数配置。
你需要考虑数据特性、预测难度、算法适用性和计算效率等因素。
请以JSON格式提供你的分析结果和算法推荐。"""
        self.algorithm_history = {}  # 存储历史算法效果
        self.knowledge_manager = None
        self.available_algorithms = {
            'arima': 'ARIMA',  # 自回归集成移动平均模型
            'sarima': 'SARIMA',  # 季节性ARIMA
            'prophet': 'Prophet',  # Facebook Prophet
            'lstm': 'LSTM',  # 长短期记忆网络
            'xgboost': 'XGBoost',  # XGBoost时序版本
            'croston': 'Croston',  # Croston方法（适用于间歇性需求）
            'tsb': 'TSB',  # Teunter-Syntetos-Babai间歇性需求模型
            'sba': 'SBA',  # Syntetos-Boylan Approximation
            'rf': 'RF',  # 随机森林回归
            'random_forest': 'RF',  # 随机森林回归别名
            'lightgbm': 'LightGBM',  # LightGBM梯度提升树
            'lgbm': 'LightGBM',  # LightGBM别名
            'ses': 'Simple Exponential Smoothing',  # 简单指数平滑
            'des': 'Double Exponential Smoothing',  # 双指数平滑
            'tes': 'Triple Exponential Smoothing'  # 三指数平滑
        }
        
    def set_knowledge_manager(self, knowledge_manager):
        """设置知识库管理器
        
        Args:
            knowledge_manager: 智能体知识库管理器实例
        """
        self.knowledge_manager = knowledge_manager
        
    def get_knowledge(self, query: str, top_k: int = 3):
        """从预测模型选择智能体知识库中检索知识
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 相关文档列表
        """
        if self.knowledge_manager:
            return self.knowledge_manager.search_knowledge("forecasting_model_select", query, top_k)
        return []

    def generate_prediction_strategy(self, cluster_info: Dict, evaluation_results: Dict, analysis: Dict, knowledge: str) -> Dict:
        """生成最终的预测策略及解释
        
        Args:
            cluster_info: 聚类分析结果
            evaluation_results: 评估结果
            analysis: 分析结果
            knowledge: 从知识库检索到的预测策略相关知识
            
        Returns:
            Dict: 预测策略
        """
        demand_types = list(evaluation_results.keys())
        class1 = len(evaluation_results[demand_types[0]]['clusters'])
        class2 = len(evaluation_results[demand_types[1]]['clusters'])

        # 构建提示词
        prompt = f"""
根据以下信息，生成最终的备件需求预测策略：

以下均为{demand_types[0]}需求模式下的聚类信息、算法评估结果、结果分析与反馈:
{json.dumps(cluster_info[demand_types[0]], indent=2, ensure_ascii=False)}
{json.dumps(evaluation_results[demand_types[0]], indent=2, ensure_ascii=False)}
{json.dumps(analysis[demand_types[0]], indent=2, ensure_ascii=False)}
以下均为{demand_types[1]}需求模式下的聚类信息、算法评估结果、结果分析与反馈:
{json.dumps(cluster_info[demand_types[1]], indent=2, ensure_ascii=False)}
{json.dumps(evaluation_results[demand_types[1]], indent=2, ensure_ascii=False)}
{json.dumps(analysis[demand_types[1]], indent=2, ensure_ascii=False)}

请根据上述内容，按照以下步骤进行分析:
1. 每个类别的最终预测策略
2. 实施预测的具体步骤
3. 预测结果的解释方法
4. 预测不确定性的处理建议

以JSON格式返回，格式如下:
{{
  "{demand_types[0]}": {{
    "prediction_strategies": {{
      "类别ID1": {{
        "label": "类别标签",
        "final_algorithm": "算法名称"
      }},
      ...,
      "类别ID7": {{
        "label": "类别标签",
        "final_algorithm": "算法名称"
      }}
    }}
  }},
  "{demand_types[1]}": {{
    "prediction_strategies": {{
      "类别ID1": {{
        "label": "类别标签",
        "final_algorithm": "算法名称"
      }},
    ...,
      "类别ID12": {{
        "label": "类别标签",
        "final_algorithm": "算法名称"
      }}
    }}
  }},
  "implementation_plan": {{
    "data_preparation":  [准备步骤1, 准备步骤2],
    "execution_sequence": [执行步骤1, 执行步骤2],
  }},
  "is_recommend_adjustments": {{
    "feature_extraction": "true/false",
    "classification": "true/false",
    "forecast_algorithm": "true/false"
  }}
}}
注意：请严格按照上述格式返回，确保所有字段都包含在JSON中。
1. {demand_types[0]}中包含{class1}个类别，{demand_types[1]}中包含{class2}个类别，类别ID应该是从1开始，且连续，每种需求模式下各类别都需要完整写出来，不能省略，注意两种需求模式的类别ID个数不一样。
2. is_recommend表示基于预测结果分析情况，是否需要对特征提取、分类、预测算法进行调整，true表示需要调整，false表示不需要调整。
3. implementation_plan表示完整的预测策略实施计划，包括特征提取智能体、备件分类智能体和模型选择智能体。
"""

        response = self.model.query(prompt, system_prompt=self.system_prompt)
        prediction_strategy = self.model.parse_json_response(response['content'])

        return prediction_strategy

    def analyze_evaluation(self, evaluation_results: Dict, cluster_analysis: Dict, knowledge: str) -> Dict:
        """分析评估结果并生成反馈
        
        Args:
            evaluation_results: 评估结果
            cluster_analysis: 聚类分析结果
            knowledge: 从知识库检索到的评估分析相关知识
            
        Returns:
            Dict: 分析结果
        """
        demand_types = list(evaluation_results.keys())
        demand_type1 = {cluster['id']: cluster['label'] for cluster in
                        cluster_analysis[demand_types[0]]['clusters']}
        demand_type2 = {cluster['id']: cluster['label'] for cluster in
                        cluster_analysis[demand_types[1]]['clusters']}
        class1 = len(evaluation_results[demand_types[0]]['clusters'])
        class2 = len(evaluation_results[demand_types[1]]['clusters'])
        print("······································")
        print(class1, class2)
        print("······································")
        algorithms = self._get_algorithm_display_names()
        prompt = f"""
针对基于{len(demand_types)}种需求特征划分的不同聚类类别，系统评估了备件库存预测模型的性能表现，并基于分析结果提出特征提取、备件聚类两个维度的优化建议。
具体而言：首先阐述各需求类别的预测精度对比数据；其次揭示预测误差的分布特征；最后给出特征提取、备件聚类两个层面给出改进方案。
1. {demand_types[0]}需求模式，包含有{class1}个类别，所以类别ID应该是从1至{class1}，各类别预测效果如下:
{json.dumps(evaluation_results[demand_types[0]], indent=2, ensure_ascii=False)}
2. {demand_types[1]}需求模式，包含有{class2}个类别，所以类别ID应该是从1至{class2}，各类别预测效果如下:
{json.dumps(evaluation_results[demand_types[1]], indent=2, ensure_ascii=False)}
3. {demand_types[1]}需求模式下的类别ID与label特征对应关系：
{json.dumps(demand_type2, indent=2, ensure_ascii=False)}
4. {demand_types[0]}需求模式下的类别ID与label特征对应关系：
{json.dumps(demand_type1, indent=2, ensure_ascii=False)}
5. 预测算法列表：
请给出以下内容，下面的分析过程和结果都是基于evaluation_results进行的；评估指标包括MAE、RMSE、RRMSE、MASE、RelMAE、sMAPE、NRMSE，数值越低通常代表预测越优。分析内容包括:
1. 每个类别的预测效果评估
2. 预测算法调整建议，基于已有的算法性能表现，给出调整建议。
3. 需要将对预测结果的分析情况反馈给特征提取和分类智能体
4. 整体预测策略的改进方向
5. {demand_types[0]}需求模式，包含有{class1}个类别，所以类别ID应该是从1至{class1}，严格按照类别ID数量给出分析;
6. {demand_types[1]}需求模式，包含有{class2}个类别，所以类别ID应该是从1至{class2}，严格按照类别ID数量给出分析。

{{
  "{demand_types[0]}": {{
    "forecast_algorithms": {{
      "类别ID1": {{
        "label": "{demand_type1[str(1)]}，类别ID与label特征对应一一对应",
        "recommended_algorithm": "在预测效果评估后推荐预测算法名称，如果预测效果良好，保持原始算法（与各类别预测效果中各需求模式和类别下的算法对应），否则更换算法，给出具体的算法名称",
        "recommendation_reason": "说明修改算法或保持原算法的理由"
      }},
     ... ,
      "类别ID{class1}": {{
        "label": "{demand_type1[str(class1)]}，类别ID与label特征对应一一对应",
        "recommended_algorithm": "在预测效果评估后推荐预测算法名称，如果预测效果良好，保持原始算法（与各类别预测效果中各需求模式和类别下的算法对应），否则更换算法，给出具体的算法名称",
        "recommendation_reason": "说明修改算法或保持原算法的理由"
      }}
    }},
    "feature_extraction_feedback": {{
      "feature_adjustments": [
        {{
          "feature": "需要新增的特征名(备品备件消耗相关特征、间歇类特征或时序特征)",
          "adjustment": "是否需要改进调整特征，ture or false",
          "reason": "调整理由"
        }}
      ]
    }},
    "classification_feedback": {{
      "clustering_assessment": "是否需要更换其他的聚类算法,ture or false",
      "cluster_adjustments": "类别调整建议"
    }}
  }},
  "{demand_types[1]}": {{
    "forecast_algorithms": {{
      "类别ID1": {{
        "label": "{demand_type2[str(1)]}，类别ID与label特征对应一一对应",
        "recommended_algorithm": "在预测效果评估后推荐预测算法名称，如果预测效果良好，保持原始算法（与各类别预测效果中各需求模式和类别下的算法对应），否则更换算法，给出具体的算法名称",
        "recommendation_reason": "说明修改算法或保持原算法的理由"
      }},
    ... ,
      "类别ID{class2}": {{
        "label": "{demand_type2[str(class2)]}，类别ID与label特征对应一一对应",
        "recommended_algorithm": "在预测效果评估后推荐预测算法名称，如果预测效果良好，保持原始算法（与各类别预测效果中各需求模式和类别下的算法对应），否则更换算法，给出具体的算法名称",
        "recommendation_reason": "说明修改算法或保持原算法的理由"
      }}
    }},
    "feature_extraction_feedback": {{
      "feature_adjustments": [
        {{
          "feature": "需要新增的特征名(备品备件消耗相关特征、间歇类特征或时序特征)",
          "adjustment": "是否需要改进调整特征，ture or false",
          "reason": "调整理由"
        }}
      ]
    }},
    "classification_feedback": {{
      "clustering_assessment": "当前需求模式下各类的聚类效果评估",
      "cluster_adjustments": "是否需要更换其他的聚类算法,ture or false",
      "reason":"调整理由"
    }}
  }}
}}
注意：
1. 改进策略中的recommended_algorithm应该均来自{algorithms}，不需要额外添加，各需求模式下各类别使用的原始算法均在evaluation_results变量中给出。
2. important_features应与原始数据中的特征名称一致。
3. suggested_new_features中的特征名称可根据预测效果进行调整。
4. 请确保输出的格式符合JSON要求，并且包含所有必要的字段。
5. recommended_algorithm中给出的建议是针对每个需求模式下各类别的预测效果进行的；feature_extraction_feedback，classification_feedback是针对各个需求模式的。
6，需求模式下的类别ID应该是从1开始，且连续，每种需求模式下每个类别都需要完整写出来，不能省略，注意两种需求模式的类别ID个数不一样。
"""

        response = self.model.query(prompt, system_prompt=self.system_prompt)
        analysis_results = self.model.parse_json_response(response["content"])
        return analysis_results

    def recommend_algorithms(self, cluster_info: Dict, features_explanation: Dict, knowledge: str) -> Dict:
        """为每个类别推荐预测算法
        
        Args:
            cluster_info: 聚类分析结果
            features_explanation: 特征解释结果
            knowledge: 从知识库检索到的算法推荐相关知识
            
        Returns:
            Dict: 算法推荐结果
        """
        demand_types = list(cluster_info.keys())
        print(f"需求类型: {demand_types}")
        
        # 获取每种需求类型的类别数量
        class1 = len(cluster_info[demand_types[0]]['clusters'])
        class2 = len(cluster_info[demand_types[1]]['clusters'])
        print(f"类别数量: {demand_types[0]}={class1}, {demand_types[1]}={class2}")
        
        # 直接使用默认推荐，避免模型调用的不稳定性
        print("⚠️ 为了确保稳定性，直接使用默认算法推荐")
        return self._get_default_algorithm_recommendations(demand_types, class1, class2)

    def _validate_algorithm_recommendations(self, recommendations: Dict, demand_types: List, class1: int, class2: int) -> bool:
        """验证算法推荐结果格式"""
        try:
            # 检查基本结构
            if not isinstance(recommendations, dict):
                return False
            
            # 检查需求类型是否完整
            for demand_type in demand_types:
                if demand_type not in recommendations:
                    return False
                
                if not isinstance(recommendations[demand_type], list):
                    return False
            
            # 检查第一个需求类型的类别数量
            if len(recommendations[demand_types[0]]) != class1:
                return False
                
            # 检查第二个需求类型的类别数量  
            if len(recommendations[demand_types[1]]) != class2:
                return False
            
            # 检查每个推荐项的必要字段
            algorithms = self._get_algorithm_display_names()
            
            for demand_type in demand_types:
                for rec in recommendations[demand_type]:
                    if not isinstance(rec, dict):
                        return False
                    
                    required_fields = ['id', 'recommended_algorithm', 'recommendation_reason', 'algorithm_parameters']
                    for field in required_fields:
                        if field not in rec:
                            return False
                    
                    # 检查算法名称是否在可用列表中
                    if rec['recommended_algorithm'] not in algorithms:
                        return False
            
            return True
            
        except Exception as e:
            print(f"验证异常: {e}")
            return False

    def _get_default_algorithm_recommendations(self, demand_types: List, class1: int, class2: int) -> Dict:
        """生成默认的算法推荐"""
        default_recommendations = {}
        
        # 为间歇性需求推荐算法（循环使用算法列表）
        intermittent_algorithms = self._deduplicate_algorithms(
            ["Croston", "TSB", "SBA", "ARIMA", "移动平均", "指数平滑"]
        )
        default_recommendations[demand_types[0]] = []
        for i in range(class1):
            algo = intermittent_algorithms[i % len(intermittent_algorithms)]
            default_recommendations[demand_types[0]].append({
                "id": str(i + 1),
                "recommended_algorithm": algo,
                "recommendation_reason": f"基于{demand_types[0]}特征的默认推荐 - {algo}适合间歇性需求模式",
                "algorithm_parameters": self._get_default_parameters(algo)
            })
        
        # 为块状需求推荐算法（循环使用算法列表）
        lumpy_algorithms = self._deduplicate_algorithms(
            ["LightGBM", "RF", "XGBoost", "LSTM", "Prophet", "ARIMA", "指数平滑"]
        )
        default_recommendations[demand_types[1]] = []
        for i in range(class2):
            algo = lumpy_algorithms[i % len(lumpy_algorithms)]
            default_recommendations[demand_types[1]].append({
                "id": str(i + 1),
                "recommended_algorithm": algo,
                "recommendation_reason": f"基于{demand_types[1]}特征的默认推荐 - {algo}适合块状需求模式", 
                "algorithm_parameters": self._get_default_parameters(algo)
            })
        
        print(f"生成默认推荐: {demand_types[0]}({class1}个类别), {demand_types[1]}({class2}个类别)")
        return default_recommendations
    
    def _get_default_parameters(self, algorithm: str) -> Dict:
        """获取算法的默认参数"""
        params_map = {
            "ARIMA": {"season_length": 12},
            "指数平滑": {"season_length": 12},
            "Prophet": {"season_length": 12},
            "XGBoost": {"n_estimators": 100, "learning_rate": 0.1},
            "LightGBM": {"n_estimators": 120, "learning_rate": 0.05, "num_leaves": 31},
            "RF": {"n_estimators": 120, "max_depth": 8},
            "LSTM": {"epochs": 20, "lookback": 12, "units": 32},
            "简单平均": {},
            "移动平均": {"window": 3},
            "Croston": {"alpha": 0.1},
            "TSB": {"alpha_d": 0.1, "alpha_p": 0.1},
            "SBA": {"alpha": 0.1}
        }
        return params_map.get(algorithm, {})

    def optimize_algorithm_parameters(
        self, data: pd.DataFrame, algorithm: str, base_params: Dict = None
    ) -> Tuple[Dict, Dict]:
        """为指定预测模型执行轻量参数优化。

        采用时间序列holdout验证：将当前聚类训练数据再次按时间切分，
        使用前段拟合、后段验证，以MAE作为搜索目标。
        """
        base_params = dict(base_params or {})
        algorithm_key = self._normalize_algorithm_name(algorithm)
        search_space = self._get_parameter_search_space(algorithm_key)

        if data is None or data.shape[1] < 8:
            return base_params, {
                "enabled": False,
                "method": "time_series_holdout_grid_search",
                "reason": "时间长度不足，保留默认参数",
                "best_params": base_params
            }

        if not search_space:
            return base_params, {
                "enabled": False,
                "method": "time_series_holdout_grid_search",
                "reason": "该算法使用内置自动参数或无需显式调参",
                "best_params": base_params
            }

        validation_size = max(1, min(max(2, data.shape[1] // 5), data.shape[1] - 4))
        opt_train = data.iloc[:, :-validation_size]
        opt_val = data.iloc[:, -validation_size:]

        best_params = dict(base_params)
        best_score = float("inf")
        candidates = self._build_parameter_candidates(search_space, base_params)

        # LSTM调参成本较高，限制候选组数；其他模型也保持轻量，避免阻塞主流程。
        max_candidates = 4 if algorithm_key == "lstm" else 8
        evaluated = []

        for candidate in candidates[:max_candidates]:
            try:
                predictions, model_info = self._apply_algorithm(opt_train, opt_val, algorithm_key, candidate)
                aligned_predictions = self._align_predictions(predictions, opt_val)
                score = mean_absolute_error(opt_val.values, aligned_predictions)
                evaluated.append({
                    "params": candidate,
                    "mae": float(score),
                    "model_status": model_info.get("status", "unknown")
                })

                if score < best_score:
                    best_score = score
                    best_params = candidate
            except Exception as e:
                evaluated.append({
                    "params": candidate,
                    "mae": None,
                    "error": str(e)
                })

        if not evaluated or best_score == float("inf"):
            return base_params, {
                "enabled": True,
                "method": "time_series_holdout_grid_search",
                "reason": "候选参数评估失败，保留默认参数",
                "best_params": base_params,
                "candidates": evaluated
            }

        return best_params, {
            "enabled": True,
            "method": "time_series_holdout_grid_search",
            "metric": "MAE",
            "validation_size": validation_size,
            "best_score": float(best_score),
            "best_params": best_params,
            "candidates": evaluated
        }

    def _get_parameter_search_space(self, algorithm: str) -> Dict[str, List[Any]]:
        """定义各预测模型的参数优化搜索空间。"""
        spaces = {
            "arima": {
                "season_length": [1, 3, 6, 12]
            },
            "exponential smoothing": {
                "season_length": [1, 3, 6, 12]
            },
            "prophet": {
                "changepoint_prior_scale": [0.01, 0.05, 0.1],
                "seasonality_prior_scale": [1.0, 5.0, 10.0]
            },
            "xgboost": {
                "n_estimators": [80, 120],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 6]
            },
            "lightgbm": {
                "n_estimators": [80, 120],
                "learning_rate": [0.03, 0.05, 0.1],
                "num_leaves": [15, 31]
            },
            "rf": {
                "n_estimators": [80, 120],
                "max_depth": [5, 8, None],
                "min_samples_leaf": [1, 3]
            },
            "lstm": {
                "lookback": [6, 12],
                "units": [16, 32],
                "epochs": [10, 20]
            },
            "croston": {
                "alpha": [0.05, 0.1, 0.2, 0.3]
            },
            "sba": {
                "alpha": [0.05, 0.1, 0.2, 0.3]
            },
            "tsb": {
                "alpha_d": [0.05, 0.1, 0.2],
                "alpha_p": [0.05, 0.1, 0.2]
            },
            "adida": {
                "alpha": [0.05, 0.1, 0.2]
            },
            "imapa": {
                "alpha": [0.05, 0.1, 0.2]
            },
            "sma": {
                "window": [2, 3, 6]
            }
        }
        return spaces.get(algorithm, {})

    def _build_parameter_candidates(self, search_space: Dict[str, List[Any]], base_params: Dict) -> List[Dict]:
        """由搜索空间生成参数候选，并确保默认参数优先评估。"""
        keys = list(search_space.keys())
        candidates = []

        if base_params:
            candidates.append(dict(base_params))

        for values in product(*[search_space[key] for key in keys]):
            candidate = dict(zip(keys, values))
            merged = dict(base_params or {})
            merged.update(candidate)
            if merged not in candidates:
                candidates.append(merged)

        return candidates

    def _normalize_algorithm_name(self, algorithm: str) -> str:
        """统一算法名称，便于参数空间和执行分支复用。"""
        key = str(algorithm or "").strip().lower()
        alias_map = {
            "指数平滑": "exponential smoothing",
            "ets": "exponential smoothing",
            "simple exponential smoothing": "exponential smoothing",
            "double exponential smoothing": "exponential smoothing",
            "triple exponential smoothing": "exponential smoothing",
            "移动平均": "sma",
            "simple moving average": "sma",
            "random forest": "rf",
            "random_forest": "rf",
            "随机森林": "rf",
            "lgbm": "lightgbm",
            "lightgbm": "lightgbm",
            "xgb": "xgboost"
        }
        return alias_map.get(key, key)

    def _align_predictions(self, predictions, target: pd.DataFrame) -> np.ndarray:
        """将不同模型输出统一为与目标窗口一致的二维矩阵。"""
        if isinstance(predictions, pd.DataFrame):
            aligned = predictions.reindex(index=target.index)
            values = aligned.to_numpy(dtype=float)
        else:
            values = np.asarray(predictions, dtype=float)

        if values.ndim == 1:
            values = values.reshape(target.shape[0], -1)

        if values.shape != target.shape:
            fixed = np.zeros(target.shape, dtype=float)
            rows = min(values.shape[0], target.shape[0])
            cols = min(values.shape[1], target.shape[1])
            fixed[:rows, :cols] = values[:rows, :cols]
            if cols < target.shape[1]:
                fill_values = np.nanmean(values[:, :cols], axis=1) if cols > 0 else np.nanmean(target.values, axis=1)
                for row_idx in range(rows):
                    fixed[row_idx, cols:] = fill_values[row_idx]
            values = fixed

        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _calculate_forecast_metrics(
        self,
        actual,
        predicted,
        train_actual=None,
        epsilon: float = 1e-8
    ) -> Dict[str, Optional[float]]:
        """计算预测评估指标：MAE、RMSE、RRMSE、MASE、RelMAE、sMAPE、NRMSE。"""
        y_true = np.asarray(actual, dtype=float)
        y_pred = np.asarray(predicted, dtype=float)
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)

        if not valid_mask.any():
            return {
                "mae": None,
                "rmse": None,
                "rrmse": None,
                "mase": None,
                "relmae": None,
                "smape": None,
                "nrmse": None
            }

        true_values = y_true[valid_mask]
        pred_values = y_pred[valid_mask]
        errors = pred_values - true_values
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(np.square(errors))))

        mean_abs_actual = float(np.mean(np.abs(true_values)))
        actual_range = float(np.max(true_values) - np.min(true_values))

        smape_denominator = np.abs(true_values) + np.abs(pred_values)
        smape_terms = np.divide(
            2.0 * abs_errors,
            smape_denominator,
            out=np.zeros_like(abs_errors, dtype=float),
            where=smape_denominator > epsilon
        )
        smape = float(np.mean(smape_terms) * 100.0)

        mase_denominator = self._calculate_mase_denominator(train_actual, epsilon)
        relmae_denominator = self._calculate_naive_last_value_mae(y_true, train_actual, epsilon)

        return {
            "mae": mae,
            "rmse": rmse,
            "rrmse": self._safe_divide(rmse, mean_abs_actual, epsilon),
            "mase": self._safe_divide(mae, mase_denominator, epsilon),
            "relmae": self._safe_divide(mae, relmae_denominator, epsilon),
            "smape": smape,
            "nrmse": self._safe_divide(rmse, actual_range, epsilon)
        }

    def _calculate_mase_denominator(self, train_actual, epsilon: float = 1e-8) -> Optional[float]:
        """MASE分母：训练窗口内一阶朴素预测的平均绝对误差。"""
        if train_actual is None:
            return None

        train_values = np.asarray(train_actual, dtype=float)
        if train_values.ndim == 1:
            train_values = train_values.reshape(1, -1)
        if train_values.shape[1] < 2:
            return None

        diffs = np.abs(np.diff(train_values, axis=1))
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            return None

        denominator = float(np.mean(diffs))
        return denominator if denominator > epsilon else None

    def _calculate_naive_last_value_mae(
        self,
        actual,
        train_actual,
        epsilon: float = 1e-8
    ) -> Optional[float]:
        """RelMAE分母：以上一训练窗口末值作为预测基准的MAE。"""
        if train_actual is None:
            return None

        y_true = np.asarray(actual, dtype=float)
        train_values = np.asarray(train_actual, dtype=float)
        if train_values.ndim == 1:
            train_values = train_values.reshape(1, -1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(train_values.shape[0], -1)
        if train_values.shape[0] != y_true.shape[0] or train_values.shape[1] == 0:
            return None

        last_values = train_values[:, -1].reshape(-1, 1)
        benchmark = np.repeat(last_values, y_true.shape[1], axis=1)
        valid_mask = np.isfinite(y_true) & np.isfinite(benchmark)
        if not valid_mask.any():
            return None

        denominator = float(np.mean(np.abs(y_true[valid_mask] - benchmark[valid_mask])))
        return denominator if denominator > epsilon else None

    def _safe_divide(self, numerator: float, denominator: Optional[float], epsilon: float = 1e-8) -> Optional[float]:
        """分母不可用或接近0时返回None，避免无穷大指标进入评估结果。"""
        if denominator is None or not np.isfinite(denominator) or abs(denominator) <= epsilon:
            return None
        return float(numerator / denominator)

    def _get_algorithm_display_names(self) -> List[str]:
        """返回去重后的模型选择候选算法名称。"""
        names = list(self.available_algorithms.values()) + [
            "指数平滑", "简单平均", "移动平均", "ADIDA", "IMAPA"
        ]
        return self._deduplicate_algorithms(names)

    def _deduplicate_algorithms(self, algorithms: List[str]) -> List[str]:
        """按大小写无关方式去重，保留首次出现的显示名称。"""
        deduplicated = []
        seen = set()
        for algorithm in algorithms:
            key = str(algorithm).strip().lower()
            if key and key not in seen:
                deduplicated.append(algorithm)
                seen.add(key)
        return deduplicated

    def evaluate_algorithms(self, data: pd.DataFrame, features_with_labels: Dict, recommendations: Dict, knowledge: str) -> Dict:
        """评估推荐算法的预测效果
        
        Args:
            data: 原始数据
            features_with_labels: 带标签的特征数据
            recommendations: 算法推荐结果
            knowledge: 从知识库检索到的算法评估相关知识
            
        Returns:
            Dict: 评估结果
        """
        self.model_interface = OllamaInterface()
        results = {}
        demand_types = list(features_with_labels.keys())
        print(f"评估算法 - 需求类型: {demand_types}")
        
        for demand_type in demand_types:
            print(f"处理需求类型: {demand_type}")
            
            # 获取特征数据和聚类标签
            if demand_type not in features_with_labels:
                print(f"警告: 未找到需求类型 {demand_type} 的特征数据")
                continue
                
            features_for_eval = features_with_labels[demand_type].copy()
            if 'cluster' in features_for_eval.columns and features_for_eval['cluster'].min() == 0:
                features_for_eval['cluster'] = features_for_eval['cluster'] + 1
            label = features_for_eval['cluster']
            evaluation_results = {"clusters": []}
            unique_clusters = np.unique(label)

            for cluster_id in unique_clusters:
                print(f"处理类别 {cluster_id}")
                
                # 查找当前类别的推荐算法 - 适配新的数据结构
                cluster_rec = None
                if demand_type in recommendations and isinstance(recommendations[demand_type], list):
                    for rec in recommendations[demand_type]:
                        if str(rec.get('id', '')) == str(cluster_id):
                            cluster_rec = rec
                            break

                if not cluster_rec:
                    print(f"警告: 未找到类别 {cluster_id} 的算法推荐")
                    continue

                # 选择该类别的数据
                cluster_mask = pd.DataFrame(features_for_eval['cluster'] == cluster_id)
                filtered_data = cluster_mask[cluster_mask['cluster'] == True].index
                cluster_data = data.loc[filtered_data]
                print(f"类别 {cluster_id}: {len(cluster_data)} 个序列")
                
                if len(cluster_data) < 10:  # 数据点太少无法有效评估
                    print(f"跳过类别 {cluster_id}: 数据点太少")
                    continue
                    
                # 准备训练集和测试集（简单的时间序列分割）
                train_size = int(cluster_data.shape[1] * 0.8)
                train_data = cluster_data.iloc[:, :train_size]
                test_data = cluster_data.iloc[:, train_size:]
                
                # 确保输出目录存在
                output_dir = "./forecasting_outputs/"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                self.model_interface.save_to_pickle([train_data, test_data],
                                                   os.path.join(output_dir, 'train-test.pkl'))
                
                # 先进行参数优化，再根据推荐算法进行预测
                algo_name = cluster_rec.get('recommended_algorithm', '').lower()
                algo_name = self._normalize_algorithm_name(algo_name)
                base_params = cluster_rec.get('algorithm_parameters', {})
                optimized_params, optimization_info = self.optimize_algorithm_parameters(
                    train_data, algo_name, base_params
                )
                cluster_rec['algorithm_parameters'] = optimized_params
                cluster_rec['parameter_optimization'] = optimization_info
                predictions, model_info = self._apply_algorithm(train_data, test_data, algo_name, optimized_params)
                model_info["parameter_optimization"] = optimization_info
                print(f"类别 {cluster_id}, 算法 {algo_name}, 预测数量: {len(predictions)}")
                
                if len(predictions) > 0:
                    try:
                        # 计算评估指标
                        aligned_predictions = self._align_predictions(predictions, test_data)
                        metrics = self._calculate_forecast_metrics(
                            test_data.values,
                            aligned_predictions,
                            train_data.values
                        )

                        # 存储评估结果
                        evaluation_results['clusters'].append({
                            "cluster_id": str(cluster_id),
                            "algorithm": algo_name,
                            "metrics": metrics,
                            "model_info": model_info,
                            "optimized_parameters": optimized_params,
                            "data_points": len(cluster_data)
                        })

                        # 更新算法历史
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if algo_name not in self.algorithm_history:
                            self.algorithm_history[algo_name] = []

                        self.algorithm_history[algo_name].append({
                            "dematype": demand_type,
                            "timestamp": timestamp,
                            "cluster_id": str(cluster_id),
                            "metrics": metrics,
                            "data_points": len(cluster_data)
                        })
                        
                    except Exception as eval_error:
                        print(f"评估指标计算失败: {eval_error}")
                        
            results[demand_type] = evaluation_results
        
        return results

    def _apply_algorithm(self, train_data: pd.DataFrame, test_data: pd.DataFrame, algorithm: str, params: Dict) -> \
            Tuple[np.ndarray, Dict]:
        """应用指定的算法进行预测"""
        predictions = []
        model_info = {"status": "failed", "message": "Algorithm not implemented"}

        try:
            if algorithm == 'arima':
                # 使用AutoARIMA自动选择参数
                from statsforecast.models import AutoARIMA
                series_list = []
                for idx in train_data.index:
                    series = train_data.loc[idx]
                    series_df = pd.DataFrame({
                        'unique_id': idx,
                        'ds': series.index,
                        'y': series.values
                    })
                    series_list.append(series_df)

                sf_train = pd.concat(series_list)
                sf = StatsForecast(
                    models=[AutoARIMA(season_length=params.get('season_length', 12))],
                    freq='ME',
                    n_jobs=-1
                )

                sf.fit(sf_train)
                forecast = sf.predict(h=len(test_data.columns))

                # 转换为宽格式
                predictions = forecast.pivot(
                    index="unique_id",
                    columns="ds",
                    values="AutoARIMA"
                ).values

                model_info = {
                    "status": "success",
                    "model_type": "AutoARIMA",
                    "parameters": "自动选择最优参数",
                    "num_series": train_data.shape[0]
                }

            elif algorithm in [
                'exponential smoothing', '指数平滑', 'ets',
                'simple exponential smoothing', 'double exponential smoothing',
                'triple exponential smoothing'
            ]:
                # 使用AutoETS自动选择参数
                from statsforecast.models import AutoETS

                series_list = []
                for idx in train_data.index:
                    series = train_data.loc[idx]
                    series_df = pd.DataFrame({
                        'unique_id': idx,
                        'ds': series.index,
                        'y': series.values
                    })
                    series_list.append(series_df)

                sf_train = pd.concat(series_list)
                sf = StatsForecast(
                    models=[AutoETS(season_length=params.get('season_length', 12))],
                    freq='ME',
                    n_jobs=-1
                )

                sf.fit(sf_train)
                forecast = sf.predict(h=len(test_data.columns))

                # 转换为宽格式
                predictions = forecast.pivot(
                    index="unique_id",
                    columns="ds",
                    values="AutoETS"
                ).values

                model_info = {
                    "status": "success",
                    "model_type": "AutoETS",
                    "parameters": "自动选择最优参数",
                    "num_series": train_data.shape[0]
                }

            elif algorithm == 'prophet':
                changepoint_prior_scale = params.get('changepoint_prior_scale', 0.05)
                seasonality_prior_scale = params.get('seasonality_prior_scale', 10.0)
                forecast_horizon = len(test_data.columns)
                predictions_list = []

                for idx in train_data.index:
                    series = train_data.loc[idx]
                    y_values = pd.to_numeric(series.values, errors="coerce")
                    ds = pd.to_datetime(series.index, errors="coerce")
                    if ds.isna().any():
                        ds = pd.date_range("2000-01-31", periods=len(series), freq="ME")

                    prophet_train = pd.DataFrame({
                        "ds": ds,
                        "y": y_values
                    }).dropna()

                    if len(prophet_train) < 3:
                        fallback_value = np.nanmean(y_values) if np.isfinite(y_values).any() else 0.0
                        fallback = np.repeat(fallback_value, forecast_horizon)
                        predictions_list.append(fallback)
                        continue

                    model = Prophet(
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale
                    )
                    model.fit(prophet_train)
                    future = model.make_future_dataframe(periods=forecast_horizon, freq="ME")
                    forecast = model.predict(future)
                    predictions_list.append(forecast["yhat"].tail(forecast_horizon).to_numpy())

                predictions = np.vstack(predictions_list)

                model_info = {
                    "status": "success",
                    "model_type": "Prophet",
                    "parameters": f"changepoint_prior_scale={changepoint_prior_scale}, seasonality_prior_scale={seasonality_prior_scale}",
                    "num_series": train_data.shape[0]
                }

            elif algorithm == 'xgboost':
                learning_rate = params.get('learning_rate', 0.1)
                n_estimators = params.get('n_estimators', 100)
                max_depth = params.get('max_depth', 6)
                model = XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                predictions, model_info = self._apply_tabular_regressor(
                    train_data, test_data, model, "XGBoost",
                    f"max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}"
                )

            elif algorithm in ['rf', 'random forest', 'random_forest', '随机森林']:
                n_estimators = params.get('n_estimators', 120)
                max_depth = params.get('max_depth', 8)
                min_samples_leaf = params.get('min_samples_leaf', 1)
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                )
                predictions, model_info = self._apply_tabular_regressor(
                    train_data, test_data, model, "RF",
                    f"max_depth={max_depth}, n_estimators={n_estimators}, min_samples_leaf={min_samples_leaf}"
                )

            elif algorithm in ['lightgbm', 'lgbm']:
                try:
                    from lightgbm import LGBMRegressor

                    n_estimators = params.get('n_estimators', 120)
                    learning_rate = params.get('learning_rate', 0.05)
                    num_leaves = params.get('num_leaves', 31)
                    model = LGBMRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        num_leaves=num_leaves,
                        random_state=42,
                        verbosity=-1
                    )
                    predictions, model_info = self._apply_tabular_regressor(
                        train_data, test_data, model, "LightGBM",
                        f"num_leaves={num_leaves}, learning_rate={learning_rate}, n_estimators={n_estimators}"
                    )
                except ImportError:
                    predictions, model_info = self._moving_average_fallback(
                        train_data, test_data, "LightGBM依赖未安装，已回退到移动平均"
                    )

            elif algorithm == 'lstm':
                predictions, model_info = self._apply_lstm_or_fallback(train_data, test_data, params)

            elif algorithm == 'sma' or algorithm == 'simple moving average' or algorithm == '移动平均':
                # 提取参数
                window = params.get('window', 3)

                # 对每条序列计算移动平均预测
                preds = []
                for i in range(len(train_data)):
                    values = train_data.iloc[i].values
                    last_window = values[-window:]
                    pred = np.full(len(test_data.columns), np.mean(last_window))
                    preds.append(pred)

                predictions = np.array(preds)

                model_info = {
                    "status": "success",
                    "model_type": "移动平均",
                    "parameters": f"window={window}",
                    "num_series": train_data.shape[0]
                }

            elif algorithm == 'croston':
                alpha = params.get('alpha', 0.1)
                # 提取参数
                series_list = []
                for idx in train_data.index:
                    series = train_data.loc[idx]
                    time_idx = series.index.values
                    series_df = pd.DataFrame({
                        'unique_id': idx,
                        'ds': time_idx,
                        'y': series.values
                    })
                    series_list.append(series_df)
                # 合并所有序列
                croston_train = pd.concat(series_list)
                croston_train["ds"] = pd.to_datetime(croston_train["ds"])
                sf = StatsForecast(
                    models=[CrostonClassic()],
                    freq='ME',
                    n_jobs=1
                )
                sf_forecast = sf.forecast(
                    h=test_data.shape[1],
                    df=croston_train,
                )
                # 转换为宽格式
                predictions = sf_forecast.pivot(
                    index="unique_id",
                    columns="ds",
                    values="CrostonClassic"
                )
                model_info = {
                    "status": "success",
                    "model_type": "Croston",
                    "parameters": f"alpha={alpha}"
                }
            elif algorithm == 'sba':
                if CrostonSBA is None:
                    predictions, model_info = self._moving_average_fallback(
                        train_data, test_data, "当前statsforecast版本不支持CrostonSBA，已回退到移动平均"
                    )
                else:
                    predictions, model_info = self._apply_statsforecast_model(
                        train_data,
                        test_data,
                        CrostonSBA(),
                        "CrostonSBA",
                        f"alpha={params.get('alpha', 0.1)}"
                    )

            elif algorithm == 'tsb':
                if TSB is None:
                    predictions, model_info = self._moving_average_fallback(
                        train_data, test_data, "当前statsforecast版本不支持TSB，已回退到移动平均"
                    )
                else:
                    alpha_d = params.get('alpha_d', 0.1)
                    alpha_p = params.get('alpha_p', 0.1)
                    try:
                        tsb_model = TSB(alpha_d=alpha_d, alpha_p=alpha_p)
                    except TypeError:
                        tsb_model = TSB(alpha=alpha_d)
                    predictions, model_info = self._apply_statsforecast_model(
                        train_data,
                        test_data,
                        tsb_model,
                        "TSB",
                        f"alpha_d={alpha_d}, alpha_p={alpha_p}"
                    )

            elif algorithm == 'adida':
                # 提取参数
                alpha = params.get('alpha', 0.1)

                # 准备数据格式
                series_list = []
                for idx in train_data.index:
                    series = train_data.loc[idx]
                    series_df = pd.DataFrame({
                        'unique_id': idx,
                        'ds': series.index,
                        'y': series.values
                    })
                    series_list.append(series_df)

                adida_train = pd.concat(series_list)
                adida_train["ds"] = pd.to_datetime(adida_train["ds"])

                # 使用StatsForecast进行预测
                sf = StatsForecast(
                    models=[ADIDA(alpha=alpha)],
                    freq='ME',
                    n_jobs=-1
                )

                sf.fit(adida_train)
                forecast = sf.predict(h=len(test_data.columns))

                # 转换为宽格式
                predictions = forecast.pivot(
                    index="unique_id",
                    columns="ds",
                    values="ADIDA"
                ).values

                model_info = {
                    "status": "success",
                    "model_type": "ADIDA",
                    "parameters": f"alpha={alpha}",
                    "num_series": train_data.shape[0]
                }
            elif algorithm == 'imapa':
                # 提取参数
                alpha = params.get('alpha', 0.1)

                # 准备数据格式
                series_list = []
                for idx in train_data.index:
                    series = train_data.loc[idx]
                    series_df = pd.DataFrame({
                        'unique_id': idx,
                        'ds': series.index,
                        'y': series.values
                    })
                    series_list.append(series_df)

                imapa_train = pd.concat(series_list)
                imapa_train["ds"] = pd.to_datetime(imapa_train["ds"])

                # 使用StatsForecast进行预测
                sf = StatsForecast(
                    models=[IMAPA(alpha=alpha)],
                    freq='ME',
                    n_jobs=-1
                )

                sf.fit(imapa_train)
                forecast = sf.predict(h=len(test_data.columns))

                # 转换为宽格式
                predictions = forecast.pivot(
                    index="unique_id",
                    columns="ds",
                    values="IMAPA"
                ).values

                model_info = {
                    "status": "success",
                    "model_type": "IMAPA",
                    "parameters": f"alpha={alpha}",
                    "num_series": train_data.shape[0]
                }
            else:
                # 默认使用简单平均
                mean_value = train_data.mean(axis=1)
                predictions = np.tile(mean_value.values[:, np.newaxis], (1, len(test_data.columns)))

                model_info = {
                    "status": "success",
                    "model_type": "简单平均",
                    "parameters": "None"
                }

        except Exception as e:
            model_info = {
                "status": "failed",
                "message": str(e),
                "algorithm": algorithm
            }

            # 使用历史平均作为备选预测
            mean_value = train_data.mean(axis=1)
            predictions = np.tile(mean_value.values[:, np.newaxis], (1, len(test_data.columns)))

        return predictions, model_info

    def _apply_tabular_regressor(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, model, model_type: str, parameters: str
    ) -> Tuple[np.ndarray, Dict]:
        """使用滞后特征训练树模型，并递归预测测试窗口。"""
        train_long = self.melt_data(train_data)
        train_long = self.create_features(train_long)
        train_clean = train_long.dropna(subset=[f'lag_{lag}' for lag in range(1, 13)])
        features = ['year', 'month'] + [f'lag_{lag}' for lag in range(1, 13)] + [
            f'rolling_{w}_mean' for w in [3, 6, 12]
        ]

        if train_clean.empty:
            return self._moving_average_fallback(train_data, test_data, f"{model_type}训练样本不足，已回退到移动平均")

        split_idx = max(1, int(len(train_clean) * 0.9))
        X_train = train_clean[features].iloc[:split_idx]
        y_train = train_clean['value'].iloc[:split_idx]
        X_val = train_clean[features].iloc[split_idx:]
        y_val = train_clean['value'].iloc[split_idx:]

        if model_type == "XGBoost" and len(X_val) > 0:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        forecast_steps = test_data.shape[1]
        predictions = pd.DataFrame(index=test_data.index, columns=test_data.columns)

        for series in test_data.index:
            series_history = train_long[train_long['series'] == series].sort_values('time')
            last_row = series_history.iloc[[-1]][features].copy()
            preds = self.recursive_forecast(model, last_row, forecast_steps, features)
            predictions.loc[series, :] = preds

        model_info = {
            "status": "success",
            "model_type": model_type,
            "parameters": parameters,
            "num_series": train_data.shape[0]
        }
        return predictions.values.astype(float), model_info

    def _apply_statsforecast_model(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, model, value_column: str, parameters: str
    ) -> Tuple[np.ndarray, Dict]:
        """统一执行StatsForecast模型并转回宽格式预测矩阵。"""
        series_list = []
        for idx in train_data.index:
            series = train_data.loc[idx]
            series_df = pd.DataFrame({
                'unique_id': idx,
                'ds': series.index,
                'y': series.values
            })
            series_list.append(series_df)

        sf_train = pd.concat(series_list)
        sf_train["ds"] = pd.to_datetime(sf_train["ds"])
        sf = StatsForecast(models=[model], freq='ME', n_jobs=1)
        forecast = sf.forecast(h=test_data.shape[1], df=sf_train)

        predictions_df = forecast.pivot(
            index="unique_id",
            columns="ds",
            values=value_column
        )
        predictions_df = predictions_df.reindex(train_data.index)

        model_info = {
            "status": "success",
            "model_type": value_column,
            "parameters": parameters,
            "num_series": train_data.shape[0]
        }
        return predictions_df.values.astype(float), model_info

    def _apply_lstm_or_fallback(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, params: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """使用轻量LSTM训练；未安装tensorflow或样本不足时回退到移动平均。"""
        try:
            from tensorflow.keras.layers import LSTM, Dense
            from tensorflow.keras.models import Sequential
        except ImportError:
            return self._moving_average_fallback(train_data, test_data, "tensorflow未安装，LSTM已回退到移动平均")

        lookback = params.get('lookback', 12)
        epochs = params.get('epochs', 20)
        units = params.get('units', 32)
        values = train_data.to_numpy(dtype=float)
        X, y = [], []
        for row in values:
            for i in range(lookback, len(row)):
                X.append(row[i - lookback:i])
                y.append(row[i])

        if not X:
            return self._moving_average_fallback(train_data, test_data, "LSTM训练样本不足，已回退到移动平均")

        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)
        model = Sequential([
            LSTM(units, input_shape=(lookback, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

        forecast_steps = test_data.shape[1]
        predictions = []
        for _, row in train_data.iterrows():
            history = row.to_numpy(dtype=float).tolist()
            preds = []
            for _ in range(forecast_steps):
                model_input = np.array(history[-lookback:]).reshape(1, lookback, 1)
                pred = float(model.predict(model_input, verbose=0)[0][0])
                preds.append(pred)
                history.append(pred)
            predictions.append(preds)

        model_info = {
            "status": "success",
            "model_type": "LSTM",
            "parameters": f"lookback={lookback}, units={units}, epochs={epochs}",
            "num_series": train_data.shape[0]
        }
        return np.array(predictions), model_info

    def _moving_average_fallback(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, message: str
    ) -> Tuple[np.ndarray, Dict]:
        """模型不可用时的稳定回退预测。"""
        window = min(3, train_data.shape[1])
        preds = []
        for i in range(len(train_data)):
            values = train_data.iloc[i].values
            pred = np.full(len(test_data.columns), np.mean(values[-window:]))
            preds.append(pred)

        return np.array(preds), {
            "status": "success",
            "model_type": "移动平均",
            "parameters": f"window={window}",
            "fallback_reason": message,
            "num_series": train_data.shape[0]
        }

    def melt_data(self, df):
        """将宽格式数据（列名为时间）转换为长格式（列为 `series`, `time`, `value`）"""
        df_melted = df.reset_index().melt(id_vars='index', var_name='time', value_name='value')
        df_melted = df_melted.rename(columns={'index': 'series'})
        df_melted['time'] = pd.to_datetime(df_melted['time'])  # 确保时间为datetime类型
        return df_melted

    def create_features(self, df, window_sizes=[3, 6, 12]):
        df = df.sort_values(['series', 'time'])

        # 生成 lag_1 到 lag_12（覆盖所有必要滞后步长）
        lags = list(range(1, 13))  # 生成 lag_1 到 lag_12
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby('series')['value'].shift(lag)

        # 其他特征（滚动统计、时间特征等）
        for window in window_sizes:
            df[f'rolling_{window}_mean'] = df.groupby('series')['value'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rolling_{window}_std'] = df.groupby('series')['value'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        return df

    def recursive_forecast(self, model, initial_data, forecast_steps, features):
        predictions = []
        current_data = initial_data.copy()

        for _ in range(forecast_steps):
            # 预测下一步
            pred = model.predict(current_data[features])
            predictions.append(pred[0])

            # 更新滞后特征（从 lag_12 到 lag_1 反向更新）
            # 注意：需确保所有 lag_1 到 lag_12 在特征工程中已生成
            for lag in reversed(range(1, 13)):  # 从 lag_12 到 lag_1
                if lag == 1:
                    current_data[f'lag_{lag}'] = pred[0]
                else:
                    current_data[f'lag_{lag}'] = current_data[f'lag_{lag - 1}'].values[0]  # 直接取前一步的值

            # 更新时间特征
            current_data['year'] = current_data['year'] + (current_data['month'] + 1) // 12
            current_data['month'] = (current_data['month'] % 12) + 1

        return predictions

    def process_feedback(self, feedback: Dict) -> Dict:
        """处理预测算法反馈并生成优化建议
        
        Args:
            feedback: 包含预测效果评估和调整建议的字典
            
        Returns:
            Dict: 包含算法优化建议的字典
        """
        # 处理不同类型的反馈格式
        if isinstance(feedback, dict):
            # 如果是字典格式，检查是否有特定的键
            if 'algorithm_feedback' in feedback:
                # 单一反馈格式
                feedback_content = feedback['algorithm_feedback']
                prompts = [f"预测算法反馈: {feedback_content}"]
            else:
                # 多需求类型格式
                demand_types = list(feedback.keys())
                prompts = []
                for demand_type in demand_types:
                    if isinstance(feedback[demand_type], dict):
                        if not feedback[demand_type].get('forecast_algorithms'):
                            prompts.append(f"{demand_type}预测良好，不需要进行额外调整。")
                        else:
                            prompts.append(f"{demand_type}的预测反馈:\n" + 
                                         json.dumps(feedback[demand_type]['forecast_algorithms'], 
                                                  indent=2, ensure_ascii=False))
                    else:
                        # 如果值是字符串或其他类型
                        prompts.append(f"{demand_type}的预测反馈: {feedback[demand_type]}")
        else:
            # 如果反馈本身是字符串
            prompts = [f"预测算法反馈: {feedback}"]

        # 检索相关知识
        forecasting_knowledge = self.get_knowledge("时间序列预测算法选择和参数优化", top_k=3)
        knowledge_prompt = ""
        if forecasting_knowledge:
            knowledge_prompt = "\n相关知识：\n" + "\n".join([doc["content"] for doc in forecasting_knowledge])

        # 获取可用算法列表
        available_algorithms = list(self.available_algorithms.keys())

        prompt = f"""
请分析以下预测算法反馈，并提供具体的优化建议：

{chr(10).join(prompts)}

{knowledge_prompt}

可用的预测算法包括：{', '.join(available_algorithms)}

请考虑以下方面：
1. 当前算法是否适合数据特征
2. 算法参数是否需要调整
3. 是否需要数据预处理
4. 是否需要集成多个模型

请以JSON格式返回分析结果，格式如下：
{{
    "algorithm_changes": {{
        "{demand_type[0]}": {{
            "类别ID1": {{
                "current_algorithm": "当前使用的算法",
                "new_algorithm": "建议使用的算法",
                "parameters": {{
                    "参数名": "参数值"
                }},
                "preprocessing": [预处理步骤1, 预处理步骤2],
                "reason": "更换算法的原因"
            }},
            ...
            "类别IDn": {{
                "current_algorithm": "当前使用的算法",
                "new_algorithm": "建议使用的算法",
                "parameters": {{
                    "参数名": "参数值"
                }},
                "preprocessing": [预处理步骤1, 预处理步骤2],
                "reason": "更换算法的原因"
            }}
        }},
        "{demand_type[1]}": {{
            "类别ID1": {{
                "current_algorithm": "当前使用的算法",
                "new_algorithm": "建议使用的算法",
                "parameters": {{
                    "参数名": "参数值"
                }},
                "preprocessing": [预处理步骤1, 预处理步骤2],
                "reason": "更换算法的原因"
            }},
            ...
            "类别IDn": {{
                "current_algorithm": "当前使用的算法",
                "new_algorithm": "建议使用的算法",
                "parameters": {{
                    "参数名": "参数值"
                }},
                "preprocessing": [预处理步骤1, 预处理步骤2],
                "reason": "更换算法的原因"
            }}
        }}
    }}
}}

注意：
1. 算法选择必须从上述可用算法列表中选择
2. 上述内容包括有多种需求类型，每种需求类型中包括有多种类别ID，请根据需求类型和类别ID进行优化
3. 预处理步骤必须是可执行的具体操作
4. 如果当前算法表现良好，可以保持不变，但需要说明原因
"""

        # 请求大模型分析反馈
        response = self.model.query(prompt, system_prompt=self.system_prompt)
        result = self.model.parse_json_response(response['content'])
        
        return result

    def apply_optimization(self, optimization_plan: Dict) -> Dict:
        """应用预测算法优化建议
        
        Args:
            optimization_plan: 包含算法优化建议的字典
            
        Returns:
            Dict: 优化后的预测策略
        """
        if not optimization_plan or 'algorithm_changes' not in optimization_plan:
            return {
                "prediction_strategies": {},
                "implementation_plan": {
                    'data_preparation': [
                        '数据清洗和标准化',
                        '特征工程',
                        '数据分割（训练集、验证集、测试集）'
                    ],
                    'execution_sequence': [
                        '应用预处理步骤',
                        '训练预测模型',
                        '模型验证和调优',
                        '生成预测结果'
                    ]
                },
                'is_recommend_adjustments': {
                    'feature_extraction': 'false',
                    'classification': 'false',
                    'forecast_algorithm': 'false'
                },
                "message": "无有效的优化计划"
            }

        # 创建新的预测策略
        prediction_strategy = {
            demand_type: {
                'prediction_strategies': {}
            } for demand_type in optimization_plan['algorithm_changes'].keys()
        }

        # 应用算法更改
        for demand_type, classes in optimization_plan['algorithm_changes'].items():
            if isinstance(classes, dict):
                for class_id, changes in classes.items():
                    try:
                        # 验证算法是否可用
                        new_algorithm = changes.get('new_algorithm', '').lower()
                        if new_algorithm not in self.available_algorithms:
                            print(f"警告：未知的预测算法 {new_algorithm}，将保持使用当前算法")
                            new_algorithm = changes.get('current_algorithm', 'arima').lower()
                            if new_algorithm not in self.available_algorithms:
                                new_algorithm = 'arima'  # 使用默认算法

                        # 获取算法参数
                        params = changes.get('parameters', {})
                        
                        # 获取预处理步骤
                        preprocessing = changes.get('preprocessing', [])
                        
                        # 更新预测策略
                        prediction_strategy[demand_type]['prediction_strategies'][class_id] = {
                            'final_algorithm': self.available_algorithms[new_algorithm],
                            'parameters': params,
                            'preprocessing': preprocessing,
                            'reason': changes.get('reason', '算法更新')
                        }

                        print(f"{demand_type} - 类别 {class_id} 预测算法优化完成")
                        print(f"新算法: {self.available_algorithms[new_algorithm]}")
                        print(f"参数: {params}")
                        if preprocessing:
                            print(f"预处理步骤: {preprocessing}")
                    except Exception as e:
                        print(f"优化 {demand_type} - 类别 {class_id} 时出错: {e}")
                        # 添加默认策略
                        prediction_strategy[demand_type]['prediction_strategies'][class_id] = {
                            'final_algorithm': 'ARIMA',
                            'parameters': {},
                            'preprocessing': [],
                            'reason': '默认算法（优化失败）'
                        }

        # 添加实施计划
        prediction_strategy['implementation_plan'] = {
            'data_preparation': [
                '数据清洗和标准化',
                '特征工程',
                '数据分割（训练集、验证集、测试集）'
            ],
            'execution_sequence': [
                '应用预处理步骤',
                '训练预测模型',
                '模型验证和调优',
                '生成预测结果'
            ]
        }

        # 添加是否建议进行调整的标志
        prediction_strategy['is_recommend_adjustments'] = {
            'feature_extraction': 'false',  # 默认不建议调整特征提取
            'classification': 'false',      # 默认不建议调整分类
            'forecast_algorithm': 'true'    # 由于进行了算法优化，标记为true
        }

        return prediction_strategy
