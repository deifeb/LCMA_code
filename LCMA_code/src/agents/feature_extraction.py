#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征提取智能体
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.feature_selection import VarianceThreshold
from src.model_interface import OllamaInterface
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os



class FeatureExtractionAgent:
    """优化后的间歇性需求特征提取智能体"""

    def __init__(self, model_interface: OllamaInterface):
        self.model = model_interface
        self.system_prompt = """你是一名专业的间歇性时间序列需求特征提取专家。
你的任务是分析不同类型的备件需求数据（间歇性、平滑性、缓慢移动和平稳性），提取能够表征其特征的关键指标。
需要重点考虑以下方面：
- 需求间隔时间统计（平均间隔、间隔变异系数）
- 非零需求的统计特征（规模、变异系数）
- 时间序列特征（趋势、季节性、自相关等）
- 近似熵和序列复杂性
- 零值比例和稀疏性指标
- Croston分解相关特征
- 间歇性特征（零值比例、需求聚集度等）
请为每种需求类型选择最具代表性的特征集合，并以JSON格式提供分析结果。"""
        self.feature_history = {}
        self.knowledge_manager = None
        self.original_features = None  # 存储原始特征
        self.optimized_features = None  # 存储优化后的特征
        
    def set_knowledge_manager(self, knowledge_manager):
        """设置知识库管理器
        
        Args:
            knowledge_manager: 智能体知识库管理器实例
        """
        self.knowledge_manager = knowledge_manager
        
    def get_knowledge(self, query: str, top_k: int = 3):
        """从特征提取智能体知识库中检索知识
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 相关文档列表
        """
        if self.knowledge_manager:
            return self.knowledge_manager.search_knowledge("feature_extraction", query, top_k)
        return []

    def _calculate_sbc_classification(self, series: pd.Series) -> str:
        """
        计算SBC需求分类
        使用ADI(平均需求间隔)和CV²(非零需求变异系数平方)进行分类
        """
        # 获取非零需求点
        non_zero = series[series != 0]
        if len(non_zero) < 2:
            return "intermittent"

        # 计算ADI (Average Demand Interval)
        non_zero_idx = np.where(series != 0)[0]
        intervals = np.diff(non_zero_idx)
        adi = np.mean(intervals) if len(intervals) > 0 else float('inf')

        # 计算CV² (Square of Coefficient of Variation)
        cv_squared = (non_zero.std() / non_zero.mean()) ** 2 if len(non_zero) > 0 else float('inf')

        # SBC分类阈值
        ADI_cutoff = 1.32  # ADI的标准阈值
        CV2_cutoff = 0.49  # CV²的标准阈值

        if adi > ADI_cutoff:
            if cv_squared > CV2_cutoff:
                return "块状需求Lumpy"  # 不规则需求
            else:
                return "间歇性需求Intermittent"  # 间歇性需求
        else:
            if cv_squared > CV2_cutoff:
                return "不稳定需求Erratic"  # 突发性需求
            else:
                return "平稳需求Smooth"  # 平稳需求

    def _extract_tsfresh_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """使用tsfresh提取时间序列特征"""
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import impute
        # 准备tsfresh所需的数据格式
        series_list = []
        for idx in data.index:
            series = data.loc[idx]
            # 创建时间索引
            # time_idx = np.arange(len(series))
            time_idx = series.index.values
            series_df = pd.DataFrame({
                'id': idx,
                'time': time_idx,
                'value': series.values
            })
            series_list.append(series_df)
        # 合并所有序列
        all_series = pd.concat(series_list)

        # 提取特征
        features_df = extract_features(all_series,
                                       column_id='id',
                                       column_sort='time',
                                       column_value='value', impute_function=impute, show_warnings=False)
        features_select_df = self._select_features(features_df)
        return features_select_df

    def _generate_intermittent_stats(self, features_df: Dict) -> dict:
        """生成面向间歇性特征的统计摘要"""
        feature_deal = {}
        for feature_type in list(features_df.keys()):
            print(feature_type)
            stats = {}
            feature = features_df[feature_type]
            if '类型' in feature.columns:
                feature = feature.drop(columns=['类型'])
            for col in feature.columns:
                # print(col)
                s = feature[col]
                stats[col] = {
                    "mean": round(s.mean(), 4),
                    "std": round(s.std(), 4),
                    "zero_ratio": round((s == 0).mean(), 4),
                    "outlier_ratio": round((s > s.quantile(0.95)).mean(), 4)
                }
                # 添加间歇性特征特有指标
                if "interval" in col:
                    stats[col].update({
                        "min_interval": s.min(),
                        "max_interval": s.max(),
                        "cv_interval": s.std() / s.mean()
                    })
            df = pd.DataFrame.from_dict(stats, orient='columns')  # 合并内层键值对，重复键会被覆盖
            feature_deal[feature_type] = df

        return feature_deal

    def apply_optimization(self, optimization_plan: Dict) -> Dict:
        """应用特征优化建议并评估优化效果
        
        Args:
            optimization_plan: 包含特征优化建议的字典
            
        Returns:
            Dict: 优化结果，包含优化后的特征和评估结果
        """
        if not optimization_plan.get("needs_optimization", False):
            return {
                "optimized_features": self.original_features,
                "evaluation": {
                    "message": "无需优化",
                    "comparison": None
                }
            }

        # 1. 应用优化
        optimized_features = {}
        adjustments = optimization_plan.get('feature_adjustments', {})

        for demand_type, adjustment in adjustments.items():
            # 获取原始特征数据
            features_df = self.original_features[demand_type].copy()
            
            # 1.1 移除不需要的特征
            for remove_feature in adjustment.get('remove_features', []):
                feature_name = remove_feature['name']
                if feature_name in features_df.columns:
                    features_df = features_df.drop(columns=[feature_name])
                    print(f"移除特征: {feature_name}")

            # 1.2 修改现有特征
            for modify_feature in adjustment.get('modify_features', []):
                feature_name = modify_feature['name']
                if feature_name in features_df.columns:
                    # 应用新的计算方法
                    new_values = self._calculate_feature(
                        features_df,
                        modify_feature['new_calculation'],
                        modify_feature.get('parameters', {})
                    )
                    features_df[feature_name] = new_values
                    print(f"修改特征: {feature_name}")

            # 1.3 添加新特征
            for new_feature in adjustment.get('new_features', []):
                feature_name = new_feature['name']
                # 计算新特征
                new_values = self._calculate_feature(
                    features_df,
                    new_feature['calculation'],
                    new_feature.get('parameters', {})
                )
                features_df[feature_name] = new_values
                print(f"添加新特征: {feature_name}")

            # 1.4 应用预处理步骤
            preprocessing_steps = optimization_plan.get('preprocessing', {}).get('steps', [])
            preprocessing_params = optimization_plan.get('preprocessing', {}).get('parameters', {})
            features_df = self._apply_preprocessing(features_df, preprocessing_steps, preprocessing_params)

            optimized_features[demand_type] = features_df

        # 2. 评估优化效果
        # evaluation = self.evaluate_optimization(self.original_features, optimized_features)
        
        # 3. 存储优化后的特征
        self.optimized_features = optimized_features

        # 4. 记录优化历史
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.feature_history[timestamp] = {
            "optimization_plan": optimization_plan,
            "feature_changes": {
                demand_type: {
                    "original_columns": list(self.original_features[demand_type].columns),
                    "optimized_columns": list(optimized_features[demand_type].columns)
                } for demand_type in optimized_features.keys()
            }
        }

        # 5. 返回优化结果
        return {
            "optimized_features": optimized_features,
            "timestamp": timestamp
        }

    def _analyze_features(self, features: Dict) -> Dict:
        """分析特征统计信息"""
        stats = {}
        for demand_type, df in features.items():
            feature_stats = {}
            for col in df.columns:
                if col != '类型':  # 排除类型列
                    feature_stats[col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "missing_ratio": float((df[col].isna().sum() / len(df)) * 100),
                        "zero_ratio": float((df[col] == 0).sum() / len(df) * 100)
                    }
            stats[demand_type] = feature_stats
        return stats

    def _calculate_feature(self, df: pd.DataFrame, calculation_method: str, parameters: Dict) -> pd.Series:
        """计算特征值
        
        这里需要实现各种特征计算方法，例如：
        - 移动平均
        - 标准差
        - 趋势指标
        - 季节性指标
        - 间歇性指标
        等等
        """
        # TODO: 实现具体的特征计算方法
        # 这里返回一个示例值
        return pd.Series(np.zeros(len(df)))

    def _apply_preprocessing(self, df: pd.DataFrame, steps: List[str], parameters: Dict) -> pd.DataFrame:
        """应用预处理步骤"""
        result = df.copy()
        
        for step in steps:
            if step == "standardization":
                scaler = StandardScaler()
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
            elif step == "missing_value_imputation":
                strategy = parameters.get("imputation_strategy", "mean")
                if strategy == "mean":
                    result = result.fillna(result.mean())
                elif strategy == "median":
                    result = result.fillna(result.median())
                elif strategy == "zero":
                    result = result.fillna(0)
            elif step == "outlier_handling":
                threshold = parameters.get("outlier_threshold", 3)
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    mean = result[col].mean()
                    std = result[col].std()
                    result[col] = result[col].clip(mean - threshold * std, mean + threshold * std)
            
        return result

    def evaluate_optimization(self, original_features: Dict, optimized_features: Dict) -> Dict:
        """评估特征优化效果
        
        Args:
            original_features: 原始特征数据
            optimized_features: 优化后的特征数据
            
        Returns:
            Dict: 评估结果
        """
        evaluation = {}
        
        for demand_type in original_features.keys():
            orig_df = original_features[demand_type]
            opt_df = optimized_features[demand_type]
            
            # 1. 基础统计评估
            evaluation[demand_type] = {
                "feature_count": {
                    "original": len(orig_df.columns),
                    "optimized": len(opt_df.columns),
                    "difference": len(opt_df.columns) - len(orig_df.columns)
                },
                "missing_values": {
                    "original": float(orig_df.isna().sum().sum() / (orig_df.shape[0] * orig_df.shape[1]) * 100),
                    "optimized": float(opt_df.isna().sum().sum() / (opt_df.shape[0] * opt_df.shape[1]) * 100),
                    "improvement": float((orig_df.isna().sum().sum() / (orig_df.shape[0] * orig_df.shape[1]) - 
                                       opt_df.isna().sum().sum() / (opt_df.shape[0] * opt_df.shape[1])) * 100)
                },
                "zero_values": {
                    "original": float((orig_df == 0).sum().sum() / (orig_df.shape[0] * orig_df.shape[1]) * 100),
                    "optimized": float((opt_df == 0).sum().sum() / (opt_df.shape[0] * opt_df.shape[1]) * 100),
                    "improvement": float(((orig_df == 0).sum().sum() / (orig_df.shape[0] * orig_df.shape[1]) - 
                                       (opt_df == 0).sum().sum() / (opt_df.shape[0] * opt_df.shape[1])) * 100)
                }
            }
            
            # 2. 特征相关性评估
            orig_corr = orig_df.corr().abs().mean().mean()
            opt_corr = opt_df.corr().abs().mean().mean()
            evaluation[demand_type]["feature_correlation"] = {
                "original": float(orig_corr),
                "optimized": float(opt_corr),
                "improvement": float(opt_corr - orig_corr)
            }
            
            # 3. 特征重要性评估（使用随机森林特征重要性）
            try:
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # 为原始特征计算重要性
                rf.fit(orig_df, np.zeros(len(orig_df)))  # 使用虚拟目标变量
                orig_importance = pd.Series(rf.feature_importances_, index=orig_df.columns)
                
                # 为优化后特征计算重要性
                rf.fit(opt_df, np.zeros(len(opt_df)))  # 使用虚拟目标变量
                opt_importance = pd.Series(rf.feature_importances_, index=opt_df.columns)
                
                evaluation[demand_type]["feature_importance"] = {
                    "original": {
                        "top_features": orig_importance.nlargest(5).to_dict(),
                        "mean_importance": float(orig_importance.mean())
                    },
                    "optimized": {
                        "top_features": opt_importance.nlargest(5).to_dict(),
                        "mean_importance": float(opt_importance.mean())
                    }
                }
            except Exception as e:
                print(f"特征重要性评估失败: {str(e)}")
                
            # 4. 计算优化效果得分
            improvement_score = (
                evaluation[demand_type]["missing_values"]["improvement"] +
                evaluation[demand_type]["zero_values"]["improvement"] +
                evaluation[demand_type]["feature_correlation"]["improvement"] * 100
            ) / 3
            
            evaluation[demand_type]["overall_score"] = {
                "improvement_score": float(improvement_score),
                "assessment": "显著改善" if improvement_score > 10 else "轻微改善" if improvement_score > 0 else "无明显改善"
            }
        
        return evaluation

    def get_optimization_history(self) -> Dict:
        """获取特征优化历史"""
        return self.feature_history

    def _select_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # 创建方差过滤器
            selector = VarianceThreshold(threshold=0.1)
            # 应用过滤器
            filtered_features = selector.fit_transform(features_df)
        except Exception as e:
            print(f"方差过滤时出现错误: {e}")
            # 可以根据实际情况进行处理，例如使用默认值或跳过该步骤
            filtered_features = features_df

            # 获取保留的特征索引
        retained_indices = selector.get_support(indices=True)
        # 从原始特征列名中获取保留的特征列名
        filtered_feature_names = features_df.columns[retained_indices]
        # 创建一个新的 DataFrame，包含保留的特征及其列名
        features_df = pd.DataFrame(filtered_features, columns=filtered_feature_names)
        # 计算相关系数矩阵
        corr_matrix = features_df.corr().abs()
        # 选择与其他特征相关性高于阈值的特征
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
        # 删除高相关性的特征
        filtered = features_df.drop(columns=to_drop)
        return filtered

    def analyze_data(self, data: pd.DataFrame, knowledge: str) -> dict:
        """分析数据并提取特征
        
        Args:
            data: 输入的时间序列数据
            prompt: 从知识库检索到的特征提取相关知识
            
        Returns:
            dict: 提取的特征数据
        """
        # 1. 使用tsfresh提取基础特征
        # tsfresh_features = self._extract_tsfresh_features(data)
        # tsfresh_features.to_excel('tsfresh_features.xlsx', index=False)
        print("tsfresh特征提取完成")
        # 修改路径，从data文件夹读取tsfresh特征文件
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tsfresh_path = os.path.join(current_dir, "data", "tsfresh_features.xlsx")
        tsfresh_features = pd.read_excel(tsfresh_path)
        tsfresh_features.columns = tsfresh_features.columns.str.replace(r'["()]', '', regex=True)
        tsfresh_features.index = data.index
        # 2. 创建基于SBC分类的数据描述
        statistic_feature, feature, SBC_class = self._create_intermittent_description(data)

        # 3. 构建更具描述性的提示词
        demand_patterns_desc = ""
        for i in range(len(statistic_feature)):
            pattern_type = statistic_feature[i][0]
            stats = statistic_feature[i][1]

            desc = f"""
需求类型: {pattern_type}
特征统计:
- 非零需求变异系数的平方: 均值={stats.loc['mean', '需求变异系数']:.2f}, 方差={stats.loc['var', '需求变异系数']:.2f}，反映序列的波动性
- 零值序列特征:
  * {pattern_type}中每条序列的连续出现零值的最长序列的长度: 均值={stats.loc['mean', '零值序列最大长度']:.2f}, 方差={stats.loc['var', '零值序列最大长度']:.2f}
  * {pattern_type}中每条序列的零值序列的平均持续时间: 均值={stats.loc['mean', '零值序列的平均长度']:.2f}, 方差={stats.loc['var', '零值序列的平均长度']:.2f}
  * {pattern_type}中每条序列的零值序列长度的波动情况: 均值={stats.loc['mean', '零值序列长度的标准差']:.2f}, 方差={stats.loc['var', '零值序列长度的标准差']:.2f}
  * {pattern_type}中每条序列的零值序列的总数: 均值={stats.loc['mean', '零值序列的数量']:.2f}, 方差={stats.loc['var', '零值序列的数量']:.2f}
- 需求间隔特征:
  * {pattern_type}中每条序列的平均需求间隔: 均值={stats.loc['mean', 'ADI']:.2f}, 方差={stats.loc['var', 'ADI']:.2f}
  * {pattern_type}中每条序列的需求间隔的非零需求变异系数的平方: 均值={stats.loc['mean', '需求间隔CV²']:.2f}, 方差={stats.loc['var', '需求间隔CV²']:.2f}
"""
            demand_patterns_desc += desc

        # 创建特征索引映射
        feature_mapping = {i: col for i, col in enumerate(tsfresh_features.columns)}
        # 4. 请求大模型分析并选择特征
        # 构建提示词，包含从知识库检索到的特征提取相关知识
        prompt = f"""
        {knowledge}
        该数据集的特征模式包含有{[demand_mode[0] for demand_mode in statistic_feature]}{len(statistic_feature)}种需求模式，请作为需求预测专家分析以下每一种备件需求数据的特征模式：

            {demand_patterns_desc}

            请对每一种需求模式下的统计特征进行分析，具体分为以下三点：
        该数据集的特征模式包含有{[demand_mode[0] for demand_mode in statistic_feature]}{len(statistic_feature)}种需求模式，请作为需求预测专家分析以下每一种备件需求数据的特征模式：

            {demand_patterns_desc}

            请对每一种需求模式下的统计特征进行分析，具体分为以下三点：
            1. 需求模式特征画像：
               - 从统计指标揭示的行为特征
               - 各类模式的显著统计差异
               - 统计特征之间的相互关系
            2. 关键统计指标筛选，通过需求模式特征画像，筛选典型特征：
                - 最具区分度的统计特征
                - 预测效果相关性最强的指标
                - 建议重点关注的统计量
            3. 特征选择，依据筛选出的典型特征，从集合中选择代表每种需求模式下的典型特征集合：
                - 从时间序列特征集合中选择最具代表性的特征集合
                - 考虑特征的预测效果和相关性
                - 确保选择的特征能够准确反映需求模式

            下面是可用的时间序列特征集合及其对应的索引编号，这些特征都是使用tsfresh库从备品备件的需求序列中提出来的时间序列特征，请仅从这些特征中进行选择：
            {json.dumps({i: col for i, col in enumerate(tsfresh_features.columns)}, indent=2)}
            【重要提示】：
            1. 请确保每个特征选择时，索引编号和特征名称必须完全匹配上面提供的特征列表
            2. 特征名称必须与索引对应的名称完全一致，不要修改或简化特征名称
            3. 索引编号必须在0到{len(tsfresh_features.columns) - 1}之间
            4. 每种需求类型建议选择10-20个特征，这些特征将用于后续的备件需求模式聚类
            5. 不能包含重复的特找那个名称或索引编号
            请为{[demand_mode[0] for demand_mode in statistic_feature]}需求类型从上述特征列表中选择最具代表性的特征集合(每种类型分别选择10-20个反映需求模式特点的特征)，并说明选择理由。返回形式严格按照以下json格式：
            {{
                "特征选择": {{
                    "块状需求Lumpy": [
                        {{"index": 0, "name": "特征列表中索引0对应的完整特征名称"}},
                        {{"index": 5, "name": "特征列表中索引5对应的完整特征名称"}},
                        ...
                    ],
                    "间歇性需求Intermittent": [
                       {{"index": 2, "name": "特征列表中索引2对应的完整特征名称"}},
                       {{"index": 7, "name": "特征列表中索引7对应的完整特征名称"}},
                        ...
                    ]
                }}
            }}
            请注意：特征名称必须与特征列表中的完全一致，不要简化或修改特征名称，不包含重复的特征名称或索引编号。
            对输出的JSON文本中的符号问题进行检查和修复。确保属性名使用双引号括起来，所有括号、方括号和逗号都正确闭合和匹配
            """
        response = self.model.query(prompt, system_prompt=self.system_prompt)
        features_plan = self.model.parse_json_response(response['content'])

        # 验证和转换特征选择
        if "特征选择" in features_plan:
            for demand_type in features_plan["特征选择"]:
                selected_features = features_plan["特征选择"][demand_type]
                # 验证并获取实际特征
                validated_features = self.validate_feature_indices(
                    selected_features,
                    feature_mapping
                )
                features_plan["特征选择"][demand_type] = validated_features
            SBC_lu = SBC_class['块状需求Lumpy']
            SBC_it = SBC_class['间歇性需求Intermittent']
            # 间歇性
            it_ = feature[1]
            lu_ = feature[0]
            lu_.index = SBC_lu
            it_.index = SBC_it

            it_ts = tsfresh_features.loc[SBC_it]
            lu_ts = tsfresh_features.loc[SBC_lu]
            it_select = features_plan['特征选择']['间歇性需求Intermittent']
            lu_select = features_plan['特征选择']['块状需求Lumpy']
            it_tsfresh = it_ts[it_select]
            lu_tsfresh = lu_ts[lu_select]
            it_feature = pd.concat([it_tsfresh, it_], axis=1)
            lu_feature = pd.concat([lu_tsfresh, lu_], axis=1)
            feature_plan = {
                "间歇性需求Intermittent": it_feature,
                "块状需求Lumpy": lu_feature
            }
            return feature_plan
        else:
            return {}

    def _create_intermittent_description(self, data: pd.DataFrame) -> list:
        """创建基于SBC分类的数据描述"""
        # 对每个序列进行SBC分类
        classifications = {}
        for idx in data.index:
            series = data.loc[idx]
            sbc_class = self._calculate_sbc_classification(series)
            if sbc_class not in classifications:
                classifications[sbc_class] = []
            classifications[sbc_class].append(idx)

        # 为每种类型计算特征统计
        desc_parts = []
        for sbc_class, indices in classifications.items():
            class_data = data.loc[indices]

            # 计算该类别的统计特征
            stats = {
                "类型": [sbc_class] * len(indices),
                # "序列长度": [len(series) for _, series in class_data.iterrows()],
                "需求变异系数": [self._calc_demand_cv_squared(series) for _, series in class_data.iterrows()],
            }
            statses = pd.DataFrame(stats)
            tf = self._get_typical_features(class_data, sbc_class)
            feature = pd.concat([statses, tf], axis=1)
            desc_parts.append(feature)
        statistic_feature = []
        for i in range(len(desc_parts)):
            desc_part = desc_parts[i].select_dtypes(include='number')
            mean_var = desc_part.agg(['mean', 'var'])
            statistic_feature.append([desc_parts[i]['类型'][0], mean_var])
        # 转换为字符串描述
        return statistic_feature, desc_parts, classifications

    def _calc_demand_cv_squared(self, series: pd.Series) -> float:
        """计算需求量的变异系数平方（CV²）"""
        non_zero = series[series != 0]
        if len(non_zero) < 2:
            return float('inf')
        return float((non_zero.std() / non_zero.mean()) ** 2)

    def _get_typical_features(self, data: pd.DataFrame, sbcclass: str) -> pd.DataFrame:
        """获取每种需求类型的典型特征"""
        max_seqs = []
        mean_seqs = []
        std_seqs = []
        seq_counts = []
        mas = []
        sess = []
        dss = []
        cas = []
        cicss = []
        if sbcclass == "块状需求Lumpy" or sbcclass == "间歇性需求Intermittent":
            for idx in data.index:
                max_seq, mean_seq, std_seq, seq_count = self._calc_zero_sequence_distribution(data.loc[idx])
                ca = self._calc_adi(data.loc[idx])
                cics = self._calc_interval_cv_squared(data.loc[idx])
                max_seqs.append(max_seq)
                mean_seqs.append(mean_seq)
                std_seqs.append(std_seq)
                seq_counts.append(seq_count)
                cas.append(ca)
                cicss.append(cics)
            typical_features = {
                "零值序列最大长度": max_seqs,
                "零值序列的平均长度": mean_seqs,
                "零值序列长度的标准差": std_seqs,
                "零值序列的数量": seq_counts,
                "ADI": cas,
                "需求间隔CV²": cicss,
            }
            return pd.DataFrame(typical_features)
        else:
            for idx in data.index:
                ma = self._calc_ma_variance(data.loc[idx])
                ses = self._calc_ses_residuals(data.loc[idx])
                ds = self._calc_demand_smoothness(data.loc[idx])
                mas.append(ma)
                sess.append(ses)
                dss.append(ds)
            typical_features = {
                "移动平均方差": mas,
                "指数平滑残差": sess,
                "需求平滑度": dss
            }
            return pd.DataFrame(typical_features)

    def _calc_zero_sequence_distribution(self, series: pd.Series) -> np.array:
        """计算零值序列分布"""
        zero_sequences = []
        current_seq = 0

        for val in series:
            if val == 0:
                current_seq += 1
            elif current_seq > 0:
                zero_sequences.append(current_seq)
                current_seq = 0

        if current_seq > 0:
            zero_sequences.append(current_seq)

        max_seq = max(zero_sequences) if zero_sequences else 0
        mean_seq = np.mean(zero_sequences) if zero_sequences else 0
        std_seq = np.std(zero_sequences) if zero_sequences else 0
        seq_count = len(zero_sequences)

        return max_seq, mean_seq, std_seq, seq_count

    def _calc_adi(self, series: pd.Series) -> float:
        """计算平均需求间隔（ADI）"""
        non_zero_idx = np.where(series != 0)[0]
        if len(non_zero_idx) < 2:
            return float('inf')
        intervals = np.diff(non_zero_idx)
        return float(np.mean(intervals))

    def _calc_interval_cv_squared(self, series: pd.Series) -> float:
        """计算需求间隔的变异系数平方（CV²）"""
        non_zero_idx = np.where(series != 0)[0]
        if len(non_zero_idx) < 2:
            return float('inf')
        intervals = np.diff(non_zero_idx)
        return float((intervals.std() / intervals.mean()) ** 2)

    def _calc_ma_variance(self, series: pd.Series) -> float:
        """计算移动平均方差"""
        try:
            ma = series.rolling(window=3).mean()
            return float(ma.var())
        except:
            return float('inf')

    def _calc_ses_residuals(self, series: pd.Series) -> dict:
        """计算指数平滑残差"""
        try:
            model = ExponentialSmoothing(series).fit()
            residuals = model.resid
            return {
                "mean": float(residuals.mean()),
                "variance": float(residuals.var())
            }
        except:
            return {"error": "指数平滑计算失败"}

    def _calc_demand_smoothness(self, series: pd.Series) -> float:
        """计算需求平滑度"""
        try:
            diff = np.diff(series)
            return float(1 / (1 + np.std(diff)))
        except:
            return 0.0

    def validate_feature_indices(self, selected_features: list, feature_mapping: dict) -> list:
        """验证特征索引并返回实际特征名称"""
        validated_features = []
        max_index = max(feature_mapping.keys())

        for feature in selected_features:
            try:
                index = feature.get('index')
                name = feature.get('name')

                # 验证索引是否有效
                if index is not None and 0 <= index <= max_index:
                    actual_name = feature_mapping[index]
                    # 验证名称是否匹配
                    if name != actual_name:
                        print(f"警告: 特征名称不匹配 - 索引 {index} 预期 '{actual_name}' 但获得 '{name}'")
                    validated_features.append(actual_name)
                else:
                    print(f"警告: 无效的特征索引 {index}")
            except Exception as e:
                validated_features.append(feature_mapping[feature])
                # print(f"特征验证错误: {e}")

        return validated_features

    def process_feedback(self, feedback: Dict, exist_feature: Dict) -> Dict:
        """处理特征提取反馈并生成优化建议
        
        Args:
            feedback: 包含特征评估和调整建议的字典
            exist_feature: 现有特征数据
            
        Returns:
            Dict: 包含特征优化建议的字典
        """
        # 存储原始特征
        self.original_features = exist_feature.copy()
        
        # 获取需求类型
        demand_types = list(feedback.keys())
        
        # 检查是否需要优化
        needs_optimization = False
        for demand_type in demand_types:
            feature_adjustments = feedback[demand_type].get('feature_adjustments', [])
            if feature_adjustments and len(feature_adjustments) > 0:
                if feature_adjustments[0].get('adjustment') not in [False, [], None]:
                    needs_optimization = True
                    break
            
        # 如果不需要优化，直接返回
        if not needs_optimization:
            return {
                "needs_optimization": False,
                "message": "当前特征表现良好，无需调整"
            }

        # 构建提示词
        prompts = []
        for demand_type in demand_types:
            if not feedback[demand_type].get('feature_adjustments'):
                prompts.append(f"{demand_type}特征良好，不需要进行额外调整。")
            else:
                prompts.append(f"{demand_type}的特征反馈:\n" + 
                             json.dumps(feedback[demand_type], indent=2, ensure_ascii=False))

        # 检索相关知识
        feature_knowledge = self.get_knowledge("时间序列特征提取和优化", top_k=3)
        knowledge_prompt = ""
        if feature_knowledge:
            knowledge_prompt = "\n相关知识：\n" + "\n".join([doc["content"] for doc in feature_knowledge])

        prompt = f"""
对于备件需求特征主要分为{len(demand_types)}类，请根据备件需求预测相关专业知识处理以下反馈：

{prompts[0]}

{prompts[1] if len(prompts) > 1 else ""}

{knowledge_prompt}

分析时请着重考虑：
1. 零膨胀数据特性对特征工程的影响
2. Croston方法与传统时序模型的差异需求
3. 改进间隔时间计算的方法
4. 处理多重零值序列的交互特征
5. 捕捉需求恢复期的模式特征

请以JSON格式返回分析结果，格式如下：
{{
    "feature_adjustments": {{
        "{demand_types[0]}": {{
            "new_features": [
                {{
                    "name": "新特征名称",
                    "calculation": "特征计算方法",
                    "parameters": {{
                        "参数名": "参数值"
                    }},
                    "reason": "添加原因"
                }}
            ],
            "remove_features": [
                {{
                    "name": "要移除的特征名称",
                    "reason": "移除原因"
                }}
            ],
            "modify_features": [
                {{
                    "name": "要修改的特征名称",
                    "new_calculation": "新的计算方法",
                    "parameters": {{
                        "参数名": "参数值"
                    }},
                    "reason": "修改原因"
                }}
            ]
        }}{f',"{ demand_types[1]}": {{"new_features": [{{}}],"remove_features": [{{}}],"modify_features": [{{}}]}}' if len(demand_types) > 1 else ""}
    }},
    "preprocessing": {{
        "steps": ["预处理步骤1", "预处理步骤2"],
        "parameters": {{
            "步骤1参数名": "参数值"
        }}
    }}
}}

注意：
1. 新特征必须与备件需求预测相关
2. 计算方法必须具体且可实现
3. 参数必须是具体的数值
4. 所有修改都需要提供充分的理由
5. 要移除的特征名称必须是已有的特征，而不是凭空捏造的
"""

        try:
            # 请求大模型分析反馈
            response = self.model.query(prompt, system_prompt=self.system_prompt)
            result = self.model.parse_json_response(response['content'])
        except Exception as e:
            print(f"特征反馈处理错误: {e}")
            # 返回一个默认的响应
            result = {
                "feature_adjustments": {
                    demand_types[0]: {
                        "new_features": [],
                        "remove_features": [],
                        "modify_features": []
                    }
                },
                "preprocessing": {
                    "steps": [],
                    "parameters": {}
                }
            }
        
        # 添加优化标志
        result["needs_optimization"] = True

        return result
    

    def explain_features(self, features_df: Dict, knowledge: str) -> Dict:
        """优化后的间歇性特征解释方法
        
        Args:
            features_df: 特征数据
            knowledge: 从知识库检索到的特征解释相关知识
            
        Returns:
            Dict: 特征解释结果
        """
        # 生成包含间歇性特征的统计描述
        self.feature_stats = self._generate_intermittent_stats(features_df)
        demand_types = list(self.feature_stats.keys())
        
        # 获取所有特征名称
        all_features = set()
        for demand_type in demand_types:
            if demand_type in features_df:
                all_features.update(features_df[demand_type].columns.tolist())
        all_features = sorted(list(all_features))
        
        prompt = f"""作为备件需求预测专家，请分析以下间歇性需求特征数据。

特征统计摘要：
需求类型1: {demand_types[0]}
{self.feature_stats[demand_types[0]]}

需求类型2: {demand_types[1]}  
{self.feature_stats[demand_types[1]]}

所有特征列表: {all_features}

请严格按照以下JSON格式返回分析结果，不要添加任何其他文本：

{{
    "需求模式分析": {{
        "{demand_types[0]}": {{
            "特征描述": "描述该需求模式的特点",
            "聚类建议": {{
                "算法选择": "推荐的聚类算法名称",
                "原因": "选择该算法的理由"
            }}
        }},
        "{demand_types[1]}": {{
            "特征描述": "描述该需求模式的特点", 
            "聚类建议": {{
                "算法选择": "推荐的聚类算法名称",
                "原因": "选择该算法的理由"
            }}
        }}
    }},
    "特征解释": {{
        "{demand_types[0]}": {{
            "特征描述": "该需求类型的整体特征描述",
            "主要特征": ["特征1", "特征2", "特征3"]
        }},
        "{demand_types[1]}": {{
            "特征描述": "该需求类型的整体特征描述",
            "主要特征": ["特征1", "特征2", "特征3"]
        }}
    }}
}}

重要要求：
1. 输出必须是有效的JSON格式
2. 所有字符串必须用双引号包围
3. 不要在JSON中使用单引号
4. 确保所有括号正确闭合
5. 不要在JSON外添加任何解释文字
6. 聚类算法选择从DBSCAN、KMeans、层次聚类中选择"""

        try:
            response = self.model.query(prompt, system_prompt=self.system_prompt)
            
            # 清理响应内容，移除可能的markdown标记
            content = response['content'].strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # 尝试解析JSON
            explain_feature = json.loads(content)
            
            return explain_feature
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            # 返回简化的默认结构
            return {
                "需求模式分析": {
                    demand_types[0]: {
                        "特征描述": "需求不规律，存在零需求期",
                        "聚类建议": {
                            "算法选择": "DBSCAN",
                            "原因": "适合处理噪声和密度变化的数据"
                        }
                    },
                    demand_types[1]: {
                        "特征描述": "需求呈块状分布",
                        "聚类建议": {
                            "算法选择": "KMeans", 
                            "原因": "适合处理紧凑的球形聚类"
                        }
                    }
                },
                "特征解释": {
                    demand_types[0]: {
                        "特征描述": "间歇性需求特征，包含时间序列统计特征和需求模式特征",
                        "主要特征": all_features[:5] if len(all_features) >= 5 else all_features
                    },
                    demand_types[1]: {
                        "特征描述": "块状需求特征，包含时间序列统计特征和需求模式特征", 
                        "主要特征": all_features[:5] if len(all_features) >= 5 else all_features
                    }
                }
            }
        except Exception as e:
            print(f"特征解释过程出错: {e}")
            # 返回最基本的默认结构
            return {
                "需求模式分析": {
                    "间歇性需求Intermittent": {
                        "特征描述": "需求不规律，存在零需求期",
                        "聚类建议": {
                            "算法选择": "DBSCAN",
                            "原因": "适合处理噪声和密度变化的数据"
                        }
                    },
                    "块状需求Lumpy": {
                        "特征描述": "需求呈块状分布",
                        "聚类建议": {
                            "算法选择": "KMeans",
                            "原因": "适合处理紧凑的球形聚类"
                        }
                    }
                },
                "特征解释": {
                    "间歇性需求Intermittent": {
                        "特征描述": "间歇性需求特征，包含时间序列统计特征和需求模式特征",
                        "主要特征": ["需求变异系数", "零值序列最大长度", "ADI"]
                    },
                    "块状需求Lumpy": {
                        "特征描述": "块状需求特征，包含时间序列统计特征和需求模式特征",
                        "主要特征": ["需求变异系数", "零值序列最大长度", "ADI"]
                    }
                }
            }
