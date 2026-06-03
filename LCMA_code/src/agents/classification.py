#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分类智能体: 负责将备件按需求模式进行分类
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from src.model_interface import OllamaInterface
from kscorer.kscorer import KScorer


class KScorerWrapper:
    """KScorer包装类，使其接口与sklearn聚类器一致"""
    
    def __init__(self, **kwargs):
        self.kscorer = KScorer()
        self.labels_ = None
        self.centroids_ = None
        self.optimal_k_ = None
        
    def fit_predict(self, X):
        """执行聚类并返回标签"""
        labels, centroids, _ = self.kscorer.fit_predict(X, retall=True)
        self.labels_ = labels
        self.centroids_ = centroids
        self.optimal_k_ = self.kscorer.optimal_
        return labels
    
    def fit(self, X):
        """训练聚类器"""
        self.fit_predict(X)
        return self
    
    def predict(self, X):
        """预测新数据的标签"""
        if self.labels_ is None:
            raise ValueError("必须先调用fit方法")
        return self.labels_


class ClassificationAgent:
    """分类智能体: 负责将备件按需求模式进行分类"""

    def __init__(self, model_interface: OllamaInterface):
        self.model = model_interface
        self.system_prompt = """你是一名专业的数据聚类与分类专家。
你的任务是分析备件需求特征，将相似需求模式的备件分类并标记。
你需要考虑聚类算法的选择、评估聚类质量，并为每个类别生成描述性标签。
请以JSON格式提供你的分析结果和分类建议。"""
        self.classification_history = {}  # 存储历史分类方案及其效果
        self.clustering_methods = {
            'kmeans': KScorerWrapper,
            'dbscan': DBSCAN,
            'hierarchical': AgglomerativeClustering,
            'gmm': GaussianMixture
        }
        self.knowledge_manager = None
        self.features_df = None  # 存储特征数据
        
    def set_knowledge_manager(self, knowledge_manager):
        """设置知识库管理器
        
        Args:
            knowledge_manager: 智能体知识库管理器实例
        """
        self.knowledge_manager = knowledge_manager
        
    def get_knowledge(self, query: str, top_k: int = 3):
        """从分类智能体知识库中检索知识
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict]: 相关文档列表
        """
        if self.knowledge_manager:
            return self.knowledge_manager.search_knowledge("classification", query, top_k)
        return []

    def design_clustering_strategy(self, features_df: Dict, features_explanation: Dict, knowledge: str) -> Dict:
        """设计聚类策略
        
        Args:
            features_df: 特征数据
            features_explanation: 特征解释结果
            knowledge: 从知识库检索到的聚类策略相关知识
            
        Returns:
            Dict: 聚类策略
        """
        print(features_explanation)
        feature_description = features_explanation['需求模式分析']
        demand_type = list(feature_description.keys())
        algo1 = feature_description[demand_type[0]]['聚类建议']['算法选择']
        reason1 = feature_description[demand_type[0]]['聚类建议']['原因']
        algo2 = feature_description[demand_type[1]]['聚类建议']['算法选择']
        reason2 = feature_description[demand_type[1]]['聚类建议']['原因']
        feature_stats = self._generate_intermittent_stats(features_df)
        # 构建提示词
        prompt = f"""
        请分析以下{len(demand_type)}种备件需求模式，并设计合适的聚类策略：

        特征统计摘要:
        需求模式1：{demand_type[0]}，包含有{features_df[demand_type[0]].shape[0]}条序列,{features_df[demand_type[0]].shape[1]}个特征
        统计特征：其中列名表示在特征提取过程中所提取出的特征名称，行表示在不同的需求序列中，特征的统计值分布。
        {feature_stats[demand_type[0]]}
        推荐聚类方法：{algo1}
        推荐理由：{reason1}
        需求模式2：{demand_type[1]}，包含有{features_df[demand_type[1]].shape[0]}条序列,{features_df[demand_type[1]].shape[1]}个特征
        统计特征：其中列名表示在特征提取过程中所提取出的特征名称，行表示在不同的需求序列中，特征的统计值分布。
        {feature_stats[demand_type[1]]}
        推荐聚类方法：{algo2}
        推荐理由：{reason2}

    请给出以下内容:
    1. 聚类前是否需要降维及方法
    2. 建议的类别数量范围及确定方法
    3. 评估聚类质量的指标
    
    以JSON格式返回，格式如下，需要对输出的JSON文本中的符号问题进行检查和修复。确保属性名使用双引号括起来，所有括号、方括号和逗号都正确闭合和匹配:
{{
    "间歇性需求Intermittent": {{
      "降维分析": {{
        "是否需要": 根据特征数量feature_stats中的特征名称来判断：'需要'/'不需要',
        "推荐方法": "降维方法名称",
        "降维后特征数量": 6
      }},
      "类别数量": {{
        "最小值": 2,
        "最大值": 5,
        "确定方法": "确定范围的方法"
      }},
      "评估指标": ["指标1", "指标2"],
      "聚类方法":"{algo1}"
    }},
    "块状需求Lumpy": {{
      "降维分析": {{
        "是否需要": 根据特征数量feature_stats中的特征名称来判断：'需要'/'不需要',
        "推荐方法": "降维方法名称",
        "降维后特征数量": 5
      }},
      "类别数量": {{
        "最小值": 3,
        "最大值": 6,
        "确定方法": "确定范围的方法"
      }},
      "评估指标": ["指标1", "指标2"],
      "聚类方法":"{algo2}"
    }}
}}

注意：
1. 每种需求模式的分析都应基于其特有的特征和统计特性
2. 降维分析要考虑数据的复杂度和特征相关性
3. 类别数量建议要考虑业务实际需求
4. 评估指标要能够有效衡量聚类效果
5. 对输出的JSON文本中的符号问题进行检查和修复。确保属性名使用双引号括起来，所有括号、方括号和逗号都正确闭合和匹配
"""

        response = self.model.query(prompt, system_prompt=self.system_prompt)
        clustering_strategy = self.model.parse_json_response(response['content'])

        return clustering_strategy

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

    def perform_clustering(self, features_df: pd.DataFrame, clustering_strategy: Dict, knowledge: str) -> Tuple[
        Dict, np.ndarray]:
        """执行聚类分析
        
        Args:
            features_df: 特征数据
            clustering_strategy: 聚类策略
            knowledge: 从知识库检索到的聚类执行相关知识
            
        Returns:
            Tuple[Dict, np.ndarray]: 聚类结果和降维特征
        """
        # 存储特征数据以供后续使用
        self.features_df = features_df
        
        """执行聚类分析"""
        # 数据标准化
        scaler = StandardScaler()
        features_with_labels = []
        reduced_feature = []
        demand = list(features_df.keys())
        for demand_type in demand:
            data = features_df[demand_type]
            data = data.drop(columns=['类型'])
            data = data.dropna(axis=1, how='any')
            scaled_features = scaler.fit_transform(data)

            # 降维处理（如果策略中需要）
            if clustering_strategy[demand_type].get('降维分析', {}).get('是否需要', '不需要') == '需要':
                method = clustering_strategy[demand_type].get('降维分析', {}).get('推荐方法', 'PCA')
                dimensions = clustering_strategy[demand_type].get('降维分析', {}).get('降维后特征数量', 2)

                if method.upper() == 'PCA':
                    pca = PCA(n_components=min(dimensions, scaled_features.shape[1]))
                    reduced_features = pca.fit_transform(scaled_features)
                else:
                    # 默认使用PCA
                    pca = PCA(n_components=min(2, scaled_features.shape[1]))
                    reduced_features = pca.fit_transform(scaled_features)
            else:
                reduced_features = scaled_features
            reduced_feature.append(reduced_features)
            # 确定类别数量
            min_clusters = clustering_strategy[demand_type].get('类别数量', {}).get('最小值', 2)
            max_clusters = clustering_strategy[demand_type].get('类别数量', {}).get('最大值', 5)

            # 简单方法：使用中间值作为类别数
            n_clusters = min(max(min_clusters, (min_clusters + max_clusters) // 2), len(features_df))

            # 执行聚类
            algorithm = clustering_strategy[demand_type].get('聚类方法', 'KMeans').lower()

            if algorithm == 'kmeans':
                ks = KScorer()
                labels, centroids, _ = ks.fit_predict(reduced_features, retall=True)
                ks.show()  # 聚类点以及相应的得分高亮显示。这些带标签的点对应于所有指标的平均分数中的局部最大值，因此是选择最佳聚类数的最佳选项
                K = ks.optimal_
                print(f'最佳聚类数为：{K}')
            elif algorithm == 'dbscan':
                # 使用DBSCAN时需要估计eps参数
                from sklearn.neighbors import NearestNeighbors
                
                # 使用更好的eps估计方法
                k = 4  # 通常使用4作为k值
                nn = NearestNeighbors(n_neighbors=k)
                nn.fit(reduced_features)
                distances, _ = nn.kneighbors(reduced_features)
                distances = np.sort(distances[:, k-1])  # 取第k个最近邻的距离
                
                # 使用肘部方法找到合适的eps
                gradients = np.gradient(distances)
                knee_point = np.argmax(gradients)
                eps = distances[knee_point]
                
                print(f"自动估计的DBSCAN eps参数: {eps:.4f}")

                clusterer = DBSCAN(eps=eps, min_samples=3)
                labels = clusterer.fit_predict(reduced_features)
                
                # 检查聚类结果
                unique_labels = np.unique(labels)
                if len(unique_labels) == 1 and unique_labels[0] == -1:
                    # 如果所有点都是噪声，尝试更小的eps
                    eps = eps * 0.5
                    print(f"调整DBSCAN eps参数: {eps:.4f}")
                    clusterer = DBSCAN(eps=eps, min_samples=3)
                    labels = clusterer.fit_predict(reduced_features)
                    
                    # 再次检查
                    unique_labels = np.unique(labels)
                    if len(unique_labels) == 1 and unique_labels[0] == -1:
                        print("DBSCAN参数调整失败，使用KScorer作为后备方案")
                        ks_backup = KScorer()
                        labels, _, _ = ks_backup.fit_predict(reduced_features, retall=True)
                        print(f'使用KScorer后备方案，最佳聚类数为：{ks_backup.optimal_}')

                # 处理DBSCAN可能的噪声点（标签为-1）
                if -1 in labels:
                    noise_indices = np.where(labels == -1)[0]
                    non_noise_labels = labels[labels != -1]
                    
                    if len(non_noise_labels) > 0:  # 确保有非噪声点
                        print(f"处理 {len(noise_indices)} 个噪声点")
                        for idx in noise_indices:
                            # 计算到所有非噪声点的距离
                            non_noise_indices = np.where(labels != -1)[0]
                            if len(non_noise_indices) > 0:
                                distances = np.sqrt(np.sum((reduced_features[idx].reshape(1, -1) -
                                                            reduced_features[non_noise_indices]) ** 2, axis=1))
                                # 分配到最近的类别
                                nearest_idx = non_noise_indices[np.argmin(distances)]
                                labels[idx] = labels[nearest_idx]
            else:
                ks = KScorer()
                labels, centroids, _ = ks.fit_predict(reduced_features, retall=True)
                ks.show()  # 聚类点以及相应的得分高亮显示。这些带标签的点对应于所有指标的平均分数中的局部最大值，因此是选择最佳聚类数的最佳选项
                K = ks.optimal_
                print(f'最佳聚类数为：{K}')

            # 将聚类标签添加到特征集
            features_with_label = features_df[demand_type].copy()
            features_with_label['cluster'] = labels
            features_with_labels.append(features_with_label)
        feature_labels = {
            demand[0]: features_with_labels[0],
            demand[1]: features_with_labels[1]
        }
        reduce_feature = {
            demand[0]: reduced_feature[0],
            demand[1]: reduced_feature[1]
        }
        return feature_labels, reduce_feature

    def analyze_clusters(self, features_with_labels: Dict, features_explanation: Dict, knowledge: str) -> Dict:
        """分析聚类结果并生成类别标签
        
        Args:
            features_with_labels: 带标签的特征数据
            features_explanation: 特征解释结果
            knowledge: 从知识库检索到的聚类分析相关知识
            
        Returns:
            Dict: 聚类分析结果
        """
        # 获取类别统计信息
        demand_types = list(features_with_labels.keys())
        stats = []
        for demand_type in demand_types:
            data = features_with_labels[demand_type]
            data = data.drop(columns=['类型'])
            clusters = data['cluster'].unique()
            cluster_stats = {}

            for cluster in clusters:
                cluster_data = data[data['cluster'] == cluster]
                cluster_stats["类别" + str(cluster)] = {
                    "size": len(cluster_data),
                    "percentage": round(len(cluster_data) / len(data) * 100, 2)
                }

                # 计算每个特征的均值
                for col in data.columns:
                    if col != 'cluster':
                        cluster_stats["类别" + str(cluster)][col] = {
                            "mean": round(cluster_data[col].mean(), 4),
                            "std": round(cluster_data[col].std(), 4)
                        }
            stats.append(cluster_stats)
        class1 = len(np.unique(features_with_labels[demand_types[0]]['cluster']))
        class2 = len(np.unique(features_with_labels[demand_types[1]]['cluster']))
        # 构建提示词
        prompt = f"""
以下包括{demand_types}{len(demand_types)}种需求模式下的聚类结果，并为每种需求模式下每个类别均生成描述性标签：
需求模式1：{demand_types[0]}被划分为{class1}个类别，每个类别的统计特征如下：
{stats[0]}
需求模式2：{demand_types[1]}被划分为{class2}个类别，每个类别的统计特征如下：
{stats[1]}
特征解释:
{json.dumps(features_explanation['特征解释'], indent=2, ensure_ascii=False)}

请为每个需求模式下的所有需求类别均提供以下分析,其中{demand_types[0]}有{class1}个类别，所以其类别ID应该是从1至7；{demand_types[1]}有{class2}个类别，所以其类别ID应该是从1至12:
1. 描述性标签（简短但信息量丰富）
2. 关键特征描述（详细说明特征的重要性）
3. 在备品备件领域的业务解释及业务特点
4. 每种需求模式下的类别ID及特征描述应该与stats中的类别特点一一对应

请以JSON格式返回，格式如下:
{{"{demand_types[0]}": {{
  "clusters": [
    {{
      "id": "类别ID1",
      "label": "描述性标签",
      "key_characteristics": [特征1, 特征2],
      "business_interpretation": "在备品备件领域的业务解释，表明该类需求的业务特点"
    }},
    ...,
    {{
      "id": "类别ID7",
      "label": "描述性标签",
      "key_characteristics": [特征1, 特征2],
      "business_interpretation": "在备品备件领域的业务解释，表明该类需求的业务特点"
    }}
  ],
  "overall_assessment": "整体聚类质量评估"
    }},
"{demand_types[1]}": {{
  "clusters": [
    {{
      "id": "类别ID1",
      "label": "描述性标签",
      "key_characteristics": [特征1, 特征2],
      "business_interpretation": "在备品备件领域的业务解释，表明该类需求的业务特点"
    }},
    ...,
    {{
      "id": "类别ID12",
      "label": "描述性标签",
      "key_characteristics": [特征1, 特征2],
      "business_interpretation": "在备品备件领域的业务解释，表明该类需求的业务特点"
    }}
  ],
  "overall_assessment": "整体聚类质量评估"
  }}
}}
注意：
1. 每个需求模式下的类别ID应唯一，且每个需求模式下的类别都需要进行描述
2. 标签应简洁明了，不超过10个字
3. 每种需求模式下的类别ID及特征描述应该与stats中的类别特点一一对应
4. 每种需求模式下的类别是不一样的，{demand_types[0]}有{class1}个类别，所以类别ID应该是从1至{class1}；{demand_types[1]}有{class2}个类别，所以其类别ID应该是从1至{class2}
"""
        response = self.model.query(prompt, system_prompt=self.system_prompt)
        cluster_analysis = self.model.parse_json_response(response['content'])

        return cluster_analysis

    def process_feedback(self, feedback: Dict) -> Dict:
        """处理分类反馈并生成优化建议
        
        Args:
            feedback: 包含聚类效果评估和调整建议的字典
            
        Returns:
            Dict: 包含聚类优化建议的字典
        """
        # 处理不同类型的反馈格式
        if isinstance(feedback, dict):
            # 如果是字典格式，检查是否有特定的键
            if 'clustering_feedback' in feedback:
                # 单一反馈格式
                feedback_content = feedback['clustering_feedback']
                prompts = [f"聚类反馈: {feedback_content}"]
            else:
                # 多需求类型格式
                demand_types = list(feedback.keys())
                prompts = []
                for demand_type in demand_types:
                    if isinstance(feedback[demand_type], dict):
                        if not feedback[demand_type].get('classification_feedback'):
                            prompts.append(f"{demand_type}分类良好，不需要进行额外调整。")
                        else:
                            prompts.append(f"{demand_type}的分类反馈:\n" + 
                                         json.dumps(feedback[demand_type]['classification_feedback'], 
                                                  indent=2, ensure_ascii=False))
                    else:
                        # 如果值是字符串或其他类型
                        prompts.append(f"{demand_type}的分类反馈: {feedback[demand_type]}")
        else:
            # 如果反馈本身是字符串
            prompts = [f"聚类反馈: {feedback}"]

        # 检索相关知识
        clustering_knowledge = self.get_knowledge("聚类算法选择和参数优化", top_k=3)
        knowledge_prompt = ""
        if clustering_knowledge:
            knowledge_prompt = "\n相关知识：\n" + "\n".join([doc["content"] for doc in clustering_knowledge])

        prompt = f"""
请分析以下备件分类反馈，并提供具体的优化建议：

{chr(10).join(prompts)}

{knowledge_prompt}

请考虑以下方面：
1. 聚类算法是否适合当前数据特征
2. 聚类参数是否需要调整
3. 是否需要预处理步骤
4. 类别数量是否合理

请以JSON格式返回分析结果，格式如下：
{{
    "cluster_adjustments": {{
        "{demand_types[0] if 'demand_types' in locals() and len(demand_types) > 0 else '间歇性需求Intermittent'}": {{
            "algorithm": "建议使用的聚类算法",
            "parameters": {{
                "参数名": "参数值"
            }},
            "preprocessing": ["预处理步骤1", "预处理步骤2"],
            "reason": "调整建议的原因"
        }},
        "{demand_types[1] if 'demand_types' in locals() and len(demand_types) > 1 else '块状需求Lumpy'}": {{
            "algorithm": "建议使用的聚类算法",
            "parameters": {{
                "参数名": "参数值"
            }},
            "preprocessing": ["预处理步骤1", "预处理步骤2"],
            "reason": "调整建议的原因"
        }}
    }}
}}

注意：
1. 算法选择必须从以下选项中选择：kmeans, dbscan, hierarchical, gmm
2. 对于DBSCAN算法，参数名必须使用：eps（邻域距离）和min_samples（最小样本数），不要使用minPts
3. 对于kmeans算法，参数名使用：n_clusters（聚类数量）
4. 对于hierarchical算法，参数名使用：n_clusters（聚类数量）、linkage（链接方法）
5. 对于gmm算法，参数名使用：n_components（组件数量）
6. 参数必须是具体的数值
7. 预处理步骤必须是可执行的具体操作
"""

        try:
            # 请求大模型分析反馈
            response = self.model.query(prompt, system_prompt=self.system_prompt)
            result = self.model.parse_json_response(response['content'])
        except Exception as e:
            print(f"分类反馈处理错误: {e}")
            # 返回一个默认的响应
            result = {
                "cluster_adjustments": {
                    "默认": {
                        "algorithm": "kmeans",
                        "parameters": {"n_clusters": 3},
                        "preprocessing": [],
                        "reason": "默认设置"
                    }
                }
            }
        
        return result

    def apply_optimization(self, optimization_plan: Dict, features_df: Dict = None) -> Dict:
        """应用聚类优化建议
        
        Args:
            optimization_plan: 包含聚类优化建议的字典
            features_df: 特征数据，如果为None则使用存储的特征数据
            
        Returns:
            Dict: 优化后的聚类结果，格式与perform_clustering相同
        """
        if not optimization_plan or 'cluster_adjustments' not in optimization_plan:
            return None

        # 存储优化后的特征数据（带聚类标签）
        optimized_features_with_labels = {}
        adjustments = optimization_plan['cluster_adjustments']
        optimization_summary = []

        # 首先复制所有原始的分类结果，确保不会丢失任何需求类型
        if hasattr(self, 'features_with_labels') and self.features_with_labels:
            optimized_features_with_labels = self.features_with_labels.copy()
            print(f"初始化优化结果，包含需求类型: {list(optimized_features_with_labels.keys())}")

        for demand_type, adjustment in adjustments.items():
            try:
                # 获取聚类算法
                algorithm_name = adjustment.get('algorithm', 'kmeans').lower()
                if algorithm_name not in self.clustering_methods:
                    print(f"警告：未知的聚类算法 {algorithm_name}，将使用默认的 KMeans")
                    algorithm_name = 'kmeans'

                # 获取算法参数并修复DBSCAN参数
                params = adjustment.get('parameters', {})
                
                # 修复DBSCAN参数：将minPts转换为min_samples
                if algorithm_name == 'dbscan' and 'minPts' in params:
                    params['min_samples'] = params.pop('minPts')
                    print(f"修复DBSCAN参数：minPts -> min_samples")
                
                # 创建聚类器实例
                clusterer = None
                try:
                    if algorithm_name == 'kmeans':
                        # KScorerWrapper不需要参数，忽略传入的参数
                        clusterer = self.clustering_methods[algorithm_name]()
                    else:
                        clusterer = self.clustering_methods[algorithm_name](**params)
                except Exception as param_error:
                    print(f"参数错误，使用默认参数: {param_error}")
                    clusterer = self.clustering_methods[algorithm_name]()
                
                # 应用预处理步骤（如果有）
                preprocessing_steps = adjustment.get('preprocessing', [])
                
                # 获取原始特征数据（用于添加聚类标签）
                original_feature_data = None
                
                # 获取用于聚类的特征数据
                if features_df is not None and demand_type in features_df:
                    # 使用传入的特征数据（可能是优化后的特征）
                    original_feature_data = features_df[demand_type].copy()
                    feature_data = features_df[demand_type].copy()
                    if '类型' in feature_data.columns:
                        feature_data = feature_data.drop(columns=['类型'])
                    feature_data = feature_data.dropna(axis=1, how='any')
                    
                    # 数据标准化
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(feature_data)
                    
                    print(f"使用 {demand_type} 的传入特征数据进行聚类，特征维度: {scaled_features.shape}")
                elif self.features_df is not None and demand_type in self.features_df:
                    # 使用存储的特征数据
                    original_feature_data = self.features_df[demand_type].copy()
                    feature_data = self.features_df[demand_type].copy()
                    if '类型' in feature_data.columns:
                        feature_data = feature_data.drop(columns=['类型'])
                    feature_data = feature_data.dropna(axis=1, how='any')
                    
                    # 数据标准化
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(feature_data)
                    
                    print(f"使用 {demand_type} 的存储特征数据进行聚类，特征维度: {scaled_features.shape}")
                else:
                    # 如果没有提供特征数据，保留原始分类结果
                    print(f"警告：未提供 {demand_type} 的特征数据，保留原始分类结果")
                    if hasattr(self, 'features_with_labels') and self.features_with_labels and demand_type in self.features_with_labels:
                        optimized_features_with_labels[demand_type] = self.features_with_labels[demand_type].copy()
                        optimization_summary.append(f"{demand_type}: 保留原始分类结果")
                    continue
                
                # 执行聚类
                labels = None
                clustering_failed = False
                
                try:
                    labels = clusterer.fit_predict(scaled_features)
                except Exception as clustering_error:
                    print(f"聚类执行失败: {clustering_error}")
                    clustering_failed = True
                
                # 特殊处理DBSCAN的聚类结果
                if algorithm_name == 'dbscan' and labels is not None:
                    # 检查是否所有点都被标记为噪声
                    unique_labels = np.unique(labels)
                    if len(unique_labels) == 1 and unique_labels[0] == -1:
                        print(f"警告：DBSCAN将所有点标记为噪声，尝试自动调整eps参数")
                        
                        # 使用k-distance方法自动估计eps参数
                        from sklearn.neighbors import NearestNeighbors
                        k = params.get('min_samples', 5)
                        nn = NearestNeighbors(n_neighbors=min(k, scaled_features.shape[0]))
                        nn.fit(scaled_features)
                        distances, _ = nn.kneighbors(scaled_features)
                        distances = np.sort(distances[:, min(k-1, distances.shape[1]-1)])  # 取第k个最近邻的距离
                        
                        # 使用肘部方法找到合适的eps
                        # 计算距离的梯度变化
                        gradients = np.gradient(distances)
                        # 找到梯度变化最大的点作为eps
                        knee_point = np.argmax(gradients)
                        estimated_eps = distances[knee_point]
                        
                        print(f"自动估计的eps参数: {estimated_eps:.4f}")
                        
                        # 使用估计的eps重新聚类
                        new_params = params.copy()
                        new_params['eps'] = estimated_eps
                        try:
                            clusterer = DBSCAN(**new_params)
                            labels = clusterer.fit_predict(scaled_features)
                        except Exception as retry_error:
                            print(f"重新聚类失败: {retry_error}")
                            clustering_failed = True
                        
                        # 再次检查结果
                        if labels is not None:
                            unique_labels = np.unique(labels)
                            if len(unique_labels) == 1 and unique_labels[0] == -1:
                                # 如果仍然失败，尝试更小的eps
                                estimated_eps = estimated_eps * 0.5
                                print(f"进一步调整eps参数: {estimated_eps:.4f}")
                                new_params['eps'] = estimated_eps
                                try:
                                    clusterer = DBSCAN(**new_params)
                                    labels = clusterer.fit_predict(scaled_features)
                                except Exception as final_retry_error:
                                    print(f"最终重试失败: {final_retry_error}")
                                    clustering_failed = True
                                
                                # 最后检查，如果还是失败就标记为需要KMeans后备
                                if labels is not None:
                                    unique_labels = np.unique(labels)
                                    if len(unique_labels) == 1 and unique_labels[0] == -1:
                                        clustering_failed = True
                    
                    # 处理DBSCAN的噪声点（标签为-1）
                    if labels is not None and -1 in labels and not clustering_failed:
                        noise_indices = np.where(labels == -1)[0]
                        non_noise_labels = labels[labels != -1]
                        
                        if len(non_noise_labels) > 0:  # 确保有非噪声点
                            print(f"处理 {len(noise_indices)} 个噪声点")
                            for idx in noise_indices:
                                # 计算到所有非噪声点的距离
                                non_noise_indices = np.where(labels != -1)[0]
                                if len(non_noise_indices) > 0:
                                    distances = np.sqrt(np.sum((scaled_features[idx].reshape(1, -1) -
                                                                scaled_features[non_noise_indices]) ** 2, axis=1))
                                    # 分配到最近的类别
                                    nearest_idx = non_noise_indices[np.argmin(distances)]
                                    labels[idx] = labels[nearest_idx]
                
                # 如果聚类失败或结果不佳，使用KScorer作为后备方案
                if clustering_failed or labels is None:
                    print(f"{demand_type} 聚类优化失败，使用KScorer作为后备方案")
                    try:
                        from kscorer.kscorer import KScorer
                        ks_backup = KScorer()
                        labels, _, _ = ks_backup.fit_predict(scaled_features, retall=True)
                        print(f'使用KScorer后备方案，最佳聚类数为：{ks_backup.optimal_}')
                        optimization_summary.append(f"{demand_type}: 使用KScorer后备方案，{len(np.unique(labels))}个类别")
                    except Exception as kscorer_error:
                        print(f"KScorer后备方案也失败: {kscorer_error}")
                        # 如果KScorer也失败，保留原始分类结果
                        if hasattr(self, 'features_with_labels') and self.features_with_labels and demand_type in self.features_with_labels:
                            optimized_features_with_labels[demand_type] = self.features_with_labels[demand_type].copy()
                            optimization_summary.append(f"{demand_type}: 所有聚类方法失败，保留原始分类结果")
                        continue
                else:
                    # 如果是KScorerWrapper，显示聚类评分信息
                    if algorithm_name == 'kmeans' and hasattr(clusterer, 'kscorer'):
                        clusterer.kscorer.show()  # 显示聚类评分
                        print(f'最佳聚类数为：{clusterer.optimal_k_}')
                    
                    optimization_summary.append(f"{demand_type}: {algorithm_name}算法，{len(np.unique(labels))}个类别")
                
                # 将聚类标签添加到原始特征数据中（与perform_clustering格式一致）
                if labels is not None:
                    features_with_label = original_feature_data.copy()
                    features_with_label['cluster'] = labels
                    optimized_features_with_labels[demand_type] = features_with_label
                    
                    print(f"{demand_type} 聚类优化完成")
                    print(f"使用算法: {algorithm_name if not clustering_failed else 'KScorer后备'}")
                    print(f"参数: {params if algorithm_name != 'kmeans' and not clustering_failed else {}}")
                    print(f"生成类别数: {len(np.unique(labels))}")
                    if preprocessing_steps:
                        print(f"预处理步骤: {preprocessing_steps}")
                    
            except Exception as e:
                print(f"{demand_type} 聚类优化失败: {str(e)}")
                # 优化失败时尝试使用KScorer后备方案
                try:
                    print(f"尝试为 {demand_type} 使用KScorer后备方案")
                    if features_df is not None and demand_type in features_df:
                        feature_data = features_df[demand_type].copy()
                    elif self.features_df is not None and demand_type in self.features_df:
                        feature_data = self.features_df[demand_type].copy()
                    else:
                        raise Exception("无可用特征数据")
                    
                    original_feature_data = feature_data.copy()
                    if '类型' in feature_data.columns:
                        feature_data = feature_data.drop(columns=['类型'])
                    feature_data = feature_data.dropna(axis=1, how='any')
                    
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(feature_data)
                    
                    from kscorer.kscorer import KScorer
                    ks_backup = KScorer()
                    labels, _, _ = ks_backup.fit_predict(scaled_features, retall=True)
                    print(f'使用KScorer后备方案，最佳聚类数为：{ks_backup.optimal_}')
                    
                    features_with_label = original_feature_data.copy()
                    features_with_label['cluster'] = labels
                    optimized_features_with_labels[demand_type] = features_with_label
                    optimization_summary.append(f"{demand_type}: KScorer后备方案，{len(np.unique(labels))}个类别")
                    
                except Exception as backup_error:
                    print(f"KScorer后备方案也失败: {backup_error}")
                    # 最后保留原始分类结果
                    if hasattr(self, 'features_with_labels') and self.features_with_labels and demand_type in self.features_with_labels:
                        optimized_features_with_labels[demand_type] = self.features_with_labels[demand_type].copy()
                        optimization_summary.append(f"{demand_type}: 所有方法失败，保留原始分类结果")
                continue

        # 如果没有任何结果（包括原始结果），返回None
        if not optimized_features_with_labels:
            print("所有需求类型的聚类优化都失败了，且没有原始分类结果可用")
            return None

        print(f"\n聚类优化总结: {'; '.join(optimization_summary)}")
        print(f"最终优化结果包含需求类型: {list(optimized_features_with_labels.keys())}")
        return optimized_features_with_labels
