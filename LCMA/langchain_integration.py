#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain + DeepSeek 智能备件需求预测助手
支持自然语言交互，智能问题分解和工具调用
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, TypedDict
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
from langchain.tools import tool, Tool
from langchain.tools.render import render_text_description
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    END = START = StateGraph = None

# 添加forecast_model路径
forecast_model_path = os.path.join(os.getcwd(), "..", "forecast_model")
current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(forecast_model_path)

# 导入forecast_model模块
from src.agents.feature_extraction import FeatureExtractionAgent
from src.agents.classification import ClassificationAgent
from src.agents.forecasting_model_select import ForecastingModelSelectAgent
from src.model_interface import OllamaInterface
from src.config import (
    EMBEDDING_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MODELSCOPE_API_KEY,
    MODELSCOPE_BASE_URL,
    MODELSCOPE_MODEL,
    KNOWLEDGE_BASE_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_CHROMA_PATH,
    RAG_ENABLED,
    RAG_HISTORY_PATH,
    RAG_KEYWORD_WEIGHT,
    RAG_OLLAMA_BASE_URL,
    RAG_SEMANTIC_WEIGHT,
)
from src.rag.agent_knowledge_manager import AgentKnowledgeManager
from src.rag.rag_manager import RAGManager


WORKFLOW_TOOL_ORDER = [
    "load_data", "quality_scanning_evaluation", "detect_anomalies_statistical",
    "dynamic_calibration_correction", "normalize_clean_data",
    "analyze_data", "explain_features", "design_clustering_strategy",
    "perform_clustering", "analyze_clusters", "recommend_algorithms",
    "evaluate_algorithms", "analyze_evaluation", "coordinate_agent_feedback",
    "generate_prediction_strategy"
]

TOOL_AGENT_LABELS = {
    "load_data": "数据接入节点",
    "quality_scanning_evaluation": "DMAgent",
    "detect_anomalies_statistical": "DMAgent",
    "dynamic_calibration_correction": "DMAgent",
    "normalize_clean_data": "DMAgent",
    "analyze_data": "FeatureExtractionAgent",
    "explain_features": "FeatureExtractionAgent",
    "design_clustering_strategy": "ClassificationAgent",
    "perform_clustering": "ClassificationAgent",
    "analyze_clusters": "ClassificationAgent",
    "recommend_algorithms": "ForecastingModelSelectAgent",
    "evaluate_algorithms": "ForecastingModelSelectAgent",
    "analyze_evaluation": "ForecastingModelSelectAgent",
    "coordinate_agent_feedback": "FeedbackCoordinator",
    "generate_prediction_strategy": "ForecastingModelSelectAgent",
}

TOOL_GRAPH_NODES = {
    tool_name: f"{tool_name}_node" for tool_name in WORKFLOW_TOOL_ORDER
}


class ForecastGraphState(TypedDict, total=False):
    """LangGraph共享状态：在各智能体节点之间传递任务、执行结果和评估。"""

    user_input: str
    assistant_response: str
    task_plan: Dict[str, Any]
    validation: Dict[str, Any]
    tools_to_execute: List[str]
    completed_steps: List[str]
    remaining_tools: List[str]
    execution_results: Dict[str, Any]
    evaluation: Dict[str, Any]
    graph_trace: List[str]
    last_tool: str
    last_tool_success: bool
    success: bool
    error: str
    message: str


class IntelligentForecastAgent:
    """智能备件需求预测助手"""
    
    def __init__(self):
        # 初始化LLM：默认接入ModelScope DeepSeek V3.1，Ollama作为可选回退
        self.llm = self._create_chat_llm()
        self.embedding_model = EMBEDDING_MODEL
        self.chroma_path = RAG_CHROMA_PATH
        self.rag_enabled = RAG_ENABLED
        self.agent_knowledge_manager = None
        self.history_rag = None
        self._initialize_rag_system()
        
        # 数据文件路径（数据文件在data文件夹中）
        self.data_file_path = os.path.join(current_dir, "data", "异常值处理之后的数据.xlsx")
        
        # 初始化forecast_model系统组件
        try:
            # 使用配置文件中的LLM provider，与LangChain保持一致
            self.model_interface = OllamaInterface()
            
            # 初始化特征提取、分类聚类和模型选择三个领域智能体
            self.feature_agent = FeatureExtractionAgent(self.model_interface)
            self.classification_agent = ClassificationAgent(self.model_interface)
            self.forecasting_agent = ForecastingModelSelectAgent(self.model_interface)
            self._attach_rag_to_agents()
            
            self.use_forecast_model = True  # 启用真正的forecast_model功能
            print("✅ 成功初始化DMAgent + forecast_model智能体系统")
            
        except Exception as e:
            print(f"⚠️ forecast_model智能体初始化失败: {e}")
            print("🔄 回退到简化实现模式")
            self.use_forecast_model = False
        
        # 系统状态 - 严格按照流程顺序
        self.raw_data = None
        self.current_data = None
        self.cleaned_data = None
        self.data_quality_report = None
        self.anomaly_detection_results = None
        self.correction_results = None
        self.global_state = {
            "theta_QS": 0.75,
            "theta_AS": 0.65,
            "theta_P": 0.70,
            "feedback": None,
            "business_events": [],
            "normalization": {},
            "agent_feedback": {},
            "pending_feedback_actions": []
        }
        self.agent_feedback_state = {}
        self.feedback_history = []
        self.loaded_data = None
        self.analyzed_data = None
        self.features_explanation = None
        self.clustering_strategy = None
        self.clustering_results = None
        self.cluster_analysis = None
        self.algorithm_recommendations = None
        self.evaluation_results = None
        self.analysis_results = None
        self.prediction_strategy = None
        self.visualization_results = None
        
        # 对话历史
        self.conversation_history = []
        
        # 定义LangChain工具
        self.tools = self._define_langchain_tools()
        
        # 创建智能助手链
        self.assistant_chain = self._create_assistant_chain()
        self.task_decomposer = self._create_task_decomposer()
        self.tool_executor = self._create_tool_executor()
        self.forecast_graph = self._build_forecast_graph()

    def _create_chat_llm(self):
        """Create the LangChain chat model from .env configuration."""
        if LLM_PROVIDER == "modelscope":
            if ChatOpenAI is None:
                raise ImportError("缺少langchain-openai依赖，无法接入ModelScope API")
            return ChatOpenAI(
                model=MODELSCOPE_MODEL,
                api_key=MODELSCOPE_API_KEY or "EMPTY",
                base_url=MODELSCOPE_BASE_URL,
                temperature=LLM_TEMPERATURE,
                timeout=LLM_TIMEOUT
            )

        if ChatOllama is None:
            raise ImportError("缺少langchain-ollama依赖，无法接入Ollama")
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT
        )

    def _initialize_rag_system(self) -> None:
        """初始化ChromaDB RAG和历史记录库。"""
        try:
            knowledge_path = os.path.join(current_dir, KNOWLEDGE_BASE_PATH)
            history_path = os.path.join(current_dir, RAG_HISTORY_PATH)
            self.agent_knowledge_manager = AgentKnowledgeManager(
                base_path=knowledge_path,
                embedding_model=self.embedding_model,
                chroma_path=self.chroma_path,
                embedding_base_url=RAG_OLLAMA_BASE_URL,
                semantic_weight=RAG_SEMANTIC_WEIGHT,
                keyword_weight=RAG_KEYWORD_WEIGHT
            )
            self.history_rag = RAGManager(
                storage_path=history_path,
                embedding_model=self.embedding_model,
                chroma_path=self.chroma_path,
                embedding_base_url=RAG_OLLAMA_BASE_URL,
                semantic_weight=RAG_SEMANTIC_WEIGHT,
                keyword_weight=RAG_KEYWORD_WEIGHT
            )
            print(f"✅ RAG系统初始化完成: ChromaDB + {self.embedding_model} + 混合检索")
        except Exception as e:
            self.rag_enabled = False
            self.agent_knowledge_manager = None
            self.history_rag = None
            print(f"⚠️ RAG系统初始化失败，将跳过检索增强: {e}")

    def _attach_rag_to_agents(self) -> None:
        """为各智能体注入专属知识库管理器。"""
        if not self.agent_knowledge_manager:
            return
        for agent in [self.feature_agent, self.classification_agent, self.forecasting_agent]:
            if hasattr(agent, "set_knowledge_manager"):
                agent.set_knowledge_manager(self.agent_knowledge_manager)

    def _retrieve_rag_context(self, agent_id: str, query: str, top_k: int = 3) -> str:
        """检索专属知识库和历史记录，用于强化提示词。"""
        if not self.rag_enabled:
            return ""

        context_blocks = []
        try:
            if self.agent_knowledge_manager:
                knowledge_docs = self.agent_knowledge_manager.search_knowledge(agent_id, query, top_k=top_k)
                if knowledge_docs:
                    context_blocks.append(self._format_retrieved_docs("相关领域知识", knowledge_docs))
            if self.history_rag:
                history_docs = self.history_rag.search(query, top_k=top_k, hybrid=True)
                if history_docs:
                    context_blocks.append(self._format_retrieved_docs("相关历史记录", history_docs))
        except Exception as e:
            print(f"RAG检索失败({agent_id}): {e}")

        return "\n\n".join(context_blocks)

    def _format_retrieved_docs(self, title: str, docs: List[Dict[str, Any]]) -> str:
        lines = [f"{title}:"]
        for i, doc in enumerate(docs, start=1):
            content = str(doc.get("content", "")).replace("\n", " ").strip()
            if len(content) > 800:
                content = content[:800] + "..."
            score = doc.get("hybrid_score", doc.get("similarity_score", 0))
            lines.append(f"{i}. score={score:.4f} | {content}")
        return "\n".join(lines)

    def _augment_prompt_with_rag(self, agent_id: str, prompt: str, query: str = None) -> str:
        """把RAG上下文拼接到智能体提示词前。"""
        rag_context = self._retrieve_rag_context(agent_id, query or prompt)
        if not rag_context:
            return prompt
        return f"""请优先参考以下RAG检索上下文，再完成任务。

{rag_context}

当前任务:
{prompt}
"""

    def _store_history_record(
        self,
        user_input: str,
        task_plan: Dict[str, Any],
        execution_results: Dict[str, Any],
        evaluation: Dict[str, Any],
        agent_feedback: Dict[str, Any] = None
    ) -> None:
        """将本轮交互摘要写入历史记录库，供后续提示词增强。"""
        if not self.history_rag:
            return
        try:
            record_id = f"history_{len(self.conversation_history)}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
            content = json.dumps({
                "user_input": user_input,
                "task_plan": task_plan,
                "executed_steps": list(execution_results.keys()),
                "evaluation": evaluation,
                "agent_feedback": agent_feedback or {},
            }, ensure_ascii=False, default=str)
            self.history_rag.add_document(
                record_id,
                content,
                metadata={
                    "type": "interaction_history",
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "steps": ",".join(execution_results.keys())
                }
            )
        except Exception as e:
            print(f"历史记录写入失败: {e}")
        
    def _define_langchain_tools(self) -> List[Tool]:
        """定义LangChain格式的工具 - 严格按照流程顺序"""
        
        @tool
        def load_data() -> Dict[str, Any]:
            """加载备件需求数据 - 系统基础，所有其他功能都必须先执行此步骤
            
            使用固定的Excel数据文件：E:/ToolBench-master/forecast_model/异常值处理之后的数据.xlsx
            
            Returns:
                包含数据加载结果的字典
            """
            try:
                if not os.path.exists(self.data_file_path):
                    return {
                        "success": False,
                        "error": f"数据文件不存在: {self.data_file_path}",
                        "message": "请确保数据文件存在"
                    }
                
                # 加载Excel数据，Draw作为DMAgent的原始感知输入
                data = pd.read_excel(self.data_file_path, parse_dates=True, index_col=0)
                self.raw_data = data.T  # 转置数据，与forecast_model保持一致
                self.current_data = self.raw_data.copy()
                self.cleaned_data = None
                self.loaded_data = None
                self.data_quality_report = None
                self.anomaly_detection_results = None
                self.correction_results = None
                self._reset_downstream_analysis_state()
                
                return {
                    "success": True,
                    "data_shape": self.raw_data.shape,
                    "series_count": len(self.raw_data.index),     # 5000个时间序列 (行数)
                    "time_periods": len(self.raw_data.columns),   # 84个时间期间 (列数)
                    "data_file": self.data_file_path,
                    "state": "Draw",
                    "message": f"成功加载Draw: {self.raw_data.shape[0]}个时间序列 x {self.raw_data.shape[1]}个时间期间"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "数据加载失败"
                }

        @tool
        def quality_scanning_evaluation() -> Dict[str, Any]:
            """DMAgent步骤1：质量扫描与评价，计算QS并决定是否进入深度异常检测。"""
            if self.raw_data is None:
                return {
                    "success": False,
                    "error": "未加载Draw",
                    "message": "必须先执行load_data"
                }

            try:
                self._apply_monitoring_feedback()
                data = self.raw_data.copy()
                numeric_data = data.apply(pd.to_numeric, errors="coerce")
                total_points = int(numeric_data.size)
                missing_count = int(numeric_data.isna().sum().sum())
                duplicate_count = int(numeric_data.duplicated().sum())
                zero_count = int((numeric_data.fillna(np.nan) == 0).sum().sum())

                if total_points == 0:
                    quality_score = 0.0
                    missing_ratio = 1.0
                    zero_ratio = 0.0
                    duplicate_ratio = 0.0
                else:
                    missing_ratio = missing_count / total_points
                    zero_ratio = zero_count / total_points
                    duplicate_ratio = duplicate_count / max(len(numeric_data.index), 1)
                    quality_score = 1.0 - ((missing_count + duplicate_count) / total_points)
                    quality_score = float(np.clip(quality_score, 0.0, 1.0))

                theta_qs = self.global_state["theta_QS"]
                deep_detection_required = quality_score < theta_qs

                if deep_detection_required:
                    self.current_data = numeric_data.copy()
                    routing_decision = "QS低于阈值，进入深度异常检测"
                else:
                    self.current_data = self._zscore_regularize(numeric_data)
                    routing_decision = "QS达到阈值，跳过深度检测并执行轻量Z-score规整"

                self.data_quality_report = {
                    "quality_score": round(quality_score, 4),
                    "theta_QS": theta_qs,
                    "missing_count": missing_count,
                    "duplicate_count": duplicate_count,
                    "total_points": total_points,
                    "missing_ratio": round(missing_ratio, 4),
                    "zero_ratio": round(zero_ratio, 4),
                    "duplicate_ratio": round(duplicate_ratio, 4),
                    "deep_detection_required": deep_detection_required,
                    "routing_decision": routing_decision
                }

                return {
                    "success": True,
                    "quality_report": self.data_quality_report,
                    "message": f"质量扫描完成: QS={quality_score:.4f}, {routing_decision}"
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "质量扫描失败"
                }

        @tool
        def detect_anomalies_statistical() -> Dict[str, Any]:
            """DMAgent步骤2：深度异常监测，结合Isolation Forest异常评分与LLM语义确认。"""
            if self.data_quality_report is None:
                return {
                    "success": False,
                    "error": "未完成质量扫描",
                    "message": "必须先执行quality_scanning_evaluation"
                }

            try:
                if not self.data_quality_report.get("deep_detection_required", False):
                    self.anomaly_detection_results = {
                        "skipped": True,
                        "reason": "QS达到阈值，按闭环策略跳过深度检测",
                        "suspected_count": 0,
                        "confirmed_count": 0,
                        "confirmed_anomalies": []
                    }
                    return {
                        "success": True,
                        "detection": self.anomaly_detection_results,
                        "message": "深度异常检测已跳过"
                    }

                data = self.current_data.copy()
                suspected_anomalies = self._detect_anomalies_with_isolation_forest(
                    data, self.global_state["theta_AS"]
                )
                confirmed_anomalies = self._semantic_confirm_anomalies(
                    suspected_anomalies, data, self.global_state["theta_P"]
                )

                self.anomaly_detection_results = {
                    "skipped": False,
                    "theta_AS": self.global_state["theta_AS"],
                    "theta_P": self.global_state["theta_P"],
                    "suspected_count": len(suspected_anomalies),
                    "confirmed_count": len(confirmed_anomalies),
                    "suspected_sample": suspected_anomalies[:20],
                    "confirmed_anomalies": confirmed_anomalies
                }

                return {
                    "success": True,
                    "detection": self.anomaly_detection_results,
                    "message": (
                        f"深度异常检测完成: 疑似{len(suspected_anomalies)}个, "
                        f"语义确认{len(confirmed_anomalies)}个"
                    )
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "深度异常检测失败"
                }

        @tool
        def dynamic_calibration_correction() -> Dict[str, Any]:
            """DMAgent步骤3：基于确认异常类型动态选择修正策略并更新Dcurrent。"""
            if self.anomaly_detection_results is None:
                return {
                    "success": False,
                    "error": "未完成异常检测",
                    "message": "必须先执行detect_anomalies_statistical"
                }

            try:
                confirmed_anomalies = self.anomaly_detection_results.get("confirmed_anomalies", [])
                if not confirmed_anomalies:
                    self.correction_results = {
                        "corrected_count": 0,
                        "strategy_usage": {},
                        "corrections": [],
                        "message": "无确认异常点，Dcurrent保持不变"
                    }
                    return {
                        "success": True,
                        "correction": self.correction_results,
                        "message": "动态校准完成: 无需修正"
                    }

                data = self.current_data.copy()
                corrections = []
                strategy_usage = {}

                for anomaly in confirmed_anomalies:
                    strategy = self._select_correction_strategy(anomaly)
                    new_value = self._execute_correction_strategy(data, anomaly, strategy)
                    series_id = anomaly["series_id"]
                    time_id = anomaly["time_id"]
                    old_value = data.at[series_id, time_id]
                    data.at[series_id, time_id] = new_value

                    strategy_usage[strategy["name"]] = strategy_usage.get(strategy["name"], 0) + 1
                    corrections.append({
                        "series_id": str(series_id),
                        "time_id": str(time_id),
                        "old_value": None if pd.isna(old_value) else float(old_value),
                        "new_value": float(new_value),
                        "anomaly_type": anomaly.get("anomaly_type", "unknown"),
                        "strategy": strategy["name"],
                        "utility": round(strategy["utility"], 4)
                    })

                self.current_data = data
                self.correction_results = {
                    "corrected_count": len(corrections),
                    "strategy_usage": strategy_usage,
                    "corrections": corrections[:100],
                    "message": "异常点动态校准修正完成"
                }

                return {
                    "success": True,
                    "correction": self.correction_results,
                    "message": f"动态校准完成: 已修正{len(corrections)}个确认异常点"
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "动态校准修正失败"
                }

        @tool
        def normalize_clean_data() -> Dict[str, Any]:
            """DMAgent步骤4：对Dcurrent做Min-Max归一化，输出Dclean并写入Sglobal。"""
            if self.current_data is None:
                return {
                    "success": False,
                    "error": "未生成Dcurrent",
                    "message": "必须先完成DMAgent前置步骤"
                }

            try:
                data = self.current_data.apply(pd.to_numeric, errors="coerce")
                data = data.interpolate(axis=1, limit_direction="both").fillna(0)

                row_min = data.min(axis=1)
                row_max = data.max(axis=1)
                denominator = (row_max - row_min).replace(0, 1)
                cleaned = data.sub(row_min, axis=0).div(denominator, axis=0)
                cleaned = cleaned.clip(0, 1)

                self.cleaned_data = cleaned
                self.loaded_data = cleaned
                self._reset_downstream_analysis_state()
                self.global_state["normalization"] = {
                    "method": "per_series_min_max",
                    "min": row_min.to_dict(),
                    "max": row_max.to_dict(),
                    "feature_range": [0, 1]
                }

                return {
                    "success": True,
                    "state": "Dclean",
                    "data_shape": self.loaded_data.shape,
                    "normalization": {
                        "method": "per_series_min_max",
                        "feature_range": [0, 1],
                        "saved_to": "Sglobal.normalization"
                    },
                    "message": "DMAgent清洗闭环完成，已输出Dclean并保存Min-Max参数"
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "归一化输出失败"
                }
        
        @tool
        def analyze_data() -> Dict[str, Any]:
            """分析数据 - 依赖load_data
            
            对加载的备件需求数据进行初步分析，识别数据特征和模式
            
            Returns:
                数据分析结果
            """
            if self.loaded_data is None:
                return {
                    "success": False,
                    "error": "未加载数据",
                    "message": "必须先执行load_data"
                }
            
            try:
                if self.use_forecast_model:
                    try:
                        # 使用真实的特征提取智能体
                        knowledge = self.feature_agent.get_knowledge("时间序列特征提取方法", top_k=3)
                        knowledge_text = "\n".join([doc.get("content", "") for doc in knowledge]) if knowledge else ""
                        rag_context = self._retrieve_rag_context("feature_extraction", "时间序列特征提取 数据分析 历史记录")
                        if rag_context:
                            knowledge_text = f"{knowledge_text}\n\n{rag_context}"
                        self.analyzed_data = self.feature_agent.analyze_data(self.loaded_data, knowledge_text)
                    except Exception as agent_error:
                        print(f"⚠️ 智能体分析失败，使用简化实现: {agent_error}")
                        # 降级到简化实现
                        self.analyzed_data = {
                            "basic_stats": self.loaded_data.describe().to_dict(),
                            "null_values": self.loaded_data.isnull().sum().to_dict(),
                            "data_types": self.loaded_data.dtypes.to_dict(),
                            "fallback_mode": True
                        }
                else:
                    # 简化实现
                    self.analyzed_data = {
                        "basic_stats": self.loaded_data.describe().to_dict(),
                        "null_values": self.loaded_data.isnull().sum().to_dict(),
                        "data_types": self.loaded_data.dtypes.to_dict()
                    }
                
                return {
                    "success": True,
                    "analysis_completed": True,
                    "data_characteristics": self.analyzed_data,
                    "message": "数据分析完成"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "数据分析失败"
                }
        
        @tool
        def explain_features() -> Dict[str, Any]:
            """解释特征 - 依赖analyze_data"""
            if self.analyzed_data is None:
                return {
                    "success": False,
                    "error": "未完成数据分析",
                    "message": "必须先执行analyze_data"
                }
            
            try:
                if self.use_forecast_model:
                    knowledge = self.feature_agent.get_knowledge("时间序列特征工程和特征解释方法", top_k=3)
                    knowledge_text = "\n".join([doc.get("content", "") for doc in knowledge]) if knowledge else ""
                    rag_context = self._retrieve_rag_context("feature_extraction", "时间序列特征解释 特征工程 历史反馈")
                    if rag_context:
                        knowledge_text = f"{knowledge_text}\n\n{rag_context}"
                    
                    # 多次尝试获取有效的特征解释
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            self.features_explanation = self.feature_agent.explain_features(
                                self.analyzed_data, knowledge_text
                            )
                            
                            # 验证返回的数据格式
                            if isinstance(self.features_explanation, dict) and len(self.features_explanation) > 0:
                                # 检查是否包含必要的键
                                if '需求模式分析' in self.features_explanation:
                                    break
                                else:
                                    print(f"尝试 {attempt + 1}: 特征解释格式不完整，重新生成...")
                            else:
                                print(f"尝试 {attempt + 1}: 特征解释为空，重新生成...")
                                
                        except Exception as e:
                            print(f"尝试 {attempt + 1}: 特征解释失败 - {e}")
                            
                        if attempt == max_retries - 1:
                            # 最后一次失败时，使用默认值
                            print("使用默认特征解释...")
                            self.features_explanation = {
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
                                }
                            }
                else:
                    self.features_explanation = {
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
                        }
                    }
                
                return {
                    "success": True,
                    "features_explained": True,
                    "explanation": self.features_explanation,
                    "message": "特征解释完成"
                }
                
            except Exception as e:
                # 发生异常时也提供默认值
                self.features_explanation = {
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
                    }
                }
                
                return {
                    "success": True,  # 即使出错也返回成功，因为有默认值
                    "features_explained": True,
                    "explanation": self.features_explanation,
                    "message": f"特征解释完成(使用默认值): {str(e)}"
                }
        
        @tool
        def design_clustering_strategy() -> Dict[str, Any]:
            """设计聚类策略 - 依赖explain_features"""
            if self.features_explanation is None:
                return {
                    "success": False,
                    "error": "未完成特征解释",
                    "message": "必须先执行explain_features"
                }
            
            try:
                # 验证features_explanation的格式
                if not isinstance(self.features_explanation, dict) or len(self.features_explanation) == 0:
                    # 提供默认的特征解释
                    self.features_explanation = {
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
                        }
                    }
                
                if self.use_forecast_model and '需求模式分析' in self.features_explanation:
                    knowledge = self.classification_agent.get_knowledge("时间序列数据聚类方法和最佳实践", top_k=3)
                    knowledge_text = "\n".join([doc.get("content", "") for doc in knowledge]) if knowledge else ""
                    rag_context = self._retrieve_rag_context("classification", "聚类策略 需求分类 历史聚类反馈")
                    if rag_context:
                        knowledge_text = f"{knowledge_text}\n\n{rag_context}"
                    self.clustering_strategy = self.classification_agent.design_clustering_strategy(
                        self.analyzed_data, self.features_explanation, knowledge_text
                    )
                else:
                    # 使用简化的聚类策略
                    self.clustering_strategy = {
                        "间歇性需求Intermittent": {
                            "降维分析": {
                                "是否需要": "需要",
                                "推荐方法": "PCA",
                                "降维后特征数量": 6
                            },
                            "类别数量": {
                                "最小值": 2,
                                "最大值": 5,
                                "确定方法": "肘部法则"
                            },
                            "评估指标": ["轮廓系数", "Calinski-Harabasz指数"],
                            "聚类方法": "DBSCAN"
                        },
                        "块状需求Lumpy": {
                            "降维分析": {
                                "是否需要": "需要",
                                "推荐方法": "PCA",
                                "降维后特征数量": 5
                            },
                            "类别数量": {
                                "最小值": 3,
                                "最大值": 6,
                                "确定方法": "肘部法则"
                            },
                            "评估指标": ["轮廓系数", "Calinski-Harabasz指数"],
                            "聚类方法": "KMeans"
                        }
                    }
                
                return {
                    "success": True,
                    "strategy_designed": True,
                    "strategy": self.clustering_strategy,
                    "message": "聚类策略设计完成"
                }
                
            except Exception as e:
                # 发生异常时也提供默认策略
                self.clustering_strategy = {
                    "间歇性需求Intermittent": {
                        "降维分析": {
                            "是否需要": "需要",
                            "推荐方法": "PCA",
                            "降维后特征数量": 6
                        },
                        "类别数量": {
                            "最小值": 2,
                            "最大值": 5,
                            "确定方法": "肘部法则"
                        },
                        "评估指标": ["轮廓系数", "Calinski-Harabasz指数"],
                        "聚类方法": "DBSCAN"
                    },
                    "块状需求Lumpy": {
                        "降维分析": {
                            "是否需要": "需要",
                            "推荐方法": "PCA",
                            "降维后特征数量": 5
                        },
                        "类别数量": {
                            "最小值": 3,
                            "最大值": 6,
                            "确定方法": "肘部法则"
                        },
                        "评估指标": ["轮廓系数", "Calinski-Harabasz指数"],
                        "聚类方法": "KMeans"
                    }
                }
                
                return {
                    "success": True,  # 即使出错也返回成功，因为有默认值
                    "strategy_designed": True,
                    "strategy": self.clustering_strategy,
                    "message": f"聚类策略设计完成(使用默认值): {str(e)}"
                }
        
        @tool
        def perform_clustering() -> Dict[str, Any]:
            """执行聚类 - 依赖design_clustering_strategy"""
            if self.clustering_strategy is None:
                return {
                    "success": False,
                    "error": "未设计聚类策略",
                    "message": "必须先执行design_clustering_strategy"
                }
            
            try:
                if self.use_forecast_model and isinstance(self.analyzed_data, dict):
                    knowledge = self.classification_agent.get_knowledge("时间序列聚类算法实现和参数优化", top_k=3)
                    knowledge_text = "\n".join([doc.get("content", "") for doc in knowledge]) if knowledge else ""
                    rag_context = self._retrieve_rag_context("classification", "时间序列聚类 参数优化 历史执行结果")
                    if rag_context:
                        knowledge_text = f"{knowledge_text}\n\n{rag_context}"
                    
                    try:
                        # self.analyzed_data 是特征提取的结果，包含两种需求类型的DataFrame
                        features_with_labels, reduced_features = self.classification_agent.perform_clustering(
                            self.analyzed_data, self.clustering_strategy, knowledge_text
                        )
                        self.clustering_results = {
                            "features_with_labels": features_with_labels,
                            "reduced_features": reduced_features
                        }
                    except Exception as clustering_error:
                        print(f"⚠️ 智能体聚类失败，使用简化实现: {clustering_error}")
                        # 使用简化的聚类实现
                        self.clustering_results = self._simplified_clustering()
                else:
                    # 使用简化的聚类实现
                    self.clustering_results = self._simplified_clustering()
                
                return {
                    "success": True,
                    "clustering_completed": True,
                    "results": self.clustering_results,
                    "message": "聚类执行完成"
                }
                
            except Exception as e:
                # 发生异常时也提供默认结果
                self.clustering_results = self._simplified_clustering()
                return {
                    "success": True,  # 即使出错也返回成功，因为有默认值
                    "clustering_completed": True,
                    "results": self.clustering_results,
                    "message": f"聚类执行完成(使用默认值): {str(e)}"
                }
        
        @tool
        def analyze_clusters() -> Dict[str, Any]:
            """分析聚类结果 - 依赖perform_clustering"""
            if self.clustering_results is None:
                return {
                    "success": False,
                    "error": "未完成聚类",
                    "message": "必须先执行perform_clustering"
                }
            
            try:
                if self.use_forecast_model and self.features_explanation and '需求模式分析' in self.features_explanation:
                    knowledge = self.classification_agent.get_knowledge("时间序列聚类结果分析和评估方法", top_k=3)
                    knowledge_text = "\n".join([doc.get("content", "") for doc in knowledge]) if knowledge else ""
                    rag_context = self._retrieve_rag_context("classification", "聚类结果分析 业务解释 历史记录")
                    if rag_context:
                        knowledge_text = f"{knowledge_text}\n\n{rag_context}"
                    
                    # 确保features_explanation格式正确
                    if '特征解释' not in self.features_explanation:
                        # 添加特征解释部分
                        self.features_explanation['特征解释'] = {
                            "间歇性需求Intermittent": {
                                "特征描述": "需求不规律，存在零需求期，特征包括需求间隔、波动性等",
                                "主要特征": ["需求间隔", "需求量变异系数", "零需求期比例"]
                            },
                            "块状需求Lumpy": {
                                "特征描述": "需求呈块状分布，大量需求集中在某些时期",
                                "主要特征": ["需求集中度", "突发需求强度", "需求频率"]
                            }
                        }
                    
                    try:
                        self.cluster_analysis = self.classification_agent.analyze_clusters(
                            self.clustering_results["features_with_labels"],
                            self.features_explanation, knowledge_text
                        )
                    except Exception as agent_error:
                        print(f"⚠️ 智能体聚类分析失败，使用简化实现: {agent_error}")
                        # 使用简化的聚类分析
                        self.cluster_analysis = self._simplified_cluster_analysis()
                else:
                    # 使用简化的聚类分析
                    self.cluster_analysis = self._simplified_cluster_analysis()
                
                # 按照用户要求：把analyze_clusters函数的输出传给大模型，构建提示词请求大模型用一段话进行分析
                cluster_analysis_prompt = f"""
请对以下聚类分析结果进行总结分析，用一段话说明主要发现和业务价值：

聚类分析结果：
{json.dumps(self.cluster_analysis, indent=2, ensure_ascii=False)}

请从以下角度进行分析：
1. 识别出了哪些主要的需求模式和类别
2. 每种需求模式下的聚类数量和特点
3. 对备件库存管理的实际指导意义
4. 关键的业务洞察和建议

要求：用一段连贯的话进行分析，突出实际应用价值。
"""
                cluster_analysis_prompt = self._augment_prompt_with_rag(
                    "classification",
                    cluster_analysis_prompt,
                    "聚类分析结果总结 业务价值 历史聚类记录"
                )
                
                # 调用大模型分析
                llm_response = self.llm.invoke(cluster_analysis_prompt)
                
                # 提取大模型回复内容
                if hasattr(llm_response, 'content'):
                    llm_analysis = llm_response.content.strip()
                else:
                    llm_analysis = str(llm_response).strip()
                
                # 按照用户要求：把结果分析print出来
                print("\n" + "="*80)
                print("🎯 聚类分析结果总结")
                print("="*80)
                print(llm_analysis)
                print("="*80)
                
                return {
                    "success": True,
                    "analysis_completed": True,
                    "cluster_analysis": self.cluster_analysis,
                    "llm_analysis": llm_analysis,
                    "message": "聚类分析完成，大模型分析已输出"
                }
                
            except Exception as e:
                # 发生异常时也提供默认分析
                self.cluster_analysis = self._simplified_cluster_analysis()
                
                # 即使出错也要调用大模型分析
                try:
                    cluster_analysis_prompt = f"""
请对以下聚类分析结果进行总结分析，用一段话说明主要发现和业务价值：

聚类分析结果：
{json.dumps(self.cluster_analysis, indent=2, ensure_ascii=False)}

请从以下角度进行分析：
1. 识别出了哪些主要的需求模式和类别
2. 每种需求模式下的聚类数量和特点
3. 对备件库存管理的实际指导意义
4. 关键的业务洞察和建议

要求：用一段连贯的话进行分析，突出实际应用价值。
"""
                    cluster_analysis_prompt = self._augment_prompt_with_rag(
                        "classification",
                        cluster_analysis_prompt,
                        "聚类分析结果总结 默认分析 历史记录"
                    )
                    
                    llm_response = self.llm.invoke(cluster_analysis_prompt)
                    
                    if hasattr(llm_response, 'content'):
                        llm_analysis = llm_response.content.strip()
                    else:
                        llm_analysis = str(llm_response).strip()
                    
                    # 按照用户要求：把结果分析print出来
                    print("\n" + "="*80)
                    print("🎯 聚类分析结果总结")
                    print("="*80)
                    print(llm_analysis)
                    print("="*80)
                    
                except Exception as llm_error:
                    print(f"⚠️ 大模型分析失败: {llm_error}")
                    llm_analysis = "聚类分析已完成，但大模型总结生成失败"
                    print("\n" + "="*80)
                    print("🎯 聚类分析结果总结")
                    print("="*80)
                    print(llm_analysis)
                    print("="*80)
                
                return {
                    "success": True,  # 即使出错也返回成功，因为有默认值
                    "analysis_completed": True,
                    "cluster_analysis": self.cluster_analysis,
                    "llm_analysis": llm_analysis,
                    "message": f"聚类分析完成(使用默认值): {str(e)}"
                }
        
        @tool
        def recommend_algorithms() -> Dict[str, Any]:
            """推荐算法 - 依赖analyze_clusters"""
            if self.cluster_analysis is None:
                return {
                    "success": False,
                    "error": "未完成聚类分析",
                    "message": "必须先执行analyze_clusters"
                }
            
            try:
                if self.use_forecast_model:
                    knowledge = self.forecasting_agent.get_knowledge("时间序列预测模型选择和参数优化", top_k=3)
                    knowledge_text = "\n".join([doc.get("content", "") for doc in knowledge]) if knowledge else ""
                    rag_context = self._retrieve_rag_context("forecasting_model_select", "预测模型选择 算法推荐 历史评估")
                    if rag_context:
                        knowledge_text = f"{knowledge_text}\n\n{rag_context}"
                    self.algorithm_recommendations = self.forecasting_agent.recommend_algorithms(
                        self.cluster_analysis, self.features_explanation, knowledge_text
                    )
                else:
                    self.algorithm_recommendations = {
                        "cluster_0": {"algorithm": "ARIMA", "confidence": 0.85},
                        "cluster_1": {"algorithm": "Croston", "confidence": 0.90},
                        "cluster_2": {"algorithm": "TSB", "confidence": 0.86},
                        "cluster_3": {"algorithm": "SBA", "confidence": 0.84},
                        "cluster_4": {"algorithm": "RF", "confidence": 0.82},
                        "cluster_5": {"algorithm": "LightGBM", "confidence": 0.83},
                        "cluster_6": {"algorithm": "LSTM", "confidence": 0.78},
                        "cluster_7": {"algorithm": "ETS", "confidence": 0.80}
                    }
                
                return {
                    "success": True,
                    "recommendations_generated": True,
                    "algorithms": self.algorithm_recommendations,
                    "message": "算法推荐完成"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "算法推荐失败"
                }
        
        @tool
        def evaluate_algorithms() -> Dict[str, Any]:
            """评估算法 - 依赖recommend_algorithms"""
            if self.algorithm_recommendations is None:
                return {
                    "success": False,
                    "error": "未完成算法推荐",
                    "message": "必须先执行recommend_algorithms"
                }
            
            if self.use_forecast_model:
                knowledge = self.forecasting_agent.get_knowledge("时间序列预测算法评估方法和指标选择", top_k=3)
                knowledge_text = "\n".join([doc.get("content", "") for doc in knowledge]) if knowledge else ""
                rag_context = self._retrieve_rag_context("forecasting_model_select", "预测算法评估 参数优化 历史效果")
                if rag_context:
                    knowledge_text = f"{knowledge_text}\n\n{rag_context}"
                self.evaluation_results = self.forecasting_agent.evaluate_algorithms(
                    self.loaded_data,
                    self.clustering_results["features_with_labels"],
                    self.algorithm_recommendations, knowledge_text
                )
            else:
                self.evaluation_results = {
                    "performance_metrics": {
                        "ARIMA": {"MAE": 0.15, "RMSE": 0.20, "RRMSE": 0.28, "MASE": 0.82, "RelMAE": 0.90, "sMAPE": 12.0, "NRMSE": 0.25},
                        "Croston": {"MAE": 0.18, "RMSE": 0.22, "RRMSE": 0.31, "MASE": 0.98, "RelMAE": 1.05, "sMAPE": 15.0, "NRMSE": 0.28},
                        "TSB": {"MAE": 0.16, "RMSE": 0.20, "RRMSE": 0.29, "MASE": 0.88, "RelMAE": 0.95, "sMAPE": 13.0, "NRMSE": 0.25},
                        "SBA": {"MAE": 0.17, "RMSE": 0.21, "RRMSE": 0.30, "MASE": 0.93, "RelMAE": 1.00, "sMAPE": 14.0, "NRMSE": 0.27},
                        "RF": {"MAE": 0.16, "RMSE": 0.19, "RRMSE": 0.27, "MASE": 0.87, "RelMAE": 0.94, "sMAPE": 13.0, "NRMSE": 0.24},
                        "LightGBM": {"MAE": 0.15, "RMSE": 0.18, "RRMSE": 0.26, "MASE": 0.81, "RelMAE": 0.89, "sMAPE": 12.0, "NRMSE": 0.23},
                        "LSTM": {"MAE": 0.18, "RMSE": 0.23, "RRMSE": 0.33, "MASE": 0.99, "RelMAE": 1.08, "sMAPE": 15.0, "NRMSE": 0.29},
                        "ETS": {"MAE": 0.17, "RMSE": 0.21, "RRMSE": 0.30, "MASE": 0.92, "RelMAE": 0.99, "sMAPE": 14.0, "NRMSE": 0.27}
                    },
                    "best_algorithm": "LightGBM"
                }
            
            return {
                "success": True,
                "evaluation_completed": True,
                "results": self.evaluation_results,
                "message": "算法评估完成"
            }
        
        @tool
        def analyze_evaluation() -> Dict[str, Any]:
            """分析评估结果 - 依赖evaluate_algorithms"""
            if self.evaluation_results is None:
                return {
                    "success": False,
                    "error": "未完成算法评估",
                    "message": "必须先执行evaluate_algorithms"
                }
            
            try:
                if self.use_forecast_model:
                    knowledge = self.forecasting_agent.get_knowledge("时间序列预测结果分析和解释方法", top_k=3)
                    knowledge_text = "\n".join([doc.get("content", "") for doc in knowledge]) if knowledge else ""
                    rag_context = self._retrieve_rag_context("forecasting_model_select", "预测评估分析 反馈优化 历史记录")
                    if rag_context:
                        knowledge_text = f"{knowledge_text}\n\n{rag_context}"
                    self.analysis_results = self.forecasting_agent.analyze_evaluation(
                        self.evaluation_results, self.cluster_analysis, knowledge_text
                    )
                else:
                    self.analysis_results = {
                        "best_performers": ["LightGBM", "ARIMA", "RF"],
                        "improvement_suggestions": ["增加特征工程", "调整模型参数"],
                        "confidence_level": "高"
                    }
                self.agent_feedback_state = {}
                
                return {
                    "success": True,
                    "analysis_completed": True,
                    "analysis": self.analysis_results,
                    "message": "评估分析完成"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "评估分析失败"
                }

        @tool
        def coordinate_agent_feedback() -> Dict[str, Any]:
            """协调智能体间反馈 - 依赖analyze_evaluation。

            将模型评估分析结果拆分为:
            1. 模型选择智能体自反馈
            2. 特征提取智能体反馈
            3. 分类聚类智能体反馈
            4. DMAgent数据监控反馈
            并写入Sglobal，供下一轮闭环优化使用。
            """
            if self.analysis_results is None:
                return {
                    "success": False,
                    "error": "未完成评估分析",
                    "message": "必须先执行analyze_evaluation"
                }

            try:
                feedback_result = self._coordinate_agent_feedback()
                return {
                    "success": True,
                    "feedback_coordinated": True,
                    "feedback": feedback_result,
                    "message": "智能体间反馈协调完成"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "智能体间反馈协调失败"
                }
        
        @tool
        def generate_prediction_strategy() -> Dict[str, Any]:
            """生成预测策略 - 依赖coordinate_agent_feedback"""
            if self.analysis_results is None:
                return {
                    "success": False,
                    "error": "未完成评估分析",
                    "message": "必须先执行analyze_evaluation"
                }
            if not self.agent_feedback_state:
                return {
                    "success": False,
                    "error": "未完成智能体反馈协调",
                    "message": "必须先执行coordinate_agent_feedback"
                }
            
            try:
                self.prediction_strategy = {
                    "recommended_models": self.analysis_results.get("best_performers", []),
                    "implementation_plan": {
                        "phase_1": "数据预处理和特征工程",
                        "phase_2": "模型训练和验证", 
                        "phase_3": "生产部署和监控"
                    },
                    "performance_targets": {
                        "accuracy": ">85%",
                        "response_time": "<1s",
                        "update_frequency": "每周"
                    }
                }
                
                return {
                    "success": True,
                    "strategy_generated": True,
                    "strategy": self.prediction_strategy,
                    "message": "预测策略生成完成"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": "预测策略生成失败"
                }
        
        return [
            load_data, quality_scanning_evaluation, detect_anomalies_statistical,
            dynamic_calibration_correction, normalize_clean_data,
            analyze_data, explain_features, design_clustering_strategy,
            perform_clustering, analyze_clusters, recommend_algorithms,
            evaluate_algorithms, analyze_evaluation, coordinate_agent_feedback,
            generate_prediction_strategy
        ]
    
    def _create_assistant_chain(self):
        """创建智能助手链 - 理解用户需求"""
        
        system_prompt = """你是一名专业的备件需求预测专家助手。你需要理解用户的自然语言需求，并提供专业的回答和建议。

你的专业领域包括：
- 异常感知数据监控与闭环清洗
- 备件需求数据分析
- 时间序列特征提取
- 需求模式聚类分析
- 预测算法选择和评估
- 预测策略制定

你需要：
1. 理解用户的具体需求和目标
2. 识别用户可能需要的分析步骤
3. 提供专业的建议和解释
4. 必要时提醒用户工作流的依赖关系

RAG检索上下文：
{rag_context}

请用专业但易懂的语言回答用户问题。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_input}")
        ])
        
        return prompt | self.llm | StrOutputParser()
    
    def _create_task_decomposer(self):
        """创建任务分解器 - 分析需要调用哪些工具"""
        
        # 渲染工具描述
        rendered_tools = render_text_description(self.tools)
        
        system_prompt = f"""你是一名专业的任务分解专家。根据用户的需求，你需要确定需要调用哪些工具函数来完成任务。

可用的工具函数（按依赖顺序）：
{rendered_tools}

工具依赖关系和使用场景：
1. load_data：所有功能的基础，必须首先执行
2. quality_scanning_evaluation：DMAgent质量扫描，计算QS，依赖load_data
3. detect_anomalies_statistical：DMAgent深度异常监测，依赖quality_scanning_evaluation
4. dynamic_calibration_correction：DMAgent动态校准修正，依赖detect_anomalies_statistical
5. normalize_clean_data：DMAgent归一化输出Dclean，依赖dynamic_calibration_correction
6. analyze_data：特征提取数据分析，依赖normalize_clean_data
7. explain_features：特征解释，依赖analyze_data
8. design_clustering_strategy：设计聚类策略，依赖explain_features
9. perform_clustering：执行聚类，依赖design_clustering_strategy
10. analyze_clusters：分析聚类结果，依赖perform_clustering
11. recommend_algorithms：推荐算法，依赖analyze_clusters
12. evaluate_algorithms：评估算法，依赖recommend_algorithms
13. analyze_evaluation：分析评估结果，依赖evaluate_algorithms
14. coordinate_agent_feedback：协调DMAgent、特征提取、分类聚类、模型选择之间的闭环反馈，依赖analyze_evaluation
15. generate_prediction_strategy：生成预测策略，依赖coordinate_agent_feedback

用户需求类型判断：
- 只要数据加载/查看 → 只需要：['load_data']
- 数据清洗/异常检测/质量监控 → 需要：['load_data', 'quality_scanning_evaluation', 'detect_anomalies_statistical', 'dynamic_calibration_correction', 'normalize_clean_data']
- 数据分析 → 需要：DMAgent清洗流程 + ['analyze_data']
- 特征分析 → 需要：DMAgent清洗流程 + ['analyze_data', 'explain_features']
- 聚类分析 → 需要：DMAgent清洗流程 + ['analyze_data', 'explain_features', 'design_clustering_strategy', 'perform_clustering', 'analyze_clusters']
- 算法推荐 → 需要：聚类分析的基础上 + ['recommend_algorithms']
- 完整预测方案 → 需要：所有工具

请仔细分析用户的具体需求，确定合适的工具调用边界。不要过度执行！

RAG检索上下文：
{rag_context}

请分析用户需求，返回一个包含三个字段的JSON对象：
- 第一个字段名为"analysis"，值为对用户需求的文字分析
- 第二个字段名为"required_tools"，值为需要调用的工具名称列表
- 第三个字段名为"reasoning"，值为选择这些工具的理由说明

只返回JSON格式，不要其他内容。"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{user_input}")
        ])
        
        return prompt | self.llm | JsonOutputParser()
    
    def _create_tool_executor(self):
        """创建工具执行器"""
        def execute_tool(tool_name: str):
            for tool in self.tools:
                if tool.name == tool_name:
                    return tool.invoke({})
            return {"success": False, "error": f"未找到工具: {tool_name}"}
        
        return execute_tool
    
    def _validate_workflow_order(self, required_tools: List[str]) -> Dict[str, Any]:
        """验证工作流顺序并返回需要执行的工具列表"""
        # 定义严格的工具顺序
        tool_order = WORKFLOW_TOOL_ORDER
        
        if not required_tools:
            return {
                "valid": False,
                "error": "没有指定需要执行的工具",
                "tools_to_execute": []
            }
        
        # 验证所有工具都存在
        invalid_tools = [tool for tool in required_tools if tool not in tool_order]
        if invalid_tools:
            return {
                "valid": False,
                "error": f"未知的工具: {invalid_tools}",
                "tools_to_execute": []
            }
        
        # 自动补齐前置依赖，确保DMAgent始终位于特征提取、分类和模型选择之前
        max_required_index = max(tool_order.index(tool) for tool in required_tools)
        tools_to_execute = tool_order[:max_required_index + 1]
        
        return {
            "valid": True,
            "tools_to_execute": tools_to_execute,
            "message": f"工具序列验证通过，需要执行{len(tools_to_execute)}个步骤: {' → '.join(tools_to_execute)}"
        }
    
    def _get_completed_steps(self) -> List[str]:
        """获取已完成的步骤"""
        step_status = {
            "load_data": self.raw_data is not None,
            "quality_scanning_evaluation": self.data_quality_report is not None,
            "detect_anomalies_statistical": self.anomaly_detection_results is not None,
            "dynamic_calibration_correction": self.correction_results is not None,
            "normalize_clean_data": self.cleaned_data is not None,
            "analyze_data": self.analyzed_data is not None,
            "explain_features": self.features_explanation is not None,
            "design_clustering_strategy": self.clustering_strategy is not None,
            "perform_clustering": self.clustering_results is not None,
            "analyze_clusters": self.cluster_analysis is not None,
            "recommend_algorithms": self.algorithm_recommendations is not None,
            "evaluate_algorithms": self.evaluation_results is not None,
            "analyze_evaluation": self.analysis_results is not None,
            "coordinate_agent_feedback": bool(self.agent_feedback_state),
            "generate_prediction_strategy": self.prediction_strategy is not None
        }
        
        return [step for step, completed in step_status.items() if completed]

    def _reset_downstream_analysis_state(self) -> None:
        """Reset outputs derived from Dclean so reruns do not reuse stale agent state."""
        self.analyzed_data = None
        self.features_explanation = None
        self.clustering_strategy = None
        self.clustering_results = None
        self.cluster_analysis = None
        self.algorithm_recommendations = None
        self.evaluation_results = None
        self.analysis_results = None
        self.prediction_strategy = None
        self.visualization_results = None
        self.agent_feedback_state = {}
        self.global_state["agent_feedback"] = {}
        self.global_state["pending_feedback_actions"] = []

    def update_monitoring_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """向Sglobal写入DMAgent闭环反馈信号，下一轮质量扫描时生效。"""
        self.global_state["feedback"] = feedback or {}
        return {
            "success": True,
            "feedback": self.global_state["feedback"],
            "message": "反馈信号已写入Sglobal，将在下一轮DMAgent清洗周期中应用"
        }

    def _apply_monitoring_feedback(self) -> None:
        """根据Sglobal.feedback自适应调整质量阈值和异常检测参数。"""
        feedback = self.global_state.get("feedback")
        if not feedback:
            return

        if "theta_QS" in feedback:
            self.global_state["theta_QS"] = float(np.clip(feedback["theta_QS"], 0.0, 1.0))
        if "theta_AS" in feedback:
            self.global_state["theta_AS"] = float(np.clip(feedback["theta_AS"], 0.0, 1.0))
        if "theta_P" in feedback:
            self.global_state["theta_P"] = float(np.clip(feedback["theta_P"], 0.0, 1.0))
        if "business_events" in feedback:
            self.global_state["business_events"] = feedback.get("business_events") or []

        signal = str(feedback.get("signal", "")).lower()
        if signal in {"too_many_anomalies", "false_positive_high", "过度清洗"}:
            self.global_state["theta_AS"] = min(0.95, self.global_state["theta_AS"] + 0.05)
            self.global_state["theta_P"] = min(0.95, self.global_state["theta_P"] + 0.05)
        elif signal in {"missed_anomalies", "false_negative_high", "漏检"}:
            self.global_state["theta_AS"] = max(0.30, self.global_state["theta_AS"] - 0.05)
            self.global_state["theta_P"] = max(0.50, self.global_state["theta_P"] - 0.05)

        self.global_state["feedback"] = None

    def _zscore_regularize(self, data: pd.DataFrame) -> pd.DataFrame:
        """轻量规整：按序列执行Z-score并保留缺失插补。"""
        numeric_data = data.apply(pd.to_numeric, errors="coerce")
        filled = numeric_data.interpolate(axis=1, limit_direction="both")
        row_mean = filled.mean(axis=1)
        row_std = filled.std(axis=1).replace(0, 1).fillna(1)
        return filled.sub(row_mean, axis=0).div(row_std, axis=0).replace([np.inf, -np.inf], 0).fillna(0)

    def _detect_anomalies_with_isolation_forest(
        self, data: pd.DataFrame, theta_as: float
    ) -> List[Dict[str, Any]]:
        """使用Isolation Forest评分ASi，缺失值作为强疑似异常纳入候选。"""
        numeric_data = data.apply(pd.to_numeric, errors="coerce")
        arr = numeric_data.to_numpy(dtype=float)
        row_ids = numeric_data.index.to_list()
        col_ids = numeric_data.columns.to_list()
        missing_positions = np.argwhere(np.isnan(arr))

        suspected = []
        for row_pos, col_pos in missing_positions[:1000]:
            suspected.append({
                "candidate_id": f"{row_pos}:{col_pos}",
                "series_id": row_ids[row_pos],
                "time_id": col_ids[col_pos],
                "series_pos": int(row_pos),
                "time_pos": int(col_pos),
                "value": None,
                "AS": 1.0,
                "anomaly_type": "missing",
                "detector": "missing_value_rule"
            })

        finite_positions = np.argwhere(np.isfinite(arr))
        if len(finite_positions) == 0:
            return suspected

        rows = finite_positions[:, 0]
        cols = finite_positions[:, 1]
        values = arr[rows, cols]
        row_mean = np.nanmean(arr, axis=1)
        row_std = np.nanstd(arr, axis=1)
        row_std = np.where(row_std == 0, 1, row_std)
        col_mean = np.nanmean(arr, axis=0)
        col_std = np.nanstd(arr, axis=0)
        col_std = np.where(col_std == 0, 1, col_std)
        zero_flag = (values == 0).astype(float)

        features = np.column_stack([
            values,
            (values - row_mean[rows]) / row_std[rows],
            (values - col_mean[cols]) / col_std[cols],
            cols / max(arr.shape[1] - 1, 1),
            zero_flag
        ])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            from sklearn.ensemble import IsolationForest

            fit_features = features
            if len(features) > 50000:
                rng = np.random.default_rng(42)
                fit_idx = rng.choice(len(features), size=50000, replace=False)
                fit_features = features[fit_idx]

            detector = IsolationForest(
                n_estimators=120,
                contamination="auto",
                random_state=42,
                n_jobs=-1
            )
            detector.fit(fit_features)
            raw_scores = -detector.score_samples(features)
            score_min = float(np.min(raw_scores))
            score_max = float(np.max(raw_scores))
            anomaly_scores = (raw_scores - score_min) / (score_max - score_min + 1e-12)
            detector_name = "improved_isolation_forest"
        except Exception:
            robust_z = np.abs((values - row_mean[rows]) / row_std[rows])
            anomaly_scores = np.clip(robust_z / 6.0, 0.0, 1.0)
            detector_name = "robust_zscore_fallback"

        candidate_positions = np.where(anomaly_scores >= theta_as)[0]
        for idx in candidate_positions:
            row_pos = int(rows[idx])
            col_pos = int(cols[idx])
            value = float(values[idx])
            local_values = self._get_local_values(arr, row_pos, col_pos)
            local_median = float(np.nanmedian(local_values)) if len(local_values) else float(row_mean[row_pos])
            anomaly_type = "spike" if value >= local_median else "drop"

            suspected.append({
                "candidate_id": f"{row_pos}:{col_pos}",
                "series_id": row_ids[row_pos],
                "time_id": col_ids[col_pos],
                "series_pos": row_pos,
                "time_pos": col_pos,
                "value": value,
                "AS": round(float(anomaly_scores[idx]), 4),
                "anomaly_type": anomaly_type,
                "detector": detector_name
            })

        suspected.sort(key=lambda item: item["AS"], reverse=True)
        return suspected[:1000]

    def _semantic_confirm_anomalies(
        self, suspected_anomalies: List[Dict[str, Any]], data: pd.DataFrame, theta_p: float
    ) -> List[Dict[str, Any]]:
        """用LLM语义推理确认异常；LLM不可用时使用局部上下文启发式推理。"""
        if not suspected_anomalies:
            return []

        llm_confirmations = self._llm_confirm_anomaly_batch(suspected_anomalies[:20], data)
        arr = data.to_numpy(dtype=float)
        confirmed = []

        for anomaly in suspected_anomalies:
            candidate_id = anomaly["candidate_id"]
            llm_result = llm_confirmations.get(candidate_id)
            if llm_result:
                confidence = float(llm_result.get("Preason", 0.0))
                reasoning = llm_result.get("reasoning", "LLM语义确认")
            else:
                confidence, reasoning = self._heuristic_reasoning_confidence(anomaly, arr)

            fd = int(confidence > theta_p)
            enriched = dict(anomaly)
            enriched.update({
                "FDi": fd,
                "Preason": round(confidence, 4),
                "semantic_reasoning": reasoning
            })
            if fd:
                confirmed.append(enriched)

        confirmed.sort(key=lambda item: (item["Preason"], item["AS"]), reverse=True)
        return confirmed[:300]

    def _build_dmagent_semantic_confirmation_prompt(self, contexts: List[Dict[str, Any]]) -> str:
        """构造DMAgent语义异常确认提示词，固定角色、判定准则和JSON输出格式。"""
        return f"""
你是LCMA框架中的DMAgent（Anomaly-Aware Data Monitoring Agent），负责对备件需求序列执行“感知-推理-行动-观察”的闭环数据清洗。

【任务目标】
请对下列疑似异常点进行语义确认，判断统计异常是否属于真实数据异常，而不是正常业务波动。你的结论将用于后续动态校准修正，因此必须保守、可解释、结构化。

【输入字段说明】
- candidate_id：疑似异常点ID，输出时必须原样保留。
- series_id/time_id：备件需求序列与时间点标识。
- value：当前疑似点需求值，null表示缺失。
- AS：统计异常评分，越接近1表示统计层面越可疑。
- anomaly_type：统计检测阶段给出的初步类型，例如missing、spike、drop。
- local_context_pm5：疑似点前后5个时点的需求上下文。
- business_events_3_months：过去3个月相关业务事件。

【判定依据】
1. 时间上下文：比较疑似点前后5个时点，识别突刺、断崖式下降、孤立缺失、连续缺失或与局部趋势明显不一致的情况。
2. 业务语义：如果维修工单、生产计划变化、促销或检修日历能合理解释需求变化，应降低异常确认置信度。
3. 统计证据：AS只能作为辅助证据，不能单独决定FDi。
4. 数据质量影响：缺失值、重复记录、明显录入错误会破坏后续特征提取和预测，应优先确认为异常。

【判定规则】
- FDi=1：确认该点是真实数据异常，需要进入后续校准修正。
- FDi=0：该点更可能是正常业务波动、可解释需求变化，或证据不足。
- Preason必须是0到1之间的小数，表示语义推理置信度。
- 只有当Preason高于系统阈值theta_P时，系统才会最终确认异常；因此不要虚高打分。

【输出格式】
只输出严格JSON数组，不要输出Markdown、解释性段落或额外文本。数组中每个对象必须包含以下字段：
[
  {{
    "candidate_id": "候选点ID，必须与输入一致",
    "FDi": 0,
    "Preason": 0.0,
    "anomaly_type": "missing | spike | drop | duplicate | business_fluctuation | uncertain",
    "reasoning": "一句话说明判断依据，需同时提到统计证据、局部上下文或业务事件"
  }}
]

【疑似异常点上下文】
{json.dumps(contexts, ensure_ascii=False, default=str)}
"""

    def _llm_confirm_anomaly_batch(
        self, candidates: List[Dict[str, Any]], data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """批量调用LLM确认少量高置信疑似点，失败时返回空结果。"""
        if not candidates:
            return {}

        contexts = []
        for item in candidates:
            row_pos = item["series_pos"]
            col_pos = item["time_pos"]
            context_values = self._get_context_window(data, row_pos, col_pos, radius=5)
            contexts.append({
                "candidate_id": item["candidate_id"],
                "series_id": str(item["series_id"]),
                "time_id": str(item["time_id"]),
                "value": item["value"],
                "AS": item["AS"],
                "local_context_pm5": context_values,
                "local_context_±5": context_values,
                "business_events_3_months": self.global_state.get("business_events", [])
            })

        prompt = self._build_dmagent_semantic_confirmation_prompt(contexts)
        prompt = self._augment_prompt_with_rag(
            "dmagent",
            prompt,
            "异常检测 语义确认 备件需求 数据清洗 历史异常"
        )
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            start = content.find("[")
            end = content.rfind("]")
            if start == -1 or end == -1:
                return {}
            parsed = json.loads(content[start:end + 1])
            return {
                str(item["candidate_id"]): item
                for item in parsed
                if isinstance(item, dict) and "candidate_id" in item
            }
        except Exception:
            return {}

    def _heuristic_reasoning_confidence(
        self, anomaly: Dict[str, Any], arr: np.ndarray
    ) -> Tuple[float, str]:
        """LLM不可用时的语义确认替代：使用局部窗口、缺失状态和业务事件降噪。"""
        if anomaly.get("anomaly_type") == "missing":
            return 0.95, "缺失值破坏序列连续性，按数据质量规则确认异常"

        row_pos = anomaly["series_pos"]
        col_pos = anomaly["time_pos"]
        value = anomaly.get("value")
        local_values = self._get_local_values(arr, row_pos, col_pos)
        if len(local_values) == 0 or value is None:
            return 0.72, "局部上下文不足，依据统计异常评分进行确认"

        median = float(np.nanmedian(local_values))
        mad = float(np.nanmedian(np.abs(local_values - median))) or 1.0
        local_deviation = abs(float(value) - median) / (1.4826 * mad + 1e-9)
        confidence = min(0.98, 0.45 + 0.08 * local_deviation + 0.35 * anomaly.get("AS", 0))

        business_events = self.global_state.get("business_events", [])
        if business_events:
            confidence = max(0.0, confidence - 0.15)
            reason = "统计上偏离局部上下文，但存在业务事件，降低异常确认置信度"
        else:
            reason = "需求值显著偏离前后5期局部上下文，且未发现业务事件解释"

        return float(confidence), reason

    def _get_local_values(self, arr: np.ndarray, row_pos: int, col_pos: int, radius: int = 5) -> np.ndarray:
        start = max(0, col_pos - radius)
        end = min(arr.shape[1], col_pos + radius + 1)
        values = np.delete(arr[row_pos, start:end], col_pos - start)
        return values[np.isfinite(values)]

    def _get_context_window(
        self, data: pd.DataFrame, row_pos: int, col_pos: int, radius: int = 5
    ) -> List[Dict[str, Any]]:
        start = max(0, col_pos - radius)
        end = min(data.shape[1], col_pos + radius + 1)
        series = data.iloc[row_pos, start:end]
        return [
            {"time_id": str(time_id), "value": None if pd.isna(value) else float(value)}
            for time_id, value in series.items()
        ]

    def _select_correction_strategy(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """按0.6*有效性+0.4*效率选择最优修正工具。"""
        strategy_library = {
            "linear_interpolation": {"effectiveness": 0.78, "efficiency": 0.95},
            "stl_seasonal_trend_interpolation": {"effectiveness": 0.88, "efficiency": 0.55},
            "similar_spare_parts_mean": {"effectiveness": 0.72, "efficiency": 0.80}
        }

        anomaly_type = anomaly.get("anomaly_type")
        if anomaly_type == "missing":
            strategy_library["linear_interpolation"]["effectiveness"] += 0.05
        elif anomaly_type in {"spike", "drop"}:
            strategy_library["stl_seasonal_trend_interpolation"]["effectiveness"] += 0.03
            strategy_library["linear_interpolation"]["effectiveness"] += 0.02

        best_name = None
        best_score = -1.0
        for name, score in strategy_library.items():
            utility = 0.6 * score["effectiveness"] + 0.4 * score["efficiency"]
            if utility > best_score:
                best_name = name
                best_score = utility

        return {"name": best_name, "utility": best_score, **strategy_library[best_name]}

    def _execute_correction_strategy(
        self, data: pd.DataFrame, anomaly: Dict[str, Any], strategy: Dict[str, Any]
    ) -> float:
        series_id = anomaly["series_id"]
        time_id = anomaly["time_id"]
        row = data.loc[series_id].astype(float).copy()
        time_pos = anomaly["time_pos"]
        strategy_name = strategy["name"]

        if strategy_name == "linear_interpolation":
            row.iloc[time_pos] = np.nan
            corrected = row.interpolate(limit_direction="both").iloc[time_pos]
        elif strategy_name == "stl_seasonal_trend_interpolation":
            corrected = self._stl_interpolate_point(row, time_pos)
        else:
            column_values = data[time_id].drop(index=series_id, errors="ignore")
            corrected = pd.to_numeric(column_values, errors="coerce").mean()
            if pd.isna(corrected):
                row.iloc[time_pos] = np.nan
                corrected = row.interpolate(limit_direction="both").iloc[time_pos]

        if pd.isna(corrected):
            corrected = 0.0 if pd.isna(row.mean()) else row.mean()
        return float(max(corrected, 0.0))

    def _stl_interpolate_point(self, row: pd.Series, time_pos: int) -> float:
        try:
            from statsmodels.tsa.seasonal import STL

            working = row.astype(float).copy()
            working.iloc[time_pos] = np.nan
            filled = working.interpolate(limit_direction="both").fillna(working.mean())
            period = 12 if len(filled) >= 24 else max(2, min(6, len(filled) // 2))
            stl_result = STL(filled, period=period, robust=True).fit()
            corrected = stl_result.trend.iloc[time_pos] + stl_result.seasonal.iloc[time_pos]
            return float(corrected)
        except Exception:
            working = row.astype(float).copy()
            working.iloc[time_pos] = np.nan
            return float(working.interpolate(limit_direction="both").iloc[time_pos])

    def _coordinate_agent_feedback(self) -> Dict[str, Any]:
        """建立智能体间反馈通路，并将优化计划写入Sglobal。"""
        feedback_bundle = self._extract_feedback_bundle()
        feedback_result = {
            "source": "ForecastingModelSelectAgent.analyze_evaluation",
            "routes": {},
            "pending_actions": [],
            "applied_runtime_updates": []
        }

        # 1. 模型选择智能体自反馈：根据评估结果生成算法调整建议
        forecast_feedback = feedback_bundle.get("forecast_model_select", {})
        if forecast_feedback:
            try:
                if self.use_forecast_model:
                    forecast_plan = self.forecasting_agent.process_feedback(forecast_feedback)
                else:
                    forecast_plan = self._fallback_forecast_feedback_plan(forecast_feedback)
                feedback_result["routes"]["ForecastingModelSelectAgent"] = {
                    "feedback": forecast_feedback,
                    "optimization_plan": forecast_plan
                }
                feedback_result["pending_actions"].append("下一轮模型推荐/评估时使用预测算法反馈和已优化参数")
            except Exception as e:
                feedback_result["routes"]["ForecastingModelSelectAgent"] = {
                    "feedback": forecast_feedback,
                    "error": str(e)
                }

        # 2. 反馈给特征提取智能体
        feature_feedback = feedback_bundle.get("feature_extraction", {})
        if feature_feedback:
            try:
                if self.use_forecast_model and isinstance(self.analyzed_data, dict):
                    feature_plan = self.feature_agent.process_feedback(feature_feedback, self.analyzed_data)
                    can_apply = bool(feature_plan.get("needs_optimization")) and self._is_feature_plan_executable(feature_plan)
                    if can_apply:
                        optimized_features = self.feature_agent.apply_optimization(feature_plan)
                        feedback_result["applied_runtime_updates"].append("FeatureExtractionAgent已生成优化特征候选")
                        feature_plan["runtime_result"] = self._summarize_feedback_payload(optimized_features)
                    else:
                        feedback_result["pending_actions"].append("下一轮特征提取时应用特征调整建议")
                else:
                    feature_plan = self._fallback_feature_feedback_plan(feature_feedback)
                feedback_result["routes"]["FeatureExtractionAgent"] = {
                    "feedback": feature_feedback,
                    "optimization_plan": feature_plan
                }
            except Exception as e:
                feedback_result["routes"]["FeatureExtractionAgent"] = {
                    "feedback": feature_feedback,
                    "error": str(e)
                }

        # 3. 反馈给分类聚类智能体
        classification_feedback = feedback_bundle.get("classification", {})
        if classification_feedback:
            try:
                if self.use_forecast_model:
                    classification_plan = self.classification_agent.process_feedback(classification_feedback)
                else:
                    classification_plan = self._fallback_classification_feedback_plan(classification_feedback)
                feedback_result["routes"]["ClassificationAgent"] = {
                    "feedback": classification_feedback,
                    "optimization_plan": classification_plan
                }
                feedback_result["pending_actions"].append("下一轮聚类策略设计/执行时应用分类反馈")
            except Exception as e:
                feedback_result["routes"]["ClassificationAgent"] = {
                    "feedback": classification_feedback,
                    "error": str(e)
                }

        # 4. 反馈给DMAgent，调整数据质量和异常检测阈值
        dm_feedback = feedback_bundle.get("dmagent", {})
        if dm_feedback:
            self.update_monitoring_feedback(dm_feedback)
            feedback_result["routes"]["DMAgent"] = {
                "feedback": dm_feedback,
                "target": "Sglobal.feedback"
            }
            feedback_result["pending_actions"].append("下一轮质量扫描时自适应调整DMAgent阈值")

        self.agent_feedback_state = feedback_result
        self.global_state["agent_feedback"] = feedback_result
        self.global_state["pending_feedback_actions"] = feedback_result["pending_actions"]
        self.feedback_history.append(feedback_result)
        return feedback_result

    def _extract_feedback_bundle(self) -> Dict[str, Any]:
        """从评估分析和运行指标中抽取发往各智能体的反馈。"""
        bundle = {
            "feature_extraction": {},
            "classification": {},
            "forecast_model_select": {},
            "dmagent": {}
        }

        if isinstance(self.analysis_results, dict):
            for demand_type, analysis in self.analysis_results.items():
                if not isinstance(analysis, dict):
                    continue
                if "feature_extraction_feedback" in analysis:
                    bundle["feature_extraction"][demand_type] = analysis["feature_extraction_feedback"]
                if "classification_feedback" in analysis:
                    bundle["classification"][demand_type] = {
                        "classification_feedback": analysis["classification_feedback"]
                    }
                if "forecast_algorithms" in analysis:
                    bundle["forecast_model_select"][demand_type] = {
                        "forecast_algorithms": analysis["forecast_algorithms"]
                    }

        # 当LLM评估分析缺少结构化字段时，基于评估指标生成保守反馈。
        metric_feedback = self._derive_metric_feedback()
        if metric_feedback:
            bundle["forecast_model_select"].setdefault("metric_feedback", metric_feedback)
            if metric_feedback.get("needs_feature_review"):
                bundle["feature_extraction"].setdefault("metric_feedback", {
                    "feature_adjustments": [
                        {
                            "feature": "intermittency_and_recent_demand_features",
                            "adjustment": True,
                            "reason": "预测误差偏高，需要强化间歇性和近期需求特征"
                        }
                    ]
                })
            if metric_feedback.get("needs_cluster_review"):
                bundle["classification"].setdefault("metric_feedback", {
                    "classification_feedback": {
                        "clustering_assessment": "预测误差在部分类别中偏高，建议复核聚类边界",
                        "cluster_adjustments": "检查类别数量和DBSCAN/KMeans参数"
                    }
                })

        bundle["dmagent"] = self._derive_dmagent_feedback(metric_feedback)
        return {key: value for key, value in bundle.items() if value}

    def _derive_metric_feedback(self) -> Dict[str, Any]:
        """根据模型评估指标生成跨智能体反馈信号。"""
        if not isinstance(self.evaluation_results, dict):
            return {}

        cluster_metrics = []
        for demand_type, demand_result in self.evaluation_results.items():
            for cluster in demand_result.get("clusters", []):
                metrics = cluster.get("metrics", {})
                if "mae" in metrics or "rmse" in metrics:
                    cluster_metrics.append({
                        "demand_type": demand_type,
                        "cluster_id": cluster.get("cluster_id"),
                        "algorithm": cluster.get("algorithm"),
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "rrmse": metrics.get("rrmse"),
                        "mase": metrics.get("mase"),
                        "relmae": metrics.get("relmae"),
                        "smape": metrics.get("smape"),
                        "nrmse": metrics.get("nrmse"),
                        "optimized_parameters": cluster.get("optimized_parameters", {})
                    })

        if not cluster_metrics:
            return {}

        mean_metrics = {}
        for metric_name in ["mae", "rmse", "rrmse", "mase", "relmae", "smape", "nrmse"]:
            values = [
                item.get(metric_name) for item in cluster_metrics
                if item.get(metric_name) is not None
            ]
            mean_metrics[f"mean_{metric_name}"] = float(np.mean(values)) if values else None

        mean_mae = mean_metrics.get("mean_mae")
        high_error_clusters = [
            item for item in cluster_metrics
            if item.get("mae") is not None and mean_mae is not None and item["mae"] > mean_mae * 1.25
        ]

        return {
            **mean_metrics,
            "high_error_clusters": high_error_clusters,
            "needs_feature_review": len(high_error_clusters) > 0,
            "needs_cluster_review": len(high_error_clusters) >= max(1, len(cluster_metrics) // 3),
            "recommendation": "针对高误差类别回传特征与聚类反馈，并保留已搜索到的模型参数"
        }

    def _derive_dmagent_feedback(self, metric_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """把数据质量和预测误差信号转为DMAgent阈值反馈。"""
        if not self.data_quality_report:
            return {}

        confirmed_count = 0
        if isinstance(self.anomaly_detection_results, dict):
            confirmed_count = self.anomaly_detection_results.get("confirmed_count", 0)

        total_points = max(self.data_quality_report.get("total_points", 1), 1)
        confirmed_ratio = confirmed_count / total_points
        feedback = {}

        if confirmed_ratio > 0.05:
            feedback["signal"] = "too_many_anomalies"
        elif metric_feedback and metric_feedback.get("needs_feature_review"):
            feedback["signal"] = "missed_anomalies"

        if feedback:
            feedback.update({
                "reason": "由预测评估和异常处理结果回传至DMAgent",
                "quality_score": self.data_quality_report.get("quality_score"),
                "confirmed_anomaly_ratio": confirmed_ratio
            })
        return feedback

    def _is_feature_plan_executable(self, feature_plan: Dict[str, Any]) -> bool:
        """只对已有安全实现的特征计划立即执行，其他计划排入下一轮。"""
        adjustments = feature_plan.get("feature_adjustments", {})
        if not isinstance(adjustments, dict):
            return False
        for adjustment in adjustments.values():
            for new_feature in adjustment.get("new_features", []):
                if new_feature.get("calculation") not in {"moving_average", "standard_deviation"}:
                    return False
        return True

    def _summarize_feedback_payload(self, payload: Any) -> Any:
        """压缩反馈结果，避免把大型DataFrame放进对话历史。"""
        if isinstance(payload, dict):
            summary = {}
            for key, value in payload.items():
                if isinstance(value, pd.DataFrame):
                    summary[key] = {"type": "DataFrame", "shape": value.shape}
                elif isinstance(value, dict):
                    summary[key] = self._summarize_feedback_payload(value)
                else:
                    summary[key] = value
            return summary
        return payload

    def _fallback_feature_feedback_plan(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "needs_optimization": bool(feedback),
            "feature_adjustments": feedback,
            "source": "fallback_feedback_router"
        }

    def _fallback_classification_feedback_plan(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "cluster_adjustments": feedback,
            "source": "fallback_feedback_router"
        }

    def _fallback_forecast_feedback_plan(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "algorithm_changes": feedback,
            "source": "fallback_feedback_router"
        }

    def _build_forecast_graph(self):
        """构建LangGraph智能体编排图。"""
        if StateGraph is None:
            print("⚠️ 未安装langgraph，当前将使用顺序执行兼容模式")
            return None

        graph_builder = StateGraph(ForecastGraphState)

        graph_builder.add_node("understand_request", self._graph_understand_request)
        graph_builder.add_node("decompose_task", self._graph_decompose_task)
        graph_builder.add_node("validate_workflow", self._graph_validate_workflow)
        graph_builder.add_node("finalize_response", self._graph_finalize_response)

        for tool_name, node_name in TOOL_GRAPH_NODES.items():
            graph_builder.add_node(node_name, self._make_tool_node(tool_name))

        route_map = {
            "finalize_response": "finalize_response",
            **{tool_name: node_name for tool_name, node_name in TOOL_GRAPH_NODES.items()}
        }

        graph_builder.add_edge(START, "understand_request")
        graph_builder.add_edge("understand_request", "decompose_task")
        graph_builder.add_edge("decompose_task", "validate_workflow")
        graph_builder.add_conditional_edges(
            "validate_workflow",
            self._route_after_validation,
            route_map
        )

        for node_name in TOOL_GRAPH_NODES.values():
            graph_builder.add_conditional_edges(
                node_name,
                self._route_after_tool,
                route_map
            )

        graph_builder.add_edge("finalize_response", END)
        return graph_builder.compile()

    def _graph_understand_request(self, state: ForecastGraphState) -> Dict[str, Any]:
        """需求理解节点：由LLM生成面向用户的专业解释。"""
        user_input = state["user_input"]
        print(f"\n🤖 用户需求: {user_input}")
        print("=" * 60)
        print("📋 LangGraph节点1: 需求理解...")

        try:
            rag_context = self._retrieve_rag_context("feedback_coordinator", user_input)
            assistant_response = self.assistant_chain.invoke({
                "user_input": user_input,
                "rag_context": rag_context
            })
            print(f"💭 助手分析: {assistant_response[:200]}...")
        except Exception as e:
            assistant_response = f"需求理解阶段LLM调用失败，已切换到规则任务分解: {e}"
            print(f"⚠️ {assistant_response}")

        return {
            "assistant_response": assistant_response,
            "graph_trace": ["understand_request"]
        }

    def _graph_decompose_task(self, state: ForecastGraphState) -> Dict[str, Any]:
        """任务分解节点：决定本轮需要经过哪些智能体节点。"""
        user_input = state["user_input"]
        print("\n📋 LangGraph节点2: 任务分解和智能体路由...")

        query_lower = user_input.lower()
        clustering_keywords = ["聚类分析", "聚类", "cluster", "clustering", "进行聚类分析", "需求聚类分析"]

        if any(keyword in query_lower for keyword in clustering_keywords):
            print("🎯 检测到聚类分析需求，使用严格边界控制...")
            task_plan = self._fallback_task_analysis(user_input)
        else:
            try:
                task_plan = self.task_decomposer.invoke({
                    "user_input": user_input,
                    "rag_context": self._retrieve_rag_context("feedback_coordinator", user_input)
                })
            except Exception as e:
                print(f"❌ 任务分解失败，切换到关键词规则: {e}")
                task_plan = self._fallback_task_analysis(user_input)

        print(f"🔍 任务分析: {task_plan.get('analysis', '无分析')}")
        print(f"🛠️ 需要的工具: {task_plan.get('required_tools', [])}")
        print(f"💡 选择原因: {task_plan.get('reasoning', '无说明')}")

        return {
            "task_plan": task_plan,
            "graph_trace": state.get("graph_trace", []) + ["decompose_task"]
        }

    def _graph_validate_workflow(self, state: ForecastGraphState) -> Dict[str, Any]:
        """工作流校验节点：补齐依赖顺序并跳过已完成节点。"""
        print("\n📋 LangGraph节点3: 工作流验证...")
        task_plan = state.get("task_plan", {})
        validation = self._validate_workflow_order(task_plan.get("required_tools", []))

        if not validation["valid"]:
            print(f"❌ 工作流验证失败: {validation.get('error')}")
            return {
                "validation": validation,
                "success": False,
                "error": validation.get("error", "工作流验证失败"),
                "graph_trace": state.get("graph_trace", []) + ["validate_workflow"]
            }

        tools_to_execute = validation["tools_to_execute"]
        completed_steps = self._get_completed_steps()
        remaining_tools = [tool for tool in tools_to_execute if tool not in completed_steps]

        print(f"✅ 已完成步骤: {completed_steps}")
        print(f"🔄 待执行步骤: {remaining_tools}")

        return {
            "validation": validation,
            "tools_to_execute": tools_to_execute,
            "completed_steps": completed_steps,
            "remaining_tools": remaining_tools,
            "execution_results": {},
            "graph_trace": state.get("graph_trace", []) + ["validate_workflow"]
        }

    def _make_tool_node(self, tool_name: str):
        """把现有LangChain工具包装成LangGraph节点。"""
        def run_tool(state: ForecastGraphState) -> Dict[str, Any]:
            agent_label = TOOL_AGENT_LABELS.get(tool_name, "工具节点")
            print(f"   🔄 {agent_label} 执行 {tool_name}...")

            result = self.tool_executor(tool_name)
            execution_results = dict(state.get("execution_results", {}))
            execution_results[tool_name] = result

            remaining_tools = [
                name for name in state.get("remaining_tools", [])
                if name != tool_name
            ]
            completed_steps = list(state.get("completed_steps", []))

            if result.get("success"):
                print(f"   ✅ {tool_name} 完成")
                if tool_name not in completed_steps:
                    completed_steps.append(tool_name)
            else:
                print(f"   ❌ {tool_name} 失败: {result.get('error')}")

            return {
                "execution_results": execution_results,
                "remaining_tools": remaining_tools,
                "completed_steps": completed_steps,
                "last_tool": tool_name,
                "last_tool_success": bool(result.get("success")),
                "graph_trace": state.get("graph_trace", []) + [TOOL_GRAPH_NODES[tool_name]]
            }

        return run_tool

    def _route_after_validation(self, state: ForecastGraphState) -> str:
        """校验节点之后选择第一个待执行工具，或直接收尾。"""
        validation = state.get("validation", {})
        if not validation.get("valid"):
            return "finalize_response"

        remaining_tools = state.get("remaining_tools", [])
        if not remaining_tools:
            return "finalize_response"

        return remaining_tools[0]

    def _route_after_tool(self, state: ForecastGraphState) -> str:
        """工具节点之后继续下一个智能体节点，失败则进入收尾。"""
        if not state.get("last_tool_success", True):
            return "finalize_response"

        remaining_tools = state.get("remaining_tools", [])
        if not remaining_tools:
            return "finalize_response"

        return remaining_tools[0]

    def _graph_finalize_response(self, state: ForecastGraphState) -> Dict[str, Any]:
        """收尾节点：生成评估、记录历史并返回统一结果。"""
        print("\n📋 LangGraph收尾节点: 生成执行评估和报告...")

        user_input = state.get("user_input", "")
        assistant_response = state.get("assistant_response", "")
        task_plan = state.get("task_plan", {})
        validation = state.get("validation", {})
        execution_results = state.get("execution_results", {})

        if not validation.get("valid", True):
            evaluation = {
                "task_completion": 0,
                "successful_tools": [],
                "failed_tools": [],
                "key_insights": [],
                "recommendations": [validation.get("error", "工作流验证失败")],
                "overall_assessment": "需要改进"
            }
            result_success = False
            error = validation.get("error", "工作流验证失败")
        elif not execution_results:
            completed_target_steps = [
                tool for tool in state.get("tools_to_execute", [])
                if tool in self._get_completed_steps()
            ]
            evaluation = {
                "task_completion": 1.0,
                "successful_tools": completed_target_steps,
                "failed_tools": [],
                "key_insights": ["目标流程节点此前已完成，本轮无需重复执行"],
                "recommendations": [],
                "overall_assessment": "优秀"
            }
            result_success = True
            error = ""
        else:
            evaluation = self._generate_execution_evaluation(
                user_input, task_plan, execution_results, assistant_response
            )
            result_success = len(evaluation.get("failed_tools", [])) == 0
            error = ""

        completed_steps = self._get_completed_steps()
        graph_trace = state.get("graph_trace", []) + ["finalize_response"]

        self.conversation_history.append({
            "user_input": user_input,
            "assistant_response": assistant_response,
            "task_plan": task_plan,
            "execution_results": execution_results,
            "evaluation": evaluation,
            "agent_feedback": self.agent_feedback_state,
            "graph_trace": graph_trace
        })
        self._store_history_record(
            user_input,
            task_plan,
            execution_results,
            evaluation,
            self.agent_feedback_state
        )

        print(f"\n✅ LangGraph任务完成! 本轮执行了 {len(execution_results)} 个步骤")

        return {
            "success": result_success,
            "error": error,
            "evaluation": evaluation,
            "agent_feedback": self.agent_feedback_state,
            "completed_steps": completed_steps,
            "graph_trace": graph_trace,
            "message": "LangGraph智能体工作流执行完成"
        }

    def get_graph_structure(self) -> Dict[str, Any]:
        """返回当前LangGraph智能体拓扑，便于调试和展示架构。"""
        return {
            "architecture": "LangGraph StateGraph",
            "available": self.forecast_graph is not None,
            "rag": {
                "enabled": self.rag_enabled,
                "vector_database": "ChromaDB",
                "embedding_model": self.embedding_model,
                "retrieval": "semantic + keyword hybrid search",
                "history_store": "knowledge_base/history",
                "prompt_augmentation": "agent knowledge + interaction history"
            },
            "entry": "understand_request",
            "planner_nodes": ["understand_request", "decompose_task", "validate_workflow"],
            "agent_links": [
                {
                    "agent": "DMAgent",
                    "nodes": [
                        "quality_scanning_evaluation", "detect_anomalies_statistical",
                        "dynamic_calibration_correction", "normalize_clean_data"
                    ],
                    "depends_on": ["load_data"],
                    "output": "Dclean"
                },
                {
                    "agent": "FeatureExtractionAgent",
                    "nodes": ["analyze_data", "explain_features"],
                    "depends_on": ["DMAgent"]
                },
                {
                    "agent": "ClassificationAgent",
                    "nodes": ["design_clustering_strategy", "perform_clustering", "analyze_clusters"],
                    "depends_on": ["FeatureExtractionAgent"]
                },
                {
                    "agent": "ForecastingModelSelectAgent",
                    "nodes": [
                        "recommend_algorithms", "evaluate_algorithms",
                        "analyze_evaluation"
                    ],
                    "depends_on": ["ClassificationAgent"]
                },
                {
                    "agent": "FeedbackCoordinator",
                    "nodes": ["coordinate_agent_feedback"],
                    "depends_on": ["ForecastingModelSelectAgent"],
                    "feedback_loops": [
                        "ForecastingModelSelectAgent -> FeatureExtractionAgent",
                        "ForecastingModelSelectAgent -> ClassificationAgent",
                        "ForecastingModelSelectAgent -> DMAgent",
                        "Evaluation metrics -> ForecastingModelSelectAgent"
                    ]
                },
                {
                    "agent": "ForecastingModelSelectAgent",
                    "nodes": ["generate_prediction_strategy"],
                    "depends_on": ["FeedbackCoordinator"]
                }
            ],
            "tool_order": WORKFLOW_TOOL_ORDER,
            "exit": "finalize_response"
        }
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """主要对话接口 - 通过LangGraph编排多个智能体。"""
        if self.forecast_graph is None:
            print("⚠️ LangGraph不可用，切换到顺序执行兼容模式")
            return self._chat_sequential(user_input)

        try:
            final_state = self.forecast_graph.invoke({"user_input": user_input})
            return {
                "success": final_state.get("success", False),
                "user_input": user_input,
                "assistant_response": final_state.get("assistant_response", ""),
                "task_plan": final_state.get("task_plan", {}),
                "execution_results": final_state.get("execution_results", {}),
                "evaluation": final_state.get("evaluation", {}),
                "agent_feedback": self.agent_feedback_state,
                "completed_steps": final_state.get("completed_steps", []),
                "graph_trace": final_state.get("graph_trace", []),
                "message": final_state.get("message", ""),
                "error": final_state.get("error", "")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "LangGraph智能体工作流处理失败"
            }

    def _chat_sequential(self, user_input: str) -> Dict[str, Any]:
        """主要对话接口 - 实现完整的智能交互流程"""
        
        try:
            print(f"\n🤖 用户需求: {user_input}")
            print("=" * 60)
            
            # 1. 用户提问 → 大模型思考(RAG)
            print("📋 步骤1: 理解用户需求...")
            rag_context = self._retrieve_rag_context("feedback_coordinator", user_input)
            assistant_response = self.assistant_chain.invoke({
                "user_input": user_input,
                "rag_context": rag_context
            })
            print(f"💭 助手分析: {assistant_response[:200]}...")
            
            # 2. 问题分解 → 确定需要调用的工具
            print("\n📋 步骤2: 分解任务和工具选择...")
            
            # 对于聚类分析相关需求，直接使用fallback方案确保边界控制
            query_lower = user_input.lower()
            clustering_keywords = ["聚类分析", "聚类", "cluster", "clustering", "进行聚类分析", "需求聚类分析"]
            
            if any(keyword in query_lower for keyword in clustering_keywords):
                print("🎯 检测到聚类分析需求，使用严格边界控制...")
                task_plan = self._fallback_task_analysis(user_input)
                print(f"🔍 任务分析: {task_plan.get('analysis', '无分析')}")
                print(f"🛠️ 需要的工具: {task_plan.get('required_tools', [])}")
                print(f"💡 选择原因: {task_plan.get('reasoning', '无说明')}")
            else:
                try:
                    task_plan = self.task_decomposer.invoke({
                        "user_input": user_input,
                        "rag_context": self._retrieve_rag_context("feedback_coordinator", user_input)
                    })
                    print(f"🔍 任务分析: {task_plan.get('analysis', '无分析')}")
                    print(f"🛠️ 需要的工具: {task_plan.get('required_tools', [])}")
                    print(f"💡 选择原因: {task_plan.get('reasoning', '无说明')}")
                except Exception as e:
                    print(f"❌ 任务分解失败: {e}")
                    # 回退到简单的关键词匹配
                    task_plan = self._fallback_task_analysis(user_input)
            
            # 3. 工具确认 → 验证工作流顺序
            print("\n📋 步骤3: 工具确认和工作流验证...")
            validation = self._validate_workflow_order(task_plan.get('required_tools', []))
            
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation.get("error", "工作流验证失败"),
                    "assistant_response": assistant_response,
                    "task_plan": task_plan
                }
            
            tools_to_execute = validation["tools_to_execute"]
            completed_steps = self._get_completed_steps()
            
            # 只执行尚未完成的步骤
            remaining_tools = [tool for tool in tools_to_execute if tool not in completed_steps]
            
            print(f"✅ 已完成步骤: {completed_steps}")
            print(f"🔄 待执行步骤: {remaining_tools}")
            
            # 4. 工具调用 → 执行分析任务
            print("\n📋 步骤4: 执行分析任务...")
            execution_results = {}
            
            for tool_name in remaining_tools:
                print(f"   🔄 执行 {tool_name}...")
                result = self._create_tool_executor()(tool_name)
                execution_results[tool_name] = result
                
                if not result.get("success"):
                    print(f"   ❌ {tool_name} 失败: {result.get('error')}")
                    break
                else:
                    print(f"   ✅ {tool_name} 完成")
            
            # 5. 执行评估 → 生成最终报告
            print("\n📋 步骤5: 生成执行评估和报告...")
            evaluation = self._generate_execution_evaluation(
                user_input, task_plan, execution_results, assistant_response
            )
            
            # 更新对话历史
            self.conversation_history.append({
                "user_input": user_input,
                "assistant_response": assistant_response,
                "task_plan": task_plan,
                "execution_results": execution_results,
                "evaluation": evaluation
            })
            self._store_history_record(
                user_input,
                task_plan,
                execution_results,
                evaluation,
                self.agent_feedback_state
            )
            
            print(f"\n✅ 任务完成! 执行了 {len(remaining_tools)} 个步骤")
            
            return {
                "success": True,
                "user_input": user_input,
                "assistant_response": assistant_response,
                "task_plan": task_plan,
                "execution_results": execution_results,
                "evaluation": evaluation,
                "agent_feedback": self.agent_feedback_state,
                "completed_steps": completed_steps + remaining_tools
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "智能对话处理失败"
            }
    
    def _fallback_task_analysis(self, user_input: str) -> Dict[str, Any]:
        """回退的任务分析方法 - 基于关键词匹配确定工具需求"""
        query_lower = user_input.lower()
        dmagent_tools = [
            "load_data", "quality_scanning_evaluation", "detect_anomalies_statistical",
            "dynamic_calibration_correction", "normalize_clean_data"
        ]
        
        # 更精确的关键词映射 - 按优先级排序
        keyword_patterns = [
            # 完整预测流程
            (["完整", "预测策略", "全流程", "端到端", "完整方案", "反馈", "闭环", "智能体反馈", "反馈机制"], 
             dmagent_tools + ["analyze_data", "explain_features", "design_clustering_strategy", 
              "perform_clustering", "analyze_clusters", "recommend_algorithms", 
              "evaluate_algorithms", "analyze_evaluation", "coordinate_agent_feedback",
              "generate_prediction_strategy"]),
            
            # 算法相关
            (["算法推荐", "模型推荐", "推荐算法", "选择算法", "推荐合适的算法", "推荐"], 
             dmagent_tools + ["analyze_data", "explain_features", "design_clustering_strategy", 
              "perform_clustering", "analyze_clusters", "recommend_algorithms"]),
            
            (["算法评估", "模型评估", "评估算法", "算法效果"], 
             dmagent_tools + ["analyze_data", "explain_features", "design_clustering_strategy", 
              "perform_clustering", "analyze_clusters", "recommend_algorithms", "evaluate_algorithms"]),
            
            # 聚类分析
            (["聚类分析", "聚类", "分类分析", "cluster", "clustering", "进行聚类分析", "需求聚类分析"], 
             dmagent_tools + ["analyze_data", "explain_features", "design_clustering_strategy", 
              "perform_clustering", "analyze_clusters"]),
            
            (["聚类策略", "聚类方法", "聚类设计"], 
             dmagent_tools + ["analyze_data", "explain_features", "design_clustering_strategy"]),
            
            # 特征分析
            (["特征分析", "特征解释", "特征工程", "分析特征", "feature"], 
             dmagent_tools + ["analyze_data", "explain_features"]),
            
            # 数据分析
            (["数据分析", "分析数据", "analyze"], 
             dmagent_tools + ["analyze_data"]),

            # DMAgent数据监控与清洗
            (["异常检测", "异常监测", "数据清洗", "质量扫描", "数据质量", "闭环清洗", "anomaly", "clean", "monitor"],
             dmagent_tools),
            
            # 数据加载
            (["加载数据", "读取数据", "导入数据", "load", "数据"], 
             ["load_data"]),
        ]
        
        # 按优先级匹配
        for keywords, tools in keyword_patterns:
            for keyword in keywords:
                if keyword in query_lower:
                    return {
                        "analysis": f"关键词匹配'{keyword}' → 确定用户需求为{self._get_task_type(tools)}",
                        "required_tools": tools,
                        "reasoning": f"基于关键词'{keyword}'匹配到对应的工具序列"
                    }
        
        # 默认情况
        return {
            "analysis": "未能明确识别用户需求，使用最小工具集",
            "required_tools": ["load_data"],
            "reasoning": "默认至少执行数据加载步骤"
        }
    
    def _get_task_type(self, tools: List[str]) -> str:
        """根据工具列表确定任务类型"""
        if len(tools) == 1:
            return "数据加载"
        elif len(tools) == 5:
            return "DMAgent数据监控与闭环清洗"
        elif len(tools) == 6:
            return "数据分析"
        elif len(tools) == 7:
            return "特征分析"
        elif len(tools) == 10:
            return "聚类分析"
        elif len(tools) == 11:
            return "算法推荐"
        elif len(tools) == 12:
            return "算法评估"
        elif len(tools) == 14:
            return "反馈协调"
        elif len(tools) == 15:
            return "完整预测流程"
        else:
            return f"部分流程({len(tools)}步)"
    
    def _generate_execution_evaluation(self, user_input: str, task_plan: Dict, 
                                     execution_results: Dict, assistant_response: str) -> Dict[str, Any]:
        """生成执行评估报告"""
        
        successful_tools = [name for name, result in execution_results.items() if result.get("success")]
        failed_tools = [name for name, result in execution_results.items() if not result.get("success")]
        
        evaluation = {
            "task_completion": len(successful_tools) / len(execution_results) if execution_results else 0,
            "successful_tools": successful_tools,
            "failed_tools": failed_tools,
            "key_insights": [],
            "recommendations": []
        }
        
        # 基于执行结果生成关键洞察
        if "load_data" in successful_tools:
            data_result = execution_results["load_data"]
            evaluation["key_insights"].append(
                f"数据加载成功: {data_result.get('series_count', 0)}个时间序列, "
                f"{data_result.get('time_periods', 0)}个时间期间"
            )

        if "quality_scanning_evaluation" in successful_tools:
            quality_result = execution_results["quality_scanning_evaluation"].get("quality_report", {})
            evaluation["key_insights"].append(
                f"DMAgent质量扫描完成: QS={quality_result.get('quality_score')}, "
                f"缺失率={quality_result.get('missing_ratio')}, 零值率={quality_result.get('zero_ratio')}"
            )

        if "detect_anomalies_statistical" in successful_tools:
            detection = execution_results["detect_anomalies_statistical"].get("detection", {})
            evaluation["key_insights"].append(
                f"异常监测完成: 疑似{detection.get('suspected_count', 0)}个, "
                f"确认{detection.get('confirmed_count', 0)}个"
            )

        if "normalize_clean_data" in successful_tools:
            evaluation["key_insights"].append("DMAgent已输出Dclean，并将Min-Max反归一化参数保存到Sglobal")
        
        if "analyze_clusters" in successful_tools:
            evaluation["key_insights"].append("完成聚类分析，识别了不同的需求模式")
        
        if "recommend_algorithms" in successful_tools:
            evaluation["key_insights"].append("完成算法推荐，提供了预测模型建议")

        if "coordinate_agent_feedback" in successful_tools:
            feedback = execution_results["coordinate_agent_feedback"].get("feedback", {})
            routes = feedback.get("routes", {})
            evaluation["key_insights"].append(
                f"完成智能体反馈协调: 已建立{len(routes)}条反馈路线"
            )
            pending_actions = feedback.get("pending_actions", [])
            if pending_actions:
                evaluation["recommendations"].extend(pending_actions[:3])
        
        # 生成建议
        if failed_tools:
            evaluation["recommendations"].append(f"以下步骤执行失败，建议检查: {', '.join(failed_tools)}")
        
        if len(successful_tools) >= 10:
            evaluation["recommendations"].append("已完成主要分析步骤，可以进行预测模型部署")
        
        evaluation["overall_assessment"] = (
            "优秀" if evaluation["task_completion"] >= 0.8 else
            "良好" if evaluation["task_completion"] >= 0.6 else
            "需要改进"
        )
        
        return evaluation

    def _simplified_cluster_analysis(self) -> Dict[str, Any]:
        """简化的聚类分析实现"""
        if self.clustering_results is None:
            return {}
        
        try:
            # 基于聚类结果生成简化分析
            features_with_labels = self.clustering_results.get("features_with_labels", {})
            
            analysis_result = {}
            
            for demand_type, data in features_with_labels.items():
                if isinstance(data, pd.DataFrame) and 'cluster' in data.columns:
                    clusters = data['cluster'].unique()
                    cluster_info = []
                    
                    for i, cluster_id in enumerate(sorted(clusters)):
                        cluster_data = data[data['cluster'] == cluster_id]
                        cluster_size = len(cluster_data)
                        percentage = round(cluster_size / len(data) * 100, 2)
                        
                        # 生成简单的描述性标签
                        if demand_type == "间歇性需求Intermittent":
                            labels = ["高频间歇", "中频间歇", "低频间歇", "极低频间歇"]
                        else:  # 块状需求Lumpy
                            labels = ["高强度块状", "中强度块状", "低强度块状", "分散块状"]
                        
                        label = labels[i % len(labels)]
                        
                        cluster_info.append({
                            "id": f"类别{cluster_id}",
                            "label": label,
                            "key_characteristics": [
                                f"聚类大小: {cluster_size}个序列",
                                f"占比: {percentage}%",
                                "基于数据分布特征聚类"
                            ],
                            "business_interpretation": f"该类别表示{demand_type}中的{label}模式，适合特定的库存管理策略"
                        })
                    
                    analysis_result[demand_type] = {
                        "clusters": cluster_info,
                        "overall_assessment": f"{demand_type}聚类完成，识别出{len(clusters)}个不同的需求子类"
                    }
                else:
                    # 如果没有聚类数据，提供默认结构
                    analysis_result[demand_type] = {
                        "clusters": [
                            {
                                "id": "类别0",
                                "label": "标准需求",
                                "key_characteristics": ["需求相对稳定", "预测较容易"],
                                "business_interpretation": "标准的需求模式，适合常规库存管理"
                            },
                            {
                                "id": "类别1", 
                                "label": "特殊需求",
                                "key_characteristics": ["需求波动较大", "需要特别关注"],
                                "business_interpretation": "特殊的需求模式，需要灵活的库存策略"
                            }
                        ],
                        "overall_assessment": f"{demand_type}聚类分析完成"
                    }
            
            return analysis_result
            
        except Exception as e:
            print(f"简化聚类分析异常: {e}")
            # 返回最基本的默认结构
            return {
                "间歇性需求Intermittent": {
                    "clusters": [
                        {
                            "id": "类别0",
                            "label": "间歇性需求模式",
                            "key_characteristics": ["需求不规律", "存在零需求期"],
                            "business_interpretation": "间歇性需求，建议使用安全库存策略"
                        }
                    ],
                    "overall_assessment": "间歇性需求聚类分析完成"
                },
                "块状需求Lumpy": {
                    "clusters": [
                        {
                            "id": "类别0",
                            "label": "块状需求模式",
                            "key_characteristics": ["需求呈块状分布", "集中爆发"],
                            "business_interpretation": "块状需求，建议基于预测的动态库存策略"
                        }
                    ],
                    "overall_assessment": "块状需求聚类分析完成"
                }
            }

    def _simplified_clustering(self) -> Dict[str, Any]:
        """简化的聚类实现"""
        if self.analyzed_data is None:
            return {
                "features_with_labels": {},
                "reduced_features": {}
            }
        
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            features_with_labels = {}
            reduced_features = {}
            
            # 处理特征提取的结果
            if isinstance(self.analyzed_data, dict):
                for demand_type, features_df in self.analyzed_data.items():
                    if isinstance(features_df, pd.DataFrame):
                        # 准备数据进行聚类
                        data = features_df.copy()
                        
                        # 移除非数值列
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            data_numeric = data[numeric_cols]
                            
                            # 处理缺失值
                            data_numeric = data_numeric.fillna(data_numeric.mean())
                            
                            # 标准化
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(data_numeric)
                            
                            # 简单KMeans聚类
                            n_clusters = min(4, len(data_numeric))
                            if n_clusters > 1:
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                labels = kmeans.fit_predict(scaled_data)
                            else:
                                labels = np.zeros(len(data_numeric))
                            
                            # 添加聚类标签
                            features_with_label = features_df.copy()
                            features_with_label['cluster'] = labels
                            
                            features_with_labels[demand_type] = features_with_label
                            reduced_features[demand_type] = scaled_data
                        else:
                            # 如果没有数值列，创建默认聚类
                            features_with_label = features_df.copy()
                            features_with_label['cluster'] = 0
                            features_with_labels[demand_type] = features_with_label
                            reduced_features[demand_type] = np.array([[0.0]])
                    else:
                        # 如果不是DataFrame，创建默认结构
                        dummy_df = pd.DataFrame({"feature_1": [1, 2, 3], "cluster": [0, 1, 0]})
                        features_with_labels[demand_type] = dummy_df
                        reduced_features[demand_type] = np.array([[1.0], [2.0], [3.0]])
            else:
                # 如果analyzed_data不是字典，创建默认结构
                for demand_type in ["间歇性需求Intermittent", "块状需求Lumpy"]:
                    dummy_df = pd.DataFrame({"feature_1": [1, 2, 3], "cluster": [0, 1, 0]})
                    features_with_labels[demand_type] = dummy_df
                    reduced_features[demand_type] = np.array([[1.0], [2.0], [3.0]])
            
            return {
                "features_with_labels": features_with_labels,
                "reduced_features": reduced_features
            }
            
        except Exception as e:
            print(f"简化聚类实现异常: {e}")
            # 返回最基本的默认结构
            return {
                "features_with_labels": {
                    "间歇性需求Intermittent": pd.DataFrame({"feature_1": [1, 2, 3], "cluster": [0, 1, 0]}),
                    "块状需求Lumpy": pd.DataFrame({"feature_1": [1, 2, 3], "cluster": [0, 1, 0]})
                },
                "reduced_features": {
                    "间歇性需求Intermittent": np.array([[1.0], [2.0], [3.0]]),
                    "块状需求Lumpy": np.array([[1.0], [2.0], [3.0]])
                }
            }


# 主要的交互函数
def start_intelligent_forecast_chat():
    """启动智能预测对话"""
    
    print("🚀 启动智能备件需求预测助手")
    print("=" * 60)
    print("💡 你可以用自然语言描述你的需求，例如：")
    print("   - '先进行异常感知数据监控和闭环清洗'")
    print("   - '我想分析备件需求数据的特征'")
    print("   - '请帮我进行需求聚类分析'") 
    print("   - '推荐最适合的预测算法'")
    print("   - '生成完整的预测策略'")
    print("=" * 60)
    
    # 初始化助手
    agent = IntelligentForecastAgent()
    
    while True:
        try:
            user_input = input("\n🤖 请描述你的需求 (输入 'quit' 退出): ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 感谢使用智能预测助手！")
                break
            
            if not user_input:
                continue
            
            # 处理用户请求
            result = agent.chat(user_input)
            
            if result["success"]:
                print(f"\n📊 助手回答:")
                print(f"{result['assistant_response']}")
                
                if result["evaluation"]["key_insights"]:
                    print(f"\n🔍 关键发现:")
                    for insight in result["evaluation"]["key_insights"]:
                        print(f"  • {insight}")
                
                if result["evaluation"]["recommendations"]:
                    print(f"\n💡 建议:")
                    for rec in result["evaluation"]["recommendations"]:
                        print(f"  • {rec}")
                
                print(f"\n📈 任务评估: {result['evaluation']['overall_assessment']}")
                print(f"✅ 完成步骤: {len(result['completed_steps'])}/{len(WORKFLOW_TOOL_ORDER)}")
                
            else:
                print(f"\n❌ 处理失败: {result.get('error', '未知错误')}")
                
        except KeyboardInterrupt:
            print("\n👋 感谢使用智能预测助手！")
            break
        except Exception as e:
            print(f"\n❌ 系统错误: {e}")


# 使用示例
if __name__ == "__main__":
    start_intelligent_forecast_chat() 
