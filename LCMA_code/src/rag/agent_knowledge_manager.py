#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能体知识库管理器：管理不同智能体的专属知识库
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from .rag_manager import RAGManager


class AgentKnowledgeManager:
    """智能体知识库管理器，负责管理不同智能体的专属知识库"""

    def __init__(
        self,
        base_path: str = "knowledge_base",
        embedding_model: str = "mxbai-embed-large",
        chroma_path: str = "knowledge_base/.chroma",
        embedding_base_url: str = "http://localhost:11434",
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """初始化智能体知识库管理器

        Args:
            base_path: 知识库基础路径
        """
        self.base_path = Path(base_path)
        self.embedding_model = embedding_model
        self.chroma_path = chroma_path
        self.embedding_base_url = embedding_base_url
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.agent_rag_managers = {}
        
        # 创建智能体知识库目录结构
        self.agent_dirs = {
            "dmagent": "数据监控智能体",
            "feature_extraction": "特征提取智能体",
            "classification": "分类智能体",
            "forecasting_model_select": "预测模型选择智能体",
            "feedback_coordinator": "反馈协调智能体"
        }
        
        # 初始化各智能体知识库目录
        for agent_id, agent_name in self.agent_dirs.items():
            agent_dir = self.base_path / agent_id
            agent_dir.mkdir(exist_ok=True)
            # 为每个智能体创建独立的RAG管理器
            self.agent_rag_managers[agent_id] = RAGManager(
                str(agent_dir),
                embedding_model=self.embedding_model,
                chroma_path=self.chroma_path,
                embedding_base_url=self.embedding_base_url,
                semantic_weight=self.semantic_weight,
                keyword_weight=self.keyword_weight
            )
            self.agent_rag_managers[agent_id].add_documents_from_directory(str(agent_dir), recursive=True)
    
    def get_agent_rag_manager(self, agent_id: str) -> Optional[RAGManager]:
        """获取指定智能体的RAG管理器

        Args:
            agent_id: 智能体ID

        Returns:
            Optional[RAGManager]: 智能体RAG管理器或None
        """
        if agent_id not in self.agent_rag_managers:
            return None
        return self.agent_rag_managers[agent_id]
    
    def add_knowledge(self, agent_id: str, doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """添加知识到指定智能体的知识库

        Args:
            agent_id: 智能体ID
            doc_id: 文档唯一标识符
            content: 文档内容
            metadata: 文档元数据

        Returns:
            bool: 是否添加成功
        """
        rag_manager = self.get_agent_rag_manager(agent_id)
        if not rag_manager:
            return False
        return rag_manager.add_document(doc_id, content, metadata)
    
    def search_knowledge(self, agent_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """搜索指定智能体的知识库

        Args:
            agent_id: 智能体ID
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[Dict]: 相关文档列表
        """
        rag_manager = self.get_agent_rag_manager(agent_id)
        if not rag_manager:
            return []
        return rag_manager.search(query, top_k)
    
    def add_documents_from_directory(self, agent_id: str, directory_path: str, recursive: bool = True) -> Dict[str, bool]:
        """从指定目录批量添加文档到指定智能体的知识库

        Args:
            agent_id: 智能体ID
            directory_path: 文档目录路径
            recursive: 是否递归处理子目录

        Returns:
            Dict[str, bool]: 文件路径到处理结果的映射
        """
        rag_manager = self.get_agent_rag_manager(agent_id)
        if not rag_manager:
            return {}
        return rag_manager.add_documents_from_directory(directory_path, recursive)
    
    def clear_agent_knowledge(self, agent_id: str) -> bool:
        """清空指定智能体的知识库

        Args:
            agent_id: 智能体ID

        Returns:
            bool: 是否清空成功
        """
        rag_manager = self.get_agent_rag_manager(agent_id)
        if not rag_manager:
            return False
        return rag_manager.clear_all_documents()
    
    def clear_all_knowledge(self) -> bool:
        """清空所有智能体的知识库

        Returns:
            bool: 是否清空成功
        """
        success = True
        for agent_id in self.agent_rag_managers:
            if not self.clear_agent_knowledge(agent_id):
                success = False
        return success
