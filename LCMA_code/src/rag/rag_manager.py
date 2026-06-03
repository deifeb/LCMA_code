#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG管理器：整合文档存储和向量存储，提供统一的知识库检索接口
"""

from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import os
from .document_store import DocumentStore
from .vector_store import VectorStore
from .document_processor import DocumentProcessor

class RAGManager:
    """RAG管理器，整合文档存储和向量存储功能"""

    def __init__(self, 
                 storage_path: str = "knowledge_base",
                 embedding_model: str = "mxbai-embed-large",
                 chroma_path: str = "knowledge_base/.chroma",
                 embedding_base_url: str = "http://localhost:11434",
                 semantic_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        """初始化RAG管理器

        Args:
            storage_path: 知识库存储路径
            embedding_model: 向量化模型名称
        """
        self.storage_path = Path(storage_path)
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.doc_store = DocumentStore(storage_path)
        self.vector_store = VectorStore(
            embedding_model,
            persist_directory=chroma_path,
            collection_name=str(self.storage_path).replace("\\", "_").replace("/", "_"),
            embedding_base_url=embedding_base_url
        )
        self.supported_extensions = {'.md', '.docx'}
        self._sync_stores()

    def _sync_stores(self):
        """同步文档存储和向量存储"""
        for doc in self.doc_store.get_all_documents():
            if doc["id"] not in self.vector_store.texts:
                self.vector_store.add_text(doc["id"], doc["content"], doc.get("metadata", {}))

    def add_document(self, doc_id: str, content: str = None, metadata: Optional[Dict] = None, file_path: str = None) -> bool:
        """添加新文档到知识库

        Args:
            doc_id: 文档唯一标识符
            content: 文档内容（可选，如果提供file_path则不需要）
            metadata: 文档元数据
            file_path: 文档文件路径（可选）

        Returns:
            bool: 是否添加成功
        """
        if file_path:
            file_path = str(Path(file_path))
            file_ext = Path(file_path).suffix.lower()
            if file_ext in self.supported_extensions:
                processed_content = DocumentProcessor.process_document(file_path, content)
                if processed_content:
                    content = processed_content
                else:
                    return False
            elif not content:
                return False

        if self.doc_store.add_document(doc_id, content, metadata):
            return self.vector_store.add_text(doc_id, content, metadata)
        return False

    def add_documents_from_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, bool]:
        """从指定目录批量添加文档到知识库

        Args:
            directory_path: 文档目录路径
            recursive: 是否递归处理子目录，默认为True

        Returns:
            Dict[str, bool]: 文件路径到处理结果的映射
        """
        results = {}
        directory_path = Path(directory_path)

        if not directory_path.exists() or not directory_path.is_dir():
            return results

        def process_directory(path: Path):
            for item in path.iterdir():
                if item.is_file() and item.suffix.lower() in self.supported_extensions:
                    doc_id = str(item.relative_to(directory_path))
                    results[str(item)] = self.add_document(doc_id, file_path=str(item))
                elif item.is_dir() and recursive:
                    process_directory(item)

        process_directory(directory_path)
        return results

    def search(self, query: str, top_k: int = 5, hybrid: bool = True) -> List[Dict]:
        """搜索相关文档

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[Dict]: 相关文档列表，包含文档内容和相似度得分
        """
        results = []
        if hybrid:
            hybrid_docs = self.vector_store.hybrid_search(
                query,
                top_k,
                semantic_weight=self.semantic_weight,
                keyword_weight=self.keyword_weight
            )
            for item in hybrid_docs:
                doc = self.doc_store.get_document(item["id"]) or {
                    "id": item["id"],
                    "content": item.get("text", ""),
                    "metadata": item.get("metadata", {})
                }
                doc["similarity_score"] = float(item.get("semantic_score", 0.0))
                doc["keyword_score"] = float(item.get("keyword_score", 0.0))
                doc["hybrid_score"] = float(item.get("hybrid_score", 0.0))
                results.append(doc)
        else:
            similar_docs = self.vector_store.search(query, top_k)
            for doc_id, score in similar_docs:
                doc = self.doc_store.get_document(doc_id)
                if doc:
                    doc["similarity_score"] = float(score)
                    results.append(doc)
        
        # 判断检索结果并输出提示
        if not results:
            print(f"没有检索到相关的知识")
        else:
            print(f"已检索到{len(results)}条相关的知识")
        
        return results

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """获取文档

        Args:
            doc_id: 文档ID

        Returns:
            Optional[Dict]: 文档数据或None
        """
        return self.doc_store.get_document(doc_id)

    def update_document(self, doc_id: str, content: str = None, metadata: Dict = None) -> bool:
        """更新文档

        Args:
            doc_id: 文档ID
            content: 新的文档内容
            metadata: 新的元数据

        Returns:
            bool: 是否更新成功
        """
        if self.doc_store.update_document(doc_id, content, metadata):
            if content is not None:
                doc = self.doc_store.get_document(doc_id) or {}
                return self.vector_store.add_text(doc_id, content, doc.get("metadata", metadata or {}))
            return True
        return False

    def delete_document(self, doc_id: str) -> bool:
        """删除文档

        Args:
            doc_id: 文档ID

        Returns:
            bool: 是否删除成功
        """
        if self.doc_store.delete_document(doc_id):
            return self.vector_store.delete_text(doc_id)
        return False
        
    def clear_all_documents(self) -> bool:
        """清空知识库中的所有文档

        Returns:
            bool: 是否清空成功
        """
        all_docs = self.doc_store.get_all_documents()
        success = True
        
        for doc in all_docs:
            if not self.delete_document(doc["id"]):
                success = False
                
        return success
