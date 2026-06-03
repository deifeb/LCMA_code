#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档存储模块：负责管理和存储知识库文档
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import json

class DocumentStore:
    """文档存储类，管理知识库文档的存储和检索"""

    def __init__(self, storage_path: str = "knowledge_base"):
        """初始化文档存储

        Args:
            storage_path: 知识库存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.documents: Dict[str, Dict] = {}
        self._load_documents()

    def _get_document_file_path(self, doc_id: str) -> Path:
        """Return a filesystem-safe JSON path while preserving the original document id."""
        safe_doc_id = str(doc_id)
        for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
            safe_doc_id = safe_doc_id.replace(char, "__")
        return self.storage_path / f"{safe_doc_id}.json"

    def _load_documents(self):
        """加载已存储的文档"""
        if not self.storage_path.exists():
            return

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)
                    self.documents[doc_data["id"]] = doc_data
            except Exception as e:
                print(f"加载文档 {file_path} 时出错: {str(e)}")

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """添加新文档

        Args:
            doc_id: 文档唯一标识符
            content: 文档内容
            metadata: 文档元数据

        Returns:
            bool: 是否添加成功
        """
        if doc_id in self.documents:
            return False

        document = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {}
        }

        file_path = self._get_document_file_path(doc_id)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            self.documents[doc_id] = document
            return True
        except Exception as e:
            print(f"保存文档时出错: {str(e)}")
            return False

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """获取文档

        Args:
            doc_id: 文档ID

        Returns:
            Optional[Dict]: 文档数据或None
        """
        return self.documents.get(doc_id)

    def get_all_documents(self) -> List[Dict]:
        """获取所有文档

        Returns:
            List[Dict]: 文档列表
        """
        return list(self.documents.values())

    def update_document(self, doc_id: str, content: str = None, metadata: Dict = None) -> bool:
        """更新文档

        Args:
            doc_id: 文档ID
            content: 新的文档内容
            metadata: 新的元数据

        Returns:
            bool: 是否更新成功
        """
        if doc_id not in self.documents:
            return False

        document = self.documents[doc_id]
        if content is not None:
            document["content"] = content
        if metadata is not None:
            document["metadata"].update(metadata)

        file_path = self._get_document_file_path(doc_id)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"更新文档时出错: {str(e)}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """删除文档

        Args:
            doc_id: 文档ID

        Returns:
            bool: 是否删除成功
        """
        if doc_id not in self.documents:
            return False

        file_path = self._get_document_file_path(doc_id)
        try:
            file_path.unlink()
            del self.documents[doc_id]
            return True
        except Exception as e:
            print(f"删除文档时出错: {str(e)}")
            return False
