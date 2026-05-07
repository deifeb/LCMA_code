#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""向量存储模块：负责ChromaDB持久向量化、语义检索和关键词混合检索。"""

import re
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import requests
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    """向量存储类，优先使用ChromaDB，缺失依赖时回退到内存向量检索。"""

    def __init__(
        self,
        model_name: str = "mxbai-embed-large",
        persist_directory: str = "knowledge_base/.chroma",
        collection_name: str = "default",
        embedding_base_url: str = "http://localhost:11434"
    ):
        """初始化向量存储

        Args:
            model_name: 向量化模型名称
            persist_directory: ChromaDB持久化目录
            collection_name: Chroma collection名称
        """
        self.model_name = model_name
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.persist_directory = str(Path(persist_directory))
        self.collection_name = self._sanitize_collection_name(collection_name)
        self.vectors: Dict[str, np.ndarray] = {}
        self.texts: Dict[str, str] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}
        self.collection = None
        self._init_chroma_collection()

    def _init_chroma_collection(self):
        """初始化ChromaDB collection。"""
        try:
            import chromadb

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"ChromaDB初始化失败，回退到内存向量库: {e}")
            self.collection = None

    def _sanitize_collection_name(self, name: str) -> str:
        """Chroma collection名称只保留安全字符。"""
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name)).strip("_")
        if len(safe) < 3:
            safe = f"collection_{uuid.uuid4().hex[:8]}"
        return safe[:63]

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Chroma metadata只支持简单标量，这里做安全转换。"""
        safe = {}
        for key, value in metadata.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                safe[str(key)] = value
            else:
                safe[str(key)] = str(value)
        return safe
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """使用Ollama API获取文本向量
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 文本向量
        """
        try:
            response = requests.post(
                f"{self.embedding_base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            )
            if response.status_code == 200:
                return np.array(response.json()["embedding"])
            else:
                raise Exception(f"API调用失败: {response.status_code}")
        except Exception as e:
            print(f"获取文本向量时出错: {str(e)}")
            return None

    def add_text(self, text_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加文本并进行向量化

        Args:
            text_id: 文本唯一标识符
            text: 文本内容
            metadata: 元数据

        Returns:
            bool: 是否添加成功
        """
        try:
            metadata = self._sanitize_metadata(metadata or {})
            vector = self._get_embedding(text)
            if vector is not None:
                self.vectors[text_id] = vector
                self.texts[text_id] = text
                self.metadatas[text_id] = metadata
                if self.collection is not None:
                    self.collection.upsert(
                        ids=[text_id],
                        documents=[text],
                        embeddings=[vector.tolist()],
                        metadatas=[metadata]
                    )
                return True
            return False
        except Exception as e:
            print(f"向量化文本时出错: {str(e)}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """搜索最相似的文本

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[Tuple[str, float]]: 文本ID和相似度得分列表
        """
        if self.collection is not None:
            chroma_results = self._semantic_search_chroma(query, top_k)
            if chroma_results:
                return [(item["id"], item["semantic_score"]) for item in chroma_results]

        if not self.vectors:
            return []

        try:
            query_vector = self._get_embedding(query)
            if query_vector is None:
                return []
                
            scores = {}
            for text_id, vector in self.vectors.items():
                similarity = cosine_similarity(
                    query_vector.reshape(1, -1),
                    vector.reshape(1, -1)
                )[0][0]
                scores[text_id] = similarity

            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_scores[:top_k]
        except Exception as e:
            print(f"搜索相似文本时出错: {str(e)}")
            return []

    def get_text(self, text_id: str) -> Optional[str]:
        """获取文本内容

        Args:
            text_id: 文本ID

        Returns:
            Optional[str]: 文本内容或None
        """
        if text_id in self.texts:
            return self.texts.get(text_id)
        if self.collection is not None:
            try:
                result = self.collection.get(ids=[text_id], include=["documents"])
                docs = result.get("documents") or []
                return docs[0] if docs else None
            except Exception:
                return None
        return None

    def delete_text(self, text_id: str) -> bool:
        """删除文本及其向量

        Args:
            text_id: 文本ID

        Returns:
            bool: 是否删除成功
        """
        if text_id not in self.vectors:
            return False

        try:
            if self.collection is not None:
                self.collection.delete(ids=[text_id])
            del self.vectors[text_id]
            del self.texts[text_id]
            self.metadatas.pop(text_id, None)
            return True
        except Exception as e:
            print(f"删除文本时出错: {str(e)}")
            return False

    def clear(self):
        """清空所有文本和向量"""
        if self.collection is not None:
            try:
                ids = self.collection.get().get("ids", [])
                if ids:
                    self.collection.delete(ids=ids)
            except Exception as e:
                print(f"清空ChromaDB集合失败: {e}")
        self.vectors.clear()
        self.texts.clear()
        self.metadatas.clear()

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """语义 + 关键词混合检索。"""
        semantic_results = self._semantic_search_chroma(query, max(top_k * 4, 10))
        if not semantic_results:
            semantic_results = self._semantic_search_memory(query, max(top_k * 4, 10))

        all_texts = self._get_all_texts()
        candidate_map: Dict[str, Dict[str, Any]] = {}

        for item in semantic_results:
            candidate_map[item["id"]] = item

        keyword_scores = {
            text_id: self._keyword_score(query, text)
            for text_id, text in all_texts.items()
        }
        for text_id, keyword_score in keyword_scores.items():
            if keyword_score > 0 or text_id in candidate_map:
                item = candidate_map.setdefault(text_id, {
                    "id": text_id,
                    "text": all_texts.get(text_id, ""),
                    "metadata": self.metadatas.get(text_id, {}),
                    "semantic_score": 0.0
                })
                item["keyword_score"] = keyword_score

        for item in candidate_map.values():
            item["keyword_score"] = item.get("keyword_score", 0.0)
            item["hybrid_score"] = (
                semantic_weight * item.get("semantic_score", 0.0) +
                keyword_weight * item.get("keyword_score", 0.0)
            )

        return sorted(candidate_map.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]

    def _semantic_search_chroma(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.collection is None:
            return []
        try:
            query_vector = self._get_embedding(query)
            if query_vector is None:
                return []
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            items = []
            for text_id, doc, metadata, distance in zip(ids, docs, metadatas, distances):
                # cosine distance越小越相关，转换为0-1之间的相似分。
                semantic_score = float(max(0.0, 1.0 - distance))
                items.append({
                    "id": text_id,
                    "text": doc,
                    "metadata": metadata or {},
                    "semantic_score": semantic_score
                })
            return items
        except Exception as e:
            print(f"ChromaDB语义检索失败: {e}")
            return []

    def _semantic_search_memory(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        results = []
        for text_id, score in self.search(query, top_k):
            results.append({
                "id": text_id,
                "text": self.texts.get(text_id, ""),
                "metadata": self.metadatas.get(text_id, {}),
                "semantic_score": float(score)
            })
        return results

    def _get_all_texts(self) -> Dict[str, str]:
        if self.collection is None:
            return dict(self.texts)
        try:
            results = self.collection.get(include=["documents", "metadatas"])
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            for text_id, doc, metadata in zip(ids, docs, metadatas):
                self.texts[text_id] = doc
                self.metadatas[text_id] = metadata or {}
            return dict(self.texts)
        except Exception:
            return dict(self.texts)

    def _keyword_score(self, query: str, text: str) -> float:
        query_terms = self._tokenize(query)
        text_terms = self._tokenize(text)
        if not query_terms or not text_terms:
            return 0.0
        overlap = query_terms.intersection(text_terms)
        return len(overlap) / max(len(query_terms), 1)

    def _tokenize(self, text: str) -> set:
        tokens = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", str(text).lower())
        expanded = set(tokens)
        for token in tokens:
            if len(token) > 2 and re.search(r"[\u4e00-\u9fff]", token):
                expanded.update(token[i:i + 2] for i in range(len(token) - 1))
        return expanded
