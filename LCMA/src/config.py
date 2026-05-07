#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
备件需求预测系统配置
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default

# 设置显示中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
plt.rcParams["axes.unicode_minus"] = False

# LLM配置：默认接入ModelScope OpenAI-compatible API
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "modelscope").strip().lower()
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.1)
LLM_TIMEOUT = _get_int("LLM_TIMEOUT", 60)
LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 2000)

MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY", "")
MODELSCOPE_BASE_URL = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1/")
MODELSCOPE_MODEL = os.getenv("MODELSCOPE_MODEL", "deepseek-ai/DeepSeek-V3.1")

# Ollama配置：作为本地回退与RAG嵌入服务
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")

LLM_MODEL = MODELSCOPE_MODEL if LLM_PROVIDER == "modelscope" else OLLAMA_MODEL

# RAG配置
RAG_ENABLED = _get_bool("RAG_ENABLED", True)
KNOWLEDGE_BASE_PATH = os.getenv("RAG_KNOWLEDGE_BASE_PATH", "knowledge_base")
RAG_HISTORY_PATH = os.getenv("RAG_HISTORY_PATH", "knowledge_base/interaction_history")
RAG_CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", "knowledge_base/.chroma")
RAG_EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "ollama").strip().lower()
RAG_OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL", OLLAMA_BASE_URL)
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "mxbai-embed-large")
RAG_SEARCH_MODE = os.getenv("RAG_SEARCH_MODE", "hybrid")
RAG_TOP_K = _get_int("RAG_TOP_K", 5)
RAG_SEMANTIC_WEIGHT = _get_float("RAG_SEMANTIC_WEIGHT", 0.7)
RAG_KEYWORD_WEIGHT = _get_float("RAG_KEYWORD_WEIGHT", 0.3)
