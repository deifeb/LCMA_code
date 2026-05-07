#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大模型接口封装
"""

import re
import json
import pickle
import requests
from typing import Dict, List, Optional
from src.config import (
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    MODELSCOPE_API_KEY,
    MODELSCOPE_BASE_URL,
    MODELSCOPE_MODEL,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
)


class OllamaInterface:
    """大模型接口封装：兼容ModelScope OpenAI API与Ollama本地API。"""

    def __init__(
        self,
        model_name: str = None,
        api_url: str = None,
        memory_manager=None,
        provider: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = LLM_TEMPERATURE,
    ):
        self.provider = (provider or LLM_PROVIDER).strip().lower()
        if self.provider == "modelscope":
            self.model_name = model_name or MODELSCOPE_MODEL
            self.base_url = (base_url or MODELSCOPE_BASE_URL).rstrip("/")
            self.api_url = api_url or f"{self.base_url}/chat/completions"
            self.api_key = api_key if api_key is not None else MODELSCOPE_API_KEY
        else:
            self.model_name = model_name or OLLAMA_MODEL
            self.api_url = api_url or OLLAMA_API_URL
            self.api_key = api_key or ""
            self.base_url = base_url or ""
        self.temperature = temperature
        self.memory_manager = memory_manager
        self.agent_name = "default"

    def set_memory_manager(self, memory_manager):
        """设置记忆管理器
        
        Args:
            memory_manager: 记忆管理器实例
        """
        self.memory_manager = memory_manager
        
    def set_agent_name(self, agent_name: str):
        """设置当前智能体名称，用于记忆存储
        
        Args:
            agent_name: 智能体名称
        """
        self.agent_name = agent_name

    def query(self, prompt: str, system_prompt: str = None, max_tokens: int = 2000, use_memory: bool = True, agent_id: str = None) -> str:
        """向Ollama发送请求并获取响应
        
        Args:
            prompt: 提示词
            system_prompt: 系统提示词
            max_tokens: 最大生成token数
            use_memory: 是否使用记忆增强
            agent_id: 智能体ID，用于从特定智能体知识库中检索知识
            
        Returns:
            str: 模型响应
        """
        headers = {"Content-Type": "application/json"}
        
        # 如果未指定agent_id，则使用当前智能体名称
        if agent_id is None:
            agent_id = self.agent_name
        
        # 如果启用记忆增强且记忆管理器存在
        enhanced_prompt = prompt
        if use_memory and self.memory_manager:
            # 检索相关记忆
            relevant_memories = self.memory_manager.retrieve_relevant_memories(prompt, top_k=3, agent_id=agent_id)
            
            # 检索相关知识库内容
            relevant_knowledge = []
            if hasattr(self.memory_manager, 'agent_knowledge_manager') and self.memory_manager.agent_knowledge_manager:
                relevant_knowledge = self.memory_manager.agent_knowledge_manager.search_knowledge(agent_id, prompt, top_k=2)
            
            # 构建增强提示词
            context_parts = []
            
            if relevant_memories:
                memory_context = "相关历史记忆:\n"
                for i, memory in enumerate(relevant_memories):
                    memory_content = memory.get('content', '')
                    memory_context += f"记忆 {i+1}:\n{memory_content}\n\n"
                context_parts.append(memory_context)
            
            if relevant_knowledge:
                knowledge_context = "相关专业知识:\n"
                for i, knowledge in enumerate(relevant_knowledge):
                    knowledge_content = knowledge.get('content', '')[:500]
                    knowledge_context += f"知识 {i+1}:\n{knowledge_content}...\n\n"
                context_parts.append(knowledge_context)
            
            if context_parts:
                context = "\n\n".join(context_parts)
                # 增强提示词
                enhanced_prompt = f"{context}\n当前问题: {prompt}\n请基于上述相关信息和当前问题进行回答。"

        # 构建请求数据
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": enhanced_prompt})

        try:
            if self.provider == "modelscope":
                headers["Authorization"] = f"Bearer {self.api_key}"
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": max_tokens or LLM_MAX_TOKENS,
                    "stream": False
                }
                response = requests.post(self.api_url, headers=headers, json=data, timeout=1000)
                response.raise_for_status()
                response_data = response.json()["choices"][0]["message"]
            else:
                data = {
                    "model": self.model_name,
                    "options": {
                        "temperature": self.temperature
                    },
                    "stream": False,
                    "messages": messages
                }
                response = requests.post(self.api_url, headers=headers, json=data, timeout=1000)
                response.raise_for_status()
                response_data = response.json()["message"]
            
            # 存储交互记忆
            if use_memory and self.memory_manager:
                try:
                    self.memory_manager.store_interaction(
                        self.agent_name, 
                        prompt, 
                        response_data['content'],
                        {"enhanced": enhanced_prompt != prompt}
                    )
                except Exception as e:
                    print(f"记忆存储错误: {e}")
                
            return response_data
        except Exception as e:
            print(f"{self.provider}模型请求错误: {e}")
            return {"content": f"模型请求失败: {e}", "role": "assistant"}

    def parse_json_response(self, response: str) -> dict:
        """尝试从响应中解析JSON格式内容"""
        try:
            # 1. 预处理响应文本，修复常见的JSON格式错误
            # 处理可能缺少逗号的情况
            response = re.sub(r'(\"\w+\")\s*\n\s*(\"\w+\")', r'\1,\n\2', response)
            # 处理可能缺少冒号的情况
            response = re.sub(r'(\"\w+\")\s+(\{|\[|\")', r'\1: \2', response)

            # 2. 如果响应中包含```json，则提取json代码块
            if "```json" in response:
                json_block = response.split("```json")[1].split("```")[0].strip()
                try:
                    return json.loads(json_block)
                except json.JSONDecodeError:
                    # 使用大模型接口检查和修复json_block
                    prompt = f"请检查并修复以下JSON文本中的符号问题，保持格式和内容不变，返回形式严格按照以下JSON格式，返回修正后的JSON文本时以```json开头。\n{json_block}"
                    system_prompt = "请检查以下JSON文本中的符号问题，并进行修复。确保属性名使用双引号括起来，所有括号、方括号和逗号都正确闭合和匹配，请将修改后的文本以JSON格式输出，不允许使用其他格式。"
                    print(f"JSON格式修复重试中，请求大模型...")
                    fix_response = OllamaInterface.query(self, prompt, system_prompt=system_prompt)
                    if "```json" in fix_response['content']:
                        fix_json_block = fix_response['content'].split("```json")[1].split("```")[0].strip()
                        return json.loads(fix_json_block)
                    else:
                        fix_json_block = self.filter_think_tags(fix_response['content'])
                        return json.loads(fix_json_block)
            else:
                fix_json_block = self.filter_think_tags(response)
                try:
                    return json.loads(fix_json_block)
                except json.JSONDecodeError as e:
                    # 添加更详细的错误处理，例如记录日志或抛出带提示的异常
                    error_msg = f"修复后的 JSON 无效，内容：{fix_json_block}"
                    raise ValueError(error_msg)  # 移除 from e

        except Exception as e:
            print(f"JSON解析错误: {e}")
            return {}

    def filter_think_tags(self, content):  # 定义正则表达式模式，用于匹配 <think> *</think> 这种格式的内容
        pattern = r'<think>.*?</think>'  # 使用 re.sub 函数将匹配到的内容替换为空字符串
        filtered_text = re.sub(pattern, '', content, flags=re.DOTALL)  # 去除多余的空白字符
        filtered_text = filtered_text.strip()
        return filtered_text

    def save_to_pickle(self, obj, filename, protocol=pickle.HIGHEST_PROTOCOL):
        """
        将对象序列化保存到Pickle文件中

        参数:
            obj: 要保存的Python对象
            filename: 目标文件名（包含路径）
            protocol: 使用的Pickle协议版本，默认为最高版本
        """
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol)

    def load_from_pickle(self, filename):
        """
        从Pickle文件中读取并反序列化对象

        参数:
            filename: Pickle文件名（包含路径）

        返回:
            反序列化的Python对象
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
