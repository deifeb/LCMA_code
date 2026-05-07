#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档处理器：负责处理不同格式的文档内容
"""

from typing import Dict, Optional

class DocumentProcessor:
    """文档处理器，支持不同格式文档的内容提取和处理"""

    @staticmethod
    def process_markdown(content: str) -> str:
        """处理Markdown格式文档

        Args:
            content: Markdown格式的文档内容

        Returns:
            str: 提取的纯文本内容
        """
        try:
            import markdown
            # 将Markdown转换为HTML
            html = markdown.markdown(content)
            # TODO: 如果需要，可以添加HTML到纯文本的转换
            return html
        except ImportError:
            print("警告: markdown 模块未安装，直接返回原始内容")
            return content

    @staticmethod
    def process_docx(file_path: str) -> str:
        """处理Word文档

        Args:
            file_path: Word文档路径

        Returns:
            str: 提取的文本内容
        """
        try:
            import docx
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except ImportError:
            print("警告: python-docx 模块未安装，无法处理Word文档")
            return ""
        except Exception as e:
            print(f"处理Word文档时出错: {str(e)}")
            return ""

    @staticmethod
    def process_document(file_path: str, content: Optional[str] = None) -> str:
        """处理文档内容

        Args:
            file_path: 文档路径
            content: 可选的文档内容（用于直接处理文本内容）

        Returns:
            str: 处理后的文本内容
        """
        if content is not None:
            # 如果是.md文件，处理Markdown内容
            if file_path.endswith('.md'):
                return DocumentProcessor.process_markdown(content)
            return content
        
        # 处理.md文件
        if file_path.endswith('.md'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return DocumentProcessor.process_markdown(f.read())
            except Exception as e:
                print(f"读取Markdown文件时出错: {str(e)}")
                return ""
        
        # 处理.docx文件
        if file_path.endswith('.docx'):
            return DocumentProcessor.process_docx(file_path)
        
        return ""