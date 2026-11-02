"""检索相关的核心数据类型：文档切片与检索命中。

所有 doc 级元数据（doc_id/version/effective_date/topic_key 等）都会
随每个 chunk 一起保留，供检索工具返回与引用校验使用。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """一个文档切片，附带完整的文档级元数据。"""

    chunk_id: str
    text: str
    doc_id: str = ""
    title: str = ""
    version: str = ""
    effective_date: str = ""
    department: str = ""
    applies_to: str = ""
    supersedes: str = ""
    topic_key: str = ""
    status: str = ""
    section: str = ""


@dataclass
class RetrievedChunk(Chunk):
    """一次检索命中的切片，在 Chunk 基础上增加分数与排名。"""

    score: float = 0.0
    rank: int = 0
