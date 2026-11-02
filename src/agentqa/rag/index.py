"""稀疏检索索引：jieba 分词 + BM25，纯函数实现，不依赖 Settings。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi

from agentqa.rag.types import Chunk, RetrievedChunk

# 抑制 jieba 首次加载词典的日志噪音
jieba.setLogLevel(logging.WARNING)


@dataclass
class SparseIndex:
    """基于 BM25 的离线检索索引。"""

    _chunks: List[Chunk]
    _bm25: BM25Okapi
    _tokenized: List[List[str]]

    @classmethod
    def from_chunks(cls, chunks: List[Chunk]) -> "SparseIndex":
        tokenized = [list(jieba.cut(c.text)) for c in chunks]
        bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
        return cls(_chunks=chunks, _bm25=bm25, _tokenized=tokenized)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """按查询检索，只返回与查询有词元重叠的切片，按 BM25 分数降序。"""
        q = list(jieba.cut(query))
        qset = set(q)
        scores = self._bm25.get_scores(q)
        scored: List[Tuple[int, float]] = []
        for i in range(len(self._chunks)):
            if qset.isdisjoint(self._tokenized[i]):
                continue
            scored.append((i, float(scores[i])))
        scored.sort(key=lambda x: x[1], reverse=True)

        out: List[RetrievedChunk] = []
        for rank, (i, score) in enumerate(scored[:top_k], start=1):
            c = self._chunks[i]
            out.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    doc_id=c.doc_id,
                    title=c.title,
                    version=c.version,
                    effective_date=c.effective_date,
                    department=c.department,
                    applies_to=c.applies_to,
                    supersedes=c.supersedes,
                    topic_key=c.topic_key,
                    status=c.status,
                    section=c.section,
                    score=score,
                    rank=rank,
                )
            )
        return out
