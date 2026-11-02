"""工具：search_documents —— 检索制度片段，只返回摘要与冲突候选，不返回全文。"""
from __future__ import annotations

from typing import Any, Dict, List

from agentqa.rag.conflict import detect_conflict_candidates
from agentqa.rag.index import SparseIndex
from agentqa.rag.types import RetrievedChunk

SNIPPET_MAX = 200


def _snippet(text: str, limit: int = SNIPPET_MAX) -> str:
    t = text.strip()
    return t if len(t) <= limit else t[:limit] + "……"


def _result_dict(h: RetrievedChunk) -> Dict[str, Any]:
    return {
        "doc_id": h.doc_id,
        "chunk_id": h.chunk_id,
        "title": h.title,
        "version": h.version,
        "effective_date": h.effective_date,
        "section": h.section,
        "snippet": _snippet(h.text),
        "score": round(h.score, 4),
    }


def search_documents(index: SparseIndex, query: str, top_k: int = 5) -> Dict[str, Any]:
    """执行检索并组装 observation。异常统一转为 ok=false，不向上抛出。"""
    try:
        hits: List[RetrievedChunk] = index.search(query, top_k)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error_code": "SEARCH_ERROR", "message": str(e)}

    return {
        "ok": True,
        "query": query,
        "results": [_result_dict(h) for h in hits],
        "conflict_candidates": detect_conflict_candidates(hits),
    }
