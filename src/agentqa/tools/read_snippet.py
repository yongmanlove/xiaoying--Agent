"""工具：read_snippet —— 读取指定文档片段的完整原文与元数据。"""
from __future__ import annotations

from typing import Any, Dict

from agentqa.rag.types import Chunk


def read_snippet(
    chunks_by_id: Dict[str, Chunk],
    doc_id: str,
    chunk_id: str = "",
    section: str = "",
) -> Dict[str, Any]:
    """按 chunk_id 或 doc_id（可选 section）定位并返回原文片段与元数据。

    引用前必须调用本工具，保证 quote 能逐字核验。
    """
    chunk: Chunk | None = None
    if chunk_id:
        chunk = chunks_by_id.get(chunk_id)
    if chunk is None and doc_id:
        candidates = [c for c in chunks_by_id.values() if c.doc_id == doc_id]
        if section:
            candidates = [c for c in candidates if c.section == section]
        if candidates:
            chunk = candidates[0]
    if chunk is None:
        return {
            "ok": False,
            "error_code": "DOC_NOT_FOUND",
            "message": f"未找到文档片段: doc_id={doc_id} chunk_id={chunk_id} section={section}",
        }

    return {
        "ok": True,
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "section": chunk.section,
        "text": chunk.text,
        "title": chunk.title,
        "version": chunk.version,
        "effective_date": chunk.effective_date,
        "department": chunk.department,
        "applies_to": chunk.applies_to,
        "supersedes": chunk.supersedes,
        "topic_key": chunk.topic_key,
        "status": chunk.status,
    }
