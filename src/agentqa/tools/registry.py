"""工具注册表：按名称分发，任何异常统一转为 ok=false。"""
from __future__ import annotations

from typing import Any, Dict, List

from agentqa.rag.index import SparseIndex
from agentqa.rag.types import Chunk
from agentqa.tools.read_snippet import read_snippet
from agentqa.tools.search_documents import search_documents
from agentqa.tools.submit_answer import submit_answer


class ToolRegistry:
    """绑定检索索引与切片库，提供统一的 execute(name, args)。"""

    def __init__(self, index: SparseIndex, chunks: List[Chunk]):
        self._index = index
        self._chunks_by_id: Dict[str, Chunk] = {c.chunk_id: c for c in chunks}

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "search_documents":
                return search_documents(
                    self._index,
                    str(args.get("query", "")),
                    int(args.get("top_k", 5)),
                )
            if name == "read_snippet":
                return read_snippet(
                    self._chunks_by_id,
                    str(args.get("doc_id", "")),
                    str(args.get("chunk_id", "")),
                    str(args.get("section", "")),
                )
            if name == "submit_answer":
                return submit_answer(args, self._chunks_by_id)
            return {"ok": False, "error_code": "UNKNOWN_TOOL", "message": f"unknown tool: {name}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error_code": "TOOL_ERROR", "message": str(e)}


class FailingSearchRegistry(ToolRegistry):
    """让 search_documents 前 fail_times 次返回超时错误，用于演示工具失败路径。"""

    def __init__(self, index: SparseIndex, chunks: List[Chunk], fail_times: int = 1):
        super().__init__(index, chunks)
        self._fail_remaining = fail_times

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "search_documents" and self._fail_remaining > 0:
            self._fail_remaining -= 1
            return {"ok": False, "error_code": "TOOL_TIMEOUT", "message": "search_documents 超时"}
        return super().execute(name, args)
