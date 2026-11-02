"""工具：submit_answer —— 提交结构化答案，经确定性校验后返回结果。"""
from __future__ import annotations

from typing import Any, Dict

from agentqa.agent.citation import validate_submission
from agentqa.rag.types import Chunk


def submit_answer(payload: Dict[str, Any], chunks_by_id: Dict[str, Chunk]) -> Dict[str, Any]:
    """校验并提交答案。校验失败返回 ok=false 与错误，由模型修正。"""
    status = str(payload.get("status", ""))
    citations = payload.get("citations") or []
    gaps = payload.get("gaps") or []

    errors = validate_submission(status, citations, gaps, chunks_by_id)
    if errors:
        return {"ok": False, "error_code": "INVALID_SUBMISSION", "errors": errors}

    return {
        "ok": True,
        "status": status,
        "answer": payload.get("answer", ""),
        "citations": citations,
        "gaps": gaps,
        "clarification_question": payload.get("clarification_question", ""),
    }
