"""任务状态枚举与最终结果数据类。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class TaskStatus(str, Enum):
    """Agent 任务的终态。"""

    ANSWERED = "answered"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    CONFLICT_NEEDS_CLARIFICATION = "conflict_needs_clarification"
    TOOL_ERROR = "tool_error"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"


@dataclass
class Citation:
    """一条引用：文档 ID + 切片 ID + 原文子串。"""

    doc_id: str
    chunk_id: str
    quote: str

    def to_dict(self) -> Dict[str, str]:
        return {"doc_id": self.doc_id, "chunk_id": self.chunk_id, "quote": self.quote}


@dataclass
class AgentResult:
    """Agent 的结构化最终结果。"""

    status: TaskStatus
    answer: str
    citations: List[Citation] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    clarification_question: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "gaps": self.gaps,
            "clarification_question": self.clarification_question,
            "trace": self.trace,
        }
