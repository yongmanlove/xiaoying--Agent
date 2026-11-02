"""离线 Mock 模型：按最近一次工具观察结果分支，驱动完整 ReAct 流程。

Mock 不是固定回答，而是脚本化的决策器：根据 observation 的 ok / 命中 /
冲突候选，决定下一步调用哪个工具或提交什么状态。仅用于验证流程与状态机，
不证明真实模型决策效果。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agentqa.llm.base import AssistantTurn, ToolCall

_QUOTES: Dict[str, str] = {
    "D01": "30 天内提交报销申请",
    "D02": "15 天内提交报销申请",
    "D03": "10 天带薪年假",
    "D04": "15 天带薪年假",
    "D05": "至少提前 24 小时",
}

_ANSWERS: Dict[str, str] = {
    "D01": "根据 D01 号制度，报销期限为 30 天。",
    "D02": "根据 D02 号制度，报销期限为 15 天。",
    "D03": "根据 D03 号制度，年假为 10 天。",
    "D04": "根据 D04 号制度，年假为 15 天。",
    "D05": "病假需至少提前 24 小时申请。",
}


class MockLLMClient:
    """实现 LLMClient 协议的离线脚本模型。"""

    def __init__(self) -> None:
        self._last_called: Optional[str] = None
        self._search_failures: int = 0
        self._conflict_doc_ids: List[str] = []
        self._call_seq: int = 0

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> AssistantTurn:
        question = self._find_question(messages)
        if self._last_called is None:
            return self._tool_call("search_documents", {"query": question})
        obs = self._last_observation(messages)
        return self._decide(self._last_called, obs, question)

    # ---- 决策分支 ----

    def _decide(self, name: str, obs: Dict[str, Any], question: str) -> AssistantTurn:
        if name == "search_documents":
            return self._after_search(obs, question)
        if name == "read_snippet":
            return self._after_read(obs)
        return self._submit("tool_error", "未知工具结果", gaps=["工具返回无法识别"])

    def _after_search(self, obs: Dict[str, Any], question: str) -> AssistantTurn:
        if not obs.get("ok"):
            self._search_failures += 1
            if self._search_failures >= 2:
                return self._submit(
                    "tool_error", "检索工具不可用", gaps=["search_documents 反复失败"]
                )
            return self._tool_call("search_documents", {"query": self._rewrite(question)})

        results = obs.get("results") or []
        conflicts = obs.get("conflict_candidates") or []
        if conflicts:
            doc_ids = list(conflicts[0].get("doc_ids", []))
            self._conflict_doc_ids = doc_ids
            return self._tool_call("read_snippet", {"doc_id": doc_ids[0]})
        if results:
            self._conflict_doc_ids = []
            first = results[0]
            return self._tool_call(
                "read_snippet", {"doc_id": first["doc_id"], "chunk_id": first["chunk_id"]}
            )
        # 命中为空 → 改写重试一次，再空则资料不足
        self._search_failures += 1
        if self._search_failures >= 2:
            return self._submit(
                "insufficient_information",
                "资料中未找到相关答案",
                gaps=["资料中未包含该问题相关内容"],
            )
        return self._tool_call("search_documents", {"query": self._rewrite(question)})

    def _after_read(self, obs: Dict[str, Any]) -> AssistantTurn:
        if not obs.get("ok"):
            return self._submit("tool_error", "读取原文失败", gaps=[obs.get("message", "")])
        doc_id = obs.get("doc_id", "")
        # 冲突路径：读第一篇后继续读第二篇
        if (
            self._conflict_doc_ids
            and doc_id == self._conflict_doc_ids[0]
            and len(self._conflict_doc_ids) > 1
        ):
            return self._tool_call("read_snippet", {"doc_id": self._conflict_doc_ids[1]})
        # 冲突路径：两篇都读完后提交冲突
        if self._conflict_doc_ids:
            citations = [self._citation(d) for d in self._conflict_doc_ids]
            return self._submit(
                "conflict_needs_clarification",
                answer="发现多篇同主题文档给出了不同规定，无法判定适用哪份，需向您确认。",
                citations=citations,
                clarification_question="请问您适用哪一份制度？",
            )
        # 正常路径：单文档作答
        return self._submit(
            "answered",
            answer=self._answer_for(doc_id),
            citations=[self._citation(doc_id)],
        )

    # ---- 辅助 ----

    def _citation(self, doc_id: str) -> Dict[str, str]:
        return {"doc_id": doc_id, "chunk_id": f"{doc_id}#1", "quote": _QUOTES.get(doc_id, "")}

    @staticmethod
    def _answer_for(doc_id: str) -> str:
        return _ANSWERS.get(doc_id, f"根据 {doc_id} 号制度作答。")

    @staticmethod
    def _rewrite(question: str) -> str:
        # 只去掉疑问语气词，不追加无关关键词，避免改写后误命中无关文档
        return question.rstrip("？?吗呢啊 ").strip() or question

    @staticmethod
    def _find_question(messages: List[Dict[str, Any]]) -> str:
        question = ""
        for m in messages:
            if m.get("role") == "user":
                question = str(m.get("content", ""))
        return question

    @staticmethod
    def _last_observation(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        for m in reversed(messages):
            if m.get("role") == "tool":
                content = m.get("content", "")
                try:
                    return json.loads(content)
                except (TypeError, json.JSONDecodeError):
                    return {"ok": False, "error_code": "PARSE_ERROR", "message": str(content)}
        return {}

    def _tool_call(self, name: str, arguments: Dict[str, Any]) -> AssistantTurn:
        self._last_called = name
        self._call_seq += 1
        return AssistantTurn(
            tool_calls=[ToolCall(id=f"call_{self._call_seq}", name=name, arguments=arguments)]
        )

    def _submit(
        self,
        status: str,
        answer: str,
        citations: Optional[List[Dict[str, str]]] = None,
        gaps: Optional[List[str]] = None,
        clarification_question: str = "",
    ) -> AssistantTurn:
        self._last_called = "submit_answer"
        self._call_seq += 1
        return AssistantTurn(
            tool_calls=[
                ToolCall(
                    id=f"call_{self._call_seq}",
                    name="submit_answer",
                    arguments={
                        "status": status,
                        "answer": answer,
                        "citations": citations or [],
                        "gaps": gaps or [],
                        "clarification_question": clarification_question,
                    },
                )
            ]
        )
