"""ReAct 主循环：模型选择动作 → 执行工具 → 观察结果 → 调整下一步或结束。"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agentqa.agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS
from agentqa.agent.status import AgentResult, Citation, TaskStatus
from agentqa.llm.base import AssistantTurn, LLMClient
from agentqa.tools.registry import ToolRegistry


class ReActAgent:
    """驱动模型与工具之间的多轮交互，带步数/工具次数上限与 trace 记录。"""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        max_steps: int = 6,
        max_tool_calls: int = 8,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls

    def run(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentResult:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history or []:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": question})

        trace: List[Dict[str, Any]] = []
        tool_call_count = 0

        for step in range(1, self._max_steps + 1):
            turn: AssistantTurn = self._llm.chat(messages, TOOL_SCHEMAS, temperature=0.0)
            trace.append(
                {
                    "step": step,
                    "content": turn.content,
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments} for tc in turn.tool_calls
                    ],
                }
            )

            if not turn.tool_calls:
                # 模型未调用工具：视为最终自然语言回答，做结构化兜底
                return self._wrap_final(turn.content, trace)

            messages.append(self._assistant_message(turn))

            for tc in turn.tool_calls:
                if tool_call_count >= self._max_tool_calls:
                    return AgentResult(
                        status=TaskStatus.MAX_STEPS_EXCEEDED,
                        answer="工具调用次数达到上限",
                        gaps=[],
                        trace=trace,
                    )
                tool_call_count += 1
                obs = self._registry.execute(tc.name, tc.arguments)
                trace.append(
                    {
                        "step": step,
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "ok": bool(obs.get("ok")),
                        "summary": self._summarize(obs),
                    }
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(obs, ensure_ascii=False)}
                )

                if tc.name == "submit_answer" and obs.get("ok"):
                    return self._result_from_obs(obs, trace)

        return AgentResult(
            status=TaskStatus.MAX_STEPS_EXCEEDED,
            answer="执行步数达到上限",
            gaps=[],
            trace=trace,
        )

    # ---- 内部辅助 ----

    @staticmethod
    def _assistant_message(turn: AssistantTurn) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in turn.tool_calls
            ]
        return msg

    @staticmethod
    def _result_from_obs(obs: Dict[str, Any], trace: List[Dict[str, Any]]) -> AgentResult:
        citations = [
            Citation(
                doc_id=c.get("doc_id", ""),
                chunk_id=c.get("chunk_id", ""),
                quote=c.get("quote", ""),
            )
            for c in obs.get("citations", [])
        ]
        return AgentResult(
            status=TaskStatus(obs.get("status", "")),
            answer=obs.get("answer", ""),
            citations=citations,
            gaps=obs.get("gaps", []),
            clarification_question=obs.get("clarification_question", ""),
            trace=trace,
        )

    @staticmethod
    def _wrap_final(content: str, trace: List[Dict[str, Any]]) -> AgentResult:
        content = (content or "").strip()
        return AgentResult(
            status=TaskStatus.ANSWERED,
            answer=content or "(空回答)",
            citations=[],
            gaps=[],
            trace=trace,
        )

    @staticmethod
    def _summarize(obs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(obs, dict):
            return {"ok": False, "message": str(obs)}
        if "results" in obs or "conflict_candidates" in obs:
            return {
                "ok": obs.get("ok"),
                "hit_count": len(obs.get("results") or []),
                "conflict_doc_ids": [c.get("doc_ids") for c in obs.get("conflict_candidates") or []],
            }
        if "text" in obs:
            return {"ok": obs.get("ok"), "doc_id": obs.get("doc_id"), "chunk_id": obs.get("chunk_id")}
        if "answer" in obs and "status" in obs:
            return {"ok": obs.get("ok"), "status": obs.get("status"), "errors": obs.get("errors", [])}
        return {"ok": obs.get("ok"), "error_code": obs.get("error_code"), "message": obs.get("message", "")}
