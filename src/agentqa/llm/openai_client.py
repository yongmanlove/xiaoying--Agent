"""OpenAI 兼容模型客户端：适用于 DeepSeek / DashScope / OpenAI 等兼容端点。"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

from agentqa.llm.base import AssistantTurn, ToolCall


class OpenAICompatLLMClient:
    """通过 OpenAI 兼容接口调用远程模型。"""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> AssistantTurn:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools or None,
            temperature=temperature,
        )
        return self._to_assistant_turn(resp.choices[0].message)

    @staticmethod
    def _to_assistant_turn(message: Any) -> AssistantTurn:
        tool_calls: List[ToolCall] = []
        for tc in message.tool_calls or []:
            args: Dict[str, Any] = {}
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return AssistantTurn(content=message.content or "", tool_calls=tool_calls)
