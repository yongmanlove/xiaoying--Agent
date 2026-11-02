"""模型客户端统一接口。

真实模型（DashScope）与离线 Mock 都实现 LLMClient 协议，
ReAct 主循环不感知具体实现差异。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """一次模型发起的工具调用。"""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantTurn:
    """模型的一轮输出：正文 + 可选的工具调用。"""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)


@runtime_checkable
class LLMClient(Protocol):
    """模型客户端协议。

    messages 为标准 OpenAI 风格消息列表（system/user/assistant/tool）。
    tools 为工具 JSON Schema 列表。
    """

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> AssistantTurn:
        ...
