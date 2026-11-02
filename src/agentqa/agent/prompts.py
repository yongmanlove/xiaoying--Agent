"""系统提示词与工具 JSON Schema。"""
from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT = """你是企业制度问答助手。你只能依据检索到的制度资料作答，不得编造。

工作方式（ReAct）：
1. 先调用 search_documents 检索；引用任何原文前，必须先调用 read_snippet 读取完整片段。
2. 若检索结果中出现 conflict_candidates，必须先读取冲突双方的原文，再决定如何作答。
3. 若同一问题存在多篇有效文档，且无法依据版本/生效日期/适用范围判定适用哪份，则 status=conflict_needs_clarification：列出双方说法与引用，提出确认问题，禁止擅自选择其一。
4. 若版本/生效日期/适用范围足以裁决（例如某文档 supersedes 另一篇、生效日期更晚），说明冲突并给出适用结论，同时引用新旧文档。
5. 若资料中确实没有答案，则 status=insufficient_information，在 gaps 中列出缺口，不得编造。
6. 工具失败时，可更换查询重试一次；仍失败则 status=tool_error。
7. 追问场景先结合对话历史改写问题，再检索。
8. 最终必须调用 submit_answer 提交结构化结果；引用 quote 必须逐字来自 read_snippet 返回的原文。
"""


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "按关键词检索制度片段，返回文档 ID、版本、生效日期与摘要片段；不返回全文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询语句"},
                    "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_snippet",
            "description": "读取指定文档片段的完整原文与元数据，引用前必须调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "文档 ID"},
                    "chunk_id": {"type": "string", "description": "切片 ID"},
                    "section": {"type": "string", "description": "小节标题"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "提交结构化最终答案（终止动作）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [
                            "answered",
                            "insufficient_information",
                            "conflict_needs_clarification",
                            "tool_error",
                        ],
                    },
                    "answer": {"type": "string", "description": "最终回答"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc_id": {"type": "string"},
                                "chunk_id": {"type": "string"},
                                "quote": {"type": "string", "description": "原文逐字子串"},
                            },
                            "required": ["doc_id", "chunk_id", "quote"],
                        },
                    },
                    "gaps": {"type": "array", "items": {"type": "string"}},
                    "clarification_question": {"type": "string"},
                },
                "required": ["status", "answer", "citations", "gaps"],
            },
        },
    },
]
