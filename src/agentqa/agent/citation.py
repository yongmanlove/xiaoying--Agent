"""引用校验：保证引用的 doc_id/chunk_id 存在且 quote 是原文子串。"""
from __future__ import annotations

from typing import Dict, List

from agentqa.agent.status import TaskStatus
from agentqa.rag.types import Chunk

_VALID_STATUSES = {s.value for s in TaskStatus}


def normalize_whitespace(text: str) -> str:
    """忽略一切空白差异（空格、换行、全角空格等），保留标点。"""
    return "".join(str(text).split())


def validate_submission(
    status: str,
    citations: List[Dict[str, str]],
    gaps: List[str],
    chunks_by_id: Dict[str, Chunk],
) -> List[str]:
    """对一次提交做确定性校验，返回错误列表（空列表 = 通过）。

    五项规则：
    1. status 合法；
    2. 每条引用的 chunk_id 存在且 doc_id 与 chunk 一致；
    3. quote 是该 chunk 原文的子串（忽略空白差异）；
    4. answered / conflict_needs_clarification 至少一条有效引用；
    5. insufficient_information 必须给出 gaps。
    """
    errors: List[str] = []
    if status not in _VALID_STATUSES:
        return [f"非法状态: {status}"]

    citations = citations or []
    gaps = gaps or []
    valid_ref_count = 0

    for i, c in enumerate(citations):
        doc_id = c.get("doc_id", "")
        chunk_id = c.get("chunk_id", "")
        quote = c.get("quote", "")
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            errors.append(f"引用 {i}: chunk_id 不存在 -> {chunk_id}")
            continue
        if doc_id and chunk.doc_id != doc_id:
            errors.append(f"引用 {i}: doc_id 与 chunk 不匹配 -> {doc_id}")
            continue
        if not quote:
            errors.append(f"引用 {i}: quote 为空")
            continue
        if normalize_whitespace(quote) not in normalize_whitespace(chunk.text):
            errors.append(f"引用 {i}: quote 不是原文子串")
            continue
        valid_ref_count += 1

    if status in (TaskStatus.ANSWERED.value, TaskStatus.CONFLICT_NEEDS_CLARIFICATION.value):
        if valid_ref_count == 0:
            errors.append(f"{status} 状态至少需要一条有效引用")

    if status == TaskStatus.INSUFFICIENT_INFORMATION.value and not gaps:
        errors.append("insufficient_information 状态必须给出 gaps")

    return errors
