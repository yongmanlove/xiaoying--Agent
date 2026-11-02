"""冲突候选检测：只提供观察提示，不替模型裁决。"""
from __future__ import annotations

from typing import Any, Dict, List

from agentqa.rag.types import RetrievedChunk


def detect_conflict_candidates(hits: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    """按 topic_key 分组，同主题下出现 ≥2 个不同 doc_id 时返回候选组。

    仅作为检索观察的一部分返回给模型，由模型决定是否读取双方原文、
    判定可否依据版本/生效日期裁决，或需要向用户追问。
    """
    by_topic: Dict[str, Dict[str, RetrievedChunk]] = {}
    for h in hits:
        if not h.topic_key:
            continue
        by_topic.setdefault(h.topic_key, {})[h.doc_id] = h

    candidates: List[Dict[str, Any]] = []
    for topic_key, docs in by_topic.items():
        if len(docs) >= 2:
            candidates.append(
                {
                    "topic_key": topic_key,
                    "doc_ids": sorted(docs.keys()),
                    "hint": "同主题存在多篇有效文档，事实值可能不一致，建议逐篇读取原文",
                }
            )
    return candidates
