"""文档加载与切分。

纯 Python 实现，不依赖 langchain：
- 手写解析 frontmatter（--- 包裹的 key: value 标量）；
- 按 # / ## / ### / #### 标题切分；
- 超长片段按 chunk_size / chunk_overlap 二次切分；
- 每个 chunk 注入 doc_id/version/effective_date/topic_key 等元数据。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from agentqa.rag.types import Chunk

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """解析 markdown frontmatter，返回 (元数据, 正文)。"""
    if not text.startswith("---"):
        return {}, text
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta, m.group(2)


def split_by_headers(body: str) -> List[Tuple[str, str]]:
    """按 Markdown 标题把正文切分为 (标题, 内容) 片段列表。"""
    matches = list(_HEADER_RE.finditer(body))
    if not matches:
        stripped = body.strip()
        return [("", stripped)] if stripped else []
    sections: List[Tuple[str, str]] = []
    pre = body[: matches[0].start()].strip()
    if pre:
        sections.append(("", pre))
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        if content:
            sections.append((title, content))
    return sections


def split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """按固定窗口切分长文本，相邻片段保留 overlap 字符重叠。"""
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - chunk_overlap)
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        chunks.append(text[start : start + chunk_size])
        if start + chunk_size >= n:
            break
        start += step
    return chunks


def load_chunks(docs_dir: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Chunk]:
    """加载目录下所有 .md 文件并切分为带元数据的 Chunk 列表。"""
    root = Path(docs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"语料目录不存在: {docs_dir}")
    chunks: List[Chunk] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        doc_id = meta.get("doc_id") or path.stem
        idx = 0
        for section_title, content in split_by_headers(body):
            for piece in split_long_text(content, chunk_size, chunk_overlap):
                idx += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}#{idx}",
                        text=piece,
                        doc_id=doc_id,
                        title=meta.get("title", ""),
                        version=meta.get("version", ""),
                        effective_date=meta.get("effective_date", ""),
                        department=meta.get("department", ""),
                        applies_to=meta.get("applies_to", ""),
                        supersedes=meta.get("supersedes", ""),
                        topic_key=meta.get("topic_key", ""),
                        status=meta.get("status", ""),
                        section=section_title,
                    )
                )
    return chunks
