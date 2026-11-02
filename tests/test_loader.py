"""Step 3 测试：frontmatter 解析、标题切分、长文二次切分、元数据传播。"""
import re
from pathlib import Path

from agentqa.data.loader import (
    load_chunks,
    parse_frontmatter,
    split_by_headers,
    split_long_text,
)

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def test_parse_frontmatter_scalars_and_empty_quotes():
    text = '---\ndoc_id: D01\ntitle: 差旅\nauthor: ""\n---\n\n# 正文\n'
    meta, body = parse_frontmatter(text)
    assert meta["doc_id"] == "D01"
    assert meta["title"] == "差旅"
    assert meta["author"] == ""
    assert "# 正文" in body


def test_split_by_headers_returns_titles_and_contents():
    body = "# 标题\n\n## 第一节\n内容一\n\n## 第二节\n内容二\n"
    secs = split_by_headers(body)
    assert [s for s, _ in secs] == ["第一节", "第二节"]
    assert "内容一" in secs[0][1]


def test_split_long_text_has_overlap():
    text = "字" * 1000
    parts = split_long_text(text, chunk_size=500, chunk_overlap=50)
    assert len(parts) >= 2
    for a, b in zip(parts, parts[1:]):
        assert a[-50:] == b[:50]


def test_load_chunks_unique_and_well_formed_ids():
    chunks = load_chunks(str(DOCS_DIR))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"D\d{2}#\d+", i) for i in ids)


def test_doc_metadata_propagates_to_every_chunk():
    chunks = load_chunks(str(DOCS_DIR))
    d01 = [c for c in chunks if c.doc_id == "D01"]
    assert d01
    assert all(c.topic_key == "travel_reimbursement" for c in d01)
    assert all(c.version == "v1.0" for c in d01)
    assert all(c.effective_date == "2025-01-01" for c in d01)

    d02 = next(c for c in chunks if c.doc_id == "D02")
    assert d02.version == "v2.0"
    assert d02.applies_to == ""


def test_section_title_is_preserved():
    chunks = load_chunks(str(DOCS_DIR))
    d01_sections = {c.section for c in chunks if c.doc_id == "D01"}
    assert "报销时限" in d01_sections
    assert "报销范围" in d01_sections
