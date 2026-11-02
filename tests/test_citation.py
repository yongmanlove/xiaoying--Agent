"""Step 5 测试：引用确定性校验。"""
from pathlib import Path

import pytest

from agentqa.agent.citation import normalize_whitespace, validate_submission
from agentqa.data.loader import load_chunks

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@pytest.fixture(scope="module")
def chunks_by_id():
    chunks = load_chunks(str(DOCS_DIR))
    return {c.chunk_id: c for c in chunks}


def test_normalize_whitespace_removes_all_whitespace():
    assert normalize_whitespace("30  天内\n提交 报销申请") == "30天内提交报销申请"


def test_legal_citation_passes(chunks_by_id):
    errors = validate_submission(
        "answered",
        [{"doc_id": "D01", "chunk_id": "D01#1", "quote": "30 天内提交报销申请"}],
        [],
        chunks_by_id,
    )
    assert errors == []


def test_whitespace_normalized_quote_passes(chunks_by_id):
    errors = validate_submission(
        "answered",
        [{"doc_id": "D01", "chunk_id": "D01#1", "quote": "30  天内\n提交 报销申请"}],
        [],
        chunks_by_id,
    )
    assert errors == []


def test_tampered_quote_rejected(chunks_by_id):
    errors = validate_submission(
        "answered",
        [{"doc_id": "D01", "chunk_id": "D01#1", "quote": "30 天内提交（虚构内容）"}],
        [],
        chunks_by_id,
    )
    assert any("不是原文子串" in e for e in errors)


def test_fake_chunk_id_rejected(chunks_by_id):
    errors = validate_submission(
        "answered",
        [{"doc_id": "D01", "chunk_id": "D99#9", "quote": "30 天"}],
        [],
        chunks_by_id,
    )
    assert any("不存在" in e for e in errors)


def test_answered_requires_at_least_one_citation(chunks_by_id):
    errors = validate_submission("answered", [], [], chunks_by_id)
    assert any("有效引用" in e for e in errors)


def test_insufficient_requires_gaps(chunks_by_id):
    errors = validate_submission("insufficient_information", [], [], chunks_by_id)
    assert any("gaps" in e for e in errors)


def test_invalid_status_rejected(chunks_by_id):
    errors = validate_submission("not_a_status", [], [], chunks_by_id)
    assert any("非法状态" in e for e in errors)
