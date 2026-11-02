"""Step 5 测试：read_snippet、submit_answer、工具注册表。"""
from pathlib import Path

import pytest

from agentqa.data.loader import load_chunks
from agentqa.rag.index import SparseIndex
from agentqa.tools.read_snippet import read_snippet
from agentqa.tools.registry import ToolRegistry

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@pytest.fixture(scope="module")
def chunks():
    return load_chunks(str(DOCS_DIR))


@pytest.fixture(scope="module")
def chunks_by_id(chunks):
    return {c.chunk_id: c for c in chunks}


@pytest.fixture(scope="module")
def registry(chunks):
    return ToolRegistry(SparseIndex.from_chunks(chunks), chunks)


def test_read_snippet_returns_full_text_and_metadata(chunks_by_id):
    res = read_snippet(chunks_by_id, doc_id="D01", chunk_id="D01#1")
    assert res["ok"] is True
    assert res["doc_id"] == "D01"
    assert "30 天" in res["text"]
    assert res["version"] == "v1.0"
    assert res["effective_date"] == "2025-01-01"


def test_read_snippet_missing_returns_not_found(chunks_by_id):
    res = read_snippet(chunks_by_id, doc_id="D99", chunk_id="D99#1")
    assert res["ok"] is False
    assert res["error_code"] == "DOC_NOT_FOUND"


def test_registry_search_ok(registry):
    res = registry.execute("search_documents", {"query": "病假", "top_k": 5})
    assert res["ok"] is True


def test_registry_unknown_tool(registry):
    res = registry.execute("no_such_tool", {})
    assert res["ok"] is False
    assert res["error_code"] == "UNKNOWN_TOOL"


def test_registry_converts_exception_to_tool_error(registry):
    res = registry.execute("search_documents", {"query": "病假", "top_k": "not_a_number"})
    assert res["ok"] is False
    assert res["error_code"] == "TOOL_ERROR"


def test_submit_answer_valid_passes(registry):
    res = registry.execute(
        "submit_answer",
        {
            "status": "answered",
            "answer": "30 天",
            "citations": [{"doc_id": "D01", "chunk_id": "D01#1", "quote": "30 天内提交报销申请"}],
            "gaps": [],
        },
    )
    assert res["ok"] is True


def test_submit_answer_invalid_rejected(registry):
    res = registry.execute(
        "submit_answer",
        {"status": "answered", "answer": "x", "citations": [], "gaps": []},
    )
    assert res["ok"] is False
    assert res["error_code"] == "INVALID_SUBMISSION"
