"""Step 4 测试：检索命中、无命中、摘要返回、冲突候选。"""
from pathlib import Path

import pytest

from agentqa.data.loader import load_chunks
from agentqa.rag.conflict import detect_conflict_candidates
from agentqa.rag.index import SparseIndex
from agentqa.rag.types import RetrievedChunk
from agentqa.tools.search_documents import search_documents

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@pytest.fixture(scope="module")
def index():
    chunks = load_chunks(str(DOCS_DIR))
    return SparseIndex.from_chunks(chunks)


def _doc_ids(hits):
    return [h.doc_id for h in hits]


def test_search_paraphrase_hits_d01_and_d02(index):
    hits = index.search("出差回来多久内要报销", top_k=5)
    ids = _doc_ids(hits)
    assert "D01" in ids
    assert "D02" in ids


def test_search_sick_leave_hits_d05(index):
    hits = index.search("病假需要提前多久申请", top_k=5)
    assert "D05" in _doc_ids(hits)


def test_search_out_of_scope_returns_no_hit(index):
    hits = index.search("公司给宠物买保险吗", top_k=5)
    assert hits == []


def test_search_documents_returns_snippet_not_full_text(index):
    res = search_documents(index, "报销期限", top_k=3)
    assert res["ok"] is True
    assert res["results"]
    for r in res["results"]:
        assert "snippet" in r
        assert "text" not in r  # 不返回全文字段


def test_conflict_candidates_group_d01_d02(index):
    hits = index.search("报销期限多久", top_k=5)
    cands = detect_conflict_candidates(hits)
    tr = next(c for c in cands if c["topic_key"] == "travel_reimbursement")
    assert set(tr["doc_ids"]) == {"D01", "D02"}


def test_conflict_candidates_different_topic_not_grouped():
    hits = [
        RetrievedChunk(chunk_id="D01#1", text="x", doc_id="D01", topic_key="travel_reimbursement"),
        RetrievedChunk(chunk_id="D10#1", text="x", doc_id="D10", topic_key="travel_lodging"),
    ]
    assert detect_conflict_candidates(hits) == []


def test_conflict_candidates_single_doc_no_group():
    hits = [
        RetrievedChunk(chunk_id="D01#1", text="x", doc_id="D01", topic_key="travel_reimbursement"),
    ]
    assert detect_conflict_candidates(hits) == []
