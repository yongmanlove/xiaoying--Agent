"""Step 7 测试：ReAct 主循环端到端（Mock 模型 + 真实检索/工具）。"""
from pathlib import Path

import pytest

from agentqa.agent.react import ReActAgent
from agentqa.agent.status import TaskStatus
from agentqa.data.loader import load_chunks
from agentqa.llm.mock_client import MockLLMClient
from agentqa.rag.index import SparseIndex
from agentqa.tools.registry import ToolRegistry

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@pytest.fixture(scope="module")
def registry():
    chunks = load_chunks(str(DOCS_DIR))
    return ToolRegistry(SparseIndex.from_chunks(chunks), chunks)


@pytest.fixture()
def agent(registry):
    # 每个用例用全新的 Mock（Mock 有状态），registry 可复用（无状态）
    return ReActAgent(MockLLMClient(), registry)


def test_normal_question_answered(agent):
    r = agent.run("病假需要提前多久申请？")
    assert r.status == TaskStatus.ANSWERED
    assert r.citations and r.citations[0].doc_id == "D05"
    assert "24 小时" in r.answer


def test_conflict_question_asks_clarification(agent):
    r = agent.run("报销期限多久？")
    assert r.status == TaskStatus.CONFLICT_NEEDS_CLARIFICATION
    assert {c.doc_id for c in r.citations} == {"D01", "D02"}
    assert r.clarification_question


def test_paraphrase_question_triggers_conflict(agent):
    r = agent.run("出差回来多久内要报销？")
    assert r.status == TaskStatus.CONFLICT_NEEDS_CLARIFICATION
    assert {c.doc_id for c in r.citations} == {"D01", "D02"}


def test_out_of_scope_question_insufficient(agent):
    r = agent.run("公司给宠物买保险吗？")
    assert r.status == TaskStatus.INSUFFICIENT_INFORMATION
    assert r.gaps


def test_trace_records_full_pipeline(agent):
    r = agent.run("病假需要提前多久申请？")
    tools = [t["tool"] for t in r.trace if "tool" in t]
    assert "search_documents" in tools
    assert "read_snippet" in tools
    assert "submit_answer" in tools


def test_followup_question_uses_latest_user_message(agent):
    history = [
        {"role": "user", "content": "年假有多少天？"},
        {"role": "assistant", "content": "根据 D04 号制度，年假为 15 天。"},
    ]
    r = agent.run("那病假呢？", history=history)
    assert r.status == TaskStatus.ANSWERED
    assert r.citations and r.citations[0].doc_id == "D05"
