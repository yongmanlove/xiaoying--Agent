"""Step 7 测试：工具失败不崩溃，重试后转为 tool_error。"""
from pathlib import Path

import pytest

from agentqa.agent.react import ReActAgent
from agentqa.agent.status import TaskStatus
from agentqa.data.loader import load_chunks
from agentqa.llm.mock_client import MockLLMClient
from agentqa.rag.index import SparseIndex
from agentqa.tools.registry import ToolRegistry

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


class FailingSearchRegistry(ToolRegistry):
    """让 search_documents 恒返回超时错误，模拟工具故障。"""

    def execute(self, name, args):
        if name == "search_documents":
            return {"ok": False, "error_code": "TOOL_TIMEOUT", "message": "timeout"}
        return super().execute(name, args)


@pytest.fixture()
def agent():
    chunks = load_chunks(str(DOCS_DIR))
    registry = FailingSearchRegistry(SparseIndex.from_chunks(chunks), chunks)
    return ReActAgent(MockLLMClient(), registry)


def test_tool_failure_recovers_to_tool_error(agent):
    r = agent.run("报销期限多久？")
    # 不应崩溃，最终明确返回 tool_error
    assert r.status == TaskStatus.TOOL_ERROR
    # trace 中应体现两次失败的检索
    failed = [t for t in r.trace if "tool" in t and t["tool"] == "search_documents" and t["ok"] is False]
    assert len(failed) >= 2
