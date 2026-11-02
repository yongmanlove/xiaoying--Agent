"""Step 6 测试：Mock 模型按 observation 分支。"""
import json

from agentqa.llm.base import LLMClient
from agentqa.llm.mock_client import MockLLMClient


def _user(q):
    return {"role": "user", "content": q}


def _tool(obs):
    return {"role": "tool", "tool_call_id": "t", "content": json.dumps(obs, ensure_ascii=False)}


def _only(mock, messages):
    turn = mock.chat(messages, [], 0.0)
    assert len(turn.tool_calls) == 1
    return turn.tool_calls[0]


def test_implements_llm_client_protocol():
    assert isinstance(MockLLMClient(), LLMClient)


def test_initial_turn_calls_search():
    mock = MockLLMClient()
    call = _only(mock, [_user("报销期限多久？")])
    assert call.name == "search_documents"
    assert call.arguments["query"] == "报销期限多久？"


def test_search_hit_then_read_first_result():
    mock = MockLLMClient()
    _only(mock, [_user("病假需要提前多久申请？")])  # search
    obs = {
        "ok": True,
        "results": [{"doc_id": "D05", "chunk_id": "D05#1", "snippet": "..."}],
        "conflict_candidates": [],
    }
    call = _only(mock, [_user("病假需要提前多久申请？"), _tool(obs)])
    assert call.name == "read_snippet"
    assert call.arguments["doc_id"] == "D05"
    assert call.arguments["chunk_id"] == "D05#1"


def test_conflict_path_reads_both_then_submits_conflict():
    mock = MockLLMClient()
    q = "报销期限多久？"
    _only(mock, [_user(q)])  # search
    obs_search = {
        "ok": True,
        "results": [],
        "conflict_candidates": [{"topic_key": "travel_reimbursement", "doc_ids": ["D01", "D02"]}],
    }
    c1 = _only(mock, [_user(q), _tool(obs_search)])
    assert c1.name == "read_snippet" and c1.arguments["doc_id"] == "D01"

    c2 = _only(mock, [_user(q), _tool({"ok": True, "doc_id": "D01", "chunk_id": "D01#1", "text": "30 天"})])
    assert c2.name == "read_snippet" and c2.arguments["doc_id"] == "D02"

    c3 = _only(mock, [_user(q), _tool({"ok": True, "doc_id": "D02", "chunk_id": "D02#1", "text": "15 天"})])
    assert c3.name == "submit_answer"
    assert c3.arguments["status"] == "conflict_needs_clarification"
    assert len(c3.arguments["citations"]) == 2
    assert c3.arguments["clarification_question"]


def test_read_then_submit_answered():
    mock = MockLLMClient()
    q = "病假需要提前多久申请？"
    _only(mock, [_user(q)])  # search
    _only(mock, [_user(q), _tool({"ok": True, "results": [{"doc_id": "D05", "chunk_id": "D05#1"}], "conflict_candidates": []})])  # read
    c = _only(mock, [_user(q), _tool({"ok": True, "doc_id": "D05", "chunk_id": "D05#1", "text": "至少提前 24 小时"})])
    assert c.name == "submit_answer"
    assert c.arguments["status"] == "answered"
    assert c.arguments["citations"][0]["doc_id"] == "D05"


def test_search_failure_retries_then_tool_error():
    mock = MockLLMClient()
    q = "报销期限多久？"
    _only(mock, [_user(q)])  # search
    c1 = _only(mock, [_user(q), _tool({"ok": False, "error_code": "TOOL_TIMEOUT", "message": "timeout"})])
    assert c1.name == "search_documents"  # 改写重试

    c2 = _only(mock, [_user(q), _tool({"ok": False, "error_code": "TOOL_TIMEOUT", "message": "timeout"})])
    assert c2.name == "submit_answer"
    assert c2.arguments["status"] == "tool_error"


def test_empty_results_then_insufficient():
    mock = MockLLMClient()
    q = "公司给宠物买保险吗？"
    _only(mock, [_user(q)])  # search
    c1 = _only(mock, [_user(q), _tool({"ok": True, "results": [], "conflict_candidates": []})])
    assert c1.name == "search_documents"  # 改写重试

    c2 = _only(mock, [_user(q), _tool({"ok": True, "results": [], "conflict_candidates": []})])
    assert c2.name == "submit_answer"
    assert c2.arguments["status"] == "insufficient_information"
    assert c2.arguments["gaps"]
