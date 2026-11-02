"""Step 1 测试：类型与状态机定义。"""
from agentqa.agent.status import AgentResult, Citation, TaskStatus
from agentqa.llm.base import AssistantTurn, ToolCall
from agentqa.rag.types import Chunk, RetrievedChunk


def test_task_status_enum_values():
    assert {s.value for s in TaskStatus} == {
        "answered",
        "insufficient_information",
        "conflict_needs_clarification",
        "tool_error",
        "max_steps_exceeded",
    }


def test_chunk_holds_doc_metadata():
    c = Chunk(
        chunk_id="D01#1",
        text="员工应在差旅费用发生后 30 天内提交报销申请。",
        doc_id="D01",
        version="v1.0",
        effective_date="2025-01-01",
        applies_to="",
        supersedes="",
        topic_key="travel_reimbursement",
        section="报销时限",
    )
    assert c.doc_id == "D01"
    assert c.version == "v1.0"
    assert c.effective_date == "2025-01-01"
    assert c.topic_key == "travel_reimbursement"


def test_retrieved_chunk_extends_chunk():
    r = RetrievedChunk(chunk_id="D01#1", text="x", doc_id="D01", score=8.31, rank=1)
    assert isinstance(r, Chunk)
    assert r.score == 8.31
    assert r.rank == 1


def test_agent_result_defaults():
    r = AgentResult(status=TaskStatus.ANSWERED, answer="答案是 30 天")
    assert r.citations == []
    assert r.gaps == []
    assert r.clarification_question == ""
    assert r.trace == []


def test_agent_result_serialization():
    r = AgentResult(
        status=TaskStatus.ANSWERED,
        answer="答案是 30 天",
        citations=[Citation(doc_id="D01", chunk_id="D01#1", quote="30 天")],
        gaps=[],
    )
    d = r.to_dict()
    assert d["status"] == "answered"
    assert d["citations"][0] == {"doc_id": "D01", "chunk_id": "D01#1", "quote": "30 天"}
    assert d["gaps"] == []


def test_assistant_turn_tool_calls():
    t = AssistantTurn(
        tool_calls=[ToolCall(id="1", name="search_documents", arguments={"query": "报销"})]
    )
    assert t.content == ""
    assert t.tool_calls[0].name == "search_documents"
    assert t.tool_calls[0].arguments == {"query": "报销"}
