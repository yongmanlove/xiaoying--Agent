"""Step 10 测试：OpenAI 兼容客户端响应转换（离线，不发网络请求）。"""
from types import SimpleNamespace

from agentqa.llm.base import LLMClient
from agentqa.llm.openai_client import OpenAICompatLLMClient


def test_implements_llm_client_protocol():
    client = OpenAICompatLLMClient(api_key="sk-test", base_url="https://api.example.com/v1", model="m")
    assert isinstance(client, LLMClient)


def test_to_assistant_turn_with_tool_calls():
    msg = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="search_documents", arguments='{"query": "报销"}'),
            )
        ],
    )
    turn = OpenAICompatLLMClient._to_assistant_turn(msg)
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "search_documents"
    assert turn.tool_calls[0].arguments == {"query": "报销"}


def test_to_assistant_turn_content_only():
    msg = SimpleNamespace(content="直接回答", tool_calls=None)
    turn = OpenAICompatLLMClient._to_assistant_turn(msg)
    assert turn.content == "直接回答"
    assert turn.tool_calls == []


def test_to_assistant_turn_bad_arguments_json():
    msg = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="x", arguments="not json"))],
    )
    turn = OpenAICompatLLMClient._to_assistant_turn(msg)
    assert turn.tool_calls[0].arguments == {}
