"""Step 7 测试：步数 / 工具调用次数上限触发 max_steps_exceeded。"""
from agentqa.agent.react import ReActAgent
from agentqa.agent.status import TaskStatus
from agentqa.llm.base import AssistantTurn, ToolCall


class LoopingLLMClient:
    """每轮都返回一个 search_documents 调用，永不提交。"""

    def chat(self, messages, tools, temperature=0.0):
        return AssistantTurn(
            tool_calls=[ToolCall(id="loop", name="search_documents", arguments={"query": "x"})]
        )


class MultiToolLLMClient:
    """每轮返回多个工具调用。"""

    def chat(self, messages, tools, temperature=0.0):
        return AssistantTurn(
            tool_calls=[
                ToolCall(id="a", name="search_documents", arguments={"query": "x"}),
                ToolCall(id="b", name="search_documents", arguments={"query": "x"}),
                ToolCall(id="c", name="search_documents", arguments={"query": "x"}),
            ]
        )


class DummyRegistry:
    def execute(self, name, args):
        return {"ok": True, "results": [], "conflict_candidates": []}


def test_max_steps_exceeded():
    agent = ReActAgent(LoopingLLMClient(), DummyRegistry(), max_steps=3, max_tool_calls=100)
    r = agent.run("任意问题")
    assert r.status == TaskStatus.MAX_STEPS_EXCEEDED
    assert len(r.trace) >= 3


def test_max_tool_calls_exceeded():
    agent = ReActAgent(MultiToolLLMClient(), DummyRegistry(), max_steps=10, max_tool_calls=2)
    r = agent.run("任意问题")
    assert r.status == TaskStatus.MAX_STEPS_EXCEEDED
