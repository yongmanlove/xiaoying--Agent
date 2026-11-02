"""Step 0 冒烟测试：验证包可导入、默认配置正确。"""
from agentqa import __version__
from agentqa.config import LLMProvider, Settings


def test_package_imports_and_version():
    assert __version__ == "0.1.0"


def test_default_settings():
    s = Settings()
    assert s.llm_provider is LLMProvider.MOCK
    assert s.max_steps == 6
    assert s.max_tool_calls == 8
    assert s.search_top_k == 5


def test_settings_from_env_has_valid_provider():
    s = Settings.from_env()
    assert s.llm_provider in (LLMProvider.MOCK, LLMProvider.OPENAI)
