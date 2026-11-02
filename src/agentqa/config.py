"""全局配置：环境变量读取与默认值。

所有可调参数集中于此，离线 Mock 模式无需任何环境变量即可运行。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv


class LLMProvider(str, Enum):
    """支持的模型提供方。"""

    MOCK = "mock"
    OPENAI = "openai"  # 任何 OpenAI 兼容端点（DeepSeek / DashScope / OpenAI 等）


@dataclass(frozen=True)
class Settings:
    """运行时配置，字段均可由 .env / 环境变量覆盖。"""

    llm_provider: LLMProvider = LLMProvider.MOCK
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "deepseek-chat"

    max_steps: int = 6
    max_tool_calls: int = 8

    docs_dir: str = "data/docs"
    chunk_size: int = 500
    chunk_overlap: int = 50
    search_top_k: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        """从项目根目录 .env 与环境变量构造配置，缺失时回退默认值。"""
        load_dotenv()
        provider_raw = os.getenv("LLM_PROVIDER", "mock").strip().lower()
        if provider_raw in {"dashscope", "deepseek", "openai", "openai-compatible"}:
            provider = LLMProvider.OPENAI
        else:
            provider = LLMProvider.MOCK

        return cls(
            llm_provider=provider,
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_model=os.getenv("LLM_MODEL", "deepseek-chat"),
            max_steps=int(os.getenv("MAX_STEPS", "6")),
            max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "8")),
            docs_dir=os.getenv("AGENTQA_DOCS_DIR", "data/docs"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            search_top_k=int(os.getenv("SEARCH_TOP_K", "5")),
        )
