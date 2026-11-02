"""命令行入口：python -m agentqa.cli ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentqa.agent.react import ReActAgent
from agentqa.agent.status import AgentResult
from agentqa.config import Settings
from agentqa.data.loader import load_chunks
from agentqa.llm.base import LLMClient
from agentqa.llm.mock_client import MockLLMClient
from agentqa.llm.openai_client import OpenAICompatLLMClient
from agentqa.rag.index import SparseIndex
from agentqa.tools.registry import FailingSearchRegistry, ToolRegistry


def build_llm_client(provider: str, settings: Settings) -> LLMClient:
    if provider == "mock":
        return MockLLMClient()
    if not settings.llm_api_key:
        raise ValueError("使用 openai provider 需要设置 LLM_API_KEY（或创建 .env）")
    return OpenAICompatLLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )


def build_registry(settings: Settings, fail_search: int = 0) -> ToolRegistry:
    chunks = load_chunks(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
    index = SparseIndex.from_chunks(chunks)
    if fail_search > 0:
        return FailingSearchRegistry(index, chunks, fail_times=fail_search)
    return ToolRegistry(index, chunks)


def build_agent(registry: ToolRegistry, provider: str, settings: Settings) -> ReActAgent:
    llm = build_llm_client(provider, settings)
    return ReActAgent(llm, registry, max_steps=settings.max_steps, max_tool_calls=settings.max_tool_calls)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, result: AgentResult) -> None:
    lines = ["# 状态: " + result.status.value, "", "## 回答", "", result.answer, ""]
    if result.citations:
        lines += ["## 引用", ""]
        lines += [f"- [{c.doc_id}] {c.chunk_id} — {c.quote}" for c in result.citations]
        lines.append("")
    if result.gaps:
        lines += ["## 资料缺口", ""] + [f"- {g}" for g in result.gaps] + [""]
    if result.clarification_question:
        lines += ["## 澄清问题", "", result.clarification_question, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_case(registry: ToolRegistry, provider: str, settings: Settings, case_path: Path) -> AgentResult:
    data = json.loads(case_path.read_text(encoding="utf-8"))
    agent = build_agent(registry, provider, settings)
    result = agent.run(data.get("question", ""), history=data.get("history"))
    out_dir = case_path.parent
    _write_json(out_dir / "result.json", result.to_dict())
    _write_json(out_dir / "trace.json", result.trace)
    _write_markdown(out_dir / "output.md", result)
    return result


def _run_batch(
    registry: ToolRegistry,
    provider: str,
    settings: Settings,
    questions_path: Path,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for i, item in enumerate(questions, start=1):
        # 每次用新 Mock，避免其内部状态在多个问题间泄漏
        agent = build_agent(registry, provider, settings)
        result = agent.run(item.get("question", ""), history=item.get("history"))
        case_dir = out_dir / f"case{i:02d}"
        _write_json(case_dir / "result.json", result.to_dict())
        _write_json(case_dir / "trace.json", result.trace)
        _write_markdown(case_dir / "output.md", result)
        summary.append({"index": i, "question": item.get("question", ""), "status": result.status.value})
    _write_json(out_dir / "summary.json", summary)
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agentqa", description="企业知识问答 Agent CLI")
    p.add_argument("--question", help="单条问题")
    p.add_argument("--case", help="用例 input.json 路径")
    p.add_argument("--batch", help="批量 questions.json 路径")
    p.add_argument("--provider", choices=["mock", "openai"], default="mock")
    p.add_argument("--out", help="批量输出目录")
    p.add_argument("--trace", help="单条问题的 trace 输出文件路径")
    p.add_argument("--fail-search", type=int, default=0, help="让 search_documents 前 N 次失败（演示工具失败路径）")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()

    if args.question:
        registry = build_registry(settings, args.fail_search)
        agent = build_agent(registry, args.provider, settings)
        result = agent.run(args.question)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        if args.trace:
            _write_json(Path(args.trace), result.trace)
        return 0

    if args.case:
        registry = build_registry(settings, args.fail_search)
        _run_case(registry, args.provider, settings, Path(args.case))
        return 0

    if args.batch:
        registry = build_registry(settings, args.fail_search)
        out_dir = Path(args.out) if args.out else Path("examples/mock")
        summary = _run_batch(registry, args.provider, settings, Path(args.batch), out_dir)
        for s in summary:
            print(f"{s['index']:02d} [{s['status']}] {s['question']}")
        return 0

    print("请提供 --question / --case / --batch 之一", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
