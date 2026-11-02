# 02 · 解决方案

## 总体架构

```
CLI (cli.py)
  └─ ReActAgent.run()            agent/react.py
       ├─ LLMClient               llm/（Mock 或 OpenAI 兼容）
       ├─ ToolRegistry.execute()  tools/registry.py
       │    ├─ search_documents   tools/search_documents.py
       │    ├─ read_snippet       tools/read_snippet.py
       │    └─ submit_answer      tools/submit_answer.py
       └─ 检索内核                rag/（BM25 + 冲突候选）
            └─ 语料切分            data/loader.py
```

## 调用流程（ReAct 主循环）

```text
messages = [system, history..., user_question]
for step in 1..MAX_STEPS:
    turn = llm.chat(messages, TOOL_SCHEMAS)
    if 无 tool_calls: 返回自然语言兜底
    for 每个 tool_call:
        obs = registry.execute(name, args)      # 异常统一转 ok=false
        messages.append(tool_observation(obs))
        if name == submit_answer and obs.ok:     # 终止动作
            return AgentResult(...)
超限 → max_steps_exceeded
```

- 所有动作由模型发起，**应用层不强制首轮检索、不自动裁决冲突**。
- 工具异常不抛出主循环，统一转 `{ok:false, error_code, message}`。
- `MAX_STEPS=6`、`MAX_TOOL_CALLS=8`，超限返回明确未完成状态 `max_steps_exceeded`。

## 三个工具

| 工具 | 行为 | 关键约束 |
|---|---|---|
| `search_documents` | 返回摘要片段 + `conflict_candidates` | 不返回全文；冲突只给观察提示 |
| `read_snippet` | 返回完整原文 + 元数据 | 引用前必须调用 |
| `submit_answer` | 提交结构化结果（终止） | 确定性校验（见下） |

## 引用校验（确定性，应用层硬保证）

`submit_answer` 后校验：① status 合法；② chunk_id 存在且 doc_id 一致；③ quote 是原文子串（忽略空白）；④ answered/conflict 至少一条有效引用；⑤ insufficient 必须给 gaps。失败时把错误作为 observation 回给模型修正。

## 冲突策略

- `detect_conflict_candidates` 按 `topic_key` 分组，同主题 ≥2 个 doc_id 时提示「可能存在事实不一致」。
- **裁决交给模型**：可依据 `supersedes` / `effective_date` 判定时说明取舍并引用新旧；无法判定时 `conflict_needs_clarification` 请求确认。
- 语料设计：D01(30天)/D02(15天) 无适用范围、无取代关系 → 不可解冲突；D03/D04 有 `supersedes` + 更晚日期 → 可解冲突；D10 同域不同 `topic_key` → 不误报。

## 技术取舍

1. **不用 LangGraph/ADK**：本题核心是展示模型如何按工具观察调整动作，原生 function-calling 循环更透明、trace 更易解释；框架不加分。
2. **不用向量库（Milvus 等）**：语料仅 10 篇，BM25 + 可选 RRF 足够，去掉外部依赖才能让评审一键复现。
3. **保留 `read_snippet`**：只给检索摘要会诱导模型凭摘要编引用；强制读原文才能保证 `quote` 可核验。
4. **用 `submit_answer` 工具**：把「停止」变成显式动作，同时获得结构化状态/引用/缺口，便于测试与评测。
5. **冲突只给提示不自动裁决**：题目要求模型依据检索结果决策，应用层只提供「同主题多文档」观察。

## 依赖清单

`jieba`（分词）、`rank-bm25`（稀疏检索）、`openai`（OpenAI 兼容客户端）、`pytest`（测试）、`python-dotenv`（环境变量）。均为原生轻量组合，未引入 Agent 框架。
