# 03 · 评测与结果

## 测试

`python -m pytest -q` → **68 passed**，输出存档 `examples/pytest-output.txt`。

| 测试文件 | 覆盖 |
|---|---|
| test_smoke / test_types_status | 包导入、类型与状态机 |
| test_corpus / test_loader | 语料完整性、frontmatter/切分 |
| test_retrieval | BM25 命中、无命中、冲突候选 |
| test_citation / test_tools | 引用校验、read_snippet、注册表 |
| test_mock_llm | Mock 按 observation 分支 |
| test_react_loop | 端到端（正常/冲突/改写/资料不足/追问） |
| test_tool_failure / test_max_steps | 工具失败、步数/工具上限 |
| test_cli_offline | CLI 离线返回合法 JSON |
| test_openai_client | OpenAI 兼容响应转换 |

## 六场景结果（Mock 离线复现）

| # | 场景 | 输入 | 状态 | 产物 |
|---|---|---|---|---|
| 1 | 明确答案 | 病假需要提前多久申请？ | `answered`（D05） | `examples/case01_normal/` |
| 2 | 文档冲突 | 报销期限多久？ | `conflict_needs_clarification`（D01+D02） | `examples/case02_conflict/` |
| 3 | 资料外 | 公司给宠物买保险吗？ | `insufficient_information` | `examples/case03_insufficient/` |
| 4 | 改写问法 | 出差回来多久内要报销？ | `conflict_needs_clarification` | `examples/case04_paraphrase/` |
| 5 | 上下文追问 | （历史）年假 → 那病假呢？ | `answered`（D05） | `examples/case05_followup/` |
| 6 | 工具失败 | 病假…（注入超时） | `answered`（重试后恢复） | `examples/case06_tool_failure/` |

批量复现：`python -m agentqa.cli --batch data/questions.json --provider mock --out examples/mock/`（`examples/mock/summary.json`）。

## 两条执行路径（体现「工具结果 → 调整动作」）

- **路径 A · 冲突追问**：`examples/case02_conflict/trace.json`
  `search` 看到 `conflict_candidates [D01,D02]` → 依次 `read D01`、`read D02` → `submit conflict_needs_clarification`。
- **路径 B · 工具失败重试**：`examples/case06_tool_failure/trace.json`
  `search` 超时 `ok=false` → 改写重试 `search` → `read D05` → `submit answered`。

## 真实模型结果（DeepSeek，OpenAI 兼容）

| 场景 | 状态 | 产物 |
|---|---|---|
| 正常问答 | `answered`（引用 D05，自发 2 个并行检索） | `examples/real/case01_normal/` |
| 冲突问答 | `conflict_needs_clarification`（同时引用 D01+D02，追问适用哪份） | `examples/real/case02_conflict/` |

真实模型未擅自选 15 或 30 天，符合「不凭空选一个答案」要求。

## 失败与限制

- Mock 不执行真正的查询改写与冲突裁决（按脚本分支），仅证明流程与状态机；这两点由真实模型验证。
- 检索为纯 BM25，未接向量/重排；语料规模小，已满足本题，但语义近义改写能力有限（靠词元重叠）。
- 冲突检测按 `topic_key` 分组，若检索把同主题文档带进 top_k 才会提示；极端长查询下的精确召回未专门调优。
- 追问场景的 Mock 只取最新 user 消息，未做真实历史改写；真实改写依赖真实模型 + 系统提示词第 7 条。
