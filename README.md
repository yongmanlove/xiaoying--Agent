# 企业知识问答 Agent（xiaoying-agent-knowledge-qa）

面向企业内部制度问答的 **ReAct Agent**。给定「问题 + 可选对话历史」，模型自主调用
「文档检索 + 片段读取」工具，产出**带引用、资料缺口与任务状态**的结构化回答；
资料冲突时说明冲突并请求确认，不凭空选一个答案。

- 运行方式：CLI（`python -m agentqa.cli ...`）
- 离线 Mock：无 API Key、无外部服务即可完整复现决策流程
- 真实模型：OpenAI 兼容端点（DeepSeek / DashScope / OpenAI 均可）
- 原生实现：不依赖 LangGraph / ADK / 向量库，仅 jieba + BM25 本地检索

## 快速运行

Python 3.10+。离线 Mock 无需密钥即可运行：

```sh
pip install -r requirements.txt
pip install -e .

# 单条问题（输出结构化 JSON）
python -m agentqa.cli --question "报销期限多久？" --provider mock

# 单个用例（读 input.json，写 result.json / trace.json / output.md）
python -m agentqa.cli --case examples/case02_conflict/input.json --provider mock

# 批量运行六类场景
python -m agentqa.cli --batch data/questions.json --provider mock --out examples/mock/

# 测试
python -m pytest -q
```

真实模型（OpenAI 兼容端点，如 DeepSeek）：`cp .env.example .env` 填入 `LLM_API_KEY` 后，
加 `--provider openai` 运行；真实模型运行记录见 `examples/real/`。

## 项目目录

```text
.
├── README.md                 # 安装、运行、评审导航
├── pyproject.toml            # 包定义 + pytest 配置
├── requirements.txt          # 依赖清单
├── .env.example              # 环境变量模板（占位符，无密钥）
├── .gitignore
├── docs/
│   ├── 01-requirements.md    # 用户、范围、指标
│   ├── 02-solution.md        # 流程、模块、取舍
│   ├── 03-evaluation.md      # 用例和实际测试记录
│   ├── 04-ai-usage.md        # AI 使用及验证
│   └── 05-retrospective.md   # 不足和迭代
├── src/agentqa/              # 项目代码与模型/工具接入
│   ├── cli.py                # CLI 入口
│   ├── config.py             # 配置（.env / 环境变量）
│   ├── agent/                # ReAct 循环、状态机、引用校验、提示词
│   ├── llm/                  # LLMClient 接口 + Mock + OpenAI 兼容客户端
│   ├── data/                 # frontmatter 解析 + 文档切分
│   ├── rag/                  # BM25 索引、冲突候选、数据类型
│   └── tools/                # search / read_snippet / submit_answer + 注册表
├── tests/                    # 68 项测试（pytest 全离线）
├── data/
│   ├── docs/                 # 10 篇合成制度（D01~D10）
│   └── questions.json        # 批量运行问题集
└── examples/
    ├── case01~06/            # 六场景 输入/输出/Trace
    ├── real/                 # 真实模型运行记录
    ├── mock/                 # 批量运行产物
    └── pytest-output.txt     # 测试执行记录
```

## 评审导航

| 检查项 | 入口 |
|---|---|
| 需求与验收点 | [需求](docs/01-requirements.md) |
| 流程、动态决策与取舍 | [方案](docs/02-solution.md) |
| 测试、预期、结果 | [评估](docs/03-evaluation.md) + [测试记录](examples/pytest-output.txt) |
| AI 辅助边界 | [AI 使用](docs/04-ai-usage.md) |
| 不足及改进 | [复盘](docs/05-retrospective.md) |
| 六场景运行记录 | [examples/](examples/)（case01~06 + real） |

## 离线 Mock 边界

- Mock 是**按 observation 分支的脚本模型**，只验证流程、状态机与工具契约；
- **不证明真实模型决策效果**。真实决策质量见 `examples/real/`（DeepSeek 运行记录）。

## 已知限制

- 检索为纯 BM25（jieba 分词），未接向量/重排；语料规模小，已满足本题需求。
- 冲突检测按 `topic_key` 分组只做「观察提示」，裁决由模型完成。
- Mock 不执行真正的查询改写与冲突裁决（按脚本分支）；需真实模型体现。
- 工具失败路径通过 `--fail-search N` 注入演示；`tests/test_tool_failure.py` 覆盖二次失败转 `tool_error`。

## 提交约定

所有代码、文档、示例与结果放同一仓库。提交题号、GitHub URL 与最终 commit SHA；
README 标明入口。私有仓库需让指定评审账号可读。不得提交密钥或隐私数据
（`.env` 已在 `.gitignore`）。
