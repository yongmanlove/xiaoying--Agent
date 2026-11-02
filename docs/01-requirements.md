# 01 · 需求文档

## 用户与目标

- **用户**：企业内部员工，需要快速、可靠地查询制度/FAQ 的答案。
- **目标**：对「问题 + 可选对话历史」，Agent 依据合成制度资料给出**可核验**的回答——带文档 ID、版本、原文引用，并明确资料缺口与任务状态；遇到资料冲突时不凭空选答案，而是说明冲突并请求确认。

## 范围

- **输入**：问题（字符串）+ 可选对话历史（user/assistant 轮次）。
- **材料**：带 `doc_id / version / effective_date` 的合成制度（Markdown + frontmatter）。
- **工具**：文档检索（`search_documents`）、片段读取（`read_snippet`）、答案提交（`submit_answer`）。
- **输出**：`status / answer / citations / gaps / clarification_question`。

## 成功标准（验收点）

| 验收点 | 期望 |
|---|---|
| 明确答案 | `answered`，引用单篇原文 |
| 文档冲突 | `conflict_needs_clarification`，列出双方 + 引用 + 请求确认，不擅自选 |
| 资料外问题 | `insufficient_information` + 缺口，不编造 |
| 改写后相似问题 | 检索到相关文档 |
| 上下文追问 | 结合历史定位问题 |
| 工具失败 | 重试或明确 `tool_error`，不崩溃 |

## 非目标

界面、线上部署、多 Agent、生产级 Runtime、向量库、联网搜索均非本题要求，未纳入实现。
