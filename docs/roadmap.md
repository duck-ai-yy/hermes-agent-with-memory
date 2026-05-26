# Mneme Roadmap

来源：M1-M4 milestone plan（用户给的） + 旧 v0.8+ 计划 + PRINCIPLES.md。
整合原则：M3（持久记忆）现在已经 done（三层记忆就是它），其余 M1/M2/M4 织入版本节奏。

## 已发版 ✅

| 版 | 主题 | 测试 |
|---|---|---|
| v0.1 | scaffold + 三层核心（Brain / Soul / Memory） | 15 |
| v0.2 | recall 排除自身 | 25 |
| v0.3 | embed provider 与 chat 解耦 | 25 |
| v0.4 | 流式 CLI + 真实 token 跟踪 | 33 |
| v0.5 | 每日云 token 预算硬墙 | 39 |
| v0.6 | `mneme search` CLI（首次多 agent 协作） | 60 |
| v0.7 | 每轮 $ 成本显示 | 93 |
| v0.8 | Agent loop (M1)：respond() user→LLM→tool→LLM→response | — |
| v0.9 | Tool registry (M2)：@tool 装饰器 + file_read / web_fetch / python_exec | — |
| v0.10 | Multi-turn session context：session_id 贯通 agent / server / events | 325 |
| v0.13 | HTTP `/chat` SSE 流式（Accept-协商，JSON path 字节不变） | — |

## 接下来 ⏳（M1-M4 织入）

| 版 | 来源 | 主题 | 大小 | 依赖 |
|---|---|---|---|---|
| **v0.11** | **M3** | 上下文窗口管理：token 计数 → 压缩 → 摘要；持久记忆已 done，只补 token-budget-aware prompt 组装 | M | v0.10 |
| **v0.12** | **M4** | Skill system：`~/.mneme/skills/<name>/SKILL.md` + 触发条件 + 可选 `script.py`；agent 完成复杂任务后可自主生成 skill | M | v0.9, v0.11 |
| **v0.14** | 旧 v0.10 | `mneme dream`：nightly cron 合成今日 concept node | M | v0.11 |
| **v0.15** | 旧 v0.11 | 飞书 webhook adapter | L | v0.9, v0.10, v0.13 |

## 明确不做 ❌（PRINCIPLE 1 否决）

- **Langfuse**：`events.jsonl` 已满足 PRINCIPLE 5，不引第三方观测平台
- **multi sub-agents 抽象层**：等真有具体 use case 再引；M1+M2 已让 agent 能自循环 + 用 tool，目前够
- **Tauri 桌面壳**：CLI 已是主入口，不为"以后好看"造 desktop app

## 推后到有真实需求才做 🕒

- `mneme graph` 可视化（调试要用了再做）
- 真实 Ollama 集成测试（fake LLM + 手动验证够用前）
- snapshot 自动旋转（用户抱怨备份膨胀时）

## 关键架构决策（v0.8 开始）

- **Agent loop 不抛弃现有 `respond()` 契约**：保留 ingest / retrieve / citation 五段流，只是把"单次 LLM 调用"换成"多步 tool-using 循环"。memory / soul / events 三个层都不动。
- **Tool execution 走显式 audit + confirm**（PRINCIPLE 4）：shell / file 写 / 网络访问等"有副作用"的 tool 调用前在 `events.jsonl` 写 audit，破坏性操作要 CLI 层 confirm prompt。
- **Skill 默认本地**：`~/.mneme/skills/`，类似 prompts 的种子模式（`examples/skills/` 仅做 init seed）。
- **Multi-agent 协作流程从 v0.6 开始已沉淀**：每版照走 architect → dev/tester 并行 → test lead → 复盘判断 → PR → merge；lessons 落 `docs/lessons/`，spike 笔记落 `docs/knowledge/`。
