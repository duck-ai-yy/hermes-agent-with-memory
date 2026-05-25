# Mneme

本地优先的 **chat agent + 长期记忆**。单用户、原生栈、可溯源。

跨会话记住你说过的事：三天前提到的概念，今天再聊它还在。每条回复都能点回它依据的原始记录。数据默认永不离开本机。

> 设计原则见 [`PRINCIPLES.md`](./PRINCIPLES.md) —— 任何新增功能先过那五条。

---

## 它和普通 chatbot 的 6 个差异

1. **跨会话记忆** —— 三天前说的概念今天还在
2. **可溯源** —— 每条回复带 `[^slice_id]`，`/explain` 一键看依据
3. **可遗忘** —— `/forget` 真删（slice + 边 + 向量同事务）
4. **永远本地** —— 默认 Ollama，云 provider 需 opt-in 且写 audit
5. **快** —— 稳定 prompt prefix 命中缓存，7B 模型也秒回
6. **零运维** —— 单文件数据库，备份 = 复制文件，迁移 = 拷过去

---

## 安装

```sh
pip install -e .
mneme init
```

`init` 会检测本地 Ollama，并在 `~/.mneme/` 下创建 `db.sqlite`、`events.jsonl`，以及从仓库
`examples/` 种入 `blueprint.md` 和 `prompts/*.yaml`。**身份（blueprint + prompts）从此归你**
—— 编辑 `~/.mneme/blueprint.md` 与 `~/.mneme/prompts/*.yaml` 即生效；仓库里的 `examples/`
只在首次 init 时被读一次，之后不再回头。

## 使用

```sh
mneme chat                  # 进入对话
mneme explain <trace_id>    # 溯源某次回复
mneme forget <slice_id>     # 删除一条记忆（级联边/向量）
mneme search <query>        # 直接对长期记忆做向量+图检索
mneme snapshot              # SQLite .backup 到 snapshots/
mneme blueprint             # $EDITOR 打开系统蓝图
mneme stats                 # slice/node/edge 数、db 大小
mneme serve [--port 7890]   # 启动 HTTP API
```

对话内可用斜杠命令：`/explain`、`/forget <id>`、`/quit`。

## HTTP API

```
POST /chat            { messages, session_id? }   → JSON
GET  /explain/{tid}
POST /forget/{sid}
POST /snapshot
GET  /stats
```

---

## 架构

```
mneme/
├─ cli.py              CLI 入口
├─ server.py           FastAPI（按需启动）
├─ agent.py            chat loop: ingest → retrieve → respond
├─ ids.py              不可变 ID 生成（ULID，纯 stdlib）
├─ memory/             记忆层 —— 见 docs/MEMORY.md
│   ├─ schema.sql      DDL（5 张表，一个 sqlite 文件）
│   ├─ store.py        连接 + 事务上下文管理器
│   ├─ embed.py        embed() + embeddings_cache 命中
│   ├─ ingest.py       消息 → slice + 向量 + 概念抽取 + 建边
│   ├─ retrieve.py     向量 top-K → 图 BFS 扩展 → 稳定排序
│   ├─ forget.py       级联删 slice / edge / vector
│   ├─ concept.py      LLM 抽概念（prompts/concept_extract.yaml）
│   └─ graph.py        BFS（纯 stdlib）
├─ llm/
│   └─ client.py       httpx singleton（Ollama / OpenAI opt-in）
└─ trace/
    └─ events.py       events.jsonl 单一 append-only 日志
examples/              用户本地身份的种子模板（首次 init 拷到 ~/.mneme/，仓库不持续承载身份）
  ├─ prompts/*.yaml    prompt 模板（运行时改 ~/.mneme/prompts/）
  └─ blueprint.md      blueprint 模板（运行时改 ~/.mneme/blueprint.md）
```

### 一次 chat turn 的事件流

```
User msg
  ├─① ingest      slice + 向量入库（同事务）→ LLM 抽概念 → nodes/edges
  ├─② retrieve    query embed → 向量 top-K → 图 BFS 1–2 跳 → 稳定排序
  ├─③ build       stable prefix（blueprint + tools + system）+ dynamic suffix
  ├─④ LLM call    前置写 trace → 输出 → parse [^sid] 引用
  └─⑤ persist     assistant 回复也入库（下轮可被检索）
```

### 进程模型

没有 daemon。CLI in-process 跑完即退；`serve` 才有常驻进程；snapshot 用户手动或 cron 触发。

---

## 技术栈

- Python stdlib：`sqlite3`、`json`、`hashlib`、`collections`、`asyncio`、`dataclasses`
- `sqlite-vec`：SQLite 原生向量扩展（图与向量同库、同事务）
- `fastapi` + `uvicorn`：HTTP API 层
- `httpx`：调 LLM
- 默认 LLM：本地 Ollama；云 provider 走 opt-in

不用 ORM、不用 networkx、不用专门图/向量数据库。理由见 `PRINCIPLES.md` 原则 3。

---

## 云 provider opt-in（PRINCIPLES.md 原则 4）

默认走本地 Ollama。要接 OpenAI 兼容的云 API（DeepSeek / Zhipu / OpenAI / Moonshot
/ OpenRouter 等），用环境变量覆盖默认：

```sh
# 例 1：DeepSeek（chat-only，embed 走 hash 兜底，仅精确文本召回）
export MNEME_PROVIDER=openai
export MNEME_BASE_URL=https://api.deepseek.com
export MNEME_API_KEY=sk-...
export MNEME_CHAT_MODEL=deepseek-chat
export MNEME_EMBED_VIA=hash
mneme init && mneme chat

# 例 2：Anthropic chat + OpenAI embed（chat / embed 可独立配置）
export MNEME_PROVIDER=anthropic
export MNEME_BASE_URL=https://api.anthropic.com
export MNEME_API_KEY=sk-ant-...
export MNEME_CHAT_MODEL=claude-haiku-4-5
export MNEME_EMBED_PROVIDER=openai
export MNEME_EMBED_BASE_URL=https://api.openai.com
export MNEME_EMBED_API_KEY=sk-...
export MNEME_EMBED_MODEL=text-embedding-3-small
```

embed env 不设时回退到 chat 同套。OpenAI 兼容的 embed 请求带 `dimensions=768`，
和 `vec_slices FLOAT[768]` schema 对齐。

所有 cloud 调用都会在 `~/.mneme/events.jsonl` 先写一条 `audit` 事件（provider /
endpoint / model / 估算 token），API key 本身**不会**被记录。
