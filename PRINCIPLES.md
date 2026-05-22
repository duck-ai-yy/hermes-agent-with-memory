# Mneme · 设计原则

本文件只列**原则**。具体架构、模块、决策在别处。任何新增功能、新依赖、新抽象，先过这五条。

---

## 1. 非必要勿增实体

奥卡姆剃刀。每一个类、表、文件、依赖都要论证"为什么没有它不行"。

- 不写未来可能用到的抽象层
- 不为假想的扩展性预留接口
- 三处相似的代码 < 一处错误的抽象
- 删除比新增更难，因此新增要更慎重

## 2. token 效率最高，尤其是缓存命中

LLM 调用的成本是项目的主要成本中心，prompt 缓存是降本的最大杠杆。

- prompt 结构分段：**稳定 prefix（系统/工具/蓝图）→ 中等稳定（用户摘要）→ 动态（本次检索 + query）**
- 检索结果必须放在 prompt 末尾，永远不污染 prefix
- 相同输入得相同 prompt：稳定 ID + 稳定排序（`ORDER BY score DESC, id ASC`）
- 同一段文本只 embed 一次（本地 embedding 缓存表）

## 3. 原生优先

能 stdlib 不引第三方；能用 SQLite 不引专门数据库；能写 50 行不引框架。

- 允许：Python stdlib、SQLite 及其原生扩展（如 sqlite-vec）、FastAPI/uvicorn、httpx
- 禁止：ORM、networkx、专门图/向量数据库（Neo4j / Chroma / Lance / Qdrant 等）作为强依赖
- 例外需在 PR 描述里说明"为什么 stdlib 不够"

## 4. 安全、隐私、本地

用户的数据默认永不出本机；任何外发都需要明确授权与可审计记录。

- LLM/Embedding 默认本地（Ollama / 本地模型），云 provider 走 opt-in
- 外发调用前置 audit log（写了什么、发去哪、多少 token）
- 写操作走白名单 + 一次性确认 prompt + append-only 决策日志
- 备份/恢复用 SQLite 原生 `.backup`，不发明新机制

## 5. 可解释 / 可溯源

不要黑箱。每个对外输出都能点回它依赖的数据与 prompt。

- 所有持久化对象有不可变 ID（ULID 或 `sha256(content)[:16]`）
- 每次 LLM 调用前置 trace 记录（used_slices / used_edges / prompt_hash / model）
- 输出强制引用契约：事实必须以 `[^slice_id]` 标注，未引用 = 软警告
- `/explain/{trace_id}` 一键 dump：完整 prompt + 用了哪些数据 + 模型版本

---

## 冲突时的优先级

当原则之间冲突，按编号优先：**1 > 2 > 3 > 4 > 5**。

例：可溯源要求 LLM 输出引用（原则 5）会增加 token（违反原则 2），但 token 增量小、收益高，保留。
反例：为了"以后好换图数据库"加一个 `GraphRepo` 抽象层 —— 违反原则 1，**拒绝**，哪怕原则 3 不反对。
