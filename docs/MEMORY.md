# 记忆层实现细则

记忆层负责四件事：**存**（ingest）、**取**（retrieve）、**删**（forget）、**管**（embedding 缓存 / 事务 / snapshot）。

所有写操作落在 **一个 sqlite 文件**（`~/.mneme/db.sqlite`）+ 一个 append-only `events.jsonl`。零外部图库、零外部向量库、零 ORM。

---

## 核心抽象（v0 只有 4 个一等公民）

| 实体 | 是什么 | 不可变 |
|---|---|---|
| **slice** | 一段不可变文本（用户消息 / agent 回复 / 提取出的事实） | ✅ |
| **vector** | slice 对应的 embedding，存在 sqlite-vec 虚表 | ✅ |
| **node** | 概念（人 / 物 / 时间 / 想法），按 `name` 归一化复用 | ✅ |
| **edge** | 有向有类型的关系，带 `slice_id` 记住"是哪句话引入了我" | ✅ |

**没有** entity/activity/intent 这类分类。原则 1：用 `slices.kind` 列预留，真有需要再启用。

---

## 模块分工

```
memory/
├─ schema.sql      DDL（一次性建表）
├─ store.py        低层：connect / 事务上下文管理器 / 通用 execute
├─ embed.py        embed(text)，带 embeddings_cache 命中
├─ ingest.py       save_user_message / save_assistant_message
├─ retrieve.py     recall(query, k=10) → list[Slice]
├─ forget.py       forget(slice_id) → 级联结果
├─ concept.py      LLM 抽概念（prompts/concept_extract.yaml）
└─ graph.py        BFS（约 20 行 stdlib）
```

---

## Schema（见 `mneme/memory/schema.sql`）

5 张表，一个文件：

- `slices(id, role, text, turn_id, created_at)` —— 主表
- `nodes(id, name UNIQUE, kind, first_seen)` —— 概念，name 唯一
- `edges(id, src, dst, type, slice_id, created_at)` —— 关系，`slice_id` 带 `ON DELETE CASCADE`
- `embeddings_cache(text_hash, vector, created_at)` —— 任意文本的 embed 缓存
- `vec_slices(slice_id, embedding)` —— sqlite-vec 虚表，与 slices 1:1

`edges.slice_id ON DELETE CASCADE` 是关键：`/forget` 删 slice 时自动级联删边，应用层不用操心。

---

## 主路径 1：ingest

**两段事务**：

1. **事务 A** —— `slices` INSERT + `vec_slices` INSERT。永远成功。
2. **LLM 抽概念** —— 调 `concept.extract()`，输出 `{nodes, edges}`。
3. **事务 B** —— `nodes` INSERT OR IGNORE（按 name 归一化）+ `edges` INSERT。

**关键设计**：概念抽取失败只记 warning，**不回滚事务 A**。slice + vector 已可搜索；图变稀疏不影响对话流畅性。原则 1：放弃完美一致性，省下重试逻辑。

---

## 主路径 2：retrieve

`recall(query, k=10, hops=2)`：

1. **向量 top-K** —— query embed → `vec_slices ... WHERE embedding MATCH ? AND k=?`。
2. **种子节点** —— 这 K 条 slice 涉及哪些 node（查 `edges.slice_id`）。
3. **图扩展** —— 从种子节点沿边 BFS 1–2 跳（`graph.bfs`）。
4. **反查 slice** —— 扩展出的 node 又被哪些 slice 提到。
5. **合并 + 重打分** —— `vec_score * 0.7 + graph_score * 0.3`。
6. **稳定排序** —— `ORDER BY score DESC, id ASC`，截断到 token 预算。

**关键设计**：向量找"语义相近"，图找"概念相邻"，70/30 加权。稳定排序保证相同检索得相同 prompt → 缓存命中（原则 2）。

---

## 主路径 3：forget

`forget(slice_id, consent)`：

1. 无 `consent` 直接拒绝（`PermissionError`）。
2. 单事务：`DELETE vec_slices` + `DELETE slices`（`edges` 经 `ON DELETE CASCADE` 自动清）。
3. `events.jsonl` append `{kind: "forget", slice_id, cascade}`。

**orphan node 不主动清**。某 node 删完所有引入 slice 后仍留在 `nodes` 表，下次被提到时复用恢复。原则 1：不为边界情况写 GC。

---

## embedding 缓存

`embed(text)`：

1. `h = sha256(text)[:16]` → 查 `embeddings_cache`，命中直接返回。
2. 未命中 → 调 `llm.client` 的 Ollama `/api/embed` → `INSERT OR IGNORE` 入缓存 → 返回。

`embeddings_cache` 缓存**任意文本**（含 query、重复短语），与 `vec_slices`（只存 slice 的可搜索副本）功能不重叠。

---

## 图 BFS

见 `mneme/memory/graph.py`，约 20 行纯 stdlib（`collections.deque`）。每跳一次 `SELECT ... WHERE src=? OR dst=?`，靠 `idx_edges_src` / `idx_edges_dst` 走索引，10 万 edge 量级毫秒级。

---

## 关键 trade-off

| 决策 | 选了 | 没选 | 为什么 |
|---|---|---|---|
| slice 切分 | 一条消息 = 1 slice | 按句/按 token 切 | 原则 1，>1000 tokens 时再考虑 |
| edge type | 固定 7 个 | LLM 自由发挥 | 自由发挥导致类型爆炸 → 破坏 retrieval 一致性 |
| 概念抽取失败 | warning，不回滚 | 整事务回滚 | 原则 1，对话流畅 > 图完整 |
| 并发 | sqlite 锁 + `asyncio.to_thread()` | aiosqlite | 原则 3，stdlib 够用 |
| orphan node | 留着 | 后台 GC | 原则 1，复用概率 > 清理收益 |
| vector ↔ slice 一致性 | 同事务保证 | 异步对账 | sqlite-vec 同库的核心红利 |

---

## 规模估算

约 **350 行 Python + 30 行 SQL**。
