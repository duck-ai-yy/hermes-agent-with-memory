# Developer — lessons learnt

每版一句，由本轮迭代结束的复盘产生。核心原则：**非必要勿增实体**。

- **v0.6 / `mneme search`**：`except Exception as exc: secho("embed provider unreachable: {exc}")` 把所有失败混标成同一种错（sqlite-vec 的 `OperationalError` 被错指向 embed provider），用户被误导去查 Ollama / 网络——错误消息要么具体到 error type，要么写中性（如 `"search failed: {exc}"`），别替用户做错误归因。
