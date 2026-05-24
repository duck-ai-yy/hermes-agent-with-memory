# Developer — lessons learnt

每版一句，由本轮迭代结束的复盘产生。核心原则：**非必要勿增实体**。

- **v0.6 / `mneme search`**：`except Exception as exc: secho("embed provider unreachable: {exc}")` 把所有失败混标成同一种错（sqlite-vec 的 `OperationalError` 被错指向 embed provider），用户被误导去查 Ollama / 网络——错误消息要么具体到 error type，要么写中性（如 `"search failed: {exc}"`），别替用户做错误归因。
- **v0.7 / cost display**（win）：`pricing._load_table` 的 4 种失败模式（文件缺 / yaml parse / 顶层非 mapping / IO error）各对应一条具名 stderr warning，绝不写"unknown error"——上一版 lesson 完整吸收。另外 spike 工具被正确判断为"不需要"（yaml 包内加载 + `functools.cache` 都是 30 分钟内能自查的 stdlib 知识），ROI 判断没浪费一次 agent 调用。模式保留：**report 里列出"micro-decisions not in the design"**，把"我自作主张了什么"摊开，建立 reviewer 信任。
