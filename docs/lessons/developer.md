# Developer — lessons learnt

每版一句，由本轮迭代结束的复盘产生。核心原则：**非必要勿增实体**。

- **v0.6 / `mneme search`**：`except Exception as exc: secho("embed provider unreachable: {exc}")` 把所有失败混标成同一种错（sqlite-vec 的 `OperationalError` 被错指向 embed provider），用户被误导去查 Ollama / 网络——错误消息要么具体到 error type，要么写中性（如 `"search failed: {exc}"`），别替用户做错误归因。
- **v0.7 / cost display**（win）：`pricing._load_table` 的 4 种失败模式（文件缺 / yaml parse / 顶层非 mapping / IO error）各对应一条具名 stderr warning，绝不写"unknown error"——上一版 lesson 完整吸收。另外 spike 工具被正确判断为"不需要"（yaml 包内加载 + `functools.cache` 都是 30 分钟内能自查的 stdlib 知识），ROI 判断没浪费一次 agent 调用。模式保留：**report 里列出"micro-decisions not in the design"**，把"我自作主张了什么"摊开，建立 reviewer 信任。
- **v0.8 / agent loop M1**（mixed）：7-step self-validate 通行无阻 + report 里 flag 7 条 micro-decisions，v0.7 模式继续生效。**但**：撞到 2 处架构师没明 pin 的 user-visible 行为（abort 字符串字面量 / `tool_calls` 在 reject 时是否计数），我**直接选了一套合理实现**而没停下来问；orchestrator 那时正在跟 tester 答开放问题、给了另一套答案，两边对不上，lead-e2e 才捡出来。教训：**dev step 里凡撞到"设计没说具体值"的 user-visible 行为，先停**——挂起当前 step、用一行问 orchestrator/architect，等回应再继续；猜测 + 标"micro-decision" 不够，因为同一时间架构师可能在别处给出冲突答复，事后 reconcile 比一开始问一次贵 10×。另外做对了一件事：subprocess + DEVNULL stdin + timeout 这种"30 分钟内可自查"的领域没 spawn spike——ROI 判断保持准。
