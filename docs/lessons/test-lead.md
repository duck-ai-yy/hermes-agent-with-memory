# Test Lead — lessons learnt

每版一句，由本轮迭代结束的复盘产生。核心原则：**独立挑刺、不信 implementer 的自报告、跑自己的 mutation**。

v0.8 起新设此角色——前几版（v0.5-v0.7）的"test lead 评审"是 orchestrator / 主 session 临时戴的帽子；v0.8 起拆出来作为并行流程的独立环节，分两次发挥作用：(1) **早期 plan review**（tester 写完 plan 但还没写代码时），(2) **final e2e**（dev + tester 都收工后）。

- **v0.8 / agent loop M1**（双 pass 立功 + 模式确立）：
  - **Pass 1 (plan review)** 抓出 tester 的 3 条 must-fix：B4 用 substring 而非 exact-string、B13 stream 语义读反、B6 缺第 2 轮 `is_error` spy（v0.6 "happy-path 伪装 error-path" 的换皮版）。在 tester 写代码前抓到 → 省一轮返工。**模式固化**：plan review 阶段必须独立读架构师设计的 `# why` 注释、不只是核对 tester 1:1 boundary 列表，因为 tester 容易漏掉"实现里 explain 了但没列进边界"的 invariant。
  - **Pass 2 (final e2e)** 独立跑 3 条 mutation（M2 / M4 / M11，跟 orchestrator 跑的 M1 / M15 不重叠）→ 验证 plan 矩阵真有效；再追加 2 条 ad-hoc mutation（删 HW4 stream-guard / 删 B18 redaction-guard），**两条都过了 139 测试**，暴露 3 个 test-suite 缺口：HW4 没测、B18 测的是另一个 invariant（client.config 不变性 vs 真正声称的"events.jsonl 不含 stdout"）、B22 只 mimic 没真用 TestClient。**这是这版最大教训**：implementer（orchestrator）自报"139 passed + 2 mutations caught"不够——lead 必须**自己挑 3-5 条针对性 mutation 跑**，特别是针对架构师设计里写了"out of scope" / "deferred" / "M1 forces..." 这种"我们承诺不会做什么"的 invariant——这类**否定式 contract** 最容易 ship 时无人钉住，因为正向行为永远绿。
  - **可挑刺清单**（每次 e2e 必走）：(a) 跑 3+ 条 mutation 验 plan 矩阵；(b) 逐条 cross-check PR description 的 claim 跟实际测试断言；(c) 找架构师设计里所有"M1 forces / does not / never / out of scope" 句子，每条 grep 对应测试存不存在；(d) read-only contracts（events.jsonl 不含 X / 不写 Y）必须有 before/after byte-snapshot 类断言，不能只看 chat_calls 之类间接信号。
  - **不该做的事**：lead 不写测试不写补丁，只 review；最终结论用 "放行 / 补 X 后放行 / 返工" 三档，不模糊；建议的 PR 评论给主 session triage，不要自己 post 到 PR 上避免双人协同混乱。
- **v0.9 / tool registry M2**（早期 review 加值最大化 + 终 e2e 持续 ROI）：
  - **Pass 1 (plan review) 加了一招**：除了核对 tester 边界 + 扫架构师 "why" 注释，**主动读了 dev 当时已 push 的实现代码**——结果在 tester 还没写一行测试时就抓到 dev 实际代码的 2 个 bug（`registry.py:305` ArgumentError 用 Python `__name__` 不是 JSON-schema name；`registry.py:253` unknown provider 静默 fallback 不 raise）。这两条都是 orchestrator 答给 tester 但没传到 dev 的 pin。如果只看 plan 文本不读代码，dev bug 会一直 ship 到终 e2e 才被发现。**模式升级固化**：plan review 时**必须 git log + 读 dev 当前已 commit 的实现**，对照架构师 design + orchestrator 答复 grep 字面值不一致——这是早期 review 最高 ROI 的动作，比纯审 plan 文本贵 5 分钟、省 reconciliation tax 10×。
  - **Pass 2 (final e2e)** 跑 3 条独立 mutation（D7 vocab / R-Reg-6 raise / R-WF-5 SSRF-no-audit）+ 1 条 bonus（agent allowed_tools=[] guard），全部按预期 fail 对应测试集 + 全部恢复干净——matrix 真实有效。同时找出 3 个 minor negative-form gaps：`tool_registry_warn` 没正向测、新 tool 的"never write events.jsonl" 没像 shell 那样 byte-snapshot、`_parse_docstring` 续行没测。**none ship-blocker，但都是 v0.8 教训的细化变种**——negative-form contract 已经覆盖大部分了，仍有"best-effort / silently drops" 这一类细分需要 tester 在下版 plan 阶段额外 grep。
  - **维持纪律**：v0.9 lead 输出仍是 "放行 / 补 X 后放行 / 返工" 三档结论 + 不直接 post 到 PR + 不写补丁——双 pass + 三档 + read-only 这套约束在 v0.9 继续是 ROI 最高的姿势。
