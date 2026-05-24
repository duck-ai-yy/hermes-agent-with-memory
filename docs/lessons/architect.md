# Architect — lessons learnt

每版一句，由本轮迭代结束的复盘产生。

- **v0.6 / `mneme search`**：设计契约里写了"`-k` 超大不报错"，但没去验底层依赖的物理硬墙（sqlite-vec 的 4096 上限），导致 developer 实现后仍撞墙被 tester 抓到——契约写完必须主动验真所有所依赖组件的边界值，不能假设"按需求写就够"。
- **v0.7 / cost display**（win）：设计文档第 5 节把"边界清单"列成 13 条 1:1 编号交给 tester，tester 直接照单全收 100% 覆盖；下一版继续这个模式——**架构师产出 = 设计 + 边界清单**，把"tester 该测什么"也变成架构师的可交付物，避免 v0.6 那种"漏一条物理硬墙"的事再发生。
- **v0.8 / agent loop M1**（win + 一条 gap）：在设计前主动跑 spike 验真三家 provider 的 tool-calling 协议，落地成"四条物理硬墙表"进 `docs/knowledge/provider-tool-calling.md`（Ollama arguments 是 dict / Anthropic tool_result 走 user role / Ollama 无 tool_call_id / FakeLLM 必须扩展）——dev 实现时**零返工**，把 v0.6 教训完整内化。**gap**：abort 字符串、`tool_calls` 在 reject 时是否计数这两条 user-visible 行为，原设计只写"abort with fixed text"和模糊的"counts attempts"，没 pin 具体字符串和具体计数语义；等到 tester 开放问题阶段才被 orchestrator 临时答复，结果 dev 没收到那份答复、自己选了另一套合理实现，引发 lead-e2e 阶段的 reconciliation tax。下一版起：**设计文档里凡是 user-visible 的字符串字面量 / counter 语义 / 错误前缀，必须在 §X 显式 pin 到字面值或精确公式，不能留到 Q&A**——架构师答案不会自动传播到正在并行写代码的 dev。
