# Orchestrator (main session) — lessons learnt

每版一句，由本轮迭代结束的复盘产生。核心原则：**串起 architect / dev / tester / lead 四个角色 + 兜底他们之间的协议裂缝**。

orchestrator 自 v0.6 起就在干（main session 直接戴帽子做 plan / review / 串接），但 v0.10 起拆出独立 lessons——v0.8 + v0.9 两版都暴露同一类协议 gap，值得专门记录。

- **v0.8 + v0.9 / Q&A 答复无法广播给并行 dev**：两版都发生了同样的 reconciliation tax —— architect 设计有歧义 / 漏 pin 字面值，tester phase 1 plan 阶段把开放问题列给 orchestrator，orchestrator 答了、传给了 tester 和（v0.8 一次）test lead，**但 dev 已经在并行 spawn 跑着，没有渠道把答复同步过去**。`SendMessage` 在本环境不可用，无法 mid-flight 通知正在跑的 agent；下次 spawn dev 会拿到新 prompt 但当前这次拿不到。结果两版都是 dev 默选了"看起来合理"的反例，被 lead 抓出来后 orchestrator 打 5-10 行补丁修。**v0.10 起的应对**：(a) spawn dev 前先和 tester phase 1 plan **串行而非并行**，让 tester 把开放问题暴露完、orchestrator 答完、并入 dev prompt 再 spawn dev——损失一点 wall-clock 并行度，但消灭整类 gap；(b) 如果非要并行，spawn dev 时附带"已知开放问题清单"和"凡撞到这些里的字面值缺失，立刻停下来用 commit message 写 `WAIT-FOR-ORCHESTRATOR: <q>` 然后挂起 step" 的明文规则；(c) 把"早期 lead plan review 同时读 dev 已 commit 的代码"这条 v0.9 lead 教训作为兜底，最差也是 lead 抓到、补丁修。三层防御。
- **v0.8 / 监测 silent agent death 滞后**：tester phase 2 静默挂掉，orchestrator 是用户问"咋样了"才发现，否则会无限静默。下版起 spawn 长任务（dev / tester phase 2）时**心里有 wall-clock 估算**（dev 7 步约 30-60min、tester 5 个文件约 30-45min）——超过 2× 估算且没收到 task-notification 时主动 `git log --oneline` 看分支是否还在动；30 分钟以上无新 commit 视为可能 stall，准备接手。增量 commit + push 约束在 v0.9 已完整落到 dev/tester 的 prompt 里，本条作为 orchestrator 自己的监测兜底。
- **v0.8 + v0.9 / PR description 的诚实度由 lead 终 e2e 兜底**：两版 orchestrator 写的初版 PR description 都有 "claims vs reality" 偏差（v0.8 是 B18 / B22 描述过强；v0.9 是没写 web_fetch SSRF reason label collapse 等 caveat），都是 lead 终 e2e 列出来后我才补的。下版起 PR description 草稿**写完就主动让 lead 终 e2e 阶段做 cross-check**（"每条 claim 找到对应测试名"），不指望自己一遍写对。
