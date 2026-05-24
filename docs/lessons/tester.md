# Tester — lessons learnt

每版一句，由本轮迭代结束的复盘产生。核心原则：**复杂度棘轮、error path > happy path、plan-first、单元 + 回归并行**。

v0.7 起本文件转向：**前瞻性策略观察**，不只是"这一版我犯了什么错"——记录可在下一版直接照搬的测试模式。

- **v0.6 / `mneme search`**：plan-first 和 ratchet 自检都到位（19 个合约级测试，抓到 BUG-v0.6-1），但 special-char / provider-down 这类 case 只 assert `exit_code == 0` 是"happy-path 伪装成 error-path"，且对"纯 read 契约"（events.jsonl 字节不变 / `chat_calls == 0`）零监控——error-path 测试必须同时 assert **可观察输出**（substring / 不变量），read-only 契约要主动写 before/after snapshot，不能只信 exit code。
- **v0.7 / cost display**（观察 → 策略）：架构师这次给的 13 条边界清单让 plan-first 1:1 落地极其顺手（test lead 验证 13 条全中），下一版起**要求架构师产出明确编号的边界清单**作为 test-plan 起点；本轮自创的 **mutation sanity check**（强行改 `cost_usd` 返回值看哪些测试失败——18/32 / 2/32）证明了测试断的是真合约而非实现噪音，**下一版起列为标配收尾步骤**；但有一个"meta-invariant"（"今日 cost sum 不重新查价格表"）虽在设计第 10 条里隐含、却没单独成测，被 test lead 抓到——下一版起**要在 plan 阶段把架构师每条 invariant 显式枚举成"test name"，凡是"实现里有 comment 解释为什么这样做"的地方都必须有同名测试钉住**，避免"实现对了但没人测"的回归裂缝。
