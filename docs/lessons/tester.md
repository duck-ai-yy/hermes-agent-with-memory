# Tester — lessons learnt

每版一句，由本轮迭代结束的复盘产生。核心原则：**复杂度棘轮、error path > happy path、plan-first**。

- **v0.6 / `mneme search`**：plan-first 和 ratchet 自检都到位（19 个合约级测试，抓到 BUG-v0.6-1），但 special-char / provider-down 这类 case 只 assert `exit_code == 0` 是"happy-path 伪装成 error-path"，且对"纯 read 契约"（events.jsonl 字节不变 / `chat_calls == 0`）零监控——error-path 测试必须同时 assert **可观察输出**（substring / 不变量），read-only 契约要主动写 before/after snapshot，不能只信 exit code。
