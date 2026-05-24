# Architect — lessons learnt

每版一句，由本轮迭代结束的复盘产生。

- **v0.6 / `mneme search`**：设计契约里写了"`-k` 超大不报错"，但没去验底层依赖的物理硬墙（sqlite-vec 的 4096 上限），导致 developer 实现后仍撞墙被 tester 抓到——契约写完必须主动验真所有所依赖组件的边界值，不能假设"按需求写就够"。
