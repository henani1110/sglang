# 视觉编码与 Prefill 拆步实验：A/B/C 完成归档

本地运行目录对应远端 project `/home/chx/project/vl-stage-split-20261106-191642` 和 SSD `/media/chx/Win10/vl-stage-split-20261106-191642`。数据、权重、GPU 缓存和远端原始结果优先放在 `/media/chx/Win10`。实验使用 GPU0（RTX 4090，负载 PCIe Gen4 ×16）；GPU1（RTX 4090，负载 PCIe Gen4 ×4）未用于本轮实验。

## 当前状态

- A：校正打点和异步采集复核完成，原始和补充证据保留。
- B：固定 E→D→P 拆步实现及语义/取消路径核验完成。
- C：F 校准、N/S 单图预检、U 双图队列预检，以及 512/768 图×2/6 路文本×3 块×N/F/S/U 主矩阵全部完成。主矩阵 576/576 episode 有效，无缺失配对键。
- D：本目录提供下一阶段方案和可粘贴的 `/goal` 提示词；D 尚未执行，不纳入 A/B/C 实验结论。

## C 结果摘要

- F 选定 `prefill_interval=2, lm_budget=1024`。其优势集中在 512 图/2 路文本；没有显示跨全部负载的固定优势。
- S 在 512 图负载下减少文本最大间隔；在 768 图负载下也减少文本间隔，但图文 TTFT P95 约增加 4–5 ms，联合 30/55 ms 达标率从 N 的 91.7% 降到 27.8%（2 路）和 8.3%（6 路）。
- U/N 单图配对 token IDs 完全一致，单图联合指标差异小；双图探针证实 encoder budget 576 可将两张同时到达的图像分到不同 step。
- C 的策略单元实际采用全局随机顺序，而非预注册的块内均衡顺序；已有名次调整作为描述性敏感性分析，但该设计偏差仍是限制。
- F/N 图文输出 token IDs 为 108/144 完全一致，S/N 文本为 143/144，U/N 两类均为 144/144。差异及其解释边界见输出差异复核。

## 关键文件

- [持续完成版方案 v4](视觉编码与Prefill拆步实验方案.md)
- [下一阶段 D `/goal` 提示词](continuation/docs/goal-prompt.txt)
- [C 主矩阵分析](results/C-main-analysis.md) 与 [JSON](results/C-main-analysis.json)
- [C 输出 token 差异复核](results/C-output-divergence-review.md)
- [A 异步复核](results/A-async-analysis.md)
- [B 语义核验](results/B-semantic-analysis.md)
- [A 原始证据归档](continuation/raw/A-async-evidence.tar.gz)
- [B 原始证据归档](continuation/raw/B-evidence.tar.gz)
- C 原始证据归档：`continuation/raw/C-evidence.tar.gz`（SHA-256 记录在同名 `.sha256`）
- [研究方向与收敛建议](可行性与收敛建议.md)

运行目录的逐文件 SHA-256 清单为 [SHA256SUMS.txt](SHA256SUMS.txt)。完整交付包写在此目录的同级 `vl-stage-split-20261106-191642-complete.tar.gz`，其哈希旁置于 `.sha256` 文件。
