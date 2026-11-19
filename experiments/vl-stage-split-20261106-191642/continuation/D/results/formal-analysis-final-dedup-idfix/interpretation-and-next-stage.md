# D 正式分析解读与下一阶段决策

最终权威统计见本目录的 `formal-analysis.json`、`formal-cell-summary.json`、`formal-paired-bootstrap.json`、`formal-state-partition.json` 和 `formal-request-decisions-and-images.json`。本报告使用第四遍完整分析；前三遍均保留为分析修复审计材料。

## 完整性

- 432/432 个预注册 run 成功，36 个策略格，每格 12 个配对 seed；196 组配对比较，9,504 条 A 决策。
- 首 token server/API/client 顺序和首 token ID 均为 46,656/46,656。
- A 状态分区共 9,504 条决策：N=4,888、S=4,616、可拆步 eligible=6,545；4,616/4,616 条选 S 的请求都实际执行 E→P 拆步，trace 顺序 9,504/9,504 有效。
- 108/108 个唯一 server-event async metadata 文件均正常关闭；共写入 153,424,042 行，丢弃 0 行，写错误 0。
- Schedule SHA-256：`811cfc12e174455ac6386e9f008f5031e9bc72f936272beccc4000810300cbb7`；冻结估计器 SHA-256：`92bb071b94599533a79113f2bf519dff785b75788ffd27ccd7d8ffe0c9ac9483`；最终分析器 SHA-256：`78f75c666a2087cfca92d49302e46d95c7b0c78dc712371cd98b0f83cd3d05fc`。

## 主要结果

联合 30/55 ms goodput 的 A−N 配对差以 12 个 trace seed 为独立单位，burst 负载如下（单位 passing image/s）：

| Grid / 文本流 | A−N 均值 [95% CI] | S−N 均值 [95% CI] |
|---|---:|---:|
| 512 / 2 | +0.00139 [−0.04861, +0.04722] | −0.02037 [−0.08657, +0.03056] |
| 512 / 6 | +0.00093 [−0.01343, +0.01435] | −0.01713 [−0.03148, −0.00278] |
| 768 / 2 | −0.00556 [−0.01019, ≈0] | −0.00694 [−0.01019, −0.00370] |
| 768 / 6 | +0.00231 [+0.00046, +0.00463] | 0.00000 [0.00000, 0.00000] |

A 在多数负载格相对 N 接近零；768×2 burst 有负向差值，768×6 burst 有小幅正向差值。S 在 512×6 与 768×2 burst 出现显著负向差值。所有策略格的图文、文本完成率均为 1.0。768 网格各策略格的 run-level 图像 TTFT P95 的 seed 中位数约 74.2–84.5 ms；A 在 768×2/6 burst 的 30/55 goodput 分别约 0.0014/0.0023 passing image/s，放宽到 40/75 后约 0.714/0.656。瓶颈主要表现为期限违约，而不是服务失败或漏答。

## 解释边界与下一步

D 说明真实在线 E/P 拆步和逐请求 N/S 决策可以按校准规则执行，但现有结果不支持把仅有 N/S deadline-slack 选择称为跨负载普遍有效的策略。正式留出 serving 没有 CUDA-event 视觉计时；`visual_forward.cpu_elapsed_ms` 仅是 CPU dispatch-side proxy，不能称为 GPU E latency。图文答案最大输出固定为 16 token，未评估视觉答案质量；因此 D 不证明 token 压缩的准确率或安全性。

下一阶段建议收敛为**单卡混合图文共服务中的内容风险感知视觉预算与文本/图文双期限联合控制**。冻结 600 个质量样本（POPE、GQA、TextVQA、DocVQA、ChartQA 各 120）；在线使用 512/768 grid × 2/6 文本流 × uniform/burst，5 臂 N、D 的 A、GeoPrune 固定预算、QCR/DIG 风格内容预算、内容预算加实时 D/I slack，共 480 个配对 run（12 seeds、每 run 60 s warmup+180 s measure，名义 32 GPU 小时）。GeoPrune 与 SieveServe 是直接近邻，不主张首次提出 token pruning 或 deadline-aware serving；其与单卡同时承载活跃文本流和开放环图像请求的关系，仍需文献复核和实验验证。

SieveServe 提出 QCR/DIG 请求级内容预算，正式 serving 在 4×A100 上分别测图像密集和文本密集 workload；“没有直接验证本课题这类单卡混合双 deadline workload”是对论文评测设置的推断。[SieveServe，EMNLP 2026](https://aclanthology.org/2026.emnlp-main.648/) GeoPrune针对 Qwen3-VL/Qwen3.5提出视觉输出空间的 engine-friendly token 排名，并报告 vLLM 集成。[GeoPrune，Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1913/) 高分辨率同时增加视觉编码与 LM prefill 成本的准确率—时延张力见 [FastVLM，Apple ML Research](https://pr-mlr-shield-prod.apple.com/research/fast-vision-language-models)。

## 图表

- [联合 30/55 goodput](plots/joint-goodput-30-55.svg)
- [A/F/S 相对 N 的配对差](plots/paired-joint-goodput-delta.svg)
- [A 决策状态与实际动作核对](plots/adaptive-state-audit.svg)
