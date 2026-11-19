# D 正式矩阵分析口径

记录日期：2026-11-08。此分析口径在正式矩阵运行期间、未查看策略延迟数值或策略差值之前写定；只检查了 trace 字段结构、写入完整性元数据和已完成 run 的状态计数。

## 分析单位与配对

- 主要独立单位为 trace seed。每个 grid × 文本流数 × 到达模式 × seed 下的各策略按完全相同的外生到达与 seed 配对。
- 每个 seed-policy run 内计算请求指标；每个策略条件用 12 个 run-level 结果汇总。请求和 token 不作为独立重复样本。
- 对每个候选策略与 N 的 run-level 差值，按 12 个配对 seed 做 bootstrap 均值区间；固定分析 seed `20261107`，重采样 5000 次。

## 延迟与 goodput

- 文本 token gap 来自客户端各文本 lane 的连续 token 交付时间，跨文本请求边界连接，并仅统计与 180 秒测量窗相交的间隔。报告每个 run 的 P50/P95/P99 和最大值，再对 12 个 seed 的 run-level 指标汇总。
- 图文 TTFT 为客户端首 token 时间减去客户端实际发送时间；按测量期所有图像请求计算每个 run 的 P50/P95/P99。
- 联合 SLO goodput 按图像请求计数。图像首 token 必须在对应 TTFT 阈值内；从该图像实际发送至首 token 返回的重叠区间内，每条配置文本 lane 的最大 client token gap 必须在文本阈值内。报告通过图像数/180 秒和通过比例。
- 主阈值为文本 30 ms、图文 55 ms；敏感性阈值为 25/45 ms 与 40/75 ms。
- 图像队列等待从实际发送到首个 encoder scheduler step；记录开放环实际发送相对计划发送的抖动。另报告 engine-ready → API-yield → client-first-token 分段时延。

## 完成率与 trace 核对

- 图像完整响应定义为无错误、finish reason 为 `length` 且至少返回 16 token；超时按错误字符串识别，其他缺少完整响应归为 incomplete。漏掉的 client response 由 planned/sent 与 response 计数差报告。
- 文本完整响应定义为无错误、finish reason 为 `length` 且返回 192 token；分母是测量窗内记录到开始时间的 response。超时、其他错误和不完整响应分别报告。由于客户端没有独立的文本请求发送事件，未落成 response 行的取消无法精确计数，保留为限制。
- 图像 server/API/client 首 token 以完全相同 request ID 连接；检查首 token ID一致和 engine-ready ≤ API-yield ≤ client-first-token。服务端异步写入器的 submitted/written/dropped/write_errors 汇总也进入审计。

## 估计器与调度动作

- 冻结 profile 保持不变。N/F/U 的首次 engine-ready 减 LM prefill scheduler step 与 `E_hat + P_hat` 比较；确认实际分步的 S/A 请求以 engine-ready 减 P scheduler step 与 `P_hat` 比较；E step 到 P step 的额外间隔与 `split_gap_hat` 比较。decode scheduler inter-step gap 与 `D_step_hat` 比较。
- formal serving 未启用 calibration-only CUDA event flush，因此留出期不能单独测得 GPU 视觉前向 E；同时报告合并 E+P 误差。`visual_forward.cpu_elapsed_ms` 仅是 CPU dispatch-side proxy，不称为 GPU execution latency。
- A 的 request-level decisions 按 D slack、I slack 符号及 E/P 比值中位数分箱；逐箱报告选择的 N/S、split eligibility、实际 E/P 是否分步、trace 完整性和联合 SLO 结果。

## 可复现标识

- Schedule SHA-256：`811cfc12e174455ac6386e9f008f5031e9bc72f936272beccc4000810300cbb7`
- Frozen estimator profile SHA-256：`92bb071b94599533a79113f2bf519dff785b75788ffd27ccd7d8ffe0c9ac9483`
- 分析实现：`continuation/D/scripts/analyze_formal_matrix_d.py`

## 2026-11-09 分析关联键审计补充

矩阵完成后的第一次分析保留在 `results/formal-analysis-pre-id-join-fix/`。检查状态分区时发现内部 scheduler/decision request ID 带多段后缀，不能直接与客户端 image request ID 精确相等；第一次输出的主客户端指标仍可用，但动作关联和状态分区不作为最终结论。

第二次分析修复了运行级/A 决策汇总的 request-ID 解析，并确认 4,616/4,616 个 A 选 S 决策都对应实际 E→P 拆步；随后发现状态分区末端又直接用 decision ID 查找 image ID，故该次输出只作为部分修复审计记录，保留在 `results/formal-analysis-partial-idfix/`。

第三次分析进一步在状态分区末端统一通过已知图像 request ID 集合解析内部 ID，再关联客户端图像结果。最终唯一权威目录为 `results/formal-analysis-final-idfix/`；最终报告必须核对状态分区的 trace 顺序、A 选 S 数与实际拆步数、联合 SLO 通过计数。三个目录及对应脚本版本均保留，不删除中间证据。最终分析脚本 SHA-256：`b5a6f85536a44d5b53690265190684a44af4881a1f0f4c499d126b5424f109b8`。

留出正式 trace 没有启用 calibration-only CUDA event flush；因此不能从 `visual_forward.cpu_elapsed_ms` 推断 GPU E 时长。它只是 CPU dispatch-side proxy。留出报告须继续保留完整视觉答案质量未测、图像最大输出固定16 token等限制。

### 2026-11-09 异步写入元数据去重

第三遍状态修复版已将每个 A 决策按归一化后的 request ID 连接至唯一客户端图像结果，状态分区确认决策总数 9,504、实际选择 S 共 4,616，且这 4,616 条均实际执行 E→P 拆步；状态分区关联的 trace 顺序均有效。该中间结果在 `results/formal-analysis-statefix-pre-dedup/` 保留。

进一步审计发现：四个 paired seed 共用一个 server-event log，因此从每个 run metric 累加 `.async-meta.json` 会把 writer 统计乘以 4。原始 108 份唯一元数据的核对值为 `closed_cleanly=108/108`、`written=153,424,042`、`dropped=0`、`write_errors=0`。分析器现以唯一 server log 路径去重聚合这些计数；修正版 SHA-256 为 `78f75c666a2087cfca92d49302e46d95c7b0c78dc712371cd98b0f83cd3d05fc`。最终输出目录为 `results/formal-analysis-final-dedup-idfix/`；完整重跑完成后，仅该目录作为正式最终结果。

### 2026-11-09 最终分析输出与去重复核

第四轮完整分析的唯一权威输出为 `continuation/D/results/formal-analysis-final-dedup-idfix/`。432/432 个正式 run、36 cells、196 paired comparisons 均入账；A 决策关联 9,504/9,504，trace 顺序和首 token ID 各 46,656/46,656。108 个唯一 `.async-meta.json` 对应 108 个 server-event log（每个 log 被四个配对 seed 共用），故按 log 路径去重后写入总数为 153,424,042，dropped 与 write errors 均为 0。该目录的 cell summary、paired bootstrap、run metrics、state partition 和 request decisions/images 均与第三轮状态关联修正版逐项相同，差异仅在被重复四次的 writer 计数。

最终分析器 SHA-256 为 `78f75c666a2087cfca92d49302e46d95c7b0c78dc712371cd98b0f83cd3d05fc`；退出码 0，分析 CPU 耗时约 23 分 23 秒。旧版 hash `b5a6f855...` 和 `formal-analysis-final-idfix/` 属于先前审计轮次，不能再标为最终权威版本。
