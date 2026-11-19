# D 阶段执行记录

当前目标是完成“按两类期限余量选择 E/P”的预注册开放环评估。执行规则以 `ai_infra_3gpu_research/视觉编码干扰_后续研究/视觉编码与Prefill拆步实验方案.md` v4.3 为准：不设任务级效果、SLO、证据、资源、墙钟、会话或失败次数停止门槛；未完成即保存检查点并续做。安全问题只暂停并收尾当前批次，诊断修复后继续。

## 固定环境

- GPU：`chx-gpu-4090-2` GPU0 RTX 4090，负载 PCIe Gen4 ×16。GPU1 RTX 4090、负载 PCIe Gen4 ×4，不参加。
- 模型：`Qwen/Qwen3-VL-2B-Instruct`，revision `89644892e4d85e24eaac8bacfd4f463576704203`。
- vLLM：0.33.0，commit `4ba787ec9959e566d3b587a796dff07673324da5`。只读复用SSD权重 `/media/chx/Win10/vl-encode-pilot-20261106-085428/models/Qwen3-VL-2B-Instruct`；D的HF/vLLM/Triton/FlashInfer cache 与新结果写入当前D SSD目录。
- 新图片、权重缓存、trace 与结果写入 `/media/chx/Win10` SSD；代码只写入本 D continuation 的隔离 overlay。

## 矩阵

512/768 processor grid × 2/6 条文本流 × uniform/burst 开环图像到达。uniform 为 0.1 image/s；burst 为每 2 秒两张，间隔 100 ms。策略为 N/F/S/A，burst 另含 U；每个 cell 12 个配对 seeds，3 块、每块 4 seeds，每 seed 60 秒预热 +180 秒测量。共 36 个 policy cells、432 个 seed-policy runs，估算测量时间 28.8 GPU-hours（仅排程，不是上限）。

## 当前检查点（2026-11-08）

- 本地 `datasets/holdout-final/` 已完成 32 张留出图最终版：32/32 源文件 SHA-256 唯一，逐文件 CC0/CC-Zero 与 CC0 1.0 URL 核验通过，512/768 两档共64张处理图哈希核验通过。清单 SHA-256 为 `9791de5f45af94c58e2be26ae8f62fb603887d9ae80ccc0cf86ed5e0d68fa571`；审计细节见 `datasets/holdout-final/holdout-qa.json`。视觉复核发现并移除一张同场景近重复图，替换为独立来源图。
- 最终数据已同步至远端 SSD，并逐项重算源文件、512/768处理图与几何控制图 SHA-256；32/32 source 和64/64 processed files通过。最终远端manifest SHA-256为 `9791de5f45af94c58e2be26ae8f62fb603887d9ae80ccc0cf86ed5e0d68fa571`。
- 独立校准已完成并冻结：64/64个analysis-eligible runs，16个cell覆盖512/768×2/6文本×N/S×uniform/burst；seeds 300–303每cell 4/4，1056张计划测量图全部通过精确请求ID、首token ID和engine→API→client顺序QA。索引68个尝试中4个早期a0无效记录原样保留并排除。四格profile SHA-256为92bb071b94599533a79113f2bf519dff785b75788ffd27ccd7d8ffe0c9ac9483；参数见 manifests/estimator-profile.json，详细样本见 results/calibration-report.json。
- 早先attempt a0的4个运行和原始trace均保留，但因图文Chat Completions serving路径缺少engine/API hook，12个图像首token未关联，索引标记为 `analysis_eligible=false` 并由a1补跑；这些记录不用于 `P_hat`。新的D overlay patcher校验vLLM源SHA-256 `fbe6d3360b3a407ff4b3e3539e73d4565dc0370112a382689bc268520b86c245`，patched SHA-256 `f2bf349f50162ca25090c2bc383641033d666d32ae85dd70ac27f0c2643f8632`。
- 校准完成；正式432-run配对schedule已按v4.3预注册，SHA-256为811cfc12e174455ac6386e9f008f5031e9bc72f936272beccc4000810300cbb7，覆盖36个策略条件、108个块内策略单元。截至2026-11-08 21:06 UTC，远端权威manifest确认 D 正式矩阵 432/432 个运行成功，未见失败记录；最新完成 `D-S-G768-L6-burst-s1011-a0`。 burst 已完成 240/240 次运行满足180/180测量图像响应且图像与文本错误均为0。GPU0 RTX 4090（负载PCIe Gen4×16）承担实验，GPU1 RTX 4090（负载PCIe Gen4×4）保持未使用；数据、权重、缓存、trace与结果位于SSD `/media/chx/Win10`，剩余约165.9 GiB。正式效果分析与归档校验尚待完成，继续执行。远端manifest由编排器维护并只读拉取。
- 正式分析器人工审阅修复 A 决策 trace 路径（优先读取 `D/logs/`，兼容旧 `D/logs/formal/`），并保留测量窗口内未能关联到客户端响应的决策事件；分析器只在矩阵完成后运行。
- 正式统计口径记录在 `continuation/docs/formal-analysis-method.md`；按 seed 配对做 bootstrap 区间，联合goodput按每个图像请求及其同时在途文本lane定义。formal serving 不采 CUDA event，不能把 dispatch CPU 代理称为 GPU 视觉执行时间。
- 原始候选、采集失败清单及退避记录均保留；所有执行、失败、修复与补跑追加到 `logs/progress.jsonl`；计划单元状态保存在 `manifests/d-run-manifest.json`。

## 矩阵结束与分析修正（2026-11-09）

- D 正式矩阵已完成 432/432 个唯一计划运行，36 个策略条件每格 12 个配对 seed；权威 manifest 记录 invalid attempts 为 0、编排器退出码均为 0。正式 trace 首 token server/API/client 顺序和首 token ID 核验通过 46,656/46,656；服务端异步 writer 写入 613,696,168 行、丢弃 0 行、写错误 0。
- 首轮 5,000 次 paired bootstrap 已生成并保留在 `results/formal-analysis-pre-id-join-fix/`。审计时发现 vLLM scheduler/策略事件 request ID 会在公开 `chatcmpl-*` ID 后附加内部十六进制后缀；首轮分析器未剥离该后缀，导致 scheduler E/P 事件及 A 决策结果不能与客户端图像请求完整关联。检查证明 A 的 9,504 条决策都能通过已知请求 ID 精确前缀唯一关联，故没有丢弃或重跑任何正式实验。
- 第二遍分析已修复运行级/A 动作的内部 ID 后缀匹配，结果保留在 `results/formal-analysis-idfix/`。后续审计又发现状态分区末端仍有 raw ID join，且 server async writer 元数据按共享日志重复累计；因此第二遍只作部分修复审计记录，不作为最终正式报告。各修复版本和最终权威目录见下文“矩阵结束与分析修正”及后续审计段落。
- 本地截至本次检查 D 目录仍只含正式分析输出和原始小型材料，约 41 GiB 原始 formal JSONL 事件保留在 SSD `/media/chx/Win10`；最终总归档计划以 SSD 原始 D 目录为准，校验后复制完整归档回本地。

- 第二轮检查确认：策略指标里的 `policy_actual_split_count=4616/4616` 已正确关联，但最终状态分区汇总仍因另一处精确 ID join 而将实际拆步、trace 与联合 SLO 计数记为 0。该完整性问题已定位；中间输出另存 `results/formal-analysis-partial-idfix/`，现在将最后一层 decision→image outcome 关联也改为基于已知图像 ID 的逐段归一化，并用相同 432-run 原始 SSD trace、5000 次 bootstrap 和 seed `20261107` 第三次重跑至 `results/formal-analysis-final-idfix/`。

### 异步写入汇总再次核对（2026-11-09）

- 第三遍 `formal-analysis-final-idfix/` 已完整生成并保留为 `results/formal-analysis-statefix-pre-dedup/`（本地与 SSD 同名）。这版状态关联正确：9,504 条 A 决策中，N=4,888、S=4,616；S 实际 E→P 拆步 4,616/4,616；状态分区中 trace 顺序有效 9,504/9,504。
- 对照原始 `.async-meta.json` 后发现，108 个 server-event log 按块由4个配对 seed共享；第三遍将同一元数据按 run 重复累计四次。原始 108 份元数据全部 `closed_cleanly=true`，实际 `written=153,424,042`、`dropped=0`、`write_errors=0`。该问题只影响 writer 计数，不影响 run/client 延迟、goodput、paired bootstrap 或动作关联。
- 分析器现按唯一 server log 路径聚合计数，SHA-256 为 `78f75c666a2087cfca92d49302e46d95c7b0c78dc712371cd98b0f83cd3d05fc`，项目与 SSD 副本哈希一致。第四遍完整重算输出目录为 `results/formal-analysis-final-dedup-idfix/`；仅此目录作为最终权威结果，前三个分析目录均保留作修复审计。

## 最终权威分析与阶段结论（2026-11-09）

第四轮全量分析已成功结束（退出码 0，分析 CPU 耗时约 23 分 23 秒）。唯一权威结果目录为 `results/formal-analysis-final-dedup-idfix/`；该轮使用修复后的分析器（SHA-256 `78f75c666a2087cfca92d49302e46d95c7b0c78dc712371cd98b0f83cd3d05fc`）及相同冻结 schedule/profile、5,000 次 bootstrap 和分析 seed `20261107`。432/432 run、36 cells、196 paired comparisons；request trace 顺序和首 token ID 均为 46,656/46,656。9,504 条 A 决策的状态关联完整（N=4,888、S=4,616），所有选择 S 的请求均实际拆步。108 个唯一 server-event writer metadata 文件全部正常关闭；真实累计写入 153,424,042 行，丢弃/错误均为 0。第三轮状态修正版保留为 `formal-analysis-statefix-pre-dedup/`；其 cell、paired、run、状态与请求输出和第四轮一致，只有按四个 seed 重复计算的 writer 行数经第四轮去重修正。

主结果与研究限制见 `results/formal-analysis-final-dedup-idfix/interpretation-and-next-stage.md`。简述：A 相对 N 的多数 burst 格差值接近零；768×2 burst 为负，768×6 burst 为小幅正值；S 在两个 burst 格出现负向差值。所有格完成率为 1.0，主要失败是 deadline miss。不能从本阶段推断视觉回答质量或把 CPU dispatch proxy 解释为 GPU E latency。

当前研究方向收敛为内容风险感知的视觉 token 预算与文本/图文双期限联合控制。可复制的后续 Goal prompt 在 `continuation/docs/goal-prompt-next-stage-E.txt`；本记录覆盖此前仍显示“正式分析待完成”的检查点文字。
