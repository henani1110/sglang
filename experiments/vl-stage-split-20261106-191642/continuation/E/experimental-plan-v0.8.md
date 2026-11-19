# E 阶段实验方案 v0.8

日期：2026-11-09  
状态：内容路由使用未校准的 pre-E 候选预算路径；live callbacks 已完成 512/768 grid × 2/6 lane × uniform/burst 八格短 replay，策略仍 keep-all；600 项质量集与 formal online matrix 均未开始。

## 本版收敛

v0.6 预想在真实 S_out DIG 后把 K 反馈给 scheduler 并执行 pre-P 原子压缩。检查固定 vLLM 0.33 源码后发现，特征张量留在 GPU worker 的 EncoderCache；scheduler 的 `Request`、multimodal spans、KV blocks 位于 EngineCore。忠实实现需要 worker→EngineCore 的 per-request DIG/keep-index 输出，以及 scheduler→worker 的下一步索引/encoder-cache 更新。若只改视觉输出行数，`prompt_token_ids`、`mm_features.mm_position`、KV 与 worker embeddings 无法一致。

近期具体路线因此改为在 scheduler admission 前计算 pre-E query 与 raw-pixel 风险代理，使用 E 已验证的 `mm_processor_kwargs.images_kwargs.e_keep_ratio` 将整数 K 写入 placeholders；E 仍处理完整 ViT patch rows，再按相同 K 选 merger groups 并同步 DeepStack/M-RoPE。这是一条 `SieveServe-inspired pre-E proxy` 路线，不是论文中的 ModernBERT QCR / embedding DIG。真实 DIG 可在校准阶段离线采集比较；post-E feedback 暂不作为当期必需实现。

## 策略原语与 shutdown 诊断

- `scripts/e_content_budget_policy.py` 新增 raw RGB 空间方差 proxy、pre-E 双守卫 route、受 `K_safe` 下界约束的内容 cap + 双期限 keep-count 选择，以及单图 request kwargs helper。无 calibration ID 或缺失信号一律 `effective_keep_ratio=1.0`。
- vLLM E processor 的计数规则为 `ceil(N × ratio)`；`K/N` 的浮点乘积偶尔上舍入。helper 改用区间中点 ratio `(K−0.5)/N`；远端 Python/Torch 自检穷举 N=1..2048 的所有 K，无一计数偏差。该自检只验证数学/API 参数路径，不验证答案质量。
- 搜索 E 与更早原始 server logs，共 133 份日志出现单个 resource_tracker semaphore 警告，其中 97 份在 raw 阶段、36 份在 E。固定 vLLM 0.33 `MultiprocExecutor` 创建 `context.Lock()`，`--shutdown-timeout` 默认 0 秒；旧日志曾记录 EngineCore 被立即 force-kill。这些事实支持“默认关闭等待与警告有关”的解释，尚非单独对象的栈级证明。
- `scripts/run_server_e.sh` 现显式设 `--shutdown-timeout 15`。两个非正式诊断（文本 completion 与单张 512×512 synthetic chart）均 HTTP 200；EngineCore 以 drain mode 退出，API shutdown complete，无 force-kill 字样、无 semaphore warning。图像诊断 writer 为 61/61 和 33/33 written，drop/error=0、clean close；端口释放，GPU0/1 恢复 782/45 MiB。原始材料见 [shutdown-timeout 调查报告](runs/processor-smoke/shutdown-timeout-investigation-v0.1.md)。

## 固定环境与硬件

- 只使用 `chx-gpu-4090-2` GPU0 RTX 4090；负载 PCIe Gen4 ×16。GPU1 是 Gen4 ×4，不进入 E 的实验臂。
- 权重、缓存、输入、trace 与结果放 SSD `/media/chx/Win10/vl-stage-split-20261106-191642/continuation/E/`；本地 E 目录保留可校验副本。
- 沿用 D 固定的 Qwen3-VL-2B-Instruct revision、vLLM 0.33.0 与 PyTorch 2.14.0+cu130；A/B/C/D 保持只读。
- 所有新 E service 用 15 秒 shutdown timeout，确保 worker 正常 drain；该项只影响 teardown，不改变 measurement treatment。

## 具体执行顺序

1. **完成 rights review**：POPE/GQA/TextVQA/DocVQA/ChartQA 各筛 120 个项目；逐项核对 annotation grant 与图片 source/license。当前所有候选仍 pending，eligible=0，未下载、未评分。
2. **冻结 calibration/heldout**：按唯一图像去重；保存 query、label、source/hash、raw RGB、processor tensor、512/768 grid 与 prompt。全量输入与多个 K 做配对，按官方 scorer 逐例评分；同时提取真实 `S_out` DIG 和 raw-pixel proxy，验证 proxy 与内容风险排序。
3. **冻结 `K_safe(x,q)`**：仅用 calibration split 选每类任务/风险的最小保真预算，在 heldout 上报告配对质量差与区间；失败项目保留。若 pre-E proxy 对高风险样本排序无效，继续改 proxy 或采用明确可用的 QCR 模型，不能打开不安全 pruning。
4. **测 K→系统成本**：GPU0 单卡，512/768 × 2/6 text lanes × uniform/burst arrival；分别量 E encoder、LM P、text max-gap、image TTFT、CPU API/dispatch、H2D、KV 与队列。full ViT E work 和 reduced decoder/KV token 数分开报。
5. **接入外部 request controller**：`scripts/e_live_joint_controller_e.py`、`smoke_stage_split_e.py`、`replay_live_joint_controller_e.py` 已连接 text SSE fragment callback、lane start/finish、预定 client arrival timestamp 与逐图 `mm_processor_kwargs`。512/768 grid × 2/6 lane × uniform/burst 八格、每格 4 张图的短 replay 完成；请求均 HTTP 200。文本 gap max 范围 17.5–53.3 ms；512 的联合 30/55 逐图通过数为 4/4、4/4、3/4、1/4，768 为 2/4、0/4、2/4、0/4（按 2-lane uniform/burst、6-lane uniform/burst 排列）。每格 n=4 且跨网格并非配对，只作测量预检。下一步延长窗口、平衡/反序条件顺序并重复 paired seeds，报告每 lane gap 分布与 image TTFT；保留 N/S 对照。
6. **冻结并执行正式矩阵**：8 workload cells × N/A/B-fixed/B-content/B-joint × 12 paired seeds = 480 runs；每个 60s warmup +180s measurement，同 seed 各臂共用图像、请求与 arrival times，按区组平衡/反序运行。threshold、K_safe、cost estimator、config hash、seed、trace 与分析代码先冻结。
7. **分析归档**：主报联合文本 30ms 最大间隔 / 图像 55ms TTFT goodput，并报告 25/45 与 40/75 敏感性、答案质量、时延分位、错误/超时/饥饿、实际 K、E/P CUDA 与 CPU/H2D/KV。按 seed 做系统配对 bootstrap、按图像做质量配对区间。所有失败 trace 保留，修复后补跑。

## 不设置停止门槛

效果、SLO、精度、置信区间、资源和墙钟都不是停止条件。OOM/Xid、磁盘/服务异常时只安全结束当前批次、记录并修复资源问题，再从缺失 seed 续跑。仅用户明确取消 Goal 才终止。

## 当前计数

- rights-cleared quality items：0/600；正式质量 runs：0。
- 480 formal online runs：0/480。
- 当前策略仍未校准，默认有效 keep ratio=1.0；v0.4 CPU/远端策略 self-check 与 shutdown smoke 不计入正式 runs。
- live controller self-check 覆盖双 lane slack、安全 fallback、内容 cap 与双期限选择；2/6 lane mixed smoke 验证了 callbacks、request kwargs、S scheduler 切步和事件关联。策略因 dense proxy route 且未校准而保持 N/N、ratio 1.0，不能算 pruning、质量或 deadline 改善证据。
- Goal E 保持 active；本次 Mac 无法解锁，模型 UI 切换未执行，工作继续在当前模型上推进。

## 证据索引

- [内容风险预算设计 v0.2](content-risk-budget-design-v0.2.md)
- [v0.6 旧方案](experimental-plan-v0.6.md)
- [SieveServe 与 GeoPrune 文献审计](literature-and-novelty-audit.md)
- [E pre-P 实现边界审查](implementation-boundary-review-v0.1.md)
- [策略原语](scripts/e_content_budget_policy.py)
- [live D/I deadline controller](scripts/e_live_joint_controller_e.py)
- [策略 self-check v0.4](runs/processor-smoke/e-content-budget-policy-self-check-v0.4.json)
- [controller self-check](runs/processor-smoke/e-live-joint-controller-self-check-v0.1.json)
- [controller→request smoke](runs/processor-smoke/e-preencoder-controller-request-smoke-v0.1.json)
- [live 2/6 lane mixed smoke 分析](runs/processor-smoke/e-live-joint-controller-mixed-smoke-v0.1.md)
- [live 2-lane JSON](runs/processor-smoke/e-live-joint-controller-smoke-v0.1.json)
- [live 6-lane JSON](runs/processor-smoke/e-live-joint-controller-smoke-6lane-v0.1.json)
- [固定到达 replay harness](scripts/replay_live_joint_controller_e.py)
- [固定到达 replay 初始分析（768 grid）](runs/processor-smoke/e-fixed-arrival-live-replay-analysis-v0.1.md)
- [固定到达 replay 八格分析（512/768 grid）](runs/processor-smoke/e-fixed-arrival-live-replay-analysis-v0.2.md)
- [E v0.8 SHA-256 manifest（78 项）](archive/e-v0.8-live-controller-fixed-replay-sha256-v0.1.txt)
- [E v0.9 SHA-256 manifest（106 项）](archive/e-v0.9-eight-cell-replay-sha256-v0.1.txt)
- [Shutdown-timeout 调查](runs/processor-smoke/shutdown-timeout-investigation-v0.1.md)

## 2026-11-09 controller implementation update

- `[事实]` `TextDeadlineTracker`/`prepare_image_request_budget()` 合成自检 `valid=true`，通过多 lane 紧迫度、30/55ms slack、uncalibrated keep-all、dense bypass、deadline overrun 后保留质量 floor 等分支。
- `[事实 / API smoke]` 一张 512×512 synthetic chart 通过 API；controller 的 raw pixel score=0.0668625，触发 candidate dense-bypass。由于无标定，它设置 K=256/N=256、ratio=1.0；API 200，prompt/completion 277/41，finish stop。请求关联视觉事件为 output `[256,8197]`、ViT input `[1024,1536]`，正好对应 256 decoder-visible tokens 与 1024 patch rows。
- `[边界]` 此 smoke 只证明 policy→request kwargs→processor placeholders→encoder rows 的 keep-all 贯通；无同时活动文本 lane、无压缩 K、无校准数据，模型将 synthetic chart 标题回答错，故不作质量结论、不计 B-joint 或 formal matrix。
- `[事实 / 退出]` 15 秒 shutdown timeout 后 EngineCore drain、application shutdown complete；无强杀和 semaphore warning。server event writer 与 scheduler traces 已复制并纳入下一轮 SHA 清单。

## 2026-11-09 live 2/6-lane mixed-load smoke

- `[事实 / 请求]` GPU0 的 E-only scheduler S 服务分别承载两条和六条 `/v1/completions` text stream；所有文本请求和一张 768×768 synthetic chart 请求均 HTTP 200。图像提交时 tracker 分别记录 2/6 条活跃 lane。图像 prompt=596 tokens，processor grid `[1,48,48]`，N=576。
- `[事实 / 控制器]` 两次 image slack 快照约 55.0 ms，text slack 约 29.8 ms。raw pixel proxy=0.06047 命中 dense bypass，query proxy=0.685；当前 threshold 与 backbone 未校准，所以 `request_budget_active=false`、K=576、ratio=1.0。
- `[事实 / scheduler]` 两个 trace 的第 14 行均有一张图的 `scheduled_encoder_inputs`，同时所有在途 text lanes 各安排 1 个 token，图像 LM token 为 0。图像 ID 关联到 `visual_forward` output `[576,8197]` 与 ViT input `[2304,1536]`。这验证 S-mode 切步和 keep-all placeholder/encoder 一致，不验证视觉压缩。
- `[观察 / gap]` client SSE content-fragment 最大间隔：2-lane 为 51.524/51.523 ms；6-lane 为 40.893–40.960 ms，均高于 30 ms 参考目标。片段间隔是用户侧可见输出 gap 指标，不直接代表模型内部 token 边界；决策时的正 slack 只描述最近片段之后剩余时间，不能说明历史整段 max gap 达标。
- `[事实 / 清理]` 两次 API/scheduler writer 均 submitted=written、drop/error=0、clean close；服务按 15 秒 drain 正常退出，无 force-kill 或 semaphore warning。8285 释放，GPU 显存回到 782/45 MiB。详细 JSON、事件和退出日志已保存到 SSD 与本地。
- `[边界 / 决定]` 两次均为短 synthetic smoke，不计质量 run 或 480 formal runs，不能支持质量、压缩、latency、SLO 或 goodput 结论。下一步做固定 prompt/arrival trace 的长时 stream replay，保留每 lane 全量 SSE 时间戳并报告 p50/p95/p99/max gap 与 image TTFT；继续把当前 backbone 的 proxy/K_safe/cost calibration 作为策略启用前置工作。

## 2026-11-09 固定到达 2/6-lane × uniform/burst replay

- `[事实 / 请求]` 在同一 E-only S 服务上按固定顺序执行 2-lane uniform（300 ms 图像间隔）、2-lane burst（25 ms）、6-lane uniform（300 ms）和 6-lane burst（25 ms）；每格 256 个 SSE text fragments/lane、4 个图像请求。全部文本与图像请求 HTTP 200，四格所有图像 arrival 时均观察到全部目标 text lanes 活跃。
- `[事实 / SSE gap]` 每路超过 30 ms 的 content-fragment gap 数：2-lane uniform 1/255，2-lane burst 2/255，6-lane uniform 0/255，6-lane burst 2/255。max gap 分别约 51.5、53.3、21.6、39.1 ms。指标按客户端 SSE fragment 到达时间计算。
- `[事实 / image TTFT]` 从预定 client arrival 至首个 SSE 片段：uniform 的 2/6-lane p50 为 55.7/54.4 ms，burst p50 为 100.6/130.0 ms；每格仅 4 个观测，p95/p99 仅描述样本。
- `[事实 / policy]` 每格 proxy score≈0.06047 命中 dense bypass，current-backbone calibration 缺失，因此 16/16 图像均 `K=576`、ratio=1.0。未验证 pruning 或联合预算实际效果。
- `[事实 / traces 与退出]` API/EngineCore writers 合计 9,895 条、scheduler 1,041 条，全部 submitted=written、drop/error=0、clean close；service drain 无 force-kill/semaphore warning，8285 释放，GPU 恢复 782/45 MiB。
- `[边界 / 推断]` 这是单进程一次性四格短 replay，运行顺序未随机化/平衡，可能混入预热与顺序影响。6-lane uniform 的 max gap 小于 2-lane，不能据此得出并发增加改善；burst p50 较高也不能作因果结论。无质量、SLO、latency/goodput 或 formal evidence。
- `[决定 / 待办]` replay harness 已把预定到达时刻带入 TTFT，纳入 client worker queue。下一轮应长时、平衡并重复 arrival traces，保存每请求 scheduler join 与 SSE timing；继续 rights closure、current-backbone proxy vs `S_out`/quality calibration。详细表格见 [replay 分析](runs/processor-smoke/e-fixed-arrival-live-replay-analysis-v0.1.md)。

## 2026-11-09 512-grid replay 补齐与八格预检

- `[事实 / 请求]` 512 图像 processor grid 为 `[1,32,32]`、N=256；按固定 shuffle seed `20261109` 依序执行 6-lane uniform、6-lane burst、2-lane uniform、2-lane burst。每格 4 张图，文本每 lane 256 个 SSE fragments；所有 16 张图和 16 条文本请求 HTTP 200，图像到达时目标 lanes 都活跃。
- `[事实 / 联合测量]` 按正式口径用预定图像 arrival 至首个 SSE 的 TTFT，并计算该重叠区间各 lane 的 max gap：512 的联合 30/55 逐图通过数为 2-lane uniform/burst 4/4、4/4，6-lane uniform/burst 3/4、1/4。768 的对应值为 2/4、0/4、2/4、0/4。
- `[观察 / 限制]` 512 两个 2-lane cell 本次通过 4/4，512 6-lane burst 为 1/4；768 burst 均为 0/4。但 n=4、768 顺序固定、512 与 768 分开服务且非配对，预热/顺序效应未隔离。不能从本次跨网格数值推断因果或稳定 goodput。策略全程 keep-all，D 30/55 的开放环预检，不是 formal 运行。
- `[事实 / 执行形态]` 512 请求-linked visual event 有 16 个单图编码，输出 `[256,8197]`、ViT input `[1024,1536]`；768 16 张图落在 15 个 encoder events，其中一行合并 2 张、输出 `[1152,8197]`。完整 trace 未见 image-token 压缩。
- `[事实 / 清理]` 512 事件 writer API/EngineCore 合计 9,897 条、scheduler 1,041 条，全部 written、drop/error=0、clean close；EngineCore drain 正常、无 force-kill/semaphore warning，8285 释放，GPU 回到 782/45 MiB。
- `[决定 / 下一步]` 将八格短 replay 作为 harness/量测预检；接下来需使用跨 grid 相同的 paired seeds、平衡/反序运行和长时窗口复测，再继续完成 rights closure、内容/质量与成本 calibration。详细结果见 [八格分析](runs/processor-smoke/e-fixed-arrival-live-replay-analysis-v0.2.md)。
