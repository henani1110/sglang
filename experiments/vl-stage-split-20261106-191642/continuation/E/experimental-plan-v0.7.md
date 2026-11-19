# E 阶段实验方案 v0.7

日期：2026-11-09  
状态：内容路由实现转为 pre-E 候选预算路径；live text/image deadline controller 原语和单图 API smoke 已通过；600 项质量集与 formal online matrix 均未开始。

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
5. **接入外部 request controller**：`scripts/e_live_joint_controller_e.py` 已实现 `TextDeadlineTracker`、30ms 文本 gap / 55ms image TTFT slack 快照、风险预算 + `K_safe` 下的 keep-count 选择和精确 per-request `mm_processor_kwargs` 输出。下一步把每个 text SSE lane 的 token callback 与 request start/finish 接到正式 load generator；按图像 arrival 算 slack，deadline 已过时仍保留 floor 并报告 overdue。再做 2/6 text lane × uniform/burst 的 API replay，并在决策可执行时比较 D 的 N/S placement。
6. **冻结并执行正式矩阵**：8 workload cells × N/A/B-fixed/B-content/B-joint × 12 paired seeds = 480 runs；每个 60s warmup +180s measurement，同 seed 各臂共用图像、请求与 arrival times，按区组平衡/反序运行。threshold、K_safe、cost estimator、config hash、seed、trace 与分析代码先冻结。
7. **分析归档**：主报联合文本 30ms 最大间隔 / 图像 55ms TTFT goodput，并报告 25/45 与 40/75 敏感性、答案质量、时延分位、错误/超时/饥饿、实际 K、E/P CUDA 与 CPU/H2D/KV。按 seed 做系统配对 bootstrap、按图像做质量配对区间。所有失败 trace 保留，修复后补跑。

## 不设置停止门槛

效果、SLO、精度、置信区间、资源和墙钟都不是停止条件。OOM/Xid、磁盘/服务异常时只安全结束当前批次、记录并修复资源问题，再从缺失 seed 续跑。仅用户明确取消 Goal 才终止。

## 当前计数

- rights-cleared quality items：0/600；正式质量 runs：0。
- 480 formal online runs：0/480。
- 当前策略仍未校准，默认有效 keep ratio=1.0；v0.4 CPU/远端策略 self-check 与 shutdown smoke 不计入正式 runs。
- live controller self-check 覆盖双 lane slack、安全 fallback、内容 cap 与双期限选择；单图 API smoke 也已用其输出参数成功完成，但没有 live text lane callback，不能算 B-joint mixed-load 验证。
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
- [Shutdown-timeout 调查](runs/processor-smoke/shutdown-timeout-investigation-v0.1.md)

## 2026-11-09 controller implementation update

- `[事实]` `TextDeadlineTracker`/`prepare_image_request_budget()` 合成自检 `valid=true`，通过多 lane 紧迫度、30/55ms slack、uncalibrated keep-all、dense bypass、deadline overrun 后保留质量 floor 等分支。
- `[事实 / API smoke]` 一张 512×512 synthetic chart 通过 API；controller 的 raw pixel score=0.0668625，触发 candidate dense-bypass。由于无标定，它设置 K=256/N=256、ratio=1.0；API 200，prompt/completion 277/41，finish stop。请求关联视觉事件为 output `[256,8197]`、ViT input `[1024,1536]`，正好对应 256 decoder-visible tokens 与 1024 patch rows。
- `[边界]` 此 smoke 只证明 policy→request kwargs→processor placeholders→encoder rows 的 keep-all 贯通；无同时活动文本 lane、无压缩 K、无校准数据，模型将 synthetic chart 标题回答错，故不作质量结论、不计 B-joint 或 formal matrix。
- `[事实 / 退出]` 15 秒 shutdown timeout 后 EngineCore drain、application shutdown complete；无强杀和 semaphore warning。server event writer 与 scheduler traces 已复制并纳入下一轮 SHA 清单。
