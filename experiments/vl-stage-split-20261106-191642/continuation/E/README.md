# E 阶段工作区

**Goal：** 单卡图文共服务下，内容风险感知的视觉 token 预算与文本/图文双期限联合控制。

**最近状态：** 截至 2026-11-09，E-v0.9 已完成 N baseline 合成负载校准。v0.1 的 96 个 replay 全部成功，但 750 ms uniform 到达后段没有活跃文本 lane，保留为负载诊断。v0.2 以 200 ms uniform 重跑 96 个 replay；1,152/1,152 次测量图像到达都有 2 或 6 条目标文本 lane 活跃；1,536 个测量 API 请求全部 HTTP 200，1156 张 PNG 的文件 SHA 与请求 JSON 一致。两个块都只用于工程校准，正式质量评估仍为 0，正式在线策略矩阵仍为 0/480。600 个样本仍未完成权利清查：eligible=0/600；此前 120 个 Flickr 页面自动读取仅保留为过程材料，不作授权证据。GPU0 负载 PCIe Gen4 ×16 是唯一主测卡；GPU1 Gen4 ×4 排除；数据/权重/缓存/trace/输出放 SSD `/media/chx/Win10`，本地留副本。A/B/C/D 只读保留。

E-v0.8 的八格短 replay、keep-all 状态和旧的 0–4/4 短时 deadline 观测保留为前序开发记录，不用来推断 v0.9 或正式策略效果。详见[实验方案 v0.9](experimental-plan-v0.9.md)及[N 基线校准审计](runs/n-baseline-calibration-v0.2/20261109T095608Z/analysis-notes.md)。

## 当前收敛判断

- `[事实]` D 的 432/432 个正式 run、配对 trace、最终分析和 SHA-256 归档校验已完成。D 的图文输出最多 16 token，正式 serving 未启用可单独测 GPU 视觉编码的 CUDA-event flush；不能据此推断视觉答案质量，也不能将 `visual_forward.cpu_elapsed_ms` 称为 GPU E 时延。
- `[事实]` 负载期间 GPU0 RTX 4090 为 PCIe Gen4 ×16，GPU1 为 Gen4 ×4；E 只使用 GPU0。最近一次 S2 服务退出后 GPU 回到 782/45 MiB，8285 端口释放，SSD `/media/chx/Win10` 可用约 158 GiB。
- `[事实]` 远端 E 基线环境位于 `/home/chx/project/vl-encode-pilot-20261106-085428/env/serve311`：Python 3.11、vLLM 0.33.0、PyTorch 2.14.0+cu130、Transformers 5.20.2、flashinfer-python 0.7.2.post1。模型 revision 和权重沿用 D 已固定版本，只读复用 SSD 权重。
- `[观察]` DuetInfer 已直接研究单 GPU 文本/图文混合请求，并用 RTX 4090/Qwen2-VL-2B 做消融。因此“单卡混跑”不是新颖点。其软件版本和模型代际与 E 不同，但消费卡与 2B 规模的重叠必须正面处理。
- `[观察]` SieveServe 已提出 query/image 风险守卫并按请求设定视觉预算；ET-Prune 也做 evidence-aware 动态预算。独立差异只能落在“安全视觉预算范围内，当前文本 decode 与图像 TTFT slack 如何共同改变预算及 E/P 放置”，且必须证明超过强固定策略与可复现的 deadline-only 对照。
- `[推断]` GeoPrune 在视觉编码器输出后剪枝，优先节省 LLM prefill/KV，不会直接缩短前面的 vision encoder E。D 中 E/P 都可占时，故应分段测出压缩究竟改善了哪一段；若仅 token 数下降而 E、KV、队列和 goodput 无改善，仍需完整报告。
- `[决定]` 先完成三项并行前置工作：复核相关系统及实现；逐样本建立标注许可与媒体来源两层 manifest；完成 Qwen3-VL 图像 token 物理压缩的无剪枝等价与状态一致性预检。通过后冻结参数，再启动 600 样本质量评估与在线矩阵。

## 最新机制验证：e-trace-v2-7

- `[事实]` 在真实 `_nested_tensors_h2d` 的 CPU tensor `.to(device, non_blocking=True)` 调用前后记录 CUDA event；CPU pin/collation 留在 event 区间之外。样本级 H2D stream interval 为 0.294/0.132/0.047 ms，载荷为 3,538,944/1,572,864/393,216 B。该值描述 event 所在 CUDA stream 的单次 copy interval，不作为 PCIe 带宽结论。
- `[事实]` 3 个图像 request 的 prompt token 为 595/275/83；视觉 event、H2D event 均为 3/3 关联到 scheduler request 与 API result/SSE yield。host batching+enqueue wall envelope 为 2.241/1.152/0.271 ms，包含 Python 批处理和异步 H2D enqueue，不是纯 CPU dispatch。
- `[事实]` 运行时 KV geometry 为 512 blocks、每 block 16 tokens、28 层、每层 page 65,536 B，即每 logical block 1,835,008 B；3 个图像请求峰值分别占 40/19/8 blocks（73,400,320/34,865,152/14,680,064 B）。这些是基于实际 scheduler block IDs 和 runtime geometry 的 reservation 重建，不等同于 `nvidia-smi` 的请求级实测显存。
- `[边界]` 每个 smoke 预算档只有一个串行合成图请求，启动 profiling event 被保留并在 join 中排除。文本请求在 event flush 尾部仍未出现 `finished_req_ids`，其末端 KV 生命周期为右删失；图像请求均看到释放事件。所有正式 run 仍为 0。

## 同 batch、多图与长宽比验证：e-trace-v2-8/9

- `[事实]` v2-8 同一 request 含两个 768×768 图像；H2D event 带两条 `mm_hash`，视觉 CUDA event 也携带两个 hash 与同一 request ID。`audit_batch_multi_image_e.py` 通过。两张 768×768 图的 prompt 为 1173 tokens；768×512 横图 prompt 为 403 tokens。
- `[事实]` 同一 4-request 并发 burst 中，v2-8 的两条并发请求未共批。v2-9 将 4 条不同图像请求通过 barrier 同时发出，scheduler 将其中 3 条排进同一 encoder step；单条 H2D 与视觉 event 都带 3 个 `mm_hash` 和 3 个 request ID。第四条独立调度。4/4 均 HTTP 200，scheduler/API/SSE/H2D/E join 通过。
- `[事实]` 同版 CPU processor 确认 square-a/b 为 `[1,48,48]`、每张 576 merged tokens；768×512 landscape 为 `[1,32,48]`、384 merged tokens。processor grid、prompt usage 和 scheduler block audit JSON 均归档。
- `[观察]` 多图视觉 event 的 `mm_hash` 主字段为 null，但 `mm_hashes` 有两个值。已把 join 条件改为要求非空 hash 列表与 request IDs，因此既收录多图 batch，也继续过滤 `mm_hashes=[]` 的启动 profiling。
- `[边界]` 12 个 request 的 scheduler trace KV audit 通过，峰值 reservation 为 278,921,216 B、容量无超限、trace 结束无存活 request。H2D bytes 是 batch tensor 的总载荷；共同 batch 中的 bytes 不拆分给单 request。以上均是合成输入、最多 8 token completion 的测量链验证，不是质量或延迟效果结论。

## 原生默认与显式默认等价：e-trace-v2-10

- `[事实]` 固定 Transformers processor 报告默认 size 为 `shortest_edge=65536`、`longest_edge=16777216`。同一 PNG、同一 prompt、greedy 生成下，native default 与显式传入这两个默认 size 在 768 和 512 两档均得到完全相同的完整响应文本、prompt/completion token 数，且 `finish_reason=stop`。
- `[事实]` processor 网格：768 square `[1,48,48]` →576 tokens；512 square `[1,32,32]` →256 tokens。相应 Chat prompt 为 597/277；固定 prompt/template 部分均为 21 tokens。262144/65536 caps 返回 prompt 277/85，即 256/64 image tokens +21。
- `[事实]` 6/6 图像 response 的 scheduler/API/SSE/E/H2D 关联审计通过；KV trace 覆盖 10 个 request，reservations 审计无容量错误且 trace 结束无存活请求。服务停止后 GPU0/1 恢复 782/45 MiB，8285 释放，SSD 可用约 161 GiB。
- `[观察 / 边界]` 65536 pixel cap 的单个短输出把 synthetic chart 错读成美国人口图；这是一个合成图质量风险提示，不是基准质量统计，也不影响默认参数等价审计。正式质量集和正式策略 runs 仍为 0。

## 文件索引

- [相关工作与新颖性审计](literature-and-novelty-audit.md)
- [数据与图像来源授权审计](data-rights-audit.md)
- [vLLM/Qwen3-VL 实现接口预检](implementation-preflight.md)
- [GeoPrune 后编码器压缩与 vLLM 调度边界审查](implementation-boundary-review-v0.1.md)
- [实验方案 v0.2（待预注册冻结）](experimental-plan-v0.2.md)
- [实验方案 v0.3（processor-level token budget 路径）](experimental-plan-v0.3.md)
- [实验方案 v0.4（GeoPrune-style scorer / S-stage 更新）](experimental-plan-v0.4.md)
- [实验方案 v0.5（请求级图像 token budget）](experimental-plan-v0.5.md)
- [实验方案 v0.6（风险预算与 keep-all 等价）](experimental-plan-v0.6.md)
- [实验方案 v0.7（pre-E 风险代理与双期限联合预算）](experimental-plan-v0.7.md)
- [实验方案 v0.8（live 2/6-lane mixed smoke）](experimental-plan-v0.8.md)
- [实验方案 v0.9（N baseline calibration 与样本权利状态）](experimental-plan-v0.9.md)
- [固定到达 replay harness](scripts/replay_live_joint_controller_e.py)
- [固定到达 replay 分析](runs/processor-smoke/e-fixed-arrival-live-replay-analysis-v0.1.md)
- [512/768 八格固定到达 replay 分析](runs/processor-smoke/e-fixed-arrival-live-replay-analysis-v0.2.md)
- [E v0.8 SHA-256 manifest（78 项）](archive/e-v0.8-live-controller-fixed-replay-sha256-v0.1.txt)
- [E v0.9 SHA-256 manifest（106 项）](archive/e-v0.9-eight-cell-replay-sha256-v0.1.txt)
- [E/N v0.1 诊断块分析](runs/n-baseline-calibration-v0.1/20261109T092927Z/analysis-notes.md)
- [E/N v0.2 负载与完整性审计](runs/n-baseline-calibration-v0.2/20261109T095608Z/analysis-notes.md)
- [内容风险预算设计 v0.1](content-risk-budget-design-v0.1.md)
- [内容风险预算设计 v0.2（pre-E proxy）](content-risk-budget-design-v0.2.md)
- [resource_tracker semaphore 调查](runs/processor-smoke/shutdown-timeout-investigation-v0.1.md)
- `scripts/e_content_budget_policy.py`：双守卫和期限预算 policy primitive（默认未校准/keep-all）
- `scripts/e_live_joint_controller_e.py`：文本 max-gap / 图像 TTFT slack admission controller primitive
- `logs/progress.jsonl`：阶段进展追加记录
- `datasets/`：许可、哈希、抽样与逐例质量结果
- `configs/`、`traces/`、`runs/`、`analysis/`、`archive/`：正式材料分目录保存

远端副本与正式数据统一写入 SSD：`/media/chx/Win10/vl-stage-split-20261106-191642/continuation/E/`。本地副本位于本目录。禁止改写 A/B/C/D 结果。

## 2026-11-09 继续记录：queue timing 与 TextVQA 候选

- `[事实 / queue trace]` `e-trace-v2-11-queue` 在 GPU0 上运行 6 个合成图请求。6/6 HTTP 200，512/768 native 与显式默认参数的完整输出逐项相同；所有请求均成功连接到首次 scheduler schedule，无缺项/负时长。API/engine 队列 event 共 30 条（每请求 5 个阶段）；原始 server/scheduler JSONL、SSE、patch、manifest、合成 PNG 和审计 JSON 均已保存。
- `[观察 / 分段时间]` 此次低并发 smoke 的 API `render_chat_request` 均值 10.78 ms，AsyncLLM 输入处理 0.111 ms，本地 handoff 0.045 ms，engine submit 到首次 scheduler schedule 均值 1.345 ms（p50 1.237、p95 2.434）；ingress 到首次 schedule 均值 13.37 ms（p50 8.779、p95 34.775）。只有 6 个 synthetic 请求，且运行时 `VL_PILOT_PROBE=1`，这些数值只验证分段可观测性，不代表在线策略性能或纯 CPU dispatch 结果。
- `[事实 / instrument]` queue events 通过独立 `VL_PILOT_QUEUE_PROBE` 开关，采用 `perf_counter_ns`，关联 public request ID、engine request ID 和 scheduler `scheduled_new_reqs`。此阶段尚未完成 probe-off / queue-only / full-probe 的配对开销校准。
- `[事实 / 资源清理]` smoke 退出后端口 8285 释放，GPU0/1 分别回到 782/45 MiB，SSD 剩余约 160 GiB。GPU0 负载链路 Gen4 ×16；GPU1 Gen4 ×4 不参与。
- `[事实 / TextVQA]` 600-item rights audit 新增 120 个 TextVQA/OI V3 候选（80 train calibration、40 val heldout）；均匹配 V3 CC BY 2.0 metadata、author 和 source URL，但原始源页核验数仍为 0，未下载图片、未标为 eligible。正式质量样本仍未冻结。
- `[待办]` 做 probe-off / queue-only / full-probe overhead paired calibration；独立核验/替换 TextVQA 图像来源；继续核对其余四个子集、GeoPrune/SieveServe 实现差异与 quality scoring；之后冻结控制器、600 项和在线 workload，正式矩阵尚未启动。

### Probe overhead 配对校准（2026-11-09；非正式 run）

- 两个区组采用相反 profile 次序；每个 server profile 有 2 次 warmup、6 次测量。相同合成图、prompt、greedy 参数、seed 与到达间隔，共形成 12 个完整配对；三组各 12/12 HTTP 200，回答内容 hash 与 usage 全部一致。
- `queue-only − probe-off` 的配对首个流式内容时间中位数为 +0.070 ms（均值 +0.281 ms），完整响应中位数 +0.093 ms（均值 +0.440 ms）。`full-probe − probe-off` 分别为 +1.791 ms（均值 +1.788 ms）和 +2.015 ms（均值 +2.061 ms）。
- queue-only 共写入 80 条 queue timing；full-probe 共 516 条事件（含 80 条 queue timing）。4 个启用 writer 的区组均 clean close，submitted=written、dropped=0、write_errors=0；两个 full-probe 区组分别 40/40 queue rows、8/8 请求连到首次 scheduler schedule，负时长 0。
- `[边界]` 这是固定 synthetic 短请求、低并发、单台 RTX 4090 的观测者效应估计。probe-off 仍载入 E overlay 并经过 hook/gate 调用，因此不是未打补丁的 vLLM 基线；首内容及完整响应还包含模型/运行时波动。不得把这些单点差值当成正式负载下的通用开销或从正式延迟中机械扣除。正式 arms 应保持相同 probe profile，原始 instrumented latency 为主，并把 profile 差作为敏感性说明。
- 原始客户端 JSONL、server/scheduler events、server/launcher logs、writer metadata、GPU 收尾状态、两个 queue audit 和分析 JSON 保存在 `logs/` 与 `runs/processor-smoke/`，同时镜像到 `/media/chx/Win10/.../continuation/E/`。
- `archive/probe-overhead-sha256-v0.1.txt` 列出这批 68 个脚本、说明和校准产物的 SHA-256；此清单也随 E 目录同步到 SSD。
- `[待办]` 继续完成授权样本筛选、预算质量校准、E/P 与 deadline 控制器实现和 480-run formal matrix；本校准不计入 480 个正式 runs。


## 2026-11-09 GQA/COCO 候选抽取（未清权）

- `[事实]` 只提取 GQA `val_balanced_questions.json` 与 COCO/Visual Genome 元数据，没有取得任何 benchmark 图片。对 132,062 个 GQA 问答流式关联后，得到 1,326 个 COCO license-metadata ID 4/5 的唯一媒体候选；固定 seed `20261109` 选出 120 个不同 COCO image ID，metadata ID 4 为 80 项、ID 5 为 40 项。
- `[事实 / 权利状态]` GQA Q/A annotation license 未从官方页确认，逐图原始媒体源页面也未核实，所以 source page、annotation permission、eligible 均为 0/120，图片下载为 0。候选脚本、CSV、摘要和 source hashes 均归档在 `datasets/`。
- `[观察]` 67,808 个问答没有 Visual Genome 映射；其余映射记录未发现 COCO image ID 缺失。Visual Genome metadata 有 290 条重复 COCO ID 记录；抽样在 COCO image ID 上去重。
- `[待办]` 继续查明问答标注的授权范围、逐图 source page 与合规获取路径；不把该候选集作为 120 个已许可 GQA 项，也继续推进其它数据子集、token 压缩实现和 controller。


- [数据权利 / GQA 新证据] 后续研究中多处将 GQA 标为 CC BY 4.0，且 Salesforce AI Research README 观察到官方页底部存在 CC-BY 4.0 badge；但 Google DeepMind ProEval README 的 GQA exception 又写 MIT。原页面可检索正文没有直读授权范围，所以本轮暂记为“reported CC BY 4.0; grant scope pending primary confirmation”，120 候选保持 pending。证据对照见 datasets/manifests/gqa-license-evidence-review-v0.1.md。

## 2026-11-09 GeoPrune 算法复核与 S-only 调度 smoke

- `[事实]` GeoPrune 正式论文给出 post-ViT/pre-merger `S_out` rank-one deflation 和 residual-L2 token ranking；参数描述含 power iteration `T=10`、eigengap `τ=2`、`K_max=3`。正文 eigengap 与 Algorithm 1 的同轮赋值/比较存在疑似排版矛盾；pre-merger patch index 到 Qwen3-VL merger 后 token 的映射也未写明。没有作者仓库或 vLLM pin 可消除歧义，故 E 暂不标“忠实复现”。
- `[观察]` `e-stage-split-s1` 暴露 Completion SSE parser 字段错误；两条文本请求均 200，图片未发送。修复脚本并保留失败 trace 后，S2 的两条文本流都先产出首 token，再提交合成图像请求。
- `[事实]` S2 的调度审计找到 image encoder work 和两条 text decode 同 step、image LM token 数为 0；KV trace 104 rows / 11 requests、0 error、trace end 0 live requests；视觉 CUDA 与 H2D event 各 1 个，均关联到 scheduler 和 API response。H2D payload 为 3,538,944 B。
- `[事实]` 独立 `geoprune_ranker_e.py` fixed-K residual scorer primitive 通过 CPU synthetic numerical self-check：`K=1`、`T=10`，per-patch score 与 exact SVD top-1 residual 的最大绝对误差为 `3.3826e-6`；4 个 merger groups 中选 2 个，输出索引顺序及 tie 处理稳定。Group mean 与 whole-group keep 是显式工程 mapping；没有 Qwen model integration。
- `[边界]` 这只证明 S-only 调度切步和事件关联机制可触发，不证明视觉 token 压缩、GeoPrune、质量、延迟改善或正式负载效果。失败与成功 trace、审计 JSON 及命令配置见 [smoke 报告](runs/processor-smoke/s-stage-split-smoke-report-v0.1.md)。
- `[事实 / 资源]` S2 服务退出后 GPU0/1 回到 782/45 MiB、8285 端口释放，SSD 可用约 158 GiB；GPU0 负载链路 Gen4 ×16、GPU1 Gen4 ×4 未参与。
- `[待办]` 冻结 residual-ranking 停止条件与 pre-merger→decoder merge-group 映射的工程解释；实现单图 fresh-request keep-all 等价和 pre-P compaction，同时让 patch encoder 数、decoder-visible token 数、DeepStack/M-RoPE/placeholders/KV 分别可审计。正式 600-item 与 480-run 仍未开始。

## 2026-11-09 v0.7 策略路径与 semaphore shutdown 诊断

- `[观察 / 路径]` 固定 vLLM 源码把实际 S_out 特征留在 GPU worker 的 EncoderCache；EngineCore scheduler 独立持有 Request、mm span 与 KV state。真实 post-E DIG 改 K 需要 worker→scheduler→worker 双向状态协议，不能只截短 encoder output。当前已验证的 request-level `e_keep_ratio` 可以在 processor 形成 placeholders 前确定 K，所以近期待落地方案改为 pre-E query/原始像素代理，并把真实 S_out DIG 留作 offline calibration/comparison。
- `[事实 / policy]` 新策略代码计算 RGB `[0,1]` raw-pixel spatial variance；与 lexical simple score 合成 dense bypass/aggressive/conservative 候选。没有 calibration ID 时仍强制 effective ratio=1.0。联合预算在 content cap 与 `K_safe` 之间使用双 deadline slack 选最大可行 K。ratio helper 采用 `(K−0.5)/N` 中点以规避 `ceil(N×K/N)` 上舍入；固定远端环境穷举 N=1..2048 全部 K 均 exact。
- `[观察 / shutdown]` 133 份历史日志中有单 semaphore warning，97 份来自 E 前 raw 阶段、36 份来自 E。源码候选是 `MultiprocExecutor` 的单个 `context.Lock()` 加默认 `shutdown_timeout=0`；两次 15 秒 shutdown 的文本/单图诊断均无 warning 和强杀，并完成 EngineCore drain。该证据支持关闭等待相关解释，但还不是 resource_tracker register 栈级根因证明。
- `[事实 / 资源]` 单图 smoke HTTP 200，prompt/completion=277/28、finish=stop；API/EngineCore writers 分别 61/61、33/33 written，0 drop/error，clean close。两次诊断后端口 8285 释放、GPU0/1 显存 782/45 MiB，SSD 剩余约 158 GiB。旧证据保持不变；E 新启动脚本显式 `--shutdown-timeout 15`。
- `[待办]` 继续逐项授权 600 项数据，完成 query/pixel proxy 与真实 S_out DIG 的配对校准、heldout `K_safe` 和 K→成本曲线；实现 client/admission controller 后先做真实 API replay，再冻结 480 paired runs。质量 run=0；online formal run=0/480。

## 2026-11-09 controller→request smoke

- `[事实]` controller 使用当前 query 和 raw RGB 生成 route 后，构造 request `mm_processor_kwargs` 并通过 Chat API 提交一张 512×512 synthetic chart。Qwen processor grid `[1,32,32]`、N=256；pixel proxy `0.0668625` 命中 candidate `dense_pixel_bypass`。因 calibration ID 缺失，最终 K=256、ratio=1.0。
- `[事实]` API 返回 HTTP 200，prompt/completion=277/41，finish=stop。request-linked visual event 记录 ViT input rows=1024、model output rows=256；说明 controller 参数到 request placeholder/encoder 对齐。模型误读 synthetic chart，明确不作为质量样本。
- `[观察 / 限制]` 合成 slack self-check 覆盖多个 text lane 的最紧 30ms slack、图像 55ms TTFT、deadline-overdue 后保留质量 floor；实际 API smoke 没有活动文本 lane，SSE timestamp callback 仍未接入在线负载生成器，不能算 B-joint mixed-load 验证。

## 2026-11-09 live controller 2/6-lane smoke

- `[事实]` client SSE callback 在两条、六条文本流的 E-only S-mode 服务中均接通；文本 lane 与 768×768 synthetic chart 请求均 HTTP 200。图片到达时 tracker 分别观察到 2/6 个活跃 lane，text slack 约 29.8 ms、image slack 约 55.0 ms。
- `[事实]` pixel proxy=0.06047 命中 dense bypass，仍无 current-backbone calibration ID，因此两次都选择 K=576、ratio=1.0。scheduler 第 14 行同时含 image encoder input 与全部文本 lane 各 1 个 decode token；image visual event rows=576、ViT patch rows=2304。
- `[观察]` 客户端 SSE content-fragment 最大间隔：2 lane 为 51.524/51.523 ms，6 lane 为 40.893–40.960 ms。点时 slack snapshot 不等于整段最大间隔合格；这些短 smoke 不构成 SLA、质量、压缩收益或正式矩阵证据。
- `[事实 / 资源]` writer submitted=written、drop/error=0、clean close；两次均由 EngineCore drain 正常关闭，无 force-kill 或 semaphore warning，GPU0/1 回到 782/45 MiB，8285 释放。
- `[待办]` 运行固定 prompt/arrival trace 的长时 replay，报告各 lane 的 p50/p95/p99/max SSE gap 和图像 TTFT；继续 rights closure 与 proxy/真实 S_out/质量 calibration。完整证据见 [mixed smoke 报告](runs/processor-smoke/e-live-joint-controller-mixed-smoke-v0.1.md)。

## 2026-11-09 固定到达 replay 四格

- `[事实]` 同一服务上完成 2/6 text lanes × uniform/burst，每格 4 张图，每路文本最多 256 个 SSE 内容片段；所有请求 HTTP 200。每次图像 arrival 时目标文本 lane 均在途。
- `[观察]` 2-lane uniform/burst 每路分别有 1/255、2/255 个间隔超过 30 ms；6-lane uniform 为 0/255，6-lane burst 为 2/255。图像 TTFT p50 uniform 为 55.7/54.4 ms，burst 为 100.6/130.0 ms。四格固定顺序且每格 n=4，预热/顺序因素未隔离。
- `[事实 / 边界]` 内容代理触发 dense bypass、无校准，所有图片 K=576/576、ratio=1.0；未测试剪枝收益。API/EngineCore writers 9,895 条、scheduler 1,041 条全部 written，0 drop/error；服务正常 drain，GPU0/1 782/45 MiB，8285 端口释放。
- `[决定]` 把此结果作为 harness/trace 预检；继续做长时、顺序平衡、paired-seed replay，并完成 rights 与风险/质量/成本标定。详细范围和限制见 [分析报告](runs/processor-smoke/e-fixed-arrival-live-replay-analysis-v0.1.md)。

## 2026-11-09 512-grid 补齐：八格固定到达 replay

- `[事实]` 512 grid 的 2/6 lane × uniform/burst 四格已在随机顺序 seed `20261109` 下完成；加上此前 768 四格，共 32 张 synthetic 图、64 个文本/图像 API 请求，全部 HTTP 200。图像到达时目标文本 lane 均仍活跃。
- `[观察]` 逐图联合 30/55 通过数：512 的 2-lane uniform/burst 4/4、4/4，6-lane uniform/burst 3/4、1/4；768 对应 2/4、0/4、2/4、0/4。每格 n=4，且跨 grid 非配对，不能推断因果或稳定 goodput。
- `[事实 / 路由]` 512 N=256，768 N=576；pixel proxy 命中 dense bypass / calibration 缺失，所有图像 keep-all。512 trace 为 16 个单图 encoder events `[256,8197]`；768 trace 16 张图落在15个 encoder events（一次双图 batch），完整输入 rows 无压缩。
- `[事实 / 归档]` 512 API/EngineCore writer 9,897 行、scheduler 1,041 行，全部 submitted=written、0 drop/error、clean close；服务正常 drain，无 force-kill/semaphore warning，GPU0/1回到782/45 MiB、8285释放。
- `[决定 / 待办]` 保留为固定 arrival harness 的八格预检；下一步跨 grid 使用同一 paired seeds，平衡顺序并延长窗口，补全长时文本与图像请求轨迹。完整八格数值与限制见 [分析报告](runs/processor-smoke/e-fixed-arrival-live-replay-analysis-v0.2.md)。
- `[事实 / 资源]` shutdown timeout 15s 后服务正常 drain，无 force kill/semaphore warning；writer clean close，GPU0/1 782/45 MiB，8285 已释放。日志、scheduler trace、writer metadata 与合成图片保存在 SSD 和本地。
