# E 内容风险预算与双期限联动设计 v0.2

日期：2026-11-09  
状态：pre-E 策略、精确请求预算映射与 live text/image controller 接线已实现；512/768 grid × 2/6 lane × uniform/burst 八格短 replay 通过调度和 request plumbing。未做当前 backbone 标定，线上仍 keep-all。

## 实现边界审查

固定 vLLM 0.33/Qwen3-VL 路径要求在 scheduler admission 前确定 prompt placeholders、multimodal spans 与 KV 长度。若等到真实 S_out DIG 在 E 后选 K，需要 worker 把每请求分数/索引回传 EngineCore，scheduler 再改 `Request.prompt_token_ids`、`mm_features.mm_position`、KV reservation，并把选择索引回发 worker 以更新 EncoderCache gather；这不是只在视觉模型输出处截短张量就能完成的改动。

E 当前已经支持 request-scoped `mm_processor_kwargs.images_kwargs.e_keep_ratio`。processor 在展开 placeholders 前读到该 ratio，选取整数 K；视觉编码仍按完整 image grid 执行，Qwen3-VL 的 E patch 在 final ViT block 后选择完整 merger group，并同步主视觉/DeepStack/M-RoPE 行。因此近期可验证路线是在 admission 前产生内容风险候选预算，利用已存在的每请求接口确定 K；真实 S_out DIG 留作质量校准和离线比较，不伪称在线闭环。

## 信号与预算守卫

- `query_simple_proxy`：现有透明词法特征反向得到的 simple 排名分；不是 ModernBERT QCR，也不是概率。
- `pixel_density_score`：对 raw RGB `[C,H,W]`、值域 `[0,1]` 输入计算通道空间方差均值；是 pre-E 像素代理，不是 SieveServe embedding DIG。它必须在与 Qwen 实际预处理一致的图像样本上校准。
- 缺失任一信号时 route 为 keep-all。像素代理高于阈值时 dense-bypass、K=N；否则 simple proxy 足够高可选 aggressive candidate，其他走 conservative candidate。
- 当前代码中 `0.70` simple 阈值、`0.03` pixel 方差阈值及 `0.222/0.333` keep ratio 均为分支测试/待标定值；只有提供明确 calibration ID 才能启用。`0.222/0.333` 来自 SieveServe pruning ratio 的反向 keep-ratio 工程候选，不代表对 Qwen3-VL-2B 有效。

## 双期限联合控制

对 512/768 单图，输入 full merged-token 数 N、质量校准得到的 `K_safe(x,q)`、内容 route 的 ratio cap、在途文本 lane 当前 max-gap slack 与图像 TTFT slack。构造只包含 `K_safe ≤ K ≤ content_cap` 的候选；根据独立校准的 K→文本影响/图像 TTFT 代价曲线，选取同时满足两类 slack 的最大 K。若没有候选可行，保留 `K_safe` 并报告两种预计超时，不通过降低质量下界制造虚假可行性。dense-bypass 固定 K=N；resolution cap 是单独动作，当前未实现。

文本 slack 由 client/load harness 对每条在途流记录上次 token 时间：`30 ms − (now − last_text_token_time)`，多 lane 取最紧者；图像 slack 由 `55 ms − (now − image_arrival_time)` 计算。正式 trace 需同时保留原始到达时刻、token 时间与策略作出的 slack 快照。

当前 E API 只接收 ratio。为避免浮点 `ceil(N × K/N)` 偶尔上舍入成 K+1，`request_ratio_for_exact_keep_count()` 对 K<N 返回 `(K−0.5)/N`，K=N 返回 1.0。远端固定 Python/Torch self-check 穷举 N=1..2048 的全部 K，验证 `ceil(N × ratio)=K`。

## 策略接口与实验臂

`scripts/e_content_budget_policy.py` 提供原始像素代理、pre-E 双守卫、质量下界内的双期限 keep-count 选择，以及生成 `{"images_kwargs":{"e_keep_ratio":...}}` 的单图请求 kwargs。当前配置的 `calibrated_for_current_backbone=false` 时有效 ratio 固定为 1.0。在线 client/admission harness 尚待实现；不要把原语自检算作 serving controller 验证。

新增 `scripts/e_live_joint_controller_e.py`：`TextDeadlineTracker` 保存每条在途 text stream 最近 SSE content 时间；`prepare_image_request_budget()` 取最紧 text gap slack 和 image arrival→TTFT slack，再调用风险预算 helper 生成 `mm_processor_kwargs`。没有 calibration ID、K_safe 或 cost curve 时直接返回 N/N、ratio=1；dense bypass 或风险信号缺失也不允许 deadline controller 改成剪枝。`smoke_stage_split_e.py` 已把多路文本 callback 接入短混合 workload；`replay_live_joint_controller_e.py` 按预定到达时刻发图并将客户端 worker queue 纳入 TTFT。512/768 grid 的八格短 replay 全部跑通，但仍未做正式 online matrix。

远端 controller self-check 覆盖双 lane tightest slack、keep-all fail-safe、合成双期限选 K、dense veto、超时保留 `K_safe` 与完成 lane 清理。随后单图 smoke 在 Qwen3-VL processor 得到 512 square `[1,32,32]`、N=256；raw RGB pixel variance=0.0668625，candidate route 为 `dense_pixel_bypass`，未校准有效 ratio=1.0；HTTP 200、prompt/completion=277/41、finish=stop。request-linked visual event 的输出形状 `[256,8197]`，ViT pixel 输入形状 `[1024,1536]`。这仅验证 request kwargs 在 placeholders/encoder/trace 中一致，生成文本出现 chart hallucination，明确不计质量正确率。

- **B-fixed**：使用同一固定 residual ranking，选一个固定 K；按独立 calibration split 与 B-content 平均保留量匹配。
- **B-content**：使用同一 ranking，以 pre-E proxy route 选内容候选 cap；`K_safe` 仍不可被越过。
- **B-joint**：在 B-content 的安全候选范围内，以文本 gap 与图像 TTFT slack 决定 K，并在 D 已验证的 N/S 放置中选可执行项。

三个策略对照都要使用同一请求 API、同一 scheduler/trace instrumentation 和相同图像/arrival seed。研究结论将写成 `SieveServe-inspired pre-E proxy policy + deadline joint controller`，不能写成 SieveServe QCR/DIG 的原样复现。

## 校准顺序与限制

1. 闭合 POPE/GQA/TextVQA/DocVQA/ChartQA 各 120 个项目的 annotation 与逐图媒体权利；当前 eligible=0，不下载 pending 图像、不评分。
2. 对 rights-cleared 校准集保存 raw image、实际 processor 输出、full-token 与候选 K 回答；同时离线提取真实 S_out DIG 与 raw-pixel proxy，检查 proxy 分布、排序相关性和任务质量退化是否可用。按唯一图像切分 calibration/heldout。
3. 用官方 scorer 冻结按 query/task 风险分层的 `K_safe`。若 proxy 无法区分高风险图像，则调整/替换预处理 estimator；不能因调度 slack 放宽质量底线。
4. GPU0 上测 K/grid/lane/placement 的 E/P、text max-gap、image TTFT、CPU dispatch/H2D/KV 成本与预测误差；再实现流式 client controller 并对真实 API 做单请求和混合负载重放。
5. 冻结配置、阈值、cost estimator、seed 和请求 trace hash 后运行 8 cells × 5 arms × 12 paired seeds = 480 runs；结果以 seed 为系统配对单位、以图像为质量配对单位。

当前限制：像素统计不是语义内容风险；词法 proxy 不是 QCR；GPU E 仍执行完整视觉 grid；固定 scorer 的 merger-group 映射仍属显式工程适配；6 类数据权利、K_safe、成本曲线、在线 controller 与 480-run 结果均未完成。

## live text/image mixed smoke 的观测与边界

两路与六路文本 SSE callback 均与一张 768×768 图像请求同时运行成功。图像到达时记录到 2/6 条在途 lane；tracker 的最近片段 slack 约 29.8 ms，image TTFT slack 约 55.0 ms。内容 route 因 raw-pixel score=0.06047 进入 dense bypass，且未配置 calibration ID，故 K=576、ratio=1.0。

S-mode scheduler trace 在同一调度行安排 image encoder input 和全部文本 lane 各 1 个 decode token；该步不含 image LM prefill token。API-linked visual event 保留 576 行、输入完整 2304 ViT patch rows，说明这次只打通了调度切步与 keep-all request metadata。短 smoke 中 client SSE 内容片段最大间隔分别为 2 lane 的 51.524/51.523 ms 与 6 lane 的 40.893–40.960 ms，超过 30 ms 参考目标；“最新片段 slack”是决策瞬间的局部信息，不意味着历史 max gap 已达标。

这些 synthetic smoke 不提供压缩、质量、延迟或 goodput 结论。长时 replay 应固定 prompt 和 arrival trace，并从原始 SSE 时间戳报告每路 gap 分布与 image TTFT；启用预算之前仍须完成数据权利、当前 backbone 内容代理/真实 `S_out` 对比、`K_safe` 和 K→cost 校准。完整 trace 和退出记录见 `runs/processor-smoke/e-live-joint-controller-mixed-smoke-v0.1.md`。

## 固定到达 replay 初测

`scripts/replay_live_joint_controller_e.py` 已将 scheduled image arrival timestamp 传入 client stream recorder，故 image TTFT 包含客户端 worker queue。2/6 text lane × 300 ms uniform/25 ms burst 的四格短 replay 全部 HTTP 200，每格 4 个图像 arrival，目标 text lanes 均保持活跃。

2-lane uniform/burst 每路 255 个 SSE content-fragment gap 中分别有 1/2 个超过 30 ms；6-lane uniform 为 0 个、6-lane burst 为 2 个。image TTFT p50 分别为 uniform 55.7/54.4 ms、burst 100.6/130.0 ms。测试执行顺序固定、每格图像样本 n=4，因此只能说明当前 harness 捕捉到了到达模式差异；不能作因果结论或作为稳定分位估计。768 图仍 K=576、ratio=1.0，不存在预算剪枝。

随后 512 grid（processor `[1,32,32]`，N=256）四格按 shuffle seed `20261109` 的顺序 6-uniform、6-burst、2-uniform、2-burst 执行；512/768 八格共 32 张图，每格 n=4，全部 HTTP 200。按文本 30 ms gap / image 55 ms TTFT 的逐图重叠窗口计算，512 的联合通过分别为 2-lane uniform/burst 4/4、4/4，6-lane uniform/burst 3/4、1/4；768 对应 2/4、0/4、2/4、0/4。768 四格固定顺序，网格间非配对；这些短数据不支持稳定 goodput、因果或跨-grid结论。所有图像保持完整视觉预算。

下一步用跨 grid 共用的 paired seeds 平衡/反序条件并延长窗口，保留每 lane 全量 SSE 时间戳、逐图 TTFT 与 scheduler/E event join。正式质量集、current-backbone `K_safe` 与成本标定未完成前，预算策略继续 keep-all。八格分析见 `runs/processor-smoke/e-fixed-arrival-live-replay-analysis-v0.2.md`，768 初次四格见 `...-v0.1.md`。
