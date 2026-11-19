# E 实验方案 v0.9：N 基线负载校准与后续工作顺序

日期：2026-11-09  
状态：N keep-all synthetic load calibration 完成；质量集权利、有效安全预算、最终策略配置和 480-run formal matrix 未完成。Goal 保持 active。

## 1. 研究问题与预注册主框架

核心问题仍为：在单张 GPU 上同时服务持续文本 decode 与开放环图像到达时，能否以内容风险定义视觉预算安全下界，再结合实时文本 decode slack 与图像 TTFT slack，在安全范围内联合决定视觉 token 数与 E/P 放置，从而在保留稠密图像任务答案质量的同时提高双期限 goodput？

预注册矩阵保持不变：512/768 processor grid × 2/6 条活跃文本 lane × uniform/burst 到达，共 8 格；策略为 N、D 阶段 deadline-only A、GeoPrune-style fixed budget、SieveServe-inspired content-only budget、content + joint deadline budget；每格每臂 12 个配对 seed，共 480 个正式 run。正式 run 仍为 60 秒 warmup + 180 秒 measurement，主期限 30 ms text max-gap / 55 ms image TTFT，并报告 25/45 与 40/75 ms 敏感性。N v0.1/v0.2 都不计入该矩阵。

文献与实现边界见 [`literature-and-novelty-audit.md`](literature-and-novelty-audit.md)、[`implementation-boundary-review-v0.1.md`](implementation-boundary-review-v0.1.md) 和 [`content-risk-budget-design-v0.2.md`](content-risk-budget-design-v0.2.md)。不得把一般图像 token pruning、内容感知预算、单 GPU 混跑或 deadline-only N/S 调度本身写成新颖点；目标差异是安全预算下界与双实时 slack 的联合控制，仍是需要实验验证的假设。

## 2. 本轮 N 基线工程校准结果

| 校准块 | 设计与覆盖 | 观察 | 结论边界 |
|---|---|---|---|
| v0.1 | 8 格 × 12 seed = 96 replay；12 图/replay；uniform 750 ms、burst 25 ms | 所有 1,536 个测量 API 请求 HTTP 200，但 uniform 后段图像到达发生在文本输出完毕后，文本 lane count 归零 | 全部保留为诊断记录；不能认为 uniform cell 验证了持续 mixed workload |
| v0.2 | 8 格 × 12 seed = 96 replay；12 图/replay；uniform 200 ms、burst 25 ms | 97/97 输出 JSON 完整，96/96 seed-cell 全覆盖；1,152/1,152 测量图像到达均观察到目标 2/6 条活跃文本 lane；1,536 个测量请求 HTTP 200。1156 张 PNG 与请求 JSON 的 image SHA-256 全部匹配，跨 lane/arrival 条件的同 seed 同 grid 图像一致 | 证实 workload harness 可以维持计划内的 text/image overlap；仍是 synthetic、keep-all、instrumented N 工程校准，不验证质量、压缩收益或正式 SLO goodput |

v0.2 各格每 run 的图像 TTFT P95 中位数约 33.9–537.4 ms，某些 6-lane burst 出现更高 tail；文本 SSE 最大间隔也有超过 30 ms 的值。具体统计口径与每格数值见 [`v0.1 analysis`](runs/n-baseline-calibration-v0.1/20261109T092927Z/analysis-notes.md)、[`v0.2 analysis`](runs/n-baseline-calibration-v0.2/20261109T095608Z/analysis-notes.md)、[`v0.2 integrity audit`](runs/n-baseline-calibration-v0.2/20261109T095608Z/integrity-audit-v0.2.json)。这些仅用于构造 workload 和排查计量，不能当作策略效果或单请求独立样本推断。

### 运行环境与资源记录

- 固定服务器：`chx-gpu-4090-2`，Qwen3-VL-2B-Instruct、固定 vLLM 0.33.0 环境；只使用 GPU0 RTX 4090。负载时 GPU0 为 PCIe Gen4 ×16；GPU1 RTX 4090 为 Gen4 ×4，不参与实验。
- 本轮完成时 GPU0/GPU1 显存为 782/45 MiB，端口 8285 已释放。所有新数据、模型权重/缓存和输出继续放 SSD `/media/chx/Win10`，本地保存复核副本。
- 这两块服务启用了 E instrumentation，图片为脚本生成的 synthetic charts，视觉预算保持 keep-all，图像回答 16 tokens、文本回答上限 512 tokens。它们不替代 32-run/per-arm formal setting。
- 本次用户请求的 GPT-6 Luna / Max 切换未执行：Mac 当前无法解锁，模型选择器不可操作；任务按当前线程模型继续，没有因此中止运行。

## 3. 数据权利状态

质量集目标仍是 POPE、GQA、TextVQA、DocVQA、ChartQA 各 120 项；当前 `rights_manifest.csv` 仍为空，eligible 为 0/600，图片下载数为 0，质量 runs 为 0。各 dataset 的 annotation/license 与底层 image/document rights 必须分开核验。

- **TextVQA / Open Images：** Open Images V3 的官方 metadata 可记录原始 landing URL、license、author、title 与 MD5；V3 README 同时明确不担保每张图片的 license，要求逐图核验。因此 120 条 Open Images CC BY 2.0 metadata 候选仍是 pending。官方来源：[Open Images V3 README](https://github.com/openimages/dataset/blob/main/READMEV3.md)。
- **Flickr 页面处理：** 已停止直接自动化读取 Flickr landing pages。此前 120 条页面 GET 的 102/200、17/404、1/403，以及 100 个 license 文本匹配和 2 个不匹配，只作为过程审计保存，不作为清权或 eligibility 证据，也未下载图片。当前 Flickr Terms 禁止未经授权的 scraping/data mining/robots 数据提取；使用 API 也需要 application key 并遵守另一组 API Terms，照片 owner 对具体照片的要求仍适用。官方来源：[Flickr Terms](https://www.flickr.com/help/terms/)、[Flickr API Terms](https://www.flickr.com/help/terms/api)、[API keys](https://www.flickr.com/services/api/misc.api_keys.html)。不得再次执行页面自动请求。
- **POPE/GQA：** 仓库或 annotation license 不等于 COCO/Flickr 图像许可；分别保存 annotation grant 证据和逐图媒体来源、许可与 attribution。
- **DocVQA：** 官方说明图像来自 UCSF Industry Documents Library，并要求在 RRC challenge 下载页登录后阅读条款；下载页条款仍需实际复核，逐份文件来源范围也不能由数据集代码 license 推定。[DocVQA 官方页](https://site.docvqa.org/datasets/docvqa)。
- **ChartQA：** 作者仓库 GPL-3.0 不自动覆盖第三方图表原图；保留 source organization/page 与逐图许可证据，无法确认的候选不进入 cleared manifest。

Flickr/API 或 RRC 账户所需材料未就绪时，继续推进 rights-independent 的实现和合成工具校验，但不能把合成数据或未清权候选伪装成官方 benchmark 子集。必要时将替代来源作为单独 supplementary set，不混入上述 600 项正式质量集。

## 4. 接下来具体执行顺序

1. **清理矩阵的原始证据与同步。** 本地保留 v0.1/v0.2 每次 replay JSON、stdout、seeded PNG、server log、config/index/completion、分析和审计。同步 README、方案、分析脚本与 rights audit 到 SSD；建立新版本归档 manifest，逐文件校验本地与 SSD SHA-256。A/B/C/D 只读。
2. **继续完成可直接推进的授权核查。** 为 POPE/GQA/TextVQA/DocVQA/ChartQA 分别填 annotation 与 media/document scope、source URL、权限证据、attribution、hash。Flickr 不再网页抓取；如有合规 API/权利人许可路径，按其条款逐项取得元数据并再次核实所有者要求。DocVQA 使用已登录的 RRC 正式页面条款；ChartQA 逐图追溯来源。记录每次排除与替代；pending 样本不下载、不跑质量评分。
3. **权利无关的剪枝实现审计。** 依据当前 vLLM 0.33/Qwen3-VL 路径，实现 GeoPrune-style residual ranking 的显式 pre-merger → decoder merger-group 映射；明确论文 eigengap 算法排版歧义，不声称忠实复现。任何真实压缩都需同步 base/DeepStack 特征、placeholders、视觉 mask、M-RoPE rows、prompt span、scheduler 长度和 KV reservation。先在已验证接口上检查 keep-all 等价与完整响应，再做 fresh-request pruning；失败尝试完整保留。GeoPrune 主要可能节省 decoder prefill/KV，不能预设它会省掉已执行的 ViT encoder E。
4. **取得可评分的 rights-cleared calibration/heldout 集。** 每数据集各 120 条，记录 seed、官方 prompt/split、image/document SHA、license/attribution、预处理和 scorer 版本。将 calibration 与 heldout 按唯一图像切分；同图多问题不可跨分区。若正式 600-item rights 集仍在核查，可并行建立自己生成的合成 calibration fixtures 用于实现测试，但不能用其冻结正式质量阈值或替代 benchmark 结论。
5. **冻结质量下界与成本曲线。** 先分别测 full-token 与 GeoPrune-style fixed-K 配对答案质量，使用官方 scoring；按鲁棒与稠密信息类报告配对质量损失与区间。结合真实 `S_out` 特征、pre-E proxy、query 类型与 grid，验证 risk proxy 能否找出高风险样本并确定 `K_safe`。再测 K/grid/lane/placement 的 GPU E/P CUDA-event、CPU dispatch、H2D、队列、KV、文本 gap 和 image TTFT，冻结独立校准出的 K→cost 曲线与估计误差。
6. **实现并核查五臂控制。** N、D deadline-only A、GeoPrune fixed K、SieveServe-inspired content-only、content + joint slack。固定预算按独立内容校准集的平均保留 token 匹配；B-joint 不得突破 `K_safe`。记录实际 K、deadline slack、放置动作、各阶段时延、请求结果与策略失败。
7. **冻结正式矩阵并执行 480 runs。** 按 8 cells × 5 arms × 12 paired seeds 执行；同 seed 的 request、PNG、文本计划与到达时刻完全匹配；平衡/反序块顺序。N v0.2 的 200 ms uniform 到达间隔作为保持 lane overlap 的校准候选，但在正式配置冻结前还须按 60 s warmup/180 s measurement 长窗口复核到达/输出生命周期，并将配置 hash 固定。不能把本轮短回放复制成正式结果。
8. **完成配对统计、可复现包和资源清理。** 以 seed 为在线系统推断配对单位、以图像为质量配对单位；报告主 SLO、两组敏感性、TTFT/max-gap 分位、质量与错误/超时、最长饥饿、token、E/P/H2D/KV/排队开销。生成中文报告、SVG、run manifest、配置/seed/补丁/失败记录、资源账本和 SHA-256 清单；SSD 与本地两份逐文件核验，抽取副本复核，结束后确认端口释放、GPU0 回到桌面基线、GPU1 不受影响、SSD 空间可用。

不设置效果、准确率、SLO、CI、证据覆盖率、GPU/墙钟、会话或普通故障次数门槛。阴性结果、一般可修复问题或候选集未满时保存证据并继续其余工作；所有结果如实写入，不通过改变指标或删去失败样本制造正向结论。正式目标仍为完成整个 E 阶段，而不是完成 N 校准块。

## 5. 当前计数

| 项目 | 当前 | 目标/状态 |
|---|---:|---|
| N v0.1/v0.2 测量 replay | 192 | engineering calibration；不计 formal |
| v0.2 图像到达时目标文本 lane 覆盖 | 1,152/1,152 | 已通过负载 harness 校验 |
| 已清权质量样本 | 0 | 600，五个 benchmark 各 120 |
| 正式质量评分 | 0 | 600 项 full-token / candidate 配对 |
| 正式在线策略 run | 0 | 480 |
| 已退出的本轮服务 / 端口 | 是 / 8285 空闲 | 资源回到基线 |

