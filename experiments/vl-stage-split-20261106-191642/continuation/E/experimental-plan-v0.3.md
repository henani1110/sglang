# E 阶段实验方案 v0.3（实现路径修订草案）

更新时间：2026-11-09。v0.2 保留为历史版本；本版把预算执行器从尚未落地的 GeoPrune 后编码器 mask，改为 vLLM/Qwen3-VL 原生的逐请求图像分辨率预算。两者并非等价方法，结果中分别命名。

## 1. 研究问题

在单张 GPU 上图文与文本持续共服务时，能否用内容风险估计建立每个图像请求的可用视觉 token 区间，再用文本 decode slack 和图像 TTFT slack 在区间内调节图像分辨率预算，并结合已验证的 N/S 执行策略，提高双 SLO goodput 且报告完整视觉任务质量？

该问题不是“首次单卡混跑”或“首次内容感知 pruning”。DuetInfer、UnifiedServe、SieveServe、ET-Prune 等都是强近邻。研究结论只描述这套 processor-level 预算与 deadline controller 的实测组合效果。

## 2. 实现路径

- 每个 Chat Completions 图像请求使用 vLLM 0.33 原生 mm_processor_kwargs，按请求传入 images_kwargs.size.longest_edge。保留 processor 默认 shortest_edge。
- 这个动作限制源图 resize 后的总像素数，Qwen smart_resize/patch merge 随后产生真实 image_grid_thw、placeholder 数及 decoder 输入长度。控制器将“目标 token 档位”映射到 max-pixel cap；每个请求同时记录 cap、image_grid_thw、实际 placeholder 数和 scheduler/KV token 数。
- square 768×768 合成图的 CPU 预检观察到默认 576、262144 pixel cap 为 256、65536 pixel cap 为 64 个 merged image tokens。512/768 在线负载、自然长宽比图片及所有目标版本都必须实测实际数量；不能把 pixel cap 当成精确 token 数。
- 分辨率缩放发生在视觉编码器之前，可能同时降低 E 和 P，但会丢弃图像细节。它不同于 GeoPrune post-ViT ranking。v0.3 不以“GeoPrune 复现”描述此策略；GeoPrune 和 PixelPrune 保留为独立外部工作/后续 comparator。
- 通过真实 serving 验证 per-request kwargs、image placeholder 数、输入 token 数、KV 计数和完整响应后，再将该 actuator 用于在线矩阵。检查原生不传 kwargs 与显式传默认 size 的 token 和完整输出一致性。

## 3. 内容风险与预算控制器

- 质量校准使用官方任务评分在 full/native、medium、low 三档分辨率下形成逐样本分数/答案变化标签；sample/question cluster 是统计单位。
- 为 runtime risk estimator 记录 query complexity 特征与图像视觉密度特征；候选模型优先用轻量、可审计的 CPU 逻辑回归/规则模型，不在模型答案或留出结果上调阈值。模型拟合、特征处理、阈值、风险分箱及校准样本清单全部版本化并哈希。
- 运行时给每张图片预测“低预算可能伤害任务质量”的风险分数。低风险请求允许更宽的预算区间；高风险请求保留校准支持的较高下界。风险输出、估计耗时、CPU 特征提取开销与选定 budget 均逐请求记录。风险估计错误和质量回退要报告，不能删除。
- deadline controller 按 D 冻结 A 的规则，在每个质量预算允许的分辨率档中估计 E/P 代价，再用 D_slack/I_slack 选择 budget 档和 N/S。紧急期限不能绕过高风险预算下界。
- 若风险特征或模型本身无法实现，先记录失败，继续完成其他工作并修复；可运行版本必须在最终报告中明确实际采用的算法及其边界。

## 4. 对照臂与矩阵

保留 5 个臂，与 v0.2 比较问题一致，但 B 臂使用 processor-level resolution budget：

1. N：Qwen3-VL 默认输入分辨率，原生 serving。
2. A：默认分辨率下复用 D 冻结的 deadline-only N/S 决策。
3. B-fixed：固定 pixel cap；通过独立校准样本将其实际平均 merged token 数匹配 B-content。
4. B-content：内容风险只决定分辨率/token 预算，不使用实时 slack；策略称为 risk-aware resolution budget，注明与 SieveServe 的方法差异。
5. B-joint：内容风险给出允许区间，实时 D/I slack 决定区间内分辨率档及 N/S 放置。

8 个负载 cell：grid {512,768} × 活跃文本 lane {2,6}（各目标 192 output tokens）× 图像 open-loop 到达 {uniform,burst}。每臂 12 个配对 trace seeds，5×8×12=480 个 run；每次 60 秒 warmup + 180 秒 measure，共约 32 GPU 小时名义测量量。该数值用于排程，不作任务停止条件。每 seed/臂用同图片、问题、文本请求、计划发送时刻；按 block 平衡臂顺序，下一 block 反序。

主要指标沿用 D 的文本最大 client token gap 30 ms、图文 client TTFT 55 ms 联合 goodput；敏感性阈值为 {25,45}、{40,75} ms。同步报告 per-task official quality、high-risk slice、token/grid 分布、E/P CUDA event、queue/dispatch/API/SSE/client 时间、KV、显存、文本与图文完成/超时/取消率、控制器开销。

## 5. 样本和数据权利

目标质量集仍为 POPE/GQA/TextVQA/DocVQA/ChartQA 各 120，共 600；保留每 item 的标注来源、媒体 source、许可依据、使用限制和 source SHA-256。问答标注、软件许可、底层媒体和下载条款分开登记。不得仅凭 benchmark repo 的代码 license 推断其图像可自由使用。无法核实权利的图片不得进入冻结评测清单，但该项未闭合不阻止 synthetic-path、source audit、serving、计时、队列 trace 与独立的已授权样本工作继续推进。

质量集和在线图文 trace 分开。质量侧保存完整答案和官方 scorer；在线测量 output length 固定以维持与 D 对比，不能把固定短输出当成答案质量实验。校准和 heldout 按 item/source cluster 分组，避免同图多问泄漏；所有选择按校准划分完成，heldout 只做最终报告。

## 6. 运行顺序

1. 完成真实 Chat Completions smoke：同一合成/授权图传默认、256 档、64 档，检查 API 接受参数、返回 token usage、processor grid/placeholder 和 KV/输入计数。
2. 加入真实 serving 事件采集。用 CUDA Events 分别记录视觉编码 E、LM prefill P；flush 后读事件，不在每步同步。另记 CPU 处理/风险估计、H2D、队列、API/SSE/client。
3. 在现有授权图片上生成 native/medium/low 完整答案，检查官方评分、风险特征与预算档；逐项记录原始输出。运行失败保留证据并修复/补跑。
4. 构建 open-loop 配对 traces 和预算/估计器 profile，冻结源码、配置、sample manifest、prompt、seeds、策略、analysis plan 和 SHA。
5. 运行 480 个配对 cells/arms，失败 seed 诊断修复后补跑，保留原始 attempts。随后完成统计、质量分析、归档和资源清理。

## 7. 运行和归档约束

- 仅 chx-gpu-4090-2 GPU0 RTX 4090，负载期间 PCIe Gen4 ×16；GPU1 RTX 4090 Gen4 ×4 不参与。
- 权重只读复用 SSD；所有 E 数据/缓存/trace/log/run/分析/archive 写入 /media/chx/Win10/vl-stage-split-20261106-191642/continuation/E/，同步本地镜像并逐文件校验 SHA-256。
- A/B/C/D 只读。若实现、样本、策略或 schedule 需要修订，创建新版本并说明原因，不覆盖前一版本。
- 没有研究效果、CI、资源估算、失败数量或会话耗时的任务级停止门槛。每个计划样本均要有成功或可追溯失败记录；当前不可用时保存进度并继续修复/续跑。Goal 只有在所有计划评测、分析、归档、释放资源和日志更新真实完成后才结束。

## 8. 实现预检增补（2026-11-09；非正式实验结果）

- CPU processor 完整调用：同一合成 768×768 图在 native、262144、65536 pixel caps 下，merged visual placeholder 数分别为 576、256、64。
- 端到端 OpenAI Chat Completions：三个版本全部 HTTP 200；prompt_tokens 分别为 595、275、83，固定非图像模板贡献 19 token，因此与 processor grid 展开计数逐项一致。
- 服务默认关闭 mm_processor_kwargs，带预算的请求返回 HTTP 400，错误提示要求显式信任 request kwargs。已在 loopback-only 的 smoke 服务中启用 --trust-request-mm-kwargs 后成功；正式服务仍应只向可信 client 暴露。
- 低预算图像得到不同的自然语言描述并误读一个 synthetic 标题。这是定性机制样本，不是 benchmark 得分或质量显著性结论。
- 首次服务启动缺少 Python.h include path；按固定 D runner 提供的 python-runtime include path 设置 CPATH/C_INCLUDE_PATH 后启动成功。原失败日志保留。
- 本次没有 per-request scheduler/KV trace 或 CUDA Events，故端到端 prompt length 一致不等于已实测 KV bytes 或 GPU E/P。下一次接入 E-owned probe overlay 后逐项测量。

## 9. 实际 V2 E/P tracing smoke（2026-11-09；仍非正式 run）

- 当前 vLLM 配置选择 V2 `vllm.v1.worker.gpu.model_runner.GPUModelRunner`。早期 V1 `_model_forward` hook 未覆盖线上执行路径，保留为 exploratory failure；正式 E smoke 现改为包围 V2 的 FULL/PIECEWISE replay、ubatch 和 eager launch。
- `e-trace-v2-4` 对 native/256/64 merged-image-token 档分别看到 processor grid 对应的 595/275/83 total prompt tokens；V2 记录 3 个对应 prefill batch 与 90 个图像请求 FULL replay。纯文本 17-token prefill 和 15 个 replay 也成功记录。
- 真实 Chat 与 text completion stream 的 engine/API token 序列均逐 token 一致。worker 与 API 的 async writer 按 PID 分开汇报；`collective_rpc` flush 之前会 barrier flush worker event queue，观察到 97/97 和 114/114 worker rows、216/216 server rows，均无 drop/write error。
- 启动 profiling 的 `mm_hash=null` E event 与内部 `_warmup_` P request 必须排除；V2 patcher 已不记录 `_warmup_` 的 P event。保留完整原始 flush 结果用于复核。
- 同一合成负载的 E/P CUDA-event 在多次 smoke 间有明显变化，当前只证明测量链可用。不能作为低分辨率改善 E/P 或 goodput 的证据；正式矩阵需固定时钟/负载记录并按 seed 报分布。
- 逐请求物理 KV bytes、CPU dispatch、H2D、queue decomposition、visual E 与 API request ID 的稳定 join、显式默认 kwargs 等价、多图/aspect-ratio 边界和 instrumentation overhead 仍待验证。数据 rights audit 和 600-item quality set 也未冻结；480 个正式在线 run 尚未启动。

## 10. V2 visual-event request-ID join 预检（2026-11-09；非正式 run）

- E-only runner 在调用 `model_state.prepare_inputs_embeds` 前，从 `scheduled_encoder_inputs` 和 `encoder_cache.mm_features` 建立 `mm_hash → request IDs` 上下文；hook 将 `request_ids`、`mm_hashes` 和 CUDA elapsed time 一起返回。一个聚合 encoder batch 允许映射多个 scheduler request。
- `e-trace-v2-5` 的三张图均 HTTP 200。离线连接审计将三条真实视觉事件逐条关联到 scheduler encoder-input row、OpenAI `response_id`、engine result row 与 SSE yield row，3/3 通过。请求与响应 ID 的关联要求 scheduler request ID 以唯一 API response ID 加内部 suffix 开始，并须同时在 server trace 中出现。
- 视觉 CUDA-event 单次样本为 19.270/14.130/13.853 ms；仅作 instrumentation smoke 记录，不作为不同 pixel cap 的效果结论。完整连接表在 `runs/processor-smoke/e-trace-v2-5-event-join.json`。
- 仅串行单图通过，multi-image、同 batch 多 request 的关联仍需验证。KV block/bytes、CPU dispatch、H2D、queue decomposition 和 trace overhead 尚未观测；`formal_runs_started=0`，数据许可和 600-item quality set 仍未冻结。
