# E 阶段实验方案 v0.2（预注册前草案）

**更新时间：2026-11-09。当前不是冻结预注册。** 本版在旧计划上增加直接近邻、token 路径和数据权利审计；任何参数或策略代码将在校准材料独立后单独写入并哈希冻结。

## 1. 问题与可证伪假设

在 GPU0 单卡，持续文本 decode 与开放环图像请求同一时段共服务时，以文本 decode slack、图像 TTFT slack 控制调度动作；以图像/问题风险确定不能再压缩的视觉 token 安全预算。检验内容预算与双期限联合控制是否在稠密信息图像质量约束内，提高联合 SLO goodput，并解释改善来自 E、P、队列/KV 中的哪一段。

组合差异假设：`B-joint` 在某些负载状态相对 N、D 的 deadline-only A/S、固定预算和 content-only B 能增加双 SLO 通过的图文请求，而质量退化不集中到高风险子集。DuetInfer、UnifiedServe、SieveServe、ET-Prune 已覆盖相邻机制；结果只支持或反驳这个组合状态空间，不做绝对首创声明。

## 2. 环境和隔离

- 单卡：`chx-gpu-4090-2` GPU0 RTX 4090，负载 PCIe Gen4 ×16。GPU1 RTX 4090 Gen4 ×4 不参与。
- 模型：Qwen3-VL-2B-Instruct，revision `89644892e4d85e24eaac8bacfd4f463576704203`；vLLM 0.33.0，复用 D 固定环境和权重。
- 所有 E 数据/缓存/trace/log/result/archive 存 SSD `/media/chx/Win10/vl-stage-split-20261106-191642/continuation/E/`，并保留本地可复核副本。A/B/C/D 只读。
- 记录源码/补丁 commit/hash、driver/CUDA/torch/backend/实际 PCIe link、模型 revision、sampling、GPU power/clock 与服务启动配置。

## 3. 样本与评分

- 每数据集 120 个 image-question-answer item：POPE、GQA、TextVQA、DocVQA、ChartQA，总数 600。数据权利两层 manifest，只有来源与本地研究使用依据已审计的 item 可进入冻结清单；每 item 保存 SHA-256、问题/答案、源 ID、媒体来源、许可与评分代码版本。
- 质量集与开放环在线 trace 完全分离。按固定 seed 在每任务内建立校准/留出划分，并只用校准样本选风险阈值、预算下界与固定臂保留量；留出质量集只做最终报告。最终拆分 seed 和比例待工程评估后冻结。
- 在 processor grid 512/768 做 full-token 与候选预算的配对质量评估，保留 dataset 官方 scoring protocol（POPE 官方分类指标、GQA accuracy、TextVQA official VQA accuracy/ANLS 对照、DocVQA official ANLS、ChartQA official relaxed accuracy）。记录 exact prompts、模板、decode 参数及完整输出。
- 候选预算应形成多级质量-视觉 token 保留曲线；在线固定预算臂匹配 content 臂在独立校准样本上的平均真实 keep count，而非事后匹配结果。

## 4. 实现及预检序列

依次完成：完整原生路径/输出等价；token 选择索引可审计；Qwen3 `DeepStack + placeholders + M-RoPE` 全部一致压紧；scheduler/KV 反映剪后长度；分段 CUDA-event E/P 与队列/API/client 计时；官方评分。GeoPrune 是首选 fixed backend，SieveServe-inspired guard 与相同 backend组成 content 策略。若正式 API/版本不兼容，保留失败证据，定位并适配；PixelPrune 作为不同插入点的 pre-ViT 候选单独标记，不能混为 GeoPrune 等价复现。

## 5. 在线计划矩阵

- 8 cells：grid `{512,768}` × 活跃文本 lane `{2,6}`（每条目标 192-token）× 图像开放环到达 `{uniform, burst}`。
- 核心 5 臂、每臂 12 个配对 trace seeds，计划 480 个正式 run：
  - `N` 原生 vLLM。
  - `A` D 冻结的 slack-based N/S 期限控制，压缩关闭；它是同运行时 deadline-only baseline，不等同于 QoServe 或 DuetInfer。
  - `B-fixed` GeoPrune 固定 keep count；在独立校准样本上与 B-content 平均实际 keep count 配平。
  - `B-content` 同 GeoPrune selector，按 query complexity/image dense-risk 选择预算；若非逐项重现原论文，统一标为 `SieveServe-inspired` 并写出差异。
  - `B-joint` 在 B-content 的质量安全预算下界内，再按实时 `D_slack/I_slack` 调整预算与 D 已测 N/S 放置；绝不以紧迫期限为由压穿高风险质量 floor。
- 同 seed 各臂使用相同图像、问题、计划发送时间和文本请求；open-loop 发压并记录计划/实际发送抖动。按 block 平衡 run 顺序，下一 block 反序。每 run 60 秒 warmup + 180 秒测量；在线图文输出长度保持与 D 可比较的固定值，质量结论只取独立完整输出的 600 样本离线评测。
- DuetInfer / UnifiedServe 由于论文公开的 runtime 和本地 vLLM 代际不同，先尝试找到作者版本和精确依赖。若无法得到原始代码，报告算法/架构差异及“未直接复现”；不把 A 等同于这些系统。若可审计地加入 same-runtime fairness/aging 或 MPS 对照，则增列独立 arm，并更新新版本的 schedule 与总 run 数，不覆盖 v0.2。

## 6. 指标与统计

- 主指标沿用 D：文本最大 client token gap 30 ms 与图文 client TTFT 55 ms 同时满足的 joint goodput；敏感性 `{25,45}` 与 `{40,75}` ms。
- 同时报 text gap 和 image TTFT 的 P50/P95/P99/max、请求级违约率、完成/错误/超时/取消、图文队列等待、最长饥饿、视觉质量、实际 keep tokens、KV/显存、E/P GPU-event、CPU dispatch/H2D/API/SSE 分段时延、控制器开销。
- 统计单位：配对 trace seed；报告同 seed 差值和 bootstrap 区间。质量指标以配对 image-question 样本为簇，报告鲁棒/稠密信息分组；不把请求/token 当独立重复。
- 留出种子不调阈值或改 prompt；每次策略修复留存原始故障，并从缺失 seed 续跑。

## 7. 实验量和完整交付

核心矩阵 nominal 为 480 个 run × 240 秒 = 32 GPU 小时，不是上限或停止门槛。计划运行结束后完成全部配对分析、诊断/补跑、逐文件 SHA-256、SSD/本地双副本、抽取归档复核、服务与端口退出、GPU0/GPU1/SSD 资源检查，再更新 D/E 可行性报告、实验方案和主日志。

## 8. 继续的前置动作

1. 完成论文/作者代码/硬件与版本差异表，更新相关工作。
2. 继续逐数据集/逐媒体 item 核验 600 样本许可。DocVQA RRC terms 与 ChartQA/COCO 上游图片权利是当前明确未闭合项。
3. 定位 vLLM 0.33 input placeholder、positions/M-RoPE 和 scheduler KV accounting 的 exact hook，完成完整正确性校验，不改动 D。
4. 冻结风险定义、预算动作空间、token 对齐、prompt、official scorer、校准/留出切分、arrivals/seeds 和 run manifest 后再启动第一批质量评估。
