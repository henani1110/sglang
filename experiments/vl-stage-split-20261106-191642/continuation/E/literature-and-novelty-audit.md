# 相关工作与新颖性审计（2026-11-09，初稿）

本表区分“论文覆盖”“作者代码可用性”和“本地兼容性”。本文不是穷尽性 novelty 证明；网页检索没有发现作者代码，不能写成作者从未公开代码。

| 工作 | 已覆盖的问题与方法 | 公开代码 / 硬件 /版本 | 对 E 的约束 |
|---|---|---|---|
| **SieveServe (EMNLP 2026)** | QCR 根据查询复杂度路由；DIG 根据视觉特征的通道方差检测高密度风险；对可压缩请求做激进/保守预算，对高风险请求绕过 token pruning 并限 tile。它已直接覆盖“内容/风险感知视觉预算”。 | 质量实验为单 A100 80GB；serving 为 4×A100、DP=4、TP=1。论文报告 vLLM 0.16 实现和 ModernBERT-base QCR；ACL 论文页/PDF 未提供仓库链接，本次没有确认可直接复用的作者代码。目标为 vLLM 0.33/Qwen3-VL-2B，算法和接口均需重验。 | E 不得把风险预算本身宣称为新贡献。差异候选是将其安全预算与已观测的在途 text decode slack、图像 TTFT slack 及 E/P 放置耦合；结果应标 `SieveServe-inspired`，并逐项标明与原文差别。 |
| **GeoPrune (Findings EMNLP 2026)** | 对 Qwen3-VL/Qwen3.5 的视觉编码器输出做主导方向 deflation，再用 residual L2 排序；论文集成进 vLLM。它已覆盖 Qwen3 系视觉 token 重要性和固定预算剪枝。 | 论文覆盖 Qwen3-VL-8B/3.5-8B 等；公开论文页未列作者仓库或 vLLM commit/version。本次未找到可核验作者实现。 | 是 E 的固定预算候选，不是可声称的新算法。需由同一保留索引压缩所有 DeepStack planes、image placeholders 和位置元数据；目标 2B / vLLM 0.33 兼容未验证。 |
| **DuetInfer (DATE 2026)** | 单 GPU 上文字与图文混流；并行 CPU preprocessing、独立 vision encoder/LM worker、共享权重/encoder cache、fairness-aware queue/aging。 | vLLM 0.8.2、CUDA 12.4、PyTorch 2.6；Qwen2-VL-2B/RTX 4090 用于消融与敏感性，端到端评测主要为 A800 上的 7B。论文使用 ShareGPT 与 COCO Captions 混合流量。会议论文未给代码 URL。 | **直接系统近邻。** “单卡混合图文”、VE/LM 解耦、避免 text 被视觉编码阻塞都已覆盖。E 需聚焦内容风险与双 deadline 联合决策，不能以 GPU 或模型版本差异单独主张新颖；将公平/aging 思路列为实现对照或明确复现限制。 |
| **UnifiedServe (arXiv:2512.17574)** | GPU 内部阶段逻辑解耦、MPS 资源共享；将 decode 独立 worker，encode 与 prefill 同进程编排，并做图像粒度 encode、token/chunk 预算及 TTFT/TBT SLO 评测。 | 本次核对到论文；未确认可复用仓库和 4090/Qwen3-VL 兼容性。 | “按 TBT 目标控制 encoder 工作量”和 GPU 内阶段共享已出现。对比须强调 E 的样本风险 floor 与双类余量在线联动，并检查原文策略可否作为强基线。 |
| **QoServe (ASPLOS 2026) / Niyama artifact** | Text-only LLM interactive 请求分别按 TTFT/TBT 管 deadline，按 decode slack 做动态 chunk 与优先级。 | 论文与 artifact 可查；论文实验为多 GPU A100，不是视觉服务。 | 通用 slack 控制不是新颖点。E 的 `A`/`B-joint` 至少要同运行时比较 deadline-only 策略；不得称为复现 QoServe，除非执行其实际策略/代码。 |
| **TriInfer / HydraInfer** | Encode–Prefill–Decode 级别解耦、异构实例映射、阶段 batching、SLO goodput；已讨论 E/P 对 decode 的干扰及阶段预算。 | MLSys 2026 正式版为多节点 H20 系统；早期 arXiv 名 HydraInfer。官方仓库的发布/版本状态待冻结时再查。 | 是分阶段和预算的强集群先例；不同于单卡资源约束，但“更小硬件”本身不足以构成贡献。 |
| **ET-Prune (arXiv 2608.01979)** | 对 Qwen3-VL-8B 等做 query-conditioned evidence、text-like region safeguard 和每样本 token floor；直接覆盖文本密集图像的风险预算。 | 作者 GitHub 当前 README 公开说明代码待发布；实现插入在 decoder 中间层，目标 vLLM 0.33 部署尚无可验证代码路径。 | E 的风险标签与预算不能只改名后复述该方法。其 decoder 中层成本与 E 的 pre-LLM serving/KV 节省要分层比较。 |
| **PixelPrune (作者 GitHub)** | Qwen3-VL/Qwen3.5 的 pre-ViT 2D predictive coding，可在进入视觉塔前减少冗余像素/视觉 patch。 | 仓库提供 vLLM plugin，示例含 Qwen3-VL-2B；测试版本是 vLLM 0.18.0/Transformers 4.57.6，而 E 是 0.33.0/5.20.2。 | 是可能节省 E 的近邻与备用实现。版本跨度较大，须单独验证/适配；其像素预测编码不等于 query-aware 质量风险预算。 |
| **Hi-Lo Prune (CVPR 2026)** | 分层 token selection、prune-aware fusion、早期 LLM pruning；论文评测包括 Qwen3-VL。 | 作者公开包当前只提供 Qwen3-VL-8B、单图 batch=1 的图像推理路径；原始依赖 Python 3.10/PyTorch 2.6/Transformers 特定 commit。 | 可作质量/选择算法对照，不能假设可直接改进 vLLM 0.33 生产 serving。 |
| **DART (NVIDIA TensorRT-Edge-LLM)** | 官方实现支持 Qwen2.5/3-VL 的 pre-LLM visual token pruning，并要求同步 compact DeepStack features 与 M-RoPE rows；fresh KV 有约束。 | TensorRT-Edge-LLM，不是 vLLM；可作为 Qwen3 结构一致性的独立参考。 | 给 E 的正确性测试提供明确检查项，但不能直接作为同 runtime 性能臂。 |
| **TEVA (ICCV 2025)** | 高分辨率图像的动态区域 proposal 与 token 效率。 | 研究型模型/评测工作，不是 text/image deadline serving 系统。 | 高分辨率 token 减少已有方法，系统新颖性需要另证。 |

## 精确研究假设（可证伪）

令 `K_safe(x,q)` 为图像/问题在冻结质量校准中允许的最低视觉预算，`D_slack` 为在途文本最紧下一 token deadline slack，`I_slack` 为图像首 token 的剩余 TTFT slack。`B-joint` 只可在 `K_safe` 范围内调节视觉预算，并可选 D 已验证的 N/S 放置；高风险样本不可因为 deadline 紧而跌破安全预算。假设：在特定的 grid、decode 并发和图像到达负载区间，`B-joint` 相对 (i) Native N、(ii) D deadline-only A/S、(iii) fixed-budget GeoPrune、(iv) content-only risk budget，提升配对的联合 SLO goodput，且质量损失没有集中在高风险样本。此为待测组合差异，不是“首创”的断言。

## 必须避免的结论

1. 不说“首次单卡图文混合服务”“首次视觉编码与 LM 解耦”或“首次动态视觉预算”。
2. 不将作者没有公开/链接代码写成确定事实；当前只能写“在论文/作者仓库中未找到可复用实现或明确 pin”。
3. 不把 vLLM 版本/硬件差异当作核心科学新颖性。
4. 不用压缩 token 数代替 E2E、TTFT、KV、显存、队列和联合 goodput。
5. 若目标候选只减少 decoder 输入，明确它不减少前置 ViT encoder E；不能归因于 text decode gap 改善，除非阶段计时证据支持。

## 核心来源

- [SieveServe, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.648/)；[论文 PDF](https://aclanthology.org/2026.emnlp-main.648.pdf)
- [GeoPrune, Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1913/)
- [DuetInfer, DATE 2026 PDF](https://past.date-conference.com/proceedings-archive/2026/DATA/782.pdf)
- [UnifiedServe, arXiv 2512.17574](https://arxiv.org/abs/2512.17574)
- [QoServe, ASPLOS 2026](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/10/QoServe_CC_final_v2.pdf)；[Niyama artifact](https://zenodo.org/records/18218177)
- [TriInfer, MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/f068c65585985c25c17f221390774ec7-Abstract-Conference.html)
- [ET-Prune paper](https://arxiv.org/abs/2608.01979)；[author repository](https://github.com/Labyrinth0419/ET-Prune)
- [PixelPrune author repository](https://github.com/OPPO-Mente-Lab/PixelPrune)
- [Hi-Lo Prune author repository](https://github.com/sealost/Hi-Lo_Prune)
- [NVIDIA DART visual-token pruning](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/features/visual-token-pruning.html)
- [TEVA, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Token-Efficient_VLM_High-Resolution_Image_Understanding_via_Dynamic_Region_Proposal_ICCV_2025_paper.pdf)
