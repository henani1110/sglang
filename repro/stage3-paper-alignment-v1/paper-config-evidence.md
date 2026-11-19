# 指定 TokenFlow PDF 配置证据表

核查日期：2026-10-28。范围是用户指定的 EuroSys 2026 PDF、作者当前公开 TokenFlow 仓库说明、服务器上的历史候选与官方 SGLang 源码，以及已经完成的 v2 运行记录。本文件用于冻结下一轮设置，不把不同来源互相替代。

## 版本、基线与开关

| 项目 | PDF / 作者材料 | 本机与本轮实测 | 证据状态与下一步 |
|---|---|---|---|
| 论文实现基础 | PDF第6节说在SGLang上替换默认scheduler并加入hierarchical KV manager；没有报告实验固定commit。作者当前README称TokenFlow是SGLang fork，并以`paper_v1`暴露策略。 | 候选仓库`SJTU-RTEAS/TokenFlow`，HEAD `e754998d6980656707b9628fb5121a4ac4f985df`，2026-04-16 merge `kvoffload`；该版另回移缓存修复并加写回生产者流同步补丁，完整tracked diff SHA256 `1099c3fc37f782c057ad691af9288f22c537384cabf796ef6c3a1e4feea710fd`。 | `[事实]` 候选适配版本可运行；`[未知]` 它不是已确认的论文artifact。需继续检查作者分支、release与论文发布时点，不能把4月16日提交直接标成原论文commit。 |
| 上游 SGLang 关系 | PDF和作者README均说“implemented on top of SGLang”；这不等于已经合入官方上游。 | 独立官方仓库`/home/chx/projects/sglang`为`02d244513871bb426bfd8a8d9de399f660ed24b5`；其`server_args.py`没有`enable_hierarchical_cache`或`kv_offload_policy`字段。候选仓库remote为作者TokenFlow，不保留可用于本机一键匹配的上游remote。 | `[事实]` 不能在这份官方SGLang当前代码里靠一个参数开启论文模块；候选fork和该官方commit也不构成论文年代的公平版本匹配。下轮先做候选基底与作者历史的祖先关系核查，再选择并明确“作者fork关闭路径”或“对齐基底”的基线。 |
| 普通 SGLang 与 chunked | PDF第7.1.4及图16/17将普通SGLang、SGLang(chunked)、Andes分别列出。作者当前README复现说明建议基线省略HiCache及`paper_v1`；chunked另设固定`--chunked-prefill-size`，但公开说明未给4090所用数值。 | v2只测候选fork内关闭TokenFlow功能的路径，实际`chunked_prefill_size=2048`。没单独建立普通SGLang配置，也未运行Andes。 | `[实测]` 现有baseline只代表“候选fork关闭路径 + chunked2048”。`[缺口]` 三系统方案需加明确普通基线，并审计候选关闭路径与可获得的SGLang基底差异；Andes后置。 |
| chunked关闭语义 | PDF未给普通SGLang与chunked的完整CLI。 | 候选scheduler将`chunked_prefill_size<=0`设为`None`；硬件默认逻辑可能自动填2048。当前官方源码的实际版本及依赖须单独对照，不能只把参数省略当作关闭。 | `[代码事实]` 下一轮普通配置应显式设`--chunked-prefill-size -1`并由server-info和执行日志确认无chunk；chunked配置显式固定值。该设定是工程显式化，非论文原CLI事实。 |

## 模型、负载与指标

| 项目 | PDF / 作者材料 | 本轮实测 | 下一轮冻结方式 |
|---|---|---|---|
| 4090请求规模 | PDF第7.3、表1：短突发60请求，长突发80；Poisson短负载2或4请求/s。短负载均值输入512/输出1024，长负载1024/2048。 | E1/E2/P2/P4分别60/80/120/120请求；P2/P4的120是本地方案的适配请求数，非已核实PDF请求数。 | E1/E2数量按PDF；Poisson数量仍标适配值，先查更多作者脚本/完整配置说明。 |
| 长度分布 | 指定PDF文字称长度服从正态分布并给均值，但没给标准差、截断、取整和重采样规则。 | v2沿作者公开benchmark随机请求生成方式，按上限20%～100%均匀取长度；实际均值、上下限和seed保存在workload文件。 | `[来源冲突]` 作者当前公开README的“0.2 range ratio”与随机请求示例更接近区间长度采样，而非PDF描述的正态分布。下一轮若无新增证据，采用明确适配假设`σ=0.2μ`、重采样非法/超context值；同时另存作者脚本式均匀负载作为单独协议，不混称论文原分布。 |
| 请求内容 | PDF和当前public README要求跨系统固定prompt与请求序列；PDF未明确给出ShareGPT语料或文件revision。 | v2调用`sample_random_requests(..., return_text=False)`且未覆盖`random_sample=True`默认值；源码默认加载`anon8231489123/ShareGPT_Vicuna_unfiltered:ShareGPT_V3_unfiltered_cleaned_split.json`，取首轮对话并tokenize后截断/重复。归档token序列不符合随机递增token-ID生成器。工作负载文件本身被保存并用于全部配对运行。 | `[已纠正]` v2是ShareGPT token化prompt，不是随机token ID。原语料下载版本/文件SHA未记录；新生成器要求提供语料文件并记录SHA。输出仍`ignore_eos=true`强制长度，不作为语义质量测试。 |
| 模型 | 作者当前README称评估Llama3与Qwen2.5家族；PDF列多个模型，尚未把每个4090图/场景逐一映射到精确权重revision。 | 本轮用`Qwen2-7B-Instruct` revision `f2826a00ceef68f0f2b946d945ecc0477ce4450c`。 | `[适配模型]` 不能称为已确认的PDF Qwen2.5型号。下轮优先按图核对权重名称与revision；未知时明确写成Qwen2-7B适配复现。 |
| 输出消费速率 | PDF/作者材料报告20、25、30 tokens/s目标速率实验；分图条件尚需继续拆分。 | v2及E2诊断固定20 tokens/s。 | 20是可追溯的适配起点；确认目标图后再冻结场景对应值，不能拿20代表所有论文结果。 |
| 硬件和公共运行设置 | PDF受控4090平台为单GPU实验；两基线完整CLI、显存预算和所有内核设置未列齐。 | GPU1 RTX4090 24GiB、TP=1；BF16、mem-fraction 0.80、deterministic开启、piecewise graph共同关闭、Triton attention、Pytorch sampling、max-running128。 | `[实测设置]` 保留为当前候选适配配置；若下轮改变公共设置，三配置一起冻结并重新验收，不写成论文原值。 |
| 指标 | PDF有效吞吐按客户端buffer的10%/20%阈值计权，并和原始吞吐、TTFT同时报告。 | v2记录SSE接收时间和客户端buffer指标；有效吞吐使用事件时间精度的本地实现，曾明确与作者离散分析实现区分。 | 新矩阵沿用同一版分析器；与作者当前脚本逐项核对输出窗口、TTFT口径和离散tick定义后再对外比较数值。 |

## 下一轮矩阵准入

1. 固定“作者fork关闭路径”与“论文年代上游基底”之间的证据，并为普通SGLang、chunked2048、TokenFlow各生成实际启动参数快照。若无法恢复论文上游基底，则将三组明确称为候选fork内配置对照，不称普通官方SGLang复现。
2. 冻结两套负载协议：PDF文字正态描述的明确适配版，与作者公开0.2 range-ratio脚本版。均保存工作负载JSON、生成脚本、取样统计和SHA256；主分析预先指定其中一种，另一种只做敏感性对照。
3. 现有长生成差异在同prefix探针中对应并列候选和retraction/批次变化，数值敏感性证据强；实际恢复KV逐层状态仍未覆盖。新矩阵可进入准备阶段，但结果报告保留该正确性边界。优先尝试在下一次自然retraction中记录request15/78的缓存节点与重算长度。

来源：用户指定PDF本地文件`/Users/chenhanxi/Zotero/storage/X9HTJM9N/Chen 等 - 2026 - TokenFlow Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling.pdf`第6节、第7.1.4节、第7.3节和表1；作者当前公开[README](https://github.com/SJTU-RTEAS/TokenFlow)及[技术报告](https://github.com/SJTU-RTEAS/TokenFlow/blob/tokenflow/kvoffload/technical_report.md)；服务器固定候选和官方SGLang源码快照；阶段二v2 `server-info`、`workloads`及有效统计。作者README当前写了四种scheduler、两条SGLang基线的做法、4090请求规模和0.2 range ratio；它是公开复现说明，不自动覆盖PDF正文中的分布文字，也不能证明该README当前分支commit就是当年实验artifact。

## 执行核查补记（2026-10-28）

- 作者公开分支本机快照`origin/tokenflow`=`2f2ec646a0ab495df9262d961c5ad30743bf49b4`（2026-08-04），候选`e754998d6`是其祖先；这证明候选在作者公开历史链内，不证明它就是论文实验commit。
- README说明SGLang baseline在同一仓库省略hierarchical cache和`paper_v1`；chunked baseline另显式设置固定chunk。两种基线仍需分开。README写`range_ratio=0.2`，源码实际按`[max(int(mean*0.2),1), mean]`均匀整数采样。
- 已从候选源码确认`--chunked-prefill-size -1`禁用chunked prefill；4090上若省略该参数会默认得到2048。因此普通profile显式传`-1`，chunked和TokenFlow profile显式传2048。
- 已生成三套同fork候选启动profile和双套负载生成器，并依次完成仅启动预检；普通SGLang配置标为“作者fork关闭路径”，不冒称官方上游基线。没有发送推理请求或采集性能。
- `[纠正 / prompt语料]` 本次读取v2归档的`E2-s101.json`：request0 token相邻递增比例为0，request1开头为`[5158,311,1855,18287,...]`等自然token序列；结合`sample_random_requests`默认`random_sample=True`及其ShareGPT代码，修正旧报告“随机token合成负载”的说法。v2使用ShareGPT首轮文本tokenized后截断/重复。下载文件SHA未保存在归档中；旧工作负载JSON本身及其哈希仍可重放。
- 新工作负载脚本要求本地ShareGPT JSON输入并写入SHA256，沿用同一提示样本和序列规则。PDF正态方案采用`σ=0.2μ`、正数且context内拒绝重采样；作者README方案按源码复现range-ratio均匀采样。两者作为明确分离的适配协议。
- 三种profile启动预检均通过`/server_info`断言，包括plain的CLI值-1、两种chunked值2048，以及TokenFlow开关。GPU1恢复到34MiB/0%，端口30000未监听，源diff哈希未变。摘要和日志见`stage3-paper-alignment-v1/preflight/结果.md`，归档SHA为`3f7195ca4af663fadfea9828ee1275c96ffc1b4cd05aa6763df8a31ad0e47d20`。

## E2适配矩阵执行后更新（2026-10-28）

- `[事实]` 固定ShareGPT数据和24个工作负载后，按E2（80个突发请求、输入/输出长度均值1024/2048）完成三profile×三seed尝试。正态分布取σ=0.2μ、越界样本重采样；该分布参数仍是适配假设。
- `[观察]` Plain与TokenFlow各3/3轮完成，chunked 1/3轮完成；chunked seed101为72/80、seed102为79/80、seed103为80/80。总计720请求中711个响应完成，其中560个响应来自整轮完成的7次运行，可纳入完整轮指标均值。
- `[观察]` 完整轮均值中TokenFlow相对plain有效吞吐更高、TTFT更低；原始吞吐变化很小，端到端均值和stall fraction更高。chunked缺少两轮，不作三seed总体比较。具体指标和逐请求失败详情见`e2-paper-normal-v1/结果.md`与`summary.json`。
- `[推断 / 待诊断]` 两次未完整chunked轮的尾部日志显示服务端停滞且仍有请求挂起；根因尚未区分客户端SSE处理、SGLang scheduler/retraction或其他路径。
- `[诊断观察]` seed101/request9和seed102/request66在相同chunked配置、相同预热与cache flush后单独复放均完成（87.27s/72.73s）；因此请求内容/长度本身不是必然失败原因，问题依赖突发并发期。常规日志没有request-ID级retraction与重新调度链，尚不能定位具体机制，见`chunked-timeout-diagnostic-v1/结果.md`。
- `[下一步]` E2突发期错误原因尚未闭合，且暂停恢复KV逐层正确性仍未核实；按预案不扩展E1/P2/P4。若继续根因定位，先给retraction、waiting queue、prefill/decode和finish/abort加短期request-ID跟踪，再决定是否补跑缺失chunked seed。候选源码、参数和工作负载仍仅代表论文适配方案，不升级称为原图复现。
