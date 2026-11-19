# vLLM 0.33 / Qwen3-VL-2B 图像 token 压缩接口预检

## 已核对环境

远端 `chx-gpu-4090-2` 当前固定环境：

- Python 3.11.13（venv: `/home/chx/project/vl-encode-pilot-20261106-085428/env/serve311`）
- `vllm==0.33.0`
- `torch==2.14.0+cu130`
- `transformers==5.20.2`
- `flashinfer-python==0.7.2.post1`
- Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`；权重只读复用 SSD
- vLLM Qwen 源码 SHA-256 `d46644daaf541fc2b78992c7af0563cf1762d8089d2a55d916dded1366b3c0e8`

## 源码观察

核对文件：`/home/chx/project/vl-encode-pilot-20261106-085428/env/serve311/lib/python3.11/site-packages/vllm/model_executor/models/qwen3_vl.py`。

- `_process_image_input`（约 2460–2479 行）由 `image_grid_thw` 调 `self.visual(pixel_values, grid_thw)`，再按每张图的 merger 后 token 数切分 embedding。
- `embed_multimodal`（约 3058–3086 行）图像路径目前调用 image forward 和 EVS 后处理，但与视频路径不同，没有在该分支直接套用 `is_multimodal_pruning_enabled` 的 image-token keep mask。
- 视觉编码器返回主视觉特征与多个 DeepStack 特征拼接后的输出（约 899–910 行）；`_compute_deepstack_embeds`（约 3088–3127 行）把它拆成主特征和多尺度特征，再按 `is_multimodal` mask 合并到语言 token embedding。
- 视觉旋转/位置元数据由原 `grid_thw` 生成（约 718–813 行）。实际从 input IDs/processor 生成的位置索引还必须继续核对；不得仅切片输出 embeddings 后继续沿用旧 image placeholder 数和旧位置。
- 模型配置使用 DeepStack 和 Qwen3-VL 的 interleaved M-RoPE。将一个 image token 物理删除时，必须对齐地改动图像占位 token、主视觉 embedding、所有 DeepStack plane、视觉 mask、原始位置行，并确保 scheduler 与 KV reservation 使用压缩后的真实长度。

## 候选实现路线

1. **GeoPrune（首选固定预算）**：在 visual encoder 输出的 `S_out` 计算 deflation/residual-L2 排名，用同一 keep indices 选全部主/DeepStack feature。其后还要把对应视觉输入占位与 M-RoPE 一起压紧。论文公开报告“集成 vLLM”，但尚无可用仓库/精确 vLLM pin；此处是忠实方法适配，不能声称复现其开源代码。
2. **SieveServe-inspired（内容预算）**：复用上面同一 token selector，另由 query complexity + image dense-risk guard 选保守/激进/不剪枝预算；记录与 QCR/DIG 原文差异、CPU/GPU 代价、阈值来源和风险 false-negative。高风险样本只能保留安全预算，不能为追 deadline 越过质量 floor。
3. **PixelPrune compatibility branch**：作者仓库直接提供 Qwen3-VL-2B 示例和 vLLM plugin，但仅说明测试 `vllm==0.18.0` / `transformers==4.57.6`。对照 E 的 0.33.0/5.20.2 版本差异，先做 API 差异核对；仅在兼容适配可审计时增加 pre-ViT 分支。它有机会降低 E，本身不是 GeoPrune 排名的等价实现。
4. **NATIVE fallback**：若 post-ViT 物理删 token 无法完整更新 scheduler/KV/M-RoPE，就保留错误证据，先沿 vLLM processor/placeholder 路径实现；若仍不行，尝试 PixelPrune 的 pre-ViT路径。不得用只裁剪分析数组、padding 代替真实 decoder token 删除。

## 正确性/效益前置验证（逐项保存原始材料）

- full-token 路径与未剪枝 patch 在 greedy 解码下完整 response 一致；request ID、image token 计数与 M-RoPE 维度逐项相同。
- 压缩样本的保留索引跨 main/DeepStack feature 一致且顺序稳定；processor `grid_thw`、input IDs placeholder 数、positions/M-RoPE 行、模型输入 token 数和 KV 实际增长一致。
- 单图、同样本多问题、不同 grid 512/768、动态预算边界和 batch 中多图混合都检查；初期只声明支持且测量当前已过验证的范围。
- 无剪枝输出与原路径的 full response 先完整一致；压缩路径保存完整 response、官方评分，不采用 D 的 16-token 截断来报告质量。
- 用 CUDA Events 分别测 E 与 P，延后读结果，另记 CPU dispatch、H2D、scheduler queue、API yield 与客户端首 token；报告方法本身的开销和事件是否扰动 serving。
- 同时读 KV cache/每请求 token 计数和实际模型侧输入长度。token 数下降但 KV 和 prefill 未下降时，不报告成系统收益。

## 未解决项

- vLLM 0.33 Qwen3-VL 的 multimodal placeholder replacement 和 M-RoPE positions 的真实生成/压紧接口，还需用同版本 processor/worker 源码定位并形成补丁前后审计。
- 是否能在当前 vLLM 0.33 eager/CUDA graph/异步调度下动态改变单请求视觉序列长度，需要实测。若动态图重捕获或 padding 吞掉收益，保留该结果并报告。
- GeoPrune 原作者代码、精确 vLLM commit 与 2B 配置未找到，需要持续查找，且不得把当前自适配算作者实现。

## 2026-11-09 补充：每请求分辨率预算路径已通过 CPU processor 预检

### 固定源码接口

- vllm/model_executor/models/qwen3_vl.py 的 Qwen3VLProcessingInfo.get_supported_mm_processor_kwargs() 从图像处理器继承 image kwargs，并另行补充 video min_pixels/max_pixels；图像参数进入 images_kwargs 路由。
- Qwen3VLMultiModalProcessor._apply_hf_processor_main() 对图像 size/min_pixels/max_pixels 解析为完整 size={shortest_edge,longest_edge} 后再调用 HF processor。
- 同文件 _get_prompt_updates() 的 image replacement 从已经完成 resize 的 out_mm_kwargs["image"][i]["image_grid_thw"] 计算 prod(grid_thw)/merge_length 个 image placeholder。缩小图像会同时改变 patch grid 和真实 placeholder 数；LM 输入长度由处理结果计算，scheduler/KV 无需另外使用原图 token 数。最后仍须通过真实 OpenAI serving 请求核实端到端计数。
- OpenAI Chat Completions schema 含 mm_processor_kwargs。计划以每请求 mm_processor_kwargs.images_kwargs.size 选择图像最大像素数。Transformers 5.20.2 仍接受 max_pixels，但发出 v5.23 将移除的弃用警告；正式配置改用 images_kwargs.size.longest_edge，并显式保留默认 shortest_edge。

### 已执行的 CPU 合成图预检

远端固定 venv、SSD 本地模型 processor 和一张合成 768×768 RGB 图像；没有启动 GPU。

| processor 参数 | image_grid_thw | image_pad 数 | pixel_values shape |
|---|---:|---:|---:|
| 默认 | [1,48,48] | 576 | [2304,1536] |
| max_pixels=262144 | [1,32,32] | 256 | [1024,1536] |
| max_pixels=65536 | [1,16,16] | 64 | [256,1536] |
| images_kwargs.size={shortest_edge:3136,longest_edge:262144} | [1,32,32] | 256 | [1024,1536] |

processor 完整 __call__ 的默认、顶层旧式 max_pixels、作用域 images_kwargs.max_pixels、作用域 images_kwargs.size 四种输入都通过。前两档/第三档占位符分别与上表一致；作用域 size 与 262144 档一致。这里“预算”是由像素上限经过 Qwen smart-resize 和 patch merge 得到的实际 token 数，不能把 max_pixels 直接当成 token 数；实际 image_grid_thw 和 placeholder count 应逐请求记入 trace。

### 对 E 实验的影响

这条原生 per-request processor 路径可在 ViT 前降低分辨率，并由 stock processor 同步缩短 image placeholders；它不是 GeoPrune 的 post-ViT token ranking。它避免当前 v0.2 设想中手工同步裁剪 visual/DeepStack/M-RoPE 的高风险改动，同时有机会降低 E 和 P，但会改变可见像素、必须重新做完整答案质量评估。后续 v0.3 将把它作为主要可运行预算 actuator；GeoPrune/PixelPrune 作为独立后续比较，不把结果混称为剪枝复现。尚未验证的事项：真实 Chat Completions 端到端 mm_processor_kwargs 接受性、服务端 prompt token 计数、CUDA-event E/P、输出与 scheduler/KV 实测。

### OpenAI serving smoke 结果

2026-11-09 在 GPU0 的 loopback Chat Completions 服务上，对一张本地程序生成的合成 768×768 柱状图依次发送原生默认、262144 pixel cap、65536 pixel cap。请求需要在仅绑定 127.0.0.1 的实验服务加 --trust-request-mm-kwargs；默认关闭时，第一条无 kwargs 请求成功，带 kwargs 请求返回 400。失败服务及完整日志保留。另一次启动因远端自带 Python 头文件未加入 CPATH 而失败，随后按 D runner 的现有路径设置 CPATH/C_INCLUDE_PATH 修复；失败日志保留。

重试服务三条均返回 HTTP 200。usage.prompt_tokens 为 595、275、83。结合 CPU processor 检查的 image placeholder 数 576、256、64，三者都恰为 19 个文本/模板 token 加视觉 token 数，验证 Chat Completions 使用每请求 cap 后，返回的 prompt 长度与处理器的 image grid 展开一致。三档输出内容有差异；低档误读了合成图标题。该单样本定性观察不能作为质量统计。请求输出和日志的 SHA-256 已记录在 E progress log。

服务未插入 CUDA-event / scheduler / per-request KV instrumentation，因此这次不能声称测得 E/P GPU 时长或逐请求物理 KV 增量。服务已停止，8285 端口释放；GPU0 回到 782 MiB、GPU1 为 45 MiB，SSD 剩余 162 GiB。

## 2026-11-09 补充：实际 V2 runner 的 E/P 与 API trace smoke

### Runner 路径核验与补丁修正

- `[事实]` vLLM 0.33.0 的 `Worker._make_model_runner()` 在当前配置选择 `vllm.v1.worker.gpu.model_runner.GPUModelRunner`（V2），不是 `vllm.v1.worker.gpu_model_runner.GPUModelRunner`（V1）。远端 import preflight 显示 `gpu_worker_selects_v2_runner=true`，scheduler 未加载 D policy，所有入口都解析到 E overlay。
- `[事实 / 失败材料]` 第一版 E P-event 插在 V1 `_model_forward`，实际服务只看到视觉与 scheduler event，没有 LM stage event。这次未计作成功计时；脚本、diff、运行 trace 均保留为 `patch_model_runner_v1_exploratory_e.py` / `gpu_model_runner_v1_exploratory_e.diff`。
- `[决定]` 当前 V2 补丁包围真实 FULL CUDA Graph replay、PIECEWISE replay、ubatch 与 eager 模型 launch；event 记录 scheduler request token map 和 encoder-input map。排除 dummy、profile 与 `_warmup_` request。E trace 的 `mm_hash=null` startup profiling event 仍保留在原始 flush 结果中，分析时必须剔除。

### 端到端 trace 结果

- `[事实]` `e-trace-v2-4` 同一 synthetic image 的三个逐请求档位均返回 HTTP 200，prompt token usage 为 595、275、83，与 CPU processor 的 576、256、64 image placeholders 加 19 个文本/模板 token 一致。纯文本 `/v1/completions` 也返回 HTTP 200。
- `[事实 / token join]` Chat 响应各 32/28/32 个 token，completion 响应 16 个 token；四个 request ID 的 `engine_result_ready_to_api` 和 `api_yield` token ID 序列逐项相同，index 范围连续且无缺口。
- `[事实 / CUDA-event]` V2 trace 为三张图记录 3 个 eager prefill batch（scheduler 分别为 595/275/83 tokens）和 90 个单-token FULL graph replay；文本请求记录 1 个 17-token prefill 和 15 个单-token graph replay。图像 decode CUDA event 的 smoke 中位数约 3.75 ms，文本约 3.82 ms；这些是低并发 mechanism smoke 的单次观察，不是正式延迟或性能结果。
- `[观察]` 重复 smoke 的 visual CUDA-event 有波动：native 约 18.6–19.3 ms，256-token 档约 7.1–14.2 ms，64-token 档约 6.3–13.9 ms；prefill P 时间同样有运行间变化。该现象要求在正式开放环矩阵保留 run/seed 分层、时钟/负载信息并报告分布；不能据此推断分辨率 actuator 的平均效益。
- `[事实 / 完整性]` worker 的 collective flush 先为 async event queue 写入 barrier summary，再返回；本轮 worker snapshot 分别为 97/97 与 114/114 rows，dropped=0、write_errors=0。API/server writer 为 216/216 rows，dropped=0、write_errors=0。事件行增加 PID 与 async-state，summary 文件按 PID 命名，避免多进程共用同一 meta 文件名。
- `[事实 / 清理]` 服务停止后 GPU0 为 782 MiB、GPU1 为 45 MiB，8285 已释放，SSD 剩余约 161 GiB。`e-trace-v2-4` 原始输出、scheduler/server JSONL、server log、per-process summary、patch/diff/source 与本地 review copy 都已分别保留。

### 仍缺少的计时和状态观测

- `[事实]` 当前 P CUDA-event 是模型 launch/replay 边界的 GPU 时间；尚未观测逐请求物理 KV bytes、每项 image grid 与 E CUDA event 的显式 request-ID 关联、CPU dispatch、H2D duration、排队分解或 trace 开销。
- `[待办]` 增加 KV block/实际 token 观测与 CPU/H2D trace；为 visual E event 补充可稳定连接到 request ID 的 key；完成显式默认 processor kwargs 等价、长宽比/512/768、多图与 batch smoke；再开始权利闭合后的 600-item quality set、风险估计器校准及 480-run open-loop matrix。

## 2026-11-09 补充：visual E 到 request ID 的直接关联

- `[事实 / 接口]` 在 V2 `prepare_inputs_embeds` 调用前，`scheduler_output.scheduled_encoder_inputs` 保留 request ID → multimodal input index；V2 runner 的 encoder cache 保留 request ID → `mm_features`，其中每项有 `identifier`。E-only runner patch 将两者组成 `mm_hash → request IDs` 映射，并在同一执行上下文中传给视觉 hook。批量编码时一个 CUDA event 可关联多个 request ID。
- `[事实 / smoke]` `e-trace-v2-5` 的三条合成图 Chat 请求均 HTTP 200，response IDs 为 `chatcmpl-8e775fcdebe9746e`、`chatcmpl-96d5b6bf4fd6fc85`、`chatcmpl-84f9f13baddc3c33`，prompt token 分别为 595、275、83。`runs/processor-smoke/e-trace-v2-5-event-join.json` 检查结果 `valid=true`：3/3 非启动 profiling 的视觉事件各有 request ID；3/3 都能在 scheduler encoder-input rows 与 API `engine_result_ready_to_api`/`api_yield` 中找到对应请求。
- `[观察 / 边界]` 这三次独立 smoke 的视觉 CUDA-event 分别为 19.270、14.130、13.853 ms，每档仅一次、串行低并发机制验证。它们不支持不同预算的性能比较。约 3.2 秒的启动 profiling event 因 `mm_hash=null` 被连接审计剔除。
- `[事实 / 清理]` text completion smoke 返回 HTTP 200。服务停止后 8285 端口释放，GPU0/GPU1 回到 782/45 MiB，SSD 可用 161 GiB。原始 flush、scheduler/server JSONL、API 输出、worker summary、overlay/diff/manifest 均保留在 E 工作区。
- `[待办]` 该 smoke 每次只串行处理一张图；还要验证同一 batch 多 request、多图 request 下的一对多关联。physical KV bytes/block 和 resident-token 增量、CPU dispatch、H2D、queue 分解、instrumentation overhead、显式默认 kwargs 等价、多尺寸/aspect ratio 以及授权闭合的质量集仍未完成；formal run 仍为 0。
