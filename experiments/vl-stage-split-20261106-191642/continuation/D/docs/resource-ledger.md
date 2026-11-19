# D 阶段资源账本

更新时间：2026-11-09（Asia/Shanghai）

## 固定资源与路径

- 远端主机：`chx-gpu-4090-2`（系统名 `flyvideo-MS-7E06`）。
- 实验 GPU：GPU0 RTX 4090，负载 PCIe Gen4 ×16；GPU1 RTX 4090，负载 PCIe Gen4 ×4，未参与矩阵。
- SSD：`/media/chx/Win10/vl-stage-split-20261106-191642`。原始 formal traces 留存在 SSD 的 `continuation/D/logs/formal/`。
- 本地复现根：`/Users/chenhanxi/Desktop/infra/reproduction_runs/vl-stage-split-20261106-191642`。
- 正式 schedule SHA-256：`811cfc12e174455ac6386e9f008f5031e9bc72f936272beccc4000810300cbb7`。
- 冻结 estimator profile SHA-256：`92bb071b94599533a79113f2bf519dff785b75788ffd27ccd7d8ffe0c9ac9483`。

## 完成批次

- D 正式矩阵：432/432 个唯一运行成功，36 个策略条件、每格 12 个配对 seed；`invalid_attempts=0`，运行 manifest 的 process exit 均为 0。
- trace QA：server/API/client 首 token 顺序和首 token ID 均通过 46,656/46,656；异步 writer 写入 613,696,168 行、丢弃 0、写错误 0。
- GPU0 使用完毕后：`nvidia-smi` 显示 0% GPU 利用率、782 MiB 显存；GPU1 为 0% 利用率、45 MiB。可见显存来自主机桌面/Xorg/GNOME/远程桌面进程；已无本实验推理服务。端口 8271 无监听。
- 编排器及 vLLM 服务均已退出。首轮 CPU 分析成功；修复内部 request-ID 关联的最终 CPU 分析目前正在 SSD trace 上重跑，结束后将再次记录 PID、退出码及资源快照。
- 检查时 SSD 约有 166 GiB 可用。本地文件系统约有 74 GiB 可用；正式 raw JSONL 事件仍在 SSD，最终压缩包校验后再复制至本地。

## 分析修复审计

- 首轮分析输出保留在 `results/formal-analysis-pre-id-join-fix/`。
- 第二轮分析输出保留在 `results/formal-analysis-partial-idfix/`：已修复 scheduler 和运行指标的 ID 关联，确认 4,616/4,616 个 A 选中 S 的决策均实际执行拆步；状态分区末端的决策结果关联仍有遗漏，因此该轮不作最终状态结论。
- 最终分析输出目录预定为 `results/formal-analysis-final-idfix/`，使用相同的冻结输入、5,000 次配对 bootstrap 和分析 seed `20261107`。仅最终目录用于主结果结论；中间目录均作为审计证据归档。

## 最终归档待补

- 最终 CPU 分析的完成时间、退出码、摘要和状态分区核验结果。
- 远端资源释放复核时间，归档压缩和提取校验结果。
- 本地/SSD 完整归档路径、文件数、大小、SHA-256 及逐文件 manifest 校验状态。

## 最终分析与资源复核（2026-11-09）

- 第四轮正式分析在 SSD 原始 trace 上完成，退出码 0，分析 CPU 耗时约 23 分 23 秒。唯一最终输出为 `results/formal-analysis-final-dedup-idfix/`。摘要：432/432 runs，36 cells，196 paired comparisons；9,504 条 A decision；trace 顺序/首 token ID 46,656/46,656；108/108 个唯一 writer metadata 关闭正常，`written=153,424,042`、`dropped=0`、`write_errors=0`。最终分析器 SHA-256：`78f75c666a2087cfca92d49302e46d95c7b0c78dc712371cd98b0f83cd3d05fc`。
- 2026-11-09 07:11 +08:00 远端收尾快照：GPU0/GPU1 利用率均 0%；空闲显存分别 785/45 MiB；空闲即时链路读数 Gen1 ×16 / Gen1 ×4（负载时已记录为 Gen4 ×16 / Gen4 ×4）。8271 无监听；未发现 vLLM、campaign 或正式分析器进程（命令自身的 grep 行除外）。SSD 是 `/dev/nvme0n1p3` 上的 `/media/chx/Win10`，剩余 166 GiB。
- 本地工作区文件系统可用 74 GiB。归档、校验与临时解包结果在下文补记；在完成逐文件校验前不得删除原始 SSD trace 或本地来源文件。

## 最终归档位置与清单（2026-11-09）

- 完整证据包固定路径：/media/chx/Win10/vl-stage-split-20261106-191642/final-bundles/D-evidence-20261109/D-evidence-20261109.tar.gz；本地镜像位于 reproduction_runs/vl-stage-split-20261106-191642/artifacts/archive-tars/D-evidence-20261109.tar.gz。
- 内容为本地 A/B/C 与 D 运行材料、SSD 权威 D 数据集/日志/配置/分析/图表、相关研究方案、AGENTS 与主日志。包内 SHA256SUMS.txt 为每个归档文件记录 SHA-256；同目录 .tar.gz.sha256 文件记录整个压缩包摘要。归档校验流程与最终本地/SSD 文件信息由当前 Goal 的完成记录给出。

