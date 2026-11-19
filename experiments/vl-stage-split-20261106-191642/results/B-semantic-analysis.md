# B 阶段 E→D→P 正确性复核

|单元|episode|错误|文本 token 数|图文 token 数|与 N token IDs 一致|split D 数|后续 P token 数|E 次数|
|---|---|---:|---|---:|---|---:|---:|---:|
|B-N-r0-retry1|B-N-r0-retry1-block1-R-00|0|[192, 192, 192, 192]|16|None|0|N/A|0|
|B-S-r0|B-S-r0-block1-R-00|0|[192, 192, 192, 192]|16|True|4|595|1|
|B-S-r1-2|B-S-r1-2-block1-R-00|0|[192, 192, 192, 192]|16|True|4|595|1|
|B-S-r1-2|B-S-r1-2-block1-R-01|0|[192, 192, 192, 192]|16|True|4|595|1|

## 判读

S 的正确性只在出现真实 encoder-only scheduler step、其间推进已有文本 D、随后图文 P 才记为有效；token ID 不一致或任何校验错误均明确保留并单列。

取消样本：客户端断开记录=True，图像请求随后由 scheduler finished_req_ids 回收=True，4 路文本 token 数=[192, 192, 192, 192]。

详细逐事件证据：`/Users/chenhanxi/Desktop/infra/reproduction_runs/vl-stage-split-20261106-191642/results/B-semantic-analysis.json`。
