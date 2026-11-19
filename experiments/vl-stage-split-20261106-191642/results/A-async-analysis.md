# A 阶段异步探针续做复算

## 有效 episode 与客户端 token 覆盖

|探针|episode|T|R|无效|期望/观测文本 token|期望/观测图文 token|
|---:|---:|---:|---:|---:|---:|---:|
|0|32|16|16|0|24576/24576|256/256|
|1|32|16|16|0|24576/24576|256/256|

## 分块窗口最大文本间隔 P95（ms）

|探针|块|T P95|R P95|R−T|
|---:|---:|---:|---:|---:|
|0|1|5.479|24.023|18.544|
|0|2|5.412|24.246|18.834|
|1|1|5.337|23.408|18.071|
|1|2|5.387|23.753|18.366|

## Probe-on/off 描述性差异

|臂|off P95|on P95|差值|允许差|
|---|---:|---:|---:|---:|
|T|5.463|5.370|0.093|1.000|
|R|24.157|23.627|0.530|1.208|

## 异步队列与事件覆盖

- writer：submitted=49664，written=49664，dropped=0，write_errors=0，closed_cleanly=True。
- probe-on server event：49682 行；scheduler event：6312 行。
- 诊断 episode 单独保留在原始 JSON 中，不计入正式窗口统计。
- 这是单轮续做的描述性对照；不得与上轮正式 256 episode 简单混并当作独立重复。

详细逐 episode 数据：`/Users/chenhanxi/Desktop/infra/reproduction_runs/vl-stage-split-20261106-191642/results/A-async-analysis.json`。
