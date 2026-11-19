# A 阶段：缓冲式逐 token 复测（buffered512）

## 分块客户端 P95（ms）

|probe|块|T n|T P95|R n|R P95|R−T|相对 T|门槛|
|---:|---:|---:|---:|---:|---:|---:|---:|---|
|0|1|8|5.462|8|24.220|18.758|343.4%|True|
|0|2|8|5.348|8|24.851|19.503|364.7%|True|
|1|1|8|5.548|8|30.640|25.092|452.3%|True|
|1|2|8|5.388|8|26.984|21.596|400.8%|True|

## 插桩开销

|臂|probe-off pooled P95|probe-on pooled P95|绝对差|允许差|通过|
|---|---:|---:|---:|---:|---|
|T|5.450|5.522|0.072|1.000|True|
|R|24.761|29.714|4.953|1.238|False|

## 校验与事件对齐

- probe-off：32 个测量 episode，校验错误 0；SSE 文本 token 覆盖 24576/24576，图像 token 覆盖 256/256。
- probe-on：32 个测量 episode，校验错误 0；SSE 文本 token 覆盖 24576/24576，图像 token 覆盖 256/256。
- probe-on 的 R episode：16；首个图文混合 scheduler step 16；视觉 forward 16；最大间隔的 engine/API 双端 token join 16，token ID 一致 16。
- 对 128 个正式文本 completion request 逐 token ID 复核：engine-ready 128/128 完全一致，API-yield 128/128 完全一致。图文 chat 输出 token 与 TTFT 由客户端 SSE 计数和计时；hook 未覆盖 chat engine/API endpoint。

## 代表性对齐时间线

选择最大客户端间隔 episode `A-default-probe1-buffered512-probe1-1-R-00`。时间均相对 episode anchor，单位 ms。

|事件|相对 anchor|
|---|---:|
|首个混合 scheduler step|24.597181|
|visual forward 起点|28.578741|
|最大间隔左侧 engine-ready|25.101055|
|最大间隔左侧 API yield|25.116857|
|最大间隔左侧客户端 SSE|25.188492|
|最大间隔右侧 engine-ready|56.548087|
|最大间隔右侧 API yield|56.565943|
|最大间隔右侧客户端 SSE|56.638016|
|客户端间隔 / engine-ready 间隔 / API-yield 间隔|31.449524 / 31.447032 / 31.449086|

详见 `A-buffered512-analysis.json` 和 `A-buffered512-aligned-timeline.json`。事件顺序能定位长间隔发生在图文混合步及视觉 forward 之后；它支持这组负载下的时间关联，不能单独证明唯一因果机制。
