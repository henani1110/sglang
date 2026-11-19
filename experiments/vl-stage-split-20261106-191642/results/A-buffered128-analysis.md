# A 阶段：缓冲式逐 token 复测（buffered）

## 分块客户端 P95（ms）

|probe|块|T n|T P95|R n|R P95|R−T|相对 T|门槛|
|---:|---:|---:|---:|---:|---:|---:|---:|---|
|0|1|8|5.324|8|24.179|18.854|354.1%|True|
|0|2|8|5.320|8|24.206|18.886|355.0%|True|
|1|1|8|6.479|8|23.797|17.318|267.3%|True|
|1|2|8|7.023|8|23.510|16.486|234.7%|True|

## 插桩开销

|臂|probe-off pooled P95|probe-on pooled P95|绝对差|允许差|通过|
|---|---:|---:|---:|---:|---|
|T|5.326|7.016|1.690|1.000|False|
|R|24.219|23.716|0.503|1.211|True|

## 校验与事件对齐

- probe-off：32 个测量 episode，校验错误 0；SSE 文本 token 覆盖 24576/24576，图像 token 覆盖 256/256。
- probe-on：32 个测量 episode，校验错误 0；SSE 文本 token 覆盖 24576/24576，图像 token 覆盖 256/256。
- probe-on 的 R episode：16；首个图文混合 scheduler step 16；视觉 forward 16；最大间隔的 engine/API 双端 token join 16，token ID 一致 16。
- 对 128 个正式文本 completion request 逐 token ID 复核：engine-ready 128/128 完全一致，API-yield 128/128 完全一致。图文 chat 输出 token 与 TTFT 由客户端 SSE 计数和计时；hook 未覆盖 chat engine/API endpoint。

## 代表性对齐时间线

选择最大客户端间隔 episode `A-default-probe1-buffered-probe1-1-R-07`。时间均相对 episode anchor，单位 ms。

|事件|相对 anchor|
|---|---:|
|首个混合 scheduler step|14.247067|
|visual forward 起点|15.399944|
|最大间隔左侧 engine-ready|26.887913|
|最大间隔左侧 API yield|26.904089|
|最大间隔左侧客户端 SSE|27.259553|
|最大间隔右侧 engine-ready|50.766731|
|最大间隔右侧 API yield|50.77862|
|最大间隔右侧客户端 SSE|51.126903|
|客户端间隔 / engine-ready 间隔 / API-yield 间隔|23.86735 / 23.878818 / 23.874531|

详见 `A-buffered-analysis.json` 和 `A-buffered-aligned-timeline.json`。事件顺序能定位长间隔发生在图文混合步及视觉 forward 之后；它支持这组负载下的时间关联，不能单独证明唯一因果机制。
