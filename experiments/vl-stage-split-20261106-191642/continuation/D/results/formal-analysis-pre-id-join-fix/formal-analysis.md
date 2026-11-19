# D 正式矩阵分析

- 有效 seed-policy runs：432/432；策略条件：36。
- Schedule SHA-256：811cfc12e174455ac6386e9f008f5031e9bc72f936272beccc4000810300cbb7。
- Frozen profile SHA-256：92bb071b94599533a79113f2bf519dff785b75788ffd27ccd7d8ffe0c9ac9483。
- Bootstrap：以配对 trace seed 为单位，5000 次重采样；分析随机种子 20261107。
- 每个策略条件的 P50/P95/P99 指标先在 run 内计算，再以 12 个 seed 的 run-level 分布汇总；配对区间比较候选策略减 N。
- 联合 SLO goodput：按测量期图像请求计数；图文 TTFT 达标，且各文本 lane 在该图像首 token 等待区间内的最大 client token gap 达标；报告每秒通过图像数与通过比例。
- formal 模式没有 CUDA-event 视觉时长；E 单项留出误差不可直接观测。报告 engine-ready−P-step 与合并 E+P 估计误差，split 策略另报 P 与拆步间隔误差。

## 各条件结果

| Grid | 文本流 | 到达 | 策略 | Seed runs | 文本最大间隔 P95 (ms) | 图文 TTFT P95 中位值 (ms) | 联合 goodput (req/s) | 图像完成率 |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 512 | 2 | burst | A | 12 | 51.409 | 52.770 | 0.963 | 1.000 |
| 512 | 2 | burst | F | 12 | 47.330 | 51.057 | 0.959 | 1.000 |
| 512 | 2 | burst | N | 12 | 54.249 | 50.732 | 0.962 | 1.000 |
| 512 | 2 | burst | S | 12 | 53.832 | 53.223 | 0.941 | 1.000 |
| 512 | 2 | burst | U | 12 | 54.171 | 51.598 | 0.981 | 1.000 |
| 512 | 2 | uniform | A | 12 | 52.677 | 53.011 | 0.097 | 1.000 |
| 512 | 2 | uniform | F | 12 | 44.100 | 50.976 | 0.098 | 1.000 |
| 512 | 2 | uniform | N | 12 | 45.858 | 51.796 | 0.097 | 1.000 |
| 512 | 2 | uniform | S | 12 | 53.614 | 51.157 | 0.098 | 1.000 |
| 512 | 6 | burst | A | 12 | 73.676 | 52.274 | 0.912 | 1.000 |
| 512 | 6 | burst | F | 12 | 59.959 | 53.711 | 0.904 | 1.000 |
| 512 | 6 | burst | N | 12 | 68.682 | 53.411 | 0.911 | 1.000 |
| 512 | 6 | burst | S | 12 | 74.110 | 54.463 | 0.894 | 1.000 |
| 512 | 6 | burst | U | 12 | 69.392 | 53.218 | 0.904 | 1.000 |
| 512 | 6 | uniform | A | 12 | 67.450 | 56.363 | 0.088 | 1.000 |
| 512 | 6 | uniform | F | 12 | 48.924 | 54.008 | 0.088 | 1.000 |
| 512 | 6 | uniform | N | 12 | 61.104 | 55.490 | 0.091 | 1.000 |
| 512 | 6 | uniform | S | 12 | 58.980 | 55.208 | 0.092 | 1.000 |
| 768 | 2 | burst | A | 12 | 61.900 | 79.795 | 0.001 | 1.000 |
| 768 | 2 | burst | F | 12 | 59.450 | 75.390 | 0.005 | 1.000 |
| 768 | 2 | burst | N | 12 | 62.044 | 74.543 | 0.007 | 1.000 |
| 768 | 2 | burst | S | 12 | 62.203 | 79.557 | 0.000 | 1.000 |
| 768 | 2 | burst | U | 12 | 62.286 | 74.645 | 0.007 | 1.000 |
| 768 | 2 | uniform | A | 12 | 61.750 | 81.243 | 0.000 | 1.000 |
| 768 | 2 | uniform | F | 12 | 50.509 | 74.151 | 0.000 | 1.000 |
| 768 | 2 | uniform | N | 12 | 59.829 | 75.537 | 0.000 | 1.000 |
| 768 | 2 | uniform | S | 12 | 58.760 | 79.191 | 0.000 | 1.000 |
| 768 | 6 | burst | A | 12 | 86.900 | 81.807 | 0.002 | 1.000 |
| 768 | 6 | burst | F | 12 | 75.683 | 76.553 | 0.002 | 1.000 |
| 768 | 6 | burst | N | 12 | 82.676 | 78.616 | 0.000 | 1.000 |
| 768 | 6 | burst | S | 12 | 77.427 | 83.018 | 0.000 | 1.000 |
| 768 | 6 | burst | U | 12 | 84.395 | 78.512 | 0.002 | 1.000 |
| 768 | 6 | uniform | A | 12 | 60.941 | 83.190 | 0.000 | 1.000 |
| 768 | 6 | uniform | F | 12 | 61.492 | 76.713 | 0.000 | 1.000 |
| 768 | 6 | uniform | N | 12 | 71.260 | 76.182 | 0.000 | 1.000 |
| 768 | 6 | uniform | S | 12 | 75.411 | 84.510 | 0.000 | 1.000 |

## 配对差值区间

| Grid | 文本流 | 到达 | 候选减 N | 指标 | 均值差 | Bootstrap 95% CI | n |
|---:|---:|---|---|---|---:|---:|---:|
| 512 | 2 | burst | F | text_max_gap_ms | 2.112 | [-5.698, 9.698] | 12 |
| 512 | 2 | burst | F | text_gap_p95_ms | -0.028 | [-0.035, -0.020] | 12 |
| 512 | 2 | burst | F | image_ttft_p95_ms | -1.286 | [-3.115, 0.404] | 12 |
| 512 | 2 | burst | F | joint_30_55_goodput_per_s | -0.002 | [-0.036, 0.038] | 12 |
| 512 | 2 | burst | F | image_queue_wait_p95_ms | -0.555 | [-4.212, 3.398] | 12 |
| 512 | 2 | burst | F | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | burst | F | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | burst | S | text_max_gap_ms | -2.591 | [-10.391, 5.537] | 12 |
| 512 | 2 | burst | S | text_gap_p95_ms | 0.006 | [-0.003, 0.015] | 12 |
| 512 | 2 | burst | S | image_ttft_p95_ms | 1.993 | [0.339, 3.878] | 12 |
| 512 | 2 | burst | S | joint_30_55_goodput_per_s | -0.020 | [-0.087, 0.031] | 12 |
| 512 | 2 | burst | S | image_queue_wait_p95_ms | -0.468 | [-3.630, 2.450] | 12 |
| 512 | 2 | burst | S | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | burst | S | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | burst | A | text_max_gap_ms | -6.092 | [-14.954, 2.765] | 12 |
| 512 | 2 | burst | A | text_gap_p95_ms | 0.008 | [-0.002, 0.019] | 12 |
| 512 | 2 | burst | A | image_ttft_p95_ms | 0.650 | [-1.799, 2.919] | 12 |
| 512 | 2 | burst | A | joint_30_55_goodput_per_s | 0.001 | [-0.049, 0.047] | 12 |
| 512 | 2 | burst | A | image_queue_wait_p95_ms | 1.267 | [-2.484, 4.234] | 10 |
| 512 | 2 | burst | A | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | burst | A | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | burst | U | text_max_gap_ms | -4.447 | [-11.282, 1.973] | 12 |
| 512 | 2 | burst | U | text_gap_p95_ms | -0.008 | [-0.020, 0.004] | 12 |
| 512 | 2 | burst | U | image_ttft_p95_ms | -0.834 | [-2.642, 0.730] | 12 |
| 512 | 2 | burst | U | joint_30_55_goodput_per_s | 0.019 | [-0.002, 0.044] | 12 |
| 512 | 2 | burst | U | image_queue_wait_p95_ms | -1.156 | [-3.640, 1.366] | 12 |
| 512 | 2 | burst | U | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | burst | U | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | uniform | F | text_max_gap_ms | -2.770 | [-5.952, 0.919] | 12 |
| 512 | 2 | uniform | F | text_gap_p95_ms | -0.044 | [-0.066, -0.024] | 12 |
| 512 | 2 | uniform | F | image_ttft_p95_ms | 0.074 | [-1.724, 2.208] | 12 |
| 512 | 2 | uniform | F | joint_30_55_goodput_per_s | 0.000 | [-0.001, 0.003] | 12 |
| 512 | 2 | uniform | F | image_queue_wait_p95_ms | 2.932 | [2.209, 3.655] | 2 |
| 512 | 2 | uniform | F | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | uniform | F | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | uniform | S | text_max_gap_ms | 2.460 | [-3.448, 8.479] | 12 |
| 512 | 2 | uniform | S | text_gap_p95_ms | -0.014 | [-0.036, 0.011] | 12 |
| 512 | 2 | uniform | S | image_ttft_p95_ms | 0.656 | [-1.561, 3.777] | 12 |
| 512 | 2 | uniform | S | joint_30_55_goodput_per_s | 0.001 | [-0.001, 0.003] | 12 |
| 512 | 2 | uniform | S | image_queue_wait_p95_ms | NA | [NA, NA] | 0 |
| 512 | 2 | uniform | S | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | uniform | S | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | uniform | A | text_max_gap_ms | 1.978 | [-3.325, 7.769] | 12 |
| 512 | 2 | uniform | A | text_gap_p95_ms | 0.010 | [-0.010, 0.030] | 12 |
| 512 | 2 | uniform | A | image_ttft_p95_ms | 0.952 | [-0.381, 2.133] | 12 |
| 512 | 2 | uniform | A | joint_30_55_goodput_per_s | -0.000 | [-0.003, 0.002] | 12 |
| 512 | 2 | uniform | A | image_queue_wait_p95_ms | -3.533 | [-3.533, -3.533] | 1 |
| 512 | 2 | uniform | A | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 2 | uniform | A | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | burst | F | text_max_gap_ms | -8.268 | [-12.912, -3.435] | 12 |
| 512 | 6 | burst | F | text_gap_p95_ms | 0.399 | [0.283, 0.509] | 12 |
| 512 | 6 | burst | F | image_ttft_p95_ms | 0.472 | [-1.309, 2.042] | 12 |
| 512 | 6 | burst | F | joint_30_55_goodput_per_s | -0.007 | [-0.021, 0.007] | 12 |
| 512 | 6 | burst | F | image_queue_wait_p95_ms | 1.897 | [-0.033, 4.049] | 11 |
| 512 | 6 | burst | F | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | burst | F | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | burst | S | text_max_gap_ms | 2.653 | [-2.181, 7.569] | 12 |
| 512 | 6 | burst | S | text_gap_p95_ms | 0.029 | [-0.009, 0.066] | 12 |
| 512 | 6 | burst | S | image_ttft_p95_ms | 1.464 | [0.183, 2.787] | 12 |
| 512 | 6 | burst | S | joint_30_55_goodput_per_s | -0.017 | [-0.031, -0.003] | 12 |
| 512 | 6 | burst | S | image_queue_wait_p95_ms | -1.001 | [-3.862, 1.738] | 12 |
| 512 | 6 | burst | S | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | burst | S | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | burst | A | text_max_gap_ms | 1.507 | [-4.941, 7.918] | 12 |
| 512 | 6 | burst | A | text_gap_p95_ms | 0.034 | [-0.041, 0.116] | 12 |
| 512 | 6 | burst | A | image_ttft_p95_ms | -0.395 | [-1.600, 0.880] | 12 |
| 512 | 6 | burst | A | joint_30_55_goodput_per_s | 0.001 | [-0.013, 0.014] | 12 |
| 512 | 6 | burst | A | image_queue_wait_p95_ms | 0.425 | [-2.106, 2.879] | 11 |
| 512 | 6 | burst | A | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | burst | A | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | burst | U | text_max_gap_ms | 2.947 | [-1.647, 7.193] | 12 |
| 512 | 6 | burst | U | text_gap_p95_ms | 0.016 | [-0.038, 0.073] | 12 |
| 512 | 6 | burst | U | image_ttft_p95_ms | -0.143 | [-2.240, 1.621] | 12 |
| 512 | 6 | burst | U | joint_30_55_goodput_per_s | -0.006 | [-0.023, 0.011] | 12 |
| 512 | 6 | burst | U | image_queue_wait_p95_ms | -0.407 | [-2.928, 2.093] | 12 |
| 512 | 6 | burst | U | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | burst | U | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | uniform | F | text_max_gap_ms | -5.856 | [-10.691, -1.410] | 12 |
| 512 | 6 | uniform | F | text_gap_p95_ms | 0.245 | [0.133, 0.360] | 12 |
| 512 | 6 | uniform | F | image_ttft_p95_ms | -0.432 | [-6.230, 5.341] | 12 |
| 512 | 6 | uniform | F | joint_30_55_goodput_per_s | -0.003 | [-0.008, 0.001] | 12 |
| 512 | 6 | uniform | F | image_queue_wait_p95_ms | -1.832 | [-5.936, 2.786] | 3 |
| 512 | 6 | uniform | F | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | uniform | F | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | uniform | S | text_max_gap_ms | 0.295 | [-3.584, 4.469] | 12 |
| 512 | 6 | uniform | S | text_gap_p95_ms | 0.032 | [0.010, 0.054] | 12 |
| 512 | 6 | uniform | S | image_ttft_p95_ms | 2.842 | [-2.042, 8.065] | 12 |
| 512 | 6 | uniform | S | joint_30_55_goodput_per_s | 0.000 | [-0.003, 0.004] | 12 |
| 512 | 6 | uniform | S | image_queue_wait_p95_ms | 1.776 | [-0.599, 4.152] | 2 |
| 512 | 6 | uniform | S | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | uniform | S | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | uniform | A | text_max_gap_ms | 2.358 | [-1.577, 6.367] | 12 |
| 512 | 6 | uniform | A | text_gap_p95_ms | 0.029 | [-0.004, 0.067] | 12 |
| 512 | 6 | uniform | A | image_ttft_p95_ms | 3.879 | [-1.293, 9.238] | 12 |
| 512 | 6 | uniform | A | joint_30_55_goodput_per_s | -0.004 | [-0.009, 0.001] | 12 |
| 512 | 6 | uniform | A | image_queue_wait_p95_ms | -3.206 | [-6.974, -0.499] | 3 |
| 512 | 6 | uniform | A | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 512 | 6 | uniform | A | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | burst | F | text_max_gap_ms | -21.461 | [-27.075, -14.526] | 12 |
| 768 | 2 | burst | F | text_gap_p95_ms | -0.028 | [-0.042, -0.015] | 12 |
| 768 | 2 | burst | F | image_ttft_p95_ms | 1.167 | [0.190, 2.064] | 12 |
| 768 | 2 | burst | F | joint_30_55_goodput_per_s | -0.002 | [-0.007, 0.003] | 12 |
| 768 | 2 | burst | F | image_queue_wait_p95_ms | 3.531 | [0.518, 6.881] | 12 |
| 768 | 2 | burst | F | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | burst | F | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | burst | S | text_max_gap_ms | -4.093 | [-10.536, 1.575] | 12 |
| 768 | 2 | burst | S | text_gap_p95_ms | 0.013 | [0.004, 0.022] | 12 |
| 768 | 2 | burst | S | image_ttft_p95_ms | 5.937 | [4.912, 6.928] | 12 |
| 768 | 2 | burst | S | joint_30_55_goodput_per_s | -0.007 | [-0.010, -0.004] | 12 |
| 768 | 2 | burst | S | image_queue_wait_p95_ms | 2.444 | [-1.763, 5.998] | 12 |
| 768 | 2 | burst | S | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | burst | S | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | burst | A | text_max_gap_ms | -4.813 | [-10.144, -0.159] | 12 |
| 768 | 2 | burst | A | text_gap_p95_ms | 0.016 | [0.007, 0.027] | 12 |
| 768 | 2 | burst | A | image_ttft_p95_ms | 6.036 | [4.802, 7.244] | 12 |
| 768 | 2 | burst | A | joint_30_55_goodput_per_s | -0.006 | [-0.010, -0.000] | 12 |
| 768 | 2 | burst | A | image_queue_wait_p95_ms | 1.865 | [-1.730, 5.486] | 11 |
| 768 | 2 | burst | A | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | burst | A | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | burst | U | text_max_gap_ms | -0.167 | [-2.772, 2.558] | 12 |
| 768 | 2 | burst | U | text_gap_p95_ms | 0.011 | [0.006, 0.016] | 12 |
| 768 | 2 | burst | U | image_ttft_p95_ms | 0.423 | [-0.315, 1.109] | 12 |
| 768 | 2 | burst | U | joint_30_55_goodput_per_s | 0.000 | [-0.004, 0.005] | 12 |
| 768 | 2 | burst | U | image_queue_wait_p95_ms | 2.572 | [0.365, 4.771] | 12 |
| 768 | 2 | burst | U | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | burst | U | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | uniform | F | text_max_gap_ms | -11.931 | [-18.222, -5.507] | 12 |
| 768 | 2 | uniform | F | text_gap_p95_ms | -0.028 | [-0.049, -0.007] | 12 |
| 768 | 2 | uniform | F | image_ttft_p95_ms | -1.809 | [-3.703, -0.097] | 12 |
| 768 | 2 | uniform | F | joint_30_55_goodput_per_s | 0.000 | [-0.001, 0.001] | 12 |
| 768 | 2 | uniform | F | image_queue_wait_p95_ms | 7.116 | [2.596, 11.635] | 2 |
| 768 | 2 | uniform | F | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | uniform | F | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | uniform | S | text_max_gap_ms | -7.817 | [-14.452, -0.895] | 12 |
| 768 | 2 | uniform | S | text_gap_p95_ms | -0.023 | [-0.042, -0.004] | 12 |
| 768 | 2 | uniform | S | image_ttft_p95_ms | 3.831 | [2.268, 5.398] | 12 |
| 768 | 2 | uniform | S | joint_30_55_goodput_per_s | -0.000 | [-0.001, 0.000] | 12 |
| 768 | 2 | uniform | S | image_queue_wait_p95_ms | 5.189 | [-4.686, 15.063] | 2 |
| 768 | 2 | uniform | S | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | uniform | S | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | uniform | A | text_max_gap_ms | -5.003 | [-12.202, 1.590] | 12 |
| 768 | 2 | uniform | A | text_gap_p95_ms | -0.014 | [-0.029, 0.003] | 12 |
| 768 | 2 | uniform | A | image_ttft_p95_ms | 5.222 | [3.713, 6.705] | 12 |
| 768 | 2 | uniform | A | joint_30_55_goodput_per_s | -0.000 | [-0.001, 0.000] | 12 |
| 768 | 2 | uniform | A | image_queue_wait_p95_ms | -3.935 | [-3.935, -3.935] | 1 |
| 768 | 2 | uniform | A | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 2 | uniform | A | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | burst | F | text_max_gap_ms | -13.611 | [-18.272, -8.237] | 12 |
| 768 | 6 | burst | F | text_gap_p95_ms | 0.417 | [0.279, 0.548] | 12 |
| 768 | 6 | burst | F | image_ttft_p95_ms | -1.848 | [-4.652, 0.874] | 12 |
| 768 | 6 | burst | F | joint_30_55_goodput_per_s | 0.002 | [0.000, 0.005] | 12 |
| 768 | 6 | burst | F | image_queue_wait_p95_ms | -2.662 | [-7.902, 2.274] | 10 |
| 768 | 6 | burst | F | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | burst | F | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | burst | S | text_max_gap_ms | -5.334 | [-9.457, -1.640] | 12 |
| 768 | 6 | burst | S | text_gap_p95_ms | 0.033 | [-0.086, 0.126] | 12 |
| 768 | 6 | burst | S | image_ttft_p95_ms | 3.911 | [1.795, 6.105] | 12 |
| 768 | 6 | burst | S | joint_30_55_goodput_per_s | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | burst | S | image_queue_wait_p95_ms | -6.055 | [-10.475, -2.184] | 8 |
| 768 | 6 | burst | S | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | burst | S | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | burst | A | text_max_gap_ms | -1.128 | [-6.514, 4.176] | 12 |
| 768 | 6 | burst | A | text_gap_p95_ms | 0.055 | [-0.045, 0.190] | 12 |
| 768 | 6 | burst | A | image_ttft_p95_ms | 2.136 | [-0.522, 4.815] | 12 |
| 768 | 6 | burst | A | joint_30_55_goodput_per_s | 0.002 | [0.000, 0.005] | 12 |
| 768 | 6 | burst | A | image_queue_wait_p95_ms | -2.230 | [-9.364, 4.457] | 9 |
| 768 | 6 | burst | A | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | burst | A | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | burst | U | text_max_gap_ms | -4.087 | [-8.002, 0.373] | 12 |
| 768 | 6 | burst | U | text_gap_p95_ms | -0.027 | [-0.080, 0.015] | 12 |
| 768 | 6 | burst | U | image_ttft_p95_ms | -1.711 | [-3.773, 0.519] | 12 |
| 768 | 6 | burst | U | joint_30_55_goodput_per_s | 0.002 | [0.000, 0.004] | 12 |
| 768 | 6 | burst | U | image_queue_wait_p95_ms | -3.792 | [-8.065, 0.566] | 10 |
| 768 | 6 | burst | U | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | burst | U | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | uniform | F | text_max_gap_ms | -6.867 | [-13.788, 0.168] | 12 |
| 768 | 6 | uniform | F | text_gap_p95_ms | 0.180 | [0.092, 0.271] | 12 |
| 768 | 6 | uniform | F | image_ttft_p95_ms | -3.842 | [-9.121, 0.782] | 12 |
| 768 | 6 | uniform | F | joint_30_55_goodput_per_s | 0.000 | [0.000, 0.001] | 12 |
| 768 | 6 | uniform | F | image_queue_wait_p95_ms | 2.743 | [2.743, 2.743] | 1 |
| 768 | 6 | uniform | F | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | uniform | F | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | uniform | S | text_max_gap_ms | -2.742 | [-7.735, 2.379] | 12 |
| 768 | 6 | uniform | S | text_gap_p95_ms | -0.015 | [-0.043, 0.011] | 12 |
| 768 | 6 | uniform | S | image_ttft_p95_ms | 5.050 | [1.241, 8.331] | 12 |
| 768 | 6 | uniform | S | joint_30_55_goodput_per_s | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | uniform | S | image_queue_wait_p95_ms | NA | [NA, NA] | 0 |
| 768 | 6 | uniform | S | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | uniform | S | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | uniform | A | text_max_gap_ms | -4.873 | [-10.823, 0.665] | 12 |
| 768 | 6 | uniform | A | text_gap_p95_ms | -0.013 | [-0.032, 0.004] | 12 |
| 768 | 6 | uniform | A | image_ttft_p95_ms | 3.168 | [-1.625, 7.247] | 12 |
| 768 | 6 | uniform | A | joint_30_55_goodput_per_s | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | uniform | A | image_queue_wait_p95_ms | NA | [NA, NA] | 0 |
| 768 | 6 | uniform | A | image_completion_rate | 0.000 | [0.000, 0.000] | 12 |
| 768 | 6 | uniform | A | text_completion_rate | 0.000 | [0.000, 0.000] | 12 |

## Trace 完整性

- 首 token server/API/client 顺序通过：46656/46656 个已关联图像请求。
- 首 token ID 一致：46656 个请求。
- server async writer 写入 613696168 行；丢弃 0 行；写错误 0。
- 被补跑前的失败 attempt 数：0，保留在 JSON 报告。

## A 策略状态分区

见 formal-state-partition.json；按 D slack 符号、I slack 符号、E/P 比值中位数分箱，并给出选择、实际拆步与联合 SLO 结果。

## 限制

- Trace seed is the independent unit for paired bootstrap intervals; requests and tokens are not treated as independent samples.
- Joint SLO goodput is counted per measurement image: image TTFT must meet the image deadline and every configured text lane must have complete interval coverage with its overlapping maximum token gap within the text deadline.
- Formal first-token engine-ready minus P-step can benchmark E+P in unsplit arms or P after E in split arms; formal mode does not collect CUDA-event visual durations, so E itself is not isolated on holdout.
- The formal visual-forward CPU elapsed time is a dispatch-side diagnostic proxy and is not interpreted as GPU execution latency.
- Missing client records and async writer metadata are reported per run; raw attempt evidence remains authoritative.
