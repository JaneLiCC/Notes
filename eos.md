[Nvidia EOS](https://github.com/mlcommons/training_results_v3.1/tree/main/NVIDIA)

### GPT-3 175B
| Name | Platform | GPUs | TP | PP | DP | Batch size |1B Training Time(minutes) | Aggregate PFlops/s | Throughout tokens/s | Iteration time/Batch(seconds) |
| ---- | ---- | ---- | ----| ---- | ---- | ---- | ---- | ---- |---- |---- |
| MegaScale | A100 +   Tomahawk4 200G| 12288(1536 nodes) | 8 | 8 |192 |6144 |8.4(300B 1.75days) | 2166.3| 1984k| 6.34|
| Nvidia EOS| Xeon 3.8GHz + H100 + IB NDR 400G|10752(1344 Nodes) | 4 | 8 | 336 |2688 | 3.2(1211105280, 3.9minutes)|7215.1|5155k|1.07|

[raw result](https://raw.githubusercontent.com/mlcommons/training_results_v3.1/main/NVIDIA/results/eos-dfw_n1344_ngc23.09_nemo/gpt3/result_1.txt)

:::MLLOG {"namespace": "", "time_ms": 1696565401886, "event_type": "POINT_IN_TIME", "key": "tracked_stats", "value": {"train_epoch_timing": 234.93810319900513}, "metadata": {"file": "/workspace/llm/custom_callbacks.py", "lineno": 277, "step": 1211105280}}
:::MLLOG {"namespace": "", "time_ms": 1696565401886, "event_type": "POINT_IN_TIME", "key": "tracked_stats", "value": {"throughput": 5154997.267404211}, "metadata": {"file": "/workspace/llm/custom_callbacks.py", "lineno": 284, "step": 1211105280}}

### Traffic size
- B: global batch size, the number of training examples in one forward/backward pass
- b: micro batch size
- n: GPU numbers, n=p*t*d
- l: layers 96
- h: hidden size, vector length, 12288
- a: attention-heads 96
- s: sequence length 2048 
- V: vacabulary size 51200
- p: pp
- t: tp
- d: dp
- T: all training tokens
- X: average FLOPs per GPU, $\approx peak FLOPs * utilization$ 

[GPT parameters](https://docs.google.com/spreadsheets/d/10Y4GLc28UgeKr2qSYEZuRqELn1D-w5EiQpAGg-_y4Xg/edit#gid=899002403)
$$模型参数P = 12lh^2(1+\frac{13}{12h}+\frac{V+s}{12lh})$$
$$单次迭代iteration(1个batch，B*s个token)浮点运算次数F = 96Bslh^2(1+\frac{s}{6h}+\frac{V}{16lh}) $$
$$单次训练epoch(全部Token)浮点运算次数F = 96Tlh^2(1+\frac{s}{6h}+\frac{V}{16lh}) $$
$$估计训练用时 = \frac{8TP}{nX}$$
$$迭代次数=\frac{T}{Bs}$$
$模型内存占用大小(without Zero) = 16P$
$模型内存占用大小(with Zero-2)=(2+\frac{14}{d})P$

| parallelism | size | Communication times/layer| Data | Level |
| ---- | ---- | ---- | ---- |----|
| DP | O(h * h) | FW:0 BW: 2*AllReduce| BW:gradients |GB-10GB, 20GB/s单卡AllReduce带宽|
| PP | O(b * s * h) | P2P PP层间 |FW: calc results BW:gradients|MB|
| TP | O(B * s * h) | FW: 2*AllReduce BW:2*AllReduce| FW: calc results BW:gradients |10G-100GB|

![3D parallem](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/Screenshot%202024-04-09%20142106.png)
