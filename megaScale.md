# MegaScale
## Objective
a production system for training LLMs at the scale of more than 10,000 GPUs. based on Megatron-LM([NV open source on Github](https://github.com/NVIDIA/Megatron-LM), [paper DOC](https://arxiv.org/pdf/1909.08053.pdf))
## Challenge
- **efficiencty**
    - heavily communication
    - GPU memory consumption
    - operator optimization
    - data preprocessing

- **stability**
	- failures
	- stragglers

## Performance
- baseline: megatron-LM      
- PTB: parallel transformer block
- 55.2% Model FLOPs Utilization (MFU) when training a 175B LLM model on 12,288 GPUs, improving the MFU by 1.34× compared to Megatron-LM.
MFU: the ratio of the observed throughput to the theoretical maximum throughput assuming 100% of peak FLOPs 
parameter: 175B, sequence length 2048, vocabulary size 64K, 300B token, pipeline-parallel 6 interleave
![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/Screenshot%202024-03-12%20175628.png)
![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/per175.png)  
<br/>


**MFU has near-linear scalability as GPUs increasing**
![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/per530.png)
<br/>


**Contribution distribution:**
![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/percomp.png)
<br/>

## Principles
- **algorithm-system co-design(full-stack)**, see solutions
- **in-depth observability**
    - monitoring + visualization + multidimensional view of system performance
    - fault localization and recovery training framework
    - performance analysis tool: record fine grained CUDA events and generate system-wide heat-map and timeline trace from a distributed view
    - 3D parallel training visualization tool: show data dependencies between ranks
  
## Solutions <a name="solutions"></a>
### Network performance tuning
- **topology**
  - 3-layer CLOS (>10KGPU) production environment
  - Tomahawk-4 64*400G
  - downlink:uplink BW 1:1 (port 32:32)
  - ToR: breakout to 2*200G+AOC cable (can connct 64 NICs)
  - Node: 8 200G NICs rail-optimize, Ampere GPU
- **reduce ECMP hash conflict**
  - Schedule task on nodes under the same ToR
  - TP=8 and constrain in one node
  - CC: Swift+DCQCN+RTT, rapid response of ECN, reduce PFC
- **Retransmit timeout tuning**
  - NIC: adap_retrans feature
  - NCCL: retransmit timer+retry count, fast recovery on link flap
### Algorithm optimization
- **Parallel transformer block(PTB)**: enable Attention/MLP parallel to reduce computation time
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/algoOpt.png)
- **Sliding window attention(SWA)**: 
  fixed-size window on each token instead of sequence length to reduce computation complexity from O(s*s) to O(s*w) 
  without accuracy compromised because of stacking layers of windowed attention
- **LAMB optimizer**
Layer-wise adaptive moments optimizer for batch training
increase batch size and adapt layer-wise learning rate to reduce training time and help model converge more quickly
experiments show scale batch size to 4* without loss 
combined with pipeline interleave can reduces 87.5 of pipeline bubbles
### Communication Overlapping in 3D Parallelism
tp: single node, dp > pp: inter-node
- **Overlapping in dp**
  all-gather---> DP ---> reduce-scatter
  - middle communication hide on a module chunk basis
  - 1st all-gather use prefetch to overlap with data loading
  - high priority communication
<br/>

- **overlapping in pp**
  ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/opp.png)
  - warm-up + cool-down: decouple Send and Receive to, enable Send to overlap with computation
  - Steady: independent of adjacent communication
    - FWD: send step j BWD result to stage i-1, and receive step j+1 BWD data from stage i
    - BWD: send step j FWD result to stage i+1, and receive step j+1 FWD data from stage i-1
<br/>

- **overlapping in tp/sp**
  ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/otp.png)
  LayerNorm + Dropout: sequence parallelism
    - fuse all-gather with the ColParaLinear, reduce-scatter with the RowParaLinear
    - break GEMM kernel into small chunks and pipeline execution with communication 
            
### Efficient operators
already have GEMM in baseline
Attention: FlashAttention-2, improves work partitioning between thread blocks and warps
LayerNorm+ GeLU: fusing fine-grained kernels in previous implementations

### Fault Tolerance
![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/ft.png)
- Data collection: Executor(pre node)      
  - ip, pod, hardware info, training process status      
  - log(stdout/stderr), aggregate,filter,analysis      
  - real-time diag when spec err/warn occur      
  - RDMA traffic metrics for network utilization and efficiency      
  - ms-level monitoring: congestion, speed  
- Diagnostic Tests: lightweight(focus HD&SF fault)      
  - Intra-host network tests:       	 
    - loopback test: RNICs to host(full-mesh) to detect link-specific BW/PCI
    - RNIC to RNIC: test conn, BW, routing      
  - NCCL test: alltoall with node, all-reduce with neighbor
- Fast checkpoint and recovery      
  - Stage1: GPU worker writes on-chip states to host memory      
  - Stage2: daemon process async trans host mem to HDFS            
  - Recovery: latest checkpoint      
    - a worker in a group read state from HDFS, broadcast to other GPU to save BW of HDFS
- result
  - Cover 90% exceptions such as CUDA error and segmentation fault.
  - detect + diagnostic tests < 10min
  - resume training job again < 15min
### others
- Asynchronous data preprocessing while synchronizing gradients at end of training step
- 1 data loader to share mem, 2-layer tree-based copy to each GPU memory with in the node when using tp
- torch.distributed profiling: replace blocking TCPStore with non-blocking Redis + minimize the need of global barrier when NCCL communication groups initialize, decrease the init time from 1047s to 5s on 2048 GPUs, and achieve 30s on 10K+ GPUs
## Experience
- stragglers
    tool: CUDA event timers
    Phenomena: specific hosts took 10%+ time across different jobs
    solution: isolate and remove these hosts from cluster
- MFU decreasing
    tool: CUDA event timers
    Phenomena: MFU gradually decrease as training progresses
    solution: code error cause random fluctuation of launch time of reduce-scatter on some ranks
- network interface flapping
    Phenomena: network interface goes down and up in several seconds
    solution:
    - set NCCL timeout threshold to a larger value;
    - lower level quality control over NIC/Switch signal strength, cable quality.
 
## Trouble shooting
Performance Diagnosis with CUDA Event Monitor
3D Parallel Training Visualization

## background
- **Data parallelism**
  并行的多个GPU前向计算全部完成之后，从最后一层逐层反向传播，每个GPU先计算本地梯度，然后ReduceScatter将同一梯度的数据聚合到一个GPU，由各个GPU计算根据全局梯度+Adam算法更新自己负责计算的那部分参数，计算完成之后，AllGather将更新好的参数同步到每个GPU，最终完成本层反向计算。
  注意：DP做前向计算时，因为每层有全部参数，不依赖与其他GPU，可以单独计算。DP做后向计算梯度时，下一层的梯度计算不依赖于上一层的梯度聚合结果，也可以单独进行，可以做到通信和计算重叠。(每层计算梯度时不仅需要后一层计算好的值，还需要前向计算过程中的中间值)
  Zero-2：每个GPU维护 所有的参数，自己负责那部分的梯度+Adam中间变量(精度32，12*F)
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/dp.png)
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/zero.png)

    劣势：
    1. 受限于memory的大小，大模型不可能放到单个GPU上
    2. dp并行数最大为batchsize，而batchsize不能无限扩大
    3. 当每个GPU的batch太小时，无法充分利用GPU的计算能力
- **Pipeline parallelism**: micro batch + vitual stage
    [pipeline compare](https://zhuanlan.zhihu.com/p/432969288)
    Gpipe->pipedream->1f1b
    每个Worker中类似TP再切分virtual stage，经历3个阶段：
    - warm-up：forward
    - steady：1F1B
    - cool-down: backward
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/Screenshot%202024-03-15%20093922.png)
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/pp.png)
- **Tensor parallelism**: intra-node
    f：duplicate    g: sum
    heavy communication, need allReduce, block
    劣势：
    通信量大，通常只能利用单个Node内的NVlink传输，无法跨Node并行
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/tp.png)
- **sequence parallelism**
    当序列长度特别长时，单个GPU无法内存无法容纳，适用
    对于Attention阶段，每个GPU上包含完整QKV参数，将序列分段后分别将一部分Token序列发给不同的GPU，各个GPU分别计算自己收到的这些Token的QKV值，通过N-1次的key传递，依次得到$Q_i$与不同$K_i$的乘积，进而得到$Q_iK^T$的值(矩阵维度$\frac{s}{n}*s$)，之后再通过N-1次的value传递，最终得到$Q_iK^TV$的值。
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/sp1.png)
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/sp2.png)
- **3D parallel**
  ![3D](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png)
- **transformer**
  [blog](https://www.cnblogs.com/rossiXYZ/p/15840803.html)
  ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/trans.png)
  ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/transformerBlock.png)
## Relative papers
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)  
[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)  
[Efficient large-scale language model training on gpu clusters using megatron-lm](https://arxiv.org/pdf/2104.04473.pdf)  
[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf)  
[Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/pdf/2105.13120.pdf)  
  