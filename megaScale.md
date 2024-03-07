# MegaScale
## Objective
a production system for training LLMs at the scale of more than 10,000 GPUs. based on Megatron-LM([NV open source on Github](https://github.com/NVIDIA/Megatron-LM), [paper](https://arxiv.org/abs/1909.08053))
## Challenge
- efficiencty
    - heavily communicate
    - operator optimization
    - data preprocessing
    - GPU memory consumption
- stability
	- failures
	- stragglers
## Principles
- algorithm-system co-design(full-stack), see [solutions]
- in-depth observability
    - monitoring + visualization + multidimensional view of system performance
    - fault localization and recovery training framework
    - performance analysis tool: record fine grained CUDA events and generate system-wide heat-map and timeline trace from a distributed view
    - 3D parallel training visualization tool: show data dependencies between ranks
  
## Performance
![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/per175.png)
- baseline: megatron-LM      PTB: parallel transformer block
- 55.2% Model FLOPs Utilization (MFU) when training a 175B LLM model on 12,288 GPUs, improving the MFU by 1.34× compared to Megatron-LM.
MFU: the ratio of the observed throughput to the theoretical maximum throughput assuming 100% of peak FLOPs 
parameter: 175B, sequence length 2048, vocabulary size 64K, 300B token, pipeline-parallel 6 interleave

![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/per530.png)
MFU has near-linear scalability as GPUs increasing

![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/percomp.png)

## Solutions <a name="solutions"></a>
### Algorithm optimization
- Parallel transformer block(PTB): enable Attention / MLP parallel to reduce computation time
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/algoOpt.png)
- Sliding window attention(SWA): 
  fixed-size window on each token instead of sequence length to reduce computation complexity from O(s*s) to O(s*w) 
  without accuracy compromised because of stacking layers of windowed attention
- LAMB optimizer
Layer-wise adaptive moments optimizer for batch training
increase batch size and adapt layer-wise learning rate to reduce training time and help model converge more quickly
experiments show scale batch size to 4* without loss + pipeline interleave
### Communication Overlapping in 3D Parallelism
tp: single node, dp > pp: inter-node
- Overlapping in dp
    module chunk basis, prefetch all-reduce with data loading, high priority communication

- overlapping in pp
  ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/opp.png)
        
- overlapping in tp/sp
  ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/otp.png)
            
### Efficient operators
    Attention: FlashAttention-2(GEMM in baseline)
    LayerNorm+GeLU: fusing kernels with previous implementations
### Network performance tuning
- topology
        3-layer CLOS (>10KGPU) production environment
        Tomahawk-4 64*400G
        downlink:uplink BW 1:1 (port 32:32)
        ToR: breakout to 2*200G+AOC cable (can connct 64 NICs)
        Node: 8 200G NICs rail-optimize, Ampere GPU
- reduce ECMP hash conflict
        Schedule task on nodes under the same ToR
        TP=8 and constrain in one node
        CC: Swift+DCQCN+RTT, rapid response of ECN, reduce PFC
- Retransmit timeout tuning
        NIC: adap_retrans feature
        NCCL: retransmit timer+retry count, fast recovery on link flap
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
    - Asynchronous data preprocessing
    - 1 data load to share mem, 2-layer tree-based copy in node
    - replace blocking + global barrier when init
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
- Data parallelism
  并行的多个GPU前向计算全部完成之后，从最后一层逐层反向传播，ReduceScatter+AllGather，每层计算梯度时不仅需要后一层计算好的值，还需要前向计算过程中的中间值，计算完成之后，继续ReduceScatter+AllGather
  Zero-2：每个GPU维护 所有的参数，自己负责那部分的梯度+中间变量(精度32，12*F)
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/dp.png)
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/zero.png)
- Pipeline parallelism: micro batch + vitual stage
    每个Worker中类似TP再切分virtual stage，经历3个阶段：
    - warm-up：forward
    - steady：1F1B
    - cool-down: backward
    ？每个micor-batch结束都进行后向传播吗？
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/pp.png)
- Tensor parallelism: intra-node
    f：duplicate    g: sum
    heavy communication, need allReduce, block
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/tp.png)
- sequence parallelism
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/sp1.png)
    ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/sp2.png)
- transformer
  ![](https://raw.githubusercontent.com/JaneLiCC/testDemo/main/images/trans.png)