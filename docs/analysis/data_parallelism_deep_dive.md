# Data Parallelism 深度解析

> 通俗易懂地讲解 Axolotl 中的数据并行实现，延续"搬桌子"比喻

---

## 目录

1. [什么是 Data Parallelism？](#1-什么是-data-parallelism)
2. [为什么需要 Data Parallelism？](#2-为什么需要-data-parallelism)
3. [DP 的工作原理](#3-dp-的工作原理)
4. [FSDP vs DDP：两种 DP 策略](#4-fsdp-vs-ddp两种-dp-策略)
5. [Axolotl 的 ND 并行架构](#5-axolotl-的-nd-并行架构)
6. [配置与实战](#6-配置与实战)
7. [常见问题排查](#7-常见问题排查)

---

## 1. 什么是 Data Parallelism？

### 1.1 一句话总结

**Data Parallelism (DP)** 是指**多个人（GPU）同时搬相同的桌子（模型），但每个人搬的是不同批次的货物（数据）**。

### 1.2 继续"搬桌子"比喻

回顾我们之前的比喻系统：

| 并行类型 | 搬桌子比喻 | 英文名称 |
|---------|-----------|---------|
| **Tensor Parallelism** | 多个人搬**同一张桌子的不同部分**（按宽度切） | TP |
| **Context Parallelism** | 多个人搬**同一张超长桌子的不同部分**（按长度切） | CP |
| **Data Parallelism** | 多个人**各自搬一张完整的桌子**，但搬的货物不同 | DP |

#### 形象类比

假设你要用 4 个人（4 个 GPU）搬家：

**TP 方式**（横向切桌子）：
```
桌子太宽，需要 4 个人一起抬一张桌子
人 A：负责桌子左 1/4
人 B：负责桌子左中 1/4
人 C：负责桌子右中 1/4
人 D：负责桌子右 1/4

→ 优点：可以搬特别宽的桌子
→ 缺点：4 个人必须同步协调，速度受限
```

**CP 方式**（纵向切桌子）：
```
桌子太长，需要 4 个人一起抬一张桌子
人 A：负责桌子前 1/4
人 B：负责桌子前中 1/4
人 C：负责桌子后中 1/4
人 D：负责桌子后 1/4

→ 优点：可以搬特别长的桌子
→ 缺点：4 个人必须同步协调，速度受限
```

**DP 方式**（数据并行）：
```
每个人搬一张完整的桌子，但搬的货物不同

人 A：搬完整的桌子 + 货物批次 1（沙发、椅子）
人 B：搬完整的桌子 + 货物批次 2（书籍、文件）
人 C：搬完整的桌子 + 货物批次 3（衣物、鞋子）
人 D：搬完整的桌子 + 货物批次 4（厨具、餐具）

→ 优点：4 个人可以并行工作，速度翻 4 倍！
→ 缺点：每个人都要有一张完整的桌子（显存需求高）
```

### 1.3 技术定义

**Data Parallelism** 是一种**数据级并行策略**：

- **模型复制**：每个 GPU 持有完整的模型副本
- **数据切分**：训练数据被切分成多个批次，每个 GPU 处理不同的数据批次
- **梯度同步**：每个 GPU 独立计算梯度，然后通过 AllReduce 操作同步梯度
- **参数更新**：所有 GPU 使用同步后的平均梯度更新模型参数

核心思想：**相同的模型，不同的数据**

---

## 2. 为什么需要 Data Parallelism？

### 2.1 解决的核心问题

#### 问题 1：训练速度慢

单个 GPU 训练大模型非常耗时：

```python
# 单 GPU 训练 Llama-7B
序列长度：2048
Batch size：8
单个样本耗时：~500ms

总训练时间（100K 样本）：
100,000 / 8 × 0.5s = 6,250 秒 ≈ 1.7 小时
```

使用 8 个 GPU 做 Data Parallelism：

```python
# 8-GPU Data Parallelism
每个 GPU 处理：100,000 / 8 = 12,500 样本
理论加速比：8×

总训练时间：
12,500 / 8 × 0.5s = 781 秒 ≈ 13 分钟  # ← 快了 ~8 倍！
```

#### 问题 2：有效 Batch Size 太小

大的 batch size 通常能带来更好的训练稳定性和收敛速度，但单 GPU 显存有限：

```python
# 单 GPU：A100 80GB
模型：Llama-7B (~14GB)
序列长度：2048
单 GPU 最大 batch size：~8

# 8-GPU Data Parallelism
每个 GPU：batch size = 8
有效 batch size = 8 × 8 = 64  # ← 大 batch 训练！
```

### 2.2 DP vs TP vs CP 对比

| 维度 | Data Parallelism | Tensor Parallelism | Context Parallelism |
|------|------------------|-------------------|---------------------|
| **解决问题** | 训练慢、batch 小 | 模型太大（显存不够） | 序列太长（显存不够） |
| **切分对象** | 数据（Data） | 模型参数（Tensor） | 输入序列（Context） |
| **模型副本** | 每个 GPU 一份完整模型 | 每个 GPU 一部分模型 | 每个 GPU 一份完整模型 |
| **通信模式** | AllReduce（梯度） | AllGather/ReduceScatter（激活） | Ring（K/V） |
| **加速效果** | 接近线性（理想 N×） | 有通信开销（<N×） | 有通信开销（<N×） |
| **显存节省** | 无（FSDP 除外） | 显著（~1/N） | 显著（~1/N） |
| **适用场景** | 模型能放进单 GPU | 模型太大 | 超长上下文 |

### 2.3 什么时候用 DP？

#### ✅ 适合使用 DP 的场景

1. **模型能放进单个 GPU 显存**
   - 小型模型（<7B）
   - 或使用了量化（4-bit/8-bit）

2. **想要提高训练速度**
   - 有多个 GPU 可用
   - 数据集很大，需要加速训练

3. **想要使用更大的 Batch Size**
   - 提高训练稳定性
   - 更好的梯度估计

#### ❌ 不适合单纯使用 DP 的场景

1. **模型太大**
   - 70B+ 模型单 GPU 放不下
   - → 需要 TP 或 FSDP

2. **序列太长**
   - 超长上下文（>16K）
   - → 需要 CP

3. **GPU 间通信带宽低**
   - 跨节点且网络慢
   - DP 的梯度同步开销会很大

---

## 3. DP 的工作原理

### 3.1 标准 Data Parallelism 流程

#### 完整的训练迭代流程

```
┌─────────────────────────────────────────────────────────────┐
│ 第 0 步：初始化                                              │
├─────────────────────────────────────────────────────────────┤
│ GPU 0: 完整模型 θ₀                                          │
│ GPU 1: 完整模型 θ₀  ← 从 GPU 0 广播（Broadcast）             │
│ GPU 2: 完整模型 θ₀  ← 从 GPU 0 广播                         │
│ GPU 3: 完整模型 θ₀  ← 从 GPU 0 广播                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 第 1 步：数据分发（Data Distribution）                       │
├─────────────────────────────────────────────────────────────┤
│ 完整 Batch (size=32) → 切分成 4 份                          │
│                                                             │
│ GPU 0: Batch[0:8]   → Data 1                               │
│ GPU 1: Batch[8:16]  → Data 2                               │
│ GPU 2: Batch[16:24] → Data 3                               │
│ GPU 3: Batch[24:32] → Data 4                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 第 2 步：前向传播（Forward Pass）                            │
├─────────────────────────────────────────────────────────────┤
│ GPU 0: Loss₀ = f(θ, Data 1)  ← 并行执行，互不影响            │
│ GPU 1: Loss₁ = f(θ, Data 2)  ←                             │
│ GPU 2: Loss₂ = f(θ, Data 3)  ←                             │
│ GPU 3: Loss₃ = f(θ, Data 4)  ←                             │
│                                                             │
│ ✅ 没有通信，完全并行！                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 第 3 步：反向传播（Backward Pass）                           │
├─────────────────────────────────────────────────────────────┤
│ GPU 0: 梯度 g₀ = ∂Loss₀/∂θ                                 │
│ GPU 1: 梯度 g₁ = ∂Loss₁/∂θ                                 │
│ GPU 2: 梯度 g₂ = ∂Loss₂/∂θ                                 │
│ GPU 3: 梯度 g₃ = ∂Loss₃/∂θ                                 │
│                                                             │
│ ✅ 没有通信，完全并行！                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 第 4 步：梯度同步（Gradient AllReduce）← 关键！              │
├─────────────────────────────────────────────────────────────┤
│ AllReduce 操作：计算所有 GPU 梯度的平均值                    │
│                                                             │
│ g_avg = (g₀ + g₁ + g₂ + g₃) / 4                            │
│                                                             │
│ GPU 0: g_avg  ←┐                                           │
│ GPU 1: g_avg  ←├─ AllReduce (Ring or Tree)                 │
│ GPU 2: g_avg  ←│                                           │
│ GPU 3: g_avg  ←┘                                           │
│                                                             │
│ ⚠️ 有通信！这是 DP 的主要开销                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 第 5 步：参数更新（Parameter Update）                        │
├─────────────────────────────────────────────────────────────┤
│ GPU 0: θ₀ ← θ₀ - lr × g_avg                                │
│ GPU 1: θ₁ ← θ₁ - lr × g_avg                                │
│ GPU 2: θ₂ ← θ₂ - lr × g_avg                                │
│ GPU 3: θ₃ ← θ₃ - lr × g_avg                                │
│                                                             │
│ ✅ 所有 GPU 使用相同的梯度，参数保持同步！                     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 AllReduce 通信详解

#### 什么是 AllReduce？

AllReduce 是一个分布式计算中的集合通信操作：

- **输入**：每个 GPU 有一个本地梯度张量 `g_i`
- **操作**：对所有梯度求和（或求平均）
- **输出**：每个 GPU 都得到相同的聚合结果 `g_avg`

#### Ring-AllReduce 算法（高效）

NCCL（NVIDIA Collective Communications Library）使用 Ring-AllReduce：

```
示例：4 个 GPU，每个有梯度向量 [a, b, c, d]

GPU 0: [a₀, b₀, c₀, d₀]
GPU 1: [a₁, b₁, c₁, d₁]
GPU 2: [a₂, b₂, c₂, d₂]
GPU 3: [a₃, b₃, c₃, d₃]

Ring 拓扑：GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0

步骤 1（Reduce-Scatter）：
  每个 GPU 累积一部分梯度
  Round 1: GPU 0 → GPU 1 (传递 a), GPU 1 → GPU 2 (传递 b), ...
  Round 2: GPU 0 → GPU 1 (传递 d), ...
  ...
  结果：每个 GPU 有一部分完整的梯度和

步骤 2（AllGather）：
  将每部分完整梯度传递给所有 GPU
  Round 1: GPU 0 → GPU 1, ...
  ...
  结果：所有 GPU 都有完整的 g_avg

时间复杂度：O(2(N-1)/N × M / B)
  N = GPU 数量
  M = 梯度大小
  B = 带宽

关键：通信量与 GPU 数量几乎无关！
```

#### 通信开销分析

```python
# Llama-7B 模型参数量
总参数：7B
每个参数：2 字节（bf16）
梯度大小：7B × 2 = 14 GB

# AllReduce 通信量（Ring 算法）
理论通信量 ≈ 2 × 14GB = 28 GB

# 通信时间（NVLink，300 GB/s）
通信时间 = 28 GB / 300 GB/s ≈ 93 ms

# 计算时间（A100）
前向 + 反向 ≈ 500 ms

# 通信占比
通信占比 = 93 / 500 ≈ 19%

# 理论加速比（8 GPU）
理想加速 = 8×
实际加速 ≈ 6.7×（考虑通信）
```

### 3.3 代码示例

#### PyTorch DDP 基本用法

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. 初始化进程组
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 2. 创建模型并移到对应 GPU
model = MyModel().to(local_rank)

# 3. 包装成 DDP 模型
model = DDP(model, device_ids=[local_rank])

# 4. 创建分布式数据加载器
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True,
)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    sampler=train_sampler,
    num_workers=4,
)

# 5. 训练循环
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)  # ← 重要！确保每个 epoch 数据不同

    for batch in train_loader:
        # 每个 GPU 得到不同的 batch
        inputs, labels = batch
        inputs = inputs.to(local_rank)
        labels = labels.to(local_rank)

        # 前向传播（并行）
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播（并行）
        optimizer.zero_grad()
        loss.backward()

        # ← 梯度自动同步（DDP 在 backward 后自动执行 AllReduce）

        # 参数更新（并行）
        optimizer.step()
```

#### 关键点

1. **DistributedSampler**：确保每个 GPU 获得不同的数据切片
2. **DDP 自动梯度同步**：在 `loss.backward()` 结束时自动执行 AllReduce
3. **set_epoch()**：每个 epoch 需要设置，确保数据随机性

---

## 4. FSDP vs DDP：两种 DP 策略

### 4.1 DDP (Distributed Data Parallelism)

#### 基本原理

```
传统 DDP：每个 GPU 持有完整的模型副本

GPU 0: [完整模型 θ] + 梯度 g₀
GPU 1: [完整模型 θ] + 梯度 g₁
GPU 2: [完整模型 θ] + 梯度 g₂
GPU 3: [完整模型 θ] + 梯度 g₃

显存占用（每个 GPU）：
  - 模型参数：100%
  - 优化器状态：100%
  - 梯度：100%

总计：每个 GPU 需要容纳完整模型 + 优化器 + 梯度
```

#### 搬桌子类比

```
DDP = 每个人都有一张完整的桌子

人 A：[完整的桌子] + 搬货物 1
人 B：[完整的桌子] + 搬货物 2
人 C：[完整的桌子] + 搬货物 3
人 D：[完整的桌子] + 搬货物 4

优点：
  - 简单直接，容易理解
  - 每个人独立工作，互不干扰
  - 只在最后同步一次（AllReduce 梯度）

缺点：
  - 每个人都要有完整的桌子（显存占用高）
  - 桌子太大时，单个人搬不动
```

### 4.2 FSDP (Fully Sharded Data Parallelism)

#### 基本原理

FSDP 基于 **ZeRO (Zero Redundancy Optimizer)** 论文：

```
FSDP：切分模型参数、梯度、优化器状态到不同 GPU

GPU 0: [模型参数 1/4] + [优化器状态 1/4] + [梯度 1/4] + Data 0
GPU 1: [模型参数 1/4] + [优化器状态 1/4] + [梯度 1/4] + Data 1
GPU 2: [模型参数 1/4] + [优化器状态 1/4] + [梯度 1/4] + Data 2
GPU 3: [模型参数 1/4] + [优化器状态 1/4] + [梯度 1/4] + Data 3

关键思想：
  - 每个 GPU 只持有 1/N 的模型参数
  - 前向传播时：AllGather 完整参数 → 计算 → 释放
  - 反向传播时：AllGather 完整参数 → 计算梯度 → ReduceScatter
  - 参数更新：只更新自己持有的 1/N 参数

显存节省：
  - 模型参数：1/N
  - 优化器状态：1/N
  - 梯度：1/N

总计：每个 GPU 只需 ~1/N 的显存！
```

#### 搬桌子类比

```
FSDP = 每个人只持有 1/4 的桌子，但搬货时临时组装完整桌子

人 A：[持有桌腿 A] + 搬货物 1
人 B：[持有桌腿 B] + 搬货物 2
人 C：[持有桌腿 C] + 搬货物 3
人 D：[持有桌腿 D] + 搬货物 4

搬货物时：
  1. 临时组装：4 个人把桌腿拿出来，组装成完整桌子
  2. 搬运：人 A 用完整桌子搬货物 1
  3. 拆解：搬完后拆解桌子，每个人拿回自己的桌腿
  4. 重复：人 B、C、D 也重复这个过程

优点：
  - 每个人只需要存储 1/4 的桌腿（显存节省）
  - 可以搬很大的桌子（支持更大模型）

缺点：
  - 每次搬货都要组装 + 拆解（通信开销）
  - 需要 4 个人协调（更复杂）
```

#### FSDP 执行流程详解

```python
# FSDP Forward Pass（前向传播）

对于每一层 Layer：

  # 1. AllGather：收集完整参数
  layer.params = AllGather(layer.shard_params)
  # GPU 0: 收集 [shard_0, shard_1, shard_2, shard_3] → 完整参数
  # GPU 1: 收集 [shard_0, shard_1, shard_2, shard_3] → 完整参数
  # ...

  # 2. Forward：使用完整参数计算
  output = layer.forward(input)

  # 3. Release：释放不属于自己的参数（节省显存）
  layer.params = None  # 只保留 layer.shard_params
  # GPU 0: 只保留 shard_0，释放其他
  # GPU 1: 只保留 shard_1，释放其他
  # ...

# FSDP Backward Pass（反向传播）

对于每一层 Layer（反向顺序）：

  # 1. AllGather：再次收集完整参数（反向传播需要）
  layer.params = AllGather(layer.shard_params)

  # 2. Backward：计算梯度
  grad = layer.backward(grad_output)

  # 3. ReduceScatter：聚合梯度并分片
  layer.shard_grad = ReduceScatter(grad)
  # GPU 0: 得到 shard_0 的梯度和
  # GPU 1: 得到 shard_1 的梯度和
  # ...

  # 4. Release：释放完整参数
  layer.params = None

# Parameter Update（参数更新）

optimizer.step()
# 每个 GPU 只更新自己持有的参数切片
# GPU 0: 更新 shard_0
# GPU 1: 更新 shard_1
# ...
```

### 4.3 FSDP vs DDP 对比表

| 维度 | DDP | FSDP |
|------|-----|------|
| **显存占用** | 100%（每个 GPU） | ~1/N（N 是 GPU 数） |
| **通信量** | 梯度 AllReduce（1×） | AllGather(2×) + ReduceScatter(1×) = 3× |
| **通信频率** | 每个 iteration 1 次 | 每层 2 次（前向 + 反向） |
| **计算效率** | 高（无额外开销） | 中（AllGather 开销） |
| **支持模型大小** | 受单 GPU 显存限制 | 可扩展到极大模型 |
| **实现复杂度** | 简单 | 复杂 |
| **适用场景** | 模型能放进单 GPU | 模型太大，单 GPU 放不下 |

### 4.4 何时用 DDP？何时用 FSDP？

#### 决策树

```
模型能放进单个 GPU 显存？
├─ Yes → 用 DDP
│   └─ 优点：简单、高效、通信少
│
└─ No → 用 FSDP
    ├─ 显存节省：~1/N
    ├─ 代价：通信增加 ~3×
    └─ 建议：
        - 开启 reshard_after_forward（节省显存）
        - 使用 TRANSFORMER_BASED_WRAP（按层切分）
        - 确保 NVLink（通信带宽）
```

#### 实战建议

| 模型大小 | 单节点 8 卡配置 | 推荐策略 |
|---------|----------------|---------|
| ≤7B | DDP only | 简单高效 |
| 13B | FSDP only | 显存刚好 |
| 30B | FSDP + GC | 梯度检查点 |
| 70B | FSDP + TP(2) | 混合并行 |
| 405B | FSDP + TP(8) + 多节点 | 全面并行 |

---

## 5. Axolotl 的 ND 并行架构

### 5.1 四维 DeviceMesh

Axolotl 支持 **4D 并行**：

```python
DeviceMesh 的 4 个维度：

1. TP 维度（tensor_parallel_size）
   → 模型参数横向切分

2. CP 维度（context_parallel_size）
   → 序列纵向切分

3. DP_Shard 维度（dp_shard_size）
   → FSDP，参数/优化器/梯度切分

4. DP_Replicate 维度（dp_replicate_size）
   → DDP，完整模型副本

总 GPU 数 = TP × CP × DP_Shard × DP_Replicate
```

### 5.2 配置优先级

Axolotl 的配置优先级（见 `distributed.py:319-370`）：

```python
def _get_parallel_config_kwargs(
    world_size: int,
    tensor_parallel_size: int = 1,
    context_parallel_size: int = 1,
    dp_shard_size: int | None = None,
    dp_replicate_size: int | None = None,
    is_fsdp: bool = False,
):
    remaining_world_size = world_size

    # 步骤 1：分配 TP
    if tensor_parallel_size > 1:
        pc_kwargs["tp_size"] = tensor_parallel_size
        remaining_world_size //= tensor_parallel_size

    # 步骤 2：分配 CP
    if context_parallel_size > 1:
        pc_kwargs["cp_size"] = context_parallel_size
        remaining_world_size //= context_parallel_size

    # 步骤 3：分配 DP_Replicate
    if dp_replicate_size > 1:
        pc_kwargs["dp_replicate_size"] = dp_replicate_size
        remaining_world_size //= dp_replicate_size

    # 步骤 4：分配 DP_Shard
    if remaining_world_size > 1 and is_fsdp:
        pc_kwargs["dp_shard_size"] = remaining_world_size

    return pc_kwargs
```

#### 配置示例

```yaml
# 示例 1：纯 DDP（8 卡）
base_model: meta-llama/Llama-3.1-8B
# dp_shard_size 和 dp_replicate_size 都不配置
# Axolotl 自动推断：dp_shard_size = 8（如果启用 FSDP）
```

```yaml
# 示例 2：TP + FSDP（8 卡）
base_model: meta-llama/Llama-3.1-70B
tensor_parallel_size: 2
dp_shard_size: 4
fsdp_version: 2

# 计算：
# TP = 2
# DP_Shard = 4
# 总计 = 2 × 4 = 8 ✅
```

```yaml
# 示例 3：TP + FSDP + DDP（16 卡，双节点）
base_model: meta-llama/Llama-3.1-70B
tensor_parallel_size: 2
dp_shard_size: 4
dp_replicate_size: 2
fsdp_version: 2

# 计算：
# TP = 2（节点内）
# DP_Shard = 4（节点内 FSDP）
# DP_Replicate = 2（跨节点 DDP）
# 总计 = 2 × 4 × 2 = 16 ✅
```

### 5.3 DeviceMesh 拓扑

#### 单节点 8 卡：TP(2) + FSDP(4)

```
DeviceMesh: [TP=2, DP_Shard=4]

       TP 维度 →
       GPU0  GPU1
DP     GPU2  GPU3
维     GPU4  GPU5
度     GPU6  GPU7
↓

TP 组：
  Group 0: [GPU0, GPU1]  ← 模型切分
  Group 1: [GPU2, GPU3]
  Group 2: [GPU4, GPU5]
  Group 3: [GPU6, GPU7]

FSDP 组：
  Group 0: [GPU0, GPU2, GPU4, GPU6]  ← 参数切分
  Group 1: [GPU1, GPU3, GPU5, GPU7]
```

#### 双节点 16 卡：TP(2) + FSDP(4) + DDP(2)

```
DeviceMesh: [TP=2, DP_Shard=4, DP_Replicate=2]

节点 0：
       GPU0  GPU1
       GPU2  GPU3
       GPU4  GPU5
       GPU6  GPU7

节点 1：
       GPU8  GPU9
       GPU10 GPU11
       GPU12 GPU13
       GPU14 GPU15

TP 组（8 组）：
  [GPU0, GPU1], [GPU2, GPU3], ...

FSDP 组（4 组）：
  [GPU0, GPU2, GPU4, GPU6]
  [GPU1, GPU3, GPU5, GPU7]
  [GPU8, GPU10, GPU12, GPU14]
  [GPU9, GPU11, GPU13, GPU15]

DDP 组（8 组）：
  [GPU0, GPU8]
  [GPU1, GPU9]
  ...
  → 跨节点梯度同步
```

---

## 6. 配置与实战

### 6.1 基础配置（纯 FSDP）

#### 单节点 8 卡，Llama-13B

```yaml
base_model: meta-llama/Llama-3.1-13B

# === Data Parallelism 配置 ===
# 不显式配置 dp_shard_size，Axolotl 自动推断为 8
fsdp_version: 2  # ← 启用 FSDP

# === FSDP 详细配置 ===
fsdp_config:
  # Sharding 策略
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer

  # 显存优化
  reshard_after_forward: true  # ← 前向传播后立即释放参数

  # Checkpoint 配置
  state_dict_type: FULL_STATE_DICT  # 保存完整模型

  # 其他
  sync_module_states: true
  use_orig_params: true

# === 训练配置 ===
sequence_len: 2048
micro_batch_size: 4  # 每个 GPU 的 batch size
gradient_accumulation_steps: 4

# 有效 batch size = 4 × 4 × 8 = 128

# === 优化器 ===
optimizer: adamw_torch_fused
learning_rate: 2e-5
lr_scheduler: cosine

# === 性能优化 ===
bf16: true
flash_attention: true
gradient_checkpointing: true

# === 输出 ===
output_dir: ./outputs/llama-13b-fsdp/
logging_steps: 10
save_steps: 500
```

### 6.2 混合并行（TP + FSDP）

#### 单节点 8 卡，Llama-70B

```yaml
base_model: meta-llama/Llama-3.1-70B

# === 混合并行配置 ===
tensor_parallel_size: 2  # TP
dp_shard_size: 4         # FSDP
# 总计：2 × 4 = 8 GPUs

fsdp_version: 2

# === FSDP 配置 ===
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT

# === 训练配置 ===
sequence_len: 2048
micro_batch_size: 2
gradient_accumulation_steps: 8

# 有效 batch size = 2 × 8 × 4 = 64
# 注意：TP 不参与 batch size 计算！

# === 优化器 ===
optimizer: adamw_torch_fused
learning_rate: 1e-5

# === 性能优化 ===
bf16: true
flash_attention: true
gradient_checkpointing: true

output_dir: ./outputs/llama-70b-tp-fsdp/
```

### 6.3 多节点训练（TP + FSDP + DDP）

#### 双节点 16 卡，Llama-70B

```yaml
base_model: meta-llama/Llama-3.1-70B

# === 4D 并行配置 ===
tensor_parallel_size: 2     # TP（节点内）
dp_shard_size: 4            # FSDP（节点内）
dp_replicate_size: 2        # DDP（跨节点）
# 总计：2 × 4 × 2 = 16 GPUs

fsdp_version: 2

# === FSDP 配置 ===
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT

# === 训练配置 ===
sequence_len: 4096
micro_batch_size: 1
gradient_accumulation_steps: 16

# 有效 batch size = 1 × 16 × 4 × 2 = 128
# 计算：micro_batch × grad_accum × dp_shard × dp_replicate

# === 优化器 ===
optimizer: adamw_torch_fused
learning_rate: 1e-5

# === 性能优化 ===
bf16: true
flash_attention: true
gradient_checkpointing: true

output_dir: ./outputs/llama-70b-multi-node/
```

#### 运行命令

```bash
# === Node 0 (Master) ===
axolotl train config.yaml \
    --num-processes 16 \
    --num-machines 2 \
    --machine-rank 0 \
    --main-process-ip <NODE0_IP> \
    --main-process-port 29500

# === Node 1 ===
axolotl train config.yaml \
    --num-processes 16 \
    --num-machines 2 \
    --machine-rank 1 \
    --main-process-ip <NODE0_IP> \
    --main-process-port 29500
```

### 6.4 数据集配置

#### DistributedSampler 自动处理

Axolotl 内部使用 `DistributedSampler` 或 `MultipackBatchSampler`：

```python
# 在 Axolotl 内部（简化版）

if is_distributed():
    from axolotl.utils.samplers.multipack import MultipackBatchSampler

    sampler = MultipackBatchSampler(
        batch_max_len=micro_batch_size * sequence_len,
        lengths=dataset_lengths,
        # 自动处理分布式
        # 每个 GPU 得到不同的 batch
    )
else:
    sampler = RandomSampler(dataset)

train_loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=4,
)
```

你**不需要**手动配置数据分发，Axolotl 会自动处理！

---

## 7. 常见问题排查

### 7.1 显存 OOM

#### 症状

```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

#### 诊断步骤

```bash
# 1. 检查模型大小
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('meta-llama/Llama-3.1-70B')
print(f'参数量: {model.num_parameters() / 1e9:.1f}B')
print(f'显存需求（bf16）: {model.num_parameters() * 2 / 1e9:.1f}GB')
"

# 2. 检查 FSDP 配置
grep -E "fsdp|dp_shard" config.yaml

# 3. 检查 batch size
grep -E "micro_batch|gradient_acc" config.yaml
```

#### 解决方案

```yaml
# 选项 1：开启 FSDP
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true  # ← 关键！

# 选项 2：增大 dp_shard_size
dp_shard_size: 8  # 从 4 增加到 8

# 选项 3：减小 batch size
micro_batch_size: 1  # 从 2 减小到 1
gradient_accumulation_steps: 16  # 增大以补偿

# 选项 4：开启梯度检查点
gradient_checkpointing: true

# 选项 5：使用 TP 切分模型
tensor_parallel_size: 2  # 显存减半
```

### 7.2 训练速度慢

#### 症状

```
Tokens/s/GPU 远低于预期
通信时间占比过高
```

#### 诊断

```bash
# 1. 检查 GPU 互连
nvidia-smi topo -m

# 应该看到 NVLink (NV12/NV18)
# 如果是 PHB (PCIe)，通信会很慢

# 2. 检查 NCCL 日志
NCCL_DEBUG=INFO axolotl train config.yaml

# 观察 AllReduce/AllGather 时间

# 3. 使用 profiler
nsys profile -o profile.qdrep \
    python -m axolotl.cli.train config.yaml --max-steps 10
```

#### 解决方案

```yaml
# 选项 1：减少 FSDP 通信
fsdp_config:
  # 使用 TRANSFORMER_BASED_WRAP（按层切分，通信少）
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer

# 选项 2：使用 Fused Optimizer
optimizer: adamw_torch_fused  # ← 比 adamw_torch 快

# 选项 3：优化数据加载
dataloader_num_workers: 4  # 增加 worker
dataloader_pin_memory: true

# 选项 4：如果模型能放进单 GPU，改用 DDP
# 注释掉 fsdp_config，Axolotl 会自动用 DDP

# 选项 5：减少梯度累积
gradient_accumulation_steps: 4  # 从 16 减小
# 代价：有效 batch size 也会减小
```

### 7.3 Loss NaN 或发散

#### 症状

```
Epoch 1: loss=2.5
Epoch 2: loss=1.8
Epoch 3: loss=NaN
```

#### 诊断

```python
# 检查梯度
import torch.distributed as dist

# 在训练脚本中添加
if dist.get_rank() == 0:
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                print(f"异常梯度：{name}, norm={grad_norm}")
```

#### 解决方案

```yaml
# 选项 1：使用 bf16（比 fp16 更稳定）
bf16: true
fp16: false

# 选项 2：梯度裁剪
max_grad_norm: 1.0

# 选项 3：降低学习率
learning_rate: 5e-6  # 从 2e-5 降低

# 选项 4：增加 warmup
warmup_steps: 100
warmup_ratio: 0.05

# 选项 5：检查数据质量
# 确保数据集没有异常值
```

### 7.4 多节点通信失败

#### 症状

```
[Rank 8] Timeout waiting for connection
NCCL error: unhandled system error
```

#### 诊断

```bash
# 1. 检查网络连通性
# Node 0:
ping <NODE1_IP>

# 2. 检查端口
# Node 0:
nc -l 29500

# Node 1:
nc <NODE0_IP> 29500

# 3. 检查 NCCL 环境变量
echo $NCCL_SOCKET_IFNAME
echo $NCCL_IB_DISABLE
```

#### 解决方案

```bash
# 选项 1：指定网络接口
export NCCL_SOCKET_IFNAME=eth0  # 或 ib0（InfiniBand）

# 选项 2：禁用 InfiniBand（如果没有）
export NCCL_IB_DISABLE=1

# 选项 3：增加超时时间
export NCCL_TIMEOUT=7200  # 2 小时

# 选项 4：使用不同的后端
# 在配置中：
distributed_backend: gloo  # 而非 nccl（仅用于调试）

# 选项 5：检查防火墙
# 确保端口 29500-29600 开放
sudo ufw allow 29500:29600/tcp
```

### 7.5 Checkpoint 保存失败

#### 症状

```
Rank 0 saves checkpoint successfully
Rank 1+ hangs...
```

#### 解决方案

```yaml
# 选项 1：使用 FULL_STATE_DICT（推荐）
fsdp_config:
  state_dict_type: FULL_STATE_DICT
  # Rank 0 收集完整模型并保存

# 选项 2：使用 SHARDED_STATE_DICT
fsdp_config:
  state_dict_type: SHARDED_STATE_DICT
  # 每个 rank 保存自己的切片
  # 恢复时需要所有 rank 一起加载

# 选项 3：确保所有 rank 同步
# Axolotl 内部使用 barrier，通常不需要手动处理
```

---

## 总结

### Data Parallelism 核心要点

1. **基本概念**
   - 相同的模型，不同的数据
   - 梯度同步是关键（AllReduce）

2. **两种策略**
   - DDP：简单高效，但显存占用高
   - FSDP：显存节省 ~1/N，但通信增加

3. **配置原则**
   - 模型能放进单 GPU → 用 DDP
   - 模型太大 → 用 FSDP
   - 模型极大 → FSDP + TP

4. **性能优化**
   - 确保 NVLink（通信带宽）
   - 使用 Fused Optimizer
   - 开启 `reshard_after_forward`

5. **搬桌子类比**
   - DDP：每个人一张完整桌子，搬不同货物
   - FSDP：每个人 1/N 桌腿，搬货时临时组装

---

## 下一步

- 详细源码解析 → [dp_source_code_walkthrough.md](./dp_source_code_walkthrough.md)
- 快速参考卡片 → [dp_quick_reference.md](./dp_quick_reference.md)
- 返回主索引 → [README.md](./README.md)

---

*本文档由 Claude AI 辅助创作，旨在通俗易懂地讲解 Axolotl 的 Data Parallelism 实现。*
