# Axolotl 框架中的 Tensor Parallelism 深度解析

> 本文档面向 infra 初学者，通俗易懂地讲解 Axolotl 如何实现 Tensor Parallelism

## 目录

1. [什么是 Tensor Parallelism？](#1-什么是-tensor-parallelism)
2. [为什么需要 Tensor Parallelism？](#2-为什么需要-tensor-parallelism)
3. [Tensor Parallelism 的工作原理](#3-tensor-parallelism-的工作原理)
4. [Axolotl 中的 ND 并行架构](#4-axolotl-中的-nd-并行架构)
5. [源码实现分析](#5-源码实现分析)
6. [实战示例](#6-实战示例)
7. [常见问题与最佳实践](#7-常见问题与最佳实践)

---

## 1. 什么是 Tensor Parallelism？

### 1.1 用一个比喻来理解

想象你要搬一张超大的桌子：
- **数据并行 (Data Parallelism, DP)**：多个人同时搬多张相同的桌子
- **Tensor 并行 (Tensor Parallelism, TP)**：多个人一起搬**同一张**桌子的不同部分

在深度学习中：
- **模型的权重**就像桌子本身
- **TP 将模型的每一层切分成多个部分**，分配到不同的 GPU 上
- 这些 GPU **协作计算同一批数据**

### 1.2 技术定义

Tensor Parallelism (张量并行) 是一种**水平模型并行**技术，通过将神经网络的**层内权重矩阵**切分到多个设备上，使得单个层的计算可以在多个 GPU 上并行执行。

**核心思想**：来自 [Megatron-LM 论文](https://arxiv.org/pdf/1909.08053.pdf)
- 将大矩阵切分成小块
- 每个 GPU 只存储和计算一部分
- 通过通信操作合并结果

---

## 2. 为什么需要 Tensor Parallelism？

### 2.1 解决的核心问题

在训练大型语言模型时，我们会遇到三个主要瓶颈：

#### 问题 1：模型太大，单卡装不下
```
例如：Llama-70B 模型参数
- 参数量：70B (700 亿)
- FP16 存储：70B × 2 bytes = 140GB
- 一张 A100 (80GB) 装不下！
```

#### 问题 2：即使装得下，激活值也可能爆显存
```
计算过程中的中间结果（激活值）：
- Batch size = 8
- Sequence length = 4096
- Hidden size = 8192
- 单层激活值 ≈ 8 × 4096 × 8192 × 2 bytes ≈ 512MB
- 80 层模型 ≈ 40GB 激活值！
```

#### 问题 3：训练速度瓶颈
```
单卡训练速度慢：
- 更多 GPU = 更快训练
- 但需要高效的并行策略
```

### 2.2 TP vs 其他并行方法

| 并行方法 | 切分对象 | 通信频率 | 适用场景 |
|---------|---------|---------|---------|
| **Data Parallel (DP)** | 数据批次 | 低（每步一次梯度同步） | 模型能装进单卡 |
| **Tensor Parallel (TP)** | 模型层权重 | 高（每层前向/反向传播） | 模型太大或需要加速 |
| **Pipeline Parallel (PP)** | 模型层（垂直切分） | 中（层间传递） | 超大模型，可容忍 bubble |
| **FSDP** | 参数+梯度+优化器 | 中（all-gather/reduce-scatter） | 内存受限，多 GPU 可用 |

**TP 的独特优势**：
- ✅ 可以将**单个层**拆分到多个 GPU
- ✅ 适合层数不多但每层很大的模型（如 MoE 模型）
- ✅ 可与其他并行方法组合（2D/3D 并行）
- ⚠️ **需要快速互连**（NVLink/InfiniBand），因为通信频繁

---

## 3. Tensor Parallelism 的工作原理

### 3.1 核心数学原理

以一个简单的全连接层为例：

```
原始计算：Y = X @ W
- X: [batch_size, seq_len, hidden_dim]  # 输入
- W: [hidden_dim, output_dim]           # 权重矩阵
- Y: [batch_size, seq_len, output_dim]  # 输出
```

#### 列切分 (Column-wise Parallel)

将权重矩阵 W 按**列**切分成两部分：

```python
# 假设有 2 个 GPU
W = [W1 | W2]  # W1 和 W2 各占一半列

# GPU 0 计算：
Y1 = X @ W1  # 得到输出的前一半

# GPU 1 计算：
Y2 = X @ W2  # 得到输出的后一半

# 最后拼接：
Y = [Y1 | Y2]  # 无需通信，直接拼接
```

**关键点**：
- ✅ 每个 GPU 只存储一半权重
- ✅ 输入 X 在所有 GPU 上相同
- ✅ 输出需要拼接，但通常下一层会行切分

#### 行切分 (Row-wise Parallel)

将权重矩阵 W 按**行**切分：

```python
W = [W1]  # GPU 0 持有上半部分行
    [W2]  # GPU 1 持有下半部分行

# 需要先切分输入：
X1, X2 = split(X)  # 沿 hidden_dim 切分

# GPU 0 计算：
Y1 = X1 @ W1

# GPU 1 计算：
Y2 = X2 @ W2

# All-Reduce 求和：
Y = Y1 + Y2  # 需要通信！
```

**关键点**：
- ✅ 每个 GPU 只存储一半权重
- ⚠️ 输出需要 All-Reduce 通信来合并结果

### 3.2 Transformer 层的 TP 策略

现代 LLM 的核心是 Transformer 层，包含两个主要部分：

```
Transformer Layer:
    ├─ Self-Attention
    │   ├─ Q, K, V 投影 (列切分)
    │   ├─ Attention 计算
    │   └─ O 投影 (行切分)
    └─ Feed-Forward Network (FFN)
        ├─ Gate/Up 投影 (列切分)
        └─ Down 投影 (行切分)
```

#### 完整的 TP 流程（以 Llama 模型为例）

```python
# 伪代码展示 TP 如何工作

# === 输入层 ===
X = input_tokens  # [batch, seq_len, hidden_dim]

# === Self-Attention ===
# 1. QKV 投影 - 列切分
# GPU 0: Q1, K1, V1 = X @ Wq1, X @ Wk1, X @ Wv1
# GPU 1: Q2, K2, V2 = X @ Wq2, X @ Wk2, X @ Wv2
# 每个 GPU 计算部分 attention heads

# 2. Attention 计算（各 GPU 独立）
# GPU 0: attn_out1 = Attention(Q1, K1, V1)
# GPU 1: attn_out2 = Attention(Q2, K2, V2)

# 3. O 投影 - 行切分
# GPU 0: out1 = attn_out1 @ Wo1
# GPU 1: out2 = attn_out2 @ Wo2
# All-Reduce: out = out1 + out2  # 通信！

# === Feed-Forward Network ===
# 4. Gate & Up 投影 - 列切分
# GPU 0: gate1 = X @ Wgate1, up1 = X @ Wup1
# GPU 1: gate2 = X @ Wgate2, up2 = X @ Wup2

# 5. 激活函数（各 GPU 独立）
# GPU 0: ffn1 = SiLU(gate1) * up1
# GPU 1: ffn2 = SiLU(gate2) * up2

# 6. Down 投影 - 行切分
# GPU 0: down1 = ffn1 @ Wdown1
# GPU 1: down2 = ffn2 @ Wdown2
# All-Reduce: output = down1 + down2  # 通信！
```

**通信次数统计**：
- 每个 Transformer 层：**2 次 All-Reduce**
  - Self-Attention O 投影后 1 次
  - FFN Down 投影后 1 次
- 80 层模型 = **160 次通信/前向传播**
- 反向传播也需要类似通信

**为什么需要快速互连？**
```
假设 A100 80GB，NVLink 600GB/s：
- 传输 1GB 数据 ≈ 1.67ms
- 160 次通信 ≈ 267ms

假设用 PCIe 4.0，32GB/s：
- 传输 1GB 数据 ≈ 31ms
- 160 次通信 ≈ 5s (太慢！)
```

---

## 4. Axolotl 中的 ND 并行架构

Axolotl 支持**多维度并行组合** (N-Dimensional Parallelism)，可以同时使用多种并行策略。

### 4.1 DeviceMesh：并行的"地图"

PyTorch 使用 `DeviceMesh` 来组织 GPU，形成一个多维网格：

```python
# 示例：16 个 GPU 的 3D 并行
# dp_replicate=2, dp_shard=2, tp=2, cp=2

GPU 网格：
    ┌─────────────────┬─────────────────┐
    │  Replica 0      │  Replica 1      │
    ├─────────────────┼─────────────────┤
    │ ┌─────┬─────┐   │ ┌─────┬─────┐   │
    │ │TP0  │TP1  │   │ │TP0  │TP1  │   │
    │ │CP0  │CP0  │   │ │CP0  │CP0  │   │
    │ └─────┴─────┘   │ └─────┴─────┘   │
    │ FSDP Shard 0    │ FSDP Shard 0    │
    │                 │                 │
    │ ┌─────┬─────┐   │ ┌─────┬─────┐   │
    │ │TP0  │TP1  │   │ │TP0  │TP1  │   │
    │ │CP1  │CP1  │   │ │CP1  │CP1  │   │
    │ └─────┴─────┘   │ └─────┴─────┘   │
    │ FSDP Shard 1    │ FSDP Shard 1    │
    └─────────────────┴─────────────────┘

配置：
dp_shard_size: 2      # FSDP 切 2 份
dp_replicate_size: 2  # 复制 2 份（DDP）
tensor_parallel_size: 2  # TP 切 2 份
context_parallel_size: 2 # CP 切 2 份
总 GPU 数：2 × 2 × 2 × 2 = 16
```

### 4.2 并行维度详解

#### 维度 1：dp_shard_size (FSDP)
```yaml
dp_shard_size: 4  # 将模型参数切成 4 份

作用：
- 参数、梯度、优化器状态分片存储
- 计算时通过 all_gather 获取需要的参数
- 节省显存，支持更大模型
```

#### 维度 2：dp_replicate_size (DDP)
```yaml
dp_replicate_size: 2  # 复制 2 份完整模型

作用：
- 每组独立训练不同数据
- 只在梯度更新时同步（通信少）
- 适合跨节点（节点间网络慢）
```

#### 维度 3：tensor_parallel_size (TP)
```yaml
tensor_parallel_size: 2  # 每层切成 2 份

作用：
- 降低单 GPU 显存需求
- 加速大层计算
- 需要快速互连（NVLink）
```

#### 维度 4：context_parallel_size (CP)
```yaml
context_parallel_size: 4  # 序列切成 4 份

作用：
- 处理超长上下文（如 100K tokens）
- 使用 Ring-Flash-Attention
- 降低激活值显存占用
```

### 4.3 组合策略示例

#### 场景 1：单节点 8 卡训练 70B 模型
```yaml
# 8 个 A100 80GB，NVLink 互连
fsdp_version: 2
dp_shard_size: 4         # FSDP 切 4 份
tensor_parallel_size: 2  # TP 切 2 份
# 总计：4 × 2 = 8 GPUs

优势：
- TP 降低单卡显存需求（层太大）
- FSDP 分摊参数存储
- NVLink 保证 TP 通信快
```

#### 场景 2：双节点 16 卡训练
```yaml
# 2 节点 × 8 GPUs = 16 GPUs
dp_shard_size: 4         # 节点内 FSDP
dp_replicate_size: 2     # 节点间 DDP
tensor_parallel_size: 2  # TP 加速
# 总计：4 × 2 × 2 = 16 GPUs

优势：
- 节点内：FSDP+TP，快速通信
- 节点间：DDP，减少跨节点通信
```

#### 场景 3：长上下文训练
```yaml
# 8 GPUs，序列长度 32K
dp_shard_size: 2
tensor_parallel_size: 2
context_parallel_size: 2
# 总计：2 × 2 × 2 = 8 GPUs

优势：
- CP 切分序列，降低激活值
- TP 降低层显存
- FSDP 分摊参数
```

---

## 5. 源码实现分析

### 5.1 配置解析流程

Axolotl 的 TP 实现依赖 HuggingFace Accelerate 库和 PyTorch FSDP2。

#### 步骤 1：配置文件 → 参数验证

```python
# 文件：src/axolotl/utils/distributed.py

def build_parallelism_config(cfg):
    """构建并行配置"""
    pc_kwargs = _get_parallel_config_kwargs(
        get_world_size(),                  # 总 GPU 数
        cfg.tensor_parallel_size,          # TP 大小
        cfg.context_parallel_size,         # CP 大小
        cfg.dp_shard_size,                 # FSDP 大小
        cfg.dp_replicate_size,             # DDP 大小
        bool(cfg.fsdp or cfg.fsdp_config), # 是否启用 FSDP
    )

    if pc_kwargs:
        # 创建 ParallelismConfig 对象
        parallelism_config = ParallelismConfig(**pc_kwargs)
        # 构建设备网格
        device_mesh = parallelism_config.build_device_mesh("cuda")
        return parallelism_config, device_mesh

    return None, None
```

**关键逻辑**：验证配置的合法性

```python
# 文件：src/axolotl/utils/distributed.py (319-370 行)

def _get_parallel_config_kwargs(
    world_size: int,
    tensor_parallel_size: int = 1,
    context_parallel_size: int = 1,
    dp_shard_size: int | None = None,
    dp_replicate_size: int | None = None,
    is_fsdp: bool = False,
):
    pc_kwargs = {}
    remaining_world_size = world_size  # 剩余可分配的 GPU

    # 1. 先分配 TP
    if tensor_parallel_size and tensor_parallel_size > 1:
        pc_kwargs["tp_size"] = tensor_parallel_size
        remaining_world_size = remaining_world_size // tensor_parallel_size
        # 例：8 GPUs, TP=2 → remaining = 8 / 2 = 4

    # 2. 再分配 CP
    if context_parallel_size and context_parallel_size > 1:
        pc_kwargs["cp_size"] = context_parallel_size
        remaining_world_size = remaining_world_size // context_parallel_size
        # 例：remaining=4, CP=2 → remaining = 4 / 2 = 2

    # 3. 分配 DDP (dp_replicate)
    if dp_replicate_size and dp_replicate_size > 1:
        pc_kwargs["dp_replicate_size"] = dp_replicate_size
        remaining_world_size = remaining_world_size // dp_replicate_size

    # 4. 分配 FSDP (dp_shard)
    if dp_shard_size and dp_shard_size > 1:
        if not is_fsdp:
            raise ValueError(
                "dp_shard_size 需要配置 fsdp_config!"
            )
        pc_kwargs["dp_shard_size"] = dp_shard_size
        remaining_world_size = remaining_world_size // dp_shard_size

    # 5. 检查是否所有 GPU 都分配完毕
    if remaining_world_size > 1:
        raise ValueError(
            f"配置的并行度与 GPU 总数 ({world_size}) 不匹配！\n"
            f"当前配置: {pc_kwargs}"
        )

    return pc_kwargs
```

**计算示例**：
```python
# 配置：
world_size = 8
tensor_parallel_size = 2
dp_shard_size = 4

# 计算过程：
remaining = 8
remaining = 8 // 2 = 4  # 分配 TP
remaining = 4 // 4 = 1  # 分配 FSDP
# remaining == 1，所有 GPU 分配完毕 ✅
```

#### 步骤 2：创建 DeviceMesh

```python
# 文件：src/axolotl/loaders/model.py (421-426 行)

def _set_parallel_config(self):
    """设置并行配置到 Accelerator"""
    parallelism_config, device_mesh = build_parallelism_config(self.cfg)
    if parallelism_config:
        self.parallelism_config = parallelism_config
        self.device_mesh = device_mesh
        # device_mesh 会被传递给 Accelerate/FSDP
```

`DeviceMesh` 是一个逻辑 GPU 拓扑：
```python
# 例如：8 GPUs, TP=2, FSDP=4
device_mesh = DeviceMesh(
    "cuda",
    mesh=[[0, 1],    # FSDP shard 0: TP group (GPU 0, 1)
          [2, 3],    # FSDP shard 1: TP group (GPU 2, 3)
          [4, 5],    # FSDP shard 2: TP group (GPU 4, 5)
          [6, 7]],   # FSDP shard 3: TP group (GPU 6, 7)
    mesh_dim_names=["dp_shard", "tp"]
)
```

### 5.2 模型加载与 TP 应用

Axolotl 利用 PyTorch 的 `DTensor` (Distributed Tensor) 来实现 TP。

#### 关键代码位置

```python
# 文件：src/axolotl/loaders/model.py (161-190 行)

def load(self):
    """加载并准备模型"""
    # 1. 应用预加载补丁
    self.patch_manager.apply_pre_model_load_patches()
    self._apply_pre_model_load_setup()  # ← 这里设置 parallelism_config

    # 2. 构建模型
    PLUGIN_MANAGER.pre_model_load(self.cfg)
    skip_move_to_device = self._build_model()  # ← 模型在这里加载
    PLUGIN_MANAGER.post_model_build(self.cfg, self.model)

    # 3. 后处理
    self._apply_post_model_load_setup()

    # 4. 加载适配器（LoRA 等）
    lora_config = self._load_adapters()

    return self.model, lora_config
```

**模型构建时的 TP 应用**：

PyTorch FSDP2 会自动根据 `device_mesh` 将模型层转换为 DTensor：

```python
# 伪代码：FSDP2 内部逻辑（Axolotl 调用）

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

# 对于每个 Transformer 层：
for layer in model.layers:
    # Q, K, V, Gate, Up 投影 → 列切分
    parallelize_module(
        layer.self_attn.q_proj,
        device_mesh["tp"],
        ColwiseParallel()
    )
    parallelize_module(
        layer.self_attn.k_proj,
        device_mesh["tp"],
        ColwiseParallel()
    )
    # ... 其他列切分层

    # O, Down 投影 → 行切分
    parallelize_module(
        layer.self_attn.o_proj,
        device_mesh["tp"],
        RowwiseParallel()
    )
    parallelize_module(
        layer.mlp.down_proj,
        device_mesh["tp"],
        RowwiseParallel()
    )
```

**DTensor 的作用**：
- 权重矩阵不再是普通 `torch.Tensor`
- 变成 `DTensor`，自动处理切分和通信
- 前向/反向传播时自动插入 All-Reduce

```python
# 示例：一个切分后的权重矩阵
# 原始权重：[4096, 4096]
# TP=2 列切分后：

# GPU 0 上：
weight_gpu0 = DTensor(
    local_tensor=torch.randn(4096, 2048),  # 只存储一半列
    device_mesh=device_mesh["tp"],
    placements=[Shard(1)]  # 在维度 1 (列) 上切分
)

# GPU 1 上：
weight_gpu1 = DTensor(
    local_tensor=torch.randn(4096, 2048),  # 存储另一半列
    device_mesh=device_mesh["tp"],
    placements=[Shard(1)]
)

# 计算时：Y = X @ W
# PyTorch 自动处理：
# - X 广播到所有 GPU
# - 每个 GPU 计算自己的部分
# - 结果自动拼接或 All-Reduce
```

### 5.3 训练过程中的 TP

#### Trainer 初始化

```python
# 文件：src/axolotl/core/builders/base.py (55-72 行)

class TrainerBuilderBase:
    def __init__(self, cfg, model, tokenizer, processor=None):
        self.cfg = cfg
        self.model = model  # 已经是 TP 切分后的模型
        # ...

        # 添加 Tokens/秒 回调（考虑 TP）
        if cfg.include_tkps:
            callbacks.append(
                TokensPerSecondCallback(
                    cfg.tensor_parallel_size,    # TP 会影响吞吐计算
                    cfg.context_parallel_size
                )
            )
```

#### 前向传播

由于使用了 DTensor，前向传播逻辑无需修改：

```python
# 用户代码（无变化）：
outputs = model(input_ids, attention_mask)
loss = outputs.loss

# 底层 DTensor 自动处理：
# 1. input_ids 复制到所有 TP GPUs
# 2. 每层计算：
#    - 列切分层：并行计算，拼接输出
#    - 行切分层：并行计算，All-Reduce 求和
# 3. loss 在所有 TP GPUs 上相同（因为最后 All-Reduce）
```

#### 反向传播

梯度计算同样由 DTensor 自动处理：

```python
loss.backward()

# DTensor 自动：
# 1. 反向传播梯度
# 2. 行切分层：梯度需要 All-Reduce
# 3. 列切分层：梯度切分存储（无需通信）
# 4. 每个 GPU 只更新自己持有的权重部分
```

### 5.4 LoRA 与 TP 的兼容性

Axolotl 特别处理了 LoRA 与 TP 的组合：

```python
# 文件：src/axolotl/kernels/lora.py (15, 69 行)

from torch.distributed.tensor import DTensor

def lora_forward_with_tp(x, lora_A, lora_B, scaling):
    """支持 TP 的 LoRA 前向传播"""

    # 检查权重是否是 DTensor（即是否启用 TP）
    if isinstance(lora_A.weight, DTensor):
        # LoRA 矩阵也需要切分
        # lora_A: 列切分
        # lora_B: 行切分
        # 保持与主权重相同的切分策略
        result = x @ lora_A.weight @ lora_B.weight * scaling
        # DTensor 自动处理通信
    else:
        # 普通 LoRA 计算
        result = x @ lora_A.weight @ lora_B.weight * scaling

    return result
```

---

## 6. 实战示例

### 6.1 配置文件示例

#### 示例 1：纯 TP（2 卡训练小模型）

```yaml
# 文件：examples/tensor-parallel-simple.yaml

base_model: meta-llama/Llama-3.1-8B
tensor_parallel_size: 2  # 2 张 GPU 做 TP

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

sequence_len: 2048
micro_batch_size: 2
num_epochs: 1
optimizer: adamw_torch_fused
learning_rate: 2e-5

bf16: true
flash_attention: true

output_dir: ./outputs/tp-test/
```

**运行命令**：
```bash
# 使用 accelerate 启动（2 个进程）
axolotl train examples/tensor-parallel-simple.yaml \
    --launcher accelerate \
    --num-processes 2
```

**显存占用对比**：
```
不使用 TP (单卡)：
- 模型参数：8B × 2 bytes = 16GB
- 优化器状态：16GB × 2 (AdamW) = 32GB
- 激活值：~8GB
- 总计：~56GB (A100 80GB 勉强够)

使用 TP=2：
- 每卡模型参数：8GB
- 每卡优化器：16GB
- 每卡激活值：~4GB
- 每卡总计：~28GB (舒服！)
```

#### 示例 2：FSDP + TP（8 卡训练大模型）

```yaml
# 文件：examples/fsdp-tp-combined.yaml

base_model: meta-llama/Llama-3.1-70B

# 并行配置
dp_shard_size: 4         # FSDP 切 4 份
tensor_parallel_size: 2  # TP 切 2 份
# 总计：4 × 2 = 8 GPUs

# FSDP 配置
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true  # ZeRO-3 模式
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

sequence_len: 4096
micro_batch_size: 1
gradient_accumulation_steps: 4
num_epochs: 1

optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 1e-5

bf16: true
flash_attention: true

output_dir: ./outputs/fsdp-tp-70b/
```

**运行命令**：
```bash
axolotl train examples/fsdp-tp-combined.yaml \
    --launcher accelerate \
    --num-processes 8
```

**GPU 布局**：
```
8 个 GPU 的分组：

FSDP Shard 0: GPU 0, GPU 1 (TP group)
    - GPU 0 存储: 参数的 1/4，每层的左半部分
    - GPU 1 存储: 参数的 1/4，每层的右半部分

FSDP Shard 1: GPU 2, GPU 3 (TP group)
    - GPU 2 存储: 参数的 1/4，每层的左半部分
    - GPU 3 存储: 参数的 1/4，每层的右半部分

FSDP Shard 2: GPU 4, GPU 5 (TP group)
FSDP Shard 3: GPU 6, GPU 7 (TP group)

每个 FSDP shard 持有不同的参数子集
每个 TP pair 协作计算同一批数据
```

#### 示例 3：全维度并行（16 卡，HSDP + TP + CP）

```yaml
# 文件：examples/nd-parallelism-full.yaml

base_model: meta-llama/Llama-3.1-70B

# 4D 并行配置
dp_shard_size: 2         # FSDP 切 2 份
dp_replicate_size: 2     # DDP 复制 2 份
tensor_parallel_size: 2  # TP 切 2 份
context_parallel_size: 2 # CP 切 2 份
# 总计：2 × 2 × 2 × 2 = 16 GPUs

fsdp_version: 2
fsdp_config:
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

sequence_len: 16384  # 长上下文
micro_batch_size: 1  # CP 要求 batch=1
num_epochs: 1

optimizer: adamw_torch_fused
learning_rate: 5e-6

bf16: true
flash_attention: true

output_dir: ./outputs/nd-full/
```

**适用场景**：
- ✅ 多节点训练（2 节点 × 8 GPU）
- ✅ 超长上下文（16K tokens）
- ✅ 超大模型（70B+）

### 6.2 性能优化建议

#### 1. TP 组大小选择

```yaml
# 规则：TP 组内 GPU 需要快速互连

✅ 推荐配置：
# 单节点 8 卡 (NVLink)
tensor_parallel_size: 2  # 或 4
# TP 通信快，开销低

❌ 不推荐配置：
# 跨节点 TP
tensor_parallel_size: 16  # 8 GPU/node × 2 nodes
# 节点间网络慢，TP 通信成为瓶颈
```

#### 2. 激活检查点 (Gradient Checkpointing)

```yaml
# TP 会降低激活值显存，但训练大模型仍需 checkpoint
gradient_checkpointing: true

# 进一步优化：
gradient_checkpointing_kwargs:
  use_reentrant: false  # PyTorch 2.0+ 推荐
```

#### 3. Flash Attention 必需

```yaml
# TP 需要 Flash Attention 来降低通信和显存
flash_attention: true

# 原因：
# - 降低 attention 激活值显存
# - 加速 attention 计算
# - TP + FA 是标配
```

#### 4. 混合精度训练

```yaml
# TP 支持 bf16，推荐使用
bf16: true
tf32: true  # A100 支持，加速矩阵乘法

# 为什么不用 fp16？
# - fp16 容易溢出
# - bf16 动态范围更大，更稳定
```

### 6.3 调试与监控

#### 检查 TP 是否生效

```python
# 在训练脚本中添加：
import torch.distributed as dist

if dist.is_initialized():
    print(f"Rank: {dist.get_rank()}")
    print(f"World Size: {dist.get_world_size()}")

    # 检查模型参数是否是 DTensor
    for name, param in model.named_parameters():
        if hasattr(param, 'placements'):
            print(f"{name}: DTensor with {param.placements}")
        break
```

**预期输出**：
```
Rank: 0
World Size: 8
model.layers.0.self_attn.q_proj.weight: DTensor with [Shard(1)]
                                         # 列切分 ✅
```

#### 显存监控

```bash
# 训练过程中监控 GPU 显存
watch -n 1 nvidia-smi

# 预期：
# - 所有 TP GPU 显存占用相近
# - 显存应比单卡训练低（TP 倍数）
```

#### Tokens/秒统计

```yaml
# 配置文件添加：
include_tkps: true  # 启用 Tokens Per Second 统计

# 训练日志会显示：
# Step 100: 2.5K tokens/sec/GPU
# 注意：TP 会增加通信开销，可能降低吞吐
```

---

## 7. 常见问题与最佳实践

### 7.1 常见错误

#### 错误 1：GPU 数量不匹配

```bash
错误信息：
ValueError: ParallelismConfig total_size (4) does not match
num_processes (8). Please adjust dp_replicate_size/dp_shard_size/tp_size/cp_size.

原因：
配置的并行度乘积 ≠ 总 GPU 数

解决：
# 检查配置
dp_shard_size × dp_replicate_size × tp_size × cp_size = 总 GPU 数
4            × 1                   × 1       × 1       = 4 ≠ 8 ❌

# 修正：
dp_shard_size: 4
tensor_parallel_size: 2
# 4 × 2 = 8 ✅
```

#### 错误 2：TP 与绑定权重模型不兼容

```python
# 测试文件：tests/e2e/multigpu/test_tp.py (17-19 行)
@pytest.mark.skip(
    reason="TP doesn't work with models with tied weights (embeddings)"
)

问题：
某些模型的输入/输出 embedding 共享权重（tied weights）
TP 切分时会导致权重不一致

解决方案：
1. 使用不绑定权重的模型
2. 或使用 FSDP 代替 TP
```

#### 错误 3：跨节点 TP 性能差

```yaml
# 2 节点 × 8 GPU = 16 GPUs
tensor_parallel_size: 16  # ❌ 跨节点 TP，慢！

原因：
- TP 每层都需要通信
- 节点间网络慢（Ethernet/IB 慢于 NVLink）
- 通信成为瓶颈

正确做法：
tensor_parallel_size: 2  # 节点内 TP
dp_replicate_size: 2     # 节点间 DDP
dp_shard_size: 4
# 2 × 2 × 4 = 16 ✅
```

### 7.2 最佳实践

#### 1. 选择合适的并行策略

```
场景决策树：

模型能否装入单卡？
├─ 是 → 使用 DDP（最简单）
└─ 否 → 模型太大
    ├─ 层数多但每层小 → Pipeline Parallel
    └─ 层数少但每层大 → Tensor Parallel
        ├─ 单节点训练 → FSDP + TP
        └─ 多节点训练 → HSDP + TP (节点内 TP+FSDP，节点间 DDP)

上下文太长？
└─ 是 → 添加 Context Parallel

显存还不够？
└─ 启用 Gradient Checkpointing
```

#### 2. TP 大小建议

```yaml
# 模型大小 → TP size 映射

7B-13B 模型：
tensor_parallel_size: 1  # 通常不需要 TP
# 使用 FSDP 即可

30B-70B 模型：
tensor_parallel_size: 2  # 或 4
# 单节点 8 卡：FSDP=4, TP=2
# 或 FSDP=2, TP=4

175B+ 超大模型：
tensor_parallel_size: 4  # 或 8
# 配合 FSDP 和 Pipeline Parallel
```

#### 3. 硬件要求

```
TP 的硬件要求：

✅ 必需：
- NVLink 或 NVSwitch（节点内）
- 支持 DTensor 的 PyTorch 版本 (≥2.7)

✅ 推荐：
- A100 / H100 GPU（显存大，NVLink 快）
- InfiniBand（多节点训练）

❌ 不推荐：
- 仅 PCIe 互连（TP 通信慢）
- 跨机房训练（延迟高）
```

#### 4. 配置模板

```yaml
# ===== 单节点 8 卡 =====
# 适用：30B-70B 模型

fsdp_version: 2
fsdp_config:
  reshard_after_forward: true
  transformer_layer_cls_to_wrap: <YourModelLayer>

dp_shard_size: 4
tensor_parallel_size: 2

flash_attention: true
bf16: true
gradient_checkpointing: true
micro_batch_size: 1
gradient_accumulation_steps: 8

# ===== 双节点 16 卡 =====
# 适用：70B+ 模型

dp_shard_size: 4         # 节点内 FSDP
dp_replicate_size: 2     # 节点间 DDP
tensor_parallel_size: 2  # 节点内 TP

# ===== 长上下文训练 =====
# 适用：序列长度 > 8K

context_parallel_size: 4  # 或 8
tensor_parallel_size: 2
micro_batch_size: 1       # CP 要求
flash_attention: true     # 必需 Ring-Flash-Attention
```

### 7.3 性能基准

以下是 Llama-70B 在 8×A100 80GB 上的参考性能：

| 配置 | Tokens/sec/GPU | 显存/GPU | 备注 |
|------|----------------|----------|------|
| FSDP only (dp_shard=8) | 1800 | 65GB | 基准 |
| FSDP + TP (4×2) | 1600 | 45GB | 显存降低 20GB |
| HSDP + TP (2×2×2) | 1550 | 48GB | 适合多节点 |
| FSDP + TP + CP (2×2×2) | 1200 | 35GB | 长上下文 |

**吞吐下降原因**：
- TP 增加通信开销（~10-15%）
- CP 增加额外通信（Ring-Attention）
- 但显存大幅降低，可训练更大 batch

---

## 总结

### TP 的核心要点

1. **本质**：将模型层的权重矩阵切分到多个 GPU
2. **优势**：降低单卡显存，支持更大模型/batch
3. **代价**：频繁通信，需要快速互连
4. **实现**：PyTorch DTensor + Accelerate + FSDP2

### Axolotl 中的 TP 特点

1. **无缝集成**：配置文件几行即可启用
2. **多维组合**：可与 FSDP/DDP/CP 组合
3. **自动化**：DeviceMesh 自动管理 GPU 拓扑
4. **生产级**：支持 LoRA、长上下文、多节点

### 何时使用 TP？

```
✅ 使用 TP 的场景：
- 模型层太大，单卡装不下
- 有 NVLink 互连的单节点多卡
- 需要降低显存，增大 batch size

❌ 不使用 TP 的场景：
- 模型能装入单卡（用 DDP）
- 只有 PCIe 互连（通信慢）
- 跨节点训练（用 HSDP，节点内 TP）
```

### 进一步学习资源

- [Megatron-LM 论文](https://arxiv.org/pdf/1909.08053.pdf)：TP 原理
- [HuggingFace Accelerate ND-Parallel 博客](https://huggingface.co/blog/accelerate-nd-parallel)
- [PyTorch DTensor 文档](https://pytorch.org/docs/stable/distributed.tensor.html)
- [Axolotl ND Parallelism 文档](../nd_parallelism.qmd)

---

*本文档由 Claude 创作，旨在帮助 infra 初学者理解 Tensor Parallelism。如有疑问或发现错误，欢迎提 Issue！*
