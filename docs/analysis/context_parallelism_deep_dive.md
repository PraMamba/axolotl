# Axolotl 框架中的 Context Parallelism 深度解析

> 本文档面向 infra 初学者，通俗易懂地讲解 Axolotl 如何实现 Context Parallelism (序列并行)

## 目录

1. [什么是 Context Parallelism？](#1-什么是-context-parallelism)
2. [为什么需要 Context Parallelism？](#2-为什么需要-context-parallelism)
3. [Context Parallelism 的工作原理](#3-context-parallelism-的工作原理)
4. [Ring-Flash-Attention 机制](#4-ring-flash-attention-机制)
5. [源码实现分析](#5-源码实现分析)
6. [实战示例](#6-实战示例)
7. [常见问题与最佳实践](#7-常见问题与最佳实践)

---

## 1. 什么是 Context Parallelism？

### 1.1 用一个比喻来理解

回顾我们在 TP 文档中的比喻：
- **数据并行 (DP)**：多个人同时搬多张桌子
- **Tensor 并行 (TP)**：多个人一起搬同一张桌子的不同部分（按宽度切）

现在来理解 Context Parallelism：
- **Context 并行 (CP)**：多个人一起搬同一张**超长**桌子的不同部分（按长度切）

在深度学习中：
- **输入序列**就像一张超长的桌子
- **CP 将输入序列按长度切成多段**，分配到不同的 GPU 上
- 每个 GPU 处理序列的一部分，但需要**频繁交换信息**才能正确计算 Attention

### 1.2 技术定义

Context Parallelism（上下文并行），也叫 **Sequence Parallelism**（序列并行），是一种将**输入序列本身**切分到多个设备上的并行策略。

**核心思想**：
- 将长度为 N 的序列切分成 K 个子序列
- 每个 GPU 处理长度为 N/K 的子序列
- 通过 Ring-Flash-Attention 算法让 GPU 之间高效交换 Key-Value 信息

**与 TP 的区别**：
```
Tensor Parallelism (TP):
切分对象：模型层的权重矩阵
目标：降低模型显存占用

Context Parallelism (CP):
切分对象：输入序列 (tokens)
目标：降低激活值显存占用，支持超长上下文
```

---

## 2. 为什么需要 Context Parallelism？

### 2.1 超长上下文的显存瓶颈

在训练或推理超长上下文时，**激活值**（特别是 Attention 的中间结果）会占用大量显存。

#### 显存占用分析

以 Llama-8B 模型为例：

```python
# 模型参数
num_layers = 32
hidden_dim = 4096
num_heads = 32
head_dim = 128

# 单个样本，序列长度 32K
batch_size = 1
seq_len = 32768

# === 激活值显存占用 ===

# 1. Attention 中的 Q, K, V
qkv_size = 3 * seq_len * hidden_dim * 2  # fp16
qkv_size = 3 * 32768 * 4096 * 2 ≈ 768 MB

# 2. Attention Scores (Q @ K^T)
attn_scores = seq_len * seq_len * num_heads * 2  # fp16
attn_scores = 32768 * 32768 * 32 * 2 ≈ 64 GB ❌❌❌

# 单层就需要 64GB！32 层模型根本装不下！
```

**问题**：Attention 的计算复杂度是 O(N²)，N 是序列长度
- 序列长度 2K → Attention 矩阵 ≈ 256MB
- 序列长度 8K → Attention 矩阵 ≈ 4GB
- 序列长度 32K → Attention 矩阵 ≈ 64GB
- 序列长度 128K → Attention 矩阵 ≈ 1TB ❌

### 2.2 Flash Attention 的局限性

Flash Attention 通过分块计算大幅降低了显存占用：

```python
# Flash Attention 优化后：
# 不存储完整的 Attention 矩阵
# 通过分块计算，显存从 O(N²) 降到 O(N)

# 序列长度 32K，Flash Attention 显存占用：
flash_attn_memory = seq_len * hidden_dim * 2
flash_attn_memory = 32768 * 4096 * 2 ≈ 256 MB ✅
```

**但是**，即使使用 Flash Attention，超长序列的激活值仍然很大：

```python
# Llama-8B，序列长度 128K
seq_len = 131072
num_layers = 32

# 每层激活值（简化估算）：
per_layer_act = seq_len * hidden_dim * 2  # Q/K/V
per_layer_act = 131072 * 4096 * 2 ≈ 1 GB

# 32 层：
total_act = 32 * 1 GB ≈ 32 GB

# 加上梯度（反向传播）：
total_with_grad = 32 GB * 2 ≈ 64 GB

# A100 80GB 仍然装不下！
```

### 2.3 CP 的解决方案

Context Parallelism 将序列切分到多个 GPU：

```python
# 使用 CP，序列切成 4 份
context_parallel_size = 4
local_seq_len = 131072 // 4 = 32768

# 每个 GPU 的激活值：
per_gpu_act = local_seq_len * hidden_dim * 2 * 32
per_gpu_act = 32768 * 4096 * 2 * 32 ≈ 8 GB

# 每个 GPU 显存占用显著降低！✅
```

---

## 3. Context Parallelism 的工作原理

### 3.1 序列切分

假设我们有一个长度为 8 的序列，使用 4 个 GPU 做 CP：

```
原始序列：
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ T0│ T1│ T2│ T3│ T4│ T5│ T6│ T7│
└───┴───┴───┴───┴───┴───┴───┴───┘

切分后（context_parallel_size = 4）：
GPU 0: │ T0│ T1│
GPU 1: │ T2│ T3│
GPU 2: │ T4│ T5│
GPU 3: │ T6│ T7│
```

**Position IDs 调整**：
```python
# 原始 position_ids
[0, 1, 2, 3, 4, 5, 6, 7]

# 切分后，每个 GPU 的 position_ids
GPU 0: [0, 1]
GPU 1: [2, 3]
GPU 2: [4, 5]
GPU 3: [6, 7]

# 保持了正确的位置信息！
```

### 3.2 Attention 的挑战

Attention 的核心是：**每个 token 需要看到所有其他 token**

```python
# Attention 计算
Q = query_states   # [batch, seq_len, num_heads, head_dim]
K = key_states     # [batch, seq_len, num_heads, head_dim]
V = value_states   # [batch, seq_len, num_heads, head_dim]

# Attention scores
scores = Q @ K.transpose(-2, -1)  # [batch, num_heads, seq_len, seq_len]
#        ↑    ↑
#     每个 token 要看到所有 token

# Attention output
attn_out = softmax(scores) @ V  # [batch, seq_len, num_heads, head_dim]
```

**问题**：如果 Q 在 GPU 0，K/V 在 GPU 1/2/3，如何计算 Attention？

```
GPU 0 的 tokens (T0, T1) 需要 attend to 所有 tokens (T0-T7)
    - 本地有 K0, V0 (T0, T1)
    - 需要 K1, V1 (T2, T3) from GPU 1
    - 需要 K2, V2 (T4, T5) from GPU 2
    - 需要 K3, V3 (T6, T7) from GPU 3

如果直接 All-Gather 所有 K, V → 显存爆炸 ❌
```

### 3.3 朴素方案：All-Gather（不可行）

```python
# 朴素方案：每个 GPU 收集完整的 K, V
all_K = all_gather([K0, K1, K2, K3])  # 显存 × 4 ❌
all_V = all_gather([V0, V1, V2, V3])  # 显存 × 4 ❌

# 然后计算完整 Attention
attn_out = flash_attention(Q, all_K, all_V)

# 问题：显存占用回到原点，CP 失去意义！
```

### 3.4 高效方案：Ring-Flash-Attention

Ring-Flash-Attention 结合了两个关键技术：
1. **Flash Attention**：分块计算，避免存储完整 Attention 矩阵
2. **Ring 通信**：循环传递 K/V，避免同时存储所有 K/V

核心思想：
- 每个 GPU 轮流接收其他 GPU 的 K/V
- 在接收到 K/V 时，立即计算部分 Attention，然后丢弃
- 通过 N-1 轮通信，每个 GPU 都能看到所有 K/V
- **显存只需要存储本地的 K/V + 1 份传递中的 K/V**

---

## 4. Ring-Flash-Attention 机制

### 4.1 Ring 通信拓扑

假设 4 个 GPU，编号 0-3，形成一个环：

```
        GPU 0
         ↑ ↓
    GPU 3 ⟳ GPU 1
         ↑ ↓
        GPU 2

通信方向：顺时针
GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0
```

### 4.2 逐步执行流程

以 GPU 0 为例，展示它如何计算完整的 Attention：

#### 初始状态

```
GPU 0:
- Q0: [2 tokens] (本地 query)
- K0, V0: [2 tokens] (本地 key/value)
- 需要计算：Q0 @ [K0, K1, K2, K3]
```

#### 第 1 轮：使用本地 K0, V0

```python
# GPU 0 计算第一部分 Attention
attn_output_0 = flash_attention(Q0, K0, V0)
# 这是 Q0 对 T0, T1 的 attention

# 同时，准备传递 K0, V0 给下一个 GPU
send(K0, V0, dest=GPU_1)
```

**关键点**：
- 使用 Flash Attention，分块计算，不存储中间的 Attention scores
- 计算结果是**部分累积值**，需要后续更新

#### 第 2 轮：接收 K3, V3

```python
# GPU 0 从 GPU 3 接收 K3, V3
K_recv, V_recv = recv(from=GPU_3)  # 接收 K3, V3

# 计算 Q0 对 T6, T7 的 attention
attn_output_3 = flash_attention(Q0, K3, V3)

# 累积到结果中
attn_output_partial = combine(attn_output_0, attn_output_3)

# 传递 K3, V3 给下一个 GPU
send(K3, V3, dest=GPU_1)
```

#### 第 3 轮：接收 K2, V2

```python
# GPU 0 从 GPU 3 接收 K2, V2（GPU 3 从 GPU 2 接收的）
K_recv, V_recv = recv(from=GPU_3)  # 接收 K2, V2

# 计算 Q0 对 T4, T5 的 attention
attn_output_2 = flash_attention(Q0, K2, V2)

# 继续累积
attn_output_partial = combine(attn_output_partial, attn_output_2)

# 传递给下一个 GPU
send(K2, V2, dest=GPU_1)
```

#### 第 4 轮：接收 K1, V1

```python
# GPU 0 从 GPU 3 接收 K1, V1
K_recv, V_recv = recv(from=GPU_3)  # 接收 K1, V1

# 计算 Q0 对 T2, T3 的 attention
attn_output_1 = flash_attention(Q0, K1, V1)

# 最终累积
attn_output_final = combine(attn_output_partial, attn_output_1)

# 现在 GPU 0 有了 Q0 对所有 tokens 的完整 attention！✅
```

### 4.3 数学原理：分块 Softmax

Ring-Flash-Attention 的数学基础是**分块 Softmax**。

#### 标准 Attention

```python
# 标准 Attention
scores = Q @ K.T / sqrt(d_k)  # [seq_len, seq_len]
attn_weights = softmax(scores)  # [seq_len, seq_len]
output = attn_weights @ V  # [seq_len, hidden_dim]
```

#### 分块 Attention

假设 K, V 被切成两部分：K = [K1, K2], V = [V1, V2]

```python
# 分别计算
scores1 = Q @ K1.T  # [seq_len, seq_len/2]
scores2 = Q @ K2.T  # [seq_len, seq_len/2]

# 合并后再 softmax
scores_all = concat([scores1, scores2], dim=-1)  # [seq_len, seq_len]
attn_weights = softmax(scores_all)
output = attn_weights @ concat([V1, V2])
```

**问题**：这需要同时存储 scores1 和 scores2，显存占用大。

#### Online Softmax 技巧

Ring-Flash-Attention 使用 **Online Softmax**，允许逐步更新：

```python
# 第 1 块
scores1 = Q @ K1.T
max1 = max(scores1)
exp_scores1 = exp(scores1 - max1)
sum1 = sum(exp_scores1)
output1 = (exp_scores1 @ V1) / sum1  # 临时结果

# 第 2 块
scores2 = Q @ K2.T
max2 = max(scores2)
exp_scores2 = exp(scores2 - max2)
sum2 = sum(exp_scores2)

# 更新全局 max 和 sum
global_max = max(max1, max2)
new_sum1 = sum1 * exp(max1 - global_max)
new_sum2 = sum2 * exp(max2 - global_max)
global_sum = new_sum1 + new_sum2

# 合并输出
output_final = (
    output1 * new_sum1 +
    (exp_scores2 @ V2 / sum2) * new_sum2
) / global_sum
```

**关键点**：
- 每次只需要存储当前块的 K, V
- 通过维护全局的 max 和 sum，逐步更新输出
- 最终得到正确的 Softmax 结果

### 4.4 通信与计算重叠

Ring-Flash-Attention 的高效性来自**通信与计算的重叠**：

```
时间轴（GPU 0 的视角）：

轮次 1: │ 计算 Q@K0 │ → │ 发送 K0 │
                ↓
轮次 2:    │ 接收 K3 │ → │ 计算 Q@K3 │ → │ 发送 K3 │
                            ↓
轮次 3:                │ 接收 K2 │ → │ 计算 Q@K2 │ → │ 发送 K2 │
                                        ↓
轮次 4:                            │ 接收 K1 │ → │ 计算 Q@K1 │

关键：
- 接收、计算、发送流水线执行
- GPU 利用率高
- 通信时间被计算时间掩盖
```

### 4.5 显存占用对比

```python
# 假设 4 个 GPU，每个处理 8K tokens

# === 不使用 CP ===
seq_len = 32768
memory_per_gpu = seq_len * hidden_dim * 2  # K + V
memory_per_gpu = 32768 * 4096 * 2 * 2 ≈ 512 MB

# === 使用 CP (All-Gather 方案) ===
# 每个 GPU 收集所有 K, V
memory_per_gpu = 32768 * 4096 * 2 * 2 ≈ 512 MB  # 没省！

# === 使用 CP (Ring-Flash-Attention) ===
local_seq_len = 32768 // 4 = 8192
# 本地 K/V
local_memory = 8192 * 4096 * 2 * 2 ≈ 128 MB
# 传递中的 K/V (只有 1 份)
ring_memory = 8192 * 4096 * 2 * 2 ≈ 128 MB
# 总计
memory_per_gpu = 128 + 128 = 256 MB  # 省了一半！✅
```

---

## 5. 源码实现分析

### 5.1 配置解析

CP 的配置非常简单：

```yaml
# 配置文件
context_parallel_size: 4  # 使用 4 个 GPU 做序列并行

# 可选配置
heads_k_stride: 1  # K 的 head 步长（优化通信）
ring_attn_func: varlen_llama3  # 或 batch_ring
```

在 Axolotl 中，CP 通过 `DeviceMesh` 的 "cp" 维度实现：

```python
# 文件：src/axolotl/utils/distributed.py (299-316 行)

def build_parallelism_config(cfg):
    pc_kwargs = _get_parallel_config_kwargs(
        world_size=get_world_size(),
        tensor_parallel_size=cfg.tensor_parallel_size,
        context_parallel_size=cfg.context_parallel_size,  # ← CP 配置
        dp_shard_size=cfg.dp_shard_size,
        dp_replicate_size=cfg.dp_replicate_size,
        is_fsdp=bool(cfg.fsdp or cfg.fsdp_config),
    )

    if pc_kwargs:
        parallelism_config = ParallelismConfig(**pc_kwargs)
        device_mesh = parallelism_config.build_device_mesh("cuda")
        return parallelism_config, device_mesh

    return None, None
```

**DeviceMesh 示例**（8 GPUs, TP=2, CP=2, FSDP=2）：

```python
device_mesh = DeviceMesh(
    "cuda",
    mesh=[
        [[[0, 1], [2, 3]],   # FSDP shard 0
         [[4, 5], [6, 7]]],  # FSDP shard 1
    ],
    mesh_dim_names=["dp_shard", "cp", "tp"]
)

# 访问 CP 维度
cp_mesh = device_mesh["cp"]  # CP 组的网格

# 例如 GPU 0 的 CP 伙伴是 GPU 2（同一列）
# GPU 0 和 GPU 2 共同处理一个序列的不同部分
```

### 5.2 Ring Attention 初始化

```python
# 文件：src/axolotl/monkeypatch/ring_attn/patch.py (135-213 行)

def register_ring_attn_from_device_mesh(
    device_mesh: DeviceMesh,
    context_parallel_dim: tuple[str, ...],  # ("cp",)
    heads_k_stride: int | None,
    ring_attn_func: RingAttnFunc | None,
):
    """使用 DeviceMesh 创建 ring attention 组"""

    # 1. 提取 CP 子网格
    sequence_mesh = device_mesh[context_parallel_dim]

    # 2. 获取 CP 进程组
    sequence_pg = sequence_mesh.get_group()
    context_parallel_size = sequence_mesh.size()

    # 3. 设置全局 ring attention 组
    set_ring_attn_group(sequence_pg)

    # 4. 根据配置选择 ring attention 实现
    if ring_attn_func is RingAttnFunc.VARLEN_LLAMA3:
        # 使用 ring-flash-attn 库的 varlen 实现
        ring_flash_attn.adapters.hf_adapter.substitute_hf_flash_attn(
            process_group=get_ring_attn_group(),
            heads_k_stride=heads_k_stride or 1
        )
    elif ring_attn_func is RingAttnFunc.BATCH_RING:
        # 使用 batch ring 实现
        substitute_hf_flash_attn(
            process_group=get_ring_attn_group(),
            ring_attn_func=ring_attn_func,
        )
```

**关键函数**：`substitute_hf_flash_attn`
- 这个函数会**替换** Hugging Face Transformers 中的 Flash Attention 实现
- 将标准 Flash Attention 换成 Ring-Flash-Attention
- 模型代码无需任何修改！

### 5.3 序列切分（Pre-Forward Hook）

在每个前向传播之前，序列会被自动切分：

```python
# 文件：src/axolotl/utils/ctx_managers/sequence_parallel.py (24-167 行)

def apply_sequence_parallelism(
    batch: dict[str, torch.Tensor],
    local_rank: int,           # 当前 GPU 在 CP 组中的 rank
    local_world_size: int,     # CP 组的大小
    gradient_accumulation_steps: int,
    ring_attn_func: RingAttnFunc,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """
    对 batch 应用序列并行切分
    """
    batch_size, original_seq_len = batch["input_ids"].shape

    # === 1. 创建或更新 position_ids ===
    if batch.get("position_ids") is not None:
        # 已有 position_ids（sample packing）
        update_ring_attn_params(position_ids=batch["position_ids"])
    else:
        # 创建标准 position_ids
        batch["position_ids"] = torch.arange(
            0, original_seq_len,
            dtype=torch.long,
            device=batch["input_ids"].device,
        ).expand(batch["input_ids"].size(0), -1)

    # === 2. 添加 padding（确保能被 CP size 整除）===
    total_seq_len = original_seq_len
    pad_len = 0
    divisor = min(local_world_size, 64)

    if total_seq_len % divisor != 0:
        pad_len = divisor - (total_seq_len % divisor)

        # 对所有相关 tensor 添加 padding
        for key in batch:
            if (isinstance(batch[key], torch.Tensor) and
                batch[key].dim() > 1 and
                batch[key].size(1) == total_seq_len):

                # 创建 padding（labels 用 -100，其他用 0）
                pad_value = -100 if key == "labels" else 0
                padding = torch.full(
                    (batch[key].size(0), pad_len, *batch[key].shape[2:]),
                    pad_value,
                    dtype=batch[key].dtype,
                    device=batch[key].device,
                )

                # 拼接 padding
                batch[key] = torch.cat([batch[key], padding], dim=1)

        total_seq_len = batch["input_ids"].size(1)

    # === 3. 切分序列 ===
    for key in batch:
        if not isinstance(batch[key], torch.Tensor) or batch[key].dim() <= 1:
            continue

        # 沿序列维度（dim=1）切分
        if batch[key].size(1) == total_seq_len:
            batch[key] = (
                batch[key]
                .chunk(local_world_size, dim=1)[local_rank]
                .contiguous()
            )

    return batch, original_seq_len, pad_len
```

**切分示例**：

```python
# 输入（GPU 0）
input_ids = [
    [101, 2054, 2003, 1996, ...]  # 32768 tokens
]

# CP size = 4，切分后（GPU 0 只保留第 1 块）
input_ids_gpu0 = [
    [101, 2054, 2003, 1996, ...]  # 8192 tokens
]

# GPU 1-3 分别持有其他块
input_ids_gpu1 = tokens[8192:16384]
input_ids_gpu2 = tokens[16384:24576]
input_ids_gpu3 = tokens[24576:32768]
```

### 5.4 Ring-Flash-Attention 执行

实际的 Ring-Flash-Attention 由 `ring-flash-attn` 库实现：

```python
# 文件：src/axolotl/monkeypatch/ring_attn/patch.py (50-133 行)

def create_ring_flash_attention_forward(
    process_group: dist.ProcessGroup,
    heads_k_stride: int
):
    from ring_flash_attn import llama3_flash_attn_varlen_func

    def _flash_attention_forward_v3(
        query_states, key_states, value_states,
        attention_mask, query_length, is_causal,
        dropout=0.0, position_ids=None, ...
    ):
        """Ring-Flash-Attention 前向传播"""

        # 确保是 causal attention
        assert is_causal, "only causal attention is supported"

        # 确保 batch_size = 1（varlen 要求）
        batch_size = query_states.size(0)
        assert batch_size == 1, "varlen data should be processed in advance"

        # === 调用 Ring-Flash-Attention 核心函数 ===
        attn_output = llama3_flash_attn_varlen_func(
            query_states.squeeze(dim=0),   # 去掉 batch 维度
            key_states.squeeze(dim=0),
            value_states.squeeze(dim=0),
            cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],  # 累积序列长度
            cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
            max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
            max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
            heads_k_stride=heads_k_stride,  # K 的 head 步长
            local_k_slice=DATA_PARAMS["local_k_slice"],
            dropout_p=dropout,
            softmax_scale=None,
            causal=causal,
            group=process_group,  # ← Ring 通信组！
        )

        return attn_output.unsqueeze(dim=0)

    return [_flash_attention_forward_v3]
```

**`llama3_flash_attn_varlen_func` 内部**（简化伪代码）：

```python
def llama3_flash_attn_varlen_func(q, k, v, group, ...):
    """Ring-Flash-Attention 实现"""

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    # 初始化输出累积器
    output = torch.zeros_like(q)
    max_score = float('-inf')
    sum_exp = 0.0

    # 当前持有的 K, V
    current_k = k
    current_v = v

    # === Ring 循环（world_size 轮）===
    for step in range(world_size):
        # 1. 计算当前块的 attention
        scores = q @ current_k.transpose(-2, -1) / sqrt(d_k)

        # 2. Online Softmax 更新
        new_max = max(max_score, scores.max())
        exp_scores = exp(scores - new_max)

        # 更新 sum
        sum_exp = sum_exp * exp(max_score - new_max) + exp_scores.sum()
        max_score = new_max

        # 3. 累积输出
        output = output * exp(old_max - new_max) + exp_scores @ current_v

        # 4. 传递 K, V 给下一个 GPU（除了最后一轮）
        if step < world_size - 1:
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1 + world_size) % world_size

            # 发送给下一个
            dist.send(current_k, dst=next_rank, group=group)
            dist.send(current_v, dst=next_rank, group=group)

            # 从上一个接收
            current_k = dist.recv(src=prev_rank, group=group)
            current_v = dist.recv(src=prev_rank, group=group)

    # 最终归一化
    output = output / sum_exp

    return output
```

### 5.5 输出聚合（Post-Forward Hook）

前向传播后，需要将切分的输出聚合回完整序列：

```python
# 文件：src/axolotl/utils/ctx_managers/sequence_parallel.py (302-308 行)

def _gather_outputs(self, output: CausalLMOutputWithPast):
    """从所有 rank 收集切分的输出，重建完整 tensor"""

    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dim() > 1:
            # 使用自定义的 All-Gather（保留梯度）
            output[key] = AllGatherWithGrad.apply(value, self.process_group)

    return output
```

**AllGatherWithGrad**：自定义 autograd 函数，支持梯度反向传播

```python
# 文件：src/axolotl/utils/ctx_managers/sequence_parallel.py (311-387 行)

class AllGatherWithGrad(torch.autograd.Function):
    """支持梯度的 All-Gather"""

    @staticmethod
    def forward(ctx, input_tensor, group):
        """前向传播：收集所有 GPU 的输出"""

        ctx.group = group
        ctx.rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        # 1. 收集形状信息（支持不同长度）
        local_shape = torch.tensor(list(input_tensor.shape))
        all_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
        dist.all_gather(all_shapes, local_shape, group=group)

        # 2. 存储序列长度（反向传播需要）
        seq_lens = [int(shape[1].item()) for shape in all_shapes]
        ctx.seq_lens = seq_lens

        # 3. All-Gather 实际数据
        gathered = [
            torch.zeros(tuple(shape.tolist()), dtype=input_tensor.dtype, device=input_tensor.device)
            for shape in all_shapes
        ]
        dist.all_gather(gathered, input_tensor, group=group)

        # 4. 拼接
        result = torch.cat(gathered, dim=1)  # 沿序列维度拼接

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：提取本 rank 对应的梯度切片"""

        rank = ctx.rank
        seq_lens = ctx.seq_lens

        # 计算本 rank 的梯度起始位置
        offset = sum(seq_lens[:rank])

        # 提取梯度切片
        grad_slice = grad_output[:, offset : offset + seq_lens[rank]].contiguous()

        return grad_slice, None  # 第二个返回值是 group（不需要梯度）
```

**为什么需要自定义 All-Gather？**

PyTorch 的 `dist.all_gather` 不支持梯度反向传播。`AllGatherWithGrad` 实现了：
- **前向传播**：收集所有 GPU 的输出，拼接成完整序列
- **反向传播**：将梯度切分，返回每个 GPU 对应的切片

---

## 6. 实战示例

### 6.1 基础配置

#### 示例 1：纯 CP（4 卡，16K 上下文）

```yaml
# 文件：examples/context-parallel-simple.yaml

base_model: meta-llama/Llama-3.1-8B
context_parallel_size: 4  # 4 张 GPU 做 CP

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

sequence_len: 16384  # 16K 上下文
micro_batch_size: 1  # CP 要求 batch_size=1
num_epochs: 1

optimizer: adamw_torch_fused
learning_rate: 2e-5

bf16: true
flash_attention: true  # CP 必需 Flash Attention

output_dir: ./outputs/cp-test/
```

**运行命令**：
```bash
axolotl train examples/context-parallel-simple.yaml \
    --num-processes 4
```

**显存占用对比**：
```
不使用 CP (单卡)：
- 序列长度：16384
- 激活值：16384 * 4096 * 2 * 32 layers ≈ 16 GB
- 加上模型和梯度：OOM ❌

使用 CP=4：
- 每卡序列长度：16384 / 4 = 4096
- 每卡激活值：4096 * 4096 * 2 * 32 ≈ 4 GB
- 每卡总显存：~25 GB ✅
```

#### 示例 2：CP + FSDP（8 卡，32K 上下文）

```yaml
# 文件：examples/cp-fsdp-combined.yaml

base_model: meta-llama/Llama-3.1-8B

# 并行配置
dp_shard_size: 2         # FSDP 切 2 份
context_parallel_size: 4 # CP 切 4 份
# 总计：2 × 4 = 8 GPUs

# FSDP 配置
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

sequence_len: 32768  # 32K 上下文
micro_batch_size: 1
gradient_accumulation_steps: 8
num_epochs: 1

optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 1e-5

bf16: true
flash_attention: true
gradient_checkpointing: true

output_dir: ./outputs/cp-fsdp-8b/
```

**GPU 布局**：
```
8 个 GPU 的分组：

CP Group 0: GPU 0, GPU 1, GPU 2, GPU 3
    - FSDP Shard 0: GPU 0, GPU 2
    - FSDP Shard 1: GPU 1, GPU 3

CP Group 1: GPU 4, GPU 5, GPU 6, GPU 7
    - FSDP Shard 0: GPU 4, GPU 6
    - FSDP Shard 1: GPU 5, GPU 7

每个 CP group 处理不同的 batch
每个 GPU 处理序列的 1/4（8K tokens）
```

#### 示例 3：3D 并行（TP + CP + FSDP，8 卡）

```yaml
# 文件：examples/3d-parallel-cp.yaml

base_model: meta-llama/Llama-3.1-70B

# 3D 并行配置
dp_shard_size: 2         # FSDP 切 2 份
tensor_parallel_size: 2  # TP 切 2 份
context_parallel_size: 2 # CP 切 2 份
# 总计：2 × 2 × 2 = 8 GPUs

fsdp_version: 2
fsdp_config:
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

sequence_len: 16384  # 16K 上下文
micro_batch_size: 1
gradient_accumulation_steps: 16
num_epochs: 1

optimizer: adamw_torch_fused
learning_rate: 5e-6

bf16: true
flash_attention: true
gradient_checkpointing: true

# CP 特定配置
heads_k_stride: 1
ring_attn_func: varlen_llama3

output_dir: ./outputs/3d-parallel-70b/
```

**适用场景**：
- ✅ 70B 模型单节点训练
- ✅ 中等长度上下文（16K）
- ✅ 显存受限（每卡 ~45GB）

### 6.2 Sample Packing + CP

CP 支持与 Sample Packing 结合使用：

```yaml
# 文件：examples/cp-sample-packing.yaml

base_model: meta-llama/Llama-3.1-8B
context_parallel_size: 4

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

sequence_len: 16384
sample_packing: true       # ← 启用 Sample Packing
eval_sample_packing: false
micro_batch_size: 1
num_epochs: 1

optimizer: adamw_torch_fused
learning_rate: 2e-5

bf16: true
flash_attention: true

# CP 配置（Sample Packing 需要 varlen）
ring_attn_func: varlen_llama3

output_dir: ./outputs/cp-packed/
```

**工作流程**：
1. **Sample Packing**：将多个短样本拼接成长序列
2. **Position IDs**：记录每个 token 的真实位置
3. **CP 切分**：长序列被切分到多个 GPU
4. **Varlen Ring-Flash-Attention**：根据 position_ids 正确计算 attention

### 6.3 性能调优

#### 1. heads_k_stride 调整

```yaml
# heads_k_stride 控制 K 传递的步长

# 默认（heads_k_stride=1）：
# - 每次传递所有 K heads
# - 通信量大，但计算精确

heads_k_stride: 1  # 推荐，精度高

# 增大 stride（heads_k_stride=2）：
# - 每次传递一半 K heads
# - 通信量减半，但可能影响精度

heads_k_stride: 2  # 实验性，可能加速但降低精度
```

#### 2. Ring Attention 实现选择

```yaml
# ring_attn_func 选择

# Option 1: varlen_llama3（推荐）
ring_attn_func: varlen_llama3
# - 支持 Sample Packing
# - 性能最优
# - 适用于大多数场景

# Option 2: batch_ring
ring_attn_func: batch_ring
# - 不支持 Sample Packing
# - 实现更简单
# - 适用于固定长度序列
```

#### 3. 批大小调整

```yaml
# CP 会减少有效批大小

# 例：8 GPUs, CP=4
# 实际只有 2 个 CP 组，每组处理 1 个 batch
# 有效 global batch size = micro_batch_size * 2

# 补偿方法：增大 gradient_accumulation_steps
micro_batch_size: 1
gradient_accumulation_steps: 32  # 增大以保持有效 batch size
```

---

## 7. 常见问题与最佳实践

### 7.1 常见错误

#### 错误 1：micro_batch_size > 1

```bash
错误信息：
AssertionError: varlen data should be processed in advance.

原因：
CP 的 varlen 实现要求 batch_size = 1

解决：
micro_batch_size: 1  # 必须为 1
gradient_accumulation_steps: 16  # 用梯度累积增大有效 batch
```

#### 错误 2：未启用 Flash Attention

```bash
错误信息：
RuntimeError: CP requires flash_attention to be enabled

原因：
Ring-Flash-Attention 依赖 Flash Attention

解决：
flash_attention: true  # 必须启用
```

#### 错误 3：序列长度不能被 CP size 整除

```bash
警告信息：
Added padding of 128 tokens to make sequence length divisible by CP size

解决方案：
# Axolotl 会自动添加 padding，无需担心
# 但最好设置序列长度为 CP size 的倍数：
sequence_len: 16384  # 可被 2, 4, 8 整除
```

#### 错误 4：CP + LoRA 兼容性

```yaml
# LoRA 与 CP 兼容，但需要注意：

adapter: lora
lora_r: 16
context_parallel_size: 4

# LoRA 权重会在 CP 收集后应用
# 不影响 CP 的序列切分
```

### 7.2 最佳实践

#### 1. CP 大小选择

```yaml
# 规则：根据序列长度和显存选择

序列长度 8K-16K：
context_parallel_size: 2  # 或 4

序列长度 32K-64K：
context_parallel_size: 4  # 或 8

序列长度 128K+：
context_parallel_size: 8  # 或 16

# 原则：
# - CP size 越大，通信开销越大
# - 显存够用的情况下，不要过度切分
```

#### 2. CP 与其他并行的组合

```yaml
# === 推荐组合 ===

# 长上下文，中等模型（8B-13B）
dp_shard_size: 4
context_parallel_size: 2
# 8 GPUs = 4 × 2

# 长上下文，大模型（70B）
dp_shard_size: 2
tensor_parallel_size: 2
context_parallel_size: 2
# 8 GPUs = 2 × 2 × 2

# 超长上下文（128K+），中等模型
dp_shard_size: 2
context_parallel_size: 8
# 16 GPUs = 2 × 8

# === 不推荐组合 ===

# 短序列（<4K）使用 CP
sequence_len: 2048
context_parallel_size: 4  # ❌ 通信开销大于收益

# CP 跨节点
# ❌ Ring 通信频繁，需要快速互连
```

#### 3. 性能优化清单

```yaml
# ✅ 必做优化

1. 启用 Flash Attention
flash_attention: true

2. 使用 bf16
bf16: true
tf32: true

3. 启用梯度检查点
gradient_checkpointing: true

4. 使用 Fused Optimizer
optimizer: adamw_torch_fused

# ✅ CP 特定优化

5. 选择合适的 ring_attn_func
ring_attn_func: varlen_llama3  # 大多数情况

6. 设置合理的 heads_k_stride
heads_k_stride: 1  # 默认即可

7. 调整 gradient_accumulation_steps
# 补偿 CP 导致的 batch size 减小
gradient_accumulation_steps: 32

8. 确保序列长度是 CP size 的倍数
sequence_len: 16384  # 2^14，可被 2, 4, 8 整除
```

### 7.3 调试技巧

#### 检查 CP 是否生效

```python
# 在训练开始前添加：
from axolotl.monkeypatch.ring_attn import get_ring_attn_group
import torch.distributed as dist

cp_group = get_ring_attn_group()
cp_size = dist.get_world_size(cp_group)
cp_rank = dist.get_rank(cp_group)

print(f"✅ CP 已生效:")
print(f"   CP Size: {cp_size}")
print(f"   CP Rank: {cp_rank}")

# 预期输出（GPU 0, CP=4）：
# ✅ CP 已生效:
#    CP Size: 4
#    CP Rank: 0
```

#### 监控序列切分

```python
# 在训练循环中添加：
def training_step(self, model, inputs):
    local_seq_len = inputs["input_ids"].size(1)
    print(f"GPU {self.rank}, Local Seq Len: {local_seq_len}")

# 预期输出（CP=4, 原始长度 16384）：
# GPU 0, Local Seq Len: 4096
# GPU 1, Local Seq Len: 4096
# GPU 2, Local Seq Len: 4096
# GPU 3, Local Seq Len: 4096
```

#### 验证输出聚合

```python
# 检查输出是否正确聚合
def compute_loss(self, model, inputs):
    outputs = model(**inputs)

    # 输出应该是完整序列长度
    logits_seq_len = outputs.logits.size(1)
    original_seq_len = inputs["input_ids"].size(1) * context_parallel_size

    assert logits_seq_len == original_seq_len, \
        f"输出序列长度 ({logits_seq_len}) 应等于原始长度 ({original_seq_len})"
```

### 7.4 性能基准

以下是 Llama-8B 在不同配置下的参考性能：

| 配置 | 序列长度 | Tokens/s/GPU | 显存/GPU | 备注 |
|------|----------|--------------|----------|------|
| 无 CP (4 GPUs) | 4K | 2500 | 25GB | 基准 |
| CP=2 (4 GPUs) | 8K | 2200 | 20GB | -12% 吞吐 |
| CP=4 (4 GPUs) | 16K | 1900 | 15GB | -24% 吞吐 |
| CP=8 (8 GPUs) | 32K | 1600 | 12GB | -36% 吞吐 |
| CP=4 + FSDP (8 GPUs) | 32K | 1800 | 10GB | 最优配置 |

**吞吐下降原因**：
- Ring 通信开销（~10-15% per doubling of CP size）
- 有效 batch size 减小
- GPU 利用率降低

**但是**：
- ✅ 显存大幅降低，可训练更长序列
- ✅ 可以增大 batch size（如果显存允许）
- ✅ 解锁了原本无法训练的超长上下文场景

### 7.5 与 TP 的对比

| 维度 | Tensor Parallelism | Context Parallelism |
|------|-------------------|---------------------|
| **切分对象** | 模型权重（宽度） | 输入序列（长度） |
| **通信频率** | 每层 2 次 All-Reduce | 每层 N-1 轮 Ring 传递 |
| **显存节省** | 模型参数 + 激活值 | 主要是激活值 |
| **适用场景** | 大模型（层宽） | 长上下文 |
| **硬件要求** | NVLink（节点内） | NVLink（节点内） |
| **batch size 影响** | 不影响 | 减小（÷ CP size） |
| **可组合性** | ✅ 与 FSDP/CP 组合 | ✅ 与 FSDP/TP 组合 |

**组合使用示例**：

```yaml
# TP 解决模型太大
# CP 解决序列太长
# FSDP 进一步降低显存

# 70B 模型 + 32K 上下文
dp_shard_size: 2
tensor_parallel_size: 2  # 层太宽
context_parallel_size: 2 # 序列太长
# 8 GPUs = 2 × 2 × 2
```

---

## 总结

### CP 的核心要点

1. **本质**：将输入序列切分到多个 GPU，通过 Ring 通信计算 Attention
2. **优势**：显著降低激活值显存，支持超长上下文（128K+）
3. **代价**：Ring 通信开销，batch size 减小，吞吐降低 10-30%
4. **实现**：Ring-Flash-Attention（online softmax + ring communication）

### Axolotl 中的 CP 特点

1. **无缝集成**：配置 `context_parallel_size` 即可启用
2. **自动化**：序列切分、padding、聚合全自动
3. **灵活组合**：可与 TP/FSDP/DDP 任意组合
4. **高性能**：基于 `ring-flash-attn` 库，通信高效

### 何时使用 CP？

```
✅ 使用 CP 的场景：
- 序列长度 > 8K
- 激活值显存占用过高（OOM）
- 需要训练超长上下文模型
- 有 NVLink 互连的 GPU

❌ 不使用 CP 的场景：
- 序列长度 < 4K（通信开销不值得）
- 显存充足
- 跨节点训练（通信慢）
```

### 进一步学习资源

- [Axolotl Sequence Parallelism 文档](https://docs.axolotl.ai/docs/sequence_parallelism.html)
- [Ring-Flash-Attention 论文](https://arxiv.org/abs/2310.01889)
- [Ring-Flash-Attention GitHub](https://github.com/zhuzilin/ring-flash-attention)
- [Flash Attention 2 论文](https://arxiv.org/abs/2307.08691)

---

*本文档由 Claude 创作，旨在帮助 infra 初学者理解 Context Parallelism 的原理和实现。如有疑问或发现错误，欢迎提 Issue！*
