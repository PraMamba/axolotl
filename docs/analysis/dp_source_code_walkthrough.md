# Data Parallelism 源码执行流程详解

> 从配置到训练的完整执行流程，结合源码深入理解 FSDP 和 DDP 实现

---

## 目录

1. [执行流程概览](#1-执行流程概览)
2. [配置解析与验证](#2-配置解析与验证)
3. [DeviceMesh 构建](#3-devicemesh-构建)
4. [FSDP 模型准备](#4-fsdp-模型准备)
5. [DDP 配置](#5-ddp-配置)
6. [数据加载与分发](#6-数据加载与分发)
7. [训练循环](#7-训练循环)
8. [梯度同步机制](#8-梯度同步机制)
9. [Checkpoint 保存与加载](#9-checkpoint-保存与加载)
10. [调试与监控](#10-调试与监控)

---

## 1. 执行流程概览

### 1.1 完整调用链

```
用户命令：axolotl train config.yaml
  ↓
src/axolotl/cli/main.py
  ├─ 解析命令行参数
  └─ 调用 do_train()
      ↓
src/axolotl/train.py:do_train()
  ├─ 加载配置 (load_cfg())
  ├─ 构建 DeviceMesh (build_parallelism_config())
  ├─ 加载模型 (load_model())
  ├─ 准备 FSDP (accelerator.prepare())
  ├─ 加载数据集 (load_datasets())
  ├─ 创建 Trainer (builder.build_trainer())
  └─ 开始训练 (trainer.train())
      ↓
transformers/Trainer.train()
  └─ 训练循环（每个 iteration）
      ├─ 前向传播 (model.forward())
      ├─ 反向传播 (loss.backward())
      │   └─ FSDP AllGather + ReduceScatter
      ├─ 梯度累积
      └─ 参数更新 (optimizer.step())
          └─ DDP AllReduce（如果启用）
```

### 1.2 关键文件地图

| 文件路径 | 功能 | 关键类/函数 |
|---------|------|-----------|
| `src/axolotl/train.py` | 训练主入口 | `do_train()` |
| `src/axolotl/utils/distributed.py` | 并行配置构建 | `build_parallelism_config()` (行 299) |
| `src/axolotl/utils/schemas/fsdp.py` | FSDP 配置 Schema | `FSDPConfig` |
| `src/axolotl/core/builders/base.py` | Trainer 构建基类 | `TrainerBuilderBase._set_base_training_args()` (行 461) |
| `src/axolotl/core/builders/causal.py` | Causal Trainer 构建 | `HFCausalTrainerBuilder` |
| `src/axolotl/monkeypatch/accelerate/fsdp2.py` | FSDP2 相关 patch | `fsdp2_load_full_state_dict()` (行 20) |
| `src/axolotl/utils/samplers/multipack.py` | 分布式数据采样 | `MultipackBatchSampler` |
| `src/axolotl/utils/distributed.py` | 分布式工具函数 | `reduce_and_broadcast()`, `gather_scalar_from_all_ranks()` |

---

## 2. 配置解析与验证

### 2.1 配置文件加载

**文件**: `src/axolotl/train.py`

```python
# train.py:do_train()

def do_train(cfg, cli_args):
    # 1. 验证配置
    cfg = validate_config(cfg)

    # 2. 解析 FSDP 配置
    if cfg.fsdp_config:
        # 将 YAML 配置映射到 FSDPConfig Pydantic 模型
        from axolotl.utils.schemas.fsdp import FSDPConfig
        cfg.fsdp_config = FSDPConfig(**cfg.fsdp_config)

    # 3. 计算并行参数
    parallelism_config, device_mesh = build_parallelism_config(cfg)
```

### 2.2 FSDP 配置 Schema

**文件**: `src/axolotl/utils/schemas/fsdp.py`

```python
class FSDPConfig(BaseModel):
    """FSDP 配置 Schema"""

    # === Sharding 策略 ===
    auto_wrap_policy: Literal[
        "TRANSFORMER_BASED_WRAP",
        "SIZE_BASED_WRAP"
    ] | None = None

    transformer_layer_cls_to_wrap: str | None = None
    # 例如: "LlamaDecoderLayer", "MistralDecoderLayer"

    # === 显存优化 ===
    reshard_after_forward: bool | None = None
    # ✅ 推荐开启！前向传播后立即释放参数

    # === CPU Offload ===
    offload_params: bool | None = None
    # 将参数 offload 到 CPU（极端显存不足时）

    # === Checkpoint ===
    state_dict_type: Literal[
        "FULL_STATE_DICT",       # Rank 0 收集完整模型
        "LOCAL_STATE_DICT",      # 每个 rank 保存本地切片
        "SHARDED_STATE_DICT"     # 分片保存
    ] | None = None

    # === 其他 ===
    use_orig_params: bool | None = None
    sync_module_states: bool | None = None
    mixed_precision_policy: str | None = None
```

### 2.3 配置验证逻辑

```python
# distributed.py:_get_parallel_config_kwargs()

def _get_parallel_config_kwargs(
    world_size: int,
    tensor_parallel_size: int = 1,
    context_parallel_size: int = 1,
    dp_shard_size: int | None = None,
    dp_replicate_size: int | None = None,
    is_fsdp: bool = False,
):
    """
    根据 GPU 数量自动推断并行配置

    优先级顺序：
    1. TP (tensor_parallel_size)
    2. CP (context_parallel_size)
    3. DP_Replicate (dp_replicate_size)
    4. DP_Shard (dp_shard_size)
    """
    pc_kwargs = {}
    remaining_world_size = world_size

    # 步骤 1：分配 TP
    if tensor_parallel_size > 1:
        pc_kwargs["tp_size"] = tensor_parallel_size
        remaining_world_size //= tensor_parallel_size

    # 步骤 2：分配 CP
    if context_parallel_size > 1:
        pc_kwargs["cp_size"] = context_parallel_size
        remaining_world_size //= context_parallel_size

    # 步骤 3：分配 DP_Replicate（跨节点 DDP）
    if dp_replicate_size > 1:
        pc_kwargs["dp_replicate_size"] = dp_replicate_size
        remaining_world_size //= dp_replicate_size

    # 步骤 4：分配剩余 GPU 到 DP_Shard（FSDP）
    if remaining_world_size > 1:
        if not is_fsdp:
            raise ValueError(
                "dp_shard_size was configured without a corresponding fsdp_config!"
            )
        pc_kwargs["dp_shard_size"] = remaining_world_size

    # 验证所有 GPU 都已分配
    if remaining_world_size > 1:
        raise ValueError(
            f"Parallelism config incompatible with world size {world_size}!"
        )

    return pc_kwargs
```

#### 示例推断

```python
# 示例 1：8 GPU，仅配置 TP=2
world_size = 8
tensor_parallel_size = 2
is_fsdp = True

# 推断结果：
# TP = 2
# remaining = 8 / 2 = 4
# DP_Shard = 4（自动推断）
# 最终：2 × 4 = 8 ✅

# 示例 2：16 GPU，TP=2, dp_shard=4, dp_replicate=2
world_size = 16
tensor_parallel_size = 2
dp_shard_size = 4
dp_replicate_size = 2
is_fsdp = True

# 推断结果：
# TP = 2
# DP_Replicate = 2
# DP_Shard = 4
# 最终：2 × 4 × 2 = 16 ✅
```

---

## 3. DeviceMesh 构建

### 3.1 构建并行配置

**文件**: `src/axolotl/utils/distributed.py:299`

```python
def build_parallelism_config(cfg):
    """构建并行配置和 DeviceMesh"""

    # 1. 获取并行参数
    pc_kwargs = _get_parallel_config_kwargs(
        get_world_size(),
        cfg.tensor_parallel_size,
        cfg.context_parallel_size,
        cfg.dp_shard_size,
        cfg.dp_replicate_size,
        bool(cfg.fsdp or cfg.fsdp_config),
    )

    if pc_kwargs:
        # 2. 创建 ParallelismConfig
        from torch.distributed.pipelining import ParallelismConfig

        parallelism_config = ParallelismConfig(**pc_kwargs)

        # 3. 构建 DeviceMesh
        device_mesh = parallelism_config.build_device_mesh("cuda")

        return parallelism_config, device_mesh

    return None, None
```

### 3.2 DeviceMesh 结构

```python
# 示例：8 GPU, TP=2, DP_Shard=4

device_mesh = DeviceMesh(
    device_type="cuda",
    mesh=[
        # TP 维度 →
        [0, 1],  # DP_Shard 维度 ↓
        [2, 3],
        [4, 5],
        [6, 7],
    ],
    mesh_dim_names=["tp", "dp_shard"],
)

# 查询示例：
# Rank 0 的坐标: (tp=0, dp_shard=0)
# Rank 5 的坐标: (tp=1, dp_shard=2)

# TP 组（模型并行）:
#   Group 0: [0, 1]
#   Group 1: [2, 3]
#   Group 2: [4, 5]
#   Group 3: [6, 7]

# FSDP 组（数据并行）:
#   Group 0: [0, 2, 4, 6]
#   Group 1: [1, 3, 5, 7]
```

---

## 4. FSDP 模型准备

### 4.1 训练参数配置

**文件**: `src/axolotl/core/builders/base.py:461`

```python
def _set_base_training_args(self, total_num_steps):
    """配置基础训练参数"""
    training_args_kwargs = {}
    trainer_kwargs = {}

    # ... 其他配置 ...

    # === FSDP 配置传递 ===
    if self.cfg.fsdp_config or self.cfg.fsdp:
        training_args_kwargs["fsdp_config"] = self.cfg.fsdp_config
        training_args_kwargs["fsdp"] = self.cfg.fsdp if self.cfg.fsdp else True

    # Batch size
    training_args_kwargs["per_device_train_batch_size"] = self.cfg.micro_batch_size

    # 梯度累积
    training_args_kwargs["gradient_accumulation_steps"] = (
        self.cfg.gradient_accumulation_steps
    )

    return training_args_kwargs, trainer_kwargs
```

### 4.2 FSDP 模型包装

FSDP 包装由 HuggingFace Accelerate 自动处理：

```python
# train.py:do_train()

from accelerate import Accelerator

accelerator = Accelerator(
    fsdp_plugin=...,  # 从 training_args 传递
)

# 模型包装
model = accelerator.prepare(model)

# Accelerate 内部做了什么？
# 1. 识别 FSDP 配置
# 2. 根据 auto_wrap_policy 切分模型
# 3. 为每一层应用 FSDP 包装
# 4. 分片参数/梯度/优化器状态到各个 GPU
```

### 4.3 FSDP2 参数加载

**文件**: `src/axolotl/monkeypatch/accelerate/fsdp2.py:20`

```python
def fsdp2_load_full_state_dict(
    _accelerator,
    model: torch.nn.Module,
    full_sd: dict,
    offload_to_cpu: bool = False
):
    """
    将完整的 state dict（仅在 rank 0）加载到分片模型

    执行流程：
    1. Rank 0 持有完整的 checkpoint
    2. 遍历每个参数：
       - Rank 0: 准备完整参数
       - 其他 ranks: 创建空 tensor
       - distribute_tensor(): 根据 DeviceMesh 分发参数
    3. 每个 rank 只保留自己的切片
    """
    from torch.distributed.tensor import distribute_tensor

    # 1. 获取模型的分片 state dict（meta 状态）
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

    for param_name, sharded_meta_param in meta_sharded_sd.items():
        # 2. Rank 0 准备完整参数
        full_tensor = None
        if _accelerator.is_main_process:
            full_tensor = full_sd[param_name]
            full_tensor = full_tensor.to(sharded_meta_param.dtype)

        # 3. 检查参数是否是 DTensor（分片）
        if hasattr(sharded_meta_param, "device_mesh"):
            device_mesh = sharded_meta_param.device_mesh

            # 其他 ranks 创建空 tensor
            if not _accelerator.is_main_process:
                full_tensor = torch.empty(
                    sharded_meta_param.size(),
                    device=device_mesh.device_type,
                    dtype=sharded_meta_param.dtype,
                )

            # 4. 分发参数（从 rank 0）
            sharded_param = distribute_tensor(
                full_tensor,
                device_mesh,
                sharded_meta_param.placements,  # 分片策略
                src_data_rank=0,
            )
        else:
            # 非分片参数（广播到所有 ranks）
            if not _accelerator.is_main_process:
                sharded_param = torch.empty_like(sharded_meta_param)
            else:
                sharded_param = full_tensor.to(torch.device("cuda"))

            dist.broadcast(sharded_param, src=0)

        # 5. 可选：offload 到 CPU
        if offload_to_cpu:
            sharded_param = sharded_param.cpu()

        sharded_sd[param_name] = nn.Parameter(sharded_param)

    # 6. 加载到模型
    model.load_state_dict(sharded_sd, assign=True, strict=True)
    return model
```

---

## 5. DDP 配置

### 5.1 何时启用 DDP？

DDP 在以下情况自动启用：

```python
# 条件 1：dp_replicate_size > 1（多节点）
dp_replicate_size: 2

# 条件 2：未启用 FSDP 且有多个 GPU
# 如果没有配置 fsdp_config，且 world_size > 1
# Accelerate 会自动使用 DDP
```

### 5.2 DDP 模型包装

```python
# 由 Accelerate 自动处理

from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    broadcast_buffers=True,
    find_unused_parameters=False,  # Axolotl 默认 False（性能更好）
)

# DDP 做了什么？
# 1. 在 backward() 后自动执行梯度 AllReduce
# 2. 使用 bucketing 优化通信（按组聚合小梯度）
# 3. 支持梯度压缩（可选）
```

---

## 6. 数据加载与分发

### 6.1 MultipackBatchSampler

**文件**: `src/axolotl/utils/samplers/multipack.py`

Axolotl 使用 `MultipackBatchSampler` 高效打包变长序列：

```python
class MultipackBatchSampler(BatchSampler):
    """
    高效的变长序列 batch sampler

    核心思想：
    - 使用 First-Fit Decreasing (FFD) 算法打包序列
    - 确保每个 batch 的总 token 数 ≤ batch_max_len
    - 最大化 GPU 利用率（减少 padding）
    """

    def __init__(
        self,
        batch_max_len: int,  # 例如: micro_batch_size × sequence_len
        lengths: list[int],  # 每个样本的长度
        sampler: Sampler | None = None,
        # ...
    ):
        self.batch_max_len = batch_max_len
        self.lengths = lengths

    def __iter__(self):
        # 1. 排序序列（降序）
        sorted_indices = np.argsort(self.lengths)[::-1]

        # 2. FFD 打包
        bins = []  # 每个 bin 是一个 batch
        for idx in sorted_indices:
            seq_len = self.lengths[idx]

            # 尝试放入现有 bin
            placed = False
            for bin in bins:
                if bin.remaining_space >= seq_len:
                    bin.add(idx)
                    placed = True
                    break

            # 创建新 bin
            if not placed:
                bins.append(Bin(self.batch_max_len))
                bins[-1].add(idx)

        # 3. 返回 batch indices
        for bin in bins:
            yield bin.indices
```

### 6.2 分布式采样

```python
# 在分布式训练中，每个 GPU 需要不同的数据

from torch.utils.data.distributed import DistributedSampler

# Axolotl 内部使用
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # 总 GPU 数
    rank=global_rank,         # 当前 GPU rank
    shuffle=True,
    seed=cfg.seed,
)

# DistributedSampler 做了什么？
# 1. 将数据集切分成 world_size 份
# 2. 每个 rank 只访问自己的那一份
# 3. 确保每个 epoch 的数据顺序不同（shuffle）

# 示例：8 个样本，4 个 GPU
# Rank 0: [样本 0, 样本 4]
# Rank 1: [样本 1, 样本 5]
# Rank 2: [样本 2, 样本 6]
# Rank 3: [样本 3, 样本 7]
```

### 6.3 数据加载器创建

```python
# 完整的数据加载器创建（简化版）

def get_dataloader(cfg, dataset, sampler=None):
    """创建分布式数据加载器"""

    if sampler is None:
        # 自动创建分布式采样器
        if is_distributed():
            sampler = DistributedSampler(dataset, ...)
        else:
            sampler = RandomSampler(dataset)

    # 使用 MultipackBatchSampler
    batch_sampler = MultipackBatchSampler(
        batch_max_len=cfg.micro_batch_size * cfg.sequence_len,
        lengths=dataset.lengths,
        sampler=sampler,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.dataloader_num_workers or 4,
        pin_memory=cfg.dataloader_pin_memory or True,
        collate_fn=collate_fn,
    )

    return dataloader
```

---

## 7. 训练循环

### 7.1 Trainer 训练循环（简化）

```python
# transformers/Trainer.train()（概念性简化）

def train(self):
    model.train()

    for epoch in range(num_epochs):
        # 设置 epoch（重要！确保每个 epoch 数据不同）
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            # === 1. 数据移到 GPU ===
            batch = {k: v.to(device) for k, v in batch.items()}

            # === 2. 前向传播 ===
            with autocast():  # 混合精度
                outputs = model(**batch)
                loss = outputs.loss

            # 梯度累积归一化
            loss = loss / gradient_accumulation_steps

            # === 3. 反向传播 ===
            accelerator.backward(loss)

            # ← FSDP 在这里执行 AllGather + ReduceScatter

            # === 4. 梯度累积 ===
            if (step + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm
                    )

                # === 5. 参数更新 ===
                optimizer.step()

                # ← DDP 在这里执行梯度 AllReduce（如果启用）

                # 清空梯度
                optimizer.zero_grad()

                # 学习率调度
                scheduler.step()

            # === 6. 日志记录 ===
            if step % logging_steps == 0:
                log_metrics(loss, lr, ...)

            # === 7. Checkpoint 保存 ===
            if step % save_steps == 0:
                save_checkpoint(model, optimizer, ...)
```

### 7.2 FSDP 前向传播详解

```python
# FSDP 前向传播（每一层）

class FSDPLayer(nn.Module):
    def forward(self, input):
        # 1. AllGather 参数（从所有 ranks 收集完整参数）
        with torch.distributed.fsdp.fully_shard.summon_full_params(self):
            # self.weight 现在是完整的参数
            output = F.linear(input, self.weight, self.bias)

        # 2. 自动释放不属于当前 rank 的参数
        # （如果开启 reshard_after_forward）
        # self.weight 现在只保留当前 rank 的切片

        return output

# 通信流程：
# Step 1: AllGather
#   Rank 0: 发送 shard_0 → 接收 [shard_1, shard_2, shard_3]
#   Rank 1: 发送 shard_1 → 接收 [shard_0, shard_2, shard_3]
#   Rank 2: 发送 shard_2 → 接收 [shard_0, shard_1, shard_3]
#   Rank 3: 发送 shard_3 → 接收 [shard_0, shard_1, shard_2]

# Step 2: Compute（使用完整参数）
#   所有 ranks: output = input @ full_weight

# Step 3: Reshard（释放其他 ranks 的参数）
#   Rank 0: 只保留 shard_0
#   Rank 1: 只保留 shard_1
#   ...
```

### 7.3 FSDP 反向传播详解

```python
# FSDP 反向传播（每一层）

class FSDPLayer(nn.Module):
    def backward(self, grad_output):
        # 1. AllGather 参数（再次收集完整参数）
        with torch.distributed.fsdp.fully_shard.summon_full_params(self):
            # 2. 计算梯度
            grad_input = grad_output @ self.weight.T
            grad_weight = grad_output.T @ input

        # 3. ReduceScatter 梯度（聚合并分片）
        # 每个 rank 只保留自己的梯度切片
        self.weight.grad = reduce_scatter(grad_weight)
        # Rank 0: 保留 grad_weight[0:N/4]
        # Rank 1: 保留 grad_weight[N/4:N/2]
        # ...

        return grad_input

# 通信流程：
# Step 1: AllGather（同前向）
#   收集完整参数用于梯度计算

# Step 2: Compute
#   计算完整的 grad_weight

# Step 3: ReduceScatter
#   Rank 0: 发送 grad_weight → 接收所有 ranks 的 shard_0 求和
#   Rank 1: 发送 grad_weight → 接收所有 ranks 的 shard_1 求和
#   ...
```

---

## 8. 梯度同步机制

### 8.1 FSDP 梯度同步

FSDP 使用 **ReduceScatter** 操作：

```python
# ReduceScatter 伪代码

def reduce_scatter(grad_tensor):
    """
    将梯度求和并分片

    输入（每个 rank）：
      Rank 0: grad = [g0_0, g0_1, g0_2, g0_3]
      Rank 1: grad = [g1_0, g1_1, g1_2, g1_3]
      Rank 2: grad = [g2_0, g2_1, g2_2, g2_3]
      Rank 3: grad = [g3_0, g3_1, g3_2, g3_3]

    输出（每个 rank 只保留一部分）：
      Rank 0: [sum(g*_0)] = [g0_0 + g1_0 + g2_0 + g3_0]
      Rank 1: [sum(g*_1)] = [g0_1 + g1_1 + g2_1 + g3_1]
      Rank 2: [sum(g*_2)] = [g0_2 + g1_2 + g2_2 + g3_2]
      Rank 3: [sum(g*_3)] = [g0_3 + g1_3 + g2_3 + g3_3]
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    chunk_size = len(grad_tensor) // world_size

    # 1. 切分梯度
    chunks = grad_tensor.chunk(world_size)

    # 2. 每个 rank 收集同一个 chunk 的所有副本并求和
    result = torch.zeros_like(chunks[rank])
    for r in range(world_size):
        # 通信：收集 chunk[rank] 从所有 ranks
        chunk_from_rank_r = recv_from(r, chunks[rank])
        result += chunk_from_rank_r

    return result
```

### 8.2 DDP 梯度同步

DDP 使用 **AllReduce** 操作：

```python
# AllReduce 伪代码

def all_reduce(grad_tensor):
    """
    计算所有 ranks 的梯度平均值，并广播给所有 ranks

    输入（每个 rank）：
      Rank 0: grad_0
      Rank 1: grad_1
      Rank 2: grad_2
      Rank 3: grad_3

    输出（所有 ranks 相同）：
      All ranks: (grad_0 + grad_1 + grad_2 + grad_3) / 4
    """
    world_size = dist.get_world_size()

    # 1. 求和
    sum_grad = sum_across_ranks(grad_tensor)

    # 2. 平均
    avg_grad = sum_grad / world_size

    # 3. 广播给所有 ranks
    return avg_grad
```

### 8.3 分布式工具函数

**文件**: `src/axolotl/utils/distributed.py`

```python
def reduce_and_broadcast(value, device=None):
    """
    在所有 ranks 之间求和并广播结果

    用于：同步统计量（如 loss、准确率）
    """
    if not dist.is_initialized():
        return value

    if device is None:
        device = torch.cuda.current_device()

    # 转换为 tensor
    tensor = torch.tensor(value, device=device)

    # AllReduce（求和）
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor.item()


def gather_scalar_from_all_ranks(scalar_value, device=None):
    """
    从所有 ranks 收集标量值到 rank 0

    用于：收集每个 rank 的指标
    """
    if not dist.is_initialized():
        return [scalar_value]

    if device is None:
        device = torch.cuda.current_device()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 创建 tensor
    local_tensor = torch.tensor([scalar_value], device=device)

    if rank == 0:
        # Rank 0 收集所有值
        gathered_tensors = [
            torch.zeros(1, device=device) for _ in range(world_size)
        ]
        dist.gather(local_tensor, gathered_tensors, dst=0)
        return [t.item() for t in gathered_tensors]
    else:
        # 其他 ranks 发送值
        dist.gather(local_tensor, dst=0)
        return None
```

---

## 9. Checkpoint 保存与加载

### 9.1 FSDP FULL_STATE_DICT 保存

```python
# accelerate/fsdp2.py:get_state_dict()

def save_checkpoint_fsdp(model, save_path):
    """
    FSDP FULL_STATE_DICT 保存流程

    1. Rank 0 收集所有参数
    2. Rank 0 保存完整 checkpoint
    3. 其他 ranks 等待（barrier）
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    # 配置 FULL_STATE_DICT
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()

    # Rank 0 保存
    if dist.get_rank() == 0:
        torch.save(state_dict, save_path)
        LOG.info(f"Checkpoint saved to {save_path}")

    # 同步所有 ranks
    dist.barrier()
```

### 9.2 FSDP SHARDED_STATE_DICT 保存

```python
def save_checkpoint_fsdp_sharded(model, save_dir):
    """
    FSDP SHARDED_STATE_DICT 保存流程

    1. 每个 rank 保存自己的参数切片
    2. 保存到不同的文件
    """
    from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

    rank = dist.get_rank()

    # 配置 SHARDED_STATE_DICT
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(),
    ):
        state_dict = model.state_dict()

    # 每个 rank 保存自己的切片
    shard_path = os.path.join(save_dir, f"model_shard_rank{rank}.pt")
    torch.save(state_dict, shard_path)
    LOG.info(f"Rank {rank} saved shard to {shard_path}")

    dist.barrier()
```

### 9.3 Checkpoint 加载

```python
def load_checkpoint_fsdp(model, checkpoint_path):
    """
    加载 FSDP checkpoint

    根据 checkpoint 类型（FULL vs SHARDED）采取不同策略
    """
    from axolotl.monkeypatch.accelerate.fsdp2 import fsdp2_load_full_state_dict

    # 1. Rank 0 加载 checkpoint
    if dist.get_rank() == 0:
        full_sd = torch.load(checkpoint_path, map_location="cpu")
    else:
        full_sd = None

    # 2. 分发到所有 ranks
    model = fsdp2_load_full_state_dict(
        accelerator,
        model,
        full_sd,
        offload_to_cpu=False,
    )

    LOG.info("Checkpoint loaded successfully")
    return model
```

---

## 10. 调试与监控

### 10.1 显存监控

```python
# axolotl/utils/bench.py

def log_gpu_memory_usage(logger, message, device=0):
    """记录 GPU 显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        logger.info(
            f"{message} - "
            f"Allocated: {allocated:.2f}GB, "
            f"Reserved: {reserved:.2f}GB"
        )

# 使用
log_gpu_memory_usage(LOG, "After model loading", 0)
log_gpu_memory_usage(LOG, "After FSDP preparation", 0)
log_gpu_memory_usage(LOG, "After first forward", 0)
```

### 10.2 通信性能分析

```bash
# 1. 启用 NCCL 调试日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL  # 只显示集合通信

axolotl train config.yaml

# 观察输出：
# NCCL INFO Ring 00 : 0 -> 1 -> 2 -> 3 -> 0
# NCCL INFO AllReduce: 14GB in 93ms (150 GB/s)

# 2. 使用 nsys 性能分析
nsys profile -o profile.qdrep \
    python -m axolotl.cli.train config.yaml --max-steps 10

# 3. 可视化
nsys-ui profile.qdrep
# 查看 NCCL kernels 和 通信时间
```

### 10.3 梯度检查

```python
# 添加梯度检查回调

class GradientMonitorCallback:
    def on_after_backward(self, trainer):
        """检查梯度异常"""
        model = trainer.model

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()

                # 检查 NaN/Inf
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    LOG.warning(
                        f"异常梯度: {name}, "
                        f"norm={grad_norm}"
                    )

                # 检查梯度爆炸
                if grad_norm > 1000:
                    LOG.warning(
                        f"梯度过大: {name}, "
                        f"norm={grad_norm}"
                    )
```

### 10.4 分布式同步检查

```python
# 检查所有 ranks 是否同步

def check_model_sync(model):
    """验证所有 ranks 的模型参数是否一致"""
    import torch.distributed as dist

    if not dist.is_initialized():
        return

    rank = dist.get_rank()

    for name, param in model.named_parameters():
        # 计算本地参数的 hash
        local_hash = hash(param.data.cpu().numpy().tobytes())

        # 收集所有 ranks 的 hash
        all_hashes = [None] * dist.get_world_size()
        dist.all_gather_object(all_hashes, local_hash)

        # Rank 0 检查一致性
        if rank == 0:
            if len(set(all_hashes)) > 1:
                LOG.error(
                    f"参数不同步: {name}, "
                    f"hashes={all_hashes}"
                )
            else:
                LOG.debug(f"参数同步 ✅: {name}")
```

---

## 总结

### 关键执行流程

1. **配置解析** → `_get_parallel_config_kwargs()`
2. **DeviceMesh 构建** → `ParallelismConfig.build_device_mesh()`
3. **模型包装** → `accelerator.prepare(model)`（FSDP/DDP）
4. **数据分发** → `MultipackBatchSampler` + `DistributedSampler`
5. **训练循环**:
   - 前向：FSDP AllGather 参数
   - 反向：FSDP ReduceScatter 梯度
   - 更新：DDP AllReduce 梯度（如果启用）
6. **Checkpoint** → FULL_STATE_DICT 或 SHARDED_STATE_DICT

### 关键代码位置

| 功能 | 文件 | 行数 |
|------|------|------|
| 并行配置构建 | `src/axolotl/utils/distributed.py` | 299-370 |
| FSDP 配置 Schema | `src/axolotl/utils/schemas/fsdp.py` | 10-72 |
| 训练参数设置 | `src/axolotl/core/builders/base.py` | 461-544 |
| FSDP2 checkpoint 加载 | `src/axolotl/monkeypatch/accelerate/fsdp2.py` | 20-91 |
| Multipack 采样 | `src/axolotl/utils/samplers/multipack.py` | 24-150 |
| 分布式工具 | `src/axolotl/utils/distributed.py` | 全文 |

---

## 下一步

- 快速参考卡片 → [dp_quick_reference.md](./dp_quick_reference.md)
- 返回深度解析 → [data_parallelism_deep_dive.md](./data_parallelism_deep_dive.md)
- 返回主索引 → [README.md](./README.md)

---

*本文档由 Claude AI 辅助创作，旨在帮助开发者深入理解 Axolotl 的 Data Parallelism 源码实现。*
