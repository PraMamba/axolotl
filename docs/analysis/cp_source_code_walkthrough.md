# Context Parallelism 源码执行流程详解

> 本文档详细追踪 Axolotl 从配置读取到 Ring-Flash-Attention 执行的完整 CP 实现流程

## 执行流程概览

```
用户执行: axolotl train config.yaml
    ↓
1. CLI 入口与配置解析
    ↓
2. 并行配置构建（包含 CP 维度）
    ↓
3. DeviceMesh 创建（CP 子网格）
    ↓
4. Ring Attention 组注册
    ↓
5. Flash Attention 替换为 Ring-Flash-Attention
    ↓
6. SequenceParallelContextManager 初始化
    ↓
7. 训练循环开始
    ↓
8. Pre-Forward Hook：序列切分
    ↓
9. 模型前向传播（Ring-Flash-Attention）
    ↓
10. Post-Forward Hook：输出聚合
    ↓
11. 反向传播（梯度自动切分）
```

---

## 第一阶段：配置解析与初始化

### 1.1 配置文件解析

```yaml
# config.yaml
base_model: meta-llama/Llama-3.1-8B
context_parallel_size: 4
sequence_len: 16384
flash_attention: true
ring_attn_func: varlen_llama3
heads_k_stride: 1
```

Axolotl 读取配置并验证：

```python
# 文件：src/axolotl/utils/schemas/validation.py

class AxolotlConfigValidator:
    def validate_context_parallel(self, cfg):
        """验证 CP 配置"""

        # 1. 检查 Flash Attention
        if cfg.get("context_parallel_size", 1) > 1:
            if not cfg.get("flash_attention"):
                raise ValueError(
                    "context_parallel_size > 1 需要启用 flash_attention"
                )

        # 2. 检查 batch size
        if cfg.get("context_parallel_size", 1) > 1:
            # varlen 实现要求 micro_batch_size = 1
            if cfg.get("ring_attn_func") == "varlen_llama3":
                if cfg.get("micro_batch_size", 1) > 1:
                    raise ValueError(
                        "varlen_llama3 requires micro_batch_size = 1"
                    )

        # 3. 检查 GPU 数量匹配
        world_size = get_world_size()
        cp_size = cfg.get("context_parallel_size", 1)

        if world_size % cp_size != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"context_parallel_size ({cp_size})"
            )

        return True
```

### 1.2 DeviceMesh 构建（包含 CP 维度）

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
    """计算并行配置参数"""

    pc_kwargs = {}
    remaining_world_size = world_size

    # 1. 先分配 TP
    if tensor_parallel_size and tensor_parallel_size > 1:
        pc_kwargs["tp_size"] = tensor_parallel_size
        remaining_world_size = remaining_world_size // tensor_parallel_size

    # 2. 再分配 CP（关键！）
    if context_parallel_size and context_parallel_size > 1:
        pc_kwargs["cp_size"] = context_parallel_size
        remaining_world_size = remaining_world_size // context_parallel_size
        # 例：8 GPUs, TP=2, CP=2 → remaining = 8 / 2 / 2 = 2

    # 3. 分配 DDP
    if dp_replicate_size and dp_replicate_size > 1:
        pc_kwargs["dp_replicate_size"] = dp_replicate_size
        remaining_world_size = remaining_world_size // dp_replicate_size

    # 4. 分配 FSDP
    if dp_shard_size and dp_shard_size > 1:
        pc_kwargs["dp_shard_size"] = dp_shard_size
        remaining_world_size = remaining_world_size // dp_shard_size

    # 5. 验证所有 GPU 都分配完毕
    if remaining_world_size != 1:
        raise ValueError(
            f"配置的并行度与 GPU 总数不匹配！\n"
            f"配置: {pc_kwargs}, 总 GPU: {world_size}"
        )

    return pc_kwargs
```

**DeviceMesh 创建**：

```python
# 文件：src/axolotl/utils/distributed.py (299-316 行)

def build_parallelism_config(cfg):
    pc_kwargs = _get_parallel_config_kwargs(...)

    if pc_kwargs:
        # 创建 ParallelismConfig
        parallelism_config = ParallelismConfig(**pc_kwargs)

        # 构建 DeviceMesh
        # 例：{"tp_size": 2, "cp_size": 2, "dp_shard_size": 2}
        device_mesh = parallelism_config.build_device_mesh("cuda")

        return parallelism_config, device_mesh

    return None, None
```

**DeviceMesh 示例**（8 GPUs, TP=2, CP=2, FSDP=2）：

```python
# 生成的 DeviceMesh 结构：
device_mesh = DeviceMesh(
    "cuda",
    mesh=[
        # FSDP shard 0
        [
            [[0, 1],   # TP group 0, CP rank 0
             [2, 3]],  # TP group 1, CP rank 0
        ],
        # FSDP shard 1
        [
            [[4, 5],   # TP group 0, CP rank 1
             [6, 7]],  # TP group 1, CP rank 1
        ],
    ],
    mesh_dim_names=["dp_shard", "cp", "tp"]
)

# 提取 CP 子网格
cp_mesh = device_mesh["cp"]

# GPU 0 的视角：
# - CP 组: [GPU 0/1, GPU 4/5] (跨 TP 的同列)
# - TP 组: [GPU 0, GPU 1]
# - FSDP 组: [GPU 0/1, GPU 2/3] (同一 CP rank)
```

---

## 第二阶段：Ring Attention 初始化

### 2.1 注册 Ring Attention 组

```python
# 文件：src/axolotl/utils/ctx_managers/sequence_parallel.py (207-228 行)

class SequenceParallelContextManager:
    def __init__(
        self,
        models: list[nn.Module],
        context_parallel_size: int,
        gradient_accumulation_steps: int,
        ring_attn_func: RingAttnFunc,
        heads_k_stride: int | None,
        gather_outputs: bool,
        device_mesh: DeviceMesh | None = None,
    ):
        self.models = models
        self.context_parallel_size = context_parallel_size
        self.ring_attn_func = ring_attn_func
        self.heads_k_stride = heads_k_stride
        self.device_mesh = device_mesh

        # === 关键：注册 ring attention ===
        self._register_ring_attn()

        # 获取 CP 进程组
        self.process_group = get_ring_attn_group()
        self.local_rank = dist.get_rank(self.process_group)
        self.local_world_size = dist.get_world_size(self.process_group)
```

```python
# 文件：src/axolotl/utils/ctx_managers/sequence_parallel.py (243-250 行)

def _register_ring_attn(self):
    """初始化 ring attention 序列并行"""

    register_ring_attn_from_device_mesh(
        device_mesh=self.device_mesh,
        context_parallel_dim=("cp",),  # ← CP 维度名称
        heads_k_stride=self.heads_k_stride,
        ring_attn_func=self.ring_attn_func,
    )
```

### 2.2 提取 CP 子网格并创建进程组

```python
# 文件：src/axolotl/monkeypatch/ring_attn/patch.py (135-186 行)

def register_ring_attn_from_device_mesh(
    device_mesh: DeviceMesh,
    context_parallel_dim: tuple[str, ...],  # ("cp",)
    heads_k_stride: int | None,
    ring_attn_func: RingAttnFunc | None,
):
    """使用 DeviceMesh 创建 ring attention 组"""

    rank = dist.get_rank()

    LOG.info(
        f"启用 ring attention 序列并行，使用 DeviceMesh 维度 '{context_parallel_dim}'",
        main_process_only=True,
    )

    # === 1. 提取 CP 子网格 ===
    try:
        sequence_mesh = device_mesh[context_parallel_dim]
        # 例：device_mesh["cp"] 提取 CP 维度
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"维度 '{context_parallel_dim}' 在 device_mesh 中不存在。"
            f"可用维度: {device_mesh.mesh_dim_names}"
        ) from e

    # === 2. 获取 CP 进程组 ===
    sequence_pg = sequence_mesh.get_group()
    context_parallel_size = sequence_mesh.size()

    if rank == 0:
        LOG.info(
            f"序列并行度: {context_parallel_size}, "
            f"mesh 形状: {sequence_mesh.mesh.shape}"
        )

    # === 3. 记录进程组成员 ===
    if sequence_pg != dist.GroupMember.WORLD:
        ranks_in_group = dist.get_process_group_ranks(sequence_pg)
        LOG.info(f"当前序列并行组 ranks: {ranks_in_group}")

    # === 4. 设置全局 ring attention 组 ===
    set_ring_attn_group(sequence_pg)

    # === 5. 替换 Flash Attention 实现 ===
    if ring_attn_func is RingAttnFunc.VARLEN_LLAMA3:
        # 使用 varlen 实现（支持 sample packing）
        _substitute_varlen_flash_attn(sequence_pg, heads_k_stride)
    elif ring_attn_func is RingAttnFunc.BATCH_RING:
        # 使用 batch ring 实现
        _substitute_batch_ring_flash_attn(sequence_pg)
```

**进程组示例**（8 GPUs, TP=2, CP=2, FSDP=2）：

```python
# CP 组划分（每组 4 个 GPU）
CP_Group_0 = [GPU 0, GPU 1, GPU 4, GPU 5]  # CP rank 0
CP_Group_1 = [GPU 2, GPU 3, GPU 6, GPU 7]  # CP rank 1

# 每个 CP 组内的 Ring 拓扑：
# GPU 0 → GPU 1 → GPU 4 → GPU 5 → GPU 0 (环形)
# GPU 2 → GPU 3 → GPU 6 → GPU 7 → GPU 2 (环形)

# 注意：TP 组（如 GPU 0-1）在同一 CP 组内
# 它们协作处理同一序列的同一部分
```

### 2.3 替换 Flash Attention 实现

```python
# 文件：src/axolotl/monkeypatch/ring_attn/patch.py (187-213 行)

def _substitute_varlen_flash_attn(process_group, heads_k_stride):
    """替换为 varlen ring-flash-attention"""

    # === 1. 创建自定义的 flash attention 前向函数 ===
    from ring_flash_attn.adapters.hf_adapter import (
        create_ring_flash_attention_forward as create_original,
    )

    # 用我们的实现替换原始实现
    custom_forward = create_ring_flash_attention_forward(
        process_group=process_group,
        heads_k_stride=heads_k_stride or 1
    )

    # === 2. Monkey-patch ring-flash-attn 库 ===
    import ring_flash_attn.adapters.hf_adapter
    ring_flash_attn.adapters.hf_adapter.create_ring_flash_attention_forward = (
        lambda pg, stride: custom_forward
    )

    # === 3. 替换 HuggingFace 的 Flash Attention ===
    ring_flash_attn.adapters.hf_adapter.substitute_hf_flash_attn(
        process_group=process_group,
        heads_k_stride=heads_k_stride or 1
    )

    LOG.info("✅ 已将 Flash Attention 替换为 Ring-Flash-Attention (varlen)")
```

**substitute_hf_flash_attn 内部**（ring-flash-attn 库）：

```python
# 伪代码：ring-flash-attn 库的实现

def substitute_hf_flash_attn(process_group, heads_k_stride):
    """替换 transformers 库中的 Flash Attention"""

    # 1. 找到所有使用 Flash Attention 的模块
    # 例如：LlamaFlashAttention2, Qwen2FlashAttention2, etc.

    # 2. 替换它们的 _flash_attention_forward 方法
    for model_class in [LlamaFlashAttention2, Qwen2FlashAttention2, ...]:
        # 保存原始方法
        model_class._original_flash_attention_forward = (
            model_class._flash_attention_forward
        )

        # 替换为 ring-flash-attention
        model_class._flash_attention_forward = (
            create_ring_flash_attention_forward(process_group, heads_k_stride)[0]
        )

    LOG.info("✅ Flash Attention 已全局替换")
```

---

## 第三阶段：训练循环与序列切分

### 3.1 SequenceParallelContextManager 上下文

```python
# 文件：src/axolotl/train.py (核心训练逻辑)

def train():
    # 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # 创建 trainer
    trainer_builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
    trainer = trainer_builder.build()

    # === 如果启用 CP，包装 trainer ===
    if cfg.context_parallel_size > 1:
        # 创建 SequenceParallelContextManager
        sp_manager = SequenceParallelContextManager(
            models=[trainer.model],
            context_parallel_size=cfg.context_parallel_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            ring_attn_func=cfg.ring_attn_func,
            heads_k_stride=cfg.heads_k_stride,
            gather_outputs=True,
            device_mesh=model.device_mesh,
        )

        # 使用上下文管理器
        with sp_manager:
            trainer.train()
    else:
        # 标准训练
        trainer.train()
```

**SequenceParallelContextManager 的作用**：
- `__enter__` 时：注册 pre-forward 和 post-forward hooks
- `__exit__` 时：移除 hooks

### 3.2 注册前向传播 Hooks

```python
# 文件：src/axolotl/utils/ctx_managers/sequence_parallel.py (230-240, 252-300 行)

class SequenceParallelContextManager:
    def __enter__(self):
        """进入上下文：注册 hooks"""
        self._register_model_hooks()
        return self

    def _register_model_hooks(self):
        """注册 pre-forward 和 post-forward hooks"""

        # === Pre-Forward Hook：切分序列 ===
        def sequence_parallel_pre_hook(_, args, kwargs):
            """前向传播前：切分输入序列"""

            # 1. 获取模型 forward 的参数名
            forward_params = list(
                inspect.signature(self.models[0].forward).parameters.keys()
            )

            # 2. 将 args 转换为 kwargs
            updated_kwargs = kwargs.copy()
            for i, arg in enumerate(args):
                if i < len(forward_params):
                    updated_kwargs[forward_params[i]] = arg

            # 3. 应用序列并行切分
            updated_kwargs, self.original_seq_len, self.pad_len = (
                self.apply_sequence_parallelism(updated_kwargs)
            )
            # ↑ 核心！在这里切分 input_ids, labels, etc.

            return (), updated_kwargs  # 返回空 args，所有参数在 kwargs

        # === Post-Forward Hook：聚合输出 ===
        def sequence_parallel_post_hook(_, __, output: ModelOutput):
            """前向传播后：聚合输出"""

            # 1. 收集所有 GPU 的输出
            output = self._gather_outputs(output)

            # 2. 移除之前添加的 padding
            if self.pad_len > 0:
                for key, value in output.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 1:
                        if value.size(1) == self.original_seq_len + self.pad_len:
                            # 去掉 padding
                            output[key] = value[:, : self.original_seq_len].contiguous()

            return output

        # === 注册 hooks ===
        for model in self.models:
            # Pre-forward hook
            self.hook_handles.append(
                model.register_forward_pre_hook(
                    sequence_parallel_pre_hook,
                    with_kwargs=True
                )
            )

            # Post-forward hook
            if self.gather_outputs:
                self.hook_handles.append(
                    model.register_forward_hook(sequence_parallel_post_hook)
                )
```

### 3.3 序列切分详细流程

```python
# 文件：src/axolotl/utils/ctx_managers/sequence_parallel.py (24-167 行)

def apply_sequence_parallelism(
    batch: dict[str, torch.Tensor],
    local_rank: int,
    local_world_size: int,
    gradient_accumulation_steps: int,
    ring_attn_func: RingAttnFunc,
):
    """应用序列并行切分"""

    batch_size, original_seq_len = batch["input_ids"].shape
    # 例：batch_size=1, original_seq_len=16384

    # === 步骤 1：处理 position_ids ===
    if batch.get("position_ids") is not None and batch_size == 1:
        # Sample packing：已有 position_ids
        # 更新 ring attention 参数（告诉它每个样本的边界）
        update_ring_attn_params(position_ids=batch["position_ids"])
    else:
        # 创建标准 position_ids
        batch["position_ids"] = torch.arange(
            0, original_seq_len,
            dtype=torch.long,
            device=batch["input_ids"].device,
        ).expand(batch_size, -1)
        # [0, 1, 2, ..., 16383]

    # === 步骤 2：添加 padding（确保能被 CP size 整除）===
    total_seq_len = original_seq_len
    pad_len = 0
    divisor = min(local_world_size, 64)  # 通常是 CP size

    if total_seq_len % divisor != 0:
        # 计算需要的 padding
        pad_len = divisor - (total_seq_len % divisor)
        # 例：16384 % 4 = 0，不需要 padding
        # 例：16000 % 4 = 0，需要 padding 384

        # 对所有相关 tensor 添加 padding
        for key in batch:
            if (isinstance(batch[key], torch.Tensor) and
                batch[key].dim() > 1 and
                batch[key].size(1) == total_seq_len):

                # 确定 padding 值
                pad_value = -100 if key == "labels" else 0

                # 创建 padding tensor
                padding = torch.full(
                    (batch[key].size(0), pad_len, *batch[key].shape[2:]),
                    pad_value,
                    dtype=batch[key].dtype,
                    device=batch[key].device,
                )

                # 拼接到右侧
                batch[key] = torch.cat([batch[key], padding], dim=1)

        total_seq_len = batch["input_ids"].size(1)

    # === 步骤 3：切分序列 ===
    for key in batch:
        if not isinstance(batch[key], torch.Tensor) or batch[key].dim() <= 1:
            continue  # 跳过标量

        # 沿序列维度（dim=1）切分
        if batch[key].size(1) == total_seq_len:
            chunks = batch[key].chunk(local_world_size, dim=1)
            # 例：[1, 16384, 4096] → 4 个 [1, 4096, 4096]

            # 取当前 rank 的 chunk
            batch[key] = chunks[local_rank].contiguous()
            # GPU 0: 取第 0 块 (tokens 0-4095)
            # GPU 1: 取第 1 块 (tokens 4096-8191)
            # GPU 2: 取第 2 块 (tokens 8192-12287)
            # GPU 3: 取第 3 块 (tokens 12288-16383)

    # === 步骤 4：更新 num_items_in_batch（如果存在）===
    if "num_items_in_batch" in batch:
        # 计算本地有效 token 数
        local_valid_tokens = (batch["labels"] != -100).sum()

        # All-Reduce 跨 CP 组，获取全局 token 数
        cp_group = get_ring_attn_group()
        global_valid_tokens = local_valid_tokens.clone()
        dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.AVG, group=cp_group)

        # 更新 batch 中的 token 数
        batch["num_items_in_batch"] = (
            int(global_valid_tokens.item()) * gradient_accumulation_steps
        )

    return batch, original_seq_len, pad_len
```

**切分示例**（可视化）：

```
原始序列（16384 tokens）：
┌─────────────────────────────────────────────────┐
│ T0  T1  T2  ...  T4095 T4096 ... T12287 ... T16383 │
└─────────────────────────────────────────────────┘

切分后（CP=4）：
GPU 0: │ T0  T1  T2  ...  T4095 │
GPU 1:                 │ T4096 ... T8191 │
GPU 2:                             │ T8192 ... T12287 │
GPU 3:                                         │ T12288 ... T16383 │

Position IDs（保持不变）：
GPU 0: [0, 1, 2, ..., 4095]
GPU 1: [4096, 4097, ..., 8191]
GPU 2: [8192, 8193, ..., 12287]
GPU 3: [12288, 12289, ..., 16383]
```

---

## 第四阶段：Ring-Flash-Attention 执行

### 4.1 模型前向传播入口

```python
# 用户代码（Trainer 内部）：
outputs = model(
    input_ids=input_ids_local,  # 已切分
    attention_mask=attention_mask_local,
    labels=labels_local,
    ...
)

# 这会调用模型的 forward 方法
# 例如：LlamaForCausalLM.forward()
```

### 4.2 Attention 层执行

```python
# 文件：transformers/models/llama/modeling_llama.py (简化)

class LlamaAttention(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        ...
    ):
        # 1. 计算 Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. Reshape for multi-head attention
        query_states = query_states.view(bsz, seq_len, num_heads, head_dim)
        key_states = key_states.view(bsz, seq_len, num_heads, head_dim)
        value_states = value_states.view(bsz, seq_len, num_heads, head_dim)

        # === 3. 调用 Flash Attention（已被替换）===
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states,
            attention_mask, seq_len, is_causal=True, ...
        )
        # ↑ 实际调用的是 Ring-Flash-Attention！

        # 4. 输出投影
        attn_output = self.o_proj(attn_output)

        return attn_output
```

### 4.3 Ring-Flash-Attention 核心实现

```python
# 文件：src/axolotl/monkeypatch/ring_attn/patch.py (56-129 行)

def _flash_attention_forward_v3(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    is_causal,
    dropout=0.0,
    position_ids=None,
    ...
):
    """Ring-Flash-Attention 前向传播（Axolotl 的包装）"""

    # 验证输入
    assert is_causal, "只支持 causal attention"
    batch_size = query_states.size(0)
    assert batch_size == 1, "varlen 要求 batch_size = 1"

    # === 调用 ring-flash-attn 库 ===
    from ring_flash_attn import llama3_flash_attn_varlen_func
    from ring_flash_attn.adapters.hf_adapter import DATA_PARAMS

    attn_output = llama3_flash_attn_varlen_func(
        query_states.squeeze(dim=0),  # [seq_len, num_heads, head_dim]
        key_states.squeeze(dim=0),
        value_states.squeeze(dim=0),
        cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],  # 累积序列长度
        cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
        max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
        max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
        heads_k_stride=self.heads_k_stride,  # K 的步长
        local_k_slice=DATA_PARAMS["local_k_slice"],
        dropout_p=dropout,
        causal=is_causal,
        group=self.process_group,  # ← Ring 通信组
        **flash_kwargs,
    )

    return attn_output.unsqueeze(dim=0)
```

### 4.4 Ring-Flash-Attention 底层（ring-flash-attn 库）

```python
# 伪代码：ring-flash-attn 库的核心实现

def llama3_flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    heads_k_stride,
    local_k_slice,
    dropout_p=0.0,
    causal=True,
    group=None,
    ...
):
    """
    Ring-Flash-Attention 核心实现

    参数：
        q: [total_seq_len, num_heads, head_dim]
        k, v: 同上
        group: CP 进程组
    """

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    # === 初始化累积器 ===
    # 使用 online softmax 技巧
    attn_output = torch.zeros_like(q)
    lse = torch.zeros(q.size(0), q.size(1), dtype=torch.float32, device=q.device)
    # lse: log-sum-exp，用于 softmax 归一化

    # 当前持有的 K, V（初始为本地）
    current_k = k
    current_v = v

    # === Ring 循环（world_size 轮）===
    for step in range(world_size):
        # 1. 计算当前块的 attention
        # 使用 Flash Attention 分块计算（避免显存爆炸）
        block_output, block_lse = flash_attention_varlen(
            q, current_k, current_v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            causal=causal,
            ...
        )

        # 2. Online Softmax 更新
        # 合并当前块的结果到累积器
        attn_output, lse = _merge_attn_outputs(
            attn_output, lse,
            block_output, block_lse
        )

        # 3. 传递 K, V 给下一个 GPU
        if step < world_size - 1:
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1 + world_size) % world_size

            # 异步发送和接收（流水线）
            send_req = dist.isend(current_k, dst=next_rank, group=group)
            recv_req = dist.irecv(new_k_buffer, src=prev_rank, group=group)

            # 同时发送 V
            send_v_req = dist.isend(current_v, dst=next_rank, group=group)
            recv_v_req = dist.irecv(new_v_buffer, src=prev_rank, group=group)

            # 等待接收完成
            recv_req.wait()
            recv_v_req.wait()

            # 更新当前 K, V
            current_k = new_k_buffer
            current_v = new_v_buffer

            # 等待发送完成
            send_req.wait()
            send_v_req.wait()

    # === 最终归一化 ===
    # attn_output 已经是正确的 attention 结果
    return attn_output


def _merge_attn_outputs(
    old_output, old_lse,
    new_output, new_lse
):
    """Online Softmax：合并两个 attention 输出"""

    # 计算新的 log-sum-exp
    max_lse = torch.maximum(old_lse, new_lse)
    old_scale = torch.exp(old_lse - max_lse)
    new_scale = torch.exp(new_lse - max_lse)

    # 更新输出（加权平均）
    merged_output = (
        old_output * old_scale.unsqueeze(-1) +
        new_output * new_scale.unsqueeze(-1)
    ) / (old_scale + new_scale).unsqueeze(-1)

    # 更新 lse
    merged_lse = max_lse + torch.log(old_scale + new_scale)

    return merged_output, merged_lse
```

**Ring 通信时间线**（GPU 0 的视角，CP=4）：

```
时间 →

Step 0: │ 计算 Q@K0 │ → │ 发送 K0 给 GPU 1 │
              ↓
Step 1:    │ 接收 K3 from GPU 3 │ → │ 计算 Q@K3 │ → │ 发送 K3 给 GPU 1 │
                                        ↓
Step 2:                            │ 接收 K2 │ → │ 计算 Q@K2 │ → │ 发送 K2 │
                                                      ↓
Step 3:                                          │ 接收 K1 │ → │ 计算 Q@K1 │
                                                                    ↓
                                                                完成！

特点：
- 接收、计算、发送流水线重叠
- 显存只需要存储 2 份 K, V（本地 + 传递中）
- 通信与计算时间部分掩盖
```

---

## 第五阶段：输出聚合与反向传播

### 5.1 输出聚合（Post-Forward Hook）

```python
# 文件：src/axolotl/utils/ctx_managers/sequence_parallel.py (302-308 行)

def _gather_outputs(self, output: CausalLMOutputWithPast):
    """收集所有 GPU 的输出，重建完整 tensor"""

    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dim() > 1:
            # 使用自定义 All-Gather（支持梯度）
            output[key] = AllGatherWithGrad.apply(value, self.process_group)

    return output
```

**AllGatherWithGrad 前向传播**：

```python
# 文件：src/axolotl/utils/ctx_managers/sequence_parallel.py (314-359 行)

class AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, group):
        """
        前向传播：从所有 rank 收集输出

        输入：
            input_tensor: [batch, local_seq_len, hidden_dim]
                          例如 [1, 4096, 4096]

        输出：
            result: [batch, total_seq_len, hidden_dim]
                    例如 [1, 16384, 4096]
        """

        ctx.group = group
        ctx.rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        # === 1. 收集形状信息（支持变长）===
        local_shape = torch.tensor(list(input_tensor.shape), device=input_tensor.device)
        all_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
        dist.all_gather(all_shapes, local_shape, group=group)

        # 提取序列长度
        seq_lens = [int(shape[1].item()) for shape in all_shapes]
        ctx.seq_lens = seq_lens  # 存储，反向传播需要

        # === 2. All-Gather 数据 ===
        gathered = [
            torch.zeros(
                tuple(shape.tolist()),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            for shape in all_shapes
        ]
        dist.all_gather(gathered, input_tensor, group=group)

        # === 3. 拼接 ===
        result = torch.cat(gathered, dim=1)  # 沿序列维度拼接
        # [1, 4096, 4096] + [1, 4096, 4096] + ... = [1, 16384, 4096]

        return result
```

**All-Gather 可视化**：

```
GPU 0 输出: [1, 4096, 4096] (logits for tokens 0-4095)
GPU 1 输出: [1, 4096, 4096] (logits for tokens 4096-8191)
GPU 2 输出: [1, 4096, 4096] (logits for tokens 8192-12287)
GPU 3 输出: [1, 4096, 4096] (logits for tokens 12288-16383)

All-Gather 后（每个 GPU 上）:
[1, 16384, 4096]  ← 完整序列的 logits
```

### 5.2 Loss 计算

```python
# Trainer 内部（所有 GPU 上相同）：
outputs = model(...)  # 已聚合，完整序列

# 计算 loss（标准 cross-entropy）
logits = outputs.logits  # [1, 16384, 4096]
labels = labels_full     # [1, 16384] (已聚合)

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=-100
)

# Loss 在所有 CP GPUs 上相同
```

### 5.3 反向传播（自动梯度切分）

```python
# Trainer 内部：
loss.backward()
```

**AllGatherWithGrad 反向传播**：

```python
# 文件：src/axolotl/utils/ctx_managers/sequence_parallel.py (361-387 行)

class AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：提取本 rank 的梯度切片

        输入：
            grad_output: [batch, total_seq_len, hidden_dim]
                         例如 [1, 16384, 4096]

        输出：
            grad_input: [batch, local_seq_len, hidden_dim]
                        例如 [1, 4096, 4096]
        """

        rank = ctx.rank
        seq_lens = ctx.seq_lens  # [4096, 4096, 4096, 4096]

        # === 计算本 rank 的切片位置 ===
        offset = sum(seq_lens[:rank])
        # GPU 0: offset = 0
        # GPU 1: offset = 4096
        # GPU 2: offset = 8192
        # GPU 3: offset = 12288

        # === 提取梯度切片 ===
        grad_slice = grad_output[:, offset : offset + seq_lens[rank]].contiguous()
        # GPU 0: grad_output[:, 0:4096, :]
        # GPU 1: grad_output[:, 4096:8192, :]
        # GPU 2: grad_output[:, 8192:12288, :]
        # GPU 3: grad_output[:, 12288:16384, :]

        return grad_slice, None  # 返回梯度，None 是 group 参数
```

**反向传播流程**：

```
损失 loss (标量，所有 GPU 相同)
    ↓ backward()
梯度传播到 logits
    ↓
grad_logits: [1, 16384, 4096] (完整序列)
    ↓ AllGatherWithGrad.backward()
切分梯度:
    GPU 0: grad_logits[:, 0:4096, :]
    GPU 1: grad_logits[:, 4096:8191, :]
    GPU 2: grad_logits[:, 8192:12287, :]
    GPU 3: grad_logits[:, 12288:16383, :]
    ↓
继续反向传播到模型参数
    ↓
Ring-Flash-Attention 的反向传播
    ↓ (自动处理 Ring 通信)
梯度传播到 Q, K, V
    ↓
梯度传播到参数（每个 GPU 只更新本地参数）
```

### 5.4 Ring-Flash-Attention 反向传播

```python
# ring-flash-attn 库内部（简化伪代码）

def llama3_flash_attn_varlen_func_backward(
    grad_output,  # 对 attention 输出的梯度
    q, k, v,      # 前向传播保存的输入
    attn_output,  # 前向传播的输出
    lse,          # log-sum-exp
    group,
):
    """Ring-Flash-Attention 反向传播"""

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    # 初始化梯度累积器
    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    # 当前持有的 K, V（与前向传播相同的顺序）
    current_k = k
    current_v = v

    # === Ring 循环（与前向传播相同）===
    for step in range(world_size):
        # 1. 计算当前块的梯度
        grad_q_block, grad_k_block, grad_v_block = flash_attention_backward(
            grad_output, q, current_k, current_v, ...
        )

        # 2. 累积梯度
        grad_q += grad_q_block

        # grad_k 和 grad_v 需要传回原始 GPU
        # 使用 Ring 逆向传递

        # 3. 传递 K, V（与前向传播相同）
        if step < world_size - 1:
            # 发送/接收 K, V
            ...

    # === Ring 逆向传递：将 grad_k, grad_v 送回原始 GPU ===
    # 类似前向传播，但方向相反

    return grad_q, grad_k, grad_v
```

---

## 调试与监控

### 1. 打印 CP 组信息

```python
# 在训练开始前添加：
from axolotl.monkeypatch.ring_attn import get_ring_attn_group
import torch.distributed as dist

cp_group = get_ring_attn_group()
cp_size = dist.get_world_size(cp_group)
cp_rank = dist.get_rank(cp_group)
cp_ranks = dist.get_process_group_ranks(cp_group)

print(f"✅ CP 配置:")
print(f"   CP Size: {cp_size}")
print(f"   CP Rank: {cp_rank}")
print(f"   CP Group Members: {cp_ranks}")
```

**预期输出**（8 GPUs, TP=2, CP=2, FSDP=2）：

```
# GPU 0
✅ CP 配置:
   CP Size: 2
   CP Rank: 0
   CP Group Members: [0, 4]  # GPU 0 和 GPU 4 在同一 CP 组

# GPU 1
✅ CP 配置:
   CP Size: 2
   CP Rank: 0
   CP Group Members: [1, 5]
```

### 2. 监控序列切分

```python
# 在 training_step 中添加：
def training_step(self, model, inputs):
    local_seq_len = inputs["input_ids"].size(1)
    global_seq_len = local_seq_len * self.args.context_parallel_size

    print(f"GPU {dist.get_rank()}:")
    print(f"  Local Seq Len: {local_seq_len}")
    print(f"  Global Seq Len: {global_seq_len}")

    # 检查 position_ids
    if "position_ids" in inputs:
        pos_min = inputs["position_ids"].min().item()
        pos_max = inputs["position_ids"].max().item()
        print(f"  Position IDs: [{pos_min}, {pos_max}]")
```

**预期输出**（CP=4, 序列长度 16384）：

```
GPU 0:
  Local Seq Len: 4096
  Global Seq Len: 16384
  Position IDs: [0, 4095]

GPU 1:
  Local Seq Len: 4096
  Global Seq Len: 16384
  Position IDs: [4096, 8191]

GPU 2:
  Local Seq Len: 4096
  Global Seq Len: 16384
  Position IDs: [8192, 12287]

GPU 3:
  Local Seq Len: 4096
  Global Seq Len: 16384
  Position IDs: [12288, 16383]
```

### 3. 验证输出聚合

```python
# 在 compute_loss 后添加：
def compute_loss(self, model, inputs):
    outputs = model(**inputs)

    # 检查输出是否正确聚合
    logits_seq_len = outputs.logits.size(1)
    expected_seq_len = inputs["input_ids"].size(1) * self.args.context_parallel_size

    if logits_seq_len != expected_seq_len:
        raise AssertionError(
            f"输出未正确聚合！\n"
            f"  期望序列长度: {expected_seq_len}\n"
            f"  实际序列长度: {logits_seq_len}"
        )

    print(f"✅ 输出已正确聚合：{logits_seq_len} tokens")
```

### 4. 监控通信时间

```python
# 使用 PyTorch Profiler
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for step in range(10):
        outputs = model(input_ids)
        loss = outputs.loss
        loss.backward()

# 查看通信操作
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# 寻找 ring attention 相关操作：
# - nccl:send
# - nccl:recv
# - flash_attn_varlen
```

---

## 性能优化技巧

### 1. heads_k_stride 调优

```python
# heads_k_stride 控制每次传递多少 K heads

# 默认值（heads_k_stride=1）：
# - 每次传递所有 K heads
# - 通信量大，但精度高

# 增大 stride（heads_k_stride=2）：
# - 每次传递一半 K heads
# - 通信量减半，可能加速
# - 但可能影响精度

# 建议：
heads_k_stride: 1  # 先用默认值
# 如果通信是瓶颈，再尝试 2
```

### 2. 通信与计算重叠

Ring-Flash-Attention 自动重叠通信与计算，但可以进一步优化：

```python
# 确保使用异步通信
# ring-flash-attn 库已经使用 dist.isend / dist.irecv

# 优化建议：
# 1. 使用 NVLink 互连（必须）
# 2. 调整 NCCL 参数
os.environ['NCCL_MIN_NCHANNELS'] = '4'
os.environ['NCCL_MAX_NCHANNELS'] = '16'
```

### 3. Batch Size 补偿

```yaml
# CP 会减少有效 batch size
# 原因：每个 CP 组处理 1 个 batch

# 例：8 GPUs, CP=4
# CP 组数 = 8 / 4 = 2
# 每组处理 1 个 batch
# 有效 global batch = micro_batch_size * 2

# 补偿方法：增大 gradient_accumulation_steps
micro_batch_size: 1
gradient_accumulation_steps: 32  # 增大以保持总 batch size

# 有效总 batch = 1 * 32 * 2 = 64
```

---

## 总结

### CP 源码的核心流程

1. **配置解析** → 验证 CP 参数合法性
2. **DeviceMesh 构建** → 创建包含 "cp" 维度的网格
3. **Ring Attention 初始化** → 注册 CP 进程组
4. **Flash Attention 替换** → 全局替换为 Ring-Flash-Attention
5. **Context Manager** → 注册 pre/post-forward hooks
6. **序列切分** → Pre-hook 自动切分输入
7. **Ring-Flash-Attention** → 循环传递 K/V，online softmax
8. **输出聚合** → Post-hook All-Gather 输出
9. **反向传播** → 自动切分梯度，Ring 反向传播

### 关键技术点

1. **Online Softmax**：允许分块计算 Attention 并逐步合并
2. **Ring 通信**：循环传递 K/V，避免同时存储所有
3. **通信计算重叠**：异步通信，流水线执行
4. **自定义 Autograd**：AllGatherWithGrad 支持梯度反向传播
5. **Hook 机制**：无侵入式地切分/聚合序列

### 与 TP 的对比

| 维度 | Tensor Parallelism | Context Parallelism |
|------|-------------------|---------------------|
| **切分对象** | 权重矩阵 | 输入序列 |
| **关键算法** | 列切分 + 行切分 | Ring-Flash-Attention |
| **通信方式** | All-Reduce (同步) | Ring Send/Recv (异步) |
| **通信次数** | 每层 2 次 | 每层 N-1 轮 |
| **实现方式** | DTensor (PyTorch) | Hook + ring-flash-attn |
| **显存节省** | 模型参数 | 激活值（主要） |

---

*本文档详细解析了 Axolotl 的 CP 源码执行流程，帮助开发者深入理解 Ring-Flash-Attention 的实现细节。*
