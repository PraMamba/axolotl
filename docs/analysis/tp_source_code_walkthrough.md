# Tensor Parallelism 源码执行流程详解

> 本文档详细追踪 Axolotl 从配置读取到模型训练的完整 TP 实现流程

## 执行流程概览

```
用户执行: axolotl train config.yaml
    ↓
1. CLI 入口 (cli/main.py)
    ↓
2. 配置解析 (cli/config.py)
    ↓
3. 并行配置构建 (utils/distributed.py)
    ↓
4. 模型加载器初始化 (loaders/model.py)
    ↓
5. ParallelismConfig 设置 (accelerate)
    ↓
6. DeviceMesh 构建 (torch.distributed)
    ↓
7. 模型实例化 (transformers)
    ↓
8. DTensor 转换 (torch.distributed.tensor)
    ↓
9. Trainer 构建 (core/builders/causal.py)
    ↓
10. 训练循环 (transformers.Trainer)
```

---

## 第一阶段：配置解析

### 1.1 CLI 入口点

```python
# 文件：src/axolotl/cli/main.py (77-100 行)

@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option("--launcher", default="accelerate")
def train(ctx, config: str, launcher: str, **kwargs):
    """训练命令入口"""

    # 读取 YAML 配置文件
    with open(config, encoding="utf-8") as file:
        cfg_dict = yaml.safe_load(file)

    # 合并命令行参数
    cfg_dict.update(kwargs)

    # 转换为 DictDefault 对象
    from axolotl.utils.dict import DictDefault
    cfg = DictDefault(cfg_dict)

    # 启动训练
    if launcher == "accelerate":
        # 使用 accelerate launch 启动分布式训练
        launch_training(cfg, ...)
    elif launcher == "torchrun":
        # 使用 torchrun 启动
        ...
```

### 1.2 配置验证

```python
# 文件：src/axolotl/utils/schemas/validation.py

class AxolotlConfigValidator:
    """配置验证器"""

    def validate_parallelism(self, cfg):
        """验证并行配置"""
        world_size = get_world_size()

        # 计算总并行度
        total_parallel = (
            cfg.get("tensor_parallel_size", 1) *
            cfg.get("context_parallel_size", 1) *
            cfg.get("dp_shard_size", 1) *
            cfg.get("dp_replicate_size", 1)
        )

        if total_parallel != world_size:
            raise ValueError(
                f"并行配置 ({total_parallel}) 与 GPU 数量 ({world_size}) 不匹配"
            )

        # TP 需要 FSDP2
        if cfg.get("tensor_parallel_size", 1) > 1:
            if cfg.get("fsdp_version") != 2:
                raise ValueError("TP 需要 FSDP version 2")

        return True
```

---

## 第二阶段：分布式初始化

### 2.1 Accelerate 初始化

```python
# 文件：accelerate 库内部（Axolotl 调用）

from accelerate import Accelerator, PartialState

# 在训练脚本启动时自动执行
state = PartialState()

# 这会初始化：
# - torch.distributed.init_process_group()
# - 设置环境变量：RANK, WORLD_SIZE, LOCAL_RANK
# - 创建默认 process group
```

**环境变量示例**（8 GPU 训练）：
```bash
# GPU 0:
RANK=0
LOCAL_RANK=0
WORLD_SIZE=8

# GPU 1:
RANK=1
LOCAL_RANK=1
WORLD_SIZE=8

# ... 以此类推
```

### 2.2 构建 ParallelismConfig

```python
# 文件：src/axolotl/utils/distributed.py (299-316 行)

def build_parallelism_config(cfg):
    """
    根据配置构建 ParallelismConfig 对象
    这个对象会告诉 Accelerate 如何组织 GPU
    """

    # 提取并行参数
    pc_kwargs = _get_parallel_config_kwargs(
        world_size=get_world_size(),           # 8
        tensor_parallel_size=cfg.tensor_parallel_size,  # 2
        context_parallel_size=cfg.context_parallel_size,  # 1
        dp_shard_size=cfg.dp_shard_size,      # 4
        dp_replicate_size=cfg.dp_replicate_size,  # 1
        is_fsdp=bool(cfg.fsdp or cfg.fsdp_config),  # True
    )

    if pc_kwargs:
        # 创建 ParallelismConfig
        # pc_kwargs = {"tp_size": 2, "dp_shard_size": 4}
        parallelism_config = ParallelismConfig(**pc_kwargs)

        # 构建 DeviceMesh
        device_mesh = parallelism_config.build_device_mesh("cuda")

        return parallelism_config, device_mesh

    return None, None
```

### 2.3 DeviceMesh 结构详解

```python
# 假设配置：8 GPUs, TP=2, FSDP=4

device_mesh = DeviceMesh(
    "cuda",
    mesh=[
        [0, 1],  # FSDP shard 0, TP group
        [2, 3],  # FSDP shard 1, TP group
        [4, 5],  # FSDP shard 2, TP group
        [6, 7],  # FSDP shard 3, TP group
    ],
    mesh_dim_names=["dp_shard", "tp"]
)

# 访问不同维度的子网格：
device_mesh["tp"]        # TP 维度的网格
device_mesh["dp_shard"]  # FSDP 维度的网格

# 示例：GPU 2 的视角
# - 它在 FSDP shard 1 中
# - 它的 TP 伙伴是 GPU 3
# - 它的 FSDP 伙伴是 GPU 0, 4, 6 (同列)
```

**可视化**：
```
        TP 维度 →
FSDP   ┌─────┬─────┐
维     │ 0   │ 1   │  Shard 0
度     ├─────┼─────┤
↓      │ 2   │ 3   │  Shard 1
       ├─────┼─────┤
       │ 4   │ 5   │  Shard 2
       ├─────┼─────┤
       │ 6   │ 7   │  Shard 3
       └─────┴─────┘

通信组：
- TP group 0: [0, 1] - 高频通信（NVLink）
- TP group 1: [2, 3]
- TP group 2: [4, 5]
- TP group 3: [6, 7]

- FSDP group 0: [0, 2, 4, 6] - 中频通信
- FSDP group 1: [1, 3, 5, 7]
```

---

## 第三阶段：模型加载与 TP 应用

### 3.1 模型加载器流程

```python
# 文件：src/axolotl/loaders/model.py (161-190 行)

class ModelLoader:
    def load(self):
        """完整的模型加载流程"""

        # === 第 1 步：预处理 ===
        self.patch_manager.apply_pre_model_load_patches()
        self._apply_pre_model_load_setup()
        # ↑ 在这里设置 self.parallelism_config

        # === 第 2 步：加载模型权重 ===
        PLUGIN_MANAGER.pre_model_load(self.cfg)
        skip_move_to_device = self._build_model()
        # ↑ 核心！模型在这里被加载和转换

        PLUGIN_MANAGER.post_model_build(self.cfg, self.model)

        # === 第 3 步：后处理 ===
        self._apply_post_model_load_setup()

        # === 第 4 步：加载 LoRA 等适配器 ===
        lora_config = self._load_adapters()

        return self.model, lora_config
```

### 3.2 设置并行配置

```python
# 文件：src/axolotl/loaders/model.py (192-216 行)

def _apply_pre_model_load_setup(self):
    """模型加载前的配置"""

    # 检查是否需要并行配置
    self.use_parallel_config = (
        self.cfg.fsdp_config or
        (self.cfg.tensor_parallel_size and self.cfg.tensor_parallel_size > 1) or
        (self.cfg.context_parallel_size and self.cfg.context_parallel_size > 1)
    )

    # 如果使用 FSDP1（旧版），不支持 TP
    if self.cfg.fsdp_config and self.cfg.fsdp_version != 2:
        self.use_parallel_config = False

    # 构建 ParallelismConfig
    if self.use_parallel_config:
        self._set_parallel_config()  # ← 关键调用

    # 设置其他配置...
    self._set_auto_model_loader()
    self._set_device_map_config()
    self._set_quantization_config()
    self._set_attention_config()
```

```python
# 文件：src/axolotl/loaders/model.py (421-426 行)

def _set_parallel_config(self):
    """设置并行配置"""
    parallelism_config, device_mesh = build_parallelism_config(self.cfg)

    if parallelism_config:
        # 保存到实例变量
        self.parallelism_config = parallelism_config
        self.device_mesh = device_mesh

        # 这些会被传递给 Accelerator
        # Accelerator 会在模型包装时使用它们
```

### 3.3 模型实例化

```python
# 文件：src/axolotl/loaders/model.py (_build_model 方法简化版)

def _build_model(self):
    """构建模型实例"""

    # 获取模型配置
    model_config = self.model_config

    # 根据是否使用 FSDP 选择加载方式
    if self.is_fsdp_enabled:
        # FSDP 模式：在 meta 设备上初始化（不占显存）
        with init_empty_weights():
            model = self.auto_model_loader.from_config(
                model_config,
                torch_dtype=self.cfg.torch_dtype,
                trust_remote_code=self.cfg.trust_remote_code,
            )
        # 稍后在 FSDP 包装时加载权重

    else:
        # 普通模式：直接加载到 GPU
        model = self.auto_model_loader.from_pretrained(
            self.base_model,
            config=model_config,
            torch_dtype=self.cfg.torch_dtype,
            device_map=self.model_kwargs.get("device_map"),
            **self.model_kwargs,
        )

    self.model = model
    return skip_move_to_device
```

### 3.4 FSDP2 + TP 包装

这是最关键的步骤！模型在 Trainer 初始化时被 FSDP2 包装，同时应用 TP。

```python
# 文件：transformers.Trainer 内部（简化版）

class Trainer:
    def __init__(self, model, args, ...):
        # 创建 Accelerator
        self.accelerator = Accelerator(
            fsdp_plugin=args.fsdp_config,  # FSDP 配置
        )

        # 如果提供了 device_mesh（来自 ModelLoader）
        if hasattr(model, 'device_mesh') and model.device_mesh:
            self.accelerator.state.device_mesh = model.device_mesh

        # 准备模型（这里应用 FSDP 和 TP）
        self.model = self.accelerator.prepare_model(model)
```

**Accelerator.prepare_model 内部流程**：

```python
# accelerate 库内部（简化版）

def prepare_model(self, model):
    """准备模型以用于分布式训练"""

    # 如果配置了 FSDP2
    if self.state.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import FullyShardedDataParallel

        # 获取 device_mesh
        device_mesh = self.state.device_mesh

        # 包装每个 Transformer 层
        for layer in model.layers:
            # 1. 先应用 TP（转换为 DTensor）
            if device_mesh and "tp" in device_mesh.mesh_dim_names:
                layer = apply_tensor_parallel(layer, device_mesh["tp"])

            # 2. 再应用 FSDP
            layer = FullyShardedDataParallel(
                layer,
                device_mesh=device_mesh["dp_shard"] if device_mesh else None,
                **fsdp_kwargs
            )

        return model
```

### 3.5 DTensor 转换细节

```python
# PyTorch 内部：apply_tensor_parallel (简化版)

def apply_tensor_parallel(module, tp_mesh):
    """
    将模块的权重转换为 DTensor，实现 TP
    """

    # 遍历模块的所有子模块
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # 判断是列切分还是行切分
            if name in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]:
                # 列切分
                parallelize_linear(child, tp_mesh, style="colwise")
            elif name in ["o_proj", "down_proj"]:
                # 行切分
                parallelize_linear(child, tp_mesh, style="rowwise")

    return module


def parallelize_linear(linear, tp_mesh, style):
    """将 Linear 层的权重转换为 DTensor"""

    from torch.distributed.tensor import distribute_tensor
    from torch.distributed.tensor.placement_types import Shard, Replicate

    # 获取当前权重
    weight = linear.weight  # [out_features, in_features]

    if style == "colwise":
        # 列切分：在 out_features 维度切分
        placement = [Shard(0)]  # 维度 0 = out_features
        # GPU 0: weight[:out_features//2, :]
        # GPU 1: weight[out_features//2:, :]

    elif style == "rowwise":
        # 行切分：在 in_features 维度切分
        placement = [Shard(1)]  # 维度 1 = in_features
        # GPU 0: weight[:, :in_features//2]
        # GPU 1: weight[:, in_features//2:]

    # 转换为 DTensor
    linear.weight = nn.Parameter(
        distribute_tensor(weight, tp_mesh, placement)
    )

    # 偏置（如果有）通常复制到所有 GPU
    if linear.bias is not None:
        linear.bias = nn.Parameter(
            distribute_tensor(linear.bias, tp_mesh, [Replicate()])
        )
```

**转换后的权重示例**：

```python
# 原始权重（单 GPU）：
linear.weight.shape = [4096, 4096]  # 16MB (fp16)

# TP=2 列切分后：
# GPU 0:
linear.weight.local_tensor.shape = [2048, 4096]  # 8MB
linear.weight.placements = [Shard(0)]

# GPU 1:
linear.weight.local_tensor.shape = [2048, 4096]  # 8MB
linear.weight.placements = [Shard(0)]

# 两个 GPU 合起来才是完整的权重
```

---

## 第四阶段：训练执行

### 4.1 前向传播

```python
# 用户代码（无变化）：
outputs = model(input_ids, attention_mask)

# DTensor 自动处理的幕后操作：

# 1. 输入广播
# input_ids 复制到所有 TP GPUs
input_ids_replicated = DTensor(
    local_tensor=input_ids,
    placements=[Replicate()]  # 复制到所有 GPU
)

# 2. Embedding 层（通常不切分）
# hidden_states = embedding(input_ids)
# 输出：[batch, seq_len, hidden_dim]，在所有 TP GPUs 上相同

# 3. 第一个 Transformer 层
# 3.1 QKV 投影 (列切分)
# GPU 0: Q1 = hidden @ Wq1  (计算前一半 heads)
# GPU 1: Q2 = hidden @ Wq2  (计算后一半 heads)
# 无需通信！输出自动是 DTensor[Shard(2)]

# 3.2 Attention 计算（各 GPU 独立）
# GPU 0: attn_out1 = softmax(Q1 @ K1.T) @ V1
# GPU 1: attn_out2 = softmax(Q2 @ K2.T) @ V2

# 3.3 O 投影 (行切分)
# GPU 0: out1 = attn_out1 @ Wo1
# GPU 1: out2 = attn_out2 @ Wo2
# DTensor 自动插入 All-Reduce：
# out = all_reduce_sum([out1, out2])  ← 通信！

# 4. FFN 层（类似流程）
# Gate/Up 列切分 → 激活函数 → Down 行切分 → All-Reduce

# 5. 所有层重复上述过程...

# 6. 最终输出
# logits 在所有 TP GPUs 上相同（因为最后做了 All-Reduce）
```

**通信可视化**：

```
时间线 (单个 Transformer 层)：

GPU 0              GPU 1
  │                  │
  ├─ QKV 投影 ───────┤  (无通信，列切分)
  │                  │
  ├─ Attention ──────┤  (无通信，各自计算)
  │                  │
  ├─ O 投影 ─────────┤
  │                  │
  └─→ All-Reduce ←───┘  (通信！同步结果)
  │                  │
  ├─ Gate/Up 投影 ───┤  (无通信)
  │                  │
  ├─ SiLU 激活 ──────┤  (无通信)
  │                  │
  ├─ Down 投影 ──────┤
  │                  │
  └─→ All-Reduce ←───┘  (通信！同步结果)
  │                  │
```

### 4.2 反向传播

```python
# 用户代码：
loss = outputs.loss
loss.backward()

# DTensor 自动处理的梯度计算：

# 假设最后一层是行切分的 down_proj
# GPU 0: out1 = x1 @ W1
# GPU 1: out2 = x2 @ W2
# out = out1 + out2  (All-Reduce)

# 反向传播：
# d_out 是损失对 out 的梯度（所有 GPU 相同）

# 1. 梯度反向传播到 out1, out2
# d_out1 = d_out  (GPU 0)
# d_out2 = d_out  (GPU 1)

# 2. 计算权重梯度
# GPU 0: d_W1 = x1.T @ d_out1
# GPU 1: d_W2 = x2.T @ d_out2
# 无需通信！每个 GPU 只更新自己的权重

# 3. 计算输入梯度
# GPU 0: d_x1 = d_out1 @ W1.T
# GPU 1: d_x2 = d_out2 @ W2.T
# All-Reduce 求和：
# d_x = d_x1 + d_x2  ← 通信！

# 4. 继续反向传播到前一层...
```

**梯度通信规则**：
```
列切分层（QKV, Gate, Up）：
- 前向：无通信
- 反向：需要 All-Reduce 输入梯度

行切分层（O, Down）：
- 前向：需要 All-Reduce 输出
- 反向：无通信（权重梯度）
```

### 4.3 优化器更新

```python
# 每个 TP GPU 只更新自己持有的权重部分

# GPU 0 的优化器：
optimizer.step()
# 更新：
# - q_proj.weight (前一半)
# - k_proj.weight (前一半)
# - o_proj.weight (左半部分行)
# ...

# GPU 1 的优化器：
optimizer.step()
# 更新：
# - q_proj.weight (后一半)
# - k_proj.weight (后一半)
# - o_proj.weight (右半部分行)
# ...

# 无需同步！每个 GPU 管理自己的参数
```

---

## 第五阶段：保存与加载

### 5.1 Checkpoint 保存

```python
# 文件：transformers.Trainer (save_model 方法)

def save_model(self, output_dir):
    """保存模型检查点"""

    # FSDP + TP 模式下：
    # 1. 收集分片参数到主进程
    # 2. 从 DTensor 转换回普通 Tensor
    # 3. 保存完整模型

    if self.args.fsdp:
        # 使用 FSDP 的状态字典收集
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import StateDictType

        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = self.model.state_dict()

        # 只有主进程保存
        if self.is_world_process_zero():
            # state_dict 现在是完整的、未切分的权重
            self.model.save_pretrained(output_dir, state_dict=state_dict)
```

**状态字典转换过程**：

```python
# TP + FSDP 分片状态：
# GPU 0: layers.0.q_proj.weight = DTensor([2048, 4096])
# GPU 1: layers.0.q_proj.weight = DTensor([2048, 4096])
# GPU 2: layers.0.q_proj.weight = DTensor([2048, 4096])
# GPU 3: layers.0.q_proj.weight = DTensor([2048, 4096])

# FSDP 收集后（在 TP group 0 的 rank 0 上）：
# GPU 0: layers.0.q_proj.weight = DTensor([4096, 4096])  ← 从 TP GPUs 0,1 收集

# 最终全局收集（在全局 rank 0 上）：
# GPU 0: layers.0.q_proj.weight = Tensor([4096, 4096])  ← 完整权重
```

### 5.2 Checkpoint 加载

```python
# 从保存的检查点恢复训练

# 1. 初始化模型（使用相同的并行配置）
model = ModelLoader(cfg, tokenizer).load()

# 2. 加载检查点
if cfg.resume_from_checkpoint:
    # FSDP2 会自动：
    # - 读取完整状态字典
    # - 分片到各个 GPU
    # - 转换为 DTensor（如果使用 TP）

    trainer = Trainer(model, ...)
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
```

---

## 调试技巧

### 1. 打印 DTensor 信息

```python
# 在模型加载后添加：
for name, param in model.named_parameters():
    if hasattr(param, 'placements'):
        print(f"{name}:")
        print(f"  - 全局形状: {param.shape}")
        print(f"  - 本地形状: {param.local_tensor.shape}")
        print(f"  - 切分方式: {param.placements}")
        print(f"  - DeviceMesh: {param.device_mesh}")
        break  # 只打印第一个参数
```

**预期输出**：
```
layers.0.self_attn.q_proj.weight:
  - 全局形状: torch.Size([4096, 4096])
  - 本地形状: torch.Size([2048, 4096])
  - 切分方式: [Shard(dim=0)]
  - DeviceMesh: DeviceMesh('cuda', [0, 1])
```

### 2. 监控通信

```python
# 设置环境变量以查看 NCCL 通信日志
import os
os.environ['NCCL_DEBUG'] = 'INFO'

# 训练时会打印：
# NCCL INFO AllReduce: size 16777216 (16MB), time 1.2ms
# NCCL INFO Broadcast: size 8388608 (8MB), time 0.8ms
```

### 3. 验证 TP 正确性

```python
# 比较 TP 和非 TP 的输出

# 1. 不使用 TP 训练几步，保存输出
# tensor_parallel_size: 1

# 2. 使用 TP 训练，比较输出
# tensor_parallel_size: 2

# 前几步的 loss 应该基本一致（浮点误差）
```

### 4. 性能分析

```python
# 使用 PyTorch Profiler
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    # 训练几个 step
    for step in range(10):
        outputs = model(input_ids)
        loss = outputs.loss
        loss.backward()

# 查看通信时间
print(prof.key_averages().table(sort_by="cuda_time_total"))
# 寻找 nccl:all_reduce 等通信操作
```

---

## 常见问题排查

### 问题 1：显存爆炸

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate 20.00 GiB
```

**排查步骤**：
1. 检查是否所有 GPU 都参与了 TP
```python
print(f"TP size: {cfg.tensor_parallel_size}")
print(f"World size: {get_world_size()}")
```

2. 检查 DTensor 是否生效
```python
# 应该看到本地 tensor 更小
print(f"Local shape: {model.layers[0].q_proj.weight.local_tensor.shape}")
```

3. 检查 FSDP 配置
```yaml
fsdp_config:
  reshard_after_forward: true  # 必须开启以节省显存
```

### 问题 2：训练速度慢

**症状**：
```
TP=2 比单卡还慢
```

**排查步骤**：
1. 检查 GPU 互连
```bash
nvidia-smi topo -m

# 应该看到 NVLink：
#   GPU0    GPU1
# GPU0   X     NV12
# GPU1  NV12    X

# 如果看到 PHB (PCIe)，TP 会很慢
```

2. 检查通信占比
```python
# 使用 profiler 查看 All-Reduce 时间
# All-Reduce 时间不应超过总时间的 20%
```

### 问题 3：Loss 不收敛或 NaN

**症状**：
```
Step 10: loss = nan
```

**排查步骤**：
1. 检查混合精度配置
```yaml
bf16: true  # TP 推荐 bf16
fp16: false # 不推荐 fp16
```

2. 检查梯度裁剪
```yaml
max_grad_norm: 1.0  # 防止梯度爆炸
```

3. 检查学习率
```yaml
learning_rate: 1e-5  # TP 可能需要更小的学习率
```

### 问题 4：Checkpoint 加载失败

**症状**：
```
RuntimeError: Error(s) in loading state_dict
```

**解决方案**：
```yaml
# 确保训练和加载时的并行配置一致
tensor_parallel_size: 2  # 必须相同
dp_shard_size: 4         # 必须相同

# 或使用 FULL_STATE_DICT 格式（兼容性更好）
fsdp_config:
  state_dict_type: FULL_STATE_DICT
```

---

## 性能优化清单

### ✅ 必做优化

1. **启用 Flash Attention**
```yaml
flash_attention: true
```

2. **使用 bf16**
```yaml
bf16: true
tf32: true
```

3. **启用梯度检查点**
```yaml
gradient_checkpointing: true
```

4. **合理配置 batch size**
```yaml
micro_batch_size: 1
gradient_accumulation_steps: 32
# 有效 batch = 1 × 32 × num_gpus = 256
```

### 🔧 可选优化

5. **启用编译（PyTorch 2.0+）**
```yaml
torch_compile: true
torch_compile_backend: "inductor"
```

6. **使用 Fused Optimizer**
```yaml
optimizer: adamw_torch_fused  # 比 adamw_torch 快
```

7. **开启 CCE (Cut Cross Entropy)**
```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
```

8. **调整 FSDP 参数**
```yaml
fsdp_config:
  forward_prefetch: true  # 提前预取参数
  backward_prefetch: backward_pre  # 反向传播预取
```

---

## 总结

Axolotl 的 TP 实现流程：

1. **配置解析** → 验证并行参数
2. **ParallelismConfig** → 定义 GPU 拓扑
3. **DeviceMesh** → 创建逻辑 GPU 网格
4. **DTensor 转换** → 自动切分权重
5. **前向/反向传播** → 自动通信
6. **保存/加载** → 自动收集/分发

关键点：
- ✅ **自动化**：用户只需配置，底层自动处理
- ✅ **透明性**：代码无需修改，DTensor 自动处理
- ✅ **灵活性**：可与 FSDP/DDP/CP 组合

核心依赖：
- PyTorch ≥ 2.7（DTensor 支持）
- FSDP2（新版 FSDP）
- Accelerate（并行配置）
- transformers（Trainer 集成）

---

*本文档详细解析了 Axolotl 的 TP 源码执行流程，帮助开发者深入理解实现细节。*
