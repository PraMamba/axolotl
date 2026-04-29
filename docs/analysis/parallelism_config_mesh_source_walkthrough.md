# Axolotl 源码走读：Accelerate ParallelismConfig 的 mesh 构建实现解析

在多卡训练里，“开几路并行”只是用户配置层面的说法；真正进入训练循环后，系统需要的是更低层的东西：哪些 rank 共享同一个 batch，哪些 rank 共同切参数，哪些 rank 组成 attention 的通信环，以及保存/加载时应该沿哪一个维度 gather。对于 Axolotl 来说，`tensor_parallel_size`、`context_parallel_size`、`dp_shard_size`、`dp_replicate_size` 最终都要落到 Accelerate 的 `ParallelismConfig.build_device_mesh()` 上，变成一个带名字的 PyTorch `DeviceMesh`。

本文不展开 FSDP、Tensor Parallel 或 Ring Attention 的理论推导，而是顺着 Axolotl 当前源码读一遍：用户在 YAML 里写下几个并行度之后，框架如何归一化配置、如何构造 `mesh_dim_names`、rank 为什么按现在的顺序排列、这个 mesh 又如何被 FSDP、TP、CP、DataLoader、保存逻辑分别消费。主线会特别关注显存、通信、调度、初始化、保存、patch 与性能瓶颈。

# 前言

## 业务 / 工程背景

Axolotl 的目标是用一份 YAML 驱动大模型训练。单一并行策略还比较容易理解：FSDP 管参数/梯度/优化器状态分片，TP 管层内矩阵切分，CP/Sequence Parallel 管长序列激活显存。但当它们组合时，问题不再是“是否开启某个功能”，而是：

- **同一组 rank 到底应该拿同一份数据还是不同数据？** TP/CP rank 必须看到同一条样本的不同切片；DP/FSDP 视角下又要推进不同 batch。
- **FSDP 应该沿哪个 group shard？** 如果 CP 也参与 FSDP shard，那么 FSDP 的 process group 不再只是 `dp_shard`，而是 `dp_shard × cp` 的 flattened mesh。
- **TP/CP/FSDP 的 rank 排列是否匹配硬件拓扑？** `build_device_mesh()` 默认按 row-major rank map 建 mesh，维度顺序决定了哪些 rank 互为邻居。

这就是 `Accelerate ParallelismConfig` 的价值：它把多个并行维度从“几个整数”变成命名 mesh，并让下游用 `device_mesh["tp"]`、`device_mesh[("dp_shard", "cp")]` 这样的语义化切片拿到通信组。

## 核心矛盾

本文的核心矛盾可以概括为三句话：

1. **用户想用简单配置表达 N-D 并行，但运行时需要明确的 rank 分组。**
2. **TP/CP 想共享 batch，DP/FSDP 想区分 batch；FSDP 又可能要把 CP 维度一起 flatten 进 shard group。**
3. **mesh 构建本身不搬 tensor，却决定了后续每一步通信、显存峰值与保存路径。**

## 本文主线

本文按机制展开：

1. 配置如何从 YAML/CLI 变成 Accelerate 环境变量与 `ParallelismConfig` 参数。
2. `build_device_mesh()` 如何命名维度、排序维度、排列 rank，并创建 flattened mesh。
3. 这个 mesh 如何被模型加载、Accelerator、FSDP2、CP ring attention、DataLoader 和保存逻辑消费。
4. 真实训练主路径、shape/rank/state 流、性能收益与坑点。
5. 测试覆盖、维护风险与后续优化方向。

## 不展开的内容

本文不讲 FSDP 原理、不讲 Megatron TP 数学、不讲 Ring Attention 论文细节，也不讲 PyTorch DTensor 的完整设计；只讲 Axolotl 如何把这些机制接入训练链路，以及源码中能确认的行为。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/main.py` | 用户 `axolotl train` 入口，决定是否通过 `accelerate launch`/`torchrun` 启动。 |
| `src/axolotl/cli/config.py` | 读取 YAML/CLI override，调用配置校验、环境变量准备与归一化。 |
| `src/axolotl/utils/schemas/config.py` / `validation.py` | 定义并校验 `context_parallel_size`、`tensor_parallel_size`、`dp_*` 等字段。 |
| `src/axolotl/utils/trainer.py` | 把 Axolotl 配置写成 Accelerate/FSDP/parallelism 环境变量，并注入 CP patch。 |
| `src/axolotl/utils/distributed.py` | Axolotl 自己构造 `ParallelismConfig` kwargs，并调用 `build_device_mesh("cuda")`。 |
| `src/axolotl/loaders/model.py` | 模型加载前构建 parallel config；TP 加载时把 `device_mesh` 传给 Transformers。 |
| `src/axolotl/loaders/patch_manager.py` | 根据 CP/FSDP2 配置注入 Accelerate、Transformers、FSDP2 相关 monkey patch。 |
| `src/axolotl/train.py` | 训练时进入 `SequenceParallelContextManager`，把 trainer 的 `torch_device_mesh` 交给 CP。 |
| `src/axolotl/utils/ctx_managers/sequence_parallel.py` | 前向 hook 中切分输入序列、必要时 gather 输出与修正 eval loss。 |
| `/usr/local/lib/python3.12/dist-packages/accelerate/parallelism_config.py` | 当前环境中 Accelerate 1.13.0 的 `ParallelismConfig` 与 `build_device_mesh()` 实现。 |

> 版本依据：Axolotl 当前 `pyproject.toml` 固定 `accelerate==1.13.0`、`transformers==5.5.4`、`torch>=2.9.1`（`pyproject.toml:15-21`）。

# 一、配置入口：把“几个整数”变成可执行拓扑

## 1.1 设计哲学与核心问题

在 Axolotl 里，用户不会直接写 `ParallelismConfig(tp_size=..., cp_size=...)`，而是在 YAML 里写：

```yaml
dp_shard_size: 2
context_parallel_size: 2
tensor_parallel_size: 2
fsdp_version: 2
```

这层机制要解决的是**配置语义归一化**问题：用户给的是 Axolotl 术语，Accelerate 需要的是 `tp_size/cp_size/dp_shard_size/dp_replicate_size`；用户可能省略 `dp_shard_size`，但 FSDP 又需要一个完整 world-size 分解；用户可能只开 CP，Accelerate upstream 默认又倾向于把 torch CP 绑定到 FSDP2。

如果没有这一层，后续 `build_device_mesh()` 不知道 world size 应该拆成几维，DataLoader 不知道 TP/CP rank 应拿同一 batch，FSDP2 也不知道该沿哪个 group shard 参数。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：Click 命令入口，默认走 accelerate launch。
  - launch_training：构造实际子进程命令。

src/axolotl/cli/config.py
  - load_cfg：读取 YAML，调用 validate_config / prepare_optim_env / normalize_config。

src/axolotl/utils/trainer.py
  - prepare_optim_env：统一设置 FSDP、DeepSpeed、parallelism 环境。
  - setup_parallelism_envs：把 Axolotl 并行配置写入 PARALLELISM_CONFIG_*。

src/axolotl/utils/distributed.py
  - _get_parallel_config_kwargs：把 world_size 和各并行度归一化为 ParallelismConfig kwargs。
  - build_parallelism_config：创建 Accelerate ParallelismConfig 并调用 build_device_mesh("cuda")。
```

## 1.3 主流程拆解

用户最常见路径是：

```text
User: axolotl train config.yaml
  -> src/axolotl/cli/main.py:train
    -> launch_training(..., launcher="accelerate")
      -> accelerate launch -m axolotl.cli.train config.yaml
        -> src/axolotl/cli/train.py:do_cli
          -> load_cfg(config)
            -> validate_config(...)
            -> prepare_optim_env(cfg)
              -> setup_parallelism_envs(cfg)
            -> normalize_config(cfg)
```

入口证据：`axolotl train` 的 Click 命令定义在 `src/axolotl/cli/main.py:78-128`，默认 launcher 是 `accelerate`（`src/axolotl/cli/main.py:83-87`）；实际命令在 `_launch_accelerate_training()` 里拼成 `accelerate launch ... -m axolotl.cli.train`（`src/axolotl/cli/utils/train.py:157-192`）。

配置加载在 `load_cfg()`：它先读 YAML（`src/axolotl/cli/config.py:244-253`），应用 CLI override（`src/axolotl/cli/config.py:265-299`），再调用 `validate_config()`（`src/axolotl/cli/config.py:308-320`）、`prepare_optim_env()`（`src/axolotl/cli/config.py:326`）和 `normalize_config()`（`src/axolotl/cli/config.py:327`）。

第一个真正改变行为的函数是 `setup_parallelism_envs()`：

```python
# src/axolotl/utils/trainer.py:621-640
if cfg.tensor_parallel_size and cfg.tensor_parallel_size > 1:
    os.environ["PARALLELISM_CONFIG_TP_SIZE"] = str(cfg.tensor_parallel_size)
if cfg.dp_shard_size and cfg.dp_shard_size > 1:
    os.environ["PARALLELISM_CONFIG_DP_SHARD_SIZE"] = str(cfg.dp_shard_size)
if cfg.dp_replicate_size and cfg.dp_replicate_size > 1:
    os.environ["PARALLELISM_CONFIG_DP_REPLICATE_SIZE"] = str(cfg.dp_replicate_size)
if cfg.context_parallel_size and cfg.context_parallel_size > 1:
    os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cfg.context_parallel_size)
    os.environ["ACCELERATE_ALLOW_CP_STANDALONE"] = "true"
    patch_prepare_cp()
if set_accelerate_parallelism_config:
    os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
```

这段代码有两个副作用：

1. **给 Accelerate 构造 `ParallelismConfig()` 的环境入口。** Accelerate `Accelerator.__init__` 会在 `ACCELERATE_USE_PARALLELISM_CONFIG=true` 时创建 `ParallelismConfig()`（`/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py:453-459`）。
2. **给 Axolotl 自己的 CP 路径开绿灯。** `ACCELERATE_ALLOW_CP_STANDALONE=true` 和 `patch_prepare_cp()` 会改变 Accelerate 对 CP 的校验与 prepare 行为（后文详解）。

另一路是在模型加载前 Axolotl 自己主动构建 mesh：

```text
ModelLoader.load
  -> _apply_pre_model_load_setup
    -> _set_parallel_config
      -> build_parallelism_config(cfg)
        -> _get_parallel_config_kwargs(...)
        -> ParallelismConfig(**pc_kwargs)
        -> parallelism_config.build_device_mesh("cuda")
```

源码在 `src/axolotl/loaders/model.py:196-213` 和 `src/axolotl/loaders/model.py:437-442`。`build_parallelism_config()` 本身很短：`_get_parallel_config_kwargs()` 生成 kwargs 后，`ParallelismConfig(**pc_kwargs)` 并调用 `build_device_mesh("cuda")`（`src/axolotl/utils/distributed.py:299-316`）。

`_get_parallel_config_kwargs()` 是 Axolotl 的关键归一化函数：

```python
# src/axolotl/utils/distributed.py:327-368
remaining_world_size = world_size
if tensor_parallel_size > 1:
    pc_kwargs["tp_size"] = tensor_parallel_size
    remaining_world_size //= tensor_parallel_size
if context_parallel_size > 1:
    pc_kwargs["cp_size"] = context_parallel_size
    remaining_world_size //= context_parallel_size
if dp_shard_size is None and dp_replicate_size in (None, 1):
    if remaining_world_size > 1:
        pc_kwargs["dp_shard_size"] = remaining_world_size
        remaining_world_size = 1
...
if remaining_world_size > 1:
    raise ValueError(...)
```

直觉上，它先把非数据并行维度 TP/CP 从 world size 里除掉，再把剩余 rank 尽量分给 FSDP shard。如果用户没有显式设置 `dp_shard_size`，且没有 HSDP replication，它会把剩余 world size 自动当作 `dp_shard_size`（`src/axolotl/utils/distributed.py:338-341`）。

`normalize_config()` 还会重写 batch size：当 `world_size != 1` 且启用 FSDP/DDP 时，`effective_world_size = world_size // context_parallel_size // tensor_parallel_size`，再把 `cfg.batch_size` 乘上这个有效 DP world size（`src/axolotl/utils/config/__init__.py:134-142`）。这就是为什么 CP/TP 不是增加数据吞吐的维度：同一 TP/CP 组内 rank 处理的是同一 batch 的不同模型/序列切片。

## 1.4 关键细节与误区澄清

> 容易误解点 1：`context_parallel_size` 在 Axolotl 文档里常被称为 sequence parallelism，但它传给 Accelerate 的字段是 `cp_size`，不是 `sp_size`。

源码依据是 `setup_parallelism_envs()` 写入 `PARALLELISM_CONFIG_CP_SIZE`（`src/axolotl/utils/trainer.py:632-635`），`_get_parallel_config_kwargs()` 写入 `pc_kwargs["cp_size"]`（`src/axolotl/utils/distributed.py:334-336`）。Accelerate 自己的 `sp_size` 是 DeepSpeed ALST/Ulysses SP 路径；`build_device_mesh()` 在 `sp_backend == "deepspeed" and sp_size > 1` 时直接返回 `None`（`/usr/local/lib/python3.12/dist-packages/accelerate/parallelism_config.py:218-221`）。Axolotl 的 ring attention 路径实际使用 `cp` 维度。

> 容易误解点 2：`tensor_parallel_size` 的 schema 描述写着 “Only supported with DeepSpeed AutoTP”，但当前源码中它也会进入 FSDP/Transformers TP 路径。

字段描述在 `src/axolotl/utils/schemas/config.py:993-997`，但 `ModelLoader._build_model()` 在 `tensor_parallel_size > 1` 时设置 `tp_size`、`tp_plan="auto"`、`device_mesh` 并传给 `from_pretrained()`（`src/axolotl/loaders/model.py:749-755`）。另外 DeepSpeed 路径也会在 validation 中改写 deepspeed json 的 `tensor_parallel.autotp_size`（`src/axolotl/utils/schemas/validation.py:1121-1148`）。所以它不是单一路径字段。

> 容易误解点 3：`ParallelismConfig` 不是只由 Axolotl 显式传入 Trainer。

Axolotl 的 TrainingArguments 构造没有显式填 `parallelism_config`；它主要通过环境变量让 Accelerate 在 `Accelerator.__init__` 中自己创建（`/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py:453-459`）。同时，Axolotl 在模型加载阶段也会单独构建一个 `device_mesh` 给 TP 加载使用（`src/axolotl/loaders/model.py:437-442`）。这意味着“模型加载用的 mesh”和“Trainer/Accelerator state 里的 mesh”是两条构建路径，配置必须一致。

## 1.5 本章小结

> 💡 **小结**
>
> * Axolotl 的并行配置先经 schema/validation，再通过环境变量与显式 `build_parallelism_config()` 进入 Accelerate。
> * `context_parallel_size` 在源码中映射到 Accelerate `cp_size`，不是 `sp_size`。
> * `_get_parallel_config_kwargs()` 的核心策略是先扣除 TP/CP，再把剩余 world size 自动归入 FSDP shard 或 HSDP replicate/shard。
> * batch size 会按有效 DP world size 重算，TP/CP 组内 rank 不应被当成独立数据并行 rank。

# 二、Device Mesh 构建：维度命名与 rank 排列策略

## 2.1 设计哲学与核心问题

配置归一化之后，真正决定通信拓扑的是 `build_device_mesh()`。这一层解决的是**命名维度与 rank 排列**问题：同样是 8 个 rank，`(dp_shard=2, cp=2, tp=2)` 可以有很多排列方式。不同排列会导致：

- TP group 是 `[0,1]` 还是 `[0,4]`；
- CP ring 是相邻 GPU 还是跨节点 GPU；
- FSDP shard group 是否跨 CP 维度；
- DataLoader 计算“谁拿同一个 batch”时是否与 mesh 一致。

Accelerate 的设计选择是：先收集启用维度，再按 canonical order 排序，然后调用 PyTorch `init_device_mesh()`。这让维度顺序稳定，但也意味着它不会自动理解节点拓扑；rank 的物理排列仍然取决于 launcher 给的 global rank 顺序。

## 2.2 源码入口与关键对象

```text
/usr/local/lib/python3.12/dist-packages/accelerate/parallelism_config.py
  - dp_dim_names / non_dp_dim_names：声明哪些维度是 DP，哪些是非 DP。
  - active_mesh_dims：所有启用维度。
  - _get_mesh：按 canonical order 生成 mesh_dim_names 与 mesh_shape。
  - build_device_mesh：调用 torch.distributed.device_mesh.init_device_mesh，并创建 flattened mesh。

/usr/local/lib/python3.12/dist-packages/torch/distributed/device_mesh.py
  - init_device_mesh：创建 row-major rank_map 的 DeviceMesh。
  - DeviceMesh._flatten：把多个命名维度 flatten 成一个逻辑维度。
```

## 2.3 主流程拆解

Accelerate `ParallelismConfig` 先定义各种“命名维度”：

```python
# accelerate/parallelism_config.py:113-155
@property
def dp_dim_names(self):
    dims = []
    if self.dp_replicate_enabled: dims += ["dp_replicate"]
    if self.dp_shard_enabled: dims += ["dp_shard"]
    return dims

@property
def non_dp_dim_names(self):
    dims = []
    if self.tp_enabled: dims += ["tp"]
    if self.cp_enabled: dims += ["cp"]
    if self.sp_enabled: dims += ["sp"]
    return dims
```

注意这里 `non_dp_dim_names` 的返回顺序是 `tp -> cp -> sp`，但最终 mesh 顺序不是这个顺序。真正排序发生在 `_get_mesh()`：

```python
# accelerate/parallelism_config.py:260-272
mesh_dims = {parallelism: self._sizes[parallelism] for parallelism in self.active_mesh_dims}
mesh_order = ["dp_replicate", "dp_shard", "cp", "sp", "tp"]
sorted_items = sorted(mesh_dims.items(), key=lambda x: mesh_order.index(x[0]))
return tuple(zip(*sorted_items))
```

也就是说，最终 canonical order 是：

```text
dp_replicate -> dp_shard -> cp -> sp -> tp
```

随后 `build_device_mesh()` 调用 PyTorch：

```python
# accelerate/parallelism_config.py:228-244
mesh_dim_names, mesh_shape = self._get_mesh()
device_mesh = init_device_mesh(
    device_type,
    mesh_shape,
    mesh_dim_names=mesh_dim_names,
)
if self.dp_dim_names:
    device_mesh[self.dp_dim_names]._flatten("dp")
if self.dp_shard_cp_dim_names:
    device_mesh[self.dp_shard_cp_dim_names]._flatten("dp_shard_cp")
if self.dp_cp_dim_names:
    device_mesh[self.dp_cp_dim_names]._flatten("dp_cp")
return device_mesh
```

PyTorch `init_device_mesh()` 的 rank map 是 CPU 上的 `torch.arange(layout.numel())`（`/usr/local/lib/python3.12/dist-packages/torch/distributed/device_mesh.py:1357-1368`），即默认 row-major 排列。`DeviceMesh` 文档也说明 mesh 数组里的值是 default process group 的 global rank（`/usr/local/lib/python3.12/dist-packages/torch/distributed/device_mesh.py:129-147`）。

以 `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml` 为例，它设置：

```yaml
# examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:6-9
dp_shard_size: 2
context_parallel_size: 2
tensor_parallel_size: 2
```

在 `world_size=8` 下，Axolotl 生成：

```text
pc_kwargs = {tp_size: 2, cp_size: 2, dp_shard_size: 2}
mesh_dim_names = ("dp_shard", "cp", "tp")
mesh_shape     = (2, 2, 2)
```

rank 布局是：

```text
维度顺序: [dp_shard, cp, tp]

rank_map = arange(8).reshape(2, 2, 2)

          tp=0  tp=1

dp=0 cp=0   0     1
dp=0 cp=1   2     3

dp=1 cp=0   4     5
dp=1 cp=1   6     7
```

由此得到：

```text
TP groups（固定 dp_shard, cp）:
  [0,1], [2,3], [4,5], [6,7]

CP groups（固定 dp_shard, tp）:
  [0,2], [1,3], [4,6], [5,7]

原始 dp_shard groups（固定 cp, tp）:
  [0,4], [1,5], [2,6], [3,7]

flatten 后 dp_shard_cp groups（固定 tp）:
  tp=0: [0,2,4,6]
  tp=1: [1,3,5,7]
```

这里最值得注意的是最后一行。`fsdp_dim_names` 在没有 `dp_replicate` 时返回 `['dp_shard_cp']`（`/usr/local/lib/python3.12/dist-packages/accelerate/parallelism_config.py:157-164`），而 `dp_shard_cp` 是 `dp_shard + cp` 的 flatten 维度（`/usr/local/lib/python3.12/dist-packages/accelerate/parallelism_config.py:136-143`、`237-240`）。所以 FSDP2 实际 shard group 不是单独的 `dp_shard`，而是 **在固定 TP rank 下跨 `dp_shard × cp`**。

再看 HSDP + TP 示例：`examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml` 设置 `dp_shard_size=4`、`dp_replicate_size=2`、`tensor_parallel_size=2`（`examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml:6-9`），得到：

```text
mesh_dim_names = ("dp_replicate", "dp_shard", "tp")
mesh_shape     = (2, 4, 2)

rep=0: [[0,1], [2,3], [4,5], [6,7]]
rep=1: [[8,9], [10,11], [12,13], [14,15]]

TP groups:        [0,1], [2,3], ...
FSDP shard groups: flatten dp_shard at fixed dp_replicate,tp
FSDP dim names:   ["dp_replicate", "dp_shard_cp"]
```

由于没有 CP，`dp_shard_cp` 退化为 `dp_shard` 的 flattened 名字；有 `dp_replicate` 时，FSDP 维度又包含 `dp_replicate`（`fsdp_dim_names` 的实现见 `accelerate/parallelism_config.py:157-164`）。

## 2.4 关键细节与误区澄清

> 容易误解点 4：`non_dp_dim_names` 返回 `tp, cp`，但最终 mesh 顺序是 `cp` 在 `tp` 前面。

`active_mesh_dims = dp_dim_names + non_dp_dim_names`（`accelerate/parallelism_config.py:206-209`），但 `_get_mesh()` 又按 `mesh_order = ["dp_replicate", "dp_shard", "cp", "sp", "tp"]` 重新排序（`accelerate/parallelism_config.py:263-272`）。因此不要用 `non_dp_dim_names` 推断 rank layout，必须看 `_get_mesh()`。

> 容易误解点 5：`context_parallel_size=2` 不一定意味着 CP group 是 `[0,1]`、`[2,3]`。

在 `(dp_shard, cp, tp)` 布局中，TP 是最后一维、rank 连续变化最快；CP group 是固定 `dp_shard,tp` 后沿 cp 维度取 rank，所以是 `[0,2]`、`[1,3]` 这种跨步 group。这个结论来自 PyTorch `init_device_mesh()` 的 row-major rank map（`torch.arange`，`device_mesh.py:1357-1361`）和 Accelerate canonical order。

> 容易误解点 6：`build_device_mesh()` 本身不节省显存。

它只创建 `DeviceMesh` 和 process group：PyTorch 会根据 layout 初始化每个 mesh dimension 的 process group（`device_mesh.py:472-494`），但参数分片、激活切片、输出 gather 都发生在后续消费者里。mesh 是“拓扑元数据 + 通信组”，不是 sharding 算子。

## 2.5 本章小结

> 💡 **小结**
>
> * Accelerate 的最终维度顺序是 `dp_replicate -> dp_shard -> cp -> sp -> tp`，不是用户配置顺序。
> * PyTorch `init_device_mesh()` 默认用 `arange(world_size)` 做 row-major rank map；rank 物理相邻性取决于 launcher 排名。
> * `dp_shard_cp` 是关键 flattened mesh：有 CP 时，FSDP2 可能沿 `dp_shard × cp` shard。
> * TP 通常是最后一维，因此 TP group rank 连续；CP group 在 TP 存在时通常是跨步 rank。

# 三、Mesh 的消费者：同一个拓扑如何驱动 TP、CP、FSDP 与 DataLoader

## 3.1 设计哲学与核心问题

`DeviceMesh` 构建完成后，真正重要的问题是：谁读取它？读哪个维度？读出来之后做什么？

Axolotl 里至少有四类消费者：

1. **模型加载 TP**：Transformers `from_pretrained(..., tp_plan="auto", device_mesh=...)` 用 `tp` 子 mesh 切模型层。
2. **FSDP2 prepare**：Accelerate/Axolotl FSDP2 用 `fsdp_dim_names` 选择 shard mesh。
3. **CP / Sequence Parallel**：Axolotl ring attention 从 `device_mesh[("cp",)]` 拿 process group。
4. **DataLoader dispatch**：Accelerate 根据 mesh 中 TP/CP 大小调整 `process_index` 和 `num_processes`，让 TP/CP rank 拿同一 batch。

这层解决的是**拓扑消费边界**问题：mesh 本身不做事，只有被这些模块读取时才影响显存、通信和调度。

## 3.2 源码入口与关键对象

```text
src/axolotl/loaders/model.py
  - _set_parallel_config：模型加载前构建 parallelism_config/device_mesh。
  - _build_model：TP 开启时把 tp_size/tp_plan/device_mesh 传给 Transformers。

/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py
  - __init__：从 env 创建 ParallelismConfig，并写入 state.device_mesh。
  - torch_device_mesh：暴露 AcceleratorState.device_mesh。
  - _prepare_cp：upstream torch CP hooks；Axolotl 会 patch 成 no-op。

src/axolotl/train.py
  - execute_training：训练前进入 SequenceParallelContextManager，并传入 trainer.accelerator.torch_device_mesh。

src/axolotl/monkeypatch/ring_attn/patch.py
  - register_ring_attn_from_device_mesh：从 cp 维度拿 group，替换 flash attention。

/usr/local/lib/python3.12/dist-packages/accelerate/data_loader.py
  - prepare_data_loader：根据 torch_device_mesh 调整数据分发。
```

## 3.3 主流程拆解

### 3.3.1 模型加载阶段：TP 先用 mesh

`ModelLoader._apply_pre_model_load_setup()` 会在 `fsdp_config`、TP 或 CP 开启时构建 parallel config（`src/axolotl/loaders/model.py:196-213`）。真正写入对象的是：

```python
# src/axolotl/loaders/model.py:437-442
def _set_parallel_config(self):
    parallelism_config, device_mesh = build_parallelism_config(self.cfg)
    if parallelism_config:
        self.parallelism_config = parallelism_config
        self.device_mesh = device_mesh
```

当 TP 开启时，模型加载路径会把这个 mesh 传给 Transformers：

```python
# src/axolotl/loaders/model.py:749-755
if self.cfg.tensor_parallel_size > 1:
    self.model_kwargs["tp_size"] = self.cfg.tensor_parallel_size
    self.model_kwargs["tp_plan"] = "auto"
    self.model_kwargs["device_mesh"] = self.device_mesh
    if "device_map" in self.model_kwargs:
        del self.model_kwargs["device_map"]
```

这一步发生在 `from_pretrained()` 之前（`src/axolotl/loaders/model.py:735-747`、`843-847`），所以它影响的是模型初始化/权重加载阶段，而不是训练 step 中临时切分。

### 3.3.2 Accelerator 阶段：env 再构建一份 state mesh

Accelerate 自己会在初始化时读取环境变量：

```python
# accelerate/accelerator.py:453-475
elif os.environ.get("ACCELERATE_USE_PARALLELISM_CONFIG", "false").lower() == "true":
    parallelism_config = ParallelismConfig()
...
self.state = AcceleratorState(..., parallelism_config=parallelism_config)
if self.parallelism_config:
    self.state.device_mesh = self.parallelism_config.get_device_mesh(self.device.type)
    self.parallelism_config._validate_accelerator(self)
```

`trainer.accelerator.torch_device_mesh` 只是返回 `state.device_mesh`（`accelerate/accelerator.py:761-763`）。Axolotl 在训练前把它交给自己的 CP context：

```python
# src/axolotl/train.py:205-219
if cfg.context_parallel_size > 1:
    stack.enter_context(
        SequenceParallelContextManager(
            ...,
            device_mesh=trainer.accelerator.torch_device_mesh,
        )
    )
```

### 3.3.3 CP 阶段：从 `cp` 命名维度拿 ring group

`SequenceParallelContextManager.__init__()` 里先注册 ring attention（`src/axolotl/utils/ctx_managers/sequence_parallel.py:207-213`），具体调用：

```python
# src/axolotl/utils/ctx_managers/sequence_parallel.py:246-253
register_ring_attn_from_device_mesh(
    device_mesh=self.device_mesh,
    context_parallel_dim=("cp",),
    heads_k_stride=self.heads_k_stride,
    ring_attn_func=self.ring_attn_func,
)
```

`register_ring_attn_from_device_mesh()` 做三件事：

```python
# src/axolotl/monkeypatch/ring_attn/patch.py:159-184
sequence_mesh = device_mesh[context_parallel_dim]
sequence_pg = sequence_mesh.get_group()
context_parallel_size = sequence_mesh.size()
set_ring_attn_group(sequence_pg)
```

然后根据 `ring_attn_func` 替换 HF flash attention：`VARLEN_LLAMA3` 会 patch `ring_flash_attn.adapters.hf_adapter.create_ring_flash_attention_forward` 并调用 `substitute_hf_flash_attn()`（`src/axolotl/monkeypatch/ring_attn/patch.py:186-202`）；`BATCH_RING` 会调用 Axolotl 自己的 batch adapter（`src/axolotl/monkeypatch/ring_attn/patch.py:203-211`）。

这说明 CP 的通信组完全来自 mesh 的 `cp` 维度。如果 `build_device_mesh()` 没有把维度命名为 `cp`，这里会直接报错并打印可用维度（`src/axolotl/monkeypatch/ring_attn/patch.py:160-166`）。

### 3.3.4 DataLoader 阶段：让 TP/CP rank 拿同一 batch

Accelerate 的 DataLoader 准备函数会读取 `torch_device_mesh`：

```python
# accelerate/data_loader.py:1119-1155
if torch_device_mesh:
    submesh_tp_size = torch_device_mesh["tp"].size() if "tp" in names else 1
    submesh_cp_size = torch_device_mesh["cp"].size() if "cp" in names else 1
    submesh_dp_size = torch_device_mesh["dp_replicate"].size() if "dp_replicate" in names else 1
    submesh_fsdp_size = torch_device_mesh["dp_shard"].size() if "dp_shard" in names else 1
    process_index = process_index // (submesh_tp_size * submesh_cp_size)
    num_processes = submesh_fsdp_size * submesh_dp_size
```

这段逻辑的注释非常直白：TP/CP rank 需要 same batch；不同 FSDP/DP rank 才拿 different batch（`accelerate/data_loader.py:1132-1141`）。也就是说，mesh 不只影响模型通信，还影响数据调度。

### 3.3.5 FSDP 阶段：沿 flattened `fsdp_dim_names` shard

Accelerate upstream FSDP2 prepare 使用：

```python
# accelerate/utils/fsdp_utils.py:643-652
mesh = getattr(accelerator, "torch_device_mesh", None)
fsdp2_kwargs = {
    ...,
    "mesh": mesh[tuple(accelerator.parallelism_config.fsdp_dim_names)] if mesh is not None else None,
}
```

Axolotl 自己也 patch 了一份 FSDP2 prepare，实现里同样使用 `mesh[tuple(accelerator.state.parallelism_config.fsdp_dim_names)]`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:344-360`）。因此 FSDP2 的 shard 维度取决于 Accelerate `fsdp_dim_names`，不是 Axolotl 另写的 group 逻辑。

## 3.4 关键细节与误区澄清

> 容易误解点 7：Transformers 的 `_prepare_context_parallel_inputs()` 会被调用，但 Axolotl 标准 CP 主切分不是由它完成。

Transformers Trainer 在 training step 里确实调用 `_prepare_context_parallel_inputs()`（`transformers/trainer.py:1890-1896`），它会构造 `accelerator.maybe_context_parallel` 的 buffers（`transformers/trainer.py:2236-2283`）。但 Axolotl 在 `setup_parallelism_envs()` 中调用 `patch_prepare_cp()`（`src/axolotl/utils/trainer.py:632-638`），把 Accelerate `_prepare_cp` 改成 no-op CP context（`src/axolotl/monkeypatch/accelerate/parallelism_config.py:80-98`）。真正切分 input_ids/labels/position_ids 的是 Axolotl 的 forward pre-hook：`apply_sequence_parallelism()`（`src/axolotl/utils/ctx_managers/sequence_parallel.py:24-167`）。

> 容易误解点 8：文档说“data collator handles chunking”，但当前主路径是模型 forward pre-hook chunking。

`docs/sequence_parallelism.qmd:40-45` 写到 data collator 处理 chunking；但源码中 `SequenceParallelContextManager._register_model_hooks()` 注册 `model.register_forward_pre_hook(..., with_kwargs=True)`，在 hook 里调用 `self.apply_sequence_parallelism(updated_kwargs)`（`src/axolotl/utils/ctx_managers/sequence_parallel.py:255-288`）。DataLoader 的作用是让 TP/CP rank 拿到同一 batch；真正沿 sequence 维切 tensor 发生在前向 hook。

> 容易误解点 9：`build_parallelism_config()` 被多处调用不代表它在每个 step 都建 mesh。

模型加载会调用一次，Accelerator 初始化会调用一次；自定义 Muon/Dion optimizer 配置也可能调用一次拿 `device_mesh`（`src/axolotl/core/builders/base.py:299-328`）。训练 step 中不会反复调用 `build_device_mesh()`。每 step 发生的是 CP hook 切分、ring attention 通信、FSDP/TP 下游通信等。

## 3.5 本章小结

> 💡 **小结**
>
> * 同一个 `DeviceMesh` 会被 TP 加载、FSDP2 shard、CP ring attention、DataLoader 调度分别消费。
> * Axolotl CP 的核心切分路径是 `SequenceParallelContextManager` 的 forward hook，不是 upstream torch `context_parallel`。
> * `device_mesh[("cp",)]` 的命名正确性直接决定 ring attention group 是否能创建。
> * DataLoader 会按 TP×CP 缩小数据并行视角，让同组 rank 拿同一 batch。

# 四、完整主路径串联

## 4.1 完整调用栈

```text
User: axolotl train examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml
  │
  ├─ Step 1: CLI 启动
  │     └─ src/axolotl/cli/main.py:train
  │        -> accelerate launch -m axolotl.cli.train
  │
  ├─ Step 2: 配置加载与校验
  │     └─ src/axolotl/cli/config.py:load_cfg
  │        -> validate_config
  │        -> prepare_optim_env
  │        -> normalize_config
  │
  ├─ Step 3: 环境变量与 patch
  │     └─ src/axolotl/utils/trainer.py:setup_parallelism_envs
  │        -> PARALLELISM_CONFIG_TP_SIZE / CP_SIZE / DP_SHARD_SIZE
  │        -> ACCELERATE_USE_PARALLELISM_CONFIG=true
  │        -> patch_prepare_cp()
  │
  ├─ Step 4: 模型加载前 mesh
  │     └─ src/axolotl/loaders/model.py:ModelLoader.load
  │        -> _set_parallel_config
  │        -> src/axolotl/utils/distributed.py:build_parallelism_config
  │        -> accelerate.ParallelismConfig.build_device_mesh("cuda")
  │        -> from_pretrained(tp_plan="auto", device_mesh=device_mesh)  # TP only
  │
  ├─ Step 5: Trainer/Accelerator 初始化
  │     └─ Transformers Trainer.create_accelerator_and_postprocess
  │        -> Accelerator(...)
  │        -> env 触发 ParallelismConfig()
  │        -> state.device_mesh = parallelism_config.get_device_mesh(device.type)
  │
  ├─ Step 6: DataLoader / FSDP / TP prepare
  │     ├─ accelerate.data_loader.prepare_data_loader(..., torch_device_mesh=device_mesh)
  │     ├─ FSDP2: mesh[tuple(parallelism_config.fsdp_dim_names)]
  │     └─ TP: model 已在加载时按 tp_plan 处理；Accelerate 校验 model.tp_size
  │
  ├─ Step 7: 训练 step 前后
  │     └─ src/axolotl/train.py:execute_training
  │        -> SequenceParallelContextManager(..., device_mesh=trainer.accelerator.torch_device_mesh)
  │        -> register_ring_attn_from_device_mesh(device_mesh[("cp",)])
  │        -> model forward pre-hook 切 sequence
  │        -> ring_flash_attn 在 cp group 内通信
  │        -> 可选 all_gather 输出 / eval loss all_reduce
  │
  └─ Step 8: 保存
        ├─ FSDP/parallelism: trainer.accelerator.get_state_dict(model)
        ├─ CP: AxolotlTrainer._save 对 state_dict tensor detach().cpu()
        └─ FSDP2 patch: DTensor.full_tensor() / barrier / rank0 state_dict
```

## 4.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 是否通信 | 是否每 step |
|---|---|---|---|---|
| CLI 启动 | config path、launcher 参数 | 子进程命令与分布式 env | 启动器建立默认进程组前置条件 | 否 |
| 配置加载 | YAML + CLI override | `DictDefault`，并行字段默认值/校验后值 | 否 | 否 |
| parallelism env | `cfg.tensor_parallel_size/context_parallel_size/dp_*` | `PARALLELISM_CONFIG_*`、`ACCELERATE_USE_PARALLELISM_CONFIG` | 否 | 否 |
| Axolotl mesh | `cfg` + `WORLD_SIZE` | `ParallelismConfig`、`DeviceMesh` | `init_device_mesh` 可创建 process groups | 否 |
| 模型加载 TP | `tp_size/tp_plan/device_mesh` | TP 化模型参数/模块状态 | 取决于 Transformers TP 加载 | 否 |
| Accelerator mesh | env | `accelerator.state.device_mesh` | `init_device_mesh` 可创建 process groups | 否 |
| DataLoader | `torch_device_mesh` | TP/CP rank 共享 batch 的调度视角 | 否 | DataLoader 迭代时分发 batch |
| FSDP2 prepare | `fsdp_dim_names` | `fully_shard(..., mesh=...)` | FSDP 后续每层通信 | prepare 否，训练通信是每层 |
| CP forward hook | batch tensors | `[B,S] -> [B,S/cp]` local chunk | hook 本身不通信；token count 可能 all_reduce | 是 |
| Ring attention | local Q/K/V | 全局等价 attention 输出 local chunk | cp group 内 ring 通信，每 attention layer | 是 |
| 保存 | model state_dict | full/sharded state_dict、rank0 输出 | FSDP gather/full_tensor/barrier | 保存时 |

## 4.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| Accelerate CLI `--use_parallelism_config` | Accelerate 自带 launch 参数看起来是入口 | Axolotl 常规 `axolotl train` 不依赖用户显式传它 | Axolotl 自己在 `setup_parallelism_envs()` 写 `ACCELERATE_USE_PARALLELISM_CONFIG=true`。 |
| Accelerate `sp_size` / DeepSpeed SP | 名字叫 sequence parallelism | Axolotl ring attention CP 不走 `sp_size` | Axolotl 用 `context_parallel_size -> cp_size`，`sp_size` 是 DeepSpeed ALST/Ulysses 路径。 |
| Transformers `_prepare_context_parallel_inputs()` | training step 中确实调用 | 不是 Axolotl CP 的主要切分者 | 它构造 context，但 Axolotl patch Accelerate `_prepare_cp` 为 no-op；真正切分在 forward pre-hook。 |
| `docs/sequence_parallelism.qmd` 的 collator chunking 描述 | 文档说 data collator 处理 chunking | 与当前源码主路径不一致 | 当前主路径是 `SequenceParallelContextManager` pre-hook 切 tensor。 |
| `ModelLoader.parallelism_config` | 名字像 Trainer 的配置来源 | 不直接传入 TrainingArguments | 模型加载用；Trainer/Accelerator 主要通过 env 自建 `ParallelismConfig()`。 |
| `tests/e2e/multigpu/test_tp.py` | 看起来覆盖 TP e2e | 当前标记 skip | skip reason 是 tied weights 不兼容（`tests/e2e/multigpu/test_tp.py:17-19`）。 |
| `tests/e2e/multigpu/patched/test_sp.py` | 看起来覆盖 CP e2e | 当前标记 skip | skip reason 是 `ring_flash_attn` 与 transformers imports upstream 问题（`tests/e2e/multigpu/patched/test_sp.py:102-104`）。 |

## 4.4 本章小结

> 💡 **小结**
>
> * 真实主路径包含两次 mesh 构建：一次给模型加载/TP，一次给 Accelerator state。
> * `build_device_mesh()` 是初始化期行为；训练期通信由消费者触发。
> * Axolotl 标准 CP 是“DataLoader 同 batch + forward hook 切序列 + ring attention 通信”。
> * 一些看似主流程的 upstream CP 函数在 Axolotl patch 后退化为兼容壳。

# 五、关键数据流、状态流与 shape 流程

## 5.1 Tensor shape 变化

CP/Sequence Parallel 的 shape 主线在 `apply_sequence_parallelism()`：

```text
原始 batch:
  input_ids      [B, S]
  labels         [B, S]
  attention_mask [B, S]
  position_ids   [B, S]  # 若没有则创建

padding:
  divisor = min(cp_size, 64)
  S' = ceil(S / divisor) * divisor
  input_ids/labels/... [B, S']

按 cp group local_rank 切分:
  rank_i input_ids [B, S' / cp_size]
  rank_i labels    [B, S' / cp_size]

模型前向:
  hidden/logits 等只覆盖 local sequence chunk

可选 gather:
  GRPO/EBFT gather_outputs=True 时，AllGatherWithGrad 沿 dim=1 拼回 [B, S', ...]

去 padding:
  若 pad_len > 0，post-hook 截回 [B, S, ...]
```

源码依据：

- 创建 `position_ids`：`src/axolotl/utils/ctx_managers/sequence_parallel.py:53-64`。
- padding 到可切分长度：`src/axolotl/utils/ctx_managers/sequence_parallel.py:96-133`。
- 沿 sequence 维 `chunk(local_world_size, dim=1)[local_rank]`：`src/axolotl/utils/ctx_managers/sequence_parallel.py:135-148`。
- 可选输出 gather：`src/axolotl/utils/ctx_managers/sequence_parallel.py:350-363`。
- `AllGatherWithGrad` forward 用 `dist.all_gather` 收 shape 与 tensor，再 `torch.cat(..., dim=1)`：`src/axolotl/utils/ctx_managers/sequence_parallel.py:389-416`。
- backward 只取本 rank 对应梯度 slice：`src/axolotl/utils/ctx_managers/sequence_parallel.py:418-444`。

显存收益主要发生在模型前向/反向期间：attention 激活、MLP 输入输出等按 local sequence chunk 变小。输入 batch 在 DataLoader/进入 hook 前仍可能是完整 `[B,S]`，hook 替换 kwargs 后才进入局部 shape。因此它节省的是模型计算路径的激活显存，不是“数据加载阶段永远不持有完整 batch”。

## 5.2 Rank / Mesh / Process Group 变化

以 `world_size=8, dp_shard=2, cp=2, tp=2` 为例：

```text
mesh_dim_names = ("dp_shard", "cp", "tp")
mesh_shape     = (2, 2, 2)

rank layout:

          tp=0  tp=1

dp=0 cp=0   0     1
dp=0 cp=1   2     3

dp=1 cp=0   4     5
dp=1 cp=1   6     7

TP communication:
  [0,1], [2,3], [4,5], [6,7]

CP / ring attention communication:
  [0,2], [1,3], [4,6], [5,7]

FSDP2 shard mesh (fsdp_dim_names = ["dp_shard_cp"]):
  [0,2,4,6] for tp=0
  [1,3,5,7] for tp=1

DataLoader data-parallel view:
  process_index // (tp_size * cp_size)
  num_processes = dp_shard_size * dp_replicate_size
```

`DeviceMesh.__getitem__` 支持按维度名切 submesh，并且文档明确同一 3D mesh 下 `mesh_3d["dp", "cp"]` 与 `mesh_3d["cp", "dp"]` 会得到不同 rank 排布（`device_mesh.py:549-591`）。所以维度名和顺序是语义，不只是 label。

## 5.3 状态切换

这条链路里有三类状态：

```text
环境变量状态:
  写入者: setup_parallelism_envs
  读取者: Accelerate ParallelismConfig.__post_init__ / Accelerator.__init__
  生命周期: 进程级，初始化前必须一致

Accelerator state:
  写入者: Accelerator.__init__
  字段: state.parallelism_config, state.device_mesh
  读取者: trainer.accelerator.torch_device_mesh, FSDP2, DataLoader, CP context

全局/monkey patch 状态:
  RING_ATTN_GROUP: axolotl.monkeypatch.ring_attn.patch module global
  Trainer._prepare_context_parallel_inputs: class method replacement
  Accelerator._prepare_cp: class method replacement
  Accelerate FSDP2 functions: module/class monkey patch
```

`RING_ATTN_GROUP` 是 module-level global，`get_ring_attn_group()` 在未注册时会报错（`src/axolotl/monkeypatch/ring_attn/patch.py:34-41`），`set_ring_attn_group()` 直接改全局变量（`src/axolotl/monkeypatch/ring_attn/patch.py:44-48`）。这在单进程单模型训练里足够简单；但它不是线程局部状态，也没有在 `SequenceParallelContextManager.__exit__()` 恢复。源码里还留下 TODO：退出时未 un-patch attention 和 accelerate functions（`src/axolotl/utils/ctx_managers/sequence_parallel.py:238-245`）。

## 5.4 本章小结

> 💡 **小结**
>
> * CP 的 shape 收益来自 forward hook 后的 `[B,S] -> [B,S/cp]`，不是 DataLoader 阶段天然只取局部序列。
> * rank group 由 canonical mesh order + row-major rank map 决定；TP 通常连续，CP 在 TP 存在时跨步。
> * `dp_shard_cp` flatten 让 FSDP2 shard group 可能包含 CP 维度，这是 mesh 构建里最关键的隐含设计。
> * patch 和 ring attention group 都是进程内全局状态，生命周期管理是维护风险点。

# 六、核心机制深挖

## 6.1 配置归一化：用户配置如何变成真实行为

**它解决什么问题？** 让 YAML 中的 Axolotl 字段变成 Accelerate 能消费的 `ParallelismConfig` sizes，并保证 sizes 与 world size 相乘/相除后闭合。

**为什么不能更简单？** 因为用户可能省略 `dp_shard_size`，还可能组合 HSDP、TP、CP。简单把字段原样传给 Accelerate 会留下未分配 world size，或者让纯 DDP + TP/CP 这种组合静默进入错误调度。

**源码怎么实现？** `_get_parallel_config_kwargs()` 先处理 TP/CP，再自动填 `dp_shard_size`，最后如果 `remaining_world_size > 1` 就报错（`src/axolotl/utils/distributed.py:327-368`）。Accelerate 自身也在 `ParallelismConfig.__post_init__()` 校验 CP/SP 互斥、纯 DP + TP/CP 不支持（`accelerate/parallelism_config.py:328-341`）。

**上下游衔接？** 上游来自 `validate_config()` 和 env；下游进入 `build_device_mesh()`、DataLoader、FSDP2、TP 加载。

**隐藏假设？** `WORLD_SIZE` 是真实总进程数；launcher 的 rank 顺序能表达期望拓扑；没有自动根据节点信息重排 mesh。

**副作用与风险？** `normalize_config()` 会改变 `cfg.batch_size`（`src/axolotl/utils/config/__init__.py:134-142`），日志或用户预期中的 global batch 可能和原始 YAML 不一样。`sequence_parallel_degree` 仍会映射到 `context_parallel_size`，但已 deprecated（`src/axolotl/utils/schemas/validation.py:1508-1514`）。

## 6.2 Flattened mesh：为什么 FSDP 要看 `dp_shard_cp`

**它解决什么问题？** 当 CP 切序列时，如果 FSDP 只沿 `dp_shard` 分片，CP rank 之间仍各自持有参数分片/副本关系较复杂；Accelerate 选择创建 `dp_shard_cp` 这个 joint mesh，让 FSDP 可以把 CP 维度一起纳入 sharding 语义。

**为什么不能只用 `dp_shard`？** 从源码行为看，Accelerate `fsdp_dim_names` 总是包含 `dp_shard_cp`，而 `dp_shard_cp_dim_names` 在 CP enabled 时返回 `['dp_shard', 'cp']`（`accelerate/parallelism_config.py:136-143`、`157-164`）。这是基于源码行为确认的设计：FSDP2 prepare 读取的就是 `mesh[tuple(fsdp_dim_names)]`（`accelerate/utils/fsdp_utils.py:643-652`；Axolotl patch 同样见 `src/axolotl/monkeypatch/accelerate/fsdp2.py:351-360`）。

**上下游衔接？** `build_device_mesh()` 创建 flatten 名称（`accelerate/parallelism_config.py:237-242`），FSDP2 prepare 使用这个名称，保存时 `get_state_dict()` 又要处理由 FSDP2/DTensor 产生的 sharded state。

**隐藏假设？** CP 与 FSDP 的组合希望把参数 shard group 扩展到 CP 维度；这提升参数/优化器状态分片范围，但也可能改变通信范围。

**副作用与风险？** 如果 CP group 跨节点，`dp_shard_cp` flatten 后的 FSDP group 也可能跨节点，参数 all-gather/reduce-scatter 的成本会上升。源码没有自动拓扑感知重排 rank。

## 6.3 Monkey Patch：零侵入接入还是维护风险？

Axolotl 对 Accelerate/Transformers 的 patch 分三类：

1. **放宽校验 / 允许 standalone CP。** `patch_parallelism_config()` 替换 `ParallelismConfig._validate_accelerator`，并修补 `AcceleratorState.is_fsdp2`（`src/axolotl/monkeypatch/accelerate/parallelism_config.py:11-78`）。它允许 `cp_size > 1` 且没有 dp_shard 时，通过 `ACCELERATE_ALLOW_CP_STANDALONE=true` 放行（`src/axolotl/monkeypatch/accelerate/parallelism_config.py:28-45`）。
2. **把 upstream torch CP prepare 变成 no-op。** `patch_prepare_cp()` 替换 `Accelerator._prepare_cp`，设置 `_cp_context` 为 yield-only context（`src/axolotl/monkeypatch/accelerate/parallelism_config.py:80-98`）。
3. **放宽 Transformers attention guard。** `patch_prepare_context_parallel_inputs()` 把 Trainer 里 “只允许 sdpa” 的 guard 替换为允许 `sdpa` 或 `flash_attention_2`（`src/axolotl/monkeypatch/transformers/trainer_context_parallel.py:15-71`）。测试验证了 guard 被替换且 patch 幂等（`tests/monkeypatch/test_trainer_context_parallel_patch.py:36-66`）。

patch 发生在 `PatchManager.apply_pre_model_load_patches()` 中：`_apply_transformers_patches()` 在 CP 开启时 patch Trainer（`src/axolotl/loaders/patch_manager.py:135-149`），`_apply_fsdp_patches()` 在 CP 或 FSDP2 时 patch Accelerate parallelism config（`src/axolotl/loaders/patch_manager.py:270-286`）。另外 `setup_parallelism_envs()` 也会在 CP 开启时 patch `Accelerator._prepare_cp`（`src/axolotl/utils/trainer.py:632-638`）。

**隐藏假设与风险：** 这些 patch 是进程级替换，不是 context-local。`SequenceParallelContextManager.__exit__()` 只移除 model hooks，不恢复 attention/accelerate patch（`src/axolotl/utils/ctx_managers/sequence_parallel.py:238-245`）。如果同一进程中先后训练不同模型或测试不恢复 monkey patch，就可能污染后续路径。测试里专门用 fixture 恢复 Trainer 方法（`tests/monkeypatch/test_trainer_context_parallel_patch.py:13-33`），说明维护者也意识到这一点。

## 6.4 通信原语：哪些通信来自 mesh，哪些来自消费者

`build_device_mesh()` 初始化时会为每个 mesh dimension 创建 process group：PyTorch `DeviceMesh._init_process_groups()` 遍历每个维度调用 `_init_one_process_group()`（`device_mesh.py:472-494`），后者可能用 `split_group` 或 `new_group` 创建 subgroup（`device_mesh.py:420-469`）。这是一次性初始化通信/建组成本。

训练期通信来自消费者：

- **CP ring attention**：`register_ring_attn_from_device_mesh()` 取 `sequence_mesh.get_group()`（`src/axolotl/monkeypatch/ring_attn/patch.py:168-184`），后续 ring-flash-attn 在这个 group 内传 KV。
- **CP token/loss 修正**：`apply_sequence_parallelism()` 对 `num_items_in_batch` 做 `dist.all_reduce(..., AVG, group=cp_group)`（`src/axolotl/utils/ctx_managers/sequence_parallel.py:150-165`）；eval loss correction 做两个 `SUM all_reduce`（`src/axolotl/utils/ctx_managers/sequence_parallel.py:305-340`）。
- **输出 gather**：`AllGatherWithGrad.forward()` 先 all_gather shape，再 all_gather tensor，沿 sequence dim 拼接（`src/axolotl/utils/ctx_managers/sequence_parallel.py:389-416`）；backward 不再 all-reduce，只切回本 rank 梯度 slice（`src/axolotl/utils/ctx_managers/sequence_parallel.py:418-444`）。
- **FSDP2**：通信由 `fully_shard(..., mesh=...)` 后的 FSDP runtime 触发，典型是参数 all-gather、梯度 reduce-scatter/reshard；Axolotl 源码只负责选择 mesh（`src/axolotl/monkeypatch/accelerate/fsdp2.py:351-360`）。
- **保存**：Axolotl patch 的 `get_state_dict()` 对 FSDP2 DTensor 调 `full_tensor()`，rank0 收 CPU state_dict，并在每个参数后 barrier（`src/axolotl/monkeypatch/accelerate/fsdp2.py:158-173`）。

## 6.5 本章小结

> 💡 **小结**
>
> * 配置归一化解决 world size 分解，flattened mesh 解决 FSDP 与 CP 的 joint shard 语义。
> * Axolotl 用 monkey patch 把 upstream torch CP 路径改成兼容壳，保留 Trainer 入口但接管实际 sequence slicing。
> * 通信不是 `build_device_mesh()` 直接触发的，而是 CP/FSDP/TP/DataLoader/save 等消费者按命名维度触发。
> * patch 的全局性和上游源码依赖是这个实现的主要维护成本。

# 七、显存、性能与通信分析

## 7.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ✅ / 取决于 FSDP/TP | `build_device_mesh()` 本身不省；FSDP2 沿 `fsdp_dim_names` shard 参数，TP 加载可切层权重。 |
| 梯度 | ✅ / 取决于 FSDP | FSDP shard/reduce-scatter 负责；CP 本身不切参数梯度。 |
| optimizer state | ✅ / 取决于 FSDP/优化器 | FSDP shard optimizer state；Muon/Dion 可拿 `device_mesh` 做分布式优化器配置（`src/axolotl/core/builders/base.py:299-328`）。 |
| 激活值 | ✅ | CP pre-hook 把 sequence 切成 `[B,S/cp]`，模型前向只处理 local chunk。 |
| attention KV/score 中间量 | ✅ | Ring attention 让每 rank 只保留 local query chunk，并通过 ring 看其他 KV。 |
| logits | ✅ / ❌ | 普通 SFT 若不 gather，则 local logits 变小；GRPO/EBFT `gather_outputs=True` 会 all-gather 拼回全序列（`src/axolotl/train.py:217`、`sequence_parallel.py:350-363`）。 |
| 输入 batch | ⚠️ 部分 | DataLoader 让 CP rank 拿同一 batch，hook 前仍可能存在完整 `[B,S]`；进入模型后才是 local chunk。 |
| 保存时 state_dict | ❌ / 峰值可能上升 | FSDP2 full state dict 需要 `full_tensor()` 和 rank0 CPU 聚合；CP 保存还会把 tensor clone/detach 到 CPU。 |
| 中间通信 buffer | ❌ | Ring attention、all_gather、FSDP gather 都引入通信 buffer。 |

真正的大头取决于模型与序列长度：长上下文场景里，activation 和 attention 中间量通常是 CP 的收益点；超大参数场景里，FSDP/TP 才是参数/优化器状态的收益点。mesh 构建把这些维度组合起来，但显存收益来自后续算子。

## 7.2 通信开销

| 阶段 | 通信类型 | group | 频率 | 源码依据 |
|---|---|---|---|---|
| mesh 初始化 | process group 创建 / split | 每个 mesh dimension | 初始化一次 | `device_mesh.py:472-494`、`376-469` |
| DataLoader | 无直接 collective | 通过 process_index 调度 | 每个 dataloader 准备 | `accelerate/data_loader.py:1119-1155` |
| CP forward | Ring KV 交换 | `device_mesh[("cp",)]` | 每个 attention layer | `ring_attn/patch.py:168-184`、`186-211` |
| CP token count | `all_reduce(AVG)` | CP group | 有 `num_items_in_batch` 时 | `sequence_parallel.py:150-165` |
| Eval loss | 两次 `all_reduce(SUM)` | CP group | eval forward 后 | `sequence_parallel.py:305-340` |
| 输出 gather | shape `all_gather` + tensor `all_gather` | CP group | `gather_outputs=True` 时 | `sequence_parallel.py:389-416` |
| FSDP2 | 参数 all-gather / 梯度 reduce-scatter 等 | `fsdp_dim_names` 对应 mesh | 每层/每 wrapper | mesh 选择见 `fsdp2.py:351-360`，通信由 FSDP runtime 触发 |
| 保存 | `DTensor.full_tensor()`、barrier | FSDP mesh / world | 保存时每参数 | `fsdp2.py:158-173` |

需要特别关注的是 `dp_shard_cp`：如果 FSDP shard group 包含 CP 维度，那么 FSDP 参数通信与 CP ring attention 通信会共享部分 rank，但语义不同。前者是参数/梯度通信，后者是 attention KV 通信；它们不能简单 overlap，实际性能取决于 wrapper 粒度、NCCL 调度与计算图。

## 7.3 性能取舍

这个实现本质上是几组 trade-off：

- **通信换显存。** CP 用 ring attention 的跨 rank KV 通信换取 sequence activation 显存下降；FSDP 用参数 all-gather/reduce-scatter 换参数/优化器状态显存下降。
- **命名 mesh 换组合性。** `DeviceMesh` 让 FSDP、TP、CP、DataLoader 共享拓扑语义，但要求维度命名和 rank 排列严格一致。
- **patch 复杂度换兼容性。** Axolotl 通过 patch 复用 Transformers Trainer 的入口，同时接管 CP 实现；这降低侵入性，但上游函数改名/改源码字符串时会破。
- **rank0/full_tensor 保存换通用输出。** full state dict 易于加载，但保存时 rank0 CPU 内存和 barrier 串行化可能成为瓶颈。
- **row-major rank map 换确定性。** 默认 rank 排列简单稳定，但不自动拓扑感知。多机时若 launcher rank 不是按节点连续排列，TP/CP/FSDP group 可能跨慢链路。

## 7.4 本章小结

> 💡 **小结**
>
> * `build_device_mesh()` 是性能结果的“拓扑前提”，真正显存节省由 FSDP/TP/CP 消费者产生。
> * CP 主要节省长序列激活和 attention 中间量，不直接节省参数/optimizer state。
> * `dp_shard_cp` 可能扩大 FSDP shard group，参数通信范围要结合 rank 拓扑评估。
> * 保存 full state dict 是典型瓶颈：rank0 聚合、CPU clone、barrier 都可能拉高尾延迟。

# 八、配置项、边界条件与坑点

## 8.1 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `context_parallel_size > 1` | `validation.py:1508-1579`、`trainer.py:632-638`、`distributed.py:334-336` | 映射为 `cp_size`，启用 ring_flash_attn 校验、CP env、CP patch、SequenceParallelContextManager | 必须 `flash_attention: true`；需要 `ring_flash_attn`；不是 Accelerate `sp_size`。 |
| `tensor_parallel_size > 1` | `distributed.py:330-332`、`model.py:749-755` | 映射为 `tp_size`；模型加载传 `tp_plan="auto"` 和 `device_mesh` | tied embeddings 会被拒绝（`loaders/utils.py:139-148`）；部分 bitsandbytes optimizer 不支持（`validation.py:1600-1608`）。 |
| `dp_shard_size` | `distributed.py:347-357` | 明确 FSDP shard 维度大小 | 非 FSDP 配置下设置会报错（`distributed.py:347-352`）。 |
| 省略 `dp_shard_size` 且 FSDP | `distributed.py:338-341`、`359-362` | 剩余 world size 自动归入 `dp_shard_size` | 用户可能以为只开 TP/CP，实际也得到 FSDP shard 维度。 |
| `dp_replicate_size > 1` | `distributed.py:343-357` | 形成 HSDP replication 维度 | Accelerate 禁止 pure DP + TP/CP，即 `dp_replicate>1, dp_shard==1, tp/cp>1`（`accelerate/parallelism_config.py:336-341`）。 |
| `fsdp_version` / `fsdp_config` | `model.py:196-211`、`patch_manager.py:279-293` | FSDP2 时保留 parallel config，并 patch FSDP2 prepare/save/load | Axolotl 对 FSDP1 不启用 `use_parallel_config`（`model.py:207-208`）。 |
| `flash_attention` | `validation.py:1516-1520` | CP 必须开启 | 未开启直接报错。 |
| `sample_packing + micro_batch_size > 1` | `validation.py:1522-1526` | CP + sample packing 限制 micro batch | ring-flash-attn 要求导致。 |
| `ring_attn_func` | `validation.py:1563-1579`、`ring_attn/patch.py:186-211` | 决定 VARLEN_LLAMA3 或 BATCH_RING patch | 默认值取决于 `sample_packing`；某些函数仍在 TODO 中。 |
| `heads_k_stride` | `config.py:981-985`、`ring_attn/patch.py:200-202` | 传给 ring_flash_attn K head stride | 必须整除 KV heads；值大可能更快但更占内存。 |
| `sequence_parallel_degree` | `config.py:969-974`、`validation.py:1508-1514` | deprecated alias 到 `context_parallel_size` | 仍生效但会 warning；新配置应改用 `context_parallel_size`。 |
| `deepspeed + tensor_parallel_size` | `validation.py:1121-1148` | 改写 DeepSpeed json，加入 `tensor_parallel.autotp_size` 与 save gather 配置 | 会写临时 json；与 FSDP/Transformers TP 是不同路径。 |
| `fsdp_config.cpu_ram_efficient_loading` | `model.py:756-780`、`fsdp2.py:371-425` | rank0/非 rank0 加载策略、meta sharding、broadcast full state dict | QLoRA/bnb Params4bit 有特殊绕过；加载/广播可能有 CPU/GPU 峰值。 |
| `include_tkps` | `causal.py:72-79`、`tokens_per_second.py:33-38` | token/s 统计除以 TP×CP 非 DP 并行度 | 只是指标修正，不影响 mesh 构建。 |

## 8.2 默认行为与静默失效条件

默认情况下 `context_parallel_size`、`tensor_parallel_size` 都会在 validation 后变成 1（`validation.py:1501-1516`），这意味着不会写 `PARALLELISM_CONFIG_CP_SIZE/TP_SIZE`，也不会进入 mesh 构建路径。

可能的“看起来配了但没走预期路径”：

- `fsdp_config` 存在但 `fsdp_version != 2`：`ModelLoader` 会把 `use_parallel_config=False`（`src/axolotl/loaders/model.py:207-208`）。
- `tensor_parallel_size > 1` 且模型 `tie_word_embeddings=True`：`load_model_config()` 会报错而不是进入 TP（`src/axolotl/loaders/utils.py:139-148`）。
- `context_parallel_size > 1` 但未安装 `ring_flash_attn`：validation 抛 ImportError（`src/axolotl/utils/schemas/validation.py:1528-1550`）。
- Accelerate `sp_size`：Axolotl 没有把 `context_parallel_size` 映射到它；如果用户通过外部 env 开了 `sp_size`，`build_device_mesh()` 可能因为 DeepSpeed SP 返回 `None`（`accelerate/parallelism_config.py:218-221`），这不是 Axolotl ring attention 主路径。

## 8.3 单机 / 多机差异

源码没有根据 host/node 自动重排 mesh。`init_device_mesh()` 使用 global rank 的 row-major `arange`（`device_mesh.py:1357-1361`），因此多机上“TP 是否留在节点内、CP 是否跨节点、FSDP shard 是否跨节点”取决于 launcher 的 rank 排布与用户选择的 dimension sizes。

示例文档建议 HSDP + TP 用 `dp_shard_size=4, tensor_parallel_size=2, dp_replicate_size=2`，意图是 FSDP/TP 在节点内、replicate 跨组（`examples/distributed-parallel/README.md:36-45`）。但这只是配置意图；源码不会检测硬件拓扑是否真的匹配。

## 8.4 本章小结

> 💡 **小结**
>
> * 最小 CP 配置不是只写 `context_parallel_size`，还要满足 flash attention、ring_flash_attn、sample packing 限制。
> * TP 在当前源码中有 Transformers TP 与 DeepSpeed AutoTP 两条路径，schema 描述不足以概括实际行为。
> * `dp_shard_size` 省略时可能被自动填充，这会改变 FSDP group 与显存/通信行为。
> * 多机拓扑匹配依赖 launcher rank 顺序，Axolotl/Accelerate 不会自动优化 rank placement。

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/test_loaders.py::test_get_parallel_config_kwargs` | `_get_parallel_config_kwargs()` 的 world-size 分解 | 覆盖 TP/CP/dp_shard/dp_replicate 的 kwargs 生成（`tests/test_loaders.py:181-218`）。 |
| `tests/test_context_parallel_batch_size.py` | CP 下 batch size 按有效 DP world size 缩放 | mock `ring_flash_attn`，验证 `normalize_config()` 结果（`tests/test_context_parallel_batch_size.py:29-56`）。 |
| `tests/test_tensor_parallel_batch_size.py` | TP 下 batch size 缩放 | mock model config，避免下载模型和 tied embedding 校验（`tests/test_tensor_parallel_batch_size.py:28-55`）。 |
| `tests/monkeypatch/test_trainer_context_parallel_patch.py` | Trainer CP guard patch | 验证 guard 替换与 patch 幂等（`tests/monkeypatch/test_trainer_context_parallel_patch.py:36-66`）。 |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml` | FSDP + TP + CP 示例配置 | 展示 8 GPU 单节点三维并行配置（`examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:6-19`）。 |
| `examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml` | HSDP + TP 示例配置 | 展示 `dp_replicate × dp_shard × tp`（`examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml:6-22`）。 |
| `docs/nd_parallelism.qmd` | N-D parallelism 概念与 support matrix | 说明 FSDP/TP/CP/HSDP 组合与限制（`docs/nd_parallelism.qmd:51-108`）。 |
| `tests/e2e/multigpu/solo/test_grpo.py` / `test_gdpo.py` | RL 路径中 CP 配置 | GRPO/GDPO 测试配置包含 `context_parallel_size: 2`（如 `tests/e2e/multigpu/solo/test_grpo.py:300-319`，`test_gdpo.py:443-463`）。 |

## 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| `build_device_mesh()` 的 `mesh_dim_names/mesh_shape` 精确顺序 | 未见 Axolotl 单测直接断言 `_get_mesh()` 或 `device_mesh.mesh_dim_names` | Accelerate 上游顺序变化会改变 rank group，Axolotl 测试可能只在运行时才暴露。 |
| `dp_shard_cp` flatten 后 FSDP group 是否符合预期 | 未见直接断言 | FSDP 参数通信范围可能与用户/文档预期不一致。 |
| TP e2e | 当前 `tests/e2e/multigpu/test_tp.py` 被 skip（`tests/e2e/multigpu/test_tp.py:17-19`） | TP 加载/保存/训练回归可能缺少 CI 保护。 |
| CP SFT e2e | 当前 `tests/e2e/multigpu/patched/test_sp.py` 被 skip（`tests/e2e/multigpu/patched/test_sp.py:102-104`） | ring attention + hook + save 主路径缺少稳定 e2e。 |
| 多机 rank 拓扑 | 未见多机 mesh 排列测试 | TP/CP group 跨节点导致性能严重下降但功能上不报错。 |
| 保存 / resume + CP/TP/FSDP 组合 | 只有分散路径，未见三维组合端到端断言 | full state dict、adapter rename、DTensor full_tensor、CPU clone 可能在组合场景下回归。 |
| patch 恢复 | Trainer patch 有测试 fixture；Accelerate/ring attention patch 恢复未见同等覆盖 | 同进程多测试/多训练任务可能互相污染。 |
| 性能/显存收益 | 未见基准断言 | 功能可用但通信开销或峰值显存不符合预期时难以及时发现。 |

## 9.3 本章小结

> 💡 **小结**
>
> * 当前单测较好覆盖了配置归一化与 batch size 缩放。
> * mesh 命名、rank 排列、flattened group 语义缺少直接测试。
> * TP 和 CP 的关键 e2e 当前存在 skip，三维组合的保存/resume 风险仍然较高。
> * 文档/示例说明了意图，但不能替代对真实 `DeviceMesh` group 的断言。

# 十、局限性与已知优化点

## 10.1 硬约束

1. **world size 必须能被并行维度整除并完全分解。** `_get_parallel_config_kwargs()` 最终 `remaining_world_size > 1` 会报错（`src/axolotl/utils/distributed.py:359-368`）。
2. **CP 需要 flash attention 与 ring_flash_attn。** validation 明确要求 `flash_attention: true`（`validation.py:1516-1520`）并导入 `ring_flash_attn`（`validation.py:1528-1550`）。
3. **sample packing + CP 要求 `micro_batch_size=1`。** 源码直接报错（`validation.py:1522-1526`）。
4. **CP 与 Accelerate SP 互斥。** Accelerate `ParallelismConfig` 禁止 `cp_size > 1` 且 `sp_size > 1`（`accelerate/parallelism_config.py:328-334`）。
5. **pure DP + TP/CP 不支持。** Accelerate 禁止 `dp_replicate_size > 1 && dp_shard_size == 1 && (tp/cp)>1`（`accelerate/parallelism_config.py:336-341`）。
6. **TP 与 tied embeddings 不兼容。** Axolotl 在 load model config 时检查并报错（`src/axolotl/loaders/utils.py:139-148`）。
7. **部分 bitsandbytes optimizer 不支持 TP。** `paged_adamw_8bit`、`adamw_8bit`、`adamw_bnb_8bit` 会报错（`validation.py:1600-1608`）。
8. **PyTorch 版本要求。** Accelerate `build_device_mesh()` 要求 torch >= 2.2（`accelerate/parallelism_config.py:223-226`）；Axolotl 当前依赖已要求 `torch>=2.9.1`（`pyproject.toml:15`）。
9. **`init_device_mesh()` 的 device_type 不能带 index。** PyTorch 对 `cuda:0` 这种 device_type 会报错（`device_mesh.py:1350-1355`）；Axolotl 固定传 `"cuda"`（`src/axolotl/utils/distributed.py:313`）。

## 10.2 维护成本

- **源码字符串 patch 脆弱。** `patch_prepare_context_parallel_inputs()` 依赖原始源码包含精确字符串 `if model.config._attn_implementation != "sdpa":`（`trainer_context_parallel.py:15-38`）。上游 Trainer 改写 guard 后 patch 会跳过。
- **进程级 patch 难恢复。** `Accelerator._prepare_cp`、`ParallelismConfig._validate_accelerator`、ring flash attention adapter 都是全局替换。`SequenceParallelContextManager.__exit__()` 目前只移除 hook，并 TODO 说明未 un-patch（`sequence_parallel.py:238-245`）。
- **两条 mesh 构建路径需要保持一致。** ModelLoader 自建 mesh，Accelerator 又从 env 自建 mesh；如果未来某处加入额外 kwargs 或 env，可能产生不一致。
- **文档与源码已有轻微偏差。** docs 说 data collator chunking（`docs/sequence_parallelism.qmd:40-45`），当前源码是 forward hook chunking。
- **依赖上游 ParallelismConfig 语义。** `dp_shard_cp`、`fsdp_dim_names`、canonical order 都来自 Accelerate；上游升级会直接影响 Axolotl。

## 10.3 性能瓶颈

- **CP 每层 ring attention 通信。** 长序列降低显存，但每个 attention layer 都要在 CP group 内传 KV。
- **FSDP shard group 可能变大。** `dp_shard_cp` 把 CP 纳入 FSDP shard group 后，参数通信范围可能扩大。
- **输出 gather 可能抵消 logits/activation 节省。** `gather_outputs=True` 时 all-gather 拼回完整 sequence，GRPO/EBFT 路径尤其要注意。
- **保存 full state dict 串行化。** Axolotl patch 的 FSDP2 `get_state_dict()` 每个参数 `full_tensor()` 后 rank0 CPU 保存，并 barrier（`fsdp2.py:158-173`）。大模型保存可能慢且占 CPU 内存。
- **rank 排列不拓扑感知。** 如果 TP group 跨节点，频繁小通信会很慢；如果 CP/FSDP group 跨节点，也会影响 ring/FSDP 通信。

## 10.4 已知优化点

源码中的 TODO 与可推断优化方向包括：

- `SequenceParallelContextManager.__exit__()` TODO：未来可以恢复 attention/accelerate patch，降低测试污染风险（`sequence_parallel.py:238-245`）。
- `apply_sequence_parallelism()` TODO：当前主要关注 batch ring 与 varlen llama3，zigzag/stripe pattern 还未完整实现（`sequence_parallel.py:22-23`）。
- `ModelLoader._build_model()` 中 TP workaround：Transformers 4.54.0 未设置 `_tp_size/_device_mesh` 的 workaround 有 TODO 移除（`src/axolotl/loaders/model.py:852-857`）。
- FSDP2 保存路径可优化：当前 per-param `full_tensor()` + barrier 简单可靠，但可考虑分块、异步、减少 barrier 或直接 sharded checkpoint 后处理。
- mesh rank placement 可增强：允许用户显式指定 rank layout 或按 node/local_rank 自动排列，让 TP 优先留在 NVLink 内、DP replicate 跨节点。
- 对 `build_device_mesh()` 结果加单测：直接断言 `mesh_dim_names`、`mesh_shape`、关键 submesh ranks，可在上游 Accelerate 改动时快速报警。

## 10.5 本章小结

> 💡 **小结**
>
> * 当前实现依赖严格的 world-size 分解、flash attention/ring_flash_attn、FSDP2/Transformers TP 约束。
> * 维护风险主要来自 monkey patch、两套 mesh 构建路径和上游 Accelerate 语义变化。
> * 性能瓶颈集中在 CP 每层通信、FSDP flattened group 通信和 full state dict 保存。
> * 最值得补的工程保护是 mesh rank/group 单测、多机拓扑验证、保存/resume e2e。

# 小结与展望

Axolotl 的 `Accelerate ParallelismConfig` mesh 构建实现，可以用几个关键词概括。

## 关键词一：配置分解

用户写的是 `context_parallel_size/tensor_parallel_size/dp_shard_size/dp_replicate_size`，源码先把它们分解为 Accelerate 的 `cp_size/tp_size/dp_*`，再校验 world size 是否完全闭合。这里的关键不是“字段搬运”，而是把非数据并行维度从 global batch 语义里拿掉。

## 关键词二：命名 mesh

`build_device_mesh()` 的核心价值是命名：`dp_shard`、`cp`、`tp`、`dp_shard_cp`。后续模块不是按裸 rank 列表通信，而是按名字切 submesh。这个设计让 FSDP、TP、CP、DataLoader 能共享拓扑语义。

## 关键词三：canonical rank order

Accelerate 固定使用 `dp_replicate -> dp_shard -> cp -> sp -> tp`，PyTorch 默认 row-major rank map。因此 TP rank 通常连续，CP rank 在 TP 存在时跨步，FSDP 的 `dp_shard_cp` flatten 可能跨 CP 维度。这是理解通信组的关键。

## 关键词四：patch 接管

Axolotl 没有完全采用 upstream torch context parallel，而是 patch 掉 Accelerate `_prepare_cp`，保留 Trainer 入口，再用自己的 `SequenceParallelContextManager` 做前向切分和 ring attention 注入。这让它可以用 FlashAttention/ring-flash-attn 处理长序列，但也带来全局 patch 的维护成本。

## 关键词五：通信换显存

CP 降低长序列激活显存，FSDP/TP 降低参数相关显存；代价是更多、更复杂的通信：CP ring、FSDP gather/reduce、TP collectives、保存 full_tensor。mesh 不直接省显存，但它决定这些通信发生在哪些 rank 之间。

这个实现适合：

- 单机多 GPU 的长上下文 SFT，尤其是希望 CP/TP/FSDP 组合时；
- FSDP2 + TP/CP 的实验性 N-D 并行；
- 需要用 Axolotl YAML 管理复杂并行配置的训练任务。

它不太适合：

- rank 拓扑不可控的多机环境，尤其 TP/CP 可能跨慢链路时；
- 强依赖稳定 e2e 覆盖的生产训练，目前 TP/CP e2e 有 skip；
- 同一 Python 进程中反复切换不同并行模式的场景，patch/global state 恢复还不完整。

与手写 process group 相比，`ParallelismConfig + DeviceMesh` 的好处是语义清晰、下游复用度高；代价是必须理解 Accelerate 的维度命名和 rank 排列规则。后续值得继续走读的方向包括：Transformers TP 如何消费 `device_mesh`、FSDP2 DTensor 保存/加载路径、以及 Axolotl ring attention patch 与不同模型 attention 实现的兼容边界。
