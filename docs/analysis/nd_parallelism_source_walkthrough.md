# Axolotl 源码走读：ND Parallelism（CP + TP + FSDP）实现解析

在长序列和大模型训练里，“单个 GPU 放不下”通常不是一个单点问题：参数可能放不下，长上下文激活也可能放不下；多机时，跨节点通信又会把本来能跑通的方案拖慢。Axolotl 在 2025/07 的更新中明确写到：ND Parallelism 已接入，可以在单机和多机内组合 Context Parallelism（CP）、Tensor Parallelism（TP）和 Fully Sharded Data Parallelism（FSDP）（`README.md:49-56`）。

这篇文章不把 ND Parallelism 讲成“配置表大全”，而是顺着一次真实训练调用去看：用户写进 YAML 的几个 size，如何变成 Accelerate 的 `ParallelismConfig`、PyTorch `DeviceMesh`、Transformers TP、Axolotl 的 sequence-parallel forward hook、ring-flash-attention 的 monkey patch，以及 FSDP2 的 `fully_shard` 与保存路径。读完这条主线，读者应该能判断：哪些显存真的被省掉，哪些通信是新增的，哪些配置只是把行为交给下游库，哪些 patch 是为了绕过当前上游限制。

# 前言

## 业务 / 工程背景

Axolotl 是一个以 YAML 配置驱动的 LLM 微调框架。用户最常见的入口不是手写训练脚本，而是：

```bash
axolotl train examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml
```

ND Parallelism 出现在训练主链路里，目标是把几类瓶颈拆到不同维度上处理：

- FSDP：解决参数、梯度、optimizer state 的冗余。
- TP：把模型层内部的矩阵计算切到多个 rank 上。
- CP：把长序列沿 sequence 维切开，降低单卡激活和 logits 压力。
- HSDP / 多机：在节点内做 FSDP，在节点间复制，减少频繁跨节点参数 gather。

## 核心矛盾

ND Parallelism 背后的核心冲突可以概括为三句话：

1. FSDP 能分参数，但它不天然减少长序列 attention / logits 的 sequence 维显存。
2. CP 能切 sequence，但 attention 不是局部算子，必须通过 ring attention 让每个 rank 看到全局 KV。
3. TP/FSDP/CP 必须共享同一套 rank 拓扑；否则数据分发、参数 sharding、attention group 会互相踩错。

Axolotl 的实现选择不是自己重写完整分布式运行时，而是在几个关键点上“接管”：配置归一化、环境变量、DeviceMesh 构建、forward hook、ring attention patch、FSDP2 patch 和最终保存。

## 本文主线

本文按机制而不是按文件展开：

1. 配置如何从 YAML 变成真实行为；
2. DeviceMesh / ProcessGroup 如何表达 ND 拓扑；
3. TP 与 FSDP2 如何在模型初始化和 prepare 阶段接入；
4. CP 如何在 forward 前后切分 / 修正 / 可选 gather；
5. ring-flash-attention 如何被注入；
6. 保存与 state_dict 为什么需要额外 patch；
7. 最后串起一次完整训练调用，并分析 shape、rank、通信、显存、测试覆盖和坑点。

## 不展开的内容

本文不讲 FSDP、Megatron TP、Ring Attention 或 LoRA 的完整理论；这些机制的数学和论文背景不在这里展开。本文只分析 Axolotl 如何把这些机制接进自己的训练链路，以及源码中能确认的收益、代价和风险。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/main.py` / `src/axolotl/cli/train.py` | 用户训练入口：启动 launcher、加载配置、进入训练主函数。 |
| `src/axolotl/cli/config.py` | YAML/CLI 配置读取、校验、环境准备、归一化。 |
| `src/axolotl/utils/schemas/config.py` / `validation.py` | ND/FSDP/CP/TP 配置字段和约束。 |
| `src/axolotl/utils/trainer.py` | 写入 FSDP 与 ParallelismConfig 相关环境变量。 |
| `src/axolotl/utils/distributed.py` | 从 Axolotl 配置推导 Accelerate `ParallelismConfig` 和 `DeviceMesh`。 |
| `src/axolotl/loaders/model.py` | 模型加载时接入 TP、FSDP cpu/meta loading、DeviceMesh。 |
| `src/axolotl/utils/ctx_managers/sequence_parallel.py` | CP 主执行逻辑：forward hook 切分输入、修正 loss、可选输出 gather。 |
| `src/axolotl/monkeypatch/ring_attn/patch.py` | 从 DeviceMesh 注册 CP group，并替换 HF flash attention 为 ring attention。 |
| `src/axolotl/monkeypatch/accelerate/fsdp2.py` | FSDP2 prepare、state_dict、LoRA/QLoRA 兼容 patch。 |
| `src/axolotl/train.py` | 训练执行、SequenceParallelContextManager 进入点、最终保存路径。 |

> 💡 **小结**
>
> * ND Parallelism 不是一个单独模块，而是一组配置、拓扑、patch、hook 和保存逻辑的组合。
> * Axolotl 把拓扑交给 Accelerate / PyTorch，把模型内 TP 交给 Transformers，把 CP attention 交给 ring-flash-attn。
> * 框架自己真正强接管的是配置归一化、CP forward hook、FSDP2 兼容 patch 和最终保存。

# 一、入口与配置归一化：把“几维并行”变成可执行意图

## 1.1 设计哲学与核心问题

用户只想写：`dp_shard_size: 2`、`tensor_parallel_size: 2`、`context_parallel_size: 2`。但运行时需要回答的问题远比这多：

- launcher 应该启动多少进程？
- FSDP2 是否启用，是否需要 CPU RAM efficient loading？
- TP 是 DeepSpeed AutoTP，还是 Transformers 原生 TP？
- CP 是否需要 flash attention、ring-flash-attn、特殊 batch size 约束？
- 全局 batch size 是否还等于 `micro_batch_size * world_size * grad_accum`？

Axolotl 的第一层职责就是把“用户意图”翻译成三类状态：Pydantic 校验后的 config、环境变量、归一化后的 batch / dtype / device 字段。

如果没有这一层，后面会出现两个典型问题：第一，Accelerate 无法从环境变量知道要创建 `ParallelismConfig`；第二，Trainer 仍按 world_size 个不同 batch 计算 global batch，而 CP/TP 的同组 rank 实际在协同处理同一个 batch 或同一个 layer。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：用户 `axolotl train` click 入口，决定 launcher。

src/axolotl/cli/utils/train.py
  - _launch_accelerate_training / _launch_torchrun_training：拼出 `accelerate launch` 或 `torchrun -m axolotl.cli.train`。

src/axolotl/cli/train.py
  - do_cli：加载配置，进入 do_train。
  - do_train：加载数据集，调用 axolotl.train.train。

src/axolotl/cli/config.py
  - load_cfg：读取 YAML、应用 CLI override、validate_config、prepare_optim_env、normalize_config。

src/axolotl/utils/schemas/config.py
  - AxolotlInputConfig 中的 fsdp / dp_shard_size / dp_replicate_size / context_parallel_size / tensor_parallel_size 字段。

src/axolotl/utils/schemas/validation.py
  - check_context_parallel_size / validate_ring_attn_func / check_tensor_parallel_optimizer / FSDP 相关 validators。

src/axolotl/utils/trainer.py
  - setup_fsdp_envs / setup_parallelism_envs / prepare_optim_env。

src/axolotl/utils/config/__init__.py
  - normalize_config：计算 world_size、device_map、effective batch_size。
```

## 1.3 主流程拆解

从用户入口开始，训练会先经历两段启动：外层 CLI 负责 launcher，内层 `axolotl.cli.train` 才真正加载配置和训练。

```text
User: axolotl train config.yaml --launcher accelerate
  -> src/axolotl/cli/main.py:train(config, launcher, kwargs)
     -> launch_training(...)
        -> accelerate launch -m axolotl.cli.train config.yaml
           -> src/axolotl/cli/train.py:do_cli(config)
              -> load_cfg(config)
              -> do_train(cfg, cli_args)
                 -> axolotl.train.train(cfg, dataset_meta)
```

源码上，外层 click 入口位于 `src/axolotl/cli/main.py:98-128`，它读取 config path 和 launcher，然后调用 `launch_training`。`launch_training` 根据 launcher 分发到 `_launch_accelerate_training` 或 `_launch_torchrun_training`（`src/axolotl/cli/utils/train.py:109-128`）。Accelerate 路径拼出的真实命令是 `accelerate launch ... -m axolotl.cli.train`（`src/axolotl/cli/utils/train.py:157-185`），torchrun 路径则会自动补 rendezvous 参数并执行 `torchrun ... -m axolotl.cli.train`（`src/axolotl/cli/utils/train.py:195-218`）。

真正改变 ND 行为的第一处，不在 `train.py`，而在配置加载：

```text
src/axolotl/cli/config.py:load_cfg
  -> validate_config(cfg, capabilities, env_capabilities)
  -> prepare_debug_log(cfg)
  -> prepare_optim_env(cfg)
       -> setup_fsdp_envs(cfg)
       -> setup_parallelism_envs(cfg)
  -> normalize_config(cfg)
```

`load_cfg` 在 `src/axolotl/cli/config.py:230-333` 完成读取 YAML、应用 CLI overrides、准备 plugins、Pydantic 校验、写环境变量和归一化。这里的顺序很关键：先校验，再写环境变量，再算 batch。也就是说，后面的 Accelerator / Trainer 看到的是已经规范化后的字段和环境。

### 配置字段：用户能写什么

ND 相关字段集中在 `src/axolotl/utils/schemas/config.py:933-998`：

- `fsdp` 已被标记 deprecated，推荐 `fsdp_config`（`config.py:933-940`）。
- `fsdp_version` 和 `fsdp_final_state_dict_type` 位于 `config.py:941-950`。
- `dp_shard_size` / `dp_replicate_size` 位于 `config.py:959-968`。
- `sequence_parallel_degree` 已废弃，迁移到 `context_parallel_size`（`config.py:969-980`）。
- `heads_k_stride`、`ring_attn_func`、`tensor_parallel_size` 位于 `config.py:981-998`。

这里有一个细节：`tensor_parallel_size` 的 schema 描述仍写着 “Only supported with DeepSpeed AutoTP”（`config.py:993-997`），但模型加载主路径已经会把它传给 Transformers 原生 TP（后文会看 `ModelLoader._build_model`）。这就是文档 / schema 描述滞后的一个例子，最终以源码主路径为准。

### 校验：哪些组合会直接被拒绝

CP 的约束在 `check_context_parallel_size`：

- 没设置时默认变成 1（`validation.py:1502-1515`）。
- `context_parallel_size > 1` 必须启用 `flash_attention`（`validation.py:1516-1520`）。
- 如果 `sample_packing` 开启，`micro_batch_size` 必须是 1，这是 ring-flash-attn 的要求（`validation.py:1522-1526`）。
- 运行时会尝试导入 `ring_flash_attn`，失败则抛出安装提示（`validation.py:1528-1550`）。
- 默认 `ring_attn_func`：sample packing 用 `varlen_llama3`，否则用 `batch_ring`（`validation.py:1563-1579`）。

TP 的优化器约束很直接：`tensor_parallel_size > 1` 时拒绝 `paged_adamw_8bit`、`adamw_8bit`、`adamw_bnb_8bit`（`validation.py:1600-1608`）。FSDP2 相关约束包括：

- `fsdp_config` 内 `fsdp_` 前缀会被剥离并告警（`validation.py:1050-1070`）。
- `version` / `fsdp_version` 会被归一到顶层和 `fsdp_config` 内（`validation.py:1072-1085`）。
- FSDP2 + bitsandbytes 8bit optimizer 某些组合会报错（`validation.py:1102-1117`）。
- FSDP2 要求 torch version >= 2.7.0（`config.py:1720-1735`）。当前项目依赖已是 `torch>=2.9.1`、`transformers==5.5.4`、`accelerate==1.13.0`（`pyproject.toml:13-23`）。

### 环境变量：把配置交给 Accelerate

`prepare_optim_env` 是第一个把用户配置写入运行时环境的函数。FSDP 环境在 `setup_fsdp_envs` 写入：

```text
cfg.fsdp_config
  -> ACCELERATE_USE_FSDP=true
  -> FSDP_VERSION=2
  -> FSDP_CPU_RAM_EFFICIENT_LOADING=true/...
  -> FSDP_STATE_DICT_TYPE=...
  -> FSDP_AUTO_WRAP_POLICY=...
```

源码对应 `src/axolotl/utils/trainer.py:589-618`。ND 并行环境在 `setup_parallelism_envs` 写入：

```text
tensor_parallel_size > 1  -> PARALLELISM_CONFIG_TP_SIZE
context_parallel_size > 1 -> PARALLELISM_CONFIG_CP_SIZE + ACCELERATE_ALLOW_CP_STANDALONE=true
dp_shard_size > 1        -> PARALLELISM_CONFIG_DP_SHARD_SIZE
dp_replicate_size > 1    -> PARALLELISM_CONFIG_DP_REPLICATE_SIZE
任一维度启用            -> ACCELERATE_USE_PARALLELISM_CONFIG=true
```

这段在 `src/axolotl/utils/trainer.py:621-640`。注意：`context_parallel_size > 1` 还会调用 `patch_prepare_cp()`（`trainer.py:632-638`），后文会解释为什么 Axolotl 不直接使用 Accelerate 的 CP 输入切分。

### Batch size：CP/TP 不是数据并行维度

`normalize_config` 会读取 `WORLD_SIZE` 和 `LOCAL_RANK`（`src/axolotl/utils/config/__init__.py:112-124`）。当 world_size > 1 且启用 FSDP/DDP 时，它计算：

```python
effective_world_size = world_size // context_parallel_size // tensor_parallel_size
cfg.batch_size = cfg.batch_size * effective_world_size
```

源码在 `src/axolotl/utils/config/__init__.py:133-143`。这说明 CP 和 TP 被视作 non-data-parallel 维度：同一组 CP rank 处理同一条序列的不同片段，同一组 TP rank 处理同一层的不同分片；它们不应该把 global batch 乘上去。

测试也锁住了这个行为：

- CP：`WORLD_SIZE=4, context_parallel_size=2` 时 batch 从 32 变 16（`tests/test_context_parallel_batch_size.py:29-56`）。
- TP：`WORLD_SIZE=4, tensor_parallel_size=2` 时 batch 同样从 32 变 16（`tests/test_tensor_parallel_batch_size.py:28-55`）。

## 1.4 关键细节与误区澄清

> 容易误解一：`context_parallel_size` 只是换了一个名字的 `sequence_parallel_degree`。

不是。`sequence_parallel_degree` 已经是 deprecated 字段，只在未设置 `context_parallel_size` 时迁移过去，并打印 warning（`validation.py:1508-1514`）。主路径读的是 `context_parallel_size`，环境变量写的是 `PARALLELISM_CONFIG_CP_SIZE`。

> 容易误解二：`tensor_parallel_size` 只会走 DeepSpeed AutoTP。

源码不是这样。DeepSpeed 配置确实会在校验中写 `tensor_parallel.autotp_size`（`validation.py:1121-1148`），但非 DeepSpeed 主路径里，`ModelLoader._build_model` 会把 `tp_size`、`tp_plan="auto"`、`device_mesh` 传给 Transformers `from_pretrained`（`src/axolotl/loaders/model.py:749-755`）。所以当前 Axolotl 同时存在 DeepSpeed AutoTP 兼容路径和 Transformers 原生 TP 主路径。

> 容易误解三：全局 batch size 默认等于 `micro_batch_size * gradient_accumulation_steps * world_size`。

在 ND 下不是。Axolotl 会除掉 CP 和 TP 维度，只把数据并行维度乘进去（`utils/config/__init__.py:133-143`）。否则同一条数据被 CP/TP 协同处理时会被错误统计成多份 batch。

## 1.5 本章小结

> 💡 **小结**
>
> * ND Parallelism 的第一个行为改变发生在 `load_cfg -> prepare_optim_env -> normalize_config`，不是在模型 forward。
> * FSDP 和 ParallelismConfig 主要通过环境变量传给 Accelerate；TP 还会在模型加载时显式传 `tp_size/tp_plan/device_mesh`。
> * CP/TP 会缩小有效 data-parallel world size，因此 batch size 归一化是正确训练语义的一部分。

# 二、DeviceMesh 与 ProcessGroup：ND 拓扑不是一张扁平 world

## 2.1 设计哲学与核心问题

把 `world_size=8` 看成 8 个等价 rank，是 DDP 时代的简化。ND Parallelism 需要的是一个多维坐标系：某些 rank 一起做 TP，某些 rank 一起做 CP，某些 rank 一起做 FSDP shard，跨节点还可能做 replicate。

这层要解决的是“通信组边界”的问题：

- TP 的通信应该发生在 TP group 内。
- CP 的 ring attention 应该发生在 CP group 内。
- FSDP 的参数 all-gather / reduce-scatter 应该发生在 FSDP mesh 内。
- HSDP 要把 shard 维和 replicate 维区分出来。

如果拓扑错了，最坏的结果不是性能下降，而是不同 rank 拿到不同 batch、attention group 不一致、FSDP 在错误的 group 里 gather 参数，直接挂死或 silently wrong。

## 2.2 源码入口与关键对象

```text
src/axolotl/utils/distributed.py
  - build_parallelism_config：根据 cfg 构造 Accelerate ParallelismConfig 和 DeviceMesh。
  - _get_parallel_config_kwargs：从 world_size 与各 size 推导 tp/cp/dp_shard/dp_replicate。

/usr/local/lib/python3.12/dist-packages/accelerate/parallelism_config.py
  - ParallelismConfig.__post_init__：从 env 或参数读取各维度 size，并做组合合法性校验。
  - build_device_mesh：按维度构建 PyTorch DeviceMesh，并 flatten dp、dp_shard_cp、dp_cp。

src/axolotl/monkeypatch/accelerate/parallelism_config.py
  - patch_parallelism_config：放宽 Axolotl 需要的 ParallelismConfig 校验。
  - patch_prepare_cp：把 Accelerate 的 CP prepare 替换成 no-op。

src/axolotl/monkeypatch/ring_attn/patch.py
  - register_ring_attn_from_device_mesh：从 DeviceMesh 中取 `cp` 维 process group。
```

## 2.3 主流程拆解

Axolotl 自己不会手写 `dist.new_group([...])`。它先推导 `ParallelismConfig` 参数，再让 Accelerate / PyTorch 建 mesh：

```text
build_parallelism_config(cfg)
  -> _get_parallel_config_kwargs(world_size, tp, cp, dp_shard, dp_replicate, is_fsdp)
  -> ParallelismConfig(**pc_kwargs)
  -> parallelism_config.build_device_mesh("cuda")
  -> return parallelism_config, device_mesh
```

源码在 `src/axolotl/utils/distributed.py:299-316`。

### `_get_parallel_config_kwargs` 的推导规则

核心逻辑在 `src/axolotl/utils/distributed.py:319-370`。它从 `remaining_world_size = world_size` 开始：

1. 先消耗 TP：`tp_size = tensor_parallel_size`，`remaining //= tp`（`distributed.py:330-332`）。
2. 再消耗 CP：`cp_size = context_parallel_size`，`remaining //= cp`（`distributed.py:334-336`）。
3. 如果用户没有显式 `dp_shard_size` 且没有 replicate，就把剩余 world size 当作 `dp_shard_size`（`distributed.py:338-342`）。
4. 如果有 `dp_replicate_size`，先消耗 replicate（`distributed.py:343-345`）。
5. 如果剩余 world size 仍大于 1 且用户配置了 `dp_shard_size`，必须有 FSDP，否则报错（`distributed.py:347-354`）。
6. 若最后还有剩余且是 FSDP，则补成 `dp_shard_size`（`distributed.py:359-362`）；否则报 “parallelisms incompatible”（`distributed.py:364-368`）。

测试覆盖了典型组合：`world_size=16,tp=2,cp=2,dp_shard=2,dp_replicate=2` 推导为 `(tp=2, cp=2, dp_shard=2, dp_replicate=2)`；FSDP 默认会把剩余 16 当作 `dp_shard_size`（`tests/test_loaders.py:181-218`）。

### Accelerate 如何把维度变成 mesh

Accelerate 的 `ParallelismConfig` 会从环境变量读取默认 size（`accelerate/parallelism_config.py:274-289`），当 `tp_size > 1` 或 `cp_size > 1` 时创建对应 handler（`parallelism_config.py:291-307`），并拒绝 “纯 DDP + TP/CP” 的组合（`parallelism_config.py:336-341`）。

构建 mesh 时，它按固定顺序排列维度：

```text
mesh_order = ["dp_replicate", "dp_shard", "cp", "sp", "tp"]
```

源码在 `accelerate/parallelism_config.py:260-272`。`build_device_mesh` 还会创建几个 flatten mesh：

- `dp`：所有 data-parallel 维。
- `dp_shard_cp`：把 `dp_shard` 和 `cp` 合并。
- `dp_cp`：把 `dp_replicate`、`dp_shard`、`cp` 合并，用于 loss 平均语义。

对应 `accelerate/parallelism_config.py:211-244`。

这解释了一个非常关键的现象：在 FSDP2 prepare 中，Axolotl 后续取的是 `mesh[tuple(accelerator.state.parallelism_config.fsdp_dim_names)]`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:351-360`），而 Accelerate 的 `fsdp_dim_names` 包含 `dp_replicate`（如果启用）和 `dp_shard_cp`（`accelerate/parallelism_config.py:157-164`）。也就是说，在 CP + FSDP2 下，CP 维会被折进 FSDP 使用的 shard mesh 中。

### 一个 8 卡例子

以 `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml` 为例：

```yaml
dp_shard_size: 2
context_parallel_size: 2
tensor_parallel_size: 2
fsdp_version: 2
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  reshard_after_forward: true
```

源码例子位于 `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:1-19`。world_size=8 时，推导出的 mesh 可以理解为：

```text
mesh shape: [dp_shard=2, cp=2, tp=2]

rank -> (dp_shard, cp, tp)
0 -> (0,0,0)   1 -> (0,0,1)
2 -> (0,1,0)   3 -> (0,1,1)
4 -> (1,0,0)   5 -> (1,0,1)
6 -> (1,1,0)   7 -> (1,1,1)
```

基于 Accelerate 的 mesh order，这是一个源码行为推断：

- TP group 大致是固定 `(dp_shard, cp)`，沿 `tp` 维通信，例如 `(0,1)`、`(2,3)`。
- CP group 大致是固定 `(dp_shard, tp)`，沿 `cp` 维通信，例如 `(0,2)`、`(1,3)`。
- FSDP group 使用 flatten 后的 `dp_shard_cp`，固定 `tp`，例如 `(0,2,4,6)` 与 `(1,3,5,7)`。

实际 rank group 以 PyTorch `DeviceMesh` 为准；Axolotl 在 ring attention 注册时会打印当前 CP group ranks（`src/axolotl/monkeypatch/ring_attn/patch.py:178-181`）。

### Axolotl 为什么 patch Accelerate 的 ParallelismConfig

`src/axolotl/monkeypatch/accelerate/parallelism_config.py` 做了两件事：

1. `_validate_accelerator` 允许 “pure CP standalone” 场景：当 `cp_size > 1`、`dp_shard_size <= 1` 且环境变量 `ACCELERATE_ALLOW_CP_STANDALONE=true` 时放行（`parallelism_config.py:28-45`）。
2. `patched_prepare_cp` 把 `Accelerator._prepare_cp` 改成只设置一个 no-op context 并返回 args（`parallelism_config.py:80-98`）。

这意味着 Axolotl 不完全依赖 Accelerate 的 CP 数据切分。Accelerate 负责建立 `ParallelismConfig` / `DeviceMesh`，Axolotl 自己在 `SequenceParallelContextManager` 里做 forward 前切分。

## 2.4 关键细节与误区澄清

> 容易误解一：`dp_shard_size: 8` 和 `context_parallel_size: 8` 在 8 卡上一定会让两个维度相乘成 64。

不会。Axolotl 的 `_get_parallel_config_kwargs` 是按 `remaining_world_size` 逐步消耗的。如果 world_size=8 且先消耗 `cp_size=8`，剩余为 1，显式 `dp_shard_size=8` 不会再进入 “remaining_world_size > 1” 的 dp_shard 分支（`distributed.py:334-354`）。不过 Accelerate 会把 CP 维 flatten 成 `dp_shard_cp`，FSDP2 prepare 仍可在这个 flatten mesh 上工作。这是一个源码层面的细节，和文档 `docs/nd_parallelism.qmd:88-90` 的表达需要结合源码理解。

> 容易误解二：CP group 是 Axolotl 手工 new_group 出来的。

不是。Axolotl 从 `trainer.accelerator.torch_device_mesh` 里取 `device_mesh[("cp",)]`，再 `get_group()` 得到 process group（`src/axolotl/monkeypatch/ring_attn/patch.py:159-184`）。它依赖的是 Accelerate / PyTorch DeviceMesh 的分组结果。

> 容易误解三：有 `ACCELERATE_USE_PARALLELISM_CONFIG=true` 就一定使用了 Axolotl 自己构造的 `parallelism_config` 对象。

不是。Axolotl 在 `ModelLoader._set_parallel_config` 里也会构造一个 `ParallelismConfig`（`src/axolotl/loaders/model.py:437-443`），主要用于拿 `device_mesh` 传给 Transformers TP。但 Trainer 内部的 `Accelerator` 通常是通过环境变量自己构造 `ParallelismConfig()`：Accelerate 在 `ACCELERATE_USE_PARALLELISM_CONFIG=true` 时创建对象（`accelerate/accelerator.py:453-459`）。这是两个同源但不同生命周期的对象。

## 2.5 本章小结

> 💡 **小结**
>
> * ND Parallelism 的核心状态是 `ParallelismConfig + DeviceMesh`，而不是一组散落的 `dist.new_group`。
> * Axolotl 负责从 YAML 推导各维 size；Accelerate 负责 mesh shape、flatten group 和合法性校验。
> * CP 维既是 ring attention group 的来源，也会在 FSDP2 下被 flatten 到 `dp_shard_cp`，这是 CP + FSDP 能组合的关键。

# 三、模型初始化与 TP/FSDP2：谁消费拓扑，谁只传环境变量

## 3.1 设计哲学与核心问题

配置和 mesh 只解决了“rank 怎么分组”。模型初始化还要解决另一个问题：不同并行维度到底在哪个时刻改变模型结构或参数状态？

- TP 必须在 `from_pretrained` 时告诉 Transformers：这个模型要按 `tp_plan="auto"` 和 `device_mesh` 加载。
- FSDP2 必须在 Trainer/Accelerate prepare 阶段把模块 `fully_shard`，并处理 CPU/meta loading。
- QLoRA + FSDP2 又引入 bitsandbytes `Params4bit`，它不能像普通 tensor 那样随意搬到 meta 或 DTensor。

因此 Axolotl 的模型初始化层不是单纯 `AutoModelForCausalLM.from_pretrained`，而是一条带 patch、device_map、quantization、TP kwargs、FSDP 特殊路径的加载流水线。

## 3.2 源码入口与关键对象

```text
src/axolotl/train.py
  - setup_model_and_tokenizer：创建 ModelLoader 并调用 load。

src/axolotl/loaders/model.py
  - ModelLoader.load：完整模型加载流水线。
  - _apply_pre_model_load_setup：决定是否使用 parallel config。
  - _set_parallel_config：构建 parallelism_config/device_mesh。
  - _set_device_map_config：FSDP/QLoRA 下 device_map 逻辑。
  - _build_model：TP kwargs、FSDP cpu/meta loading、sharded quant loading。

src/axolotl/loaders/patch_manager.py
  - _apply_fsdp_patches：FSDP2、ParallelismConfig、TRL FSDP prepare patch。

src/axolotl/monkeypatch/accelerate/fsdp2.py
  - fsdp2_prepare_model：Axolotl 替换后的 FSDP2 prepare。
  - fsdp2_load_full_state_dict：rank0 full state_dict 广播到 sharded model。
  - patch_peft_param_wrapper_for_fsdp2：PEFT LoRA 参数 wrapper 的 DTensor 兼容。

src/axolotl/utils/model_shard_quant.py
  - load_sharded_model_quant：QLoRA + FSDP cpu_ram_efficient_loading 的自定义量化加载。
```

## 3.3 主流程拆解

模型加载从 `setup_model_and_tokenizer` 进入：

```text
setup_model_and_tokenizer(cfg)
  -> tokenizer = load_tokenizer(cfg)
  -> model_loader = ModelLoader(cfg, tokenizer, processor)
  -> model, peft_config = model_loader.load()
```

源码在 `src/axolotl/train.py:54-84`。`ModelLoader.load` 的主流程是：

```text
ModelLoader.load
  -> patch_manager.apply_pre_model_load_patches()
  -> _apply_pre_model_load_setup()
  -> PLUGIN_MANAGER.pre_model_load()
  -> patch_manager.apply_post_plugin_pre_model_load_patches()
  -> _build_model()
  -> patch_manager.apply_post_model_build_patches(model)
  -> _apply_post_model_load_setup()
  -> _load_adapters()
  -> _apply_post_lora_load_setup(skip_move_to_device)
```

源码在 `src/axolotl/loaders/model.py:161-194`。

### TP：在 `from_pretrained` 前注入 `tp_size/tp_plan/device_mesh`

`ModelLoader._apply_pre_model_load_setup` 会判断是否需要 parallel config：只要启用了 FSDP config、TP 或 CP，就会进入 `_set_parallel_config`；但如果是 FSDP1，会关闭这条 parallel config 路径（`src/axolotl/loaders/model.py:196-212`）。

`_set_parallel_config` 调用前一章的 `build_parallelism_config` 并保存 `self.device_mesh`（`model.py:437-443`）。到 `_build_model` 时，TP 真正影响模型加载：

```text
if cfg.tensor_parallel_size > 1:
  model_kwargs["tp_size"] = cfg.tensor_parallel_size
  model_kwargs["tp_plan"] = "auto"
  model_kwargs["device_mesh"] = self.device_mesh
  del model_kwargs["device_map"]  # tp_plan 不兼容 device_map
```

对应 `src/axolotl/loaders/model.py:745-755`。模型加载后还有一个 workaround：如果 Transformers 4.54.0 没设置 `_tp_size` / `_device_mesh`，Axolotl 手动补上（`model.py:852-857`）。虽然当前项目 pin 到 Transformers 5.5.4（`pyproject.toml:20`），这个兼容代码仍存在。

TP 还有一个模型配置限制：如果模型 `tie_word_embeddings=True`，Axolotl 会直接拒绝 TP（`src/axolotl/loaders/utils.py:139-148`）。e2e TP 测试也因此被 skip：`tests/e2e/multigpu/test_tp.py:17-21` 标注 “TP doesn't work with models with tied weights”。

### FSDP2：不是加载时 shard，而是 prepare 阶段 fully_shard

FSDP 配置首先通过 env 和 TrainingArguments 交给 Accelerate。`setup_fsdp_envs` 写 `ACCELERATE_USE_FSDP`、`FSDP_VERSION`、`FSDP_STATE_DICT_TYPE` 等（`src/axolotl/utils/trainer.py:589-618`）；builder 会把 `fsdp_config` 和 `fsdp` 放进 training args（`src/axolotl/core/builders/base.py:604-607`）。

但 Axolotl 对 FSDP2 做了 patch：

```text
PatchManager._apply_fsdp_patches
  -> patch_initialize_missing_keys_for_fsdp()
  -> patch_parallelism_config()
  -> patch_accelerate_fsdp2()
  -> if cpu_ram_efficient_loading: patch_tied_keys_for_meta_device()
  -> if rl: patch_trl_prepare_fsdp2()
```

源码在 `src/axolotl/loaders/patch_manager.py:270-299`。

`patch_accelerate_fsdp2` 把 `accelerate.accelerator.fsdp2_prepare_model` 替换成 Axolotl 自己的 `fsdp2_prepare_model`，同时替换 `Accelerator.get_state_dict`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:529-538`）。

`fsdp2_prepare_model` 做的事情比较重：

1. 取 `accelerator.state.fsdp_plugin` 和 `original_sd = model.state_dict()`（`fsdp2.py:279-310`）。
2. 根据 FSDP plugin 设置 auto wrap policy，并在需要时先做 activation checkpointing（`fsdp2.py:316-342`）。
3. 从 `accelerator.state.device_mesh` 里取 FSDP mesh：`mesh[tuple(parallelism_config.fsdp_dim_names)]`（`fsdp2.py:344-360`）。
4. 处理 CPU offload pin_memory（`fsdp2.py:346-349`）。
5. 构造 `fsdp2_kwargs`：`reshard_after_forward`、`offload_policy`、mixed precision policy、mesh（`fsdp2.py:351-360`）。
6. 对匹配 auto wrap policy 的子模块 `fully_shard`，最后对整个 model `fully_shard`（`fsdp2.py:403-415`）。
7. 如果开启 `cpu_ram_efficient_loading`，再通过 `fsdp2_load_full_state_dict` 从 rank0 广播权重到 sharded model（`fsdp2.py:422-425`）。

这说明 FSDP2 的 sharding 不是发生在 `from_pretrained` 内部，而是在 Trainer/Accelerate prepare 模型阶段通过 patched `fully_shard` 发生。

### CPU/meta loading：省 CPU 内存，但引入广播和 patch

`ModelLoader._build_model` 在 FSDP 下会改变 device placement：

- `cpu_ram_efficient_loading=True` 时 `skip_move_to_device=True`，非 QLoRA 情况删除 `device_map`（`model.py:756-765`）。
- 如果是 FSDP2 + CPU RAM efficient + 非 TP，rank0 用 `device_map="cpu"`，其他 rank 用 `device_map="meta"`（`model.py:769-780`）。

随后 `fsdp2_prepare_model` 在 cpu_ram_efficient_loading 且不是 Params4bit 时，把模型移到 meta，避免 `fully_shard` 先搬到 GPU 导致峰值显存翻倍（`src/axolotl/monkeypatch/accelerate/fsdp2.py:371-388`）。真实权重再通过 `fsdp2_load_full_state_dict` 分发：

```text
for each param in model.state_dict():
  rank0: full_tensor = full_sd[param]
  if DTensor shard:
    distribute_tensor(full_tensor, device_mesh, placements, src_data_rank=0)
  else:
    dist.broadcast(sharded_param, src=0)
```

源码在 `fsdp2.py:20-97`。这条路径的收益是非 rank0 不需要完整 CPU 权重；代价是初始化时按参数广播 / distribute，且 rank0 仍要短暂持有 full state dict。

### QLoRA + FSDP2：Params4bit 是特殊对象

QLoRA 的 4bit 参数不是普通 Tensor。`load_sharded_model_quant` 会先在 `init_empty_weights` 下创建模型、替换 Linear 为 `Linear4bit`，然后逐个 safetensors shard 加载并量化（`src/axolotl/utils/model_shard_quant.py:167-227`）。加载时，rank0 可把量化参数放 CPU，非 rank0 放 meta（`model_shard_quant.py:243-264`）。

源码注释解释了为什么需要这么绕：FSDP 只同步 parameters 和 buffers，不同步 bitsandbytes 的 `quant_state`，所以 meta `Params4bit` 需要先量化初始化 quant_state，再把 data 换成 meta/CPU 以释放内存（`model_shard_quant.py:103-117`）。

FSDP2 prepare 里也有 Params4bit 检测：如果模型里有 Params4bit，就不走“把模型移到 meta 再 shard”的优化，因为 torch 操作不会保留原始参数类，移动到 meta 会破坏类型（`fsdp2.py:362-370`）。这意味着 QLoRA 路径可能保留更高的 VRAM 峰值。

## 3.4 关键细节与误区澄清

> 容易误解一：`ModelLoader._set_parallel_config()` 创建的 `parallelism_config` 就是 Trainer 内部用的对象。

不是。`ModelLoader` 保存的 `device_mesh` 主要用于模型加载阶段，例如 Transformers TP 的 `device_mesh`。Trainer 内部 Accelerator 的 `ParallelismConfig` 往往是由环境变量 `ACCELERATE_USE_PARALLELISM_CONFIG` 和 `PARALLELISM_CONFIG_*` 在 `Accelerator` 初始化时重建的（Accelerate 逻辑见 `accelerate/accelerator.py:453-459`）。

> 容易误解二：FSDP2 的参数分片发生在 `from_pretrained`。

不是。`from_pretrained` 只是加载模型；真正的 FSDP2 sharding 在 Accelerate prepare 阶段调用 patched `fsdp2_prepare_model`，通过 `fully_shard` 包装子模块和整个模型（`fsdp2.py:403-415`）。

> 容易误解三：`cpu_ram_efficient_loading` 对所有模型都同样省显存。

源码显示它对 Params4bit 有特殊限制：检测到 `Params4bit` 后不会把模型整体移到 meta（`fsdp2.py:362-371`）。因此 QLoRA + FSDP2 的 CPU RAM 省内存逻辑和普通 BF16/FP16 FSDP2 不完全相同。

## 3.5 本章小结

> 💡 **小结**
>
> * TP 是模型加载时的行为：Axolotl 显式传 `tp_size/tp_plan/device_mesh` 给 Transformers。
> * FSDP2 是 prepare 时的行为：Axolotl patch Accelerate 的 `fsdp2_prepare_model`，再调用 PyTorch `fully_shard`。
> * CPU/meta loading 解决初始化内存峰值，但把复杂度转移到 state_dict 广播、meta tensor tied weights 和量化参数兼容上。

# 四、Context Parallelism 主流程：切序列不是 dataloader 做的，而是 forward hook 做的

## 4.1 设计哲学与核心问题

CP 的目标是把 `[batch, seq]` 的 sequence 维拆到多个 GPU 上，让每张卡只持有 `seq / cp_size` 的激活和 logits。但这会立刻破坏三个东西：

1. attention 需要全局 KV，不能只看局部 chunk；
2. loss 的 token 数统计不能只看本 rank；
3. 某些 RL 路径需要 gather 输出，否则后续 reward / logprob 维度不对。

Axolotl 没有把切分放进 data collator，而是把它做成训练期间的 forward pre-hook / post-hook。这是一个重要设计：dataloader 仍产出完整 batch，同一 CP group 的 rank 拿到同一份输入，然后在模型 forward 前切成各自片段。

## 4.2 源码入口与关键对象

```text
src/axolotl/train.py
  - execute_training：当 context_parallel_size > 1 时进入 SequenceParallelContextManager。

src/axolotl/utils/ctx_managers/sequence_parallel.py
  - SequenceParallelContextManager：注册 / 移除 forward hooks。
  - apply_sequence_parallelism：创建 position_ids、padding、按 sequence 维 chunk、修正 num_items_in_batch。
  - AllGatherWithGrad：可选输出 gather，并在 backward 中切回本 rank 梯度。
```

## 4.3 主流程拆解

CP 主路径从 `execute_training` 开始：

```text
execute_training(cfg, trainer, resume)
  -> if cfg.context_parallel_size > 1:
       models = [trainer.model] (+ trainer.ref_model if exists)
       enter SequenceParallelContextManager(
         models,
         context_parallel_size,
         gradient_accumulation_steps,
         ring_attn_func,
         heads_k_stride,
         gather_outputs = cfg.rl in {GRPO, EBFT},
         device_mesh = trainer.accelerator.torch_device_mesh,
       )
  -> trainer.train(...)
```

源码在 `src/axolotl/train.py:183-229`。注意 `gather_outputs` 只在 GRPO / EBFT 下为 true（`train.py:210-219`）。普通 SFT 训练不会 gather logits；这正是 CP 节省 logits / activation 显存的关键之一。

### 进入 context：注册 ring attention 和 hooks

`SequenceParallelContextManager.__init__` 会先注册 ring attention group，然后取当前 rank 在 CP group 内的 local rank / world size：

```text
_register_ring_attn()
  -> register_ring_attn_from_device_mesh(device_mesh, context_parallel_dim=("cp",), ...)
process_group = get_ring_attn_group()
local_rank = dist.get_rank(process_group)
local_world_size = dist.get_world_size(process_group)
```

对应 `src/axolotl/utils/ctx_managers/sequence_parallel.py:189-231`。进入 `with` 时注册 forward pre-hook / post-hook；退出时只移除 hook handle（`sequence_parallel.py:233-244`）。

### pre-hook：完整 batch 到本 rank chunk

`apply_sequence_parallelism` 的输入是模型 forward 的 kwargs，最核心的 shape 变化是：

```text
原始 batch:
  input_ids:      [B, S]
  attention_mask: [B, S]
  labels:         [B, S]

如果没有 position_ids:
  position_ids:   [B, S]

如果 S 不能被 divisor 整除:
  input_ids/attention_mask/position_ids: pad 0 到 [B, S']
  labels: pad -100 到 [B, S']

after chunk on dim=1:
  rank_i input_ids: [B, S' / cp_size]
  rank_i labels:    [B, S' / cp_size]
```

源码细节如下：

- 输入 batch shape 由 `batch["input_ids"].shape` 得到（`sequence_parallel.py:51`）。
- 如果 batch 有 `position_ids` 且 batch_size=1，则调用 `update_ring_attn_params`，用于 sample packing 的 cu_seqlens（`sequence_parallel.py:53-56`）。否则创建标准 position_ids（`sequence_parallel.py:57-64`）。
- 如果有整数 `logits_to_keep`，会转换成本 rank chunk 上的 boolean mask（`sequence_parallel.py:65-95`）。
- sequence 长度 pad 到 `min(local_world_size, 64)` 的倍数（`sequence_parallel.py:96-134`）。通常 cp_size <=64 时就是 cp_size。
- 对所有 sequence 维等于 total_seq_len 的 tensor，按 dim=1 chunk 并取 local_rank（`sequence_parallel.py:135-148`）。
- 如果存在 `num_items_in_batch`，统计本地有效 label token，再在 CP group 内 all_reduce AVG，最后乘上 gradient_accumulation_steps（`sequence_parallel.py:150-165`）。

这里的 all_reduce AVG 很微妙。注释明确写着：SUM 会过度计数并把 loss scale down，所以使用 AVG（`sequence_parallel.py:156-160`）。这说明 loss token 计数并不是简单“所有 CP rank 求和”。

### post-hook：普通 SFT 不 gather，GRPO/EBFT 才 gather 输出

`SequenceParallelContextManager` 注册两个 post hooks：

1. `sequence_parallel_post_hook`：当 `gather_outputs=True` 时，对 output tensor 做 all-gather，并去掉 padding（`sequence_parallel.py:290-303`）。
2. `eval_loss_correction_post_hook`：始终注册，用本地有效 token 加权 all-reduce 修正 eval loss（`sequence_parallel.py:305-341`）。

输出 gather 用 `AllGatherWithGrad`：forward 里先 all_gather shape，再 all_gather tensor，最后沿 sequence 维 concat（`sequence_parallel.py:368-416`）。backward 不做 reduce-scatter，而是直接从 full gradient 中切出本 rank 对应 sequence slice（`sequence_parallel.py:418-444`）。

这意味着 `AllGatherWithGrad` 的通信语义是：

```text
forward:  local output [B, S_i, ...] --all_gather--> [B, sum(S_i), ...]
backward: full grad [B, S, ...] --slice only--> local grad [B, S_i, ...]
```

它适合“前向后续逻辑需要完整 sequence 输出”的场景；普通 SFT 不走这条输出 gather，避免把 logits 又拼回 full sequence。

### eval loss：为什么需要第二个 post-hook

训练时，模型 forward 可能通过 `num_items_in_batch` 缩放 loss；eval 时，Axolotl 在 pre-hook 中如果发现模型不在 training，会记录本地有效 token，并移除 `num_items_in_batch`，让模型使用 reduction mean（`sequence_parallel.py:276-285`）。post-hook 再把 `loss * local_valid_tokens` 和 `local_valid_tokens` 在 CP group 内 SUM，计算全局加权 mean（`sequence_parallel.py:312-335`）。

这个设计避免了某些 rank 没有有效 label token 导致 NaN 的问题：如果 local_valid 为 0，weighted_loss 被置 0（`sequence_parallel.py:315-318`）。

## 4.4 关键细节与误区澄清

> 容易误解一：文档说 “data collator handles chunking”，所以 CP 切分发生在 collator。

当前源码主路径不是这样。`docs/sequence_parallelism.qmd:40-45` 的描述更像概念说明，但真正切分发生在 `SequenceParallelContextManager` 的 forward pre-hook 中（`sequence_parallel.py:255-288`），进入点是 `train.py:205-220`。

> 容易误解二：CP 总是会 gather logits，因此 logits 显存不会省。

不对。`gather_outputs=cfg.rl in {GRPO, EBFT}`（`train.py:217`），普通 SFT 不 gather 输出。SFT 的 logits 只在本 rank 的 sequence chunk 上存在，显存收益保留。GRPO/EBFT 因后续算法需要完整输出，才会用 `AllGatherWithGrad` 拼回。

> 容易误解三：`ring_attn_func` 在 `apply_sequence_parallelism` 里没用，所以配置无效。

它对“batch slicing”确实没用，函数 docstring 也写了 currently unused（`sequence_parallel.py:37-43`）。但它在 `_register_ring_attn` 中传给 `register_ring_attn_from_device_mesh`，决定是替换成 `varlen_llama3` 还是 `batch_ring` attention（`sequence_parallel.py:246-253`）。所以它不是无效字段，只是不参与切片算法。

## 4.5 本章小结

> 💡 **小结**
>
> * Axolotl 的 CP 主路径是 forward hook：dataloader 给完整 batch，pre-hook 按 sequence 维切分。
> * 普通 SFT 不 gather 输出，因此 logits / activation 的 sequence 维显存收益能保留。
> * Eval loss 和 `num_items_in_batch` 都有 CP group 内修正逻辑，否则 token 计数会错。

# 五、Ring Flash Attention 注入：让局部 sequence chunk 看见全局 KV

## 5.1 设计哲学与核心问题

CP 把 sequence 切开之后，MLP 和 logits 这类逐 token 计算天然能局部执行；attention 不行。一个 token 的 causal attention 需要看到它之前的全局 KV。如果每张卡只保留本地 KV，结果就错了。

Axolotl 的方案是把 HuggingFace Transformers 的 flash attention 调用替换成 ring-flash-attn 实现。这样模型 forward 代码本身不用改，attention kernel 内部通过 CP process group 做 ring 通信，让每个 rank 为自己的 query chunk 逐步看到其他 rank 的 KV chunk。

这层解决的是通信问题和侵入性问题：用 monkey patch 降低对模型源码的侵入，但代价是强依赖 Transformers / ring-flash-attn 的函数签名。

## 5.2 源码入口与关键对象

```text
src/axolotl/monkeypatch/ring_attn/patch.py
  - RING_ATTN_GROUP：模块级全局 CP group。
  - register_ring_attn_from_device_mesh：从 DeviceMesh 取 cp group 并替换 flash attention。
  - create_ring_flash_attention_forward：varlen_llama3 适配 HF flash attention 签名。
  - update_ring_attn_params：根据 position_ids 计算 cu_seqlens 并传给 ring_flash_attn。

src/axolotl/monkeypatch/ring_attn/adapters/batch.py
  - create_flash_attn_forward_varlen_llama3：batch_ring 的 HF flash attention adapter。
  - substitute_hf_flash_attn：替换 transformers.modeling_flash_attention_utils._flash_attention_forward。

src/axolotl/utils/schemas/enums.py
  - RingAttnFunc：当前只启用 varlen_llama3 和 batch_ring。
```

## 5.3 主流程拆解

注册入口在 `SequenceParallelContextManager._register_ring_attn`：

```text
SequenceParallelContextManager._register_ring_attn
  -> register_ring_attn_from_device_mesh(
       device_mesh=trainer.accelerator.torch_device_mesh,
       context_parallel_dim=("cp",),
       heads_k_stride=cfg.heads_k_stride,
       ring_attn_func=cfg.ring_attn_func,
     )
```

`register_ring_attn_from_device_mesh` 做三步：

1. 从 `device_mesh[("cp",)]` 取 CP submesh（`src/axolotl/monkeypatch/ring_attn/patch.py:159-166`）。
2. `sequence_mesh.get_group()` 得到 process group，并写入模块级 `RING_ATTN_GROUP`（`patch.py:168-184`）。
3. 根据 `ring_attn_func` 替换 HF flash attention：
   - `VARLEN_LLAMA3`：替换 `ring_flash_attn.adapters.hf_adapter.create_ring_flash_attention_forward`，再调用上游 `substitute_hf_flash_attn`（`patch.py:186-202`）。
   - `BATCH_RING`：调用 Axolotl 自己的 `adapters.batch.substitute_hf_flash_attn`（`patch.py:203-211`）。

`RING_ATTN_GROUP` 是模块级全局变量，getter 在未注册时报错（`patch.py:34-47`）。这保证 CP 训练时任何需要 CP group 的函数都读到同一个 group，但也意味着 patch 状态不是 thread-local。

### varlen_llama3：sample packing 的 cu_seqlens 路径

当 `ring_attn_func=varlen_llama3` 时，Axolotl 创建的 `_flash_attention_forward_v3` 会调用：

```text
llama3_flash_attn_varlen_func(
  query_states.squeeze(0),
  key_states.squeeze(0),
  value_states.squeeze(0),
  cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
  cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
  max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
  local_k_slice=DATA_PARAMS["local_k_slice"],
  heads_k_stride=heads_k_stride,
  group=process_group,
)
```

源码在 `src/axolotl/monkeypatch/ring_attn/patch.py:50-132`。它有几个硬假设：

- `softcap is None`，否则 assert（`patch.py:99-102`）。
- 只支持 causal attention（`patch.py:105-107`）。
- batch_size 必须是 1，注释说 varlen data 应提前处理（`patch.py:107-109`）。

`update_ring_attn_params` 会根据 `position_ids` 计算 `cu_seqlens`，然后调用 `ring_flash_attn.update_ring_flash_attn_params(cu_seqlens, group)`（`patch.py:214-226`）。pre-hook 只有在 batch_size=1 且已有 position_ids 时会调用它（`sequence_parallel.py:53-56`），这正对应 sample packing 路径。

### batch_ring：非 sample packing 的 batch API 路径

`batch.py` 里，Axolotl 定义了一个和 Transformers `_flash_attention_forward` 签名兼容的函数，然后调用 `ring_flash_attn_func(..., group=process_group)`（`src/axolotl/monkeypatch/ring_attn/adapters/batch.py:61-151`）。替换逻辑会检查新旧函数参数是否匹配，匹配后覆盖：

```python
transformers.modeling_flash_attention_utils._flash_attention_forward = new_flash_attention_forward
ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
```

对应 `batch.py:156-196`。

当前 enum 只开放两个值：`varlen_llama3` 和 `batch_ring`；`batch_zigzag` / `batch_stripe` 在 enum 中是注释状态（`src/axolotl/utils/schemas/enums.py:100-108`）。`sequence_parallel.py` 顶部 TODO 也写着 zigzag / stripe 还没实现（`sequence_parallel.py:22-23`）。

## 5.4 关键细节与误区澄清

> 容易误解一：CP 的主要通信就是 `AllGatherWithGrad`。

不是。`AllGatherWithGrad` 只用于 forward 输出需要恢复完整 sequence 的路径，例如 GRPO/EBFT。attention 的关键通信发生在 ring-flash-attn kernel 内部，Axolotl 只是把 CP process group 传进去（`patch.py:110-124`、`batch.py:137-149`）。普通 SFT 即使不 gather logits，attention 层仍会在每层通过 ring attention 交换 KV。

> 容易误解二：ring attention patch 是局部 context manager，退出后会恢复。

不是完全恢复。`SequenceParallelContextManager.__exit__` 只移除模型 hooks；TODO 明确写着未来再 un-patch attention 和 accelerate functions（`sequence_parallel.py:238-245`）。`RING_ATTN_GROUP` 和被替换的 HF flash attention 函数会留在进程全局命名空间中。

> 容易误解三：文档里的所有 `ring_attn_func` 名称都可用。

源码 enum 只启用了 `varlen_llama3` 和 `batch_ring`（`enums.py:100-108`）。`batch_zigzag` / `batch_stripe` 目前是注释状态，不能当作稳定主路径。

## 5.5 本章小结

> 💡 **小结**
>
> * CP 能正确训练的关键不是简单切 input，而是 attention 层被替换成带 CP group 的 ring-flash-attn。
> * `varlen_llama3` 偏 sample packing，依赖 position_ids / cu_seqlens；`batch_ring` 偏普通 batch。
> * monkey patch 降低了模型适配成本，但带来全局状态污染和上游签名兼容风险。

# 六、FSDP 保存与 state_dict：训练结束不是简单 `save_pretrained`

## 6.1 设计哲学与核心问题

ND Parallelism 训练完成后，模型权重可能处在多种状态中：FSDP2 DTensor、FSDP1 wrapper、TP sharded module、PEFT adapter、CP eval 后被污染的 tensor storage、SHARDED_STATE_DICT checkpoint。最终用户需要的是一个能加载的输出目录。

这层要解决的是保存和恢复问题：

- FSDP2 FULL_STATE_DICT 要把 DTensor 汇成普通 tensor。
- SHARDED_STATE_DICT 需要提示或自动 merge。
- CP 场景下 safetensors 可能遇到无效 storage，需要 CPU clone。
- FSDP wrapper 会把 architecture 名字加前缀，需要清理。
- optimizer/scheduler 在部分 FSDP2 场景无法保存，不能让训练直接崩。

## 6.2 源码入口与关键对象

```text
src/axolotl/train.py
  - save_trained_model：训练结束后的统一保存入口。
  - _rename_fsdp_merged_to_adapter：FSDP sharded PEFT 权重 merge 后重命名。

src/axolotl/core/trainers/mixins/distributed_parallel.py
  - DistributedParallelMixin._save：dp_shard_enabled 时显式 get_state_dict。

src/axolotl/core/trainers/base.py
  - AxolotlTrainer._save：CP 下 clone tensor 到 CPU，避免 safetensors storage 问题。

src/axolotl/monkeypatch/accelerate/fsdp2.py
  - get_state_dict：替换 Accelerator.get_state_dict，处理 FSDP2 DTensor full_tensor。

src/axolotl/core/trainers/mixins/checkpoints.py
  - CheckpointSaveMixin._save_optimizer_and_scheduler：FSDP optimizer 保存失败时降级告警。
```

## 6.3 主流程拆解

训练结束后，`axolotl.train.train` 会调用：

```text
execute_training(...)
  -> trainer.train(...)
  -> save_trained_model(cfg, trainer, model)
  -> tokenizer.save_pretrained(...)
  -> create_model_card(...)
```

源码在 `src/axolotl/train.py:624-640`。

FSDP 分支在 `save_trained_model` 中：

1. 如果 trainer FSDP enabled 或 cfg.fsdp_config 存在，先设置最终 state_dict_type：优先 `final_state_dict_type`，否则 `state_dict_type`（`src/axolotl/train.py:294-300`）。
2. 调用 `trainer.save_model(cfg.output_dir)`（`train.py:301`）。
3. 如果是 `SHARDED_STATE_DICT`，打印 merge 提示，并在特定条件下调用 `merge_fsdp_weights` 自动 merge 到 output_dir（`train.py:302-333`）。
4. 主进程清理 `config.json` 里的 `FSDP` architecture 前缀（`train.py:334-349`）。

`DistributedParallelMixin._save` 在 `accelerator.parallelism_config.dp_shard_enabled` 时会显式调用 `self.accelerator.get_state_dict(self.model)`（`src/axolotl/core/trainers/mixins/distributed_parallel.py:14-21`）。而这个 `get_state_dict` 已被 Axolotl FSDP2 patch 替换。

### FSDP2 get_state_dict：full_tensor + barrier

Axolotl patched `Accelerator.get_state_dict` 中，FSDP2 分支会遍历 `model.state_dict()`：

```text
for param_name, param in sharded_state_dict.items():
  if param is CPU: move to cuda
  if isinstance(param, DTensor): param = param.full_tensor()
  if rank == 0: state_dict[param_name] = param.cpu()
  torch.distributed.barrier()
```

源码在 `src/axolotl/monkeypatch/accelerate/fsdp2.py:158-173`。这是一条简单可靠但可能慢的路径：每个参数都可能触发 full_tensor gather，并且每个参数后有 barrier。大模型保存时，rank0 聚合和 per-param barrier 都可能成为瓶颈。

### CP 保存：为什么要 clone 到 CPU

`AxolotlTrainer._save` 中有一个 CP 专用修复：如果 `state_dict` 非空且 `context_parallel_size > 1`，会对每个 tensor 执行 `v.detach().cpu()`（`src/axolotl/core/trainers/base.py:805-823`）。注释说明：CP eval 会让 tensor storage pointer 失效，因此 clone 到 CPU 以便 safetensors 获得新的有效 storage（`base.py:812-814`）。

这不是显存优化，而是保存正确性 / 兼容性修复。代价是保存时 CPU 内存峰值会上升。

### checkpoint optimizer：失败时告警而不是崩溃

`CheckpointSaveMixin._save_optimizer_and_scheduler` 捕获 `NotImplementedError` / `KeyError`，并 warning：optimizer 和 scheduler 未保存，无法 resume（`src/axolotl/core/trainers/mixins/checkpoints.py:10-22`）。注释里还有 TODO：fix fsdp2 optimizer saving（`checkpoints.py:17`）。

这说明“模型权重保存成功”不等价于“完整训练状态可 resume”。对于长时间多机训练，这个风险比普通 SFT 更大。

## 6.4 关键细节与误区澄清

> 容易误解一：FSDP2 保存就是 Accelerate 原生行为。

不是完全原生。Axolotl 替换了 `Accelerator.get_state_dict`（`fsdp2.py:529-538`），FSDP2 分支手动 `DTensor.full_tensor()` 并 rank0 CPU 聚合（`fsdp2.py:158-173`）。

> 容易误解二：`trainer.save_model` 后一定得到最终完整模型。

如果 `state_dict_type == SHARDED_STATE_DICT`，Axolotl 会提示用户需要 `merge-sharded-fsdp-weights`，并只在某些情况下自动 merge（`train.py:302-333`）。因此保存结果取决于 state_dict_type 和 checkpoint 目录是否存在。

> 容易误解三：Checkpoint 保存失败一定会中断训练。

optimizer/scheduler 保存失败会被 `CheckpointSaveMixin` 捕获并降级为 warning（`checkpoints.py:13-22`）。这保护了训练不中断，但牺牲了 resume 能力。

## 6.5 本章小结

> 💡 **小结**
>
> * FSDP/CP/TP 组合下，保存是单独的工程问题：权重状态、wrapper、storage、adapter 命名都可能影响可加载性。
> * FSDP2 FULL_STATE_DICT 走 rank0 聚合，可靠但有 rank0 CPU 内存和 per-param barrier 成本。
> * 当前源码明确存在 FSDP2 optimizer 保存 TODO，resume 语义不能和普通单卡训练等同看待。

# 七、完整主路径串联

## 7.1 完整调用栈

以 8 卡单机的示例配置为例：`examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml` 同时配置 `dp_shard_size: 2`、`context_parallel_size: 2`、`tensor_parallel_size: 2`、`fsdp_version: 2`（`examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:6-19`）。一次训练可以串成：

```text
User: axolotl train examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml
  │
  ├─ Step 1: CLI launcher
  │     └─ src/axolotl/cli/main.py:train
  │        -> accelerate launch -m axolotl.cli.train
  │
  ├─ Step 2: 配置读取 / 校验 / 环境变量
  │     └─ src/axolotl/cli/config.py:load_cfg
  │        -> validate_config
  │        -> prepare_optim_env
  │        -> normalize_config
  │
  ├─ Step 3: Patch 注册
  │     └─ src/axolotl/loaders/patch_manager.py:_apply_fsdp_patches
  │        -> patch_parallelism_config
  │        -> patch_accelerate_fsdp2
  │        -> patch_prepare_context_parallel_inputs
  │
  ├─ Step 4: 模型加载
  │     └─ src/axolotl/loaders/model.py:ModelLoader.load
  │        -> _set_parallel_config
  │        -> _build_model(tp_size/tp_plan/device_mesh)
  │
  ├─ Step 5: Trainer / Accelerator 创建
  │     └─ src/axolotl/utils/trainer.py:setup_trainer
  │        -> HFCausalTrainerBuilder / HFRLTrainerBuilder
  │        -> TrainingArguments(fsdp_config, fsdp)
  │        -> Accelerator reads PARALLELISM_CONFIG_* env
  │
  ├─ Step 6: FSDP2 prepare
  │     └─ patched accelerate.fsdp2_prepare_model
  │        -> fully_shard(child modules)
  │        -> fully_shard(model)
  │        -> optional full state_dict broadcast
  │
  ├─ Step 7: CP training context
  │     └─ src/axolotl/train.py:execute_training
  │        -> SequenceParallelContextManager
  │        -> register_ring_attn_from_device_mesh
  │        -> model forward pre/post hooks
  │
  ├─ Step 8: 每个 training step
  │     └─ Trainer.train
  │        -> dataloader gives full batch
  │        -> pre-hook slices [B,S] -> [B,S/cp]
  │        -> ring attention exchanges KV in cp group
  │        -> FSDP gathers/shards params in fsdp mesh
  │        -> TP executes model shards in tp group
  │
  └─ Step 9: 保存
        └─ src/axolotl/train.py:save_trained_model
           -> fsdp_plugin.set_state_dict_type
           -> trainer.save_model
           -> optional sharded merge
           -> config architecture prefix cleanup
```

## 7.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| CLI launcher | config path, launcher args | 多进程环境：`WORLD_SIZE/RANK/LOCAL_RANK` | launcher rendezvous | 无直接影响 | 启动一次 |
| `load_cfg` | YAML + CLI overrides | validated cfg、FSDP/Parallelism env、归一化 batch | 无 | batch 语义改变 | 每进程一次 |
| PatchManager | cfg | 替换 Accelerate / Transformers / PEFT 函数 | 无 | 为后续省内存/兼容铺路 | 每进程一次 |
| ModelLoader | cfg、tokenizer | model、peft_config、TP kwargs、device_map | 一般无；远端下载除外 | TP 加载 / CPU-meta loading 影响初始化峰值 | 每进程一次 |
| Accelerator / FSDP prepare | model、TrainingArguments、env | wrapped/sharded model、device_mesh | FSDP 初始化可能同步 / broadcast | 参数、梯度、optimizer state 分片 | 初始化一次 |
| CP context enter | trainer.model、device_mesh | CP group、ring attention patch、forward hooks | 无；仅取 group | 无直接影响 | 训练前一次 |
| 每个 forward | full batch | local sequence chunk、局部 logits/loss | ring attention；可选 output all_gather；loss all_reduce | 激活/logits 按 CP 降低；通信增加 | 每 step / 每 forward |
| 保存 | sharded/wrapped model | output_dir 权重、config、tokenizer | FSDP full_tensor / barrier / merge | rank0 CPU 内存峰值；保存时可能复制 | 保存时 |

## 7.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `docs/sequence_parallelism.qmd` 的 “data collator handles chunking” | 文档写法像数据层切分 | 否 | 当前源码主路径是在 forward pre-hook 中切分（`sequence_parallel.py:255-288`）。 |
| `Accelerator._prepare_cp` | 名字像 CP 主切分入口 | 被 patch 成 no-op | Axolotl 只让 Accelerate 建 topology，实际 CP slicing 自己做（`parallelism_config.py:80-98`）。 |
| `AllGatherWithGrad` | 名字像 CP attention 通信 | 只在 gather_outputs=True 时参与 | Attention 通信在 ring-flash-attn 内部；AllGather 主要服务 GRPO/EBFT 输出恢复。 |
| DeepSpeed AutoTP JSON 修改 | TP 配置校验里会写 `autotp_size` | 仅 DeepSpeed 路径 | Transformers TP 主路径走 `tp_size/tp_plan/device_mesh`（`model.py:749-755`）。 |
| `sequence_parallel_degree` | 老配置名看起来还能用 | 兼容迁移路径 | 主字段是 `context_parallel_size`，老字段只在未设置新字段时迁移（`validation.py:1508-1514`）。 |
| `CheckpointSaveMixin` | 名字像完整 checkpoint 支持 | 失败降级告警 | FSDP2 optimizer/scheduler 保存仍有 TODO，失败后不能 resume（`checkpoints.py:13-22`）。 |

> 💡 **小结**
>
> * 完整主路径可以理解为：配置 env 化 -> mesh 化 -> 模型 TP/FSDP 化 -> forward CP 化 -> 保存 state_dict 化。
> * CP 和 FSDP 都是运行时行为：CP 每个 forward 切数据，FSDP 每层 / 每参数按 shard 策略通信。
> * 文章中最容易误判的地方，是把名字相似的 “Accelerate CP prepare” 当成 Axolotl CP 主实现。

# 八、关键数据流、状态流与 shape 流程

## 8.1 Tensor shape 变化

以普通 causal SFT、`context_parallel_size=2` 为例：

```text
原始输入（dataloader 输出，每个 CP group 内 rank 拿同一份）:
  input_ids:      [B, S]
  attention_mask: [B, S]
  labels:         [B, S]

pre-hook 创建或更新 position_ids:
  position_ids:   [B, S]

padding（若 S % cp_size != 0）:
  input_ids:      [B, S']       pad value = 0
  attention_mask: [B, S']       pad value = 0
  labels:         [B, S']       pad value = -100

按 sequence 维切分:
  rank0 input_ids: [B, S'/2]
  rank1 input_ids: [B, S'/2]

模型本地输出（普通 SFT 不 gather）:
  logits_i: [B, S'/2, vocab]
  loss_i:   scalar

评估 loss 修正:
  weighted_loss_i = loss_i * valid_tokens_i
  all_reduce SUM(weighted_loss_i), all_reduce SUM(valid_tokens_i)
  eval_loss = global_weighted_loss / global_valid_tokens
```

显存节省主要发生在模型 forward 内部：attention / MLP 激活和 logits 的 sequence 维从 `S` 变成 `S/cp_size`。但是 attention 层为了保持全局语义，会通过 ring-flash-attn 在 CP group 内交换 KV；因此 CP 是“通信换显存”。

对于 GRPO/EBFT，后处理需要完整输出，post-hook 会把 local output all-gather 成：

```text
local output: [B, S_i, ...]
all_gather:  [B, S, ...]
backward:    grad_output[:, offset_i:offset_i+S_i] -> local grad
```

这一步会恢复完整 sequence tensor，因此 GRPO/EBFT 的部分显存收益会被后处理阶段抵消。

## 8.2 Rank / Mesh / Process Group 变化

以 world_size=8、`dp_shard_size=2`、`context_parallel_size=2`、`tensor_parallel_size=2` 为例，mesh 可以理解为：

```text
mesh dims = [dp_shard, cp, tp]
shape     = [2,        2,  2]

rank0: (dp0, cp0, tp0)
rank1: (dp0, cp0, tp1)
rank2: (dp0, cp1, tp0)
rank3: (dp0, cp1, tp1)
rank4: (dp1, cp0, tp0)
rank5: (dp1, cp0, tp1)
rank6: (dp1, cp1, tp0)
rank7: (dp1, cp1, tp1)
```

基于 Accelerate mesh order 的推断：

```text
TP group（固定 dp_shard, cp）:
  (rank0, rank1), (rank2, rank3), (rank4, rank5), (rank6, rank7)

CP group（固定 dp_shard, tp）:
  (rank0, rank2), (rank1, rank3), (rank4, rank6), (rank5, rank7)

FSDP mesh（flatten dp_shard + cp，固定 tp）:
  (rank0, rank2, rank4, rank6)
  (rank1, rank3, rank5, rank7)
```

TP 和 CP 都属于 non-data-parallel 维度；它们不增加不同样本数。FSDP mesh flatten 了 `dp_shard_cp`，所以 CP 与 FSDP 的组合并不是“两个完全独立的通信世界”，它们共享 DeviceMesh 维度，只是在不同操作中取不同 group。

多机场景中，示例 `examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml` 配置 `dp_shard_size: 4`、`dp_replicate_size: 2`、`tensor_parallel_size: 2`（`examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml:6-22`）。文档解释其意图是节点内 FSDP/TP，节点间 DDP replicate（`examples/distributed-parallel/README.md:34-45`）。源码层面实际分组仍由全局 rank 顺序和 launcher 环境决定；Axolotl 没有额外读取“节点拓扑”来重排 rank。

## 8.3 状态切换

ND Parallelism 中有几类状态：

```text
进程环境变量:
  写入者: setup_fsdp_envs / setup_parallelism_envs
  读取者: Accelerate Accelerator / ParallelismConfig / FSDP plugin
  生命周期: 进程级

DeviceMesh / ParallelismConfig:
  写入者: Accelerate 或 ModelLoader.build_parallelism_config
  读取者: ModelLoader TP、FSDP2 prepare、ring attention registration
  生命周期: Accelerator / ModelLoader 对象级

RING_ATTN_GROUP:
  写入者: register_ring_attn_from_device_mesh
  读取者: apply_sequence_parallelism、update_ring_attn_params、GRPO trainer
  生命周期: 模块全局，非 thread-local

Transformers / Accelerate monkey patch:
  写入者: PatchManager、setup_parallelism_envs、SequenceParallelContextManager ring registration
  读取者: 后续 Trainer / Accelerate / model forward
  生命周期: 进程全局，当前源码未自动恢复

Forward hooks:
  写入者: SequenceParallelContextManager.__enter__
  读取者: PyTorch module forward
  生命周期: with 块内；__exit__ 会 remove
```

最需要警惕的是全局 patch：hooks 会退出恢复，但 attention 替换和 Accelerate patch 没有恢复。测试里 `test_trainer_context_parallel_patch.py` 会手动 restore `Trainer._prepare_context_parallel_inputs`（`tests/monkeypatch/test_trainer_context_parallel_patch.py:13-34`），这本身也说明 patch 状态需要测试层面清理。

> 💡 **小结**
>
> * CP 的 shape 收益来自 forward 前把 `[B,S]` 切成 `[B,S/cp]`，普通 SFT 不会再 gather logits。
> * TP/CP/FSDP 不是三套孤立 group，而是同一个 DeviceMesh 上的不同切片和 flatten。
> * 环境变量、global group、monkey patch、forward hook 生命周期不同；排查 bug 时必须先判断状态污染在哪一层。

# 九、核心机制深挖

## 9.1 Monkey Patch：低侵入接入还是维护风险？

Axolotl 在 ND Parallelism 上使用了多处 monkey patch：

| Patch | 替换对象 | 触发条件 | 是否恢复 | 作用 |
|---|---|---|---|---|
| `patch_prepare_cp` | `Accelerator._prepare_cp` | `context_parallel_size > 1` | 否 | 让 Accelerate CP prepare 变 no-op，由 Axolotl hook 接管切分。 |
| `patch_parallelism_config` | `ParallelismConfig._validate_accelerator`、`AcceleratorState.is_fsdp2` | CP 或 FSDP2 | 否 | 放宽 pure CP、修复 fsdp_plugin None 判断。 |
| `patch_prepare_context_parallel_inputs` | `Trainer._prepare_context_parallel_inputs` | CP | 测试手动恢复；运行时不恢复 | 放宽 Transformers CP 对 SDPA 的限制，允许 flash_attention_2。 |
| ring attention patch | HF flash attention 函数 / `ALL_ATTENTION_FUNCTIONS` | 进入 CP context | 否 | 把 attention kernel 替换成 ring-flash-attn。 |
| `patch_accelerate_fsdp2` | `accelerate.fsdp2_prepare_model`、`Accelerator.get_state_dict` | FSDP2 | 否 | 自定义 FSDP2 prepare 和保存。 |
| `patch_peft_param_wrapper_for_fsdp2` | PEFT `_LoraParameterProxy.forward` | FSDP2 + ParamWrapper | 否 | DTensor / Tensor 相加兼容。 |

为什么不能更简单？因为这些能力分散在 Transformers、Accelerate、PyTorch、PEFT 和 ring-flash-attn 中。Axolotl 作为上层框架，既要跟上最新上游，又要支持已有 YAML 入口；直接 fork 下游库成本更高。

隐藏假设也很明显：

- Transformers 的 `_prepare_context_parallel_inputs` 源码里必须存在特定 guard 字符串，否则 patch 跳过（`src/axolotl/monkeypatch/transformers/trainer_context_parallel.py:15-38`）。
- batch ring adapter 会检查新旧 `_flash_attention_forward` 签名，签名不匹配就报 Transformers 版本不支持（`src/axolotl/monkeypatch/ring_attn/adapters/batch.py:167-190`）。
- FSDP2 patch 假设 Accelerate 会调用 `accelerate.accelerator.fsdp2_prepare_model`，并假设 `accelerator.state.parallelism_config.fsdp_dim_names` 可用（`fsdp2.py:351-360`）。

这类 patch 的收益是用户配置简单；代价是升级 Transformers/Accelerate/ring-flash-attn 时，最容易坏的正是这些函数签名和内部字段。

## 9.2 通信原语：前向和反向是否对称？

ND Parallelism 的通信分三层：

### CP attention 通信

Axolotl 不直接写 all-to-all / send-recv，而是调用 ring-flash-attn，并把 CP group 传进去：

- varlen path：`llama3_flash_attn_varlen_func(..., group=process_group)`（`ring_attn/patch.py:110-124`）。
- batch path：`ring_flash_attn_func(..., group=process_group)`（`adapters/batch.py:137-149`）。

通信发生频率：每个 attention layer 的 forward/backward 内部。具体 send/recv/all-gather 细节在外部 `ring_flash_attn` 包内，Axolotl 源码未展开，因此这里不凭空推断具体 primitive 次数。

### CP output gather 通信

`AllGatherWithGrad` 是显式 autograd function：forward 先 all_gather shape，再 all_gather tensor（`sequence_parallel.py:393-414`）；backward 只 slice，不 reduce（`sequence_parallel.py:418-444`）。这不是对称 all-gather / reduce-scatter，而是 “forward concat，backward 切片”。适用于后续只需要完整输出、梯度天然可按 sequence 切回的场景。

### FSDP 通信

FSDP2 通信主要由 PyTorch `fully_shard` / DTensor / FSDP runtime 负责；Axolotl 只传入 mesh、reshard_after_forward、offload_policy、mixed precision policy（`fsdp2.py:351-360`）。保存和加载时 Axolotl 有显式通信：

- 初始化加载：`distribute_tensor(..., src_data_rank=0)` 或 `dist.broadcast`（`fsdp2.py:47-82`）。
- 保存 FULL_STATE_DICT：`DTensor.full_tensor()` + per-param barrier（`fsdp2.py:158-173`）。

### 梯度缩放

CP 的 `num_items_in_batch` 修正使用 AVG 而不是 SUM（`sequence_parallel.py:156-160`），eval loss 使用 weighted SUM / SUM（`sequence_parallel.py:321-335`）。这说明 Axolotl 对 loss token 归一化做了显式补偿，避免 CP 维度重复平均或重复计数。

## 9.3 配置归一化：用户配置如何变成真实行为？

一个配置项通常会经过四层转换：

```text
YAML 字段
  -> Pydantic schema / validation
  -> env var 或 model_kwargs / TrainingArguments
  -> Accelerate / Transformers / PyTorch / ring-flash-attn 行为
```

以 `context_parallel_size` 为例：

1. schema 字段在 `config.py:975-980`。
2. validation 要求 flash attention、ring_flash_attn、micro_batch_size 约束，并默认 ring_attn_func（`validation.py:1508-1579`）。
3. env 写入 `PARALLELISM_CONFIG_CP_SIZE` 和 `ACCELERATE_ALLOW_CP_STANDALONE`（`trainer.py:632-638`）。
4. `build_parallelism_config` 创建 `cp_size` mesh（`distributed.py:299-316`）。
5. `SequenceParallelContextManager` 在训练时用 CP group 切分输入并注册 ring attention（`train.py:205-220`、`sequence_parallel.py:246-253`）。

以 `tensor_parallel_size` 为例：

1. schema 字段在 `config.py:993-998`。
2. validation 拒绝某些 8bit optimizer（`validation.py:1600-1608`），DeepSpeed 路径还会改临时 JSON（`validation.py:1121-1148`）。
3. env 写入 `PARALLELISM_CONFIG_TP_SIZE`（`trainer.py:623-625`）。
4. ModelLoader 把 `tp_size/tp_plan/device_mesh` 传给 Transformers（`model.py:749-755`）。
5. Tokens/sec callback 也把 TP 算作 non-data-parallel 维（`tokens_per_second.py:27-38`、`tokens_per_second.py:86-92`）。

以 `fsdp_config` 为例：

1. schema 在 `utils/schemas/fsdp.py:10-76`。
2. validation 归一 prefix / version，并做 optimizer/torch version 约束（`validation.py:1004-1117`、`config.py:1720-1735`）。
3. env 写入 `FSDP_*`（`trainer.py:589-618`）。
4. TrainingArguments 接收 `fsdp_config` / `fsdp`（`core/builders/base.py:604-607`）。
5. FSDP2 prepare 被 Axolotl patch 后执行 `fully_shard`（`fsdp2.py:279-449`）。

> 💡 **小结**
>
> * Axolotl 的 ND Parallelism 通过 monkey patch “接缝式”嵌入多个下游库，低侵入但维护成本高。
> * CP 的显式通信分两类：attention 内 ring 通信和可选输出 all_gather；不要把两者混为一谈。
> * 配置项的真实行为通常不在 schema，而在 env、model_kwargs、TrainingArguments 和 patched 下游函数共同作用后才确定。

# 十、显存、性能与通信分析

## 10.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ✅ FSDP2 节省；TP 也可分层内参数 | FSDP2 `fully_shard` 把参数放到 FSDP mesh；TP 通过 Transformers `tp_plan` 切 layer。 |
| 梯度 | ✅ FSDP2 节省 | FSDP shard 梯度；具体通信由 PyTorch FSDP runtime 管理。 |
| optimizer state | ✅ FSDP2 节省，但保存有 TODO | FSDP sharding 覆盖 optimizer state；但 checkpoint optimizer 保存可能失败并降级告警（`checkpoints.py:13-22`）。 |
| 激活值 | ✅ CP 节省 sequence 维；activation checkpointing 另行节省 | CP pre-hook 把 `[B,S]` 切成 `[B,S/cp]`；FSDP activation checkpointing 在 prepare 前应用（`fsdp2.py:327-342`）。 |
| logits | ✅ 普通 SFT 节省；GRPO/EBFT 部分恢复冗余 | `gather_outputs` 仅 GRPO/EBFT 为 true（`train.py:217`）；普通 SFT 不 all_gather logits。 |
| 输入 batch | ❌ dataloader 输出仍完整；forward 前切分 | 每个 CP group rank 拿同一份 batch，然后 hook 切分。输入本身不靠 dataloader 省内存。 |
| 中间 buffer | ⚠️ CP 有 padding / all_gather buffer；保存有 CPU clone | padding 到可整除长度；GRPO/EBFT output gather 需要 full tensor；CP 保存会 `detach().cpu()`。 |
| 初始化 CPU 内存 | ✅ 对非 rank0 有明显收益 | FSDP2 cpu_ram_efficient_loading 下非 rank0 meta loading；rank0 仍持有 full weights。 |
| 保存 CPU 内存 | ❌ rank0 可能升高 | FSDP2 FULL_STATE_DICT rank0 聚合，CP save clone 到 CPU。 |

真正的大头取决于场景：

- 长序列 SFT：activation / attention / logits 往往是 CP 的主要收益点。
- 大模型全参：参数、梯度、optimizer state 是 FSDP 的主要收益点。
- TP：主要降低单 rank layer 计算和参数驻留，但通信频繁，适合高速互联。
- QLoRA：参数本身量化后已经小，FSDP2 的收益更多受 adapter、activation、optimizer、初始化峰值限制。

## 10.2 通信开销

| 阶段 | 通信类型 | group | 频率 | 源码依据 |
|---|---|---|---|---|
| CP attention | ring-flash-attn 内部通信 | CP group | 每层 attention forward/backward | Axolotl 传 `group=process_group` 给 ring attention（`patch.py:110-124`、`batch.py:137-149`）。 |
| CP token count | `dist.all_reduce(AVG)` | CP group | 有 `num_items_in_batch` 时每 forward | `sequence_parallel.py:150-165`。 |
| CP eval loss | `dist.all_reduce(SUM)` 两次 | CP group | eval forward | `sequence_parallel.py:321-335`。 |
| CP output gather | `dist.all_gather` shape + tensor | CP group | `gather_outputs=True` 时每 forward | `sequence_parallel.py:393-414`。 |
| FSDP2 training | parameter all-gather / reduce-scatter 等 | FSDP mesh | 每层 / 每参数，取决于 FSDP runtime | Axolotl 传 `mesh` / `reshard_after_forward` 给 `fully_shard`（`fsdp2.py:351-415`）。 |
| FSDP2 load | `distribute_tensor` 或 `dist.broadcast` | FSDP / world 相关 | 初始化时每参数 | `fsdp2.py:47-82`。 |
| FSDP2 save | `DTensor.full_tensor()` + `barrier` | DTensor mesh / world | 保存时每参数 | `fsdp2.py:158-173`。 |
| 多机 launcher | rendezvous / NCCL | 全局 | 启动 / 通信 runtime | torchrun docs and CLI path（`docs/multi-node.qmd:58-92`、`cli/utils/train.py:195-218`）。 |

CP 和 TP 都是高频通信维度。文档也明确提醒 TP 需要高速互联，不推荐跨节点（`docs/nd_parallelism.qmd:28-34`）。HSDP 的多机设计意图正是把频繁的 FSDP/TP 尽量限制在节点内，把跨节点做成较低频的 replication / gradient sync（`docs/nd_parallelism.qmd:43-49`）。

## 10.3 性能取舍

ND Parallelism 的本质是多种资源交换：

- CP：用 ring attention 通信换 sequence 维激活 / logits 显存。
- FSDP2：用参数 all-gather / reduce-scatter 换参数、梯度、optimizer state 显存。
- TP：用层内频繁小通信换单 rank 计算 / 参数压力。
- CPU RAM efficient loading：用 rank0 广播和 meta patch 换非 rank0 CPU 内存。
- Sharded save：用后处理 merge 换保存时内存峰值。
- Monkey patch：用维护风险换接入速度和用户配置简洁。

源码里几个潜在瓶颈很明显：

1. `fsdp2.get_state_dict` 每个参数后 `barrier()`，大模型保存可能串行化严重（`fsdp2.py:171-173`）。
2. CP output gather 会额外 all_gather shape 和 tensor，且在 GRPO/EBFT 中恢复完整输出（`sequence_parallel.py:393-414`）。
3. `ring_flash_attn_func` 被 `torch.compile` 包装（`adapters/batch.py:35-37`），首次编译可能带来 warmup 开销。
4. QLoRA Params4bit 路径不能完全享受 meta sharding 优化，可能仍有初始化 VRAM 峰值（`fsdp2.py:362-371`）。
5. TP tied embeddings 直接不兼容（`loaders/utils.py:139-148`），真实可用模型范围受限。

> 💡 **小结**
>
> * ND Parallelism 没有“免费午餐”：FSDP/CP/TP 都是在用通信换不同维度的显存压力。
> * 普通 SFT 的 CP 显存收益最干净，因为 logits 不 gather；RL 路径会因为输出恢复而部分抵消收益。
> * 保存和初始化也是性能路径，尤其 rank0 聚合、per-param barrier、CPU clone 都可能成为大模型瓶颈。

# 十一、配置项、边界条件与坑点

## 11.1 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `context_parallel_size > 1` | `validation.py:1508-1579`、`trainer.py:632-638`、`train.py:205-220` | 启用 CP 校验、写 `PARALLELISM_CONFIG_CP_SIZE`、进入 `SequenceParallelContextManager` | 必须 `flash_attention: true`；需要 ring-flash-attn；sample_packing 时 `micro_batch_size=1`。 |
| `ring_attn_func` | `validation.py:1563-1579`、`ring_attn/patch.py:186-211` | 选择 `varlen_llama3` 或 `batch_ring` patch | enum 只启用两个；zigzag/stripe 不在主路径。 |
| `heads_k_stride` | `train.py:216`、`ring_attn/patch.py:118-124` | 传给 varlen llama3 ring attention | stride 越大文档说更快但更耗内存；源码未校验 KV heads 整除。 |
| `tensor_parallel_size > 1` | `trainer.py:623-625`、`model.py:749-755` | 写 TP env，Transformers `tp_plan="auto"` | tied word embeddings 不兼容；部分 8bit optimizer 被拒绝；不建议跨节点。 |
| `dp_shard_size` | `distributed.py:347-354`、Accelerate mesh | 显式设置 FSDP shard 维 | 没有 FSDP 时配置 dp_shard 会报错；与 CP/TP 的乘积必须匹配 world_size。 |
| `dp_replicate_size` | `distributed.py:343-357` | 启用 HSDP / replicate 维 | 纯 DDP + TP/CP 被 Accelerate 拒绝；rank 顺序需和节点布局匹配。 |
| `fsdp_config` | `trainer.py:589-618`、`builders/base.py:604-607`、`patch_manager.py:270-299` | 启用 FSDP env、TrainingArguments、FSDP patch | FSDP1 deprecated；FSDP2 要 torch >= 2.7；部分 optimizer/quant/RL 组合不兼容。 |
| `fsdp_config.cpu_ram_efficient_loading` | `model.py:756-780`、`fsdp2.py:371-425` | rank0 CPU / 非 rank0 meta loading，prepare 后广播 | rank0 仍持有 full weights；Params4bit 路径受限；需要 tied-key/meta patch。 |
| `fsdp_config.state_dict_type` / `final_state_dict_type` | `train.py:294-333` | 控制最终保存 FULL/SHARDED/LOCAL | SHARDED 可能需要 merge；FULL rank0 CPU 内存高。 |
| `sample_packing: true` + CP | `validation.py:1522-1526`、`sequence_parallel.py:53-56` | 默认 `varlen_llama3`，通过 position_ids 更新 cu_seqlens | `micro_batch_size` 必须 1；position_ids / packed boundaries 很关键。 |
| `use_liger_loss` + GRPO + CP | `validation.py:719-729` | 直接拒绝 | 当前 GRPO + SP + Liger 不支持。 |
| `liger_fused_linear_cross_entropy` + TP | `integrations/liger/args.py:108-113` | 直接拒绝 | Liger loss 与 TP 不兼容。 |

## 11.2 开启该特性的最小配置

单独 CP 的最小关键字段是：

```yaml
flash_attention: true
context_parallel_size: 2
# 可选：ring_attn_func: varlen_llama3 / batch_ring
```

如果 sample packing：

```yaml
sample_packing: true
micro_batch_size: 1
context_parallel_size: 2
```

FSDP + TP + CP 的最小骨架类似示例：

```yaml
fsdp_version: 2
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3DecoderLayer
  state_dict_type: FULL_STATE_DICT
  reshard_after_forward: true

dp_shard_size: 2
tensor_parallel_size: 2
context_parallel_size: 2
flash_attention: true
```

这不是完整训练配置，只是 ND 并行相关骨架。实际还需要 model、dataset、optimizer、batch、dtype 等。

## 11.3 静默失效和不兼容组合

- `ring_attn_func` 如果未设置，会根据 sample_packing 默认选择；不是静默无效，但可能和用户预期不同（`validation.py:1571-1577`）。
- `sequence_parallel_degree` 仍可迁移，但会 warning，未来风险更高（`validation.py:1508-1514`）。
- TP + tied word embeddings 直接报错（`loaders/utils.py:139-148`），e2e TP 测试目前也被 skip（`tests/e2e/multigpu/test_tp.py:17-21`）。
- `dp_shard_size` 不配 FSDP 会报错（`distributed.py:347-352`）。
- `context_parallel_size > 1` 但没装 ring-flash-attn 会 ImportError（`validation.py:1528-1550`）。
- GRPO + CP + Liger loss 被拒绝（`validation.py:719-729`）。
- FSDP2 + DPO/KTO/ORPO/IPO + 4bit/8bit base model 被拒绝（`validation.py:1034-1048`）。
- FSDP2 + `adamw_8bit` / `adamw_bnb_8bit` 被拒绝，提示用 `adamw_torch_8bit`（`validation.py:1102-1117`）。

## 11.4 单机 / 多机差异

Axolotl CLI 支持 Accelerate 和 torchrun。多机文档建议每台机器使用同 commit、同配置文件，主节点可达（`docs/multi-node.qmd:6-14`）。Accelerate 多机依赖默认 config 中 `machine_rank`、`main_process_ip`、`num_machines`、`num_processes` 等字段（`docs/multi-node.qmd:16-39`）。torchrun 路径建议设置 NCCL IB 环境，并用 `axolotl train config.yaml --launcher torchrun -- --nnodes ...` 启动（`docs/multi-node.qmd:58-92`）。

源码层面，`_launch_torchrun_training` 只是拼接 torchrun 参数并补 rendezvous backend/id（`src/axolotl/cli/utils/train.py:195-218`），并不会根据物理节点自动调整 mesh。HSDP 配置能否把 FSDP 留在节点内，很大程度取决于全局 rank 排布是否与 mesh order 对齐。

> 💡 **小结**
>
> * 配置项的影响不是“开关表”，而是决定校验、env、mesh、model_kwargs、hooks、保存路径的组合。
> * CP 最小配置看似简单，但 flash attention、ring-flash-attn、sample packing batch size 都是硬前提。
> * 多机 ND 并行尤其依赖 rank 顺序和 launcher 配置，Axolotl 源码没有额外的拓扑感知重排。

# 十二、测试、示例与覆盖缺口

## 12.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/test_loaders.py:181-218` | `_get_parallel_config_kwargs` 推导 | 覆盖 TP/CP/FSDP/HSDP 组合的参数推导，但不启动真实 mesh 通信。 |
| `tests/test_context_parallel_batch_size.py:29-56` | CP 下 batch size 归一化 | Mock ring_flash_attn，验证 effective world_size 会除以 CP。 |
| `tests/test_tensor_parallel_batch_size.py:28-55` | TP 下 batch size 归一化 | Mock model config，验证 effective world_size 会除以 TP。 |
| `tests/monkeypatch/test_trainer_context_parallel_patch.py:36-66` | Transformers CP guard patch | 验证 patch 替换 guard 且幂等；测试后手动恢复。 |
| `tests/e2e/multigpu/test_fsdp2.py:49-100` | FSDP2 FFT SFT + cpu_ram_efficient_loading True/False | 2 GPU smoke 级训练，覆盖 FSDP2 基础路径。 |
| `tests/e2e/multigpu/test_llama.py:476-520` | FSDP2 + sample packing + SHARDED_STATE_DICT | 覆盖 sharded state dict 配置路径。 |
| `tests/utils/schemas/validation/test_fsdp.py:16-174` | FSDP schema / validation | 覆盖 version 迁移、prefix 剥离、optimizer/quant/RL 不兼容。 |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:1-46` | 单机 FSDP + TP + CP 示例 | 展示 8 GPU 组合配置。 |
| `examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml:1-47` | 多机 HSDP + TP 示例 | 展示 dp_replicate + dp_shard + TP。 |
| `docs/nd_parallelism.qmd:51-108` | ND 用户文档和支持矩阵 | 说明 FSDP/TP/CP/HSDP 组合意图。 |

## 12.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| CP e2e 主训练 | ⚠️ 有测试但 skip | `tests/e2e/multigpu/patched/test_sp.py:102-137` 因 ring_flash_attn / Transformers upstream 维护问题被 skip，主路径缺少新鲜 e2e 保护。 |
| FSDP + TP + CP 三维组合真实训练 | 未在搜索结果中确认 | 只有 config/helper 推导和示例，未确认 e2e 覆盖所有三维同时启用。 |
| 多机 HSDP + TP/CP | 未在源码测试中确认 | rank 排布、跨节点通信、NCCL 环境可能在真实集群暴露问题。 |
| CP 保存 / resume | 部分保存修复，未确认完整 resume e2e | CP 下 CPU clone 保护保存，但 optimizer/scheduler FSDP2 TODO 可能影响 resume。 |
| ring attention patch 恢复 | 单测手动恢复 Trainer patch，但运行时无恢复 | 同进程多模型/多测试可能受全局 patch 污染。 |
| TP + tied embeddings | 被 skip / 被拒绝 | 可用模型范围受限；用户可能误以为所有 causal LM 都能 TP。 |
| QLoRA + FSDP2 cpu_ram_efficient_loading 显存峰值 | 有特殊路径，未确认性能/显存测试 | Params4bit 不能完全 meta 优化，可能 OOM。 |
| FSDP2 FULL_STATE_DICT 大模型保存性能 | 未见性能测试 | rank0 聚合 + per-param barrier 可能导致保存极慢或 CPU OOM。 |
| `heads_k_stride` 合法性 | 未见 schema 校验 | 如果不能整除 KV heads，错误可能来自 ring-flash-attn 下游。 |

## 12.3 示例证明了什么，没证明什么

`examples/distributed-parallel/README.md:21-45` 给了两个强信号：

- 单机 8 GPU 的目标组合是 FSDP + TP + CP。
- 多机目标组合是 HSDP + TP，把跨节点通信降到 replicate 维。

但示例不是测试。它们证明“项目期望用户这样配置”，不证明“所有模型、所有 optimizer、所有保存/resume 组合都稳定”。测试层面真正比较扎实的是 FSDP2 基础训练和配置校验；CP 主训练 e2e 被 skip，是当前最大的覆盖缺口之一。

> 💡 **小结**
>
> * 单元测试覆盖了配置推导和 batch size 语义，FSDP2 有多条 2 GPU e2e。
> * CP 主 e2e 目前被 skip，三维 ND 组合和多机 HSDP 主要依赖示例与下游库能力。
> * 保存 / resume / 大模型性能缺少系统性测试，尤其 FSDP2 optimizer state 和 rank0 full state dict。

# 十三、局限性与已知优化点

## 13.1 硬约束

- CP 必须启用 `flash_attention: true`（`validation.py:1516-1520`）。
- CP 需要安装 `ring_flash_attn`，项目 optional extra 是 `ring-flash-attn>=0.1.7` 和 `flash-attn==2.8.3`（`pyproject.toml:91-96`）。
- sample packing + CP 要求 `micro_batch_size=1`（`validation.py:1522-1526`）。
- TP 不支持 tied word embeddings（`loaders/utils.py:139-148`）。
- TP 不支持部分 bitsandbytes 8bit optimizer（`validation.py:1600-1608`）。
- FSDP2 要求 torch >= 2.7.0（`config.py:1720-1735`），当前项目依赖更高。
- `dp_shard_size` 不配 FSDP 会报错（`distributed.py:347-352`）。
- GRPO + CP + Liger 不支持（`validation.py:719-729`）。
- Tensor parallelism 与 Liger fused linear cross entropy 不兼容（`integrations/liger/args.py:108-113`）。

## 13.2 维护成本

- Transformers `_prepare_context_parallel_inputs` patch 依赖源码字符串 `GUARD_PATTERN`（`trainer_context_parallel.py:15-38`）。一旦上游改写 guard，patch 会跳过。
- Ring attention adapter 检查 `_flash_attention_forward` 签名，不匹配就报版本不支持（`adapters/batch.py:167-190`）。
- `SequenceParallelContextManager.__exit__` TODO 表示 attention 和 accelerate patch 不会恢复（`sequence_parallel.py:238-245`）。
- FSDP2 patch 覆盖 Accelerate 函数，依赖 `accelerator.state.parallelism_config.fsdp_dim_names` 等内部结构（`fsdp2.py:351-360`）。
- QLoRA / Params4bit 需要单独加载与 quant_state 处理，维护面横跨 bitsandbytes、Transformers、FSDP2（`model_shard_quant.py:103-117`）。

## 13.3 性能瓶颈

- CP attention 每层都要在 CP group 内通信；长序列越长，显存收益越明显，但通信也越关键。
- TP 每层内部有频繁小通信，不适合慢速跨节点互联，文档也提醒需要 NVLink 等高速连接（`docs/nd_parallelism.qmd:28-34`）。
- FSDP2 `get_state_dict` 每个参数 full_tensor 后 barrier，保存大模型可能串行瓶颈明显（`fsdp2.py:158-173`）。
- `fsdp2_load_full_state_dict` 初始化时逐参数 distribute/broadcast，rank0 是权重源（`fsdp2.py:20-97`）。
- CP output gather 在 GRPO/EBFT 会恢复完整输出，降低 CP 的显存收益（`sequence_parallel.py:359-416`）。
- CP save CPU clone 会增加保存时 CPU 内存（`core/trainers/base.py:812-823`）。

## 13.4 已知优化点

源码中能看到几个直接的 TODO / FIXME：

- `sequence_parallel.py` TODO：zigzag、stripe patterns 尚未实现（`sequence_parallel.py:22-23`）。
- `SequenceParallelContextManager.__exit__` TODO：尚未 un-patch attention / accelerate functions（`sequence_parallel.py:244`）。
- `checkpoints.py` TODO：fix FSDP2 optimizer saving（`checkpoints.py:17`）。
- `fsdp2.py` TODO：review whether LoRA ParamWrapper needs separate sharding（`fsdp2.py:240-242`）。
- `model.py` TODO：Transformers TP `_tp_size/_device_mesh` workaround 待上游修复后移除（`model.py:852-857`）。

工程上可以继续优化的方向包括：

1. FSDP2 保存改为分块 / 异步 / 减少 per-param barrier，降低 rank0 串行瓶颈。
2. CP patch 增加可恢复机制，避免同进程多任务污染。
3. 为 `heads_k_stride` 增加模型 KV head 可整除校验，提前报错。
4. 增加三维 ND e2e 和多机拓扑测试，尤其验证 rank order 与 HSDP 组是否符合预期。
5. 对 QLoRA Params4bit + FSDP2 初始化做更细粒度的显存 profiling，避免“理论省内存，实际初始化 OOM”。

> 💡 **小结**
>
> * 当前 ND Parallelism 的硬约束主要来自 CP/ring-flash-attn、TP tied embeddings、FSDP2 optimizer/quant 组合。
> * 维护风险集中在 monkey patch 和下游库内部签名。
> * 性能优化空间最大的是保存路径、CP patch 恢复、三维组合 e2e 和多机拓扑验证。

# 小结与展望

Axolotl 的 ND Parallelism 实现可以用几个关键词概括。

## 关键词一：环境变量驱动的拓扑注入

用户配置先被归一化成 `PARALLELISM_CONFIG_*` 和 `FSDP_*` 环境变量，再由 Accelerate 在创建 Accelerator 时读取。这让 Axolotl 不必侵入 Accelerator 构造过程，但也让“谁创建了 parallelism_config”变得不直观：ModelLoader 有自己的 `device_mesh`，Trainer/Accelerator 也会从 env 重建一份。

## 关键词二：DeviceMesh 统一 rank 语义

TP、CP、FSDP 不再靠散乱 process group，而是共享 `DeviceMesh`。CP group 从 `device_mesh[("cp",)]` 取，FSDP2 mesh 从 `dp_shard_cp` flatten 维取，TP 由 Transformers `tp_plan` 消费。这个设计让三维并行能组合，但也要求 world_size、rank order、dp_shard/cp/tp size 精确匹配。

## 关键词三：Forward hook 切 sequence

Axolotl 的 CP 主流程不是数据层切分，而是在模型 forward 前把完整 batch 切成 local sequence chunk。普通 SFT 不 gather logits，因此真正节省 logits / activation 显存；GRPO/EBFT 因算法需要会 gather 输出，收益会部分回撤。

## 关键词四：Monkey patch 作为接缝

为了让 Transformers flash attention、Accelerate CP/FSDP2、PEFT LoRA DTensor、ring-flash-attn 一起工作，Axolotl 选择了多处 monkey patch。它让用户配置保持简单，也让框架能快速跟进上游能力；但长期维护上，patch 恢复、签名检测、版本升级都会是风险点。

## 关键词五：通信换显存

这套实现最适合的场景是：模型参数大、上下文长、GPU 间互联足够快，且用户愿意接受更复杂的保存 / resume / patch 风险。比如单机 8×H100/NVLink 上的 FSDP + TP + CP，或多机中节点内 FSDP/TP、节点间 replicate 的 HSDP。

它不适合的场景也很明确：

- 低速跨节点网络上强行 TP/CP；
- tied embeddings 且需要 TP 的模型；
- 强依赖完整 optimizer checkpoint resume 的长任务，但当前 FSDP2 optimizer 保存仍可能降级；
- 不愿承担 ring-flash-attn / Transformers 内部签名变化风险的生产环境。

和替代方案相比，DeepSpeed ZeRO-3 更成熟但不解决所有 CP/TP 组合；纯 FSDP2 更简洁但长序列激活仍可能 OOM；纯 CP 能救长上下文但参数仍冗余。Axolotl 的 ND Parallelism 把这些方案拼在一起，优势是组合能力，代价是调试复杂度。

后续值得继续走读的方向有三个：

1. Transformers 原生 TP 在不同模型架构上的 `tp_plan="auto"` 细节；
2. ring-flash-attn 内部 forward/backward 通信和显存曲线；
3. FSDP2 + PEFT/QLoRA 保存与 resume 的完整状态一致性。

如果用一句话收束：Axolotl 的 ND Parallelism 不是“多加几个配置项”，而是在配置层、拓扑层、模型加载层、forward 层和保存层分别打了一组补丁，把 CP、TP、FSDP2 拼成一条可训练链路。它的工程价值很高，但也要求使用者理解每个维度到底在省什么、通信什么、以及哪里可能坏。
