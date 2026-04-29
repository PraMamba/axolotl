# Axolotl 源码走读：Distributed Muon Optimizer support for FSDP2 pretraining 实现解析

在大模型预训练里，优化器从来不只是“更新一下参数”这么简单。FSDP2 已经把模型参数、梯度和部分状态切成了 shard，但 Muon 这类带矩阵正交化的优化器又天然想看见“完整矩阵”——这就是 Axolotl 这次 Distributed Muon Optimizer 支持背后的核心矛盾。

本文不展开 Muon 论文本身，也不重新讲 FSDP2 的理论。我们只顺着 Axolotl 的源码，看一个用户写下 `optimizer: muon` 和 `fsdp_version: 2` 之后，配置如何变成真正的训练行为：谁创建 DeviceMesh，谁选择 DistMuon，optimizer 为什么先于 FSDP2 wrap 创建却仍能更新 DTensor shard，Muon 的 all-to-all 到底在传什么，以及保存 / resume / 测试覆盖有哪些边界。

# 前言

## 业务 / 工程背景

目标特性来自 README 的更新说明：`Distributed Muon Optimizer support has been added for FSDP2 pretraining`（`README.md:38-42`）。对应示例是 `examples/qwen2/muon-pretrain-fsdp2.yaml`：它选择 Qwen2.5-0.5B、`pretraining_dataset`、`optimizer: muon`、`learning_rate: 0.02`，并开启 `fsdp_config.fsdp_version: 2`、`state_dict_type: FULL_STATE_DICT`、`reshard_after_forward: true`（`examples/qwen2/muon-pretrain-fsdp2.yaml:12-68`）。

从业务上看，它服务的是“多卡 FSDP2 预训练 / 全参训练”场景，而不是典型小 batch 指令微调。Muon 的本地实现文档也提醒：它可能不适合小 batch，也未充分测试 finetuning 场景（外部依赖 `axolotl-contribs-mit==0.0.6` 中 `muon.py:53-56`）。

## 核心矛盾

这个特性有三层工程冲突：

1. **FSDP2 希望参数始终分片，Muon 正交化希望看见矩阵。** 如果每张卡只拿到 `[out/world, in]` 的局部 shard，直接对 shard 做 Newton-Schulz 正交化，语义就不是对完整矩阵做正交化。
2. **Transformers / Accelerate 的 FSDP2 要求 model 和 optimizer 一起 `prepare()`。** 但优化器通常在 wrap 前创建，wrap 后参数对象会变成 DTensor，需要可靠地把 optimizer param group 切换到新参数。
3. **Muon 的矩阵更新比 AdamW 多了额外通信。** 它为了节省 FSDP2 参数/状态显存，引入 optimizer step 阶段的 all-to-all / all-gather 调度。

## 本文主线

本文按机制而不是按文件展开：

1. 配置入口：用户如何真正开启 DistMuon，哪些配置只是约束或下游环境变量。
2. FSDP2 初始化：为什么 optimizer 创建在 wrap 前，仍能在 step 时拿到 DTensor。
3. DistMuon 调度：参数如何分组，Muon 与 AdamW fallback 如何分流。
4. 通信与 shape：一次 optimizer step 中 all-to-all / all-gather 分别解决什么问题。
5. 保存、显存、性能、测试与边界：哪些路径已覆盖，哪些风险还暴露在源码中。

## 不展开的内容

本文不讲 Muon 数学推导，不讲 FSDP2 内部 flatten / unflatten 细节，不讲 Qwen2 模型结构，也不比较 Muon 与 AdamW 的收敛曲线。相关判断都以 Axolotl 仓库和其 pin 住的 `axolotl-contribs-mit==0.0.6` 源码为准；涉及 Transformers / Accelerate 的部分，只用于还原 Axolotl 依赖的下游主路径。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `README.md` | 声明 Distributed Muon for FSDP2 pretraining 已加入项目。 |
| `examples/qwen2/muon-pretrain-fsdp2.yaml` | 推荐的 FSDP2 pretraining + Muon 示例配置。 |
| `src/axolotl/cli/config.py` | 读取 YAML、校验配置、准备 FSDP / parallelism 环境变量。 |
| `src/axolotl/utils/schemas/validation.py` | 拒绝 Muon + DeepSpeed、Muon + FSDP1 等不兼容组合。 |
| `src/axolotl/core/builders/base.py` | 根据 `optimizer: muon` 和 DeviceMesh 选择 `DistMuonOptimizerFactory` 或本地 `MuonOptimizerFactory`。 |
| `src/axolotl/core/trainers/mixins/optimizer.py` | 在 Trainer 创建 optimizer 时识别 `BaseOptimizerFactory` 并调用工厂。 |
| `src/axolotl/loaders/patch_manager.py` | 在 FSDP2 场景注册 Accelerate / ParallelismConfig monkey patch。 |
| `src/axolotl/monkeypatch/accelerate/fsdp2.py` | 替换 Accelerate 的 FSDP2 prepare / get_state_dict 逻辑，并处理 rank0 loading。 |
| `axolotl.contribs.mit.muon.dist_muon.py` | 外部 MIT contrib 包中的 Distributed Muon 主实现。 |
| `src/axolotl/core/trainers/mixins/checkpoints.py` | 捕获 FSDP2 optimizer state 保存异常，暴露 resume 风险。 |

> 注：`axolotl.contribs.mit.muon.dist_muon.py` 不在仓库 `src/` 目录内，而来自 `pyproject.toml:70-72` pin 住的 `axolotl-contribs-mit==0.0.6`。Axolotl 主仓库通过 import 接入它。

# 一、配置入口与行为开关：`optimizer: muon` 如何变成真正的 DistMuon

## 1.1 设计哲学与核心问题

这一层解决的是“**用户意图如何被框架安全识别**”的问题。

Muon 在 Axolotl 里不是 Transformers 原生 optimizer，而是一个 custom optimizer。用户只写 `optimizer: muon`，框架必须完成三件事：

- schema 允许这个 optimizer 名字；
- validation 拒绝已知错误组合；
- builder 把这个名字映射成真正的 optimizer factory。

如果缺少 validation，Muon 可能被放进 DeepSpeed 或 FSDP1 路径，结果不是静默退化，就是在训练中途用很难定位的错误崩掉。如果缺少 builder 里的 DeviceMesh 判断，多卡 FSDP2 就会落回单机 Muon 逻辑，无法对 sharded matrix 做正确调度。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：Click CLI 入口，最终 launch `axolotl.cli.train`

src/axolotl/cli/train.py
  - do_cli：加载配置，调用 do_train

src/axolotl/cli/config.py
  - load_cfg：读取 YAML，validate_config，prepare_optim_env

src/axolotl/utils/schemas/enums.py
  - CustomSupportedOptimizers.muon：把 muon 放入自定义 optimizer 枚举

src/axolotl/utils/schemas/validation.py
  - check_muon_deepspeed_fsdp：拒绝 DeepSpeed / FSDP1 组合

src/axolotl/core/builders/base.py
  - TrainerBuilderBase._configure_optimizer：选择 Muon / DistMuon factory
```

## 1.3 主流程拆解

用户入口不是某个 Python API，而是标准训练命令：

```text
User: axolotl train examples/qwen2/muon-pretrain-fsdp2.yaml --num-processes N
  -> src/axolotl/cli/main.py:98 train(...)
     -> src/axolotl/cli/utils/train.py:109 launch_training(...)
        -> accelerate launch -m axolotl.cli.train <config>
           -> src/axolotl/cli/train.py:55 do_cli(...)
              -> src/axolotl/cli/config.py:230 load_cfg(...)
```

`cli/main.py` 的 `train` 命令接收 config、launcher 和 CLI overrides（`src/axolotl/cli/main.py:98-128`），默认通过 `accelerate launch -m axolotl.cli.train` 启动 worker（`src/axolotl/cli/utils/train.py:157-185`）。worker 内部的 `do_cli` 再调用 `load_cfg`（`src/axolotl/cli/train.py:55-91`）。

`load_cfg` 负责把 YAML 变成 `DictDefault`，合并 CLI override，然后执行校验与环境准备：

```text
load_cfg(config)
  -> yaml.safe_load
  -> prepare_plugins(cfg)
  -> validate_config(cfg)
  -> prepare_optim_env(cfg)
  -> normalize_config(cfg)
```

对应源码在 `src/axolotl/cli/config.py:244-333`。Muon 本身在 schema 中被允许，是因为 `CustomSupportedOptimizers` 包含 `muon = "muon"`（`src/axolotl/utils/schemas/enums.py:82-92`），而训练配置的 `optimizer` 字段接受 `OptimizerNames | CustomSupportedOptimizers`（`src/axolotl/utils/schemas/training.py:77-85`）。

真正的“行为改变”发生在 builder：

```text
TrainerBuilderBase._configure_optimizer(...)
  if cfg.optimizer == "muon":
      _, device_mesh = build_parallelism_config(cfg)
      if device_mesh is not None:
          optimizer_cls = DistMuonOptimizerFactory
          optimizer_kwargs["device_mesh"] = device_mesh
      else:
          optimizer_cls = MuonOptimizerFactory
```

源码位置是 `src/axolotl/core/builders/base.py:277-316`。也就是说：

- `optimizer: muon` 只是名字；
- **是否 Distributed Muon，取决于 `build_parallelism_config(cfg)` 是否返回 DeviceMesh**；
- 单卡或无 parallelism config 时，仍会使用本地 `MuonOptimizerFactory`。

## 1.4 关键细节与误区澄清

> 容易误解点 1：README 写的是 “FSDP2 pretraining support”，是不是代码只允许 pretraining？

不是。源码没有检查 `pretraining_dataset` 才允许 Muon。validation 只拒绝 DeepSpeed，以及 FSDP 场景下非 FSDP2（`src/axolotl/utils/schemas/validation.py:906-918`）。测试里甚至有 SFT + FSDP2 的 DistMuon e2e（`tests/e2e/multigpu/test_dist_muon_fsdp2.py:51-107`）和 LoRA SFT + FSDP2（同文件 `108-168`）。所以“pretraining”是该特性的主推场景，不是硬编码限制。

> 容易误解点 2：`optimizer: muon` 一定就是分布式 Muon？

也不是。`base.py:299-314` 明确：只有 `build_parallelism_config` 返回 DeviceMesh 时才导入 `DistMuonOptimizerFactory`；否则导入本地 `MuonOptimizerFactory`。单卡 e2e 测试 `tests/e2e/test_optimizers.py:117-161` 断言的是 `Muon`，不是 `DistMuon`。

> 容易误解点 3：文档里的 `optimizer: muon` 足够用于 FSDP2 吗？

`docs/optimizers.qmd:106-113` 只写了最小 optimizer 配置，但 FSDP2 路径还需要 `fsdp_config` / `fsdp_version: 2`。示例文件给出了完整组合（`examples/qwen2/muon-pretrain-fsdp2.yaml:61-68`）。如果配置 FSDP 但不是 FSDP2，会在 validation 阶段报错。

## 1.5 本章小结

> 💡 **小结**
>
> * `optimizer: muon` 只是入口开关，真正 DistMuon 分支由 DeviceMesh 决定。
> * Axolotl 的 validation 明确拒绝 Muon + DeepSpeed 和 Muon + FSDP1。
> * README 的 “pretraining” 是目标场景，不是源码里的硬限制。

# 二、FSDP2 初始化与 patch：为什么 optimizer 先创建，step 时却能更新 DTensor

## 2.1 设计哲学与核心问题

这一层解决的是“**FSDP2 wrap 前后的参数对象如何保持一致**”。

在常规 PyTorch 训练里，optimizer 通常持有 model parameter 的引用；如果 FSDP2 wrap 后把参数替换成 DTensor，而 optimizer 还拿着旧参数，训练会出现一个严重错误：loss 正常 backward，但 optimizer 更新的是旧 tensor，真实模型参数不变。

Axolotl 并没有自己重写整个训练循环，而是利用 Transformers / Accelerate 的 FSDP2 prepare 机制，并对其中几个不满足 Axolotl 场景的点做 monkey patch。

## 2.2 源码入口与关键对象

```text
src/axolotl/utils/trainer.py
  - prepare_optim_env / setup_fsdp_envs / setup_parallelism_envs：写入 FSDP2 和 ParallelismConfig 环境变量

src/axolotl/loaders/model.py
  - ModelLoader.load：在模型加载前注册 patch，构建 parallelism_config/device_mesh

src/axolotl/loaders/patch_manager.py
  - _apply_fsdp_patches：注册 FSDP2 与 ParallelismConfig patch

src/axolotl/monkeypatch/accelerate/fsdp2.py
  - patch_accelerate_fsdp2：替换 accelerate.accelerator.fsdp2_prepare_model / Accelerator.get_state_dict
  - fsdp2_prepare_model：用 fully_shard 包装模型，并在需要时做 rank0 full state broadcast

transformers.trainer.Trainer._prepare_for_training
  - 下游主训练准备逻辑：FSDP2 下 model 和 optimizer 一起 accelerator.prepare

accelerate.accelerator.Accelerator.prepare
  - 下游逻辑：FSDP2 prepare 后用 mapping 切换 optimizer 参数
```

## 2.3 主流程拆解

配置加载阶段先把 FSDP2 变成环境变量：

```text
load_cfg
  -> prepare_optim_env(cfg)
     -> setup_fsdp_envs(cfg)
        - ACCELERATE_USE_FSDP=true
        - FSDP_VERSION=2
        - FSDP_STATE_DICT_TYPE=...
     -> setup_parallelism_envs(cfg)
        - PARALLELISM_CONFIG_* envs
```

源码依据：`prepare_optim_env` 在 `src/axolotl/utils/trainer.py:643-668`；`setup_fsdp_envs` 写入 `ACCELERATE_USE_FSDP`、`FSDP_VERSION`、`FSDP_STATE_DICT_TYPE` 等环境变量（`src/axolotl/utils/trainer.py:589-619`）；parallelism env 在 `src/axolotl/utils/trainer.py:621-640`。

模型加载阶段，PatchManager 注册 FSDP2 patch：

```text
ModelLoader.load
  -> patch_manager.apply_pre_model_load_patches()
  -> _apply_pre_model_load_setup()
       -> _set_parallel_config()
          -> build_parallelism_config(cfg)
  -> _build_model()
```

对应源码：`ModelLoader.load` 在 `src/axolotl/loaders/model.py:161-194`；`_apply_pre_model_load_setup` 在 `src/axolotl/loaders/model.py:196-223`；`_set_parallel_config` 在 `src/axolotl/loaders/model.py:437-442`。

FSDP2 patch 的注册在 `PatchManager._apply_fsdp_patches`：

```text
if cfg.fsdp_config and fsdp_version == 2:
    patch_accelerate_fsdp2()
    if cpu_ram_efficient_loading:
        patch_tied_keys_for_meta_device()
```

源码见 `src/axolotl/loaders/patch_manager.py:270-299`。这里是全局 monkey patch：`patch_accelerate_fsdp2` 直接替换 `accelerate.accelerator.fsdp2_prepare_model` 和 `accelerate.Accelerator.get_state_dict`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:529-533`）。

接下来最容易绕晕的是 optimizer 创建时机。当前 Transformers 训练准备逻辑中，FSDP2 不 delay optimizer creation：

```text
Trainer._prepare_for_training
  is_fsdp2 -> delay_optimizer_creation = False
  create_optimizer()
  model, optimizer = accelerator.prepare(model, optimizer)
```

源码见下游 `transformers/trainer.py:1555-1604`。Accelerate 的 FSDP2 prepare 又要求 model 和 optimizer 一起传入（`accelerate/accelerator.py:1505-1519`）。在 prepare 内部，它先调用 `fsdp2_prepare_model`，再构造 old parameter 到 new sharded parameter 的 mapping，并对 optimizer 执行 `fsdp2_switch_optimizer_parameters`（`accelerate/accelerator.py:1723-1761`；`accelerate/utils/fsdp_utils.py:557-585`）。

这解释了一个关键问题：**DistMuon optimizer 对象虽然在 FSDP2 wrap 前创建，但它的 param group 会在 Accelerate.prepare 里被替换成 FSDP2 sharded DTensor 参数。** 因此 DistMuon 的 `step()` 里看到的 `param` 可以是 DTensor。

## 2.4 关键细节与误区澄清

> 容易误解点 4：Axolotl 的 DistMuon 是不是自己 wrap FSDP2？

不是。DistMuon 不负责 FSDP2 wrapping。wrap 发生在 Accelerate 的 `fsdp2_prepare_model` 路径；Axolotl 只是通过 `patch_accelerate_fsdp2()` 替换了 prepare 函数实现（`src/axolotl/monkeypatch/accelerate/fsdp2.py:279-449`）。DistMuon 只消费 wrap 后的参数形态。

> 容易误解点 5：optimizer 先创建就一定拿不到 DTensor？

在 FSDP2 下不成立。Transformers 先 `create_optimizer()`，再 `accelerator.prepare(model, optimizer)`；Accelerate 在 prepare 中调用 `fsdp2_switch_optimizer_parameters`，把 optimizer param group 从旧 parameter 切到新 sharded parameter。这个切换不是 Axolotl 自己写的，但 Axolotl 的 DistMuon 正是依赖这个下游语义。

> 容易误解点 6：所有 patch 都是局部生效的吗？

不是。`patch_accelerate_fsdp2`、`patch_parallelism_config` 都是改模块 / 类属性的全局 monkey patch（`src/axolotl/monkeypatch/accelerate/parallelism_config.py:73-77`，`src/axolotl/monkeypatch/accelerate/fsdp2.py:529-533`）。它们没有上下文恢复逻辑；在同一 Python 进程内会持续影响后续 Accelerator 行为。

## 2.5 本章小结

> 💡 **小结**
>
> * FSDP2 的模型 wrap 由 Accelerate 完成，Axolotl 通过 monkey patch 改写关键 prepare / state_dict 行为。
> * optimizer 创建在 FSDP2 wrap 前，但 Accelerate.prepare 会把 optimizer 参数引用切换到 sharded DTensor。
> * patch 是进程级全局替换，不是 `with` 块内的局部状态。

# 三、DeviceMesh 与参数分组：DistMuon 如何决定谁做 Muon，谁退回 AdamW

## 3.1 设计哲学与核心问题

这一层解决的是“**Muon 并不适合所有参数**”。

Muon 的核心更新是对矩阵做 momentum + orthogonalization。对 embedding、lm_head、bias、norm 这类参数，强行做 Muon 没有意义甚至会破坏训练。因此 DistMuon 的 factory 必须在创建 optimizer 时把参数拆成多个 group：

- 二维及以上、非 embedding/head 的权重走 Muon；
- embedding/head、一维参数、bias/norm 等走 AdamW fallback；
- weight decay 与 no weight decay 再进一步拆组。

同时，它还必须从 Axolotl 的 DeviceMesh 里选出真正用于 FSDP shard 的通信组。

## 3.2 源码入口与关键对象

```text
src/axolotl/utils/distributed.py
  - build_parallelism_config：基于 world_size / tp / cp / dp_shard / dp_replicate 构建 ParallelismConfig 和 DeviceMesh

src/axolotl/core/builders/base.py
  - _configure_optimizer：把 device_mesh 放进 optimizer_kwargs

axolotl.contribs.mit.muon.dist_muon.py
  - DistMuonOptimizerFactory.__call__：抽取 dp_shard mesh，构造参数组
  - DistMuon.__init__：校验 mesh 类型，保存 rank/world/group
```

## 3.3 主流程拆解

Axolotl 构建 DeviceMesh 的逻辑集中在 `build_parallelism_config`：

```text
build_parallelism_config(cfg)
  -> _get_parallel_config_kwargs(
       world_size,
       tensor_parallel_size,
       context_parallel_size,
       dp_shard_size,
       dp_replicate_size,
       is_fsdp,
     )
  -> ParallelismConfig(**pc_kwargs)
  -> parallelism_config.build_device_mesh("cuda")
```

源码在 `src/axolotl/utils/distributed.py:299-316`。如果是 FSDP 且没有显式 `dp_shard_size`，剩余 world size 会默认变成 `dp_shard_size`（`src/axolotl/utils/distributed.py:338-362`）。这意味着标准 FSDP2 多卡训练会自然得到 `dp_shard` mesh dim。

DistMuon factory 再从 DeviceMesh 中抽出通信维度：

```text
if "dp_shard" in device_mesh.mesh_dim_names:
    distributed_mesh = device_mesh["dp_shard"]
elif device_mesh.ndim == 1:
    distributed_mesh = device_mesh
```

源码见外部 `dist_muon.py:602-610`。随后它遍历 `opt_model.named_parameters()`：

```text
for name, param in opt_model.named_parameters():
    if not param.requires_grad:
        continue
    if name endswith modules_to_save or contains embed_tokens/lm_head:
        AdamW embeddings group
    elif param.ndim < 2:
        AdamW decay/no_decay group
    else:
        Muon decay/no_decay group
```

源码在外部 `dist_muon.py:626-645`，group 构造在 `dist_muon.py:647-689`，最后返回 `DistMuon(..., distributed_mesh=distributed_mesh, lr=lr, ...)`（`dist_muon.py:691-696`）。

DistMuon 初始化时保存 rank、world size 和 process group：

```text
if DeviceMesh:
    require ndim == 1
    device_rank = mesh.get_local_rank()
    world_size = mesh.size()
    process_group = mesh.get_group()
elif ProcessGroup:
    device_rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
else:
    world_size = 1
```

源码在外部 `dist_muon.py:96-117`。这里的 `world_size` 不是全局 `WORLD_SIZE` 的抽象概念，而是 DistMuon 当前通信组大小。

## 3.4 关键细节与误区澄清

> 容易误解点 7：Muon 会更新所有 requires_grad 参数？

不会。DistMuon factory 明确把 `embed_tokens`、`lm_head`、`modules_to_save.default.weight` 放进 AdamW fallback（外部 `dist_muon.py:630-634`），一维参数也走 AdamW（外部 `dist_muon.py:636-640`）。真正走 Muon 的是二维及以上、非 embedding/head 的参数。

> 容易误解点 8：`device_mesh` 一定直接传给 DistMuon 吗？

不是完整 mesh 直接传。factory 优先抽取 `device_mesh["dp_shard"]`（外部 `dist_muon.py:604-608`）。这很重要：Muon 的矩阵 shard 调度应该沿 FSDP shard 维度，而不是 TP/CP/replicate 维度。

> 容易误解点 9：DistMuon 支持任意 N-D 并行吗？

源码注释写得很直接：tensor parallelism 当前不支持，只支持 1D data parallel sharding（外部 `dist_muon.py:54`）。Axolotl validation 目前没有单独拒绝 `optimizer: muon` + `tensor_parallel_size > 1`，所以这属于运行时风险：如果参数带多个 sharded placement，DistMuon 会在 step 创建任务时抛 `NotImplementedError`（外部 `dist_muon.py:232-235`）。

## 3.5 本章小结

> 💡 **小结**
>
> * DeviceMesh 是 DistMuon 与 FSDP2 对齐的关键，标准路径使用 `dp_shard` 子 mesh。
> * Muon 只处理适合矩阵正交化的参数，embedding/head/1D 参数走 AdamW fallback。
> * TP / 多 sharded dimension 在源码中不是主路径，属于需要小心规避的边界。

# 四、一次 optimizer step：All-to-All 不是梯度同步，而是分摊矩阵正交化

## 4.1 设计哲学与核心问题

这一层解决的是“**如何在不永久 unshard 参数的情况下，对完整矩阵做 Muon 更新**”。

FSDP2 已经负责梯度同步和参数分片。DistMuon 的通信不是为了替代 FSDP2 的 gradient reduce-scatter，而是为了 Muon 自己的正交化：Newton-Schulz 需要完整矩阵语义，但完整矩阵不能长期复制在每张卡上，否则会抵消 FSDP2 的显存收益。

DistMuon 的做法是：每次 optimizer step，把同 shape / 同 sharding / 同 dtype 的参数按 `world_size` 分批，让每个 rank 临时负责一个完整矩阵的正交化，再把正交化后的 shard 发回原 rank。

## 4.2 源码入口与关键对象

```text
axolotl.contribs.mit.muon.dist_muon.py
  - DistMuon.step：按 algorithm 拆分任务，启动 AsyncRuntime
  - DistMuon._create_muon_tasks：按参数 shape/sharding/dtype 分批，识别 DTensor shard
  - muon_update_batch_async：执行 pre-orthogonalize、通信、Newton-Schulz、post-update
  - zeropower_via_newtonschulz5：bf16 Newton-Schulz 近似正交化

axolotl.contribs.mit.dion.opt_utils.py
  - create_param_batches：按 shape / placement / dtype 分组
  - pad_batch：把不足 world_size 的 batch 补到固定大小
  - AsyncTask / AsyncRuntime：在通信 wait 期间推进其他任务
```

## 4.3 主流程拆解

DistMuon 的 `step()` 先把 param group 按 algorithm 分开：

```text
DistMuon.step()
  -> group["step"] += 1
  -> muon_groups / lion_groups / adamw_groups
  -> _create_muon_tasks(...)
  -> _create_adamw_tasks(...)
  -> AsyncRuntime(all_tasks, max_concurrent_tasks=3).run()
```

源码见外部 `dist_muon.py:121-153`。当前 Axolotl factory 主要生成 `muon` 和 `adamw` group；`lion` 是 DistMuon 实现保留的备用算法分支（外部 `dist_muon.py:129-148`），Axolotl factory 并不会生成 `algorithm: lion`。

Muon task 创建阶段：

```text
_create_muon_tasks
  -> group_params = params with grad
  -> create_param_batches(group_params, batch_size=world_size)
  -> gradients = [p.grad]
  -> momentum state = zeros_like(param) lazily
  -> if DTensor:
       inspect placements
       find shard_dim
       verify process_group matches DTensor mesh group
  -> yield muon_update_batch_async(...)
```

源码依据：外部 `dist_muon.py:165-266`。状态是 lazy 初始化：`_get_or_initialize_state` 在第一次见到参数时创建 `momentum = torch.zeros_like(param)`，AdamW 还会创建 `variance`（外部 `dist_muon.py:156-163`）。如果此时 param 已被 Accelerate 切到 DTensor，`zeros_like(param)` 得到的也是与参数同 placement 的 sharded state。

对 FSDP2 sharded matrix，核心通信在 `muon_update_batch_async`：

```text
U = momentum_update(local_grad, local_momentum)

if shard_dim is not None:
    # 第一轮 all_to_all：每个 rank 收集某一个矩阵的所有 shard
    all_to_all(single_matrix_shards, U)
    single_matrix = cat(single_matrix_shards, dim=shard_dim)
    single_matrix = NewtonSchulz(single_matrix)

    # 第二轮 all_to_all：把正交化后的 shard 发回原 rank
    split(single_matrix, world_size, dim=shard_dim)
    all_to_all(U, single_matrix_shards)

post_update(local_param_shards, U)
```

源码对应外部 `dist_muon.py:341-455`，其中 sharded matrix 分支在 `dist_muon.py:370-407`。如果不是 sharded matrix，但有多个矩阵组成一个 batch，则每个 rank 正交化其中一个矩阵，再 `all_gather` 给所有 rank（外部 `dist_muon.py:408-428`）。如果 batch 长度为 1，则完全本地执行（外部 `dist_muon.py:430-437`）。

## 4.4 Shape / rank 图示

假设 FSDP2 `dp_shard` world size = 4，一个线性层权重全局形状为：

```text
W: [out, in]
FSDP2 shard over dim 0:
rank0 W_local: [out/4, in]
rank1 W_local: [out/4, in]
rank2 W_local: [out/4, in]
rank3 W_local: [out/4, in]
```

DistMuon 按 4 个同 shape 参数组成一个 batch：

```text
本地 U 列表（每张卡都有 4 个矩阵的本地 shard）:
rank0: [P0_shard0, P1_shard0, P2_shard0, P3_shard0]
rank1: [P0_shard1, P1_shard1, P2_shard1, P3_shard1]
rank2: [P0_shard2, P1_shard2, P2_shard2, P3_shard2]
rank3: [P0_shard3, P1_shard3, P2_shard3, P3_shard3]

第一次 all_to_all 后:
rank0: [P0_shard0, P0_shard1, P0_shard2, P0_shard3] -> cat -> P0_full
rank1: [P1_shard0, P1_shard1, P1_shard2, P1_shard3] -> cat -> P1_full
rank2: [P2_shard0, P2_shard1, P2_shard2, P2_shard3] -> cat -> P2_full
rank3: [P3_shard0, P3_shard1, P3_shard2, P3_shard3] -> cat -> P3_full

Newton-Schulz:
rank_i 只正交化自己负责的 P_i_full

第二次 all_to_all 后:
每张卡重新拿回所有参数的本地 shard update
```

这就是“通信换显存”的核心：完整矩阵只在某个 rank 上临时出现，而不是所有 rank 同时 all-gather 所有矩阵。

## 4.5 关键细节与误区澄清

> 容易误解点 10：DistMuon 的 all-to-all 是 FSDP2 梯度同步吗？

不是。FSDP2 梯度同步由 PyTorch FSDP2 / DTensor 负责；DistMuon 的 all-to-all 发生在 optimizer step 内，是为了 Muon 正交化临时重组完整矩阵。源码上它只操作 `G`、`M`、`U` 和 local parameter shard（外部 `dist_muon.py:363-455`）。

> 容易误解点 11：每个 rank 都会正交化所有矩阵吗？

不会。sharded matrix 分支里，每个 rank 通过 all-to-all 收到“某一个矩阵”的所有 shard，并只对这个矩阵做 Newton-Schulz（外部 `dist_muon.py:381-395`）。这避免了“所有 rank 重复做所有矩阵”的计算浪费。

> 容易误解点 12：AdamW fallback 也会触发 DistMuon 的 all-to-all 吗？

不会。AdamW fallback 走 `adamw_update_foreach_async`，输入先经过 `to_local`，然后本地 foreach 更新（外部 `dist_muon.py:301-338`；`scalar_opts.py:93-153`）。它不做 Muon 的矩阵重组通信。

## 4.6 本章小结

> 💡 **小结**
>
> * DistMuon 的核心通信是 optimizer step 内的矩阵正交化调度，不是梯度同步。
> * FSDP2 sharded matrix 路径通常是“两次 all-to-all + 一次本地 Newton-Schulz”。
> * 通过按 `world_size` 分批，每个 rank 临时负责一个完整矩阵，避免所有 rank 重复正交化全部参数。

# 五、完整主路径串联

## 5.1 完整调用栈

一次真实用户调用可以串成下面这条主路径：

```text
User: axolotl train examples/qwen2/muon-pretrain-fsdp2.yaml --num-processes 2
  │
  ├─ Step 1: CLI launch
  │     └─ src/axolotl/cli/main.py:98 train
  │        -> src/axolotl/cli/utils/train.py:157 _launch_accelerate_training
  │        -> accelerate launch -m axolotl.cli.train
  │
  ├─ Step 2: 配置加载与校验
  │     └─ src/axolotl/cli/config.py:230 load_cfg
  │        -> validate_config
  │        -> src/axolotl/utils/schemas/validation.py:906 check_muon_deepspeed_fsdp
  │        -> src/axolotl/utils/trainer.py:643 prepare_optim_env
  │
  ├─ Step 3: 模型加载与 FSDP2 patch
  │     └─ src/axolotl/train.py:522 setup_model_and_trainer
  │        -> src/axolotl/loaders/model.py:161 ModelLoader.load
  │        -> src/axolotl/loaders/patch_manager.py:270 _apply_fsdp_patches
  │        -> src/axolotl/monkeypatch/accelerate/fsdp2.py:529 patch_accelerate_fsdp2
  │
  ├─ Step 4: Trainer 构建与 optimizer factory 注入
  │     └─ src/axolotl/utils/trainer.py:679 setup_trainer
  │        -> src/axolotl/core/builders/causal.py:163 build
  │        -> src/axolotl/core/builders/base.py:277 _configure_optimizer
  │        -> trainer_kwargs["optimizer_cls_and_kwargs"] = (DistMuonOptimizerFactory, kwargs)
  │
  ├─ Step 5: 训练准备
  │     └─ src/axolotl/train.py:183 execute_training
  │        -> trainer.train(...)
  │        -> transformers.Trainer._prepare_for_training
  │        -> create_optimizer()
  │        -> accelerator.prepare(model, optimizer)
  │        -> FSDP2 wrap + optimizer param switch
  │
  ├─ Step 6: 每个 optimizer step
  │     └─ DistMuon.step
  │        -> _create_muon_tasks
  │        -> muon_update_batch_async
  │        -> all_to_all / all_gather + Newton-Schulz + local update
  │
  └─ Step 7: 保存
        └─ src/axolotl/train.py:257 save_trained_model
           -> fsdp_plugin.set_state_dict_type(...)
           -> trainer.save_model(...)
           -> patched Accelerator.get_state_dict when needed
```

## 5.2 每一层做了什么

| 层次 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| CLI launch | YAML 路径、launcher 参数 | 多进程 worker | launcher 初始化 | 无直接影响 | 一次 |
| `load_cfg` | YAML + CLI overrides | `cfg`、FSDP env、parallelism env | 无 | 决定后续 FSDP/mesh | 一次 |
| ModelLoader patch | `cfg.fsdp_config` | 全局替换 Accelerate FSDP2 函数 | 无 | 影响加载 / wrap 峰值 | 一次 |
| `build_parallelism_config` | world/tp/cp/dp 配置 | `ParallelismConfig`、DeviceMesh | 构建 mesh 本身无显式 collective | 决定 sharding 维度 | 初始化 |
| builder optimizer | `optimizer: muon`、DeviceMesh | `DistMuonOptimizerFactory` 注入 trainer | 无 | 无直接影响 | 初始化 |
| `accelerator.prepare` | model + optimizer | 模型参数变 DTensor，optimizer param group 切到新参数 | FSDP2 初始化可能通信 | 参数/状态进入 sharded 语义 | 训练前一次 |
| `DistMuon.step` | local grad/state shard | local param shard 更新 | all-to-all / all-gather | 临时 full matrix buffer | 每 optimizer step |
| 保存 | FSDP state dict type | full/sharded model weights | full_tensor / barrier 等 | rank0 CPU 聚合或 sharded 输出 | checkpoint / end |

## 5.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `axolotl.contribs.mit.muon.muon.Muon` | 名字也是 Muon | FSDP2 多卡 DeviceMesh 路径不是主实现 | 单卡 / 无 DeviceMesh fallback；DistMuon 在 FSDP2 多卡主路径中接管。 |
| `DistMuon._create_lion_tasks` | DistMuon step 里有 lion 分支 | Axolotl factory 不生成 `algorithm: lion` | 这是外部实现保留能力，不是当前 `optimizer: muon` 主路径。 |
| `process_pretraining_datasets_for_packing` | 目标是 pretraining，容易以为 optimizer 依赖它 | Optimizer 不依赖 | 它只处理 streaming pretraining sample packing 的 position_ids/attention_mask（`src/axolotl/utils/trainer.py:386-405`）。 |
| `DistributedParallelMixin._save` | 名字像 FSDP 保存主路径 | 对 `AxolotlTrainer` 未必是最终覆盖点 | `AxolotlTrainer` 自身定义 `_save`（`src/axolotl/core/trainers/base.py:805-861`）；最终保存从 `train.py:294-349` 进入。 |
| `fsdp2_load_full_state_dict` | 名字像每次训练都加载 | 只在 `cpu_ram_efficient_loading` 时触发 | 在 `fsdp2_prepare_model` 里受 `fsdp2_plugin.cpu_ram_efficient_loading` 控制（`src/axolotl/monkeypatch/accelerate/fsdp2.py:422-425`）。 |
| `docs/optimizers.qmd` 的 `optimizer: muon` | 看起来是完整配置 | 只是 optimizer 最小文档 | FSDP2 pretraining 还需要 fsdp_config、launcher、数据配置等。 |

> 💡 **小结**
>
> * 主路径的关键转折点是 builder 选择 DistMuon factory，以及 Accelerate.prepare 切换 optimizer 参数。
> * DistMuon 不参与 forward/backward；它只在 optimizer step 内参与更新。
> * 很多看似相关的 pretraining / saving / fallback 函数只服务局部场景，不能误认为主链路。

# 六、关键数据流 / 状态流 / shape 流程

## 6.1 Tensor shape 变化

以一个 FSDP2 sharded linear weight 为例：

```text
全局参数:
  W: [out, in]
  grad: [out, in]
  momentum: [out, in]

FSDP2 local shard（假设 shard_dim=0, dp_shard_size=4）:
  W_local: [out/4, in]
  G_local: [out/4, in]
  M_local: [out/4, in]

DistMuon pre-orthogonalize:
  U_local = momentum * mu + grad
  U_local dtype -> bf16
  U_local: [out/4, in]

第一次 all_to_all + cat:
  rank_i single_matrix: [out, in]

Newton-Schulz:
  single_matrix_orthogonalized: [out, in]

split + 第二次 all_to_all:
  U_local_orthogonalized: [out/4, in]

post-update:
  W_local -= adjusted_lr * U_local_orthogonalized
```

关键源码：pre-orthogonalize 在外部 `dist_muon.py:488-510`；Newton-Schulz wrapper 在 `dist_muon.py:528-541`；具体 Newton-Schulz 在 `dist_muon.py:567-591`；post-update 在 `dist_muon.py:513-525`。

真正节省显存的是 FSDP2 让参数、梯度和 optimizer state 平时保持 shard；DistMuon 只在某个 rank 上临时拼一个矩阵。性能瓶颈也在这里：大矩阵会在 step 阶段产生临时 full matrix buffer，并触发两次 all-to-all。

## 6.2 Rank / Mesh / Process Group 变化

标准 8 卡 FSDP2，无 TP/CP 时，Axolotl 的 `_get_parallel_config_kwargs` 会把剩余 world size 变成 `dp_shard_size=8`（`src/axolotl/utils/distributed.py:338-362`）：

```text
world_size = 8
fsdp_config = true

dp_shard mesh:
  [rank0, rank1, rank2, rank3, rank4, rank5, rank6, rank7]

DistMuon:
  distributed_mesh = device_mesh["dp_shard"]
  process_group = distributed_mesh.get_group()
  world_size = distributed_mesh.size()
```

如果是 HSDP，例如：

```text
world_size = 8
dp_replicate_size = 2
dp_shard_size = 4

replica 0 dp_shard group: rank0 rank1 rank2 rank3
replica 1 dp_shard group: rank4 rank5 rank6 rank7
```

DistMuon factory 仍会抽取 `dp_shard` 子 mesh。基于源码行为推断，它只在各自 shard group 内做 Muon 通信；replicate 维度上的梯度同步仍属于 FSDP2 / PyTorch 下游职责。这里的 HSDP 多维行为未在 DistMuon 专门测试中确认。

## 6.3 状态切换

这个特性涉及三类状态：

```text
环境变量状态:
  prepare_optim_env 写入 ACCELERATE_USE_FSDP / FSDP_VERSION / PARALLELISM_CONFIG_*
  下游 Accelerate / Transformers 初始化时读取

全局 monkey patch 状态:
  patch_parallelism_config 替换 ParallelismConfig._validate_accelerator
  patch_accelerate_fsdp2 替换 accelerate.accelerator.fsdp2_prepare_model 与 Accelerator.get_state_dict

optimizer 内部状态:
  DistMuon.state[param]["momentum"] = zeros_like(param)
  AdamW fallback 还保存 variance
  param_group["step"] 每次 step += 1
```

线程安全 / 进程安全上要区分：环境变量和 monkey patch 都是**进程内全局状态**；在 accelerate 多进程训练里每个 worker 进程各自设置，通常不会跨进程污染。但在同一 Python 进程连续跑多个配置时，monkey patch 不会自动恢复，这是测试和 notebook 场景需要注意的维护成本。

> 💡 **小结**
>
> * Shape 主线是 local shard -> 临时 full matrix -> local shard，而不是永久 all-gather。
> * DistMuon 的 process group 来自 FSDP `dp_shard` mesh；这决定 optimizer step 的通信边界。
> * 环境变量、monkey patch、optimizer state 是三类不同状态，生命周期完全不同。

# 七、核心机制深挖

## 7.1 Monkey Patch：零侵入接入还是维护风险？

Axolotl 没有 fork Transformers / Accelerate，而是在模型加载阶段 patch 下游函数：

- `patch_parallelism_config()`：替换 `ParallelismConfig._validate_accelerator`，并给 `AcceleratorState.is_fsdp2` 加上空 fsdp_plugin guard（`src/axolotl/monkeypatch/accelerate/parallelism_config.py:61-77`）。
- `patch_accelerate_fsdp2()`：替换 `accelerate.accelerator.fsdp2_prepare_model` 和 `accelerate.Accelerator.get_state_dict`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:529-533`）。
- `patch_initialize_missing_keys_for_fsdp()`：patch Transformers `PreTrainedModel._initialize_missing_keys`，避免非 rank0 meta loading 时重复初始化所有参数（`src/axolotl/monkeypatch/accelerate/fsdp2.py:489-527`）。

它解决的问题是兼容性：FSDP2、rank0-only loading、ParallelismConfig 和 PEFT/LoRA 组合变化很快，完全等上游修复会阻塞 Axolotl 功能。

但代价也很明确：

- patch 是全局的，没有恢复函数；
- patch 依赖下游函数名和模块变量名；
- `patch_accelerate_fsdp2` 本身没有 idempotence guard，重复赋值问题不大，但也说明它不是严格的 patch manager registry；
- 一旦 Accelerate 内部 prepare 流程变化，Axolotl 的替换函数需要同步维护。

## 7.2 通信原语：前向和反向并不对称

DistMuon 没有自定义 autograd Function，也不改 forward/backward。它的通信只发生在 optimizer step：

| 场景 | 通信 | 代码位置 | 语义 |
|---|---|---|---|
| FSDP2 matrix-sharded 参数 | 两次 `dist.all_to_all` | 外部 `dist_muon.py:383-407` | 第一次聚合完整矩阵，第二次把正交化 shard 发回。 |
| 非 matrix-sharded 但 batch 大于 1 | 一次 `dist.all_gather` | 外部 `dist_muon.py:424-428` | 每个 rank 计算一个矩阵更新，再同步给所有 rank。 |
| AdamW fallback | 无 Muon collective | 外部 `dist_muon.py:301-338` | 本地 foreach 更新 local tensor。 |
| FSDP2 模型本身 | 下游 FSDP collectives | `fully_shard(..., reshard_after_forward=...)` 在 `fsdp2.py:351-415` | 参数 all-gather / reshard / gradient sync 由 PyTorch FSDP2 负责。 |

因此不能把 DistMuon 的通信理解成“反向传播通信”。它不负责梯度平均，也没有梯度缩放逻辑。它默认输入的 `p.grad` 已经符合 FSDP2 sharded 参数应有的梯度语义。

## 7.3 配置归一化：字段如何变成真实行为

几个配置字段有不同层次的消费方式：

- `optimizer: muon`：被 Axolotl builder 直接消费，决定 factory（`src/axolotl/core/builders/base.py:299-316`）。
- `fsdp_version: 2` / `fsdp_config.fsdp_version`：validation 归一化并校验；`check_fsdp_version_in_fsdp_config` 会在顶层和嵌套字段之间同步（`src/axolotl/utils/schemas/validation.py:1072-1085`）。
- `fsdp_config` 中带 `fsdp_` 前缀的旧字段：会被去前缀迁移（`src/axolotl/utils/schemas/validation.py:1050-1070`）。示例里仍使用 `fsdp_state_dict_type` 这类旧写法（`examples/qwen2/muon-pretrain-fsdp2.yaml:61-68`）。
- `state_dict_type`：写到 `FSDP_STATE_DICT_TYPE` 环境变量（`src/axolotl/utils/trainer.py:605-606`），保存阶段又通过 `fsdp_plugin.set_state_dict_type` 显式切换（`src/axolotl/train.py:294-301`）。
- `dp_shard_size` / `dp_replicate_size` / `tensor_parallel_size` / `context_parallel_size`：既可能写入 `PARALLELISM_CONFIG_*` 环境变量，也会在 `build_parallelism_config` 中构建 DeviceMesh（`src/axolotl/utils/trainer.py:621-640`；`src/axolotl/utils/distributed.py:299-370`）。

> 💡 **小结**
>
> * Axolotl 用 monkey patch 换来了较低接入成本，但维护风险集中在下游 API 变化。
> * DistMuon 的通信与 forward/backward 不对称；它只服务 optimizer update。
> * 配置字段有的由 Axolotl 直接消费，有的只是写入环境变量或交给下游库。

# 八、显存、性能与通信分析

## 8.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---|---|
| 参数 | ✅ | FSDP2 `fully_shard` 后参数以 DTensor shard 存放；DistMuon step 后只更新 local shard。 |
| 梯度 | ✅ | FSDP2 梯度语义是 sharded 参数对应的 local grad；DistMuon 不额外保存 full grad。 |
| Muon momentum state | ✅ | `zeros_like(param)` 在 param 已切到 DTensor 后创建，状态随 shard 存储（外部 `dist_muon.py:156-163`）。 |
| AdamW variance state | ✅ | AdamW fallback 的 variance 也是 lazy `zeros_like(param)`。 |
| 激活值 | ❌ / 取决于 FSDP2 AC | Muon 不影响 forward 激活；激活节省来自 FSDP activation checkpointing 或其他机制。 |
| logits | ❌ | Muon 不改 loss / logits 计算。 |
| optimizer 临时 full matrix buffer | ❌ | sharded matrix 分支会在某个 rank 临时 cat 出 `[out, in]` 完整矩阵（外部 `dist_muon.py:381-390`）。 |
| 保存时 full state dict | ❌ | `get_state_dict` 的 FSDP2 分支会对 DTensor 调 `full_tensor()`，rank0 收 CPU state dict，并每个参数 barrier（`src/axolotl/monkeypatch/accelerate/fsdp2.py:158-173`）。 |

真正的显存收益来自 FSDP2 参数/梯度/optimizer state 的常驻分片；DistMuon 的贡献是没有把 Muon 状态设计成全量 replicated state。但具体显存节省比例未在源码中确认，需要通过 benchmark 才能量化。它必须为了正交化临时 materialize full matrix，因此大矩阵层会出现 step 内峰值。

## 8.2 通信开销

每个 optimizer step 的新增通信可以分层看：

```text
FSDP2 模型训练通信（下游 PyTorch）:
  - forward 前参数 materialize / all-gather（由 fully_shard 语义决定）
  - backward 梯度同步 / reshard
  - 受 reshard_after_forward 影响

DistMuon 额外 optimizer 通信:
  - 对每个 matrix-sharded Muon batch：2 次 all_to_all
  - 对非 sharded 但 distributed batch：1 次 all_gather
  - AdamW fallback：无额外 Muon collective

保存通信:
  - FSDP2 get_state_dict 中 DTensor.full_tensor()
  - 每个参数后 barrier
```

`AsyncRuntime(max_concurrent_tasks=3)` 试图在异步 collective 等待期间推进其他任务（外部 `dist_muon.py:150-152`；`opt_utils.py:83-149`）。但这不是完整的通信/计算 overlap 框架：每个 `muon_update_batch_async` 内部仍有明确 `work.wait()`，并且 Newton-Schulz 本身在单 rank 上执行。

## 8.3 性能取舍

DistMuon 的取舍可以概括为：

- 用 optimizer step 阶段的通信换取 FSDP2 sharded state 的显存收益；
- 用每个 rank 负责一个矩阵的调度，避免所有 rank 重复正交化所有矩阵；
- 用全局 monkey patch 换取与当前 Accelerate / Transformers FSDP2 的兼容；
- 用 AdamW fallback 保守处理 embedding/head/1D 参数，避免 Muon 泛化到不适合的参数形态。

性能瓶颈也很清楚：

1. **大矩阵正交化是单 rank 临时 full matrix 计算。** 模型越大，单层投影矩阵越大，rank 内峰值和 matmul 开销越明显。
2. **all-to-all 是 step 级额外通信。** 与 AdamW 相比，Muon 多了 optimizer 侧 collectives。
3. **保存 full state dict 可能成为 rank0 CPU / I/O 瓶颈。** FSDP2 `get_state_dict` 会逐参数 full_tensor，并在每个参数后 barrier（`src/axolotl/monkeypatch/accelerate/fsdp2.py:158-173`）。

> 💡 **小结**
>
> * DistMuon 不节省激活和 logits；它主要保持 optimizer state 与参数 shard 对齐。
> * 新增通信集中在 optimizer step，尤其是 sharded matrix 的两次 all-to-all。
> * 保存路径可能比训练 step 更容易出现 rank0 聚合瓶颈。

# 九、配置项、边界条件与坑点

## 9.1 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `optimizer: muon` | `src/axolotl/core/builders/base.py:299-316` | 进入 Muon custom optimizer factory | 没有 DeviceMesh 时是本地 Muon，不是 DistMuon。 |
| `fsdp_config` + `fsdp_version: 2` | `validation.py:906-918`、`patch_manager.py:287-299` | 允许 Muon + FSDP，并注册 FSDP2 patch | Torch < 2.7 会被 FSDP2 校验拒绝（`config.py:1731-1734`）。 |
| `deepspeed` | `validation.py:907-911` | Muon 直接报错 | 当前不支持 Muon + DeepSpeed。 |
| `fsdp_version: 1` | `validation.py:912-917` | Muon + FSDP1 报错 | 必须迁移 FSDP2。 |
| `fsdp_config.state_dict_type` | `utils/trainer.py:605-606`、`train.py:294-302` | 控制保存 full / sharded state dict | SHARDED final path需要 merge；FULL 可能 rank0 聚合重。 |
| `fsdp_config.cpu_ram_efficient_loading` | `loaders/model.py:756-779`、`fsdp2.py:371-425` | rank0 CPU / 非 rank meta loading，并 broadcast full state | 节省 CPU RAM，但 load 阶段有 broadcast 和 meta patch 复杂度。 |
| `dp_shard_size` / `dp_replicate_size` | `utils/distributed.py:319-370` | 改变 DeviceMesh 与 DistMuon process group | HSDP 未在 DistMuon 测试中充分覆盖。 |
| `tensor_parallel_size > 1` | `utils/distributed.py:330-332`、外部 `dist_muon.py:54` | 可能构建 TP mesh | DistMuon 注释称 TP 不支持；Axolotl validation 未显式拒绝。 |
| `optim_args` | `base.py:389-397` | 透传额外 optimizer kwargs | 字符串解析很薄：`key=value` 不做类型强转，可能把 bool/float 传成字符串。 |
| `pretraining_dataset` | `causal.py:326`、`utils/data/sft.py:63-64` | 进入 pretraining/streaming 数据路径 | 不决定是否 DistMuon；只是目标场景数据路径。 |

## 9.2 硬约束与静默失效条件

- **FSDP2 需要 torch >= 2.7.0。** `check_fsdp_torch_version` 在 `src/axolotl/utils/schemas/config.py:1720-1735` 检查。
- **Muon + DeepSpeed 不支持。** validation 直接抛错（`src/axolotl/utils/schemas/validation.py:907-911`）。
- **Muon + FSDP1 不支持。** validation 要求 FSDP 场景下 `fsdp_version == 2`（`validation.py:912-917`）。
- **DistMuon 只支持一个 sharded matrix dimension。** 多个 sharded dimensions 会 `NotImplementedError`（外部 `dist_muon.py:232-235`）。
- **shard 维度必须能被 world size 整除。** `muon_update_batch_async` 断言 `X[0].size(shard_dim) % world_size == 0`（外部 `dist_muon.py:370-379`）。
- **DeviceMesh 必须和 DTensor placement 的 group 匹配。** 否则 runtime error（外部 `dist_muon.py:237-244`）。
- **optimizer state 保存 / resume 有缺口。** `CheckpointSaveMixin` 捕获 FSDP2 optimizer saving 的 `NotImplementedError` / `KeyError` 并警告无法 resume optimizer/scheduler（`src/axolotl/core/trainers/mixins/checkpoints.py:13-22`）。

> 💡 **小结**
>
> * 最小 FSDP2 DistMuon 配置不是只有 `optimizer: muon`，还需要 FSDP2 与多进程 launcher。
> * 多维并行、TP、非整除 shard 维度是主要风险区。
> * `optim_args` 是透传口，灵活但缺少强类型保护。

# 十、测试、示例与覆盖缺口

## 10.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `examples/qwen2/muon-pretrain-fsdp2.yaml` | 推荐配置 | 覆盖 pretraining dataset、Muon、FSDP2、FULL_STATE_DICT 的组合意图。 |
| `tests/utils/schemas/validation/test_fsdp.py:139-148` | Muon + FSDP1 rejected | 验证 FSDP1 组合会报错。 |
| `tests/test_validation_dataset.py:331-368` | Muon + DeepSpeed / FSDP1 rejected | 覆盖用户常见错误配置。 |
| `tests/core/test_builders.py:542-578` | 无 FSDP 时 builder 使用本地 `MuonOptimizerFactory` | 证明单卡 / 无 DeviceMesh 分支存在。 |
| `tests/e2e/test_optimizers.py:117-161` | 单卡 Muon 训练 smoke | 训练 5 step，检查输出和 optimizer 类名。 |
| `tests/e2e/multigpu/test_dist_muon_fsdp2.py:51-107` | 2 进程 FSDP2 + Muon FFT/SFT | 执行 `axolotl train --num-processes 2`，检查模型文件、checkpoint、loss 非 NaN。 |
| `tests/e2e/multigpu/test_dist_muon_fsdp2.py:108-168` | 2 进程 FSDP2 + Muon + LoRA SFT | 覆盖 LoRA 组合 smoke。 |
| `tests/e2e/utils.py:81-90` | torch >= 2.7 gate | 多卡 DistMuon 测试会在 torch 版本不足时 skip。 |

## 10.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| 真正 `pretraining_dataset` + DistMuon e2e | 示例有，未看到专门 e2e | README 主推场景没有独立断言收敛/保存语义。 |
| 多机训练 | 未看到 DistMuon 专项多机测试 | RDMA/NCCL all-to-all 拓扑问题可能只在多机暴露。 |
| HSDP `dp_replicate + dp_shard` | 未看到 DistMuon 专项测试 | DistMuon 只在 dp_shard 组通信，replicate 语义依赖 FSDP2 下游，需验证。 |
| TP + Muon | 未看到拒绝或测试 | DistMuon 注释称 TP 不支持，可能 runtime 报错或语义不正确。 |
| SHARDED_STATE_DICT final save + DistMuon | e2e 使用 FULL_STATE_DICT | sharded merge / adapter rename 路径未专项覆盖。 |
| optimizer state resume | 源码 TODO 明确缺口 | checkpoint 可保存模型，但 optimizer/scheduler resume 可能不可用。 |
| 显存 / 性能收益 | 未看到 benchmark assert | 无法从测试证明 Muon 通信换显存的收益边界；具体收益未在源码中确认。 |
| patch 是否恢复 | 未看到恢复测试 | 同进程多配置/测试污染风险。 |
| 非整除 shard_dim | 未看到异常配置测试 | 训练中途 assertion，用户体验较差。 |

> 💡 **小结**
>
> * 当前测试更像 smoke + validation，证明“能跑”和“错误组合能拒绝”。
> * 主推的 pretraining 示例存在，但 DistMuon 专项 e2e 覆盖的是 SFT / LoRA SFT。
> * 多机、HSDP、TP、resume、性能收益仍是主要覆盖缺口。

# 十一、局限性与已知优化点

## 11.1 硬约束

源码层面可以确认的硬约束包括：

- Muon 与 DeepSpeed 不兼容（`validation.py:907-911`）。
- FSDP 场景只能用 FSDP2（`validation.py:912-917`）。
- FSDP2 要求 torch >= 2.7.0（`config.py:1731-1734`）。
- DistMuon 只接受 1D DeviceMesh；多维 mesh 要先抽取 1D `dp_shard` 子 mesh（外部 `dist_muon.py:96-104`、`dist_muon.py:602-610`）。
- 多个 sharded dimensions 不支持（外部 `dist_muon.py:232-235`）。
- tensor parallelism 在 DistMuon docstring 中明确“不当前支持”（外部 `dist_muon.py:54`）。

## 11.2 维护成本

- **外部 contrib 包版本耦合。** Axolotl 主仓库只 pin `axolotl-contribs-mit==0.0.6`（`pyproject.toml:70-72`），核心 DistMuon 源码不在主仓库中。升级 contrib 包可能改变 optimizer 语义。
- **下游 FSDP2 prepare 耦合。** Axolotl patch 直接替换 Accelerate 模块变量（`fsdp2.py:529-533`），依赖下游 `Accelerator.prepare` 仍通过该变量调用 prepare。
- **全局 patch 无恢复。** 长生命周期进程和测试套件可能受污染。
- **配置经过多层转换。** `fsdp_` 前缀迁移、顶层 `fsdp_version` 与 nested `fsdp_config.fsdp_version` 同步、环境变量写入、builder 再重建 DeviceMesh，使问题定位需要跨多个模块。

## 11.3 性能瓶颈

- **step 内 all-to-all 可能成为瓶颈。** 每个 matrix-sharded batch 两次 all-to-all，且 batch 粒度受同 shape 参数数量限制。
- **Newton-Schulz 是矩阵级计算。** 每个 rank 临时处理完整矩阵，单 rank 峰值与计算量随最大矩阵增长。
- **保存路径每参数 barrier。** FSDP2 `get_state_dict` 在遍历参数时对 rank0 CPU state dict 写入后执行 barrier（`src/axolotl/monkeypatch/accelerate/fsdp2.py:168-173`），大模型保存可能串行化明显。
- **optimizer state resume 未完善。** `CheckpointSaveMixin` 的 TODO 表明 FSDP2 optimizer saving 还不是完整闭环（`src/axolotl/core/trainers/mixins/checkpoints.py:13-22`）。

## 11.4 已知优化点

源码中已经透露出几个可优化方向：

1. **FSDP2 optimizer saving。** `# TODO: fix fsdp2 optimizer saving` 是最直接的工程缺口（`checkpoints.py:17`）。
2. **减少保存 barrier / 分块 gather。** 当前 FSDP2 `get_state_dict` 逐参数 `full_tensor()` + barrier，可考虑更粗粒度 coalescing 或 sharded final 默认路径。
3. **更强的配置 validation。** 例如提前拒绝 Muon + TP、多 sharded dim、非整除 shard dim，而不是等 optimizer step 报错。
4. **更细粒度 overlap。** 当前 `AsyncRuntime` 最多并发 3 个任务，但每个任务内部通信 wait 边界仍明显；后续可以探索更系统的 compute/comm overlap。
5. **文档补齐。** `docs/optimizers.qmd` 只写了 `optimizer: muon`，没有解释 FSDP2 / DistMuon 条件、TP 限制和 resume 限制。

> 💡 **小结**
>
> * 当前实现适合 1D FSDP2 data-shard 预训练 / 全参训练主线。
> * 多维并行和完整 checkpoint resume 是最需要谨慎的区域。
> * 未来优化重点不在“能不能接入”，而在保存、validation、通信 overlap 与文档化。

# 小结与展望

Axolotl 的 Distributed Muon Optimizer for FSDP2 pretraining 实现，可以用四个关键词概括。

## 关键词一：配置驱动的工厂切换

用户只写 `optimizer: muon`，但 builder 会根据 DeviceMesh 决定使用本地 Muon 还是 DistMuon（`src/axolotl/core/builders/base.py:299-316`）。这让单卡和多卡共享一个配置入口，也带来一个需要澄清的点：`optimizer: muon` 不等于一定启用分布式 Muon。

## 关键词二：FSDP2 参数引用切换

DistMuon 能更新 DTensor，不是因为它自己 wrap 模型，而是因为 Transformers / Accelerate 的 FSDP2 主路径要求 model 和 optimizer 一起 `prepare()`，并在 wrap 后切换 optimizer 参数引用。Axolotl 的 monkey patch 改写了 FSDP2 prepare 细节，但 optimizer param switch 仍依赖 Accelerate 主路径。

## 关键词三：All-to-All 分摊正交化

DistMuon 的核心不是梯度同步，而是矩阵正交化调度。对 FSDP2 sharded matrix，它用两次 all-to-all 把“每个 rank 拥有所有矩阵的一块”转换成“每个 rank 临时拥有一个完整矩阵”，做完 Newton-Schulz 再切回 shard。这是该特性最有工程味的部分：用通信和临时 buffer 换常驻显存下降。

## 关键词四：保守 fallback 与明确边界

Embedding、lm_head、一维参数走 AdamW fallback；DeepSpeed / FSDP1 被 validation 拒绝；TP 和多 sharded dimension 在源码中不是主路径。这个实现适合标准 FSDP2 预训练 / 全参训练，尤其是希望尝试 Muon 但不能放弃 FSDP2 sharding 的场景。不适合依赖 DeepSpeed、复杂 TP/HSDP 组合、强 resume 需求或对 step 通信极端敏感的训练。

与替代方案相比，DistMuon 没有像低秩优化器那样彻底减少矩阵正交化通信，也没有完全隐藏 FSDP2 保存复杂度；它的价值在于以相对小的 Axolotl 接入面，把一个矩阵型 optimizer 接进现有 FSDP2 Trainer 主链路。后续值得继续走读的方向，是 `Dion` 这类更强调分布式正交化通信效率的 optimizer、Axolotl 的 ND parallelism 与 TP/CP/FSDP 组合，以及 FSDP2 checkpoint / optimizer state resume 的完整闭环。

