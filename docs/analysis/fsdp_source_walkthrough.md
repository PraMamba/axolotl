# Axolotl 源码走读：Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes 实现解析

在大模型微调里，最朴素的数据并行会很快撞上显存墙：每张卡都要完整保存参数、梯度和优化器状态。对于 Axolotl 这样的“用一份 YAML 组织训练”的框架来说，FSDP 的价值不只是“省显存”，还在于把 PyTorch / Accelerate / Transformers / PEFT / bitsandbytes 这些组件的分布式语义接到同一条训练链路上。

本文不复述 FSDP 论文和 PyTorch 原理，而是沿着 Axolotl 的真实源码路径，看它如何让用户在单机多卡和多机多卡上开启 FSDP / HSDP，配置如何变成环境变量和 `TrainingArguments`，模型何时被 shard，rank0-only loading 为什么需要 patch，保存阶段又为什么可能成为新的瓶颈。

# 前言

## 业务 / 工程背景

Axolotl 的定位是 LLM fine-tuning 框架：用户主要通过 `axolotl train config.yml` 启动训练。FSDP 出现在以下场景：

- 全参微调 7B+ / 30B+ 模型，单卡无法容纳参数、梯度和 optimizer state；
- LoRA / QLoRA 下仍希望把大模型底座或 adapter 相关状态分散到多张卡；
- 多节点训练时，希望节点内做 FSDP、节点间做复制式数据并行，即 HSDP；
- 保存 / resume / merge sharded checkpoint 时，需要把 PyTorch 分布式 checkpoint 和 HuggingFace `save_pretrained()` 生态接起来。

## 核心矛盾

FSDP 的核心矛盾可以概括为三句话：

1. **显存收益来自参数生命周期缩短**：参数、梯度、优化器状态被分片，但计算某层时又必须临时 all-gather 出完整权重。
2. **初始化越省内存，保存 / 广播越复杂**：`cpu_ram_efficient_loading` 让非 rank0 使用 meta tensor，但必须保证后续广播、tied weights、missing keys 初始化都正确。
3. **Axolotl 自己不重写 FSDP 算法**：它主要负责配置归一化、拓扑构建、patch 注入和保存适配，真正的逐层通信由 PyTorch FSDP2 / Accelerate 接管。

## 本文主线

本文按机制而不是按文件展开：

1. 配置如何从 YAML 进入 Axolotl，并在第一个行为改变点变成 FSDP 环境变量；
2. 单机 FSDP 与多机 HSDP 如何由 `dp_shard_size` / `dp_replicate_size` 映射到 DeviceMesh；
3. 模型加载、patch、`fully_shard()` 的顺序为什么不能随便换；
4. 训练主路径中 FSDP 改变的是参数生命周期，而不是 batch shape；
5. 保存 / checkpoint 为什么是 FSDP 另一个高风险路径；
6. 最后分析配置坑点、测试覆盖、显存与通信取舍。

## 不展开的内容

本文不讲 FSDP 数学原理、不讲 ZeRO 论文、不讲 LoRA / QLoRA 基础、不讲 Accelerate 完整设计。涉及这些外部理论时，只解释 Axolotl 源码如何接入。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/main.py` | 用户 CLI 入口，负责选择 `accelerate` / `torchrun` / `python` launcher。 |
| `src/axolotl/cli/utils/train.py` | 组装真实启动命令，单机和多机 torchrun 参数从这里传入。 |
| `src/axolotl/cli/config.py` | 读取 YAML、合并 CLI override、校验配置、准备优化器 / FSDP 环境。 |
| `src/axolotl/utils/trainer.py` | `setup_fsdp_envs()` / `setup_parallelism_envs()` 把配置写入环境变量。 |
| `src/axolotl/utils/distributed.py` | 构建 `ParallelismConfig` / DeviceMesh，处理 world size 与并行维度关系。 |
| `src/axolotl/loaders/model.py` | 模型加载主类，决定 device_map、cpu/meta loading、TP mesh、量化加载策略。 |
| `src/axolotl/loaders/patch_manager.py` | 模型加载前后统一注入 FSDP2 / QLoRA / Accelerate patch。 |
| `src/axolotl/monkeypatch/accelerate/fsdp2.py` | 替换 Accelerate 的 FSDP2 prepare / get_state_dict，并实现 rank0 broadcast loading。 |
| `src/axolotl/monkeypatch/fsdp2_qlora.py` | 修补 FSDP2 与 bitsandbytes Params4bit / Int8Params 的 shard/unshard 兼容。 |
| `src/axolotl/train.py` | 串联 setup、train、final save，并处理 sharded checkpoint merge。 |

# 一、配置入口与归一化：FSDP 首先是“把用户意图翻译给下游库”

## 1.1 设计哲学与核心问题

用户看到的是一段 YAML：

```yaml
fsdp_version: 2
fsdp_config:
  offload_params: false
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  state_dict_type: FULL_STATE_DICT
  reshard_after_forward: true
```

但 FSDP 真正生效时，参与者至少有三层：

- Transformers `TrainingArguments` 需要知道 `fsdp` / `fsdp_config`；
- Accelerate 需要通过环境变量或 `AcceleratorState` 初始化 FSDP plugin；
- PyTorch FSDP2 需要在 `Accelerator.prepare()` 阶段对模型调用 `fully_shard()`。

所以 Axolotl 的第一层工作不是 shard 参数，而是**把用户配置拆成多个下游系统能理解的状态**。如果没有这一层，FSDP 可能在 YAML 中“看起来开启了”，但 Accelerate 没有收到 `ACCELERATE_USE_FSDP`，或者 Transformers 收不到 `fsdp_config`，最终训练仍走普通 DDP / 单卡路径。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：用户命令入口，接收 config 与 launcher。

src/axolotl/cli/utils/train.py
  - launch_training：根据 launcher 分发到 accelerate / torchrun / python。
  - _launch_accelerate_training：生成 accelerate launch -m axolotl.cli.train。
  - _launch_torchrun_training：生成 torchrun -m axolotl.cli.train。

src/axolotl/cli/train.py
  - do_cli：load_cfg 后进入 do_train。
  - do_train：加载数据集并调用 axolotl.train.train。

src/axolotl/cli/config.py
  - load_cfg：读取 YAML、合并 CLI 参数、validate_config、prepare_optim_env、normalize_config。

src/axolotl/utils/trainer.py
  - prepare_optim_env：决定 FSDP / DeepSpeed / parallelism env。
  - setup_fsdp_envs：第一个真正改变 FSDP 行为的函数。
```

## 1.3 主流程拆解

用户入口在 `src/axolotl/cli/main.py:78-128`。`train()` 不直接训练，而是把 config 和 launcher 传给 `launch_training()`：

```text
axolotl train config.yml
  -> src/axolotl/cli/main.py:98 train(...)
    -> src/axolotl/cli/utils/train.py:109 launch_training(...)
      -> accelerate launch -m axolotl.cli.train config.yml
         或 torchrun -m axolotl.cli.train config.yml
```

`accelerate` 路径在 `src/axolotl/cli/utils/train.py:157-192`：它拼出 `accelerate launch ... -m axolotl.cli.train`。多节点 `torchrun` 路径在 `src/axolotl/cli/utils/train.py:195-218`，并且 `_add_default_rdzv_args()` 会在用户给了 `--rdzv_endpoint` 但没给 backend / id 时补默认值（`src/axolotl/cli/utils/train.py:15-43`）。

真正读取 YAML 的地方是 `load_cfg()`：

```text
src/axolotl/cli/train.py:55 do_cli
  -> src/axolotl/cli/config.py:230 load_cfg
    -> validate_config(...)
    -> prepare_optim_env(cfg)
    -> normalize_config(cfg)
```

关键顺序在 `src/axolotl/cli/config.py:300-328`：

- `validate_config()` 在 `src/axolotl/cli/config.py:308-320`；
- `prepare_optim_env(cfg)` 在 `src/axolotl/cli/config.py:326`；
- `normalize_config(cfg)` 在 `src/axolotl/cli/config.py:327`。

第一个真正改变 FSDP 行为的是 `setup_fsdp_envs()`：

```python
# src/axolotl/utils/trainer.py:589-618，简化
os.environ["ACCELERATE_USE_FSDP"] = "true"
if str(cfg.fsdp_version) == "2":
    os.environ["FSDP_VERSION"] = "2"
if cfg.fsdp_config.cpu_ram_efficient_loading:
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
if cfg.fsdp_config.auto_wrap_policy:
    os.environ["FSDP_AUTO_WRAP_POLICY"] = cfg.fsdp_config.auto_wrap_policy
if cfg.fsdp_config.transformer_layer_cls_to_wrap:
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = cfg.fsdp_config.transformer_layer_cls_to_wrap
if cfg.fsdp_config.reshard_after_forward:
    os.environ["FSDP_RESHARD_AFTER_FORWARD"] = "true"
```

这段代码说明：Axolotl 并不是自己解析所有 FSDP 细节，而是把关键配置写成 Accelerate 能读的环境变量。`prepare_optim_env()` 决定何时调用它：只要 `cfg.fsdp` 或 `cfg.fsdp_config` 存在，就设置 FSDP 环境；否则才进入 DeepSpeed 分支（`src/axolotl/utils/trainer.py:643-666`）。

同时，Trainer 参数也会收到 FSDP 配置。在 `TrainerBuilderBase._set_base_training_args()` 中，Axolotl 把 `fsdp_config` 和 `fsdp` 放进 training args（`src/axolotl/core/builders/base.py:604-607`）：

```python
if self.cfg.fsdp_config or self.cfg.fsdp:
    training_args_kwargs["fsdp_config"] = self.cfg.fsdp_config
    training_args_kwargs["fsdp"] = self.cfg.fsdp if self.cfg.fsdp else True
```

于是同一份 YAML 被拆成两路：

```text
YAML fsdp_config
  ├─ setup_fsdp_envs -> ACCELERATE_USE_FSDP / FSDP_VERSION / FSDP_* env
  └─ TrainingArguments -> fsdp=True, fsdp_config=<FSDPConfig>
```

## 1.4 关键细节与误区澄清

> 容易误解一：`fsdp_config` 出现在 YAML 里，就表示 Axolotl 自己会执行 shard。

不对。Axolotl 在配置阶段主要写环境变量和 Trainer 参数。真正的 `fully_shard()` 发生在 Accelerate prepare 阶段，并且被 Axolotl patch 到 `src/axolotl/monkeypatch/accelerate/fsdp2.py:279-449`。

> 容易误解二：FSDP2 的 `sharding_strategy: FULL_SHARD` 是主配置。

从源码看，FSDP2 schema `FSDPConfig` 没有 `sharding_strategy` 字段（`src/axolotl/utils/schemas/fsdp.py:10-76`），文档迁移表也把 FSDP1 的 `fsdp_sharding_strategy` 映射为 FSDP2 的 `reshard_after_forward`（`docs/multi-gpu.qmd:91-130`）。因此对 FSDP2 主路径来说，真正影响 ZeRO-2 / ZeRO-3 语义的是 `reshard_after_forward`，不是 `sharding_strategy`。

> 容易误解三：`fsdp_final_state_dict_type` 会控制最终保存。

源码中确实有这个 deprecated 顶层字段（`src/axolotl/utils/schemas/config.py:945-950`），但最终保存只读取 `cfg.fsdp_config.final_state_dict_type`，否则回退到 `cfg.fsdp_config.state_dict_type`（`src/axolotl/train.py:294-300`）。未在源码中看到顶层 `fsdp_final_state_dict_type` 被映射到 `fsdp_config.final_state_dict_type`。

## 1.5 本章小结

> 💡 **小结**
>
> * Axolotl 的 FSDP 入口不是一个 wrapper 函数，而是一条“YAML -> env -> TrainingArguments -> Accelerate”的翻译链。
> * `setup_fsdp_envs()` 是第一个真正改变行为的函数，它让 Accelerate 进入 FSDP 模式。
> * FSDP2 配置里最关键的是 `fsdp_version: 2`、`auto_wrap_policy`、`transformer_layer_cls_to_wrap`、`reshard_after_forward` 和 state dict 类型。

# 二、DeviceMesh 与并行拓扑：单机 FSDP 和多机 HSDP 的差别不在模型代码里

## 2.1 设计哲学与核心问题

FSDP 可以只在一个 data-parallel group 内做参数分片；多节点时，更常见的做法是 HSDP：节点内 FSDP，节点间复制。这样做的工程直觉很直接：

- 节点内 GPU 通常有 NVLink / NVSwitch，适合频繁 all-gather 参数；
- 节点间网络更慢，更适合较低频的复制式梯度同步；
- 如果把 FSDP group 跨到慢网络上，每层 all-gather 都可能放大瓶颈。

Axolotl 并不在模型里写“这是单机 / 这是多机”。它用 `dp_shard_size`、`dp_replicate_size`、`tensor_parallel_size`、`context_parallel_size` 描述逻辑网格，再交给 Accelerate / PyTorch `DeviceMesh`。

## 2.2 源码入口与关键对象

```text
src/axolotl/utils/trainer.py
  - setup_parallelism_envs：把 dp_shard / dp_replicate / tp / cp 写入 PARALLELISM_CONFIG_* 环境变量。

src/axolotl/utils/distributed.py
  - build_parallelism_config：创建 Accelerate ParallelismConfig 和 DeviceMesh。
  - _get_parallel_config_kwargs：根据 world_size 推导 dp_shard / dp_replicate。

/usr/local/lib/python3.12/dist-packages/accelerate/parallelism_config.py
  - ParallelismConfig：下游 Accelerate 的网格定义与 mesh dim 名称。
```

## 2.3 主流程拆解

Axolotl 设置 parallelism env 的逻辑在 `src/axolotl/utils/trainer.py:621-640`：

```python
if cfg.dp_shard_size and cfg.dp_shard_size > 1:
    os.environ["PARALLELISM_CONFIG_DP_SHARD_SIZE"] = str(cfg.dp_shard_size)
if cfg.dp_replicate_size and cfg.dp_replicate_size > 1:
    os.environ["PARALLELISM_CONFIG_DP_REPLICATE_SIZE"] = str(cfg.dp_replicate_size)
if set_accelerate_parallelism_config:
    os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
```

`ModelLoader` 也会在模型加载前构造一次 parallelism config（`src/axolotl/loaders/model.py:196-213`）：

```text
ModelLoader.load
  -> apply_pre_model_load_patches()
  -> _apply_pre_model_load_setup()
    -> _set_parallel_config()
      -> build_parallelism_config(cfg)
```

`build_parallelism_config()` 在 `src/axolotl/utils/distributed.py:299-316`，核心是 `_get_parallel_config_kwargs()`：

```python
# src/axolotl/utils/distributed.py:338-362，简化
if dp_shard_size is None and dp_replicate_size in (None, 1):
    if remaining_world_size > 1:
        pc_kwargs["dp_shard_size"] = remaining_world_size

if dp_replicate_size and dp_replicate_size > 1:
    pc_kwargs["dp_replicate_size"] = dp_replicate_size

if remaining_world_size > 1 and dp_shard_size and dp_shard_size > 1:
    if not is_fsdp:
        raise ValueError(...)
    pc_kwargs["dp_shard_size"] = dp_shard_size
    if remaining_world_size > 1 and "dp_replicate_size" not in pc_kwargs:
        pc_kwargs["dp_replicate_size"] = remaining_world_size
```

这段逻辑有两个重要结论：

1. 纯 FSDP：如果用户没写 `dp_shard_size`，但 `world_size > 1` 且开启 FSDP，Axolotl 会把剩余 world size 推导为 `dp_shard_size`。
2. HSDP：如果用户写了 `dp_shard_size`，剩余 world size 可以自动落到 `dp_replicate_size`，除非用户显式给了 `dp_replicate_size`。

下游 Accelerate 的 `ParallelismConfig` 定义也能印证这种语义：`dp_shard_size > 1` 是 FSDP，`dp_replicate_size > 1 and dp_shard_size > 1` 是 HSDP（`/usr/local/lib/python3.12/dist-packages/accelerate/parallelism_config.py:61-66`）。mesh 维度顺序为 `dp_replicate -> dp_shard -> cp -> sp -> tp`（同文件 `:260-272`），并会额外 flatten 出 `dp_shard_cp` 和 `dp`（同文件 `:237-242`）。

一个 2 节点、每节点 8 卡的 HSDP + TP 示例来自 `examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml:6-22`：

```yaml
dp_shard_size: 4
dp_replicate_size: 2
tensor_parallel_size: 2
fsdp_version: 2
```

概念上可以理解为：

```text
world_size = 16
mesh_shape = [dp_replicate=2, dp_shard=4, tp=2]

每个 TP 坐标下：
  dp_shard 维度负责 FSDP 参数分片
  dp_replicate 维度负责复制式数据并行 / HSDP 复制
```

## 2.4 关键细节与误区澄清

> 容易误解一：Axolotl 会自动识别“哪些 rank 在同一节点”，然后把 FSDP 限制在节点内。

源码中没有看到这样的自动识别。`_get_parallel_config_kwargs()` 只看 `WORLD_SIZE` 和用户配置的 size（`src/axolotl/utils/distributed.py:319-370`），不读取 `machine_rank`、hostname 或 local world size。文档和示例建议 “FSDP within nodes, DDP across nodes”（`examples/distributed-parallel/README.md:34-45`），但这依赖用户设置的 `dp_shard_size` / `dp_replicate_size` 与 launch rank 排布匹配。

> 容易误解二：`dp_shard_size` 可以在未开启 FSDP 时单独使用。

不可以。源码明确检查：如果配置了 `dp_shard_size > 1` 但 `is_fsdp` 为 false，会抛出 `ValueError`（`src/axolotl/utils/distributed.py:347-352`）。

> 容易误解三：Context Parallelism 和 FSDP 完全独立。

在 Accelerate mesh 中，`dp_shard` 和 `cp` 会被 flatten 成 `dp_shard_cp`（Accelerate `parallelism_config.py:136-143`、`:237-242`）。Axolotl 的 FSDP2 prepare 会取 `mesh[parallelism_config.fsdp_dim_names]`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:344-360`）。因此 CP 不是简单“旁路功能”，它会影响 FSDP mesh 的维度选择。

## 2.5 本章小结

> 💡 **小结**
>
> * 单机 FSDP 与多机 HSDP 的差异主要来自 mesh size，而不是两套模型代码。
> * Axolotl 负责推导 `dp_shard_size` / `dp_replicate_size`，但不自动验证 rank 是否真的按节点排列。
> * 多维并行下，FSDP mesh 会与 CP / TP 的 mesh 维度发生组合关系，配置错误会直接变成拓扑错误。

# 三、模型加载与 Sharding：为什么必须先 patch、再加载、最后 prepare

## 3.1 设计哲学与核心问题

FSDP 最危险的阶段往往不是训练 step，而是**模型刚加载和刚 shard 的瞬间**。如果每个 rank 都完整加载 70B 模型，CPU RAM 可能先炸；如果模型在 shard 前被搬到 GPU，VRAM 会先炸；如果 QLoRA 的 `Params4bit` 在 shard/unshard 中丢失量化元数据，训练可能直接错或 silent corruption。

所以 Axolotl 的模型加载主线有一个很明确的顺序：

```text
先装 patch
  -> 再设置 parallelism / device_map / quantization kwargs
    -> 再 from_pretrained / 自定义量化加载
      -> 再加载 LoRA / QLoRA adapter
        -> 最后由 Accelerate.prepare 触发 FSDP2 fully_shard
```

## 3.2 源码入口与关键对象

```text
src/axolotl/train.py
  - setup_model_and_tokenizer：创建 ModelLoader 并调用 load。

src/axolotl/loaders/model.py
  - ModelLoader.load：模型加载总入口。
  - _apply_pre_model_load_setup：设置 parallel config、device_map、quantization。
  - _build_model：处理 cpu_ram_efficient_loading、QLoRA sharded loading、from_pretrained。

src/axolotl/loaders/patch_manager.py
  - apply_pre_model_load_patches：加载前注入 FSDP / QLoRA / Accelerate patch。

src/axolotl/monkeypatch/accelerate/fsdp2.py
  - fsdp2_prepare_model：替换 Accelerate FSDP2 prepare，调用 PyTorch fully_shard。
```

## 3.3 主流程拆解

训练 setup 从 `src/axolotl/train.py:522-570` 进入，模型加载发生在 `setup_model_and_tokenizer()`：

```text
setup_model_and_trainer
  -> setup_model_and_tokenizer
    -> ModelLoader(cfg, tokenizer)
    -> model_loader.load()
```

`ModelLoader.load()` 的顺序非常关键（`src/axolotl/loaders/model.py:162-194`）：

```text
ModelLoader.load
  -> patch_manager.apply_pre_model_load_patches()
  -> _apply_pre_model_load_setup()
  -> _build_model()
  -> patch_manager.apply_post_model_build_patches(model)
  -> _apply_post_model_load_setup()
  -> _load_adapters()
  -> _apply_post_lora_load_setup()
  -> patch_manager.apply_post_model_load_patches(model)
```

对于 FSDP，`_apply_pre_model_load_setup()` 会在 FSDP2 时启用 parallel config（`src/axolotl/loaders/model.py:196-213`）。注意这里有一个分支：如果配置的是 FSDP1，`use_parallel_config` 会被设为 false（同文件 `:207-208`）。

加载阶段的显存控制在 `_build_model()` 中：

- `cpu_ram_efficient_loading=True` 时，设置 `skip_move_to_device=True`（`src/axolotl/loaders/model.py:756-758`）；
- FSDP2 且无 TP 时，rank0 的 `device_map` 是 CPU，其他 rank 是 meta（`src/axolotl/loaders/model.py:769-779`）；
- QLoRA + FSDP 在部分模型或 `qlora_sharded_model_loading` 下走 `load_sharded_model_quant()`（`src/axolotl/loaders/model.py:781-808`）。

真正的 sharding 不是在 `ModelLoader._build_model()` 中完成，而是在 Accelerate prepare 模型时完成。Axolotl 通过 patch 替换了 Accelerate 的 FSDP2 prepare（`src/axolotl/monkeypatch/accelerate/fsdp2.py:529-538`），其核心函数是 `fsdp2_prepare_model()`：

```python
# src/axolotl/monkeypatch/accelerate/fsdp2.py:403-415，简化
auto_wrap_policy = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)
if auto_wrap_policy is not None:
    for module in get_module_children_bottom_up(model)[:-1]:
        if auto_wrap_policy(module) and not isinstance(module, FSDPModule):
            fully_shard(module, **fsdp2_kwargs)

fully_shard(model, **fsdp2_kwargs)
```

这段代码说明 FSDP2 wrap 是 bottom-up：先对满足 auto-wrap policy 的子模块 shard，再对根模型 shard。`activation_checkpointing` 也必须在 `fully_shard()` 之前应用（`src/axolotl/monkeypatch/accelerate/fsdp2.py:327-342`），否则 checkpoint wrapper 和 FSDP wrapper 的组合顺序会不一致。

## 3.4 关键细节与误区澄清

> 容易误解一：`ModelLoader._set_parallel_config()` 构建的 mesh 就是 FSDP wrapping 一定使用的 mesh。

不完全是。`ModelLoader` 保存了 `self.device_mesh`，并在 TP 加载时传给 `from_pretrained()`（`src/axolotl/loaders/model.py:749-752`）。但 FSDP2 prepare 使用的是 `accelerator.state.device_mesh`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:344-360`）。纯 FSDP 没有显式 parallelism config 时，mesh 可以是 `None`，PyTorch FSDP 使用默认进程组。

> 容易误解二：`cpu_ram_efficient_loading` 只是一个加载优化，不影响后续 patch。

不对。它触发了 rank0 CPU / 非 rank0 meta 加载（`src/axolotl/loaders/model.py:769-779`），还会触发 `patch_tied_keys_for_meta_device()`（`src/axolotl/loaders/patch_manager.py:293-295`）以及 `fsdp2_load_full_state_dict()` 广播逻辑（`src/axolotl/monkeypatch/accelerate/fsdp2.py:422-425`）。这是一个跨加载、shard、广播、保存的状态设计。

> 容易误解三：`load_sharded_model()` 是 FSDP 主加载函数。

不是。`src/axolotl/utils/model_shard_quant.py:140-164` 定义的 `load_sharded_model()` 在当前源码搜索中未发现调用点。FSDP + QLoRA 的特殊路径调用的是 `load_sharded_model_quant()`，且只在 `is_qlora_and_fsdp_enabled and cpu_ram_efficient_loading and (dbrx or qlora_sharded_model_loading)` 时触发（`src/axolotl/loaders/model.py:781-808`）。

## 3.5 本章小结

> 💡 **小结**
>
> * FSDP2 wrapping 发生在 Accelerate prepare 阶段，Axolotl 通过 patch 改写 prepare 行为。
> * `cpu_ram_efficient_loading` 的本质是 rank0 真实加载、其他 rank meta 占位，再由 FSDP2 broadcast / distribute。
> * QLoRA + FSDP2 不是普通 FSDP 的小分支，而是需要 bitsandbytes 参数类型补丁的兼容路径。

# 四、完整主路径串联

## 4.1 完整调用栈

```text
User: axolotl train config.yml --launcher accelerate/torchrun
  │
  ├─ Step 1: CLI launcher
  │     ├─ src/axolotl/cli/main.py:98 train
  │     └─ src/axolotl/cli/utils/train.py:109 launch_training
  │
  ├─ Step 2: 配置加载与 FSDP env
  │     ├─ src/axolotl/cli/train.py:55 do_cli
  │     ├─ src/axolotl/cli/config.py:230 load_cfg
  │     ├─ src/axolotl/utils/schemas/fsdp.py:10 FSDPConfig
  │     └─ src/axolotl/utils/trainer.py:589 setup_fsdp_envs
  │
  ├─ Step 3: 模型加载与 patch
  │     ├─ src/axolotl/train.py:83 ModelLoader(...).load()
  │     ├─ src/axolotl/loaders/patch_manager.py:95 apply_pre_model_load_patches
  │     └─ src/axolotl/loaders/model.py:745 _build_model
  │
  ├─ Step 4: Trainer 构建与 Accelerate prepare
  │     ├─ src/axolotl/utils/trainer.py:679 setup_trainer
  │     ├─ src/axolotl/core/builders/base.py:604 fsdp args
  │     └─ src/axolotl/monkeypatch/accelerate/fsdp2.py:279 fsdp2_prepare_model
  │
  ├─ Step 5: 训练循环
  │     ├─ src/axolotl/train.py:183 execute_training
  │     ├─ src/axolotl/train.py:227 trainer.train(...)
  │     └─ src/axolotl/core/trainers/base.py:366 compute_loss
  │
  └─ Step 6: 保存 / merge
        ├─ src/axolotl/train.py:254 save_trained_model
        ├─ src/axolotl/core/trainers/mixins/distributed_parallel.py:14 _save
        ├─ src/axolotl/monkeypatch/accelerate/fsdp2.py:100 get_state_dict
        └─ src/axolotl/cli/merge_sharded_fsdp_weights.py:108 merge_fsdp_weights
```

## 4.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 是否通信 | 是否每 step 执行 |
|---|---|---|---|---|
| CLI launcher | config path、launcher args | 启动 `accelerate` 或 `torchrun` 子进程 | 否 | 否 |
| `load_cfg` | YAML + CLI override | `DictDefault`、FSDP env、parallelism env | 否 | 否 |
| `ModelLoader.load` | cfg、tokenizer | 已加载模型、adapter config、patch 已安装 | 可能触发模型下载/加载；非 FSDP 通信 | 否 |
| `Accelerator.prepare` / FSDP2 prepare | model、TrainingArguments、AcceleratorState | 模型被 `fully_shard()`，参数变为 sharded / DTensor 语义 | 可能初始化 process group；cpu-efficient loading 会广播 | 否，prepare 阶段 |
| `trainer.train` | dataloader、FSDP-wrapped model | loss、grad、optimizer step | 是，FSDP 每层 all-gather / reduce-scatter 由 PyTorch 处理 | 是 |
| final save | trainer、model、state_dict_type | full / sharded checkpoint，必要时 merge | 是，full state dict / DCP load 会通信或 CPU 聚合 | 否 |

一个关键状态切换在 `AxolotlTrainer.create_accelerator_and_postprocess()`：它会在创建 Accelerator 前重置 `AcceleratorState`，确保 Accelerate 从当前环境变量重新配置（`src/axolotl/core/trainers/base.py:666-674`）。这就是为什么 `setup_fsdp_envs()` 必须早于 Trainer 创建。

## 4.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---:|---|
| `src/axolotl/utils/model_shard_quant.py:140 load_sharded_model` | 名字像 FSDP sharded loading | 否，当前搜索未发现调用点 | QLoRA 特殊路径用的是 `load_sharded_model_quant()`。 |
| `src/axolotl/cli/merge_sharded_fsdp_weights.py` | 名字包含 FSDP | 训练 step 不调用 | 只在 final save 后自动 merge 某些 sharded 输出，或用户手动 CLI 调用。 |
| `SequenceParallelContextManager` | FSDP + CP 示例中会出现 | 仅 `context_parallel_size > 1` 时进入 | 它处理序列维度，不是 FSDP 参数分片本身。入口在 `src/axolotl/train.py:205-220`。 |
| `setup_deepspeed_env` | 同样是省显存分布式 | 与 FSDP 互斥分支 | `prepare_optim_env()` 中 FSDP 分支优先于 DeepSpeed（`src/axolotl/utils/trainer.py:648-664`）。 |
| `patch_prepare_cp()` | 看起来影响并行 prepare | 只在 CP standalone 时 patch | 在 `context_parallel_size > 1` 时设置，用于 CP 兼容（`src/axolotl/utils/trainer.py:632-638`）。 |

> 💡 **小结**
>
> * FSDP 的完整主路径横跨 CLI、config、ModelLoader、Trainer、Accelerate patch、save。
> * 训练 step 中 Axolotl 自己并不显式写 all-gather；它把模型交给 PyTorch FSDP2。
> * 很多 FSDP 相关文件是初始化、兼容或保存路径，不应误认为每 step 主流程。

# 五、关键数据流 / 状态流 / shape 流程

## 5.1 Tensor shape：FSDP 不切输入，它切参数状态

FSDP 与 context parallel 不同。FSDP 不把 `input_ids: [batch, seq]` 切成多个 seq chunk；输入 batch 的 shape 在 Axolotl 主训练路径中仍由 data collator / Trainer 决定。

```text
普通 SFT batch:
  input_ids:      [micro_batch_size, sequence_len]
  attention_mask: [micro_batch_size, sequence_len]
  labels:         [micro_batch_size, sequence_len]

FSDP 后：
  batch shape 不因为 FSDP 改变
  参数 P 的存储生命周期改变
```

参数侧可以抽象成：

```text
逻辑参数:
  W: [hidden_out, hidden_in]

FSDP sharded idle 状态:
  rank0 local shard: W_0
  rank1 local shard: W_1
  ...

某个 wrapped module forward 前:
  all-gather -> 每个 rank 临时得到该 module 的完整 W

forward 后且 reshard_after_forward=True:
  释放完整 W，只保留 local shard

backward:
  再次需要完整参数 / 或使用 FSDP 内部调度
  grad reduce-scatter -> 每个 rank 只保留 grad shard
```

源码证据在 `fsdp2_prepare_model()`：Axolotl 把 `reshard_after_forward` 放进 `fully_shard()` kwargs（`src/axolotl/monkeypatch/accelerate/fsdp2.py:351-360`），再对模块和根模型调用 `fully_shard()`（同文件 `:403-415`）。逐层 all-gather / reduce-scatter 是 PyTorch FSDP2 的职责，Axolotl 只负责把策略传进去。

如果启用 CP，才会出现序列维度变化。`execute_training()` 在 `context_parallel_size > 1` 时进入 `SequenceParallelContextManager`，并传入 `device_mesh=trainer.accelerator.torch_device_mesh`（`src/axolotl/train.py:205-220`）。这属于 FSDP + CP 组合，不是 FSDP 单独行为。

## 5.2 Rank / Mesh / Process Group 变化

纯 FSDP，`world_size=8` 且没有显式 TP/CP/HSDP 时，可以理解为：

```text
world_size = 8
remaining_world_size = 8

_get_parallel_config_kwargs(..., is_fsdp=True)
  -> dp_shard_size = 8

FSDP group:
  [rank0, rank1, rank2, rank3, rank4, rank5, rank6, rank7]
```

HSDP 示例：2 节点，每节点 8 卡，总 16 rank，配置来自 `examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml:6-22`：

```text
world_size = 16
dp_shard_size = 4
dp_replicate_size = 2
tensor_parallel_size = 2

逻辑 mesh:
  [dp_replicate=2, dp_shard=4, tp=2]
```

这意味着：

- `dp_shard` 维度负责 FSDP 参数分片；
- `dp_replicate` 维度负责复制 / HSDP；
- `tp` 维度交给 Transformers / Accelerate TP handler；
- 具体 rank 是否正好落在同一节点，源码没有自动校验。

## 5.3 状态切换：环境变量、AcceleratorState、全局 monkey patch

FSDP 启动至少有三类状态：

```text
进入配置阶段:
  setup_fsdp_envs 写入 ACCELERATE_USE_FSDP / FSDP_VERSION / FSDP_*
  setup_parallelism_envs 写入 PARALLELISM_CONFIG_*

创建 Trainer / Accelerator 前:
  AxolotlTrainer.create_accelerator_and_postprocess
    -> AcceleratorState._reset_state(reset_partial_state=True)

模型加载前:
  PatchManager.apply_pre_model_load_patches
    -> 替换 Accelerate / Transformers / PyTorch FSDPParam / PEFT 方法
```

这些状态的线程 / 进程安全边界也不同：

- 环境变量是进程级；每个 launcher worker 都会各自读取；
- `AcceleratorState` 是 Python 进程内全局状态；Axolotl 会 reset 以避免旧状态污染；
- monkey patch 是模块级全局替换，多数没有 restore，只靠进程生命周期隔离。

> 💡 **小结**
>
> * FSDP 的 shape 变化主要发生在参数和梯度的本地 shard / full tensor 生命周期，而不是输入 batch。
> * HSDP 的“节点内 / 节点间”语义来自 mesh 配置与 rank 排布约定，源码不自动识别物理节点。
> * FSDP 依赖多个全局状态：env、AcceleratorState、monkey patch；这也是维护风险来源。

# 六、核心机制深挖

## 6.1 Monkey Patch：零侵入接入还是维护风险？

### 它解决什么问题？

Axolotl 要同时兼容 Transformers、Accelerate、PEFT、bitsandbytes 和 PyTorch FSDP2。很多问题无法只靠公开 API 解决，例如：

- Accelerate 原生 FSDP2 prepare 不满足 Axolotl 的 cpu-efficient loading / LoRA 处理需求；
- Transformers 在 meta tensor tied keys 上可能误判；
- bitsandbytes 的 `Params4bit` / `Int8Params` 在 FSDP2 shard/unshard 中需要保留额外量化元数据。

### 源码怎么实现？

Patch 入口集中在 `PatchManager.apply_pre_model_load_patches()`（`src/axolotl/loaders/patch_manager.py:95-123`）。FSDP 相关逻辑在 `_apply_fsdp_patches()`：

```text
src/axolotl/loaders/patch_manager.py:270-299
  -> patch_initialize_missing_keys_for_fsdp()
  -> patch_parallelism_config()
  -> patch_accelerate_fsdp2()
  -> patch_tied_keys_for_meta_device()  # cpu_ram_efficient_loading only
  -> patch_trl_prepare_fsdp2()          # RL only
```

`patch_accelerate_fsdp2()` 是直接替换：

```python
# src/axolotl/monkeypatch/accelerate/fsdp2.py:529-538
accelerate.accelerator.fsdp2_prepare_model = fsdp2_prepare_model
accelerate.Accelerator.get_state_dict = get_state_dict
```

QLoRA / bitsandbytes patch 入口在 `src/axolotl/loaders/patch_manager.py:590-608`，条件是 FSDP2 且 `load_in_4bit` 或 `load_in_8bit`。它会修改 PyTorch 内部 `FSDPParam` 方法：

- `_init_sharded_param`：创建 sharded param 时保留 `Params4bit` / `Int8Params` 类型与量化元数据（`src/axolotl/monkeypatch/fsdp2_qlora.py:19-94`）；
- `init_unsharded_param`：unshard 时重建 bitsandbytes 参数对象（同文件 `:96-170`）；
- `init_dtype_attrs`：避免 mixed precision 把非浮点量化参数错误 cast（同文件 `:205-236`）；
- `Linear8bitLt._save_to_state_dict`：保存时临时 unwrap DTensor（同文件 `:172-202`）。

### 隐藏假设和副作用

- `fsdp2_qlora.py` 通过 `inspect.getsource()` + 字符串替换 + `exec()` patch PyTorch 内部函数（`src/axolotl/monkeypatch/fsdp2_qlora.py:25-90`、`:102-166`）。这强依赖上游源码形态。
- 一些 patch 有 `_axolotl_patched` guard，例如 `_initialize_missing_keys`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:508-526`）和 QLoRA patch；但 `patch_accelerate_fsdp2()` 本身只是赋值，没有 restore。
- patch 是进程级全局替换。测试进程内如果先后跑不同配置，理论上存在污染风险；目前主要依赖 idempotent guard 和测试隔离。

## 6.2 通信原语：前向和反向是否对称？

Axolotl 源码中直接出现的 FSDP 通信主要在保存 / 加载 patch，而训练时逐层通信由 PyTorch FSDP2 内部完成。

训练阶段：

- forward 前对 wrapped module 参数 all-gather；
- `reshard_after_forward=True` 时 forward 后释放 full param；
- backward 需要参数时再次调度 gather；
- gradient 通过 reduce-scatter 落回 shard；
- optimizer state 也按 shard 管理。

这部分语义在项目文档中有明确描述：FSDP shard 参数、梯度、optimizer state，并在计算前 all-gather，`reshard-after-forward` 后丢弃（`docs/nd_parallelism.qmd:23-26`）。源码层面，Axolotl 把 `reshard_after_forward` 传给 `fully_shard()`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:351-360`）。

保存 / 加载阶段，Axolotl 自己写了通信：

- `fsdp2_load_full_state_dict()` 对 sharded DTensor 使用 `distribute_tensor(..., src_data_rank=0)`，对非 sharded 参数使用 `dist.broadcast()`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:39-91`）；
- `get_state_dict()` 在 FSDP2 下对每个 DTensor 调用 `full_tensor()`，rank0 收集到 CPU，并且每个参数后有 `torch.distributed.barrier()`（同文件 `:158-173`）。

这意味着保存 full state dict 时通信不是“每 step”，但可能非常重：它按参数逐个 full-gather，并在参数粒度 barrier。

## 6.3 配置归一化：用户配置如何变成真实行为？

配置链路可以概括为：

```text
YAML
  -> Pydantic schema / validation
  -> DictDefault cfg
  -> env vars
  -> TrainingArguments
  -> AcceleratorState / FSDPPlugin
```

关键源码：

- `FSDPConfig` 定义字段（`src/axolotl/utils/schemas/fsdp.py:10-76`）；
- legacy `fsdp_` 前缀会被移除（`src/axolotl/utils/schemas/validation.py:1052-1069`）；
- `fsdp_version` 可以从 `fsdp_config.version` 或 `fsdp_config.fsdp_version` 推导（同文件 `:1074-1085`）；
- FSDP2 要求 torch >= 2.7（`src/axolotl/utils/schemas/config.py:1720-1735`）；
- `normalize_config()` 会在多卡下按 effective world size 放大全局 batch size（`src/axolotl/utils/config/__init__.py:133-143`）。

一个容易忽略的点：Axolotl 在训练参数中设置 `average_tokens_across_devices=False`（`src/axolotl/core/builders/base.py:588-590`）。这意味着 token 统计和 loss 归一化要避免被 Transformers 默认逻辑重复跨设备平均；同时 `compute_loss()` 中吞吐统计会显式 all-reduce tokens（`src/axolotl/core/trainers/base.py:376-390`）。

> 💡 **小结**
>
> * Axolotl 的 FSDP 集成是“配置翻译 + 下游 patch + PyTorch FSDP2 执行”的组合。
> * 训练阶段通信主要由 PyTorch FSDP2 完成；Axolotl 直接写通信的地方集中在 state dict 保存 / rank0 broadcast loading。
> * monkey patch 带来低侵入接入能力，也带来上游版本耦合和全局污染风险。

# 七、显存、性能与通信分析

## 7.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ✅ | FSDP2 `fully_shard()` 后 idle 状态只保留 local shard；`reshard_after_forward=True` 时 forward 后释放 full param。 |
| 梯度 | ✅ | backward 后梯度 reduce-scatter，rank 只保留 grad shard。 |
| optimizer state | ✅ | FSDP / HSDP 下 optimizer state 随参数 shard 分布；Muon 等特殊 optimizer 要求 FSDP2。 |
| 激活值 | ❌ / 间接 | FSDP 不切激活；需要 activation checkpointing / activation offloading / CP 才能省激活。 |
| 输入 batch | ❌ | FSDP 不改变 `input_ids [batch, seq]`；batch 分发由 data parallel sampler / Trainer 处理。 |
| logits | ❌ | 标准 FSDP 不切 vocab logits；Cut Cross Entropy / Liger 等插件才影响 logits/loss 显存路径。 |
| 初始化 CPU RAM | ✅（可选） | `cpu_ram_efficient_loading` 让非 rank0 meta loading，rank0 后续广播。 |
| 保存期显存 / CPU 内存 | ⚠️ | FULL_STATE_DICT 会 full-gather；rank0 CPU 聚合可能成为峰值。SHARDED_STATE_DICT 降低聚合但需要 merge。 |

真正的显存大头是参数、梯度、optimizer state 和激活。FSDP 主要解决前三者；激活要靠 `activation_checkpointing`、`activation_offloading` 或 CP。示例 `examples/gpt-oss/gpt-oss-120b-fft-fsdp2-offload.yaml:53-72` 同时启用 gradient checkpointing、activation offloading、FSDP2 offload 和 `cpu_ram_efficient_loading`，这说明工程上通常需要多个机制叠加。

## 7.2 通信开销

| 阶段 | 通信类型 | 发生频率 | 源码 / 依据 |
|---|---|---|---|
| FSDP forward | all-gather 参数 | 每个 wrapped module / layer | PyTorch FSDP2 内部；Axolotl 传入 `fully_shard(... reshard_after_forward ...)`（`fsdp2.py:351-415`）。 |
| FSDP backward | all-gather / reduce-scatter | 每个 wrapped module / backward | PyTorch FSDP2 内部；文档说明参数 gather 与 ZeRO-3 语义（`docs/nd_parallelism.qmd:23-26`）。 |
| rank0 full state loading | distribute / broadcast | 初始化加载一次，每个参数 | `fsdp2_load_full_state_dict()`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:39-91`）。 |
| final FULL_STATE_DICT save | full_tensor + barrier | 保存时每个参数 | `get_state_dict()` FSDP2 分支（同文件 `:158-173`）。 |
| sharded checkpoint merge | DCP CPU load | final merge / 手动 CLI | `merge_fsdp_weights()` 与 `_distributed_checkpoint_to_merged_weights()`（`src/axolotl/cli/merge_sharded_fsdp_weights.py:38-104`、`:108-167`）。 |
| token 统计 | all_reduce | `include_tkps` 且训练中每次 compute_loss | `AxolotlTrainer.compute_loss()`（`src/axolotl/core/trainers/base.py:376-390`）。 |

FSDP 的性能本质是通信换显存。`reshard_after_forward=True` 越省显存，backward 重新 gather 的概率越高；`reshard_after_forward=False` 更像 ZeRO-2，少一些参数重 gather，但参数显存收益下降。Axolotl 文档也把二者对应到 ZeRO-2 / ZeRO-3（`docs/nd_parallelism.qmd:24-26`）。

## 7.3 性能取舍

- **通信换显存**：FSDP 把参数/梯度/optimizer state 分片，但每层计算前要 gather。
- **初始化内存换广播复杂度**：`cpu_ram_efficient_loading` 降低非 rank0 CPU RAM，但需要 meta-device patch、rank0 full state broadcast、tied weight 修复。
- **保存便利换 rank0 峰值**：`FULL_STATE_DICT` 最方便上传 / 加载，但 rank0 聚合 CPU 内存大；`SHARDED_STATE_DICT` 更分布式，但需要 merge。
- **patch 复杂度换生态兼容**：QLoRA + FSDP2 的 bitsandbytes 参数类型不是 PyTorch 原生 Parameter，必须 patch 内部 shard/unshard。
- **多节点效率依赖 rank 排布**：HSDP 的收益来自“节点内 FSDP、节点间复制”，但源码不验证物理拓扑，配置错误会把频繁 all-gather 放到跨节点网络上。

> 💡 **小结**
>
> * FSDP 真正节省的是参数、梯度、optimizer state，不直接节省输入 batch / logits / 激活。
> * 保存和初始化阶段同样可能触发大规模通信与 rank0 峰值，不应只关注 forward OOM。
> * 多节点 FSDP 的性能高度依赖 `dp_shard_size` 是否匹配节点内高速互联。

# 八、配置项、边界条件与坑点

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `fsdp_config` | `prepare_optim_env()` -> `setup_fsdp_envs()`；`TrainerBuilderBase._set_base_training_args()` | 开启 FSDP env，并传入 Trainer 参数 | 只有 YAML 中出现还不够，必须被 validate 后保留下来。 |
| `fsdp_version: 2` | `setup_fsdp_envs()` 设置 `FSDP_VERSION=2` | 进入 FSDP2 路径，启用 Axolotl FSDP2 patch | schema 要求 torch >= 2.7（`config.py:1720-1735`）。 |
| `auto_wrap_policy` | `FSDP_AUTO_WRAP_POLICY`；`fsdp2_prepare_model()` auto wrap | 决定哪些 transformer layer 被独立 shard | `transformer_layer_cls_to_wrap` 写错会导致 wrap 粒度异常或下游报错。 |
| `transformer_layer_cls_to_wrap` | `FSDP_TRANSFORMER_CLS_TO_WRAP` | 指定 layer 类名 | 多模态 / MoE 模型常有特殊 layer 名，需参考 `_no_split_modules`。 |
| `reshard_after_forward` | `FSDP_RESHARD_AFTER_FORWARD`；`fully_shard()` kwargs | true 更省参数显存，false 减少重 gather | true 通信更重；false 显存收益下降。 |
| `cpu_ram_efficient_loading` | `ModelLoader._build_model()`；`patch_tied_keys_for_meta_device()`；`fsdp2_load_full_state_dict()` | rank0 CPU 加载，其他 rank meta，占用更低 CPU RAM | 不支持某些量化配置，如 Mxfp4Config 校验会拒绝（`validation.py:1430-1436`）。 |
| `offload_params` | `FSDP_OFFLOAD_PARAMS`；`fsdp2_prepare_model()` CPUOffloadPolicy | 参数可 offload 到 CPU | 可能极慢；`cpu_offload_pin_memory:false` 还要求 `offload_params:true`（`validation.py:1019-1032`）。 |
| `cpu_offload_pin_memory:false` | `FSDP_CPU_OFFLOAD_PIN_MEMORY`；`fsdp2_prepare_model()` 修改 pin_memory | 允许资源紧张场景使用 swap | 只有 FSDP2 支持；性能开销大。 |
| `state_dict_type` | `setup_fsdp_envs()`；`save_trained_model()` | 控制 checkpoint / final save 默认 state dict 类型 | `SHARDED_STATE_DICT` 需要 merge，FULL 可能 rank0 峰值大。 |
| `final_state_dict_type` | `save_trained_model()` | 只覆盖最终保存类型 | 顶层 `fsdp_final_state_dict_type` 未见映射到这里。 |
| `dp_shard_size` | `setup_parallelism_envs()`；`_get_parallel_config_kwargs()` | 指定 FSDP shard group size | 未开启 FSDP 时配置会报错；多节点需用户确保 rank 排布。 |
| `dp_replicate_size` | 同上 | 开启 HSDP 复制维度 | 源码只看 size，不验证是否跨节点。 |
| `tensor_parallel_size` | `ModelLoader._build_model()` 传 `tp_size` / `device_mesh` | 与 FSDP 组成多维并行 | 会影响 `effective_world_size` batch 计算（`normalize_config`:133-143）。 |
| `context_parallel_size` | `SequenceParallelContextManager`；Accelerate mesh | 与 FSDP 组合处理长上下文 | 不是 FSDP 本身；会改变序列维度和 mesh。 |
| `load_in_4bit/load_in_8bit + fsdp_version=2` | `PatchManager._apply_fsdp2_bnb_patches()` | 安装 bitsandbytes FSDP2 patch | RL 的 DPO/KTO/ORPO/IPO + base quant 会被拒绝（`validation.py:1036-1046`）。 |
| `optimizer: adamw_8bit` | validation | FSDP2 下拒绝，建议 `adamw_torch_8bit` | `validation.py:1102-1117` 明确报错。 |
| `optimizer: muon` | validation | 只允许 FSDP2 | `validation.py:906-918`。 |
| `deepspeed + fsdp_config` | `prepare_optim_env()` 分支、Trainer args | 源码未见对 `deepspeed + fsdp_config` 的同等显式互斥校验 | `check_fsdp_deepspeed()` 只检查 `deepspeed and fsdp`（`validation.py:1189-1191`）；这是基于源码的风险判断。 |

最小 FSDP2 配置通常应包含：

```yaml
fsdp_version: 2
fsdp_config:
  offload_params: false
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: <YourDecoderLayer>
  reshard_after_forward: true
```

多机 HSDP 则额外配置：

```yaml
dp_shard_size: <每个 FSDP shard group 的 rank 数>
dp_replicate_size: <复制组数，通常等于节点数或节点组数>
```

并通过 torchrun 或 accelerate 多机启动。文档给出的 torchrun 入口是 `axolotl train config.yaml --launcher torchrun -- --nnodes ...`（`docs/multi-node.qmd:70-82`）。

> 💡 **小结**
>
> * 配置项的真实含义要看它进入 env、TrainingArguments、ModelLoader 还是 final save。
> * FSDP2 已经是主推路径；FSDP1 legacy 字段仍存在兼容代码，但文档明确建议迁移。
> * 多节点 FSDP / HSDP 的最大坑不是语法，而是 rank 拓扑与网络拓扑不匹配。

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/utils/schemas/validation/test_fsdp.py:16-38` | `fsdp_version` 从 `fsdp_config` 推导 | 覆盖配置归一化基本路径。 |
| `tests/utils/schemas/validation/test_fsdp.py:40-118` | offload、8bit optimizer、pin_memory 约束 | 覆盖若干不兼容组合。 |
| `tests/utils/schemas/validation/test_fsdp.py:120-138` | legacy `fsdp_` 前缀移除 | 证明旧字段会迁移为无前缀字段。 |
| `tests/test_loaders.py:181-218` | `_get_parallel_config_kwargs()` | 覆盖 world_size、TP、CP、dp_shard、dp_replicate 的组合推导。 |
| `tests/e2e/multigpu/test_fsdp2.py:52-110` | FSDP2 full fine-tune + cpu_ram_efficient_loading true/false | 2 GPU smoke/e2e，检查产物和 loss 非 NaN。 |
| `tests/e2e/multigpu/test_fsdp2.py:112-302` | LoRA / QLoRA + FSDP2 | 覆盖 adapter 路径和 4bit 路径。 |
| `tests/e2e/multigpu/test_llama.py:467-547` | FSDP2 + sample packing + flash/flex + reshard true/false | 覆盖 packed batch 与 state_dict_type=SHARDED_STATE_DICT。 |
| `tests/e2e/patched/test_fsdp2_qlora.py:10-30` | FSDP2 Params4bit patch 是否替换方法 | 只验证 patch 安装，不验证真实训练正确性。 |
| `tests/cli/test_cli_merge_sharded_fsdp_weights.py:8-108` | merge CLI launcher 参数 | 验证 python / torchrun / accelerate launcher 命令组装。 |
| `examples/distributed-parallel/*.yaml` | FSDP + TP + CP、HSDP + TP 示例 | 展示单机 8 卡、多机 16 卡配置。 |

## 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| 真实多节点 HSDP | 未在本地测试中确认 | rank 拓扑错误可能导致跨节点 all-gather 过慢或 hang。 |
| 保存 FULL_STATE_DICT 的 rank0 CPU 峰值 | 部分 e2e 间接覆盖 | 大模型时 rank0 OOM，small model 测不出来。 |
| `SHARDED_STATE_DICT` 自动 merge 大模型 | 有 CLI mock / 小规模路径，缺少大模型压力 | DCP merge CPU 时间、磁盘空间、dtype cast 风险。 |
| QLoRA patch 对上游 PyTorch FSDPParam 源码变化 | 只测试 patch 能替换 | 上游函数源码稍变，字符串替换失败或 silent fallback warning。 |
| `deepspeed + fsdp_config` 组合 | 未见显式测试 | 可能同时传 deepspeed 和 fsdp TrainingArguments，行为依赖下游。 |
| `fsdp_final_state_dict_type` deprecated 顶层字段 | 未见测试 | 用户以为生效，实际 final save 仍用 `fsdp_config.state_dict_type`。 |
| 性能 / 显存收益 | e2e 只看训练完成和 loss | 无法证明通信 overlap、显存峰值或吞吐符合预期。 |
| 多模态 / MoE 特殊 layer wrap | 有部分示例 / 文档，覆盖不完整 | `transformer_layer_cls_to_wrap` 错误会导致未 shard 或报错。 |

一个值得注意的 skipped 测试：FSDP2 DPO 相关测试被标记 slow skip（`tests/e2e/multigpu/test_fsdp2.py:368-428`），FSDP1 DPO 也有 skip（`tests/e2e/multigpu/test_fsdp1.py:189-205`）。这说明 RL + FSDP 的长尾路径仍然不是最稳的主线。

> 💡 **小结**
>
> * Axolotl 对 FSDP2 的基础 SFT、LoRA、QLoRA、packed batch 有多 GPU e2e 覆盖。
> * 多节点 HSDP、保存峰值、性能收益更多依赖示例和工程经验，测试保护不足。
> * patch 类测试多是“能否安装”，不等价于“上游版本升级后语义一定正确”。

# 十、局限性与已知优化点

## 10.1 硬约束

- FSDP2 要求 torch >= 2.7（`src/axolotl/utils/schemas/config.py:1720-1735`）。
- `dp_shard_size` 只能与 FSDP 一起使用（`src/axolotl/utils/distributed.py:347-352`）。
- Falcon 模型被显式拒绝 FSDP（`src/axolotl/utils/schemas/validation.py:1347-1350`）。
- ReLoRA 与 FSDP / DeepSpeed 都不兼容（`src/axolotl/utils/schemas/validation.py:1468-1477`）。
- FSDP2 + DPO/KTO/ORPO/IPO 不支持 base model 4bit/8bit（`src/axolotl/utils/schemas/validation.py:1036-1046`）。
- FSDP2 + `adamw_8bit` / `adamw_bnb_8bit` 被拒绝，建议 `adamw_torch_8bit`（`src/axolotl/utils/schemas/validation.py:1102-1117`）。
- `cpu_offload_pin_memory:false` 只支持 FSDP2 且要求 `offload_params:true`（`src/axolotl/utils/schemas/validation.py:1019-1032`）。

## 10.2 维护成本

- `fsdp2_qlora.py` 依赖 PyTorch 内部 `FSDPParam` 的源码文本，字符串替换失败只 warning（`src/axolotl/monkeypatch/fsdp2_qlora.py:63-94`、`:139-170`）。
- `patch_accelerate_fsdp2()` 直接替换 Accelerate 命名空间，没有 restore（`src/axolotl/monkeypatch/accelerate/fsdp2.py:529-538`）。
- FSDP1 兼容字段、FSDP2 新字段、文档迁移表同时存在，配置追踪成本高。
- 多节点拓扑由用户配置 size 表达，源码不验证物理节点边界。

## 10.3 性能瓶颈

- `reshard_after_forward=True` 下，每层参数生命周期更短，但通信更频繁。
- `get_state_dict()` FSDP2 分支对每个参数 `full_tensor()` 后 barrier（`src/axolotl/monkeypatch/accelerate/fsdp2.py:158-173`），大模型保存可能串行瓶颈明显。
- `fsdp2_load_full_state_dict()` 每个参数从 rank0 分发，并对非 sharded 参数使用 `dist.broadcast()`（同文件 `:39-91`），初始化时间随参数数和模型大小增长。
- SHARDED_STATE_DICT merge 是 CPU-bound，源码注释也直接说明 “This is a CPU-bound process”（`src/axolotl/cli/merge_sharded_fsdp_weights.py:113-118`）。
- 多节点若把 `dp_shard` 跨慢网络，FSDP 的 per-layer all-gather 会比 HSDP 预期慢很多。

## 10.4 已知优化点

- `CheckpointSaveMixin` 中有 TODO：FSDP2 optimizer saving 仍需修复（`src/axolotl/core/trainers/mixins/checkpoints.py:13-22`）。
- `save_trained_model()` 对 sharded final save 的注释提到 Transformers / HF 上游待修复（`src/axolotl/train.py:334-349`）。
- `fsdp2_prepare_model()` 里对 PEFT ParamWrapper 有 TODO：是否需要单独 shard LoRA 参数仍待 review（`src/axolotl/monkeypatch/accelerate/fsdp2.py:240-243`）。
- FSDP2 + QLoRA 对 Params4bit 的 meta 移动仍有临时 workaround，遇到 Params4bit 时可能绕过 meta 节省而导致 VRAM spike（`src/axolotl/monkeypatch/accelerate/fsdp2.py:362-375`）。
- FP8 all-gather 已有配置和 patch（`src/axolotl/monkeypatch/trainer_accelerator_args.py:41-83`），但只在特定硬件 / torchao / torch.compile 组合下有收益，测试是 smoke 级（`tests/e2e/multigpu/test_fp8_fsdp2.py:51-99`）。

> 💡 **小结**
>
> * FSDP2 是当前主线，但仍依赖多处上游内部实现 patch。
> * 保存和 optimizer state 是比 forward 更容易被忽视的短板。
> * 后续优化方向集中在减少保存期聚合、修复 optimizer resume、降低 patch 对源码文本的依赖。

# 小结与展望

Axolotl 的 FSDP 实现可以用几个关键词概括。

## 关键词一：配置翻译

Axolotl 不重新实现 FSDP，而是把 YAML 翻译成 Accelerate env、TrainingArguments 和 parallelism config。`setup_fsdp_envs()`、`setup_parallelism_envs()`、`_set_base_training_args()` 共同构成了这条翻译链。

## 关键词二：延迟 Sharding

模型先被加载和 patch，真正 `fully_shard()` 延迟到 Accelerate prepare 阶段。这让 Axolotl 可以在 shard 前处理 device_map、量化、LoRA、activation checkpointing 和 meta loading。

## 关键词三：rank0 loading + meta 占位

`cpu_ram_efficient_loading` 是大模型初始化的关键：rank0 持有真实权重，其他 rank 用 meta 占位，再通过 `fsdp2_load_full_state_dict()` 分发到 sharded 参数。这节省 CPU RAM，但引入 tied keys、missing keys、broadcast 与保存复杂度。

## 关键词四：patch 驱动兼容

FSDP2 + QLoRA、FP8 all-gather、Accelerate FSDP2 prepare、Transformers meta tied keys 都通过 monkey patch 补齐。这是 Axolotl 适配新训练技术的速度来源，也是维护成本来源。

## 关键词五：通信换显存

FSDP 的收益来自参数、梯度、optimizer state 分片；代价是 forward/backward 中更频繁的 gather / reduce-scatter，以及保存阶段 full-gather 或 DCP merge 的成本。单机高速互联更适合大 shard group，多节点更适合 HSDP：节点内 shard，节点间 replicate。

总体来看，Axolotl 的 FSDP 实现适合：

- 单机多卡训练中大型模型全参 / LoRA / QLoRA；
- 多节点训练中明确知道 rank 排布、能配置 HSDP 的工程团队；
- 愿意接受保存 / merge / patch 复杂度，以换取显存可行性的训练任务。

它不适合：

- 网络拓扑不可控、跨节点带宽较弱但仍想把 FSDP shard group 拉满的场景；
- 强依赖稳定 resume optimizer state 的长训练任务，除非已验证当前组合可保存；
- 不愿承担上游版本升级带来的 monkey patch 维护风险的生产环境。

后续值得继续走读的方向有三个：

1. **Sequence / Context Parallelism 与 FSDP 的组合**：尤其是 CP 如何改变 batch dispatch、attention 通信和 loss gather；
2. **Cut Cross Entropy / Liger / FP8 与 FSDP 的组合**：这些机制才真正影响 logits / loss 显存；
3. **FSDP checkpoint resume 语义**：包括 optimizer state、scheduler state、sharded DCP 和 final HF checkpoint 的一致性。

