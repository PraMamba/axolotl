# Axolotl 源码走读：FSDP checkpoint resume 语义实现解析

在大模型微调里，FSDP 往往先被理解成“把参数、梯度和 optimizer state 分片”的显存优化手段。但真正跑过长任务的人很快会发现：**能不能省显存只是第一层问题，能不能从中间 checkpoint 准确恢复训练，才是工程上决定它是否可靠的第二层问题**。

对 Axolotl 来说，FSDP checkpoint/resume 不是一个单点功能，而是一条横跨配置归一化、Accelerate 环境变量、模型加载 patch、Transformers Trainer 保存策略、PyTorch Distributed Checkpoint（DCP）以及最终 HuggingFace 格式导出的链路。本文不展开 FSDP 理论本身，而是沿着源码回答一个更实际的问题：当用户在 Axolotl 中开启 FSDP 后，中间 checkpoint、optimizer state、scheduler state、sharded DCP，以及最终可发布的 HF checkpoint，究竟如何保持一致？哪些地方又并不保证一致？

# 前言

## 业务 / 工程背景

FSDP 出现在 Axolotl 的多 GPU 训练场景中。它的目标不是让训练逻辑看起来更优雅，而是让模型参数、梯度和 optimizer state 不再在每张 GPU 上完整复制，从而让单机多卡或多机多卡承载更大的全参微调、LoRA/QLoRA 微调与部分混合并行组合。

但是 checkpoint 语义把这个问题复杂化了：

- **训练恢复**需要恢复模型权重、optimizer state、scheduler state、scaler、RNG、TrainerState 等状态；
- **分布式保存**希望每个 rank 只保存自己负责的 shard，避免 rank0 聚合成为内存和 I/O 瓶颈；
- **最终发布**又希望产物是普通 `from_pretrained()` 可读取的 `model.safetensors` / `adapter_model.safetensors`，而不是 DCP 分片目录。

这三个目标天然冲突：**resume 想要“状态完整”，FSDP 想要“状态分片”，HF checkpoint 想要“权重合并”**。

## 核心矛盾

本文围绕三个工程矛盾展开：

1. **FSDP 分片 vs. resume 完整性**：训练时参数和 optimizer state 是分片状态，但 resume 必须恢复到与中断前等价的模型、优化器和 scheduler 状态。
2. **SHARDED_STATE_DICT vs. HF 最终格式**：DCP 分片适合分布式恢复，却不能直接当作普通 HuggingFace 模型发布。
3. **显存/CPU 内存节省 vs. 初始化与保存通信**：`cpu_ram_efficient_loading`、rank0-only loading、DCP 分片保存都在省内存，但它们会把复杂度转移到广播、all-gather、metadata 和 patch 维护上。

## 本文主线

文章按机制而不是按文件展开：

1. 用户入口与配置归一化：用户如何开启 FSDP checkpoint 语义；
2. 初始化与 patch：FSDP2 为什么在训练前就要替换 Accelerate 和 Transformers 行为；
3. 中间 checkpoint 保存：模型、optimizer、scheduler、RNG 分别保存在哪里；
4. resume 恢复：从 `resume_from_checkpoint` 到恢复全训练状态；
5. sharded DCP 与 final HF checkpoint：训练恢复格式和最终发布格式如何转换；
6. 状态流、rank 流和通信流：FSDP checkpoint 没有 batch shape 变化，但有状态 shape 和 rank 语义；
7. 核心机制深挖：patch、DCP、配置路径的隐藏假设；
8. 显存、性能与通信分析；
9. 配置项、边界条件与坑点；
10. 测试、示例与覆盖缺口；
11. 设计评价与展望。

## 不展开的内容

本文不讲 FSDP 的数学原理，不讲 ZeRO/FSDP 的完整论文背景，也不解释 HuggingFace Trainer 的所有通用 checkpoint 逻辑。只有当这些下游库逻辑直接影响 Axolotl 的 FSDP checkpoint/resume 语义时，才会引用本机安装依赖源码作为依据。本文源码观察基于当前仓库 `/root/axolotl`，以及本地依赖 `transformers 5.3.0`、`accelerate 1.13.0`、`torch 2.10.0+cu129`。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/main.py` | `axolotl train` 与 `merge-sharded-fsdp-weights` 的用户 CLI 入口 |
| `src/axolotl/cli/config.py` | YAML/CLI 配置加载、校验、环境变量准备与归一化入口 |
| `src/axolotl/utils/schemas/fsdp.py` | FSDP 配置 schema，定义 `state_dict_type` / `final_state_dict_type` 等字段 |
| `src/axolotl/utils/schemas/validation.py` | FSDP 版本、optimizer、量化、旧字段前缀等兼容校验 |
| `src/axolotl/utils/trainer.py` | 将 Axolotl 配置写入 Accelerate/FSDP 环境变量 |
| `src/axolotl/loaders/model.py` | 模型加载、FSDP device_map / cpu_ram_efficient_loading 初始化策略 |
| `src/axolotl/loaders/patch_manager.py` | 在模型加载前注入 FSDP2、parallelism、QLoRA 等 patch |
| `src/axolotl/monkeypatch/accelerate/fsdp2.py` | FSDP2 prepare、rank0 full state broadcast、`Accelerator.get_state_dict` patch |
| `src/axolotl/train.py` | 主训练流程、resume 入口、最终模型保存与 sharded merge |
| `src/axolotl/cli/merge_sharded_fsdp_weights.py` | 将 DCP sharded FSDP 权重转换成 HF safetensors |

# 一、入口与配置归一化：用户配置如何变成 FSDP checkpoint 语义

## 1.1 设计哲学与核心问题

FSDP checkpoint 语义不是在保存时才决定的。用户看见的是 YAML：

```yaml
fsdp_version: 2
fsdp_config:
  state_dict_type: SHARDED_STATE_DICT
  final_state_dict_type: FULL_STATE_DICT
  cpu_ram_efficient_loading: true
```

但真正影响行为的，是这份配置经过 schema、validation、环境变量、TrainingArguments、Accelerate FSDP plugin 之后形成的运行时状态。也就是说，Axolotl 在这里解决的是一个**配置到下游库行为的语义映射问题**：用户配置并不直接调用 DCP，也不直接调用 FSDP save/load，而是被翻译为 Transformers Trainer 与 Accelerate FSDP plugin 能识别的状态。

如果这一层没有做好，后面会出现非常隐蔽的问题：同一个字段在旧配置中叫 `fsdp_state_dict_type`，在 FSDP2 迁移后叫 `state_dict_type`；`save_only_model` 看起来只是保存优化，但它会直接破坏 resume；`final_state_dict_type` 只影响训练结束后的最终保存，不等价于中间 checkpoint 的格式。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：用户命令入口，负责选择 launcher 并进入 axolotl.cli.train
  - merge_sharded_fsdp_weights：sharded FSDP 权重合并入口

src/axolotl/cli/train.py
  - do_cli：加载配置并进入 do_train
  - do_train：加载数据集后调用 axolotl.train.train

src/axolotl/cli/config.py
  - load_cfg：读取 YAML、CLI override、validate_config、prepare_optim_env、normalize_config

src/axolotl/utils/schemas/fsdp.py
  - FSDPConfig：定义 FSDP 字段，包括 checkpoint state dict 类型

src/axolotl/utils/schemas/validation.py
  - check_fsdp_config_kwargs_prefix：迁移旧式 fsdp_ 前缀
  - check_fsdp_version_in_fsdp_config：统一 fsdp_version 来源
  - check_fsdp2_w_8bit_optimizer：拒绝不兼容 optimizer
```

## 1.3 主流程拆解

用户最常见的入口是：

```text
User: axolotl train config.yml
  -> src/axolotl/cli/main.py:train(...)
    -> src/axolotl/cli/train.py:do_cli(config, **kwargs)
      -> src/axolotl/cli/config.py:load_cfg(config, **kwargs)
        -> validate_config(...)
        -> prepare_optim_env(cfg)
        -> normalize_config(cfg)
      -> do_train(parsed_cfg, parsed_cli_args)
        -> axolotl.train.train(cfg, dataset_meta)
```

`src/axolotl/cli/main.py:78-125` 定义 `train` click 命令，支持 `accelerate`、`torchrun`、`python` 三类 launcher。真正加载配置的入口在 `src/axolotl/cli/train.py:55-91`：`do_cli()` 调用 `load_cfg()`，然后进入 `do_train()`。

配置加载的关键顺序在 `src/axolotl/cli/config.py:229-346`：

```text
load_cfg
  -> 读取 YAML / 远程配置
  -> 应用 CLI override
  -> prepare_plugins
  -> validate_config
  -> prepare_debug_log
  -> prepare_optim_env
  -> normalize_config
  -> normalize_cfg_datasets
  -> plugin_set_cfg
```

这个顺序很重要：`validate_config()` 先把字段合法性和兼容性确定下来；`prepare_optim_env()` 再根据 FSDP/DeepSpeed 配置写环境变量；`normalize_config()` 最后根据 world size、rank、dtype、batch size 等信息改写运行时配置。

FSDP schema 本身在 `src/axolotl/utils/schemas/fsdp.py:10-76`：

```python
class FSDPConfig(BaseModel):
    fsdp_version: int | None = Field(...)
    offload_params: bool | None = Field(...)
    cpu_ram_efficient_loading: bool | None = Field(...)
    state_dict_type: Literal[
        "FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"
    ] | None = Field(...)
    final_state_dict_type: Literal[
        "FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"
    ] | None = Field(...)
```

这里的 `state_dict_type` 是中间 checkpoint 与默认最终保存的核心字段；`final_state_dict_type` 则允许训练结束时覆盖一次最终保存格式。二者分工在 `save_trained_model()` 里才真正体现，后文会展开。

配置归一化还处理旧式字段。`src/axolotl/utils/schemas/validation.py:1052-1070` 会把 `fsdp_config` 内部的 `fsdp_state_dict_type`、`fsdp_reshard_after_forward` 等旧字段去掉 `fsdp_` 前缀；`src/axolotl/utils/schemas/validation.py:1072-1085` 则把 `fsdp_version` 从 top-level 或 `fsdp_config.version` 统一到两处。

此外，`normalize_config()` 会按分布式 world size 改写 batch 语义。`src/axolotl/utils/config/__init__.py:122-143` 中：

```text
cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
...
if cfg.world_size != 1:
    cfg.device_map = {"": LOCAL_RANK}
    if cfg.fsdp or cfg.fsdp_config or cfg.ddp:
        effective_world_size = world_size // cp_size // tp_size
        cfg.batch_size = cfg.batch_size * effective_world_size
```

也就是说，FSDP 开启后，用户配置的 batch 会被转换为全局意义下的 batch。这个变化不直接保存到 checkpoint 文件名里，但会影响 Trainer state、scheduler step 计算和 resume 后的训练节奏。

## 1.4 关键细节与误区澄清

> 误区一：`fsdp_config.state_dict_type` 只影响最终保存。

正确结论：它首先影响**中间 checkpoint**，最终保存默认复用它，但可以被 `fsdp_config.final_state_dict_type` 覆盖。源码依据是 `src/axolotl/train.py:294-300`：如果 `final_state_dict_type` 存在就用它，否则退回 `state_dict_type`。

> 误区二：旧配置里的 `fsdp_state_dict_type` 已经完全无效。

正确结论：它在 `fsdp_config` 内部仍会被 validation 迁移成 `state_dict_type`。源码依据是 `src/axolotl/utils/schemas/validation.py:1052-1070`，测试依据是 `tests/test_normalize_config.py:170-196`。

> 误区三：top-level `fsdp_final_state_dict_type` 会自然生效。

谨慎结论：schema 中确实存在 top-level `fsdp_final_state_dict_type`（`src/axolotl/utils/schemas/config.py:945-950`），但主保存路径 `save_trained_model()` 只读取 `cfg.fsdp_config.final_state_dict_type`（`src/axolotl/train.py:296-300`）。在本次源码检索中未确认 top-level 字段被迁移到 `fsdp_config.final_state_dict_type`，因此应优先使用 `fsdp_config.final_state_dict_type`。

## 1.5 本章小结

> 💡 **小结**
>
> * Axolotl 的 FSDP checkpoint 语义从 `load_cfg()` 就开始确定，而不是保存时才临时决定。
> * `state_dict_type` 控制中间 checkpoint 格式，`final_state_dict_type` 只在最终保存阶段覆盖。
> * 旧式 `fsdp_` 前缀字段会被迁移，但 top-level deprecated 字段不应作为可靠主路径使用。

# 二、初始化与 patch：为什么 resume 语义从训练前就决定

## 2.1 设计哲学与核心问题

FSDP checkpoint/resume 的难点之一，是“保存和加载”并不只发生在保存和加载函数里。FSDP2 的包装方式、rank0-only loading、meta device、DTensor、QLoRA 参数类型，都会决定后续 checkpoint 能不能恢复。

Axolotl 在初始化阶段要解决三个问题：

1. 告诉 Accelerate：当前训练要使用 FSDP，且 FSDP version/state_dict/offload 等设置是什么；
2. 在模型真正加载前安装必要 patch，让 Transformers/Accelerate 的默认行为能处理 FSDP2；
3. 对 `cpu_ram_efficient_loading` 这种 rank0-only 加载方式，提前安排好 meta device、broadcast 和 tied weights 修复。

这层解决的是**初始化状态一致性问题**。如果模型一开始加载方式不对，后续 resume 即使找到 checkpoint，也可能因为参数在 meta、rank 权重不同步、量化参数元数据丢失而失败。

## 2.2 源码入口与关键对象

```text
src/axolotl/utils/trainer.py
  - prepare_optim_env：根据 fsdp_config 写 Accelerate/FSDP 环境变量
  - setup_fsdp_envs：写 FSDP_VERSION、FSDP_STATE_DICT_TYPE 等

src/axolotl/loaders/model.py
  - ModelLoader.load：模型加载总入口
  - _apply_pre_model_load_setup：设置 parallelism/device_map/quantization
  - _build_model：处理 FSDP cpu_ram_efficient_loading 与 rank device_map

src/axolotl/loaders/patch_manager.py
  - apply_pre_model_load_patches：模型加载前 patch 总入口
  - _apply_fsdp_patches：FSDP2 / parallelism / TRL prepare patch
  - _apply_fsdp2_bnb_patches：QLoRA + FSDP2 量化参数 patch

src/axolotl/monkeypatch/accelerate/fsdp2.py
  - fsdp2_prepare_model：FSDP2 fully_shard 主逻辑
  - fsdp2_load_full_state_dict：rank0 full state -> sharded model
  - get_state_dict：替换 Accelerator.get_state_dict
```

## 2.3 主流程拆解

配置环境变量阶段在 `src/axolotl/utils/trainer.py:589-651`：

```text
prepare_optim_env(cfg)
  -> if cfg.fsdp or cfg.fsdp_config:
       cfg.fsdp = True if not cfg.fsdp else cfg.fsdp
       setup_fsdp_envs(cfg)
```

`setup_fsdp_envs()` 写入：

```text
ACCELERATE_USE_FSDP=true
FSDP_VERSION=2                      # fsdp_version == 2
FSDP_ACTIVATION_CHECKPOINTING=true   # activation_checkpointing
FSDP_OFFLOAD_PARAMS=true             # offload_params
FSDP_CPU_RAM_EFFICIENT_LOADING=true  # cpu_ram_efficient_loading
FSDP_STATE_DICT_TYPE=...             # state_dict_type
FSDP_AUTO_WRAP_POLICY=...
FSDP_TRANSFORMER_CLS_TO_WRAP=...
FSDP_RESHARD_AFTER_FORWARD=true
```

源码在 `src/axolotl/utils/trainer.py:589-618`。注意，这里不是直接构造 FSDP 对象，而是通过环境变量影响 Accelerate 的后续初始化。

模型加载阶段从 `src/axolotl/train.py:54-84` 的 `setup_model_and_tokenizer()` 进入 `ModelLoader(cfg, tokenizer).load()`。`ModelLoader.load()` 的主流程在 `src/axolotl/loaders/model.py:161-194`：

```text
ModelLoader.load
  -> patch_manager.apply_pre_model_load_patches()
  -> _apply_pre_model_load_setup()
  -> PLUGIN_MANAGER.pre_model_load
  -> _build_model()
  -> patch_manager.apply_post_model_build_patches(model)
  -> _apply_post_model_load_setup()
  -> _load_adapters()
  -> _apply_post_lora_load_setup()
```

FSDP patch 发生在模型加载前。`PatchManager.apply_pre_model_load_patches()` 在 `src/axolotl/loaders/patch_manager.py:95-122` 调用 `_apply_fsdp_patches()`。具体逻辑在 `src/axolotl/loaders/patch_manager.py:270-307`：

```text
if cfg.fsdp_config:
    patch_initialize_missing_keys_for_fsdp()

if context_parallel_size > 1 or (fsdp_config and fsdp_version == 2):
    patch_parallelism_config()

if fsdp_config and fsdp_version == 2:
    patch_accelerate_fsdp2()
    if cpu_ram_efficient_loading:
        patch_tied_keys_for_meta_device()
    if cfg.rl:
        patch_trl_prepare_fsdp2()
```

`patch_accelerate_fsdp2()` 替换两个关键入口，源码在 `src/axolotl/monkeypatch/accelerate/fsdp2.py:529-538`：

```python
accelerate.accelerator.fsdp2_prepare_model = fsdp2_prepare_model
accelerate.Accelerator.get_state_dict = get_state_dict
```

这说明 Axolotl 的 FSDP2 行为不是完全使用 Accelerate 默认实现，而是在模型 prepare 与最终 full state 导出上接管了一部分语义。

`cpu_ram_efficient_loading` 的路径更值得注意。`src/axolotl/loaders/model.py:756-780` 中：

```text
if FSDP and cpu_ram_efficient_loading:
    skip_move_to_device = True
    if fsdp_version == 2 and not tensor_parallel:
        local_rank == 0 -> device_map = "cpu"
        local_rank != 0 -> device_map = "meta"
```

这意味着 rank0 加载真实 CPU 权重，其他 rank 先拿 meta 参数，后面再通过 FSDP2 broadcast/distribute 填充真实 shard。

FSDP2 wrapping 的核心在 `fsdp2_prepare_model()`，源码 `src/axolotl/monkeypatch/accelerate/fsdp2.py:279-449`。其中 `cpu_ram_efficient_loading` 的关键流程是：

```text
original_sd = model.state_dict()
if cpu_ram_efficient_loading and not Params4bit:
    保存 non-persistent buffers
    model.to("meta")
    model.tie_weights()

fully_shard(child modules)
fully_shard(model)

if cpu_ram_efficient_loading:
    fsdp2_load_full_state_dict(accelerator, model, original_sd, offload_to_cpu)

if cpu_ram_efficient_loading and not Params4bit:
    恢复 non-persistent buffers
    model.tie_weights()
```

真正把 rank0 full state 分发到所有 rank 的函数是 `fsdp2_load_full_state_dict()`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:20-97`）：

```text
for param_name, sharded_meta_param in model.state_dict().items():
    rank0: full_tensor = full_sd[param_name]
    if DTensor / device_mesh:
        distribute_tensor(full_tensor, device_mesh, placements, src_data_rank=0)
    else:
        dist.broadcast(sharded_param, src=0)
    model.load_state_dict(sharded_sd, assign=True, strict=True)
```

这个函数的副作用很明确：它不只是“加载权重”，而是把 rank0 的 full tensor 按 FSDP2 的 sharding placement 变成各 rank 的 local shard。

## 2.4 关键细节与误区澄清

> 误区四：`cpu_ram_efficient_loading` 只是降低 CPU 内存，不影响 checkpoint/resume。

正确结论：它直接影响初始化权重如何进入 FSDP sharded model。rank0 加载 CPU 权重，其他 rank 使用 meta；随后 `fsdp2_load_full_state_dict()` 通过 `distribute_tensor(..., src_data_rank=0)` 或 `dist.broadcast()` 分发真实权重。源码依据是 `src/axolotl/loaders/model.py:756-780` 与 `src/axolotl/monkeypatch/accelerate/fsdp2.py:20-97`。

> 误区五：`src/axolotl/monkeypatch/trainer_fsdp_optim.py` 是当前 optimizer save 主路径。

正确结论：不是。`PatchManager._apply_fsdp_patches()` 中关于 `patch_training_loop_for_fsdp()` 的调用被注释掉了（`src/axolotl/loaders/patch_manager.py:301-307`）。当前中间 checkpoint 的 optimizer 保存主路径来自 Transformers/Accelerate 的 `save_fsdp_optimizer()`，Axolotl 只通过 `CheckpointSaveMixin` 捕获保存失败并给出 warning。

> 误区六：FSDP2 + QLoRA 只是普通量化加载。

正确结论：不是。`src/axolotl/loaders/patch_manager.py:590-608` 在 FSDP2 且 `load_in_4bit/load_in_8bit` 时会 patch FSDPParam 的 sharded/unsharded 参数构造、dtype attrs、8bit Linear 保存逻辑。`src/axolotl/monkeypatch/fsdp2_qlora.py:1-7` 明确说明这些 patch 用来保留 bitsandbytes 参数的 quantization metadata。

## 2.5 本章小结

> 💡 **小结**
>
> * FSDP checkpoint/resume 的可靠性从模型加载前就开始决定。
> * Axolotl 通过环境变量驱动 Accelerate，通过 monkey patch 修正 FSDP2 prepare 与 full state 导出。
> * `cpu_ram_efficient_loading` 是 rank0 full state 到各 rank shard 的初始化协议，不只是一个内存开关。

# 三、中间 checkpoint：optimizer、scheduler、RNG 的可恢复边界

## 3.1 设计哲学与核心问题

训练中间 checkpoint 和最终模型保存是两件事。中间 checkpoint 的目标是：**从某一步继续训练，尽可能恢复与中断前一致的训练状态**。因此它不能只保存模型权重，还要保存 optimizer、scheduler、scaler、RNG、TrainerState。

FSDP 让这个目标变复杂：模型权重和 optimizer state 可能是 full state，也可能是 sharded DCP；scheduler state 通常是普通小文件；RNG 在分布式环境下每个 rank 都有自己的文件。Axolotl 的策略是尽量沿用 Transformers Trainer 的 checkpoint 语义，并对 FSDP 失败场景做容错。

## 3.2 源码入口与关键对象

```text
/usr/local/lib/python3.12/dist-packages/transformers/trainer.py
  - Trainer._save_checkpoint：中间 checkpoint 总入口
  - Trainer._save_optimizer_and_scheduler：optimizer/scheduler 保存主逻辑
  - Trainer.save_model：checkpoint 中模型权重保存入口

/usr/local/lib/python3.12/dist-packages/accelerate/utils/fsdp_utils.py
  - save_fsdp_model：FSDP model state 保存
  - save_fsdp_optimizer：FSDP optimizer state 保存

src/axolotl/core/trainers/mixins/checkpoints.py
  - CheckpointSaveMixin._save_optimizer_and_scheduler：保存失败时 warning 而不是中断训练

src/axolotl/utils/callbacks/dynamic_checkpoint.py
  - DynamicCheckpointCallback：文件触发的 on-demand checkpoint
```

## 3.3 主流程拆解

Transformers 的中间 checkpoint 入口在本地依赖 `transformers/trainer.py:3029-3080`：

```text
Trainer._save_checkpoint
  -> output_dir = checkpoint-{global_step}
  -> self.save_model(output_dir, _internal_call=True)
  -> if not save_only_model:
       self._save_optimizer_and_scheduler(output_dir)
       self._save_scaler(output_dir)
       self._save_rng_state(output_dir)
  -> self.state.save_to_json(trainer_state.json)
```

这个顺序解释了 checkpoint 目录为什么可能同时出现 HF 权重文件、FSDP 专用权重、optimizer 目录、scheduler 文件和 RNG 文件。

当 `self.is_fsdp_enabled` 为真时，`Trainer._save_optimizer_and_scheduler()` 在本地依赖 `transformers/trainer.py:3225-3232` 调用：

```python
save_fsdp_model(fsdp_plugin, accelerator, self.model, output_dir, **get_fsdp_ckpt_kwargs())
save_fsdp_optimizer(fsdp_plugin, accelerator, self.optimizer, self.model, output_dir)
```

然后在 `transformers/trainer.py:3237-3248` 保存 scheduler：

```python
torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
```

也就是说，FSDP 中间 checkpoint 的 optimizer state 和 scheduler state 不是同一个机制保存的：

```text
模型 FSDP state     -> accelerate.utils.fsdp_utils.save_fsdp_model
optimizer FSDP state -> accelerate.utils.fsdp_utils.save_fsdp_optimizer
scheduler state     -> torch.save(lr_scheduler.state_dict(), scheduler.pt)
RNG state           -> rng_state_{rank}.pth
Trainer state       -> trainer_state.json
```

Accelerate 的 `save_fsdp_model()` 在本地依赖 `accelerate/utils/fsdp_utils.py:103-158` 根据 `state_dict_type` 分两条路径：

```text
FULL_STATE_DICT:
  rank0 写 pytorch_model_fsdp.bin

SHARDED_STATE_DICT:
  所有 rank 通过 torch.distributed.checkpoint.save
  写 checkpoint-{step}/pytorch_model_fsdp_0/
```

optimizer 类似。`save_fsdp_optimizer()` 在 `accelerate/utils/fsdp_utils.py:233-278`：

```text
FSDP2:
  optim_state = get_optimizer_state_dict(model, optimizer, options=sd_options)

FULL_STATE_DICT:
  rank0 写 optimizer.bin

非 FULL（包括 SHARDED_STATE_DICT）:
  dist_cp.save({"optimizer": optim_state})
  写 checkpoint-{step}/optimizer_0/
```

因此，一个典型 FSDP2 `SHARDED_STATE_DICT` 中间 checkpoint 结构可以理解为：

```text
checkpoint-100/
  trainer_state.json
  scheduler.pt
  rng_state_0.pth
  rng_state_1.pth
  pytorch_model_fsdp_0/
    .metadata
    __0_0.distcp
    __1_0.distcp
    ...
  optimizer_0/
    .metadata
    __0_0.distcp
    __1_0.distcp
    ...
```

如果是 `FULL_STATE_DICT`，则中间 checkpoint 往往还会有 HF `model.safetensors` 以及 FSDP 专用 `pytorch_model_fsdp.bin`。原因是 `Trainer._save_checkpoint()` 先调用 `save_model()`，而 `Trainer.save_model()` 在 FSDP 且 plugin state_dict_type 包含 `FULL_STATE_DICT` 时会调用 `accelerator.get_state_dict()` 并 `_save()`（本地依赖 `transformers/trainer.py:3760-3764`）。随后 `_save_optimizer_and_scheduler()` 又保存 FSDP 专用 model/optimizer state。

Axolotl 自己对这条路径最重要的改动，是 `CheckpointSaveMixin`。`src/axolotl/core/trainers/mixins/checkpoints.py:10-22`：

```python
class CheckpointSaveMixin(Trainer):
    def _save_optimizer_and_scheduler(self, output_dir):
        try:
            super()._save_optimizer_and_scheduler(output_dir)
        except (NotImplementedError, KeyError) as exc:
            LOG.warning_once(
                "Optimizer and scheduler states were not saved - resuming ... will not be possible."
            )
```

这个 mixin 的意义不是让 resume 更强，而是让“保存 optimizer/scheduler 失败”不至于直接杀死训练。代价也很明确：一旦进入 except，本次 checkpoint 就不是完整可恢复 checkpoint。

动态 checkpoint 也复用同一套 Trainer 保存机制。`DynamicCheckpointCallback` 在 `src/axolotl/utils/callbacks/dynamic_checkpoint.py:64-128` 中，由 rank0 检查触发文件，分布式下 broadcast 一个 trigger tensor，然后设置 `control.should_save = True`。真正保存仍然由 Trainer 的 `_save_checkpoint()` 完成。

## 3.4 关键细节与误区澄清

> 误区七：只要目录叫 `checkpoint-*` 就一定可 resume。

正确结论：不一定。若 `save_only_model: true`，Transformers `_save_checkpoint()` 会跳过 optimizer/scheduler/scaler/RNG 保存（本地依赖 `transformers/trainer.py:3061-3067`）。Axolotl schema 也明确说明 `save_only_model` 会导致不能 resume（`src/axolotl/utils/schemas/config.py:1103-1107`）。

> 误区八：FSDP optimizer state 保存失败会让训练立刻失败。

正确结论：在 Axolotl 的 trainer mixin 下，`NotImplementedError` / `KeyError` 会被捕获并警告，训练继续，但该 checkpoint 不能用于完整 resume。源码依据是 `src/axolotl/core/trainers/mixins/checkpoints.py:13-22`。

> 误区九：动态 checkpoint 是另一种 checkpoint 格式。

正确结论：不是。动态 checkpoint 只是把 `control.should_save` 置为 True，最终仍进入 Transformers Trainer 的 `_save_checkpoint()` 主路径。源码依据是 `src/axolotl/utils/callbacks/dynamic_checkpoint.py:123-128`。

## 3.5 本章小结

> 💡 **小结**
>
> * 中间 checkpoint 的 resume 能力来自模型、optimizer、scheduler、RNG、TrainerState 的组合，而不是单一权重文件。
> * FSDP model/optimizer state 由 Accelerate FSDP utils 保存；scheduler 仍是普通 `scheduler.pt`。
> * Axolotl 对 optimizer/scheduler 保存失败采取“不中断训练但警告不可 resume”的策略。

# 四、resume：从 `resume_from_checkpoint` 到恢复全状态

## 4.1 设计哲学与核心问题

Resume 的目标不是“加载一个模型”，而是**恢复一个训练进程**。这意味着它必须在 Trainer 已经构建好 optimizer、scheduler、FSDP wrapper、Accelerate state 之后，把 checkpoint 中的状态写回对应对象。

这一章要澄清一个关键点：Axolotl 的 `ModelLoader` 负责加载 base model 和 adapter，但中间 checkpoint resume 的模型权重加载不是在 `ModelLoader.load()` 主路径完成的，而是由 `trainer.train(resume_from_checkpoint=...)` 进入 Transformers Trainer 的恢复逻辑完成。

## 4.2 源码入口与关键对象

```text
src/axolotl/utils/train.py
  - determine_last_checkpoint：显式/自动 resume 路径选择

src/axolotl/train.py
  - train：训练总入口，调用 determine_last_checkpoint
  - execute_training：把 resume_from_checkpoint 传给 trainer.train

src/axolotl/core/builders/base.py
  - get_callbacks：resume 时添加 SkipEvalOnResumeCallback

src/axolotl/utils/callbacks/__init__.py
  - SkipEvalOnResumeCallback：跳过 resume step 的重复 eval

/usr/local/lib/python3.12/dist-packages/transformers/trainer.py
  - _load_from_checkpoint：恢复模型权重
  - _load_optimizer_and_scheduler：恢复 optimizer/scheduler

src/axolotl/core/trainers/mixins/rng_state_loader.py
  - RngLoaderMixin._load_rng_state：安全加载 rank RNG state
```

## 4.3 主流程拆解

Axolotl 的 resume 路径从 `src/axolotl/train.py:624-627` 开始：

```python
resume_from_checkpoint = determine_last_checkpoint(cfg)
execute_training(cfg, trainer, resume_from_checkpoint)
```

`determine_last_checkpoint()` 在 `src/axolotl/utils/train.py:11-47` 做两件事：

1. 扫描 `output_dir/checkpoint-*`，按数字后缀选择最大 step；
2. 如果 `cfg.resume_from_checkpoint is None` 且 `auto_resume_from_checkpoints` 为真，就把 last checkpoint 写回 `cfg.resume_from_checkpoint`。

伪代码如下：

```text
checkpoints = sorted(output_dir.glob("checkpoint-*"), key=step)
last_checkpoint = checkpoints[-1] if exists else None

if update=False:
    return last_checkpoint

if cfg.resume_from_checkpoint is None and cfg.auto_resume_from_checkpoints and last_checkpoint:
    cfg.resume_from_checkpoint = last_checkpoint

return cfg.resume_from_checkpoint
```

随后 `execute_training()` 在 `src/axolotl/train.py:183-228` 进入：

```python
trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

这一步以后，恢复逻辑主要在 Transformers Trainer 内部。模型权重恢复入口是本地依赖 `transformers/trainer.py:3273-3400` 的 `_load_from_checkpoint()`。它先判断 checkpoint 是否为 FSDP checkpoint：

```text
is_fsdp_ckpt = checkpoint dir 中存在 pytorch_model_fsdp_* 目录
            或存在 pytorch_model_fsdp.bin
```

源码在本地依赖 `transformers/trainer.py:3285-3294`。如果发现 FSDP checkpoint 但当前 Trainer 没开启 FSDP，会直接报错（`transformers/trainer.py:3310-3311`）。

FSDP 模型恢复调用：

```text
_load_from_checkpoint
  -> if self.is_fsdp_enabled:
       load_fsdp_model(fsdp_plugin, accelerator, model, resume_from_checkpoint, ...)
```

源码在本地依赖 `transformers/trainer.py:3348-3355`。

optimizer/scheduler 恢复在本地依赖 `transformers/trainer.py:3553-3655`。关键判断是：

```text
checkpoint_file_exists =
  optimizer.pt exists
  or optimizer.bin exists
  or checkpoint 内存在 optimizer_0 这样的目录

if checkpoint_file_exists and scheduler.pt exists:
  if self.is_fsdp_enabled:
      load_fsdp_optimizer(...)
  else:
      torch.load(optimizer.pt)
  torch.load(scheduler.pt)
```

这说明 FSDP sharded optimizer 恢复依赖 `optimizer_0/` 目录，同时 scheduler 仍依赖普通 `scheduler.pt` 文件。如果 optimizer DCP 目录存在但 scheduler 文件缺失，`_load_optimizer_and_scheduler()` 不会进入完整恢复分支。

RNG 恢复由 Axolotl 自己覆盖。`RngLoaderMixin` 在 `src/axolotl/core/trainers/mixins/rng_state_loader.py:24-68` 中根据 world size 选择：

```text
world_size > 1 -> rng_state_{process_index}.pth
world_size <= 1 -> rng_state.pth
```

它还用 `transformers.trainer.safe_globals()` 包住 `torch.load()`（`src/axolotl/core/trainers/mixins/rng_state_loader.py:52-55`），这是为了兼容 PyTorch 2.6+ 的安全反序列化限制。

Resume 还有一个小但实用的行为：`TrainerBuilderBase.get_callbacks()` 在 `src/axolotl/core/builders/base.py:122-124` 中，如果 `cfg.resume_from_checkpoint` 存在，会添加 `SkipEvalOnResumeCallback()`。这个 callback 在 `src/axolotl/utils/callbacks/__init__.py:101-148` 中记录恢复时的 `global_step`，并跳过这一 step 的重复 eval，避免恢复后第一时间重跑已完成的 evaluation。

## 4.4 关键细节与误区澄清

> 误区十：resume 会在 `ModelLoader.load()` 阶段加载 checkpoint 权重。

正确结论：中间 checkpoint resume 的主路径在 `trainer.train(resume_from_checkpoint=...)` 之后，由 Transformers `_load_from_checkpoint()` 处理。`ModelLoader` 负责 base model / adapter 初始化和 FSDP prepare 前的 patch，不负责读取 `checkpoint-*` 中的 FSDP optimizer/scheduler 状态。

> 误区十一：FSDP resume 只要模型权重恢复即可。

正确结论：不够。完整 resume 还要求 optimizer、scheduler、RNG、TrainerState 等状态存在。特别是 FSDP sharded optimizer 位于 `optimizer_0/`，scheduler 仍是 `scheduler.pt`。源码依据是本地依赖 `transformers/trainer.py:3569-3655`。

> 误区十二：`auto_resume_from_checkpoints` 会校验 checkpoint 是否属于当前模型。

正确结论：不会。`determine_last_checkpoint()` 只按目录名数字后缀找最新 checkpoint（`src/axolotl/utils/train.py:22-47`）。schema 文档也提醒“between different models” 要小心（`src/axolotl/utils/schemas/config.py:205-209`）。

## 4.5 本章小结

> 💡 **小结**
>
> * Axolotl 只负责选择 resume 路径并传给 Trainer；模型/optimizer/scheduler 的实际恢复由 Transformers/Accelerate 完成。
> * FSDP sharded optimizer 恢复依赖 `optimizer_0/`，scheduler 恢复仍依赖 `scheduler.pt`。
> * `auto_resume_from_checkpoints` 是目录选择器，不是语义校验器。

# 五、Sharded DCP 与 final HF checkpoint：训练恢复和发布保存为什么不是一个格式

## 5.1 设计哲学与核心问题

中间 checkpoint 为 resume 服务，最终 checkpoint 为发布或推理服务。前者可以是分布式 DCP 目录，因为恢复时仍在同一个 FSDP/Accelerate 语境中；后者最好是普通 HuggingFace 格式，因为用户希望直接 `from_pretrained(output_dir)`。

Axolotl 在这里解决的是**格式桥接问题**：训练期间允许 `SHARDED_STATE_DICT`，但训练结束后尽量自动合并成 HF safetensors。如果自动合并失败，用户还可以调用 `merge-sharded-fsdp-weights` CLI 手动转换。

## 5.2 源码入口与关键对象

```text
src/axolotl/train.py
  - save_trained_model：训练结束后的最终保存入口
  - _rename_fsdp_merged_to_adapter：PEFT sharded merge 后改名为 adapter_model*

src/axolotl/cli/main.py
  - merge_sharded_fsdp_weights：CLI 转发到 axolotl.cli.merge_sharded_fsdp_weights

src/axolotl/cli/merge_sharded_fsdp_weights.py
  - merge_fsdp_weights：DCP -> HF safetensors
  - _distributed_checkpoint_to_merged_weights：实际读取 DCP 并 split safetensors
  - do_cli：从 output_dir 或 last checkpoint 自动找 pytorch_model_fsdp_0
```

## 5.3 主流程拆解

最终保存入口在 `src/axolotl/train.py:254-386`。FSDP 分支从 `src/axolotl/train.py:294` 开始：

```text
if trainer.is_fsdp_enabled or cfg.fsdp_config:
    if final_state_dict_type:
        state_dict_type = final_state_dict_type
    else:
        state_dict_type = state_dict_type
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type(state_dict_type)
    trainer.save_model(cfg.output_dir)

    if state_dict_type == "SHARDED_STATE_DICT":
        checkpoint_dir = determine_last_checkpoint(cfg, update=False)
        if output_dir/model.safetensors.index.json 不存在 and checkpoint_dir:
            merge_fsdp_weights(checkpoint_dir/pytorch_model_fsdp_0, output_dir/merged)
            move merged/* -> output_dir/
            if PEFT: model* -> adapter_model*
```

源码对应 `src/axolotl/train.py:294-333`。

这里有一个容易忽略的下游行为：Transformers `Trainer.save_model()` 在 FSDP 分支只有当 plugin state dict type 包含 `FULL_STATE_DICT` 时才真正保存 HF 权重。本地依赖 `transformers/trainer.py:3760-3764`：

```python
elif self.is_fsdp_enabled:
    if "FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type):
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.args.should_save:
            self._save(output_dir, state_dict=state_dict)
```

这解释了 Axolotl 代码里的注释：`trainer.save_model(cfg.output_dir)  # only handles FULL_STATE_DICT`（`src/axolotl/train.py:301`）。如果最终选择 `SHARDED_STATE_DICT`，`trainer.save_model()` 不会直接生成普通 HF 权重；Axolotl 后面会尝试从最近的 checkpoint 的 `pytorch_model_fsdp_0/` 自动 merge。

合并逻辑在 `src/axolotl/cli/merge_sharded_fsdp_weights.py:38-104`：

```text
_distributed_checkpoint_to_merged_weights(checkpoint_dir, save_path)
  -> dist_cp_format_utils._load_state_dict(..., FileSystemReader(checkpoint_dir), no_dist=True)
  -> 如果 state_dict 是 {"model": {...}}，展开内部 dict
  -> 将 tensor 转成 bfloat16
  -> split_torch_state_dict_into_shards(..., max_shard_size="5GB")
  -> safe_save_file(..., metadata={"format": "pt"})
  -> 如果分片，写 model.safetensors.index.json
```

手动 CLI 路径在 `src/axolotl/cli/main.py:245-274` 和 `src/axolotl/cli/merge_sharded_fsdp_weights.py:169-203`。`do_cli()` 会先找：

```text
output_dir/pytorch_model_fsdp_0
```

如果没有，再通过 `determine_last_checkpoint(parsed_cfg, update=False)` 找：

```text
output_dir/checkpoint-{last}/pytorch_model_fsdp_0
```

这和最终保存自动 merge 的路径略有差异：自动 merge 直接从 last checkpoint 下找 `pytorch_model_fsdp_0`（`src/axolotl/train.py:307-318`），手动 CLI 会先看 `output_dir/pytorch_model_fsdp_0`。

## 5.4 关键细节与误区澄清

> 误区十三：最终 `output_dir` 就是可 resume checkpoint。

正确结论：通常不是。`save_trained_model()` 的目标是保存最终模型权重，不保存 optimizer/scheduler/RNG/TrainerState。完整 resume 应指向 `checkpoint-*` 目录，而不是最终 `output_dir`。

> 误区十四：`SHARDED_STATE_DICT` 最终保存一定会自动生成 HF safetensors。

正确结论：源码是“尽量自动 merge”。它需要存在最近的 `checkpoint-*`，并且其中有 `pytorch_model_fsdp_0/`。如果没有中间 checkpoint（例如关闭保存策略），自动 merge 可能没有可用输入。源码依据是 `src/axolotl/train.py:307-318`。

> 误区十五：DCP merge 会同时合并 optimizer state。

正确结论：`merge_sharded_fsdp_weights.py` 的最终目标是权重 safetensors。虽然错误提示会识别 `optimizer_0` 目录（`src/axolotl/cli/merge_sharded_fsdp_weights.py:137-153`），但 CLI `do_cli()` 主路径寻找的是 `pytorch_model_fsdp_0`（`src/axolotl/cli/merge_sharded_fsdp_weights.py:180-184`）。不要把它理解成“恢复训练状态”的工具。

## 5.5 本章小结

> 💡 **小结**
>
> * 中间 checkpoint 面向 resume，最终 checkpoint 面向 HF 发布，两者格式目标不同。
> * `SHARDED_STATE_DICT` 下最终 HF 权重依赖 DCP merge，自动 merge 使用最近 checkpoint 的 `pytorch_model_fsdp_0/`。
> * `merge-sharded-fsdp-weights` 合并的是模型权重，不是 optimizer/scheduler 完整训练状态。

# 六、完整主路径串联

## 6.1 设计哲学与核心问题

前面几章把配置、初始化、保存、恢复、最终合并拆开讲。现在把它们串成一次真实调用：用户启动 FSDP 训练，训练中保存 checkpoint，随后从 checkpoint resume，训练结束再生成最终 HF 权重。

这条路径的核心不是“哪个函数调用哪个函数”，而是每层在什么时候修改状态：配置阶段写环境变量，模型加载阶段 patch，下游 Trainer 阶段保存/恢复训练状态，最终阶段再转换发布格式。

## 6.2 完整调用栈

```text
User: axolotl train config.yml
  │
  ├─ Step 1: CLI 与配置加载
  │     ├─ src/axolotl/cli/main.py:train
  │     ├─ src/axolotl/cli/train.py:do_cli
  │     └─ src/axolotl/cli/config.py:load_cfg
  │
  ├─ Step 2: FSDP 配置校验与环境变量
  │     ├─ src/axolotl/utils/schemas/validation.py
  │     └─ src/axolotl/utils/trainer.py:prepare_optim_env / setup_fsdp_envs
  │
  ├─ Step 3: 模型加载与 FSDP patch
  │     ├─ src/axolotl/loaders/model.py:ModelLoader.load
  │     ├─ src/axolotl/loaders/patch_manager.py:_apply_fsdp_patches
  │     └─ src/axolotl/monkeypatch/accelerate/fsdp2.py:fsdp2_prepare_model
  │
  ├─ Step 4: Trainer 构建与训练启动
  │     ├─ src/axolotl/core/builders/base.py:build_training_args
  │     ├─ src/axolotl/train.py:execute_training
  │     └─ trainer.train(resume_from_checkpoint=...)
  │
  ├─ Step 5: 中间 checkpoint 保存
  │     ├─ transformers.Trainer._save_checkpoint
  │     ├─ transformers.Trainer._save_optimizer_and_scheduler
  │     └─ accelerate.utils.fsdp_utils.save_fsdp_model / save_fsdp_optimizer
  │
  ├─ Step 6: resume 恢复
  │     ├─ src/axolotl/utils/train.py:determine_last_checkpoint
  │     ├─ transformers.Trainer._load_from_checkpoint
  │     ├─ transformers.Trainer._load_optimizer_and_scheduler
  │     └─ src/axolotl/core/trainers/mixins/rng_state_loader.py:_load_rng_state
  │
  └─ Step 7: 最终保存 / 合并
        ├─ src/axolotl/train.py:save_trained_model
        └─ src/axolotl/cli/merge_sharded_fsdp_weights.py:merge_fsdp_weights
```

## 6.3 每一层做了什么

| 层级 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| 配置加载 | YAML + CLI kwargs | `cfg.fsdp_config`、`cfg.fsdp_version`、batch/world/rank 归一化 | 无 | 无直接影响 | 进程启动一次 |
| 环境变量 | `cfg.fsdp_config` | `ACCELERATE_USE_FSDP`、`FSDP_STATE_DICT_TYPE` 等 | 无 | 决定后续加载和保存策略 | 进程启动一次 |
| 模型加载 patch | `cfg`、model config | 替换 Accelerate FSDP2 prepare/get_state_dict | 无 | 为 rank0/meta 加载和 full state 导出铺路 | 模型加载前一次 |
| FSDP prepare | model、Accelerator state | `fully_shard()` 后的 FSDP2 model | 初始化 broadcast / distribute | 参数变为 shard，降低常驻显存 | trainer prepare 阶段一次 |
| 训练 step | batch、model shard | loss、grad、optimizer update | FSDP all-gather / reduce-scatter | 参数/梯度/optimizer shard 节省显存 | 每 step / 每 FSDP unit |
| 中间保存 | trainer state | checkpoint-* 目录 | FULL 可能 all-gather；SHARDED DCP 写本地 shard | FULL rank0 CPU/内存峰值；SHARDED 更分散 | save_steps/epoch/动态触发 |
| resume | checkpoint-* | 恢复模型、optimizer、scheduler、RNG | SHARDED DCP load；FULL rank0 broadcast | 依 state_dict_type 变化 | 训练启动时一次 |
| final save | trained model | HF safetensors 或 sharded merge 后 safetensors | FULL all-gather 或 DCP no_dist merge | FULL 聚合峰值；merge CPU/I/O 开销 | 训练结束一次 |

## 6.4 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `src/axolotl/monkeypatch/trainer_fsdp_optim.py:patch_training_loop_for_fsdp` | 文件名写着 optimizer save fix | 当前未启用 | `PatchManager` 中调用被注释，主路径是 Transformers/Accelerate FSDP utils |
| `src/axolotl/train.py:setup_signal_handler` | Ctrl+C 会保存模型 | 不构成完整 resume checkpoint | 只调用 `model.save_pretrained(cfg.output_dir)`，文档说明不保存 optimizer state |
| `merge-sharded-fsdp-weights` CLI | 名字像“checkpoint 恢复工具” | 不是训练恢复路径 | 它把 DCP model shards 合并成 HF 权重，不恢复 optimizer/scheduler |
| `ModelLoader.load()` | 名字像加载所有模型状态 | 不加载 `checkpoint-*` 中的训练状态 | 中间 checkpoint resume 由 `trainer.train(resume_from_checkpoint=...)` 触发 |
| top-level `fsdp_final_state_dict_type` | schema 中存在 | 未在主保存函数中确认消费 | 应使用 `fsdp_config.final_state_dict_type` |

## 6.5 本章小结

> 💡 **小结**
>
> * Axolotl 的 FSDP resume 主路径是“配置/patch 初始化 + 下游 Trainer checkpoint 机制”的组合。
> * 中间 checkpoint 和最终模型保存是两个阶段，不能混为一谈。
> * 若要判断一个 checkpoint 是否可恢复，必须检查 optimizer、scheduler、RNG、TrainerState，而不是只看权重文件。

# 七、关键数据流 / 状态流 / rank 流程

## 7.1 设计哲学与核心问题

FSDP checkpoint/resume 不像序列并行那样改变 `input_ids: [batch, seq]` 的 shape。它改变的是**状态张量的存在形态**：训练中是 shard，保存时可能 all-gather 成 full，也可能以 DCP shard 写盘；恢复时再从 full 或 shard 回到每个 rank 的 local shard。

因此，本章的“shape”不是 batch shape，而是 parameter / optimizer state 的 shape 与 placement。

## 7.2 Tensor / state shape 变化

假设某个参数 `W` 的 full shape 是：

```text
W.full: [hidden, hidden]
numel = hidden * hidden
fsdp_shard_size = 4
```

训练中 FSDP2 近似状态：

```text
rank0: W.local_shard ≈ [numel / 4]
rank1: W.local_shard ≈ [numel / 4]
rank2: W.local_shard ≈ [numel / 4]
rank3: W.local_shard ≈ [numel / 4]

optimizer state:
  exp_avg.local_shard ≈ [numel / 4]
  exp_avg_sq.local_shard ≈ [numel / 4]
```

`FULL_STATE_DICT` 保存时：

```text
rank0:
  W.full: [hidden, hidden]   # CPU offload / rank0_only
  optimizer full state       # rank0 保存 optimizer.bin

rank1-3:
  不写 full 权重 / 或空 state_dict
```

`SHARDED_STATE_DICT` 保存时：

```text
checkpoint-100/pytorch_model_fsdp_0/
  .metadata
  rank0 shard files
  rank1 shard files
  rank2 shard files
  rank3 shard files

checkpoint-100/optimizer_0/
  .metadata
  optimizer state shard files
```

`cpu_ram_efficient_loading` 初始化时：

```text
加载后但 FSDP broadcast 前:
  rank0: full_sd[param] 在 CPU
  rank1-3: meta tensor / empty state

fsdp2_load_full_state_dict:
  full_sd[param] --distribute_tensor(src_data_rank=0)--> DTensor local shards

加载后:
  rank_i: param.local_shard
```

源码依据是 `src/axolotl/monkeypatch/accelerate/fsdp2.py:39-91`。

## 7.3 Rank / Mesh / Process Group 变化

Axolotl 构建 parallelism config 的入口在 `src/axolotl/utils/distributed.py:299-370`。当未显式配置 `dp_shard_size` / `dp_replicate_size`，但开启 FSDP 且 world size 大于 1 时，剩余 world size 会成为 `dp_shard_size`：

```text
world_size = 8
tensor_parallel_size = 1
context_parallel_size = 1
is_fsdp = true

remaining_world_size = 8
pc_kwargs["dp_shard_size"] = 8
```

如果组合 TP/CP/HSDP，则 `remaining_world_size` 会先除以 TP/CP，再用于 DP shard/replicate。这意味着 FSDP sharding group 不一定等于全局 world，尤其在 TP/CP/HSDP 组合时，checkpoint 的 shard 语义跟 device mesh 相关。

FSDP2 prepare 中 mesh 被传给 `fully_shard()`：

```python
"mesh": (
    mesh[tuple(accelerator.state.parallelism_config.fsdp_dim_names)]
    if mesh is not None
    else None
)
```

源码在 `src/axolotl/monkeypatch/accelerate/fsdp2.py:351-360`。这说明真正参与参数分片的是 FSDP 维度对应的 mesh，而不是所有并行维度。

## 7.4 状态切换

FSDP checkpoint 相关的全局/进程状态主要有三类：

```text
进入训练前:
  prepare_optim_env 写 os.environ
  PatchManager 替换 Accelerate / Transformers / PEFT / BnB 函数

Trainer 构建后:
  accelerator.state.fsdp_plugin 持有 state_dict_type、offload、auto_wrap 等状态

最终保存时:
  save_trained_model 临时 set_state_dict_type(final_state_dict_type or state_dict_type)
```

关键状态写入点：

- 环境变量：`src/axolotl/utils/trainer.py:589-618`；
- patch：`src/axolotl/loaders/patch_manager.py:270-307`；
- final state dict type：`src/axolotl/train.py:296-301`。

线程/进程安全方面，环境变量和 monkey patch 都是**进程内全局状态**。分布式训练每个 rank 是独立进程，因此 patch 不跨进程共享；但在同一 Python 进程内，它们没有自动恢复机制。FSDP2 QLoRA patch 使用 `_axolotl_patched` 标记避免重复 patch（如 `src/axolotl/monkeypatch/fsdp2_qlora.py:19-23`），但这不是上下文管理器式的可逆 patch。

## 7.5 本章小结

> 💡 **小结**
>
> * FSDP checkpoint 的核心 shape 变化发生在参数/optimizer state，而不是输入 batch。
> * `FULL_STATE_DICT` 把 shard 聚合到 rank0/CPU；`SHARDED_STATE_DICT` 通过 DCP 保存各 rank shard。
> * parallelism config 决定 FSDP shard group；TP/CP/HSDP 组合下不能简单把 world size 等同于 FSDP shard size。

# 八、核心机制深挖

## 8.1 设计哲学与核心问题

Axolotl 没有重写一个完整的 FSDP checkpoint 系统，而是采用“上游 Trainer/Accelerate 主路径 + 局部 patch + 最终格式转换”的方式集成。这种方式减少了框架自有代码量，但把正确性压力转移到了 patch 的版本兼容、配置映射和下游库行为理解上。

下面深挖三个机制：FSDP2 patch、DCP save/load、配置归一化。

## 8.2 Monkey Patch：零侵入接入还是维护风险？

### 它解决什么问题

FSDP2 仍在快速演进，Accelerate/Transformers/PEFT/bitsandbytes 的默认组合并不能覆盖 Axolotl 需要的所有路径，例如：

- FSDP2 + rank0/meta efficient loading；
- FSDP2 + QLoRA 的 `Params4bit` / `Int8Params` 元数据保持；
- PEFT ParamWrapper 与 DTensor 相加；
- `Accelerator.get_state_dict` 的 full tensor 导出。

### 源码怎么实现

`patch_accelerate_fsdp2()` 直接替换：

```python
accelerate.accelerator.fsdp2_prepare_model = fsdp2_prepare_model
accelerate.Accelerator.get_state_dict = get_state_dict
```

源码在 `src/axolotl/monkeypatch/accelerate/fsdp2.py:529-538`。

FSDP2 QLoRA patch 更激进，它读取上游 `FSDPParam._init_sharded_param` / `init_unsharded_param` 源码字符串，然后做字符串替换并 `exec()` 新函数。源码在 `src/axolotl/monkeypatch/fsdp2_qlora.py:19-169`。这带来一个隐藏假设：上游函数源码必须包含预期片段，否则 patch 会记录 warning 而不是生效。

### 副作用和维护风险

- patch 是进程级全局替换，不是 `with` 上下文；
- 对上游源码文本结构敏感；
- 测试里只验证部分 patch 被替换（`tests/e2e/patched/test_fsdp2_qlora.py:10-30`），没有覆盖完整训练 resume；
- `patch_training_loop_for_fsdp()` 文件存在但当前未启用，容易让读者误判 optimizer save 主路径。

## 8.3 DCP：为什么不能简单 `torch.save(model.state_dict())`

### 它解决什么问题

对于大模型，`FULL_STATE_DICT` 会把参数聚合到 rank0，即使 offload 到 CPU，也可能造成 CPU 内存和 I/O 峰值。`SHARDED_STATE_DICT` 使用 PyTorch Distributed Checkpoint，让每个 rank 写自己的 shard，同时保存 metadata 用于恢复。

### 源码怎么实现

Accelerate 的 FSDP model 保存逻辑在本地依赖 `accelerate/utils/fsdp_utils.py:147-157`：

```python
ckpt_dir = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_{model_index}")
state_dict = {"model": state_dict}
dist_cp.save(
    state_dict=state_dict,
    storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
    planner=DefaultSavePlanner(),
)
```

optimizer 保存类似，在 `accelerate/utils/fsdp_utils.py:269-278` 写入 `optimizer_0/`。

加载时，model DCP 路径在 `accelerate/utils/fsdp_utils.py:213-229`：先构造当前 model 的 state dict 形状，再 `dist_cp.load()` 填充；optimizer 在 `accelerate/utils/fsdp_utils.py:307-335` 中先调用 `get_optimizer_state_dict()` 得到目标结构，再从 `optimizer_0/` 读回，最后用 `set_optimizer_state_dict()` 写回 optimizer。

### 隐藏假设

- 恢复时的 FSDP wrapping、参数名、optimizer 参数组必须与保存时兼容；
- `state_dict_type` 与目录结构要匹配；
- scheduler 不是 DCP 保存，仍依赖普通 `scheduler.pt`；
- DCP model merge 只解决最终权重，不解决训练状态恢复。

## 8.4 配置归一化：用户字段如何改变源码路径

`fsdp_config.state_dict_type` 最终会变成 Accelerate FSDP plugin 的 `state_dict_type`，并在下游 save/load 中决定走 FULL 还是 SHARDED。Axolotl 自己还会在最终保存时读取：

```python
if cfg.fsdp_config.final_state_dict_type:
    state_dict_type = cfg.fsdp_config.final_state_dict_type
else:
    state_dict_type = cfg.fsdp_config.state_dict_type
trainer.accelerator.state.fsdp_plugin.set_state_dict_type(state_dict_type)
```

源码在 `src/axolotl/train.py:296-301`。

这说明配置改变的是**多个源码路径**：

- `state_dict_type=FULL_STATE_DICT`：中间 checkpoint 保存 HF full 权重 + FSDP full state；最终 `trainer.save_model()` 直接保存 HF 权重；
- `state_dict_type=SHARDED_STATE_DICT`：中间 checkpoint 保存 DCP shard；最终 `trainer.save_model()` 不直接写 HF 权重，Axolotl 尝试从 last checkpoint merge；
- `final_state_dict_type`：只在最终保存前临时切换 FSDP plugin state dict type。

## 8.5 本章小结

> 💡 **小结**
>
> * Axolotl 选择复用下游 Trainer/Accelerate checkpoint 主路径，而不是自研完整状态机。
> * Monkey patch 降低了侵入性，但把风险转移到上游源码结构和版本兼容上。
> * DCP 是 sharded resume 的核心格式，但最终 HF checkpoint 需要额外 merge。

# 九、显存、性能与通信分析

## 9.1 设计哲学与核心问题

FSDP 的核心收益是把参数、梯度和 optimizer state shard 掉；checkpoint 语义的核心代价是：当你需要 full state 或最终 HF 权重时，分片状态必须被某种方式重新聚合或转换。

因此，这一章关注的不是“FSDP 是否省显存”这个泛泛结论，而是：Axolotl 的 checkpoint/resume 路径在哪些时刻节省显存，哪些时刻又会产生峰值。

## 9.2 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数常驻显存 | ✅ | FSDP shard 后每个 rank 常驻 local shard；`reshard_after_forward=True` 更接近 ZeRO-3 语义 |
| 梯度显存 | ✅ | FSDP 对梯度分片 / reduce-scatter，避免每 rank 保存完整梯度 |
| optimizer state | ✅ | `save_fsdp_optimizer()` / FSDP 训练状态使用 sharded optimizer state |
| 激活值 | 取决于配置 | FSDP 本身不自动解决所有激活峰值；`activation_checkpointing` / `gradient_checkpointing` 另行影响 |
| 中间 checkpoint `SHARDED_STATE_DICT` 保存 | ✅ | DCP 分片写盘，不需要 rank0 聚合完整模型写单文件 |
| 中间 checkpoint `FULL_STATE_DICT` 保存 | ❌/⚠️ | rank0 需要 full state；即使 CPU offload，也有 CPU 内存和 I/O 峰值 |
| 最终 HF checkpoint | ❌/⚠️ | HF 格式需要合并权重；FULL 走 all-gather，SHARDED 走 CPU-bound DCP merge |
| 输入 batch | ❌ | FSDP checkpoint/resume 不改变 batch tensor shape |

真正的显存大头通常是参数、梯度、optimizer state、激活值。FSDP checkpoint/resume 主要影响前三者的保存和恢复；激活值更多由 activation checkpointing、context parallel、flash attention 等机制决定。

`cpu_ram_efficient_loading` 的收益在初始化：非 rank0 不加载完整 CPU 权重，而是 meta/empty 后通过 broadcast 得到 shard。源码在 `src/axolotl/loaders/model.py:756-780` 和 `src/axolotl/monkeypatch/accelerate/fsdp2.py:20-97`。代价是初始化通信和每参数处理逻辑更复杂。

## 9.3 通信开销

| 阶段 | 通信类型 | 参与 group | 是否每 step | 说明 |
|---|---|---|---:|---|
| FSDP forward | all-gather 参数 | FSDP shard group | ✅ | 由 PyTorch FSDP 管理，Axolotl 不直接调用 |
| FSDP backward | reduce-scatter / gradient sync | FSDP shard group | ✅ | optimizer state shard 语义依赖它 |
| `reshard_after_forward=True` | forward 后释放/reshard | FSDP shard group | ✅ | 省显存但增加后续再 gather 成本 |
| `cpu_ram_efficient_loading` 初始化 | `distribute_tensor` / `dist.broadcast` | device mesh / default process group | ❌ | 每个参数从 rank0 分发到 shard |
| FULL checkpoint 保存 | all-gather/full tensor | FSDP group | 按保存频率 | rank0 聚合完整 state，CPU offload/rank0_only 降 GPU 峰值但不消除聚合成本 |
| SHARDED DCP 保存 | DCP metadata + shard 写盘 | 各 rank | 按保存频率 | 避免 full gather，但多 rank I/O 与 metadata 管理更复杂 |
| SHARDED DCP load | DCP shard load | 各 rank | resume 时 | 按当前 model/optimizer state 结构读回 shard |
| dynamic checkpoint trigger | broadcast small tensor | 默认分布式 group | 按 check_interval | 只同步是否触发保存 |
| final merge | `no_dist=True` CPU load | 通常 rank0/主进程 | 训练结束或手动 | 主要是 CPU/I/O，不是 GPU all-gather |

一个特殊性能点在 Axolotl patch 的 `Accelerator.get_state_dict()`。`src/axolotl/monkeypatch/accelerate/fsdp2.py:158-173` 对 FSDP2 DTensor 调用 `param.full_tensor()`，rank0 收集 CPU tensor，并在每个参数后 `torch.distributed.barrier()`。这有利于控制状态一致性和释放节奏，但也意味着 full state 导出可能是串行参数级瓶颈。

## 9.4 性能取舍

Axolotl 这里的取舍可以概括为：

- **用通信换显存**：FSDP 每 step all-gather/reduce-scatter，换取参数/梯度/optimizer state 分片；
- **用 DCP metadata/I/O 复杂度换 rank0 内存峰值**：`SHARDED_STATE_DICT` 避免直接 full gather 保存；
- **用 patch 复杂度换生态兼容性**：FSDP2 + QLoRA、PEFT ParamWrapper、rank0/meta loading 都依赖 patch；
- **用 CPU-bound merge 换最终 HF 兼容性**：DCP merge 让最终产物可以是 safetensors，但训练结束会多一次 CPU/I/O 过程。

## 9.5 本章小结

> 💡 **小结**
>
> * FSDP checkpoint resume 真正节省的是参数、梯度、optimizer state 的常驻和保存形态，不节省输入 batch。
> * `FULL_STATE_DICT` 更方便 HF 保存，但有 rank0 聚合峰值；`SHARDED_STATE_DICT` 更适合大模型中间保存，但最终要 merge。
> * 性能瓶颈常出现在 full state 导出、DCP merge、rank0 CPU/I/O，而不是普通训练 step 本身。

# 十、配置项、边界条件与坑点

## 10.1 设计哲学与核心问题

配置表如果只列字段名没有意义。对 FSDP checkpoint/resume 来说，关键是字段如何改变源码路径：是否写环境变量、是否改变 Trainer save/load、是否导致无法 resume、是否只影响最终保存。

## 10.2 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `fsdp_config` | `utils/trainer.py:648-651`、`builders/base.py:604-607` | 开启 FSDP 环境变量并传给 TrainingArguments | 与 DeepSpeed 不能混用；旧 `fsdp` 已 deprecated |
| `fsdp_version: 2` | `validation.py:1072-1085`、`patch_manager.py:287-299` | 启用 FSDP2 patch / parallelism patch | torch < 2.7 会被拒绝（`config.py:1722-1735`） |
| `fsdp_config.state_dict_type` | Accelerate FSDP plugin、`save_trained_model()` | 控制中间 checkpoint model/optimizer state 格式 | `SHARDED_STATE_DICT` 最终 HF 需要 merge |
| `fsdp_config.final_state_dict_type` | `train.py:296-301` | 训练结束前临时覆盖最终保存格式 | 只影响 final save，不改变已写的中间 checkpoint |
| `save_only_model: true` | Transformers `_save_checkpoint` | 跳过 optimizer/scheduler/scaler/RNG | schema 明确说明不能 resume（`config.py:1103-1107`） |
| `resume_from_checkpoint` | `utils/train.py:37-47`、`train.py:625-627` | 显式指定恢复目录 | 必须指向完整 `checkpoint-*`，不应指最终 `output_dir` |
| `auto_resume_from_checkpoints` | `utils/train.py:22-47` | 自动选择最新 `checkpoint-*` | 不校验模型/配置一致性 |
| `cpu_ram_efficient_loading` | `loaders/model.py:756-780`、`fsdp2.py:371-425` | rank0 CPU / 非 rank0 meta，后续 broadcast 到 shard | MXFP4 量化不支持；Params4bit 路径会绕开部分 meta 优化 |
| `offload_params` | `setup_fsdp_envs`、FSDP plugin | 参数可 offload 到 CPU | 与 8bit optimizer / pin memory 有约束 |
| `cpu_offload_pin_memory: false` | `validation.py:1017-1032`、`fsdp2.py:347-349` | FSDP2 CPU offload 可禁用 pinned memory | 必须 `offload_params: true` 且 FSDP2 |
| `optimizer: adamw_8bit` | `validation.py:1088-1117` | FSDP2 下被拒绝 | 应使用 `adamw_torch_8bit` 或非 8bit optimizer |
| `dynamic_checkpoint.enabled` | `builders/base.py:128-133`、`dynamic_checkpoint.py` | 文件触发中间 checkpoint | 仍依赖完整 Trainer save；不是新格式 |
| `save_steps` / `save_strategy` | `builders/base.py:443-452` | 决定是否产生 `checkpoint-*` | `SHARDED_STATE_DICT` final auto merge 依赖 last checkpoint |

## 10.3 静默失效和不兼容组合

几个风险尤其值得注意：

1. **`save_only_model` 与 resume 目标冲突**：它会跳过 optimizer/scheduler/RNG 保存；
2. **`SHARDED_STATE_DICT` + 没有中间 checkpoint**：最终自动 merge 可能找不到 `checkpoint-*/pytorch_model_fsdp_0`；
3. **`auto_resume_from_checkpoints` 与不同模型复用 output_dir**：只看目录名，不看模型身份；
4. **top-level `fsdp_final_state_dict_type`**：schema 存在但主保存路径未确认消费；
5. **FSDP2 + `adamw_8bit`**：validation 明确拒绝；
6. **FSDP2 + RL + `load_in_4bit/load_in_8bit`**：DPO/KTO/ORPO/IPO 等路径 validation 拒绝（`validation.py:1034-1048`）；
7. **FSDP2 + MXFP4 + `cpu_ram_efficient_loading`**：validation 拒绝（`validation.py:1430-1437`）；
8. **ReLoRA + FSDP**：validation 拒绝（`validation.py:1465-1477`）。

## 10.4 本章小结

> 💡 **小结**
>
> * FSDP checkpoint 语义由多个配置共同决定：`state_dict_type`、`save_only_model`、`save_strategy`、resume 字段都重要。
> * `SHARDED_STATE_DICT` 是大模型中间保存友好格式，但最终发布依赖 merge。
> * 自动 resume 和 deprecated 字段迁移能提升易用性，但不会替代用户对 checkpoint 语义的判断。

# 十一、测试、示例与覆盖缺口

## 11.1 设计哲学与核心问题

测试要回答的不是“有没有 FSDP 测试”，而是：测试证明了哪些 checkpoint/resume 语义？哪些关键风险没有被保护？

从当前源码看，Axolotl 对 FSDP2 训练、配置迁移、sharded state dict、merge CLI 都有覆盖，但对“FSDP checkpoint 保存后再 resume 并验证 optimizer/scheduler 连续性”的直接覆盖仍然不足。

## 11.2 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/utils/schemas/validation/test_fsdp.py:16-174` | FSDP version 迁移、8bit optimizer 不兼容、cpu_offload_pin_memory、RL + quant 限制 | 覆盖配置合法性，不跑真实 checkpoint |
| `tests/utils/test_train.py:9-24` | `determine_last_checkpoint()` 选择最新 checkpoint、auto resume 写回 cfg | 覆盖目录选择逻辑 |
| `tests/e2e/multigpu/test_fsdp2.py:52-110` | FSDP2 FFT SFT，`cpu_ram_efficient_loading=True/False`，`FULL_STATE_DICT` | 验证训练成功和 checkpoint 文件存在 |
| `tests/e2e/multigpu/test_fsdp2.py:112-177` | FSDP2 LoRA + DoRA 参数化 | 验证 LoRA/FSDP2 训练基本可行 |
| `tests/e2e/multigpu/test_llama.py:476-520` | FSDP2 packed + `SHARDED_STATE_DICT` | 覆盖 sharded checkpoint 保存场景 |
| `tests/cli/test_cli_merge_sharded_fsdp_weights.py:8-109` | merge CLI launcher 行为 | 验证 CLI 转发，不验证真实 DCP 合并内容 |
| `tests/e2e/patched/test_fsdp2_qlora.py:10-30` | FSDP2 QLoRA patch 是否替换 FSDPParam 方法 | 验证 patch 注入，不验证完整训练恢复 |
| `tests/e2e/patched/test_resume.py:25-120` | 非 FSDP LoRA packed resume，tokens state 与 TensorBoard first_step | 覆盖通用 resume，不覆盖 FSDP sharded optimizer |
| `examples/gpt-oss/gpt-oss-20b-fft-fsdp2-offload.yaml:62-69` | FSDP2 + offload + `SHARDED_STATE_DICT` 推荐配置 | 展示大模型 offload/sharded 保存场景 |
| `examples/gpt-oss/README.md:58-65` | sharded final checkpoint 自动/手动 merge 指引 | 文档说明 final HF merge 预期 |

## 11.3 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| FSDP2 `SHARDED_STATE_DICT` 保存后 resume，并验证 optimizer/scheduler 连续性 | 未在源码中确认直接覆盖 | optimizer/scheduler 缺失或 DCP load 失败可能只在用户长训中暴露 |
| `CheckpointSaveMixin` 捕获 optimizer save 失败后的 checkpoint 不可 resume | 未见专门测试 | 用户可能以为 checkpoint 可恢复，实际缺 optimizer/scheduler |
| final `SHARDED_STATE_DICT` 自动 merge 使用 last checkpoint 而不是 output_dir shard 的边界 | 未见专门测试 | 无中间 checkpoint 时可能没有最终 HF 权重 |
| top-level `fsdp_final_state_dict_type` 行为 | 未见测试 | 用户使用 deprecated 字段可能静默不符合预期 |
| 多机多节点 DCP save/load | 未在本地测试中确认 | metadata、shared filesystem、rank 映射问题可能暴露 |
| FSDP2 + QLoRA + sharded optimizer resume | 未见端到端覆盖 | 量化元数据、optimizer state 与 DTensor placement 组合风险高 |
| 性能/显存收益回归 | 主要是 smoke/e2e 成功检查 | full state 导出或 merge 变慢不一定被 CI 捕获 |
| dynamic checkpoint + FSDP sharded resume | 未见组合测试 | 文件触发保存可能生成不可恢复状态时缺少保护 |

## 11.4 示例与文档给出的推荐信号

`docs/multi-gpu.qmd:72-80` 明确推荐新用户使用 FSDP2，并说明 FSDP 会 shard 参数、梯度和 optimizer state。`docs/multi-gpu.qmd:86-130` 给出 FSDP1 到 FSDP2 的字段迁移，其中 `fsdp_state_dict_type -> state_dict_type` 是 checkpoint 语义关键迁移。

`examples/gpt-oss/README.md:58-65` 说明：使用 `SHARDED_STATE_DICT` 时，最终 checkpoint 应该自动 merge 到 `output_dir`；如果由于磁盘空间等原因失败，可以手动运行 `axolotl merge-sharded-fsdp-weights ...`。

这和源码一致，但源码还显示了一个更细的边界：自动 merge 依赖 last checkpoint 下的 `pytorch_model_fsdp_0`。

## 11.5 本章小结

> 💡 **小结**
>
> * 当前测试覆盖了 FSDP2 训练成功、配置迁移、部分 sharded 保存和 merge CLI 转发。
> * 直接验证 FSDP sharded checkpoint resume optimizer/scheduler 连续性的测试仍是缺口。
> * 示例文档对 `SHARDED_STATE_DICT` final merge 有说明，但源码边界比文档更细。

# 十二、局限性与已知优化点

## 12.1 设计哲学与核心问题

一个成熟的 checkpoint/resume 设计，不只要看主路径能不能跑，还要看边界条件是否可解释。Axolotl 的实现很务实：大部分训练状态交给 Transformers/Accelerate，自己补 patch 和最终 merge。但这种方式也继承了下游库的限制，并引入了 patch 维护成本。

## 12.2 硬约束

1. **FSDP2 torch 版本约束**：`src/axolotl/utils/schemas/config.py:1722-1735` 要求 FSDP2 使用 torch >= 2.7.0。
2. **DeepSpeed 与 FSDP 互斥**：`src/axolotl/utils/schemas/validation.py:1187-1192` 检查 `deepspeed` 和 `fsdp` 不能同时使用。
3. **FSDP2 与部分 8bit optimizer 不兼容**：`adamw_8bit` / `adamw_bnb_8bit` 在 FSDP2 下被拒绝（`validation.py:1102-1117`）。
4. **FSDP2 + RL + base quant 限制**：DPO/KTO/ORPO/IPO 与 `load_in_8bit/load_in_4bit` 组合被拒绝（`validation.py:1034-1048`）。
5. **MXFP4 + cpu_ram_efficient_loading 不支持**：`validation.py:1430-1437` 明确报错。
6. **ReLoRA 不支持 FSDP/DeepSpeed**：`validation.py:1465-1477`。

## 12.3 维护成本

- `fsdp2_qlora.py` 依赖上游 `FSDPParam` 函数字符串片段，PyTorch 内部实现变化会破坏 patch；
- `patch_accelerate_fsdp2()` 替换 `Accelerator.get_state_dict`，会覆盖 Accelerate 默认行为；
- `PatchManager` 中存在未启用的 `trainer_fsdp_optim` patch 注释，说明历史上 optimizer save 语义曾经需要修补，也增加读者理解成本；
- `final_state_dict_type`、旧式 `fsdp_state_dict_type`、top-level deprecated 字段并存，配置追踪复杂。

## 12.4 性能瓶颈

1. **FULL_STATE_DICT 导出**：需要聚合 full tensor，rank0 CPU/I/O 峰值明显；
2. **Axolotl patched `get_state_dict` 参数级 barrier**：`src/axolotl/monkeypatch/accelerate/fsdp2.py:158-173` 每个参数 `full_tensor()` 后 barrier，可能成为串行瓶颈；
3. **DCP merge CPU-bound**：`merge_fsdp_weights()` 注释明确“CPU-bound process”（`src/axolotl/cli/merge_sharded_fsdp_weights.py:117`）；
4. **rank0/meta 初始化广播**：`cpu_ram_efficient_loading` 节省 CPU 内存，但每参数从 rank0 分发到 shards；
5. **保存频率过高**：`SHARDED_STATE_DICT` 虽避免 full gather，但频繁 DCP metadata 和多 rank I/O 仍会拖慢训练。

## 12.5 已知优化点

源码中几个 TODO / 注释值得关注：

- `src/axolotl/core/trainers/mixins/checkpoints.py:17` 有 `TODO: fix fsdp2 optimizer saving`，说明 optimizer save 容错并不是最终理想状态；
- `src/axolotl/loaders/patch_manager.py:301-307` 注释了 `trainer_fsdp_optim` patch，未来如果 upstream 行为再次变化，可能会重新启用或删除；
- `src/axolotl/train.py:334-349` 对 FSDP prefix 清理有 TODO，引用 Transformers PR；
- `src/axolotl/monkeypatch/accelerate/fsdp2.py:240-242` 对 ParamWrapper 是否需要单独 shard 有 TODO；
- DCP merge 可以考虑更显式地支持从 `output_dir/pytorch_model_fsdp_0` 自动 merge，减少 final sharded 无中间 checkpoint 时的困惑；
- 对 sharded resume 可以增加“checkpoint 完整性检查”：检查 `pytorch_model_fsdp_0/`、`optimizer_0/`、`scheduler.pt`、`trainer_state.json`、`rng_state_{rank}.pth` 是否齐全。

## 12.6 本章小结

> 💡 **小结**
>
> * Axolotl 的 FSDP checkpoint 可靠性高度依赖下游库和 patch 的版本兼容。
> * 最大性能瓶颈集中在 full state 导出、DCP merge 和初始化广播。
> * 最值得补强的是 FSDP sharded resume 完整性测试与 checkpoint 完整性诊断。

# 小结与展望

`Axolotl` 的 `FSDP checkpoint resume` 实现可以用几个关键词概括。

## 关键词一：配置翻译

用户配置不会直接调用 FSDP save/load，而是通过 schema、validation、环境变量和 TrainingArguments 翻译成 Accelerate FSDP plugin 行为。`state_dict_type`、`final_state_dict_type`、`save_only_model`、`resume_from_checkpoint` 共同决定 checkpoint 是否可恢复、最终是否可发布。

## 关键词二：下游主路径复用

Axolotl 没有自研完整 checkpoint 状态机。中间 checkpoint 的模型、optimizer、scheduler、RNG 保存/恢复主要复用 Transformers Trainer 与 Accelerate FSDP utils。这样代码量较小，也更贴近生态默认行为；代价是读源码时必须跨 Axolotl、Transformers、Accelerate、PyTorch DCP 多层追踪。

## 关键词三：patch 补缝

FSDP2 + QLoRA、rank0/meta loading、PEFT ParamWrapper、full state 导出都通过 patch 补齐。这些 patch 解决了真实工程问题，但也带来上游版本敏感性。尤其是通过源码字符串替换 FSDPParam 方法的实现，维护成本不低。

## 关键词四：DCP 分片与 HF 合并的双格式

中间 checkpoint 适合用 sharded DCP 保存和恢复；最终模型适合用 HF safetensors 发布。Axolotl 用 `merge_fsdp_weights()` 在两种格式之间架桥。这个设计适合大模型训练，但要求用户理解：`checkpoint-*` 是训练恢复目录，`output_dir` 是最终模型目录，两者不能混用。

## 关键词五：通信换显存

FSDP 本质是用 all-gather/reduce-scatter 和更复杂的 checkpoint I/O 换取参数、梯度、optimizer state 的显存节省。`SHARDED_STATE_DICT` 进一步减少 rank0 聚合压力，但最终 merge 和 resume 都需要正确的 DCP metadata、rank mapping 和 state dict 结构。

## 适合什么场景

这套实现适合：

- 多 GPU 全参微调或大模型 LoRA/QLoRA；
- 模型过大，FULL checkpoint 聚合成本过高，需要 sharded DCP 中间保存；
- 需要最终导出 HF safetensors，但训练期间希望保留 sharded resume 能力；
- 能接受 Accelerate/FSDP2/PyTorch 版本约束的工程环境。

## 不适合什么场景

不太适合：

- 需要极简、单文件、无分布式依赖 checkpoint 的场景；
- 频繁保存 full checkpoint 且 rank0 CPU/I/O 资源有限的场景；
- 下游库版本频繁漂移、无法锁定 PyTorch/Accelerate/Transformers 组合的环境；
- 希望 `output_dir` 同时承担最终模型发布和完整训练 resume 的用户。

## 后续值得继续走读的方向

如果继续深挖，可以沿三条线展开：

1. **FSDP2 + QLoRA patch 全链路**：从 bitsandbytes 参数类型到 FSDPParam shard/unshard；
2. **Accelerate FSDP plugin 内部状态**：`state_dict_type`、`StateDictOptions`、`optim_state_dict_config` 如何组合；
3. **DCP metadata 与多机恢复**：`.metadata`、rank mapping、shared filesystem 对 resume 一致性的影响。

总体来看，Axolotl 的实现不是“把 FSDP 打开”这么简单，而是在训练可恢复性、最终模型可发布性和大模型显存压力之间做了一套务实折中：中间状态交给分布式 checkpoint，最终权重再合并成 HF 格式；能省显存，但必须尊重 checkpoint 语义边界。
