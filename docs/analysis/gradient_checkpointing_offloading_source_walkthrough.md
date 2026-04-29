# Axolotl 源码走读：Gradient Checkpointing, Activation Offloading, and Layer Offloading 实现解析

在大模型微调里，显存优化常常被概括成一句话：把不该常驻 GPU 的东西挪走，或者在反向时再算一遍。但真正落到训练框架里，问题会复杂得多：哪些配置由框架自己消费？哪些只是传给 HuggingFace / PyTorch / TRL / Accelerate？LoRA 的冻结参数和激活值是不是同一类“可 offload 对象”？FSDP 的 activation checkpointing 和普通 `gradient_checkpointing` 能不能同时开？

本文以 `/root/axolotl` 当前源码为准，走读 Axolotl 对 **Gradient Checkpointing、Activation Offloading、Layer Offloading** 三类显存优化的接入方式。重点不是重复这些技术的论文原理，而是追踪一次 `axolotl train config.yml` 从配置到训练 step 的真实路径：配置如何被改写，模型什么时候被 wrap，patch 影响的是局部还是全局，每个 step 中到底发生了哪些 CPU/GPU 迁移，以及保存、FSDP、DeepSpeed、LoRA 会带来哪些边界问题。

# 前言

## 业务 / 工程背景

Axolotl 面向的是大模型 SFT / RLHF / LoRA / QLoRA / FSDP / DeepSpeed 微调。长序列、MoE、VLM、多卡分片会把显存压力推到三个方向：

1. **激活值显存**：训练时 autograd 要保存中间张量用于反向，序列越长、batch 越大越明显。
2. **冻结参数显存**：LoRA / QLoRA 只训练少量 adapter，但原模型 decoder layer 的 frozen weights 仍然常驻 GPU。
3. **分布式包装显存**：FSDP / ZeRO 虽然处理参数/优化器状态，但和 activation checkpointing 的重计算边界、all-gather 时机、参数 offload 策略会互相影响。

Axolotl 的这组三个特性正是在这个交界处工作：

- `gradient_checkpointing`：用重计算减少激活保存。
- `activation_offloading`：把 autograd 保存的激活搬到 CPU 或 legacy/disk 后端。
- `layer_offloading`：把 frozen decoder layer 参数搬到 CPU，训练时按层流式搬回 GPU。

## 核心矛盾

这套实现背后的核心矛盾可以压缩成三句话：

- **Gradient checkpointing 省激活，但牺牲算力**：反向时需要重新执行 forward，而且不同后端对 reentrant / non-reentrant 的兼容性不同。
- **Activation offloading 继续省激活，但把压力转移到 CPU 内存、PCIe/NVLink 传输和 stream 同步**：它不是分布式通信，而是每个 rank 本地的设备间数据迁移。
- **Layer offloading 省的是 frozen 参数，不是激活**：它最适合 LoRA/QLoRA，但会把每层 forward/backward 变成参数搬运调度问题。

## 本文主线

本文按机制而不是按文件展开：

1. 配置如何从 YAML 归一化成三个不同执行路径。
2. 标准 gradient checkpointing 的主路径：Axolotl 只是把开关交给 HF Trainer / Transformers。
3. activation offloading 的两条路线：现代 TRL `saved_tensors_hooks` 路线与 legacy 全局 checkpoint patch。
4. layer offloading 的 hook 调度：按层 offload / prefetch frozen 参数。
5. 一次真实训练调用如何串起上述机制。
6. shape、状态、rank、通信、显存与性能取舍。
7. 测试、示例、缺口、坑点与维护风险。

## 不展开的内容

本文不讲 PyTorch checkpoint 的完整数学原理，不讲 FSDP/ZeRO 的完整设计，也不讲 LoRA/QLoRA 的训练原理；只讲 Axolotl 当前源码如何把这些能力接入训练链路。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/main.py` | `axolotl train` 用户入口，最终启动 `axolotl.cli.train` |
| `src/axolotl/cli/config.py` | YAML/CLI 配置读取、校验、normalize、环境准备入口 |
| `src/axolotl/utils/schemas/config.py` | `gradient_checkpointing` / `activation_offloading` / `layer_offloading` schema |
| `src/axolotl/utils/schemas/validation.py` | offload 旧字段改写、兼容性校验、FSDP/DeepSpeed/模型限制 |
| `src/axolotl/loaders/model.py` | 模型加载、activation checkpoint wrapper、adapter 前后处理 |
| `src/axolotl/loaders/patch_manager.py` | legacy/disk gradient checkpointing patch 与 DeepSpeed patch 注册 |
| `src/axolotl/core/builders/base.py` | 将 Axolotl 配置转成 Trainer/TrainingArguments 行为 |
| `src/axolotl/core/trainers/mixins/activation_checkpointing.py` | `activation_offloading: true` 的训练 step context 与 HF model wrap |
| `src/axolotl/core/trainers/mixins/layer_offloading.py` | frozen decoder layer 参数 offload/prefetch/hook 调度 |
| `src/axolotl/monkeypatch/gradient_checkpointing/*` | legacy CPU / disk offloaded checkpointing autograd Function |

> 💡 **小结**
>
> * 这三个特性解决的是同一个大问题：训练显存不够，但省的是不同对象。
> * `gradient_checkpointing` 主要省激活；`activation_offloading` 省 autograd saved tensors；`layer_offloading` 省 frozen 参数。
> * Axolotl 自己并不重写完整训练循环，而是在配置、模型加载、Trainer mixin、monkey patch 四个点接入下游库。

# 一、配置归一化：同一组显存开关为什么会分成三条路径

## 1.1 设计哲学与核心问题

用户在 YAML 里只会写几行配置：

```yaml
gradient_checkpointing: true
activation_offloading: true
layer_offloading: true
```

但源码里这三行不会走同一个后端。Axolotl 首先要解决的是**配置语义归一化**问题：

- `gradient_checkpointing: true` 是交给 Transformers/HF Trainer 的常规重计算开关。
- `activation_offloading: true` 走 Axolotl + TRL 的 `saved_tensors_hooks` 路径，而且会关闭 `TrainingArguments.gradient_checkpointing`。
- `activation_offloading: legacy` 走全局 monkey patch，把 `transformers.modeling_utils.checkpoint` 替换成 Axolotl 自己的 autograd Function。
- `layer_offloading: true` 不处理激活，而是在 Trainer 初始化后创建 `LayerOffloadManager`。

如果没有这一层归一化，用户可能以为“offload”只是 checkpointing 的一个参数，实际却可能需要 model wrap、Trainer context、全局 patch、DeepSpeed workaround 同时配合。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：Click 入口，接收 config 和 CLI override

src/axolotl/cli/config.py
  - load_cfg：读取 YAML、应用 CLI override、validate_config、normalize_config、prepare_optim_env

src/axolotl/utils/schemas/config.py
  - AxolotlInputConfig：声明三类显存开关

src/axolotl/utils/schemas/validation.py
  - check_gradient_checkpointing_w_offload：把旧的 gradient_checkpointing: offload/offload_disk 改写为新字段
  - check_activation_offloading_wo_gc：强制 activation_offloading 必须配合 gradient_checkpointing

src/axolotl/core/builders/base.py
  - _configure_gradient_checkpointing：把 cfg 写入 TrainingArguments
```

## 1.3 主流程拆解

用户入口从 `axolotl train` 进入。`src/axolotl/cli/main.py:98-128` 中的 `train()` 只负责生成/分发 config 文件，并调用 launcher；真正训练模块是 `accelerate launch -m axolotl.cli.train`，命令拼装在 `src/axolotl/cli/utils/train.py:179-185`。

进入 Python 训练进程后，`src/axolotl/cli/train.py:55-91` 做三件事：

```text
axolotl.cli.train.do_cli(config)
  -> load_cfg(config, **kwargs)
  -> HfArgumentParser(TrainerCliArgs)
  -> do_train(parsed_cfg, parsed_cli_args)
       -> load_datasets(...)
       -> axolotl.train.train(cfg, dataset_meta)
```

配置加载的关键在 `src/axolotl/cli/config.py:230-333`：

```text
load_cfg
  -> yaml.safe_load
  -> apply CLI flat/nested overrides
  -> prepare_plugins
  -> validate_config
  -> prepare_debug_log
  -> prepare_optim_env
  -> normalize_config
  -> plugin_set_cfg
```

三类字段在 schema 中的声明非常直接：`src/axolotl/utils/schemas/config.py:566-588` 中，`gradient_checkpointing` 允许 `true/false/'offload'/'offload_disk'`，`activation_offloading` 允许 `true/false/'legacy'/'disk'`，`layer_offloading` 是 bool。

真正改变语义的是 validation。`src/axolotl/utils/schemas/validation.py:1377-1398` 做了两次旧字段改写：

```text
if gradient_checkpointing == "offload":
    gradient_checkpointing = True
    activation_offloading = True

if gradient_checkpointing == "offload_disk":
    gradient_checkpointing = True
    activation_offloading = "disk"

if activation_offloading and not gradient_checkpointing:
    raise ValueError
```

这说明用户虽然可以写旧式 `gradient_checkpointing: offload`，但源码里的主语已经被改成 `activation_offloading`。

随后 `src/axolotl/core/builders/base.py:514-532` 把配置分流到 Trainer arguments：

```text
if cfg.layer_offloading:
    training_args_kwargs["layer_offloading"] = True

if cfg.activation_offloading is True:
    training_args_kwargs["gradient_checkpointing"] = False
    training_args_kwargs["activation_offloading"] = True
elif cfg.gradient_checkpointing is not None:
    training_args_kwargs["gradient_checkpointing"] = cfg.gradient_checkpointing
    training_args_kwargs["gradient_checkpointing_kwargs"] = cfg.gradient_checkpointing_kwargs or {"use_reentrant": False}
```

这里的设计意图很明显：现代 activation offloading 不再让 HF Trainer 自己启用 gradient checkpointing，而是由 Axolotl 在模型加载阶段 wrap，再在 `training_step` 外层挂 `saved_tensors_hooks`。

## 1.4 关键细节与误区澄清

> 容易误解一：`activation_offloading: true` 并不是 “HF gradient checkpointing + offload 参数”。

源码上，`activation_offloading is True` 时，builder 明确把 `TrainingArguments.gradient_checkpointing` 设成 `False`（`src/axolotl/core/builders/base.py:517-520`）。这意味着 HF Trainer 的标准启用点不会执行。标准 HF Trainer 启用 GC 的位置在本地安装的 `transformers/trainer.py:1374-1376`：

```text
if args.gradient_checkpointing:
    self.model.gradient_checkpointing_enable(...)
```

Axolotl 现代 offload 路径绕开的是这个 Trainer 开关。

> 容易误解二：`gradient_checkpointing` 默认 reentrant 不是一个固定答案。

`normalize_config` 中，非 RL、无 `unfrozen_parameters` 且用户没写 kwargs 时，会把 `gradient_checkpointing_kwargs` 设为 `{"use_reentrant": True}`（`src/axolotl/utils/config/__init__.py:263-269`）。但 builder 的兜底默认是 `{"use_reentrant": False}`（`src/axolotl/core/builders/base.py:525-532`）。所以实际默认取决于 normalize 是否已经填过 kwargs。文档 `docs/training_stability.qmd:205-216` 又推荐某些场景用 `false`，并说明 ZeRO-3 / flex_attention 例外。

> 容易误解三：`activation_offloading: disk` 在 schema/docs 中存在，但当前主路径并没有顺利接到 disk patch。

文档 `docs/gradient_checkpointing.qmd:28-29` 写了 `activation_offloading: disk`，validation 也会把旧式 `gradient_checkpointing: offload_disk` 改成 `activation_offloading = "disk"`（`src/axolotl/utils/schemas/validation.py:1387-1392`）。但 patch 注册检查的是 `self.cfg.activation_offloading == "offload_disk"`（`src/axolotl/loaders/patch_manager.py:510-520`），builder 又只 special-case `activation_offloading is True`。因此按当前源码，`"disk"` 这个值不会触发 `hf_grad_checkpoint_disk_offload_wrapper`。这不是推断设计意图，而是源码条件分支本身显示出的不一致。

## 1.5 本章小结

> 💡 **小结**
>
> * Axolotl 的显存开关不是简单布尔值，而是配置归一化后的多后端分流。
> * `activation_offloading: true` 会关闭 HF Trainer 的标准 GC 开关，改走 model wrap + training_step context。
> * `legacy` 和 `disk` 属于 patch 路线，但当前 `disk` 字段和 patch 条件存在源码级不一致。

# 二、Gradient Checkpointing：把“保存激活”变成“反向重算”的主路径

## 2.1 设计哲学与核心问题

普通训练中，每个 decoder layer forward 产生的中间激活都要被 autograd 保存。长序列下，这些激活往往比 LoRA adapter 参数大得多。Gradient checkpointing 的思路是：forward 只保存边界输入，反向时重新执行被 checkpoint 的 layer，换取显存下降。

在 Axolotl 中，标准 `gradient_checkpointing: true` 的核心并不在 Axolotl 自己，而在 Transformers：Axolotl 负责把配置传到 TrainingArguments，Transformers 负责设置各层的 `_gradient_checkpointing_func`，模型层在 `__call__` 中决定是否调用 checkpoint。

## 2.2 源码入口与关键对象

```text
src/axolotl/core/builders/base.py
  - _configure_gradient_checkpointing：写入 TrainingArguments.gradient_checkpointing / kwargs

src/axolotl/loaders/model.py
  - _configure_embedding_dtypes：adapter 路径提前调用 model.gradient_checkpointing_enable
  - _prepare_model_for_quantization：QLoRA/k-bit 路径把 use_gradient_checkpointing 传给 PEFT

/usr/local/lib/python3.12/dist-packages/transformers/trainer.py
  - Trainer._inner_training_loop：根据 args.gradient_checkpointing 启用 model.gradient_checkpointing_enable

/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py
  - PreTrainedModel.gradient_checkpointing_enable：生成 functools.partial(checkpoint, **kwargs)

/usr/local/lib/python3.12/dist-packages/transformers/modeling_layers.py
  - GradientCheckpointingLayer.__call__：训练时真正调用 checkpoint
```

## 2.3 主流程拆解

标准 SFT 主路径可以简化成：

```text
_configure_gradient_checkpointing(cfg)
  -> TrainingArguments.gradient_checkpointing = True
  -> TrainingArguments.gradient_checkpointing_kwargs = {...}

Trainer.train()
  -> transformers.Trainer: if args.gradient_checkpointing
       -> model.gradient_checkpointing_enable(kwargs)
          -> module._gradient_checkpointing_func = partial(checkpoint, **kwargs)
          -> module.gradient_checkpointing = True

decoder_layer.__call__(*args, **kwargs)
  -> if self.gradient_checkpointing and self.training
       -> self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
```

外部 Transformers 源码给出了关键执行点：

- `transformers/trainer.py:1374-1376`：Trainer 根据 `args.gradient_checkpointing` 调 `model.gradient_checkpointing_enable`。
- `transformers/modeling_utils.py:3092-3132`：生成 `partial(checkpoint, **gradient_checkpointing_kwargs)`，并启用 input grads。
- `transformers/modeling_layers.py:34-93`：`GradientCheckpointingLayer.__call__` 在训练时把 layer call 包进 `_gradient_checkpointing_func`。

从 shape 直觉看，典型 decoder layer 的输入/输出不会因为 checkpointing 改形：

```text
hidden_states: [B, T, H]
attention_mask / position_ids: 结构不变
layer output: [B, T, H]
```

变化的是 autograd 保存策略：原来可能保存 attention、MLP、norm 等多个中间结果；checkpoint 后通常只保存 layer 边界输入，反向时重新跑 layer forward。

Axolotl 对 adapter/k-bit 路径还有两个前置动作：

1. `src/axolotl/loaders/model.py:330-337`：如果 `cfg.adapter in ["lora", "qlora"]` 且 `cfg.gradient_checkpointing`，模型加载阶段会提前调用 `self.model.gradient_checkpointing_enable(...)`。
2. `src/axolotl/loaders/model.py:906-914`：对 8bit/4bit PEFT 模型，`prepare_model_for_kbit_training(..., use_gradient_checkpointing=cfg.gradient_checkpointing)` 也会收到这个开关。

这两个动作解决的是 LoRA/QLoRA 下梯度要穿过 frozen base model 的问题，和纯 full fine-tuning 的路径不完全相同。

## 2.4 关键细节与误区澄清

> 容易误解：`enable_input_require_grads()` 不是训练 step 里反复执行的逻辑。

它是初始化期动作。`src/axolotl/loaders/adapter.py:370-374` 在加载 adapter 前，如果模型支持，就调用 `model.enable_input_require_grads()`；Transformers 的 `gradient_checkpointing_enable` 也会在需要时启用 input grads（`transformers/modeling_utils.py:3124-3132`）。这保证了 PEFT 场景中 frozen embedding 输出仍能把梯度传到 LoRA 层，但它不是每 step 的显存优化逻辑。

> 容易误解：普通 GC 不引入新的分布式通信。

标准 checkpointing 的代价是**重计算**，不是 all-gather / reduce-scatter。多卡训练中仍然会有 DDP/FSDP/ZeRO 自己的通信，但那不是 checkpointing 本身新增的通信。Transformers 的 FSDP warning 也说明了两者会相互影响：`training_args.py:2669-2676` 提醒 FSDP full shard 下建议用 FSDP activation checkpointing，否则可能引入 backward redundant all-gather。

> 容易误解：`use_cache=True` 会继续生效。

Transformers 的 `GradientCheckpointingLayer.__call__` 会在 checkpointing 训练时把 `use_cache` / `past_key_values` 等缓存参数置空或关闭（`transformers/modeling_layers.py:60-92`）。所以训练路径下，KV cache 不是显存优化对象，通常会被禁掉。

## 2.5 本章小结

> 💡 **小结**
>
> * 标准 `gradient_checkpointing` 的真正执行点在 Transformers layer `__call__`，Axolotl 主要负责配置传递。
> * shape 不变，autograd 保存策略改变；省的是 layer 内部中间激活。
> * adapter/k-bit 路径会在模型加载阶段提前启用 input grads / checkpointing，不能只看 Trainer arguments。

# 三、Activation Offloading：从重计算走向 CPU/Disk saved tensor 调度

## 3.1 设计哲学与核心问题

Gradient checkpointing 已经减少了“保存多少激活”，activation offloading 进一步问：剩下必须保存的 tensor 能不能不要留在 GPU？

Axolotl 这里有两套实现：

1. **现代路径：`activation_offloading: true`**
   - 模型加载阶段用 PyTorch checkpoint wrapper 包 HF `GradientCheckpointingLayer`。
   - Trainer 每个 `training_step` 外层进入 TRL `OffloadActivations` context。
   - TRL 通过 `torch.autograd.graph.saved_tensors_hooks` 拦截 autograd 保存/读取 tensor，把大 tensor 搬到 CPU pinned memory，并用 CUDA stream overlap。

2. **legacy 路径：`activation_offloading: legacy`**
   - PatchManager 全局替换 `transformers.modeling_utils.checkpoint`。
   - 每个 checkpointed decoder layer 调 Axolotl 自定义 autograd Function，只保存 CPU 版 hidden_states。

这两条路解决的问题相似，但接入点完全不同：前者是 context + wrapper，后者是全局函数替换。

## 3.2 源码入口与关键对象

```text
src/axolotl/loaders/model.py
  - _apply_activation_checkpointing：activation_offloading is True 时 ac_wrap_hf_model(model)

src/axolotl/core/trainers/mixins/activation_checkpointing.py
  - ac_wrap_hf_model：对 GradientCheckpointingLayer 应用 apply_activation_checkpointing
  - ActivationOffloadingMixin.__init__：构建 TRL activation_offload_context
  - ActivationOffloadingMixin.training_step：每 step 进入 context
  - get_lora_act_offloading_ctx_manager：LoRA/PEFT 特化，跳过输出头、Liger、checkpoint_wrapped_module 子模块

src/axolotl/loaders/patch_manager.py
  - _apply_gradient_checkpointing_patches：legacy/disk patch 注册

src/axolotl/monkeypatch/gradient_checkpointing/offload_cpu.py
  - CPU_Offloaded_Gradient_Checkpointer：保存 CPU hidden_states，反向搬回 GPU 重算

src/axolotl/monkeypatch/gradient_checkpointing/offload_disk.py
  - DiskOffloadManager / Disco：异步保存 tensor 到临时文件，反向 prefetch/load
```

## 3.3 主流程拆解

### 现代 `activation_offloading: true`

现代路径从模型加载开始：`src/axolotl/loaders/model.py:224-253` 中 `_apply_post_model_load_setup()` 会调用 `_apply_activation_checkpointing()`，只有 `self.cfg.activation_offloading is True` 时才执行：

```text
ModelLoader.load
  -> _apply_post_model_load_setup
     -> _apply_activation_checkpointing
        -> ac_wrap_hf_model(self.model)
```

`ac_wrap_hf_model` 在 `src/axolotl/core/trainers/mixins/activation_checkpointing.py:49-52`：

```text
auto_wrap_policy = ModuleWrapPolicy({GradientCheckpointingLayer})
apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy)
```

也就是说，它不是扫描任意层名，而是依赖模型层是 Transformers 的 `GradientCheckpointingLayer` 或其子类。

第二个入口在 Trainer 初始化。`AxolotlTrainer` 的继承顺序包含 `LayerOffloadingMixin` 和 `ActivationOffloadingMixin`（`src/axolotl/core/trainers/base.py:64-74`）。`ActivationOffloadingMixin.__init__` 在 `src/axolotl/core/trainers/mixins/activation_checkpointing.py:30-42`：

```text
if self.args.activation_offloading:
    if isinstance(self.model, PeftModel):
        context = get_lora_act_offloading_ctx_manager(self.model, use_streams=True)
    else:
        context = get_act_offloading_ctx_manager(self.model, use_streams=True)
else:
    context = nullcontext()
```

每个训练 step 的执行点在 `src/axolotl/core/trainers/mixins/activation_checkpointing.py:44-46`：

```text
with self.activation_offload_context:
    return super().training_step(...)
```

这个 context 的核心来自 TRL，本地安装的 `trl/models/activation_offloading.py:79-118` 声明 `OffloadActivations(saved_tensors_hooks)`；`pack_tensor` 在 forward 保存 tensor 时被调用，`unpack_tensor` 在 backward 读取 tensor 时被调用。关键行为包括：

```text
forward 保存 tensor:
  - 小于 min_offload_size 的 tensor 不 offload
  - CPU tensor / Parameter / Buffer / FP8 tensor 不 offload
  - 记录 shape/stride/storage_offset
  - 分配 CPU pinned tensor
  - cuda stream s1 非阻塞 copy GPU -> CPU
  - tracker[tensor_id] = CPU tensor + 元数据

backward 读取 tensor:
  - 从 CPU copy 回 GPU
  - 按原 shape/stride/storage_offset 恢复视图
  - 用 stream/event 控制释放时机
```

这些逻辑分别能在 TRL 文件中对应到 `pack_tensor` 的筛选与 copy（`trl/models/activation_offloading.py:194-327`）、stream 版 unpack（`trl/models/activation_offloading.py:371-527`）、参数 storage 过滤（`trl/models/activation_offloading.py:529-559`）。

### legacy `activation_offloading: legacy`

legacy 路径不使用 TRL context，而是在 model load 前 patch Transformers 全局变量。PatchManager 的预加载 patch 顺序在 `src/axolotl/loaders/patch_manager.py:95-110`，其中 `_apply_gradient_checkpointing_patches()` 位于 pre-model-load 阶段。

legacy 条件在 `src/axolotl/loaders/patch_manager.py:499-520`：

```text
if cfg.gradient_checkpointing and cfg.activation_offloading == "legacy":
    transformers.modeling_utils.checkpoint = hf_grad_checkpoint_offload_wrapper
```

Transformers `PreTrainedModel.gradient_checkpointing_enable` 会用 `modeling_utils.checkpoint` 构造 `_gradient_checkpointing_func`（`transformers/modeling_utils.py:3092-3116`）。因此这个全局 patch 会影响后续模型启用 checkpointing 时绑定的函数。

legacy wrapper 最终调用 `CPU_Offloaded_Gradient_Checkpointer.apply`（`src/axolotl/monkeypatch/gradient_checkpointing/__init__.py:28-42`）。自定义 autograd Function 在 `src/axolotl/monkeypatch/gradient_checkpointing/offload_cpu.py:38-72`：

```text
forward:
  saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
  with torch.no_grad():
      output = forward_function(hidden_states, *args)
  save CPU hidden_states + forward_function + args

backward:
  hidden_states = saved_hidden_states.to("cuda", non_blocking=True).detach()
  hidden_states.requires_grad = True
  with torch.enable_grad():
      output = forward_function(hidden_states, *args)
  torch.autograd.backward(output, dY)
  return hidden_states.grad
```

这个路径的保存对象更窄：主要是 decoder layer 的 `hidden_states` 边界，而不是所有 autograd saved tensors。

## 3.4 关键细节与误区澄清

> 容易误解一：modern activation offloading 和 legacy offload 的 offload 粒度一样。

不一样。modern `OffloadActivations` 拦截的是 autograd saved tensors，带有大小过滤、Parameter/Buffer/FP8 过滤、storage dedup 和 stride 恢复；legacy `CPU_Offloaded_Gradient_Checkpointer` 保存的是 checkpointed layer 边界 `hidden_states`。前者更通用，但状态机更复杂；后者更直接，但依赖全局 checkpoint patch。

> 容易误解二：activation offloading 会 offload 输出头 logits。

Axolotl/TRL 都刻意避免这一点。`get_lora_act_offloading_ctx_manager` 会对 `lm_head`、`output`、`decoder.output`、`head` 等注册 `NoOpManager` hook（`src/axolotl/core/trainers/mixins/activation_checkpointing.py:120-200`）；TRL 自带实现也有类似逻辑（`trl/models/activation_offloading.py:629-700`）。原因很工程化：`[B, T, vocab]` 相关输出很快就要算 loss，来回搬 CPU 可能比省显存更亏。

> 容易误解三：patch 是局部生效的。

legacy patch 是对 `transformers.modeling_utils.checkpoint` 的全局替换（`src/axolotl/loaders/patch_manager.py:509`），没有生产路径的 restore。测试里倒是有 fixture 在用例结束后恢复：`tests/e2e/patched/test_activation_checkpointing.py:17-20`。所以它是“进程级全局 patch”，不是某个模型对象的局部属性。

> 容易误解四：`activation_offloading: true` 在 LoRA 路径完全不会启用 HF gradient checkpointing。

builder 确实把 TrainingArguments 的 `gradient_checkpointing` 设为 False（`src/axolotl/core/builders/base.py:517-520`），但模型加载阶段如果 `cfg.adapter in ["lora", "qlora"]` 且 `cfg.gradient_checkpointing`，仍会调用 `self.model.gradient_checkpointing_enable(...)`（`src/axolotl/loaders/model.py:330-337`）。因此源码行为不是一句“完全禁用 HF GC”能概括；更准确地说，是**不会由 HF Trainer 再次启用**，但 adapter 初始化路径可能已经设置了模型 checkpointing 状态。

## 3.5 本章小结

> 💡 **小结**
>
> * modern offload 是 `CheckpointWrapper + saved_tensors_hooks + CUDA stream`，legacy offload 是全局 checkpoint 函数替换。
> * modern 路径省的是 autograd saved tensors；legacy 路径更像“CPU 版 checkpoint hidden_states”。
> * 输出头、Liger、部分 checkpoint wrapped 子模块被显式排除，避免把昂贵但短生命周期的 tensor 搬来搬去。

# 四、Layer Offloading：把 frozen 参数从常驻显存变成按层流式调度

## 4.1 设计哲学与核心问题

Activation offloading 处理的是激活值；Layer offloading 处理的是参数。它的目标场景非常明确：LoRA/QLoRA 中绝大部分 base model 参数是 frozen 的，真正训练的是少量 adapter。如果所有 frozen decoder layer 权重仍然常驻 GPU，就会浪费大量显存。

Layer offloading 的工程问题是：参数不是中间 tensor，它是 module 的 `nn.Parameter`，训练过程中 forward/backward 都会访问。简单把它搬到 CPU 会导致计算时 device mismatch；简单每层同步搬回又会严重拖慢。所以 Axolotl 用 forward/backward hook 做了一个按层状态机：当前层执行前搬回，下一层提前 prefetch，上一层执行完 offload。

## 4.2 源码入口与关键对象

```text
src/axolotl/core/trainers/mixins/layer_offloading.py
  - _find_decoder_layers：BFS 找第一个 decoder layer ModuleList
  - _get_frozen_params：只收集 requires_grad=False 的参数
  - LayerOffloadManager：维护 CPU pinned buffer、_on_gpu 状态、transfer stream
  - setup_hooks：注册 forward_pre/forward/full_backward_pre/full_backward hook
  - pre_step / post_step：每个 training_step 前后整理状态
  - LayerOffloadingMixin.training_step：把 Trainer step 包进 _LayerOffloadContext

src/axolotl/core/builders/base.py
  - _configure_gradient_checkpointing：将 layer_offloading 写入 training_args_kwargs

src/axolotl/core/training_args_base.py
  - AxolotlTrainingMixins.layer_offloading：TrainingArguments 扩展字段
```

## 4.3 主流程拆解

配置进入 TrainingArguments 的位置很短：`src/axolotl/core/builders/base.py:514-517` 中，只要 `cfg.layer_offloading` 为真，就写入 `training_args_kwargs["layer_offloading"] = True`。字段定义在 `src/axolotl/core/training_args_base.py:238-242`。

Trainer 初始化时，`LayerOffloadingMixin.__init__` 会创建 manager（`src/axolotl/core/trainers/mixins/layer_offloading.py:288-300`）：

```text
if self.args.layer_offloading:
    self._layer_offload_manager = LayerOffloadManager(model=self.model, num_prefetch=1)
    self._layer_offload_manager.setup_hooks()
    self._layer_offload_ctx = _LayerOffloadContext(manager)
else:
    self._layer_offload_ctx = nullcontext()
```

manager 初始化有几个关键状态：

```text
_find_decoder_layers(model)
  -> 找到 ModuleList，要求 child class name 包含 DecoderLayer 或 TransformerBlock

_frozen_params[i]
  -> 第 i 层中 requires_grad=False 的参数列表

_cpu_data[i][name]
  -> 首次 offload 时创建的 CPU pinned tensor

_on_gpu
  -> 哪些层的 frozen 参数当前在 GPU 上

_transfer_stream
  -> torch.cuda.Stream，用于 prefetch overlap
```

这些分别在 `src/axolotl/core/trainers/mixins/layer_offloading.py:24-49`、`52-116` 中建立。

每层的状态机在 `setup_hooks()`（`src/axolotl/core/trainers/mixins/layer_offloading.py:177-235`）：

```text
forward pre-hook(layer i):
  if layer i not on GPU: load layer i
  default stream wait for transfer stream
  prefetch layer i+1

forward post-hook(layer i):
  offload layer i-1
  if i is last: offload layer i

backward pre-hook(layer i):
  if layer i not on GPU: load layer i
  wait transfer
  prefetch layer i-1

backward post-hook(layer i):
  offload layer i+1
  if i is first: offload layer i
```

参数搬运本身在 `_offload_layer` / `_load_layer`：

- `_offload_layer`：首次为每个 frozen param 分配 `torch.empty_like(..., device="cpu", pin_memory=True)`，把 GPU 参数 copy 到 CPU buffer，然后令 `param.data = cpu_buf`（`src/axolotl/core/trainers/mixins/layer_offloading.py:130-147`）。
- `_load_layer`：把 CPU tensor `.to(self._device, non_blocking=True)` 回 GPU，然后 `param.data = gpu_data`（`src/axolotl/core/trainers/mixins/layer_offloading.py:149-164`）。
- `_prefetch_layer`：让 transfer stream 等 default stream，再在 transfer stream 上 load（`src/axolotl/core/trainers/mixins/layer_offloading.py:166-175`）。

每个 step 的外层 context 在 `src/axolotl/core/trainers/mixins/layer_offloading.py:266-304`：

```text
training_step:
  with _LayerOffloadContext:
      pre_step(): offload currently-on-GPU layers; prefetch layer 0
      super().training_step(...)
      post_step(): offload leftovers; prefetch layer 0 for next step
```

## 4.4 关键细节与误区澄清

> 容易误解一：Layer offloading 会 offload LoRA 参数。

不会。`_get_frozen_params` 明确只返回 `not p.requires_grad` 的参数（`src/axolotl/core/trainers/mixins/layer_offloading.py:47-49`）。LoRA/QLoRA adapter 权重是 trainable，应该常驻 GPU，否则 optimizer step 和梯度更新会变得非常复杂。

> 容易误解二：Layer offloading 是模型加载阶段的功能。

不是。它在 Trainer 初始化时创建 manager 并注册 hook，真正搬运发生在 training step 和 layer forward/backward hook 中。模型加载阶段不会为 layer offloading 改写模型结构。

> 容易误解三：`remove_hooks()` 会在训练结束自动恢复模型。

当前源码没有调用 `remove_hooks()`；全仓搜索只看到定义和初始化引用，未看到生产路径调用。保存阶段 `src/axolotl/train.py:294-373` 也没有显式把所有 offloaded 参数搬回 GPU 或移除 hook。因此 final save 依赖 PyTorch/Transformers 能保存当前 `param.data` 所在设备的 tensor；训练后如果继续复用同一个 Python 模型对象，hooks 仍然存在。

## 4.5 本章小结

> 💡 **小结**
>
> * Layer offloading 省的是 frozen decoder layer 参数，不是 activation。
> * 它通过 forward/backward hooks 做局部调度，每次只让当前/预取层的 frozen 参数回到 GPU。
> * 当前实现没有自动 remove hooks 的收尾路径，这是保存后继续复用模型时需要注意的维护风险。

# 五、完整主路径串联

## 5.1 完整调用栈

下面用一次标准 SFT 调用串起三类特性：

```text
User: axolotl train config.yml
  │
  ├─ Step 1: CLI launcher
  │     └─ src/axolotl/cli/main.py:train
  │        -> src/axolotl/cli/utils/train.py:_launch_accelerate_training
  │        -> accelerate launch -m axolotl.cli.train config.yml
  │
  ├─ Step 2: 配置加载与校验
  │     └─ src/axolotl/cli/train.py:do_cli
  │        -> src/axolotl/cli/config.py:load_cfg
  │        -> src/axolotl/utils/config/__init__.py:validate_config
  │        -> src/axolotl/utils/config/__init__.py:normalize_config
  │        -> src/axolotl/utils/trainer.py:prepare_optim_env
  │
  ├─ Step 3: pre-model-load patch
  │     └─ src/axolotl/loaders/model.py:ModelLoader.load
  │        -> src/axolotl/loaders/patch_manager.py:apply_pre_model_load_patches
  │        -> _apply_gradient_checkpointing_patches (legacy/disk only)
  │
  ├─ Step 4: 模型构建与 wrap
  │     └─ ModelLoader._apply_post_model_load_setup
  │        -> _apply_activation_checkpointing (activation_offloading is True)
  │        -> _configure_embedding_dtypes / gradient_checkpointing_enable (adapter path)
  │        -> _load_adapters
  │
  ├─ Step 5: Trainer 构建
  │     └─ src/axolotl/utils/trainer.py:setup_trainer
  │        -> HFCausalTrainerBuilder.build
  │        -> TrainerBuilderBase._configure_gradient_checkpointing
  │        -> AxolotlTrainer(...)
  │        -> LayerOffloadingMixin.__init__
  │        -> ActivationOffloadingMixin.__init__
  │
  ├─ Step 6: 每个 training step
  │     └─ AxolotlTrainer.training_step
  │        -> LayerOffloadingMixin.training_step
  │           -> _LayerOffloadContext.pre_step/post_step
  │        -> ActivationOffloadingMixin.training_step
  │           -> OffloadActivations saved_tensors_hooks
  │        -> transformers.Trainer.training_step
  │           -> model forward / loss / backward
  │
  └─ Step 7: 训练完成与保存
        └─ src/axolotl/train.py:save_trained_model
           -> trainer.save_model / model.save_pretrained
           -> FSDP/DeepSpeed 分支处理
```

## 5.2 每一层做了什么

| 层 | 输入 | 输出/状态变化 | 是否每 step | 是否触发通信/迁移 | 显存影响 |
|---|---|---|---|---|---|
| 配置校验 | YAML + CLI override | 改写 `offload` 字段，填 kwargs，报错/警告不兼容组合 | 否 | 否 | 决定后续路径 |
| pre-model patch | cfg + transformers module | legacy 时替换 `modeling_utils.checkpoint` | 否 | 否 | 改变之后 GC 函数绑定 |
| model wrap | model | `apply_activation_checkpointing` 包 `GradientCheckpointingLayer` | 否 | 否 | 减少后续 autograd 保存 |
| Trainer init | model + TrainingArguments | 创建 activation context / layer manager；注册 hooks | 否 | 初始 layer offload 会 GPU->CPU | Layer offload 立刻释放 frozen 参数显存 |
| training_step | batch tensors | forward/loss/backward | 是 | activation/layer offload 的 CPU<->GPU copy；FSDP/ZeRO 另有 collectives | 激活和 frozen 参数峰值下降 |
| save | trainer/model | state dict / safetensors / sharded output | 否/按 save step | FSDP/DeepSpeed save 可能有 gather/barrier | offloaded CPU tensor 可能影响保存时 CPU 内存 |

## 5.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `src/axolotl/monkeypatch/gradient_checkpointing/offload_disk.py:Disco` | 文档写了 disk offload，文件也存在 | 当前 `activation_offloading: disk` 不会命中 patch 条件 | patch 条件检查 `"offload_disk"`，validation 写入 `"disk"`；需要修正才能成为主路径 |
| `LayerOffloadManager.remove_hooks()` | 看起来像训练结束清理 | 未在生产路径调用 | 只是可用工具函数，当前保存前不会自动恢复/移除 hook |
| `Trainer.args.gradient_checkpointing` | 以为它控制所有 checkpointing | `activation_offloading: true` 时被设 False | modern offload 由 `ac_wrap_hf_model` 和 `ActivationOffloadingMixin` 控制 |
| `load_adapter().enable_input_require_grads()` | 以为是 offload 逻辑 | 不是 offload 主流程 | 它解决 PEFT 梯度穿过 frozen embedding 的问题 |
| `FSDP_ACTIVATION_CHECKPOINTING` 环境变量 | 名字很像 `activation_offloading` | 是 FSDP checkpointing 分支，不是 TRL offload | 由 `fsdp_config.activation_checkpointing` 写入 env，走 Accelerate/FSDP wrapper |
| `tests/e2e/patched/test_activation_checkpointing.py` | 参数含 `offload_disk` | 只证明训练能跑完 | 没断言 `Disco` 被使用，也没有验证磁盘 I/O/offload 文件 |

> 💡 **小结**
>
> * 主路径不是单线：配置会把用户意图分到 HF Trainer、TRL context、legacy patch、layer hook、FSDP wrapper 等不同位置。
> * `activation_offloading: true` 的主流程跨模型加载和每 step context；`layer_offloading` 跨 Trainer init 和每层 hook。
> * 有些看似相关的代码当前不在实际主路径，尤其是 disk offload 和 hook cleanup。

# 六、关键数据流、状态流与 shape 流程

## 6.1 Tensor shape 变化

以典型 CausalLM decoder layer 为例：

```text
原始 batch:
  input_ids:        [B, T]
  attention_mask:   [B, T] 或扩展 mask

embedding 后:
  hidden_states:    [B, T, H]

Gradient Checkpointing:
  forward 保存边界 hidden_states: [B, T, H]
  不保存或少保存 layer 内 q/k/v、MLP 中间激活
  backward 重新计算 layer forward

Activation Offloading (modern):
  autograd saved tensor x: shape = x.shape, stride = x.stride()
  pack:   GPU x -> CPU pinned buffer，记录 stride/storage_offset/original_shape
  unpack: CPU buffer -> GPU tensor，再 as_strided 恢复视图

Legacy CPU offload:
  saved_hidden_states: [B, T, H] 从 GPU copy 到 CPU
  backward: CPU -> CUDA，requires_grad=True，重算 forward

Layer Offloading:
  layer_i frozen param，例如:
    attention.q_proj.weight: [H, H]
    mlp.gate_proj.weight:   [4H, H]
  offload: param.data 指向 CPU pinned tensor，shape 不变，device 变 CPU
  load:    param.data 指向 GPU tensor，shape 不变，device 变 CUDA
```

这里真正“节省显存”的步骤不同：

- GC 省掉 layer 内部中间激活的持久保存。
- modern activation offload 省掉 autograd saved tensors 在 GPU 上的驻留。
- legacy offload 省掉 checkpoint boundary hidden_states 的 GPU 驻留。
- layer offload 省掉 frozen 参数在 GPU 上的常驻。

性能瓶颈也不同：GC 瓶颈是重计算；activation offload 瓶颈是 CPU pinned memory 和设备间 copy；layer offload 瓶颈是每层参数 H2D/D2H 传输与 stream 等待。

## 6.2 Rank / Mesh / Process Group 变化

这三个特性本身不创建新的 process group。每个 rank 都在本地对自己的模型副本/分片做 offload：

```text
DDP rank 0: 本地 forward -> 本地 activation CPU offload -> DDP grad all-reduce
DDP rank 1: 本地 forward -> 本地 activation CPU offload -> DDP grad all-reduce
...
```

如果叠加 FSDP2，则会出现两层机制：

```text
FSDP2:
  参数 shard/unshard 需要 FSDP group 内通信
  fully_shard 前可 apply_activation_checkpointing

Activation / Layer Offload:
  每个 rank 本地 CPU<->GPU copy
  不新增 all_gather / reduce_scatter
```

FSDP2 patch 的关键在 `src/axolotl/monkeypatch/accelerate/fsdp2.py:327-342`：当 `fsdp2_plugin.activation_checkpointing` 为真时，会在 `fully_shard` 前调用 PyTorch `apply_activation_checkpointing`，并使用 non-reentrant checkpoint wrapper。随后 `fsdp2_kwargs` 中会把 mesh slice 传给 `fully_shard`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:344-360`）。

因此，FSDP activation checkpointing 的通信来自 FSDP 参数 all-gather/reduce-scatter；Axolotl activation offloading 的“通信”更准确地说是本地 CPU/GPU DMA，不是分布式 collective。

## 6.3 状态切换

### modern activation offloading 状态

```text
进入 training_step:
  ActivationOffloadingMixin: with self.activation_offload_context

forward 中:
  saved_tensors_hooks.pack_tensor 被 autograd 调用
  tracker[tensor_id] = (cpu_tensor, modified, stride, offset, shape)
  fwd_stash 保存仍在传输中的 GPU tensor/event

backward 中:
  saved_tensors_hooks.unpack_tensor 被 autograd 调用
  CPU tensor copy 回 GPU
  bwd_tensor_stash / bwd_ev_stash 控制生命周期

退出 backward:
  tracker 清空后 is_first_forward_call=True
```

状态定义在 TRL `OffloadActivations` 对象中，不是全局变量；但 hooks 在 context 范围内影响所有 autograd saved tensors。

### legacy patch 状态

```text
pre-model-load:
  transformers.modeling_utils.checkpoint = Axolotl wrapper

model.gradient_checkpointing_enable:
  module._gradient_checkpointing_func = partial(patched_checkpoint, kwargs)

training:
  layer.__call__ 调用已绑定的 patched checkpoint
```

这里的状态是进程级 monkey patch，没有生产 restore。

### layer offloading 状态

```text
初始化:
  _on_gpu = {0, 1, ..., n_layers-1}
  _cpu_data = [{} for each layer]
  _offload_all() 后 _on_gpu 变空

step 前:
  offload 所有残留 GPU layer
  prefetch layer 0

forward/backward:
  hook 根据 layer index 在 CPU/GPU 之间切换 param.data

step 后:
  offload 所有残留 GPU layer
  prefetch layer 0
```

状态定义在 `LayerOffloadManager` 实例中，每个 Trainer 一个 manager；线程安全主要依赖 CUDA stream 同步，不涉及 Python 多线程锁。

## 6.4 本章小结

> 💡 **小结**
>
> * 三个机制都基本保持 tensor shape 不变，改变的是保存位置、重算策略或 parameter device。
> * 它们不创建新的分布式通信组；真正的 collectives 来自 DDP/FSDP/ZeRO。
> * 状态最复杂的是 TRL `OffloadActivations` 和 Axolotl `LayerOffloadManager`，前者管理 autograd saved tensors，后者管理 per-layer frozen params。

# 七、核心机制深挖

## 7.1 Monkey Patch：零侵入接入还是维护风险？

legacy offload 的 patch 点非常小：`src/axolotl/loaders/patch_manager.py:499-520` 直接替换 `transformers.modeling_utils.checkpoint`。好处是不用改 Transformers 模型源码；坏处也很明显：

1. **影响范围是进程级**：所有之后调用 `model.gradient_checkpointing_enable` 并绑定 `modeling_utils.checkpoint` 的模型都会受影响。
2. **依赖上游命名空间**：Transformers 当前在 `modeling_utils.py:44` 从 `torch.utils.checkpoint` import 了 `checkpoint`，`gradient_checkpointing_enable` 再使用这个模块全局变量（`transformers/modeling_utils.py:3092-3116`）。如果上游改成其他引用路径，patch 可能失效。
3. **恢复机制只在测试中出现**：`tests/e2e/patched/test_activation_checkpointing.py:17-20` 手动把 `transformers.modeling_utils.checkpoint` 恢复为 torch 原函数，生产训练没有 restore。
4. **版本分支存在**：`src/axolotl/monkeypatch/gradient_checkpointing/__init__.py:15-26` 判断 Transformers 版本大于 4.51.3 时按 `GradientCheckpointingLayer` 的 partial 结构取 `decoder_layer.func.__self__`，否则走旧格式。

这是一种典型的“框架集成型 patch”：实现成本低、覆盖面广，但维护风险随上游变动增加。

## 7.2 通信语义：前向和反向是否对称？

这里要区分三种“通信”：

| 机制 | 前向 | 反向 | 是否 collective |
|---|---|---|---|
| 标准 GC | 少保存激活，正常 forward | 重新 forward，再反向 | ❌ |
| modern activation offload | pack tensor：GPU -> CPU | unpack tensor：CPU -> GPU | ❌，本地 DMA |
| layer offload | layer param CPU -> GPU，上一层 GPU -> CPU | 反向相反方向 prefetch/offload | ❌，本地 DMA |
| FSDP/ZeRO | 参数 unshard/all-gather 等 | reduce-scatter/all-reduce 等 | ✅ |

modern offload 的前向/反向不是严格“对称函数”，而是 pack/unpack 生命周期。`OffloadActivations` 在 forward 保存时可能跳过小 tensor、CPU tensor、Parameter、Buffer、FP8 tensor，还会对相同 storage 去重；backward 则按 tensor id 恢复。它保存 shape/stride/storage_offset（`trl/models/activation_offloading.py:279-316`），在恢复时用 `torch.as_strided` 还原视图（`trl/models/activation_offloading.py:430-438`）。

legacy CPU offload 更对称：forward 把 `hidden_states` 搬 CPU；backward 搬回 CUDA、重算 forward、对输出执行 `torch.autograd.backward`（`src/axolotl/monkeypatch/gradient_checkpointing/offload_cpu.py:46-72`）。

Layer offload 的对称性体现在 hook 顺序：forward 从 0 到 N-1 预取下一层，backward 从 N-1 到 0 预取上一层（`src/axolotl/core/trainers/mixins/layer_offloading.py:185-229`）。

## 7.3 DeepSpeed/FSDP patch：兼容性为什么会进入显存优化代码？

activation checkpoint wrapper 会改变 module 外壳。DeepSpeed ZeRO-3 可能需要在 module 上写入 `ds_*` 属性，如果 wrapper 不转发，这些属性留在外层 wrapper，内层真实 module 看不到，就可能出错。

Axolotl 的 workaround 在 `src/axolotl/monkeypatch/deepspeed_utils.py:9-65`：

```text
patch CheckpointWrapper.__setattr__:
  if name startswith "ds_" and has _checkpoint_wrapped_module:
      setattr(inner_module, name, value)
  else:
      original_setattr(...)
```

PatchManager 只在 `activation_offloading is True` 且 DeepSpeed ZeRO-3 启用时应用它（`src/axolotl/loaders/patch_manager.py:793-803`）。这说明 activation offloading 并不是纯显存逻辑，它会影响下游分布式库观察 module 的方式。

FSDP2 也有独立分支。`src/axolotl/utils/trainer.py:589-618` 把 `fsdp_config.activation_checkpointing`、`offload_params`、`cpu_ram_efficient_loading`、`cpu_offload_pin_memory` 等写成 Accelerate/FSDP 环境变量；`src/axolotl/monkeypatch/accelerate/fsdp2.py:327-342` 再在 `fully_shard` 前应用 checkpoint wrapper。这条路径和 Axolotl `activation_offloading: true` 不同，不要混为一谈。

## 7.4 本章小结

> 💡 **小结**
>
> * legacy patch 是全局函数替换，易接入但强依赖 Transformers 内部结构。
> * activation/layer offload 的“通信”主要是 CPU/GPU copy；分布式 collectives 来自 FSDP/ZeRO，而不是这些 offload 机制本身。
> * DeepSpeed/FSDP patch 说明 checkpoint wrapper 会改变 module 可见性，兼容性问题会反过来影响显存优化路径。

# 八、显存、性能与通信分析

## 8.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | `layer_offloading` ✅；GC/activation offload ❌ | layer offload 把 frozen decoder params 放 CPU；GC/offload 不改变参数常驻策略 |
| 激活值 | GC ✅；activation offload ✅ | GC 减少保存；offload 把 saved tensors 搬 CPU |
| logits | 通常 ❌ | 输出头区域被 NoOp hook 排除，避免 `[B,T,V]` 来回搬 |
| optimizer state | ❌ | 三个机制都不改变 optimizer state；FSDP/ZeRO/8bit optimizer 才处理它 |
| 输入 batch | ❌ | `input_ids/labels` 等仍由 dataloader/Trainer 管理 |
| 中间 buffer | 部分 ✅ | TRL saved_tensors_hooks 会处理 autograd saved tensors，但会跳过小 tensor、Parameter、Buffer、FP8 tensor |
| frozen decoder weights | layer offloading ✅ | `_get_frozen_params` 只处理 `requires_grad=False` |
| FSDP shard/unshard 峰值 | 间接 | FSDP activation checkpointing 可改变 all-gather/recompute 时机；不是 Axolotl layer/activation offload 本身 |

真正大头取决于训练方式：

- full fine-tuning 长序列：激活值常常是大头，GC/activation offload 更有效。
- LoRA/QLoRA 大模型：frozen 参数仍可能占很大 GPU 常驻显存，layer offload 的收益更直接。
- FSDP/ZeRO：参数和 optimizer state 已被分片/offload，activation 部分可能重新变成瓶颈。

## 8.2 通信开销

| 路径 | 每 step/layer 开销 | 类型 | group |
|---|---|---|---|
| 标准 GC | 反向重算每个 checkpointed layer | 计算开销 | 无 group |
| modern activation offload | 每个符合条件 saved tensor 一次 GPU->CPU，backward 一次 CPU->GPU | 本地 DMA，CUDA stream/event | 无 group |
| legacy CPU offload | 每个 checkpointed layer hidden_states GPU->CPU/CPU->GPU | 本地 DMA | 无 group |
| disk offload | 设计上每个 checkpoint tensor 写临时文件并反向读回 | 磁盘 I/O + CPU/GPU copy | 无 group；但当前主路径未确认触发 |
| layer offload | 每层 forward/backward 前后 frozen params CPU<->GPU | 本地 DMA，transfer stream | 无 group |
| FSDP activation_checkpointing | FSDP wrapped module 前后可能 all-gather/reduce-scatter | collective | FSDP mesh/group |
| DeepSpeed ZeRO-3 | 参数 gather、grad reduce、optimizer partition | collective | ZeRO data parallel group |

能否 overlap 也不同：

- TRL `OffloadActivations` 用 `s1` comm stream 与 default compute stream 配合（`trl/models/activation_offloading.py:157-170`、`257-327`、`371-527`）。
- Layer offload 用 `_transfer_stream` 预取下一层/上一层（`src/axolotl/core/trainers/mixins/layer_offloading.py:95-96`、`166-175`）。
- legacy CPU offload 只做 non_blocking copy，但没有现代路径那样完整的 stash/event 状态机。

## 8.3 性能取舍

这三类策略本质上分别用不同成本换显存：

- **GC：用计算换显存**。反向时重算 forward，训练速度通常下降，但不引入 CPU 内存压力。
- **Activation offload：用 CPU 内存和 H2D/D2H 传输换显存**。长序列大 tensor 最受益；CPU pinned memory、stream 同步和 PCIe 带宽会成为瓶颈。
- **Layer offload：用每层参数搬运换常驻显存**。LoRA/QLoRA 下最合适；full fine-tuning 下大部分参数 trainable，收益明显下降甚至不适用。
- **FSDP offload/checkpointing：用集体通信和重计算换单卡显存**。它和本地 offload 叠加时，要警惕 all-gather 与 CPU/GPU copy 的串行化。

## 8.4 本章小结

> 💡 **小结**
>
> * GC 主要牺牲算力；activation/layer offload 主要牺牲设备间传输和 CPU 内存。
> * 三者本身不新增分布式 collective，但和 FSDP/ZeRO 叠加后，整体 step 时间可能由 all-gather、CPU/GPU copy、重计算共同决定。
> * 输出头 offload 被刻意规避，说明源码关注的不只是“省显存”，还有短生命周期大 tensor 的吞吐代价。

# 九、配置项、边界条件与坑点

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `gradient_checkpointing: true` | `core/builders/base.py:_configure_gradient_checkpointing`；HF Trainer | 传入 TrainingArguments，Trainer 调 `model.gradient_checkpointing_enable` | FSDP full shard 下 Transformers 提示建议用 FSDP activation checkpointing；reentrant 默认受 normalize 影响 |
| `gradient_checkpointing_kwargs.use_reentrant` | `utils/config/__init__.py:263-269`；Transformers GC | 改变 checkpoint 实现 | ZeRO-3/flex_attention 可能要求 `true`；Gemma4 DDP 要求 `false` |
| `gradient_checkpointing: offload` | `validation.py:1377-1386` | deprecated，改写为 `gradient_checkpointing=True` + `activation_offloading=True` | 用户以为 legacy，实际默认走现代 stream implementation；legacy 要写 `activation_offloading: legacy` |
| `gradient_checkpointing: offload_disk` | `validation.py:1387-1392` | deprecated，改写为 `activation_offloading="disk"` | 当前 PatchManager 检查 `"offload_disk"`，源码上未确认能触发 Disco |
| `activation_offloading: true` | `loaders/model.py:_apply_activation_checkpointing`；`ActivationOffloadingMixin` | model wrap + training_step context；TrainingArguments GC 设 False | adapter 路径仍可能提前 `gradient_checkpointing_enable`；需要 CUDA/TRL hooks；CPU pinned memory 压力 |
| `activation_offloading: legacy` | `PatchManager._apply_gradient_checkpointing_patches` | 全局替换 Transformers checkpoint | 进程级 patch，无生产 restore；依赖 Transformers 内部命名空间 |
| `activation_offloading: disk` | schema/docs/validation | 文档声称 disk offload | 当前源码条件不匹配；测试未断言真正使用 `Disco` |
| `layer_offloading: true` | `LayerOffloadingMixin.__init__` | 创建 manager，offload frozen decoder params | 需要 CUDA；只找类名含 `DecoderLayer`/`TransformerBlock` 的 ModuleList；无测试覆盖 |
| `fsdp_config.activation_checkpointing` | `utils/trainer.py:589-618`；`accelerate/fsdp2.py:327-342` | FSDP wrapper 前应用 activation checkpointing | 不能和 HF TrainingArguments GC 同时开；不是 Axolotl `activation_offloading` |
| `fsdp_config.offload_params` | `utils/trainer.py:597-598`；FSDP plugin | 参数 CPU offload | 和 optimizer/量化有额外限制；FSDP2 `cpu_offload_pin_memory: false` 需配合它 |
| `fsdp_config.cpu_offload_pin_memory: false` | `validation.py:1019-1031`；`accelerate/fsdp2.py:346-349` | 关闭 FSDP CPU offload pin memory | 只支持 FSDP2 且必须 `offload_params: true` |
| Gemma4 + DDP | `utils/config/__init__.py:271-300` | GC 强制 `use_reentrant=False`；activation_offloading true 时跳过 `ddp_find_unused_parameters` | 要用 `freeze_mm_modules` 处理 unused multimodal params |
| MPT | `validation.py:1353-1358` | 禁止 gradient_checkpointing | 直接报错 |
| Nemotron-H | `validation.py:1361-1374`；`patch_manager.py:361-376` | 只有 sample_packing patch 后才支持 GC | 不开 sample_packing 会报错 |
| EBFT strided + flex_attention | `validation.py:1651-1688` | 强制 reentrant 或禁止 activation_offloading | activation_offloading true 与 flex_attention strided 不兼容 |

另外还有一个文档坑：`docs/faq.qmd:148-150` 提到失败时可用 `offload_activations: legacy`，但 schema 字段是 `activation_offloading`（`src/axolotl/utils/schemas/config.py:578-583`）。按源码，应以 schema 字段为准。

> 💡 **小结**
>
> * 配置项改变的不是一个参数，而是源码路径：Trainer args、model wrap、global patch、FSDP env、hook manager 都可能被影响。
> * `disk`、`legacy`、`true` 三个 offload 值不是同一种实现，当前 `disk` 还存在源码条件不一致。
> * reentrant 选择高度依赖后端组合：FSDP、ZeRO-3、flex_attention、Gemma4、RL 都有不同约束。

# 十、测试、示例与覆盖缺口

## 10.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/utils/schemas/validation/test_activation_offloading.py:12-35` | `gradient_checkpointing: offload` 被改写为 `gradient_checkpointing=True` + `activation_offloading=True` | 只测 validation，不测训练执行 |
| `tests/e2e/test_activation_offloading.py:20-79` | `activation_offloading: true` 在 `adapter in [lora, qlora, None]` 下能完成小模型训练和保存 | 覆盖 modern path smoke test，未量化显存收益 |
| `tests/e2e/patched/test_activation_checkpointing.py:28-80` | `gradient_checkpointing in [offload, offload_disk]` 的训练 smoke test | 结束后恢复 global checkpoint patch；但未断言 CPU/disk wrapper 是否实际被绑定 |
| `tests/utils/schemas/validation/test_fsdp.py:40-118` | FSDP offload 与 optimizer、pin memory 的配置约束 | 覆盖 validation，不覆盖真实 FSDP training step |
| `tests/patched/test_validation.py:69-86` | QLoRA + ZeRO-3 + non-reentrant warning | 证明风险提示存在，不证明运行时一定失败/成功 |
| `tests/patched/test_validation.py:513-528` | MPT + gradient_checkpointing 报错 | 覆盖模型级硬限制 |
| `examples/gpt-oss/*-fsdp2-offload.yaml` | `gradient_checkpointing + activation_offloading + fsdp_config.offload_params` 推荐组合 | 示例展示大模型 FSDP2 offload 配法 |
| `examples/alst/*-alst.yaml` | `activation_offloading: legacy` | 示例展示 legacy 路径仍被推荐用于特定场景 |
| `docs/gradient_checkpointing.qmd:8-59` | 三类功能的用户文档 | 文档简洁，但部分字段与当前源码条件不一致 |

## 10.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| `activation_offloading: disk` 是否真的触发 `Disco` | 未看到断言 | 用户以为走磁盘 offload，实际可能只是普通 GC |
| `LayerOffloadManager` 行为 | 未看到测试 | hook 顺序、保存前状态、CPU/GPU 参数切换可能回归而无人发现 |
| 多机/FSDP2 + activation_offloading true | 未看到专门 e2e | wrapper、FSDP all-gather、CPU/GPU copy 组合风险未覆盖 |
| DeepSpeed ZeRO-3 + activation_offloading true patch 是否生效 | 只有 patch 源码和 warning/validation | `ds_*` 属性转发回归可能导致运行时错误 |
| 显存收益/性能回归 | 未看到 benchmark assertion | 功能能跑但显存不降或吞吐大幅下降难以及时发现 |
| 保存/resume 与 layer offloading | 未看到测试 | offloaded CPU param + hooks 未恢复可能影响 checkpoint/resume 或后续推理 |
| 输出头 NoOp hook 覆盖所有模型 | 未完全覆盖 | 未识别 output head 时可能 offload 大 logits/输出相关 tensor，训练变慢 |
| TRL `OffloadActivations` tracker edge case | FAQ 有问题说明 | `Backward pass should have cleared tracker` 类错误可能只在复杂图出现 |

## 10.3 本章小结

> 💡 **小结**
>
> * 现有测试更偏 smoke/validation，能证明“能跑”和“能改写配置”，不能充分证明显存收益和路径命中。
> * Layer offloading 当前几乎没有测试保护，是这组三个特性里覆盖最薄的一块。
> * `offload_disk` 测试没有断言 `Disco`，结合源码条件不一致，是最值得优先补的测试缺口。

# 十一、局限性与已知优化点

## 11.1 硬约束

1. **Activation offloading 需要 gradient_checkpointing**  
   `check_activation_offloading_wo_gc` 在 `src/axolotl/utils/schemas/validation.py:1395-1398` 直接报错。

2. **Layer offloading 需要 CUDA 参数**  
   `LayerOffloadManager` 找不到 CUDA 参数会禁用（`src/axolotl/core/trainers/mixins/layer_offloading.py:85-93`）。CPU-only 不支持。

3. **Layer offloading 依赖 layer 命名/结构**  
   `_find_decoder_layers` 只找第一个 `ModuleList`，且 child 类名要包含 `DecoderLayer` 或 `TransformerBlock`（`src/axolotl/core/trainers/mixins/layer_offloading.py:24-44`）。特殊架构可能静默禁用。

4. **MPT 不支持 gradient_checkpointing**  
   `src/axolotl/utils/schemas/validation.py:1353-1358` 直接报错。

5. **Nemotron-H 需要 sample_packing patch**  
   validation 要求 `sample_packing: true`（`src/axolotl/utils/schemas/validation.py:1361-1374`），PatchManager 只在 sample_packing 后设置 `supports_gradient_checkpointing=True`（`src/axolotl/loaders/patch_manager.py:361-376`）。

6. **EBFT strided + flex_attention 不兼容 modern activation_offloading**  
   `src/axolotl/utils/schemas/validation.py:1671-1688` 直接报错。

## 11.2 维护成本

- **全局 patch 维护成本高**：legacy/disk wrapper 依赖 `transformers.modeling_utils.checkpoint` 变量名和 `GradientCheckpointingLayer` 调用结构。
- **现代 offload 依赖 TRL 内部状态机**：Axolotl 只创建 context，但真正 pack/unpack 逻辑在 TRL；TRL 升级可能改变行为。
- **Layer offload 直接改 `param.data`**：这种方式高效但危险，和 FSDP/ZeRO/optimizer state/compiled graph 组合时要非常谨慎。
- **字段语义历史包袱明显**：`gradient_checkpointing: offload`、`activation_offloading: legacy/disk/true`、FSDP `activation_checkpointing` 很容易混淆。
- **清理路径不足**：layer hooks 和 legacy patch 都缺少生产级 restore/cleanup。

## 11.3 性能瓶颈

- **CPU pinned memory 压力**：modern offload 和 layer offload 都可能大量使用 pinned memory；CPU RAM 或 pinned memory 不足时，收益会变成 stalls。
- **PCIe/NVLink 带宽**：layer offload 每层都搬 frozen params，模型越大、层越宽，传输压力越明显。
- **stream overlap 不一定成功**：如果计算太短或参数太大，prefetch 来不及，default stream 仍要等待 transfer stream。
- **FSDP + activation checkpointing 的 all-gather 时机**：Transformers warning 已指出普通 GC 和 FSDP full shard 可能导致 backward 冗余 all-gather（`training_args.py:2669-2676`）。
- **disk offload I/O**：`Disco` 设计了线程池、prefetch queue、临时文件 cleanup（`src/axolotl/monkeypatch/gradient_checkpointing/offload_disk.py:43-104`、`223-243`、`376-423`），但如果接入主路径，磁盘 I/O 会成为极强瓶颈。

## 11.4 已知优化点

1. **修正 disk offload 字段与 patch 条件**  
   要么 PatchManager 接受 `activation_offloading == "disk"`，要么 validation 写入 `"offload_disk"`；并补充断言 `Disco.apply` 被绑定的测试。

2. **为 LayerOffloadingManager 增加单元/集成测试**  
   至少覆盖：找层、只 offload frozen params、hook 顺序、保存前恢复/不恢复行为、无 CUDA 时禁用。

3. **增加显存/吞吐 smoke metric**  
   不一定要做严格 benchmark，但可以在小模型上记录 `torch.cuda.max_memory_allocated` 的相对变化，防止 offload 失效。

4. **显式 cleanup / restore**  
   legacy patch 可以有上下文或训练结束恢复；layer offload 可以在 save 前可选 `remove_hooks()` 或至少提供 callback。

5. **把 output head 识别变成更可靠的模型接口**  
   当前靠属性名猜测，特殊模型可能漏掉。可考虑让模型架构 registry 提供 output head 路径。

## 11.5 本章小结

> 💡 **小结**
>
> * 这组三个特性不是“开了就安全省显存”，而是强依赖模型结构、后端版本和分布式策略。
> * 最大维护风险来自全局 patch、`param.data` 切换和字段历史兼容。
> * 最直接的优化点是修正 disk path、补 layer offload 测试、增加 cleanup 与显存回归验证。

# 小结与展望

Axolotl 的 `Gradient Checkpointing, Activation Offloading, and Layer Offloading` 实现可以用几个关键词概括。

## 关键词一：配置分流

同一个“省显存”目标被拆成多条源码路径：HF Trainer 的 `gradient_checkpointing`、TRL 的 `OffloadActivations`、PyTorch checkpoint wrapper、legacy global patch、LayerOffloadManager hook、FSDP/Accelerate env。理解这个特性，首先要理解配置不是终点，而是路由器。

## 关键词二：重计算与迁移并存

Gradient checkpointing 用重计算换激活显存；activation offloading 用 CPU/GPU 迁移换 saved tensor 显存；layer offloading 用按层参数迁移换 frozen weight 常驻显存。它们能叠加，但代价也会叠加：重算、CPU pinned memory、PCIe/NVLink 带宽、stream wait、FSDP collectives 都可能成为瓶颈。

## 关键词三：hook / wrapper / patch 三种接入方式

Axolotl 没有重写模型 forward 主体，而是选择三种低侵入接入：

- Transformers gradient checkpointing layer 自身的 `_gradient_checkpointing_func`。
- PyTorch/TRL context manager 与 saved tensor hooks。
- 进程级 monkey patch 和 per-layer forward/backward hooks。

这让功能能覆盖很多模型，但也让路径判断变复杂。

## 关键词四：兼容性优先，但需要测试兜底

源码里能看到大量兼容性处理：Gemma4、Nemotron-H、MPT、EBFT、ZeRO-3、FSDP2、Liger output head 排除、FP8 tensor skip、DTensor storage 过滤。这说明该实现不是理论上的单一算法，而是深度学习工程里不断补齐的兼容层。

适合使用这些特性的场景：

- 长序列 SFT / full fine-tuning，优先考虑 `gradient_checkpointing` 和 `activation_offloading`。
- LoRA/QLoRA 大模型，且 GPU 显存主要被 frozen base weights 占用，可尝试 `layer_offloading`。
- FSDP2 大模型训练，优先明确区分 `fsdp_config.activation_checkpointing` 和 Axolotl `activation_offloading`。

不太适合的场景：

- 极端依赖吞吐、显存并不紧张的训练；重计算和 CPU/GPU copy 会拖慢速度。
- 特殊模型结构不含标准 decoder `ModuleList`，layer offload 可能找不到层。
- 对 checkpoint/resume 和训练后同进程推理有严格要求的流程；当前 layer hooks/legacy patch cleanup 需要额外注意。

与替代方案相比，这套实现的取舍是：**以低侵入集成换维护复杂度，以本地迁移/重计算换单卡显存，以兼容多个后端换更复杂的配置语义**。后续值得继续走读的方向包括：FSDP2 `fully_shard` 与 activation checkpointing 的准确通信时机、DeepSpeed ZeRO-3 与 checkpoint wrapper 的交互、TRL `OffloadActivations` 在复杂 autograd 图下的 tracker 生命周期，以及 layer offloading 和 optimizer/compile/FSDP 的组合边界。

> 💡 **小结**
>
> * Axolotl 的实现不是单一算法，而是一组配置路由 + wrapper/hook/patch 的组合。
> * 真正要判断是否省显存，必须区分激活、saved tensor、frozen 参数、optimizer state、FSDP shard 五类对象。
> * 当前最值得补强的是 disk offload 路径一致性、layer offload 测试覆盖、训练结束 cleanup 与显存/性能回归验证。
