# Axolotl 源码走读：FSDP2 / TP / ZeRO-3 下的 loss 层通信实现解析

在普通单卡训练里，Cut Cross Entropy（下文简称 CCE）看起来是一个很“清爽”的优化：不再显式生成 `[batch, seq, vocab]` 的 logits，而是直接用 hidden states 和 `lm_head.weight` 计算交叉熵。问题在于，大模型训练很少停留在普通单卡。只要把 FSDP2、Tensor Parallel（TP）或 DeepSpeed ZeRO-3 加进来，loss 层就不再只是一个 kernel 替换问题，而变成了“谁持有 `lm_head.weight`、谁持有完整 vocab、谁负责把梯度还原到正确并行语义”的协同问题。

本文不讲 CCE 论文原理，也不展开 FSDP、ZeRO 或 Megatron TP 的通用教程；我们只顺着 Axolotl 当前源码，把“用户打开 CCE 后，到底哪段 forward 被替换、loss kernel 在并行下触发哪些通信、保存阶段又会遇到什么额外成本”这条主线走完。

> 说明：Axolotl 通过安装提示固定依赖 `axolotl-ai-cloud/ml-cross-entropy.git@fec1a88`，下文涉及 CCE 核心 kernel 的源码行号来自本次本地克隆的 `ml-cross-entropy@fec1a888e6f4ad7e6270ea7b02186e56c76f5ac2`。

# 前言

## 业务 / 工程背景

LLM 训练的最后一层 loss 往往很尴尬：Transformer 主体已经可以用 FlashAttention、activation checkpointing、FSDP/ZeRO、TP 等手段压显存，但语言模型头部仍然天然面对一个巨大 vocab。普通 causal LM forward 会先算：

```text
hidden_states: [B, T, H]
lm_head.weight: [V, H]
logits = hidden_states @ lm_head.weight.T  -> [B, T, V]
loss = cross_entropy(logits, labels)
```

当 `V` 是十几万、`B*T` 又因为长序列和 packing 变大时，`[B, T, V]` 很容易成为训练 step 的显存峰值。CCE 的价值正是在这里：它把 `lm_head + cross_entropy` 融合成“只计算 log-sum-exp 和目标 token logit”的 kernel，避免常驻完整 logits。

但组合并行会改变这个直觉：

* FSDP2 / ZeRO-3 会分片参数，loss 层不一定随时拿得到完整 `lm_head.weight`。
* TP 会按 vocab 或线性层维度切 `lm_head.weight`，单个 rank 只能看到一段 vocab。
* CCE 又恰好绕过了常规 `lm_head(hidden_states)`，因此它必须自己补齐 loss 所需的通信语义。

## 核心矛盾

这条特性的核心矛盾可以压缩成三句话：

1. **CCE 想消灭 logits，但 cross entropy 的数学语义仍然需要全局 vocab 的归一化。**
2. **FSDP2 / ZeRO-3 想分片参数，但 CCE backward 仍然需要与 `lm_head.weight` 一致的梯度。**
3. **TP 想把 vocab 切到多个 rank 上，但 loss 的 LSE、正确类别 logit、hidden gradient 必须跨 TP group 对齐。**

所以，单卡 CCE 是“kernel 优化”；组合并行下的 CCE 是“kernel + process group + 参数生命周期”的协同。

## 本文主线

本文按机制而不是按文件展开：

1. 用户入口与配置归一化：CCE 为什么必须在模型加载前 patch。
2. 单卡 CCE 主路径：它没有改 Trainer，而是改模型 `forward()`。
3. 组合并行下的 loss 通信：FSDP2、TP、ZeRO-3 分别改变了什么。
4. 完整主路径串联：从 `axolotl train` 到 loss kernel 和保存。
5. shape / rank / state 流：张量形状、mesh、全局状态如何变化。
6. 深挖 patch、通信原语、配置归一化、显存与测试缺口。

## 不展开的内容

本文不讲 FSDP2、ZeRO-3、TP、FlashAttention、LoRA 的理论推导；只解释 Axolotl 如何把 CCE 接入现有训练链路，以及这件事在并行训练中带来的收益、边界和风险。

## 核心文件表

这里只列主线文件，不是完整文件索引。

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/config.py` | 读取 YAML、注册 plugin、合并 plugin schema、触发配置校验与环境准备 |
| `src/axolotl/integrations/base.py` | `PluginManager` 单例，负责在模型加载前调用 plugin hook |
| `src/axolotl/integrations/cut_cross_entropy/__init__.py` | Axolotl 侧 CCE plugin：检查依赖、注册模型 forward patch |
| `src/axolotl/integrations/cut_cross_entropy/args.py` | CCE 配置校验：dtype 与 chunked CE 互斥 |
| `src/axolotl/loaders/model.py` | 模型加载主路径，构造 DeviceMesh、传入 TP 参数、处理 ZeRO-3/FSDP2 加载 |
| `src/axolotl/utils/trainer.py` | FSDP、DeepSpeed、ParallelismConfig 环境变量初始化 |
| `src/axolotl/utils/distributed.py` | 根据 `dp_shard/tp/cp` 构建 Accelerate `ParallelismConfig` 与 DeviceMesh |
| `src/axolotl/core/trainers/base.py` | Axolotl Trainer 的 loss 入口；标准路径最终回到 HF Trainer 的 `model(**inputs)` |
| `src/axolotl/monkeypatch/accelerate/fsdp2.py` | FSDP2 prepare/save/load patch，处理 DTensor 与 state dict |
| `ml-cross-entropy@fec1a88/cut_cross_entropy/*` | CCE forward patch、linear CE、自定义 autograd、TP vocab-parallel 通信 |

# 一、入口与配置归一化：为什么 loss 优化必须先于模型加载发生

## 1.1 设计哲学与核心问题

CCE 在 Axolotl 里不是一个 Trainer 参数，也不是一个 `compute_loss` 回调。它的真正入口是 plugin。这个设计的关键原因是：CCE 需要替换模型类的 `forward()`，而这个替换必须发生在 `AutoModelForCausalLM.from_pretrained()` 之前或至少在模型实例参与训练之前。

如果等 Trainer 创建后再改 loss，就已经错过了两个关键点：

* 模型自身 forward 已经决定了是否 materialize logits；
* FSDP2 / ZeRO-3 / TP 的 wrapping 可能已经包住模型，后续再 patch 类方法会更难判断作用范围。

因此 Axolotl 把 CCE 设计成“模型加载前 hook”：用户在 YAML 里写 plugin，配置加载阶段注册 plugin，模型加载阶段执行 `pre_model_load()`，再进入 `from_pretrained()`。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/config.py
  - load_cfg：读取 YAML，调用 prepare_plugins、validate_config、prepare_optim_env
  - prepare_plugins：根据 cfg.plugins 注册 plugin，并调用 plugin.register

src/axolotl/integrations/base.py
  - PluginManager.register：按字符串 import plugin
  - PluginManager.pre_model_load：模型加载前遍历 plugin hook

src/axolotl/integrations/cut_cross_entropy/__init__.py
  - CutCrossEntropyPlugin.get_input_args：把 CCE 参数类并入 Axolotl schema
  - CutCrossEntropyPlugin.pre_model_load：检查依赖并调用 cce_patch

src/axolotl/integrations/cut_cross_entropy/args.py
  - CutCrossEntropyArgs：校验 fp16/bf16 与 chunked CE 互斥
```

## 1.3 主流程拆解

用户最小开启方式不是单独写 `cut_cross_entropy: true`，而是加载 CCE plugin：

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
# cut_cross_entropy 默认由 plugin args 提供，显式写 true 更直观
cut_cross_entropy: true
bf16: true
```

真实入口链路如下：

```text
axolotl train config.yml
  -> src/axolotl/cli/main.py:train()                 # Click 命令入口，约 98-120 行
    -> src/axolotl/cli/train.py:do_cli()              # 63 行 load_cfg
      -> src/axolotl/cli/config.py:load_cfg()         # 230-346 行
        -> prepare_plugins(cfg)                       # 306 行
          -> PluginManager.register(plugin_name)      # integrations/base.py:370-383
          -> plugin.register(cfg)                     # cli/config.py:219-220
        -> validate_config(cfg)                       # cli/config.py:308-320
        -> prepare_optim_env(cfg)                     # cli/config.py:326
        -> plugin_set_cfg(cfg)                        # cli/config.py:333
    -> do_train(parsed_cfg, cli_args)
      -> train(cfg, dataset_meta)
        -> setup_model_and_tokenizer(cfg)
          -> ModelLoader.load()
            -> PLUGIN_MANAGER.pre_model_load(cfg)     # loaders/model.py:172-174
              -> CutCrossEntropyPlugin.pre_model_load # CCE plugin:86-103
```

CCE plugin 在 `pre_model_load()` 中做三件事：

1. `cfg.cut_cross_entropy` 为真才继续；
2. `_check_requirements()` 检查 PyTorch 版本、`cut_cross_entropy` 包、`transformers` support，以及 Axolotl fork 标记 `AXOLOTL_CCE_FORK`（`src/axolotl/integrations/cut_cross_entropy/__init__.py:50-84`）；
3. 调用 `cce_patch(cfg.model_config_type, remote_model_id=...)`（同文件 92-103 行）。

这里的安装信息也透露了一个重要事实：Axolotl 不是依赖 Apple 原仓的普通版本，而是要求自己的 fork：

```text
pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@fec1a88"
```

这条字符串就在 `src/axolotl/integrations/cut_cross_entropy/__init__.py:36-39`。

## 1.4 关键细节与误区澄清

> 容易误解一：`cut_cross_entropy: true` 看起来像普通 schema 字段，但没有加载 plugin 时，Axolotl 不会自动 patch 模型。

源码依据是 `CutCrossEntropyPlugin.get_input_args()` 返回 `axolotl.integrations.cut_cross_entropy.CutCrossEntropyArgs`（`__init__.py:47-48`），而 plugin schema 是在 `prepare_plugins()` 之后动态合并进配置类的（`src/axolotl/integrations/config.py:27-57`）。也就是说，CCE 的配置字段和行为都挂在 plugin 生命周期上。

> 容易误解二：CCE 和 `chunked_cross_entropy` 都是 loss 显存优化，似乎可以叠加。

不能。`CutCrossEntropyArgs.check_chunked_cross_entropy_not_set()` 在 `src/axolotl/integrations/cut_cross_entropy/args.py:46-54` 明确拒绝 `chunked_cross_entropy`。两者都想接管 loss 计算路径，Axolotl 选择让 CCE 独占这条路径。

> 容易误解三：CCE 是运行时每 step 动态开关。

不是。Axolotl 侧是在模型加载前做一次 patch；CCE fork 内部再用 module-level `_PATCH_OPTS` 控制是否在有 labels 且满足训练/eval条件时走 LCE。这个状态是进程内全局的，不是每个 batch 新建的对象。

## 1.5 本章小结

> 💡 **小结**
>
> * CCE 在 Axolotl 中是 plugin 驱动的模型 forward patch，不是 Trainer 的 `compute_loss` 插件。
> * plugin 必须在模型加载前执行，才能避免原始模型 forward 先 materialize logits。
> * `bf16/fp16` 是硬约束，`chunked_cross_entropy` 与 CCE 互斥。
> * CCE 的真实核心实现来自 Axolotl fork 的 `ml-cross-entropy@fec1a88`。

# 二、单卡 CCE 主路径：不是改 Trainer，而是改模型 forward

## 2.1 设计哲学与核心问题

理解组合并行前，必须先看单卡主路径。单卡 CCE 的本质不是“换一个 cross entropy 函数”，而是绕过 `logits = lm_head(hidden_states)` 这一步。

在普通 HF causal LM 中，模型 forward 通常返回：

```text
loss:   scalar
logits: [B, T, V]
```

CCE patch 后，在 labels 存在且启用 LCE 的路径上，模型 forward 返回：

```text
loss:   scalar
logits: None
```

这不是小差异。它意味着后续若有 callback 或 eval 逻辑依赖 logits，需要确认它是否处在 `prediction_loss_only` 或无 labels 生成路径。Axolotl 的标准训练 step 只需要 loss，所以这条路径能成立。

## 2.2 源码入口与关键对象

```text
src/axolotl/core/trainers/base.py
  - AxolotlTrainer.compute_loss：标准 SFT 大多最终调用 HF Trainer.compute_loss

ml-cross-entropy@fec1a88/cut_cross_entropy/transformers/patch.py
  - cce_patch：根据 model_type 找到模型专属 patch 函数

ml-cross-entropy@fec1a88/cut_cross_entropy/transformers/llama.py
  - cce_forward：以 LLaMA 类模型为代表的 patched forward

ml-cross-entropy@fec1a88/cut_cross_entropy/transformers/utils.py
  - apply_lce：把 hidden_states、lm_head.weight、labels 转成 linear_cross_entropy 调用

ml-cross-entropy@fec1a88/cut_cross_entropy/cce.py
  - LinearCrossEntropyFunction：自定义 autograd Function
```

## 2.3 主流程拆解

Axolotl 自己的 `AxolotlTrainer.compute_loss()` 并没有显式调用 CCE。普通 SFT 路径在做完 token 计数、Gemma 特例、ORPO 分支后，直接回到 HF Trainer：

```text
AxolotlTrainer.compute_loss
  -> super().compute_loss(model, inputs, ...)
    -> model(**inputs)  # inputs 中包含 labels
      -> patched_model.forward(..., labels=labels)
```

源码依据是 `src/axolotl/core/trainers/base.py:365-460`：除了 ORPO 和 Gemma4 特例，最后 455-460 行返回 `super().compute_loss(...)`。

以 LLaMA patch 为例，CCE fork 的 `cce_forward()` 逻辑非常直接（`ml-cross-entropy@fec1a88/cut_cross_entropy/transformers/llama.py:53-98`）：

```text
self.model(...) -> hidden_states
if _PATCH_OPTS.use_lce(labels, self.training):
    loss = apply_lce(hidden_states[:, slice_indices, :], self.lm_head.weight, labels, ...)
    logits = None
else:
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = self.loss_function(logits, labels, ...)
return CausalLMOutputWithPast(loss=loss, logits=logits, ...)
```

`apply_lce()` 继续把数据交给 `linear_cross_entropy()`（`transformers/utils.py:97-165`）。到 CCE kernel 层，`cce_linear_cross_entropy()` 做了几件关键事（`cce.py:238-294`）：

```text
e:       [B, T, H]
targets: [B, T]

assert e.size()[0:-1] == targets.size()
e = e.contiguous().flatten(0, -2)      # [B*T, H]
targets = targets.contiguous().flatten() # [B*T]
valids = _build_flat_valids(targets, ignore_index, shift)
LinearCrossEntropyFunction.apply(e, c, bias, params)
```

真正省显存的地方在 `LinearCrossEntropyFunction.forward()`：

* `cce_lse_forward_kernel()` 计算每个 token 的 log-sum-exp（`cce.py:59-76`）；
* `indexed_neg_dot_forward_kernel()` 只计算 label 对应类别的负 dot（`cce.py:100-109`）；
* `nll = neg_dot + lse` 后再 reduction（`cce.py:120-130`）；
* 保存 backward 需要的 `e, c, bias, lse, targets, valids, logit_avg`，而不是保存 `[B*T, V]` logits（`cce.py:132-135`）。

单卡 shape 可以写成：

```text
输入:
  input_ids / labels: [B, T]

模型主体:
  hidden_states: [B, T, H]

CCE flatten:
  e:       [N, H], N = B * T
  c:       [V, H]  # lm_head.weight
  targets: [N]

CCE 中间量:
  lse:      [N_valid]
  neg_dot:  [N_valid]
  loss:     scalar

被避免的中间量:
  logits: [B, T, V]
```

## 2.4 关键细节与误区澄清

> 容易误解一：CCE 是 Axolotl Trainer 的一个 loss function。

不是。Axolotl Trainer 的主路径仍是 `super().compute_loss()`；CCE 靠模型 forward patch 生效。判断 CCE 是否真的生效，要看最终被训练的模型类 forward 是否被替换，而不是只看 Trainer 类。

> 容易误解二：有 labels 时 patched forward 仍会返回 logits。

在 CCE 路径不会。`cce_forward()` 初始化 `logits = None`，只有 `_PATCH_OPTS.use_lce(...)` 为假才计算 `self.lm_head(...)`（`llama.py:64-83`）。因此训练路径节省的是 logits 常驻显存；但如果某个评估/回调需要 logits，就必须确认它是否绕开 labels 或关闭 CCE。

> 容易误解三：所有模型类型都走同一个 LLaMA forward。

不是。CCE fork 的 `PATCH_FNS` 为大量模型注册了专属 patch（`transformers/patch.py:15-88`）。Axolotl 只在 model type 不在 `PATCH_FNS` 时，通过 `patch_llama_like()` 增加一个实验性的 generic fallback（`src/axolotl/integrations/cut_cross_entropy/__init__.py:105-150`）。多模态模型尤其不能假设 LLaMA patch 足够。

## 2.5 本章小结

> 💡 **小结**
>
> * 单卡 CCE 的核心收益来自不 materialize `[B, T, V]` logits。
> * Axolotl 标准 SFT loss 主路径没有显式调用 CCE；模型 forward patch 才是关键。
> * CCE 保存的是 `lse`、targets 等 backward 所需状态，而不是完整 logits。
> * 判断 CCE 是否生效，要看模型类 forward 是否被正确 patch。

# 三、组合并行中的 loss 通信：FSDP2 管参数、TP 管 vocab、ZeRO-3 管生命周期

## 3.1 设计哲学与核心问题

单卡 CCE 避免 logits 后，loss 计算看起来只剩一个本地 kernel。但在组合并行下，`lm_head.weight` 不再是一个简单 Tensor：

* **FSDP2** 关心参数何时 unshard、reshard，以及 state dict 如何收集；
* **TP** 可能把 `lm_head.weight` 的 vocab 维切成 `V/tp_size`；
* **ZeRO-3** 会让参数对象在 forward/backward 不同阶段呈现不同的分片生命周期。

CCE 必须识别这些状态。否则，要么算出的 softmax 只覆盖本地 vocab shard，要么 backward 用到的 `lm_head.weight` 已经被 ZeRO-3 重新分片，要么保存时 rank0 聚合出错。

## 3.2 源码入口与关键对象

```text
src/axolotl/utils/trainer.py
  - setup_fsdp_envs：写入 FSDP2 相关 env
  - setup_deepspeed_env：写入 DeepSpeed / ZeRO stage env
  - setup_parallelism_envs：写入 TP/CP ParallelismConfig env

src/axolotl/utils/distributed.py
  - build_parallelism_config：构建 Accelerate ParallelismConfig 与 DeviceMesh

src/axolotl/loaders/model.py
  - _set_parallel_config：保存 device_mesh
  - _build_model：TP 时传 tp_size、tp_plan、device_mesh 给 Transformers
  - _configure_zero3_memory_efficient_loading：ZeRO-3 预加载 HfTrainerDeepSpeedConfig

src/axolotl/monkeypatch/accelerate/fsdp2.py
  - fsdp2_prepare_model：按 FSDP mesh fully_shard
  - get_state_dict：FSDP2 / DeepSpeed 保存时收集权重

ml-cross-entropy@fec1a88/cut_cross_entropy/transformers/utils.py
  - apply_lce：识别 DTensor TP 与 ZeRO-3 ds_id

ml-cross-entropy@fec1a88/cut_cross_entropy/vocab_parallel/utils.py
  - VocabParallelOptions / vp_reduce_lse / vp_reduce_correct_logit / vp_reduce_e_grad
```

## 3.3 主流程拆解

### 3.3.1 FSDP2：CCE 不负责参数分片，但必须适应 DTensor

FSDP2 和 TP 的并行配置从环境变量开始。`prepare_optim_env()` 会按配置选择 FSDP 或 DeepSpeed，然后总是执行 `setup_parallelism_envs(cfg)`（`src/axolotl/utils/trainer.py:643-667`）。

FSDP2 相关 env 在 `setup_fsdp_envs()` 中写入，例如：

* `ACCELERATE_USE_FSDP=true`；
* `FSDP_VERSION=2`；
* `FSDP_RESHARD_AFTER_FORWARD=true`；
* `FSDP_STATE_DICT_TYPE=...`。

这些位于 `src/axolotl/utils/trainer.py:589-618`。

TP/CP 相关 env 则由 `setup_parallelism_envs()` 写入：

```text
PARALLELISM_CONFIG_TP_SIZE = cfg.tensor_parallel_size
PARALLELISM_CONFIG_CP_SIZE = cfg.context_parallel_size
ACCELERATE_USE_PARALLELISM_CONFIG = true
```

源码在 `src/axolotl/utils/trainer.py:621-640`。

模型加载时，`ModelLoader._set_parallel_config()` 调用 `build_parallelism_config()`（`src/axolotl/loaders/model.py:437-443`），后者用当前 world size 和 `tensor_parallel_size/context_parallel_size/dp_shard_size/dp_replicate_size` 创建 Accelerate `ParallelismConfig` 并 `build_device_mesh("cuda")`（`src/axolotl/utils/distributed.py:299-316`）。

真正把 TP 交给 Transformers 的地方在 `_build_model()`：

```text
if cfg.tensor_parallel_size > 1:
    model_kwargs["tp_size"] = cfg.tensor_parallel_size
    model_kwargs["tp_plan"] = "auto"
    model_kwargs["device_mesh"] = self.device_mesh
```

对应 `src/axolotl/loaders/model.py:749-755`。

FSDP2 wrapping 则在 Axolotl patch 过的 `fsdp2_prepare_model()` 中完成。它取 `accelerator.state.device_mesh`，并把 FSDP 用的 mesh 设为：

```text
mesh[tuple(accelerator.state.parallelism_config.fsdp_dim_names)]
```

源码在 `src/axolotl/monkeypatch/accelerate/fsdp2.py:344-361`。Accelerate 的 `fsdp_dim_names` 会包含 `dp_replicate` 和一个扁平的 `dp_shard_cp` 维（本地安装的 `accelerate/parallelism_config.py:157-164`），而 TP 维仍单独叫 `
`。这就是后面 CCE 可以从 DTensor 的 `device_mesh.get_group("tp")` 找到 TP process group 的前提。

### 3.3.2 TP：loss 不再看到完整 vocab，需要自己 all-reduce

CCE fork 在 `apply_lce()` 中先判断 `lm_head.weight` 是否是 DTensor：

```text
if isinstance(c, DTensor):
    device_mesh = c.device_mesh
    process_group = device_mesh.get_group("tp")
    placement = c.placements[0]
    if isinstance(placement, Shard):
        vocab_parallel_options = VocabParallelOptions.from_vocab(
            vocab_size, process_group, reduce_e_grad=True
        )
    c_local = c.to_local()
else:
    c_local = c
```

源码位于 `ml-cross-entropy@fec1a88/cut_cross_entropy/transformers/utils.py:113-135`。

这段代码说明 CCE 并没有让 DTensor 继续流入 `linear_cross_entropy()`。事实上 `linear_cross_entropy()` 自己会拒绝 DTensor 输入（`linear_cross_entropy.py:62-67`）。Axolotl fork 的策略是：在 Transformers patch 层识别 DTensor，提取本地 shard，同时把 TP group 和 vocab range 编码进 `VocabParallelOptions`。

到了 CCE autograd forward，TP loss 的语义分成三步：

1. 每个 rank 用本地 `c_local: [V_i, H]` 计算局部 LSE；
2. `vp_reduce_lse()` 先 all-reduce max，再 all-reduce exp-sum，得到全局 LSE；
3. 只有目标 token 落在本 rank vocab range 时才计算 correct logit，然后 `vp_reduce_correct_logit()` all-reduce 到所有 TP rank。

源码依据：

* `LinearCrossEntropyFunction.forward()` 在 `cce.py:75-118` 调用 `vp_reduce_lse()` 与 `vp_reduce_correct_logit()`；
* `vp_reduce_lse()` 的两次 all-reduce 在 `vocab_parallel/utils.py:47-54`；
* `vp_reduce_correct_logit()` 的 all-reduce 在 `vocab_parallel/utils.py:57-65`。

TP 下的 forward shape 可以写成：

```text
全局语义:
  E: [N, H]
  C: [V, H]

TP rank i:
  C_i: [V_i, H]
  local_lse_i: [N]
  local_correct_i: [N]  # 只有 target 在 [start_i, stop_i) 时非零

跨 TP group:
  all_reduce(MAX): local_lse_i -> lse_max
  all_reduce(SUM): exp(local_lse_i - lse_max) -> global_lse
  all_reduce(SUM): local_correct_i -> global_correct

输出:
  loss = global_lse - global_correct
```

Backward 也不是完全本地的。由于每个 TP rank 的 `C_i` 都参与了同一个 hidden vector `E` 的 softmax，`dE` 必须跨 TP rank 求和。CCE 在 `cce_backward_kernel()` 后，如果 `reduce_e_grad=True`，会调用 `vp_reduce_e_grad()`（`cce_backward.py:474-475`），而 `vp_reduce_e_grad()` 内部是一次 all-reduce（`vocab_parallel/utils.py:68-76`）。

因此，TP 下 CCE 每次 loss 大致新增：

```text
forward:
  all_reduce(MAX) on [N]
  all_reduce(SUM) on [N]
  all_reduce(SUM) on [N]

backward:
  all_reduce(SUM) on dE [N, H]
```

注意它仍然没有 all-gather `[N, V]` logits；通信对象从“大矩阵 logits”变成了几个向量和 hidden gradient。

### 3.3.3 ZeRO-3：不是 vocab-parallel，而是参数生命周期问题

DeepSpeed ZeRO-3 的问题不同。它不一定把 `lm_head.weight` 表达成 DTensor；它会给参数对象挂上 DeepSpeed 元数据，例如 `ds_id`。CCE fork 在 `apply_lce()` 中检查：

```text
zero3_params = []
if hasattr(c, "ds_id"):
    zero3_params.append(c)
if bias is not None and hasattr(bias, "ds_id"):
    zero3_params.append(bias)
if zero3_params:
    cce_kwargs["zero3_params"] = zero3_params
```

源码在 `ml-cross-entropy@fec1a88/cut_cross_entropy/transformers/utils.py:137-147`。

真正的特殊处理发生在 backward。CCE forward 保存下来的 `c` / `bias` tensor 到 backward 时可能已经变成 ZeRO-3 的局部分片或 stale reference，因此 `LinearCrossEntropyFunction.backward()` 会在调用 Triton backward kernel 前进入 DeepSpeed 的 `GatheredParameters`：

```text
if params.zero3_params:
    gather_ctx = GatheredParameters(params.zero3_params, modifier_rank=None)
else:
    gather_ctx = nullcontext()

with gather_ctx:
    if params.zero3_params:
        c = params.zero3_params[0].data
        bias = params.zero3_params[1].data if exists
    de, dc, dbias = cce_backward_kernel(...)
```

源码在 `ml-cross-entropy@fec1a88/cut_cross_entropy/cce.py:179-218`。

这就是 ZeRO-3 下 CCE 的 loss 层通信：它不是为了全局 softmax 做 TP all-reduce，而是为了在 backward kernel 执行期间重新拿到完整 `lm_head.weight/bias`。这个 gather 的成本取决于 `lm_head` 的大小和 ZeRO-3 参数分片策略。

## 3.4 关键细节与误区澄清

> 容易误解一：FSDP2 开启后，CCE 一定会看到 DTensor vocab shard。

不一定。FSDP2 负责参数在 FSDP mesh 上的 shard/unshard；CCE 的 DTensor vocab-parallel 分支是由 TP 维触发的。源码上，CCE 判断的是 `isinstance(c, DTensor)` 且 placement 是 `Shard`（`transformers/utils.py:113-131`），不是判断 `cfg.fsdp_version == 2`。

> 容易误解二：TP 下 CCE 会 all-gather logits。

不会。CCE TP 分支用 all-reduce 汇总 LSE 和正确类别 logit，用 all-reduce 汇总 `dE`，没有构造 `[N, V]` logits。它是通信换显存，但通信对象比 logits 小得多。

> 容易误解三：ZeRO-3 下的 `GatheredParameters` 是 forward 阶段发生的 CCE 主通信。

从 CCE 源码看，`GatheredParameters` 位于自定义 autograd 的 backward 中（`cce.py:179-218`）。forward 只把 `zero3_params` 记录进 params；真正 gather 是 backward kernel 前。

> 容易误解四：`tensor_parallel_size` 在 schema 中写着“Only supported with DeepSpeed AutoTP”，所以 FSDP2+TP 不走 Transformers TP。

源码和文档/描述存在历史痕迹。`src/axolotl/utils/schemas/config.py:993-997` 的描述仍写 DeepSpeed AutoTP，但 `ModelLoader._build_model()` 确实在 `tensor_parallel_size > 1` 时传入 `tp_size/tp_plan/device_mesh` 给 Transformers（`src/axolotl/loaders/model.py:749-755`），而 `docs/nd_parallelism.qmd:96-103` 也列出了 FSDP+TP、FSDP+TP+CP 支持矩阵。本文以源码主路径为准。

## 3.5 本章小结

> 💡 **小结**
>
> * FSDP2、TP、ZeRO-3 改变的是 `lm_head.weight` 的可见形态与生命周期。
> * TP 下 CCE 用 LSE / correct logit / dE all-reduce 保持全局 cross entropy 语义。
> * ZeRO-3 下 CCE 在 backward 用 `GatheredParameters` 重新聚合 loss kernel 所需权重。
> * FSDP2 本身不是 CCE vocab-parallel 的触发条件；DTensor + TP group 才是。

# 四、完整主路径串联

## 4.1 完整调用栈

以一个包含 CCE、FSDP2、TP 的配置为例：

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
cut_cross_entropy: true
bf16: true

tensor_parallel_size: 2
dp_shard_size: 4
fsdp_version: 2
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3DecoderLayer
  reshard_after_forward: true
```

一次真实调用可以串成：

```text
User: axolotl train config.yml
  │
  ├─ Step 1: 配置加载与 plugin 注册
  │     └─ cli/config.py:load_cfg
  │        ├─ prepare_plugins(cfg)
  │        ├─ validate_config(cfg)
  │        ├─ prepare_optim_env(cfg)
  │        └─ normalize_config(cfg)
  │
  ├─ Step 2: 分布式环境与 DeviceMesh
  │     └─ utils/trainer.py:prepare_optim_env
  │        ├─ setup_fsdp_envs(cfg)
  │        ├─ setup_deepspeed_env(cfg)   # DeepSpeed 路径二选一
  │        └─ setup_parallelism_envs(cfg)
  │
  ├─ Step 3: 模型加载前 patch
  │     └─ loaders/model.py:ModelLoader.load
  │        ├─ PatchManager.apply_pre_model_load_patches()
  │        ├─ ModelLoader._set_parallel_config()
  │        ├─ PLUGIN_MANAGER.pre_model_load(cfg)
  │        │   └─ CutCrossEntropyPlugin.pre_model_load
  │        │      └─ cce_patch(model_type)
  │        └─ AutoModelForCausalLM.from_pretrained(...)
  │
  ├─ Step 4: wrapping / prepare
  │     └─ Accelerate / Axolotl FSDP2 patch
  │        └─ fsdp2_prepare_model(... fully_shard ...)
  │
  ├─ Step 5: 训练 step
  │     └─ AxolotlTrainer.compute_loss
  │        └─ HF Trainer.compute_loss
  │           └─ patched_model.forward(labels=...)
  │              └─ apply_lce(hidden_states, lm_head.weight, labels)
  │                 ├─ TP: vocab-parallel all-reduce
  │                 ├─ ZeRO-3: record zero3_params
  │                 └─ CCE Triton autograd Function
  │
  └─ Step 6: 保存
        └─ train.py:save_trained_model
           ├─ FSDP2: trainer.save_model + Accelerator.get_state_dict patch
           ├─ ZeRO-3: trainer.save_model + proxy cleanup
           └─ SHARDED_STATE_DICT: merge_sharded_fsdp_weights
```

## 4.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 是否通信 | 是否每 step 执行 |
|---|---|---|---|---|
| 配置加载 | YAML + CLI overrides | `DictDefault`，plugin schema 合并，dtype 校验 | 否 | 否，仅启动 |
| 环境准备 | `fsdp_config/deepspeed/tensor_parallel_size` | env vars，Accelerate/DeepSpeed 初始化条件 | DeepSpeed 单卡初始化可能 init process group | 否，仅启动 |
| CCE patch | `cfg.model_config_type` | 替换目标模型类 `forward`，设置 CCE `_PATCH_OPTS` | 否 | 否，模型加载前一次 |
| DeviceMesh | world size + dp/tp/cp sizes | `ParallelismConfig`、`device_mesh` | 可能由 torch distributed 初始化 group | 否，启动/Accelerate prepare |
| 模型加载 | base model + model_kwargs | TP 参数传给 Transformers，FSDP2/ZeRO3 加载策略生效 | ZeRO/FSDP 加载可能广播/分发 | 否，启动/恢复 |
| 训练 forward | batch tensors | hidden states、本地或全局 loss | TP loss all-reduce；FSDP/ZeRO 参数通信 | 是，每 step |
| 训练 backward | scalar loss | dE/dC/dbias，参数梯度 | TP dE all-reduce；ZeRO3 gather；FSDP grad 通信 | 是，每 step |
| 保存 | wrapped model | full 或 sharded checkpoint | FSDP full_tensor / ZeRO consolidate | 否，保存时 |

## 4.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `trainer_weighted_loss()` in `src/axolotl/utils/trainer.py:78-87` | 名字里有 loss，且手写 CE | 标准 CCE SFT 不走 | `AxolotlTrainer.compute_loss()` 中 sample packing 的 weighted loss 分支被注释（`base.py:369-374`） |
| `patch_chunked_ce_loss_fn()` | 也是 loss 显存优化 | CCE 开启时被 schema 拒绝 | `CutCrossEntropyArgs` 明确禁止 `chunked_cross_entropy` |
| `PatchManager._apply_patch_deepspeed_zero3()` | 名字像 ZeRO-3 loss patch | 只在 `activation_offloading=True` 且 ZeRO-3 时 patch checkpoint wrapper | CCE 的 ZeRO-3 处理在 CCE fork backward 的 `GatheredParameters` |
| `merge_sharded_fsdp_weights` | 看起来影响训练中 loss | 不在训练 step | 只在保存/合并 sharded FSDP checkpoint 时运行 |
| `preprocess` CLI | 数据预处理会生成 labels | 不触发模型 forward / CCE kernel | CCE 是训练/模型 forward 路径，不是 dataset preprocess 路径 |
| `linear_cross_entropy()` 直接接收 DTensor | 以为 CCE kernel 原生支持 DTensor | 直接调用会报错 | Axolotl fork 在 `apply_lce()` 先把 DTensor 转 local shard，再传 `VocabParallelOptions` |
| 无 labels 的 generate/inference | 模型已 patch，所以以为仍走 CCE | 不走 LCE | `_PATCH_OPTS.use_lce()` 在 labels 为 None 时返回 False（`transformers/utils.py:87-94`） |

## 4.4 本章小结

> 💡 **小结**
>
> * CCE 的主路径横跨 config、plugin、model loader、patched forward、autograd kernel。
> * 训练 step 中 CCE 只在 labels 存在时接管 loss；无 labels 推理仍回到 logits 路径。
> * TP/ZeRO-3 的通信发生在 CCE fork 内部，FSDP2 的参数通信由 Accelerate/FSDP wrapping 负责。
> * 保存阶段不是 CCE 的功能，但 FSDP2/ZeRO-3/TP 会改变保存通信和 rank0 聚合成本。

# 五、关键数据流 / 状态流 / shape 流程

## 5.1 Tensor shape 变化

单卡 CCE 的 shape 变化：

```text
原始 batch:
  input_ids: [B, T]
  labels:    [B, T]

Transformer body:
  hidden_states: [B, T, H]

CCE:
  e = hidden_states.flatten(0, -2): [N, H]
  c = lm_head.weight:               [V, H]
  targets = labels.flatten():       [N]
  lse:                              [N_valid]
  correct_logit:                    [N_valid]
  loss:                             scalar

不再常驻:
  logits: [B, T, V]
```

TP 下，`c` 变成每个 rank 的本地 shard：

```text
全局:
  c: [V, H]

TP size = P
rank i:
  c_i: [V_i, H]
  vocab range: [start_i, stop_i)

局部计算:
  local_lse_i:     [N_valid]
  local_correct_i: [N_valid]

通信后:
  global_lse:      [N_valid]
  global_correct:  [N_valid]
  loss:            scalar
```

显存收益来自没有 `[N, V]` logits；通信成本来自 `[N]` 向量 all-reduce 和 `[N, H]` 的 `dE` all-reduce。

ZeRO-3 下，shape 不一定变成 vocab shard，但参数生命周期变了：

```text
forward 时:
  c 参数对象可能由 DeepSpeed gather 后参与计算
  CCE 记录 zero3_params=[lm_head.weight, maybe bias]

backward 时:
  saved c 可能是 stale/local shard
  with GatheredParameters(zero3_params):
      c = full lm_head.weight.data
      run cce_backward_kernel
```

这一步节省的不是 logits，而是配合 ZeRO-3 参数分片，避免 backward 用错局部分片。

## 5.2 Rank / Mesh / Process Group 变化

以 `world_size=8, dp_shard_size=2, tensor_parallel_size=4` 为例，Accelerate 的 canonical mesh order 是 `dp_replicate -> dp_shard -> cp -> sp -> tp`（本地 `accelerate/parallelism_config.py:266-272`）。无 `dp_replicate/cp/sp` 时，可以理解为：

```text
mesh shape: [dp_shard=2, tp=4]

          tp0   tp1   tp2   tp3
dp0       r0    r1    r2    r3
dp1       r4    r5    r6    r7
```

对应 group 语义：

```text
TP group for dp0: r0, r1, r2, r3
TP group for dp1: r4, r5, r6, r7

FSDP dp_shard groups roughly按 tp 坐标成列:
  group tp0: r0, r4
  group tp1: r1, r5
  group tp2: r2, r6
  group tp3: r3, r7
```

CCE 只关心 `device_mesh.get_group("tp")`。FSDP2 patch 则把 FSDP mesh 设置成 `mesh[fsdp_dim_names]`，即 DP/FSDP 相关维度（`src/axolotl/monkeypatch/accelerate/fsdp2.py:351-360`）。这两个 group 是不同维度，通信不能混为一谈。

## 5.3 状态切换

这个特性涉及几类状态：

```text
配置状态:
  cfg.plugins / cfg.cut_cross_entropy / cfg.tensor_parallel_size / cfg.fsdp_config

进程环境变量:
  ACCELERATE_USE_FSDP
  FSDP_VERSION
  ACCELERATE_USE_DEEPSPEED
  ACCELERATE_DEEPSPEED_ZERO_STAGE
  PARALLELISM_CONFIG_TP_SIZE
  ACCELERATE_USE_PARALLELISM_CONFIG

进程内全局 patch:
  transformers.models.<model>.modeling_<model>.<Class>.forward = cce_forward
  cut_cross_entropy.transformers.<model>._PATCH_OPTS = PatchOptions(...)
  accelerate.Accelerator.get_state_dict = axolotl patched get_state_dict

运行时对象状态:
  accelerator.state.device_mesh
  accelerator.state.parallelism_config
  DeepSpeed parameter ds_id / ZeRO partition state
  FSDP2 DTensor placements
```

这些状态大多是进程级的，不是线程局部变量。训练通常是单进程单 GPU rank 模式，所以可控；但在测试套件或同一 Python 进程内连续加载多个不同模型时，patch 的全局性会成为维护风险。

## 5.4 本章小结

> 💡 **小结**
>
> * CCE 消灭的是 `[N, V]` logits，不是所有 loss 相关中间量。
> * TP 下 loss 语义通过 `[N]` LSE/correct all-reduce 与 `[N,H]` dE all-reduce 保持一致。
> * FSDP2 和 TP 使用不同 mesh 维度；CCE 只读取 TP group。
> * CCE、FSDP2、Accelerate 的 patch 都是进程级状态，生命周期需要谨慎理解。

# 六、核心机制深挖

## 6.1 Monkey Patch：零侵入接入还是维护风险？

### 它解决什么问题

Axolotl 不想 fork 每个 Transformers 模型类，也不想要求用户换 Trainer。plugin patch 的好处是：用户只需在 YAML 中加入 plugin，模型 forward 就被替换成 CCE 版本。

### 为什么不能更简单

如果只在 Trainer 层拿 `outputs.logits` 后换 loss，就已经 materialize 了 `[B,T,V]`，显存收益消失。CCE 必须在 logits 生成之前接管，所以只能在模型 forward 或模型 head 层介入。

### 源码如何实现

Axolotl 侧：

* `CutCrossEntropyPlugin.pre_model_load()` 调用 `patch_llama_like()` 和 fork 的 `cce_patch()`（`src/axolotl/integrations/cut_cross_entropy/__init__.py:86-103`）；
* 对未知 llama-like 类型，`patch_llama_like()` 直接给 `PATCH_FNS` 塞入 generic patch，并执行 `model_cls.forward = cce_forward`（同文件 114-150 行）。

CCE fork 侧：

* `PATCH_FNS` 映射 model type 到 patch 函数（`transformers/patch.py:15-88`）；
* `cce_patch()` 构造 `PatchOptions`，再调用具体 patch（`transformers/patch.py:151-199`）；
* 以 LLaMA 为例，`patch_llama()` 设置 module-level `_PATCH_OPTS` 并替换类或实例 forward（`transformers/llama.py:101-128`）。

### 隐藏假设与副作用

隐藏假设是：模型类 forward 签名、返回对象、`self.model`、`self.lm_head.weight` 等结构与 patch 代码匹配。上游 Transformers 一旦改 forward 参数，patch 就可能失效。

副作用是：patch 默认没有 restore。对单次训练无所谓，但对同进程多模型测试会污染后续模型类型。

## 6.2 通信原语：前向和反向是否对称？

TP 下不完全对称。

Forward 做的是 loss 标量语义还原：

```text
local logits not materialized globally
  -> local LSE
  -> all_reduce max/sum 得到 global LSE
  -> all_reduce correct logit
```

Backward 做的是 hidden gradient 还原：

```text
local dC_i: 只对应本地 vocab shard
local dE_i: 每个 TP rank 对同一个 E 的部分贡献
  -> all_reduce dE across TP group
```

源码上，forward 的三次 all-reduce 分布在 `cce.py:75-118` 与 `vocab_parallel/utils.py:47-65`；backward 的 dE all-reduce 在 `cce_backward.py:474-475`。

ZeRO-3 下也不对称：forward 只记录 `zero3_params`，backward 才进入 `GatheredParameters`。这是参数生命周期导致的通信，而不是 softmax 数学语义导致的通信。

## 6.3 配置归一化：用户配置如何变成真实行为

几个配置路径尤其值得注意：

* `plugins`：决定 CCE plugin 是否注册；没有 plugin 就没有 patch。
* `cut_cross_entropy`：plugin schema 字段，默认 `True`，但只在 plugin 注册后才有意义。
* `bf16/fp16`：`CutCrossEntropyArgs` 要求其一为真，否则报错（`args.py:35-44`）。
* `tensor_parallel_size`：一方面写入 `PARALLELISM_CONFIG_TP_SIZE`（`utils/trainer.py:623-625`），另一方面模型加载时传 `tp_size/tp_plan/device_mesh`（`loaders/model.py:749-755`）。
* `deepspeed + tensor_parallel_size`：validation 会改写一个临时 DeepSpeed JSON，加入 `tensor_parallel.autotp_size` 与 `gather_16bit_weights_on_model_save`（`utils/schemas/validation.py:1121-1148`）。
* `world_size`：`normalize_config()` 会把 global batch 按 `context_parallel_size` 和 `tensor_parallel_size` 折算有效 data parallel world size（`utils/config/__init__.py:134-142`）。

这说明“配置项是否存在”和“框架是否直接消费”是两回事。比如 DeepSpeed AutoTP JSON 是下游 DeepSpeed 消费；CCE fork 的 TP vocab branch 则是通过 DTensor / DeviceMesh 自己判断。

## 6.4 本章小结

> 💡 **小结**
>
> * CCE 必须 patch forward，因为 Trainer 层接管太晚。
> * TP forward all-reduce 的是 LSE/correct logit，backward all-reduce 的是 dE。
> * ZeRO-3 的特殊通信来自参数重新 gather，而不是 vocab-parallel softmax。
> * 配置会被 Axolotl、Accelerate、Transformers、DeepSpeed 多层消费，不能只看 schema 描述。

# 七、显存、性能与通信分析

## 7.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | CCE 本身 ❌；FSDP2/ZeRO3/TP ✅ | CCE 不分片参数；FSDP2/ZeRO-3 shard 参数，TP shard 线性层/vocab |
| 激活值 | 部分 ✅ | CCE 不保存 logits；Transformer 主体激活仍由 checkpointing/CP 等机制处理 |
| logits `[B,T,V]` | ✅✅ | CCE 最大收益：不 materialize 完整 logits |
| optimizer state | CCE ❌；FSDP2/ZeRO ✅ | optimizer state 分片是 FSDP/ZeRO 的职责 |
| 输入 batch | ❌ | CCE 不改变 dataloader；TP/CP 会影响有效 batch/sequence 分配 |
| LSE / correct buffer | ❌（新增少量） | CCE 需要 `[N]` 级别中间量，比 `[N,V]` 小很多 |
| ZeRO-3 backward gather buffer | 可能 ❌ | backward `GatheredParameters` 会临时聚合 `lm_head.weight/bias` |
| 保存时 full state dict | 可能 ❌ | FSDP2 `full_tensor()` 和 DeepSpeed consolidate 会制造 rank0/CPU 峰值 |

真正的大头是 logits：`B*T*V*dtype_bytes`。例如 `B*T=8192, V=151936, bf16=2 bytes`，单个 logits 就接近 2.5GB，还不算 softmax 临时量。CCE 通过 kernel 避免它，但 TP/ZeRO-3 下为了保持语义会增加通信和临时 gather。

## 7.2 通信开销

| 场景 | 通信类型 | 触发频率 | group | 源码依据 |
|---|---|---:|---|---|
| TP CCE forward | all_reduce MAX `[N]` | 每次 loss forward | TP group | `vocab_parallel/utils.py:47-54` |
| TP CCE forward | all_reduce SUM `[N]` | 每次 loss forward | TP group | `vocab_parallel/utils.py:52-54` |
| TP CCE forward | all_reduce correct logit `[N]` | 每次 loss forward | TP group | `vocab_parallel/utils.py:57-65` |
| TP CCE backward | all_reduce dE `[N,H]` | 每次 loss backward | TP group | `cce_backward.py:474-475` |
| ZeRO-3 CCE backward | GatheredParameters for `lm_head/bias` | 每次 loss backward | DeepSpeed ZeRO group | `cce.py:179-218` |
| FSDP2 forward/backward | 参数 all-gather / grad reduce 等 | 每 wrapped module | FSDP mesh | Axolotl patch 调用 `fully_shard`，具体通信由 PyTorch FSDP2 执行 |
| FSDP2 save | DTensor `full_tensor()` + barrier | 保存时每参数 | FSDP mesh / global | `accelerate/fsdp2.py:158-173` |
| DeepSpeed save | consolidated state dict | 保存时 | DeepSpeed | `accelerate/fsdp2.py:129-157` |

CCE TP 的通信可以 overlap 吗？源码中 `vp_reduce_lse()` 和 `vp_reduce_correct_logit()` 是同步 `torch.distributed.all_reduce`，没有显式 async handle；是否被后端内部 overlap，源码中未确认。ZeRO-3 的 `GatheredParameters` 也是一个同步上下文。

## 7.3 性能取舍

CCE 的取舍不是“免费加速”，而是：

```text
减少显存:
  避免 [N,V] logits 和部分 softmax 临时量

增加/暴露成本:
  TP: 每次 loss 多个 all_reduce
  ZeRO-3: backward 临时 gather lm_head/bias
  Patch: 依赖模型 forward 结构
  Save: FSDP/ZeRO/TP 聚合可能成为尾部瓶颈
```

对大 vocab、长序列、sample packing 的 SFT，CCE 的显存收益非常直接；对小 vocab、小 batch 或强通信瓶颈的多机 TP，TP loss all-reduce 和 ZeRO gather 的相对成本会更明显。

## 7.4 本章小结

> 💡 **小结**
>
> * CCE 真正节省的是 logits 显存，不是参数/optimizer state。
> * TP 下它用多次 all-reduce 换掉 logits all-gather 或 logits materialization。
> * ZeRO-3 下 backward gather `lm_head` 可能成为额外峰值与通信点。
> * 保存阶段的 full state dict / consolidate 与 CCE 无关，但会影响组合并行训练的整体体验。

# 八、配置项、边界条件与坑点

## 8.1 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `plugins: [CutCrossEntropyPlugin]` | `cli/config.py:prepare_plugins` -> `PluginManager.register` | 注册 CCE plugin，并把 CCE args 合入 schema | 没有 plugin 时 `cut_cross_entropy` 不会自动 patch |
| `cut_cross_entropy: true` | `CutCrossEntropyPlugin.pre_model_load` | 模型加载前执行 CCE patch | plugin args 默认 true；显式写更清晰 |
| `bf16` / `fp16` | `cut_cross_entropy/args.py:35-44` | 满足 CCE backward dtype 要求 | 都不设会报错；`bf16:auto` 是否为真取决于配置解析后值 |
| `chunked_cross_entropy` | `cut_cross_entropy/args.py:46-54` | 与 CCE 互斥 | 不能叠加两种 loss 显存优化 |
| `tensor_parallel_size > 1` | `utils/trainer.py:621-640`; `loaders/model.py:749-755` | 写 TP env，并传 `tp_size/tp_plan/device_mesh` 给模型 | tied embeddings 会被 `loaders/utils.py:139-148` 拒绝；部分 e2e TP 测试仍 skip |
| `deepspeed + tensor_parallel_size` | `validation.py:1121-1148` | 临时改写 DS JSON，加入 AutoTP 配置和保存 gather | 这是下游 DeepSpeed 消费，不等同于 CCE 直接支持 DS AutoTP |
| `fsdp_version: 2` + `fsdp_config` | `setup_fsdp_envs`; `patch_accelerate_fsdp2` | 启用 FSDP2 env 与 Axolotl FSDP2 patch | `adamw_8bit/adamw_bnb_8bit` 被拒绝，见 `validation.py:1102-1117` |
| `fsdp_config.cpu_ram_efficient_loading` | `loaders/model.py:756-780`; `fsdp2.py:371-425` | rank0 CPU / others meta 加载，再 broadcast/distribute | 量化模型有特殊限制；GPT-OSS Mxfp4 禁用该组合（`validation.py:1430-1436`） |
| `fsdp_config.state_dict_type` / `final_state_dict_type` | `train.py:294-333` | 选择 FULL 或 SHARDED 保存路径 | SHARDED 需要 merge；FULL 可能 rank0 CPU 峰值大 |
| `deepspeed: zero3*.json` | `setup_deepspeed_env`; `loaders/model.py:679-716` | 设置 ZeRO-3 env，创建 HfTrainerDeepSpeedConfig | ZeRO-3 + CCE backward 会 `GatheredParameters` lm_head |
| `context_parallel_size > 1` | `validation.py:1507-1579`; `train.py:205-220` | 进入 SequenceParallelContextManager，按序列切分 | 与本文 loss TP group 不同；CP 影响 `N` 的本地长度 |
| `experimental_skip_move_to_device` | `loaders/model.py:859-860` | 覆盖 skip move 行为 | 示例中用于避免大模型加载 OOM，但会改变加载峰值与调试直觉 |

## 8.2 静默失效与不兼容组合

* **模型类 patch 不匹配**：日志显示 CCE applied，但如果实际模型类不是被 patch 的类，loss 仍可能走原始 logits。`docs/agents/new_model_support.md:147-148` 专门提醒这个坑。
* **无 labels 路径**：generate / inference 不走 CCE；这是设计，不是 bug。
* **Gemma4 特例风险**：`AxolotlTrainer.compute_loss()` 对 `_model_type == "gemma4"` 会 pop labels，先 `outputs = model(**inputs)` 再调用 `unwrapped.loss_function(...)`（`core/trainers/base.py:438-453`）。这条路径可能绕开 forward 内 labels 触发的 CCE 分支；本文未在源码中确认另有 hook 把 Gemma4 标准 SFT 重新导回 CCE，需要实际测试验证。
* **TP tied embeddings**：`loaders/utils.py:139-148` 明确拒绝 `tie_word_embeddings=True` 的模型开启 TP。
* **TP + bnb 8bit optimizer**：`validation.py:1600-1608` 拒绝 `paged_adamw_8bit/adamw_8bit/adamw_bnb_8bit`。
* **FSDP 与 DeepSpeed 同时开启**：`validation.py:1187-1192` 拒绝 `deepspeed` 和 `fsdp` 同时为真。

## 8.3 本章小结

> 💡 **小结**
>
> * CCE 的最小有效配置是 plugin + fp16/bf16；`cut_cross_entropy` 字段本身不是自动魔法。
> * TP、FSDP2、ZeRO-3 分别通过不同源码路径改变模型加载与 loss 通信。
> * schema 描述、文档、源码之间存在历史痕迹，关键判断应以当前源码调用链为准。
> * 最容易踩坑的是“patch 了错误模型类”和“以为 TP/ZeRO/FSDP 的通信是同一件事”。

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/e2e/integrations/test_cut_cross_entropy.py:53-66` | LLaMA/SmolLM2 + CCE 训练并保存 | 覆盖单机 CCE plugin 主路径 |
| `tests/e2e/integrations/test_cut_cross_entropy.py:68-110` | Qwen2.5 + CCE | 覆盖另一个模型类型 patch |
| `tests/e2e/integrations/test_cut_cross_entropy.py:112-138` | CCE + flash/sdp attention | 证明 CCE 与部分 attention 配置可共存 |
| `tests/test_tensor_parallel_batch_size.py:28-55` | TP 下 batch size 按有效 DP world size 折算 | 是配置/归一化单元测试，不跑 TP forward |
| `tests/e2e/multigpu/test_fsdp2.py:52-90` | FSDP2 SFT，含 cpu_ram_efficient_loading 参数化 | 覆盖 FSDP2 训练保存基本路径，但未启用 CCE |
| `tests/e2e/multigpu/test_llama.py:629-710` | DeepSpeed ZeRO-3 packed 训练 | 覆盖 ZeRO-3 + packing，但未启用 CCE |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:3-19` | CCE + FSDP2 + TP + CP 示例 | 是推荐配置样例，不是测试 |
| `ml-cross-entropy@fec1a88/tests/test_vocab_parallel.py:26-104` | vocab-parallel loss 与 full loss 梯度对齐 | 外部 CCE fork 测试；注意参数化未包含默认 `cce` impl |

## 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| Axolotl e2e 同时开启 CCE + FSDP2 + TP | 未看到 | DTensor vocab branch 在 Axolotl 集成中可能退化或失效 |
| Axolotl e2e 同时开启 CCE + ZeRO-3 | 未看到 | backward `GatheredParameters` 的性能/正确性缺少框架级保护 |
| CCE vocab-parallel 默认 Triton `cce` impl | 外部测试参数化只列 `torch_compile` / `cce_exact` | 默认 kernel 的 TP 通信路径缺少显式测试覆盖 |
| 多机 TP / HSDP + TP | 未看到 | TP all-reduce 跨慢网络可能成为严重瓶颈 |
| CCE patch restore | 未看到 | 同进程多测试/多模型可能被全局 patch 污染 |
| Gemma4 Trainer 特例与 CCE | 未在本文源码中确认 | 可能出现日志 applied 但 loss 走原始 logits 的情况 |
| 保存/resume + CCE + TP/FSDP2 | 未看到组合测试 | 保存聚合、恢复后 patch 顺序可能出错 |
| 性能/显存断言 | 未看到 | 只能证明训练能跑，不能证明显存收益或通信成本达标 |

`tests/e2e/multigpu/test_tp.py` 中 TP e2e 当前被 skip，原因写明 “TP doesn't work with models with tied weights (embeddings)”（`test_tp.py:17-19`）。这也解释了为什么 TP 相关覆盖更多停留在 batch size 和示例层面。

## 9.3 本章小结

> 💡 **小结**
>
> * Axolotl 对单机 CCE 有 e2e 覆盖，对 FSDP2、ZeRO-3 也分别有 e2e 覆盖。
> * 最关键的组合：CCE + TP + FSDP2 / ZeRO-3，在 Axolotl 测试中没有形成闭环。
> * 外部 CCE fork 有 vocab-parallel 测试，但默认 Triton `cce` impl 的参数化覆盖不足。
> * 当前测试更像 smoke / correctness 起点，不是性能或显存收益证明。

# 十、局限性与已知优化点

## 10.1 硬约束

* CCE plugin 要求 PyTorch >= 2.4（`src/axolotl/integrations/cut_cross_entropy/__init__.py:54-59`）。
* CCE args 要求 `bf16` 或 `fp16`（`args.py:35-44`）。
* CCE kernel backward 要求 embeddings 和 classifier dtype 为 fp16/bf16（`ml-cross-entropy@fec1a88/cut_cross_entropy/cce_backward.py:349-356`）。
* `cce_linear_cross_entropy()` 要求 Ampere 或更新 GPU，否则报错建议使用 torch_compile 路径（`cce.py:257-261`）。
* TP 不支持 tied word embeddings（`src/axolotl/loaders/utils.py:139-148`）。
* FSDP 与 DeepSpeed 不能同时开启（`validation.py:1187-1192`）。
* `chunked_cross_entropy` 与 CCE 互斥（`cut_cross_entropy/args.py:46-54`）。

## 10.2 维护成本

* 模型 forward patch 依赖 Transformers 上游源码结构；CCE fork 里的 `llama.py` 注释标明它适配 Transformers v4.56.2。
* Axolotl generic fallback 假设模型有 `{Prefix}ForCausalLM` 且结构类似 LLaMA；多模态模型通常需要专属 patch。
* `_PATCH_OPTS` 是 module-level 全局状态，不是 per-model state。
* FSDP2 patch 会替换 `accelerate.Accelerator.get_state_dict`（`src/axolotl/monkeypatch/accelerate/fsdp2.py:529-538`），这对保存路径有效，但也增加了上游升级时的兼容风险。
* ZeRO-3 CCE backward 依赖 DeepSpeed 参数对象存在 `ds_id`，以及 `GatheredParameters` 行为稳定。

## 10.3 性能瓶颈

* TP CCE forward 至少有三次 `[N]` all-reduce；backward 还有一次 `[N,H]` all-reduce。
* ZeRO-3 backward 聚合 `lm_head.weight`，对于大 vocab 模型可能成为显著通信和显存峰值。
* FSDP2 full state dict 保存逐参数 `full_tensor()` 并在每个参数后 barrier（`fsdp2.py:168-173`），保存大模型时容易成为尾部瓶颈。
* DeepSpeed consolidate state dict 对 ZeRO-3 / TP 保存也可能在 rank0 或 CPU 侧形成瓶颈（`fsdp2.py:129-157`）。
* 这些通信在源码中没有显式 async overlap 控制。

## 10.4 已知优化点

基于源码行为，可以看到几个优化方向：

* **TP loss 通信融合**：LSE max/sum 和 correct logit all-reduce 当前分散在多个同步调用中，理论上可探索更细粒度融合或异步 overlap。
* **ZeRO-3 lm_head gather 分块**：`GatheredParameters` 聚合完整 `lm_head`，大 vocab 下可探索分块 backward 或与 ZeRO 参数预取协同。
* **组合 e2e 测试**：补 CCE + FSDP2 + TP、CCE + ZeRO-3、保存/resume 组合测试，比继续增加单卡模型覆盖更关键。
* **patch 生命周期管理**：增加可恢复 patch 或测试隔离，降低同进程污染风险。
* **文档/schema 对齐**：`tensor_parallel_size` schema 描述仍偏 DeepSpeed AutoTP，但源码已有 Transformers TP / DeviceMesh 路径，值得更新。

## 10.5 本章小结

> 💡 **小结**
>
> * CCE 的硬约束集中在 dtype、GPU 架构、模型 forward 结构和并行参数形态。
> * 组合并行下的瓶颈不再是 logits 显存，而是 TP all-reduce、ZeRO gather、保存聚合。
> * 维护风险主要来自 monkey patch 与上游 Transformers/Accelerate/DeepSpeed 版本耦合。
> * 最值得补的是组合并行 e2e 与性能/显存断言。

# 小结与展望

Axolotl 的 CCE 在 FSDP2 / TP / ZeRO-3 下，可以用几个关键词概括。

## 关键词一：前向替换

CCE 的接入点不是 Trainer loss，而是模型 `forward()`。这让它能在 logits 出现之前接管 loss，直接避免 `[B,T,V]` 的显存峰值。代价是 patch 必须紧贴模型类结构，上游模型 forward 一变，就需要跟着维护。

## 关键词二：vocab-parallel all-reduce

TP 下，CCE 没有 gather logits，而是在 loss 层重建全局 softmax 语义：LSE 两次 all-reduce、correct logit 一次 all-reduce、backward 的 dE 一次 all-reduce。它把大矩阵显存压力转成了较小张量的通信压力。

## 关键词三：ZeRO-3 参数生命周期补偿

ZeRO-3 下的问题不是 vocab range，而是 backward 时 `lm_head.weight` 可能已经回到分片状态。CCE fork 通过 `GatheredParameters` 在 backward kernel 前重新拿到完整参数。这保证正确性，但也把 `lm_head` 聚合成本暴露在 loss 层。

## 关键词四：DeviceMesh 分维协作

FSDP2 使用 DP/FSDP mesh 维度，TP 使用 `tp` mesh 维度，CP 又可能切序列维。CCE 只读取 TP group；FSDP2 参数通信和保存聚合由 Accelerate/PyTorch FSDP2 patch 负责。理解这个边界，才能避免把所有通信都误归因给 CCE。

## 关键词五：通信换显存

单卡 CCE 的收益非常直接；组合并行下，它仍然节省 logits 显存，但需要付出 TP all-reduce、ZeRO-3 gather、保存 consolidate 的系统成本。它适合大 vocab、长序列、logits 显存成为瓶颈的 SFT/预训练场景；不适合通信网络弱、TP 跨节点、或依赖 logits 输出做复杂训练后处理的路径。

后续最值得继续走读的方向有三个：

1. **Transformers TP 的 `_tp_plan` 如何具体 sharding `lm_head`**，这决定 CCE DTensor 分支的实际覆盖范围。
2. **FSDP2 + TP + CP 三维 mesh 下每层通信顺序**，尤其是 attention ring 与 loss TP all-reduce 的调度关系。
3. **保存/resume 的组合语义**，因为训练 step 能跑通不代表 full/sharded checkpoint 在大规模组合并行下没有 rank0 或 CPU 峰值问题。

最终评价是：Axolotl 的 CCE 集成并不是一个“漂亮但孤立的 kernel 插件”，而是一个典型的深度学习系统工程折中。它把最显眼的 logits 显存峰值切掉，同时把原本藏在 logits 后面的全局 softmax 语义显式搬到了 TP group 和 ZeRO-3 参数生命周期里。单卡时它很清爽；一旦进入 FSDP2 / TP / ZeRO-3，它就变成了 kernel、DeviceMesh、参数分片和保存策略共同参与的一条系统链路。
