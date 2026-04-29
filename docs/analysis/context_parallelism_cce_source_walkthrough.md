# Axolotl 源码走读：Context Parallelism + CCE 实现解析

在长序列训练里，很多显存问题并不来自同一个地方。FSDP/ZeRO 可以切参数、梯度和 optimizer state，但序列越长，Transformer 激活、attention 中间状态、最后一层 logits 仍然会一起膨胀。对 Axolotl 来说，`Context Parallelism`（源码和文档中也常称 Sequence Parallelism，下文简称 CP）和 `Cut Cross Entropy`（下文简称 CCE）正好打在两段不同的瓶颈上：CP 把序列维切到多个 rank 上，CCE 则避免 loss 阶段物化 `[batch, seq, vocab]` logits。

本文不展开 Ring Attention 或 CCE 论文的数学推导，而是顺着 Axolotl 的源码主路径看：用户在 YAML 里同时打开 `context_parallel_size` 和 `CutCrossEntropyPlugin` 后，框架到底在哪些地方改变了训练路径？哪些通信属于 CP，哪些属于 CCE？二者组合后，显存收益会不会被某个 gather 悄悄吃掉？

# 前言

## 业务 / 工程背景

目标场景是 **长序列 SFT / RL 训练**，尤其是 ALST、500K context、FSDP2、sample packing、大词表模型这类配置。示例里已经能看到这套组合拳：

```yaml
# examples/alst/llama3-8b-fsdp2-alst.yaml
sequence_len: 500_000
sample_packing: true
context_parallel_size: 8
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
flash_attention: true
bf16: auto
```

以及 N-D parallel 示例：

```yaml
# examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
dp_shard_size: 2
context_parallel_size: 2
tensor_parallel_size: 2
flash_attention: true
bf16: true
```

## 核心矛盾

这套特性背后的工程冲突可以概括为三句话：

1. **FSDP 切参数，不天然切序列**：长上下文的激活、attention 和 logits 仍然随 `seq_len` 增长。
2. **CP 切序列，但 attention 需要全局上下文**：每个 rank 只拿局部 token 后，必须靠 ring attention 在 CP group 内交换 K/V。
3. **CCE 省 logits，但必须在 `lm_head` 之前接管 forward**：如果标准 forward 已经生成了 `[B, S, V]` logits，再换 loss function 已经晚了。

组合起来看，CP 和 CCE 不是一个统一后端，而是两个不同边界的 patch：CP 作用在 **输入序列和 attention 通信**，CCE 作用在 **模型 forward 内部的 loss 计算**。

## 本文主线

本文按机制而不是按文件展开：

1. 配置如何从 YAML 变成 CP 拓扑和 CCE plugin；
2. DeviceMesh / rank / batch 调度如何保证 CP group 看到同一份样本；
3. CP 和 CCE 的 monkey patch 分别注入在哪里；
4. 一次真实 forward 中 shape 如何从 `[B,S]` 变成本地 chunk，再进入 CCE loss；
5. 显存、通信、保存、测试和边界条件如何评价。

## 不展开的内容

本文不讲：

- Ring Attention 算法推导；
- FSDP / ZeRO 原理；
- CCE 论文数学细节；
- Triton kernel 逐行实现。

只讲 Axolotl 如何把这些能力接入训练链路。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/main.py` / `cli/train.py` | 用户 `axolotl train` 到 `axolotl.train.train()` 的入口 |
| `src/axolotl/cli/config.py` / `utils/config/__init__.py` | 读取 YAML、注册插件、校验配置、归一 batch size |
| `src/axolotl/utils/schemas/config.py` / `validation.py` | CP schema、默认值、互斥与依赖校验 |
| `src/axolotl/utils/trainer.py` | 写入 Accelerate parallelism env，触发 CP 相关 patch |
| `src/axolotl/utils/distributed.py` | 构造 `ParallelismConfig` / `DeviceMesh` |
| `src/axolotl/loaders/model.py` / `loaders/patch_manager.py` | 模型加载前后 patch 顺序、CCE/CP 兼容 patch |
| `src/axolotl/utils/ctx_managers/sequence_parallel.py` | CP forward hook、sequence slicing、output gather |
| `src/axolotl/monkeypatch/ring_attn/patch.py` | 从 DeviceMesh 取 CP group，替换 HF FlashAttention |
| `src/axolotl/integrations/cut_cross_entropy/*` | CCE plugin、schema、pre-model-load patch |
| 外部 [`ml-cross-entropy@fec1a88`](https://github.com/axolotl-ai-cloud/ml-cross-entropy/tree/fec1a888e6f4ad7e6270ea7b02186e56c76f5ac2) | `cce_forward` / `linear_cross_entropy` 的实际 loss kernel |

# 一、配置入口：从一个 YAML 开关到两套执行边界

## 1.1 设计哲学与核心问题

CP 和 CCE 都不是“普通布尔开关”。

CP 一旦开启，就会影响：

- world size 如何拆成 CP / DP / TP 维度；
- Accelerate 是否构造 `ParallelismConfig`；
- dataloader 是否按 mesh 语义分发；
- forward 前是否切 sequence；
- attention 是否被替换成 ring attention；
- 保存时是否需要修复 tensor storage。

CCE 一旦开启，则会影响：

- plugin schema 是否合并；
- PyTorch / 外部包 / fork 标志是否满足；
- 模型类的 `forward()` 是否在加载前被替换；
- loss 阶段是否物化 logits。

也就是说，CP 是 **分布式拓扑 + forward hook + attention patch**，CCE 是 **plugin + model forward patch + loss kernel**。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：Click 命令入口，最终 launch `axolotl.cli.train`

src/axolotl/cli/train.py
  - do_cli：调用 load_cfg()
  - do_train：加载数据集后进入 axolotl.train.train()

src/axolotl/cli/config.py
  - load_cfg：读 YAML、prepare_plugins、validate_config、normalize_config
  - prepare_plugins：注册 CCE plugin

src/axolotl/integrations/config.py
  - merge_input_args：把插件参数动态合并进 Pydantic config

src/axolotl/utils/schemas/validation.py
  - check_context_parallel_size
  - validate_ring_attn_func
  - check_cross_entropy_conflicts

src/axolotl/integrations/cut_cross_entropy/args.py
  - CutCrossEntropyArgs：CCE 插件自己的配置 schema
```

## 1.3 主流程拆解

用户入口是：

```text
axolotl train config.yml
  -> cli/main.py:train()
    -> launch_training(...)
      -> python -m axolotl.cli.train config.yml
        -> cli/train.py:do_cli()
          -> load_cfg(config)
```

`load_cfg()` 的关键顺序在 `src/axolotl/cli/config.py:230-330`：

```text
load_cfg()
  -> 读取 YAML 为 DictDefault
  -> prepare_plugins(cfg)          # 先注册插件
  -> validate_config(cfg)          # 再做 Pydantic 校验
  -> normalize_config(cfg)
  -> prepare_optim_env(cfg)
```

这里顺序很关键。CCE 的字段 `cut_cross_entropy` 并不是基础 `AxolotlInputConfig` 里的字段，而是插件提供的。`CutCrossEntropyPlugin.get_input_args()` 返回：

```python
# src/axolotl/integrations/cut_cross_entropy/__init__.py:47-48
return "axolotl.integrations.cut_cross_entropy.CutCrossEntropyArgs"
```

然后 `merge_input_args()` 动态创建继承插件参数的新 config 类（`src/axolotl/integrations/config.py:27-57`）。因此正确开启方式是：

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
cut_cross_entropy: true
bf16: true
```

严格说，`cut_cross_entropy: true` 不是独立入口；没有 plugin，它不会触发 `pre_model_load()` 的 patch。

CP 的 schema 则是基础字段。`context_parallel_size` 定义在 `src/axolotl/utils/schemas/config.py:975-980`，旧字段 `sequence_parallel_degree` 在 `:969-974` 标记为 deprecated。

校验集中在 `ValidationMixin.check_context_parallel_size()`：

```python
# src/axolotl/utils/schemas/validation.py:1508-1519
if self.sequence_parallel_degree and not self.context_parallel_size:
    self.context_parallel_size = self.sequence_parallel_degree

if not self.context_parallel_size:
    self.context_parallel_size = 1
elif self.context_parallel_size > 1:
    if not self.flash_attention:
        raise ValueError("flash_attention: true must be set ...")
```

后面还有两个关键约束：

```python
# validation.py:1522-1526
if self.sample_packing and self.micro_batch_size > 1:
    raise ValueError("micro_batch_size must be set to 1 ...")
```

以及对 `ring_flash_attn` 的 import 检查（`validation.py:1528-1550`）。如果没装，直接报错，不会静默 fallback。

CCE 的校验在插件 schema 里：

```python
# src/axolotl/integrations/cut_cross_entropy/args.py:35-42
if data.get("cut_cross_entropy") and not (data.get("bf16") or data.get("fp16")):
    raise ValueError("Cut Cross Entropy requires fp16/bf16 training ...")
```

同时不能和其他 CE 优化叠加：

```python
# src/axolotl/utils/schemas/validation.py:974-1002
ce_options = {
    "cut_cross_entropy": data.get("cut_cross_entropy"),
    "chunked_cross_entropy": data.get("chunked_cross_entropy"),
    "liger_cross_entropy": data.get("liger_cross_entropy"),
    "liger_fused_linear_cross_entropy": data.get("liger_fused_linear_cross_entropy"),
}
if len(enabled_options) > 1:
    raise ValueError("Only one cross entropy optimization can be enabled ...")
```

## 1.4 关键细节与误区澄清

> 容易误解点一：`context_parallel_size` 和 `sequence_parallel_degree` 是两个开关。

不是。`sequence_parallel_degree` 只是旧字段，校验时会被迁移到 `context_parallel_size`（`validation.py:1508-1514`）。

> 容易误解点二：写了 `cut_cross_entropy: true` 就一定启用 CCE。

不一定。CCE 是 plugin。真正调用 `cce_patch()` 的地方是 `CutCrossEntropyPlugin.pre_model_load()`（`src/axolotl/integrations/cut_cross_entropy/__init__.py:86-103`）。如果没有注册 plugin，就没有这个 hook。

> 容易误解点三：CP 会自动安装 ring-flash-attn。

不会。源码只是 import 检查，失败就报错（`validation.py:1528-1550`）。依赖在 `pyproject.toml:91-96` 的 optional extra `ring-flash-attn` 里。

## 1.5 本章小结

> 💡 **小结**
>
> * CP 是基础 schema 字段；CCE 是插件字段。
> * CP 校验会强制依赖 `flash_attention` 和 `ring_flash_attn`。
> * CCE 必须通过 plugin 注册，且要求 fp16/bf16。
> * CCE 与 chunked CE / Liger CE 是互斥 loss 优化。

# 二、DeviceMesh 与同 batch 调度：CP 为什么先改变 rank 语义

## 2.1 设计哲学与核心问题

CP 和 DP 最大的区别是：**DP 希望不同 rank 处理不同 batch，CP 希望同一个 CP group 内的 rank 处理同一份 batch 的不同 sequence chunk**。

如果 rank0 拿样本 A 的前半段，rank1 拿样本 B 的后半段，ring attention 再怎么通信都是错的。因此 CP 的第一件事不是切 tensor，而是让分布式拓扑知道：哪些 rank 属于同一个 sequence/context group。

## 2.2 源码入口与关键对象

```text
src/axolotl/utils/trainer.py
  - setup_parallelism_envs：写 PARALLELISM_CONFIG_CP_SIZE 等 env

src/axolotl/utils/distributed.py
  - build_parallelism_config：构造 ParallelismConfig 和 DeviceMesh
  - _get_parallel_config_kwargs：把 TP/CP/DP/FSDP size 归一化

src/axolotl/loaders/model.py
  - _set_parallel_config：模型加载阶段也会构造 parallelism_config/device_mesh

src/axolotl/core/trainers/base.py
  - create_accelerator_and_postprocess：重置 AcceleratorState，让 env 生效
```

## 2.3 主流程拆解

`prepare_optim_env()` 会调用 `setup_parallelism_envs()`（`src/axolotl/utils/trainer.py:643-667`）。CP 相关逻辑是：

```python
# src/axolotl/utils/trainer.py:621-640
if cfg.context_parallel_size and cfg.context_parallel_size > 1:
    os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cfg.context_parallel_size)
    os.environ["ACCELERATE_ALLOW_CP_STANDALONE"] = "true"
    patch_prepare_cp()

if set_accelerate_parallelism_config:
    os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
```

然后 `build_parallelism_config()` 根据 world size 和用户配置生成拓扑：

```python
# src/axolotl/utils/distributed.py:299-315
pc_kwargs = _get_parallel_config_kwargs(
    get_world_size(),
    cfg.tensor_parallel_size,
    cfg.context_parallel_size,
    cfg.dp_shard_size,
    cfg.dp_replicate_size,
    bool(cfg.fsdp or cfg.fsdp_config),
)

parallelism_config = ParallelismConfig(**pc_kwargs)
device_mesh = parallelism_config.build_device_mesh("cuda")
```

`_get_parallel_config_kwargs()` 中，CP 会先消耗一部分 world size：

```python
# distributed.py:334-336
if context_parallel_size and context_parallel_size > 1:
    pc_kwargs["cp_size"] = context_parallel_size
    remaining_world_size = remaining_world_size // context_parallel_size
```

这意味着：

```text
world_size = 8
context_parallel_size = 4
tensor_parallel_size = 1

effective DP size = 8 / 4 = 2
```

Axolotl 的 batch size 归一化也遵循这个语义：

```python
# src/axolotl/utils/config/__init__.py:134-142
effective_world_size = (
    cfg.world_size
    // (cfg.context_parallel_size or 1)
    // (cfg.tensor_parallel_size or 1)
)
cfg.batch_size = cfg.batch_size * effective_world_size
```

测试 `tests/test_context_parallel_batch_size.py:29-56` 直接覆盖了这个行为：`world_size=4, context_parallel_size=2` 时，global batch 从 32 变成 16。

## 2.4 关键细节与误区澄清

> 容易误解点四：CP 的 rank group 是 Axolotl 手写出来的。

不是。Axolotl 调用 Accelerate 的 `ParallelismConfig.build_device_mesh("cuda")` 生成 `DeviceMesh`（`distributed.py:309-315`），后面 ring attention 从 mesh 的 `"cp"` 维度取 group。

> 容易误解点五：文档说 data collator 负责 chunking，所以切 sequence 发生在 collator。

当前源码主路径不是这样。`docs/sequence_parallelism.qmd:40-45` 写了 “data collator handles the chunking”，但 SFT 主路径里真正切 tensor 的是 `SequenceParallelContextManager` 的 forward pre-hook（`sequence_parallel.py:255-288`）。文档与源码不一致时，应以源码为准。

> 容易误解点六：CP 只影响 forward，不影响 batch size。

不对。`normalize_config()` 会用 `world_size // context_parallel_size // tensor_parallel_size` 调整全局 batch（`utils/config/__init__.py:134-142`），测试也覆盖了这一点。

## 2.5 本章小结

> 💡 **小结**
>
> * CP group 内 rank 必须拿同一份 batch 的不同 sequence chunk。
> * Axolotl 通过 Accelerate `ParallelismConfig` / `DeviceMesh` 表达 CP 维度。
> * `context_parallel_size` 会改变 effective DP size，从而影响 global batch。
> * SFT 的 sequence chunking 不在 collator，而在 forward pre-hook。

# 三、Patch 注入：CP 和 CCE 都必须在 forward 前抢位置

## 3.1 设计哲学与核心问题

CP 和 CCE 都是“低侵入接入”：

- CP 不重写模型类，而是用 hook 改输入，用 monkey patch 替换 FlashAttention；
- CCE 不重写 Trainer，而是在模型加载前替换 `ForCausalLM.forward()`。

这种方式接入快、侵入小，但代价是 patch 的生命周期、版本兼容和污染范围都要非常小心。

## 3.2 源码入口与关键对象

```text
src/axolotl/loaders/model.py
  - ModelLoader.load：模型加载主流程，决定 plugin 与 patch 顺序

src/axolotl/loaders/patch_manager.py
  - apply_pre_model_load_patches
  - _apply_transformers_patches
  - _apply_fsdp_patches

src/axolotl/monkeypatch/ring_attn/patch.py
  - register_ring_attn_from_device_mesh
  - set_ring_attn_group
  - update_ring_attn_params

src/axolotl/integrations/cut_cross_entropy/__init__.py
  - CutCrossEntropyPlugin.pre_model_load
  - patch_llama_like
```

## 3.3 主流程拆解

模型加载顺序在 `src/axolotl/loaders/model.py:161-194`：

```text
ModelLoader.load()
  -> patch_manager.apply_pre_model_load_patches()
  -> _apply_pre_model_load_setup()
  -> PLUGIN_MANAGER.pre_model_load(cfg)
  -> patch_manager.apply_post_plugin_pre_model_load_patches()
  -> _build_model()
  -> post model build/load/lora hooks
```

CCE 就挂在 `PLUGIN_MANAGER.pre_model_load(cfg)` 这一步：

```python
# src/axolotl/integrations/cut_cross_entropy/__init__.py:86-103
if cfg.cut_cross_entropy:
    self._check_requirements()
    self.patch_llama_like(cfg.model_config_type)

    from cut_cross_entropy.transformers.patch import cce_patch
    cce_patch(
        cfg.model_config_type,
        remote_model_id=cfg.base_model if cfg.trust_remote_code else None,
    )
```

它必须在 `_build_model()` 之前执行，因为它替换的是模型类的 `forward()`。如果模型实例已经创建，再 patch 类方法未必覆盖所有路径。

CP 的 patch 分两层。

第一层是初始化期兼容 patch：

```python
# src/axolotl/loaders/patch_manager.py:135-149
if self.cfg.context_parallel_size > 1:
    patch_prepare_context_parallel_inputs()
```

以及 Accelerate parallelism patch：

```python
# patch_manager.py:279-286
if self.cfg.context_parallel_size > 1 or (... fsdp2 ...):
    patch_parallelism_config()
```

第二层是真正 ring attention 注册，发生在训练执行期：

```python
# src/axolotl/train.py:205-219
if cfg.context_parallel_size > 1:
    stack.enter_context(
        SequenceParallelContextManager(
            models=models,
            context_parallel_size=cfg.context_parallel_size,
            ring_attn_func=cfg.ring_attn_func,
            device_mesh=trainer.accelerator.torch_device_mesh,
            ...
        )
    )
```

`SequenceParallelContextManager.__init__()` 会调用 `_register_ring_attn()`：

```python
# src/axolotl/utils/ctx_managers/sequence_parallel.py:207-253
register_ring_attn_from_device_mesh(
    device_mesh=self.device_mesh,
    context_parallel_dim=("cp",),
    heads_k_stride=self.heads_k_stride,
    ring_attn_func=self.ring_attn_func,
)
```

`register_ring_attn_from_device_mesh()` 从 mesh 中取 CP group：

```python
# src/axolotl/monkeypatch/ring_attn/patch.py:159-184
sequence_mesh = device_mesh[context_parallel_dim]
sequence_pg = sequence_mesh.get_group()
set_ring_attn_group(sequence_pg)
```

然后按 `ring_attn_func` 选择 patch 路径：

```python
# ring_attn/patch.py:186-212
if ring_attn_func is RingAttnFunc.VARLEN_LLAMA3:
    ring_flash_attn.adapters.hf_adapter.substitute_hf_flash_attn(...)
elif ring_attn_func is RingAttnFunc.BATCH_RING:
    substitute_hf_flash_attn(...)
```

## 3.4 关键细节与误区澄清

> 容易误解点七：`Trainer._prepare_context_parallel_inputs()` 是 Axolotl CP 的主切分路径。

不是。`patch_prepare_context_parallel_inputs()` 只是放宽 Transformers 内部 guard，让 FlashAttention + CP 不被 SDPA-only 限制挡住（`src/axolotl/monkeypatch/transformers/trainer_context_parallel.py:19-72`）。真正切输入的是 `SequenceParallelContextManager` 的 pre-hook。

> 容易误解点八：退出 `SequenceParallelContextManager` 后，所有 patch 都恢复。

不是。`__exit__()` 只移除 model forward hooks（`sequence_parallel.py:238-245`），并且源码 TODO 明确写着还没 un-patch attention 和 accelerate functions。`RING_ATTN_GROUP` 是模块级全局变量（`ring_attn/patch.py:34-47`），CCE 的类级 `forward` patch 也不是局部恢复型 patch。

> 容易误解点九：CCE 打印 “Applying Cut Cross Entropy” 就代表一定命中实际训练类。

不一定。Axolotl 的 generic fallback 会尝试导入 `transformers.models.{model_type}.modeling_{model_type}` 并替换 `{Prefix}ForCausalLM.forward`（`cut_cross_entropy/__init__.py:123-135`）。如果实际加载的是 `ForConditionalGeneration` 或 remote custom class，可能日志显示启用但显存没有下降；项目文档也提醒了这个坑（`docs/agents/new_model_support.md:136-148`）。

## 3.5 本章小结

> 💡 **小结**
>
> * CCE 必须在模型加载前 patch `forward()`。
> * CP 的 ring attention patch 发生在训练进入 context manager 时。
> * CP hook 是局部的，但 attention / accelerate / CCE forward patch 多数是模块级或类级全局状态。
> * patch 带来低侵入接入，也带来版本与测试隔离成本。

# 四、一次 SFT forward：CP 切 sequence，CCE 切 logits

## 4.1 设计哲学与核心问题

CP 和 CCE 组合时，最关键的问题是：**loss 到底在本地 chunk 上算，还是要先 all-gather 回完整 sequence？**

SFT 主路径的答案是：本地算。CP pre-hook 先把输入切成 `[B, S/CP]`，模型只在本地 chunk 上 forward；CCE patched forward 再在本地 hidden states 上计算 loss，不物化 `[B, S/CP, V]` logits。SFT 默认不会在 post-hook all-gather 输出，因此显存收益能保留下来。

## 4.2 源码入口与关键对象

```text
src/axolotl/train.py
  - execute_training：进入 SequenceParallelContextManager 后调用 trainer.train()

src/axolotl/utils/ctx_managers/sequence_parallel.py
  - sequence_parallel_pre_hook
  - apply_sequence_parallelism
  - sequence_parallel_post_hook
  - AllGatherWithGrad

外部 ml-cross-entropy@fec1a88
  - cut_cross_entropy/transformers/llama.py:cce_forward
  - cut_cross_entropy/transformers/utils.py:apply_lce
  - cut_cross_entropy/cce.py:LinearCrossEntropyFunction
```

## 4.3 主流程拆解

训练入口：

```text
execute_training()
  -> with SequenceParallelContextManager(...)
    -> trainer.train()
      -> model(**batch)
        -> forward_pre_hook: apply_sequence_parallelism
        -> patched model.forward: cce_forward
        -> ring_flash_attn 替换后的 attention
        -> CCE loss
```

CP pre-hook 在 `sequence_parallel.py:255-288`。它先把 positional args 转成 kwargs，再调用：

```python
# sequence_parallel.py:271-274
updated_kwargs, self.original_seq_len, self.pad_len = (
    self.apply_sequence_parallelism(updated_kwargs)
)
```

`apply_sequence_parallelism()` 的核心动作：

```python
# sequence_parallel.py:51-64
batch_size, original_seq_len = batch["input_ids"].shape
if batch.get("position_ids") is not None and batch_size == 1:
    update_ring_attn_params(position_ids=batch["position_ids"])
else:
    batch["position_ids"] = torch.arange(...).expand(...)
```

sample packing 时，`position_ids` 用来推导 packed sequence 的 `cu_seqlens`，再交给 ring attention：

```python
# ring_attn/patch.py:214-226
cu_seqlens, _ = get_cu_seqlens_from_pos_ids(position_ids)
update_ring_flash_attn_params(cu_seqlens, get_ring_attn_group())
```

然后对 sequence pad 到可整除长度，并沿 dim=1 切分：

```python
# sequence_parallel.py:96-144
if total_seq_len % divisor != 0:
    # pad input_ids / labels / attention_mask ...

for key in batch:
    if batch[key].size(1) == total_seq_len:
        batch[key] = batch[key].chunk(local_world_size, dim=1)[local_rank].contiguous()
```

shape 直觉如下：

```text
进入 hook 前:
  input_ids:      [B, S]
  labels:         [B, S]
  attention_mask: [B, S]

CP=4 后，rank i:
  input_ids_i:    [B, S/4]
  labels_i:       [B, S/4]
  position_ids_i: [B, S/4]

模型主体:
  hidden_i:       [B, S/4, H]
```

接着进入 CCE patched forward。以 Llama patch 为例，外部 fork 的 `cce_forward()` 先调用原模型主体：

```python
# ml-cross-entropy@fec1a88
# cut_cross_entropy/transformers/llama.py:53-64
outputs = self.model(...)
hidden_states = outputs.last_hidden_state
loss = None
logits = None
```

如果 labels 存在且允许 CCE：

```python
# llama.py:72-80
if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
    loss = apply_lce(
        hidden_states[:, slice_indices, :],
        self.lm_head.weight,
        labels,
        _PATCH_OPTS,
        **kwargs,
    )
else:
    logits = self.lm_head(hidden_states[:, slice_indices, :])
```

CCE 的本质是把：

```text
hidden_states: [B, S_local, H]
lm_head.weight: [V, H]
labels: [B, S_local]
```

送入 `linear_cross_entropy()`，而不是先生成：

```text
logits: [B, S_local, V]
```

外部 `cce_linear_cross_entropy()` 会检查 shape 并 flatten：

```python
# cut_cross_entropy/cce.py:255-272
assert e.size()[0:-1] == targets.size()
assert e.size(-1) == c.size(1)

e = e.flatten(0, -2)
targets = targets.flatten()
```

随后 forward 里只分配 `lse` 和正确类别 logit 相关的一维中间量：

```python
# cce.py:59-120
lse = cce_lse_forward_kernel(...)
neg_dot = indexed_neg_dot_forward_kernel(...)
nll = neg_dot.add_(lse)
```

`cce_lse_forward_kernel()` 分配的是：

```python
# cce_lse_forward.py:194-205
lse = e.new_full((B,), -float("inf"), dtype=torch.float32)
logit_avg = e.new_full((V,), 0.0, dtype=torch.float32)  # 可选
```

`indexed_neg_dot_forward_kernel()` 输出也是 `[B]` 级别（`indexed_dot.py:101-158`）。这就是 CCE 避开 logits 显存的源码证据。

## 4.4 关键细节与误区澄清

> 容易误解点十：CP + CCE 需要先 gather 完整 sequence 再算 loss。

SFT 主路径不需要。`gather_outputs` 在 `train.py:217` 只有 GRPO / EBFT 时为 true；普通 SFT 不注册 output gather hook。CCE 在本地 `[B, S/CP, H]` 上算 loss。

> 容易误解点十一：CCE 分支仍会返回 logits。

通常不会。外部 `cce_forward()` 初始化 `logits = None`，CCE 分支只设置 `loss`，最后返回 `CausalLMOutputWithPast(logits=logits)`（`llama.py:64-98`）。训练中依赖 `outputs.logits` 的 callback 或自定义 Trainer 可能不兼容。

> 容易误解点十二：`ring_attn_func` 决定 sequence slicing 策略。

不决定。`apply_sequence_parallelism()` 的 docstring 明确说 `ring_attn_func` 当前在 slicing 中 unused（`sequence_parallel.py:42-43`）。它真正影响的是 ring attention patch 后端（`ring_attn/patch.py:186-212`）。

## 4.5 本章小结

> 💡 **小结**
>
> * CP 先把 `[B,S]` 切成 `[B,S/CP]`。
> * ring attention 在每层 attention 中恢复全局上下文语义。
> * CCE 在本地 hidden states 上直接算 loss，避免 `[B,S/CP,V]` logits。
> * SFT 默认不 all-gather 输出，因此 CP + CCE 的显存收益可以叠加。

# 五、完整主路径串联

## 5.1 完整调用栈

```text
User: axolotl train config.yml
  │
  ├─ Step 1: CLI launch
  │     └─ src/axolotl/cli/main.py:train()
  │        -> accelerate/torchrun/python -m axolotl.cli.train
  │
  ├─ Step 2: 配置加载与插件注册
  │     └─ src/axolotl/cli/config.py:load_cfg()
  │        -> prepare_plugins()
  │        -> merge_input_args()
  │        -> validate_config()
  │
  ├─ Step 3: CP env 与 batch size 归一化
  │     └─ utils/config/__init__.py:normalize_config()
  │     └─ utils/trainer.py:setup_parallelism_envs()
  │
  ├─ Step 4: 模型加载前 patch
  │     └─ loaders/model.py:ModelLoader.load()
  │        -> PatchManager.apply_pre_model_load_patches()
  │        -> PluginManager.pre_model_load()
  │           -> CutCrossEntropyPlugin.pre_model_load()
  │
  ├─ Step 5: Trainer / Accelerator / DeviceMesh
  │     └─ core/trainers/base.py:create_accelerator_and_postprocess()
  │     └─ utils/distributed.py:build_parallelism_config()
  │
  ├─ Step 6: 训练上下文
  │     └─ train.py:execute_training()
  │        -> SequenceParallelContextManager(...)
  │        -> register_ring_attn_from_device_mesh()
  │
  ├─ Step 7: 每次 forward
  │     └─ sequence_parallel_pre_hook()
  │        -> apply_sequence_parallelism()
  │        -> cce_forward()
  │        -> linear_cross_entropy()
  │
  └─ Step 8: 保存
        └─ train.py:save_trained_model()
        └─ core/trainers/base.py:_save()
```

## 5.2 每一层做了什么

| 层级 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 频率 |
|---|---|---|---|---|---|
| 配置校验 | YAML | `context_parallel_size` 默认/校验，CCE schema 合并 | 无 | 无 | 初始化一次 |
| env 设置 | cfg | `PARALLELISM_CONFIG_CP_SIZE` 等 | 无 | 无 | 初始化一次 |
| DeviceMesh | world size + CP/TP/DP | `ParallelismConfig` / `DeviceMesh` | process group 初始化 | 无 | 初始化一次 |
| CCE patch | model type | 替换 `ForCausalLM.forward` | 无 | 避免后续 logits | 模型加载前一次 |
| CP context | trainer model | 注册 pre/post hooks，设置 ring group | 无 / group 绑定 | 无 | trainer.train 外层一次 |
| pre-hook | batch `[B,S]` | local batch `[B,S/CP]` | token count 可能 all_reduce | 降低模型主体输入规模 | 每 forward |
| attention | local Q/K/V | local attention output | ring attention group 通信 | 用通信换 attention 显存 | 每层 |
| CCE loss | local hidden + labels | loss，logits 通常 None | 无；TP/ZeRO3 例外 | 避免 local `[B,S/CP,V]` | 每 forward |
| output gather | local output | full output | all_gather | 只在 GRPO/EBFT 默认开启 | 部分路径 |
| save | state_dict | checkpoint / final model | FSDP/DS 可能通信 | CP 下 CPU clone 可能增 CPU 峰值 | 保存时 |

## 5.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在 SFT 主流程 | 正确理解 |
|---|---|---|---|
| `docs/sequence_parallelism.qmd` 中 “collator handles chunking” | 文档这样写 | ❌ | 当前源码由 forward pre-hook 切 sequence |
| `Trainer._prepare_context_parallel_inputs` patch | 名字像 CP 输入准备 | ⚠️ 兼容 patch | 不是 Axolotl 的 slicing 主路径 |
| `AllGatherWithGrad` | 容易以为 CP 总 gather 输出 | ❌ SFT 默认不走 | 只在 `gather_outputs=True`，即 GRPO/EBFT |
| `chunked_cross_entropy` patch | 同样是 loss 显存优化 | ❌ 与 CCE 互斥 | `validation.py` 禁止同时启用 |
| CCE `post_model_load` | 插件常见 hook | ❌ | CCE 只实现 `pre_model_load` 主 patch |
| GRPO `SequenceParallelRepeatRandomSampler` | 也是 CP 数据调度 | ❌ SFT 不走 | 仅 GRPO sequence-parallel trainer |
| `_save()` CPU clone | CP 相关 | ❌ 不在训练 step | 保存时修 safetensors storage，不省训练显存 |

## 5.4 本章小结

> 💡 **小结**
>
> * CP + CCE 的主路径横跨 CLI、config、Accelerate、ModelLoader、Trainer、hook 和外部 loss kernel。
> * 初始化期主要是声明拓扑和 patch；每 step 主要是 sequence slicing、ring attention 和 CCE loss。
> * SFT 的关键收益来自“不 gather 输出 + 不物化 logits”。
> * 很多看似相关的函数是兼容路径、RL 路径或保存路径，不是 SFT 主链路。

# 六、关键数据流 / 状态流 / shape 流程

## 6.1 Tensor shape 变化

以 SFT、`context_parallel_size=4`、CCE 开启为例：

```text
原始 batch:
  input_ids:      [B, S]
  labels:         [B, S]
  attention_mask: [B, S]

CP pre-hook 后，每个 CP rank:
  input_ids_i:    [B, S/4]
  labels_i:       [B, S/4]
  position_ids_i: [B, S/4]

Transformer local forward:
  hidden_i:       [B, S/4, H]

CCE:
  e = hidden_i.flatten(0, -2): [B*S/4, H]
  c = lm_head.weight:          [V, H]
  targets:                    [B*S/4]

CCE 中间量:
  lse:      [B*S/4]
  neg_dot:  [B*S/4]

不会产生:
  logits_i: [B, S/4, V]
```

这说明 CP 和 CCE 的收益是乘法关系：

- CP 把 `S` 变成 `S/CP`；
- CCE 再把 logits 的 `V` 维物化去掉。

标准路径 local logits 大小约为：

```text
B * (S / CP) * V * dtype_bytes
```

CCE 避开的正是这块。

## 6.2 Rank / Mesh / Process Group 变化

以概念示例说明：

```text
world_size = 8
context_parallel_size = 4
effective_dp_size = 2

CP group 0: rank0, rank1, rank2, rank3
CP group 1: rank4, rank5, rank6, rank7
```

真实 group 由 `DeviceMesh["cp"].get_group()` 决定，而不是 Axolotl 手写 rank list。源码依据：

```python
# ring_attn/patch.py:159-170
sequence_mesh = device_mesh[context_parallel_dim]
sequence_pg = sequence_mesh.get_group()
context_parallel_size = sequence_mesh.size()
```

每个 CP group 内：

```text
rank0: sequence chunk 0
rank1: sequence chunk 1
rank2: sequence chunk 2
rank3: sequence chunk 3
```

但它们对应的是同一份样本。不同 CP group 才对应不同数据并行 batch。

## 6.3 状态切换

CP 和 CCE 都使用了全局状态：

```text
进入 SequenceParallelContextManager:
  1. 从 DeviceMesh 取 CP process group
  2. 写入 ring_attn/patch.py 的 RING_ATTN_GROUP
  3. patch HF flash attention
  4. 给 model 注册 forward pre-hook/post-hook

执行 forward:
  pre-hook 读取 local_rank/local_world_size
  ring attention 通过 get_ring_attn_group() 读全局 group
  CCE patched forward 通过外部 _PATCH_OPTS 判断是否走 LCE

退出 context:
  移除 model hooks
  不恢复 attention / accelerate / CCE class patch
```

源码证据：

- `RING_ATTN_GROUP` 定义与 setter：`ring_attn/patch.py:34-47`
- hooks 移除：`sequence_parallel.py:238-245`
- CCE `_PATCH_OPTS`：外部 `cut_cross_entropy/transformers/llama.py:37-40, 101-128`

这类状态是 **进程级**，不是线程隔离的。单进程多模型、长生命周期服务或测试复用进程时，需要注意污染。

## 6.4 通信边界

CP 显式通信：

| 位置 | 通信 | group | 频率 |
|---|---|---|---|
| ring attention | ring KV 交换，具体 primitive 在外部 `ring_flash_attn` | CP group | 每层 attention |
| `apply_sequence_parallelism` | `all_reduce(..., AVG)` 修正 token count | CP group | forward 输入处理 |
| eval loss correction | 两次 `all_reduce(SUM)` | CP group | eval forward |
| output gather | shape `all_gather` + tensor `all_gather` | CP group | GRPO/EBFT output gather |

CCE 默认没有 CP 通信。但如果 `lm_head.weight` 是 DTensor，外部 CCE 会走 vocab-parallel 分支：

```python
# ml-cross-entropy transformers/utils.py:113-133
device_mesh = c.device_mesh
process_group = device_mesh.get_group("tp")
vocab_parallel_options = VocabParallelOptions.from_vocab(...)
```

随后在 CCE 内部：

```python
# vocab_parallel/utils.py:47-76
vp_reduce_lse       # MAX all_reduce + SUM all_reduce
vp_reduce_correct_logit
vp_reduce_e_grad
```

如果是 DeepSpeed ZeRO-3，CCE backward 还可能 gather full `lm_head.weight`：

```python
# cut_cross_entropy/cce.py:179-195
GatheredParameters(params.zero3_params, modifier_rank=None)
```

## 6.5 本章小结

> 💡 **小结**
>
> * shape 主线是 `[B,S] -> [B,S/CP] -> [B*S/CP,H]`。
> * CP 通信发生在 sequence / attention 维度；CCE 通信只在 TP vocab-parallel 或 ZeRO-3 兼容分支出现。
> * SFT 默认不会把输出 gather 回完整 sequence。
> * CP 和 CCE 都有全局 patch 状态，需要关注污染与恢复。

# 七、核心机制深挖

## 7.1 Monkey Patch：零侵入接入，还是维护风险？

CP patch 主要替换 Hugging Face FlashAttention 调用点。

`VARLEN_LLAMA3` 路径中，Axolotl 替换 `ring_flash_attn.adapters.hf_adapter.create_ring_flash_attention_forward`，再调用上游 `substitute_hf_flash_attn()`（`ring_attn/patch.py:186-202`）。

`BATCH_RING` 路径中，Axolotl 自己实现 adapter，替换：

```python
# src/axolotl/monkeypatch/ring_attn/adapters/batch.py:167-179
transformers.modeling_flash_attention_utils._flash_attention_forward = (
    new_flash_attention_forward
)
```

CCE patch 则替换模型 `forward()`。外部 fork 的 `PATCH_FNS` 把 model type 映射到 patch function（外部 `cut_cross_entropy/transformers/patch.py:15-88`），`cce_patch()` lazy import 后调用对应 patch（`:91-199`）。Llama 路径最终是：

```python
# external llama.py:127
modeling_llama.LlamaForCausalLM.forward = cce_forward
```

这解释了为什么 CCE 维护成本高：它绑定具体 Transformers 模型类和 forward 签名。外部 `llama.py:1` 还注明适配自 Transformers v4.56.2，而当前 Axolotl `pyproject.toml:20` 固定 `transformers==5.5.4`，这类版本漂移必须靠测试兜住。

## 7.2 通信原语：前向和反向是否对称？

`AllGatherWithGrad` 是一个很好的例子。它前向 all-gather：

```python
# sequence_parallel.py:393-414
dist.all_gather(all_shapes, local_shape, group=group)
dist.all_gather(gathered, input_tensor, group=group)
result = torch.cat(gathered, dim=1)
```

反向却不做通信，只切回本 rank 的 gradient slice：

```python
# sequence_parallel.py:437-443
offset = sum(seq_lens[:rank])
grad_slice = grad_output[:, offset : offset + seq_lens[rank]].contiguous()
```

这不是 bug，而是 concat 的反向语义：每个 rank 只需要属于自己 sequence chunk 的梯度。

ring attention 的 forward/backward 具体通信 primitive 在 `ring_flash_attn` 外部包里，Axolotl 源码只把 `group=process_group` 传进去：

- `ring_attn/patch.py:110-124`
- `ring_attn/adapters/batch.py:137-149`

因此不能在 Axolotl 仓库内断言它是 send/recv、all-to-all 还是其他自定义通信，只能确认通信发生在 CP group。

## 7.3 配置归一化：用户配置如何变成真实行为？

有三个关键归一化点：

1. `context_parallel_size=None` 会变成 1（`validation.py:1514-1515`）；
2. `ring_attn_func=None` 会按 `sample_packing` 选择默认值：packing 时 `VARLEN_LLAMA3`，否则 `BATCH_RING`（`validation.py:1568-1577`）；
3. batch size 会按 `world_size / CP / TP` 缩放（`utils/config/__init__.py:134-142`）。

CCE 的归一化则来自 plugin schema：

- `cut_cross_entropy` 默认是 `True`（`args.py:33`），但只有 plugin 注册后才进入 schema；
- CCE 要求 fp16/bf16（`args.py:35-42`）；
- CCE 禁止 chunked CE（`args.py:46-54`）；
- 全局禁止多种 CE 优化同开（`validation.py:974-1002`）。

## 7.4 本章小结

> 💡 **小结**
>
> * CP 和 CCE 都靠 monkey patch 接入，侵入小但维护成本高。
> * `AllGatherWithGrad` 前向通信、反向本地切片；ring attention 通信委托外部包。
> * CCE 必须 patch forward，而不是只替换 loss function。
> * 用户配置经过 schema、env、DeviceMesh、plugin hook 多层转换后才变成真实行为。

# 八、显存、性能与通信分析

## 8.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | CP/CCE 不切参数；需 FSDP/ZeRO/TP 解决 |
| optimizer state | ❌ | 不属于 CP/CCE 范围 |
| Transformer 激活 | ✅ CP 节省 | 每 rank 只处理 `S/CP` token |
| attention 中间状态 | ✅ 但换通信 | ring attention 避免单卡完整长序列 attention |
| logits `[B,S,V]` | ✅ CCE 节省 | CCE 不物化 `[B,S/CP,V]` |
| 输入 batch | ⚠️ 部分 | forward 前会拿到 batch，hook 后才替换为 local chunk |
| output tensors | ⚠️ | SFT 不 gather；GRPO/EBFT 可能 all-gather |
| 保存期 CPU 内存 | ❌ | CP `_save()` 会 `detach().cpu()`，增加 CPU copy |

真正的大头通常是：

```text
激活/attention: 随 S 增长
logits: B * S * V * dtype_bytes
```

CP 解决前者，CCE 解决后者。组合后，单 rank 上标准 logits 峰值从：

```text
B * S * V * dtype
```

先因 CP 变成：

```text
B * (S / CP) * V * dtype
```

再被 CCE 基本消掉。

## 8.2 通信开销

| 阶段 | 通信类型 | group | 频率 | 源码依据 |
|---|---|---|---|---|
| CP attention | ring K/V 通信 | CP group | 每层 attention | `ring_attn/patch.py:110-124`, `adapters/batch.py:137-149` |
| CP token count | `all_reduce(AVG)` | CP group | forward 输入处理 | `sequence_parallel.py:156-165` |
| eval loss correction | 两次 `all_reduce(SUM)` | CP group | eval forward | `sequence_parallel.py:321-334` |
| output gather | shape + tensor `all_gather` | CP group | GRPO/EBFT | `sequence_parallel.py:393-414` |
| CCE vocab parallel | 多次 `all_reduce` | TP group | loss forward/backward | 外部 `vocab_parallel/utils.py:47-76` |
| CCE ZeRO-3 | parameter gather | ZeRO group | loss backward | 外部 `cce.py:179-195` |
| FSDP | parameter/grad/state 通信 | DP/FSDP group | 每层 / 保存 | FSDP 后端 |

这里要特别注意：**CCE 本身不是 CP 通信的一部分**。如果没有 TP/DTensor 或 ZeRO-3 特例，它只是在本 rank 上用 kernel 避免 logits。

## 8.3 性能取舍

CP 是典型的通信换显存：

- 序列越长，单卡 OOM 越严重，CP 越有价值；
- 序列较短时，ring 通信和 patch 成本可能不划算；
- CP + FSDP + TP 叠加时，通信维度变多，Axolotl 源码没有显式 overlap 调度。

CCE 是 kernel 复杂度换显存：

- 避免巨大 logits；
- 但引入 Triton kernel 分块、locks、可选 fp32 accum；
- TP vocab-parallel 下还要 loss 层 all-reduce；
- ZeRO-3 下 backward 可能 gather `lm_head.weight`。

外部 CCE backward 明确要求 embedding/classifier 是 fp16/bf16：

```python
# external cce_backward.py:349-356
assert e.dtype in (torch.float16, torch.bfloat16)
assert c.dtype in (torch.float16, torch.bfloat16)
```

这也解释了 Axolotl 插件为什么 schema 层要求 `bf16` 或 `fp16`。

## 8.4 本章小结

> 💡 **小结**
>
> * CP 主要省 sequence 相关激活和 attention 显存。
> * CCE 主要省 loss 阶段 logits 显存。
> * CP 的主要代价是每层 attention 通信；CCE 的主要代价是复杂 kernel 和可选 TP/ZeRO3 通信。
> * SFT 不 gather 输出是组合收益能保留的关键。

# 九、配置项、边界条件与坑点

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `context_parallel_size` | `validation.py:1508-1561`, `train.py:205-219` | 开启 CP context 和 sequence slicing | world topology 不合法会在后续 mesh/Accelerate 阶段暴露 |
| `sequence_parallel_degree` | `validation.py:1508-1514` | 迁移到 `context_parallel_size` | deprecated，不应继续使用 |
| `flash_attention` | `validation.py:1517-1520` | CP 必须开启 | 没开直接报错 |
| `sample_packing` + `micro_batch_size>1` | `validation.py:1522-1526` | 禁止组合 | ring-flash-attn 要求导致 |
| `ring_attn_func` | `validation.py:1564-1579`, `ring_attn/patch.py:186-212` | 选择 varlen 或 batch ring patch | 不影响 slicing，只影响 attention backend |
| `heads_k_stride` | `config.py:981-985`, `ring_attn/patch.py:200-202` | 传给 varlen llama3 ring attention | 描述要求整除 KV heads，但未在 Axolotl schema 中看到显式校验 |
| `plugins` | `cli/config.py:208-220` | 注册 CCE plugin | 不写 plugin，CCE hook 不会执行 |
| `cut_cross_entropy` | `args.py:33`, `__init__.py:86-103` | pre-model-load patch forward | plugin schema 字段，不是基础字段 |
| `bf16` / `fp16` | `args.py:35-42` | CCE backward dtype 前置要求 | fp32 训练会报错 |
| `chunked_cross_entropy` / Liger CE | `validation.py:974-1002` | 与 CCE 互斥 | 不能叠加多个 CE 优化 |
| `tensor_parallel_size` | 外部 `apply_lce()` | CCE 可能走 vocab-parallel all-reduce | loss 层通信增加 |
| `fsdp_config.state_dict_type` | `train.py:294-334` | 影响保存格式 | CP 不改变 FSDP 保存语义 |
| `trust_remote_code` | `cut_cross_entropy/__init__.py:100-103` | 下游 patch 可能 patch remote class | Axolotl generic fallback 不真正使用 remote class |
| GRPO async | `core/trainers/grpo/__init__.py:39-48` | CP + async GRPO 禁止 | 会直接 ValueError |

## 本章小结

> 💡 **小结**
>
> * CP 的坑主要来自 topology、FlashAttention、sample packing 和 batch size。
> * CCE 的坑主要来自 plugin 注册、dtype、模型类 patch 命中和 CE 互斥。
> * `heads_k_stride` 这类字段有文档约束，但源码校验有限。
> * CP + CCE 组合时，最重要的是确认两者都真的进入主路径。

# 十、测试、示例与覆盖缺口

## 10.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/test_context_parallel_batch_size.py` | CP 改变 effective batch size | CPU 测试，mock `ring_flash_attn` |
| `tests/test_loaders.py:181-218` | `_get_parallel_config_kwargs` | 覆盖 TP/CP/DP/FSDP size 组合 |
| `tests/e2e/multigpu/patched/test_sp.py` | sequence parallel 训练 | 当前被 skip，原因是 ring_flash_attn + transformers import upstream 问题 |
| `tests/e2e/integrations/test_cut_cross_entropy.py` | CCE e2e | Llama / Qwen2 / attention 配置，检查训练输出 |
| `tests/core/test_async_grpo.py` | GRPO async 与 CP 冲突 | 覆盖 trainer class 选择 |
| `examples/alst/*` | 超长上下文 CP + CCE 示例 | 500K sequence、sample packing、CCE plugin |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml` | FSDP + TP + CP + CCE 示例 | 多维并行组合示例 |

## 10.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| CP + CCE 同时开启的真实 e2e | ⚠️ 示例有，未见专门测试 | patch 边界问题只能运行时暴露 |
| ring attention 正确性 | ⚠️ e2e 被 skip | 长序列 correctness / loss parity 不稳 |
| 多机 CP | ❌ | rank mapping / group 通信问题 |
| 保存 / resume + CP + CCE | ❌ | storage 修复、CPU 峰值、resume patch 缺失 |
| CCE patch 是否命中真实模型类 | ⚠️ 少数 e2e 间接覆盖 | 日志显示启用但显存不降 |
| 性能 / 显存收益断言 | ❌ | 回归难以及时发现 |
| output logits 依赖 | ❌ | CCE 返回 logits=None 破坏 callback |
| CCE TP / ZeRO-3 分支 | ⚠️ 依赖外部分支 | loss all-reduce / gather 行为未被 Axolotl 专测 |

## 10.3 本章小结

> 💡 **小结**
>
> * CP 的 batch size 和 topology helper 有单元测试。
> * CP 真正多 GPU e2e 当前被 skip，是最大缺口。
> * CCE 有独立 e2e，但缺少 CP+CCE 组合测试。
> * 显存收益、patch 命中和保存/resume 仍主要靠人工验证。

# 十一、局限性与已知优化点

## 11.1 硬约束

1. `context_parallel_size > 1` 必须开启 `flash_attention`。
2. 必须安装 `ring_flash_attn`。
3. sample packing 下 `micro_batch_size` 必须为 1。
4. CCE 必须 fp16/bf16。
5. CCE 目标模型必须在外部 `PATCH_FNS` 或 Axolotl generic fallback 能正确 patch 的范围内。
6. 多种 CE 优化不能同时开启。
7. GRPO async 与 CP 不能同时开启。
8. `heads_k_stride` 的整除约束主要在描述中体现，源码未见完整校验。

## 11.2 维护成本

- CP patch 修改 HF FlashAttention 函数，依赖 Transformers 内部 API。
- CCE patch 替换模型 `forward()`，依赖模型类结构和 forward 签名。
- `SequenceParallelContextManager.__exit__()` 不恢复 attention / accelerate patch。
- CCE 类级 patch 和 `_PATCH_OPTS` 是全局状态。
- 文档与源码存在不一致：sequence chunking 当前不在 collator。

## 11.3 性能瓶颈

- CP 每层 attention 都有 CP group 通信；
- GRPO/EBFT output gather 会恢复完整 sequence 输出；
- eval loss correction 有额外 all-reduce；
- CCE Triton kernel 有 block/lock/accum 成本；
- CCE TP vocab-parallel 有 loss 层 all-reduce；
- ZeRO-3 下 CCE backward 可能 gather full lm_head；
- CP 保存时 `_save()` 会把 state_dict tensor 搬到 CPU：

```python
# src/axolotl/core/trainers/base.py:812-823
if state_dict is not None and self.axolotl_cfg.context_parallel_size > 1:
    state_dict = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in state_dict.items()
    }
```

这解决 safetensors storage 指针问题，不是显存优化。

## 11.4 已知优化点

源码里已有几个明显方向：

- `sequence_parallel.py:22-23` TODO：zigzag / stripe pattern 还没完整实现；
- `sequence_parallel.py:244` TODO：退出时 un-patch attention 和 accelerate；
- `core/trainers/grpo/trainer.py:254-258` TODO：未来可用 Accelerate dataloader dispatch + sequence slice function 替代 GRPO 当前手写路径；
- `core/trainers/base.py:805` TODO：保存修复等待上游 Transformers 合并后移除；
- CCE 下游参数如 `impl`、`filter_eps`、`accum_e_fp32` 当前 Axolotl 未暴露，未来可作为性能调参面。

## 11.5 本章小结

> 💡 **小结**
>
> * 这套实现的硬约束主要来自 FlashAttention、ring_flash_attn、dtype 和模型 patch 命中。
> * 最大维护成本是 monkey patch 与上游模型 forward 漂移。
> * 最大性能瓶颈是 CP 每层通信和 CCE 特殊分支通信。
> * 最值得补的是 CP+CCE e2e、保存/resume、显存收益和 patch 恢复测试。

# 小结与展望

Axolotl 的 Context Parallelism + CCE 实现可以用几个关键词概括。

## 关键词一：多层开关

`context_parallel_size` 不是一个简单 int；它会穿过 validation、env、Accelerate、DeviceMesh、Trainer context、attention patch 和保存路径。CCE 也不是裸字段，而是 plugin schema + pre-model-load patch。

## 关键词二：DeviceMesh 上的同 batch 分发

CP 的本质不是让更多 GPU 处理更多样本，而是让一个 CP group 协作处理同一条长序列。这个 rank 语义变化，是后续 sequence slicing 和 ring attention 正确性的前提。

## 关键词三：forward hook 切 sequence

Axolotl 没有重写模型，也没有把 chunking 放在 collator 主路径，而是在模型 forward 前用 hook 把 `[B,S]` 切成 `[B,S/CP]`。这让现有 Trainer 链路可以复用，但也带来 hook 与 patch 的状态管理成本。

## 关键词四：hidden states + lm_head 直接算 loss

CCE 的关键不是“换一个 CrossEntropyLoss”，而是在 `lm_head(hidden_states)` 之前接管 forward。它从 local hidden states 和 `lm_head.weight` 直接计算 loss，避免 logits 物化。组合 CP 后，CCE 处理的是本地 sequence chunk，因此 SFT 下不需要 gather 完整 logits。

## 关键词五：通信换显存

CP 用每层 ring attention 通信换长序列激活/attention 显存；CCE 用复杂 loss kernel 换 logits 显存。二者解决的是不同瓶颈，通信边界也不同：CP 是 CP group 内 sequence 通信，CCE 默认本地计算，只有 TP vocab-parallel 或 ZeRO-3 分支会额外通信。

这套实现适合：

- 超长上下文；
- sample packing；
- 大词表模型；
- logits 峰值明显的 SFT；
- FSDP/TP/CP 多维组合训练。

不适合：

- 短序列、通信开销大于收益的场景；
- 依赖训练时 `outputs.logits` 的自定义逻辑；
- 未被 CCE 正确 patch 的 remote / multimodal / wrapper 模型；
- 缺少高速互联的多机 CP；
- 对 monkey patch 隔离要求很高的长生命周期进程。

后续最值得继续走读的方向，是 `ring_flash_attn` 内部前后向通信、CP + FSDP2 + QLoRA 的保存边界，以及 GRPO/vLLM 下 CP 与生成、reward、advantage 的完整链路。只有把这些外部和 RL 分支也串起来，才能完整评估 Axolotl 在超长上下文训练中的真实扩展边界。
