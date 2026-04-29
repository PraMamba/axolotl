# Axolotl 源码走读：ring-flash-attn 内部 forward/backward 通信和显存曲线实现解析

在长序列训练里，FlashAttention 已经把注意力矩阵从“显式物化”改成了块状在线计算，但它并没有改变一个更朴素的事实：每个 GPU 仍然要拿到完整序列上的 Q/K/V、MLP 激活、labels 和 logits。FSDP / ZeRO 可以切参数、optimizer state，却不会天然切掉序列维度上的激活峰值。

Axolotl 的 `context_parallel_size`，在文档中也称 Sequence Parallelism / Context Parallelism，正是沿着这个缺口接入的：同一条长序列被切到一个 CP group 的多个 rank 上，rank 内只保留局部 token；真正跨 token 的 attention 由 `ring-flash-attn` 在 attention kernel 边界内通过通信补回来。

本文不重新讲 Ring Attention 或 FlashAttention 的论文原理，而是从 Axolotl 源码出发，回答一个工程问题：**用户在 YAML 里写下 `context_parallel_size: 2/4/8` 后，框架到底在哪些地方改变了训练路径？forward/backward 通信如何发生？显存曲线又是哪一段真的下降、哪一段只是转移成通信和 buffer？**

> 说明：当前本地环境没有安装 `ring_flash_attn` Python 包；Axolotl 在 `pyproject.toml:93-96` 声明的可选依赖是 `ring-flash-attn>=0.1.7`。为了分析下游 forward/backward 内部通信，本文额外下载并阅读了 `ring-flash-attn==0.1.7` 的源码，文中用 `ring_flash_attn-0.1.7/...` 标注该外部依赖源码路径。Axolotl 自身源码判断仍以 `/root/axolotl` 仓库为准。

# 前言

## 业务 / 工程背景

`ring-flash-attn` 在 Axolotl 里出现于**长上下文训练**，尤其是几十 K 乃至数百 K token 的 SFT / 预训练 / 部分 RL 场景。典型配置来自 `docs/sequence_parallelism.qmd:21-30`：

```yaml
flash_attention: true
context_parallel_size: 4
heads_k_stride: 1
ring_attn_func:
```

它解决的不是模型参数放不下的问题，而是**序列维度上的激活与 attention 计算上下文放不下**的问题。示例 `examples/alst/llama3-8b-fsdp2-alst.yaml:18-24` 甚至把 `sequence_len` 设到 `500_000`，同时打开 `sample_packing: true`、`context_parallel_size: 8` 和 CCE 插件，这种配置如果每张卡都吃完整序列，基本不可能靠普通 FlashAttention 扛住。

## 核心矛盾

这套实现的核心矛盾可以压缩成三句话：

1. **FSDP 切参数，不切序列。** 即使参数、梯度、optimizer state 被 sharding，每层 hidden states、Q/K/V、logits 仍会随 sequence length 线性增长。
2. **序列可以切，但 attention 不是局部算子。** rank 只拿自己的 token 后，本地 Q 仍然需要看到前面所有可见 token 的 K/V，否则 causal attention 语义就错了。
3. **通信不能把显存收益抵消掉。** 如果每层都 all-gather 完整 K/V 到每张卡，显存会回到接近完整序列；如果每步只流动一块 K/V，则通信多了，但峰值显存下降。

Axolotl 的答案不是把所有模块都改写成 sequence-parallel aware，而是在三个边界下注入：

- 配置 / Accelerate / DeviceMesh 边界：建立 CP group；
- model forward 边界：用 hook 切输入、必要时 gather 输出；
- HF FlashAttention 边界：monkey patch 到 `ring-flash-attn`。

## 本文主线

本文按机制而不是文件展开：

1. 配置与初始化：`context_parallel_size` 如何变成 env、DeviceMesh 和 process group；
2. forward hook：Axolotl 为什么在 model forward 外切输入，而不是靠 DataLoader；
3. attention patch：HF 的 `_flash_attention_forward` 如何被替换；
4. 完整主路径：从 `axolotl train` 到每个 training step；
5. shape / state / rank 流程：同一条序列在 rank 间怎么切；
6. forward/backward 深挖：batch ring 与 varlen llama3 两条路径的通信差异；
7. 显存、性能、配置、测试与边界。

## 不展开的内容

本文不讲 FlashAttention kernel 内部 tile 算法，不讲 FSDP / ZeRO 原理，不讲 LoRA / QLoRA 原理，也不展开 ring attention 的数学证明。我们只看 Axolotl 怎么把下游 `ring-flash-attn` 接进训练链路，以及源码暴露出的收益与代价。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/train.py` | CLI 训练入口，加载配置和数据后调用 `axolotl.train.train`。 |
| `src/axolotl/cli/config.py` | 读取 YAML、CLI override，执行 `validate_config()`、`prepare_optim_env()`、`normalize_config()`。 |
| `src/axolotl/utils/schemas/config.py` / `validation.py` | 声明 `context_parallel_size`、`heads_k_stride`、`ring_attn_func`，并做 CP 约束校验和默认值选择。 |
| `src/axolotl/utils/trainer.py` | 写入 Accelerate parallelism env，并 patch Accelerate CP prepare 路径。 |
| `src/axolotl/utils/distributed.py` | 根据 world size / TP / CP / DP 构造 `ParallelismConfig` 和 `DeviceMesh`。 |
| `src/axolotl/loaders/patch_manager.py` | 模型加载前注册 Transformers / Accelerate 相关 monkey patch。 |
| `src/axolotl/train.py` | 训练前进入 `SequenceParallelContextManager`，训练后保存模型。 |
| `src/axolotl/utils/ctx_managers/sequence_parallel.py` | forward pre-hook 切 batch，post-hook 可选 gather 输出，eval loss 修正。 |
| `src/axolotl/monkeypatch/ring_attn/patch.py` / `adapters/batch.py` | 从 DeviceMesh 获取 CP group，并替换 HF FlashAttention 实现。 |
| `ring_flash_attn-0.1.7/ring_flash_attn/*.py` | 下游 `ring-flash-attn` 的 batch ring、varlen llama3 forward/backward 通信实现。 |

> 💡 **小结**
>
> * `ring-flash-attn` 在 Axolotl 中是为长序列训练服务的，不是参数 sharding 方案。
> * 真实主线是“配置建拓扑 → forward hook 切输入 → attention patch 内通信”。
> * sample packing 与非 sample packing 走的 `ring_attn_func` 默认不同，后面的通信形态也不同。

# 一、配置与初始化：把“长序列切分”变成可执行拓扑

## 1.1 设计哲学与核心问题

一个容易低估的问题是：`context_parallel_size` 不能只是一个 trainer 参数。它必须同时影响：

- 配置合法性：是否安装 `ring_flash_attn`，是否打开 `flash_attention`；
- 全局 batch size 估算：同一 CP group 的 rank 拿的是同一 batch 的不同序列片段；
- Accelerate 拓扑：需要有名为 `cp` 的 mesh dimension；
- HF Trainer 兼容：Transformers 原生 context parallel 默认偏向 SDPA，Axolotl 要允许 FlashAttention；
- ring attention patch：attention forward 必须知道当前 CP process group。

如果这层缺失，后面即使能切 tensor，也找不到正确的通信 group；如果只建 group 不修 Trainer，HF 可能在 `_prepare_context_parallel_inputs` 阶段拒绝 `flash_attention_2`；如果只修 Trainer 不改 batch size，日志中的 effective batch 会被按 world size 过度放大。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/train.py
  - do_cli：读取 YAML 并进入 do_train。
  - do_train：加载数据，调用 axolotl.train.train。

src/axolotl/cli/config.py
  - load_cfg：读取配置、校验、准备 env、归一化。

src/axolotl/utils/schemas/config.py
  - context_parallel_size / heads_k_stride / ring_attn_func：用户可见配置。

src/axolotl/utils/schemas/validation.py
  - check_context_parallel_size：CP 基础约束。
  - validate_ring_attn_func：根据 sample_packing 选择默认 ring 函数。

src/axolotl/utils/trainer.py
  - setup_parallelism_envs：写 Accelerate parallelism env，并 patch CP prepare。

src/axolotl/utils/distributed.py
  - build_parallelism_config：构造 ParallelismConfig 和 DeviceMesh。
```

## 1.3 主流程拆解

用户入口很普通：

```text
axolotl train config.yml
  -> src/axolotl/cli/train.py:55-91 do_cli(config, **kwargs)
    -> load_cfg(config, **kwargs)
    -> do_train(parsed_cfg, parsed_cli_args)
      -> train(cfg, dataset_meta)
```

`load_cfg()` 的关键顺序在 `src/axolotl/cli/config.py:244-346`：先读 YAML，再合并 CLI overrides，随后调用 `validate_config()`，再执行 `prepare_optim_env()` 和 `normalize_config()`。这意味着 `context_parallel_size` 在模型加载前已经被校验、归一化并写入 env。

配置字段本身在 schema 里：

```text
src/axolotl/utils/schemas/config.py:969-991
  sequence_parallel_degree: deprecated，转到 context_parallel_size
  context_parallel_size: 按 GPU 数切分序列
  heads_k_stride: KV head 分块步长
  ring_attn_func: varlen_llama3 / batch_ring 等
```

真正的校验在 `validation.py`：

```python
# src/axolotl/utils/schemas/validation.py:1508-1577 的简化版
if sequence_parallel_degree and not context_parallel_size:
    context_parallel_size = sequence_parallel_degree
if not context_parallel_size:
    context_parallel_size = 1
elif context_parallel_size > 1:
    if not flash_attention:
        raise ValueError(...)
    if sample_packing and micro_batch_size > 1:
        raise ValueError(...)
    import ring_flash_attn

if context_parallel_size > 1 and ring_attn_func is None:
    ring_attn_func = VARLEN_LLAMA3 if sample_packing else BATCH_RING
```

这里解决了两个早期错误：

- CP 必须和 FlashAttention 一起用（`validation.py:1517-1520`）；
- sample packing 下 `micro_batch_size` 必须为 1（`validation.py:1522-1526`），因为 varlen llama3 路径假设 packed data 预处理成 batch size 1，Axolotl wrapper 里也有 `assert batch_size == 1`（`src/axolotl/monkeypatch/ring_attn/patch.py:105-110`）。

然后 `prepare_optim_env()` 写入 Accelerate env：

```python
# src/axolotl/utils/trainer.py:621-640 的简化版
if cfg.context_parallel_size and cfg.context_parallel_size > 1:
    os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cfg.context_parallel_size)
    os.environ["ACCELERATE_ALLOW_CP_STANDALONE"] = "true"
    patch_prepare_cp()
if set_accelerate_parallelism_config:
    os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
```

这一步不是 attention 通信本身，但它让后续 `Accelerator` 有机会构造 parallelism config。`utils/distributed.py:299-316` 又提供了一条显式构造路径：

```python
pc_kwargs = _get_parallel_config_kwargs(world_size, tp, cp, dp_shard, dp_replicate, is_fsdp)
parallelism_config = ParallelismConfig(**pc_kwargs)
device_mesh = parallelism_config.build_device_mesh("cuda")
```

其中 CP 会进入 `pc_kwargs["cp_size"]`，并从 `remaining_world_size` 中除掉（`utils/distributed.py:334-336`）。例如：

```text
world_size = 8
context_parallel_size = 4
tensor_parallel_size = 1
默认 dp_shard_size = remaining_world_size = 2

DeviceMesh 逻辑维度近似为:
  dp_shard x cp = 2 x 4

CP group 0: rank0, rank1, rank2, rank3
CP group 1: rank4, rank5, rank6, rank7
```

`normalize_config()` 还会修正 batch size：

```python
# src/axolotl/utils/config/__init__.py:134-142
if cfg.world_size != 1:
    effective_world_size = world_size // context_parallel_size // tensor_parallel_size
    cfg.batch_size = cfg.batch_size * effective_world_size
```

这就是文档中“effective global batch size 会除以 `context_parallel_size`”的源码依据（`docs/sequence_parallelism.qmd:90-100`）。同一 CP group 的 rank 不是处理不同样本，而是共同处理同一批样本的不同 token chunk。

## 1.4 关键细节与误区澄清

> **误区 1：`context_parallel_size` 会在 schema 阶段校验一定整除 world size。**
>
> 源码中没有在 `validation.py:1508-1579` 直接检查 `WORLD_SIZE % context_parallel_size == 0`。它只校验 `flash_attention`、sample packing 的 `micro_batch_size` 和依赖导入。拓扑合法性更多依赖 Accelerate `ParallelismConfig` / `build_device_mesh()` 以及 `_validate_accelerator`，后者会检查 `total_size` 与进程数是否一致（`src/axolotl/monkeypatch/accelerate/parallelism_config.py:21-26`）。因此某些不整除或组合不合理的配置不是在 schema 层报错，而是在分布式初始化或 device mesh 构造阶段暴露。

> **误区 2：`ring_attn_func` 是用户必须显式设置的开关。**
>
> 不是。`validate_ring_attn_func()` 会在 CP 开启后选择默认值：`sample_packing=True` 时为 `varlen_llama3`，否则为 `batch_ring`（`validation.py:1563-1577`）。这两个默认值背后的通信机制差异很大，前者不是传统 P2P ring，而是 all-gather / reduce-scatter，后文会展开。

> **误区 3：文档说 DataCollator 负责 chunking，所以切序列发生在 collator。**
>
> 文档 `docs/sequence_parallelism.qmd:40-45` 的表述是“data collator handles chunking”，但当前主路径源码里真正切分 batch 的是 `SequenceParallelContextManager` 的 forward pre-hook：`sequence_parallel_pre_hook()` 调用 `apply_sequence_parallelism()`（`src/axolotl/utils/ctx_managers/sequence_parallel.py:255-288`）。collator 主要负责 padding labels / position_ids（`src/axolotl/utils/collators/batching.py:61-104`），不是 CP 主切分点。这里应以源码为准。

## 1.5 本章小结

> 💡 **小结**
>
> * `context_parallel_size` 首先改变配置校验、batch size 估算和 Accelerate parallelism env。
> * CP group 依赖 `DeviceMesh` 的 `cp` 维度，不是随便用 world group 通信。
> * schema 层没有完整兜住所有拓扑错误，部分约束会延迟到 Accelerate / DeviceMesh 初始化阶段。

# 二、Hook 与数据切分：为什么 Axolotl 不把切分交给 DataLoader

## 2.1 设计哲学与核心问题

序列并行最容易踩坑的地方，是“数据切分”必须和“模型实际 forward”对齐。

如果在 DataLoader 里直接让不同 rank 拿不同 sequence chunk，会遇到两个问题：

1. HF Trainer / TRL 仍会在 forward 前后处理 `labels`、`num_items_in_batch`、`position_ids`、`logits_to_keep` 等字段；如果切分太早，后续逻辑可能按完整序列假设处理局部片段。
2. CP 只需要在模型 forward 期间生效；保存、日志、部分生成路径并不应该永久看到被切过的 batch。

Axolotl 选择在 model forward 外层注册 hook：进入训练上下文后，每次 forward 前切输入；必要时 forward 后 gather 输出；退出上下文时移除 hook。这是一种“贴着模型边界切”的实现，避免大面积侵入 Trainer。

## 2.2 源码入口与关键对象

```text
src/axolotl/train.py
  - execute_training：在 trainer.train() 外进入 SequenceParallelContextManager。

src/axolotl/utils/ctx_managers/sequence_parallel.py
  - SequenceParallelContextManager.__enter__ / __exit__：注册与移除 hooks。
  - apply_sequence_parallelism：padding、position_ids、序列维度 chunk。
  - AllGatherWithGrad：输出 gather 的自定义 autograd。
```

## 2.3 主流程拆解

训练入口在 `src/axolotl/train.py:183-229`：

```python
with ExitStack() as stack:
    if cfg.context_parallel_size > 1:
        models = [trainer.model]
        if hasattr(trainer, "ref_model") and trainer.ref_model:
            models.append(trainer.ref_model)
        stack.enter_context(
            SequenceParallelContextManager(
                models=models,
                context_parallel_size=cfg.context_parallel_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                ring_attn_func=cfg.ring_attn_func,
                heads_k_stride=cfg.heads_k_stride,
                gather_outputs=cfg.rl in {RLType.GRPO, RLType.EBFT},
                device_mesh=trainer.accelerator.torch_device_mesh,
            )
        )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

这段代码说明两个重要事实：

- CP hook 包裹的是整个 `trainer.train()`，不是模型加载阶段，也不是保存阶段；
- `gather_outputs` 只在 GRPO / EBFT 这类 RL 路径打开，普通 SFT 不会把 logits 再 all-gather 回完整序列。

`SequenceParallelContextManager.__init__()` 先注册 ring attention，然后记录本 rank 在 CP group 内的 rank / world size（`sequence_parallel.py:207-213`）：

```text
self._register_ring_attn()
self.process_group = get_ring_attn_group()
self.local_rank = dist.get_rank(self.process_group)
self.local_world_size = dist.get_world_size(self.process_group)
```

进入上下文时，`__enter__()` 调用 `_register_model_hooks()`（`sequence_parallel.py:233-236`）。pre-hook 的关键逻辑是：

```python
# src/axolotl/utils/ctx_managers/sequence_parallel.py:257-288 的简化版
updated_kwargs = kwargs.copy()
# 将 positional args 映射为 forward 参数名
updated_kwargs, self.original_seq_len, self.pad_len = self.apply_sequence_parallelism(updated_kwargs)

if "labels" in updated_kwargs and not self.models[0].training:
    self._local_valid_tokens = (updated_kwargs["labels"] != -100).sum().float()
    updated_kwargs.pop("num_items_in_batch", None)
return remaining_args, updated_kwargs
```

真正切分发生在 `apply_sequence_parallelism()`：

```python
# sequence_parallel.py:51-167 的简化版
batch_size, original_seq_len = batch["input_ids"].shape

if batch.get("position_ids") is not None and batch_size == 1:
    update_ring_attn_params(position_ids=batch["position_ids"])
else:
    batch["position_ids"] = arange(original_seq_len).expand(batch_size, -1)

# pad 到 local_world_size 可整除，最多按 64 做 divisor
if total_seq_len % min(local_world_size, 64) != 0:
    对 input_ids / labels / attention_mask / position_ids 等右侧 padding

for key in batch:
    if tensor.dim() > 1 and tensor.size(1) == total_seq_len:
        batch[key] = tensor.chunk(local_world_size, dim=1)[local_rank].contiguous()
```

shape 可以写成：

```text
进入模型前（每个 CP rank 初始看到同一完整 batch）:
  input_ids:      [B, S]
  labels:         [B, S]
  position_ids:   [B, S]

pre-hook padding 后:
  S' = ceil_div(S, cp_size) * cp_size

pre-hook chunk 后，CP rank r:
  input_ids:      [B, S'/cp_size]
  labels:         [B, S'/cp_size]
  position_ids:   [B, S'/cp_size]
```

如果 batch 中带 `num_items_in_batch`，代码会在 CP group 内 all-reduce token count：

```python
# sequence_parallel.py:150-165
local_valid_tokens = (batch["labels"] != -100).sum()
global_valid_tokens = local_valid_tokens.clone()
dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.AVG, group=cp_group)
batch["num_items_in_batch"] = global_valid_tokens * gradient_accumulation_steps
```

注释说明这里刻意用 `AVG` 而不是 `SUM`，因为 SUM 会让 loss 缩放过度（`sequence_parallel.py:156-160`）。这不是 ring attention 的通信，而是 loss normalization 的辅助通信。

post-hook 只在 `gather_outputs=True` 时收集输出：

```python
# sequence_parallel.py:350-363
if self.gather_outputs:
    model.register_forward_hook(sequence_parallel_post_hook)
...
output[key] = AllGatherWithGrad.apply(value, self.process_group)
```

`AllGatherWithGrad.forward()` 先 all-gather shape，再 all-gather tensor，最后沿 sequence 维拼接（`sequence_parallel.py:389-415`）；反向时不再通信，只从完整梯度中切回本 rank 的 slice（`sequence_parallel.py:437-443`）。

## 2.4 关键细节与误区澄清

> **误区 4：普通 SFT 会在 forward 后 all-gather logits，所以 logits 显存没有省。**
>
> 普通 SFT 的 `gather_outputs=False`，因为 `execute_training()` 只对 `cfg.rl in {GRPO, EBFT}` 打开 gather（`train.py:217`）。因此 SFT 主路径中，模型输出和 logits 通常保持局部序列长度，显存收益不会被一个 post-forward logits all-gather 抵消。GRPO / EBFT 是另一条路径，需要完整输出或跨 rank 后处理，才打开 `AllGatherWithGrad`。

> **误区 5：Accelerate 的 context parallel 负责真实切分，Axolotl hook 只是补充。**
>
> Axolotl 在 `setup_parallelism_envs()` 中调用 `patch_prepare_cp()`（`utils/trainer.py:632-638`）。这个 patch 把 `Accelerator._prepare_cp` 改成：非 deepspeed CP 后端下设置 `_cp_context` 为 no-op，并直接返回 args（`monkeypatch/accelerate/parallelism_config.py:80-97`）。基于这段源码可以推断，Axolotl 主路径避免让 Accelerate 自己切 buffers，而是由 `SequenceParallelContextManager` 的 pre-hook 负责切分。

> **误区 6：eval loss 的 NaN 是 ring attention 算错。**
>
> 更准确地说，CP 切片后某些 rank 的 local labels 可能全是 `-100`，局部 loss 变成 NaN。Axolotl 一方面 patch Transformers eval / logging 逻辑使用 `nanmean`（`src/axolotl/monkeypatch/transformers/trainer_loss_calc.py:22-36`），另一方面在 eval post-hook 里用 `weighted_loss / total_valid` 修正 CP group 内 loss（`sequence_parallel.py:305-340`）。这是 loss 聚合问题，不是 attention 数值公式本身的问题。

## 2.5 本章小结

> 💡 **小结**
>
> * Axolotl 的序列切分发生在 model forward pre-hook，而不是 DataLoader 主路径。
> * SFT 默认不 gather 输出，GRPO / EBFT 才通过 `AllGatherWithGrad` 恢复完整序列输出。
> * CP 还带来 loss normalization / eval NaN 修正通信，这些通信不属于 ring attention kernel，但会出现在 step 内。

# 三、Attention Patch：把 HF FlashAttention 换成 ring-flash-attn

## 3.1 设计哲学与核心问题

切完输入以后，模型内部每层 attention 仍然以为自己只需要处理局部序列。如果直接调用原生 FlashAttention，它只能在 `[B, S/cp]` 上做 attention，语义变成本地窗口 attention，和完整 causal attention 不一致。

Axolotl 没有重写每个模型的 attention module，而是替换 HuggingFace Transformers 的 FlashAttention 入口：只要模型使用 `flash_attention_2`，最终就会走到被替换的 `_flash_attention_forward`。这是一种“低侵入，高耦合”的设计：模型代码基本不用改，但 patch 依赖 Transformers 和 `ring_flash_attn` 的函数签名。

## 3.2 源码入口与关键对象

```text
src/axolotl/loaders/patch_manager.py
  - _apply_transformers_patches：注册 Trainer loss / CP guard patches。
  - _apply_fsdp_patches：注册 Accelerate ParallelismConfig patch。

src/axolotl/monkeypatch/transformers/trainer_context_parallel.py
  - patch_prepare_context_parallel_inputs：允许 flash_attention_2 通过 HF CP guard。

src/axolotl/monkeypatch/ring_attn/patch.py
  - register_ring_attn_from_device_mesh：提取 CP group，设置全局 RING_ATTN_GROUP，替换 attention。
  - update_ring_attn_params：sample packing 时更新 varlen cu_seqlens。

src/axolotl/monkeypatch/ring_attn/adapters/batch.py
  - substitute_hf_flash_attn：batch_ring 路径的 HF adapter。
```

## 3.3 主流程拆解

PatchManager 在模型加载阶段运行。`ModelLoader.load()` 的顺序是：

```text
src/axolotl/loaders/model.py:162-194
  -> patch_manager.apply_pre_model_load_patches()
  -> _apply_pre_model_load_setup()
  -> _build_model()
  -> _apply_post_model_load_setup()
  -> _load_adapters()
  -> patch_manager.apply_post_model_load_patches(model)
```

其中 Transformers 相关 patch 在 `patch_manager.py:135-149`：

```python
patch_evaluation_loop()
patch_maybe_log_save_evaluate()
if self.cfg.context_parallel_size > 1:
    patch_prepare_context_parallel_inputs()
```

`patch_prepare_context_parallel_inputs()` 做的是源码字符串替换。原来的 guard 是：

```python
if model.config._attn_implementation != "sdpa":
```

Axolotl 替换为允许 `sdpa` 或 `flash_attention_2`（`trainer_context_parallel.py:15-17`），然后通过 `exec()` 生成新函数并挂回 `Trainer._prepare_context_parallel_inputs`（`trainer_context_parallel.py:38-69`）。

注意，这个 patch 只是放行 HF Trainer 的 CP guard；真正的 attention 替换发生在训练上下文初始化：

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

```text
src/axolotl/monkeypatch/ring_attn/patch.py:159-184
  1. sequence_mesh = device_mesh[("cp",)]
  2. sequence_pg = sequence_mesh.get_group()
  3. set_ring_attn_group(sequence_pg)
```

然后按 `ring_attn_func` 分支 patch：

```python
# patch.py:186-211 的简化版
if ring_attn_func is VARLEN_LLAMA3:
    ring_flash_attn.adapters.hf_adapter.create_ring_flash_attention_forward = create_ring_flash_attention_forward
    ring_flash_attn.adapters.hf_adapter.substitute_hf_flash_attn(process_group, heads_k_stride)
elif ring_attn_func is BATCH_RING:
    axolotl.monkeypatch.ring_attn.adapters.batch.substitute_hf_flash_attn(process_group, ring_attn_func)
```

`batch_ring` 分支最终把 `transformers.modeling_flash_attention_utils._flash_attention_forward` 替换成 Axolotl wrapper，并把 `ALL_ATTENTION_FUNCTIONS["flash_attention_2"]` 指向 `ring_flash_attn.adapters.hf_adapter.flash_attention_forward`（`adapters/batch.py:167-196`）。

`varlen_llama3` 分支更绕：Axolotl 先把下游 `ring_flash_attn.adapters.hf_adapter.create_ring_flash_attention_forward` 这个工厂函数替换成自己的版本（`patch.py:186-198`），再调用下游 package 的 `substitute_hf_flash_attn()`（`patch.py:200-202`）。Axolotl 这样做的目的，是适配当前 Transformers 版本的 `_flash_attention_forward` 签名，并传入 `DATA_PARAMS` 中的 varlen 信息。

## 3.4 关键细节与误区澄清

> **误区 7：patch 只在 `with SequenceParallelContextManager` 内局部生效。**
>
> hook 是局部的，但 attention patch 不是。`SequenceParallelContextManager.__exit__()` 只移除 hook handles（`sequence_parallel.py:238-242`），并留下 TODO：“Un-patch attention and accelerate functions”（`sequence_parallel.py:244`）。`RING_ATTN_GROUP` 也是模块级全局变量（`patch.py:34-47`）。因此 attention monkey patch 在进程内是全局污染式的，只是训练结束后通常进程也结束。

> **误区 8：`ring_attn_func=batch_zigzag` / `batch_stripe` 文档可用。**
>
> schema 描述中提到 `batch_zigzag`、`batch_stripe`（`config.py:987-991`），但 enum 当前只开放 `VARLEN_LLAMA3` 和 `BATCH_RING`，其他项被注释掉（`src/axolotl/utils/schemas/enums.py:100-108`）。`sequence_parallel.py:22-23` 也有 TODO 说明 zigzag / stripe 尚未实现到这个模块里。用户把这些字符串写进配置，按当前 enum 校验会失败。

> **误区 9：`update_ring_attn_params()` 是 batch_ring 的必要步骤。**
>
> `update_ring_attn_params()` 只在 batch 有 `position_ids` 且 `batch_size == 1` 时调用（`sequence_parallel.py:53-55`），它内部计算 `cu_seqlens` 并调用下游 `update_ring_flash_attn_params()`（`patch.py:214-226`）。这主要服务 sample packing + `varlen_llama3`。非 sample packing 默认走 `batch_ring`，通常由 wrapper 直接把 `[B, S_local, H, D]` 传给 `ring_flash_attn_func`，不依赖 `DATA_PARAMS`。

## 3.5 本章小结

> 💡 **小结**
>
> * Axolotl 没有重写每个模型 attention，而是替换 Transformers FlashAttention 入口。
> * CP group 来自 `DeviceMesh[("cp",)]`，随后写入模块级 `RING_ATTN_GROUP`。
> * attention patch 当前不恢复，属于进程级全局 monkey patch，维护风险主要来自上游函数签名变化。

# 四、完整主路径串联

## 4.1 完整调用栈

一次典型 SFT 长序列训练可以串成下面这条主路径：

```text
User: axolotl train config.yml
  │
  ├─ Step 1: 配置加载与校验
  │     ├─ src/axolotl/cli/train.py:55-91 do_cli()
  │     ├─ src/axolotl/cli/config.py:230-346 load_cfg()
  │     ├─ src/axolotl/utils/schemas/validation.py:1508-1579 CP 校验与默认 ring_attn_func
  │     └─ src/axolotl/utils/trainer.py:621-640 写 PARALLELISM_CONFIG_CP_SIZE 等 env
  │
  ├─ Step 2: patch 与模型加载
  │     ├─ src/axolotl/loaders/model.py:162-194 ModelLoader.load()
  │     ├─ src/axolotl/loaders/patch_manager.py:135-149 patch HF Trainer CP guard / loss
  │     └─ src/axolotl/loaders/model.py:437-443 构造 parallelism_config / device_mesh
  │
  ├─ Step 3: Trainer 构建
  │     ├─ src/axolotl/core/builders/base.py:588-590 average_tokens_across_devices=False
  │     └─ src/axolotl/core/builders/causal.py:431-439 创建 AxolotlTrainer
  │
  ├─ Step 4: 训练上下文启动
  │     ├─ src/axolotl/train.py:205-219 进入 SequenceParallelContextManager
  │     ├─ src/axolotl/monkeypatch/ring_attn/patch.py:159-184 取 CP group
  │     └─ src/axolotl/monkeypatch/ring_attn/patch.py:186-211 patch HF FlashAttention
  │
  ├─ Step 5: 每次 model.forward
  │     ├─ sequence_parallel.py:257-288 pre-hook 切 batch
  │     ├─ HF model attention -> patched _flash_attention_forward
  │     ├─ ring_flash_attn-0.1.7/... forward/backward 内部通信
  │     └─ sequence_parallel.py:291-363 可选 post-hook all-gather 输出
  │
  └─ Step 6: 训练结束与保存
        ├─ sequence_parallel.py:238-244 移除 hooks，但不 unpatch attention
        ├─ src/axolotl/train.py:254-374 save_trained_model()
        └─ src/axolotl/core/trainers/base.py:812-823 CP 保存时 CPU clone state_dict
```

## 4.2 每一层做了什么

| 层次 | 输入 | 输出 / 状态 | 是否通信 | 是否每 step 执行 | 显存影响 |
|---|---|---|---|---|---|
| 配置校验 | YAML / CLI args | `context_parallel_size` 归一到 int，`ring_attn_func` 默认值 | 否 | 否 | 无直接影响 |
| env 初始化 | cfg | `PARALLELISM_CONFIG_CP_SIZE`、`ACCELERATE_USE_PARALLELISM_CONFIG` | 否 | 否 | 让后续 mesh 可建立 |
| DeviceMesh | world size / TP / CP / DP | `torch_device_mesh`，含 `cp` 维度 | 初始化阶段可能有 distributed setup | 否 | 无直接激活节省 |
| ring patch | DeviceMesh `cp` group | `RING_ATTN_GROUP`，HF attention 函数被替换 | 否 | 否 | 改变每层 attention 实现 |
| pre-hook | 完整 batch tensor | 局部 sequence chunk | `num_items_in_batch` 时 all_reduce | 是 | input / labels / hidden 后续按 `S/P` 增长 |
| attention forward | local Q/K/V | local out / lse | ring P2P 或 all-gather | 每层每 forward | 避免完整 Q/K/V 和 attention 输出常驻 |
| attention backward | local dout / saved tensors | local dq/dk/dv | ring P2P 或 reduce-scatter | 每层每 backward | 需要临时 grad buffers |
| post-hook | local output | 可选完整 output | GRPO/EBFT 下 all_gather | 每 forward（若开启） | 会恢复输出显存峰值 |
| save | state_dict | checkpoint | FSDP/ZeRO 自身保存通信，CP 本身无 ring attention 通信 | checkpoint 时 | CP 下 state_dict tensor clone 到 CPU |

## 4.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `docs/sequence_parallelism.qmd:40-45` “collator handles chunking” | 文档把 chunking 归到 data collator | 当前训练主路径不是 | 真实切分在 `SequenceParallelContextManager` pre-hook。 |
| `Accelerator._prepare_cp` | Accelerate 原生 CP 也会 split buffers | Axolotl patch 后非 deepspeed CP 是 no-op | Axolotl 自己切 sequence；Accelerate 主要提供 topology / mesh。 |
| `ring_flash_attn_varlen.py` | 名字里有 varlen ring | Axolotl sample packing 默认不是这条 | Axolotl `VARLEN_LLAMA3` 分支调用 `llama3_flash_attn_varlen_func`。 |
| `zigzag_ring_flash_attn.py` / `stripe_flash_attn.py` | 下游 package 有实现，schema 文案提到 | Axolotl enum 未开放，sequence manager TODO | 当前不要把它们当作可用主流程。 |
| `AllGatherWithGrad` | 名字很核心 | SFT 不调用 | 只在 `gather_outputs=True`，当前 GRPO / EBFT 打开。 |
| `merge_lora` | 训练后可能 merge adapter | CP 被显式关闭 | `merge_lora.py:144-155` 重新 `load_cfg()` 时设置 `flash_attention=False`、`context_parallel_size=None`。 |

> 💡 **小结**
>
> * 主路径不是单个函数，而是配置、patch、hook、attention kernel 四层协作。
> * 训练 step 内最关键的是 pre-hook sequence chunk 和每层 attention 通信。
> * 保存阶段不再执行 ring attention，但 CP 会影响 state_dict 保存前的 tensor storage 处理。

# 五、关键数据流 / 状态流 / shape 流程

## 5.1 Tensor shape 变化

### 非 sample packing：`batch_ring`

假设：

```text
B = micro_batch_size
S = sequence_len after collator padding
P = context_parallel_size
H = num_heads
D = head_dim
V = vocab_size
```

主 shape 流程是：

```text
pre-hook 前，每个 CP rank 都看到完整 batch:
  input_ids:    [B, S]
  labels:       [B, S]

pre-hook 后，rank r:
  input_ids_r:  [B, S/P]
  labels_r:     [B, S/P]

模型 embedding / hidden:
  hidden_r:     [B, S/P, hidden]

每层 attention projection:
  q_r:          [B, S/P, Hq, D]
  k_r/v_r:      [B, S/P, Hkv, D]

ring_flash_attn_func forward:
  输入常驻:      q_r, 当前 k/v block
  临时接收:      next_k/next_v: [B, S/P, Hkv, D]
  输出:          out_r: [B, S/P, Hq, D]

LM head / loss（SFT，未 gather_outputs）:
  logits_r:     [B, S/P, V]
  loss_r:       scalar
```

真正节省显存的点是：embedding 后的每层 hidden、Q/K/V、MLP 中间激活、logits 都跟 `S/P` 走。增加的显存是 ring attention 里的 `next_k/next_v` 和 backward buffer，它们通常是局部 chunk 大小，而不是完整序列大小。

### sample packing：`varlen_llama3`

sample packing 下 `micro_batch_size` 被要求为 1。输入 shape 更接近：

```text
pre-hook 前:
  input_ids:     [1, S]
  position_ids:  [1, S]   # 多个样本拼接时 position_ids 会重置到 0

update_ring_attn_params() 用完整 position_ids 得到:
  cu_seqlens:    [num_packed_sequences + 1]

pre-hook 后 rank r:
  input_ids_r:   [1, S/P]

attention wrapper 内 squeeze:
  q_r/k_r/v_r:   [S/P, H, D]
```

下游 `llama3_flash_attn_varlen_forward()` 又按 KV head stride 做 all-gather：

```text
对每个 KV head block，大小 heads_k_stride:
  local k_slice:     [S/P, heads_k_stride, D]
  all_gather buffer: [S,   heads_k_stride, D]
  local_k_slice:     可能是 [0 : 当前 q 可见的 K 范围]
  flash_attn_varlen: q_i attends k_i/v_i
```

源码依据是 `ring_flash_attn-0.1.7/ring_flash_attn/llama3_flash_attn_varlen.py:84-119`：它分配 `kv_buffer`，异步 all-gather 当前 KV head block，然后用 `local_k_slice` 取本 rank 的可见 K/V 范围。

## 5.2 Rank / Mesh / Process Group 变化

设：

```text
world_size = 8
context_parallel_size = 4
tensor_parallel_size = 1
remaining dp size = 2
```

一个典型 CP 分组是：

```text
DP group 0 / CP group 0: rank0 rank1 rank2 rank3
DP group 1 / CP group 1: rank4 rank5 rank6 rank7

同一 CP group 内:
  rank0 拿 tokens [0      : S/4]
  rank1 拿 tokens [S/4    : S/2]
  rank2 拿 tokens [S/2    : 3S/4]
  rank3 拿 tokens [3S/4   : S]

不同 CP group:
  处理不同 data-parallel batch
```

`register_ring_attn_from_device_mesh()` 从 `device_mesh[("cp",)]` 提取 `sequence_pg`（`patch.py:159-170`），并写入 `RING_ATTN_GROUP`（`patch.py:183-184`）。之后：

- `apply_sequence_parallelism()` 用 `dist.get_rank(self.process_group)` 得到 CP local rank（`sequence_parallel.py:210-212`）；
- `ring_flash_attn` 内部通信使用同一个 process group；
- GRPO 的 sampler 也按 `rank // context_parallel_size` 推导 SP group（`src/axolotl/core/trainers/grpo/sampler.py:79-83`）。

## 5.3 状态切换

CP 相关状态有三类：

```text
进程级全局状态:
  RING_ATTN_GROUP = None / sequence_pg
  transformers.modeling_flash_attention_utils._flash_attention_forward = patched_fn

每次 forward 更新状态:
  ring_flash_attn.adapters.hf_adapter.DATA_PARAMS = {
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    local_k_slice,
  }

上下文局部状态:
  hook_handles
  original_seq_len / pad_len
  _local_valid_tokens
```

进入 `SequenceParallelContextManager`：

```text
1. 从 DeviceMesh 取 cp group
2. set_ring_attn_group(cp_group)
3. patch HF FlashAttention
4. register forward_pre_hook / forward_hook
```

每次 forward：

```text
1. pre-hook 根据完整 batch 更新 DATA_PARAMS（sample packing）
2. pre-hook 切 input_ids / labels / position_ids
3. attention 调用 patched _flash_attention_forward
4. 下游 ring_flash_attn 从 group / DATA_PARAMS 读状态
5. post-hook 可选 gather 输出 / 修正 eval loss
```

退出上下文：

```text
1. 移除 model hooks
2. 不恢复 RING_ATTN_GROUP
3. 不恢复 HF attention patch
```

这不是线程安全设计。训练通常是多进程单线程模型 forward，因此问题不大；但如果同一 Python 进程内交替训练多个模型或跑测试，patch 污染需要额外清理。Axolotl 自己的测试也只对 `Trainer._prepare_context_parallel_inputs` patch 做了 fixture 恢复（`tests/monkeypatch/test_trainer_context_parallel_patch.py:13-33`），没有覆盖 ring attention patch 的恢复。

> 💡 **小结**
>
> * CP 的核心 shape 收益来自把 sequence 维从 `S` 改成 `S/P`。
> * CP group 是通信边界，不是 world group；同组 rank 处理同一 batch 的不同 token。
> * `DATA_PARAMS` 和 HF attention patch 都是全局状态，sample packing 每次 forward 会刷新 varlen 参数。

# 六、核心机制深挖：Batch Ring 的 forward/backward 通信

## 6.1 设计哲学与核心问题

非 sample packing 默认走 `batch_ring`。它要解决的问题是：**rank r 的 Q 只在本地，但它需要看到所有 causal 可见的 K/V chunk；同时又不能把完整 K/V all-gather 到每张卡。**

最简单的做法是每层 all-gather K/V，所有 rank 都得到 `[B, S, Hkv, D]`，然后用 FlashAttention 计算本地 Q 对全局 K/V 的输出。但这会让 K/V 显存回到完整序列规模。

Batch ring 的做法是：K/V 沿环流动；每个 step 只持有当前 K/V block 和下一个接收 buffer；每拿到一个 block，就调用一次 FlashAttention，并用 log-sum-exp 公式把局部输出合并到最终 out。

## 6.2 源码入口与关键对象

```text
ring_flash_attn-0.1.7/ring_flash_attn/ring_flash_attn.py
  - ring_flash_attn_forward：前向 K/V ring 和 block 输出合并。
  - ring_flash_attn_backward：反向 K/V ring + dK/dV 回传 ring。
  - RingFlashAttnFunc：自定义 autograd Function。

ring_flash_attn-0.1.7/ring_flash_attn/utils.py
  - RingComm：P2P isend/irecv 封装。
  - update_out_and_lse：稳定合并 block attention 输出。
```

## 6.3 主流程拆解：forward

`RingComm` 的通信方向定义在 `utils.py:98-151`：

```python
send_rank = (rank + 1) % world_size
recv_rank = (rank - 1) % world_size
send_recv_kv(k, v):
    isend(k -> send_rank), irecv(next_k <- recv_rank)
    isend(v -> send_rank), irecv(next_v <- recv_rank)
    batch_isend_irecv(...)
```

这意味着每一轮 rank 把自己的当前 K/V 传给下一个 rank，同时从前一个 rank 收到另一块 K/V。

`ring_flash_attn_forward()` 的核心循环在 `ring_flash_attn.py:19-67`：

```python
comm = RingComm(process_group)
for step in range(comm.world_size):
    if step + 1 != comm.world_size:
        next_k, next_v = comm.send_recv_kv(k, v)

    if not causal or step <= comm.rank:
        block_out, block_lse = _flash_attn_forward(q, k, v, causal=(causal and step == 0), ...)
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)

    if step + 1 != comm.world_size:
        comm.wait()
        k, v = next_k, next_v
```

以 `P=4` 的 causal attention 为例，可以画成：

```text
初始:
  rank0: Q0 K0 V0
  rank1: Q1 K1 V1
  rank2: Q2 K2 V2
  rank3: Q3 K3 V3

step0:
  所有 rank 计算本地块，causal=True
  rank0: Q0 attends K0
  rank1: Q1 attends K1
  rank2: Q2 attends K2
  rank3: Q3 attends K3
  同时 K/V 发送给下一个 rank

step1:
  rank0 收到 K3，但 step=1 > rank0，不计算（causal 下不可见）
  rank1 收到 K0，计算 Q1 attends K0
  rank2 收到 K1，计算 Q2 attends K1
  rank3 收到 K2，计算 Q3 attends K2

step2:
  rank2 计算 Q2 attends K0
  rank3 计算 Q3 attends K1

step3:
  rank3 计算 Q3 attends K0
```

所以 causal 下每个 rank 的计算量不完全均衡：rank0 只算 1 个 block，rank P-1 算 P 个 block。源码中的条件就是 `step <= comm.rank`（`ring_flash_attn.py:30`）。这也是朴素 ring causal 的一个性能特征。

`update_out_and_lse()` 负责把多个 K/V block 的 attention 输出合并成等价于全局 softmax 的输出。它不是简单相加，而是用 log-sum-exp 权重合并：

```python
# ring_flash_attn-0.1.7/ring_flash_attn/utils.py:32-73
out = out - sigmoid(block_lse - lse) * (out - block_out)
lse = lse - logsigmoid(lse - block_lse)
```

这就是为什么每个 block 可以单独调用 FlashAttention，最后仍能组合出全局 attention 结果。

## 6.4 主流程拆解：backward

反向更复杂，因为梯度分两类：

- `dQ` 属于本地 Q owner，多个 K/V block 对它都有贡献，直接累加即可；
- `dK/dV` 属于 K/V owner，但每个 owner 的 K/V 在 forward 中被多个 rank 使用，所以反向贡献需要沿环送回原 owner。

`ring_flash_attn_backward()` 在 `ring_flash_attn.py:85-154` 同时建了两个通信器：

```python
kv_comm = RingComm(process_group)      # 继续旋转 K/V，重放 forward 的可见块
d_kv_comm = RingComm(process_group)    # 反向旋转 dK/dV，把梯度送回 owner
```

核心流程可以写成：

```text
for step in range(P):
  1. kv_comm 异步发送当前 K/V，并接收下一块 K/V
  2. 若该 K/V block 对本 rank Q 可见：
       _flash_attn_backward(dout, q, k, v, out, lse)
       dq += block_dq
       dk/dv = 当前 block 贡献 + 从下游收到的 next_dk/next_dv
     否则：
       只等待并转发 next_dk/next_dv
  3. kv_comm.wait(); k/v = next_k/next_v
  4. d_kv_comm.send_recv_kv(dk, dv)
最后 wait，返回 dq 和回到本 rank 的 dk/dv
```

源码里对应位置：

- 分配临时 buffer：`block_dq_buffer`、`block_dk_buffer`、`block_dv_buffer`（`ring_flash_attn.py:90-92`）；
- 调用 FlashAttention backward：`_flash_attn_backward(**params)`（`ring_flash_attn.py:101-131`）；
- 累加 `dq`，等待并合并 `next_dk/next_dv`（`ring_flash_attn.py:133-144`）；
- 发送当前 `dk/dv` 到下一跳（`ring_flash_attn.py:150`）。

这条 backward 链路和 forward 并不完全对称：forward 只需要把 K/V 流给使用者；backward 还要把使用者产生的 dK/dV 贡献送回原 K/V owner。因此 backward 的通信通常比 forward 更重：除了重放 K/V ring，还多了一条 dK/dV ring。

## 6.5 关键细节与误区澄清

> **误区 10：forward 用 ring，backward 自动由 PyTorch all-reduce 梯度。**
>
> 不对。`RingFlashAttnFunc.backward()` 显式调用 `ring_flash_attn_backward()`（`ring_flash_attn.py:202-220`），后者内部自己发送 K/V 和 dK/dV。DDP / FSDP 感知的是参数梯度同步；它们不会自动知道序列维度上 K/V block 的所有权。

> **误区 11：ring forward 每个 rank 都计算 P 个 K/V block。**
>
> 非 causal 是这样；causal 下源码条件是 `step <= comm.rank`（`ring_flash_attn.py:30`）。因此 rank0 计算最少，最后一个 rank 计算最多。这个负载不均衡是 batch ring causal 路径的潜在性能瓶颈。

> **误区 12：batch ring 完全不需要额外显存。**
>
> 它不保存完整 K/V，但仍需要 `next_k/next_v` 接收 buffer、`block_*` backward buffers，以及 `out/lse` 的 FP32 累积形式。源码中 `update_out_and_lse()` 会把 block output 转成 float32（`utils.py:40,63-64`），backward 也把 `dq/dk/dv` 初始累积到 float32（`ring_flash_attn.py:133-136`）。显存下降来自“局部序列 + 流式 K/V”，不是零开销。

> **误区 13：梯度 dtype 完全遵循输入 dtype。**
>
> `ring_flash_attn_backward()` 最后返回 `dq.to(torch.bfloat16), next_dk.to(q.dtype), next_dv.to(q.dtype)`（`ring_flash_attn.py:154`）；`ring_flash_attn_varlen_backward()` 也有同样的 `dq.to(torch.bfloat16)`（`ring_flash_attn_varlen.py:192`）。这对非 bf16 训练是维护风险。Axolotl 常见长序列配置多用 bf16，但源码层面这里确实是硬编码。

## 6.6 本章小结

> 💡 **小结**
>
> * batch ring forward 的本质是 K/V P2P 流动，Q 留在本地，out/lse 用 log-sum-exp 合并。
> * batch ring backward 需要两条通信链：重放 K/V 可见性，并把 dK/dV 贡献送回 owner。
> * causal batch ring 有 rank 间计算不均衡；显存节省来自不 all-gather 完整 K/V，而不是没有临时 buffer。

# 七、核心机制深挖：Varlen Llama3 的 all-gather / reduce-scatter

## 7.1 设计哲学与核心问题

sample packing 让一个 `[1, S]` 序列里拼进多个样本。此时 attention 不能简单按位置做一个完整 causal mask：每个 packed sample 内部 causal，sample 之间不能串扰。Axolotl 已经通过 `position_ids` 重置来描述样本边界；`ring-flash-attn` 的 `llama3` varlen 路径则需要把这些边界转成 `cu_seqlens_q/k` 和 `local_k_slice`。

这条路径的设计取舍与 batch ring 不同。它不是 K/V 一块块沿 P2P ring 转，而是按 `heads_k_stride` 对 KV heads 分块，每次 all-gather 一个 KV head block，再用 `reduce_scatter_tensor` 把 dK/dV 梯度分回 owner。它牺牲了一部分 buffer 显存，换取更规整的 varlen FlashAttention 调用。

## 7.2 源码入口与关键对象

```text
src/axolotl/utils/trainer.py
  - add_position_ids：sample packing 前为样本生成 position_ids。

src/axolotl/monkeypatch/utils.py
  - get_cu_seqlens_from_pos_ids：从 position_ids 重置点推导 cu_seqlens。

src/axolotl/monkeypatch/ring_attn/patch.py
  - update_ring_attn_params：把 cu_seqlens 写入下游 DATA_PARAMS。
  - create_ring_flash_attention_forward：Axolotl 版 varlen llama3 HF wrapper。

ring_flash_attn-0.1.7/ring_flash_attn/llama3_flash_attn_varlen.py
  - llama3_flash_attn_prepare_cu_seqlens
  - llama3_flash_attn_varlen_forward
  - llama3_flash_attn_varlen_backward
```

## 7.3 主流程拆解

sample packing 先在数据准备阶段加入 `position_ids`。`add_position_ids()` 对单样本生成 `0..seq_len-1`，对 batch 逐条生成（`src/axolotl/utils/trainer.py:99-130`）；当多样本 pack 成一条序列时，collator 会拼接 position ids（`src/axolotl/utils/collators/batching.py:181-194`）。

每次 forward pre-hook 中，如果看到 `position_ids` 且 `batch_size == 1`，就调用：

```python
# sequence_parallel.py:53-55
update_ring_attn_params(position_ids=batch["position_ids"])
```

Axolotl 的 `update_ring_attn_params()` 做两步：

```python
# src/axolotl/monkeypatch/ring_attn/patch.py:222-226
cu_seqlens, _ = get_cu_seqlens_from_pos_ids(position_ids)
cu_seqlens = cu_seqlens.squeeze().to(device=torch.cuda.current_device())
update_ring_flash_attn_params(cu_seqlens, get_ring_attn_group())
```

`get_cu_seqlens_from_pos_ids()` 根据 position id 从 0 重启的位置找 sample 边界（`src/axolotl/monkeypatch/utils.py:94-152`）。下游 package 的 `update_ring_flash_attn_params()` 再调用 `llama3_flash_attn_prepare_cu_seqlens()`，把全局 `cu_seqlens` 变成本 rank 的：

```text
ring_flash_attn-0.1.7/ring_flash_attn/adapters/hf_adapter.py:42-62
  DATA_PARAMS = {
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    local_k_slice,
  }
```

`llama3_flash_attn_prepare_cu_seqlens()` 要求总 token 数能被 world size 整除（`llama3_flash_attn_varlen.py:24-26`），然后按 rank 的 token 区间计算本地 Q 的 cu_seqlens 和可见 K 的 slice（`llama3_flash_attn_varlen.py:27-60`）。

forward 通信在 `llama3_flash_attn_varlen_forward()`：

```python
# ring_flash_attn-0.1.7/.../llama3_flash_attn_varlen.py:84-119 的简化版
kv_buffer = empty((2, total_k * world_size, heads_k_stride, head_dim))
kv_buffer_copy = empty_like(kv_buffer)

comm.all_gather(kv_buffer_copy[0], k[:, :heads_k_stride])
comm.all_gather(kv_buffer_copy[1], v[:, :heads_k_stride])

for i in range(0, nheads_k, heads_k_stride):
    comm.wait()
    swap(kv_buffer, kv_buffer_copy)
    if next head block exists:
        comm.all_gather(next k/v head block)

    q_i = q[:, q_head_slice]
    k_i = kv_buffer[0][local_k_slice]
    v_i = kv_buffer[1][local_k_slice]
    _flash_attn_varlen_forward(q_i, k_i, v_i, cu_seqlens_q, cu_seqlens_k, ...)
```

这里的显存曲线和 batch ring 明显不同：

```text
batch_ring:
  常驻/临时 K/V ≈ local_seq_len * all_kv_heads

varlen_llama3:
  KV all_gather buffer ≈ full_seq_len * heads_k_stride
  若 heads_k_stride 小，按 head block 限制峰值；若 heads_k_stride == nheads_k，则接近完整 KV heads all-gather。
```

backward 在 `llama3_flash_attn_varlen_backward()`：

```python
# ring_flash_attn-0.1.7/.../llama3_flash_attn_varlen.py:186-299 的简化版
kv_buffer      = [2, S, heads_k_stride, D]
kv_buffer_copy = [2, S, heads_k_stride, D]
dkv_buffer     = [2, S, heads_k_stride, D]

for each KV head block:
    all_gather K/V block
    _flash_attn_varlen_backward(..., dk=dk_i, dv=dv_i)
    dist.reduce_scatter_tensor(dk_local, dkv_buffer[0], group=process_group)
    dist.reduce_scatter_tensor(dv_local, dkv_buffer[1], group=process_group)
```

所以 varlen llama3 的 backward 通信语义是：

- forward：每个 KV head block 两次 all-gather（K 和 V）；
- backward：每个 KV head block 两次 all-gather（K 和 V）+ 两次 reduce-scatter（dK 和 dV）。

## 7.4 关键细节与误区澄清

> **误区 14：sample packing 默认的 `varlen_llama3` 也是 P2P ring。**
>
> 从源码看不是。Axolotl `VARLEN_LLAMA3` 分支调用的是 `llama3_flash_attn_varlen_func`（`src/axolotl/monkeypatch/ring_attn/patch.py:110-124`），下游实现使用 `AllGatherComm`（`llama3_flash_attn_varlen.py:7`）和 `dist.reduce_scatter_tensor`（`llama3_flash_attn_varlen.py:292-293`）。这和 `batch_ring` 的 `RingComm` P2P isend/irecv 是两种通信模型。

> **误区 15：`heads_k_stride` 越大越省显存。**
>
> schema 描述说“更大的 stride 用更多内存但可能更快”（`config.py:981-985`）。源码原因很直接：`kv_buffer` 的 head 维是 `heads_k_stride`（`llama3_flash_attn_varlen.py:88-93`）。stride 越大，每次 all-gather 的 KV head block 越大，通信次数更少，但 buffer 峰值更高。

> **误区 16：`DATA_PARAMS` 是初始化时固定的。**
>
> 不是。`update_ring_attn_params()` 在每次 forward pre-hook 中根据当前 batch 的 `position_ids` 更新（`sequence_parallel.py:53-55`），下游 `DATA_PARAMS.update(...)` 也是运行期更新（`hf_adapter.py:54-62`）。sample packing 下每个 batch 的 packed 边界可能不同，因此不能只在初始化阶段计算一次。

## 7.5 本章小结

> 💡 **小结**
>
> * sample packing 默认走 `varlen_llama3`，它依赖 `position_ids -> cu_seqlens -> DATA_PARAMS`。
> * 这条路径 forward 用 KV head block all-gather，backward 用 all-gather + reduce-scatter。
> * `heads_k_stride` 是显存 / 通信次数的旋钮：更大通常更快，但 KV buffer 峰值更高。

# 八、显存、性能与通信分析

## 8.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | CP 不切参数；参数节省来自 FSDP / ZeRO / TP / quantization。 |
| optimizer state | ❌ | CP 不改变 optimizer state；仍由 FSDP / ZeRO / optimizer 自身决定。 |
| gradients（参数梯度） | ❌ | 参数梯度同步不属于 CP；CP 只处理序列维中间张量。 |
| input batch tensor | ✅ | pre-hook 把 `[B,S]` 切为 `[B,S/P]`。 |
| hidden / MLP 激活 | ✅ | 模型主体只看到局部 sequence chunk。 |
| Q/K/V 激活 | ✅ | attention projection 后 Q/K/V 都是局部 token。 |
| attention score 矩阵 | ✅ / 原本已由 FA 节省 | FlashAttention 本身不物化完整 attention matrix；ring 继续保持 block 化。 |
| logits（SFT） | ✅ | SFT 不 post-gather outputs，LM head 通常只产生 `[B,S/P,V]`。 |
| logits（GRPO/EBFT） | 部分消失 | `gather_outputs=True` 会 all-gather 多维输出，恢复完整序列 tensor。 |
| batch_ring K/V buffer | ❌ 额外 | 需要 `next_k/next_v` 和 backward block buffers，但规模是局部 chunk。 |
| varlen_llama3 KV buffer | ❌ 额外 | 每个 head stride all-gather 到 `[S, heads_k_stride, D]`，stride 越大越占显存。 |
| save state_dict | ❌ 额外 CPU 内存 | CP 下 `_save()` 会将 state_dict tensor detach 到 CPU，避免 safetensors storage 问题（`core/trainers/base.py:812-823`）。 |

显存收益最大的部分，是每层随 sequence length 线性增长的激活：

```text
无 CP:
  per rank activation ~ O(B * S * hidden) + O(B * S * H * D) + logits

CP size = P:
  per rank activation ~ O(B * S/P * hidden) + O(B * S/P * H * D)
  + ring/all_gather 临时通信 buffer
```

对于超长序列，`S/P` 的下降通常远大于 ring buffer 的增加；但对于较短序列或跨节点慢互联，通信开销可能超过显存收益。

## 8.2 通信开销

按每个 transformer layer 估算：

| 路径 | Forward 通信 | Backward 通信 | group | 频率 |
|---|---|---|---|---|
| `batch_ring` | `P-1` 轮 P2P K + V send/recv | `P-1` 轮 K/V 重放 + `P` 轮 dK/dV P2P | CP group | 每层每 step |
| `varlen_llama3` | 每个 KV head block：K all-gather + V all-gather | 每个 KV head block：K/V all-gather + dK/dV reduce-scatter | CP group | 每层每 step |
| `num_items_in_batch` 修正 | all-reduce AVG | 无 | CP group | 每次 forward，若 batch 带该字段 |
| eval loss correction | all-reduce SUM weighted_loss + all-reduce SUM total_valid | 无 | CP group | eval forward |
| GRPO/EBFT output gather | shape all-gather + tensor all-gather | backward 只 slice，无通信 | CP group | model forward output |
| FSDP save / 参数通信 | FSDP 自己决定 | FSDP 自己决定 | DP shard group | 不属于 CP |

Batch ring 的一个直观时序：

```text
Forward, P=4:
  step0: async send K/V -> compute block0 -> wait
  step1: async send K/V -> compute visible block -> wait
  step2: async send K/V -> compute visible block -> wait
  step3: no next K/V send -> compute visible block

Backward:
  step0..P-1:
    rotate K/V
    compute block backward if visible
    rotate accumulated dK/dV back to owner
```

这就是“通信换显存”：不 all-gather 完整 K/V 常驻，但每层都插入细粒度通信。能否 overlap 取决于异步 P2P/all-gather 和 FlashAttention compute 的相对耗时。源码上 batch ring 会先发起 `send_recv_kv()` 再计算当前 block，最后 `wait()`（`ring_flash_attn.py:26-63`），具备一定 overlap 形态；varlen llama3 则用双 buffer，把下一 head block 的 all-gather 与当前 head block compute 交叠（`llama3_flash_attn_varlen.py:101-116`）。

## 8.3 性能取舍

这套实现的性能取舍可以概括为：

1. **用 CP group 内通信换序列激活显存。** 序列越长、模型 hidden 越大，收益越明显；互联越慢、层数越深，通信越可能成为瓶颈。
2. **用 monkey patch 换模型覆盖面。** 只要模型走 HF FlashAttention 入口，就能接入；但 Transformers 签名变化会破坏 patch。
3. **用 head stride 换 varlen 路径速度。** `heads_k_stride` 增大后通信次数减少，但 all-gather buffer 变大。
4. **用局部 logits 换 SFT 显存收益。** SFT 不 gather output；但 RL 某些路径为了后处理会 gather，收益变小。
5. **用 CPU clone 换保存可靠性。** CP save workaround 避免 safetensors storage pointer 问题，但增加 checkpoint 时 CPU 内存压力。

> 💡 **小结**
>
> * CP 不省参数和 optimizer state，主要省序列激活、Q/K/V、logits。
> * batch ring 与 varlen llama3 的通信模型不同：P2P ring vs all-gather/reduce-scatter。
> * 性能瓶颈通常出现在“每层每 step 的 CP 通信”和“varlen all-gather buffer / reduce-scatter”。

# 九、配置项、边界条件与坑点

## 9.1 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `context_parallel_size > 1` | `validation.py:1508-1561`、`train.py:205-219` | 开启 CP 校验、env、SequenceParallelContextManager | 不在 schema 直接校验整除 world size；拓扑错误可能延迟。 |
| `flash_attention: true` | `validation.py:1517-1520` | CP 必须依赖 HF FlashAttention 入口 | 未开启直接报错；SDPA 不是 Axolotl ring 主路径。 |
| `sample_packing: true` | `validation.py:1571-1577` | 默认 `ring_attn_func=varlen_llama3` | `micro_batch_size > 1` 会报错；通信变成 all-gather/reduce-scatter。 |
| `sample_packing: false` | `validation.py:1571-1577` | 默认 `ring_attn_func=batch_ring` | causal 下 rank 计算量不均衡。 |
| `micro_batch_size` | `validation.py:1522-1526` | sample packing + CP 要求为 1 | 非 sample packing 可以 >1，但长序列显存仍按 B 增长。 |
| `heads_k_stride` | `patch.py:200-202`、`llama3_flash_attn_varlen.py:84-119` | varlen llama3 按 KV heads 分块 all-gather | 必须整除 KV heads；源码中下游 assert `nheads_k % heads_k_stride == 0`。 |
| `ring_attn_func` | `patch.py:186-211` | 选择 patch 分支 | enum 当前只开放 `varlen_llama3` / `batch_ring`。 |
| `dp_shard_size` / `dp_replicate_size` | `utils/distributed.py:319-355` | 决定剩余 world size 如何进 DP mesh | 组合不当会在 parallelism config 校验报错。 |
| `tensor_parallel_size` | `utils/config/__init__.py:137-142`、`distributed.py:330-336` | 从 effective DP size 中除掉 TP | TP + CP + FSDP 属于多维并行，debug 难度高。 |
| `rl: grpo` | `core/builders/rl.py:64-68` | 选择 `AxolotlGRPOSequenceParallelTrainer` | async GRPO 与 sequence_parallel 冲突（`grpo/__init__.py:39-45`）。 |
| `use_liger_loss` + GRPO + CP | `validation.py:721-728` | 直接报错 | 当前不支持 GRPO + SP + Liger。 |
| `save_strategy` / FSDP state dict | `train.py:294-334`、`base.py:812-823` | save 时可能走 FSDP gather / merge，CP 下 clone CPU | 大模型保存可能出现 CPU 内存峰值。 |

## 9.2 最小可用配置与默认行为

最小思路：

```yaml
flash_attention: true
context_parallel_size: 2
# 非 sample packing：默认 ring_attn_func=batch_ring
# sample_packing: true 时：默认 ring_attn_func=varlen_llama3，并要求 micro_batch_size: 1
```

如果启用 sample packing：

```yaml
sample_packing: true
micro_batch_size: 1
context_parallel_size: 4
heads_k_stride: 1   # 更稳妥的低显存默认
```

如果追求 varlen 路径速度，可以增大 `heads_k_stride`，但要接受 KV all-gather buffer 峰值上升。

## 9.3 静默失效与不兼容组合

- `context_parallel_size=1`：`validate_ring_attn_func()` 直接返回，不设置 ring 函数（`validation.py:1565-1566`），后续 `execute_training()` 也不会进入 CP context。
- `ring_attn_func` 留空：不是失效，会根据 sample packing 默认选择。
- `batch_zigzag` / `batch_stripe`：下游 package 有代码，但 Axolotl enum 注释掉，不能当作可用配置。
- `GRPO + async_grpo + CP`：`GRPOStrategy.get_trainer_class()` 明确报错（`core/trainers/grpo/__init__.py:39-45`）。
- `GRPO + Liger loss + CP`：schema before validator 报错（`validation.py:721-728`）。
- `merge_lora`：重新加载配置时显式关闭 `flash_attention` 和 `context_parallel_size`（`cli/merge_lora.py:144-155`），所以不要期待 merge 阶段也跑 CP。

> 💡 **小结**
>
> * 开 CP 的真正最小条件是 `flash_attention: true` + `context_parallel_size > 1` + 已安装 `ring_flash_attn`。
> * `sample_packing` 会把默认通信路径从 batch ring 切到 varlen llama3。
> * 一些字段在文档或下游包里存在，但当前 Axolotl enum / 主路径并未开放。

# 十、测试、示例与覆盖缺口

## 10.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/test_context_parallel_batch_size.py:29-56` | CP 下 batch size 按 `world_size // context_parallel_size` 缩放 | 使用 mock `ring_flash_attn`，覆盖配置/normalize，不覆盖真实 kernel。 |
| `tests/test_loaders.py:181-218` | `_get_parallel_config_kwargs()` 中 TP / CP / DP size 组合 | 证明 `cp_size` 会进入 parallel config kwargs。 |
| `tests/monkeypatch/test_trainer_context_parallel_patch.py:36-66` | HF Trainer CP guard patch 替换且幂等 | 覆盖字符串 patch，不覆盖真实训练。 |
| `tests/e2e/multigpu/patched/test_sp.py:28-73` | 构造 CP=2 的端到端训练配置 | 覆盖意图完整，但测试本身被 skip。 |
| `tests/e2e/multigpu/solo/test_grpo.py:300` / `test_gdpo.py:451` | GRPO/GDPO 配置含 `context_parallel_size=2` | 说明 RL 路径有覆盖入口，但需结合测试运行条件判断。 |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:6-32` | FSDP + TP + CP 示例 | 展示 ND parallelism 推荐组合。 |
| `examples/alst/llama3-8b-fsdp2-alst.yaml:18-59` | 超长上下文 + CP + CCE + FSDP2 示例 | 更接近真实长序列压力场景。 |
| `ring_flash_attn-0.1.7/test/test_ring_flash_attn_func.py:46-92` | batch ring forward/backward 与 full FlashAttention 对齐 | 下游包测试，用 broadcast 后局部 chunk 对比。 |
| `ring_flash_attn-0.1.7/test/test_llama3_flash_attn_varlen_func.py:52-119` | llama3 varlen forward/backward 与 full varlen FlashAttention 对齐 | 下游包测试，覆盖 all-gather/reduce-scatter 语义。 |

## 10.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| Axolotl SP e2e 主测试被 skip | `test_sp.py:102-104` 明确 skip | 主 SFT + ring_flash_attn + Transformers 组合缺少 CI 保护。 |
| ring attention patch 是否恢复 | 未看到恢复测试 | 同进程多模型 / 多测试可能被全局 patch 污染。 |
| `varlen_llama3` 在当前 Transformers 5.x 签名下稳定性 | 只有 patch 逻辑，e2e skipped | 上游签名变化可能运行时报错。 |
| 多机 CP group 性能与正确性 | 未看到专门多机覆盖 | 跨节点 P2P/all-gather 延迟可能严重放大。 |
| 保存 / resume + CP + FSDP2 + sample packing | 只有保存 workaround 源码，缺少针对性测试 | checkpoint storage / CPU clone / resume 可能有边界问题。 |
| `heads_k_stride` 非 1、多 KV head 模型 | 未看到 Axolotl 侧参数化测试 | stride 不整除 KV heads 时下游 assert；大 stride 显存峰值未验证。 |
| `num_items_in_batch` all_reduce AVG 的 loss 缩放 | 未看到单测 | loss 统计与非 CP 可能存在细微差异，源码也 warning。 |
| GRPO + CP + vLLM 路径去重 / broadcast | 有代码路径，覆盖不充分 | 同一 SP group prompt 去重错误会影响 reward / completion 对齐。 |

> 💡 **小结**
>
> * Axolotl 对配置、parallel config、Trainer guard patch 有单测。
> * 最关键的 SFT ring attention e2e 当前被 skip，这是最大的覆盖缺口。
> * 下游 `ring-flash-attn` 自己有 forward/backward 对齐测试，但不能替代 Axolotl 集成测试。

# 十一、局限性与已知优化点

## 11.1 硬约束

1. **必须开启 FlashAttention。** `context_parallel_size > 1` 但 `flash_attention` false 会直接报错（`validation.py:1517-1520`）。
2. **sample packing + CP 要求 `micro_batch_size=1`。** 这是 ring-flash-attn varlen 路径要求（`validation.py:1522-1526`）。
3. **varlen 总 token 数需能被 CP world size 整除。** 下游 `llama3_flash_attn_prepare_cu_seqlens()` assert `total_length % world_size == 0`（`llama3_flash_attn_varlen.py:24-26`）。Axolotl pre-hook 会 pad 到 `local_world_size` 可整除（`sequence_parallel.py:96-133`），这是它满足下游要求的关键。
4. **`heads_k_stride` 必须整除 KV heads。** 下游 forward/backward 都有 `assert nheads_k % heads_k_stride == 0`（`llama3_flash_attn_varlen.py:84-87,182-184`）。
5. **GRPO + async 不兼容 CP。** `GRPOStrategy.get_trainer_class()` 显式禁止（`core/trainers/grpo/__init__.py:39-45`）。
6. **GRPO + Liger loss + CP 不支持。** schema validator 报错（`validation.py:721-728`）。

## 11.2 维护成本

- **Monkey patch 依赖上游签名。** `trainer_context_parallel.py` 和 ring adapter 都通过检查/替换函数签名或源码字符串实现；Transformers 版本升级很容易破坏。
- **attention patch 不恢复。** `SequenceParallelContextManager.__exit__()` 只有 TODO，没有 unpatch（`sequence_parallel.py:238-244`）。
- **全局 DATA_PARAMS。** sample packing 的 `DATA_PARAMS` 是模块级 dict，下游 attention forward 隐式读取；这对并发、多模型、多线程都不是隔离设计。
- **文档和源码有偏差。** 文档说 collator chunking，源码主路径是 forward hook；文档依赖版本写 `>=0.1.4`，`pyproject.toml` optional dependency 已经是 `>=0.1.7`（`pyproject.toml:93-96`）。
- **dtype 硬编码风险。** 下游 batch/varlen ring backward 对 `dq` 使用 `to(torch.bfloat16)`，对非 bf16 训练不够泛化。

## 11.3 性能瓶颈

- **batch ring causal 负载不均。** rank 越靠后，计算的 K/V block 越多，`step <= rank` 导致最后一个 rank 最重。
- **每层通信频繁。** 每个 attention layer 都要执行 CP group 内通信，层数越多通信越密集。
- **varlen llama3 buffer 峰值随 `heads_k_stride` 增长。** 大 stride 减少通信轮数，但 all-gather buffer 更大。
- **GRPO/EBFT output gather 会恢复部分显存压力。** `AllGatherWithGrad` 会把多维输出沿 sequence 维拼回完整长度。
- **save CPU clone 可能产生 CPU 内存峰值。** CP 下 state_dict tensor detach 到 CPU 是保存可靠性 workaround，不是显存优化。

## 11.4 已知优化点

源码中的 TODO 和结构暗示了几个方向：

- `sequence_parallel.py:22-23` 提到 zigzag / stripe patterns 尚未接入。下游 package 已有 `zigzag_ring_flash_attn.py`、`stripe_flash_attn.py`，但 Axolotl enum 未开放。zigzag / stripe 可能改善 causal ring 的负载均衡。
- `sequence_parallel.py:244` TODO unpatch attention / accelerate。更稳妥的实现应提供可恢复 patch 或上下文隔离。
- `core/trainers/grpo/trainer.py:254-257` TODO 提到未来可能用 Accelerate 的 dataloader preparation，加 `dispatch_batches` 和 `slice_fn_for_dispatch`。这可能减少自定义 sampler / dataloader 分支。
- varlen llama3 可以继续优化 `heads_k_stride` 的自动选择：根据 KV heads、显存余量、互联带宽选择 stride，而不是完全交给用户手调。
- 保存阶段可以探索分块 CPU clone 或 safetensors storage 修复，降低大模型 checkpoint 的 CPU 峰值。

> 💡 **小结**
>
> * 当前实现可用但明显依赖 monkey patch 和下游签名，维护成本不低。
> * batch ring 的主要性能问题是 causal 负载不均与每层 P2P 通信。
> * varlen llama3 的主要调优旋钮是 `heads_k_stride`，本质是在 buffer 显存和通信轮数之间取舍。

# 小结与展望

Axolotl 的 `ring-flash-attn` / context parallelism 实现可以用几个关键词概括。

## 关键词一：边界注入

Axolotl 没有把每个模型层都改造成 sequence-parallel module，而是在配置、model forward hook、HF FlashAttention 三个边界注入逻辑。这让它能覆盖大量 Transformers 模型，但代价是 patch 链路变长：配置校验、Accelerate env、DeviceMesh、Trainer guard、attention wrapper 任一环节变动都会影响最终行为。

## 关键词二：局部序列

真正的显存收益来自 `apply_sequence_parallelism()` 把 `[B,S]` 切成 `[B,S/P]`。这让 hidden states、Q/K/V、MLP 激活和 SFT logits 都按局部序列长度增长。FSDP 负责参数维度，CP 负责序列维度，两者解决的是不同显存大头。

## 关键词三：通信补语义

切掉序列后，attention 语义必须靠通信补回来。非 sample packing 的 `batch_ring` 用 P2P K/V ring 和 dK/dV ring；sample packing 默认的 `varlen_llama3` 用 KV head block all-gather 和 dK/dV reduce-scatter。二者都在每层 attention 内部发生，不是 Trainer 外层的普通 all-reduce。

## 关键词四：全局 patch

`RING_ATTN_GROUP`、HF `_flash_attention_forward`、下游 `DATA_PARAMS` 都是进程级或模块级状态。它们让接入路径非常短，却带来测试污染、上游版本耦合、多模型隔离不足等维护风险。

## 关键词五：通信换显存

CP 适合长序列、高激活显存压力、GPU 间互联较快的训练场景。它不适合短序列、跨慢网络频繁通信、或依赖复杂 RL output gather 的场景。与纯 FSDP 相比，它补上了序列维度；与直接 all-gather K/V 相比，batch ring 更省显存但更复杂；与 Ulysses / DeepSpeed SP 等替代方案相比，Axolotl 的优势是贴合 HF FlashAttention 生态，代价是 monkey patch 维护成本。

后续值得继续走读的方向有三个：

1. **CP + FSDP2 + TP 的 ND parallelism**：DeviceMesh 多维组合如何影响参数通信和 attention 通信的交错；
2. **CP + CCE / Liger / GRPO**：局部 logits、局部 loss 与 RL gather 之间如何保持数值一致；
3. **zigzag / stripe 或更负载均衡的 ring backend**：是否能缓解 batch ring causal 路径 rank 计算不均。

> 💡 **最终小结**
>
> * Axolotl 的实现不是“打开一个 ring attention 开关”这么简单，而是一条从 YAML 到 HF attention kernel 的集成链。
> * forward 显存下降靠序列 chunk；forward 语义正确靠 K/V 通信；backward 正确靠 dK/dV 回流。
> * 当前最值得警惕的不是算法原理，而是 patch 全局性、测试缺口、以及 sample packing 默认路径与“ring”直觉不一致。
