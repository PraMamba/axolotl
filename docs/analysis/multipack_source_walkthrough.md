# Axolotl 源码走读：Multipack (Sample Packing) 实现解析

在大模型 SFT 或继续预训练里，很多样本都远短于 `sequence_len`。如果每条样本都独占一个固定长度窗口，GPU 做了大量 padding token 的无效前向；如果直接把多条样本拼到一起，又会引入跨样本 attention 泄漏，模型可能在样本 A 的 token 上看到样本 B 的历史。Axolotl 的 Multipack（Sample Packing）就是围绕这个矛盾展开的：它不是新的并行策略，也不是新的 checkpoint 格式，而是一套把“样本重排、batch fetch、collate、attention mask / unpad patch、Trainer dataloader”串起来的工程集成。

本文不讲 bin packing 的理论最优性，也不展开 FlashAttention / FSDP / DDP 的基础原理；只沿着 Axolotl 当前源码，说明用户配置 `sample_packing: true` 后，训练链路里到底哪些对象变了、哪些状态被 patch、shape 如何变化、通信发生在哪里，以及哪些看似相关的代码其实不在标准主路径上。

# 前言

## 业务 / 工程背景

Multipack 出现在训练数据侧，主要服务于 SFT、继续预训练和部分插件训练路径。它解决的不是参数显存，而是“每步有效 token 利用率”和“padding 带来的激活 / logits / kernel 浪费”。官方文档对它的定位很直接：把多个短序列组合进一个 packed sequence 来提高 GPU 利用率，配置入口是 `sample_packing: true`（`docs/optimizations.qmd:14-19`）。`docs/multipack.qmd:8-10` 进一步说明，在 Flash Attention 路径下，Axolotl 不构造 4D mask，而是拼接序列并告诉 attention 每条序列从哪里开始。

## 核心矛盾

Multipack 背后的工程冲突可以压缩成三句话：

1. **吞吐需要把短样本塞满固定 token 窗口**，否则一个 batch 里大量 padding token 会白白占用计算和显存。
2. **训练语义要求 packed 样本彼此隔离**，否则 causal attention 会跨样本泄漏，loss 可能异常升高。
3. **HuggingFace Trainer / PyTorch DataLoader 原生不认识“一个 batch 由若干个 bin、每个 bin 又由若干样本 index 组成”的嵌套索引结构**，因此 Axolotl 必须 patch DataLoader fetcher，并选择特定 collator 与 attention patch。

## 本文主线

本文按机制而不是文件展开：

- 先看配置如何进入数据准备，为什么要写入 `position_ids` 和 `length`。
- 再看 `MultipackBatchSampler` 如何做装箱、缓存、估算步数和跨 rank 长度同步。
- 接着看 DataLoader patch 与 collator 如何把嵌套 index 变成模型输入张量。
- 然后看 attention patch 如何保证 packed 样本之间不互相看见。
- 最后串起完整主路径，分析 shape、状态、通信、显存收益、配置边界、测试覆盖和维护风险。

## 不展开的内容

本文不讲 FlashAttention varlen kernel 的内部实现，不讲 FSDP / DeepSpeed 参数分片原理，也不讲 LoRA / QLoRA 算法。涉及这些模块时，只讨论 Axolotl 的 Multipack 如何与它们衔接。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/train.py`、`src/axolotl/cli/config.py` | 用户命令入口与 YAML/CLI 配置加载、校验、归一化 |
| `src/axolotl/utils/schemas/config.py`、`src/axolotl/utils/schemas/validation.py` | `sample_packing` 相关字段、默认值、互斥约束和提示 |
| `src/axolotl/utils/data/sft.py` | 标准 / streaming 数据集准备，决定何时调用 packing 预处理 |
| `src/axolotl/utils/trainer.py` | 写入 `position_ids/length`，估算 packed steps 和 packing efficiency |
| `src/axolotl/utils/samplers/multipack.py` | Multipack 装箱算法、batch sampler、跨 rank 长度同步 |
| `src/axolotl/monkeypatch/data/batch_dataset_fetcher.py` | patch PyTorch DataLoader fetcher，使嵌套 index 可被正确取样 |
| `src/axolotl/utils/collators/batching.py` | packed batch collator，拼接样本并构造 attention / position 语义 |
| `src/axolotl/core/builders/causal.py`、`src/axolotl/core/trainers/base.py` | 把配置变成 TrainingArguments、选择 collator、创建 sampler/dataloader |
| `src/axolotl/loaders/patch_manager.py`、`src/axolotl/monkeypatch/multipack.py` | 模型加载前注册 multipack attention/DataLoader patch |
| `src/axolotl/monkeypatch/utils.py` | 根据 attention mask / position ids 生成 `cu_seqlens` 等 varlen 元数据 |

# 一、配置与数据准备：把“少填 padding”变成训练语义

## 1.1 设计哲学与核心问题

Multipack 不是只在 DataLoader 里把 index 混一混就结束了。它必须提前给每条样本补充两类语义：

- `length`：告诉 sampler 每条样本占多少 token，装箱算法才能判断能否放进同一个 bin。
- `position_ids`：每条样本内部从 0 开始计数，多个样本拼接后 position 会出现“归零点”，attention 侧可以据此识别样本边界。

如果只有 `input_ids`，sampler 可以知道长度，但 attention 不知道边界；如果只有 `attention_mask=1/0`，FlashAttention 路径只知道有效 token 与 padding，不知道 packed 序列中哪些 token 属于不同样本。因此 Axolotl 在数据准备阶段把“装箱调度”和“attention 隔离”需要的元数据一起写进去。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：Click 命令入口，接收 config 路径和 CLI override

src/axolotl/cli/config.py
  - load_cfg：读取 YAML、应用 CLI override、validate_config、normalize_config

src/axolotl/utils/schemas/config.py
  - sample_packing / eval_sample_packing / pad_to_sequence_len / multipack_real_batches 等字段

src/axolotl/utils/schemas/validation.py
  - check_eval_packing / hint_sample_packing_padding / check_sample_packing_without_attention 等校验与默认化

src/axolotl/utils/data/sft.py
  - prepare_datasets / _prepare_standard_dataset / _load_raw_datasets：决定标准数据集何时做 packing 预处理

src/axolotl/utils/trainer.py
  - add_position_ids：给样本写入 position_ids 和 length
  - process_datasets_for_packing：过滤无训练 token，并在 sample_packing 下 map add_position_ids
```

## 1.3 主流程拆解

用户最常见入口是：

```text
axolotl train examples/llama-3/lora-1b.yml
  -> src/axolotl/cli/main.py:98 train(...)
    -> src/axolotl/cli/train.py:63 load_cfg(config, **kwargs)
      -> src/axolotl/cli/config.py:249-252 读取 YAML 到 DictDefault
      -> src/axolotl/cli/config.py:308-320 validate_config(...)
      -> src/axolotl/cli/config.py:326-328 prepare_optim_env / normalize_config / normalize_cfg_datasets
    -> src/axolotl/cli/train.py:37-43 load_datasets(...)
      -> src/axolotl/common/datasets.py:58-65 load_tokenizer + prepare_datasets
      -> src/axolotl/utils/data/sft.py:63-65 标准或 streaming 分支
```

示例配置里，`examples/llama-3/lora-1b.yml:15-17` 就是最小直觉入口：

```yaml
sequence_len: 2048
sample_packing: true
eval_sample_packing: true
```

真正改变数据集内容的第一处，不在 CLI，而在 `process_datasets_for_packing()`：

```text
src/axolotl/utils/data/sft.py:346-354
  if not skip_prepare_dataset and not streaming:
      handle_long_seq_in_dataset(...)
      if split == "train" and cfg.sample_packing:
          dataset, _ = process_datasets_for_packing(cfg, dataset, None)
```

进入 `process_datasets_for_packing()` 后，Axolotl 先过滤掉 labels 全是 `-100` 的样本（`src/axolotl/utils/trainer.py:267-306`），避免 packed batch 里塞入完全没有监督信号的样本。随后，如果 `cfg.sample_packing` 为真，就调用 `add_position_ids`：

```text
src/axolotl/utils/trainer.py:365-381
  train_dataset = train_dataset.map(add_position_ids, batched=True, ...)
  if cfg.eval_sample_packing:
      eval_dataset = eval_dataset.map(add_position_ids, ...)
```

`add_position_ids()` 本身很朴素：单样本时写 `position_ids=[0..seq_len-1]` 与 `length=seq_len`；batched map 时对每条子序列分别写入（`src/axolotl/utils/trainer.py:99-135`）。这一步看似只是加两列，但后面 sampler、collator、attention patch 都依赖它。

配置校验层也提前做了一些“防呆”：

- `sample_packing` 字段定义在 `src/axolotl/utils/schemas/config.py:636-640`，描述明确提到 block diagonal attention 与 per sequence `position_ids`。
- `eval_sample_packing` 默认是 `None`（`config.py:664-668`），但当训练开启 sample packing 且没有 eval table 时，会被自动设为 `True`（`validation.py:145-152`）。
- `pad_to_sequence_len` 如果未配置，会在 `sample_packing` 下自动设为 `True`，理由是减少 memory fragmentation / OOM 风险（`validation.py:251-263`）。测试 `tests/utils/schemas/validation/test_default_values.py:10-21` 专门验证了这一点。

配置归一化还会计算全局 batch：`normalize_config()` 先补齐 `gradient_accumulation_steps` / `batch_size`，再根据 `WORLD_SIZE`、`context_parallel_size`、`tensor_parallel_size` 扩大全局 `batch_size`（`src/axolotl/utils/config/__init__.py:112-142`）。这会影响 `calculate_total_num_steps()` 对 packed step 数的估算。

## 1.4 关键细节与误区澄清

> 容易误解点一：`preprocess` 不是 Multipack 训练的必经入口。

`axolotl preprocess` 会设置 `AXOLOTL_IS_PREPROCESS=1` 并调用 `load_datasets()`（`src/axolotl/cli/preprocess.py:111-120`），因此它确实可以预先生成带 `position_ids/length` 的 prepared dataset。但标准训练路径也会在 `_load_raw_datasets()` 中直接调用 `process_datasets_for_packing()`（`sft.py:346-354`）。所以 Multipack 的主路径是 `axolotl train`；`preprocess` 是可选的缓存/稳定性路径，不是训练时必须经过的函数。

> 容易误解点二：`max_packed_sequence_len` 已经不是有效开关。

旧配置项 `max_packed_sequence_len` 还出现在 deprecated schema 中，但 validator 会直接抛 `DeprecationWarning`（`src/axolotl/utils/schemas/deprecated.py:15-32`），测试 `tests/patched/test_validation.py:657-670` 覆盖了这个行为。当前有效容量由 `sequence_len`、`micro_batch_size` 和 `multipack_real_batches` 共同决定，而不是这个旧字段。

> 容易误解点三：`eval_sample_packing` 没写不等于关闭。

当 `sample_packing: true` 且未设置 eval table 时，`eval_sample_packing` 会被自动设为 `True`（`validation.py:145-152`）。如果用户显式设为 `False`，框架还会在默认情况下把 `remove_unused_columns` 设为 `False`，用于处理 train/eval collator 不一致（`validation.py:155-162`）。

## 1.5 本章小结

> 💡 **小结**
>
> * Multipack 的第一个行为变化是数据列增加：`position_ids` 和 `length` 是后续 sampler 与 attention 隔离的桥梁。
> * `sample_packing` 会联动默认开启 `pad_to_sequence_len`，这是为了固定 buffer、减少碎片，不是为了“让序列更短”。
> * `preprocess` 可以提前产出 prepared dataset，但训练主路径也能即时处理；旧的 `max_packed_sequence_len` 已废弃。

# 二、装箱调度：MultipackBatchSampler 如何让每步装更多有效 token

## 2.1 设计哲学与核心问题

有了每条样本长度之后，核心问题变成：怎样把变长序列塞进固定容量的“箱子”（bin）？Axolotl 的设计并不是改 Dataset，而是改 BatchSampler：Dataset 仍然存储单条样本，BatchSampler 输出的是嵌套 index：

```text
普通 batch_sampler:
  [idx0, idx1, idx2, idx3]

Multipack batch_sampler:
  [
    [idx0, idx7, idx9],   # 一个 packed bin
    [idx2, idx4],         # 另一个 packed bin
  ]
```

这样做的好处是：Dataset/tokenization 缓存不必变成“预拼接样本”，每个 epoch 可以根据 sampler 顺序重新装箱；代价是 PyTorch 默认 DataLoader fetcher 不能正确处理这种嵌套 index，后面必须 monkey patch。

## 2.2 源码入口与关键对象

```text
src/axolotl/utils/samplers/multipack.py
  - pack_group：First-Fit Decreasing 风格地把样本放入 bin
  - pack_parallel：按 group 切分后用 ProcessPoolExecutor 并行装箱
  - allocate_sequentially：顺序装箱，保留样本顺序
  - MultipackBatchSampler：对外的 BatchSampler，负责缓存、迭代、长度同步、效率统计

src/axolotl/core/trainers/base.py
  - _create_multipack_sampler：Trainer 里真正创建 MultipackBatchSampler
  - _get_train_sampler / _get_eval_sampler：决定训练/评估是否走 packing sampler

src/axolotl/utils/trainer.py
  - calculate_total_num_steps：训练前估算 packed dataloader 长度和 sample_packing_eff_est
```

## 2.3 主流程拆解

Trainer 创建 dataloader 时会先选择 sampler：

```text
AxolotlTrainer.get_train_dataloader()
  -> _get_train_sampler(...)
    -> use_sample_packing = args.sample_packing and not args.pretraining
    -> base_sampler = RandomSampler(...) 或 SequentialSampler(...)
    -> _create_multipack_sampler(base_sampler, dataset)
```

源码位置是 `src/axolotl/core/trainers/base.py:172-207`。训练集开启 packing 且不是 pretraining trainer 内置路径时，`_create_multipack_sampler()` 会计算两个关键参数（`base.py:145-167`）：

```text
if multipack_real_batches:
    batch_size = per_device_train_batch_size
    batch_max_len = max_seq_length
else:
    batch_size = 1
    batch_max_len = train_batch_size * max_seq_length
```

这就是 Axolotl Multipack 最重要的 shape 分叉：

- 默认在 Flash/Flex/Xformers 路径下，`multipack_real_batches` 会倾向于 `False`（`src/axolotl/core/builders/causal.py:259-273`），于是 sampler 只产出 1 个 bin，但这个 bin 容量是 `micro_batch_size * sequence_len`。
- 如果 `multipack_real_batches=True`，则保持真实 batch 维，每个 bin 容量是 `sequence_len`，一个 batch 里有多个 bin。

装箱算法本身在 `MultipackBatchSampler.generate_batches()`：

```text
src/axolotl/utils/samplers/multipack.py:323-365
  indices = [idx for idx in self.sampler]
  lengths = self.lengths[indices]

  if sequential:
      bins = allocate_sequentially(lengths, rank=0, num_ranks=1)
  else:
      all_bins = pack_parallel(lengths, bin_capacity=batch_max_len, group_size=..., bin_size=...)

  batches = [bins[i:i + batch_size] ...]
```

`pack_group()` 是一个保守的 First-Fit 过程：对每条 sequence，找第一个剩余容量足够、且 bin 内样本数量未超过 `bin_size` 的 bin；找不到就新建 bin（`multipack.py:85-107`）。`pack_parallel()` 按 `group_size` 切块，可用 `ProcessPoolExecutor` 并行处理（`multipack.py:149-188`）。如果用户设了 `sample_packing_sequentially`，则走 `allocate_sequentially()`，它按原顺序 next-fit 装箱，并把第 `rank, rank+n, ...` 个 bin 分给 rank；但注意当前 `generate_batches()` 调用它时传的是 `rank=0, num_ranks=1`（`multipack.py:330-336`），也就是说标准 sampler 主路径没有在这里做真实 distributed rank 切分。

在分布式下，Axolotl 真正做的同步是“长度一致性”：`__len__()` 会多次采样 batch 数，取本 rank 的最小值，再通过 `reduce_and_broadcast()` 收集所有 rank 的数值并广播全局最小值（`multipack.py:445-473`）。底层通信是 gather 到 rank0，再 broadcast 给所有 rank（`src/axolotl/utils/distributed.py:274-296`）。`__iter__()` 之后会用 `_len_across_ranks` 截断本地 batch 列表（`multipack.py:383-393`），避免某些 rank 先耗尽。

训练前步数估算也会构造一个临时 `MultipackBatchSampler`。`calculate_total_num_steps()` 在 `sample_packing` 下，如果没有用户提供 `sample_packing_eff_est`，会创建 sampler、DataLoader，并用 `len(data_loader)` 估算 `total_num_steps`（`src/axolotl/utils/trainer.py:442-516`）。这一步还会通过 `reduce_and_broadcast()` 聚合 packing efficiency，并写回 `cfg.sample_packing_eff_est`（`trainer.py:503-516`）。源码里的 FIXME 明确提醒这里“total_num_steps depends on the agreed on value for sample_packing_eff_est”（`trainer.py:496-497`）。

## 2.4 关键细节与误区澄清

> 容易误解点一：MultipackBatchSampler 不是模型并行，也不创建新的 process group。

它只在估算长度/效率时使用 `reduce_and_broadcast()`，底层是 rank0 gather + broadcast（`distributed.py:237-296`）。训练前向过程中没有因为 Multipack 新增 all-to-all、reduce-scatter 或每层通信。DDP/FSDP 的梯度/参数通信仍由原并行框架处理。

> 容易误解点二：`sample_packing_eff_est` 不是用户必须填的开关。

字段定义在 `config.py:1242-1246`，描述说它通常是训练启动后由 trainer 给出的优化值。源码也显示：如果没有设置，Axolotl 会临时跑 sampler 来估算；如果设置了，则直接用公式估算步数（`trainer.py:446-460`）。

> 容易误解点三：`sample_packing_sequentially` 不等于分布式顺序切 shard。

`allocate_sequentially()` 的签名支持 `rank` 与 `num_ranks`（`multipack.py:193-241`），但标准 `generate_batches()` 固定传 `rank=0, num_ranks=1`（`multipack.py:330-336`）。所以在当前主路径中，它的核心作用是保留样本顺序，而不是在 sampler 内完成 distributed rank mapping。最终 dataloader 的进程分发仍交给 Accelerate / Trainer 准备后的 dataloader 逻辑。

## 2.5 本章小结

> 💡 **小结**
>
> * Axolotl 把 packing 做在 BatchSampler 层：Dataset 保持单样本，Sampler 输出嵌套 index。
> * Flash 类路径常把 `micro_batch_size * sequence_len` 合成一个容量更大的 bin，因此 batch 维可能变成 1。
> * Multipack 的通信只用于同步 dataloader 长度/效率估计，不是训练前向里的新通信维度。

# 三、DataLoader 与 Collator：从嵌套 index 到 packed tensor

## 3.1 设计哲学与核心问题

Sampler 输出嵌套 index 后，PyTorch DataLoader 的默认 `_MapDatasetFetcher` 会把它当成普通 batched index，无法区分“一个 batch 里的多个 bin”和“一个 bin 里的多个样本”。Axolotl 的解决方案是两段式：

1. patch DataLoader fetcher，让它遇到 `[[idx...], [idx...]]` 时按 bin 取样。
2. 用专门的 collator 把每个 bin 内的样本拼接成一条长序列，并把 position / attention 语义保留下来。

这层解决的是数据结构适配问题：上游 sampler 输出的是嵌套 Python index，下游 model 要的是 dense tensor。

## 3.2 源码入口与关键对象

```text
src/axolotl/monkeypatch/data/batch_dataset_fetcher.py
  - _MapDatasetFetcher.fetch：识别 nested batch index，逐 bin 调 dataset.__getitems__
  - apply_multipack_dataloader_patch：替换 PyTorch fetcher 与 worker loop
  - remove_multipack_dataloader_patch：测试/清理用恢复函数

src/axolotl/utils/collators/batching.py
  - BatchSamplerDataCollatorForSeq2Seq：拼接样本，attention_mask 保持 1
  - V2BatchSamplerDataCollatorForSeq2Seq：拼接样本，attention_mask 写成样本编号
  - PretrainingBatchSamplerDataCollatorForSeq2Seq：streaming/pretraining 专用拼接

src/axolotl/core/builders/causal.py
  - build_collator：根据 sample_packing、模型类型、attention backend 选择 collator
```

## 3.3 主流程拆解

DataLoader patch 在模型加载前由 PatchManager 触发。只要 `cfg.sample_packing` 为真，就会调用：

```text
src/axolotl/loaders/patch_manager.py:582-588
  if self.cfg.sample_packing:
      apply_multipack_dataloader_patch()
```

patch 内容是全局替换 PyTorch 内部 fetcher：

```text
src/axolotl/monkeypatch/data/batch_dataset_fetcher.py:45-48
  torch.utils.data._utils.fetch._MapDatasetFetcher = _MapDatasetFetcher
  torch.utils.data.dataloader._utils.fetch._MapDatasetFetcher = _MapDatasetFetcher
```

自定义 fetcher 的关键判断是 `possibly_batched_index[0]` 是否为 list（`batch_dataset_fetcher.py:18-42`）：

```text
if isinstance(possibly_batched_index[0], list):
    for each bin_index_list:
        data[i] = dataset.__getitems__(bin_index_list) or [dataset[idx] ...]
else:
    # 标准 DataLoader 行为
return self.collate_fn(data)
```

collator 负责把 bin 内样本拼起来。`V2BatchSamplerDataCollatorForSeq2Seq` 对 `attention_mask` 做了一个很关键的变换：第 i 条样本的 mask 会乘以 `i+1`（`src/axolotl/utils/collators/batching.py:174-180`）。例如三个样本长度分别为 3、2、4：

```text
原始 attention_mask:
  A: [1,1,1]
  B: [1,1]
  C: [1,1,1,1]

V2 拼接后:
  attention_mask: [1,1,1, 2,2, 3,3,3,3]
  position_ids:   [0,1,2, 0,1, 0,1,2,3]
```

这不是普通二值 mask，而是“样本编号”。后面的 `_get_unpad_data()` 会统计每个编号出现多少次，从而得到每个 packed 子序列的长度。

`BatchSamplerDataCollatorForSeq2Seq` 则把 attention mask 保持为 1（`batching.py:142-148`），更适合那些依赖 position reset 或其他路径识别 packed 边界的模型/后端。`HFCausalTrainerBuilder.build_collator()` 决定用哪一个：在 `sample_packing` train 或 eval 下，若启用 flex attention、模型类型在 `SUPPORTED_MULTIPACK_MODEL_TYPES`、或非 flash 的 llama，就选 V2；否则选旧的 BatchSampler collator（`src/axolotl/core/builders/causal.py:474-520`）。

最后，基础 `DataCollatorForSeq2Seq` 会 pad `labels` 和 `position_ids`，并调用 tokenizer.pad（`batching.py:55-125`）。如果原 feature 没有 `attention_mask`，它会在 tokenizer.pad 后删除自动生成的 attention mask（`batching.py:111-112`）。

测试 `tests/test_packed_dataset.py:92-106` 给出了一个非常直观的 shape 证据：`sequence_len=1024`、`micro_batch_size=8`、`multipack_real_batches=False` 时，train/eval dataloader 的 `input_ids.shape` 都是 `(1, 8192)`。

## 3.4 关键细节与误区澄清

> 容易误解点一：V2 collator 的 `attention_mask` 不是 0/1 mask。

它用 `i+1` 标记同一个 packed row 中第几条原始样本（`batching.py:174-180`）。这正是 `get_max_seqlen_in_batch()` 能按编号统计长度的前提（`src/axolotl/monkeypatch/utils.py:13-22`）。把它当二值 mask 理解，会误判 FlashAttention varlen 元数据来源。

> 容易误解点二：DataLoader patch 是全局生效，不是某个 trainer 实例局部生效。

`apply_multipack_dataloader_patch()` 直接改写 `torch.utils.data` 命名空间，并用 `_IS_PATCHED` 防重复（`batch_dataset_fetcher.py:57-76`）。虽然文件提供 `remove_multipack_dataloader_patch()`（`batch_dataset_fetcher.py:79-96`），但训练主路径没有在结束时自动恢复；测试 `tests/test_packed_batch_sampler.py:110-120` 会手动 finally 恢复。

> 容易误解点三：collator 只是拼接，不负责做 attention kernel。

collator 输出的是 `input_ids/labels/attention_mask/position_ids` 张量；真正阻止跨样本 attention 的逻辑发生在模型 attention mask / FlashAttention unpad 元数据生成处，而不是 collator 本身完成计算隔离。

## 3.5 本章小结

> 💡 **小结**
>
> * Multipack 的 batch 结构是嵌套 index，因此必须 patch DataLoader fetcher。
> * Collator 把多个样本拼成 packed row；V2 collator 用非二值 attention mask 编码样本边界。
> * Flash/Flex 等路径下，常见 shape 是 `[1, micro_batch_size * sequence_len]`，不是传统 `[micro_batch_size, sequence_len]`。

# 四、Attention Patch：跨样本隔离如何接入模型前向

## 4.1 设计哲学与核心问题

Multipack 真正危险的地方不在“拼接”，而在“拼接后是否会串味”。一个 packed row 在张量上是连续 token：

```text
[A0 A1 A2 B0 B1 C0 C1 C2]
```

普通 causal mask 只会阻止看未来，不会阻止 `B0` 看 `A0..A2`。因此 Axolotl 必须把样本边界传给 attention 实现。不同 attention backend / 模型版本的入口不一样，Axolotl 的策略是：

- FlashAttention 路径：patch `_get_unpad_data`，让它从带编号的 `attention_mask` 生成多段 `cu_seqlens`。
- 新版 Transformers masking 路径：让模型从 `position_ids` reset 检测 packed sequence；必要时删除 `attention_mask`。
- 特殊模型：在 PatchManager 中注册模型专用 patch。

## 4.2 源码入口与关键对象

```text
src/axolotl/loaders/patch_manager.py
  - apply_pre_model_load_patches：模型加载前统一注册 patch
  - _apply_multipack_patches：注册 FlashAttention unpad patch 与 DataLoader patch
  - _apply_model_specific_patches：Qwen3.5/Qwen3_Next/Nemotron-H 等特殊 packing patch

src/axolotl/monkeypatch/multipack.py
  - SUPPORTED_MULTIPACK_MODEL_TYPES：允许走 multipack attention patch 的模型类型
  - patch_for_multipack：替换 transformers 或 remote modeling 模块的 _get_unpad_data

src/axolotl/monkeypatch/utils.py
  - get_unpad_data：根据 attention_mask 编号生成 indices/cu_seqlens/max_seqlen
  - get_cu_seqlens_from_pos_ids：根据 position_ids reset 生成 cu_seqlens

src/axolotl/core/trainers/base.py
  - compute_loss：Gemma3/Gemma4 sample packing 下删除 attention_mask
```

## 4.3 主流程拆解

模型加载时，`ModelLoader.load()` 先调用 patch manager：

```text
src/axolotl/loaders/model.py:161-191
  load()
    -> patch_manager.apply_pre_model_load_patches()
    -> _build_model()
    -> patch_manager.apply_post_model_build_patches(...)
    -> _load_adapters()
    -> patch_manager.apply_post_model_load_patches(...)
```

`apply_pre_model_load_patches()` 的顺序里包含 `_apply_multipack_patches()`（`src/axolotl/loaders/patch_manager.py:95-118`）。Multipack patch 分两层：

```text
src/axolotl/loaders/patch_manager.py:552-588
  if model_type in SUPPORTED_MULTIPACK_MODEL_TYPES
     and (flash_attention or flex_attention)
     and sample_packing:
       patch_for_multipack(...)

  if sample_packing:
       apply_multipack_dataloader_patch()
```

`patch_for_multipack()` 的核心是替换 Transformers 的 FlashAttention unpad 函数：

```text
src/axolotl/monkeypatch/multipack.py:72-78
  assert hasattr(transformers.modeling_flash_attention_utils, "_get_unpad_data")
  transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data
```

如果是 remote code 模型，它会先加载 config 和空权重模型，以便 import 对应 `modeling_*` 模块，再替换该模块中的 `_get_unpad_data`（`multipack.py:83-93`）。这说明 patch 发生在模块命名空间，而不是某个模型实例方法上。

`get_unpad_data()` 处理的是 V2 collator 生成的带编号 `attention_mask`：

```text
src/axolotl/monkeypatch/utils.py:25-40
  seqlens_in_batch = get_max_seqlen_in_batch(attention_mask)
  indices = torch.nonzero(attention_mask.flatten()).flatten()
  cu_seqlens = pad(cumsum(seqlens_in_batch), (1, 0))
  return indices, cu_seqlens, max_seqlen
```

如果前面例子的 mask 是 `[1,1,1,2,2,3,3,3,3,0,0]`，则：

```text
seqlens_in_batch = [3, 2, 4]
cu_seqlens       = [0, 3, 5, 9]
indices          = 非零 token 的 flat index
max_seqlen       = 4
```

FlashAttention varlen kernel 据此只在每个 `[cu_seqlens[i], cu_seqlens[i+1])` 子序列内部做 causal attention。

对 Gemma3/Gemma4 这类新版 Transformers masking 系统，Axolotl 走另一条修复：`compute_loss()` 中，如果 sample packing active 且输入同时有 `attention_mask` 与 `position_ids`，就删除 `attention_mask`（`src/axolotl/core/trainers/base.py:417-428`）。配套文档解释了原因：`transformers.masking_utils._preprocess_mask_arguments()` 只有在 `attention_mask is None` 时才会从 `position_ids` reset 检测 packed sequence（`docs/agents/new_model_support.md:90-118`）。

## 4.4 关键细节与误区澄清

> 容易误解点一：Multipack 没有自定义 backward 通信。

当前源码没有为 sample packing 定义自定义 autograd Function，也没有在 attention patch 中加入 all-gather/all-to-all。它改变的是 attention metadata / mask 构造。反向传播仍沿着原 attention kernel 和 PyTorch autograd 走。

> 容易误解点二：`patch_for_multipack` 不是所有 sample_packing 都会触发。

PatchManager 要求模型类型在 `SUPPORTED_MULTIPACK_MODEL_TYPES`，且 `flash_attention` 或 `flex_attention` 为真，且 `sample_packing` 为真（`patch_manager.py:552-558`）。如果用户只开了 `sample_packing` 但没开受支持 attention backend，校验只会 warning：不处理 cross sample decontamination（`validation.py:201-213`）。

> 容易误解点三：Gemma3/Gemma4 删除 attention_mask 是局部训练输入处理，不是全局 DataLoader patch。

这段逻辑只在 `AxolotlTrainer.compute_loss()` 里对当前 `inputs` dict 删除 key（`base.py:422-428`）。它不改变 Dataset，也不改变 collator 类；其目的只是让 Transformers masking_utils 走 position reset 检测。

## 4.5 本章小结

> 💡 **小结**
>
> * Multipack 的语义正确性依赖 attention 边界元数据，不是依赖 loss 后处理。
> * FlashAttention 路径通过替换 `_get_unpad_data` 把非二值 attention mask 转成 `cu_seqlens`。
> * 新版 masking 路径则让 `position_ids` reset 触发 packed sequence 检测，必要时删除 `attention_mask`。

# 五、完整主路径串联

## 5.1 完整调用栈

```text
User: axolotl train examples/llama-3/lora-1b.yml
  │
  ├─ Step 1: 配置加载与校验
  │     ├─ src/axolotl/cli/main.py:98 train
  │     ├─ src/axolotl/cli/train.py:63 load_cfg
  │     ├─ src/axolotl/cli/config.py:308-320 validate_config
  │     └─ src/axolotl/utils/config/__init__.py:112-142 normalize_config
  │
  ├─ Step 2: 数据准备与 step 估算
  │     ├─ src/axolotl/common/datasets.py:58-65 load_datasets
  │     ├─ src/axolotl/utils/data/sft.py:63-65 prepare_datasets
  │     ├─ src/axolotl/utils/data/sft.py:346-354 process_datasets_for_packing
  │     ├─ src/axolotl/utils/trainer.py:99-135 add_position_ids
  │     └─ src/axolotl/utils/trainer.py:442-516 calculate_total_num_steps
  │
  ├─ Step 3: 模型加载前 patch
  │     ├─ src/axolotl/train.py:547-569 setup_model_and_trainer
  │     ├─ src/axolotl/loaders/model.py:161-191 ModelLoader.load
  │     └─ src/axolotl/loaders/patch_manager.py:552-588 _apply_multipack_patches
  │
  ├─ Step 4: Trainer / collator 构建
  │     ├─ src/axolotl/core/builders/causal.py:259-291 写入 TrainingArguments
  │     ├─ src/axolotl/core/builders/causal.py:454-544 build_collator
  │     └─ src/axolotl/core/builders/causal.py:431-439 创建 trainer
  │
  ├─ Step 5: dataloader 与 packed batch
  │     ├─ src/axolotl/core/trainers/base.py:172-207 _get_train_sampler
  │     ├─ src/axolotl/core/trainers/base.py:131-170 _create_multipack_sampler
  │     ├─ src/axolotl/core/trainers/base.py:241-336 _get_dataloader
  │     ├─ src/axolotl/monkeypatch/data/batch_dataset_fetcher.py:18-42 nested fetch
  │     └─ src/axolotl/utils/collators/batching.py:166-196 V2 collate
  │
  └─ Step 6: 训练、保存与清理
        ├─ src/axolotl/train.py:183-229 execute_training -> trainer.train
        ├─ src/axolotl/core/trainers/base.py:365-460 compute_loss 局部输入修正
        ├─ src/axolotl/train.py:632-637 save_trained_model / tokenizer save
        └─ src/axolotl/train.py:638-640 cleanup_distributed
```

## 5.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| 配置加载 | YAML + CLI override | `cfg.sample_packing`、`eval_sample_packing`、`pad_to_sequence_len` 等被校验/默认化 | 无 | 间接影响后续 batch shape | 一次 |
| 数据准备 | tokenized Dataset | 增加 `position_ids`、`length`，过滤超长/无监督样本 | 多进程 map 但非分布式通信 | 决定每步有效 token 密度 | preprocess 或训练前 |
| step 估算 | Dataset + cfg | 临时 sampler、`cfg.sample_packing_eff_est` | `reduce_and_broadcast` | 无直接 GPU 显存收益 | 训练前 |
| patch 注册 | cfg + model_config | 替换 `_get_unpad_data`、DataLoader fetcher | 无 | 避免构造部分大 mask，保证正确 attention | 模型加载前一次 |
| sampler | Dataset length + base sampler | 嵌套 batch index、全 rank 最短长度缓存 | `__len__` 中 gather+broadcast | 减少 padding 浪费 | dataloader 创建 / epoch |
| collator | 嵌套样本列表 | packed tensor，如 `[1, micro*seq]` | 无 | 提高 token slot 利用率；pad_to_sequence_len 固定 buffer | 每 step |
| forward/loss | packed tensor | logits/loss；Gemma3/4 可能删除 attention_mask | 模型并行框架自己的通信 | 激活按 packed slot 计算，非参数节省 | 每 step |
| save/resume | Trainer/model state | 通用 checkpoint / tokenizer / config | 取决于 FSDP/DeepSpeed，不是 Multipack 特有 | 无 Multipack 专用状态 | save step / 结束 |

## 5.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `src/axolotl/cli/preprocess.py` | 会调用 `load_datasets()` 并能生成 packed prepared dataset | 可选，不是 `axolotl train` 必经 | 预处理是缓存/稳定性路径；训练也会即时处理 |
| `trainer_weighted_loss()` / `create_weighted_mask()` (`utils/trainer.py:28-87`) | 代码注释里提到 sample_packing loss | 否，`compute_loss()` 中相关分支被注释（`base.py:369-374`） | 当前 Multipack 不靠加权 loss 保证语义 |
| `allocate_sequentially(rank, num_ranks)` | 函数签名像 distributed rank 分配 | 标准调用固定 `rank=0,num_ranks=1` | 主要用于顺序装箱，不是实际 rank mapping |
| `remove_multipack_dataloader_patch()` | 看起来训练结束会恢复 patch | 训练主路径未调用 | 测试清理用；生产进程内 patch 持续生效 |
| `src/axolotl/core/trainers/grpo/trainer.py` 的 sample_packing 判断 | GRPO dataloader 也有 sample_packing 条件 | 一般被 validation 阻断 | `validation.py:703-705` 禁止 `sample_packing` 与 `rl` 同开 |
| `max_packed_sequence_len` | 名字像容量配置 | 否 | deprecated，设置即抛 warning/异常 |
| save/load 特殊函数 | 目标问题包含保存/加载，容易寻找 Multipack save path | 没有 Multipack 专用 checkpoint | 保存走通用 Trainer；resume 测试主要保护 token 统计 |

## 5.4 本章小结

> 💡 **小结**
>
> * 主路径从配置到数据列、sampler、collator、attention patch 是连续的；缺任何一环都可能变成“只拼接不隔离”。
> * Multipack 的保存/加载没有独立状态，checkpoint 仍是通用 Trainer / PEFT / tokenizer 流程。
> * 很多看似相关的兼容代码是备用或被 validation 阻断的路径，不能当作标准执行链路理解。

# 六、关键数据流 / 状态流 / shape 流程

## 6.1 Tensor shape 变化

以 `sequence_len=1024`、`micro_batch_size=8`、`flash_attention=True`、`multipack_real_batches=False` 为例，源码与测试共同指向这个 shape：

```text
原始 tokenized dataset（逻辑上）:
  sample_i.input_ids: [len_i]
  sample_i.position_ids: [0, 1, ..., len_i-1]
  sample_i.length: len_i

MultipackBatchSampler 输出:
  batch = [ [idx0, idx7, idx9, ...] ]
  # batch_size=1, batch_max_len=micro_batch_size * sequence_len = 8192

DataLoader fetcher 后:
  features = [ [sample0, sample7, sample9, ...] ]

V2 collator 拼接后:
  input_ids:      [1, <=8192] -> pad_to_sequence_len 后 [1, 8192]
  labels:         [1, 8192]
  position_ids:   [1, 8192]，每条样本起点重置为 0
  attention_mask: [1, 8192]，不同样本段为 1、2、3...，padding 为 0

FlashAttention unpad 元数据:
  indices:    [total_nonzero_tokens]
  cu_seqlens: [num_packed_sequences + 1]
  max_seqlen: max(len_i)

模型输出:
  logits: [1, 8192, vocab_size] 或由下游 loss/内核进一步处理
```

`tests/test_packed_dataset.py:92-106` 直接断言了 `(1, 8192)`。`docs/multipack.qmd:54-64` 的示意图也强调 “true bsz of 1” 和 `cu_seqlens`。

如果 `multipack_real_batches=True`，则 shape 更接近：

```text
batch = [ [idx...], [idx...], ... ]   # 多个 bin
input_ids: [micro_batch_size, sequence_len]
```

因此，Multipack 的核心不是“总 slot 一定变少”，而是“同样的 slot 里装更多有效 token”。当 `pad_to_sequence_len=True` 时，每步 buffer 反而被固定到上限，这有利于减少碎片，但也意味着单步峰值不一定比动态 padding 更低。

## 6.2 Rank / Process Group / 通信变化

Multipack 自身不创建新的 rank group。典型 `world_size=4` 下，Axolotl 的 sampler 层更像这样：

```text
rank0: local MultipackBatchSampler -> len estimate = L0
rank1: local MultipackBatchSampler -> len estimate = L1
rank2: local MultipackBatchSampler -> len estimate = L2
rank3: local MultipackBatchSampler -> len estimate = L3

reduce_and_broadcast:
  gather [L0,L1,L2,L3] 到 rank0
  rank0 计算 min_len
  broadcast min_len 给所有 rank

每个 rank:
  batches = batches[:min_len]
```

源码证据是 `MultipackBatchSampler.gather_len_batches()`（`multipack.py:432-443`）与 `reduce_and_broadcast()`（`distributed.py:274-296`）。

训练 step 内新增通信主要不是 Multipack 本身，而是已有框架行为：

- DDP/FSDP 仍会做梯度/参数同步。
- `include_tkps` 开启时，`compute_loss()` 会 all_reduce trainable/total token 统计（`src/axolotl/core/trainers/base.py:376-389`）。这是吞吐统计，不是 sample packing 必需的 attention 通信。
- 如果同时启用 `context_parallel_size>1`，会进入 SequenceParallelContextManager（`src/axolotl/train.py:205-220`），那是另一个特性；validation 还要求 sample_packing + context parallel 时 `micro_batch_size == 1`（`validation.py:1515-1526`）。

## 6.3 状态切换与全局污染面

Multipack 有两类全局状态：

```text
DataLoader patch:
  _ORIGINAL_MAP_DATASET_FETCHER
  _ORIGINAL_WORKER_LOOP
  _IS_PATCHED

Attention patch:
  transformers.modeling_flash_attention_utils._get_unpad_data = axolotl.get_unpad_data
  remote modeling_xxx._get_unpad_data = axolotl.get_unpad_data
```

DataLoader patch 保存旧对象并设置 `_IS_PATCHED=True`（`batch_dataset_fetcher.py:57-76`），可通过 `remove_multipack_dataloader_patch()` 恢复（`batch_dataset_fetcher.py:79-96`）。但训练主路径没有自动恢复。

Attention patch 没有对应的 unpatch 函数；`patch_for_multipack()` 直接改模块函数（`multipack.py:72-78`、`83-93`）。这意味着同一个 Python 进程里，如果后续又加载别的模型，patch 仍然存在。通常 CLI 训练进程生命周期较短，这个风险可接受；但在长生命周期服务或测试套件中，需要特别注意隔离。

## 6.4 本章小结

> 💡 **小结**
>
> * Flash 路径下，Multipack 常把 `micro_batch_size` 合并进 sequence 维，形成 `[1, micro*seq]`。
> * Multipack 自身没有新 process group；它只在长度/效率估算阶段做 gather + broadcast。
> * DataLoader 与 FlashAttention patch 都是进程级全局状态，测试可恢复 DataLoader patch，但 attention patch 没有通用 unpatch。

# 七、核心机制深挖

## 7.1 Monkey Patch：零侵入接入还是维护风险？

### 它解决什么问题

PyTorch DataLoader 和 Transformers FlashAttention unpad 都不是为 Axolotl 的嵌套 packed index / 非二值 attention mask 设计的。要不改上游库源码，又要让主流程工作，最直接的方式就是 monkey patch。

### 为什么不能更简单

如果不 patch DataLoader，就需要把 Dataset 物化成 packed samples，牺牲缓存复用和 epoch 级重排能力；如果不 patch `_get_unpad_data`，FlashAttention 只能看到普通 padding mask，无法从 `[1,1,2,2,3,0]` 得到多段 `cu_seqlens`。

### 源码实现

- DataLoader patch：`apply_multipack_dataloader_patch()` 替换 `_MapDatasetFetcher` 和 worker loop（`batch_dataset_fetcher.py:45-76`）。
- Attention patch：`patch_for_multipack()` 替换 Transformers 或 remote modeling 模块的 `_get_unpad_data`（`multipack.py:69-93`）。
- Patch 注册时机：`PatchManager.apply_pre_model_load_patches()` 在模型构建前调用 `_apply_multipack_patches()`（`patch_manager.py:95-118`、`552-588`）。

### 隐藏假设与副作用

隐藏假设是上游内部路径稳定：比如 `transformers.modeling_flash_attention_utils._get_unpad_data` 必须存在，源码用 assert 检查这一点（`multipack.py:72-76`）。副作用是 patch 进程级生效；训练主路径没有恢复 attention patch。维护风险集中在 Transformers / PyTorch 内部 API 升级。

## 7.2 装箱算法：吞吐收益来自 CPU 调度，不来自模型魔法

`pack_group()` 是近似装箱：把样本按 sampler 顺序传入，依次放入第一个容量足够的 bin（`multipack.py:85-107`）。`pack_parallel()` 用 group 切分和进程池并行（`multipack.py:149-188`）。这意味着 packing efficiency 受三个因素影响：

- 样本长度分布：越多短样本，越容易填满。
- `sample_packing_group_size`：每次可共同考虑的样本越多，装箱空间越大，但 CPU/内存开销也更大。
- `sample_packing_bin_size`：一个 bin 最多容纳多少样本，太小会限制效率，太大可能增加样本边界数量和调度成本。

`MultipackBatchSampler` 缓存 `_batches`（`multipack.py:254-255`、`320-321`），`set_epoch()` 会清空缓存（`multipack.py:305-309`）。这说明它不是每个 step 重新装箱，而是在 dataloader/epoch 粒度生成并复用。

## 7.3 Attention 隔离：同一条 tensor 里的 block diagonal 语义

文档说“with Flash Attention ... true bsz of 1”，但源码里并没有真的构造一个显式 block diagonal 4D mask。隔离来自两种 metadata：

- `attention_mask` 编号 -> `get_unpad_data()` -> `cu_seqlens`（`utils.py:25-40`）。
- `position_ids` reset -> Transformers masking_utils packed sequence 检测；Gemma3/4 需要删掉 `attention_mask` 才能触发（`base.py:417-428`、`docs/agents/new_model_support.md:90-118`）。

这是一种“把 block diagonal 语义交给 attention backend”的设计。好处是避免显式 `[B,1,S,S]` mask 的内存；代价是每个 backend / 模型族都要正确理解这些 metadata。

## 7.4 Streaming / Pretraining：同名 sample_packing，路径不完全一样

标准 SFT 的 packing 发生在 prepared Dataset + Trainer dataloader 中；streaming/pretraining 则在 `wrap_streaming_dataset()` 里提前把一个 buffer 内的数据 encode 成 packed chunks（`src/axolotl/utils/data/streaming.py:179-251`）。当 `cfg.sample_packing` 为真，函数会构造 `PretrainingBatchSamplerDataCollatorForSeq2Seq`，并把 `cfg.micro_batch_size` 改成 1（`streaming.py:185-212`）。

`encode_packed_streaming()` 内部也会调用 `process_pretraining_datasets_for_packing()` 和 `MultipackBatchSampler`（`streaming.py:254-304`），但它输出的是 iterable dataset 里的 packed tensor，而不是训练时再由 Trainer sampler 动态装箱。

还有一个重要差异：SFT streaming 会强制 `multipack_attn=True`，pretraining 才尊重 `pretrain_multipack_attn`（`streaming.py:185-190`）。测试 `tests/test_streaming.py:166-234` 分别覆盖了这两个行为。

## 7.5 本章小结

> 💡 **小结**
>
> * Monkey patch 是 Axolotl Multipack 的关键接入方式，减少侵入但增加上游 API 维护成本。
> * 装箱收益来自 CPU 侧调度和更高有效 token 密度，不是模型参数或 optimizer 的变化。
> * 标准 SFT 与 streaming/pretraining 都叫 sample_packing，但一个在 Trainer dataloader 装箱，一个在 streaming map buffer 中预先产出 packed chunk。

# 八、显存、性能与通信分析

## 8.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---|---|
| 参数 | ❌ | Multipack 不改变模型参数、FSDP/DeepSpeed 分片策略或 LoRA 参数量 |
| optimizer state | ❌ | optimizer state 与参数相关，packing 不减少参数个数 |
| 梯度 | ❌ | 梯度张量形状仍由参数决定 |
| 激活值 | ✅/取决于配置 | 对同样有效 token 数，减少 padding token 的激活浪费；但 `pad_to_sequence_len=True` 会固定每步 slot，上限内存不一定降低 |
| logits | ✅/取决于 slot | 如果减少无效 slot，会减少 logits；但 `[1, micro*seq]` 固定 padding 时，每步 logits slot 仍可能等于上限 |
| 输入 batch tensor | ✅/取决于 packing efficiency | 更少纯 padding；但 packed row 可能固定到 `micro_batch_size * sequence_len` |
| attention mask | ✅（Flash 路径） | 不构造显式 4D block mask，改用 `cu_seqlens` metadata |
| CPU prepared dataset | ❌/略增 | 额外存 `position_ids` 和 `length` 列；streaming buffer 越大 CPU 内存越高 |
| 中间 buffer / 碎片 | ✅/间接 | validation 默认设置 `pad_to_sequence_len=True`，目的是复用固定大小 buffer、减少 fragmentation（`validation.py:251-263`） |

真正的显存大头仍然是参数、optimizer state、激活和 logits。Multipack 不动参数和 optimizer，它改善的是“每个激活/logits slot 对应多少有效训练 token”。因此它更像提高吞吐 / token utilization 的技术，而不是像 FSDP、QLoRA 那样直接降低参数显存。

## 8.2 通信开销

Multipack 自身新增的通信很少：

| 触发位置 | 通信类型 | 频率 | group |
|---|---|---|---|
| `MultipackBatchSampler.__len__()` -> `gather_len_batches()` | gather 到 rank0 + broadcast | dataloader 长度估算时；可能每次 sampler 长度初始化 | 默认 torch distributed world |
| `calculate_total_num_steps()` efficiency 聚合 | gather + broadcast | 训练前估算步数时 | 默认 world |
| `AxolotlTrainer.compute_loss()` tokens/sec 统计 | all_reduce | 每 step（仅 `include_tkps` 且训练中） | 默认 world |
| DDP/FSDP/DeepSpeed | all-reduce / all-gather / reduce-scatter 等 | 原训练框架决定 | 原并行组 |
| Context Parallel（若同时开启） | ring attention 相关通信 | 每层/每 step，取决于 SP 实现 | CP group |

需要强调：Multipack 没有在每层 attention 里引入额外 all-to-all；它只是让 attention kernel 使用不同的 varlen 元数据。真正可能成为瓶颈的是 CPU 端装箱、DataLoader worker、以及 packing buffer 大小，而不是 GPU 间通信。

## 8.3 性能取舍

Multipack 的收益来自三类取舍：

1. **CPU 调度换 GPU 利用率**：`pack_parallel()` 可能用多进程装箱，`group_size` 大时 CPU 开销和内存都增加，但 batch 的有效 token 密度更高。
2. **全局 patch 复杂度换无侵入接入**：不改 PyTorch/Transformers 源码，但依赖内部 API。
3. **固定 buffer 换显存稳定性**：`pad_to_sequence_len=True` 可能牺牲某些动态 padding 场景下的单步最小内存，换来更稳定的 allocator 行为。

从源码看，性能最容易被误判的点是：`sample_packing` 不必然让每步张量更短。默认 Flash 路径常把 batch 合成 `[1, micro*seq]`，它提升的是同样 token slot 内的有效样本填充率。对于样本本来都接近 `sequence_len` 的数据，packing efficiency 提升有限，CPU 调度反而可能成为负担。

## 8.4 本章小结

> 💡 **小结**
>
> * Multipack 主要节省 padding 带来的激活/logits/attention 浪费，不节省参数和 optimizer state。
> * 它新增的分布式通信只在长度/效率同步和统计上，训练前向没有新的 per-layer collective。
> * `pad_to_sequence_len=True` 是稳定显存碎片的选择，不等价于降低每步峰值显存。

# 九、配置项、边界条件与坑点

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `sample_packing` | `validation.py:145-163`、`sft.py:351-354`、`base.py:189-205` | 开启数据列写入、sampler、collator、patch | 若无合适 attention backend，可能无法防跨样本泄漏 |
| `eval_sample_packing` | `validation.py:131-164`、`base.py:221-238` | 评估集也使用 MultipackBatchSampler | eval table / causal lm eval 有冲突；小 eval 集可能 steps 为 0 |
| `pad_to_sequence_len` | `validation.py:251-263`、`causal.py:370-381` | collator pad 到固定长度倍数 | 降低碎片但可能固定较大 buffer |
| `flash_attention` / `flex_attention` / `xformers_attention` | `patch_manager.py:552-588`、`causal.py:259-273` | 决定是否 patch unpad、是否默认 `multipack_real_batches=False` | 依赖后端支持和模型类型；xformers 会先 patch over FA2（`patch_manager.py:253-260`） |
| `sdp_attention` | `validation.py:201-213`、`tests/e2e/patched/test_4d_multipack_llama.py:20-65` | 可走非 Flash 4D/masking 路径 | docs 提醒 `sample_packing + SDPA + bf16` 可能 0 loss（`docs/agents/sft.md:83-87`） |
| `multipack_real_batches` | `base.py:145-154`、`causal.py:265-273` | 控制 shape 是 `[1, micro*seq]` 还是 `[micro, seq]` | 默认值由 attention backend 推导，容易被忽略 |
| `sample_packing_group_size` | `multipack.py:155-158`、`342-349` | 每组参与装箱的样本数量 | 越大可能效率更好但 CPU/内存更高 |
| `sample_packing_bin_size` | `multipack.py:91-97`、`346` | 单个 packed bin 最多样本数 | 太小限制短样本 packing；太大增加边界数 |
| `sample_packing_sequentially` | `multipack.py:330-339` | 顺序 next-fit 装箱 | 不等于 distributed shard；建议配合 curriculum_sampling |
| `sample_packing_mp_start_method` | `multipack.py:163-175` | 控制 multiprocessing context | `fork/spawn/forkserver` 与平台、Numba/PyTorch 交互有关 |
| `pretrain_multipack_attn` | `streaming.py:185-190`、`254-274` | pretraining streaming 是否阻止跨样本 attention | SFT streaming 强制 True；pretraining 可设 False 但 loss 阈值更宽（`tests/e2e/test_llama_pretrain.py:64-71`） |
| `streaming_multipack_buffer_size` | `streaming.py:227-249` | streaming map 的 buffer/batch size | 大 buffer 提高 packing 但占更多 CPU 内存；旧字段会被迁移/报错 |
| `pretrain_multipack_buffer_size` | `validation.py:404-426` | deprecated alias | 与新字段同时设置会报错 |
| `batch_flattening` | `validation.py:939-971` | 非 packing 时的 flatten 优化 | 显式 batch_flattening 与 sample_packing 不兼容；auto 会在 packing 下设 False |
| `rl` / `kto` | `validation.py:701-713` | RLHF 禁止 sample_packing | GRPO trainer 里有残留判断，但标准配置会被 validation 阻断 |
| `s2_attention` | `validation.py:215-223` | shifted-sparse attention | 与 sample_packing 同开直接报错 |
| `context_parallel_size` | `validation.py:1515-1526`、`train.py:205-220` | 与 sequence parallel 组合 | sample_packing 下要求 `micro_batch_size=1`，否则 ring-flash-attn 约束报错 |
| `processor_type` / multimodal | `docs/multimodal.qmd:36-42` | 多模态示例关闭 sample_packing | 文档标注 not yet supported；部分模型需要 `skip_prepare_dataset` |
| `max_packed_sequence_len` | `deprecated.py:15-32` | 无 | 已废弃，设置会抛 DeprecationWarning |

## 9.1 最小可用配置

标准 SFT 最小思路是：

```yaml
sequence_len: 2048
sample_packing: true
flash_attention: true   # 推荐，或选择其他受支持 attention backend
micro_batch_size: 1     # 与 context parallel 组合时必须为 1；普通 packing 可大于 1
```

示例 `examples/llama-3/lora-1b.yml:15-17` 展示了 `sequence_len + sample_packing + eval_sample_packing`。但从 validation 看，仅开 `sample_packing` 而不开任何 attention backend 只会 warning（`validation.py:201-213`），不是硬错误；这也是一个静默风险。

## 9.2 本章小结

> 💡 **小结**
>
> * 配置项不是孤立开关，`sample_packing` 会联动 eval、padding、collator、patch 和 dataloader。
> * 真正影响 shape 的是 `multipack_real_batches`，而它默认由 attention backend 推导。
> * 边界条件集中在 attention backend、RL/多模态、context parallel、streaming buffer 和 deprecated 字段。

# 十、测试、示例与覆盖缺口

## 10.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/test_packed_batch_sampler.py:30-120` | sampler + DataLoader patch + V2 collator | 参数化 batch_size、num_workers、max_seq_length、sequential；验证 shape 和样本不重复 |
| `tests/test_packed_dataset.py:33-106` | Trainer 构建后的 train/eval sampler 与 collator | 验证 `MultipackBatchSampler`、`V2BatchSamplerDataCollatorForSeq2Seq`、`(1,8192)` shape |
| `tests/test_packed_pretraining.py:49-110` | streaming/pretraining packing | 验证 packed streaming 输出 `[1, original_bsz*sequence_len]` 且 attention_mask 暂时不存在 |
| `tests/test_streaming.py:19-78` | streaming buffer 字段迁移和冲突 | 覆盖 deprecated `pretrain_multipack_buffer_size` |
| `tests/test_streaming.py:166-234` | SFT streaming 强制 `multipack_attn=True`，pretraining 尊重配置 | 覆盖 streaming 语义分叉 |
| `tests/patched/test_validation.py:657-709` | deprecated 字段、pad_to_sequence_len warning/autoset | 验证配置默认化 |
| `tests/patched/test_validation.py:968-1023` | eval table 与 eval_sample_packing 冲突 | 验证 eval 边界 |
| `tests/patched/test_validation.py:1248-1260` | s2_attention 与 sample_packing 冲突 | 硬错误路径 |
| `tests/e2e/patched/test_lora_llama_multipack.py:23-69` | LLaMA/SmolLM LoRA + flash + packing 训练 | 端到端训练并检查输出 |
| `tests/e2e/patched/test_4d_multipack_llama.py:20-112` | 非 Flash SDPA / torch attention packing | 端到端覆盖 4D/masking 路径 |
| `tests/e2e/patched/test_model_patches.py:21-92` | multipack monkey patch 是否生效 | Mixtral/Mistral 模型加载；断言 `_get_unpad_data` 被 patch |
| `tests/e2e/multigpu/test_eval.py:23-92` | 多 GPU eval sample packing | accelerate 2 进程评估 loss |
| `tests/e2e/patched/test_resume.py:25-120` | packed resume 与 token 统计 | 检查 resume 后 total_num_tokens 保持一致 |
| `examples/llama-3/lora-1b.yml:15-17` | 常规样例配置 | 推荐 train/eval packing |
| `examples/streaming/README.md:28-50` | streaming packing 参数说明 | 说明 buffer size 与效率/内存取舍 |

## 10.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| 多机多节点 packing 长度同步 | 未在源码中确认有专门多节点测试 | rank 间 batch 数、shuffle、sampler 截断问题可能只在多节点暴露 |
| packing efficiency / 显存收益量化 | 没有基准测试断言 | 性能回退不容易被 CI 捕获 |
| attention patch 自动恢复 | DataLoader 有测试恢复；FlashAttention patch 未见通用恢复测试 | 长生命周期进程或测试污染 |
| 所有 `SUPPORTED_MULTIPACK_MODEL_TYPES` 都端到端覆盖 | 只有部分模型 smoke/e2e | 新模型加入列表后可能只通过加载，不保证 loss 正常 |
| context parallel + sample_packing | `tests/e2e/multigpu/patched/test_sp.py:102-120` 当前 skip | SP 组合路径风险较高，尤其 ring attention upstream 变动 |
| streaming attention_mask 路径 | `tests/test_packed_pretraining.py:106-110` 注释 FIXME | pretraining packed attention mask/unpad/pad 尚有已知缺口 |
| 小数据集 / eval steps 为 0 | 标准路径有 ValueError（`sft.py:105-112`）但组合有限 | 用户小 eval split 容易报错，需要关闭 eval_sample_packing |
| RL trainer 内残留 sample_packing 分支 | validation 禁止 RL + packing | 代码存在但不代表可用；未来放开时需补测试 |
| 保存 / 加载 sampler 状态 | resume 测 token 统计，不保存 packing sampler 状态 | epoch 内精确恢复 sampler 顺序未在源码中确认 |

## 10.3 本章小结

> 💡 **小结**
>
> * 测试较好覆盖了 sampler、collator、patch、单机 e2e 和部分多 GPU eval。
> * 最大缺口在性能/显存量化、多节点、全模型矩阵和 context parallel 组合。
> * 源码中已有 FIXME 指向 streaming pretraining attention_mask 路径，说明这块仍有维护风险。

# 十一、局限性与已知优化点

## 11.1 硬约束

- `sample_packing` 与 `rl` 同开会报错（`validation.py:701-705`），KTO 也显式禁止 packing（`validation.py:708-713`）。
- `sample_packing` 与 `s2_attention` 同开会报错（`validation.py:215-223`）。
- 显式 `batch_flattening=True` 与 `sample_packing` 不兼容；`batch_flattening: auto` 会在 packing 下设为 False（`validation.py:939-971`，测试 `tests/patched/test_validation.py:1356-1370`）。
- `context_parallel_size>1` 时必须 `flash_attention: true`；如果同时 sample_packing 且 `micro_batch_size>1` 会报 ring-flash-attn 约束错误（`validation.py:1515-1526`）。
- 多模态文档仍建议 `sample_packing: false`（`docs/multimodal.qmd:36-42`）。
- `max_packed_sequence_len` 已废弃（`deprecated.py:15-32`）。

## 11.2 维护成本

维护成本集中在三处：

1. **PyTorch DataLoader 内部类 patch**：`_MapDatasetFetcher` 和 worker loop 是私有 API（`batch_dataset_fetcher.py:45-76`）。
2. **Transformers FlashAttention util patch**：`_get_unpad_data` 也是内部函数，源码只用 assert 做 API 存在性检查（`multipack.py:72-76`）。
3. **模型族特殊逻辑扩散**：PatchManager 对 Qwen3_Next、Qwen3.5、Nemotron-H 等有 sample_packing 分支（`patch_manager.py:361-399`），新增模型时容易漏掉 position_ids / mask 语义。

## 11.3 性能瓶颈

- **CPU 装箱**：`pack_parallel()` 虽支持多进程，但会把 lengths 切 group 并在 Python/Numba/进程池之间调度（`multipack.py:149-188`）。极大数据集和大 group size 下，训练前 dataloader 准备可能变慢。
- **固定 pad 上限**：默认 `pad_to_sequence_len=True` 有利于稳定，但对本来较短的 batch，单步 slot 可能固定到 `micro*seq`。
- **全局最短长度截断**：跨 rank 取 min len 保证平衡，但某些 rank 生成的额外 batch 会被截断（`multipack.py:389-393`、`445-473`），有效数据利用可能略降。
- **streaming buffer 内存**：`streaming_multipack_buffer_size` 越大 packing 越好，但文档也说明会增加内存（`examples/streaming/README.md:34-37`）。

## 11.4 已知优化点

源码注释里最明确的优化点有两个：

- `calculate_total_num_steps()` 对 `sample_packing_eff_est` 的一致性还有 FIXME（`src/axolotl/utils/trainer.py:496-497`）。未来可以把 efficiency 估算和 dataloader 长度同步做得更确定，减少启动时重复 sampler 生成。
- streaming pretraining 的 attention mask/unpad 路径有 FIXME：当前用 position id workaround，测试里也注释“add back once we fix packing unpad/pad with attention mask”（`src/axolotl/utils/data/streaming.py:271-273`、`tests/test_packed_pretraining.py:106-110`）。

可考虑的工程优化包括：缓存 lengths 与 packing plan、异步预取下一 epoch packing、在更细粒度上 overlap CPU packing 与 GPU 训练、为 attention patch 提供统一 unpatch/上下文管理、以及增加真实显存/吞吐 benchmark 作为回归保护。

## 11.5 本章小结

> 💡 **小结**
>
> * Multipack 的硬约束主要来自 attention backend、RL/多模态和 context parallel 组合。
> * 最大维护风险是对 PyTorch/Transformers 私有 API 的 monkey patch。
> * 优化方向不是改模型参数，而是减少 CPU 装箱成本、增强 patch 生命周期管理、补齐 streaming attention mask 与性能基准。

# 小结与展望

Axolotl 的 Multipack 实现可以用几个关键词概括。

## 关键词一：数据列契约

`sample_packing` 不是单一开关，而是一份贯穿数据准备、sampler、collator、attention 的契约。`length` 给 sampler，`position_ids` 给 masking 系统，带编号的 `attention_mask` 给 FlashAttention unpad patch。理解这三个字段，比记住文件列表更重要。

## 关键词二：嵌套 BatchSampler

Axolotl 没有把 Dataset 永久改造成 packed samples，而是让 `MultipackBatchSampler` 输出嵌套 index。这保留了数据缓存与 epoch 重排的灵活性，但迫使框架 patch DataLoader fetcher。它是一个典型的工程取舍：用一层运行时适配，换更少的数据物化成本。

## 关键词三：metadata 驱动的 attention 隔离

真正防止跨样本泄漏的不是 loss，也不是显式大 mask，而是 `cu_seqlens` 或 `position_ids` reset 这类 metadata。FlashAttention 路径通过 `_get_unpad_data` patch 得到 varlen 边界；Gemma3/Gemma4 则通过删除 `attention_mask` 让 Transformers 从 `position_ids` 检测 packed sequence。

## 关键词四：通信少、patch 重

Multipack 本身几乎不新增训练 step 内 GPU 间通信；它只在 sampler 长度/效率估算时 gather + broadcast。它的复杂度主要在 patch 生命周期、模型族兼容、attention backend 语义上。因此它不是“通信换显存”，更准确地说是“CPU 调度 + attention metadata + monkey patch 换 token 利用率”。

## 关键词五：吞吐优化而非万能省显存

Multipack 最适合样本长度分布离散、短样本多、padding 浪费严重的 SFT / 继续预训练场景。它不适合已经接近满长的样本集，也不适合当前 validation 禁止的 RLHF/KTO、多模态未支持组合，或对上游私有 API patch 极度敏感的长期服务进程。与 batch flattening、sequence parallel、FSDP 这类方案相比，Multipack 的位置更靠近数据与 attention 语义层：它提高每个 token slot 的有效训练密度，而不是切分参数或切分序列维。

后续值得继续走读的方向有三个：第一，Sequence Parallel 与 sample_packing 组合下 ring attention 如何消费 `position_ids/cu_seqlens`；第二，Qwen3.5 / Qwen3_Next 这类线性 attention 模型为什么需要额外 FLA patch；第三，Cut Cross Entropy / Liger Loss 与 packed logits 如何共同影响显存峰值。把这些拼起来，才能完整理解 Axolotl 在“数据密度、attention kernel、loss kernel、并行策略”之间做的系统级取舍。
