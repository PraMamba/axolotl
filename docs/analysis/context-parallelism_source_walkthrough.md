# Axolotl 源码走读：Context Parallelism (CP) 实现解析

在长上下文微调里，最先撞墙的往往不是参数，而是随序列长度增长的激活、Attention 中间状态和 logits。FSDP 可以把参数、梯度和优化器状态切开，但它并不会自然把一个 128K、500K token 的样本沿序列维度切到多张卡上。Axolotl 的 Context Parallelism（代码和文档里也常叫 Sequence Parallelism，下面统一简称 CP）正是为这个矛盾接入的：让一组 GPU 共同处理同一条序列的不同片段，同时用 ring-flash-attn 在注意力层恢复“全局可见性”。

本文不展开 Ring Attention、FSDP 或 FlashAttention 的外部算法推导，而是顺着 Axolotl 源码回答一个更工程化的问题：用户在 YAML 里写下 `context_parallel_size: 2` 之后，Axolotl 到底在哪些地方改变了训练路径？这些改动如何影响数据分发、进程组、前向 shape、通信、显存、保存以及测试可靠性？

# 前言

## 业务 / 工程背景

CP 出现在 Axolotl 的长上下文训练场景，尤其是 SFT / pretraining / GRPO 等训练链路中。它解决的不是“模型权重太大”这个问题，而是“单个样本序列太长，单卡放不下完整序列上的训练激活和注意力计算”这个问题。文档 `docs/nd_parallelism.qmd:35-42` 对 CP 的定位很直接：输入序列沿 sequence length 切分，但注意力不是局部的，所以需要 ring-flash-attn 在 CP 组内交换 KV。

## 核心矛盾

这个特性背后的核心冲突可以概括成三句话：

1. **FSDP 切参数，不切序列**：FSDP 负责参数、梯度和 optimizer state 的分片，但长序列导致的激活 / logits / attention 显存仍然会随序列长度增长。
2. **切序列会破坏注意力语义**：每个 rank 只拿到局部 token 后，普通 FlashAttention 看不到其他 rank 的 K/V，需要 ring 通信恢复全局上下文。
3. **Trainer / Accelerate 已有 CP 入口，但 Axolotl 选择了自己的执行方式**：Transformers 的 `Trainer.training_step()` 会调用 `accelerator.maybe_context_parallel()`，但 Axolotl 又通过 hook 自己切 batch、patch ring attention，并把 Accelerate 原生 CP 上下文改成 no-op。

## 本文主线

本文按机制而不是按文件展开：

1. 用户配置如何变成真实的 CP 行为；
2. DeviceMesh / process group 如何把 world 切成 CP 组；
3. 为什么需要多层 monkey patch；
4. 一次 forward 中 batch shape 如何切分、attention 如何通信、输出如何收集；
5. GRPO 为什么有单独的数据分发路径；
6. 保存、显存、通信、性能和测试缺口。

## 不展开的内容

本文不讲 Ring Attention 数学原理，不讲 FlashAttention kernel 内部实现，不讲 FSDP/TP 的完整原理，也不评价 ring-flash-attn 上游包的实现质量。所有判断以当前 `/root/axolotl` 源码和本地已安装依赖的可见行为为准；`ring_flash_attn` 包在当前环境未安装，因此其内部 kernel / autograd 细节未在源码中确认。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/utils/schemas/config.py` | 定义 `context_parallel_size`、`ring_attn_func`、`heads_k_stride` 等配置字段 |
| `src/axolotl/utils/schemas/validation.py` | 校验 CP 依赖、默认值、deprecated 字段、ring_attn_func 默认选择 |
| `src/axolotl/utils/trainer.py` | 设置 Accelerate/FSDP/parallelism 环境变量，并 patch Accelerate CP prepare |
| `src/axolotl/utils/distributed.py` | 把 TP/CP/DP/FSDP 配置归一成 `ParallelismConfig` 与 `DeviceMesh` |
| `src/axolotl/loaders/model.py` | 模型加载前构建 parallelism_config/device_mesh |
| `src/axolotl/loaders/patch_manager.py` | 在模型加载前注入 Transformers / Accelerate / FSDP / CP 相关 patch |
| `src/axolotl/utils/ctx_managers/sequence_parallel.py` | CP 主执行入口：forward hook、序列切分、loss 修正、output gather |
| `src/axolotl/monkeypatch/ring_attn/patch.py` | 从 DeviceMesh 取 CP process group，并替换 HF FlashAttention 为 ring_flash_attn |
| `src/axolotl/core/trainers/grpo/trainer.py` | GRPO + CP 的采样、生成、reward/advantage 分发特殊路径 |
| `src/axolotl/core/trainers/base.py` | CP 保存时修复 safetensors storage 指针问题 |

# 一、配置入口：`context_parallel_size` 如何从 YAML 变成行为开关

## 1.1 设计哲学与核心问题

CP 的第一层问题不是“如何通信”，而是“什么时候可以安全打开”。长序列切分依赖 FlashAttention、ring-flash-attn、合法的 world size、正确的数据分发和特定 loss 语义。如果只是把 `context_parallel_size` 当作普通 int 传下去，用户很容易在单卡、无 flash attention、缺少 ring_flash_attn 或 batch size 语义错误的情况下得到静默错误。

Axolotl 的做法是把配置入口拆成三段：

1. schema 暴露用户可写字段；
2. validation 做强依赖检查和默认值归一；
3. config normalize / env setup 把 CP 影响传播到 batch size、Accelerate ParallelismConfig 和后续 patch。

## 1.2 源码入口与关键对象

```text
src/axolotl/utils/schemas/config.py
  - AxolotlInputConfig.context_parallel_size：用户开启 CP 的主配置
  - sequence_parallel_degree：deprecated 兼容字段
  - heads_k_stride / ring_attn_func：传给 ring attention 的调度参数

src/axolotl/utils/schemas/validation.py
  - check_context_parallel_size：默认值、flash_attention、ring_flash_attn、sample_packing 限制
  - validate_ring_attn_func：根据 sample_packing 选择 varlen_llama3 或 batch_ring

src/axolotl/utils/config/__init__.py
  - normalize_config：按 CP/TP 折算 effective world size 与 batch_size
```

## 1.3 主流程拆解

用户入口通常是：

```bash
axolotl train examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml
```

Click CLI 在 `src/axolotl/cli/main.py:78-125` 接收命令，然后通过 `src/axolotl/cli/utils/train.py:157-185` 拼成：

```text
accelerate launch ... -m axolotl.cli.train <config.yaml>
```

真正的训练进程进入 `src/axolotl/cli/train.py:55-91`：

```text
do_cli(config)
  -> load_cfg(config, **kwargs)
  -> do_train(parsed_cfg, parsed_cli_args)
  -> axolotl.train.train(cfg, dataset_meta)
```

配置加载在 `src/axolotl/cli/config.py:250-327` 完成：读 YAML、应用 CLI override、调用 `validate_config()`，然后执行 `prepare_optim_env(cfg)` 和 `normalize_config(cfg)`。

CP 字段本身定义在 `src/axolotl/utils/schemas/config.py:969-991`：

```python
sequence_parallel_degree: int | None = Field(default=None, ...)
context_parallel_size: int | None = Field(default=None, ...)
heads_k_stride: int | None = Field(default=None, ...)
ring_attn_func: RingAttnFunc | None = Field(default=None, ...)
```

校验逻辑集中在 `src/axolotl/utils/schemas/validation.py:1508-1579`：

```text
check_context_parallel_size
  -> sequence_parallel_degree 迁移到 context_parallel_size
  -> 未设置则归一为 1
  -> context_parallel_size > 1 时要求 flash_attention
  -> sample_packing 且 micro_batch_size > 1 报错
  -> patch transformers flash support 标志并 import ring_flash_attn
  -> 打 warning：SP loss 可能与非 SP 略有差异

validate_ring_attn_func
  -> CP 未开启则返回
  -> 用户指定则转 RingAttnFunc enum
  -> 否则 sample_packing=True 选 VARLEN_LLAMA3，反之选 BATCH_RING
```

这里的第一个行为改变点不是 forward hook，而是 `prepare_optim_env()`。它在 `src/axolotl/utils/trainer.py:621-640` 写入 Accelerate 相关环境变量：

```python
if cfg.context_parallel_size and cfg.context_parallel_size > 1:
    os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cfg.context_parallel_size)
    os.environ["ACCELERATE_ALLOW_CP_STANDALONE"] = "true"
    patch_prepare_cp()
os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
```

这一步的副作用是：Accelerate 启动时会看到 CP size；同时 Axolotl 会把 Accelerate 的 `_prepare_cp()` 替换成自己的 no-op 版本，后面第三章会解释为什么。

batch size 的变化在 `src/axolotl/utils/config/__init__.py:112-142`：

```python
effective_world_size = (
    cfg.world_size
    // (cfg.context_parallel_size or 1)
    // (cfg.tensor_parallel_size or 1)
)
cfg.batch_size = cfg.batch_size * effective_world_size
```

直觉上，CP 组内多张卡协作处理同一份 batch，不应该把这些 rank 当作数据并行 rank 计入全局 batch。因此 `world_size=8, cp=4` 时有效 DP 规模只有 2。

## 1.4 关键细节与误区澄清

> 误区一：`sequence_parallel_degree` 和 `context_parallel_size` 是两个并存开关。

不是。`sequence_parallel_degree` 只是 deprecated 兼容字段。源码在 `validation.py:1508-1515` 中明确把它迁移到 `context_parallel_size`，随后 CP 主路径只看 `context_parallel_size`。

> 误区二：只要配置了 `context_parallel_size`，Axolotl 就会自己安装或启用 ring_flash_attn。

不会。`validation.py:1528-1550` 只是尝试 `import ring_flash_attn`，失败则报 ImportError。依赖声明在 `pyproject.toml:91-96` 的 optional extra：`ring-flash-attn = ["flash-attn==2.8.3", "ring-flash-attn>=0.1.7"]`。当前环境中 `ring_flash_attn` 未安装，因此不能在本地确认其内部通信 kernel。

> 误区三：文档中“micro_batch_size must be 1 when using context parallel”是绝对限制。

示例 `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:31-32` 写了 `micro_batch_size: 1  # must be 1 when using context parallel`，但源码强校验更窄：只有 `sample_packing` 且 `micro_batch_size > 1` 才报错（`validation.py:1522-1526`）。非 sample packing 情况下，e2e 测试甚至参数化过 `micro_batch_size=2`（`tests/e2e/multigpu/patched/test_sp.py:106-110`），只是该 e2e 当前被 skip。

## 1.5 本章小结

> 💡 **小结**
>
> * CP 的用户开关是 `context_parallel_size > 1`，`sequence_parallel_degree` 只是兼容入口。
> * 配置阶段已经产生行为副作用：检查依赖、选择 ring attention 后端、写 Accelerate env、折算 effective batch size。
> * batch size 不再按 world size 全量放大，而是按 `world_size / cp_size / tp_size` 放大。
> * CP 开启依赖 `flash_attention: true` 和 optional `ring-flash-attn` 包。

# 二、DeviceMesh 与数据分发：CP 组为什么必须拿到同一份输入

## 2.1 设计哲学与核心问题

CP 不是数据并行。数据并行要求不同 rank 拿不同样本；CP 恰好相反，同一个 CP group 内的 rank 必须拿到同一份样本，然后沿 sequence 维切成不同片段。如果 CP 组内 rank0 处理样本 A 的前半段，rank1 却处理样本 B 的后半段，ring attention 的通信语义就完全错了。

因此第二层核心问题是：如何在 world rank 上构造一个包含 TP/CP/DP/FSDP 维度的逻辑网格，并让 dataloader 按“数据并行维度”而不是“所有 rank”分发数据。

## 2.2 源码入口与关键对象

```text
src/axolotl/utils/distributed.py
  - build_parallelism_config：构造 ParallelismConfig 与 DeviceMesh
  - _get_parallel_config_kwargs：按 world_size 分解 tp/cp/dp_shard/dp_replicate

src/axolotl/loaders/model.py
  - _apply_pre_model_load_setup：模型加载前决定是否使用 parallelism_config
  - _set_parallel_config：调用 build_parallelism_config

/usr/local/lib/python3.12/dist-packages/accelerate/data_loader.py
  - prepare_data_loader：用 torch_device_mesh 调整 process_index / num_processes
```

## 2.3 主流程拆解

模型加载前，`ModelLoader.load()` 会先执行 patch 和并行配置准备：

```text
ModelLoader.load()                         # src/axolotl/loaders/model.py:161-194
  -> patch_manager.apply_pre_model_load_patches()
  -> _apply_pre_model_load_setup()
       -> if fsdp_config / tensor_parallel_size>1 / context_parallel_size>1:
            _set_parallel_config()
              -> build_parallelism_config(cfg)
```

`_apply_pre_model_load_setup()` 在 `src/axolotl/loaders/model.py:196-212` 判断只要存在 FSDP、TP 或 CP，就构建 parallelism config。真正分解 world 的逻辑在 `src/axolotl/utils/distributed.py:299-370`：

```python
if tensor_parallel_size > 1:
    pc_kwargs["tp_size"] = tensor_parallel_size
    remaining_world_size //= tensor_parallel_size

if context_parallel_size > 1:
    pc_kwargs["cp_size"] = context_parallel_size
    remaining_world_size //= context_parallel_size

# 未显式配置 dp_shard/dp_replicate 时，剩余 world_size 默认给 dp_shard
if dp_shard_size is None and dp_replicate_size in (None, 1):
    if remaining_world_size > 1:
        pc_kwargs["dp_shard_size"] = remaining_world_size
```

随后 `ParallelismConfig(**pc_kwargs).build_device_mesh("cuda")` 生成 `DeviceMesh`（`distributed.py:309-315`）。

以 `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:6-19` 为例：

```yaml
dp_shard_size: 2
context_parallel_size: 2
tensor_parallel_size: 2
fsdp_version: 2
```

这意味着单机 8 卡可以理解为：

```text
world_size = 8
TP = 2, CP = 2, DP-shard = 2

逻辑网格约等于：dp_shard × cp × tp = 2 × 2 × 2

数据分发视角：
  同一个 (cp,tp) 小组内 rank 应拿同一份 batch
  不同 dp_shard / dp_replicate 组拿不同 batch
```

这里最容易漏掉的是：数据分发不完全由 Axolotl 自己写。Accelerate 的 `prepare_data_loader()` 会根据 `torch_device_mesh` 调整 `process_index` 和 `num_processes`。本地安装的 Accelerate 1.13.0 中，`/usr/local/lib/python3.12/dist-packages/accelerate/data_loader.py:1119-1155` 写明：

```python
if "tp" in torch_device_mesh.mesh_dim_names:
    submesh_tp_size = torch_device_mesh["tp"].size()
if "cp" in torch_device_mesh.mesh_dim_names:
    submesh_cp_size = torch_device_mesh["cp"].size()
...
process_index = process_index // (submesh_tp_size * submesh_cp_size)
num_processes = submesh_fsdp_size * submesh_dp_size
```

注释也明确说 “for CP the same as TP applies”。这就是 CP 组内 rank 拿同一份输入的关键：不是 collator 切 batch，而是 dataloader 的分发视角把 CP/TP 维度从数据并行 rank 中剥离。

GRPO 是例外：`AxolotlGRPOSequenceParallelTrainer._prepare_dataloader()` 在 `src/axolotl/core/trainers/grpo/trainer.py:254-262` 中 CP 开启时直接返回未 `accelerator.prepare_data_loader()` 的 dataloader，因此它必须自己实现 sampler；第五章单独讲。

## 2.4 关键细节与误区澄清

> 误区四：文档说“data collator handles chunking”，所以切分发生在 collator。

`docs/sequence_parallelism.qmd:40-45` 确实写了 “data collator handles the chunking”。但当前源码主路径不是这样：SFT 主路径中，数据分发依赖 Accelerate 的 mesh-aware dataloader；真正沿 sequence 维切 tensor 发生在 `SequenceParallelContextManager` 的 forward pre-hook（`sequence_parallel.py:255-288`）。如果文档和源码不一致，以源码为准。

> 误区五：`context_parallel_size` 只影响 forward，不影响 dataloader。

不对。只要 `torch_device_mesh` 进入 Accelerate dataloader 准备流程，CP size 就会影响 `process_index` / `num_processes`，从而影响每个 rank 拿到哪一份 batch。否则 CP 组内 rank 会处理不同样本，注意力通信语义错误。

> 误区六：world size 不能整除 CP size 会在 schema 阶段报错。

源码中 `validation.py` 没有直接做 world size divisibility 校验。`_get_parallel_config_kwargs()` 用整数除法构造 kwargs，最终合法性主要依赖 Accelerate `ParallelismConfig` 的 total size 校验和后续 mesh 构建。也就是说，不合法拓扑更可能在初始化阶段报错，而不是 YAML schema 阶段。

## 2.5 本章小结

> 💡 **小结**
>
> * CP 组内 rank 必须拿同一份 batch，不同 rank 只处理不同 sequence chunk。
> * Axolotl 通过 `ParallelismConfig` / `DeviceMesh` 表达 CP/TP/DP/FSDP 维度。
> * SFT 数据分发依赖 Accelerate mesh-aware dataloader；GRPO 因为绕开 prepare_data_loader，需要自定义 sampler。
> * “collator 切分”不是当前主路径，真正切 tensor 的地方是 forward pre-hook。

# 三、Monkey Patch 注入：为什么 CP 不是一个纯配置开关

## 3.1 设计哲学与核心问题

理想情况下，CP 应该只是传给 Accelerate / PyTorch 的一个并行配置。但 Axolotl 的实现面对三个现实约束：

1. Transformers Trainer 原生 CP 检查偏向 SDPA，而 Axolotl 要用 FlashAttention + ring_flash_attn；
2. Accelerate 原生 CP 会用 PyTorch DTensor experimental context_parallel 切 buffer，而 Axolotl 要用自己的 hook + ring attention；
3. ring_flash_attn 需要替换 HuggingFace FlashAttention 调用点，把 process group 和 varlen 参数塞进去。

所以 CP 实现不是单点开关，而是一组按时机安装的 monkey patch。

## 3.2 源码入口与关键对象

```text
src/axolotl/loaders/patch_manager.py
  - apply_pre_model_load_patches：模型加载前安装 patch
  - _apply_transformers_patches：Trainer loss + CP guard patch
  - _apply_fsdp_patches：Accelerate ParallelismConfig patch

src/axolotl/monkeypatch/transformers/trainer_context_parallel.py
  - patch_prepare_context_parallel_inputs：放宽 HF Trainer 的 sdpa-only guard

src/axolotl/monkeypatch/accelerate/parallelism_config.py
  - patch_parallelism_config：允许 pure CP / 修正 is_fsdp2 判定
  - patch_prepare_cp：把 Accelerator._prepare_cp 改成 no-op CP context

src/axolotl/monkeypatch/ring_attn/patch.py
  - register_ring_attn_from_device_mesh：注册 CP group，替换 HF flash attention
```

## 3.3 主流程拆解

PatchManager 的安装时机在模型加载前：`src/axolotl/loaders/model.py:168-176` 先执行 `apply_pre_model_load_patches()`，再 build model。

`PatchManager.apply_pre_model_load_patches()` 在 `src/axolotl/loaders/patch_manager.py:95-122` 会依次应用 Transformers、FlashAttention、FSDP、adapter、model-specific 等 patch。和 CP 直接相关的有两处。

第一处是 Transformers Trainer patch（`patch_manager.py:135-149`）：

```python
patch_evaluation_loop()
patch_maybe_log_save_evaluate()

if self.cfg.context_parallel_size > 1:
    patch_prepare_context_parallel_inputs()
```

`trainer_context_parallel.py:15-69` 的策略不是重写整个 Trainer，而是拿到 `Trainer._prepare_context_parallel_inputs` 的源码字符串，把：

```python
if model.config._attn_implementation != "sdpa":
```

替换成允许 `sdpa` 或 `flash_attention_2` 的 guard，然后 `exec()` 成新函数并挂回 `Trainer._prepare_context_parallel_inputs`。这是一种源码文本级 patch，测试 `tests/monkeypatch/test_trainer_context_parallel_patch.py:36-66` 覆盖了 guard 被替换和幂等性。

第二处是 Accelerate patch（`patch_manager.py:270-286`）：

```python
if self.cfg.context_parallel_size > 1 or (self.cfg.fsdp_config and str(self.cfg.fsdp_version) == "2"):
    patch_parallelism_config()
```

`src/axolotl/monkeypatch/accelerate/parallelism_config.py:11-59` 替换 `ParallelismConfig._validate_accelerator`，其中 `ACCELERATE_ALLOW_CP_STANDALONE=true` 时允许 pure CP（`cp_size > 1` 且 `dp_shard_size <= 1`）。同文件 `:80-98` 的 `patch_prepare_cp()` 更关键：

```python
def patched_prepare_cp(self, *args):
    if self.parallelism_config.cp_backend == "deepspeed":
        return args

    @contextlib.contextmanager
    def _noop_cp_context(...):
        yield

    self._cp_context = _noop_cp_context
    return args

Accelerator._prepare_cp = patched_prepare_cp
```

对比本地 Accelerate 1.13.0 原生 `_prepare_cp()`：它会把 `_cp_context` 设成 `torch.distributed.tensor.experimental.context_parallel`，并给 module 挂 context_parallel hooks（`/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py:1657-1670`）。Axolotl 把这条路径换成 no-op，是为了避免 Accelerate 原生 CP 再切一次 buffer。

最后，真正的 ring attention patch 在训练执行期由 `SequenceParallelContextManager.__init__()` 触发。它调用 `register_ring_attn_from_device_mesh()`（`sequence_parallel.py:207-253`），后者在 `src/axolotl/monkeypatch/ring_attn/patch.py:135-212` 中：

1. 从 `device_mesh[("cp",)]` 取 CP submesh；
2. `sequence_mesh.get_group()` 得到 CP process group；
3. 写入模块级全局变量 `RING_ATTN_GROUP`；
4. 根据 `ring_attn_func` 替换 HF FlashAttention：
   - `VARLEN_LLAMA3`：patch `ring_flash_attn.adapters.hf_adapter.create_ring_flash_attention_forward` 并调用上游 `substitute_hf_flash_attn()`；
   - `BATCH_RING`：调用 Axolotl 自己的 `adapters/batch.py:156-196` 替换 `transformers.modeling_flash_attention_utils._flash_attention_forward`。

## 3.4 关键细节与误区澄清

> 误区七：Transformers `Trainer._prepare_context_parallel_inputs()` 是 Axolotl CP 的切分主路径。

不是。HF Trainer 的 `training_step()` 确实会调用 `_prepare_context_parallel_inputs()`，随后进入 `accelerator.maybe_context_parallel()`；但 Axolotl 在 `utils/trainer.py:632-638` 调用 `patch_prepare_cp()`，把 Accelerate `_cp_context` 改成 no-op。也就是说，HF/Accelerate 这层主要保留 position_ids / shift_labels / guard 兼容逻辑，真正切分发生在 Axolotl model forward pre-hook。

> 误区八：patch 是局部的，退出 context 后会恢复。

不是完全局部。`SequenceParallelContextManager.__exit__()` 只移除 model forward hooks（`sequence_parallel.py:238-245`），并明确 TODO：未 un-patch attention 和 accelerate functions。`RING_ATTN_GROUP` 是 `ring_attn/patch.py:34-47` 的模块级全局状态，HF flash attention 替换也是模块级生效。这在单进程多模型、测试隔离、长生命周期服务中有污染风险。

> 误区九：`ring_attn_func` 只在配置层出现，不影响实际实现。

不是。`validate_ring_attn_func()` 选择 enum 后，`SequenceParallelContextManager` 把它传给 `register_ring_attn_from_device_mesh()`；该函数按 `VARLEN_LLAMA3` / `BATCH_RING` 走完全不同的 patch 路径（`ring_attn/patch.py:186-212`）。不过 `apply_sequence_parallelism()` 的 docstring 也承认该参数在 batch slicing 里“Currently unused”（`sequence_parallel.py:42-43`）。

## 3.5 本章小结

> 💡 **小结**
>
> * Axolotl CP 依赖三类 patch：Trainer guard/loss、Accelerate parallelism/no-op CP、ring_flash_attn attention 替换。
> * Accelerate 原生 CP context 被 no-op 化，避免和 Axolotl 自己的 hook 切分重复。
> * ring attention patch 是模块级替换，context 退出时不会恢复，这是维护风险。
> * `ring_attn_func` 决定 attention 替换路径，但不决定 batch slicing 逻辑。

# 四、Forward 主路径：序列切分、ring attention 与输出收集

## 4.1 设计哲学与核心问题

进入训练 step 后，CP 要同时满足两个目标：

1. 每张卡只保留本地 sequence chunk，降低激活 / logits 显存；
2. attention 层仍能看到全序列 K/V，保持训练语义。

Axolotl 的切入点非常克制：不改模型源码，不改大部分 Trainer 逻辑，而是在模型 forward 前后挂 hook。pre-hook 改输入；attention 调用点通过 monkey patch 通信；post-hook 只在部分 RL 路径收集输出。

## 4.2 源码入口与关键对象

```text
src/axolotl/train.py
  - execute_training：训练前进入 SequenceParallelContextManager

src/axolotl/utils/ctx_managers/sequence_parallel.py
  - SequenceParallelContextManager.__enter__/__exit__：注册/移除 hooks
  - apply_sequence_parallelism：pad、position_ids、按 sequence 维切 batch
  - _gather_outputs：可选 all-gather 输出
  - AllGatherWithGrad：前向 all_gather，反向切片梯度

src/axolotl/monkeypatch/ring_attn/patch.py
  - update_ring_attn_params：sample packing 下更新 cu_seqlens
```

## 4.3 主流程拆解

训练执行入口在 `src/axolotl/train.py:183-227`：

```text
execute_training(cfg, trainer, resume_from_checkpoint)
  -> with ExitStack()
      -> if cfg.context_parallel_size > 1:
           enter SequenceParallelContextManager(...)
      -> trainer.train(resume_from_checkpoint=...)
```

`SequenceParallelContextManager` 初始化时会注册 ring attention，并记录本 rank 在 CP group 内的位置（`sequence_parallel.py:207-231`）：

```python
self.process_group = get_ring_attn_group()
self.local_rank = dist.get_rank(self.process_group)
self.local_world_size = dist.get_world_size(self.process_group)
self.apply_sequence_parallelism = functools.partial(
    apply_sequence_parallelism,
    local_rank=self.local_rank,
    local_world_size=self.local_world_size,
    ...
)
```

进入 context 后，`_register_model_hooks()` 给每个 model 注册 forward pre-hook 和 eval loss correction hook；只有 `gather_outputs=True` 时再注册 output gather hook（`sequence_parallel.py:343-357`）。`gather_outputs` 在 `train.py:217` 只对 `RLType.GRPO` 和 `RLType.EBFT` 打开。

一次标准 SFT forward 可以抽象为：

```text
HF Trainer.training_step
  -> _prepare_context_parallel_inputs(...)       # HF/Accelerate CP 兼容层，Axolotl 下 no-op 切分
  -> with accelerator.maybe_context_parallel():  # _cp_context 已被 Axolotl patch 成 no-op
      -> model(**inputs)
          -> Axolotl pre-hook: apply_sequence_parallelism(kwargs)
          -> model forward
              -> patched HF FlashAttention
                  -> ring_flash_attn(..., group=cp_group)
          -> eval loss correction hook（eval 时）
          -> optional output gather hook（GRPO/EBFT）
```

`apply_sequence_parallelism()` 的核心逻辑在 `sequence_parallel.py:51-167`：

```python
batch_size, original_seq_len = batch["input_ids"].shape

# position_ids: 有则更新 ring attn varlen 参数，否则创建 [0..seq-1]
if batch.get("position_ids") is not None and batch_size == 1:
    update_ring_attn_params(position_ids=batch["position_ids"])
else:
    batch["position_ids"] = torch.arange(...).expand(batch_size, -1)

# pad 到可被 CP group 切分的长度
if total_seq_len % divisor != 0:
    pad labels 用 -100，其他 tensor 用 0

# 沿 dim=1 切分 input_ids / labels / attention_mask / position_ids 等二维以上 tensor
batch[key] = batch[key].chunk(local_world_size, dim=1)[local_rank].contiguous()
```

shape 直觉如下：

```text
CP 前，每个 CP group rank 拿到同一份 batch：
  input_ids      [B, S]
  attention_mask [B, S]
  labels         [B, S]
  position_ids   [B, S]

pre-hook 后，第 i 个 CP rank：
  input_ids      [B, S / CP]
  attention_mask [B, S / CP]
  labels         [B, S / CP]
  position_ids   [B, S / CP]

attention 层：
  Q 本地 chunk
  K/V 通过 ring_flash_attn 在 CP group 内环形交换
  output         [B, S / CP, hidden]

SFT 默认：
  loss 在本地 chunk 上计算
  通过 Trainer / DDP/FSDP 梯度同步进入优化

GRPO/EBFT gather_outputs=True：
  output tensor all_gather 回 [B, S, ...]
```

对于 sample packing，`update_ring_attn_params()` 会根据 `position_ids` 计算 cu_seqlens，并调用上游 `update_ring_flash_attn_params(cu_seqlens, group)`（`ring_attn/patch.py:214-226`）。这说明 varlen path 的 packed sample 边界不是靠 attention_mask 传给 kernel，而是靠 position_ids 推导出的 cu_seqlens 全局状态。

输出收集由 `AllGatherWithGrad` 实现（`sequence_parallel.py:368-444`）：

```python
# forward
all_shapes = all_gather(local_shape)
gathered = all_gather(input_tensor)
result = torch.cat(gathered, dim=1)

# backward
offset = sum(seq_lens[:rank])
grad_slice = grad_output[:, offset: offset + seq_lens[rank]].contiguous()
```

它的反向不是 reduce-scatter，而是从 full gradient 中取回本 rank 原本那段序列梯度。对于“前向 concat 只是重建序列维”的语义，这是合理的。

## 4.4 关键细节与误区澄清

> 误区十：CP 总会在 forward 后 all-gather logits / hidden states。

不会。`SequenceParallelContextManager` 的 `gather_outputs` 由 `train.py:217` 控制，只在 GRPO / EBFT 为 true。SFT 主路径不会 gather 模型输出，这也是显存收益保留的关键。否则每步 logits 恢复成完整 `[B, S, vocab]`，长序列下显存收益会被吃掉。

> 误区十一：`num_items_in_batch` 在 CP 下只是普通 token count。

源码做了特殊处理。切分后如果 batch 里有 `num_items_in_batch`，`apply_sequence_parallelism()` 会统计本地 valid token，并在 CP group 内 `all_reduce(..., AVG)`，再乘 `gradient_accumulation_steps`（`sequence_parallel.py:150-165`）。注释说明不用 SUM，因为 SUM 会让 loss 被过度缩放。这是一个框架语义修正，不是 ring attention 算法本身。

> 误区十二：`logits_to_keep` 的切片逻辑已经完整覆盖。

源码里有处理 `logits_to_keep` int 的分支（`sequence_parallel.py:65-95`），也有看似切 1D mask 的分支（`:145-148`）。但由于循环一开始会跳过 `dim() <= 1` 的 tensor（`:136-138`），1D `logits_to_keep` mask 的切片分支按当前代码不可达。这可能依赖下游模型接受全局 mask，也可能是待修复风险；当前测试未看到专门覆盖这个细节。

## 4.5 本章小结

> 💡 **小结**
>
> * CP forward 的核心是 model pre-hook 切输入，而不是 collator 或 Accelerate 原生 CP 切 buffer。
> * attention 正确性依赖 ring_flash_attn 替换后的 FlashAttention 调用点。
> * SFT 默认不 all-gather 输出，GRPO/EBFT 才通过 `AllGatherWithGrad` 恢复完整序列输出。
> * loss token count、eval loss、`logits_to_keep` 都有 CP 特殊语义，不能按普通 Trainer 路径理解。

# 五、GRPO 的特殊路径：为什么 RL 不能只复用 SFT 的 CP hook

## 5.1 设计哲学与核心问题

GRPO 的训练数据流比 SFT 更复杂：同一个 prompt 要生成多个 completion，reward 需要跨生成样本归一，vLLM 可能只在主进程生成，再广播回所有 rank。CP 又要求同一 CP group 内 rank 拿同一份 prompt。普通 dataloader + hook 只能解决“模型 forward 前切 sequence”，不能解决“采样和生成阶段哪些 rank 拿哪些 prompt”。

因此 Axolotl 给 GRPO 单独写了 sequence-parallel trainer 和 sampler。

## 5.2 源码入口与关键对象

```text
src/axolotl/core/builders/rl.py
  - HFRLTrainerBuilder：context_parallel_size > 1 时选择 sequence_parallel trainer

src/axolotl/core/trainers/grpo/__init__.py
  - GRPOStrategy.get_trainer_class：禁止 sequence_parallel + async_grpo 同时开启
  - set_training_args_kwargs：把 context_parallel_size 传入 GRPOConfig

src/axolotl/core/trainers/grpo/sampler.py
  - SequenceParallelRepeatRandomSampler：让 CP group 内 rank 拿同一批 indices

src/axolotl/core/trainers/grpo/trainer.py
  - AxolotlGRPOSequenceParallelTrainer：CP 下自定义 dataloader / vLLM slicing / advantage slicing
```

## 5.3 主流程拆解

RL builder 在 `src/axolotl/core/builders/rl.py:54-69` 中：

```python
trainer_cls = GRPOStrategy.get_trainer_class(
    sequence_parallel=self.cfg.context_parallel_size > 1,
    async_grpo=async_grpo,
)
```

`GRPOStrategy.get_trainer_class()` 在 `src/axolotl/core/trainers/grpo/__init__.py:29-48` 中明确禁止 `sequence_parallel and async_grpo`，测试 `tests/core/test_async_grpo.py:62-94` 覆盖了这个冲突。

GRPO sampler 的设计直接写在 docstring 里：同一 SP group 内 GPU 收到相同数据，不同 SP group 收不同数据（`src/axolotl/core/trainers/grpo/sampler.py:13-39`）。实现上它用：

```python
self.num_sp_groups = world_size // context_parallel_size
self.sp_group_id = rank // context_parallel_size
```

然后每轮从全局 indices 中按 `sp_group_id` 取对应 batch（`sampler.py:128-150`）。这相当于在 GRPO 自己的 sampler 层复刻了“CP/TP rank 不计入数据并行 rank”的原则。

`AxolotlGRPOSequenceParallelTrainer._prepare_dataloader()` 更关键：

```python
# src/axolotl/core/trainers/grpo/trainer.py:254-262
if self.args.context_parallel_size > 1:
    return dataloader
return self.accelerator.prepare_data_loader(dataloader)
```

也就是说 GRPO + CP 不走 Accelerate prepare_data_loader，而是完全信任自己的 sampler。

生成阶段也有 CP 特殊切片。vLLM path 中，所有 prompt 先 gather 到主进程（`trainer.py:328-330`），主进程只从每个 CP group 的 leader rank 取一份 prompt（`:331-353`），生成 completion 后 broadcast 给所有进程（`:379-380`），每个 CP group 再拿回相同 slice（`:382-395`）。reward / advantage 最后也按同样的 SP group slice 保持一致（`:611-630`）。

## 5.4 关键细节与误区澄清

> 误区十三：GRPO + CP 只是 SFT CP 外加 reward loss。

不是。GRPO + CP 要改 sampler、dataloader、vLLM prompt 去重、completion broadcast、advantage slicing。否则 CP group 内 rank 可能生成不同 completion 或 reward 对不齐。

> 误区十四：async GRPO 可以和 CP 自然组合。

源码明确拒绝。`GRPOStrategy.get_trainer_class()` 在 `grpo/__init__.py:39-43` 报错，提示禁用 `context_parallel_size > 1` 或 async prefetch / data producer。测试 `test_async_grpo.py:65-71` 覆盖了这个行为。

> 误区十五：GRPO 仍依赖 Accelerate mesh-aware dataloader。

CP 开启时，GRPO dataloader 明确不调用 `accelerator.prepare_data_loader()`（`trainer.py:254-262`），因此正确数据分发依赖 `SequenceParallelRepeatRandomSampler`，不是 Accelerate mesh slicing。

## 5.5 本章小结

> 💡 **小结**
>
> * GRPO 的 CP 复杂性主要在数据和生成阶段，而不只是 forward hook。
> * CP group 内 prompt/completion/reward/advantage 必须一致，因此 sampler 和 slicing 都要 CP-aware。
> * async GRPO 与 CP 当前互斥，这是源码级硬约束。
> * vLLM path 通过 group leader 去重 + broadcast 降低重复生成。

# 六、保存与恢复：CP 为什么会影响 safetensors storage

## 6.1 设计哲学与核心问题

CP 的 forward hook 会临时切分输入，ring attention / eval loss correction 也会在评估阶段改变一些 tensor 生命周期。训练结束保存时，最怕的是 state_dict tensor 的 storage 指针不再满足 safetensors 写入要求。Axolotl 没有为 CP 写一套独立 checkpoint 格式，而是在 Trainer `_save()` 中做了保守修复。

## 6.2 源码入口与关键对象

```text
src/axolotl/core/trainers/base.py
  - AxolotlTrainer._save：CP 开启且传入 state_dict 时，把 tensor detach().cpu()

src/axolotl/core/trainers/mixins/distributed_parallel.py
  - DistributedParallelMixin._save：dp_shard_enabled 时通过 accelerator.get_state_dict 获取 state_dict
  - create_accelerator_and_postprocess：pure CP 无 FSDP plugin 时修正 distributed_type

src/axolotl/train.py
  - save_trained_model：按 FSDP / DeepSpeed / 普通路径保存最终模型
```

## 6.3 主流程拆解

最终保存入口在 `src/axolotl/train.py:632-637`：训练结束后调用 `save_trained_model(cfg, trainer, model)`。

FSDP / CP 相关路径在 `src/axolotl/train.py:294-334`：如果 `trainer.is_fsdp_enabled or cfg.fsdp_config`，先设置 FSDP state dict type，再 `trainer.save_model(cfg.output_dir)`。如果是 `SHARDED_STATE_DICT`，后面可能调用 `merge_fsdp_weights()` 合并，并处理 PEFT adapter 文件名。

CP 特有修复在 `src/axolotl/core/trainers/base.py:805-823`：

```python
# fix for Context Parallel save: CP eval invalidates tensor storage pointers
if state_dict is not None and self.axolotl_cfg.context_parallel_size > 1:
    state_dict = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in state_dict.items()
    }
```

这个逻辑只在 `_save()` 已经拿到 `state_dict` 时触发；它不改变 checkpoint 拓扑，也不主动 gather CP 输出。它只是把 tensor 复制到 CPU，获得新的有效 storage，避免 safetensors 写入时踩到被 CP eval 影响过的 storage pointer。

另一个 CP 兼容点在 `DistributedParallelMixin.create_accelerator_and_postprocess()`（`src/axolotl/core/trainers/mixins/distributed_parallel.py:23-31`）：如果 Accelerate 把 distributed_type 标成 FSDP，但没有 fsdp_plugin，则将其修正为 `MULTI_GPU`。注释写得很直白：`handle Context Parallelism without FSDP`。

## 6.4 关键细节与误区澄清

> 误区十六：CP 有独立的 save/load/checkpoint 格式。

没有。CP 复用 Trainer / FSDP / DeepSpeed 的保存路径，只在 `_save()` 中对 state_dict tensor 做 CPU clone 修复。resume 仍主要走普通 `determine_last_checkpoint()` 和 Trainer resume。

> 误区十七：CP 保存修复会节省显存。

不一定。`detach().cpu()` 可以修复 storage，但会引入 CPU 内存复制和 CPU 侧峰值。它解决的是 safetensors 兼容性，不是训练期显存优化。

> 误区十八：pure CP 被当成 FSDP 是正常状态。

不是最终状态。Axolotl 用 `DistributedParallelMixin` 在没有 FSDP plugin 时把 distributed_type 改回 `MULTI_GPU`，避免后续保存/Trainer 分支误判。

## 6.5 本章小结

> 💡 **小结**
>
> * CP 不定义新 checkpoint 格式，只在保存时修复 state_dict tensor storage。
> * FSDP final state dict / sharded merge 仍由原有 FSDP 路径处理。
> * CP save 修复可能增加 CPU 内存峰值，解决的是 safetensors 写入可靠性。
> * pure CP 需要把 Accelerate 的 distributed_type 从伪 FSDP 状态修正回 MULTI_GPU。

# 七、完整主路径串联

## 7.1 完整调用栈

```text
User: axolotl train config.yaml
  │
  ├─ Step 1: CLI 启动
  │     ├─ src/axolotl/cli/main.py:78-125
  │     └─ src/axolotl/cli/utils/train.py:157-185
  │        生成 accelerate launch -m axolotl.cli.train config.yaml
  │
  ├─ Step 2: 配置加载与校验
  │     ├─ src/axolotl/cli/config.py:250-327
  │     ├─ src/axolotl/utils/schemas/config.py:969-991
  │     └─ src/axolotl/utils/schemas/validation.py:1508-1579
  │        归一 context_parallel_size / ring_attn_func，检查 flash_attention 与 ring_flash_attn
  │
  ├─ Step 3: 环境变量与 patch 准备
  │     ├─ src/axolotl/utils/trainer.py:621-640
  │     └─ src/axolotl/loaders/patch_manager.py:95-149,270-286
  │        写 PARALLELISM_CONFIG_CP_SIZE，patch Trainer / Accelerate
  │
  ├─ Step 4: DeviceMesh / parallelism_config 构建
  │     ├─ src/axolotl/loaders/model.py:196-212,437-443
  │     └─ src/axolotl/utils/distributed.py:299-370
  │        生成 ParallelismConfig 和 cuda DeviceMesh
  │
  ├─ Step 5: Trainer 构建与数据分发
  │     ├─ SFT: Accelerate prepare_data_loader 按 mesh 剥离 CP/TP 维度
  │     └─ GRPO: src/axolotl/core/trainers/grpo/sampler.py 自定义 sampler
  │
  ├─ Step 6: 训练执行
  │     ├─ src/axolotl/train.py:183-227
  │     └─ src/axolotl/utils/ctx_managers/sequence_parallel.py:170-357
  │        进入 SequenceParallelContextManager，注册 ring attention 和 forward hooks
  │
  ├─ Step 7: 每次 forward
  │     ├─ pre-hook 切 [B,S] -> [B,S/CP]
  │     ├─ patched FlashAttention 调 ring_flash_attn(group=cp_group)
  │     ├─ eval loss correction 可触发 CP group all_reduce
  │     └─ GRPO/EBFT 可触发 output all_gather
  │
  └─ Step 8: 保存与清理
        ├─ src/axolotl/train.py:254-334
        └─ src/axolotl/core/trainers/base.py:805-823
           CP state_dict tensor detach().cpu() 修复 safetensors storage
```

## 7.2 每一层做了什么

| 层级 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| 配置校验 | YAML / CLI override | `context_parallel_size` 默认 1；`ring_attn_func` 默认选择；检查依赖 | 无 | 无 | 初始化一次 |
| env setup | validated cfg | `PARALLELISM_CONFIG_CP_SIZE`、`ACCELERATE_USE_PARALLELISM_CONFIG` | 无 | 无 | 初始化一次 |
| patch manager | cfg / model_config | Trainer / Accelerate / attention 模块被替换 | 无 | 无 | 模型加载前一次 |
| DeviceMesh | world size + TP/CP/DP | `ParallelismConfig`、`DeviceMesh` | 可能初始化 process group | 无 | 初始化一次 |
| dataloader | dataset / sampler / mesh | CP/TP 组内同 batch，不同 DP 组不同 batch | 无或 sampler 同步 | 避免 batch 语义错误 | 每 epoch / dataloader 构造 |
| pre-hook | full local batch `[B,S]` | chunk batch `[B,S/CP]` | `num_items_in_batch` 可能 all_reduce | 节省后续激活/logits | 每 forward |
| ring attention | local Q/K/V | local attention output | ring_flash_attn 内部 CP group 通信 | 避免完整 attention materialization | 每 attention layer |
| output gather | local output | full sequence output | all_gather + shape all_gather | GRPO/EBFT 会恢复部分显存压力 | GRPO/EBFT 每 forward |
| save | state_dict | CPU tensor copy | FSDP save 可能通信 | 可能增加 CPU 内存 | checkpoint/final save |

## 7.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `Trainer._prepare_context_parallel_inputs` | HF Trainer 每 step 都调用，名字很像 CP 主入口 | 会调用，但 Axolotl 下不负责真正切分 | 主要做 guard/position_ids/shift_labels 兼容；Accelerate CP context 被 patch 成 no-op |
| `Accelerator._prepare_cp` 原生实现 | Accelerate 1.13 自带 torch DTensor CP | 被 `patch_prepare_cp()` 替换 | Axolotl 避免原生 CP 与自身 hook 重复切分 |
| docs 中的 “data collator handles chunking” | 文档描述像是 collator 切序列 | 当前源码未看到主路径由 collator 切 | SFT 依赖 mesh-aware dataloader 同 batch，forward pre-hook 切 tensor |
| `ring_attn_func` 在 `apply_sequence_parallelism` 参数中 | 看起来会决定 slicing 策略 | slicing 中未使用 | 它决定 attention patch 后端，而不是 batch 切法 |
| `AllGatherWithGrad` | 容易以为所有任务都会 gather 输出 | 仅 `gather_outputs=True` 时注册 | SFT 默认不 gather；GRPO/EBFT gather |
| `logits_to_keep` 切片分支 | 源码有 `elif key == "logits_to_keep"` | 对 1D mask 当前不可达 | 前置 `dim() <= 1` 会 continue，是潜在风险 |


## 7.4 本章小结

> 💡 **小结**
>
> * 一次真实训练调用中，CP 同时穿过 CLI、配置校验、env、patch、DeviceMesh、Trainer 和保存路径。
> * 初始化期的工作主要是声明拓扑和替换第三方行为；每 step 的工作主要是 pre-hook 切分和 attention 通信。
> * HF/Accelerate 原生 CP 名义上在调用栈里，但 Axolotl 将其切分上下文 no-op 化，避免双重切分。

# 八、关键数据流 / 状态流 / shape 流程

## 8.1 Tensor shape 变化

以 `B=1, S=8192, CP=4` 为例：

```text
Dataloader 输出（CP group 内每个 rank 同一份）:
  input_ids:      [1, 8192]
  attention_mask: [1, 8192]
  labels:         [1, 8192]

pre-hook padding（如需要）:
  labels pad value = -100
  others pad value = 0

pre-hook chunk 后:
  rank0 input_ids: [1, 2048]  position_ids: [1, 2048]
  rank1 input_ids: [1, 2048]  position_ids: [1, 2048]
  rank2 input_ids: [1, 2048]  position_ids: [1, 2048]
  rank3 input_ids: [1, 2048]  position_ids: [1, 2048]

attention 层:
  local Q: [1, 2048, n_heads, head_dim]
  local K/V: [1, 2048, n_kv_heads, head_dim]
  ring_flash_attn 在 CP group 内交换 K/V
  local output: [1, 2048, hidden]

SFT loss:
  logits 近似 [1, 2048, vocab]
  labels [1, 2048]
  不恢复 [1, 8192, vocab]

GRPO/EBFT output gather:
  all_gather local output -> [1, 8192, ...]
  backward: 从 grad_output 中切回本 rank 的 [1, 2048, ...]
```

真正节省显存的是 pre-hook 之后的模型主体计算：MLP 激活、layer hidden states、local logits 都按 `S/CP` 规模走。注意力本身通过 ring 通信避免把全序列 K/V 或 attention matrix 全量常驻在单卡上。收益会在需要 output gather 或保存 CPU clone 时部分消失。

## 8.2 Rank / Mesh / Process Group 变化

以 `world_size=8, tp=2, cp=2, dp_shard=2` 为例，逻辑上：

```text
总 rank: 0 1 2 3 4 5 6 7
mesh: dp_shard × cp × tp = 2 × 2 × 2

数据分发视角:
  tp 和 cp 维不增加数据并行样本数
  effective data parallel size = dp_shard × dp_replicate

CP group:
  每个 group 内 rank 共同处理同一 batch 的不同 sequence chunk
  ring_flash_attn 只在该 group 内通信
```

实际 rank 排布由 Accelerate `ParallelismConfig.build_device_mesh()` 决定；Axolotl 源码不手写 rank list，而是通过 `device_mesh[("cp",)].get_group()` 获取当前 rank 所在 CP group（`ring_attn/patch.py:159-184`）。

## 8.3 状态切换

CP 状态有三类：

```text
初始化 / 模型加载前:
  env:
    PARALLELISM_CONFIG_CP_SIZE
    ACCELERATE_ALLOW_CP_STANDALONE
    ACCELERATE_USE_PARALLELISM_CONFIG
  monkey patch:
    Trainer._prepare_context_parallel_inputs
    Accelerator._prepare_cp
    ParallelismConfig._validate_accelerator

进入 SequenceParallelContextManager:
  register_ring_attn_from_device_mesh()
    -> RING_ATTN_GROUP = cp process group
    -> patch HF FlashAttention namespace
  register model forward hooks

每次 forward:
  pre-hook 修改 kwargs 中的 tensor
  sample packing 时 update_ring_attn_params() 写 ring_flash_attn 上游全局参数
  eval 时 all_reduce weighted loss / total valid

退出 context:
  移除 model hooks
  不恢复 attention / accelerate patch
  不清空 RING_ATTN_GROUP
```

线程安全 / 进程安全方面：分布式训练通常是多进程单模型，因此模块级全局状态在每个进程内可接受。但如果同一 Python 进程内连续加载多个模型或运行不同测试，未恢复 patch 可能污染后续路径。


## 8.4 本章小结

> 💡 **小结**
>
> * shape 流程的关键变化是 `[B,S] -> [B,S/CP]`，attention 语义由 ring_flash_attn 在 CP group 内补回来。
> * CP group 是运行时从 DeviceMesh 取出的 process group，不是 Axolotl 手写 rank list。
> * CP 状态包含 env、module-level patch、global process group 和 forward hooks；只有 hooks 会在 context 退出时恢复。

# 九、核心机制深挖

## 9.1 Monkey Patch：零侵入接入还是维护风险？

它解决的问题是：在不 fork Transformers / Accelerate 的情况下，把 CP 接入现有 Trainer。它不能简单通过继承实现，因为 HF Trainer 的 CP 入口、Accelerate `_prepare_cp` 和 HF FlashAttention 函数都在第三方库内部。

源码中 patch 的粒度很不一样：

| patch | 替换对象 | 版本/签名保护 | 是否恢复 | 风险 |
|---|---|---|---|---|
| `trainer_context_parallel.py` | `Trainer._prepare_context_parallel_inputs` 函数 | 查找源码字符串 `GUARD_PATTERN` | 测试中 fixture 恢复，生产不恢复 | 上游源码变动导致 patch 跳过 |
| `parallelism_config.py` | `ParallelismConfig._validate_accelerator` / `AcceleratorState.is_fsdp2` / `Accelerator._prepare_cp` | 基本无版本检查 | 不恢复 | Accelerate 内部 API 变动风险 |
| `ring_attn/patch.py` | HF flash attention namespace / ring_flash_attn adapter | `batch.py` 用 `check_params` 比对签名 | 不恢复 | Transformers / ring_flash_attn 接口变动风险 |
| `trainer_loss_calc.py` | `Trainer.evaluation_loop` / `_maybe_log_save_evaluate` | 查找源码字符串 | 不恢复 | 多 patch 互斥、上游源码变动风险 |

这种设计的好处是主训练链路改动少，坏处是升级 Transformers / Accelerate / ring_flash_attn 时，行为可能不是编译期失败，而是 patch 静默跳过或运行时才暴露。

## 9.2 通信原语：前向和反向是否对称？

Axolotl 自己显式写出的通信有三类：

1. `apply_sequence_parallelism()` 中 token count 的 `dist.all_reduce(..., AVG)`（`sequence_parallel.py:156-165`）；
2. eval loss correction 的两个 `dist.all_reduce(..., SUM)`（`sequence_parallel.py:321-331`）；
3. `AllGatherWithGrad` 的 shape all_gather + tensor all_gather（`sequence_parallel.py:393-414`），反向切片（`:419-444`）。

attention 内部通信不在 Axolotl 源码中，而是委托给 `ring_flash_attn`。`batch.py:137-149` 调用 `ring_flash_attn_func(..., group=process_group)`；`patch.py:110-124` 调用 `llama3_flash_attn_varlen_func(..., group=process_group)`。基于文档 `docs/nd_parallelism.qmd:39-41`，这是 ring KV 交换，但其具体 primitive（send/recv、all-to-all 或自定义通信）未在当前仓库源码中确认。

`AllGatherWithGrad` 的前后向不是通信对称：前向 all_gather，反向只本地 slice，不做 reduce。这是因为 concat 的梯度天然按 sequence 维分块，不需要跨 rank 求和。相比之下 ring attention 的反向是否有额外通信由上游 `ring_flash_attn` 决定，本文不编造。

## 9.3 配置归一化：字段如何改变源码路径？

`context_parallel_size` 改变了至少六条路径：

1. validation 阶段要求 `flash_attention` 与 `ring_flash_attn`；
2. `prepare_optim_env()` 写 Accelerate parallelism env；
3. `PatchManager` 安装 Trainer CP guard patch 和 Accelerate parallelism patch；
4. `ModelLoader` 构建 `ParallelismConfig` / `DeviceMesh`；
5. `execute_training()` 进入 `SequenceParallelContextManager`；
6. GRPO builder 选择 `AxolotlGRPOSequenceParallelTrainer`。

因此它不是一个“传给下游库就完事”的字段，而是 Axolotl、Transformers、Accelerate、ring_flash_attn 四层协同的行为开关。


## 9.4 本章小结

> 💡 **小结**
>
> * CP 最核心的工程取舍是低侵入：用 patch 与 hook 接入第三方 Trainer/Accelerate/FlashAttention。
> * 显式通信中，output gather 的反向只是本地切片；attention 通信细节委托给 ring_flash_attn。
> * 配置字段会改变多条源码路径，不能只看 schema 判断真实行为。

# 十、显存、性能与通信分析

## 10.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌（靠 FSDP/TP 才节省） | CP 不切参数；示例常与 FSDP/TP 组合，但那是其他并行维度的收益 |
| optimizer state | ❌（靠 FSDP/ZeRO） | CP 不改变 optimizer state 持有方式 |
| 梯度 | ❌/间接 | 参数梯度不因 CP 分片；激活反传中间量随 sequence chunk 变小 |
| 输入 batch | ⚠️ 部分 | dataloader 输出到 rank 时仍是完整 `[B,S]`，pre-hook 后才切成 `[B,S/CP]` |
| 激活值 | ✅ | 模型主体只处理本地 sequence chunk |
| attention 中间状态 | ✅ 但换通信 | ring attention 避免单卡完整序列注意力；具体 kernel 内存由 ring_flash_attn 决定 |
| logits | ✅（SFT）/ ⚠️（GRPO/EBFT） | SFT 不 gather 输出；GRPO/EBFT 可能 all_gather 恢复完整序列输出 |
| 中间 buffer | ⚠️ | padding、shape all_gather、output gather、CPU clone save 会产生额外 buffer |
| 保存期 CPU 内存 | ❌ | CP `_save()` 会把 state_dict tensor detach 到 CPU，可能增加 CPU 峰值 |

真正显存大头通常是长序列下的激活、attention 相关中间量和 logits。CP 对这些最有效；对参数和 optimizer state 没直接帮助。

## 10.2 通信开销

| 阶段 | 通信类型 | group | 频率 | 源码依据 |
|---|---|---|---|---|
| attention forward/backward | ring KV 通信（具体 primitive 未在仓库确认） | CP group | 每层 attention | `ring_attn/patch.py:110-124`, `adapters/batch.py:137-149` |
| token count 修正 | all_reduce AVG | CP group | 含 `num_items_in_batch` 的 forward | `sequence_parallel.py:156-165` |
| eval loss correction | all_reduce SUM ×2 | CP group | eval forward 且有 loss | `sequence_parallel.py:321-331` |
| output gather | all_gather shape + all_gather tensor | CP group | GRPO/EBFT forward | `sequence_parallel.py:393-414` |
| GRPO vLLM | gather_object + broadcast_object_list | 全进程 / 主进程广播 | 生成阶段 | `grpo/trainer.py:328-380` |
| FSDP 保存 / 训练 | all_gather 等 FSDP 通信 | DP/FSDP group | 由 FSDP 决定 | `train.py:294-334` 间接触发 |

CP 是典型的“通信换显存”：每层 attention 增加 CP group 内通信；如果还开启 FSDP/TP，训练 step 同时存在参数 all-gather、TP 通信和 CP ring 通信。能否 overlap 主要取决于下游 kernel / backend，Axolotl 源码没有显式 overlap 调度。

## 10.3 性能取舍

CP 的收益随序列长度增长更明显。对于短序列，pre-hook、ring communication、patch 层复杂度和 all_reduce/gather 可能不划算。对于超长上下文，单卡 OOM 是硬约束，通信成本是换取可训练性的代价。

几个性能敏感点：

* `heads_k_stride` 文档和 schema 都提示“更大值更耗内存但可能更快”（`config.py:981-985`, `docs/sequence_parallelism.qmd:25-30`），这是 ring_flash_attn varlen path 的调度旋钮；
* SFT 默认不 output gather，保留 logits 显存收益；
* GRPO/EBFT 因为算法需要完整输出或对齐奖励，可能引入 all_gather；
* 保存期 `detach().cpu()` 是可靠性修复，可能带来 CPU 内存和拷贝开销；
* e2e SP 当前 skip，性能收益没有自动测试保护。


## 10.4 本章小结

> 💡 **小结**
>
> * CP 主要节省长序列激活、attention 中间量和 SFT logits，不节省参数或 optimizer state。
> * 通信开销集中在每层 ring attention、eval loss all_reduce、GRPO/EBFT output gather 和 RL 生成广播。
> * 序列越长、单卡越容易 OOM，CP 的收益越明显；短序列下通信与 patch 成本可能不划算。

# 十一、配置项、边界条件与坑点

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `context_parallel_size` | `validation.py:1508-1579`, `trainer.py:621-640`, `train.py:205-219` | 开启 CP、写 env、进入 context manager | 必须与 world topology 匹配；schema 不直接校验整除 |
| `sequence_parallel_degree` | `validation.py:1508-1515` | 迁移到 `context_parallel_size` | deprecated，不应新配置 |
| `flash_attention` | `validation.py:1516-1520` | CP > 1 时必须 true | 未开直接报错 |
| `ring_attn_func` | `validation.py:1563-1579`, `ring_attn/patch.py:186-212` | 选择 `varlen_llama3` 或 `batch_ring` patch path | enum 中 zigzag/stripe 被注释，当前不支持 |
| `sample_packing` | `validation.py:1522-1526`, `1569-1577` | sample packing 默认 `VARLEN_LLAMA3`，且 micro_batch_size 必须 1 | docs 与源码约束表述不完全一致 |
| `heads_k_stride` | `ring_attn/patch.py:200-202` | 传给 varlen llama3 ring attention | 更大可能更快但占更多内存；依赖上游实现 |
| `tensor_parallel_size` | `distributed.py:330-336`, Accelerate dataloader | 与 CP 一起从数据并行维度剥离 | TP/CP/FSDP 组合要求 world size 合法 |
| `dp_shard_size` / `dp_replicate_size` | `distributed.py:338-368` | 决定剩余 world 如何给 FSDP/HSDP | `dp_shard_size` 无 FSDP 会报错 |
| `trl.async_prefetch` / `trl.use_data_producer` | `grpo/__init__.py:39-43` | async GRPO 与 CP 互斥 | 配置同时开启直接报错 |
| `fsdp_config.state_dict_type` / `final_state_dict_type` | `train.py:294-334` | 影响 final save / merge | CP 只修 storage，不改变 FSDP 保存语义 |

最小可用配置大致是：

```yaml
context_parallel_size: 2
flash_attention: true
# 安装 axolotl[ring-flash-attn]
```

如果是 sample packing 长上下文，通常还需要：

```yaml
sample_packing: true
micro_batch_size: 1
ring_attn_func: varlen_llama3  # 可省略，源码会默认选择
```

如果组合 N-D parallelism，则需要显式规划：

```yaml
dp_shard_size: 2
tensor_parallel_size: 2
context_parallel_size: 2
fsdp_version: 2
fsdp_config:
  ...
```


## 11.1 本章小结

> 💡 **小结**
>
> * `context_parallel_size` 是总开关，但它的风险来自组合：sample packing、ring_attn_func、TP/FSDP、GRPO async、保存格式都会改变路径。
> * 部分约束在 schema 阶段报错，world topology 等问题更可能在 Accelerate/DeviceMesh 初始化阶段暴露。
> * 示例给出推荐姿势，但个别说明比源码约束更保守，排查时应以 validation 和主路径代码为准。

# 十二、测试、示例与覆盖缺口

## 12.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/test_context_parallel_batch_size.py:26-56` | CP 下 batch_size 按 `world_size / cp_size` 折算 | CPU 单测 mock `ring_flash_attn` |
| `tests/test_loaders.py:181-218` | `_get_parallel_config_kwargs` 对 TP/CP/DP/FSDP 的组合分解 | 覆盖 16 卡组合和默认 FSDP shard |
| `tests/monkeypatch/test_trainer_context_parallel_patch.py:36-66` | Trainer CP guard patch 替换与幂等 | 不验证真实 training_step |
| `tests/core/test_async_grpo.py:62-94` | GRPO CP 与 async_grpo 冲突 | 覆盖 trainer class selection |
| `tests/e2e/multigpu/solo/test_grpo.py:300-315` | GRPO 配置中包含 CP | 具体执行依赖 e2e 环境 |
| `tests/e2e/multigpu/solo/test_gdpo.py:442-465` | GDPO + CP + vLLM 配置 | 覆盖 RL 组合意图 |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:6-19` | FSDP + TP + CP N-D 示例 | 展示 8 GPU 组合 |
| `examples/alst/llama3-8b-fsdp2-alst.yaml:18-55` | 500K context + CP=8 + activation offloading + FSDP2 | 说明 CP 用于极长上下文 |

## 12.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| SFT CP e2e 真实训练 | ⚠️ 有测试但 skip | `tests/e2e/multigpu/patched/test_sp.py:102-104` 因 ring_flash_attn/transformers upstream 问题跳过 |
| ring_flash_attn patch 后真实 attention 正确性 | ❌ 当前环境未安装，测试有限 | 上游签名或行为变化导致训练错误 |
| `logits_to_keep` 1D mask 切片不可达 | ❌ 未见专门测试 | GRPO / generation 相关 logits mask 可能 shape 不一致 |
| 多机 CP / CP+TP+FSDP 真实拓扑 | ⚠️ 示例有，自动测试有限 | rank mapping / 通信 group 问题在多机才暴露 |
| 保存 / resume + CP eval storage 修复 | ❌ 未见专门 e2e | safetensors、CPU 峰值、resume 状态问题 |
| patch 恢复 / 多模型同进程隔离 | ⚠️ Trainer guard 单测 fixture 恢复，生产不恢复 | 测试污染或服务长生命周期污染 |
| 性能 / 显存收益 | ❌ 未见自动断言 | 退化、overlap 失败或通信瓶颈不易发现 |
| DeepSpeed 与 CP 组合 | ⚠️ 文档称 DeepSpeed 仅兼容 TP，ALST 有 deepspeed+CP 示例 | 行为边界需要进一步实测确认 |


## 12.3 本章小结

> 💡 **小结**
>
> * 当前测试覆盖了配置归一、parallelism kwargs、Trainer guard patch 和 GRPO 互斥，但真实 SP e2e 被 skip。
> * 最缺的是 ring attention 正确性、多机拓扑、保存/resume、性能显存收益与 patch 恢复测试。
> * 示例很丰富，但示例不能替代自动化正确性与性能回归保护。

# 十三、局限性与已知优化点

## 13.1 硬约束

* `context_parallel_size > 1` 必须 `flash_attention: true`（`validation.py:1516-1520`）。
* 必须安装 `ring_flash_attn` optional dependency（`validation.py:1528-1550`, `pyproject.toml:91-96`）。
* sample packing + CP 要求 `micro_batch_size == 1`（`validation.py:1522-1526`）。
* `ring_attn_func` 当前 enum 只开放 `varlen_llama3` 和 `batch_ring`，zigzag/stripe 被注释（`enums.py:100-108`）。
* GRPO sequence_parallel 与 async_grpo 互斥（`grpo/__init__.py:39-43`）。
* 非法 world topology 主要在 parallelism_config / mesh 初始化阶段暴露，不是 schema 阶段。

## 13.2 维护成本

* 多处源码字符串 patch 依赖上游 Transformers 函数文本；上游改一行 guard，patch 可能跳过。
* Accelerate `_prepare_cp` 被替换成 no-op，依赖 Accelerate 内部 API 名称和调用时机。
* ring attention patch 修改模块级 namespace，退出 context 不恢复。
* `RING_ATTN_GROUP` 是模块级全局变量；多模型/多阶段切换时需要小心。
* 文档与源码存在不一致：文档说 collator chunking，源码主路径是 forward pre-hook。

## 13.3 性能瓶颈

* 每层 attention 都需要 CP group 内 ring 通信；长序列收益大，短序列可能得不偿失。
* GRPO/EBFT output gather 会恢复完整 sequence output，削弱显存收益。
* eval loss correction 每次 eval forward 可能有两个 all_reduce。
* 保存期 CPU clone 可能造成 CPU 内存峰值和拷贝延迟。
* CP + TP + FSDP 叠加时，通信维度多，源码中没有显式 overlap 调度。

## 13.4 已知优化点

源码中已有几个 TODO / 暗示方向：

* `sequence_parallel.py:22-23` TODO：实现 zigzag、stripe patterns，目前只关注 batch ring 和 varlen llama3。
* `sequence_parallel.py:244` TODO：退出 context 时 un-patch attention 和 accelerate functions。
* `grpo/trainer.py:254-257` TODO：未来可能利用 Accelerate dataloader 的 `dispatch_batches` 和 `slice_fn_for_dispatch`，而不是完全绕开 prepare_data_loader。
* `base.py:805` TODO：保存修复等上游 Transformers PR 合并后可移除。
* `heads_k_stride` 是已有但需按模型/硬件调优的性能旋钮。


## 13.5 本章小结

> 💡 **小结**
>
> * CP 的硬约束集中在 flash attention、ring_flash_attn、sample packing micro batch、GRPO async 互斥和合法 world topology。
> * 维护成本主要来自第三方源码 patch 与不恢复的全局状态。
> * 已知优化方向包括更多 ring pattern、patch 可恢复、GRPO dataloader 与 Accelerate dispatch 融合、保存修复上游化。

# 小结与展望

Axolotl 的 Context Parallelism 实现可以用几个关键词概括。

## 关键词一：配置驱动但不是纯配置

`context_parallel_size` 从 YAML 进入后，会影响 validation、env、Accelerate parallelism、DeviceMesh、Trainer class、forward hook 和保存路径。它不是简单传给下游库的字段，而是跨 Axolotl / Transformers / Accelerate / ring_flash_attn 的组合开关。

## 关键词二：DeviceMesh 上的同 batch 分发

CP 的正确性首先来自数据分发：CP group 内 rank 必须拿同一份 batch。SFT 主要依赖 Accelerate mesh-aware dataloader；GRPO 则用自己的 `SequenceParallelRepeatRandomSampler` 和 vLLM slicing 维护这个不变量。

## 关键词三：hook 切序列，ring attention 补语义

Axolotl 不改模型源码，而是在 forward pre-hook 把 `[B,S]` 切成 `[B,S/CP]`。切完之后，patched FlashAttention 通过 ring_flash_attn 在 CP group 内交换 K/V，让每个本地 chunk 仍能看到全局上下文。

## 关键词四：通信换显存

CP 不节省参数和 optimizer state；它主要节省长序列激活、attention 中间状态和 SFT logits。代价是每层 attention 的 CP group 通信、eval loss all_reduce、GRPO/EBFT output gather，以及保存期 CPU clone。

## 关键词五：低侵入与高维护成本并存

用 monkey patch 接入第三方 Trainer/Accelerate/FlashAttention 的好处是改动集中、兼容现有训练链路；坏处是版本脆弱、全局状态难恢复、测试隔离成本高。当前 e2e SP 测试被 skip，也说明这个区域仍然依赖上游生态稳定。

这个实现适合：长上下文 SFT / pretraining、模型参数已通过 FSDP/TP 解决但 sequence 激活仍 OOM、单机或拓扑清晰的多 GPU 训练。不适合：短序列、对 patch 稳定性要求极高的生产服务、需要 async GRPO 的 RL 训练、无法安装或维护 ring_flash_attn 的环境。

后续值得继续走读的方向有三个：一是 ring_flash_attn 上游 kernel 的前后向通信细节；二是 CP + FSDP2 + QLoRA / torchao optimizer 的保存和量化边界；三是 GRPO/vLLM 下 CP 与 generation、reward、advantage 的全链路性能画像。只有把这些下游细节也串起来，才能完整评估 Axolotl 在超长上下文训练中的真实扩展边界。
