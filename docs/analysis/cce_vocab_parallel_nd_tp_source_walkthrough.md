# Axolotl 源码走读：CCE fork 的 vocab-parallel loss 与 ND parallelism / TP DeviceMesh 实现解析

在大词表语言模型训练里，最后一层 `lm_head` 和 cross entropy 经常是一个容易被低估的显存黑洞。Tensor Parallelism（TP）已经把 `lm_head.weight` 沿 vocab 维切到了不同 GPU 上；但如果普通 forward 仍然把每个 rank 的局部 logits all-gather 回完整 `[tokens, vocab]`，那么 TP 在 loss 侧的显存收益会被“还原完整 logits”这一步吃掉。

Axolotl 当前的做法很有意思：它自己不实现 TP，也不把 CCE 写进 Trainer，而是把 **Accelerate / PyTorch DeviceMesh、Transformers `tp_plan="auto"`、以及 Axolotl fork 的 Cut Cross Entropy（CCE）** 串在一起。真正关键的不是“启用 CCE”四个字，而是 CCE fork 能识别 `lm_head.weight` 已经是 DTensor，并把它解释成 vocab-parallel shard：每个 TP rank 只拿自己的 vocab 区间做局部 loss，然后用 TP process group 做少量 all-reduce，还原全局 softmax 语义。

本文不展开 Megatron TP、DTensor 或 CCE 论文的数学推导，而是顺着源码主路径回答一个工程问题：**CCE fork 的 vocab-parallel loss 到底如何和 Axolotl 的 ND parallelism / TP device mesh 交互？它省了哪些显存，又新增了哪些通信和维护风险？**

# 前言

## 业务 / 工程背景

目标特性出现在多 GPU 微调场景，特别是同时满足以下条件时：

- 模型较大，需要 FSDP / HSDP 分片参数、梯度和 optimizer state；
- 词表较大，`lm_head(hidden_states)` 生成的 `[batch, seq, vocab]` logits 显存很高；
- 用户启用 `tensor_parallel_size > 1`，希望 `lm_head.weight` 在 TP 维度上切 vocab；
- 用户启用 Axolotl 的 `CutCrossEntropyPlugin`，希望 loss 阶段不物化完整 logits；
- 可能还叠加 `context_parallel_size > 1`，让长序列在 CP 维度切开。

## 核心矛盾

这里的工程矛盾可以压缩成三句话：

1. **Transformers TP 的 `colwise_gather_output` 默认会把 `lm_head` 输出 gather 回完整 vocab logits**，这对推理/普通 forward 友好，但对训练 loss 显存不友好。
2. **CCE 要避免完整 logits，但 TP 下 `lm_head.weight` 已经不是普通 Tensor，而是 DTensor shard**，loss kernel 必须理解 vocab 分片和 TP process group。
3. **Axolotl 的 ND parallelism 还可能同时存在 CP/FSDP 维度**，因此 CCE 不能随便用 world group 通信，必须只沿 TP 维度还原 softmax 语义。

## 本文主线

本文按机制而不是按文件展开：

1. 配置如何从 YAML 变成 CCE plugin、TP/CP/FSDP 并行度；
2. DeviceMesh 如何提供命名 TP group；
3. 模型加载时 `tp_plan="auto"` 如何让 `lm_head.weight` 变成 vocab-sharded DTensor；
4. CCE patched forward 如何绕开 logits all-gather；
5. vocab-parallel loss 的 forward/backward 通信语义；
6. 一次真实训练调用的完整主路径；
7. shape / rank / state 流程；
8. 显存、通信、性能、配置、测试和边界风险。

## 不展开的内容

本文不讲：FSDP 原理、Megatron TP 数学、DTensor 内部 dispatcher、Ring Attention 论文细节、CCE Triton kernel 的每条指令。本文只讲 Axolotl 如何把这些机制接入训练链路，以及源码中能确认的真实执行逻辑。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/config.py` | 读取 YAML，注册 plugin，动态合并 CCE schema，执行配置校验与归一化 |
| `src/axolotl/utils/trainer.py` | 写入 `PARALLELISM_CONFIG_*` / `ACCELERATE_USE_PARALLELISM_CONFIG` 环境变量 |
| `src/axolotl/utils/distributed.py` | 根据 TP/CP/DP/FSDP 配置构造 `ParallelismConfig` 与 `DeviceMesh` |
| `src/axolotl/loaders/model.py` | 模型加载前构建 mesh；TP 开启时传入 `tp_plan="auto"` 和 `device_mesh` |
| `src/axolotl/integrations/cut_cross_entropy/__init__.py` | CCE plugin；检查 fork；在模型加载前 patch `ForCausalLM.forward` |
| `src/axolotl/train.py` | 训练主流程；CP context 进入点；保存路径 |
| `src/axolotl/utils/ctx_managers/sequence_parallel.py` | CP forward hook、输出 gather、eval loss 修正 |
| `cut_cross_entropy/transformers/utils.py`（外部 fork `fec1a88`） | `apply_lce()` 识别 DTensor，并生成 vocab-parallel options |
| `cut_cross_entropy/cce.py`（外部 fork `fec1a88`） | CCE autograd Function；vocab-parallel forward/backward 主逻辑 |
| `cut_cross_entropy/vocab_parallel/utils.py`（外部 fork `fec1a88`） | TP group 内 all-reduce：LSE、correct logit、hidden grad |

> 外部 CCE fork 由 Axolotl 安装脚本固定到 `axolotl-ai-cloud/ml-cross-entropy.git@fec1a88`，见 `scripts/cutcrossentropy_install.py:30-33` 和 `src/axolotl/integrations/cut_cross_entropy/README.md:20-23`。本文的外部源码行号基于临时检出的 `axolotl-ai-cloud/ml-cross-entropy@fec1a88`。

# 一、配置入口：为什么 `cut_cross_entropy` 不是一个孤立开关

## 1.1 设计哲学与核心问题

Axolotl 的用户入口是 YAML。用户看起来只是在配置里写：

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
cut_cross_entropy: true

tensor_parallel_size: 2
context_parallel_size: 2
fsdp_version: 2
fsdp_config:
  ...
```

但这几行会进入两个完全不同的系统：

- `cut_cross_entropy` 属于 **plugin 动态 schema + pre-model-load patch**；
- `tensor_parallel_size/context_parallel_size/dp_*` 属于 **Accelerate ParallelismConfig / DeviceMesh**；
- `fsdp_config` 属于 **Trainer/Accelerate/FSDP2 准备与保存路径**。

如果没有配置归一化层，后续会出现三个问题：

1. CCE 字段没有注册 plugin 时不会进入 schema，也不会触发 patch；
2. TP/CP rank 应该拿同一份 batch，而不是像普通 DP 那样各拿不同 batch；
3. CCE vocab-parallel loss 需要 TP group，但 TP group 只有在 DeviceMesh 构造后才有语义。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/train.py
  - do_cli：训练 CLI 入口，调用 load_cfg() 后进入 do_train()

src/axolotl/cli/config.py
  - load_cfg：读取 YAML、覆盖 CLI 参数、注册 plugins、validate_config、normalize_config
  - prepare_plugins：按 cfg.plugins 注册插件

src/axolotl/integrations/config.py
  - merge_input_args：把插件提供的 Pydantic schema 动态并入 AxolotlInputConfig

src/axolotl/integrations/cut_cross_entropy/args.py
  - CutCrossEntropyArgs：CCE 插件配置字段与校验

src/axolotl/utils/trainer.py
  - setup_parallelism_envs：把 TP/CP/DP 配置写入环境变量

src/axolotl/utils/config/__init__.py
  - normalize_config：根据 TP/CP 计算 effective data-parallel world size，重写 batch_size
```

## 1.3 主流程拆解

用户执行训练时，CLI 先走 `do_cli()`：

```text
axolotl train config.yaml
  -> src/axolotl/cli/train.py:55-91 do_cli(config, **kwargs)
    -> src/axolotl/cli/config.py:230-346 load_cfg(config, **kwargs)
      -> prepare_plugins(cfg)
      -> validate_config(cfg)
      -> prepare_optim_env(cfg)
      -> normalize_config(cfg)
      -> plugin_set_cfg(cfg)
    -> do_train(parsed_cfg, parsed_cli_args)
```

关键顺序在 `load_cfg()` 内：

- `prepare_plugins(cfg)` 在 `validate_config()` 之前执行（`src/axolotl/cli/config.py:306-308`）；
- `validate_config()` 如果发现 `cfg.plugins`，会调用 `merge_input_args()` 动态合并插件 schema（`src/axolotl/utils/config/__init__.py:324-337`）；
- `merge_input_args()` 会从 `PluginManager.get_input_args()` 收集插件 schema，并用 `exec()` 构造新的 `AxolotlInputConfig` / `AxolotlConfigWCapabilities`（`src/axolotl/integrations/config.py:27-57`）。

CCE 插件提供的 schema 很小，但很关键：

```python
# src/axolotl/integrations/cut_cross_entropy/args.py:28-54
class CutCrossEntropyArgs(BaseModel):
    cut_cross_entropy: Optional[bool] = True

    @model_validator(mode="before")
    def check_dtype_is_half(cls, data):
        if data.get("cut_cross_entropy") and not (data.get("bf16") or data.get("fp16")):
            raise ValueError("Cut Cross Entropy requires fp16/bf16 training ...")

    @model_validator(mode="before")
    def check_chunked_cross_entropy_not_set(cls, data):
        if data.get("chunked_cross_entropy"):
            raise ValueError("Cut Cross Entropy does not support chunked cross entropy ...")
```

与此同时，TP/CP 并行配置会被写成 Accelerate 可读的环境变量：

```python
# src/axolotl/utils/trainer.py:621-640
if cfg.tensor_parallel_size and cfg.tensor_parallel_size > 1:
    os.environ["PARALLELISM_CONFIG_TP_SIZE"] = str(cfg.tensor_parallel_size)
if cfg.context_parallel_size and cfg.context_parallel_size > 1:
    os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cfg.context_parallel_size)
    os.environ["ACCELERATE_ALLOW_CP_STANDALONE"] = "true"
    patch_prepare_cp()
if set_accelerate_parallelism_config:
    os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
```

`normalize_config()` 还会把全局 batch size 按有效 DP world size 放大：

```python
# src/axolotl/utils/config/__init__.py:134-142
if cfg.world_size != 1:
    if cfg.fsdp or cfg.fsdp_config or cfg.ddp:
        effective_world_size = (
            cfg.world_size
            // (cfg.context_parallel_size or 1)
            // (cfg.tensor_parallel_size or 1)
        )
        cfg.batch_size = cfg.batch_size * effective_world_size
```

这段代码说明：TP/CP 不是“更多不同 batch 的数据并行维度”。同一个 TP/CP 组内 rank 要处理同一份样本的不同模型/序列切片，所以全局 batch 的有效 DP 维度要除掉 `tp_size * cp_size`。

## 1.4 关键细节与误区澄清

> 容易误解点一：写了 `cut_cross_entropy: true` 就一定启用 CCE。

不一定。CCE 是插件字段。真正把字段并入 schema、并在模型加载前调用 `cce_patch()` 的前提是 YAML 里注册了 `axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin`。插件入口 `get_input_args()` 返回 `axolotl.integrations.cut_cross_entropy.CutCrossEntropyArgs`（`src/axolotl/integrations/cut_cross_entropy/__init__.py:47-48`），patch 发生在 `pre_model_load()`（同文件 `86-103`）。没有 plugin，就不会有这个生命周期 hook。

> 容易误解点二：`tensor_parallel_size` 的 schema 描述只说 DeepSpeed AutoTP，所以 FSDP2 + TP 不走这条配置。

源码与描述不完全一致。字段描述在 `src/axolotl/utils/schemas/config.py:993-997` 写着 “Only supported with DeepSpeed AutoTP”，但 `ModelLoader._build_model()` 在 `tensor_parallel_size > 1` 时显式设置 `tp_plan="auto"` 和 `device_mesh`（`src/axolotl/loaders/model.py:749-755`），这是 Transformers / DTensor TP 路径。DeepSpeed 只是另一路：`validation.py:1121-1148` 会把 `tensor_parallel.autotp_size` 注入 DeepSpeed JSON。

> 容易误解点三：TP/CP 只影响模型，不影响数据分发。

不对。`normalize_config()` 已经用 `world_size // cp_size // tp_size` 计算 effective world size（`src/axolotl/utils/config/__init__.py:134-142`），Accelerate 的 DataLoader 也会在有 `torch_device_mesh` 时按 `tp` 和 `cp` 调整 `process_index/num_processes`（`/usr/local/lib/python3.12/dist-packages/accelerate/data_loader.py:1119-1155`）。同组 rank 需要拿到同一份 batch。

## 1.5 本章小结

> 💡 **小结**
>
> * CCE 是 plugin 生命周期能力，不是基础 schema 中的普通布尔开关。
> * TP/CP 配置会被写入 Accelerate 环境变量，并影响 batch size 与 DataLoader rank 映射。
> * CCE 与 TP 的真正交汇点还不在配置层，而是在后续 `lm_head.weight` 变成 DTensor 后。

# 二、DeviceMesh：TP group 如何成为 vocab-parallel loss 的通信边界

## 2.1 设计哲学与核心问题

vocab-parallel loss 的核心问题是：**softmax 的分母需要全词表，但每个 TP rank 只持有词表的一段**。

这意味着 loss 不能只看本地 vocab shard；但也不应该用 world group 通信。假设同时启用 FSDP、CP、TP：

- FSDP 维度负责参数 shard / all-gather；
- CP 维度负责长序列切分和 ring attention；
- TP 维度负责 `lm_head.weight` vocab shard；
- CCE vocab-parallel loss 的 LSE、correct logit、hidden grad 同步只能发生在 **TP group** 内。

DeviceMesh 的价值就在这里：它把 “world_size=8” 这种扁平 rank 集合变成带名字的多维拓扑，让下游可以精确拿 `device_mesh["tp"]`。

## 2.2 源码入口与关键对象

```text
src/axolotl/utils/distributed.py
  - _get_parallel_config_kwargs：把 world_size、tp/cp/dp 配置变成 ParallelismConfig kwargs
  - build_parallelism_config：创建 ParallelismConfig 并 build_device_mesh("cuda")

/usr/local/lib/python3.12/dist-packages/accelerate/parallelism_config.py
  - ParallelismConfig.__post_init__：从 PARALLELISM_CONFIG_* 环境变量读取大小
  - ParallelismConfig._get_mesh：按 canonical order 排列维度
  - ParallelismConfig.build_device_mesh：调用 PyTorch init_device_mesh，并 flatten dp / dp_shard_cp

/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py
  - Accelerator.__init__：ACCELERATE_USE_PARALLELISM_CONFIG=true 时构造 ParallelismConfig
  - torch_device_mesh：暴露 state.device_mesh
```

## 2.3 主流程拆解

Axolotl 自己构建 mesh 的代码很短：

```python
# src/axolotl/utils/distributed.py:299-316
def build_parallelism_config(cfg):
    pc_kwargs = _get_parallel_config_kwargs(
        get_world_size(),
        cfg.tensor_parallel_size,
        cfg.context_parallel_size,
        cfg.dp_shard_size,
        cfg.dp_replicate_size,
        bool(cfg.fsdp or cfg.fsdp_config),
    )
    if pc_kwargs:
        parallelism_config = ParallelismConfig(**pc_kwargs)
        device_mesh = parallelism_config.build_device_mesh("cuda")
        return parallelism_config, device_mesh
    return None, None
```

`_get_parallel_config_kwargs()` 先扣掉 TP，再扣掉 CP，然后把剩余 world size 分配给 DP shard / replicate（`src/axolotl/utils/distributed.py:319-370`）：

```text
world_size
  -> / tensor_parallel_size  => tp_size
  -> / context_parallel_size => cp_size
  -> remaining_world_size    => dp_shard_size / dp_replicate_size
```

Accelerate 的 `ParallelismConfig` 则从环境变量读取默认值：

```python
# /usr/local/lib/python3.12/dist-packages/accelerate/parallelism_config.py:274-289
if self.tp_size is None:
    self.tp_size = int(os.environ.get("PARALLELISM_CONFIG_TP_SIZE", "1"))
if self.cp_size is None:
    self.cp_size = int(os.environ.get("PARALLELISM_CONFIG_CP_SIZE", "1"))
```

mesh 维度的顺序不是随意的。Accelerate 使用固定 canonical order：

```python
# accelerate/parallelism_config.py:260-272
mesh_order = ["dp_replicate", "dp_shard", "cp", "sp", "tp"]
sorted_items = sorted(mesh_dims.items(), key=lambda x: mesh_order.index(x[0]))
return tuple(zip(*sorted_items))
```

然后 `build_device_mesh()` 调用 PyTorch `init_device_mesh()`，并创建 flatten 后的 `dp` / `dp_shard_cp` / `dp_cp` mesh（`accelerate/parallelism_config.py:211-244`）。其中 FSDP2 后续会用 `fsdp_dim_names`：

```python
# accelerate/parallelism_config.py:158-164
@property
def fsdp_dim_names(self):
    dims = []
    if self.dp_replicate_enabled:
        dims += ["dp_replicate"]
    dims += ["dp_shard_cp"]
    return dims
```

这说明在 FSDP + CP 场景，FSDP shard 维度会把 `dp_shard` 与 `cp` flatten 成 `dp_shard_cp`；而 TP 维度保持正交，不参与 FSDP 的参数 shard mesh。

## 2.4 关键细节与误区澄清

> 容易误解点四：Axolotl 只构造一次 DeviceMesh。

不是。至少有两条构造路径：

1. `setup_parallelism_envs()` 写环境变量，Accelerate 在 `Accelerator.__init__` 看到 `ACCELERATE_USE_PARALLELISM_CONFIG=true` 时创建 `ParallelismConfig()`（`accelerate/accelerator.py:453-459`），并放入 `accelerator.state.device_mesh`；
2. `ModelLoader._set_parallel_config()` 直接调用 `build_parallelism_config()`，把 `device_mesh` 放到 `self.device_mesh`，用于 `from_pretrained(tp_plan="auto", device_mesh=...)`（`src/axolotl/loaders/model.py:437-442`、`749-755`）。

这两条路径应由同一组配置驱动，逻辑上得到同一个 mesh，但服务的消费者不同：一个给 Trainer/Accelerate/FSDP/CP，一个给 Transformers TP 模型加载。

> 容易误解点五：DeviceMesh 本身节省显存。

DeviceMesh 只是拓扑元数据和 process group。显存收益来自后续消费者：Transformers TP 切参数，CP hook 切序列，FSDP2 `fully_shard()` 切参数/梯度/optimizer state，CCE 避免 logits。`build_device_mesh()` 本身没有任何 tensor sharding 算子。

## 2.5 本章小结

> 💡 **小结**
>
> * CCE vocab-parallel loss 需要的是 `tp` 子 mesh，而不是全局 world group。
> * Axolotl 通过环境变量和显式 `build_parallelism_config()` 两条路径把同一套并行配置传给 Accelerate 与 Transformers。
> * FSDP/CP/TP 在 mesh 中是命名维度；FSDP2 使用 `dp_shard_cp`，CCE 使用 `tp`。

# 三、模型加载：`tp_plan` 让 `lm_head.weight` 变成 vocab-sharded DTensor

## 3.1 设计哲学与核心问题

Axolotl 没有自己写 TP 切分规则。它选择把 TP 委托给 HuggingFace Transformers：用户设置 `tensor_parallel_size: 2` 后，Axolotl 在 `from_pretrained()` 中传入：

```python
tp_size=2
tp_plan="auto"
device_mesh=<DeviceMesh(..., mesh_dim_names=(..., "tp"))>
```

这个设计的好处是 Axolotl 不需要为每个模型维护 `q_proj/k_proj/v_proj/lm_head` 的切分表；坏处是它依赖 Transformers 模型类是否提供 `_tp_plan`，以及上游 TP implementation 的 DTensor 语义。

对 CCE 来说，最重要的是 `lm_head.weight` 被 `colwise_gather_output` 切在输出特征维，也就是 vocab 维。只有这样，外部 CCE fork 才能把 DTensor local shard 当成 vocab shard。

## 3.2 源码入口与关键对象

```text
src/axolotl/loaders/model.py
  - ModelLoader.load：patch 顺序和模型加载总控
  - _apply_pre_model_load_setup：决定是否构建 parallel config
  - _set_parallel_config：构建 DeviceMesh
  - _build_model：TP 开启时设置 tp_size/tp_plan/device_mesh

src/axolotl/integrations/cut_cross_entropy/__init__.py
  - CutCrossEntropyPlugin.pre_model_load：模型加载前调用 cce_patch()

/usr/local/lib/python3.12/dist-packages/transformers/models/llama/modeling_llama.py
  - LlamaForCausalLM._tp_plan：lm_head -> colwise_gather_output

/usr/local/lib/python3.12/dist-packages/transformers/integrations/tensor_parallel.py
  - ColwiseParallel：weight dim -2 sharding；gather_output=True 时 all_gather outputs
```

## 3.3 主流程拆解

模型加载顺序在 `ModelLoader.load()`：

```text
src/axolotl/loaders/model.py:161-194
ModelLoader.load()
  -> patch_manager.apply_pre_model_load_patches()
  -> _apply_pre_model_load_setup()
       -> _set_parallel_config()    # 如果 TP/CP/FSDP parallel config 启用
  -> PLUGIN_MANAGER.pre_model_load(cfg)
       -> CutCrossEntropyPlugin.pre_model_load(cfg)
       -> cce_patch(model_config_type)
  -> patch_manager.apply_post_plugin_pre_model_load_patches()
  -> _build_model()
       -> from_pretrained(..., tp_plan="auto", device_mesh=self.device_mesh)
```

这里有一个顺序细节：**CCE patch 发生在 `_build_model()` 前**。也就是说，Axolotl 先把 Transformers 的 `ForCausalLM.forward` 类方法换成 CCE 版本，然后再加载模型实例。源码依据是 `PLUGIN_MANAGER.pre_model_load(self.cfg)` 位于 `_build_model()` 之前（`src/axolotl/loaders/model.py:172-176`）。

TP 设置发生在 `_build_model()`：

```python
# src/axolotl/loaders/model.py:749-755
if self.cfg.tensor_parallel_size > 1:
    self.model_kwargs["tp_size"] = self.cfg.tensor_parallel_size
    self.model_kwargs["tp_plan"] = "auto"
    self.model_kwargs["device_mesh"] = self.device_mesh
    if "device_map" in self.model_kwargs:
        del self.model_kwargs["device_map"]
```

以 Llama / Qwen3 为例，Transformers 当前模型类提供的 `_tp_plan` 是：

```python
# transformers/models/llama/modeling_llama.py:441-450
class LlamaForCausalLM(...):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

# transformers/models/qwen3/modeling_qwen3.py:455-464
class Qwen3ForCausalLM(...):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
```

`colwise_gather_output` 在 Transformers TP implementation 中对应 `ColwiseParallel(gather_output=True)`（`transformers/integrations/tensor_parallel.py:1194-1204`）。`ColwiseParallel` 的注释和实现说明：weight 沿 dim `-2` 切分，输出默认是局部 vocab 维；如果 `gather_output=True`，forward 输出会 all-gather 成完整 logits（同文件 `684-713`）。对 `lm_head.weight` 形状 `[vocab, hidden]` 来说，dim `-2` 就是 vocab 维。

## 3.4 关键细节与误区澄清

> 容易误解点六：CCE patch 之后就不需要 Transformers TP 了。

不对。CCE patch 只是替换 forward 内 loss 计算；`lm_head.weight` 是否是 DTensor、怎么切 vocab、TP group 是什么，仍然由 Transformers TP + PyTorch DTensor 决定。Axolotl 只负责把 `tp_plan="auto"` 和 `device_mesh` 传给 `from_pretrained()`。

> 容易误解点七：`tp_plan="auto"` 保证所有模型都支持 TP。

不保证。`auto` 的意思是查模型类的 `_tp_plan`。如果某个模型没有 `_tp_plan`，或者 `_tp_plan` 没有覆盖 `lm_head`，CCE 的 vocab-parallel 分支就不会被触发。Axolotl 没有在 `_build_model()` 后检查 `lm_head.weight` 是否真的变成 DTensor，只在 `src/axolotl/loaders/model.py:852-857` 用 TODO workaround 修补 `_tp_size/_device_mesh` 元信息。

> 容易误解点八：`colwise_gather_output` 一定会产生完整 logits，所以 CCE+TP 仍然会 gather logits。

标准 forward 会。但 CCE patched forward 不调用 `self.lm_head(...)`，而是直接读取 `self.lm_head.weight` 传给 `apply_lce()`。因此 Transformers 的 module output hook（`ColwiseParallel._prepare_output_fn` 中的 all-gather）不会执行。这正是 CCE vocab-parallel loss 的关键交互点。

## 3.5 本章小结

> 💡 **小结**
>
> * Axolotl TP 的实际切分由 Transformers `_tp_plan` 和 PyTorch DTensor 完成。
> * `lm_head: colwise_gather_output` 意味着 `lm_head.weight` 沿 vocab 维切分。
> * CCE patch 发生在模型加载前；TP shard 发生在模型加载时；二者在 `lm_head.weight` DTensor 处汇合。

# 四、前向主流程：绕开 logits all-gather 的关键分叉

## 4.1 设计哲学与核心问题

标准 causal LM forward 是：

```text
hidden_states = model(input_ids).last_hidden_state
logits = lm_head(hidden_states)          # [B, S, V]
loss = cross_entropy(logits, labels)
```

在 TP 下，`lm_head` 的局部输出是 `[B, S, V/tp]`，但 `colwise_gather_output` 会把它 gather 成 `[B, S, V]`。这对返回完整 logits 很自然，但对训练 loss 来说非常贵。

CCE 的策略是把这条路径改成：

```text
hidden_states = model(input_ids).last_hidden_state
loss = linear_cross_entropy(hidden_states, lm_head.weight, labels)
logits = None
```

TP 下再进一步变成：

```text
lm_head.weight: DTensor(global=[V,H], local=[V/tp,H])
apply_lce(...)
  -> c.to_local()                    # 只拿本地 vocab shard
  -> local lse / local correct logit
  -> all_reduce over tp group         # 还原全局 softmax 语义
```

## 4.2 源码入口与关键对象

```text
cut_cross_entropy/transformers/llama.py（外部 fork）
  - cce_forward：替换 ForCausalLM.forward；labels 存在时调用 apply_lce()

cut_cross_entropy/transformers/utils.py（外部 fork）
  - PatchOptions.use_lce：决定是否走 CCE 分支
  - apply_lce：识别 DTensor，构造 VocabParallelOptions，调用 linear_cross_entropy

cut_cross_entropy/linear_cross_entropy.py（外部 fork）
  - linear_cross_entropy：分发到 cce_linear_cross_entropy 或 torch_compile 路径

cut_cross_entropy/cce.py（外部 fork）
  - cce_linear_cross_entropy：flatten tokens，构造 CCEParams
  - LinearCrossEntropyFunction：forward/backward 自定义 autograd
```

## 4.3 主流程拆解

以 Llama patch 为例，外部 fork 的 `cce_forward()` 先正常调用 backbone：

```python
# cut_cross_entropy/transformers/llama.py:53-66
outputs = self.model(...)
hidden_states = outputs.last_hidden_state
loss = None
logits = None
```

当 labels 存在且 `PatchOptions.use_lce()` 允许 CCE 时，它不会调用 `self.lm_head(hidden_states)`：

```python
# cut_cross_entropy/transformers/llama.py:68-83
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

最后返回时，CCE 分支下 `logits` 仍然是 `None`：

```python
# cut_cross_entropy/transformers/llama.py:92-98
return CausalLMOutputWithPast(
    loss=loss,
    logits=logits,
    ...
)
```

进入 `apply_lce()` 后，fork 会检查 `c` 是否为 DTensor：

```python
# cut_cross_entropy/transformers/utils.py:113-135
if isinstance(c, DTensor):
    device_mesh = c.device_mesh
    vocab_dim = 0
    process_group = device_mesh.get_group("tp")
    placement = c.placements[vocab_dim]
    if isinstance(placement, Shard):
        vocab_size = c.size(vocab_dim)
        vocab_parallel_options = VocabParallelOptions.from_vocab(
            vocab_size,
            process_group,
            reduce_e_grad=True,
        )
        cce_kwargs["vocab_parallel_options"] = vocab_parallel_options
    c_local = c.to_local()
else:
    c_local = c
```

这段代码是本文目标特性的核心证据：CCE fork 不是从 Axolotl 配置里直接读取 `tensor_parallel_size`，而是通过 **DTensor 自带的 `device_mesh`** 发现 TP group，通过 `to_local()` 取本地 vocab shard。

然后 `apply_lce()` 调用：

```python
# cut_cross_entropy/transformers/utils.py:152-160
loss = linear_cross_entropy(
    e,
    c_local,
    labels.to(e.device),
    bias=bias,
    shift=True,
    softcap=softcap,
    **cce_kwargs,
)
```

`linear_cross_entropy()` 会拒绝直接传入 DTensor（`cut_cross_entropy/linear_cross_entropy.py:62-67`），所以 `apply_lce()` 必须先把 `c` unwrap 成 local Tensor。它还会校验本地 `c.size(0)` 是否等于本 rank 的 vocab 区间长度（同文件 `74-77`）。这相当于在 loss 层确认：“你给我的分类器权重确实是本 rank 的 vocab shard”。

## 4.4 关键细节与误区澄清

> 容易误解点九：CCE 是全局替换 `torch.nn.functional.cross_entropy`。

不是。CCE fork 替换的是模型类的 `forward()`，让 forward 在 labels 存在时直接调用 `apply_lce()`。Axolotl 的 `CutCrossEntropyPlugin.pre_model_load()` 调用 `cce_patch(cfg.model_config_type, ...)`（`src/axolotl/integrations/cut_cross_entropy/__init__.py:86-103`）；外部 `cce_patch()` 再根据模型类型找到 patch 函数（`cut_cross_entropy/transformers/patch.py:151-199`）。

> 容易误解点十：CCE 分支仍会返回 logits，方便 callback 使用。

通常不会。外部 `cce_forward()` 初始化 `logits = None`，CCE 分支只设置 `loss`（`cut_cross_entropy/transformers/llama.py:64-80`），返回时 `logits=logits`（`92-98`）。如果某个自定义 callback、metric 或 Trainer 在训练时依赖 `outputs.logits`，它可能和 CCE 不兼容。

> 容易误解点十一：CCE 的 vocab-parallel 分支由 Axolotl 显式传入 TP group。

不是。Axolotl 没有向 CCE 传 `process_group`。CCE fork 是通过 `lm_head.weight` 这个 DTensor 的 `device_mesh.get_group("tp")` 找到 TP group（`cut_cross_entropy/transformers/utils.py:113-119`）。这要求 Transformers TP 生成的 DTensor mesh 维度名必须包含 `tp`。

## 4.5 本章小结

> 💡 **小结**
>
> * CCE patched forward 绕过 `self.lm_head(...)`，因此绕过了 `colwise_gather_output` 的完整 logits gather。
> * vocab-parallel loss 的触发条件是 `lm_head.weight` 为 DTensor 且 placement 是 `Shard`。
> * CCE 通过 DTensor 的 `device_mesh.get_group("tp")` 获取通信组，而不是读 Axolotl 配置。

# 五、通信语义：vocab-parallel loss 用 all-reduce 还原 softmax，而不是 all-gather logits

## 5.1 设计哲学与核心问题

softmax cross entropy 需要两部分：

```text
loss_i = logsumexp(logits_i over all vocab) - logits_i[target]
```

TP vocab shard 后，每个 rank 只能算：

```text
local_logits_i = hidden_i @ local_weight.T     # [tokens, V/tp]
local_lse_i = logsumexp(local_logits_i)
local_correct_i = target 若落在本 rank vocab 区间，则取对应 logit，否则 0
```

要恢复全局 loss，需要两类通信：

1. 对 `local_lse` 做跨 TP rank 的 max/sum 归约，得到全局 `logsumexp`；
2. 对 `local_correct_i` 做跨 TP rank 求和，因为只有目标 token 所在 shard 有非零 correct logit。

反向还要注意 hidden states 梯度：每个 vocab shard 都对 `dE` 有贡献，所以 `dE` 需要跨 TP rank all-reduce；而 `dC` 是本地 vocab shard 的权重梯度，不需要变成完整 `[V,H]`。

## 5.2 源码入口与关键对象

```text
cut_cross_entropy/vocab_parallel/utils.py
  - VocabParallelOptions.from_vocab：根据 TP group rank/world_size 算本 rank vocab [start, stop)
  - vp_reduce_lse：all_reduce MAX + all_reduce SUM
  - vp_reduce_correct_logit：all_reduce correct logit
  - vp_reduce_e_grad：all_reduce hidden gradient

cut_cross_entropy/cce.py
  - LinearCrossEntropyFunction.forward：local lse / correct logit + TP all-reduce
  - LinearCrossEntropyFunction.backward：target localize + cce_backward_kernel + optional dE all-reduce

cut_cross_entropy/cce_backward.py
  - cce_backward_kernel：生成 de/dc/dbias；末尾 reduce_e_grad 时 all-reduce de
```

## 5.3 主流程拆解

`VocabParallelOptions.from_vocab()` 用 TP group 内 rank 计算 vocab 区间：

```python
# cut_cross_entropy/vocab_parallel/utils.py:32-44
rank = torch.distributed.get_rank(group)
world_size = torch.distributed.get_world_size(group)
start, stop = partition_n_into_range(vocab_size, rank, world_size)
return cls(start, stop, group, reduce_e_grad)
```

forward 中，CCE 先算本地 LSE：

```python
# cut_cross_entropy/cce.py:59-76
ret = cce_lse_forward_kernel(e=e, c=c, ...)
lse = ret
if (vp_opts := params.vocab_parallel_options) is not None:
    lse = vp_reduce_lse(lse, pg=vp_opts.group)
```

`vp_reduce_lse()` 不是简单 sum，而是数值稳定的 logsumexp 归约：

```python
# cut_cross_entropy/vocab_parallel/utils.py:47-54
lse_max = vp_lse.clone()
torch.distributed.all_reduce(lse_max, op=torch.distributed.ReduceOp.MAX, group=pg)
lse = (vp_lse - lse_max).exp()
torch.distributed.all_reduce(lse, group=pg)
return lse_max + lse.log()
```

接着，CCE 找出 target 是否落在当前 vocab shard：

```python
# cut_cross_entropy/cce.py:83-95
vp_valids = ((targets >= vp_opts.start) & (targets < vp_opts.stop)).nonzero().to(torch.int32)
neg_dot_targets = params.targets - vp_opts.start
```

本地 correct logit 通过 `indexed_neg_dot_forward_kernel()` 计算后，再 all-reduce：

```python
# cut_cross_entropy/cce.py:100-120
neg_dot = indexed_neg_dot_forward_kernel(...)
if params.vocab_parallel_options is not None:
    global_neg_dot = neg_dot.new_zeros(lse.size())
    global_neg_dot[vp_valids] = neg_dot
    neg_dot = vp_reduce_correct_logit(global_neg_dot, pg=params.vocab_parallel_options.group)
nll = neg_dot.add_(lse)
```

反向中，非本 rank 的 target 被改成 padding sentinel，避免本地 kernel 对不存在的 vocab index 读写：

```python
# cut_cross_entropy/cce.py:160-177
if (vp_opts := params.vocab_parallel_options) is not None:
    is_my_target = (targets >= vp_opts.start) & (targets < vp_opts.stop)
    targets = torch.where(
        is_my_target,
        targets - vp_opts.start,
        targets.new_full((), c.size(0) + 1),
    )
    reduce_e_grad = vp_opts.reduce_e_grad
    pg = vp_opts.group
```

`cce_backward_kernel()` 返回 `de, dc, dbias` 后，如果 `reduce_e_grad=True`，会 all-reduce hidden gradient：

```python
# cut_cross_entropy/cce_backward.py:474-475
if reduce_e_grad and de is not None:
    de = vp_reduce_e_grad(de, pg)
```

`vp_reduce_e_grad()` 会把 `e_grad` 转成 fp32 做 all-reduce，再 cast 回原 dtype（`cut_cross_entropy/vocab_parallel/utils.py:68-76`）。

## 5.4 关键细节与误区澄清

> 容易误解点十二：vocab-parallel CCE 的前向通信是 all-gather vocab logits。

不是。源码中没有 gather `[tokens, vocab]`。前向通信是三个向量级 all-reduce：`lse_max`、`lse_sum`、`correct_logit`，都在 TP group 内（`cut_cross_entropy/vocab_parallel/utils.py:47-65`）。这比 all-gather 完整 logits 小得多。

> 容易误解点十三：前向省了 logits，反向就不需要 TP 通信。

不对。`dE` 是所有 vocab shard 对 hidden states 的贡献之和，CCE backward 仍要对 `de` 做 TP all-reduce（`cut_cross_entropy/cce_backward.py:474-475`）。省掉的是完整 logits 及其梯度 buffer，不是所有通信。

> 容易误解点十四：CCE 使用 CP group 做 vocab loss 通信。

不对。`apply_lce()` 明确调用 `device_mesh.get_group("tp")`（`cut_cross_entropy/transformers/utils.py:113-119`）。CP group 用于 sequence/ring attention，CCE vocab-parallel loss 用 TP group。二者在同一个 DeviceMesh 上，但维度不同。

## 5.5 本章小结

> 💡 **小结**
>
> * vocab-parallel CCE 用 all-reduce 还原 LSE 和 correct logit，不 all-gather 完整 logits。
> * backward 仍需要对 hidden gradient 做 TP all-reduce；这是用通信换 logits 显存的代价。
> * CCE 通信边界是 `tp` group；CP/FSDP 通信来自各自模块，不由 CCE 合并调度。

# 六、完整主路径串联

## 6.1 完整调用栈

以 `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml` 为例，这个配置同时注册 CCE plugin、设置 `dp_shard_size: 2`、`context_parallel_size: 2`、`tensor_parallel_size: 2`（`examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:1-19`），主路径可以串成：

```text
User: axolotl train examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml
  │
  ├─ Step 1: 配置加载与 plugin schema 合并
  │     ├─ src/axolotl/cli/train.py:55-91 do_cli()
  │     ├─ src/axolotl/cli/config.py:230-346 load_cfg()
  │     └─ src/axolotl/integrations/config.py:27-57 merge_input_args()
  │
  ├─ Step 2: 并行配置归一化与 env 写入
  │     ├─ src/axolotl/utils/trainer.py:621-640 setup_parallelism_envs()
  │     ├─ src/axolotl/utils/config/__init__.py:134-142 normalize_config()
  │     └─ accelerate/accelerator.py:453-459 ParallelismConfig()
  │
  ├─ Step 3: 模型加载前 patch + DeviceMesh 构建
  │     ├─ src/axolotl/loaders/model.py:161-194 ModelLoader.load()
  │     ├─ src/axolotl/loaders/model.py:437-442 _set_parallel_config()
  │     └─ src/axolotl/integrations/cut_cross_entropy/__init__.py:86-103 pre_model_load()
  │
  ├─ Step 4: TP 模型加载
  │     ├─ src/axolotl/loaders/model.py:749-755 tp_size/tp_plan/device_mesh
  │     ├─ transformers/models/qwen3/modeling_qwen3.py:455-464 _tp_plan + lm_head
  │     └─ transformers/integrations/tensor_parallel.py:684-713 ColwiseParallel
  │
  ├─ Step 5: Trainer 构建与 CP context
  │     ├─ src/axolotl/train.py:522-570 setup_model_and_trainer()
  │     └─ src/axolotl/train.py:183-227 execute_training()
  │
  ├─ Step 6: 每次 forward
  │     ├─ src/axolotl/utils/ctx_managers/sequence_parallel.py:255-288 CP pre-hook 切 sequence
  │     ├─ cut_cross_entropy/transformers/llama.py:53-83 patched forward
  │     ├─ cut_cross_entropy/transformers/utils.py:113-160 DTensor -> local vocab shard
  │     └─ cut_cross_entropy/cce.py:47-135 VP CCE forward
  │
  ├─ Step 7: backward
  │     ├─ cut_cross_entropy/cce.py:137-218 target localize + backward kernel
  │     └─ cut_cross_entropy/cce_backward.py:474-475 all_reduce dE over TP group
  │
  └─ Step 8: 保存 / resume
        ├─ src/axolotl/train.py:254-386 save_trained_model()
        ├─ src/axolotl/core/trainers/base.py:806-850 _save()
        └─ src/axolotl/monkeypatch/accelerate/fsdp2.py:100-193 get_state_dict()
```

## 6.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| 配置加载 | YAML + CLI overrides | `cfg`；plugin 注册；动态 schema | 无 | 无 | 初始化一次 |
| env 写入 | `cfg.tensor_parallel_size/context_parallel_size/dp_*` | `PARALLELISM_CONFIG_*` | 无 | 无 | 初始化一次 |
| DeviceMesh | world size + 并行度 | `DeviceMesh(..., "tp", "cp", ...)` | 初始化 process group | 无直接节省 | 初始化一次 |
| CCE patch | `model_config_type` | 替换 `ForCausalLM.forward` | 无 | 后续避免 logits | 模型加载前一次 |
| TP 加载 | `tp_plan="auto"` + mesh | `lm_head.weight` 等变为 DTensor | DTensor 初始化 / 权重分发 | 参数按 TP shard | 模型加载一次 |
| CP pre-hook | batch `[B,S]` | local batch `[B,S/cp]` | `num_items_in_batch` 可能 all_reduce | 激活按 CP 降低 | 每 forward |
| CCE VP loss | local hidden + DTensor weight | loss；`logits=None` | TP all_reduce LSE/correct | 避免 `[B,S,V]` logits | 每 forward |
| CCE backward | loss grad | `de/dc/dbias` | TP all_reduce `de` | 避免 logits grad | 每 backward |
| FSDP2 save | sharded state_dict | rank0 full state_dict | `DTensor.full_tensor()` / barrier | rank0 CPU/GPU 峰值 | save/checkpoint |

## 6.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `src/axolotl/monkeypatch/loss/chunked.py` | 名字也是 cross entropy 显存优化 | CCE 开启时不应进入 | `CutCrossEntropyArgs` 和全局校验都禁止与 `chunked_cross_entropy` 同时启用（`args.py:46-54`、`validation.py:974-1002`） |
| `src/axolotl/integrations/liger/*` | Liger FLCE 也避免 logits | 与 CCE 互斥；TP 下 Liger loss 被拦截 | Liger args 明确禁止 `tensor_parallel_size > 1` + `liger_fused_linear_cross_entropy`（`src/axolotl/integrations/liger/args.py:108-113`） |
| `PluginManager.post_model_load/post_trainer_create/post_train` | plugin 生命周期 hook 很多 | CCE 插件没有实现这些主逻辑 | CCE 只靠 `get_input_args()` 和 `pre_model_load()` 接入 |
| `cut_cross_entropy.transformers.patch.cce_patch(model_instance)` | 下游支持直接 patch 实例 | Axolotl 传 `model_config_type` 字符串 | Axolotl 选择类级 patch，在模型实例创建前替换 forward |
| `SequenceParallelContextManager._gather_outputs()` | 名字像 CP 必然 gather 回完整 sequence | SFT 默认不走 | `gather_outputs=cfg.rl in {GRPO, EBFT}`（`src/axolotl/train.py:217`）；普通 SFT 不 gather outputs |
| `Accelerator._prepare_cp` 原生 torch CP | Accelerate 有 CP 实现 | Axolotl patch 成 no-op | `patch_prepare_cp()` 把 `_prepare_cp` 改成 no-op context（`src/axolotl/monkeypatch/accelerate/parallelism_config.py:80-98`），实际 CP 由 Axolotl ring-attn context 处理 |

## 6.4 本章小结

> 💡 **小结**
>
> * 主路径不是“Trainer 调自定义 loss”，而是“模型 forward 被 patch 后自己返回 loss”。
> * TP 的 mesh 和 DTensor 在模型加载时形成；CCE 在每次 forward 时读取 DTensor 的 mesh。
> * 普通 SFT 下 CP 不 gather outputs，因此 CP 切 sequence 与 CCE 避免 logits 可以叠加。

# 七、关键数据流 / 状态流 / shape 流程

## 7.1 Tensor shape 变化

假设：

```text
B = micro_batch_size
S = sequence_len
H = hidden_size
V = vocab_size
CP = context_parallel_size
TP = tensor_parallel_size
T = B * (S / CP)   # 单 rank 上进入 loss 的 token 数，忽略 padding/shift
```

普通无 CP/TP/CCE 路径：

```text
input_ids:       [B, S]
hidden_states:   [B, S, H]
lm_head.weight:  [V, H]
logits:          [B, S, V]
loss:            scalar
```

CP + TP + CCE vocab-parallel 路径：

```text
原始 batch（同一 TP/CP 组内相同）:
  input_ids:      [B, S]
  labels:         [B, S]

CP pre-hook 后（每个 CP rank 一段 sequence）:
  input_ids:      [B, S / CP]
  labels:         [B, S / CP]
  position_ids:   [B, S / CP]

Transformer backbone 输出:
  hidden_states:  [B, S / CP, H]

TP 后的 lm_head.weight:
  global DTensor logical shape: [V, H]
  local tensor shape:           [V / TP, H]

CCE 内部 flatten:
  e:              [T, H]
  c_local:        [V / TP, H]

局部计算:
  local_lse:      [T]
  local_correct:  [T]  # 只有 target 落在本 rank vocab 范围时非零

TP all-reduce 后:
  global_lse:     [T]
  correct_logit:  [T]
  nll/loss:       [T] -> scalar

返回:
  loss:           scalar
  logits:         None
```

哪一步节省显存？

- CP：从模型输入开始把 sequence 维切成 `S/CP`，减少每 rank 的 attention/MLP 激活；
- TP：把 `lm_head.weight` 的 vocab 维切成 `V/TP`，减少每 rank 持有的输出层参数；
- CCE：不生成 `[B,S/CP,V]`，只在 kernel 内用 block 方式计算局部 lse/correct；
- CCE+TP：进一步避免 `colwise_gather_output` 把 `[T,V/TP]` gather 成 `[T,V]`。

哪一步恢复冗余？

- CCE forward 通过 all-reduce 恢复的是 **标量/向量级 softmax 语义**，不是完整 logits；
- CCE backward all-reduce 的是 `de: [T,H]`，不是 `dlogits: [T,V]`；
- 保存 full state dict 时，FSDP2/DTensor 可能通过 `full_tensor()` 恢复完整参数到 rank0（`src/axolotl/monkeypatch/accelerate/fsdp2.py:158-173`）。

## 7.2 Rank / Mesh / Process Group 变化

以 `world_size=8, dp_shard_size=2, context_parallel_size=2, tensor_parallel_size=2` 为例，Accelerate canonical order 是：

```text
mesh dims = (dp_shard, cp, tp)
mesh shape = (2, 2, 2)
row-major rank map:

(dp=0, cp=0, tp=0) -> rank0
(dp=0, cp=0, tp=1) -> rank1
(dp=0, cp=1, tp=0) -> rank2
(dp=0, cp=1, tp=1) -> rank3
(dp=1, cp=0, tp=0) -> rank4
(dp=1, cp=0, tp=1) -> rank5
(dp=1, cp=1, tp=0) -> rank6
(dp=1, cp=1, tp=1) -> rank7
```

因此：

```text
TP groups（固定 dp, cp，沿 tp 维）:
  [0, 1], [2, 3], [4, 5], [6, 7]

CP groups（固定 dp, tp，沿 cp 维）:
  [0, 2], [1, 3], [4, 6], [5, 7]

FSDP2 shard mesh:
  使用 dp_shard_cp flatten（不包含 tp）
```

这能解释一个重要现象：同一个 global rank 可能同时属于：

- 一个 TP group：用于 `lm_head.weight` vocab shard 和 CCE all-reduce；
- 一个 CP group：用于 sequence 切分、ring attention、eval loss 修正；
- 一个 FSDP group：用于参数 shard / all-gather / save gather。

这些通信组不能混用。

## 7.3 状态切换

这一特性有三类状态：

```text
1. 环境变量状态（进程级）
   写入: setup_parallelism_envs()
   读取: Accelerate ParallelismConfig.__post_init__()

2. 模块/类级 monkey patch 状态（Python 进程级）
   CCE: Transformers ForCausalLM.forward 被替换
   CP: ring_flash_attn / HF flash attention 被替换
   Accelerate: ParallelismConfig._validate_accelerator / Accelerator._prepare_cp 被替换

3. runtime group 状态（进程内全局变量 + model hook）
   RING_ATTN_GROUP: axolotl.monkeypatch.ring_attn.patch.RING_ATTN_GROUP
   model forward pre/post hooks: SequenceParallelContextManager.__enter__ 注册，__exit__ 移除
```

CP context 的局部 hook 会在退出时移除：

```python
# src/axolotl/utils/ctx_managers/sequence_parallel.py:233-245
def __enter__(self):
    self._register_model_hooks()

def __exit__(...):
    for handle in self.hook_handles:
        handle.remove()
    # TODO: Un-patch attention and accelerate functions
```

但注意 TODO：attention 和 accelerate patches 没有恢复。CCE 的类级 forward patch也不是 context-local。`RING_ATTN_GROUP` 是模块级全局变量（`src/axolotl/monkeypatch/ring_attn/patch.py:34-47`）。因此这些 patch 是进程级行为，测试或多模型场景要小心污染。

## 7.4 关键细节与误区澄清

> 容易误解点十五：CP context 退出后，所有 patch 都恢复了。

没有。`SequenceParallelContextManager.__exit__()` 只移除 model hooks，并明确有 TODO “Un-patch attention and accelerate functions”（`sequence_parallel.py:238-245`）。CCE forward patch 也没有恢复逻辑。它们适合一次训练进程，不适合在同一 Python 进程里反复切不同并行模式而不重启。

> 容易误解点十六：TP group 和 CP group 在 rank 上总是连续相邻。

不一定。由于 mesh order 是 `(dp_replicate, dp_shard, cp, sp, tp)`，TP 通常是最后一维，rank 连续；CP group 在 `tp` 固定时常常是跨步 rank，如 `[0,2]`。物理拓扑是否友好取决于 launcher 的 global rank 排列，DeviceMesh 不会自动按 NVLink 拓扑重排。

## 7.5 本章小结

> 💡 **小结**
>
> * CCE+TP 的关键 shape 是 `hidden [T,H]` 与本地 `lm_head [V/TP,H]`，不是完整 logits `[T,V]`。
> * TP group、CP group、FSDP group 是同一 DeviceMesh 的不同切片，通信语义不能混用。
> * 多个 monkey patch 是进程级状态，CP hook 可移除，但 attention/Accelerate/CCE patch 不完整恢复。

# 八、显存、性能与通信分析

## 8.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| `lm_head` 参数 | ✅ | Transformers TP 把 `[V,H]` 沿 vocab 维切成 `[V/TP,H]` |
| Transformer 层参数 | 取决于 `_tp_plan` / FSDP | Axolotl 自己不切；Transformers TP 和 FSDP2 决定 |
| 激活值 | ✅（CP）/ 部分（checkpointing） | `context_parallel_size` 把 sequence 切成 `S/CP`；CCE 不处理 attention/MLP 激活 |
| logits | ✅✅ | CCE 不物化 `[B,S/CP,V]`；TP 下还绕过 `colwise_gather_output` 完整 logits gather |
| logits grad buffer | ✅ | backward 直接算 `de/dc/dbias`，不保留完整 `dlogits` |
| optimizer state | ❌（CCE）/ ✅（FSDP/ZeRO） | CCE 不管理 optimizer；FSDP/ZeRO/optimizer 后端决定 |
| 输入 batch | ❌ | TP/CP rank 往往拿同一 batch；CP 只是 forward 前切 sequence |
| 中间 `[T]` buffer | ❌（新增小 buffer） | vocab-parallel CCE 需要 `lse/correct_logit` 等 `[T]` 向量 |
| 保存时 full state | ❌ | FSDP2 full save 会 `DTensor.full_tensor()`，rank0 可能有峰值 |

真正的大头通常是：

```text
logits bytes ≈ B * S_local * V * dtype_size
```

例如 `B=1, S=8192, CP=2, V=128k, bf16`：单 rank 完整 logits 约 `1 * 4096 * 128000 * 2 ≈ 1GB`，这还没算 softmax / CE 中间 buffer。CCE 避开的就是这部分；TP 再避免从 `[4096, 64000]` gather 回 `[4096, 128000]`。

## 8.2 通信开销

| 来源 | 通信类型 | group | 频率 | 说明 |
|---|---|---|---|---|
| Transformers TP 常规层 | 取决于 Colwise/Rowwise/DTensor | `tp` | 每层/每模块 | 由 Transformers/PyTorch DTensor 管理，Axolotl 不直接实现 |
| 标准 `lm_head colwise_gather_output` | all-gather logits | `tp` | 每 forward | CCE 分支绕过，因为不调用 `lm_head(...)` |
| CCE vocab-parallel forward | all-reduce MAX、all-reduce SUM、all-reduce correct logit | `tp` | 每 loss forward | buffer 主要是 `[T]` |
| CCE vocab-parallel backward | all-reduce hidden grad `de` | `tp` | 每 backward | buffer 是 `[T,H]` |
| CP pre-hook token count | all-reduce AVG | `cp` | 有 `num_items_in_batch` 时每 forward | `sequence_parallel.py:150-165` |
| CP ring attention | ring P2P / collectives（下游 ring_flash_attn） | `cp` | 每 attention layer | Axolotl 注册 group，通信在 ring_flash_attn 内 |
| CP eval loss 修正 | all-reduce SUM loss 与 token count | `cp` | eval forward | `sequence_parallel.py:305-340` |
| CP output gather（GRPO/EBFT） | all-gather + backward slice | `cp` | RL output gather 时 | SFT 默认不启用 |
| FSDP2 | parameter all-gather / reduce-scatter 等 | `dp_shard_cp` | 每层/每 step | 由 PyTorch FSDP2 管理 |
| FSDP2 save | `DTensor.full_tensor()` + barrier | FSDP mesh | checkpoint/save | `fsdp2.py:158-173`，可能串行瓶颈 |

CCE 的通信取舍很明确：用几个 TP all-reduce 换掉完整 vocab logits all-gather。前向通信从 `[T,V]` 级别降到多个 `[T]` 向量；反向仍有 `[T,H]` 的 hidden gradient all-reduce，但这通常比 `[T,V]` 小得多，尤其当 `V >> H`。

## 8.3 性能取舍

这套实现的收益与代价可以概括为：

```text
收益:
  - 避免完整 logits 显存峰值
  - TP 下避免 lm_head output all-gather
  - 不需要 Axolotl 自己实现 TP 切分规则

代价:
  - 每个 loss forward 多个 TP all-reduce
  - backward 对 hidden grad 做 TP all-reduce
  - CCE fork 依赖 DTensor placement / mesh 命名 / Transformers patch 结构
  - 训练中 outputs.logits=None，兼容性下降
```

这不是“免费省显存”。它是典型的 **通信换显存 + patch 复杂度换低侵入集成**。

## 8.4 关键细节与误区澄清

> 容易误解点十七：CCE+TP 会减少所有训练显存。

不会。CCE 只解决 loss 侧 logits 与相关 buffer；CP 解决序列激活；FSDP/ZeRO 解决参数/梯度/optimizer state。注意力 KV、中间 MLP 激活、optimizer state 都不由 CCE 直接处理。

> 容易误解点十八：CCE+TP 的性能一定更快。

不一定。它通常减少显存和大 logits 操作，但增加 TP all-reduce，并引入自定义 Triton / torch.compile kernel。小 vocab、短序列、慢互联或通信拥塞时，收益可能不明显。

## 8.5 本章小结

> 💡 **小结**
>
> * CCE+TP 最大收益是避免 `[tokens, vocab]` 级别 logits 物化和 gather。
> * 新增通信主要是 TP all-reduce：前向 `[T]`，反向 `[T,H]`。
> * CP/FSDP/TP/CCE 各管一段显存瓶颈，不能把某一个开关理解成全局显存优化。

# 九、配置项、边界条件与坑点

## 9.1 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `plugins: [CutCrossEntropyPlugin]` | `cli/config.py:208-220`、`integrations/config.py:27-57` | 注册 CCE plugin 并合并 `CutCrossEntropyArgs` | 没有 plugin 时 `cut_cross_entropy` 不会触发 patch |
| `cut_cross_entropy: true` | `integrations/cut_cross_entropy/__init__.py:86-103` | 模型加载前调用 `cce_patch()` | 需要 Axolotl fork；普通 upstream CCE 会被 `_check_requirements()` 拒绝（同文件 `74-84`） |
| `bf16` / `fp16` | `cut_cross_entropy/args.py:35-44` | 允许 CCE backward | 未开启半精度会直接校验失败 |
| `tensor_parallel_size > 1` | `trainer.py:623-626`、`model.py:749-755` | 写 TP env；`from_pretrained(tp_plan="auto")` | 依赖模型 `_tp_plan`；8-bit bnb optimizer 被禁止（`validation.py:1600-1608`） |
| `context_parallel_size > 1` | `trainer.py:632-638`、`train.py:205-220` | 写 CP env；进入 `SequenceParallelContextManager` | 必须 `flash_attention: true`；需要 `ring_flash_attn`（`validation.py:1516-1550`） |
| `dp_shard_size` / `dp_replicate_size` | `distributed.py:319-370` | 决定 mesh 中 DP/FSDP/HSDP 维度 | `dp_shard_size` 无 FSDP 会报错（`distributed.py:347-352`） |
| `fsdp_config` + `fsdp_version: 2` | `patch_manager.py:270-299`、`fsdp2.py:279-449` | 使用 FSDP2 patch；mesh 用 `fsdp_dim_names` | 保存/加载时 DTensor full/gather 可能有峰值 |
| `deepspeed` + `tensor_parallel_size` | `validation.py:1121-1148` | 自动注入 `tensor_parallel.autotp_size` 与保存 gather 配置 | 修改的是临时 JSON 路径，不是用户原文件 |
| `chunked_cross_entropy` | `cut_cross_entropy/args.py:46-54`、`validation.py:974-1002` | 与 CCE 互斥 | 不能同时启用两个 CE 优化 |
| `liger_fused_linear_cross_entropy` | `liger/args.py:108-113` | Liger FLCE 路径 | TP 下被显式禁止；CCE fork 反而有 DTensor VP 分支 |
| `sequence_parallel_degree` | `validation.py:1508-1515` | deprecated，迁移到 `context_parallel_size` | 容易和 Accelerate `sp_size` 混淆；Axolotl CP 实际用 `cp_size` |
| `trust_remote_code` | `cut_cross_entropy/__init__.py:100-103` | CCE patch 传 `remote_model_id` | 下游 remote patch 支持不完整时可能失败或不命中 |

## 9.2 最小可用配置

只看 CCE+TP vocab-parallel loss，最小核心配置形态是：

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
cut_cross_entropy: true
bf16: true

tensor_parallel_size: 2
# 通常还需要适合 TP 的模型、Transformers _tp_plan、launcher num_processes >= 2
```

若叠加 CP/FSDP：

```yaml
fsdp_version: 2
fsdp_config:
  state_dict_type: FULL_STATE_DICT
  reshard_after_forward: true

dp_shard_size: 2
context_parallel_size: 2
flash_attention: true
micro_batch_size: 1  # sample_packing 场景下 CP 要求
```

## 9.3 静默失效与硬失败

- **硬失败**：未安装 Axolotl fork 的 CCE，`_check_requirements()` 抛 ImportError（`src/axolotl/integrations/cut_cross_entropy/__init__.py:61-84`）。
- **硬失败**：`cut_cross_entropy` 开启但无 bf16/fp16，Pydantic validator 报错（`args.py:35-42`）。
- **硬失败**：`chunked_cross_entropy` 与 CCE 同时启用（`args.py:46-54`）。
- **硬失败**：TP + 8-bit bnb optimizer（`validation.py:1600-1608`）。
- **可能静默不达预期**：模型 `_tp_plan` 没有把 `lm_head` 切成 DTensor，则 CCE 不走 vocab-parallel 分支，只是普通 CCE。
- **可能静默不达预期**：自定义/remote 模型 forward class 不被 CCE patch 命中；Axolotl generic fallback 只尝试 `{Prefix}ForCausalLM.forward`（`src/axolotl/integrations/cut_cross_entropy/__init__.py:114-150`）。
- **运行时报错**：如果 `apply_lce()` 没有提前 unwrap DTensor，`linear_cross_entropy()` 会拒绝 DTensor 输入（`cut_cross_entropy/linear_cross_entropy.py:62-67`）。当前 fork 只显式处理 `c`，bias DTensor 是潜在风险；多数 causal LM `lm_head` 是 `bias=False`。

## 9.4 本章小结

> 💡 **小结**
>
> * CCE+TP 的最小有效条件是：plugin 注册、半精度、TP mesh、模型 `_tp_plan` 让 `lm_head.weight` 成为 vocab-sharded DTensor。
> * Axolotl 对 Liger+TP 做了显式禁止，但 CCE fork 专门支持 DTensor vocab-parallel。
> * `context_parallel_size` 使用 Accelerate 的 `cp_size` 命名，不等同于 DeepSpeed ALST `sp_size`。

# 十、测试、示例与覆盖缺口

## 10.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/e2e/integrations/test_cut_cross_entropy.py:53-66` | Llama/SmolLM CCE e2e 训练并检查输出 | 覆盖 plugin、patch、训练、保存基本路径；不是 TP/CP |
| `tests/e2e/integrations/test_cut_cross_entropy.py:68-110` | Qwen2.5 CCE e2e | 覆盖另一个模型族的 CCE patch |
| `tests/e2e/integrations/test_cut_cross_entropy.py:112-138` | CCE + flash/sdpa attention | 覆盖 attention 配置组合；不是 vocab-parallel |
| `tests/test_tensor_parallel_batch_size.py:28-55` | TP 下 batch_size 按 effective DP world size 缩放 | 覆盖 `normalize_config()` 的 TP batch 语义 |
| `tests/test_context_parallel_batch_size.py:29-56` | CP 下 batch_size 缩放 | 覆盖 CP batch 语义，mock `ring_flash_attn` |
| `tests/test_loaders.py:181-218` | `_get_parallel_config_kwargs()` 组合 | 覆盖 TP/CP/DP/FSDP kwargs 推导 |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:3-19` | CCE + FSDP2 + TP + CP 示例 | 提供官方推荐配置形态，但不是测试断言 |
| `examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml:3-22` | CCE + HSDP + TP 示例 | 覆盖文档级配置组合 |

## 10.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| CCE vocab-parallel DTensor 分支是否真实触发 | 未见 Axolotl 本地测试 | 用户以为省掉 TP logits gather，实际模型未 TP 或未命中 DTensor 分支 |
| CCE + `tensor_parallel_size > 1` e2e | TP e2e 当前 skip | regressions 难以及时发现 |
| CCE + CP + TP 同时训练 | 仅有示例，未见 e2e | CP group / TP group 混用、shape/padding 问题可能漏测 |
| CCE + FSDP2 save/resume | 未见专门测试 | full state dict / DTensor / patch resume 组合风险 |
| 多机 TP/CP 拓扑 | 未见多机 e2e | rank map 不贴合硬件拓扑导致通信瓶颈 |
| bias 为 DTensor 的 `lm_head` | 未见测试 | `linear_cross_entropy()` 可能拒绝 DTensor bias |
| outputs.logits 依赖方 | 未见兼容性测试 | callback/metric/custom Trainer 在 CCE 下拿到 `None` |
| patch 恢复 / 多模型同进程 | 未见隔离测试 | 类级 patch 污染后续模型 |

两个 skip 尤其值得注意：

- `tests/e2e/multigpu/test_tp.py:17-20` 的 TP SFT 测试被 skip，原因是 tied weights 模型 TP 不工作；
- `tests/e2e/multigpu/patched/test_sp.py:102-120` 的 sequence parallel e2e 被 skip，原因是 `ring_flash_attn` 与 Transformers imports 上游维护问题。

这意味着本文分析的组合路径，虽然有源码支撑和示例配置，但测试保护并不充分。

## 10.3 本章小结

> 💡 **小结**
>
> * CCE 单独路径有 e2e；TP/CP 的 batch 和 config 有单测。
> * CCE+TP vocab-parallel loss 这个关键交汇点，在 Axolotl 本地测试中未见直接覆盖。
> * 当前最薄弱的是多维组合 e2e、保存/resume、以及 patch 是否真实命中 DTensor 分支。

# 十一、局限性与已知优化点

## 11.1 硬约束

- CCE 要求 PyTorch >= 2.4，且安装 Axolotl fork；`_check_requirements()` 会检查 `AXOLOTL_CCE_FORK`（`src/axolotl/integrations/cut_cross_entropy/__init__.py:50-84`）。
- CCE backward 要求 fp16/bf16 配置（`src/axolotl/integrations/cut_cross_entropy/args.py:35-42`）。
- CP 要求 `flash_attention: true` 和 `ring_flash_attn`（`src/axolotl/utils/schemas/validation.py:1516-1550`）。
- TP 不支持 `paged_adamw_8bit`、`adamw_8bit`、`adamw_bnb_8bit`（`validation.py:1600-1608`）。
- vocab-parallel CCE 假设 `lm_head.weight` DTensor 是 vocab 维 shard：外部 `apply_lce()` 固定 `vocab_dim = 0`，并取 `device_mesh.get_group("tp")`（`cut_cross_entropy/transformers/utils.py:113-129`）。
- `linear_cross_entropy()` 明确拒绝 DTensor 输入（`cut_cross_entropy/linear_cross_entropy.py:62-67`），因此 `apply_lce()` 的 unwrap 逻辑是必要前置。

## 11.2 维护成本

这套实现跨了四个上游边界：

```text
Axolotl plugin / ModelLoader
  -> Transformers model class forward patch
  -> Transformers tp_plan / tensor_parallel integration
  -> PyTorch DTensor / DeviceMesh
  -> CCE fork Triton/autograd kernel
```

任一层变化都可能破坏行为。例如：

- Transformers 模型 forward 签名变化，会影响 CCE fork 的模型专用 patch；
- `_tp_plan` 命名或 `colwise_gather_output` 语义变化，会影响 `lm_head.weight` placement；
- DTensor `device_mesh.get_group("tp")` API 或 mesh dim name 变化，会影响 CCE 获取 process group；
- Axolotl 的 generic CCE fallback 对未列入 `PATCH_FNS` 的模型只是实验性支持，并打印 warning（`src/axolotl/integrations/cut_cross_entropy/__init__.py:142-150`）。

## 11.3 性能瓶颈

- **TP loss all-reduce 串行化**：forward 至少有 LSE max、LSE sum、correct logit 三类归约；backward 有 `de` all-reduce。当前源码未显示这些通信与其他计算 overlap。
- **CP ring attention 每层通信**：长序列场景 CP 降低激活，但每层 attention 要在 CP group 内传 KV。
- **FSDP2 save full tensor**：`get_state_dict()` 对每个 DTensor 调 `full_tensor()` 后 rank0 放 CPU，并有 barrier（`src/axolotl/monkeypatch/accelerate/fsdp2.py:158-173`），大模型保存可能成为串行瓶颈。
- **rank map 与硬件拓扑未自动匹配**：Accelerate mesh 使用 canonical order 和 row-major rank map；跨节点使用 TP/CP 可能踩慢互联。
- **patch 不恢复**：长生命周期 Python 进程可能因为类级 patch 和全局 group 状态产生意外复用。

## 11.4 已知优化点

源码中可以看到几个 TODO / 改进方向：

- `ModelLoader._build_model()` 对 Transformers 4.54.0 的 `_tp_size/_device_mesh` workaround 标注 TODO（`src/axolotl/loaders/model.py:852-857`），上游修复后应移除；
- `SequenceParallelContextManager.__exit__()` TODO 提到未 un-patch attention 和 accelerate functions（`src/axolotl/utils/ctx_managers/sequence_parallel.py:238-245`）；
- `liger_fused_linear_cross_entropy` 与 TP 目前被禁止，注释写 “larger fix - investigate”（`src/axolotl/integrations/liger/args.py:108-113`），而 CCE fork 已经提供了一个可借鉴的 DTensor vocab-parallel loss 接入方式；
- CCE vocab-parallel all-reduce 可以考虑更细粒度 overlap，但当前源码未确认已有 overlap；
- 保存路径可以探索分块 full_tensor / 异步 CPU offload / sharded-safe final artifact，避免 rank0 峰值。

## 11.5 本章小结

> 💡 **小结**
>
> * 当前实现依赖 CCE fork、Transformers TP、PyTorch DTensor 和 Accelerate mesh 的多层契约。
> * 最大维护风险来自 monkey patch 与上游 forward / `_tp_plan` / DTensor API 变化。
> * 最大性能风险来自 TP/CP/FSDP 多维通信叠加，以及保存时 full tensor 聚合。

# 小结与展望

Axolotl 的 “CCE fork vocab-parallel loss × ND parallelism / TP DeviceMesh” 实现，可以用五个关键词概括。

## 关键词一：YAML 驱动的命名 mesh

用户只写 `tensor_parallel_size/context_parallel_size/dp_*`，Axolotl 通过环境变量和 `build_parallelism_config()` 把它们变成 Accelerate / PyTorch `DeviceMesh`。真正重要的是维度命名：CCE 找 `tp`，CP 找 `cp`，FSDP2 找 `dp_shard_cp`。

## 关键词二：类级 forward patch

CCE 没有改 Trainer，也没有全局替换 `torch.nn.functional.cross_entropy`。它在模型加载前替换 `ForCausalLM.forward`，让模型在 labels 存在时直接返回 loss，并让 `logits=None`。这低侵入，但带来 patch 恢复和 forward 签名维护成本。

## 关键词三：DTensor bridge

CCE fork 的关键增强是 `apply_lce()` 能识别 `lm_head.weight` 是 DTensor：从 DTensor 拿 `device_mesh.get_group("tp")`，从 placement 判断 shard，再 `to_local()` 取本地 vocab 权重。Axolotl 不需要显式把 TP group 传给 CCE，DTensor 成了两套系统之间的桥。

## 关键词四：all-reduce 替代 logits all-gather

标准 TP `lm_head` 的 `colwise_gather_output` 会 all-gather 完整 logits；CCE patched forward 绕过 `lm_head(...)`，在本地 vocab shard 上计算 LSE/correct logit，并用 TP all-reduce 恢复全局 softmax。它省掉 `[tokens, vocab]`，但反向仍要 all-reduce `[tokens, hidden]` 的 `dE`。

## 关键词五：通信换显存

这套实现适合长序列、大词表、大模型、多 GPU 训练：CP 降序列激活，TP 降输出层参数，CCE 降 logits 峰值，FSDP 降参数/optimizer state。它不适合小模型短序列、慢互联、多 patch 混用、或者强依赖 `outputs.logits` 的自定义训练逻辑。

与替代方案相比，CCE fork 的优势是能在 TP DTensor 下避免完整 logits gather；Liger FLCE 当前在 Axolotl 中被禁止与 TP loss 组合。代价是维护一个紧贴 Transformers forward 和 DTensor placement 的 fork。后续值得继续走读的方向包括：Transformers `_tp_plan` 在更多模型上的具体切分、FSDP2 + TP + LoRA 的 DTensor 类型桥接、CP ring attention 的 forward/backward 通信细节，以及保存/resume 在多维 mesh 下的峰值内存与性能。

最终判断：**这不是一个“打开 CCE 就完事”的优化，而是 Axolotl 用 DeviceMesh 把多个并行维度命名，再让 CCE fork 借 DTensor 找到 TP 维度，从而在 loss 层精确避开 logits all-gather。** 它的设计很工程化，也很脆弱：收益来自跨库协作，风险也来自跨库协作。
