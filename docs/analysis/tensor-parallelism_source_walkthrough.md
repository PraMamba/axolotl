# Axolotl 源码走读：Tensor Parallelism (TP) 实现解析

在大规模语言模型的微调场景中，模型参数量的增长速度远远超出单卡显存的扩展速度。FSDP 通过参数分片解决了存储问题，但每个 GPU 在前向/反向计算时仍然需要持有完整的单层参数。当单层参数量足够大（例如 70B 模型的 attention 和 FFN 线性层），即使 FSDP 也无法让单卡装下一层完整的参数和激活值。Tensor Parallelism（TP）正是为了解决这个问题而引入的——它将每一层的线性变换**按列或按行切分到多张卡上**，使得每张卡只需要计算和存储部分参数与部分激活值。

本文不展开 Megatron-LM 风格的 TP 原理，也不深入 PyTorch DTensor 的底层实现，而是聚焦 Axolotl 源码，分析这个微调框架如何将 TP 接入现有的训练链路，它在工程上做了哪些适配、打了哪些 patch、踩了哪些坑，以及当前实现的收益和局限。

---

# 前言

## 业务 / 工程背景

Axolotl 是一个以 YAML 配置驱动的 LLM 微调框架，构建在 HuggingFace Transformers、TRL、PEFT 和 Accelerate 之上。它的核心价值在于让用户只写一份 YAML，就能完成 SFT、LoRA、DPO、GRPO 等各种训练任务。但随着用户需要微调的模型越来越大（8B、70B、甚至 405B），单卡和纯 FSDP 都不够用了。TP 作为一种**层内并行**方案，可以在多卡之间切分每一层的线性变换，是突破单层显存瓶颈的关键手段。

Axolotl 从 2024 年底开始引入 TP 支持，目前标记为 **[Experimental]** 状态。

## 核心矛盾

Axolotl 的 TP 实现面临的核心工程矛盾是：

1. **框架自身不实现 TP，而是依赖 PyTorch DTensor + HuggingFace Transformers 的 `tp_plan="auto"`**。这意味着 Axolotl 需要在不侵入 TP 核心计算的前提下，把 TP 正确地"接入"现有的模型加载、FSDP 分片、LoRA 适配、状态保存等链路。
2. **FSDP2 和 TP 操作在同一个多维 DeviceMesh 上，但维度正交**。FSDP 在数据并行维度上分片，TP 在模型维度上分片。两者的参数都变成了 DTensor，但分片语义完全不同，状态保存和恢复时需要正确处理两种分片的叠加。
3. **LoRA/PEFT 适配器不是 DTensor 原生的**。当底层参数是 DTensor、LoRA 增量是普通 Tensor 时，两者的加法运算会崩溃。Axolotl 需要在 PEFT 层面打 patch 来桥接这个类型不匹配。

## 本文主线

本文分为以下几个核心机制章节：

1. **配置归一化与环境注入**：用户 YAML 如何变成运行时并行配置
2. **DeviceMesh 构建与维度分配**：多维 mesh 如何拆分 TP / DP / CP 维度
3. **模型加载与 TP 分片**：`tp_plan="auto"` 在 `from_pretrained` 中如何生效
4. **FSDP2 与 TP 的正交组合**：mesh 切片、模型准备、LoRA 兼容
5. **状态保存与恢复**：DTensor 的 `full_tensor()` 和 `distribute_tensor()` 如何处理双重分片

## 不展开的内容

- Megatron-LM 的 Column/Row Parallel Linear 原理
- PyTorch DTensor 的底层调度和通信实现
- HuggingFace Transformers 各模型的 `_tp_plan` 定义
- FSDP2 的 shard/unshard 机制
- LoRA/QLoRA 原理

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/utils/schemas/config.py` | 定义 `tensor_parallel_size` 配置字段 |
| `src/axolotl/utils/schemas/validation.py` | TP 相关的配置校验与 DeepSpeed 自动注入 |
| `src/axolotl/utils/trainer.py` | 通过环境变量向 Accelerate 注入并行维度 |
| `src/axolotl/utils/distributed.py` | 构建 `ParallelismConfig` 和 `DeviceMesh` |
| `src/axolotl/loaders/model.py` | 模型加载时传入 `tp_plan="auto"` |
| `src/axolotl/loaders/utils.py` | `tie_word_embeddings` 兼容性校验 |
| `src/axolotl/monkeypatch/accelerate/fsdp2.py` | FSDP2 模型准备、mesh 切片、状态保存/恢复、LoRA DTensor 兼容 |
| `src/axolotl/monkeypatch/accelerate/parallelism_config.py` | 放宽 Accelerate 的 ParallelismConfig 校验 |
| `src/axolotl/utils/config/__init__.py` | batch size 按有效 DP 维度缩放 |

---

# 一、配置归一化与环境注入：用户 YAML 如何变成运行时并行配置

## 1.1 设计哲学与核心问题

Axolotl 的核心设计原则是"一切通过 YAML 配置驱动"。用户开启 TP 只需在 YAML 中加一行 `tensor_parallel_size: 2`，不需要修改训练脚本。但这行配置需要经过**三层转换**才能真正影响运行时行为：

1. **Pydantic 校验层**：类型检查、默认值填充、兼容性校验
2. **环境变量注入层**：将配置写入 `os.environ`，供 Accelerate 读取
3. **运行时对象构建层**：根据配置构建 `ParallelismConfig` 和 `DeviceMesh`

如果没有这套转换管线，用户配置和实际并行行为之间就会出现断层——要么框架不知道用户要求 TP，要么底层库不知道该在哪个维度上做分片。

## 1.2 源码入口与关键对象

```text
src/axolotl/utils/schemas/config.py
  - tensor_parallel_size: int | None（字段定义，line 993）

src/axolotl/utils/schemas/validation.py
  - check_tensor_parallel_size（默认值归一化，line 1501）
  - check_tensor_parallel_size_update_ds_json（DeepSpeed 配置自动注入，line 1119）
  - check_tensor_parallel_optimizer（8-bit 优化器拦截，line 1600）

src/axolotl/utils/trainer.py
  - setup_parallelism_envs（环境变量注入，line 621）

src/axolotl/loaders/utils.py
  - check_model_config（tie_word_embeddings 校验，line 139）
```

## 1.3 主流程拆解

### 第一层：Pydantic 校验

用户在 YAML 中写 `tensor_parallel_size: 2` 后，这个值进入 Pydantic V2 的 `AxolotlInputConfig` 进行校验。关键的 model validator 有三个：

**默认值填充**（`validation.py:1501`）：

```python
@model_validator(mode="after")
def check_tensor_parallel_size(self):
    if not self.tensor_parallel_size:
        self.tensor_parallel_size = 1
    return self
```

这个校验器将 `None` 或 `0` 统一归一化为 `1`，确保后续代码可以直接用 `> 1` 判断是否开启 TP，而不需要到处做 None 检查。

**8-bit 优化器拦截**（`validation.py:1600`）：

```python
@model_validator(mode="after")
def check_tensor_parallel_optimizer(self):
    if self.tensor_parallel_size > 1:
        if self.optimizer in ["paged_adamw_8bit", "adamw_8bit", "adamw_bnb_8bit"]:
            raise ValueError(
                "tensor_parallel_size is not supported with paged_adamw_8bit, ..."
            )
```

TP 将参数变为 DTensor，而 bitsandbytes 的 8-bit 优化器不支持对 DTensor 做量化状态管理，因此必须在配置阶段就拦截。

**DeepSpeed 配置自动注入**（`validation.py:1119`）：

当用户同时配置了 `tensor_parallel_size > 1` 和 `deepspeed` JSON 文件时，Axolotl 会自动将 `tensor_parallel.autotp_size` 注入 DeepSpeed 配置中，并确保 `gather_16bit_weights_on_model_save` 为 `True`。这个逻辑直接修改临时文件中的 JSON，然后将配置路径替换为临时文件路径。

### 第二层：环境变量注入

校验通过后，`prepare_optim_env()` 调用 `setup_parallelism_envs()`（`trainer.py:621`），将并行维度写入环境变量：

```python
def setup_parallelism_envs(cfg):
    set_accelerate_parallelism_config = False
    if cfg.tensor_parallel_size and cfg.tensor_parallel_size > 1:
        set_accelerate_parallelism_config = True
        os.environ["PARALLELISM_CONFIG_TP_SIZE"] = str(cfg.tensor_parallel_size)
    # ... dp_shard_size, dp_replicate_size, context_parallel_size 类似
    if set_accelerate_parallelism_config:
        os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
```

这些环境变量是 Axolotl 和 Accelerate 之间的通信协议。当 Accelerate 初始化 `PartialState` / `Accelerator` 时，会读取这些变量来构建自己的 `ParallelismConfig` 并存入 `accelerator.state`。

### 第三层：模型级校验

在模型配置加载阶段，`check_model_config()`（`loaders/utils.py:139`）检查 `tie_word_embeddings`：

```python
if (
    cfg.tensor_parallel_size and cfg.tensor_parallel_size > 1
    and hasattr(model_config, "tie_word_embeddings")
    and model_config.tie_word_embeddings
):
    raise ValueError(
        "Tensor parallelism is incompatible with models configured with "
        "`tie_word_embeddings` enabled."
    )
```

这是一个**运行时校验**而非配置校验——因为它需要读取模型的 HuggingFace config 才能知道是否 tie embeddings。

## 1.4 关键细节与误区澄清

> **误区一：`tensor_parallel_size` 的 schema 描述说 "Only supported with DeepSpeed AutoTP"，但实际上 FSDP2 也支持 TP。**
>
> schema 定义处（`config.py:993`）的 `json_schema_extra` 描述是过时的。从源码看，TP 同时支持两条路径：DeepSpeed AutoTP（通过自动注入 `tensor_parallel.autotp_size`）和 FSDP2 + HuggingFace Transformers 的 `tp_plan="auto"`。文档 `docs/nd_parallelism.qmd` 也明确列出了 FSDP+TP 的支持矩阵。这里 schema 描述和实际实现不一致，应以源码为准。

> **误区二：DeepSpeed 配置自动注入不是"可选的"——它会修改实际使用的配置文件路径。**
>
> `check_tensor_parallel_size_update_ds_json`（`validation.py:1121`）不只是校验，它会创建临时文件并将 `data["deepspeed"]` 替换为新路径。这意味着用户原始的 DeepSpeed JSON 不会被修改，但训练时使用的是框架临时生成的版本。如果用户依赖 DeepSpeed 的 `_consolidated_16bit_state_dict()` 保存，这个自动注入是前提条件。

> **误区三：`tie_word_embeddings` 不兼容并非 Axolotl 的限制，而是 PyTorch TP 的根本限制。**
>
> 当 embedding 和 lm_head 共享同一个物理 weight tensor 时，TP 的 `tp_plan` 需要对 embedding 做 column-wise 切分、对 lm_head 做 row-wise 切分——但同一个 tensor 不可能同时有两种不同的切分方式。这是 DTensor placement 语义上的矛盾，不是 Axolotl 的实现缺陷。

## 1.5 本章小结

> 💡 **小结**
>
> - 用户只需在 YAML 中设置 `tensor_parallel_size: N` 即可开启 TP，经过 Pydantic 校验、环境变量注入、模型配置校验三层转换后生效。
> - 环境变量 `PARALLELISM_CONFIG_TP_SIZE` 和 `ACCELERATE_USE_PARALLELISM_CONFIG` 是 Axolotl 与 Accelerate 之间的通信桥梁。
> - TP 与 `tie_word_embeddings` 模型不兼容，与 8-bit 优化器不兼容——这些限制在配置校验阶段就会被拦截。
> - schema 描述称 "Only supported with DeepSpeed AutoTP" 是过时的，实际同时支持 FSDP2 路径。

---

# 二、DeviceMesh 构建与维度分配：多卡如何被划分为 TP / DP / CP 组

## 2.1 设计哲学与核心问题

当一个训练任务同时使用 TP、CP 和 FSDP 时，8 张 GPU 可能需要被划分为：2 个 TP rank × 2 个 CP rank × 2 个 DP shard rank。这种多维并行要求一个统一的"坐标系"来描述每张 GPU 在每个并行维度上的角色。

PyTorch 的 `DeviceMesh` 正是这个坐标系。它是一个多维数组，每个维度对应一种并行策略，每个元素是一个 GPU 的 rank 编号。Axolotl 需要解决的问题是：**给定用户配置的各维度大小和 world_size，如何正确构建这个多维 mesh？**

## 2.2 源码入口与关键对象

```text
src/axolotl/utils/distributed.py
  - build_parallelism_config：构建 ParallelismConfig + DeviceMesh（line 299）
  - _get_parallel_config_kwargs：计算各维度大小（line 319）
```

## 2.3 主流程拆解

`_get_parallel_config_kwargs()` 的维度分配算法是一个**贪心减法过程**：

```text
输入: world_size=8, tp_size=2, cp_size=2, dp_shard_size=None, dp_replicate_size=None

Step 1: 分配 TP → remaining = 8 // 2 = 4
Step 2: 分配 CP → remaining = 4 // 2 = 2
Step 3: dp_shard_size 和 dp_replicate_size 都未指定，自动将 remaining 分给 dp_shard
        → dp_shard_size = 2, remaining = 1
Step 4: remaining == 1，分配完毕

输出: {tp_size: 2, cp_size: 2, dp_shard_size: 2}
```

分配顺序是固定的：**TP → CP → dp_replicate → dp_shard → 兜底**。如果最终 `remaining > 1`，抛出 `ValueError`，说明配置的并行维度之积与 world_size 不匹配。

构建完 kwargs 后，`build_parallelism_config()` 调用 Accelerate 的 `ParallelismConfig(**pc_kwargs)` 和 `.build_device_mesh("cuda")` 来创建实际的 `DeviceMesh`。Accelerate 内部会根据维度名称（如 `"dp_shard"`, `"tp"`, `"cp"`）创建一个多维 tensor，其中每个元素是对应位置的 global rank。

以 8 GPU、`(dp_shard=2, tp=2, cp=2)` 为例，mesh 的逻辑结构为：

```text
DeviceMesh("cuda", shape=(2, 2, 2), dim_names=("dp_shard", "tp", "cp"))

rank 布局（一种可能）:
dp_shard=0, tp=0, cp=0 → rank 0
dp_shard=0, tp=0, cp=1 → rank 1
dp_shard=0, tp=1, cp=0 → rank 2
dp_shard=0, tp=1, cp=1 → rank 3
dp_shard=1, tp=0, cp=0 → rank 4
dp_shard=1, tp=0, cp=1 → rank 5
dp_shard=1, tp=1, cp=0 → rank 6
dp_shard=1, tp=1, cp=1 → rank 7
```

同一个 TP group 内的 rank（如 rank 0 和 rank 2）会处理相同的数据，但各自持有参数的不同切片。

## 2.4 关键细节与误区澄清

> **误区四：Axolotl 构建了两次 DeviceMesh，但作用不同。**
>
> 第一次是 `setup_parallelism_envs()` 通过环境变量告知 Accelerate，Accelerate 在 `Accelerator.__init__()` 时自己构建 mesh 并存入 `accelerator.state.device_mesh`。第二次是 `ModelLoader._set_parallel_config()` 直接调用 `build_parallelism_config()` 构建 mesh 传给 `from_pretrained()`。两次构建使用相同的配置，产生相同的 mesh，但服务于不同的消费者：前者给 Accelerate/FSDP2 用，后者给 Transformers 的 TP 初始化用。

> **关键约束：world_size 必须等于所有并行维度之积。**
>
> `_get_parallel_config_kwargs()` 在分配完所有维度后检查 `remaining_world_size`。如果不能整除或有剩余，直接报错。这意味着用户不能随意组合维度——例如 6 卡环境下无法配置 `tp_size=4`。

## 2.5 本章小结

> 💡 **小结**
>
> - DeviceMesh 是 TP/CP/FSDP 多维并行的统一坐标系，由 `_get_parallel_config_kwargs()` 按 TP → CP → DP 的优先级分配维度。
> - Axolotl 构建两次 mesh：一次通过环境变量给 Accelerate 使用，一次直接传给 `from_pretrained()` 进行 TP 初始化。两次结果一致。
> - world_size 必须等于各维度之积，否则配置校验直接报错。

---

# 三、模型加载与 TP 分片：`tp_plan="auto"` 如何让每张卡只持有部分参数

## 3.1 设计哲学与核心问题

这是 Axolotl TP 实现中最核心的设计决策：**Axolotl 自己不实现任何 TP 切分逻辑**。它将 `tp_plan="auto"` 和 `device_mesh` 传给 HuggingFace Transformers 的 `from_pretrained()`，由 Transformers 调用 PyTorch 的 `parallelize_module()` 来完成实际的参数切分。

这个决策的好处是显而易见的——Axolotl 不需要为每种模型架构编写 TP 切分规则，HuggingFace 各模型类自带 `_tp_plan`（如 `LlamaForCausalLM._tp_plan`）。但代价是 Axolotl 对 TP 的控制力很弱：它无法自定义切分策略，也无法在不修改 Transformers 的情况下支持 Transformers 尚未适配的模型。

## 3.2 源码入口与关键对象

```text
src/axolotl/loaders/model.py
  - ModelLoader（类定义，line 85）
    - use_parallel_config / parallelism_config / device_mesh（class attrs，line 94-96）
    - _set_parallel_config（构建 mesh，line 437）
    - _apply_pre_model_load_setup（判断是否启用 parallel config，line 196）
    - _build_model（传入 tp_plan="auto"，line 749）
    - post-build workaround（修复 transformers 4.54.0 bug，line 852）
```

## 3.3 主流程拆解

模型加载的完整 TP 调用链：

```text
ModelLoader.load()
  └── _apply_pre_model_load_setup()                    [line 196]
        ├── 判断 use_parallel_config = True             [当 tp_size > 1]
        └── _set_parallel_config()                      [line 437]
              └── build_parallelism_config(cfg)          [构建 mesh]

  └── _build_model()                                    [line 745]
        ├── 设置 model_kwargs:
        │     tp_size = cfg.tensor_parallel_size         [line 750]
        │     tp_plan = "auto"                           [line 751]
        │     device_mesh = self.device_mesh             [line 752]
        │     删除 device_map（与 tp_plan 不兼容）        [line 753-754]
        │
        ├── AutoModelForCausalLM.from_pretrained(**model_kwargs)
        │     └── [Transformers 内部]
        │           └── parallelize_module(model, device_mesh["tp"], tp_plan)
        │                 └── 对每个匹配的 Linear 层应用 ColwiseParallel / RowwiseParallel
        │                       └── weight → DTensor（按 TP 维度分片）
        │
        └── 后处理 workaround                            [line 852-857]
              └── 修复 model._tp_size 和 model._device_mesh
```

**关键步骤详解**：

当 `tp_size > 1` 时，`_build_model()` 做了三件事：

1. **设置 `tp_plan="auto"`**：告诉 Transformers 使用模型类自带的 `_tp_plan` 字典来自动切分参数。每个模型类（如 LlamaForCausalLM）定义了哪些层用 ColumnParallel、哪些用 RowParallel。
2. **传入 `device_mesh`**：Transformers 从这个 mesh 中提取 TP 维度（通常是 `mesh["tp"]`），构建 TP 维度的 process group。
3. **删除 `device_map`**：`device_map`（用于多卡放置模型层）和 `tp_plan` 互斥。TP 通过 DTensor 管理参数分布，不需要也不能和 `device_map` 共存。

**CPU RAM efficient loading 与 TP 的冲突**：

正常情况下，FSDP2 配合 `cpu_ram_efficient_loading` 时，rank 0 将参数加载到 CPU、其他 rank 加载到 meta device，然后通过广播分发。但这个逻辑依赖 `device_map` 来指定初始设备，而 TP 删除了 `device_map`。因此当 TP 开启时，`cpu_ram_efficient_loading` 的 `device_map` 路径被跳过：

```python
# model.py:769-779
if (
    self.cfg.tensor_parallel_size <= 1          # ← 只在非 TP 时设置 device_map
    and self.cfg.fsdp_config.cpu_ram_efficient_loading
    and self.cfg.fsdp_version == 2
):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank == 0:
        self.model_kwargs["device_map"] = "cpu"
    else:
        self.model_kwargs["device_map"] = "meta"
```

这意味着 **TP + FSDP2 的 `cpu_ram_efficient_loading` 行为和非 TP 时不同**——TP 下模型会直接加载到 GPU 上进行 TP 切分，不走 CPU 中转路径。

## 3.4 关键细节与误区澄清

> **误区五：Axolotl 的 TP 并不是"Axolotl 实现的 TP"。**
>
> Axolotl 在整个 TP 链路中只做了"配置传递"和"兼容性修补"，实际的参数切分、通信原语插入、前向/反向 all-reduce 全部由 PyTorch DTensor 和 HuggingFace Transformers 完成。Axolotl 源码中没有任何自定义的 `ColwiseParallel`、`RowwiseParallel`、`all_gather` 或 `reduce_scatter` 调用。

> **误区六：`model._tp_size` 的 workaround 说明上游 Transformers 的 TP 支持本身仍不完善。**
>
> `model.py:852-857` 中的 TODO 注释写着 "workaround for upstream 4.54.0 not setting _tp_size or _device_mesh"。这意味着即使 Transformers 声称支持 `tp_plan="auto"`，在某些版本中模型对象上的 TP 元信息可能没有被正确设置。Axolotl 通过事后修补来弥补这个缺陷。

> **误区七：`tp_plan="auto"` 不等于"所有模型都支持 TP"。**
>
> `"auto"` 只是让 Transformers 查找模型类的 `_tp_plan` 属性。如果某个模型类没有定义 `_tp_plan`，TP 将不会对任何层生效——参数仍然是完整的，不会被切分。Axolotl 没有对此做任何检查或警告。

## 3.5 本章小结

> 💡 **小结**
>
> - Axolotl 通过 `tp_plan="auto"` 将 TP 切分完全委托给 PyTorch + Transformers，自身不实现任何 TP 计算逻辑。
> - TP 与 `device_map` 互斥，因此 `cpu_ram_efficient_loading` 在 TP 下的行为与非 TP 时不同。
> - Transformers 4.54.0 存在 `_tp_size` 未正确设置的 bug，Axolotl 通过 post-build workaround 修复。
> - `tp_plan="auto"` 并不保证所有模型都能 TP——只有定义了 `_tp_plan` 的模型类才会被切分。

---

# 四、FSDP2 与 TP 的正交组合：mesh 切片、模型准备与 LoRA 兼容

## 4.1 设计哲学与核心问题

FSDP2 和 TP 都将参数变成 DTensor，但它们在不同维度上操作。FSDP2 在数据并行维度上分片（每个 DP rank 持有参数的不同 shard），TP 在模型维度上分片（每个 TP rank 持有每层参数的不同列/行）。两者通过多维 DeviceMesh 实现正交组合：FSDP2 只看 `mesh["dp_shard"]` 维度，TP 只看 `mesh["tp"]` 维度。

但这种正交性在工程上并不是"自动成立"的。Axolotl 需要在 FSDP2 的模型准备阶段正确切片 mesh，确保 `fully_shard()` 只在 DP 维度上操作而不影响 TP 维度。同时，LoRA 适配器作为"非 DTensor 原生"的组件，需要额外的 patch 来兼容。

## 4.2 源码入口与关键对象

```text
src/axolotl/monkeypatch/accelerate/fsdp2.py
  - fsdp2_prepare_model：模型准备主入口（line 279）
  - _process_lora_module_for_fsdp：LoRA 模块 FSDP 处理（line 235）
  - patch_peft_param_wrapper_for_fsdp2：DTensor + Tensor 加法兼容 patch（line 196）
  - patch_accelerate_fsdp2：patch 注册入口（line 529）

src/axolotl/monkeypatch/accelerate/parallelism_config.py
  - _validate_accelerator：放宽 ParallelismConfig 校验（line 11）
  - patch_parallelism_config：patch 注册入口（line 73）
```

## 4.3 主流程拆解

### Mesh 切片：FSDP2 如何"无视" TP 维度

`fsdp2_prepare_model()` 在构建 `fsdp2_kwargs` 时做了关键的 mesh 切片：

```python
# fsdp2.py:344-360
mesh = getattr(accelerator.state, "device_mesh", None)

fsdp2_kwargs = {
    "reshard_after_forward": ...,
    "offload_policy": ...,
    "mp_policy": ...,
    "mesh": (
        mesh[tuple(accelerator.state.parallelism_config.fsdp_dim_names)]
        if mesh is not None
        else None
    ),
}
```

`accelerator.state.parallelism_config.fsdp_dim_names` 返回的是 FSDP 相关的维度名列表——例如 `("dp_shard",)` 或 `("dp_replicate", "dp_shard")`。`mesh[("dp_shard",)]` 从完整的多维 mesh 中提取出只包含 DP shard 维度的子 mesh。这个子 mesh 被传给 `fully_shard()`，确保 FSDP2 只在 DP 维度上做分片。

以 `(dp_shard=2, tp=2)` 的 4 GPU mesh 为例：

```text
完整 mesh:
  dim_names = ("dp_shard", "tp")
  shape = (2, 2)
  [[rank0, rank1],
   [rank2, rank3]]

切片后 FSDP 子 mesh:
  mesh[("dp_shard",)] for rank0: [rank0, rank2]  (dp_shard 维度)
  mesh[("dp_shard",)] for rank1: [rank1, rank3]  (dp_shard 维度)

TP 子 mesh（由 Transformers 在 from_pretrained 时提取）:
  mesh[("tp",)] for rank0: [rank0, rank1]  (tp 维度)
  mesh[("tp",)] for rank2: [rank2, rank3]  (tp 维度)
```

rank 0 的参数在 TP 维度上被切分（和 rank 1 共享一层的不同列），在 DP 维度上也被切分（和 rank 2 持有不同的 FSDP shard）。两种分片在 DTensor 内部表示为多维的 `placements`，例如 `(Shard(0), Shard(1))` 表示第一个维度（DP）按维度 0 分片、第二个维度（TP）按维度 1 分片。

### LoRA DTensor 兼容 patch

当使用 PEFT 的 ParamWrapper（用于 target_parameters / 3D expert params）时，PEFT 内部通过 `torch.nn.utils.parametrize` 注册了一个 `_LoraParameterProxy`，在前向传播时计算 `W + delta_weight`。但在 FSDP2 下，`W` 被 unshard 后是一个 DTensor，而 `delta_weight` 是普通 Tensor——两者的加法会报 `RuntimeError`。

`patch_peft_param_wrapper_for_fsdp2()`（`fsdp2.py:196`）通过替换 `_LoraParameterProxy.forward()` 来解决：

```python
def _patched_forward(self, W):
    delta = self.delta_weight
    w_is_dt = isinstance(W, DTensor)
    d_is_dt = isinstance(delta, DTensor)

    with torch.nn.utils.parametrize.cached():
        if w_is_dt == d_is_dt:     # 类型一致，直接加
            return W + delta
        if w_is_dt:                 # W 是 DTensor，delta 不是
            return W + DTensor.from_local(delta, W.device_mesh, W.placements)
        # delta 是 DTensor，W 不是
        return DTensor.from_local(W, delta.device_mesh, delta.placements) + delta
```

`DTensor.from_local()` 在 `Replicate` placement 下只是元数据包装，不产生通信。这是一个"零成本"的类型适配。

### LoRA 模块的 FSDP 独立分片

`_process_lora_module_for_fsdp()`（`fsdp2.py:235`）对每个 LoRA 模块的 `lora_A`、`lora_B` 和 `lora_magnitude_vector` 分别调用 `fully_shard()`。这确保 LoRA 参数也被 FSDP 管理——在不需要时 reshard 到 shard 状态节省显存，在计算时 unshard 到完整状态。

值得注意的是，`ParamWrapper` 类型的模块被显式跳过（`fsdp2.py:243`）——它的 lora_A/B 不能被独立分片，必须由父层的 FSDP wrapper 统一管理。

### ParallelismConfig 校验 patch

Accelerate 原生的 `ParallelismConfig._validate_accelerator` 要求使用 FSDP 才能开启 ParallelismConfig。但 Axolotl 支持不带 FSDP 的纯 TP 或纯 CP 场景。`parallelism_config.py:11` 的 patch 放宽了这个限制，允许在 `ACCELERATE_ALLOW_CP_STANDALONE=true` 时不带 FSDP 使用 ParallelismConfig。

## 4.4 关键细节与误区澄清

> **误区八：FSDP2 的 `fully_shard()` 不会影响已经被 TP 切分的参数维度。**
>
> 这不是因为 `fully_shard()` 主动避开了 TP 维度，而是因为传给 `fully_shard()` 的 mesh 只包含 DP 维度。DTensor 会在这个子 mesh 上做分片，结果是参数同时有 TP 和 FSDP 两层分片——体现为 DTensor 的 `placements` 包含多个 `Shard` 条目。

> **关键事实：LoRA 的 `lora_A` / `lora_B` 参数在 FSDP2 + TP 下可能既是 DTensor（被 FSDP 管理）又需要与 TP 切分的 base weight 交互。** Axolotl 的 Triton LoRA kernel（`kernels/lora.py:86-88`）通过 `linear_A.unshard()` / `linear_B.unshard()` 手动 unshard LoRA 参数来避开这个复杂性，注释写道"LoRA parameters are generally small enough that this is not an issue"。这是一个正确但略显粗暴的策略。

## 4.5 本章小结

> 💡 **小结**
>
> - FSDP2 通过 `mesh[fsdp_dim_names]` 切片获取 DP 子 mesh，确保 `fully_shard()` 只在 DP 维度操作，与 TP 维度正交。
> - PEFT ParamWrapper 的 DTensor + Tensor 混合加法通过 `DTensor.from_local()` 零成本适配解决。
> - LoRA 参数被 FSDP 独立分片管理，但在 Triton kernel 中被完整 unshard 后使用。
> - ParallelismConfig 的校验 patch 允许不带 FSDP 使用 TP/CP。

---

# 五、状态保存与恢复：双重分片参数如何正确序列化

## 5.1 设计哲学与核心问题

当模型参数同时被 TP 和 FSDP 分片后，每个 rank 只持有参数的一小部分。保存 checkpoint 时需要将这些碎片收集回完整参数，加载时又需要将完整参数重新分发到各 rank。这个"收集-分发"过程必须正确处理两种分片语义的叠加。

## 5.2 源码入口与关键对象

```text
src/axolotl/monkeypatch/accelerate/fsdp2.py
  - get_state_dict：保存时收集 DTensor → full tensor（line 100）
  - fsdp2_load_full_state_dict：加载时分发 full tensor → DTensor（line 20）

src/axolotl/core/trainers/mixins/distributed_parallel.py
  - DistributedParallelMixin._save：触发 get_state_dict（line 14）
```

## 5.3 主流程拆解

### 保存：DTensor → 完整 tensor

`get_state_dict()` 的 FSDP2 分支（`fsdp2.py:158-173`）：

```python
elif self.is_fsdp2:
    from torch.distributed.tensor import DTensor
    state_dict = {}
    sharded_state_dict = model.state_dict()
    for param_name, param in sharded_state_dict.items():
        if param.is_cpu:
            param = param.to(torch.device("cuda"))
        if isinstance(param, DTensor):
            param = param.full_tensor()      # ← 关键：gather 回完整 tensor
        if torch.distributed.get_rank() == 0:
            state_dict[param_name] = param.cpu()
        torch.distributed.barrier()
```

`param.full_tensor()` 是 DTensor 提供的方法，它根据参数的 placement 信息自动执行必要的通信（all-gather）将分片收集回完整 tensor。对于同时有 TP 和 FSDP 分片的参数，`full_tensor()` 会在两个维度上都做 gather。最终只有 rank 0 保留完整参数（`param.cpu()` 移到 CPU），其他 rank 在 `barrier()` 后释放。

**DeepSpeed TP 保存**（`fsdp2.py:129-144`）有不同的路径：

```python
tp_sharding = (
    self.deepspeed_config.get("tensor_parallel", {}).get("autotp_size", 0) > 1
)
if tp_sharding:
    state_dict = model._consolidated_16bit_state_dict()  # DeepSpeed 内置方法
```

DeepSpeed 使用自己的 `_consolidated_16bit_state_dict()` 来收集 TP 分片，而非 DTensor 的 `full_tensor()`。这要求 DeepSpeed >= 0.16.4。

### 加载：完整 tensor → DTensor

`fsdp2_load_full_state_dict()`（`fsdp2.py:20-97`）的逻辑更复杂：

```text
对于每个参数 param_name:
  1. rank 0 从 full_sd 获取完整 tensor
  2. 检查 meta model 中该参数是否有 device_mesh 属性
     ├── 有 device_mesh（DTensor 参数）：
     │     使用 distribute_tensor(full_tensor, device_mesh, placements) 分发
     │     自动处理 TP + FSDP 两个维度的分片
     │     克隆 local shard 以允许释放完整 tensor
     │
     └── 无 device_mesh（非分片参数）：
           rank 0 将 tensor 移到 GPU
           其他 rank 创建空 tensor
           dist.broadcast(tensor, src=0) 广播

  3. 构建 sharded_sd，最后 model.load_state_dict(sharded_sd, assign=True)
```

`distribute_tensor()` 是 DTensor 的核心分发函数，它根据 `placements`（如 `[Shard(0), Shard(1)]`）和 `device_mesh` 自动计算每个 rank 应该持有的 shard，并执行相应的通信。`src_data_rank=0` 指定数据源是 rank 0。

一个重要的优化细节在 `fsdp2.py:64-69`：

```python
if (
    sharded_param._local_tensor.untyped_storage().size()
    > sharded_param._local_tensor.nelement() * sharded_param._local_tensor.element_size()
):
    sharded_param = sharded_param.clone()
```

当 `distribute_tensor` 的结果是原始 tensor 的一个 view（存储大于实际元素数量），`.clone()` 确保 local shard 有自己独立的存储，从而允许完整 tensor 被 GC 回收。

## 5.4 关键细节与误区澄清

> **误区九：保存时的 `barrier()` 看起来是"多余的同步"，但实际上是为了控制内存峰值。**
>
> 如果所有参数同时调用 `full_tensor()`，每个 rank 都会临时持有多个完整参数的副本，峰值显存可能爆炸。逐参数 `full_tensor()` + `barrier()` 的策略确保在一个参数处理完、rank 0 移到 CPU 之后，其他 rank 才释放这个参数的完整副本，然后处理下一个。代价是**序列化的通信**——每个参数一次 all-gather + 一次 barrier，无法 overlap。

> **checkpoint resume 的隐患：FSDP2 optimizer 保存可能失败。**
>
> `CheckpointSaveMixin._save_optimizer_and_scheduler()`（`core/trainers/mixins/checkpoints.py`）将 FSDP2 的 optimizer 保存包在 try/except 中，失败时只打 warning。这意味着 TP + FSDP2 训练可能无法从 checkpoint 恢复 optimizer state，只能恢复模型权重和 scheduler。

## 5.5 本章小结

> 💡 **小结**
>
> - 保存时通过 `DTensor.full_tensor()` 在 TP + FSDP 两个维度上 all-gather 回完整参数，只在 rank 0 保留。
> - 加载时通过 `distribute_tensor()` 按 placement 自动分发到各 rank，处理 DTensor 和非 DTensor 参数。
> - 逐参数保存 + barrier 控制显存峰值，但带来串行通信开销。
> - DeepSpeed TP 走独立的 `_consolidated_16bit_state_dict()` 路径，要求 >= 0.16.4。

---

# 六、完整主路径串联

## 6.1 完整调用栈

以一次最典型的 TP 训练为例：用户配置 `tensor_parallel_size: 2`，`fsdp_version: 2`，使用 Llama 3.1 8B。

```text
User: axolotl train config.yml
  │
  ├─ Step 1: 配置加载与校验
  │     ├─ load_cfg() → Pydantic 校验
  │     │     ├─ check_tensor_parallel_size: None → 1 (默认值)
  │     │     ├─ check_tensor_parallel_optimizer: 拦截 8-bit optimizer
  │     │     └─ check_tensor_parallel_size_update_ds_json: (仅 DeepSpeed)
  │     │
  │     └─ prepare_optim_env() → setup_parallelism_envs()
  │           └─ 写入 PARALLELISM_CONFIG_TP_SIZE=2
  │              ACCELERATE_USE_PARALLELISM_CONFIG=true
  │
  ├─ Step 2: PatchManager 注册 patches
  │     ├─ patch_parallelism_config()
  │     │     └─ 放宽 ParallelismConfig 校验
  │     └─ patch_accelerate_fsdp2()
  │           └─ 替换 accelerate 的 fsdp2_prepare_model 和 get_state_dict
  │
  ├─ Step 3: 模型加载
  │     ├─ load_model_config()
  │     │     └─ check_model_config()
  │     │           └─ 检查 tie_word_embeddings（不兼容则报错）
  │     │
  │     ├─ ModelLoader._apply_pre_model_load_setup()
  │     │     ├─ use_parallel_config = True
  │     │     └─ _set_parallel_config()
  │     │           └─ build_parallelism_config(cfg)
  │     │                 └─ ParallelismConfig(tp_size=2, dp_shard_size=N)
  │     │                       .build_device_mesh("cuda")
  │     │
  │     └─ ModelLoader._build_model()
  │           ├─ model_kwargs["tp_plan"] = "auto"
  │           ├─ model_kwargs["tp_size"] = 2
  │           ├─ model_kwargs["device_mesh"] = mesh
  │           ├─ del model_kwargs["device_map"]
  │           │
  │           ├─ AutoModelForCausalLM.from_pretrained(**model_kwargs)
  │           │     └─ [HF Transformers 内部]
  │           │           └─ parallelize_module → DTensor 化所有 TP 层
  │           │
  │           └─ 修复 model._tp_size / _device_mesh（workaround）
  │
  ├─ Step 4: Accelerator.prepare(model)
  │     └─ fsdp2_prepare_model(accelerator, model)
  │           ├─ mesh[("dp_shard",)] → 提取 DP 子 mesh
  │           ├─ [如果 LoRA] patch_peft_param_wrapper_for_fsdp2()
  │           ├─ 遍历子模块 bottom-up:
  │           │     ├─ LoRA 模块 → _process_lora_module_for_fsdp()
  │           │     └─ 匹配 wrap policy → fully_shard(module, mesh=dp_mesh)
  │           └─ fully_shard(model, mesh=dp_mesh)  # 顶层
  │
  ├─ Step 5: 训练循环
  │     └─ HF Trainer.train()
  │           ├─ 每个 step:
  │           │     ├─ 前向: DTensor 自动处理 TP 通信 (all-reduce)
  │           │     ├─ 反向: DTensor autograd 自动插入梯度通信
  │           │     └─ 优化器更新: 在 sharded 参数上直接更新
  │           │
  │           └─ 保存 checkpoint:
  │                 └─ DistributedParallelMixin._save()
  │                       └─ accelerator.get_state_dict(model)
  │                             └─ 逐参数 full_tensor() + barrier
  │
  └─ Step 6: 训练结束，保存最终模型
        └─ 同 Step 5 的保存逻辑
```

## 6.2 每一层做了什么

| 阶段 | 接收 | 产出 / 副作用 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| 配置校验 | YAML dict | 归一化后的 config | 无 | 无 | 一次 |
| 环境变量注入 | config | os.environ 写入 | 无 | 无 | 一次 |
| Patch 注册 | - | 替换 accelerate 函数 | 无 | 无 | 一次 |
| DeviceMesh 构建 | config | ParallelismConfig + DeviceMesh | process group 初始化 | 极小 | 一次 |
| 模型加载 + TP | model_kwargs | DTensor 化的模型参数 | TP 初始化通信 | 每卡只持有部分参数 | 一次 |
| FSDP2 准备 | DTensor 模型 | FSDP + TP 双重分片 | 可能涉及广播 | 进一步分片节省显存 | 一次 |
| 前向计算 | input batch | loss | TP all-reduce (每层) | 激活值按 TP 切分 | 每 step |
| 反向传播 | loss | 梯度 | TP all-reduce (每层) | 梯度按 TP 切分 | 每 step |
| Checkpoint 保存 | sharded params | 完整 state_dict (rank 0) | 逐参数 all-gather | rank 0 峰值高 | 按 save_steps |

## 6.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `monkeypatch/trainer_fsdp_optim.py` | 文件名含 "fsdp"，与 TP+FSDP 场景相关 | ❌ 已被注释掉 | 是 transformers 4.47.0 的临时修复，已不再使用 |
| `utils/ctx_managers/sequence_parallel.py` | 含 "parallel" 关键字 | ❌ 仅用于 CP | CP（Context Parallelism）的序列切分，与 TP 完全无关 |
| `monkeypatch/ring_attn/` | 含 "分布式注意力" 相关逻辑 | ❌ 仅用于 CP | Ring attention 是 CP 的核心机制，TP 不需要修改注意力 |
| `vllm.tensor_parallel_size` | 字段名含 "tensor_parallel" | ❌ 是推理时的 TP | 仅用于 GRPO/EBFT 的 vLLM 推理服务，与训练 TP 独立 |
| `integrations/cut_cross_entropy/` | 两个 TP 示例都使用 CCE | ✅ 推荐但非必需 | CCE 减少 logits 显存，在 TP 下尤其有价值，但不是 TP 必需 |
| `monkeypatch/transformers/trainer_loss_calc.py` | 修改了 loss 计算 | ❌ 仅用于 CP | 将 `mean()` 改为 `nanmean()` 处理 CP 的 masked 序列 |

---

# 七、关键数据流 / 状态流 / shape 流程

## 7.1 Tensor shape 变化

以 Llama 3.1 8B 的一个 attention 层为例，`hidden_size=4096`, `num_heads=32`, `head_dim=128`, `tp_size=2`：

```text
模型加载前（完整参数）:
  q_proj.weight: [4096, 4096]
  k_proj.weight: [1024, 4096]   (GQA, num_kv_heads=8)
  v_proj.weight: [1024, 4096]
  o_proj.weight: [4096, 4096]

TP 切分后（tp_plan="auto", ColwiseParallel for q/k/v, RowwiseParallel for o）:
  每个 TP rank 持有:
  q_proj.weight: DTensor, local shape [2048, 4096]  (列切分，输出维度减半)
  k_proj.weight: DTensor, local shape [512, 4096]
  v_proj.weight: DTensor, local shape [512, 4096]
  o_proj.weight: DTensor, local shape [4096, 2048]  (行切分，输入维度减半)

FSDP2 额外分片后（假设 dp_shard_size=2）:
  每个 rank 持有:
  q_proj.weight: DTensor, local shape [1024, 4096]  (TP + FSDP 双重分片)
  ...但 FSDP 在计算前会 unshard 回 TP shard 大小

前向计算时:
  输入: hidden_states [batch, seq, 4096]  (所有 TP rank 相同输入)

  q_proj 前向 (ColwiseParallel):
    本地输出: [batch, seq, 2048]  (只有一半 head 的 query)
    无通信

  attention 计算:
    本地: 只计算分到的 16 个 head

  o_proj 前向 (RowwiseParallel):
    本地乘法: [batch, seq, 2048] × [4096, 2048]^T → [batch, seq, 4096] (partial sum)
    all-reduce: 在 TP group 内求和 → [batch, seq, 4096] (完整结果)
```

每一层的 attention 和 FFN 各有一次 all-reduce（RowwiseParallel 输出时），通信量约为 `batch × seq × hidden_size × dtype_size`。

## 7.2 Rank / Mesh / Process Group 变化

以 `world_size=8, tp_size=2, dp_shard_size=4` 为例：

```text
DeviceMesh shape: (4, 2), dim_names = ("dp_shard", "tp")

TP groups (mesh["tp"]):
  Group 0: [rank0, rank1]  — 这两个 rank 持有同一层的不同列
  Group 1: [rank2, rank3]
  Group 2: [rank4, rank5]
  Group 3: [rank6, rank7]

DP shard groups (mesh["dp_shard"]):
  Group 0: [rank0, rank2, rank4, rank6]  — 这些 rank 持有参数的不同 FSDP shard
  Group 1: [rank1, rank3, rank5, rank7]

数据分发:
  rank0 和 rank1 处理相同的 micro-batch（同一 TP group 必须看到相同数据）
  rank0 和 rank2 处理不同的 micro-batch（不同 DP rank 看不同数据）

effective_dp_size = world_size // tp_size = 8 // 2 = 4
4 个 DP rank 各处理一份独立的 micro-batch
```

## 7.3 状态切换

TP 的实现中没有显式的 context manager 或全局状态切换——TP 信息完全编码在 DTensor 的 `device_mesh` 和 `placements` 属性中，是**参数级别**的元数据，不是进程级别的全局状态。这与 CP（Context Parallelism）不同，CP 使用 `SequenceParallelContextManager` 做全局状态切换。

但有几个隐式的全局状态：

1. **`os.environ["PARALLELISM_CONFIG_TP_SIZE"]`**：进程级环境变量，初始化后不变。
2. **`accelerator.state.parallelism_config`**：Accelerator 对象上的属性，初始化后不变。
3. **`model._tp_size` 和 `model._device_mesh`**：模型对象上的属性，由 Transformers 设置（或由 Axolotl workaround 补设）。

这些状态都是"写一次、读多次"的，不涉及运行时切换。

---

# 八、核心机制深挖

## 8.1 Monkey Patch：必要的适配还是维护负担？

Axolotl 的 TP 链路涉及两个核心 monkey patch：

### `patch_accelerate_fsdp2()`

**替换了什么**：`accelerate.accelerator.fsdp2_prepare_model` 和 `accelerate.Accelerator.get_state_dict`。

**为什么不能用更简单的方式**：Accelerate 的原生 FSDP2 准备逻辑不支持多维 mesh 切片、不处理 LoRA DTensor 兼容性、不支持 DeepSpeed TP 的 state dict 收集。这些都是 Axolotl 特有的需求组合。

**影响范围**：
- `fsdp2_prepare_model` 是模块级函数替换（`accelerate.accelerator.fsdp2_prepare_model = ...`），影响所有通过 Accelerator 准备的 FSDP2 模型。
- `get_state_dict` 是实例方法替换（`accelerate.Accelerator.get_state_dict = ...`），影响所有 Accelerator 实例。
- 两者都是**一次性替换**，不可恢复（没有保存原始函数引用的恢复逻辑）。
- **没有版本保护**——不检查 Accelerate 版本就直接替换。

**维护风险**：每次 Accelerate 更新 FSDP2 相关逻辑时，Axolotl 的 patch 都可能因为函数签名变化或行为变化而失效。`fsdp2.py` 文件长达 539 行，包含大量从 Accelerate 原始代码中复制并修改的逻辑。

### `patch_parallelism_config()`

**替换了什么**：`ParallelismConfig._validate_accelerator` 和 `AcceleratorState.is_fsdp2`。

**为什么需要**：Accelerate 原生要求 FSDP 才能使用 ParallelismConfig，但 Axolotl 支持纯 TP（无 FSDP）和纯 CP（无 FSDP）场景。patch 放宽了这个校验。

**维护风险**：相对较低，只是校验逻辑的放宽，不涉及核心功能修改。

## 8.2 通信原语：Axolotl 不做通信，但通信无处不在

Axolotl 源码中**没有任何 TP 相关的自定义通信调用**。所有 TP 通信由 PyTorch DTensor 在以下时机自动插入：

| 时机 | 通信类型 | 触发者 | 频率 |
|---|---|---|---|
| ColwiseParallel 前向 | 无通信 | PyTorch DTensor | 每层每 step |
| RowwiseParallel 前向 | all-reduce | PyTorch DTensor | 每层每 step |
| ColwiseParallel 反向 | all-reduce (梯度) | PyTorch autograd | 每层每 step |
| RowwiseParallel 反向 | 无通信 (梯度) | PyTorch autograd | 每层每 step |
| FSDP unshard | all-gather | PyTorch FSDP2 | 每层每 step |
| FSDP reshard | 无通信 (本地切片) | PyTorch FSDP2 | 每层每 step |
| Checkpoint 保存 | all-gather (full_tensor) | Axolotl get_state_dict | 按 save_steps |
| Checkpoint 加载 | distribute_tensor / broadcast | Axolotl fsdp2_load | 一次 |

前向过程中，每个 Transformer 层大约产生 2 次 all-reduce（attention 和 FFN 各一次 RowwiseParallel 输出），通信在 TP group 内发生。FSDP 的 unshard 通信在 DP group 内发生。两者互不干扰。

## 8.3 配置归一化：batch size 的隐式缩放

`normalize_config()`（`utils/config/__init__.py:137-142`）中，`batch_size` 被 `effective_world_size` 缩放：

```python
effective_world_size = (
    cfg.world_size
    // (cfg.context_parallel_size or 1)
    // (cfg.tensor_parallel_size or 1)
)
cfg.batch_size = cfg.batch_size * effective_world_size
```

**这里有一个隐含假设：同一个 TP group 内的所有 rank 处理相同的 micro-batch。** 这意味着数据加载器不需要给 TP rank 分配不同的数据——Accelerate 的 DataLoader 会根据 `effective_world_size` 来分割数据集，而不是 `world_size`。

但 Axolotl 的 `normalize_config` 实际上只缩放了 `batch_size`，没有显式调整 `gradient_accumulation_steps` 或 DataLoader 的 sharding。TP rank 的数据一致性依赖 Accelerate/Trainer 正确处理 TP group——如果 Accelerate 的 DataLoader 仍然按 `world_size` 分配数据，每个 TP rank 就会拿到不同的数据，导致前向计算结果不一致。

**这是基于源码行为的推断**：Accelerate 在构建 `ParallelismConfig` 时，内部应当能够根据 mesh 结构正确设置 DataLoader 的分片行为，但 Axolotl 源码中没有显式验证这一点。

---

# 九、显存、性能与通信分析

## 9.1 显存收益范围

| 内容 | 是否因 TP 节省 | 原因 |
|---|---|---|
| 线性层参数 | ✅ 节省 ~(1-1/tp_size) | ColwiseParallel / RowwiseParallel 将权重按列/行切分 |
| Embedding 参数 | ✅ 节省（如支持） | TP plan 中 embedding 也被切分 |
| Attention 激活值 | ✅ 节省 | 每个 TP rank 只计算部分 head |
| FFN 激活值 | ✅ 节省 | 中间维度被 TP 切分 |
| Logits | ✅ 节省 | lm_head 是 TP 切分的，输出 logits 在 TP 维度上分片 |
| Optimizer state | ✅ 节省 | 优化器只管理本地 shard 的参数 |
| 输入 batch | ❌ 不节省 | 同一 TP group 内所有 rank 持有相同的输入 |
| 非线性层（LayerNorm 等） | ❌ 不节省 | 通常被 replicate（每个 rank 持有完整副本） |
| KV Cache（如适用） | ✅ 节省 | 每个 rank 只有部分 head 的 KV |

**真正的显存大头**是线性层参数和激活值。对于 8B 模型，TP=2 大约能将每卡的参数显存需求减半，激活值也显著降低。

**显存收益消失的地方**：
- Checkpoint 保存时，rank 0 需要临时持有每个参数的完整版本（通过 `full_tensor()` gather），这会在 rank 0 上产生显存峰值。
- LoRA 的 `lora_A`/`lora_B` 在 Triton kernel 中被完整 unshard 后使用，不享受 TP 分片的显存节省。但 LoRA 参数通常很小，影响有限。

## 9.2 通信开销

| 通信事件 | 类型 | Group | 频率 | 数据量 |
|---|---|---|---|---|
| RowwiseParallel 前向输出 | all-reduce | TP group | 每层每 step ×2 (attn + FFN) | `batch × seq × hidden_size × dtype` |
| ColwiseParallel 反向梯度 | all-reduce | TP group | 每层每 step ×2 | `batch × seq × hidden_size × dtype` |
| FSDP2 unshard | all-gather | DP group | 每层每 step | 参数量 × dtype |
| FSDP2 grad reduce | reduce-scatter | DP group | 每层每 step | 梯度量 × dtype |
| Checkpoint save | all-gather (full_tensor) | 全局 | 按 save_steps | 全部参数 |
| Checkpoint load | distribute_tensor / broadcast | 全局 | 一次 | 全部参数 |

**关键观察**：TP 的通信在 TP group 内发生（通常是同一节点内的 NVLink 连接），FSDP 的通信在 DP group 内发生（可能跨节点）。因此 TP 的通信延迟通常远低于 FSDP 的通信延迟。文档（`docs/nd_parallelism.qmd`）也提到 TP 推荐在 NVLink 连接的 GPU 之间使用。

TP 通信**不能与计算 overlap**——RowwiseParallel 的 all-reduce 发生在线性层前向计算之后、下一步计算之前，是同步的。但 PyTorch DTensor 可能在未来支持异步通信。

## 9.3 性能取舍

Axolotl 的 TP 实现是一个经典的"通信换显存"方案：

| 获得 | 代价 |
|---|---|
| 每卡参数显存减少约 (1-1/tp_size) | 每层增加 2 次 TP all-reduce |
| 每卡激活值显存减少 | NVLink 带宽消耗增加 |
| 可训练更大模型 | Checkpoint 保存变慢（串行 all-gather） |
| TP 通信在 NVLink 上，延迟低 | 增加工程复杂度（DTensor、mesh 管理） |
| 与 FSDP2 正交组合 | 部分功能不兼容（tied embeddings、8-bit optimizer、Liger kernels） |

---

# 十、配置项、边界条件与坑点

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `tensor_parallel_size: N` | `loaders/model.py:749-754` | 传入 `tp_plan="auto"` 进行 TP 切分 | 必须整除 `num_attention_heads`；必须整除 `world_size` |
| `fsdp_version: 2` | `loaders/patch_manager.py` | 启用 FSDP2 patch，与 TP 正交组合 | 必须配合 `fsdp_config` 使用；FSDP1 不支持 TP |
| `dp_shard_size` / `dp_replicate_size` | `utils/distributed.py:319` | 显式指定 DP 维度大小 | 各维度之积必须等于 `world_size` |
| `optimizer` | `schemas/validation.py:1600` | 8-bit optimizer 与 TP 不兼容 | `paged_adamw_8bit`、`adamw_8bit`、`adamw_bnb_8bit` 被拦截 |
| `fsdp_config.cpu_ram_efficient_loading` | `loaders/model.py:769` | TP 下不走 CPU 中转路径 | TP 时 `device_map` 被删除，模型直接加载到 GPU |
| `plugins: [axolotl.integrations.liger.LigerPlugin]` | `integrations/liger/args.py:88` | `liger_rms_norm` 和 `liger_fused_linear_cross_entropy` 被禁用 | 静默禁用，不报错 |
| `deepspeed` | `schemas/validation.py:1119` | 自动注入 `tensor_parallel.autotp_size` | 修改临时文件，原始 DS config 不变 |
| `model_config.tie_word_embeddings` | `loaders/utils.py:139` | TP 直接报错 | 很多小模型默认 tie=True（如 Qwen2.5-0.5B） |
| `vllm.tensor_parallel_size` | `core/trainers/grpo/__init__.py:77` | 仅用于 vLLM 推理 TP | 与训练 TP 完全独立，不要混淆 |

### 开启 TP 的最小配置

```yaml
base_model: meta-llama/Llama-3.1-8B  # 必须无 tie_word_embeddings
tensor_parallel_size: 2
fsdp_version: 2
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
```

### 静默失效条件

- **模型未定义 `_tp_plan`**：`tp_plan="auto"` 不会报错，但不会切分任何层——模型仍然是完整的，每个 TP rank 持有全部参数。
- **Liger kernel 与 TP 不兼容时**：`liger_rms_norm` 和 `liger_fused_linear_cross_entropy` 被静默设为 `False`（`integrations/liger/args.py:88-112`），不抛异常。

### 不兼容组合

- TP + `tie_word_embeddings=True` → 报错
- TP + 8-bit optimizer → 报错
- TP + DDP（无 FSDP）→ ParallelismConfig 校验失败
- TP + FSDP1 → 不支持（需要 `fsdp_version: 2`）
- TP + `device_map` → 内部删除 `device_map`
- TP + Liger RMSNorm / Fused Linear CE → 静默禁用

---

# 十一、测试、示例与覆盖缺口

## 11.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/test_tensor_parallel_batch_size.py` | batch_size 缩放逻辑 | 验证 effective_world_size 正确排除 TP rank |
| `tests/test_loaders.py` (line 181-217) | `_get_parallel_config_kwargs` | 验证 6 种维度分解组合 |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml` | 3D 并行配置示例 | FSDP + TP + CP，8 GPU |
| `examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml` | HSDP + TP 配置示例 | 16 GPU 多节点 |
| `docs/nd_parallelism.qmd` | 功能文档 + 支持矩阵 | 标记为 [Experimental] |

## 11.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| TP E2E 训练正确性 | ❌ 唯一的 E2E 测试被 skip | 无法验证 TP 训练是否产生正确的 loss/梯度 |
| TP + LoRA | ❌ 无测试 | DTensor + LoRA 交互可能在 unshard/reshard 时出错 |
| TP + QLoRA | ❌ 无测试 | Params4bit + DTensor 的交互未验证 |
| TP checkpoint save/resume | ❌ 无测试 | `full_tensor()` 收集和 `distribute_tensor()` 分发的正确性未验证 |
| TP + DeepSpeed AutoTP | ❌ 无测试 | DeepSpeed TP 路径从 JSON 注入到 `_consolidated_16bit_state_dict` 全链路未测 |
| TP 多节点 | ❌ 无测试 | 跨节点 TP 通信延迟和正确性未验证 |
| 模型无 `_tp_plan` 的静默失效 | ❌ 无测试 | 用户可能误以为 TP 生效，实际未切分 |
| TP + sample packing | ❌ 无测试 | packing 后的 batch 在 TP rank 间的一致性未验证 |
| Liger 静默禁用 | ❌ 无测试 | 用户可能不知道 Liger kernel 被禁用 |
| TP 显存/性能收益 | ❌ 无 benchmark | 实际收益无量化数据 |

**最大的测试缺口**：唯一的 E2E TP 测试（`tests/e2e/multigpu/test_tp.py`）因为 `Qwen2.5-0.5B` 有 tied embeddings 而被 skip。这意味着 TP 在 CI 中**完全没有 E2E 回归保护**。修复方案很简单——换一个没有 tied embeddings 的小模型（如 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`）。

---

# 十二、局限性与已知优化点

## 12.1 硬约束

- **`num_attention_heads` 必须整除 `tensor_parallel_size`**：TP 按 head 切分 attention，不能整除就无法切分。
- **`tie_word_embeddings` 必须为 False**：DTensor 不支持同一个 tensor 有两种不同的 placement。
- **`world_size` 必须整除 `tensor_parallel_size`**：mesh 构建要求精确分配。
- **8-bit 优化器不兼容**：bitsandbytes 的量化状态管理不支持 DTensor。
- **DDP 后端不支持**：只能配合 FSDP2 或 DeepSpeed 使用。
- **Liger RMSNorm 和 Fused Linear CE 不兼容**：自定义 kernel 不支持 TP 分片的参数。

## 12.2 维护成本

- **FSDP2 monkeypatch 长达 539 行**：从 Accelerate 源码复制并修改，每次 Accelerate 升级都需要同步审查。没有版本保护。
- **Transformers `_tp_size` workaround**：依赖特定版本的 Transformers 内部实现细节。
- **TP + LoRA 的 DTensor 兼容**：涉及 PEFT、PyTorch、Accelerate 三方交互，任一方升级都可能破坏。
- **测试缺失**：没有 E2E 测试意味着 TP 链路的回归只能靠人工验证。

## 12.3 性能瓶颈

- **Checkpoint 保存串行化**：逐参数 `full_tensor()` + `barrier()` 不能 overlap，保存时间与参数数量线性增长。
- **rank 0 保存瓶颈**：只有 rank 0 保留完整 state_dict，其 CPU 内存和 GPU 显存可能成为瓶颈。
- **LoRA 参数完整 unshard**：在 Triton kernel 中 LoRA A/B 被完整 unshard，没有利用 TP 分片。
- **TP 通信不能 overlap**：all-reduce 是同步的，不能与下一层计算重叠。

## 12.4 已知优化点

1. **E2E 测试修复**：将测试模型从 `Qwen2.5-0.5B`（tied embeddings）换成不 tie 的小模型即可解除 skip。
2. **Schema 描述更新**：`config.py:993` 的 "Only supported with DeepSpeed AutoTP" 描述应更新为包含 FSDP2。
3. **静默失效检测**：当模型未定义 `_tp_plan` 时应发出 warning，而非静默不切分。
4. **异步 checkpoint 保存**：可以用 `DCP`（Distributed Checkpoint）替代逐参数 `full_tensor()`，实现分布式并行保存。
5. **LoRA kernel TP 优化**：在 Triton kernel 中直接处理 DTensor，避免完整 unshard。
6. **版本保护**：为 FSDP2 和 ParallelismConfig patch 添加 Accelerate 版本检查。
7. **`cpu_ram_efficient_loading` 与 TP 的兼容**：当前 TP 下跳过了 CPU 中转路径，可以探索通过 DTensor 的 `distribute_tensor` 实现类似的内存优化。

---

# 十三、小结与展望

Axolotl 的 Tensor Parallelism 实现可以用几个关键词概括。

### 关键词一：委托式集成

Axolotl 选择了"最轻量"的 TP 集成方式——不实现任何 TP 计算或通信逻辑，而是将 `tp_plan="auto"` 和 `device_mesh` 传给 HuggingFace Transformers，由 PyTorch DTensor 处理所有底层细节。这个决策让 Axolotl 只需约 200 行核心代码就接入了 TP，但也意味着框架对 TP 行为的控制力几乎为零——无法自定义切分策略，无法为 Transformers 未适配的模型提供 TP 支持。

### 关键词二：Mesh 切片与维度正交

FSDP2 和 TP 通过多维 DeviceMesh 实现正交组合的设计是 Axolotl TP 实现中最精巧的部分。`mesh[fsdp_dim_names]` 一行代码就将 FSDP 的操作范围限制在 DP 维度上，而 TP 维度由 Transformers 在模型构建时独立处理。这种"各管各维度"的设计清晰且可组合。

### 关键词三：DTensor 统一抽象

无论是 TP 分片还是 FSDP 分片，参数最终都被表示为 DTensor。这让状态保存（`full_tensor()`）和恢复（`distribute_tensor()`）可以用统一的接口处理双重分片，不需要区分"这个参数是 TP 分的还是 FSDP 分的"。PEFT 的 DTensor 兼容 patch 也得益于这个统一抽象。

### 关键词四：通信换显存，NVLink 是前提

TP 每层引入 2 次 all-reduce，将参数和激活值的显存需求减少到 `1/tp_size`。这个交换在 NVLink 连接的 GPU 之间是划算的（NVLink 带宽远高于 PCIe/InfiniBand），但在跨节点场景下通信延迟可能抵消收益。

### 适用场景

- 单节点多卡微调超大模型（FSDP2 + TP 2D 并行）
- 模型单层参数量超过单卡容量（纯 FSDP 无法装下一层）
- NVLink 连接的 GPU（A100/H100 同节点）

### 不适合的场景

- 小模型（TP 的通信开销会超过显存节省的收益）
- 跨节点 TP（通信延迟太高）
- 需要 8-bit 优化器或 Liger kernel 的场景
- 使用 tied embeddings 的模型（如大部分 Qwen 2.5 小参数变体）
- 需要可靠 checkpoint resume 的生产训练（optimizer state 保存可能失败）

### 后续值得走读的方向

1. **HuggingFace Transformers 的 `_tp_plan` 定义**：各模型类如何定义 ColwiseParallel / RowwiseParallel
2. **PyTorch DTensor 的通信调度**：all-reduce 如何被自动插入前向/反向图
3. **Accelerate ParallelismConfig 的 mesh 构建**：`build_device_mesh` 的维度命名和 rank 排列策略
4. **FSDP2 与 DTensor 的 shard/unshard 生命周期**：参数何时从 shard 状态 unshard、何时 reshard
5. **Axolotl 的 Context Parallelism (CP) 实现**：与 TP 正交但实现方式完全不同——CP 使用 ring attention 和序列切分，值得对比走读
