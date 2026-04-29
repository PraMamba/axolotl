# Axolotl 源码走读：HuggingFace Transformers `_tp_plan` 定义与 Colwise/Rowwise Tensor Parallel 实现解析

在大模型训练里，“把模型放进多张 GPU”可以有很多种含义：FSDP 把参数、梯度、优化器状态切开；Context Parallel 把长序列切开；而 Tensor Parallel 则更直接——把一个线性层本身切开。对 Axolotl 来说，`tensor_parallel_size` 并不是自己重新实现一套 Megatron-LM，而是把用户配置转换成 HuggingFace Transformers 的 `tp_plan="auto"`，再由 Transformers 根据各模型类内置的 `_tp_plan` / `base_model_tp_plan` 把 `q_proj`、`o_proj`、`gate_proj`、`down_proj` 等模块映射到 `ColwiseParallel`、`RowwiseParallel` 等并行样式。

本文不展开 Megatron Tensor Parallel 的数学原理，也不讲 FSDP、DeepSpeed、LoRA 的完整背景；我们只顺着 Axolotl 的训练入口往下读：用户如何打开 TP，Axolotl 在哪里第一次改变行为，Transformers 如何读取模型类里的 `_tp_plan`，Colwise/Rowwise 如何注入通信，加载和保存时又怎样处理被切开的权重。

> 说明：Axolotl 项目 `pyproject.toml` 当前 pin 了 `transformers==5.5.4`（`pyproject.toml:20`），本文源码证据来自本地仓库 `/root/axolotl` 以及当前运行环境中安装的 Transformers 源码 `/usr/local/lib/python3.12/dist-packages/transformers`（本地版本为 5.3.0）。机制主线以源码行为为准，具体行号在不同 Transformers 小版本中可能轻微漂移。

# 前言

## 业务 / 工程背景

在 Axolotl 的常规 SFT 或 RLHF 训练中，用户面对的入口仍然是一份 YAML：

```yaml
base_model: Qwen/Qwen3-8B
dp_shard_size: 2
context_parallel_size: 2
tensor_parallel_size: 2
fsdp_version: 2
fsdp_config:
  state_dict_type: FULL_STATE_DICT
```

示例配置就放在 `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:6-19`。文档把这个组合描述成 “FSDP + TP + CP”：FSDP 解决参数/优化器状态，CP 解决长上下文激活，TP 负责把层内矩阵运算切到多个 GPU 上（`docs/nd_parallelism.qmd:28-41`, `docs/nd_parallelism.qmd:96-105`）。

但是 TP 的工程难点不在“写一个 all-reduce”这么简单，而在：不同模型的线性层命名、融合方式、norm 位置和 MoE 专家结构都不一样。Llama 的 `q_proj/k_proj/v_proj` 可以自然列切，Phi3 的 `qkv_proj` 被一个融合线性层输出后马上 slice，Mixtral 的专家权重是三维 packed 参数，Qwen3 的 `q_norm/k_norm` 夹在列切输出和注意力之间。于是 Transformers 让每个模型配置类声明一张计划表：哪些模块用 colwise，哪些模块用 rowwise，哪些地方需要 gather、split 或特殊梯度同步。

## 核心矛盾

这套实现背后的核心冲突可以压缩成三句话：

1. **层内切分能节省参数和局部激活显存，但会破坏模型代码对 shape 的隐含假设。** 例如 `chunk(2, dim=-1)` 或 q/k/v slice 需要看到完整最后一维，否则逻辑错位。
2. **Colwise 和 Rowwise 必须成对设计通信语义。** 前者让输出分片，后者把分片输入投影后 all-reduce 回完整 hidden；反向传播还要做互补通信。
3. **Axolotl 自己不定义模型级 TP 计划，只负责把配置、mesh 和加载参数交给 Transformers。** 因此真正决定 “Llama 怎么切、Qwen 怎么切、MoE 怎么切” 的，是 Transformers 模型类里的 `_tp_plan` / `base_model_tp_plan`。

## 本文主线

本文按机制而不是文件拆解：

1. Axolotl 如何把 `tensor_parallel_size` 从 YAML 变成真实行为；
2. Transformers 如何从模型类收集 `_tp_plan`；
3. Colwise/Rowwise 在 forward/backward 中如何注入通信；
4. 权重加载与保存如何切分/恢复 state dict；
5. shape、rank、mesh、显存和性能的真实收益与代价；
6. 配置坑点、测试覆盖和维护风险。

## 不展开的内容

本文不讲 Megatron-LM TP 论文推导，不讲 FSDP/ZeRO 全流程，不讲 ring attention 和 Context Parallel 的完整实现，也不讲 vLLM 推理侧 TP。它们只在与 Axolotl `tensor_parallel_size` 或 Transformers `_tp_plan` 主路径相交时出现。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/main.py` | `axolotl train` CLI 入口，启动训练配置解析与 launcher。 |
| `src/axolotl/cli/config.py` | 读取 YAML、应用 CLI override、执行 schema validation 和 normalize。 |
| `src/axolotl/utils/trainer.py` | 写入 Accelerate 并行环境变量，包括 `PARALLELISM_CONFIG_TP_SIZE`。 |
| `src/axolotl/utils/distributed.py` | 根据 TP/CP/DP 配置构建 `ParallelismConfig` 和 `DeviceMesh`。 |
| `src/axolotl/loaders/model.py` | 模型加载主入口；TP 开启时传入 `tp_size`、`tp_plan="auto"`、`device_mesh`。 |
| `transformers/modeling_utils.py` | `PreTrainedModel` 收集 `_tp_plan`，`from_pretrained()` 触发 TP 初始化、分发、保存 gather。 |
| `transformers/integrations/tensor_parallel.py` | `ColwiseParallel`、`RowwiseParallel`、通信 autograd Function、state dict gather。 |
| `transformers/core_model_loading.py` | 权重加载时根据 TP plan 只物化/设置本 rank 所需 shard。 |
| `transformers/models/*/configuration_*.py` | 各模型定义 `base_model_tp_plan`，决定 q/k/v、MLP、MoE 的并行样式。 |
| `tests/e2e/multigpu/test_tp.py` | Axolotl 当前 TP e2e 测试入口，但核心测试被 skip。 |

# 一、从 YAML 到 `tp_plan="auto"`：Axolotl 解决的是“接入”而不是“定义计划”

## 1.1 设计哲学与核心问题

TP 对 Axolotl 的第一层问题不是“怎么切矩阵”，而是“什么时候把用户意图交给正确的下游”。用户只写 `tensor_parallel_size: 2`，Axolotl 必须完成几件工程化动作：

- 配置校验：默认值、优化器兼容性、DeepSpeed 配置补丁；
- world size 归一化：TP rank 不应该再被当成数据并行 rank 来放大 batch；
- mesh 构建：如果同时有 FSDP/CP/TP，必须给 Transformers 一个带 `tp` 维度的 `DeviceMesh`；
- 模型加载参数注入：最终真正触发 Transformers TP 的，是 `from_pretrained(..., tp_plan="auto", tp_size=N, device_mesh=...)`。

如果没有这一层，Transformers 的 `_tp_plan` 即使写在模型类里也不会生效；如果这一层写错，常见后果是：`device_map` 与 `tp_plan` 冲突、batch size 被按 world size 过度放大、FSDP 和 TP 使用了错误的 group。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train(): CLI 命令入口，调用 launch_training。  

src/axolotl/cli/train.py
  - do_cli(): 调用 load_cfg() 读取 YAML。
  - do_train(): 加载 dataset 后调用 axolotl.train.train()。

src/axolotl/cli/config.py
  - load_cfg(): YAML -> DictDefault -> validate_config -> normalize_config。

src/axolotl/utils/trainer.py
  - setup_parallelism_envs(): 写 PARALLELISM_CONFIG_TP_SIZE 和 ACCELERATE_USE_PARALLELISM_CONFIG。

src/axolotl/utils/distributed.py
  - build_parallelism_config(): 创建 Accelerate ParallelismConfig 和 DeviceMesh。

src/axolotl/loaders/model.py
  - ModelLoader.load(): 模型加载总入口。
  - ModelLoader._build_model(): TP 开启时传 tp_size/tp_plan/device_mesh 给 Transformers。
```

## 1.3 主流程拆解

真实入口从 CLI 开始：`src/axolotl/cli/main.py:78-129` 定义了 `axolotl train <config>`。进入 worker 后，`src/axolotl/cli/train.py:55-90` 的 `do_cli()` 先调用 `load_cfg()`，随后 `do_train()` 在 `src/axolotl/cli/train.py:23-52` 加载数据集并调用 `axolotl.train.train()`。

调用链可以简化成：

```text
User: axolotl train examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml
  -> cli.main.train()
    -> cli.train.do_cli()
      -> load_cfg()
        -> validate_config()
        -> prepare_optim_env()
        -> normalize_config()
      -> do_train()
        -> axolotl.train.train()
          -> setup_model_and_trainer()
            -> setup_model_and_tokenizer()
              -> ModelLoader(cfg, tokenizer).load()
                -> _apply_pre_model_load_setup()
                -> _build_model()
                  -> AutoModelForCausalLM.from_pretrained(..., tp_plan="auto")
```

几个关键点：

- `load_cfg()` 从 YAML 读入 `DictDefault`，在 `src/axolotl/cli/config.py:244-320` 完成远程/目录配置处理、CLI override、插件准备和 schema validation。
- `prepare_optim_env()` 最后调用 `setup_parallelism_envs(cfg)`（`src/axolotl/utils/trainer.py:643-667`）。当 `cfg.tensor_parallel_size > 1` 时，它写入 `PARALLELISM_CONFIG_TP_SIZE`，并设置 `ACCELERATE_USE_PARALLELISM_CONFIG=true`（`src/axolotl/utils/trainer.py:621-640`）。
- `normalize_config()` 在分布式环境下会用“有效数据并行 world size”更新 `cfg.batch_size`：`world_size // context_parallel_size // tensor_parallel_size`（`src/axolotl/utils/config/__init__.py:134-142`）。这说明 TP rank 不被视为拿不同 batch 的数据并行 rank。
- `ModelLoader._apply_pre_model_load_setup()` 判断 FSDP、TP、CP 是否需要 parallel config（`src/axolotl/loaders/model.py:196-212`），随后 `_set_parallel_config()` 调用 `build_parallelism_config()`（`src/axolotl/loaders/model.py:437-443`）。
- `build_parallelism_config()` 根据 `cfg.tensor_parallel_size` 写入 `tp_size`，再调用 Accelerate 的 `ParallelismConfig.build_device_mesh("cuda")`（`src/axolotl/utils/distributed.py:299-315`）。
- 最终 `_build_model()` 在 `cfg.tensor_parallel_size > 1` 时设置：

```python
self.model_kwargs["tp_size"] = self.cfg.tensor_parallel_size
self.model_kwargs["tp_plan"] = "auto"
self.model_kwargs["device_mesh"] = self.device_mesh
```

对应源码在 `src/axolotl/loaders/model.py:749-752`。如果已有 `device_map`，Axolotl 会删除它，因为 Transformers 明确禁止 `tp_plan` 和 `device_map` 同时使用（Axolotl 删除处在 `src/axolotl/loaders/model.py:753-754`；Transformers 抛错逻辑在 `transformers/integrations/tensor_parallel.py:47-50`）。

## 1.4 关键细节与误区澄清

> 容易误解一：`tensor_parallel_size` 在 Axolotl schema 里写着 “Only supported with DeepSpeed AutoTP”，所以 FSDP + TP 不是主路径。

这个描述已经不能代表当前源码主路径。schema 字段说明在 `src/axolotl/utils/schemas/config.py:993-997`，确实写了 DeepSpeed AutoTP；但模型加载时只要 `cfg.tensor_parallel_size > 1`，`ModelLoader._build_model()` 就会传 `tp_plan="auto"` 给 Transformers（`src/axolotl/loaders/model.py:749-752`）。文档也明确列出 `FSDP + TP`、`HSDP + TP`、`FSDP + TP + CP` 为支持组合（`docs/nd_parallelism.qmd:96-105`）。因此，实际行为以 `ModelLoader` 和文档的 ND Parallelism 路径为准。

> 容易误解二：Axolotl 自己定义了 Llama/Qwen/Mixtral 的 Colwise/Rowwise 切法。

没有。Axolotl 只传 `tp_plan="auto"`。各模型层怎么切由 Transformers 的 `configuration_*.py` 和 `modeling_*.py` 决定；例如 Llama 的 plan 定义在 `transformers/models/llama/configuration_llama.py:109-118`，Qwen3 的 plan 定义在 `transformers/models/qwen3/configuration_qwen3.py:109-120`。

> 容易误解三：`preprocess` 命令会触发 TP 主流程。

标准训练主路径是 `axolotl train` -> `ModelLoader.load()`。`preprocess` CLI 入口在 `src/axolotl/cli/main.py:51-75`，它主要处理数据预处理；本文讨论的 `tp_plan="auto"` 是模型加载阶段行为，不是数据预处理阶段行为。

## 1.5 本章小结

> 💡 **小结**
>
> * Axolotl 的 TP 接入点是 `tensor_parallel_size > 1`，真正触发 Transformers TP 的是 `tp_plan="auto"`。
> * Axolotl 负责配置、batch size、DeviceMesh 和加载参数；模型级切分规则由 Transformers 定义。
> * `device_map` 与 `tp_plan` 互斥，Axolotl 会在 TP 开启时删除 `device_map`。
> * schema 的 DeepSpeed 描述存在滞后，不能单独作为真实行为依据。

# 二、`_tp_plan` 的生成：模型类把“工程 shape 假设”写成计划表

## 2.1 设计哲学与核心问题

Tensor Parallel 最容易被讲成两句话：列切输入投影，行切输出投影。但在真实模型里，问题远比这复杂：

- Llama：`q_proj/k_proj/v_proj` 是独立线性层，适合标准 colwise；`o_proj` 接受局部 heads，适合 rowwise。
- Qwen3：`q_norm/k_norm` 位于分片后的 q/k 上，norm 参数本身不切，但梯度必须跨 TP rank 汇总。
- Phi3：`qkv_proj` 和 `gate_up_proj` 是融合层，输出后立即 slice/chunk；如果保持分片输出，后续模型代码会错。
- Mixtral / Qwen3-MoE：专家权重是三维 packed 参数，还要处理 expert 维、packed gate/up 和专家输出求和。

所以 `_tp_plan` 的本质不是“静态配置表”，而是把每个模型源码里的 shape 假设显式编码：哪里可以保持分片，哪里必须 gather，哪里需要 split_input，哪里需要特殊梯度 all-reduce。

## 2.2 源码入口与关键对象

```text
transformers/modeling_utils.py
  - PreTrainedModel._tp_plan: 顶层模型或 base model 收集后的完整 TP plan。
  - PreTrainedModel.post_init(): 从 config.base_model_tp_plan 和子模块 _tp_plan 收集 plan。
  - PreTrainedModel.tp_plan.setter: 校验 style 是否在 ALL_PARALLEL_STYLES 中。

transformers/configuration_utils.py
  - PreTrainedConfig.base_model_tp_plan: base model 的默认 TP plan 类属性。

transformers/models/llama/configuration_llama.py
  - LlamaConfig.base_model_tp_plan: 标准 decoder-only 模式。

transformers/models/qwen3/configuration_qwen3.py
  - Qwen3Config.base_model_tp_plan: 增加 q_norm/k_norm 梯度同步。

transformers/models/phi3/configuration_phi3.py
  - Phi3Config.base_model_tp_plan: fused qkv / gate_up 的 gather + split 路径。

transformers/models/mixtral/configuration_mixtral.py
  - MixtralConfig.base_model_tp_plan: packed MoE 专家路径。
```

## 2.3 主流程拆解

Transformers 在 `PreTrainedModel` 上声明了 `_tp_plan`、`_tp_size` 等属性（`transformers/modeling_utils.py:1151-1157`）。真正收集 plan 的地方是 `post_init()`：

```text
PreTrainedModel.post_init()
  -> 如果 self.base_model is self:
       self._tp_plan = self.config.base_model_tp_plan.copy()
  -> 遍历 named_children():
       如果 child 有 _tp_plan，则加上 child 名字前缀合并到当前模型
  -> init_weights()
```

源码依据在 `transformers/modeling_utils.py:1287-1333`。这解释了两个层次：

1. **base model plan**：放在 config 类里，如 `LlamaConfig.base_model_tp_plan`。
2. **top-level head plan**：放在 `LlamaForCausalLM._tp_plan`，通常是 `{"lm_head": "colwise_gather_output"}`。

以 Llama 为例：

```python
# transformers/models/llama/configuration_llama.py:109-118
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}
```

而 `LlamaForCausalLM` 又补了语言模型头：

```python
# transformers/models/llama/modeling_llama.py:441-444
class LlamaForCausalLM(...):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
```

`post_init()` 在顶层模型上会把子模块 `model` 的 plan 前缀化成 `model.layers.*...`，再加上 `lm_head`。这也是为什么计划表里有时在 config 中没有 `model.` 前缀，但顶层模型最终 `_tp_plan` 会带上模块路径前缀。

再看几个模型差异：

| 模型 | 关键 plan | 工程含义 |
|---|---|---|
| Llama | `q/k/v/gate/up: colwise`，`o/down: rowwise` | 标准“局部分片激活 -> rowwise all-reduce”模式。 |
| Qwen3 | 增加 `q_norm/k_norm: replicated_with_grad_allreduce` | norm 权重复制，但梯度来自局部 heads，需要同步。 |
| Phi3 | `qkv_proj: colwise_gather_output`，`o_proj: rowwise_split_input` | fused qkv 后续 slice 需要完整输出；rowwise 前再 split。 |
| Olmo3 | `q/k/v: colwise_gather_output`，`o_proj: rowwise_split_input` | q/k/v 后有额外 norm，保守 gather 避免分片 shape 破坏。 |
| Mixtral | `experts.gate_up_proj: packed_colwise`，`experts.down_proj: rowwise`，`experts: moe_tp_experts` | 三维专家权重和 MoE 输出需要专门处理。 |
| DeepSeek-V3 | shared/dense MLP 使用 col/row，MoE experts 使用 packed + moe | 注意力 plan 在本地源码中未列入该 config；主要 TP 作用在 MLP/MoE。 |

这些定义分别可在 `transformers/models/qwen3/configuration_qwen3.py:109-120`、`transformers/models/phi3/configuration_phi3.py:107-112`、`transformers/models/olmo3/configuration_olmo3.py:104-112`、`transformers/models/mixtral/configuration_mixtral.py:115-123`、`transformers/models/deepseek_v3/configuration_deepseek_v3.py:131-141` 找到。

## 2.4 关键细节与误区澄清

> 容易误解四：`base_model_tp_plan` 会被保存到 `config.json`，所以保存后的模型自带完整 TP plan。

Transformers 的 `PreTrainedConfig` 声明了 `base_model_tp_plan`（`transformers/configuration_utils.py:82-83`, `transformers/configuration_utils.py:151`），但序列化时明确删除它：`configuration_utils.py:1044-1046` 注释写着 “Do not serialize `base_model_tp_plan` for now”。这意味着 plan 主要来自 Transformers 代码里的模型类定义，而不是 checkpoint config 文件中的普通字段。

> 容易误解五：所有模型都只需要 `colwise` / `rowwise` 两种 style。

不对。Transformers 的注册表还包括 `colwise_gather_output`、`rowwise_split_input`、`packed_colwise`、`packed_rowwise`、`replicated_with_grad_allreduce`、`moe_tp_experts`、`mla_kv_a_proj` 等（`transformers/integrations/tensor_parallel.py:1194-1213`）。这些 style 的存在就是为了处理融合层、MoE、MLA、norm 梯度等“模型源码 shape 假设”。

> 容易误解六：`_tp_plan` 只影响权重加载，不影响 forward。

不是。`distribute_model()` 会遍历 `model.named_modules()`，为命中 plan 的 module 注册 forward pre-hook / post-hook（`transformers/integrations/tensor_parallel.py:1464-1492`）。加载时会切权重，forward 时会执行通信 hook，两者都依赖同一张 plan。

## 2.5 本章小结

> 💡 **小结**
>
> * `_tp_plan` 是模型源码 shape 约束的显式化，不只是配置元数据。
> * base model 的 plan 来自 `config.base_model_tp_plan`，top-level head 的 plan 多由 `modeling_*.py` 中的 `_tp_plan` 补充。
> * `post_init()` 会把子模型 plan 前缀化并合并到顶层模型。
> * plan 不随 `config.json` 普通序列化保存，依赖 Transformers 模型类实现。

# 三、Colwise 与 Rowwise：把线性层切开后，谁负责把结果接回去？

## 3.1 设计哲学与核心问题

标准 decoder block 里有两类线性层：

```text
Attention:
  hidden -> q/k/v -> attention -> o_proj -> hidden

MLP:
  hidden -> gate/up -> activation/mul -> down_proj -> hidden
```

如果把 `q_proj` 或 `gate_proj` 的输出维度切开，每个 rank 只得到一部分 heads 或 intermediate channels。这样能节省该层权重和后续局部激活，但模型主干最后仍要回到完整 hidden size。于是 Colwise/Rowwise 的分工是：

- **ColwiseParallel**：切权重的输出维度，让每个 rank 产生局部输出；默认不 gather。
- **RowwiseParallel**：切权重的输入维度，接受局部分片输入，计算局部 partial output，再 all-reduce 成完整输出。

它解决的是 shape 和通信边界问题：分片可以在 block 内部流动，但 block 边界通常要恢复完整 hidden。

## 3.2 源码入口与关键对象

```text
transformers/integrations/tensor_parallel.py
  - ColwiseParallel: 权重按 dim -2 切；输入复制，输出默认分片。
  - RowwiseParallel: 权重按 dim -1 切；输出 all-reduce。
  - _AllReduceBackward / _AllReduceForward: 自定义 autograd 通信语义。
  - distribute_module(): 注册 forward pre-hook 和 post-hook。
```

## 3.3 主流程拆解

### Colwise：输入完整，输出分片

`ColwiseParallel` 的核心在 `transformers/integrations/tensor_parallel.py:684-713`：

```python
class ColwiseParallel(TensorParallelLayer):
    def _prepare_input_fn(self, mod, inputs, device_mesh):
        input_tensor = inputs[0] if inputs else inputs
        return all_reduce_backward(input_tensor, device_mesh)

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        if self.gather_output:
            return all_gather(outputs, device_mesh)
        return outputs

    def shard_tensor(...):
        # 2D weight: shard dim -2 (out_features)
```

对 PyTorch `nn.Linear(in, out)` 来说，weight shape 是 `[out, in]`。Colwise sharding `dim=-2` 意味着每张卡只保留 `out / tp_size` 行：

```text
W_q: [hidden, hidden]
TP=2 colwise:
  rank0 W_q0: [hidden/2, hidden]
  rank1 W_q1: [hidden/2, hidden]

input x: [B, S, hidden]  # replicated
rank0 q0 = x @ W_q0^T -> [B, S, hidden/2]
rank1 q1 = x @ W_q1^T -> [B, S, hidden/2]
```

`all_reduce_backward()` 的名字容易让人绕晕。它 forward 是 identity，backward 才 all-reduce（`transformers/integrations/tensor_parallel.py:450-465`）。这保证虽然 forward 输入是复制的，但反向时输入梯度能跨 TP rank 汇总。

### Rowwise：输入分片，输出求和

`RowwiseParallel` 在 `transformers/integrations/tensor_parallel.py:808-849`：

```python
class RowwiseParallel(TensorParallelLayer):
    def _prepare_input_fn(...):
        if self.split_input:
            return split(input_tensor, device_mesh)
        return input_tensor

    def _prepare_output_fn(...):
        outputs = all_reduce_forward(outputs, device_mesh)
        if saved_bias:
            outputs = outputs + bias
        return outputs

    def shard_tensor(...):
        # 2D weight: shard dim -1 (in_features)
```

对应 shape：

```text
W_o: [hidden, hidden]
TP=2 rowwise:
  rank0 W_o0: [hidden, hidden/2]
  rank1 W_o1: [hidden, hidden/2]

input attn_i: [B, S, hidden/2]
rank_i partial = attn_i @ W_oi^T -> [B, S, hidden]
all_reduce(sum partial) -> [B, S, hidden]
```

这里 `all_reduce_forward()` forward 做 sum，backward 是 identity（`transformers/integrations/tensor_parallel.py:468-480`, `transformers/integrations/tensor_parallel.py:602-604`）。

### Hook 注入点

Transformers 没有改写每个模型的 `forward()`。它用 `distribute_module()` 给 module 注册 hooks：

```python
# transformers/integrations/tensor_parallel.py:622-636
if input_fn is not None:
    module.register_forward_pre_hook(...)
if output_fn is not None:
    module.register_forward_hook(...)
```

`distribute_model()` 会遍历所有 module，命中 plan 后调用 `add_tensor_parallel_hooks_to_module()`，并写入 `module._hf_tp_plan`、`module._hf_device_mesh`（`transformers/integrations/tensor_parallel.py:1358-1380`, `transformers/integrations/tensor_parallel.py:1464-1492`）。这说明 TP 的 forward 通信是模块级 hook 注入，而不是模型源码每层手写通信。

## 3.4 关键细节与误区澄清

> 容易误解七：Colwise forward 没有通信，所以它完全免费。

Colwise 默认 forward 不 all-gather，但它在输入上包了 `all_reduce_backward()`，反向会产生 all-reduce。若 style 是 `colwise_gather_output`，forward 还会 all-gather 输出（`transformers/integrations/tensor_parallel.py:699-702`）。所以是否“免费”取决于 plan。

> 容易误解八：Rowwise 的 all-reduce 是梯度同步。

Rowwise 的 `all_reduce_forward()` 是 forward 阶段对 partial output 求和（`transformers/integrations/tensor_parallel.py:835-839`）。它不是 optimizer/DP 的梯度平均；backward 反而是 identity。TP 的通信维度和 DDP/FSDP 的数据并行通信维度不是同一个概念。

> 容易误解九：Rowwise 的 bias 会被每个 rank 重复加，导致放大。

源码在 rowwise pre-hook 中临时移走 `mod.bias`（`transformers/integrations/tensor_parallel.py:823-827`），post-hook 在 all-reduce 后只加一次（`transformers/integrations/tensor_parallel.py:835-839`）。这是一个很小但很关键的数值正确性细节。

## 3.5 本章小结

> 💡 **小结**
>
> * Colwise 切输出维度，Rowwise 切输入维度；二者共同维持 block 边界 hidden shape。
> * Transformers 通过 forward hooks 注入通信，不改每个模型 forward 源码。
> * Colwise 的通信主要在 backward；Rowwise 的通信主要在 forward。
> * `colwise_gather_output`、`rowwise_split_input` 是为无法保持分片 shape 的模型代码准备的保守变体。

# 四、完整主路径串联：一次 `axolotl train` 如何走到 `_tp_plan`

## 4.1 完整调用栈

```text
User: axolotl train examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml
  │
  ├─ Step 1: CLI 与配置加载
  │     └─ src/axolotl/cli/main.py:78-129
  │     └─ src/axolotl/cli/config.py:230-329
  │
  ├─ Step 2: 配置校验与环境变量
  │     └─ src/axolotl/utils/schemas/validation.py:1501-1505
  │     └─ src/axolotl/utils/trainer.py:621-640
  │     └─ src/axolotl/utils/config/__init__.py:134-142
  │
  ├─ Step 3: DeviceMesh 构建
  │     └─ src/axolotl/loaders/model.py:196-212
  │     └─ src/axolotl/utils/distributed.py:299-315
  │     └─ accelerate/parallelism_config.py:211-244
  │
  ├─ Step 4: 模型加载参数注入
  │     └─ src/axolotl/loaders/model.py:745-858
  │     └─ transformers/modeling_utils.py:3985-3988
  │
  ├─ Step 5: Transformers 收集并应用 TP plan
  │     └─ transformers/modeling_utils.py:1287-1333
  │     └─ transformers/modeling_utils.py:4113-4115
  │     └─ transformers/integrations/tensor_parallel.py:1464-1492
  │
  ├─ Step 6: 权重加载时按 TP plan 切 shard
  │     └─ transformers/core_model_loading.py:1116-1195
  │     └─ transformers/core_model_loading.py:1210-1235
  │
  ├─ Step 7: 每个 forward 通过 hooks 做 TP 通信
  │     └─ transformers/integrations/tensor_parallel.py:450-619
  │     └─ transformers/integrations/tensor_parallel.py:684-849
  │
  └─ Step 8: 保存时 gather TP shard
        └─ transformers/modeling_utils.py:3360-3363
        └─ transformers/integrations/tensor_parallel.py:1257-1355
```

## 4.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| 配置加载 | YAML + CLI overrides | `cfg.tensor_parallel_size` 被校验，默认 None -> 1 | 无 | 无 | 启动一次 |
| 环境准备 | `cfg` | `PARALLELISM_CONFIG_TP_SIZE=N`，`ACCELERATE_USE_PARALLELISM_CONFIG=true` | 无 | 无 | 启动一次 |
| batch normalize | `WORLD_SIZE`, TP/CP size | `cfg.batch_size *= world_size/(tp*cp)` | 无 | 避免数据并行重复放大 | 启动一次 |
| mesh 构建 | TP/CP/DP sizes | `DeviceMesh(..., mesh_dim_names=[..., "tp"])` | 依赖 torch.distributed 初始化 | 建立 group，不直接省显存 | 启动一次 |
| from_pretrained 参数 | model kwargs | `tp_plan="auto"`, `tp_size`, `device_mesh` | 初始化/选择 group | 后续权重按 shard 加载 | 启动一次 |
| plan 收集 | 模型类 `_tp_plan` / config plan | `model._tp_plan`, `_tp_size`, `_device_mesh` | 无 | 无 | 初始化一次 |
| 权重加载 | checkpoint tensor/slice | 每 rank 只设置本地 shard 参数 | 无显式 collective；各 rank 各自读取/切片 | 参数显存约按 TP 降低 | 初始化一次 |
| forward hook | batch tensor | 局部激活、all-reduce/all-gather 输出 | TP group 内 collective | 中间激活可能降低或恢复 | 每层/每 step |
| save | local TP state dict | full state dict | TP group all-gather | 保存峰值显存/CPU 内存上升 | checkpoint/final save |

## 4.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `src/axolotl/cli/preprocess.py` | CLI 中也有 model/config 相关逻辑 | 标准训练 TP 主路径否 | TP 的 `tp_plan="auto"` 在 `ModelLoader._build_model()` 触发。 |
| `configuration_utils.py` 文档里的 `model.tensor_parallel` | 文档提到 plan applied when `model.tensor_parallel` | 本地源码未找到 `def tensor_parallel` | 当前主路径是 `from_pretrained(tp_plan=...)`。 |
| `src/axolotl/utils/schemas/config.py` 字段说明 | 写着 DeepSpeed AutoTP | 不是唯一主路径 | 真实主路径还包括 Transformers TP + FSDP2/DeviceMesh。 |
| `setup_parallelism_envs()` | 写 env 后似乎已完成 TP | 否 | env 只是给 Accelerate；真正模型切分在 Transformers。 |
| `ColwiseParallel.shard_tensor()` | 只看加载会以为 forward 无通信 | 否 | forward hook 也会调用 `all_reduce_backward` / `all_gather`。 |
| `gather_state_dict_for_save()` | 看起来像每 step 执行 | 否 | 只在 `save_pretrained()` 且 `_tp_size` 非空时执行。 |
| DeepSpeed `autotp_size` 补丁 | 似乎是所有 TP 必需 | 仅 DeepSpeed 配置路径 | FSDP/Transformers TP 路径不依赖修改 ds json。 |

## 4.4 本章小结

> 💡 **小结**
>
> * 一次训练里，TP 的关键行为分成初始化、加载、forward、保存四段。
> * Axolotl 只在初始化和加载前注入配置；forward 通信完全由 Transformers hooks 接管。
> * 保存时 TP 的显存收益会部分消失，因为要 gather full tensor。
> * 很多看似相关的路径只是配置/兼容路径，不是每 step 主循环。

# 五、关键数据流 / 状态流 / shape 流程

## 5.1 Tensor shape 变化

以 Llama/Qwen 的 attention + MLP 为例，假设：

```text
B = batch
S = sequence length
H = hidden size
I = intermediate size
T = tensor_parallel_size
```

### Attention 标准路径

```text
输入:
  hidden_states: [B, S, H]          # 每个 TP rank 都有完整 hidden

q/k/v colwise:
  W_q_i: [H/T, H]
  q_i:   [B, S, H/T]
  k_i:   [B, S, H_kv/T]  # 取决于模型 head / kv head 设置
  v_i:   [B, S, H_kv/T]

local attention:
  attn_i: [B, S, H/T]

output projection rowwise:
  W_o_i: [H, H/T]
  partial_i: [B, S, H]

all_reduce sum:
  hidden_states_next: [B, S, H]
```

真正节省的是 q/k/v 权重、局部 heads 的 attention 中间激活，以及 rowwise 输入激活。恢复冗余发生在 `o_proj` 的 all-reduce 后，每个 rank 又拿到完整 `[B,S,H]`。

### MLP 标准路径

```text
输入:
  hidden_states: [B, S, H]

gate/up colwise:
  gate_i: [B, S, I/T]
  up_i:   [B, S, I/T]

activation/mul local:
  mlp_i: [B, S, I/T]

down_proj rowwise:
  W_down_i: [H, I/T]
  partial_i: [B, S, H]

all_reduce sum:
  hidden_states_next: [B, S, H]
```

MLP 的 intermediate activation 通常很大，`I` 往往是 `H` 的 2.5-4 倍，因此这里的 TP 对激活显存更直观。

### fused qkv / gate_up 的保守路径

Phi3 的源码里，`qkv_proj` 输出后立即按最后一维 slice（`transformers/models/phi3/modeling_phi3.py:238-246`），`gate_up_proj` 输出后 `chunk(2, dim=-1)`（`transformers/models/phi3/modeling_phi3.py:58-64`）。因此 plan 使用：

```text
qkv_proj: colwise_gather_output
o_proj:  rowwise_split_input
gate_up_proj: colwise_gather_output
down_proj: rowwise_split_input
```

shape 变成：

```text
qkv_proj local output before gather:
  [B, S, qkv_dim/T]

all_gather output:
  [B, S, qkv_dim]

model forward slice/chunk:
  query/key/value see full qkv_dim

rowwise_split_input before o_proj:
  [B, S, H] -> [B, S, H/T]
```

这条路径更稳，但通信更多，且 gather 后中间激活显存收益会阶段性消失。

## 5.2 Rank / Mesh / Process Group 变化

假设 `world_size=8`，配置：

```yaml
dp_shard_size: 2
context_parallel_size: 2
tensor_parallel_size: 2
```

Axolotl 的 `_get_parallel_config_kwargs()` 会依次写入：

```text
tp_size = 2
cp_size = 2
dp_shard_size = 2
```

源码在 `src/axolotl/utils/distributed.py:319-370`。Accelerate 的 mesh 维度排序是 `dp_replicate, dp_shard, cp, sp, tp`（`accelerate/parallelism_config.py:260-272`），并在 `build_device_mesh()` 中创建 named mesh（`accelerate/parallelism_config.py:211-244`）。

逻辑上可以理解成：

```text
DeviceMesh shape: [dp_shard=2, cp=2, tp=2]

某个 dp/cp 坐标下的 TP group:
  [..., tp0], [..., tp1]

Transformers initialize_tensor_parallelism():
  如果 device_mesh.ndim > 1:
    要求存在 "tp" dim
    device_mesh = device_mesh["tp"]
```

Transformers 的检查在 `transformers/integrations/tensor_parallel.py:99-107`：多维 mesh 必须包含 `tp` 维，并抽取 `device_mesh["tp"]` 作为 TP 通信 group。

## 5.3 状态切换

TP 主路径里有三类状态：

```text
进程环境状态:
  PARALLELISM_CONFIG_TP_SIZE=N
  ACCELERATE_USE_PARALLELISM_CONFIG=true

模型对象状态:
  model._tp_plan = {...}
  model._tp_size = N
  model._device_mesh = tp mesh

模块 hook 状态:
  module._hf_tp_plan = "colwise" / "rowwise" / ...
  module._hf_device_mesh = device_mesh
  module._is_hooked = True
```

写入者分别是：

- env：`setup_parallelism_envs()`（`src/axolotl/utils/trainer.py:621-640`）；
- `_tp_plan`：`PreTrainedModel.post_init()` 与 `distribute_model()`（`transformers/modeling_utils.py:1287-1333`, `transformers/integrations/tensor_parallel.py:1464-1475`）；
- module hook 状态：`add_tensor_parallel_hooks_to_module()`（`transformers/integrations/tensor_parallel.py:1358-1380`）。

这些状态都是进程内的；没有跨线程隔离语义。模型 hooks 是挂在具体 module 实例上，通常不会污染其他模型实例；但 Axolotl 的 FSDP/Accelerate monkey patch 是替换模块命名空间中的函数，属于全局进程级副作用，这一点在第六章展开。

## 5.4 本章小结

> 💡 **小结**
>
> * 标准 Colwise/Rowwise 在 block 内保持分片，block 输出处恢复完整 hidden。
> * fused qkv/gate_up 会引入 `colwise_gather_output`，用更多通信换取模型源码兼容。
> * 多维并行时 Transformers 只抽取 `DeviceMesh` 的 `tp` 维做 TP 通信。
> * TP 状态分成 env、model attribute、module hook 三层，作用范围不同。

# 六、核心机制深挖

## 6.1 Monkey Patch：Axolotl 没 patch Colwise，但 patch 了组合并行的边界

### 它解决什么问题？

Transformers TP 本身不需要 Axolotl monkey patch：`ColwiseParallel` / `RowwiseParallel` 在 Transformers 内部。Axolotl 的 patch 主要出现在 TP 与 FSDP2、CP、DeepSpeed、QLoRA 等组合时，修复 Accelerate 或保存路径的边界问题。

### 源码怎么实现？

`PatchManager.apply_pre_model_load_patches()` 在模型加载前执行一组 patch（`src/axolotl/loaders/patch_manager.py:95-122`）。其中与并行组合最相关的是 `_apply_fsdp_patches()`：

- 如果有 FSDP config，先 patch missing keys 初始化（`src/axolotl/loaders/patch_manager.py:270-277`）；
- 如果 `context_parallel_size > 1` 或 FSDP2，patch Accelerate 的 parallelism config 校验（`src/axolotl/loaders/patch_manager.py:279-286`）；
- 如果 FSDP2，替换 `accelerate.accelerator.fsdp2_prepare_model` 和 `Accelerator.get_state_dict`（`src/axolotl/loaders/patch_manager.py:287-295`, `src/axolotl/monkeypatch/accelerate/fsdp2.py:529-538`）。

`patch_parallelism_config()` 直接替换 `ParallelismConfig._validate_accelerator` 和 `AcceleratorState.is_fsdp2`（`src/axolotl/monkeypatch/accelerate/parallelism_config.py:73-77`）。这是全局 monkey patch，不是某个模型实例局部状态。

### 隐藏假设与副作用

- patch 在每个 worker 进程内生效；它不是线程局部，也没有自动恢复。
- patch 的条件依赖 Axolotl 配置；例如纯 Transformers TP 路径不需要它，但 FSDP2 + TP/CP 会受影响。
- 维护风险来自上游 Accelerate 内部 API 变化。Axolotl patch 文件名和注释也表明这是 workaround，例如 `src/axolotl/monkeypatch/accelerate/fsdp2.py:2`。

> 容易误解十：`tensor_parallel_size` 开启后一定会走 Axolotl monkey patch。

不一定。TP 的核心 Colwise/Rowwise 不靠 Axolotl patch；Axolotl patch 更多是 FSDP2/CP/保存/QLoRA 组合边界。纯 TP 或 FSDP+TP 的模型层通信主要由 Transformers hooks 处理。

## 6.2 通信原语：前向和反向不是同一个方向

Transformers 把通信封装成 autograd Function，并在文件注释中列出 forward/backward 对偶关系（`transformers/integrations/tensor_parallel.py:429-446`）：

| 函数 | Forward | Backward |
|---|---|---|
| `all_reduce_forward` | all-reduce sum | identity |
| `all_reduce_backward` | identity | all-reduce sum |
| `all_gather` | all-gather | split local chunk |
| `split` | split local chunk | all-gather |
| `reduce_scatter` | reduce-scatter | all-gather |

这解释了 Colwise/Rowwise 为什么能组合：

```text
Colwise q_proj:
  forward: no all-reduce, output local heads
  backward: input gradient all-reduce

Rowwise o_proj:
  forward: partial hidden all-reduce -> full hidden
  backward: no all-reduce on output gradient
```

代码里没有 `all_to_all`。本地 `tensor_parallel.py` 搜索到的是 `all_reduce`、`all_gather`、`reduce_scatter`，未在 TP 主实现中看到 `all_to_all`。这意味着当前 Transformers TP 更接近层内矩阵切分，而不是 token/expert routing 的 all-to-all dispatcher。

## 6.3 配置归一化：用户配置如何变成真实行为

关键路径如下：

```text
YAML tensor_parallel_size
  -> validation: None -> 1
  -> setup_parallelism_envs: PARALLELISM_CONFIG_TP_SIZE
  -> normalize_config: batch_size 按 effective DP 调整
  -> build_parallelism_config: tp_size 写入 DeviceMesh
  -> ModelLoader._build_model: from_pretrained(tp_plan="auto")
```

源码依据：

- 默认值归一：`check_tensor_parallel_size()` 将空值置 1（`src/axolotl/utils/schemas/validation.py:1501-1505`）；
- 8-bit bnb optimizer 不兼容：`check_tensor_parallel_optimizer()`（`src/axolotl/utils/schemas/validation.py:1600-1608`）；
- DeepSpeed 配置补丁：如果存在 `deepspeed` 文件，写入 `tensor_parallel.autotp_size` 和 `gather_16bit_weights_on_model_save`（`src/axolotl/utils/schemas/validation.py:1121-1149`）；
- batch size：`cfg.batch_size *= world_size // cp // tp`（`src/axolotl/utils/config/__init__.py:134-142`）；
- 模型 config 额外约束：`tie_word_embeddings` 与 TP 不兼容（`src/axolotl/loaders/utils.py:139-148`）。

这里有一个很重要的边界：DeepSpeed 配置补丁和 Transformers `tp_plan="auto"` 是两条可能相交但不等价的路径。若用户用 DeepSpeed，Axolotl 还会修改 ds json；若用户用 FSDP2 + TP，核心仍是 Transformers TP 和 Accelerate DeviceMesh。

## 6.4 权重加载与保存：切分发生在初始化，聚合发生在保存

Transformers 的 `from_pretrained()` 在解析 `tp_plan` 后调用 `initialize_tensor_parallelism()`（`transformers/modeling_utils.py:3985-3988`），随后在模型实例化后调用 `distribute_model()` 加 hooks（`transformers/modeling_utils.py:4113-4115`）。真正加载权重时，`convert_and_load_state_dict_in_model()` 会：

1. 为 TP plan 构造 pattern alternation（`transformers/core_model_loading.py:1116-1120`）；
2. 每个 state dict key 如果命中 TP pattern，就创建相应 parallel style 实例（`transformers/core_model_loading.py:1173-1182`）；
3. 调用 `spawn_tp_materialize()`，其 `_job()` 内部执行 `sharding_method.shard_tensor(...)`（`transformers/core_model_loading.py:810-824`）；
4. 最终 `set_param_for_module()` 用 sharded expected shape 校验并替换 module 参数（`transformers/core_model_loading.py:890-923`）。

保存则反过来。`save_pretrained()` 如果发现 `self._tp_size is not None`，调用 `gather_state_dict_for_save()`（`transformers/modeling_utils.py:3360-3363`）。后者对每个命中 plan 的 tensor 通过 `dist.all_gather` 拼回 full tensor（`transformers/integrations/tensor_parallel.py:1257-1284`, `transformers/integrations/tensor_parallel.py:1287-1355`）。packed 权重还会调用 `repack_weights()` 纠正 `[G0,U0,G1,U1]` 这类重建顺序（`transformers/integrations/tensor_parallel.py:1349-1353`）。

## 6.5 本章小结

> 💡 **小结**
>
> * Axolotl patch 的重点是并行组合边界，不是 Colwise/Rowwise 本体。
> * Transformers TP 的通信语义由自定义 autograd Function 定义，forward/backward 不对称但互补。
> * 配置归一化同时影响 env、batch size、DeviceMesh 和 model kwargs。
> * 加载时切 shard，保存时 gather full tensor；这也是显存/CPU 内存峰值的关键来源。

# 七、显存、性能与通信分析

## 7.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 线性层参数 | ✅ | 命中 plan 的 weight 按输出维或输入维切到 TP rank；`shard_tensor()` 在加载时替换本地参数。 |
| 参数梯度 | ✅ | 参数本身是 shard，梯度也随参数 shard 存在。 |
| optimizer state | ✅/取决于 optimizer | 常规每参数 optimizer state 跟随 sharded 参数减少；若再组合 FSDP/ZeRO，会由下游并行策略继续处理。 |
| Attention q/k/v 局部激活 | ✅ | 标准 `colwise` 输出 `[B,S,H/T]`，local heads 计算。 |
| MLP intermediate | ✅ | `gate/up` 输出 `[B,S,I/T]`，`down_proj` 接受局部分片。 |
| block 边界 hidden | ❌ | `rowwise` all-reduce 后每 rank 恢复 `[B,S,H]`。 |
| fused qkv/gate_up 中间输出 | 部分 ❌ | `colwise_gather_output` 会把输出 gather 回完整维度，兼容 slice/chunk 但牺牲阶段性显存收益。 |
| logits | 部分 ✅/❌ | `lm_head` 是 `colwise_gather_output`，weight 可 shard，但输出 logits 会 gather 成完整 vocab。 |
| 保存 state dict | ❌ | `save_pretrained()` 会 gather full tensor；保存峰值显存/CPU 内存上升。 |
| 输入 batch | ❌ | TP 不切 batch；Axolotl 只是避免按 TP rank 放大 global batch。 |

真正的大头取决于模型与训练设置：

- Dense decoder 的 MLP intermediate activation 常常是 TP 受益最明显的激活；
- 对长序列训练，单纯 TP 不切 sequence，仍需 CP/activation checkpointing/flash attention 解决序列激活；
- 若模型大量使用 fused qkv 或 fused gate_up 且 plan 选择 `colwise_gather_output`，显存收益会比标准 Llama/Qwen MLP 路径更小。

## 7.2 通信开销

### 每层通信模式

标准 Llama/Qwen block 可近似看成：

```text
Attention:
  q/k/v colwise:
    forward: local matmul
    backward: all_reduce_backward on input gradient

  o_proj rowwise:
    forward: all_reduce sum partial output
    backward: identity

MLP:
  gate/up colwise:
    forward: local matmul
    backward: all_reduce_backward on input gradient

  down_proj rowwise:
    forward: all_reduce sum partial output
    backward: identity
```

因此每层至少有：

- rowwise 输出处的 forward all-reduce：`o_proj` 一次，`down_proj` 一次；
- colwise 输入处的 backward all-reduce：q/k/v/gate/up 各自对应的输入梯度同步，具体是否可融合取决于 autograd 执行和后端，并未在本地源码中看到显式 overlap/fusion 逻辑；
- 如果 plan 使用 `colwise_gather_output`，还会有 forward all-gather；其 backward 是 split。

### 保存通信

保存阶段 `gather_full_tensor()` 对每个 sharded tensor 执行 `dist.all_gather`（`transformers/integrations/tensor_parallel.py:1279-1284`）。它使用 TP group：如果 mesh 有 `tp` 维，则 `device_mesh.get_group("tp")`（`transformers/integrations/tensor_parallel.py:1271-1273`）。这不是每 step 通信，但在 checkpoint 或 final save 时可能形成串行瓶颈。

### 没有看到的通信

在 Transformers TP 主文件中，本地源码未看到 `all_to_all`。MoE TP 通过 `moe_tp_experts` 对专家输出 all-reduce，而不是 token dispatcher all-to-all（`transformers/integrations/tensor_parallel.py:1130-1161`）。如果读者期待 Megatron-MoE 那种 expert parallel all-to-all，需要区分：本文目标是 `_tp_plan` 的 tensor parallel，不是完整 EP runtime。

## 7.3 性能取舍

这套实现的核心取舍是：

```text
减少每 rank 参数/激活
  换来
更多层内 collective + hook 调度 + 保存 gather
```

几类场景收益不同：

- **标准 Llama/Qwen dense block**：Colwise/Rowwise 成对，通信位置清晰，比较适合高速 NVLink 单机 TP。
- **跨节点 TP**：文档提醒 TP 有频繁小通信，通常不推荐跨慢网络（`docs/nd_parallelism.qmd:30-33`）。多机更适合把 TP 放在节点内，把 DP/FSDP replication 放在节点间。
- **fused qkv/gate_up 模型**：为了兼容源码 slice/chunk，`colwise_gather_output` + `rowwise_split_input` 增加通信，性能收益更依赖模型大小和 interconnect。
- **保存频繁的训练**：checkpoint 会 gather full tensor，保存间隔太短会放大 TP 的通信和 CPU 内存成本。

## 7.4 本章小结

> 💡 **小结**
>
> * TP 节省的是命中 plan 的参数、梯度、optimizer state 和局部激活，不节省 batch 本身。
> * 标准 Colwise/Rowwise 用通信换显存；fused 模型常用 gather/split 再换兼容性。
> * 通信主要是 TP group 内 all-reduce/all-gather/reduce-scatter，未看到主 TP 路径使用 all-to-all。
> * 保存阶段会聚合 full tensor，是显存收益暂时消失和性能瓶颈的高风险点。

# 八、配置项、边界条件与坑点

配置不要孤立看字段，要看它改变了哪条源码路径。

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `tensor_parallel_size: N>1` | `ModelLoader._build_model()` (`src/axolotl/loaders/model.py:749-752`) | 传 `tp_size=N`, `tp_plan="auto"`, `device_mesh` 给 Transformers | 模型必须有可用 `_tp_plan`；否则 plan 缺失或未命中。 |
| `dp_shard_size` / `dp_replicate_size` | `_get_parallel_config_kwargs()` (`src/axolotl/utils/distributed.py:319-370`) | 决定 mesh 中 DP/FSDP 维度 | 纯 DDP + TP/CP 被 Accelerate 禁止（`accelerate/parallelism_config.py:336-341`）。 |
| `context_parallel_size` | `build_parallelism_config()` + `SequenceParallelContextManager` | 与 TP 共同减少 effective DP size；训练时另走 CP hooks | CP 不等于 Transformers 的 `SequenceParallel` style；两套机制不要混淆。 |
| `deepspeed` + `tensor_parallel_size` | validation 修改 ds json (`src/axolotl/utils/schemas/validation.py:1121-1149`) | 写入 `tensor_parallel.autotp_size` 和保存 gather 配置 | 这是 DeepSpeed AutoTP 兼容路径，不等价于 FSDP + Transformers TP。 |
| `optimizer: adamw_8bit` 等 bnb 8-bit | `check_tensor_parallel_optimizer()` (`src/axolotl/utils/schemas/validation.py:1600-1608`) | TP>1 时报错 | 静默省显存不可用，需换 optimizer。 |
| `tie_word_embeddings: true` | `loaders/utils.py` (`src/axolotl/loaders/utils.py:139-148`) | TP>1 直接报错 | 很多 CausalLM 有 `_tied_weights_keys`，但 Axolotl 禁止 config 中 tied embeddings + TP。 |
| `liger_rms_norm` | Liger validation (`src/axolotl/integrations/liger/args.py:86-94`) | TP>1 不兼容 | Qwen3 等 norm 相关 TP 已有特殊梯度同步，Liger RMSNorm 仍被禁。 |
| `liger_fused_linear_cross_entropy` | Liger validation (`src/axolotl/integrations/liger/args.py:108-113`) | TP>1 不兼容 | 不能同时用 Liger loss 和 Transformers TP。 |
| `device_map` | `initialize_tensor_parallelism()` (`transformers/integrations/tensor_parallel.py:47-50`) | 与 `tp_plan` 互斥 | Axolotl 会删除已有 `device_map`，避免冲突。 |
| `quantization_config` | `AutoHfQuantizer.update_tp_plan()` (`transformers/quantizers/auto.py:338-345`) | 某些量化器会改写 base_model_tp_plan | FP8/MXFP4 可能增加 scale/block 的 TP plan，不能只看原始模型 config。 |
| `save_steps` / final save | `save_pretrained()` (`transformers/modeling_utils.py:3360-3363`) | 保存时 gather TP shard | checkpoint 过频会带来 all-gather 和内存峰值。 |

## 8.1 开启该特性的最小配置

最小主线配置是：

```yaml
tensor_parallel_size: 2
```

但实际训练还需要满足：

- world size 能被 TP/CP/DP 组合整除，否则 `_get_parallel_config_kwargs()` 最终会抛不兼容错误（`src/axolotl/utils/distributed.py:364-368`）；
- 模型 config 的 `tie_word_embeddings` 不能为 true（`src/axolotl/loaders/utils.py:139-148`）；
- 模型类必须有可用的 `base_model_tp_plan` 或 `_tp_plan`；`supports_tp_plan` 只检查 plan 是否存在（`transformers/modeling_utils.py:4391-4403`）。

## 8.2 默认行为与静默失效

- `tensor_parallel_size` 默认会被 normalize 成 1（`src/axolotl/utils/schemas/validation.py:1501-1505`），不开启 TP。
- 如果 plan 规则没有命中真实参数，Transformers 只在 warning level 下调用 `verify_tp_plan()` 记录 unused rules / unsharded layers（`transformers/modeling_utils.py:4193-4198`, `transformers/integrations/tensor_parallel.py:1435-1461`）。这类问题可能不是硬错误。
- `base_model_tp_plan` 不写进 config 文件；换一个 Transformers 版本或 remote code 模型时，plan 来源可能改变。

## 8.3 特殊模型限制

- fused qkv/gate_up：更可能使用 `colwise_gather_output` / `rowwise_split_input`，节省较少、通信较多；Phi3 是代表（`transformers/models/phi3/configuration_phi3.py:107-112`）。
- MoE：要求专家数或专家张量 layout 能被 TP style 正确处理。`GroupedGemmParallel` 明确要求 global experts 能被 device mesh size 整除（`transformers/integrations/tensor_parallel.py:1026-1030`）。
- Llama4 这类复合多模态模型存在多份 plan：vision、text、composite config 都可能定义不同规则（如 `transformers/models/llama4/configuration_llama4.py:79-87`, `:233-246`, `:413-415`）。复合 plan 的注册 style 兼容性是维护风险。

## 8.4 本章小结

> 💡 **小结**
>
> * `tensor_parallel_size` 真正影响模型加载参数、mesh、batch size 和保存行为。
> * 字段说明、DeepSpeed AutoTP、Transformers TP 三者要区分。
> * tied embeddings、bnb 8-bit optimizer、Liger loss/RMSNorm 是源码中明确的 TP 风险点。
> * plan 未命中可能只是 warning，需要通过日志和模型结构确认。

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/test_loaders.py:181-218` | `_get_parallel_config_kwargs()` 对 TP/CP/DP 组合的 kwargs 计算 | 覆盖 mesh size 组合逻辑，但不加载真实模型。 |
| `tests/test_tensor_parallel_batch_size.py:25-55` | TP 下 `cfg.batch_size` 使用 effective DP world size | mock 了 `load_model_config` 以绕过模型下载和 tied embeddings 校验。 |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:6-19` | 推荐 FSDP2 + TP + CP 组合 | 示例体现主路径配置。 |
| `examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml:6-22` | HSDP + TP 多机倾向配置 | 文档强调 TP 在节点内，FSDP/DP 跨更高维度。 |
| `docs/nd_parallelism.qmd:28-33` | TP 概念与高速互联要求 | 文档解释 TP 通信频繁，不建议慢互联。 |
| `docs/nd_parallelism.qmd:96-105` | 并行组合支持矩阵 | 覆盖 FSDP+TP/HSDP+TP/FSDP+TP+CP 的预期支持。 |

## 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| Axolotl TP e2e 真实训练 | `tests/e2e/multigpu/test_tp.py:17-20` 被 skip | 标准训练主路径缺少 CI 保护。 |
| Transformers `_tp_plan` 对具体模型是否命中 | Axolotl 未见专门测试 | 模型升级后 plan 规则可能 unused，只发 warning。 |
| 保存 / resume TP checkpoint | 未见针对 TP 的 Axolotl 保存恢复 e2e | `gather_state_dict_for_save()` 或 DeepSpeed/FSDP 保存组合可能回归。 |
| 多机 HSDP + TP | 示例存在，未见 e2e | TP group 跨节点或 mesh 排列错误会显著拖慢或失败。 |
| quantization 修改 TP plan | 未见 Axolotl 侧覆盖 | FP8/MXFP4 scale/block plan 与原模型 plan 冲突可能漏测。 |
| Liger / TP 不兼容错误 | validation 有代码，未见专门表驱动覆盖 | 配置组合变化时可能出现用户才发现的报错。 |
| 性能/显存收益 | 未见基准测试断言 | 无法从测试证明 TP 真正降低峰值显存或提升吞吐。 |
| 保存阶段内存峰值 | 未见压力测试 | 大模型 checkpoint 可能在 gather full tensor 时 OOM。 |

TP 相关 e2e 最值得注意：`tests/e2e/multigpu/test_tp.py` 的 `test_fft_sft` 整体被 `pytest.mark.skip`，理由是 “TP doesn't work with models with tied weights (embeddings)”（`tests/e2e/multigpu/test_tp.py:17-19`）。这并不代表 TP 主路径不可用，而是说明当前 Axolotl 仓库对“真实多 GPU 训练 + 保存 + 指标”这条链路没有持续 CI 保护。

## 9.3 本章小结

> 💡 **小结**
>
> * Axolotl 当前对 TP 的单元测试主要覆盖配置计算和 batch size 归一化。
> * 真实 TP e2e 测试存在但被 skip，主路径缺少持续保护。
> * Transformers 层面的 plan 命中、保存 gather、量化 plan 改写，是更容易在版本升级中回归的部分。
> * 示例和文档能说明推荐配置，但不能替代执行级验证。

# 十、局限性与已知优化点

## 10.1 硬约束

1. **world size 组合必须匹配。** `_get_parallel_config_kwargs()` 会逐步除以 TP/CP/DP size，剩余 world size 不能消化时抛错（`src/axolotl/utils/distributed.py:330-368`）。
2. **纯 DDP + TP/CP 不支持。** Accelerate 在 `dp_replicate_size > 1` 且 `dp_shard_size == 1` 时禁止 TP/CP（`accelerate/parallelism_config.py:336-341`）。
3. **`device_map` 与 `tp_plan` 互斥。** Transformers 初始化时直接抛错（`transformers/integrations/tensor_parallel.py:47-50`）。
4. **tied embeddings 被 Axolotl 禁止。** `tie_word_embeddings` + TP>1 会抛 ValueError（`src/axolotl/loaders/utils.py:139-148`）。
5. **部分 optimizer / Liger 组合不兼容。** bnb 8-bit optimizer、`liger_rms_norm`、`liger_fused_linear_cross_entropy` 都有显式校验（`src/axolotl/utils/schemas/validation.py:1600-1608`, `src/axolotl/integrations/liger/args.py:86-113`）。
6. **MoE 专家数/shape 有整除约束。** `GroupedGemmParallel` 要求 expert 数能被 mesh size 整除（`transformers/integrations/tensor_parallel.py:1026-1030`）。

## 10.2 维护成本

- **模型 plan 依赖命名。** plan key 如 `layers.*.self_attn.q_proj` 与模型源码命名强绑定。上游重命名或 remote code 差异会导致 plan unused。
- **hook 注入是隐式执行。** 模型 forward 源码看不到通信，读代码时必须同时看 `_tp_plan` 和 `tensor_parallel.py`。
- **Axolotl patch 是全局替换。** FSDP2/Accelerate patch 改的是模块级函数，不是局部 context manager，升级上游时维护成本高。
- **配置描述可能落后。** `tensor_parallel_size` schema 仍写 DeepSpeed AutoTP，但主路径已经包含 Transformers TP。
- **Transformers 版本差异。** Axolotl pin 和本地安装版本可能不一致；`_tp_plan` 支持矩阵会随 Transformers 版本变化。

## 10.3 性能瓶颈

- **每层 collective 多。** 标准 block 至少在 `o_proj`、`down_proj` forward 做 all-reduce；backward 还有 colwise 输入梯度 all-reduce。
- **fused 模型 gather/split 增加通信。** `colwise_gather_output` 牺牲中间激活分片，`rowwise_split_input` 又引入 split backward all-gather。
- **保存时逐 tensor all-gather。** `gather_state_dict_for_save()` 对 state dict 逐项处理，可能在大模型 checkpoint 时成为瓶颈。
- **缺少显式 overlap/fusion。** 本地源码未看到 TP collective 与计算的显式 overlap 或通信融合逻辑；性能依赖 PyTorch/NCCL 调度。
- **跨节点 TP 不友好。** 文档已经指出 TP 通信频繁，需要快速互联（`docs/nd_parallelism.qmd:30-33`）。

## 10.4 已知优化点

基于源码行为，可以看到几类优化方向：

1. **更细粒度或模型特定 fused plan。** 对 fused qkv/gate_up，减少 `colwise_gather_output` 的保守 gather，需要模型 forward 能接受分片 q/k/v 或修改 fused op。
2. **保存分块/异步 gather。** `gather_state_dict_for_save()` 当前按 tensor all-gather，未来可考虑分块、流式写 safetensors 或只在 rank0 聚合需要的 shard。
3. **通信融合与 overlap。** 将多个 colwise backward all-reduce 或 rowwise forward all-reduce 合并/overlap，可能降低小 collective 开销。
4. **测试补强。** 恢复一个不 tied embeddings 的 TP e2e，并覆盖 save/resume、FSDP+TP、quantization update_tp_plan。
5. **配置文档同步。** 更新 `tensor_parallel_size` schema 描述，明确 Transformers TP 与 DeepSpeed AutoTP 两条路径。

## 10.5 本章小结

> 💡 **小结**
>
> * TP 的硬约束来自 world size、模型 plan、embedding tie、optimizer 和特定 kernel 组合。
> * 维护风险主要在模型命名、hook 隐式行为、全局 monkey patch 和 Transformers 版本漂移。
> * 性能瓶颈不是 matmul 本身，而是层内小 collective、fused 模型 gather/split 和保存 gather。
> * 后续优化应优先补 e2e/save 测试，再考虑通信融合和保存路径优化。

# 小结与展望

`Axolotl` 的 Transformers `_tp_plan` 接入可以用几个关键词概括。

## 关键词一：配置转译

Axolotl 不重写 Tensor Parallel。它把 YAML 中的 `tensor_parallel_size` 转译成 Accelerate `ParallelismConfig`、`DeviceMesh` 和 Transformers `from_pretrained(tp_plan="auto")`。这层设计的好处是复用 Transformers 模型生态；代价是 Axolotl 对具体模型能不能 TP 的控制力有限。

## 关键词二：模型内置计划表

`_tp_plan` / `base_model_tp_plan` 是模型作者把 shape 假设写成并行规则的地方。Llama 的 q/k/v + gate/up 可以自然 colwise，o/down rowwise；Qwen3 额外处理 q_norm/k_norm 梯度；Phi3 这类 fused 模型必须 gather 再 split；Mixtral/Qwen3-MoE 需要 packed 和 MoE 专家特殊 style。理解 TP，不能只看通用 `ColwiseParallel`，还要看具体模型 plan。

## 关键词三：Hook 注入

Transformers 通过 module forward pre-hook/post-hook 注入通信，而不是修改每个模型 forward。这个设计减少侵入，但也让通信边界变得不显眼：读 `modeling_llama.py` 时你看不到 all-reduce，必须同时看 `model._tp_plan` 和 `tensor_parallel.py`。

## 关键词四：通信换显存

TP 的收益来自参数、梯度、optimizer state 和局部激活分片；代价是 TP group 内频繁 all-reduce/all-gather。标准 decoder block 比较适合这套模式，fused qkv/gate_up 或 MoE 则会引入更多兼容性通信。保存时还要 gather full tensor，训练中节省的显存不等于 checkpoint 阶段也省。

## 关键词五：组合并行的边界成本

在 Axolotl 中，TP 往往不是单独出现，而是与 FSDP2、CP、DeepSpeed、量化、LoRA、Liger 等组合。组合越多，边界 patch、保存路径、optimizer 兼容性和测试缺口就越重要。Axolotl 的价值在于把这些开关放进统一 YAML；风险也在于一行配置会跨越多个下游库的状态机。

总体来看，这套实现适合：模型已经在 Transformers 中有成熟 `_tp_plan`、GPU 间互联足够快、希望与 FSDP2/CP 组合的大模型训练。它不适合：慢速跨节点 TP、模型 remote code 没有稳定 plan、依赖 tied embeddings 或 Liger loss 的配置、以及频繁保存超大 full checkpoint 的场景。

和 Megatron-LM 这类深度侵入式训练框架相比，Transformers `_tp_plan` 的优势是接入轻、模型生态广、能和 HuggingFace 加载/保存体系复用；劣势是 hook 隐式、模型 plan 维护成本高、通信优化空间有限。后续值得继续走读的方向，是 Transformers 中量化器如何改写 `base_model_tp_plan`、FSDP2 + TP 的 state dict 如何在 Accelerate 中交接，以及 Axolotl 是否能补齐 TP save/resume 的 e2e 保护。
