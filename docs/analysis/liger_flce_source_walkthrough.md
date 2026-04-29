# Axolotl 源码走读：Liger fused linear cross entropy 实现解析

在大词表语言模型训练里，最后一层 `lm_head -> logits -> cross entropy` 往往是一个“不做计算也会 OOM”的环节：`hidden_states` 只有 `[B, T, H]`，但一旦乘上 `lm_head.weight.T`，就会展开成 `[B, T, V]`。当 `V` 是 100K、`B*T` 又因为长序列或 sample packing 被拉大时，这个 logits 张量会成为 loss 阶段最醒目的显存峰值。

Axolotl 里同时存在几类瞄准这个问题的方案：`chunked_cross_entropy`、Cut Cross Entropy（下文简称 CCE）、Liger Cross Entropy，以及本文关注的 **Liger fused linear cross entropy（FLCE）**。它们看上去都在“省 loss 显存”，但真正接入训练链路的位置、能覆盖的模型、反向语义、分布式行为并不一样。本文不讲 Liger 论文或 CCE 论文的数学推导，只沿着 Axolotl 源码追踪：用户配置如何触发 Liger FLCE，patch 在模型加载前如何生效，forward 为什么返回 `logits=None`，以及它和 CCE 在工程边界上的差异。

# 前言

## 业务 / 工程背景

Liger FLCE 出现在 **SFT / causal LM 训练的 loss 计算末端**。它解决的不是参数分片，也不是 attention 的 KV/score 显存，而是把最后一层线性投影和交叉熵融合起来，避免标准路径中长期持有完整 logits：

```text
标准路径:
  hidden_states [B, T, H]
    -> lm_head
  logits        [B, T, V]
    -> cross_entropy(labels)
  loss          []

Liger FLCE 路径:
  hidden_states [B, T, H]
  lm_head.weight [V, H]
  labels [B, T]
    -> fused linear + CE kernel
  loss []
  logits = None
```

## 核心矛盾

这个特性的核心矛盾可以概括为三句话：

1. loss 只需要目标 token 的概率和归一化项，但普通实现会先物化整个 `[B*T, V]` logits；
2. 如果想绕开 logits，就必须改写模型 `forward`，因为 HuggingFace 模型通常先返回 logits 再交给 loss 函数；
3. 改写 `forward` 是一个模型结构相关的 monkey patch：它省显存，但维护成本、模型覆盖、与 TP/FSDP/DeepSpeed 的边界都要重新审视。

## 本文主线

本文按机制而不是文件展开：

1. 先看用户如何开启 Liger FLCE，以及插件参数如何进入 Pydantic 校验；
2. 再看 patch 为什么必须发生在模型加载前，以及 Axolotl 如何选择 Liger upstream patch、Axolotl 自维护 patch 或 generic fallback；
3. 然后进入 forward 主路径，分析 shape、状态、logits 是否物化、FSDP 下 lm_head 的特殊处理；
4. 接着把它和 CCE 放在同一条 loss 显存主线里比较：两者都省 logits，但 patch 点、覆盖模型和通信语义不同；
5. 最后分析显存、通信、配置坑、测试覆盖和维护风险。

## 不展开的内容

本文不展开 Triton 编程模型、Liger Kernel 完整算子库、FSDP/ZeRO 原理、LoRA 原理，也不做性能 benchmark。所有判断只基于当前仓库 `/root/axolotl`、本环境安装的 `liger-kernel==0.7.0`，以及 Axolotl CCE 插件明确要求的 fork `axolotl-ai-cloud/ml-cross-entropy@fec1a88` 的源码行为。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/config.py` | 读取 YAML，注册插件，将插件参数并入配置校验入口 |
| `src/axolotl/integrations/base.py` | `PluginManager` 生命周期，负责调用 `pre_model_load` 等 hook |
| `src/axolotl/integrations/liger/args.py` | Liger 配置项、互斥和 TP 兼容性校验 |
| `src/axolotl/integrations/liger/plugin.py` | Liger 插件主入口：torch.compile shim、token scaling patch、模型 forward patch 分发 |
| `src/axolotl/integrations/liger/models/base.py` | Axolotl generic FLCE forward，含 FSDP/PEFT lm_head 处理 |
| `src/axolotl/integrations/liger/models/qwen3.py` | Axolotl 为 qwen3 自维护的 FLCE forward 示例 |
| `/usr/local/lib/python3.12/dist-packages/liger_kernel/transformers/model/loss_utils.py` | Liger `LigerForCausalLMLoss`：shift、flatten、调用 fused loss |
| `/usr/local/lib/python3.12/dist-packages/liger_kernel/ops/fused_linear_cross_entropy.py` | Liger FLCE autograd Function 与分块 kernel 语义 |
| `src/axolotl/integrations/cut_cross_entropy/__init__.py` | CCE 插件入口，用于对比 patch 点和覆盖范围 |
| `/tmp/ml-cross-entropy-axolotl/cut_cross_entropy/transformers/utils.py` | CCE `apply_lce`：DTensor vocab parallel 与 ZeRO-3 处理，用于对比通信语义 |

# 一、入口与配置：用户打开的不是一个 loss 函数，而是一次模型命名空间改写

## 1.1 设计哲学与核心问题

Liger FLCE 在 Axolotl 里不是通过 `Trainer.compute_loss` 临时切换 loss，也不是在 batch 进入 trainer 后包一层函数。用户开启它时，真正改变的是 **模型类的 `forward` 实现**。这要求配置系统先完成两件事：

- 让 `liger_fused_linear_cross_entropy` 这样的插件字段成为合法配置；
- 在模型实例化前注册插件并执行 `pre_model_load`，否则模型类已经创建，类级 patch 可能错过实例化时机。

如果没有这层配置合并，Axolotl 的基础 schema 并不知道 `liger_fused_linear_cross_entropy`；如果 patch 晚于模型加载，部分模型的类替换、模块命名空间替换就不会覆盖已经创建出来的对象。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：Click 命令入口，接收 config 路径和 launcher 参数

src/axolotl/cli/train.py
  - do_cli：调用 load_cfg，再进入 do_train
  - do_train：加载 dataset，调用 axolotl.train.train

src/axolotl/cli/config.py
  - load_cfg：读取 YAML，prepare_plugins，然后 validate_config
  - prepare_plugins：把 plugins 中的类注册到 PluginManager

src/axolotl/integrations/config.py
  - merge_input_args：把插件的 Pydantic args 动态混入 AxolotlInputConfig

src/axolotl/integrations/liger/args.py
  - LigerArgs：定义 liger_fused_linear_cross_entropy 等字段和校验
```

## 1.3 主流程拆解

用户最小配置大致是：

```yaml
plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_fused_linear_cross_entropy: true
bf16: true   # 或 fp16/auto；Liger 本身不在 args.py 里强制，但实际训练通常如此
```

命令入口从 Click 开始：

```text
axolotl train config.yml
  -> src/axolotl/cli/main.py:98 train(...)
    -> launch_training(...)
      -> python -m axolotl.cli.train config.yml
        -> src/axolotl/cli/train.py:55 do_cli(...)
          -> load_cfg(config)
```

`main.py` 的 `train` 命令在 `src/axolotl/cli/main.py:78-128` 定义，真正进入训练子进程后，`do_cli` 在 `src/axolotl/cli/train.py:55-91` 调用 `load_cfg` 并进入 `do_train`。

配置读取的关键顺序在 `load_cfg`：

```python
# src/axolotl/cli/config.py:249-253
with open(config, encoding="utf-8") as file:
    cfg: DictDefault = DictDefault(yaml.safe_load(file))
cfg.axolotl_config_path = config

# src/axolotl/cli/config.py:306-320
prepare_plugins(cfg)
cfg = validate_config(cfg, capabilities=..., env_capabilities=...)
```

这里的顺序非常关键：`prepare_plugins(cfg)` 先根据 YAML 中的 `plugins` 注册 `LigerPlugin`，随后 `validate_config` 才能通过 `merge_input_args()` 把 `LigerArgs` 合并进 schema。`merge_input_args` 的动态继承逻辑在 `src/axolotl/integrations/config.py:27-57`：它从 `PluginManager.get_input_args()` 收集插件参数类，然后生成新的 `AxolotlInputConfig` / `AxolotlConfigWCapabilities`。

`LigerPlugin` 暴露参数类的方式非常小：

```python
# src/axolotl/integrations/liger/plugin.py:19-20
class LigerPlugin(BasePlugin):
    def get_input_args(self):
        return "axolotl.integrations.liger.LigerArgs"
```

真正的字段定义在 `LigerArgs`：

```python
# src/axolotl/integrations/liger/args.py:31-47
liger_rope: bool | None = None
liger_rms_norm: bool | None = None
liger_layer_norm: bool | None = None
liger_swiglu: bool | None = None
liger_glu_activation: bool | None = None
liger_cross_entropy: bool | None = None
liger_fused_linear_cross_entropy: bool | None = None
liger_use_token_scaling: bool | None = Field(default=None, ...)
```

这说明一个重要事实：**Liger 插件只是把字段变成合法配置；具体 patch 还没有发生。** 第一个真正改变训练行为的函数不是 `validate_config`，而是后面模型加载前的 `LigerPlugin.pre_model_load()`。

## 1.4 关键细节与误区澄清

> 容易误解一：只写 `liger_fused_linear_cross_entropy: true` 就会生效。

不一定。Axolotl 的插件字段是通过 `plugins` 注册后动态并入 schema 的。`load_cfg` 先在 `src/axolotl/cli/config.py:306` 调用 `prepare_plugins`，然后 `validate_config` 在 `src/axolotl/utils/config/__init__.py:332-336` 才会调用 `merge_input_args()`。如果没有：

```yaml
plugins:
  - axolotl.integrations.liger.LigerPlugin
```

那么 `LigerArgs` 不会作为插件输入参数参与校验；后续也不会执行 `LigerPlugin.pre_model_load()`。

> 容易误解二：`liger_use_token_scaling` 是一个独立 loss 开关。

不是。`LigerArgs.check_liger_use_token_scaling_flce` 在 `src/axolotl/integrations/liger/args.py:96-106` 明确要求它只能在 `liger_fused_linear_cross_entropy` 开启时使用。它改变的是 FLCE 内部 scaling 行为，不是另一个 loss 路径。

> 容易误解三：Liger FLCE 可以和 CCE / chunked CE 一起叠加。

不能。全局校验 `check_cross_entropy_conflicts` 在 `src/axolotl/utils/schemas/validation.py:974-1002` 把 `cut_cross_entropy`、`chunked_cross_entropy`、`liger_cross_entropy`、`liger_fused_linear_cross_entropy` 视为互斥项，只允许一个开启。

## 1.5 本章小结

> 💡 **小结**
>
> * Liger FLCE 的用户入口是插件配置，不是 trainer loss 参数。
> * 插件参数通过 `get_input_args -> merge_input_args -> validate_config` 动态进入 schema。
> * 第一个真正改变执行行为的函数是模型加载前的 `LigerPlugin.pre_model_load()`。
> * FLCE、Liger CE、CCE、chunked CE 在 Axolotl 中是互斥关系，不是可叠加优化。

# 二、初始化与 patch 分发：为什么必须在模型加载前动手

## 2.1 设计哲学与核心问题

FLCE 要绕过 logits，必须把模型 `forward` 中这段逻辑：

```text
hidden_states -> self.lm_head(...) -> logits -> self.loss_function(...)
```

替换成：

```text
hidden_states + self.lm_head.weight + labels -> LigerForCausalLMLoss -> loss
logits = None
```

这不是一个后置 hook 能优雅完成的事情，因为标准 HuggingFace 模型会在自己的 `forward` 内部决定是否生成 logits、如何 shift labels、如何返回 `CausalLMOutputWithPast`。因此 Axolotl 选择在 `ModelLoader.load()` 的 **模型构建前** 调用插件，让 Liger 有机会改写 transformers 模块命名空间里的类或函数。

## 2.2 源码入口与关键对象

```text
src/axolotl/loaders/model.py
  - ModelLoader.load：在 _build_model 前调用 PLUGIN_MANAGER.pre_model_load

src/axolotl/integrations/base.py
  - PluginManager.pre_model_load：顺序调用所有插件 pre_model_load

src/axolotl/integrations/liger/plugin.py
  - LigerPlugin.pre_model_load：执行 Liger 兼容 shim、token scaling patch、模型类型分发
```

## 2.3 主流程拆解

模型加载主路径在 `ModelLoader.load`：

```python
# src/axolotl/loaders/model.py:168-177
self.patch_manager.apply_pre_model_load_patches()
self._apply_pre_model_load_setup()

PLUGIN_MANAGER.pre_model_load(self.cfg)
self.patch_manager.apply_post_plugin_pre_model_load_patches()

skip_move_to_device = self._build_model()
```

这段顺序解释了 Liger FLCE 的生命周期：

```text
ModelLoader.load
  -> Axolotl 内置 pre-model patches
  -> _apply_pre_model_load_setup
  -> PluginManager.pre_model_load(cfg)
       -> LigerPlugin.pre_model_load(cfg)
          -> 选择并改写 transformers 模型类 forward
  -> _build_model()
       -> AutoModelForCausalLM.from_pretrained(...)
       -> 实例化时已经拿到 patched class / patched symbols
```

`PluginManager.pre_model_load` 本身只是一个顺序分发器：

```python
# src/axolotl/integrations/base.py:439-446
for plugin in self.plugins.values():
    plugin.pre_model_load(cfg)
```

Liger 插件的 `pre_model_load` 做了几类初始化动作：

1. **TRL 兼容 shim**：`liger-kernel 0.7.0` 从旧路径导入 `ORPOTrainer`，Axolotl 在 `src/axolotl/integrations/liger/plugin.py:22-27` 把 `trl.experimental.orpo.ORPOTrainer` 塞回 `trl.trainer` 命名空间。
2. **torch.compile shim**：如果 `cfg.torch_compile` 为真，Axolotl 会用 `torch.compiler.disable` 包装 Liger FLCE 的 forward/backward kernel 函数（`src/axolotl/integrations/liger/plugin.py:29-42`，包装器定义在 `src/axolotl/integrations/liger/utils.py:10-29`）。
3. **token scaling patch**：如果 `liger_use_token_scaling` 为真，改写 Liger 的函数 API 和 `LigerFusedLinearCrossEntropyLoss.__init__`，强制传入 `use_token_scaling=True`（`src/axolotl/integrations/liger/plugin.py:57-82`）。
4. **模型类型分发**：先尝试 Liger upstream 的 `MODEL_TYPE_TO_APPLY_LIGER_FN`，再走 Axolotl 自维护特殊分支，最后在 `liger_fused_linear_cross_entropy` 开启时尝试 generic fallback（`src/axolotl/integrations/liger/plugin.py:84-290`）。

核心分发逻辑可以简化为：

```text
if cfg.model_config_type in liger_kernel.MODEL_TYPE_TO_APPLY_LIGER_FN:
    inspect signature
    apply_liger_fn(rope=..., cross_entropy=..., fused_linear_cross_entropy=..., ...)
elif model_config_type in Axolotl-maintained special cases:
    patch that model's forward / norm / MLP symbols
elif cfg.liger_fused_linear_cross_entropy:
    patch_lce_forward(cfg.model_config_type)  # generic experimental
else:
    warning unsupported
```

这里的 `inspect.signature` 很实用：不同 Liger upstream patch 函数接受 `swiglu`、`geglu`、`rope`、`layer_norm` 的组合不完全一致，Axolotl 只传该函数签名中存在的参数（`src/axolotl/integrations/liger/plugin.py:84-105`）。

## 2.4 关键细节与误区澄清

> 容易误解一：`pre_model_load` 是“加载模型之后做一些准备”。

在 Axolotl 这里恰好相反。`ModelLoader.load` 在 `_build_model()` 之前调用 `PLUGIN_MANAGER.pre_model_load`（`src/axolotl/loaders/model.py:168-176`）。Liger FLCE 依赖这一点：它改写的是类或模块命名空间，让后续 `from_pretrained` 创建出的模型直接拥有新的 `forward`。

> 容易误解二：Liger 的 patch 都来自 `liger-kernel` 上游。

不是。`cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN` 时确实走 Liger 上游映射（`src/axolotl/integrations/liger/plugin.py:84-105`），但 Axolotl 还维护了 `jamba`、`deepseek_v2`、`llama4`、`qwen3`、`qwen3_5`、`qwen3_moe`、`qwen3_5_moe`、`gemma4`、`granitemoe` 等分支（`plugin.py:106-275`）。如果仍不命中且只开了 FLCE，才会走 `models/base.py` 的 generic patch（`plugin.py:275-286`）。

> 容易误解三：patch 是可恢复的 context manager。

不是。`LigerPlugin.pre_model_load` 直接给模块或类赋值，例如 `modeling_qwen3.Qwen3ForCausalLM.forward = lce_forward`（`src/axolotl/integrations/liger/models/qwen3.py:157-158`）。没有 `__enter__/__exit__`，也没有恢复旧 forward 的逻辑。它是进程内全局生效的 monkey patch。

## 2.5 本章小结

> 💡 **小结**
>
> * Liger FLCE 必须在模型加载前 patch，因为它改变的是模型类 `forward`。
> * Axolotl 的 patch 分发有三层：Liger upstream map、Axolotl 自维护特殊模型、generic experimental fallback。
> * `torch_compile` 和 `liger_use_token_scaling` 都是初始化期的命名空间改写，不是每 step 动态开关。
> * patch 没有自动恢复机制，测试和多模型同进程使用时要警惕污染。

# 三、Forward 主路径：省显存的关键不是 cross entropy，而是“不返回 logits”

## 3.1 设计哲学与核心问题

普通 cross entropy 优化只能从 `logits` 已经存在之后开始省；Liger FLCE 的野心更靠前：**不要构造完整 logits**。这也是它名字里 “fused linear cross entropy” 的含义：最后一个线性层和 CE 被视为一个 autograd Function。

这层要解决的是显存问题，但副作用是 API 语义发生变化：训练时 `outputs.logits` 可能是 `None`。如果下游 trainer、callback 或指标逻辑假设 logits 一定存在，就会踩坑。

## 3.2 源码入口与关键对象

```text
src/axolotl/integrations/liger/models/qwen3.py
  - lce_forward：Axolotl 自维护 qwen3 FLCE forward 示例

src/axolotl/integrations/liger/models/base.py
  - lce_forward：generic FLCE forward
  - lce_maybe_trainable_lm_head：PEFT / FSDP lm_head 处理
  - _liger_for_causal_lm_loss：调用 LigerForCausalLMLoss

/usr/local/lib/python3.12/dist-packages/liger_kernel/transformers/model/loss_utils.py
  - LigerForCausalLMLoss：shift labels、flatten token、调用 fused loss

/usr/local/lib/python3.12/dist-packages/liger_kernel/ops/fused_linear_cross_entropy.py
  - LigerFusedLinearCrossEntropyFunction：forward 里计算 loss 和梯度缓存
```

## 3.3 主流程拆解

以 Axolotl 自维护的 qwen3 patch 为例，`lce_forward` 的关键逻辑是：

```python
# src/axolotl/integrations/liger/models/qwen3.py:57-83
outputs = self.model(...)
hidden_states = outputs[0]

logits = None
loss = None
if self.training and (labels is not None):
    loss = LigerForCausalLMLoss(
        hidden_states=hidden_states,
        lm_head_weight=self.lm_head.weight,
        labels=labels,
        hidden_size=self.config.hidden_size,
        **kwargs,
    )
else:
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, ...)
```

可以画成这样：

```text
训练 + labels 存在:
  input_ids [B,T]
    -> self.model
  hidden_states [B,T,H]
    -> LigerForCausalLMLoss(hidden_states, lm_head.weight [V,H], labels [B,T])
  loss []
  logits = None

推理 / 无 labels / 显式不跳过 logits:
  hidden_states [B,T,H]
    -> self.lm_head(hidden_states[:, slice_indices, :])
  logits [B,T_keep,V]
  optional standard loss
```

Liger 的 `LigerForCausalLMLoss` 继续做 shift 和 flatten：

```python
# /usr/local/.../liger_kernel/transformers/model/loss_utils.py:80-91
if shift_labels is None:
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

hidden_states = hidden_states.view(-1, hidden_size)
shift_labels = shift_labels.view(-1)
shift_labels = shift_labels.to(hidden_states.device)
result = fixed_fused_linear_cross_entropy(hidden_states, lm_head_weight, shift_labels, ...)
```

这一步把 `[B,T,H]` 变成 `[B*T,H]`，labels 变成 `[B*T]`。真正省掉的是标准路径中的 `[B*T,V]` logits。Liger op 文件里的注释把这个显存来源写得很直接：

```python
# /usr/local/.../liger_kernel/ops/fused_linear_cross_entropy.py:41-48
# inputs have shape: BT x H
# materialized activations will have shape: BT x V
# the increase in memory = BT x V
BT, H = _input.shape
V = weight.shape[0]
```

随后 Liger 按 token chunk 计算：

```python
# /usr/local/.../liger_kernel/ops/fused_linear_cross_entropy.py:52-55
inc_factor = triton.cdiv(V, H)
chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
num_chunks = triton.cdiv(BT, chunk_size)
```

每个 chunk 内仍会临时算 `logits_chunk = _input_chunk @ weight.t()`（`fused_linear_cross_entropy.py:91-99`），但它不会作为完整 `[BT,V]` 长期保留到 autograd backward。loss kernel 在 `fused_linear_cross_entropy.py:144-174` 直接把 cross entropy 梯度写回 `logits_chunk`，再用它累积：

```text
for chunk:
  logits_chunk [chunk, V]
    -> CE kernel in-place 写成 grad_logits_chunk
  grad_input[chunk] = grad_logits_chunk @ weight
  grad_weight += grad_logits_chunk.T @ input_chunk
```

对应源码在 `fused_linear_cross_entropy.py:187-199`。Autograd Function 在 forward 结束前保存的是 `grad_input / grad_weight / grad_bias`，不是保存输入再在 backward 重新展开 logits：

```python
# /usr/local/.../liger_kernel/ops/fused_linear_cross_entropy.py:324-348
loss, z_loss, token_accuracy, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(...)
ctx.save_for_backward(grad_input.detach(), grad_weight.detach() if ... else None, grad_bias.detach() if ... else None)
return loss, z_loss, token_accuracy
```

backward 主要处理外部 `grad_output` 的缩放：

```python
# /usr/local/.../liger_kernel/ops/fused_linear_cross_entropy.py:352-360
(grad_input, grad_weight, grad_bias) = ctx.saved_tensors
grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
    grad_output, grad_input, grad_weight, grad_bias
)
```

## 3.4 关键细节与误区澄清

> 容易误解一：FLCE 完全不计算 logits。

不准确。它不物化完整生命周期的 `[BT,V]` logits，但每个 chunk 内仍会临时计算 `logits_chunk = _input_chunk @ weight.t()`（`fused_linear_cross_entropy.py:96-99`）。显存收益来自 chunk 化和不把 logits 保存给 backward，而不是数学上完全跳过 logits。

> 容易误解二：`liger_cross_entropy` 和 `liger_fused_linear_cross_entropy` 只是同一个 kernel 的两个名字。

不是。`liger_cross_entropy` 在多个 patch 函数中替换的是 `nn.functional.cross_entropy` 或 `CrossEntropyLoss`（如 `src/axolotl/integrations/liger/models/qwen3.py:152-155`）；它仍然以 logits 为输入。`liger_fused_linear_cross_entropy` 替换的是模型 `ForCausalLM.forward`（如 `qwen3.py:157-158`），目标是把 `lm_head` 和 CE 融合，避免主路径返回 logits。

> 容易误解三：训练时 `outputs.logits` 一定存在。

Liger FLCE 主路径下不成立。`qwen3.lce_forward` 初始化 `logits = None`，训练且有 labels 时只计算 loss（`qwen3.py:73-83`），最后返回 `CausalLMOutputWithPast(logits=logits, ...)`（`qwen3.py:100-106`）。依赖 logits 做指标、蒸馏或额外 loss 的逻辑不能假设这个字段非空。

## 3.5 本章小结

> 💡 **小结**
>
> * Liger FLCE 的显存收益来自跳过完整 logits 的物化和 backward 保存。
> * kernel 内部仍有 chunk 级 logits，但生命周期短得多。
> * training + labels 是 FLCE 主路径；推理或无 labels 会回到标准 `lm_head -> logits`。
> * `liger_cross_entropy` 是替换 CE，`liger_fused_linear_cross_entropy` 是替换模型 forward，二者工程语义不同。

# 四、模型覆盖与 fallback：同样叫 FLCE，不同模型走的 patch 路径并不相同

## 4.1 设计哲学与核心问题

HuggingFace 的 `ForCausalLM.forward` 并没有统一到足以“一把梭”的程度。不同模型会有不同输出类型、MoE auxiliary loss、multimodal 参数、`logits_to_keep` 命名、`return_dict` 处理、softcap、router logits 等差异。FLCE 的难点不是调用一个 loss kernel，而是让替换后的 forward 仍然像原模型。

因此 Axolotl 的覆盖策略不是单一路径，而是“能用上游就用上游，不够就本地补，最后才 generic”。

## 4.2 源码入口与关键对象

```text
src/axolotl/integrations/liger/plugin.py
  - pre_model_load：模型类型分发

src/axolotl/integrations/liger/models/qwen3.py
src/axolotl/integrations/liger/models/qwen3_5.py
src/axolotl/integrations/liger/models/qwen3_moe.py
src/axolotl/integrations/liger/models/qwen3_5_moe.py
src/axolotl/integrations/liger/models/deepseekv2.py
src/axolotl/integrations/liger/models/jamba.py
src/axolotl/integrations/liger/models/base.py
  - 各模型自定义或 generic forward patch

/usr/local/lib/python3.12/dist-packages/liger_kernel/transformers/monkey_patch.py
  - MODEL_TYPE_TO_APPLY_LIGER_FN：Liger upstream 支持表
```

## 4.3 主流程拆解

Axolotl 首先查询 Liger upstream 的 `MODEL_TYPE_TO_APPLY_LIGER_FN`：

```python
# src/axolotl/integrations/liger/plugin.py:84-105
if cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN:
    apply_liger_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[cfg.model_config_type]
    liger_fn_sig = inspect.signature(apply_liger_fn)
    kwargs = {}
    if "fused_linear_cross_entropy" in liger_fn_sig.parameters:
        kwargs["fused_linear_cross_entropy"] = cfg.liger_fused_linear_cross_entropy
    ...
    apply_liger_fn(**kwargs)
```

本环境安装的 `liger-kernel` 中，upstream map 有 41 个 model type，例如 `llama`、`mistral`、`mixtral`、`gemma`、`qwen2`、`qwen3`、`qwen3_moe`、`qwen3_vl` 等。对应映射定义在 `/usr/local/lib/python3.12/dist-packages/liger_kernel/transformers/monkey_patch.py`，例如 Liger upstream 的 `apply_liger_kernel_to_llama` 会在开启 FLCE 时替换 `modeling_llama.LlamaForCausalLM.forward`（同文件 `:217-264`）。

但 Axolotl 仍然维护了自己的分支。比如 qwen3 在 `plugin.py:165-176` 会调用：

```python
apply_liger_kernel_to_qwen3(
    cross_entropy=cfg.liger_cross_entropy,
    fused_linear_cross_entropy=cfg.liger_fused_linear_cross_entropy,
    glu_activation=cfg.liger_glu_activation,
    rms_norm=cfg.liger_rms_norm,
    layer_norm=cfg.liger_layer_norm,
)
```

对应实现中，开启 FLCE 后替换的是：

```python
# src/axolotl/integrations/liger/models/qwen3.py:157-158
if fused_linear_cross_entropy:
    modeling_qwen3.Qwen3ForCausalLM.forward = lce_forward
```

MoE 模型要保留 auxiliary loss。例如 `qwen3_moe` 在训练 loss 后继续算 router auxiliary loss：

```text
hidden_states -> LigerForCausalLMLoss -> loss
outputs.router_logits -> load_balancing_loss_func -> aux_loss
loss += router_aux_loss_coef * aux_loss
```

源码依据是 `src/axolotl/integrations/liger/models/qwen3_moe.py:108-121`。这说明 MoE 不是简单地复用 llama forward；它要把 FLCE 插到主 loss 位置，同时保留 MoE 专属损失。

最后的 generic fallback 在 `src/axolotl/integrations/liger/plugin.py:275-286`：

```python
elif cfg.liger_fused_linear_cross_entropy:
    from .models.base import patch_lce_forward
    patch_lce_forward(cfg.model_config_type)
    LOG.warning_once("... generic FLCE support is experimental ...")
```

`patch_lce_forward` 通过 `transformers.models.{model_type}.modeling_{model_type}` 动态导入 `ForCausalLM` 类并替换 forward（`src/axolotl/integrations/liger/models/base.py:172-188`）。这条路径只适合“长得像 Llama”的 causal LM；如果模型 forward 需要额外返回值或特殊 loss，它可能语义不完整。

## 4.4 关键细节与误区澄清

> 容易误解一：README 的 Supported Models 就等于当前实际覆盖。

`src/axolotl/integrations/liger/README.md:26-45` 列了一组模型，但实际覆盖由 `plugin.py` 的分发逻辑、安装的 `liger-kernel` 版本和 Axolotl 自维护分支共同决定。比如 `gemma4` 在 Axolotl 分支里明确警告 FLCE 不兼容并跳过（`src/axolotl/integrations/liger/plugin.py:225-269`），即使它出现在某些相关文档或上游映射附近，也不能直接认为 FLCE 可用。

> 容易误解二：generic fallback 是稳定支持。

不是。Axolotl 自己在 `plugin.py:280-285` 对 generic FLCE 打了 warning：只应用 generic patch，并提示 experimental。它只动态找 `ForCausalLM` 类，不理解模型的特殊 forward 语义。

> 容易误解三：模型覆盖只取决于 Axolotl 仓库。

不完全。`MODEL_TYPE_TO_APPLY_LIGER_FN` 来自安装的 `liger_kernel.transformers.monkey_patch`（`plugin.py:47,84-85`），所以 Liger Kernel 版本升级会改变可用模型、函数签名和 patch 行为。Axolotl 的 `pyproject.toml:83` 当前 pin 了 `liger-kernel==0.7.0`。

## 4.5 本章小结

> 💡 **小结**
>
> * FLCE 覆盖不是一个静态列表，而是 upstream map + Axolotl 特殊分支 + generic fallback 的组合。
> * MoE / multimodal / 新模型常需要自定义 forward，不能简单套用 Llama 逻辑。
> * generic fallback 能扩大实验范围，但也最容易语义遗漏。
> * Liger Kernel 版本是实际模型覆盖的重要变量。

# 五、完整主路径串联：一次真实训练里 FLCE 什么时候执行、什么时候不执行

## 5.1 完整调用栈

下面把前面的机制串成一次用户调用：

```text
User: axolotl train examples/xxx.yml
  │
  ├─ Step 1: CLI 与配置读取
  │     └─ src/axolotl/cli/main.py:98 train
  │        src/axolotl/cli/train.py:55 do_cli
  │        src/axolotl/cli/config.py:230 load_cfg
  │
  ├─ Step 2: 插件注册与配置校验
  │     └─ src/axolotl/cli/config.py:208 prepare_plugins
  │        src/axolotl/integrations/base.py:370 PluginManager.register
  │        src/axolotl/utils/config/__init__.py:324 validate_config
  │        src/axolotl/integrations/config.py:27 merge_input_args
  │
  ├─ Step 3: 模型加载前 patch
  │     └─ src/axolotl/train.py:54 setup_model_and_tokenizer
  │        src/axolotl/loaders/model.py:162 ModelLoader.load
  │        src/axolotl/integrations/base.py:439 PluginManager.pre_model_load
  │        src/axolotl/integrations/liger/plugin.py:22 LigerPlugin.pre_model_load
  │
  ├─ Step 4: 模型实例化与 trainer 构建
  │     └─ src/axolotl/loaders/model.py:176 _build_model
  │        src/axolotl/train.py:522 setup_model_and_trainer
  │        src/axolotl/utils/trainer.py:708 setup_trainer
  │
  ├─ Step 5: 每个训练 step 的 forward
  │     └─ patched ModelForCausalLM.forward
  │        -> self.model(...)
  │        -> LigerForCausalLMLoss(hidden_states, lm_head.weight, labels)
  │        -> LigerFusedLinearCrossEntropyFunction.apply(...)
  │
  └─ Step 6: 保存 / resume
        └─ src/axolotl/train.py:254 save_trained_model
           正常保存权重；FLCE patch 不进入 state_dict
```

## 5.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 频率 |
|---|---|---|---|---|---|
| 配置读取 | YAML + CLI kwargs | `DictDefault` | 无 | 无 | 进程启动一次 |
| 插件注册 | `plugins` 字符串 | `PluginManager.plugins` 有序字典 | 无 | 无 | 配置加载一次 |
| 配置校验 | cfg + plugin args | 合并后的 cfg；互斥检查 | 无 | 无 | 配置加载一次 |
| `pre_model_load` | cfg.model_config_type + Liger flags | 改写 transformers 类 / 函数命名空间 | 无 | 无 | 模型加载前一次；reference model 也会触发 ModelLoader |
| `_build_model` | patched class namespace | 模型实例 | 取决于 FSDP/ZeRO/TP 加载配置，不是 FLCE 自己触发 | 参数加载显存 | 初始化一次 |
| patched forward | batch tensors | `loss`，训练时 `logits=None` | Liger FLCE kernel 自身无 collective；FSDP/ZeRO 可能有参数通信 | 避免完整 logits | 每 step |
| backward | loss 标量 | 预计算的 grad_input / grad_weight / grad_bias 被缩放返回 | Liger FLCE kernel 自身无 collective | 避免 backward 再展开 logits | 每 step |
| save | model/trainer | 常规 checkpoint / final model | FSDP/ZeRO 保存逻辑可能通信 | 与 FLCE 无专属保存显存 | 训练结束 / checkpoint |

## 5.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `liger_cross_entropy` | 名字里也有 Liger 和 CE | 只有 `liger_cross_entropy: true` 时 | 替换 CE，不融合 lm_head，不能替代 FLCE |
| `dpo_use_liger_kernel` | 也是 Liger loss 开关 | DPO trainer 路径，不是 SFT FLCE | `src/axolotl/core/trainers/dpo/__init__.py:41-42` 只是传给 DPO training args |
| `trl.use_liger_loss` | 文档说 GRPO 用 Liger loss | GRPO 路径，不是 causal LM forward patch | `src/axolotl/core/trainers/grpo/__init__.py:146-147` 映射到 `use_liger_kernel` |
| `chunked_cross_entropy` | 同样省 loss 显存 | Axolotl 内置 PatchManager 路径 | `src/axolotl/loaders/patch_manager.py:261-268`，与 FLCE 互斥 |
| CCE plugin | 同样融合 linear+CE | 另一套插件和外部 fork | `src/axolotl/integrations/cut_cross_entropy/__init__.py:86-103`，覆盖和通信语义不同 |
| `save_trained_model` | 保存时可能需要记录 FLCE | 没有 Liger 专属保存逻辑 | patch 是运行时代码替换，权重保存仍按普通模型走 |

## 5.4 本章小结

> 💡 **小结**
>
> * FLCE 的主路径非常短：配置校验 -> pre-model-load patch -> patched forward -> Liger fused loss。
> * patch 是初始化期动作；真正节省显存发生在每个 step 的 forward/loss 末端。
> * 保存和 resume 不保存 patch 本身，只保存权重；resume 依赖重新读取配置并重新执行插件。
> * DPO/GRPO 的 Liger loss 开关是不同 trainer 的 loss kernel，不是本文的 causal LM FLCE。

# 六、关键数据流、状态流与 shape 流程

## 6.1 Tensor shape 变化

以 causal LM SFT 为例：

```text
输入 batch:
  input_ids:      [B, T]
  labels:         [B, T]

Decoder 输出:
  hidden_states:  [B, T, H]

LigerForCausalLMLoss:
  labels pad+shift:
    labels        [B, T]
    shift_labels  [B, T]

  flatten:
    hidden_states [B*T, H]
    shift_labels  [B*T]

Liger FLCE chunk:
  weight:         [V, H]
  input_chunk:    [chunk, H]
  logits_chunk:   [chunk, V]   # 临时
  loss_1d_slice:  [chunk]
  grad_input:     [B*T, H]
  grad_weight:    [V, H]       # 如果 weight requires_grad

返回:
  loss:           []
  logits:         None         # 训练且 labels 存在
```

为什么这样变换？因为 Causal LM loss 本来就是 token 级别的 next-token CE。Liger 在 `loss_utils.py:80-90` 中完成 label shift、flatten 和 device 对齐，然后把 `[B*T,H]` 与 `[V,H]` 交给 fused kernel。

哪一步省显存？不是 `hidden_states.view(-1,H)`，而是不再让 `self.lm_head(hidden_states)` 产生完整 `[B,T,V]` 并保存到输出 / autograd。临时的 `logits_chunk [chunk,V]` 在 kernel loop 内消费掉。

哪一步可能成为瓶颈？`fused_linear_cross_entropy_forward` 仍然按 chunk 做 `_input_chunk @ weight.t()`（`fused_linear_cross_entropy.py:91-99`），而且 token scaling 会额外 `clone + softmax + gather`（同文件 `:105-133`）。因此它是“用分块和 fused backward 降峰值”，不是让最后一层计算消失。

## 6.2 Rank / Mesh / Process Group 变化

Liger FLCE 在 Axolotl 里没有自己创建 process group，也没有在 Liger op 中调用 `all_reduce` / `all_gather`。关键边界是：

```text
单卡 / DDP:
  每个 rank 在自己的 batch shard 上运行 FLCE
  无 Liger 自有 collective

FSDP:
  FSDP 是否 all-gather lm_head 参数由 FSDP wrapping 决定
  Axolotl generic path 对 FSDP-wrapped lm_head 做 forward redirection

Tensor Parallel:
  Axolotl 校验禁止 tensor_parallel_size > 1 与 liger_fused_linear_cross_entropy 同开
```

TP 禁止来自 `src/axolotl/integrations/liger/args.py:108-113`：

```python
if self.tensor_parallel_size > 1 and self.liger_fused_linear_cross_entropy:
    raise ValueError("Tensor parallelism is not compatible with liger losses.")
```

这和 CCE 形成鲜明对比：CCE 的 Axolotl fork 在 `apply_lce` 里识别 DTensor 的 vocab shard，构造 `VocabParallelOptions` 并在 loss 内做 vocab-parallel reduce（`/tmp/ml-cross-entropy-axolotl/cut_cross_entropy/transformers/utils.py:113-132`）。Liger FLCE 在 Axolotl 当前实现中选择了更保守的边界：不让 TP 与 Liger losses 同时出现。

## 6.3 状态切换

Liger FLCE 有两类全局状态改写：

```text
进入 pre_model_load:
  1. 改写 trl.trainer.ORPOTrainer
  2. 可选：用 torch.compiler.disable 包装 liger ops 函数
  3. 可选：改写 functional.liger_fused_linear_cross_entropy 和 FLCE Loss.__init__
  4. 改写 transformers.modeling_xxx.ForCausalLM.forward

执行中:
  trainer 调用 model(**batch)
  Python dispatch 落到 patched forward

退出:
  无自动恢复
```

这些状态定义在哪里？被改写的对象分布在 `trl.trainer`、`liger_kernel.transformers.functional`、`liger_kernel.transformers.fused_linear_cross_entropy.LigerFusedLinearCrossEntropyLoss`、`transformers.models.*.modeling_*`。写入者是 `LigerPlugin.pre_model_load` 和具体 `apply_liger_kernel_to_*` 函数；读取者是后续模型实例化和 trainer forward。

线程安全吗？源码中没有锁，也没有 context manager。更准确地说，它是 **进程级全局 monkey patch**。在典型 Axolotl 训练进程里这是可接受的；在同一 Python 进程里连续加载不同模型、或测试间不隔离模块状态时，就有污染风险。

## 6.4 关键细节与误区澄清

> 容易误解一：Liger FLCE 是一种通信优化。

不是。它主要是单 rank 上的 loss 显存优化。分布式通信来自外部并行策略（DDP/FSDP/ZeRO），不是 Liger FLCE kernel 自己创建。

> 容易误解二：FLCE 与 TP 只是“可能慢一点”。

在 Axolotl 配置层不是慢一点，而是直接报错：`tensor_parallel_size > 1` 且 `liger_fused_linear_cross_entropy` 为真会触发 `ValueError`（`liger/args.py:108-113`）。

> 容易误解三：`logits_to_keep` 在训练 FLCE 中照常节省显存。

对 qwen3 等自维护 path，训练且有 labels 时直接把完整 `hidden_states` 交给 `LigerForCausalLMLoss`（`qwen3.py:75-83`），没有走 `slice_indices`。generic `base.py` 会先算 `kept_hidden_states = hidden_states[:, slice_indices, :]`（`base.py:67-75`），但标准训练通常 `logits_to_keep=0` 表示全序列。不要把推理时的 last-token logits 优化误认为训练 loss 的核心收益。

## 6.5 本章小结

> 💡 **小结**
>
> * shape 主线是 `[B,T,H] -> [B*T,H]`，避免常驻 `[B*T,V]`。
> * Liger FLCE 没有 Axolotl 自建通信组；TP 与 Liger losses 被配置层禁止。
> * 状态切换是进程级 monkey patch，没有恢复动作。
> * token scaling、FSDP wrapping、generic slicing 会让不同模型路径存在细微差异。

# 七、核心机制深挖：Monkey Patch、FSDP 边界与 CCE 对照

## 7.1 Monkey Patch：零侵入接入，还是维护风险？

### 它解决什么问题？

Axolotl 不 fork transformers 模型源码，也不要求用户换一个模型类。通过 monkey patch，它可以在 `AutoModelForCausalLM.from_pretrained` 之前改写 transformers 模块中的 `ForCausalLM.forward`，从而让训练路径自然落入 FLCE。

### 为什么不能更简单？

如果只在 trainer 层重写 `compute_loss`，输入通常已经是模型返回的 `outputs`。而标准模型 forward 已经把 logits 算出来了，显存峰值已经发生。FLCE 必须发生在 `lm_head` 之前，所以必须进入模型 forward 内部。

### 源码实现

Liger upstream 路径示例：`liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_llama` 在开启 FLCE 时设置 `modeling_llama.LlamaForCausalLM.forward = llama_lce_forward`（`/usr/local/.../monkey_patch.py:217-264`）。

Axolotl 自维护 qwen3 路径示例：

```python
# src/axolotl/integrations/liger/models/qwen3.py:141-158
modeling_qwen3 = sys.modules["transformers.models.qwen3.modeling_qwen3"]
...
if fused_linear_cross_entropy:
    modeling_qwen3.Qwen3ForCausalLM.forward = lce_forward
```

Generic fallback 示例：

```python
# src/axolotl/integrations/liger/models/base.py:172-183
module_path = f"transformers.models.{model_type}.modeling_{model_type}"
model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
module = __import__(module_path, fromlist=[f"{model_cls_prefix}ForCausalLM"])
model_cls = getattr(module, f"{model_cls_prefix}ForCausalLM")
model_cls.forward = lce_forward
```

### 隐藏假设与副作用

隐藏假设包括：

- transformers 模块路径符合 `transformers.models.{model_type}.modeling_{model_type}`；
- CausalLM 类名能由 `get_causal_lm_model_cls_prefix` 推出；
- 替换后的 forward 与当前 transformers 版本返回结构兼容；
- 同一进程内没有另一个插件后续覆盖同一个 `forward`。

副作用是 patch 不可局部化。它不像 context manager 那样离开作用域后恢复；一旦赋值，后续同进程创建的同类模型都会使用 patched forward。

## 7.2 FSDP 边界：不是 FLCE 通信，而是 lm_head 权重访问时机

Generic path 有一段专门处理 FSDP 和 PEFT 的逻辑：

```python
# src/axolotl/integrations/liger/models/base.py:121-156
lm_head = self.lm_head
if PEFT_AVAILABLE and isinstance(lm_head, ModulesToSaveWrapper):
    lm_head = lm_head.modules_to_save.default

if isinstance(lm_head, FullyShardedDataParallel):
    return _FSDPForwardRedirection()(
        lm_head,
        _liger_for_causal_lm_loss,
        lm_head.module,
        hidden_states,
        hidden_size,
        labels,
        shift_labels,
        **loss_kwargs,
    )

return _liger_for_causal_lm_loss(lm_head=self.lm_head, ...)
```

这段不是在 Liger kernel 里做 collective，而是保证当 `lm_head` 本身被 FSDP 包起来时，读取 `lm_head.weight` 并执行 fused loss 的动作发生在 FSDP forward 语义内。注释 `base.py:133-135` 已经说明：如果 FSDP 下 `lm_head` 可训练，读取权重和调用 kernel 必须在 FSDP forward pass 中完成，这样完整参数会被 summon 并在 kernel 执行期间留在内存。

这也解释了一个显存边界：FLCE 可以省 logits，但不能省 `lm_head.weight` 的参数通信和参数 materialization。FSDP 是否需要 all-gather 参数由 FSDP 自己决定；FLCE 只是把 logits 这块峰值压下去。

## 7.3 CCE 对照：同样绕开 logits，但通信语义更激进

Axolotl 的 CCE 插件入口与 Liger 类似，也在 `pre_model_load` 里 patch：

```python
# src/axolotl/integrations/cut_cross_entropy/__init__.py:86-103
if cfg.cut_cross_entropy:
    self._check_requirements()
    self.patch_llama_like(cfg.model_config_type)
    from cut_cross_entropy.transformers.patch import cce_patch
    cce_patch(cfg.model_config_type, remote_model_id=cfg.base_model if cfg.trust_remote_code else None)
```

但 CCE 的外部 fork 覆盖模型更广。`src/axolotl/integrations/cut_cross_entropy/README.md:32-104` 列了大量模型，`/tmp/ml-cross-entropy-axolotl/cut_cross_entropy/transformers/patch.py:15-88` 也有完整 `PATCH_FNS` 映射。

更关键的是通信语义不同。CCE 的 `apply_lce` 会识别 DTensor：

```python
# /tmp/ml-cross-entropy-axolotl/cut_cross_entropy/transformers/utils.py:113-132
if isinstance(c, DTensor):
    device_mesh = c.device_mesh
    process_group = device_mesh.get_group("tp")
    placement = c.placements[vocab_dim]
    if isinstance(placement, Shard):
        vocab_size = c.size(vocab_dim)
        vocab_parallel_options = VocabParallelOptions.from_vocab(
            vocab_size, process_group, reduce_e_grad=True
        )
        cce_kwargs["vocab_parallel_options"] = vocab_parallel_options
    c_local = c.to_local()
```

CCE core loss 里，vocab parallel 会做 all-reduce：

```python
# /tmp/ml-cross-entropy-axolotl/cut_cross_entropy/vocab_parallel/utils.py:47-65
vp_reduce_lse:
  all_reduce(MAX) for lse_max
  all_reduce(SUM) for exp-sum
vp_reduce_correct_logit:
  all_reduce(SUM) for correct logit
```

backward 还可以 all-reduce embedding gradient（`utils.py:68-97`），并且 ZeRO-3 下用 `GatheredParameters` 重新 gather full weight（`/tmp/ml-cross-entropy-axolotl/cut_cross_entropy/cce.py:179-196`）。

这就是二者的工程分水岭：

```text
Liger FLCE in Axolotl:
  - TP: 配置层禁止
  - kernel: 不引入自有 collective
  - FSDP: generic path 处理 lm_head forward redirection

CCE in Axolotl fork:
  - DTensor vocab shard: 支持 vocab-parallel loss
  - forward: all_reduce lse / correct_logit
  - backward: 可 all_reduce e_grad
  - ZeRO-3: backward gather 参数
```

## 7.4 关键细节与误区澄清

> 容易误解一：Liger FLCE 和 CCE 只是两个不同厂牌的同类 kernel。

从 Axolotl 接入看不是。Liger 当前在配置层禁止 TP + Liger loss；CCE fork 明确处理 DTensor vocab parallel 和 ZeRO-3 参数 gather。两者都省 logits，但分布式语义差异很大。

> 容易误解二：FSDP 下 FLCE 不需要考虑 lm_head 参数。

需要。generic path 专门判断 `lm_head` 是否是 `FullyShardedDataParallel`，并通过 `_FSDPForwardRedirection` 调用 loss（`base.py:136-146`）。这说明权重访问时机是一个真实问题。

> 容易误解三：CCE 总是比 Liger 更复杂，所以一定更慢。

源码只能说明 CCE 有额外分布式语义和更多覆盖路径，不能直接推出快慢。实际性能取决于 vocab size、BT、TP/ZeRO 配置、kernel autotune、通信拓扑和 dtype。本文不基于源码编造 benchmark 结论。

## 7.5 本章小结

> 💡 **小结**
>
> * Monkey patch 是 FLCE 接入模型 forward 的必要手段，也是维护风险来源。
> * FSDP 边界主要体现在 lm_head 权重访问时机，不是 FLCE 自己创建通信。
> * CCE 与 Liger 都避免完整 logits，但 CCE fork 的 DTensor/ZeRO-3 支持让它的通信语义明显不同。
> * 对比二者时不能只看“省不省 logits”，还要看 patch 点、模型覆盖和并行约束。

# 八、显存、性能与通信分析

## 8.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | FLCE 不改变 `lm_head.weight [V,H]` 或模型参数数量 |
| optimizer state | ❌ | optimizer 仍按可训练参数维护状态 |
| attention 激活 | ❌ | FLCE 位于 decoder 之后，不改变 attention 内部激活 |
| hidden_states | ❌ / 轻微 | `[B,T,H]` 仍然需要，generic path 可能 slice，但主收益不在这里 |
| logits | ✅ | 避免常驻完整 `[B*T,V]` logits；只保留 chunk 级临时 logits |
| loss 中间 buffer | ✅ / 有代价 | 保存 `grad_input/grad_weight/grad_bias`，不保存完整 logits；但 `grad_weight [V,H]` 在 trainable lm_head 时仍可能很大 |
| 输入 batch | ❌ | dataloader / collator 不受 FLCE 影响 |
| save/load CPU 内存 | ❌ | 保存仍走 Axolotl 常规模型保存逻辑 |

真正显存大头是 `BT * V`。Liger op 注释在 `fused_linear_cross_entropy.py:41-48` 直接指出 materialized activations 会是 `BT x V`。当 `B*T=8192`、`V=128000`、bf16 时，单个 logits 理论大小约 2GB；如果 backward 还要保存或产生梯度，中间峰值会更高。FLCE 的收益集中在这部分。

但收益不是免费的：

- 每个 chunk 仍计算 `logits_chunk [chunk,V]`；
- 如果 `weight.requires_grad`，`grad_weight` 仍是 `[V,H]`，源码在 `fused_linear_cross_entropy.py:56-65` 分配；
- `liger_use_token_scaling` 会额外 `detach().clone()` logits 并 softmax（`fused_linear_cross_entropy.py:105-133`），增加 chunk 内临时内存和计算。

## 8.2 通信开销

Liger FLCE 自身没有在 Axolotl 或 Liger op 中调用 collective。按场景拆开看：

| 场景 | 每 step 通信 | 来源 | 是否 FLCE 新增 |
|---|---|---|---|
| 单卡 | 无 | 无 | 否 |
| DDP | 梯度 all-reduce | PyTorch DDP / Accelerate | 否 |
| FSDP | 参数 all-gather / reduce-scatter 等 | FSDP wrapping | 否；generic path 只保证 lm_head 在 FSDP forward 语义内使用 |
| DeepSpeed ZeRO | 参数 gather / gradient partition | DeepSpeed | 否；Liger FLCE 无 ZeRO-3 专属 gather 代码 |
| Tensor Parallel | Axolotl 禁止与 Liger FLCE 同开 | `liger/args.py:108-113` | 不适用 |
| CCE + DTensor TP | lse/correct logit/e_grad all-reduce | CCE fork vocab parallel | 是 CCE 路径新增，不是 Liger |

如果和 CCE 对比，CCE 的 `vp_reduce_lse` 至少包含两次 all-reduce（max 和 sum），`vp_reduce_correct_logit` 再一次 all-reduce（`/tmp/ml-cross-entropy-axolotl/cut_cross_entropy/vocab_parallel/utils.py:47-65`）；backward 若 `reduce_e_grad=True` 还会 all-reduce e_grad（同文件 `:68-97`）。这让 CCE 能服务 vocab-sharded lm_head，但也把 loss 阶段变成通信参与者。

## 8.3 性能取舍

Liger FLCE 的取舍可以概括为：

```text
用更复杂的 forward patch + fused autograd Function
换取 loss 末端 logits 峰值下降；
不引入新的 loss collective，
但也放弃当前 Axolotl TP 组合。
```

它不属于“通信换显存”，更像是“kernel 分块 + autograd 语义重排换显存”。forward 里预计算并保存梯度，backward 只缩放已有梯度，这降低了 backward 再展开 logits 的压力；但如果外部 `grad_output` 不是 1，backward 仍要用 Triton element-wise kernel 缩放 `grad_input/grad_weight/grad_bias`（`fused_linear_cross_entropy.py:232-276`）。

性能瓶颈主要可能出现在：

- vocab 很大时，每个 chunk 的 matmul 仍昂贵；
- `target_mask.sum().item()` 在 forward 中会触发同步风险，Liger 源码注释也写了 TODO：评估 `.item()` 导致的 CUDA synchronization 对速度的影响（`fused_linear_cross_entropy.py:74-76`）；
- `liger_use_token_scaling` 额外 softmax 让 chunk 内工作更重；
- FSDP 下如果 `lm_head` 被 shard，参数 materialization / communication 仍可能成为 step 尾部瓶颈。

## 8.4 本章小结

> 💡 **小结**
>
> * Liger FLCE 真正节省的是 loss 阶段完整 logits，不节省参数、optimizer state 或 attention 激活。
> * 它不是通信优化；loss kernel 自身没有 collective，TP 组合在配置层被禁止。
> * CCE 的 vocab-parallel 路径会把 loss 纳入通信图，这是两者的重要差异。
> * `use_token_scaling`、FSDP lm_head、`.item()` 同步都是潜在性能边界。

# 九、配置项、边界条件与坑点

## 9.1 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `plugins: [axolotl.integrations.liger.LigerPlugin]` | `cli/config.py:306` -> `prepare_plugins` | 注册 Liger 插件，插件参数并入 schema | 不写插件则 Liger hook 不执行 |
| `liger_fused_linear_cross_entropy: true` | `liger/plugin.py:84-290` | pre-model-load 替换模型 `forward` | 与 CCE / chunked CE / Liger CE 互斥 |
| `liger_cross_entropy: true` | `liger/plugin.py` 各分支 | 替换 CE 函数 / 类，不融合 lm_head | 不能和 FLCE 同开；仍可能物化 logits |
| `liger_use_token_scaling: true` | `liger/args.py:96-106`; `liger/plugin.py:57-82` | 强制 Liger FLCE 使用 token scaling | 需要 FLCE；会额外 softmax / clone |
| `tensor_parallel_size > 1` | `liger/args.py:108-113` | 与 Liger FLCE 同开时报错 | Liger loss 当前不支持 Axolotl TP |
| `liger_rms_norm: true` + TP | `liger/args.py:86-94` | 报错 | RMSNorm 与 TP 不兼容，源码引用 Liger issue #826 |
| `liger_swiglu` | `liger/args.py:57-71` | deprecated，迁移到 `liger_glu_activation` | 与 `liger_glu_activation` 同设会报错 |
| `tiled_mlp: true` + `liger_glu_activation: true` | `liger/args.py:73-84` | 除非 `tiled_mlp_use_original_mlp: true`，否则报错 | 两套 MLP patch 冲突 |
| `torch_compile: true` | `liger/plugin.py:29-42` | 对 Liger FLCE forward/backward kernel 加 `torch.compiler.disable` | 避免 compile 误优化 Triton kernel，但又是全局函数替换 |
| `cut_cross_entropy: true` | `validation.py:974-1002` | 与 Liger FLCE 冲突 | 不能把 CCE 与 FLCE 叠加比较收益 |
| `dpo_use_liger_kernel` | `core/trainers/dpo/__init__.py:41-42` | 传给 DPO trainer args | 不会触发 causal LM FLCE forward patch |
| `trl.use_liger_loss` | `core/trainers/grpo/__init__.py:146-147` | 传给 GRPO `use_liger_kernel` | 与本文 SFT FLCE 是不同路径 |

## 9.2 默认行为和静默失效

Liger args 的默认值基本是 `None`，因此只加载插件但不设置 `liger_fused_linear_cross_entropy`，不会自动启用 FLCE。对比 CCE，`CutCrossEntropyArgs.cut_cross_entropy` 默认是 `True`（`src/axolotl/integrations/cut_cross_entropy/args.py:33`）：只要用户把 CCE plugin 放进 `plugins`，默认就会开启 CCE。

Liger unsupported 模型不会总是 hard fail。`LigerPlugin.pre_model_load` 在不支持且没走 fallback 时只 warning：`Unsupported model config type ... Liger not applied`（`src/axolotl/integrations/liger/plugin.py:286-292`）。这意味着如果用户没有仔细看日志，可能以为配置生效但实际上没有 patch 到模型。

## 9.3 保存、加载与 resume

FLCE 没有专属 state_dict。训练完成后 `save_trained_model` 按 FSDP、DeepSpeed 或 rank0 普通保存逻辑处理（`src/axolotl/train.py:254-386`）。这带来两个结论：

- 保存出来的是普通权重，不会把 “forward 被 patch” 这个事实写入 checkpoint；
- resume 或重新训练时，必须再次通过配置加载 Liger plugin，重新执行 `pre_model_load`。

这不是 bug，而是 monkey patch 接入的自然结果。

## 9.4 本章小结

> 💡 **小结**
>
> * 最小开启条件是注册 `LigerPlugin` 并设置 `liger_fused_linear_cross_entropy: true`。
> * Liger FLCE 与 TP、CCE、chunked CE、Liger CE 都有明确边界。
> * CCE 插件默认启用，Liger 插件默认不自动启用 FLCE，这是配置语义差异。
> * FLCE 不进入 checkpoint；resume 依赖重新执行插件初始化。

# 十、测试、示例与覆盖缺口

## 10.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/integrations/test_liger.py:44-61` | `liger_swiglu` deprecated -> `liger_glu_activation` | 覆盖配置迁移，不覆盖 kernel |
| `tests/integrations/test_liger.py:63-77` | `liger_swiglu` 与 `liger_glu_activation` 冲突 | 覆盖 schema 互斥 |
| `tests/integrations/test_liger.py:79-93` | `liger_use_token_scaling` 必须依赖 FLCE | 覆盖 token scaling 配置约束 |
| `tests/e2e/patched/test_cli_integrations.py:18-47` | Liger plugin args 能从 YAML 进入 cfg | 覆盖插件参数合并 |
| `tests/e2e/integrations/test_liger.py:20-63` | Llama/SmolLM2 + Liger CE 训练 smoke | 覆盖非 FLCE Liger CE 路径 |
| `tests/e2e/integrations/test_liger.py:65-113` | Llama/SmolLM2 + FLCE + token scaling True/False | 覆盖基础端到端训练和输出保存 |
| `tests/e2e/integrations/test_cut_cross_entropy.py:53-66` | Llama + CCE 训练 smoke | 用于对比 CCE 主路径 |
| `tests/e2e/integrations/test_cut_cross_entropy.py:68-110` | Qwen2 + CCE 训练 smoke | 覆盖 CCE 非 Llama 模型 |
| `tests/e2e/integrations/test_cut_cross_entropy.py:112-138` | CCE + attention 组合 | 覆盖 CCE 与 flash/sdp attention smoke |

Liger e2e 使用 `require_torch_2_4_1`（`tests/e2e/utils.py:45-54`），所以低版本 torch 会 skip。测试里的 FLCE 配置使用 `HuggingFaceTB/SmolLM2-135M`、`max_steps=5`、`bf16="auto"`，目标更像 smoke test：证明训练能跑通并保存输出，而不是证明显存收益或所有模型 patch 正确。

## 10.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| qwen3/qwen3_5/qwen3_moe 自维护 FLCE forward | 未看到专门 e2e | transformers 版本变化时 forward 签名或返回结构不匹配 |
| generic fallback `models/base.py` | 未看到专门 e2e | 实验性模型 silently patch 后语义不完整 |
| FSDP-wrapped lm_head + `_FSDPForwardRedirection` | 未看到 Liger FLCE 专项多卡测试 | lm_head 权重 summon 时机、显存峰值或梯度异常难以及早发现 |
| TP + Liger FLCE 禁止 | schema 有代码，但未看到专门测试 | 未来改动可能误放开不支持组合 |
| `torch_compile` + FLCE shim | 未看到专项测试 | compile disable wrapper 失效时可能触发 kernel compile 问题 |
| `liger_use_token_scaling` 数值正确性 | 只看到 e2e smoke | scaling 语义、梯度缩放与预期不一致时难发现 |
| 输出 `logits=None` 对 callbacks/metrics 的影响 | 未看到专项测试 | 下游自定义 trainer 或 callback 访问 logits 报错 |
| 保存 / resume 后重新 patch | 未看到 Liger FLCE resume 测试 | resume 配置遗漏插件时行为退回普通 loss |
| 多机训练 | 未看到专项覆盖 | FSDP/DeepSpeed/launcher 组合问题可能只在集群出现 |
| 显存/性能收益 | 未看到 benchmark test | 无法从 CI 证明 60%/20% 类收益 |

## 10.3 关键细节与误区澄清

> 容易误解一：有 e2e 就说明所有模型的 FLCE 都稳。

当前 Liger e2e 只覆盖一个小模型路径，不能外推到 qwen3.5、MoE、Gemma4、DeepSeekV2 或 generic fallback。模型覆盖的风险仍主要靠源码维护和实际训练反馈。

> 容易误解二：测试证明了显存收益。

没有。现有 e2e 主要证明能训练和保存，没有读取 `torch.cuda.max_memory_allocated()` 对比普通 CE。显存收益来自 kernel 语义推断和 Liger 文档声明，不是 Axolotl 测试直接断言。

> 容易误解三：CCE 和 Liger 的测试覆盖等价。

不等价。CCE e2e 覆盖 Llama、Qwen2 和 attention 组合；Liger FLCE e2e 覆盖 Llama/SmolLM2 + token scaling。CCE 的 DTensor vocab parallel、ZeRO-3 gather 等路径也未被这些测试证明。

## 10.4 本章小结

> 💡 **小结**
>
> * 现有测试较好覆盖了配置合并和基础 smoke，但没有系统覆盖所有模型 patch。
> * Liger FLCE 的高风险点集中在模型 forward 兼容、FSDP lm_head、generic fallback 和 `logits=None` 下游假设。
> * 现有测试不证明显存收益，只证明部分路径能跑通。
> * CCE 的更复杂通信语义同样缺少专项多卡验证。

# 十一、局限性与已知优化点

## 11.1 硬约束

1. **互斥约束**：`cut_cross_entropy`、`chunked_cross_entropy`、`liger_cross_entropy`、`liger_fused_linear_cross_entropy` 只能开一个（`validation.py:974-1002`）。
2. **TP 约束**：`tensor_parallel_size > 1` 与 `liger_fused_linear_cross_entropy` 同开会报错（`liger/args.py:108-113`）。
3. **Liger RMSNorm + TP 约束**：`liger_rms_norm` 与 TP 不兼容（`liger/args.py:86-94`）。
4. **MLP patch 冲突**：`liger_glu_activation` 与 `tiled_mlp` 冲突，除非 `tiled_mlp_use_original_mlp: true`（`liger/args.py:73-84`）。
5. **模型支持约束**：unsupported model 可能只 warning 不报错（`liger/plugin.py:286-292`），generic fallback 是 experimental。
6. **训练输出约束**：FLCE 主路径下 `logits=None`，不适合依赖 logits 的自定义逻辑。

## 11.2 维护成本

- **依赖上游 transformers forward 签名**：qwen3 文件注释写明基于 transformers v4.51.3（`src/axolotl/integrations/liger/models/qwen3.py:1-3`），qwen3.5 基于 v5.3.0（`qwen3_5.py:1-3`）。上游 forward 一改，本地 patch 就可能漂移。
- **进程级 monkey patch 难隔离**：没有恢复逻辑，对测试顺序和同进程多模型不友好。
- **版本耦合**：`liger-kernel==0.7.0` pin 在 `pyproject.toml:83`，插件还为该版本写了 ORPOTrainer shim（`liger/plugin.py:22-27`）。
- **特殊模型分支膨胀**：MoE、multimodal、Gemma4、DeepSeekV2、Jamba 都需要不同 patch 点，维护复杂度高于普通 loss 函数替换。

## 11.3 性能瓶颈

- **chunk matmul 仍存在**：`logits_chunk = _input_chunk @ weight.t()` 是每 chunk 的主要计算（Liger op `:96-99`）。
- **同步风险**：Liger op 对 `target_mask.sum().item()` 有 TODO，提示 CUDA synchronization 可能影响速度（`fused_linear_cross_entropy.py:74-76`）。
- **token scaling 额外成本**：`detach().clone()`、softmax 和 gather 在 `fused_linear_cross_entropy.py:105-133`。
- **FSDP lm_head materialization**：generic path 需要在 FSDP forward 内访问完整 lm_head 权重，参数通信/峰值仍存在（`base.py:133-146`）。
- **无法使用 Axolotl TP 分担 vocab**：当前直接禁止 TP + Liger losses，因此大 vocab 场景不能靠 TP 分摊 `lm_head.weight` 的 vocab 维。

## 11.4 已知优化点

源码中明确出现的 TODO / 优化线索包括：

- `liger/args.py:110` 写有 TODO：`tensor_parallel_size > 1 and liger_fused_linear_cross_entropy` 是 “larger fix - investigate”。这说明 TP 不兼容是当前工程限制，不一定是理论不可行。
- Liger op `fused_linear_cross_entropy.py:74` TODO 提到评估 `.item()` 同步对速度影响。
- CCE fork 已经有 DTensor vocab parallel 和 ZeRO-3 gather 处理，可以作为未来 Liger FLCE 与 TP 结合时的参考方向，但不能直接照搬，因为 Liger forward/backward 预存梯度的语义不同。
- generic fallback 可以增加更强的 runtime verification：patch 后检查 forward 签名、返回类型、是否保留 aux loss，避免只 warning 后进入不完整训练。

## 11.5 本章小结

> 💡 **小结**
>
> * 当前最大硬约束是 TP 不兼容和模型 forward patch 覆盖有限。
> * 维护成本来自 transformers 版本漂移、进程级 monkey patch 和特殊模型分支。
> * 性能瓶颈并非消失，而是从完整 logits 峰值转为 chunk matmul、同步点和 FSDP 权重 materialization。
> * 未来优化值得关注 TP/vocab-parallel FLCE、patch 可恢复性、generic fallback 验证和 `.item()` 同步消除。

# 小结与展望

Axolotl 的 Liger fused linear cross entropy 实现可以用几个关键词概括。

## 关键词一：模型加载前 patch

FLCE 不是 trainer loss 的小替换，而是模型 `forward` 的重写。Axolotl 通过插件系统在 `ModelLoader._build_model()` 前执行 `LigerPlugin.pre_model_load()`，确保后续模型实例化时已经拿到 patched class。这个设计零侵入、配置简单，但代价是进程级 monkey patch 和上游 forward 版本耦合。

## 关键词二：不物化完整 logits

真正的显存收益来自训练路径中 `logits=None`：`hidden_states [B,T,H]` 和 `lm_head.weight [V,H]` 直接进入 Liger fused loss，避免完整 `[B*T,V]` logits 常驻。kernel 内仍有 chunk 级 logits，因此它不是免计算，而是降低峰值并改变 backward 保存语义。

## 关键词三：覆盖分层

Axolotl 不是只依赖 Liger upstream。它先查 `MODEL_TYPE_TO_APPLY_LIGER_FN`，再处理本地维护的 qwen3/qwen3.5/MoE/Jamba/DeepSeekV2 等分支，最后才尝试 generic fallback。这个策略扩大了模型覆盖，也带来了维护风险：每个模型 forward 的细节都可能影响 loss、router aux、multimodal 输出和 logits 语义。

## 关键词四：非通信型 loss 显存优化

Liger FLCE 在 Axolotl 当前实现中不是“通信换显存”。它没有自建 process group，也不在 kernel 里做 all-reduce；TP 与 Liger loss 直接互斥。与之相比，CCE fork 已经在 DTensor vocab parallel 下引入 lse / correct-logit / e_grad all-reduce，并为 ZeRO-3 backward gather 参数。这也是本文最重要的对照结论：**Liger FLCE 和 CCE 都瞄准 loss 显存，但前者在 Axolotl 里更像 forward patch + fused autograd，后者更像模型 patch + vocab/ZeRO-aware linear CE 框架。**

## 适合什么场景

Liger FLCE 适合大 vocab、长序列、SFT/causal LM 训练中 loss logits 成为显存峰值的场景，尤其是单卡、DDP、常规 FSDP/DeepSpeed 训练，希望用较少配置获得 loss 端显存下降。

## 不适合什么场景

它不适合当前依赖 Tensor Parallel 分摊 vocab 的训练；不适合下游强依赖 `outputs.logits` 的自定义 trainer / callback；也不适合模型 forward 结构与 Llama-like 假设差异很大、但只走 generic fallback 的生产训练。

## 与替代方案相比的取舍

- 相比 `liger_cross_entropy`：FLCE 更早绕开 logits，显存收益边界更靠前，但 patch 更重；
- 相比 `chunked_cross_entropy`：FLCE 融合 lm_head 与 CE，语义更深入，但模型覆盖更敏感；
- 相比 CCE：Liger 当前在 Axolotl 中分布式语义更保守，不支持 TP；CCE 覆盖更广且处理 vocab parallel/ZeRO-3，但引入了更多 loss 阶段通信和外部 fork 依赖。

## 后续值得继续走读的方向

下一步可以继续走读三条线：

1. CCE fork 的 vocab-parallel loss 如何和 Axolotl ND parallelism / TP device mesh 交互；
2. Liger GRPO / DPO loss 与本文 causal LM FLCE 的差异，尤其是为什么 GRPO 文档推荐 token-level IS；
3. FSDP2、QLoRA、trainable `lm_head` 与 FLCE 的组合，在真实多卡下参数 all-gather、梯度 reduce 和显存峰值如何叠加。

如果用一句话收束：Axolotl 的 Liger FLCE 是一个典型的深度学习系统工程折中——它用模型加载前 monkey patch 和 fused autograd，把最刺眼的 logits 显存峰值压下去；但它没有免费午餐，模型覆盖、TP 兼容、`logits=None` 语义和上游版本漂移，都是使用者必须读懂源码后再接受的代价。
