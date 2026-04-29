# Axolotl 源码走读：Cut Cross Entropy 实现解析

在大词表语言模型训练里，最后一层 loss 往往不像注意力那样显眼，却会在长序列、sample packing、MoE / VLM 大词表以及 QLoRA 场景中突然变成显存峰值：模型已经算出了 `[batch, seq, hidden]`，但标准做法还要把它投影成 `[batch, seq, vocab]` 的 logits，再把 logits 喂给 cross entropy。对于 32K、128K 甚至更大词表，这个临时张量可能比很多人直觉中的“loss 很轻量”大得多。

Axolotl 的 Cut Cross Entropy（下文简称 CCE）就是围绕这个矛盾接入的：不重写 Trainer、不改变数据集、不改变保存格式，而是在模型加载前把 Hugging Face 模型的 `forward()` 换成一个“直接从 hidden states + lm_head.weight 计算 loss”的版本。本文不展开 Apple Cut Cross Entropy 论文的数学推导，而是按源码主路径读 Axolotl 如何把这个能力接进训练链路、哪些地方真的省显存、哪些地方只是 patch 与兼容成本。

# 前言

## 业务 / 工程背景

CCE 出现在 **训练阶段的 loss 计算**，主要解决显存问题。它不是参数分片方案，不负责数据并行调度，也不改变 checkpoint 格式；它瞄准的是 causal LM 训练中 `hidden_states -> lm_head -> logits -> cross_entropy` 这段路径。Axolotl 文档也把它放在优化项里：`docs/optimizations.qmd:64-68` 只说它通过优化 cross entropy loss 降低 VRAM；集成 README 进一步说明它是“during loss calculation”的 VRAM 优化（`src/axolotl/integrations/cut_cross_entropy/README.md:1-5`）。

## 核心矛盾

核心矛盾可以压缩成三句话：

1. Trainer 期望模型前向返回 `loss`，Transformers 默认 loss 需要先物化完整 logits。
2. 完整 logits 的大小是 `batch × seq × vocab × dtype_bytes`，而训练真正需要的只是每个 token 的 log-sum-exp 和正确类别 logit。
3. Axolotl 又不能为每个模型重写 Trainer，因此选择在 **模型加载前 monkey patch 模型类的 forward**，把 loss 路径切到 CCE 下游库。

## 本文主线

本文按机制而不是按文件展开：

1. 先看用户如何开启 CCE，以及配置如何变成真实行为；
2. 再看 patch 为什么必须发生在模型加载前；
3. 然后进入一次训练 step，读 patched `forward()` 如何避开 logits；
4. 接着分析 shape、状态、通信、ZeRO-3 / TP 兼容；
5. 最后看显存收益、配置坑点、测试覆盖和维护风险。

## 不展开的内容

本文不讲 CCE 论文的完整数学推导，不讲 FSDP / DeepSpeed / Tensor Parallel 的基础原理，也不讲 Liger、FlashAttention 的算法细节。需要时只讨论它们和 Axolotl CCE 主路径的交界面。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/integrations/cut_cross_entropy/__init__.py` | Axolotl CCE 插件入口；检查依赖；在模型加载前调用下游 `cce_patch()` |
| `src/axolotl/integrations/cut_cross_entropy/args.py` | 插件配置 schema；默认启用；校验 fp16/bf16 与 chunked CE 冲突 |
| `src/axolotl/cli/config.py` | CLI 配置加载；先注册插件，再动态合并插件 schema，再校验配置 |
| `src/axolotl/loaders/model.py` | 模型加载主路径；在 `_build_model()` 前触发 `PLUGIN_MANAGER.pre_model_load()` |
| `src/axolotl/core/trainers/base.py` | Axolotl Trainer 的 `compute_loss()`；常规路径仍交给 Transformers/模型输出 loss |
| `src/axolotl/loaders/patch_manager.py` | 其他 loss/attention patch 的位置，用于区分 CCE 和 chunked/flash-attn CE |
| `axolotl-ai-cloud/ml-cross-entropy@fec1a88:cut_cross_entropy/transformers/patch.py` | 下游 CCE patch 分发器；维护模型类型到 patch 函数的映射 |
| `axolotl-ai-cloud/ml-cross-entropy@fec1a88:cut_cross_entropy/transformers/llama.py` | 典型 patched `forward()`；从 hidden states 直接调用 `apply_lce()` |
| `axolotl-ai-cloud/ml-cross-entropy@fec1a88:cut_cross_entropy/cce.py` | 自定义 autograd Function；Triton 前向/反向核心和 VP / ZeRO-3 兼容 |

> 说明：Axolotl 的仓库里只有集成层；真正的 CCE kernel 在安装脚本固定的 fork `axolotl-ai-cloud/ml-cross-entropy@fec1a88` 中。本文的 kernel 行号来自该 commit 的本地 checkout。

# 一、入口与配置归一化：把“插件名”变成一个 loss 行为开关

## 1.1 设计哲学与核心问题

CCE 在 Axolotl 里不是一个内置顶层配置项，而是一个 integration plugin。这个设计解决了一个很现实的问题：CCE 依赖外部包 `cut_cross_entropy[transformers]`，还依赖模型类型的 patch 适配。如果把它做成基础 schema 的固定字段，用户即使不用 CCE 也会被外部依赖和模型兼容性污染。

因此主路径是：用户在 YAML 中写 `plugins`，Axolotl 先注册插件，让插件把自己的 Pydantic 参数模型并入全局配置 schema，然后再校验 `cut_cross_entropy`、`bf16/fp16`、互斥 loss 优化项。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：用户侧 `axolotl train config.yml` 入口，默认 launcher 为 accelerate

src/axolotl/cli/utils/train.py
  - _launch_accelerate_training：转成 `accelerate launch -m axolotl.cli.train config.yml`

src/axolotl/cli/train.py
  - do_cli：调用 load_cfg，再进入 do_train
  - do_train：加载数据集，然后调用 axolotl.train.train

src/axolotl/cli/config.py
  - prepare_plugins：按 cfg.plugins 注册插件
  - load_cfg：读取 YAML、注册插件、校验配置、normalize

src/axolotl/integrations/config.py
  - merge_input_args：把插件提供的 Pydantic args 动态拼进 AxolotlInputConfig

src/axolotl/integrations/cut_cross_entropy/args.py
  - CutCrossEntropyArgs：定义 `cut_cross_entropy` 默认值与校验
```

## 1.3 主流程拆解

用户最常见的开启方式很短：

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
# cut_cross_entropy: true  # 可显式写；插件 schema 默认 True
bf16: auto
```

示例里大量配置只写插件名，例如 Qwen3.5 27B QLoRA 示例在 `examples/qwen3.5/27b-qlora.yaml:6-8` 写入 CCE 插件，`bf16: auto` 在 `examples/qwen3.5/27b-qlora.yaml:56-57`。Gemma3n 示例则显式写了 `cut_cross_entropy: true`（`examples/gemma3n/gemma-3n-e2b-qlora.yml:6-9`）。

从 CLI 到训练进程的入口是：

```text
axolotl train examples/qwen3.5/27b-qlora.yaml
  -> src/axolotl/cli/main.py:98-128 train(...)
    -> launch_training(...)
      -> src/axolotl/cli/utils/train.py:157-185
         accelerate launch -m axolotl.cli.train <config>
        -> src/axolotl/cli/train.py:55-91 do_cli(...)
          -> load_cfg(config)
```

`load_cfg()` 的关键顺序是：

```text
load_cfg
  -> 读取 YAML 为 DictDefault                    # src/axolotl/cli/config.py:244-253
  -> prepare_plugins(cfg)                       # src/axolotl/cli/config.py:306
      -> PluginManager.register(plugin_name)    # src/axolotl/cli/config.py:215-220
  -> validate_config(cfg)                       # src/axolotl/cli/config.py:308-320
      -> merge_input_args() if cfg.plugins       # src/axolotl/utils/config/__init__.py:324-337
          -> plugin.get_input_args()
```

`CutCrossEntropyPlugin.get_input_args()` 返回字符串形式的 args 类路径（`src/axolotl/integrations/cut_cross_entropy/__init__.py:47-48`）。`merge_input_args()` 再用动态 `exec()` 构造新类：

```python
# src/axolotl/integrations/config.py:40-57（简化）
input_args = plugin_manager.get_input_args()
for plugin_args in input_args:
    dynamic_input += f"from {plugin_module} import {plugin_cls}\n"
...
class AxolotlInputConfig(AxolotlInputConfigBase, CutCrossEntropyArgs):
    pass
```

这意味着 `cut_cross_entropy` 并不是基础 `AxolotlInputConfigBase` 字段；它只有在插件注册后才进入正式 schema。

`CutCrossEntropyArgs` 本身很小，但决定了几个重要行为：

```python
# src/axolotl/integrations/cut_cross_entropy/args.py:28-54（简化）
class CutCrossEntropyArgs(BaseModel):
    cut_cross_entropy: Optional[bool] = True

    @model_validator(mode="before")
    def check_dtype_is_half(cls, data):
        if data.get("cut_cross_entropy") and not (data.get("bf16") or data.get("fp16")):
            raise ValueError(...)

    @model_validator(mode="before")
    def check_chunked_cross_entropy_not_set(cls, data):
        if data.get("chunked_cross_entropy"):
            raise ValueError(...)
```

这里有两个输入输出变化：

- 输入：YAML 里的 `plugins`、`cut_cross_entropy`、`bf16/fp16`、`chunked_cross_entropy`；
- 输出：经过 Pydantic 校验后的 `cfg.cut_cross_entropy`，供后续 `pre_model_load()` 判断。

另一个全局校验在 `ValidationMixin.check_cross_entropy_conflicts()`：它把 `cut_cross_entropy`、`chunked_cross_entropy`、`liger_cross_entropy`、`liger_fused_linear_cross_entropy` 放在同一组，只允许开一个（`src/axolotl/utils/schemas/validation.py:974-1002`）。所以 CCE 在 Axolotl 里被定义为 **loss 路径的互斥优化**，而不是可以和其他 CE patch 任意叠加的选项。

## 1.4 关键细节与误区澄清

> 容易误解点 1：`cut_cross_entropy` 看起来像 Axolotl 核心 schema 字段，但它其实来自插件 schema。

基础 schema 里只有 `plugins`（`src/axolotl/utils/schemas/config.py:1275-1280`）和其他 loss 字段如 `chunked_cross_entropy`（`src/axolotl/utils/schemas/config.py:867-877`）。`cut_cross_entropy` 字段由 `CutCrossEntropyArgs` 注入。如果绕过 `load_cfg()`、先 `validate_config()` 再注册插件，主路径的 schema 合并顺序就不成立。CLI 主路径没有这个问题，因为 `load_cfg()` 先 `prepare_plugins()` 再 `validate_config()`（`src/axolotl/cli/config.py:306-320`）。

> 容易误解点 2：只写插件名是否足够？主路径下通常足够。

`CutCrossEntropyArgs.cut_cross_entropy` 默认是 `True`（`args.py:33`），因此只要插件被注册且没有显式 `cut_cross_entropy: false`，`cfg.cut_cross_entropy` 会为真。示例 `examples/qwen3.5/27b-qlora.yaml:6-8` 就是这种写法。

> 容易误解点 3：CCE 和 chunked cross entropy 都是“省 logits 显存”，是否可以叠加？不能。

插件自身会拒绝 `chunked_cross_entropy`（`args.py:46-54`），全局校验也会拒绝多个 CE 优化同时开启（`validation.py:974-1002`）。这是因为它们都要接管同一个 loss 计算位置，叠加不会更安全，反而容易出现重复 patch 或 loss 语义不一致。

## 1.5 本章小结

> 💡 **小结**
>
> * CCE 的用户入口是 `plugins`，不是基础 schema 的固定字段。
> * CLI 主路径先注册插件再校验配置，使 `CutCrossEntropyArgs` 能参与 Pydantic 校验。
> * 添加插件通常默认启用 CCE；禁用需要显式 `cut_cross_entropy: false`。
> * CCE 与 chunked CE、Liger CE 是互斥的 loss 优化路径。

# 二、依赖检查与 pre-model-load patch：为什么必须在模型加载前动手

## 2.1 设计哲学与核心问题

CCE 的接入点不是 Trainer，而是模型类的 `forward()`。这带来一个时序要求：如果模型实例已经被 `AutoModelForCausalLM.from_pretrained()` 创建，再 patch 类方法虽然有时也能影响实例方法查找，但对 remote code、多模型类型、类命名空间缓存都会更脆弱。Axolotl 选择在 **模型真正 build 前** 让插件执行 `pre_model_load()`，先把目标模型类的 `forward` 替换好，再加载权重。

这层解决的是 **初始化时序 + patch 命名空间** 问题。

## 2.2 源码入口与关键对象

```text
src/axolotl/loaders/model.py
  - ModelLoader.load：模型加载主路径，调用 PLUGIN_MANAGER.pre_model_load 后才 _build_model

src/axolotl/integrations/base.py
  - PluginManager.pre_model_load：遍历所有插件，调用插件 hook

src/axolotl/integrations/cut_cross_entropy/__init__.py
  - CutCrossEntropyPlugin._check_requirements：检查 PyTorch、包、transformers extra、Axolotl fork 标志
  - CutCrossEntropyPlugin.pre_model_load：调用 patch_llama_like 和下游 cce_patch
  - CutCrossEntropyPlugin.patch_llama_like：为未知 llama-like model_type 注册 generic patch

scripts/cutcrossentropy_install.py
  - 输出安装 Axolotl fork 的命令
```

## 2.3 主流程拆解

模型加载路径的关键片段在 `ModelLoader.load()`：

```text
ModelLoader.load
  -> patch_manager.apply_pre_model_load_patches()        # src/axolotl/loaders/model.py:168-169
  -> self._apply_pre_model_load_setup()                  # model_config / attention / quantization 等
  -> PLUGIN_MANAGER.pre_model_load(self.cfg)             # src/axolotl/loaders/model.py:173
  -> patch_manager.apply_post_plugin_pre_model_load_patches()
  -> self._build_model()                                 # src/axolotl/loaders/model.py:176
```

这行 `PLUGIN_MANAGER.pre_model_load(self.cfg)` 是 CCE 第一次真正改变运行行为的地方。`PluginManager.pre_model_load()` 只是简单遍历插件（`src/axolotl/integrations/base.py:439-447`），真正逻辑在 CCE 插件：

```python
# src/axolotl/integrations/cut_cross_entropy/__init__.py:86-103（简化）
def pre_model_load(self, cfg):
    if cfg.cut_cross_entropy:
        self._check_requirements()
        self.patch_llama_like(cfg.model_config_type)
        from cut_cross_entropy.transformers.patch import cce_patch
        cce_patch(
            cfg.model_config_type,
            remote_model_id=cfg.base_model if cfg.trust_remote_code else None,
        )
```

依赖检查分三层：

1. PyTorch 版本必须 `>= 2.4.0`（`__init__.py:54-59`）；
2. 必须能 import `cut_cross_entropy` 和 `cut_cross_entropy.transformers`（`__init__.py:61-72`）；
3. 必须是 Axolotl fork，检查 `cut_cross_entropy.transformers.patch.AXOLOTL_CCE_FORK`（`__init__.py:74-84`）。

安装脚本也体现了这个约束：Torch `<2.4.0` 时直接输出空字符串并退出（`scripts/cutcrossentropy_install.py:14-19`），否则安装固定 fork `axolotl-ai-cloud/ml-cross-entropy.git@fec1a88`（`scripts/cutcrossentropy_install.py:30-33`）。

`patch_llama_like()` 是 Axolotl 在下游支持表之外加的一层兜底。它读取下游 `PATCH_FNS`，如果当前 `model_config_type` 不在表里，就注册一个 `patch_generic`：

```python
# src/axolotl/integrations/cut_cross_entropy/__init__.py:142-150（简化）
if model_type_to_patch not in PATCH_FNS:
    PATCH_FNS[model_type_to_patch] = partial(
        patch_generic, model_type=model_type_to_patch
    )
```

`patch_generic()` 的策略是：按 `transformers.models.{model_type}.modeling_{model_type}` 动态导入模块，再通过 `get_causal_lm_model_cls_prefix()` 找到 `{Prefix}ForCausalLM`，最后把 `model_cls.forward = cce_forward`（`__init__.py:123-135`）。这个 generic fallback 不是“万金油”，它假设目标架构的 causal LM 类形态足够像 Llama。

下游 `cce_patch()` 的分发逻辑则在外部 fork：

```text
axolotl-ai-cloud/ml-cross-entropy@fec1a88:cut_cross_entropy/transformers/patch.py
  PATCH_FNS: model_type -> (module_path, patch_fn_name)     # lines 15-88
  _get_patch_fn(): lazy import 并缓存 patch_fn              # lines 91-120
  cce_patch(): 构造 PatchOptions，调用对应 patch_fn          # lines 151-199
```

其中 `PATCH_FNS` 列出了大量显式支持模型类型（`patch.py:15-88`），而 `_get_patch_fn()` 第一次调用时 lazy import，随后把 tuple 替换为 callable 以缓存（`patch.py:101-115`）。

## 2.4 关键细节与误区澄清

> 容易误解点 4：CCE 不是在 Trainer 创建后“换 loss function”。

Axolotl 的 `setup_trainer()` 只是选择 SFT/RL builder（`src/axolotl/utils/trainer.py:708-720`），`HFCausalTrainerBuilder` 创建 trainer 时没有给 CCE 特设 `compute_loss_func`（`src/axolotl/core/builders/causal.py:431-439`）。CCE 真正改变的是模型类 `forward()`，而且发生在 `_build_model()` 前（`loaders/model.py:173-176`）。

> 容易误解点 5：日志“Applying Cut Cross Entropy”不等于所有模型都会走到 CCE loss。

插件会调用 `cce_patch(cfg.model_config_type)` 并打印日志（`__init__.py:94-103`），但是否命中训练时的实际类取决于下游 patch 是否替换了正确类。例如文档明确提醒：如果模型实际加载为 `ForConditionalGeneration`，而 generic fallback 只 patch 了 `ForCausalLM`，显存可能没有下降（`docs/agents/new_model_support.md:136-148`）。

> 容易误解点 6：`trust_remote_code=True` 不是 generic fallback 的万能通行证。

Axolotl 调用下游 `cce_patch(..., remote_model_id=cfg.base_model if cfg.trust_remote_code else None)`（`__init__.py:100-103`）。下游显式 patch 函数如果支持 `remote_model_id`，会通过 `patch_remote_model_class()` 下载 remote code 并替换类 forward（外部 `patch.py:194-199`、`transformers/utils.py:168-199`）。但 Axolotl 的 generic `patch_generic()` 虽然参数里有 `remote_model_id`，实现中并没有用它，而是导入本地 `transformers.models.{model_type}`（`__init__.py:123-135`）。所以 remote 模型最好有下游专门 patch。

## 2.5 本章小结

> 💡 **小结**
>
> * CCE 的第一个行为改变点是 `ModelLoader.load()` 中的 `PLUGIN_MANAGER.pre_model_load()`。
> * 插件会先检查 PyTorch、外部包、transformers extra 和 Axolotl fork 标志。
> * patch 是进程内 monkey patch：替换模型类或实例的 `forward()`，不是保存到 checkpoint 的状态。
> * generic fallback 只适合 llama-like causal LM；多模态和 remote code 需要更精确的下游 patch。

# 三、Forward 主路径：避开 logits，而不是换一个 CrossEntropyLoss

## 3.1 设计哲学与核心问题

标准 causal LM loss 的显存峰值来自完整 logits：

```text
hidden_states: [B, S, H]
lm_head.weight: [V, H]
logits = hidden_states @ weight.T -> [B, S, V]
loss = cross_entropy(logits, labels)
```

CCE 的设计不是“更快的 `F.cross_entropy`”，而是 **跳过 `[B,S,V]` 的完整物化**。它直接从 hidden states `e`、分类器权重 `c`、labels 计算需要的两部分：

- 每个 token 的 `logsumexp(e @ c.T)`；
- 每个 token 对应 label 的正确类别 logit。

这层解决的是 **每 step 前向 loss 的显存峰值**。

## 3.2 源码入口与关键对象

```text
src/axolotl/core/trainers/base.py
  - AxolotlTrainer.compute_loss：常规路径仍交给 super().compute_loss，由模型 forward 返回 loss

axolotl-ai-cloud/ml-cross-entropy@fec1a88:cut_cross_entropy/transformers/llama.py
  - cce_forward：典型 patched forward；labels 存在时调用 apply_lce，否则走普通 logits
  - patch_llama：设置模块级 _PATCH_OPTS，并替换 LlamaForCausalLM.forward

axolotl-ai-cloud/ml-cross-entropy@fec1a88:cut_cross_entropy/transformers/utils.py
  - PatchOptions.use_lce：判断是否启用 linear CE
  - apply_lce：处理 reduction / DTensor / ZeRO-3 / dtype，然后调用 linear_cross_entropy

axolotl-ai-cloud/ml-cross-entropy@fec1a88:cut_cross_entropy/linear_cross_entropy.py
  - linear_cross_entropy：选择 cce / torch_compile 实现
```

## 3.3 主流程拆解

Axolotl Trainer 的 SFT 主路径并没有为 CCE 特判。`AxolotlTrainer.compute_loss()` 做了一些 token 统计和 Gemma4 特例后，普通路径直接调用父类：

```python
# src/axolotl/core/trainers/base.py:455-460
return super().compute_loss(
    model,
    inputs,
    return_outputs=return_outputs,
    num_items_in_batch=num_items_in_batch,
)
```

因此真正的 loss 仍来自 `model(**inputs)` 的输出。patched Llama forward 的核心逻辑如下：

```python
# 外部 fork cut_cross_entropy/transformers/llama.py:53-98（简化）
outputs = self.model(...)
hidden_states = outputs.last_hidden_state
loss = None
logits = None
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

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
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, ...)

return CausalLMOutputWithPast(loss=loss, logits=logits, ...)
```

这段代码有一个非常关键的状态变化：

```text
训练且 labels 存在:
  outputs.last_hidden_state: [B, S, H]
  logits: None
  loss: apply_lce(...)

推理 / 无 labels / train_only 限制:
  logits: self.lm_head(...): [B, S, V] 或 [B, keep, V]
  loss: 普通 loss 或 None
```

`PatchOptions.use_lce()` 的判断也很克制：labels 为 `None` 就不走 CCE；如果 `train_only=True` 且当前不在 training，也不走 CCE（外部 `transformers/utils.py:87-94`）。Axolotl 调用 `cce_patch()` 时没有显式传 `train_only`，下游默认是 `False`（外部 `patch.py:151-161`），所以只要有 labels，训练和评估 loss 都可能走 CCE。

`apply_lce()` 接手后做几件工程化处理：

1. 如果 Trainer 传了 `num_items_in_batch` 且 reduction 是 mean，则先改成 sum，最后手动除以 `num_items_in_batch`（外部 `transformers/utils.py:106-112`、`162-164`）；
2. 如果 `lm_head.weight` 是 DTensor，尝试设置 vocab parallel options 并取 local shard（外部 `transformers/utils.py:113-135`）；
3. 如果是 DeepSpeed ZeRO-3 sharded parameter，把原始参数引用塞进 `zero3_params`，供 backward 重新 gather（外部 `transformers/utils.py:137-147`）；
4. 如果 DoRA 等路径导致 hidden states 是 fp32、权重是 bf16，则把 `e` 转回权重 dtype（外部 `transformers/utils.py:148-150`）；
5. 最后调用 `linear_cross_entropy(e, c_local, labels.to(e.device), shift=True, ...)`（外部 `transformers/utils.py:152-160`）。

`shift=True` 对 causal LM 很关键：它表示 token `< n` 预测 token `n`，而不是外部先手动切 `logits[..., :-1, :]`。

## 3.4 关键细节与误区澄清

> 容易误解点 7：CCE 不是把 `torch.nn.functional.cross_entropy` 全局替换掉。

对比 Liger 插件：Liger 在某些路径会替换 `nn.functional.cross_entropy` 或模型模块里的 loss 类（`src/axolotl/integrations/liger/plugin.py:119-123`、`147-151`）。CCE 没有这么做；它替换模型 `forward()`，让 `forward()` 在 labels 存在时直接调用 `apply_lce()`。

> 容易误解点 8：patched forward 返回的 `logits` 可能是 `None`，这是设计结果。

在 CCE 分支里 `logits` 初始化为 `None` 后没有赋值（外部 `llama.py:64-80`），最后返回 `CausalLMOutputWithPast(logits=logits)`（`llama.py:92-98`）。如果某个 callback 或自定义 Trainer 在训练时强依赖 `outputs.logits`，它可能和 CCE 不兼容。

> 容易误解点 9：CCE 省的是 loss 阶段 logits，不是整个前向的所有激活。

`self.model(...)` 仍然会计算 Transformer 层 hidden states（外部 `llama.py:53-64`）。注意力 KV、中间 MLP 激活、checkpointing 策略不由 CCE 管。CCE 避免的是 `self.lm_head(hidden_states)` 生成完整 `[B,S,V]` logits 以及后续 cross entropy 对 logits 的处理。

## 3.5 本章小结

> 💡 **小结**
>
> * Axolotl Trainer 主路径不感知 CCE；它仍然向模型要 `loss`。
> * CCE patched forward 在 labels 存在时不物化完整 logits，而是调用 `apply_lce(hidden_states, lm_head.weight, labels)`。
> * CCE 分支下 `outputs.logits` 可能为 `None`，这对依赖 logits 的外部逻辑是兼容风险。
> * `shift=True` 把 causal LM 的 label shift 放进 CCE 内部处理。

# 四、完整主路径串联：一次真实训练调用里 CCE 在哪里生效

## 4.1 完整调用栈

```text
User: axolotl train examples/qwen3.5/27b-qlora.yaml
  │
  ├─ Step 1: CLI 进程调度
  │     ├─ pyproject.toml:163-164 注册 `axolotl = axolotl.cli.main:main`
  │     ├─ src/axolotl/cli/main.py:98-128 train()
  │     └─ src/axolotl/cli/utils/train.py:179-185
  │        生成 `accelerate launch -m axolotl.cli.train <config>`
  │
  ├─ Step 2: 配置加载与插件 schema 合并
  │     ├─ src/axolotl/cli/train.py:55-91 do_cli()
  │     ├─ src/axolotl/cli/config.py:244-253 读取 YAML
  │     ├─ src/axolotl/cli/config.py:306 prepare_plugins()
  │     └─ src/axolotl/utils/config/__init__.py:332-337 merge_input_args()
  │
  ├─ Step 3: 模型类型识别与加载准备
  │     ├─ src/axolotl/utils/config/__init__.py:171-201
  │     │  load_model_config() 并写入 cfg.model_config_type
  │     └─ src/axolotl/loaders/model.py:136-144
  │        初始化 ModelLoader 与 PatchManager
  │
  ├─ Step 4: pre-model-load patch
  │     ├─ src/axolotl/loaders/model.py:168-176
  │     │  PLUGIN_MANAGER.pre_model_load(cfg) 发生在 _build_model() 前
  │     ├─ src/axolotl/integrations/cut_cross_entropy/__init__.py:86-103
  │     │  检查依赖并调用 cce_patch(model_config_type)
  │     └─ external cut_cross_entropy/transformers/patch.py:151-199
  │        分发到具体模型 patch 函数
  │
  ├─ Step 5: Trainer 创建与训练
  │     ├─ src/axolotl/train.py:522-570 setup_model_and_trainer()
  │     ├─ src/axolotl/utils/trainer.py:708-720 选择 HFCausal/HFRL builder
  │     └─ src/axolotl/train.py:183-229 execute_training()
  │        trainer.train(...)
  │
  ├─ Step 6: 每个 step 的 loss
  │     ├─ src/axolotl/core/trainers/base.py:455-460 super().compute_loss()
  │     ├─ external transformers/llama.py:53-80 patched forward
  │     └─ external cce.py:47-218 自定义 autograd Function
  │
  └─ Step 7: 保存
        ├─ src/axolotl/train.py:632-640 save_trained_model + cleanup
        └─ CCE 不写入 state_dict；resume 时靠同一配置重新 patch
```

## 4.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 频率 |
|---|---|---|---|---|---|
| CLI launcher | config path、launcher args | 新训练进程，通常是 `accelerate launch -m axolotl.cli.train` | launcher 自身可能建分布式进程 | 无直接影响 | 一次 |
| 配置加载 | YAML + CLI overrides | `cfg.plugins` 注册；动态 schema 包含 `cut_cross_entropy` | 无 | 无 | 一次 |
| normalize | base model config | `cfg.model_config_type` 写入，如 `qwen3_5` | 可能访问 HF/local config | 无 | 一次 |
| CCE pre_model_load | `cfg.cut_cross_entropy`、`model_config_type` | 替换模型类 `forward`；设置下游 `_PATCH_OPTS` | 无 | 无直接节省 | 每次模型加载 |
| `_build_model()` | patched 类命名空间 | 加载模型实例与权重 | 取决于 FSDP/ZeRO/TP | 不由 CCE 决定 | 一次或 ref model 一次 |
| Trainer step | batch tensors | `model(**inputs)` 返回 loss；CCE 分支 logits=None | 常规 DDP/FSDP 通信；VP/ZeRO-3 特例见后文 | 避免 `[B,S,V]` logits | 每个 step |
| Backward | loss graph | CCE 自定义 backward 产出 `de/dc/dbias` | VP all_reduce 或 ZeRO-3 gather 特例 | 避免 logits grad buffer | 每个 step |
| Save/resume | model/trainer state | 普通 `save_pretrained`/FSDP save；CCE 不进 state_dict | 取决于 FSDP/ZeRO 保存策略 | 无 CCE 特有收益 | 保存时 |

保存路径证明 CCE 没有特殊状态：`save_trained_model()` 只按 FSDP、DeepSpeed、rank0 等训练策略保存模型（`src/axolotl/train.py:254-386`），没有调用 CCE 插件的 save/load hook，也没有保存 `_PATCH_OPTS`。这意味着 resume 依赖同一 YAML 再次启用插件。

## 4.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `src/axolotl/monkeypatch/loss/chunked.py` | 同样是 cross entropy 显存优化 | CCE 开启时不应进入 | `chunked_cross_entropy` 是另一条互斥路径；由 PatchManager pre-model-load patch Transformers loss mapping（`patch_manager.py:261-268`） |
| `src/axolotl/monkeypatch/llama_attn_hijack_flash.py` | 名字里有 cross entropy | 只有 `flash_attn_cross_entropy` 且 llama-derived 才走 | CCE 不依赖该 patch；全局校验禁止多个 CE 优化同时开启 |
| `src/axolotl/core/builders/causal.py` 的 `compute_loss_func` | 似乎可以注入自定义 loss | CCE 不通过这里注入 | 只有 EAFT 等路径会设置 `trainer_kwargs["compute_loss_func"]`（`causal.py:383-394`） |
| `PluginManager.post_model_load/post_trainer_create/post_train` | 插件生命周期 hook 很多 | CCE 插件没有实现这些 hook | CCE 只用 `get_input_args()` 和 `pre_model_load()` |
| `scripts/cutcrossentropy_install.py` | 像 runtime 初始化 | 不是训练主流程 | 只是安装命令生成器；训练时只做 import 检查 |
| `cut_cross_entropy.transformers.patch.cce_patch(model_instance)` | 下游支持传模型实例 | Axolotl 传的是 `model_config_type` 字符串 | Axolotl 选择类级 patch，在 `_build_model()` 前执行 |

## 4.4 本章小结

> 💡 **小结**
>
> * CCE 主路径可以概括为：配置注册 → 模型加载前 patch → Trainer 常规调用模型 → patched forward 返回 loss。
> * CCE 不改变数据集加载、Trainer 类选择、optimizer 创建和保存格式。
> * 保存 / resume 没有 CCE 专属状态；恢复训练必须再次用包含插件的配置加载模型。
> * 与 chunked CE、flash-attn CE 的相似点只在目标相同，实际 patch 点完全不同。

# 五、关键 shape / 状态 / 通信流：到底省了哪块显存

## 5.1 Tensor shape 变化

以普通 SFT、无 Tensor Parallel vocab sharding 的 CCE 路径为例：

```text
输入 batch:
  input_ids: [B, S]
  labels:    [B, S]

Transformer 主体:
  hidden_states = self.model(...).last_hidden_state
  hidden_states: [B, S, H]

CCE forward 入口:
  e = hidden_states[:, slice_indices, :]
  c = lm_head.weight
  e: [B, S, H]
  c: [V, H]
  labels: [B, S]

CCE 内部整理:
  valids = non-ignore label positions after shift
  e = e.flatten(0, -2)
  targets = labels.flatten()
  e: [B*S, H]
  targets: [B*S]

Triton LSE kernel:
  lse: [num_valid_tokens or B*S]
  不产生 [B*S, V] logits，只分 block 计算 logsumexp

Indexed correct-logit kernel:
  neg_dot: [num_valid_tokens]

loss:
  nll = neg_dot + lse
  reduction -> scalar / per-token loss
```

对应源码依据：

- `cce_linear_cross_entropy()` 要求 `e.size()[0:-1] == targets.size()`、`e.size(-1) == c.size(1)`（外部 `cce.py:255-256`）；
- 它把 `e` 和 `targets` flatten（外部 `cce.py:263-272`）；
- `_build_flat_valids()` 根据 `ignore_index` 和 `shift` 选有效 token（外部 `utils.py:25-47`）；
- `cce_lse_forward_kernel()` 只分配 `lse: [B]` 和可选 `logit_avg: [V]`，没有分配 logits 矩阵（外部 `cce_lse_forward.py:194-210`）；
- `indexed_neg_dot_forward_kernel()` 输出 `out: [B]`，只取 label 对应类别（外部 `indexed_dot.py:101-158`）。

为什么这样能省显存？因为标准路径中最大临时张量是：

```text
logits: [B, S, V]
```

CCE 把它变成一系列 block 内部计算，长期驻留的主要中间结果变成：

```text
lse:     [B*S]
neg_dot: [B*S]
可选 logit_avg: [V]
```

对于 `B=1, S=8192, V=151936, bf16`，单个 logits 张量理论大小约为：

```text
1 × 8192 × 151936 × 2 bytes ≈ 2.32 GiB
```

这还没算 logits.float()、梯度或额外 view/contiguous 的峰值。CCE 避开的正是这部分。

## 5.2 Rank / Mesh / Process Group 变化

普通 DDP/FSDP 下，Axolotl CCE 自身不创建新的 process group。它只是在每个 rank 的本地模型前向里替换 loss 计算。通信仍来自 DDP/FSDP/ZeRO，而不是 CCE 主动调度。

但下游 CCE fork 有两个分布式兼容分支。

### Vocab Parallel / Tensor Parallel DTensor

如果 `lm_head.weight` 是 DTensor，`apply_lce()` 会：

```text
c is DTensor
  -> device_mesh = c.device_mesh
  -> process_group = device_mesh.get_group("tp")
  -> VocabParallelOptions.from_vocab(vocab_size, process_group, reduce_e_grad=True)
  -> c_local = c.to_local()
```

源码在外部 `transformers/utils.py:113-135`。随后 `cce.forward()` 会在 VP 模式下额外通信：

```text
forward:
  local lse over vocab shard
  all_reduce MAX over tp group
  all_reduce SUM over tp group
  all_reduce correct_logit over tp group

backward:
  cce_backward_kernel computes local de/dc
  all_reduce de over tp group if reduce_e_grad=True
```

源码依据：

- `vp_reduce_lse()`：一次 MAX all_reduce + 一次 SUM all_reduce（外部 `vocab_parallel/utils.py:47-54`）；
- `vp_reduce_correct_logit()`：一次 all_reduce（外部 `vocab_parallel/utils.py:57-65`）；
- `vp_reduce_e_grad()`：一次 all_reduce（外部 `vocab_parallel/utils.py:68-76`）；
- CCE forward 调用 VP reduce 的位置在外部 `cce.py:75-118`；backward 设置 `reduce_e_grad` 和 `pg` 在 `cce.py:160-177`，最终 kernel 后 all_reduce 在 `cce_backward.py:474-476`。

可以画成：

```text
world_size = 4, tensor parallel vocab shard

rank0: vocab [0, V/4)       local lse_0, correct_logit_0
rank1: vocab [V/4, V/2)     local lse_1, correct_logit_1
rank2: vocab [V/2, 3V/4)    local lse_2, correct_logit_2
rank3: vocab [3V/4, V)      local lse_3, correct_logit_3

每个 token:
  global_lse = logsumexp(concat shard logits)
             = all_reduce(max) + all_reduce(sum-exp)
  correct_logit = all_reduce(one rank has value, others zero)
```

### DeepSpeed ZeRO-3

ZeRO-3 下 `lm_head.weight` 可能是分片参数。`apply_lce()` 如果看到 `c` 或 bias 有 `ds_id`，会把原始 Parameter 放入 `zero3_params`（外部 `transformers/utils.py:137-147`）。在 backward 中，CCE 会用 DeepSpeed `GatheredParameters` 重新 gather 完整参数，再读 `.data` 进入 backward kernel（外部 `cce.py:179-195`）。

这不是 Axolotl 自己调度的通信，但确实是 CCE backward 触发的兼容成本。

## 5.3 状态切换

CCE 有三类状态：

```text
Axolotl 侧:
  PluginManager.plugins: OrderedDict[str, BasePlugin]
  PluginManager.cfg: 当前配置

下游 patch 分发表:
  PATCH_FNS: model_type -> patch function 或 (module_path, function_name)
  第一次 lazy import 后会缓存成 callable

每个模型 patch 模块:
  _PATCH_OPTS: PatchOptions | None
  patched class.forward = cce_forward
```

状态读写路径：

```text
进入 pre_model_load:
  CutCrossEntropyPlugin.pre_model_load(cfg)
    -> patch_llama_like() 可能修改 PATCH_FNS
    -> cce_patch(model_type)
      -> PatchOptions(...)
      -> patch_fn(..., patch_options)
        -> module._PATCH_OPTS = patch_options
        -> ModelClass.forward = cce_forward

训练中:
  cce_forward()
    -> 读取模块级 _PATCH_OPTS
    -> _PATCH_OPTS.use_lce(labels, self.training)

退出训练:
  没有 unpatch；进程结束即释放
```

线程/进程安全层面，这些状态是 **进程内全局**。分布式训练通常是多进程，每个 rank 都会独立 patch 自己的 Python 进程；同一进程内如果先后加载不同模型类型，patch 状态不会自动恢复。

## 5.4 本章小结

> 💡 **小结**
>
> * CCE 的 shape 核心是从 `[B,S,V]` logits 转成 `[B*S]` 级别的 lse / correct-logit 中间量。
> * 常规 DDP/FSDP 下 CCE 不引入新的通信组；通信来自训练策略本身。
> * TP vocab sharding 会让 CCE 在 loss 前向和反向引入 all_reduce。
> * CCE patch 状态是进程内全局 monkey patch，没有自动 unpatch 或上下文恢复。

# 六、核心机制深挖：patch、autograd 与兼容分支

## 6.1 Monkey Patch：零侵入接入还是维护风险？

### 解决什么问题

Axolotl 希望不改 Trainer、不 fork Transformers 主体，也能让几十种模型走 CCE loss。最轻的接入点就是替换模型类 `forward()`。

### 为什么不能更简单

如果只替换 loss function，标准 `forward()` 往往已经算出了完整 logits；显存峰值已经发生。CCE 必须在 `lm_head(hidden_states)` 之前接管，所以 patch 位置必须在模型 `forward()` 内部。

### 源码怎么做

典型 Llama patch：

```python
# external cut_cross_entropy/transformers/llama.py:101-128（简化）
def patch_llama(maybe_model, patch_options, remote_model_id=None):
    global _PATCH_OPTS
    _PATCH_OPTS = patch_options

    if remote_model_id is not None:
        patch_remote_model_class(...)
        return None

    from transformers.models.llama import modeling_llama
    if isinstance(maybe_model, transformers.PreTrainedModel):
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_llama.LlamaForCausalLM.forward = cce_forward
```

Axolotl 传的是字符串 `cfg.model_config_type`，因此通常走类级 patch：`modeling_llama.LlamaForCausalLM.forward = cce_forward`。

### 隐藏假设与副作用

- 隐藏假设 1：目标模型的 forward 签名与 patched 版本兼容；
- 隐藏假设 2：实际加载类就是被 patch 的类；
- 隐藏假设 3：下游 fork 的 `PATCH_FNS` 覆盖当前 `transformers==5.5.4` 的模型实现；
- 副作用：同一进程内该类后续实例都会受影响；没有恢复函数。

## 6.2 自定义 autograd：前向和反向并不只是“省一个 matmul”

### 解决什么问题

CCE 需要在不物化 logits 的情况下完成 forward loss 和 backward 梯度。仅 forward 省 logits 不够，反向如果再构造完整 softmax/logits grad，同样会把显存吃回来。

### 源码怎么做

核心类是外部 `LinearCrossEntropyFunction`（`cce.py:47-218`）：

```text
forward:
  cce_lse_forward_kernel(e, c, ...)        -> lse
  indexed_neg_dot_forward_kernel(e, c, y)  -> -correct_logit
  nll = neg_dot + lse
  save_for_backward(e, c, bias, lse, targets, valids, logit_avg)

backward:
  cce_backward_kernel(do, e, c, lse, targets, ...)
  return de, dc, dbias
```

前向中 `cce_lse_forward_kernel()` 分 block 遍历 vocab 计算 logsumexp（外部 `cce_lse_forward.py:12-119`），输出只分配 `lse`（`cce_lse_forward.py:194-210`）。正确类别 logit 通过 indexed dot 单独取出（外部 `indexed_dot.py:101-158`）。

反向中 `cce_backward_kernel()` 明确要求 `e` 和 `c` 是 fp16/bf16（外部 `cce_backward.py:349-356`），这也解释了 Axolotl 插件为什么在 schema 层要求 `bf16` 或 `fp16`（`src/axolotl/integrations/cut_cross_entropy/args.py:35-42`）。

### 隐藏假设与维护风险

- hidden states 和 lm_head weight 的 dtype 必须满足 kernel 要求；
- CUDA / Triton kernel 是核心依赖；外部包缺失时训练无法启用；
- `filter_eps`、`accum_*_fp32` 等实现细节由下游默认值控制，Axolotl 当前没有暴露这些参数；
- 下游模型 `forward()` 改签名时，patched forward 需要同步。

## 6.3 配置归一化：用户配置如何变成真实行为

### 用户配置在哪里读取

`load_cfg()` 读取 YAML 后先 `prepare_plugins(cfg)`（`src/axolotl/cli/config.py:244-306`），再 `validate_config()`，后者在有插件时 `merge_input_args()`（`src/axolotl/utils/config/__init__.py:324-337`）。

### schema 是否校验

校验分三层：

| 校验 | 源码 | 行为 |
|---|---|---|
| 插件字段默认值 | `args.py:28-33` | 插件存在时 `cut_cross_entropy` 默认 True |
| dtype | `args.py:35-42` | CCE 开启但无 `bf16/fp16` 报错 |
| CE 优化互斥 | `validation.py:974-1002` | CCE / chunked / Liger CE 多开报错 |

### 是否写入环境变量

未在 Axolotl CCE 源码中确认有任何 CCE 专属环境变量。`prepare_optim_env()` 会设置 accelerate/deepspeed 等环境变量（例如 `src/axolotl/utils/trainer.py:528-676`），但 CCE 插件本身没有写 env。

### 是否传给下游库

Axolotl 只传两个信息给下游：

```text
model_type: cfg.model_config_type
remote_model_id: cfg.base_model if cfg.trust_remote_code else None
```

见 `src/axolotl/integrations/cut_cross_entropy/__init__.py:100-103`。下游 CCE 的 `impl/reduction/filter_eps/accum_*` 都使用默认参数（外部 `patch.py:151-161`）。

## 6.4 本章小结

> 💡 **小结**
>
> * CCE 必须 patch `forward()`，因为省显存的关键发生在 `lm_head` 物化 logits 之前。
> * 下游 autograd 同时接管 forward 和 backward，避免反向再生成完整 logits 梯度。
> * Axolotl 当前只暴露 CCE 开关，不暴露下游 CCE kernel 的细粒度参数。
> * patch 的收益来自零侵入，代价是依赖模型类签名和下游适配表。

# 七、显存、性能与通信分析

## 7.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | `lm_head.weight`、Transformer 参数仍存在；CCE 不做参数分片 |
| optimizer state | ❌ | optimizer 创建和状态管理不经 CCE |
| Transformer 激活 | ❌ | `self.model(...)` 仍正常计算 hidden states |
| logits `[B,S,V]` | ✅ | CCE 分支不调用 `self.lm_head(hidden_states)` 生成完整 logits（外部 `llama.py:72-83`） |
| logits fp32 upcast | ✅ | 标准 CE 常见 `logits.float()` 峰值被避开；CCE kernel 内部分 block fp32 累积 |
| loss 中间量 | ✅ / 部分 | 长驻中间量主要是 `lse: [B*S]`、`neg_dot: [B*S]`；但 kernel 内部仍有 block buffer |
| 输入 batch | ❌ | `input_ids/labels/attention_mask` 不变 |
| TP vocab shard 通信 buffer | ❌ / 新增 | VP 模式下 all_reduce 需要临时 buffer |
| ZeRO-3 backward gather | ❌ / 新增 | ZeRO-3 兼容分支会 gather full lm_head/bias 参数用于 backward |

真正显存大头是 logits：`B*S*V*dtype_bytes`。长序列和大词表同时出现时，CCE 收益最明显。示例中 ALST 把 CCE 称为 Tiled Loss 组件，用于长上下文训练组合拳（`examples/alst/README.md:1-7`、`20-23`）。

收益会消失或减弱的场景：

- 推理无 labels，`use_lce()` 返回 False，patched forward 仍会算 logits（外部 `transformers/utils.py:87-94`）；
- 训练逻辑要求 `outputs.logits`，CCE 分支可能不兼容；
- 模型类没被正确 patch，日志有但实际 forward 不是 CCE；
- `logits_to_keep` 只保留很少 token 的评估/推理路径，本来 logits 就不大。

## 7.2 通信开销

### 常规单卡 / DDP / FSDP

CCE 自身没有新增通信。每 step 通信仍来自：

- DDP 梯度 all_reduce；
- FSDP 参数 all_gather / reduce_scatter；
- DeepSpeed ZeRO 的参数/梯度分片通信。

CCE 只改变 loss 计算图。

### Tensor Parallel vocab shard / DTensor

如果 `lm_head.weight` 是 DTensor 且 vocab 维被 shard，下游 CCE 会在 `tp` group 内通信：

| 阶段 | 通信 | 源码 | 频率 |
|---|---|---|---|
| forward lse | all_reduce MAX + all_reduce SUM | `vocab_parallel/utils.py:47-54` | 每次 loss forward |
| forward correct logit | all_reduce SUM | `vocab_parallel/utils.py:57-65` | 每次 loss forward |
| backward hidden grad | all_reduce SUM | `vocab_parallel/utils.py:68-76`、`cce_backward.py:474-476` | 每次 loss backward |

这不是每层通信，而是 **loss 层通信**。它的好处是避免 full vocab logits；代价是在 TP group 内对 `[B*S]` 级别张量做几次 all_reduce。

### DeepSpeed ZeRO-3

ZeRO-3 分支在 backward 用 `GatheredParameters` 重新 gather `lm_head.weight` / bias（外部 `cce.py:179-195`）。这相当于在 loss backward 阶段引入一次 full parameter gather 兼容成本。它不是 Axolotl 显式调用的 all_gather，但语义上会触发 ZeRO-3 参数聚合。

## 7.3 性能取舍

CCE 的取舍可以概括为：

```text
用更复杂的 Triton kernel + patch 维护成本
换掉完整 logits 的显存峰值
```

它不一定总是更快。原因有三类：

1. **kernel 分块与 atomic / lock 成本**：`cce_lse_forward_kernel()` 使用 locks 合并不同 vocab block 的 lse（外部 `cce_lse_forward.py:103-117`）；
2. **反向更复杂**：`cce_backward_kernel()` 需要处理 `de/dc/dbias`、锁、filter、可选 fp32 accum（外部 `cce_backward.py:361-487`）；
3. **分布式兼容成本**：VP all_reduce 和 ZeRO-3 gather 会引入额外通信。

但在 logits 显存成为瓶颈时，这个 trade-off 是值得的：它让用户能把 batch/seq/vocab 组合推到原本会 OOM 的区域。

## 7.4 本章小结

> 💡 **小结**
>
> * CCE 主要节省 logits 及其 loss 计算中间量，不节省参数、optimizer state 或 Transformer 激活。
> * 常规路径没有 CCE 专属通信；TP vocab shard 和 ZeRO-3 兼容分支会新增通信/聚合。
> * CCE 是典型的“kernel 复杂度 + patch 复杂度换显存峰值”方案。
> * 长序列、大词表、小显存微调最容易受益。

# 八、配置项、边界条件与坑点

## 8.1 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `plugins: [CutCrossEntropyPlugin]` | `cli/config.py:306` → `PluginManager.register()` | 注册插件，允许 schema 合并与 pre_model_load hook | 不写插件时 `cut_cross_entropy` 即使出现在 YAML 也不是主路径字段；`DictDefault` 缺失键返回 None（`utils/dict.py:6-12`） |
| `cut_cross_entropy: true/false` | `CutCrossEntropyPlugin.pre_model_load()` `__init__.py:86-103` | true 时检查依赖并 patch；false 时不 patch | 插件 schema 默认 true；想只注册不启用需显式 false |
| `bf16` / `fp16` | `args.py:35-42`、`core/builders/base.py:255-263` | 满足 CCE backward dtype 要求，并传入 TrainingArguments | `bf16: auto` 在 schema 前为真值，实际能否 bf16 还要看后续 normalize/硬件；kernel 反向仍要求 e/c 为 fp16/bf16 |
| `chunked_cross_entropy` | `args.py:46-54`、`patch_manager.py:261-268` | 与 CCE 互斥 | 同开会报错；chunked CE 是另一套 Transformers loss mapping patch |
| `liger_cross_entropy` / `liger_fused_linear_cross_entropy` | `validation.py:974-1002`、`integrations/liger/plugin.py` | 与 CCE 互斥 | 同开会报错；不要把“多个 loss 优化”当作可叠加 |
| `trust_remote_code` | `__init__.py:100-103`、外部 `patch_remote_model_class()` | 下游专用 patch 可 patch remote class | generic fallback 不真正使用 `remote_model_id`；remote 模型需专门适配 |
| `model_config_type` | `utils/config/__init__.py:179-201` | 决定下游 patch 分发 key | 如果模型 wrapper type 与实际 CausalLM 类不匹配，可能 patch 不生效 |
| `tensor_parallel_size` / DTensor lm_head | 外部 `apply_lce()` `transformers/utils.py:113-135` | 设置 vocab parallel loss | 需要 `device_mesh.get_group("tp")` 存在；会引入 loss all_reduce |
| `deepspeed` ZeRO-3 | 外部 `apply_lce()`、`cce.py:179-195` | backward gather full lm_head/bias | 兼容成本可能抵消部分收益；峰值取决于 ZeRO gather 实现 |
| `save_*` / FSDP save | `train.py:254-386` | 不改变 CCE；照常保存模型 | checkpoint 不携带 patch 状态；resume 必须重新加载插件 |

## 8.2 最小配置、默认行为与静默失效条件

最小开启配置：

```yaml
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
bf16: true   # 或 fp16: true / bf16: auto（视硬件与 normalize 而定）
```

默认行为：

- 插件存在时，`cut_cross_entropy` 默认 True；
- Axolotl 不暴露下游 `impl`，因此默认由外部包决定，一般非 Darwin 平台走 CCE kernel（外部 `linear_cross_entropy.py:15-23`）；
- `reduction` 默认 mean（外部 `patch.py:151-160`）。

可能静默失效或“看起来启用但收益不明显”的条件：

1. 没有把插件放进 `plugins`，只有裸 `cut_cross_entropy: true`；
2. 实际模型类不是被 patch 的类，例如 VLM `ForConditionalGeneration` 需要专用 patch；
3. 当前 step 没有 labels，`use_lce()` 返回 False；
4. 下游 patch 表没有支持当前 `model_type`，generic fallback 导入失败或 patch 到错误类；
5. callback/评估逻辑强依赖 logits，迫使走非 CCE 或产生兼容问题。

## 8.3 不兼容组合与特殊模型限制

源码直接确认的不兼容：

- CCE + `chunked_cross_entropy`：插件 validator 报错（`args.py:46-54`）；
- CCE + Liger CE 任一项：全局 validator 报错（`validation.py:974-1002`）；
- PyTorch `<2.4.0`：插件 `_check_requirements()` 报 ImportError（`__init__.py:54-59`）；
- 未安装 Axolotl fork：检查 `AXOLOTL_CCE_FORK` 失败（`__init__.py:74-84`）；
- backward 输入 dtype 非 fp16/bf16：下游断言失败（外部 `cce_backward.py:349-356`）。

示例层面也能看到模型支持边界：MiMo README 明确写 CCE 当前不支持（`examples/mimo/README.md:29-31`），Trinity 配置把 CCE 插件注释为 N/A（`examples/trinity/trinity-nano-preview-qlora.yaml:7-9`）。这些不是代码 validator，但它们提醒：模型支持不是只看“能不能 import”。

## 8.4 本章小结

> 💡 **小结**
>
> * CCE 的关键配置不是单个布尔值，而是“插件注册 + dtype + 模型 patch 命中”。
> * 最常见坑点是 patch 到了错误类，尤其是多模态 / remote code / wrapper 模型。
> * CCE 与其他 CE 优化互斥，不是越多越好。
> * checkpoint 不保存 patch 状态；保存/加载差异由训练配置和模型加载时 patch 决定。

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/e2e/integrations/test_cut_cross_entropy.py:16-66` | SmolLM2 + CCE 完整训练 10 steps 并检查输出 | 覆盖插件配置、训练、保存的 happy path |
| `tests/e2e/integrations/test_cut_cross_entropy.py:68-110` | Qwen2.5 + CCE 训练 | 覆盖另一类下游模型 patch |
| `tests/e2e/integrations/test_cut_cross_entropy.py:112-138` | CCE + flash/sdpa attention | 证明 CCE 可与注意力实现组合，注意不是 CE 优化叠加 |
| `examples/qwen3.5/27b-qlora.yaml:6-8` | 大模型 QLoRA 推荐写法 | 插件名默认启用 CCE |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:3-19` | FSDP2 + TP + CP 示例 | 说明 CCE 可出现在 ND parallel 配置中，但不是专门测试 |
| `examples/gemma3n/gemma-3n-e2b-qlora.yml:6-9` | 多模态/特殊模型显式 CCE | 依赖下游专用 patch |
| `docs/agents/new_model_support.md:136-148` | CCE patch 原理和常见 pitfall | 文档直接说明 forward 替换与错误类 patch 风险 |

## 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| schema 校验：无 bf16/fp16 时应报错 | 未在 `tests` 中找到针对 `CutCrossEntropyArgs` 的单元测试 | 配置错误可能只在训练前较晚暴露 |
| CCE 与 Liger/chunked 互斥 | 全局源码有 validator；未找到 CCE 专门单测 | 未来改 schema 时可能回归 |
| patch 是否真的命中实际模型类 | e2e 间接覆盖少数模型 | 新模型日志显示启用但显存不降 |
| 多机 / 多进程 TP vocab parallel | 未看到 CCE 专门多机测试 | loss all_reduce / DTensor group 语义可能在组合并行下出错 |
| DeepSpeed ZeRO-3 backward gather | 未看到 CCE + ZeRO-3 专测 | backward 中 full lm_head gather 可能造成峰值或性能问题 |
| 保存 / resume 后重新 patch | e2e 只检查输出存在 | resume 时如果配置缺插件，forward 回到普通 loss，显存行为变化 |
| 训练时依赖 `outputs.logits` 的 callback | 未见专测 | CCE 分支 logits=None 可能破坏自定义逻辑 |
| 性能 / 显存收益断言 | 未见自动化显存阈值测试 | 只能靠 profile / 用户观测发现 patch 失效 |

`tests/e2e/integrations/test_cut_cross_entropy.py` 是 smoke/e2e 风格：它证明“能训完并保存”，但没有断言 `model.forward` 已替换、没有比较 peak VRAM，也没有覆盖 VP/ZeRO-3 的通信分支。

## 9.3 本章小结

> 💡 **小结**
>
> * 当前测试主要覆盖少数模型的 happy path，而不是 CCE 机制完整性。
> * 最缺的是 patch 命中检测、显存收益检测、ZeRO-3/TP 分布式分支测试。
> * 示例覆盖面比测试更广，但示例不能替代自动化回归。
> * 对新模型支持来说，最好增加“forward 是否来自 CCE patch”的显式断言。

# 十、局限性与已知优化点

## 10.1 硬约束

1. **版本约束**：PyTorch `>=2.4.0`（`__init__.py:54-59`、`scripts/cutcrossentropy_install.py:14-19`）。
2. **安装约束**：必须安装带 transformers extra 的 Axolotl fork（`__init__.py:61-84`）。
3. **dtype 约束**：插件 schema 要求 `bf16` 或 `fp16`（`args.py:35-42`），下游 backward 也断言 embeddings/classifier 为 fp16/bf16（外部 `cce_backward.py:349-356`）。
4. **模型约束**：目标 `model_config_type` 必须在下游 `PATCH_FNS` 或 generic fallback 能正确 patch 的范围内（外部 `patch.py:15-88`、Axolotl `__init__.py:142-150`）。
5. **loss 互斥约束**：不能和 chunked CE / Liger CE 同开（`validation.py:974-1002`）。
6. **输出语义约束**：CCE 分支可能返回 `logits=None`（外部 `llama.py:64-98`）。

## 10.2 维护成本

- **上游 forward 签名漂移**：外部 `llama.py` 注释写明适配 Transformers v4.56.2（外部 `llama.py:1`），而 Axolotl 当前依赖 `transformers==5.5.4`（`pyproject.toml:20`）。patch 文件需要持续追踪上游签名变化。
- **模型适配表维护**：下游 `PATCH_FNS` 很长（外部 `patch.py:15-88`），新增模型要么进下游表，要么冒 generic fallback 风险。
- **进程全局污染**：类级 `forward` 替换没有 unpatch。测试或长生命周期服务里，如果同进程加载不同配置，需要小心状态残留。
- **配置追踪复杂**：`cut_cross_entropy` 来自插件动态 schema，不在基础 schema 中。工具若只读取基础 JSON schema，可能看不到该字段。

## 10.3 性能瓶颈

从源码可确认的潜在瓶颈：

1. `cce_lse_forward_kernel()` 对 vocab block 做 logsumexp，并用 locks 合并结果（外部 `cce_lse_forward.py:103-117`）；
2. `indexed_neg_dot_forward_kernel()` 对 label 对应类别做 indexed dot，仍要访问 `c` 的相关行（外部 `indexed_dot.py:62-85`）；
3. backward kernel 有 `de_locks/dc_locks` 等同步结构（外部 `cce_backward.py:422-435`）；
4. VP 模式每次 loss forward/backward 多次 all_reduce（外部 `vocab_parallel/utils.py:47-76`）；
5. ZeRO-3 backward 可能 gather full lm_head（外部 `cce.py:179-195`）。

这些成本说明：CCE 不是免费午餐，它把“巨大 logits 常驻显存”换成“更复杂 kernel + 可能的 loss 层通信”。

## 10.4 已知优化点

源码和工程行为提示了几个值得改进的方向：

- **显式 patch 命中检测**：训练前检查实际 `model.forward` 是否来自 CCE patch，避免“日志启用但没生效”。
- **暴露部分 CCE 下游参数**：如 `impl="torch_compile"`、`filter_eps`、`accum_e_fp32`、`accum_c_fp32`，让用户在不同 GPU/模型上做权衡。当前 Axolotl 调用下游 `cce_patch()` 时没有传这些参数（`__init__.py:100-103` 对照外部 `patch.py:151-161`）。
- **unpatch / context 化**：为测试或多模型同进程场景提供恢复机制。
- **ZeRO-3 / TP 专测和 profile**：把 loss 层 all_reduce / gather 纳入性能基线。
- **程序化配置路径统一**：`src/axolotl/cli/config.py.prepare_plugins()` 会调用 `plugin.register()`（`cli/config.py:215-220`），而 `src/axolotl/utils/config/__init__.py.prepare_plugins()` 只注册插件（`utils/config/__init__.py:381-389`）。虽然 CCE 当前 `register()` 为空，但这类重复入口容易造成未来插件行为不一致。

## 10.5 本章小结

> 💡 **小结**
>
> * CCE 的硬约束集中在版本、dtype、模型 patch 命中和 loss 互斥。
> * 最大维护成本来自 monkey patch 跟随 Transformers 模型 forward 演进。
> * 性能瓶颈从 logits 显存转移到 Triton kernel 分块、同步和可选分布式通信。
> * 最值得补强的是 patch 命中验证、分布式测试、unpatch 和参数暴露。

# 小结与展望

Axolotl 的 Cut Cross Entropy 实现可以用几个关键词概括。

## 关键词一：插件化 schema

CCE 不在基础 schema 中硬编码，而是通过 `CutCrossEntropyPlugin.get_input_args()` 动态注入 `CutCrossEntropyArgs`。这让外部依赖和模型兼容性被隔离在 integration 层，也解释了为什么正确入口是 `plugins` 而不是裸写一个布尔字段。

## 关键词二：pre-model-load monkey patch

CCE 的第一个真实行为改变点在模型加载前：`ModelLoader.load()` 调用 `PLUGIN_MANAGER.pre_model_load()`，插件再调用下游 `cce_patch()`。这让模型实例一开始就拥有 patched `forward()`。这种设计零侵入 Trainer，但代价是依赖模型类命名、forward 签名和 patch 表。

## 关键词三：hidden_states + lm_head.weight 直接算 loss

训练 step 中，patched forward 在 labels 存在时不生成完整 logits，而是调用：

```text
apply_lce(hidden_states, lm_head.weight, labels)
```

下游 CCE kernel 分别计算 logsumexp 和正确类别 logit，再用自定义 autograd 完成 backward。显存收益来自避开 `[B,S,V]`，而不是减少 Transformer 主体激活。

## 关键词四：通信换兼容，而不是默认通信换显存

常规 CCE 不新增通信；但当 `lm_head.weight` 是 DTensor vocab shard 或 DeepSpeed ZeRO-3 参数时，下游会引入 all_reduce 或 gather 兼容逻辑。这些分支让 CCE 能进入更复杂的分布式训练，但也带来性能和峰值不确定性。

## 关键词五：收益大，边界也硬

CCE 适合长序列、大词表、小显存微调，尤其是 QLoRA/ALST 这类 logits 峰值明显的场景。不适合依赖训练时 logits 输出的自定义逻辑，也不适合尚未有正确 patch 的多模态/remote/wrapper 模型。和 Liger fused CE、chunked CE 相比，它的优势是更彻底地绕过完整 logits；代价是更强的 monkey patch 维护负担和下游 kernel 依赖。

后续如果继续走读，最值得看的方向有三个：

1. **Liger fused linear cross entropy**：和 CCE 同样瞄准 loss 显存，但 patch 点、模型覆盖和 kernel 语义不同；
2. **Context Parallelism + CCE**：长序列训练中 CP 解决序列维度激活/注意力压力，CCE 解决 loss logits 压力，二者组合的通信边界值得单独拆；
3. **FSDP2 / TP / ZeRO-3 下的 loss 层通信**：CCE 在普通单卡很清爽，在组合并行下则变成 kernel 与通信协同问题。

如果用一句话评价 Axolotl 的 CCE 集成：它是一个很典型的开源训练框架工程取舍——用插件和 monkey patch 把复杂能力快速接入主训练链路，用最小框架侵入换来显著的 logits 显存收益；但模型适配、patch 命中和分布式边界，仍然需要开发者用源码意识去验证，而不能只相信“插件已启用”的日志。
