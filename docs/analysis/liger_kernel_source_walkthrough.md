# Axolotl 源码走读：Liger Kernel 实现解析

在大模型训练里，“显存不够”经常不是参数本身造成的，而是一些看起来很普通的中间张量把显存顶爆：比如最后一层 `lm_head` 之后的 logits，形状是 `[batch, seq, vocab]`；再比如 RMSNorm、RoPE、GLU 这类每层都会出现的小算子，单次不大，但长序列、多层、梯度检查点叠加后会形成可观的 kernel launch 与临时 buffer 开销。

Axolotl 对 Liger Kernel 的接入，正是围绕这个矛盾展开：它不试图重写训练循环，也不新建一套并行通信框架，而是在“模型真正被实例化之前”把 HuggingFace/Transformers 的若干类、函数、loss 路径替换成 Liger 的 Triton 实现。本文不展开 Liger Kernel 论文或 Triton 编程原理，而是沿着 Axolotl 的源码，追踪这个特性从 YAML 配置到训练 forward、FSDP 兼容、保存、测试覆盖的真实路径。

# 前言

## 业务 / 工程背景

Liger Kernel 在 Axolotl 中属于训练性能优化特性。它主要服务于 SFT、DPO/GRPO 等后训练场景里的三类问题：

1. **显存问题**：尤其是最后 `lm_head + cross_entropy` 路径中 `[B, S, V]` logits 的物化成本。
2. **小算子性能问题**：RMSNorm、LayerNorm、RoPE、SwiGLU/GEGLU 等在 Transformer block 内高频出现，融合后可以减少中间张量与 kernel launch。
3. **兼容问题**：Axolotl 要让这些替换能和 LoRA/QLoRA、FSDP/DeepSpeed、torch.compile、不同 Transformers 模型结构共存。

## 核心矛盾

Liger 的核心收益来自“替换模型内部实现”，但 Axolotl 又必须尽量不侵入 Trainer、dataset、checkpoint 等主框架逻辑。这带来一个典型工程冲突：

> 要想省掉 logits 或融合层内算子，patch 必须发生在模型类实例化之前；但 patch 一旦写入 Transformers 模块命名空间，就是全局状态，容易和版本、测试、其他模型加载路径互相污染。

另一个容易被忽略的冲突是：

> Liger Kernel 自身不负责参数切分或进程组通信；FSDP/DeepSpeed 仍然负责分布式语义。但 Liger FLCE 会直接读取 `lm_head.weight`，因此在 FSDP 包裹 `lm_head` 时必须保证权重在正确的 forward 上下文里被 all-gather 出来。

## 本文主线

本文按机制，而不是按文件，拆成几条主线：

1. 用户如何开启 Liger，以及配置如何进入动态 Pydantic schema。
2. 为什么第一个真正改变行为的函数是 `LigerPlugin.pre_model_load()`。
3. Liger 的 monkey patch 到底替换了什么，哪些替换是全局的。
4. `liger_fused_linear_cross_entropy` 如何改变 forward shape 与显存峰值。
5. FSDP、保存、GRPO/DPO 这些路径和 Liger 的关系边界。
6. 哪些看似相关的文件其实不是标准主路径，哪些测试证明了行为，哪些风险没有覆盖。

## 不展开的内容

本文不讲 Triton kernel 的编程模型，不讲 FSDP/DeepSpeed 基础原理，不讲 LoRA/QLoRA 原理，也不把 Liger Kernel 论文作为分析重点。我们只关心 Axolotl 源码里这个特性如何被接入、何时生效、节省了什么、代价在哪里。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/main.py` | 用户 `axolotl train` 的 Click 入口，负责把配置交给 launcher。 |
| `src/axolotl/cli/config.py` | 读取 YAML、注册插件、触发校验与归一化。 |
| `src/axolotl/integrations/config.py` | 将插件输入参数动态合并进 Axolotl Pydantic 配置。 |
| `src/axolotl/integrations/base.py` | `BasePlugin` 与 `PluginManager` 的插件生命周期定义。 |
| `src/axolotl/integrations/liger/args.py` | Liger 配置字段与字段冲突校验。 |
| `src/axolotl/integrations/liger/plugin.py` | Liger 真实 patch 入口：`pre_model_load()`。 |
| `src/axolotl/integrations/liger/models/base.py` | Axolotl 的通用 FLCE fallback forward 与 FSDP `lm_head` 处理。 |
| `src/axolotl/train.py` | 模型 / Trainer 创建、训练执行、保存主路径。 |
| `src/axolotl/core/trainers/base.py` | SFT Trainer loss 入口与保存细节。 |
| `src/axolotl/core/trainers/grpo/*` | GRPO 中另一条 Liger loss 路径，不等同于 `LigerPlugin`。 |

---

# 一、配置入口：Liger 为什么先是“插件”，而不是 Trainer 分支

## 1.1 设计哲学与核心问题

从用户视角看，Liger Kernel 是几行 YAML：

```yaml
plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_rope: true
liger_rms_norm: true
liger_glu_activation: true
liger_fused_linear_cross_entropy: true
```

这组配置看起来像是在选择一种 Trainer 或 optimizer，但源码里不是这样。Axolotl 把 Liger 做成插件，是因为它真正要改变的是**模型类和若干函数的实现**，而不是训练调度本身。

如果把 Liger 接到 Trainer 里，会太晚：Trainer 创建时模型通常已经实例化，`LlamaRMSNorm`、`LlamaMLP`、`LlamaForCausalLM.forward` 这些类/方法已经绑定进对象。Liger 要替换它们，就必须赶在模型构建之前完成 patch。

因此，这一层解决的是**初始化顺序问题**：

```text
配置读取
  -> 注册插件参数
  -> 校验/归一化配置
  -> 模型加载前执行 LigerPlugin.pre_model_load()
  -> AutoModel.from_pretrained() 实例化模型
```

第一个真正改变行为的函数不是 `load_cfg()`，也不是 `validate_config()`，而是 `LigerPlugin.pre_model_load()`。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：Click CLI 入口，把 config 与 launcher 交给 launch_training。

src/axolotl/cli/train.py
  - do_cli：读取配置，然后调用 do_train。
  - do_train：加载数据集，再调用 axolotl.train.train。

src/axolotl/cli/config.py
  - load_cfg：读取 YAML，调用 prepare_plugins、validate_config、normalize_config。
  - prepare_plugins：注册插件并调用 plugin.register(cfg)。

src/axolotl/integrations/base.py
  - PluginManager.register：import 插件类并保存到单例管理器。
  - PluginManager.pre_model_load：模型加载前统一调用插件 hook。

src/axolotl/integrations/liger/plugin.py
  - LigerPlugin.get_input_args：声明插件配置类。
  - LigerPlugin.pre_model_load：真正执行 Liger patch。
```

## 1.3 主流程拆解

用户入口从 Click 命令开始。`src/axolotl/cli/main.py:98-128` 的 `train()` 会遍历配置文件并调用 `launch_training()`；`launch_training()` 在 `src/axolotl/cli/utils/train.py:109-131` 决定使用 `accelerate`、`torchrun` 还是普通 Python 方式启动训练。默认 accelerate 路径最终执行：

```text
accelerate launch -m axolotl.cli.train <config>
```

进入训练进程后，`src/axolotl/cli/train.py:55-91` 的 `do_cli()` 调用 `load_cfg(config)`，随后 `do_train()` 在 `src/axolotl/cli/train.py:23-45` 加载数据并进入 `axolotl.train.train()`。

配置加载阶段关键调用链是：

```text
axolotl.cli.train.do_cli(config)
  -> load_cfg(config)                                  # src/axolotl/cli/config.py:230-346
    -> prepare_plugins(cfg)                            # src/axolotl/cli/config.py:208-220
      -> PluginManager.register(plugin_name)           # src/axolotl/integrations/base.py:370-383
      -> plugin.register(cfg)                          # LigerPlugin 未覆写，实际 no-op
    -> validate_config(cfg)                            # src/axolotl/utils/config/__init__.py:324-378
      -> merge_input_args()                            # src/axolotl/integrations/config.py:27-57
    -> normalize_config(cfg)                           # src/axolotl/utils/config/__init__.py:112+，其中写入 model_config_type
    -> plugin_set_cfg(cfg)                             # src/axolotl/cli/config.py:223-226
```

这里的状态变化有三层：

1. `PluginManager.plugins` 保存插件实例。`PluginManager` 是单例，`plugins` 是 `OrderedDict`，定义在 `src/axolotl/integrations/base.py:325-360`。
2. `merge_input_args()` 动态创建新的 Pydantic 配置类，把 `LigerArgs` 混入 Axolotl 基础配置。源码在 `src/axolotl/integrations/config.py:40-57`，它通过插件的 `get_input_args()` 收集字符串路径，再 `exec()` 生成新类。
3. `normalize_config()` 会加载 HF model config，并把 `model_config.model_type` 写入 `cfg.model_config_type`。对应 `src/axolotl/utils/config/__init__.py:171-202`。

真正 patch 发生在模型加载阶段。`ModelLoader.load()` 的顺序很关键：

```text
ModelLoader.load()
  -> patch_manager.apply_pre_model_load_patches()
  -> self._apply_pre_model_load_setup()
  -> PLUGIN_MANAGER.pre_model_load(self.cfg)           # Liger 在这里执行
  -> patch_manager.apply_post_plugin_pre_model_load_patches()
  -> self._build_model()                               # AutoModel.from_pretrained / from_config
```

源码位置是 `src/axolotl/loaders/model.py:161-194`。也就是说，Liger patch 发生在 `_build_model()` 之前，正好赶上 Transformers 创建模型对象。

## 1.4 关键细节与误区澄清

> 容易误解一：`plugins` 配置只是“加载扩展包”，真正行为可能在 `register()` 里发生。

对 Liger 来说不是。`LigerPlugin` 没有覆写 `register()`，它只实现了 `get_input_args()` 和 `pre_model_load()`（`src/axolotl/integrations/liger/plugin.py:14-22`）。`prepare_plugins()` 确实会调用 `plugin.register(cfg)`（`src/axolotl/cli/config.py:219-220`），但对 Liger 这是继承自 `BasePlugin` 的空方法。真正改变 Transformers 行为的是 `pre_model_load()`。

> 容易误解二：只要 schema 里出现了 `liger_*` 字段，就说明 Liger 已经生效。

不是。字段进入 schema 只说明配置能被解析和校验；要真正生效，还必须满足：

1. `plugins` 中包含 `axolotl.integrations.liger.LigerPlugin`；
2. `pre_model_load()` 被调用；
3. `cfg.model_config_type` 命中 Liger 支持分支；
4. 对应的 `liger_*` 开关为真。

如果只写 `liger_fused_linear_cross_entropy: true`，但没有插件，SFT 模型 forward 不会被 Liger patch。

## 1.5 本章小结

> 💡 **小结**
>
> * Liger 在 Axolotl 中首先是模型加载前插件，而不是 Trainer 分支。
> * 配置校验只让 `liger_*` 字段合法；真正行为从 `LigerPlugin.pre_model_load()` 开始。
> * patch 必须发生在 `_build_model()` 之前，否则已经实例化的模块不会自动变成 Liger 实现。

---

# 二、配置归一化：用户字段如何变成可执行 patch 决策

## 2.1 设计哲学与核心问题

Liger 的配置看起来很多：`liger_rope`、`liger_rms_norm`、`liger_glu_activation`、`liger_layer_norm`、`liger_cross_entropy`、`liger_fused_linear_cross_entropy`、`liger_use_token_scaling`……但源码里真正的决策并不是“看到字段就 patch”，而是先做三件事：

1. 把字段合并进全局配置 schema。
2. 处理废弃字段和互斥字段。
3. 根据模型类型和下游 Liger 支持情况选择 patch 分支。

这一层解决的是**配置歧义问题**。如果没有它，用户可以同时打开多个互斥 loss 优化，或者把只适合 FLCE 的 token scaling 用在普通 CE 上，最终要么静默错误，要么在训练中才爆炸。

## 2.2 源码入口与关键对象

```text
src/axolotl/integrations/liger/args.py
  - LigerArgs：插件配置字段定义。
  - check_deprecated_swiglu：把旧字段 liger_swiglu 迁移到 liger_glu_activation。
  - check_tiled_mlp_conflict：避免 Liger GLU 与 TiledMLP 直接冲突。
  - check_liger_rms_norm_tensor_parallel：声明 Liger RMSNorm 与 TP 不兼容。
  - check_liger_use_token_scaling_flce：token scaling 必须依赖 FLCE。

src/axolotl/utils/schemas/validation.py
  - check_cross_entropy_conflicts：CCE / chunked CE / Liger CE / Liger FLCE 互斥。
  - check_grpo_liger_sequence_parallel：GRPO + SP + Liger loss 禁用。
  - check_batch_flattening_fa：说明 GRPO Liger loss 会绕过 flattened training forward。

src/axolotl/utils/config/__init__.py
  - normalize_config：写入 cfg.model_config_type，后续 Liger patch 按它分支。
```

## 2.3 主流程拆解

`LigerArgs` 字段定义在 `src/axolotl/integrations/liger/args.py:26-55`。几个校验尤其关键：

```text
liger_swiglu -> liger_glu_activation
  src/axolotl/integrations/liger/args.py:57-71

liger_glu_activation 与 tiled_mlp 冲突
  src/axolotl/integrations/liger/args.py:73-84

liger_rms_norm 与 tensor_parallel_size > 1 冲突
  src/axolotl/integrations/liger/args.py:86-94

liger_use_token_scaling 依赖 liger_fused_linear_cross_entropy
  src/axolotl/integrations/liger/args.py:96-106
```

全局 CE 优化互斥在 `src/axolotl/utils/schemas/validation.py:974-1002`：

```text
cut_cross_entropy
chunked_cross_entropy
liger_cross_entropy
liger_fused_linear_cross_entropy
```

四者只能开一个。这一点非常重要，因为这些优化都试图控制最后 CE/loss 路径；如果同时 patch，谁先写入全局函数、谁后写入 forward，很容易出现无法解释的行为。

模型类型来自 `normalize_config()`。源码在 `src/axolotl/utils/config/__init__.py:171-202`：

```text
base_model_config ||= base_model
  -> PatchManager.apply_pre_config_load_patches(cfg)
  -> load_model_config(cfg)
  -> cfg.model_config_type = model_config.model_type
```

这使得 Liger 不直接猜 `base_model` 名字，而是使用 Transformers config 的 `model_type`。例如 Llama 走 `llama`，Qwen3 走 `qwen3`，Gemma4 走 `gemma4`。

## 2.4 关键细节与误区澄清

> 容易误解三：`liger_swiglu` 和 `liger_glu_activation` 是两个不同优化。

不是。`liger_swiglu` 是废弃字段。`check_deprecated_swiglu()` 在 `args.py:57-71` 中将它迁移为 `liger_glu_activation`；如果两者同时出现，会直接报错。

> 容易误解四：`liger_use_token_scaling` 可以独立开启。

不能。`args.py:96-106` 明确要求它必须和 `liger_fused_linear_cross_entropy` 同时开启。原因从后面的 Liger kernel 可见：token scaling 作用在 FLCE 的 chunked logits/loss 路径上，而不是普通 `cross_entropy`。

> 容易误解五：README 里的支持模型列表就是最终真相。

要以源码为准。`src/axolotl/integrations/liger/README.md:26-45` 列了一批支持模型；但实际运行时首先看下游 `liger_kernel.transformers.monkey_patch.MODEL_TYPE_TO_APPLY_LIGER_FN`，Axolotl 自己还有 `jamba`、`deepseek_v2`、`qwen3_5`、`qwen3_5_moe`、`gemma4` 等自定义分支（`plugin.py:106-293`）。因此 README 更像使用简介，不是完整 runtime 路由表。

## 2.5 本章小结

> 💡 **小结**
>
> * Liger 配置的关键不是字段数量，而是互斥关系与模型类型分派。
> * `cfg.model_config_type` 是 Liger patch 路由的核心状态，由 `normalize_config()` 写入。
> * 一些字段存在但只在特定分支生效；例如 `liger_use_token_scaling` 只服务 FLCE。

---

# 三、Monkey Patch 注入：零侵入接入，还是全局状态风险

## 3.1 设计哲学与核心问题

Liger 想优化的对象大多藏在 Transformers 模型内部：`RMSNorm` 类、`MLP` 类、`apply_rotary_pos_emb()`、`ForCausalLM.forward()`、`nn.functional.cross_entropy()`。Axolotl 不可能为每个模型复制一整份模型定义，所以采用 monkey patch：在模型实例化前改写 Transformers 模块命名空间。

这带来的好处是接入成本低：原来的 `AutoModelForCausalLM.from_pretrained()` 不需要知道 Liger 存在，实例化时自然会拿到被替换后的类/函数。

代价是全局状态风险：这些替换不是局部 context manager，不会在训练结束自动恢复。

## 3.2 源码入口与关键对象

```text
src/axolotl/integrations/liger/plugin.py
  - pre_model_load：总 patch 入口。
  - torch_compile 分支：禁用 Liger FLCE forward/backward 的 torch.compile。
  - token scaling 分支：包裹 Liger FLCE 函数和 Loss 类 __init__。
  - MODEL_TYPE_TO_APPLY_LIGER_FN 分支：委托下游 liger-kernel。
  - jamba/deepseek_v2/qwen3_5/qwen3_5_moe/gemma4 等 Axolotl 自定义分支。

src/axolotl/integrations/liger/utils.py
  - patch_with_compile_disable：用 torch.compiler.disable 包装函数。

/usr/local/lib/python3.12/dist-packages/liger_kernel/transformers/monkey_patch.py
  - MODEL_TYPE_TO_APPLY_LIGER_FN：下游 Liger 支持模型路由。
  - apply_liger_kernel_to_llama 等：替换 Transformers 模块中的类/函数。
```

## 3.3 主流程拆解

`LigerPlugin.pre_model_load()` 的结构可以简化成：

```text
pre_model_load(cfg)
  -> 给 trl.trainer.ORPOTrainer 做兼容 shim
  -> 如果 torch_compile: 给 Liger FLCE forward/backward 加 torch.compiler.disable
  -> 导入 Liger Kernel 的替换类/函数
  -> 检查 cross_entropy 与 fused_linear_cross_entropy 互斥
  -> 如果 liger_use_token_scaling: patch Liger FLCE 函数与 Loss.__init__
  -> 如果 cfg.model_config_type 命中下游 MODEL_TYPE_TO_APPLY_LIGER_FN:
       inspect signature，只传支持的 kwargs
       apply_liger_fn(**kwargs)
     否则走 Axolotl 自定义模型分支
```

对应源码是 `src/axolotl/integrations/liger/plugin.py:22-293`。

下游委托分支在 `plugin.py:84-105`：

```text
if cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN:
    apply_liger_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[cfg.model_config_type]
    liger_fn_sig = inspect.signature(apply_liger_fn)
    kwargs = {}
    ... 只放入 apply 函数支持的参数 ...
    apply_liger_fn(**kwargs)
```

这段代码有一个工程上很好的细节：它不是盲目传所有 `liger_*` 参数，而是先 inspect 下游函数签名。这样不同模型支持的开关不一致时，Axolotl 不会因为多传参数而崩。

以 Llama 为例，下游 `liger_kernel` 的 `apply_liger_kernel_to_llama()` 会改写 Transformers 模块：

```text
if rope:      modeling_llama.apply_rotary_pos_emb = liger_rotary_pos_emb
if rms_norm:  modeling_llama.LlamaRMSNorm = LigerRMSNorm
if swiglu:    modeling_llama.LlamaMLP = LigerSwiGLUMLP
if cross_entropy:
    transformers.loss.loss_utils.nn.functional.cross_entropy = liger_cross_entropy
if fused_linear_cross_entropy:
    modeling_llama.LlamaForCausalLM.forward = llama_lce_forward
```

源码见 `/usr/local/lib/python3.12/dist-packages/liger_kernel/transformers/monkey_patch.py:217-265`。注意这里替换的是模块级名称；后续新建的 Llama 模型会使用这些新定义。

Axolotl 自己也有特殊分支。例如 `deepseek_v2` 需要先用 `init_empty_weights()` 加载远端模型类，从 `sys.modules` 找到实际 module，再改写对应类（`plugin.py:125-152`）。`gemma4` 分支则明确跳过 RoPE 和 FLCE，并只 patch RMSNorm/GEGLU/LayerNorm（`plugin.py:225-274`）。

## 3.4 关键细节与误区澄清

> 容易误解六：`src/axolotl/integrations/liger/models/qwen3.py`、`llama4.py` 一定是当前 qwen3/llama4 的主路径。

不一定，甚至在当前依赖版本下通常不是。Axolotl 的 `pyproject.toml:79-84` 固定 `liger-kernel==0.7.0`；该包的 `MODEL_TYPE_TO_APPLY_LIGER_FN` 已经包含 `qwen3`、`qwen3_moe`、`llama4`、`llama4_text` 等 key（见 `liger_kernel/.../monkey_patch.py:2879-2921`）。而 Axolotl 的 `plugin.py:84-105` 会先命中下游 map，只有没命中才进入后面的 `elif cfg.model_config_type == "qwen3"` 或 `"llama4"`。因此这些 Axolotl 自定义文件更像兼容/备用路径，不能看到文件名就认定它是标准主流程。

> 容易误解七：patch 是局部生效的。

不是。多数 patch 是写入 `transformers.models.xxx.modeling_xxx` 或 `torch.nn.functional` 级别的全局模块对象。例如 `nn.functional.cross_entropy = liger_cross_entropy` 出现在下游 Llama patch 的 `monkey_patch.py:255-258`，Axolotl 自定义 qwen3 分支也在 `src/axolotl/integrations/liger/models/qwen3.py:152-155` 做同类替换。源码里没有 `with`、没有 `__exit__`、也没有统一 restore。

> 容易误解八：`torch_compile` 开启后 Liger kernel 会被一起编译优化。

Axolotl 反而在 `cfg.torch_compile` 时禁用 Liger FLCE 两个底层函数的 compile：`plugin.py:29-42` 导入 `liger_kernel.ops.fused_linear_cross_entropy`，再用 `patch_with_compile_disable()` 包住 `fused_linear_cross_entropy_forward` 和 `fused_linear_cross_entropy_backward`。工具函数在 `src/axolotl/integrations/liger/utils.py:10-29`，核心是 `@torch.compiler.disable`。注释解释了原因：torch.compile 会不必要地尝试优化 Triton kernel。

## 3.5 本章小结

> 💡 **小结**
>
> * Liger 的接入点是模型实例化前的全局 monkey patch。
> * 下游 `MODEL_TYPE_TO_APPLY_LIGER_FN` 优先于 Axolotl 自定义分支，导致部分本地文件不是当前主路径。
> * patch 没有统一恢复机制；这是低侵入接入的代价，也是测试和多模型进程复用的维护风险。

---

# 四、前向主路径：FLCE 如何省掉 `[B, S, V]` logits

## 4.1 设计哲学与核心问题

Liger 中对显存最直接的优化是 `liger_fused_linear_cross_entropy`。普通 causal LM 训练大致是：

```text
hidden_states: [B, S, H]
  -> lm_head
logits: [B, S, V]
  -> shift + cross_entropy
loss: scalar
```

当 `V` 是 32k、128k 甚至更大时，`[B, S, V]` 是非常大的中间张量。长序列训练时，即使参数已经通过 FSDP 切分，logits 仍然可能成为激活显存的大头。

FLCE 的思路是把最后线性层和 CE 融合：输入仍然是 hidden states 和 `lm_head.weight`，但不把完整 logits 留在 Python/PyTorch 图里，而是分 chunk 计算 loss 与梯度。

## 4.2 源码入口与关键对象

```text
src/axolotl/integrations/liger/models/base.py
  - lce_forward：通用 fallback forward，支持 skip_logits、shift_labels、FSDP lm_head。
  - lce_maybe_trainable_lm_head：处理 PEFT 与 FSDP 包裹的 lm_head。
  - patch_lce_forward：动态替换某个模型的 ForCausalLM.forward。

src/axolotl/integrations/liger/models/qwen3_5.py
  - lce_forward：Qwen3.5 专用 forward，训练时返回 loss，logits=None。

/usr/local/lib/python3.12/dist-packages/liger_kernel/transformers/model/loss_utils.py
  - LigerForCausalLMLoss：shift labels、flatten hidden states，调用 FLCE。

/usr/local/lib/python3.12/dist-packages/liger_kernel/ops/fused_linear_cross_entropy.py
  - fused_linear_cross_entropy_forward：chunked logits、in-place CE 梯度。
  - LigerFusedLinearCrossEntropyFunction：自定义 autograd Function。
```

## 4.3 主流程拆解

SFT 主训练中，Axolotl 自己的 `AxolotlTrainer.compute_loss()` 大部分情况下会回落到 HuggingFace Trainer 默认实现：`src/axolotl/core/trainers/base.py:455-460`。也就是 Trainer 最终调用 `model(**inputs)`，并把 `labels` 传给模型。

当模型 forward 已经被 Liger patch 后，训练路径变成：

```text
Trainer.compute_loss
  -> model(input_ids, attention_mask, labels, ...)
    -> patched ForCausalLM.forward
      -> self.model(...)                         # 只跑 backbone，得到 hidden_states
      -> LigerForCausalLMLoss(hidden_states, lm_head.weight, labels, hidden_size)
        -> hidden_states.view(-1, H)
        -> shift_labels.view(-1)
        -> liger_fused_linear_cross_entropy(...)
          -> 按 chunk 计算 logits_chunk: [chunk, V]
          -> Triton CE kernel 原地写 loss 与 grad_logits_chunk
          -> grad_input: [B*S, H]
          -> grad_weight: [V, H]（如果 lm_head 可训练）
```

以 Axolotl 的通用 fallback 为例，`lce_forward()` 在 `src/axolotl/integrations/liger/models/base.py:59-118` 做了两件事：

1. 先调用 `self.model(...)` 得到 `hidden_states`（`base.py:59-67`）。
2. 如果 `skip_logits` 为真，走 `lce_maybe_trainable_lm_head()`，返回 loss 且不 materialize logits（`base.py:80-97`）。否则才执行 `self.lm_head()` 得到 logits（`base.py:98-106`）。

Qwen3.5 专用实现更直接：训练且有 labels 时，直接调用 `LigerForCausalLMLoss()`，否则才切片并计算 logits。对应 `src/axolotl/integrations/liger/models/qwen3_5.py:57-85`。

下游 Liger loss 的 shape 变化在 `loss_utils.py:64-101`：

```text
labels = pad(labels, (0, 1), ignore_index)
shift_labels = labels[..., 1:].contiguous()
hidden_states = hidden_states.view(-1, hidden_size)
shift_labels = shift_labels.view(-1)
fixed_fused_linear_cross_entropy(hidden_states, lm_head_weight, shift_labels, ...)
```

底层 FLCE 在 `fused_linear_cross_entropy.py:41-55` 明确写了注释：输入是 `BT x H`，物化 logits 会变成 `BT x V`，所以用 chunk 限制一次性 logits 的大小。实际循环在 `fused_linear_cross_entropy.py:91-207`：每个 chunk 做 `_input_chunk @ weight.t()` 得到 `[chunk, V]`，随后 Triton CE kernel 计算 loss 和梯度，并把 `grad_input`、`grad_weight` 累积出来。

## 4.4 Tensor shape 流程

```text
原始 batch:
  input_ids: [B, S]
  labels:    [B, S]

Backbone forward:
  hidden_states: [B, S, H]

LigerForCausalLMLoss:
  labels pad + shift:
    shift_labels: [B, S] -> [B*S]
  hidden flatten:
    hidden_states: [B, S, H] -> [B*S, H]

FLCE chunk loop:
  _input_chunk:  [C, H]
  weight.t():    [H, V]
  logits_chunk:  [C, V]       # 只在 chunk 内短暂存在
  loss_1d_slice: [C]
  grad_input:    [B*S, H]
  grad_weight:   [V, H]       # 如果 lm_head.weight.requires_grad

返回:
  loss: scalar
  logits: None（训练 + FLCE 主路径）
```

显存收益来自：完整 `[B*S, V]` logits 不再作为持久激活留在 forward 输出与 autograd 图中。它不是完全不算 logits，而是一次只算 `[C, V]` chunk，并尽快转成 loss 与梯度。

## 4.5 关键细节与误区澄清

> 容易误解九：FLCE 完全不产生 logits。

不准确。底层仍然在 chunk 内计算 `logits_chunk = _input_chunk @ weight.t()`，源码是 `fused_linear_cross_entropy.py:91-99`。省掉的是完整 `[B*S, V]` logits 的长期物化，而不是数学上的 logits 计算。

> 容易误解十：开启 FLCE 后训练和推理都不返回 logits。

不对。Axolotl 的 patch forward 通常只在 `self.training and labels is not None` 时走 fused loss；否则仍然会计算 logits。例如 Qwen3.5 分支在 `qwen3_5.py:61-85` 明确区分训练 loss 路径和 inference/materialize logits 路径。

> 容易误解十一：FLCE 只影响 loss，不影响梯度语义。

它确实保持了最后线性层 + CE 的数学目标，但实现上把梯度提前在 forward 中准备好。`LigerFusedLinearCrossEntropyFunction.forward()` 在 `fused_linear_cross_entropy.py:324-348` 保存的是 `grad_input`、`grad_weight`、`grad_bias`；`backward()` 在 `fused_linear_cross_entropy.py:350-376` 再按上游 `grad_output` 缩放返回。这是自定义 autograd 语义，不是普通 PyTorch CE 的组合图。

## 4.6 本章小结

> 💡 **小结**
>
> * Liger FLCE 的主收益是避免完整 `[B, S, V]` logits 成为持久激活。
> * 它仍然分 chunk 计算 `[C, V]` logits，因此计算量没有凭空消失。
> * 训练路径通常返回 `loss` 且 `logits=None`；评估/推理仍可能 materialize logits。

---

# 五、FSDP、通信与保存：Liger 不建通信组，但必须尊重分布式上下文

## 5.1 设计哲学与核心问题

Liger Kernel 本身不是并行策略。它不创建 DeviceMesh，不切分 batch，不做 all-to-all，也不替代 FSDP/DeepSpeed。它的分布式挑战主要在两个点：

1. **训练 forward 中读取 `lm_head.weight`**：如果 `lm_head` 被 FSDP 包裹，直接读权重可能拿到 sharded/flat 参数，必须进入 FSDP 正确的 forward redirection。
2. **保存时恢复权重**：Liger patch 改的是类/函数，不是 state_dict 格式；保存仍然走 Trainer/FSDP/Accelerate 的原路径。

因此这一层解决的是**兼容性问题**，不是新通信算法问题。

## 5.2 源码入口与关键对象

```text
src/axolotl/integrations/liger/models/base.py
  - lce_maybe_trainable_lm_head：如果 lm_head 是 FSDP，使用 _FSDPForwardRedirection。

src/axolotl/train.py
  - execute_training：训练期间进入 Trainer.train。
  - save_trained_model：训练结束保存，按 FSDP/DeepSpeed/普通 rank0 分支处理。

src/axolotl/core/trainers/base.py
  - _save：保存 state_dict，context parallel 下会 CPU clone；不是 Liger 专属。

src/axolotl/core/trainers/mixins/distributed_parallel.py
  - _save：dp_shard_enabled 时通过 accelerator.get_state_dict 收集。
```

## 5.3 主流程拆解

通用 FLCE fallback 对 FSDP 的处理在 `src/axolotl/integrations/liger/models/base.py:121-169`。逻辑是：

```text
lm_head = self.lm_head
if PEFT ModulesToSaveWrapper:
    lm_head = lm_head.modules_to_save.default

if isinstance(lm_head, FullyShardedDataParallel):
    return _FSDPForwardRedirection()(lm_head, _liger_for_causal_lm_loss, lm_head.module, ...)
else:
    return _liger_for_causal_lm_loss(lm_head=self.lm_head, ...)
```

这里的核心是 `base.py:133-146` 的注释：如果 FSDP 被使用且 `lm_head` 可训练，读取 `lm_head` 权重和调用 kernel 必须发生在 FSDP forward pass 内，这样完整参数才会被 summon 并在 kernel 执行期间保留。

训练执行本身没有 Liger 专属 context。`src/axolotl/train.py:183-230` 的 `execute_training()` 只在 flash optimum、context parallel 等配置下进入额外 context，最终调用 `trainer.train()`（`train.py:226-227`）。Liger 已经在模型加载前完成 patch，因此训练 step 不需要每次进入 Liger manager。

保存阶段也没有 Liger 专属保存器。`save_trained_model()` 在 `src/axolotl/train.py:254-380` 根据 FSDP、DeepSpeed、普通 rank0 分支保存：

```text
FSDP:
  trainer.accelerator.state.fsdp_plugin.set_state_dict_type(...)
  trainer.save_model(output_dir)
  如 SHARDED_STATE_DICT，可能 merge_sharded_fsdp_weights

DeepSpeed ZeRO-3:
  wait_for_everyone()
  trainer.save_model(output_dir)
  删除可能的 proxy model.safetensors

普通:
  local_rank == 0 时 model.save_pretrained(output_dir)
```

这说明 Liger patch 不改变 checkpoint 格式。保存的是参数，而不是“Liger 化后的模型源码”。重新加载时若想继续用 Liger，仍需配置插件并再次执行 patch。

## 5.4 Rank / 通信流程

以普通 SFT + FSDP + Liger FLCE 为例：

```text
每个训练 step:
  rank 内部:
    input_ids / labels -> model forward
    backbone 层参数由 FSDP 按自身策略 all-gather / reshard
    hidden_states 到 lm_head/loss

  Liger 自身:
    不创建 process group
    不 all_to_all
    不 reduce_scatter
    不 broadcast

  FSDP 可能触发:
    lm_head 参数 all-gather（如果 lm_head 被 FSDP 包裹且需要参与 forward）
    反向梯度 reduce/scatter（FSDP 原生语义）
```

因此，“Liger 的通信开销”准确说是：Liger 不主动增加分布式通信原语；它可能改变最后 loss 路径中权重访问时机，进而要求 FSDP 在正确上下文中提供完整 `lm_head.weight`。

## 5.5 关键细节与误区澄清

> 容易误解十二：Liger 会像 sequence parallel 一样切分序列维度。

不会。Axolotl 的 sequence parallel 在 `SequenceParallelContextManager`，进入条件是 `cfg.context_parallel_size > 1`，源码在 `src/axolotl/train.py:205-220`。LigerPlugin 没有 DeviceMesh、process group、dispatch/collect 逻辑。

> 容易误解十三：Liger patch 会影响模型保存格式。

不会直接影响。保存路径仍然是 `trainer.save_model()` / `model.save_pretrained()`。Liger 替换的是运行期 Python 类/函数，state_dict 仍是参数名和张量。输出目录不会自动记录“下次加载必须启用 Liger”。

> 容易误解十四：`src/axolotl/core/trainers/base.py:812-823` 的 CPU clone 是 Liger 保存修复。

不是。这段注释明确是 “Context Parallel save: CP eval invalidates tensor storage pointers”，触发条件是 `context_parallel_size > 1`。它与 Liger 没有直接关系，只是同样位于 Trainer 保存路径，容易被混在一起看。

## 5.6 本章小结

> 💡 **小结**
>
> * Liger 不新增通信组；FSDP/DeepSpeed 仍负责分布式参数和梯度语义。
> * Liger 与 FSDP 的交界点主要是 `lm_head.weight` 的读取上下文。
> * 保存路径不保存 patch 状态；重新加载要重新通过插件 patch。

---

# 六、GRPO / DPO 的另一条 Liger 路：不是 `LigerPlugin` 的主路径

## 6.1 设计哲学与核心问题

Axolotl 里还有另一类“Liger”字段：

```yaml
trl:
  use_liger_loss: true
```

以及：

```yaml
dpo_use_liger_kernel: true
```

它们容易和 `plugins: [LigerPlugin]` 混淆。实际上，这两类字段主要是把参数传给 TRL 的 DPO/GRPO trainer，让 RL loss 路径使用 Liger kernel。它们不等同于 SFT 场景下 `LigerPlugin.pre_model_load()` 对模型类的 patch。

这一层解决的是**RL loss 路径的显存问题**，而不是模型初始化前的 Transformers module 替换问题。

## 6.2 源码入口与关键对象

```text
src/axolotl/utils/schemas/config.py
  - dpo_use_liger_kernel：DPO loss Liger 开关。

src/axolotl/utils/schemas/trl.py
  - use_liger_loss：GRPO Liger loss 开关。

src/axolotl/core/trainers/dpo/__init__.py
  - DPOStrategy.set_training_args_kwargs：把 dpo_use_liger_kernel 转成 use_liger_kernel。

src/axolotl/core/trainers/grpo/__init__.py
  - GRPOStrategy.set_training_args_kwargs：把 trl.use_liger_loss 转成 use_liger_kernel。

src/axolotl/core/trainers/grpo/async_trainer.py
  - scoring path 使用 fused_selective_log_softmax。

src/axolotl/core/trainers/grpo/fast_async_trainer.py
  - compute_liger_loss：OPSM + zero-advantage skip 的自定义 Liger loss 处理。
```

## 6.3 主流程拆解

DPO 开关定义在 `src/axolotl/utils/schemas/config.py:305-308`，真正传给 trainer args 的位置是 `src/axolotl/core/trainers/dpo/__init__.py:37-43`：

```text
if cfg.dpo_use_liger_kernel is not None:
    training_args_kwargs["use_liger_kernel"] = cfg.dpo_use_liger_kernel
```

GRPO 开关定义在 `src/axolotl/utils/schemas/trl.py:174-177`，映射位置是 `src/axolotl/core/trainers/grpo/__init__.py:146-147`：

```text
if trl.use_liger_loss is not None:
    grpo_args_kwargs["use_liger_kernel"] = trl.use_liger_loss
```

GRPO async trainer 里还有一个 scoring path 优化：`src/axolotl/core/trainers/grpo/async_trainer.py:85-90` 尝试导入 `fused_selective_log_softmax`；在 no-grad scoring 路径中，如果 `self.use_liger_kernel` 为真且导入成功，就使用 fused kernel（`async_trainer.py:2907-2990`）。它融合的是 temperature、log_softmax 和 gather，不是替换 `ForCausalLM.forward()`。

Fast async GRPO 还覆写了 `compute_liger_loss()`，在 OPSM 需要 per-token logprobs 时先用 chunked `lm_head` matmul 计算 KL 相关量，再把修改后的 mask 交给 Liger fused kernel。源码在 `src/axolotl/core/trainers/grpo/fast_async_trainer.py:637-751`。

## 6.4 关键细节与误区澄清

> 容易误解十五：`trl.use_liger_loss` 会自动启用 `LigerPlugin`。

不会。`trl.use_liger_loss` 是 GRPO training args 层面的字段映射；源码路径没有调用 `PluginManager.pre_model_load()`。如果你想在 SFT/模型 forward 层替换 RMSNorm/MLP/FLCE，仍需配置 `plugins: [axolotl.integrations.liger.LigerPlugin]`。

> 容易误解十六：GRPO + Liger 与 batch flattening 完全兼容。

校验代码专门提醒它们不是同一条 forward 路径。`src/axolotl/utils/schemas/validation.py:951-960` 的注释和 warning 说明：`use_liger_loss` 会走 `compute_liger_loss` 单独路径，绕过 flattened training forward；batch flattening 只会影响 scoring/deferred logprobs 路径。

> 容易误解十七：GRPO + sequence parallel + Liger loss 已经可用。

校验直接禁止：`src/axolotl/utils/schemas/validation.py:721-728` 在 `rl == "grpo"`、`trl.use_liger_loss` 且 `context_parallel_size > 1` 时抛出 `ValueError("GRPO + SP + Liger not currently supported")`。

## 6.5 本章小结

> 💡 **小结**
>
> * SFT 的 LigerPlugin patch 和 RL 的 `use_liger_kernel/use_liger_loss` 是两条不同路径。
> * GRPO 中 Liger 不只是训练 loss，还用于 no-grad scoring 的 selective log-softmax。
> * 源码明确禁止 GRPO + sequence parallel + Liger loss 的组合。

---

# 七、完整主路径串联

## 7.0 设计哲学与核心问题

前面几章分别讲了配置、patch、forward、FSDP 和 RL 分支，但读源码最容易迷失的地方，恰恰是把这些机制误认为并列入口。本章解决的是**主路径还原问题**：把一次真实 `axolotl train` 调用从 CLI 串到 loss 和保存，明确哪些步骤只发生一次，哪些步骤每个 step 都执行。

## 7.1 完整调用栈

下面以一个典型配置为例：

```yaml
base_model: NousResearch/Meta-Llama-3.1-8B
plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_rope: true
liger_rms_norm: true
liger_glu_activation: true
liger_fused_linear_cross_entropy: true
```

完整主路径可以串成：

```text
User: axolotl train examples/llama-3/fft-8b-liger-fsdp.yaml
  │
  ├─ Step 1: CLI launcher
  │     ├─ src/axolotl/cli/main.py:98-128
  │     └─ src/axolotl/cli/utils/train.py:157-192
  │
  ├─ Step 2: 配置加载与插件注册
  │     ├─ src/axolotl/cli/train.py:55-91
  │     ├─ src/axolotl/cli/config.py:230-346
  │     ├─ src/axolotl/cli/config.py:208-220
  │     └─ src/axolotl/integrations/base.py:370-383
  │
  ├─ Step 3: 动态 schema + 配置归一化
  │     ├─ src/axolotl/integrations/config.py:27-57
  │     ├─ src/axolotl/integrations/liger/args.py:26-113
  │     └─ src/axolotl/utils/config/__init__.py:171-202
  │
  ├─ Step 4: 模型加载前 patch
  │     ├─ src/axolotl/train.py:54-84
  │     ├─ src/axolotl/loaders/model.py:161-194
  │     └─ src/axolotl/integrations/liger/plugin.py:22-293
  │
  ├─ Step 5: 模型实例化
  │     ├─ src/axolotl/loaders/model.py:745-862
  │     └─ AutoModelForCausalLM.from_pretrained / from_config
  │
  ├─ Step 6: Trainer.train()
  │     ├─ src/axolotl/train.py:183-230
  │     ├─ src/axolotl/core/trainers/base.py:366-460
  │     └─ patched ForCausalLM.forward -> LigerForCausalLMLoss
  │
  └─ Step 7: 保存
        ├─ src/axolotl/train.py:254-380
        └─ src/axolotl/core/trainers/base.py:806-875
```

## 7.2 每一层做了什么

| 层 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| CLI launcher | config 路径、CLI 参数 | 启动训练进程 | launcher 负责 | 无直接影响 | 一次 |
| load_cfg | YAML dict | `DictDefault`，注册插件 | 无 | 无直接影响 | 每个进程一次 |
| merge_input_args | 插件声明的 args 类 | 动态 Pydantic class | 无 | 无 | 每次校验 |
| normalize_config | HF model config | 写入 `cfg.model_config_type` | 可能访问 HF/local config | 无直接影响 | 一次 |
| pre_model_load | `cfg` | 改写 Transformers / Liger module 全局状态 | 无 | 决定后续省显存路径 | 每次模型 load |
| _build_model | patched module 命名空间 | 模型对象 | FSDP/DS 初始化可能参与 | 被 patch 类/forward 固化进实例 | 每个模型一次 |
| forward/loss | batch tensors | loss，通常 `logits=None` | FSDP 原生通信 | FLCE 减少 logits 激活 | 每 step |
| save | model/trainer state | state_dict/checkpoint | FSDP/DS save 通信 | 与 Liger 无直接关系 | checkpoint/end |

## 7.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `LigerPlugin.register()` | `prepare_plugins()` 会调用 `plugin.register(cfg)` | 对 Liger 否 | Liger 没覆写 `register()`；真正入口是 `pre_model_load()`。 |
| `src/axolotl/integrations/liger/models/qwen3.py` | 文件名像 qwen3 主实现 | 当前依赖下通常否 | `liger-kernel==0.7.0` map 已含 `qwen3`，先命中下游分支。 |
| `src/axolotl/integrations/liger/models/llama4.py` | 文件名像 Llama4 主实现 | 当前依赖下通常否 | 下游 map 含 `llama4`/`llama4_text`，先被委托。 |
| `src/axolotl/kernels/gemma4_fused_rope.py` | 也是 Triton fused kernel | 不是 LigerPlugin 主路径 | 由 Gemma4 fused attention patch 使用；不由 `LigerPlugin` 直接启用。 |
| `context_parallel_size` / `SequenceParallelContextManager` | 也与显存相关 | 不是 Liger 主路径 | SP 另有 dispatch/collect/attention patch，Liger 不建通信组。 |
| `dpo_use_liger_kernel` / `trl.use_liger_loss` | 名字含 Liger | 不是 SFT LigerPlugin 路径 | 传给 TRL/RL trainer args，属于 RL loss 路径。 |

## 7.4 本章小结

> 💡 **小结**
>
> * 一次真实训练里，Liger 的关键时刻只有一个：模型实例化前。
> * 每 step 执行的是被 patch 后的 forward/loss，而不是每 step 重新 patch。
> * 很多 Liger-adjacent 文件属于备用、测试或其他优化路径，不能从文件名推主流程。

---

# 八、关键数据流、状态流与 shape 流程

## 8.0 设计哲学与核心问题

Liger 的价值不是“文件被 patch 了”这个事实，而是 patch 后张量、状态和分布式边界发生了什么变化。本章解决的是**运行时可见效果问题**：哪些 shape 被缩小，哪些状态被全局改写，哪些 rank 逻辑其实没有被 Liger 触碰。

## 8.1 Tensor shape 变化

以 SFT + FLCE 为主路径：

```text
输入 batch:
  input_ids:      [B, S]
  attention_mask: [B, S]
  labels:         [B, S]

模型 backbone:
  hidden_states:  [B, S, H]

Liger loss:
  hidden_states.view(-1, H): [B*S, H]
  shift_labels.view(-1):     [B*S]

chunked FLCE:
  chunk hidden:   [C, H]
  lm_head.weight: [V, H]
  logits_chunk:   [C, V]
  loss slice:     [C]

反向保存:
  grad_input:     [B*S, H]
  grad_weight:    [V, H]（如果 weight requires_grad）
```

为什么这样能省显存？因为普通路径可能需要在 forward 输出、loss 计算和 autograd 图中持有 `[B*S, V]`，而 Liger 把它限制在 chunk 内短生命周期的 `[C, V]`。

性能瓶颈也很清晰：当 `V` 很大，`_input_chunk @ weight.t()` 仍然是大矩阵乘；FLCE 减少显存峰值和中间写回，但没有省掉最后线性层的核心 FLOPs。

## 8.2 状态切换

Liger 不是 context manager，而是全局 patch：

```text
进入 pre_model_load:
  读取 cfg.model_config_type
  读取 cfg.liger_* 开关
  替换 transformers / liger_kernel 模块中的函数或类

执行中:
  AutoModel.from_pretrained 创建模型
  模型 forward 通过模块级类/方法引用进入 Liger 实现

退出后:
  没有恢复逻辑
  当前 Python 进程内 patch 继续存在
```

状态定义位置：

* 插件状态：`PluginManager.plugins` 与 `PluginManager._cfg`，见 `src/axolotl/integrations/base.py:339-368`。
* 配置状态：`cfg.model_config_type`，见 `src/axolotl/utils/config/__init__.py:201`。
* patch 状态：写入 Transformers/Liger module 全局对象，例如 `modeling_llama.LlamaForCausalLM.forward`。

线程/进程安全角度看：多进程训练每个进程各自 import 和 patch，进程间不共享 Python module 状态；但同一进程内连续加载不同模型时，patch 可能残留。测试里 `tests/conftest.py:477-539` 会重置 `PluginManager` 并 reload 一批已知模块，但这个清理列表主要覆盖 Llama、Trainer、loss_utils，并不一定覆盖所有模型分支。

## 8.3 Rank / Mesh / Process Group

LigerPlugin 没有自己的 rank mapping。若 world size = 8 且使用 FSDP，逻辑更接近：

```text
Rank 0..7:
  每个 rank 拿自己的 data parallel micro-batch
  FSDP 控制参数 shard / all-gather / reduce-scatter
  Liger 控制 rank 本地 loss kernel 的计算方式
```

也就是说：

* 每个 rank 的输入分发由 Trainer/Accelerate/FSDP/数据 sampler 决定。
* Liger 不让同组 rank 共享同一份输入。
* Liger 不切换 process group。
* 若 `lm_head` 被 FSDP 包裹，相关 all-gather 属于 FSDP forward 上下文，不是 Liger 自己发起的通信原语。

## 8.4 本章小结

> 💡 **小结**
>
> * Liger 的 shape 主线是 `[B,S,H] -> [B*S,H] -> chunk [C,V] -> scalar loss`。
> * Liger 状态是全局 module patch，不是局部上下文。
> * Liger 不定义 rank/mesh；所有分布式通信仍由外层训练框架负责。

---

# 九、核心机制深挖

## 9.0 设计哲学与核心问题

如果只看调用链，Liger 像是几次简单赋值；如果只看效果，又像是“显存变少”。真正值得深挖的是这些赋值背后的工程假设。本章解决的是**机制代价问题**：为什么要 monkey patch，为什么 FLCE 能省显存，以及配置为什么要经过多层门控。

## 9.1 Monkey Patch：零侵入接入还是维护风险？

### 它解决什么问题

Liger 要替换的是模型内部类和 forward。通过 pre-model-load monkey patch，Axolotl 无需 fork Transformers 模型源码，也无需写一套专门的模型工厂。

### 为什么不能更简单

如果只在模型实例化后替换 `model.forward`，可以覆盖 FLCE，但 RMSNorm/MLP/RoPE 等已经嵌入每层 module，不会自动变成 Liger 实现。下游 Liger 的 `_apply_liger_kernel()` 注释也指出：在模型初始化后调用无法完全 patch 模型（`liger_kernel/.../monkey_patch.py:2924-2956`，尤其 `2930-2932`）。

### 源码实现

Axolotl 先在 `ModelLoader.load()` 调 `PLUGIN_MANAGER.pre_model_load()`（`src/axolotl/loaders/model.py:168-176`），随后才 `_build_model()`。`LigerPlugin.pre_model_load()` 根据模型类型替换 module 级对象。

### 隐藏假设

* Transformers 模型内部类名和 module 路径稳定。
* 下游 Liger 的 apply 函数签名可以通过 `inspect.signature()` 正确过滤参数。
* 用户不会在同一进程里依赖“未 patch 的同类模型”与“已 patch 的同类模型”严格隔离。

### 副作用和维护风险

* patch 无统一恢复。
* 下游 Transformers/Liger 版本升级可能导致类名或 forward 签名变化。
* 多次 `pre_model_load()` 可能重复包裹 token scaling patch，因为 `plugin.py:57-82` 没有幂等 guard。

## 9.2 FLCE 反向：前向里先算梯度，为什么合理？

### 它解决什么问题

普通 autograd 会保存足够多中间状态用于 backward，而 FLCE 的目标是避免保存完整 logits。它选择在 forward chunk 内直接计算 CE 以及对 `_input`、`weight`、`bias` 的梯度，再在 backward 中按上游标量梯度缩放。

### 源码实现

下游 `LigerFusedLinearCrossEntropyFunction.forward()` 在 `fused_linear_cross_entropy.py:324-348` 调用 `fused_linear_cross_entropy_forward()`，返回 loss 和梯度缓存，并 `ctx.save_for_backward(grad_input, grad_weight, grad_bias)`。`backward()` 在 `350-376` 调用 `fused_linear_cross_entropy_backward()`。

### 隐藏假设

* CE 是最后 loss 层，通常上游 `grad_output` 是 1。源码还对非 1 的 `grad_output` 做额外 scaling（`fused_linear_cross_entropy.py:232-276`）。
* `target`、`weight`、`hidden_states` shape 符合 causal LM loss 预期：`_input: [B*T,H]`，`target: [B*T]`，`weight: [V,H]`（`fused_linear_cross_entropy.py:299-311`）。

### 副作用

* 如果 `return_token_accuracy`、`use_token_scaling` 等开关打开，会多出额外计算或临时张量。
* token scaling 分支会对 logits 做 `detach().clone()` 和 softmax（`fused_linear_cross_entropy.py:105-134`），这是为了概率缩放，但会增加 chunk 内临时显存与计算。

## 9.3 配置归一化：字段如何改变源码路径

`liger_fused_linear_cross_entropy` 的路径不是单一开关，而是经过：

```text
YAML 字段
  -> LigerArgs 校验
  -> 全局 CE 互斥校验
  -> cfg.model_config_type 路由
  -> 对应 apply_liger_kernel_to_* 支持 fused_linear_cross_entropy 参数才传入
  -> forward 被替换
```

`plugin.py:88-104` 的参数过滤尤其关键：如果下游 apply 函数签名没有 `fused_linear_cross_entropy`，Axolotl 不会强传。结果可能是字段存在，但该模型分支不消费，或者只消费其中一部分开关。

## 9.4 本章小结

> 💡 **小结**
>
> * Monkey patch 让 Axolotl 低侵入接入 Liger，但全局状态和版本耦合是长期维护成本。
> * FLCE 的核心不是“不算 logits”，而是 chunk 化并提前计算最后层梯度。
> * 配置字段到真实行为之间还有模型类型、函数签名、互斥校验三层门控。

---

# 十、显存、性能与通信分析

## 10.0 设计哲学与核心问题

性能优化最容易被写成口号：更快、更省显存。但源码走读需要回答边界：到底省哪一块显存，新增哪一类开销，是否引入通信。本章解决的是**收益归因问题**，把参数、激活、logits、optimizer state、通信逐一拆开。

## 10.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | Liger 不切分参数；参数仍由 FSDP/DeepSpeed/QLoRA 等处理。 |
| optimizer state | ❌ | Liger 不改变 optimizer state；Adam state 等仍由 optimizer/ZeRO/FSDP 决定。 |
| backbone 激活 | 部分 ✅ | RMSNorm/MLP/RoPE 融合可减少部分中间 buffer，但不等于整层激活消失。 |
| logits `[B,S,V]` | ✅ | FLCE 避免完整 logits 长期物化，只保留 chunk logits。 |
| loss 中间张量 | ✅ | CE 与最后线性融合，loss/grad 在 chunk 内完成。 |
| 输入 batch | ❌ | Liger 不改变 dataloader 或 batch dispatch。 |
| FSDP `lm_head` 参数峰值 | ❌/⚠️ | FSDP 仍可能需要 all-gather 完整 `lm_head.weight`；Liger 只是确保在正确上下文读取。 |
| token scaling 临时 buffer | ❌/⚠️ | `use_token_scaling` 会在 chunk 内额外 clone logits 并 softmax。 |

真正的大头通常是 logits。举例：`B*S = 8192`、`V = 128000`、bf16 logits 约 2GB；如果还发生 fp32 wrapper 或额外 log_softmax 输出，峰值更高。Liger FLCE 的收益正是把这个 `[B*S,V]` 降到 `[C,V]` 的短生命周期临时张量。

## 10.2 通信开销

| 场景 | Liger 自己是否通信 | 可能出现的通信 | 说明 |
|---|---:|---|---|
| SFT + 单卡 | ❌ | 无 | 只有本地 Triton kernel。 |
| SFT + DDP | ❌ | DDP 梯度 all-reduce | Liger 不改变 DDP 梯度同步。 |
| SFT + FSDP | ❌ | FSDP 参数 all-gather / grad reduce-scatter | `lm_head` 若 FSDP 包裹，需要正确 forward context。 |
| GRPO scoring | ❌ | trainer/accelerator 自身 gather | Liger fused selective log-softmax 是本地 kernel。 |
| 保存 checkpoint | ❌ | FSDP/DeepSpeed state_dict gather/barrier | `save_trained_model()` 使用原保存逻辑。 |
| Sequence Parallel | ❌ | SP 自己的通信 | GRPO + SP + Liger loss 已被校验禁用。 |

每 step 的新增通信次数：从 LigerPlugin 源码看是 0。通信瓶颈如果出现，通常来自 FSDP/DeepSpeed/Trainer，而不是 Liger kernel 本身。

## 10.3 性能取舍

Liger 用三种复杂度换收益：

1. **用 chunked kernel 换显存**：最后 loss 路径不存完整 logits，但要循环 chunk，chunk 太小可能降低吞吐。
2. **用自定义 autograd 换中间张量减少**：forward 保存梯度缓存，而不是保存完整计算图。
3. **用 monkey patch 换低侵入集成**：不用 fork Trainer，但对上游版本和模块命名空间敏感。

在长序列、大 vocab、较大 batch 的训练中，FLCE 收益明显；在短序列、小 vocab 或推理场景中，收益可能有限，因为推理仍需 logits，且 patch 带来的维护复杂度仍在。

## 10.4 本章小结

> 💡 **小结**
>
> * Liger 最确定的显存收益来自最后 logits/loss 路径。
> * 它不替代分布式通信策略，也不减少 optimizer state。
> * 性能收益与代价都集中在 kernel fusion、chunk size、版本兼容和 patch 复杂度上。

---

# 十一、配置项、边界条件与坑点

## 11.0 设计哲学与核心问题

Liger 的配置项不是静态说明书，而是分支选择器。不同字段会进入不同源码路径，有些触发全局 patch，有些只传给 RL trainer，有些会被模型分支忽略。本章解决的是**配置到行为的映射问题**。

## 11.1 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `plugins: [axolotl.integrations.liger.LigerPlugin]` | `cli/config.py:208-220`、`base.py:370-383` | 注册 Liger 插件，后续模型加载前可 patch | 没有它，SFT `liger_*` 字段不会触发模型 patch。 |
| `liger_rope` | `plugin.py:88-89` 或自定义分支 | 替换 RoPE 实现 | 某些模型显式不支持，如 DeepSeekV2 warning、Gemma4 skip。 |
| `liger_rms_norm` | `plugin.py:96-97` | 替换 RMSNorm | `tensor_parallel_size > 1` 时被校验禁用（`args.py:86-94`）。 |
| `liger_glu_activation` | `plugin.py:100-103` | 替换 SwiGLU/GEGLU MLP | 与 `tiled_mlp` 默认冲突（`args.py:73-84`）。 |
| `liger_layer_norm` | `plugin.py:98-99` | 替换 LayerNorm | 取决于模型分支是否消费；DeepSeekV2 warning 不支持。 |
| `liger_cross_entropy` | `plugin.py:90-91` | 替换 CE 函数 | 与 FLCE/CCE/chunked CE 互斥。全局 patch 风险更高。 |
| `liger_fused_linear_cross_entropy` | `plugin.py:92-95` | 替换 ForCausalLM.forward，训练时不物化完整 logits | 某些模型不支持；Gemma4 明确 skip。 |
| `liger_use_token_scaling` | `plugin.py:57-82` | 包裹 Liger FLCE，使 `use_token_scaling=True` | 必须启用 FLCE；会多出 chunk 内 clone/softmax。 |
| `torch_compile` | `plugin.py:29-42` | 对 Liger FLCE forward/backward 加 `torch.compiler.disable` | 不是“编译 Liger”，而是绕开 compile。 |
| `dpo_use_liger_kernel` | `dpo/__init__.py:37-43` | 传给 DPO trainer args | 不等同于 LigerPlugin。 |
| `trl.use_liger_loss` | `grpo/__init__.py:146-147` | 传给 GRPO trainer args | 与 SP 冲突；绕过 batch flattening training forward。 |

## 11.2 开启该特性的最小配置

SFT 最小配置至少需要：

```yaml
plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_fused_linear_cross_entropy: true   # 或开启 rope/rms/glu/layer_norm 任一支持项
```

实践示例通常会同时开启多个 kernel。例如 `examples/llama-3/fft-8b-liger-fsdp.yaml:5-10` 同时开启 RoPE、RMSNorm、GLU 和 FLCE；`examples/qwen3/8b-qat-fsdp2.yml:9-16` 还开启了 LayerNorm。

## 11.3 静默失效与不兼容组合

* **未配置插件**：字段即使出现在 YAML，也不会触发 `LigerPlugin.pre_model_load()`。
* **模型类型不支持**：`plugin.py:290-293` 只 warning “Unsupported model config type”。
* **签名不消费字段**：`plugin.py:84-105` 只传下游函数支持的 kwargs。
* **多个 CE 优化同时开**：`validation.py:974-1002` 直接报错。
* **TP + Liger RMSNorm / TP + Liger losses**：`args.py:86-113` 禁止相关组合。
* **GRPO + SP + Liger loss**：`validation.py:721-728` 禁用。
* **Gemma4 FLCE / RoPE**：`plugin.py:258-268` warning 并 skip。
* **DeepSeekV2 RoPE / LayerNorm**：`plugin.py:137-146` warning 不支持。

## 11.4 本章小结

> 💡 **小结**
>
> * Liger 配置不是简单开关表，而是“字段 + 模型类型 + 下游签名 + 校验”的组合决策。
> * 最小开启条件是插件注册；RL Liger 字段属于另一条 trainer args 路径。
> * 不支持分支很多是 warning/skip，不一定都会硬报错。

---

# 十二、测试、示例与覆盖缺口

## 12.0 设计哲学与核心问题

测试不能只回答“有没有测试文件”，而要回答“证明了什么”。Liger 的高风险点在多卡、大 vocab、长序列和全局 patch 残留上，而这些恰恰不是轻量单元测试容易覆盖的。本章解决的是**证据边界问题**。

## 12.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/integrations/test_liger.py:44-61` | `liger_swiglu` 迁移到 `liger_glu_activation` | 覆盖废弃字段兼容。 |
| `tests/integrations/test_liger.py:63-77` | `liger_swiglu` 与 `liger_glu_activation` 冲突 | 覆盖互斥报错。 |
| `tests/integrations/test_liger.py:79-93` | `liger_use_token_scaling` 必须依赖 FLCE | 覆盖 token scaling 配置门控。 |
| `tests/e2e/patched/test_cli_integrations.py:18-47` | YAML 中 Liger 插件参数能被 `load_cfg()` 解析 | 覆盖动态 schema 合并。 |
| `tests/e2e/integrations/test_liger.py:20-113` | 小模型训练 + CE/FLCE/token scaling | 类名不是 `Test*`，源码层面像 e2e 用例，但需注意 pytest 默认收集规则。 |
| `tests/kernels/test_rms_norm_gated.py:42-180` | Fused RMSNormGated forward/backward 与 eager 对齐 | CUDA/Triton 条件测试，覆盖 Axolotl 自定义 kernel。 |
| `examples/llama-3/fft-8b-liger-fsdp.yaml:5-10` | Llama + FSDP + Liger 推荐配置 | 展示 SFT 主用法。 |
| `examples/qwen3/8b-qat-fsdp2.yml:9-16` | Qwen3 + QAT/FSDP2 + Liger | 展示 Liger 与 QAT/FSDP2 组合。 |
| `examples/deepseek-v2/qlora-fsdp-2_5.yaml:11-15` | DeepSeekV2 + QLoRA/FSDP + Liger 部分开关 | 避开 RoPE/LayerNorm，贴合源码 warning。 |

## 12.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| 多模型同进程反复 `pre_model_load()` 的 patch 幂等性 | 未看到专门覆盖 | token scaling wrapper 可能重复包裹；全局 patch 残留难定位。 |
| 下游 Liger map 与 Axolotl 自定义分支优先级 | 未看到专门覆盖 | 本地 `qwen3.py`/`llama4.py` 可能被误以为主路径。 |
| FSDP 包裹 `lm_head` + FLCE 的真实多卡行为 | 示例有，测试不充分 | 权重 summon / save / resume 边界可能只在大模型多卡暴露。 |
| Gemma4 Liger branch 的 skip 行为 | 未看到 LigerPlugin 专项测试 | 用户开 FLCE/RoPE 可能只 warning，预期与实际不一致。 |
| `liger_cross_entropy` 全局 patch 恢复 | 测试清理主要覆盖部分模块 | 其他模型模块可能被污染。 |
| 性能/显存收益回归 | 未看到 benchmark 断言 | 只能证明能跑/数值对齐，不能证明吞吐或显存收益。 |
| 多机通信场景 | 未看到 Liger 专项多机测试 | FSDP/DeepSpeed/保存行为风险难被单机测试捕获。 |
| GRPO + Liger + OPSM/async 组合 | 有源码实现，未在本文检索到专门端到端覆盖 | 可能出现 mask、IS、KL 数值边界问题。 |

## 12.3 测试覆盖的真实含义

当前测试对“配置能解析、字段冲突能报错、小 kernel 数值正确”覆盖较好；对“真实大模型 + 多卡 + 保存/resume + patch 残留”的覆盖较弱。尤其是 Liger 的价值恰恰出现在大 batch、长序列、大 vocab、多卡组合里，这类路径很难用轻量 CI 完整证明。

## 12.4 本章小结

> 💡 **小结**
>
> * 单元测试主要保护配置校验，kernel 测试保护局部数值正确。
> * 端到端示例丰富，但多卡、大模型、保存/resume、性能收益缺少强断言。
> * pytest 默认收集规则下，`LigerIntegrationTestCase` 这种类名需要额外注意是否真的被收集。

---

# 十三、局限性与已知优化点

## 13.0 设计哲学与核心问题

Liger 的源码实现很务实：能委托下游就委托，不能委托就按模型特例 patch，不能保证兼容就 warning 或报错。本章解决的是**适用边界问题**：哪些约束是硬性的，哪些成本来自维护，哪些优化点已经在源码注释中暴露。

## 13.1 硬约束

* `liger_rms_norm` 与 `tensor_parallel_size > 1` 不兼容，源码直接报错（`args.py:86-94`）。
* `liger_fused_linear_cross_entropy` 与 tensor parallel loss 不兼容，`args.py:108-113` 有 TODO 并报错。
* `liger_cross_entropy`、`liger_fused_linear_cross_entropy`、`cut_cross_entropy`、`chunked_cross_entropy` 互斥（`validation.py:974-1002`）。
* GRPO + sequence parallel + Liger loss 不支持（`validation.py:721-728`）。
* Gemma4 的 Liger RoPE 与 FLCE 在 `LigerPlugin` 中被 skip（`plugin.py:258-268`）。
* DeepSeekV2 的 Liger RoPE 与 LayerNorm 被 warning 不支持（`plugin.py:137-146`）。

## 13.2 维护成本

* 依赖下游 `liger-kernel==0.7.0`，版本固定在 `pyproject.toml:79-84`。
* 下游 Liger map 变化会改变 Axolotl 自定义分支是否可达。
* Monkey patch 写入全局模块，缺少统一恢复机制。
* `inspect.signature()` 虽然降低参数不兼容风险，但也意味着某些配置字段可能因为下游签名不支持而不生效。
* 模型专用分支依赖 Transformers 内部类名，如 `Qwen3_5RMSNorm`、`Gemma4RMSNorm`、`DeepseekV2ForCausalLM` 等，升级成本高。

## 13.3 性能瓶颈

* FLCE 仍需执行最后 `hidden @ weight.T`，只是 chunk 化和融合 CE。
* chunk 内 token scaling 会额外 `detach().clone()` logits 并 softmax（下游 `fused_linear_cross_entropy.py:105-134`）。
* 如果 `lm_head.weight` 可训练，`grad_weight: [V,H]` 仍然需要累积，不能省掉权重梯度显存/计算。
* FSDP 场景下 `lm_head` 权重读取仍可能触发 all-gather 峰值。
* Patch 本身没有性能监控；吞吐提升依赖模型、序列长度、vocab、GPU、chunk size 与 dtype。

## 13.4 已知优化点

源码中已有一些 TODO 或可改进点：

* `args.py:110` 提到 tensor parallel + Liger losses 需要更大修复。
* 下游 FLCE `fused_linear_cross_entropy.py:74` 有 TODO：评估 `.item()` 导致 CUDA synchronization 对速度的影响。
* `plugin.py` 的 token scaling patch 可以增加幂等 guard 和 restore 机制。
* 对 patch 是否实际生效，可以增加实例化后断言，例如检查 `model.forward.__module__` 或关键 module class。
* 对保存/resume 和多卡 FSDP，可以增加小模型分布式 smoke test，至少覆盖 `lm_head` FSDP redirection。
* 对 README 支持模型列表，可以改为自动生成或链接到实际 runtime map，避免文档滞后。

## 13.5 本章小结

> 💡 **小结**
>
> * Liger 的硬约束集中在 TP、SP、CE 优化互斥和模型特例上。
> * 维护成本来自全局 patch 与上游模型结构耦合。
> * 性能瓶颈没有消失，而是从显存峰值问题转为 chunk kernel、权重梯度和 FSDP 边界问题。

---

# 小结与展望

Axolotl 的 Liger Kernel 实现可以用几个关键词概括。

## 关键词一：模型加载前 patch

Liger 不是 Trainer 插件意义上的“训练循环扩展”，而是模型实例化前的 Transformers module patch。`ModelLoader.load()` 在 `_build_model()` 前调用 `PLUGIN_MANAGER.pre_model_load()`，这是整个特性成立的时间窗口。

## 关键词二：全局命名空间替换

Liger 的低侵入来自全局替换：改类、改函数、改 forward。好处是不用 fork 模型；代价是 patch 生命周期不透明，恢复困难，版本耦合强。

## 关键词三：FLCE 的 chunked logits

显存收益最大的一环是 `liger_fused_linear_cross_entropy`：它不保存完整 `[B,S,V]` logits，而是把 hidden states flatten 成 `[B*S,H]`，再按 chunk 计算 `[C,V]` logits、loss 和梯度。它省的是中间张量峰值，不是最后线性层的 FLOPs。

## 关键词四：分布式边界清晰

Liger 不创建通信组，不切 batch，不替代 FSDP/DeepSpeed。它只改变 rank 本地计算图；FSDP 参数 all-gather、梯度 reduce-scatter、checkpoint gather 仍由外层框架负责。唯一敏感点是 `lm_head.weight` 在 FLCE 中被直接读取，因此 FSDP 包裹时要进入正确 forward context。

## 关键词五：双路径 Liger

SFT 的 `LigerPlugin` 和 RL 的 `dpo_use_liger_kernel` / `trl.use_liger_loss` 是两条不同路径。前者 patch 模型类和 loss forward，后者更多是把 `use_liger_kernel` 传给 DPO/GRPO trainer，或者在 GRPO scoring/loss 中使用 fused kernel。

从适用场景看，Axolotl 的 Liger 集成最适合长序列、大 vocab、SFT/后训练中最后 logits 显存占比高的场景，也适合希望在不重写训练框架的前提下获得 kernel fusion 收益的用户。它不适合需要强隔离多模型运行态、频繁在同一进程切换 patch 状态、或依赖尚未适配模型结构的场景。

与 sequence parallel、FSDP、Cut Cross Entropy 相比，Liger 的取舍很明确：它主要是**kernel fusion + monkey patch**，不是并行策略；它用实现复杂度和版本耦合换取显存与吞吐收益。后续值得继续走读的方向，是把 Liger 与 Axolotl 的 Sequence Parallel、FSDP2/QLoRA、GRPO async/off-policy 路径放在一起看：这些特性分别从序列维度、参数维度、loss/logits 维度和 rollout 调度维度解决显存与吞吐问题，组合起来才是 Axolotl 长上下文后训练的完整工程图景。
