# Axolotl 源码走读：zigzag / stripe 与 batch ring causal 负载均衡实现解析

长序列训练里，Context / Sequence Parallelism 解决的第一层问题是“单卡放不下一整条序列”；但当序列真的被切开后，新的问题会浮上来：**causal attention 的计算量天然不均匀**。越靠后的 token 能看见越多历史 token；如果 rank 按连续序列块切分，那么 CP group 里编号越大的 rank 往往要做更多 attention block 计算。

Axolotl 的 `context_parallel_size` 已经把序列切分、DeviceMesh、FlashAttention monkey patch 串进训练链路。本文聚焦一个更窄的问题：`ring_attn_func` 文档和注释里出现的 `zigzag` / `stripe`，是否已经能缓解 `batch_ring` causal 路径里的 rank 计算不均？结论先说在前面：**从下游 `ring-flash-attn` 算法看，zigzag / stripe 正是为缓解 batch causal ring 的负载不均而设计；但在当前 Axolotl 源码中，它们尚未接入主路径，用户无法通过配置真正启用。**

> 说明：当前本地环境没有安装 `ring_flash_attn` 包；Axolotl 在 `pyproject.toml:93-96` 声明可选依赖 `ring-flash-attn>=0.1.7`。本文对下游 backend 的判断基于下载阅读的 `ring-flash-attn==0.1.7` sdist 源码，路径以 `ring_flash_attn-0.1.7/...` 标注；Axolotl 自身判断均以 `/root/axolotl` 源码为准。

# 前言

## 业务 / 工程背景

`context_parallel_size` 出现在 Axolotl 的长上下文训练场景。文档 `docs/sequence_parallelism.qmd:21-31` 给出的最小配置是：

```yaml
flash_attention: true
context_parallel_size: 4
heads_k_stride: 1
ring_attn_func:
```

它解决的不是参数显存，而是序列维度带来的激活、Q/K/V、logits 与 attention 上下文压力。`docs/nd_parallelism.qmd:37-41` 也把 CP 描述为：序列沿 length 维度切到多个 GPU 上，但 attention 需要通过 ring 通信让本地 Q 看到其他 rank 的 K/V。

## 核心矛盾

这套机制背后有两层工程矛盾：

1. **显存矛盾**：FSDP / ZeRO 可以切参数和 optimizer state，但不会自动切掉 `[batch, seq, hidden]` 这类激活；CP 必须切 sequence。
2. **调度矛盾**：连续切分虽然最简单，但 causal attention 的依赖不是均匀的。rank0 的早期 token 只看很短历史，最后一个 rank 的 token 要看几乎全序列；basic `batch_ring` 会把这种不均放大成 rank 间 kernel 调用数差异。

本文主线分为五步：先看用户配置如何进入 CP 主路径，再看 Axolotl 当前 batch ring 如何切数据、patch attention，然后进入下游 `ring-flash-attn` 比较 `batch_ring`、`zigzag`、`stripe` 的计算形态，最后分析显存、通信、测试覆盖和未接入的工程代价。

本文不展开 Ring Attention / FlashAttention / FSDP 的论文原理；只看 Axolotl 如何集成，以及 `zigzag` / `stripe` 在当前源码中到底有没有生效。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `src/axolotl/cli/main.py` / `src/axolotl/cli/train.py` | 用户从 `axolotl train` 进入配置加载与训练。 |
| `src/axolotl/cli/config.py` | 读取 YAML、CLI override，执行校验、env 准备与归一化。 |
| `src/axolotl/utils/schemas/config.py` | 声明 `context_parallel_size`、`heads_k_stride`、`ring_attn_func`。 |
| `src/axolotl/utils/schemas/enums.py` / `validation.py` | 定义真正可选的 ring backend，并选择默认值。 |
| `src/axolotl/utils/trainer.py` | 写入 Accelerate parallelism env，patch CP prepare。 |
| `src/axolotl/utils/distributed.py` | 构造 `ParallelismConfig` / `DeviceMesh`。 |
| `src/axolotl/train.py` | 训练时进入 `SequenceParallelContextManager`。 |
| `src/axolotl/utils/ctx_managers/sequence_parallel.py` | forward pre-hook 切 batch，post-hook 可选 gather 输出。 |
| `src/axolotl/monkeypatch/ring_attn/patch.py` / `adapters/batch.py` | 从 DeviceMesh 注册 CP group，并替换 HF FlashAttention。 |
| `ring_flash_attn-0.1.7/ring_flash_attn/*.py` | 下游 basic ring、zigzag、stripe backend 的真实计算/通信。 |

# 一、配置入口：`ring_attn_func` 看起来可选，实际上只剩两条路

## 1.1 设计哲学与核心问题

要判断 zigzag / stripe 是否生效，第一步不是看 backend 函数，而是看用户能不能把配置传到那里。Axolotl 的配置系统承担三个职责：

- 确保 CP 基础前提成立：开启 FlashAttention、安装 `ring_flash_attn`；
- 根据 `sample_packing` 选择默认 backend；
- 把用户字符串转换成 `RingAttnFunc` 枚举。

这层如果已经挡住 `batch_zigzag`，后面再多 backend 代码也不会进入主流程。

## 1.2 源码入口与关键对象

```text
src/axolotl/cli/main.py
  - train：Click 命令入口，最终 launch 到 axolotl.cli.train。

src/axolotl/cli/train.py
  - do_cli：调用 load_cfg，再进入 do_train。

src/axolotl/cli/config.py
  - load_cfg：读 YAML、合并 CLI override、validate_config、prepare_optim_env、normalize_config。

src/axolotl/utils/schemas/enums.py
  - RingAttnFunc：真实可用枚举。

src/axolotl/utils/schemas/validation.py
  - check_context_parallel_size：CP 前置校验。
  - validate_ring_attn_func：默认 backend 选择。
```

## 1.3 主流程拆解

入口链路如下：

```text
User: axolotl train config.yml
  -> src/axolotl/cli/main.py:78-128 train()
    -> launch_training(...)
      -> accelerate launch -m axolotl.cli.train
        -> src/axolotl/cli/train.py:55-91 do_cli()
          -> src/axolotl/cli/config.py:230-346 load_cfg()
            -> validate_config(...)
            -> prepare_optim_env(cfg)
            -> normalize_config(cfg)
```

`ring_attn_func` 字段声明在 `src/axolotl/utils/schemas/config.py:987-991`。这里的描述写着：

```text
One of 'varlen_llama3', 'batch_ring', 'batch_zigzag', 'batch_stripe'.
```

但真正的枚举在 `src/axolotl/utils/schemas/enums.py:100-108`：

```python
class RingAttnFunc(str, Enum):
    VARLEN_LLAMA3 = "varlen_llama3"
    BATCH_RING = "batch_ring"
    # VARLEN_RING = "varlen_ring"
    # VARLEN_ZIGZAG = "varlen_zigzag"
    # BATCH_ZIGZAG = "batch_zigzag"
    # BATCH_STRIPE = "batch_stripe"
```

也就是说，`batch_zigzag` 与 `batch_stripe` 只是注释，不是合法枚举。`validate_ring_attn_func()` 会把用户输入强制转换为枚举（`validation.py:1568-1569`）：

```python
if self.ring_attn_func is not None:
    self.ring_attn_func = RingAttnFunc(self.ring_attn_func)
else:
    self.ring_attn_func = RingAttnFunc.VARLEN_LLAMA3 if sample_packing else RingAttnFunc.BATCH_RING
```

我用当前源码直接验证枚举转换：

```text
varlen_llama3 -> RingAttnFunc.VARLEN_LLAMA3
batch_ring -> RingAttnFunc.BATCH_RING
batch_zigzag -> ValueError 'batch_zigzag' is not a valid RingAttnFunc
batch_stripe -> ValueError 'batch_stripe' is not a valid RingAttnFunc
```

所以用户层面的真实分叉只有：

```text
context_parallel_size <= 1
  -> 不启用 CP，不看 ring_attn_func

context_parallel_size > 1 && sample_packing == true
  -> 默认 ring_attn_func = varlen_llama3

context_parallel_size > 1 && sample_packing == false
  -> 默认 ring_attn_func = batch_ring
```

`check_context_parallel_size()` 还会做基础约束：`flash_attention` 必须为真（`validation.py:1517-1520`），`sample_packing` 下 `micro_batch_size > 1` 会报错（`validation.py:1522-1526`），并尝试 import `ring_flash_attn`（`validation.py:1528-1550`）。

## 1.4 关键细节与误区澄清

> 这里有一个容易误解的点：`config.py` 的字段描述提到 `batch_zigzag` / `batch_stripe`，但源码中 `RingAttnFunc` 并未启用这两个值。以源码为准，当前用户不能通过 YAML 打开 zigzag / stripe。

> 第二个误区是把 `ring_attn_func` 当成通用策略参数。实际上 `apply_sequence_parallelism()` 的 docstring 明确说该参数 “Currently unused”（`sequence_parallel.py:42-43`），当前它只影响 attention monkey patch 选择，不影响 batch 切分方式。

> 第三个误区是认为 `sequence_parallel_degree` 与 `context_parallel_size` 是两套逻辑。`validation.py:1508-1514` 会把 deprecated 的 `sequence_parallel_degree` 转成 `context_parallel_size`，后面主路径只看后者。

## 1.5 本章小结

> 💡 **小结**
>
> * 当前 Axolotl 合法 backend 只有 `varlen_llama3` 与 `batch_ring`。
> * `batch_zigzag` / `batch_stripe` 出现在字段描述和注释中，但没有通过枚举校验。
> * 是否 sample packing 决定默认 backend：packing 走 `varlen_llama3`，非 packing 走 `batch_ring`。

# 二、初始化与拓扑：CP group 是真的，但 backend 仍然只注册 basic ring

## 2.1 设计哲学与核心问题

CP 不是单卡优化，它必须知道“哪些 rank 共同切同一条序列”。Axolotl 借用 Accelerate 的 `ParallelismConfig` 和 PyTorch `DeviceMesh` 建立名为 `cp` 的维度，然后从这个维度拿 process group 传给 `ring_flash_attn`。

这一层解决的是状态和通信问题：同一 CP group 内 rank 需要拿同一 batch 的不同 sequence shard；attention backend 也必须只在这个 group 内做 P2P / all-gather，而不是全局 world group。

## 2.2 源码入口与关键对象

```text
src/axolotl/utils/trainer.py
  - setup_parallelism_envs：写 PARALLELISM_CONFIG_CP_SIZE 与 ACCELERATE_USE_PARALLELISM_CONFIG。

src/axolotl/monkeypatch/accelerate/parallelism_config.py
  - patch_prepare_cp：把 Accelerate 原生 torch CP context 替换成 no-op。
  - patch_parallelism_config：放宽 pure CP 的 Accelerate 校验。

src/axolotl/utils/distributed.py
  - build_parallelism_config / _get_parallel_config_kwargs：推导 cp_size、dp_shard_size、tp_size。

src/axolotl/monkeypatch/ring_attn/patch.py
  - register_ring_attn_from_device_mesh：取 `device_mesh[("cp",)]`，设置全局 RING_ATTN_GROUP，并注册 attention patch。
```

## 2.3 主流程拆解

配置加载后，`prepare_optim_env()` 会调用 `setup_parallelism_envs()`（`utils/trainer.py:643-667`）：

```python
# src/axolotl/utils/trainer.py:621-640
if cfg.context_parallel_size and cfg.context_parallel_size > 1:
    os.environ["PARALLELISM_CONFIG_CP_SIZE"] = str(cfg.context_parallel_size)
    os.environ["ACCELERATE_ALLOW_CP_STANDALONE"] = "true"
    patch_prepare_cp()
if set_accelerate_parallelism_config:
    os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
```

这里有一个关键设计：Axolotl 不直接使用 Accelerate 的 torch CP 切分上下文。`patch_prepare_cp()` 把 `Accelerator._prepare_cp` 改成设置一个 no-op `_cp_context`（`monkeypatch/accelerate/parallelism_config.py:80-96`）。与此同时，Transformers 的 `Trainer.training_step()` 仍会调用 `_prepare_context_parallel_inputs()`，但真正切 batch 的工作在 Axolotl 自己的 forward pre-hook 中完成。

拓扑推导在 `utils/distributed.py:299-370`。当 `context_parallel_size > 1` 时：

```python
pc_kwargs["cp_size"] = context_parallel_size
remaining_world_size = remaining_world_size // context_parallel_size
```

若没有显式 DP 配置，剩余 world size 会默认进入 `dp_shard_size`（`utils/distributed.py:338-341`）。例如：

```text
world_size = 8
context_parallel_size = 4
tensor_parallel_size = 1

_get_parallel_config_kwargs:
  cp_size = 4
  remaining_world_size = 2
  dp_shard_size = 2

逻辑 mesh:
  dp_shard x cp = 2 x 4

同一 CP group 内 rank 共享 batch，不同 CP group 处理不同 batch。
```

训练真正开始前，`src/axolotl/train.py:205-220` 会进入：

```python
SequenceParallelContextManager(
    models=[trainer.model, maybe_ref_model],
    context_parallel_size=cfg.context_parallel_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    ring_attn_func=cfg.ring_attn_func,
    heads_k_stride=cfg.heads_k_stride,
    gather_outputs=cfg.rl in {RLType.GRPO, RLType.EBFT},
    device_mesh=trainer.accelerator.torch_device_mesh,
)
```

`SequenceParallelContextManager.__init__()` 立即 `_register_ring_attn()`（`sequence_parallel.py:207-213`），最终进入 `register_ring_attn_from_device_mesh()`：

```python
# src/axolotl/monkeypatch/ring_attn/patch.py:159-184
sequence_mesh = device_mesh[("cp",)]
sequence_pg = sequence_mesh.get_group()
set_ring_attn_group(sequence_pg)
```

然后才按 backend 注册 patch：

```python
# src/axolotl/monkeypatch/ring_attn/patch.py:186-211
if ring_attn_func is RingAttnFunc.VARLEN_LLAMA3:
    ring_flash_attn.adapters.hf_adapter.substitute_hf_flash_attn(...)
elif ring_attn_func is RingAttnFunc.BATCH_RING:
    axolotl.monkeypatch.ring_attn.adapters.batch.substitute_hf_flash_attn(...)
```

这里没有 `BATCH_ZIGZAG` / `BATCH_STRIPE` 分支。

## 2.4 关键细节与误区澄清

> 容易误解的是：既然 DeviceMesh 里有 `cp` 维度，是否说明 Axolotl 使用了 Accelerate / torch 原生 context parallel？不是。`patch_prepare_cp()` 把 Accelerate 的 `_prepare_cp` 改成 no-op；Axolotl 自己在 model forward pre-hook 里切输入，在 FlashAttention monkey patch 里通信。

> 另一个误区是：`register_ring_attn_from_device_mesh()` 已经接收 `ring_attn_func`，所以任何下游函数都能接上。实际代码只有两个分支：`VARLEN_LLAMA3` 和 `BATCH_RING`。如果未来枚举新增 zigzag / stripe，这里仍必须新增注册逻辑。

## 2.5 本章小结

> 💡 **小结**
>
> * CP group 是通过 Accelerate `DeviceMesh[("cp",)]` 建出来的，attention 通信只在该 group 内发生。
> * Axolotl intentionally 绕开 Accelerate 原生 torch CP 切分，把切分放到自己的 forward hook。
> * 当前注册分支没有 zigzag / stripe，因此拓扑存在不等于 backend 生效。

# 三、Forward 切分：当前 batch 路径是假定“连续 rank chunk”的

## 3.1 设计哲学与核心问题

要缓解 batch ring 的 causal 负载不均，仅替换 attention 函数还不够。因为负载均衡 backend 通常要求不同的数据布局：

- basic ring：rank i 拿连续第 i 段；
- zigzag：rank i 通常拿一段靠前 token + 一段靠后 token；
- stripe：rank i 拿按 stride / stripe 交错分布的 token。

如果上游切分仍是连续 chunk，而 backend 按 zigzag / stripe 解释本地 Q/K/V，输出语义就可能错。因此需要先看 Axolotl 当前 batch 是如何切的。

## 3.2 源码入口与关键对象

```text
src/axolotl/utils/ctx_managers/sequence_parallel.py
  - apply_sequence_parallelism：对 input_ids / labels / attention_mask / position_ids 等做 padding 与切片。
  - SequenceParallelContextManager._register_model_hooks：在 model forward 前切输入。
  - AllGatherWithGrad：RL 路径可把输出按 sequence 维 all-gather 回来。
```

## 3.3 主流程拆解

`apply_sequence_parallelism()` 顶部有一个很直白的 TODO（`sequence_parallel.py:22-23`）：

```python
# TODO(djsaunde): implement zigzag, stripe patterns here (and elsewhere) in this
# module. Currently, we just focus on batch ring and varlen llama3 for simplicity.
```

这基本已经给出结论：当前 Axolotl 的切分逻辑没有实现 zigzag / stripe pattern。

真实切分代码在 `sequence_parallel.py:135-148`：

```python
for key in batch:
    if not isinstance(batch[key], torch.Tensor) or batch[key].dim() <= 1:
        continue

    if batch[key].size(1) == total_seq_len:
        batch[key] = batch[key].chunk(local_world_size, dim=1)[local_rank].contiguous()
    elif key == "logits_to_keep":
        batch[key] = batch[key].chunk(local_world_size, dim=0)[local_rank].contiguous()
```

这是一种**连续切分**：

```text
原始 input_ids: [B, S]
cp_size = 4

rank0: input_ids[:, 0       : S/4]
rank1: input_ids[:, S/4     : S/2]
rank2: input_ids[:, S/2     : 3S/4]
rank3: input_ids[:, 3S/4    : S]
```

padding 也只是为了让 sequence length 能被 group size 整除（`sequence_parallel.py:96-134`）：

```text
if S % min(cp_size, 64) != 0:
  input_ids / attention_mask / position_ids 右侧 pad 0
  labels 右侧 pad -100
```

`SequenceParallelContextManager._register_model_hooks()` 把这段逻辑注册到每个模型 forward 之前（`sequence_parallel.py:255-288`）。因此主流程是：

```text
Trainer.training_step
  -> model(**inputs)
    -> sequence_parallel_pre_hook
      -> apply_sequence_parallelism(kwargs)
        -> position_ids 准备
        -> pad 到可切长度
        -> 每个 tensor 沿 dim=1 连续 chunk
    -> 模型 forward
      -> 被 monkey-patch 的 FlashAttention
```

若是 GRPO / EBFT，`gather_outputs=True`，post-hook 会把输出 all-gather 回完整 sequence（`sequence_parallel.py:290-303`、`359-365`）。`AllGatherWithGrad` 的 forward 是先 all-gather shape，再 all-gather tensor，最后沿 sequence dim concat（`sequence_parallel.py:393-415`）；backward 则按 rank 的 offset 切回本地梯度（`sequence_parallel.py:437-443`）。这同样假设输出顺序是连续 rank chunk。

## 3.4 关键细节与误区澄清

> 这里最关键的误区是：只把 `RING_ATTN_FUNC_MAPPING` 改成 zigzag / stripe 就能工作。下游 `ring-flash-attn` 的测试并不是这样切本地数据的。`test_zigzag_ring_flash_attn_func.py:9-13` 把全序列切成 `2 * world_size` 份，rank i 拿第 i 份和倒数第 i 份；`test_stripe_flash_attn_func.py:9-14` 则把序列按 world size 做 stripe 重排。Axolotl 当前 `chunk(local_world_size, dim=1)[local_rank]` 与这两种布局都不一致。

> 第二个误区是：post-hook gather 只是性能优化，可以不改。对于需要恢复完整输出的 RL 路径，`AllGatherWithGrad` 直接按 rank 顺序 concat；一旦前向改成 zigzag / stripe 布局，gather 后还必须 inverse permutation，否则 logits / loss / metrics 的 token 顺序会错。

## 3.5 本章小结

> 💡 **小结**
>
> * 当前 Axolotl forward pre-hook 对所有序列张量做连续 chunk。
> * 源码 TODO 明确指出 zigzag / stripe pattern 尚未实现。
> * 负载均衡 backend 需要数据布局和输出反变换一起改，不能只替换 attention 函数。

# 四、Attention Patch：batch adapter 只映射到 `ring_flash_attn_func`

## 4.1 设计哲学与核心问题

Axolotl 不修改每个模型的 attention 模块，而是替换 Transformers 的 FlashAttention 入口。这样接入成本低，但维护风险集中在 monkey patch 上：Transformers 签名变了、下游包 API 变了，patch 都可能失效。

对本文的问题来说，关键是 batch adapter 的映射表：它决定了 `batch_ring` 最终调用哪个下游函数。

## 4.2 源码入口与关键对象

```text
src/axolotl/monkeypatch/ring_attn/patch.py
  - register_ring_attn_from_device_mesh：选择 varlen 或 batch adapter。

src/axolotl/monkeypatch/ring_attn/adapters/batch.py
  - RING_ATTN_FUNC_MAPPING：Axolotl 当前 batch backend 映射。
  - create_flash_attn_forward_varlen_llama3：创建兼容 HF `_flash_attention_forward` 签名的 wrapper。
  - substitute_hf_flash_attn：替换 Transformers `_flash_attention_forward`。
```

## 4.3 主流程拆解

batch adapter 的导入和映射在 `adapters/batch.py:17-39`：

```python
from ring_flash_attn import ring_flash_attn_func

RING_ATTN_FUNC_MAPPING = {
    RingAttnFunc.BATCH_RING: torch.compile(ring_flash_attn_func),
    # RingAttnFunc.BATCH_ZIGZAG: torch.compile(zigzag_ring_flash_attn_func),
    # RingAttnFunc.BATCH_STRIPE: torch.compile(stripe_flash_attn_func),
}
```

真正调用在 `adapters/batch.py:136-149`：

```python
attn_output = RING_ATTN_FUNC_MAPPING[ring_attn_func](
    query_states,
    key_states,
    value_states,
    dropout_p=dropout,
    softmax_scale=softmax_scale,
    causal=causal,
    window_size=window_size,
    group=process_group,
)
```

这说明当前 batch 路径只有一个 backend：下游 `ring_flash_attn_func`。注释里的 zigzag / stripe 没有 import，没有枚举，没有 mapping key。

替换 HF 入口发生在 `substitute_hf_flash_attn()`（`adapters/batch.py:156-196`）：

```text
old = transformers.modeling_flash_attention_utils._flash_attention_forward
new = create_flash_attn_forward_varlen_llama3(process_group, ring_attn_func)
check_params(old, new)
transformers.modeling_flash_attention_utils._flash_attention_forward = new
ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
```

函数名 `create_flash_attn_forward_varlen_llama3` 在 batch adapter 里有些误导；它实际创建的是 batch ring wrapper，并调用 `RING_ATTN_FUNC_MAPPING[ring_attn_func]`。这属于命名上的历史痕迹，不影响主路径判断。

## 4.4 关键细节与误区澄清

> 容易误解的是：下游 `ring_flash_attn` 包已经导出了 `zigzag_ring_flash_attn_func` / `stripe_flash_attn_func`，所以 Axolotl 自动支持。不是。Axolotl 只 import 了 `ring_flash_attn_func`，mapping 里只启用 `BATCH_RING`。

> 另一个误区是：`substitute_hf_flash_attn()` 是局部 patch。实际上它改的是 `transformers.modeling_flash_attention_utils._flash_attention_forward` 这个模块级函数，并且 `SequenceParallelContextManager.__exit__()` 只移除 model hooks，没有恢复 attention patch；`sequence_parallel.py:238-245` 明确 TODO “Un-patch attention and accelerate functions”。这意味着同一进程后续模型也可能受 patch 影响。

## 4.5 本章小结

> 💡 **小结**
>
> * Axolotl batch attention patch 当前只会调用下游 `ring_flash_attn_func`。
> * zigzag / stripe 在 mapping 中只是注释，尚未进入可执行路径。
> * monkey patch 是全局替换，不随 context manager 自动恢复，维护和测试隔离成本较高。

# 五、完整主路径串联：一次 `batch_ring` 训练 step 发生了什么

## 5.1 完整调用栈

```text
User: axolotl train config.yml
  │
  ├─ Step 1: 配置加载与校验
  │     └─ src/axolotl/cli/config.py:230-346 load_cfg
  │        ├─ validation.py:1508-1577 CP 校验与 ring_attn_func 默认值
  │        ├─ utils/trainer.py:621-640 写 Accelerate env
  │        └─ utils/config/__init__.py:134-142 修正 batch_size
  │
  ├─ Step 2: 模型与 Trainer 初始化
  │     └─ src/axolotl/train.py:522-569 setup_model_and_trainer
  │        ├─ loaders/model.py:162-194 ModelLoader.load
  │        ├─ patch_manager.py:144-149 patch HF Trainer CP guard
  │        └─ patch_manager.py:279-286 patch Accelerate ParallelismConfig
  │
  ├─ Step 3: DataLoader 按 DeviceMesh 分发 batch
  │     └─ AxolotlTrainer._get_dataloader -> accelerator.prepare(dataloader)
  │        └─ accelerate/data_loader.py:1119-1155 让同一 CP group 拿同一 batch
  │
  ├─ Step 4: 训练前注册 SP context
  │     └─ src/axolotl/train.py:205-220 SequenceParallelContextManager
  │        ├─ register_ring_attn_from_device_mesh(device_mesh[("cp",)])
  │        └─ batch adapter 替换 HF FlashAttention
  │
  ├─ Step 5: 每次 model forward
  │     └─ sequence_parallel_pre_hook
  │        ├─ apply_sequence_parallelism: 连续切 input_ids/labels/position_ids
  │        └─ model forward -> patched _flash_attention_forward -> ring_flash_attn_func
  │
  └─ Step 6: 保存
        └─ train.py:254-386 save_trained_model
           └─ core/trainers/base.py:812-823 CP 下 state_dict tensor detach().cpu()
```

## 5.2 每一层做了什么

| 层级 | 输入 | 输出 / 状态变化 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| 配置校验 | YAML / CLI kwargs | `cfg.context_parallel_size`、`cfg.ring_attn_func` | 无 | 无 | 初始化一次 |
| env 准备 | cfg | `PARALLELISM_CONFIG_CP_SIZE`、`ACCELERATE_USE_PARALLELISM_CONFIG` | 无 | 无 | 初始化一次 |
| DeviceMesh | world size / cp / tp / dp | `trainer.accelerator.torch_device_mesh` | 进程组初始化 | 无 | 初始化一次 |
| DataLoader prepare | dataset batch | 同一 CP group 拿同一 batch | sampler / dispatch 内部同步 | batch 冗余到 CP group | 每个 epoch/loader |
| forward pre-hook | `[B, S]` batch | `[B, S/cp]` local shard | `num_items_in_batch` 可能 all-reduce | 降低激活/logits局部长度 | 每次 forward |
| attention backend | local Q/K/V | local attention output | 每层 ring / all-gather | attention buffer 取决于 backend | 每层 forward/backward |
| post-hook | local output | GRPO/EBFT 可 all-gather 成完整输出 | all-gather | 输出显存可能恢复完整长度 | 每次 forward，按配置 |
| save | state_dict | CPU clone 后保存 | FSDP/ZeRO 另有通信 | CPU 内存峰值上升 | checkpoint / 结束 |

其中 DataLoader 分发不是 Axolotl 自己手写的 sampler（SFT 主路径），而是 Accelerate 看到 device mesh 后调整 `process_index` / `num_processes`。源码 `accelerate/data_loader.py:1119-1155` 对 CP 的处理是：如果 mesh 有 `cp`，则 `process_index = process_index // (tp_size * cp_size)`，使同一 CP group 中多个 rank 在 data 视角上得到同一个 index。

## 5.3 哪些逻辑不在主路径

| 看似相关的函数 / 文件 | 为什么容易误解 | 实际是否在主流程 | 正确理解 |
|---|---|---|---|
| `RingAttnFunc.BATCH_ZIGZAG` / `BATCH_STRIPE` 注释 | schema 描述提到了这两个值 | ❌ | 注释值，枚举未启用，用户配置会报错。 |
| `RING_ATTN_FUNC_MAPPING` 中 zigzag / stripe 注释 | 看起来只差取消注释 | ❌ | 还缺 import、枚举、注册分支、数据 layout、gather 反变换。 |
| Accelerate 原生 `maybe_context_parallel` | Transformers training_step 会调用 | 基本 no-op | Axolotl patch 了 `_prepare_cp`，真正切分在 forward pre-hook。 |
| 文档中的“DataCollator handles chunking” | 文档 `docs/sequence_parallelism.qmd:42-44` 这样写 | ❌ | 当前源码由 `SequenceParallelContextManager` 的 pre-hook 切。 |
| GRPO `SequenceParallelRepeatRandomSampler` | 也在做 CP group 数据重复 | 仅 RL/GRPO 路径 | SFT / causal 主路径依赖 Accelerate DataLoader mesh 调整。 |

> 💡 **小结**
>
> * 一次标准 batch ring step 的真实主路径是：Accelerate 保证 CP group 同 batch，Axolotl pre-hook 连续切序列，HF FlashAttention 被 patch 到 basic ring。
> * zigzag / stripe 相关代码目前全部停在注释或下游包，不在 Axolotl 主路径。
> * 保存阶段有 CP 特定 CPU clone，但它不改变 ring backend，只避免 CP eval 后 tensor storage 指针问题。

# 六、核心机制深挖：为什么 basic batch ring 会 rank 计算不均

## 6.1 设计哲学与核心问题

basic batch ring 的直觉很简单：每个 rank 持有本地 Q/K/V；K/V 沿环传递，rank 每拿到一段 K/V，就用本地 Q 做一次 FlashAttention block，并把 block 输出用 online softmax 合并。

非 causal 时，每个 rank 对每段 K/V 都要算，负载均匀。causal 时，早期 query 不能看未来 K/V，因此 rank 越靠前，可见 K/V 段越少；rank 越靠后，可见历史越多。这就是连续切分 + causal mask 的调度不均。

## 6.2 源码入口与关键对象

```text
ring_flash_attn-0.1.7/ring_flash_attn/ring_flash_attn.py
  - ring_flash_attn_forward：basic batch ring forward。
  - ring_flash_attn_backward：basic batch ring backward。

ring_flash_attn-0.1.7/ring_flash_attn/utils.py
  - RingComm：P2P isend/irecv 的 ring 通信封装。
  - update_out_and_lse：online 合并 block attention 输出。
```

## 6.3 主流程拆解

basic forward 的关键循环在 `ring_flash_attn.py:26-63`：

```python
for step in range(comm.world_size):
    if step + 1 != comm.world_size:
        next_k, next_v = comm.send_recv_kv(k, v)

    if not causal or step <= comm.rank:
        outputs = _flash_attn_forward(q=q, k=k, v=v, causal=causal and step == 0, ...)
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)

    if step + 1 != comm.world_size:
        comm.wait()
        k, v = next_k, next_v
```

注意这一行：`if not causal or step <= comm.rank`。当 `causal=True` 且 `world_size=4`：

```text
rank0: step 0 计算，step 1/2/3 不计算
rank1: step 0/1 计算，step 2/3 不计算
rank2: step 0/1/2 计算，step 3 不计算
rank3: step 0/1/2/3 全计算
```

也就是 kernel 调用数近似 `rank + 1`。虽然所有 rank 都要参与 `send_recv_kv()`，但 rank0 后续更多是在通信等待，而 rank3 还在持续计算。backward 也有类似条件，`ring_flash_attn.py:97-117` 中 `if step <= kv_comm.rank or not causal` 决定是否执行 `_flash_attn_backward`。

通信封装 `RingComm` 在 `utils.py:98-151`：每次 `send_recv_kv()` 对 K/V 分别创建 `isend` 与 `irecv`，再 `batch_isend_irecv()`。也就是说 basic ring 的通信轮数是固定的 `world_size - 1`，负载不均主要来自每轮是否执行 attention kernel，而不是某些 rank 少通信。

一个 4-rank causal basic ring 的状态可以画成：

```text
连续切分:
  rank0: Q/K/V for tokens [0, 1/4S)
  rank1: Q/K/V for tokens [1/4S, 1/2S)
  rank2: Q/K/V for tokens [1/2S, 3/4S)
  rank3: Q/K/V for tokens [3/4S, S)

可见历史:
  rank0 只需 K0
  rank1 需要 K0,K1
  rank2 需要 K0,K1,K2
  rank3 需要 K0,K1,K2,K3

结果:
  通信轮数类似，但 attention kernel 数: 1,2,3,4
```

## 6.4 关键细节与误区澄清

> 容易误解的是：basic ring 中所有 rank 都跑 `world_size` 次循环，所以计算量相同。循环次数相同不代表 kernel 调用相同；`step <= comm.rank` 会让低 rank 跳过大量 `_flash_attn_forward` / `_flash_attn_backward`。

> 另一个误区是：rank0 少算就一定更快。分布式 step 的耗时由最慢 rank 与通信同步共同决定。rank0 少算并不能让整体 step 结束，因为它仍要参与 K/V ring，并等待高 rank 完成更多 kernel。

## 6.5 本章小结

> 💡 **小结**
>
> * basic batch ring 在 causal + 连续切分下天然不均，rank 越大 kernel 调用越多。
> * 通信轮数基本相同，瓶颈来自“高 rank 多算、低 rank 等待”。
> * 这正是 zigzag / stripe backend 试图缓解的问题。

# 七、zigzag / stripe：能缓解，但 Axolotl 还缺布局接入

## 7.1 设计哲学与核心问题

负载均衡 backend 的核心思想不是减少 causal attention 总 FLOPs，而是把 FLOPs 在 rank 间分摊得更均匀。

下游 `ring-flash-attn` README 明确把 `zigzag_ring_flash_attn_func` 描述为 “more compute balanced version”，把 `stripe_flash_attn_func` 描述为 stripe attention 版本（`README.md:7-12`）。其 benchmark 中 batch API 在 8xH800 / 8xA100 上，zigzag / stripe 都明显快于 basic ring（`README.md:80-89`）。这些数据不是 Axolotl 的 e2e 结果，但说明下游 backend 的设计目标就是缓解 basic ring 不均。

## 7.2 源码入口与关键对象

```text
ring_flash_attn-0.1.7/ring_flash_attn/zigzag_ring_flash_attn.py
  - zigzag_ring_flash_attn_forward：每个 rank 持有前后两段，按 q/q1 分摊计算。

ring_flash_attn-0.1.7/ring_flash_attn/stripe_flash_attn.py
  - stripe_flash_attn_forward：每个 rank 持有交错 stripe，并在未来块上使用 shift。

ring_flash_attn-0.1.7/test/test_zigzag_ring_flash_attn_func.py
  - extract_local：展示 zigzag 期望的数据布局。

ring_flash_attn-0.1.7/test/test_stripe_flash_attn_func.py
  - extract_local：展示 stripe 期望的数据布局。
```

## 7.3 主流程拆解

zigzag forward 在 `zigzag_ring_flash_attn.py:19-80` 中要求 `causal=True`，然后把本地 q 分成前后两半：

```python
assert causal == True
block_seq_len = q.shape[1] // 2
q1 = q[:, block_seq_len:]

for step in range(comm.world_size):
    if step == 0:
        forward(q, k, v, causal=True)
    elif step <= comm.rank:
        k0 = k[:, :block_seq_len]
        forward(q, k0, v0, causal=False)
    else:
        forward(q1, k, v, causal=False)
```

这和 basic ring 最大不同是：即使 `step > rank`，也不是跳过计算，而是让本地后半段 `q1` 去算可见的另一部分。它通过“每个 rank 同时持有靠前与靠后 token”的布局，让低 rank 也有足够的后段 query 可算。

但这个布局不是 Axolotl 当前的连续 chunk。下游测试清楚展示了 zigzag 期望的本地切片（`test_zigzag_ring_flash_attn_func.py:9-13`）：

```python
value_chunks = value.chunk(2 * world_size, dim=dim)
local_value = torch.cat(
    [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
)
```

例如 world_size=4，序列先切成 8 份：

```text
C0 C1 C2 C3 C4 C5 C6 C7

zigzag layout:
  rank0: C0 + C7
  rank1: C1 + C6
  rank2: C2 + C5
  rank3: C3 + C4
```

这样每个 rank 都有一段早期 token 和一段后期 token，causal 计算更均匀。

stripe 的测试布局也不是连续 chunk（`test_stripe_flash_attn_func.py:9-14`）：

```python
value = torch.stack(value.split(world_size, dim=dim), dim=dim).transpose(dim, dim + 1)
local = value[slicer_for_rank]
```

直观上它把 token 按 stripe 分给 rank：

```text
原序列按每 4 个 token 成组:
  [t0 t1 t2 t3] [t4 t5 t6 t7] ...

stripe layout:
  rank0: t0, t4, t8, ...
  rank1: t1, t5, t9, ...
  rank2: t2, t6, t10, ...
  rank3: t3, t7, t11, ...
```

stripe forward 在 `stripe_flash_attn.py:29-97` 中每个 step 都执行 attention；当 `step > rank` 时，它改用 `q[:, 1:]` 和 `k[:, :-1]` 的 shifted causal 计算（`stripe_flash_attn.py:63-93`），避免直接看未来 token。

所以：

```text
basic batch_ring:
  目标: 最少数据重排，连续 chunk
  代价: causal 计算 rank 不均

zigzag:
  目标: 每 rank 同时持有早期+后期块
  代价: 数据 layout / gather inverse 更复杂

stripe:
  目标: token 交错分布，接近均匀计算
  代价: 位置、labels、输出恢复都必须跟随 stripe permutation
```

## 7.4 关键细节与误区澄清

> 最重要的误区：zigzag / stripe 不是“同样输入 layout 下换一个 kernel”。下游测试已经把本地数据提取方式写死为 zigzag / stripe layout。Axolotl 当前连续切片不满足这些假设。

> 另一个误区：zigzag / stripe 会显著节省显存。它们主要缓解计算负载和等待，不是新的参数 sharding 或 activation checkpointing。序列仍然被 CP 切分，K/V 仍然通信；显存收益主要来自 CP 本身，而不是 backend 负载均衡。

> 第三个误区：`stripe` 一定比 `zigzag` 更优。下游 README benchmark 中 zigzag 在 batch fwd+bwd 上高于 stripe（`README.md:80-89`），但这不是 Axolotl e2e 结论；真实表现还取决于模型、seq_len、interconnect、torch.compile、数据 layout 成本和 output gather。

## 7.5 本章小结

> 💡 **小结**
>
> * 算法层面，zigzag / stripe 确实针对 basic batch ring causal rank 不均。
> * 下游源码要求特殊本地数据布局，不能直接套在 Axolotl 的连续 chunk 上。
> * 当前 Axolotl 没有实现 layout、inverse gather、配置和测试，因此“能缓解”仍停留在下游 backend 能力，而不是 Axolotl 用户能力。

# 八、关键数据流 / 状态流 / shape 流程

## 8.1 Tensor shape 变化

以非 sample packing、`batch_ring`、`cp_size=4` 为例：

```text
DataLoader 输出（同一 CP group 各 rank 拿同一 batch）:
  input_ids:      [B, S]
  attention_mask: [B, S]
  labels:         [B, S]
  position_ids:   [B, S]

Axolotl forward pre-hook:
  若 S 不能被 min(cp_size,64) 整除，右侧 padding
  input_ids -> chunk(dim=1)[local_rank]

rank_i 输入模型:
  input_ids:      [B, S/4]
  attention_mask: [B, S/4]
  labels:         [B, S/4]
  position_ids:   [B, S/4]

模型内部 attention:
  q/k/v: [B, S/4, n_heads, head_dim]

batch_ring backend:
  K/V 在 CP group 内沿 ring 传递
  输出: [B, S/4, n_heads, head_dim]

LM head / loss:
  logits: [B, S/4, vocab]
  labels: [B, S/4]

GRPO/EBFT gather_outputs=True 时:
  output tensor all-gather -> [B, S, ...]
```

哪一步节省显存？主要是 pre-hook 后的本地 sequence length 变为 `S / cp_size`，因此 transformer block 激活、Q/K/V、MLP 激活、logits 都按局部序列长度下降。哪一步可能恢复冗余？`gather_outputs=True` 的 RL 路径会把输出 gather 回完整 sequence；保存阶段也可能把 state dict clone 到 CPU。

如果要支持 zigzag，shape 仍可能是 `[B, S/cp]`，但语义不再是连续 token：

```text
zigzag local rank0: [C0, C7]  -> shape 仍是 [B, S/4]
stripe local rank0: [t0,t4,t8,...] -> shape 仍是 [B, S/4]
```

这就是工程风险所在：shape 看起来没变，但 token 顺序变了，`labels`、`position_ids`、`logits_to_keep`、loss masking、output gather 都必须同步变化。

## 8.2 Rank / Mesh / Process Group 变化

```text
world_size = 8
context_parallel_size = 4
默认 dp_shard_size = 2

DeviceMesh:
  dp_shard dimension: 2
  cp dimension:       4

CP group 0: rank0 rank1 rank2 rank3
CP group 1: rank4 rank5 rank6 rank7

Data view:
  rank0-3 处理 batch A 的不同 token shard
  rank4-7 处理 batch B 的不同 token shard
```

`register_ring_attn_from_device_mesh()` 把 `device_mesh[("cp",)]` 转成 process group（`patch.py:159-184`）。随后：

- `get_ring_attn_group()` 给 `apply_sequence_parallelism()` 的 `num_items_in_batch` all-reduce 使用；
- batch adapter 把该 group 传给 `ring_flash_attn_func`；
- varlen llama3 adapter 把该 group 传给 `llama3_flash_attn_varlen_func`。

## 8.3 状态切换

Axolotl 有两类状态：

```text
全局状态:
  RING_ATTN_GROUP in src/axolotl/monkeypatch/ring_attn/patch.py:34-47
  Transformers _flash_attention_forward 被模块级替换
  Accelerate _prepare_cp / ParallelismConfig 校验被 monkey patch

上下文状态:
  SequenceParallelContextManager.hook_handles
  original_seq_len / pad_len
  _local_valid_tokens for eval loss correction
```

进入 `SequenceParallelContextManager`：

```text
__init__:
  register ring attn group
  patch HF attention
  记录 local_rank/local_world_size

__enter__:
  注册 forward pre-hook / post-hook

执行中:
  每次 forward 切 batch
  attention 通过全局 patch 走 ring backend

__exit__:
  移除 model hooks
  不恢复 attention / accelerate patch
```

这不是线程级隔离的局部状态；分布式训练通常是多进程，每个 rank 一个 Python 进程，所以进程间不共享全局变量。但在同一进程内连续加载多个模型或测试时，patch 污染是实际维护风险。

> 💡 **小结**
>
> * CP 的 shape 收益来自本地 sequence length 降到 `S/cp_size`。
> * rank/group 语义由 DeviceMesh 的 `cp` 维度决定，同组 rank 共享 batch、切 token。
> * zigzag / stripe 的难点是 shape 不变但 token layout 变化，所有状态和 gather 都要跟着改。

# 九、显存、通信与性能取舍

## 9.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | CP 不切参数；参数显存要靠 FSDP / ZeRO / TP / quantization。 |
| optimizer state | ❌ | CP 本身不处理 optimizer state。 |
| transformer 激活 | ✅ | 输入序列在 pre-hook 被切到 `[B, S/cp]`。 |
| Q/K/V 本地张量 | ✅ | 每 rank 只持有本地 sequence shard 的 Q/K/V。 |
| attention 中间 buffer | 部分 ✅ / 部分 ❌ | basic ring 只流动当前 K/V，但仍有通信 buffer；varlen llama3 会 all-gather KV head slice buffer。 |
| logits | ✅ | SFT 本地 loss 下 logits 是 `[B, S/cp, vocab]`。 |
| RL 输出 gather | ❌ | GRPO/EBFT `gather_outputs=True` 会 all-gather 输出，局部收益可能在输出端消失。 |
| 输入 batch | ❌ | 同一 CP group rank 先拿同一 batch，再各自切；batch 在 group 内冗余出现。 |
| 保存 state_dict | ❌ | CP 保存时可能 `detach().cpu()` clone，增加 CPU 内存峰值。 |

zigzag / stripe 对这张表的改变有限。它们不会进一步切参数，也不会减少 CP group 内通信轮数；它们主要减少负载不均造成的等待。

## 9.2 通信开销

当前 batch ring 每层 attention 近似：

```text
forward:
  world_size - 1 轮 K/V P2P send_recv
  每轮 K 和 V 各一个 isend/irecv
  causal 下 kernel 调用数按 rank 不均

backward:
  K/V 再旋转
  dK/dV 也通过 ring 归还/累积
  同样存在 causal 条件下的计算不均
```

源码依据：

- K/V P2P：`ring_flash_attn.py:26-63`；
- backward K/V 与 dK/dV 通信：`ring_flash_attn.py:97-150`；
- `RingComm` 使用 `dist.P2POp(isend/irecv)` 与 `dist.batch_isend_irecv`：`utils.py:121-150`。

Axolotl 自身还可能新增：

- `apply_sequence_parallelism()` 中 `num_items_in_batch` all-reduce（`sequence_parallel.py:150-165`）；
- eval loss correction 的两次 all-reduce：weighted loss 与 total valid tokens（`sequence_parallel.py:321-334`）；
- RL gather_outputs 的 all-gather shape + tensor（`sequence_parallel.py:393-415`）。

## 9.3 性能取舍

`batch_ring` 的取舍是“最小布局复杂度 + 明显 causal imbalance”。它适合先把长序列跑起来，但随着 `cp_size` 增大，高 rank 多算、低 rank 等待的比例会更明显。

`zigzag` / `stripe` 的取舍是“更复杂的数据布局 + 更均匀的 kernel 调度”。下游 README benchmark 中，batch API `fwd+bwd` 在 8xH800 上 basic ring 为 10.4 iter/s，zigzag 为 17.4，stripe 为 16.0；8xA100 上 basic ring 为 6.2，zigzag 为 10.6，stripe 为 9.75（`README.md:80-89`）。这支持“能缓解”的方向判断，但不能直接等同于 Axolotl e2e 收益，因为 Axolotl 尚未实现布局接入。

> 💡 **小结**
>
> * CP 主要省 sequence 维激活和 logits，不省参数与 optimizer state。
> * basic ring 通信轮数固定，但 causal kernel 调用数随 rank 增长。
> * zigzag / stripe 的价值是吞吐和等待时间，不是新的显存压缩机制。

# 十、配置项、边界条件与坑点

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `context_parallel_size` | `validation.py:1508-1559`；`trainer.py:621-640`；`train.py:205-220` | 开启 CP、写 env、注册 SP context | 需要 `ring_flash_attn`；world size / mesh 不匹配会在后续初始化报错。 |
| `ring_attn_func` | `validation.py:1563-1577`；`patch.py:186-211` | 选择 `varlen_llama3` 或 `batch_ring` | `batch_zigzag` / `batch_stripe` 当前非法。 |
| `sample_packing` | `validation.py:1522-1526`；`validation.py:1573-1577` | packing 默认 `varlen_llama3`；非 packing 默认 `batch_ring` | packing 下 `micro_batch_size > 1` 会报错。 |
| `micro_batch_size` | `validation.py:1522-1526`；`core/builders/base.py:588` | 传给 Trainer per-device batch | CP 下 effective global batch 不是简单 `world_size * micro_batch_size`。 |
| `flash_attention` | `validation.py:1517-1520` | CP 前置条件 | 关闭会直接报错。 |
| `heads_k_stride` | `patch.py:200-202`；下游 `llama3_flash_attn_varlen.py:84-118` | varlen llama3 按 KV head stride all-gather | 只对 `varlen_llama3` 有意义；batch ring adapter 不消费它。 |
| `dp_shard_size` / `dp_replicate_size` / `tensor_parallel_size` | `utils/distributed.py:299-370` | 影响 mesh 维度与 DataLoader batch 分组 | DDP + TP/CP 组合受 Accelerate 限制；mesh 顺序影响 group。 |
| `accelerator_config` | `core/builders/base.py:506-613` | 传给 Trainer / Accelerator | 若手动覆盖 dispatch/split 行为，要确认同 CP group 是否仍拿同一 batch。 |
| `fsdp_config.state_dict_type` | `train.py:294-334` | 保存 FULL/SHARDED state dict | CP 保存另有 CPU clone；FSDP 保存可能引入 gather/merge 成本。 |
| `trl.use_liger_loss` + GRPO + CP | `validation.py:721-728` | 直接报错 | GRPO + SP + Liger 当前不支持。 |
| async GRPO + CP | `core/trainers/grpo/__init__.py:39-47` | trainer class 选择时报错 | `sequence_parallel` 与 `async_grpo` 互斥。 |

最小可用 batch ring 配置大致是：

```yaml
flash_attention: true
context_parallel_size: 2
sample_packing: false
ring_attn_func: batch_ring  # 可省略，非 packing 默认就是它
```

而当前不可用配置是：

```yaml
ring_attn_func: batch_zigzag  # 会被 RingAttnFunc(...) 拒绝
ring_attn_func: batch_stripe  # 同上
```

> 💡 **小结**
>
> * 当前“更负载均衡 backend”不是隐藏配置，而是未接入能力。
> * `heads_k_stride` 主要服务 `varlen_llama3`，不是 batch zigzag/stripe 的性能旋钮。
> * CP 与 RL、Liger、async GRPO、FSDP 保存等组合有额外限制。

# 十一、测试、示例与覆盖缺口

## 11.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/test_context_parallel_batch_size.py:29-56` | CP 下 batch_size 按 `world_size // context_parallel_size` 缩放 | CPU 单测 mock 了 `ring_flash_attn`，不跑真实 attention。 |
| `tests/test_loaders.py:181-218` | `_get_parallel_config_kwargs()` 组合推导 | 覆盖 TP/CP/DP/FSDP kwargs，不覆盖真实 DeviceMesh 通信。 |
| `tests/monkeypatch/test_trainer_context_parallel_patch.py:36-65` | Trainer CP guard patch 与幂等性 | 证明 patch 改了源码 guard，不证明 CP 训练正确性。 |
| `tests/e2e/multigpu/patched/test_sp.py:102-137` | SP e2e 计划覆盖 `varlen_llama3` 与 `batch_ring` | 整个测试被 skip，原因是 `ring_flash_attn w transformers imports unmaintained upstream`。 |
| `tests/e2e/multigpu/solo/test_grpo.py:144` / `test_gdpo.py:23` | GRPO/GDPO CP e2e 配置 | 类级 skip：`flaky vllm tests in modal`。 |
| `examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml:6-32` | FSDP + TP + CP 示例 | 使用 sample packing，默认会走 `varlen_llama3`，不是 batch ring。 |
| `examples/alst/llama3-8b-fsdp2-alst.yaml:18-43` | 超长上下文 + CP + CCE 示例 | `sample_packing: true`，同样偏 `varlen_llama3`。 |

## 11.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| `batch_zigzag` / `batch_stripe` 配置非法但 schema 描述提到 | ❌ | 用户按描述配置会报错，文档/源码认知不一致。 |
| batch ring rank 计算不均的性能量化 | ❌ | 无法在 Axolotl e2e 中判断 cp_size 增大后的 idle/等待比例。 |
| zigzag / stripe Axolotl layout 接入 | ❌ | 若只改 mapping，可能输出 token 顺序或 causal 语义错误。 |
| `AllGatherWithGrad` 对非连续 layout 的 inverse permutation | ❌ | RL 输出 gather 后 token 顺序错，梯度切片也错。 |
| CP 保存 / resume e2e | 不充分 | state_dict CPU clone 与 FSDP/ZeRO/PEFT 组合可能有隐藏峰值或兼容问题。 |
| 多机 CP group / P2P 性能 | ❌ | Ring P2P 对跨节点链路敏感，可能严重拖慢。 |
| patch 恢复 | 部分单测手动恢复 Trainer 方法 | attention patch / Accelerate patch 在生产上下文不恢复，测试污染风险。 |

> 💡 **小结**
>
> * 当前测试更多覆盖配置推导和 patch 可执行性，不覆盖真实 GPU ring 性能。
> * SP 训练 e2e 和 GRPO/GDPO CP e2e 都被 skip，风险点需要人工谨慎评估。
> * zigzag / stripe 在下游包有测试，但没有 Axolotl 集成测试。

# 十二、局限性与已知优化点

## 12.1 硬约束

- `context_parallel_size > 1` 必须 `flash_attention: true`（`validation.py:1517-1520`）。
- `sample_packing=True` 且 CP 开启时，`micro_batch_size > 1` 会报错（`validation.py:1522-1526`）。
- `ring_flash_attn` 必须安装；当前环境未安装时 schema 校验会失败（`validation.py:1544-1550`）。
- 当前枚举不支持 `batch_zigzag` / `batch_stripe`（`enums.py:100-108`）。
- `heads_k_stride` 要求 KV heads 可整除；下游 varlen llama3 里有 `assert nheads_k % heads_k_stride == 0`（`llama3_flash_attn_varlen.py:84-87`）。
- GRPO + SP + Liger loss 被显式禁止（`validation.py:721-728`）。
- async GRPO 与 sequence parallel 互斥（`core/trainers/grpo/__init__.py:39-47`）。

## 12.2 维护成本

- monkey patch 替换 Transformers 模块级 `_flash_attention_forward`，且 context exit 不恢复（`sequence_parallel.py:238-245`）。
- batch adapter 依赖 Transformers 函数签名，靠 `check_params()` 做防线（`adapters/batch.py:176-183`）。
- Accelerate 的 `_prepare_cp` / `ParallelismConfig` 被 patch，升级 Accelerate 后语义可能变化。
- zigzag / stripe 若接入，需要同时改 schema、enum、adapter、pre-hook、post-hook、测试，不能局部改一处。

## 12.3 性能瓶颈

- basic batch ring causal 下高 rank 计算更多，低 rank 等待更多。
- 每层 attention 都有 ring 通信；跨节点 P2P 对网络非常敏感。
- `num_items_in_batch` all-reduce 位于 `apply_sequence_parallelism()` 的 per-key loop 中（`sequence_parallel.py:150-165`），当 batch 中多个 tensor 满足循环条件时可能重复执行，属于值得审视的小通信开销。
- RL `gather_outputs=True` 会把输出 all-gather 回完整序列，削弱 logits / output 侧显存收益。
- 保存时 CP 下 state_dict tensor `detach().cpu()`，可能增加 CPU 内存峰值（`core/trainers/base.py:812-823`）。

## 12.4 已知优化点

源码里最直接的优化点就是 `sequence_parallel.py:22-23` 的 TODO：实现 zigzag / stripe patterns。结合下游 `ring-flash-attn`，一个完整接入至少需要：

1. 在 `RingAttnFunc` 中启用 `BATCH_ZIGZAG` / `BATCH_STRIPE`；
2. 在 batch adapter 中 import `zigzag_ring_flash_attn_func` / `stripe_flash_attn_func` 并加入 mapping；
3. 在 `register_ring_attn_from_device_mesh()` 中处理新枚举；
4. 在 `apply_sequence_parallelism()` 中按 backend 选择 layout：连续 / zigzag / stripe；
5. 为 `position_ids`、`labels`、`attention_mask`、`logits_to_keep` 同步 permutation；
6. 为 `AllGatherWithGrad` 或 post-hook 实现 inverse permutation；
7. 增加 GPU e2e：至少比较 non-CP baseline、batch_ring、batch_zigzag、batch_stripe 的 loss parity、token 顺序、显存与吞吐；
8. 明确 sample packing 下是否支持 varlen zigzag，或者继续推荐 `varlen_llama3`。

> 💡 **小结**
>
> * 当前最大性能痛点是 batch ring causal 的 rank 计算不均，而不是 CP group 构建。
> * 下游有可借鉴 backend，但 Axolotl 接入需要完整 layout 工程，而非简单取消注释。
> * 测试缺口集中在真实 GPU e2e、保存/resume、patch 恢复和性能量化。

# 小结与展望

Axolotl 的 `zigzag / stripe 或更负载均衡 ring backend` 现状，可以用四个关键词概括。

## 关键词一：配置承诺尚未兑现

`config.py` 的描述提到 `batch_zigzag` / `batch_stripe`，测试里也有注释参数，但 `RingAttnFunc` 只启用 `varlen_llama3` 与 `batch_ring`。因此从用户视角看，这不是隐藏功能，而是未接入功能。

## 关键词二：连续切分是当前主路径

Axolotl 通过 `SequenceParallelContextManager` 在 forward pre-hook 中把 `[B, S]` 连续切成 `[B, S/cp]`。这条路径简单、直观、适配 basic ring，但正是 causal rank 不均的来源之一。

## 关键词三：basic ring 的不均来自 `step <= rank`

下游 `ring_flash_attn_func` 在 causal 模式下只在 `step <= comm.rank` 时执行 FlashAttention kernel。于是 rank0、rank1、rank2、rank3 的 kernel 调用数近似 1、2、3、4。通信轮数仍在，整体 step 等最慢 rank。

## 关键词四：zigzag / stripe 是通信换显存之后的调度优化

CP 已经通过切 sequence 换取显存；zigzag / stripe 进一步试图在不改变总语义的前提下重排 token，让每个 rank 的计算更均匀。它适合长序列、非 sample packing batch API、NVLink 等高速互联、basic ring 已出现明显 rank idle 的场景；不适合希望零改数据 layout、只靠配置无痛切换的场景。

与替代方案相比：

- 继续使用 `batch_ring`：最稳、接入最少，但 causal 负载不均；
- 使用 `varlen_llama3`：sample packing 默认路径，通信形态不同，适合 packed 长序列，但不是 batch zigzag；
- 接入 `zigzag` / `stripe`：有望提高 batch causal 吞吐，但需要重做 layout、gather、测试；
- 用 FSDP / TP / CCE / activation checkpointing：解决的是参数、logits 或重计算问题，不能直接替代 CP backend 的 rank 调度优化。

后续最值得继续走读的方向，是把 `apply_sequence_parallelism()` 的连续切分改造成 backend-aware layout manager：它不仅决定 `input_ids` 怎么切，还要负责 `position_ids`、`labels`、`attention_mask`、`logits_to_keep`、post-hook gather 和 backward slice 的一致性。只有这层补齐后，`zigzag` / `stripe` 才能从下游包里的“更均衡算法”，变成 Axolotl 用户真正可用的“更均衡 ring backend”。
