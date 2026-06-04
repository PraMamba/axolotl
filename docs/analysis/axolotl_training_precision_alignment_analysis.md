# Axolotl 训练精度对齐体系源码与 Commit History 深度分析

> **分析日期**: 2026-06-03  
> **分析分支**: source_code_analysis (基于 main 0ef1b301)  
> **分析方法**: 源码全量关键词搜索 + Git history 双线交叉验证  
> **分析工具**: 3 个并行子代理 (Explore/Architect-Reviewer/Code-Reviewer) + 主代理综合

---

## 0. Executive Summary

| 维度 | 评估 |
|---|---|
| **项目名称** | Axolotl — 开源 LLM 微调框架 |
| **总体判断** | **局部具备**（2.5/5），核心训练流程精度管理基本合理，但缺乏系统性精度对齐基础设施 |
| **最强能力** | (1) 自定义 Triton kernel 的算子级数值正确性测试（259 个 `assert_close` 断言）；(2) FSDP2 下的 `fp32_norms` 混合精度分层管理；(3) TensorBoard loss 回归守卫函数 |
| **最大短板** | (1) 无 golden value / 精确 loss 回归测试；(2) 无 `torch.use_deterministic_algorithms()`；(3) 无单卡 vs 多卡数值一致性对比测试；(4) 无 tensor dump / activation compare 调试工具 |
| **最值得借鉴的源码模块** | `src/axolotl/utils/fp32_norms.py`, `tests/e2e/utils.py:check_tensorboard_loss_decreased()`, `tests/e2e/multigpu/_fp32_norms_dtype_capture.py` (plugin-based dtype dump 测试模式) |
| **最值得研究的 commits/PR** | `b05ab9a0` (fp32_norms), `0ae06d75` (nanmean CP fix), `2501c1a6` (gradient accumulation loss fix), `da97285e` (MoE gate fp32), `24bb2c9c` (CP div-by-zero) |
| **是否适合作为训练精度对齐基础设施参考** | **部分适合**。Axolotl 的 loss threshold 回归测试、kernel 级 allclose 测试、fp32_norms 是好的设计模式，但不具备完整的精度对齐体系（无 golden values、无 tensor dump、无 deterministic mode、无跨并行策略一致性验证）。适合作为"训练框架中精度守护的自然演化案例"来学习，但不适合直接作为精度对齐基础设施的蓝图。 |

---

## 1. 项目训练流程与精度相关架构总览

### 1.1 训练入口链

```
CLI: src/axolotl/cli/train.py:do_cli() → do_train()
  → src/axolotl/train.py:train()
    → setup_model_and_trainer() → setup_trainer()
      → src/axolotl/utils/trainer.py:690: setup_trainer()
        → HFCausalTrainerBuilder.build() 或 HFRLTrainerBuilder.build()
    → trainer.train(resume_from_checkpoint=...)  # 委托给 HF Trainer
```

**关键发现**: Axolotl 不直接拥有训练循环。forward → backward → optimizer update 全部委托给 HuggingFace `Trainer._inner_training_loop`。Axolotl 通过以下方式影响精度：
- 配置注入（precision fields → TrainingArguments）
- Monkeypatch（替换 loss 函数、attention 实现、TRL utils）
- Callback（loss watchdog、profiler、dtype capture）
- Mixin（scheduler override、RNG state loader、checkpoint save）

### 1.2 配置系统精度传递路径

```
YAML config
  → DictDefault (src/axolotl/utils/dict.py:6)
    → validate_config() (Pydantic V2 schema at src/axolotl/utils/schemas/config.py)
      → AxolotlConfigWCapabilities (GPU 能力检测: bf16, tf32, fp8)
    → normalize_config() (src/axolotl/utils/config/__init__.py)
      → resolve_dtype()
      → tf32 backend settings
      → seed auto-default to 42
    → TrainerBuilder._configure_precision_settings()
      → training_args_kwargs["bf16"], ["fp16"], ["tf32"]
    → prepare_optim_env()
      → ACCELERATE_MIXED_PRECISION env var
```

**精度相关配置字段** (`src/axolotl/utils/schemas/config.py`):

| 字段 | 类型 | 默认值 | 行号 | 精度影响 |
|---|---|---|---|---|
| `seed` | `int \| None` | `None` → 自动设为 42 | 482 | 随机性控制 |
| `bf16` | `Literal["auto"] \| bool \| None` | `None` | 554 | AMP 混合精度 |
| `fp16` | `bool \| None` | `None` | 560 | AMP 混合精度 |
| `fp8` | 通过 TorchAO | `None` | 563 | FP8 训练 |
| `tf32` | `Literal["auto"] \| bool \| None` | `"auto"` | 587 | 矩阵乘法精度 |
| `fp32_norms` | `bool` | `False` | 980 | FSDP2 下 norm 层保持 FP32 |
| `loss_watchdog_threshold` | `float \| None` | `None` | 517 | Loss 异常检测 |
| `gradient_checkpointing` | `bool` | `False` | 595 | 激活重算精度 |
| `fsdp_version` | `int \| None` | `None` | 973 | FSDP1/2 分布式策略 |

### 1.3 混合精度配置路径

**AMP 配置** (`src/axolotl/utils/trainer.py:680-687`):
```python
if cfg.fp8:
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
elif (cfg.bf16 == "auto" and is_torch_bf16_gpu_available()) or cfg.bf16 is True:
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
elif cfg.fp16:
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
```

**TF32 控制** (`src/axolotl/utils/config/__init__.py:92-100`):
```python
if cfg.tf32 is True:
    torch.set_float32_matmul_precision("high")
    if is_torch_greater_or_equal("2.9.0"):
        torch.backends.fp32_precision = "tf32"  # 新 API
    else:
        torch.backends.cuda.matmul.allow_tf32 = True  # 旧 API
```

**Loss scaling**: Axolotl 本身不管理 GradScaler / loss scaling，完全委托给 Accelerate（fp16 路径）或 DeepSpeed（`deepspeed_configs/` 中的 `loss_scale: 0`，即 dynamic scaling）。

### 1.4 分布式训练架构

| 并行策略 | 配置入口 | 实现位置 | 精度测试 |
|---|---|---|---|
| FSDP1 | `fsdp_config` + `fsdp_version: 1` | HF Accelerate | `tests/e2e/multigpu/test_fsdp1.py` (loss threshold) |
| FSDP2 | `fsdp_config` + `fsdp_version: 2` | `monkeypatch/accelerate/fsdp2.py` | `test_fsdp2.py`, `test_fsdp2_fp32_norms.py` |
| DeepSpeed | `deepspeed:` 配置块 | HF Accelerate + DS config | `test_llama.py` (multigpu, loss threshold) |
| Tensor Parallel | `tensor_parallel_size: N` | `monkeypatch/accelerate/parallelism_config.py` | `test_tp.py` (loss decreased, **currently skipped**) |
| Context Parallel | `context_parallel_size: N` | `monkeypatch/ring_attn/`, `utils/ctx_managers/sequence_parallel.py` | `test_sp.py` (loss threshold) |
| Expert Parallel | `parallelism_config.ep_size` | `monkeypatch/accelerate/parallelism_config.py` | `test_expert_parallel.py` (unit test only) |
| DDP | `ddp: true` (默认 multi-GPU) | PyTorch DDP | `test_llama.py` (multigpu) |

### 1.5 Checkpoint 系统

- **保存**: `AxolotlTrainer._save_checkpoint()` → `src/axolotl/core/trainers/base.py:795`，保存 tokens state 后委托 HF Trainer（保存模型权重、optimizer state、scheduler state、RNG state）
- **RNG 状态恢复**: `RngLoaderMixin._load_rng_state()` → `src/axolotl/core/trainers/mixins/rng_state_loader.py:29`，恢复 Python random、NumPy、CPU torch、CUDA RNG（per-rank）
- **关键问题**: FSDP2 下 optimizer state 保存可能静默失败（`CheckpointSaveMixin._save_optimizer_and_scheduler()` 的 try/except，`src/axolotl/core/trainers/mixins/checkpoints.py:13-23`），仅记录 warning
- **无完整性校验**: 没有 hash check、tensor shape 验证、checkpoint round-trip test

---

## 2. 精度对齐能力矩阵

| 能力项 | 是否具备 | 源码证据 | Commit/PR 证据 | 成熟度 | 备注 |
|---|---|---|---|---|---|
| 配置一致性扫描 | 间接存在 | Pydantic schema 验证 (`utils/schemas/`) | 多次 config validation 改进 | 2 | 仅验证字段合法性，不验证精度一致性 |
| 随机种子/RNG 控制 | 明确存在 | `seed` config, `torch_manual_seed` fixture, `RngLoaderMixin` | `validator auto-set seed=42` | 2 | 无 `torch.use_deterministic_algorithms()`，无 CUBLAS 配置 |
| 数据加载顺序确定性 | 间接存在 | `worker_init_fn=seed_worker` (`base.py:303`) | — | 1 | 仅设置了 worker seed，无 DistributedSampler 确定性验证 |
| 初始权重一致性 | 未发现 | — | — | 0 | 无初始化权重 checksum / snapshot 机制 |
| 单步 forward loss 对齐 | 间接存在 | `check_tensorboard_loss_decreased` (threshold) | `2501c1a6` gradient accum fix | 1 | 仅有 loss threshold 回归，无精确单步 loss 对齐 |
| activation dump/compare | 未发现 | — | — | 0 | 无 activation dump 工具，仅有 gradient checkpointing offload |
| gradient dump/compare | 间接存在 | `test_batch_flattening.py:268` gradient correctness test | — | 1 | 仅 GRPO padded vs flattened 路径的梯度对比 |
| optimizer state 对齐 | 间接存在 | `_save_optimizer_and_scheduler` mixin | FSDP2 opt state save 问题 | 1 | 有保存/恢复，但 FSDP2 下可能静默丢失 |
| scheduler/lr curve 对齐 | 间接存在 | `create_scheduler` mixin, `trainer/lr.py` patch | `trainer/lr.py` DS fp16 fix | 1 | 有 DS loss scale → scheduler 对齐修复 |
| loss curve golden regression | 间接存在 | `check_tensorboard_loss_decreased()` | `6130e40c` 21 文件 tag 修复 | 2 | 仅 ratio + 绝对阈值，无精确 golden values |
| mixed precision 对齐 | 明确存在 | `fp32_norms.py`, `chunked.py` fp32 upcast | `b05ab9a0` fp32_norms, `830e9f7e` tf32 auto | 3 | fp32_norms 是体系化设计，CE upcast 是正确实践 |
| FP16/BF16/FP8 数值稳定性 | 明确存在 | FP8 smoke test, bf16/fp16 auto config | `42d4732a` KD loss fp32, `da97285e` MoE gate | 2 | FP8 仅 NaN 检查，无 vs bf16 baseline |
| TF32 控制 | 明确存在 | `normalize_config()` tf32 backend, version guard | `fc2d63ee` tf32 API 2.9+, `830e9f7e` auto | 3 | 有版本兼容，有 auto 检测 |
| NaN/Inf/overflow 检测 | 明确存在 | `nan_to_num`, `isnan`/`isinf` checks, `loss_watchdog` | `0ae06d75` nanmean fix | 2 | 运行时检测（非全局 anomaly detection） |
| checkpoint resume 一致性 | 间接存在 | `RngLoaderMixin`, per-rank RNG restore | `rng_state_loader.py` 修复 | 2 | 有 RNG 恢复，但无 round-trip 一致性测试 |
| data parallel correctness | 间接存在 | DDP e2e tests with loss thresholds | — | 2 | 仅 smoke test，无 vs 单卡对比 |
| tensor parallel correctness | 间接存在 | `test_tp.py` (但目前 **skipped**) | — | 1 | TP 测试因 tied weights 问题被跳过 |
| pipeline parallel correctness | 未发现 | — | — | 0 | Axolotl 不直接支持 PP |
| sequence parallel correctness | 明确存在 | `test_sp.py`, `sequence_parallel.py` all_reduce | `24bb2c9c` div-by-zero, `97a4f285` eval | 2 | 有 e2e test，但已知 loss 日志不准确（TODO） |
| expert parallel / MoE correctness | 间接存在 | EP-aware grad norm clip, ScatterMoE allclose tests | `da97285e` gate fp32 | 2 | kernel 级测试强，端到端测试弱 |
| collective communication correctness | 间接存在 | `_ep_aware_clip_grad_norm` with fp32 accumulation | — | 1 | 仅 EP grad norm，无通用 collective 验证 |
| CI 精度回归测试 | 明确存在 | `multi-gpu-e2e.yml`, loss threshold tests | `6130e40c` 统一 tag | 2 | 双周运行，threshold 松散 |
| 自动化二分定位能力 | 未发现 | — | — | 0 | — |
| 跨硬件/跨后端对齐能力 | 间接存在 | CI matrix (CUDA 12.8/13.0, PyTorch 2.9/2.10) | — | 1 | 仅兼容性测试，非数值对齐 |

---

## 3. 源码证据地图

### 3.1 Kernel 级数值正确性测试（最成熟领域）

**文件**: `tests/test_triton_kernels.py` (~30 `assert_close` 调用)

测试模式：针对每个自定义 Triton kernel，构造随机输入，分别用 Triton 实现和 PyTorch 参考实现计算，然后用 `torch.testing.assert_close(result, expected, atol=X, rtol=X)` 对比。

覆盖的 kernel:
- LoRA forward/backward (`atol=1e-4`)
- DoRA magnitude/direction (`atol=1e-4`)
- SwiGLU forward/backward (`atol=1e-4`)
- GeGLU forward/backward (`atol=5e-2` for bf16)
- Selective log softmax forward/backward (`atol=1e-5`)

**文件**: `tests/test_ebft_kernels.py` (~25 `assert_close` 调用)

覆盖: chunk_softmax, gated_linear, fused_cosine_similarity, partial_cross_entropy, strided structured masking

**文件**: `tests/e2e/kernels/test_lora_features.py:83`
```python
def _compare_tensors(a, b, name="", atol=1e-2, rtol=1e-2):
    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    )
```

**文件**: `tests/monkeypatch/test_gemma4_fused_attn.py:154`
```python
cos_sim = torch.nn.functional.cosine_similarity(fused_out.flatten(), stock_out.flatten(), dim=0)
assert cos_sim > 0.999, f"layer {layer_idx} fused vs stock cosine_sim={cos_sim:.6f}"
```

### 3.2 Loss 回归守卫（主要防御线）

**文件**: `tests/e2e/utils.py:204-301` — `check_tensorboard_loss_decreased()`

这是 Axolotl 最重要的精度回归防御机制。设计理念：
1. 读取 TensorBoard 日志获取 per-step loss 序列
2. 计算 initial window 和 final window 的均值
3. 断言 `final <= initial * max_loss_ratio`（默认 0.95，即 loss 至少下降 5%）
4. 可选绝对上界 `max_initial` / `max_final`（防止 loss 从异常高的起点"下降"）

使用该函数的测试文件（~15 个）:
- `test_falcon.py`, `test_phi.py`, `test_mistral.py`, `test_mixtral.py` (solo)
- `test_fsdp2.py`, `test_dist_muon_fsdp2.py`, `test_llama.py` (multigpu)
- `test_sp.py` (sequence parallel)

### 3.3 FP32 Norms — 最体系化的精度对齐功能

**文件**: `src/axolotl/utils/fp32_norms.py`

设计：在 FSDP2 混合精度训练中，将 RMSNorm/LayerNorm 模块保持在 FP32，而其他参数在 BF16。

关键函数:
- `_matches_norm_class(module, patterns)`: 按类名后缀或全限定名匹配 norm 模块
- `shard_norms_fp32(model, ...)`: 对匹配的 norm 模块应用 `MixedPrecisionPolicy(param_dtype=float32, reduce_dtype=float32)`

**测试**:
- `tests/test_fp32_norms.py` — 12 个单元测试，覆盖 suffix/qualified 匹配、meta-device、buffer dtype 保持
- `tests/e2e/multigpu/test_fsdp2_fp32_norms.py` — 多 GPU e2e 测试，通过自定义 plugin 在 step 1 后 dump 所有参数的 dtype 到 JSON，外层 pytest 断言 norm 参数为 float32、非 norm 参数为 bfloat16

**Plugin-based dtype dump 模式** (`tests/e2e/multigpu/_fp32_norms_dtype_capture.py`):
```python
class _DtypeCaptureCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step != 1 or model is None:
            return
        norm_dtypes, non_norm_dtypes = {}, {}
        for name, param in model.named_parameters():
            if "norm" in name.lower():
                norm_dtypes[name] = str(param.dtype).removeprefix("torch.")
            else:
                non_norm_dtypes[name] = str(param.dtype).removeprefix("torch.")
        json.dump({"norms": norm_dtypes, "non_norms": non_norm_dtypes}, fout)
```

**启发**: 这个 plugin → callback → dump → 外层断言 的模式是可复用的精度对齐测试设计模式。

### 3.4 Chunked Cross-Entropy FP32 Upcast

**文件**: `src/axolotl/monkeypatch/loss/chunked.py:24-35`
```python
def compute_cross_entropy(self, logits, labels, normalize=True):
    """Upcast logits to fp32 and compute cross entropy loss."""
    return F.cross_entropy(logits.float(), labels, ignore_index=self.ignore_index, reduction="sum")
```

这是从 torchtune 移植的做法。逐 chunk upcast 到 FP32 计算 CE loss，避免 BF16 下的数值不稳定。

### 3.5 EP-Aware Gradient Norm Clipping

**文件**: `src/axolotl/monkeypatch/accelerate/parallelism_config.py:193-241`

当 Expert Parallel + FSDP 组合使用时，不同参数可能在不同的 DeviceMesh 上。标准的 `clip_grad_norm_` 无法跨 mesh 正确计算。修复：
1. 在 FP32 下计算 local p-norm
2. 通过 `dist.all_reduce` 聚合
3. 计算全局 total_norm 并 clip

### 3.6 Loss Watchdog

**文件**: `src/axolotl/utils/callbacks/__init__.py:57-83`

运行时 loss 异常检测：如果连续 `patience` 步 loss 超过 `threshold`，停止训练。这不是精度对齐工具，但可以在 loss 爆炸时快速发现问题。

### 3.7 RNG State Loader Mixin

**文件**: `src/axolotl/core/trainers/mixins/rng_state_loader.py`

修复了 upstream transformers 在 PyTorch 2.6+ 下 RNG state 反序列化的 bug（`safe_globals` context manager）。恢复 Python random、NumPy、CPU torch、CUDA RNG 四种状态。

### 3.8 Profiler Callback

**文件**: `src/axolotl/utils/callbacks/profiler.py:17-100`

提供 `torch.profiler` Chrome trace 和 CUDA memory snapshot。这不是精度对齐工具，但可以辅助排查异常的内存使用模式（可能暗示精度问题，如意外的 FP32 复制）。

### 3.9 OpenTelemetry Metrics

**文件**: `src/axolotl/utils/callbacks/opentelemetry.py`

导出训练指标到 Prometheus/OTEL，包括 `grad_norm_gauge`。可用于监控 gradient norm 变化趋势，辅助精度问题排查。

---

## 4. Commit / PR / Issue 历史演进时间线

### 时间线概览

```
2023-08   96deb6bd  QLoRA norm/gate/embed fp32 模式建立（基础性）
2024-01   da97285e  MoE gate 保持 fp32 + mixtral test（首个闭环案例）
2024-06   9c1af1a9  eval_sample_packing 必须匹配 sample_packing
2024-10   2501c1a6  gradient accumulation loss fix + test_packing_loss（闭环）
2025-01   42d4732a  KD loss FP32 upcast（无测试）
2025-02   68e97d03  KD Triton kernel overflow chunking（无测试）
2025-04   954b989e  SP 下 loss 日志不准确 warning + TODO
2025-07   208fb7b8  FP8 混合精度训练 + smoke test
2025-08   0ae06d75  nanmean CP eval loss fix + 测试（闭环）
2026-01   24bb2c9c  CP division-by-zero fix（无测试）
2026-02   97a4f285  CP eval loss weighted correction
2026-03   fc2d63ee  tf32 API 2.9+ 适配
2026-03   830e9f7e  tf32 auto 检测
2026-03   f56efdb4  eval sample packing high loss fix
2026-03   6130e40c  21 个测试文件统一 loss tag（闭环）
2026-05   b05ab9a0  fp32_norms for FSDP2 + 完整测试（闭环）
2026-05   5352d41d  多模态 loss masking 系统化
```

### 详细分析（Top 10）

---

**时间**: 2026-05  
**Commit / PR**: `b05ab9a0` — "feat(fsdp2): add fp32_norms for keeping RMSNorm/LayerNorm in fp32 (#3670)"  
**涉及文件**: `utils/fp32_norms.py` (新), `utils/schemas/config.py`, `monkeypatch/accelerate/fsdp2.py`, `tests/e2e/multigpu/test_fsdp2_fp32_norms.py` (新), `tests/test_fp32_norms.py` (新)  
**问题背景**: FSDP1 强制所有参数使用相同 dtype，无法实现"norm 层 FP32 + 其他 BF16"。FSDP2 的 per-module MixedPrecisionPolicy 使此成为可能。  
**修改内容**: 新增 `fp32_norms: true` 配置，自动识别 RMSNorm/LayerNorm 并应用 FP32 policy；支持自定义 `fp32_norm_classes` 列表。  
**新增测试**: 12 个单元测试 + 1 个 multi-GPU e2e 测试（plugin dump dtype + assert）  
**影响范围**: 所有使用 FSDP2 + BF16 的训练配置  
**启发**: **最佳实践案例**。完整的 config → source → test 闭环，plugin-based 测试模式可复用。

---

**时间**: 2025-08  
**Commit / PR**: `0ae06d75` — "use nanmean for loss aggregation (CP fix) (#3033)"  
**涉及文件**: `monkeypatch/transformers/trainer_loss_calc.py` (新), `loaders/patch_manager.py`  
**问题背景**: Context Parallelism 下，某些 rank 可能收到全 padding 的输入，产生 NaN 局部 loss。标准 `.mean()` 将 NaN 传播到全局 loss。  
**修改内容**: Monkeypatch HF Trainer 的 `evaluation_loop` 和 `maybe_log_save_evaluate`，用 `nanmean` 替代 `mean`。  
**新增测试**: `tests/monkeypatch/test_trainer_loss_calc.py`（验证 patch 可应用性）  
**启发**: 分布式训练下 loss 聚合的数值正确性问题。NaN 传播是常见陷阱。

---

**时间**: 2024-10  
**Commit / PR**: `2501c1a6` — "Fix: Gradient Accumulation issue (#1980)"  
**涉及文件**: `core/trainer_builder.py`, `train.py`, `tests/e2e/test_packing_loss.py` (新)  
**问题背景**: HF Transformers/TRL 更改了 loss API，新增 `num_items_in_batch` 参数，从 per-sample 切换到 per-token loss 归一化。Axolotl 未转发此参数，导致 gradient accumulation 下梯度缩放错误。  
**新增测试**: `test_packing_loss.py`（首个 loss threshold 冒烟测试）  
**启发**: 上游 API 变更可能静默改变数值行为。需要回归测试来捕获。

---

**时间**: 2024-01  
**Commit / PR**: `da97285e` — "keep gate in fp32 for 16 bit loras (#1105)"  
**涉及文件**: `utils/models.py`, `train.py`, `tests/e2e/test_mixtral.py`  
**问题背景**: MoE gate 层在 BF16 下导致训练不稳定（gate routing 精度敏感）。  
**修改内容**: 扩展 norm-in-fp32 规则到 gate 层。  
**新增测试**: `test_mixtral.py` dtype assertion  
**启发**: 模型特定的关键模块可能需要更高精度。

---

**时间**: 2026-01  
**Commit / PR**: `24bb2c9c` — "fix(sequence_parallel): prevent division by zero in num_items_in_batch calculation"  
**涉及文件**: `utils/ctx_managers/sequence_parallel.py`  
**问题背景**: `int()` 截断 `global_valid_tokens` 在 CP 大组 + `train_on_inputs=False` 时可能产生 0，导致 loss 变为 Inf/NaN。  
**修改内容**: 保持 float + `clamp(min=1.0)` + fail-fast validation  
**新增测试**: **无**  
**启发**: 分布式 loss 归一化中的 division-by-zero 是常见精度问题。

---

**时间**: 2026-03  
**Commit / PR**: `6130e40c` — "fix flaky tests; should be using train loss from final step rather than final avg train loss"  
**涉及文件**: 21 个测试文件  
**问题背景**: TensorBoard tag `"train/train_loss"` 是 epoch 聚合，`"train/loss"` 是 per-step。测试使用错误 tag 导致间歇性失败。  
**修改内容**: 统一所有测试使用 per-step tag + `check_tensorboard_loss_decreased()`  
**启发**: 测试基础设施的一致性直接影响回归检测能力。

---

**时间**: 2026-03  
**Commit / PR**: `fc2d63ee` — "use new tf32 APIs for torch 2.9+ (#3467)"  
**涉及文件**: `utils/config/__init__.py`  
**问题背景**: PyTorch 2.9 新增 `torch.backends.fp32_precision = "tf32"` 替代旧 API。  
**启发**: 精度相关的上游 API 变更需要跟踪。

---

**时间**: 2025-01  
**Commit / PR**: `42d4732a` — "kd loss needs to be calculated in full precision"  
**涉及文件**: `integrations/kd/topk_logprob/forward_kl.py`  
**问题背景**: KL 散度在低精度下数值不稳定。  
**修改内容**: 在 KD loss 计算前 `.float()` upcast  
**新增测试**: **无**  
**启发**: Loss 函数的精度是关键。

---

**时间**: 2026-02  
**Commit / PR**: `97a4f285` — "fix: saving state dict and eval for Context Parallel (#3382)"  
**涉及文件**: `core/trainers/base.py`, `utils/ctx_managers/sequence_parallel.py`  
**问题背景**: CP 下 eval loss 不正确，因为 `num_items_in_batch` 未在 eval 时正确处理。  
**修改内容**: forward hook 追踪 `_local_valid_tokens`，eval 时 weighted all-reduce  
**启发**: 分布式 eval 和 train 的 loss 计算逻辑需要分别验证。

---

**时间**: 2025-04  
**Commit / PR**: `954b989e` — "log warning re: logged losses / gradient scaling per rank"  
**涉及文件**: `utils/schemas/config.py`  
**问题背景**: SP 模式下日志中的 loss 值不等于真实全局 loss，因为 per-rank gradient scaling 未被修正。  
**修改内容**: 仅添加 `LOG.warning` 和 TODO  
**新增测试**: **无**  
**启发**: **未关闭的精度问题**。已知但未修复。

---

## 5. 典型精度问题案例复盘

### Case 1: Gradient Accumulation 下 Loss 错误 (`2501c1a6`)

**现象**: 使用 gradient accumulation + sample packing 时，训练 loss 偏高，学习效果差。  
**根因**: HF Transformers 更改了 `compute_loss` API，引入 `num_items_in_batch` 进行 per-token 归一化。Axolotl 未转发此参数，导致 gradient accumulation 的多个 micro-batch 之间 loss 归一化不一致。  
**定位方式**: 通过对比 upstream 代码变更发现。  
**修复**: 在 `compute_loss` 中正确转发 `num_items_in_batch`。  
**新增测试**: `tests/e2e/test_packing_loss.py` — loss threshold 冒烟测试。  
**借鉴**: 上游 API 数值语义变更是隐蔽的精度问题来源，需要 loss 回归测试来捕获。

### Case 2: Context Parallel 下 NaN Loss (`0ae06d75`)

**现象**: CP 训练中 eval loss 显示 NaN。  
**根因**: 某些 rank 收到全 padding 输入（labels 全为 -100），局部 loss 为 NaN。标准 `.mean()` 将 NaN 传播到聚合结果。  
**定位方式**: 在多 GPU 环境中观察 per-rank loss 日志。  
**修复**: Monkeypatch HF Trainer 使用 `nanmean`。  
**新增测试**: `test_trainer_loss_calc.py`（验证 patch 可应用）  
**借鉴**: 分布式训练中 NaN 传播是系统性问题，需要在所有聚合点使用 `nanmean` 或显式 NaN 过滤。

### Case 3: BF16 下 MoE Gate 不稳定 (`da97285e`)

**现象**: MoE 模型在 BF16 训练时 loss 异常振荡。  
**根因**: MoE gate 层的 routing 对精度敏感，BF16 的有限尾数导致 expert 选择不稳定。  
**定位方式**: 观察 gate 输出的分布发现异常。  
**修复**: gate 层保持 FP32。  
**新增测试**: `test_mixtral.py` dtype 断言。  
**借鉴**: 并非所有模块都应使用相同精度。需要识别精度敏感的关键模块（norm、gate、loss）。

### Case 4: CP Division-by-Zero → Inf/NaN Loss (`24bb2c9c`)

**现象**: `train_on_inputs=False` + 大 CP 组时出现 Inf loss。  
**根因**: `int()` 截断 `global_valid_tokens` 在 token 数很少时可能变为 0，导致 `loss = loss * local_valid / global_valid` 中除零。  
**修复**: 保持 float 并 clamp(min=1.0)。  
**新增测试**: **无** — 这是一个未关闭的案例。  
**借鉴**: 分布式 loss 归一化中的除零风险需要系统性检查。

### Case 5: FSDP1 下 Norm 精度限制 → FSDP2 fp32_norms (`b05ab9a0`)

**现象**: FSDP1 混合精度训练时，所有参数被强制转为相同 dtype，无法保持 norm 层在 FP32。  
**根因**: FSDP1 使用 flat-param 结构，所有子参数共享同一 dtype。  
**修复**: 利用 FSDP2 的 per-module MixedPrecisionPolicy，新增 `fp32_norms` 功能。  
**新增测试**: 完整的单元测试 + 多 GPU e2e dtype dump 测试。  
**借鉴**: 混合精度的粒度需要达到模块级别，不能一刀切。

### Case 6: KD Loss 低精度计算 (`42d4732a`)

**现象**: 知识蒸馏训练中 loss 不稳定。  
**根因**: KL 散度在 BF16 下计算，log/exp 运算精度不足。  
**修复**: `.float()` upcast。  
**新增测试**: **无**  
**借鉴**: 所有涉及 log/exp 的 loss 函数都应在 FP32 下计算。

### Case 7: SP 下 Loss 日志不准确 (`954b989e`)

**现象**: 使用 Sequence Parallelism 时，日志中的 loss 值不代表真实的全局 loss。  
**根因**: per-rank gradient scaling 未被修正，导致 logged loss 仅反映单 rank 视角。  
**修复**: 仅添加 warning + TODO。**至今未修复**。  
**借鉴**: 分布式训练的 loss 日志需要全局正确的聚合，否则会误导调优。

### Case 8: TensorBoard Tag 混淆导致测试 Flaky (`6130e40c`)

**现象**: CI 中 loss threshold 测试间歇性失败。  
**根因**: `train/train_loss`（epoch 聚合）和 `train/loss`（per-step）是不同的 tag。测试混用导致在不同 step 数下行为不一致。  
**修复**: 统一使用 per-step tag + `check_tensorboard_loss_decreased`。  
**新增测试**: 修复的本身就是测试。  
**借鉴**: 测试基础设施需要标准化，特别是指标读取方式。

### Case 9: FSDP2 Optimizer State 静默丢失 (已知问题)

**现象**: checkpoint resume 后 loss 不一致。  
**根因**: `CheckpointSaveMixin._save_optimizer_and_scheduler()` 的 try/except 静默吞掉 `NotImplementedError`/`KeyError`，optimizer state 可能未保存。  
**修复**: 当前仅 LOG.warning。  
**借鉴**: Checkpoint 保存不能静默失败。需要校验机制。

### Case 10: Eval Loss 异常高 (`f56efdb4`)

**现象**: 使用 `sample_packing: true` + `eval_sample_packing: false` 时 eval loss 异常高。  
**根因**: dataset 处理路径将 sample packing 错误地应用到了 eval split，导致 eval batch 结构与预期不符。  
**修复**: 正确检查 `eval_sample_packing` 配置。  
**借鉴**: 训练/评估路径的配置分叉需要独立测试。

---

## 6. 三阶段精度对齐流程映射

### 阶段一：训练前准备与基础对齐

| 检查项 | Axolotl 状态 | 证据 | 评价 |
|---|---|---|---|
| 配置一致性 | ✅ Pydantic V2 schema 验证 | `utils/schemas/config.py` | 验证合法性，不验证精度语义 |
| 环境一致性 | ⚠️ 部分 | CI matrix 覆盖多 CUDA/PyTorch 版本 | 无跨环境数值对比 |
| seed/RNG | ✅ auto seed=42, test fixture | `validation.py:29-35`, `conftest.py:524` | 无 CUBLAS_WORKSPACE_CONFIG |
| 数据顺序 | ⚠️ 部分 | `worker_init_fn=seed_worker` | 无 DistributedSampler 确定性验证 |
| 模型结构 | ❌ 未发现 | — | 无结构 hash / 参数 count 校验 |
| 初始化权重 | ❌ 未发现 | — | 无初始权重 checksum |
| dropout/正则 | ✅ 配置化 | `lora_dropout` config field | 无 dropout 一致性验证 |
| deterministic flags | ❌ 缺失 | 无 `torch.use_deterministic_algorithms()` | **关键缺失** |

**评价**: 阶段一基础部分（seed、dtype config）做得到位，但缺乏系统性的"开训前对齐检查"机制。无 deterministic mode、无初始状态 snapshot。

### 阶段二：单卡/单步对齐

| 检查项 | Axolotl 状态 | 证据 | 评价 |
|---|---|---|---|
| forward loss | ⚠️ 间接 | `check_tensorboard_loss_decreased` (threshold) | 松散阈值，无精确值 |
| activation | ❌ 未发现 | — | 无 activation dump/compare |
| backward gradient | ⚠️ 间接 | `test_batch_flattening.py` gradient correctness | 仅 GRPO padded/flattened 对比 |
| optimizer update | ❌ 未发现 | — | 无 optimizer state 对比 |
| scheduler | ✅ | `trainer/lr.py` DS fp16 fix | 有 DS loss scale → scheduler 对齐 |
| loss scaling | ⚠️ 委托 | DeepSpeed config `loss_scale: 0` | Axolotl 不直接管理 |
| tensor dump | ❌ 未发现 | — | 无 tensor dump 工具 |
| operator-level compare | ✅ 强 | 259 个 `assert_close` 在 kernel 测试中 | Triton kernel 级对比 |

**评价**: kernel 级对比非常强，但端到端单步对齐几乎不存在。无 activation dump、无 optimizer state dump、无单步精确 loss 对比。

### 阶段三：多步/分布式/长稳对齐

| 检查项 | Axolotl 状态 | 证据 | 评价 |
|---|---|---|---|
| loss curve | ✅ | `check_tensorboard_loss_decreased` | ratio + 绝对阈值 |
| checkpoint resume | ⚠️ | `RngLoaderMixin`, 但 FSDP2 opt 可能丢失 | 有 RNG 恢复，无 round-trip test |
| DP correctness | ⚠️ | e2e loss threshold tests | 无 vs 单卡对比 |
| TP correctness | ❌ | `test_tp.py` 被 skip | tied weights 问题未解决 |
| PP correctness | ❌ | 不支持 | — |
| SP correctness | ⚠️ | `test_sp.py`, 但已知 loss 日志不准 | TODO 未关闭 |
| EP correctness | ⚠️ | EP-aware grad norm, kernel tests | 无端到端精度对比 |
| gradient accumulation | ✅ | `2501c1a6` fix + test | 修复后有回归测试 |
| communication collectives | ⚠️ | `_ep_aware_clip_grad_norm` fp32 累加 | 仅 EP grad norm |
| mixed precision stability | ✅ | `fp32_norms`, chunked CE fp32 upcast | 体系化 |
| NaN/Inf monitoring | ✅ | `loss_watchdog`, `nan_to_num`, per-step checks | 运行时检测 |
| CI regression | ✅ | `multi-gpu-e2e.yml` 双周运行 | loss threshold 检查 |

**评价**: 阶段三在 loss curve 回归和混合精度稳定性上做得不错，但分布式 correctness 验证严重不足。无 DP/TP/SP 的 vs 单卡一致性测试。

---

## 7. 可复用设计模式

### 模式 1: TensorBoard Loss 回归守卫

**设计目标**: 防止训练流程静默退化  
**源码位置**: `tests/e2e/utils.py:204-301`  
**工作流程**: 训练结束 → 读 TensorBoard 日志 → 计算 initial/final window mean → 断言 ratio + 绝对阈值  
**优点**: 简单、不依赖 golden values、自动适应不同模型规模  
**局限**: 松散阈值无法捕获细微精度退化；不能定位具体退化原因  
**迁移建议**: 直接复用，但增加更严格的 golden loss JSON 机制作为补充

### 模式 2: Plugin-Based Dtype Dump 测试

**设计目标**: 验证分布式训练中的精度策略是否正确应用  
**源码位置**: `tests/e2e/multigpu/_fp32_norms_dtype_capture.py`  
**工作流程**: 注册 plugin → 在 step 1 后 dump 参数 dtype 到 JSON → 外层 pytest 读取并断言  
**优点**: 非侵入式、跨进程（subprocess 训练 + 主进程验证）、可扩展  
**局限**: 仅验证 dtype，不验证数值  
**迁移建议**: 扩展为通用的"训练状态 dump + 外层验证"框架，增加 tensor value dump

### 模式 3: Kernel 算子级 assert_close 测试

**设计目标**: 验证自定义算子（Triton kernel）与参考实现（PyTorch）的数值一致性  
**源码位置**: `tests/test_triton_kernels.py`, `tests/test_ebft_kernels.py`  
**工作流程**: 构造随机输入 → 分别用 Triton 和 PyTorch 计算 → `torch.testing.assert_close(atol, rtol)`  
**优点**: 精确到算子级别、可参数化 dtype 和 tolerance  
**局限**: 仅覆盖自定义 kernel，不覆盖标准 PyTorch 算子  
**迁移建议**: 对所有精度敏感的自定义算子建立此类测试

### 模式 4: Gradient Correctness Cross-Path 测试

**设计目标**: 验证不同计算路径（padded vs flattened）产生一致的梯度  
**源码位置**: `tests/e2e/solo/test_batch_flattening.py:268-339`  
**工作流程**: 两个模型 → 同一数据 → 不同路径计算 loss → backward → 逐参数对比梯度  
**优点**: 端到端梯度验证  
**局限**: 仅覆盖 GRPO 的 padded/flattened 两条路径  
**迁移建议**: 扩展为通用的"多路径梯度一致性"测试框架

### 模式 5: Cosine Similarity 阈值测试

**设计目标**: 验证 fused 实现与 stock 实现的输出高度相似  
**源码位置**: `tests/monkeypatch/test_gemma4_fused_attn.py:154`, `tests/kernels/test_gemma4_fused_rope.py:99`  
**工作流程**: 计算两个实现输出的 cosine similarity，断言 > 0.999  
**优点**: 对高维向量更稳定（比 allclose 对 outlier 更鲁棒）  
**局限**: 不能检测 scale shift  
**迁移建议**: 作为 allclose 的补充，用于 attention/embedding 输出对比

### 模式 6: Loss Watchdog 运行时保护

**设计目标**: 训练过程中实时检测 loss 异常  
**源码位置**: `src/axolotl/utils/callbacks/__init__.py:57-83`  
**工作流程**: 每步检查 loss 是否超过阈值 → 连续超过 patience 次则停止训练  
**优点**: 简单有效，零开销  
**局限**: 仅检测 loss 爆炸，不检测缓慢漂移  
**迁移建议**: 增加 loss slope 检测、NaN/Inf 检测、gradient norm 异常检测

### 模式 7: EP-Aware FP32 Gradient Norm

**设计目标**: 在 Expert Parallel + FSDP 组合下正确计算跨 mesh 梯度 norm  
**源码位置**: `src/axolotl/monkeypatch/accelerate/parallelism_config.py:193-241`  
**工作流程**: local FP32 累加 → dist.all_reduce → global norm → clip  
**优点**: 解决了跨 DeviceMesh 的 DTensor 梯度 norm 不兼容问题  
**局限**: 仅处理 EP+FSDP 场景  
**迁移建议**: 对所有涉及多 mesh 的分布式策略建立类似的 FP32 聚合机制

---

## 8. 缺口分析与改造建议

### P0: 必须补齐

#### P0-1: Golden Value Loss 回归测试

**问题**: 当前仅有松散的 loss threshold 测试，无法检测细微精度退化。  
**为什么重要**: 一个数值语义变更可能将最终 loss 从 3.45 变为 3.52，在 `< 5.0` 的阈值下不会被捕获，但实际影响了训练质量。  
**当前实现**: `check_tensorboard_loss_decreased()` 使用 ratio + 绝对上界，但无精确 golden 值。  
**建议设计**: 为每个 CI 测试配置维护一个 `golden_loss.json`（包含 step → expected_loss 映射），使用更紧的 tolerance（如 5%）。CI 失败时自动输出 actual vs expected diff。  
**涉及模块**: `tests/e2e/utils.py`, 新增 `tests/golden/` 目录  
**预期收益**: 可检测 ±5% 的 loss 回归，而非当前的 100%+ 幅度变化。

#### P0-2: Deterministic Mode 支持

**问题**: 无 `torch.use_deterministic_algorithms(True)` 和 `CUBLAS_WORKSPACE_CONFIG`。  
**为什么重要**: 无法进行 bitwise 可复现的训练，无法排除随机性干扰来定位精度问题。  
**当前实现**: 仅 `seed` 配置 + `FLASH_ATTENTION_DETERMINISTIC` env var。  
**建议设计**: 新增 `deterministic: true` 配置，自动设置所有 backend flags + env vars + 禁用非确定性算子。  
**涉及模块**: `utils/schemas/config.py`, `utils/config/__init__.py`, `core/builders/base.py`  
**预期收益**: 支持 bitwise 可复现训练，是精度对齐的基础。

#### P0-3: 单卡 vs 多卡 Loss 一致性测试

**问题**: 无测试验证分布式训练的 loss 是否等于单卡 loss。  
**为什么重要**: 这是分布式训练 correctness 的核心验证。当前仅验证"loss 下降了"，不验证"和单卡一样"。  
**当前实现**: `validation.py:1646` 仅有 LOG.warning 建议用户自行对比。  
**建议设计**: 对小模型在 1/2/4 GPU 上跑相同 config+seed，比较前 N 步 loss 的 `allclose`。  
**涉及模块**: `tests/e2e/multigpu/`, 新增 `tests/e2e/precision/`  
**预期收益**: 自动捕获分布式通信引入的数值偏差。

### P1: 强烈建议补齐

#### P1-1: Tensor Dump / Activation Compare 工具

**问题**: 无法在训练过程中 dump 中间 tensor 进行对比。  
**为什么重要**: 当 loss 偏差时，需要逐层定位哪里开始 diverge。  
**当前实现**: 仅有 `_fp32_norms_dtype_capture` 作为 dtype dump 的 prototype。  
**建议设计**: 通用的 `ActivationDumpPlugin`，可在指定 step 和指定层 dump tensor 到文件。配套 `compare_activations(dump1, dump2, tolerance)` 工具。  
**涉及模块**: 新增 `src/axolotl/utils/precision/`, `src/axolotl/integrations/precision_debug.py`  
**预期收益**: 将精度问题定位时间从"盲猜"缩短到"逐层对比"。

#### P1-2: Checkpoint Round-Trip 一致性测试

**问题**: 无测试验证 save → load → 继续训练 的数值一致性。  
**为什么重要**: FSDP2 下 optimizer state 可能静默丢失（`checkpoints.py:13-23`），resume 后 loss curve 可能 diverge。  
**当前实现**: 有 RNG state 恢复，但无 round-trip 数值测试。  
**建议设计**: 训练 N 步 → checkpoint → load → 训练 N 步，vs 直接训练 2N 步，比较 loss 曲线。  
**涉及模块**: `tests/e2e/`, `core/trainers/mixins/checkpoints.py`  
**预期收益**: 确保 checkpoint resume 的数值正确性。

#### P1-3: 修复已知架构 Bug

**问题 A**: ORPO loss 硬编码 bfloat16 cast (`core/trainers/base.py:652`)  
**问题 B**: `bf16="auto"` 字符串可能泄漏到 TrainingArguments (`core/builders/base.py:261`)  
**问题 C**: SP 下 loss 日志不准确 (`954b989e` 的 TODO 未关闭)  
**涉及模块**: `core/trainers/base.py`, `core/builders/base.py`, `utils/ctx_managers/sequence_parallel.py`  
**预期收益**: 消除已知的数值不一致源。

#### P1-4: NaN/Inf 全局检测

**问题**: 当前 NaN 检测分散在各处（GRPO `nan_to_num`、SonicMoE test `isnan` 检查），无全局机制。  
**建议设计**: 新增 `NaNWatchdogCallback`，在每步检查 loss、grad_norm 是否为 NaN/Inf，可选开启 `torch.autograd.set_detect_anomaly(True)`。  
**涉及模块**: `utils/callbacks/`, `utils/schemas/config.py`

### P2: 长期优化项

#### P2-1: 跨精度策略一致性矩阵

**建议**: 自动化测试矩阵 `{FP32, BF16, FP8} × {单卡, DDP, FSDP2} × {小模型}` → 比较 loss curves。

#### P2-2: 配置快照与 Diff

**建议**: 训练前 dump 完整 resolved config（含所有 auto-resolved 值）到 JSON，支持两次训练的 config diff。

#### P2-3: 自动化二分定位

**建议**: 当 CI loss regression 触发时，自动对最近 N 个 commit 进行 `git bisect`，找到引入回归的 commit。

#### P2-4: Nightly Full Training Regression

**建议**: 在 nightly CI 中跑一个完整的小规模训练（100 steps），与 golden loss curve 精确对比。当前 nightly 仅测试兼容性。

---

## 9. 推荐学习路线

### 第 1 步：读文档与配置

1. `CLAUDE.md` — 项目整体架构
2. `.claude/rules/testing.md` — 测试规范
3. `.claude/rules/monkeypatch.md` — Monkeypatch 安全规范
4. `.claude/rules/config-schema.md` — 配置系统
5. `src/axolotl/utils/schemas/config.py:554-610` — 精度相关配置字段
6. `deepspeed_configs/zero1.json` — DeepSpeed loss scaling 配置

### 第 2 步：跑 examples/tests

1. `pytest tests/test_triton_kernels.py -v` — 感受 kernel 级 assert_close
2. `pytest tests/test_fp32_norms.py -v` — 感受 fp32_norms 单元测试
3. `pytest tests/test_ebft_kernels.py -v` — 感受 EBFT kernel 精度测试
4. `pytest tests/test_chunked_xentropy.py -v` — chunked CE vs 标准 CE 对比
5. （需 GPU）`pytest tests/e2e/test_packing_loss.py -v` — loss threshold 测试
6. （需 2 GPU）`pytest tests/e2e/multigpu/test_fsdp2_fp32_norms.py -v` — dtype dump 模式

### 第 3 步：读源码

按以下顺序阅读核心文件：

1. `src/axolotl/train.py` — 训练入口（特别关注 precision 相关路径）
2. `src/axolotl/utils/config/__init__.py:70-110` — `resolve_dtype()` 和 tf32 设置
3. `src/axolotl/utils/trainer.py:680-690` — `prepare_optim_env()` 混合精度环境变量
4. `src/axolotl/core/builders/base.py:255-263` — `_configure_precision_settings()`
5. `src/axolotl/utils/fp32_norms.py` — fp32_norms 完整实现
6. `src/axolotl/monkeypatch/accelerate/fsdp2.py:365-436` — FSDP2 混合精度 + fp32 norms sharding
7. `src/axolotl/monkeypatch/loss/chunked.py` — Chunked CE FP32 upcast
8. `src/axolotl/monkeypatch/trainer/utils.py` — Triton fused entropy + selective_log_softmax
9. `src/axolotl/monkeypatch/accelerate/parallelism_config.py:193-268` — EP-aware grad norm
10. `src/axolotl/utils/ctx_managers/sequence_parallel.py` — SP loss 聚合
11. `src/axolotl/core/trainers/mixins/rng_state_loader.py` — RNG 状态恢复
12. `src/axolotl/core/trainers/mixins/checkpoints.py` — Checkpoint 保存（含 FSDP2 问题）
13. `tests/e2e/utils.py:204-301` — `check_tensorboard_loss_decreased()`
14. `tests/e2e/multigpu/_fp32_norms_dtype_capture.py` — Plugin-based dtype dump

### 第 4 步：复现 commit/PR 中的问题

1. `2501c1a6`: 注释掉 `num_items_in_batch` 传递，观察 gradient accumulation 下 loss 变化
2. `da97285e`: 将 MoE gate 层改为 bf16，观察训练不稳定
3. `0ae06d75`: 在 CP 下使用全 padding 输入，观察 NaN loss
4. `b05ab9a0`: 禁用 `fp32_norms`，对比 FSDP2 bf16 训练的 loss curve

### 第 5 步：抽象设计模式

1. TensorBoard loss 回归守卫 → 通用化为 JSON golden values
2. Plugin-based dtype dump → 通用化为 activation/gradient dump
3. Kernel assert_close → 标准化为所有自定义算子的测试模板
4. Loss watchdog → 扩展为多维度运行时异常检测
5. EP-aware FP32 grad norm → 通用化为跨 mesh 聚合框架

### 第 6 步：迁移到自己的训练系统

详见下节。

---

## 10. 对自研分布式训练系统的迁移建议

### 10.1 直接可复用的组件

| 组件 | 来源 | 迁移复杂度 | 优先级 |
|---|---|---|---|
| `check_tensorboard_loss_decreased()` | `tests/e2e/utils.py` | 低 | P0 |
| `fp32_norms` 模式 | `utils/fp32_norms.py` | 中 | P0 |
| Kernel assert_close 测试模板 | `tests/test_triton_kernels.py` | 低 | P0 |
| Loss Watchdog callback | `utils/callbacks/__init__.py` | 低 | P1 |
| Plugin-based dtype dump | `_fp32_norms_dtype_capture.py` | 低 | P1 |
| EP-aware FP32 grad norm | `parallelism_config.py` | 中 | P1 |

### 10.2 需要自建的组件（Axolotl 缺失）

| 组件 | 建议设计 | 优先级 |
|---|---|---|
| Deterministic mode | `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | P0 |
| Golden loss JSON | 为每个测试配置维护 step→loss 映射，CI 对比 | P0 |
| 单卡 vs 多卡一致性 | 同 config+seed 在 1/N GPU 跑，loss `allclose` | P0 |
| Activation dump/compare | 通用 hook → HDF5/safetensors dump → allclose compare | P1 |
| Gradient dump/compare | 逐层 grad norm + grad value dump | P1 |
| Optimizer state round-trip | save → load → N steps 的 loss curve 对比 | P1 |
| 配置快照 + diff | resolved config JSON dump + jq diff | P2 |
| 自动化 git bisect | CI 失败时自动二分定位 commit | P2 |

### 10.3 架构建议

1. **Precision Manager**: 集中管理所有精度设置（dtype、TF32、deterministic、loss scaling），而非分散在 config/builder/monkeypatch 中
2. **Precision Test Framework**: 标准化的测试基类，提供 `assert_loss_close(actual, golden, rtol)`, `assert_grad_close(model1, model2, rtol)`, `dump_activations(model, step)` 等方法
3. **CI Precision Gate**: 在 CI pipeline 中加入精度回归 gate，使用紧密的 tolerance
4. **Checkpoint Integrity Verifier**: 在 checkpoint save 后立即 load 并验证 tensor checksum

---

## Appendix A. 检索关键词与命令记录

### 源码搜索关键词（均已执行）

| 关键词组 | 搜索方式 | 命中数 |
|---|---|---|
| deterministic, determinism, seed, manual_seed, set_seed | `grep -rn` in src/ + tests/ | ~80 |
| golden, baseline, expected, allclose, rtol, atol, assert_close | `grep -rn` in tests/ | ~259 (assert_close/allclose) |
| fp32, tf32, fp16, bf16, fp8, amp, mixed_precision | `grep -rn` in src/ + tests/ | ~150+ |
| nan, inf, isnan, isinf, detect_anomaly, nan_to_num | `grep -rn` in src/ + tests/ | ~40 |
| checkpoint, resume, save_state, load_state, rng_state | `grep -rn` in src/ | ~80 |
| all_reduce, reduce_scatter, distributed, fsdp, deepspeed | `grep -rn` in src/ + tests/ | ~100+ |
| loss, compute_loss, training_step, loss_scale, loss_watchdog | `grep -rn` in src/ + tests/ | ~200+ |
| grad_norm, clip_grad, gradient_checkpointing | `grep -rn` in src/ | ~20 |
| profiler, trace, torch.profiler | `grep -rn` in src/ | ~15 |
| dump, hook, register_hook | `grep -rn` in src/ | ~30 |
| cosine_similarity, cos_sim | `grep -rn` in tests/ | ~15 |

### Git 搜索命令（均已执行）

```bash
git log --oneline --all --grep="<keyword>" | head -N
# 对以下关键词：precision, accuracy, determin, golden, seed, loss, gradient,
# bf16, fp16, fp8, tf32, checkpoint, distributed, fsdp, deepspeed,
# nan, inf, overflow, numerical, reproducib, regression, mismatch,
# resume, optimizer, all_reduce, tensor parallel, mixed precision,
# loss scale, grad_norm, dropout, rng, random, stability, correctness, alignment

git grep -n "<pattern>"
# 对以下 pattern：allclose, rtol, atol, torch.testing, assert_close,
# golden, expected_loss, baseline, set_seed, manual_seed, deterministic,
# detect_anomaly, isnan, isinf, grad_scaler, loss_scale, CUBLAS,
# tensor_dump, dump_tensor, save_tensor
```

### 已检查目录

| 目录 | 存在 | 精度相关内容 |
|---|---|---|
| `tests/` | ✅ | 全部精度测试所在 |
| `tests/e2e/` | ✅ | loss threshold + dtype 测试 |
| `tests/e2e/multigpu/` | ✅ | 分布式精度测试 |
| `.github/workflows/` | ✅ | 11 个 workflow 文件 |
| `src/axolotl/monkeypatch/` | ✅ | 精度相关 patch |
| `src/axolotl/core/trainers/` | ✅ | loss 计算、gradient 处理 |
| `src/axolotl/utils/schemas/` | ✅ | 精度配置定义 |
| `deepspeed_configs/` | ✅ | loss scaling 配置 |
| `benchmarks/` | ✅ | kernel perf benchmark (非精度) |
| `scripts/` | ✅ | 安装脚本 (非精度) |
| `debug/` | ❌ 不存在 | — |
| `ci/` | ❌ 不存在 | — |
| `tools/` | ❌ 不存在 | — |

---

## Appendix B. 关键文件清单

### 精度配置

| 文件 | 行号 | 内容 |
|---|---|---|
| `src/axolotl/utils/schemas/config.py` | 482, 554-610, 965-1005, 1588-1632 | seed, bf16/fp16/fp8/tf32, fsdp, fp32_norms, validators |
| `src/axolotl/utils/schemas/validation.py` | 29-35, 1274, 1644-1646 | seed auto-default, NPU tf32 warning, CP baseline warning |
| `src/axolotl/utils/config/__init__.py` | 70-110 | resolve_dtype(), tf32 backend settings |
| `src/axolotl/core/builders/base.py` | 255-263, 583-600 | _configure_precision_settings, training args |
| `src/axolotl/utils/trainer.py` | 533-690 | setup_deepspeed_env, setup_fsdp_envs, prepare_optim_env |

### 精度实现

| 文件 | 内容 |
|---|---|
| `src/axolotl/utils/fp32_norms.py` | FSDP2 fp32 norm 管理 |
| `src/axolotl/monkeypatch/accelerate/fsdp2.py` | FSDP2 mixed precision, fp32 norms sharding |
| `src/axolotl/monkeypatch/loss/chunked.py` | Chunked CE FP32 upcast |
| `src/axolotl/monkeypatch/trainer/utils.py` | Triton fused entropy + selective_log_softmax |
| `src/axolotl/monkeypatch/accelerate/parallelism_config.py` | EP-aware grad norm clipping |
| `src/axolotl/monkeypatch/transformers/trainer_loss_calc.py` | nanmean eval loss patch |
| `src/axolotl/monkeypatch/trainer/lr.py` | DS fp16 loss scale → scheduler fix |
| `src/axolotl/monkeypatch/torchao_optim.py` | TorchAO optim state dtype fix |
| `src/axolotl/utils/ctx_managers/sequence_parallel.py` | SP loss normalization all_reduce |
| `src/axolotl/core/trainers/mixins/rng_state_loader.py` | RNG state restore |
| `src/axolotl/core/trainers/mixins/checkpoints.py` | Checkpoint save (FSDP2 issue) |
| `src/axolotl/core/trainers/base.py` | compute_loss, ORPO loss, worker_init_fn |
| `src/axolotl/utils/callbacks/__init__.py` | LossWatchDogCallback |

### 精度测试

| 文件 | 内容 |
|---|---|
| `tests/e2e/utils.py` | check_tensorboard, check_tensorboard_loss_decreased |
| `tests/test_triton_kernels.py` | ~30 assert_close for Triton kernels |
| `tests/test_ebft_kernels.py` | ~25 assert_close for EBFT kernels |
| `tests/test_fp32_norms.py` | 12 unit tests for fp32_norms |
| `tests/test_chunked_xentropy.py` | chunked CE vs standard CE |
| `tests/e2e/multigpu/test_fsdp2_fp32_norms.py` | dtype dump e2e test |
| `tests/e2e/multigpu/test_fp8_fsdp2.py` | FP8 NaN check |
| `tests/e2e/multigpu/test_fsdp2.py` | FSDP2 loss decreased |
| `tests/e2e/multigpu/test_tp.py` | TP loss decreased (skipped) |
| `tests/e2e/multigpu/patched/test_sp.py` | SP loss threshold |
| `tests/e2e/multigpu/test_dist_muon_fsdp2.py` | Muon optimizer + FSDP2 |
| `tests/e2e/test_packing_loss.py` | Packing loss threshold |
| `tests/e2e/solo/test_batch_flattening.py` | Gradient correctness |
| `tests/e2e/solo/test_trainer_loss_calc.py` | nanmean patch patchability |
| `tests/monkeypatch/test_gemma4_fused_attn.py` | Cosine similarity > 0.999 |
| `tests/kernels/test_gemma4_fused_rope.py` | Cosine similarity > 0.999 |
| `tests/e2e/kernels/test_lora_features.py` | _compare_tensors utility |
| `tests/e2e/integrations/test_sonicmoe_lora.py` | Per-step NaN/Inf assertion |
| `tests/conftest.py` | torch_manual_seed(42) autouse fixture |

---

## Appendix C. 关键 Commits / PR / Issues 清单

| # | Commit | 日期 | 描述 | 精度影响 | 测试 |
|---|---|---|---|---|---|
| 1 | `b05ab9a0` | 2026-05 | fp32_norms for FSDP2 | 高 | ✅ 完整闭环 |
| 2 | `0ae06d75` | 2025-08 | nanmean CP eval loss | 高 | ✅ 闭环 |
| 3 | `2501c1a6` | 2024-10 | gradient accumulation loss fix | 高 | ✅ 闭环 |
| 4 | `da97285e` | 2024-01 | MoE gate fp32 | 高 | ✅ 闭环 |
| 5 | `24bb2c9c` | 2026-01 | CP div-by-zero | 高 | ❌ 无 |
| 6 | `97a4f285` | 2026-02 | CP eval loss weighted correction | 高 | ❌ 无 |
| 7 | `42d4732a` | 2025-01 | KD loss fp32 upcast | 中 | ❌ 无 |
| 8 | `68e97d03` | 2025-02 | KD Triton kernel overflow | 中 | ❌ 无 |
| 9 | `fc2d63ee` | 2026-03 | tf32 API 2.9+ | 中 | ❌ 隐式 |
| 10 | `830e9f7e` | 2026-03 | tf32 auto detection | 中 | ❌ 隐式 |
| 11 | `954b989e` | 2025-04 | SP loss logging warning | 低 | ❌ TODO |
| 12 | `6130e40c` | 2026-03 | 21 file test tag fix | 测试基础设施 | ✅ |
| 13 | `f56efdb4` | 2026-03 | eval sample packing fix | 高 | ❌ 无 |
| 14 | `5352d41d` | 2026-05 | multimodal loss masking | 高 | ✅ |
| 15 | `96deb6bd` | 2023-08 | QLoRA norm/embed fp32 (基础) | 基础性 | ❌ 无 |

---

## 附：7 个已发现的架构级精度风险

1. **ORPO 硬编码 bfloat16 cast** — `core/trainers/base.py:652`，不论训练精度一律 `.to(dtype=torch.bfloat16)`
2. **`bf16="auto"` 字符串泄漏** — `core/builders/base.py:261`，`cfg.bf16 or cfg.bfloat16` 可能将 `"auto"` 传给 TrainingArguments
3. **FSDP2 optimizer state 静默丢失** — `core/trainers/mixins/checkpoints.py:13-23`，try/except 吞掉保存失败
4. **Deterministic mode 未启用** — 无 `torch.use_deterministic_algorithms()`
5. **Ring attention 默认非确定性** — `monkeypatch/ring_attn/patch.py:96-98`，默认 `deterministic=False`
6. **Eval loop patch 脆弱** — `monkeypatch/transformers/trainer_loss_calc.py` 使用源码字符串匹配，upstream 格式变化会静默失效
7. **SP loss 日志不准确** — `954b989e` 的 TODO 至今未关闭
