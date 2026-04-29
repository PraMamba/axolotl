# Deep Analysis: Fully Sharded Data Parallelism (FSDP) in Axolotl

**Analysis Date**: 2026-04-27
**Codebase Version**: Commit `a7adb056` (main branch)
**Scope**: Single-node and multi-node FSDP (v1 and v2) implementation

---

## 1. High-Level Summary

Axolotl implements FSDP support through a **six-layer architecture** that translates a single YAML configuration into a fully sharded distributed training setup:

1. **Configuration Layer** (`FSDPConfig` Pydantic schema) — Validates and normalizes FSDP settings
2. **Environment Translation Layer** (`setup_fsdp_envs()`) — Converts config to `FSDP_*` environment variables for HuggingFace Accelerate
3. **Monkeypatch Layer** (8 patch modules) — Replaces upstream Accelerate/PyTorch/TRL/PEFT/BnB internals to support FSDP2 + QLoRA, cpu_ram_efficient_loading, and DTensor compatibility
4. **Model Loading Layer** (`ModelLoader`) — Handles device_map, dtype, meta-device, and quantization-aware loading for FSDP
5. **Training Layer** (`TrainerBuilder` + Trainer Mixins) — Configures `TrainingArguments`, handles FSDP-aware save/checkpoint, and manages the Accelerator lifecycle
6. **Post-Training Layer** (`save_trained_model()` + `merge_sharded_fsdp_weights`) — Gathers FSDP sharded state dicts, merges checkpoints, and cleans up FSDP-prefixed architecture names

**Key dependencies**: PyTorch >= 2.7.0 (for FSDP2), HuggingFace Accelerate (FSDPPlugin, ParallelismConfig), HuggingFace Transformers (TrainingArguments, Trainer), PEFT (LoRA/QLoRA), bitsandbytes (Params4bit/Int8Params), TRL (RL trainers).

**Entry point**: `axolotl train config.yml` → `accelerate launch` → per-process training with FSDP wrapping via `accelerator.prepare(model)`.

**FSDP v1 vs v2**: Axolotl supports both but v1 is deprecated (with warnings). FSDP v2 is the primary path receiving new patches and features (QLoRA, FP8, LoRA kernels, cpu_ram_efficient_loading).

---

## 2. Source Code Map

| File Path | Key Class / Function | Role | Relationship to FSDP |
|---|---|---|---|
| `src/axolotl/utils/schemas/fsdp.py` | `FSDPConfig` | Configuration schema | Pydantic model defining all FSDP fields (13 fields) |
| `src/axolotl/utils/schemas/config.py:933-950` | `AxolotlInputConfig` (FSDP fields) | Top-level config | `fsdp`, `fsdp_config`, `fsdp_version`, `fsdp_final_state_dict_type` |
| `src/axolotl/utils/schemas/validation.py` | 18+ validators | Config validation | FSDP compatibility checks (DeepSpeed mutual exclusion, optimizer compat, torch version, model restrictions) |
| `src/axolotl/utils/trainer.py:589-618` | `setup_fsdp_envs()` | Env var setup | Translates FSDPConfig → `FSDP_*` environment variables for Accelerate |
| `src/axolotl/utils/trainer.py:621-640` | `setup_parallelism_envs()` | Parallelism env | Sets `PARALLELISM_CONFIG_*` for TP/CP/DP mesh |
| `src/axolotl/utils/distributed.py:299-370` | `build_parallelism_config()` | Device mesh | Builds `ParallelismConfig` + `DeviceMesh` for FSDP2 |
| `src/axolotl/loaders/patch_manager.py:270-308` | `_apply_fsdp_patches()` | Patch orchestration | Applies all FSDP monkeypatches in correct order |
| `src/axolotl/loaders/patch_manager.py:590-608` | `_apply_fsdp2_bnb_patches()` | BnB patch orchestration | Applies 4 QLoRA/8bit patches for FSDP2 |
| `src/axolotl/monkeypatch/accelerate/fsdp2.py` | `fsdp2_prepare_model()`, `get_state_dict()`, `fsdp2_load_full_state_dict()` | Core FSDP2 patches (539 lines) | Replaces Accelerate's FSDP2 model preparation, state dict gathering, weight broadcasting |
| `src/axolotl/monkeypatch/fsdp2_qlora.py` | `apply_init_sharded_param_patch()`, etc. | FSDP2+QLoRA patches (237 lines) | Preserves BnB quantization metadata through FSDP2 shard/unshard cycle |
| `src/axolotl/monkeypatch/accelerate/parallelism_config.py` | `patch_parallelism_config()` | AcceleratorState fixes | Guards `is_fsdp2` property, allows standalone CP |
| `src/axolotl/monkeypatch/trainer/trl.py` | `patch_trl_prepare_fsdp2()` | TRL redirect | Redirects TRL's FSDP prep to Axolotl's custom implementation |
| `src/axolotl/monkeypatch/trainer_fsdp_optim.py` | `patch_training_loop_for_fsdp()` | Trainer loop fix (disabled) | Fixes FSDP optimizer save; currently commented out |
| `src/axolotl/monkeypatch/trainer_accelerator_args.py` | `patch_create_accelerate_code_for_fp8()` | FP8+FSDP | Injects `enable_fsdp_float8_all_gather` into Accelerator |
| `src/axolotl/monkeypatch/torchao_optim.py` | torchao optimizer patches | Optimizer compat | Fixes DTensor view operations for low-bit optimizer states |
| `src/axolotl/loaders/model.py:152-159` | `ModelLoader.is_fsdp_enabled` | Model loading | FSDP detection property; gates all FSDP-specific loading logic |
| `src/axolotl/loaders/model.py:456-537` | `_set_device_map_config()` | Device map | FSDP: no device_map; QLoRA+FSDP: `{"": LOCAL_RANK}`; FSDP2+cpu_efficient: rank0→cpu, others→meta |
| `src/axolotl/loaders/model.py:756-808` | `_build_model()` | Model build | cpu_ram_efficient_loading, meta device, sharded quantized loading |
| `src/axolotl/loaders/adapter.py:303-357` | `load_lora()` (FSDP sections) | Adapter loading | `setup_quantized_meta_for_peft` / `setup_quantized_peft_meta_for_training` |
| `src/axolotl/utils/model_shard_quant.py` | `load_sharded_model_quant()` | Sharded QLoRA loading | Shard-by-shard quantized weight loading for FSDP |
| `src/axolotl/core/builders/base.py:604-606` | `_set_base_training_args()` | Builder | Passes `fsdp_config`/`fsdp` to `TrainingArguments` |
| `src/axolotl/core/trainers/mixins/distributed_parallel.py` | `DistributedParallelMixin` | Trainer mixin | `_save()` override for FSDP state dict; `create_accelerator_and_postprocess()` for CP fallback |
| `src/axolotl/core/trainers/mixins/checkpoints.py` | `CheckpointSaveMixin` | Trainer mixin | Gracefully handles FSDP2 optimizer save failures |
| `src/axolotl/core/trainers/mixins/activation_checkpointing.py` | `ActivationOffloadingMixin` | Trainer mixin | FSDP-compatible activation checkpointing via `ModuleWrapPolicy` |
| `src/axolotl/train.py:294-349` | `save_trained_model()` | Post-training save | FSDP state dict type selection, sharded weight merge, FSDP prefix cleanup |
| `src/axolotl/cli/merge_sharded_fsdp_weights.py` | `merge_fsdp_weights()` | CLI utility | Merges `SHARDED_STATE_DICT` checkpoints to safetensors |
| `src/axolotl/cli/main.py:245-274` | `merge_sharded_fsdp_weights` CLI cmd | CLI entry | Click command for standalone checkpoint merging |
| `docs/fsdp_qlora.qmd` | Documentation | User docs | FSDP+QLoRA usage guide |

---

## 3. Complete Call Chain

### 3.1 Primary Training Flow

```
User: axolotl train config.yml
  │
  ├─> cli/main.py:train()                              [Click command]
  │     └─> cli/utils/train.py:launch_training()        [Choose launcher]
  │           ├─> _launch_accelerate_training()          [Default: accelerate launch]
  │           └─> _launch_torchrun_training()            [Alternative: torchrun]
  │                 └─> _add_default_rdzv_args()         [Multi-node RDZV defaults]
  │
  ├─> accelerate launch -m axolotl.cli.train config.yml [Per-process spawn]
  │
  └─> cli/train.py:do_train()                           [Per-process entry]
        │
        ├─> cli/config.py:load_cfg()                    [Config loading]
        │     ├─> YAML → DictDefault
        │     ├─> validate_config()                     [Pydantic validation]
        │     │     ├─> check_fsdp_config_kwargs_prefix()     [Strip fsdp_ prefix]
        │     │     ├─> check_fsdp_version_in_fsdp_config()   [Sync version fields]
        │     │     ├─> check_fsdp_version()                  [FSDP1 deprecation warn]
        │     │     ├─> check_fsdp_deepspeed()                [Mutual exclusion]
        │     │     ├─> check_fsdp_torch_version()            [torch >= 2.7.0]
        │     │     ├─> check_fsdp2_cpu_offload_pin_memory()  [pin_memory constraints]
        │     │     ├─> check_fsdp2_base_model_quant_rl()     [Quant+RL guard]
        │     │     ├─> check_muon_deepspeed_fsdp()           [Muon→FSDP2 only]
        │     │     ├─> check_flashoptim_deepspeed_fsdp()     [Flash opt→FSDP2 only]
        │     │     ├─> check_fsdp_offload_w_8bit_optimizer() [FSDP1 offload guard]
        │     │     ├─> check_fsdp2_w_8bit_optimizer()        [FSDP2 bnb 8bit guard]
        │     │     ├─> check_falcon_fsdp()                   [Falcon guard]
        │     │     └─> check_multigpu_lora_kernels()         [FSDP1 LoRA kernels guard]
        │     │
        │     ├─> model_dump(exclude_none=True) → DictDefault  [Lose Pydantic types]
        │     │
        │     ├─> prepare_optim_env(cfg)                [Environment setup]
        │     │     ├─> setup_fsdp_envs(cfg)            [Set FSDP_* env vars]
        │     │     │     ├─> ACCELERATE_USE_FSDP=true
        │     │     │     ├─> FSDP_VERSION=2
        │     │     │     ├─> FSDP_ACTIVATION_CHECKPOINTING=true
        │     │     │     ├─> FSDP_OFFLOAD_PARAMS=true
        │     │     │     ├─> FSDP_SYNC_MODULE_STATES=true
        │     │     │     ├─> FSDP_CPU_RAM_EFFICIENT_LOADING=true
        │     │     │     ├─> FSDP_USE_ORIG_PARAMS=true
        │     │     │     ├─> FSDP_STATE_DICT_TYPE=...
        │     │     │     ├─> FSDP_AUTO_WRAP_POLICY=...
        │     │     │     ├─> FSDP_TRANSFORMER_CLS_TO_WRAP=...
        │     │     │     └─> FSDP_RESHARD_AFTER_FORWARD=true
        │     │     │
        │     │     ├─> setup_parallelism_envs(cfg)     [Set PARALLELISM_CONFIG_*]
        │     │     └─> ACCELERATE_MIXED_PRECISION=bf16|fp16|fp8|no
        │     │
        │     └─> normalize_config(cfg)                 [Batch size scaling w/ world_size]
        │
        ├─> train.py:train()
        │     ├─> setup_model_and_trainer()
        │     │     ├─> setup_model_and_tokenizer()
        │     │     │     └─> ModelLoader.load()
        │     │     │           ├─> PatchManager.apply_pre_model_load_patches()
        │     │     │           │     ├─> _apply_torchao_patches()       [DTensor compat]
        │     │     │           │     ├─> _apply_fsdp_patches()
        │     │     │           │     │     ├─> patch_initialize_missing_keys_for_fsdp()
        │     │     │           │     │     ├─> patch_parallelism_config()  [FSDP2/CP]
        │     │     │           │     │     ├─> patch_accelerate_fsdp2()    [FSDP2]
        │     │     │           │     │     ├─> patch_tied_keys_for_meta_device() [FSDP2+cpu_efficient]
        │     │     │           │     │     └─> patch_trl_prepare_fsdp2()   [FSDP2+RL]
        │     │     │           │     └─> _apply_fsdp2_bnb_patches()
        │     │     │           │           ├─> apply_init_sharded_param_patch()
        │     │     │           │           ├─> apply_init_unsharded_param_patch()
        │     │     │           │           ├─> apply_init_dtype_attrs_patch()
        │     │     │           │           └─> apply_linear8bitlt_save_patch() [8bit only]
        │     │     │           │
        │     │     │           ├─> _set_device_map_config()
        │     │     │           │     ├─> No device_map (FSDP without QLoRA)
        │     │     │           │     ├─> {"": LOCAL_RANK} (QLoRA+FSDP)
        │     │     │           │     ├─> "cpu" (FSDP2+cpu_efficient, rank 0)
        │     │     │           │     └─> "meta" (FSDP2+cpu_efficient, rank != 0)
        │     │     │           │
        │     │     │           ├─> _build_model()
        │     │     │           │     ├─> AutoModelForCausalLM.from_pretrained()
        │     │     │           │     └─> load_sharded_model_quant() [QLoRA+FSDP sharded]
        │     │     │           │
        │     │     │           ├─> _configure_embedding_dtypes()   [Skip under FSDP]
        │     │     │           └─> _prepare_model_for_quantization() [Skip kbit prep under FSDP+QLoRA]
        │     │     │
        │     │     └─> setup_trainer()
        │     │           └─> HFCausalTrainerBuilder.build() / HFRLTrainerBuilder.build()
        │     │                 ├─> _set_base_training_args()
        │     │                 │     └─> training_args["fsdp_config"] = cfg.fsdp_config
        │     │                 │         training_args["fsdp"] = True
        │     │                 │
        │     │                 └─> AxolotlTrainer(**kwargs)
        │     │                       └─> Trainer.__init__()
        │     │                             ├─> create_accelerator_and_postprocess()  [DistributedParallelMixin]
        │     │                             │     └─> Downgrade FSDP→MULTI_GPU if fsdp_plugin is None (pure CP)
        │     │                             └─> accelerator.prepare(model)
        │     │                                   └─> fsdp2_prepare_model()  [PATCHED]
        │     │                                         ├─> fsdp2_plugin.set_auto_wrap_policy(model)
        │     │                                         ├─> apply_activation_checkpointing() [if configured]
        │     │                                         ├─> Build fsdp2_kwargs (reshard, offload, mp_policy, mesh)
        │     │                                         ├─> model.to(meta) [if cpu_ram_efficient + no Params4bit]
        │     │                                         ├─> patch_peft_param_wrapper_for_fsdp2() [if ParamWrapper]
        │     │                                         ├─> Walk modules bottom-up:
        │     │                                         │     ├─> LoraLayer → _process_lora_module_for_fsdp()
        │     │                                         │     │     ├─> fully_shard(lora_A[adapter])
        │     │                                         │     │     ├─> fully_shard(lora_B[adapter])
        │     │                                         │     │     └─> fully_shard(lora_magnitude_vector[adapter])
        │     │                                         │     └─> auto_wrap_policy(module) → fully_shard(module)
        │     │                                         ├─> fully_shard(model)  [Root model]
        │     │                                         ├─> fsdp2_load_full_state_dict()  [Broadcast from rank 0]
        │     │                                         │     ├─> distribute_tensor() for DTensor params
        │     │                                         │     └─> dist.broadcast() for non-sharded params
        │     │                                         ├─> Re-register non-persistent buffers
        │     │                                         └─> model.tie_weights()
        │     │
        │     ├─> execute_training()
        │     │     └─> trainer.train()                 [HF Trainer training loop]
        │     │           ├─> training_step()
        │     │           │     ├─> model.forward()     [FSDP all-gather before forward]
        │     │           │     └─> loss.backward()     [FSDP reduce-scatter on gradients]
        │     │           └─> optimizer.step()          [Every gradient_accumulation_steps]
        │     │
        │     └─> save_trained_model()
        │           ├─> fsdp_plugin.set_state_dict_type(final_state_dict_type)
        │           ├─> trainer.save_model()
        │           │     └─> DistributedParallelMixin._save()
        │           │           └─> accelerator.get_state_dict(model)  [PATCHED]
        │           │                 ├─> FSDP2: param.full_tensor() per DTensor param
        │           │                 └─> FSDP1: FSDP.state_dict_type() context manager
        │           ├─> merge_fsdp_weights() [if SHARDED_STATE_DICT]
        │           └─> Strip "FSDP" prefix from config.json architectures
```

### 3.2 Multi-Node Setup

Multi-node FSDP in Axolotl uses the standard `torchrun` / `accelerate launch` distributed launch mechanisms:

```
# torchrun launcher:
torchrun --nproc-per-node=N --nnodes=M --rdzv_endpoint=MASTER:PORT \
    -m axolotl.cli.train config.yml

# accelerate launcher:
accelerate launch --multi_gpu --num_processes=TOTAL \
    -m axolotl.cli.train config.yml
```

**File**: `src/axolotl/cli/utils/train.py:15-43` — `_add_default_rdzv_args()` provides defaults:
- `rdzv_backend=c10d`
- Random `rdzv_id` if not specified
- Accepts user-provided `--rdzv_endpoint`, `--rdzv_backend`, `--rdzv_id`

The actual multi-node FSDP sharding is handled entirely by PyTorch's distributed runtime and the `DeviceMesh` built from `ParallelismConfig`. Axolotl's code is node-agnostic — the same FSDP wrapping logic applies regardless of whether processes are on 1 or N nodes.

---

## 4. Detailed Implementation Analysis

### 4.1 Configuration System

#### FSDPConfig Schema
**File**: `src/axolotl/utils/schemas/fsdp.py:10-77`

A Pydantic `BaseModel` with 13 fields, all optional (defaulting to `None`):

| Field | Type | Description |
|---|---|---|
| `fsdp_version` | `int \| None` | 1 or 2; has `AliasChoices("fsdp_version", "version")` |
| `activation_checkpointing` | `bool \| None` | Reduces memory by recomputing activations |
| `offload_params` | `bool \| None` | CPU parameter offloading |
| `sync_module_states` | `bool \| None` | Sync module states across processes |
| `cpu_ram_efficient_loading` | `bool \| None` | Only rank 0 loads real weights; others get meta tensors |
| `cpu_offload_pin_memory` | `bool \| None` | `false` enables swap memory for constrained setups |
| `use_orig_params` | `bool \| None` | Use original (not flattened) parameters |
| `state_dict_type` | `Literal["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"] \| None` | Checkpoint format during training |
| `final_state_dict_type` | Same literal | Checkpoint format for final save |
| `auto_wrap_policy` | `Literal["TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP"] \| None` | Module wrapping strategy |
| `transformer_layer_cls_to_wrap` | `str \| None` | Class name (e.g., `"LlamaDecoderLayer"`) |
| `reshard_after_forward` | `bool \| None` | Free shards after forward (ZeRO-3 style memory savings) |
| `mixed_precision_policy` | `str \| None` | e.g., `"fp16"`, `"bf16"` |

#### Validation Pipeline
**File**: `src/axolotl/utils/schemas/validation.py`

18+ model validators enforce FSDP compatibility constraints:

1. **`check_fsdp_deepspeed`** (line 1189): FSDP and DeepSpeed are mutually exclusive
2. **`check_fsdp_version`** (line 1006): Warns FSDP1 is deprecated
3. **`check_fsdp_torch_version`** (config.py line 1720): FSDP2 requires torch >= 2.7.0
4. **`check_fsdp_config_kwargs_prefix`** (line 1052): Strips legacy `fsdp_` prefix from inner keys (e.g., `fsdp_offload_params` → `offload_params`)
5. **`check_fsdp_version_in_fsdp_config`** (line 1074): Syncs `fsdp_version` between top-level and nested config
6. **`check_fsdp2_cpu_offload_pin_memory`** (line 1019): `pin_memory: false` requires FSDP2 + `offload_params: true`
7. **`check_fsdp2_base_model_quant_rl`** (line 1036): FSDP2 + quantization blocked with DPO/KTO/ORPO/IPO
8. **`check_muon_deepspeed_fsdp`** (line 906): Muon optimizer only works with FSDP2
9. **`check_flashoptim_deepspeed_fsdp`** (line 922): Flash optimizers only work with FSDP2
10. **`check_fsdp_offload_w_8bit_optimizer`** (line 1088): FSDP1 offload incompatible with 8-bit optimizers
11. **`check_fsdp2_w_8bit_optimizer`** (line 1103): FSDP2 incompatible with `adamw_8bit`/`adamw_bnb_8bit`; suggests `adamw_torch_8bit`
12. **`check_falcon_fsdp`** (line 1348): FSDP not supported for Falcon
13. **`check_gpt_oss_fsdp_loading`** (line 1430): Mxfp4 incompatible with `cpu_ram_efficient_loading`
14. **`check_relora`** (line 1473): ReLoRA incompatible with FSDP
15. **`check_multigpu_lora_kernels`** (config.py line 1491): LoRA kernels blocked on FSDP1, allowed on FSDP2
16. **`check_fp8_config`** (line 447): FP8 + FSDP2 + activation checkpointing warning; `fp8_enable_fsdp_float8_all_gather` requires FSDP2

#### DictDefault Behavior
After Pydantic validation, config is dumped via `model_dump(exclude_none=True)` and reconverted to `DictDefault` (an `addict.Dict` subclass). This means:
- All `None` fields are stripped
- `cfg.fsdp_config` becomes a plain dict
- Attribute access is via `addict.Dict` (typos silently return `None`)
- Boolean check: `cfg.fsdp_config is not None or cfg.fsdp is not None`

### 4.2 Environment Variable Translation

**File**: `src/axolotl/utils/trainer.py:589-618`

`setup_fsdp_envs(cfg)` is called from `prepare_optim_env()` (line 649) when `cfg.fsdp or cfg.fsdp_config` is truthy. It sets:

| Axolotl Config | Environment Variable |
|---|---|
| (always) | `ACCELERATE_USE_FSDP=true` |
| `fsdp_version: 2` | `FSDP_VERSION=2` |
| `fsdp_config.activation_checkpointing` | `FSDP_ACTIVATION_CHECKPOINTING=true` |
| `fsdp_config.offload_params` | `FSDP_OFFLOAD_PARAMS=true` |
| `fsdp_config.sync_module_states` | `FSDP_SYNC_MODULE_STATES=true` |
| `fsdp_config.cpu_ram_efficient_loading` | `FSDP_CPU_RAM_EFFICIENT_LOADING=true` |
| `fsdp_config.use_orig_params` | `FSDP_USE_ORIG_PARAMS=true` |
| `fsdp_config.state_dict_type` | `FSDP_STATE_DICT_TYPE=<value>` |
| `fsdp_config.cpu_offload_pin_memory` | `FSDP_CPU_OFFLOAD_PIN_MEMORY=<true\|false>` |
| `fsdp_config.auto_wrap_policy` | `FSDP_AUTO_WRAP_POLICY=<value>` |
| `fsdp_config.transformer_layer_cls_to_wrap` | `FSDP_TRANSFORMER_CLS_TO_WRAP=<value>` |
| `fsdp_config.reshard_after_forward` | `FSDP_RESHARD_AFTER_FORWARD=true` |

These are consumed by HuggingFace Accelerate to construct the `FSDPPlugin`.

### 4.3 Monkeypatch System

Axolotl's FSDP support relies heavily on 8 monkeypatch modules applied via `PatchManager`. These are necessary because upstream libraries (Accelerate, PyTorch, PEFT, BnB, TRL) don't fully support the combinations Axolotl needs (FSDP2+QLoRA, cpu_ram_efficient_loading, DTensor+PEFT).

#### 4.3.1 Core FSDP2 Patches (`monkeypatch/accelerate/fsdp2.py`, 539 lines)

**`fsdp2_prepare_model()`** (lines 279-449) — The heart of FSDP2 support. Replaces `accelerate.accelerator.fsdp2_prepare_model`:

1. Checks if already FSDP-wrapped (early return)
2. Saves `original_sd` for later broadcast
3. Configures `auto_wrap_policy` via `fsdp2_plugin.set_auto_wrap_policy(model)`
4. Applies activation checkpointing **before** sharding (using `torch.distributed.algorithms._checkpoint`)
5. Builds `fsdp2_kwargs`: `reshard_after_forward`, `offload_policy` (CPUOffloadPolicy), `mp_policy` (MixedPrecisionPolicy), `mesh` (from device_mesh)
6. Detects `Params4bit` — if present, skips meta-device optimization (BnB params can't be moved to meta)
7. If `cpu_ram_efficient_loading` and no Params4bit:
   - Saves non-persistent buffer FQNs
   - Moves model to `torch.device("meta")`
   - Calls `model.tie_weights()`
8. If PeftModel with ParamWrapper modules: patches `_LoraParameterProxy.forward` for DTensor compatibility
9. Walks modules **bottom-up**:
   - `LoraLayer` → `_process_lora_module_for_fsdp()` (individually shards `lora_A`, `lora_B`, `lora_magnitude_vector`)
   - `auto_wrap_policy(module)` matches → `fully_shard(module, **fsdp2_kwargs)`
10. `fully_shard(model, **fsdp2_kwargs)` — root model sharding
11. If `cpu_ram_efficient_loading`: broadcasts full state dict via `fsdp2_load_full_state_dict()`
12. Re-registers non-persistent buffers, re-ties weights

**`get_state_dict()`** (lines 100-193) — Replaces `Accelerator.get_state_dict`:
- **FSDP2**: Iterates sharded state dict, calls `param.full_tensor()` per DTensor, only rank 0 stores on CPU
- **FSDP1**: Uses `FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig)` context manager
- **DeepSpeed**: Handles Zero3/TP consolidation

**`fsdp2_load_full_state_dict()`** (lines 20-97) — Broadcasts weights from rank 0:
- For DTensor params: `distribute_tensor(full_tensor, device_mesh, placements, src_data_rank=0)`
- For non-sharded params: `dist.broadcast(param, src=0)`
- Clones local shard if storage oversized (to free full tensor memory)
- Deletes `full_sd[param_name]` after each param to free memory incrementally

**`patch_peft_param_wrapper_for_fsdp2()`** (lines 196-232) — Fixes DTensor + regular Tensor mixing in PEFT's `_LoraParameterProxy.forward()`. Promotes the non-DTensor operand using `DTensor.from_local()` (free for Replicate placement).

**`patch_tied_keys_for_meta_device()`** (lines 452-486) — Meta tensors all share `data_ptr()==0`, causing false "tied" detection. Skips meta tensors.

**`patch_initialize_missing_keys_for_fsdp()`** (lines 489-527) — Marks all params with `_is_hf_initialized=True` on non-rank-0 processes to prevent expensive re-initialization. Upstream fix: `transformers/pull/44473`.

#### 4.3.2 FSDP2+QLoRA Patches (`monkeypatch/fsdp2_qlora.py`, 237 lines)

**CRITICAL**: These patches use `inspect.getsource()` + string replacement + `exec()` — extremely fragile.

**`apply_init_sharded_param_patch()`** (lines 19-93) — Patches `FSDPParam._init_sharded_param`:
- Original: wraps with `nn.Parameter(to_sharded_dtensor(sharded_param))` which destroys BnB metadata
- Patched: Checks parameter type, constructs `Params4bit` or `Int8Params` with all quantization metadata preserved (quant_state, blocksize, compress_statistics, quant_type, quant_storage, bnb_quantized)

**`apply_init_unsharded_param_patch()`** (lines 96-169) — Same for unshard path (`FSDPParam.init_unsharded_param`):
- Accesses `self.sharded_param._local_tensor` to get the BnB parameter type
- Reconstructs with metadata from the local shard

**`apply_init_dtype_attrs_patch()`** (lines 205-236) — Prevents FSDP2 mixed precision from casting non-float quantized params:
- Without patch: FSDP2 sets `param_dtype=bf16` for ALL params, destroying uint8/int8 quantized data
- Fix: Sets `self.param_dtype = None` for non-floating-point params lacking `fsdp_pre_all_gather` extensions

**`apply_linear8bitlt_save_patch()`** (lines 172-202) — Temporarily unwraps DTensor during `Linear8bitLt._save_to_state_dict` so BnB can find the `SCB` attribute.

#### 4.3.3 Other FSDP Patches

**`monkeypatch/accelerate/parallelism_config.py`** (99 lines):
- `patched_is_fsdp2()`: Guards against `fsdp_plugin` being `None`
- `_validate_accelerator()`: Allows standalone CP without FSDP when `ACCELERATE_ALLOW_CP_STANDALONE=true`

**`monkeypatch/trainer/trl.py`** (13 lines):
- Redirects `trl.models.utils.prepare_fsdp` to Axolotl's `fsdp2_prepare_model`

**`monkeypatch/trainer_accelerator_args.py`** (84 lines):
- Injects `enable_fsdp_float8_all_gather` into Accelerator constructor for FP8 support

**`monkeypatch/torchao_optim.py`** (155 lines):
- Fixes DTensor view operations for `OptimState8bit`, `OptimState4bit`, `OptimStateFp8`
- Required for FSDP2 + torchao low-bit optimizers (upstream fix: `pytorch/ao/pull/4216`)

#### 4.3.4 Patch Application Order

In `PatchManager.apply_pre_model_load_patches()`:

1. `_apply_torchao_patches()` — DTensor optimizer state compat
2. `_apply_fsdp_patches()`:
   - `patch_initialize_missing_keys_for_fsdp()` (always with any FSDP config)
   - `patch_parallelism_config()` (FSDP2 or CP)
   - `patch_accelerate_fsdp2()` (FSDP2 only)
   - `patch_tied_keys_for_meta_device()` (FSDP2 + cpu_ram_efficient)
   - `patch_trl_prepare_fsdp2()` (FSDP2 + RL)
3. Other patches (attention, model-specific, etc.)
4. `_apply_fsdp2_bnb_patches()` (FSDP2 + 4bit/8bit):
   - `apply_init_sharded_param_patch()`
   - `apply_init_unsharded_param_patch()`
   - `apply_init_dtype_attrs_patch()`
   - `apply_linear8bitlt_save_patch()` (8bit only)

### 4.4 Model Loading with FSDP

**File**: `src/axolotl/loaders/model.py`

Key properties (lines 152-159):
```python
@property
def is_fsdp_enabled(self):
    return self.cfg.fsdp_config is not None or self.cfg.fsdp is not None

@property
def is_qlora_and_fsdp_enabled(self):
    return self.is_fsdp_enabled and self.cfg.adapter == "qlora"
```

**Device map configuration** (lines 498-506):
- FSDP without QLoRA: No `device_map` (FSDP controls placement)
- QLoRA+FSDP: `device_map = {"": LOCAL_RANK}`
- FSDP2 + cpu_ram_efficient_loading (lines 769-779): Rank 0 → `"cpu"`, others → `"meta"`

**Embedding dtype** (lines 312-321): Embedding float32 conversion is **skipped** under FSDP to avoid mixed dtypes.

**kbit training prep** (lines 895-900): `prepare_model_for_kbit_training` is **skipped** under QLoRA+FSDP or FSDP+cpu_ram_efficient_loading (would create mixed dtypes).

**Sharded quantized loading** (lines 781-808): For QLoRA+FSDP with specific architectures, `load_sharded_model_quant()` (`utils/model_shard_quant.py`) loads and quantizes weights shard-by-shard.

### 4.5 Adapter Loading with FSDP

**File**: `src/axolotl/loaders/adapter.py`

When FSDP + `cpu_ram_efficient_loading` and rank != 0 (lines 305-356):
1. **Before** `get_peft_model()`: `setup_quantized_meta_for_peft(model)` — Replaces `quant_state.to` with a no-op to prevent PEFT from moving quant_state to meta device
2. **Call** `get_peft_model(model, lora_config)` — PEFT wraps the model
3. **After**: `setup_quantized_peft_meta_for_training(model)` — Restores original `quant_state.to`

### 4.6 Training Execution with FSDP

**TrainerBuilder** (`core/builders/base.py:604-606`): Passes `fsdp_config` and `fsdp=True` to `TrainingArguments`. HuggingFace creates `FSDPPlugin` internally.

**Trainer Mixins**:

1. **`DistributedParallelMixin`** (`core/trainers/mixins/distributed_parallel.py:9-33`):
   - `_save()`: When FSDP is active (dp_shard_enabled), calls `accelerator.get_state_dict(model)` for full state dict gathering
   - `create_accelerator_and_postprocess()`: Downgrades FSDP → MULTI_GPU if `fsdp_plugin` is None (pure Context Parallelism)

2. **`CheckpointSaveMixin`** (`core/trainers/mixins/checkpoints.py:10-23`):
   - Wraps `_save_optimizer_and_scheduler` in try/except for `NotImplementedError`/`KeyError`
   - FSDP2 optimizer saving is known-incomplete (TODO comment at line 17)

3. **`ActivationOffloadingMixin`** (`core/trainers/mixins/activation_checkpointing.py`):
   - Uses `torch.distributed.fsdp.wrap.ModuleWrapPolicy` + `apply_activation_checkpointing` for FSDP-compatible checkpointing

### 4.7 Post-Training Save

**File**: `src/axolotl/train.py:294-349`

1. Selects `final_state_dict_type` (fallback to `state_dict_type`)
2. Sets on `fsdp_plugin` via `set_state_dict_type()`
3. Calls `trainer.save_model(output_dir)`
4. For `SHARDED_STATE_DICT`: Merges shards via `merge_fsdp_weights()`, renames for PEFT models
5. Strips `FSDP` prefix from `config.json` architecture names (lines 336-349)

**Merge utility** (`cli/merge_sharded_fsdp_weights.py`):
- `BFloat16CastPlanner`: Custom planner casting to bf16 during DCP loading
- `_distributed_checkpoint_to_merged_weights()`: Loads DCP, saves as merged safetensors
- Available as CLI: `axolotl merge-sharded-fsdp-weights`

---

## 5. Pseudocode Restatement

### 5.1 FSDP2 Model Preparation (Core Algorithm)

```python
def fsdp2_prepare_model(accelerator, model):
    fsdp2_plugin = accelerator.state.fsdp_plugin
    original_sd = model.state_dict()

    # Configure auto-wrap policy
    fsdp2_plugin.set_auto_wrap_policy(model)

    # Apply activation checkpointing BEFORE sharding
    if fsdp2_plugin.activation_checkpointing:
        apply_activation_checkpointing(model, auto_wrap_policy=fsdp2_plugin.auto_wrap_policy)

    # Build FSDP kwargs
    mesh = accelerator.state.device_mesh[parallelism_config.fsdp_dim_names] if device_mesh else None
    fsdp2_kwargs = {
        reshard_after_forward, offload_policy, mp_policy, mesh
    }

    # Detect if model has Params4bit (can't move to meta)
    model_has_params4bit = any(p.__class__.__name__ == "Params4bit" for p in model.parameters())

    # cpu_ram_efficient_loading: move to meta device
    if cpu_ram_efficient_loading and not model_has_params4bit:
        non_persistent_buffers = get_non_persistent_buffers(model)
        model = model.to("meta")
        model.tie_weights()

    # Patch PEFT ParamWrapper if present
    if isinstance(model, PeftModel) and has_ParamWrapper(model):
        patch_peft_param_wrapper_for_fsdp2()

    # Walk modules bottom-up and shard
    auto_wrap_policy = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)
    for module in get_module_children_bottom_up(model)[:-1]:
        if isinstance(module, LoraLayer):
            _process_lora_module_for_fsdp(module, fsdp2_kwargs)
        if auto_wrap_policy(module) and not isinstance(module, FSDPModule):
            fully_shard(module, **fsdp2_kwargs)

    fully_shard(model, **fsdp2_kwargs)  # Root model

    # Broadcast weights from rank 0
    if cpu_ram_efficient_loading:
        fsdp2_load_full_state_dict(accelerator, model, original_sd)
        # Re-register buffers, re-tie weights
        for fqn, buffer in non_persistent_buffers.items():
            parent.register_buffer(name, buffer.to(device), persistent=False)
        model.tie_weights()

    return model
```

### 5.2 FSDP2 State Dict Gathering

```python
def get_state_dict(self, model):
    if self.is_fsdp2:
        state_dict = {}
        for param_name, param in model.state_dict().items():
            if param.is_cpu:
                param = param.to("cuda")
            if isinstance(param, DTensor):
                param = param.full_tensor()  # All-gather
            if rank == 0:
                state_dict[param_name] = param.cpu()
            dist.barrier()
        return state_dict

    elif distributed_type == FSDP:  # FSDP1
        with FSDP.state_dict_type(model, FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            return model.state_dict()
```

### 5.3 FSDP2 Full State Dict Broadcast

```python
def fsdp2_load_full_state_dict(accelerator, model, full_sd):
    meta_sharded_sd = model.state_dict()  # On meta device

    for param_name, sharded_meta_param in meta_sharded_sd.items():
        if is_main_process:
            full_tensor = full_sd[param_name].to(sharded_meta_param.dtype)
        else:
            full_tensor = torch.empty(sharded_meta_param.size(), ...)

        if hasattr(sharded_meta_param, "device_mesh"):
            # DTensor: distribute according to placement spec
            sharded_param = distribute_tensor(full_tensor, device_mesh, placements, src_data_rank=0)
            if oversized_storage(sharded_param):
                sharded_param = sharded_param.clone()  # Free full tensor memory
        else:
            # Non-sharded: manual broadcast
            sharded_param = full_tensor.to("cuda")
            dist.broadcast(sharded_param, src=0)

        sharded_sd[param_name] = nn.Parameter(sharded_param)
        full_sd[param_name] = None  # Free memory incrementally

    model.load_state_dict(sharded_sd, assign=True, strict=True)
```

---

## 6. Configuration Items and Behavior Switches

### 6.1 YAML Configuration Fields

| Config Field | Type | Default | Effect |
|---|---|---|---|
| `fsdp` | `list[str]` | `None` | **Deprecated**. Old-style FSDP flags |
| `fsdp_config` | `FSDPConfig` | `None` | Main FSDP configuration block |
| `fsdp_version` | `int` | `None` | `1` or `2`; can be set at top-level or inside `fsdp_config` |
| `fsdp_config.activation_checkpointing` | `bool` | `None` | FSDP-compatible activation checkpointing |
| `fsdp_config.offload_params` | `bool` | `None` | Offload parameters to CPU |
| `fsdp_config.sync_module_states` | `bool` | `None` | Synchronize module states across processes |
| `fsdp_config.cpu_ram_efficient_loading` | `bool` | `None` | Rank 0 loads weights; broadcast to others |
| `fsdp_config.cpu_offload_pin_memory` | `bool` | `None` | `false` enables swap memory |
| `fsdp_config.use_orig_params` | `bool` | `None` | Use original parameters (required for some features) |
| `fsdp_config.state_dict_type` | `str` | `None` | `FULL_STATE_DICT`, `LOCAL_STATE_DICT`, `SHARDED_STATE_DICT` |
| `fsdp_config.final_state_dict_type` | `str` | `None` | State dict type for final save |
| `fsdp_config.auto_wrap_policy` | `str` | `None` | `TRANSFORMER_BASED_WRAP` or `SIZE_BASED_WRAP` |
| `fsdp_config.transformer_layer_cls_to_wrap` | `str` | `None` | e.g., `"LlamaDecoderLayer"`, `"Qwen2DecoderLayer"` |
| `fsdp_config.reshard_after_forward` | `bool` | `None` | Free memory after forward (ZeRO-3 style) |
| `fsdp_config.mixed_precision_policy` | `str` | `None` | e.g., `"bf16"` |
| `fsdp_final_state_dict_type` | `str` | `None` | **Deprecated**. Use `fsdp_config.final_state_dict_type` |
| `fp8_enable_fsdp_float8_all_gather` | `bool` | `None` | FP8 all-gather with FSDP2 |
| `dp_shard_size` | `int` | `None` | Data-parallel shard dimension (mesh) |
| `dp_replicate_size` | `int` | `None` | Data-parallel replicate dimension (mesh) |
| `context_parallel_size` | `int` | `None` | Context/sequence parallelism dimension |
| `tensor_parallel_size` | `int` | `None` | Tensor parallelism dimension |

### 6.2 Environment Variables

| Variable | Set By | Read By | Purpose |
|---|---|---|---|
| `ACCELERATE_USE_FSDP` | `setup_fsdp_envs` | Accelerate | Master FSDP enable switch |
| `FSDP_VERSION` | `setup_fsdp_envs` | Accelerate | Select FSDP v1 or v2 |
| `FSDP_ACTIVATION_CHECKPOINTING` | `setup_fsdp_envs` | Accelerate | Activation checkpointing |
| `FSDP_OFFLOAD_PARAMS` | `setup_fsdp_envs` | Accelerate | CPU offloading |
| `FSDP_SYNC_MODULE_STATES` | `setup_fsdp_envs` | Accelerate | Module state sync |
| `FSDP_CPU_RAM_EFFICIENT_LOADING` | `setup_fsdp_envs` | Accelerate | Efficient loading |
| `FSDP_USE_ORIG_PARAMS` | `setup_fsdp_envs` | Accelerate | Original params |
| `FSDP_STATE_DICT_TYPE` | `setup_fsdp_envs` | Accelerate | State dict format |
| `FSDP_CPU_OFFLOAD_PIN_MEMORY` | `setup_fsdp_envs` | `fsdp2_prepare_model` | Pin memory toggle |
| `FSDP_AUTO_WRAP_POLICY` | `setup_fsdp_envs` | Accelerate | Wrap policy |
| `FSDP_TRANSFORMER_CLS_TO_WRAP` | `setup_fsdp_envs` | Accelerate | Layer class |
| `FSDP_RESHARD_AFTER_FORWARD` | `setup_fsdp_envs` | Accelerate | Reshard behavior |
| `ACCELERATE_USE_PARALLELISM_CONFIG` | `setup_parallelism_envs` | Accelerate | Multi-dim parallelism |
| `PARALLELISM_CONFIG_DP_SHARD_SIZE` | `setup_parallelism_envs` | Accelerate | DP shard dim |
| `PARALLELISM_CONFIG_DP_REPLICATE_SIZE` | `setup_parallelism_envs` | Accelerate | DP replicate dim |
| `PARALLELISM_CONFIG_TP_SIZE` | `setup_parallelism_envs` | Accelerate | TP dim |
| `PARALLELISM_CONFIG_CP_SIZE` | `setup_parallelism_envs` | Accelerate | CP dim |
| `ACCELERATE_ALLOW_CP_STANDALONE` | `setup_parallelism_envs` | Patched validator | Allow CP without FSDP |
| `ACCELERATE_MIXED_PRECISION` | `prepare_optim_env` | Accelerate | bf16/fp16/fp8/no |
| `WORLD_SIZE` | torchrun/accelerate | `normalize_config` | Batch size scaling |
| `LOCAL_RANK` | torchrun/accelerate | `ModelLoader` | Device placement |
| `NCCL_P2P_DISABLE` | `prepare_optim_env` | NCCL | P2P support check |
| `AXOLOTL_NCCL_TIMEOUT` | User | `distributed.py` | NCCL timeout |

---

## 7. Test and Example Analysis

### 7.1 Test Coverage

| Test File | Test Class | Tests | FSDP Version | Coverage |
|---|---|---|---|---|
| `tests/e2e/multigpu/test_fsdp2.py` | `TestFSDP2` | 7 tests | FSDP2 | FFT SFT (±cpu_efficient), LoRA (±DoRA), LoRA+kernels, QLoRA, QLoRA+kernels, DPO FFT (skipped), DPO LoRA (skipped) |
| `tests/e2e/multigpu/test_fsdp1.py` | `TestFSDP1` | 4 tests | FSDP1 | FFT SFT (±cpu_efficient), LoRA/QLoRA SFT, DPO FFT (skipped), DPO LoRA (skipped/broken in transformers v5) |
| `tests/e2e/multigpu/test_fsdp2_lora_kernels.py` | `TestFSDP2LoRAKernels` | Multiple | FSDP2 | LoRA kernels with various configs (dropout, DoRA, bias) |
| `tests/e2e/multigpu/test_fp8_fsdp2.py` | `TestFP8FSDP2` | 1 test | FSDP2 | FP8 + torch.compile + FSDP2 smoke test |
| `tests/e2e/multigpu/test_dist_muon_fsdp2.py` | - | 2 tests | FSDP2 | DistMuon optimizer with FSDP2 (FFT and LoRA) |
| `tests/e2e/patched/test_fsdp2_qlora.py` | `TestFSDPPatchIntegration` | Patch verification | FSDP2 | Verifies FSDP2 patches actually modify FSDPParam methods |
| `tests/e2e/multigpu/test_llama.py:324-601` | - | 4 tests | Both | LLaMA-specific FSDP1/FSDP2 multi-GPU tests |
| `tests/e2e/multigpu/test_ray.py:149-178` | - | 1 test | FSDP2 | Ray + FSDP2 packed |
| `tests/utils/schemas/validation/test_fsdp.py` | `TestFSDPValidation` | 7+ tests | Both | Version resolution, offload+8bit, cpu_efficient, pin_memory, prefix strip, Muon+FSDP1, FSDP2+DPO RL |
| `tests/cli/test_cli_merge_sharded_fsdp_weights.py` | - | 4 tests | - | CLI merge command (no accelerate, torchrun, accelerate, backward compat) |
| `tests/test_normalize_config.py:105-196` | - | Multiple | - | fsdp_config migration (prefix removal, version extraction) |
| `tests/patched/test_validation.py:471-493` | - | 1 test | - | Falcon FSDP rejection |
| `tests/test_validation_dataset.py:349-367` | - | 1 test | - | Muon+FSDP1 rejection |

### 7.2 Verification Pattern

All e2e FSDP tests follow this pattern:
1. Create config as `DictDefault`
2. Write to YAML file
3. Launch via `execute_subprocess_async(["axolotl", "train", ...])`
4. Call `verify_training_success(temp_dir)`:
   - Assert model files exist (`.bin` or `.safetensors`)
   - Assert checkpoint files exist
   - Read TensorBoard logs, assert loss is not NaN

### 7.3 Missing Test Coverage

1. **Multi-node testing**: No multi-node e2e tests (all use `--num-processes 2` on single node)
2. **SHARDED_STATE_DICT round-trip**: No test verifying save→merge→load cycle
3. **cpu_offload_pin_memory=false (swap)**: Not tested in e2e
4. **Checkpoint resume with FSDP2**: Not tested (optimizer save is TODO)
5. **FSDP2 + 8bit LoRA**: Only patch verification test, no training e2e
6. **FSDP2 DPO/RL tests**: Skipped with reason "slow test"
7. **Monkeypatch cleanup**: `tests/conftest.py:cleanup_monkeypatches` does NOT clean up FSDP patches — only restores LlamaAttention, LlamaForCausalLM, and Trainer methods
8. **SIZE_BASED_WRAP policy**: No tests use this policy
9. **mixed_precision_policy field**: Not tested explicitly
10. **FP8 + FSDP + activation_checkpointing**: Only a warning in validation, no negative test

### 7.4 Example YAML Configs

35+ example files with FSDP configurations exist in `examples/`, including:
- `llama-3/qlora-fsdp-70b.yaml`, `llama-3/qlora-fsdp-405b.yaml` — Large model QLoRA+FSDP
- `llama-3/3b-fp8-fsdp2.yaml` — FP8 + FSDP2
- `qwen2/muon-pretrain-fsdp2.yaml` — Muon optimizer + FSDP2
- `distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml` — FSDP + TP + CP hybrid parallelism
- `gpt-oss/gpt-oss-120b-fft-fsdp2-offload.yaml` — Large scale FSDP2 with offloading

---

## 8. Design Evaluation

### 8.1 Strengths

1. **Comprehensive FSDP2 support**: Axolotl's FSDP2 support covers FFT, LoRA, QLoRA, FP8, LoRA kernels, and RL training — more combinations than most frameworks
2. **cpu_ram_efficient_loading**: The rank-0-loads-then-broadcasts pattern dramatically reduces memory footprint during initialization
3. **Unified configuration**: A single `fsdp_config` block in YAML controls all FSDP behavior, with clean validation
4. **Backward compatibility**: Legacy `fsdp_` prefix keys are auto-stripped; `fsdp` list form still works
5. **Incremental memory management**: `fsdp2_load_full_state_dict()` frees `full_sd[param_name] = None` after each parameter broadcast — critical for large models
6. **Hierarchical sharding**: Bottom-up module walk ensures proper parent-child FSDP wrapping order
7. **Plugin architecture**: FSDP patches are applied via PatchManager lifecycle, not global module-level side effects

### 8.2 Complexity

1. **8 monkeypatch modules** touching 5 upstream libraries (Accelerate, PyTorch, PEFT, BnB, TRL)
2. **Source code string replacement** in 3 files (`fsdp2_qlora.py`, `trainer_fsdp_optim.py`, `trainer_accelerator_args.py`) using `inspect.getsource()` + `exec()`
3. **Multi-layer config translation**: YAML → Pydantic → DictDefault → env vars → Accelerate FSDPPlugin → PyTorch FSDP
4. **Conditional branching**: ~15 distinct code paths based on FSDP v1 vs v2, QLoRA, cpu_ram_efficient_loading, Params4bit presence, RL training
5. **Type erasure**: After Pydantic validation, `model_dump(exclude_none=True)` back to `DictDefault` loses type safety

### 8.3 Potential Risks

1. **Source code patching fragility**: The `inspect.getsource() + exec()` pattern in `fsdp2_qlora.py` breaks on ANY whitespace, naming, or logic change in PyTorch's `FSDPParam._init_sharded_param`. This is the highest-risk patch in the codebase.
2. **No version guards on FSDP2 patches**: Unlike the documented monkeypatch rules requiring version checks, FSDP patches only use idempotency flags (`_axolotl_patched`). A PyTorch upgrade could silently break QLoRA+FSDP2.
3. **FSDP2 optimizer saving is incomplete**: `CheckpointSaveMixin` catches `NotImplementedError` and logs a warning — checkpoint resume may not work for some configurations.
4. **Monkeypatch cleanup gap**: `tests/conftest.py:cleanup_monkeypatches` does not restore any FSDP patches, risking test cross-contamination.
5. **Meta-device optimization skipped for Params4bit**: When QLoRA is used, the cpu_ram_efficient_loading meta-device optimization is bypassed, causing a VRAM spike during loading (acknowledged in code comments).
6. **DTensor barrier per-parameter**: `get_state_dict()` calls `torch.distributed.barrier()` after each parameter's `full_tensor()`, which could be slow for models with many parameters.

### 8.4 Extensibility

1. **Adding new model architectures**: Requires adding `transformer_layer_cls_to_wrap` value to YAML config — no code changes needed
2. **Adding new FSDP features**: Requires adding field to `FSDPConfig`, env var to `setup_fsdp_envs()`, and handler in `fsdp2_prepare_model()`
3. **Plugin-aware**: Integrations like Liger, Spectrum, and SwanLab correctly interact with FSDP (e.g., Spectrum requires `use_orig_params=True`, SwanLab logs `fsdp_enabled`)

### 8.5 Maintainability

1. **Strong**: Configuration validation catches most invalid combinations early with clear error messages
2. **Weak**: Monkeypatches have upstream version coupling documented in comments (e.g., "Remove once transformers includes fix") but no automated alerts
3. **Medium**: The test suite covers the main paths but skips RL+FSDP and multi-node scenarios
4. **Concerning**: Three separate comment blocks reference upstream PRs to track for removal — these could become stale

### 8.6 Performance Considerations

1. **`reshard_after_forward: true`** — Trades compute (extra all-gather) for memory (ZeRO-3 style). Critical for fitting large models.
2. **`cpu_ram_efficient_loading`** — Saves N-1 copies of model weights during initialization but adds broadcast overhead
3. **DTensor per-parameter barrier** — In `get_state_dict()`, could be batched for efficiency
4. **Params4bit VRAM spike** — Meta-device optimization bypassed; acknowledged limitation
5. **`cpu_offload_pin_memory: false`** — Enables swap at the cost of ~2-5x slower data transfers
6. **Activation checkpointing applied before sharding** — Correct order ensures FSDP wraps checkpoint-aware modules

---

## 9. Unconfirmed Items

1. **Multi-node FSDP testing in CI**: Not confirmed whether any CI infrastructure runs multi-node tests. All observed e2e tests use `--num-processes 2` which is single-node multi-GPU.

2. **FSDP2 + SHARDED_STATE_DICT resume**: The `merge_fsdp_weights` utility exists for final merge, but whether mid-training checkpoint resume works with SHARDED_STATE_DICT on FSDP2 is not confirmed in tests.

3. **SIZE_BASED_WRAP policy behavior**: The schema supports it but no tests or examples use this policy. Whether it works correctly with the bottom-up module walk in `fsdp2_prepare_model` is not confirmed.

4. **mixed_precision_policy field usage**: The field exists in `FSDPConfig` but is never referenced in `setup_fsdp_envs()` or any patch code. It's unclear whether this field has any effect — Accelerate may read it from the environment variable translation of `ACCELERATE_MIXED_PRECISION`.

5. **DeepSpeed configs and FSDP coexistence**: Confirmed mutually exclusive by validator, but the `deepspeed_configs/` directory only contains DeepSpeed configs — no FSDP equivalents.

6. **Performance overhead of per-parameter barrier**: In `get_state_dict()`, `torch.distributed.barrier()` is called after each parameter. The performance impact for models with thousands of parameters is not benchmarked in the codebase.

7. **`fsdp_config.final_state_dict_type` vs `fsdp_final_state_dict_type`**: The deprecated top-level field is documented but the actual migration/forwarding logic (whether the deprecated field value is copied to `fsdp_config.final_state_dict_type` during normalization) is not confirmed in the validation code.

---

## Appendix A: Canonical FSDP2 YAML Configuration

```yaml
# Based on test_fsdp2.py canonical pattern
base_model: Qwen/Qwen2.5-0.5B
sequence_len: 2048

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

adapter: lora          # or qlora, or omit for FFT
lora_r: 8
lora_alpha: 16
lora_target_linear: true

fsdp_version: 2
fsdp_config:
  offload_params: false
  cpu_ram_efficient_loading: true
  transformer_layer_cls_to_wrap: Qwen2DecoderLayer
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  reshard_after_forward: true

optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 0.00001
bf16: true
flash_attention: true
micro_batch_size: 2
gradient_accumulation_steps: 1
num_epochs: 1
```

## Appendix B: FSDP-Related File Index

```
Configuration:
  src/axolotl/utils/schemas/fsdp.py          # FSDPConfig Pydantic model
  src/axolotl/utils/schemas/config.py         # Top-level FSDP fields
  src/axolotl/utils/schemas/validation.py     # 18+ FSDP validators
  src/axolotl/utils/trainer.py                # setup_fsdp_envs(), setup_parallelism_envs()
  src/axolotl/utils/distributed.py            # build_parallelism_config(), DeviceMesh

Monkeypatches:
  src/axolotl/monkeypatch/accelerate/fsdp2.py           # Core FSDP2 patches (539 lines)
  src/axolotl/monkeypatch/fsdp2_qlora.py                # FSDP2+QLoRA patches (237 lines)
  src/axolotl/monkeypatch/accelerate/parallelism_config.py  # is_fsdp2, CP patches
  src/axolotl/monkeypatch/trainer/trl.py                # TRL FSDP2 redirect
  src/axolotl/monkeypatch/trainer_fsdp_optim.py         # FSDP optimizer fix (disabled)
  src/axolotl/monkeypatch/trainer_accelerator_args.py   # FP8+FSDP
  src/axolotl/monkeypatch/torchao_optim.py              # DTensor optimizer compat

Model Loading:
  src/axolotl/loaders/model.py                # ModelLoader FSDP logic
  src/axolotl/loaders/adapter.py              # Adapter loading with FSDP
  src/axolotl/loaders/patch_manager.py        # Patch orchestration
  src/axolotl/utils/model_shard_quant.py      # Sharded quantized loading

Training:
  src/axolotl/core/builders/base.py           # TrainingArguments setup
  src/axolotl/core/builders/rl.py             # RL builder FSDP handling
  src/axolotl/core/trainers/mixins/distributed_parallel.py  # _save(), accelerator postprocess
  src/axolotl/core/trainers/mixins/checkpoints.py          # Optimizer save resilience
  src/axolotl/core/trainers/mixins/activation_checkpointing.py  # FSDP-compatible AC

Post-Training:
  src/axolotl/train.py                        # save_trained_model()
  src/axolotl/cli/merge_sharded_fsdp_weights.py  # Sharded checkpoint merge

CLI:
  src/axolotl/cli/main.py                     # merge-sharded-fsdp-weights command
  src/axolotl/cli/utils/train.py              # Distributed launch

Tests:
  tests/e2e/multigpu/test_fsdp2.py            # FSDP2 e2e tests
  tests/e2e/multigpu/test_fsdp1.py            # FSDP1 e2e tests
  tests/e2e/multigpu/test_fsdp2_lora_kernels.py  # LoRA kernels + FSDP2
  tests/e2e/multigpu/test_fp8_fsdp2.py        # FP8 + FSDP2
  tests/e2e/multigpu/test_dist_muon_fsdp2.py  # Muon + FSDP2
  tests/e2e/patched/test_fsdp2_qlora.py       # Patch integration test
  tests/utils/schemas/validation/test_fsdp.py # Validation tests
  tests/cli/test_cli_merge_sharded_fsdp_weights.py  # CLI merge tests

Documentation:
  docs/fsdp_qlora.qmd                         # FSDP+QLoRA guide

Examples (35+ files):
  examples/llama-3/qlora-fsdp-70b.yaml
  examples/llama-3/3b-fp8-fsdp2.yaml
  examples/qwen2/muon-pretrain-fsdp2.yaml
  examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml
  ... (and 30+ more)
```
