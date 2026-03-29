# PR Review: Change Type Detection Reference

Referenced by: `.claude/commands/review-pr.md`

---

## CRITICAL Level (Must use Opus)

| Change Type              | File Path Pattern                                  | Code Pattern                                              |
| ------------------------ | -------------------------------------------------- | --------------------------------------------------------- |
| **MONKEYPATCH_CORE**     | `src/axolotl/monkeypatch/`                         | Module replacement, `inspect.getsource`, `exec()`         |
| **PATCH_MANAGER**        | `src/axolotl/loaders/patch_manager.py`             | `PatchManager`, `apply_*_patches`                         |
| **FSDP2_PATCHES**        | `monkeypatch/accelerate/fsdp2.py`, `fsdp2_qlora.py`| `fully_shard`, `FSDP`, bitsandbytes internals            |
| **ATTENTION_PATCHES**    | `monkeypatch/attention/`                           | Flash Attention, Flex, SDPA, xformers replacements        |
| **MODEL_SPECIFIC_PATCH** | `monkeypatch/models/`                              | Model-specific forward method replacements                |
| **ASYNC_GRPO**           | `core/trainers/grpo/async_trainer.py`              | `AxolotlAsyncGRPOTrainer`, multi-process sync             |

## HIGH Level (Recommend Opus)

| Change Type              | File Path Pattern                                  | Code Pattern                                              |
| ------------------------ | -------------------------------------------------- | --------------------------------------------------------- |
| **TRAINER_CORE**         | `src/axolotl/core/trainers/`                       | `AxolotlTrainer`, `AxolotlDPOTrainer`, trainer mixins     |
| **MODEL_LOADER**         | `src/axolotl/loaders/model.py`                     | `ModelLoader`, `_build_model`, `_set_device_map_config`   |
| **ADAPTER_LOADING**      | `src/axolotl/loaders/adapter.py`                   | `load_lora`, `load_adapter`, PEFT integration             |
| **CONFIG_SCHEMA**        | `src/axolotl/utils/schemas/`                       | `AxolotlInputConfig`, `ValidationMixin`, validators       |
| **BUILDER_PATTERN**      | `src/axolotl/core/builders/`                       | `TrainerBuilderBase`, `HFCausalTrainerBuilder`            |
| **TRAINING_ARGS**        | `src/axolotl/core/training_args.py`                | `AxolotlTrainingArguments`                                |
| **DISTRIBUTED**          | `src/axolotl/utils/distributed.py`                 | Process groups, device mesh, parallelism config           |
| **MULTIPACK**            | `src/axolotl/monkeypatch/multipack.py`             | `SUPPORTED_MULTIPACK_MODEL_TYPES`, packing logic          |

## MEDIUM Level (Use Sonnet)

| Change Type              | File Path Pattern                                  | Code Pattern                                              |
| ------------------------ | -------------------------------------------------- | --------------------------------------------------------- |
| **PROMPT_STRATEGY**      | `src/axolotl/prompt_strategies/`                   | `PromptTokenizingStrategy`, `ChatTemplatePrompter`        |
| **DATA_LOADING**         | `src/axolotl/utils/data/`                          | `prepare_datasets`, `load_datasets`                       |
| **COLLATOR**             | `src/axolotl/utils/collators/`                     | `DataCollatorForSeq2Seq`, `MultiModalChatDataCollator`    |
| **PLUGIN_SYSTEM**        | `src/axolotl/integrations/`                        | `BasePlugin`, `PluginManager`, plugin lifecycle           |
| **CALLBACK**             | `src/axolotl/utils/callbacks/`                     | Training callbacks                                        |
| **CLI**                  | `src/axolotl/cli/`                                 | Click commands, config loading                            |
| **TRAINING_ORCHESTRATION** | `src/axolotl/train.py`                           | `train()`, `setup_model_and_tokenizer()`                  |
| **KERNELS**              | `src/axolotl/kernels/`                             | Triton kernels (LoRA, DoRA, SwiGLU)                       |
| **GRADIENT_CHECKPOINT**  | `monkeypatch/gradient_checkpointing/`              | CPU/disk gradient checkpoint offloading                   |
| **RING_ATTENTION**       | `monkeypatch/ring_attn/`                           | Ring attention for sequence parallelism                   |

## LOW Level (Use Haiku)

| Change Type     | File Path Pattern              | Code Pattern |
| --------------- | ------------------------------ | ------------ |
| **TESTS**       | `tests/`                       | -            |
| **DOCS**        | `docs/`, `*.md`                | -            |
| **EXAMPLES**    | `examples/`                    | -            |
| **CONFIG_ONLY** | `*.yaml`, `*.json`, `*.toml`   | -            |

---

## Framework-Specific Risk Identification

### Monkeypatch Risks

- **Version coupling**: Patch targets specific upstream internal API that can change
- **Order dependency**: PatchManager applies patches in specific sequence
- **Global state mutation**: Some patches modify module-level globals
- **Source code patching**: `inspect.getsource()` + `exec()` (extremely fragile)
- **Missing cleanup**: New patch without conftest.py cleanup fixture

### Model Loading Risks

- **Device map conflicts**: Incorrect device mapping for multi-GPU setups
- **Quantization incompatibility**: BitsAndBytes version vs model architecture
- **Adapter target mismatch**: LoRA target modules don't exist in model
- **Multimodal model mapping**: Missing `MULTIMODAL_AUTO_MODEL_MAPPING` entry

### Config Schema Risks

- **DictDefault silent None**: Typo in config key access returns None
- **Validation gap**: Config validated by Pydantic but accessed via DictDefault
- **Plugin config conflict**: Multiple plugins defining same config fields
- **Cross-field dependency**: Validation rule doesn't catch invalid combinations

### Trainer Risks

- **Mixin MRO issues**: Incorrect class declaration ordering
- **Callback duplication**: Same callback registered via both builder and plugin
- **State dict format**: FSDP/DeepSpeed checkpoint incompatibility

---

## Risk Linkage Rules

| Detected Change             | Auto-Linked Review                                  |
| --------------------------- | --------------------------------------------------- |
| Monkeypatch changes         | Version guard check, conftest cleanup check         |
| Model loader changes        | PatchManager interaction check                      |
| Config schema changes       | Validation rule check, DictDefault access check     |
| Trainer changes             | Mixin MRO check, builder integration check          |
| Multipack changes           | Model type list completeness check                  |
| Plugin changes              | Lifecycle hook ordering check                       |
| Attention patch changes     | Model compatibility check                           |
| FSDP2 changes               | QLoRA patch interaction check                       |

---

## Core Framework Paths (Must Use Opus)

**Monkeypatch Core**:
- `src/axolotl/monkeypatch/` (entire directory)
- `src/axolotl/loaders/patch_manager.py`

**Training Core**:
- `src/axolotl/core/trainers/`
- `src/axolotl/core/builders/`
- `src/axolotl/train.py`

**Model Loading Core**:
- `src/axolotl/loaders/model.py`
- `src/axolotl/loaders/adapter.py`

**Config Core**:
- `src/axolotl/utils/schemas/config.py`
- `src/axolotl/utils/schemas/validation.py`
