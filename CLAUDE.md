# CLAUDE.md - Axolotl

## WHAT: Project Overview

Axolotl is an open-source LLM fine-tuning framework that streamlines post-training for
large language models via a single YAML configuration file.

**Tech Stack**: Python 3.10+ | PyTorch 2.9+ | HuggingFace Transformers 5.x | TRL | PEFT
| Accelerate

**Core Directories**:

- `src/axolotl/` - Main source package
  - `cli/` - Click CLI entry points and config loading
  - `core/` - Trainers, builders, training args, attention, chat formatting
    - `trainers/` - AxolotlTrainer + RL trainers (DPO, GRPO, KTO, ORPO, EBFT, etc.)
    - `trainers/mixins/` - Trainer mixins (optimizer, scheduler, packing, distributed)
    - `builders/` - TrainerBuilder pattern (causal, RL)
  - `loaders/` - Model, tokenizer, adapter, processor loading; PatchManager
  - `integrations/` - Plugin system (BasePlugin, PluginManager) + 12 integration plugins
  - `monkeypatch/` - Runtime patches (attention, FSDP, models, trainer, loss, etc.)
  - `prompt_strategies/` - Dataset format handlers (chat_template, alpaca, DPO, etc.)
  - `utils/` - Config schemas, data prep, callbacks, collators, samplers, distributed
    - `schemas/` - Pydantic V2 configuration validation
    - `data/` - Dataset loading and preparation
    - `callbacks/` - Training callbacks
    - `collators/` - Data collators
  - `kernels/` - Custom Triton kernels (LoRA, DoRA, SwiGLU)
- `tests/` - Test suite (unit + e2e)
- `examples/` - Model-specific YAML configs
- `deepspeed_configs/` - DeepSpeed ZeRO configs

## WHY: Purpose

- Enable efficient fine-tuning of LLMs with minimal configuration
- Support SFT, LoRA/QLoRA, DPO, GRPO, KTO, ORPO, EBFT, reward modeling, pretraining
- Multi-GPU and multi-node training via FSDP, DeepSpeed, Ray
- Multimodal support (vision, audio) across many model families

## HOW: Core Commands

```bash
# Install
pip install axolotl[flash-attn,deepspeed]
# or
pip install -e '.[flash-attn,deepspeed]'

# Pre-commit hooks
pre-commit install
pre-commit run --all-files

# Train
axolotl train examples/llama-3/lora-1b.yml

# Preprocess datasets
axolotl preprocess config.yml

# Evaluate
axolotl evaluate config.yml

# Merge LoRA
axolotl merge-lora config.yml

# Run tests
pytest tests/ -m 'not slow'

# Run e2e tests (requires GPU)
pytest tests/e2e/solo/ -v
```

## Boundaries

### Constraints

- Designed for GPU training; many features require CUDA
- E2e tests require GPU hardware; skip gracefully when unavailable
- Heavy dependency on HuggingFace ecosystem internals (transformers, TRL, PEFT)
- Monkeypatch system is tightly coupled to upstream library versions

### Always Do

- Read relevant files before modifying code
- Run `pre-commit run --all-files` before committing
- Follow existing code patterns in the same module
- Add tests for new functionality
- Use `LOG = get_logger(__name__)` for logging (never `print` or `logging.getLogger`)
- Add new config keys to Pydantic schemas in `src/axolotl/utils/schemas/`
- Update `SUPPORTED_MULTIPACK_MODEL_TYPES` when adding new model architectures

### Ask First

- Modifying Pydantic config schemas in `src/axolotl/utils/schemas/`
- Adding new monkeypatches to `src/axolotl/monkeypatch/`
- Adding new dependencies to `setup.py` or `requirements.txt`
- Changing the plugin interface in `src/axolotl/integrations/base.py`
- Running GPU/distributed tests (check GPU first:
  `python -c "import torch; print('GPU:', torch.cuda.is_available())"`)

### Never Do

- Hardcode secrets, paths, or endpoints
- Skip pre-commit hooks
- Use wildcard imports (`from x import *`)
- Use `logging.getLogger(__name__)` instead of `get_logger(__name__)`
- Access config keys without considering DictDefault returns `None` for missing keys
- Add monkeypatches without version guards

## Progressive Disclosure: Detailed Guides

| Task                    | Reference                                                          |
| ----------------------- | ------------------------------------------------------------------ |
| Add Prompt Strategy     | `src/axolotl/prompt_strategies/chat_template.py`                   |
| Add Dataset Format      | `src/axolotl/utils/data/`, `src/axolotl/common/datasets.py`       |
| Add Integration/Plugin  | `src/axolotl/integrations/base.py`, `src/axolotl/integrations/liger/` |
| Add Trainer             | `src/axolotl/core/trainers/`, `src/axolotl/core/builders/rl.py`   |
| Add Model Support       | `src/axolotl/monkeypatch/multipack.py`, `src/axolotl/loaders/`    |
| Config Schema           | `src/axolotl/utils/schemas/config.py`                             |
| Monkeypatch Guide       | `src/axolotl/loaders/patch_manager.py`                            |
| Example Configs         | `examples/`                                                        |
| Contributing            | `.github/CONTRIBUTING.md`                                          |

## Git Workflow

- **Commits**: Imperative mood, capitalized verb, ~72 chars subject
  (e.g., `Add new feature`, `Fix bug in function`)
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations
- **PR template**: Description, motivation, testing, AI usage disclaimer
- **CI skip**: `[skip ci]` or `[skip-e2e]` in commit message/PR title

## Extended Configuration

See `.claude/agents/`, `.claude/skills/`, `.claude/commands/`, and `.claude/rules/` for
specialized instructions.

### Agents

| Agent                      | Purpose                                    | Activation Trigger                                                |
| -------------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| `planner`                  | Implementation planning                    | Before multi-file changes, new features, architectural decisions  |
| `simple-code-reviewer`     | Quick code quality checks                  | After code changes, before committing                             |
| `code-verifier`            | Formatting/linting/tests                   | After code changes, before committing                             |
| `model-loading-expert`     | ModelLoader, PatchManager, adapters        | Model loading code changes or questions                           |
| `training-expert`          | Trainers, builders, RL algorithms          | Trainer/builder code changes or questions                         |
| `data-processing-expert`   | Prompt strategies, datasets, collators     | Data pipeline code changes or questions                           |
| `config-schema-expert`     | Pydantic schemas, DictDefault, validation  | Config schema changes or validation questions                     |
| `plugin-integration-expert`| Plugin system and integrations             | Plugin/integration code changes or questions                      |
| `monkeypatch-expert`       | Monkeypatch system and model patches       | Monkeypatch code changes or compatibility questions               |

**Stage-by-Stage Agent Guidance**:

1. **Planning Stage** (Before coding): Use `planner` for architecture design
2. **Code Formatting & Linting** (After coding): Use `code-verifier` for automated checks
3. **Code Quality Check** (After formatting): Use `simple-code-reviewer` for logic review

### Skills (Guided Development Workflows)

- `/add-prompt-strategy` - Prompt strategy creation guide
- `/add-integration` - Plugin/integration creation guide
- `/add-trainer` - New trainer type guide
- `/add-model-support` - New model architecture support guide
- `/debug-training` - Training debugging guide
- `/add-unit-tests` - Test development guide

### Commands (User-invoked Actions)

- `/create-pr` - Rebase, squash commits, and create/update PR
- `/gen-commit-msg` - Generate commit messages from staged changes
- `/review-pr` - PR code review with dynamic agent allocation

### Rules (Code Quality Standards)

- `code-style.md` - Logging, naming, imports, type annotations
- `config-schema.md` - Pydantic schema patterns and DictDefault usage
- `monkeypatch.md` - Monkeypatch safety and version compatibility
- `testing.md` - Testing patterns, GPU handling, fixtures
- `plugin-system.md` - Plugin interface and lifecycle rules

## Code Intelligence & Navigation

When navigating and understanding code:

1. **ALWAYS prefer LSP tools over text search for code relationships**:
   - Use `goToDefinition` to jump to symbol definitions
   - Use `findReferences` to find all usages across the codebase
   - Use `goToImplementation` for interfaces/abstract methods
   - Use `workspaceSymbol` to search symbols across entire project

2. **Use Grep/Glob/Read ONLY for**:
   - Text/pattern searches in comments or strings
   - Searching configuration files (YAML, JSON, TOML)
   - Exploratory "fuzzy" searches when unsure what you're looking for

3. **Workflow**:
   - First: Use LSP to understand code structure and relationships
   - Second: Use text tools only when LSP cannot help
   - NEVER read entire large files to find references; use LSP instead
