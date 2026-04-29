# AGENTS.md - Axolotl

## Codex Operating Contract

This file is the project-level instruction surface for Codex agents working in this repository.
Follow higher-priority system, developer, and user instructions first; then apply this document to
all files under `/root/axolotl`. Use `.codex/config.toml` for project-scoped Codex agent/skill
registration and `.codex/` for reusable Codex-native guidance.

Do not treat legacy Claude-oriented files as the active agent interface. The active Codex interface is:

- `.codex/agents/*.toml` for custom agents
- `.codex/skills/*/SKILL.md` for workflow skills
- `.codex/rules/*.md` for reusable coding rules
- `.codex/data/*.md` for review templates/reference data

## WHAT: Project Overview

Axolotl is an open-source LLM fine-tuning framework that streamlines post-training for
large language models via a single YAML configuration file.

**Tech Stack**: Python 3.10+ | PyTorch 2.9+ | HuggingFace Transformers 5.x | TRL | PEFT
| Accelerate | DeepSpeed | FSDP | vLLM

**Core Directories**:

- `src/axolotl/` - Main source package
  - `cli/` - Click CLI entry points and config loading
  - `core/` - Trainers, builders, training args, attention, chat formatting
    - `trainers/` - AxolotlTrainer + RL trainers (DPO, GRPO, KTO, ORPO, EBFT, etc.)
    - `trainers/mixins/` - Trainer mixins (optimizer, scheduler, packing, distributed)
    - `builders/` - TrainerBuilder pattern (causal, RL)
  - `loaders/` - Model, tokenizer, adapter, processor loading; PatchManager
  - `integrations/` - Plugin system (BasePlugin, PluginManager) + integration plugins
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
- `.codex/` - Project-scoped Codex agents, skills, rules, and review data

## WHY: Purpose

- Enable efficient fine-tuning of LLMs with minimal configuration
- Support SFT, LoRA/QLoRA, DPO, GRPO, KTO, ORPO, EBFT, reward modeling, pretraining
- Multi-GPU and multi-node training via FSDP, DeepSpeed, Ray
- Multimodal support (vision, audio) across many model families
- Provide Codex-native project guidance through `.codex/config.toml`, custom agents, and skills

## HOW: Core Commands

```bash
# Install
pip install axolotl[flash-attn,deepspeed]
# or
pip install -e '.[flash-attn,deepspeed]'

# Pre-commit checks
# CI tool versions are pinned in .pre-commit-config.yaml — never run system ruff/mypy
pre-commit install
pre-commit run --all-files

# Auto-fix with the pinned ruff (<rev> = the ruff-pre-commit rev in .pre-commit-config.yaml)
uvx ruff@<rev> check --fix && uvx ruff@<rev> format

# Train
axolotl train examples/llama-3/lora-1b.yml

# Preprocess datasets
axolotl preprocess config.yml
axolotl preprocess config.yml --debug

# Evaluate
axolotl evaluate config.yml

# Merge LoRA
axolotl merge-lora config.yml

# Interactive inference
axolotl inference config.yml

# Serve with vLLM for online RL workflows
axolotl vllm-serve config.yml

# Run tests
pytest tests/ -m 'not slow'

# Run e2e tests (requires GPU)
pytest tests/e2e/solo/ -v

# Agent-optimized docs bundled with Axolotl
axolotl agent-docs
axolotl agent-docs grpo
axolotl config-schema
```

## Boundaries

### Constraints

- Designed for GPU training; many features require CUDA
- E2e tests require GPU hardware; skip gracefully when unavailable
- Heavy dependency on HuggingFace ecosystem internals (transformers, TRL, PEFT)
- Monkeypatch system is tightly coupled to upstream library versions
- Codex project config in `.codex/config.toml` loads only when the project is trusted locally
- `.codex/` intentionally does not include hook-based automation; invoke skills/review workflows manually

### Always Do

- Read relevant files before modifying code
- Follow existing patterns in the same module before adding new abstractions
- Add tests for new functionality
- Run targeted tests for the changed area; run `pre-commit run --all-files` before committing
- Use `LOG = get_logger(__name__)` for logging (never `print` or `logging.getLogger`)
- Add new config keys to Pydantic schemas in `src/axolotl/utils/schemas/`
- Consider `DictDefault` behavior: missing config keys return `None` silently
- Update `SUPPORTED_MULTIPACK_MODEL_TYPES` when adding new model architectures that need multipack support
- Keep `.codex` agent/skill/rule references current when project workflows change

### Ask First

- Adding new dependencies to packaging or requirements files
- Running GPU/distributed tests; first check GPU availability:

  ```bash
  python -c "import torch; print('GPU:', torch.cuda.is_available())"
  ```

- Changing public plugin interfaces in `src/axolotl/integrations/base.py`
- Adding new monkeypatches or widening existing monkeypatch behavior
- Making broad schema migrations in `src/axolotl/utils/schemas/`
- Rewriting training orchestration, distributed behavior, or checkpoint semantics beyond the requested scope

### Never Do

- Hardcode secrets, local-only paths, credentials, or service endpoints
- Skip required verification and claim completion without fresh evidence
- Use wildcard imports (`from x import *`)
- Use `logging.getLogger(__name__)` instead of `get_logger(__name__)`
- Access config keys without considering `DictDefault` silent `None` behavior
- Add monkeypatches without version guards and cleanup coverage
- Delete tests to make a change pass
- Add hook configuration under `.codex`; hooks are intentionally out of scope for this project setup

## Progressive Disclosure: Detailed Guides

| Work Area               | Codex / Repository Reference                                                          |
| ----------------------- | ------------------------------------------------------------------ |
| Add Prompt Strategy     | `.codex/skills/add-prompt-strategy/SKILL.md`, `src/axolotl/prompt_strategies/chat_template.py` |
| Add Dataset Format      | `src/axolotl/utils/data/`, `src/axolotl/common/datasets.py`       |
| Add Integration/Plugin  | `.codex/skills/add-integration/SKILL.md`, `.codex/rules/plugin-system.md`, `src/axolotl/integrations/base.py` |
| Add Trainer             | `.codex/skills/add-trainer/SKILL.md`, `src/axolotl/core/trainers/`, `src/axolotl/core/builders/rl.py` |
| Add Model Support       | `.codex/skills/add-model-support/SKILL.md`, `src/axolotl/monkeypatch/multipack.py`, `src/axolotl/loaders/` |
| Add Unit Tests          | `.codex/skills/add-unit-tests/SKILL.md`, `.codex/rules/testing.md` |
| Debug Training          | `.codex/skills/debug-training/SKILL.md`                           |
| Config Schema           | `.codex/rules/config-schema.md`, `src/axolotl/utils/schemas/config.py` |
| Monkeypatch Guide       | `.codex/rules/monkeypatch.md`, `src/axolotl/loaders/patch_manager.py` |
| Code Style              | `.codex/rules/code-style.md`                                      |
| Example Configs         | `examples/`                                                        |
| Feature Support Matrix  | `docs/support-matrix.qmd`                                          |
| Contributing            | `.github/CONTRIBUTING.md`                                          |

## Git Workflow

- **Commits**: Imperative mood, capitalized verb, ~72 chars subject
  (e.g., `Add new feature`, `Fix bug in function`)
- **Lore protocol**: When creating commits in this environment, prefer structured decision-record
  trailers (`Constraint:`, `Rejected:`, `Confidence:`, `Tested:`, `Not-tested:`) when they add value
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations
- **PR template**: Description, motivation, testing, AI usage disclaimer
- **CI skip**: `[skip ci]` or `[skip-e2e]` in commit message/PR title
- **Codex workflows**: Use `.codex/skills/create-pr`, `.codex/skills/gen-commit-msg`, and
  `.codex/skills/review-pr` for PR/commit/review guidance

## Extended Codex Configuration

Codex project configuration lives under `.codex/`. The project config is `.codex/config.toml` and
registers custom agents plus skills. This directory is intentionally Codex-native and does not rely
on Codex hook wiring. When updating these assets, keep TOML parseable and keep every `SKILL.md`
frontmatter block at the top of the file.

### Agents

| Agent                         | Purpose                                      | Activation Trigger                                               |
| ----------------------------- | -------------------------------------------- | ---------------------------------------------------------------- |
| `implementation_planner`      | Implementation planning                      | Before multi-file changes, new features, architectural decisions |
| `simple_code_reviewer`        | Quick code quality checks                    | After code changes, before committing                            |
| `code_verifier`               | Formatting, linting, tests, verification     | After code changes, before committing                            |
| `model_loading_expert`        | ModelLoader, PatchManager, adapters          | Model loading code changes or questions                          |
| `training_expert`             | Trainers, builders, RL algorithms            | Trainer/builder code changes or questions                        |
| `data_processing_expert`      | Prompt strategies, datasets, collators       | Data pipeline code changes or questions                          |
| `config_schema_expert`        | Pydantic schemas, DictDefault, validation    | Config schema changes or validation questions                    |
| `plugin_integration_expert`   | Plugin system and integrations               | Plugin/integration code changes or questions                     |
| `monkeypatch_expert`          | Monkeypatch system and model patches         | Monkeypatch code changes or compatibility questions              |

**Stage-by-Stage Agent Guidance**:

1. **Planning Stage** (before coding): use `implementation_planner` for architecture and scope
2. **Domain Stage** (during design or review): use the relevant domain expert for risky areas
3. **Verification Stage** (after coding): use `code_verifier` for formatting/linting/tests
4. **Quality Stage** (after verification): use `simple_code_reviewer` for logic and maintainability review

### Skills (Guided Development Workflows)

- `add-prompt-strategy` - Prompt strategy creation guide
- `add-integration` - Plugin/integration creation guide
- `add-trainer` - New trainer type guide
- `add-model-support` - New model architecture support guide
- `debug-training` - Training debugging guide
- `add-unit-tests` - Test development guide

### Command-Derived Codex Skills (User-Invoked Actions)

- `create-pr` - Rebase, squash commits, and create/update PR guidance
- `gen-commit-msg` - Generate commit messages from staged changes
- `review-pr` - PR code review with dynamic Codex agent/reasoning allocation

Invoke these as Codex skills/workflows by name in user prompts or through the configured Codex skill
surface; do not describe them as Claude slash commands.

### Rules (Code Quality Standards)

- `.codex/rules/code-style.md` - Logging, naming, imports, type annotations
- `.codex/rules/config-schema.md` - Pydantic schema patterns and DictDefault usage
- `.codex/rules/monkeypatch.md` - Monkeypatch safety and version compatibility
- `.codex/rules/testing.md` - Testing patterns, GPU handling, fixtures
- `.codex/rules/plugin-system.md` - Plugin interface and lifecycle rules

### Review Data and Templates

- `.codex/data/review-pr-change-types.md` - PR risk/change-type classification
- `.codex/data/review-pr-templates.md` - Review work templates for PR analysis

## Code Intelligence & Navigation

When navigating and understanding code:

1. **Prefer LSP/code-intelligence tools over text search for code relationships**:
   - Use go-to-definition to jump to symbol definitions
   - Use find-references to find usages across the codebase
   - Use implementation/symbol search for interfaces, abstract methods, and workspace-wide symbols

2. **Use grep/glob/read for**:
   - Text or pattern searches in comments and strings
   - Searching configuration files (YAML, JSON, TOML)
   - Exploratory fuzzy searches when unsure what symbol to target
   - Inspecting `.codex` Markdown/TOML guidance files

3. **Workflow**:
   - First: use code-intelligence tools to understand structure and relationships
   - Second: use text tools when code-intelligence cannot answer the question
   - Avoid reading entire large files just to find references; use symbol/reference tools instead

## Verification Before Completion

Codex agents must verify before claiming completion. Before finishing:

- Identify what proves the change works
- Run the relevant tests, lint, type checks, or static checks
- Read the output and report concrete evidence
- For `.codex` changes, at minimum verify:

  ```bash
  python - <<'PY'
  from pathlib import Path
  import tomllib
  for path in [Path('.codex/config.toml'), *Path('.codex/agents').glob('*.toml')]:
      tomllib.loads(path.read_text())
      print('OK', path)
  PY
  ```

- If touching skills, ensure each `SKILL.md` begins with YAML frontmatter (`---`)
- If touching hooks or hook-like automation, stop and re-check scope: this project setup intentionally avoids Codex hook configuration
