# Code Style Rules

Rules beyond pre-commit (ruff format/lint, mypy, bandit).

## Logging

- Use `from axolotl.utils.logging import get_logger; LOG = get_logger(__name__)`
  - Returns a `MultiProcessAdapter` that filters to main process by default
  - Good: `LOG = get_logger(__name__)`
  - Avoid: `LOG = logging.getLogger(__name__)` (bypasses distributed-aware adapter)
- The module-level logger is always named `LOG` (not `logger` or `log`)
- Use f-string formatting: `LOG.info(f"Loaded {count} samples")`
- Log levels:
  - DEBUG: Detailed tracing (avoid in hot paths)
  - INFO: Milestones (training start, checkpoint saved, model loaded)
  - WARNING: Recoverable issues
  - ERROR: Failures requiring attention
- Controlled by `AXOLOTL_LOG_LEVEL` and `LOG_LEVEL` environment variables

## Naming Conventions

| Type              | Pattern          | Example                                  |
| ----------------- | ---------------- | ---------------------------------------- |
| Trainer class     | `AxolotlXxxTrainer` | `AxolotlDPOTrainer`, `AxolotlGRPOTrainer` |
| Builder class     | `HFXxxBuilder`   | `HFCausalTrainerBuilder`, `HFRLTrainerBuilder` |
| Config schema     | `XxxConfig`      | `LoraConfig`, `FSDPConfig`               |
| Prompt strategy   | `XxxPrompter`    | `ChatTemplatePrompter`, `AlpacaPrompter` |
| Plugin class      | `XxxPlugin`      | `LigerPlugin`, `SpectrumPlugin`          |
| Module-level logger| `LOG`           | `LOG = get_logger(__name__)`             |
| Constants         | `UPPER_SNAKE`    | `SUPPORTED_MULTIPACK_MODEL_TYPES`        |
| Private methods   | `_snake_case`    | `_build_model`, `_set_attention_config`  |

## Type Annotations

- Prefer modern syntax: `X | None` over `Optional[X]`, `X | Y` over `Union[X, Y]`
- Use `from __future__ import annotations` when needed for forward references
- New functions should have full type annotations
- Pydantic models have strong typing; respect this in schema code

## Import Style

- Group: stdlib, third-party, local `axolotl.*` (ruff handles order)
- Avoid `from x import *`
- Prefer deferred/lazy imports inside functions for heavy optional deps
- Use `TYPE_CHECKING` guard for imports needed only by type checkers

## Design Patterns

- **Builder pattern**: Use `TrainerBuilderBase` for trainer construction
- **Plugin pattern**: New features should go through `BasePlugin` interface when possible
- **Mixin pattern**: Trainer mixins for reusable behavior (8 mixins available)
- **Strategy pattern**: Prompt strategies loaded dynamically via `importlib`
- Keep inheritance shallow; prefer explicit delegation

## Performance Patterns

- **Avoid GPU-CPU sync**: `.item()`, `.tolist()`, `print(tensor)` cause sync
- **Prefer batch operations**: Avoid Python loops over tensor elements
- **Lazy imports**: Use for heavy optional deps (reduces startup time)

## Error Handling

- Use `raise ValueError(f"descriptive message: {value}")` for config validation
- Optional import pattern: `with suppress(ImportError)` or `try/except ImportError`
- Don't swallow exceptions silently; at minimum log the error
- Signal handling: SIGINT saves model gracefully (rank 0 only)
