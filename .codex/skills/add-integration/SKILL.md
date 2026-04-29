---
name: add-integration
description: Guide for creating a new plugin/integration for axolotl. Use when user wants to add a new integration package.
---

## Codex Reference Paths

- `.codex/rules/plugin-system.md`
- `.codex/rules/code-style.md`
- `.codex/rules/testing.md`

# Add Integration

Add a new plugin/integration to axolotl using the BasePlugin system.

## When to Use

- User asks "how do I create a plugin?"
- User wants to integrate a new library or optimization
- User mentions adding a new integration package

## Step-by-Step Guide

### Step 1: Create Integration Package

Create `src/axolotl/integrations/<name>/`:

```
src/axolotl/integrations/<name>/
  __init__.py     # Plugin class + registration
  args.py         # Pydantic config model (optional)
```

### Step 2: Implement Plugin Class

`src/axolotl/integrations/<name>/__init__.py`:

```python
"""<Name> integration for axolotl."""

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class MyPlugin(BasePlugin):
    """Plugin for <name> integration."""

    def get_input_args(self) -> str:
        """Return path to Pydantic config model."""
        return "axolotl.integrations.<name>.args.MyPluginConfig"

    def register(self, cfg):
        """Validate plugin requirements."""
        try:
            import my_dependency
        except ImportError as exc:
            raise ImportError(
                "Please install <name>: pip install <name>"
            ) from exc

    def pre_model_load(self, cfg):
        """Called before model is loaded."""
        if cfg.my_plugin_enabled:
            LOG.info("Applying <name> optimizations")
            # Apply pre-model patches/setup
            ...

    def post_model_build(self, cfg, model):
        """Called after model is built."""
        # Apply post-build modifications
        ...

    def add_callbacks_post_trainer(self, cfg, trainer):
        """Add training callbacks."""
        callbacks = []
        # Add custom callbacks
        return callbacks
```

### Step 3: Define Config Model (Optional)

`src/axolotl/integrations/<name>/args.py`:

```python
"""Configuration for <name> integration."""

from pydantic import BaseModel, field_validator


class MyPluginConfig(BaseModel):
    """Config fields for <name> plugin.

    These fields become available in the YAML config when the plugin is enabled.
    """

    my_plugin_enabled: bool = False
    my_plugin_setting: str = "default"

    @field_validator("my_plugin_setting")
    @classmethod
    def validate_setting(cls, value):
        if value not in ("option_a", "option_b", "default"):
            raise ValueError(f"Invalid setting: {value}")
        return value
```

### Step 4: Configure in YAML

```yaml
plugins:
  - axolotl.integrations.<name>

my_plugin_enabled: true
my_plugin_setting: option_a
```

### Step 5: Add Tests

`tests/integrations/test_<name>.py`:

```python
import pytest
from axolotl.integrations.<name> import MyPlugin


class TestMyPlugin:
    def test_register(self):
        """Test plugin registration."""
        plugin = MyPlugin()
        # Mock cfg
        ...

    def test_config_validation(self):
        """Test plugin config validation."""
        ...
```

## Available Lifecycle Hooks

| Hook                         | When Called                   | Use For                   |
| ---------------------------- | ---------------------------- | ------------------------- |
| `register(cfg)`              | Plugin registration          | Validate requirements     |
| `get_input_args()`           | Config merging               | Extend config schema      |
| `load_datasets(cfg)`         | Dataset loading              | Custom data loading       |
| `pre_model_load(cfg)`        | Before model instantiation   | Pre-model setup           |
| `post_model_build(cfg, model)` | After model construction   | Model modifications       |
| `pre_lora_load(cfg, model)`  | Before LoRA application      | Pre-adapter setup         |
| `post_lora_load(cfg, model)` | After LoRA application       | Post-adapter setup        |
| `post_model_load(cfg, model)`| After full model load        | Final model setup         |
| `get_trainer_cls(cfg, model)`| Trainer selection            | Custom trainer class      |
| `post_trainer_create(cfg, trainer)` | After trainer created  | Trainer modifications     |
| `create_optimizer(cfg, trainer)` | Optimizer creation        | Custom optimizer          |
| `create_lr_scheduler(...)` | Scheduler creation             | Custom LR scheduler       |
| `add_callbacks_pre_trainer(cfg, model)` | Before trainer    | Early callbacks           |
| `add_callbacks_post_trainer(cfg, trainer)` | After trainer  | Late callbacks            |
| `post_train(cfg, model)`    | After training               | Post-training cleanup     |

## Reference Implementations

| Plugin     | Package                    | Complexity |
| ---------- | -------------------------- | ---------- |
| Liger      | `integrations/liger/`      | Medium     |
| Spectrum   | `integrations/spectrum/`   | Simple     |
| SwanLab    | `integrations/swanlab/`    | Simple     |
| KD         | `integrations/kd/`         | Complex    |

## Key Requirements

1. **Extend BasePlugin**: All plugins must inherit from `BasePlugin`
2. **Graceful dependency handling**: Handle missing optional deps with clear error
3. **Self-contained**: Don't modify globals outside plugin scope
4. **Config model**: Use Pydantic for any new config fields
5. **Tests**: Add tests in `tests/integrations/`
