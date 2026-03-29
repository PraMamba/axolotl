---
name: add-trainer
description: Guide for adding a new trainer type to axolotl. Use when user wants to implement a new training algorithm.
---

# Add Trainer

Add a new trainer implementation to axolotl for a new training algorithm.

## When to Use

- User wants to add a new RL algorithm (e.g., new preference optimization method)
- User wants a custom training loop
- User mentions implementing a new trainer variant

## Step-by-Step Guide

### Step 1: Create Trainer Class

Create `src/axolotl/core/trainers/<name>/trainer.py`:

```python
"""<Name> trainer implementation."""

from axolotl.core.trainers.mixins import (
    ActivationOffloadingMixin,
    CheckpointSaveMixin,
    OptimizerMixin,
    PackingMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.utils.logging import get_logger
from trl import <BaseTrainer>  # or transformers.Trainer

LOG = get_logger(__name__)


class Axolotl<Name>Trainer(
    CheckpointSaveMixin,
    PackingMixin,
    SchedulerMixin,
    OptimizerMixin,
    RngLoaderMixin,
    ActivationOffloadingMixin,
    <BaseTrainer>,
):
    """Axolotl trainer for <algorithm>.

    Extends TRL/HF <BaseTrainer> with axolotl mixins for:
    - Custom optimizer/scheduler support via plugins
    - Sample packing support
    - Checkpoint save customization
    - RNG state persistence
    """

    tag_names = ["axolotl", "<name>"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization
        ...
```

### Step 2: Register in `__init__.py`

Add to `src/axolotl/core/trainers/__init__.py`:

```python
from axolotl.core.trainers.<name>.trainer import Axolotl<Name>Trainer

__all__ = [
    # ... existing trainers
    "Axolotl<Name>Trainer",
]
```

### Step 3: Add to Builder

Update `src/axolotl/core/builders/rl.py` (for RL methods):

```python
# In HFRLTrainerBuilder.get_trainer_cls()
if cfg.rl == "<name>":
    return Axolotl<Name>Trainer
```

Or add config enum in `src/axolotl/utils/schemas/enums.py`:

```python
class RLType(str, Enum):
    # ... existing types
    MY_METHOD = "<name>"
```

### Step 4: Add Training Arguments (if needed)

In `src/axolotl/core/training_args.py`:

```python
@dataclass
class Axolotl<Name>Config(AxolotlTrainingArguments):
    """Training arguments for <name> algorithm."""
    my_param: float = 0.1
```

### Step 5: Add Prompt Strategy (if needed)

If the new trainer requires a different data format, create a prompt strategy
(see `/add-prompt-strategy` skill).

### Step 6: Add Config Schema Fields

Add new config fields to `src/axolotl/utils/schemas/trl.py` or create new schema:

```python
class MyMethodConfig(BaseModel):
    my_param: float = 0.1
```

### Step 7: Add Tests

Create `tests/core/trainers/test_<name>.py`:

```python
import pytest
from axolotl.core.trainers.<name>.trainer import Axolotl<Name>Trainer


class TestMyTrainer:
    def test_trainer_init(self):
        ...

    def test_loss_computation(self):
        ...
```

## Available Mixins

| Mixin                      | Purpose                                    |
| -------------------------- | ------------------------------------------ |
| `PackingMixin`             | Sample packing / multipack batch sampler   |
| `SchedulerMixin`           | Custom LR scheduler creation               |
| `OptimizerMixin`           | Custom optimizer via plugin delegation     |
| `OptimizerInitMixin`       | Optimizer initialization for GRPO trainers |
| `RngLoaderMixin`           | RNG state persistence                      |
| `CheckpointSaveMixin`      | Model save customization                   |
| `LayerOffloadingMixin`     | Layer offloading support                   |
| `ActivationOffloadingMixin`| Activation checkpointing                   |
| `DistributedParallelMixin` | FSDP2/TP device mesh management            |

## Reference Implementations

| Trainer                 | File                                    | Complexity |
| ----------------------- | --------------------------------------- | ---------- |
| AxolotlDPOTrainer       | `core/trainers/dpo/trainer.py`          | Medium     |
| AxolotlGRPOTrainer      | `core/trainers/grpo/trainer.py`         | High       |
| AxolotlKTOTrainer       | `core/trainers/trl.py`                  | Simple     |
| AxolotlORPOTrainer      | `core/trainers/trl.py`                  | Simple     |
| AxolotlEBFTTrainer      | `core/trainers/ebft/trainer.py`         | High       |

## Key Requirements

1. **Extend HF/TRL Trainer**: Base on `transformers.Trainer` or TRL trainer class
2. **Use mixins**: Include relevant mixins for axolotl features
3. **Prefix with Axolotl**: Class name must be `Axolotl<Name>Trainer`
4. **Register in builder**: Add to `HFRLTrainerBuilder` or `HFCausalTrainerBuilder`
5. **Config enum**: Add to `RLType` if it's an RL method
6. **Logger**: Use `LOG = get_logger(__name__)`
