---
name: add-model-support
description: Guide for adding new model architecture support to axolotl. Use when user wants to enable fine-tuning for a new model family.
---

## Codex Reference Paths

- `.codex/rules/monkeypatch.md`
- `.codex/rules/testing.md`

# Add Model Support

Add support for a new model architecture in axolotl's fine-tuning pipeline.

## When to Use

- User wants to fine-tune a new model family
- User needs sample packing for a new architecture
- User wants to add multimodal model support

## Step-by-Step Guide

### Step 1: Check Existing Support

First verify the model isn't already supported:

```bash
# Check multipack support
grep "<model_type>" src/axolotl/monkeypatch/multipack.py

# Check multimodal support
grep "<model_type>" src/axolotl/loaders/constants.py

# Check for model-specific patches
ls src/axolotl/monkeypatch/models/
```

### Step 2: Add to Multipack Model Types

If the model supports sample packing, add its model type string to
`SUPPORTED_MULTIPACK_MODEL_TYPES` in `src/axolotl/monkeypatch/multipack.py`:

```python
SUPPORTED_MULTIPACK_MODEL_TYPES = [
    # ... existing types
    "my_new_model",
]
```

The model type string comes from the HuggingFace `config.json`'s `model_type` field.

### Step 3: Add Model-Specific Patches (If Needed)

If the model requires patches for Flash Attention, custom attention, or other
fixes, create `src/axolotl/monkeypatch/models/<name>.py`:

```python
"""Patches for <ModelName> model architecture."""

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_<name>_model():
    """Apply patches for <ModelName>.

    Workaround for: <upstream issue link>
    Can be removed when: transformers >= X.Y.Z
    """
    # Apply minimal patches
    ...
```

Then register in PatchManager (`src/axolotl/loaders/patch_manager.py`):

```python
# In apply_pre_model_load_patches() or appropriate lifecycle point
if cfg.model_config_type == "my_new_model":
    from axolotl.monkeypatch.models.<name> import patch_<name>_model
    patch_<name>_model()
```

### Step 4: Add Multimodal Support (If Applicable)

For vision/audio models, update `src/axolotl/loaders/constants.py`:

```python
MULTIMODAL_AUTO_MODEL_MAPPING = {
    # ... existing mappings
    "my_vision_model": "AutoModelForImageTextToText",
}
```

And add a processing strategy in `src/axolotl/processing_strategies.py` if the
model requires custom multimodal data handling.

### Step 5: Add Example Config

Create `examples/<model_name>/`:

```yaml
# examples/<model_name>/lora.yml
base_model: org/model-name
model_type: AutoModelForCausalLM

load_in_8bit: false
load_in_4bit: true

adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_linear: true

datasets:
  - path: dataset/name
    type: chat_template

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 1
learning_rate: 2e-4

output_dir: ./outputs/model-name-lora
```

### Step 6: Add Tests

Create an e2e test in `tests/e2e/solo/`:

```python
import pytest
from axolotl.utils.dict import DictDefault
from axolotl.utils.config import validate_config, normalize_config

class TestMyModelSupport:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_lora_training(self):
        cfg = DictDefault({
            "base_model": "org/model-name",
            "adapter": "lora",
            ...
        })
        validate_config(cfg)
        normalize_config(cfg)
        # Run training
        ...
```

## Checklist

- [ ] Model type added to `SUPPORTED_MULTIPACK_MODEL_TYPES` (if packing supported)
- [ ] Model-specific patches created and registered (if needed)
- [ ] Version guards on patches (if needed)
- [ ] conftest.py cleanup updated (if patches added)
- [ ] Multimodal mapping updated (if multimodal model)
- [ ] Processing strategy created (if custom multimodal handling)
- [ ] Example config created in `examples/`
- [ ] E2e test added (if GPU available)

## Key Files

| File                                        | Purpose                           |
| ------------------------------------------- | --------------------------------- |
| `src/axolotl/monkeypatch/multipack.py`      | Multipack model type list         |
| `src/axolotl/loaders/constants.py`          | Multimodal model mapping          |
| `src/axolotl/loaders/patch_manager.py`      | Patch registration                |
| `src/axolotl/monkeypatch/models/`           | Model-specific patches            |
| `src/axolotl/processing_strategies.py`       | Multimodal processing strategies  |
