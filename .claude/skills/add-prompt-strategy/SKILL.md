---
name: add-prompt-strategy
description: Guide for adding a new prompt strategy to axolotl. Use when user wants to create a new dataset format handler.
---

# Add Prompt Strategy

Add a new prompt strategy implementation to axolotl for handling a custom dataset format.

## When to Use

- User asks "how do I add a prompt strategy?"
- User wants to support a new dataset format
- User mentions implementing a custom tokenization strategy

## Prerequisites

- Understand the dataset format (fields, structure)
- Know which fields should be trained on vs masked
- Know if this is for SFT, DPO, or another training mode

## Step-by-Step Guide

### Step 1: Create Strategy File

Create `src/axolotl/prompt_strategies/<name>.py`:

```python
"""Prompt strategy for <format description>."""

from typing import Any, Generator

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class MyPrompter:
    """Prompter for <format> dataset format."""

    def build_prompt(self, instruction: str, input_text: str = "", output: str = "") -> str:
        """Build prompt string from dataset fields."""
        # Construct the prompt from dataset fields
        ...
        return prompt


class MyStrategy(PromptTokenizingStrategy):
    """Tokenization strategy for <format> datasets."""

    def tokenize_prompt(self, prompt: dict[str, Any]) -> Generator[dict, None, None]:
        """Tokenize a single dataset row.

        Args:
            prompt: Dictionary from the dataset row.

        Yields:
            Tokenized example with input_ids, attention_mask, labels.
        """
        # 1. Extract fields from prompt dict
        # 2. Build prompt text
        # 3. Tokenize with self.tokenizer
        # 4. Construct labels (mask non-training tokens with -100)
        # 5. Yield result dict
        ...


def load(tokenizer, cfg, ds_cfg=None, **kwargs):
    """Factory function called by axolotl to instantiate this strategy.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        cfg: DictDefault config.
        ds_cfg: Dataset-specific config (optional).

    Returns:
        MyStrategy instance.
    """
    return MyStrategy(MyPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len)
```

### Step 2: Configure in YAML

Reference the strategy in your training config:

```yaml
datasets:
  - path: my_dataset
    type: my_strategy_name  # matches filename without .py
```

### Step 3: Add Tests

Create `tests/prompt_strategies/test_<name>.py`:

```python
import pytest
from axolotl.prompt_strategies.<name> import MyStrategy, MyPrompter, load


class TestMyStrategy:
    def setup_method(self):
        # Set up mock tokenizer and config
        ...

    def test_basic_tokenization(self):
        """Test basic prompt tokenization."""
        ...

    def test_label_masking(self):
        """Test that non-training tokens are masked with -100."""
        ...

    def test_empty_input(self):
        """Test handling of empty/missing fields."""
        ...
```

## Reference Implementations

| Strategy          | File                                  | Description                    |
| ----------------- | ------------------------------------- | ------------------------------ |
| chat_template     | `prompt_strategies/chat_template.py`  | HF chat templates (default)    |
| completion        | `prompt_strategies/completion.py`     | Raw text completion            |
| alpaca_chat       | `prompt_strategies/alpaca_chat.py`    | Alpaca chat format             |
| input_output      | `prompt_strategies/input_output.py`   | Simple input/output pairs      |
| user_defined      | `prompt_strategies/user_defined.py`   | Custom user strategies         |

## Key Requirements

1. **`load()` factory**: Module MUST expose a `load(tokenizer, cfg, ds_cfg)` function
2. **Label masking**: Mask non-training tokens with `-100` in labels
3. **`train_on_inputs`**: Respect this config flag for input masking
4. **Sequence length**: Truncate to `cfg.sequence_len`
5. **Logger**: Use `LOG = get_logger(__name__)`

## Common Mistakes

- Forgetting the `load()` factory function
- Not masking labels correctly (training on system/user tokens)
- Not handling empty/missing dataset fields
- Not respecting `train_on_inputs` config
