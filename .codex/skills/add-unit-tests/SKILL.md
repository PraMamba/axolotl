---
name: add-unit-tests
description: Guide for adding unit tests to axolotl. Use when user wants to add tests for new or existing functionality.
---

## Codex Reference Paths

- `.codex/rules/testing.md`

# Add Unit Tests

Guide for writing tests in axolotl following project conventions.

## When to Use

- Adding tests for new functionality
- Improving test coverage for existing code
- User asks "how should I test this?"

## Test Conventions

### Framework

- **pytest** (not unittest for new tests)
- Class-based tests: `class TestMyFeature:`
- Markers: `@pytest.mark.slow`, `@pytest.mark.asyncio`

### File Placement

| Test Type         | Location                         | GPU Required |
| ----------------- | -------------------------------- | ------------ |
| Unit tests        | `tests/test_<module>.py`         | No           |
| Prompt strategies | `tests/prompt_strategies/`       | No           |
| CLI tests         | `tests/cli/`                     | No           |
| Schema tests      | `tests/utils/schemas/`           | No           |
| E2e solo tests    | `tests/e2e/solo/`                | Yes          |
| E2e patched       | `tests/e2e/patched/`             | Yes          |
| E2e multi-GPU     | `tests/e2e/multigpu/`            | Yes, multi   |
| Integration tests | `tests/e2e/integrations/`        | Yes          |

### Test Template

```python
"""Tests for <module description>."""

import pytest
from unittest.mock import MagicMock, patch

from axolotl.utils.dict import DictDefault


class TestMyFeature:
    """Tests for MyFeature."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cfg = DictDefault({
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "sequence_len": 512,
        })

    def test_basic_functionality(self):
        """Test that feature does X when Y."""
        # Arrange
        input_data = ...

        # Act
        result = my_function(input_data)

        # Assert
        assert result == expected

    @pytest.mark.parametrize("input_val,expected", [
        ("a", 1),
        ("b", 2),
        (None, 0),
    ])
    def test_parameterized(self, input_val, expected):
        """Test with various inputs."""
        result = my_function(input_val)
        assert result == expected

    def test_error_handling(self):
        """Test that ValueError raised for invalid input."""
        with pytest.raises(ValueError, match="must be positive"):
            my_function(-1)
```

### GPU Test Template

```python
import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()


class TestGPUFeature:

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_operation(self):
        """Test GPU-dependent functionality."""
        ...
```

### E2e Test Template

```python
"""End-to-end test for <feature>."""

import pytest
import torch
from axolotl.utils.dict import DictDefault
from axolotl.utils.config import validate_config, normalize_config
from axolotl.common.datasets import load_datasets
from axolotl.train import train


class TestE2EFeature:

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_full_training_pipeline(self, tmp_path):
        cfg = DictDefault({
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "output_dir": str(tmp_path),
            "micro_batch_size": 1,
            "num_epochs": 1,
            "max_steps": 5,
            ...
        })
        validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)
        train(cfg=cfg, dataset_meta=dataset_meta)
        # Assert output files exist
        assert (tmp_path / "pytorch_model.bin").exists() or \
               (tmp_path / "model.safetensors").exists()
```

## Key Fixtures (from conftest.py)

| Fixture                  | Scope    | Purpose                               |
| ------------------------ | -------- | ------------------------------------- |
| `reset_plugin_manager`   | function | Cleans PluginManager singleton        |
| `torch_manual_seed`      | function | Seeds RNG to 42                       |
| `cleanup_monkeypatches`  | function | Restores patched methods              |
| `disable_telemetry`      | function | Sets AXOLOTL_DO_NOT_TRACK=1           |
| `min_base_cfg`           | function | Minimal DictDefault config            |

All are `autouse=True` - they run automatically.

## Mocking Patterns

```python
# Mock config values
from unittest.mock import patch

@patch("axolotl.utils.config.is_torch_bf16_gpu_available", return_value=True)
def test_with_bf16(self, mock_bf16):
    ...

# Mock tokenizer
tokenizer = MagicMock()
tokenizer.encode.return_value = [1, 2, 3]
tokenizer.decode.return_value = "hello"

# Mock environment
def test_with_env(self, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    ...
```

## Running Tests

```bash
# All unit tests (default, excludes slow)
pytest tests/ -v

# Specific test file
pytest tests/test_normalize_config.py -v

# Specific test class
pytest tests/test_normalize_config.py::TestNormalizeConfig -v

# With coverage
pytest tests/ --cov=axolotl --cov-report=html

# E2e tests (requires GPU)
pytest tests/e2e/solo/ -v
```
