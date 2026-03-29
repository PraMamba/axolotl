---
paths:
  - tests/**
  - '*_test.py'
  - test_*.py
---

# Testing Rules

## Test Framework

- Use **pytest** (not unittest for new tests)
- Markers: `@pytest.mark.slow` (skipped by default), `@pytest.mark.asyncio`
- Plugins: pytest-cov, pytest-retry, pytest-xdist
- Config: `addopts = "-m 'not slow'"` in `pyproject.toml`

## Test Organization

| Category             | Location                    | GPU Required   |
| -------------------- | --------------------------- | -------------- |
| Unit tests           | `tests/test_*.py`           | No             |
| Prompt strategy tests| `tests/prompt_strategies/`  | No             |
| CLI tests            | `tests/cli/`                | No             |
| Schema tests         | `tests/utils/schemas/`      | No             |
| E2e solo tests       | `tests/e2e/solo/`           | Yes            |
| E2e patched tests    | `tests/e2e/patched/`        | Yes            |
| E2e multi-GPU        | `tests/e2e/multigpu/`       | Yes, multi-GPU |
| Integration tests    | `tests/e2e/integrations/`   | Yes            |

## Test Style

Use pytest class-based tests:

```python
class TestMyFeature:
    """Tests for my feature."""

    def setup_method(self):
        """Set up test fixtures."""
        ...

    def test_basic_functionality(self):
        """Test that feature does X when Y."""
        # Arrange
        ...
        # Act
        ...
        # Assert
        ...

    @pytest.mark.parametrize("input,expected", [
        ("a", 1),
        ("b", 2),
    ])
    def test_parameterized(self, input, expected):
        ...
```

## GPU Test Handling

- **Always skip gracefully** when GPU unavailable:
  ```python
  @pytest.mark.skipif(
      not torch.cuda.is_available(),
      reason="CUDA not available"
  )
  def test_gpu_feature():
      ...
  ```
- Version-gated decorators from `tests/e2e/utils.py`:
  `require_torch_2_5_1`, `require_torch_2_6_0`, etc.
- GPU capability gates: `requires_sm_ge_100`, `require_hopper`
- CI skip: `[skip-e2e]` in commit message skips e2e tests

## Fixtures

Key fixtures from `tests/conftest.py`:

- `reset_plugin_manager` (autouse) - Cleans up PluginManager singleton between tests
- `torch_manual_seed` (autouse) - Sets seed to 42
- `cleanup_monkeypatches` (autouse) - Restores patched methods after each test
- `disable_telemetry` (autouse) - Sets `AXOLOTL_DO_NOT_TRACK=1`
- `min_base_cfg` - Minimal DictDefault config for testing

## Mocking Patterns

- `unittest.mock.patch` for config/environment overrides
- `unittest.mock.MagicMock` for mock objects (tokenizers, models)
- `pytest.monkeypatch` for environment variables
- `@enable_hf_offline` for HF Hub offline mode

## E2e Test Pattern

```python
def test_e2e_training(self):
    cfg = DictDefault({...})
    validate_config(cfg)
    normalize_config(cfg)
    dataset_meta = load_datasets(cfg=cfg)
    train(cfg=cfg, dataset_meta=dataset_meta)
    # Assert on output/model/metrics
```

## Assertions

- Use `torch.testing.assert_close()` for tensor comparison
- Specify `rtol`/`atol` explicitly for numerical tests
- Use `pytest.raises(ValueError, match="expected message")` for error testing
- For model output validation, use `tbparse` to read TensorBoard logs
