---
paths:
  - src/axolotl/monkeypatch/**
  - src/axolotl/loaders/patch_manager.py
---

# Monkeypatch Rules

## Safety Requirements

1. **Version guards**: Every patch MUST check the upstream library version:
   ```python
   from transformers import __version__ as transformers_version
   if version.parse(transformers_version) >= version.parse("5.3.0"):
       # Apply patch
   ```

2. **Document motivation**: Reference the upstream issue/PR that necessitates the patch:
   ```python
   # Workaround for https://github.com/huggingface/transformers/issues/XXXXX
   # Can be removed when transformers >= X.Y.Z
   ```

3. **Test cleanup**: Add corresponding cleanup to `tests/conftest.py:cleanup_monkeypatches`:
   ```python
   @pytest.fixture(autouse=True)
   def cleanup_monkeypatches():
       yield
       # Restore original methods
       importlib.reload(module_that_was_patched)
   ```

4. **Minimal replacement**: Replace only the necessary method, not entire classes

5. **No module-level patches**: All patches applied via PatchManager lifecycle

## PatchManager Lifecycle

Patches are applied in this order (DO NOT CHANGE without understanding all dependencies):

1. `apply_pre_config_load_patches()` - Before config loading (static)
2. `apply_pre_tokenizer_load_patches()` - Before tokenizer loading (static)
3. `apply_pre_model_load_patches()` - Before model instantiation
4. `apply_post_plugin_pre_model_load_patches()` - After plugin registration
5. `apply_post_model_build_patches()` - After model construction
6. `apply_post_model_load_patches()` - After full model load

## Multipack Model Types

When adding new model architecture support:

1. Add model type string to `SUPPORTED_MULTIPACK_MODEL_TYPES` in
   `src/axolotl/monkeypatch/multipack.py`
2. Verify sample packing works with the new model type
3. Add e2e test for the new model with packing enabled

## Common Pitfalls

- **Source code patching**: The unsloth integration uses `inspect.getsource()` +
  string replacement + `exec()`. This is extremely fragile.
- **Global state**: Some patches modify `transformers.modeling_utils.checkpoint` globally
- **Order-dependent**: Attention patches must be applied before model build
- **mypy disabled**: `mypy.ini` explicitly ignores `axolotl.monkeypatch.*`

## Upstream Dependency Coupling

Monkeypatches are tightly coupled to these libraries' internals:

| Library         | Patch Areas                                    |
| --------------- | ---------------------------------------------- |
| transformers    | Attention, model internals, trainer methods    |
| TRL             | Trainer patches, vLLM integration              |
| PEFT            | LoRA integration, quantization                 |
| accelerate      | FSDP2, distributed launch                      |
| bitsandbytes    | FSDP2+QLoRA quantization internals             |

**Every upstream version update requires reviewing all patches for compatibility.**
