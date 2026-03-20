---
status: in-progress
created: '2026-01-07'
tags:
  - channel-loss
  - plugin-architecture
  - dataset-injection
  - bugfix
  - p0
priority: high
created_at: '2026-01-07T03:50:50.679Z'
updated_at: '2026-01-07T03:52:54.761Z'
transitions:
  - status: in-progress
    at: '2026-01-07T03:52:54.683Z'
depends_on:
  - 012-channel-loss-compatibility-verification
  - 013-channel-loss-optimizations-and-robustness
---

# Channel Loss: Dataset Index-Based Channel Injection

> **Status**: ⏳ In progress · **Priority**: High · **Created**: 2026-01-07 · **Tags**: channel-loss, plugin-architecture, dataset-injection, bugfix, p0

## Overview

Implement plugin context storage for dataset-to-channel mapping to survive Pydantic config validation, enabling collator-side channel injection using `dataset_idx` from batch features.

### Problem Statement

**P0-1 Implementation Defect**: The original implementation stored `_channel_loss_channel` in dataset config during `register()`, expecting it to be injected into features during dataset loading. However, this approach failed because:

1. `validate_config()` uses Pydantic `SFTDataset` schema to reconstruct dataset configs
2. `SFTDataset` schema doesn't include `_channel_loss_channel` field and doesn't allow extra fields (`extra="forbid"` is default)
3. The field gets silently dropped during `model_dump()` reconstruction
4. Dataset loading code at `sft.py:400-417` never sees the field, so injection never occurs
5. Tests passed only because they used mocks that bypassed `validate_config()`

**Impact**: Channel Loss feature completely non-functional for dataset-level channel configuration (P0-1 priority requirement from Spec 013).

### Solution Approach

**Solution A: Plugin Context Storage** (chosen over Solution B: schema modification)

Store dataset-to-channel mapping in plugin instance variable that survives config validation:

```python
# In ChannelLossPlugin
self._dataset_channels = {
    0: "math",      # dataset index → channel name
    1: "code",
    2: "reasoning"
}
```

Pass this mapping to collator wrapper, which injects channel at batch collation time using `dataset_idx` from features.

**Why Solution A over Solution B**:
- Doesn't require modifying Axolotl's core schemas
- Plugin-owned data stays in plugin scope
- Lower risk of conflicts with future Axolotl updates
- Clean separation: plugin state vs. validated config state

## Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Plugin.register()                                        │
│    - Extract 'channel' from dataset configs                 │
│    - Store in self._dataset_channels = {idx: channel}       │
│    - Pop 'channel' to avoid validation errors               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. validate_config()                                        │
│    - Pydantic validates dataset configs                     │
│    - 'channel' field already removed, no error              │
│    - self._dataset_channels preserved in plugin instance    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Plugin.post_trainer_create()                             │
│    - Wrap data collator                                     │
│    - Pass self._dataset_channels to wrapper                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Collator wrapper (each batch)                            │
│    - Extract dataset_idx from features                      │
│    - Lookup channel: dataset_channels[dataset_idx]          │
│    - Inject channel into features                           │
└─────────────────────────────────────────────────────────────┘
```

### Key Implementation Details

#### 1. Plugin Instance Storage

**File**: `src/axolotl/integrations/channel_loss/__init__.py`

**Added `__init__()` method**:
```python
def __init__(self):
    """Initialize plugin with empty dataset channel mapping."""
    super().__init__()
    self._dataset_channels = {}  # Maps dataset index to channel name
```

**Modified `register()` method** (lines 175-189):
```python
# Extract channel from dataset configs and store in plugin instance
self._dataset_channels = {}
for idx, ds in enumerate(cfg.get("datasets", [])):
    # Pop channel to avoid schema validation errors
    ch = ds.pop("channel", None)
    if ch is not None:
        # Store in plugin instance variable (survives validate_config)
        self._dataset_channels[idx] = ch

LOG.info(
    f"Channel Loss Plugin: Extracted channels from {len(self._dataset_channels)} datasets"
)
```

**Rationale**:
- Use `pop()` to remove field from config before validation
- Store in instance variable that persists through entire training lifecycle
- Log extraction count for debugging

#### 2. Collator Wrapper Integration

**File**: `src/axolotl/integrations/channel_loss/__init__.py`

**Modified `post_trainer_create()` method** (lines 237-258):
```python
# Wrap data collator with dataset channel mapping
trainer.data_collator = wrap_collator_for_channel_loss(
    inner_collator=trainer.data_collator,
    channel_field=channel_field,
    dataset_channels=self._dataset_channels,  # Pass plugin mapping
    warn_on_missing=warn_on_missing,
)

# Also wrap eval collator if exists
if hasattr(trainer, "eval_data_collator") and trainer.eval_data_collator is not None:
    trainer.eval_data_collator = wrap_collator_for_channel_loss(
        inner_collator=trainer.eval_data_collator,
        channel_field=channel_field,
        dataset_channels=self._dataset_channels,  # Pass plugin mapping
        warn_on_missing=warn_on_missing,
    )
```

#### 3. Dataset Index-Based Injection

**File**: `src/axolotl/integrations/channel_loss/collator_wrapper.py`

**Injection logic** (to be implemented):
```python
def _process_standard_batch(features, channel_field, dataset_channels, warn_on_missing):
    for feat in features:
        # First try: explicit channel field in sample
        ch = feat.pop(channel_field, None)

        # Second try: dataset-level channel via dataset_idx
        if ch is None and dataset_channels:
            dataset_idx = feat.get("dataset_idx")
            if dataset_idx is not None:
                ch = dataset_channels.get(dataset_idx)

        # Fallback
        if ch is None:
            ch = "default"

        # Inject into features
        feat[channel_field] = ch
```

#### 4. Dead Code Removal

**File**: `src/axolotl/utils/data/sft.py`

**Removed lines 400-417**: The ineffective injection logic that relied on `_channel_loss_channel` field surviving validation.

**Rationale**: This code never executed because the field was stripped by `validate_config()`. Removing it prevents confusion and reduces maintenance burden.

### Edge Cases Handled

1. **No channel specified**: Falls back to `"default"` channel
2. **dataset_idx missing**: Falls back to explicit channel field or `"default"`
3. **Mixed specification**: Explicit sample-level channel overrides dataset-level channel
4. **Eval collator**: Same logic applied to eval collator if it exists

## Plan

- [x] **Phase 1: Plugin Instance Storage**
  - [x] Add `__init__()` method to `ChannelLossPlugin`
  - [x] Initialize `self._dataset_channels = {}`
  - [x] Document purpose in docstring

- [x] **Phase 2: Extract and Store Channels**
  - [x] Modify `register()` to iterate over datasets
  - [x] Pop `channel` field from each dataset config
  - [x] Store in `self._dataset_channels[idx] = channel`
  - [x] Add logging for extraction count

- [x] **Phase 3: Pass Mapping to Collator**
  - [x] Modify `post_trainer_create()` to pass `dataset_channels` parameter
  - [x] Update both train and eval collator wrapping
  - [x] Verify parameter passing

- [x] **Phase 4: Remove Dead Code**
  - [x] Identify ineffective injection logic in `sft.py`
  - [x] Remove lines 400-417
  - [x] Verify no references remain

- [ ] **Phase 5: Collator-Side Injection** (INCOMPLETE)
  - [ ] Modify `_process_standard_batch()` to use `dataset_idx`
  - [ ] Modify `_process_packing_batch()` to use `dataset_idx`
  - [ ] Implement fallback chain: explicit → dataset-level → default
  - [ ] Handle None values consistently

- [ ] **Phase 6: Testing and Validation**
  - [ ] Update existing tests to verify real config validation flow
  - [ ] Add test for `dataset_idx` injection
  - [ ] Add test for fallback chain
  - [ ] Verify with multi-domain config

## Test

### Verification Criteria

- [x] **Config Validation**: `channel` field in dataset config doesn't cause validation errors
- [x] **Plugin Storage**: `self._dataset_channels` populated correctly in `register()`
- [x] **Collator Parameter**: `dataset_channels` passed to collator wrapper
- [ ] **Channel Injection**: Features contain correct channel based on `dataset_idx`
- [ ] **Fallback Behavior**: Falls back to `"default"` when channel not found
- [ ] **Multi-Domain Training**: Per-channel metrics logged for each dataset

### Test Cases

1. **test_channel_extracted_from_config**:
   - Given: Dataset config with `channel: "math"`
   - When: `register()` called
   - Then: `self._dataset_channels[0] == "math"`

2. **test_channel_removed_before_validation**:
   - Given: Dataset config with `channel: "math"`
   - When: `validate_config()` called
   - Then: No validation error, field removed from config

3. **test_channel_injected_via_dataset_idx**:
   - Given: Feature with `dataset_idx: 0`, plugin has `{0: "math"}`
   - When: Collator processes batch
   - Then: Feature contains `channel_field: "math"`

4. **test_fallback_to_default**:
   - Given: Feature with no channel, no dataset_idx
   - When: Collator processes batch
   - Then: Feature contains `channel_field: "default"`

### Integration Testing

**Test config**: `qwen3-8b-fsdp-tp-cp-channel-loss-multi-domain.yaml`

Expected behavior:
```yaml
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
    channel: general           # Should be extracted to _dataset_channels[0]

  - path: /path/to/math_data
    type: chat_template
    channel: math              # Should be extracted to _dataset_channels[1]
```

Expected logs:
```
Channel Loss Plugin: Extracted channels from 2 datasets
Step 100: loss=1.234, loss_general=1.456, loss_math=0.987
```

## Notes

### Alternative Solutions Considered

**Solution B: Modify SFTDataset Schema**

Add `channel` field to `SFTDataset` Pydantic schema:
```python
class SFTDataset(BaseModel):
    path: str
    type: str
    channel: Optional[str] = None  # Add this
```

**Rejected because**:
- Requires modifying Axolotl core schemas (outside plugin scope)
- Risk of conflicts with upstream updates
- Violates plugin architecture (plugin should be self-contained)
- Adds field to schema that's only relevant when plugin is active

### Additional Issues Discovered

During code review, the following issues were identified but not yet fixed:

1. **None handling inconsistency**:
   - Code checks `if channel_field not in example`
   - Test expects `example[channel_field] = None` to be treated as missing
   - Should consistently use `if not example.get(channel_field)`

2. **Incomplete batch size checking**:
   - Only validates `micro_batch_size` for packing mode
   - Misses `per_device_eval_batch_size` validation
   - Should check both train and eval batch sizes

3. **Inaccurate optimization claim**:
   - Documentation claimed "消除 GPU-CPU 同步" (eliminate GPU-CPU sync)
   - Actually only reduced from O(n) to O(1) synchronization per batch
   - Should accurately describe as "optimize" not "eliminate"

These will be addressed in future optimization work.

### Dependencies

- **Depends on**:
  - Spec 012: Channel Loss Plugin base implementation
  - Spec 013: Optimizations and robustness improvements

- **Required by**: Any multi-domain training using dataset-level channel configuration

### Implementation Status

**Completed**:
- ✅ Plugin instance storage (`__init__`, instance variable)
- ✅ Channel extraction in `register()` (pop from config, store in plugin)
- ✅ Collator parameter passing in `post_trainer_create()`
- ✅ Dead code removal from `sft.py`

**Incomplete**:
- ⏸️ Collator-side injection using `dataset_idx`
- ⏸️ Test updates for real config validation flow
- ⏸️ Integration testing with multi-domain config

**Commits**:
- `feat(channel_loss): add Channel Loss Plugin for per-channel loss tracking` (7000ee7f)
- `feat(channel_loss): enhance dataset processing for dynamic channel loss field` (835a6738)
- `feat(channel_loss): add Context Parallelism support with micro_batch_size fix` (369a43a2)
- `fix(channel_loss): prevent deadlock in distributed callback synchronization` (5e125d8f)
- `test(channel_loss): add comprehensive CP and batch size tests` (7eddb418)
- `fix(sequence_parallel): prevent division by zero in num_items_in_batch calculation` (741e61e9)
- `feat(channel-loss): implement dataset index tracking mechanism` (260df3c5)
- `perf(channel-loss): optimize segment boundary detection` (2f175d51)
- `refactor(channel-loss): reduce production logging noise` (56a6c4db)
- `test(channel-loss): update tests for dataset index tracking` (e805e23e)

**Branch**: `feature/channel-loss` (rebased on `upstream/main`, pushed to `origin/feature/channel-loss`)
