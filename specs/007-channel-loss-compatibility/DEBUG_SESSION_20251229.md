# Debug Session Summary - 2025-12-29 22:00-22:35

**Session Type**: Production Regression Investigation
**Duration**: ~35 minutes
**Outcome**: Critical bug identified, root cause confirmed, solution proposed
**Status**: 🔴 Fix Required

---

## Session Context

Following the initial "production validation" at 21:02, user reported that **per-channel loss statistics were not appearing** in training logs despite training running successfully.

### Initial Observation

```log
# Expected (MISSING):
{'loss': 3.1094, 'loss=cell_type_identification': 2.8, 'loss=cell_type_identification_from_topk_genes': 3.5, ...}
Channel Loss: Tracking new channel 'loss=cell_type_identification'

# Actual:
{'loss': 1.6562, 'grad_norm': 97.5, ...}  # No per-channel metrics!
```

---

## Investigation Timeline

### Phase 1: Data Flow Verification (22:00-22:10)

**Hypothesis**: Channel field not being extracted or passed correctly.

**Tests**:
1. ✅ Verified dataset contains `task_type` field
2. ✅ Verified collator extracts channel correctly
3. ✅ Verified channels passed to `compute_loss`

**Evidence**:
```log
[Collator] Received 1 features, first feature keys: ['task_type', 'input_ids', 'labels', 'attention_mask']
[Collator] Extracted channel from first feature: cell_type_identification
[RANK 0] Input shapes - channels: ['cell_type_identification']
```

**Conclusion**: Data flow is correct up to `_update_channel_stats`.

### Phase 2: Statistics Accumulation Investigation (22:10-22:20)

**Hypothesis**: Statistics accumulation logic has a bug.

**Tests**:
1. Added debug logging to `_update_channel_stats`
2. Discovered segments being skipped due to bounds check failure

**Critical Discovery**:
```log
[RANK 0] Channel stats accumulation - flat_channels: ['cell_type_identification'], num_segments: 1, cu_seqlens.shape: torch.Size([2]), prefix: loss=
[RANK 0] Segment 0: channel=cell_type_identification, start=0, end=4096, per_token_loss.shape=torch.Size([2047])
[RANK 0] Segment 0: Skipped (out of bounds)  # ← BUG!
```

**Root Cause Identified**: Segment boundaries (`cu_seqlens`) computed for 4096-token sequence, but `per_token_loss` only has 2047 tokens.

### Phase 3: Tensor Shape Analysis (22:20-22:30)

**Traced tensor transformations**:

| Step | Location | Labels Shape | Notes |
|------|----------|--------------|-------|
| 1 | Collator output | `(1, 2048)` | Original input |
| 2 | After CP gather | `(1, 4096)` | `AllGatherWithGrad` across CP=2 |
| 3 | After trim | `(1, 2048)` | Trimmed to match logits (1:2 ratio) |
| 4 | After shift | `(1, 2047)` | Causal LM shift (-1 token) |
| 5 | Flattened | `(2047,)` | Used for per-token loss |

**But**: `attention_mask` and `position_ids` used by `get_segment_boundaries` are still at **Step 2 (4096 tokens)**!

**Result**: Segment boundaries calculated for 4096-token sequence don't match 2047-token `per_token_loss`.

### Phase 4: Solution Exploration (22:30-22:35)

**Attempted fixes**:

1. **Attempt 1**: Use `shift_labels` instead of `labels` in `get_segment_boundaries`
   - Result: ❌ Failed - still `end=4096` (function uses `attention_mask` instead)

2. **Attempt 2**: Shift `attention_mask` and `position_ids` too
   - Result: ❌ Failed - broke segment detection (`num_segments=0`)

3. **Attempt 3**: Clamp `cu_seqlens` to `per_token_loss` length
   - Result: ⏳ Testing in progress (interrupted)

**Recommended Solution**: Trim `attention_mask` and `position_ids` to match logits length BEFORE passing to `_update_channel_stats`.

---

## Root Cause

### The Problem

```python
# In compute_loss_with_channel:
logits_to_use = AllGatherWithGrad.apply(outputs.logits, cp_group)
# logits: 1024 → 2048 tokens

labels_to_use = AllGatherWithGrad.apply(labels, cp_group)
# labels: 2048 → 4096 tokens

# Trim labels to match logits
labels_to_use = labels_to_use[:, :logits_to_use.size(1)].contiguous()
# labels: 4096 → 2048 tokens

# BUT: attention_mask and position_ids NOT trimmed!
# attention_mask: still 4096 tokens
# position_ids: still 4096 tokens

# In _update_channel_stats:
shift_logits = logits[..., :-1, :].contiguous()  # 2048 → 2047
shift_labels = labels[..., 1:].contiguous()       # 2048 → 2047
# per_token_loss computed from shift_logits and shift_labels
# per_token_loss.shape = (2047,)

# But get_segment_boundaries uses 4096-token attention_mask!
cu_seqlens = get_segment_boundaries(
    attention_mask=attention_mask,  # 4096 tokens!
    position_ids=position_ids,      # 4096 tokens!
    labels=labels,                   # 2048 tokens (pre-shift)
    mode=segment_mode,
)
# cu_seqlens = tensor([0, 4096])  ← Wrong boundaries!

# Bounds check:
if end > per_token_loss.shape[0]:  # 4096 > 2047 → TRUE
    continue  # ← Skipped! No statistics!
```

### Why This Wasn't Caught Earlier

1. **Initial validation focused on shape alignment**: Checked that logits and labels match for loss computation
2. **Did not validate statistics output**: Only checked that training doesn't crash
3. **Assumed statistics "just work"**: Didn't verify per-channel metrics in logs

**Lesson**: Always validate END-TO-END functionality, not just intermediate steps.

---

## Proposed Solution

### Option B: Trim ALL Tensors (RECOMMENDED)

Modify `compute_loss_with_channel` to trim `attention_mask` and `position_ids` along with `labels`:

```python
# File: src/axolotl/integrations/channel_loss/compute_loss_patch.py
# Location: Lines 166-190

if cp_size > 1:
    # Gather all tensors
    logits_to_use = AllGatherWithGrad.apply(outputs.logits, cp_group)
    labels_to_use = AllGatherWithGrad.apply(labels, cp_group)

    if position_ids_to_use is not None:
        position_ids_to_use = AllGatherWithGrad.apply(position_ids_to_use, cp_group)

    if attention_mask_to_use is not None:
        attention_mask_to_use = AllGatherWithGrad.apply(attention_mask_to_use, cp_group)

    # CRITICAL FIX: Trim ALL tensors to match logits length
    logits_seq_len = logits_to_use.size(1)
    labels_seq_len = labels_to_use.size(1)

    if logits_seq_len != labels_seq_len:
        LOG.warning(
            f"[RANK {dist.get_rank()}] Length mismatch after gather: "
            f"logits={logits_seq_len}, labels={labels_seq_len}. "
            f"Trimming all tensors to match logits."
        )

        # Trim labels (EXISTING)
        labels_to_use = labels_to_use[:, :logits_seq_len].contiguous()

        # Trim position_ids (NEW)
        if position_ids_to_use is not None:
            position_ids_to_use = position_ids_to_use[:, :logits_seq_len].contiguous()

        # Trim attention_mask (NEW)
        if attention_mask_to_use is not None:
            attention_mask_to_use = attention_mask_to_use[:, :logits_seq_len].contiguous()
```

**Expected Result**:
- All tensors: 2048 tokens after trim
- After shift in `_update_channel_stats`: 2047 tokens
- `cu_seqlens` computed for 2047-token sequence: `tensor([0, 2047])`
- Bounds check passes: `0 <= 2047 <= 2047` ✅
- Statistics accumulated correctly ✅

### Validation Criteria

After fix, logs should show:

```log
# 1. Tensor alignment
[RANK 0] After trim: logits=(1,2048,152696), labels=(1,2048), attention_mask=(1,2048), position_ids=(1,2048)
[RANK 0] After shift: shift_logits=(1,2047,152696), shift_labels=(1,2047)

# 2. Correct boundaries
[RANK 0] cu_seqlens: tensor([0, 2047])  # NOT [0, 4096]!

# 3. Successful accumulation
[RANK 0] Segment 0: channel=cell_type_identification, start=0, end=2047, per_token_loss.shape=torch.Size([2047])
[RANK 0] Segment 0: segment_loss.shape=torch.Size([2047]), valid_loss.numel()=1850
[RANK 0] Accumulated for key 'loss=cell_type_identification': sum=4320.1234, count=1850

# 4. Callback logging
Channel Loss: Tracking new channel 'loss=cell_type_identification'

# 5. Metrics in logs
{'loss': 3.1094, 'loss=cell_type_identification': 2.8, 'loss=cell_type_identification_from_topk_genes': 3.5, ...}
```

---

## Code Changes Required

### File: `src/axolotl/integrations/channel_loss/compute_loss_patch.py`

**Location**: Lines 166-190 (in `compute_loss_with_channel` function)

**Change Type**: Addition

**Lines to Add**: 8 lines (trimming position_ids and attention_mask)

**Risk**: Low - isolated change, similar to existing label trimming

**Testing**: Run training with CP=2, verify:
1. Training stability (should continue working)
2. Per-channel metrics appear in logs
3. Callback messages logged
4. No performance regression

---

## Cleanup Required

After fix is validated, remove excessive debug logging:

### File: `src/axolotl/integrations/channel_loss/compute_loss_patch.py`

**Remove**:
1. Lines with "Segment 0: channel=" logging (~Line 378-380)
2. Lines with "Segment 0: segment_loss.shape=" logging (~Line 395-397)
3. Lines with "Segment 0: Skipped" logging (~Line 385, 388)
4. Lines with "After finite filter" logging (~Line 405-407)
5. Lines with "Accumulated for key" logging (~Line 414-416)

**Keep**:
- High-level "Channel stats accumulation" log (Line 364-369)
- "Input shapes" log (Line 104-110)
- CP gather logs (Lines 149-164, 274-289, 304-309)

---

## Session Artifacts

### Log Files
- `/data/Mamba/Project/Single_Cell/Training/.../logs/model_training_20251229_21.log` (training with old code, channels=None)
- `/data/Mamba/Project/Single_Cell/Training/.../logs/model_training_20251229_22.log` (training with debug logging, segment skipped)

### Debug Scripts
- `/tmp/training_with_collator_debug.log`
- `/tmp/channel_loss_debug_v5.log`
- `/tmp/training_restart_debug.log`
- `/tmp/training_detailed_debug.log`
- `/tmp/training_fix_v1.log`
- `/tmp/training_fix_v2.log`
- `/tmp/training_fix_v3.log`

### Key Commands Used
```bash
# Verify dataset has task_type field
python3 -c "from datasets import load_from_disk; ds = load_from_disk('...'); print(ds.column_names)"

# Monitor training logs
tail -f .../model_training_20251229_22.log | grep -E "(Channel|Segment|Accumulated)"

# Search for specific log patterns
grep "Segment 0:" .../model_training_20251229_22.log | head -20
grep "Channel stats accumulation" .../model_training_20251229_22.log | tail -10
```

---

## Key Insights

### 1. Validation Must Be End-to-End

**Wrong**: "Training doesn't crash, so it works."

**Right**: "Training runs AND produces expected outputs (per-channel metrics)."

**Lesson**: Always validate business logic outputs, not just technical stability.

### 2. Tensor Length Consistency Is Critical

When working with sequence parallelism:
- **All tensors used together must have matching sequence lengths**
- Just trimming labels is insufficient if other tensors (attention_mask, position_ids) are used downstream
- Mismatched lengths cause silent failures (bounds checks, empty iterations)

### 3. Debug Logging Strategy

**Effective patterns used**:
1. Log shapes at transformation points: "Before gather", "After gather", "After shift"
2. Log segment processing details: "Segment 0: start, end, per_token_loss.shape"
3. Log accumulation results: "Accumulated for key X: sum, count"

**Ineffective patterns**:
- Too much logging (flooded logs, hard to find signal)
- Logging only final results (can't trace intermediate failures)

### 4. Bounds Check Philosophy

**Current code**: Silently skip segments that fail bounds check
```python
if start >= per_token_loss.shape[0] or end > per_token_loss.shape[0]:
    continue  # ← Silent failure!
```

**Better approach**: Log warning when skipping
```python
if start >= per_token_loss.shape[0] or end > per_token_loss.shape[0]:
    LOG.warning(f"[RANK {rank}] Segment {i} out of bounds: start={start}, end={end}, per_token_loss.shape={per_token_loss.shape}")
    continue
```

**Rationale**: Silent failures are hard to debug. Warnings make issues visible.

---

## Next Steps

1. **Implement Option B**: Trim attention_mask and position_ids (8 lines of code)
2. **Test with CP=2**: Verify per-channel metrics appear
3. **Remove debug logging**: Clean up excessive logs
4. **Add bounds check warning**: Make future failures visible
5. **Re-validate production**: Run 50-100 steps, verify metrics stable
6. **Update LEAN_SPEC.md**: Change status back to ✅ Production Validated
7. **Update INDEX.md**: Remove critical issue warning

---

## References

- **Bug Report**: `CP_STATISTICS_BUG.md`
- **Original Spec**: `LEAN_SPEC.md`
- **Implementation**: `src/axolotl/integrations/channel_loss/compute_loss_patch.py`
- **Configuration**: `/home/scbjtfy/RVQ-Alpha/configs/axolotl/7b-fsdp2-tp-cp_sft_channel-loss.yaml`

---

**Session Lead**: Claude (Sonnet 4.5)
**Session Date**: 2025-12-29 22:00-22:35
**Session Duration**: ~35 minutes
**Session Result**: Root cause identified, solution designed, ready for implementation
