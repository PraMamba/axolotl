# Critical Bug: Channel Loss Statistics Not Being Recorded with CP

**Status**: 🔴 REGRESSION FOUND (2025-12-29 22:30)
**Severity**: HIGH - Channel Loss功能完全失效
**Affected**: CP > 1 configurations
**Root Cause**: Segment boundary calculation incompatible with shifted tensors

---

## Problem Summary

Despite successful shape alignment and stable training with CP=2, **per-channel loss statistics are NOT being recorded**.

### Observed Symptoms

1. ✅ Training runs without errors
2. ✅ Loss values are normal (1.6562, 3.1094, 5.875, etc.)
3. ❌ **No per-channel loss metrics in logs** (missing `loss=task_A`, `loss=task_B`, etc.)
4. ❌ **No "Channel Loss: Tracking new channel" messages**
5. ✅ Collator correctly extracts channel metadata
6. ✅ Channels successfully passed to compute_loss

### Evidence from Logs

```log
# Expected behavior (MISSING):
{'loss': 3.1094, 'loss=cell_type_identification': 2.8, 'loss=cell_type_identification_from_topk_genes': 3.5, ...}
Channel Loss: Tracking new channel 'loss=cell_type_identification'

# Actual behavior:
{'loss': 1.6562, 'grad_norm': 97.5, ...}  # ← No per-channel metrics!
[Collator] Extracted channel from first feature: cell_type_identification  # ← Extraction works
[RANK 0] Input shapes - channels: ['cell_type_identification']  # ← Channels passed correctly
```

---

## Root Cause Analysis

### Data Flow Investigation

1. **Collator** ✅ Working correctly
   ```log
   [Collator] Received 1 features, first feature keys: ['task_type', ...]
   [Collator] Extracted channel from first feature: cell_type_identification
   ```

2. **Channel Extraction** ✅ Working correctly
   ```log
   [RANK 0] Input shapes - channels: ['cell_type_identification']
   ```

3. **Statistics Accumulation** 🔴 **FAILING**
   ```log
   [RANK 0] Channel stats accumulation - flat_channels: ['cell_type_identification'], num_segments: 1, cu_seqlens.shape: torch.Size([2]), prefix: loss=
   [RANK 0] Segment 0: channel=cell_type_identification, start=0, end=4096, per_token_loss.shape=torch.Size([2047])
   [RANK 0] Segment 0: Skipped (out of bounds)  # ← BUG!
   ```

### The Core Issue

**Segment Boundary Mismatch**: `cu_seqlens` contains boundaries for 4096-token sequence, but `per_token_loss` only has 2047 tokens.

#### Why This Happens

```python
# In compute_loss_with_channel (lines 154-190):
logits_to_use = AllGatherWithGrad.apply(outputs.logits, cp_group)
# logits: 1024 → 2048 (gathered across CP=2)

labels_to_use = AllGatherWithGrad.apply(labels, cp_group)
# labels: 2048 → 4096 (gathered across CP=2)

# Trim labels to match logits (handles data-specific ratio)
if logits_to_use.size(1) != labels_to_use.size(1):  # 2048 != 4096
    labels_to_use = labels_to_use[:, :logits_to_use.size(1)].contiguous()
# labels: 4096 → 2048 (trimmed)

# In _update_channel_stats (lines 279-351):
shift_logits = logits[..., :-1, :].contiguous()  # 2048 → 2047
shift_labels = labels[..., 1:].contiguous()       # 2048 → 2047

# Compute per-token loss
per_token_loss = loss_fct(shift_logits.view(-1, ...), shift_labels.view(-1))
# per_token_loss.shape = (2047,)  ← Only 2047 tokens!

# BUT: get_segment_boundaries uses ORIGINAL labels (4096 tokens before trim)
cu_seqlens = get_segment_boundaries(
    attention_mask=attention_mask,  # Still 4096 tokens!
    position_ids=position_ids,      # Still 4096 tokens!
    labels=labels,                   # Still 4096 tokens! (pre-shift)
    mode=segment_mode,
)
# cu_seqlens = tensor([0, 4096])  ← Boundaries for 4096-token sequence!

# Result: Bounds check fails
for i in range(num_segments):
    start = cu_seqlens[i].item()    # 0
    end = cu_seqlens[i + 1].item()  # 4096
    if end > per_token_loss.shape[0]:  # 4096 > 2047 → TRUE
        continue  # ← SKIPPED! No statistics accumulated!
```

### Timeline of Labels Transformations

| Step | Operation | Shape | Location |
|------|-----------|-------|----------|
| 1 | Input | `(1, 2048)` | Collator output |
| 2 | CP Gather | `(1, 4096)` | `compute_loss_with_channel:158` |
| 3 | Trim to logits | `(1, 2048)` | `compute_loss_with_channel:190` |
| 4 | Shift by 1 | `(1, 2047)` | `_update_channel_stats:281` |
| 5 | Flatten | `(2047,)` | `_update_channel_stats:339` |

But `attention_mask` and `position_ids` are still at Step 2 (4096 tokens) when `get_segment_boundaries` is called!

---

## Attempted Fixes

### Attempt 1: Use `shift_labels` instead of `labels` ❌

```python
# Fix:
cu_seqlens = get_segment_boundaries(
    attention_mask=attention_mask,
    position_ids=position_ids,
    labels=shift_labels,  # ← Use shifted labels
    mode=segment_mode,
)

# Result: FAILED - Still end=4096
# Reason: attention_mask and position_ids are still 4096 tokens,
#         get_segment_boundaries uses those instead of labels
```

### Attempt 2: Shift `attention_mask` and `position_ids` ❌

```python
# Fix:
shift_attention_mask = attention_mask[..., 1:] if attention_mask is not None else None
shift_position_ids = position_ids[..., 1:] if position_ids is not None else None

cu_seqlens = get_segment_boundaries(
    attention_mask=shift_attention_mask,
    position_ids=shift_position_ids,
    labels=shift_labels,
    mode=segment_mode,
)

# Result: FAILED - num_segments=0, cu_seqlens.shape=torch.Size([1])
# Reason: Shifting broke segment detection logic in get_segment_boundaries
```

### Attempt 3: Clamp `cu_seqlens` to `per_token_loss` length ⏳

```python
# Fix:
cu_seqlens = get_segment_boundaries(
    attention_mask=attention_mask,
    position_ids=position_ids,
    labels=labels,
    mode=segment_mode,
)

max_len = per_token_loss.shape[0]
cu_seqlens = torch.clamp(cu_seqlens, max=max_len)

# Expected: cu_seqlens = tensor([0, 2047])
# Status: Testing in progress...
```

---

## Correct Solution (Proposed)

The issue is that `get_segment_boundaries` needs to operate on tensors with the SAME sequence length as `per_token_loss`.

### Option A: Pass Trimmed Labels to `get_segment_boundaries`

```python
# In compute_loss_with_channel, pass trimmed labels to _update_channel_stats
_update_channel_stats(
    trainer=trainer,
    logits=logits_to_use,         # 2048 tokens (gathered + trimmed)
    labels=labels_to_use,          # 2048 tokens (gathered + trimmed) ← Already trimmed!
    channels=channels,
    position_ids=position_ids_to_use,     # 4096 tokens (gathered, NOT trimmed)
    attention_mask=attention_mask_to_use, # 4096 tokens (gathered, NOT trimmed)
    segment_mode=segment_mode,
    prefix=prefix,
    cp_already_gathered=True,
)

# In _update_channel_stats:
# 1. Shift all tensors FIRST
shift_logits = logits[..., :-1, :].contiguous()      # 2048 → 2047
shift_labels = labels[..., 1:].contiguous()           # 2048 → 2047
shift_attention_mask = attention_mask[..., 1:] if attention_mask is not None else None  # 4096 → 4095 (WRONG!)
shift_position_ids = position_ids[..., 1:] if position_ids is not None else None        # 4096 → 4095 (WRONG!)

# Problem: attention_mask/position_ids still don't match!
```

**Issue**: `attention_mask` and `position_ids` are passed at 4096 tokens, but `labels` are passed at 2048 tokens.

### Option B: Trim ALL Tensors Before Passing ✅ **RECOMMENDED**

```python
# In compute_loss_with_channel (MODIFY lines 166-176):
if cp_size > 1:
    # Gather all tensors
    logits_to_use = AllGatherWithGrad.apply(outputs.logits, cp_group)
    labels_to_use = AllGatherWithGrad.apply(labels, cp_group)

    if position_ids_to_use is not None:
        position_ids_to_use = AllGatherWithGrad.apply(position_ids_to_use, cp_group)
    if attention_mask_to_use is not None:
        attention_mask_to_use = AllGatherWithGrad.apply(attention_mask_to_use, cp_group)

    # CRITICAL: Trim ALL tensors to match logits length
    logits_seq_len = logits_to_use.size(1)
    labels_seq_len = labels_to_use.size(1)

    if logits_seq_len != labels_seq_len:
        LOG.warning(f"[RANK {dist.get_rank()}] Length mismatch after gather: logits={logits_seq_len}, labels={labels_seq_len}. Trimming all tensors to match logits.")

        # Trim labels
        labels_to_use = labels_to_use[:, :logits_seq_len].contiguous()

        # Trim position_ids
        if position_ids_to_use is not None:
            position_ids_to_use = position_ids_to_use[:, :logits_seq_len].contiguous()

        # Trim attention_mask
        if attention_mask_to_use is not None:
            attention_mask_to_use = attention_mask_to_use[:, :logits_seq_len].contiguous()

# Now all tensors have 2048 tokens
# After shift in _update_channel_stats, all will be 2047 tokens
# get_segment_boundaries will produce boundaries for 2047-token sequence
```

### Option C: Adjust Boundaries Arithmetically ⚠️ **RISKY**

```python
# After get_segment_boundaries:
cu_seqlens = get_segment_boundaries(...)  # Returns boundaries for 4096-token seq

# Compute scale factor
original_seq_len = labels.size(1)         # 4096
actual_seq_len = per_token_loss.shape[0]  # 2047
scale = actual_seq_len / original_seq_len # 0.499...

# Scale boundaries
cu_seqlens = (cu_seqlens.float() * scale).long()

# Problem: May produce incorrect boundaries for packing mode!
# Example: [0, 2048, 4096] → [0, 1023, 2046] (loses precision)
```

---

## Recommended Action Plan

1. **Immediate Fix**: Implement Option B (trim all tensors to logits length) ✅
   - Location: `src/axolotl/integrations/channel_loss/compute_loss_patch.py:166-176`
   - Risk: Low (isolated change)
   - Validation: Check `cu_seqlens` values in debug logs

2. **Add Debug Logging**: Log tensor shapes at key points
   ```python
   LOG.info(f"[RANK {rank}] After trim: logits={logits.shape}, labels={labels.shape}, attention_mask={attention_mask.shape if attention_mask else None}")
   LOG.info(f"[RANK {rank}] After shift: shift_logits={shift_logits.shape}, shift_labels={shift_labels.shape}")
   LOG.info(f"[RANK {rank}] cu_seqlens: {cu_seqlens}")
   ```

3. **Remove Excessive Debug Logging**: Clean up temporary debug code after validation
   - Lines with "Segment 0:" logging
   - Lines with "Accumulated for key" logging
   - Keep only high-level stats accumulation logs

4. **Update LEAN_SPEC.md**:
   - Change status from "✅ Production Validated" to "⚠️  Bug Found - Statistics Not Recording"
   - Add this issue to troubleshooting section

5. **Re-validate**: Run training with CP=2 and verify:
   - `cu_seqlens` has correct boundaries
   - Per-channel statistics are accumulated
   - Callback logs "Channel Loss: Tracking new channel" messages
   - Training metrics include `loss=<channel_name>` fields

---

## Files Requiring Modification

```
src/axolotl/integrations/channel_loss/
└── compute_loss_patch.py
    ├── Lines 166-176: Add trimming for position_ids and attention_mask
    ├── Lines 275-360: Remove excessive debug logging
    └── Lines 345-375: Verify cu_seqlens calculation
```

---

## Success Criteria

Training logs should show:

```log
# 1. Collator extraction (ALREADY WORKING)
[Collator] Extracted channel from first feature: cell_type_identification

# 2. Channel passed to compute_loss (ALREADY WORKING)
[RANK 0] Input shapes - channels: ['cell_type_identification']

# 3. Correct tensor alignment (NEEDS FIX)
[RANK 0] After trim: logits=(1,2048,152696), labels=(1,2048), attention_mask=(1,2048)
[RANK 0] After shift: shift_logits=(1,2047,152696), shift_labels=(1,2047)

# 4. Correct segment boundaries (NEEDS FIX)
[RANK 0] cu_seqlens: tensor([0, 2047])  # ← NOT [0, 4096]!

# 5. Successful accumulation (NEEDS FIX)
[RANK 0] Segment 0: channel=cell_type_identification, start=0, end=2047, per_token_loss.shape=torch.Size([2047])
[RANK 0] Segment 0: segment_loss.shape=torch.Size([2047]), valid_loss.numel()=1850
[RANK 0] Accumulated for key 'loss=cell_type_identification': sum=4320.1234, count=1850

# 6. Callback logging (NEEDS FIX)
Channel Loss: Tracking new channel 'loss=cell_type_identification'

# 7. Metrics in training logs (NEEDS FIX)
{'loss': 3.1094, 'loss=cell_type_identification': 2.8, 'loss=cell_type_identification_from_topk_genes': 3.5, ...}
```

---

## References

- **Regression Context**: Session starting 2025-12-29 22:00
- **Previous Validation**: LEAN_SPEC.md (marked production ready but statistics were not validated)
- **Related Issue**: Shape alignment works, but statistics accumulation fails
- **Code Location**: `src/axolotl/integrations/channel_loss/compute_loss_patch.py`

---

**Last Updated**: 2025-12-29 22:35
**Reported By**: Production debugging session
**Priority**: HIGH - Core functionality broken
**Next Step**: Implement Option B (trim all tensors)
