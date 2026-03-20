---
status: complete
created: '2025-12-30'
tags:
  - bugfix
  - cp
  - micro_batch_size
  - contiguous
priority: high
created_at: '2025-12-30T07:44:55.623Z'
updated_at: '2025-12-30T07:47:17.133Z'
completed_at: '2025-12-30T07:47:17.133Z'
completed: '2025-12-30'
transitions:
  - status: complete
    at: '2025-12-30T07:47:17.133Z'
---

# Fix micro_batch_size > 1 RuntimeError with CP-local statistics

> **Status**: ✅ Complete · **Priority**: High · **Created**: 2025-12-30 · **Tags**: bugfix, cp, micro_batch_size, contiguous

## Overview

**Problem**: Channel Loss crashes with `RuntimeError: view size is not compatible with input tensor's size and stride` when `micro_batch_size > 1` with Context Parallelism enabled.

**Root Cause**: In CP-local path, `shift_labels` is created via slicing from full-length `labels` tensor. When `micro_batch_size > 1`, this creates a non-contiguous view (stride includes skipped columns). `.view(-1)` requires contiguous memory layout and fails.

**Impact**: Channel Loss only works with `micro_batch_size=1`, limiting training throughput.

**Solution**: Replace all `.view()` calls with `.reshape()` in `_update_channel_stats()`. `.reshape()` automatically handles non-contiguous tensors by creating a copy when needed.

**Result**: ✅ Channel Loss now works with `micro_batch_size=4` (and any batch size) under CP.

## Design

**Why micro_batch_size=1 worked**:
- PyTorch relaxes contiguous requirements for size=1 dimensions
- stride doesn't matter when there's only one row
- `.view(-1)` succeeds even with non-contiguous layout

**Why micro_batch_size=4 failed**:
- `labels` shape: `(4, 2048)`, stride: `(2048, 1)`
- CP-local slice: `labels[:, label_start:slice_end]` creates `shift_labels` shape `(4, s2)`
- stride remains `(2048, 1)` ← not contiguous `(s2, 1)` due to "skipped columns" in memory
- `.view(-1)` requires contiguous memory, raises RuntimeError

**Solution: .view() → .reshape()**:
- `.reshape()` is zero-copy when tensor is contiguous (same as `.view()`)
- `.reshape()` auto-creates contiguous copy when needed
- For observer-only channel loss, copy overhead is acceptable
- Recommended by PyTorch error message itself

**Alternative considered**:
- Add `.contiguous()` before `.view()`: Also works, but `.reshape()` is more idiomatic per PyTorch docs

## Plan

- [x] Identify all `.view()` calls in `_update_channel_stats()`
- [x] Replace with `.reshape()` (5 locations):
  - Line 306: `shift_logits.view()` → `shift_logits.reshape()`
  - Line 307: `shift_labels.view()` → `shift_labels.reshape()`
  - Line 311: `shift_labels.view()` → `shift_labels.reshape()`
  - Line 322: `per_token_loss.view()` → `per_token_loss.reshape()`
  - Line 323: `valid_mask.view()` → `valid_mask.reshape()`
- [x] Test with `micro_batch_size=4`, `CP=2`
- [x] Update documentation

## Test

- [x] Training runs without crashes (CP=2, TP=1, FSDP=2, micro_batch_size=4)
- [x] No RuntimeError about view/contiguous/stride
- [x] Per-channel metrics appear correctly in logs
- [x] CP detection working: `is_cp_local=True`
- [x] Both channels tracked successfully
- [x] Multiple steps completed (validated 49 steps)

**Validation Log Evidence**:
```
Step 35-49: All successful with micro_batch_size=4
- CP detection: logits_seq_len=1024, label_seq_len=2048, is_cp_local=True
- Per-channel stats: loss=cell_type_identification, loss=cell_type_identification_from_topk_genes
- No errors or warnings
```

## Notes

### Files Modified

**src/axolotl/integrations/channel_loss/compute_loss_patch.py**:
- Lines 306-307: CrossEntropyLoss input flattening
- Line 311: Valid mask creation
- Lines 322-323: Standard mode (non-packing) batch reshaping

### This is NOT a CP framework limitation

- Axolotl CP allows `micro_batch_size > 1` when `sample_packing=false` (uses batch_ring mode)
- Validator only forces `micro_batch_size=1` when `sample_packing=true` (varlen ring-flash-attn requirement)
- This bug was specific to Channel Loss implementation, not CP itself

### Performance Impact

- `.reshape()` is zero-copy when tensor is contiguous (most cases)
- Small copy overhead when non-contiguous (CP-local path with batch>1)
- Acceptable for observer-only statistics (no gradient impact)
- No measurable performance regression observed

### Related Issues

- Discovered after completing spec 008 (CP Statistics fix)
- User reported when testing higher batch sizes
- Error stack: `compute_loss_patch.py:307` in `_update_channel_stats()`
