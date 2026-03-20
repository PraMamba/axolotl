---
status: complete
created: '2025-12-30'
tags:
  - bugfix
  - cp
  - statistics
  - critical
priority: high
created_at: '2025-12-30T03:04:06.544Z'
updated_at: '2025-12-30T05:38:12.294Z'
transitions:
  - status: in-progress
    at: '2025-12-30T03:05:43.861Z'
  - status: complete
    at: '2025-12-30T05:38:12.294Z'
completed_at: '2025-12-30T05:38:12.294Z'
completed: '2025-12-30'
---

# CP Statistics Segment Boundary Fix

> **Status**: ✅ Complete · **Priority**: High · **Created**: 2025-12-30 · **Tags**: bugfix, cp, statistics, critical

## Overview

**Problem**: Per-channel loss statistics not recording with CP > 1 despite successful training.

**Root Causes Identified**:
1. ✅ Misunderstanding of Axolotl CP implementation (inputs remain full-length) - Already correct in worktree
2. ✅ CP output gathering behavior (SFT defaults to `gather_outputs=False`) - Already handled correctly
3. ✅ Token-space vs loss-space boundary mismatch (causal shift off-by-1) - Already mapped correctly
4. ✅ **FIXED**: Callback collective operation deadlock (early returns on empty stats)

**Impact**: ✅ **RESOLVED** - Channel Loss now fully functional with CP=2, training validated successfully.

**Resolution**: Based on comprehensive Axolotl CP source code analysis, fixed critical callback bug while confirming existing worktree implementation correctly handles CP integration.

## Design

**Final Solution**: Fixed callback collective operation alignment while leveraging existing CP-aware implementation.

**Critical Understanding** (from Axolotl CP source code):

Axolotl's CP implementation design:
```
Pre-hook (sequence_parallel.py:254/260):
- Slices kwargs.copy(), NOT original inputs
- Result: inputs dict retains FULL-LENGTH tensors

Outputs (sequence_parallel.py:154):
- gather_outputs = cfg.rl is RLType.GRPO (False for SFT)
- Result: outputs.logits remain CP-LOCAL when gather_outputs=False
```

**Existing Worktree Implementation**:
- ✅ Uses full-length labels/attention_mask/position_ids (no gather needed)
- ✅ Detects CP-local vs gathered logits via sequence length comparison
- ✅ Computes CP-local statistics via offset slicing into full labels
- ✅ Maps token-space boundaries to loss-space via `cu_seqlens_token - 1`

**BUG 4 Fix**:
- **File**: `src/axolotl/integrations/channel_loss/callback.py`
- **Location**: Lines 87-98 (on_log), Lines 131-140 (on_evaluate)
- **Change**: Removed early returns when stats empty to ensure ALL ranks participate in collective operations (all_gather_object, all_reduce)

## Plan

- [x] Analyze Axolotl CP source code implementation (sequence_parallel.py)
- [x] Identify 4 root causes through systematic analysis
- [x] Verify BUGs 1-3 already correctly handled in worktree
- [x] Fix BUG 4: Remove early returns in callback.py
- [x] Run training with CP=2 and verify statistics appear
- [x] Document complete resolution in DEBUG_SESSION_20251230.md
- [x] Update spec status to complete

## Test

- [x] Training runs without crashes (CP=2, TP=1, FSDP=2)
- [x] Per-channel metrics appear in logs: `{'loss': 3.96, 'loss=cell_type_identification': 4.05, ...}`
- [x] Callback messages logged: "Channel Loss: Tracking new channel 'loss=...'"
- [x] CP detection working: `is_cp_local=True, cp_rank=0, cp_size=2`
- [x] Both channels tracked: cell_type_identification, cell_type_identification_from_topk_genes
- [x] No deadlocks or hangs (1000+ steps validated)
- [x] No performance regression

## Notes

### 📚 Complete Historical Context

**Master Navigation**: See [../MASTER_INDEX.md](../MASTER_INDEX.md) for unified documentation hub

**Full Timeline**: See [FULL_TIMELINE.md](./FULL_TIMELINE.md) for complete retrospective (all 5 development stages)

**Historical Documentation**: All detailed docs in [../007-channel-loss-compatibility/](../007-channel-loss-compatibility/)

### Essential Reading (Priority Order)

1. **[FULL_TIMELINE.md](./FULL_TIMELINE.md)** - Complete retrospective
2. **[../007/CP_STATISTICS_BUG.md](../007-channel-loss-compatibility/CP_STATISTICS_BUG.md)** - Detailed bug report
3. **[../007/DEBUG_SESSION_20251229.md](../007-channel-loss-compatibility/DEBUG_SESSION_20251229.md)** - Investigation (35 min)
4. **[../007/CP_IMPLEMENTATION_SUMMARY.md](../007-channel-loss-compatibility/CP_IMPLEMENTATION_SUMMARY.md)** - Implementation details
5. **[../007/CP_NATIVE_SOLUTION_DESIGN.md](../007-channel-loss-compatibility/CP_NATIVE_SOLUTION_DESIGN.md)** - Design rationale
6. **[../007/LEAN_SPEC.md](../007-channel-loss-compatibility/LEAN_SPEC.md)** - Quick reference
7. **[../007/COMPATIBILITY_ANALYSIS.md](../007-channel-loss-compatibility/COMPATIBILITY_ANALYSIS.md)** - Compatibility matrix
8. **[../007/SWIFT_COMPATIBILITY_COMPARISON.md](../007-channel-loss-compatibility/SWIFT_COMPATIBILITY_COMPARISON.md)** - ms-swift comparison

### Alternative Solutions Considered

- **Option A**: Clamp cu_seqlens to per_token_loss length (rejected - treats symptom not cause)
- **Option B**: Trim ALL tensors (THIS SPEC) ✅ SELECTED
- **Option C**: Recompute cu_seqlens after shift (rejected - inefficient)

### Key Insights

**Root Cause**: Partial trimming causes silent failures
- Labels: Trimmed to 2048 ✅
- attention_mask: NOT trimmed (still 4096) ❌ BUG
- position_ids: NOT trimmed (still 4096) ❌ BUG
- Result: `get_segment_boundaries` produces wrong boundaries → segments skipped

**The Fix**: Trim ALL related tensors together (8 lines of code)

**Lessons Learned**:
1. Validation must be end-to-end (not just "doesn't crash")
2. ALL related tensors must have matching sequence lengths
3. Silent failures are dangerous (add warnings)

### Implementation Status

- **Code**: ✅ Fixed (callback.py BUG 4, compute_loss_patch.py already correct)
- **Testing**: ✅ Validated (1000+ training steps with CP=2)
- **Production**: ✅ Ready - all bugs resolved, training successful

### Final Resolution

**See**: [DEBUG_SESSION_20251230.md](./DEBUG_SESSION_20251230.md) - Complete resolution details with:
- All 4 bugs analyzed with source code references
- BUG 4 fix implementation
- Training validation results
- Key technical insights from Axolotl CP framework
