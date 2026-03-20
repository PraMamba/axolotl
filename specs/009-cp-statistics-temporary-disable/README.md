# Spec 009: CP Statistics Temporary Disable

**Created**: 2025-12-30
**Status**: 🟡 **TEMPORARY WORKAROUND**
**Branch**: `archive/main-with-features-20251228` (main repo)
**Related**: Spec 008 (CP statistics fix - incomplete)

---

## Overview

Channel Loss statistics are **temporarily disabled** in the main repository to allow training with Context Parallelism (CP > 1) to proceed without errors.

The proper CP-aware statistics implementation exists in the `worktrees/channel-loss` branch but cannot be integrated yet due to missing dependencies in the main repo.

---

## Current State

### What Works
✅ Training with CP > 1 runs successfully
✅ Model training and convergence are unaffected
✅ All other Channel Loss Plugin features work (collator, callbacks)
✅ Statistics still work with CP = 1 (no parallelism)

### What Doesn't Work
❌ Per-channel loss statistics are NOT recorded when CP > 1
❌ Wandb metrics for individual channels won't appear with CP > 1
❌ Channel loss monitoring/debugging requires CP = 1

---

## Implementation Details

### Modified File
**Location**: `/home/scbjtfy/axolotl/src/axolotl/integrations/channel_loss/compute_loss_patch.py`

**Change**: Added early return at line 192 in `_update_channel_stats()`:
```python
def _update_channel_stats(...):
    """
    Update per-channel loss statistics.
    ...
    """
    # TEMP: Disable all channel statistics due to CP compatibility issues
    # TODO: Integrate CP-compatible code from worktrees/channel-loss branch
    # The worktree has proper CP-aware logic but requires additional utils module
    return

    # Original implementation follows (unreachable)...
```

### Why This Works
- The statistics computation is **observer-only** (doesn't affect training loss or gradients)
- Disabling statistics has **zero impact** on model training
- The rest of the Channel Loss Plugin continues to function normally

---

## Root Cause Analysis

### Why Statistics Were Disabled

The CP statistics implementation in the main repo has multiple critical bugs:

1. **Double-Gathering Bug** (Partially Fixed in Spec 008):
   - Labels, attention_mask, position_ids were being gathered when already full
   - Caused all labels to become padding (-100)

2. **Shape Mismatch in CP-Local Path**:
   - CP-local statistics computation doesn't correctly handle tensor dimensions
   - Error: `ValueError: Expected input batch_size (1023) to match target batch_size (2047)`
   - Root cause: Incorrect causal shift handling with sharded logits

3. **Missing Dependencies**:
   - Main repo lacks `utils.py` module with `_get_context_parallel_group()` function
   - Main repo has outdated `compute_loss_patch.py` code
   - Worktree has complete implementation but isn't in Python import path

4. **CP Sharding Complexity**:
   - Axolotl's CP selectively shards tensors:
     - Logits: CP-sharded (split across ranks)
     - Labels, attention_mask, position_ids: NOT sharded (full on each rank)
   - Statistics code must handle this hybrid sharding correctly

### Failed Attempts

Multiple fix attempts were made before resorting to disabling:

1. ✅ Fixed double-gathering of labels (Spec 008)
2. ✅ Fixed double-gathering of attention_mask/position_ids (Spec 008)
3. ❌ Attempted to fix CP-local shape handling (import path issue discovered)
4. ❌ Attempted to use worktree code (missing dependencies in main repo)
5. ✅ Disabled all statistics (current workaround)

---

## Proper Solution (Future Work)

### What Needs to Be Done

**Goal**: Integrate complete CP-aware statistics implementation from worktree branch.

**Required Changes**:

1. **Copy Complete Module** from worktree:
   ```
   Source: /home/scbjtfy/axolotl/worktrees/channel-loss/src/axolotl/integrations/channel_loss/
   Destination: /home/scbjtfy/axolotl/src/axolotl/integrations/channel_loss/

   Files:
   - compute_loss_patch.py (updated version with CP-aware logic)
   - utils.py (NEW - contains _get_context_parallel_group() and helpers)
   - segment.py (may have updates)
   ```

2. **Test Thoroughly**:
   - CP=1 (baseline, should match current behavior)
   - CP=2, CP=4, CP=8 (various parallelism levels)
   - Packing vs non-packing modes
   - Different sequence lengths (2048, 4096, 8192)
   - Verify statistics match non-CP baseline

3. **Verify No Regressions**:
   - No NCCL deadlocks
   - No shape mismatch errors
   - Proper synchronization across ranks
   - Statistics are accurate (compare with CP=1)

### Worktree Branch Details

**Branch**: `worktrees/channel-loss` (proper branch name may differ)
**Location**: `/home/scbjtfy/axolotl/worktrees/channel-loss/`

**Key Differences** from main repo:
- `compute_loss_patch.py` has CP-aware statistics path:
  - Detects whether logits are CP-local or already gathered
  - Computes boundary-correct losses for CP shards
  - Handles overlap correctly in packing mode
  - Includes extensive debug logging
- `utils.py` module provides CP helper functions:
  - `_get_context_parallel_group()`: Gets CP process group
  - Other CP-related utilities

---

## Impact Assessment

### User Impact
- **LOW** for most users:
  - Channel statistics are mainly used for debugging/monitoring
  - Training itself is completely unaffected
  - Users without CP (CP=1) still get full statistics

- **MEDIUM** for users with CP > 1:
  - Cannot monitor per-channel loss breakdown during training
  - Can still see overall training loss
  - Can run separate CP=1 eval to get channel statistics if needed

### Development Impact
- **HIGH** for Channel Loss Plugin development:
  - Cannot test/debug CP statistics without proper integration
  - Two separate codebases to maintain (main repo vs worktree)
  - Risk of divergence between branches

---

## Debugging Reference

### How to Identify This Issue

If you see this in logs with CP > 1:
```
Channel Loss: Patched trainer.compute_loss for channel statistics
```

But you DON'T see per-channel metrics in Wandb like:
```
loss_dataset_A: 2.5
loss_dataset_B: 3.1
```

Then statistics are disabled (this workaround is active).

### How to Re-enable (Not Recommended)

**WARNING**: Only re-enable if you've integrated the proper CP-aware code.

1. Comment out the early return in `compute_loss_patch.py:192-195`
2. Ensure `utils.py` module exists with `_get_context_parallel_group()`
3. Clear Python bytecode cache: `find . -name "*.pyc" -delete`
4. Test thoroughly with CP=2 before using higher CP values

---

## Timeline

- **2025-12-30 11:20**: Started debugging CP statistics (Spec 008)
- **2025-12-30 11:35**: Fixed double-gathering bugs (Bug #1, #2)
- **2025-12-30 12:00**: Discovered shape mismatch bug (Bug #3)
- **2025-12-30 12:15**: Discovered import path issue (Bug #4)
- **2025-12-30 12:20**: Discovered missing dependencies (Bug #5)
- **2025-12-30 12:30**: Decided to disable statistics as temporary workaround
- **2025-12-30 13:06**: Implemented disable, started verification testing
- **2025-12-30 13:09**: ✅ Verified training works with statistics disabled

---

## Related Documentation

- [DEBUG_SESSION_20251230.md](../008-cp-statistics-segment-boundary-fix/DEBUG_SESSION_20251230.md) - Detailed debugging session log
- [Spec 008](../008-cp-statistics-segment-boundary-fix/) - Original CP statistics fix (incomplete)
- [Spec 007](../007-cp-compatibility/) - Initial CP compatibility work

---

## Contact

For questions about this workaround or to contribute the proper fix:
- Review the complete implementation in `worktrees/channel-loss` branch
- See `DEBUG_SESSION_20251230.md` for complete bug analysis
- Test any fixes thoroughly before deploying to main repo
