# TODO: CP Statistics Integration

**Priority**: MEDIUM
**Status**: 📋 **PLANNED**
**Effort**: 2-4 hours
**Dependencies**: None (code is ready in worktree)

---

## Objective

Integrate the complete CP-aware statistics implementation from `worktrees/channel-loss` branch into the main repository, enabling per-channel loss monitoring with Context Parallelism (CP > 1).

---

## Current Situation

### What's Working
- ✅ Training with CP > 1 runs successfully (statistics disabled)
- ✅ Complete CP-aware implementation exists in worktree branch
- ✅ Statistics work perfectly with CP = 1

### What's Missing
- ❌ Per-channel statistics don't work with CP > 1 (disabled in main repo)
- ❌ Main repo missing `utils.py` module with CP helper functions
- ❌ Main repo has outdated `compute_loss_patch.py`

### Why It's Disabled
See [Spec 009](./009-cp-statistics-temporary-disable/README.md) for complete context.

**TL;DR**: Main repo lacks dependencies needed for CP-aware statistics. Workaround is to disable all statistics when CP > 1.

---

## Task Breakdown

### Phase 1: Copy Code from Worktree (30 minutes)

**Source Location**: `/home/scbjtfy/axolotl/worktrees/channel-loss/src/axolotl/integrations/channel_loss/`
**Destination**: `/home/scbjtfy/axolotl/src/axolotl/integrations/channel_loss/`

**Files to Copy**:

1. ✅ **`utils.py`** (NEW file):
   ```bash
   cp worktrees/channel-loss/src/axolotl/integrations/channel_loss/utils.py \
      src/axolotl/integrations/channel_loss/utils.py
   ```
   - Contains `_get_context_parallel_group()` function
   - Provides CP detection and helper utilities

2. ✅ **`compute_loss_patch.py`** (REPLACE):
   ```bash
   cp worktrees/channel-loss/src/axolotl/integrations/channel_loss/compute_loss_patch.py \
      src/axolotl/integrations/channel_loss/compute_loss_patch.py
   ```
   - Complete CP-aware statistics implementation
   - Handles CP-local vs gathered logits
   - Correct shape handling for all CP sizes

3. ⚠️ **`segment.py`** (CHECK first):
   ```bash
   diff worktrees/channel-loss/src/axolotl/integrations/channel_loss/segment.py \
        src/axolotl/integrations/channel_loss/segment.py
   ```
   - Only copy if there are meaningful differences
   - Unlikely to have CP-specific changes

**Verification**:
```bash
# Check that all required functions are present
grep -n "_get_context_parallel_group" src/axolotl/integrations/channel_loss/compute_loss_patch.py
grep -n "def _get_context_parallel_group" src/axolotl/integrations/channel_loss/utils.py
```

---

### Phase 2: Clear Bytecode Cache (5 minutes)

**Why**: Python may have cached the old disabled version.

**Command**:
```bash
find /home/scbjtfy/axolotl/src/axolotl/integrations/channel_loss \
     -name "*.pyc" -delete

find /home/scbjtfy/axolotl/src/axolotl/integrations/channel_loss \
     -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "Bytecode cache cleared"
```

---

### Phase 3: Test with CP=1 (Baseline) (30 minutes)

**Purpose**: Verify no regressions in non-CP mode.

**Test Configuration**:
```yaml
# Test config (CP=1)
context_parallel_size: 1  # No parallelism
sample_packing: true      # Test packing mode
sequence_len: 2048
```

**Run Training**:
```bash
accelerate launch -m axolotl.cli.train /path/to/test_config.yaml
```

**Verify**:
- ✅ Training starts without errors
- ✅ Per-channel statistics appear in logs
- ✅ Wandb shows channel metrics (loss_dataset_A, etc.)
- ✅ Statistics values are reasonable (not all zeros/NaN)
- ✅ At least 50 steps complete successfully

**Success Criteria**:
- No errors in logs
- Channel statistics match previous baseline (if available)
- All channel keys have non-zero counts

---

### Phase 4: Test with CP=2 (Primary Target) (60 minutes)

**Purpose**: Verify CP statistics work correctly.

**Test Configuration**:
```yaml
# Test config (CP=2)
context_parallel_size: 2  # Enable CP
sample_packing: true      # Test packing mode
sequence_len: 2048
num_processes: 4          # Need at least 2 for CP=2
```

**GPU Setup**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  # Need 4 GPUs (2x2 = DP×CP)
```

**Run Training**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --main_process_port 0 --num_processes=4 \
  -m axolotl.cli.train /path/to/test_config_cp2.yaml
```

**Verify**:
- ✅ Training starts without errors
- ✅ No shape mismatch errors
- ✅ No NCCL deadlocks or hangs
- ✅ Per-channel statistics appear in logs
- ✅ Wandb shows channel metrics
- ✅ Statistics values are reasonable
- ✅ At least 100 steps complete successfully

**Debug Logging** (if issues):
```python
# Check logs for CP detection messages
[RANK 0] CP detection: logits_seq_len=1024, label_seq_len=2048,
         expected_chunk_len=1024, is_cp_local=True, cp_rank=0, cp_size=2
```

**Success Criteria**:
- No errors in training logs
- Statistics are non-zero and finite
- Statistics are consistent across steps (no sudden jumps)

---

### Phase 5: Validation Testing (60 minutes)

**Purpose**: Ensure statistics are actually correct, not just "working".

**Test Approach**: Compare statistics with CP=1 baseline.

**Method**:
1. Run same dataset with CP=1 for 100 steps → save statistics
2. Run same dataset with CP=2 for 100 steps → save statistics
3. Compare channel statistics (should be very close, <5% difference)

**Comparison Script**:
```python
# Compare channel statistics from CP=1 vs CP=2
import json

# Load stats from Wandb or saved logs
stats_cp1 = {...}  # loss_dataset_A: 2.5, etc.
stats_cp2 = {...}

for key in stats_cp1:
    cp1_val = stats_cp1[key]
    cp2_val = stats_cp2.get(key, 0)
    diff_pct = abs(cp1_val - cp2_val) / cp1_val * 100
    print(f"{key}: CP=1: {cp1_val:.4f}, CP=2: {cp2_val:.4f}, diff: {diff_pct:.2f}%")
    assert diff_pct < 5.0, f"Statistics differ by {diff_pct}% for {key}"
```

**Success Criteria**:
- All channel statistics match within 5% between CP=1 and CP=2
- No channels have zero counts in CP=2 that had counts in CP=1
- Same number of unique channels in both runs

---

### Phase 6: Extended Testing (Optional, 30 minutes)

**Purpose**: Test edge cases and higher CP values.

**Test Matrix**:
| CP Size | Packing | Seq Len | GPU Count | Expected Result |
|---------|---------|---------|-----------|-----------------|
| 1       | False   | 2048    | 2         | ✅ Baseline     |
| 1       | True    | 2048    | 2         | ✅ Baseline     |
| 2       | False   | 2048    | 4         | ✅ Should work  |
| 2       | True    | 2048    | 4         | ✅ Should work  |
| 4       | True    | 4096    | 8         | ⚠️ Stretch goal |
| 8       | True    | 8192    | 16        | ⚠️ Not priority |

**Focus**: CP=2 with packing (most common use case).

---

## Known Issues to Watch For

### Issue #1: Shape Mismatch (should be fixed)
**Symptom**:
```
ValueError: Expected input batch_size (1023) to match target batch_size (2047)
```
**Cause**: Incorrect CP-local shape handling
**Fix**: Worktree code has correct implementation

### Issue #2: Double-Gathering (should be fixed)
**Symptom**: All statistics are zero, `valid_loss.numel()=0` in logs
**Cause**: Labels being gathered when already full
**Fix**: Worktree code checks tensor size before gathering

### Issue #3: NCCL Deadlock (watch for this)
**Symptom**: Training hangs during statistics computation
**Cause**: Improper synchronization across CP ranks
**Fix**: Ensure no unexpected collectives in statistics code
**Mitigation**: If this occurs, add timeout to detect it:
```python
import signal
signal.alarm(30)  # Timeout after 30s
# ... statistics computation ...
signal.alarm(0)
```

### Issue #4: NaN/Inf Statistics (edge case)
**Symptom**: Channel statistics become NaN or Inf
**Cause**: Numerical instability in loss computation
**Fix**: Already handled in worktree code with `torch.isfinite()` filter

---

## Rollback Plan

If integration causes issues:

**Quick Rollback** (revert to disabled state):
```bash
# Restore the early return
git checkout src/axolotl/integrations/channel_loss/compute_loss_patch.py

# Or manually re-add at line 192:
# return  # TEMP: Disable all statistics
```

**Full Rollback** (remove new code):
```bash
# Restore old versions
git checkout HEAD -- src/axolotl/integrations/channel_loss/
rm -f src/axolotl/integrations/channel_loss/utils.py
```

---

## Success Metrics

### Minimum Viable Integration
- ✅ Training works with CP=2 (no crashes or hangs)
- ✅ Statistics appear in Wandb for all channels
- ✅ Statistics values are non-zero and finite
- ✅ No regressions in CP=1 mode

### Full Success
- ✅ Statistics match CP=1 baseline within 5%
- ✅ Works with packing mode
- ✅ Works with CP=4 (if tested)
- ✅ Comprehensive logging for debugging

---

## Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1. Copy code | 30 min | 30 min |
| 2. Clear cache | 5 min | 35 min |
| 3. Test CP=1 | 30 min | 65 min |
| 4. Test CP=2 | 60 min | 125 min |
| 5. Validation | 60 min | 185 min |
| 6. Extended (optional) | 30 min | 215 min |

**Total**: ~2-4 hours depending on issues encountered

---

## Post-Integration Tasks

1. ✅ Update Spec 009 status to "RESOLVED"
2. ✅ Document CP statistics behavior in main README
3. ✅ Add CP statistics section to troubleshooting guide
4. ✅ Create tests for CP statistics (if test suite exists)
5. ✅ Remove "TEMP" comments from code

---

## References

- [Spec 009: CP Statistics Temporary Disable](./009-cp-statistics-temporary-disable/README.md)
- [DEBUG_SESSION_20251230.md](./008-cp-statistics-segment-boundary-fix/DEBUG_SESSION_20251230.md)
- Worktree Implementation: `/home/scbjtfy/axolotl/worktrees/channel-loss/src/axolotl/integrations/channel_loss/`

---

**Created**: 2025-12-30
**Status**: 📋 Ready to implement
**Next Step**: Phase 1 - Copy code from worktree
