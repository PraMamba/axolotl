# Channel Loss CP Compatibility Documentation Index

**Feature**: Context Parallelism (CP > 1) Support for Channel Loss Plugin
**Status**: 🔧 **Fix In Progress** - See Spec 008 (2025-12-30)
**Implementation**: Axolotl-Native Solution
**Critical Issue**: Statistics not recording (discovered 2025-12-29 22:35)

---

## 🔴 IMPORTANT UPDATE (2025-12-30)

This directory (007) contains **complete historical documentation** of the CP compatibility development from design through bug discovery.

**For Current Status and Active Work**: See [specs/008-cp-statistics-segment-boundary-fix/](../008-cp-statistics-segment-boundary-fix/)

**For Unified Navigation**: See [specs/MASTER_INDEX.md](../MASTER_INDEX.md)

**Complete Timeline**: See [../008/FULL_TIMELINE.md](../008-cp-statistics-segment-boundary-fix/FULL_TIMELINE.md)

---

## Quick Navigation

### 🎯 For New Readers

**Start Here**:
1. **[../MASTER_INDEX.md](../MASTER_INDEX.md)** - Unified navigation hub with current status
2. **[../008/FULL_TIMELINE.md](../008-cp-statistics-segment-boundary-fix/FULL_TIMELINE.md)** - Complete 5-stage retrospective
3. **[LEAN_SPEC.md](LEAN_SPEC.md)** - Quick reference (historical, pre-bug-fix)

### 🔴 Critical Issues

1. **[CP_STATISTICS_BUG.md](CP_STATISTICS_BUG.md)** - **REGRESSION FOUND (2025-12-29 22:35)**: Per-channel loss statistics not being recorded with CP > 1. Shape alignment works but statistics accumulation fails due to segment boundary mismatch. **Must be fixed before production deployment.**

2. **[DEBUG_SESSION_20251229.md](DEBUG_SESSION_20251229.md)** - Debug session summary (22:00-22:35): Root cause analysis, attempted fixes, and recommended solution. Read this for detailed investigation timeline and validation criteria.

### 📚 Detailed Documentation

1. **[COMPATIBILITY_ANALYSIS.md](COMPATIBILITY_ANALYSIS.md)**
   - Compatibility matrix (CP now ✅ COMPATIBLE)
   - Usage guidelines
   - Configuration examples
   - When to use Channel Loss

2. **[CP_IMPLEMENTATION_SUMMARY.md](CP_IMPLEMENTATION_SUMMARY.md)**
   - Full implementation details
   - Production validation results
   - Discovery of SFT mode logits gathering issue
   - Label trimming solution for 1:2 ratio
   - Performance impact analysis

3. **[CP_NATIVE_SOLUTION_DESIGN.md](CP_NATIVE_SOLUTION_DESIGN.md)**
   - Original design proposal
   - Why not port ms-swift GatherLoss
   - Advantages of Axolotl-native approach
   - Decision matrix

4. **[SWIFT_COMPATIBILITY_COMPARISON.md](SWIFT_COMPATIBILITY_COMPARISON.md)**
   - Comparison with ms-swift's approach
   - Architecture differences
   - Implementation trade-offs

---

## Key Results

### Production Validation

⚠️ **Configuration**: Qwen2.5-7B, CP=2, TP=1, FSDP=2, 4 GPUs
⚠️ **Training Steps**: 672-680+ (stable, but statistics not recording)
⚠️ **Date**: 2025-12-29 21:02 (Initial validation), 22:35 (Regression found)
🔴 **Status**: Critical bug - NOT ready for production deployment

### Critical Discoveries

1. **SFT Mode Behavior**: Axolotl's SFT mode (`gather_outputs=False`) does NOT auto-gather logits after model forward, requiring manual gathering in Channel Loss plugin.

2. **Data-Specific Handling**: Some data/model configurations produce logits:labels ratio ≠ 1:1. Solution: Trim labels to match logits length after gathering.

3. **Double Gathering Prevention**: Implemented `cp_already_gathered` flag to prevent redundant gathering operations.

### Files Modified

```
src/axolotl/integrations/channel_loss/
├── __init__.py              (MODIFIED: Removed CP > 1 conflict detection)
├── compute_loss_patch.py    (MODIFIED: Manual gathering + label trimming)
└── utils.py                 (NEW: _get_context_parallel_group helper)
```

**Total**: 3 files, ~150 lines of code added

---

## Implementation Pattern

```python
# High-level flow:

1. Detect CP group: _get_context_parallel_group(trainer)
2. Call original compute_loss with return_outputs=True
3. If CP > 1: Manually gather logits, labels, position_ids, attention_mask
4. Trim labels to match logits if length mismatch
5. Pass to _update_channel_stats with cp_already_gathered=True
6. Skip gathering in _update_channel_stats if already gathered
```

---

## Performance Impact

- **Overhead**: ~1.5-3ms per step
- **Impact**: < 1% of typical 100-1000ms step time
- **Memory**: +24KB per step (negligible)
- **Conclusion**: Acceptable for production use

---

## Quick Start

### Configuration

```yaml
# Enable Context Parallelism + Channel Loss
context_parallel_size: 2
tensor_parallel_size: 1
dp_shard_size: 2

plugins:
  - axolotl.integrations.channel_loss.ChannelLossPlugin
enable_channel_loss: true
channel_loss_field: "task_type"
channel_loss_prefix: "loss="
```

### Verification

Check logs for these indicators:
```
✅ "Channel Loss Plugin: Extracted channels from N datasets"
✅ "CP_SIZE=2" (or your CP size)
✅ "After manual gather: logits.shape=..."
✅ "Tensors already gathered, skipping gather"
✅ Training proceeds without ValueError
```

---

## References

- **Production Config**: `/home/scbjtfy/RVQ-Alpha/configs/axolotl/7b-fsdp2-tp-cp_sft_channel-loss.yaml`
- **Production Logs**: `/data/Mamba/Project/Single_Cell/Training/.../logs/model_training_20251229_20.log`
- **Axolotl CP Docs**:
  - `docs/analysis/context_parallelism_deep_dive.md`
  - `docs/analysis/cp_quick_reference.md`
  - `docs/analysis/cp_source_code_walkthrough.md`

---

## Document History

| Date | Time | Document | Event | Status |
|------|------|----------|-------|--------|
| 2025-12-29 | 15:00 | CP_NATIVE_SOLUTION_DESIGN.md | Design Complete | ✅ |
| 2025-12-29 | 18:00 | CP_IMPLEMENTATION_SUMMARY.md | Implementation Complete | ✅ |
| 2025-12-29 | 21:02 | CP_IMPLEMENTATION_SUMMARY.md | Production Validated | ✅ |
| 2025-12-29 | 21:09 | LEAN_SPEC.md | Created | ✅ |
| 2025-12-29 | 21:10 | COMPATIBILITY_ANALYSIS.md | Updated (CP compatible) | ✅ |
| 2025-12-29 | 21:10 | INDEX.md | Created | ✅ |
| 2025-12-29 | 22:35 | **CP_STATISTICS_BUG.md** | **Regression Found** | 🔴 |
| 2025-12-29 | 22:35 | LEAN_SPEC.md | Status Updated (Bug Found) | ⚠️ |
| 2025-12-29 | 22:36 | INDEX.md | Status Updated (Bug Found) | ⚠️ |
| 2025-12-29 | 22:40 | DEBUG_SESSION_20251229.md | Debug Session Summary Created | 📋 |
| 2025-12-29 | 22:41 | INDEX.md | Added Debug Session Link | 📋 |

---

**Last Updated**: 2025-12-29 22:41
**Maintained By**: Channel Loss Development Team
**Status**: 🔴 Critical Bug - Root Cause Identified, Solution Proposed
