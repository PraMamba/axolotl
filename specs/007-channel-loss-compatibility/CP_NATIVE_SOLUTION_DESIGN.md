# Axolotl-Native CP Compatibility Solution for Channel Loss

**Date**: 2025-12-29
**Status**: Design Proposal
**Target**: Enable Context Parallelism (CP > 1) support in Channel Loss without porting ms-swift's GatherLoss

---

## Executive Summary

**Problem**: Channel Loss is incompatible with Context Parallelism (CP > 1) due to shape mismatch between聚合后的 logits 和 切分后的 labels.

**Root Cause**:
```python
# In Axolotl's CP implementation:
# - logits: Gathered by AllGatherWithGrad → Shape: (batch, full_seq_len, vocab)
# - labels: NOT gathered → Shape: (batch, local_seq_len)

# Channel Loss computes:
shift_logits = logits[..., :-1, :].contiguous()  # Shape: (1, 4095, vocab)
shift_labels = labels[..., 1:].contiguous()       # Shape: (1, 2046)  ← MISMATCH!
```

**Proposed Solution**: Axolotl-native approach leveraging existing CP infrastructure, **without** porting ms-swift's GatherLoss.

---

## Problem Analysis

### Current Axolotl CP Workflow

```python
# File: src/axolotl/utils/ctx_managers/sequence_parallel.py

# 1. Pre-Forward Hook: Split sequence
def apply_sequence_parallelism(batch, local_rank, local_world_size, ...):
    # Split input_ids, labels, attention_mask along sequence dimension
    for key in batch:
        if batch[key].size(1) == total_seq_len:
            batch[key] = batch[key].chunk(local_world_size, dim=1)[local_rank].contiguous()
    return batch  # ← labels is now split!

# 2. Forward Pass: Model computes on local sequence
output = model(**batch)  # output.logits has local shape

# 3. Post-Forward Hook: Gather outputs
def sequence_parallel_post_hook(_, __, output):
    output = _gather_outputs(output)  # ← AllGatherWithGrad on logits
    return output

def _gather_outputs(output):
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dim() > 1:
            output[key] = AllGatherWithGrad.apply(value, process_group)  # ← Gather logits
    return output
```

**Key Observation**: `labels` is split in step 1 but **NOT** gathered in step 3!

### Channel Loss Computation

```python
# File: src/axolotl/integrations/channel_loss/compute_loss_patch.py:195-203

def track_channel_loss_statistics(trainer, logits, labels, channels, ...):
    with torch.no_grad():
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()  # ← Gathered (full seq)
        shift_labels = labels[..., 1:].contiguous()       # ← Split (local seq)

        # Compute per-token CE
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # ← full_seq_len - 1
            shift_labels.view(-1),                          # ← local_seq_len - 1
        ).detach()  # ❌ SHAPE MISMATCH!
```

**Error**:
```
ValueError: Expected input batch_size (2046) to match target batch_size (4095).
```

---

## Why Not Port ms-swift's GatherLoss?

### ms-swift's Approach

ms-swift solves this by:
1. Computing per-token loss **on each rank independently** (before gather)
2. Using custom `GatherLoss` autograd function to gather losses from all ranks
3. Maintaining position_ids alignment across ranks

```python
# ms-swift: swift/trainers/utils.py:59-91
def per_token_loss_func_sp(outputs, labels, ...):
    # Compute loss on local sequence
    loss = CrossEntropyLoss(reduction='none')(logits, labels)  # ← Local

    # Gather loss from all ranks
    loss, labels = GatherLoss.apply(loss.reshape(batch_size, -1), labels, 1, position_ids)
    return loss
```

### Why This Doesn't Fit Axolotl

1. **Architecture Mismatch**:
   - ms-swift computes loss **before** logits gather
   - Axolotl gathers logits **before** compute_loss() is called
   - Channel Loss is a **post-hook** that runs after compute_loss()

2. **Complexity**:
   - Requires restructuring Axolotl's CP data flow
   - Needs custom autograd functions (GatherLoss, ChunkedCrossEntropyLoss)
   - Requires sequence_parallel module infrastructure

3. **Maintenance Burden**:
   - Duplicates Axolotl's existing AllGatherWithGrad logic
   - Adds dependency on ms-swift's sequence parallel utilities

**Conclusion**: Porting GatherLoss would require **major refactoring** of Axolotl's CP architecture. A native solution is preferred.

---

## Proposed Axolotl-Native Solution

### Core Idea

**Reuse Axolotl's existing `AllGatherWithGrad`** to gather labels in Channel Loss compute_loss_patch.

### Design Principles

1. **Minimal Changes**: Only modify Channel Loss plugin, not core CP infrastructure
2. **Reuse Existing Code**: Leverage `AllGatherWithGrad` already used for logits
3. **No Gradient Impact**: Channel Loss statistics use `torch.no_grad()`, so gradient flow is unaffected
4. **Backward Compatible**: Works with both CP and non-CP configurations

---

## Implementation Plan

### Step 1: Detect Context Parallelism in Channel Loss

```python
# File: src/axolotl/integrations/channel_loss/compute_loss_patch.py

def track_channel_loss_statistics(trainer, logits, labels, channels, ...):
    with torch.no_grad():
        # Detect Context Parallelism
        cp_group = _get_context_parallel_group(trainer)
        cp_size = dist.get_world_size(cp_group) if cp_group else 1

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # If CP is enabled, gather labels to match gathered logits
        if cp_size > 1:
            # Import AllGatherWithGrad from Axolotl's CP module
            from axolotl.utils.ctx_managers.sequence_parallel import AllGatherWithGrad

            # Gather labels across CP ranks (no_grad context)
            shift_labels = AllGatherWithGrad.apply(shift_labels, cp_group)

        # Now shift_logits and shift_labels have matching shapes!
        # Compute per-token CE as usual
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).detach()

        # Rest of channel statistics tracking...
```

### Step 2: Helper Function to Get CP Group

```python
# File: src/axolotl/integrations/channel_loss/utils.py

import torch.distributed as dist
from typing import Optional

def _get_context_parallel_group(trainer) -> Optional[dist.ProcessGroup]:
    """
    Extract Context Parallel process group from trainer.

    Returns:
        dist.ProcessGroup if CP is enabled, None otherwise
    """
    # Check if trainer has device_mesh (modern Axolotl)
    if hasattr(trainer, 'device_mesh') and trainer.device_mesh is not None:
        try:
            # Get CP sub-mesh
            cp_mesh = trainer.device_mesh.get("cp", None)
            if cp_mesh is not None:
                return cp_mesh.get_group()
        except (KeyError, AttributeError):
            pass

    # Check if ring_attn_group is set (fallback)
    try:
        from axolotl.monkeypatch.ring_attn import get_ring_attn_group
        ring_group = get_ring_attn_group()
        if ring_group is not None:
            return ring_group
    except (ImportError, AttributeError):
        pass

    # No CP detected
    return None
```

### Step 3: Handle Position IDs Alignment (Optional Enhancement)

For sample packing with CP, we need to handle position_ids:

```python
def track_channel_loss_statistics(trainer, logits, labels, channels, position_ids, ...):
    with torch.no_grad():
        cp_group = _get_context_parallel_group(trainer)
        cp_size = dist.get_world_size(cp_group) if cp_group else 1

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # If CP is enabled, gather labels
        if cp_size > 1:
            from axolotl.utils.ctx_managers.sequence_parallel import AllGatherWithGrad

            shift_labels = AllGatherWithGrad.apply(shift_labels, cp_group)

            # Also gather position_ids if present (for segment boundary detection)
            if position_ids is not None:
                position_ids = AllGatherWithGrad.apply(position_ids, cp_group)

        # Compute per-token loss with aligned shapes
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).detach()

        # Get segment boundaries (now with gathered position_ids if CP)
        cu_seqlens = get_segment_boundaries(
            attention_mask=attention_mask,  # ← May also need gathering
            position_ids=position_ids,       # ← Already gathered above
            labels=shift_labels,             # ← Already gathered
            mode=segment_mode,
        )

        # Rest of channel statistics tracking...
```

---

## Key Advantages Over ms-swift GatherLoss

| Aspect | ms-swift GatherLoss | Axolotl-Native Solution |
|--------|---------------------|-------------------------|
| **Code Changes** | Requires new autograd functions, sequence_parallel infrastructure | Only modifies Channel Loss plugin |
| **Complexity** | High - custom backward pass, position_ids alignment | Low - reuses existing AllGatherWithGrad |
| **Integration** | Needs refactoring of compute_loss flow | Drop-in fix in existing post-hook |
| **Maintenance** | Separate codebase to sync with ms-swift | Uses Axolotl's maintained CP utilities |
| **Gradient Safety** | Custom backward logic | Runs in torch.no_grad() (no gradient concern) |
| **Position IDs** | Complex alignment logic | Reuses Axolotl's existing alignment |

---

## Implementation Details

### File Changes

```
Modified:
  src/axolotl/integrations/channel_loss/compute_loss_patch.py
  src/axolotl/integrations/channel_loss/utils.py
  src/axolotl/integrations/channel_loss/__init__.py (remove CP conflict detection)

Added:
  None (uses existing Axolotl utilities)
```

### Backward Compatibility

```python
# The solution automatically handles both CP and non-CP cases:

# Case 1: CP disabled (cp_size = 1)
cp_group = _get_context_parallel_group(trainer)  # → None
cp_size = dist.get_world_size(cp_group) if cp_group else 1  # → 1

if cp_size > 1:  # ← False, skip gathering
    shift_labels = AllGatherWithGrad.apply(shift_labels, cp_group)

# Labels remain as-is, no overhead

# Case 2: CP enabled (cp_size > 1)
cp_group = _get_context_parallel_group(trainer)  # → ProcessGroup
cp_size = dist.get_world_size(cp_group)  # → 2, 4, 8, etc.

if cp_size > 1:  # ← True, perform gathering
    shift_labels = AllGatherWithGrad.apply(shift_labels, cp_group)

# Labels gathered, matching logits shape
```

---

## Testing Plan

### Unit Tests

```python
# File: tests/integrations/test_channel_loss_cp.py

def test_channel_loss_with_cp():
    """Test Channel Loss with CP=2"""
    # Setup CP environment
    dist.init_process_group(...)
    cp_group = dist.new_group([0, 1])

    # Simulate split labels (CP rank 0)
    labels_rank0 = torch.tensor([[1, 2, 3, 4]])  # Local: 4 tokens

    # Simulate gathered logits
    logits_gathered = torch.randn(1, 8, vocab_size)  # Full: 8 tokens

    # Run Channel Loss computation
    track_channel_loss_statistics(
        trainer, logits_gathered, labels_rank0, channels, ...
    )

    # Should NOT raise ValueError
```

### Integration Tests

```yaml
# Test config: configs/axolotl/test_cp2_channel_loss.yaml
context_parallel_size: 2
tensor_parallel_size: 2
dp_shard_size: 1

plugins:
  - axolotl.integrations.channel_loss.ChannelLossPlugin
enable_channel_loss: true
channel_loss_field: "task_type"

sequence_len: 4096
micro_batch_size: 1
max_steps: 20
```

**Expected Result**: Training completes successfully, channel metrics logged.

---

## Performance Considerations

### Memory Overhead

```python
# Additional all-gather operations in Channel Loss:
# - shift_labels: (batch_size, local_seq_len) → (batch_size, full_seq_len)
# - position_ids (if present): Same as above

# Example: CP=4, seq_len=16384, batch=1
# - Local labels: 1 × 4096 × 2 bytes (int16) = 8 KB
# - Gathered labels: 1 × 16384 × 2 bytes = 32 KB
# - Overhead: 24 KB (negligible compared to model activations)
```

**Impact**: Negligible (< 0.1% of total memory)

### Compute Overhead

```python
# Additional all-gather calls: 2-3 per training step
# - AllGatherWithGrad on labels: ~0.1-1ms (depends on CP size and network)
# - AllGatherWithGrad on position_ids: ~0.1-1ms

# Total overhead: ~0.5-3ms per step
# Typical training step: 100-1000ms
# Overhead: < 1%
```

**Impact**: Negligible (< 1% of step time)

---

## Comparison: Native vs GatherLoss

### ms-swift GatherLoss Approach

**Advantages**:
- ✅ Computes loss before gather (more efficient)
- ✅ Proven in production

**Disadvantages**:
- ❌ Requires refactoring Axolotl's CP flow
- ❌ Custom autograd functions to maintain
- ❌ Complexity in position_ids alignment

### Axolotl-Native Approach

**Advantages**:
- ✅ Minimal code changes (only Channel Loss plugin)
- ✅ Reuses existing Axolotl infrastructure
- ✅ Drop-in solution
- ✅ Easy to maintain

**Disadvantages**:
- ⚠️ Slightly less efficient (extra all-gather on labels)
- ⚠️ Overhead: ~0.5-3ms per step (negligible)

---

## Decision Matrix

| Criterion | ms-swift GatherLoss | Axolotl-Native | Winner |
|-----------|---------------------|----------------|--------|
| **Development Time** | High (weeks) | Low (days) | ✅ Native |
| **Code Complexity** | High | Low | ✅ Native |
| **Maintenance** | Medium (sync with ms-swift) | Low (uses Axolotl utils) | ✅ Native |
| **Performance** | Best (no extra gather) | Good (< 1% overhead) | GatherLoss |
| **Compatibility** | Requires refactor | Drop-in | ✅ Native |
| **Risk** | High (breaks CP flow) | Low (isolated change) | ✅ Native |

**Recommendation**: **Implement Axolotl-Native solution** first, optimize later if needed.

---

## Implementation Checklist

### Phase 1: Core Functionality
- [ ] Add `_get_context_parallel_group()` helper
- [ ] Modify `track_channel_loss_statistics()` to gather labels
- [ ] Handle position_ids gathering (if present)
- [ ] Remove CP > 1 conflict detection from __init__.py
- [ ] Add unit tests

### Phase 2: Testing
- [ ] Test with CP=2, TP=2, FSDP=2 (8 GPUs)
- [ ] Test with sample packing + CP
- [ ] Verify channel metrics correctness
- [ ] Performance benchmarking

### Phase 3: Documentation
- [ ] Update COMPATIBILITY_ANALYSIS.md
- [ ] Update SWIFT_COMPATIBILITY_COMPARISON.md
- [ ] Add usage examples

---

## Future Optimizations (Optional)

If performance becomes a concern, consider:

1. **Lazy Gathering**: Only gather labels if channel loss is actually used in this step
2. **Selective Gathering**: Only gather needed portions based on segment boundaries
3. **Fusion**: Merge label gathering with logits gathering in sequence_parallel module

---

## Conclusion

**The Axolotl-native solution is preferred** because:

1. ✅ **Simpler**: Only modifies Channel Loss plugin
2. ✅ **Safer**: Reuses existing, tested AllGatherWithGrad
3. ✅ **Faster to implement**: Days vs weeks
4. ✅ **Easier to maintain**: No custom autograd functions
5. ✅ **Low risk**: Isolated change, easy to rollback
6. ⚠️ **Slight overhead**: < 1% performance impact (acceptable)

**Next Steps**:
1. Implement core functionality (Phase 1)
2. Test on 8-GPU setup with CP=2
3. Verify channel metrics correctness
4. Update documentation
5. Consider ms-swift GatherLoss only if performance becomes critical

---

## References

- Axolotl CP Implementation: `src/axolotl/utils/ctx_managers/sequence_parallel.py`
- AllGatherWithGrad: `src/axolotl/utils/ctx_managers/sequence_parallel.py:311-387`
- Channel Loss: `src/axolotl/integrations/channel_loss/`
- ms-swift GatherLoss: `swift/trainers/sequence_parallel/utils.py:19-53`
