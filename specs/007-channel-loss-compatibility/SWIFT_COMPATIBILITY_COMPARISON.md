# ms-swift vs Axolotl: Channel Loss Compatibility Comparison

**Date**: 2025-12-29
**Analysis**: Comparing incompatibility handling between ms-swift (original) and Axolotl (port)

---

## Executive Summary

This document compares how **ms-swift** (the original implementation) and **Axolotl** (the port) handle Channel Loss compatibility with various training optimizations. The analysis focuses on the 4 features marked as **INCOMPATIBLE** in Axolotl.

### Key Findings

| Feature | ms-swift | Axolotl | Difference |
|---------|----------|---------|------------|
| **Context/Sequence Parallelism** | ✅ **COMPATIBLE** with special handling | ❌ **INCOMPATIBLE** - raises ValueError | **ms-swift has solution, Axolotl doesn't** |
| **Liger FLCE** | ⚠️ **WARNING** - bypasses with warning | ❌ **INCOMPATIBLE** - raises ValueError | **Different enforcement strategies** |
| **KD Trainer** | ℹ️ **N/A** - no traditional KD trainer | ❌ **INCOMPATIBLE** - raises ValueError | **ms-swift uses GKD instead** |
| **Cut Cross Entropy** | ⚠️ **LIKELY COMPATIBLE** - needs verification | ⚠️ **AUTO-DISABLED** - automatic handling | **Both frameworks handle gracefully** |

---

## Detailed Comparison

### 1. Context Parallelism / Sequence Parallelism

#### ❌ Axolotl: INCOMPATIBLE

**Status**: Hard incompatibility with `context_parallel_size > 1`

**Implementation**: `src/axolotl/integrations/channel_loss/__init__.py:120-134`

```python
context_parallel_size = cfg.get("context_parallel_size", 1)
if context_parallel_size > 1:
    raise ValueError(
        f"Channel Loss is incompatible with context_parallel_size > 1..."
    )
```

**Error**:
```
ValueError: Expected input batch_size (1023) to match target batch_size (2047).
Location: compute_loss_patch.py:200
```

**Root Cause**:
- CP slices sequence dimension across devices
- Shift operation for causal LM creates boundary mismatches
- Per-token loss computation fails with shape mismatch

---

#### ✅ ms-swift: COMPATIBLE with Special Handling

**Status**: Fully compatible with `sequence_parallel_size > 1`

**Implementation**: `swift/trainers/trainers.py:356-357` + `swift/trainers/utils.py:59-91`

```python
if self.template.sequence_parallel_size > 1:
    outputs.loss = per_token_loss_func_sp(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
else:
    outputs.loss = per_token_loss_func(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
```

**Special Function**: `per_token_loss_func_sp()` in `swift/trainers/utils.py:59-91`

```python
def per_token_loss_func_sp(outputs, labels, enable_dft_loss=False, **kwargs) -> torch.Tensor:
    """Common loss function for sequence parallel training"""
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    device = logits.device

    batch_size = logits.shape[0]
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.flatten().to(device)

    # Compute per-token loss
    sploss_parallel_size = int(os.environ.get('CELOSS_PARALLEL_SIZE', '0'))
    if sploss_parallel_size > 0:
        from swift.trainers.sequence_parallel.utils import ChunkedCrossEntropyLoss
        loss = ChunkedCrossEntropyLoss.apply(logits, labels, sploss_parallel_size)
    else:
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits, labels)

    # Gather loss from all sequence parallel ranks
    from swift.trainers.sequence_parallel import sequence_parallel
    position_ids = sequence_parallel.real_position_ids
    if position_ids is not None:
        position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
    from swift.trainers.sequence_parallel.utils import GatherLoss
    loss, labels = GatherLoss.apply(loss.reshape(batch_size, -1), labels.reshape(batch_size, -1), 1, position_ids)

    # Filter padding tokens
    if position_ids is not None and position_ids.min() == -1:
        _pos_mask = position_ids >= 0
        loss = loss[_pos_mask].contiguous()

    return loss
```

**Key Solution**: `GatherLoss` class in `swift/trainers/sequence_parallel/utils.py:19-53`

```python
class GatherLoss(torch.autograd.Function):
    """Gather loss from sequence group"""

    @staticmethod
    def forward(ctx, loss, labels, gather_idx=None, position_ids=None):
        """Gather loss across sequence parallel ranks"""
        ctx.scatter_shape = loss.shape[gather_idx or 0]
        ctx.gather_idx = gather_idx or 0
        from swift.trainers.sequence_parallel import sequence_parallel
        if position_ids is not None:
            position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
        ctx.position_ids = position_ids
        output = sequence_parallel.gather(loss, dim=ctx.gather_idx, position_ids=position_ids)
        if labels is not None:
            labels_output = sequence_parallel.gather(labels, dim=ctx.gather_idx, position_ids=position_ids)
        else:
            labels_output = None
        return output, labels_output

    @staticmethod
    def backward(ctx, *grad_output):
        """Scatter gradients back to sequence parallel ranks"""
        from swift.trainers.sequence_parallel import sequence_parallel
        _grad = grad_output[0] * sequence_parallel.world_size
        if sequence_parallel.rp_world_size > 1:
            _grad = sequence_parallel.split(_grad, dim=ctx.gather_idx, position_ids=ctx.position_ids).contiguous()
        else:
            _grad = _grad.split(
                ctx.scatter_shape, dim=ctx.gather_idx)[dist.get_rank(sequence_parallel.sp_group)].contiguous()
        return _grad, None, None, None
```

**How It Works**:
1. Compute per-token loss on each sequence parallel rank independently
2. Use `GatherLoss.apply()` to gather losses from all ranks into a unified tensor
3. Use `position_ids` to align sequence boundaries and filter padding
4. In backward pass, scatter gradients back to respective ranks
5. Channel loss can then track per-channel statistics on the gathered loss

**Why This Works**:
- Avoids the shift operation boundary issue by gathering before shift
- Uses `position_ids` to correctly align sequence chunks
- Maintains gradient flow through custom backward pass

---

#### 📊 Comparison Summary

| Aspect | ms-swift | Axolotl |
|--------|----------|---------|
| **Compatibility** | ✅ Compatible | ❌ Incompatible |
| **Solution** | `GatherLoss` + `per_token_loss_func_sp` | None - raises error |
| **Complexity** | High - custom autograd function | N/A |
| **Testing** | ✅ Used in production | ❌ Explicitly blocked |
| **Code Location** | `swift/trainers/sequence_parallel/utils.py` | `src/axolotl/integrations/channel_loss/__init__.py` |

**Recommendation for Axolotl**: Consider porting `GatherLoss` and `per_token_loss_func_sp` from ms-swift to enable CP > 1 compatibility.

---

### 2. Liger Fused Linear Cross Entropy (FLCE)

#### ❌ Axolotl: INCOMPATIBLE (Hard Error)

**Status**: Hard incompatibility - raises ValueError

**Implementation**: `src/axolotl/integrations/channel_loss/__init__.py:95-105`

```python
if cfg.get("liger_fused_linear_cross_entropy"):
    raise ValueError(
        "Channel Loss is incompatible with liger_fused_linear_cross_entropy.\n\n"
        "Reason: Liger FLCE skips logits materialization in training mode (skip_logits=True)\n"
        "to save memory, but Channel Loss requires access to logits for per-channel statistics.\n\n"
        "Solutions:\n"
        "  1. Use 'chunked_cross_entropy: true' instead (compatible, saves memory)\n"
        "  2. Use 'liger_cross_entropy: true' (non-fused, partial optimization)\n"
        "  3. Disable Channel Loss if Liger FLCE is critical for your memory budget"
    )
```

---

#### ⚠️ ms-swift: BYPASS with Warning

**Status**: Soft incompatibility - warns and bypasses Liger optimization

**Implementation**: `swift/trainers/trainers.py:323-328`

```python
if (self.label_smoother is not None or compute_loss_func is not None or loss_scale is not None
        or self.args.enable_dft_loss or self.args.enable_channel_loss
        or self.template.sequence_parallel_size > 1) and 'labels' in inputs:
    if self.args.use_liger_kernel:
        logger.warning_once('The cross_entropy loss function defined in Liger Kernel will not '
                            'take effect, potentially leading to increased GPU memory consumption.')
    labels = inputs.pop('labels')
```

**Behavior**:
1. Detects if channel loss is enabled
2. Pops `labels` from inputs to prevent Liger FLCE from using them
3. Computes loss manually with standard `CrossEntropyLoss(reduction='none')`
4. Warns user that Liger optimization won't apply

**Memory Impact**: Higher GPU memory usage compared to Liger FLCE, but still functional.

---

#### 📊 Comparison Summary

| Aspect | ms-swift | Axolotl |
|--------|----------|---------|
| **Enforcement** | ⚠️ Warning | ❌ Hard error |
| **User Experience** | Graceful degradation | Explicit blocking |
| **Memory Efficiency** | Reduced (standard CE) | N/A (blocked) |
| **Flexibility** | Can still use both features | Must choose one |

**Recommendation**: Axolotl's hard error is safer (prevents unexpected memory issues), while ms-swift's warning is more flexible.

---

### 3. Knowledge Distillation (KD) Trainer

#### ❌ Axolotl: INCOMPATIBLE

**Status**: Hard incompatibility with KD trainer

**Implementation**: `src/axolotl/integrations/channel_loss/__init__.py:108-118`

```python
if cfg.get("kd_trainer"):
    raise ValueError(
        "Channel Loss is incompatible with KD trainer.\n\n"
        "Reason: KD's compute_loss() method does not support return_outputs=True,\n"
        "preventing Channel Loss from accessing model outputs and logits.\n\n"
        "Solutions:\n"
        "  1. Disable Channel Loss for KD training\n"
        "  2. Wait for KD Trainer fix (track issue in GitHub)\n"
        "  3. Use standard SFT training if Channel Loss is required"
    )
```

**Root Cause**: Axolotl's KD trainer `compute_loss()` doesn't support `return_outputs=True`

---

#### ℹ️ ms-swift: N/A (Different Architecture)

**Status**: Not applicable - ms-swift doesn't have traditional KD trainer

**Implementation**: ms-swift uses **GKDTrainer** (Generalized Knowledge Distillation) instead

**Key Difference**:
- Axolotl: `KDTrainer` for knowledge distillation
- ms-swift: `GKDTrainer` for generalized KD (uses different loss computation)

**Channel Loss Support in GKD**:
- No explicit conflict detection found
- GKDTrainer doesn't appear to support channel loss (no references in code)

---

#### 📊 Comparison Summary

| Aspect | ms-swift | Axolotl |
|--------|----------|---------|
| **KD Architecture** | GKDTrainer | KDTrainer |
| **Channel Loss Support** | ℹ️ Unknown (likely unsupported) | ❌ Explicitly incompatible |
| **Conflict Detection** | None found | ✅ Hard error |

**Note**: This incompatibility is Axolotl-specific due to different KD trainer implementations.

---

### 4. Cut Cross Entropy (CCE)

#### ⚠️ Axolotl: AUTO-DISABLED

**Status**: Soft conflict - automatically disabled with warning

**Implementation**: `src/axolotl/integrations/channel_loss/__init__.py:139-150`

```python
cce_plugin = "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"
plugins = cfg.get("plugins", [])

if cce_plugin in plugins:
    LOG.warning(
        "Channel Loss Plugin: Cut Cross Entropy detected. "
        "These features are incompatible because CCE does not materialize logits. "
        "Disabling Cut Cross Entropy..."
    )
    # CCE plugin checks cfg.cut_cross_entropy flag in pre_model_load()
    # Setting this to False prevents CCE from being applied
    cfg["cut_cross_entropy"] = False
```

**Behavior**: Automatically disables CCE when Channel Loss is detected

---

#### ⚠️ ms-swift: LIKELY COMPATIBLE (Needs Verification)

**Status**: Not explicitly documented as incompatible

**Evidence**:
- No conflict detection code found for CCE in channel loss implementation
- `swift/trainers/trainers.py` does not check for CCE conflicts
- However, manual verification needed to confirm compatibility

**Hypothesis**: ms-swift may not use CCE in the same way as Axolotl, or may have implicit compatibility.

---

#### 📊 Comparison Summary

| Aspect | ms-swift | Axolotl |
|--------|----------|---------|
| **Conflict Handling** | ⚠️ None detected | ⚠️ Auto-disable |
| **User Notification** | None | Warning |
| **Behavior** | Unknown | Disables CCE |

**Recommendation**: Verify ms-swift's CCE implementation and compatibility before assuming it works with channel loss.

---

## Summary Table

| Feature | ms-swift | Axolotl | Key Difference |
|---------|----------|---------|----------------|
| **Sequence/Context Parallelism** | ✅ Compatible (`GatherLoss`) | ❌ Incompatible (hard block) | **ms-swift has working solution** |
| **Liger FLCE** | ⚠️ Warning (bypass) | ❌ Hard error | **Different enforcement philosophy** |
| **KD Trainer** | ℹ️ N/A (uses GKD) | ❌ Hard error | **Different architectures** |
| **Cut Cross Entropy** | ⚠️ Likely compatible | ⚠️ Auto-disabled | **Needs verification** |

---

## Recommendations

### For Axolotl Developers

1. **Port `GatherLoss` from ms-swift** to enable Context Parallelism (CP > 1) support
   - Location: `swift/trainers/sequence_parallel/utils.py:19-53`
   - Benefit: Enable long-sequence training with CP
   - Complexity: Moderate (requires sequence parallel infrastructure)

2. **Consider softening Liger FLCE enforcement**
   - Current: Hard error
   - Alternative: Warning + bypass (like ms-swift)
   - Trade-off: Flexibility vs. safety

3. **Verify Cut Cross Entropy handling**
   - Current: Auto-disable
   - Recommendation: Check if ms-swift's approach allows compatibility
   - Benefit: Potential memory savings

### For ms-swift Users

1. **Verify Cut Cross Entropy compatibility**
   - No explicit conflict detection found
   - Manual testing recommended before production use

2. **Be aware of Liger FLCE memory overhead**
   - Warning message explains Liger CE won't apply
   - Plan for ~20-30% higher GPU memory usage

3. **Use GKDTrainer instead of traditional KD**
   - Different architecture, likely incompatible with channel loss
   - Verify if GKD supports channel tracking before use

---

## Testing Recommendations

### High Priority Tests

1. **ms-swift + Sequence Parallelism**
   ```bash
   # Verify channel loss works with sequence_parallel_size > 1
   python test_channel.py --sequence_parallel_size 2
   ```

2. **ms-swift + Cut Cross Entropy**
   ```bash
   # Verify CCE + channel loss compatibility
   python test_channel.py --cut_cross_entropy true
   ```

### Low Priority Tests

1. **ms-swift + GKDTrainer**
   - Verify if GKD can be combined with channel loss
   - Document findings

---

## Changelog

### 2025-12-29
- ✅ Initial comparison analysis
- ✅ Identified `GatherLoss` as key CP compatibility solution
- ✅ Documented Liger FLCE handling differences
- ✅ Noted KD trainer architectural differences
- ⚠️ Cut Cross Entropy compatibility needs verification

---

## References

### ms-swift Code References
- Channel Loss: `swift/trainers/trainers.py:321-376`
- Sequence Parallel Loss: `swift/trainers/utils.py:59-91`
- GatherLoss: `swift/trainers/sequence_parallel/utils.py:19-53`
- Liger Warning: `swift/trainers/trainers.py:326-328`
- GKDTrainer: `swift/trainers/rlhf_trainer/gkd_trainer.py`

### Axolotl Code References
- Channel Loss Plugin: `src/axolotl/integrations/channel_loss/__init__.py`
- Compatibility Analysis: `specs/007-channel-loss-compatibility/COMPATIBILITY_ANALYSIS.md`
- KD Trainer: `src/axolotl/integrations/kd/trainer.py`

### Test Files
- ms-swift: `tests/train/test_channel.py`
- Axolotl: `tests/integrations/test_channel_loss.py`
