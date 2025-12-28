# Channel Loss Plugin

Per-channel loss tracking for multi-domain training in Axolotl.

## Overview

The Channel Loss Plugin enables you to track loss metrics separately for different data sources (channels) during training. This is particularly useful for:

- **Multi-domain training**: Monitor how well your model learns from different domains (e.g., math, code, general text)
- **Dataset balancing**: Identify which datasets need more/less representation
- **Training diagnostics**: Detect overfitting or underfitting in specific data sources

The feature is ported from the [ms-swift](https://github.com/modelscope/swift) framework and designed to be:
- **Observer-only**: Does not modify training dynamics or gradients
- **Distributed-ready**: Works with DDP, FSDP, DeepSpeed ZeRO-2/3
- **Packing-compatible**: Supports sample packing with automatic segment detection

## Quick Start

```yaml
# config.yaml
plugins:
  - axolotl.integrations.channel_loss.ChannelLossPlugin

enable_channel_loss: true

datasets:
  - path: /data/math.jsonl
    type: alpaca
    channel: math

  - path: /data/code.jsonl
    type: alpaca
    channel: code
```

During training, you'll see:
```
Step 10: loss=2.345, loss_math=2.123, loss_code=2.567
Step 20: loss=2.234, loss_math=2.001, loss_code=2.456
```

## Compatibility Matrix

### ✅ Compatible Features

| Feature | Status | Notes |
|---------|--------|-------|
| **Sample Packing** | ✅ Fully Compatible | Automatic segment detection via position_ids/attention_mask |
| **Chunked Cross Entropy** | ✅ Fully Compatible | Logits are materialized in chunks |
| **Liger Cross Entropy** (non-fused) | ✅ Fully Compatible | Use `liger_cross_entropy: true` (not the fused version) |
| **DeepSpeed ZeRO-2/3** | ✅ Fully Compatible | Automatic distributed synchronization |
| **FSDP** | ✅ Fully Compatible | Works with all FSDP sharding strategies |
| **Tensor Parallelism** | ✅ Fully Compatible | Channel stats tracked on each TP rank |
| **Context Parallelism** | ✅ Fully Compatible | NaN/Inf filtering for CP boundaries |
| **Gradient Checkpointing** | ✅ Fully Compatible | Uses detached tensors, no gradient impact |
| **Flash Attention** | ✅ Fully Compatible | No interaction with attention mechanisms |
| **LoRA/QLoRA** | ✅ Fully Compatible | Works with all PEFT adapters |
| **4-bit/8-bit Quantization** | ✅ Fully Compatible | No precision requirements |

### ❌ Incompatible Features

| Feature | Status | Why Incompatible | Solution |
|---------|--------|------------------|----------|
| **Liger Fused Linear Cross Entropy** | ❌ **Incompatible** | Skips logits materialization with `skip_logits=True` in training mode | Use `chunked_cross_entropy: true` instead (compatible, saves memory) |
| **Knowledge Distillation (KD)** | ❌ **Incompatible** | KD's `compute_loss()` ignores `return_outputs=True` parameter | Use standard SFT training or disable Channel Loss |
| **Cut Cross Entropy** | ⚠️ Auto-Disabled | Does not materialize logits to save memory | Plugin automatically disables CCE with warning |

### ⚠️ Semantic Warnings

| Feature | Status | Notes |
|---------|--------|-------|
| **RL Training** (DPO/KTO/ORPO/SIMPO/GRPO) | ⚠️ Works but Questionable | RL uses sample-level preference loss, not per-token loss. Channel statistics may not be meaningful. |

## Configuration Options

```yaml
# Enable the plugin
plugins:
  - axolotl.integrations.channel_loss.ChannelLossPlugin

enable_channel_loss: true

# Field name containing channel info (default: "channel")
channel_loss_field: "channel"

# Metric prefix in logs (default: "loss_")
# Results in: loss_math, loss_code, etc.
channel_loss_prefix: "loss_"

# Segment detection for packing mode (default: "auto")
# Options: "auto" | "position_ids" | "attention_mask"
channel_loss_segment: "auto"

# Warn when channel field is missing (default: true)
channel_loss_warn_on_missing: true
```

## Usage Methods

### Method 1: Dataset-Level Channels

Specify channel for entire dataset:

```yaml
datasets:
  - path: /data/math.jsonl
    type: alpaca
    channel: math  # All samples tagged as "math"

  - path: /data/code.jsonl
    type: alpaca
    channel: code  # All samples tagged as "code"
```

### Method 2: Sample-Level Channels

Include channel in each data sample:

```jsonl
{"messages": [...], "channel": "math"}
{"messages": [...], "channel": "code"}
{"messages": [...], "channel": "general"}
```

Then load without dataset-level channel:

```yaml
datasets:
  - path: /data/mixed.jsonl
    type: alpaca
    # Channel comes from individual samples
```

## Troubleshooting

### Error: "Channel Loss is incompatible with liger_fused_linear_cross_entropy"

**Cause**: Liger FLCE uses `skip_logits=True` in training mode to save memory, but Channel Loss requires access to logits.

**Solutions**:
1. **Recommended**: Use `chunked_cross_entropy: true` instead
   ```yaml
   # Remove or comment out:
   # liger_fused_linear_cross_entropy: true

   # Use this instead:
   chunked_cross_entropy: true
   chunk_size: 8192  # Adjust based on memory
   ```

2. Use non-fused Liger CE (partial optimization):
   ```yaml
   liger_cross_entropy: true  # Not the fused version
   ```

3. Disable Channel Loss if Liger FLCE is critical:
   ```yaml
   enable_channel_loss: false
   liger_fused_linear_cross_entropy: true
   ```

### Error: "Channel Loss is incompatible with KD trainer"

**Cause**: Knowledge Distillation trainer's `compute_loss()` does not support `return_outputs=True`.

**Solutions**:
1. Use standard SFT training if Channel Loss is required
2. Disable Channel Loss for KD training
3. Track GitHub issue for KD Trainer fix

### Warning: "No logits available from compute_loss()"

**Cause**: An incompatible optimization is enabled at runtime (e.g., custom trainer modification).

**What happens**: Training continues normally but channel statistics are NOT recorded.

**Solutions**:
1. Check for custom trainer modifications
2. Verify no incompatible plugins are loaded
3. Review training logs for conflicting optimizations

### Warning: "Channel Loss enabled with RL training mode"

**Cause**: You're using Channel Loss with DPO/KTO/ORPO/SIMPO/GRPO.

**Implications**: RL training uses sample-level preference loss, not per-token causal loss. Channel statistics may not reflect meaningful learning signals.

**Action**: Consider whether per-channel monitoring makes sense for your RL use case. This is a semantic warning, not a technical incompatibility.

### No channel metrics appearing in logs

**Checklist**:
1. ✅ Plugin added to `plugins:` list?
2. ✅ `enable_channel_loss: true` set?
3. ✅ Channel field present in dataset/samples?
4. ✅ No incompatible features enabled? (Liger FLCE, KD, CCE)
5. ✅ Check logs for "Channel Loss: Patched trainer.compute_loss" message
6. ✅ Verify `logging_steps` is set appropriately

### Metrics show NaN or Inf

**Cause**: Context Parallel training may produce NaN/Inf at sequence boundaries.

**Solution**: Already handled automatically. The plugin filters NaN/Inf values before accumulation.

## Advanced Usage

### With Sample Packing

Channel Loss automatically detects packed sequences:

```yaml
sample_packing: true
enable_channel_loss: true
channel_loss_segment: "auto"  # Prefers attention_mask, falls back to position_ids
```

### With Distributed Training

No special configuration needed. Channel statistics are automatically synchronized across ranks:

```yaml
# DeepSpeed
deepspeed: deepspeed_configs/zero3.json
enable_channel_loss: true

# FSDP
fsdp_config:
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
enable_channel_loss: true
```

### Custom Metric Prefix

Change the prefix for channel metrics:

```yaml
channel_loss_prefix: "domain_"
# Results in: domain_math, domain_code, etc.
```

### Integration with WandB/MLflow/SwanLab

Channel metrics are added to the standard logging dictionary and automatically tracked by logging integrations:

```yaml
wandb_project: my-project
enable_channel_loss: true
# Channel metrics appear in WandB automatically
```

## Performance Impact

- **Memory**: Negligible (only stores sum/count per channel)
- **Compute**: ~1-2% overhead (one additional forward pass through CE loss in no_grad mode)
- **I/O**: No impact on data loading or checkpointing

## Examples

See `qlora-channel-loss.yaml` for a complete working example with:
- QLoRA 4-bit quantization
- Sample packing
- Multi-domain datasets
- DeepSpeed compatibility notes

## References

- **Original Implementation**: [ms-swift Channel Loss](https://github.com/modelscope/swift)
- **Compatibility Audit**: `specs/001-channel-loss-compatibility-audit.md`
- **Source Code**: `src/axolotl/integrations/channel_loss/`

## Technical Details

### How It Works

1. **Data Collation**: Channel field is preserved through custom collator wrapper
2. **Compute Loss Patch**: `trainer.compute_loss()` is wrapped to:
   - Extract channel info (side input, not passed to model)
   - Call original compute_loss with `return_outputs=True`
   - Compute per-token CE separately (detached, no gradient)
   - Accumulate sum/count per channel locally
3. **Logging Callback**: Synchronizes statistics across ranks and adds to logs

### Gradient Safety

Channel Loss uses:
- `torch.no_grad()` context for all statistics computation
- `.detach()` on all intermediate tensors
- Separate CE computation (does not affect original loss)

This ensures training dynamics are completely unchanged.

## Contributing

Found a compatibility issue? Please file an issue or submit a PR with:
1. Configuration that triggers the issue
2. Error message or unexpected behavior
3. Axolotl version and hardware setup
