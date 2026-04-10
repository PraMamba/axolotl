# BFT: Balanced Fine-Tuning

**Balanced Fine-Tuning (BFT)** extends DFT with sample-level difficulty weighting to improve learning on sparse, complex datasets (e.g., biomedical, scientific domains).

## Overview

BFT combines two levels of weighting:
- **Token-level (from DFT):** `w_{b,t} = exp(-CE)` - stabilizes gradients
- **Sample-level (BFT):** `s_b = 1 - min_window(mean(confidence))` - focuses on hard samples

Hard samples get up to 2x weight, easy samples get 0.1x weight, preventing both overfitting and underlearning.

## Quick Start

```yaml
plugins:
  - axolotl.integrations.bft.BFTPlugin

enable_bft_loss: true
bft_group_size: 256
bft_weight_floor: 0.1
bft_weight_ceiling: 2.0
bft_normalize_sample_weight: true
bft_warmup_steps: 100

# Optional: memory optimization for large vocab
dft_chunk_size: 2048  # Recommended for vocab > 100K
```

## Configuration Parameters

### Core Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_bft_loss` | `false` | Master switch for BFT loss |
| `bft_group_size` | `256` | Sliding window size for confidence computation |
| `bft_min_valid_tokens_in_window` | `32` | Skip windows with fewer valid tokens |

### Sample Weight Stabilization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bft_weight_floor` | `0.1` | Minimum sample weight (keeps ≥10% of easy samples) |
| `bft_weight_ceiling` | `2.0` | Maximum sample weight (limits hard samples to 2x) |
| `bft_normalize_sample_weight` | `true` | Normalize to mean=1.0 per batch |
| `bft_warmup_steps` | `100` | Linear warmup for sample weighting |

### Memory Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dft_chunk_size` | `None` | Chunk size for CE computation (reduces memory by 50-75%) |

## When to Use BFT

**Use BFT when:**
- Training on sparse-data domains (medical, legal, scientific)
- Model struggles with rare/complex patterns
- Need better generalization than standard SFT or DFT

**Don't use BFT when:**
- Training on abundant, high-quality data
- DFT already provides sufficient performance
- Using RL-based methods (ORPO, DPO, PPO)

## Compatibility

### ✅ Compatible With
- Context Parallel (CP) / Ring Attention
- Sample packing (v1: treats packed sequence as single sample)
- Gradient accumulation
- Mixed precision training
- Gradient checkpointing

### ❌ Incompatible With
- `enable_dft_loss=true` (mutually exclusive)
- `label_smoothing_factor > 0`
- RL paths (ORPO, DPO, PPO)

## Monitoring

BFT adds the following metrics to training logs:

| Metric | Description |
|--------|-------------|
| `bft/s_mean` | Mean sample weight (should be ≈1.0 with normalization) |
| `bft/s_min` | Min sample weight (easy samples) |
| `bft/s_max` | Max sample weight (hard samples) |
| `bft/p_conf_mean` | Mean minimum window confidence |
| `bft/valid_token_ratio` | Fraction of non-masked tokens |
| `loss/raw_ce` | Unweighted CE (for cross-method comparison) |

**Healthy training indicators:**
- `bft/s_mean ≈ 1.0` (due to normalization)
- `bft/s_max > bft/s_min` (differential weighting active)
- `bft/s_max ≤ 2.0` and `bft/s_min ≥ 0.1` (clipping working)

## Advanced: Tuning for Your Dataset

### Adjusting Difficulty Sensitivity

**More sensitive to hard samples:**
```yaml
bft_group_size: 128           # Smaller window → more local
bft_weight_ceiling: 3.0       # Allow 3x weight on hardest
```

**More conservative (closer to DFT):**
```yaml
bft_group_size: 512           # Larger window → smoother
bft_weight_ceiling: 1.5       # Limit hard sample boost
bft_weight_floor: 0.3         # Keep more easy samples
```

### Handling Excessive Masking

If your dataset has many masked tokens (e.g., packing with short documents):
```yaml
bft_min_valid_tokens_in_window: 16  # Lower threshold
```

### Warmup for Unstable Models

If loss spikes early in training:
```yaml
bft_warmup_steps: 500  # Longer warmup
```

## Technical Details

### Sample Weight Computation

1. **Sliding Windows:** For each sample, slide a window of size `bft_group_size` over the sequence
2. **Local Confidence:** Compute mean token confidence in each window
3. **Hardest Region:** Take minimum mean across all windows → `p_conf[b]`
4. **Weight:** `s_b = 1 - p_conf[b]`
5. **Stabilize:** Clamp to `[floor, ceiling]`, normalize, apply warmup

### CP (Context Parallel) Compatibility

BFT is fully compatible with ring-attention CP:
- Token confidences and masks are all-gathered across CP ranks
- Sample weights computed on full sequences (identical across ranks)
- Only confidences/masks are gathered (not full logits), keeping memory overhead low (~4MB @ B=2, L=16k)

### Packing Behavior (v1)

Current implementation treats a packed sequence as a single sample:
- `s_b` applies uniformly across all documents in the pack
- Suboptimal for multi-document packing but simple and stable
- Future v2 will support per-document weights via `segment_ids`

## Citation

```bibtex
@article{tang2025bft,
  title={Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning},
  author={Tang, Zhenchao and Wang, Fang and He, Haohuai and others},
  journal={arXiv preprint},
  year={2025}
}
```

## References

- **Paper:** Tang et al., "Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning" (2025)
- **DFT:** Li et al., "Dynamic Fine-Tuning" (2024) - arXiv:2508.05629
- **Axolotl DFT Implementation:** `src/axolotl/integrations/dft/`
