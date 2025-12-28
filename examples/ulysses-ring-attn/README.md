# Ulysses + Ring-Attention Example Configurations

This directory contains example configurations for testing the Ulysses + Ring-Attention plugin (Phase 2.2: Multi-GPU Integration Testing).

## Overview

The Ulysses + Ring-Attention plugin enables arbitrary chunk sharding in context parallelism by combining:
- **Ulysses attention**: All-to-all communication on heads dimension (sp_size)
- **Ring-Attention**: Ring communication on sequence dimension (rp_size)

Key innovation: GCD-based decomposition automatically finds optimal `sp × rp = context_parallel_size` split.

## Configuration Files

| File | GPUs | Mode | Description |
|------|------|------|-------------|
| `config_hybrid_sp3_rp2.yml` | 6 | Hybrid | sp=3 × rp=2 (GCD decomposition) |
| `config_ulysses_only_sp4.yml` | 3 | Ulysses-only | sp=3 × rp=1 (pure Ulysses) |
| `config_ring_only_rp4.yml` | 4 | Ring-only | sp=1 × rp=4 (pure Ring) |
| `config_manual_sp4_rp2.yml` | 8 | Manual override | sp=4 × rp=2 (user-specified) |
| `config_baseline_no_cp.yml` | 1 | Baseline | No context parallelism (for comparison) |

## Quick Start

### 1. Hybrid Mode (6 GPUs)

```bash
accelerate launch --num-processes 6 \
  -m axolotl.cli.train \
  examples/ulysses-ring-attn/config_hybrid_sp3_rp2.yml
```

**What happens:**
- SmolLM2-135M has 9 attention heads
- `context_parallel_size=6` → `gcd(9, 6) = 3` → `sp=3, rp=2`
- 3-way Ulysses attention (heads split across 3 ranks)
- 2-way Ring-Attention (sequence split across 2 ranks)

### 2. Ulysses-Only Mode (3 GPUs)

```bash
accelerate launch --num-processes 3 \
  -m axolotl.cli.train \
  examples/ulysses-ring-attn/config_ulysses_only_sp4.yml
```

**What happens:**
- `context_parallel_size=3` divides 9 heads → `sp=3, rp=1`
- Pure Ulysses attention (no ring communication)

### 3. Ring-Only Mode (4 GPUs)

```bash
accelerate launch --num-processes 4 \
  -m axolotl.cli.train \
  examples/ulysses-ring-attn/config_ring_only_rp4.yml
```

**What happens:**
- `context_parallel_size=4` coprime to 9 heads → `gcd(9, 4) = 1` → `sp=1, rp=4`
- Pure Ring-Attention (no Ulysses all-to-all)

### 4. Manual Override (8 GPUs)

```bash
accelerate launch --num-processes 8 \
  -m axolotl.cli.train \
  examples/ulysses-ring-attn/config_manual_sp4_rp2.yml
```

**What happens:**
- User explicitly sets `ulysses_ring_attention_sp_size=4` and `ulysses_ring_attention_rp_size=2`
- Bypasses GCD-based decomposition
- Useful for experimenting with different sp/rp ratios

### 5. Baseline (1 GPU)

```bash
accelerate launch --num-processes 1 \
  -m axolotl.cli.train \
  examples/ulysses-ring-attn/config_baseline_no_cp.yml
```

**What happens:**
- Standard training without Ulysses + Ring-Attention
- Use for convergence comparison (loss curves should match)

## Configuration Parameters

### Required

```yaml
# Enable plugin
plugins:
  - axolotl.integrations.ulysses_ring_attn

ulysses_ring_attention_enabled: true
context_parallel_size: 4  # Total parallelism (sp × rp)

# Dependencies
flash_attention: true  # Required for ring-flash-attn
sample_packing: true   # Required for Phase 1 (varlen API)
```

### Optional

```yaml
# Decomposition mode
ulysses_ring_attention_mode: auto  # auto (default), hybrid, ulysses_only, ring_only

# Manual overrides (bypass GCD decomposition)
ulysses_ring_attention_sp_size: 4
ulysses_ring_attention_rp_size: 2
```

## Phase 2.2 Testing Plan

### Test Matrix

Run all configurations and validate:

1. **Training Completion**: All runs complete without errors
2. **Loss Convergence**: Distributed loss matches baseline
3. **Gradient Correctness**: Verify gradient flow
4. **Performance**: Measure tokens/sec vs baseline

### Validation Steps

1. **Run baseline** (no context parallelism):
   ```bash
   ./run_baseline.sh
   ```

2. **Run distributed tests** (requires 4-8 GPUs):
   ```bash
   ./run_hybrid_sp3_rp2.sh
   ./run_ulysses_only_sp3.sh
   ./run_ring_only_rp4.sh
   ./run_manual_sp4_rp2.sh
   ```

3. **Compare results**:
   ```bash
   python compare_runs.py \
     --baseline outputs/baseline_no_cp/runs \
     --distributed outputs/ulysses_ring_*/runs \
     --metric train/train_loss
   ```

### Expected Results

- ✅ All training runs complete without NCCL errors or deadlocks
- ✅ Final train loss within 5% of baseline
- ✅ Loss curves have similar shape (no divergence)
- ✅ Throughput within 10-20% of baseline (communication overhead)

## Debugging

### Common Issues

#### 1. NCCL Timeout

```
RuntimeError: NCCL timeout after 600s
```

**Fix**: Increase timeout in environment:
```bash
export NCCL_TIMEOUT=1800  # 30 minutes
```

#### 2. Sequence Length Not Divisible

```
AssertionError: Sequence length 1024 must be divisible by 2 × rp_size (8)
```

**Fix**: Phase 1 constraint. Adjust sequence length:
```yaml
sequence_len: 2048  # Must be divisible by 2 × rp_size
```

#### 3. Unsupported Model Architecture

```
ValueError: UlyssesRingAttentionPlugin currently only supports Llama-style models.
```

**Fix**: Phase 2.1 supports: llama, mistral, mixtral, qwen2, phi3, gemma, gemma2, cohere.
Use a supported model or wait for Phase 2.3.

### Debug Logging

Enable verbose logging:
```yaml
# In config.yml
debug: true
```

Check plugin logs:
```bash
grep "UlyssesRingAttention" outputs/*/train.log
```

## Performance Profiling

### Measure Throughput

```bash
# Run with profiling
CUDA_PROFILE=1 accelerate launch --num-processes 4 \
  -m axolotl.cli.train config_hybrid_sp3_rp2.yml

# Check logs
grep "samples/sec" outputs/ulysses_ring_hybrid_sp3_rp2/train.log
```

### Compare Communication Overhead

Expected overhead:
- **Ulysses-only**: 5-10% (all-to-all on heads is fast)
- **Ring-only**: 10-20% (ring communication on sequence)
- **Hybrid**: 15-25% (both communication patterns)

## Next Steps: Phase 2.3

After validating Phase 2.2:
- Add support for more model architectures (GPT-NeoX, Falcon, BLOOM)
- Implement auto-padding (remove sequence length constraint)
- Optimize communication patterns
- Add GQA/MQA optimizations

## References

- Plugin Spec: `specs/006-ulysses-ring-attention-plugin/README.md`
- Phase 1 Summary: `specs/006-ulysses-ring-attention-plugin/PHASE1_COMPLETE.md`
- Phase 2.1 Summary: `specs/006-ulysses-ring-attention-plugin/PHASE2_1_COMPLETE.md`
- ms-swift implementation: `docs/analysis/ms_swift_ulysses_ring_attention_implementation.md`
