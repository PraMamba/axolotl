---
name: debug-training
description: Guide for debugging training issues in axolotl. Use when user encounters training failures, OOM, loss issues, or configuration problems.
---

# Debug Training

Comprehensive guide for diagnosing and resolving training issues in axolotl.

## When to Use

- Training crashes or hangs
- Out-of-memory (OOM) errors
- Loss not decreasing or NaN
- Checkpoint loading failures
- Configuration validation errors
- Distributed training issues

## Diagnostic Workflow

### Step 1: Identify the Error Category

| Symptom                     | Category              | Go To          |
| --------------------------- | --------------------- | -------------- |
| Python traceback            | Code Error            | Section 2      |
| CUDA OOM                    | Memory Issue          | Section 3      |
| Loss NaN / not decreasing   | Training Issue        | Section 4      |
| Training hangs               | Distributed Issue     | Section 5      |
| Config validation error      | Config Issue          | Section 6      |
| Checkpoint load failure      | Checkpoint Issue      | Section 7      |

### Step 2: Code Errors

**Common causes:**

| Error Type                 | Likely Cause                                    |
| -------------------------- | ----------------------------------------------- |
| `AttributeError`           | DictDefault returned None for missing config key|
| `ImportError`              | Missing dependency or version mismatch          |
| `ValueError` from schema   | Invalid config combination                      |
| `RuntimeError` from CUDA   | Device mismatch or driver issue                 |
| `TypeError`                | Wrong argument type (often from DictDefault None)|

**Debugging steps:**
1. Check the full traceback for the actual failing line
2. If `None` appears unexpectedly, check config key spelling (DictDefault risk)
3. If import fails, check library versions: `pip show transformers peft trl`
4. Enable verbose logging: `AXOLOTL_LOG_LEVEL=debug axolotl train config.yml`

### Step 3: Memory Issues (OOM)

**Quick fixes (try in order):**

1. Reduce `micro_batch_size` (most impactful)
2. Enable `gradient_checkpointing: true`
3. Reduce `sequence_len`
4. Enable 4-bit quantization (`load_in_4bit: true`)
5. Use FSDP: add `fsdp:` config section
6. Disable `sample_packing` (increases memory usage)
7. Enable `cpu_offload` for FSDP

**Diagnosis:**
```bash
# Check GPU memory before training
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}, Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

# Monitor during training (in another terminal)
watch -n 1 nvidia-smi
```

### Step 4: Training Issues

**Loss NaN:**
- Check learning rate (too high -> NaN)
- Check for numerical overflow in bf16/fp16
- Verify dataset has valid labels (not all -100)
- Check reward function for DPO/GRPO (returning NaN?)

**Loss not decreasing:**
- Verify labels are correct (not all masked)
- Check `train_on_inputs` setting
- Reduce learning rate
- Check dataset quality (garbage in, garbage out)
- Verify correct prompt strategy matches dataset format

**Loss spikes:**
- Enable gradient clipping: `max_grad_norm: 1.0`
- Check for outlier samples in dataset
- Reduce learning rate

### Step 5: Distributed Training Issues

**Training hangs:**
1. Check NCCL: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`
2. Check network connectivity between nodes
3. Verify GPU count matches config
4. Check for deadlock: are all ranks reaching the same barrier?

**FSDP issues:**
- Verify `fsdp_config` in YAML
- Check FSDP2 compatibility: `monkeypatch/accelerate/fsdp2.py`
- For QLoRA + FSDP2: check `monkeypatch/fsdp2_qlora.py` compatibility

**DeepSpeed issues:**
- Verify deepspeed config JSON
- Check ZeRO stage matches use case
- For ZeRO-3: special model loading required

### Step 6: Config Issues

**Validation errors:**
- Read the error message carefully - it includes the invalid field and value
- Check `src/axolotl/utils/schemas/validation.py` for the rule
- Common issues:
  - `sample_packing` + `flash_attention` requirement
  - `load_in_4bit` + `adapter` requirement
  - Incompatible model type + feature combination

**Config debugging:**
```bash
# Preprocess only (validates config + loads data, no training)
axolotl preprocess config.yml

# Enable debug logging
AXOLOTL_LOG_LEVEL=debug axolotl train config.yml
```

### Step 7: Checkpoint Issues

**Load failures:**
- Check checkpoint format (FSDP sharded vs full, DeepSpeed vs HF)
- Verify model architecture matches checkpoint
- Check `output_dir` for checkpoint files
- For FSDP: `axolotl merge-sharded-fsdp-weights config.yml`

**Resume failures:**
- Check `resume_from_checkpoint` path
- Verify optimizer state compatibility
- Check for changed config between runs

## Environment Diagnostics

```bash
# Full environment check
python -c "
import torch
import transformers
import peft
import trl
import accelerate
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
print(f'Accelerate: {accelerate.__version__}')
"
```
