# KernelWiki Training Library Analysis: Axolotl

**Framework**: axolotl-ai-cloud/axolotl
**GitHub URL**: https://github.com/axolotl-ai-cloud/axolotl
**Local Source Path**: `/root/axolotl/.worktrees/source_code_analysis`
**Analysis Date**: 2026-05-28
**Library Classification**: **Training Orchestration Framework**

---

## Dimension 1: Compute Kernels

### Kernel File Census

Total kernel files: **0 CUDA/C++ files**, **~15 Triton kernel files**

- CUDA/C++: **0 files** — Axolotl contains no native `.cu`, `.cuh`, or `.ptx` files
- Triton: **9 files** in `src/axolotl/kernels/`, **4 files** in `src/axolotl/integrations/kernels/libs/scattermoe_lora/kernels/`, **1 file** in `src/axolotl/core/trainers/ebft/kernels.py`
- Extension entry points (PYBIND11/TORCH_LIBRARY): **0 files**

**Classification**: Axolotl is an **orchestration framework**. All core GPU compute (GEMM, attention, LayerNorm, optimizer steps) is delegated to upstream libraries (PyTorch, HuggingFace Transformers, Flash Attention, Liger Kernel, TransformerEngine). Axolotl provides a thin layer of custom Triton kernels for specialized operations.

### Training-Specific Triton Kernels (Axolotl-Owned)

| Kernel | File Path | Proposed Tag | Description |
|--------|-----------|--------------|-------------|
| SwiGLU forward | `src/axolotl/kernels/swiglu.py:L14` | `fused-swiglu` | Fused SiLU(gate) × up activation in FP32 for numerical stability |
| SwiGLU backward | `src/axolotl/kernels/swiglu.py:L49` | `fused-swiglu` | In-place backward: grad_gate and grad_up computed without intermediate tensors |
| GeGLU forward | `src/axolotl/kernels/geglu.py:L13` | `fused-geglu` | Fused GELU(gate) × up activation in FP32 |
| GeGLU backward | `src/axolotl/kernels/geglu.py:L70` | `fused-geglu` | In-place backward with GELU derivative computation |
| DoRA fused norm | `src/axolotl/kernels/dora.py:L16` | `dora-norm` | Computes `magnitude / ||W + s*(B@A)||₂` row-by-row without materializing full B@A matrix |
| LoRA fused forward/backward | `src/axolotl/kernels/lora.py` | `fused-lora` | Fused LoRA linear + activation (SwiGLU/GeGLU) forward and backward with quantization support; 67KB, largest Triton kernel in the repo. 6 `torch.autograd.Function` classes: `LoRA_MLP` (L283), `LoRA_QKV` (L824), `LoRA_QK` (L1300), `LoRA_O` (L1633), `LoRA_Embedding` (L1851). Handles NF4/FP8 dequant, DTensor/FSDP2 unsharding (L86-88), DoRA magnitude scaling, dropout |
| RMSNorm + SiLU Gate | `src/axolotl/kernels/rms_norm_gated.py:L1` | `fused-rmsnorm-gated` | Fused `(W + offset) * RMSNorm(X) * silu(G)` for Qwen3.5 GatedDeltaNet; depends on Liger Kernel utilities |
| Gemma4 Fused RoPE | `src/axolotl/kernels/gemma4_fused_rope.py` | `fused-rope` | Fused rotary position embedding for Gemma4 model family |
| Quantize/Dequantize | `src/axolotl/kernels/quantize.py` | `dequantize` | BitsAndBytes NF4/FP4 dequantization helpers for LoRA kernel integration |

### EBFT (Embedding-Based Fine-Tuning) Kernels

| Kernel | File Path | Proposed Tag | Description |
|--------|-----------|--------------|-------------|
| Fused log_softmax + gather | `src/axolotl/core/trainers/ebft/kernels.py:L25` | `selective-logsoftmax` | Selective log_softmax at target index without materializing full (B,S,V) output; 2-pass (max + sum_exp) |
| Fused masked REINFORCE loss | `src/axolotl/core/trainers/ebft/kernels.py:L105` | `fused-reinforce-loss` | `-logp * advantage * mask` reduced to scalar in one kernel with block-parallel accumulation |
| Fused cosine similarity | `src/axolotl/core/trainers/ebft/kernels.py:L174` | `fused-cosine-sim` | Batched cosine similarity fusing dot product + norms + division |
| Fused diversity penalty | `src/axolotl/core/trainers/ebft/kernels.py:L244` | `fused-diversity` | Pairwise dot product with self-exclusion for diversity regularization |

### ScatterMoE LoRA Kernels

| Kernel | File Path | Proposed Tag | Description |
|--------|-----------|--------------|-------------|
| ScatterMoE LoRA ops | `src/axolotl/integrations/kernels/libs/scattermoe_lora/kernels/lora_ops.py` (72KB) | `scattermoe-lora` | 4 `@triton.autotune` + `@triton.jit` kernels: `_scatter2scatter_lora` (L462, fwd Y=X@W+s*(X@A^T)@B^T), `_scatter2scatter_lora_dX` (L1043, bwd dX), `_group_bwd_lora` (L1376, dA/dB with `tl.atomic_add`), `_group_bwd_lora_split` (L1616, split dA/dB without atomics). Autotune search: BLOCK_M∈{32,64,128}, BLOCK_N∈{32,64}, BLOCK_K∈{32,64,128}. LoRA weight layout: A=[r*E,K], B=[N,r*E] expert-major |
| ScatterMoE base ops | `src/axolotl/integrations/kernels/libs/scattermoe_lora/kernels/ops.py` (17KB) | `scattermoe` | `@triton.autotune` kernels: `_scatter2scatter` (L79, fwd GEMM registered as `torch.library.custom_op`), `_groupXtY` (L346, bwd weight grad with `tl.swizzle2d`), `_group` (L592, token scatter/gather) |
| ScatterMoE single expert | `src/axolotl/integrations/kernels/libs/scattermoe_lora/kernels/single.py` | `scattermoe` | `_single2scatter` (L12, `@triton.jit` only, fixed BLOCK_N=128, BLOCK_K=128) for single-token or small-batch MoE dispatch |
| Selective dequantization | `src/axolotl/integrations/kernels/libs/scattermoe_lora/selective_dequant_kernel.py` | `selective-dequant` | `_selective_dequant_nf4_kernel` (L44): fuses expert gather + NF4 dequant in one pass — reads packed NF4 bytes, 16-value codebook lookup, absmax multiply, bf16 output. Eliminates intermediate gather buffer |

### Kernel Dependency Graph (Upstream Providers)

Since Axolotl is an orchestration framework, the actual GPU compute workload is provided by upstream libraries:

| Provider Library | Kernel Types Provided | Integration Point |
|-----------------|----------------------|-------------------|
| **PyTorch / torch.compile** | GEMM, attention (SDPA), LayerNorm, optimizer (AdamW, SGD), loss (CrossEntropy) | Default runtime; `torch._dynamo` compile integration |
| **Flash Attention 2/3** | Flash attention forward + backward | `src/axolotl/monkeypatch/attention/flash_attn_4.py`, `llama_attn_hijack_flash.py` |
| **Liger Kernel** | `LigerRMSNorm`, `LigerSwiGLUMLP`, `LigerGEGLUMLP`, `LigerLayerNorm`, `liger_rotary_pos_emb`, `LigerCrossEntropyLoss`, `LigerFusedLinearCrossEntropyLoss` | `src/axolotl/integrations/liger/plugin.py` `LigerPlugin.pre_model_load()` (L22-293) patches: Llama, Qwen2/3/3.5/3-MoE/3.5-MoE, Gemma4, DeepSeekV2, Jamba, Llama4, GraniteMoE. Also `kd/kernels/liger.py`: `LigerFusedLinearKLTopKLogprobFunction` (L14) for chunked KL-div KD loss |
| **Cut Cross Entropy** | Fused linear + cross-entropy (avoids materializing `[B*S,V]` logits) | `src/axolotl/integrations/cut_cross_entropy/__init__.py` `CutCrossEntropyPlugin.pre_model_load()` (L86-103) patches `ForCausalLM.forward` via Apple ML's `cce_patch()` |
| **xformers** | Memory-efficient attention | `src/axolotl/monkeypatch/attention/xformers.py`, `xformers_/__init__.py` |
| **FlexAttention** | PyTorch-native flexible attention | `src/axolotl/monkeypatch/attention/flex_attn.py` |
| **SageAttention** | Quantized attention | `src/axolotl/monkeypatch/attention/sage_attn.py` |
| **BitsAndBytes** | NF4/FP4 quantization, 8-bit optimizers | `src/axolotl/loaders/model.py`, `monkeypatch/fsdp2_qlora.py` |
| **TransformerEngine** | FP8 GEMM, FP8 LayerNorm | Referenced in FP8 FSDP2 integration (`loaders/patch_manager.py`) |
| **TorchAO** | FP8 training, quantized optimizers | `src/axolotl/monkeypatch/torchao_optim.py` |
| **DeepSpeed** | ZeRO optimizer, fused kernels | `deepspeed_configs/`, `monkeypatch/deepspeed_utils.py` |
| **Unsloth** | Inspiration for LoRA/SwiGLU/GeGLU Triton kernels | Credited in kernel docstrings |

### Proposed New kernel_types

| Tag | Representative File | Description |
|-----|---------------------|-------------|
| `fused-swiglu` | `src/axolotl/kernels/swiglu.py` | Fused SwiGLU activation forward + backward (gate × SiLU × up) |
| `fused-geglu` | `src/axolotl/kernels/geglu.py` | Fused GeGLU activation forward + backward |
| `fused-lora` | `src/axolotl/kernels/lora.py` | Fused LoRA linear + activation with quantized weight support |
| `dora-norm` | `src/axolotl/kernels/dora.py` | DoRA magnitude-norm-scale without B@A materialization |
| `selective-logsoftmax` | `src/axolotl/core/trainers/ebft/kernels.py` | Selective log_softmax at target index (avoids full vocab materialization) |
| `fused-reinforce-loss` | `src/axolotl/core/trainers/ebft/kernels.py` | Masked REINFORCE loss reduction in single kernel |
| `fused-cosine-sim` | `src/axolotl/core/trainers/ebft/kernels.py` | Batched cosine similarity fusing dot + norm + divide |
| `scattermoe-lora` | `src/axolotl/integrations/kernels/libs/scattermoe_lora/` | LoRA-adapted MoE expert dispatch via Triton scatter/gather |
| `fused-rmsnorm-gated` | `src/axolotl/kernels/rms_norm_gated.py` | Fused RMSNorm + SiLU gate for Qwen3.5 GatedDeltaNet |
| `fused-rope` | `src/axolotl/kernels/gemma4_fused_rope.py` | Fused rotary position embedding for Gemma4 |

---

## Dimension 2: Communication Kernels and Strategies

### Communication Architecture

Axolotl does **not implement communication kernels directly**. All distributed communication is delegated to:
- **PyTorch Distributed** (`torch.distributed`) — AllReduce, ReduceScatter, AllGather, AllToAll
- **HuggingFace Accelerate** — FSDP2 wrapper around PyTorch distributed
- **NCCL** — as the backend runtime (accessed through PyTorch, never directly)
- **DeepSpeed** — ZeRO communication when DeepSpeed backend is selected

### Collective Operations (via PyTorch Distributed)

| Operation | Triggered By | Usage Pattern | Evidence |
|-----------|-------------|---------------|----------|
| AllGather | FSDP2 forward pass | Parameters gathered before each layer's forward | `src/axolotl/monkeypatch/accelerate/fsdp2.py` |
| ReduceScatter | FSDP2 backward pass | Gradient reduction + sharding after each layer's backward | FSDP2 internal (PyTorch native) |
| AllReduce | DDP gradient sync, TP AllReduce | Gradient synchronization across data-parallel ranks; TP reduction after MLP/Attention | `torch.distributed.all_reduce` via Accelerate |
| AllGather | Ring Attention (CP) | KV block gathering across context-parallel ranks | `src/axolotl/monkeypatch/ring_attn/patch.py` |
| AllToAll | Expert Parallel | Token routing to expert GPUs via DeepEP `Buffer.dispatch/combine` | `src/axolotl/integrations/expert_parallel/experts_fn.py` — `_DeepEPDispatch` (L81) fwd + `_DeepEPCombine` (L124) bwd, using DeepEP's NVLink/RDMA buffer pools |
| Broadcast | GRPO dataset/rollout | Rank 0 broadcasts dataset and rollout data to all ranks | `src/axolotl/core/trainers/grpo/async_trainer.py:L450` — `dist.broadcast_object_list()` |
| AllReduce | Sequence Parallel loss | Correct cross-CP eval loss normalization | `src/axolotl/utils/ctx_managers/sequence_parallel.py:L305` — `dist.all_reduce(weighted_loss, op=SUM)` |
| AllGather | Sequence Parallel output | Gather sharded sequence outputs with gradients | `src/axolotl/utils/ctx_managers/sequence_parallel.py:L368` — `AllGatherWithGrad` autograd function |

### Communication-Compute Overlap Patterns

| Pattern | Mechanism | Evidence |
|---------|-----------|----------|
| FSDP2 backward prefetch | ReduceScatter of layer N overlapped with backward of layer N-1 via `backward_prefetch` config | `src/axolotl/monkeypatch/accelerate/fsdp2.py` (PyTorch FSDP2 native) |
| FSDP2 forward prefetch | AllGather of next layer overlapped with current layer's forward computation | FSDP2 `forward_prefetch` parameter |
| Gradient accumulation | Reduces communication frequency by accumulating N micro-batches before ReduceScatter | `src/axolotl/core/builders/base.py`, config key `gradient_accumulation_steps` |
| Async GRPO generation | vLLM inference server runs asynchronously on separate GPU group while training proceeds | `src/axolotl/core/trainers/grpo/async_trainer.py`, `fast_async_trainer.py` |

### Advanced Communication Features Checklist

- [ ] Symmetric memory support (NCCL 2.27+): **No** — no references to `symm_mem`, `NVLS`, or symmetric memory APIs
- [ ] Device API support (NCCL 2.28+): **No** — LSA: No, Multimem: No, GIN: No
- [ ] Copy Engine zero-SM collectives: **No** — no references to `copy_engine` or `ZERO_CTA`
- [ ] NCCL Inspector integration: **No** — no NCCL profiler plugin usage
- [ ] PyTorch SymmetricMemory: **No** — no `SymmetricMemory` API usage
- [ ] Alternative backend support (MSCCL++): **No** — no MSCCL++ or NVSHMEM references
- [x] NCCL backend via PyTorch: **Yes** — standard NCCL through `torch.distributed`

### Proposed New Communication kernel_types

None — Axolotl does not provide communication kernel implementations. All communication is delegated to PyTorch Distributed / NCCL.

### Proposed New Communication techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `fsdp2-prefetch` | `src/axolotl/monkeypatch/accelerate/fsdp2.py` | FSDP2 backward/forward prefetch for ReduceScatter/AllGather overlap with compute |
| `async-grpo-generation` | `src/axolotl/core/trainers/grpo/async_trainer.py` | Async vLLM generation overlapped with training on separate GPU group |

---

## Dimension 3: Parallelism Strategies

### Supported Parallelism Dimensions

| Dimension | Supported | Implementation File | Communication Pattern Triggered |
|-----------|-----------|--------------------|---------------------------------|
| Data Parallel (FSDP2) | **Yes** | `src/axolotl/monkeypatch/accelerate/fsdp2.py` | AllGather before forward + ReduceScatter after backward |
| Data Parallel (DeepSpeed ZeRO) | **Yes** | `deepspeed_configs/`, `src/axolotl/monkeypatch/deepspeed_utils.py` | ZeRO-1/2/3 communication patterns |
| Tensor Parallel | **Yes** | `src/axolotl/monkeypatch/accelerate/parallelism_config.py` | AllReduce after each MLP and Attention sublayer |
| Context Parallel (Ring Attention) | **Yes** | `src/axolotl/monkeypatch/ring_attn/patch.py` | Ring AllGather across sequence dimension |
| Expert Parallel | **Yes** | `src/axolotl/integrations/expert_parallel/plugin.py`, `shard.py` | AllToAll for token routing to expert GPUs |
| Sequence Parallel | **Yes** | `src/axolotl/utils/ctx_managers/sequence_parallel.py` | AllGather/ReduceScatter for LayerNorm/Dropout regions |
| Pipeline Parallel | **No** | Not implemented | N/A |

### DeviceMesh Topology

Axolotl extends HuggingFace Accelerate's `ParallelismConfig` with Expert Parallel as a first-class mesh axis. The mesh order is:

```
(ep, dp_replicate, dp_shard, cp, sp, tp)
```

**Design rationale** (from `parallelism_config.py:L6`): "The dp axes stay contiguous (required for `_flatten('dp')`)."

The total parallelism size is:
```
total = dp_replicate × dp_shard × tp × cp × sp × ep
```

Key topology configurations from example YAMLs:
- **FSDP2 + TP + CP** (`examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml`): TP within NVLink-connected GPUs, CP across nodes, FSDP across remaining ranks
- **HSDP + TP** (`examples/distributed-parallel/llama-3_1-8b-hsdp-tp.yaml`): Hybrid shard DP (replicate across nodes, shard within node) + TP
- **EP + FSDP** (`examples/expert_parallel/qwen3_30ba3b_ep_fsdp_fft_4gpu.yaml`): Expert parallel with FSDP sharding for non-expert parameters

### Context Parallel Implementation

Ring Attention is implemented in `src/axolotl/monkeypatch/ring_attn/patch.py`:
- Sequence dimension sharded across CP mesh ranks
- SDPA replaced with ring attention kernel (zigzag/stripe patterns from existing analysis docs)
- Integrated with FSDP via flattened `dp_shard_cp` mesh dimension
- Configuration: `context_parallel_size` in YAML

### Expert Parallel Implementation

Expert Parallel via `src/axolotl/integrations/expert_parallel/`:
- `plugin.py`: Registers EP as a plugin, creates EP DeviceMesh sub-group
- `shard.py`: Shards MoE expert layers across EP ranks
- `experts_fn.py`: Custom expert dispatch with AllToAll communication
- Non-expert parameters use standard FSDP sharding across the full world
- Expert parameters are pre-wrapped on `mesh["dp_shard"]` only

### Proposed New Parallelism techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `nd-parallelism` | `src/axolotl/monkeypatch/accelerate/parallelism_config.py` | N-dimensional parallelism composition (EP + DP + TP + CP + SP) via DeviceMesh with configurable mesh order |
| `ring-attention` | `src/axolotl/monkeypatch/ring_attn/patch.py` | Context parallel via ring attention with zigzag/stripe scheduling |
| `expert-parallel` | `src/axolotl/integrations/expert_parallel/plugin.py` | MoE expert sharding across GPU groups with AllToAll dispatch |
| `hybrid-shard` | `deepspeed_configs/`, FSDP2 config | Hybrid sharding: replicate across nodes, shard within node |

---

## Dimension 4: Memory Management

### Memory Component Analysis

| Component | Storage Format | Sharding Strategy | Communication Kernel Triggered |
|-----------|---------------|-------------------|---------------------------------|
| Parameters | BF16/FP8 | FSDP2 (ZeRO-3) or DeepSpeed ZeRO-1/2/3 | AllGather before each layer's forward pass |
| Gradients | BF16 | FSDP2 ReduceScatter | ReduceScatter after each layer's backward pass |
| Optimizer States | FP32 | FSDP2 (sharded) or DeepSpeed ZeRO-1+ | None (local optimizer step after ReduceScatter) |
| Activations | BF16 | Selective recomputation / full recomputation / disk offload | None (stored or recomputed locally) |

### Activation Checkpointing Strategies

| Strategy | Description | Evidence |
|----------|-------------|----------|
| Full recompute | `gradient_checkpointing: true` — recompute all activations during backward (default `use_reentrant: False`) | `src/axolotl/core/builders/base.py:L540-558` |
| TRL activation offloading | `activation_offloading: true` — intercepts activations and moves to CPU during forward, streams back during backward. LoRA-specific path excludes `lm_head` and Liger modules to avoid bandwidth waste | `src/axolotl/core/trainers/mixins/activation_checkpointing.py:L44-236` (`OffloadActivations` from TRL with `use_pin_memory=True`, `use_streams=True`, `max_fwd_stash_size=5`) |
| CPU offload | `CPU_Offloaded_Gradient_Checkpointer` autograd function: `.to("cpu", non_blocking=True)` in forward, `.to("cuda", non_blocking=True)` + re-forward in backward | `src/axolotl/monkeypatch/gradient_checkpointing/offload_cpu.py:L38-72` |
| Disk offload (DISCO) | `DiskOffloadManager` with 4 async save threads + 2 prefetch threads. LIFO tensor deque, per-file locks, states: saving→ready→prefetching→loaded→deleted. Prefetch to GPU via `torch.load(map_location="cuda")` | `src/axolotl/monkeypatch/gradient_checkpointing/offload_disk.py:L43-533` |
| FSDP2 native | `apply_activation_checkpointing` with `CheckpointImpl.NO_REENTRANT` applied before `fully_shard()` | `src/axolotl/monkeypatch/accelerate/fsdp2.py:L341-356` |

### Layer Offloading

`src/axolotl/core/trainers/mixins/layer_offloading.py`: `LayerOffloadManager` (L52-264) offloads frozen (non-trainable) decoder layer parameters to CPU pinned memory. Trainable params (LoRA) stay on GPU.

- **Detection** (`_find_decoder_layers`, L24-44): BFS finds first `nn.ModuleList` with `DecoderLayer` or `TransformerBlock` children
- **Transfer stream**: Dedicated `torch.cuda.Stream` for async H2D prefetch with configurable `num_prefetch` layers
- **Forward hooks** (L185-202): Pre-forward loads layer i to GPU + prefetches i+1..i+num_prefetch; post-forward offloads layer i-1
- **Backward hooks** (L207-229): Pre-backward loads layer i + prefetches i-1..i-num_prefetch; post-backward offloads layer i+1
- Config key: `layer_offloading: true`. Mutually exclusive with `gradient_checkpointing`

### Gradient Accumulation

- Config key: `gradient_accumulation_steps`
- Communication frequency: ReduceScatter triggered every N micro-batches instead of every micro-batch
- Effective batch size: `micro_batch_size × gradient_accumulation_steps × data_parallel_degree`
- Implementation: `src/axolotl/core/builders/base.py`, `src/axolotl/core/trainers/base.py`

### CPU Offload Support

| Offload Type | Config Key | Evidence |
|-------------|------------|----------|
| FSDP2 CPU offload | `fsdp_config.cpu_offload` | `src/axolotl/utils/schemas/fsdp.py` |
| DeepSpeed ZeRO-3 offload | `zero3_bf16_cpuoffload_all.json` | `deepspeed_configs/zero3_bf16_cpuoffload_all.json` |
| Optimizer state offload | `fsdp_config.cpu_offload` | FSDP2 native |
| Activation offload to CPU | `gradient_checkpointing_kwargs` | `src/axolotl/monkeypatch/gradient_checkpointing/offload_cpu.py` |
| Activation offload to disk | config option | `src/axolotl/monkeypatch/gradient_checkpointing/offload_disk.py` |
| ReLoRA CPU offload | `relora_cpu_offload` | `src/axolotl/monkeypatch/relora.py` |

### Memory Profiling

| Feature | Implementation | Evidence |
|---------|---------------|----------|
| PyTorch memory snapshot | `torch.cuda.memory._snapshot()` → `snapshot.pickle` | `src/axolotl/utils/callbacks/profiler.py:L68` |
| Memory history recording | `torch.cuda.memory._record_memory_history()` | `src/axolotl/utils/callbacks/profiler.py:L31` |
| Runtime memory metrics | Peak allocated, reserved memory | `src/axolotl/telemetry/runtime_metrics.py` |
| Benchmarking utilities | GPU memory tracking | `src/axolotl/utils/bench.py` |

---

## Dimension 5: Precision Management

### FP8 Integration (via TorchAO + Accelerate)

Axolotl does **not implement FP8 scaling strategies directly**. FP8 support is provided through:

1. **TorchAO Float8Linear** — Primary FP8 path: `Float8LinearConfig` from `torchao.float8`
2. **Accelerate AORecipeKwargs** — Wraps TorchAO config for HF Trainer integration
3. **BitsAndBytes** — NF4/FP4 quantization for QLoRA (not FP8)

**FP8 activation path** (`src/axolotl/loaders/patch_manager.py:L518`): `PatchManager._apply_fp8_patches()` triggers `patch_create_accelerate_code_for_fp8()` which source-string monkeypatches `Trainer.create_accelerator_and_postprocess` to inject FP8 config into `Accelerator(**args)`.

**FP8 assembly** (`src/axolotl/core/trainers/base.py:L687-707`):
```python
Float8LinearConfig(
    enable_fsdp_float8_all_gather=...,
    force_recompute_fp8_weight_in_bwd=...
)
```
Default scaling strategy: **tensorwise** (no explicit `scaling_type` argument — TorchAO default). Wrapped in `AORecipeKwargs` + `mixed_precision="fp8"`.

**FP8 attention** (`src/axolotl/loaders/patch_manager.py:L265-270`): Separate path via `attn_implementation: "fp8"`. Uses TorchAO's FP8 low-precision SDPA. Requires SM90+ (Hopper) and PyTorch >= 2.11.

### FP8 FSDP2 AllGather

Config keys: `fp8: true` + `fp8_enable_fsdp_float8_all_gather: true` (requires `fsdp_version: 2`)

When enabled:
- Parameters stored and communicated in FP8 E4M3 format
- Halves AllGather communication volume vs BF16
- `force_recompute_fp8_weight_in_bwd=True` — recomputes FP8 weights in backward to save activation memory
- Validation: `check_fp8_config` (L424-452 in `validation.py`) warns without `torch_compile`, raises if not FSDP2
- Claims **10-15% speed improvement** (per config docstring at L570 in `config.py`)

### FP8 Scaling Strategies Found

| Strategy | Class Name | Supported | GPU Support | Evidence |
|----------|-----------|-----------|-------------|----------|
| Delayed Scaling | N/A | Via upstream TransformerEngine | Hopper + Blackwell | Referenced in patch_manager.py |
| Current Scaling | N/A | Via upstream TorchAO | Hopper + Blackwell | Referenced in FP8 FSDP2 integration |
| Block Scaling | N/A | Not directly referenced | Hopper | — |
| MXFP8 | N/A | **Yes** — via quantization schema | Blackwell only | `src/axolotl/utils/schemas/enums.py`, `quantization.py` |

### Precision per Training Component

| Component | Forward Pass | Backward Pass | Optimizer Step | Evidence |
|-----------|-------------|---------------|----------------|----------|
| Linear GEMM inputs | BF16 or FP8 (via TorchAO/TE) | BF16 or FP8 | N/A | Config: `bf16: true` or `fp8: true` |
| Weights | BF16 or FP8 (FSDP AllGather) | N/A | FP32 master copy (with FSDP) | `fsdp_config.enable_fsdp_float8_all_gather` |
| Gradients | N/A | BF16 | N/A | Standard FSDP behavior |
| Adam momentum | N/A | N/A | FP32 | PyTorch default |
| Adam variance | N/A | N/A | FP32 | PyTorch default |
| RMSNorm/LayerNorm | BF16 or FP32 | BF16 or FP32 | N/A | `fsdp_config.fp32_norms` via `src/axolotl/utils/fp32_norms.py` |

### Quantization Support (TorchAO)

`get_quantization_config()` in `src/axolotl/utils/quantization.py:L56` builds `AOBaseConfig`:

| Activation dtype | Weight dtype | Resulting TorchAO Config |
|-----------------|-------------|--------------------------|
| None | int4 | `Int4WeightOnlyConfig(group_size=..., version=2)` |
| int8 | int4 | `Int8DynamicActivationIntxWeightConfig` |
| float8_e4m3fn | float8_e4m3fn | `Float8DynamicActivationFloat8WeightConfig()` (E4M3/E4M3 FP8) |
| float8_e4m3fn | int4 | `Float8DynamicActivationInt4WeightConfig()` |
| None | nvfp4 | `NVFP4WeightOnlyConfig()` (group_size must be 16) |
| None | mxfp4 | `MXDynamicActivationMXWeightConfig(activation_dtype=float4_e2m1fn_x2, weight_dtype=float4_e2m1fn_x2, block_size=32)` |

**QAT path** (`quantization.py:L360`): `Float8FakeQuantizeConfig`, `IntxFakeQuantizeConfig`, `MXFakeQuantizeConfig`

**PTQ path** (`quantization.py:L202`): `torchao.quantization.quantize_()`

**Additional quantization methods:**

| Method | Config Key | Evidence |
|--------|------------|----------|
| QLoRA (NF4) | `adapter: qlora` | `src/axolotl/loaders/model.py`, `monkeypatch/fsdp2_qlora.py` |
| MoE expert quantization | `quantize_moe_experts` | `src/axolotl/monkeypatch/moe_quant.py` — patches `transformers.core_model_loading.set_param_for_module` for on-the-fly 4-bit/8-bit expert weight quantization during model loading |
| LLM Compressor (PTQ) | `plugins: [llm_compressor]` | `src/axolotl/integrations/llm_compressor/` |
| FP8 optimizer states | `optimizer: ao_adamw_fp8` | `TorchAOQuantDType.float8_e4m3fn` enum; `OptimStateFp8` from `torchao.optim.subclass_fp8` |

**`TorchAOQuantDType` enum** (`schemas/enums.py:L8`): `int4`, `int8`, `float8_e4m3fn` (aliases: `fp8`, `float8`), `nvfp4`, `mxfp4`. No E5M2 support.

### FP32 Norm Preservation

`src/axolotl/utils/fp32_norms.py`: Keeps RMSNorm and LayerNorm parameters in FP32 when training in BF16/FP8. Config key: `fsdp_config.fp32_norms`. Prevents numerical instability in normalization layers.

### FP8 Communication Integration

- [x] FP8 AllGather in FSDP2: **Yes** — `enable_fsdp_float8_all_gather` config key
- [ ] FP8 ReduceScatter: **No** — gradients communicated in BF16
- [ ] NVLink-SHARP FP8 in-switch reduction: **No** — no SHARP integration
- Estimated communication volume reduction vs BF16: **~50%** (parameters only, via FP8 AllGather)

### Proposed New Precision techniques

| Tag | Evidence | Description |
|-----|----------|-------------|
| `fp8-fsdp-allgather` | `src/axolotl/loaders/patch_manager.py` | FP8 parameter AllGather in FSDP2 halving communication volume |
| `fp32-norm-preservation` | `src/axolotl/utils/fp32_norms.py` | Keep normalization layers in FP32 during mixed-precision training |
| `moe-expert-quant` | `src/axolotl/monkeypatch/moe_quant.py` | Selective quantization of MoE expert weights |

---

## Dimension 6: Profiling and Observability

### Built-in Profiling Capabilities

| Feature | Supported | Integration Method | Evidence |
|---------|-----------|-------------------|----------|
| PyTorch Profiler (Chrome trace) | **Yes** | `PytorchProfilerCallback` with CPU+CUDA activities, shape recording, memory profiling, stack traces | `src/axolotl/utils/callbacks/profiler.py:L17` |
| CUDA Memory Snapshots | **Yes** | `torch.cuda.memory._snapshot()` → pickle file | `src/axolotl/utils/callbacks/profiler.py:L68` |
| Memory History Recording | **Yes** | `torch.cuda.memory._record_memory_history(enabled="all", stacks="all")` | `src/axolotl/utils/callbacks/profiler.py:L31` |
| Tokens/Second Metrics | **Yes** | `TokensPerSecondCallback` with TP/CP-aware counting, checkpoint resume support | `src/axolotl/utils/callbacks/tokens_per_second.py:L21` |
| NVTX annotations | **No** | No `torch.cuda.nvtx` usage | — |
| NCCL Inspector plugin | **No** | No NCCL profiler integration | — |
| Flight Recorder | **No** | No PyTorch Flight Recorder usage | — |
| MFU metrics | **No** | No Model FLOPs Utilization calculation | — |

### Telemetry System

Axolotl has a built-in telemetry system for observability:

| Component | File | Description |
|-----------|------|-------------|
| Telemetry Manager | `src/axolotl/telemetry/manager.py` | Central telemetry orchestration |
| Telemetry Callbacks | `src/axolotl/telemetry/callbacks.py` | Training event telemetry hooks |
| Runtime Metrics | `src/axolotl/telemetry/runtime_metrics.py` | GPU memory, step timing metrics |
| Error Telemetry | `src/axolotl/telemetry/errors.py` | Error reporting and diagnostics |

### Third-Party Profiling Integrations

| Integration | Config Key | Evidence |
|-------------|------------|----------|
| SwanLab profiling | `plugins: [swanlab]` | `src/axolotl/integrations/swanlab/profiling.py` |
| TensorBoard | `logging_dir` | Via HuggingFace Trainer |
| Weights & Biases | `wandb_*` config keys | `src/axolotl/utils/wandb_.py` |
| Comet ML | `comet_*` config keys | `src/axolotl/utils/comet_.py`, `callbacks/comet_.py` |
| MLflow | `mlflow_*` config keys | `src/axolotl/utils/mlflow_.py`, `callbacks/mlflow_.py` |
| OpenTelemetry | `opentelemetry_*` config keys | `src/axolotl/utils/callbacks/opentelemetry.py` |
| TrackIO | config keys | `src/axolotl/utils/trackio_.py` |

### Profile Analysis Script

`scripts/analyze_profile.py` (1519 lines): Comprehensive post-hoc CLI tool for profiler output analysis.

**Input files**: `profiler_trace.json` (Chrome trace) + `snapshot.pickle` (CUDA memory snapshot)

**Analysis modules:**

| Function | What It Analyzes |
|----------|-----------------|
| `analyze_trace(events)` (L184) | CUDA kernel time by category; 20 categories including ScatterMoE, BnB Dequant, Flash Attention, CCE Loss, LoRA kernels, DoRA, GEMM/CUTLASS |
| `analyze_cpu_overhead(events)` (L651) | Wall clock vs CUDA time (GPU utilization), memcpy by direction, checkpoint recompute overhead |
| `analyze_snapshot(snapshot)` (L844) | Reserved/allocated/fragmentation; active blocks; allocation churn; Python frame attribution |
| `analyze_peak_memory(snapshot)` (L975) | Peak concurrent allocated bytes with live allocation breakdown |
| `analyze_fragmentation(snapshot)` (L1079) | Segment size distribution; inactive gaps; `expandable_segments` recommendation |
| `analyze_allocation_churn(snapshot)` (L565) | Top churn sizes attributed to Python source (BnB dequant, GC recompute, ScatterMoE, optimizer) |
| `compare_traces(before, after)` (L458) | A/B comparison with per-category speedup ratios |
| `analyze_scaling(mem_a, mem_b)` (L1234) | Detects which tensor categories scale with sequence length |

**Features**: Streaming support via `ijson` for large traces (>0.5 GB). CLI flags: `--compare`, `--include-warmup`, `--memory-only`, `--quick`, `--gpu-gb`.

### Recommended Profiling Dimensions for KernelWiki

| Dimension | What It Measures | Tool | Key Metrics |
|-----------|-----------------|------|-------------|
| GPU memory lifecycle | Allocation/deallocation patterns across training steps | PyTorch memory snapshot (`profiler_callback`) | peak_allocated_bytes, allocation timeline |
| Step timing | Wall-clock time per training step | Tokens/Second callback | tokens_per_second, step_time |
| Communication overhead | Time spent in NCCL collectives vs compute | External nsys (not built-in) | comm_time / total_time |
| Activation memory peak | Effect of checkpointing strategy on peak GPU memory | PyTorch memory profiler | peak with/without checkpointing |

---

## Synthesis: Expansion Decision Summary

### S.1 Library Classification

| Property | Value |
|----------|-------|
| Library | Axolotl |
| GitHub URL | axolotl-ai-cloud/axolotl |
| Type | **training-orchestration** |
| Contains CUDA Kernels | **No** (Triton only, ~15 files) |
| Primary Knowledge Dimensions | Dim 3 (Parallelism), Dim 4 (Memory), Dim 1 (Kernel orchestration) |
| Recommended KernelWiki Priority | **P1** (important orchestration framework — does not provide core kernels but composes them from 10+ upstream libraries with sophisticated parallelism and memory strategies) |

### S.2 Proposed Tags (for controlled vocabulary YAML)

```yaml
kernel_types:
  # New from Axolotl (Triton kernels)
  - fused-swiglu          # SwiGLU forward+backward fused activation
  - fused-geglu           # GeGLU forward+backward fused activation
  - fused-lora            # LoRA linear + activation fused with quantization
  - dora-norm             # DoRA magnitude-norm-scale without B@A materialization
  - selective-logsoftmax  # Selective log_softmax at target index (EBFT)
  - fused-reinforce-loss  # Masked REINFORCE loss reduction (EBFT)
  - fused-cosine-sim      # Batched cosine similarity fused (EBFT)
  - scattermoe-lora       # LoRA-adapted MoE expert dispatch via scatter/gather
  - fused-rmsnorm-gated   # Fused RMSNorm + SiLU gate (Qwen3.5)
  - fused-rope            # Fused rotary position embedding (Gemma4)

techniques:
  # New from Axolotl
  - nd-parallelism            # N-dimensional parallelism composition via DeviceMesh
  - ring-attention            # Context parallel via ring attention scheduling
  - expert-parallel           # MoE expert sharding with AllToAll dispatch
  - fsdp2-prefetch            # FSDP2 backward/forward prefetch overlap
  - async-grpo-generation     # Async vLLM generation overlapped with training
  - fp8-fsdp-allgather        # FP8 parameter AllGather in FSDP2
  - fp32-norm-preservation    # Keep normalization layers in FP32
  - moe-expert-quant          # Selective quantization of MoE expert weights
  - activation-disk-offload   # Activation checkpointing with disk offload
  - layer-cpu-offload         # Full layer offload to CPU between passes
  - kernel-autotune           # Runtime autotuning of Triton kernel parameters

hardware_features:
  # No new hardware features — Axolotl does not exploit hardware directly

source_categories:
  - training-orchestration  # Config-driven training framework composing upstream kernels
```

### S.3 Wiki Page Topics

| # | Wiki Subdirectory | Proposed Page ID | Title | Source Evidence | Related Existing KernelWiki Pages |
|---|-------------------|------------------|-------|----------------|-----------------------------------|
| 1 | training/ | `training-fused-lora-triton` | Fused LoRA Triton Kernels: Forward+Backward with Quantization | `src/axolotl/kernels/lora.py` (67KB) | lora-adaptation, quantization |
| 2 | training/ | `training-ebft-kernels` | EBFT Triton Kernels: Selective LogSoftmax and REINFORCE Loss | `src/axolotl/core/trainers/ebft/kernels.py` | reinforcement-learning, loss-computation |
| 3 | training/ | `training-scattermoe-lora` | ScatterMoE LoRA: Triton Expert Dispatch with Adapter Fusion | `src/axolotl/integrations/kernels/libs/scattermoe_lora/` | moe-routing, lora-adaptation |
| 4 | parallelism/ | `parallel-nd-mesh` | N-Dimensional Parallelism via DeviceMesh (EP+DP+TP+CP+SP) | `src/axolotl/monkeypatch/accelerate/parallelism_config.py` | tensor-parallel, data-parallel |
| 5 | parallelism/ | `parallel-expert-fsdp` | Expert Parallel Integration with FSDP2 | `src/axolotl/integrations/expert_parallel/` | moe-routing, fsdp |
| 6 | training/ | `training-activation-offload` | Activation Checkpointing: CPU and Disk Offload Strategies | `src/axolotl/monkeypatch/gradient_checkpointing/` | activation-checkpointing |
| 7 | training/ | `training-fp8-fsdp2` | FP8 AllGather in FSDP2: Halving Communication Volume | `src/axolotl/loaders/patch_manager.py` | fp8-training, fsdp |

### S.4 Repository Mappings (slug -> org/repo)

```python
# For the PR candidate search script
"axolotl": "axolotl-ai-cloud/axolotl",

# For the PR page generation script
"axolotl": "axolotl-ai-cloud/axolotl",
```

### S.5 Keyword-to-Tag Mappings (for automated PR tagger)

```python
# keyword -> kernel_type tag
"swiglu": "fused-swiglu",
"geglu": "fused-geglu",
"lora_kernel": "fused-lora",
"lora_ops": "fused-lora",
"dora": "dora-norm",
"ebft": "fused-reinforce-loss",
"log_softmax_gather": "selective-logsoftmax",
"cosine_similarity": "fused-cosine-sim",
"scattermoe": "scattermoe-lora",
"rms_norm_gated": "fused-rmsnorm-gated",
"fused_rope": "fused-rope",

# keyword -> technique tag
"fsdp2": "nd-parallelism",
"ring_attn": "ring-attention",
"ring_attention": "ring-attention",
"context_parallel": "ring-attention",
"expert_parallel": "expert-parallel",
"async_grpo": "async-grpo-generation",
"float8_allgather": "fp8-fsdp-allgather",
"fp8_allgather": "fp8-fsdp-allgather",
"fp32_norms": "fp32-norm-preservation",
"moe_quant": "moe-expert-quant",
"offload_disk": "activation-disk-offload",
"layer_offloading": "layer-cpu-offload",
"autotune": "kernel-autotune",
"backward_prefetch": "fsdp2-prefetch",

# keyword -> hardware_feature tag
# (none — Axolotl does not exploit hardware features directly)
```

### S.6 PR Search Keywords (for candidate ledger)

```yaml
keywords_used:
  - triton
  - kernel
  - fused
  - lora_kernel
  - dora
  - swiglu
  - geglu
  - ebft
  - scattermoe
  - ring_attn
  - context_parallel
  - expert_parallel
  - fsdp2
  - tensor_parallel
  - fp8
  - float8
  - mxfp8
  - quantize
  - activation_checkpoint
  - gradient_checkpointing
  - offload
  - profiler
  - liger
  - cut_cross_entropy
  - flash_attn
```

### S.7 Inclusion Policy Lane

```yaml
training-orchestration:
  description: |
    Axolotl is a config-driven training orchestration framework. Capture PRs that
    modify Triton kernels, parallelism configuration, FSDP/DeepSpeed integration,
    memory management strategies, precision management, or upstream kernel integration
    plugins. Skip pure-config, documentation, or CI changes.
  capture_criteria:
    - changed_paths_match:
        - "src/axolotl/kernels/**"
        - "src/axolotl/integrations/kernels/**"
        - "src/axolotl/core/trainers/ebft/kernels.py"
        - "src/axolotl/monkeypatch/accelerate/**"
        - "src/axolotl/monkeypatch/ring_attn/**"
        - "src/axolotl/monkeypatch/attention/**"
        - "src/axolotl/monkeypatch/fsdp2_qlora.py"
        - "src/axolotl/monkeypatch/lora_kernels.py"
        - "src/axolotl/monkeypatch/moe_quant.py"
        - "src/axolotl/monkeypatch/gradient_checkpointing/**"
        - "src/axolotl/integrations/expert_parallel/**"
        - "src/axolotl/integrations/liger/**"
        - "src/axolotl/integrations/cut_cross_entropy/**"
        - "src/axolotl/utils/fp32_norms.py"
        - "src/axolotl/utils/quantization.py"
    - title_contains_any:
        - kernel
        - triton
        - fused
        - lora_kernel
        - dora
        - fsdp
        - fsdp2
        - tensor_parallel
        - context_parallel
        - expert_parallel
        - ring_attn
        - fp8
        - float8
        - mxfp8
        - quantize
        - liger
        - flash_attn
        - activation_checkpoint
        - offload
        - scattermoe
  skip_criteria:
    - changed_paths_match_only:
        - "docs/**"
        - "examples/**"
        - "tests/**"
        - ".github/**"
        - "*.md"
        - ".claude/**"
        - ".codex/**"
    - pure_config_only: true
```

### S.8 Schema Extensions (if any)

Proposed new optional frontmatter fields for Wiki pages from this library:

- `scope: orchestration` — distinguishes orchestration-level knowledge from kernel-level
- `upstream_providers: [liger, flash-attention, torchao, ...]` — lists kernel providers for orchestration pages
- `parallelism_dimensions: [dp, tp, cp, ep, sp]` — applicable parallelism dimensions
- `memory_strategy: [fsdp2, deepspeed-zero3, cpu-offload, disk-offload]` — memory optimization category

### S.9 Hardware Features Relevant to This Library's Training Workloads

| Hardware Feature | Inference Relevance | Training Relevance | Specific Impact on Axolotl |
|-----------------|--------------------|--------------------|---------------------------|
| NVLink 5 (1.8 TB/s) | Partial | Core | Doubles FSDP2 AllGather/ReduceScatter bandwidth for gradient sync |
| NVSwitch 4 (NVL72) | Partial | Core | Enables larger-scale ND parallelism with Axolotl's DeviceMesh |
| NVLink-SHARP FP8 | No | Core | Would reduce FP8 AllGather bandwidth by 4x (not yet integrated) |
| Symmetric Memory | Partial | Core | Would improve small-message AllReduce in TP (not yet integrated) |
| Copy Engine | Partial | Core | Would free SMs for compute during FSDP AllGather (not yet integrated) |
| MXFP8 hardware (Blackwell) | Yes | Core | Would enable hardware-native FP8 scaling via TorchAO/TE (schema support exists) |
| 192 MB L2 Cache | Partial | Beneficial | Improves Triton kernel (LoRA, SwiGLU, EBFT) cache hit rates |
| 192 GB HBM3e | Partial | Core | Enables larger models per GPU, reduces FSDP sharding requirements |

### S.10 Upstream/Downstream Dependencies to Also Track

| Slug | GitHub URL | Relationship | Justification |
|------|-----------|-------------|---------------|
| `liger-kernel` | linkedin/Liger-Kernel | kernel-provider | Primary fused kernel provider: cross-entropy, RMSNorm, RoPE, SwiGLU for 15+ model families |
| `flash-attention` | Dao-AILab/flash-attention | kernel-provider | Flash Attention 2/3 for memory-efficient attention |
| `torchao` | pytorch/ao | kernel-provider | FP8 training via Float8Linear, quantized optimizers |
| `cut-cross-entropy` | apple/ml-cross-entropy | kernel-provider | Memory-efficient chunked cross-entropy loss |
| `transformer-engine` | NVIDIA/TransformerEngine | kernel-provider | FP8 GEMM, FP8 LayerNorm for FSDP2 FP8 AllGather |
| `bitsandbytes` | bitsandbytes-foundation/bitsandbytes | kernel-provider | NF4/FP4 quantization for QLoRA, 8-bit optimizers |
| `deepspeed` | microsoft/DeepSpeed | runtime-dependency | ZeRO optimizer, fused kernels, distributed training backend |
| `accelerate` | huggingface/accelerate | runtime-dependency | FSDP2 wrapper, distributed launch, parallelism config |
| `trl` | huggingface/trl | runtime-dependency | DPO, GRPO, KTO trainer base classes |
| `xformers` | facebookresearch/xformers | kernel-provider | Memory-efficient attention alternative |

---

## Library Type Adaptation Rationale

Axolotl was classified as an **orchestration framework** based on Dimension 1 findings:
- **Zero CUDA/C++ kernel files** — no native GPU kernel implementations
- **~15 Triton kernel files** — thin layer of custom kernels for specialized operations (LoRA fusion, EBFT loss, MoE dispatch)
- **10+ upstream kernel providers** — all core GPU compute (GEMM, attention, optimizer, loss) delegated to external libraries

Dimension emphasis was adjusted accordingly:
- **Dimension 1**: Replaced kernel-by-kernel analysis with **dependency graph analysis** showing 10 upstream kernel providers and their integration points
- **Dimension 2**: **Light treatment** — Axolotl has no communication kernel implementations; documented PyTorch Distributed usage patterns only
- **Dimension 3**: **Deep analysis** — Axolotl's primary contribution is sophisticated N-dimensional parallelism composition (EP+DP+TP+CP+SP) via extended DeviceMesh
- **Dimension 4**: **Deep analysis** — Rich memory management with 4 activation checkpointing strategies (full, selective, CPU, disk), layer offloading, and ZeRO-1/2/3
- **Dimension 5**: **Moderate analysis** — FP8 support via upstream integration (TorchAO, TransformerEngine) rather than native implementation; MXFP8 schema support
- **Dimension 6**: **Moderate analysis** — Built-in PyTorch profiler callback and telemetry system; 7 third-party integrations (SwanLab, W&B, Comet, MLflow, OpenTelemetry, TensorBoard, TrackIO)
