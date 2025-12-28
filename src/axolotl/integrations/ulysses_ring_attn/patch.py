"""Attention patching for Ulysses + Ring-Attention.

This module patches the model's attention forward pass to wrap it with:
1. Ulysses all-to-all (scatter seq, gather heads) - if sp > 1
2. Ring-attention or local flash-attn - always
3. Reverse all-to-all (gather seq, scatter heads) - if sp > 1

Implements attention monkey-patching for Llama-style models in Phase 2.1.

References:
    - ms-swift: swift/trainers/sequence_parallel/ulysses.py (DistributedAttention)
    - Spec: specs/006-ulysses-ring-attention-plugin/README.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


# Global instance of DistributedAttention for use in patched forward methods
_DISTRIBUTED_ATTENTION: DistributedAttention | None = None


def set_distributed_attention(distributed_attn: DistributedAttention):
    """Set the global DistributedAttention instance used by patched attention."""
    global _DISTRIBUTED_ATTENTION
    _DISTRIBUTED_ATTENTION = distributed_attn
    LOG.info("Set global DistributedAttention instance for attention patching")


def get_distributed_attention() -> DistributedAttention:
    """Get the global DistributedAttention instance."""
    if _DISTRIBUTED_ATTENTION is None:
        raise RuntimeError(
            "DistributedAttention not initialized. "
            "Call set_distributed_attention() before patching attention."
        )
    return _DISTRIBUTED_ATTENTION


class DistributedAttention:
    """Wrapper for Ulysses + Ring-Attention distributed attention.

    Orchestrates the full distributed attention pipeline:
    1. Ulysses all-to-all: [B,L/W,H,D] → [B,L/rp,H/sp,D]
    2. Ring-attention: Compute attention over full sequence via ring comm
    3. Reverse all-to-all: [B,L/rp,H/sp,D] → [B,L/W,H,D]

    Placeholder for Phase 1.5.
    """

    def __init__(
        self,
        sp_group: ProcessGroup,
        rp_group: ProcessGroup,
        sp_size: int,
        rp_size: int,
        rp_rank: int,
        require_padding_free: bool = True,
    ):
        """Initialize distributed attention wrapper.

        Args:
            sp_group: Sequence parallel (Ulysses) process group
            rp_group: Ring parallel process group
            sp_size: Sequence parallel world size
            rp_size: Ring parallel world size
            rp_rank: This rank's position in ring parallel group [0, rp_size)
            require_padding_free: If True, enforce Phase 1 constraint (seq_len % 2*rp == 0).
                                 If False, enable auto-padding (Phase 2.3)
        """
        self.sp_group = sp_group
        self.rp_group = rp_group
        self.sp_size = sp_size
        self.rp_size = rp_size
        self.rp_rank = rp_rank
        self.require_padding_free = require_padding_free

        LOG.warning(
            f"[DistAttn] INIT: sp_size={sp_size}, rp_size={rp_size}, rp_rank={rp_rank}, "
            f"auto_padding={'disabled' if require_padding_free else 'enabled'}"
        )

    def forward(
        self,
        q,
        k,
        v,
        cu_seqlens,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    ):
        """Distributed attention forward pass.

        Orchestrates the full Ulysses + Ring-Attention pipeline:
        1. Ulysses all-to-all (if sp_size > 1): scatter seq, gather heads
        2. Ring-attention (if rp_size > 1) or local flash-attn (if rp_size == 1)
        3. Reverse all-to-all (if sp_size > 1): gather seq, scatter heads

        Args:
            q: Query tensor [batch, seq_local, num_heads, head_dim]
            k: Key tensor [batch, seq_local, num_heads, head_dim]
            v: Value tensor [batch, seq_local, num_heads, head_dim]
            cu_seqlens: Global cumulative sequence lengths [num_seqs + 1] (pre-slicing).
            dropout_p: Dropout probability (default: 0.0)
            softmax_scale: Softmax scale factor (default: 1/sqrt(head_dim))
            causal: Whether to apply causal masking (default: True)
            window_size: Sliding window size (default: (-1, -1) = no window)
            alibi_slopes: ALiBi slopes for positional bias (default: None)
            deterministic: Whether to use deterministic attention (default: False)

        Returns:
            Attention output [batch, seq_local, num_heads, head_dim]
        """
        from axolotl.integrations.ulysses_ring_attn.ulysses_all2all import (
            ulysses_all_to_all_gather_scatter,
            ulysses_all_to_all_scatter_gather,
        )

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        # Ensure cu_seqlens is on the right device/dtype for downstream kernels.
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32)

        # FlashAttention (and ring-flash-attn) only support fp16/bf16 inputs.
        # PEFT/LayerNorm/RoPE can silently promote Q/K/V to fp32, so cast back
        # to a supported dtype for kernel compatibility.
        desired_dtype = None
        for tensor in (q, k, v):
            if tensor.dtype in (torch.float16, torch.bfloat16):
                desired_dtype = tensor.dtype
                break
        if desired_dtype is None:
            if q.is_cuda and hasattr(torch.cuda, "is_bf16_supported"):
                desired_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                desired_dtype = torch.float16

        if (
            q.dtype != desired_dtype
            or k.dtype != desired_dtype
            or v.dtype != desired_dtype
        ):
            q = q.to(desired_dtype)
            k = k.to(desired_dtype)
            v = v.to(desired_dtype)

        # === Step 1: Ulysses all-to-all (if sp_size > 1) ===
        # Transform: [B, L/W, H, D] → [B, L/rp, H/sp, D]
        # Where W = sp * rp (total world size)

        LOG.warning(
            f"[DistAttn] Input: q.shape={q.shape}, cu_seqlens[-1]={int(cu_seqlens[-1].item())}"
        )

        if self.sp_size > 1:
            LOG.warning(
                f"Applying Ulysses all-to-all: "
                f"seq_len={q.shape[1]} → {q.shape[1] * self.sp_size}, "
                f"num_heads={q.shape[2]} → {q.shape[2] // self.sp_size}"
            )
            q = ulysses_all_to_all_scatter_gather(
                self.sp_group, q, scatter_idx=2, gather_idx=1
            )
            k = ulysses_all_to_all_scatter_gather(
                self.sp_group, k, scatter_idx=2, gather_idx=1
            )
            v = ulysses_all_to_all_scatter_gather(
                self.sp_group, v, scatter_idx=2, gather_idx=1
            )

            LOG.warning(f"[DistAttn] After Ulysses all-to-all: q.shape={q.shape}")

        # === Step 2: Ring-attention or local flash-attn ===

        if self.rp_size > 1:
            # Use ring-flash-attn (llama3 varlen) for distributed sequence computation.
            LOG.debug(
                f"Applying ring-attention with rp_size={self.rp_size}, rp_rank={self.rp_rank}"
            )

            try:
                from ring_flash_attn import (
                    llama3_flash_attn_prepare_cu_seqlens,
                    llama3_flash_attn_varlen_func,
                )
            except ImportError as e:
                raise ImportError(
                    "ring-flash-attn not installed. Install with: pip install ring-flash-attn"
                ) from e

            # ring-flash-attn varlen APIs follow flash-attn's convention:
            #   q/k/v: [total_tokens, nheads, headdim]
            # Axolotl sample packing produces batch_size==1, so we squeeze/unsqueeze.
            if q.shape[0] != 1:
                raise ValueError(
                    "UlyssesRingAttentionPlugin requires batch_size==1 when using "
                    "ring-flash-attn varlen kernels (sample_packing).\n\n"
                    f"Got q.shape={tuple(q.shape)}."
                )

            q_varlen = q.squeeze(0).contiguous()
            k_varlen = k.squeeze(0).contiguous()
            v_varlen = v.squeeze(0).contiguous()
            if q_varlen.ndim != 3 or k_varlen.ndim != 3 or v_varlen.ndim != 3:
                raise ValueError(
                    "ring-flash-attn varlen requires 3D q/k/v tensors: [total_tokens, nheads, headdim].\n\n"
                    f"Got q={tuple(q_varlen.shape)}, k={tuple(k_varlen.shape)}, v={tuple(v_varlen.shape)}."
                )
            if k_varlen.shape != v_varlen.shape:
                raise ValueError(
                    "ring-flash-attn varlen requires k and v to have identical shapes.\n\n"
                    f"Got k={tuple(k_varlen.shape)}, v={tuple(v_varlen.shape)}."
                )

            (
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                local_k_slice,
            ) = llama3_flash_attn_prepare_cu_seqlens(
                cu_seqlens, causal, self.rp_rank, self.rp_size
            )

            attn_output = llama3_flash_attn_varlen_func(
                q_varlen,
                k_varlen,
                v_varlen,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                heads_k_stride=1,
                local_k_slice=local_k_slice,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                group=self.rp_group,
            ).unsqueeze(0)

        else:
            # rp_size == 1: Use local flash-attn (no ring communication)
            LOG.debug("Using local flash-attn (rp_size=1)")

            try:
                from flash_attn import flash_attn_varlen_func
            except ImportError as e:
                raise ImportError(
                    "flash-attn not installed. Install with: pip install flash-attn"
                ) from e

            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_output = flash_attn_varlen_func(
                q.squeeze(0).contiguous(),
                k.squeeze(0).contiguous(),
                v.squeeze(0).contiguous(),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
            ).unsqueeze(0)

        # === Step 3: Reverse all-to-all (if sp_size > 1) ===
        # Transform back: [B, L/rp, H/sp, D] → [B, L/W, H, D]

        if self.sp_size > 1:
            LOG.warning("Applying reverse Ulysses all-to-all")
            attn_output = ulysses_all_to_all_gather_scatter(
                self.sp_group, attn_output, scatter_idx=1, gather_idx=2
            )
            LOG.warning(
                f"[DistAttn] After reverse all-to-all: attn_output.shape={attn_output.shape}"
            )

        LOG.warning(f"[DistAttn] Final output: attn_output.shape={attn_output.shape}")

        return attn_output


def pad_sequences_to_multiple(
    tensor, cu_seqlens, target_multiple, dim=1, pad_value=0.0
):
    """Pad sequences to be divisible by target_multiple.

    Args:
        tensor: Input tensor to pad (typically [batch, seq_len, ...])
        cu_seqlens: Cumulative sequence lengths [num_seqs + 1]
        target_multiple: Target divisor (typically 2 * rp_size)
        dim: Dimension to pad along (default: 1 = sequence dimension)
        pad_value: Value to use for padding (default: 0.0)

    Returns:
        padded_tensor: Tensor padded to make all sequences divisible by target_multiple
        padded_cu_seqlens: Updated cumulative sequence lengths after padding
        padding_mask: Boolean mask [total_padded_seqlen] where True = original data, False = padding

    Example:
        >>> # Sequence of length 1000, need to pad to multiple of 128
        >>> tensor = torch.randn(1, 1000, 32, 128)
        >>> cu_seqlens = torch.tensor([0, 1000])
        >>> padded, new_cu, mask = pad_sequences_to_multiple(tensor, cu_seqlens, 128)
        >>> padded.shape
        torch.Size([1, 1024, 32, 128])  # Padded to 1024 (next multiple of 128)
        >>> mask.sum().item()
        1000  # Original 1000 positions are True
    """
    import torch

    num_seqs = len(cu_seqlens) - 1
    seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]

    # Compute padding needed for each sequence
    padding_per_seq = []
    for seq_len in seq_lengths:
        remainder = seq_len.item() % target_multiple
        if remainder == 0:
            padding_per_seq.append(0)
        else:
            padding_needed = target_multiple - remainder
            padding_per_seq.append(padding_needed)

    total_padding = sum(padding_per_seq)

    if total_padding == 0:
        # No padding needed
        total_seqlen = cu_seqlens[-1].item()
        padding_mask = torch.ones(total_seqlen, dtype=torch.bool, device=tensor.device)
        return tensor, cu_seqlens, padding_mask

    # Build padding mask (True = original data, False = padding)
    original_seqlen = cu_seqlens[-1].item()
    padded_seqlen = original_seqlen + total_padding
    padding_mask = torch.zeros(padded_seqlen, dtype=torch.bool, device=tensor.device)

    # Build padded tensor by copying sequences and adding padding
    padded_chunks = []
    current_pos = 0

    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        pad_len = padding_per_seq[i]

        # Extract original sequence
        if dim == 1:
            seq_chunk = tensor[:, start:end, ...]
        else:
            raise NotImplementedError(f"Padding only supports dim=1, got dim={dim}")

        padded_chunks.append(seq_chunk)

        # Set mask for original data
        padding_mask[current_pos : current_pos + seq_len] = True
        current_pos += seq_len

        # Add padding if needed
        if pad_len > 0:
            pad_shape = list(seq_chunk.shape)
            pad_shape[dim] = pad_len
            pad_chunk = torch.full(
                pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device
            )
            padded_chunks.append(pad_chunk)
            # padding_mask already initialized to False for padding positions
            current_pos += pad_len

    # Concatenate all chunks
    padded_tensor = torch.cat(padded_chunks, dim=dim)

    # Compute new cu_seqlens
    padded_seq_lengths = (
        seq_lengths + torch.tensor(padding_per_seq, device=cu_seqlens.device)
    ).to(cu_seqlens.dtype)
    padded_cu_seqlens = torch.cumsum(
        torch.cat(
            [
                torch.tensor([0], device=cu_seqlens.device, dtype=cu_seqlens.dtype),
                padded_seq_lengths,
            ]
        ),
        dim=0,
    )

    return padded_tensor, padded_cu_seqlens, padding_mask


def unpad_sequences(tensor, padding_mask, dim=1):
    """Remove padding from tensor using padding mask.

    Args:
        tensor: Padded tensor [batch, padded_seq_len, ...]
        padding_mask: Boolean mask [padded_seq_len] where True = keep, False = remove
        dim: Dimension to unpad along (default: 1)

    Returns:
        Unpadded tensor with only original (non-padded) positions

    Example:
        >>> padded = torch.randn(1, 1024, 32, 128)
        >>> mask = torch.cat([torch.ones(1000, dtype=torch.bool), torch.zeros(24, dtype=torch.bool)])
        >>> unpadded = unpad_sequences(padded, mask, dim=1)
        >>> unpadded.shape
        torch.Size([1, 1000, 32, 128])
    """
    if dim == 1:
        return tensor[:, padding_mask, ...]
    else:
        raise NotImplementedError(f"Unpadding only supports dim=1, got dim={dim}")


def compute_zigzag_indices(cu_seqlens, rp_size, rp_rank, require_padding_free=True):
    """Compute zigzag sequence splitting indices for load balancing.

    In causal attention, later chunks have larger KV caches (more tokens to attend to).
    Zigzag splitting pairs early chunks (small KV) with late chunks (large KV) to
    balance workload across ranks.

    Pattern for rp_size=4, chunks=[0,1,2,3,4,5,6,7]:
        Rank 0: [0, 7]  # Early + Late (balanced)
        Rank 1: [1, 6]
        Rank 2: [2, 5]
        Rank 3: [3, 4]

    Args:
        cu_seqlens: Cumulative sequence lengths [num_seqs + 1]
            Example: [0, 1024, 2048, 3072] for 3 sequences of length 1024 each
        rp_size: Ring parallel world size
        rp_rank: This rank's position in ring [0, rp_size)

    Returns:
        Boolean mask tensor [total_seqlen] indicating which positions this rank handles

    Raises:
        AssertionError: If sequences are not divisible by 2*rp_size (Phase 1 constraint)

    Example:
        >>> cu_seqlens = torch.tensor([0, 1024, 2048])  # 2 sequences of length 1024
        >>> rp_size = 2
        >>> rp_rank = 0
        >>> mask = compute_zigzag_indices(cu_seqlens, rp_size, rp_rank)
        >>> # Rank 0 gets chunks [0, 3] from each sequence:
        >>> # Seq 0: [0:256, 768:1024], Seq 1: [1024:1280, 1792:2048]
    """
    import torch

    num_seqs = len(cu_seqlens) - 1
    total_seqlen = cu_seqlens[-1].item()

    # Initialize boolean mask
    mask = torch.zeros(total_seqlen, dtype=torch.bool, device=cu_seqlens.device)

    # Process each sequence independently
    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start

        # Check divisibility constraint
        if require_padding_free and seq_len % (2 * rp_size) != 0:
            # Phase 1 constraint: sequence must be divisible by 2*rp_size
            raise AssertionError(
                f"Sequence {i} length {seq_len} must be divisible by 2*rp_size={2 * rp_size}. "
                f"Set ulysses_ring_attention_require_padding_free=false to enable auto-padding (Phase 2.3)."
            )

        # Phase 2.3: sequences should already be padded if require_padding_free=False
        # If not divisible at this point with auto-padding enabled, something went wrong
        if not require_padding_free and seq_len % (2 * rp_size) != 0:
            import warnings

            warnings.warn(
                f"Sequence {i} length {seq_len} is not divisible by 2*rp_size={2 * rp_size} "
                "even with auto-padding enabled.",
                UserWarning,
                stacklevel=2,
            )

        # Split sequence into 2*rp_size chunks
        chunk_size = seq_len // (2 * rp_size)

        # Zigzag pattern: get chunks [rp_rank, 2*rp_size - 1 - rp_rank]
        # Front chunk (early in sequence, small KV cache)
        front_chunk_idx = rp_rank
        front_start = start + front_chunk_idx * chunk_size
        front_end = front_start + chunk_size

        # Back chunk (late in sequence, large KV cache)
        back_chunk_idx = 2 * rp_size - 1 - rp_rank
        back_start = start + back_chunk_idx * chunk_size
        back_end = back_start + chunk_size

        # Set mask for both chunks
        mask[front_start:front_end] = True
        mask[back_start:back_end] = True

    return mask


def split_tensor_zigzag(
    tensor, cu_seqlens, rp_size, rp_rank, dim=1, require_padding_free=True
):
    """Split tensor using zigzag pattern along sequence dimension.

    Convenience wrapper around compute_zigzag_indices for splitting tensors.

    Args:
        tensor: Input tensor to split (typically [batch, seq_len, ...])
        cu_seqlens: Cumulative sequence lengths [num_seqs + 1]
        rp_size: Ring parallel world size
        rp_rank: This rank's position in ring [0, rp_size)
        dim: Dimension to split along (default: 1 = sequence dimension)
        require_padding_free: If True, enforce Phase 1 constraint (seq_len % 2*rp == 0)

    Returns:
        Sliced tensor containing only this rank's zigzag chunks

    Example:
        >>> q = torch.randn(2, 2048, 32, 128)  # batch=2, seq=2048, heads=32, dim=128
        >>> cu_seqlens = torch.tensor([0, 1024, 2048])  # 2 sequences
        >>> q_local = split_tensor_zigzag(q, cu_seqlens, rp_size=2, rp_rank=0)
        >>> q_local.shape
        torch.Size([2, 1024, 32, 128])  # 2 chunks of 512 each = 1024 total
    """

    mask = compute_zigzag_indices(cu_seqlens, rp_size, rp_rank, require_padding_free)

    # Apply mask along the sequence dimension
    if dim == 1:
        return tensor[:, mask, ...]
    elif dim == -1 or dim == tensor.ndim - 1:
        return tensor[..., mask]
    else:
        raise NotImplementedError(
            f"Zigzag splitting only supports dim=1 or dim=-1, got dim={dim}"
        )


def compute_cu_seqlens_from_position_ids(position_ids):
    """Compute cumulative sequence lengths from position_ids for flash-attn varlen API.

    In packed sequences, position_ids resets to 0 at the start of each new sequence.
    This function detects those resets and computes the cumulative sequence lengths
    required by flash-attn's varlen API.

    Args:
        position_ids: Position IDs tensor [batch, seq_len] or [1, seq_len]
                      Each packed sequence starts with position_id=0

    Returns:
        cu_seqlens: Cumulative sequence lengths tensor [num_seqs + 1] on same device
                    Example: [0, 512, 1024, 2048] means 3 sequences of lengths [512, 512, 1024]

    Example:
        >>> # Two sequences packed: lengths [4, 3]
        >>> position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
        >>> cu_seqlens = compute_cu_seqlens_from_position_ids(position_ids)
        >>> cu_seqlens
        tensor([0, 4, 7])

        >>> # Single sequence
        >>> position_ids = torch.tensor([[0, 1, 2, 3, 4]])
        >>> cu_seqlens = compute_cu_seqlens_from_position_ids(position_ids)
        >>> cu_seqlens
        tensor([0, 5])

    References:
        - ms-swift: swift/utils/torch_utils.py (lines 378-384)
    """
    import torch

    # Handle batch dimension: flatten to [total_seq_len]
    if position_ids.dim() == 2:
        position_ids = position_ids[
            0
        ]  # Take first batch (packed sequences share same structure)
    elif position_ids.dim() != 1:
        raise ValueError(
            f"position_ids must be 1D or 2D, got shape {position_ids.shape}"
        )

    # Find sequence boundaries (where position_id == 0)
    seq_start_indices = torch.where(position_ids == 0)[0]

    if len(seq_start_indices) == 0:
        # No sequences detected (shouldn't happen with valid packed data)
        raise ValueError(
            f"No sequence starts (position_id=0) found in position_ids. "
            f"position_ids: {position_ids[:20]}... (showing first 20)"
        )

    # Compute sequence end indices
    # End of each sequence is the start of the next sequence (or end of tensor)
    seq_end_indices = torch.cat(
        [
            seq_start_indices[1:],  # Next sequence start = this sequence end
            torch.tensor(
                [len(position_ids)], device=position_ids.device
            ),  # Last sequence ends at tensor end
        ]
    )

    # Compute sequence lengths
    seq_lengths = seq_end_indices - seq_start_indices

    # Compute cumulative lengths (cu_seqlens)
    cu_seqlens = torch.cumsum(
        torch.cat(
            [
                torch.tensor([0], device=position_ids.device),  # Start from 0
                seq_lengths,
            ]
        ),
        dim=0,
    )

    # Flash-attn requires int32 dtype for cu_seqlens
    return cu_seqlens.to(torch.int32)


def patch_llama_attention():
    """Monkey-patch LlamaAttention to use Ulysses + Ring-Attention.

    This function replaces the forward method of LlamaAttention with a custom
    implementation that uses DistributedAttention for distributed sequence parallel
    computation.

    The patched forward method:
    1. Computes Q, K, V projections as normal
    2. Applies RoPE (if needed)
    3. Extracts cu_seqlens from position_ids
    4. Calls DistributedAttention.forward() for distributed attention
    5. Returns output in the expected format

    NOTE: This is Phase 2.1 implementation for Llama-style models only.
    Other architectures will be added in Phase 2.3.

    Raises:
        ImportError: If transformers.models.llama.modeling_llama is not available
    """
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            apply_rotary_pos_emb,
        )
    except ImportError as e:
        raise ImportError(
            "Cannot import LlamaAttention from transformers. "
            "Llama models may not be supported in this transformers version."
        ) from e

    # Store original forward for potential restoration
    _original_llama_attention_forward = LlamaAttention.forward

    def ulysses_ring_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple | None = None,
        **kwargs,
    ):
        """Custom forward pass for LlamaAttention using Ulysses + Ring-Attention.

        This forward method wraps the standard attention computation with our
        distributed attention pipeline.
        """
        if output_attentions:
            LOG.warning(
                "output_attentions=True is not supported with Ulysses + Ring-Attention. "
                "Falling back to standard attention."
            )
            return _original_llama_attention_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        # === Standard Q, K, V projection ===
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, seq, num_heads, head_dim]
        query_states = query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.head_dim
        )
        key_states = key_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim
        )

        # === Apply RoPE ===
        if position_embeddings is None:
            # Compute cos/sin for RoPE
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        # === Handle KV cache (if enabled) ===
        if use_cache and past_key_value is not None:
            # Concatenate past KV cache
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
            past_key_value = (key_states, value_states)
        elif use_cache:
            past_key_value = (key_states, value_states)

        # === Repeat K/V for GQA/MQA ===
        if self.config.num_key_value_heads != self.config.num_attention_heads:
            if self.config.num_attention_heads % self.config.num_key_value_heads != 0:
                raise ValueError(
                    "num_attention_heads must be divisible by num_key_value_heads for GQA/MQA.\n\n"
                    f"Got num_attention_heads={self.config.num_attention_heads}, "
                    f"num_key_value_heads={self.config.num_key_value_heads}."
                )
            n_rep = self.config.num_attention_heads // self.config.num_key_value_heads
            # NOTE: HF's `repeat_kv` expects layout [bsz, num_kv_heads, seqlen, head_dim].
            # We keep tensors in [bsz, seqlen, num_heads, head_dim] for varlen flash-attn,
            # so we repeat along the heads dimension (dim=2).
            key_states = key_states.repeat_interleave(n_rep, dim=2)
            value_states = value_states.repeat_interleave(n_rep, dim=2)

        # === Get global cu_seqlens (pre-slicing) ===
        # SequenceParallelContextManager calls update_ring_attn_params() before slicing
        # the batch, so we can reuse the global cu_seqlens here. This is required for
        # varlen ring-flash-attn kernels (Llama3) to correctly handle packed sequences
        # that span across sequence-parallel ranks.
        from axolotl.monkeypatch.ring_attn import get_ring_attn_cu_seqlens

        cu_seqlens = get_ring_attn_cu_seqlens()

        # === Call DistributedAttention ===
        distributed_attn = get_distributed_attention()

        LOG.warning(
            f"[Llama] Before DistributedAttention: q.shape={query_states.shape}, "
            f"cu_seqlens[-1]={int(cu_seqlens[-1].item())}"
        )

        attn_output = distributed_attn.forward(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens=cu_seqlens,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            softmax_scale=None,  # Use default 1/sqrt(head_dim)
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )

        LOG.warning(
            f"[Llama] After DistributedAttention: attn_output.shape={attn_output.shape}"
        )

        # === Output projection ===
        # attn_output shape: [batch, seq, num_heads, head_dim]
        # Reshape to [batch, seq, hidden_size] by flattening last two dims
        # Use actual output shape, not input q_len (may differ due to distributed ops)
        bsz_out, seq_len_out, num_heads_out, head_dim_out = attn_output.shape
        attn_output = attn_output.reshape(
            bsz_out, seq_len_out, num_heads_out * head_dim_out
        )
        attn_output = self.o_proj(attn_output)

        # Return (attn_output, attn_weights) to match original LlamaAttention signature
        # Note: attn_weights=None since we don't compute them in distributed attention
        return attn_output, None

    # Apply the monkey-patch
    LlamaAttention.forward = ulysses_ring_attention_forward

    LOG.info(
        "Patched LlamaAttention.forward() to use Ulysses + Ring-Attention "
        "(Phase 2.1: Llama-style models only)"
    )


def patch_gpt_neox_attention():
    """Monkey-patch GPTNeoXAttention to use Ulysses + Ring-Attention.

    This function replaces the forward method of GPTNeoXAttention with a custom
    implementation that uses DistributedAttention for distributed sequence parallel
    computation.

    The patched forward method:
    1. Computes Q, K, V projections as normal
    2. Applies RoPE (if needed)
    3. Extracts cu_seqlens from position_ids
    4. Calls DistributedAttention.forward() for distributed attention
    5. Returns output in the expected format

    NOTE: This is Phase 2.4 implementation for GPT-NeoX models.

    Raises:
        ImportError: If transformers.models.gpt_neox.modeling_gpt_neox is not available
    """
    try:
        from transformers.models.gpt_neox.modeling_gpt_neox import (
            GPTNeoXAttention,
            apply_rotary_pos_emb,
        )
    except ImportError as e:
        raise ImportError(
            "Cannot import GPTNeoXAttention from transformers. "
            "GPT-NeoX models may not be supported in this transformers version."
        ) from e

    # Store original forward for potential restoration
    _original_gpt_neox_attention_forward = GPTNeoXAttention.forward

    def ulysses_ring_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        layer_past=None,
        output_attentions: bool = False,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        """Custom forward pass for GPTNeoXAttention using Ulysses + Ring-Attention."""
        if output_attentions:
            LOG.warning(
                "output_attentions=True is not supported with Ulysses + Ring-Attention. "
                "Falling back to standard attention."
            )
            return _original_gpt_neox_attention_forward(
                self,
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                layer_past=layer_past,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        if position_embeddings is None:
            # This should be provided by the caller (e.g. GPTNeoXLayer). If it isn't,
            # fall back to the original implementation to avoid incorrect RoPE behavior.
            return _original_gpt_neox_attention_forward(
                self,
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                layer_past=layer_past,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)

        # Match the upstream GPTNeoXAttention layout:
        # qkv: [batch, num_heads, seq, 3 * head_dim]
        qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # === Handle KV cache (if enabled) ===
        if layer_past is not None and hasattr(layer_past, "update"):
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key_states, value_states = layer_past.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Convert to [batch, seq, num_heads, head_dim] for DistributedAttention.
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        # === Get global cu_seqlens (pre-slicing) ===
        from axolotl.monkeypatch.ring_attn import get_ring_attn_cu_seqlens

        cu_seqlens = get_ring_attn_cu_seqlens()

        # === Call DistributedAttention ===
        distributed_attn = get_distributed_attention()

        attn_output = distributed_attn.forward(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens=cu_seqlens,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            softmax_scale=None,  # Use default 1/sqrt(head_dim)
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )

        # === Output projection ===
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)

        # Match upstream return type: (attn_output, attn_weights)
        return attn_output, None

    # Apply the monkey-patch
    GPTNeoXAttention.forward = ulysses_ring_attention_forward

    LOG.info(
        "Patched GPTNeoXAttention.forward() to use Ulysses + Ring-Attention "
        "(Phase 2.4: GPT-NeoX models)"
    )


def patch_falcon_attention():
    """Monkey-patch FalconAttention to use Ulysses + Ring-Attention.

    This function replaces the forward method of FalconAttention with a custom
    implementation that uses DistributedAttention for distributed sequence parallel
    computation.

    NOTE: This is Phase 2.4 implementation for Falcon models.

    Raises:
        ImportError: If transformers.models.falcon.modeling_falcon is not available
    """
    try:
        from transformers.models.falcon.modeling_falcon import (
            FalconAttention,
        )
    except ImportError as e:
        raise ImportError(
            "Cannot import FalconAttention from transformers. "
            "Falcon models may not be supported in this transformers version."
        ) from e

    # Store original forward for potential restoration
    _original_falcon_attention_forward = FalconAttention.forward

    def ulysses_ring_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        layer_past=None,
        head_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        alibi=None,
        **kwargs,
    ):
        """Custom forward pass for FalconAttention using Ulysses + Ring-Attention."""
        if output_attentions:
            LOG.warning(
                "output_attentions=True is not supported with Ulysses + Ring-Attention. "
                "Falling back to standard attention."
            )
            return _original_falcon_attention_forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                layer_past,
                head_mask,
                use_cache,
                output_attentions,
                alibi,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        # === Q, K, V projection ===
        if self.new_decoder_architecture:
            # Falcon-40B and newer: separate Q, K, V projections
            query_states = self.query_key_value(hidden_states)
            key_states = self.query_key_value(hidden_states)
            value_states = self.query_key_value(hidden_states)
        else:
            # Falcon-7B: combined projection
            fused_qkv = self.query_key_value(hidden_states)
            # Split Q, K, V
            qkv_size = fused_qkv.size(-1) // 3
            query_states, key_states, value_states = torch.split(
                fused_qkv, qkv_size, dim=-1
            )

        # Reshape to [batch, seq, num_heads, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # === RoPE (Falcon typically uses ALiBi instead, but some variants use RoPE) ===
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(
                    q_len, dtype=torch.long, device=hidden_states.device
                )
                position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
            cos, sin = self.rotary_emb(value_states, position_ids)
            from transformers.models.falcon.modeling_falcon import apply_rotary_pos_emb

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=2
            )

        # === Handle KV cache (if enabled) ===
        if use_cache and layer_past is not None:
            key_states = torch.cat([layer_past[0], key_states], dim=1)
            value_states = torch.cat([layer_past[1], value_states], dim=1)
            layer_past = (key_states, value_states)
        elif use_cache:
            layer_past = (key_states, value_states)

        # === Repeat K/V for GQA ===
        if self.num_kv_heads != self.num_heads:
            # Import repeat_kv if available, otherwise implement inline
            try:
                from transformers.models.falcon.modeling_falcon import repeat_kv

                key_states = repeat_kv(key_states, self.num_heads // self.num_kv_heads)
                value_states = repeat_kv(
                    value_states, self.num_heads // self.num_kv_heads
                )
            except ImportError:
                n_rep = self.num_heads // self.num_kv_heads
                key_states = key_states.repeat_interleave(n_rep, dim=2)
                value_states = value_states.repeat_interleave(n_rep, dim=2)

        # === Get global cu_seqlens (pre-slicing) ===
        from axolotl.monkeypatch.ring_attn import get_ring_attn_cu_seqlens

        cu_seqlens = get_ring_attn_cu_seqlens()

        # === Call DistributedAttention ===
        distributed_attn = get_distributed_attention()

        # Handle ALiBi slopes if present
        alibi_slopes = None
        if alibi is not None and hasattr(self, "_get_alibi_slopes"):
            # Falcon uses ALiBi for position bias
            alibi_slopes = self._get_alibi_slopes(self.num_heads)

        attn_output = distributed_attn.forward(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens=cu_seqlens,
            dropout_p=self.attention_dropout if self.training else 0.0,
            softmax_scale=None,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )

        # === Output projection ===
        # Use actual output shape, not input q_len (may differ due to distributed ops)
        bsz_out, seq_len_out = attn_output.shape[0], attn_output.shape[1]
        attn_output = attn_output.reshape(bsz_out, seq_len_out, -1)
        attn_output = self.dense(attn_output)

        if use_cache:
            return attn_output, layer_past
        return (attn_output,)

    # Apply the monkey-patch
    FalconAttention.forward = ulysses_ring_attention_forward

    LOG.info(
        "Patched FalconAttention.forward() to use Ulysses + Ring-Attention "
        "(Phase 2.4: Falcon models)"
    )


def patch_bloom_attention():
    """Monkey-patch BloomAttention to use Ulysses + Ring-Attention.

    This function replaces the forward method of BloomAttention with a custom
    implementation that uses DistributedAttention for distributed sequence parallel
    computation.

    NOTE: This is Phase 2.4 implementation for BLOOM models.

    Raises:
        ImportError: If transformers.models.bloom.modeling_bloom is not available
    """
    try:
        from transformers.models.bloom.modeling_bloom import (
            BloomAttention,
        )
    except ImportError as e:
        raise ImportError(
            "Cannot import BloomAttention from transformers. "
            "BLOOM models may not be supported in this transformers version."
        ) from e

    # Store original forward for potential restoration
    _original_bloom_attention_forward = BloomAttention.forward

    def ulysses_ring_attention_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        alibi=None,
        layer_past=None,
        head_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        """Custom forward pass for BloomAttention using Ulysses + Ring-Attention."""
        if output_attentions:
            LOG.warning(
                "output_attentions=True is not supported with Ulysses + Ring-Attention. "
                "Falling back to standard attention."
            )
            return _original_bloom_attention_forward(
                self,
                hidden_states,
                residual,
                attention_mask,
                alibi,
                layer_past,
                head_mask,
                use_cache,
                output_attentions,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        # === Q, K, V projection (BLOOM uses combined query_key_value projection) ===
        fused_qkv = self.query_key_value(hidden_states)

        # Split into Q, K, V
        # BLOOM: [batch, seq, 3 * num_heads * head_dim]
        qkv_size = fused_qkv.size(-1) // 3
        query_states, key_states, value_states = torch.split(
            fused_qkv, qkv_size, dim=-1
        )

        # Reshape to [batch, seq, num_heads, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim)

        # === BLOOM uses ALiBi instead of RoPE, no position embeddings needed ===

        # === Handle KV cache (if enabled) ===
        if use_cache and layer_past is not None:
            key_states = torch.cat([layer_past[0], key_states], dim=1)
            value_states = torch.cat([layer_past[1], value_states], dim=1)
            layer_past = (key_states, value_states)
        elif use_cache:
            layer_past = (key_states, value_states)

        # === Get global cu_seqlens (pre-slicing) ===
        from axolotl.monkeypatch.ring_attn import get_ring_attn_cu_seqlens

        cu_seqlens = get_ring_attn_cu_seqlens()

        # === Call DistributedAttention ===
        distributed_attn = get_distributed_attention()

        # BLOOM uses ALiBi slopes for position bias
        alibi_slopes = alibi if alibi is not None else None

        attn_output = distributed_attn.forward(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens=cu_seqlens,
            dropout_p=0.0,  # BLOOM doesn't use attention dropout
            softmax_scale=None,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )

        # === Output projection ===
        # Use actual output shape, not input q_len (may differ due to distributed ops)
        bsz_out, seq_len_out = attn_output.shape[0], attn_output.shape[1]
        attn_output = attn_output.reshape(bsz_out, seq_len_out, -1)
        attn_output = self.dense(attn_output)

        # BLOOM returns (attn_output, layer_past)
        if use_cache:
            return attn_output, layer_past
        return (attn_output, None)

    # Apply the monkey-patch
    BloomAttention.forward = ulysses_ring_attention_forward

    LOG.info(
        "Patched BloomAttention.forward() to use Ulysses + Ring-Attention "
        "(Phase 2.4: BLOOM models)"
    )
