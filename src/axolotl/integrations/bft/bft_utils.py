"""BFT (Balanced Fine-Tuning) loss utilities.

This module implements sample-level weighting on top of DFT token weighting.
Key functions:
- compute_sample_weights: Sliding window min-mean confidence → sample weights
- compute_bft_loss: Full BFT loss with CP-aware token CE + sample weighting
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from axolotl.utils.logging import get_logger

# Import DFT utilities for reuse
from axolotl.integrations.dft.dft_utils import (
    _get_context_parallel_group,
    compute_per_token_cross_entropy,
    reduce_token_loss,
)

LOG = get_logger(__name__)


def _pad_for_cp_all_gather(
    token_confidences: torch.Tensor,
    mask: torch.Tensor,
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    trainer,
    shift_labels: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad CP-local per-target tensors so every CP rank has the same length.

    Under CP-local logits with causal shifting, the last CP rank has one fewer target
    (global last token has no target), which would make all_gather shapes mismatch.
    We pad that missing position with confidence=1.0 and mask=False.
    """
    cp_group = _get_context_parallel_group(trainer) if trainer is not None else None
    if cp_group is None or not dist.is_initialized():
        return token_confidences, mask

    cp_size = dist.get_world_size(cp_group)
    if cp_size <= 1:
        return token_confidences, mask

    label_seq_len = labels.size(1)
    logits_seq_len = logits.size(1)

    # Mirror DFT CP-local detection logic to avoid gathering full logits.
    divisor = min(cp_size, 64)
    pad_len = (divisor - (label_seq_len % divisor)) % divisor
    expected_chunk_len = (label_seq_len + pad_len) // cp_size
    is_cp_local_logits = logits_seq_len == expected_chunk_len

    if not (shift_labels and is_cp_local_logits):
        return token_confidences, mask

    target_len = expected_chunk_len
    current_len = token_confidences.size(1)
    if current_len == target_len:
        return token_confidences, mask

    if current_len < target_len:
        pad = target_len - current_len
        token_confidences = torch.cat(
            [
                token_confidences,
                token_confidences.new_full((token_confidences.size(0), pad), 1.0),
            ],
            dim=1,
        )
        mask = torch.cat(
            [mask, mask.new_zeros((mask.size(0), pad))],
            dim=1,
        )
        return token_confidences, mask

    # Unexpected: truncate (debug safety) rather than erroring inside all_gather.
    return token_confidences[:, :target_len], mask[:, :target_len]


def compute_sample_weights(
    token_confidences: torch.Tensor,
    mask: torch.Tensor,
    *,
    group_size: int = 256,
    min_valid_tokens: int = 32,
    weight_floor: float = 0.1,
    weight_ceiling: float = 2.0,
    normalize: bool = True,
    warmup_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute sample-level weights from per-token confidences via sliding windows.

    Algorithm:
    1. For each sample, slide a window of size `group_size` over the sequence
    2. Compute mean confidence in each window (only counting valid tokens)
    3. Take the minimum mean across all windows → p_conf[b] (hardest region)
    4. Sample weight: s_b = 1 - p_conf[b]
    5. Apply clamp, normalization, and warmup

    Args:
        token_confidences: Per-token confidence exp(-CE), shape [B, T]
        mask: Valid token mask (True where label != ignore_index), shape [B, T]
        group_size: Sliding window length
        min_valid_tokens: Skip windows with fewer valid tokens
        weight_floor: Minimum sample weight (e.g., 0.1)
        weight_ceiling: Maximum sample weight (e.g., 2.0)
        normalize: Whether to normalize to mean=1.0
        warmup_factor: Multiply final weights by this (0→1 during warmup)

    Returns:
        sample_weights: Shape [B], one weight per sample (stop-grad applied)
        p_conf: Shape [B], minimum window confidence per sample (for logging)
    """
    token_confidences = token_confidences.float()
    batch_size, seq_len = token_confidences.shape
    device = token_confidences.device

    if seq_len < group_size:
        # Sequence too short for windowing: use global mean
        valid_counts = mask.sum(dim=1).float()
        p_conf = torch.where(
            valid_counts > 0,
            (token_confidences * mask.float()).sum(dim=1) / valid_counts.clamp(min=1),
            torch.ones(batch_size, device=device),  # Default to 1.0 (low weight)
        )
    else:
        # Sliding window: compute mean confidence for each window
        # Using unfold for efficient windowing
        # Shape: [B, num_windows, group_size]
        c_windows = token_confidences.unfold(1, group_size, 1)
        m_windows = mask.unfold(1, group_size, 1).float()

        # Sum and count valid tokens per window
        valid_counts = m_windows.sum(dim=2)  # [B, num_windows]
        conf_sums = (c_windows * m_windows).sum(dim=2)  # [B, num_windows]

        # Mean confidence per window (mask out windows with too few valid tokens)
        window_means = torch.where(
            valid_counts >= min_valid_tokens,
            conf_sums / valid_counts.clamp(min=1),
            torch.full_like(conf_sums, float("inf")),  # Ignore invalid windows
        )

        # Min across windows (with inf-handling)
        p_conf = window_means.min(dim=1).values  # [B]
        p_conf = torch.where(
            torch.isinf(p_conf),
            torch.ones_like(p_conf),  # Fallback if all windows invalid
            p_conf,
        )

    # Compute sample weights: s_b = 1 - p_conf
    sample_weights = 1.0 - p_conf

    # Clamp to [floor, ceiling]
    sample_weights = sample_weights.clamp(min=weight_floor, max=weight_ceiling)

    # Normalize to mean=1.0 (stabilizes gradient scale)
    if normalize:
        mean_weight = sample_weights.mean().item()
        if mean_weight > 0:
            sample_weights = sample_weights / mean_weight

    # Apply warmup
    sample_weights = sample_weights * warmup_factor

    # Stop gradient (weights are observations, not learnable)
    sample_weights = sample_weights.detach()
    p_conf = p_conf.detach()

    return sample_weights, p_conf


def gather_confidences_across_cp(
    token_confidences: torch.Tensor,
    mask: torch.Tensor,
    cp_group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All-gather token confidences and masks across CP ranks.

    This is necessary because each CP rank only has a shard of the sequence,
    but sample weights must be computed from the full sequence to ensure
    consistency across ranks.

    Args:
        token_confidences: Local shard, shape [B, L_local]
        mask: Local mask, shape [B, L_local]
        cp_group: CP process group

    Returns:
        full_confidences: Shape [B, L_full]
        full_mask: Shape [B, L_full]
    """
    if cp_group is None or not dist.is_initialized():
        return token_confidences, mask

    cp_size = dist.get_world_size(cp_group)
    if cp_size == 1:
        return token_confidences, mask

    # All-gather along sequence dimension
    # Output: [cp_size, B, L_local] → reshape to [B, L_full]
    gathered_conf = [torch.zeros_like(token_confidences) for _ in range(cp_size)]
    gathered_mask = [torch.zeros_like(mask) for _ in range(cp_size)]

    dist.all_gather(gathered_conf, token_confidences, group=cp_group)
    dist.all_gather(gathered_mask, mask, group=cp_group)

    # Concatenate along sequence dimension
    full_confidences = torch.cat(gathered_conf, dim=1)
    full_mask = torch.cat(gathered_mask, dim=1)

    return full_confidences, full_mask


def compute_bft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    shift_labels: bool = True,
    num_items_in_batch: int | None = None,
    chunk_size: Optional[int] = None,
    trainer=None,
    group_size: int = 256,
    min_valid_tokens: int = 32,
    weight_floor: float = 0.1,
    weight_ceiling: float = 2.0,
    normalize_sample_weight: bool = True,
    warmup_factor: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute BFT loss with sample-level weighting.

    This function combines:
    1. Per-token CE computation (CP-aware via DFT utils)
    2. Token-level DFT weighting: w_{b,t} = exp(-CE.detach())
    3. Sample-level BFT weighting: s_b from sliding window confidence
    4. Final loss: sum(s_b * w_{b,t} * CE) / denom

    Args:
        logits: Model output logits, shape [batch, seq_len, vocab_size]
        labels: Target labels, shape [batch, seq_len]
        ignore_index: Label value to ignore
        shift_labels: Whether to shift for causal LM
        num_items_in_batch: Denominator for loss reduction
        chunk_size: Chunked CE for memory efficiency
        trainer: For CP detection
        group_size: BFT sliding window size
        min_valid_tokens: Min valid tokens per window
        weight_floor: Min sample weight
        weight_ceiling: Max sample weight
        normalize_sample_weight: Normalize to mean=1.0
        warmup_factor: Warmup multiplier (0→1)

    Returns:
        loss: Scalar loss tensor
        metrics: Dict with logging metrics (s_mean, s_min, s_max, p_conf_mean, raw_ce)
    """
    # Step 1: Compute per-token CE (reuse DFT's CP-aware logic)
    per_token_loss, valid_mask = compute_per_token_cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        shift_labels=shift_labels,
        chunk_size=chunk_size,
        trainer=trainer,
    )

    # Step 2: Token-level DFT weighting (stop-grad)
    with torch.no_grad():
        token_weights = torch.exp(-per_token_loss)  # w_{b,t} = exp(-CE)

    # Step 3: Reshape to [B, T] for sample-level processing
    batch_size = labels.size(0)
    # After shifting, sequence length is reduced
    seq_len_after_shift = per_token_loss.numel() // batch_size
    token_confidences_flat = token_weights  # Shape: [B * T]
    mask_flat = valid_mask  # Shape: [B * T]

    # Reshape to [B, T]
    token_confidences = token_confidences_flat.view(batch_size, seq_len_after_shift)
    mask_2d = mask_flat.view(batch_size, seq_len_after_shift)

    # Step 4: CP all-gather (if needed)
    cp_group = _get_context_parallel_group(trainer) if trainer is not None else None
    token_confidences_for_gather, mask_for_gather = _pad_for_cp_all_gather(
        token_confidences,
        mask_2d,
        logits=logits,
        labels=labels,
        trainer=trainer,
        shift_labels=shift_labels,
    )
    full_confidences, full_mask = gather_confidences_across_cp(
        token_confidences_for_gather, mask_for_gather, cp_group
    )

    # Step 5: Compute sample weights
    sample_weights, p_conf = compute_sample_weights(
        full_confidences,
        full_mask,
        group_size=group_size,
        min_valid_tokens=min_valid_tokens,
        weight_floor=weight_floor,
        weight_ceiling=weight_ceiling,
        normalize=normalize_sample_weight,
        warmup_factor=warmup_factor,
    )

    # Step 6: Apply sample weights to per-token losses
    sample_weights_for_loss = sample_weights.repeat_interleave(seq_len_after_shift)

    # Apply both token and sample weights
    weighted_loss = per_token_loss * token_weights * sample_weights_for_loss

    # Step 7: Reduce to scalar
    loss = reduce_token_loss(
        weighted_loss,
        valid_mask,
        num_items_in_batch=num_items_in_batch,
    )

    # Step 8: Collect metrics for logging
    with torch.no_grad():
        raw_ce = (
            per_token_loss[valid_mask].mean()
            if valid_mask.any()
            else per_token_loss.new_tensor(0.0)
        )
        metrics = {
            "bft/s_mean": sample_weights.mean().item(),
            "bft/s_min": sample_weights.min().item(),
            "bft/s_max": sample_weights.max().item(),
            "bft/p_conf_mean": p_conf.mean().item(),
            "bft/valid_token_ratio": full_mask.float().mean().item(),
            "loss/raw_ce": raw_ce.item(),
        }

    return loss, metrics
