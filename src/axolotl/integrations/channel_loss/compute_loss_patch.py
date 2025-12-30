# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compute loss patching for Channel Loss.

This module wraps trainer.compute_loss to add per-channel loss tracking
without affecting the original loss computation or gradients.

Design notes:
- Observer-only: The channel loss statistics are purely observational
- Uses torch.no_grad() and detach() to prevent gradient impact
- Independently computes per-token CE from logits/labels
- Accumulates sum/count locally, sync happens in callback
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from axolotl.utils.logging import get_logger

from .segment import flatten_channels, get_segment_boundaries
from .utils import _get_context_parallel_group

LOG = get_logger(__name__)


def patch_compute_loss_for_channel_loss(
    trainer,
    cfg: Dict[str, Any],
) -> None:
    """
    Patch trainer.compute_loss to add channel loss statistics.

    This wraps the existing compute_loss method to:
    1. Extract channel information from inputs (as side input)
    2. Call original compute_loss normally
    3. Compute per-token CE separately for channel statistics

    Args:
        trainer: The HuggingFace Trainer instance.
        cfg: Configuration dict containing channel loss settings.

    Design:
        - Observer-only: Does not modify the original loss or gradients
        - Safe: Uses no_grad and detach to ensure isolation
        - Compatible: Works with any model that outputs logits
    """
    orig_compute_loss = trainer.compute_loss
    segment_mode: Literal["auto", "position_ids", "attention_mask"] = cfg.get(
        "channel_loss_segment", "auto"
    )
    prefix: str = cfg.get("channel_loss_prefix", "loss_")

    # Initialize warning flag on trainer instance (to warn only once)
    trainer._channel_loss_warned_no_logits = False

    def compute_loss_with_channel(
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Wrapped compute_loss that adds channel statistics.

        Args:
            model: The model being trained.
            inputs: Input dict (may contain 'channel' field).
            return_outputs: Whether to return model outputs.
            num_items_in_batch: Number of items in batch (for gradient accumulation).

        Returns:
            Loss tensor, or (loss, outputs) tuple if return_outputs=True.
        """
        # Extract channel from batch (collator always uses "channel" key)
        # Note: collator extracts from dataset's channel_loss_field and stores in batch["channel"]
        channels = inputs.pop("channel", None)

        # Get tensors needed for channel statistics.
        #
        # NOTE (Axolotl Context Parallelism):
        # CP slices model kwargs inside a forward pre-hook using `kwargs.copy()`. As a result,
        # this `inputs` dict keeps the *full* tensors, while `outputs.logits` may be CP-local
        # (when `SequenceParallelContextManager(gather_outputs=False)`).
        labels = inputs.get("labels")
        position_ids = inputs.get("position_ids")
        attention_mask = inputs.get("attention_mask")

        # Call original compute_loss with return_outputs=True to get logits
        # Without Cut Cross Entropy, this should return proper logits
        result = orig_compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        if isinstance(result, tuple):
            loss, outputs = result
        else:
            # Shouldn't happen with return_outputs=True, but handle gracefully
            loss = result
            outputs = None

        # Compute channel statistics (observer-only)
        # Check if we have the required logits
        has_logits = (
            outputs is not None
            and hasattr(outputs, "logits")
            and outputs.logits is not None
        )

        if channels is not None and labels is not None:
            if has_logits:
                _update_channel_stats(
                    trainer=trainer,
                    logits=outputs.logits,
                    labels=labels,
                    channels=channels,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    segment_mode=segment_mode,
                    prefix=prefix,
                )
            else:
                # Runtime detection: logits are missing
                # This can happen if an incompatible optimization is enabled
                # despite passing compile-time checks
                if not trainer._channel_loss_warned_no_logits:
                    LOG.error(
                        "Channel Loss: No logits available from compute_loss().\n"
                        "This usually means an incompatible optimization is enabled.\n"
                        "Possible causes:\n"
                        "  - Liger FLCE with skip_logits=True (training mode)\n"
                        "  - Custom Trainer not supporting return_outputs=True\n"
                        "  - Model forward() not returning logits\n"
                        "Result: Channel statistics will NOT be recorded.\n"
                        "This warning will only be shown once."
                    )
                    trainer._channel_loss_warned_no_logits = True

        return (loss, outputs) if return_outputs else loss

    # Apply the patch
    trainer.compute_loss = compute_loss_with_channel
    LOG.info("Channel Loss: Patched trainer.compute_loss for channel statistics")


def _update_channel_stats(
    trainer,
    logits: torch.Tensor,
    labels: torch.Tensor,
    channels: Union[List[str], List[List[str]]],
    position_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    segment_mode: Literal["auto", "position_ids", "attention_mask"],
    prefix: str,
) -> None:
    """
    Update per-channel loss statistics.

    This function:
    1. Computes per-token cross entropy (detached, no gradient)
    2. Determines segment boundaries (for packing mode)
    3. Accumulates sum/count per channel locally

    Args:
        trainer: Trainer instance (to access _channel_loss_stats).
        logits: Model output logits. Shape: (batch_size, seq_len, vocab_size)
        labels: Ground truth labels. Shape: (batch_size, seq_len)
        channels: Channel info. List[str] for standard, List[List[str]] for packing.
        position_ids: Position IDs tensor. Shape: (batch_size, seq_len)
        attention_mask: Attention mask tensor. Shape: (batch_size, seq_len)
        segment_mode: Segment detection strategy.
        prefix: Prefix for channel metric keys.

    Design:
        - Uses torch.no_grad() to prevent any gradient computation
        - Detaches per-token loss from computation graph
        - CP-aware:
          - When CP is enabled and outputs are CP-local (gather_outputs=False), avoids
            all-gathering full logits and instead computes boundary-correct losses using
            the full (pre-hook) labels tensor available in `inputs`.
          - When CP outputs are already gathered (gather_outputs=True), computes only
            on CP rank 0 to avoid redundant work.
        - Filters NaN/Inf values (safety)
    """
    with torch.no_grad():
        if logits is None or labels is None or not channels:
            return

        batch_size, label_seq_len = labels.shape
        logits_seq_len = logits.size(1)

        cp_group = _get_context_parallel_group(trainer)
        cp_enabled = cp_group is not None and dist.is_initialized()
        cp_size = dist.get_world_size(cp_group) if cp_enabled else 1
        cp_rank = dist.get_rank(cp_group) if cp_enabled else 0

        # Segment detection expects position_ids in "auto" mode; CP pre-hook creates it
        # if missing, but that happens on a kwargs copy. Create it here as well if absent.
        if position_ids is None:
            position_ids = torch.arange(
                0, label_seq_len, dtype=torch.long, device=labels.device
            ).expand(batch_size, -1)

        is_packing = isinstance(channels[0], list)

        # Detect whether `logits` are CP-local (post-hook gather disabled) or already gathered.
        is_cp_local_logits = False
        if cp_size > 1:
            divisor = min(cp_size, 64)
            pad_len = (divisor - (label_seq_len % divisor)) % divisor
            expected_chunk_len = (label_seq_len + pad_len) // cp_size
            is_cp_local_logits = logits_seq_len == expected_chunk_len
            LOG.info(
                f"[RANK {dist.get_rank() if dist.is_initialized() else 0}] CP detection: "
                f"logits_seq_len={logits_seq_len}, label_seq_len={label_seq_len}, "
                f"expected_chunk_len={expected_chunk_len}, is_cp_local={is_cp_local_logits}, "
                f"cp_rank={cp_rank}, cp_size={cp_size}"
            )

        # If CP outputs are already gathered, compute only on CP-rank 0 to avoid redundant work.
        if cp_size > 1 and not is_cp_local_logits and cp_rank != 0:
            return

        loss_offset = 0  # global loss-index offset (used for packing overlap)

        if cp_size > 1 and is_cp_local_logits:
            # CP-local path (SFT gather_outputs=False):
            #
            # Compute boundary-correct losses for this CP shard without all-gathering logits.
            # For CP rank r with local token chunk [s, s+L), we compute losses for targets:
            #   tokens [s+1, s+L] (non-last ranks)
            #   tokens [s+1, s+L-1] (last rank; global last token has no target)
            #
            # This uses the full (pre-hook) `labels` tensor and pads out-of-range tokens
            # with ignore_index=-100.
            token_chunk_len = logits_seq_len
            local_token_start = cp_rank * token_chunk_len
            is_last_cp_rank = cp_rank == (cp_size - 1)

            token_count = token_chunk_len - (1 if is_last_cp_rank else 0)
            if token_count <= 0:
                return

            shift_logits = logits[:, :-1, :].contiguous() if is_last_cp_rank else logits

            label_start = local_token_start + 1
            label_end = label_start + token_count

            if label_start >= label_seq_len:
                shift_labels = torch.full(
                    (batch_size, token_count),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
            else:
                slice_end = min(label_end, label_seq_len)
                shift_labels = labels[:, label_start:slice_end]
                if shift_labels.size(1) < token_count:
                    pad = torch.full(
                        (batch_size, token_count - shift_labels.size(1)),
                        -100,
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    shift_labels = torch.cat([shift_labels, pad], dim=1)

            loss_offset = local_token_start

            if shift_logits.size(1) != shift_labels.size(1):
                LOG.warning(
                    "Channel Loss (CP): shift_logits/shift_labels length mismatch: "
                    "shift_logits=%s shift_labels=%s (cp_rank=%s cp_size=%s)",
                    tuple(shift_logits.shape),
                    tuple(shift_labels.shape),
                    cp_rank,
                    cp_size,
                )
                min_len = min(shift_logits.size(1), shift_labels.size(1))
                shift_logits = shift_logits[:, :min_len, :].contiguous()
                shift_labels = shift_labels[:, :min_len].contiguous()
        else:
            # Non-CP or CP-gathered path: standard causal shift on full tensors.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).detach()

        # Create valid token mask (exclude ignore_index=-100)
        valid_mask = shift_labels.reshape(-1) != -100

        # Filter NaN/Inf values early to keep sums/counts stable.
        valid_mask = valid_mask & torch.isfinite(per_token_loss)

        # Determine mode (train or eval)
        mode = "train" if trainer.model.training else "eval"
        stats = trainer._channel_loss_stats[mode]

        if not is_packing:
            # Standard mode: one channel per batch item.
            per_token_loss_2d = per_token_loss.reshape(batch_size, -1)
            valid_mask_2d = valid_mask.reshape(batch_size, -1)

            for batch_index, channel in enumerate(channels[:batch_size]):
                row_mask = valid_mask_2d[batch_index]
                if not row_mask.any():
                    continue

                row_loss = per_token_loss_2d[batch_index][row_mask]
                if row_loss.numel() == 0:
                    continue

                key = f"{prefix}{channel}"
                stats[key]["sum"] += row_loss.sum().item()
                stats[key]["count"] += row_loss.numel()

            return

        # Packing mode: channels map to multiple segments within a single sequence.
        if batch_size != 1:
            LOG.warning(
                "Channel Loss: packing mode expects batch_size=1; got %d. Skipping.",
                batch_size,
            )
            return

        cu_seqlens_token = get_segment_boundaries(
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            mode=segment_mode,
        )

        # Token-boundaries -> loss-index boundaries (token_pos -> token_pos - 1).
        max_loss_len = max(label_seq_len - 1, 0)
        cu_seqlens_loss = torch.clamp(cu_seqlens_token - 1, min=0, max=max_loss_len)

        flat_channels = flatten_channels(channels)
        num_segments = min(len(flat_channels), int(cu_seqlens_loss.numel()) - 1)

        local_loss_start = loss_offset
        local_loss_end = loss_offset + per_token_loss.numel()

        for i in range(num_segments):
            channel = flat_channels[i]
            seg_start = int(cu_seqlens_loss[i].item())
            seg_end = int(cu_seqlens_loss[i + 1].item())

            overlap_start = max(seg_start, local_loss_start)
            overlap_end = min(seg_end, local_loss_end)
            if overlap_start >= overlap_end:
                continue

            local_start = overlap_start - local_loss_start
            local_end = overlap_end - local_loss_start

            segment_mask = valid_mask[local_start:local_end]
            if not segment_mask.any():
                continue

            segment_loss = per_token_loss[local_start:local_end][segment_mask]
            if segment_loss.numel() == 0:
                continue

            key = f"{prefix}{channel}"
            stats[key]["sum"] += segment_loss.sum().item()
            stats[key]["count"] += segment_loss.numel()
