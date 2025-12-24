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
import torch.nn as nn

from axolotl.utils.logging import get_logger

from .segment import flatten_channels, get_segment_boundaries

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
        # Extract channel (side input, not passed to model)
        channels = inputs.pop("channel", None)

        # Remove task_type if it exists (shouldn't be passed to model)
        inputs.pop("task_type", None)

        # Get tensors needed for channel statistics (before they might be modified)
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
        if (
            channels is not None
            and labels is not None
            and outputs is not None
            and hasattr(outputs, "logits")
            and outputs.logits is not None
        ):
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
        - Filters NaN/Inf values (Context Parallel safety)
    """
    with torch.no_grad():
        # Compute per-token cross entropy
        # Shift logits and labels for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute per-token loss without reduction
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).detach()

        # Create valid token mask (exclude ignore_index=-100)
        valid_mask = shift_labels.view(-1) != -100

        # Get segment boundaries (cu_seqlens)
        cu_seqlens = get_segment_boundaries(
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            mode=segment_mode,
        )

        # Flatten channels for packing mode
        flat_channels = flatten_channels(channels)

        # Determine mode (train or eval)
        mode = "train" if trainer.model.training else "eval"
        stats = trainer._channel_loss_stats[mode]

        # Accumulate per-channel statistics
        num_segments = min(len(flat_channels), cu_seqlens.shape[0] - 1)

        for i in range(num_segments):
            channel = flat_channels[i]
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()

            # Bounds check
            if start >= per_token_loss.shape[0] or end > per_token_loss.shape[0]:
                continue
            if start >= end:
                continue

            segment_loss = per_token_loss[start:end]
            segment_mask = valid_mask[start:end]
            valid_loss = segment_loss[segment_mask]

            if valid_loss.numel() > 0:
                # Filter NaN/Inf values (Context Parallel safety)
                finite_mask = torch.isfinite(valid_loss)
                valid_loss = valid_loss[finite_mask]

                if valid_loss.numel() > 0:
                    key = f"{prefix}{channel}"
                    stats[key]["sum"] += valid_loss.sum().item()
                    stats[key]["count"] += valid_loss.numel()
