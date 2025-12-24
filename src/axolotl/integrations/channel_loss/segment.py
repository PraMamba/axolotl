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
Segment detection strategies for Channel Loss.

Handles both standard padding mode and packing (multipack) mode.

Design notes:
- Axolotl V2 Collator uses attention_mask as segment IDs (1, 2, 3, ...), not 0/1
- Swift uses position_ids == 0 to detect segment boundaries
- 'auto' strategy: prefer attention_mask segment IDs, fallback to position_ids
"""

from typing import Literal, Optional

import torch


def get_segment_boundaries(
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    labels: torch.Tensor,
    mode: Literal["auto", "position_ids", "attention_mask"] = "auto",
) -> torch.Tensor:
    """
    Get segment boundaries (cu_seqlens) for channel loss calculation.

    Args:
        attention_mask: Attention mask tensor. In V2 Collator, values are segment IDs (1,2,3...),
                       not binary 0/1. Shape: (batch_size, seq_len) or (seq_len,)
        position_ids: Position IDs tensor. In packing mode, resets to 0 at segment start.
                     Shape: (batch_size, seq_len) or (seq_len,)
        labels: Labels tensor. Shape: (batch_size, seq_len)
        mode: Segment detection mode:
              - 'auto': prefer attention_mask segment IDs, fallback to position_ids
              - 'position_ids': use position_ids == 0 as segment boundary
              - 'attention_mask': use attention_mask value changes as segment boundary

    Returns:
        cu_seqlens: Cumulative sequence lengths tensor. Shape: (num_segments + 1,)
                   Example: [0, 128, 256, 384] means 3 segments of length 128 each.
    """
    batch_size, seq_len = labels.shape
    device = labels.device

    # Flatten for easier processing (assuming batch_size=1 for packing mode)
    # For standard padding mode, we treat each batch item as a segment

    if mode == "auto":
        mode = _detect_mode(attention_mask, position_ids)

    if mode == "attention_mask" and attention_mask is not None:
        return _get_boundaries_from_attention_mask(attention_mask, device)

    if mode == "position_ids" and position_ids is not None:
        return _get_boundaries_from_position_ids(position_ids, device)

    # Fallback: each batch item is a segment (standard padding mode)
    # For labels of shape (batch_size, seq_len), after shift we have (seq_len - 1) tokens per item
    cu_seqlens = torch.arange(0, batch_size + 1, device=device) * (seq_len - 1)
    return cu_seqlens


def _detect_mode(
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
) -> Literal["attention_mask", "position_ids", "fallback"]:
    """
    Auto-detect the best segment detection mode.

    V2 Collator uses attention_mask as segment IDs (values > 1).
    Standard Collator uses binary attention_mask (0/1).
    """
    if attention_mask is not None:
        attn_flat = attention_mask.view(-1)
        max_val = attn_flat.max().item()
        if max_val > 1:
            # V2 Collator format: attention_mask contains segment IDs (1, 2, 3, ...)
            return "attention_mask"

    if position_ids is not None:
        # Swift-style: use position_ids == 0 as segment start
        return "position_ids"

    return "fallback"


def _get_boundaries_from_attention_mask(
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Get segment boundaries from attention_mask segment IDs.

    In V2 Collator, attention_mask values are segment IDs (1, 2, 3, ...),
    and 0 indicates padding.

    Example:
        attention_mask = [1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 0]
        -> boundaries at positions where value changes
        -> cu_seqlens = [0, 3, 7, 9]  (excluding padding)
    """
    attn_flat = attention_mask.view(-1)

    # Find positions where attention_mask value changes
    # This gives us segment boundaries
    changes = torch.where(attn_flat[1:] != attn_flat[:-1])[0] + 1

    # Filter out padding transitions (where mask becomes 0 or from 0)
    # We only want boundaries between valid segments
    valid_changes = []
    for i, pos in enumerate(changes):
        prev_val = attn_flat[pos - 1].item() if pos > 0 else 0
        curr_val = attn_flat[pos].item()
        # Only keep transitions between non-zero values, or from non-zero to zero (end of last segment)
        if prev_val > 0:
            valid_changes.append(pos)

    # Build cu_seqlens
    cu_seqlens = torch.tensor([0] + valid_changes + [attn_flat.shape[0]], device=device)

    # Remove duplicate endings (if last valid segment ends at the array end)
    cu_seqlens = torch.unique_consecutive(cu_seqlens)

    return cu_seqlens


def _get_boundaries_from_position_ids(
    position_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Get segment boundaries from position_ids.

    In packing mode, position_ids resets to 0 at the start of each sample.

    Example:
        position_ids = [0, 1, 2, 0, 1, 2, 3, 0, 1]
        -> segment starts at positions 0, 3, 7
        -> cu_seqlens = [0, 3, 7, 9]
    """
    pos_flat = position_ids.view(-1)

    # Find positions where position_ids == 0 (segment starts)
    seq_starts = (pos_flat == 0).nonzero(as_tuple=True)[0]

    # Add ending position
    total_len = pos_flat.shape[0]
    cu_seqlens = torch.cat([
        seq_starts,
        torch.tensor([total_len], device=device)
    ])

    return cu_seqlens


def flatten_channels(channels) -> list:
    """
    Flatten nested channel list for packing mode.

    Args:
        channels: Channel information. Can be:
                 - List[str]: standard mode, one channel per batch item
                 - List[List[str]]: packing mode, multiple channels per packed row

    Returns:
        Flattened list of channel strings.

    Example:
        [['math', 'code'], ['general']] -> ['math', 'code', 'general']
    """
    if not channels:
        return []

    if isinstance(channels[0], list):
        # Packing mode: flatten nested lists
        return [ch for sub in channels for ch in sub]

    return list(channels)
