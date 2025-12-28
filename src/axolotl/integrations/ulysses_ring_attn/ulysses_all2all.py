"""Ulysses all-to-all autograd function for sequence parallel attention.

Implements the _SeqAllToAll autograd function that scatters on sequence dimension
and gathers on heads dimension (forward), with reversed scatter/gather for backward.

This enables Ulysses-style sequence parallelism where each rank processes a subset
of heads over the full sequence length (after all-to-all).

References:
    - ms-swift: swift/trainers/sequence_parallel/ulysses.py (lines 133-156)
    - Spec: specs/006-ulysses-ring-attention-plugin/README.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


class _SeqAllToAll(torch.autograd.Function):
    """All-to-all autograd function for Ulysses sequence parallelism.

    Forward: Scatter on sequence (dim=scatter_idx), gather on heads (dim=gather_idx)
    Backward: Reverse the operation (swap scatter and gather indices)

    Shape transformation example (sp_world_size=4):
        Forward (scatter_idx=1, gather_idx=2):
            Input:  [batch, seq_len, num_heads, head_dim]  # seq_len on this rank
            Output: [batch, seq_len*4, num_heads/4, head_dim]  # full seq, subset of heads

        Backward:
            grad_output: [batch, seq_len*4, num_heads/4, head_dim]
            grad_input:  [batch, seq_len, num_heads, head_dim]
    """

    @staticmethod
    def forward(
        ctx,
        group: ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> torch.Tensor:
        """Perform all-to-all scatter-gather.

        Args:
            ctx: Autograd context for saving backward info
            group: Process group for all-to-all communication
            input: Input tensor to scatter-gather
            scatter_idx: Dimension to split (scatter) across ranks
            gather_idx: Dimension to concatenate (gather) from ranks

        Returns:
            Output tensor after all-to-all transformation
        """
        # Save for backward
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        # Get world size
        world_size = dist.get_world_size(group)

        # Split input along scatter_idx into world_size chunks
        input_list = [
            t.contiguous() for t in torch.tensor_split(input, world_size, scatter_idx)
        ]

        # Prepare output list
        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

        # Perform all-to-all
        dist.all_to_all(output_list, input_list, group=group)

        # Concatenate along gather_idx
        output = torch.cat(output_list, dim=gather_idx).contiguous()

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass: reverse all-to-all (swap scatter and gather).

        Args:
            grad_output: Gradient of output

        Returns:
            Tuple of gradients (None for group, grad_input, None, None)
        """
        # Backward is just all-to-all with swapped scatter/gather indices
        grad_input = _SeqAllToAll.apply(
            ctx.group, grad_output, ctx.gather_idx, ctx.scatter_idx
        )

        # Return gradients: (group, input, scatter_idx, gather_idx)
        return None, grad_input, None, None


def ulysses_all_to_all_scatter_gather(
    group: ProcessGroup,
    input: torch.Tensor,
    scatter_idx: int = 1,
    gather_idx: int = 2,
) -> torch.Tensor:
    """Convenience wrapper for Ulysses all-to-all.

    Args:
        group: Sequence parallel process group
        input: Input tensor [batch, seq_local, num_heads, head_dim]
        scatter_idx: Dimension to scatter (default: 1 = sequence)
        gather_idx: Dimension to gather (default: 2 = heads)

    Returns:
        Output tensor [batch, seq_full, num_heads/sp, head_dim]

    Example:
        >>> sp_world_size = 4
        >>> q = torch.randn(2, 1024, 32, 128)  # batch=2, seq=1024, heads=32, dim=128
        >>> q_all2all = ulysses_all_to_all_scatter_gather(sp_group, q)
        >>> q_all2all.shape
        torch.Size([2, 4096, 8, 128])  # seq×4, heads/4
    """
    return _SeqAllToAll.apply(group, input, scatter_idx, gather_idx)


def ulysses_all_to_all_gather_scatter(
    group: ProcessGroup,
    input: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
) -> torch.Tensor:
    """Reverse Ulysses all-to-all (gather on sequence, scatter on heads).

    This is the inverse operation used after ring-attention to restore
    the original sequence/heads layout.

    Args:
        group: Sequence parallel process group
        input: Input tensor [batch, seq_full, num_heads/sp, head_dim]
        scatter_idx: Dimension to scatter (default: 2 = heads)
        gather_idx: Dimension to gather (default: 1 = sequence)

    Returns:
        Output tensor [batch, seq_local, num_heads, head_dim]
    """
    return _SeqAllToAll.apply(group, input, scatter_idx, gather_idx)


# Placeholder for GQA/MQA support (Phase 1.3)
# TODO: Add support for different Q/K/V head counts
# def ulysses_all_to_all_gqa(...):
#     pass
