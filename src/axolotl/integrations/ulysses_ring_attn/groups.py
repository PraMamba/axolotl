"""GCD decomposition and process group creation for Ulysses + Ring-Attention.

This module implements the core logic for decomposing context_parallel_size into
sequence parallel (sp) and ring parallel (rp) dimensions using GCD, and creates
the corresponding PyTorch distributed process groups.

Key insight:
    sp_world_size = gcd(num_heads, context_parallel_size)
    rp_world_size = context_parallel_size / sp_world_size

This ensures sp_world_size always divides num_heads (Ulysses requirement),
while rp_world_size has no head constraints (Ring-Attention operates on sequence).

References:
    - ms-swift: swift/trainers/sequence_parallel/ulysses.py (lines 732-760)
    - Spec: specs/006-ulysses-ring-attention-plugin/README.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch.distributed as dist

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def compute_sp_rp(
    num_heads: int,
    context_parallel_size: int,
    sp_size_override: int | None = None,
    rp_size_override: int | None = None,
    mode: str = "auto",
) -> tuple[int, int]:
    """Compute sequence parallel (sp) and ring parallel (rp) world sizes.

    Args:
        num_heads: Number of attention heads in the model
        context_parallel_size: Total context parallelism world size (W)
        sp_size_override: Manual override for sp_world_size (optional)
        rp_size_override: Manual override for rp_world_size (optional)
        mode: Decomposition mode ('auto', 'hybrid', 'ulysses_only', 'ring_only')

    Returns:
        (sp_world_size, rp_world_size) tuple

    Raises:
        ValueError: If constraints are violated (divisibility, consistency)

    Examples:
        >>> compute_sp_rp(32, 8)   # Divisible
        (8, 1)  # Ulysses-only

        >>> compute_sp_rp(32, 24)  # Non-divisible
        (8, 3)  # Hybrid: 8-way Ulysses × 3-way Ring

        >>> compute_sp_rp(32, 7)   # Prime
        (1, 7)  # Ring-only

        >>> compute_sp_rp(40, 12)  # GQA
        (4, 3)  # Hybrid: 4-way Ulysses × 3-way Ring
    """
    # Handle manual overrides
    if sp_size_override is not None and rp_size_override is not None:
        sp = sp_size_override
        rp = rp_size_override

        # Validate consistency
        if sp * rp != context_parallel_size:
            raise ValueError(
                f"Manual sp_size={sp} and rp_size={rp} don't satisfy: "
                f"sp × rp = context_parallel_size ({context_parallel_size}). "
                f"Got: {sp} × {rp} = {sp * rp}"
            )

        # Validate head divisibility for Ulysses
        if sp > 1 and num_heads % sp != 0:
            raise ValueError(
                f"Manual sp_size={sp} doesn't divide num_heads={num_heads}. "
                f"Ulysses requires sp_size | num_heads."
            )

        LOG.info(
            f"Using manual sp/rp override: sp={sp}, rp={rp} "
            f"(mode={mode}, num_heads={num_heads}, cp={context_parallel_size})"
        )
        return sp, rp

    # Auto-compute based on mode
    if mode == "ulysses_only":
        # Force Ulysses-only (requires divisibility)
        if num_heads % context_parallel_size != 0:
            raise ValueError(
                f"Mode 'ulysses_only' requires num_heads ({num_heads}) to be divisible "
                f"by context_parallel_size ({context_parallel_size}). "
                f"Use mode='auto' for automatic decomposition."
            )
        sp = context_parallel_size
        rp = 1

    elif mode == "ring_only":
        # Force Ring-only
        sp = 1
        rp = context_parallel_size

    elif mode in ("auto", "hybrid"):
        # GCD-based decomposition
        sp = math.gcd(num_heads, context_parallel_size)
        rp = context_parallel_size // sp

        # For 'hybrid' mode, verify both sp>1 and rp>1
        if mode == "hybrid" and (sp == 1 or rp == 1):
            raise ValueError(
                f"Mode 'hybrid' requires both sp>1 and rp>1, but got sp={sp}, rp={rp}. "
                f"gcd({num_heads}, {context_parallel_size}) = {sp}. "
                f"Use mode='auto' to allow fallback to Ulysses-only or Ring-only."
            )

    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Final validation
    assert sp * rp == context_parallel_size, (
        f"Internal error: sp × rp != cp ({sp} × {rp} != {context_parallel_size})"
    )
    assert sp >= 1 and rp >= 1, f"Internal error: invalid sp={sp} or rp={rp}"
    if sp > 1:
        assert num_heads % sp == 0, (
            f"Internal error: sp={sp} doesn't divide num_heads={num_heads}"
        )

    LOG.info(
        f"Computed sp/rp decomposition: sp={sp}, rp={rp} "
        f"(mode={mode}, gcd={math.gcd(num_heads, context_parallel_size)}, "
        f"num_heads={num_heads}, cp={context_parallel_size})"
    )

    return sp, rp


def create_ulysses_ring_groups(
    context_parallel_group: ProcessGroup,
    sp_world_size: int,
    rp_world_size: int,
) -> tuple[ProcessGroup, ProcessGroup, int, int]:
    """Create sequence parallel and ring parallel process groups.

    Given a context parallel group with ranks [0, 1, ..., W-1] where W = sp × rp,
    create two orthogonal process groups:

    1. Sequence Parallel (SP) groups: Consecutive ranks for Ulysses all-to-all
       - Group 0: [0, 1, ..., sp-1]
       - Group 1: [sp, sp+1, ..., 2*sp-1]
       - ...
       - Group (rp-1): [(rp-1)*sp, ..., W-1]
       Total: rp groups, each with sp ranks

    2. Ring Parallel (RP) groups: Strided ranks for ring communication
       - Group 0: [0, sp, 2*sp, ...]
       - Group 1: [1, sp+1, 2*sp+1, ...]
       - ...
       - Group (sp-1): [sp-1, 2*sp-1, ...]
       Total: sp groups, each with rp ranks

    Args:
        context_parallel_group: The full context parallel process group
        sp_world_size: Sequence parallel world size (Ulysses dimension)
        rp_world_size: Ring parallel world size (Ring dimension)

    Returns:
        (sp_group, rp_group, sp_rank, rp_rank) tuple where:
        - sp_group: This rank's sequence parallel group (for all-to-all)
        - rp_group: This rank's ring parallel group (for ring comm)
        - sp_rank: Rank within sp_group [0, sp_world_size)
        - rp_rank: Rank within rp_group [0, rp_world_size)

    Example:
        cp_ranks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # cp=12
        sp=4, rp=3

        SP groups (consecutive):
          Group 0: [0, 1, 2, 3]      (rp_rank=0)
          Group 1: [4, 5, 6, 7]      (rp_rank=1)
          Group 2: [8, 9, 10, 11]    (rp_rank=2)

        RP groups (strided):
          Group 0: [0, 4, 8]     (sp_rank=0)
          Group 1: [1, 5, 9]     (sp_rank=1)
          Group 2: [2, 6, 10]    (sp_rank=2)
          Group 3: [3, 7, 11]    (sp_rank=3)

        Rank 5's groups:
          - SP group: [4, 5, 6, 7]   (sp_rank=1)
          - RP group: [1, 5, 9]      (rp_rank=1)
    """
    # Get all ranks in context parallel group
    cp_world_size = dist.get_world_size(context_parallel_group)
    cp_rank = dist.get_rank(context_parallel_group)

    # Validate consistency
    if sp_world_size * rp_world_size != cp_world_size:
        raise ValueError(
            f"sp_world_size × rp_world_size ({sp_world_size} × {rp_world_size}) "
            f"!= cp_world_size ({cp_world_size})"
        )

    # Get global ranks in CP group (for creating new groups)
    # Note: We assume CP group is a contiguous range of global ranks
    # This is typically true for standard distributed setups
    global_rank = dist.get_rank()
    cp_ranks = list(range(cp_world_size))  # Simplified assumption

    # Compute this rank's sp_rank and rp_rank
    # Using row-major layout: rp_rank = cp_rank // sp, sp_rank = cp_rank % sp
    sp_rank = cp_rank % sp_world_size
    rp_rank = cp_rank // sp_world_size

    # Create SP groups (consecutive ranks)
    sp_groups = []
    for i in range(rp_world_size):
        sp_group_ranks = cp_ranks[i * sp_world_size : (i + 1) * sp_world_size]
        sp_groups.append(sp_group_ranks)

    # Create RP groups (strided ranks)
    rp_groups = []
    for i in range(sp_world_size):
        rp_group_ranks = [cp_ranks[i + k * sp_world_size] for k in range(rp_world_size)]
        rp_groups.append(rp_group_ranks)

    # Create actual PyTorch process groups
    # Find which groups this rank belongs to
    my_sp_group = None
    my_rp_group = None

    for sp_group_ranks in sp_groups:
        group = dist.new_group(sp_group_ranks)
        if cp_rank in sp_group_ranks:
            my_sp_group = group

    for rp_group_ranks in rp_groups:
        group = dist.new_group(rp_group_ranks)
        if cp_rank in rp_group_ranks:
            my_rp_group = group

    if my_sp_group is None or my_rp_group is None:
        raise RuntimeError(
            f"Failed to create groups for rank {cp_rank}. "
            f"my_sp_group={my_sp_group}, my_rp_group={my_rp_group}"
        )

    LOG.info(
        f"Rank {global_rank} (cp_rank={cp_rank}): "
        f"sp_rank={sp_rank}/{sp_world_size}, rp_rank={rp_rank}/{rp_world_size}"
    )

    return my_sp_group, my_rp_group, sp_rank, rp_rank
