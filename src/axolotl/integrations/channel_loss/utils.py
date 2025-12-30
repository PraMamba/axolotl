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
Utility functions for Channel Loss integration.

This module provides helper functions for Context Parallelism detection
and other utilities needed by the Channel Loss plugin.
"""

from typing import Optional

import torch.distributed as dist


def _get_context_parallel_group(trainer) -> Optional[dist.ProcessGroup]:
    """
    Extract Context Parallel process group from trainer.

    This function attempts to detect if Context Parallelism (CP) is enabled
    by checking for CP-related attributes in the trainer instance.

    Detection Strategy:
        1. Check if trainer has device_mesh (modern Axolotl with CP)
        2. Extract CP sub-mesh from device_mesh
        3. Fallback: Check for ring_attn_group (older CP implementation)

    Args:
        trainer: The HuggingFace Trainer instance.

    Returns:
        torch.distributed.ProcessGroup if CP is enabled, None otherwise.

    Example:
        >>> cp_group = _get_context_parallel_group(trainer)
        >>> if cp_group is not None:
        >>>     cp_size = dist.get_world_size(cp_group)
        >>>     print(f"Context Parallelism detected: CP size = {cp_size}")
        >>> else:
        >>>     print("Context Parallelism not detected (CP size = 1)")
    """
    # Method 1: Check if trainer has device_mesh (modern Axolotl)
    if hasattr(trainer, "device_mesh") and trainer.device_mesh is not None:
        try:
            # Get CP sub-mesh from DeviceMesh
            # DeviceMesh structure: DeviceMesh(["dp_shard", "cp", "tp"])
            cp_mesh = trainer.device_mesh.get("cp", None)
            if cp_mesh is not None:
                return cp_mesh.get_group()
        except (KeyError, AttributeError):
            pass

    # Method 2: Check if ring_attn_group is set (fallback for older implementations)
    try:
        from axolotl.monkeypatch.ring_attn import get_ring_attn_group

        ring_group = get_ring_attn_group()
        if ring_group is not None:
            return ring_group
    except (ImportError, AttributeError):
        pass

    # No CP detected
    return None
