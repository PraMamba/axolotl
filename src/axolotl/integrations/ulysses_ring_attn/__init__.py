"""Ulysses + Ring-Attention plugin for arbitrary chunk sharding in context parallelism.

This plugin enables Axolotl to use context_parallel_size values that don't divide
num_heads by using a hybrid approach:
- Ulysses attention (all-to-all on heads dimension) for gcd(num_heads, cp_size) ranks
- Ring-attention (ring communication on sequence dimension) for remaining ranks

Example:
    num_heads=32, context_parallel_size=24
    → sp=gcd(32,24)=8, rp=24/8=3
    → 8-way Ulysses × 3-way Ring-Attention

References:
    - ms-swift implementation: swift/trainers/sequence_parallel/ulysses.py
    - Analysis: docs/analysis/ms_swift_ulysses_ring_attention_implementation.md
    - Spec: specs/006-ulysses-ring-attention-plugin/README.md
"""

from axolotl.integrations.ulysses_ring_attn.args import UlyssesRingAttentionArgs
from axolotl.integrations.ulysses_ring_attn.plugins import UlyssesRingAttentionPlugin

__all__ = ["UlyssesRingAttentionPlugin", "UlyssesRingAttentionArgs"]
