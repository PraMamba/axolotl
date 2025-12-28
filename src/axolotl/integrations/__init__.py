"""Axolotl integrations package.

Some integrations are referenced in configs as `axolotl.integrations.<name>` (without an
explicit class). The plugin loader expects `module_name.class_name`, so we provide a
lazy attribute bridge for those module-level references.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "ulysses_ring_attn":
        from axolotl.integrations.ulysses_ring_attn import UlyssesRingAttentionPlugin

        return UlyssesRingAttentionPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ulysses_ring_attn",
]
