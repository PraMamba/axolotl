"""Ulysses + Ring-Attention configuration arguments."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class UlyssesRingAttentionArgs(BaseModel):
    """Configuration for Ulysses + Ring-Attention plugin.

    Enables arbitrary context parallelism by combining Ulysses attention (all-to-all
    on heads) with Ring-Attention (ring communication on sequence). This removes the
    constraint that context_parallel_size must divide num_heads.

    The plugin automatically decomposes context_parallel_size (W) into:
        sp_world_size = gcd(num_heads, W)  # Ulysses dimension
        rp_world_size = W / sp_world_size   # Ring dimension

    Example:
        num_heads=32, context_parallel_size=24
        → sp=8 (Ulysses on 8 ranks), rp=3 (Ring over 3 groups)
    """

    ulysses_ring_attention_enabled: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Enable Ulysses + Ring-Attention for arbitrary chunk sharding in context parallelism"
        },
    )

    ulysses_ring_attention_mode: (
        Literal["auto", "hybrid", "ulysses_only", "ring_only"] | None
    ) = Field(
        default="auto",
        json_schema_extra={
            "description": (
                "Mode for Ulysses + Ring-Attention decomposition:\n"
                "  - auto: Automatically compute sp/rp using gcd(num_heads, cp_size)\n"
                "  - hybrid: Force both Ulysses and Ring (sp>1, rp>1)\n"
                "  - ulysses_only: Force Ulysses-only (sp=cp_size, rp=1, requires divisibility)\n"
                "  - ring_only: Force Ring-only (sp=1, rp=cp_size)"
            )
        },
    )

    ulysses_ring_attention_sp_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Manual override for sequence parallel (Ulysses) world size. "
                "If None, automatically computed as gcd(num_heads, context_parallel_size). "
                "Must divide both num_heads and context_parallel_size."
            )
        },
    )

    ulysses_ring_attention_rp_size: int | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Manual override for ring parallel world size. "
                "If None, automatically computed as context_parallel_size / sp_size. "
                "Must satisfy sp_size × rp_size = context_parallel_size."
            )
        },
    )

    ulysses_ring_attention_backend: str | None = Field(
        default="ring_flash_attn_llama3",
        json_schema_extra={
            "description": (
                "Ring-attention backend to use. Phase 1 only supports 'ring_flash_attn_llama3' "
                "which uses varlen flash-attn API with packed sequences."
            )
        },
    )

    ulysses_ring_attention_require_padding_free: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": (
                "Phase 1 constraint: Require sequences to be divisible by 2*rp_size. "
                "Set to False to enable auto-padding (Phase 2 feature, not yet implemented)."
            )
        },
    )

    @field_validator("ulysses_ring_attention_mode")
    @classmethod
    def validate_mode(cls, v):
        """Validate mode is one of allowed values."""
        valid_modes = ["auto", "hybrid", "ulysses_only", "ring_only"]
        if v and v not in valid_modes:
            raise ValueError(
                f"Invalid ulysses_ring_attention_mode: '{v}'.\n\n"
                f"Valid options: {', '.join(valid_modes)}\n\n"
                f"Recommended: 'auto' (computes optimal sp/rp split automatically)"
            )
        return v

    @model_validator(mode="after")
    def validate_sp_rp_consistency(self):
        """Validate that sp_size and rp_size are consistent if manually specified."""
        sp = self.ulysses_ring_attention_sp_size
        rp = self.ulysses_ring_attention_rp_size

        # If both are manually specified, they must be consistent
        # (actual validation against context_parallel_size happens in plugin.register)
        if sp is not None and rp is not None:
            if sp <= 0 or rp <= 0:
                raise ValueError(
                    f"Invalid sp_size={sp} or rp_size={rp}. Both must be positive integers."
                )

        return self

    @field_validator("ulysses_ring_attention_sp_size", "ulysses_ring_attention_rp_size")
    @classmethod
    def validate_positive(cls, v):
        """Validate that sp/rp sizes are positive if specified."""
        if v is not None and v <= 0:
            raise ValueError(f"sp_size and rp_size must be positive integers, got {v}")
        return v
