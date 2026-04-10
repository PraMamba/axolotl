"""BFT (Balanced Fine-Tuning) plugin arguments.

This module defines configuration options for Balanced Fine-Tuning (BFT) loss,
which extends DFT with sample-level difficulty weighting.

BFT combines:
- Token-level weighting from DFT: w_{b,t} = exp(-CE.detach())
- Sample-level weighting: s_b = 1 - min_window(mean(confidence))

Reference: Tang et al. "Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning" (2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field


class BFTArgs(BaseModel):
    """Input args for Balanced Fine-Tuning (BFT) loss.

    BFT extends DFT with sample-level weighting based on per-sample difficulty.
    Each sample's weight is determined by its hardest local region (sliding window
    min-mean confidence), allowing the model to focus on difficult samples while
    maintaining stable token-level gradients.

    Reference: https://arxiv.org/abs/2501.xxxxx (Tang et al. 2025)
    """

    enable_bft_loss: bool = Field(
        default=False,
        description=(
            "Enable BFT loss: L_bft = s_b * w_{b,t} * CE where s_b is sample weight "
            "based on min-window confidence. Mutually exclusive with enable_dft_loss."
        ),
    )

    bft_group_size: int = Field(
        default=256,
        description=(
            "Sliding window size for computing local confidence. "
            "Larger values smooth over more context but may miss local difficulty spikes. "
            "Paper default: 256."
        ),
    )

    bft_min_valid_tokens_in_window: int = Field(
        default=32,
        description=(
            "Minimum number of valid (non-masked) tokens required in a window. "
            "Windows with fewer valid tokens are skipped to avoid noise from "
            "packing boundaries or excessive masking."
        ),
    )

    bft_weight_floor: float = Field(
        default=0.1,
        description=(
            "Minimum sample weight (prevents easy samples from being dropped). "
            "Recommended: 0.1 (keeps ≥10% weight on easiest samples)."
        ),
    )

    bft_weight_ceiling: float = Field(
        default=2.0,
        description=(
            "Maximum sample weight (prevents gradient explosion on extreme outliers). "
            "Recommended: 2.0 (allows hard samples to be 2x overweighted)."
        ),
    )

    bft_normalize_sample_weight: bool = Field(
        default=True,
        description=(
            "Normalize sample weights to mean=1.0 per batch. "
            "Stabilizes gradient scale across batches with varying difficulty distributions."
        ),
    )

    bft_warmup_steps: int = Field(
        default=100,
        description=(
            "Linear warmup for sample weighting: s_b := s_b * min(step/warmup_steps, 1). "
            "Reduces instability in early training when model confidence is low."
        ),
    )

    dft_chunk_size: Optional[int] = Field(
        default=None,
        description=(
            "Chunk size for memory-efficient cross-entropy computation. "
            "When set (e.g., 2048), logits are processed in chunks to reduce peak memory "
            "by 50-75% for large vocabulary models (e.g., Qwen 152K vocab). "
            "Inherited from DFT for compatibility. "
            "\n\nRecommended values:"
            "\n- vocab_size < 50K: None (chunking not needed)"
            "\n- vocab_size 50K-100K: 2048-4096"
            "\n- vocab_size > 100K: 1024-2048"
        ),
    )


@dataclass
class BFTTrainingArgsMixin:
    """TrainingArguments mixin for BFT."""

    enable_bft_loss: bool = field(
        default=False,
        metadata={
            "help": "Enable BFT loss with sample-level difficulty weighting",
        },
    )

    bft_group_size: int = field(
        default=256,
        metadata={
            "help": "Sliding window size for computing local confidence (paper default: 256)",
        },
    )

    bft_min_valid_tokens_in_window: int = field(
        default=32,
        metadata={
            "help": "Minimum valid tokens in window (skip windows below this threshold)",
        },
    )

    bft_weight_floor: float = field(
        default=0.1,
        metadata={
            "help": "Minimum sample weight (default: 0.1 = 10% of easy samples)",
        },
    )

    bft_weight_ceiling: float = field(
        default=2.0,
        metadata={
            "help": "Maximum sample weight (default: 2.0 = 2x hard samples)",
        },
    )

    bft_normalize_sample_weight: bool = field(
        default=True,
        metadata={
            "help": "Normalize sample weights to mean=1.0 per batch",
        },
    )

    bft_warmup_steps: int = field(
        default=100,
        metadata={
            "help": "Linear warmup steps for sample weighting",
        },
    )

    dft_chunk_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Chunk size for memory-efficient CE computation (inherited from DFT). "
                "Recommended: 2048 for vocab > 100K, None for vocab < 50K."
            ),
        },
    )
