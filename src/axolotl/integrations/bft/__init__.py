"""Composable BFT (Balanced Fine-Tuning) plugin for Axolotl."""

from __future__ import annotations

from transformers import Trainer

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class BFTPlugin(BasePlugin):
    """Enable BFT (Balanced Fine-Tuning) loss in Axolotl SFT training.

    BFT extends DFT with sample-level weighting based on per-sample difficulty,
    allowing the model to focus on hard samples while maintaining stable gradients.
    """

    def get_input_args(self) -> str:
        return "axolotl.integrations.bft.args.BFTArgs"

    def get_training_args_mixin(self) -> str:
        return "axolotl.integrations.bft.args.BFTTrainingArgsMixin"

    def get_training_args(self, cfg: DictDefault) -> dict:
        if not cfg.enable_bft_loss:
            return {}
        return {"enable_bft_loss": True}

    def get_trainer_cls(self, cfg: DictDefault) -> None:
        return None

    def post_trainer_create(self, cfg: DictDefault, trainer: Trainer) -> None:
        if not cfg.enable_bft_loss:
            return

        # Check for incompatibilities
        if cfg.rl or cfg.reward_model or cfg.process_reward_model:
            LOG.warning("BFTPlugin is intended for SFT; skipping for RL/Reward paths.")
            return

        # Check for DFT conflict (mutually exclusive)
        if getattr(cfg, "enable_dft_loss", False):
            msg = (
                "BFT and DFT are mutually exclusive. "
                "Please set enable_dft_loss=false when using BFT. "
                "BFT already includes DFT's token-level weighting."
            )
            raise ValueError(msg)

        # Check for label smoothing (inherited incompatibility)
        if getattr(cfg, "label_smoothing_factor", 0.0) not in (0, 0.0, None):
            msg = (
                "BFT is incompatible with label smoothing (label_smoothing_factor > 0). "
                "BFT relies on accurate token probabilities from cross-entropy loss."
            )
            raise ValueError(msg)

        from .patch import patch_compute_loss_for_bft

        patch_compute_loss_for_bft(trainer, cfg)
        LOG.info("BFTPlugin: patched trainer.compute_loss with sample-level weighting")
