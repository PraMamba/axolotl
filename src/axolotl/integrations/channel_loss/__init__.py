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
Channel Loss Plugin for Axolotl.

Ported from ms-swift framework. Enables per-channel loss tracking during training
without affecting the training dynamics. Useful for monitoring loss on different
data sources (e.g., math, code, general) during multi-domain training.

Usage in YAML config:
    plugins:
      - axolotl.integrations.channel_loss.ChannelLossPlugin

    enable_channel_loss: true
    channel_loss_field: "channel"
    channel_loss_prefix: "loss_"
    channel_loss_segment: "auto"

    datasets:
      - path: /data/math.jsonl
        channel: math
      - path: /data/code.jsonl
        channel: code

Design notes:
- Uses composite Plugin pattern (post_trainer_create) instead of get_trainer_cls
- This ensures compatibility with other plugins (KD, GRPO, etc.)
- Observer-only: does not modify original loss or gradients
- Supports standard batch and packing modes
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List

from transformers import Trainer

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .callback import ChannelLossLoggingCallback
from .collator_wrapper import wrap_collator_for_channel_loss
from .compute_loss_patch import patch_compute_loss_for_channel_loss

LOG = get_logger(__name__)


class ChannelLossPlugin(BasePlugin):
    """
    Plugin for per-channel loss tracking.

    This plugin wraps the trainer's compute_loss method and data collator
    to track loss metrics per channel (data source) without affecting
    the training dynamics.

    Key features:
    - Observer-only: does not modify loss or gradients
    - Compatible with any Trainer class (works with KD, GRPO, etc.)
    - Supports standard batch and packing modes
    - Distributed training compatible (DDP, ZeRO2/3, FSDP)
    """

    def register(self, cfg: dict) -> None:
        """
        Register the plugin with the configuration.

        This runs before validate_config and handles schema limitations.
        Specifically, it extracts channel info from dataset configs since
        SFTDataset schema doesn't support arbitrary fields.

        Args:
            cfg: Raw configuration dict (before validation).
        """
        if not cfg.get("enable_channel_loss"):
            return

        LOG.info("Channel Loss Plugin: Registering...")

        # === Hard Conflicts: Features that prevent Channel Loss from working ===
        # These will raise ValueError to fail early with clear error messages

        # 1. Liger Fused Linear Cross Entropy (FLCE)
        if cfg.get("liger_fused_linear_cross_entropy"):
            raise ValueError(
                "Channel Loss is incompatible with liger_fused_linear_cross_entropy.\n\n"
                "Reason: Liger FLCE skips logits materialization in training mode (skip_logits=True)\n"
                "to save memory, but Channel Loss requires access to logits for per-channel statistics.\n\n"
                "Solutions:\n"
                "  1. Use 'chunked_cross_entropy: true' instead (compatible, saves memory)\n"
                "  2. Use 'liger_cross_entropy: true' (non-fused, partial optimization)\n"
                "  3. Disable Channel Loss if Liger FLCE is critical for your memory budget\n\n"
                "See: specs/001-channel-loss-compatibility-audit.md for details"
            )

        # 2. Knowledge Distillation (KD) Trainer
        if cfg.get("kd_trainer"):
            raise ValueError(
                "Channel Loss is incompatible with KD trainer.\n\n"
                "Reason: KD's compute_loss() method does not support return_outputs=True,\n"
                "preventing Channel Loss from accessing model outputs and logits.\n\n"
                "Solutions:\n"
                "  1. Disable Channel Loss for KD training\n"
                "  2. Wait for KD Trainer fix (track issue in GitHub)\n"
                "  3. Use standard SFT training if Channel Loss is required\n\n"
                "See: specs/001-channel-loss-compatibility-audit.md for details"
            )

        # === Soft Conflicts: Cut Cross Entropy (auto-disable) ===
        # CCE is incompatible, but we can auto-disable it for user convenience

        cce_plugin = "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"
        plugins = cfg.get("plugins", [])

        if cce_plugin in plugins:
            LOG.warning(
                "Channel Loss Plugin: Cut Cross Entropy detected. "
                "These features are incompatible because CCE does not materialize logits. "
                "Disabling Cut Cross Entropy..."
            )
            # CCE plugin checks cfg.cut_cross_entropy flag in pre_model_load()
            # Setting this to False prevents CCE from being applied
            cfg["cut_cross_entropy"] = False
            LOG.info("Channel Loss Plugin: Disabled Cut Cross Entropy (set cut_cross_entropy=False)")

        # === Semantic Warnings: RL Training ===
        # Technically compatible but semantically questionable

        rl_types = ["dpo", "kto", "orpo", "simpo", "grpo"]
        current_rl = cfg.get("rl")
        if current_rl and current_rl.lower() in rl_types:
            LOG.warning(
                f"Channel Loss enabled with RL training mode: {current_rl.upper()}\n"
                f"Note: RL training uses sample-level preference loss, not per-token causal loss.\n"
                f"Per-channel statistics may not be meaningful for this training paradigm.\n"
                f"Consider whether channel-level monitoring makes sense for your use case."
            )

        # Extract channel from dataset configs
        # This is necessary because SFTDataset schema doesn't have 'channel' field
        # and validate_config would discard it
        channel_map = []
        for ds in cfg.get("datasets", []):
            # Pop channel to avoid schema validation errors
            ch = ds.pop("channel", None)
            channel_map.append(ch)

        # Store in private config key for later use
        cfg["_channel_loss_dataset_channels"] = channel_map

        LOG.info(
            f"Channel Loss Plugin: Extracted channels from {len(channel_map)} datasets"
        )

    def get_input_args(self) -> str | None:
        """
        Returns the Pydantic model for input configuration.

        Returns:
            Fully qualified class name of the Pydantic model.
        """
        return "axolotl.integrations.channel_loss.args.ChannelLossArgs"

    def get_training_args_mixin(self) -> str | None:
        """
        Returns the dataclass mixin for training arguments.

        Returns:
            Fully qualified class name of the dataclass mixin.
        """
        return "axolotl.integrations.channel_loss.args.ChannelLossTrainingArgsMixin"

    def post_trainer_create(self, cfg: DictDefault, trainer: Trainer) -> None:
        """
        Perform setup after trainer is created.

        This is the main injection point. We:
        1. Preserve channel field in dataset (prevent Trainer from removing it)
        2. Wrap the data collator to handle channel field
        3. Patch compute_loss to track per-channel statistics
        4. Initialize the statistics accumulator

        Args:
            cfg: Axolotl configuration.
            trainer: The created Trainer instance.
        """
        if not cfg.get("enable_channel_loss"):
            return

        LOG.info("Channel Loss Plugin: Setting up trainer...")

        # Get configuration values
        channel_field = cfg.get("channel_loss_field", "channel")
        dataset_channels = cfg.get("_channel_loss_dataset_channels")
        warn_on_missing = cfg.get("channel_loss_warn_on_missing", True)

        # 0. Prevent Trainer from removing channel field
        # HuggingFace Trainer removes columns not in model signature via _remove_unused_columns
        # We need to preserve the channel field for our collator
        self._preserve_channel_field(trainer, channel_field)

        # 1. Wrap data collator
        if hasattr(trainer, "data_collator") and trainer.data_collator is not None:
            trainer.data_collator = wrap_collator_for_channel_loss(
                inner_collator=trainer.data_collator,
                channel_field=channel_field,
                dataset_channels=dataset_channels,
                warn_on_missing=warn_on_missing,
            )
            LOG.debug("Channel Loss Plugin: Wrapped train data collator")

        # Also wrap eval collator if it exists
        if hasattr(trainer, "eval_data_collator") and trainer.eval_data_collator is not None:
            trainer.eval_data_collator = wrap_collator_for_channel_loss(
                inner_collator=trainer.eval_data_collator,
                channel_field=channel_field,
                dataset_channels=dataset_channels,
                warn_on_missing=warn_on_missing,
            )
            LOG.debug("Channel Loss Plugin: Wrapped eval data collator")

        # 2. Patch compute_loss
        patch_compute_loss_for_channel_loss(trainer, cfg)

        # 3. Initialize statistics accumulator
        # Using defaultdict to auto-create entries for new channels
        trainer._channel_loss_stats = {
            "train": defaultdict(lambda: {"sum": 0.0, "count": 0}),
            "eval": defaultdict(lambda: {"sum": 0.0, "count": 0}),
        }

        LOG.info("Channel Loss Plugin: Setup complete")

    def _preserve_channel_field(self, trainer: Trainer, channel_field: str) -> None:
        """
        Prevent Trainer from removing channel field via _remove_unused_columns.

        HuggingFace Trainer automatically removes columns that are not in the model's
        forward() signature. We need to keep the channel field for our collator.

        Args:
            trainer: The Trainer instance.
            channel_field: Name of the channel field to preserve.
        """
        # Store original method
        original_remove_unused_columns = trainer._remove_unused_columns

        def _remove_unused_columns_with_channel_preservation(
            dataset, description: str | None = None
        ):
            """Wrapper that preserves channel field."""
            # Initialize _signature_columns if it's None
            # When None, Trainer uses model's forward signature automatically
            # We need to preserve channel_field plus standard training fields
            if trainer._signature_columns is None:
                # Standard fields that model expects + our channel field
                # These are the typical fields for causal LM training
                standard_fields = ["input_ids", "labels", "attention_mask", "position_ids", "token_type_ids"]

                if hasattr(dataset, 'column_names'):
                    # Only keep standard fields that exist in dataset, plus channel field
                    trainer._signature_columns = [
                        col for col in standard_fields if col in dataset.column_names
                    ]
                    # Add channel field
                    if channel_field not in trainer._signature_columns:
                        trainer._signature_columns.append(channel_field)
            else:
                # Add channel field to existing signature columns
                if channel_field not in trainer._signature_columns:
                    trainer._signature_columns = list(trainer._signature_columns) + [channel_field]

            # Call original method
            result = original_remove_unused_columns(dataset, description)

            return result

        # Replace the method
        trainer._remove_unused_columns = _remove_unused_columns_with_channel_preservation
        LOG.info(f"Channel Loss Plugin: Protected '{channel_field}' field from removal")

    def add_callbacks_post_trainer(
        self, cfg: DictDefault, trainer: Trainer
    ) -> List[Callable]:
        """
        Add callbacks after trainer creation.

        Returns the ChannelLossLoggingCallback which handles
        distributed synchronization and metric logging.

        IMPORTANT: This callback MUST execute BEFORE other logging callbacks
        (like SwanLab, WandB) to ensure channel metrics are added to logs first.
        We achieve this by directly inserting into trainer.callback_handler.callbacks
        at the beginning of the list.

        Args:
            cfg: Axolotl configuration.
            trainer: The created Trainer instance.

        Returns:
            List of callback instances to add.
        """
        if not cfg.get("enable_channel_loss"):
            return []

        # Create callback instance
        callback = ChannelLossLoggingCallback(trainer, cfg)

        # CRITICAL: Insert at the beginning of existing callbacks to ensure
        # it runs BEFORE other logging callbacks like SwanLab/WandB
        if hasattr(trainer, 'callback_handler') and hasattr(trainer.callback_handler, 'callbacks'):
            trainer.callback_handler.callbacks.insert(0, callback)
            LOG.info("Channel Loss Plugin: Inserted callback at beginning of callback list")
            return []  # Don't return it since we already added it
        else:
            LOG.debug("Channel Loss Plugin: Adding logging callback (normal flow)")
            return [callback]


# Export main classes
__all__ = ["ChannelLossPlugin"]
