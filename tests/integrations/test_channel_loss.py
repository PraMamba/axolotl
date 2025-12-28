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
Unit tests for Channel Loss Plugin.

Tests the core functionality of the Channel Loss feature ported from ms-swift.
"""

import pytest
import torch

from axolotl.integrations.channel_loss.collator_wrapper import (
    wrap_collator_for_channel_loss,
)
from axolotl.integrations.channel_loss.segment import (
    flatten_channels,
    get_segment_boundaries,
)


class TestSegmentBoundaries:
    """Tests for segment boundary detection."""

    def test_attention_mask_segment_ids(self):
        """Test V2 Collator format where attention_mask contains segment IDs."""
        # V2 Collator uses attention_mask as segment IDs: 1, 2, 3, ...
        attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 0]])
        labels = torch.ones(1, 11, dtype=torch.long)

        cu_seqlens = get_segment_boundaries(
            attention_mask=attention_mask,
            position_ids=None,
            labels=labels,
            mode="attention_mask",
        )

        # Expect boundaries at value changes: [0, 3, 7, 9, 11]
        assert cu_seqlens[0].item() == 0
        # First segment ends at position 3
        assert 3 in cu_seqlens.tolist()

    def test_position_ids_segment_detection(self):
        """Test Swift-style segment detection using position_ids == 0."""
        # Position IDs reset to 0 at each sample start
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0, 1]])
        labels = torch.ones(1, 9, dtype=torch.long)

        cu_seqlens = get_segment_boundaries(
            attention_mask=None,
            position_ids=position_ids,
            labels=labels,
            mode="position_ids",
        )

        # Expect boundaries at positions 0, 3, 7, 9
        expected = [0, 3, 7, 9]
        assert cu_seqlens.tolist() == expected

    def test_auto_mode_prefers_attention_mask(self):
        """Test auto mode prefers attention_mask when it contains segment IDs."""
        # V2 format: max value > 1
        attention_mask = torch.tensor([[1, 1, 2, 2, 2]])
        position_ids = torch.tensor([[0, 1, 0, 1, 2]])
        labels = torch.ones(1, 5, dtype=torch.long)

        cu_seqlens = get_segment_boundaries(
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            mode="auto",
        )

        # Should use attention_mask because max > 1
        # Boundaries at: 0, 2, 5
        assert cu_seqlens[0].item() == 0
        assert cu_seqlens[-1].item() == 5

    def test_auto_mode_falls_back_to_position_ids(self):
        """Test auto mode falls back to position_ids when attention_mask is binary."""
        # Standard binary mask (max = 1)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        position_ids = torch.tensor([[0, 1, 0, 1, 2]])
        labels = torch.ones(1, 5, dtype=torch.long)

        cu_seqlens = get_segment_boundaries(
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            mode="auto",
        )

        # Should fall back to position_ids
        # Boundaries at: 0, 2, 5
        expected = [0, 2, 5]
        assert cu_seqlens.tolist() == expected

    def test_fallback_single_segment(self):
        """Test fallback when neither attention_mask nor position_ids available."""
        labels = torch.ones(2, 10, dtype=torch.long)

        cu_seqlens = get_segment_boundaries(
            attention_mask=None,
            position_ids=None,
            labels=labels,
            mode="auto",
        )

        # Fallback: each batch item is a segment
        # For labels (2, 10), after shift we have (10-1)=9 tokens per item
        # cu_seqlens = [0, 9, 18]
        assert cu_seqlens[0].item() == 0
        assert cu_seqlens[1].item() == 9
        assert cu_seqlens[2].item() == 18


class TestFlattenChannels:
    """Tests for channel list flattening."""

    def test_standard_format(self):
        """Test standard format (List[str]) passes through unchanged."""
        channels = ["math", "code", "general"]
        result = flatten_channels(channels)
        assert result == ["math", "code", "general"]

    def test_packing_format(self):
        """Test packing format (List[List[str]]) is flattened."""
        channels = [["math", "code"], ["general"], ["math", "general", "code"]]
        result = flatten_channels(channels)
        assert result == ["math", "code", "general", "math", "general", "code"]

    def test_empty_input(self):
        """Test empty input returns empty list."""
        assert flatten_channels([]) == []
        assert flatten_channels(None) == []


class TestCollatorWrapper:
    """Tests for collator wrapper functionality."""

    def test_standard_batch_channel_extraction(self):
        """Test channel extraction from standard batch format."""

        def mock_collator(features):
            return {"input_ids": torch.tensor([f["input_ids"] for f in features])}

        wrapped = wrap_collator_for_channel_loss(mock_collator)

        features = [
            {"input_ids": [1, 2, 3], "channel": "math"},
            {"input_ids": [4, 5, 6], "channel": "code"},
        ]

        batch = wrapped(features)

        # Channel should be extracted and added to batch
        assert "channel" in batch
        assert batch["channel"] == ["math", "code"]

        # Original features should have channel removed
        assert "channel" not in features[0]
        assert "channel" not in features[1]

    def test_packing_batch_channel_extraction(self):
        """Test channel extraction from packing batch format."""

        def mock_collator(features):
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]])}

        wrapped = wrap_collator_for_channel_loss(mock_collator)

        # Packing format: List[List[dict]]
        features = [
            [
                {"input_ids": [1, 2, 3], "channel": "math"},
                {"input_ids": [4, 5, 6], "channel": "code"},
            ]
        ]

        batch = wrapped(features)

        # Channel should be nested list for packing mode
        assert "channel" in batch
        assert batch["channel"] == [["math", "code"]]

    def test_default_channel_when_missing(self):
        """Test default channel is used when field is missing."""

        def mock_collator(features):
            return {"input_ids": torch.tensor([f["input_ids"] for f in features])}

        wrapped = wrap_collator_for_channel_loss(mock_collator, warn_on_missing=False)

        features = [
            {"input_ids": [1, 2, 3]},  # No channel field
            {"input_ids": [4, 5, 6], "channel": "code"},
        ]

        batch = wrapped(features)

        # Should have channel in batch (because "code" is not default)
        assert "channel" in batch
        assert batch["channel"] == ["default", "code"]

    def test_all_default_channels_not_added(self):
        """Test that batch doesn't include channel key when all are default."""

        def mock_collator(features):
            return {"input_ids": torch.tensor([f["input_ids"] for f in features])}

        wrapped = wrap_collator_for_channel_loss(mock_collator, warn_on_missing=False)

        features = [
            {"input_ids": [1, 2, 3]},  # No channel
            {"input_ids": [4, 5, 6]},  # No channel
        ]

        batch = wrapped(features)

        # When all channels are "default", don't add to batch
        assert "channel" not in batch

    def test_custom_channel_field_name(self):
        """Test custom channel field name."""

        def mock_collator(features):
            return {"input_ids": torch.tensor([f["input_ids"] for f in features])}

        wrapped = wrap_collator_for_channel_loss(
            mock_collator, channel_field="data_source"
        )

        features = [
            {"input_ids": [1, 2, 3], "data_source": "arxiv"},
            {"input_ids": [4, 5, 6], "data_source": "github"},
        ]

        batch = wrapped(features)

        assert "channel" in batch
        assert batch["channel"] == ["arxiv", "github"]


class TestChannelLossPlugin:
    """Integration tests for the full plugin."""

    def test_plugin_import(self):
        """Test plugin can be imported."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()
        assert plugin is not None

    def test_plugin_get_input_args(self):
        """Test plugin returns correct input args class path."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()
        args_path = plugin.get_input_args()
        assert args_path == "axolotl.integrations.channel_loss.args.ChannelLossArgs"

    def test_plugin_get_training_args_mixin(self):
        """Test plugin returns correct training args mixin path."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()
        mixin_path = plugin.get_training_args_mixin()
        assert (
            mixin_path
            == "axolotl.integrations.channel_loss.args.ChannelLossTrainingArgsMixin"
        )

    def test_plugin_register_extracts_channels(self):
        """Test register() extracts channel from dataset configs."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        cfg = {
            "enable_channel_loss": True,
            "datasets": [
                {"path": "/data/math.jsonl", "channel": "math"},
                {"path": "/data/code.jsonl", "channel": "code"},
                {"path": "/data/general.jsonl"},  # No channel
            ],
        }

        plugin.register(cfg)

        # Channels should be extracted and stored
        assert "_channel_loss_dataset_channels" in cfg
        assert cfg["_channel_loss_dataset_channels"] == ["math", "code", None]

        # Channel should be removed from dataset configs
        assert "channel" not in cfg["datasets"][0]
        assert "channel" not in cfg["datasets"][1]

    def test_plugin_disabled_when_not_enabled(self):
        """Test plugin does nothing when enable_channel_loss is False."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        cfg = {
            "enable_channel_loss": False,
            "datasets": [
                {"path": "/data/math.jsonl", "channel": "math"},
            ],
        }

        plugin.register(cfg)

        # Should not extract channels
        assert "_channel_loss_dataset_channels" not in cfg
        # Original config should be unchanged
        assert cfg["datasets"][0].get("channel") == "math"


class TestArgsModels:
    """Tests for argument models."""

    def test_channel_loss_args_defaults(self):
        """Test ChannelLossArgs has correct defaults."""
        from axolotl.integrations.channel_loss.args import ChannelLossArgs

        args = ChannelLossArgs()

        assert args.enable_channel_loss is None
        assert args.channel_loss_field == "channel"
        assert args.channel_loss_prefix == "loss_"
        assert args.channel_loss_segment == "auto"
        assert args.channel_loss_warn_on_missing is True

    def test_channel_loss_training_args_mixin(self):
        """Test ChannelLossTrainingArgsMixin has correct defaults."""
        from axolotl.integrations.channel_loss.args import ChannelLossTrainingArgsMixin

        mixin = ChannelLossTrainingArgsMixin()

        assert mixin.enable_channel_loss is None
        assert mixin.channel_loss_field == "channel"
        assert mixin.channel_loss_prefix == "loss_"
        assert mixin.channel_loss_segment == "auto"


class TestConflictDetection:
    """Tests for compatibility conflict detection added in Phase 3."""

    def test_liger_flce_conflict_raises_error(self):
        """Test that Liger FLCE raises ValueError when combined with Channel Loss."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        cfg = {
            "enable_channel_loss": True,
            "liger_fused_linear_cross_entropy": True,
            "plugins": ["axolotl.integrations.channel_loss.ChannelLossPlugin"],
        }

        with pytest.raises(
            ValueError, match="incompatible with liger_fused_linear_cross_entropy"
        ):
            plugin.register(cfg)

    def test_kd_trainer_conflict_raises_error(self):
        """Test that KD Trainer raises ValueError when combined with Channel Loss."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        cfg = {
            "enable_channel_loss": True,
            "kd_trainer": True,
            "plugins": ["axolotl.integrations.channel_loss.ChannelLossPlugin"],
        }

        with pytest.raises(ValueError, match="incompatible with KD trainer"):
            plugin.register(cfg)

    def test_liger_flce_error_message_includes_solutions(self):
        """Test that Liger FLCE error provides solution alternatives."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        cfg = {
            "enable_channel_loss": True,
            "liger_fused_linear_cross_entropy": True,
        }

        with pytest.raises(ValueError) as exc_info:
            plugin.register(cfg)

        error_msg = str(exc_info.value)
        # Check that error message includes helpful solutions
        assert "chunked_cross_entropy" in error_msg
        assert "liger_cross_entropy" in error_msg
        assert "skip_logits" in error_msg

    def test_kd_error_message_includes_solutions(self):
        """Test that KD Trainer error provides solution alternatives."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        cfg = {
            "enable_channel_loss": True,
            "kd_trainer": True,
        }

        with pytest.raises(ValueError) as exc_info:
            plugin.register(cfg)

        error_msg = str(exc_info.value)
        # Check that error message explains the problem
        assert "return_outputs" in error_msg
        assert "compute_loss" in error_msg

    def test_rl_training_warning(self, caplog):
        """Test that RL training triggers a warning but doesn't fail."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        # Test each RL type
        rl_types = ["dpo", "kto", "orpo", "simpo", "grpo"]

        for rl_type in rl_types:
            caplog.clear()

            cfg = {
                "enable_channel_loss": True,
                "rl": rl_type,
                "datasets": [],
            }

            # Should not raise, only warn
            plugin.register(cfg)

            # Check warning was logged
            assert any(
                "sample-level preference loss" in rec.message for rec in caplog.records
            )
            assert any(rl_type.upper() in rec.message for rec in caplog.records)

    def test_cce_auto_disable_behavior(self, caplog):
        """Test that CCE is auto-disabled with appropriate logging."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        cfg = {
            "enable_channel_loss": True,
            "plugins": [
                "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin",
                "axolotl.integrations.channel_loss.ChannelLossPlugin",
            ],
            "datasets": [],
        }

        plugin.register(cfg)

        # CCE should be disabled
        assert cfg.get("cut_cross_entropy") is False

        # Check warning was logged
        assert any(
            "Disabling Cut Cross Entropy" in rec.message for rec in caplog.records
        )

    def test_no_conflicts_when_disabled(self):
        """Test that conflicts are not checked when Channel Loss is disabled."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        cfg = {
            "enable_channel_loss": False,  # Disabled
            "liger_fused_linear_cross_entropy": True,  # Would conflict if enabled
            "kd_trainer": True,  # Would conflict if enabled
        }

        # Should not raise because Channel Loss is disabled
        plugin.register(cfg)

    def test_multiple_conflicts_first_one_raises(self):
        """Test that when multiple conflicts exist, the first one raises."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        cfg = {
            "enable_channel_loss": True,
            "liger_fused_linear_cross_entropy": True,  # First check
            "kd_trainer": True,  # Second check
        }

        # Should raise for Liger FLCE (first check)
        with pytest.raises(ValueError, match="liger_fused_linear_cross_entropy"):
            plugin.register(cfg)

    def test_compatible_configurations_pass(self):
        """Test that compatible configurations don't raise errors."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        # Test chunked_cross_entropy (compatible)
        cfg = {
            "enable_channel_loss": True,
            "chunked_cross_entropy": True,
            "datasets": [],
        }
        plugin.register(cfg)  # Should not raise

        # Test sample_packing (compatible)
        cfg = {
            "enable_channel_loss": True,
            "sample_packing": True,
            "datasets": [],
        }
        plugin.register(cfg)  # Should not raise

        # Test deepspeed (compatible)
        cfg = {
            "enable_channel_loss": True,
            "deepspeed": "deepspeed_configs/zero3.json",
            "datasets": [],
        }
        plugin.register(cfg)  # Should not raise


class TestCompatibleFeatures:
    """Integration tests for features that should work with Channel Loss."""

    def test_chunked_cross_entropy_integration(self):
        """Test that Channel Loss works with Chunked Cross Entropy."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()
        cfg = {
            "enable_channel_loss": True,
            "chunked_cross_entropy": True,
            "chunk_size": 8192,
            "datasets": [],
        }

        # Should register without errors
        plugin.register(cfg)

        # Verify chunked_cross_entropy is preserved
        assert cfg["chunked_cross_entropy"] is True
        assert cfg["chunk_size"] == 8192

    def test_sample_packing_integration(self):
        """Test that Channel Loss works with sample packing."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()
        cfg = {
            "enable_channel_loss": True,
            "sample_packing": True,
            "channel_loss_segment": "auto",
            "datasets": [{"path": "test.jsonl", "channel": "test"}],
        }

        # Should register without errors
        plugin.register(cfg)

        # Verify channel was extracted
        assert "_channel_loss_dataset_channels" in cfg
        assert cfg["_channel_loss_dataset_channels"] == ["test"]

    def test_lora_qlora_integration(self):
        """Test that Channel Loss works with LoRA/QLoRA."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()
        cfg = {
            "enable_channel_loss": True,
            "adapter": "qlora",
            "lora_r": 32,
            "lora_alpha": 16,
            "load_in_4bit": True,
            "datasets": [],
        }

        # Should register without errors
        plugin.register(cfg)

    def test_distributed_training_integration(self):
        """Test that Channel Loss registers with distributed training configs."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        # FSDP
        cfg_fsdp = {
            "enable_channel_loss": True,
            "fsdp": ["full_shard", "auto_wrap"],
            "fsdp_config": {"fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer"},
            "datasets": [],
        }
        plugin.register(cfg_fsdp)  # Should not raise

        # DeepSpeed ZeRO-2
        plugin2 = ChannelLossPlugin()
        cfg_ds = {
            "enable_channel_loss": True,
            "deepspeed": "deepspeed_configs/zero2.json",
            "datasets": [],
        }
        plugin2.register(cfg_ds)  # Should not raise

    def test_gradient_checkpointing_integration(self):
        """Test that Channel Loss works with gradient checkpointing."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()
        cfg = {
            "enable_channel_loss": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "datasets": [],
        }

        # Should register without errors
        plugin.register(cfg)

    def test_flash_attention_integration(self):
        """Test that Channel Loss works with Flash Attention."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()
        cfg = {
            "enable_channel_loss": True,
            "flash_attention": True,
            "datasets": [],
        }

        # Should register without errors
        plugin.register(cfg)

    def test_liger_non_fused_integration(self):
        """Test that Channel Loss works with non-fused Liger CE."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()
        cfg = {
            "enable_channel_loss": True,
            "liger_cross_entropy": True,  # Non-fused version
            "datasets": [],
        }

        # Should register without errors
        plugin.register(cfg)

        # Verify liger_cross_entropy is preserved
        assert cfg["liger_cross_entropy"] is True


class TestRuntimeDetection:
    """Tests for runtime detection of missing logits."""

    def test_runtime_logits_missing_detection(self):
        """Test that missing logits at runtime triggers warning."""
        from unittest.mock import Mock

        import torch

        from axolotl.integrations.channel_loss.compute_loss_patch import (
            patch_compute_loss_for_channel_loss,
        )

        # Create mock trainer
        trainer = Mock()
        trainer.model = Mock()
        trainer.model.training = True
        trainer._channel_loss_warned_no_logits = False
        trainer._channel_loss_stats = {
            "train": {},
            "eval": {},
        }

        # Mock original compute_loss that returns None outputs
        def mock_compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss = torch.tensor(1.0)
            if return_outputs:
                # Return outputs without logits (simulating incompatible optimization)
                outputs = Mock()
                outputs.logits = None
                return loss, outputs
            return loss

        trainer.compute_loss = mock_compute_loss

        # Apply the patch
        cfg = {
            "channel_loss_segment": "auto",
            "channel_loss_prefix": "loss_",
            "channel_loss_field": "channel",
        }
        patch_compute_loss_for_channel_loss(trainer, cfg)

        # Test inputs with channel
        inputs = {
            "labels": torch.tensor([[1, 2, 3]]),
            "channel": ["test_channel"],
        }

        # Call patched compute_loss
        result = trainer.compute_loss(trainer.model, inputs, return_outputs=False)

        # Verify warning flag was set
        assert trainer._channel_loss_warned_no_logits is True
        assert isinstance(result, torch.Tensor)

    def test_runtime_detection_only_warns_once(self):
        """Test that runtime detection only warns once per trainer."""
        from unittest.mock import Mock

        import torch

        from axolotl.integrations.channel_loss.compute_loss_patch import (
            patch_compute_loss_for_channel_loss,
        )

        trainer = Mock()
        trainer.model = Mock()
        trainer.model.training = True
        trainer._channel_loss_warned_no_logits = False
        trainer._channel_loss_stats = {"train": {}, "eval": {}}

        def mock_compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss = torch.tensor(1.0)
            if return_outputs:
                outputs = Mock()
                outputs.logits = None
                return loss, outputs
            return loss

        trainer.compute_loss = mock_compute_loss

        cfg = {
            "channel_loss_segment": "auto",
            "channel_loss_prefix": "loss_",
            "channel_loss_field": "channel",
        }
        patch_compute_loss_for_channel_loss(trainer, cfg)

        inputs = {
            "labels": torch.tensor([[1, 2, 3]]),
            "channel": ["test_channel"],
        }

        # First call - should warn
        trainer.compute_loss(trainer.model, inputs, return_outputs=False)
        assert trainer._channel_loss_warned_no_logits is True

        # Second call - warning flag already set, should not warn again
        # (The actual logging happens inside the function, we just verify the flag)
        trainer.compute_loss(trainer.model, inputs, return_outputs=False)
        assert trainer._channel_loss_warned_no_logits is True

    def test_happy_path_with_logits_available(self):
        """Test normal operation when logits are available."""
        from collections import defaultdict
        from unittest.mock import Mock

        import torch

        from axolotl.integrations.channel_loss.compute_loss_patch import (
            patch_compute_loss_for_channel_loss,
        )

        trainer = Mock()
        trainer.model = Mock()
        trainer.model.training = True
        trainer._channel_loss_warned_no_logits = False
        trainer._channel_loss_stats = {
            "train": defaultdict(lambda: {"sum": 0.0, "count": 0}),
            "eval": defaultdict(lambda: {"sum": 0.0, "count": 0}),
        }

        # Mock compute_loss that returns proper logits
        def mock_compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss = torch.tensor(2.5)
            if return_outputs:
                outputs = Mock()
                # Create proper logits tensor
                batch_size = inputs["labels"].shape[0]
                seq_len = inputs["labels"].shape[1]
                vocab_size = 32000
                outputs.logits = torch.randn(batch_size, seq_len, vocab_size)
                return loss, outputs
            return loss

        trainer.compute_loss = mock_compute_loss

        cfg = {
            "channel_loss_segment": "auto",
            "channel_loss_prefix": "loss_",
            "channel_loss_field": "channel",
        }
        patch_compute_loss_for_channel_loss(trainer, cfg)

        # Test with proper inputs
        inputs = {
            "labels": torch.tensor([[1, 2, 3, -100]]),
            "channel": ["math"],
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        }

        # Call should succeed
        result = trainer.compute_loss(trainer.model, inputs, return_outputs=False)

        # Should not have warned
        assert trainer._channel_loss_warned_no_logits is False
        assert isinstance(result, torch.Tensor)

        # Channel stats should have been updated
        assert "loss_math" in trainer._channel_loss_stats["train"]
        assert trainer._channel_loss_stats["train"]["loss_math"]["count"] > 0
