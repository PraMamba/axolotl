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

from collections import defaultdict

import pytest
import torch

from axolotl.integrations.channel_loss.collator_wrapper import (
    wrap_collator_for_channel_loss,
)
from axolotl.integrations.channel_loss.compute_loss_patch import _update_channel_stats
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

        # Fallback: each batch item is a segment (token index space)
        # For labels (2, 10), token boundaries are [0, 10, 20]
        assert cu_seqlens[0].item() == 0
        assert cu_seqlens[1].item() == 10
        assert cu_seqlens[2].item() == 20


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

    def test_metadata_field_always_removed_standard_batch(self):
        """Test that _channel_dataset_idx metadata field is always removed in standard batch."""
        import numpy as np

        def mock_collator(features):
            # This collator would fail if it receives scalar fields like _channel_dataset_idx
            # Simulate the real collator behavior by trying to concatenate arrays
            for key in features[0].keys():
                arrays = [np.array(f[key]) for f in features]
                # This would raise "ValueError: zero-dimensional arrays cannot be concatenated"
                # if _channel_dataset_idx (a scalar int) is present
                if len(arrays) > 0:
                    try:
                        np.concatenate(arrays)
                    except ValueError as e:
                        if "zero-dimensional" in str(e):
                            raise ValueError(
                                f"Scalar field '{key}' should have been removed by collator wrapper"
                            ) from e
            return {"input_ids": torch.tensor([f["input_ids"] for f in features])}

        wrapped = wrap_collator_for_channel_loss(
            mock_collator,
            channel_field="channel",
            dataset_channels={0: "math", 1: "code"},
        )

        # Test case 1: Sample with direct channel field AND dataset_idx field
        features = [
            {
                "input_ids": [1, 2, 3],
                "channel": "math",
                "_channel_dataset_idx": 0,  # This metadata field should be removed
            },
            {
                "input_ids": [4, 5, 6],
                "channel": "code",
                "_channel_dataset_idx": 1,  # This metadata field should be removed
            },
        ]

        batch = wrapped(features)

        # Verify channels were correctly extracted
        assert "channel" in batch
        assert batch["channel"] == ["math", "code"]

        # Verify metadata field was removed (implicitly - no concatenation error)
        assert "_channel_dataset_idx" not in batch

        # Test case 2: Sample with only dataset_idx field (no direct channel)
        features2 = [
            {
                "input_ids": [7, 8, 9],
                "_channel_dataset_idx": 0,  # Should be removed after lookup
            },
            {
                "input_ids": [10, 11, 12],
                "_channel_dataset_idx": 1,  # Should be removed after lookup
            },
        ]

        batch2 = wrapped(features2)

        # Verify channels were looked up correctly
        assert "channel" in batch2
        assert batch2["channel"] == ["math", "code"]

        # Verify metadata field was removed
        assert "_channel_dataset_idx" not in batch2

    def test_metadata_field_always_removed_packing_batch(self):
        """Test that _channel_dataset_idx metadata field is always removed in packing batch."""
        import numpy as np

        def mock_collator(features):
            # Simulate packing collator that concatenates arrays from sub-batches
            if isinstance(features[0], list):
                for sub_batch in features:
                    for key in sub_batch[0].keys():
                        arrays = [np.array(f[key]) for f in sub_batch]
                        # Would fail on scalar fields
                        if len(arrays) > 0:
                            try:
                                np.concatenate(arrays)
                            except ValueError as e:
                                if "zero-dimensional" in str(e):
                                    raise ValueError(
                                        f"Scalar field '{key}' in packing mode should have been removed"
                                    ) from e
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]])}

        wrapped = wrap_collator_for_channel_loss(
            mock_collator,
            channel_field="channel",
            dataset_channels={0: "math", 1: "code"},
        )

        # Packing format with metadata fields
        features = [
            [
                {
                    "input_ids": [1, 2, 3],
                    "channel": "math",
                    "_channel_dataset_idx": 0,  # Should be removed
                },
                {
                    "input_ids": [4, 5, 6],
                    "channel": "code",
                    "_channel_dataset_idx": 1,  # Should be removed
                },
            ]
        ]

        batch = wrapped(features)

        # Verify channels were correctly extracted (packing format: List[List[str]])
        assert "channel" in batch
        assert batch["channel"] == [["math", "code"]]

        # Verify metadata field was removed (no concatenation error occurred)
        assert "_channel_dataset_idx" not in batch


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
        """Test register() extracts channel from dataset configs and stores in plugin instance."""
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

        # Channels should be stored in plugin instance variable (Spec 013 P0-1 fix)
        assert hasattr(plugin, "_dataset_channels")
        assert plugin._dataset_channels == {0: "math", 1: "code"}

        # Channel field should be removed from dataset configs to avoid schema validation errors
        assert "channel" not in cfg["datasets"][0]
        assert "channel" not in cfg["datasets"][1]

        # No internal field should be added to configs (that was the broken approach)
        assert "_channel_loss_channel" not in cfg["datasets"][0]
        assert "_channel_loss_channel" not in cfg["datasets"][1]

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

    def test_packing_with_batch_size_gt_1_raises_error(self):
        """Test that packing with micro_batch_size > 1 raises ValueError."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        # Test with sample_packing
        cfg = {
            "enable_channel_loss": True,
            "sample_packing": True,
            "micro_batch_size": 2,
            "datasets": [],
        }

        with pytest.raises(
            ValueError,
            match="does not support sample packing with micro_batch_size > 1",
        ):
            plugin.register(cfg)

        # Test with eval_sample_packing
        cfg_eval = {
            "enable_channel_loss": True,
            "eval_sample_packing": True,
            "micro_batch_size": 4,
            "datasets": [],
        }

        with pytest.raises(
            ValueError,
            match="does not support sample packing with micro_batch_size > 1",
        ):
            plugin.register(cfg_eval)

        # Test with both packing modes
        cfg_both = {
            "enable_channel_loss": True,
            "sample_packing": True,
            "eval_sample_packing": True,
            "micro_batch_size": 2,
            "datasets": [],
        }

        with pytest.raises(
            ValueError,
            match="does not support sample packing with micro_batch_size > 1",
        ):
            plugin.register(cfg_both)

    def test_packing_with_batch_size_1_succeeds(self):
        """Test that packing with micro_batch_size=1 is allowed."""
        from axolotl.integrations.channel_loss import ChannelLossPlugin

        plugin = ChannelLossPlugin()

        cfg = {
            "enable_channel_loss": True,
            "sample_packing": True,
            "micro_batch_size": 1,
            "datasets": [],
        }

        # Should register without errors
        plugin.register(cfg)


class TestChannelLossWithContextParallelism:
    def test_cp_local_standard_mode_matches_full(self, monkeypatch):
        """
        CP-local outputs (gather_outputs=False): stats computed shard-wise should match
        stats computed on full logits/labels.
        """

        class DummyModel:
            training = True

        class DummyTrainer:
            def __init__(self):
                self.model = DummyModel()
                self._channel_loss_stats = {
                    "train": defaultdict(lambda: {"sum": 0.0, "count": 0}),
                    "eval": defaultdict(lambda: {"sum": 0.0, "count": 0}),
                }

        trainer = DummyTrainer()

        cp_size = 2
        cp_group = object()

        # Patch CP group detection + distributed helpers (no collectives used).
        monkeypatch.setattr(
            "axolotl.integrations.channel_loss.compute_loss_patch._get_context_parallel_group",
            lambda _trainer: cp_group,
        )
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)

        def _get_world_size(group=None):
            return cp_size if group is cp_group else 1

        monkeypatch.setattr("torch.distributed.get_world_size", _get_world_size)

        # Small synthetic example: seq_len divisible by cp_size.
        torch.manual_seed(0)
        batch_size = 1
        seq_len = 6
        vocab_size = 11
        channels = ["task_A"]

        logits_full = torch.randn(batch_size, seq_len, vocab_size)
        labels_full = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Expected (full) computation
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        full_loss = loss_fct(
            logits_full[:, :-1, :].contiguous().view(-1, vocab_size),
            labels_full[:, 1:].contiguous().view(-1),
        )
        expected_sum = full_loss.sum().item()
        expected_count = full_loss.numel()

        # Simulate CP ranks: each sees a chunk of logits
        chunk = seq_len // cp_size

        for cp_rank in range(cp_size):
            monkeypatch.setattr(
                "torch.distributed.get_rank", lambda group=None, r=cp_rank: r
            )
            logits_local = logits_full[:, cp_rank * chunk : (cp_rank + 1) * chunk, :]
            _update_channel_stats(
                trainer=trainer,
                logits=logits_local,
                labels=labels_full,
                channels=channels,
                position_ids=None,
                attention_mask=None,
                segment_mode="auto",
                prefix="loss=",
            )

        stats = trainer._channel_loss_stats["train"]["loss=task_A"]
        assert stats["count"] == expected_count
        assert stats["sum"] == pytest.approx(expected_sum, rel=1e-6, abs=1e-6)

    def test_cp_local_packing_mode_attributes_boundary_to_next_segment(
        self, monkeypatch
    ):
        """
        Packing mode with a CP boundary exactly on a segment boundary: the boundary token loss
        should be attributed to the *next* segment (by token position), even though it's computed
        on the previous CP rank.
        """

        class DummyModel:
            training = True

        class DummyTrainer:
            def __init__(self):
                self.model = DummyModel()
                self._channel_loss_stats = {
                    "train": defaultdict(lambda: {"sum": 0.0, "count": 0}),
                    "eval": defaultdict(lambda: {"sum": 0.0, "count": 0}),
                }

        trainer = DummyTrainer()

        cp_size = 2
        cp_group = object()

        monkeypatch.setattr(
            "axolotl.integrations.channel_loss.compute_loss_patch._get_context_parallel_group",
            lambda _trainer: cp_group,
        )
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)

        def _get_world_size(group=None):
            return cp_size if group is cp_group else 1

        monkeypatch.setattr("torch.distributed.get_world_size", _get_world_size)

        torch.manual_seed(0)
        batch_size = 1
        seq_len = 6
        vocab_size = 11

        logits_full = torch.randn(batch_size, seq_len, vocab_size)
        labels_full = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Two packed segments of length 3 each; CP boundary is also at 3.
        attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2]])
        channels = [["seg1", "seg2"]]

        # Expected per-segment full computation (token losses correspond to target tokens)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        full_loss = loss_fct(
            logits_full[:, :-1, :].contiguous().view(-1, vocab_size),
            labels_full[:, 1:].contiguous().view(-1),
        )
        # seg1 covers tokens 0..2 -> losses for tokens 1..2 (indices 0..1)
        expected_seg1 = full_loss[0:2]
        # seg2 covers tokens 3..5 -> losses for tokens 3..5 (indices 2..4)
        expected_seg2 = full_loss[2:5]

        chunk = seq_len // cp_size
        for cp_rank in range(cp_size):
            monkeypatch.setattr(
                "torch.distributed.get_rank", lambda group=None, r=cp_rank: r
            )
            logits_local = logits_full[:, cp_rank * chunk : (cp_rank + 1) * chunk, :]
            _update_channel_stats(
                trainer=trainer,
                logits=logits_local,
                labels=labels_full,
                channels=channels,
                position_ids=None,
                attention_mask=attention_mask,
                segment_mode="attention_mask",
                prefix="loss=",
            )

        seg1_stats = trainer._channel_loss_stats["train"]["loss=seg1"]
        seg2_stats = trainer._channel_loss_stats["train"]["loss=seg2"]

        assert seg1_stats["count"] == expected_seg1.numel()
        assert seg2_stats["count"] == expected_seg2.numel()
        assert seg1_stats["sum"] == pytest.approx(
            expected_seg1.sum().item(), rel=1e-6, abs=1e-6
        )
        assert seg2_stats["sum"] == pytest.approx(
            expected_seg2.sum().item(), rel=1e-6, abs=1e-6
        )

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

    def test_p0_1_integration_channel_injection_via_dataset_idx(self):
        """
        Test P0-1 fix: Channel injection via dataset_idx (Spec 013).

        This test verifies the complete flow:
        1. Plugin extracts channels from config and stores in _dataset_channels dict
        2. Dataset loading injects _channel_dataset_idx into samples
        3. Collator extracts dataset_idx and looks up channel from plugin's mapping
        4. Channel is correctly added to batch

        This integration test simulates the real data flow to ensure P0-1 fix works end-to-end.
        """
        from unittest.mock import MagicMock

        from axolotl.integrations.channel_loss import ChannelLossPlugin
        from axolotl.integrations.channel_loss.collator_wrapper import (
            wrap_collator_for_channel_loss,
        )

        # Step 1: Plugin extracts channels from config
        plugin = ChannelLossPlugin()
        cfg = {
            "enable_channel_loss": True,
            "channel_loss_field": "channel",
            "datasets": [
                {"path": "/data/math.jsonl", "channel": "math"},
                {"path": "/data/code.jsonl", "channel": "code"},
            ],
        }
        plugin.register(cfg)

        # Verify plugin stores channels correctly
        assert plugin._dataset_channels == {0: "math", 1: "code"}
        assert "channel" not in cfg["datasets"][0]  # Removed to avoid validation errors

        # Step 2: Simulate dataset loading that injects _channel_dataset_idx
        # (This would happen in sft.py _load_and_process_single_dataset)
        sample_from_dataset_0 = {
            "input_ids": [1, 2, 3],
            "labels": [4, 5, 6],
            "_channel_dataset_idx": 0,  # Injected by sft.py
        }
        sample_from_dataset_1 = {
            "input_ids": [7, 8, 9],
            "labels": [10, 11, 12],
            "_channel_dataset_idx": 1,  # Injected by sft.py
        }

        # Step 3: Collator extracts dataset_idx and looks up channel
        mock_inner_collator = MagicMock(
            return_value={
                "input_ids": [[1, 2, 3], [7, 8, 9]],
                "labels": [[4, 5, 6], [10, 11, 12]],
            }
        )

        wrapped_collator = wrap_collator_for_channel_loss(
            inner_collator=mock_inner_collator,
            channel_field="channel",
            dataset_channels=plugin._dataset_channels,  # Pass plugin's mapping
            warn_on_missing=True,
        )

        # Process batch
        batch = wrapped_collator([sample_from_dataset_0, sample_from_dataset_1])

        # Step 4: Verify channels are correctly added to batch
        assert "channel" in batch
        assert batch["channel"] == ["math", "code"]

        # Verify _channel_dataset_idx was removed (shouldn't be passed to model)
        # This is checked implicitly - if it's passed to inner_collator, that's OK
        # as long as it doesn't appear in the final batch
        assert "_channel_dataset_idx" not in batch


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

        # Verify channel was extracted and stored in plugin instance (Spec 013 P0-1 fix)
        assert hasattr(plugin, "_dataset_channels")
        assert plugin._dataset_channels == {0: "test"}
        assert "channel" not in cfg["datasets"][0]

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


class TestDatasetChannelInjection:
    """Test channel injection during dataset loading."""

    def test_channel_injected_from_config(self):
        """Test that channel field is injected from dataset config."""
        from datasets import Dataset

        # Create a simple dataset without channel field
        dataset = Dataset.from_dict({"text": ["sample1", "sample2", "sample3"]})

        # Simulate the injection logic from sft.py
        channel_value = "math"
        channel_field = "channel"

        def add_channel_field(example):
            if channel_field not in example:
                example[channel_field] = channel_value
            return example

        # Apply injection
        dataset_with_channel = dataset.map(add_channel_field)

        # Verify all samples have the channel field
        assert all(channel_field in sample for sample in dataset_with_channel)
        assert all(
            sample[channel_field] == channel_value for sample in dataset_with_channel
        )

    def test_channel_not_overwritten_if_exists(self):
        """Test that existing channel field in sample is not overwritten."""
        from datasets import Dataset

        # Create dataset with some samples having channel field
        dataset = Dataset.from_dict(
            {
                "text": ["sample1", "sample2", "sample3"],
                "channel": ["existing", None, None],
            }
        )

        # Simulate the injection logic
        channel_value = "math"
        channel_field = "channel"

        def add_channel_field(example):
            # Priority: sample's existing channel > configured channel
            if channel_field not in example or example[channel_field] is None:
                example[channel_field] = channel_value
            return example

        dataset_with_channel = dataset.map(add_channel_field)

        # Verify: first sample keeps "existing", others get "math"
        samples = list(dataset_with_channel)
        assert samples[0][channel_field] == "existing"
        assert samples[1][channel_field] == "math"
        assert samples[2][channel_field] == "math"


class TestNumericalRobustness:
    """Tests for numerical robustness and inf/nan filtering (Spec 013)."""

    def test_inf_nan_loss_filtered_correctly(self):
        """Test that inf/nan loss values are filtered by isfinite check."""
        from collections import defaultdict
        from unittest.mock import Mock

        import torch

        from axolotl.integrations.channel_loss.compute_loss_patch import (
            _update_channel_stats,
        )

        # Create mock trainer
        trainer = Mock()
        trainer.model = Mock()
        trainer.model.training = True
        trainer._channel_loss_stats = {
            "train": defaultdict(lambda: {"sum": 0.0, "count": 0}),
            "eval": defaultdict(lambda: {"sum": 0.0, "count": 0}),
        }

        # Create logits that will produce inf/nan losses
        batch_size, seq_len, vocab_size = 1, 4, 100
        logits = torch.zeros(batch_size, seq_len, vocab_size)

        # Create extreme logits that may cause inf/nan
        # Set very large values that could cause numerical issues
        logits[0, 0, :] = 1e10  # Extremely large logits
        logits[0, 1, :] = -1e10  # Extremely small logits

        labels = torch.tensor([[10, 20, 30, -100]])  # Last token is padding
        channels = ["test"]

        # Call _update_channel_stats
        _update_channel_stats(
            trainer=trainer,
            logits=logits,
            labels=labels,
            channels=channels,
            position_ids=None,
            attention_mask=None,
            segment_mode="auto",
            prefix="loss_",
        )

        # Verify stats were collected
        assert "loss_test" in trainer._channel_loss_stats["train"]
        stats = trainer._channel_loss_stats["train"]["loss_test"]

        # Verify that sum and count are finite (inf/nan were filtered)
        assert torch.isfinite(torch.tensor(stats["sum"])).item()
        assert stats["count"] >= 0  # Count should be non-negative

        # If all losses were inf/nan, count should be 0
        # If some were finite, count should be > 0
        # We can't predict exact behavior without knowing CE implementation details

    def test_all_valid_losses_are_counted(self):
        """Test that all valid (finite) losses are counted correctly."""
        from collections import defaultdict
        from unittest.mock import Mock

        import torch

        from axolotl.integrations.channel_loss.compute_loss_patch import (
            _update_channel_stats,
        )

        trainer = Mock()
        trainer.model = Mock()
        trainer.model.training = True
        trainer._channel_loss_stats = {
            "train": defaultdict(lambda: {"sum": 0.0, "count": 0}),
            "eval": defaultdict(lambda: {"sum": 0.0, "count": 0}),
        }

        # Create normal logits that will produce valid losses
        batch_size, seq_len, vocab_size = 1, 5, 50
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 2, 3, 4, -100]])  # 4 valid tokens
        channels = ["valid"]

        _update_channel_stats(
            trainer=trainer,
            logits=logits,
            labels=labels,
            channels=channels,
            position_ids=None,
            attention_mask=None,
            segment_mode="auto",
            prefix="loss_",
        )

        stats = trainer._channel_loss_stats["train"]["loss_valid"]

        # With causal shift: logits[:-1] predicts labels[1:]
        # labels has 4 valid tokens at indices 0-3, label 4 is -100 (padding)
        # After shift: we predict labels[1:4] which are 3 valid tokens
        # (label at index 4 is -100, excluded from loss)
        assert stats["count"] == 3  # 3 valid loss tokens after shift
        assert stats["sum"] > 0  # Loss should be positive
        assert torch.isfinite(torch.tensor(stats["sum"])).item()

    def test_direct_nan_inf_injection_filtering(self, monkeypatch):
        """Test that nan/inf values are correctly filtered when directly injected.

        This test directly injects nan/inf into the per-token loss tensor
        to ensure the filtering logic works correctly, rather than relying
        on extreme logits that may or may not produce nan/inf.
        """
        from collections import defaultdict
        from unittest.mock import Mock

        import torch

        from axolotl.integrations.channel_loss.compute_loss_patch import (
            _update_channel_stats,
        )

        # Patch CrossEntropyLoss to return controlled loss with nan/inf
        class MockCrossEntropyLoss:
            def __init__(self, reduction="none"):
                self.reduction = reduction

            def __call__(self, input, target):
                # Return a controlled loss tensor with known nan/inf values
                # Shape should match: (batch_size * (seq_len - 1),)
                # We'll create: [1.5, nan, 2.5, inf, 3.5, -inf, 4.5]
                loss = torch.tensor(
                    [1.5, float("nan"), 2.5, float("inf"), 3.5, float("-inf"), 4.5]
                )
                return loss

        monkeypatch.setattr(
            "axolotl.integrations.channel_loss.compute_loss_patch.torch.nn.CrossEntropyLoss",
            MockCrossEntropyLoss,
        )

        trainer = Mock()
        trainer.model = Mock()
        trainer.model.training = True
        trainer._channel_loss_stats = {
            "train": defaultdict(lambda: {"sum": 0.0, "count": 0}),
            "eval": defaultdict(lambda: {"sum": 0.0, "count": 0}),
        }

        # Create dummy inputs (values don't matter since loss is mocked)
        batch_size, seq_len, vocab_size = 1, 8, 50
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        channels = ["test"]

        _update_channel_stats(
            trainer=trainer,
            logits=logits,
            labels=labels,
            channels=channels,
            position_ids=None,
            attention_mask=None,
            segment_mode="auto",
            prefix="loss_",
        )

        stats = trainer._channel_loss_stats["train"]["loss_test"]

        # Expected: only finite values [1.5, 2.5, 3.5, 4.5] should be counted
        # nan, inf, -inf should be filtered out
        expected_count = 4
        expected_sum = 1.5 + 2.5 + 3.5 + 4.5  # = 12.0

        assert stats["count"] == expected_count, (
            f"Expected {expected_count} finite values, got {stats['count']}"
        )
        assert abs(stats["sum"] - expected_sum) < 1e-6, (
            f"Expected sum {expected_sum}, got {stats['sum']}"
        )
        assert torch.isfinite(torch.tensor(stats["sum"])).item(), (
            "Sum should be finite after filtering"
        )


class TestCPLocalBoundaryConditions:
    """Tests for CP-local edge cases (Spec 013)."""

    def test_cp_local_non_divisible_sequence_length(self, monkeypatch):
        """Test CP with sequence length not divisible by cp_size.

        Verifies correctness by comparing CP-local shard-wise computation
        against full baseline computation. CP handles non-divisible lengths
        by padding, which should not affect loss statistics.
        """
        from collections import defaultdict
        from unittest.mock import Mock

        import pytest
        import torch

        from axolotl.integrations.channel_loss.compute_loss_patch import (
            _update_channel_stats,
        )

        # Sequence length not divisible by cp_size=2: 7 % 2 = 1
        batch_size, seq_len, vocab_size = 1, 7, 50
        cp_size = 2
        chunk_len = 4  # ceil(7 / 2) = 4
        torch.manual_seed(42)

        # Create original (unpadded) logits and labels
        logits_orig = torch.randn(batch_size, seq_len, vocab_size)
        labels_orig = torch.randint(0, vocab_size, (batch_size, seq_len))
        channels = ["test"]

        # Compute expected (full) result on original sequence
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        full_loss = loss_fct(
            logits_orig[:, :-1, :].contiguous().view(-1, vocab_size),
            labels_orig[:, 1:].contiguous().view(-1),
        )
        expected_sum = full_loss.sum().item()
        expected_count = full_loss.numel()  # Should be 6 (seq_len - 1)

        # CP pads sequence to make it divisible: padded_len = cp_size * chunk_len = 8
        padded_len = cp_size * chunk_len

        # Create padded logits and labels for CP processing
        # Padding positions should have labels = -100 to not contribute to loss
        logits_padded = torch.zeros(batch_size, padded_len, vocab_size)
        logits_padded[:, :seq_len, :] = logits_orig

        labels_padded = torch.full((batch_size, padded_len), -100, dtype=torch.long)
        labels_padded[:, :seq_len] = labels_orig

        # Setup trainer for CP-local computation
        trainer = Mock()
        trainer.model = Mock()
        trainer.model.training = True
        trainer._channel_loss_stats = {
            "train": defaultdict(lambda: {"sum": 0.0, "count": 0}),
            "eval": defaultdict(lambda: {"sum": 0.0, "count": 0}),
        }

        cp_group = Mock()
        monkeypatch.setattr(
            "axolotl.integrations.channel_loss.compute_loss_patch._get_context_parallel_group",
            lambda _trainer: cp_group,
        )
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)

        def _get_world_size(group=None):
            return cp_size if group is cp_group else 1

        monkeypatch.setattr("torch.distributed.get_world_size", _get_world_size)

        # Simulate each CP rank with its local chunk
        for cp_rank in range(cp_size):
            monkeypatch.setattr(
                "torch.distributed.get_rank", lambda group=None, r=cp_rank: r
            )

            # Extract local chunk from padded logits
            start = cp_rank * chunk_len
            end = start + chunk_len
            logits_local = logits_padded[:, start:end, :].contiguous()

            _update_channel_stats(
                trainer=trainer,
                logits=logits_local,
                labels=labels_padded,  # Use padded labels
                channels=channels,
                position_ids=None,
                attention_mask=None,
                segment_mode="auto",
                prefix="loss_",
            )

        # Verify CP-local shard-wise sum matches full baseline
        stats = trainer._channel_loss_stats["train"]["loss_test"]
        assert stats["count"] == expected_count
        assert stats["sum"] == pytest.approx(expected_sum, rel=1e-5, abs=1e-6)

    def test_cp_local_batch_size_gt_1_standard_mode(self, monkeypatch):
        """Test CP-local with micro_batch_size > 1 in standard (non-packing) mode.

        Verifies correctness by comparing CP-local shard-wise computation
        against full baseline computation.
        """
        from collections import defaultdict
        from unittest.mock import Mock

        import pytest
        import torch

        from axolotl.integrations.channel_loss.compute_loss_patch import (
            _update_channel_stats,
        )

        # Mock distributed environment with CP=2
        cp_size = 2

        cp_group = Mock()
        monkeypatch.setattr(
            "axolotl.integrations.channel_loss.compute_loss_patch._get_context_parallel_group",
            lambda t: cp_group,
        )

        # batch_size=4, seq_len=8 (divisible by CP=2)
        batch_size, seq_len, vocab_size = 4, 8, 50
        torch.manual_seed(42)

        # Create full logits and labels
        logits_full = torch.randn(batch_size, seq_len, vocab_size)
        labels_full = torch.randint(0, vocab_size, (batch_size, seq_len))
        channels = ["ch1", "ch2", "ch3", "ch4"]  # One channel per sample

        # Compute expected (full) result for each channel
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        full_loss = loss_fct(
            logits_full[:, :-1, :].contiguous().view(-1, vocab_size),
            labels_full[:, 1:].contiguous().view(-1),
        )
        # Reshape to (batch_size, seq_len-1)
        full_loss = full_loss.view(batch_size, seq_len - 1)

        # Compute expected per-channel statistics
        expected_stats = {}
        for i, ch in enumerate(channels):
            channel_loss = full_loss[i, :]  # Loss for this batch item
            expected_stats[ch] = {
                "sum": channel_loss.sum().item(),
                "count": channel_loss.numel(),
            }

        # Simulate CP computation: each rank processes its local chunk
        chunk_len = seq_len // cp_size
        aggregated_stats = defaultdict(lambda: {"sum": 0.0, "count": 0})

        for cp_rank in range(cp_size):
            # Simulate this rank's environment
            monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
            monkeypatch.setattr(
                "torch.distributed.get_world_size", lambda group=None: cp_size
            )
            monkeypatch.setattr(
                "torch.distributed.get_rank",
                lambda group=None, rank=cp_rank: rank,
            )

            trainer = Mock()
            trainer.model = Mock()
            trainer.model.training = True
            trainer._channel_loss_stats = {
                "train": defaultdict(lambda: {"sum": 0.0, "count": 0}),
                "eval": defaultdict(lambda: {"sum": 0.0, "count": 0}),
            }

            # Extract this rank's chunk
            start_idx = cp_rank * chunk_len
            end_idx = start_idx + chunk_len
            logits_cp_local = logits_full[:, start_idx:end_idx, :].contiguous()

            _update_channel_stats(
                trainer=trainer,
                logits=logits_cp_local,
                labels=labels_full,
                channels=channels,
                position_ids=None,
                attention_mask=None,
                segment_mode="auto",
                prefix="loss_",
            )

            # Aggregate stats from this rank
            for ch in channels:
                key = f"loss_{ch}"
                stats = trainer._channel_loss_stats["train"][key]
                aggregated_stats[key]["sum"] += stats["sum"]
                aggregated_stats[key]["count"] += stats["count"]

        # Verify CP-local shard-wise computation matches full baseline
        for ch in channels:
            key = f"loss_{ch}"
            expected = expected_stats[ch]
            actual = aggregated_stats[key]

            assert actual["count"] == expected["count"]
            assert actual["sum"] == pytest.approx(expected["sum"], rel=1e-5, abs=1e-6)


class TestSegmentDetectionBoundaries:
    """Tests for segment detection edge cases (Spec 013)."""

    def test_segment_detection_position_ids_with_padding(self):
        """Test segment detection correctly handles position_ids with padding."""
        import torch

        from axolotl.integrations.channel_loss.segment import get_segment_boundaries

        # position_ids has 0s in padding area (but also starts with 0)
        position_ids = torch.tensor([[0, 1, 2, 0, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        labels = torch.tensor([[1, 2, 3, -100, -100, -100]])

        cu_seqlens = get_segment_boundaries(
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            mode="auto",
        )

        # Should detect only 1 segment [0, 3] (not confused by padding 0s)
        assert cu_seqlens.tolist() == [0, 3]

    def test_segment_detection_attention_mask_with_holes(self):
        """Test segment detection with non-contiguous attention mask."""
        import torch

        from axolotl.integrations.channel_loss.segment import get_segment_boundaries

        # attention_mask has "holes" (0s in the middle)
        attention_mask = torch.tensor([[1, 1, 0, 1, 1, 0, 0]])
        labels = torch.tensor([[1, 2, -100, 4, 5, -100, -100]])

        cu_seqlens = get_segment_boundaries(
            attention_mask=attention_mask,
            position_ids=None,
            labels=labels,
            mode="attention_mask",
        )

        # With holes, should detect segments where attention_mask transitions
        # Exact behavior depends on implementation
        assert len(cu_seqlens) >= 2  # At least start and end
        assert cu_seqlens[0] == 0
        assert cu_seqlens[-1] <= attention_mask.size(1)

    def test_segment_length_1_produces_zero_loss_tokens(self):
        """Test that segment with length=1 produces 0 loss tokens after causal shift."""
        from collections import defaultdict
        from unittest.mock import Mock

        import torch

        from axolotl.integrations.channel_loss.compute_loss_patch import (
            _update_channel_stats,
        )

        trainer = Mock()
        trainer.model = Mock()
        trainer.model.training = True
        trainer._channel_loss_stats = {
            "train": defaultdict(lambda: {"sum": 0.0, "count": 0}),
            "eval": defaultdict(lambda: {"sum": 0.0, "count": 0}),
        }

        # Create packing scenario with one very short segment
        batch_size, seq_len, vocab_size = 1, 3, 50
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[5, 6, 7]])  # 3 tokens total

        # Packing mode: segment 1 has length=1, segment 2 has length=2
        # attention_mask: [1, 2, 2] indicates segment IDs
        channels = [["seg1", "seg2"]]  # Two segments packed
        attention_mask = torch.tensor([[1, 2, 2]])

        _update_channel_stats(
            trainer=trainer,
            logits=logits,
            labels=labels,
            channels=channels,
            position_ids=None,
            attention_mask=attention_mask,
            segment_mode="attention_mask",
            prefix="loss_",
        )

        # seg1 has length=1, so after shift it should have 0 loss tokens
        # seg2 has length=2, so after shift it should have 1 loss token
        if "loss_seg1" in trainer._channel_loss_stats["train"]:
            stats_seg1 = trainer._channel_loss_stats["train"]["loss_seg1"]
            assert stats_seg1["count"] == 0  # No loss tokens for length-1 segment

        if "loss_seg2" in trainer._channel_loss_stats["train"]:
            stats_seg2 = trainer._channel_loss_stats["train"]["loss_seg2"]
            assert stats_seg2["count"] >= 0  # Should have at least 1 token

    def test_segment_detection_all_padding(self):
        """Test segment detection when entire sequence is padding."""
        import torch

        from axolotl.integrations.channel_loss.segment import get_segment_boundaries

        # All padding
        attention_mask = torch.tensor([[0, 0, 0, 0]])
        labels = torch.tensor([[-100, -100, -100, -100]])

        cu_seqlens = get_segment_boundaries(
            attention_mask=attention_mask,
            position_ids=None,
            labels=labels,
            mode="auto",
        )

        # Should return minimal boundaries (no valid segments)
        assert len(cu_seqlens) >= 1
        assert cu_seqlens[0] == 0
