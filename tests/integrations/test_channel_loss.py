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

from axolotl.integrations.channel_loss.segment import (
    flatten_channels,
    get_segment_boundaries,
)
from axolotl.integrations.channel_loss.collator_wrapper import (
    wrap_collator_for_channel_loss,
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
