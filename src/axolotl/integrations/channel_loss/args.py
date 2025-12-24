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
Channel Loss Plugin configuration arguments.

Ported from ms-swift framework's Channel Loss feature.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class ChannelLossArgs(BaseModel):
    """
    Pydantic configuration model for Channel Loss Plugin.

    These fields will be merged into AxolotlInputConfig via plugin mechanism.
    """

    enable_channel_loss: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable Channel Loss training. "
            "When enabled, loss will be tracked per channel and logged separately."
        },
    )

    channel_loss_field: str | None = Field(
        default="channel",
        json_schema_extra={
            "description": "The field name in the dataset that contains the channel information. "
            "Default: 'channel'"
        },
    )

    channel_loss_prefix: str | None = Field(
        default="loss_",
        json_schema_extra={
            "description": "Prefix for channel loss metrics in logs. "
            "E.g., 'loss_' will result in 'loss_math', 'loss_code', etc. "
            "Default: 'loss_'"
        },
    )

    channel_loss_segment: Literal["auto", "position_ids", "attention_mask"] | None = Field(
        default="auto",
        json_schema_extra={
            "description": "Segment detection strategy for packing mode. "
            "'auto': prefer attention_mask segment ids, fallback to position_ids. "
            "'position_ids': use position_ids == 0 as segment boundary. "
            "'attention_mask': use attention_mask value changes as segment boundary. "
            "Default: 'auto'"
        },
    )

    channel_loss_warn_on_missing: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Warn when channel field is missing in sample. "
            "Default: True"
        },
    )


@dataclass
class ChannelLossTrainingArgsMixin:
    """
    TrainingArgs Mixin for Channel Loss Plugin.

    These fields will be merged into AxolotlTrainingMixins via plugin mechanism.
    """

    enable_channel_loss: bool | None = None
    channel_loss_field: str | None = "channel"
    channel_loss_prefix: str | None = "loss_"
    channel_loss_segment: str | None = "auto"
    channel_loss_warn_on_missing: bool | None = True
