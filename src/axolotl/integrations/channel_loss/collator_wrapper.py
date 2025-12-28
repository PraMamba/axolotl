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
Collator wrapper for Channel Loss.

Wraps existing collator to extract and pass through channel information
without breaking tokenizer.pad() which cannot handle string fields.

Design notes:
- We wrap the existing collator instance instead of creating a new class
- This ensures compatibility with any collator (standard, packing, KD, etc.)
- Channel strings are extracted before calling inner collator, then added back to batch
"""

from typing import Any, Callable, Dict, List, Optional

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def wrap_collator_for_channel_loss(
    inner_collator: Callable,
    channel_field: str = "channel",
    dataset_channels: Optional[List[str]] = None,
    warn_on_missing: bool = True,
) -> Callable:
    """
    Wrap an existing collator to handle channel field extraction.

    Args:
        inner_collator: The original collator function/object to wrap.
        channel_field: Field name containing channel info in each sample.
        dataset_channels: Optional pre-defined channel mapping from dataset config.
                         Used when channel is specified at dataset level, not sample level.
        warn_on_missing: Whether to warn when channel field is missing.

    Returns:
        Wrapped collator function.

    Design:
        1. Extract channel strings from features (before tokenizer.pad breaks on strings)
        2. Remove channel field from features
        3. Call inner collator to get batch tensors
        4. Add channel list back to batch dict
    """
    _warned_missing = [False]  # Use list to allow mutation in nested function

    def wrapped_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Wrapped collator that handles channel field.

        Supports both standard format (List[dict]) and packing format (List[List[dict]]).
        """
        if not features:
            return inner_collator(features)

        # Detect if this is packing format (List[List[dict]])
        is_packing = isinstance(features[0], list)

        if is_packing:
            return _process_packing_batch(
                features,  # type: ignore[arg-type]
                inner_collator,
                channel_field,
                dataset_channels,
                warn_on_missing,
                _warned_missing,
            )
        else:
            return _process_standard_batch(
                features,
                inner_collator,
                channel_field,
                dataset_channels,
                warn_on_missing,
                _warned_missing,
            )

    return wrapped_collator


def _process_standard_batch(
    features: List[Dict[str, Any]],
    inner_collator: Callable,
    channel_field: str,
    dataset_channels: Optional[List[str]],
    warn_on_missing: bool,
    warned_missing: List[bool],
) -> Dict[str, Any]:
    """
    Process standard batch format (List[dict]).

    Each dict is one sample. Extract channel from each sample.
    """
    channels = []

    for _i, feat in enumerate(features):
        # Try to get channel from sample
        ch = feat.pop(channel_field, None)

        if ch is None and dataset_channels:
            # Fallback to dataset-level channel mapping
            # This requires knowing which dataset each sample came from
            # For now, we use 'default' as fallback
            ch = "default"

        if ch is None:
            ch = "default"
            if warn_on_missing and not warned_missing[0]:
                LOG.warning(
                    f"Channel field '{channel_field}' not found in sample. "
                    f"Using 'default' as channel. This warning will only be shown once."
                )
                warned_missing[0] = True

        channels.append(ch)

    # Call inner collator (without channel strings that would break tokenizer.pad)
    batch = inner_collator(features)

    # Add channel list to batch
    if any(ch != "default" for ch in channels):
        batch["channel"] = channels

    return batch


def _process_packing_batch(
    features: List[List[Dict[str, Any]]],
    inner_collator: Callable,
    channel_field: str,
    dataset_channels: Optional[List[str]],
    warn_on_missing: bool,
    warned_missing: List[bool],
) -> Dict[str, Any]:
    """
    Process packing batch format (List[List[dict]]).

    Each inner list contains multiple samples packed into one sequence.
    We need to preserve the nested structure for segment mapping.
    """
    all_channels = []  # List[List[str]]

    for sub_batch in features:
        sub_channels = []
        for feat in sub_batch:
            ch = feat.pop(channel_field, None)

            if ch is None:
                ch = "default"
                if warn_on_missing and not warned_missing[0]:
                    LOG.warning(
                        f"Channel field '{channel_field}' not found in packed sample. "
                        f"Using 'default' as channel. This warning will only be shown once."
                    )
                    warned_missing[0] = True

            sub_channels.append(ch)

        all_channels.append(sub_channels)

    # Call inner collator
    batch = inner_collator(features)

    # Add nested channel list to batch
    # This preserves the packing structure: List[List[str]]
    has_non_default = any(ch != "default" for sub in all_channels for ch in sub)
    if has_non_default:
        batch["channel"] = all_channels

    return batch
