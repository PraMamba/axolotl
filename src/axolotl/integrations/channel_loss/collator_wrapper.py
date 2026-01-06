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
    dataset_channels: Optional[Dict[int, str]] = None,
    warn_on_missing: bool = True,
) -> Callable:
    """
    Wrap an existing collator to handle channel field extraction.

    Args:
        inner_collator: The original collator function/object to wrap.
        channel_field: Field name containing channel info in each sample.
        dataset_channels: Optional dict mapping dataset index to channel name.
                         Used when channel is specified at dataset level, not sample level.
                         Format: {0: "channel1", 1: "channel2", ...}
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

    # Precompute the dataset_idx field name
    dataset_idx_field = f"_{channel_field}_dataset_idx"

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
                dataset_idx_field,
                warn_on_missing,
                _warned_missing,
            )
        else:
            return _process_standard_batch(
                features,
                inner_collator,
                channel_field,
                dataset_channels,
                dataset_idx_field,
                warn_on_missing,
                _warned_missing,
            )

    return wrapped_collator


def _process_standard_batch(
    features: List[Dict[str, Any]],
    inner_collator: Callable,
    channel_field: str,
    dataset_channels: Optional[Dict[int, str]],
    dataset_idx_field: str,
    warn_on_missing: bool,
    warned_missing: List[bool],
) -> Dict[str, Any]:
    """
    Process standard batch format (List[dict]).

    Each dict is one sample. Extract channel from each sample.
    """
    channels = []

    # DEBUG: Log what fields we receive (use DEBUG level to avoid production noise)
    if features:
        LOG.debug(
            f"[Collator] Received {len(features)} features, first feature keys: {list(features[0].keys())}, channel_field: {channel_field}"
        )

    for _i, feat in enumerate(features):
        # Try to get channel directly from sample first
        ch = feat.pop(channel_field, None)

        # Always remove the metadata field (prevents scalar concatenation errors)
        dataset_idx = feat.pop(dataset_idx_field, None)

        # If channel not found directly, try to get from dataset_idx mapping
        if ch is None and dataset_channels and dataset_idx is not None:
            # DEBUG: Log extraction for first feature
            if _i == 0:
                LOG.debug(
                    f"[Collator] First feature: channel={ch}, dataset_idx={dataset_idx}, "
                    f"dataset_channels={dataset_channels}"
                )

            # Lookup channel from mapping
            if dataset_idx in dataset_channels:
                ch = dataset_channels[dataset_idx]
            else:
                LOG.warning(
                    f"Channel Loss: dataset_idx={dataset_idx} not found in dataset_channels mapping. "
                    f"Using 'default' as channel."
                )
                ch = "default"

        # Fallback to default if still None
        if ch is None:
            ch = "default"
            if warn_on_missing and not warned_missing[0]:
                LOG.warning(
                    f"Channel field '{channel_field}' not found in sample and no dataset_idx mapping available. "
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
    dataset_channels: Optional[Dict[int, str]],
    dataset_idx_field: str,
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
            # Try to get channel directly from sample first
            ch = feat.pop(channel_field, None)

            # Always remove the metadata field (prevents scalar concatenation errors)
            dataset_idx = feat.pop(dataset_idx_field, None)

            # If channel not found directly, try to get from dataset_idx mapping
            if ch is None and dataset_channels and dataset_idx is not None:
                # Lookup channel from mapping
                if dataset_idx in dataset_channels:
                    ch = dataset_channels[dataset_idx]
                else:
                    LOG.warning(
                        f"Channel Loss: dataset_idx={dataset_idx} not found in dataset_channels mapping. "
                        f"Using 'default' as channel."
                    )
                    ch = "default"

            # Fallback to default if still None
            if ch is None:
                ch = "default"
                if warn_on_missing and not warned_missing[0]:
                    LOG.warning(
                        f"Channel field '{channel_field}' not found in packed sample and no dataset_idx mapping available. "
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
