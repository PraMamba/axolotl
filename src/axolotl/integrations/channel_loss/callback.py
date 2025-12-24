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
Channel Loss Logging Callback.

Handles distributed synchronization and logging of per-channel loss metrics.

Design notes:
- Synchronization happens only on logging_steps (on_log) to minimize overhead
- Uses all_gather_object to align keys across ranks (prevent deadlock)
- Batch all_reduce for efficient communication
- Statistic accumulators are reset after each logging event
"""

from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ChannelLossLoggingCallback(TrainerCallback):
    """
    Callback for logging per-channel loss metrics.

    This callback:
    1. Collects accumulated statistics from trainer._channel_loss_stats
    2. Synchronizes across distributed ranks (if applicable)
    3. Computes mean loss per channel
    4. Injects metrics into logs dict
    5. Resets accumulators for next logging window

    Design:
        - Triggered on_log (training) and on_evaluate (evaluation)
        - Handles key alignment to prevent distributed deadlocks
        - Efficient batch all_reduce for multiple channels
    """

    def __init__(self, trainer, cfg: Dict[str, Any]):
        """
        Initialize callback.

        Args:
            trainer: The trainer instance (for accessing _channel_loss_stats).
            cfg: Configuration dict containing channel loss settings.
        """
        self.trainer = trainer
        self.prefix = cfg.get("channel_loss_prefix", "loss_")
        self._logged_channels = set()  # Track which channels we've seen

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        """
        Called when training logs are being recorded.

        Synchronizes and injects channel loss metrics into logs.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control object.
            logs: Dict of metrics to log (modified in place).
        """
        if logs is None:
            return

        stats = self.trainer._channel_loss_stats["train"]

        if not stats:
            return

        # Compute synchronized metrics
        channel_logs = self._compute_and_sync(stats)

        # Log new channels
        for key in channel_logs:
            if key not in self._logged_channels:
                LOG.info(f"Channel Loss: Tracking new channel '{key}'")
                self._logged_channels.add(key)

        # Inject into logs
        logs.update(channel_logs)

        # Reset accumulators for next logging window
        stats.clear()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        """
        Called at the end of evaluation.

        Synchronizes and injects channel loss metrics into evaluation metrics.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control object.
            metrics: Dict of evaluation metrics (modified in place).
        """
        if metrics is None:
            return

        stats = self.trainer._channel_loss_stats["eval"]
        if not stats:
            return

        # Compute synchronized metrics
        channel_logs = self._compute_and_sync(stats)

        # Add eval_ prefix to distinguish from training metrics
        eval_logs = {f"eval_{k}": v for k, v in channel_logs.items()}

        # Inject into metrics
        metrics.update(eval_logs)

        # Reset accumulators
        stats.clear()

    def _compute_and_sync(self, stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute mean loss per channel with distributed synchronization.

        Args:
            stats: Dict mapping channel key to {"sum": float, "count": int}.

        Returns:
            Dict mapping channel key to mean loss value.

        Design:
            For distributed training:
            1. all_gather_object to collect all keys from all ranks
            2. Ensure all ranks have same key set (prevents deadlock)
            3. Build tensor for batch all_reduce
            4. Compute mean from synchronized sum/count
        """
        logs = {}

        if not dist.is_initialized():
            # Single GPU: compute mean directly
            for key, data in stats.items():
                if data["count"] > 0:
                    logs[key] = data["sum"] / data["count"]
            return logs

        # === Distributed synchronization ===

        # Step 1: Align keys across all ranks (prevent deadlock)
        # Different ranks may have different channels in their micro-batches
        local_keys = list(stats.keys())
        world_size = dist.get_world_size()

        all_keys_list = [None] * world_size
        dist.all_gather_object(all_keys_list, local_keys)

        # Merge all keys
        all_keys = set()
        for keys in all_keys_list:
            if keys is not None:
                all_keys.update(keys)

        if not all_keys:
            return logs

        # Ensure local stats has all keys (with zeros for missing)
        for key in all_keys:
            if key not in stats:
                stats[key] = {"sum": 0.0, "count": 0}

        # Step 2: Build tensor for batch all_reduce
        # Shape: (num_channels, 2) where [:, 0] = sum, [:, 1] = count
        sorted_keys = sorted(all_keys)
        device = self._get_device()

        data_tensor = torch.zeros(len(sorted_keys), 2, dtype=torch.float64, device=device)
        for i, key in enumerate(sorted_keys):
            data_tensor[i, 0] = stats[key]["sum"]
            data_tensor[i, 1] = stats[key]["count"]

        # Step 3: All-reduce sum and count
        dist.all_reduce(data_tensor, op=dist.ReduceOp.SUM)

        # Step 4: Compute mean
        for i, key in enumerate(sorted_keys):
            total_sum = data_tensor[i, 0].item()
            total_count = data_tensor[i, 1].item()
            if total_count > 0:
                logs[key] = total_sum / total_count

        return logs

    def _get_device(self) -> torch.device:
        """
        Get the device to use for tensors.

        Returns the device of the trainer's model, or CUDA if available,
        or CPU as fallback.
        """
        if hasattr(self.trainer, "model") and hasattr(self.trainer.model, "device"):
            return self.trainer.model.device

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")
