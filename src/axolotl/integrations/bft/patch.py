"""Monkey patch for trainer.compute_loss to apply BFT loss in SFT training."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch

from axolotl.utils.distributed import is_distributed, is_main_process
from axolotl.utils.logging import get_logger

from .bft_utils import compute_bft_loss

LOG = get_logger(__name__)


def patch_compute_loss_for_bft(trainer, cfg) -> None:
    """Patch a trainer instance to apply BFT loss when enabled."""
    original_compute_loss = trainer.compute_loss

    # Defensive checks (plugin also enforces these).
    if cfg is not None:
        if getattr(cfg, "rl", False) or getattr(cfg, "reward_model", False) or getattr(
            cfg, "process_reward_model", False
        ):
            LOG.warning("BFTPlugin is intended for SFT; skipping for RL/Reward paths.")
            return

        if getattr(cfg, "enable_dft_loss", False):
            msg = (
                "BFT and DFT are mutually exclusive. "
                "Please set enable_dft_loss=false when using BFT."
            )
            raise ValueError(msg)

        if getattr(cfg, "label_smoothing_factor", 0.0) not in (0, 0.0, None):
            msg = (
                "BFT is incompatible with label smoothing (label_smoothing_factor > 0)."
            )
            raise ValueError(msg)

    # Merge BFT metrics into the next framework log call to avoid double-log lines.
    if hasattr(trainer, "log") and getattr(trainer, "_bft_log_patched", False) is not True:
        original_log = trainer.log

        def log_with_bft_metrics(self, logs: dict[str, float], *args, **kwargs):
            pending = getattr(self.state, "_bft_pending_metrics", None)
            if "loss" in logs and isinstance(pending, dict) and pending:
                for key, value in pending.items():
                    logs.setdefault(key, value)
                self.state._bft_pending_metrics = {}
            return original_log(logs, *args, **kwargs)

        trainer.log = MethodType(log_with_bft_metrics, trainer)
        trainer._bft_log_patched = True

    # Store global step for warmup calculation
    bft_warmup_steps = getattr(getattr(trainer, "args", None), "bft_warmup_steps", 100)

    def compute_loss_with_bft(
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Any]:
        # Check if BFT is enabled
        if not getattr(trainer.args, "enable_bft_loss", False):
            return original_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # Incompatibility checks (inherited from DFT)
        if getattr(trainer.args, "orpo_alpha", None):
            return original_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if getattr(trainer.args, "label_smoothing_factor", 0.0) not in (0, 0.0, None):
            msg = (
                "BFT loss is currently incompatible with label smoothing "
                "(label_smoothing_factor > 0)."
            )
            raise ValueError(msg)

        # Extract labels
        labels = inputs.get("labels")
        if labels is None:
            return original_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # Track tokens for throughput calculation (reuse DFT logic)
        if getattr(trainer.args, "include_tkps", False) and model.training:
            inputs_key = "labels" if "labels" in inputs else "input_ids"
            trainable_tokens = (inputs[inputs_key] != -100).sum()
            total_tokens = torch.tensor(
                inputs[inputs_key].numel(),
                dtype=torch.long,
                device=inputs[inputs_key].device,
            )

            if is_distributed():
                torch.distributed.all_reduce(
                    trainable_tokens, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    total_tokens, op=torch.distributed.ReduceOp.SUM
                )

            # Initialize state.tokens dict if not exists
            if not hasattr(trainer.state, "tokens"):
                trainer.state.tokens = {
                    "trainable": torch.zeros(1),
                    "total": torch.zeros(1),
                }

            # Accumulate tokens
            trainer.state.tokens["trainable"] = (
                trainer.state.tokens["trainable"] + trainable_tokens.detach().cpu()
            )
            trainer.state.tokens["total"] = trainer.state.tokens["total"] + total_tokens.cpu()
            trainer.state.tokens["trainable_tokens"] = trainable_tokens.detach().cpu()

        # Pop labels to prevent CP pre-hook from sharding them
        forward_inputs = dict(inputs)
        labels = forward_inputs.pop("labels")

        # Forward pass
        outputs = model(**forward_inputs)
        logits = _extract_logits(outputs)
        if logits is None:
            return original_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        # Calculate warmup factor
        current_step = trainer.state.global_step
        warmup_factor = min(1.0, current_step / max(1, bft_warmup_steps))

        # Get BFT parameters from trainer.args
        chunk_size = getattr(trainer.args, "dft_chunk_size", None)
        group_size = getattr(trainer.args, "bft_group_size", 256)
        min_valid_tokens = getattr(trainer.args, "bft_min_valid_tokens_in_window", 32)
        weight_floor = getattr(trainer.args, "bft_weight_floor", 0.1)
        weight_ceiling = getattr(trainer.args, "bft_weight_ceiling", 2.0)
        normalize_sample_weight = getattr(trainer.args, "bft_normalize_sample_weight", True)

        # Compute BFT loss with metrics
        loss, metrics = compute_bft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            num_items_in_batch=num_items_in_batch,
            chunk_size=chunk_size,
            trainer=trainer,
            group_size=group_size,
            min_valid_tokens=min_valid_tokens,
            weight_floor=weight_floor,
            weight_ceiling=weight_ceiling,
            normalize_sample_weight=normalize_sample_weight,
            warmup_factor=warmup_factor,
        )

        # Cache BFT metrics and merge on the next trainer.log call.
        if model.training:
            if not hasattr(trainer.state, "_bft_pending_metrics"):
                trainer.state._bft_pending_metrics = {}
            if not is_distributed() or is_main_process():
                trainer.state._bft_pending_metrics = metrics

        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = compute_loss_with_bft


def _extract_logits(outputs: Any) -> torch.Tensor | None:
    """Extract logits from model outputs (reused from DFT)."""
    if outputs is None:
        return None
    if isinstance(outputs, dict):
        return outputs.get("logits")
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    if isinstance(outputs, SimpleNamespace) and hasattr(outputs, "logits"):
        return outputs.logits
    return None
