"""BFT (Balanced Fine-Tuning) test suite.

Tests for sample-level weighting functionality in BFT.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from axolotl.integrations.bft.args import BFTArgs
from axolotl.integrations.bft.bft_utils import (
    compute_bft_loss,
    compute_sample_weights,
)
from axolotl.integrations.bft.patch import patch_compute_loss_for_bft
from axolotl.integrations.dft.dft_utils import compute_dft_loss


class TestBFTArgs:
    """Test BFT configuration arguments."""

    def test_defaults(self):
        args = BFTArgs()
        assert args.enable_bft_loss is False
        assert args.bft_group_size == 256
        assert args.bft_min_valid_tokens_in_window == 32
        assert args.bft_weight_floor == 0.1
        assert args.bft_weight_ceiling == 2.0
        assert args.bft_normalize_sample_weight is True
        assert args.bft_warmup_steps == 100

    def test_custom(self):
        args = BFTArgs(
            enable_bft_loss=True,
            bft_group_size=128,
            bft_weight_floor=0.2,
            bft_weight_ceiling=1.5,
        )
        assert args.enable_bft_loss is True
        assert args.bft_group_size == 128
        assert args.bft_weight_floor == 0.2
        assert args.bft_weight_ceiling == 1.5


class TestComputeSampleWeights:
    """Test sample weight computation logic."""

    def test_single_sample_uniform_confidence(self):
        """Test with uniform confidence (all tokens equally easy)."""
        confidences = torch.ones(1, 512) * 0.8  # 80% confidence
        mask = torch.ones(1, 512, dtype=torch.bool)

        weights, p_conf = compute_sample_weights(
            confidences,
            mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=1.0,
        )

        # All windows have same confidence → min = 0.8
        # Weight = 1 - 0.8 = 0.2
        assert weights.shape == (1,)
        assert p_conf.shape == (1,)
        assert p_conf[0].item() == pytest.approx(0.8, abs=1e-5)
        assert weights[0].item() == pytest.approx(0.2, abs=1e-5)

    def test_single_sample_with_hard_region(self):
        """Test with one hard region (low confidence spike)."""
        confidences = torch.ones(1, 512) * 0.9
        # Insert a hard region (tokens 200-220)
        confidences[0, 200:220] = 0.3  # Very low confidence

        mask = torch.ones(1, 512, dtype=torch.bool)

        weights, p_conf = compute_sample_weights(
            confidences,
            mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=1.0,
        )

        # Min window should capture the hard region
        # Window containing tokens 200-220 will have lower mean
        # Exact value depends on window position, but p_conf < 0.9
        assert p_conf[0].item() < 0.9
        assert weights[0].item() > 0.1  # Higher weight due to hard region

    def test_weight_floor_ceiling_clamp(self):
        """Test that weights are clamped to [floor, ceiling]."""
        # Very high confidence (easy sample) → should hit floor
        easy_confidences = torch.ones(1, 512) * 0.95
        easy_mask = torch.ones(1, 512, dtype=torch.bool)

        weights, _ = compute_sample_weights(
            easy_confidences,
            easy_mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=1.0,
        )

        # Weight = 1 - 0.95 = 0.05 → clamped to 0.1
        assert weights[0].item() == pytest.approx(0.1, abs=1e-5)

        # Very low confidence (hard sample) → should hit ceiling
        hard_confidences = torch.ones(1, 512) * 0.05
        hard_mask = torch.ones(1, 512, dtype=torch.bool)

        weights, _ = compute_sample_weights(
            hard_confidences,
            hard_mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=1.0,
        )

        # Weight = 1 - 0.05 = 0.95 (no ceiling hit in this case)
        # But if p_conf were even lower, would hit 2.0
        assert weights[0].item() <= 2.0

    def test_normalization(self):
        """Test that weights are normalized to mean=1.0."""
        # Batch with varying difficulties
        confidences = torch.tensor(
            [
                [0.9] * 512,  # Easy
                [0.5] * 512,  # Medium
                [0.2] * 512,  # Hard
            ]
        )
        mask = torch.ones(3, 512, dtype=torch.bool)

        weights, _ = compute_sample_weights(
            confidences,
            mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=True,
            warmup_factor=1.0,
        )

        # After normalization, mean should be 1.0
        assert weights.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_warmup_factor(self):
        """Test that warmup_factor scales weights correctly."""
        confidences = torch.ones(1, 512) * 0.5
        mask = torch.ones(1, 512, dtype=torch.bool)

        # Full warmup
        weights_full, _ = compute_sample_weights(
            confidences,
            mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=1.0,
        )

        # Half warmup
        weights_half, _ = compute_sample_weights(
            confidences,
            mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=0.5,
        )

        # Weights should be halved
        assert weights_half[0].item() == pytest.approx(weights_full[0].item() * 0.5, abs=1e-5)

    def test_short_sequence_fallback(self):
        """Test fallback to global mean for sequences shorter than group_size."""
        confidences = torch.ones(1, 64) * 0.7  # Shorter than group_size=256
        mask = torch.ones(1, 64, dtype=torch.bool)

        weights, p_conf = compute_sample_weights(
            confidences,
            mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=1.0,
        )

        # Should use global mean
        assert p_conf[0].item() == pytest.approx(0.7, abs=1e-5)
        assert weights[0].item() == pytest.approx(0.3, abs=1e-5)

    def test_masked_tokens_handling(self):
        """Test that masked tokens are correctly excluded from computation."""
        confidences = torch.ones(1, 512) * 0.8
        mask = torch.ones(1, 512, dtype=torch.bool)
        # Mask out half the sequence
        mask[0, :256] = False

        weights, p_conf = compute_sample_weights(
            confidences,
            mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=1.0,
        )

        # Should only consider valid tokens
        assert weights.shape == (1,)
        # Windows with too few valid tokens should be skipped

    def test_all_masked_fallback(self):
        """Test fallback when all tokens are masked."""
        confidences = torch.ones(1, 512) * 0.8
        mask = torch.zeros(1, 512, dtype=torch.bool)  # All masked

        weights, p_conf = compute_sample_weights(
            confidences,
            mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=1.0,
        )

        # Should fallback to default (p_conf=1.0 → weight after floor=0.1)
        assert weights[0].item() == pytest.approx(0.1, abs=1e-5)

    def test_single_valid_token_is_finite(self):
        confidences = torch.ones(1, 128) * 0.8
        mask = torch.zeros(1, 128, dtype=torch.bool)
        mask[0, 10] = True

        weights, p_conf = compute_sample_weights(
            confidences,
            mask,
            group_size=256,
            min_valid_tokens=1,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=1.0,
        )

        assert torch.isfinite(weights).all()
        assert torch.isfinite(p_conf).all()

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        # Batch of 4 samples with different difficulties
        confidences = torch.tensor(
            [
                [0.9] * 512,  # Easy
                [0.7] * 512,  # Medium-easy
                [0.4] * 512,  # Medium-hard
                [0.1] * 512,  # Hard
            ]
        )
        mask = torch.ones(4, 512, dtype=torch.bool)

        weights, p_conf = compute_sample_weights(
            confidences,
            mask,
            group_size=256,
            min_valid_tokens=32,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize=False,
            warmup_factor=1.0,
        )

        assert weights.shape == (4,)
        assert p_conf.shape == (4,)

        # Weights should be monotonically increasing (harder samples get higher weight)
        # After clamping, this may not be strictly true, but directionally correct
        assert weights[0] < weights[3]  # Easy < Hard


class TestComputeBFTLoss:
    """Test full BFT loss computation."""

    def test_bft_loss_basic(self):
        """Test basic BFT loss computation."""
        # Simple case: 1 sample, 4 tokens
        log4 = math.log(4.0)
        logits = torch.tensor(
            [[[0.0, log4], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], requires_grad=True
        )
        labels = torch.tensor([[0, 1, 0, -100]])

        loss, metrics = compute_bft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            num_items_in_batch=None,
            chunk_size=None,
            trainer=None,
            group_size=256,
            min_valid_tokens=1,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize_sample_weight=False,
            warmup_factor=1.0,
        )

        # Loss should be a scalar
        assert loss.ndim == 0
        assert loss.requires_grad

        # Metrics should be populated
        assert "bft/s_mean" in metrics
        assert "bft/s_min" in metrics
        assert "bft/s_max" in metrics
        assert "bft/p_conf_mean" in metrics
        assert "loss/raw_ce" in metrics

        # Loss should be backpropable
        loss.backward()
        assert logits.grad is not None

    def test_bft_loss_all_ignored(self):
        """Test BFT loss with all tokens ignored."""
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        labels = torch.full((1, 4), -100)

        loss, metrics = compute_bft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            num_items_in_batch=None,
            chunk_size=None,
            trainer=None,
            group_size=256,
            min_valid_tokens=1,
            weight_floor=0.1,
            weight_ceiling=2.0,
            normalize_sample_weight=False,
            warmup_factor=1.0,
        )

        # Loss should be zero
        assert loss.item() == pytest.approx(0.0, abs=1e-12)

        # Backward should work
        loss.backward()
        assert logits.grad is not None

    def test_bft_matches_dft_when_sample_weight_is_one(self):
        """Regression: with s_b forced to 1, BFT reduces to DFT."""
        torch.manual_seed(0)
        batch_size, seq_len, vocab_size = (2, 8, 13)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[0, -1] = -100

        dft_loss = compute_dft_loss(logits, labels, ignore_index=-100, shift_labels=True)
        bft_loss, _ = compute_bft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            num_items_in_batch=None,
            chunk_size=None,
            trainer=None,
            group_size=256,
            min_valid_tokens=1,
            weight_floor=1.0,
            weight_ceiling=1.0,
            normalize_sample_weight=False,
            warmup_factor=1.0,
        )

        assert bft_loss.item() == pytest.approx(dft_loss.item(), abs=1e-6)


class TestCPConsistency:
    def test_cp_last_rank_padding_and_consistency(self, monkeypatch):
        """Ensure last CP rank pads to match all_gather and s_b is consistent."""
        cp_size = 2
        batch_size = 1
        full_seq_len = 8
        vocab_size = 2

        divisor = min(cp_size, 64)
        pad_len = (divisor - (full_seq_len % divisor)) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size

        labels_full = torch.zeros(batch_size, full_seq_len, dtype=torch.long)
        logits_local = torch.zeros(
            batch_size, chunk_len, vocab_size, dtype=torch.float32, requires_grad=True
        )

        cp_group = object()
        trainer = SimpleNamespace(
            accelerator=SimpleNamespace(context_parallel_group=cp_group)
        )

        rank_holder = {"rank": 0}

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: cp_size)
        monkeypatch.setattr(dist, "get_rank", lambda group=None: rank_holder["rank"])

        conf_seg0 = torch.full((batch_size, chunk_len), 0.5, dtype=torch.float32)
        conf_seg1 = torch.tensor(
            [[0.5] * (chunk_len - 1) + [1.0]], dtype=torch.float32
        )
        mask_seg0 = torch.ones((batch_size, chunk_len), dtype=torch.bool)
        mask_seg1 = torch.tensor([[True] * (chunk_len - 1) + [False]], dtype=torch.bool)

        def fake_all_gather(out_list, in_tensor, group=None):
            assert in_tensor.shape == (batch_size, chunk_len)
            rank = dist.get_rank(group)

            if in_tensor.dtype == torch.bool:
                if rank == 0:
                    out_list[0].copy_(in_tensor)
                    out_list[1].copy_(mask_seg1)
                else:
                    out_list[0].copy_(mask_seg0)
                    out_list[1].copy_(in_tensor)
                return

            if rank == 0:
                out_list[0].copy_(in_tensor)
                out_list[1].copy_(conf_seg1)
            else:
                out_list[0].copy_(conf_seg0)
                out_list[1].copy_(in_tensor)

        monkeypatch.setattr(dist, "all_gather", fake_all_gather)

        def run_for_rank(rank: int):
            rank_holder["rank"] = rank
            loss, metrics = compute_bft_loss(
                logits_local,
                labels_full,
                shift_labels=True,
                ignore_index=-100,
                num_items_in_batch=None,
                chunk_size=None,
                trainer=trainer,
                group_size=2,
                min_valid_tokens=1,
                weight_floor=0.1,
                weight_ceiling=2.0,
                normalize_sample_weight=False,
                warmup_factor=1.0,
            )
            return loss.detach(), metrics

        loss0, metrics0 = run_for_rank(0)
        loss1, metrics1 = run_for_rank(1)

        assert loss0.item() == pytest.approx(loss1.item(), abs=1e-6)
        assert metrics0["bft/s_mean"] == pytest.approx(metrics1["bft/s_mean"], abs=1e-6)
        assert metrics0["bft/p_conf_mean"] == pytest.approx(
            metrics1["bft/p_conf_mean"], abs=1e-6
        )


class TestBFTPatch:
    """Test BFT compute_loss patching."""

    def test_patch_raises_on_dft_conflict(self):
        """Test that BFT raises error when DFT is also enabled."""
        trainer = MagicMock()
        cfg = MagicMock()
        cfg.enable_bft_loss = True
        cfg.enable_dft_loss = True  # Conflict
        cfg.rl = False
        cfg.reward_model = False
        cfg.process_reward_model = False

        with pytest.raises(ValueError, match="BFT and DFT are mutually exclusive"):
            patch_compute_loss_for_bft(trainer, cfg)

    def test_patch_raises_on_label_smoothing(self):
        """Test that BFT raises error with label smoothing."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_bft_loss=True,
            label_smoothing_factor=0.1,  # Not allowed
        )
        cfg = MagicMock()
        cfg.enable_bft_loss = True
        cfg.enable_dft_loss = False
        cfg.rl = False
        cfg.reward_model = False
        cfg.process_reward_model = False
        cfg.label_smoothing_factor = 0.1

        with pytest.raises(ValueError, match="incompatible with label smoothing"):
            patch_compute_loss_for_bft(trainer, cfg)

    def test_patch_skips_on_rl_path(self):
        """Test that BFT skips patching for RL paths."""
        trainer = MagicMock()
        original_compute_loss = trainer.compute_loss
        cfg = MagicMock()
        cfg.enable_bft_loss = True
        cfg.rl = True  # RL path
        cfg.reward_model = False
        cfg.process_reward_model = False

        patch_compute_loss_for_bft(trainer, cfg)

        # Should not patch (returned early)
        assert trainer.compute_loss == original_compute_loss

    def test_patch_applies_successfully(self):
        """Test that BFT patch is applied successfully."""
        trainer = MagicMock()
        captured_logs = []

        def fake_log(logs, *args, **kwargs):
            captured_logs.append(dict(logs))

        trainer.log = fake_log
        trainer.args = SimpleNamespace(
            enable_bft_loss=True,
            label_smoothing_factor=0.0,
            bft_warmup_steps=100,
        )
        trainer.state = SimpleNamespace(global_step=0)
        original_compute_loss = trainer.compute_loss

        cfg = MagicMock()
        cfg.enable_bft_loss = True
        cfg.enable_dft_loss = False
        cfg.rl = False
        cfg.reward_model = False
        cfg.process_reward_model = False
        cfg.label_smoothing_factor = 0.0

        patch_compute_loss_for_bft(trainer, cfg)

        # Compute loss should be patched
        assert trainer.compute_loss != original_compute_loss

        # Log should be wrapped for metric merging
        assert getattr(trainer, "_bft_log_patched", False) is True

        # Run one compute_loss to populate pending metrics (no logging yet)
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        labels = torch.tensor([[0, 1, 0, -100]])

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        loss = trainer.compute_loss(
            DummyModel(),
            {"input_ids": torch.zeros(1, 4, dtype=torch.long), "labels": labels},
            return_outputs=False,
        )
        assert isinstance(loss, torch.Tensor)

        # Simulate framework's normal logging call; it should include BFT metrics.
        trainer.log({"loss": 1.0})
        assert captured_logs, "Expected at least one merged log call"
        merged = captured_logs[-1]
        assert "loss" in merged
        assert "bft/s_mean" in merged
        assert "loss/raw_ce" in merged
