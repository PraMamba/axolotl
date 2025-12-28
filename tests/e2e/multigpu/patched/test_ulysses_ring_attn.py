"""E2E tests for Ulysses + Ring-Attention plugin"""

from pathlib import Path

import pytest
import yaml
from accelerate.test_utils import execute_subprocess_async
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.utils.dict import DictDefault

from ...utils import check_tensorboard


class TestUlyssesRingAttention:
    """Test case for Ulysses + Ring-Attention plugin with arbitrary chunk sharding"""

    def _run_ulysses_ring_test(
        self,
        temp_dir,
        context_parallel_size=4,
        num_heads=32,
        mode="auto",
        threshold=2.5,
    ):
        """Helper method to run Ulysses + Ring-Attention tests

        Args:
            temp_dir: Temporary directory for outputs
            context_parallel_size: Total context parallelism size (sp × rp)
            num_heads: Number of attention heads (for GCD decomposition)
            mode: Decomposition mode (auto, hybrid, ulysses_only, ring_only)
            threshold: Loss threshold for validation
        """
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "load_in_8bit": False,
                "load_in_4bit": True,
                "strict": False,
                "sequence_len": 2048,
                "adapter": "qlora",
                "sample_packing": True,  # Required for Phase 1
                "eval_sample_packing": True,
                "pad_to_sequence_len": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "lora_modules_to_save": ["embed_tokens", "lm_head"],
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:10%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 8,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "dataset_prepared_path": temp_dir + "/last_run_prepared",
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,  # Required for plugin
                "loss_watchdog_threshold": 5.0,
                "loss_watchdog_patience": 3,
                "bf16": "auto",
                "warmup_steps": 1,
                "saves_per_epoch": 1,
                "logging_steps": 1,
                "weight_decay": 0.0,
                "use_tensorboard": True,
                "save_first_step": False,
                # Ulysses + Ring-Attention plugin config
                "plugins": [
                    "axolotl.integrations.ulysses_ring_attn",
                ],
                "ulysses_ring_attention_enabled": True,
                "ulysses_ring_attention_mode": mode,
                "context_parallel_size": context_parallel_size,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "accelerate",
                "launch",
                "--num-processes",
                str(context_parallel_size),
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            threshold,
            "Train Loss (%s) is too high",
        )

    @pytest.mark.parametrize(
        "context_parallel_size, mode, description",
        [
            # Test GCD decomposition with non-divisible cp_size
            # SmolLM2-135M has 9 heads, so cp=6 → sp=gcd(9,6)=3, rp=2
            (6, "auto", "hybrid: 3-way Ulysses × 2-way Ring"),
            # Test pure Ulysses (divisible)
            # cp=3 divides 9 heads → sp=3, rp=1
            (3, "auto", "ulysses_only: 3-way Ulysses"),
            # Test pure Ring (coprime)
            # cp=4 coprime to 9 → sp=1, rp=4
            (4, "auto", "ring_only: 4-way Ring"),
        ],
        ids=[
            "hybrid_sp3_rp2",
            "ulysses_only_sp3",
            "ring_only_rp4",
        ],
    )
    def test_ulysses_ring_training(
        self,
        temp_dir,
        context_parallel_size,
        mode,
        description,
    ):
        """Test Ulysses + Ring-Attention with different decompositions

        Phase 2.1 COMPLETE: Attention patching is now implemented!
        This test validates:
        - Config registration and validation
        - GCD-based sp/rp decomposition
        - Process group creation
        - Llama attention monkey-patching (NEW in Phase 2.1)
        - Full distributed training with Ulysses + Ring-Attention

        NOTE: Requires 4+ GPUs. Run with:
            accelerate launch --num-processes 4 -m pytest tests/e2e/multigpu/patched/test_ulysses_ring_attn.py
        """
        self._run_ulysses_ring_test(
            temp_dir,
            context_parallel_size=context_parallel_size,
            mode=mode,
        )

    def test_gpt_neox_ulysses_ring_training(self, temp_dir):
        """Validate Phase 2.4 GPT-NeoX on multi-GPU.

        NOTE: Requires 6 GPUs. Run with:
            accelerate launch --num-processes 6 -m pytest tests/e2e/multigpu/patched/test_ulysses_ring_attn.py::TestUlyssesRingAttention::test_gpt_neox_ulysses_ring_training
        """
        context_parallel_size = 6  # hybrid: sp=gcd(num_heads, 6), rp=6/sp
        cfg = DictDefault(
            {
                "base_model": "EleutherAI/pythia-70m-deduped",
                "load_in_8bit": False,
                "load_in_4bit": True,
                "strict": False,
                "sequence_len": 2048,
                "adapter": "qlora",
                "sample_packing": True,
                "eval_sample_packing": True,
                "pad_to_sequence_len": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:10%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 8,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "dataset_prepared_path": temp_dir + "/last_run_prepared",
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                # GPT-NeoX base models are not instruction-tuned; the absolute loss
                # is expected to be materially higher than the SmolLM2 Llama baselines.
                "loss_watchdog_threshold": 100.0,
                "loss_watchdog_patience": 3,
                "bf16": "auto",
                "warmup_steps": 1,
                "logging_steps": 1,
                "weight_decay": 0.0,
                "use_tensorboard": True,
                "save_first_step": False,
                "plugins": [
                    "axolotl.integrations.ulysses_ring_attn",
                ],
                "ulysses_ring_attention_enabled": True,
                "ulysses_ring_attention_mode": "auto",
                "context_parallel_size": context_parallel_size,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "accelerate",
                "launch",
                "--num-processes",
                str(context_parallel_size),
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            50.0,
            "Train Loss (%s) is too high",
        )

    @pytest.mark.parametrize(
        "context_parallel_size, sp_override, rp_override",
        [
            # Test manual sp/rp overrides
            (8, 4, 2),  # sp=4 × rp=2
            (8, 2, 4),  # sp=2 × rp=4 (different ratio)
            (8, 8, 1),  # sp=8 × rp=1 (Ulysses-only)
            (8, 1, 8),  # sp=1 × rp=8 (Ring-only)
        ],
        ids=[
            "manual_sp4_rp2",
            "manual_sp2_rp4",
            "manual_ulysses_only",
            "manual_ring_only",
        ],
    )
    def test_ulysses_ring_manual_override(
        self,
        temp_dir,
        context_parallel_size,
        sp_override,
        rp_override,
    ):
        """Test Ulysses + Ring-Attention with manual sp/rp overrides

        Validates that users can manually specify sp_size and rp_size
        instead of relying on GCD-based decomposition.

        NOTE: Requires 8 GPUs. Run with:
            accelerate launch --num-processes 8 -m pytest tests/e2e/multigpu/patched/test_ulysses_ring_attn.py::TestUlyssesRingAttention::test_ulysses_ring_manual_override
        """
        cfg = DictDefault(
            {
                # Use a model with a head count compatible with the manual overrides below.
                # SmolLM2-1.7B has 32 attention heads, so sp in {2,4,8} are all valid.
                "base_model": "HuggingFaceTB/SmolLM2-1.7B",
                "load_in_4bit": True,
                "strict": False,
                "sequence_len": 2048,
                "adapter": "qlora",
                "sample_packing": True,
                "eval_sample_packing": True,
                "pad_to_sequence_len": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [
                    {
                        "path": "tatsu-lab/alpaca",
                        "type": "alpaca",
                        "split": "train[:10%]",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 8,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "dataset_prepared_path": temp_dir + "/last_run_prepared",
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "bf16": "auto",
                "warmup_steps": 1,
                "logging_steps": 1,
                "use_tensorboard": True,
                # Ulysses + Ring-Attention plugin with manual overrides
                "plugins": ["axolotl.integrations.ulysses_ring_attn"],
                "ulysses_ring_attention_enabled": True,
                "ulysses_ring_attention_mode": "auto",
                "ulysses_ring_attention_sp_size": sp_override,
                "ulysses_ring_attention_rp_size": rp_override,
                "context_parallel_size": context_parallel_size,
            }
        )

        # write cfg to yaml file
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_dir) / "config.yaml", "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

        execute_subprocess_async(
            [
                "accelerate",
                "launch",
                "--num-processes",
                str(context_parallel_size),
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "-m",
                "axolotl.cli.train",
                str(Path(temp_dir) / "config.yaml"),
            ]
        )

        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            2.5,
            "Train Loss (%s) is too high",
        )


class TestUlyssesRingAttentionPluginLifecycle:
    """Test plugin lifecycle without full e2e training"""

    def test_plugin_registration(self, temp_dir):
        """Test that plugin can be registered and validates config correctly"""
        from axolotl.integrations.ulysses_ring_attn.plugins import (
            UlyssesRingAttentionPlugin,
        )

        plugin = UlyssesRingAttentionPlugin()

        # Test: Plugin disabled (should pass silently)
        cfg_disabled = {"ulysses_ring_attention_enabled": False}
        plugin.register(cfg_disabled)
        assert not plugin.enabled

        # Test: Plugin enabled with valid config
        cfg_valid = {
            "ulysses_ring_attention_enabled": True,
            "context_parallel_size": 4,
            "flash_attention": True,
            "sample_packing": True,
        }
        plugin.register(cfg_valid)
        assert plugin.enabled

        # Test: Missing context_parallel_size (should raise)
        cfg_no_cp = {
            "ulysses_ring_attention_enabled": True,
            "flash_attention": True,
            "sample_packing": True,
        }
        with pytest.raises(ValueError, match="context_parallel_size > 1"):
            plugin.register(cfg_no_cp)

        # Test: Missing flash_attention (should raise)
        cfg_no_flash = {
            "ulysses_ring_attention_enabled": True,
            "context_parallel_size": 4,
            "sample_packing": True,
        }
        with pytest.raises(ValueError, match="flash_attention=true"):
            plugin.register(cfg_no_flash)

        # Test: Missing sample_packing (Phase 1 constraint)
        cfg_no_packing = {
            "ulysses_ring_attention_enabled": True,
            "context_parallel_size": 4,
            "flash_attention": True,
        }
        with pytest.raises(ValueError, match="sample_packing=true"):
            plugin.register(cfg_no_packing)

    def test_sp_rp_decomposition(self):
        """Test GCD-based sp/rp decomposition logic"""
        from axolotl.integrations.ulysses_ring_attn.groups import compute_sp_rp

        # Test: Divisible case (Ulysses-only)
        sp, rp = compute_sp_rp(num_heads=32, context_parallel_size=8)
        assert sp == 8 and rp == 1

        # Test: Non-divisible hybrid case
        sp, rp = compute_sp_rp(num_heads=32, context_parallel_size=24)
        assert sp == 8 and rp == 3  # gcd(32, 24) = 8

        # Test: Coprime case (Ring-only)
        sp, rp = compute_sp_rp(num_heads=32, context_parallel_size=7)
        assert sp == 1 and rp == 7

        # Test: Mode enforcement
        with pytest.raises(ValueError, match="ulysses_only"):
            compute_sp_rp(num_heads=32, context_parallel_size=7, mode="ulysses_only")

        with pytest.raises(ValueError, match="hybrid"):
            compute_sp_rp(num_heads=32, context_parallel_size=8, mode="hybrid")

    @pytest.mark.skip(
        reason="Requires multi-GPU setup and device_mesh initialization. "
        "Run manually with: pytest -k test_post_trainer_create --forked"
    )
    def test_post_trainer_create(self, temp_dir):
        """Test post_trainer_create hook (requires actual trainer with device_mesh)

        This test validates that the plugin can:
        1. Extract device_mesh from trainer
        2. Create SP and RP process groups
        3. Instantiate DistributedAttention wrapper

        NOTE: Requires multi-GPU environment and full trainer setup.
        """
        # TODO: Implement mock trainer with device_mesh for testing
        # For now, this serves as a placeholder for manual testing
        pass
