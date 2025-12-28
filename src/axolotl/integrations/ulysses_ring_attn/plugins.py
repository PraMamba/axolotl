"""Ulysses + Ring-Attention Plugin for Axolotl.

This plugin enables arbitrary chunk sharding in context parallelism by combining:
- Ulysses attention (all-to-all communication on heads dimension)
- Ring-Attention (ring communication on sequence dimension)

Key innovation: Decomposes context_parallel_size using GCD to separate head-constrained
Ulysses parallelism from unconstrained Ring parallelism.

References:
    - ms-swift: swift/trainers/sequence_parallel/ulysses.py
    - Analysis: docs/analysis/ms_swift_ulysses_ring_attention_implementation.md
    - Spec: specs/006-ulysses-ring-attention-plugin/README.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.ulysses_ring_attn.groups import (
    compute_sp_rp,
    create_ulysses_ring_groups,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import Trainer

    from axolotl.utils.dict import DictDefault


class UlyssesRingAttentionPlugin(BasePlugin):
    """Plugin for Ulysses + Ring-Attention arbitrary chunk sharding.

    Enables using context_parallel_size values that don't divide num_heads by
    automatically decomposing parallelism into:
        sp_world_size = gcd(num_heads, context_parallel_size)
        rp_world_size = context_parallel_size / sp_world_size

    Usage in config.yaml:
        plugins:
          - axolotl.integrations.ulysses_ring_attn

        ulysses_ring_attention_enabled: true
        ulysses_ring_attention_mode: auto  # or hybrid, ulysses_only, ring_only
        context_parallel_size: 24  # Works with 32 heads!
        flash_attention: true
        sample_packing: true

    Lifecycle:
        1. register(cfg): Validate configuration and detect conflicts
        2. post_trainer_create(cfg, trainer): Setup groups and patch attention
    """

    def __init__(self):
        """Initialize plugin state."""
        super().__init__()
        self.enabled = False
        self.sp_world_size = None
        self.rp_world_size = None
        self.sp_group = None
        self.rp_group = None
        self.sp_rank = None
        self.rp_rank = None
        LOG.info("UlyssesRingAttentionPlugin initialized")

    def get_input_args(self) -> str:
        """Return the pydantic config model for this plugin."""
        return "axolotl.integrations.ulysses_ring_attn.UlyssesRingAttentionArgs"

    def register(self, cfg: dict):
        """Register plugin and validate configuration.

        Args:
            cfg: Unparsed configuration dictionary

        Raises:
            ValueError: If configuration is invalid or has conflicts
        """
        LOG.info("Registering UlyssesRingAttentionPlugin")

        # Check if plugin is enabled
        if not cfg.get("ulysses_ring_attention_enabled"):
            LOG.info("UlyssesRingAttentionPlugin disabled")
            return

        self.enabled = True

        # === Validation: Required fields ===

        # Require context parallelism to be enabled
        if (
            not cfg.get("context_parallel_size")
            or (cfg.get("context_parallel_size") or 0) <= 1
        ):
            raise ValueError(
                "UlyssesRingAttentionPlugin requires context_parallel_size > 1.\n\n"
                "Add to your config:\n"
                "  context_parallel_size: 24  # Or any value > 1\n\n"
                "See: specs/006-ulysses-ring-attention-plugin/README.md"
            )

        # Require flash attention (dependency for ring-flash-attn)
        if not cfg.get("flash_attention"):
            raise ValueError(
                "UlyssesRingAttentionPlugin requires flash_attention=true.\n\n"
                "Add to your config:\n"
                "  flash_attention: true\n\n"
                "Ring-attention uses flash-attn kernels for efficiency."
            )

        # Phase 1 constraint: Require sample packing for varlen API
        if cfg.get("ulysses_ring_attention_require_padding_free", True):
            if not cfg.get("sample_packing"):
                raise ValueError(
                    "UlyssesRingAttentionPlugin Phase 1 requires sample_packing=true.\n\n"
                    "Add to your config:\n"
                    "  sample_packing: true\n\n"
                    "Or disable padding check (not recommended for Phase 1):\n"
                    "  ulysses_ring_attention_require_padding_free: false"
                )

        # === Validation: Compute sp/rp decomposition ===

        context_parallel_size = cfg["context_parallel_size"]
        mode = cfg.get("ulysses_ring_attention_mode", "auto")
        sp_override = cfg.get("ulysses_ring_attention_sp_size")
        rp_override = cfg.get("ulysses_ring_attention_rp_size")

        # Note: We can't access num_heads here (model not loaded yet)
        # Actual sp/rp computation happens in post_trainer_create
        # Here we just validate manual overrides if specified

        if sp_override is not None and rp_override is not None:
            if sp_override * rp_override != context_parallel_size:
                raise ValueError(
                    f"Manual sp_size={sp_override} and rp_size={rp_override} "
                    f"don't satisfy: sp × rp = context_parallel_size ({context_parallel_size}).\n"
                    f"Got: {sp_override} × {rp_override} = {sp_override * rp_override}"
                )

        # === Conflict Detection: Incompatible features ===

        # Check for potential conflicts with other plugins
        # (Add as needed when conflicts are discovered)

        LOG.info(
            f"UlyssesRingAttentionPlugin registered: "
            f"mode={mode}, cp={context_parallel_size}, "
            f"sp_override={sp_override}, rp_override={rp_override}"
        )

    def post_trainer_create(self, cfg: DictDefault, trainer: Trainer):
        """Setup process groups and patch attention after trainer creation.

        This is the main integration point where we:
        1. Compute sp/rp decomposition using model's num_heads
        2. Create SP and RP process groups
        3. Patch model's attention mechanism

        Args:
            cfg: Validated Axolotl configuration
            trainer: The HF Trainer instance

        Raises:
            ValueError: If decomposition fails or model incompatible
        """
        if not self.enabled:
            return

        LOG.info("UlyssesRingAttentionPlugin.post_trainer_create()")

        # === Step 1: Get num_heads from model ===

        model = trainer.model
        try:
            # Try to get config from unwrapped model
            model_config = getattr(model, "config", None)
            if model_config is None:
                # Handle wrapped models (FSDP, DDP, etc.)
                if hasattr(model, "module"):
                    model_config = model.module.config
                else:
                    raise AttributeError("Cannot access model.config")

            # Extract num_heads (handle different naming conventions)
            num_heads = getattr(model_config, "num_attention_heads", None)
            if num_heads is None:
                num_heads = getattr(model_config, "n_head", None)
            if num_heads is None:
                raise AttributeError("Cannot find num_attention_heads or n_head")

        except Exception as e:
            raise ValueError(
                f"Failed to extract num_heads from model: {e}\n\n"
                f"Model type: {type(model)}\n"
                f"This plugin requires access to model.config.num_attention_heads"
            ) from e

        LOG.info(f"Detected num_heads={num_heads} from model config")

        # === Step 2: Compute sp/rp decomposition ===

        context_parallel_size = cfg.context_parallel_size
        mode = cfg.get("ulysses_ring_attention_mode", "auto")
        sp_override = cfg.get("ulysses_ring_attention_sp_size")
        rp_override = cfg.get("ulysses_ring_attention_rp_size")

        try:
            self.sp_world_size, self.rp_world_size = compute_sp_rp(
                num_heads=num_heads,
                context_parallel_size=context_parallel_size,
                sp_size_override=sp_override,
                rp_size_override=rp_override,
                mode=mode,
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to compute sp/rp decomposition: {e}\n\n"
                f"Configuration:\n"
                f"  num_heads: {num_heads}\n"
                f"  context_parallel_size: {context_parallel_size}\n"
                f"  mode: {mode}\n"
                f"  sp_override: {sp_override}\n"
                f"  rp_override: {rp_override}\n\n"
                f"See: specs/006-ulysses-ring-attention-plugin/README.md"
            ) from e

        LOG.info(
            f"Decomposed context parallelism: sp={self.sp_world_size}, rp={self.rp_world_size} "
            f"(num_heads={num_heads}, cp={context_parallel_size})"
        )

        # === Step 3: Create process groups ===

        # Get device_mesh from trainer
        device_mesh = getattr(trainer.model, "device_mesh", None)
        if device_mesh is None:
            raise ValueError(
                "UlyssesRingAttentionPlugin requires device_mesh to be set on the model.\n\n"
                "This typically means context_parallel_size is configured but the model "
                "was not properly initialized with DeviceMesh.\n\n"
                "Ensure you're using Axolotl's distributed training setup with context parallelism."
            )

        LOG.info(f"Found device_mesh with dimensions: {device_mesh.mesh_dim_names}")

        # Extract context parallel submesh
        context_parallel_dim = "cp"  # Standard name used by Axolotl
        try:
            context_mesh = device_mesh[context_parallel_dim]
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Dimension '{context_parallel_dim}' not found in device_mesh. "
                f"Available dimensions: {device_mesh.mesh_dim_names}\n\n"
                f"This plugin requires context parallelism to be configured."
            ) from e

        # Get process group for context parallelism
        cp_group = context_mesh.get_group()
        cp_world_size = context_mesh.size()

        LOG.info(
            f"Extracted context parallel group: world_size={cp_world_size}, "
            f"mesh_shape={context_mesh.mesh.shape}"
        )

        # Verify consistency
        if cp_world_size != context_parallel_size:
            raise ValueError(
                f"Context parallel size mismatch: config={context_parallel_size}, "
                f"device_mesh={cp_world_size}"
            )

        # Create SP and RP process groups
        self.sp_group, self.rp_group, self.sp_rank, self.rp_rank = (
            create_ulysses_ring_groups(cp_group, self.sp_world_size, self.rp_world_size)
        )

        LOG.info(
            f"Created process groups: sp_group (size={self.sp_world_size}, rank={self.sp_rank}), "
            f"rp_group (size={self.rp_world_size}, rank={self.rp_rank})"
        )

        # === Step 4: Create DistributedAttention wrapper ===

        from axolotl.integrations.ulysses_ring_attn.patch import DistributedAttention

        self.distributed_attn = DistributedAttention(
            sp_group=self.sp_group,
            rp_group=self.rp_group,
            sp_size=self.sp_world_size,
            rp_size=self.rp_world_size,
            rp_rank=self.rp_rank,
            require_padding_free=cfg.get(
                "ulysses_ring_attention_require_padding_free", True
            ),
        )

        LOG.info("Created DistributedAttention wrapper")

        # === Step 4.5: Disable Accelerate's automatic CP context manager ===
        # Our plugin handles CP manually via attention patching, so we need to
        # disable Accelerate's automatic tensor sharding which conflicts with our approach
        from contextlib import nullcontext

        if hasattr(trainer, "accelerator") and trainer.accelerator is not None:
            # Replace maybe_context_parallel with a no-op context manager
            # This prevents Accelerate from automatically sharding inputs along sequence dimension
            trainer.accelerator.maybe_context_parallel = (
                lambda *args, **kwargs: nullcontext()
            )
            LOG.info(
                "Disabled Accelerate's automatic CP context manager "
                "(plugin handles CP via attention patching)"
            )

        # === Step 5: Monkey-patch attention ===

        # Detect model architecture
        model_type = getattr(model_config, "model_type", None)
        if model_type is None:
            raise ValueError(
                "Cannot detect model_type from model config. "
                "UlyssesRingAttentionPlugin requires model_type to be set."
            )

        LOG.info(f"Detected model_type: {model_type}")

        # Define supported model types and their corresponding patching functions
        # Phase 2.1: Llama-style models
        # Phase 2.4: GPT-NeoX, Falcon, BLOOM
        llama_model_types = {
            "llama",
            "mistral",
            "mixtral",
            "qwen2",
            "phi3",
            "gemma",
            "gemma2",
            "cohere",
        }

        gpt_neox_model_types = {"gpt_neox"}
        falcon_model_types = {"falcon", "RefinedWeb", "RefinedWebModel"}
        bloom_model_types = {"bloom"}

        all_supported_types = (
            llama_model_types
            | gpt_neox_model_types
            | falcon_model_types
            | bloom_model_types
        )

        if model_type not in all_supported_types:
            raise ValueError(
                f"UlyssesRingAttentionPlugin does not support model_type='{model_type}'. "
                f"Supported types: {sorted(all_supported_types)}\n\n"
                f"If you believe this model should be supported, please file an issue."
            )

        # Set global DistributedAttention instance for patched attention
        from axolotl.integrations.ulysses_ring_attn.patch import (
            patch_bloom_attention,
            patch_falcon_attention,
            patch_gpt_neox_attention,
            patch_llama_attention,
            set_distributed_attention,
        )

        set_distributed_attention(self.distributed_attn)

        # Apply attention monkey-patch based on model type
        if model_type in llama_model_types:
            patch_llama_attention()
            phase_info = "Phase 2.1: Llama-style"
        elif model_type in gpt_neox_model_types:
            patch_gpt_neox_attention()
            phase_info = "Phase 2.4: GPT-NeoX"
        elif model_type in falcon_model_types:
            patch_falcon_attention()
            phase_info = "Phase 2.4: Falcon"
        elif model_type in bloom_model_types:
            patch_bloom_attention()
            phase_info = "Phase 2.4: BLOOM"
        else:
            # Should never reach here due to earlier check
            raise RuntimeError(f"Unexpected model_type: {model_type}")

        LOG.info(
            f"UlyssesRingAttentionPlugin setup complete ({phase_info}): "
            f"Patched {model_type} attention with Ulysses + Ring-Attention"
        )
