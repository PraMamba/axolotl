# FSDP-1 vs FSDP-2 源码实现对比

> 深入源码理解两个 FSDP 版本在 Axolotl 中的实现差异

---

## 目录

1. [代码结构概览](#1-代码结构概览)
2. [配置解析差异](#2-配置解析差异)
3. [模型包装差异](#3-模型包装差异)
4. [Checkpoint 处理差异](#4-checkpoint-处理差异)
5. [关键代码路径对比](#5-关键代码路径对比)

---

## 1. 代码结构概览

### 1.1 关键文件

| 文件 | 功能 | FSDP-1 | FSDP-2 |
|------|------|--------|--------|
| `src/axolotl/utils/schemas/config.py` | 配置定义 | ✅ | ✅ |
| `src/axolotl/utils/schemas/validation.py` | 配置验证 | ✅ | ✅ |
| `src/axolotl/loaders/patch_manager.py` | Patch 管理 | ✅ | ✅ |
| `src/axolotl/monkeypatch/accelerate/fsdp2.py` | FSDP-2 专用 | - | ✅ |
| `src/axolotl/monkeypatch/accelerate/parallelism_config.py` | 并行配置 | - | ✅ |
| `src/axolotl/utils/trainer.py` | Trainer 配置 | ✅ | ✅ |

### 1.2 执行流程对比

#### FSDP-1 流程

```
用户配置 (fsdp_version: 1 或默认)
  ↓
src/axolotl/utils/schemas/config.py
  ├─ 解析 fsdp_config
  └─ fsdp_version 默认为 1
      ↓
src/axolotl/utils/schemas/validation.py
  ├─ 检查 FSDP-1 兼容性
  └─ 警告：建议升级到 FSDP-2
      ↓
src/axolotl/loaders/patch_manager.py
  └─ 不应用 FSDP-2 专用 patch
      ↓
Accelerate（HuggingFace）
  ├─ 使用传统 FSDP API
  └─ FullyShardedDataParallel 类包装
```

#### FSDP-2 流程

```
用户配置 (fsdp_version: 2)
  ↓
src/axolotl/utils/schemas/config.py
  ├─ 解析 fsdp_config
  └─ fsdp_version = 2
      ↓
src/axolotl/utils/schemas/validation.py
  ├─ 检查 FSDP-2 兼容性
  ├─ 验证：PyTorch >= 2.2
  └─ 验证：不支持量化+RL
      ↓
src/axolotl/loaders/patch_manager.py
  ├─ 应用 FSDP-2 专用 patch
  └─ patch_accelerate_fsdp2()
      ↓
src/axolotl/monkeypatch/accelerate/fsdp2.py
  ├─ 替换 Accelerate 的 FSDP 准备逻辑
  └─ 使用 fully_shard API
      ↓
src/axolotl/utils/trainer.py
  └─ 设置环境变量 FSDP_VERSION=2
      ↓
Accelerate（修改后）
  ├─ 使用新 FSDP API
  ├─ DTensor + DeviceMesh
  └─ fully_shard 函数包装
```

---

## 2. 配置解析差异

### 2.1 配置定义

**文件**: `src/axolotl/utils/schemas/config.py:687`

```python
class AxolotlInputConfig(BaseModel):
    """Axolotl 配置 Schema"""

    # FSDP 版本配置
    fsdp_version: int | None = Field(
        default=None,  # ← 默认为 None（等价于 1）
        description="FSDP version to use (1 or 2). Version 2 is recommended for better performance.",
    )

    # FSDP 详细配置
    fsdp_config: FSDPConfig | None = Field(
        default=None,
        description="Configuration for FSDP (Fully Sharded Data Parallelism)",
    )
```

### 2.2 配置验证

**文件**: `src/axolotl/utils/schemas/validation.py:799`

#### 验证 1：版本警告

```python
@model_validator(mode="before")
@classmethod
def check_fsdp_version(cls, data):
    """检查 FSDP 版本并给出警告"""
    fsdp_config = data.get("fsdp_config", {})

    if fsdp_config and str(data.get("fsdp_version")) != "2":
        # FSDP-1 警告
        LOG.info(
            "FSDP1 will be deprecated in an upcoming release of Axolotl. "
            "We recommend that you use FSDP version 2 for better performance and compatibility. "
            "Please see this link for more details: https://docs.axolotl.ai/docs/multi-gpu.html#sec-fsdp "
        )

    return data
```

#### 验证 2：FSDP-2 特有功能

```python
@model_validator(mode="before")
@classmethod
def check_fsdp2_cpu_offload_pin_memory(cls, data):
    """检查 cpu_offload_pin_memory（仅 FSDP-2 支持）"""
    if not (fsdp_config := data.get("fsdp_config")):
        return data

    if fsdp_config.get("cpu_offload_pin_memory") is False:
        # FSDP-1 不支持 disable pin_memory
        if str(data.get("fsdp_version")) != "2":
            raise ValueError(
                "FSDP1 does not support disabling cpu_offload_pin_memory, "
                "please set `fsdp_version` to 2"
            )

        # 必须同时启用 offload_params
        if not fsdp_config.get("offload_params"):
            raise ValueError(
                "disabling cpu_offload_pin_memory requires enabling offload_params"
            )

    return data
```

#### 验证 3：FSDP-2 限制

```python
@model_validator(mode="before")
@classmethod
def check_fsdp2_base_model_quant_rl(cls, data):
    """检查 FSDP-2 + 量化 + RL 的限制"""
    if data.get("fsdp_version") == 2 and data.get("rl") in [
        RLType.DPO,
        RLType.KTO,
        RLType.ORPO,
        RLType.IPO,
    ]:
        if data.get("load_in_8bit") or data.get("load_in_4bit"):
            # FSDP-2 不支持量化 + RL
            raise ValueError(
                f"FSDP2 does not support load_in_8bit or load_in_4bit with {data.get('rl')}. "
                f"Please use DeepSpeed or set `fsdp_version` to 1."
            )

    return data
```

---

## 3. 模型包装差异

### 3.1 Patch 应用逻辑

**文件**: `src/axolotl/loaders/patch_manager.py:117`

```python
def _apply_fsdp_patches(self):
    """应用 FSDP 相关的 patches"""

    # Context Parallelism 或 FSDP-2 需要 parallelism_config patch
    if self.cfg.context_parallel_size > 1 or (
        self.cfg.fsdp_config and str(self.cfg.fsdp_version) == "2"
    ):
        from axolotl.monkeypatch.accelerate.parallelism_config import (
            patch_parallelism_config,
        )
        patch_parallelism_config()

    # FSDP-2 专用 patch
    if self.cfg.fsdp_config and str(self.cfg.fsdp_version) == "2":
        from axolotl.monkeypatch.accelerate.fsdp2 import patch_accelerate_fsdp2

        # ← 关键：替换 Accelerate 的 FSDP 实现
        patch_accelerate_fsdp2()

        # 如果使用 RL，还需要 patch TRL
        if self.cfg.rl:
            from axolotl.monkeypatch.trainer.trl import patch_trl_prepare_fsdp2
            patch_trl_prepare_fsdp2()
```

### 3.2 FSDP-2 模型准备

**文件**: `src/axolotl/monkeypatch/accelerate/fsdp2.py:214`

```python
def fsdp2_prepare_model(accelerator, model: torch.nn.Module) -> torch.nn.Module:
    """
    为 FSDP-2 准备模型

    核心差异：
    - FSDP-1：使用 FullyShardedDataParallel 类包装
    - FSDP-2：使用 fully_shard 函数包装
    """
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,  # ← FSDP-2 核心 API
    )

    # 检查是否已经是 FSDP 模型
    is_type_fsdp = isinstance(model, FSDPModule)
    if is_type_fsdp:
        return model

    fsdp2_plugin = accelerator.state.fsdp_plugin

    # 保存原始 state_dict（用于后续加载）
    original_sd = model.state_dict()

    # 设置 auto_wrap_policy
    fsdp2_plugin.set_auto_wrap_policy(model)

    # 激活检查点（如果启用）
    if fsdp2_plugin.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            auto_wrap_policy=fsdp2_plugin.auto_wrap_policy,
        )

    # 获取 DeviceMesh（如果有）
    mesh = getattr(accelerator.state, "device_mesh", None)

    # 配置 CPU offload
    offload_to_cpu = isinstance(fsdp2_plugin.cpu_offload, CPUOffloadPolicy)
    if offload_to_cpu and os.environ.get("FSDP_CPU_OFFLOAD_PIN_MEMORY", "") == "false":
        fsdp2_plugin.cpu_offload.pin_memory = False  # ← FSDP-2 特有

    # FSDP-2 参数
    fsdp2_kwargs = {
        "reshard_after_forward": fsdp2_plugin.reshard_after_forward,
        "offload_policy": fsdp2_plugin.cpu_offload,
        "mp_policy": fsdp2_plugin.mixed_precision_policy or MixedPrecisionPolicy(),
        "mesh": (
            mesh[tuple(accelerator.state.parallelism_config.fsdp_dim_names)]
            if mesh is not None
            else None
        ),  # ← FSDP-2 使用 DeviceMesh
    }

    # CPU RAM 高效加载
    if fsdp2_plugin.cpu_ram_efficient_loading:
        # 移动到 meta device
        model = model.to(torch.device("meta"))
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # 应用 auto_wrap_policy
    auto_wrap_policy = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)

    if auto_wrap_policy is not None:
        for module in get_module_children_bottom_up(model)[:-1]:
            # 特殊处理 LoRA 层
            if isinstance(module, LoraLayer):
                _process_lora_module_for_fsdp(module, fsdp2_kwargs)

            # 应用 fully_shard
            if auto_wrap_policy(module) and not isinstance(module, FSDPModule):
                fully_shard(module, **fsdp2_kwargs)  # ← FSDP-2 核心

    # 包装整个模型
    fully_shard(model, **fsdp2_kwargs)  # ← FSDP-2 核心

    # 加载原始参数
    if fsdp2_plugin.cpu_ram_efficient_loading:
        fsdp2_load_full_state_dict(
            accelerator, model, original_sd, offload_to_cpu=offload_to_cpu
        )

        # 重新 tie weights
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    return model
```

### 3.3 FSDP-1 vs FSDP-2 API 对比

#### FSDP-1（PyTorch 原生）

```python
# Accelerate 内部（FSDP-1）

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 包装模型
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=True),
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision,
    # ... 更多参数
)
```

#### FSDP-2（Axolotl patch）

```python
# Axolotl monkeypatch（FSDP-2）

from torch.distributed.fsdp import fully_shard

# 函数式包装
fully_shard(
    model,
    mesh=device_mesh,  # ← 使用 DeviceMesh
    reshard_after_forward=True,
    offload_policy=offload_policy,
    mp_policy=mp_policy,
)
```

---

## 4. Checkpoint 处理差异

### 4.1 FSDP-2 State Dict 加载

**文件**: `src/axolotl/monkeypatch/accelerate/fsdp2.py:20`

```python
def fsdp2_load_full_state_dict(
    _accelerator, model: torch.nn.Module, full_sd: dict, offload_to_cpu: bool = False
):
    """
    加载完整 state dict 到分片模型

    FSDP-2 特点：
    - 使用 DTensor 的 distribute_tensor
    - 自动处理参数分发
    """
    from torch.distributed.tensor import distribute_tensor

    LOG.info("Broadcasting full state dict to all ranks...")

    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

    for param_name, sharded_meta_param in meta_sharded_sd.items():
        full_tensor = None
        if _accelerator.is_main_process:
            full_tensor = full_sd[param_name]
            full_tensor = full_tensor.to(sharded_meta_param.dtype)

        # 检查是否是 DTensor（FSDP-2 特有）
        if hasattr(sharded_meta_param, "device_mesh"):
            device_mesh = sharded_meta_param.device_mesh

            if not _accelerator.is_main_process:
                full_tensor = torch.empty(
                    sharded_meta_param.size(),
                    device=device_mesh.device_type,
                    dtype=sharded_meta_param.dtype,
                )

            # DTensor 自动分发
            sharded_param = distribute_tensor(
                full_tensor,
                device_mesh,
                sharded_meta_param.placements,  # ← 分片策略
                src_data_rank=0,
            )
        else:
            # 非分片参数（广播）
            if _accelerator.is_main_process:
                sharded_param = full_tensor.to(torch.device("cuda"))
            else:
                sharded_param = torch.empty_like(sharded_meta_param)

            dist.broadcast(sharded_param, src=0)

        if offload_to_cpu:
            sharded_param = sharded_param.cpu()

        sharded_sd[param_name] = nn.Parameter(sharded_param)

        # 释放内存
        del full_tensor
        full_sd[param_name] = None

    # 加载到模型
    model.load_state_dict(sharded_sd, assign=True, strict=True)

    return model
```

### 4.2 FSDP-2 State Dict 保存

**文件**: `src/axolotl/monkeypatch/accelerate/fsdp2.py:151`

```python
def get_state_dict(self, model, unwrap=True):
    """
    获取模型 state dict

    FSDP-2 分支：使用 DTensor 的 full_tensor()
    """
    # ... 其他分布式类型的处理 ...

    elif self.is_fsdp2:  # ← FSDP-2 分支
        state_dict = {}
        sharded_state_dict = model.state_dict()

        for param_name, param in sharded_state_dict.items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))

            # DTensor 自动 AllGather
            param = param.full_tensor()  # ← FSDP-2 特有

            # 只在 rank 0 保存
            if torch.distributed.get_rank() == 0:
                state_dict[param_name] = param.cpu()

            torch.distributed.barrier()

    elif self.distributed_type == DistributedType.FSDP:  # ← FSDP-1 分支
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        # FSDP-1 使用 context manager
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )

        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            state_dict = model.state_dict()

    # ...

    return state_dict
```

---

## 5. 关键代码路径对比

### 5.1 配置到模型包装流程

#### FSDP-1 路径

```
1. 用户配置
   config.yaml
   ├─ fsdp_version: 1 (或不写)
   └─ fsdp_config: {...}

2. 配置解析
   src/axolotl/utils/schemas/config.py
   └─ 解析为 AxolotlInputConfig

3. 配置验证
   src/axolotl/utils/schemas/validation.py
   ├─ check_fsdp_version() → 警告：建议用 FSDP-2
   └─ 通过验证

4. Patch 应用
   src/axolotl/loaders/patch_manager.py
   └─ _apply_fsdp_patches()
       └─ 不应用 FSDP-2 专用 patch

5. 模型加载
   src/axolotl/loaders/model.py
   └─ ModelLoader.load()
       └─ 使用标准 Accelerate

6. Accelerate 准备
   HuggingFace Accelerate（未修改）
   └─ FullyShardedDataParallel 类包装
```

#### FSDP-2 路径

```
1. 用户配置
   config.yaml
   ├─ fsdp_version: 2
   └─ fsdp_config: {...}

2. 配置解析
   src/axolotl/utils/schemas/config.py
   └─ 解析为 AxolotlInputConfig

3. 配置验证
   src/axolotl/utils/schemas/validation.py
   ├─ check_fsdp_version() → 不警告
   ├─ check_fsdp2_cpu_offload_pin_memory()
   └─ check_fsdp2_base_model_quant_rl()

4. Patch 应用
   src/axolotl/loaders/patch_manager.py
   └─ _apply_fsdp_patches()
       ├─ patch_parallelism_config()  # ← DeviceMesh 支持
       └─ patch_accelerate_fsdp2()    # ← 替换 Accelerate 逻辑

5. Patch 生效
   src/axolotl/monkeypatch/accelerate/fsdp2.py
   └─ patch_accelerate_fsdp2()
       ├─ 替换 accelerator.fsdp2_prepare_model
       └─ 替换 Accelerator.get_state_dict

6. 环境变量设置
   src/axolotl/utils/trainer.py
   └─ os.environ["FSDP_VERSION"] = "2"

7. 模型加载
   src/axolotl/loaders/model.py
   └─ ModelLoader.load()

8. Accelerate 准备（已修改）
   HuggingFace Accelerate（被 patch）
   └─ fsdp2_prepare_model()
       └─ fully_shard 函数包装 + DTensor
```

### 5.2 关键差异点代码位置

| 差异点 | FSDP-1 | FSDP-2 | 文件 |
|--------|--------|--------|------|
| **配置警告** | 有警告 | 无警告 | `validation.py:799` |
| **Patch 应用** | 不应用 | 应用 | `patch_manager.py:127` |
| **模型包装 API** | `FSDP(model)` | `fully_shard(model)` | `fsdp2.py:342` |
| **DeviceMesh** | 不使用 | 使用 | `fsdp2.py:279-295` |
| **DTensor** | 不使用 | 使用 | `fsdp2.py:57-62` |
| **State Dict 加载** | Context Manager | DTensor API | `fsdp2.py:20-90` |
| **State Dict 保存** | Context Manager | `full_tensor()` | `fsdp2.py:151-183` |
| **环境变量** | 无 | `FSDP_VERSION=2` | `trainer.py:594` |

### 5.3 调试入口点

#### 检查 FSDP 版本

```python
# 在训练脚本中添加

import os
print(f"FSDP_VERSION: {os.environ.get('FSDP_VERSION', 'Not set')}")

# FSDP-1: 输出 "Not set"
# FSDP-2: 输出 "2"
```

#### 检查是否应用了 Patch

```python
# 检查 Accelerate 是否被 patch

import accelerate

# FSDP-2 会有这个属性
if hasattr(accelerate.accelerator, 'fsdp2_prepare_model'):
    print("✅ FSDP-2 patch applied")
else:
    print("❌ FSDP-2 patch not applied (using FSDP-1)")
```

#### 检查模型类型

```python
# 检查模型是否是 FSDP-2

from torch.distributed.fsdp import FSDPModule

if isinstance(model, FSDPModule):
    print("✅ Using FSDP-2 (fully_shard)")
else:
    print("ℹ️ Using FSDP-1 (FullyShardedDataParallel)")
```

---

## 总结

### 核心实现差异

1. **API 层面**
   - FSDP-1：`FullyShardedDataParallel` 类
   - FSDP-2：`fully_shard` 函数

2. **底层技术**
   - FSDP-1：手动分片逻辑
   - FSDP-2：DTensor + DeviceMesh

3. **代码组织**
   - FSDP-1：使用原生 Accelerate
   - FSDP-2：Monkeypatch Accelerate

4. **关键文件**
   - 配置：`config.py`, `validation.py`
   - Patch：`patch_manager.py`, `fsdp2.py`
   - 环境：`trainer.py`

5. **调试要点**
   - 检查 `FSDP_VERSION` 环境变量
   - 检查是否应用了 fsdp2 patch
   - 检查模型类型（`FSDPModule`）

---

## 下一步

- 快速参考卡片 → [fsdp_versions_quick_reference.md](./fsdp_versions_quick_reference.md)
- 返回对比分析 → [fsdp_versions_comparison.md](./fsdp_versions_comparison.md)
- 返回主索引 → [README.md](./README.md)

---

*本文档由 Claude AI 辅助创作，旨在帮助开发者理解 FSDP-1 和 FSDP-2 的源码实现差异。*
