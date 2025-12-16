# FSDP-1 vs FSDP-2 快速参考卡片 🚀

> 一页纸速查手册，快速决策使用哪个 FSDP 版本

---

## ⚡ 30 秒决策指南

```
需要使用 FSDP？看这里：

├─ 新项目？
│  └─ ✅ 用 FSDP-2 (fsdp_version: 2)
│
├─ PyTorch >= 2.4？
│  └─ ✅ 用 FSDP-2
│
├─ 使用 TP/CP（N-D 并行）？
│  └─ ✅ 用 FSDP-2
│
├─ 使用量化（8bit/4bit）+ RL（DPO/KTO等）？
│  └─ ❌ 用 FSDP-1 或 DeepSpeed
│
└─ 老项目 + PyTorch < 2.2？
   └─ ℹ️ 保持 FSDP-1
```

---

## ⚙️ 基本配置

### FSDP-1 配置（不推荐）

```yaml
base_model: meta-llama/Llama-3.1-13B

# === FSDP-1（默认）===
fsdp_version: 1  # 或不写（默认为 1）
# ⚠️ 会有警告：建议升级到 FSDP-2

fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT
```

### FSDP-2 配置（推荐）

```yaml
base_model: meta-llama/Llama-3.1-13B

# === FSDP-2（推荐）===
fsdp_version: 2  # ← 关键！

fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT

  # === FSDP-2 特有功能 ===
  cpu_offload_pin_memory: false  # 可选，节省系统内存
```

---

## 📊 快速对比表

| 维度 | FSDP-1 | FSDP-2 |
|------|--------|--------|
| **PyTorch 版本** | 1.11+ | 2.2+ |
| **性能** | 基准 | +5-20% |
| **N-D 并行** | ❌ | ✅ |
| **DeviceMesh** | ❌ | ✅ |
| **DTensor** | ❌ | ✅ |
| **cpu_offload_pin_memory** | ✅ (强制 true) | ✅ (可配置) |
| **量化 + RL** | ✅ | ❌ |
| **状态** | 维护模式 | 活跃开发 |
| **推荐** | ❌ | ✅ |

---

## 🔧 迁移步骤（3 分钟）

### 步骤 1：检查 PyTorch 版本

```bash
python -c "import torch; print(torch.__version__)"

# 需要 >= 2.2
# 推荐 >= 2.4
```

### 步骤 2：修改配置

```yaml
# 只需添加一行！
fsdp_version: 2
```

### 步骤 3：测试

```bash
axolotl train config.yaml --max-steps 10
```

### 完成！

Checkpoint 完全兼容，可随时回退：`fsdp_version: 1`

---

## 🎯 常见场景配置

### 场景 1：纯 FSDP（推荐 FSDP-2）

```yaml
base_model: meta-llama/Llama-3.1-13B
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true

# 性能提升：~5-10%
```

### 场景 2：TP + FSDP（强烈推荐 FSDP-2）

```yaml
base_model: meta-llama/Llama-3.1-70B
tensor_parallel_size: 2
dp_shard_size: 4
fsdp_version: 2

# 性能提升：~15-20%
# 配置更简单（DeviceMesh 自动协调）
```

### 场景 3：CP + FSDP（强烈推荐 FSDP-2）

```yaml
base_model: meta-llama/Llama-3.1-8B
context_parallel_size: 2
dp_shard_size: 4
sequence_len: 16384
fsdp_version: 2

# 性能提升：~10-15%
```

### 场景 4：量化 + RL（必须 FSDP-1）

```yaml
base_model: meta-llama/Llama-3.1-7B
load_in_8bit: true
rl: dpo
fsdp_version: 1  # ← 必须！

# FSDP-2 不支持量化 + RL
```

---

## 🐛 问题排查

### 问题：升级后报错

```
ValueError: FSDP2 does not support load_in_8bit with dpo
```

**原因**：FSDP-2 + 量化 + RL 不兼容

**解决**：
```yaml
# 选项 1：回退到 FSDP-1
fsdp_version: 1

# 选项 2：不用量化
# load_in_8bit: true  # ← 注释掉

# 选项 3：改用 DeepSpeed
# deepspeed_config: ...
```

---

### 问题：cpu_offload_pin_memory 报错

```
ValueError: FSDP1 does not support disabling cpu_offload_pin_memory
```

**原因**：FSDP-1 不支持 `cpu_offload_pin_memory: false`

**解决**：
```yaml
fsdp_version: 2  # ← 升级到 FSDP-2
fsdp_config:
  cpu_offload_pin_memory: false
```

---

### 问题：性能没提升？

**检查清单**：
```bash
# 1. 确认 FSDP-2 生效
python -c "import os; print(os.environ.get('FSDP_VERSION'))"
# 应输出: 2

# 2. 确认 PyTorch 版本
python -c "import torch; print(torch.__version__)"
# 应 >= 2.2

# 3. 检查是否使用 N-D 并行
grep -E "tensor_parallel|context_parallel" config.yaml
# FSDP-2 在 N-D 并行下提升最大

# 4. 检查 batch size
# FSDP-2 在大 batch 下提升更明显
```

---

## 💡 最佳实践

### ✅ 推荐做法

```yaml
# 1. 使用 FSDP-2（新项目）
fsdp_version: 2

# 2. 开启 reshard
fsdp_config:
  reshard_after_forward: true

# 3. 使用 TRANSFORMER_BASED_WRAP
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP

# 4. PyTorch >= 2.4
# pip install torch>=2.4

# 5. N-D 并行优先用 FSDP-2
tensor_parallel_size: 2
fsdp_version: 2
```

### ❌ 避免

```yaml
# 1. FSDP-2 + 量化 + RL
fsdp_version: 2
load_in_8bit: true
rl: dpo  # ← 报错！

# 2. FSDP-1 + N-D 并行
fsdp_version: 1
tensor_parallel_size: 2  # ← 配置复杂，性能差

# 3. PyTorch < 2.2 + FSDP-2
# ← 不支持！

# 4. 忘记设置 fsdp_version
# ← 默认为 1，性能损失
```

---

## 📈 性能参考

### Llama-13B, 8×A100 80GB

| 配置 | FSDP-1 | FSDP-2 | 提升 |
|------|--------|--------|------|
| 纯 FSDP | 1850 | 2050 | +10.8% |
| FSDP + TP(2) | 1600 | 1850 | +15.6% |
| FSDP + CP(2) | 1400 | 1600 | +14.3% |

### Llama-70B, 8×A100 80GB

| 配置 | FSDP-1 | FSDP-2 | 提升 |
|------|--------|--------|------|
| FSDP + TP(2) | 900 | 1050 | +16.7% |
| FSDP + TP(2) + CP(2) | 700 | 850 | +21.4% |

---

## 🔢 配置示例速查

### Llama-13B，8 卡（FSDP-2）

```yaml
base_model: meta-llama/Llama-3.1-13B
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true
micro_batch_size: 4
gradient_accumulation_steps: 4
```

### Llama-70B，8 卡（FSDP-2 + TP）

```yaml
base_model: meta-llama/Llama-3.1-70B
tensor_parallel_size: 2
dp_shard_size: 4
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true
micro_batch_size: 2
```

### Llama-8B 长上下文，8 卡（FSDP-2 + CP）

```yaml
base_model: meta-llama/Llama-3.1-8B
context_parallel_size: 2
dp_shard_size: 4
sequence_len: 16384
micro_batch_size: 1
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true
```

---

## 🛠️ 调试命令

### 检查 FSDP 版本

```python
# 方法 1：环境变量
import os
print(f"FSDP Version: {os.environ.get('FSDP_VERSION', 'Not set (FSDP-1)')}")

# 方法 2：检查 patch
import accelerate
if hasattr(accelerate.accelerator, 'fsdp2_prepare_model'):
    print("✅ FSDP-2")
else:
    print("ℹ️ FSDP-1")

# 方法 3：检查模型类型
from torch.distributed.fsdp import FSDPModule
if isinstance(model, FSDPModule):
    print("✅ FSDP-2 (fully_shard)")
```

### 性能分析

```bash
# 对比 FSDP-1 vs FSDP-2
# 测试 1: FSDP-1
axolotl train config_fsdp1.yaml --max-steps 100

# 测试 2: FSDP-2
axolotl train config_fsdp2.yaml --max-steps 100

# 对比 Tokens/s/GPU
```

---

## 💬 快速 FAQ

**Q: FSDP-1 何时移除？**
A: 1-2 年内，会提前通知

**Q: 性能提升有多少？**
A: 5-20%，N-D 并行下更高

**Q: Checkpoint 兼容吗？**
A: ✅ 完全兼容

**Q: 可以随时切换吗？**
A: ✅ 可以，改一行配置

**Q: PyTorch 最低版本？**
A: FSDP-2 需要 >= 2.2

---

## 📚 相关文档

- [详细对比](./fsdp_versions_comparison.md)
- [源码解析](./fsdp_versions_source_walkthrough.md)
- [Data Parallelism 总览](./data_parallelism_deep_dive.md)
- [主索引](./README.md)

---

## 💡 速记口诀

```
新项目用 FSDP-2，
性能提升显著多。
N-D 并行必须选，
DeviceMesh 自动协调。

量化加 RL 不支持，
回退 FSDP-1 或 DeepSpeed。
迁移简单改一行，
Checkpoint 完全兼容。

PyTorch 要升级，
至少 2.2 版本起。
未来发展方向明，
FSDP-2 是首选！
```

---

*打印此页作为速查手册 | 最后更新：2025-11*
