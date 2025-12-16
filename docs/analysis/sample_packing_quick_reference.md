# Sample Packing 快速参考卡片 🚀

> 一页纸速查手册，快速上手 Sample Packing

---

## ⚡ 30 秒决策指南

```
需要启用 Sample Packing？看这里：

├─ 预训练？
│  └─ ✅ 必须启用 (收益最大，2-4倍加速)
│
├─ 指令微调/SFT？
│  └─ ✅ 强烈推荐 (2-3倍加速)
│
├─ 序列长度差异大？
│  └─ ✅ 启用 (利用率从30% → 95%)
│
├─ 序列长度均匀(>80%接近max_len)？
│  └─ ❌ 收益有限(<10%)，可不启用
│
└─ 超长序列训练(>16K)？
   └─ ⚠️ 评估收益，可能用CP更好
```

---

## ⚙️ 基本配置

### 最简配置（推荐新手）

```yaml
base_model: meta-llama/Llama-3.1-8B
sequence_len: 2048

# ✅ 启用Sample Packing
sample_packing: true

# ✅ 推荐：配合Flash Attention
flash_attention: true

# ✅ 推荐：不要pad到固定长度
pad_to_sequence_len: false

# 训练参数
micro_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 2e-5
```

### 完整配置（进阶）

```yaml
base_model: meta-llama/Llama-3.1-13B
sequence_len: 4096

# === Sample Packing核心配置 ===
sample_packing: true                    # 启用packing
sample_packing_eff_est: 0.95            # 预估效率(用于调整batch size)

# Packing模式
sample_packing_sequentially: false      # false=并行FFD(推荐), true=顺序

# Bin配置
sample_packing_bin_size: 200            # 每bin最多200个序列
sample_packing_group_size: 100000       # 每组处理100K序列

# Eval配置
eval_sample_packing: true               # Eval也启用packing

# === 必要配置 ===
flash_attention: true                   # ⚠️ 必须启用Flash Attention!
pad_to_sequence_len: false              # ⚠️ 关键：关闭固定padding

# === 优化配置 ===
gradient_checkpointing: true
bf16: true
```

---

## 📊 快速对比表

| 维度 | 无Sample Packing | 启用Sample Packing |
|------|-----------------|-------------------|
| **Token利用率** | 30-40% | 90-95% |
| **Throughput** | 基准 | +2-3倍 |
| **训练时间** | 基准 | -50-60% |
| **GPU利用率** | 40-50% | 80-90% |
| **内存使用** | 高 (大量padding) | 低 (少padding) |
| **收敛性** | 标准 | 相同 ✅ |
| **配置复杂度** | 简单 | 简单 (1-2行配置) |
| **成本** | 基准 | -50-60% 💰 |

---

## 🎯 常见场景配置

### 场景 1：预训练（收益最大）

```yaml
base_model: meta-llama/Llama-3.1-8B
sequence_len: 4096

pretraining_dataset: path/to/data

# Sample Packing
sample_packing: true
sample_packing_eff_est: 0.98  # 预训练通常效率更高
pretrain_multipack_attn: true  # 预训练专用

flash_attention: true
pad_to_sequence_len: false

# 大batch训练
micro_batch_size: 8
gradient_accumulation_steps: 16

# 预期收益: 3-4倍加速 🚀
```

### 场景 2：指令微调 (SFT)

```yaml
base_model: meta-llama/Llama-3.1-8B
sequence_len: 2048

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

# Sample Packing
sample_packing: true
sample_packing_eff_est: 0.95

flash_attention: true
pad_to_sequence_len: false

micro_batch_size: 4
gradient_accumulation_steps: 4

# 预期收益: 2-3倍加速
```

### 场景 3：DDP + Sample Packing

```yaml
base_model: meta-llama/Llama-3.1-8B
sequence_len: 2048

# Sample Packing
sample_packing: true
flash_attention: true

# DDP通过launcher自动启用
# torchrun --nproc_per_node=8 ...

micro_batch_size: 4
gradient_accumulation_steps: 4
# 有效batch: 4 × 8 (GPUs) × 4 = 128
```

### 场景 4：FSDP + Sample Packing

```yaml
base_model: meta-llama/Llama-3.1-13B
sequence_len: 2048

# FSDP-2
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true

# Sample Packing
sample_packing: true
sample_packing_eff_est: 0.95
flash_attention: true

micro_batch_size: 2
gradient_accumulation_steps: 8
```

### 场景 5：TP + DP + Sample Packing

```yaml
base_model: meta-llama/Llama-3.1-70B
sequence_len: 2048

# TP=2, DP=4 (8 GPUs total)
tensor_parallel_size: 2

# Sample Packing
sample_packing: true
sample_packing_eff_est: 0.95
flash_attention: true

micro_batch_size: 2  # 每DP rank
gradient_accumulation_steps: 4

# 预期收益: 2-3倍加速
```

### 场景 6：Curriculum Learning (顺序重要)

```yaml
base_model: meta-llama/Llama-3.1-8B
sequence_len: 2048

# Sample Packing with sequential mode
sample_packing: true
sample_packing_sequentially: true  # ← 保持数据顺序

flash_attention: true
pad_to_sequence_len: false

# 注意: 效率略低于并行模式，但保持顺序
```

---

## 🔧 参数详解

### 核心参数

```yaml
# sample_packing (bool)
sample_packing: true
# 是否启用Sample Packing
# 默认: false
# 推荐: true (除非序列长度非常均匀)

# sample_packing_eff_est (float, 0.0-1.0)
sample_packing_eff_est: 0.95
# 预估的打包效率，用于调整batch size
# 默认: 1.0 (不调整)
# 推荐: 0.90-0.95 (根据实际日志调整)
# 作用: 避免OOM (实际tokens比预期多)

# sample_packing_sequentially (bool)
sample_packing_sequentially: false
# false: 并行FFD，最高效
# true: 顺序packing，保持数据顺序
# 默认: false
# 推荐: false (除非需要保持顺序)

# sample_packing_bin_size (int)
sample_packing_bin_size: 200
# 每个bin最多容纳的序列数
# 默认: 200
# 推荐: 保持默认 (除非特殊需求)

# sample_packing_group_size (int)
sample_packing_group_size: 100000
# FFD分组大小 (并行处理)
# 默认: 100000
# 推荐: 保持默认

# eval_sample_packing (bool)
eval_sample_packing: true
# Evaluation时是否启用packing
# 默认: false
# 推荐: true (如果eval数据也长度不均)
```

### 关联参数

```yaml
# flash_attention (bool)
flash_attention: true
# ⚠️ Sample Packing几乎必须配合Flash Attention
# 推荐: 必须启用

# pad_to_sequence_len (bool)
pad_to_sequence_len: false
# ⚠️ Sample Packing时必须设为false
# true会破坏packing效果

# micro_batch_size (int)
micro_batch_size: 4
# 每个GPU的batch size (bins数量)
# Sample Packing时可能需要调小
# (因为实际tokens更多)
```

---

## 🐛 问题排查

### 问题 1：OOM (Out of Memory)

```
错误: CUDA out of memory
```

**原因**: Sample Packing提高了token利用率，实际计算量增加

**解决方案**:

```yaml
# 方案1: 减小micro_batch_size
micro_batch_size: 2  # 从4降到2

# 方案2: 降低效率估计
sample_packing_eff_est: 0.8  # 保守估计

# 方案3: 启用gradient checkpointing
gradient_checkpointing: true

# 方案4: 减小sequence_len
sequence_len: 1024  # 从2048降到1024
```

---

### 问题 2：打包效率低

```
[INFO] Sample packing efficiency: 0.65
```

**原因**: 序列长度分布不均或配置不当

**诊断**:

```python
# 检查数据集序列长度分布
from datasets import load_dataset
import matplotlib.pyplot as plt

ds = load_dataset('your_dataset')
lengths = [len(x['input_ids']) for x in ds['train']]

plt.hist(lengths, bins=50)
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.title('Sequence Length Distribution')
plt.show()

# 计算统计
import numpy as np
print(f"Mean: {np.mean(lengths):.0f}")
print(f"Std: {np.std(lengths):.0f}")
print(f"Min: {np.min(lengths)}")
print(f"Max: {np.max(lengths)}")
print(f"Median: {np.median(lengths):.0f}")
```

**解决方案**:

```yaml
# 方案1: 使用并行FFD (更高效)
sample_packing_sequentially: false

# 方案2: 增大bin_size
sample_packing_bin_size: 500

# 方案3: 调整sequence_len
# 如果大部分序列<1000，设置sequence_len=1024更合适
sequence_len: 1024  # 从2048降低

# 方案4: 过滤超长/超短序列
# 在数据预处理时过滤异常值
```

---

### 问题 3：训练不稳定

```
Loss出现NaN或震荡
```

**原因**: batch内token数量波动大

**解决方案**:

```yaml
# 方案1: 使用multipack_real_batches
multipack_real_batches: false
# false: 每bin算一个sample (默认，推荐)
# true: 每sequence算一个sample (更稳定但慢)

# 方案2: 降低学习率
learning_rate: 1e-5  # 从2e-5降低

# 方案3: 增加warmup
warmup_steps: 500  # 从100增加

# 方案4: 使用更保守的打包
sample_packing_eff_est: 0.85
```

---

### 问题 4：Eval时OOM

```
Train正常，Eval时OOM
```

**解决方案**:

```yaml
# 方案1: 关闭eval packing
eval_sample_packing: false

# 方案2: 减小eval batch size
eval_batch_size: 2  # 独立于micro_batch_size

# 方案3: 减小eval数据集
# 只eval部分数据
```

---

### 问题 5：模型不支持

```
ValueError: Model xxx does not support sample packing
```

**检查支持列表**:

```python
from axolotl.monkeypatch.multipack import SUPPORTED_MULTIPACK_MODEL_TYPES

print(SUPPORTED_MULTIPACK_MODEL_TYPES)
# ['llama', 'mistral', 'mixtral', 'qwen2', 'gemma', ...]
```

**解决方案**:

```yaml
# 如果模型不在列表中，尝试使用V2 collator
# (V2更广泛兼容)

# 或者不启用sample_packing
sample_packing: false
```

---

### 问题 6：效率没提升

```
启用Sample Packing后throughput没变化
```

**检查清单**:

```bash
# 1. 确认Sample Packing生效
# 查看训练日志，应该看到:
[INFO] Sample packing efficiency: 0.XXX
[INFO] MultipackBatchSampler: using parallel packing

# 2. 确认Flash Attention启用
# 日志中应该有:
[INFO] Using Flash Attention 2

# 3. 检查pad_to_sequence_len
# 必须是false

# 4. 检查序列长度分布
# 如果90%序列都接近sequence_len，收益有限

# 5. 监控GPU利用率
nvidia-smi dmon -s u
# 应该从40-50% → 80-90%+
```

---

## 💡 最佳实践

### ✅ 推荐做法

```yaml
# 1. 启用Sample Packing (新项目)
sample_packing: true

# 2. 配合Flash Attention
flash_attention: true

# 3. 关闭固定padding
pad_to_sequence_len: false

# 4. 使用并行FFD
sample_packing_sequentially: false

# 5. 设置合理的效率估计
sample_packing_eff_est: 0.95  # 根据日志调整

# 6. Eval也启用packing (如果数据长度不均)
eval_sample_packing: true

# 7. 启用gradient checkpointing节省内存
gradient_checkpointing: true

# 8. 使用bf16或fp16
bf16: true
```

### ❌ 避免

```yaml
# 1. Sample Packing + pad_to_sequence_len
sample_packing: true
pad_to_sequence_len: true  # ❌ 冲突！会破坏packing

# 2. Sample Packing without Flash Attention
sample_packing: true
flash_attention: false  # ❌ 效率低，不推荐

# 3. 过大的micro_batch_size
sample_packing: true
micro_batch_size: 16  # ❌ 容易OOM

# 4. 忘记设置效率估计
sample_packing: true
# sample_packing_eff_est: 0.95  ← 忘记设置，可能OOM

# 5. 序列长度均匀时强行使用
# 90%序列长度在1900-2048之间
sample_packing: true  # ❌ 收益<5%，不值得
```

---

## 📈 性能参考

### Llama-8B, 8×A100 40GB, DDP

| 配置 | Throughput (tokens/s/GPU) | 训练1B tokens时间 | GPU利用率 |
|------|--------------------------|------------------|----------|
| 无Packing | ~1800 | ~15.4h | 45% |
| + Sample Packing | ~4500 | ~6.2h | 85% |
| **提升** | **+2.5x** | **-60%** | **+89%** |

### Llama-13B, 8×A100 80GB, DDP

| 配置 | Throughput (tokens/s/GPU) | 训练1B tokens时间 | GPU利用率 |
|------|--------------------------|------------------|----------|
| 无Packing | ~1200 | ~23.1h | 42% |
| + Sample Packing | ~3000 | ~9.3h | 82% |
| **提升** | **+2.5x** | **-60%** | **+95%** |

### Llama-70B, 8×A100 80GB, TP=2, DP=4

| 配置 | Throughput (tokens/s/GPU) | 训练1B tokens时间 |
|------|--------------------------|------------------|
| 无Packing | ~600 | ~46h |
| + Sample Packing | ~1500 | ~18.5h |
| **提升** | **+2.5x** | **-60%** |

---

## 🛠️ 调试命令

### 检查Sample Packing是否生效

```bash
# 方法1: 查看训练日志
# 应该看到:
[INFO] Sample packing efficiency: 0.XXX
[INFO] MultipackBatchSampler: using parallel packing
```

```python
# 方法2: 在callback中检查batch shape
from transformers import TrainerCallback

class DebugCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            batch = kwargs.get('inputs', {})
            print(f"Step {state.global_step}:")
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  attention_mask unique: {batch['attention_mask'].unique()}")
            # Sample Packing enabled应该看到:
            # attention_mask unique: tensor([0, 1, 2, 3, ...])
            #                                ↑ 序列IDs
```

### 性能分析

```bash
# 对比 Sample Packing vs 无Sample Packing

# 测试1: 无Sample Packing
sample_packing: false
axolotl train config.yaml --max-steps 100
# 记录: Throughput, GPU利用率, 内存使用

# 测试2: Sample Packing
sample_packing: true
axolotl train config.yaml --max-steps 100
# 对比指标
```

### GPU监控

```bash
# 实时监控GPU利用率
watch -n 1 nvidia-smi

# 或使用dmon
nvidia-smi dmon -s u -d 1

# Sample Packing enabled后:
# GPU-Util应该从40-50% → 80-90%+
```

---

## 💬 快速 FAQ

**Q: Sample Packing会影响收敛吗？**
A: ✅ 不会。Attention mask确保序列隔离，梯度计算等价于非packing。

**Q: 所有模型都支持吗？**
A: ⚠️ 大部分主流模型支持。检查 `SUPPORTED_MULTIPACK_MODEL_TYPES`。

**Q: 必须配合Flash Attention吗？**
A: ✅ 强烈推荐。虽然V2 collator支持非Flash Attention，但效率会降低。

**Q: DeepSpeed兼容吗？**
A: ✅ 兼容。Axolotl会自动处理DeepSpeed配置。

**Q: 可以和FSDP/TP/CP一起用吗？**
A: ✅ DDP/FSDP/TP完美兼容。⚠️ CP需要注意，推荐分开使用。

**Q: 如何估算打包效率？**
A: 先不设置 `sample_packing_eff_est`，查看日志中的实际效率，然后设置该值。

**Q: Eval必须启用packing吗？**
A: ❌ 可选。如果eval数据长度也不均，建议启用；否则可关闭。

**Q: 会增加训练时间吗？**
A: ❌ 相反，会减少50-60%训练时间！

---

## 🔢 配置示例速查

### 最简配置 (1分钟上手)

```yaml
base_model: meta-llama/Llama-3.1-8B
sequence_len: 2048
sample_packing: true
flash_attention: true
pad_to_sequence_len: false
micro_batch_size: 4
```

### 生产环境配置

```yaml
base_model: meta-llama/Llama-3.1-13B
sequence_len: 2048

# Sample Packing
sample_packing: true
sample_packing_eff_est: 0.95
sample_packing_sequentially: false
eval_sample_packing: true

# 性能优化
flash_attention: true
gradient_checkpointing: true
bf16: true
pad_to_sequence_len: false

# FSDP-2
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true

# 训练参数
micro_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2e-5
lr_scheduler: cosine
warmup_steps: 100

# Logging
logging_steps: 10
eval_steps: 500
save_steps: 1000
```

### 大模型配置 (70B+)

```yaml
base_model: meta-llama/Llama-3.1-70B
sequence_len: 2048

# TP + DP
tensor_parallel_size: 2  # 8 GPUs → 4 TP groups

# Sample Packing
sample_packing: true
sample_packing_eff_est: 0.95
flash_attention: true
pad_to_sequence_len: false

# 内存优化
gradient_checkpointing: true
bf16: true

# 训练参数
micro_batch_size: 1  # TP后内存紧张
gradient_accumulation_steps: 16
```

---

## 📚 相关文档

- [详细解析](./sample_packing_deep_dive.md)
- [源码解析](./sample_packing_source_walkthrough.md)
- [Data Parallelism](./data_parallelism_deep_dive.md)
- [Tensor Parallelism](./tensor_parallelism_deep_dive.md)
- [Context Parallelism](./context_parallelism_deep_dive.md)
- [FSDP Versions](./fsdp_versions_comparison.md)
- [主索引](./README.md)

---

## 💡 速记口诀

```
Sample Packing 好处多，
减少 padding 效率高。
序列打包像拼图，
GPU 利用率飙升啦。

Flash Attention 必须配，
sequence_len 别固定。
效率估计要合理，
OOM 问题不用怕。

预训练收益最显著，
微调也能快两倍。
DDP FSDP 都兼容，
生产环境放心用！
```

---

*打印此页作为速查手册 | 最后更新：2025-11*
