# Context Parallelism 快速参考卡片 🚀

> 一页纸速查手册，适合快速查阅 CP 配置和命令

---

## ⚙️ 基本配置

### 最小化 CP 配置
```yaml
base_model: meta-llama/Llama-3.1-8B
context_parallel_size: 2  # 仅此一行！

# CP 要求配置
micro_batch_size: 1  # 必须为 1
sequence_len: 8192   # 长上下文

# 其他必需配置
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
output_dir: ./outputs/cp-test/
bf16: true
flash_attention: true
```

### 推荐的完整配置
```yaml
base_model: meta-llama/Llama-3.1-8B

# === 并行配置 ===
dp_shard_size: 2           # FSDP
tensor_parallel_size: 2    # TP
context_parallel_size: 2   # CP
# 总计：2 × 2 × 2 = 8 GPUs

# === FSDP 配置 ===
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer

# === 长上下文配置 ===
sequence_len: 16384        # 16K 上下文
micro_batch_size: 1        # CP 强制要求
gradient_accumulation_steps: 16  # 补偿小 batch

# === 训练配置 ===
num_epochs: 1
optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 2e-5

# === 性能优化 ===
bf16: true
tf32: true
flash_attention: true      # 必需！
gradient_checkpointing: true

# === 输出 ===
output_dir: ./outputs/cp-long-context/
logging_steps: 1
```

---

## 🎯 常用场景配置

### 场景 1：单节点 8 卡，16K 上下文
```yaml
# 选项 A：只使用 CP（适合小模型）
dp_shard_size: 4
context_parallel_size: 2
# 4 × 2 = 8 GPUs

# 选项 B：TP + CP 组合（适合大模型）
dp_shard_size: 2
tensor_parallel_size: 2
context_parallel_size: 2  # ← 推荐
# 2 × 2 × 2 = 8 GPUs

sequence_len: 16384
micro_batch_size: 1
```

### 场景 2：单节点 8 卡，32K 超长上下文
```yaml
# 更激进的 CP 切分
dp_shard_size: 2
context_parallel_size: 4
# 2 × 4 = 8 GPUs

sequence_len: 32768
micro_batch_size: 1
gradient_accumulation_steps: 32  # 增大以补偿
```

### 场景 3：双节点 16 卡，64K 极长上下文
```yaml
# 4D 并行（FSDP + DDP + TP + CP）
dp_shard_size: 2           # 节点内 FSDP
dp_replicate_size: 2       # 节点间 DDP
tensor_parallel_size: 2    # 模型并行
context_parallel_size: 2   # 序列并行
# 2 × 2 × 2 × 2 = 16 GPUs

sequence_len: 65536
micro_batch_size: 1
```

### 场景 4：测试 CP（最小配置）
```yaml
# 2 卡快速测试
dp_shard_size: 1
context_parallel_size: 2
sequence_len: 4096
micro_batch_size: 1
max_steps: 10  # 只跑 10 步测试
```

---

## 🚀 运行命令

### 基本命令
```bash
# 单节点训练
axolotl train config.yaml

# 指定 GPU 数量（CP 需要精确匹配配置）
axolotl train config.yaml --num-processes 8

# 使用 torchrun（推荐）
axolotl train config.yaml --launcher torchrun
```

### 多节点训练
```bash
# === Node 0 (master) ===
axolotl train config.yaml \
    --num-processes 16 \
    --num-machines 2 \
    --machine-rank 0 \
    --main-process-ip <NODE0_IP> \
    --main-process-port 29500

# === Node 1 ===
axolotl train config.yaml \
    --num-processes 16 \
    --num-machines 2 \
    --machine-rank 1 \
    --main-process-ip <NODE0_IP> \
    --main-process-port 29500
```

### 调试命令
```bash
# 测试 Ring 通信
NCCL_DEBUG=INFO axolotl train config.yaml --max-steps 2

# 检查序列切分是否正确
# 在配置中添加：
# logging_steps: 1
# 观察日志中的 sequence length

# 验证显存节省
nvidia-smi dmon -s mu -c 10
```

---

## 🔍 调试速查

### 问题：显存仍然 OOM（即使开启 CP）

#### 检查清单
```bash
✓ 确认 micro_batch_size = 1
✓ 确认 flash_attention = true
✓ 确认 sequence_len 能被 context_parallel_size 整除
✓ 开启 gradient_checkpointing
✓ 增大 context_parallel_size
```

#### 配置调整
```yaml
# 之前
context_parallel_size: 2
sequence_len: 16384
gradient_accumulation_steps: 8

# 之后
context_parallel_size: 4    # 增大 CP
sequence_len: 16384
gradient_accumulation_steps: 32  # 增大以补偿
# 或降低序列长度
sequence_len: 8192         # 减半
```

---

### 问题：训练速度极慢

#### 检查 GPU 互连
```bash
nvidia-smi topo -m

# ✅ 好（NVLink）:
#   GPU0  GPU1  GPU2  GPU3
# 0   X    NV12  NV12  NV12
# 1  NV12   X    NV12  NV12
# 2  NV12  NV12   X    NV12
# 3  NV12  NV12  NV12   X

# ❌ 差（PCIe）:
#   GPU0  GPU1
# 0   X    PHB
# 1  PHB   X
```

**CP 对通信带宽极度敏感！**
- Ring-Flash-Attention 需要每个 GPU 与相邻 GPU 频繁通信
- PCIe 带宽可能导致 10-100× 性能下降
- **强烈建议**：同一 CP 组的 GPU 必须在同一节点且有 NVLink

#### 性能优化配置
```yaml
# Flash Attention（必需）
flash_attention: true

# 混合精度
bf16: true
tf32: true

# Fused 算子
optimizer: adamw_torch_fused

# 梯度检查点
gradient_checkpointing: true

# 数据加载优化
dataloader_num_workers: 4
dataloader_pin_memory: true
```

---

### 问题：Loss NaN 或不稳定

#### 配置调整
```yaml
# 使用 bf16（比 fp16 更稳定）
bf16: true
fp16: false

# 梯度裁剪（CP 尤其重要）
max_grad_norm: 1.0

# 降低学习率
learning_rate: 1e-5  # CP 可能需要更小的 LR

# Warmup（让模型适应 Ring-Attention）
warmup_steps: 100
warmup_ratio: 0.1

# 检查序列长度是否过长
sequence_len: 8192  # 先从较短序列开始测试
```

---

### 问题：序列切分错误

#### 诊断
```python
# 在训练脚本中添加：
print(f"原始序列长度: {input_ids.shape[1]}")
print(f"CP size: {context_parallel_size}")
print(f"每个 GPU 序列长度: {input_ids.shape[1] // context_parallel_size}")
```

#### 要求
```yaml
# sequence_len 必须能被 context_parallel_size 整除
sequence_len: 16384
context_parallel_size: 4  # 16384 / 4 = 4096 ✅

# 错误示例：
sequence_len: 10000
context_parallel_size: 3  # 10000 / 3 = 3333.33... ❌
```

---

## 📊 性能对比表

### 序列长度 → CP 配置映射

| 序列长度 | 单节点 8 卡 | 显存节省 | 备注 |
|---------|------------|---------|------|
| 2K-4K | CP=1 (不需要) | - | 常规 FSDP 即可 |
| 8K | CP=2, TP=2, FSDP=2 | ~30% | 推荐起点 |
| 16K | CP=2, TP=2, FSDP=2 | ~50% | 常用配置 |
| 32K | CP=4, TP=2, FSDP=1 | ~70% | 需要 NVLink |
| 64K+ | CP=4, TP=2 (多节点) | ~75% | 需要多节点 |

### Llama-8B 长上下文性能参考 (8×A100 80GB)

| Seq Len | CP Size | 显存/GPU | Tokens/s/GPU | 通信开销 |
|---------|---------|---------|--------------|---------|
| 4K | 1 (纯FSDP) | ~35GB | 3000 | 0% |
| 8K | 2 | ~25GB | 2400 | ~20% |
| 16K | 2 | ~30GB | 2000 | ~33% |
| 32K | 4 | ~25GB | 1200 | ~60% |

**注意**：通信开销随 CP size 线性增长

---

## 🛠️ 常用代码片段

### 检查 CP 是否生效
```python
# 在训练前添加
import torch.distributed as dist

if dist.is_initialized():
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 检查序列切分
    for batch in train_dataloader:
        local_seq_len = batch['input_ids'].shape[1]
        print(f"Rank {rank}: 本地序列长度 = {local_seq_len}")

        # 验证总长度
        expected_total = local_seq_len * world_size  # 如果只有 CP
        print(f"预期总序列长度 = {expected_total}")
        break  # 只检查第一个 batch
```

### 监控 Ring 通信
```python
# 添加 NCCL 日志
import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'COLL'

# 训练时观察日志，应该看到类似：
# NCCL INFO Ring 00 : 0 -> 1 -> 2 -> 3 -> 0
```

### 计算有效 Batch Size (CP 场景)
```python
# CP 下的有效 batch size 计算
effective_batch_size = (
    micro_batch_size *              # 必须为 1
    gradient_accumulation_steps *   # 补偿小 batch
    dp_shard_size *                 # FSDP 并行度
    dp_replicate_size               # DDP 并行度（如果有）
)

# 例如：
# micro_batch_size = 1
# gradient_accumulation_steps = 32
# dp_shard_size = 4
# dp_replicate_size = 1
# effective_batch_size = 1 × 32 × 4 × 1 = 128

# 注意：CP 不参与 effective batch size 计算！
# 因为 CP 是序列并行，不是数据并行
```

### 检查序列对齐
```python
# 验证序列长度是否正确切分
def check_sequence_alignment(sequence_len, cp_size):
    if sequence_len % cp_size != 0:
        raise ValueError(
            f"序列长度 {sequence_len} 不能被 CP size {cp_size} 整除！"
            f"建议调整为 {(sequence_len // cp_size + 1) * cp_size}"
        )
    chunk_size = sequence_len // cp_size
    print(f"✅ 每个 GPU 处理 {chunk_size} tokens")
    return chunk_size

# 使用
check_sequence_alignment(16384, 4)  # ✅ 输出：每个 GPU 处理 4096 tokens
check_sequence_alignment(10000, 3)  # ❌ 报错
```

---

## ⚡ 性能调优检查清单

### 必做优化 ✅
- [ ] 启用 Flash Attention (`flash_attention: true`) **← CP 强制要求**
- [ ] 使用 bf16 (`bf16: true`)
- [ ] 设置 `micro_batch_size: 1` **← CP 强制要求**
- [ ] 验证序列长度可整除 (`sequence_len % cp_size == 0`)
- [ ] 启用梯度检查点 (`gradient_checkpointing: true`)

### 通信优化 🚀
- [ ] 确保 CP 组内 GPU 有 NVLink (`nvidia-smi topo -m`)
- [ ] CP 组内 GPU 在同一节点（避免跨节点 Ring）
- [ ] 启用 NCCL 优化 (`NCCL_IB_DISABLE=0` for InfiniBand)
- [ ] 使用高速网络（至少 100Gbps）

### 显存优化 💾
- [ ] 增大 `context_parallel_size`（线性减少显存）
- [ ] 开启 FSDP reshard (`reshard_after_forward: true`)
- [ ] 增大 `gradient_accumulation_steps` 补偿小 batch
- [ ] 考虑降低 `sequence_len`（如果业务允许）

### 调试优化 🐛
- [ ] 检查序列切分日志
- [ ] 监控 NCCL 通信时间 (`NCCL_DEBUG=INFO`)
- [ ] 验证每个 GPU 的序列长度一致
- [ ] 测试不同 CP size 的性能曲线

---

## 🔄 Ring-Flash-Attention 核心原理

### 一句话总结
**每个 GPU 保留完整 Q，但 K/V 在 Ring 上轮流传递，每一轮计算部分 Attention 并用 Online Softmax 增量合并。**

### 执行流程（4 个 GPU 示例）
```
Step 0: 初始状态
GPU 0: Q₀, K₀, V₀ → 计算 Attn₀
GPU 1: Q₁, K₁, V₁ → 计算 Attn₁
GPU 2: Q₂, K₂, V₂ → 计算 Attn₂
GPU 3: Q₃, K₃, V₃ → 计算 Attn₃

Step 1: Ring 传递 K/V
GPU 0: Q₀, K₃, V₃ → 计算 Attn₀' 并合并
GPU 1: Q₁, K₀, V₀ → 计算 Attn₁' 并合并
GPU 2: Q₂, K₁, V₁ → 计算 Attn₂' 并合并
GPU 3: Q₃, K₂, V₂ → 计算 Attn₃' 并合并

Step 2: 继续传递...
（总共 4 轮，每个 GPU 看到所有 K/V）

最终: 每个 GPU 得到完整的 Attention 输出
```

### 关键技术：Online Softmax
```python
# 传统 Softmax（需要完整序列）
scores = Q @ K^T / sqrt(d)
attn_weights = softmax(scores)  # 需要知道所有 scores
output = attn_weights @ V

# Online Softmax（增量更新）
# 第 1 轮
scores₁ = Q @ K₁^T
max₁ = max(scores₁)
exp_scores₁ = exp(scores₁ - max₁)
sum₁ = sum(exp_scores₁)
output₁ = (exp_scores₁ @ V₁) / sum₁

# 第 2 轮（合并）
scores₂ = Q @ K₂^T
max₂ = max(max₁, max(scores₂))
# 重新缩放之前的结果
exp_scores₁ *= exp(max₁ - max₂)
exp_scores₂ = exp(scores₂ - max₂)
sum₂ = sum₁ * exp(max₁ - max₂) + sum(exp_scores₂)
output₂ = (output₁ * sum₁ * exp(max₁ - max₂) + exp_scores₂ @ V₂) / sum₂

# 继续迭代...
```

---

## 📐 配置公式

### GPU 数量计算
```
总 GPU 数 = dp_shard_size × dp_replicate_size × tensor_parallel_size × context_parallel_size
```

### 显存节省估算
```
显存节省 ≈ 1 - (1 / context_parallel_size)

例如：
CP=1 (无CP): 节省 0%
CP=2: 节省 ~50%
CP=4: 节省 ~75%
CP=8: 节省 ~87.5%
```

### 通信开销估算
```
额外通信时间 ≈ (context_parallel_size - 1) × (单次 K/V 传输时间)

通信与计算重叠后，实际开销约为：
通信开销% ≈ 20% + 15% × (context_parallel_size - 1)

例如：
CP=2: ~35% 开销
CP=4: ~65% 开销
CP=8: ~125% 开销（可能变慢！）
```

---

## 💡 最佳实践

### ✅ 推荐
- **序列长度 ≥ 8K** 时才考虑 CP（否则通信开销不划算）
- **CP size = 2 或 4**（更大的 CP 通信开销过高）
- **CP 组内 GPU 在同一节点**（必须有 NVLink）
- **TP × CP ≤ 8**（单节点内）
- **先用 TP 切模型，显存还不够再用 CP 切序列**

### ❌ 避免
- CP size > 8（通信开销爆炸）
- CP 跨节点（Ring 延迟过高）
- 短序列（< 4K）使用 CP（得不偿失）
- micro_batch_size > 1（CP 不支持）
- 不检查 `sequence_len % cp_size == 0`

### 🎯 决策树
```
显存不够？
├─ 模型太大 → 用 TP
└─ 序列太长 → 用 CP
    ├─ 8K-16K → CP=2
    ├─ 32K → CP=4
    └─ 64K+ → CP=4 + 多节点
```

---

## 📚 快速链接

- [详细教程](./context_parallelism_deep_dive.md)
- [源码解析](./cp_source_code_walkthrough.md)
- [TP 快速参考](./tp_quick_reference.md)
- [分析文档索引](./README.md)

---

## 💡 速记口诀

```
CP 专治序列长，
Ring 传递显存降。
Flash Attention 是基础，
Batch 必须等于一。

序列长度能整除，
NVLink 通信效率高。
先用 TP 切模型，
再用 CP 切序列好。

显存节省看 CP 数，
通信开销也随之涨。
八千以下不建议，
长上下文显神通。
```

---

## 🔢 配置示例速查

### 16K 上下文，8 卡
```yaml
dp_shard_size: 2
tensor_parallel_size: 2
context_parallel_size: 2
sequence_len: 16384
micro_batch_size: 1
gradient_accumulation_steps: 16
```

### 32K 上下文，8 卡
```yaml
dp_shard_size: 2
context_parallel_size: 4
sequence_len: 32768
micro_batch_size: 1
gradient_accumulation_steps: 32
```

### 64K 上下文，16 卡（双节点）
```yaml
dp_shard_size: 2
dp_replicate_size: 2
tensor_parallel_size: 2
context_parallel_size: 2
sequence_len: 65536
micro_batch_size: 1
gradient_accumulation_steps: 64
```

---

*打印此页作为速查手册 | 最后更新：2025-11*
