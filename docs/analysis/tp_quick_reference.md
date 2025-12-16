# Tensor Parallelism 快速参考卡片 🚀

> 一页纸速查手册，适合快速查阅配置和命令

---

## ⚙️ 基本配置

### 最小化 TP 配置
```yaml
base_model: meta-llama/Llama-3.1-8B
tensor_parallel_size: 2  # 仅此一行！

# 其他必需配置
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
output_dir: ./outputs/tp-test/
bf16: true
flash_attention: true
```

### 推荐的完整配置
```yaml
base_model: meta-llama/Llama-3.1-70B

# === 并行配置 ===
dp_shard_size: 4         # FSDP
tensor_parallel_size: 2  # TP
# 总计：4 × 2 = 8 GPUs

# === FSDP 配置 ===
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer

# === 训练配置 ===
sequence_len: 2048
micro_batch_size: 1
gradient_accumulation_steps: 8
num_epochs: 1

optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 2e-5

# === 性能优化 ===
bf16: true
tf32: true
flash_attention: true
gradient_checkpointing: true

# === 输出 ===
output_dir: ./outputs/tp-70b/
logging_steps: 1
```

---

## 🎯 常用场景配置

### 场景 1：单节点 8 卡，30B-70B 模型
```yaml
# 选项 A：更多 FSDP
dp_shard_size: 8
tensor_parallel_size: 1

# 选项 B：平衡 FSDP + TP
dp_shard_size: 4
tensor_parallel_size: 2  # ← 推荐

# 选项 C：更多 TP
dp_shard_size: 2
tensor_parallel_size: 4
```

### 场景 2：双节点 16 卡，70B+ 模型
```yaml
dp_shard_size: 4         # 节点内 FSDP
dp_replicate_size: 2     # 节点间 DDP
tensor_parallel_size: 2  # 节点内 TP
# 4 × 2 × 2 = 16 GPUs
```

### 场景 3：长上下文 (16K tokens)
```yaml
dp_shard_size: 2
tensor_parallel_size: 2
context_parallel_size: 2
sequence_len: 16384
micro_batch_size: 1  # CP 要求
# 2 × 2 × 2 = 8 GPUs
```

### 场景 4：超大模型 (175B+)，4D 并行
```yaml
dp_shard_size: 2
dp_replicate_size: 2
tensor_parallel_size: 4
context_parallel_size: 2
# 2 × 2 × 4 × 2 = 32 GPUs
```

---

## 🚀 运行命令

### 基本命令
```bash
# 单节点训练
axolotl train config.yaml

# 指定 GPU 数量
axolotl train config.yaml --num-processes 8

# 指定 launcher
axolotl train config.yaml --launcher accelerate
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
# 只运行 2 个 step 测试
axolotl train config.yaml --max-steps 2

# 启用详细日志
NCCL_DEBUG=INFO axolotl train config.yaml

# 使用 PyTorch Profiler
axolotl train config.yaml --use-profiler
```

---

## 🔍 调试速查

### 问题：显存 OOM

#### 检查清单
```bash
✓ 确认 TP size 配置正确
✓ 开启 reshard_after_forward
✓ 开启 gradient_checkpointing
✓ 降低 micro_batch_size
✓ 增大 gradient_accumulation_steps
```

#### 配置调整
```yaml
# 之前
micro_batch_size: 4
gradient_accumulation_steps: 2

# 之后
micro_batch_size: 1          # 降低
gradient_accumulation_steps: 8  # 增大
# 有效 batch size 保持不变：4×2 = 1×8
```

---

### 问题：训练速度慢

#### 检查 GPU 互连
```bash
nvidia-smi topo -m

# ✅ 好（NVLink）:
#   GPU0  GPU1
# 0   X    NV12
# 1  NV12   X

# ❌ 差（PCIe）:
#   GPU0  GPU1
# 0   X    PHB
# 1  PHB   X
```

#### 性能优化配置
```yaml
# 编译优化
torch_compile: true
torch_compile_backend: "inductor"

# Fused 算子
optimizer: adamw_torch_fused

# Flash Attention
flash_attention: true

# 混合精度
bf16: true
tf32: true

# CCE 插件
plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
```

---

### 问题：Loss NaN 或不收敛

#### 配置调整
```yaml
# 使用 bf16（更稳定）
bf16: true
fp16: false

# 梯度裁剪
max_grad_norm: 1.0

# 降低学习率
learning_rate: 1e-5  # 或更小

# Warmup
warmup_steps: 100
# 或
warmup_ratio: 0.1
```

---

## 📊 性能对比表

### 模型大小 → TP 配置映射

| 模型大小 | 单节点 8 卡 | 双节点 16 卡 | 备注 |
|---------|------------|-------------|------|
| 7B-13B | TP=1, FSDP=8 | TP=1, HSDP=8×2 | 不需要 TP |
| 30B | TP=2, FSDP=4 | TP=2, HSDP=4×2 | 推荐 TP |
| 70B | TP=2, FSDP=4 | TP=2, HSDP=4×2 | 必需 TP |
| 175B+ | TP=4, FSDP=2 | TP=4, HSDP=4×2 | + Pipeline |

### Llama-70B 性能参考 (8×A100 80GB)

| TP Size | 显存/GPU | Tokens/s/GPU | 适用场景 |
|---------|---------|--------------|---------|
| 1 (纯FSDP) | ~65GB | 1800 | 基准 |
| 2 | ~45GB | 1600 | 推荐 |
| 4 | ~30GB | 1400 | 显存受限 |

---

## 🛠️ 常用代码片段

### 检查 DTensor 是否生效
```python
# 在训练开始前添加
for name, param in model.named_parameters():
    if hasattr(param, 'placements'):
        print(f"✅ TP 已生效: {name}")
        print(f"   全局形状: {param.shape}")
        print(f"   本地形状: {param.local_tensor.shape}")
        break
    else:
        print(f"❌ TP 未生效（这是普通 Tensor）")
        break
```

### 监控显存使用
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或在 Python 中
import torch
print(f"显存已用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### 计算有效 Batch Size
```python
effective_batch_size = (
    micro_batch_size *
    gradient_accumulation_steps *
    num_gpus  # 如果使用数据并行
)

# 例如：
# micro_batch_size = 1
# gradient_accumulation_steps = 8
# num_gpus = 8 (FSDP)
# effective_batch_size = 1 × 8 × 8 = 64
```

---

## ⚡ 性能调优检查清单

### 必做优化 ✅
- [ ] 启用 Flash Attention (`flash_attention: true`)
- [ ] 使用 bf16 (`bf16: true`)
- [ ] 启用梯度检查点 (`gradient_checkpointing: true`)
- [ ] 使用 Fused Optimizer (`optimizer: adamw_torch_fused`)
- [ ] 开启 TF32 (`tf32: true`)

### 可选优化 🔧
- [ ] 启用 Torch Compile (`torch_compile: true`)
- [ ] 使用 CCE 插件 (Cut Cross Entropy)
- [ ] 调整 FSDP 预取 (`forward_prefetch: true`)
- [ ] 优化数据加载 (`dataloader_num_workers: 4`)

### 调试优化 🐛
- [ ] 检查 GPU 拓扑 (`nvidia-smi topo -m`)
- [ ] 监控通信时间 (`NCCL_DEBUG=INFO`)
- [ ] 验证 batch size (`effective_batch_size`)
- [ ] 检查 DTensor 生效（见上方代码）

---

## 📚 快速链接

- [详细教程](./tensor_parallelism_deep_dive.md)
- [源码解析](./tp_source_code_walkthrough.md)
- [Axolotl 官方文档](https://docs.axolotl.ai/)
- [示例配置](../../examples/distributed-parallel/)

---

## 💡 速记口诀

```
TP 大小看层宽，
快速互连是关键。
FSDP 节省显存量，
bf16 稳定训练好。

单节点内用 TP，
跨节点间用 DDP。
长上下文加 CP，
四维并行最强大。

显存不够开 reshard，
速度不快查拓扑。
Loss 爆炸降学习率，
调试先看 DTensor。
```

---

*打印此页作为速查手册 | 最后更新：2025-11*
