# Data Parallelism 快速参考卡片 🚀

> 一页纸速查手册，适合快速查阅 DP/FSDP/DDP 配置和命令

---

## ⚙️ 基本配置

### 最小化 FSDP 配置
```yaml
base_model: meta-llama/Llama-3.1-13B
fsdp_version: 2  # ← 仅此一行启用 FSDP！

# 自动推断：dp_shard_size = GPU 数量

# 其他必需配置
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
output_dir: ./outputs/fsdp-test/
bf16: true
flash_attention: true
```

### 推荐的完整 FSDP 配置
```yaml
base_model: meta-llama/Llama-3.1-13B

# === FSDP 配置 ===
fsdp_version: 2
dp_shard_size: 8  # 可选，默认为所有 GPU

fsdp_config:
  # Wrapping 策略（按层切分）
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer

  # 显存优化（关键！）
  reshard_after_forward: true  # ← 前向传播后立即释放参数

  # Checkpoint 策略
  state_dict_type: FULL_STATE_DICT  # Rank 0 收集完整模型

  # 其他
  sync_module_states: true
  use_orig_params: true

# === 训练配置 ===
sequence_len: 2048
micro_batch_size: 4
gradient_accumulation_steps: 4

# === 优化器 ===
optimizer: adamw_torch_fused  # Fused 版本更快
learning_rate: 2e-5
lr_scheduler: cosine

# === 性能优化 ===
bf16: true
flash_attention: true
gradient_checkpointing: true

# === 输出 ===
output_dir: ./outputs/fsdp-llama-13b/
logging_steps: 10
save_steps: 500
```

---

## 🎯 常用场景配置

### 场景 1：单节点 8 卡，Llama-13B（纯 FSDP）

```yaml
base_model: meta-llama/Llama-3.1-13B

# === FSDP 配置 ===
fsdp_version: 2
# dp_shard_size 自动推断为 8

fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT

# === 训练配置 ===
sequence_len: 2048
micro_batch_size: 4
gradient_accumulation_steps: 4

# 有效 batch size = 4 × 4 × 8 = 128

bf16: true
flash_attention: true
gradient_checkpointing: true
output_dir: ./outputs/llama-13b-fsdp/
```

### 场景 2：单节点 8 卡，Llama-70B（TP + FSDP）

```yaml
base_model: meta-llama/Llama-3.1-70B

# === 混合并行 ===
tensor_parallel_size: 2  # TP
dp_shard_size: 4         # FSDP
# 总计：2 × 4 = 8 GPUs

fsdp_version: 2

fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT

# === 训练配置 ===
sequence_len: 2048
micro_batch_size: 2
gradient_accumulation_steps: 8

# 有效 batch size = 2 × 8 × 4 = 64
# 注意：TP 不参与 batch size 计算！

bf16: true
flash_attention: true
gradient_checkpointing: true
output_dir: ./outputs/llama-70b-tp-fsdp/
```

### 场景 3：双节点 16 卡，Llama-70B（TP + FSDP + DDP）

```yaml
base_model: meta-llama/Llama-3.1-70B

# === 4D 并行 ===
tensor_parallel_size: 2     # TP（节点内）
dp_shard_size: 4            # FSDP（节点内）
dp_replicate_size: 2        # DDP（跨节点）
# 总计：2 × 4 × 2 = 16 GPUs

fsdp_version: 2

fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
  state_dict_type: FULL_STATE_DICT

# === 训练配置 ===
sequence_len: 4096
micro_batch_size: 1
gradient_accumulation_steps: 16

# 有效 batch size = 1 × 16 × 4 × 2 = 128

bf16: true
flash_attention: true
gradient_checkpointing: true
output_dir: ./outputs/llama-70b-multi-node/
```

### 场景 4：单节点 8 卡，Llama-8B（纯 DDP，不用 FSDP）

```yaml
base_model: meta-llama/Llama-3.1-8B

# === 不配置 fsdp_config，自动使用 DDP ===
# Axolotl 会自动用 DDP 包装模型

# === 训练配置 ===
sequence_len: 2048
micro_batch_size: 8
gradient_accumulation_steps: 2

# 有效 batch size = 8 × 2 × 8 = 128

bf16: true
flash_attention: true
output_dir: ./outputs/llama-8b-ddp/
```

---

## 🚀 运行命令

### 单节点训练
```bash
# 基本命令
axolotl train config.yaml

# 指定 GPU 数量
axolotl train config.yaml --num-processes 8

# 使用 torchrun（推荐）
axolotl train config.yaml --launcher torchrun
```

### 多节点训练
```bash
# === Node 0 (Master) ===
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
# 测试 FSDP 配置
axolotl train config.yaml --max-steps 5

# 启用 NCCL 调试
NCCL_DEBUG=INFO axolotl train config.yaml --max-steps 2

# 监控显存使用
watch -n 1 nvidia-smi

# 性能分析
nsys profile -o profile.qdrep \
    python -m axolotl.cli.train config.yaml --max-steps 10
```

---

## 🔍 调试速查

### 问题：显存 OOM（Out of Memory）

#### 检查清单
```bash
✓ 启用 FSDP (fsdp_version: 2)
✓ 开启 reshard_after_forward: true
✓ 开启 gradient_checkpointing: true
✓ 减小 micro_batch_size
✓ 考虑增加 dp_shard_size
```

#### 配置调整
```yaml
# === 选项 1：启用 FSDP ===
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true  # ← 关键！

# === 选项 2：增大 FSDP 切分 ===
dp_shard_size: 8  # 从 4 增加到 8

# === 选项 3：减小 Batch Size ===
micro_batch_size: 1  # 从 4 减小
gradient_accumulation_steps: 16  # 增大以补偿

# === 选项 4：开启梯度检查点 ===
gradient_checkpointing: true

# === 选项 5：混合 TP + FSDP ===
tensor_parallel_size: 2
dp_shard_size: 4

# === 选项 6：CPU Offload（极端情况）===
fsdp_config:
  offload_params: true  # 参数 offload 到 CPU
  cpu_offload_pin_memory: false
```

---

### 问题：训练速度慢

#### 诊断步骤
```bash
# 1. 检查 GPU 互连
nvidia-smi topo -m

# ✅ 好（NVLink）:
#   GPU0  GPU1
# 0   X    NV12
# 1  NV12   X

# ❌ 差（PCIe）:
#   GPU0  GPU1
# 0   X    PHB
# 1  PHB   X

# 2. 检查通信时间
NCCL_DEBUG=INFO axolotl train config.yaml 2>&1 | grep "AllGather\|AllReduce"

# 3. 查看 tokens/s
# 在日志中查找 "Tokens/s/GPU"
```

#### 性能优化配置
```yaml
# === 优化 1：使用 Fused Optimizer ===
optimizer: adamw_torch_fused  # ← 比 adamw_torch 快 ~10%

# === 优化 2：优化 FSDP Wrapping ===
fsdp_config:
  auto_wrap_policy: TRANSFORMER_BASED_WRAP  # ← 比 SIZE_BASED_WRAP 快
  transformer_layer_cls_to_wrap: LlamaDecoderLayer

# === 优化 3：数据加载优化 ===
dataloader_num_workers: 4
dataloader_pin_memory: true
dataloader_prefetch_factor: 2

# === 优化 4：减少通信频率 ===
gradient_accumulation_steps: 8  # 增大
# 梯度累积期间不通信，减少 AllReduce 次数

# === 优化 5：混合精度 ===
bf16: true
tf32: true

# === 优化 6：如果模型能放进单 GPU，改用 DDP ===
# 注释掉 fsdp_config，Axolotl 会自动用 DDP
# DDP 通信量 < FSDP（约 1/3）
```

---

### 问题：Loss NaN 或发散

#### 诊断
```python
# 在训练脚本中添加梯度检查

import math

for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            print(f"异常梯度: {name}, norm={grad_norm}")
```

#### 配置调整
```yaml
# === 选项 1：使用 bf16（比 fp16 更稳定）===
bf16: true
fp16: false

# === 选项 2：梯度裁剪 ===
max_grad_norm: 1.0

# === 选项 3：降低学习率 ===
learning_rate: 5e-6  # 从 2e-5 降低

# === 选项 4：增加 Warmup ===
warmup_steps: 100
warmup_ratio: 0.05

# === 选项 5：检查数据质量 ===
# 确保数据集没有异常值、损坏的样本
```

---

### 问题：多节点通信失败

#### 诊断
```bash
# 1. 检查网络连通性
ping <NODE1_IP>

# 2. 检查端口
# Node 0:
nc -l 29500

# Node 1:
nc <NODE0_IP> 29500

# 3. 检查 NCCL 环境变量
env | grep NCCL
```

#### 解决方案
```bash
# === 选项 1：指定网络接口 ===
export NCCL_SOCKET_IFNAME=eth0  # 或 ib0（InfiniBand）
export GLOO_SOCKET_IFNAME=eth0

# === 选项 2：禁用 InfiniBand（如果没有）===
export NCCL_IB_DISABLE=1

# === 选项 3：增加超时时间 ===
export NCCL_TIMEOUT=7200  # 2 小时

# === 选项 4：启用 NCCL 调试 ===
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# === 选项 5：检查防火墙 ===
sudo ufw allow 29500:29600/tcp
sudo ufw allow 29500:29600/udp
```

---

### 问题：Checkpoint 保存/加载失败

#### 症状
```
Rank 0 saves successfully
Rank 1+ hangs...
```

#### 解决方案
```yaml
# === 选项 1：使用 FULL_STATE_DICT（推荐）===
fsdp_config:
  state_dict_type: FULL_STATE_DICT
  # Rank 0 收集并保存完整模型
  # 其他 ranks 等待（barrier）

# === 选项 2：使用 SHARDED_STATE_DICT ===
fsdp_config:
  state_dict_type: SHARDED_STATE_DICT
  # 每个 rank 保存自己的切片
  # 恢复时需要相同的 GPU 配置

# === 选项 3：最终保存完整模型 ===
fsdp_config:
  state_dict_type: SHARDED_STATE_DICT  # 训练中
  final_state_dict_type: FULL_STATE_DICT  # 训练结束
```

---

## 📊 FSDP vs DDP 对比表

| 维度 | FSDP | DDP |
|------|------|-----|
| **显存占用** | ~1/N（N 是 GPU 数） | 100%（每个 GPU） |
| **通信量** | AllGather + ReduceScatter（~3×） | AllReduce（~1×） |
| **通信频率** | 每层 2 次（前向+反向） | 每个 iteration 1 次 |
| **适用模型大小** | 可扩展到极大模型 | 受单 GPU 显存限制 |
| **配置复杂度** | 复杂（需要 wrap policy 等） | 简单（自动） |
| **速度** | 较慢（通信多） | 较快（通信少） |

### 决策树
```
模型能放进单个 GPU 显存？
├─ Yes → 用 DDP
│   └─ 配置：不启用 fsdp_config
│
└─ No → 用 FSDP
    ├─ 配置：fsdp_version: 2 + fsdp_config
    └─ 建议：
        - reshard_after_forward: true（节省显存）
        - TRANSFORMER_BASED_WRAP（性能更好）
        - 确保 NVLink（通信带宽）
```

---

## 🛠️ 常用代码片段

### 检查 FSDP 是否生效
```python
import torch.distributed as dist

if dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 检查模型参数大小
    total_params = sum(p.numel() for p in model.parameters())
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    print(f"Rank {rank}: 参数数量 = {total_params:,}")
    print(f"Rank {rank}: 参数显存 = {param_memory / 1e9:.2f} GB")

    # 如果使用 FSDP，每个 rank 的参数显存应该约为 total_memory / world_size
```

### 监控梯度同步
```python
# 添加通信监控

import time

class CommunicationTimer:
    def __init__(self):
        self.comm_time = 0
        self.compute_time = 0

    def on_before_backward(self):
        self.backward_start = time.time()

    def on_after_backward(self):
        self.backward_end = time.time()
        self.compute_time += self.backward_end - self.backward_start

    def on_before_optimizer_step(self):
        self.step_start = time.time()

    def on_after_optimizer_step(self):
        self.step_end = time.time()
        self.comm_time += self.step_end - self.step_start

    def report(self):
        total_time = self.comm_time + self.compute_time
        comm_ratio = self.comm_time / total_time * 100
        print(f"计算时间: {self.compute_time:.2f}s")
        print(f"通信时间: {self.comm_time:.2f}s ({comm_ratio:.1f}%)")
```

### 计算有效 Batch Size
```python
# Data Parallelism 下的有效 batch size

effective_batch_size = (
    micro_batch_size *              # 每个 GPU 的 batch size
    gradient_accumulation_steps *   # 梯度累积步数
    dp_shard_size *                 # FSDP 并行度
    dp_replicate_size               # DDP 并行度（如果有）
)

# 例子 1：单节点 8 卡 FSDP
# micro_batch_size = 4
# gradient_accumulation_steps = 4
# dp_shard_size = 8
# dp_replicate_size = 1
# effective_batch_size = 4 × 4 × 8 × 1 = 128

# 例子 2：双节点 16 卡 FSDP + DDP
# micro_batch_size = 1
# gradient_accumulation_steps = 16
# dp_shard_size = 4
# dp_replicate_size = 2
# effective_batch_size = 1 × 16 × 4 × 2 = 128

# 注意：TP 和 CP 不参与 batch size 计算！
```

### 检查参数同步
```python
# 验证所有 ranks 的模型参数是否一致

import torch.distributed as dist

def check_model_sync(model):
    if not dist.is_initialized():
        return

    rank = dist.get_rank()

    for name, param in model.named_parameters():
        # 计算参数的 checksum
        checksum = param.data.sum().item()

        # 收集所有 ranks 的 checksum
        all_checksums = [torch.tensor(0.0) for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_checksums, checksum)

        # Rank 0 检查
        if rank == 0:
            if not all(abs(c - checksum) < 1e-5 for c in all_checksums):
                print(f"❌ 参数不同步: {name}")
            else:
                print(f"✅ 参数同步: {name}")

# 使用
check_model_sync(model)
```

---

## ⚡ 性能调优检查清单

### 必做优化 ✅
- [ ] 使用 bf16 (`bf16: true`)
- [ ] 启用 Flash Attention (`flash_attention: true`)
- [ ] FSDP 开启 reshard (`reshard_after_forward: true`)
- [ ] 使用 Fused Optimizer (`optimizer: adamw_torch_fused`)
- [ ] 开启梯度检查点 (`gradient_checkpointing: true`)

### 通信优化 🚀
- [ ] 确保 NVLink (`nvidia-smi topo -m`)
- [ ] 使用 TRANSFORMER_BASED_WRAP（比 SIZE_BASED 快）
- [ ] 增大梯度累积（减少通信频率）
- [ ] 多节点：配置高速网络（InfiniBand/100Gbps+）

### 显存优化 💾
- [ ] FSDP `reshard_after_forward: true`
- [ ] 减小 `micro_batch_size`，增大 `gradient_accumulation_steps`
- [ ] 开启 `gradient_checkpointing`
- [ ] 极端情况：CPU offload (`offload_params: true`)

### 调试优化 🐛
- [ ] 启用 NCCL 日志 (`NCCL_DEBUG=INFO`)
- [ ] 监控显存使用 (`nvidia-smi dmon`)
- [ ] 检查梯度异常（NaN/Inf）
- [ ] 验证模型参数同步

---

## 📐 配置公式

### GPU 数量计算
```
总 GPU 数 = tensor_parallel_size × context_parallel_size × dp_shard_size × dp_replicate_size
```

### 有效 Batch Size 计算
```
有效 batch size = micro_batch_size × gradient_accumulation_steps × dp_shard_size × dp_replicate_size

注意：TP 和 CP 不参与 batch size 计算！
```

### FSDP 显存节省估算
```
显存节省 ≈ 1 / dp_shard_size

例如：
dp_shard_size = 1 (无FSDP): 节省 0%
dp_shard_size = 4: 节省 ~75%
dp_shard_size = 8: 节省 ~87.5%
```

### 通信开销估算
```
FSDP 通信量 ≈ 3 × DDP 通信量

原因：
- DDP: 1 次 AllReduce（梯度）
- FSDP: 每层 2 次（AllGather 参数 + ReduceScatter 梯度）
```

---

## 💡 最佳实践

### ✅ 推荐
- **模型 ≤7B** → 用 DDP（简单高效）
- **模型 >7B** → 用 FSDP（节省显存）
- **模型 >30B** → FSDP + TP（混合并行）
- **多节点** → FSDP + DDP（节点内 FSDP，节点间 DDP）
- **FSDP wrapping** → TRANSFORMER_BASED_WRAP（性能更好）
- **Checkpoint** → FULL_STATE_DICT（易于使用）

### ❌ 避免
- 小模型使用 FSDP（通信开销不划算）
- FSDP 不开启 `reshard_after_forward`（显存节省少）
- 使用 SIZE_BASED_WRAP（性能较差）
- 忘记设置 `sampler.set_epoch()`（每个 epoch 数据相同）

### 🎯 决策树
```
选择并行策略？
├─ 模型能放进单 GPU → DDP
├─ 模型太大 → FSDP
├─ 模型极大 + 序列短 → FSDP + TP
└─ 模型极大 + 序列长 → FSDP + TP + CP
```

---

## 📚 快速链接

- [详细教程](./data_parallelism_deep_dive.md)
- [源码解析](./dp_source_code_walkthrough.md)
- [TP 快速参考](./tp_quick_reference.md)
- [CP 快速参考](./cp_quick_reference.md)
- [分析文档索引](./README.md)

---

## 💡 速记口诀

```
模型小用 DDP，
模型大用 FSDP。
Reshard 要开启，
显存节省多。

Batch 累积算，
梯度少通信。
NVLink 是关键，
速度飞起来。

FULL_STATE_DICT，
保存最简单。
Fused Optimizer，
训练更高效。
```

---

## 🔢 配置示例速查

### Llama-8B, 8 卡 (DDP)
```yaml
# 不配置 fsdp，自动用 DDP
micro_batch_size: 8
gradient_accumulation_steps: 2
# 有效 batch = 8 × 2 × 8 = 128
```

### Llama-13B, 8 卡 (FSDP)
```yaml
fsdp_version: 2
fsdp_config:
  reshard_after_forward: true
micro_batch_size: 4
gradient_accumulation_steps: 4
# 有效 batch = 4 × 4 × 8 = 128
```

### Llama-70B, 8 卡 (TP + FSDP)
```yaml
tensor_parallel_size: 2
dp_shard_size: 4
fsdp_version: 2
micro_batch_size: 2
gradient_accumulation_steps: 8
# 有效 batch = 2 × 8 × 4 = 64
```

### Llama-70B, 16 卡双节点 (TP + FSDP + DDP)
```yaml
tensor_parallel_size: 2
dp_shard_size: 4
dp_replicate_size: 2
fsdp_version: 2
micro_batch_size: 1
gradient_accumulation_steps: 16
# 有效 batch = 1 × 16 × 4 × 2 = 128
```

---

*打印此页作为速查手册 | 最后更新：2025-11*
