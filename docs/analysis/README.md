# Axolotl 深度技术分析文档

本目录包含 Axolotl 框架的深度技术分析文档，适合对框架内部实现感兴趣的开发者和 infra 工程师。

## 📚 文档列表

### 1. Tensor Parallelism 深度解析
**文件**: [tensor_parallelism_deep_dive.md](./tensor_parallelism_deep_dive.md)

**适合人群**: infra 初学者、想理解 TP 原理的开发者

**内容概要**:
- ✅ 什么是 Tensor Parallelism？（通俗易懂的比喻）
- ✅ 为什么需要 TP？（解决的核心问题）
- ✅ TP 的工作原理（数学推导 + 图解）
- ✅ Axolotl 中的 ND 并行架构
- ✅ 配置示例与最佳实践
- ✅ 常见问题排查

**阅读时间**: ~30 分钟

---

### 2. TP 源码执行流程详解
**文件**: [tp_source_code_walkthrough.md](./tp_source_code_walkthrough.md)

**适合人群**: 想深入理解实现细节的高级开发者

**内容概要**:
- ✅ 从配置到训练的完整执行流程
- ✅ DeviceMesh 构建过程
- ✅ DTensor 转换细节
- ✅ 前向/反向传播的通信机制
- ✅ Checkpoint 保存与加载
- ✅ 调试技巧与性能优化

**阅读时间**: ~45 分钟

---

### 3. Context Parallelism 深度解析
**文件**: [context_parallelism_deep_dive.md](./context_parallelism_deep_dive.md)

**适合人群**: infra 初学者、想理解 CP 原理的开发者

**内容概要**:
- ✅ 什么是 Context Parallelism？（延续 TP 的搬桌子比喻）
- ✅ 为什么需要 CP？（超长上下文的显存瓶颈）
- ✅ Ring-Flash-Attention 机制（分块 Softmax + Ring 通信）
- ✅ 完整的执行流程图解
- ✅ 配置示例与性能调优
- ✅ 常见问题排查

**阅读时间**: ~35 分钟

---

### 4. CP 源码执行流程详解
**文件**: [cp_source_code_walkthrough.md](./cp_source_code_walkthrough.md)

**适合人群**: 想深入理解 Ring-Flash-Attention 实现的高级开发者

**内容概要**:
- ✅ 从配置到 Ring Attention 的完整流程
- ✅ SequenceParallelContextManager 机制
- ✅ 序列切分与 Hook 注册
- ✅ Ring-Flash-Attention 底层实现
- ✅ Online Softmax 数学原理
- ✅ 输出聚合与梯度反向传播

**阅读时间**: ~40 分钟

---

### 5. TP 快速参考卡片
**文件**: [tp_quick_reference.md](./tp_quick_reference.md)

**适合人群**: 需要快速查阅 TP 配置和命令的实践者

**内容概要**:
- ✅ 常用配置速查（单节点、多节点、不同模型规模）
- ✅ 运行命令与调试技巧
- ✅ 性能调优检查清单
- ✅ 问题诊断与解决方案

**阅读时间**: ~10 分钟

---

### 6. CP 快速参考卡片
**文件**: [cp_quick_reference.md](./cp_quick_reference.md)

**适合人群**: 需要快速查阅 CP 配置和命令的实践者

**内容概要**:
- ✅ 长上下文场景配置速查
- ✅ Ring-Flash-Attention 核心原理一页纸总结
- ✅ 显存与通信开销计算公式
- ✅ 最佳实践与常见陷阱

**阅读时间**: ~10 分钟

---

### 7. Data Parallelism 深度解析
**文件**: [data_parallelism_deep_dive.md](./data_parallelism_deep_dive.md)

**适合人群**: infra 初学者、想理解 DP/FSDP/DDP 原理的开发者

**内容概要**:
- ✅ 什么是 Data Parallelism？（延续"搬桌子"比喻）
- ✅ 为什么需要 DP？（训练速度与 batch size）
- ✅ FSDP vs DDP 详细对比（ZeRO 优化器原理）
- ✅ AllReduce 和 ReduceScatter 通信机制
- ✅ Axolotl 的 4D 并行架构
- ✅ 配置示例与最佳实践

**阅读时间**: ~35 分钟

---

### 8. DP 源码执行流程详解
**文件**: [dp_source_code_walkthrough.md](./dp_source_code_walkthrough.md)

**适合人群**: 想深入理解 FSDP/DDP 实现的高级开发者

**内容概要**:
- ✅ 从配置到训练的完整执行流程
- ✅ DeviceMesh 的 DP 维度构建
- ✅ FSDP 模型包装与参数分片
- ✅ 前向/反向传播的 AllGather/ReduceScatter
- ✅ MultipackBatchSampler 数据分发
- ✅ Checkpoint 保存与加载机制

**阅读时间**: ~40 分钟

---

### 9. DP 快速参考卡片
**文件**: [dp_quick_reference.md](./dp_quick_reference.md)

**适合人群**: 需要快速查阅 FSDP/DDP 配置的实践者

**内容概要**:
- ✅ FSDP vs DDP 配置速查
- ✅ 常见场景配置（小/中/大模型）
- ✅ 显存/通信开销计算公式
- ✅ 性能优化检查清单
- ✅ 最佳实践与决策树

**阅读时间**: ~10 分钟

---

### 10. FSDP-1 vs FSDP-2 深度对比
**文件**: [fsdp_versions_comparison.md](./fsdp_versions_comparison.md)

**适合人群**: 想理解两个 FSDP 版本差异的开发者

**内容概要**:
- ✅ FSDP-1 vs FSDP-2 核心差异（延续"搬桌子"比喻）
- ✅ 技术架构对比（API、底层实现）
- ✅ 功能差异详解（N-D 并行、CPU Offload 等）
- ✅ 性能对比（实测数据）
- ✅ 迁移指南（3 分钟完成）
- ✅ 兼容性与限制

**阅读时间**: ~30 分钟

---

### 11. FSDP 版本源码实现对比
**文件**: [fsdp_versions_source_walkthrough.md](./fsdp_versions_source_walkthrough.md)

**适合人群**: 想深入理解 FSDP 版本实现差异的高级开发者

**内容概要**:
- ✅ 代码结构概览（关键文件）
- ✅ 配置解析差异（validation、patch）
- ✅ 模型包装差异（FSDP vs fully_shard）
- ✅ Checkpoint 处理差异（DTensor）
- ✅ 关键代码路径对比

**阅读时间**: ~25 分钟

---

### 12. FSDP 版本快速参考卡片
**文件**: [fsdp_versions_quick_reference.md](./fsdp_versions_quick_reference.md)

**适合人群**: 需要快速决策使用哪个 FSDP 版本的实践者

**内容概要**:
- ✅ 30 秒决策指南
- ✅ 配置对比（FSDP-1 vs FSDP-2）
- ✅ 迁移步骤（3 分钟）
- ✅ 常见场景配置
- ✅ 问题排查速查
- ✅ 性能参考数据

**阅读时间**: ~8 分钟

---

### 13. Sample Packing 深度解析
**文件**: [sample_packing_deep_dive.md](./sample_packing_deep_dive.md)

**适合人群**: infra 初学者、想理解 Sample Packing 原理的开发者

**内容概要**:
- ✅ 什么是 Sample Packing？（延续"搬桌子"比喻）
- ✅ 为什么需要 Sample Packing？（减少 padding 浪费）
- ✅ FFD 打包算法原理（Sequential vs Parallel）
- ✅ Attention Mask 处理机制
- ✅ Sample Packing 与 DDP/FSDP/TP/CP 的结合
- ✅ Sample Packing vs 非 Sample Packing 详细对比
- ✅ 配置示例与最佳实践

**阅读时间**: ~35 分钟

---

### 14. Sample Packing 源码执行流程详解
**文件**: [sample_packing_source_walkthrough.md](./sample_packing_source_walkthrough.md)

**适合人群**: 想深入理解 Sample Packing 实现的高级开发者

**内容概要**:
- ✅ MultipackBatchSampler 详细实现
- ✅ FFD 打包算法源码（Numba 加速）
- ✅ Data Collator 机制（V1 vs V2）
- ✅ Attention Mask 处理（get_unpad_data）
- ✅ Monkeypatch 机制
- ✅ 与训练流程的集成
- ✅ 分布式训练支持

**阅读时间**: ~40 分钟

---

### 15. Sample Packing 快速参考卡片
**文件**: [sample_packing_quick_reference.md](./sample_packing_quick_reference.md)

**适合人群**: 需要快速查阅 Sample Packing 配置的实践者

**内容概要**:
- ✅ 30 秒决策指南（是否启用）
- ✅ 常见场景配置速查
- ✅ 参数详解与调优
- ✅ 问题排查速查（OOM、效率低等）
- ✅ 性能参考数据
- ✅ 最佳实践与避坑指南

**阅读时间**: ~10 分钟

---

## 🚀 快速开始

### 我想快速了解 TP 是什么
👉 阅读 [tensor_parallelism_deep_dive.md](./tensor_parallelism_deep_dive.md) 的第 1-3 章

### 我想快速了解 CP 是什么
👉 阅读 [context_parallelism_deep_dive.md](./context_parallelism_deep_dive.md) 的第 1-3 章

### 我想快速了解 DP/FSDP/DDP 是什么
👉 阅读 [data_parallelism_deep_dive.md](./data_parallelism_deep_dive.md) 的第 1-4 章

### 我想配置 TP 训练
👉 阅读 [tensor_parallelism_deep_dive.md](./tensor_parallelism_deep_dive.md) 第 6 章（实战示例）

### 我想配置 CP 训练（超长上下文）
👉 阅读 [context_parallelism_deep_dive.md](./context_parallelism_deep_dive.md) 第 6 章（实战示例）

### 我想配置 FSDP 训练（大模型）
👉 阅读 [data_parallelism_deep_dive.md](./data_parallelism_deep_dive.md) 第 6 章（实战示例）

### 我想理解 TP 源码实现
👉 阅读 [tp_source_code_walkthrough.md](./tp_source_code_walkthrough.md)

### 我想理解 CP 和 Ring-Flash-Attention
👉 阅读 [cp_source_code_walkthrough.md](./cp_source_code_walkthrough.md)

### 我想理解 FSDP 和 DDP 实现
👉 阅读 [dp_source_code_walkthrough.md](./dp_source_code_walkthrough.md)

### 我需要快速查配置（TP）
👉 阅读 [tp_quick_reference.md](./tp_quick_reference.md)

### 我需要快速查配置（CP）
👉 阅读 [cp_quick_reference.md](./cp_quick_reference.md)

### 我需要快速查配置（FSDP/DDP）
👉 阅读 [dp_quick_reference.md](./dp_quick_reference.md)

### 我应该用 FSDP-1 还是 FSDP-2？
👉 阅读 [fsdp_versions_quick_reference.md](./fsdp_versions_quick_reference.md) 的 30 秒决策指南

### 我想了解 FSDP-1 和 FSDP-2 的区别
👉 阅读 [fsdp_versions_comparison.md](./fsdp_versions_comparison.md) 的第 1-2 章

### 我想快速了解 Sample Packing 是什么
👉 阅读 [sample_packing_deep_dive.md](./sample_packing_deep_dive.md) 的第 1-3 章

### 我想启用 Sample Packing 加速训练
👉 阅读 [sample_packing_quick_reference.md](./sample_packing_quick_reference.md) 的配置速查章节

### 我想理解 Sample Packing 源码实现
👉 阅读 [sample_packing_source_walkthrough.md](./sample_packing_source_walkthrough.md)

### 我遇到了问题
👉 TP 问题：查看 [tp_quick_reference.md](./tp_quick_reference.md) 调试速查章节
👉 CP 问题：查看 [cp_quick_reference.md](./cp_quick_reference.md) 调试速查章节
👉 DP 问题：查看 [dp_quick_reference.md](./dp_quick_reference.md) 调试速查章节
👉 FSDP 版本问题：查看 [fsdp_versions_quick_reference.md](./fsdp_versions_quick_reference.md) 问题排查章节
👉 Sample Packing 问题：查看 [sample_packing_quick_reference.md](./sample_packing_quick_reference.md) 问题排查章节
👉 详细排查：查看各 deep dive 文档第 7 章或第 8 章

---

## 📖 推荐阅读顺序

### 路径 1：从零开始学习并行技术
```
1. data_parallelism_deep_dive.md (第 1-4 章)
   ↓ 理解 DP/FSDP/DDP 基本概念（最常用！）
2. tensor_parallelism_deep_dive.md (第 1-3 章)
   ↓ 理解 TP 基本概念
3. context_parallelism_deep_dive.md (第 1-3 章)
   ↓ 理解 CP 基本概念
4. 各自的第 5 章
   ↓ 了解 Axolotl 架构
5. 各自的第 6 章
   ↓ 动手实践
```

### 路径 2：快速上手配置
```
# 优化训练效率（减少 padding 浪费）
1. sample_packing_quick_reference.md（30秒决策 + 配置速查）
   ↓ 或 sample_packing_deep_dive.md (第 1-3 章了解原理)
   ↓ 启用 Sample Packing（预期：2-3倍加速）

# 小模型训练（≤7B，加速训练）
1. dp_quick_reference.md（场景 4：纯 DDP）
   ↓ 或 data_parallelism_deep_dive.md (第 6.4 节)
   ↓ 配置 DDP

# 中等模型训练（13B-30B，显存优化）
1. dp_quick_reference.md（场景 1：纯 FSDP）
   ↓ 或 data_parallelism_deep_dive.md (第 6.1 节)
   ↓ 配置 FSDP

# 大模型训练（70B+，模型太大）
1. tp_quick_reference.md + dp_quick_reference.md（场景 2）
   ↓ 或 tensor_parallelism_deep_dive.md + data_parallelism_deep_dive.md (第 6 章)
   ↓ 配置 TP + FSDP

# 长上下文训练（序列太长）
1. cp_quick_reference.md (速查配置和命令)
   ↓ 或 context_parallelism_deep_dive.md (第 6 章详解)
   ↓ 配置 CP + FSDP

# 超大模型 + 超长上下文
1. 结合 tp_quick_reference.md + cp_quick_reference.md + dp_quick_reference.md
   ↓ 配置 4D 并行 (TP + CP + FSDP + DDP)

# 遇到问题时
1. 先查快速参考卡片的调试速查章节
   ↓ 如果没解决，再看 deep dive 文档第 7 章
```

### 路径 3：源码研究
```
# TP 源码
1. tp_source_code_walkthrough.md (完整阅读)
   ↓
2. 结合源码阅读：
   - src/axolotl/utils/distributed.py
   - src/axolotl/loaders/model.py
   - src/axolotl/core/builders/causal.py

# CP 源码
1. cp_source_code_walkthrough.md (完整阅读)
   ↓
2. 结合源码阅读：
   - src/axolotl/utils/ctx_managers/sequence_parallel.py
   - src/axolotl/monkeypatch/ring_attn/patch.py
   - src/axolotl/monkeypatch/transformers/trainer_context_parallel.py

# DP 源码
1. dp_source_code_walkthrough.md (完整阅读)
   ↓
2. 结合源码阅读：
   - src/axolotl/utils/distributed.py
   - src/axolotl/utils/schemas/fsdp.py
   - src/axolotl/core/builders/base.py
   - src/axolotl/monkeypatch/accelerate/fsdp2.py
   - src/axolotl/utils/samplers/multipack.py

# Sample Packing 源码
1. sample_packing_source_walkthrough.md (完整阅读)
   ↓
2. 结合源码阅读：
   - src/axolotl/utils/samplers/multipack.py
   - src/axolotl/utils/collators/batching.py
   - src/axolotl/monkeypatch/multipack.py
   - src/axolotl/monkeypatch/utils.py
   - src/axolotl/core/builders/causal.py
```

---

## 🔧 实用资源

### 配置模板

#### 单节点 8 卡 (70B 模型) + Sample Packing
```yaml
base_model: meta-llama/Llama-3.1-70B
dp_shard_size: 4
tensor_parallel_size: 2
fsdp_version: 2
flash_attention: true
bf16: true
gradient_checkpointing: true

# Sample Packing (强烈推荐，2-3倍加速)
sample_packing: true
sample_packing_eff_est: 0.95
pad_to_sequence_len: false
```

#### 双节点 16 卡 (70B+ 模型)
```yaml
base_model: meta-llama/Llama-3.1-70B
dp_shard_size: 4
dp_replicate_size: 2
tensor_parallel_size: 2
fsdp_version: 2
```

#### 长上下文训练
```yaml
base_model: meta-llama/Llama-3.1-8B
dp_shard_size: 2
tensor_parallel_size: 2
context_parallel_size: 2
sequence_len: 16384
micro_batch_size: 1
```

### 运行命令

```bash
# 基本训练
axolotl train config.yaml

# 指定 GPU 数量
axolotl train config.yaml --num-processes 8

# 使用 torchrun
axolotl train config.yaml --launcher torchrun

# 多节点训练
# Node 0:
axolotl train config.yaml --num-processes 16 --num-machines 2 --machine-rank 0

# Node 1:
axolotl train config.yaml --num-processes 16 --num-machines 2 --machine-rank 1
```

---

## 🐛 调试速查表

### 显存问题
```bash
# 检查配置
tensor_parallel_size × dp_shard_size × dp_replicate_size × context_parallel_size = 总 GPU 数

# 开启 reshard
fsdp_config:
  reshard_after_forward: true

# 开启梯度检查点
gradient_checkpointing: true
```

### 速度问题
```bash
# 检查 GPU 互连
nvidia-smi topo -m

# 应该看到 NVLink (NV12/NV18)，而非 PHB (PCIe)

# 开启编译
torch_compile: true

# 使用 Fused Optimizer
optimizer: adamw_torch_fused
```

### Loss 问题
```yaml
# 使用 bf16
bf16: true
fp16: false

# 梯度裁剪
max_grad_norm: 1.0

# 降低学习率
learning_rate: 5e-6  # 原来的一半
```

---

## 📊 性能参考

### Llama-70B on 8×A100 80GB

| 配置 | Tokens/s/GPU | 显存/GPU | Batch Size | Seq Len |
|------|--------------|----------|------------|---------|
| FSDP only | 1800 | 65GB | 256 | 2048 |
| FSDP + TP(2) | 1600 | 45GB | 256 | 2048 |
| FSDP + TP + CP | 1200 | 35GB | 128 | 8192 |

*注：实际性能受模型、数据集、硬件等因素影响*

---

## 🔗 相关链接

### 官方文档
- [Axolotl ND Parallelism 文档](https://docs.axolotl.ai/docs/nd_parallelism.html)
- [HuggingFace Accelerate ND-Parallel 博客](https://huggingface.co/blog/accelerate-nd-parallel)
- [PyTorch DTensor 文档](https://pytorch.org/docs/stable/distributed.tensor.html)

### 学术论文
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

### 代码仓库
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [PyTorch Distributed](https://github.com/pytorch/pytorch/tree/main/torch/distributed)

---

## 🤝 贡献

发现文档错误或想补充内容？欢迎提交 PR 或 Issue！

---

## 📝 更新日志

- **2025-11-23**:
  - 创建 Sample Packing 深度解析文档
  - 创建 Sample Packing 源码执行流程详解文档
  - 创建 Sample Packing 快速参考卡片
  - 更新 README 包含 Sample Packing 文档
  - 完善快速开始指南（Sample Packing 配置）
  - 更新推荐阅读顺序（优先推荐 Sample Packing）

- **2025-11-22 (晚)**:
  - 创建 FSDP-1 vs FSDP-2 深度对比文档
  - 创建 FSDP 版本源码实现对比文档
  - 创建 FSDP 版本快速参考卡片
  - 更新 README 包含 FSDP 版本对比文档
  - 完善快速开始指南（FSDP 版本决策）

- **2025-11-08 (晚)**:
  - 创建 Data Parallelism 深度解析文档
  - 创建 DP 源码执行流程详解文档
  - 创建 DP 快速参考卡片
  - 更新 README 包含完整的 TP/CP/DP 文档套件
  - 更新阅读路径，优先推荐 DP（最常用）

- **2025-11-08 (中)**:
  - 创建 Context Parallelism 深度解析文档
  - 创建 CP 源码执行流程详解文档
  - 创建 TP 快速参考卡片
  - 创建 CP 快速参考卡片

- **2025-11-08 (早)**: 初始版本
  - 创建 Tensor Parallelism 深度解析文档
  - 创建 TP 源码执行流程详解文档
  - 创建本 README

---

*这些文档由 Claude AI 辅助创作，旨在帮助开发者深入理解 Axolotl 的并行训练实现。*

## 📈 文档统计

- **总文档数**: 16 份（3 种并行策略 × 3 层文档 + FSDP 版本对比 × 3 + Sample Packing × 3 + README）
- **总字数**: ~100,000+ 字
- **总代码示例**: 200+ 个
- **覆盖主题**:
  - Tensor Parallelism (TP)
  - Context Parallelism (CP)
  - Data Parallelism (FSDP/DDP)
  - FSDP-1 vs FSDP-2 版本对比
  - Sample Packing（数据打包优化）
- **阅读路径**: 3 条（零基础、快速上手、源码研究）
