# Quantization Aware Training (QAT) 深度解读

## 目录

1. [什么是 Quantization Aware Training (QAT)？](#第一章什么是-quantization-aware-training-qat)
2. [为什么需要 QAT？](#第二章为什么需要-qat)
3. [QAT 的原理](#第三章qat-的原理)
4. [Axolotl 中的 QAT 实现](#第四章axolotl-中的-qat-实现)
5. [源码解析](#第五章源码解析)
6. [实战示例](#第六章实战示例)
7. [常见问题与最佳实践](#第七章常见问题与最佳实践)

---

## 第一章：什么是 Quantization Aware Training (QAT)？

### 1.1 搬桌子比喻：提前适应窄门

继续用我们的"搬桌子"比喻来理解 QAT。

**传统方式（Post-Training Quantization，PTQ）：**

想象你要把一张大桌子从房间 A 搬到房间 B，但两个房间之间有个窄门。传统做法是：
1. 在房间 A 里正常组装好桌子（正常训练模型）
2. 搬到门口时，才发现门太窄，桌子过不去
3. 只能临时拆掉桌子的一些部分，或者强行挤压变形（PTQ 量化）
4. 结果桌子可能不太稳定，质量下降（精度损失）

**QAT 方式：**

而 QAT 的做法是：
1. 一开始就知道要经过窄门（知道模型要量化部署）
2. 在组装桌子时，就按照窄门的尺寸来调整设计（训练时就模拟量化）
3. 组装过程中不断练习通过窄门（训练中持续应用 fake quantization）
4. 最终组装完成的桌子，刚好能完美通过窄门（量化后精度保持良好）

```
传统 PTQ：
┌─────────────┐      ┌───┐      ┌─────────────┐
│  正常训练    │ ──>  │窄门│ ──>  │ 精度下降？   │
│  (FP32)     │      │❌ │      │  (INT8)     │
└─────────────┘      └───┘      └─────────────┘
                强行压缩

QAT：
┌─────────────┐      ┌───┐      ┌─────────────┐
│  训练+模拟   │ ──>  │窄门│ ──>  │ 精度保持！   │
│  量化约束    │      │✅ │      │  (INT8)     │
└─────────────┘      └───┘      └─────────────┘
         提前适应
```

### 1.2 技术定义

**Quantization Aware Training (QAT)** 是一种在训练过程中模拟量化效果的技术，通过在前向传播中插入 "fake quantization" 节点，让模型的权重和激活值在训练时就适应低精度表示，从而在最终量化部署时保持更高的精度。

**核心思想：**
- **Train with quantization in mind**: 训练时就考虑量化约束
- **Simulate low precision**: 使用 fake quantization 模拟低精度计算
- **Adapt weights**: 让权重和激活值自适应到量化格式
- **Maintain accuracy**: 量化后精度损失最小化

### 1.3 QAT vs PTQ

| 特性 | PTQ (Post-Training Quantization) | QAT (Quantization Aware Training) |
|------|----------------------------------|-----------------------------------|
| **训练成本** | 无需重新训练，成本低 | 需要训练/微调，成本高 |
| **精度保持** | 低精度量化时可能损失较大 | 精度损失小，尤其是低比特量化 |
| **实现难度** | 简单，直接应用 | 需要修改训练流程 |
| **适用场景** | INT8/FP8 高精度量化 | INT4 及以下低精度量化 |
| **部署速度** | 快速部署 | 需要训练完成后转换 |
| **比喻** | 搬完桌子再压缩 | 边搬桌子边适应窄门 |

### 1.4 支持的量化格式

Axolotl 通过 **torchao** 库支持多种量化格式：

#### Weight 量化格式
- **INT4**: 4-bit 整数量化（最常用于低精度 QAT）
- **INT8**: 8-bit 整数量化
- **FP8 (float8_e4m3fn)**: 8-bit 浮点量化
- **NVFP4**: NVIDIA 4-bit 浮点量化（实验性）

#### Activation 量化格式
- **INT8**: 8-bit 整数激活量化
- **FP8 (float8_e4m3fn)**: 8-bit 浮点激活量化
- **None**: 仅量化权重，激活保持 FP16/BF16

#### 常见组合
| Weight | Activation | 用途 | 压缩比 |
|--------|-----------|------|--------|
| INT4 | None | 推理加速（仅权重） | 4x |
| INT4 | INT8 | 推理加速（激活+权重） | 3-4x |
| INT8 | INT8 | 平衡精度和性能 | 2x |
| FP8 | FP8 | 训练加速（需硬件支持） | 2x |

---

## 第二章：为什么需要 QAT？

### 2.1 PTQ 的局限性

#### 问题 1：低精度量化时精度损失严重

**场景**：将 Llama-3-8B 模型从 FP16 量化到 INT4

```python
# PTQ 量化（直接量化）
Original FP16 模型:
├─ Perplexity: 8.23
└─ HellaSwag Accuracy: 76.5%

PTQ INT4 量化后:
├─ Perplexity: 12.87 ↑ (恶化 56%)
└─ HellaSwag Accuracy: 71.2% ↓ (下降 5.3%)
```

**搬桌子类比**：就像把一张精心打磨的实木桌子，为了通过窄门，不得不用锯子锯掉一些边角。结果虽然能通过，但桌子表面凹凸不平，稳定性下降。

#### 问题 2：权重分布不均导致量化误差

**问题**：神经网络的权重分布往往不均匀，某些层的权重范围很大，某些层很小。

```
权重分布示例（某个 Linear 层）:
                    ┌─────┐
                    │     │
         ┌──────┐   │     │   ┌──────┐
         │      │   │     │   │      │
    ─────┴──────┴───┴─────┴───┴──────┴─────
   -2.0        -0.5  0.0  0.5       2.0
           离群值会压缩主要分布的量化精度

如果用统一的量化范围 [-2.0, 2.0]，
大部分权重在 [-0.5, 0.5] 的精度会很差。
```

**PTQ 的问题**：量化时只能根据现有权重分布来设定量化参数，无法优化权重分布本身。

**QAT 的解决方案**：训练过程中，权重会自动调整分布，适应量化约束。

#### 问题 3：激活值量化误差累积

在深度神经网络中，每一层的激活值都会被量化，误差会逐层累积：

```
Layer 1: FP16 → INT8 (误差 ε₁)
   ↓
Layer 2: INT8 → INT8 (误差 ε₂)
   ↓
Layer 3: INT8 → INT8 (误差 ε₃)
   ↓
...
   ↓
Layer N: 累积误差 Σεᵢ → 精度大幅下降
```

**搬桌子类比**：每经过一个窄门，桌子就要被挤压变形一次。经过 32 层窄门后，桌子可能已经完全变形了。

### 2.2 QAT 的优势

#### 优势 1：权重自适应量化约束

QAT 训练时，模型权重会主动学习适应量化带来的约束：

```python
# QAT 训练过程
for epoch in range(num_epochs):
    for batch in dataloader:
        # 前向传播：模拟量化
        weight_fp32 = model.linear.weight  # FP32 权重
        weight_quantized = fake_quantize(weight_fp32)  # 模拟 INT4
        output = F.linear(input, weight_quantized)  # 用量化权重计算

        # 反向传播：更新 FP32 权重
        loss.backward()
        optimizer.step()  # 权重更新时考虑了量化误差
```

**关键**：权重更新时，梯度反映了量化后的误差，因此权重会朝着"量化友好"的方向调整。

**搬桌子类比**：在组装桌子时，不断测量窄门的尺寸，调整桌子的每个零件，确保最终成品刚好能通过窄门。

#### 优势 2：显著提升低精度量化精度

**实验数据**（Llama-3-8B，torchao 官方数据）：

| 方法 | Perplexity (WikiText) | HellaSwag Accuracy | 精度恢复率 |
|------|----------------------|-------------------|-----------|
| FP16 Baseline | 8.23 | 76.5% | - |
| PTQ INT4 | 12.87 | 71.2% | - |
| QAT INT4 | 9.71 | 74.7% | 68% perplexity, 96% accuracy |

**搬桌子类比**：
- PTQ：强行挤压，桌子变形严重
- QAT：提前适应，桌子形状基本保持

#### 优势 3：支持更激进的量化策略

QAT 可以支持一些 PTQ 难以实现的量化策略：

- **Per-group quantization**: 将权重分组，每组独立量化
- **Mixed precision**: 关键层用高精度，其他层用低精度
- **Learned quantization parameters**: 量化范围和零点通过学习得到

### 2.3 QAT 的成本

#### 成本 1：训练时间增加

QAT 需要额外的训练时间：

```
Full Fine-tuning QAT:
├─ 从头训练模型 + QAT: 100% 训练成本
└─ 时间成本：几天到几周

LoRA + QAT:
├─ LoRA 微调 + QAT: 10-20% 训练成本
└─ 时间成本：几小时到一天

PTQ:
├─ 无需训练，直接量化: ~0% 训练成本
└─ 时间成本：几分钟
```

**权衡**：如果精度要求高（如 INT4 量化），QAT 的成本是值得的。如果精度要求一般（如 INT8），PTQ 足够了。

#### 成本 2：实现复杂度增加

QAT 需要修改训练流程：
1. 插入 fake quantization 节点
2. 管理 FP32 主权重 + INT4 量化权重
3. 训练后转换为实际量化模型
4. 调试量化误差

**搬桌子类比**：不仅要组装桌子，还要不断测量窄门尺寸，调整设计，比直接组装复杂得多。

### 2.4 何时使用 QAT？

**推荐使用 QAT 的场景：**

✅ **低精度量化**（INT4 及以下）
✅ **精度要求高**（生产环境部署）
✅ **有训练资源**（GPU 集群、时间充足）
✅ **模型尺寸大**（7B+ 参数，量化收益明显）

**推荐使用 PTQ 的场景：**

✅ **高精度量化**（INT8/FP8）
✅ **快速部署**（原型验证、实验）
✅ **训练资源有限**（个人开发者）
✅ **精度容忍度高**（非关键应用）

---

## 第三章：QAT 的原理

### 3.1 Fake Quantization 核心机制

#### 3.1.1 什么是 Fake Quantization？

**定义**：Fake Quantization 是一种在训练时模拟量化效果的技术，在前向传播中将 FP32 值量化再反量化回 FP32，在反向传播中使用直通估计器（Straight-Through Estimator, STE）传递梯度。

**核心思想**：

```python
# Fake Quantization 伪代码
def fake_quantize(x, scale, zero_point, quant_min, quant_max):
    # 1. 量化：FP32 → INT4
    x_quant = torch.clamp(
        torch.round(x / scale) + zero_point,
        quant_min,
        quant_max
    )

    # 2. 反量化：INT4 → FP32
    x_dequant = (x_quant - zero_point) * scale

    # 3. 返回 FP32（但已经模拟了量化误差）
    return x_dequant

# 反向传播：使用 STE（Straight-Through Estimator）
# ∂L/∂x ≈ ∂L/∂x_dequant（忽略 round 的梯度）
```

**搬桌子类比**：
1. **量化**：测量桌子通过窄门时需要压缩多少
2. **反量化**：记录压缩后的形状，再恢复到原尺寸
3. **训练更新**：根据"压缩-恢复"后的误差，调整桌子设计

#### 3.1.2 量化公式

对于 **对称量化**（Symmetric Quantization）：

```
x_quant = round(x / scale)
x_dequant = x_quant * scale

其中:
scale = max(|x|) / (2^(bits-1) - 1)
```

对于 **非对称量化**（Affine Quantization）：

```
x_quant = round(x / scale) + zero_point
x_dequant = (x_quant - zero_point) * scale

其中:
scale = (max(x) - min(x)) / (2^bits - 1)
zero_point = round(-min(x) / scale)
```

**示例**（INT4 对称量化）：

```python
# 权重范围: [-0.8, 0.8]
# INT4 范围: [-7, 7] (4-bit signed)

scale = 0.8 / 7 = 0.114

原始权重:    [-0.8, -0.4,  0.0,  0.4,  0.8]
量化值:      [  -7,   -4,    0,    4,    7]
反量化值:    [-0.798, -0.456, 0.0, 0.456, 0.798]
量化误差:    [0.002, -0.056, 0.0, 0.056, -0.002]
```

**关键观察**：
- 量化误差是不可避免的（除非权重恰好是 scale 的整数倍）
- QAT 的目标：让权重分布尽量对齐量化格子点（grid points）

#### 3.1.3 Per-Tensor vs Per-Channel vs Per-Group 量化

**Per-Tensor Quantization**：整个张量共享一个 scale

```
Weight Tensor: [out_features, in_features]
Scale: 一个标量

优点: 计算简单，硬件友好
缺点: 离群值会压缩整体精度
```

**Per-Channel Quantization**：每个输出通道独立 scale

```
Weight Tensor: [out_features, in_features]
Scale: [out_features] 向量

优点: 每个通道有独立动态范围，精度更高
缺点: 计算开销略大
```

**Per-Group Quantization**：将权重分组，每组独立 scale

```
Weight Tensor: [out_features, in_features]
Group Size: 32 (例如)
Scale: [out_features, in_features // 32]

优点: 平衡精度和性能
缺点: 需要调整 group_size 超参数
```

**搬桌子类比**：
- **Per-Tensor**：整张桌子用同一个压缩比例
- **Per-Channel**：桌子的每条腿独立压缩
- **Per-Group**：桌子每个小区域独立压缩

**Axolotl 默认配置**：

```yaml
qat:
  group_size: 32  # Per-group quantization
  weight_dtype: int4
  activation_dtype: int8
```

### 3.2 Fake Quantized Linear 层

#### 3.2.1 FakeQuantizedLinear 结构

torchao 的 `FakeQuantizedLinear` 层是 QAT 的核心组件：

```python
class FakeQuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ...):
        super().__init__()
        # 主权重（始终保持 FP32/BF16）
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # 权重 fake quantizer
        self.weight_fake_quantizer = WeightFakeQuantizer(
            dtype=weight_dtype,  # INT4/INT8/FP8
            group_size=group_size,
        )

        # 激活 fake quantizer（可选）
        self.activation_fake_quantizer = ActivationFakeQuantizer(
            dtype=activation_dtype,  # INT8/FP8
        ) if activation_dtype else None

    def forward(self, x):
        # 1. 量化权重（模拟）
        if self.weight_fake_quantizer.enabled:
            weight = self.weight_fake_quantizer(self.weight)
        else:
            weight = self.weight

        # 2. 量化激活（模拟）
        if self.activation_fake_quantizer and self.activation_fake_quantizer.enabled:
            x = self.activation_fake_quantizer(x)

        # 3. 线性计算（用量化后的值）
        return F.linear(x, weight, self.bias)
```

**关键点**：
- `self.weight` 始终是 FP32/BF16（用于梯度更新）
- `weight_fake_quantizer(self.weight)` 返回模拟量化后的 FP32 值
- 反向传播时，梯度流经 fake quantizer，更新 `self.weight`

**搬桌子类比**：
- `self.weight`：桌子的实际尺寸（高精度）
- `weight_fake_quantizer`：模拟通过窄门后的形状
- 前向传播：用"模拟窄门后的形状"来计算 loss
- 反向传播：根据 loss 调整"实际尺寸"

#### 3.2.2 Straight-Through Estimator (STE)

量化操作 `round()` 是不可微的，QAT 使用 **Straight-Through Estimator** 来传递梯度：

```python
class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 直通：梯度直接传递，忽略 round
        return grad_output

# 使用
x_rounded = StraightThroughRound.apply(x)
```

**数学表达**：

```
前向: y = round(x / scale) * scale
反向: ∂L/∂x ≈ ∂L/∂y  (忽略 round 的梯度)
```

**为什么可以这样做？**

虽然 `round()` 的梯度理论上是 0，但 STE 假设在量化点附近，`round()` 可以近似为恒等函数。实验表明这个近似在 QAT 中效果很好。

**搬桌子类比**：
- 前向：测量桌子能否通过窄门（离散的"能"或"不能"）
- 反向：根据"通过窄门后的效果"，连续地调整桌子尺寸

### 3.3 QAT 训练流程

#### 3.3.1 训练三阶段

```
阶段 1: Prepare (准备阶段)
├─ 将模型的 nn.Linear 替换为 FakeQuantizedLinear
├─ 将模型的 nn.Embedding 替换为 FakeQuantizedEmbedding (可选)
└─ 初始化 fake quantizers

阶段 2: Training (训练阶段)
├─ 前向传播：使用 fake quantization
├─ 计算 loss
├─ 反向传播：通过 STE 传递梯度
└─ 更新 FP32 主权重

阶段 3: Convert (转换阶段)
├─ 移除 fake quantizers
├─ 将 FP32 权重转换为实际 INT4/INT8 权重
└─ 导出量化模型
```

**搬桌子类比**：
1. **Prepare**：标记出所有需要通过窄门的桌子零件
2. **Training**：不断模拟通过窄门，调整零件尺寸
3. **Convert**：最终组装完成，实际搬运通过窄门

#### 3.3.2 Fake Quantization 开关控制

Axolotl 支持在训练中途启用 fake quantization：

```yaml
qat:
  fake_quant_after_n_steps: 1000
```

**训练流程**：

```
Step 0 - 999:
├─ Fake quantization 禁用
└─ 正常 FP32/BF16 训练（预热）

Step 1000+:
├─ Fake quantization 启用
└─ 开始模拟量化训练
```

**为什么要延迟启用？**

1. **权重初始化阶段**：模型刚初始化时，权重分布可能不适合量化，直接 QAT 可能导致训练不稳定
2. **梯度预热**：先用 FP32 训练几百步，让权重分布收敛到合理范围，再启用 QAT
3. **学习率配合**：通常在学习率开始衰减后启用 QAT，避免剧烈震荡

**搬桌子类比**：
- Step 0-999：先正常组装桌子，不管窄门
- Step 1000+：桌子基本成型后，开始练习通过窄门

#### 3.3.3 训练超参数调整

QAT 训练通常需要调整超参数：

| 超参数 | 正常训练 | QAT 训练 | 说明 |
|--------|---------|---------|------|
| **学习率** | 5e-5 | 1e-5 ~ 2e-5 | QAT 需要更小学习率，避免量化误差放大 |
| **Warmup Steps** | 500 | 1000 | 更长预热，让权重分布稳定 |
| **Weight Decay** | 0.01 | 0.001 | 更小正则化，避免权重分布过于稀疏 |
| **Batch Size** | 32 | 64+ | 更大 batch size，减少量化噪声影响 |
| **训练步数** | 10,000 | 12,000 | 需要更多步数让权重适应量化约束 |

**经验法则**：
- 学习率降低 2-5 倍
- 训练步数增加 20-50%
- Batch size 增加有助于稳定训练

### 3.4 Per-Group Quantization 详解

#### 3.4.1 为什么需要 Per-Group？

**问题场景**：神经网络权重分布不均

```python
# 假设一个 Linear 层的权重
Weight: [4096, 4096]

# 如果用 Per-Tensor 量化
全局最大值: 3.2
全局最小值: -2.8
Scale: (3.2 - (-2.8)) / 15 = 0.4  # INT4: 16 levels

# 但实际上大部分权重在 [-0.5, 0.5] 范围内
量化精度: 0.4 (很粗糙！)

# 如果用 Per-Group 量化 (group_size=32)
每组 32 个元素独立量化
Group 1 范围: [-0.5, 0.5], Scale: 0.067
Group 2 范围: [-0.3, 0.4], Scale: 0.047
Group 3 范围: [-2.8, 3.2], Scale: 0.4  # 只有离群值组精度低
...
```

**优势**：大部分组的量化精度显著提升。

#### 3.4.2 Group Size 选择

| Group Size | 精度 | 计算开销 | 内存开销 | 适用场景 |
|-----------|------|---------|---------|---------|
| **Per-Tensor (∞)** | 最低 | 最小 | 最小 | 快速推理 |
| **128** | 中等 | 小 | 小 | 平衡性能和精度 |
| **32** | 高 | 中等 | 中等 | **推荐默认值** |
| **16** | 很高 | 较大 | 较大 | 精度敏感任务 |
| **Per-Channel** | 最高 | 大 | 大 | 训练阶段 |

**Axolotl 默认**：`group_size: 32`

**调整建议**：
- 模型精度要求高 → 减小 `group_size` (如 16)
- 推理速度优先 → 增大 `group_size` (如 128)
- INT4 量化 → 建议 32 或更小
- INT8 量化 → 可以用 128 甚至 Per-Tensor

#### 3.4.3 实现细节

```python
# torchao Per-Group Quantization 伪代码
def per_group_quantize(weight, group_size):
    """
    weight: [out_features, in_features]
    group_size: 每组元素数量
    """
    out_features, in_features = weight.shape
    num_groups = in_features // group_size

    # Reshape: [out_features, num_groups, group_size]
    weight_grouped = weight.view(out_features, num_groups, group_size)

    # 每组计算 scale
    scales = weight_grouped.abs().max(dim=-1).values / 7  # INT4: [-7, 7]

    # 量化每组
    weight_quant = torch.round(weight_grouped / scales.unsqueeze(-1))
    weight_quant = torch.clamp(weight_quant, -7, 7).to(torch.int8)

    # 反量化（fake quantization）
    weight_dequant = weight_quant * scales.unsqueeze(-1)

    # Reshape back
    return weight_dequant.view(out_features, in_features)
```

**内存布局**：

```
原始权重: [4096, 4096] FP16 = 32 MB

Per-Group 量化 (group_size=32):
├─ 量化权重: [4096, 4096] INT4 = 8 MB
├─ Scales: [4096, 4096/32] FP16 = 1 MB
└─ 总计: 9 MB (压缩 3.6x)
```

---

## 第四章：Axolotl 中的 QAT 实现

### 4.1 架构概览

Axolotl 的 QAT 实现基于 **torchao** 库，通过以下组件集成：

```
用户配置 (YAML)
    ↓
QATConfig Schema (Pydantic 验证)
    ↓
ModelLoader._configure_qat()
    ↓
prepare_model_for_qat() (替换 Linear/Embedding)
    ↓
HFCausalTrainerBuilder (添加 QATCallback)
    ↓
训练循环 (Fake Quantization 开关控制)
    ↓
训练完成后转换 (convert_qat_model)
    ↓
导出量化模型 (TorchAoConfig)
```

**设计理念**：
1. **声明式配置**：用户只需在 YAML 中配置 `qat` 字段
2. **自动集成**：Axolotl 自动处理模型替换和训练流程
3. **灵活控制**：支持延迟启用、embedding 量化等高级选项
4. **标准兼容**：导出符合 HuggingFace TorchAoConfig 格式

### 4.2 配置系统

#### 4.2.1 QATConfig Schema

文件：`src/axolotl/utils/schemas/quantization.py`

```python
class QATConfig(BaseModel):
    """QAT 配置模式"""

    activation_dtype: TorchAOQuantDType | None = Field(
        default=None,
        description="激活量化格式 (int8/fp8/None)",
    )

    weight_dtype: TorchAOQuantDType = Field(
        default=TorchAOQuantDType.int8,
        description="权重量化格式 (int4/int8/fp8)",
    )

    quantize_embedding: bool | None = Field(
        default=False,
        description="是否量化 Embedding 层",
    )

    group_size: int | None = Field(
        default=32,
        description="Per-group 量化的组大小",
    )

    fake_quant_after_n_steps: int | None = Field(
        default=None,
        description="在第几步后启用 fake quantization",
    )
```

**YAML 配置示例**：

```yaml
# INT4 Weight + INT8 Activation QAT
qat:
  weight_dtype: int4
  activation_dtype: int8
  group_size: 32
  quantize_embedding: true
  fake_quant_after_n_steps: 1000
```

**支持的 dtype 字符串**：

| 配置值 | 映射到 | 说明 |
|--------|--------|------|
| `int4` | `TorchAOQuantDType.int4` | 4-bit 整数 |
| `int8` | `TorchAOQuantDType.int8` | 8-bit 整数 |
| `fp8` / `float8` / `float8_e4m3fn` | `TorchAOQuantDType.float8_e4m3fn` | 8-bit 浮点 |
| `nvfp4` | `TorchAOQuantDType.nvfp4` | NVIDIA 4-bit 浮点 |

#### 4.2.2 配置验证

```python
# 不支持的组合会在配置验证时报错
qat:
  weight_dtype: int4
  activation_dtype: int4  # ❌ 不支持

# Error: Int4DynamicActivationInt4WeightConfig is not supported by torchao QAT.

qat:
  weight_dtype: int8
  activation_dtype: int8  # ❌ 不支持

# Error: Int8DynamicActivationInt8WeightConfig is not supported by torchao QAT.
```

**支持的组合**（参考 `get_quantization_config()`）：

| Weight | Activation | 支持 | Config 类 |
|--------|-----------|------|----------|
| INT4 | None | ✅ | `Int4WeightOnlyConfig` |
| INT4 | INT8 | ✅ | `Int8DynamicActivationInt4WeightConfig` |
| INT4 | FP8 | ✅ | `Float8DynamicActivationInt4WeightConfig` |
| INT8 | None | ❌ | - |
| INT8 | INT8 | ❌ | - |
| FP8 | FP8 | ✅ | `Float8DynamicActivationFloat8WeightConfig` |
| NVFP4 | None | ✅ | `NVFP4InferenceConfig` |

### 4.3 模型准备流程

#### 4.3.1 ModelLoader 集成

文件：`src/axolotl/loaders/model.py`

```python
class ModelLoader:
    def _configure_qat(self):
        """在模型加载后、LoRA 加载前配置 QAT"""
        if self.cfg.qat:
            from axolotl.utils.quantization import prepare_model_for_qat

            prepare_model_for_qat(
                self.model,
                self.cfg.qat.weight_dtype,
                self.cfg.qat.group_size,
                self.cfg.qat.activation_dtype,
                self.cfg.qat.quantize_embedding,
            )
```

**调用时机**：

```python
# ModelLoader 初始化流程
def __init__(self, ...):
    self._load_model()              # 1. 加载基础模型
    self._post_model_load()         # 2. 后处理（类型转换等）
    self._configure_qat()           # 3. 配置 QAT ⬅️
    self._load_adapters()           # 4. 加载 LoRA/adapter
```

**为什么在 LoRA 之前？**

QAT 需要替换所有 `nn.Linear` 层，包括 LoRA 会注入的层。如果先加载 LoRA，QAT 的替换逻辑可能会错过某些层。

#### 4.3.2 prepare_model_for_qat() 实现

文件：`src/axolotl/utils/quantization.py`

```python
def prepare_model_for_qat(
    model,
    weight_dtype: TorchAOQuantDType,
    group_size: int | None = None,
    activation_dtype: TorchAOQuantDType | None = None,
    quantize_embedding: bool = False,
):
    """准备模型进行 QAT 训练"""

    # 1. 构建 base config（决定量化策略）
    base_config = get_quantization_config(
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        group_size=group_size,
    )

    # 2. 包装为 QATConfig
    qat_config = QATConfig(base_config)

    # 3. 替换 Linear 层为 FakeQuantizedLinear
    quantize_(model, qat_config)

    # 4. 可选：替换 Embedding 层为 FakeQuantizedEmbedding
    if quantize_embedding:
        # Embedding 不支持 activation quantization
        embedding_base_config = get_quantization_config(
            weight_dtype=weight_dtype,
            activation_dtype=None,  # ⬅️ 强制 None
            group_size=group_size,
        )
        embedding_qat_config = QATConfig(embedding_base_config)
        quantize_(
            model,
            embedding_qat_config,
            filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding),
        )
```

**关键函数**：`torchao.quantization.quantize_()`

这是 torchao 库的核心 API，会递归遍历模型，根据 `filter_fn` 替换符合条件的层。

**替换逻辑**（伪代码）：

```python
def quantize_(model, config, filter_fn=None):
    """递归替换模型中的层"""
    for name, module in model.named_modules():
        # 默认 filter: nn.Linear 层
        if filter_fn is None:
            should_replace = isinstance(module, nn.Linear)
        else:
            should_replace = filter_fn(module, name)

        if should_replace:
            # 创建 FakeQuantizedLinear
            fake_quant_module = FakeQuantizedLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                weight_dtype=config.weight_dtype,
                activation_dtype=config.activation_dtype,
                group_size=config.group_size,
            )

            # 拷贝权重
            fake_quant_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                fake_quant_module.bias.data.copy_(module.bias.data)

            # 替换
            setattr(parent_module, child_name, fake_quant_module)
```

#### 4.3.3 Embedding 量化特殊处理

**为什么 Embedding 不支持 activation quantization？**

```python
# Embedding 层的前向传播
def embedding_forward(input_ids, weight):
    # input_ids: [batch, seq_len] - 整数索引
    # weight: [vocab_size, hidden_dim] - Embedding 矩阵

    # 查表操作
    output = weight[input_ids]  # [batch, seq_len, hidden_dim]

    # ❌ 没有"激活值"的概念
    # Embedding 的输出就是权重的某几行
    # 量化激活 = 量化权重，会重复量化
```

**正确做法**：

```yaml
qat:
  quantize_embedding: true
  weight_dtype: int4
  activation_dtype: null  # Embedding 忽略此配置
```

**Axolotl 的处理**：

```python
if quantize_embedding:
    # 强制 activation_dtype=None
    embedding_base_config = get_quantization_config(
        weight_dtype=weight_dtype,
        activation_dtype=None,  # ⬅️
        group_size=group_size,
    )
```

### 4.4 训练流程集成

#### 4.4.1 QATCallback

文件：`src/axolotl/utils/callbacks/qat.py`

```python
class QATCallback(TrainerCallback):
    """控制 Fake Quantization 的启用/禁用"""

    def __init__(self, cfg: QATConfig):
        self.cfg = cfg

    def on_step_begin(self, args, state, control, model, **kwargs):
        """每步开始时检查是否需要切换 fake quantization"""

        if self.cfg.fake_quant_after_n_steps is not None:
            # 第 0 步：禁用 fake quantization
            if state.global_step == 0:
                LOG.info(f"Disabling fake quantization at step {state.global_step}")
                model.apply(partial(toggle_fake_quant, enable=False))

            # 第 N 步：启用 fake quantization
            elif state.global_step == self.cfg.fake_quant_after_n_steps:
                LOG.info(f"Enabling fake quantization at step {state.global_step}")
                model.apply(partial(toggle_fake_quant, enable=True))
```

**toggle_fake_quant() 函数**：

```python
def toggle_fake_quant(mod: nn.Module, enable: bool):
    """递归切换模型中所有 fake quantizer 的状态"""

    if isinstance(mod, (FakeQuantizedLinear, FakeQuantizedEmbedding)):
        # 切换 activation fake quantizer
        if (
            isinstance(mod, FakeQuantizedLinear)
            and mod.activation_fake_quantizer is not None
        ):
            mod.activation_fake_quantizer.enabled = enable

        # 切换 weight fake quantizer
        mod.weight_fake_quantizer.enabled = enable
```

**状态切换示例**：

```python
# Step 0
model.apply(partial(toggle_fake_quant, enable=False))
# FakeQuantizedLinear.weight_fake_quantizer.enabled = False
# FakeQuantizedLinear.activation_fake_quantizer.enabled = False
# ➜ 等价于普通 Linear 层

# Step 1000
model.apply(partial(toggle_fake_quant, enable=True))
# FakeQuantizedLinear.weight_fake_quantizer.enabled = True
# FakeQuantizedLinear.activation_fake_quantizer.enabled = True
# ➜ 开始模拟量化
```

#### 4.4.2 添加 Callback 到 Trainer

文件：`src/axolotl/core/builders/causal.py`

```python
class HFCausalTrainerBuilder(TrainerBuilderBase):
    def hook_pre_create_training_args(self, **kwargs):
        """在创建 Trainer 前注册 callbacks"""
        callbacks = kwargs.get("callbacks", [])

        # 添加 QATCallback
        if self.cfg.qat:
            callbacks.append(QATCallback(self.cfg.qat))

        kwargs["callbacks"] = callbacks
        return kwargs
```

**训练流程**：

```
Trainer 初始化
├─ 加载模型（已被 prepare_model_for_qat 替换）
├─ 注册 QATCallback
└─ 开始训练

Step 0:
├─ QATCallback.on_step_begin()
├─ 禁用 fake quantization
├─ 前向传播（正常 FP32/BF16）
└─ 反向传播

Step 1 - 999:
├─ Fake quantization 保持禁用
└─ 正常训练

Step 1000:
├─ QATCallback.on_step_begin()
├─ 启用 fake quantization
├─ 前向传播（模拟 INT4/INT8）
└─ 反向传播（STE 梯度）

Step 1001+:
├─ Fake quantization 保持启用
└─ QAT 训练
```

### 4.5 转换和导出

#### 4.5.1 训练后转换

训练完成后，需要将 `FakeQuantizedLinear` 转换为实际量化层：

```python
from axolotl.utils.quantization import convert_qat_model

# 转换模型
convert_qat_model(
    model,
    quantize_embedding=cfg.qat.quantize_embedding,
)
```

**convert_qat_model() 实现**：

```python
def convert_qat_model(
    model,
    quantize_embedding: bool = False,
):
    """将 FakeQuantizedLinear 转换为实际量化层"""

    # 创建 convert config
    config = QATConfig(step="convert")  # ⬅️ 特殊参数

    # 转换 Linear 层
    quantize_(model, config)

    # 转换 Embedding 层
    if quantize_embedding:
        quantize_(
            model,
            config,
            filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding),
        )
```

**torchao 的 "convert" 模式**：

```python
# step="convert" 时，quantize_() 的行为
if config.step == "convert":
    # 移除 fake quantizer
    # 将 FP32 权重量化为 INT4/INT8
    # 替换为实际量化层（如 AffineQuantizedLinear）
```

**转换前后对比**：

```
训练阶段:
FakeQuantizedLinear
├─ weight: [4096, 4096] FP32 = 64 MB
├─ weight_fake_quantizer: 量化参数
└─ activation_fake_quantizer: 量化参数

转换后（推理阶段）:
AffineQuantizedLinear
├─ weight: [4096, 4096] INT4 = 8 MB  ⬅️ 实际量化
├─ scales: [4096, 128] FP16 = 1 MB
└─ zero_points: [4096, 128] INT4 = 0.25 MB
总计: 9.25 MB (压缩 6.9x)
```

#### 4.5.2 导出到 HuggingFace 格式

文件：`src/axolotl/cli/quantize.py`

```python
def do_quantize(config, cli_args):
    """导出量化模型"""

    # 1. 加载训练好的模型
    model = AutoModelForCausalLM.from_pretrained(cfg.output_dir, ...)

    # 2. 应用 PTQ（如果从 QAT checkpoint 继续）
    # 或直接从 QAT 训练的 checkpoint 加载

    # 3. 构建 TorchAoConfig
    quantization_config = get_quantization_config(
        weight_dtype, activation_dtype, group_size
    )

    ao_config = TorchAoConfig(
        quant_type=quantization_config,
        include_input_output_embeddings=quantize_embedding,
    )

    # 4. 保存配置到 model.config
    model.config.quantization_config = ao_config

    # 5. 导出
    model.save_pretrained(output_dir / "quantized")
    tokenizer.save_pretrained(output_dir / "quantized")

    # 6. 可选：上传到 HuggingFace Hub
    if hub_model_id:
        model.push_to_hub(hub_model_id)
```

**导出后的文件结构**：

```
quantized/
├── config.json  # 包含 quantization_config
├── model.safetensors  # 量化后的权重
├── tokenizer_config.json
└── ...

# config.json 中的 quantization_config
{
  "quantization_config": {
    "quant_type": "int8int4",
    "include_input_output_embeddings": true
  }
}
```

**加载量化模型**：

```python
# HuggingFace Transformers 会自动识别 TorchAoConfig
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "path/to/quantized",
    device_map="auto"
)

# 模型会自动加载为量化版本
print(model.model.layers[0].mlp.gate_proj)
# Output: AffineQuantizedLinear(in_features=4096, out_features=11008, dtype=int4)
```

---

## 第五章：源码解析

### 5.1 QATCallback 完整源码

文件：`src/axolotl/utils/callbacks/qat.py`

```python
"""QAT Callback for HF Causal Trainer"""

from functools import partial

from torch import nn
from torchao.quantization.qat.embedding import FakeQuantizedEmbedding
from torchao.quantization.qat.linear import FakeQuantizedLinear
from transformers import TrainerCallback

from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.quantization import QATConfig

LOG = get_logger(__name__)


def toggle_fake_quant(mod: nn.Module, enable: bool):
    """
    Toggle fake quantization for any fake quantized linear or embedding layers in the model.

    Args:
        mod: The module to toggle fake quantization for.
        enable: Whether to enable or disable fake quantization.
    """
    if isinstance(mod, (FakeQuantizedLinear, FakeQuantizedEmbedding)):
        # 1. 切换 activation fake quantizer（仅 Linear 层）
        if (
            isinstance(mod, FakeQuantizedLinear)
            and mod.activation_fake_quantizer is not None
        ):
            mod.activation_fake_quantizer.enabled = enable

        # 2. 切换 weight fake quantizer（Linear 和 Embedding 都有）
        mod.weight_fake_quantizer.enabled = enable


class QATCallback(TrainerCallback):
    """
    Callback to toggle fake quantization for the model.
    """

    def __init__(self, cfg: QATConfig):
        self.cfg = cfg

    def on_step_begin(self, args, state, control, model, **kwargs):
        """在每步开始时检查是否需要切换 fake quantization 状态"""

        if self.cfg.fake_quant_after_n_steps is not None:
            # Step 0: 禁用 fake quantization
            if state.global_step == 0:
                LOG.info(f"Disabling fake quantization at step {state.global_step}")
                model.apply(partial(toggle_fake_quant, enable=False))

            # Step N: 启用 fake quantization
            elif state.global_step == self.cfg.fake_quant_after_n_steps:
                LOG.info(f"Enabling fake quantization at step {state.global_step}")
                model.apply(partial(toggle_fake_quant, enable=True))
```

**代码亮点**：

1. **functools.partial**：优雅地传递 `enable` 参数给 `model.apply()`
2. **类型检查**：区分 `FakeQuantizedLinear` 和 `FakeQuantizedEmbedding`
3. **条件控制**：只在 `fake_quant_after_n_steps` 配置时才切换状态
4. **日志记录**：便于调试和监控

### 5.2 quantization.py 核心函数

文件：`src/axolotl/utils/quantization.py`

#### 5.2.1 get_quantization_config()

```python
def get_quantization_config(
    weight_dtype: TorchAOQuantDType,
    activation_dtype: TorchAOQuantDType | None = None,
    group_size: int | None = None,
) -> AOBaseConfig:
    """
    构建 torchao 量化配置对象

    根据 weight_dtype 和 activation_dtype 组合，返回对应的 Config 类。
    """

    # Case 1: 仅权重量化
    if activation_dtype is None:
        if weight_dtype == TorchAOQuantDType.int8:
            # INT8 仅权重量化不支持 QAT
            raise ValueError("Int8WeightOnlyConfig is not supported by torchao QAT.")

        if weight_dtype == TorchAOQuantDType.int4:
            # INT4 仅权重量化（最常用）
            from torchao.quantization.quant_api import Int4WeightOnlyConfig

            if group_size is not None:
                return Int4WeightOnlyConfig(group_size=group_size, version=2)
            else:
                return Int4WeightOnlyConfig(version=2)

    # Case 2: INT4/INT4 不支持
    if (
        activation_dtype == TorchAOQuantDType.int4
        and weight_dtype == TorchAOQuantDType.int4
    ):
        raise ValueError(
            "Int4DynamicActivationInt4WeightConfig is not supported by torchao QAT."
        )

    # Case 3: INT8/INT8 不支持
    if (
        activation_dtype == TorchAOQuantDType.int8
        and weight_dtype == TorchAOQuantDType.int8
    ):
        raise ValueError(
            "Int8DynamicActivationInt8WeightConfig is not supported by torchao QAT."
        )

    # Case 4: INT8 激活 + INT4 权重（推荐组合）
    if (
        activation_dtype == TorchAOQuantDType.int8
        and weight_dtype == TorchAOQuantDType.int4
    ):
        if group_size is not None:
            return Int8DynamicActivationInt4WeightConfig(group_size=group_size)
        else:
            return Int8DynamicActivationInt4WeightConfig()

    # Case 5: FP8 激活 + FP8 权重
    if (
        activation_dtype == TorchAOQuantDType.float8_e4m3fn
        and weight_dtype == TorchAOQuantDType.float8_e4m3fn
    ):
        return Float8DynamicActivationFloat8WeightConfig()

    # Case 6: FP8 激活 + INT4 权重
    if (
        activation_dtype == TorchAOQuantDType.float8_e4m3fn
        and weight_dtype == TorchAOQuantDType.int4
    ):
        return Float8DynamicActivationInt4WeightConfig()

    # Case 7: NVFP4（实验性）
    if weight_dtype == TorchAOQuantDType.nvfp4:
        from torchao.prototype.mx_formats import NVFP4InferenceConfig

        if group_size is not None and group_size != 16:
            raise ValueError("NVFP4 quantization must use a group_size of 16")
        return NVFP4InferenceConfig()

    # 不支持的组合
    raise ValueError(
        f"Invalid activation/weight dtype combination: {activation_dtype}/{weight_dtype}"
    )
```

**设计模式**：
- **Factory Pattern**：根据输入参数返回不同的 Config 对象
- **Fail-fast**：不支持的组合立即抛出异常，避免后续错误
- **条件导入**：只在需要时 import，避免依赖问题

#### 5.2.2 prepare_model_for_qat()

```python
def prepare_model_for_qat(
    model,
    weight_dtype: TorchAOQuantDType,
    group_size: int | None = None,
    activation_dtype: TorchAOQuantDType | None = None,
    quantize_embedding: bool = False,
):
    """
    准备模型进行 QAT 训练

    步骤:
    1. 构建 base quantization config
    2. 包装为 QATConfig
    3. 替换 Linear 层为 FakeQuantizedLinear
    4. 可选替换 Embedding 层为 FakeQuantizedEmbedding
    """

    # 1. 构建 base config
    base_config = get_quantization_config(
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype,
        group_size=group_size,
    )

    # 2. 包装为 QATConfig
    qat_config = QATConfig(base_config)

    # 3. 替换 Linear 层
    quantize_(model, qat_config)

    # 4. 替换 Embedding 层（如果启用）
    if quantize_embedding:
        # Embedding 不支持 activation quantization
        embedding_base_config = get_quantization_config(
            weight_dtype=weight_dtype,
            activation_dtype=None,  # ⬅️ 强制 None
            group_size=group_size,
        )
        embedding_qat_config = QATConfig(embedding_base_config)

        # 仅替换 Embedding 层
        quantize_(
            model,
            embedding_qat_config,
            filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding),
        )
```

**关键点**：
- `quantize_(model, qat_config)` 会递归遍历模型，替换所有 `nn.Linear`
- Embedding 的 `activation_dtype` 必须是 `None`
- `filter_fn` 用于精确控制替换哪些层

#### 5.2.3 convert_qat_model()

```python
def convert_qat_model(
    model,
    quantize_embedding: bool = False,
):
    """
    将 QAT 训练的模型转换为实际量化模型

    步骤:
    1. 创建 "convert" 模式的 QATConfig
    2. 转换 Linear 层（FakeQuantizedLinear → AffineQuantizedLinear）
    3. 转换 Embedding 层（如果启用）
    """

    # "convert" 是 torchao QATConfig 的特殊参数
    config = QATConfig(step="convert")

    # 转换 Linear 层
    quantize_(model, config)

    # 转换 Embedding 层
    if quantize_embedding:
        quantize_(
            model,
            config,
            filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding),
        )
```

**torchao 的 "convert" 机制**：

当 `QATConfig(step="convert")` 时，`quantize_()` 会：
1. 识别 `FakeQuantizedLinear` 层
2. 从 `weight_fake_quantizer` 中提取量化参数（scale, zero_point）
3. 将 FP32 权重量化为 INT4/INT8
4. 创建 `AffineQuantizedLinear` 并替换原层

### 5.3 torchao 核心组件解析

虽然 torchao 是外部库，但理解其核心组件有助于调试和优化。

#### 5.3.1 FakeQuantizedLinear 结构

```python
# torchao.quantization.qat.linear
class FakeQuantizedLinear(nn.Module):
    """
    模拟量化的 Linear 层

    前向传播:
    1. 量化权重（FP32 → INT4 → FP32）
    2. 量化激活（FP32 → INT8 → FP32）
    3. 执行线性计算

    反向传播:
    使用 Straight-Through Estimator 传递梯度
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_dtype: torch.dtype = torch.int4,
        activation_dtype: torch.dtype | None = torch.int8,
        group_size: int = 32,
    ):
        super().__init__()

        # 主权重（始终 FP32/BF16，用于梯度更新）
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # 权重 fake quantizer
        self.weight_fake_quantizer = WeightFakeQuantizer(
            dtype=weight_dtype,
            group_size=group_size,
            enabled=True,  # 默认启用
        )

        # 激活 fake quantizer（可选）
        if activation_dtype is not None:
            self.activation_fake_quantizer = ActivationFakeQuantizer(
                dtype=activation_dtype,
                enabled=True,
            )
        else:
            self.activation_fake_quantizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 量化权重
        if self.weight_fake_quantizer.enabled:
            weight = self.weight_fake_quantizer(self.weight)
        else:
            weight = self.weight

        # 2. 量化激活
        if self.activation_fake_quantizer and self.activation_fake_quantizer.enabled:
            x = self.activation_fake_quantizer(x)

        # 3. 线性计算
        return F.linear(x, weight, self.bias)
```

#### 5.3.2 WeightFakeQuantizer

```python
class WeightFakeQuantizer(nn.Module):
    """
    权重 Fake Quantizer

    支持:
    - Per-group quantization
    - 对称/非对称量化
    - INT4/INT8/FP8
    """

    def __init__(self, dtype, group_size, enabled=True):
        super().__init__()
        self.dtype = dtype
        self.group_size = group_size
        self.enabled = enabled

        # 量化范围
        if dtype == torch.int4:
            self.quant_min = -7
            self.quant_max = 7
        elif dtype == torch.int8:
            self.quant_min = -127
            self.quant_max = 127
        else:  # FP8
            # FP8 的范围由格式定义
            pass

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        weight: [out_features, in_features]
        """
        if not self.enabled:
            return weight

        # Reshape for per-group quantization
        # [out_features, in_features] → [out_features, num_groups, group_size]
        out_features, in_features = weight.shape
        num_groups = in_features // self.group_size
        weight_grouped = weight.view(out_features, num_groups, self.group_size)

        # 计算每组的 scale（对称量化）
        # scale = max(|weight|) / quant_max
        scales = weight_grouped.abs().amax(dim=-1, keepdim=True) / self.quant_max
        scales = torch.clamp(scales, min=1e-5)  # 避免除零

        # 量化
        weight_quant = torch.round(weight_grouped / scales)
        weight_quant = torch.clamp(weight_quant, self.quant_min, self.quant_max)

        # 反量化（回到 FP32）
        weight_dequant = weight_quant * scales

        # Reshape back
        return weight_dequant.view(out_features, in_features)
```

**数学推导**：

```
输入: W ∈ ℝ^(m×n), group_size = g
输出: W_dequant ∈ ℝ^(m×n)

1. Reshape: W → W_grouped ∈ ℝ^(m × n/g × g)

2. 计算 scale (每组):
   s_ij = max(|W_grouped[i,j,:]|) / 7  # INT4: [-7, 7]

3. 量化:
   W_quant[i,j,k] = clamp(round(W_grouped[i,j,k] / s_ij), -7, 7)

4. 反量化:
   W_dequant[i,j,k] = W_quant[i,j,k] * s_ij

5. Reshape back: W_dequant → ℝ^(m×n)
```

#### 5.3.3 Straight-Through Estimator 实现

```python
class STERound(torch.autograd.Function):
    """
    Straight-Through Estimator for round()

    前向: y = round(x)
    反向: dy/dx = 1 (直通)
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 梯度直接传递，忽略 round 的不可微性
        return grad_output

# 在 WeightFakeQuantizer 中使用
weight_quant = STERound.apply(weight_grouped / scales)
```

**为什么 STE 有效？**

虽然 `round()` 在数学上不可微，但在 QAT 中：
1. 权重更新时，大部分权重变化幅度远小于量化步长
2. STE 近似假设：`round(x + δx) ≈ round(x) + δx`（当 `δx` 很小时）
3. 实验表明这个近似在 QAT 中工作良好

**搬桌子类比**：
- 前向：测量桌子能否通过窄门（离散判断）
- 反向：根据"通过窄门的难度"，连续地微调桌子尺寸

### 5.4 模型转换细节

#### 5.4.1 FakeQuantizedLinear → AffineQuantizedLinear

转换时，torchao 执行以下步骤：

```python
# 伪代码
def convert_fake_to_real(fake_quant_linear: FakeQuantizedLinear):
    """
    转换 FakeQuantizedLinear 为 AffineQuantizedLinear
    """

    # 1. 提取权重和量化参数
    weight_fp32 = fake_quant_linear.weight.data  # [out, in] FP32
    group_size = fake_quant_linear.weight_fake_quantizer.group_size

    # 2. 计算量化参数（与训练时相同）
    out_features, in_features = weight_fp32.shape
    num_groups = in_features // group_size
    weight_grouped = weight_fp32.view(out_features, num_groups, group_size)

    # 计算 scales
    scales = weight_grouped.abs().amax(dim=-1) / 7  # INT4: [-7, 7]
    scales = torch.clamp(scales, min=1e-5)

    # 3. 量化权重（实际量化，不再反量化）
    weight_quant = torch.round(weight_grouped / scales.unsqueeze(-1))
    weight_quant = torch.clamp(weight_quant, -7, 7).to(torch.int8)  # 存储为 INT8

    # 4. 打包为 INT4（2个 INT4 值打包到 1个 UINT8）
    # weight_quant: [out, num_groups, group_size] INT8
    # → weight_packed: [out, num_groups, group_size//2] UINT8
    weight_packed = pack_int4(weight_quant)

    # 5. 创建 AffineQuantizedLinear
    affine_quant_linear = AffineQuantizedLinear(
        in_features=in_features,
        out_features=out_features,
        bias=fake_quant_linear.bias is not None,
        group_size=group_size,
    )

    # 6. 拷贝量化权重和参数
    affine_quant_linear.weight_packed.data.copy_(weight_packed)
    affine_quant_linear.scales.data.copy_(scales)
    if fake_quant_linear.bias is not None:
        affine_quant_linear.bias.data.copy_(fake_quant_linear.bias.data)

    return affine_quant_linear
```

**INT4 打包**：

```python
def pack_int4(x: torch.Tensor) -> torch.Tensor:
    """
    将 INT8 存储的 INT4 值打包为 UINT8

    输入: [N] INT8 (值范围 [-7, 7])
    输出: [N//2] UINT8 (每个字节存储 2 个 INT4 值)
    """
    # 转换到 [0, 15] 范围
    x_unsigned = x + 7  # [-7, 7] → [0, 14]

    # 打包：高 4 位存储第一个值，低 4 位存储第二个值
    x_packed = (x_unsigned[::2] << 4) | x_unsigned[1::2]

    return x_packed.to(torch.uint8)

# 示例
x = torch.tensor([-7, -3, 0, 4, 7], dtype=torch.int8)
x_unsigned = x + 7  # [0, 4, 7, 11, 14]
x_packed = (x_unsigned[::2] << 4) | x_unsigned[1::2]
# [0<<4 | 4, 7<<4 | 11] = [0x04, 0x7B]
```

**内存节省**：

```
FakeQuantizedLinear:
├─ weight: [4096, 4096] FP32 = 64 MB
└─ 其他参数: ~1 MB
总计: 65 MB

AffineQuantizedLinear:
├─ weight_packed: [4096, 4096//2] UINT8 = 8 MB  # INT4 打包
├─ scales: [4096, 4096//32] FP16 = 1 MB
└─ bias: [4096] FP32 = 16 KB
总计: 9 MB (压缩 7.2x)
```

#### 5.4.2 AffineQuantizedLinear 推理

```python
class AffineQuantizedLinear(nn.Module):
    """
    实际量化的 Linear 层（推理用）
    """

    def __init__(self, in_features, out_features, group_size, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # 量化权重（INT4 打包为 UINT8）
        num_groups = in_features // group_size
        self.register_buffer(
            'weight_packed',
            torch.empty(out_features, num_groups, group_size // 2, dtype=torch.uint8)
        )

        # Scales（每组一个）
        self.register_buffer(
            'scales',
            torch.empty(out_features, num_groups, dtype=torch.float16)
        )

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        高效推理前向传播
        """
        # 1. 解包 INT4 权重
        weight_int4 = unpack_int4(self.weight_packed)  # [out, num_groups, group_size]

        # 2. 反量化（INT4 → FP16）
        weight_fp16 = (weight_int4 - 7).to(torch.float16) * self.scales.unsqueeze(-1)

        # 3. Reshape
        weight_fp16 = weight_fp16.view(self.out_features, self.in_features)

        # 4. 线性计算（FP16）
        return F.linear(x, weight_fp16, self.bias)
```

**推理优化**：

实际的 `AffineQuantizedLinear` 会使用 CUDA kernel 优化：
- **Fused kernel**：解包 + 反量化 + 矩阵乘法融合为一个 kernel
- **INT4 GEMM**：直接使用 INT4 进行矩阵乘法（Ampere+ GPU）
- **Tensor Core**：利用 INT4 Tensor Core 加速（H100）

**性能对比**：

```
Llama-3-8B 推理 (A100 GPU)

FP16:
├─ 延迟: 42 ms/token
├─ 显存: 16 GB
└─ 吞吐: 24 tokens/s

INT4 (AffineQuantizedLinear):
├─ 延迟: 22 ms/token  (1.9x faster)
├─ 显存: 6 GB  (2.7x smaller)
└─ 吞吐: 45 tokens/s  (1.9x higher)
```

---

## 第六章：实战示例

### 6.1 基础 QAT 训练

#### 6.1.1 配置文件

文件：`examples/qat/llama3-qat-int4.yml`

```yaml
# 基础模型配置
base_model: meta-llama/Meta-Llama-3-8B
model_type: llama
tokenizer_type: AutoTokenizer

# 数据集
datasets:
  - path: mlabonne/FineTome-100k
    type: chat_template
    field_messages: conversations
    message_property_mappings:
      role: from
      content: value
    drop_system_message: true
    split: train[:10%]  # 使用 10% 数据

chat_template: llama3

# QAT 配置 ⬅️ 核心部分
qat:
  weight_dtype: int4            # 权重量化为 INT4
  activation_dtype: int8        # 激活量化为 INT8
  group_size: 32                # Per-group 量化，每组 32 个元素
  quantize_embedding: true      # 量化 Embedding 层
  fake_quant_after_n_steps: 1000  # 第 1000 步后启用量化

# 训练超参数（QAT 调整）
sequence_len: 2048
num_epochs: 2
micro_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 0.00001  # ⬅️ 比正常训练低 5 倍
warmup_steps: 1000      # ⬅️ 更长的预热
lr_scheduler: cosine
optimizer: adamw_bnb_8bit

# 性能优化
flash_attention: true
bf16: true              # 使用 BF16 混合精度
gradient_checkpointing: true

# 保存配置
output_dir: ./outputs/llama3-qat-int4
save_strategy: steps
save_steps: 500
save_total_limit: 3
```

#### 6.1.2 启动训练

```bash
# 单 GPU 训练
axolotl train examples/qat/llama3-qat-int4.yml

# 多 GPU 训练 (FSDP)
accelerate launch -m axolotl.cli.train examples/qat/llama3-qat-int4.yml \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"

# 多 GPU 训练 (DeepSpeed ZeRO-3)
accelerate launch -m axolotl.cli.train examples/qat/llama3-qat-int4.yml \
    --deepspeed ds_config_zero3.json
```

#### 6.1.3 训练日志解读

```
Step 0:
├─ [INFO] Disabling fake quantization at step 0
├─ [INFO] Training with FP32/BF16 (warmup)
└─ Loss: 2.456

Step 500:
├─ Loss: 1.234
└─ (仍然是 FP32/BF16 训练)

Step 1000:
├─ [INFO] Enabling fake quantization at step 1000
├─ [INFO] Starting QAT training
└─ Loss: 1.345  ⬅️ Loss 可能略微上升（量化误差）

Step 1100:
├─ Loss: 1.312  ⬅️ 模型开始适应量化约束
└─ (QAT 训练中)

Step 2000:
├─ Loss: 0.987
└─ (QAT 训练中)

Training completed!
```

**关键观察**：
- Step 1000 启用 QAT 后，loss 可能略微上升（5-10%）
- 随后 loss 会逐渐下降，模型适应量化约束
- 最终 loss 应接近 FP32 训练的 loss

### 6.2 LoRA + QAT

#### 6.2.1 配置文件

```yaml
base_model: meta-llama/Meta-Llama-3-8B
model_type: llama

# LoRA 配置
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# QAT 配置
qat:
  weight_dtype: int4
  activation_dtype: int8
  group_size: 32
  quantize_embedding: false  # ⬅️ Embedding 不量化（LoRA 不修改 Embedding）
  fake_quant_after_n_steps: 500  # ⬅️ LoRA 训练更快，提前启用

# 数据集
datasets:
  - path: mlabonne/FineTome-100k
    type: chat_template
    split: train[:5%]

# 训练超参数
sequence_len: 2048
num_epochs: 3
micro_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 0.0002  # ⬅️ LoRA 用更高学习率
warmup_steps: 500
lr_scheduler: cosine
optimizer: adamw_8bit

# 其他配置
flash_attention: true
bf16: true
output_dir: ./outputs/llama3-lora-qat-int4
```

#### 6.2.2 训练和部署

```bash
# 1. 训练 LoRA + QAT
axolotl train examples/qat/llama3-lora-qat.yml

# 2. 转换模型（移除 fake quantizers）
python -c "
from axolotl.utils.quantization import convert_qat_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    './outputs/llama3-lora-qat-int4/checkpoint-final',
    device_map='cpu'
)
convert_qat_model(model, quantize_embedding=False)
model.save_pretrained('./outputs/llama3-lora-qat-int4/quantized')
"

# 3. 推理测试
axolotl inference examples/qat/llama3-lora-qat.yml \
    --lora_model_dir="./outputs/llama3-lora-qat-int4/quantized"
```

**LoRA + QAT 优势**：
- 训练更快（只更新 LoRA 参数 + 量化参数）
- 显存占用更小
- 适合快速实验和迭代

### 6.3 DPO + QAT

#### 6.3.1 配置文件

```yaml
base_model: HuggingFaceTB/SmolLM2-135M
model_type: llama

# DPO 配置
rl: dpo
chat_template: chatml

# 数据集（偏好数据）
datasets:
  - path: fozziethebeat/alpaca_messages_2k_dpo_test
    type: chat_template.default
    field_messages: conversation
    field_chosen: chosen
    field_rejected: rejected
    message_field_role: role
    message_field_content: content

# QAT 配置
qat:
  weight_dtype: int4
  activation_dtype: int8
  group_size: 8  # ⬅️ 小模型用更小的 group_size
  quantize_embedding: true
  fake_quant_after_n_steps: null  # ⬅️ 从一开始就启用 QAT

# 训练超参数
sequence_len: 2048
sample_packing: false
pad_to_sequence_len: true
val_set_size: 0.01
num_epochs: 1
max_steps: 5000
micro_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.00001
warmup_steps: 500
optimizer: adamw_torch_fused
lr_scheduler: cosine

# 其他配置
flash_attention: true
bf16: true
output_dir: ./outputs/smollm-dpo-qat-int4
```

#### 6.3.2 训练流程

```bash
# 训练
axolotl train examples/qat/smollm-dpo-qat.yml

# 评估
axolotl eval examples/qat/smollm-dpo-qat.yml \
    --checkpoint_dir="./outputs/smollm-dpo-qat-int4/checkpoint-final"
```

**DPO + QAT 注意事项**：
- DPO 对量化误差更敏感（需要精确计算 chosen/rejected log probs）
- 建议使用更小的学习率
- 可能需要更多训练步数才能收敛

### 6.4 FP8 QAT（H100）

#### 6.4.1 配置文件

```yaml
base_model: meta-llama/Meta-Llama-3-70B
model_type: llama

# FP8 QAT 配置
qat:
  weight_dtype: fp8          # ⬅️ FP8 权重
  activation_dtype: fp8      # ⬅️ FP8 激活
  group_size: null           # ⬅️ FP8 不使用 per-group
  quantize_embedding: false  # ⬅️ Embedding 保持 BF16
  fake_quant_after_n_steps: 2000

# 数据集
datasets:
  - path: mlabonne/FineTome-100k
    type: chat_template
    split: train[:20%]

# 训练超参数
sequence_len: 4096
num_epochs: 1
micro_batch_size: 1
gradient_accumulation_steps: 64
learning_rate: 0.00003
warmup_steps: 2000
lr_scheduler: cosine
optimizer: adamw_torch_fused

# 分布式训练（FSDP）
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer

# 其他配置
flash_attention: true
bf16: true
output_dir: ./outputs/llama3-70b-fp8-qat
```

#### 6.4.2 H100 优化

```bash
# 启用 FP8 Transformer Engine (NVIDIA)
export NVTE_FLASH_ATTN=1
export NVTE_FP8_DPA_BWD=1

# 启动训练（8x H100）
torchrun --nproc_per_node=8 -m axolotl.cli.train \
    examples/qat/llama3-70b-fp8-qat.yml
```

**FP8 QAT 性能**：

```
Llama-3-70B 训练 (8x H100)

BF16:
├─ 吞吐: 1200 tokens/s
├─ 显存: 72 GB/GPU
└─ 训练时间: 48 hours

FP8 QAT:
├─ 吞吐: 1800 tokens/s  (1.5x faster)
├─ 显存: 48 GB/GPU  (1.5x smaller)
└─ 训练时间: 32 hours  (1.5x faster)

精度对比:
├─ BF16 Perplexity: 8.23
└─ FP8 QAT Perplexity: 8.31  (仅下降 1%)
```

### 6.5 分阶段 QAT 训练

#### 6.5.1 策略：先 FP32 预训练，再 QAT 微调

```yaml
# Stage 1: 正常训练（保存 checkpoint）
base_model: meta-llama/Meta-Llama-3-8B
datasets:
  - path: mlabonne/FineTome-100k
    split: train[:50%]

num_epochs: 3
learning_rate: 0.0001
output_dir: ./outputs/llama3-pretrain

# (不配置 qat)
```

```yaml
# Stage 2: QAT 微调（从 checkpoint 继续）
base_model: ./outputs/llama3-pretrain/checkpoint-final
datasets:
  - path: mlabonne/FineTome-100k
    split: train[:10%]  # ⬅️ 使用更少数据

# QAT 配置
qat:
  weight_dtype: int4
  activation_dtype: int8
  group_size: 32
  quantize_embedding: true
  fake_quant_after_n_steps: null  # ⬅️ 立即启用

num_epochs: 1  # ⬅️ 更少 epochs
learning_rate: 0.00001  # ⬅️ 更小学习率
output_dir: ./outputs/llama3-qat-finetune
```

**训练命令**：

```bash
# Stage 1: 正常训练
axolotl train stage1-pretrain.yml

# Stage 2: QAT 微调
axolotl train stage2-qat-finetune.yml
```

**优势**：
- Stage 1 训练更快（无量化开销）
- Stage 2 只需少量数据即可适应量化约束
- 适合大规模预训练 + 量化部署的场景

### 6.6 导出和推理

#### 6.6.1 导出量化模型

```bash
# 方法 1: 使用 axolotl CLI
axolotl quantize examples/qat/llama3-qat-int4.yml \
    --output_dir ./outputs/llama3-qat-int4/quantized

# 方法 2: 使用 Python 脚本
python -c "
from axolotl.utils.quantization import convert_qat_model, quantize_model
from transformers import AutoModelForCausalLM, TorchAoConfig

# 加载 QAT 训练的模型
model = AutoModelForCausalLM.from_pretrained(
    './outputs/llama3-qat-int4/checkpoint-final',
    device_map='cpu'
)

# 转换为实际量化模型
convert_qat_model(model, quantize_embedding=True)

# 构建 TorchAoConfig
from axolotl.utils.quantization import get_quantization_config, TorchAOQuantDType

quantization_config = get_quantization_config(
    weight_dtype=TorchAOQuantDType.int4,
    activation_dtype=TorchAOQuantDType.int8,
    group_size=32,
)

ao_config = TorchAoConfig(
    quant_type=quantization_config,
    include_input_output_embeddings=True,
)

# 保存配置
model.config.quantization_config = ao_config

# 导出
model.save_pretrained('./outputs/llama3-qat-int4/quantized')
"
```

#### 6.6.2 推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "./outputs/llama3-qat-int4/quantized",
    device_map="auto",  # 自动分配到 GPU
)

tokenizer = AutoTokenizer.from_pretrained("./outputs/llama3-qat-int4/quantized")

# 推理
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### 6.6.3 上传到 HuggingFace Hub

```bash
# 上传
axolotl quantize examples/qat/llama3-qat-int4.yml \
    --output_dir ./outputs/llama3-qat-int4/quantized \
    --hub_model_id "username/llama3-8b-qat-int4"

# 模型会自动命名为: username/llama3-8b-qat-int4-int8int4
```

**加载已发布的模型**：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "username/llama3-8b-qat-int4-int8int4",
    device_map="auto"
)
# 自动加载量化配置并应用
```

---

## 第七章：常见问题与最佳实践

### 7.1 常见问题排查

#### 问题 1：QAT 训练后精度显著下降

**症状**：

```
FP16 Baseline: Perplexity 8.23, Accuracy 76.5%
QAT INT4:      Perplexity 15.67, Accuracy 62.3%  ❌ 精度下降太多
```

**可能原因**：

1. **学习率过高**
   ```yaml
   # ❌ 错误
   learning_rate: 0.0001  # QAT 应该用更小学习率

   # ✅ 正确
   learning_rate: 0.00001  # 降低 5-10 倍
   ```

2. **过早启用 fake quantization**
   ```yaml
   # ❌ 错误
   fake_quant_after_n_steps: 100  # 权重还没收敛

   # ✅ 正确
   fake_quant_after_n_steps: 1000  # 让权重先稳定
   ```

3. **Group size 过大**
   ```yaml
   # ❌ 错误
   group_size: 128  # INT4 + 大 group size → 精度差

   # ✅ 正确
   group_size: 32   # 或更小
   ```

4. **训练步数不足**
   ```yaml
   # ❌ 错误
   max_steps: 5000  # 不够让模型适应量化约束

   # ✅ 正确
   max_steps: 10000  # 增加 50-100%
   ```

**调试步骤**：

```bash
# 1. 检查训练日志中的 loss 曲线
grep "train/loss" outputs/llama3-qat-int4/runs/*/events.out.tfevents.*

# 2. 对比 Step 1000 前后的 loss
# 如果启用 QAT 后 loss 暴涨（>50%），说明学习率太高或 group_size 太大

# 3. 评估中间 checkpoint
axolotl eval examples/qat/llama3-qat-int4.yml \
    --checkpoint_dir outputs/llama3-qat-int4/checkpoint-2000
```

#### 问题 2：训练过程中 OOM（显存溢出）

**症状**：

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**原因**：QAT 需要额外显存

```
FP32 训练:
├─ 模型参数: 16 GB
├─ 激活值: 8 GB
├─ 优化器状态: 32 GB
└─ 总计: 56 GB

QAT 训练:
├─ 模型参数: 16 GB
├─ Fake quantization 参数: 2 GB  ⬅️ 额外开销
├─ 激活值: 8 GB
├─ 优化器状态: 32 GB
└─ 总计: 58 GB
```

**解决方案**：

1. **减小 batch size**
   ```yaml
   micro_batch_size: 2  # 从 4 降到 2
   gradient_accumulation_steps: 8  # 相应增加
   ```

2. **启用 gradient checkpointing**
   ```yaml
   gradient_checkpointing: true
   gradient_checkpointing_kwargs:
     use_reentrant: false  # 节省更多显存
   ```

3. **使用 FSDP/DeepSpeed**
   ```yaml
   fsdp:
     - full_shard
     - auto_wrap
   fsdp_config:
     fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
   ```

4. **延迟启用 QAT**
   ```yaml
   fake_quant_after_n_steps: 2000  # 前期不启用，节省显存
   ```

#### 问题 3：转换后的模型推理结果不一致

**症状**：

```python
# QAT 训练时的输出
"Quantum computing uses qubits that can be in superposition..."

# 转换后的量化模型输出
"Quantum computing uses qubits that can be in superposition..."  # 相同

# 但 loss/perplexity 不同
QAT 训练最后一步 loss: 0.987
转换后评估 loss: 1.234  ❌ 差异太大
```

**原因**：转换时量化参数计算不一致

**解决方案**：

```python
# 确保使用相同的量化配置
from axolotl.utils.quantization import convert_qat_model, get_quantization_config

# 转换时必须指定与训练时相同的参数
convert_qat_model(
    model,
    quantize_embedding=True,  # ⬅️ 与训练时一致
)

# 检查 config
print(model.config.quantization_config)
# 确保 group_size, dtype 等与训练时一致
```

#### 问题 4：QAT 训练速度慢

**症状**：

```
FP32 训练: 1200 tokens/s
QAT 训练:   800 tokens/s  ❌ 慢 33%
```

**原因**：Fake quantization 有额外计算开销

**优化方案**：

1. **使用 torch.compile**（PyTorch 2.0+）
   ```yaml
   torch_compile: true
   torch_compile_backend: inductor
   ```

2. **禁用不必要的 activation quantization**
   ```yaml
   qat:
     activation_dtype: null  # 仅量化权重，推理时再量化激活
   ```

3. **增大 batch size**
   ```yaml
   micro_batch_size: 4  # 更大 batch 摊薄量化开销
   ```

4. **使用 Flash Attention**
   ```yaml
   flash_attention: true
   ```

#### 问题 5：LoRA + QAT 不兼容

**症状**：

```
RuntimeError: Cannot apply LoRA to FakeQuantizedLinear
```

**原因**：LoRA 尝试替换已被 QAT 替换的层

**解决方案**：确保正确的初始化顺序

```python
# Axolotl 的正确顺序（已内置）
ModelLoader:
1. _load_model()          # 加载基础模型
2. _configure_qat()       # 替换为 FakeQuantizedLinear ⬅️
3. _load_adapters()       # 在 FakeQuantizedLinear 上应用 LoRA ⬅️

# 如果手动实现，确保先 QAT 再 LoRA
prepare_model_for_qat(model, ...)  # 1. QAT
peft_model = get_peft_model(model, lora_config)  # 2. LoRA
```

### 7.2 最佳实践

#### 最佳实践 1：超参数选择指南

| 场景 | weight_dtype | activation_dtype | group_size | learning_rate | fake_quant_after_n_steps |
|------|--------------|------------------|------------|---------------|--------------------------|
| **通用推荐** | int4 | int8 | 32 | 1e-5 | 1000 |
| **极致精度** | int4 | int8 | 16 | 5e-6 | 2000 |
| **快速训练** | int4 | null | 64 | 2e-5 | 500 |
| **FP8 (H100)** | fp8 | fp8 | null | 3e-5 | 2000 |
| **小模型 (<1B)** | int8 | int8 | 16 | 1e-5 | 500 |
| **大模型 (>70B)** | int4 | int8 | 32 | 5e-6 | 3000 |

#### 最佳实践 2：分阶段训练策略

**推荐流程**：

```
Stage 1: 正常 FP32 预训练
├─ 目标: 让权重分布收敛
├─ 数据: 50-80% 训练数据
├─ 训练步数: 80% 总步数
└─ 输出: checkpoint-pretrain

Stage 2: QAT 微调
├─ 目标: 适应量化约束
├─ 数据: 10-20% 训练数据
├─ 训练步数: 20% 总步数
├─ fake_quant_after_n_steps: 0  # 立即启用
└─ 输出: checkpoint-qat

Stage 3: 转换和验证
├─ convert_qat_model()
├─ 评估量化模型精度
└─ 导出部署
```

**示例时间线**（Llama-3-8B）：

```
Day 1-3: Stage 1 正常训练 (8000 steps)
Day 4:   Stage 2 QAT 微调 (2000 steps)
Day 4:   Stage 3 转换和评估
```

#### 最佳实践 3：评估和监控

**训练时监控指标**：

```yaml
# 配置
use_tensorboard: true
logging_steps: 10
eval_steps: 500
```

**关键指标**：

1. **Loss 曲线**
   ```
   正常模式:
   Step 0-1000:     Loss 2.5 → 1.2  (平滑下降)

   QAT 模式:
   Step 1000:       Loss 1.2 → 1.35  (略微上升，正常)
   Step 1000-2000:  Loss 1.35 → 1.15 (适应后下降)
   ```

2. **Gradient Norm**
   ```python
   # 监控梯度范数
   from torch.nn.utils import clip_grad_norm_

   grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)

   # 正常范围: 0.1 - 1.0
   # 如果 > 5.0，说明梯度爆炸，降低学习率
   ```

3. **Weight 量化误差**
   ```python
   # 训练中检查量化误差
   for name, module in model.named_modules():
       if isinstance(module, FakeQuantizedLinear):
           weight_fp32 = module.weight.data
           weight_quant = module.weight_fake_quantizer(weight_fp32)
           error = (weight_fp32 - weight_quant).abs().mean()
           print(f"{name}: quantization error = {error:.6f}")

   # 正常范围: 0.001 - 0.01
   # 如果 > 0.1，说明 group_size 太大或权重分布异常
   ```

#### 最佳实践 4：数据集选择

**QAT 对数据集的要求**：

1. **数据多样性 > 数据量**
   ```
   ❌ 不推荐: 100 万条相似数据
   ✅ 推荐: 10 万条多样化数据
   ```

2. **高质量数据**
   ```yaml
   # 数据清洗很重要
   datasets:
     - path: mlabonne/FineTome-100k  # 高质量过滤后数据集
       split: train[:10%]
   ```

3. **Task-specific 数据优先**
   ```
   如果目标是代码生成，用代码数据集进行 QAT
   如果目标是对话，用对话数据集
   ```

#### 最佳实践 5：硬件选择

| GPU | 适用模型 | QAT 性能 | 推荐配置 |
|-----|---------|---------|---------|
| **A100 (40GB)** | ≤13B | 良好 | `bf16: true, micro_batch_size: 2` |
| **A100 (80GB)** | ≤30B | 优秀 | `bf16: true, micro_batch_size: 4` |
| **H100 (80GB)** | ≤70B | 优秀 | `fp8 QAT, micro_batch_size: 8` |
| **4090 (24GB)** | ≤7B | 一般 | `gradient_checkpointing: true` |
| **A6000 (48GB)** | ≤13B | 良好 | `bf16: true, micro_batch_size: 2` |

**FP8 QAT 硬件要求**：
- NVIDIA H100/A100 (Ampere+ 架构)
- CUDA 12.0+
- PyTorch 2.1+

#### 最佳实践 6：调试和可视化

**检查 FakeQuantizedLinear 状态**：

```python
# 打印模型结构
print(model)

# 应该看到 FakeQuantizedLinear
# LlamaForCausalLM(
#   ...
#   (layers): ModuleList(
#     (0): LlamaDecoderLayer(
#       (self_attn): LlamaAttention(
#         (q_proj): FakeQuantizedLinear(in_features=4096, out_features=4096)
#         (k_proj): FakeQuantizedLinear(in_features=4096, out_features=1024)
#         ...

# 检查 fake quantizer 状态
for name, module in model.named_modules():
    if isinstance(module, FakeQuantizedLinear):
        print(f"{name}:")
        print(f"  weight_fake_quantizer.enabled: {module.weight_fake_quantizer.enabled}")
        if module.activation_fake_quantizer:
            print(f"  activation_fake_quantizer.enabled: {module.activation_fake_quantizer.enabled}")
```

**可视化权重分布**：

```python
import matplotlib.pyplot as plt
import torch

# 对比量化前后的权重分布
module = model.model.layers[0].self_attn.q_proj

weight_fp32 = module.weight.data.cpu().flatten()
weight_quant = module.weight_fake_quantizer(module.weight.data).cpu().flatten()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(weight_fp32.numpy(), bins=100, alpha=0.7, label='FP32')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('FP32 Weight Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(weight_quant.numpy(), bins=100, alpha=0.7, label='Quantized', color='orange')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Quantized Weight Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('weight_distribution.png')
```

### 7.3 故障排除流程图

```
QAT 训练出现问题
    ↓
Loss 是否正常下降？
    ├─ 否 → 检查学习率（降低 2-5x）
    │       检查 fake_quant_after_n_steps（增加）
    │       检查数据质量
    └─ 是 ↓
        ↓
转换后精度是否保持？
    ├─ 否 → 检查 group_size（减小）
    │       检查训练步数（增加）
    │       检查转换参数是否一致
    └─ 是 ↓
        ↓
推理速度是否满足？
    ├─ 否 → 检查硬件是否支持 INT4
    │       检查是否使用了 fused kernel
    │       考虑仅量化权重（activation_dtype: null）
    └─ 是 ↓
        ↓
部署成功！ ✅
```

### 7.4 性能优化 Checklist

**训练阶段**：

- [ ] 使用 `bf16: true`（Ampere+ GPU）
- [ ] 启用 `flash_attention: true`
- [ ] 启用 `gradient_checkpointing: true`（显存受限时）
- [ ] 启用 `torch_compile: true`（PyTorch 2.0+）
- [ ] 使用合适的 `micro_batch_size`（尽量大，但不 OOM）
- [ ] 多 GPU 时使用 FSDP 或 DeepSpeed
- [ ] 延迟启用 fake quantization（`fake_quant_after_n_steps > 0`）

**推理阶段**：

- [ ] 使用 `torch.compile()` 编译模型
- [ ] 使用 `torch.inference_mode()`
- [ ] 启用 KV cache
- [ ] 使用 `device_map="auto"` 自动分配
- [ ] 考虑使用 vLLM/TGI 等推理框架
- [ ] 对于 FP8，确保硬件支持 Tensor Core

**部署阶段**：

- [ ] 验证量化模型精度（与 FP32 baseline 对比）
- [ ] 测试推理速度和延迟
- [ ] 测试不同 batch size 的吞吐量
- [ ] 监控显存占用
- [ ] 准备 fallback 方案（如精度不足，回退到 FP16）

### 7.5 进阶话题

#### 7.5.1 Mixed Precision QAT

**场景**：关键层用高精度，其他层用低精度

```python
# 自定义量化策略
def custom_quantize_fn(module, name):
    # Attention 层用 INT8
    if 'attn' in name:
        return Int8DynamicActivationInt4WeightConfig(group_size=32)
    # MLP 层用 INT4
    elif 'mlp' in name:
        return Int8DynamicActivationInt4WeightConfig(group_size=64)
    # Embedding 保持 FP16
    else:
        return None

# 应用
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        config = custom_quantize_fn(module, name)
        if config:
            qat_config = QATConfig(config)
            # 替换当前层
            ...
```

#### 7.5.2 Knowledge Distillation + QAT

**思路**：用 FP32 teacher 指导 QAT student

```python
# 训练循环
teacher_model = ...  # FP32 模型
student_model = ...  # QAT 模型

for batch in dataloader:
    # Teacher 前向（无梯度）
    with torch.no_grad():
        teacher_logits = teacher_model(batch)

    # Student 前向
    student_logits = student_model(batch)

    # KD Loss
    loss_ce = cross_entropy(student_logits, labels)
    loss_kd = kl_divergence(student_logits, teacher_logits)
    loss = 0.5 * loss_ce + 0.5 * loss_kd

    # 反向传播
    loss.backward()
    optimizer.step()
```

**优势**：进一步减少量化精度损失（额外 2-5% 精度提升）

#### 7.5.3 Quantization-Aware Pruning

**结合 QAT 和剪枝**：

```python
# 1. 先剪枝
from torch.nn.utils import prune

for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# 2. 再 QAT
prepare_model_for_qat(model, ...)

# 3. 训练
# 模型同时学习剪枝掩码和量化约束
```

**优势**：进一步压缩模型（剪枝 30% + 量化 4x = 11x 压缩）

---

## 总结

### QAT 核心要点回顾

**比喻总结**：
- **PTQ**：搬完桌子再压缩 → 可能变形严重
- **QAT**：边搬边适应窄门 → 形状保持更好

**技术总结**：

1. **什么是 QAT**：训练时模拟量化，让模型适应低精度表示
2. **为什么需要**：低精度量化（INT4）时，PTQ 精度损失大，QAT 可恢复 68-96% 精度
3. **如何工作**：Fake Quantization + Straight-Through Estimator
4. **Axolotl 实现**：基于 torchao，通过配置即可启用，自动处理模型替换和训练流程
5. **最佳实践**：分阶段训练、调整超参数、监控指标、选择合适硬件

**适用场景**：
- ✅ INT4/FP8 低精度量化
- ✅ 精度要求高的生产部署
- ✅ 有充足训练资源
- ❌ 快速原型验证（用 PTQ）
- ❌ INT8 量化（PTQ 足够）

**配置模板**：

```yaml
# 通用 QAT 配置
qat:
  weight_dtype: int4
  activation_dtype: int8
  group_size: 32
  quantize_embedding: true
  fake_quant_after_n_steps: 1000

# 训练超参数调整
learning_rate: 0.00001  # 降低 5-10x
warmup_steps: 1000      # 增加 2x
max_steps: 12000        # 增加 20-50%
```

通过 QAT，我们可以在保持高精度的同时，实现 4-8 倍的模型压缩和 2-4 倍的推理加速，这对于在资源受限环境（如边缘设备、移动端）部署大语言模型至关重要。

**下一步**：
- 尝试在自己的模型上应用 QAT
- 对比 PTQ 和 QAT 的精度差异
- 探索 Mixed Precision 和 Knowledge Distillation
- 在生产环境中部署量化模型
