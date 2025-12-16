# Sample Packing 源码解析 🔍

> **深入代码层面**：逐行解读 Axolotl 的 Sample Packing 实现

---

## 目录

- [1. 源码架构总览](#1-源码架构总览)
- [2. MultipackBatchSampler 详解](#2-multipackbatchsampler-详解)
- [3. FFD 打包算法实现](#3-ffd-打包算法实现)
- [4. Data Collator 实现](#4-data-collator-实现)
- [5. Attention Mask 处理](#5-attention-mask-处理)
- [6. Monkeypatch 机制](#6-monkeypatch-机制)
- [7. 与训练流程的集成](#7-与训练流程的集成)
- [8. 分布式训练支持](#8-分布式训练支持)

---

## 1. 源码架构总览

### 1.1 核心文件结构

```
axolotl/
├── src/axolotl/
│   ├── utils/samplers/
│   │   └── multipack.py                    # ⭐ 核心: Batch Sampler
│   │
│   ├── utils/collators/
│   │   └── batching.py                     # ⭐ 核心: Data Collator
│   │
│   ├── monkeypatch/
│   │   ├── multipack.py                    # ⭐ 模型patch
│   │   └── utils.py                        # ⭐ Attention处理
│   │
│   ├── core/builders/
│   │   └── causal.py                       # 集成到trainer
│   │
│   └── core/trainers/mixins/
│       └── packing.py                      # Trainer mixin
│
└── tests/
    └── test_multipack.py                   # 测试
```

### 1.2 数据流图

```
┌────────────────────────────────────────────────────────────────┐
│                         训练数据流                               │
└────────────────────────────────────────────────────────────────┘

1. 数据集加载
   ↓
   Dataset.__getitem__() → {input_ids, attention_mask, labels}
   ↓

2. Batch Sampling  ← multipack.py: MultipackBatchSampler
   ↓
   返回: [[idx1, idx2, idx3], [idx4, idx5], ...]  # bins of indices
   ↓

3. Data Collation  ← batching.py: V2BatchSamplerDataCollatorForSeq2Seq
   ↓
   {
     input_ids: [packed_seq],
     attention_mask: [seq_ids],  ← 关键: 序列ID标记
     position_ids: [0,1,2, 0,1,2,3, ...],
     labels: [packed_seq]
   }
   ↓

4. Model Forward  ← monkeypatch/utils.py: get_unpad_data()
   ↓
   提取: indices, cu_seqlens, max_seqlen
   ↓

5. Attention Computation  ← Flash Attention / Xformers
   ↓
   使用cu_seqlens确保序列隔离
   ↓

6. Loss Calculation
   ↓
   标准交叉熵，packed序列不影响loss计算
```

### 1.3 关键类关系

```
DataLoader
    │
    ├─ sampler: MultipackBatchSampler
    │   ├─ _lengths: List[int]          # 每个样本的长度
    │   ├─ _batches: List[List[List[int]]]  # 缓存的batches
    │   └─ generate_batches() → List[List[List[int]]]
    │
    └─ collate_fn: V2BatchSamplerDataCollatorForSeq2Seq
        └─ __call__(features) → Dict[str, Tensor]

Trainer
    ├─ train_dataset
    ├─ data_collator: V2BatchSamplerDataCollatorForSeq2Seq
    └─ args.sample_packing = True

Model (patched)
    └─ attention.forward()
        └─ get_unpad_data(attention_mask)
            └─ flash_attn_varlen_func(..., cu_seqlens=...)
```

---

## 2. MultipackBatchSampler 详解

### 2.1 类定义

**文件**: `src/axolotl/utils/samplers/multipack.py`

```python
class MultipackBatchSampler(BatchSampler):
    """
    Batch sampler for efficient packing of variable-length sequences.

    核心职责:
    1. 接收数据集和序列长度
    2. 使用FFD算法将序列打包成bins
    3. 返回打包后的batch索引
    """

    _batches: list[list[list[int]]] | None = None  # 缓存batches
    _epoch: int = 0
    _efficiency: float = 0.0
    _len_packed_dataset: int = 0
```

### 2.2 初始化

```python
def __init__(
    self,
    sampler: Sampler[int],              # 底层sampler (RandomSampler等)
    batch_size: int,                    # 每个batch包含多少bins
    drop_last: bool,                    # 是否丢弃最后不完整的batch
    batch_max_len: int,                 # 每个bin的最大token容量
    lengths: list[int],                 # 预先计算的序列长度
    packing_efficiency_estimate: float = 1.0,  # 预估打包效率
    group_size: int = 100000,           # FFD分组大小
    bin_size: int = 200,                # 每个bin最多容纳序列数
    packing_sequentially: bool = False, # 是否顺序打包
):
    super().__init__(sampler, batch_size, drop_last)

    # 保存参数
    self.batch_max_len = batch_max_len
    self.lengths = lengths
    self.packing_efficiency_estimate = packing_efficiency_estimate
    self.group_size = group_size
    self.bin_size = bin_size
    self.packing_sequentially = packing_sequentially

    # 分布式设置
    if dist.is_available() and dist.is_initialized():
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()
    else:
        self.rank = 0
        self.num_replicas = 1

    # 计算打包后的数据集大小
    self._estimate_packed_length()
```

**关键点解析**:

```python
# batch_size vs batch_max_len 的区别:

# batch_size: 每个training step处理多少个bins
# 例: batch_size=4 → 每次forward 4个bins

# batch_max_len: 每个bin的token容量
# 例: batch_max_len=2048 → 每个bin最多2048 tokens

# 实际batch的token数:
total_tokens_per_batch = batch_size × batch_max_len × efficiency
# 例: 4 × 2048 × 0.95 = ~7782 tokens/batch
```

### 2.3 核心方法: generate_batches()

```python
def generate_batches(self, set_stats: bool = False) -> list[list[list[int]]]:
    """
    生成打包后的batches

    返回格式:
    [
        [[idx1, idx2], [idx3, idx4, idx5]],  # Batch 1: 2 bins
        [[idx6], [idx7, idx8, idx9, idx10]], # Batch 2: 2 bins
        ...
    ]
    """
    # 1. 获取当前epoch的样本索引
    sampler_indices = list(self.sampler)
    if hasattr(self.sampler, "generator"):
        # 设置随机种子以保证可复现
        self.sampler.generator.manual_seed(self.epoch)

    # 2. 提取对应的序列长度
    sequence_lengths = np.array(
        [self.lengths[i] for i in sampler_indices],
        dtype=np.int32
    )

    # 3. 根据模式选择打包算法
    if self.packing_sequentially:
        # 顺序打包 (保持原始顺序)
        batches_indices = allocate_sequentially(
            sequence_lengths=sequence_lengths,
            rank=self.rank,
            bin_capacity=self.batch_max_len,
            num_ranks=self.num_replicas,
        )
    else:
        # 并行打包 (FFD算法，更高效)
        batches_indices = pack_parallel(
            sequence_lengths=sequence_lengths,
            bin_capacity=self.batch_max_len,
            group_size=self.group_size,
            bin_size=self.bin_size,
            num_processes=None,  # 自动检测CPU核心数
        )

    # 4. 将bin内的索引映射回原始数据集索引
    batches = [
        [sampler_indices[i] for i in batch_bin]
        for batch_bin in batches_indices
    ]

    # 5. 分组成batches (每batch_size个bins为一组)
    batches = [
        batches[i : i + self.batch_size]
        for i in range(0, len(batches), self.batch_size)
    ]

    # 6. 处理最后一个不完整的batch
    if self.drop_last and len(batches[-1]) != self.batch_size:
        batches = batches[:-1]

    # 7. 统计效率 (可选)
    if set_stats:
        self._compute_efficiency(batches, sequence_lengths)

    return batches
```

**逐步解析**:

#### 步骤1: 获取样本索引

```python
sampler_indices = list(self.sampler)
# 例如 RandomSampler 会返回打乱的索引:
# [2045, 67, 1234, 89, 3456, ...]

# 这些是数据集中的原始索引
```

#### 步骤2: 提取序列长度

```python
sequence_lengths = np.array([self.lengths[i] for i in sampler_indices])
# self.lengths是预先计算好的所有序列长度
# 例如: lengths = [512, 1024, 256, 2048, 800, ...]

# 提取后:
# sequence_lengths = [800, 256, 512, ...]  (按sampler_indices顺序)
```

#### 步骤3-4: 打包并映射

```python
# pack_parallel返回的是相对索引 (相对于sequence_lengths数组)
batches_indices = [[0, 5, 8], [1, 2], [3, 4, 6, 7], ...]
#                   ↑ 这些是sequence_lengths中的位置

# 映射回原始数据集索引:
batches = [[sampler_indices[0], sampler_indices[5], sampler_indices[8]], ...]
#        = [[2045, 3456, ...], ...]  # 原始数据集索引
```

#### 步骤5: 分组成batches

```python
# 假设 batch_size=2
# batches_indices (bins): [[0,5,8], [1,2], [3,4,6,7], [9,10], [11], ...]

# 分组:
batches = [
    [[0,5,8], [1,2]],        # Batch 1: 2 bins
    [[3,4,6,7], [9,10]],     # Batch 2: 2 bins
    [[11]],                  # Batch 3: 1 bin (不完整)
]

# 如果 drop_last=True:
batches = [
    [[0,5,8], [1,2]],
    [[3,4,6,7], [9,10]],
]  # 丢弃最后不完整的batch
```

### 2.4 效率统计

```python
def _compute_efficiency(self, batches, sequence_lengths):
    """计算实际打包效率"""
    total_tokens = 0
    total_capacity = 0

    for batch in batches:
        for bin_indices in batch:
            # 统计bin内的实际tokens
            bin_tokens = sum(sequence_lengths[i] for i in bin_indices)
            total_tokens += bin_tokens
            total_capacity += self.batch_max_len

    self._efficiency = total_tokens / total_capacity if total_capacity > 0 else 0.0

    LOG.info(f"Sample packing efficiency: {self._efficiency:.3f}")
```

**输出示例**:

```
[INFO] Sample packing efficiency: 0.923
# 意味着: 92.3%的token slots被有效利用，只有7.7%是padding
```

### 2.5 Epoch管理

```python
def set_epoch(self, epoch: int):
    """
    设置epoch，触发重新生成batches

    为什么需要这个方法?
    - 每个epoch需要重新shuffle数据
    - 重新shuffle后需要重新打包
    - 清空_batches缓存
    """
    self.epoch = epoch
    self._batches = None  # 清空缓存，强制下次重新生成

def __iter__(self):
    """迭代器接口"""
    if self._batches is None:
        # 首次调用或缓存被清空，生成新batches
        self._batches = self.generate_batches(set_stats=True)

    for batch in self._batches:
        yield batch
```

**使用示例**:

```python
# 在训练循环中
for epoch in range(num_epochs):
    # 设置新epoch
    train_loader.batch_sampler.set_epoch(epoch)
    # ↑ 这会清空缓存，触发重新shuffle和打包

    for batch in train_loader:
        # 训练...
        pass
```

---

## 3. FFD 打包算法实现

### 3.1 并行打包入口

**文件**: `src/axolotl/utils/samplers/multipack.py:125-190`

```python
def pack_parallel(
    sequence_lengths: np.ndarray,  # [N,] 序列长度数组
    bin_capacity: int,             # bin容量 (如2048)
    group_size: int,               # 分组大小 (如100000)
    bin_size: int,                 # 每bin最多序列数 (如200)
    num_processes: int = None,     # 并行进程数
) -> list[list[int]]:
    """
    并行FFD打包算法

    核心思想:
    1. 按长度降序排序 (D in FFD)
    2. 分成多组并行处理
    3. 每组内使用FFD算法
    4. 合并结果
    """

    # 步骤1: 按长度降序排序
    sorted_indices = np.argsort(-sequence_lengths)  # 负号实现降序
    sorted_lengths = sequence_lengths[sorted_indices]

    # 示例:
    # 原始: lengths = [512, 2048, 256, 1024]
    # 排序后: sorted_lengths = [2048, 1024, 512, 256]
    #        sorted_indices = [1, 3, 0, 2]

    # 步骤2: 分组
    num_groups = (len(sorted_lengths) + group_size - 1) // group_size
    groups = []

    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, len(sorted_lengths))
        groups.append((
            sorted_lengths[start:end],  # 该组的序列长度
            start,                      # 该组的起始offset
        ))

    # 示例: 如果有250K序列，group_size=100K
    # groups = [
    #     (sorted_lengths[0:100K], 0),
    #     (sorted_lengths[100K:200K], 100K),
    #     (sorted_lengths[200K:250K], 200K),
    # ]

    # 步骤3: 估算需要的bins数量
    total_length = np.sum(sorted_lengths)
    num_bins = int(np.ceil(total_length / bin_capacity))
    # 理论最少bins: 总长度 / bin容量
    # 实际会稍多 (因为打包不完美)

    # 步骤4: 多进程并行打包
    if num_processes is None:
        import multiprocessing as mp
        num_processes = mp.cpu_count()

    if num_processes > 1 and num_groups > 1:
        # 使用进程池
        import multiprocessing as mp
        with mp.Pool(num_processes) as pool:
            results = pool.starmap(
                pack_group,
                [
                    (
                        group_lengths,
                        group_offset,
                        bin_capacity,
                        num_bins,  # 每组共享bin池
                        bin_size,
                        True,  # safe_mode
                    )
                    for group_lengths, group_offset in groups
                ],
            )
    else:
        # 单进程处理 (数据量小或num_processes=1)
        results = [
            pack_group(
                group_lengths,
                group_offset,
                bin_capacity,
                num_bins,
                bin_size,
                True,
            )
            for group_lengths, group_offset in groups
        ]

    # 步骤5: 合并所有组的结果
    all_bins = []
    for group_bins in results:
        for bin_content in group_bins:
            if len(bin_content) > 0:
                all_bins.append(bin_content)

    # 步骤6: 映射回原始索引
    final_bins = []
    for bin_content in all_bins:
        # bin_content包含的是sorted_indices中的位置
        # 需要映射回原始数据集索引
        original_indices = [sorted_indices[i] for i in bin_content]
        final_bins.append(original_indices)

    return final_bins
```

**为什么要分组?**

```python
# 问题: 如果有1M个序列，直接FFD会很慢
# O(N × M) 其中N=序列数, M=bins数
# 1M × 10K = 10B 次操作!

# 解决: 分组处理
# - 每组100K序列
# - 10组并行处理
# - 每组复杂度: O(100K × 1K) = 100M
# - 总时间: 100M / 10 (并行) = 10M 操作时间
# 加速比: 100倍!
```

### 3.2 核心FFD实现

```python
@numba.njit  # ← Numba JIT编译，加速30-50倍!
def pack_group(
    sequence_lengths: np.ndarray,  # 该组的序列长度
    group_offset: int,             # 该组的起始offset
    bin_capacity: int,             # bin容量
    max_bins: int,                 # 最多bins数
    bin_size: int,                 # 每bin最多序列数
    safe_mode: bool = True,
) -> list[list[int]]:
    """
    First-Fit Decreasing bin packing算法

    算法流程:
    1. 遍历每个序列 (已按长度降序排列)
    2. 尝试放入第一个能容纳它的bin (First-Fit)
    3. 如果所有bin都放不下，创建新bin
    """

    # 初始化数据结构
    bins_remaining = np.full(max_bins, bin_capacity, dtype=np.int32)
    # bins_remaining[i]: bin i 的剩余容量
    # 初始: [2048, 2048, 2048, ...]

    bin_contents = [[] for _ in range(max_bins)]
    # bin_contents[i]: bin i 包含的序列索引列表

    # 遍历每个序列
    for i, length in enumerate(sequence_lengths):
        global_index = group_offset + i  # 全局索引

        # 安全检查: 跳过超长序列
        if safe_mode and length > bin_capacity:
            continue

        # First-Fit: 找第一个能放下的bin
        placed = False
        for b in range(max_bins):
            # 检查两个条件:
            # 1. 容量足够
            # 2. 序列数量未达上限
            if (bins_remaining[b] >= length and
                len(bin_contents[b]) < bin_size):

                # 放入该bin
                bins_remaining[b] -= length
                bin_contents[b].append(global_index)
                placed = True
                break  # First-Fit: 找到第一个就停止

        # 如果所有现有bin都放不下，创建新bin
        if not placed:
            # 找第一个空bin
            for b in range(max_bins):
                if len(bin_contents[b]) == 0:
                    bins_remaining[b] = bin_capacity - length
                    bin_contents[b].append(global_index)
                    break

    # 返回非空bins
    result = []
    for b in range(max_bins):
        if len(bin_contents[b]) > 0:
            result.append(bin_contents[b])

    return result
```

**逐步示例**:

```python
# 输入:
sequence_lengths = [2048, 1024, 1024, 800, 512, 256]
bin_capacity = 2048
bin_size = 10

# 初始状态:
bins_remaining = [2048, 2048, 2048, ...]
bin_contents = [[], [], [], ...]

# 处理 seq[0]=2048:
# - 检查 bin[0]: 2048 >= 2048 ✅ → 放入bin[0]
bins_remaining = [0, 2048, 2048, ...]
bin_contents = [[0], [], [], ...]

# 处理 seq[1]=1024:
# - 检查 bin[0]: 0 >= 1024 ❌
# - 检查 bin[1]: 2048 >= 1024 ✅ → 放入bin[1]
bins_remaining = [0, 1024, 2048, ...]
bin_contents = [[0], [1], [], ...]

# 处理 seq[2]=1024:
# - 检查 bin[0]: 0 >= 1024 ❌
# - 检查 bin[1]: 1024 >= 1024 ✅ → 放入bin[1]
bins_remaining = [0, 0, 2048, ...]
bin_contents = [[0], [1,2], [], ...]

# 处理 seq[3]=800:
# - 检查 bin[0]: 0 >= 800 ❌
# - 检查 bin[1]: 0 >= 800 ❌
# - 检查 bin[2]: 2048 >= 800 ✅ → 放入bin[2]
bins_remaining = [0, 0, 1248, ...]
bin_contents = [[0], [1,2], [3], ...]

# 处理 seq[4]=512:
# - 检查 bin[0]: 0 >= 512 ❌
# - 检查 bin[1]: 0 >= 512 ❌
# - 检查 bin[2]: 1248 >= 512 ✅ → 放入bin[2]
bins_remaining = [0, 0, 736, ...]
bin_contents = [[0], [1,2], [3,4], ...]

# 处理 seq[5]=256:
# - 检查 bin[0]: 0 >= 256 ❌
# - 检查 bin[1]: 0 >= 256 ❌
# - 检查 bin[2]: 736 >= 256 ✅ → 放入bin[2]
bins_remaining = [0, 0, 480, ...]
bin_contents = [[0], [1,2], [3,4,5], ...]

# 最终结果:
# Bin 0: [seq0] = 2048 tokens (100%利用率)
# Bin 1: [seq1, seq2] = 2048 tokens (100%利用率)
# Bin 2: [seq3, seq4, seq5] = 1568 tokens (76.6%利用率)
# 平均利用率: (2048+2048+1568)/(3×2048) = 91.4%
```

### 3.3 顺序打包 (Sequential)

```python
@numba.njit
def allocate_sequentially(
    sequence_lengths: np.ndarray,
    rank: int,           # 当前rank
    bin_capacity: int,
    num_ranks: int,      # 总rank数
) -> list[list[int]]:
    """
    顺序打包: 不排序，按原始顺序处理

    用途:
    - Curriculum learning (数据顺序很重要)
    - 需要保持数据顺序的场景

    区别于并行FFD:
    - 不排序 (效率稍低)
    - 单进程 (不并行)
    - 保持原始顺序
    """

    bins_remaining = []
    bin_contents = []

    # 遍历序列 (按原始顺序)
    for i, length in enumerate(sequence_lengths):
        # 跳过超长序列
        if length > bin_capacity:
            continue

        # First-Fit
        placed = False
        for b in range(len(bins_remaining)):
            if bins_remaining[b] >= length:
                bins_remaining[b] -= length
                bin_contents[b].append(i)
                placed = True
                break

        # 创建新bin
        if not placed:
            bins_remaining.append(bin_capacity - length)
            bin_contents.append([i])

    # 分布式训练: 只返回属于当前rank的bins
    if num_ranks > 1:
        # 轮询分配: rank 0 获取bin 0,3,6,...
        #          rank 1 获取bin 1,4,7,...
        result = []
        for b in range(rank, len(bin_contents), num_ranks):
            result.append(bin_contents[b])
        return result
    else:
        return bin_contents
```

**顺序 vs 并行 FFD 对比**:

```python
# 数据: [100, 2000, 200, 1800, 300, 1700]
# bin_capacity = 2048

# 并行FFD (排序):
# 排序后: [2000, 1800, 1700, 300, 200, 100]
# Bin 1: [2000] → 2000/2048 = 97.7%
# Bin 2: [1800, 200] → 2000/2048 = 97.7%
# Bin 3: [1700, 300] → 2000/2048 = 97.7%
# Bin 4: [100] → 100/2048 = 4.9%
# 平均: 74.5%

# 顺序FFD (不排序):
# 按原始: [100, 2000, 200, 1800, 300, 1700]
# Bin 1: [100, 200, 300, ...] → 尝试填满
# Bin 2: [2000] → 2000/2048 = 97.7%
# Bin 3: [1800] → 1800/2048 = 87.9%
# Bin 4: [1700] → 1700/2048 = 83.0%
# 效率通常较低

# 但顺序很重要时，必须使用Sequential!
```

---

## 4. Data Collator 实现

### 4.1 V2BatchSamplerDataCollatorForSeq2Seq

**文件**: `src/axolotl/utils/collators/batching.py:159-196`

```python
@dataclass
class V2BatchSamplerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    V2 Collator: 支持更广泛的模型

    关键改进:
    - Attention mask使用序列ID (1,2,3,...)
    - 更好的position_ids处理
    - 支持非Flash Attention模型
    """

    squash_position_ids: bool = False  # 是否压平position_ids

    def __call__(self, features, return_tensors=None):
        """
        输入: List[List[dict]] 或 List[dict]
              外层List: batch中的bins
              内层List: bin中的sequences

        输出: Dict[str, Tensor]
              拼接并pad后的batch
        """

        # 规范化输入格式
        if not isinstance(features[0], list):
            features: List[List[dict]] = [features]
        # 现在 features = [[seq1, seq2], [seq3, seq4, seq5], ...]
        #                  \_bin1_/      \_____bin2______/

        # 为每个bin创建输出dict
        out_features = [{} for _ in features]

        # 处理每个bin
        for i, bin_sequences in enumerate(features):
            # bin_sequences = [seq1_dict, seq2_dict, seq3_dict]

            # 遍历所有feature keys
            for feature_name in bin_sequences[0].keys():
                if feature_name == "length":
                    continue  # 跳过辅助字段

                if feature_name == "attention_mask":
                    # ⭐ 关键处理: attention_mask
                    arrays = [
                        (seq_idx + 1) * np.array(seq[feature_name])
                        for seq_idx, seq in enumerate(bin_sequences)
                    ]
                    # 示例:
                    # seq 0: [1,1,1] → (0+1)*[1,1,1] = [1,1,1]
                    # seq 1: [1,1,1,1] → (1+1)*[1,1,1,1] = [2,2,2,2]
                    # seq 2: [1,1] → (2+1)*[1,1] = [3,3]

                    out_features[i][feature_name] = np.concatenate(arrays)
                    # → [1,1,1, 2,2,2,2, 3,3]

                elif feature_name == "position_ids" and self.squash_position_ids:
                    # 可选: 压平position_ids
                    # (某些模型需要连续的position_ids)
                    arrays = [
                        np.array(seq[feature_name])
                        for seq in bin_sequences
                    ]
                    position_ids = np.concatenate(arrays)
                    total_length = position_ids.shape[0]
                    # 重新生成连续的position_ids
                    position_ids = np.arange(total_length)
                    out_features[i][feature_name] = position_ids

                else:
                    # 其他字段: 直接拼接
                    # input_ids, labels, position_ids (默认)
                    arrays = [
                        np.array(seq[feature_name])
                        for seq in bin_sequences
                    ]
                    out_features[i][feature_name] = np.concatenate(arrays)

        # 调用父类的__call__进行padding
        # 将所有bins pad到相同长度
        return super().__call__(out_features, return_tensors=return_tensors)
```

**完整示例**:

```python
# 输入: batch包含2个bins
features = [
    # Bin 1: 2个序列
    [
        {
            'input_ids': [101, 102, 103],
            'attention_mask': [1, 1, 1],
            'position_ids': [0, 1, 2],
            'labels': [101, 102, 103],
        },
        {
            'input_ids': [201, 202, 203, 204],
            'attention_mask': [1, 1, 1, 1],
            'position_ids': [0, 1, 2, 3],
            'labels': [201, 202, 203, 204],
        },
    ],
    # Bin 2: 1个序列
    [
        {
            'input_ids': [301, 302],
            'attention_mask': [1, 1],
            'position_ids': [0, 1],
            'labels': [301, 302],
        },
    ],
]

# 处理后 (拼接但未padding):
out_features = [
    # Bin 1 (拼接后)
    {
        'input_ids': np.array([101,102,103, 201,202,203,204]),
        'attention_mask': np.array([1,1,1, 2,2,2,2]),  # ← 序列ID!
        'position_ids': np.array([0,1,2, 0,1,2,3]),    # ← 独立计数
        'labels': np.array([101,102,103, 201,202,203,204]),
    },
    # Bin 2 (拼接后)
    {
        'input_ids': np.array([301, 302]),
        'attention_mask': np.array([1, 1]),
        'position_ids': np.array([0, 1]),
        'labels': np.array([301, 302]),
    },
]

# 调用父类padding (pad到batch内最长=7):
final_output = {
    'input_ids': torch.tensor([
        [101,102,103, 201,202,203,204],  # Bin 1
        [301,302, 0,0,0,0,0],            # Bin 2 + padding
    ]),
    'attention_mask': torch.tensor([
        [1,1,1, 2,2,2,2],
        [1,1, 0,0,0,0,0],  # ← padding的mask=0
    ]),
    'position_ids': torch.tensor([
        [0,1,2, 0,1,2,3],
        [0,1, 0,0,0,0,0],  # ← padding的position_ids=0
    ]),
    'labels': torch.tensor([
        [101,102,103, 201,202,203,204],
        [301,302, -100,-100,-100,-100,-100],  # ← padding的label=-100
    ]),
}
```

### 4.2 V1 vs V2 Collator

```python
# V1: BatchSamplerDataCollatorForSeq2Seq
class V1:
    def __call__(self, features):
        # attention_mask: 所有序列都乘以1
        arrays = [
            (1) * np.array(item[feature])  # ← 注意这里
            for item in features
        ]
        # 结果: [1,1,1, 1,1,1,1, 1,1]
        #       ↑ 无法区分不同序列!

# V2: V2BatchSamplerDataCollatorForSeq2Seq
class V2:
    def __call__(self, features):
        # attention_mask: 每个序列乘以不同ID
        arrays = [
            (i + 1) * np.array(item[feature])  # ← 关键差异
            for i, item in enumerate(features)
        ]
        # 结果: [1,1,1, 2,2,2,2, 3,3]
        #       ↑ 可以区分不同序列!
```

**为什么需要V2?**

```python
# V1适用于: Flash Attention (native支持multipack)
# - Flash Attention可以直接处理packed sequences
# - 通过cu_seqlens参数知道序列边界

# V2适用于: 非Flash Attention模型 (如标准Attention, SDPA)
# - 需要通过attention_mask区分序列
# - 在mask_2d_to_4d中构建block-diagonal mask
```

---

## 5. Attention Mask 处理

### 5.1 get_unpad_data()

**文件**: `src/axolotl/monkeypatch/utils.py:31-45`

```python
@torch.jit.script  # ← JIT编译，加速推理
def get_unpad_data(attention_mask: torch.Tensor):
    """
    从packed attention_mask中提取有效token位置和序列边界

    输入: attention_mask with sequence IDs
          shape: [batch, total_tokens]
          示例: [[1,1,1, 2,2,2,2, 3,3, 0,0]]

    输出: (indices, cu_seqlens, max_seqlen)
    """
    device = attention_mask.device

    # 1. 获取每个序列的长度
    seqlens_in_batch = get_max_seqlen_in_batch(attention_mask)
    # 示例: [3, 4, 2]  (3个序列，长度分别为3,4,2)

    # 2. 获取所有非零token的位置
    indices = torch.nonzero(attention_mask.flatten()).flatten()
    # 示例: tensor([0,1,2, 3,4,5,6, 7,8])
    #              ↑ 位置0-8是有效tokens，9-10是padding

    # 3. 计算最长序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 示例: 4

    # 4. 计算累积序列长度 (cu_seqlens)
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32),
        (1, 0)  # 在前面pad一个0
    ).to(device=device).detach()
    # cumsum([3,4,2]) = [3, 7, 9]
    # pad: [0, 3, 7, 9]  ← 每个序列的起始位置

    return (
        indices,           # 有效token位置
        cu_seqlens,        # 序列边界
        max_seqlen_in_batch,  # 最长序列长度
    )
```

**逐步解析**:

```python
# 输入:
attention_mask = torch.tensor([[1,1,1, 2,2,2,2, 3,3, 0,0]])

# 步骤1: get_max_seqlen_in_batch
seqlens_in_batch = get_max_seqlen_in_batch(attention_mask)
# 内部逻辑:
max_num = 3  # 最大序列ID
counts = torch.zeros((1, 3))
# 统计每个ID的出现次数:
# ID=1: 3次 → counts[0,0] = 3
# ID=2: 4次 → counts[0,1] = 4
# ID=3: 2次 → counts[0,2] = 2
# 结果: [3, 4, 2]

# 步骤2: 非零位置
attention_mask.flatten()  # [1,1,1,2,2,2,2,3,3,0,0]
indices = torch.nonzero(...)  # [[0],[1],[2],[3],[4],[5],[6],[7],[8]]
indices = indices.flatten()   # [0,1,2,3,4,5,6,7,8]

# 步骤3: 最长序列
max_seqlen_in_batch = max([3,4,2]) = 4

# 步骤4: 累积长度
cumsum([3,4,2]) = [3, 7, 9]
pad (1, 0):     = [0, 3, 7, 9]
# 含义:
# - 序列1: tokens 0-2   (cu_seqlens[0]=0 to cu_seqlens[1]=3)
# - 序列2: tokens 3-6   (cu_seqlens[1]=3 to cu_seqlens[2]=7)
# - 序列3: tokens 7-8   (cu_seqlens[2]=7 to cu_seqlens[3]=9)
```

### 5.2 get_max_seqlen_in_batch()

```python
@torch.jit.script
def get_max_seqlen_in_batch(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    从attention_mask中提取每个序列的长度

    输入: [batch, total_tokens] 包含序列ID的mask
    输出: [num_sequences] 每个序列的长度
    """
    # 找到最大序列ID
    max_num = int(torch.max(attention_mask).item())
    batch_size, _ = attention_mask.shape

    # 为每个ID统计出现次数
    counts = torch.zeros((batch_size, max_num), dtype=torch.int32)

    for i in range(1, max_num + 1):
        # 创建mask: 当前ID的位置
        mask = (attention_mask == i)
        # 统计每行该ID出现的次数
        counts[:, i - 1] = torch.sum(mask, dim=-1).to(dtype=torch.int32)

    # 展平并去除0 (可能有空序列)
    result = counts.flatten()
    nonzero_indices = torch.nonzero(result).squeeze(-1)
    return result[nonzero_indices]
```

**示例**:

```python
# attention_mask包含2个bins:
attention_mask = torch.tensor([
    [1,1,1, 2,2,2,2, 0,0],     # Bin 1: seq1(3), seq2(4)
    [1,1, 2,2,2, 0,0,0,0],     # Bin 2: seq3(2), seq4(3)
])

max_num = 2  # 每个bin内最多2个序列

counts = torch.zeros((2, 2), dtype=torch.int32)

# 处理ID=1:
mask = (attention_mask == 1)
# [[1,1,1, 0,0,0,0, 0,0],
#  [1,1, 0,0,0, 0,0,0,0]]
counts[:, 0] = torch.sum(mask, dim=-1)  # [3, 2]

# 处理ID=2:
mask = (attention_mask == 2)
# [[0,0,0, 1,1,1,1, 0,0],
#  [0,0, 1,1,1, 0,0,0,0]]
counts[:, 1] = torch.sum(mask, dim=-1)  # [4, 3]

# counts = [[3, 4],
#           [2, 3]]

result = counts.flatten()  # [3, 4, 2, 3]
# 即: bin1的seq1=3, seq2=4, bin2的seq3=2, seq4=3
```

### 5.3 在模型forward中的使用

```python
# 典型的Attention层forward函数
def forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    ...
):
    bsz, q_len, _ = hidden_states.size()

    # 计算QKV
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape for multi-head
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

    # ⭐ Sample Packing: 提取有效tokens和序列边界
    if attention_mask is not None and torch.any(attention_mask > 1):
        # 检测到packed sequences (attention_mask包含序列ID)
        indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)

        # 去除padding
        query_states = query_states.flatten(0, 1)[indices]
        key_states = key_states.flatten(0, 1)[indices]
        value_states = value_states.flatten(0, 1)[indices]

        # Flash Attention with variable-length sequences
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=True,
        )

        # Flash Attention内部会根据cu_seqlens:
        # - 只在序列内部计算attention
        # - 自动屏蔽跨序列的attention
        # - 完全跳过padding部分

        # Reshape回原始shape (包含padding)
        attn_output_unpad = attn_output
        attn_output = torch.zeros(
            bsz * q_len, self.num_heads, self.head_dim,
            dtype=attn_output.dtype, device=attn_output.device
        )
        attn_output[indices] = attn_output_unpad

    else:
        # 标准attention (非packed)
        attn_output = self.standard_attention(
            query_states, key_states, value_states, attention_mask
        )

    return attn_output
```

---

## 6. Monkeypatch 机制

### 6.1 patch_for_multipack()

**文件**: `src/axolotl/monkeypatch/multipack.py:53-65`

```python
SUPPORTED_MULTIPACK_MODEL_TYPES = [
    "llama", "mistral", "mixtral", "qwen2", "gemma", "phi3",
    "deepseek_v2", "deepseek_v3", ...
]

def patch_for_multipack(model_type, model_name=None, has_remote_code=False):
    """
    为模型打patch以支持Sample Packing

    核心思想:
    - 替换transformers库中的_get_unpad_data函数
    - 使其能正确处理packed sequences
    """

    if has_remote_code:
        # 远程代码模型需要特殊处理
        patch_remote(model_name)

    elif hasattr(transformers, "modeling_flash_attention_utils"):
        # Transformers >= 4.36版本
        # 替换全局的_get_unpad_data函数
        assert hasattr(
            transformers.modeling_flash_attention_utils,
            "_get_unpad_data"
        ), "transformers API changed!"

        # ⭐ 核心: 替换为我们的实现
        transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data

    # Mixtral + DeepSpeed Zero3需要额外patch
    if model_type == "mixtral" and is_deepspeed_zero3_enabled():
        patch_mixtral_moe_forward_zero3()
```

**为什么需要monkeypatch?**

```python
# Transformers原生的_get_unpad_data:
def _get_unpad_data(attention_mask):
    """
    原生实现假设attention_mask是binary (0/1)
    不支持序列ID (1,2,3,...)
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 对于 [1,1,1, 2,2,2,2, 0,0]:
    # sum = 1+1+1+2+2+2+2 = 11  ← 错误! 应该是9

    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 这部分是对的

    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 错误的seqlens导致错误的max_seqlen

    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
    )
    # 错误的seqlens导致错误的cu_seqlens

    return (indices, cu_seqlens, max_seqlen_in_batch)

# Axolotl的实现:
# 正确处理序列ID，提取真实的序列长度
```

### 6.2 patch_remote()

```python
def patch_remote(model_name):
    """
    为remote code模型打patch

    挑战:
    - Remote code模型的modeling文件在运行时动态加载
    - 不在transformers库中，需要找到实际模块
    """
    # 1. 加载模型配置
    model_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 2. 加载模型 (触发remote code下载)
    with init_empty_weights():
        AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    # 现在modeling_xxx.py已经被import

    # 3. 找到modeling module
    parts = model_config.__class__.__module__.split(".")
    # 例: "transformers_modules.model_name.configuration_xxx"
    parts[-1] = parts[-1].replace("configuration_", "modeling_", 1)
    # → "transformers_modules.model_name.modeling_xxx"

    module_name = ".".join(parts)
    modeling_arch = importlib.import_module(module_name)

    # 4. 替换_get_unpad_data
    if hasattr(modeling_arch, "_get_unpad_data"):
        modeling_arch._get_unpad_data = get_unpad_data
```

---

## 7. 与训练流程的集成

### 7.1 在HFCausalTrainerBuilder中的配置

**文件**: `src/axolotl/core/builders/causal.py:250-284`

```python
class HFCausalTrainerBuilder(TrainerBuilderBase):
    def build(self, total_num_steps):
        # ...

        # ⭐ Sample Packing配置传递给TrainingArguments
        training_arguments_kwargs["sample_packing"] = bool(self.cfg.sample_packing)

        # 是否drop attention_mask (Flash Attention可以drop)
        training_arguments_kwargs["sample_packing_drop_attention_mask"] = bool(
            self.cfg.flash_attention
            or self.cfg.xformers_attention
            or self.cfg.flex_attention
        )

        # 是否使用real batches (legacy设置)
        training_arguments_kwargs["multipack_real_batches"] = (
            self.cfg.multipack_real_batches
            if self.cfg.multipack_real_batches is not None
            else not (
                self.cfg.flash_attention
                or self.cfg.flex_attention
                or self.cfg.xformers_attention
            )
        )

        # Eval也启用packing
        training_arguments_kwargs["eval_sample_packing"] = bool(
            self.cfg.eval_sample_packing
        )

        # Packing模式
        if self.cfg.sample_packing_sequentially is not None:
            training_arguments_kwargs["sample_packing_sequentially"] = (
                self.cfg.sample_packing_sequentially
            )

        # Bin配置
        if self.cfg.sample_packing_bin_size is not None:
            training_arguments_kwargs["sample_packing_bin_size"] = (
                self.cfg.sample_packing_bin_size
            )

        if self.cfg.sample_packing_group_size is not None:
            training_arguments_kwargs["sample_packing_group_size"] = (
                self.cfg.sample_packing_group_size
            )

        # 效率估计
        if self.cfg.sample_packing_eff_est:
            training_arguments_kwargs["sample_packing_efficiency"] = (
                self.cfg.sample_packing_eff_est
            )

        # ...

        training_args = AxolotlTrainingArguments(**training_arguments_kwargs)

        # ...
        return trainer
```

### 7.2 Collator选择逻辑

```python
def build_collator(self, training_args, is_eval=False, **kwargs):
    """选择合适的data collator"""

    # 检查是否需要packing collator
    use_batch_sampler_collator = False
    if is_eval is False and training_args.sample_packing:
        use_batch_sampler_collator = True
    if is_eval and training_args.eval_sample_packing:
        use_batch_sampler_collator = True

    if use_batch_sampler_collator:
        # 选择V1 vs V2
        if (
            self.cfg.flex_attention
            or self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
            or (
                self.cfg.model_config_type in ["llama"]
                and self.cfg.flash_attention is not True
            )
        ):
            # 使用V2 (更广泛兼容)
            collator = V2BatchSamplerDataCollatorForSeq2Seq
        else:
            # 使用V1 (Flash Attention专用)
            collator = BatchSamplerDataCollatorForSeq2Seq
    else:
        # 标准collator
        collator = DataCollatorForSeq2Seq

    return collator(self.tokenizer, **kwargs)
```

### 7.3 DeepSpeed特殊处理

```python
# 在trainer创建后
if self.cfg.deepspeed and self.cfg.sample_packing:
    # DeepSpeed需要知道真实的micro_batch_size
    # (因为Sample Packing改变了batch结构)
    trainer.accelerator.state.deepspeed_plugin.deepspeed_config[
        "train_micro_batch_size_per_gpu"
    ] = self.cfg.micro_batch_size
```

---

## 8. 分布式训练支持

### 8.1 DDP集成

```python
class MultipackBatchSampler:
    def __init__(self, ...):
        # 自动检测分布式环境
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()
        else:
            self.rank = 0
            self.num_replicas = 1

    def generate_batches(self):
        # 所有rank生成相同的batches (使用相同的seed)
        batches = pack_parallel(...)

        # 但每个rank只处理自己的分片
        # 通过sampler自动处理 (DistributedSampler)
```

**DistributedSampler + MultipackBatchSampler**:

```python
# 训练代码
from torch.utils.data import DistributedSampler

# 创建sampler
base_sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=42,
)

# 包装为MultipackBatchSampler
batch_sampler = MultipackBatchSampler(
    sampler=base_sampler,
    batch_size=4,
    batch_max_len=2048,
    lengths=precomputed_lengths,
    ...
)

# 创建DataLoader
train_loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,  # ← 使用batch_sampler
    collate_fn=collator,
)

# 训练循环
for epoch in range(num_epochs):
    base_sampler.set_epoch(epoch)  # 重要! 设置epoch
    batch_sampler.set_epoch(epoch)  # 也要设置

    for batch in train_loader:
        # 每个rank处理不同的bins
        ...
```

### 8.2 FSDP/TP集成

```python
# FSDP和TP通过DeviceMesh协调

# 例: 8 GPUs, TP=2, DP=4
from torch.distributed.device_mesh import init_device_mesh

device_mesh = init_device_mesh(
    "cuda",
    (4, 2),  # (DP, TP)
    mesh_dim_names=("dp", "tp"),
)

# MultipackBatchSampler自动处理:
# - 检测当前rank在DP维度的位置
# - 只返回属于该DP rank的数据

# 伪代码:
dp_rank = device_mesh.get_local_rank("dp")
dp_world_size = device_mesh.size("dp")

# 在generate_batches中:
my_bins = all_bins[dp_rank::dp_world_size]
# DP rank 0: bins [0, 4, 8, 12, ...]
# DP rank 1: bins [1, 5, 9, 13, ...]
# DP rank 2: bins [2, 6, 10, 14, ...]
# DP rank 3: bins [3, 7, 11, 15, ...]
```

### 8.3 效率同步

```python
class MultipackBatchSampler:
    def gather_efficiency(self) -> float:
        """收集所有ranks的效率统计"""
        if not dist.is_available() or not dist.is_initialized():
            return self._efficiency

        # 创建tensor
        efficiency_tensor = torch.tensor(
            [self._efficiency],
            dtype=torch.float32,
            device="cuda"
        )

        # AllGather: 收集所有ranks的效率
        gathered = [torch.zeros_like(efficiency_tensor) for _ in range(self.num_replicas)]
        dist.all_gather(gathered, efficiency_tensor)

        # 计算平均效率
        efficiencies = [t.item() for t in gathered]
        avg_efficiency = sum(efficiencies) / len(efficiencies)

        return avg_efficiency

    def gather_len_batches(self) -> int:
        """收集所有ranks的batch数量"""
        if not dist.is_available() or not dist.is_initialized():
            return len(self._batches) if self._batches else 0

        len_tensor = torch.tensor(
            [len(self._batches) if self._batches else 0],
            dtype=torch.int64,
            device="cuda"
        )

        # AllGather
        gathered = [torch.zeros_like(len_tensor) for _ in range(self.num_replicas)]
        dist.all_gather(gathered, len_tensor)

        # 返回最小值 (确保所有ranks同步)
        lengths = [t.item() for t in gathered]
        return min(lengths)
```

**为什么需要gather_len_batches?**

```python
# 问题: 不同ranks可能生成不同数量的batches
# Rank 0: 1000 batches
# Rank 1: 999 batches  ← 数据分片可能不均

# 如果不同步:
# - Rank 0会等待batch 1000
# - Rank 1已经结束
# - 训练hang!

# 解决: 取最小值，确保所有ranks在相同步数结束
min_batches = batch_sampler.gather_len_batches()
# 所有ranks都只运行999个batches
```

---

## 总结

### 核心源码组件

1. **MultipackBatchSampler** (`multipack.py:244-474`)
   - 负责生成打包后的batch索引
   - 支持FFD并行和Sequential两种模式
   - 自动处理分布式训练

2. **FFD算法** (`multipack.py:61-190`)
   - `pack_parallel()`: 并行FFD，最高效
   - `pack_group()`: 单组FFD，Numba加速
   - `allocate_sequentially()`: 顺序打包，保持顺序

3. **Data Collator** (`batching.py:159-196`)
   - `V2BatchSamplerDataCollatorForSeq2Seq`: 推荐，广泛兼容
   - 关键: attention_mask使用序列ID区分序列

4. **Attention处理** (`monkeypatch/utils.py:31-96`)
   - `get_unpad_data()`: 提取有效tokens和序列边界
   - `get_max_seqlen_in_batch()`: 从mask提取序列长度
   - `get_cu_seqlens()`: 计算累积序列长度

5. **Monkeypatch** (`monkeypatch/multipack.py`)
   - 替换transformers的`_get_unpad_data`
   - 确保模型正确处理packed sequences

### 关键技术点

- **Numba JIT**: FFD算法加速30-50倍
- **序列ID标记**: attention_mask=[1,2,3,...] 区分不同序列
- **cu_seqlens**: Flash Attention的序列边界参数
- **分布式兼容**: 自动检测DDP/FSDP/TP环境

### 数据流总结

```
Dataset
  ↓ __getitem__
{input_ids, attention_mask, labels}
  ↓ MultipackBatchSampler
[[idx1,idx2], [idx3,idx4,idx5], ...]  # bins
  ↓ V2Collator
{input_ids: [packed], attention_mask: [1,1,2,2,2,3,3], ...}
  ↓ get_unpad_data
(indices, cu_seqlens=[0,2,5,7], max_seqlen=3)
  ↓ Flash Attention
正确隔离序列，高效计算
  ↓ Loss
标准交叉熵，不受packing影响
```

---

## 相关文档

- [Sample Packing 深度解析](./sample_packing_deep_dive.md)
- [Sample Packing 快速参考](./sample_packing_quick_reference.md)
- [主索引](./README.md)

---

*文档版本: v1.0 | 最后更新: 2025-11*
