# 第10章：LLM 架构设计基础

> 🟢 入门 → 🔴 高级 | 阅读时长 25-30分钟 | 第四部分：大模型架构

---

## 目录

- [10.1 Transformer 架构回顾](#101-transformer-架构回顾)
- [10.2 大模型训练架构](#102-大模型训练架构)
- [10.3 模型并行与流水线并行](#103-模型并行与流水线并行)
- [10.4 内存优化技术](#104-内存优化技术)
- [10.5 混合精度训练](#105-混合精度训练)
- [📝 练习：搭建小规模分布式训练环境](#-练习搭建小规模分布式训练环境)
- [本章小结](#本章小结)
- [参考文献](#参考文献)

---

## 10.1 Transformer 架构回顾

### 10.1.1 自注意力机制

Transformer 架构由 Vaswani 等人于2017年在论文"Attention Is All You Need"中提出，彻底改变了自然语言处理领域，现已成为几乎所有大语言模型（LLM）的核心骨架。在深入大规模模型的架构设计之前，我们首先需要夯实基础构建模块的理解。

📌 **核心概念**：自注意力机制计算序列的加权表示，其中每个 token 的表示依赖于序列中所有其他 token。

自注意力的核心运算可以表示为：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中：
- **Q**（Query，查询）：我在寻找什么？
- **K**（Key，键）：我包含什么信息？
- **V**（Value，值）：我提供什么信息？
- **d_k**：键向量的维度（缩放因子）

多头注意力（Multi-Head Attention）通过并行运行多个注意力计算来扩展这一机制：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

标准 Transformer 块的架构图：

```
┌─────────────────────────────────────────────────────┐
│                Transformer 块                        │
│                                                      │
│  输入嵌入（Input Embedding）                          │
│       │                                              │
│       ▼                                              │
│  ┌─────────────────────────────────┐                │
│  │    多头自注意力（Multi-Head Self-Attention）      │
│  │  ┌─────┐ ┌─────┐ ┌─────┐      │                │
│  │  │  Q  │ │  K  │ │  V  │      │                │
│  │  └──┬──┘ └──┬──┘ └──┬──┘      │                │
│  │     └───────┼───────┘          │                │
│  │             ▼                   │                │
│  │     缩放点积注意力              │                │
│  │  (Scaled Dot-Product Attention)│                │
│  └─────────────┬───────────────────┘                │
│                │                                     │
│       ───────────── （残差连接）                      │
│                │                                     │
│         层归一化（Layer Normalization）               │
│                │                                     │
│                ▼                                     │
│  ┌─────────────────────────────────┐                │
│  │   前馈网络（FFN）                │                │
│  │   Linear → Activation → Linear   │                │
│  └─────────────┬───────────────────┘                │
│                │                                     │
│       ───────────── （残差连接）                      │
│                │                                     │
│         层归一化（Layer Normalization）               │
│                │                                     │
│                ▼                                     │
│  输出嵌入（Output Embedding）                        │
└─────────────────────────────────────────────────────┘
```

### 10.1.2 从 Transformer 到 LLM：缩放定律

从标准 Transformer 过渡到大语言模型，需要沿着三个维度进行系统化缩放：

| 维度 | 小型 Transformer | 大型 LLM（如 LLaMA-3 405B） |
|------|-------------------|------------------------------|
| 参数量 | 10M - 100M | 10B - 405B+ |
| 层数 | 6 - 12 | 80 - 128 |
| 隐藏维度 | 256 - 768 | 8192 - 16384 |
| 注意力头数 | 4 - 12 | 64 - 128 |
| 上下文长度 | 512 - 2048 | 8K - 128K+ |
| 训练数据量 | GB 级 | TB 级 |

📌 **核心概念**：Chinchilla 缩放定律（Hoffmann 等，2022）表明，最优性能需要平衡模型大小和数据大小。一个 70B 参数的模型应在约 1.4T tokens 上训练，以实现最优计算效率。

缩放定律遵循幂律关系：

```
L(N, D) ∝ (N_c / N)^α_N + (D_c / D)^α_D
```

其中 L 是损失，N 是参数数量，D 是数据集大小，N_c、D_c、α_N、α_D 是拟合常数。

### 10.1.3 现代架构创新（2024-2026）

现代 LLM 已经从原始 Transformer 演进了显著变化。关键创新包括：

**分组查询注意力（Grouped Query Attention, GQA）**：
与标准多头注意力（每个头有独立的 Q、K、V 投影）不同，GQA 在查询头组之间共享 K 和 V 投影。这在推理时大幅减少了 KV 缓存的内存占用。

```
标准 MHA：Q_heads=64, K_heads=64, V_heads=64
GQA：     Q_heads=64, K_heads=8,  V_heads=8   （8组，每组8个Q头）
MQA：     Q_heads=64, K_heads=1,  V_heads=1   （所有头共享同一个K,V）
```

**旋转位置编码（RoPE）**：
RoPE 不使用学习的绝对位置嵌入或相对位置偏置，而是通过旋转矩阵直接将位置信息编码到注意力计算中。这使得长度泛化能力更好。

```python
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算旋转嵌入的频率。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """将旋转嵌入应用于查询和键。"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**SwiGLU 激活函数**：
现代 LLM 在前馈网络中用 SwiGLU 替代 ReLU/GELU：

```python
class SwiGLU(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
```

**RMSNorm 替代 LayerNorm**：
RMSNorm 去除了均值中心化步骤，使计算更高效：

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
```

---

## 10.2 大模型训练架构

### 10.2.1 大规模训练流水线

训练大语言模型是现代计算中最密集的计算任务之一。像 LLaMA-3 405B 这样的前沿模型在 H100 GPU 上训练大约需要 3084万 GPU 小时。

```
┌──────────────────────────────────────────────────────────────────┐
│                   大模型训练流水线                                 │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  数据     │───▶│  分词化   │───▶│ 分片与   │───▶│  数据     │  │
│  │  采集     │    │(Tokenize)│    │  加载     │    │  加载器   │  │
│  └──────────┘    └──────────┘    └──────────┘    └─────┬────┘  │
│                                                         │        │
│                                                         ▼        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              分布式训练循环                                 │   │
│  │                                                           │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  前向传播（所有 GPU 上）                          │     │   │
│  │  │  ├── GPU 0: 层 0-19    ─────────────┐          │     │   │
│  │  │  ├── GPU 1: 层 20-39  ──────────────┤          │     │   │
│  │  │  ├── GPU 2: 层 40-59  ──────────────┤          │     │   │
│  │  │  └── GPU 3: 层 60-79  ──────────────┘          │     │   │
│  │  └──────────────────────┬──────────────────────────┘     │   │
│  │                         │                                 │   │
│  │                         ▼                                 │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  损失计算与梯度计算                               │     │   │
│  │  └──────────────────────┬──────────────────────────┘     │   │
│  │                         │                                 │   │
│  │                         ▼                                 │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  反向传播 + 梯度全局归约                          │     │   │
│  │  │  ├── ZeRO Stage 1: 优化器状态分片                │     │   │
│  │  │  ├── ZeRO Stage 2: + 梯度分片                   │     │   │
│  │  │  └── ZeRO Stage 3: + 参数分片                   │     │   │
│  │  └──────────────────────┬──────────────────────────┘     │   │
│  │                         │                                 │   │
│  │                         ▼                                 │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  优化器更新步（AdamW / Adam-mini）                │     │   │
│  │  └──────────────────────┬──────────────────────────┘     │   │
│  │                         │                                 │   │
│  │                    [重复执行]                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              检查点与监控                                   │   │
│  │  ├── 周期性检查点保存（每 N 步）                           │   │
│  │  ├── 损失/梯度范数日志                                    │   │
│  │  └── GPU 利用率与内存监控                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 10.2.2 数据并行（Data Parallelism, DP）

最简单的分布式训练形式。每个 GPU 持有完整的模型副本，处理不同的数据批次，然后同步梯度。

```
┌──────────────────────────────────────────────────┐
│              数据并行（Data Parallelism）          │
│                                                    │
│  GPU 0          GPU 1          GPU 2          GPU 3 │
│  ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐│
│  │完整 │       │完整 │       │完整 │       │完整 ││
│  │模型 │       │模型 │       │模型 │       │模型 ││
│  └──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘│
│     │             │             │             │    │
│     ▼             ▼             ▼             ▼    │
│  ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐│
│  │Batch│       │Batch│       │Batch│       │Batch││
│  │  0  │       │  1  │       │  2  │       │  3  ││
│  └──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘│
│     │             │             │             │    │
│     ▼             ▼             ▼             ▼    │
│  ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐│
│  │梯度 │       │梯度 │       │梯度 │       │梯度 ││
│  │  0  │       │  1  │       │  2  │       │  3  ││
│  └──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘│
│     │             │             │             │    │
│     └─────────────┴──────┬──────┴─────────────┘    │
│                          │                          │
│                    全局归约（All-Reduce）             │
│                   （梯度平均）                        │
│                          │                          │
│              ┌───────────┴───────────┐              │
│              ▼                       ▼              │
│         GPU 0,1,2,3              所有 GPU           │
│         更新参数                使用更新后的参数       │
└──────────────────────────────────────────────────┘
```

**局限性**：内存以 O(N) 的比例扩展，其中 N 是模型大小。对于一个 70B 参数的模型（fp16），每个 GPU 仅参数就需要约 140GB，超过了单个 GPU 的最大容量。

### 10.2.3 DeepSpeed ZeRO（零冗余优化器）

DeepSpeed 引入了 ZeRO，系统地消除了数据并行 GPU 之间的内存冗余：

```
┌──────────────────────────────────────────────────────────┐
│           ZeRO 内存优化阶段                                 │
│                                                            │
│  标准 DP（无 ZeRO）：                                       │
│  ┌─────────────────────────────────────────┐              │
│  │ GPU 0: [参数] [梯度] [优化器状态]         │ = 4x 模型    │
│  │ GPU 1: [参数] [梯度] [优化器状态]         │   内存       │
│  └─────────────────────────────────────────┘              │
│                                                            │
│  ZeRO Stage 1（os）：                                      │
│  ┌─────────────────────────────────────────┐              │
│  │ GPU 0: [参数] [梯度] [优化器 1/4]         │ = 2.25x      │
│  │ GPU 1: [参数] [梯度] [优化器 1/4]         │   模型       │
│  │ GPU 2: [参数] [梯度] [优化器 1/4]         │   内存       │
│  │ GPU 3: [参数] [梯度] [优化器 1/4]         │              │
│  └─────────────────────────────────────────┘              │
│                                                            │
│  ZeRO Stage 2（os + g）：                                   │
│  ┌─────────────────────────────────────────┐              │
│  │ GPU 0: [参数] [梯度 1/4] [优化器 1/4]     │ = 1.5x       │
│  │ GPU 1: [参数] [梯度 1/4] [优化器 1/4]     │   模型       │
│  │ GPU 2: [参数] [梯度 1/4] [优化器 1/4]     │   内存       │
│  │ GPU 3: [参数] [梯度 1/4] [优化器 1/4]     │              │
│  └─────────────────────────────────────────┘              │
│                                                            │
│  ZeRO Stage 3（os + g + p）：                               │
│  ┌─────────────────────────────────────────┐              │
│  │ GPU 0: [参数 1/4] [梯度 1/4] [优化 1/4]   │ = 1x         │
│  │ GPU 1: [参数 1/4] [梯度 1/4] [优化 1/4]   │   模型       │
│  │ GPU 2: [参数 1/4] [梯度 1/4] [优化 1/4]   │   内存       │
│  │ GPU 3: [参数 1/4] [梯度 1/4] [优化 1/4]   │              │
│  └─────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

#### 内存预算计算

对于使用混合精度（bf16/fp16 + fp32 优化器状态）训练的模型，参数数量为 Φ：

| 组件 | 每参数字节数 | 70B 模型内存 |
|------|------------|-------------|
| 模型参数（bf16） | 2 | 140 GB |
| 梯度（bf16） | 2 | 140 GB |
| 优化器状态（fp32 参数 + 动量 + 方差） | 12 | 840 GB |
| **总计（标准 DP）** | **16** | **1,120 GB** |
| **ZeRO-3** | 16 / GPU数 | 1,120 / N GB |

#### DeepSpeed ZeRO 配置

```python
import deepspeed

# 70B 模型的 DeepSpeed ZeRO Stage 3 配置
ds_config = {
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "bfloat16": {
        "enabled": True
    },
    "zero_allow_untested_optimizer": True
}

# 初始化模型和引擎
model = YourLLMModel(config)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters()
)

# 训练循环
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### 10.2.4 FSDP（全分片数据并行）

PyTorch 原生的 FSDP 提供了与 ZeRO-3 类似的功能，与 PyTorch 集成更紧密：

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 定义要包装的层
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={TransformerBlock}
)

# 初始化 FSDP 模型
model = FSDP(
    LLMModel(config),
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    forward_prefetch=True,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,
)

# 标准 PyTorch 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch)
    loss.backward()  # FSDP 自动处理梯度通信
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
```

---

## 10.3 模型并行与流水线并行

### 10.3.1 张量并行（Tensor Parallelism, TP）

当单个模型层太大而无法放入一个 GPU 时，张量并行将单个层分割到多个 GPU 上。

📌 **核心概念**：张量并行在单个层内将计算分割到多个 GPU，减少每个 GPU 的内存占用并支持更大的批处理大小。

```
┌──────────────────────────────────────────────────────────┐
│              张量并行（线性层）                              │
│                                                            │
│  GPU 0                        GPU 1                       │
│  ┌──────────────────┐        ┌──────────────────┐        │
│  │                  │        │                  │        │
│  │  A₀ = A[:, :n/2] │        │  A₁ = A[:, n/2:] │        │
│  │       │          │        │       │          │        │
│  │       ▼          │        │       ▼          │        │
│  │  Y₀ = X · A₀     │        │  Y₁ = X · A₁     │        │
│  │       │          │        │       │          │        │
│  │       ▼          │        │       ▼          │        │
│  │  Output₀         │        │  Output₁         │        │
│  └──────────────────┘        └──────────────────────────┘│
│                                                            │
│  全收集或归约散射以合并结果                                   │
└──────────────────────────────────────────────────────────┘
```

#### Megatron-LM 风格的列-行并行

最常见的 TP 方案以特定模式分割权重矩阵：

```python
# Megatron-LM 风格的列并行 + 行并行 MLP
class ParallelMLP(nn.Module):
    def __init__(self, hidden_size, ffn_size, tp_size):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = dist.get_rank() % tp_size

        # 列并行：分割输出维度
        self.w1 = ColumnParallelLinear(hidden_size, ffn_size // tp_size)
        self.w3 = ColumnParallelLinear(hidden_size, ffn_size // tp_size)

        # 行并行：分割输入维度
        self.w2 = RowParallelLinear(ffn_size // tp_size, hidden_size)

    def forward(self, x):
        # x 已经在 TP 组中分片
        h = F.silu(self.w1(x)) * self.w3(x)
        return self.w2(h)  # All-Reduce 在 RowParallelLinear 内部完成
```

```
┌──────────────────────────────────────────────────────────┐
│         Megatron-LM 列-行并行 MLP                          │
│                                                            │
│        输入 X（已复制）                                      │
│              │                                             │
│     ┌────────┴────────┐                                    │
│     │                 │                                    │
│     ▼                 ▼                                    │
│  ┌──────┐         ┌──────┐                               │
│  │  W1  │         │  W3  │   列并行                        │
│  │(col) │         │(col) │   分割输出维度                   │
│  └──┬───┘         └──┬───┘                                │
│     │                 │                                    │
│     ▼                 ▼                                    │
│  ┌──────┐         ┌──────┐                               │
│  │ SiLU │         │      │   激活函数                      │
│  └──┬───┘         └──┬───┘                                │
│     │                 │                                    │
│     └────────┬────────┘                                    │
│              ▼                                             │
│         ┌────────┐                                        │
│         │  H = h1 * h3  │                                   │
│         └───┬────┘                                        │
│             │                                              │
│             ▼                                              │
│         ┌──────┐                                          │
│         │  W2  │   行并行                                   │
│         │(row) │   分割输入维度 + All-Reduce                 │
│         └──┬───┘                                           │
│            │                                               │
│            ▼                                               │
│        输出 Y（已归约）                                      │
└──────────────────────────────────────────────────────────┘
```

### 10.3.2 流水线并行（Pipeline Parallelism, PP）

流水线并行将模型的层分割到不同 GPU，创建一个流水线，其中不同阶段处理不同的微批次。

```
┌──────────────────────────────────────────────────────────────────┐
│              流水线并行（GPipe 风格）                               │
│                                                                    │
│  时间 ──────────────────────────────────────────────────────▶     │
│                                                                    │
│  Stage 0    │  ┌────┐  │        │        │        │             │
│  (GPU 0)    │  │MB 0│  │        │        │        │             │
│  层 0-19    │  └────┘  │        │        │        │             │
│             │        │  ┌────┐  │        │        │             │
│             │        │  │MB 1│  │        │        │             │
│             │        │  └────┘  │        │        │             │
│                                                                    │
│  Stage 1    │        │        │  ┌────┐  │        │             │
│  (GPU 1)    │        │        │  │MB 0│  │        │             │
│  层 20-39   │        │        │  └────┘  │        │             │
│             │        │        │        │  ┌────┐  │             │
│             │        │        │        │  │MB 1│  │             │
│             │        │        │        │  └────┘  │             │
│                                                                    │
│  Stage 2    │        │        │        │        │  ┌────┐       │
│  (GPU 2)    │        │        │        │        │  │MB 0│       │
│  层 40-59   │        │        │        │        │  └────┘       │
│                                                                    │
│  Stage 3    │        │        │        │        │        │       │
│  (GPU 3)    │        │        │        │        │        │       │
│  层 60-79   │        │        │        │        │        │       │
│                                                                    │
│  MB = 微批次（Micro-Batch）                                        │
│  Bubble = 阶段之间的空闲时间（流水线气泡）                           │
└──────────────────────────────────────────────────────────────────┘
```

#### 流水线气泡分析

流水线气泡是关键的效率问题：

```
流水线效率 = (T - 气泡) / T
其中：
  T = P 个阶段处理 M 个微批次的总时间
  气泡 = (P - 1) × 每微批次处理时间

GPipe 效率：
  效率 ≈ (M - 1) / (M + P - 1)
  对于 P=4, M=32：效率 ≈ 31/35 ≈ 88.6%
  对于 P=4, M=8：  效率 ≈ 7/11 ≈ 63.6%
```

#### 1F1B 调度

1F1B（一次前向一次反向）通过交错前向和反向传播来减少流水线气泡：

```
┌──────────────────────────────────────────────────────────────┐
│                    1F1B 调度                                    │
│                                                                │
│  GPU 0: F0  F1  F2  F3  B0  F4  B1  F5  B2  F6  B3  F7    │
│  GPU 1:     F0  F1  F2  F3  B0  F4  B1  F5  B2  F6  B3     │
│  GPU 2:         F0  F1  F2  F3  B0  F4  B1  F5  B2  F6     │
│  GPU 3:             F0  F1  F2  F3  B0  F4  B1  F5  B2     │
│                                                                │
│  F = 前向传播    B = 反向传播                                   │
│  数字 = 微批次索引                                              │
│                                                                │
│  预热阶段后，每个 GPU 交替执行 F 和 B：                           │
│  气泡减少为：(P-1) / (P + M*(P-1)) × 时间                      │
└──────────────────────────────────────────────────────────────┘
```

### 10.3.3 3D 并行

生产环境训练结合所有三种并行策略：

```
┌──────────────────────────────────────────────────────────────┐
│                    3D 并行                                      │
│                                                                │
│  数据并行（DP）：跨节点                                         │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                                                      │    │
│  │  ┌─────────────────┐    ┌─────────────────┐        │    │
│  │  │   Node 0        │    │   Node 1        │  DP    │    │
│  │  │  TP=4, PP=2     │    │  TP=4, PP=2     │ ◄────▶│    │
│  │  │  DP Rank 0      │    │  DP Rank 1      │        │    │
│  │  └─────────────────┘    └─────────────────┘        │    │
│  │                                                      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  节点内：                                                       │
│  ┌──────────────────────────────────────────────┐            │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │            │
│  │  │GPU 0 │ │GPU 1 │ │GPU 2 │ │GPU 3 │       │            │
│  │  │TP=0  │ │TP=1  │ │TP=2  │ │TP=3  │       │            │
│  │  │PP=0  │ │PP=0  │ │PP=1  │ │PP=1  │       │            │
│  │  └──────┘ └──────┘ └──────┘ └──────┘       │            │
│  │                                              │            │
│  │  TP 在阶段内，PP 跨阶段                       │            │
│  └──────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

### 10.3.4 专家并行（MoE）

混合专家（MoE）增加了不同维度的并行：

```
┌──────────────────────────────────────────────────────────┐
│              MoE 层架构                                     │
│                                                            │
│  输入 token: "模型微调的最佳实践"                             │
│       │                                                    │
│       ▼                                                    │
│  ┌─────────────────────────┐                              │
│  │   路由器/门控网络          │                              │
│  │   （可学习的）             │                              │
│  └───────────┬─────────────┘                              │
│              │                                             │
│     ┌────────┼────────┬────────┬────────┐                │
│     ▼        ▼        ▼        ▼        ▼                │
│  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐              │
│  │专家 0││专家 1││专家 2││专家 3││专家 4│              │
│  │(GPU0)││(GPU1)││(GPU2)││(GPU3)││(GPU4)│              │
│  └──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘              │
│     │       │       │       │       │                    │
│     │       ▼       │       │       │                    │
│     │  ┌────────┐   │       │       │                    │
│     │  │Top-K   │   │       │       │                    │
│     │  │选择    │   │       │       │                    │
│     │  │(K=2)   │   │       │       │                    │
│     │  └───┬────┘   │       │       │                    │
│     │      │        │       │       │                    │
│     │      ├────────┘       │       │                    │
│     │      │                │       │                    │
│     │      └────────────────┘       │                    │
│     │                               │                    │
│     └───────────┬───────────────────┘                    │
│                 │                                         │
│                 ▼                                         │
│           组合输出                                         │
└──────────────────────────────────────────────────────────┘

MoE 缩放：E 个专家，Top-K 路由：
  - 每个 token 激活 E 个专家中的 K 个
  - 总参数：E ×（FFN 参数）
  - 每个 token 的活跃参数：K ×（FFN 参数）
  - 示例：Mixtral 8x7B 总参数 47B，每个 token 活跃约 13B
```

---

## 10.4 内存优化技术

### 10.4.1 激活检查点（梯度检查点）

基本的权衡：计算量 vs 内存。

```
┌──────────────────────────────────────────────────────────┐
│          激活检查点权衡                                      │
│                                                            │
│  无检查点：                                                  │
│  ┌────────────────────────────────────┐                  │
│  │ 存储所有激活：O(L × B × S × H)      │                  │
│  │ 内存：高                            │                  │
│  │ 计算量：标准                         │                  │
│  └────────────────────────────────────┘                  │
│                                                            │
│  完全检查点：                                                │
│  ┌────────────────────────────────────┐                  │
│  │ 仅存储边界：O(√L × B × S × H)       │                  │
│  │ 内存：低（降低 √L 倍）              │                  │
│  │ 计算量：+33%（重新计算激活）         │                  │
│  └────────────────────────────────────┘                  │
│                                                            │
│  选择性检查点：                                               │
│  ┌────────────────────────────────────┐                  │
│  │ 检查点注意力，不检查点 FFN           │                  │
│  │ 内存：中等                          │                  │
│  │ 计算量：+10-20%                     │                  │
│  └────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────┘
```

```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerBlockWithCheckpointing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = SwiGLU(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, x):
        # 对整个块使用检查点
        return checkpoint(self._forward_impl, x, use_reentrant=False)

    def _forward_impl(self, x):
        h = x + self.attention(self.norm1(x))
        return h + self.ffn(self.norm2(h))

# 选择性检查点 - 仅检查点注意力
def selective_checkpointing_forward(block, x):
    h = x + checkpoint(block.attention, block.norm1(x),
                        use_reentrant=False)
    # FFN 不使用检查点计算（更快，更多内存）
    return h + block.ffn(block.norm2(h))
```

### 10.4.2 Flash Attention

Flash Attention（Dao 等，2022）是一种内存高效的注意力算法，避免了完整注意力矩阵的物化：

```
┌──────────────────────────────────────────────────────────┐
│          标准注意力 vs Flash Attention                      │
│                                                            │
│  标准注意力：                                                │
│  ┌────────────────────────────────────┐                  │
│  │ Q, K, V (N × d)                    │                  │
│  │     │                              │                  │
│  │     ▼                              │                  │
│  │ S = Q @ K^T (N × N)   ← 存储完整 N×N 矩阵             │
│  │     │                              │                  │
│  │     ▼                              │                  │
│  │ P = softmax(S) (N × N)            │                  │
│  │     │                              │                  │
│  │     ▼                              │                  │
│  │ O = P @ V (N × d)                  │                  │
│  │                                    │                  │
│  │ 内存：O(N²)  ← 长序列时成为瓶颈     │                  │
│  └────────────────────────────────────┘                  │
│                                                            │
│  Flash Attention：                                          │
│  ┌────────────────────────────────────┐                  │
│  │ Q, K, V 分块处理                    │                  │
│  │ 在 SRAM 中处理块（快速！）           │                  │
│  │ 在线 softmax（无需完整 N×N）         │                  │
│  │                                    │                  │
│  │ for Q 的每个块：                    │                  │
│  │   for K,V 的每个块：               │                  │
│  │     加载到 SRAM                     │                  │
│  │     计算部分注意力                   │                  │
│  │     用运行最大值更新输出             │                  │
│  │                                    │                  │
│  │ 内存：O(N)   ← 线性！              │                  │
│  │ 速度：2-4x 更快（更好的内存访问模式）│                  │
│  └────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────┘
```

```python
# 使用 Flash Attention（通过 PyTorch 2.0+ 或 flash-attn）
import torch
import torch.nn.functional as F

# 方法 1：PyTorch 2.0 SDPA（缩放点积注意力）
# 当可能时自动使用 Flash Attention
def attention_forward(q, k, v, mask=None):
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        is_causal=True,
        # PyTorch 自动选择：Flash、Memory-Efficient 或 Math
    )

# 方法 2：flash-attn 库（更多控制）
from flash_attn import flash_attn_func

def flash_attention_forward(q, k, v):
    # q, k, v 形状：(batch, seqlen, nheads, headdim)
    return flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        causal=True,
        window_size=(-1, -1),  # 全注意力
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False
    )
```

### 10.4.3 CPU 卸载

对于即使使用 ZeRO-3 也装不下的模型，优化器状态和参数可以卸载到 CPU：

```
┌──────────────────────────────────────────────────────────┐
│              CPU 卸载策略                                    │
│                                                            │
│  GPU 内存（80GB H100）                                     │
│  ┌────────────────────────────────────────┐              │
│  │  ┌──────────┐  ┌──────────┐           │              │
│  │  │ 激活      │  │ 前向传播  │           │              │
│  │  │（动态）   │  │          │           │              │
│  │  └──────────┘  └──────────┘           │              │
│  │  ┌──────────────────────────┐         │              │
│  │  │ 当前层参数               │         │              │
│  │  │（固定内存）              │         │              │
│  │  └──────────────────────────┘         │              │
│  └────────────────────────────────────────┘              │
│                                                            │
│  CPU 内存（512GB+）                                        │
│  ┌────────────────────────────────────────┐              │
│  │  ┌──────────────────────────────────┐  │              │
│  │  │ 优化器状态（fp32）                │  │              │
│  │  └── AdamW: 参数 + 动量 + 方差       │  │              │
│  │  │ 内存：每参数 12 字节             │  │              │
│  │  └──────────────────────────────────┘  │              │
│  │  ┌──────────────────────────────────┐  │              │
│  │  │ 卸载的参数（bf16）                │  │              │
│  │  │ 内存：每参数 2 字节              │  │              │
│  │  └──────────────────────────────────┘  │              │
│  │  ┌──────────────────────────────────┐  │              │
│  │  │ 梯度累积缓冲区                    │  │              │
│  │  └──────────────────────────────────┘  │              │
│  └────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

### 10.4.4 内存预算汇总

训练 70B 参数模型的完整内存分析：

```
┌────────────────────────────────────────────────────────────┐
│        70B 模型内存预算（训练，bf16）                         │
│                                                              │
│  组件               │ 每 GPU (DP=1)  │ 每 GPU (ZeRO-3, 64 GPUs)
│  ───────────────────│───────────────│──────────────────────
│  模型参数 (bf16)     │    140 GB     │     2.19 GB
│  梯度 (bf16)         │    140 GB     │     2.19 GB
│  优化器 (fp32)       │    840 GB     │    13.13 GB
│  激活 (8K ctx)       │    ~200 GB    │    ~200 GB（不分片）
│  ───────────────────│───────────────│──────────────────────
│  总计                │  ~1,320 GB    │   ~218 GB
│                                                              │
│  使用 CPU 卸载 (ZeRO-3)：                                    │
│  ───────────────────│───────────────│──────────────────────
│  GPU 内存            │     N/A       │    ~218 GB（需要 3× H100）
│  CPU 内存            │     N/A       │    ~993 GB
│                                                              │
│  使用激活检查点：                                              │
│  ───────────────────│───────────────│──────────────────────
│  激活（减少后）      │    ~20 GB     │    ~20 GB
│  总 GPU              │     N/A       │    ~38 GB (ZeRO-3 + CKPT)
└────────────────────────────────────────────────────────────┘
```

---

## 10.5 混合精度训练

### 10.5.1 数值格式比较

📌 **核心概念**：混合精度训练使用低精度格式（bf16/fp16）进行大部分计算，同时维护 fp32 的主副本来处理关键操作。

```
┌────────────────────────────────────────────────────────────┐
│           数值格式比较                                        │
│                                                              │
│  格式       │ 位数 │ 符号 │ 指数   │ 尾数    │ 范围         │
│  ───────────│──────│──────│────────│─────────│──────────────│
│  fp32       │  32  │  1   │    8   │    23   │ ±3.4e38     │
│  fp16       │  16  │  1   │    5   │    10   │ ±65504      │
│  bf16       │  16  │  1   │    8   │     7   │ ±3.4e38     │
│  fp8 (E4M3) │   8  │  1   │    4   │     3   │ ±448        │
│  fp8 (E5M2) │   8  │  1   │    5   │     2   │ ±57344      │
│  int8       │   8  │  1   │    -   │     -   │ -128~127    │
│  int4       │   4  │  1   │    -   │     -   │ -8~7        │
│  nf4        │   4  │  -   │    -   │     -   │ 归一化       │
│                                                              │
│  关键洞察：                                                   │
│  - fp16：范围小，需要损失缩放                                │
│  - bf16：与 fp32 相同范围，精度较低                           │
│  - fp8：两种变体适用于不同需求                                │
│  - E4M3：更高精度（推理）                                    │
│  - E5M2：更高范围（训练梯度）                                 │
└────────────────────────────────────────────────────────────┘
```

### 10.5.2 BF16 vs FP16 训练

```
┌────────────────────────────────────────────────────────────┐
│              BF16 vs FP16 训练对比                            │
│                                                              │
│  BF16（Brain Float 16）：                                     │
│  ✅ 与 fp32 相同的指数范围（±3.4e38）                         │
│  ✅ 无需损失缩放                                             │
│  ✅ 训练更稳定                                               │
│  ❌ 精度较低（7位尾数 vs 10位）                               │
│  ❌ 旧 GPU 不支持（A100 之前）                               │
│                                                              │
│  FP16（半精度）：                                              │
│  ✅ 更高精度（10位尾数）                                     │
│  ✅ GPU 支持更广泛                                           │
│  ❌ 范围小（±65504），需要损失缩放                            │
│  ❌ 容易溢出/下溢                                            │
│  ❌ 需要仔细调整缩放因子                                      │
│                                                              │
│  推荐（2024-2026）：                                          │
│  ┌────────────────────────────────────────────┐            │
│  │  GPU >= A100/H100 → 使用 BF16              │            │
│  │  GPU < A100        → 使用 FP16 + GradScaler│            │
│  │  H100/H200         → 考虑 FP8 获得额外加速  │            │
│  └────────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────────┘
```

### 10.5.3 FP8 训练（H100/H200）

H100 GPU 上的 FP8 训练为矩阵运算提供 2 倍吞吐量提升：

```python
import torch
import transformer_engine.pytorch as te

class FP8TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Transformer Engine 自动处理 FP8
        self.attention = te.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            bias=True,
            params_dtype=torch.bfloat16,
            tp_size=config.tensor_parallel_size,
        )
        self.mlp = te.Sequential(
            te.Linear(config.hidden_size, config.ffn_size, bias=True),
            torch.nn.SiLU(),
            te.Linear(config.ffn_size, config.hidden_size, bias=True),
        )
        self.norm1 = te.LayerNorm(config.hidden_size)
        self.norm2 = te.LayerNorm(config.hidden_size)

    def forward(self, x):
        h = x + self.attention(self.norm1(x), self.norm1(x),
                               self.norm1(x), is_self_attn=True)
        return h + self.mlp(self.norm2(h))

# H100 上的 FP8 训练配置
fp8_recipe = te.common.recipe.DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=te.common.recipe.Format.HYBRID,
    amax_history_length=16,
    amax_compute_algo="max",
)
```

### 10.5.4 梯度累积

当所需的批处理大小超过 GPU 内存容量时：

```python
import torch
import deepspeed

def train_with_gradient_accumulation(
    model_engine,
    dataloader,
    gradient_accumulation_steps=16,
):
    """
    梯度累积：通过在多个微批次上累积梯度来模拟大批处理，
    然后才进行参数更新。
    """
    model_engine.train()
    optimizer = model_engine.optimizer

    for step, batch in enumerate(dataloader):
        # 前向 + 反向（梯度累积）
        loss = model_engine(batch) / gradient_accumulation_steps
        model_engine.backward(loss)

        # 每 N 步才更新一次
        if (step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model_engine.module.parameters(), max_norm=1.0
            )
            # 优化器更新（这里发生 all-reduce）
            model_engine.step()
            optimizer.zero_grad()

            # 日志
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}: loss={loss.item():.4f}, lr={current_lr:.2e}")

# 有效批处理大小 = micro_batch_size × num_gpus × gradient_accumulation_steps
# 示例：2 × 64 × 16 = 每步 2048 个样本
```

---

## 📝 练习：搭建小规模分布式训练环境

### 目标

在 2 个 GPU 上使用 DeepSpeed ZeRO Stage 2 搭建分布式训练环境，并在文本数据集上训练一个小型语言模型（约125M参数）。

### 前提条件

- 2× GPU，每个至少 16GB 显存
- Python 3.10+，PyTorch 2.0+，DeepSpeed 0.12+

### 步骤 1：环境搭建

```bash
# 创建虚拟环境
conda create -n llm-train python=3.11
conda activate llm-train

# 安装依赖
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed==0.12.0 transformers datasets accelerate
pip install flash-attn --no-build-isolation

# 验证环境
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
python -c "import deepspeed; print(f'DeepSpeed版本: {deepspeed.__version__}')"
```

### 步骤 2：模型配置

```python
# model_config.py
from transformers import LlamaConfig

config = LlamaConfig(
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,  # GQA
    max_position_embeddings=2048,
    vocab_size=32000,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)

# 约 125M 参数
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

### 步骤 3：DeepSpeed 配置

```json
// ds_config.json
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "bf16": {
        "enabled": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 1e-5,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 100,
            "total_num_steps": 10000
        }
    },
    "wall_clock_breakdown": false
}
```

### 步骤 4：训练脚本

```python
# train.py
import torch
import deepspeed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def main():
    # 加载分词器并创建模型
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        hidden_size=768, intermediate_size=2048,
        num_hidden_layers=12, num_attention_heads=12,
        num_key_value_heads=4, max_position_embeddings=2048,
        vocab_size=32000,
    )
    model = AutoModelForCausalLM.from_config(config)

    # 加载数据集
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                     remove_columns=["text"])
    dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

    # 初始化 DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config="ds_config.json",
        model_parameters=model.parameters()
    )

    # 训练循环
    model_engine.train()
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(model_engine.device)
        labels = input_ids.clone()

        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()

        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

        if step >= 1000:
            break

    # 保存检查点
    model_engine.save_checkpoint("./checkpoints", tag="step_1000")

if __name__ == "__main__":
    main()
```

### 步骤 5：启动分布式训练

```bash
# 使用 DeepSpeed 在 2 个 GPU 上启动
deepspeed --num_gpus=2 train.py \
    --deepspeed ds_config.json \
    --output_dir ./output \
    --logging_steps 10 \
    --save_steps 500

# 监控 GPU 使用情况
watch -n 1 nvidia-smi

# 监控训练指标
tensorboard --logdir ./output/runs
```

### 预期输出

```
[2026-01-15 10:00:00] Step 0   | Loss: 10.8234 | LR: 1.00e-05
[2026-01-15 10:00:15] Step 100 | Loss: 5.2341  | LR: 3.00e-04
[2026-01-15 10:00:30] Step 200 | Loss: 4.8723  | LR: 2.98e-04
[2026-01-15 10:00:45] Step 300 | Loss: 4.6512  | LR: 2.96e-04
...
[2026-01-15 10:02:30] Step 1000 | Loss: 4.1234 | LR: 2.80e-04
```

### 探索问题

1. **如果将梯度累积步数从 8 增加到 32 会怎样？** 这如何影响收敛速度和最终损失？

2. **比较 ZeRO Stage 1 vs Stage 2 vs Stage 3。** 使用 `nvidia-smi` 测量每个阶段的 GPU 内存使用量。

3. **尝试启用激活检查点。** 内存节省和训练速度之间的权衡是什么？

4. **实验学习率调度。** 预热时长如何影响早期训练稳定性？

---

## 本章小结

本章建立了构建和训练大语言模型的基础架构知识：

| 主题 | 关键要点 |
|------|---------|
| **Transformer 回顾** | 现代 LLM 标准使用 RoPE、GQA、SwiGLU、RMSNorm |
| **缩放定律** | 最优训练需要平衡模型大小和数据大小（Chinchilla） |
| **ZeRO 阶段** | Stage 3 支持以可管理的单 GPU 内存训练 70B+ 模型 |
| **3D 并行** | TP + PP + DP 是训练 100B+ 模型的标准方案 |
| **Flash Attention** | O(N) 内存，比标准注意力快 2-4 倍 |
| **混合精度** | BF16 是现代 GPU 的默认选择；H100+ 上使用 FP8 可获 2 倍加速 |
| **激活检查点** | 33% 计算开销换取大量内存节省 |

### 章节路线图

```
第10章（当前）：LLM 架构设计基础
    │
    ├── 第11章：LLM 推理架构
    │   └── 如何高效地服务训练好的模型
    │
    ├── 第12章：RAG 系统架构
    │   └── 如何用外部知识增强 LLM
    │
    └── 第13章：模型微调架构
        └── 如何将预训练模型适配到特定任务
```

---

## 参考文献

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Rajbhandari, S., et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC'20.
3. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS.
4. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." NeurIPS.
5. Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv.
6. Korthikanti, V., et al. (2022). "Reducing Activation Recomputation in Large Transformer Models." MLSys.
7. Narayanan, D., et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." SC'21.
8. NVIDIA Transformer Engine 文档. https://docs.nvidia.com/deeplearning/transformer-engine/
9. DeepSpeed 文档. https://www.deepspeed.ai/
10. PyTorch FSDP 教程. https://pytorch.org/docs/stable/fsdp.html

---

*下一章：[第11章 - LLM 推理架构](chapter-11.md) →*
