# Chapter 10: LLM Architecture Design Fundamentals

> 🟢 Beginner → 🔴 Advanced | 25-30 min read | Part 4: Large Model Architecture

---

## Table of Contents

- [10.1 Transformer Architecture Review](#101-transformer-architecture-review)
- [10.2 Large Model Training Architecture](#102-large-model-training-architecture)
- [10.3 Model Parallelism & Pipeline Parallelism](#103-model-parallelism--pipeline-parallelism)
- [10.4 Memory Optimization Techniques](#104-memory-optimization-techniques)
- [10.5 Mixed Precision Training](#105-mixed-precision-training)
- [📝 Exercise: Building a Small-Scale Distributed Training Environment](#-exercise-building-a-small-scale-distributed-training-environment)
- [Summary](#summary)
- [References](#references)

---

## 10.1 Transformer Architecture Review

### 10.1.1 The Self-Attention Mechanism

The Transformer architecture, introduced by Vaswani et al. (2017) in "Attention Is All You Need," fundamentally reshaped natural language processing and now serves as the backbone of virtually all large language models (LLMs). Before diving into the architectural considerations for large-scale models, we must first solidify our understanding of the foundational building blocks.

📌 **Key Concept**: Self-attention computes a weighted representation of a sequence where each token's representation depends on all other tokens in the sequence.

The core self-attention operation can be expressed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q** (Query): What am I looking for?
- **K** (Key): What do I contain?
- **V** (Value): What information do I provide?
- **d_k**: Dimension of the key vectors (scaling factor)

Multi-Head Attention extends this by running multiple attention computations in parallel:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

The architectural diagram of a standard Transformer block:

```
┌─────────────────────────────────────────────────────┐
│                Transformer Block                     │
│                                                      │
│  Input Embedding                                     │
│       │                                              │
│       ▼                                              │
│  ┌─────────────────────────────────┐                │
│  │    Multi-Head Self-Attention     │                │
│  │  ┌─────┐ ┌─────┐ ┌─────┐      │                │
│  │  │  Q  │ │  K  │ │  V  │      │                │
│  │  └──┬──┘ └──┬──┘ └──┬──┘      │                │
│  │     └───────┼───────┘          │                │
│  │             ▼                   │                │
│  │     Scaled Dot-Product         │                │
│  │         Attention              │                │
│  └─────────────┬───────────────────┘                │
│                │                                     │
│       ───────────── (Residual Connection)            │
│                │                                     │
│         Layer Normalization                          │
│                │                                     │
│                ▼                                     │
│  ┌─────────────────────────────────┐                │
│  │   Feed-Forward Network (FFN)     │                │
│  │                                  │                │
│  │   Linear → Activation → Linear   │                │
│  └─────────────┬───────────────────┘                │
│                │                                     │
│       ───────────── (Residual Connection)            │
│                │                                     │
│         Layer Normalization                          │
│                │                                     │
│                ▼                                     │
│  Output Embedding                                    │
└─────────────────────────────────────────────────────┘
```

### 10.1.2 From Transformer to LLM: Scaling Laws

The transition from a vanilla Transformer to a large language model involves systematic scaling along three axes:

| Dimension | Small Transformer | Large LLM (e.g., LLaMA-3 405B) |
|-----------|-------------------|--------------------------------|
| Parameters | 10M - 100M | 10B - 405B+ |
| Layers | 6 - 12 | 80 - 128 |
| Hidden Dim | 256 - 768 | 8192 - 16384 |
| Attention Heads | 4 - 12 | 64 - 128 |
| Context Length | 512 - 2048 | 8K - 128K+ |
| Training Data | GBs | TBs |

📌 **Key Concept**: Chinchilla Scaling Laws (Hoffmann et al., 2022) showed that optimal performance requires balancing model size and data size. A 70B parameter model should be trained on roughly 1.4T tokens for optimal compute efficiency.

The scaling laws follow power-law relationships:

```
L(N, D) ∝ (N_c / N)^α_N + (D_c / D)^α_D
```

Where L is the loss, N is the number of parameters, D is the dataset size, and N_c, D_c, α_N, α_D are fitted constants.

### 10.1.3 Modern Architectural Innovations (2024-2026)

Modern LLMs have evolved significantly from the original Transformer. Key innovations include:

**Grouped Query Attention (GQA)**:
Instead of standard multi-head attention where each head has its own Q, K, V projections, GQA shares K and V projections across groups of query heads. This dramatically reduces KV cache memory during inference.

```
Standard MHA:  Q_heads=64, K_heads=64, V_heads=64
GQA:           Q_heads=64, K_heads=8,  V_heads=8   (8 groups of 8 Q heads)
MQA:           Q_heads=64, K_heads=1,  V_heads=1   (all share same K,V)
```

**Rotary Position Embeddings (RoPE)**:
Instead of learned absolute position embeddings or relative position biases, RoPE encodes position information directly into the attention computation through rotation matrices. This allows better length generalization.

```python
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute RoPE frequencies for rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to queries and keys."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**SwiGLU Activation in FFN**:
Modern LLMs replace ReLU/GELU with SwiGLU in the feed-forward network:

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

**RMSNorm over LayerNorm**:
RMSNorm removes the mean-centering step, making it computationally cheaper:

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

## 10.2 Large Model Training Architecture

### 10.2.1 The Training Pipeline at Scale

Training a large language model is one of the most computationally intensive tasks in modern computing. A state-of-the-art model like LLaMA-3 405B required approximately 30.84 million GPU hours on H100 GPUs.

```
┌──────────────────────────────────────────────────────────────────┐
│                   Large Model Training Pipeline                   │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   Data    │───▶│  Token   │───▶│ Sharding │───▶│   Data    │  │
│  │ Ingestion │    │ ization  │    │  & Load  │    │  Loader   │  │
│  └──────────┘    └──────────┘    └──────────┘    └─────┬────┘  │
│                                                         │        │
│                                                         ▼        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Distributed Training Loop                    │   │
│  │                                                           │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  Forward Pass (across all GPUs)                  │     │   │
│  │  │  ├── GPU 0: Layers 0-19    ─────────────┐      │     │   │
│  │  │  ├── GPU 1: Layers 20-39  ──────────────┤      │     │   │
│  │  │  ├── GPU 2: Layers 40-59  ──────────────┤      │     │   │
│  │  │  └── GPU 3: Layers 60-79  ──────────────┘      │     │   │
│  │  └──────────────────────┬──────────────────────────┘     │   │
│  │                         │                                 │   │
│  │                         ▼                                 │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  Loss Computation & Gradient Calculation         │     │   │
│  │  └──────────────────────┬──────────────────────────┘     │   │
│  │                         │                                 │   │
│  │                         ▼                                 │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  Backward Pass + Gradient All-Reduce             │     │   │
│  │  │  ├── ZeRO Stage 1: Optimizer State Sharding     │     │   │
│  │  │  ├── ZeRO Stage 2: + Gradient Sharding          │     │   │
│  │  │  └── ZeRO Stage 3: + Parameter Sharding         │     │   │
│  │  └──────────────────────┬──────────────────────────┘     │   │
│  │                         │                                 │   │
│  │                         ▼                                 │   │
│  │  ┌─────────────────────────────────────────────────┐     │   │
│  │  │  Optimizer Step (AdamW / Adam-mini)              │     │   │
│  │  └──────────────────────┬──────────────────────────┘     │   │
│  │                         │                                 │   │
│  │                    [Repeat]                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Checkpointing & Monitoring                    │   │
│  │  ├── Periodic checkpoint saves (every N steps)           │   │
│  │  ├── Loss/gradient norm logging                          │   │
│  │  └── GPU utilization & memory monitoring                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 10.2.2 Data Parallelism (DP)

The simplest form of distributed training. Each GPU holds a complete copy of the model, processes different data batches, and synchronizes gradients.

```
┌──────────────────────────────────────────────────┐
│              Data Parallelism                      │
│                                                    │
│  GPU 0          GPU 1          GPU 2          GPU 3 │
│  ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐│
│  │Full │       │Full │       │Full │       │Full ││
│  │Model│       │Model│       │Model│       │Model││
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
│  │Grad │       │Grad │       │Grad │       │Grad ││
│  │  0  │       │  1  │       │  2  │       │  3  ││
│  └──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘│
│     │             │             │             │    │
│     └─────────────┴──────┬──────┴─────────────┘    │
│                          │                          │
│                    All-Reduce                       │
│                   (Gradient Average)                 │
│                          │                          │
│              ┌───────────┴───────────┐              │
│              ▼                       ▼              │
│         GPU 0,1,2,3              All GPUs           │
│         Update Parameters       Use Updated Params  │
└──────────────────────────────────────────────────┘
```

**Limitation**: Memory scales as O(N) per GPU where N is model size. For a 70B model in fp16, each GPU needs ~140GB just for parameters, which exceeds even the largest single GPUs.

### 10.2.3 DeepSpeed ZeRO (Zero Redundancy Optimizer)

DeepSpeed introduces ZeRO, which systematically eliminates memory redundancy across data-parallel GPUs:

```
┌──────────────────────────────────────────────────────────┐
│           ZeRO Memory Optimization Stages                  │
│                                                            │
│  Standard DP (No ZeRO):                                    │
│  ┌─────────────────────────────────────────┐              │
│  │ GPU 0: [Params] [Grads] [Optim States]  │ = 4x model   │
│  │ GPU 1: [Params] [Grads] [Optim States]  │   memory     │
│  └─────────────────────────────────────────┘              │
│                                                            │
│  ZeRO Stage 1 (os):                                        │
│  ┌─────────────────────────────────────────┐              │
│  │ GPU 0: [Params] [Grads] [Optim 1/4]     │ = 2.25x      │
│  │ GPU 1: [Params] [Grads] [Optim 1/4]     │   model      │
│  │ GPU 2: [Params] [Grads] [Optim 1/4]     │   memory     │
│  │ GPU 3: [Params] [Grads] [Optim 1/4]     │              │
│  └─────────────────────────────────────────┘              │
│                                                            │
│  ZeRO Stage 2 (os + g):                                     │
│  ┌─────────────────────────────────────────┐              │
│  │ GPU 0: [Params] [Grad 1/4] [Optim 1/4]  │ = 1.5x       │
│  │ GPU 1: [Params] [Grad 1/4] [Optim 1/4]  │   model      │
│  │ GPU 2: [Params] [Grad 1/4] [Optim 1/4]  │   memory     │
│  │ GPU 3: [Params] [Grad 1/4] [Optim 1/4]  │              │
│  └─────────────────────────────────────────┘              │
│                                                            │
│  ZeRO Stage 3 (os + g + p):                                 │
│  ┌─────────────────────────────────────────┐              │
│  │ GPU 0: [Param 1/4] [Grad 1/4] [Opt 1/4] │ = 1x         │
│  │ GPU 1: [Param 1/4] [Grad 1/4] [Opt 1/4] │   model      │
│  │ GPU 2: [Param 1/4] [Grad 1/4] [Opt 1/4] │   memory     │
│  │ GPU 3: [Param 1/4] [Grad 1/4] [Opt 1/4] │              │
│  └─────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

#### Memory Budget Calculation

For a model with Φ parameters in mixed precision (bf16/fp16 + fp32 optimizer states):

| Component | Bytes per Parameter | 70B Model Memory |
|-----------|-------------------|-------------------|
| Model Parameters (bf16) | 2 | 140 GB |
| Gradients (bf16) | 2 | 140 GB |
| Optimizer States (fp32 params + momentum + variance) | 12 | 840 GB |
| **Total (Standard DP)** | **16** | **1,120 GB** |
| **ZeRO-3** | 16 / num_gpus | 1,120 / N GB |

#### DeepSpeed ZeRO Configuration

```python
import deepspeed

# DeepSpeed ZeRO Stage 3 configuration for a 70B model
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

# Initialize model and engine
model = YourLLMModel(config)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters()
)

# Training loop
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### 10.2.4 FSDP (Fully Sharded Data Parallelism)

PyTorch's native FSDP provides similar functionality to ZeRO-3 with tighter PyTorch integration:

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Define which layers to wrap
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={TransformerBlock}
)

# Initialize FSDP model
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

# Standard PyTorch training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch)
    loss.backward()  # FSDP handles gradient communication
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
```

---

## 10.3 Model Parallelism & Pipeline Parallelism

### 10.3.1 Tensor Parallelism (TP)

When a single model layer is too large to fit on one GPU, tensor parallelism splits individual layers across multiple GPUs.

📌 **Key Concept**: Tensor parallelism splits the computation within a single layer across multiple GPUs, reducing memory per GPU and enabling larger batch sizes.

```
┌──────────────────────────────────────────────────────────┐
│              Tensor Parallelism for a Linear Layer         │
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
│  └──────────────────┘        └──────────────────┘        │
│                                                            │
│  All-Gather or Reduce-Scatter to combine results           │
└──────────────────────────────────────────────────────────┘
```

#### Megatron-LM Style Column-Row Parallelism

The most common TP scheme splits the weight matrix in a specific pattern:

```python
# Megatron-LM style column-parallel + row-parallel for MLP
class ParallelMLP(nn.Module):
    def __init__(self, hidden_size, ffn_size, tp_size):
        super().__init__()
        self.tp_size = tp_size
        self.tp_rank = dist.get_rank() % tp_size

        # Column parallel: split output dimension
        self.w1 = ColumnParallelLinear(hidden_size, ffn_size // tp_size)
        self.w3 = ColumnParallelLinear(hidden_size, ffn_size // tp_size)

        # Row parallel: split input dimension
        self.w2 = RowParallelLinear(ffn_size // tp_size, hidden_size)

    def forward(self, x):
        # x is already sharded across TP group
        h = F.silu(self.w1(x)) * self.w3(x)
        return self.w2(h)  # All-reduce happens inside RowParallelLinear
```

```
┌──────────────────────────────────────────────────────────┐
│         Megatron-LM Column-Row Parallel MLP              │
│                                                            │
│        Input X (replicated)                                │
│              │                                             │
│     ┌────────┴────────┐                                    │
│     │                 │                                    │
│     ▼                 ▼                                    │
│  ┌──────┐         ┌──────┐                               │
│  │  W1  │         │  W3  │   Column Parallel              │
│  │(col) │         │(col) │   Split output dim             │
│  └──┬───┘         └──┬───┘                                │
│     │                 │                                    │
│     ▼                 ▼                                    │
│  ┌──────┐         ┌──────┐                               │
│  │ SiLU │         │      │   Activation                   │
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
│         │  W2  │   Row Parallel                            │
│         │(row) │   Split input dim                         │
│         └──┬───┘   + All-Reduce                           │
│            │                                               │
│            ▼                                               │
│        Output Y (reduced)                                  │
└──────────────────────────────────────────────────────────┘
```

### 10.3.2 Pipeline Parallelism (PP)

Pipeline parallelism splits the model's layers across GPUs, creating a pipeline where different stages process different micro-batches.

```
┌──────────────────────────────────────────────────────────────────┐
│              Pipeline Parallelism (GPipe Style)                    │
│                                                                    │
│  Time ──────────────────────────────────────────────────────▶     │
│                                                                    │
│  Stage 0    │  ┌────┐  │        │        │        │             │
│  (GPU 0)    │  │MB 0│  │        │        │        │             │
│  Layers 0-19│  └────┘  │        │        │        │             │
│             │        │  ┌────┐  │        │        │             │
│             │        │  │MB 1│  │        │        │             │
│             │        │  └────┘  │        │        │             │
│                                                                    │
│  Stage 1    │        │        │  ┌────┐  │        │             │
│  (GPU 1)    │        │        │  │MB 0│  │        │             │
│  Layers 20-39       │        │  └────┘  │        │             │
│             │        │        │        │  ┌────┐  │             │
│             │        │        │        │  │MB 1│  │             │
│             │        │        │        │  └────┘  │             │
│                                                                    │
│  Stage 2    │        │        │        │        │  ┌────┐       │
│  (GPU 2)    │        │        │        │        │  │MB 0│       │
│  Layers 40-59       │        │        │        │  └────┘       │
│             │        │        │        │        │        │       │
│                                                                    │
│  Stage 3    │        │        │        │        │        │       │
│  (GPU 3)    │        │        │        │        │        │       │
│  Layers 60-79       │        │        │        │        │       │
│                                                                    │
│  MB = Micro-Batch                                                │
│  Bubble = idle time between stages (pipeline bubble)              │
└──────────────────────────────────────────────────────────────────┘
```

#### Pipeline Bubble Analysis

The pipeline bubble is a critical efficiency concern:

```
Pipeline Efficiency = (T - bubble) / T
Where:
  T = total time for P stages and M micro-batches
  bubble = (P - 1) * time_per_micro_batch

For GPipe:
  Efficiency ≈ (M - 1) / (M + P - 1)
  For P=4, M=32: Efficiency ≈ 31/35 ≈ 88.6%
  For P=4, M=8:  Efficiency ≈ 7/11 ≈ 63.6%
```

#### 1F1B Scheduling

1F1B (One Forward One Backward) reduces pipeline bubbles by interleaving forward and backward passes:

```
┌──────────────────────────────────────────────────────────────┐
│                    1F1B Schedule                               │
│                                                                │
│  GPU 0: F0  F1  F2  F3  B0  F4  B1  F5  B2  F6  B3  F7    │
│  GPU 1:     F0  F1  F2  F3  B0  F4  B1  F5  B2  F6  B3     │
│  GPU 2:         F0  F1  F2  F3  B0  F4  B1  F5  B2  F6     │
│  GPU 3:             F0  F1  F2  F3  B0  F4  B1  F5  B2     │
│                                                                │
│  F = Forward pass    B = Backward pass                        │
│  Numbers = Micro-batch index                                  │
│                                                                │
│  After warmup phase, each GPU alternates F and B:              │
│  Bubble reduced to: (P-1) / (P + M*(P-1)) * time              │
└──────────────────────────────────────────────────────────────┘
```

### 10.3.3 3D Parallelism

Production training combines all three parallelism strategies:

```
┌──────────────────────────────────────────────────────────────┐
│                    3D Parallelism                              │
│                                                                │
│  Data Parallel (DP): Across nodes                              │
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
│  Within each Node:                                             │
│  ┌──────────────────────────────────────────────┐            │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │            │
│  │  │GPU 0 │ │GPU 1 │ │GPU 2 │ │GPU 3 │       │            │
│  │  │TP=0  │ │TP=1  │ │TP=2  │ │TP=3  │       │            │
│  │  │PP=0  │ │PP=0  │ │PP=1  │ │PP=1  │       │            │
│  │  └──────┘ └──────┘ └──────┘ └──────┘       │            │
│  │                                              │            │
│  │  TP within stage, PP across stages           │            │
│  └──────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

#### Communication Patterns

```python
# Different communication primitives for different parallelism strategies

# Tensor Parallelism: All-Reduce (high bandwidth, low latency)
# - Synchronous communication within a node (NVLink)
# - Frequent communication (every layer)

# Pipeline Parallelism: Point-to-point send/recv
# - Between adjacent stages
# - Less frequent (between stages)

# Data Parallelism: All-Reduce or Reduce-Scatter + All-Gather
# - Cross-node communication (InfiniBand)
# - Once per training step

# Communication volume analysis:
# TP: 2 * layer_size * 2 bytes (bf16) per layer  [per GPU]
# PP: layer_size * 2 bytes per stage boundary     [per step]
# DP: 2 * model_size * 2 bytes per step           [per GPU]

# For a 70B model on 64 GPUs (8 nodes × 8 GPUs):
# TP=8 (within node), PP=8 (across nodes), DP=1
# TP comm per layer: 2 * 8.75B * 2 bytes = 35 GB  [within NVLink]
# PP comm per step:  8.75B * 2 bytes = 17.5 GB     [across InfiniBand]
```

### 10.3.4 Expert Parallelism (MoE)

Mixture of Experts (MoE) adds a different dimension of parallelism:

```
┌──────────────────────────────────────────────────────────┐
│              MoE Layer Architecture                        │
│                                                            │
│  Input Token: "The cat sat on the mat"                     │
│       │                                                    │
│       ▼                                                    │
│  ┌─────────────────────────┐                              │
│  │   Router/Gate Network    │                              │
│  │   (Learned)              │                              │
│  └───────────┬─────────────┘                              │
│              │                                             │
│     ┌────────┼────────┬────────┬────────┐                │
│     ▼        ▼        ▼        ▼        ▼                │
│  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐              │
│  │Exp 0 ││Exp 1 ││Exp 2 ││Exp 3 ││Exp 4 │              │
│  │(GPU0)││(GPU1)││(GPU2)││(GPU3)││(GPU4)│              │
│  └──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘              │
│     │       │       │       │       │                    │
│     │       ▼       │       │       │                    │
│     │  ┌────────┐   │       │       │                    │
│     │  │Top-K   │   │       │       │                    │
│     │  │Selected│   │       │       │                    │
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
│           Combined Output                                 │
└──────────────────────────────────────────────────────────┘

MoE Scaling: With E experts and top-K routing:
  - Each token activates K out of E experts
  - Total parameters: E × (FFN params)
  - Active parameters per token: K × (FFN params)
  - Example: Mixtral 8x7B has 47B total, ~13B active per token
```

---

## 10.4 Memory Optimization Techniques

### 10.4.1 Activation Checkpointing (Gradient Checkpointing)

The fundamental trade-off: compute vs. memory.

```
┌──────────────────────────────────────────────────────────┐
│          Activation Checkpointing Trade-off                │
│                                                            │
│  Without Checkpointing:                                    │
│  ┌────────────────────────────────────┐                  │
│  │ All activations stored: O(L × B × S × H)            │
│  │ Memory: HIGH                        │                  │
│  │ Compute: STANDARD                   │                  │
│  └────────────────────────────────────┘                  │
│                                                            │
│  With Full Checkpointing:                                  │
│  ┌────────────────────────────────────┐                  │
│  │ Only boundaries stored: O(√L × B × S × H)            │
│  │ Memory: LOW (by factor of √L)      │                  │
│  │ Compute: +33% (recompute activations) │                  │
│  └────────────────────────────────────┘                  │
│                                                            │
│  With Selective Checkpointing:                             │
│  ┌────────────────────────────────────┐                  │
│  │ Checkpoint attention, not FFN       │                  │
│  │ Memory: MEDIUM                      │                  │
│  │ Compute: +10-20%                    │                  │
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
        # Use checkpoint for the entire block
        return checkpoint(self._forward_impl, x, use_reentrant=False)

    def _forward_impl(self, x):
        h = x + self.attention(self.norm1(x))
        return h + self.ffn(self.norm2(h))

# Selective checkpointing - only checkpoint attention
def selective_checkpointing_forward(block, x):
    h = x + checkpoint(block.attention, block.norm1(x),
                        use_reentrant=False)
    # FFN computed without checkpointing (faster, more memory)
    return h + block.ffn(block.norm2(h))
```

### 10.4.2 Flash Attention

Flash Attention (Dao et al., 2022) is a memory-efficient attention algorithm that avoids materializing the full attention matrix:

```
┌──────────────────────────────────────────────────────────┐
│          Standard vs Flash Attention                       │
│                                                            │
│  Standard Attention:                                        │
│  ┌────────────────────────────────────┐                  │
│  │ Q, K, V (N × d)                    │                  │
│  │     │                              │                  │
│  │     ▼                              │                  │
│  │ S = Q @ K^T (N × N)   ← FULL N×N matrix stored      │
│  │     │                              │                  │
│  │     ▼                              │                  │
│  │ P = softmax(S) (N × N)            │                  │
│  │     │                              │                  │
│  │     ▼                              │                  │
│  │ O = P @ V (N × d)                  │                  │
│  │                                    │                  │
│  │ Memory: O(N²)  ← PROBLEM for long  │                  │
│  │           sequences                 │                  │
│  └────────────────────────────────────┘                  │
│                                                            │
│  Flash Attention:                                           │
│  ┌────────────────────────────────────┐                  │
│  │ Q, K, V partitioned into blocks    │                  │
│  │ Process blocks in SRAM (fast!)     │                  │
│  │ Online softmax (no full N×N)       │                  │
│  │                                    │                  │
│  │ for each block of Q:              │                  │
│  │   for each block of K,V:          │                  │
│  │     load into SRAM                 │                  │
│  │     compute partial attention      │                  │
│  │     update output with running max │                  │
│  │                                    │                  │
│  │ Memory: O(N)   ← Linear!          │                  │
│  │ Speed: 2-4x FASTER (better memory  │                  │
│  │         access patterns)           │                  │
│  └────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────┘
```

```python
# Using Flash Attention via PyTorch 2.0+ or flash-attn
import torch
import torch.nn.functional as F

# Method 1: PyTorch 2.0 SDPA (Scaled Dot Product Attention)
# Automatically uses Flash Attention when possible
def attention_forward(q, k, v, mask=None):
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        is_causal=True,
        # PyTorch auto-selects: Flash, Memory-Efficient, or Math
    )

# Method 2: flash-attn library (more control)
from flash_attn import flash_attn_func

def flash_attention_forward(q, k, v):
    # q, k, v shape: (batch, seqlen, nheads, headdim)
    return flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        causal=True,
        window_size=(-1, -1),  # Full attention
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False
    )
```

### 10.4.3 CPU Offloading

For models too large to fit even with ZeRO-3, optimizer states and parameters can be offloaded to CPU:

```
┌──────────────────────────────────────────────────────────┐
│              CPU Offloading Strategy                        │
│                                                            │
│  GPU Memory (80GB H100)                                    │
│  ┌────────────────────────────────────────┐              │
│  │  ┌──────────┐  ┌──────────┐           │              │
│  │  │Activations│  │ Forward  │           │              │
│  │  │(dynamic)  │  │  Pass    │           │              │
│  │  └──────────┘  └──────────┘           │              │
│  │  ┌──────────────────────────┐         │              │
│  │  │ Current Layer Parameters │         │              │
│  │  │ (pinned memory)          │         │              │
│  │  └──────────────────────────┘         │              │
│  └────────────────────────────────────────┘              │
│                                                            │
│  CPU Memory (512GB+)                                       │
│  ┌────────────────────────────────────────┐              │
│  │  ┌──────────────────────────────────┐  │              │
│  │  │ Optimizer States (fp32)           │  │              │
│  │  │ └── AdamW: params + m + v         │  │              │
│  │  │ Memory: 12 bytes per param        │  │              │
│  │  └──────────────────────────────────┘  │              │
│  │  ┌──────────────────────────────────┐  │              │
│  │  │ Offloaded Parameters (bf16)       │  │              │
│  │  │ Memory: 2 bytes per param         │  │              │
│  │  └──────────────────────────────────┘  │              │
│  │  ┌──────────────────────────────────┐  │              │
│  │  │ Gradient Accumulation Buffer      │  │              │
│  │  └──────────────────────────────────┘  │              │
│  └────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

### 10.4.4 Memory Budget Summary

Complete memory analysis for training a 70B parameter model:

```
┌────────────────────────────────────────────────────────────┐
│        70B Model Memory Budget (Training, bf16)              │
│                                                              │
│  Component               │ Per GPU (DP=1) │ Per GPU (ZeRO-3, 64 GPUs)
│  ─────────────────────── │ ────────────── │ ───────────────────
│  Model Parameters (bf16) │    140 GB      │     2.19 GB
│  Gradients (bf16)        │    140 GB      │     2.19 GB
│  Optimizer (fp32)        │    840 GB      │    13.13 GB
│  Activations (8K ctx)    │    ~200 GB     │    ~200 GB (not sharded)
│  ─────────────────────── │ ────────────── │ ───────────────────
│  TOTAL                   │  ~1,320 GB     │   ~218 GB
│                                                              │
│  With CPU Offload (ZeRO-3):                                  │
│  ─────────────────────── │ ────────────── │ ───────────────────
│  GPU Memory              │     N/A        │    ~218 GB (need 3× H100)
│  CPU Memory              │     N/A        │    ~993 GB
│                                                              │
│  With Activation Checkpointing:                              │
│  ─────────────────────── │ ────────────── │ ───────────────────
│  Activations (reduced)   │    ~20 GB      │    ~20 GB
│  TOTAL GPU               │     N/A        │    ~38 GB (ZeRO-3 + CKPT)
└────────────────────────────────────────────────────────────┘
```

---

## 10.5 Mixed Precision Training

### 10.5.1 Numerical Formats Comparison

📌 **Key Concept**: Mixed precision training uses lower-precision formats (bf16/fp16) for most computations while maintaining a master copy in fp32 for critical operations.

```
┌────────────────────────────────────────────────────────────┐
│           Numerical Format Comparison                        │
│                                                              │
│  Format     │ Bits │ Sign │ Exponent │ Mantissa │ Range    │
│  ───────────│──────│──────│──────────│──────────│──────────│
│  fp32       │  32  │  1   │    8     │    23    │ ±3.4e38  │
│  fp16       │  16  │  1   │    5     │    10    │ ±65504   │
│  bf16       │  16  │  1   │    8     │     7    │ ±3.4e38  │
│  fp8 (E4M3) │   8  │  1   │    4     │     3    │ ±448     │
│  fp8 (E5M2) │   8  │  1   │    5     │     2    │ ±57344   │
│  int8       │   8  │  1   │    -     │     -    │ -128~127 │
│  int4       │   4  │  1   │    -     │     -    │ -8~7     │
│  nf4        │   4  │  -   │    -     │     -    │ Normalized│
│                                                              │
│  Key Insight:                                                │
│  - fp16: Small range, needs loss scaling                    │
│  - bf16: Same range as fp32, less precise                   │
│  - fp8:  Two variants for different needs                   │
│  - E4M3: Higher precision (inference)                       │
│  - E5M2: Higher range (training gradients)                  │
└────────────────────────────────────────────────────────────┘
```

### 10.5.2 BF16 vs FP16 for Training

```
┌────────────────────────────────────────────────────────────┐
│              BF16 vs FP16 Training Comparison                │
│                                                              │
│  BF16 (Brain Float 16):                                     │
│  ✅ Same exponent range as fp32 (±3.4e38)                  │
│  ✅ No loss scaling required                                │
│  ✅ More stable training                                   │
│  ❌ Lower precision (7 mantissa bits vs 10)                │
│  ❌ Not supported on older GPUs (pre-A100)                 │
│                                                              │
│  FP16 (Half Precision):                                      │
│  ✅ More precise (10 mantissa bits)                        │
│  ✅ Broader GPU support                                     │
│  ❌ Small range (±65504), needs loss scaling               │
│  ❌ Prone to overflow/underflow                             │
│  ❌ Requires careful scaling factor tuning                  │
│                                                              │
│  Recommendation (2024-2026):                                 │
│  ┌────────────────────────────────────────────┐            │
│  │  GPU >= A100/H100 → Use BF16               │            │
│  │  GPU < A100        → Use FP16 + GradScaler │            │
│  │  H100/H200         → Consider FP8 for      │            │
│  │                      additional speedup     │            │
│  └────────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────────┘
```

### 10.5.3 FP8 Training (H100/H200)

FP8 training on H100 GPUs offers 2x throughput improvement for matrix operations:

```python
import torch
import transformer_engine.pytorch as te

class FP8TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Transformer Engine handles FP8 automatically
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

# FP8 training recipe for H100
fp8_recipe = te.common.recipe.DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=te.common.recipe.Format.HYBRID,
    amax_history_length=16,
    amax_compute_algo="max",
)
```

### 10.5.4 Gradient Accumulation

When the desired effective batch size exceeds GPU memory capacity:

```python
import torch
import deepspeed

def train_with_gradient_accumulation(
    model_engine,
    dataloader,
    gradient_accumulation_steps=16,
):
    """
    Gradient accumulation: simulate large batch by accumulating
    gradients over multiple micro-batches before updating.
    """
    model_engine.train()
    optimizer = model_engine.optimizer

    for step, batch in enumerate(dataloader):
        # Forward + backward (gradients accumulate)
        loss = model_engine(batch) / gradient_accumulation_steps
        model_engine.backward(loss)

        # Only update every N steps
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model_engine.module.parameters(), max_norm=1.0
            )
            # Optimizer step (all-reduce happens here)
            model_engine.step()
            optimizer.zero_grad()

            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}: loss={loss.item():.4f}, lr={current_lr:.2e}")

# Effective batch size = micro_batch_size * num_gpus * gradient_accumulation_steps
# Example: 2 * 64 * 16 = 2048 samples per step
```

---

## 📝 Exercise: Building a Small-Scale Distributed Training Environment

### Objective

Set up a distributed training environment on 2 GPUs using DeepSpeed ZeRO Stage 2, and train a small language model (~125M parameters) on a text dataset.

### Prerequisites

- 2× GPUs with at least 16GB VRAM each
- Python 3.10+, PyTorch 2.0+, DeepSpeed 0.12+

### Step 1: Environment Setup

```bash
# Create virtual environment
conda create -n llm-train python=3.11
conda activate llm-train

# Install dependencies
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed==0.12.0 transformers datasets accelerate
pip install flash-attn --no-build-isolation

# Verify setup
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
```

### Step 2: Model Configuration

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

# ~125M parameters
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

### Step 3: DeepSpeed Configuration

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

### Step 4: Training Script

```python
# train.py
import torch
import deepspeed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def main():
    # Load tokenizer and create model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        hidden_size=768, intermediate_size=2048,
        num_hidden_layers=12, num_attention_heads=12,
        num_key_value_heads=4, max_position_embeddings=2048,
        vocab_size=32000,
    )
    model = AutoModelForCausalLM.from_config(config)

    # Load dataset
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

    # Initialize DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config="ds_config.json",
        model_parameters=model.parameters()
    )

    # Training loop
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

    # Save checkpoint
    model_engine.save_checkpoint("./checkpoints", tag="step_1000")

if __name__ == "__main__":
    main()
```

### Step 5: Launch Distributed Training

```bash
# Launch with DeepSpeed on 2 GPUs
deepspeed --num_gpus=2 train.py \
    --deepspeed ds_config.json \
    --output_dir ./output \
    --logging_steps 10 \
    --save_steps 500

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training metrics
tensorboard --logdir ./output/runs
```

### Expected Output

```
[2026-01-15 10:00:00] Step 0   | Loss: 10.8234 | LR: 1.00e-05
[2026-01-15 10:00:15] Step 100 | Loss: 5.2341  | LR: 3.00e-04
[2026-01-15 10:00:30] Step 200 | Loss: 4.8723  | LR: 2.98e-04
[2026-01-15 10:00:45] Step 300 | Loss: 4.6512  | LR: 2.96e-04
...
[2026-01-15 10:02:30] Step 1000 | Loss: 4.1234 | LR: 2.80e-04
```

### Questions to Explore

1. **What happens if you increase gradient accumulation steps from 8 to 32?** How does this affect convergence speed and final loss?

2. **Compare ZeRO Stage 1 vs Stage 2 vs Stage 3.** Measure GPU memory usage at each stage using `nvidia-smi`.

3. **Try enabling activation checkpointing.** What is the trade-off between memory savings and training speed?

4. **Experiment with learning rate schedules.** How does warmup duration affect early training stability?

---

## Summary

This chapter established the foundational architecture knowledge for building and training large language models:

| Topic | Key Takeaway |
|-------|-------------|
| **Transformer Review** | Modern LLMs use RoPE, GQA, SwiGLU, RMSNorm as standard components |
| **Scaling Laws** | Optimal training requires balancing model size and data size (Chinchilla) |
| **ZeRO Stages** | Stage 3 enables training 70B+ models with manageable per-GPU memory |
| **3D Parallelism** | TP + PP + DP is the standard for training 100B+ models |
| **Flash Attention** | O(N) memory and 2-4x speedup over standard attention |
| **Mixed Precision** | BF16 is the default for modern GPUs; FP8 on H100+ for 2x speedup |
| **Activation Checkpointing** | 33% compute overhead for massive memory savings |

### Chapter Roadmap

```
Chapter 10 (Current): LLM Architecture Design Fundamentals
    │
    ├── Chapter 11: LLM Inference Architecture
    │   └── How to serve trained models efficiently
    │
    ├── Chapter 12: RAG System Architecture
    │   └── How to augment LLMs with external knowledge
    │
    └── Chapter 13: Model Fine-tuning Architecture
        └── How to adapt pre-trained models to specific tasks
```

---

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Rajbhandari, S., et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC'20.
3. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS.
4. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." NeurIPS.
5. Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv.
6. Korthikanti, V., et al. (2022). "Reducing Activation Recomputation in Large Transformer Models." MLSys.
7. Narayanan, D., et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." SC'21.
8. NVIDIA Transformer Engine Documentation. https://docs.nvidia.com/deeplearning/transformer-engine/
9. DeepSpeed Documentation. https://www.deepspeed.ai/
10. PyTorch FSDP Tutorial. https://pytorch.org/docs/stable/fsdp.html

---

*Next Chapter: [Chapter 11 - LLM Inference Architecture](chapter-11.md) →*
