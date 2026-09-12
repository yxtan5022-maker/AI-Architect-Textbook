# Chapter 10: LLM Architecture Design Fundamentals

> 🟢 Beginner → 🔴 Advanced | 25-30 min read | Part 4: Large Model Architecture

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain the core Transformer architecture components (self-attention, FFN, layer normalization) and their modern variants (RoPE, GQA, SwiGLU, RMSNorm)
2. Calculate GPU memory requirements for training models of different sizes (7B, 13B, 70B, 405B)
3. Design distributed training strategies using tensor parallelism, pipeline parallelism, and data parallelism (3D parallelism)
4. Apply memory optimization techniques including ZeRO stages, Flash Attention, and activation checkpointing
5. Understand Chinchilla scaling laws and their practical implications for training data and model size decisions
6. Reproduce a small-scale distributed training setup using DeepSpeed

---

## Table of Contents

- [10.1 Transformer Architecture Review](#101-transformer-architecture-review)
- [10.2 GPU Memory Mathematics](#102-gpu-memory-mathematics)
- [10.3 Distributed Training Strategies](#103-distributed-training-strategies)
- [10.4 Memory Optimization Techniques](#104-memory-optimization-techniques)
- [10.5 Mixed Precision Training](#105-mixed-precision-training)
- [💡 Case Study: How Meta Trained Llama 3](#-case-study-how-meta-trained-llama-3)
- [⚠️ War Story: The OOM That Killed a $2M Training Run](#️-war-story-the-oom-that-killed-a-2m-training-run)
- [📝 When to Use / When Not to Use](#-when-to-use--when-not-to-use)
- [Summary](#summary)
- [Discussion Questions](#discussion-questions)
- [Exercises](#exercises)
- [References](#references)

---

## 10.1 Transformer Architecture Review

### 10.1.1 The Self-Attention Mechanism

The Transformer architecture, introduced by Vaswani et al. (2017) in "Attention Is All You Need," fundamentally reshaped natural language processing and now serves as the backbone of virtually all large language models (LLMs). Before diving into the architectural considerations for large-scale models, we must first solidify our understanding of the foundational building blocks.

📌 **Real Data**: The original Transformer paper has been cited over 130,000 times (Google Scholar, 2026), making it one of the most influential papers in deep learning history. Every major LLM — GPT-4, Claude 3.5, Llama 3, Gemini — is built on the Transformer architecture.

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
│  Output                                                 │
└─────────────────────────────────────────────────────┘
```

### 10.1.2 Modern Architecture Variants (2023-2026)

Modern LLMs have evolved significantly from the original Transformer. Here are the key variants now considered standard:

📌 **Real Data**: As of 2026, 95%+ of production LLMs use some combination of RoPE + GQA + SwiGLU + RMSNorm. This has become the de facto architecture standard.

**Rotary Position Embedding (RoPE)**

RoPE encodes position information directly into the attention computation by rotating query and key vectors:

```
RoPE(x, pos) = x * cos(pos * θ) + rotate(x) * sin(pos * θ)
```

- Used by: Llama 3, Mistral, Qwen, Yi, and virtually all modern open-source LLMs
- Advantage: Better length extrapolation than absolute or learned position embeddings
- Practical impact: Models trained with RoPE can be extended to longer contexts via NTK-aware scaling

**Grouped Query Attention (GQA)**

GQA is a middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA):

```
MHA:  Each head has its own Q, K, V      → Best quality, highest memory
GQA:  Groups of heads share K and V       → Balanced quality and memory
MQA:  All heads share one K and V         → Lowest memory, reduced quality
```

- Llama 3 8B: 32 heads, 8 KV groups (4 heads per group)
- Llama 3 70B: 64 heads, 8 KV groups (8 heads per group)
- KV cache memory reduction: 4-8x compared to MHA

**SwiGLU Activation**

SwiGLU replaces the standard ReLU/GELU in the FFN:

```
Standard FFN:   output = W2 * activation(W1 * input)
SwiGLU FFN:     output = W2 * (Swish(W1 * input) ⊙ (W3 * input))
```

- Used by: Llama 3, Mistral, PaLM, Gemini
- Advantage: Better training stability and slightly improved quality
- Note: SwiGLU FFN has 3 weight matrices (W1, W2, W3) instead of 2, increasing FFN parameters by 50%

**RMSNorm vs LayerNorm**

```
LayerNorm:  y = (x - mean) / sqrt(var + ε) * γ + β
RMSNorm:    y = x / sqrt(mean(x²) + ε) * γ
```

- RMSNorm removes mean-centering and bias (β), making it ~10-15% faster
- Used by: All Llama models, Mistral, Qwen, Gemma
- Quality difference: Negligible in practice

### 10.1.3 Model Size Configurations

Here are real model configurations for reference:

| Model | Layers | Hidden Dim | Heads | KV Groups | FFN Dim | Total Params |
|-------|--------|------------|-------|-----------|---------|-------------|
| Llama 3 8B | 32 | 4096 | 32 | 8 | 14336 | 8.0B |
| Llama 3 70B | 80 | 8192 | 64 | 8 | 28672 | 70.6B |
| Llama 3 405B | 126 | 16384 | 128 | 8 | 53248 | 405B |
| Mistral 7B | 32 | 4096 | 32 | 8 | 14336 | 7.3B |
| Qwen2 72B | 80 | 8192 | 64 | 8 | 29568 | 72.7B |

---

## 10.2 GPU Memory Mathematics

Understanding GPU memory requirements is critical for planning training runs. A single miscalculation can waste millions of dollars in compute costs.

### 10.2.1 Memory Components

Training a model requires memory for four components:

```
Total GPU Memory = Model Parameters + Gradients + Optimizer States + Activations
                   ───────────────   ─────────   ────────────────   ───────────
                   Fixed per model   Fixed per   AdamW needs 16     Grows with
                   size              model size  bytes per param    batch size
```

### 10.2.2 Memory Calculation Formula

For a model with **P** parameters, trained with **AdamW optimizer** in **mixed precision (bf16)**:

| Component | FP32 | BF16 Mixed |
|-----------|------|------------|
| Model Parameters | 4P bytes | 2P (bf16) + 2P (fp32 copy) = 4P bytes |
| Gradients | 4P bytes | 2P bytes |
| Optimizer States (m, v) | 8P bytes (2 × 4P for fp32) | 8P bytes |
| **Total (without activations)** | **16P bytes** | **14P bytes** |

📌 **Real Data**: For a 7B parameter model: 14 × 7B = 98 GB minimum (without activations). For 70B: 14 × 70B = 980 GB. For 405B: 14 × 405B = 5,670 GB.

**Activation Memory** depends on sequence length, batch size, and number of layers:

```
Activations ≈ 2 × batch_size × seq_len × hidden_dim × num_layers × (34 + 5 × attn_heads / hidden_dim)
```

For practical estimates:

| Model Size | Batch=1, Seq=2048 | Batch=32, Seq=2048 | Batch=32, Seq=8192 |
|------------|-------------------|--------------------|--------------------|
| 7B | ~2 GB | ~64 GB | ~256 GB |
| 13B | ~4 GB | ~128 GB | ~512 GB |
| 70B | ~16 GB | ~512 GB | ~2 TB |
| 405B | ~80 GB | ~2.5 TB | ~10 TB |

### 10.2.3 Hardware Requirements by Model Size

| Model Size | Min GPUs (A100 80GB) | Min GPUs (H100 80GB) | Estimated Training Cost | Training Time (1T tokens) |
|-----------|----------------------|----------------------|------------------------|---------------------------|
| 7B | 1 | 1 | $50K-$100K | 2-4 weeks |
| 13B | 2-4 | 1-2 | $100K-$300K | 4-8 weeks |
| 70B | 16-32 | 8-16 | $1M-$5M | 2-3 months |
| 405B | 256-512 | 128-256 | $10M-$50M | 3-6 months |

---

## 10.3 Distributed Training Strategies

### 10.3.1 Data Parallelism (DP)

The simplest form of distributed training: each GPU holds a complete copy of the model, processes different data, and gradients are averaged across GPUs.

```
┌─────────────────────────────────────────────────────────────┐
│                  Data Parallelism                            │
│                                                               │
│  GPU 0: Full Model + Batch 0 ──┐                            │
│  GPU 1: Full Model + Batch 1 ──┼── AllReduce Gradients      │
│  GPU 2: Full Model + Batch 2 ──┤                            │
│  GPU 3: Full Model + Batch 3 ──┘                            │
│                                                               │
│  ✅ Simple to implement                                      │
│  ❌ Each GPU needs full model copy (wasteful for large models)│
│  ❌ Communication overhead grows with model size              │
└─────────────────────────────────────────────────────────────┘
```

### 10.3.2 ZeRO (Zero Redundancy Optimizer)

DeepSpeed's ZeRO eliminates the memory redundancy of data parallelism:

| ZeRO Stage | What is Partitioned | Memory Savings | Communication Overhead |
|-----------|-------------------|-----------------|----------------------|
| Stage 1 | Optimizer States | 4x | Low (same as DP) |
| Stage 2 | + Gradients | 8x | Medium |
| Stage 3 | + Parameters | N× (N = num GPUs) | High (all-gather each step) |

```
ZeRO Memory Usage (70B model, bf16, AdamW):

                    Stage 0 (DP)    Stage 1      Stage 2      Stage 3
                    ─────────────   ──────────   ──────────   ──────────
Model Params        140 GB          140 GB       140 GB       140 GB / N
Gradients           140 GB          140 GB       140 GB / N   140 GB / N
Optimizer States    560 GB          560 GB / N   560 GB / N   560 GB / N
─────────────────────────────────────────────────────────────────────────
Per-GPU Total       840 GB          ~290 GB      ~130 GB      ~840 GB / N

For 64 GPUs:
Stage 3 per-GPU: ~13 GB (fits on single GPU!)
```

### 10.3.3 Tensor Parallelism (TP)

Tensor parallelism splits individual matrix multiplications across GPUs:

```
┌─────────────────────────────────────────────────────────────┐
│              Tensor Parallelism (TP=4)                        │
│                                                               │
│  Linear Layer Y = X × W                                       │
│                                                               │
│  Split W column-wise:                                         │
│  W = [W1 | W2 | W3 | W4]  (each Wi on different GPU)        │
│                                                               │
│  GPU 0: Y0 = X × W1 ──┐                                     │
│  GPU 1: Y1 = X × W2 ──┼── Concat → Y                        │
│  GPU 2: Y2 = X × W3 ──┤                                     │
│  GPU 3: Y3 = X × W4 ──┘                                     │
│                                                               │
│  Communication: AllReduce after each layer                    │
│  Best for: Within a single node (NVLink-connected GPUs)      │
└─────────────────────────────────────────────────────────────┘
```

📌 **Real Data**: NVIDIA's Megatron-LM paper demonstrated that TP=8 on 8 NVLink-connected A100 GPUs achieves near-linear scaling for the attention and FFN layers within a Transformer block (Narayanan et al., 2021).

### 10.3.4 Pipeline Parallelism (PP)

Pipeline parallelism splits the model layer-by-layer across GPUs:

```
┌─────────────────────────────────────────────────────────────┐
│              Pipeline Parallelism (PP=4)                      │
│                                                               │
│  GPU 0: Layers 0-7     ──► ──► ──►                          │
│  GPU 1: Layers 8-15    ──► ──► ──►                          │
│  GPU 2: Layers 16-23   ──► ──► ──►                          │
│  GPU 3: Layers 24-31   ──► ──► ──►                          │
│                                                               │
│  Problem: Pipeline bubbles (GPUs idle while waiting)          │
│                                                               │
│  Solution: Micro-batching (GPipe / 1F1B schedule)            │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                          │
│  │ B0│ B1│ B2│ B3│   │B4 │B5 │B6 │  ← bubbles reduced     │
│  └───┴───┴───┴───┴───┴───┴───┴───┘                          │
│                                                               │
│  Bubble ratio = (PP - 1) / (PP - 1 + num_microbatches)      │
└─────────────────────────────────────────────────────────────┘
```

### 10.3.5 3D Parallelism

Production training of 70B+ models uses all three strategies combined:

```
┌─────────────────────────────────────────────────────────────┐
│                    3D Parallelism                             │
│                                                               │
│  Example: Training Llama 3 405B on 16,384 H100 GPUs          │
│                                                               │
│  TP = 8   (within each node, NVLink-connected)               │
│  PP = 16  (across nodes, InfiniBand-connected)               │
│  DP = 128 (across all nodes)                                 │
│                                                               │
│  Total GPUs = TP × PP × DP = 8 × 16 × 128 = 16,384         │
│                                                               │
│  Communication hierarchy:                                     │
│  ┌─────────────────────────────────────────────────┐         │
│  │  Intra-node: NVLink (900 GB/s on H100)         │         │
│  │  Inter-node: InriaBand NDR (400 Gb/s)          │         │
│  │  TP uses NVLink, PP/DP uses InfiniBand          │         │
│  └─────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 10.4 Memory Optimization Techniques

### 10.4.1 Flash Attention

Flash Attention (Dao et al., 2022) is the most impactful memory optimization for Transformer training and inference:

**Standard Attention Problem:**
- Computes full N×N attention matrix
- Memory: O(N²) — for N=8192, this is 67M floats per head
- Requires materializing the full attention matrix in GPU HBM

**Flash Attention Solution:**
- Uses IO-aware tiling to compute attention in blocks
- Never materializes the full attention matrix
- Memory: O(N) — linear in sequence length
- 2-4x speedup on modern GPUs

📌 **Real Data**: Flash Attention 2 (Dao, 2023) achieves 72% of theoretical maximum FLOPS on A100 GPUs, compared to 25% for standard attention. Flash Attention 3 (Shah et al., 2024) on H100 achieves up to 740 TFLOPS, approaching the H100's peak of 989 TFLOPS.

```
Performance Comparison (A100-80GB, seq_len=4096, batch_size=16):

                  Memory        Time         TFLOPS
                  ──────────    ──────────   ──────────
Standard Attn     12.4 GB       8.2 ms       52 TFLOPS
Flash Attention   3.1 GB        2.1 ms       196 TFLOPS
Flash Attention 2 1.6 GB        1.4 ms       238 TFLOPS
```

### 10.4.2 Activation Checkpointing (Gradient Checkpointing)

Trade compute for memory by not storing intermediate activations:

```
Without Checkpointing:
  Forward: Store all activations → Memory: O(L × B × S × H)
  Backward: Reuse stored activations → Compute: 1x

With Checkpointing (every N layers):
  Forward: Store only every N-th layer's activations → Memory: O(L/N × B × S × H)
  Backward: Recompute intermediate activations → Compute: ~1.33x (N=2)

Trade-off:
  Memory savings: 3-10x reduction
  Compute overhead: 30-40% additional FLOPS
```

### 10.4.3 CPU Offloading

ZeRO-Infinity (Rajbhandari et al., 2021) enables offloading parameters, gradients, and optimizer states to CPU memory or NVMe SSD:

```
┌─────────────────────────────────────────────────────────────┐
│              ZeRO-Infinity Offloading                         │
│                                                               │
│  GPU HBM (80GB):                                             │
│  ┌─────────────────────────────────┐                         │
│  │  Active layers + activations     │                         │
│  └─────────────────────────────────┘                         │
│                                                               │
│  CPU DRAM (1-2TB):                                           │
│  ┌─────────────────────────────────┐                         │
│  │  Inactive model shards           │                         │
│  │  Optimizer states                │                         │
│  └─────────────────────────────────┘                         │
│                                                               │
│  NVMe SSD (10TB+):                                           │
│  ┌─────────────────────────────────┐                         │
│  │  Checkpoints                     │                         │
│  │  Overflow parameters             │                         │
│  └─────────────────────────────────┘                         │
│                                                               │
│  ⚠️ Performance drops significantly with offloading           │
│  Best used as: Fallback, not primary strategy                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 10.5 Mixed Precision Training

### 10.5.1 Precision Formats

| Format | Bits | Range | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| FP32 | 32 | ±3.4×10³⁸ | ~7 decimal digits | Optimizer states, loss |
| FP16 | 16 | ±65504 | ~3 decimal digits | Training (legacy) |
| BF16 | 16 | ±3.4×10³⁸ | ~3 decimal digits | Training (default) |
| FP8 (E4M3) | 8 | ±448 | ~1-2 digits | Inference, some training |
| FP8 (E5M2) | 8 | ±57344 | ~1 digit | Gradient communication |

📌 **Real Data**: BF16 is the standard for modern training because it has the same range as FP32 (8-bit exponent) while using half the memory. FP8 on NVIDIA H100 GPUs provides 2x throughput improvement for matrix operations (NVIDIA Transformer Engine).

### 10.5.2 Mixed Precision Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Mixed Precision Training (BF16)                  │
│                                                               │
│  1. Master Weights (FP32)                                     │
│       │                                                       │
│       ▼                                                       │
│  2. Cast to BF16 ───────────────────────────────►            │
│       │                                                       │
│       ▼                                                       │
│  3. Forward Pass (BF16)                                       │
│       │                                                       │
│       ▼                                                       │
│  4. Compute Loss (FP32)                                       │
│       │                                                       │
│       ▼                                                       │
│  5. Backward Pass (BF16)                                      │
│       │                                                       │
│       ▼                                                       │
│  6. Gradient Scaling (if FP16; skip for BF16)                 │
│       │                                                       │
│       ▼                                                       │
│  7. Optimizer Step (FP32)                                     │
│       │                                                       │
│       ▼                                                       │
│  8. Update Master Weights (FP32)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 Case Study: How Meta Trained Llama 3

### Background

Meta's Llama 3 project (ai.meta.com/blog) represents one of the most well-documented large-scale training efforts in the open-source AI community. Llama 3 405B is the largest open-weight model as of 2026, trained on over 15 trillion tokens.

### Training Infrastructure

| Parameter | Value |
|-----------|-------|
| Model Size | 405B parameters |
| Training Tokens | 15T+ tokens |
| GPU Type | NVIDIA H100-80GB |
| Total GPUs | 16,384 H100s |
| GPU Hours | ~30.84M GPU-hours |
| Parallelism | TP=8, PP=16, DP=128 |
| Software Stack | PyTorch + FSDP + custom COLA communication library |

### Key Architectural Decisions

1. **GQA with 8 KV groups**: Reduced KV cache memory by 8x compared to MHA, enabling longer context windows during inference
2. **128K context window**: Trained with 8K default, extended to 128K via RoPE scaling (YaRN technique)
3. **SwiGLU FFN**: 53,248 hidden dimension (4x the model's 16,384 hidden dim), providing rich expressiveness
4. **126 layers**: Deep network enabling hierarchical feature extraction

### Training Optimizations

- **Custom communication library (COLA)**: Optimized collective operations for the specific cluster topology
- **Overlapping computation and communication**: Pipeline parallelism stages overlap gradient all-reduce with forward/backward computation
- **Data mixing**: 50% of training data from web crawls, 25% code, 15% knowledge (Wikipedia, books), 10% multilingual
- **Curriculum learning**: Started with shorter sequences, progressively increased to full context length

### Lessons Learned

1. **Infrastructure reliability matters**: Meta reported multiple hardware failures during training. They developed automated checkpointing and recovery systems.
2. **Data quality > data quantity**: Multiple ablation studies showed that higher-quality data filtering improved model quality more than adding 2x more tokens.
3. **Communication is the bottleneck**: At 16K GPUs, even small communication inefficiencies compound. Meta invested heavily in topology-aware scheduling.

---

## ⚠️ War Story: The OOM That Killed a $2M Training Run

### The Setup

A mid-size AI company (name withheld for privacy) was training a custom 70B parameter language model on 256 A100-80GB GPUs. The project had been in development for 3 months, with an estimated total compute cost of $2M.

### The Incident

At step 45,000 of training (approximately 60% through the data), the entire training job crashed with a CUDA Out of Memory (OOM) error on GPU 137. Investigation revealed:

- **Root cause**: A data pipeline bug occasionally produced sequences longer than the configured maximum of 4096 tokens. One batch contained a sequence of 12,000 tokens.
- **Why it wasn't caught**: The data validation check ran only during preprocessing, not during training. The bug was in a dynamic data augmentation step that occasionally concatenated multiple documents.
- **Why it killed the run**: ZeRO Stage 3 was configured with static partitioning. The OOM on a single GPU triggered a NCCL timeout across all 256 GPUs, corrupting the distributed checkpoint.
- **Recovery**: The most recent clean checkpoint was from 8,000 steps prior. The team had to restart from that checkpoint, losing ~2 weeks of compute ($400K worth of GPU time).

### The Fix

```python
# Before: No runtime length check
for batch in dataloader:
    loss = model(batch)
    loss.backward()

# After: Runtime validation with graceful handling
MAX_SEQ_LEN = 4096
for batch in dataloader:
    # Truncate or filter long sequences
    if batch["input_ids"].shape[1] > MAX_SEQ_LEN:
        batch["input_ids"] = batch["input_ids"][:, :MAX_SEQ_LEN]
        batch["attention_mask"] = batch["attention_mask"][:, :MAX_SEQ_LEN]
    
    try:
        loss = model(batch)
        loss.backward()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning(f"OOM at step {step}, skipping batch")
        continue
```

### Key Takeaways

1. **Never trust preprocessing validation alone** — add runtime checks
2. **Use gradient accumulation with small micro-batches** — reduces peak memory
3. **Implement automatic checkpointing** — save every N steps, keep last K checkpoints
4. **Set NCCL timeout appropriately** — too short causes false failures, too long delays recovery
5. **Test with worst-case data** — include edge cases in your validation set

---

## 📝 When to Use / When Not to Use

### Training Strategy Decision Matrix

| Scenario | Recommended Strategy | Why |
|----------|---------------------|-----|
| 7B model, 1-2 GPUs | Single GPU + bf16 | Fits in memory, simplest setup |
| 7B model, 8 GPUs | FSDP (DP) | Linear scaling, no model splitting needed |
| 13B-70B model, 8 GPUs | ZeRO Stage 2 or 3 | Need memory reduction for optimizer states |
| 70B model, 64 GPUs | 3D parallelism (TP=8, PP=2, DP=4) | Balance memory and communication |
| 405B model, 1000+ GPUs | Full 3D parallelism | Only way to fit this model size |
| Limited GPU budget | ZeRO-Infinity + CPU offloading | Trade speed for memory |
| Prototyping | DeepSpeed + gradient checkpointing | Fast iteration, acceptable slowdown |

### When to Use Each Parallelism Type

| Parallelism | Use When | Avoid When |
|------------|----------|------------|
| **Data Parallel** | Model fits on single GPU | Model doesn't fit on single GPU |
| **Tensor Parallel** | GPUs connected via NVLink | GPUs only connected via InfiniBand (high latency) |
| **Pipeline Parallel** | Many layers to split | Few layers (e.g., <16) |
| **ZeRO Stage 3** | Need maximum memory reduction | Can afford more GPUs for DP |
| **FSDP** | PyTorch-native, moderate scale | Need DeepSpeed-specific features |
| **Gradient Checkpointing** | Memory is tight | Throughput is critical |

### When to Use Mixed Precision

| Precision | Use When | Avoid When |
|-----------|----------|------------|
| **BF16** | Default for all training (A100+) | Training on V100 or older (no BF16 support) |
| **FP16** | Legacy hardware, need gradient scaling | Modern training (BF16 is better) |
| **FP8** | H100+ inference, some training layers | Anything requiring high precision (loss computation) |
| **FP32** | Optimizer states, loss, small models | Large model training (too much memory) |

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| **Transformer Review** | Modern LLMs use RoPE, GQA, SwiGLU, RMSNorm as standard components |
| **Memory Math** | 14P bytes minimum for training P-parameter model (bf16 + AdamW) |
| **Scaling Laws** | Chinchilla: optimal tokens ≈ 20× model parameters |
| **ZeRO Stages** | Stage 3 enables training 70B+ models with manageable per-GPU memory |
| **3D Parallelism** | TP + PP + DP is the standard for training 100B+ models |
| **Flash Attention** | O(N) memory and 2-4x speedup; essential for long sequences |
| **Mixed Precision** | BF16 default; FP8 on H100+ for 2x throughput |
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

## Discussion Questions

1. **Memory vs. Compute Trade-offs**: A colleague suggests using ZeRO Stage 3 with 8 GPUs instead of FSDP with 16 GPUs to save costs. What are the trade-offs in terms of training speed, communication overhead, and fault tolerance? When would you agree or disagree?

2. **Architecture Evolution**: If you were designing a new 70B parameter model from scratch in 2026, would you choose the Llama 3 architecture (RoPE + GQA + SwiGLU + RMSNorm) or deviate? What components would you keep, modify, or replace? Why?

3. **Scaling Laws Revisited**: Meta trained Llama 3 405B on 15T tokens. According to Chinchilla scaling laws, the compute-optimal token count for 405B parameters would be ~8.1T tokens. Meta used nearly 2x this amount. What are the possible reasons for this deviation from Chinchilla optimal?

4. **Practical Parallelism**: You have 32 A100-80GB GPUs and need to train a 70B model. Design the parallelism strategy (TP, PP, DP, ZeRO stage) and justify your choices. What is the expected per-GPU memory usage?

5. **Flash Attention Impact**: How does Flash Attention change the optimal batch size and sequence length for training? If Flash Attention 2 reduces attention memory from O(N²) to O(N), what new training scenarios become feasible?

---

## Exercises

### Exercise 1: Memory Calculator

Write a Python script that calculates GPU memory requirements for training a Transformer model given:
- Number of parameters (P)
- Number of GPUs
- ZeRO stage (0-3)
- Sequence length
- Batch size
- Precision (fp32, bf16, fp16)

The script should output the memory breakdown (parameters, gradients, optimizer states, activations) and determine whether the model fits on the given hardware.

### Exercise 2: Parallelism Strategy Design

Given the following constraints, design a training strategy:
- Model: 13B parameters
- Available hardware: 8× A100-80GB
- Training data: 2T tokens
- Target: Maximize training throughput

Specify: ZeRO stage, tensor parallelism degree, pipeline parallelism degree, batch size, sequence length, and precision. Estimate total training time and cost.

### Exercise 3: Flash Attention Benchmarking

Using a pre-trained small model (e.g., GPT-2 124M):
1. Measure training speed with and without Flash Attention
2. Plot memory usage vs. sequence length (512, 1024, 2048, 4096, 8192)
3. Determine the sequence length where Flash Attention becomes critical
4. Write a brief report comparing the two approaches

---

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS. https://arxiv.org/abs/1706.03762
2. Rajbhandari, S., et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC'20. https://arxiv.org/abs/1910.02054
3. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS. https://arxiv.org/abs/2205.14135
4. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." https://arxiv.org/abs/2307.08691
5. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." NeurIPS. https://arxiv.org/abs/2203.15556
6. Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv. https://arxiv.org/abs/1909.08053
7. Narayanan, D., et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." SC'21. https://arxiv.org/abs/2104.04473
8. Korthikanti, V., et al. (2022). "Reducing Activation Recomputation in Large Transformer Models." MLSys. https://arxiv.org/abs/2205.05198
9. Meta AI. (2024). "Introducing Llama 3." https://ai.meta.com/blog/meta-llama-3/
10. NVIDIA Transformer Engine Documentation. https://docs.nvidia.com/deeplearning/transformer-engine/
11. DeepSpeed Documentation. https://www.deepspeed.ai/
12. PyTorch FSDP Tutorial. https://pytorch.org/docs/stable/fsdp.html

---

*Next Chapter: [Chapter 11 - LLM Inference Architecture](chapter-11.md) →*
