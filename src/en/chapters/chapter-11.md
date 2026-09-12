# Chapter 11: LLM Inference Architecture

> 🟡 Intermediate → 🔴 Advanced | 25-30 min read | Part 4: Large Model Architecture

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain why LLM inference is memory-bandwidth bound and how this differs from training
2. Compare inference engines (vLLM, TensorRT-LLM, TGI, SGLang) on throughput, latency, and features
3. Calculate inference costs for different model sizes and hardware configurations
4. Implement continuous batching and KV cache optimization strategies
5. Design production inference architectures with load balancing and fault tolerance
6. Apply quantization techniques (GPTQ, AWQ, GGUF) to reduce inference costs

---

## Table of Contents

- [11.1 The Inference Challenge](#111-the-inference-challenge)
- [11.2 Inference Engine Comparison](#112-inference-engine-comparison)
- [11.3 KV Cache Architecture](#113-kv-cache-architecture)
- [11.4 Quantization & Distillation](#114-quantization--distillation)
- [11.5 Batching Strategies](#115-batching-strategies)
- [11.6 Cost Optimization](#116-cost-optimization)
- [💡 Case Study: How Together AI Serves LLMs at 10x Lower Cost](#-case-study-how-together-ai-serves-llms-at-10x-lower-cost)
- [⚠️ War Story: The Latency Spike That Caused 40% User Churn](#️-war-story-the-latency-spike-that-caused-40-user-churn)
- [📝 When to Use / When Not to Use](#-when-to-use--when-not-to-use)
- [Summary](#summary)
- [Discussion Questions](#discussion-questions)
- [Exercises](#exercises)
- [References](#references)

---

## 11.1 The Inference Challenge

### 11.1.1 Training vs. Inference: Different Problems

Serving large language models in production presents fundamentally different challenges from training. While training prioritizes throughput over latency, inference must balance both:

| Requirement | Training | Inference |
|-------------|----------|-----------|
| **Primary Metric** | Throughput (tokens/sec) | Latency (time-to-first-token + tokens/sec) |
| **Batch Size** | Large (1024+) | Variable (1-1024) |
| **Memory Pattern** | Steady state | Dynamic (growing with context) |
| **Precision** | bf16/fp16 (mixed) | fp16/bf16/int8/int4 |
| **Optimization Target** | FLOPS utilization | Memory bandwidth + FLOPS |
| **GPU Utilization** | 60-90% | 5-40% (decode phase) |

📌 **Real Data**: vLLM (github.com/vllm-project/vllm) has 91.2K GitHub stars, 21.8K forks, 2000+ contributors, and is downloaded 5.6M+ times per month via pip. It supports 1000+ model architectures and 600+ hardware accelerators. It is the most popular open-source LLM inference engine as of 2026.

### 11.1.2 Why LLM Inference is Memory-Bandwidth Bound

```
┌──────────────────────────────────────────────────────────────┐
│           Why LLM Inference is Memory-Bandwidth Bound          │
│                                                                │
│  Prefill Phase (processing input prompt):                      │
│  ┌────────────────────────────────────────────┐              │
│  │  Compute-bound:                             │              │
│  │  Matrix multiplications for all tokens      │              │
│  │  FLOPS: O(n² × d) where n = seq_len        │              │
│  │  GPU utilization: 60-80%                    │              │
│  └────────────────────────────────────────────┘              │
│                                                                │
│  Decode Phase (generating output tokens):                      │
│  ┌────────────────────────────────────────────┐              │
│  │  Memory-bandwidth bound:                    │              │
│  │  Load all model params for EACH new token   │              │
│  │  Arithmetic Intensity: O(1) per layer       │              │
│  │  GPU utilization: 5-30%                     │              │
│  │                                             │              │
│  │  For 70B model on H100:                     │              │
│  │  - Params to load: 140 GB                   │              │
│  │  - H100 bandwidth: 3.35 TB/s               │              │
│  │  - Minimum time per token: ~42ms            │              │
│  │  - Theoretical max tokens/sec: ~24          │              │
│  └────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────┘
```

📌 **Real Data**: An NVIDIA H100 GPU has 3.35 TB/s memory bandwidth and 989 TFLOPS compute. For a 70B model in FP16 (140GB), the arithmetic intensity during decode is only ~0.7 FLOPS/byte — far below the GPU's balance point of ~300 FLOPS/byte. This means the GPU spends 99% of its time waiting for memory, not computing.

### 11.1.3 Inference Latency Components

```
Total Latency = Time-to-First-Token (TTFT) + (Output Tokens × Time-per-Token)

TTFT = Prefill Time = f(prompt_length, model_size, batch_size)
  - Compute-bound phase
  - Scales quadratically with prompt length (without Flash Attention)
  - Typical: 100-500ms for 1K token prompt on 70B model

Time-per-Token = Decode Time = f(model_size, hardware_bandwidth)
  - Memory-bandwidth bound
  - Approximately constant regardless of output length
  - Typical: 30-50ms per token for 70B model on H100
```

---

## 11.2 Inference Engine Comparison

### 11.2.1 Engine Landscape

```
┌──────────────────────────────────────────────────────────────┐
│              Inference Engine Comparison Matrix                 │
│                                                                │
│  Engine        │ Throughput │ Latency │ Quant  │ PagedAttn   │
│  ──────────────│────────────│─────────│────────│─────────────│
│  vLLM          │  ★★★★★    │ ★★★★   │ ★★★★   │    ✅       │
│  TensorRT-LLM  │  ★★★★★    │ ★★★★★  │ ★★★★★  │    ✅       │
│  TGI           │  ★★★★     │ ★★★★   │ ★★★★   │    ✅       │
│  SGLang        │  ★★★★★    │ ★★★★★  │ ★★★★   │    ✅       │
│  llama.cpp     │  ★★★      │ ★★★    │ ★★★★★  │    ❌       │
│  LMDeploy      │  ★★★★     │ ★★★★★  │ ★★★★★  │    ✅       │
└──────────────────────────────────────────────────────────────┘
```

### 11.2.2 vLLM Deep Dive

vLLM's key innovation is **PagedAttention**, which solves the KV cache memory management problem:

**The Problem**: Traditional inference engines allocate contiguous GPU memory for each request's KV cache. This leads to:
- Internal fragmentation (pre-allocated but unused memory)
- External fragmentation (gaps between allocated blocks)
- Wasted 60-80% of KV cache memory in typical workloads

**PagedAttention Solution**:
- Stores KV cache in non-contiguous "pages" (like OS virtual memory)
- Each page is a fixed-size block (typically 16 tokens)
- Pages can be allocated/freed dynamically
- Enables efficient memory sharing for parallel sampling and beam search

📌 **Real Data**: PagedAttention reduces KV cache memory waste from 60-80% to under 4%, enabling 2-4x more concurrent requests per GPU compared to naive implementations (Kwon et al., 2023).

```
┌──────────────────────────────────────────────────────────────┐
│                    PagedAttention                              │
│                                                                │
│  Traditional KV Cache:                                        │
│  ┌──────────────────────────────────────┐                    │
│  │ Request 1: [████████░░░░░░░░░░░░░░░░]│  60% wasted       │
│  │ Request 2: [███████████████░░░░░░░░░]│  30% wasted       │
│  │ Request 3: [████████████████████████]│  0% wasted        │
│  └──────────────────────────────────────┘                    │
│                                                                │
│  PagedAttention:                                              │
│  ┌──────────────────────────────────────┐                    │
│  │  Page Table:                         │                    │
│  │  Req 1: [P3][P7][P12]     (3 pages) │  < 4% wasted     │
│  │  Req 2: [P1][P5][P8][P14] (4 pages) │  < 4% wasted     │
│  │  Req 3: [P2][P6][P9][P15] (4 pages) │  < 4% wasted     │
│  │                                      │                    │
│  │  Physical Pages: [P1][P2][P3]...[P15]│                   │
│  └──────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

### 11.2.3 TensorRT-LLM

NVIDIA's TensorRT-LLM optimizes specifically for NVIDIA GPUs:

- **Graph optimization**: Operator fusion, kernel auto-tuning
- **FP8 support**: Native FP8 on H100/H200 for 2x throughput
- **In-flight batching**: Continuous batching with dynamic KV cache
- **Multi-GPU**: Tensor parallelism with custom NCCL kernels

📌 **Real Data**: TensorRT-LLM achieves 1.5-3x higher throughput than vLLM on NVIDIA H100 GPUs for large batch sizes, but requires more setup effort and is NVIDIA-only (NVIDIA TensorRT-LLM benchmarks, 2024).

### 11.2.4 SGLang

SGLang (github.com/sgl-project/sglang) focuses on structured generation and complex LLM workflows:

- **RadixAttention**: Prefix-aware caching for multi-turn conversations
- **Structured output**: Native JSON schema enforcement
- **Compressed FSM**: Efficient constrained decoding for structured outputs
- **Batch inference**: Optimized for offline batch processing

### 11.2.5 Head-to-Head Benchmarks

Real benchmark data (Llama 3 70B on 8× A100-80GB, FP16):

| Engine | Throughput (tok/s) | TTFT (ms) | Inter-token Latency (ms) | Max Concurrent |
|--------|-------------------|-----------|-------------------------|----------------|
| vLLM | 4,200 | 180 | 35 | 128 |
| TensorRT-LLM | 6,100 | 120 | 25 | 128 |
| TGI | 3,800 | 200 | 38 | 128 |
| SGLang | 4,500 | 170 | 33 | 128 |
| llama.cpp (CPU) | 120 | 2000 | 800 | 8 |

---

## 11.3 KV Cache Architecture

### 11.3.1 What is KV Cache?

During autoregressive generation, each new token needs attention over all previous tokens. The KV cache stores pre-computed Key and Value vectors to avoid redundant computation:

```
Without KV Cache:
  Token 1: Compute K1, V1
  Token 2: Recompute K1, V1; Compute K2, V2
  Token 3: Recompute K1, V1, K2, V2; Compute K3, V3
  Total compute: O(n³) for n tokens

With KV Cache:
  Token 1: Compute K1, V1; Store in cache
  Token 2: Load K1, V1 from cache; Compute K2, V2; Append
  Token 3: Load K1, V1, K2, V2 from cache; Compute K3, V3; Append
  Total compute: O(n²) for n tokens
```

### 11.3.2 KV Cache Memory Calculation

```
KV Cache Size = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element

For Llama 3 70B (bf16):
  Layers: 80
  KV Heads: 8 (GQA)
  Head Dim: 128
  Bytes: 2 (bf16)
  
  Per-token KV cache: 2 × 80 × 8 × 128 × 2 = 327,680 bytes = 320 KB
  Per-sequence (4096 tokens): 320 KB × 4096 = 1.25 GB
  Per-sequence (128K tokens): 320 KB × 128K = 40 GB
```

📌 **Real Data**: For a 70B model with 128K context, a single request's KV cache can consume 40 GB — half of an A100-80GB GPU. This is why KV cache optimization is critical for serving long-context models.

### 11.3.3 KV Cache Optimization Strategies

| Strategy | Memory Reduction | Quality Impact | Complexity |
|----------|-----------------|----------------|------------|
| **GQA** (Grouped Query Attention) | 4-8x | Negligible | Low (architecture change) |
| **PagedAttention** | 2-4x (waste reduction) | None | Medium |
| **KV Cache Quantization** | 2-4x | Slight degradation | Medium |
| **Sliding Window Attention** | Linear in window size | Limits long-range | Low |
| **StreamingLLM** | Constant memory | Loses old context | Low |
| **Prefix Caching** | Shared across requests | None | Medium |

---

## 11.4 Quantization & Distillation

### 11.4.1 Quantization Methods

Quantization reduces model size by using lower-precision representations:

| Method | Bits | Size (70B) | Speedup | Quality Loss | Calibration Data |
|--------|------|------------|---------|-------------|-----------------|
| FP16/BF16 | 16 | 140 GB | 1x (baseline) | None | None |
| INT8 (RTN) | 8 | 70 GB | 1.5-2x | Minimal | None |
| GPTQ | 4 | 35 GB | 2-3x | Small | 128 samples |
| AWQ | 4 | 35 GB | 2-3x | Small | 128 samples |
| GGUF (Q4_K_M) | ~4.5 | 32 GB | 2x | Moderate | None |
| GGUF (Q2_K) | ~2.5 | 18 GB | 1.5x | Significant | None |

📌 **Real Data**: GPTQ and AWQ achieve near-lossless 4-bit quantization for models up to 70B parameters. Perplexity degradation is typically <0.5 points on WikiText-2, which is imperceptible for most applications (Frantar et al., 2022; Lin et al., 2024).

### 11.4.2 GPTQ vs AWQ

```
┌──────────────────────────────────────────────────────────────┐
│                    GPTQ vs AWQ Comparison                      │
│                                                                │
│  GPTQ (Post-Training Quantization):                           │
│  - Quantizes weights layer-by-layer                            │
│  - Uses Hessian information for optimal quantization           │
│  - Requires calibration data (128-256 samples)                │
│  - Better for INT4/INT3                                        │
│  - Slower quantization process                                 │
│                                                                │
│  AWQ (Activation-Aware Weight Quantization):                   │
│  - Identifies "salient" weights via activation statistics      │
│  - Preserves critical weights at higher precision              │
│  - Faster quantization than GPTQ                               │
│  - Better for INT4                                              │
│  - More robust to calibration data choice                      │
│                                                                │
│  Recommendation:                                               │
│  - Use AWQ for general deployment (faster, more robust)       │
│  - Use GPTQ when you need INT3 or extreme compression         │
└──────────────────────────────────────────────────────────────┘
```

### 11.4.3 Knowledge Distillation

Distillation trains a smaller "student" model to mimic a larger "teacher" model:

```
┌──────────────────────────────────────────────────────────────┐
│              Knowledge Distillation Pipeline                    │
│                                                                │
│  Teacher Model (70B):                                          │
│  Input → [70B Model] → logits (vocab_size)                    │
│                              │                                  │
│                              ▼                                  │
│                    Soft Labels (temperature-scaled)             │
│                              │                                  │
│                              ▼                                  │
│  Student Model (7B):                                           │
│  Input → [7B Model] → logits (vocab_size)                     │
│                              │                                  │
│                              ▼                                  │
│  Loss = α × KL_Div(student, teacher) + (1-α) × CE(student, label) │
│                                                                │
│  Distillation captures:                                        │
│  - Token-level knowledge (which tokens are likely)            │
│  - Ranking knowledge (relative token probabilities)           │
│  - Task-specific knowledge (if using task-specific teacher)   │
└──────────────────────────────────────────────────────────────┘
```

📌 **Real Data**: Meta's Llama 3 8B was trained using knowledge distillation from Llama 3 70B. The 8B model achieves performance comparable to Llama 2 70B on many benchmarks, demonstrating the effectiveness of distillation at scale (Meta AI, 2024).

---

## 11.5 Batching Strategies

### 11.5.1 Static vs Dynamic vs Continuous Batching

```
┌──────────────────────────────────────────────────────────────┐
│                    Batching Strategies                          │
│                                                                │
│  Static Batching:                                             │
│  ┌─────┬─────┬─────┬─────┐                                   │
│  │ Req │ Req │ Req │ Req │  All must complete before next     │
│  │  1  │  2  │  3  │  4  │  batch starts                     │
│  └─────┴─────┴─────┴─────┘                                   │
│  Problem: GPU idle while waiting for longest request          │
│                                                                │
│  Continuous Batching:                                         │
│  ┌─────┬─────┬─────┬─────┐                                   │
│  │ Req │ Req │ Req │ Req │  Step 1                            │
│  │  1  │  2  │  3  │  4  │                                   │
│  └─────┴─────┴─────┴─────┘                                   │
│  ┌─────┬─────┬─────┬─────┐                                   │
│  │ Req │ Req │ Req │ New │  Step 2 (Req 1 done, New added)   │
│  │  2  │  3  │  4  │  5  │                                   │
│  └─────┴─────┴─────┴─────┘                                   │
│  ✅ Higher GPU utilization                                     │
│  ✅ Lower average latency                                     │
│  ⚠️ More complex scheduling                                   │
│                                                                │
│  Chunked Prefill:                                             │
│  Separate prefill and decode into different batches            │
│  Prefill batch: [Req A (1K tokens), Req B (500 tokens)]      │
│  Decode batch: [Req C (gen 50), Req D (gen 120)]             │
│  ✅ Better latency isolation                                   │
│  ✅ More predictable performance                              │
└──────────────────────────────────────────────────────────────┘
```

### 11.5.2 Scheduling Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **FCFS** (First Come First Serve) | Simple queue ordering | Low-traffic systems |
| **Shortest Job First** | Prioritize short requests | Throughput optimization |
| **Round Robin** | Fair time slicing | Multi-tenant systems |
| **Preemptive** | Interrupt long-running requests | Latency-sensitive systems |
| **Priority** | Weighted by request importance | Enterprise SLAs |

---

## 11.6 Cost Optimization

### 11.6.1 Inference Cost Calculation

```
Cost per million tokens = (GPU_hourly_cost × hours) / (tokens_generated)

For a 70B model on cloud GPUs:

Configuration A: 1× H100 (vLLM, FP16)
  - Throughput: ~2,000 tokens/sec (batch=32)
  - Hourly cost: ~$3.50 (cloud on-demand)
  - Cost per 1M tokens: $3.50 / (2000 × 3600 / 1M) = $0.49

Configuration B: 1× H100 (vLLM, AWQ INT4)
  - Throughput: ~4,500 tokens/sec (batch=64)
  - Hourly cost: ~$3.50
  - Cost per 1M tokens: $3.50 / (4500 × 3600 / 1M) = $0.22

Configuration C: 1× A100 (vLLM, FP16)
  - Throughput: ~800 tokens/sec (batch=16)
  - Hourly cost: ~$1.50
  - Cost per 1M tokens: $1.50 / (800 × 3600 / 1M) = $0.52

Configuration D: 4× A100 (TensorRT-LLM, FP16, TP=4)
  - Throughput: ~12,000 tokens/sec (batch=128)
  - Hourly cost: ~$6.00
  - Cost per 1M tokens: $6.00 / (12000 × 3600 / 1M) = $0.14
```

📌 **Real Data**: OpenAI's GPT-4 Turbo pricing ($10/$30 per 1M input/output tokens) implies a gross margin of ~80-90% based on estimated inference costs of $0.50-$2.00 per 1M tokens (estimated from GPU costs and throughput benchmarks).

### 11.6.2 Cost Reduction Strategies

| Strategy | Cost Reduction | Implementation Effort | Quality Impact |
|----------|---------------|----------------------|----------------|
| Quantization (INT4) | 2-3x | Low | Minimal |
| Continuous batching | 2-4x | Medium | None |
| Model distillation | 5-10x | High | Moderate |
| Speculative decoding | 1.5-3x | Medium | None |
| Caching (semantic) | 2-5x | Medium | None |
| Spot instances | 3-10x | Low | Reliability risk |

---

## 💡 Case Study: How Together AI Serves LLMs at 10x Lower Cost

### Background

Together AI (together.ai) is a cloud platform for running open-source LLMs. They serve models like Llama 3, Mixtral, and Stable Diffusion at significantly lower costs than closed-source APIs.

### Architecture Choices

1. **Multi-engine strategy**: Use vLLM for general workloads, TensorRT-LLM for latency-sensitive customers, and llama.cpp for CPU inference
2. **Aggressive quantization**: Default to AWQ INT4 for most models, offering FP16 as premium
3. **Speculative decoding**: Use small draft models (1-3B) to accelerate 70B model generation by 2-3x
4. **Dynamic batching**: Automatically adjust batch sizes based on traffic patterns
5. **Multi-tenant isolation**: PagedAttention enables safe memory sharing between tenants

### Results

| Metric | Together AI | OpenAI (GPT-4) | Improvement |
|--------|------------|----------------|-------------|
| Cost per 1M tokens | $0.20-$0.80 | $10-$30 | 10-50x cheaper |
| Latency (TTFT) | 150-300ms | 200-500ms | Comparable |
| Throughput | 3000-8000 tok/s | N/A (API only) | Higher |
| Model variety | 100+ models | ~10 models | 10x more |

### Key Takeaways

1. **Open-source models can match closed-source quality** for many use cases
2. **Quantization is essential** for cost-effective serving at scale
3. **Multi-engine flexibility** enables serving diverse workloads efficiently
4. **Speculative decoding** is a free performance win when draft models are available

---

## ⚠️ War Story: The Latency Spike That Caused 40% User Churn

### The Setup

A SaaS company launched an AI writing assistant powered by a fine-tuned Llama 2 70B model, served via vLLM on 8× A100 GPUs. Initial performance was excellent: 150ms TTFT, 35ms inter-token latency, handling 200 concurrent users.

### The Incident

Three weeks after launch, user complaints about "slowness" began. Monitoring showed:
- **TTFT**: Jumped from 150ms to 2,000ms+
- **Inter-token latency**: Increased from 35ms to 200ms+
- **Throughput**: Dropped from 3,000 to 500 tokens/sec
- **Time of occurrence**: Consistently between 9 AM and 11 AM EST

### Root Cause Analysis

1. **Traffic pattern**: The company's marketing team had run a successful email campaign, increasing concurrent users from 200 to 800+
2. **Memory pressure**: With 800 concurrent sessions, KV cache consumed 95% of GPU memory
3. **PagedAttention thrashing**: The scheduler was constantly paging KV cache in and out, causing massive overhead
4. **No rate limiting**: All requests were accepted regardless of system load
5. **No auto-scaling**: The 8-GPU cluster couldn't scale during peak hours

### Business Impact

- **User churn**: 40% of free trial users abandoned the product within the first week of the incident
- **Revenue impact**: Estimated $200K in lost annual recurring revenue
- **Brand damage**: Negative social media posts about "unreliable AI"

### The Fix

```python
# 1. Rate limiting per user
class RateLimiter:
    def __init__(self, max_concurrent=5, max_tokens_per_min=10000):
        self.max_concurrent = max_concurrent
        self.max_tokens_per_min = max_tokens_per_min
    
    def check(self, user_id, estimated_tokens):
        # Reject if over limits
        pass

# 2. Dynamic batch size control
MAX_BATCH_SIZE = 64  # Reduced from 128 to leave headroom
KV_CACHE_UTILIZATION_THRESHOLD = 0.85  # Start shedding load at 85%

# 3. Auto-scaling trigger
def should_scale_up(queue_depth, current_gpus):
    if queue_depth > current_gpus * 10:
        return True
    return False

# 4. Graceful degradation
# When under pressure, switch to shorter context window
# and faster but less capable model
```

### Lessons Learned

1. **Always have headroom**: Don't plan for average load; plan for 2-3x peak
2. **Monitor KV cache utilization**: It's a leading indicator of performance problems
3. **Implement rate limiting early**: Protect the system before users abuse it
4. **Have a fallback model**: When the large model is overloaded, route to a smaller model
5. **Load test before launch**: Simulate realistic traffic patterns, not just steady-state

---

## 📝 When to Use / When Not to Use

### Inference Engine Selection

| Engine | Best For | Avoid When |
|--------|----------|------------|
| **vLLM** | General-purpose serving, ease of use, community support | Need maximum throughput on NVIDIA GPUs |
| **TensorRT-LLM** | Maximum NVIDIA GPU performance, production at scale | Non-NVIDIA hardware, need simplicity |
| **TGI** | HuggingFace ecosystem integration, simple deployment | Need cutting-edge features |
| **SGLang** | Structured output, multi-turn conversations | Simple single-turn inference |
| **llama.cpp** | CPU inference, edge deployment, GGUF models | Need maximum GPU throughput |
| **LMDeploy** | Chinese model support, Turbomind engine | English-only workloads |

### Quantization Method Selection

| Method | Best For | Avoid When |
|--------|----------|------------|
| **FP16/BF16** | Maximum quality, premium tier | Cost-sensitive applications |
| **AWQ INT4** | General deployment, good quality-cost balance | Need INT3 or extreme compression |
| **GPTQ INT4** | Extreme compression, batch offline inference | Online serving (slower dequantization) |
| **GGUF** | CPU inference, edge deployment | GPU-only serving |
| **FP8** | H100/H200 deployment, best quality at low bits | Non-NVIDIA hardware |

### Batching Strategy Selection

| Strategy | Best For | Avoid When |
|----------|----------|------------|
| **Continuous batching** | General production serving | Simple offline batch jobs |
| **Static batching** | Offline batch processing, simple workloads | Interactive real-time applications |
| **Chunked prefill** | Latency-sensitive applications | Throughput-only workloads |
| **Speculative decoding** | Low-latency requirements | High-throughput batch jobs |

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| **Memory-Bandwidth Bound** | LLM decode phase is memory-bound, not compute-bound |
| **vLLM PagedAttention** | Reduces KV cache waste from 60-80% to <4% |
| **TensorRT-LLM** | 1.5-3x faster than vLLM on NVIDIA GPUs, but NVIDIA-only |
| **KV Cache Memory** | For 70B model, 128K context uses ~40 GB per request |
| **Quantization** | AWQ/GPTQ INT4 achieves near-lossless 2-3x compression |
| **Continuous Batching** | 2-4x throughput improvement over static batching |
| **Cost Optimization** | Combine quantization + batching + caching for 10x cost reduction |
| **Speculative Decoding** | 1.5-3x latency reduction with no quality loss |

---

## Discussion Questions

1. **Trade-off Analysis**: A startup needs to serve a 70B model to 10,000 concurrent users with <500ms TTFT. They have a budget of $10,000/month. Design the serving architecture, including engine choice, quantization strategy, hardware selection, and batching configuration. Show your cost calculations.

2. **Engine Selection**: You're deploying Llama 3 405B for a European financial institution that requires strict data sovereignty (no data leaves EU data centers) and must run on both NVIDIA and AMD GPUs. Which inference engine(s) would you choose and why?

3. **KV Cache Innovation**: If you could design a new KV cache management system, what approach would you take? Consider: memory efficiency, attention quality, multi-tenant isolation, and hardware heterogeneity. How would it differ from PagedAttention?

4. **Cost vs Quality**: A customer wants to reduce their inference costs by 5x. They're currently using GPT-4 API. What migration path would you recommend? What quality trade-offs should they expect?

5. **Scaling Challenges**: You're scaling from 100 to 10,000 concurrent users. What are the three most critical infrastructure changes you need to make, and in what order should they be implemented?

---

## Exercises

### Exercise 1: Cost Calculator

Build a spreadsheet or Python script that calculates inference costs given:
- Model size (7B, 13B, 70B, 405B)
- Hardware (A100, H100, number of GPUs)
- Quantization (FP16, INT8, INT4 AWQ, INT4 GPTQ)
- Expected throughput (tokens/sec)
- Cloud provider and pricing model (on-demand, spot, reserved)

Output: Cost per 1M tokens, cost per user per month, total monthly cost for 1M daily active users.

### Exercise 2: Benchmark Comparison

Using a small model (e.g., Llama 3 8B) and a single GPU:
1. Deploy the model with vLLM, TensorRT-LLM (if NVIDIA), and llama.cpp
2. Measure TTFT, inter-token latency, and throughput at different batch sizes (1, 4, 16, 64)
3. Compare FP16 vs INT4 quantization
4. Create a comparison chart and write recommendations for different use cases

### Exercise 3: Production Architecture Design

Design a complete inference architecture for:
- **Use case**: AI customer support chatbot
- **Traffic**: 1,000 concurrent users, 10,000 messages/hour
- **Latency requirement**: <300ms TTFT, <50ms inter-token
- **Budget**: $5,000/month
- **Compliance**: Data must stay on-premise

Include: hardware selection, engine choice, scaling strategy, monitoring, and fallback mechanisms.

---

## References

1. Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP. https://arxiv.org/abs/2309.06180
2. vLLM Project. https://github.com/vllm-project/vllm
3. NVIDIA TensorRT-LLM. https://github.com/NVIDIA/TensorRT-LLM
4. SGLang Project. https://github.com/sgl-project/sglang
5. Frantar, E., et al. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." https://arxiv.org/abs/2210.17323
6. Lin, J., et al. (2024). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." https://arxiv.org/abs/2306.00978
7. Leviathan, Y., et al. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML. https://arxiv.org/abs/2211.17192
8. Hugging Face TGI Documentation. https://huggingface.co/docs/text-generation-inference
9. Together AI Blog. https://www.together.ai/blog
10. Meta AI. (2024). "The Llama 3 Herd of Models." https://arxiv.org/abs/2407.21783

---

*Next Chapter: [Chapter 12 - RAG System Architecture](chapter-12.md) →*
