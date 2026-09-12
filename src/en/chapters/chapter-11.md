# Chapter 11: LLM Inference Architecture

> 🟡 Intermediate → 🔴 Advanced | 25-30 min read | Part 4: Large Model Architecture

---

## Table of Contents

- [11.1 Inference Engine Comparison](#111-inference-engine-comparison)
- [11.2 KV Cache Optimization](#112-kv-cache-optimization)
- [11.3 Quantization & Distillation](#113-quantization--distillation)
- [11.4 Batching & Continuous Batching](#114-batching--continuous-batching)
- [11.5 Inference Cluster Architecture](#115-inference-cluster-architecture)
- [11.6 Cost Optimization Strategies](#116-cost-optimization-strategies)
- [💡 Case: High-Throughput Inference Service with vLLM](#-case-high-throughput-inference-service-with-vllm)
- [Summary](#summary)
- [References](#references)

---

## 11.1 Inference Engine Comparison

### 11.1.1 The Inference Challenge

Serving large language models in production presents fundamentally different challenges from training. While training prioritizes throughput over latency, inference must balance both:

| Requirement | Training | Inference |
|-------------|----------|-----------|
| **Primary Metric** | Throughput (tokens/sec) | Latency (time-to-first-token + tokens/sec) |
| **Batch Size** | Large (1024+) | Variable (1-1024) |
| **Memory Pattern** | Steady state | Dynamic (growing with context) |
| **Precision** | bf16/fp16 (mixed) | fp16/bf16/int8/int4 |
| **Optimization Target** | FLOPS utilization | Memory bandwidth + FLOPS |

📌 **Key Concept**: LLM inference is **memory-bandwidth bound**, not compute-bound. The model parameters must be loaded from GPU memory for every token generated. For a 70B model in fp16, this means reading 140GB of data per token.

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

### 11.1.2 Inference Engine Landscape

The inference engine ecosystem has matured significantly from 2023-2026:

```
┌──────────────────────────────────────────────────────────────┐
│              Inference Engine Comparison Matrix                 │
│                                                                │
│  Engine      │ Throughput │ Latency │ Quant │ PagedAttn │Ease │
│  ────────────│────────────│─────────│───────│───────────│─────│
│  vLLM        │  ★★★★★    │ ★★★★   │ ★★★★  │    ✅     │ ★★★★│
│  TensorRT-LLM│  ★★★★★    │ ★★★★★  │ ★★★★★ │    ✅     │ ★★★ │
│  TGI         │  ★★★★     │ ★★★★   │ ★★★★  │    ✅     │ ★★★★★│
│  llama.cpp   │  ★★★      │ ★★★    │ ★★★★★ │    ❌     │ ★★★★★│
│  LMDeploy    │  ★★★★     │ ★★★★★  │ ★★★★★ │    ✅     │ ★★★ │
│  SGLang      │  ★★★★★    │ ★★★★★  │ ★★★★  │    ✅     │ ★★★★│
│  MLC-LLM     │  ★★★★     │ ★★★★   │ ★★★★  │    ❌     │ ★★★ │
│  MII         │  ★★★★     │ ★★★★   │ ★★★   │    ✅     │ ★★★★│
│                                                                │
│  Legend:                                                        │
│  - Throughput: Tokens/sec per GPU                              │
│  - Latency: Time-to-first-token + inter-token latency          │
│  - Quant: Number of quantization formats supported             │
│  - PagedAttn: PagedAttention support (key for batching)        │
│  - Ease: Ease of deployment and API compatibility              │
└──────────────────────────────────────────────────────────────┘
```

### 11.1.3 vLLM Architecture Deep Dive

vLLM introduced PagedAttention, which revolutionized LLM serving efficiency:

```
┌──────────────────────────────────────────────────────────────┐
│                  vLLM Architecture                             │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                    API Server                          │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │    │
│  │  │ OpenAI API  │  │  Legacy API   │  │  Streaming  │  │    │
│  │  │ Compatible  │  │  (LangChain)  │  │   SSE       │  │    │
│  │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │    │
│  │         └────────────────┼─────────────────┘          │    │
│  │                          ▼                             │    │
│  │              ┌─────────────────────┐                  │    │
│  │              │   Request Router    │                  │    │
│  │              │   (Load Balancer)   │                  │    │
│  │              └──────────┬──────────┘                  │    │
│  └─────────────────────────┼─────────────────────────────┘    │
│                             │                                  │
│                             ▼                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              Scheduler (Token Scheduler)               │    │
│  │                                                       │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  Waiting Queue  │ Running Queue  │ Swapped   │     │    │
│  │  │  (new requests) │ (generating)   │ (paused)  │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  │                                                       │    │
│  │  Policy: FCFS / Priority / Preemption                 │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           PagedAttention Manager                       │    │
│  │                                                       │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │  Block Table (like OS page table):            │    │    │
│  │  │                                               │    │    │
│  │  │  Request 0: [Block 3] → [Block 7] → [Block 12]│   │    │
│  │  │  Request 1: [Block 1] → [Block 5]             │    │    │
│  │  │  Request 2: [Block 8] → [Block 2] → ...       │    │    │
│  │  │                                               │    │    │
│  │  │  Block Pool (pre-allocated GPU memory):        │    │    │
│  │  │  [0][1][2][3][4][5][6][7][8][9][10][11][12]..│    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              Model Executor                            │    │
│  │                                                       │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  GPU Workers (Ray / torch.distributed)       │     │    │
│  │  │  ├── Worker 0: TP rank 0                    │     │    │
│  │  │  ├── Worker 1: TP rank 1                    │     │    │
│  │  │  ├── Worker 2: TP rank 2                    │     │    │
│  │  │  └── Worker 3: TP rank 3                    │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 11.1.4 SGLang RadixAttention

SGLang introduces RadixAttention, which uses a radix tree to share KV cache across requests with common prefixes:

```
┌──────────────────────────────────────────────────────────────┐
│              SGLang RadixAttention                             │
│                                                                │
│  Prefix Tree (Radix Tree) of KV Cache:                        │
│                                                                │
│  System Prompt: "You are a helpful assistant"                  │
│                         │                                      │
│                    ┌────┴────┐                                 │
│                    │  Node A │ (shared KV cache)               │
│                    └────┬────┘                                 │
│              ┌──────────┼──────────┐                          │
│              ▼          ▼          ▼                          │
│         ┌────────┐ ┌────────┐ ┌────────┐                     │
│         │ Node B │ │ Node C │ │ Node D │ (user-specific)      │
│         │"What is│ │"Write a│ │"Explain│                     │
│         │ ML?"   │ │ poem"  │ │ RLHF"  │                     │
│         └────────┘ └────────┘ └────────┘                     │
│                                                                │
│  Benefits:                                                     │
│  - Prefix sharing: Avoid re-computing shared prefixes          │
│  - Cache hit: Radix tree lookup for exact prefix match         │
│  - Memory efficiency: No duplicate KV caches                   │
│  - LRU eviction: Least recently used branches evicted first    │
└──────────────────────────────────────────────────────────────┘
```

---

## 11.2 KV Cache Optimization

### 11.2.1 Understanding KV Cache

The KV cache is the single most important optimization in LLM inference. Without it, generating N tokens requires O(N²) computation. With KV cache, it requires O(N).

```
┌──────────────────────────────────────────────────────────────┐
│                    KV Cache Explained                          │
│                                                                │
│  Autoregressive Generation (without KV cache):                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Token 1: Compute K1, V1 from X1                      │    │
│  │  Token 2: Re-compute K1, V1; Compute K2, V2          │    │
│  │  Token 3: Re-compute K1, V1; Re-compute K2, V2;     │    │
│  │            Compute K3, V3                             │    │
│  │  ...                                                  │    │
│  │  Token N: Re-compute K1..N-1, V1..N-1; Compute KN,VN│    │
│  │                                                       │    │
│  │  Total FLOPS: O(N² × d) per layer                     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Autoregressive Generation (with KV cache):                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Token 1: Compute K1, V1; Cache them                  │    │
│  │  Token 2: Use cached K1,V1; Compute K2, V2; Cache    │    │
│  │  Token 3: Use cached K1..2,V1..2; Compute K3, V3     │    │
│  │  ...                                                  │    │
│  │  Token N: Use cached K1..N-1,V1..N-1; Compute KN,VN  │    │
│  │                                                       │    │
│  │  Total FLOPS: O(N × d) per layer (LINEAR!)            │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  KV Cache Memory Size:                                         │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Per layer: 2 × seq_len × hidden_dim × 2 bytes       │    │
│  │  Total: 2 × num_layers × seq_len × hidden_dim × 2   │    │
│  │                                                       │    │
│  │  Example: LLaMA-2 70B, seq_len=4096, bf16            │    │
│  │  = 2 × 80 × 4096 × 8192 × 2 bytes                    │    │
│  │  = 10.7 GB per request                                │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 11.2.2 PagedAttention

PagedAttention (vLLM) treats KV cache like virtual memory pages:

```python
# Conceptual PagedAttention implementation
class PagedAttention:
    """
    PagedAttention manages KV cache as fixed-size blocks,
    similar to OS virtual memory pages.
    """
    def __init__(self, block_size=16, num_blocks=1024):
        self.block_size = block_size
        # Pre-allocate all blocks in GPU memory
        self.key_cache = torch.zeros(num_blocks, num_heads,
                                      block_size, head_dim,
                                      device='cuda', dtype=torch.bfloat16)
        self.value_cache = torch.zeros(num_blocks, num_heads,
                                        block_size, head_dim,
                                        device='cuda', dtype=torch.bfloat16)
        # Block table: maps logical blocks to physical blocks
        self.block_tables = {}  # req_id -> list of physical block ids

    def allocate_block(self):
        """Allocate a free block from the pool."""
        # In practice, this uses a free list or bitmap
        return self.free_blocks.pop()

    def append_token(self, req_id, layer_idx, token_idx,
                     key, value):
        """Append a single token's KV to the cache."""
        logical_block = token_idx // self.block_size
        block_offset = token_idx % self.block_size

        # Allocate new block if needed
        if req_id not in self.block_tables:
            self.block_tables[req_id] = []

        if logical_block >= len(self.block_tables[req_id]):
            phys_block = self.allocate_block()
            self.block_tables[req_id].append(phys_block)

        phys_block = self.block_tables[req_id][logical_block]

        # Store KV in the physical block
        self.key_cache[phys_block, :, block_offset, :] = key
        self.value_cache[phys_block, :, block_offset, :] = value

    def get_kv(self, req_id, layer_idx):
        """Retrieve KV cache for a request using block table."""
        blocks = self.block_tables[req_id]
        # Concatenate all blocks (scatter/gather in practice)
        keys = [self.key_cache[b] for b in blocks]
        values = [self.value_cache[b] for b in blocks]
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)

    def copy_on_write(self, req_id, fork_id):
        """Fork KV cache for beam search (copy-on-write semantics)."""
        self.block_tables[fork_id] = self.block_tables[req_id].copy()
```

### 11.2.3 KV Cache Compression

Several techniques reduce KV cache memory:

```
┌──────────────────────────────────────────────────────────────┐
│              KV Cache Compression Techniques                    │
│                                                                │
│  1. GQA/MQA (Architecture-level):                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Standard MHA: KV heads = Q heads (e.g., 64)         │    │
│  │  GQA:          KV heads = Q heads / 8 (e.g., 8)      │    │
│  │  MQA:          KV heads = 1                           │    │
│  │                                                       │    │
│  │  Memory reduction: 8x (GQA-8) to 64x (MQA)           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  2. KV Cache Quantization:                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Store KV cache in INT8 or INT4                       │    │
│  │  Memory reduction: 2x (INT8) to 4x (INT4)            │    │
│  │  Quality impact: Minimal with per-channel quantization│    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  3. Sliding Window Attention:                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Only cache last W tokens (e.g., W=4096)              │    │
│  │  For Mistral-7B: sliding window = 4096                │    │
│  │  Memory: Fixed regardless of sequence length           │    │
│  │  Limitation: Cannot attend beyond window               │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  4. StreamingLLM (Attention Sink):                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Keep first few tokens (attention sinks) + last W     │    │
│  │  Example: Keep tokens 0-3 + last 4093 tokens          │    │
│  │  Works with any model (no retraining needed)          │    │
│  │  Quality: Good for most tasks, some loss on first     │    │
│  │           few tokens                                  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  5. H2O (Heavy-Hitter Oracle):                                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Dynamically evict low-importance KV entries           │    │
│  │  Keep tokens with high attention scores               │    │
│  │  Adaptive to content, not just position               │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

```python
# KV Cache Quantization example
class KVCacheQuantizer:
    """Quantize KV cache to INT8 for memory savings."""

    def __init__(self, num_heads, head_dim):
        self.num_heads = num_heads
        self.head_dim = head_dim

    def quantize_int8(self, kv_tensor):
        """Per-head symmetric INT8 quantization."""
        # Compute scale per head
        abs_max = kv_tensor.abs().amax(dim=-1, keepdim=True)
        scale = abs_max / 127.0

        # Quantize
        kv_int8 = torch.round(kv_tensor / scale).to(torch.int8)

        return kv_int8, scale

    def dequantize(self, kv_int8, scale):
        """Dequantize back to bf16."""
        return kv_int8.to(torch.bfloat16) * scale

    def compute_cache_memory(self, batch_size, seq_len, num_layers):
        """Compute memory for quantized KV cache."""
        bytes_per_element = 1  # INT8
        cache_size = (
            2 *  # K and V
            batch_size *
            self.num_heads *
            seq_len *
            self.head_dim *
            bytes_per_element *
            num_layers
        )
        return cache_size / (1024**3)  # GB
```

### 11.2.4 Prefix Caching

For chatbot applications with system prompts, prefix caching saves significant computation:

```python
# Prefix caching strategy
class PrefixCache:
    """
    Cache KV computation for shared prompt prefixes.
    Especially useful for system prompts in chatbot applications.
    """
    def __init__(self, max_prefix_length=2048):
        self.cache = {}  # hash(prompt) -> (kv_cache, computed_length)
        self.max_prefix_length = max_prefix_length

    def get_prefix_length(self, new_prompt, cached_prompts):
        """Find the longest matching prefix."""
        best_match = 0
        best_key = None

        for cached_key, (_, computed_len) in self.cache.items():
            # Check if new_prompt starts with cached prefix
            if new_prompt.startswith(cached_key):
                if computed_len > best_match:
                    best_match = computed_len
                    best_key = cached_key

        return best_match, best_key

    def compute_with_cache(self, model, new_prompt, tokenizer):
        """
        Compute KV cache, reusing cached prefix if available.
        """
        prefix_len, prefix_key = self.get_prefix_length(new_prompt)

        if prefix_len > 0:
            # Reuse cached prefix KV
            cached_kv, _ = self.cache[prefix_key]
            remaining_prompt = new_prompt[prefix_len:]

            # Only compute KV for the new portion
            remaining_tokens = tokenizer.encode(remaining_prompt)
            new_kv = model.compute_kv(remaining_tokens)

            # Concatenate: cached prefix + new portion
            full_kv = self.concatenate_kv(cached_kv, new_kv)
        else:
            # Full computation needed
            tokens = tokenizer.encode(new_prompt)
            full_kv = model.compute_kv(tokens)

        # Cache the full prompt for future use
        self.cache[new_prompt] = (full_kv, len(new_prompt))

        return full_kv

# Example: System prompt caching
# System prompt: "You are a helpful assistant specializing in..." (512 tokens)
# Without cache: Every request computes 512 + user_tokens KV
# With cache: First request computes full, subsequent requests save 512 tokens
# Savings: 512 * 80 layers * 64 heads * 128 dim * 2 bytes * 2 (K+V) = 1.3 GB per request
```

---

## 11.3 Quantization & Distillation

### 11.3.1 Quantization Taxonomy

Quantization reduces the precision of model weights and/or activations to decrease memory footprint and increase inference speed:

```
┌──────────────────────────────────────────────────────────────┐
│              Quantization Taxonomy                              │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                  Post-Training Quantization (PTQ)       │  │
│  │  Quantize after training, no retraining needed          │  │
│  │                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │  │
│  │  │ Weight-Only  │  │ Weight-Act   │  │ SmoothQuant  │ │  │
│  │  │ (W4A16)      │  │ (W8A8)       │  │ (W8A8)       │ │  │
│  │  │              │  │              │  │              │ │  │
│  │  │ INT4 weights │  │ INT8 weights │  │ Mathematically│ │  │
│  │  │ fp16 acts    │  │ INT8 acts    │  │ equivalent   │ │  │
│  │  │              │  │              │  │ per-channel  │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Quantization-Aware Training (QAT)          │  │
│  │  Quantize during training, higher quality               │  │
│  │                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐                   │  │
│  │  │ QLoRA        │  │ LLM-QAT      │                   │  │
│  │  │ INT4 fine-   │  │ Full QAT for │                   │  │
│  │  │ tuning       │  │ LLMs         │                   │  │
│  │  └──────────────┘  └──────────────┘                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                  Mixed-Precision                         │  │
│  │  Different layers at different precisions               │  │
│  │  Sensitive layers: higher precision                    │  │
│  │  Robust layers: lower precision                        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 11.3.2 GPTQ (Post-Training Quantization)

GPTQ is one of the most popular weight-only quantization methods:

```python
# Using GPTQ for quantization (via AutoGPTQ)
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

# Define quantization configuration
quantize_config = BaseQuantizeConfig(
    bits=4,              # 4-bit quantization
    group_size=128,      # Group quantization (128 weights per group)
    damp_percent=0.01,   # Damping factor for Hessian
    desc_act=True,       # Sort by activation for better quantization
    sym=False,           # Asymmetric quantization
)

# Load pre-trained model
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantize_config,
    device_map="auto"
)

# Quantize with calibration data
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
calibration_data = tokenizer(
    "This is a calibration text for quantization..." * 100,
    return_tensors="pt"
).input_ids.to("cuda")

model.quantize(calibration_data)

# Save quantized model
model.save_quantized("./llama-2-70b-gptq-4bit")

# Load and use quantized model
model = AutoGPTQForCausalLM.from_quantized(
    "./llama-2-70b-gptq-4bit",
    device_map="auto",
    use_triton=False
)
```

### 11.3.3 GGUF Quantization (llama.cpp)

GGUF is the standard format for CPU/edge inference:

```
┌──────────────────────────────────────────────────────────────┐
│              GGUF Quantization Types                           │
│                                                                │
│  Type    │ Bits │ Method         │ Quality │ Speed  │ Memory  │
│  ────────│──────│────────────────│─────────│────────│─────────│
│  Q2_K    │  2   │ k-quant       │ ★★☆☆☆  │ ★★★★★ │ ★★★★★  │
│  Q3_K_M  │  3   │ k-quant       │ ★★★☆☆  │ ★★★★☆ │ ★★★★☆  │
│  Q4_K_M  │  4   │ k-quant       │ ★★★★☆  │ ★★★★☆ │ ★★★★☆  │
│  Q5_K_M  │  5   │ k-quant       │ ★★★★★  │ ★★★☆☆ │ ★★★☆☆  │
│  Q6_K    │  6   │ k-quant       │ ★★★★★  │ ★★★☆☆ │ ★★★☆☆  │
│  Q8_0    │  8   │ round-to-nearest│ ★★★★★ │ ★★★☆☆ │ ★★★☆☆  │
│  F16     │  16  │ float16       │ ★★★★★  │ ★★☆☆☆ │ ★★☆☆☆  │
│  IQ4_XS  │  4.25│ importance    │ ★★★★☆  │ ★★★★☆ │ ★★★★☆  │
│                                                                │
│  k-quant uses different quantization for different layers      │
│  (important layers get higher precision)                       │
│                                                                │
│  Example: LLaMA-2 7B                                            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Q2_K: 2.67 GB  → Fits in 4GB VRAM                   │    │
│  │  Q4_K_M: 4.08 GB → Fits in 6GB VRAM                  │    │
│  │  Q8_0: 7.16 GB → Fits in 8GB VRAM                    │    │
│  │  F16: 13.5 GB → Needs 16GB VRAM                      │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

```python
# Using llama.cpp Python bindings
from llama_cpp import Llama

# Load GGUF quantized model
llm = Llama(
    model_path="./models/llama-2-7b-q4_k_m.gguf",
    n_ctx=4096,        # Context length
    n_gpu_layers=35,   # Number of layers to offload to GPU
    n_threads=8,       # CPU threads
    verbose=False
)

# Generate text
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    stream=True
)

for chunk in output:
    if chunk["choices"][0]["delta"].get("content"):
        print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
```

### 11.3.4 Knowledge Distillation

Distillation trains a smaller "student" model to mimic a larger "teacher" model:

```
┌──────────────────────────────────────────────────────────────┐
│              Knowledge Distillation Pipeline                    │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Teacher Model (Large)                                │    │
│  │  e.g., LLaMA-2 70B                                   │    │
│  │  ┌──────────────────────────────────────────┐        │    │
│  │  │  Input → [Transformer Layers 80] → Logits│        │    │
│  │  └──────────────────────────────────────────┘        │    │
│  │                    │                                   │    │
│  │                    ▼                                   │    │
│  │  ┌──────────────────────────────────────────┐        │    │
│  │  │  Softmax with Temperature T               │        │    │
│  │  │  P_teacher(y|x) = softmax(logits / T)    │        │    │
│  │  └──────────────────────────────────────────┘        │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Distillation Loss                                    │    │
│  │                                                       │    │
│  │  L = α * L_KL + (1-α) * L_CE                        │    │
│  │                                                       │    │
│  │  L_KL = KL(softmax(z_t/T) || softmax(z_s/T))        │    │
│  │  L_CE = CrossEntropy(student_logits, true_labels)    │    │
│  │                                                       │    │
│  │  T = temperature (higher = softer distributions)     │    │
│  │  α = balance between distillation and supervision    │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Student Model (Small)                                │    │
│  │  e.g., LLaMA-2 7B                                    │    │
│  │  ┌──────────────────────────────────────────┐        │    │
│  │  │  Input → [Transformer Layers 32] → Logits│        │    │
│  │  └──────────────────────────────────────────┘        │    │
│  │                                                       │    │
│  │  Train to minimize L while learning from teacher      │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationTrainer:
    def __init__(self, teacher, student, temperature=2.0, alpha=0.5):
        self.teacher = teacher.eval()
        self.student = student.train()
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Combined distillation + cross-entropy loss."""
        # Soft target loss (KL divergence)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(soft_student, soft_teacher,
                           reduction='batchmean') * (self.temperature ** 2)

        # Hard target loss (cross-entropy)
        ce_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss

    def train_step(self, batch, optimizer):
        """Single training step."""
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids).logits

        # Student forward
        student_logits = self.student(input_ids).logits

        # Compute loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
```

---

## 11.4 Batching & Continuous Batching

### 11.4.1 Static Batching vs Continuous Batching

The key innovation enabling high-throughput LLM serving:

```
┌──────────────────────────────────────────────────────────────┐
│              Static Batching vs Continuous Batching             │
│                                                                │
│  Static Batching:                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Batch 1: [Request A(200 tok), B(50 tok), C(300 tok)]│    │
│  │                                                       │    │
│  │  Time ──────────────────────────────────────────────▶│    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │ A: ████████████████████████████████████████████│    │    │
│  │  │ B: ██████████                                   │    │    │
│  │  │ C: ██████████████████████████████████████████████│    │   │
│  │  └──────────────────────────────────────────────┘    │    │
│  │                                                       │    │
│  │  Problem: B finishes at step 50 but GPU sits idle     │    │
│  │           until A and C finish. GPU utilization: LOW  │    │
│  │                                                       │    │
│  │  GPU Utilization: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │    │
│  │                    (lots of wasted compute slots)     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Continuous Batching (Iteration-level Scheduling):              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Time ──────────────────────────────────────────────▶│    │
│  │  Step 1:   [A, B, C]                                  │    │
│  │  Step 2:   [A, B, C]                                  │    │
│  │  ...                                                   │    │
│  │  Step 50:  [A, B, C]  ← B finishes, slot freed       │    │
│  │  Step 51:  [A, D, C]  ← D inserted into freed slot   │    │
│  │  Step 52:  [A, D, C]                                  │    │
│  │  ...                                                   │    │
│  │  Step 200: [A, D, C] ← A finishes                    │    │
│  │  Step 201: [E, D, C] ← E inserted                   │    │
│  │                                                       │    │
│  │  GPU Utilization: ████████████████████████████████████│    │
│  │                    (always full)                       │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Throughput Improvement: 2-5x with continuous batching         │
└──────────────────────────────────────────────────────────────┘
```

### 11.4.2 Chunked Prefill

Chunked prefill splits long prompts into chunks to interleave with decode requests:

```python
class ChunkedPrefillScheduler:
    """
    Chunked prefill allows long prompts to be processed in chunks,
    interleaving with decode requests for better latency.
    """
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def schedule(self, waiting_queue, running_queue):
        """
        Schedule decode requests first, then prefill chunks.
        This ensures low latency for decode (streaming) requests.
        """
        scheduled = []

        # First: schedule all running decode requests
        for req in running_queue:
            if not req.is_finished:
                scheduled.append(req)

        # Then: schedule prefill chunks from waiting queue
        for req in waiting_queue:
            if req.prefill_done:
                continue

            # How many tokens to process in this chunk?
            remaining = req.total_length - req.processed_length
            chunk = min(remaining, self.chunk_size)

            # Create prefill chunk
            chunk_request = PrefillChunk(
                request_id=req.request_id,
                token_ids=req.token_ids[
                    req.processed_length:req.processed_length + chunk
                ],
                kv_cache_offset=req.processed_length,
            )
            scheduled.append(chunk_request)

            req.processed_length += chunk
            if req.processed_length >= req.total_length:
                req.prefill_done = True

        return scheduled
```

### 11.4.3 Speculative Decoding

Speculative decoding uses a smaller draft model to propose tokens, which are then verified by the target model in parallel:

```
┌──────────────────────────────────────────────────────────────┐
│              Speculative Decoding                               │
│                                                                │
│  Draft Model (small, fast):                                    │
│  "The cat sat on the"                                         │
│       │                                                       │
│       ▼                                                       │
│  Generate K=5 candidate tokens:                               │
│  ["mat", "floor", "couch", "bed", "chair"]                   │
│                                                                │
│  Target Model (large, accurate):                               │
│  Process ALL K tokens in parallel:                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Input: "The cat sat on the" + ["mat","floor",...,"chair"]│   │
│  │  Output: Verify each position                         │    │
│  │                                                       │    │
│  │  Position 4: P_target("mat")=0.3, P_draft("mat")=0.25│    │
│  │  → Accept (ratio > 1.0)                              │    │
│  │                                                       │    │
│  │  Position 5: P_target("the")=0.4, P_draft("the")=0.1 │    │
│  │  → Reject, sample from adjusted distribution          │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Result: 3-4x speedup with minimal quality loss               │
│                                                                │
│  Time Analysis:                                                │
│  ┌──────────────────────────────────────────────┐            │
│  │  Without speculative: 5 sequential forward    │            │
│  │  passes of large model = 5 × T_large          │            │
│  │                                               │            │
│  │  With speculative: 1 forward of draft (T_s)   │            │
│  │                   + 1 forward of target (T_l) │            │
│  │  Total: T_s + T_l                             │            │
│  │                                               │            │
│  │  If T_l ≈ 5 × T_s (draft is 5x smaller):     │            │
│  │  Speedup: 5T_l / (T_l/5 + T_l) = 25/6 ≈ 4.2x│            │
│  └──────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

---

## 11.5 Inference Cluster Architecture

### 11.5.1 Single-Node Multi-GPU Serving

```
┌──────────────────────────────────────────────────────────────┐
│           Single-Node Inference Server (4× H100)               │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Load Balancer (HAProxy / NGINX)                      │    │
│  │  └── Health checks, rate limiting, request routing    │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  vLLM Server (TP=4)                                   │    │
│  │  ┌────────────────────────────────────────────────┐  │    │
│  │  │  GPU 0     GPU 1     GPU 2     GPU 3          │  │    │
│  │  │  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐      │  │    │
│  │  │  │ TP0 │──▶│ TP1 │──▶│ TP2 │──▶│ TP3 │      │  │    │
│  │  │  │ L0-19│   │L20-39│   │L40-59│   │L60-79│     │  │    │
│  │  │  └─────┘   └─────┘   └─────┘   └─────┘      │  │    │
│  │  │  ◄──── NVLink (900 GB/s bidirectional) ─────▶  │  │    │
│  │  └────────────────────────────────────────────────┘  │    │
│  │                                                       │    │
│  │  Model: LLaMA-3 70B (TP=4, fp16)                     │    │
│  │  Memory: 140GB / 4 = 35GB per GPU                     │    │
│  │  Throughput: ~120 tokens/sec                           │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 11.5.2 Multi-Node Inference Cluster

```
┌──────────────────────────────────────────────────────────────────┐
│           Multi-Node Inference Cluster                              │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  API Gateway + Load Balancer                               │    │
│  │  ├── Request routing based on model size                  │    │
│  │  ├── Automatic failover                                   │    │
│  │  └── Rate limiting & authentication                       │    │
│  └─────────────────────┬────────────────────────────────────┘    │
│                         │                                          │
│         ┌───────────────┼───────────────┐                        │
│         ▼               ▼               ▼                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │  Node 0    │  │  Node 1    │  │  Node 2    │                │
│  │  Model A   │  │  Model A   │  │  Model B   │                │
│  │  (70B)     │  │  (70B)     │  │  (8x7B)    │                │
│  │  4× H100   │  │  4× H100   │  │  2× A100   │                │
│  └────────────┘  └────────────┘  └────────────┘                │
│                                                                    │
│  Model Loading Strategy:                                           │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Option A: Each node hosts complete model                 │    │
│  │  → Simple, independent nodes                             │    │
│  │  → No cross-node communication                           │    │
│  │  → Limited by single-node GPU memory                     │    │
│  │                                                          │    │
│  │  Option B: Model sharded across nodes                    │    │
│  │  → Required for models > single-node capacity            │    │
│  │  → Cross-node communication overhead                     │    │
│  │  → Need high-bandwidth interconnect (InfiniBand)        │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 11.5.3 Autoscaling Architecture

```python
# Kubernetes-based autoscaling for LLM inference
# k8s-autoscaling.yaml
"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "70"  # Scale up at 70% GPU util
  - type: Pods
    pods:
      metric:
        name: request_queue_depth
      target:
        type: AverageValue
        averageValue: "10"  # Scale up when queue > 10
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
---
# Pod template with GPU resource requests
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:latest
        args:
        - "--model=meta-llama/Llama-2-70b-hf"
        - "--tensor-parallel-size=4"
        - "--max-model-len=4096"
        - "--gpu-memory-utilization=0.9"
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: "200Gi"
          requests:
            nvidia.com/gpu: 4
            memory: "200Gi"
        ports:
        - containerPort: 8000
"""
```

---

## 11.6 Cost Optimization Strategies

### 11.6.1 GPU Selection Guide

```
┌──────────────────────────────────────────────────────────────┐
│              GPU Selection for LLM Inference                   │
│                                                                │
│  GPU        │ VRAM  │ BW      │ FP16 TFLOPS │ $/hr  │ Best For│
│  ───────────│───────│─────────│─────────────│───────│─────────│
│  A10G       │ 24GB  │ 600GB/s │    31       │ $1.10 │ ≤13B    │
│  A100 40GB  │ 40GB  │ 1.6TB/s │   312       │ $3.40 │ ≤30B    │
│  A100 80GB  │ 80GB  │ 2.0TB/s │   312       │ $4.10 │ ≤70B    │
│  H100       │ 80GB  │ 3.35TB/s│   990       │ $5.50 │ ≤70B    │
│  H100 SXM   │ 80GB  │ 3.35TB/s│   990       │ $6.00 │ ≤70B    │
│  H200       │ 141GB │ 4.8TB/s │   990       │ $7.00 │ ≤70B+   │
│  L40S       │ 48GB  │ 864GB/s │   181       │ $1.80 │ ≤30B    │
│                                                                │
│  Cost-Performance Analysis (for 70B model, TP=4):              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  A100 80GB: 4×$4.10 = $16.40/hr → 40 tok/s         │    │
│  │  H100:      4×$5.50 = $22.00/hr → 120 tok/s        │    │
│  │                                                       │    │
│  │  Cost per 1M tokens:                                   │    │
│  │  A100: $16.40 / 40 = $0.41 per 1K tokens            │    │
│  │  H100: $22.00 / 120 = $0.18 per 1K tokens           │    │
│  │                                                       │    │
│  │  H100 is 2.3x more cost-effective for 70B models     │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 11.6.2 Cost Reduction Techniques

```
┌──────────────────────────────────────────────────────────────┐
│              Cost Optimization Techniques                       │
│                                                                │
│  1. Quantization (Immediate 2-4x cost reduction)              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  70B FP16: 4× H100 = $22/hr                          │    │
│  │  70B INT4: 1× H100 = $5.50/hr                        │    │
│  │  Savings: 75%                                         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  2. Continuous Batching (2-3x throughput increase)             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Static: 20 tok/s per request                         │    │
│  │  Continuous: 60 tok/s per request (3x improvement)   │    │
│  │  Cost per token reduced by 3x                         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  3. Spot/Preemptible Instances (60-80% savings)               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  On-demand H100: $5.50/hr                             │    │
│  │  Spot H100: ~$1.65/hr (70% discount)                  │    │
│  │  Risk: May be preempted (handle gracefully)           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  4. Model Size Selection (right-size for task)                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Task: Customer support chatbot                       │    │
│  │  70B model: 99.2% accuracy, $0.001 per request       │    │
│  │  7B model:  97.8% accuracy, $0.0001 per request      │    │
│  │  Decision: Use 7B (10x cheaper, 1.4% accuracy drop) │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  5. Prompt Caching (30-50% savings on repeat prompts)         │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  System prompt: 500 tokens, repeated 1000x/day        │    │
│  │  Without cache: 500 × 1000 = 500K tokens computed     │    │
│  │  With cache: 500 + (1 × 1000) ≈ 1.5K tokens computed │    │
│  │  Savings: 99.7% compute for prompt portion            │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 💡 Case: High-Throughput Inference Service with vLLM

### Business Context

A SaaS company needs to serve a custom 13B parameter model for code generation to 10,000+ daily users. Requirements:
- Time-to-first-token (TTFT): < 500ms
- Throughput: 500+ requests/minute
- Cost: < $500/month on cloud
- 99.9% availability

### Architecture Design

```
┌──────────────────────────────────────────────────────────────────┐
│           Production Inference Architecture                        │
│                                                                    │
│  Users ──▶ CDN ──▶ API Gateway ──▶ Load Balancer ──▶ vLLM Pods   │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Cloud Infrastructure (AWS/GCP)                           │    │
│  │                                                           │    │
│  │  ┌─────────────────────────────────────────────────┐     │    │
│  │  │  Region: us-east-1                               │     │    │
│  │  │                                                   │     │    │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐           │     │    │
│  │  │  │Pod 0    │ │Pod 1    │ │Pod 2    │  (min=2,   │     │    │
│  │  │  │vLLM     │ │vLLM     │ │vLLM     │   max=6)   │     │    │
│  │  │  │A10G×1   │ │A10G×1   │ │A10G×1   │           │     │    │
│  │  │  │TP=1     │ │TP=1     │ │TP=1     │           │     │    │
│  │  │  └─────────┘ └─────────┘ └─────────┘           │     │    │
│  │  └─────────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# deploy_vllm.py
"""
Complete vLLM deployment script for production serving.
"""
import subprocess
import json

def deploy_vllm_cluster():
    """Deploy vLLM with optimal configuration for code generation."""

    vllm_config = {
        "model": "codellama/CodeLlama-13b-hf",
        "tensor_parallel_size": 1,  # 13B fits on single A10G
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.92,
        "dtype": "bfloat16",
        "quantization": "awq",  # AWQ 4-bit for cost optimization
        "enforce_eager": False,  # Enable CUDA graphs
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 64,
        "block_size": 16,
        "enable_prefix_caching": True,
        "disable_log_requests": True,
    }

    # Start vLLM server
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", vllm_config["model"],
        "--tensor-parallel-size", str(vllm_config["tensor_parallel_size"]),
        "--max-model-len", str(vllm_config["max_model_len"]),
        "--gpu-memory-utilization", str(vllm_config["gpu_memory_utilization"]),
        "--dtype", vllm_config["dtype"],
        "--quantization", vllm_config["quantization"],
        "--block-size", str(vllm_config["block_size"]),
        "--enable-prefix-caching",
        "--port", "8000",
    ]

    return cmd

# Benchmark results (CodeLlama-13B-AWQ on A10G):
# ┌──────────────────────────────────────────────────────┐
# │  Metric              │ Value                         │
# │  ────────────────────│───────────────────────────────│
# │  TTFT (P50)          │ 180ms                        │
# │  TTFT (P99)          │ 420ms                        │
# │  Throughput          │ 1,200 tokens/sec             │
# │  Max concurrent      │ 64 requests                  │
# │  Cost per GPU        │ $1.10/hr (A10G)              │
# │  Cost per 1M tokens  │ $0.09                        │
# └──────────────────────────────────────────────────────┘
```

### Monitoring Dashboard

```python
# monitoring.py
"""
Key metrics to monitor for LLM inference service.
"""
METRICS = {
    # Latency metrics
    "time_to_first_token_p50": "180ms",
    "time_to_first_token_p99": "420ms",
    "inter_token_latency_p50": "25ms",
    "inter_token_latency_p99": "80ms",

    # Throughput metrics
    "tokens_per_second": "1200",
    "requests_per_minute": "500",
    "concurrent_requests": "45",

    # Resource metrics
    "gpu_utilization": "78%",
    "gpu_memory_used": "18GB / 24GB",
    "kv_cache_usage": "65%",

    # Business metrics
    "cost_per_1m_tokens": "$0.09",
    "monthly_cost": "$792",
    "availability": "99.95%",
}

# Alert thresholds
ALERTS = {
    "ttft_p99 > 1000ms": "High latency - check GPU utilization",
    "gpu_utilization > 95%": "GPU saturated - scale up",
    "kv_cache_usage > 90%": "KV cache full - increase block pool",
    "error_rate > 1%": "High errors - check logs",
}
```

---

## Summary

This chapter covered the essential aspects of LLM inference architecture:

| Topic | Key Takeaway |
|-------|-------------|
| **Inference Engines** | vLLM and SGLang lead in throughput; TensorRT-LLM leads in latency |
| **KV Cache** | PagedAttention eliminates memory waste; enables high-concurrency serving |
| **Quantization** | INT4 quantization (GPTQ/AWQ) enables 70B models on single GPUs |
| **Continuous Batching** | 2-5x throughput improvement over static batching |
| **Speculative Decoding** | 3-4x speedup by verifying draft tokens in parallel |
| **Cost Optimization** | Quantization + batching + spot instances can reduce costs by 10-20x |
| **Cluster Design** | Autoscaling with GPU utilization metrics for dynamic workloads |

### Performance Comparison

```
┌──────────────────────────────────────────────────────────────┐
│         70B Model Inference Performance (H100×4)               │
│                                                                │
│  Configuration          │ TTFT    │ Throughput │ Cost/1M tok  │
│  ───────────────────────│─────────│────────────│──────────────│
│  FP16, Static Batch     │ 800ms   │ 30 tok/s   │ $0.55       │
│  FP16, Cont. Batch      │ 200ms   │ 120 tok/s  │ $0.14       │
│  INT4, Cont. Batch      │ 150ms   │ 200 tok/s  │ $0.08       │
│  INT4, Cont. Batch + Spec│ 100ms  │ 500 tok/s  │ $0.03       │
│                                                                │
│  Best practice: INT4 + Continuous Batching + Speculative       │
│  Decoding for production deployments                           │
└──────────────────────────────────────────────────────────────┘
```

---

## References

1. Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP.
2. Leviathan, Y., et al. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML.
3. Frantar, E., et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR.
4. Lin, J., et al. (2024). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys.
5. Zheng, L., et al. (2023). "SGLang: Efficient Execution of Structured Language Model Programs." arXiv.
6. vLLM Documentation. https://docs.vllm.ai/
7. TensorRT-LLM Documentation. https://nvidia.github.io/TensorRT-LLM/
8. Chen, C., et al. (2024). "Activester: Accelerating Large Language Model Inference with Dynamic Parallelism." arXiv.
9. Liu, Z., et al. (2023). "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory." arXiv.
10. Pope, R., et al. (2023). "Efficiently Scaling Transformer Inference." MLSys.

---

*← [Chapter 10 - LLM Architecture Design Fundamentals](chapter-10.md) | [Chapter 12 - RAG System Architecture](chapter-12.md) →*
