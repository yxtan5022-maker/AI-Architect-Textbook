# Chapter 21: Performance Engineering for AI Systems

**Reader Level:** 🔴 Advanced | **Pages:** 28 | **Code:** Python, C++, CUDA

---

## 21.1 AI Performance Fundamentals

### 21.1.1 The Performance Triangle

AI system performance rests on three pillars: latency, throughput, and cost. Optimizing one often impacts the others. The goal is finding the right balance for your use case.

```
                    Latency
                   (p50/p99)
                      /\\
                     /  \\
                    /    \\
                   / AIM  \\
                  / Optimal\\
                 /  Balance \\
                /____________\\
         Throughput          Cost
        (tokens/sec)      ($/token)
```

### 21.1.2 Key Performance Metrics

| Metric | Definition | Target (Production) |
|--------|-----------|---------------------|
| Time to First Token (TTFT) | Time until first output token | 200-500ms |
| Inter-Token Latency | Time between subsequent tokens | 20-50ms |
| Throughput | Tokens generated per second | 100-1000 tok/s |
| Requests per Second | Concurrent request handling | 100-10,000 RPS |
| GPU Utilization | Percentage of GPU compute used | 70-90% |
| Memory Bandwidth | HBM throughput utilization | 60-80% |
| Cost per 1M Tokens | Total cost including compute | $0.10-$2.00 |

### 21.1.3 Bottleneck Analysis Framework

```
┌────────────────────────────────────────────────────────────┐
│                 BOTTLENECK ANALYSIS                         │
│                                                             │
│  Input Processing                                           │
│  ├── Tokenization speed                                    │
│  ├── Prompt caching hit rate                               │
│  └── Batch formation efficiency                            │
│                                                             │
│  Compute                                                    │
│  ├── GPU utilization (SM occupancy)                        │
│  ├── Memory bandwidth saturation                           │
│  └── Tensor core utilization                               │
│                                                             │
│  Memory                                                     │
│  ├── KV cache size and eviction                            │
│  ├── Activation memory                                     │
│  └── Model weight memory                                   │
│                                                             │
│  Output                                                     │
│  ├── Token generation speed                                │
│  ├── Output buffering                                      │
│  └── Network transfer                                      │
└────────────────────────────────────────────────────────────┘
```


## 21.2 Inference Optimization

### 21.2.1 Quantization Strategies

Quantization reduces model precision from FP32/FP16 to INT8/INT4, dramatically reducing memory and compute requirements.

```python
# quantization_example.py
import torch
from torch.quantization import quantize_dynamic

def quantize_model(model, dtype=torch.qint8):
    quantized = quantize_dynamic(
        model,
        {torch.nn.Linear},  # Layers to quantize
        dtype=dtype
    )
    return quantized

def measure_quantization_impact(model, input_tensor):
    # Original model
    original_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    
    # Quantized model
    quantized = quantize_model(model)
    quantized_size = sum(
        p.numel() * p.element_size() for p in quantized.parameters()
    )
    
    compression_ratio = original_size / quantized_size
    print(f"Original: {original_size / 1e6:.1f} MB")
    print(f"Quantized: {quantized_size / 1e6:.1f} MB")
    print(f"Compression: {compression_ratio:.1f}x")
    
    return quantized
```

### 21.2.2 KV Cache Optimization

The KV cache stores key-value pairs from previous tokens to avoid recomputation. Managing it efficiently is critical for long-context inference.

```
┌────────────────────────────────────────────────────────────┐
│                    KV CACHE MANAGEMENT                      │
│                                                             │
│  Full Cache (No Optimization)                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [Token 1 KV] [Token 2 KV] ... [Token N KV]         │   │
│  │ Memory: O(N * d_model * n_layers * 2)               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Paged Attention (vLLM-style)                               │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │
│  │Page 0│ │Page 1│ │Page 2│ │Page 3│ │Page 4│            │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘            │
│  Benefits: No fragmentation, dynamic allocation             │
│                                                             │
│  Sliding Window (Mistral-style)                             │
│  ┌──────────────────────────────────────┐                  │
│  │  [Window: Last W tokens only]        │                  │
│  │  Memory: O(W * d_model * n_layers)   │                  │
│  └──────────────────────────────────────┘                  │
└────────────────────────────────────────────────────────────┘
```

### 21.2.3 Continuous Batching

```python
# continuous_batching.py
from dataclasses import dataclass
from typing import List
import heapq

@dataclass
class InferenceRequest:
    request_id: str
    prompt_tokens: int
    max_output_tokens: int
    priority: int = 0

class ContinuousBatchScheduler:
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.batch: List[InferenceRequest] = []
        self.queue: List[InferenceRequest] = []
    
    def add_request(self, req: InferenceRequest):
        self.queue.append(req)
        self.queue.sort(key=lambda r: -r.priority)
    
    def form_batch(self) -> List[InferenceRequest]:
        batch = []
        remaining = []
        for req in self.queue:
            if len(batch) < self.max_batch_size:
                batch.append(req)
            else:
                remaining.append(req)
        self.queue = remaining
        return batch
    
    def dynamic_batch_adjustment(self, current_load: float):
        if current_load > 0.8:
            self.max_batch_size = min(64, self.max_batch_size + 4)
        elif current_load < 0.3:
            self.max_batch_size = max(8, self.max_batch_size - 4)
```


## 21.3 Distributed Inference

### 21.3.1 Tensor Parallelism

Split model tensors across multiple GPUs for parallel computation.

```
┌────────────────────────────────────────────────────────────┐
│                  TENSOR PARALLELISM                         │
│                                                             │
│  Layer with weight matrix W:                                │
│                                                             │
│  ┌─────────────────────┐                                    │
│  │         W           │                                    │
│  │    (4096 x 4096)    │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│     ┌───────┴───────┐                                       │
│     │               │                                       │
│     v               v                                       │
│  ┌──────┐       ┌──────┐                                    │
│  │ GPU 0│       │ GPU 1│                                    │
│  │ W[:,0:2048]  │ W[:,2048:4096]                            │
│  └──────┘       └──────┘                                    │
│     │               │                                       │
│     v               v                                       │
│  ┌──────┐       ┌──────┐                                    │
│  │Output│       │Output│                                    │
│  │ Part │       │ Part │                                    │
│  └──┬───┘       └──┬───┘                                    │
│     │              │                                        │
│     └──────┬───────┘                                        │
│            v                                                │
│     ┌────────────┐                                          │
│     │  AllReduce │                                          │
│     │  Combine   │                                          │
│     └────────────┘                                          │
└────────────────────────────────────────────────────────────┘
```

### 21.3.2 Pipeline Parallelism

```python
# pipeline_parallelism.py
from typing import List
from dataclasses import dataclass

@dataclass
class PipelineStage:
    stage_id: int
    layers: List[str]
    device: str

class PipelineScheduler:
    def __init__(self, stages: List[PipelineStage], micro_batch_size: int = 4):
        self.stages = stages
        self.micro_batch_size = micro_batch_size
    
    def schedule_forward(self, batch):
        pipeline = []
        for i, stage in enumerate(self.stages):
            pipeline.append({
                "stage": stage.stage_id,
                "device": stage.device,
                "batch_idx": i,
                "forward": True,
            })
        return pipeline
    
    def schedule_backward(self, pipeline):
        return list(reversed(pipeline))
```

### 21.3.3 Expert Parallelism (MoE)

```python
# moe_expert_parallelism.py
import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_dim, num_experts)
    
    def forward(self, x):
        gate_scores = torch.softmax(self.gate(x), dim=-1)
        top_k = torch.topk(gate_scores, k=2, dim=-1)
        
        output = torch.zeros_like(x)
        for k in range(2):
            expert_idx = top_k.indices[:, :, k]
            weight = top_k.values[:, :, k].unsqueeze(-1)
            for i in range(self.num_experts):
                mask = (expert_idx == i)
                if mask.any():
                    expert_input = x[mask]
                    output[mask] += weight[mask] * self.experts[i](expert_input)
        return output
```


## 21.4 Caching and Memory Management

### 21.4.1 Prompt Caching Strategies

```python
# prompt_cache.py
import hashlib
from typing import Dict, Optional
from collections import OrderedDict

class PromptCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.stats = {"hits": 0, "misses": 0}
    
    def get_prefix_match(self, prompt: str) -> Optional[str]:
        for cached_prompt in reversed(list(self.cache.keys())):
            if prompt.startswith(cached_prompt):
                self.stats["hits"] += 1
                self.cache.move_to_end(cached_prompt)
                return cached_prompt
        self.stats["misses"] += 1
        return None
    
    def store(self, prompt: str, kv_cache):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[prompt] = kv_cache
    
    @property
    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0
```

### 21.4.2 Memory Pool Management

```python
# memory_pool.py
import torch
from typing import Dict, List

class CUDAMemoryPool:
    def __init__(self, device: torch.device):
        self.device = device
        self.allocated: Dict[int, torch.Tensor] = {}
        self.free_blocks: List[torch.Tensor] = []
    
    def allocate(self, shape, dtype=torch.float16) -> torch.Tensor:
        if self.free_blocks:
            for i, block in enumerate(self.free_blocks):
                if block.shape == shape and block.dtype == dtype:
                    return self.free_blocks.pop(i)
        return torch.empty(shape, dtype=dtype, device=self.device)
    
    def free(self, tensor: torch.Tensor):
        self.free_blocks.append(tensor)
    
    def get_memory_stats(self) -> Dict:
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        return {
            "allocated_gb": allocated / 1e9,
            "reserved_gb": reserved / 1e9,
            "utilization": allocated / reserved if reserved > 0 else 0,
        }
```

## 21.5 Profiling and Benchmarking

### 21.5.1 Profiling Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                 PROFILING PIPELINE                          │
│                                                             │
│  1. Macro Profiling                                         │
│  ├── End-to-end latency measurement                        │
│  ├── Throughput under load                                 │
│  └── Resource utilization (GPU/CPU/Memory)                 │
│                                                             │
│  2. Micro Profiling                                         │
│  ├── Operator-level timing                                 │
│  ├── Memory allocation patterns                            │
│  └── Kernel launch overhead                                │
│                                                             │
│  3. Comparative Profiling                                   │
│  ├── Before/after optimization                             │
│  ├── Across hardware configurations                        │
│  └── Different quantization levels                         │
│                                                             │
│  4. Continuous Profiling                                    │
│  ├── Production traffic analysis                           │
│  ├── Anomaly detection in metrics                          │
│  └── Capacity planning trends                              │
└────────────────────────────────────────────────────────────┘
```

### 21.5.2 Benchmark Script

```python
# benchmark_inference.py
import time
import statistics
from typing import List, Dict

def benchmark_inference(model_fn, prompts: List[str], runs: int = 10) -> Dict:
    latencies = []
    throughputs = []
    
    for _ in range(runs):
        start = time.perf_counter()
        output = model_fn(prompts)
        end = time.perf_counter()
        
        elapsed = end - start
        tokens = sum(len(p.split()) for p in output)
        latencies.append(elapsed)
        throughputs.append(tokens / elapsed)
    
    return {
        "p50_latency_ms": statistics.median(latencies) * 1000,
        "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] * 1000,
        "avg_throughput": statistics.mean(throughputs),
        "std_throughput": statistics.stdev(throughputs),
    }
```


## 21.6 Cost Optimization

### 21.6.1 Cost Model for AI Inference

```python
# cost_calculator.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class HardwareConfig:
    gpu_type: str  # A100, H100, etc.
    gpu_count: int
    gpu_hourly_cost: float
    cpu_hourly_cost: float
    memory_gb: float
    memory_hourly_cost: float

@dataclass
class InferenceConfig:
    model_params_billions: float
    tokens_per_request: int
    requests_per_day: int
    hardware: HardwareConfig

class AICostCalculator:
    def __init__(self, config: InferenceConfig):
        self.config = config
    
    def calculate_monthly_cost(self) -> Dict[str, float]:
        hw = self.config.hardware
        
        # GPU cost
        gpu_daily = hw.gpu_hourly_cost * hw.gpu_count * 24
        gpu_monthly = gpu_daily * 30
        
        # CPU cost
        cpu_daily = hw.cpu_hourly_cost * 24
        cpu_monthly = cpu_daily * 30
        
        # Memory cost
        mem_daily = hw.memory_hourly_cost * hw.memory_gb * 24
        mem_monthly = mem_daily * 30
        
        total_monthly = gpu_monthly + cpu_monthly + mem_monthly
        
        # Cost per token
        total_tokens = self.config.tokens_per_request * self.config.requests_per_day * 30
        cost_per_token = total_monthly / total_tokens if total_tokens > 0 else 0
        
        return {
            "gpu_monthly": gpu_monthly,
            "cpu_monthly": cpu_monthly,
            "memory_monthly": mem_monthly,
            "total_monthly": total_monthly,
            "cost_per_1m_tokens": cost_per_token * 1e6,
        }
    
    def optimize_for_cost(self) -> Dict[str, str]:
        recommendations = []
        
        if self.config.hardware.gpu_count > 1:
            recommendations.append(
                "Consider tensor parallelism to reduce GPU count"
            )
        
        if self.config.model_params_billions > 70:
            recommendations.append(
                "Use quantization (INT4/INT8) to reduce memory requirements"
            )
        
        if self.config.requests_per_day < 10000:
            recommendations.append(
                "Consider serverless inference for low-traffic scenarios"
            )
        
        return {"recommendations": recommendations}
```

### 21.6.2 Optimization Decision Matrix

| Scenario | Primary Bottleneck | Recommended Optimization |
|----------|-------------------|--------------------------|
| High traffic, low latency | Throughput | Continuous batching + tensor parallelism |
| Long documents | Memory (KV cache) | Sliding window + paged attention |
| Cost-sensitive | Compute | Quantization + model distillation |
| Real-time chat | Latency | Speculative decoding + prompt caching |
| Batch processing | Throughput | Dynamic batching + pipeline parallelism |
