# 第21章：AI系统性能工程

**读者级别：** 🔴 高级 | **页数：** 28 | **代码：** Python, C++, CUDA

---

## 21.1 AI性能基础

### 21.1.1 性能三角

AI系统性能基于三大支柱：延迟、吞吐量和成本。优化其中一个往往会影响其他。目标是为您的用例找到合适的平衡。

```
                    延迟
                   (p50/p99)
                      /\\
                     /  \\
                    /    \\
                   / 最优 \\
                  /  平衡  \\
                 /__________\\
         吞吐量              成本
        (tokens/sec)      ($/token)
```

### 21.1.2 关键性能指标

| 指标 | 定义 | 目标（生产环境） |
|------|------|-----------------|
| 首token时间(TTFT) | 到第一个输出token的时间 | 200-500ms |
| token间延迟 | 后续token间的时间 | 20-50ms |
| 吞吐量 | 每秒生成的token数 | 100-1000 tok/s |
| 每秒请求数 | 并发请求处理 | 100-10,000 RPS |
| GPU利用率 | GPU计算使用百分比 | 70-90% |
| 内存带宽 | HBM吞吐利用率 | 60-80% |
| 每百万token成本 | 包含计算的总成本 | $0.10-$2.00 |

## 21.2 推理优化

### 21.2.1 量化策略

量化将模型精度从FP32/FP16降低到INT8/INT4，显著减少内存和计算需求。

```python
# quantization_example.py
import torch
from torch.quantization import quantize_dynamic

def quantize_model(model, dtype=torch.qint8):
    quantized = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=dtype
    )
    return quantized
```

### 21.2.2 KV缓存优化

KV缓存存储先前token的键值对以避免重新计算。高效管理对长上下文推理至关重要。

```
┌────────────────────────────────────────────────────────────┐
│                    KV缓存管理                                │
│                                                             │
│  全缓存（无优化）                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [Token 1 KV] [Token 2 KV] ... [Token N KV]         │   │
│  │ 内存: O(N * d_model * n_layers * 2)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  分页注意力（vLLM风格）                                      │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │
│  │页 0  │ │页 1  │ │页 2  │ │页 3  │ │页 4  │            │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘            │
│  优点: 无碎片化，动态分配                                    │
│                                                             │
│  滑动窗口（Mistral风格）                                     │
│  ┌──────────────────────────────────────┐                  │
│  │  [窗口: 仅最后W个token]              │                  │
│  │  内存: O(W * d_model * n_layers)     │                  │
│  └──────────────────────────────────────┘                  │
└────────────────────────────────────────────────────────────┘
```

## 21.3 分布式推理

### 21.3.1 张量并行

```python
# tensor_parallelism.py
class TensorParallelLayer:
    def __init__(self, weight, num_gpus):
        self.shards = [
            weight[:, i*chunk:(i+1)*chunk]
            for i in range(num_gpus)
        ]
    
    def forward(self, x):
        outputs = [shard @ x for shard in self.shards]
        return sum(outputs)  # AllReduce
```

### 21.3.2 流水线并行

```python
# pipeline_parallelism.py
class PipelineScheduler:
    def __init__(self, stages, micro_batch_size=4):
        self.stages = stages
        self.micro_batch_size = micro_batch_size
    
    def schedule_forward(self, batch):
        return [
            {"stage": i, "device": stage.device}
            for i, stage in enumerate(self.stages)
        ]
```

## 21.4 缓存与内存管理

### 21.4.1 提示缓存策略

```python
# prompt_cache.py
from collections import OrderedDict

class PromptCache:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.stats = {"hits": 0, "misses": 0}
    
    def get_prefix_match(self, prompt):
        for cached in reversed(list(self.cache.keys())):
            if prompt.startswith(cached):
                self.stats["hits"] += 1
                return cached
        self.stats["misses"] += 1
        return None
    
    def store(self, prompt, kv_cache):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[prompt] = kv_cache
```

## 21.5 性能分析与基准测试

### 21.5.1 基准测试脚本

```python
# benchmark_inference.py
import time
import statistics

def benchmark_inference(model_fn, prompts, runs=10):
    latencies = []
    throughputs = []
    
    for _ in range(runs):
        start = time.perf_counter()
        output = model_fn(prompts)
        elapsed = time.perf_counter() - start
        
        tokens = sum(len(p.split()) for p in output)
        latencies.append(elapsed)
        throughputs.append(tokens / elapsed)
    
    return {
        "p50_latency_ms": statistics.median(latencies) * 1000,
        "avg_throughput": statistics.mean(throughputs),
    }
```

## 21.6 成本优化

| 场景 | 主要瓶颈 | 推荐优化 |
|------|---------|---------|
| 高流量、低延迟 | 吞吐量 | 持续批处理+张量并行 |
| 长文档 | 内存(KV缓存) | 滑动窗口+分页注意力 |
| 成本敏感 | 计算 | 量化+模型蒸馏 |
| 实时聊天 | 延迟 | 推测解码+提示缓存 |
| 批处理 | 吞吐量 | 动态批处理+流水线并行 |
