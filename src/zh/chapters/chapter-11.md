# 第11章：LLM 推理架构

> 🟡 中级 → 🔴 高级 | 阅读时长 25-30分钟 | 第四部分：大模型架构

---

## 目录

- [11.1 推理引擎对比](#111-推理引擎对比)
- [11.2 KV Cache 优化](#112-kv-cache-优化)
- [11.3 量化与蒸馏](#113-量化与蒸馏)
- [11.4 批处理与连续批处理](#114-批处理与连续批处理)
- [11.5 推理集群架构](#115-推理集群架构)
- [11.6 成本优化策略](#116-成本优化策略)
- [💡 案例：基于 vLLM 的高吞吐推理服务](#-案例基于-vllm-的高吞吐推理服务)
- [本章小结](#本章小结)
- [参考文献](#参考文献)

---

## 11.1 推理引擎对比

### 11.1.1 推理的挑战

在生产环境中服务大语言模型与训练有着根本不同的挑战。训练优先考虑吞吐量而非延迟，而推理必须同时平衡两者：

| 需求 | 训练 | 推理 |
|------|------|------|
| **主要指标** | 吞吐量（tokens/sec） | 延迟（首token时间 + tokens/sec） |
| **批处理大小** | 大（1024+） | 可变（1-1024） |
| **内存模式** | 稳态 | 动态（随上下文增长） |
| **精度** | bf16/fp16（混合） | fp16/bf16/int8/int4 |
| **优化目标** | FLOPS 利用率 | 内存带宽 + FLOPS |

📌 **核心概念**：LLM 推理是**内存带宽瓶颈**，而非计算瓶颈。每个生成的 token 都需要从 GPU 内存加载模型参数。对于一个 70B 参数的 fp16 模型，每个 token 需要读取 140GB 数据。

```
┌──────────────────────────────────────────────────────────────┐
│           为什么 LLM 推理是内存带宽瓶颈                          │
│                                                                │
│  预填充阶段（处理输入提示）：                                     │
│  ┌────────────────────────────────────────────┐              │
│  │  计算密集型：                                │              │
│  │  所有 token 的矩阵乘法                       │              │
│  │  FLOPS: O(n² × d)，n = 序列长度              │              │
│  │  GPU 利用率：60-80%                          │              │
│  └────────────────────────────────────────────┘              │
│                                                                │
│  解码阶段（生成输出 token）：                                    │
│  ┌────────────────────────────────────────────┐              │
│  │  内存带宽瓶颈：                              │              │
│  │  每个新 token 都需加载所有模型参数            │              │
│  │  算术强度：O(1) 每层                        │              │
│  │  GPU 利用率：5-30%                          │              │
│  │                                             │              │
│  │  70B 模型在 H100 上：                        │              │
│  │  - 需加载参数：140 GB                       │              │
│  │  - H100 带宽：3.35 TB/s                     │              │
│  │  - 每 token 最小时间：~42ms                  │              │
│  │  - 理论最大 tokens/sec：~24                  │              │
│  └────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────┘
```

### 11.1.2 推理引擎生态

推理引擎生态从 2023-2026 年已显著成熟：

```
┌──────────────────────────────────────────────────────────────┐
│              推理引擎对比矩阵                                   │
│                                                                │
│  引擎         │ 吞吐量 │ 延迟   │ 量化  │ PagedAttn │ 易用性  │
│  ─────────────│────────│────────│───────│───────────│─────────│
│  vLLM         │ ★★★★★ │ ★★★★  │ ★★★★ │    ✅     │ ★★★★   │
│  TensorRT-LLM │ ★★★★★ │ ★★★★★ │ ★★★★★│    ✅     │ ★★★    │
│  TGI          │ ★★★★  │ ★★★★  │ ★★★★ │    ✅     │ ★★★★★  │
│  llama.cpp    │ ★★★   │ ★★★   │ ★★★★★│    ❌     │ ★★★★★  │
│  LMDeploy     │ ★★★★  │ ★★★★★ │ ★★★★★│    ✅     │ ★★★    │
│  SGLang       │ ★★★★★ │ ★★★★★ │ ★★★★ │    ✅     │ ★★★★   │
│  MLC-LLM      │ ★★★★  │ ★★★★  │ ★★★★ │    ❌     │ ★★★    │
│  MII          │ ★★★★  │ ★★★★  │ ★★★  │    ✅     │ ★★★★   │
│                                                                │
│  图例：                                                        │
│  - 吞吐量：每 GPU 的 tokens/sec                                │
│  - 延迟：首token时间 + token间延迟                              │
│  - 量化：支持的量化格式数量                                     │
│  - PagedAttn：分页注意力支持（批处理的关键）                     │
│  - 易用性：部署和 API 兼容性                                   │
└──────────────────────────────────────────────────────────────┘
```

### 11.1.3 vLLM 架构深入分析

vLLM 引入了 PagedAttention，革新了 LLM 服务效率：

```
┌──────────────────────────────────────────────────────────────┐
│                  vLLM 架构                                     │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                    API 服务器                           │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │    │
│  │  │ OpenAI API  │  │  兼容 API     │  │  流式 SSE  │  │    │
│  │  │ 兼容接口    │  │ (LangChain)  │  │            │  │    │
│  │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │    │
│  │         └────────────────┼─────────────────┘          │    │
│  │                          ▼                             │    │
│  │              ┌─────────────────────┐                  │    │
│  │              │   请求路由器          │                  │    │
│  │              │  （负载均衡器）        │                  │    │
│  │              └──────────┬──────────┘                  │    │
│  └─────────────────────────┼─────────────────────────────┘    │
│                             │                                  │
│                             ▼                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              调度器（Token 调度器）                      │    │
│  │                                                       │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  等待队列  │ 运行队列   │ 交换队列            │     │    │
│  │  │  （新请求） │（生成中）  │ （已暂停）          │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  │                                                       │    │
│  │  策略：FCFS / 优先级 / 抢占                           │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           PagedAttention 管理器                        │    │
│  │                                                       │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │  块表（类似 OS 页表）：                         │    │    │
│  │  │                                               │    │    │
│  │  │  请求 0: [Block 3] → [Block 7] → [Block 12]  │    │    │
│  │  │  请求 1: [Block 1] → [Block 5]                │    │    │
│  │  │  请求 2: [Block 8] → [Block 2] → ...          │    │    │
│  │  │                                               │    │    │
│  │  │  块池（预分配 GPU 内存）：                       │    │    │
│  │  │  [0][1][2][3][4][5][6][7][8][9][10][11][12].. │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              模型执行器                                 │    │
│  │                                                       │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  GPU 工作器（Ray / torch.distributed）        │     │    │
│  │  │  ├── 工作器 0: TP rank 0                    │     │    │
│  │  │  ├── 工作器 1: TP rank 1                    │     │    │
│  │  │  ├── 工作器 2: TP rank 2                    │     │    │
│  │  │  └── 工作器 3: TP rank 3                    │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 11.1.4 SGLang RadixAttention

SGLang 引入了 RadixAttention，使用基数树在具有公共前缀的请求之间共享 KV 缓存：

```
┌──────────────────────────────────────────────────────────────┐
│              SGLang RadixAttention                             │
│                                                                │
│  前缀树（基数树）形式的 KV 缓存：                                │
│                                                                │
│  系统提示: "你是一个有用的助手"                                  │
│                         │                                      │
│                    ┌────┴────┐                                 │
│                    │ 节点 A  │（共享 KV 缓存）                  │
│                    └────┬────┘                                 │
│              ┌──────────┼──────────┐                          │
│              ▼          ▼          ▼                          │
│         ┌────────┐ ┌────────┐ ┌────────┐                     │
│         │ 节点 B │ │ 节点 C │ │ 节点 D │（用户特定）           │
│         │"什么是 │ │"写一首  │ │"解释   │                     │
│         │ ML?"  │ │  诗"   │ │ RLHF"  │                     │
│         └────────┘ └────────┘ └────────┘                     │
│                                                                │
│  优势：                                                         │
│  - 前缀共享：避免重新计算共享前缀                               │
│  - 缓存命中：基数树查找精确前缀匹配                             │
│  - 内存效率：无重复 KV 缓存                                    │
│  - LRU 淘汰：最近最少使用的分支优先淘汰                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 11.2 KV Cache 优化

### 11.2.1 理解 KV 缓存

KV 缓存是 LLM 推理中最重要的优化。没有它，生成 N 个 token 需要 O(N²) 的计算量。有了 KV 缓存，只需要 O(N)。

```
┌──────────────────────────────────────────────────────────────┐
│                    KV 缓存详解                                  │
│                                                                │
│  自回归生成（无 KV 缓存）：                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Token 1: 从 X1 计算 K1, V1                           │    │
│  │  Token 2: 重新计算 K1, V1; 计算 K2, V2               │    │
│  │  Token 3: 重新计算 K1, V1; 重新计算 K2, V2;           │    │
│  │            计算 K3, V3                                │    │
│  │  ...                                                  │    │
│  │  Token N: 重新计算 K1..N-1, V1..N-1; 计算 KN, VN     │    │
│  │                                                       │    │
│  │  总 FLOPS: O(N² × d) 每层                             │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  自回归生成（有 KV 缓存）：                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Token 1: 计算 K1, V1; 缓存它们                       │    │
│  │  Token 2: 使用缓存的 K1,V1; 计算 K2, V2; 缓存         │    │
│  │  Token 3: 使用缓存的 K1..2,V1..2; 计算 K3, V3        │    │
│  │  ...                                                  │    │
│  │  Token N: 使用缓存的 K1..N-1,V1..N-1; 计算 KN, VN    │    │
│  │                                                       │    │
│  │  总 FLOPS: O(N × d) 每层（线性！）                     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  KV 缓存内存大小：                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  每层: 2 × seq_len × hidden_dim × 2 bytes            │    │
│  │  总计: 2 × num_layers × seq_len × hidden_dim × 2    │    │
│  │                                                       │    │
│  │  示例: LLaMA-2 70B, seq_len=4096, bf16               │    │
│  │  = 2 × 80 × 4096 × 8192 × 2 bytes                    │    │
│  │  = 每请求 10.7 GB                                     │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 11.2.2 PagedAttention

PagedAttention（vLLM）将 KV 缓存视为虚拟内存页面：

```python
# 概念性的 PagedAttention 实现
class PagedAttention:
    """
    PagedAttention 将 KV 缓存管理为固定大小的块，
    类似于操作系统虚拟内存页面。
    """
    def __init__(self, block_size=16, num_blocks=1024):
        self.block_size = block_size
        # 在 GPU 内存中预分配所有块
        self.key_cache = torch.zeros(num_blocks, num_heads,
                                      block_size, head_dim,
                                      device='cuda', dtype=torch.bfloat16)
        self.value_cache = torch.zeros(num_blocks, num_heads,
                                        block_size, head_dim,
                                        device='cuda', dtype=torch.bfloat16)
        # 块表：逻辑块到物理块的映射
        self.block_tables = {}  # req_id -> 物理块ID列表

    def allocate_block(self):
        """从池中分配一个空闲块。"""
        return self.free_blocks.pop()

    def append_token(self, req_id, layer_idx, token_idx,
                     key, value):
        """将单个 token 的 KV 追加到缓存。"""
        logical_block = token_idx // self.block_size
        block_offset = token_idx % self.block_size

        # 如果需要，分配新块
        if req_id not in self.block_tables:
            self.block_tables[req_id] = []

        if logical_block >= len(self.block_tables[req_id]):
            phys_block = self.allocate_block()
            self.block_tables[req_id].append(phys_block)

        phys_block = self.block_tables[req_id][logical_block]

        # 将 KV 存储到物理块中
        self.key_cache[phys_block, :, block_offset, :] = key
        self.value_cache[phys_block, :, block_offset, :] = value

    def get_kv(self, req_id, layer_idx):
        """使用块表检索请求的 KV 缓存。"""
        blocks = self.block_tables[req_id]
        keys = [self.key_cache[b] for b in blocks]
        values = [self.value_cache[b] for b in blocks]
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)

    def copy_on_write(self, req_id, fork_id):
        """用于束搜索的 KV 缓存复制（写时复制语义）。"""
        self.block_tables[fork_id] = self.block_tables[req_id].copy()
```

### 11.2.3 KV 缓存压缩

几种技术可以减少 KV 缓存内存：

```
┌──────────────────────────────────────────────────────────────┐
│              KV 缓存压缩技术                                    │
│                                                                │
│  1. GQA/MQA（架构层面）：                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  标准 MHA: KV 头数 = Q 头数（例如 64）                  │    │
│  │  GQA:      KV 头数 = Q 头数 / 8（例如 8）              │    │
│  │  MQA:      KV 头数 = 1                                │    │
│  │                                                       │    │
│  │  内存减少：8x (GQA-8) 到 64x (MQA)                    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  2. KV 缓存量化：                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  以 INT8 或 INT4 存储 KV 缓存                         │    │
│  │  内存减少：2x (INT8) 到 4x (INT4)                     │    │
│  │  质量影响：通过逐通道量化影响极小                       │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  3. 滑动窗口注意力：                                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  仅缓存最后 W 个 token（例如 W=4096）                  │    │
│  │  Mistral-7B: 滑动窗口 = 4096                          │    │
│  │  内存：固定，与序列长度无关                             │    │
│  │  限制：无法关注窗口之外的内容                           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  4. StreamingLLM（注意力汇聚）：                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  保留前几个 token（注意力汇聚）+ 最后 W 个 token        │    │
│  │  示例：保留 token 0-3 + 最后 4093 个 token             │    │
│  │  适用于任何模型（无需重训练）                            │    │
│  │  质量：对大多数任务良好，前几个 token 有轻微损失        │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  5. H2O（重击手预言机）：                                        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  动态淘汰低重要性 KV 条目                              │    │
│  │  保留注意力分数高的 token                              │    │
│  │  自适应内容，非仅位置                                  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

```python
# KV 缓存量化示例
class KVCacheQuantizer:
    """将 KV 缓存量化为 INT8 以节省内存。"""

    def __init__(self, num_heads, head_dim):
        self.num_heads = num_heads
        self.head_dim = head_dim

    def quantize_int8(self, kv_tensor):
        """逐头对称 INT8 量化。"""
        abs_max = kv_tensor.abs().amax(dim=-1, keepdim=True)
        scale = abs_max / 127.0
        kv_int8 = torch.round(kv_tensor / scale).to(torch.int8)
        return kv_int8, scale

    def dequantize(self, kv_int8, scale):
        """反量化回 bf16。"""
        return kv_int8.to(torch.bfloat16) * scale

    def compute_cache_memory(self, batch_size, seq_len, num_layers):
        """计算量化 KV 缓存的内存占用。"""
        bytes_per_element = 1  # INT8
        cache_size = (
            2 *  # K 和 V
            batch_size *
            self.num_heads *
            seq_len *
            self.head_dim *
            bytes_per_element *
            num_layers
        )
        return cache_size / (1024**3)  # GB
```

### 11.2.4 前缀缓存

对于具有系统提示的聊天机器人应用，前缀缓存可以节省大量计算：

```python
# 前缀缓存策略
class PrefixCache:
    """
    缓存共享提示前缀的 KV 计算。
    在聊天机器人应用中对系统提示特别有用。
    """
    def __init__(self, max_prefix_length=2048):
        self.cache = {}  # hash(prompt) -> (kv_cache, computed_length)
        self.max_prefix_length = max_prefix_length

    def get_prefix_length(self, new_prompt, cached_prompts):
        """找到最长的匹配前缀。"""
        best_match = 0
        best_key = None

        for cached_key, (_, computed_len) in self.cache.items():
            if new_prompt.startswith(cached_key):
                if computed_len > best_match:
                    best_match = computed_len
                    best_key = cached_key

        return best_match, best_key

    def compute_with_cache(self, model, new_prompt, tokenizer):
        """计算 KV 缓存，如果可用则重用缓存前缀。"""
        prefix_len, prefix_key = self.get_prefix_length(new_prompt)

        if prefix_len > 0:
            cached_kv, _ = self.cache[prefix_key]
            remaining_prompt = new_prompt[prefix_len:]
            remaining_tokens = tokenizer.encode(remaining_prompt)
            new_kv = model.compute_kv(remaining_tokens)
            full_kv = self.concatenate_kv(cached_kv, new_kv)
        else:
            tokens = tokenizer.encode(new_prompt)
            full_kv = model.compute_kv(tokens)

        self.cache[new_prompt] = (full_kv, len(new_prompt))
        return full_kv

# 示例：系统提示缓存
# 系统提示："你是一个专门从事..."（512 个 token）
# 无缓存：每个请求计算 512 + user_tokens 的 KV
# 有缓存：首次请求计算完整，后续请求节省 512 个 token
# 节省：512 × 80 层 × 64 头 × 128 维 × 2 字节 × 2（K+V）= 每请求 1.3 GB
```

---

## 11.3 量化与蒸馏

### 11.3.1 量化分类

量化降低模型权重和/或激活的精度，以减少内存占用并提升推理速度：

```
┌──────────────────────────────────────────────────────────────┐
│              量化分类                                            │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              训练后量化（PTQ）                            │  │
│  │  训练后量化，无需重训练                                   │  │
│  │                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │  │
│  │  │ 仅权重量化    │  │ 权重量化     │  │ SmoothQuant  │ │  │
│  │  │ (W4A16)      │  │ (W8A8)       │  │ (W8A8)       │ │  │
│  │  │              │  │              │  │              │ │  │
│  │  │ INT4 权重    │  │ INT8 权重    │  │ 数学等价     │ │  │
│  │  │ fp16 激活    │  │ INT8 激活    │  │ 逐通道量化   │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │          量化感知训练（QAT）                              │  │
│  │  训练期间量化，质量更高                                   │  │
│  │                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐                   │  │
│  │  │ QLoRA        │  │ LLM-QAT      │                   │  │
│  │  │ INT4 微调    │  │ LLM 全量化   │                   │  │
│  │  └──────────────┘  └──────────────┘                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │          混合精度                                        │  │
│  │  不同层使用不同精度                                      │  │
│  │  敏感层：更高精度                                       │  │
│  │  鲁棒层：更低精度                                       │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 11.3.2 GPTQ（训练后量化）

GPTQ 是最流行的仅权重量化方法之一：

```python
# 使用 GPTQ 进行量化（通过 AutoGPTQ）
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

# 定义量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,              # 4 位量化
    group_size=128,      # 分组量化（每组 125 个权重）
    damp_percent=0.01,   # Hessian 阻尼因子
    desc_act=True,       # 按激活排序以获得更好量化
    sym=False,           # 非对称量化
)

# 加载预训练模型
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantize_config,
    device_map="auto"
)

# 使用校准数据进行量化
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
calibration_data = tokenizer(
    "这是一个用于量化的校准文本..." * 100,
    return_tensors="pt"
).input_ids.to("cuda")

model.quantize(calibration_data)

# 保存量化模型
model.save_quantized("./llama-2-70b-gptq-4bit")

# 加载和使用量化模型
model = AutoGPTQForCausalLM.from_quantized(
    "./llama-2-70b-gptq-4bit",
    device_map="auto",
    use_triton=False
)
```

### 11.3.3 GGUF 量化（llama.cpp）

GGUF 是 CPU/边缘推理的标准格式：

```
┌──────────────────────────────────────────────────────────────┐
│              GGUF 量化类型                                     │
│                                                                │
│  类型     │ 位数 │ 方法          │ 质量   │ 速度  │ 内存      │
│  ─────────│──────│───────────────│────────│───────│───────────│
│  Q2_K     │  2   │ k-quant       │ ★★☆☆☆ │ ★★★★★│ ★★★★★   │
│  Q3_K_M   │  3   │ k-quant       │ ★★★☆☆ │ ★★★★☆│ ★★★★☆   │
│  Q4_K_M   │  4   │ k-quant       │ ★★★★☆ │ ★★★★☆│ ★★★★☆   │
│  Q5_K_M   │  5   │ k-quant       │ ★★★★★ │ ★★★☆☆│ ★★★☆☆   │
│  Q6_K     │  6   │ k-quant       │ ★★★★★ │ ★★★☆☆│ ★★★☆☆   │
│  Q8_0     │  8   │ 四舍五入      │ ★★★★★ │ ★★★☆☆│ ★★★☆☆   │
│  F16      │  16  │ float16       │ ★★★★★ │ ★★☆☆☆│ ★★☆☆☆   │
│  IQ4_XS   │ 4.25 │ 重要性量化    │ ★★★★☆ │ ★★★★☆│ ★★★★☆   │
│                                                                │
│  k-quant 对不同层使用不同量化                                   │
│  （重要层获得更高精度）                                         │
│                                                                │
│  示例：LLaMA-2 7B                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Q2_K: 2.67 GB  → 适配 4GB 显存                      │    │
│  │  Q4_K_M: 4.08 GB → 适配 6GB 显存                     │    │
│  │  Q8_0: 7.16 GB → 适配 8GB 显存                       │    │
│  │  F16: 13.5 GB → 需要 16GB 显存                       │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

```python
# 使用 llama.cpp Python 绑定
from llama_cpp import Llama

# 加载 GGUF 量化模型
llm = Llama(
    model_path="./models/llama-2-7b-q4_k_m.gguf",
    n_ctx=4096,        # 上下文长度
    n_gpu_layers=35,   # 卸载到 GPU 的层数
    n_threads=8,       # CPU 线程数
    verbose=False
)

# 生成文本
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "用简单的话解释量子计算。"}
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

### 11.3.4 知识蒸馏

蒸馏训练较小的"学生"模型来模仿较大的"教师"模型：

```
┌──────────────────────────────────────────────────────────────┐
│              知识蒸馏流水线                                      │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  教师模型（大）                                         │    │
│  │  例如 LLaMA-2 70B                                     │    │
│  │  ┌──────────────────────────────────────────┐        │    │
│  │  │  输入 → [Transformer 层 80] → Logits      │        │    │
│  │  └──────────────────────────────────────────┘        │    │
│  │                    │                                   │    │
│  │                    ▼                                   │    │
│  │  ┌──────────────────────────────────────────┐        │    │
│  │  │  温度 T 的 Softmax                        │        │    │
│  │  │  P_teacher(y|x) = softmax(logits / T)    │        │    │
│  │  └──────────────────────────────────────────┘        │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  蒸馏损失                                              │    │
│  │                                                       │    │
│  │  L = α * L_KL + (1-α) * L_CE                        │    │
│  │                                                       │    │
│  │  L_KL = KL(softmax(z_t/T) || softmax(z_s/T))        │    │
│  │  L_CE = CrossEntropy(student_logits, true_labels)    │    │
│  │                                                       │    │
│  │  T = 温度（越高分布越软）                              │    │
│  │  α = 蒸馏和监督之间的平衡                             │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  学生模型（小）                                         │    │
│  │  例如 LLaMA-2 7B                                      │    │
│  │  ┌──────────────────────────────────────────┐        │    │
│  │  │  输入 → [Transformer 层 32] → Logits      │        │    │
│  │  └──────────────────────────────────────────┘        │    │
│  │                                                       │    │
│  │  训练以最小化 L，同时从教师学习                        │    │
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
        """组合蒸馏 + 交叉熵损失。"""
        # 软目标损失（KL 散度）
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(soft_student, soft_teacher,
                           reduction='batchmean') * (self.temperature ** 2)

        # 硬目标损失（交叉熵）
        ce_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss

    def train_step(self, batch, optimizer):
        """单步训练。"""
        input_ids = batch['input_ids']
        labels = batch['labels']

        # 教师前向（无梯度）
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids).logits

        # 学生前向
        student_logits = self.student(input_ids).logits

        # 计算损失
        loss = self.distillation_loss(student_logits, teacher_logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
```

---

## 11.4 批处理与连续批处理

### 11.4.1 静态批处理 vs 连续批处理

实现高吞吐量 LLM 服务的关键创新：

```
┌──────────────────────────────────────────────────────────────┐
│              静态批处理 vs 连续批处理                            │
│                                                                │
│  静态批处理：                                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  批次 1: [请求 A(200 tok), B(50 tok), C(300 tok)]    │    │
│  │                                                       │    │
│  │  时间 ──────────────────────────────────────────────▶ │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │ A: ████████████████████████████████████████████│    │    │
│  │  │ B: ██████████                                   │    │    │
│  │  │ C: ██████████████████████████████████████████████│   │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  │                                                       │    │
│  │  问题: B 在第 50 步完成但 GPU 空闲                     │    │
│  │        直到 A 和 C 完成。GPU 利用率：低                 │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  连续批处理（迭代级调度）：                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  时间 ──────────────────────────────────────────────▶ │    │
│  │  第 1 步:  [A, B, C]                                  │    │
│  │  第 2 步:  [A, B, C]                                  │    │
│  │  ...                                                   │    │
│  │  第 50 步: [A, B, C]  ← B 完成，槽位释放              │    │
│  │  第 51 步: [A, D, C]  ← D 插入释放的槽位              │    │
│  │  第 52 步: [A, D, C]                                  │    │
│  │  ...                                                   │    │
│  │  第 200 步: [A, D, C] ← A 完成                        │    │
│  │  第 201 步: [E, D, C] ← E 插入                        │    │
│  │                                                       │    │
│  │  GPU 利用率: ████████████████████████████████████████  │    │
│  │              （始终满载）                               │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  吞吐量提升：使用连续批处理可提升 2-5 倍                        │
└──────────────────────────────────────────────────────────────┘
```

### 11.4.2 分块预填充

分块预填充将长提示分割为块，以便与解码请求交错处理：

```python
class ChunkedPrefillScheduler:
    """
    分块预填充允许将长提示分块处理，
    与解码请求交错以获得更好的延迟。
    """
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def schedule(self, waiting_queue, running_queue):
        """
        优先调度解码请求，然后调度预填充块。
        这确保了低延迟的解码（流式）请求。
        """
        scheduled = []

        # 首先：调度所有运行中的解码请求
        for req in running_queue:
            if not req.is_finished:
                scheduled.append(req)

        # 然后：调度等待队列中的预填充块
        for req in waiting_queue:
            if req.prefill_done:
                continue

            remaining = req.total_length - req.processed_length
            chunk = min(remaining, self.chunk_size)

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

### 11.4.3 推测解码

推测解码使用较小的草稿模型提议 token，然后由目标模型并行验证：

```
┌──────────────────────────────────────────────────────────────┐
│              推测解码                                            │
│                                                                │
│  草稿模型（小，快速）：                                          │
│  "模型微调的最佳实践是"                                         │
│       │                                                       │
│       ▼                                                       │
│  生成 K=5 个候选 token：                                       │
│  ["使用", "LoRA", "技术", "进行", "高效"]                     │
│                                                                │
│  目标模型（大，精确）：                                          │
│  并行处理所有 K 个 token：                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  输入: "模型微调的最佳实践是" + ["使用","LoRA",...,"高效"]│   │
│  │  输出: 验证每个位置                                    │    │
│  │                                                       │    │
│  │  位置 4: P_target("使用")=0.3, P_draft("使用")=0.25  │    │
│  │  → 接受（比率 > 1.0）                                │    │
│  │                                                       │    │
│  │  位置 5: P_target("的")=0.4, P_draft("的")=0.1       │    │
│  │  → 拒绝，从调整后的分布采样                            │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  结果: 3-4 倍加速，质量损失极小                                │
│                                                                │
│  时间分析：                                                     │
│  ┌──────────────────────────────────────────────┐            │
│  │  无推测: 5 次大模型顺序前向                     │            │
│  │  = 5 × T_large                                │            │
│  │                                               │            │
│  │  有推测: 1 次草稿前向 (T_s)                     │            │
│  │         + 1 次目标前向 (T_l)                   │            │
│  │  总计: T_s + T_l                              │            │
│  │                                               │            │
│  │  如果 T_l ≈ 5 × T_s（草稿模型小 5 倍）：       │            │
│  │  加速: 5T_l / (T_l/5 + T_l) = 25/6 ≈ 4.2x    │            │
│  └──────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

---

## 11.5 推理集群架构

### 11.5.1 单节点多 GPU 服务

```
┌──────────────────────────────────────────────────────────────┐
│           单节点推理服务器（4× H100）                            │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  负载均衡器（HAProxy / NGINX）                         │    │
│  │  └── 健康检查、限流、请求路由                          │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  vLLM 服务器（TP=4）                                   │    │
│  │  ┌────────────────────────────────────────────────┐  │    │
│  │  │  GPU 0     GPU 1     GPU 2     GPU 3          │  │    │
│  │  │  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐      │  │    │
│  │  │  │ TP0 │──▶│ TP1 │──▶│ TP2 │──▶│ TP3 │      │  │    │
│  │  │  │L0-19 │   │L20-39│   │L40-59│   │L60-79│    │  │    │
│  │  │  └─────┘   └─────┘   └─────┘   └─────┘      │  │    │
│  │  │  ◄──── NVLink (900 GB/s 双向) ─────▶          │  │    │
│  │  └────────────────────────────────────────────────┘  │    │
│  │                                                       │    │
│  │  模型: LLaMA-3 70B (TP=4, fp16)                      │    │
│  │  内存: 140GB / 4 = 每 GPU 35GB                        │    │
│  │  吞吐量: ~120 tokens/sec                              │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 11.5.2 多节点推理集群

```
┌──────────────────────────────────────────────────────────────────┐
│           多节点推理集群                                            │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  API 网关 + 负载均衡器                                     │    │
│  │  ├── 基于模型大小的请求路由                                 │    │
│  │  ├── 自动故障转移                                         │    │
│  │  └── 限流和身份验证                                        │    │
│  └─────────────────────┬────────────────────────────────────┘    │
│                         │                                          │
│         ┌───────────────┼───────────────┐                        │
│         ▼               ▼               ▼                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │  节点 0    │  │  节点 1    │  │  节点 2    │                │
│  │  模型 A   │  │  模型 A   │  │  模型 B   │                │
│  │  (70B)    │  │  (70B)    │  │  (8×7B)   │                │
│  │  4× H100  │  │  4× H100  │  │  2× A100  │                │
│  └────────────┘  └────────────┘  └────────────┘                │
│                                                                    │
│  模型加载策略：                                                     │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  方案 A: 每个节点托管完整模型                              │    │
│  │  → 简单，独立节点                                         │    │
│  │  → 无跨节点通信                                          │    │
│  │  → 受限于单节点 GPU 内存                                  │    │
│  │                                                          │    │
│  │  方案 B: 模型跨节点分片                                   │    │
│  │  → 模型超过单节点容量时必须使用                            │    │
│  │  → 跨节点通信开销                                         │    │
│  │  → 需要高带宽互联（InfiniBand）                           │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 11.5.3 自动扩展架构

```python
# 基于 Kubernetes 的 LLM 推理自动扩展
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
        averageValue: "70"  # GPU 利用率 70% 时扩展
  - type: Pods
    pods:
      metric:
        name: request_queue_depth
      target:
        type: AverageValue
        averageValue: "10"  # 队列深度 > 10 时扩展
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
# 带 GPU 资源请求的 Pod 模板
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

## 11.6 成本优化策略

### 11.6.1 GPU 选择指南

```
┌──────────────────────────────────────────────────────────────┐
│              GPU 选择指南（LLM 推理）                           │
│                                                                │
│  GPU        │ 显存  │ 带宽    │ FP16 TFLOPS │ $/小时 │ 适用   │
│  ───────────│───────│─────────│─────────────│───────│────────│
│  A10G       │ 24GB  │ 600GB/s │    31       │ $1.10 │ ≤13B  │
│  A100 40GB  │ 40GB  │ 1.6TB/s │   312       │ $3.40 │ ≤30B  │
│  A100 80GB  │ 80GB  │ 2.0TB/s │   312       │ $4.10 │ ≤70B  │
│  H100       │ 80GB  │ 3.35TB/s│   990       │ $5.50 │ ≤70B  │
│  H100 SXM   │ 80GB  │ 3.35TB/s│   990       │ $6.00 │ ≤70B  │
│  H200       │ 141GB │ 4.8TB/s │   990       │ $7.00 │ ≤70B+ │
│  L40S       │ 48GB  │ 864GB/s │   181       │ $1.80 │ ≤30B  │
│                                                                │
│  性价比分析（70B 模型，TP=4）：                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  A100 80GB: 4×$4.10 = $16.40/hr → 40 tok/s          │    │
│  │  H100:      4×$5.50 = $22.00/hr → 120 tok/s         │    │
│  │                                                       │    │
│  │  每 100 万 token 成本：                                │    │
│  │  A100: $16.40 / 40 = 每 1K token $0.41              │    │
│  │  H100: $22.00 / 120 = 每 1K token $0.18             │    │
│  │                                                       │    │
│  │  对于 70B 模型，H100 的性价比高 2.3 倍                 │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 11.6.2 成本降低技术

```
┌──────────────────────────────────────────────────────────────┐
│              成本优化技术                                        │
│                                                                │
│  1. 量化（立即降低 2-4 倍成本）                                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  70B FP16: 4× H100 = $22/hr                          │    │
│  │  70B INT4: 1× H100 = $5.50/hr                        │    │
│  │  节省: 75%                                           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  2. 连续批处理（2-3 倍吞吐量提升）                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  静态: 每请求 20 tok/s                                │    │
│  │  连续: 每请求 60 tok/s（3 倍提升）                     │    │
│  │  每 token 成本降低 3 倍                                │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  3. 抢占式实例（节省 60-80%）                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  按需 H100: $5.50/hr                                  │    │
│  │  抢占式 H100: ~$1.65/hr（70% 折扣）                   │    │
│  │  风险: 可能被抢占（优雅处理）                          │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  4. 模型大小选择（为任务选择合适大小）                           │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  任务: 客服聊天机器人                                  │    │
│  │  70B 模型: 99.2% 准确率, 每请求 $0.001               │    │
│  │  7B 模型:  97.8% 准确率, 每请求 $0.0001              │    │
│  │  决策: 使用 7B（便宜 10 倍，准确率仅降 1.4%）         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  5. 提示缓存（重复提示节省 30-50%）                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  系统提示: 500 tokens, 每天重复 1000 次               │    │
│  │  无缓存: 500 × 1000 = 500K tokens 计算               │    │
│  │  有缓存: 500 + (1 × 1000) ≈ 1.5K tokens 计算        │    │
│  │  提示部分节省: 99.7% 计算量                           │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 💡 案例：基于 vLLM 的高吞吐推理服务

### 业务背景

一家 SaaS 公司需要为 10,000+ 日活用户提供自定义 13B 参数代码生成模型服务。要求：
- 首token 时间（TTFT）：< 500ms
- 吞吐量：500+ 请求/分钟
- 成本：< $500/月（云上）
- 可用性：99.9%

### 架构设计

```
┌──────────────────────────────────────────────────────────────────┐
│           生产推理架构                                              │
│                                                                    │
│  用户 ──▶ CDN ──▶ API 网关 ──▶ 负载均衡器 ──▶ vLLM Pod            │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  云基础设施（AWS/GCP）                                     │    │
│  │                                                           │    │
│  │  ┌─────────────────────────────────────────────────┐     │    │
│  │  │  区域: us-east-1                                 │     │    │
│  │  │                                                   │     │    │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐           │     │    │
│  │  │  │Pod 0    │ │Pod 1    │ │Pod 2    │ (min=2,    │     │    │
│  │  │  │vLLM     │ │vLLM     │ │vLLM     │  max=6)    │     │    │
│  │  │  │A10G×1   │ │A10G×1   │ │A10G×1   │           │     │    │
│  │  │  │TP=1     │ │TP=1     │ │TP=1     │           │     │    │
│  │  │  └─────────┘ └─────────┘ └─────────┘           │     │    │
│  │  └─────────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 实现

```python
# deploy_vllm.py
"""
用于生产服务的完整 vLLM 部署脚本。
"""
import subprocess
import json

def deploy_vllm_cluster():
    """使用最优配置部署 vLLM 进行代码生成服务。"""

    vllm_config = {
        "model": "codellama/CodeLlama-13b-hf",
        "tensor_parallel_size": 1,  # 13B 适合单 A10G
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.92,
        "dtype": "bfloat16",
        "quantization": "awq",  # AWQ 4位以优化成本
        "enforce_eager": False,  # 启用 CUDA 图
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 64,
        "block_size": 16,
        "enable_prefix_caching": True,
        "disable_log_requests": True,
    }

    # 启动 vLLM 服务器
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

# 基准测试结果（CodeLlama-13B-AWQ 在 A10G 上）：
# ┌──────────────────────────────────────────────────────┐
# │  指标                 │ 值                           │
# │  ────────────────────│──────────────────────────────│
# │  TTFT (P50)          │ 180ms                       │
# │  TTFT (P99)          │ 420ms                       │
# │  吞吐量              │ 1,200 tokens/sec            │
# │  最大并发            │ 64 请求                      │
# │  每 GPU 成本         │ $1.10/hr (A10G)             │
# │  每 100万 token 成本  │ $0.09                       │
# └──────────────────────────────────────────────────────┘
```

### 监控仪表板

```python
# monitoring.py
"""
LLM 推理服务的关键监控指标。
"""
METRICS = {
    # 延迟指标
    "time_to_first_token_p50": "180ms",
    "time_to_first_token_p99": "420ms",
    "inter_token_latency_p50": "25ms",
    "inter_token_latency_p99": "80ms",

    # 吞吐量指标
    "tokens_per_second": "1200",
    "requests_per_minute": "500",
    "concurrent_requests": "45",

    # 资源指标
    "gpu_utilization": "78%",
    "gpu_memory_used": "18GB / 24GB",
    "kv_cache_usage": "65%",

    # 业务指标
    "cost_per_1m_tokens": "$0.09",
    "monthly_cost": "$792",
    "availability": "99.95%",
}

# 告警阈值
ALERTS = {
    "ttft_p99 > 1000ms": "高延迟 - 检查 GPU 利用率",
    "gpu_utilization > 95%": "GPU 饱和 - 扩展",
    "kv_cache_usage > 90%": "KV 缓存已满 - 增加块池",
    "error_rate > 1%": "高错误率 - 检查日志",
}
```

---

## 本章小结

本章涵盖了 LLM 推理架构的关键方面：

| 主题 | 关键要点 |
|------|---------|
| **推理引擎** | vLLM 和 SGLang 在吞吐量领先；TensorRT-LLM 在延迟领先 |
| **KV 缓存** | PagedAttention 消除内存浪费；支持高并发服务 |
| **量化** | INT4 量化（GPTQ/AWQ）使 70B 模型可在单 GPU 上运行 |
| **连续批处理** | 比静态批处理提升 2-5 倍吞吐量 |
| **推测解码** | 通过并行验证草稿 token 实现 3-4 倍加速 |
| **成本优化** | 量化 + 批处理 + 抢占实例可降低 10-20 倍成本 |
| **集群设计** | 基于 GPU 利用率指标的自动扩展应对动态工作负载 |

### 性能对比

```
┌──────────────────────────────────────────────────────────────┐
│         70B 模型推理性能（H100×4）                              │
│                                                                │
│  配置               │ TTFT    │ 吞吐量      │ 每 1M token 成本 │
│  ───────────────────│─────────│─────────────│─────────────────│
│  FP16, 静态批处理    │ 800ms   │ 30 tok/s    │ $0.55          │
│  FP16, 连续批处理    │ 200ms   │ 120 tok/s   │ $0.14          │
│  INT4, 连续批处理    │ 150ms   │ 200 tok/s   │ $0.08          │
│  INT4, 连续+推测     │ 100ms   │ 500 tok/s   │ $0.03          │
│                                                                │
│  最佳实践：INT4 + 连续批处理 + 推测解码                         │
│  用于生产部署                                                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 参考文献

1. Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP.
2. Leviathan, Y., et al. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML.
3. Frantar, E., et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR.
4. Lin, J., et al. (2024). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys.
5. Zheng, L., et al. (2023). "SGLang: Efficient Execution of Structured Language Model Programs." arXiv.
6. vLLM 文档. https://docs.vllm.ai/
7. TensorRT-LLM 文档. https://nvidia.github.io/TensorRT-LLM/
8. Chen, C., et al. (2024). "Activester: Accelerating Large Language Model Inference with Dynamic Parallelism." arXiv.
9. Liu, Z., et al. (2023). "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory." arXiv.
10. Pope, R., et al. (2023). "Efficiently Scaling Transformer Inference." MLSys.

---

*← [第10章 - LLM 架构设计基础](chapter-10.md) | [第12章 - RAG 系统架构](chapter-12.md) →*
