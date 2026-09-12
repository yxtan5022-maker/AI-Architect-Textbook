# 附录

---

## 附录A：AI架构术语表

| 术语 | 定义 |
|------|------|
| **注意力机制** | 计算输入token之间加权关系的神经网络组件 |
| **Transformer** | 基于自注意力的架构，支持序列并行处理 |
| **KV缓存** | 存储自回归生成中间注意力计算的键值缓存 |
| **量化** | 降低模型权重数值精度（FP32 → INT8/INT4） |
| **MoE** | 混合专家 - 将token路由到专门子网络的稀疏架构 |
| **RAG** | 检索增强生成 - 将检索与生成模型结合 |
| **分词** | 将文本转换为模型处理的数值token |
| **嵌入** | token或概念的密集向量表示 |
| **微调** | 在特定任务数据上额外训练以适应预训练模型 |
| **推理** | 使用训练好的模型从新输入生成输出 |
| **蒸馏** | 训练较小模型模仿较大模型行为 |
| **LoRA** | 低秩适应 - 参数高效微调技术 |
| **RLHF** | 基于人类反馈的强化学习 |
| **DPO** | 直接偏好优化 - RLHF的替代方案 |

## 附录B：技术参考

### B.1 模型服务框架

| 框架 | 用例 | 关键特性 |
|------|------|---------|
| vLLM | 高吞吐量服务 | 分页注意力、持续批处理 |
| TensorRT-LLM | NVIDIA优化推理 | GPU内核优化 |
| TGI | 生产文本生成 | 量化、流式传输 |
| Ollama | 本地开发 | 简单设置、模型管理 |
| llama.cpp | CPU推理 | GGUF量化、可移植性 |

### B.2 向量数据库

| 数据库 | 类型 | 最佳用途 |
|--------|------|---------|
| Pinecone | 托管 | 生产、易用性 |
| Weaviate | 开源 | 多模态、GraphQL |
| Qdrant | 开源 | 高性能、过滤 |
| Chroma | 轻量级 | 开发、原型 |
| Milvus | 分布式 | 大规模生产 |

### B.3 监控工具

| 工具 | 类别 | 关键能力 |
|------|------|---------|
| Prometheus | 指标 | 时序数据 |
| Grafana | 可视化 | 仪表板 |
| LangSmith | LLM特定 | 追踪、评估 |
| Weights & Biases | ML运维 | 实验跟踪 |
| Datadog | 全栈 | 基础设施+APM |

## 附录C：配置模板

### C.1 模型配置

```yaml
# model_config.yaml
model:
  name: "llama-3-70b"
  quantization: "awq_int4"
  max_context_length: 8192
  tensor_parallel: 4
  
serving:
  max_batch_size: 32
  max_concurrent_requests: 100
  timeout_seconds: 30
  
safety:
  content_filter: true
  injection_detection: true
  pii_redaction: true
  
monitoring:
  metrics_port: 9090
  log_level: "info"
  tracing_enabled: true
```

### C.2 Kubernetes部署

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-model
  template:
    spec:
      containers:
      - name: model
        image: ai-model:latest
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: 32Gi
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
```

## 附录D：常见陷阱与解决方案

| 陷阱 | 症状 | 解决方案 |
|------|------|---------|
| 推理时OOM | CUDA内存不足 | 减少批大小，启用量化 |
| 高延迟 | 响应时间慢 | 启用KV缓存，使用Flash注意力 |
| 幻觉 | 错误输出 | 添加RAG，改进基础 |
| 提示注入 | 不安全输出 | 实现输入验证 |
| token限制超出 | 输出截断 | 优化提示，增加上下文 |
| 吞吐量低 | 低RPS | 启用持续批处理 |
| 内存泄漏 | 内存使用增长 | 监控GPU内存，实现清理 |

## 附录E：推荐阅读

### 基础论文
- "Attention Is All You Need"（Vaswani等，2017）
- "BERT: Pre-training of Deep Bidirectional Transformers"（Devlin等，2019）
- "Language Models are Few-Shot Learners"（Brown等，2020）
- "Training Compute-Optimal Large Language Models"（Hoffmann等，2022）

### 架构论文
- "FlashAttention"（Dao等，2022）
- "Mixture of Experts"（Fedus等，2021）
- "State Space Models"（Gu等，2021）

### 安全与对齐
- "Constitutional AI"（Bai等，2022）
- "Training Language Models to Follow Instructions"（Ouyang等，2022）
- "Red Teaming Language Models"（Perez等，2022）
