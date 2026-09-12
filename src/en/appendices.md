# Appendices

---

## Appendix A: Glossary of AI Architecture Terms

| English Term | 中文术语 | Definition |
|--------------|----------|------------|
| **Attention Mechanism** | 注意力机制 | Neural network component that computes weighted relationships between input tokens |
| **Transformer** | Transformer架构 | Architecture based on self-attention, enabling parallel processing of sequences |
| **KV Cache** | KV缓存 | Key-Value cache that stores intermediate attention computations for autoregressive generation |
| **Quantization** | 量化 | Reducing numerical precision of model weights (FP32 → INT8/INT4) |
| **MoE** | 混合专家 | Mixture of Experts - sparse architecture routing tokens to specialized sub-networks |
| **RAG** | 检索增强生成 | Retrieval-Augmented Generation - combining retrieval with generative models |
| **Tokenization** | 分词 | Converting text into numerical tokens for model processing |
| **Embedding** | 嵌入 | Dense vector representation of tokens or concepts |
| **Fine-tuning** | 微调 | Additional training on task-specific data to adapt a pre-trained model |
| **Inference** | 推理 | Using a trained model to generate outputs from new inputs |
| **Distillation** | 蒸馏 | Training a smaller model to mimic a larger model's behavior |
| **LoRA** | 低秩适应 | Low-Rank Adaptation - parameter-efficient fine-tuning technique |
| **RLHF** | 基于人类反馈的强化学习 | Reinforcement Learning from Human Feedback |
| **DPO** | 直接偏好优化 | Direct Preference Optimization - alternative to RLHF |
| **PagedAttention** | 分页注意力 | Memory management technique for efficient attention computation in vLLM |
| **Continuous Batching** | 持续批处理 | Dynamic batching strategy that adds/removes requests during generation |
| **Tensor Parallelism** | 张量并行 | Splitting model tensors across multiple GPUs for distributed inference |
| **Pipeline Parallelism** | 流水线并行 | Distributing model layers across multiple devices |
| **Flash Attention** | Flash注意力 | IO-aware exact attention algorithm that reduces memory usage |
| **Speculative Decoding** | 投机解码 | Using a small draft model to accelerate large model generation |
| **Prefill** | 预填充 | Processing input tokens before starting autoregressive generation |
| **Decode** | 解码 | Autoregressive token generation phase |
| **Context Window** | 上下文窗口 | Maximum number of tokens a model can process in a single forward pass |
| **Token Throughput** | Token吞吐量 | Number of tokens generated per second across all requests |
| **Latency** | 延迟 | Time from request submission to first token response (TTFT) |
| **Time to First Token (TTFT)** | 首Token延迟 | Time from request submission to first token generation |
| **Inter-Token Latency** | Token间延迟 | Time between consecutive token generations |
| **Batch Size** | 批大小 | Number of requests processed simultaneously |
| **Model Parallelism** | 模型并行 | Distributing a single model across multiple devices |
| **Data Parallelism** | 数据并行 | Replicating model across devices, splitting input data |
| **Expert Parallelism** | 专家并行 | Distributing MoE expert layers across devices |
| **Feature Store** | 特征存储 | Centralized repository for storing, accessing, and managing ML features |
| **Offline Store** | 离线存储 | Storage for historical features used in model training |
| **Online Store** | 在线存储 | Low-latency storage for real-time feature serving |
| **Feature Engineering** | 特征工程 | Process of creating, selecting, and transforming input variables for ML models |
| **Point-in-Time Correctness** | 时间点正确性 | Ensuring features are extracted at the correct historical timestamp |
| **ML Pipeline** | ML流水线 | End-to-end workflow for training, evaluating, and deploying ML models |
| **Orchestration** | 编排 | Automated coordination of complex computational workflows |
| **DAG** | 有向无环图 | Directed Acyclic Graph representing task dependencies in workflows |
| **Workflow Engine** | 工作流引擎 | System for defining, scheduling, and monitoring computational workflows |
| **Container Orchestration** | 容器编排 | Automated deployment, scaling, and management of containerized applications |
| **Kubernetes** | Kubernetes | Open-source container orchestration platform for automating deployment and scaling |
| **Helm** | Helm | Package manager for Kubernetes, simplifying application deployment |
| **Model Registry** | 模型注册表 | Centralized repository for storing, versioning, and managing ML models |
| **Experiment Tracking** | 实验跟踪 | Recording and comparing ML training runs and their parameters |
| **Hyperparameter Tuning** | 超参数调优 | Automated search for optimal model training configurations |
| **A/B Testing** | A/B测试 | Comparing model performance between two variants in production |
| **Canary Deployment** | 金丝雀部署 | Gradually rolling out model changes to a subset of traffic |
| **Shadow Deployment** | 影子部署 | Running new models alongside production models without affecting users |
| **Model Monitoring** | 模型监控 | Continuous tracking of model performance and behavior in production |
| **Data Drift** | 数据漂移 | Change in input data distribution over time |
| **Concept Drift** | 概念漂移 | Change in the relationship between input features and target variable |
| **Model Degradation** | 模型退化 | Decline in model performance over time |
| **Observability** | 可观测性 | Ability to understand system state through external outputs |
| **Metrics** | 指标 | Quantitative measurements of system performance and behavior |
| **Alerting** | 告警 | Automated notification when metrics exceed defined thresholds |
| **Tracing** | 追踪 | Recording the path of requests through distributed systems |
| **Log Aggregation** | 日志聚合 | Centralized collection and analysis of distributed system logs |
| **Vector Database** | 向量数据库 | Database optimized for storing and querying high-dimensional vectors |
| **Similarity Search** | 相似度搜索 | Finding vectors closest to a query vector in embedding space |
| **ANN** | 近似最近邻 | Approximate Nearest Neighbor - efficient similarity search algorithm |
| **HNSW** | 层次可导航小世界图 | Hierarchical Navigable Small World - graph-based similarity index |
| **IVF** | 倒排文件索引 | Inverted File Index - partitioning-based similarity index |
| **Flat Index** | 扁平索引 | Brute-force exact similarity search without indexing |
| **Metadata Filtering** | 元数据过滤 | Filtering search results based on non-vector attributes |
| **Hybrid Search** | 混合搜索 | Combining vector similarity with keyword/metadata filtering |
| **Sparse Vector** | 稀疏向量 | Vector with mostly zero values, typically from keyword models like BM25 |
| **Dense Vector** | 稠密向量 | Vector with mostly non-zero values, from neural embedding models |
| **Multi-Vector** | 多向量 | Storing multiple vectors per document for different aspects |
| **Quantization (Vector)** | 向量量化 | Reducing vector dimensionality or precision for storage/search efficiency |
| **Content Filter** | 内容过滤 | System for detecting and blocking harmful or inappropriate content |
| **Prompt Injection** | 提示注入 | Adversarial attack manipulating model behavior through crafted inputs |
| **PII Redaction** | PII脱敏 | Automatically detecting and removing personally identifiable information |
| **Guardrails** | 护栏 | Safety mechanisms controlling model inputs and outputs |
| **Hallucination** | 幻觉 | Model generating factually incorrect or fabricated information |
| **Grounding** | 基础化 | Anchoring model outputs to verifiable facts or sources |
| **Constitutional AI** | 宴会AI | AI training approach using principles to guide model behavior |
| **Red Teaming** | 红队测试 | Adversarial testing to identify model vulnerabilities |
| **Jailbreak** | 越狱 | Techniques to bypass model safety restrictions |
| **Watermarking** | 水印 | Embedding detectable signals in model outputs for provenance |
| **Edge Deployment** | 边缘部署 | Running models on edge devices with limited compute resources |
| **ONNX** | ONNX格式 | Open Neural Network Exchange - cross-platform model format |
| **Model Compression** | 模型压缩 | Techniques to reduce model size while preserving performance |
| **Pruning** | 剪枝 | Removing redundant weights or neurons from a trained model |
| **Knowledge Graph** | 知识图谱 | Structured representation of entities and their relationships |
| **Vector Embedding** | 向量嵌入 | Learned numerical representation of data in continuous vector space |
| **Semantic Search** | 语义搜索 | Search based on meaning rather than exact keyword matching |
| **Prompt Engineering** | 提示工程 | Designing inputs to elicit desired model outputs |
| **Chain-of-Thought** | 思维链 | Prompting technique that encourages step-by-step reasoning |
| **Few-Shot Learning** | 少样本学习 | Learning from a small number of examples provided in context |
| **Zero-Shot Learning** | 零样本学习 | Performing tasks without any task-specific training examples |
| **In-Context Learning** | 上下文学习 | Learning from examples provided in the prompt without weight updates |
| **Tool Calling** | 工具调用 | Model generating structured calls to external APIs or functions |
| **Function Calling** | 函数调用 | Model generating structured function invocations for external tools |
| **Structured Output** | 结构化输出 | Model generating responses in a predefined format (JSON, XML, etc.) |
| **Streaming** | 流式传输 | Sending model outputs token-by-token as they are generated |
| **Token** | Token | Basic unit of text processing in language models |
| **Vocabulary** | 词表 | Set of all tokens a model can process |
| **Positional Encoding** | 位置编码 | Mechanism to inject sequence order information into transformers |
| **Layer Normalization** | 层归一化 | Normalizing activations within each layer for training stability |
| **Residual Connection** | 残差连接 | Skip connections that add input directly to layer output |
| **Dropout** | Dropout | Regularization technique randomly disabling neurons during training |
| **Learning Rate** | 学习率 | Step size for parameter updates during optimization |
| **Gradient Descent** | 梯度下降 | Optimization algorithm iteratively adjusting parameters to minimize loss |
| **Backpropagation** | 反向传播 | Algorithm for computing gradients in neural networks |
| **Loss Function** | 损失函数 | Mathematical function measuring prediction error |
| **Cross-Entropy** | 交叉熵 | Common loss function for classification tasks |
| **Softmax** | Softmax函数 | Function converting logits to probability distributions |

---

## Appendix B: Tool Selection Guide (工具选型指南)

### B.1 Inference Engines (推理引擎)

| Tool | GitHub Stars | License | Primary Use Case | Learning Curve | Production Readiness |
|------|-------------|---------|------------------|----------------|---------------------|
| **vLLM** | 90,265+ | Apache-2.0 | High-throughput LLM serving with PagedAttention | Medium | High - used in production by major companies |
| **TensorRT-LLM** | 14,556+ | Apache-2.0 | NVIDIA GPU-optimized inference with kernel fusion | High | High - NVIDIA production-grade |
| **TGI** | 10,889+ | Apache-2.0 | HuggingFace ecosystem text generation (maintenance mode) | Low | Medium - in maintenance, recommend vLLM/SGLang |
| **SGLang** | 26,656+ | Apache-2.0 | Structured generation with RadixAttention | Medium | High - production-ready, active development |

**Key Insights:**
- **vLLM**: Best overall choice for most LLM serving. 90K+ stars, massive community, PagedAttention for memory efficiency, continuous batching, supports 200+ model architectures. Ideal for OpenAI-compatible API serving.
- **TensorRT-LLM**: Best for maximum NVIDIA GPU performance. Kernel fusion, INT4/INT8/FP8 quantization, in-flight batching. Higher complexity but best throughput on NVIDIA hardware.
- **TGI**: Now in maintenance mode. HuggingFace recommends migrating to vLLM or SGLang. Still usable but new projects should choose alternatives.
- **SGLang**: Best for structured output and complex prompting. RadixAttention for prefix caching, native support for JSON mode and constrained decoding. Growing rapidly.

### B.2 Feature Stores (特征存储)

| Tool | GitHub Stars | License | Primary Use Case | Learning Curve | Production Readiness |
|------|-------------|---------|------------------|----------------|---------------------|
| **Feast** | 7,111+ | Apache-2.0 | Open-source feature store for ML training and serving | Medium | High - 12M+ downloads, used by major companies |
| **Tecton** | N/A (Commercial) | Proprietary | Enterprise managed feature platform | Low (Managed) | High - fully managed service |
| **Hopsworks** | N/A (Open Core) | AGPL-3.0 | Full ML platform with feature store | High | High - enterprise features |

**Key Insights:**
- **Feast**: Best open-source choice. 5.5K+ Slack community, 12M+ downloads. Supports offline (batch) and online (real-time) serving. Integrates with Snowflake, BigQuery, Redis, DynamoDB. Point-in-time correct feature retrieval prevents data leakage.
- **Tecton**: Best for teams wanting fully managed. Built by Feast creators. Higher cost but zero operational overhead.
- **Hopsworks**: Best for full ML platform needs. Includes feature store, model serving, and monitoring.

### B.3 Orchestration (编排)

| Tool | GitHub Stars | License | Primary Use Case | Learning Curve | Production Readiness |
|------|-------------|---------|------------------|----------------|---------------------|
| **Kubeflow** | 33,100+ (total) | Apache-2.0 | End-to-end ML workflows on Kubernetes | High | High - CNCF Graduated project |
| **Airflow** | 46,724+ | Apache-2.0 | General workflow orchestration and scheduling | Medium | High - battle-tested at scale |
| **Ray** | 42,525+ | Apache-2.0 | Distributed AI/ML compute engine | Medium-High | High - used by major AI companies |

**Key Insights:**
- **Kubeflow**: Best for Kubernetes-native ML pipelines. CNCF Graduated (2026). Includes Training Operator, Pipelines, Katib (HPO), Notebooks. 258M+ PyPI downloads. Best for teams already on Kubernetes.
- **Airflow**: Most mature workflow orchestrator. 46K+ stars. Rich ecosystem of operators. Better for general data pipelines that include ML steps. Python DAG definition.
- **Ray**: Best for distributed computing. Ray Train for distributed training, Ray Serve for model serving, Ray Tune for HPO. Unified compute framework. Best for teams needing both training and serving in one system.

### B.4 Monitoring (监控)

| Tool | GitHub Stars | License | Primary Use Case | Learning Curve | Production Readiness |
|------|-------------|---------|------------------|----------------|---------------------|
| **Prometheus** | 65,998+ | Apache-2.0 | Time-series metrics collection and alerting | Medium | High - CNCF Graduated, industry standard |
| **Grafana** | 76,600+ | AGPL-3.0 | Metrics visualization and dashboarding | Low-Medium | High - industry standard visualization |
| **Evidently AI** | 7,786+ | Apache-2.0 | ML/LLM observability and evaluation | Low-Medium | High - 40M+ downloads, production-ready |

**Key Insights:**
- **Prometheus**: Industry standard for metrics. Pull-based model, PromQL for queries, integrates with everything. Best for infrastructure and application metrics.
- **Grafana**: Best visualization layer. Connects to Prometheus, Loki, Elasticsearch, and 100+ data sources. Rich alerting capabilities.
- **Evidently AI**: Purpose-built for ML/LLM monitoring. Data drift detection, model quality metrics, LLM evaluation (hallucinations, toxicity, relevance). 100+ built-in metrics. Best for model-specific monitoring.

### B.5 Vector Databases (向量数据库)

| Tool | GitHub Stars | License | Primary Use Case | Learning Curve | Production Readiness |
|------|-------------|---------|------------------|----------------|---------------------|
| **Chroma** | 29,230+ | Apache-2.0 | Lightweight development and prototyping | Low | Medium - good for small-medium scale |
| **Pinecone** | N/A (Managed) | Proprietary | Managed vector search, zero operations | Low | High - fully managed service |
| **Weaviate** | 16,751+ | BSD-3-Clause | Multi-modal vector search with GraphQL | Medium | High - cloud-native, production-ready |
| **Qdrant** | 34,449+ | Apache-2.0 | High-performance filtering and search | Medium | High - written in Rust, fast and reliable |
| **Milvus** | 44,588+ | Apache-2.0 | Large-scale distributed vector search | High | High - handles billions of vectors |

**Key Insights:**
- **Chroma**: Simplest to start with. `pip install chromadb`. Great for prototyping and small projects. Less suitable for large-scale production.
- **Pinecone**: Zero-ops managed service. Best for teams that want to focus on application logic, not infrastructure. Higher cost.
- **Weaviate**: Best for multi-modal search. Built-in vectorization with OpenAI, Cohere, HuggingFace. GraphQL API. Good for RAG applications.
- **Qdrant**: Best performance for filtered search. Written in Rust. Advanced filtering (nested, geo, text). Good balance of performance and features.
- **Milvus**: Best for massive scale. Handles billions of vectors with horizontal scaling. Distributed architecture. Most complex to operate.

---

## Appendix C: Architecture Design Templates (架构设计模板)

### C.1 AI System Architecture Review Template

```markdown
## AI System Architecture Review

### System Overview
- **System Name**: [Name]
- **Version**: [Version]
- **Review Date**: [Date]
- **Reviewers**: [Names]

### 1. Model Layer
- [ ] Model architecture documented
- [ ] Training data lineage tracked
- [ ] Model versioning in place
- [ ] Performance benchmarks established
- [ ] Bias/fairness evaluation completed

### 2. Inference Layer
- [ ] Serving framework selected and justified
- [ ] Resource requirements documented (GPU/CPU/memory)
- [ ] Latency SLAs defined (TTFT, inter-token)
- [ ] Throughput requirements defined (tokens/sec)
- [ ] Auto-scaling strategy defined

### 3. Data Layer
- [ ] Feature store integrated
- [ ] Data pipeline documented
- [ ] Data quality checks in place
- [ ] Privacy compliance verified (GDPR/CCPA)
- [ ] Backup and recovery plan documented

### 4. Security Layer
- [ ] Authentication and authorization implemented
- [ ] Input validation and sanitization in place
- [ ] Content safety filters configured
- [ ] PII detection and redaction enabled
- [ ] Audit logging enabled

### 5. Monitoring Layer
- [ ] Model performance metrics defined
- [ ] Data drift detection configured
- [ ] Alerting rules established
- [ ] Dashboard created for visibility
- [ ] Incident response plan documented

### 6. Operations Layer
- [ ] CI/CD pipeline configured
- [ ] Rollback strategy defined
- [ ] A/B testing framework in place
- [ ] Cost monitoring and optimization
- [ ] Disaster recovery plan tested
```

### C.2 Model Deployment Checklist

```markdown
## Model Deployment Checklist

### Pre-Deployment
- [ ] Model artifacts registered in model registry
- [ ] Model validation tests passed
- [ ] Performance benchmarks meet SLA requirements
- [ ] Security scan completed
- [ ] Resource requirements confirmed (GPU, memory, storage)

### Infrastructure
- [ ] Kubernetes cluster ready (or cloud equivalent)
- [ ] GPU nodes provisioned and tested
- [ ] Network policies configured
- [ ] Secrets and configuration injected
- [ ] Monitoring endpoints configured

### Deployment
- [ ] Canary deployment initiated (5% traffic)
- [ ] Health checks passing
- [ ] Latency metrics within threshold
- [ ] Error rate below threshold
- [ ] Memory utilization stable

### Validation
- [ ] A/B test results analyzed
- [ ] User feedback collected
- [ ] Model quality metrics stable
- [ ] No data drift detected
- [ ] Cost within budget

### Post-Deployment
- [ ] Full traffic rollout completed
- [ ] Old model version archived
- [ ] Documentation updated
- [ ] Runbook updated
- [ ] Team notified of changes
```

### C.3 MLOps Maturity Assessment

```markdown
## MLOps Maturity Assessment

Rate each capability from 1 (Initial) to 5 (Optimized):

### Level 1: Initial
| Capability | Score (1-5) | Notes |
|------------|-------------|-------|
| Manual model training | | |
| Ad-hoc deployment | | |
| No monitoring | | |
| Manual testing | | |

### Level 2: Managed
| Capability | Score (1-5) | Notes |
|------------|-------------|-------|
| Scripted training pipelines | | |
| Basic CI/CD for models | | |
| Basic logging | | |
| Automated unit tests | | |

### Level 3: Defined
| Capability | Score (1-5) | Notes |
|------------|-------------|-------|
| Automated training pipelines | | |
| Model registry in use | | |
| Centralized monitoring | | |
| Integration testing | | |
| Feature store integrated | | |

### Level 4: Quantitatively Managed
| Capability | Score (1-5) | Notes |
|------------|-------------|-------|
| Experiment tracking | | |
| Automated model validation | | |
| Data drift detection | | |
| A/B testing framework | | |
| Automated rollback | | |

### Level 5: Optimized
| Capability | Score (1-5) | Notes |
|------------|-------------|-------|
| AutoML/hyperparameter optimization | | |
| Continuous training | | |
| Predictive monitoring | | |
| Full lineage tracking | | |
| Cost optimization | | |

### Overall Maturity Level: [1-5]
### Recommended Next Steps:
1. [Priority action]
2. [Priority action]
3. [Priority action]
```

### C.4 AI Security Audit Template

```markdown
## AI Security Audit

### System Information
- **System**: [Name]
- **Audit Date**: [Date]
- **Auditor**: [Name/Team]

### 1. Input Security
- [ ] Input validation implemented
- [ ] Prompt injection protection enabled
- [ ] Input length limits enforced
- [ ] Content type validation in place
- [ ] Rate limiting configured

### 2. Model Security
- [ ] Model weights encrypted at rest
- [ ] Model access controlled (RBAC)
- [ ] Adversarial robustness tested
- [ ] Model extraction protection evaluated
- [ ] Watermarking implemented (if required)

### 3. Output Security
- [ ] Content filtering enabled
- [ ] PII detection and redaction active
- [ ] Output validation implemented
- [ ] Toxicity detection configured
- [ ] Hallucination detection in place

### 4. Data Security
- [ ] Training data access controlled
- [ ] Data encryption in transit and at rest
- [ ] Data retention policies defined
- [ ] Right to deletion supported
- [ ] Cross-border data transfer compliant

### 5. Infrastructure Security
- [ ] Network segmentation implemented
- [ ] Secrets management in place
- [ ] Container security scanning enabled
- [ ] API authentication required
- [ ] Audit logging comprehensive

### 6. Operational Security
- [ ] Incident response plan documented
- [ ] Red team testing conducted
- [ ] Penetration testing completed
- [ ] Security training for team completed
- [ ] Third-party dependencies audited

### Risk Summary
| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| [Risk 1] | High/Med/Low | High/Med/Low | [Action] |
| [Risk 2] | High/Med/Low | High/Med/Low | [Action] |

### Recommendations:
1. [Critical recommendation]
2. [Important recommendation]
3. [Nice-to-have recommendation]
```

---

## Appendix D: References (参考文献)

### D.1 Foundational Papers (基础论文)

| Paper | Authors | Year | arXiv/URL |
|-------|---------|------|-----------|
| Attention Is All You Need | Vaswani et al. | 2017 | https://arxiv.org/abs/1706.03762 |
| BERT: Pre-training of Deep Bidirectional Transformers | Devlin et al. | 2019 | https://arxiv.org/abs/1810.04805 |
| Language Models are Few-Shot Learners (GPT-3) | Brown et al. | 2020 | https://arxiv.org/abs/2005.14165 |
| Training Compute-Optimal Large Language Models (Chinchilla) | Hoffmann et al. | 2022 | https://arxiv.org/abs/2203.15556 |
| PaLM: Scaling Language Modeling with Pathways | Chowdhery et al. | 2022 | https://arxiv.org/abs/2204.02311 |
| LLaMA: Open and Efficient Foundation Language Models | Touvron et al. | 2023 | https://arxiv.org/abs/2302.13971 |

### D.2 Architecture Papers (架构论文)

| Paper | Authors | Year | arXiv/URL |
|-------|---------|------|-----------|
| FlashAttention: Fast and Memory-Efficient Exact Attention | Dao et al. | 2022 | https://arxiv.org/abs/2205.14135 |
| FlashAttention-2: Faster Attention with Better Parallelism | Dao | 2023 | https://arxiv.org/abs/2307.08691 |
| Mixtral of Experts | Jiang et al. | 2024 | https://arxiv.org/abs/2401.04088 |
| DeepSeek-V3 Technical Report | DeepSeek-AI | 2024 | https://arxiv.org/abs/2412.19437 |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | Kwon et al. | 2023 | https://arxiv.org/abs/2309.06180 |
| SGLang: Efficient Execution of Structured Language Model Programs | Zheng et al. | 2024 | https://arxiv.org/abs/2312.07104 |

### D.3 Safety and Alignment (安全与对齐)

| Paper | Authors | Year | arXiv/URL |
|-------|---------|------|-----------|
| Constitutional AI: Harmlessness from AI Feedback | Bai et al. | 2022 | https://arxiv.org/abs/2212.08073 |
| Training Language Models to Follow Instructions with Human Feedback | Ouyang et al. | 2022 | https://arxiv.org/abs/2203.02155 |
| Red Teaming Language Models to Reduce Harms | Perez et al. | 2022 | https://arxiv.org/abs/2209.07858 |
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafailov et al. | 2023 | https://arxiv.org/abs/2305.18290 |
| Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations | Inan et al. | 2023 | https://arxiv.org/abs/2312.06674 |

### D.4 Retrieval and RAG (检索与RAG)

| Paper | Authors | Year | arXiv/URL |
|-------|---------|------|-----------|
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis et al. | 2020 | https://arxiv.org/abs/2005.11401 |
| REPLUG: Retrieval-Augmented Black-Box Language Models | Shi et al. | 2023 | https://arxiv.org/abs/2301.12652 |
| Self-RAG: Learning to Retrieve, Generate, and Critique | Asai et al. | 2023 | https://arxiv.org/abs/2310.11511 |

### D.5 Quantization and Efficiency (量化与效率)

| Paper | Authors | Year | arXiv/URL |
|-------|---------|------|-----------|
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Frantar et al. | 2022 | https://arxiv.org/abs/2210.17323 |
| QLoRA: Efficient Finetuning of Quantized LLMs | Dettmers et al. | 2023 | https://arxiv.org/abs/2305.14314 |
| LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | Dettmers et al. | 2022 | https://arxiv.org/abs/2208.07339 |

### D.6 Official Documentation (官方文档)

| Resource | URL |
|----------|-----|
| vLLM Documentation | https://docs.vllm.ai/ |
| TensorRT-LLM Documentation | https://nvidia.github.io/TensorRT-LLM/ |
| SGLang Documentation | https://sgl-project.github.io/ |
| Feast Documentation | https://docs.feast.dev/ |
| Kubeflow Documentation | https://www.kubeflow.org/docs/ |
| Apache Airflow Documentation | https://airflow.apache.org/docs/ |
| Ray Documentation | https://docs.ray.io/ |
| Prometheus Documentation | https://prometheus.io/docs/ |
| Grafana Documentation | https://grafana.com/docs/ |
| Evidently AI Documentation | https://docs.evidentlyai.com/ |
| Chroma Documentation | https://docs.trychroma.com/ |
| Weaviate Documentation | https://weaviate.io/developers/weaviate |
| Qdrant Documentation | https://qdrant.tech/documentation/ |
| Milvus Documentation | https://milvus.io/docs |

### D.7 Engineering Blog Posts (工程博客)

| Title | Source | URL |
|-------|--------|-----|
| vLLM: Easy, Fast, and Cheap LLM Serving | vLLM Blog | https://blog.vllm.ai/ |
| SGLang: Fast Structured Generation | SGLang Blog | https://sgl-project.github.io/ |
| Building LLM Applications at Scale | Ray Blog | https://www.ray.io/blog |
| MLOps Best Practices | Kubeflow Blog | https://www.kubeflow.org/blog/ |
| ML Observability Guide | Evidently AI Blog | https://www.evidentlyai.com/blog |

### D.8 Books (书籍)

| Title | Authors | ISBN | Publisher |
|-------|---------|------|-----------|
| Designing Machine Learning Systems | Chip Huyen | 978-1098107963 | O'Reilly Media |
| Machine Learning Engineering | Andriy Burkov | 978-1735765449 | True Positive |
| Reliable Machine Learning | Kakushadze, Shukla | 978-1735765456 | COCONUT |
| Building Machine Learning Pipelines | Hannes Hapke, Catherine Nelson | 978-1492053194 | O'Reilly Media |
| Distributed Machine Learning with Spark | Muhammad Asif Iqbal | 978-1800208100 | Packt |

---

## Appendix E: Open Source Project Index (开源项目索引)

| Project | GitHub URL | License | Stars | Primary Use Case |
|---------|-----------|---------|-------|------------------|
| **vLLM** | https://github.com/vllm-project/vllm | Apache-2.0 | 90,265+ | High-throughput LLM inference and serving |
| **TensorRT-LLM** | https://github.com/NVIDIA/TensorRT-LLM | Apache-2.0 | 14,556+ | NVIDIA GPU-optimized LLM inference |
| **TGI** | https://github.com/huggingface/text-generation-inference | Apache-2.0 | 10,889+ | HuggingFace text generation (maintenance mode) |
| **SGLang** | https://github.com/sgl-project/sglang | Apache-2.0 | 26,656+ | Structured generation and LLM serving |
| **Feast** | https://github.com/feast-dev/feast | Apache-2.0 | 7,111+ | ML feature store for training and serving |
| **Kubeflow** | https://github.com/kubeflow/kubeflow | Apache-2.0 | 15,846+ | ML toolkit for Kubernetes |
| **Kubeflow Pipelines** | https://github.com/kubeflow/pipelines | Apache-2.0 | 4,198+ | ML workflow orchestration |
| **Apache Airflow** | https://github.com/apache/airflow | Apache-2.0 | 46,724+ | Workflow orchestration and scheduling |
| **Ray** | https://github.com/ray-project/ray | Apache-2.0 | 42,525+ | Distributed AI/ML compute engine |
| **Prometheus** | https://github.com/prometheus/prometheus | Apache-2.0 | 65,998+ | Time-series metrics and monitoring |
| **Grafana** | https://github.com/grafana/grafana | AGPL-3.0 | 76,600+ | Metrics visualization and dashboards |
| **Evidently AI** | https://github.com/evidentlyai/evidently | Apache-2.0 | 7,786+ | ML/LLM observability and evaluation |
| **Chroma** | https://github.com/chroma-core/chroma | Apache-2.0 | 29,230+ | Vector database for AI applications |
| **Weaviate** | https://github.com/weaviate/weaviate | BSD-3-Clause | 16,751+ | Multi-modal vector database |
| **Qdrant** | https://github.com/qdrant/qdrant | Apache-2.0 | 34,449+ | High-performance vector search |
| **Milvus** | https://github.com/milvus-io/milvus | Apache-2.0 | 44,588+ | Distributed vector database at scale |
| **llama.cpp** | https://github.com/ggerganov/llama.cpp | MIT | 70,000+ | CPU/GPU inference with GGUF quantization |
| **Ollama** | https://github.com/ollama/ollama | MIT | 100,000+ | Local LLM management and serving |
| **Hugging Face Transformers** | https://github.com/huggingface/transformers | Apache-2.0 | 140,000+ | Pre-trained model library |
| **LangChain** | https://github.com/langchain-ai/langchain | MIT | 95,000+ | LLM application framework |
| **LlamaIndex** | https://github.com/run-llama/llama_index | MIT | 35,000+ | Data framework for LLM applications |
| **OpenTelemetry** | https://github.com/open-telemetry/opentelemetry-java | Apache-2.0 | 2,500+ | Distributed tracing and observability |

*Note: Star counts are approximate and based on data collected in September 2026.*
