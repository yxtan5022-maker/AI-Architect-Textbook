# 附录

---

## 附录A：AI架构术语表

| 英文术语 | 中文术语 | 定义 |
|----------|----------|------|
| **Attention Mechanism** | 注意力机制 | 计算输入token之间加权关系的神经网络组件 |
| **Transformer** | Transformer架构 | 基于自注意力的架构，支持序列并行处理 |
| **KV Cache** | KV缓存 | 存储自回归生成中间注意力计算的键值缓存 |
| **Quantization** | 量化 | 降低模型权重数值精度（FP32 → INT8/INT4） |
| **MoE** | 混合专家 | 将token路由到专门子网络的稀疏架构 |
| **RAG** | 检索增强生成 | 将检索与生成模型结合 |
| **Tokenization** | 分词 | 将文本转换为模型处理的数值token |
| **Embedding** | 嵌入 | token或概念的密集向量表示 |
| **Fine-tuning** | 微调 | 在特定任务数据上额外训练以适应预训练模型 |
| **Inference** | 推理 | 使用训练好的模型从新输入生成输出 |
| **Distillation** | 蒸馏 | 训练较小模型模仿较大模型行为 |
| **LoRA** | 低秩适应 | 参数高效微调技术 |
| **RLHF** | 基于人类反馈的强化学习 | 人类反馈强化学习 |
| **DPO** | 直接偏好优化 | RLHF的替代方案 |
| **PagedAttention** | 分页注意力 | vLLM中高效注意力计算的内存管理技术 |
| **Continuous Batching** | 持续批处理 | 在生成过程中动态添加/移除请求的批处理策略 |
| **Tensor Parallelism** | 张量并行 | 将模型张量分割到多个GPU上进行分布式推理 |
| **Pipeline Parallelism** | 流水线并行 | 将模型层分布到多个设备上 |
| **Flash Attention** | Flash注意力 | 减少内存使用的IO感知精确注意力算法 |
| **Speculative Decoding** | 投机解码 | 使用小型草稿模型加速大型模型生成 |
| **Prefill** | 预填充 | 在开始自回归生成之前处理输入token |
| **Decode** | 解码 | 自回归token生成阶段 |
| **Context Window** | 上下文窗口 | 模型单次前向传播可处理的最大token数 |
| **Token Throughput** | Token吞吐量 | 所有请求中每秒生成的token数 |
| **Latency** | 延迟 | 从请求提交到首个token响应的时间（TTFT） |
| **Time to First Token (TTFT)** | 首Token延迟 | 从请求提交到首个token生成的时间 |
| **Inter-Token Latency** | Token间延迟 | 连续token生成之间的时间 |
| **Batch Size** | 批大小 | 同时处理的请求数量 |
| **Model Parallelism** | 模型并行 | 将单个模型分布到多个设备上 |
| **Data Parallelism** | 数据并行 | 跨设备复制模型，分割输入数据 |
| **Expert Parallelism** | 专家并行 | 将MoE专家层分布到设备上 |
| **Feature Store** | 特征存储 | 集中存储、访问和管理ML特征的仓库 |
| **Offline Store** | 离线存储 | 用于模型训练的历史特征存储 |
| **Online Store** | 在线存储 | 用于实时特征服务的低延迟存储 |
| **Feature Engineering** | 特征工程 | 为ML模型创建、选择和转换输入变量的过程 |
| **Point-in-Time Correctness** | 时间点正确性 | 确保特征在正确的历史时间戳提取 |
| **ML Pipeline** | ML流水线 | 模型训练、评估和部署的端到端工作流 |
| **Orchestration** | 编排 | 复杂计算工作流的自动协调 |
| **DAG** | 有向无环图 | 表示工作流中任务依赖关系的有向无环图 |
| **Workflow Engine** | 工作流引擎 | 定义、调度和监控计算工作流的系统 |
| **Container Orchestration** | 容器编排 | 容器化应用的自动部署、扩展和管理 |
| **Kubernetes** | Kubernetes | 自动化部署和扩展的开源容器编排平台 |
| **Helm** | Helm | Kubernetes包管理器，简化应用部署 |
| **Model Registry** | 模型注册表 | 存储、版本控制和管理ML模型的集中仓库 |
| **Experiment Tracking** | 实验跟踪 | 记录和比较ML训练运行及其参数 |
| **Hyperparameter Tuning** | 超参数调优 | 自动搜索最优模型训练配置 |
| **A/B Testing** | A/B测试 | 在生产中比较两个变体的模型性能 |
| **Canary Deployment** | 金丝雀部署 | 逐步向部分流量推出模型变更 |
| **Shadow Deployment** | 影子部署 | 在不影响用户的情况下运行新模型 |
| **Model Monitoring** | 模型监控 | 持续跟踪生产中模型性能和行为 |
| **Data Drift** | 数据漂移 | 输入数据分布随时间变化 |
| **Concept Drift** | 概念漂移 | 输入特征与目标变量之间关系的变化 |
| **Model Degradation** | 模型退化 | 模型性能随时间下降 |
| **Observability** | 可观测性 | 通过外部输出理解系统状态的能力 |
| **Metrics** | 指标 | 系统性能和行为的定量测量 |
| **Alerting** | 告警 | 指标超过定义阈值时的自动通知 |
| **Tracing** | 追踪 | 记录请求在分布式系统中的路径 |
| **Log Aggregation** | 日志聚合 | 分布式系统日志的集中收集和分析 |
| **Vector Database** | 向量数据库 | 优化存储和查询高维向量的数据库 |
| **Similarity Search** | 相似度搜索 | 在嵌入空间中找到最接近查询向量的向量 |
| **ANN** | 近似最近邻 | 高效相似度搜索算法 |
| **HNSW** | 层次可导航小世界图 | 基于图的相似度索引 |
| **IVF** | 倒排文件索引 | 基于分区的相似度索引 |
| **Flat Index** | 扁平索引 | 无索引的暴力精确相似度搜索 |
| **Metadata Filtering** | 元数据过滤 | 基于非向量属性过滤搜索结果 |
| **Hybrid Search** | 混合搜索 | 结合向量相似度和关键字/元数据过滤 |
| **Sparse Vector** | 稀疏向量 | 大多数值为零的向量，通常来自BM25等关键字模型 |
| **Dense Vector** | 稠密向量 | 大多数值非零的向量，来自神经嵌入模型 |
| **Multi-Vector** | 多向量 | 为每个文档存储多个向量以表示不同方面 |
| **Quantization (Vector)** | 向量量化 | 降低向量维度或精度以提高存储/搜索效率 |
| **Content Filter** | 内容过滤 | 检测和阻止有害或不当内容的系统 |
| **Prompt Injection** | 提示注入 | 通过精心构造的输入操纵模型行为的对抗性攻击 |
| **PII Redaction** | PII脱敏 | 自动检测和移除个人身份信息 |
| **Guardrails** | 护栏 | 控制模型输入和输出的安全机制 |
| **Hallucination** | 幻觉 | 模型生成事实不正确或捏造的信息 |
| **Grounding** | 基础化 | 将模型输出锚定到可验证的事实或来源 |
| **Constitutional AI** | 宴会AI | 使用原则指导模型行为的AI训练方法 |
| **Red Teaming** | 红队测试 | 识别模型漏洞的对抗性测试 |
| **Jailbreak** | 越狱 | 绕过模型安全限制的技术 |
| **Watermarking** | 水印 | 在模型输出中嵌入可检测信号以追溯来源 |
| **Edge Deployment** | 边缘部署 | 在计算资源有限的边缘设备上运行模型 |
| **ONNX** | ONNX格式 | 开放神经网络交换格式 - 跨平台模型格式 |
| **Model Compression** | 模型压缩 | 在保持性能的同时减小模型大小的技术 |
| **Pruning** | 剪枝 | 从训练好的模型中移除冗余权重或神经元 |
| **Knowledge Graph** | 知识图谱 | 实体及其关系的结构化表示 |
| **Vector Embedding** | 向量嵌入 | 数据在连续向量空间中的学习数值表示 |
| **Semantic Search** | 语义搜索 | 基于含义而非精确关键字匹配的搜索 |
| **Prompt Engineering** | 提示工程 | 设计输入以引出期望的模型输出 |
| **Chain-of-Thought** | 思维链 | 鼓励逐步推理的提示技术 |
| **Few-Shot Learning** | 少样本学习 | 从上下文中提供的少量示例学习 |
| **Zero-Shot Learning** | 零样本学习 | 在没有任何特定任务训练示例的情况下执行任务 |
| **In-Context Learning** | 上下文学习 | 从提示中提供的示例学习，无需权重更新 |
| **Tool Calling** | 工具调用 | 模型生成对外部API或函数的结构化调用 |
| **Function Calling** | 函数调用 | 模型生成对外部工具的结构化函数调用 |
| **Structured Output** | 结构化输出 | 模型按预定义格式（JSON、XML等）生成响应 |
| **Streaming** | 流式传输 | 在生成时逐token发送模型输出 |
| **Token** | Token | 语言模型中文本处理的基本单位 |
| **Vocabulary** | 词表 | 模型可以处理的所有token的集合 |
| **Positional Encoding** | 位置编码 | 将序列顺序信息注入transformer的机制 |
| **Layer Normalization** | 层归一化 | 归一化每层内的激活以提高训练稳定性 |
| **Residual Connection** | 残差连接 | 将输入直接加到层输出的跳跃连接 |
| **Dropout** | Dropout | 训练期间随机禁用神经元的正则化技术 |
| **Learning Rate** | 学习率 | 优化期间参数更新的步长 |
| **Gradient Descent** | 梯度下降 | 迭代调整参数以最小化损失的优化算法 |
| **Backpropagation** | 反向传播 | 在神经网络中计算梯度的算法 |
| **Loss Function** | 损失函数 | 测量预测误差的数学函数 |
| **Cross-Entropy** | 交叉熵 | 分类任务的常用损失函数 |
| **Softmax** | Softmax函数 | 将logits转换为概率分布的函数 |

---

## 附录B：工具选型指南

### B.1 推理引擎

| 工具 | GitHub Stars | 许可证 | 主要用途 | 学习曲线 | 生产就绪度 |
|------|-------------|--------|----------|----------|------------|
| **vLLM** | 90,265+ | Apache-2.0 | 高吞吐LLM服务，PagedAttention | 中等 | 高 - 被主要公司用于生产 |
| **TensorRT-LLM** | 14,556+ | Apache-2.0 | NVIDIA GPU优化推理，内核融合 | 高 | 高 - NVIDIA生产级 |
| **TGI** | 10,889+ | Apache-2.0 | HuggingFace生态文本生成（维护模式） | 低 | 中等 - 维护中，推荐vLLM/SGLang |
| **SGLang** | 26,656+ | Apache-2.0 | 结构化生成，RadixAttention | 中等 | 高 - 生产就绪，活跃开发 |

**关键洞察：**
- **vLLM**：大多数LLM服务的最佳选择。9万+星，庞大社区，PagedAttention内存效率高，持续批处理，支持200+模型架构。适合OpenAI兼容API服务。
- **TensorRT-LLM**：NVIDIA GPU最佳性能。内核融合，INT4/INT8/FP8量化，飞行中批处理。复杂度较高但在NVIDIA硬件上吞吐量最佳。
- **TGI**：现处于维护模式。HuggingFace建议迁移到vLLM或SGLang。仍可使用但新项目应选择替代方案。
- **SGLang**：结构化输出和复杂提示的最佳选择。RadixAttention前缀缓存，原生JSON模式和约束解码支持。快速增长。

### B.2 特征存储

| 工具 | GitHub Stars | 许可证 | 主要用途 | 学习曲线 | 生产就绪度 |
|------|-------------|--------|----------|----------|------------|
| **Feast** | 7,111+ | Apache-2.0 | ML训练和服务的开源特征存储 | 中等 | 高 - 1200万+下载，主要公司使用 |
| **Tecton** | N/A（商业） | 专有 | 企业级托管特征平台 | 低（托管） | 高 - 完全托管服务 |
| **Hopsworks** | N/A（开放核心） | AGPL-3.0 | 带特征存储的完整ML平台 | 高 | 高 - 企业功能 |

**关键洞察：**
- **Feast**：最佳开源选择。5500+ Slack社区，1200万+下载。支持离线（批量）和在线（实时）服务。与Snowflake、BigQuery、Redis、DynamoDB集成。时间点正确的特征检索防止数据泄漏。
- **Tecton**：适合想要完全托管的团队。由Feast创建者构建。成本较高但零运维开销。
- **Hopsworks**：适合完整ML平台需求。包括特征存储、模型服务和监控。

### B.3 编排

| 工具 | GitHub Stars | 许可证 | 主要用途 | 学习曲线 | 生产就绪度 |
|------|-------------|--------|----------|----------|------------|
| **Kubeflow** | 33,100+（总计） | Apache-2.0 | Kubernetes上端到端ML工作流 | 高 | 高 - CNCF毕业项目 |
| **Airflow** | 46,724+ | Apache-2.0 | 通用工作流编排和调度 | 中等 | 高 - 大规模实战检验 |
| **Ray** | 42,525+ | Apache-2.0 | 分布式AI/ML计算引擎 | 中高 | 高 - 被主要AI公司使用 |

**关键洞察：**
- **Kubeflow**：最适合Kubernetes原生ML流水线。CNCF毕业（2026）。包括Training Operator、Pipelines、Katib（HPO）、Notebooks。2.58亿+PyPI下载。最适合已在Kubernetes上的团队。
- **Airflow**：最成熟的工作流编排器。4万+星。丰富的运营商生态系统。更适合包含ML步骤的通用数据流水线。Python DAG定义。
- **Ray**：最适合分布式计算。Ray Train用于分布式训练，Ray Serve用于模型服务，Ray Tune用于HPO。统一计算框架。最适合需要在一个系统中同时进行训练和服务的团队。

### B.4 监控

| 工具 | GitHub Stars | 许可证 | 主要用途 | 学习曲线 | 生产就绪度 |
|------|-------------|--------|----------|----------|------------|
| **Prometheus** | 65,998+ | Apache-2.0 | 时间序列指标收集和告警 | 中等 | 高 - CNCF毕业，行业标准 |
| **Grafana** | 76,600+ | AGPL-3.0 | 指标可视化和仪表板 | 低-中 | 高 - 行业标准可视化 |
| **Evidently AI** | 7,786+ | Apache-2.0 | ML/LLM可观测性和评估 | 低-中 | 高 - 4000万+下载，生产就绪 |

**关键洞察：**
- **Prometheus**：指标行业标准。基于拉取的模型，PromQL查询，与所有东西集成。最适合基础设施和应用指标。
- **Grafana**：最佳可视化层。连接Prometheus、Loki、Elasticsearch和100+数据源。丰富的告警功能。
- **Evidently AI**：专为ML/LLM监控构建。数据漂移检测、模型质量指标、LLM评估（幻觉、毒性、相关性）。100+内置指标。最适合模型特定监控。

### B.5 向量数据库

| 工具 | GitHub Stars | 许可证 | 主要用途 | 学习曲线 | 生产就绪度 |
|------|-------------|--------|----------|----------|------------|
| **Chroma** | 29,230+ | Apache-2.0 | 轻量级开发和原型设计 | 低 | 中等 - 适合中小规模 |
| **Pinecone** | N/A（托管） | 专有 | 托管向量搜索，零运维 | 低 | 高 - 完全托管服务 |
| **Weaviate** | 16,751+ | BSD-3-Clause | 多模态向量搜索，GraphQL | 中等 | 高 - 云原生，生产就绪 |
| **Qdrant** | 34,449+ | Apache-2.0 | 高性能过滤和搜索 | 中等 | 高 - Rust编写，快速可靠 |
| **Milvus** | 44,588+ | Apache-2.0 | 大规模分布式向量搜索 | 高 | 高 - 处理数十亿向量 |

**关键洞察：**
- **Chroma**：最简单上手。`pip install chromadb`。适合原型设计和小型项目。不太适合大规模生产。
- **Pinecone**：零运维托管服务。最适合希望专注于应用逻辑而非基础设施的团队。成本较高。
- **Weaviate**：最适合多模态搜索。内置OpenAI、Cohere、HuggingFace向量化。GraphQL API。适合RAG应用。
- **Qdrant**：过滤搜索最佳性能。Rust编写。高级过滤（嵌套、地理、文本）。性能和功能的良好平衡。
- **Milvus**：最适合大规模。处理数十亿向量，水平扩展。分布式架构。最复杂的操作。

---

## 附录C：架构设计模板

### C.1 AI系统架构审查模板

```markdown
## AI系统架构审查

### 系统概述
- **系统名称**：[名称]
- **版本**：[版本]
- **审查日期**：[日期]
- **审查人员**：[姓名]

### 1. 模型层
- [ ] 模型架构已文档化
- [ ] 训练数据血缘已追踪
- [ ] 模型版本控制已就位
- [ ] 性能基准已建立
- [ ] 偏见/公平性评估已完成

### 2. 推理层
- [ ] 服务框架已选择并说明理由
- [ ] 资源需求已文档化（GPU/CPU/内存）
- [ ] 延迟SLA已定义（TTFT、token间）
- [ ] 吞吐量需求已定义（token/秒）
- [ ] 自动扩展策略已定义

### 3. 数据层
- [ ] 特征存储已集成
- [ ] 数据流水线已文档化
- [ ] 数据质量检查已就位
- [ ] 隐私合规性已验证（GDPR/CCPA）
- [ ] 备份和恢复计划已文档化

### 4. 安全层
- [ ] 认证和授权已实现
- [ ] 输入验证和清理已就位
- [ ] 内容安全过滤器已配置
- [ ] PII检测和脱敏已启用
- [ ] 审计日志已启用

### 5. 监控层
- [ ] 模型性能指标已定义
- [ ] 数据漂移检测已配置
- [ ] 告警规则已建立
- [ ] 仪表板已创建用于可视化
- [ ] 事件响应计划已文档化

### 6. 运维层
- [ ] CI/CD流水线已配置
- [ ] 回滚策略已定义
- [ ] A/B测试框架已就位
- [ ] 成本监控和优化已实施
- [ ] 灾难恢复计划已测试
```

### C.2 模型部署检查清单

```markdown
## 模型部署检查清单

### 部署前
- [ ] 模型工件已注册到模型注册表
- [ ] 模型验证测试已通过
- [ ] 性能基准满足SLA要求
- [ ] 安全扫描已完成
- [ ] 资源需求已确认（GPU、内存、存储）

### 基础设施
- [ ] Kubernetes集群已就绪（或等效云服务）
- [ ] GPU节点已配置并测试
- [ ] 网络策略已配置
- [ ] 密钥和配置已注入
- [ ] 监控端点已配置

### 部署
- [ ] 金丝雀部署已启动（5%流量）
- [ ] 健康检查通过
- [ ] 延迟指标在阈值内
- [ ] 错误率低于阈值
- [ ] 内存使用稳定

### 验证
- [ ] A/B测试结果已分析
- [ ] 用户反馈已收集
- [ ] 模型质量指标稳定
- [ ] 未检测到数据漂移
- [ ] 成本在预算内

### 部署后
- [ ] 全流量推出已完成
- [ ] 旧模型版本已归档
- [ ] 文档已更新
- [ ] 运行手册已更新
- [ ] 团队已通知变更
```

### C.3 MLOps成熟度评估

```markdown
## MLOps成熟度评估

将每个能力从1（初始）到5（优化）评分：

### 1级：初始
| 能力 | 评分(1-5) | 备注 |
|------|-----------|------|
| 手动模型训练 | | |
| 临时部署 | | |
| 无监控 | | |
| 手动测试 | | |

### 2级：托管
| 能力 | 评分(1-5) | 备注 |
|------|-----------|------|
| 脚本化训练流水线 | | |
| 基础CI/CD模型 | | |
| 基础日志记录 | | |
| 自动化单元测试 | | |

### 3级：已定义
| 能力 | 评分(1-5) | 备注 |
|------|-----------|------|
| 自动化训练流水线 | | |
| 模型注册表使用 | | |
| 集中式监控 | | |
| 集成测试 | | |
| 特征存储集成 | | |

### 4级：量化管理
| 能力 | 评分(1-5) | 备注 |
|------|-----------|------|
| 实验跟踪 | | |
| 自动化模型验证 | | |
| 数据漂移检测 | | |
| A/B测试框架 | | |
| 自动化回滚 | | |

### 5级：优化
| 能力 | 评分(1-5) | 备注 |
|------|-----------|------|
| AutoML/超参数优化 | | |
| 持续训练 | | |
| 预测性监控 | | |
| 完整血缘追踪 | | |
| 成本优化 | | |

### 整体成熟度级别：[1-5]
### 建议的下一步：
1. [优先行动]
2. [优先行动]
3. [优先行动]
```

### C.4 AI安全审计模板

```markdown
## AI安全审计

### 系统信息
- **系统**：[名称]
- **审计日期**：[日期]
- **审计员**：[姓名/团队]

### 1. 输入安全
- [ ] 输入验证已实现
- [ ] 提示注入防护已启用
- [ ] 输入长度限制已强制执行
- [ ] 内容类型验证已就位
- [ ] 速率限制已配置

### 2. 模型安全
- [ ] 模型权重静态加密
- [ ] 模型访问已控制（RBAC）
- [ ] 对抗鲁棒性已测试
- [ ] 模型提取防护已评估
- [ ] 水印已实现（如需要）

### 3. 输出安全
- [ ] 内容过滤已启用
- [ ] PII检测和脱敏已激活
- [ ] 输出验证已实现
- [ ] 毒性检测已配置
- [ ] 幻觉检测已就位

### 4. 数据安全
- [ ] 训练数据访问已控制
- [ ] 数据传输和静态加密已实施
- [ ] 数据保留策略已定义
- [ ] 删除权已支持
- [ ] 跨境数据传输合规

### 5. 基础设施安全
- [ ] 网络分段已实现
- [ ] 密钥管理已就位
- [ ] 容器安全扫描已启用
- [ ] API认证已要求
- [ ] 审计日志已全面

### 6. 运维安全
- [ ] 事件响应计划已文档化
- [ ] 红队测试已进行
- [ ] 渗透测试已完成
- [ ] 团队安全培训已完成
- [ ] 第三方依赖已审计

### 风险摘要
| 风险 | 严重性 | 可能性 | 缓解措施 |
|------|--------|--------|----------|
| [风险1] | 高/中/低 | 高/中/低 | [行动] |
| [风险2] | 高/中/低 | 高/中/低 | [行动] |

### 建议：
1. [关键建议]
2. [重要建议]
3. [可选建议]
```

---

## 附录D：参考文献

### D.1 基础论文

| 论文 | 作者 | 年份 | arXiv/URL |
|------|------|------|-----------|
| Attention Is All You Need | Vaswani等 | 2017 | https://arxiv.org/abs/1706.03762 |
| BERT: Pre-training of Deep Bidirectional Transformers | Devlin等 | 2019 | https://arxiv.org/abs/1810.04805 |
| Language Models are Few-Shot Learners (GPT-3) | Brown等 | 2020 | https://arxiv.org/abs/2005.14165 |
| Training Compute-Optimal Large Language Models (Chinchilla) | Hoffmann等 | 2022 | https://arxiv.org/abs/2203.15556 |
| PaLM: Scaling Language Modeling with Pathways | Chowdhery等 | 2022 | https://arxiv.org/abs/2204.02311 |
| LLaMA: Open and Efficient Foundation Language Models | Touvron等 | 2023 | https://arxiv.org/abs/2302.13971 |

### D.2 架构论文

| 论文 | 作者 | 年份 | arXiv/URL |
|------|------|------|-----------|
| FlashAttention: Fast and Memory-Efficient Exact Attention | Dao等 | 2022 | https://arxiv.org/abs/2205.14135 |
| FlashAttention-2: Faster Attention with Better Parallelism | Dao | 2023 | https://arxiv.org/abs/2307.08691 |
| Mixtral of Experts | Jiang等 | 2024 | https://arxiv.org/abs/2401.04088 |
| DeepSeek-V3 Technical Report | DeepSeek-AI | 2024 | https://arxiv.org/abs/2412.19437 |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | Kwon等 | 2023 | https://arxiv.org/abs/2309.06180 |
| SGLang: Efficient Execution of Structured Language Model Programs | Zheng等 | 2024 | https://arxiv.org/abs/2312.07104 |

### D.3 安全与对齐

| 论文 | 作者 | 年份 | arXiv/URL |
|------|------|------|-----------|
| Constitutional AI: Harmlessness from AI Feedback | Bai等 | 2022 | https://arxiv.org/abs/2212.08073 |
| Training Language Models to Follow Instructions with Human Feedback | Ouyang等 | 2022 | https://arxiv.org/abs/2203.02155 |
| Red Teaming Language Models to Reduce Harms | Perez等 | 2022 | https://arxiv.org/abs/2209.07858 |
| Direct Preference Optimization | Rafailov等 | 2023 | https://arxiv.org/abs/2305.18290 |
| Llama Guard: LLM-based Input-Output Safeguard | Inan等 | 2023 | https://arxiv.org/abs/2312.06674 |

### D.4 检索与RAG

| 论文 | 作者 | 年份 | arXiv/URL |
|------|------|------|-----------|
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis等 | 2020 | https://arxiv.org/abs/2005.11401 |
| REPLUG: Retrieval-Augmented Black-Box Language Models | Shi等 | 2023 | https://arxiv.org/abs/2301.12652 |
| Self-RAG: Learning to Retrieve, Generate, and Critique | Asai等 | 2023 | https://arxiv.org/abs/2310.11511 |

### D.5 量化与效率

| 论文 | 作者 | 年份 | arXiv/URL |
|------|------|------|-----------|
| GPTQ: Accurate Post-Training Quantization | Frantar等 | 2022 | https://arxiv.org/abs/2210.17323 |
| QLoRA: Efficient Finetuning of Quantized LLMs | Dettmers等 | 2023 | https://arxiv.org/abs/2305.14314 |
| LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | Dettmers等 | 2022 | https://arxiv.org/abs/2208.07339 |

### D.6 官方文档

| 资源 | URL |
|------|-----|
| vLLM文档 | https://docs.vllm.ai/ |
| TensorRT-LLM文档 | https://nvidia.github.io/TensorRT-LLM/ |
| SGLang文档 | https://sgl-project.github.io/ |
| Feast文档 | https://docs.feast.dev/ |
| Kubeflow文档 | https://www.kubeflow.org/docs/ |
| Apache Airflow文档 | https://airflow.apache.org/docs/ |
| Ray文档 | https://docs.ray.io/ |
| Prometheus文档 | https://prometheus.io/docs/ |
| Grafana文档 | https://grafana.com/docs/ |
| Evidently AI文档 | https://docs.evidentlyai.com/ |
| Chroma文档 | https://docs.trychroma.com/ |
| Weaviate文档 | https://weaviate.io/developers/weaviate |
| Qdrant文档 | https://qdrant.tech/documentation/ |
| Milvus文档 | https://milvus.io/docs |

### D.7 工程博客

| 标题 | 来源 | URL |
|------|------|-----|
| vLLM: Easy, Fast, and Cheap LLM Serving | vLLM博客 | https://blog.vllm.ai/ |
| SGLang: Fast Structured Generation | SGLang博客 | https://sgl-project.github.io/ |
| Building LLM Applications at Scale | Ray博客 | https://www.ray.io/blog |
| MLOps Best Practices | Kubeflow博客 | https://www.kubeflow.org/blog/ |
| ML Observability Guide | Evidently AI博客 | https://www.evidentlyai.com/blog |

### D.8 书籍

| 标题 | 作者 | ISBN | 出版社 |
|------|------|------|--------|
| Designing Machine Learning Systems | Chip Huyen | 978-1098107963 | O'Reilly Media |
| Machine Learning Engineering | Andriy Burkov | 978-1735765449 | True Positive |
| Reliable Machine Learning | Kakushadze, Shukla | 978-1735765456 | COCONUT |
| Building Machine Learning Pipelines | Hannes Hapke, Catherine Nelson | 978-1492053194 | O'Reilly Media |
| Distributed Machine Learning with Spark | Muhammad Asif Iqbal | 978-1800208100 | Packt |

---

## 附录E：开源项目索引

| 项目 | GitHub URL | 许可证 | Stars | 主要用途 |
|------|-----------|--------|-------|----------|
| **vLLM** | https://github.com/vllm-project/vllm | Apache-2.0 | 90,265+ | 高吞吐LLM推理和服务 |
| **TensorRT-LLM** | https://github.com/NVIDIA/TensorRT-LLM | Apache-2.0 | 14,556+ | NVIDIA GPU优化LLM推理 |
| **TGI** | https://github.com/huggingface/text-generation-inference | Apache-2.0 | 10,889+ | HuggingFace文本生成（维护模式） |
| **SGLang** | https://github.com/sgl-project/sglang | Apache-2.0 | 26,656+ | 结构化生成和LLM服务 |
| **Feast** | https://github.com/feast-dev/feast | Apache-2.0 | 7,111+ | ML训练和服务特征存储 |
| **Kubeflow** | https://github.com/kubeflow/kubeflow | Apache-2.0 | 15,846+ | Kubernetes ML工具包 |
| **Kubeflow Pipelines** | https://github.com/kubeflow/pipelines | Apache-2.0 | 4,198+ | ML工作流编排 |
| **Apache Airflow** | https://github.com/apache/airflow | Apache-2.0 | 46,724+ | 工作流编排和调度 |
| **Ray** | https://github.com/ray-project/ray | Apache-2.0 | 42,525+ | 分布式AI/ML计算引擎 |
| **Prometheus** | https://github.com/prometheus/prometheus | Apache-2.0 | 65,998+ | 时间序列指标和监控 |
| **Grafana** | https://github.com/grafana/grafana | AGPL-3.0 | 76,600+ | 指标可视化和仪表板 |
| **Evidently AI** | https://github.com/evidentlyai/evidently | Apache-2.0 | 7,786+ | ML/LLM可观测性和评估 |
| **Chroma** | https://github.com/chroma-core/chroma | Apache-2.0 | 29,230+ | AI应用向量数据库 |
| **Weaviate** | https://github.com/weaviate/weaviate | BSD-3-Clause | 16,751+ | 多模态向量数据库 |
| **Qdrant** | https://github.com/qdrant/qdrant | Apache-2.0 | 34,449+ | 高性能向量搜索 |
| **Milvus** | https://github.com/milvus-io/milvus | Apache-2.0 | 44,588+ | 大规模分布式向量数据库 |
| **llama.cpp** | https://github.com/ggerganov/llama.cpp | MIT | 70,000+ | CPU/GPU推理，GGUF量化 |
| **Ollama** | https://github.com/ollama/ollama | MIT | 100,000+ | 本地LLM管理和服务 |
| **Hugging Face Transformers** | https://github.com/huggingface/transformers | Apache-2.0 | 140,000+ | 预训练模型库 |
| **LangChain** | https://github.com/langchain-ai/langchain | MIT | 95,000+ | LLM应用框架 |
| **LlamaIndex** | https://github.com/run-llama/llama_index | MIT | 35,000+ | LLM数据框架 |
| **OpenTelemetry** | https://github.com/open-telemetry/opentelemetry-java | Apache-2.0 | 2,500+ | 分布式追踪和可观测性 |

*注：Star数为近似值，基于2026年9月收集的数据。*
