# 第四章：特征工程架构

---

## 学习目标

通过本章学习，你将能够：

- **设计特征存储架构**，消除训练-服务偏移并实现跨团队的特征复用
- **使用 Feast 实现开源特征存储**，包括在线和离线服务模式
- **对比特征存储解决方案**（Feast、Tecton、Hopsworks），在技术和运维维度上做出评估
- **诊断训练-服务偏移**并实施系统性的预防策略
- **构建特征管道**，维护批量特征和实时特征的一致性

---

## 4.1 为什么特征工程架构很重要

Jupyter Notebook 中运行良好的模型与生产环境中运行良好的模型之间存在巨大鸿沟。Google（ai.google）的研究表明，**85%的机器学习项目从未投入生产**。一个主要原因是缺乏适当的特征工程基础设施。

### 核心问题：训练-服务偏移

训练-服务偏移（Training-Serving Skew）发生在模型训练时使用的特征与服务时可用的特征不同。这不是理论问题——它是生产环境中最常见的静默模型退化原因。

考虑这个场景：
- 训练时，特征 `avg_transaction_amount_30d` 使用数据仓库中的复杂 SQL 联接从历史数据计算
- 服务时，同一特征使用具有不同联接逻辑的实时缓存计算
- 模型接收到微妙不同的特征值，导致预测退化

### 特征存储解决方案

特征存储通过提供以下功能解决此问题：

1. **统一特征定义** — 一个定义特征含义的地方
2. **一致计算** — 训练和服务使用相同逻辑
3. **时间点正确性** — 防止未来数据泄露
4. **特征共享** — 团队可以发现和复用现有特征
5. **低延迟服务** — 用于实时推理的在线特征

---

## 4.2 Feast：开源特征存储

### 📌 真实数据：Feast 采用情况

| 指标 | 数值 | 来源 |
|------|------|------|
| Slack 社区 | 5,500+ 成员 | feast.dev |
| GitHub 贡献者 | 293+ | github.com/feast-dev/feast |
| Docker Hub 下载量 | 12M+ | hub.docker.com/r/feastdev/feast-server |
| 使用公司 | Robinhood、NVIDIA、Shopify、IBM、Cloudflare、Walmart | feast.dev |
| 支持框架 | Python、PySpark、Pandas、TensorFlow、PyTorch | feast.dev |
| 首次发布 | 2019（Gojek） | feast.dev |

Feast（Feature Store，特征存储）最初由 Gojek 于2019年开发，现为 CNCF 沙箱项目。它是采用最广泛的开源特征存储。

### Feast 架构

```
┌─────────────────────────────────────────────────┐
│                 Feast 架构                        │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  特征    │───▶│  Registry│───▶│  在线    │  │
│  │  管道    │    │  注册表  │    │  存储    │  │
│  └──────────┘    └──────────┘    └──────────┘  │
│       │                │                │        │
│       ▼                ▼                ▼        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  离线    │    │  元数据  │    │  在线    │  │
│  │  存储    │    │  存储    │    │  存储    │  │
│  │(Parquet) │    │          │    │(Redis/   │  │
│  │          │    │          │    │ DynamoDB)│  │
│  └──────────┘    └──────────┘    └──────────┘  │
│                                                   │
│  训练（离线）         服务（在线）                 │
└─────────────────────────────────────────────────┘
```

### Feast 核心概念

**Feature Views（特征视图）**：定义特征的 Schema 和来源：

```python
from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

transaction_features = FeatureView(
    name="transaction_features",
    entities=["customer_id"],
    ttl=timedelta(days=7),
    schema=[
        Field(name="avg_transaction_amount_30d", dtype=Float32),
        Field(name="transaction_count_7d", dtype=Int64),
        Field(name="max_transaction_amount_30d", dtype=Float32),
    ],
    online=True,
    source=FileSource(
        path="s3://feature-store/transactions/",
        event_timestamp_column="event_timestamp",
    ),
)
```

**Feature Services（特征服务）**：定义特征如何被服务：

```python
from feast import FeatureService

fraud_detection_service = FeatureService(
    name="fraud_detection_v1",
    features=[
        transaction_features,
    ],
    description="欺诈检测模型 v1 的特征",
)
```

**时间点正确性**：Feast 确保在创建训练数据集时，每行只使用事件发生时实际可用的特征值：

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

training_df = store.get_historical_features(
    entity_df=entity_df,  # 包含 entity_id + event_timestamp 的 DataFrame
    features=[
        "transaction_features:avg_transaction_amount_30d",
        "transaction_features:transaction_count_7d",
    ],
).to_df()
```

### Feast 在线服务

对于实时推理，Feast 从低延迟在线存储提供特征：

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# 检索在线特征
feature_vector = store.get_online_features(
    features=[
        "transaction_features:avg_transaction_amount_30d",
        "transaction_features:transaction_count_7d",
    ],
    entity_rows=[
        {"customer_id": 12345},
    ],
).to_dict()
```

| 在线存储后端 | 延迟 | 使用场景 |
|-------------|------|----------|
| SQLite | < 1 毫秒 | 开发/测试 |
| Redis | 1-5 毫秒 | 生产环境，中等规模 |
| DynamoDB | 5-15 毫秒 | AWS，大规模 |
| Cassandra | 5-15 毫秒 | 多区域 |
| Firestore | 5-20 毫秒 | GCP |

---

## 💡 案例研究：Robinhood 如何使用 Feast 进行欺诈检测

### 问题

Robinhood 是零佣金交易平台，每天处理数百万笔交易。其欺诈检测模型需要在**100毫秒内**评估每笔交易，同时保持99%以上的准确率。

在使用 Feast 之前，Robinhood 面临：
- **特征重复**：多个团队独立计算相同特征
- **训练-服务偏移**：训练和服务使用不同方式计算特征
- **迭代缓慢**：新特征从离线实验到生产服务需要数周时间

### 架构

Robinhood 的欺诈检测特征管道：

```
┌─────────────────────────────────────────────────────┐
│            Robinhood 欺诈检测管道                     │
├─────────────────────────────────────────────────────┤
│                                                       │
│  交易事件 (Kafka)                                    │
│       │                                               │
│       ▼                                               │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐    │
│  │  特征    │────▶│  Feast   │────▶│  欺诈    │    │
│  │  计算    │     │  在线    │     │  模型    │    │
│  │ (Flink)  │     │  存储    │     │ (Serving)│    │
│  └──────────┘     └──────────┘     └──────────┘    │
│       │                               │             │
│       ▼                               ▼             │
│  ┌──────────┐                   ┌──────────┐       │
│  │  特征    │                   │  决策    │       │
│  │  管道    │                   │  引擎    │       │
│  │ (批量)   │                   │          │       │
│  └──────────┘                   └──────────┘       │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### 📌 真实数据：Robinhood 的规模

| 指标 | 数值 | 来源 |
|------|------|------|
| 每日交易量 | 5M+ | robinhood.com/blog |
| 特征服务延迟目标 | < 100ms | Robinhood 工程演讲 |
| 服务特征数 | 500+ | Feast 案例研究 |
| 模型重训练频率 | 每日 | 机器学习工程最佳实践 |

### 关键设计决策

1. **共享特征注册表**：所有团队在中心 Feast Registry 中注册特征。在计算新特征之前，工程师检查是否已存在。

2. **一致计算**：相同的 Feast Feature View 定义用于批量（训练）和在线（服务）计算。

3. **实时特征管道**：Apache Flink 处理交易事件，在事件发生后几秒内更新 Feast 的在线存储。

4. **特征版本化**：特征进行版本控制，允许安全演进而不会破坏现有模型。

### 成果

实施 Feast 后：
- **特征复用率提高60%** — 团队发现并复用现有特征，而非重新实现
- **训练-服务偏移消除** — 跨环境使用一致的特征定义
- **特征部署时间从数周缩短至数小时** — 自助服务特征注册

---

## 4.3 特征存储对比

### Feast vs. Tecton vs. Hopsworks

| 维度 | Feast（开源） | Tecton（托管） | Hopsworks（开源） |
|------|--------------|---------------|------------------|
| **部署方式** | 自托管 | 托管 SaaS | 自托管 / 托管 |
| **许可证** | Apache 2.0 | 专有 | AGPLv3 |
| **特征转换** | 用户定义（Python） | 内置转换 | 内置 + UDF |
| **在线存储** | Redis、DynamoDB 等 | 专有 | 在线特征存储 |
| **流处理** | Flink/Spark 集成 | 原生流处理 | Spark Streaming |
| **时间点正确性** | 支持 | 支持 | 支持 |
| **特征监控** | 基础 | 高级 | 高级 |
| **GitOps 集成** | 支持 | 支持 | 有限 |
| **企业支持** | 社区 | 完整 SLA | 可用 |
| **最适合** | 需要 OSS 灵活性的团队 | 需要托管服务的团队 | 需要完整 ML 平台的团队 |

### 架构对比

**Feast**：轻量级、模块化，与现有基础设施集成。最适合已有数据工程能力的团队。

**Tecton**：完整的托管服务，具有高级转换能力。最适合希望专注于机器学习而非基础设施的团队。

**Hopsworks**：完整的机器学习平台，包括特征存储、模型服务和实验管理。最适合构建端到端机器学习平台的团队。

---

## 4.4 特征工程模式

### 模式：滑动窗口聚合

最常见的特征工程模式是在时间窗口上计算聚合：

```python
from feast import FeatureView, Field
from feast.types import Float32, Int64
from datetime import timedelta

# 7天窗口
weekly_features = FeatureView(
    name="weekly_transaction_features",
    entities=["customer_id"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="sum_amount_7d", dtype=Float32),
        Field(name="count_transactions_7d", dtype=Int64),
        Field(name="avg_amount_7d", dtype=Float32),
        Field(name="max_amount_7d", dtype=Float32),
    ],
    online=True,
)

# 30天窗口
monthly_features = FeatureView(
    name="monthly_transaction_features",
    entities=["customer_id"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="sum_amount_30d", dtype=Float32),
        Field(name="count_transactions_30d", dtype=Int64),
        Field(name="avg_amount_30d", dtype=Float32),
    ],
    online=True,
)
```

### 模式：实体嵌入特征

分类实体的预计算嵌入：

```python
customer_embedding_features = FeatureView(
    name="customer_embedding",
    entities=["customer_id"],
    ttl=timedelta(days=7),
    schema=[
        Field(name="embedding", dtype=Array(Float32, dim=128)),
    ],
    online=True,
)
```

### 模式：交叉特征交互

捕获实体间关系的特征：

```python
merchant_customer_features = FeatureView(
    name="merchant_customer_interaction",
    entities=["customer_id", "merchant_id"],
    ttl=timedelta(days=1),
    schema=[
        Field(name="transaction_count_merchant_customer_30d", dtype=Int64),
        Field(name="avg_amount_merchant_customer_30d", dtype=Float32),
    ],
    online=True,
)
```

---

## 4.5 特征管道设计

### 批量特征管道

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  源数据  │───▶│  提取    │───▶│  转换    │───▶│  Feast   │
│          │    │  (SQL)   │    │ (Spark/  │    │  离线    │
│          │    │          │    │  Pandas) │    │  存储    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                       │
                                                       ▼
┌──────────┐    ┌──────────┐                   ┌──────────┐
│  模型    │◀──│  Feast   │◀──────────────────│  Feast   │
│  训练    │   │ 历史     │                   │  Registry│
│          │   │  特征    │                   │  注册表  │
└──────────┘    └──────────┘                   └──────────┘
```

### 实时特征管道

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  事件    │───▶│  流处理  │───▶│  特征    │───▶│  Feast   │
│  来源    │    │  处理器  │    │  计算    │    │  在线    │
│  (Kafka) │    │ (Flink)  │    │          │    │  存储    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                       │
                                                       ▼
┌──────────┐    ┌──────────┐                   ┌──────────┐
│  模型    │◀──│  Feast   │◀──────────────────│  在线    │
│  服务    │   │  获取    │                   │  存储    │
│          │   │  特征    │                   │ (Redis)  │
└──────────┘    └──────────┘                   └──────────┘
```

---

## ⚠️ 战争故事：训练-服务偏移导致信用评分模型失效

### 背景

一家金融科技公司部署了信用评分模型，离线评估准确率为92%。生产部署3个月后，合格申请人的批准率下降了23%，已批准申请人的违约率上升了15%。

### 调查过程

数据科学团队最初怀疑是模型漂移——数据分布随时间发生了偏移。他们重新训练了模型，但性能没有改善。

经过数周调查，他们发现了根本原因：

**训练特征**使用数据仓库中的30天回溯窗口从历史数据计算。

**服务特征**使用流式管道中的30天窗口从实时数据计算。

关键区别：流式管道排除了有24小时结算延迟的某些支付处理商的交易。这意味着服务特征系统性地低估了使用这些支付处理商客户的交易量。

### 影响

| 指标 | 之前 | 服务后 | 变化 |
|------|------|--------|------|
| 特征: transaction_count_30d | 45.2（平均） | 38.7（平均） | -14.4% |
| 特征: avg_transaction_amount | $127.30 | $142.50 | +11.9% |
| 模型准确率 | 92% | 81% | -11 个百分点 |
| 误报率 | 4.2% | 11.8% | +7.6 个百分点 |

### 根本原因

流式管道使用了仅订阅10个支付处理商事件流中7个的不同 Kafka 消费者组。缺失的3个流占交易量的14.4%——与观察到的差异完全一致。

### 预防措施

1. **特征验证测试**：自动化测试验证训练和服务特征分布在容差范围内匹配
2. **Schema 强制**：特征定义必须包含所有数据源，在管道创建时验证
3. **影子评分**：在相同数据上同时运行训练和服务管道并比较输出
4. **特征监控仪表盘**：实时跟踪服务与训练的特征分布

---

## 4.6 特征质量与验证

### 特征的数据质量检查

```python
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

# 验证特征分布
suite = ExpectationSuite(expectation_suite_name="feature_validation")

suite.add_expectation({
    "expectation_type": "expect_column_values_to_be_between",
    "kwargs": {
        "column": "avg_transaction_amount_30d",
        "min_value": 0,
        "max_value": 100000,
    }
})

suite.add_expectation({
    "expectation_type": "expect_column_values_to_not_be_null",
    "kwargs": {
        "column": "avg_transaction_amount_30d",
    }
})

suite.add_expectation({
    "expectation_type": "expect_column_mean_to_be_between",
    "kwargs": {
        "column": "transaction_count_7d",
        "min_value": 1,
        "max_value": 500,
    }
})
```

### 特征监控指标

| 指标 | 检测内容 | 告警阈值 |
|------|----------|----------|
| 特征漂移（PSI） | 分布偏移 | > 0.2 |
| 特征缺失率 | 管道故障 | > 5% |
| 特征相关性变化 | 关系变化 | > 0.3 |
| 特征延迟 | 服务退化 | > 100ms |
| 特征新鲜度 | 特征过时 | > 2x TTL |

---

## 4.7 特征存储决策框架

### 何时使用特征存储

| 标准 | 使用特征存储 | 不需要特征存储 |
|------|-------------|---------------|
| 机器学习模型数量 | 3+ 模型共享特征 | 单个模型，孤立 |
| 数据科学家数量 | 5+ 人 | 1-2 人 |
| 特征计算复杂度 | 复杂、多源 | 简单、单源 |
| 服务延迟要求 | < 100ms | 仅批量评分 |
| 训练频率 | 每日或更频繁 | 每周或更少 |
| 特征复用潜力 | 高（跨团队） | 低（团队特定） |

### 构建 vs. 购买决策

| 因素 | 构建（Feast） | 购买（Tecton） |
|------|-------------|---------------|
| 前期成本 | 工程时间 | 许可/订阅 |
| 持续维护 | 你的团队 | 供应商 |
| 定制化 | 完全控制 | 有限 |
| 价值实现时间 | 数周-数月 | 数天-数周 |
| 数据主权 | 完全控制 | 取决于供应商 |
| 最适合 | 成本敏感、定制需求 | 速度优先、标准需求 |

---

## 本章小结

特征工程架构是原始数据与生产机器学习之间的桥梁。关键要点：

1. **训练-服务偏移是机器学习系统的头号沉默杀手。**像 Feast 这样的特征存储通过确保训练和服务的一致特征计算来消除这一问题。

2. **Feast 是领先的开源特征存储**，拥有5,500+社区成员、293+贡献者，以及包括 Robinhood、NVIDIA 和 Shopify 在内的大型公司采用。

3. **特征复用是机器学习工程中最高杠杆的活动。**良好设计的特征注册表可以将特征开发时间从数周缩短至数小时。

4. **特征质量监控不是可选的。**分布漂移、缺失率和延迟必须持续跟踪，以防止静默模型退化。

5. **Feast、Tecton 和 Hopsworks 之间的选择**取决于团队的技术能力、预算以及对定制化 vs. 托管服务的需求。

---

## 讨论题

1. **架构决策**：你正在为一家拥有20名数据科学家、分布在5个团队的公司构建机器学习平台。每个团队有3-5个模型。你会实施特征存储吗？如果是，选择哪一个？请论证你的决策。

2. **权衡分析**：比较 Redis 与 DynamoDB 提供特征服务的延迟和成本权衡。在什么情况下你会选择每一个？

3. **根因分析**：你的模型准确率在2周内从94%降至88%。所有管道监控显示正常。描述你的调查流程。

4. **特征设计**：为网约车公司的动态定价模型设计一套特征集。你需要哪些实体？什么时间窗口？延迟要求是什么？

5. **特征伦理**：哪些伦理考虑应该指导贷款审批模型的特征工程？你如何检测和缓解特征中的偏见？

---

## 练习

### 练习1：Feast 安装与特征注册（动手实践）

1. 使用 `pip install feast` 在本地安装 Feast
2. 初始化 Feast 仓库：`feast init my_feature_repo`
3. 为虚构的电子商务数据集定义特征视图
4. 应用特征定义：`feast apply`
5. 使用 `feast features-retrieve` 检索在线特征

记录你的安装过程和特征检索延迟。

### 练习2：训练-服务偏移检测

根据以下场景：
- 训练数据计算于2025-06-01
- 特征 `purchase_frequency_30d` 在训练中的均值为12.3，标准差为4.1
- 服务数据显示均值为11.8，标准差为5.2

编写 Python 脚本：
1. 计算训练和服务分布之间的群体稳定性指数（PSI）
2. 确定漂移是否显著（PSI > 0.2 阈值）
3. 生成包含建议的报告

### 练习3：特征存储设计文档

为一家金融服务公司设计特征存储架构，要求：
- 10名数据科学家
- 15个机器学习模型（欺诈检测、信用评分、推荐）
- 延迟要求：实时特征 < 50ms
- 批量特征每日计算
- 合规要求：所有特征必须可审计

你的设计应包括：架构图、技术选型、治理模型和监控策略。

---

## 参考资料

1. **Feast 文档** — feast.dev
2. **Tecton 文档** — tecton.ai/docs
3. **Hopsworks 文档** — docs.hopsworks.ai
4. **Robinhood 工程：特征存储** — robinhood.com/blog
5. **NVIDIA Feast 集成** — developer.nvidia.com/blog
6. **Great Expectations 文档** — docs.greatexpectations.io
7. **机器学习特征工程** (Alice Zheng, Amanda Casari) — O'Reilly
8. **机器学习系统设计** (Chip Huyen) — oreilly.com
9. **Google：机器学习中的隐藏技术债务** — papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
10. **机器学习特征存储** (Hopsworks 博客) — hopsworks.ai/blog
