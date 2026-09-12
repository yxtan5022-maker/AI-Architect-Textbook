# 第五章：数据湖仓架构

---

## 学习目标

通过本章学习，你将能够：

- **架构数据湖仓**，结合数据湖的灵活性与数据仓库的可靠性
- **对比表格式**（Delta Lake、Apache Iceberg、Apache Hudi），在性能、特性和生态系统兼容性上做出评估
- **实现 Medallion 架构**，将数据组织为 Bronze、Silver 和 Gold 层
- **防止数据沼泽形成**，通过治理、质量和组织实践
- **为 AI/ML 工作负载设计湖仓**，同时支持分析查询和模型训练

---

## 5.1 数据湖仓：湖与仓的融合

数据湖仓是一种将数据湖和数据仓库统一的架构模式。它提供：

- **数据湖的灵活性**：以低成本存储任何格式的数据（结构化、半结构化、非结构化）
- **数据仓库的可靠性**：ACID 事务、Schema 强制和时间旅行
- **开放格式**：无厂商锁定；数据以开放文件格式（Parquet、ORC）存储
- **直接访问**：BI 工具和机器学习框架可以直接访问数据，无需 ETL

### 仅有数据湖的问题

数据湖承诺以低廉成本存储任何数据。实践中，许多变成了**数据沼泽** — 数据存储其中但很少使用的混乱仓库：

| 问题 | 后果 |
|------|------|
| 无 Schema 强制 | 数据质量不一致 |
| 无 ACID 事务 | 部分写入损坏数据集 |
| 无时间旅行 | 无法审计或回滚变更 |
| 无数据血缘 | 无法追溯数据来源 |
| 元数据混乱 | 无法发现数据 |

### 仅有数据仓库的问题

数据仓库解决了可靠性，但引入了约束：

| 问题 | 后果 |
|------|------|
| 昂贵存储 | $25-100/TB/月 vs. $0.02/TB/月（云存储） |
| 写入时 Schema | 刚性，无法处理半结构化数据 |
| 有限数据类型 | 无法原生存储图像、音频、视频、JSON |
| 专有格式 | 厂商锁定 |
| 扩展限制 | 垂直扩展成本高昂 |

### 湖仓解决方案

湖仓在云/对象存储中的开放文件格式之上添加了**元数据层**：

```
┌─────────────────────────────────────────────────┐
│              数据湖仓架构                         │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────────────────────────────────┐    │
│  │           元数据层                       │    │
│  │   (Delta Lake / Iceberg / Hudi)        │    │
│  │   - ACID 事务                           │    │
│  │   - Schema 强制                         │    │
│  │   - 时间旅行                            │    │
│  │   - 数据血缘                            │    │
│  └─────────────────────────────────────────┘    │
│                       │                           │
│  ┌─────────────────────────────────────────┐    │
│  │         开放文件格式                     │    │
│  │         (Parquet / ORC)                 │    │
│  └─────────────────────────────────────────┘    │
│                       │                           │
│  ┌─────────────────────────────────────────┐    │
│  │       云对象存储                        │    │
│  │    (S3 / GCS / ADLS / MinIO)           │    │
│  └─────────────────────────────────────────┘    │
│                                                   │
│  查询引擎：Spark、Trino、Presto、Dremio          │
│  BI 工具：Tableau、Power BI、Looker             │
│  ML 框架：PyTorch、TensorFlow、XGBoost          │
└─────────────────────────────────────────────────┘
```

---

## 5.2 表格式对比

### 📌 真实数据：Delta Lake、Iceberg 和 Hudi

| 特性 | Delta Lake | Apache Iceberg | Apache Hudi |
|------|-----------|---------------|-------------|
| **治理机构** | Linux 基金会 | Apache 软件基金会 | Apache 软件基金会 |
| **GitHub Stars** | 7,500+ | 6,500+ | 5,800+ |
| **GitHub Forks** | 1,300+ | 2,300+ | 2,100+ |
| **主要支持者** | Databricks | Netflix → Apache 社区 | Uber → Apache 社区 |
| **ACID 事务** | 支持 | 支持 | 支持 |
| **时间旅行** | 支持 | 支持 | 支持 |
| **Schema 演进** | 有限 | 完整（添加/删除/重命名） | 完整 |
| **分区演进** | 不支持（需要重写） | 支持（隐藏分区） | 支持 |
| **合并时读取** | 支持 | 支持 | 支持 |
| **写时复制** | 支持 | 支持 | 支持 |
| **Spark 集成** | 优秀（原生） | 优秀 | 优秀 |
| **Trino/Presto** | 良好 | 优秀 | 良好 |
| **Flink 集成** | 良好 | 增长中 | 优秀 |
| **最适合** | Databricks 生态、Spark 原生 | 多引擎、Iceberg REST Catalog | CDC、增量处理 |

### 架构差异

**Delta Lake** 在 Parquet 文件旁存储事务日志：

```
my_table/
├── _delta_log/
│   ├── 00000000000000000000.json
│   ├── 00000000000000000001.json
│   └── 00000000000000000002.checkpoint.parquet
├── part-00000-...parquet
├── part-00001-...parquet
└── part-00002-...parquet
```

**Apache Iceberg** 使用层次化元数据结构：

```
my_table/
├── metadata/
│   ├── v1.metadata.json
│   ├── v2.metadata.json
│   ├── snap-001.avro
│   ├── snap-002.avro
│   └── snap-001.parquet (manifest list)
├── data/
│   ├── part-00000-...parquet
│   └── part-00001-...parquet
```

**Apache Hudi** 使用基于时间线的架构：

```
my_table/
├── .hoodie/
│   ├── 20250101000000.commit
│   ├── 20250102000000.commit
│   └── 20250102000000.clean
├── part-00000-...parquet
└── part-00001-...parquet
```

### 性能对比

根据 Databricks、Netflix 和 Uber 工程团队发布的基准测试：

| 操作 | Delta Lake | Iceberg | Hudi |
|------|-----------|---------|------|
| **小文件合并** | Auto-optimize | 手动压缩 | 基于时间线压缩 |
| **分区裁剪** | 文件级统计 | 分区演进 | 分区裁剪 |
| **谓词下推** | Parquet 统计 | Parquet + 分区 | Parquet + 分区 |
| **并发写入** | 乐观并发 | 乐观并发 | OCC + 时间线 |
| **时间旅行查询** | 日志重放 | 基于快照 | 基于时间线 |
| **Schema 演进** | 仅添加列 | 完整 | 完整 |
| **Upsert 性能** | 中等 | 良好 | 优秀 |
| **读取性能** | 优秀 | 优秀 | 良好 |

### 如何选择每种格式

| 场景 | 推荐格式 | 原因 |
|------|----------|------|
| 已使用 Databricks | Delta Lake | 原生集成，最佳 Spark 支持 |
| 多引擎（Spark + Trino + Flink） | Iceberg | 最佳跨引擎兼容性 |
| CDC / 变更数据处理 | Hudi | 原生增量处理 |
| PostgreSQL/MySQL 复制 | Hudi | 内置 CDC 连接器 |
| 多云部署 | Iceberg | REST Catalog 标准 |
| 简单用例，仅 Spark | Delta Lake | 最易设置 |
| 时间旅行/审计要求 | Iceberg | 最成熟的快照隔离 |

---

## 5.3 Medallion 架构

### 📌 真实数据：Medallion 架构（Databricks）

Medallion 架构是 Databricks（databricks.com）引入的数据设计模式，用于将湖仓数据组织到渐进式质量层。它已成为湖仓数据组织的事实标准。

### 三个层级

```
┌─────────────────────────────────────────────────────────┐
│                 Medallion 架构                           │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
│  │   BRONZE    │──▶│   SILVER    │──▶│    GOLD     │  │
│  │  （铜层）    │   │  （银层）    │   │  （金层）    │  │
│  │             │   │             │   │             │  │
│  │  原始数据   │   │  清洗并     │   │  业务级     │  │
│  │  原样保留   │   │  规范化     │   │  聚合       │  │
│  │             │   │             │   │             │  │
│  │  完整保真度 │   │  去重       │   │  为查询     │  │
│  │  无转换     │   │  验证       │   │  优化       │  │
│  │             │   │  丰富       │   │             │  │
│  └─────────────┘   └─────────────┘   └─────────────┘  │
│                                                           │
│  质量：原始       质量：清洁        质量：策划            │
│  消费者：         消费者：          消费者：              │
│  数据工程师       数据科学家        BI 分析师             │
│                   机器学习工程师    仪表盘                │
└─────────────────────────────────────────────────────────┘
```

### Bronze 层：原始数据

Bronze 层按原样存储数据，不做任何转换。这提供：

- **完整数据保真度**：每条记录都保留，包括错误和重复
- **可审计性**：摄入内容的完整历史
- **重放能力**：如果转换有 Bug，可以从原始数据重新处理
- **读取时 Schema**：数据以其原始格式存储

```python
# Bronze 层摄入
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Bronze_Ingestion").getOrCreate()

# 从源读取原始数据
raw_events = spark.read \
    .format("json") \
    .option("inferSchema", "true") \
    .load("s3://raw-data/events/")

# 写入 Bronze，完整保真
raw_events.write \
    .format("delta") \
    .mode("append") \
    .partitionBy("date") \
    .save("s3://lakehouse/bronze/events/")
```

### Silver 层：清洗和规范化

Silver 层应用质量规则和标准化：

```python
# Silver 层转换
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower

spark = SparkSession.builder.appName("Silver_Transform").getOrCreate()

# 从 Bronze 读取
bronze_events = spark.read.format("delta").load("s3://lakehouse/bronze/events/")

# 应用质量规则
silver_events = bronze_events \
    .filter(col("event_id").isNotNull()) \
    .filter(col("user_id").isNotNull()) \
    .dropDuplicates(["event_id"]) \
    .withColumn("event_type", lower(trim(col("event_type")))) \
    .withColumn("timestamp", col("timestamp").cast("timestamp"))

# 写入 Silver
silver_events.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("date", "event_type") \
    .save("s3://lakehouse/silver/events/")
```

### Gold 层：业务级聚合

Gold 层包含为消费优化的业务级聚合：

```python
# Gold 层聚合
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum, avg, window

spark = SparkSession.builder.appName("Gold_Aggregate").getOrCreate()

# 从 Silver 读取
silver_events = spark.read.format("delta").load("s3://lakehouse/silver/events/")

# 计算业务级聚合
gold_daily_metrics = silver_events \
    .groupBy("date", "event_type") \
    .agg(
        count("*").alias("event_count"),
        countDistinct("user_id").alias("unique_users"),
    )

# 写入 Gold
gold_daily_metrics.write \
    .format("delta") \
    .mode("overwrite") \
    .save("s3://lakehouse/gold/daily_metrics/")
```

### Medallion 层职责

| 层级 | 数据质量 | 消费者 | Schema | 写入模式 | 成本 |
|------|----------|--------|--------|----------|------|
| **Bronze** | 原始、未验证 | 数据工程师 | 读取时 Schema | 追加 | 最低 |
| **Silver** | 清洗、去重 | 数据科学家、ML | 写入时 Schema | 合并/Upsert | 中等 |
| **Gold** | 聚合、策划 | BI 分析师、仪表盘 | 严格 Schema | 覆盖 | 最高 |

---

## 💡 案例研究：Airbnb 的数据湖架构

### 问题

Airbnb 运营着世界上最大的数据平台之一，服务于220+国家的数百万房源。他们的数据挑战包括：

- **500+ PB** 数据，分布在数千个数据集
- **数千条数据管道** 每日运行
- **多计算引擎**：Spark、Presto、Hive
- **多样化用例**：搜索排名、定价优化、欺诈检测、业务分析

### 架构

Airbnb 的数据湖架构建立在三大支柱之上：

1. **Doris** — Airbnb 的内部数据湖平台，提供跨多个存储系统的统一数据访问
2. **Apache Airflow** — 所有数据管道的编排（Airflow 在 Airbnb 创建）
3. **Presto/Trino** — 数据湖上的交互式 SQL 分析

### 📌 真实数据：Airbnb 的规模

| 指标 | 数值 | 来源 |
|------|------|------|
| 总数据量 | 500+ PB | medium.com/airbnb-engineering |
| 每日管道运行 | 10,000+ | Airbnb 工程博客 |
| 数据集 | 10,000+ | medium.com/airbnb-engineering |
| 查询引擎 | Presto | Airbnb 工程博客 |
| 编排 | Airflow | airflow.apache.org |

### 关键设计决策

**1. 读取时 Schema 而非写入时 Schema**：Airbnb 以原始格式存储数据，在查询时应用 Schema。这为多样化的数据类型和演进的 Schema 提供了灵活性。

**2. Presto 用于交互式分析**：Airbnb 选择 Presto（现 Trino）进行交互式 SQL 查询，因为它在 PB 级数据上提供亚秒延迟，且无需物化结果。

**3. Airflow 用于编排**：Airbnb 创建了 Airflow 并继续作为其最大部署者。所有数据管道都定义为 Python Airflow DAG。

**4. 成本优化**：Airbnb 使用分层存储策略：
- 热数据：SSD 支持的存储用于频繁访问的数据集
- 温数据：标准 S3 存储用于近期数据
- 冷数据：Glacier 用于历史归档

### 经验教训

1. **元数据与数据同等重要**：没有强大的元数据和编目，数据湖会变成数据沼泽。Airbnb 在数据发现工具上投入巨大。

2. **Presto 民主化了数据访问**：通过在数据湖上提供 SQL 接口，非技术用户无需编写 MapReduce 或 Spark 作业即可查询数据。

3. **Airflow 创建了自助管道平台**：工程师无需平台团队参与即可创建新数据管道，加速了迭代。

4. **成本管理至关重要**：在 500+ PB 规模下，存储格式或压缩的微小优化就能节省数百万美元。

---

## 5.4 防止数据沼泽

### 数据沼泽问题

数据沼泽是失去实用性的数据湖。迹象包括：
- 数据被摄入但从未被查询
- 没有文档或元数据
- 不同 Schema 的重复数据集
- 没有数据质量保证
- 需要数据时找不到

### 预防策略

| 策略 | 实施方式 | 影响 |
|------|----------|------|
| **数据目录** | Apache Atlas、DataHub、Amundsen | 数据可发现 |
| **Schema Registry** | Confluent Schema Registry、AWS Glue | Schema 治理 |
| **数据质量** | Great Expectations、dbt tests | 数据可信 |
| **访问策略** | 基于角色的访问控制 | 安全与合规 |
| **数据血缘** | Apache Atlas、OpenLineage | 可审计性 |
| **成本监控** | 云成本仪表盘 | 成本控制 |
| **自动归档** | 生命周期策略 | 存储优化 |

### 数据治理框架

```
┌─────────────────────────────────────────────────────┐
│              数据治理框架                             │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  数据    │    │  Schema  │    │  数据    │      │
│  │  目录    │    │ Registry │    │  质量    │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  访问    │    │  数据    │    │  成本    │      │
│  │  控制    │    │  血缘    │    │  监控    │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### Apache Atlas：元数据治理

Apache Atlas（atlas.apache.org）为 Hadoop 及其他提供元数据治理：

- **类型系统**：定义数据集、流程和用户的元数据类型
- **血缘跟踪**：跟踪数据如何流经管道
- **分类**：用敏感级别和业务术语标记数据
- **治理**：执行数据访问和保留策略

---

## 5.5 面向 AI/ML 工作负载的湖仓

### 同时支持分析和 ML

湖仓必须服务两种不同的工作负载：

1. **分析查询**：基于 SQL、聚合、联接、BI 仪表盘
2. **ML 工作负载**：特征提取、模型训练、批量推理

### 从湖仓提取特征

```python
# 从 Silver 层提取 ML 特征
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("ML_Feature_Extraction").getOrCreate()

# 从 Silver 读取清洗后的数据
events = spark.read.format("delta").load("s3://lakehouse/silver/events/")

# 计算用户级特征
user_features = events \
    .groupBy("user_id") \
    .agg(
        count("*").alias("total_events"),
        countDistinct("event_type").alias("unique_event_types"),
        avg("session_duration").alias("avg_session_duration"),
        max("timestamp").alias("last_event_time"),
    )

# 将特征写入专用特征表
user_features.write \
    .format("delta") \
    .mode("overwrite") \
    .save("s3://lakehouse/gold/ml_features/user_features/")
```

### 模型训练数据准备

```python
# 准备具有时间点正确性的训练数据集
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Training_Data").getOrCreate()

# 读取特征
features = spark.read.format("delta").load("s3://lakehouse/gold/ml_features/")
labels = spark.read.format("delta").load("s3://lakehouse/silver/labels/")

# 具有时间点正确性的联接
training_data = features \
    .join(labels, on="user_id", how="inner") \
    .filter(col("feature_date") <= col("label_date"))

# 拆分并保存
training_data.write.format("delta").mode("overwrite") \
    .save("s3://lakehouse/gold/ml_training/training_set/")
```

### 批量推理管道

```python
# 基于湖仓的批量推理
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Batch_Inference").getOrCreate()

# 读取最新特征
latest_features = spark.read.format("delta") \
    .load("s3://lakehouse/gold/ml_features/")

# 加载模型（从 MLflow 或模型注册表）
# model = mlflow.spark.load_model("models:/churn_model/Production")

# 生成预测
# predictions = model.transform(latest_features)

# 将预测写回湖仓
# predictions.write.format("delta").mode("overwrite") \
#     .save("s3://lakehouse/gold/predictions/churn_predictions/")
```

---

## ⚠️ 战争故事：数据湖如何变成数据沼泽

### 背景

一家大型医疗公司在2019年构建了数据湖，以整合来自15个不同医院系统的数据。初始目标是实现人群健康分析和预测建模。

### 第一年：前景

- 摄入200+数据集
- 数据以 Parquet 格式存储在 S3
- 构建了初始分析仪表盘
- 获得高管支持和预算批准

### 第二年：衰退

问题浮现：

1. **无 Schema 强制**：不同医院系统对相同概念使用不同 Schema（如 "patient_id" vs "pat_id" vs "mrn"）

2. **无数据质量检查**：缺失值、重复和不一致的格式未被检测

3. **无文档**：新数据科学家无法找到或理解现有数据集

4. **重复激增**：团队创建了自己版本的数据集并进行轻微修改，导致"患者人口统计"出现50+版本

5. **成本爆炸**：存储成本从$50K/月增长到$300K/月，随着原始数据积累

### 第三年：沼泽

- 仅15%的数据集被主动查询
- 查找正确数据集的平均时间：3天
- 数据科学家80%的时间花在数据准备上
- 数据信任度接近零
- 数据湖被戏称为"数据沼泽"

### 恢复

公司启动了为期12个月的数据湖恢复计划：

| 阶段 | 持续时间 | 行动 | 结果 |
|------|----------|------|------|
| **编目** | 第1-3月 | 部署 Apache Atlas，编目所有数据集 | 数据可发现 |
| **质量** | 第3-6月 | 实施 Great Expectations，数据质量评分 | 信任增加 |
| **治理** | 第6-9月 | 访问策略、Schema Registry、数据所有权 | 合规 |
| **优化** | 第9-12月 | 压缩小文件、归档冷数据、成本降低60% | 成本控制 |

### 关键教训

1. **技术不是问题**：公司拥有所有正确的工具（S3、Parquet、Spark）。失败是组织层面的。

2. **数据治理必须主动**：等到沼泽形成后再治理，成本是预防的10倍。

3. **所有权很重要**：每个数据集必须有指定的所有者负责其质量和文档。

4. **成本监控防止意外**：没有成本跟踪，存储成本可能不受控制地膨胀。

---

## 5.6 性能优化

### 小文件问题

数据湖中最常见的性能问题是小文件的积累：

| 文件大小 | 影响 | 解决方案 |
|----------|------|----------|
| < 1 MB | 读取性能极差 | 压缩作业 |
| 1-64 MB | 次优 | 定期压缩 |
| 64-256 MB | 良好 | 目标范围 |
| 256 MB - 1 GB | 最优 | 理想范围 |
| > 1 GB | 可能导致 OOM | 考虑拆分 |

### 压缩策略

```python
# Delta Lake 压缩
from delta.tables import DeltaTable

# Auto-optimize（Databricks）
# SET spark.databricks.delta.optimizeWrite.enabled = true
# SET spark.databricks.delta.autoCompact.enabled = true

# 手动压缩
delta_table = DeltaTable.forPath(spark, "s3://lakehouse/silver/events/")
delta_table.optimize().executeCompaction()
```

### 分区策略

| 策略 | 使用场景 | 示例 |
|------|----------|------|
| **日期分区** | 时间序列数据，最常见 | `date=2025-01-01/` |
| **类别分区** | 低基数分类 | `country=US/` |
| **不分区** | 小数据集（< 1GB） | 单个目录 |
| **Hive 风格分区** | 旧版兼容性 | `date=2025-01-01/hour=12/` |
| **Iceberg 隐藏分区** | 灵活、非显式分区 | Iceberg 规范 |

### Z-Order（数据聚类）

Z-Order 将相关数据共定位以提高查询性能：

```python
# Delta Lake Z-ORDER
delta_table = DeltaTable.forPath(spark, "s3://lakehouse/silver/events/")
delta_table.optimize().executeZOrderBy("user_id", "event_type")
```

---

## 📝 何时使用 / 何时不使用湖仓

| 场景 | 使用湖仓？ | 理由 |
|------|-----------|------|
| 混合分析 + ML 工作负载 | 是 | 湖仓擅长统一两者 |
| 仅 SQL 分析，简单 Schema | 否 — 使用数据仓库 | 仓库更简单且性能更好 |
| 非结构化数据（图像、视频） | 是 | 湖仓处理任何数据格式 |
| 仅实时流处理 | 部分 — 使用 Kafka + 湖仓 | 流处理需要专用基础设施 |
| 严格监管合规 | 视成熟度而定 | 仓库可能更简单 |
| 成本敏感、大数据量 | 是 | 湖仓存储成本低10-100倍 |
| 小团队、简单需求 | 否 — 使用托管仓库 | 复杂性不合理 |

---

## 本章小结

数据湖仓代表了数据湖和数据仓库的融合，提供前者的灵活性和后者的可靠性。

1. **Delta Lake、Apache Iceberg 和 Apache Hudi** 是三种领先的表格式。Delta Lake 在 Databricks 生态中表现出色；Iceberg 提供最佳多引擎兼容性；Hudi 适合 CDC 和增量处理。

2. **Medallion 架构**（Bronze → Silver → Gold）提供了按质量级别组织湖仓数据的成熟模式，每层服务不同的消费者。

3. **Airbnb 的架构**展示了最先进的实践：500+ PB 数据、10,000+ 每日管道运行，以及基于 Airflow 和 Presto 的自助平台。

4. **数据沼泽源于组织疏忽，而非技术失败。**预防需要主动的数据治理：目录、质量检查、所有权和成本监控。

5. **湖仓支持 AI/ML 工作负载**，通过直接特征提取、训练数据准备和批量推理——全部利用与分析相同的数据平台。

---

## 讨论题

1. **架构决策**：你的公司有200 TB 数据，80%结构化（来自 PostgreSQL 数据库），20%非结构化（图像和 PDF）。你需要同时支持 SQL 分析和图像分类 ML。你会使用数据仓库、数据湖还是湖仓？请论证你的决策。

2. **格式选择**：你正在启动一个新项目，Spark 作为主要计算引擎。你预计未来会添加 Trino 用于交互式查询。你会选择哪种表格式？为什么？

3. **成本分析**：一个数据湖存储500 TB 数据。如果你实施压缩（3倍比率）和压缩（减少80%小文件），以 $0.023/GB S3 定价计算，每月能节省多少？

4. **治理设计**：为一家需要遵守 HIPAA 同时让数据科学团队发现和使用患者数据的医疗公司设计数据治理框架。

5. **权衡分析**：比较自托管 Delta Lake/Iceberg 与使用 Databricks、Snowflake 或 BigQuery 等托管服务的运维复杂性。总拥有成本有何影响？

---

## 练习

### 练习1：湖仓搭建（动手实践）

1. 使用以下工具搭建本地湖仓：
   - MinIO（S3 兼容存储）
   - 带 Delta Lake 的 Apache Spark
   - Jupyter notebook

2. 实现 Medallion 架构：
   - Bronze：摄入示例 CSV 数据集
   - Silver：清洗、去重和验证
   - Gold：创建业务级聚合

3. 通过查询历史版本演示时间旅行

4. 记录你的搭建过程和结果

### 练习2：表格式对比

创建 Delta Lake、Iceberg 和 Hudi 的基准测试：

1. 在三种格式中创建相同的数据集
2. 测量：写入吞吐量、读取吞吐量、Upsert 性能、时间旅行查询延迟
3. 生成包含实际性能数据的对比报告

### 练习3：数据沼泽预防计划

你继承了一个有500个数据集、无文档、无质量检查、所有权未知的数据湖。设计6个月恢复计划：

1. 确定优先编目的数据集
2. 设计数据质量评分系统
3. 创建数据所有权分配流程
4. 定义恢复计划的成功指标

---

## 参考资料

1. **Delta Lake 文档** — docs.delta.io
2. **Apache Iceberg 文档** — iceberg.apache.org/docs
3. **Apache Hudi 文档** — hudi.apache.org
4. **Databricks：Medallion 架构** — databricks.com/glossary/medallion-architecture
5. **Airbnb 工程博客** — medium.com/airbnb-engineering
6. **Apache Atlas 文档** — atlas.apache.org/docs
7. **Great Expectations 文档** — docs.greatexpectations.io
8. **lakeFS：数据的 Git** — lakefs.io
9. **Dremio：数据湖仓** — dremio.com
10. **数据湖 vs. 数据仓库** (Martin Kleppmann) — dataintensive.net
