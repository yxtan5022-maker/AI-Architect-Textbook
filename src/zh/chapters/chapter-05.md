# 第五章：数据湖仓架构

> **学习目标**：读完本章，你将能够：
> 1. 理解数据湖、数据仓库和湖仓一体架构的核心差异
> 2. 对比 Delta Lake、Apache Iceberg、Apache Hudi 等表格式技术
> 3. 设计统一分析架构，支持 BI + AI 混合负载
> 4. 实现数据治理与合规管理体系
> 5. 针对 AI 工作负载优化数据架构
> 6. 基于 Delta Lake 搭建完整的湖仓架构

---

## 5.1 数据湖 vs 数据仓库 vs 湖仓一体 🟢

### 5.1.1 数据架构演进历程

📌 **关键概念**：数据架构经历了三个主要阶段的演进：数据仓库 → 数据湖 → 湖仓一体。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    数据架构演进时间线                                  │
│                                                                     │
│  2000s              2010s              2020s                        │
│  ────────           ────────           ────────                     │
│  数据仓库            数据湖              湖仓一体                     │
│  (Data Warehouse)  (Data Lake)        (Data Lakehouse)             │
│                                                                     │
│  特点:                  特点:                 特点:                   │
│  - 结构化数据           - 任意格式            - 结构化+非结构化        │
│  - Schema-on-Write     - Schema-on-Read     - Schema Evolution      │
│  - 高成本               - 低成本              - 中等成本              │
│  - 强一致性             - 最终一致性          - ACID事务              │
│  - 批处理为主           - 批+流              - 批+流+交互             │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.1.2 三种架构详细对比

| 维度 | 数据仓库 | 数据湖 | 湖仓一体 |
|------|---------|-------|---------|
| **数据格式** | 结构化（SQL表） | 任意格式（文件） | 结构化+半结构化 |
| **Schema管理** | Schema-on-Write | Schema-on-Read | 两者都支持 |
| **存储成本** | 高（$0.02-0.10/GB/月） | 低（$0.01-0.02/GB/月） | 中等 |
| **查询性能** | 快（列式存储+索引） | 慢（全表扫描） | 快（表格式优化） |
| **数据新鲜度** | 小时~天 | 实时~分钟 | 实时~分钟 |
| **事务支持** | ✅ 强一致性 | ❌ 无事务 | ✅ ACID事务 |
| **Schema演进** | 困难 | 灵活 | 灵活且受控 |
| **适用场景** | BI报表、OLAP | 数据科学、探索 | BI+AI统一平台 |
| **典型工具** | Snowflake、Redshift | S3、HDFS | Delta Lake、Iceberg |

### 5.1.3 湖仓一体的核心优势

📌 **关键概念**：湖仓一体（Lakehouse）是 Dan Abitbol（Databricks 联合创始人）提出的概念，它结合了数据湖的低成本灵活性和数据仓库的管理能力。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    湖仓一体架构示意图                                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    应用层                                      │  │
│  │                                                              │  │
│  │  BI/OLAP      AI/ML       数据科学       实时分析            │  │
│  │  (报表)       (训练)      (探索)        (监控)              │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    SQL/计算引擎层                               │  │
│  │                                                              │  │
│  │  Spark SQL   Presto/Trino   Flink   Delta Lake API          │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    表格式层 (Table Format)                     │  │
│  │                                                              │  │
│  │  Delta Lake    Apache Iceberg    Apache Hudi                │  │
│  │  (元数据管理、事务、版本控制)                                    │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    存储层                                      │  │
│  │                                                              │  │
│  │  S3 / ADLS / GCS / HDFS / 本地文件系统                        │  │
│  │  (低成本对象存储)                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.1.4 选型决策指南

```
选型决策树:

需要ACID事务?
├── 是 → 需要流式写入?
│        ├── 是 → Apache Hudi (增量处理优化)
│        └── 否 → 需要时间旅行?
│                 ├── 是 → Delta Lake / Apache Iceberg
│                 └── 否 → Delta Lake (更简单)
└── 否 → 只需低成本存储?
         ├── 是 → 原始S3/HDFS
         └── 否 → 需要Schema演进?
                  ├── 是 → Apache Iceberg (Schema演进最好)
                  └── 否 → Delta Lake
```

---

## 5.2 表格式（Table Format）对比 🟡

### 5.2.1 什么是表格式

📌 **关键概念**：表格式（Table Format）是介于文件系统和SQL引擎之间的一层元数据管理层。它定义了如何将小文件组织成逻辑表，并提供ACID事务、时间旅行、Schema演进等能力。

```
┌─────────────────────────────────────────────────────────────────┐
│                    表格式的位置                                    │
│                                                                 │
│  ┌──────────────┐                                              │
│  │  SQL 引擎     │  Spark / Presto / Flink / Trino             │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  表格式       │  Delta Lake / Iceberg / Hudi                │
│  │              │  (元数据、事务、版本)                           │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  文件格式     │  Parquet / ORC / Avro                       │
│  │              │  (列式存储、压缩)                               │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  存储系统     │  S3 / HDFS / ADLS / GCS                    │
│  │              │  (分布式存储)                                   │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2.2 三大表格式深度对比

| 特性 | Delta Lake | Apache Iceberg | Apache Hudi |
|------|-----------|---------------|-------------|
| **ACID事务** | ✅ | ✅ | ✅ |
| **时间旅行** | ✅ (版本号/时间戳) | ✅ (快照ID) | ✅ (时间戳) |
| **Schema演进** | ✅ (支持但有限) | ✅ (最好) | ✅ (有限) |
| **分区演进** | ❌ (需要重写) | ✅ (隐藏分区) | ❌ (需要重写) |
| **增量处理** | ✅ (Change Data Feed) | ✅ (增量读取) | ✅ (最优化) |
| **流式写入** | ✅ (Structured Streaming) | ✅ (Flink集成) | ✅ (最优化) |
| **元数据存储** | _delta_log/ (JSON) | metadata/ (Avro) | .hoodie/ (JSON) |
| **文件布局** | Parquet + 日志 | Parquet + manifest | Parquet + 索引 |
| **社区支持** | Databricks主导 | Apache基金会 | Apache基金会 |
| **学习曲线** | 低 | 中 | 中 |
| **生产就绪** | 高 | 高 | 高 |

### 5.2.3 各格式内部结构

**Delta Lake 内部结构**：
```
my_table/
├── _delta_log/
│   ├── 00000000000000000000.json  ← 版本0
│   ├── 00000000000000000001.json  ← 版本1
│   ├── 00000000000000000002.json  ← 版本2
│   └── _last_checkpoint           ← 最新检查点
├── part-00000-xxx.parquet         ← 数据文件
├── part-00001-xxx.parquet
└── part-00002-xxx.parquet
```

**Apache Iceberg 内部结构**：
```
my_table/
├── metadata/
│   ├── v1.metadata.json          ← 表元数据
│   ├── v1.manifest-list-xxx.avro ← manifest列表
│   ├── v1-xxx.manifest           ← manifest文件
│   └── snap-xxx.avro             ← 快照
├── data/
│   ├── part-00000-xxx.parquet
│   └── part-00001-xxx.parquet
└── README.md
```

**Apache Hudi 内部结构**：
```
my_table/
├── .hoodie/
│   ├── .hoodie_partition_metafile
│   ├── 00000000000000.commit       ← 提交元数据
│   ├── 00000000000000.clean        ← 清理元数据
│   └── .schema                     ← Schema文件
├── part-00000-xxx.parquet
└── part-00001-xxx.parquet
```

---

## 5.3 统一分析架构 🔴

### 5.3.1 批流一体架构

📌 **关键概念**：批流一体（Unified Batch & Streaming）是指使用同一套代码和架构同时处理批处理和流式数据，避免维护两套独立的系统。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    批流一体架构                                      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    统一数据接入层                               │  │
│  │                                                              │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐            │  │
│  │  │  CDC       │  │  日志流     │  │  批量文件   │            │  │
│  │  │ (Debezium) │  │ (Kafka)    │  │ (S3/FTP)  │            │  │
│  │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘            │  │
│  └─────────┼───────────────┼───────────────┼───────────────────┘  │
│            │               │               │                      │
│            └───────────────┼───────────────┘                      │
│                            ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    统一计算引擎层                               │  │
│  │                                                              │  │
│  │  Apache Spark                                                │  │
│  │  ├── Structured Streaming (流处理)                            │  │
│  │  ├── Batch Processing (批处理)                                │  │
│  │  └── DataFrame/SQL API (统一接口)                             │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    统一存储层 (Delta Lake)                     │  │
│  │                                                              │  │
│  │  Delta Lake Table                                           │  │
│  │  ├── 实时数据: Structured Streaming 持续写入                  │  │
│  │  ├── 历史数据: 批量回填/重算                                  │  │
│  │  └── 查询视图: 按需创建 (全量/增量/时间点)                     │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│              ┌──────────────┼──────────────┐                       │
│              ▼              ▼              ▼                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │  BI 报表    │  │  AI 训练    │  │  实时监控   │               │
│  │  (全量快照) │  │ (时间点快照) │  │ (增量数据)  │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3.2 多引擎支持架构

📌 **关键概念**：湖仓一体架构需要支持多种计算引擎，以满足不同工作负载的需求。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    多引擎统一分析架构                                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    查询/计算引擎层                              │  │
│  │                                                              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │  │
│  │  │  Spark   │ │ Presto/  │ │  Flink   │ │   Trino  │       │  │
│  │  │  SQL     │ │  Trino   │ │  SQL     │ │          │       │  │
│  │  │          │ │          │ │          │ │          │       │  │
│  │  │ 批处理   │ │ 交互查询  │ │ 流处理   │ │ 交互查询  │       │  │
│  │  │ ML训练   │ │ BI报表    │ │ 实时ETL  │ │ 数据发现  │       │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    表格式层 (统一元数据)                        │  │
│  │                                                              │  │
│  │  Delta Lake / Iceberg / Hudi                                │  │
│  │  (提供ACID事务、Schema管理、版本控制)                           │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    存储层                                      │  │
│  │                                                              │  │
│  │  S3 / ADLS / GCS / HDFS                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3.3 数据分层架构

📌 **关键概念**：湖仓一体中的数据分层（Medallion Architecture）是 Databricks 提出的数据组织模式，将数据分为铜层（Bronze）、银层（Silver）、金层（Gold）三层。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Medallion 数据分层架构                            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  铜层 (Bronze) - 原始数据层                                   │  │
│  │                                                              │  │
│  │  内容: 原始数据的忠实副本                                     │  │
│  │  格式: JSON/CSV/Parquet (原始格式)                            │  │
│  │  更新: 追加写入                                               │  │
│  │  用途: 数据溯源、审计、重处理                                  │  │
│  │  示例: s3://lake/bronze/orders/                              │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  银层 (Silver) - 清洗/标准化层                                │  │
│  │                                                              │  │
│  │  内容: 清洗、去重、标准化后的数据                              │  │
│  │  格式: Parquet (列式存储)                                     │  │
│  │  更新: 增量/合并写入                                          │  │
│  │  用途: 分析查询、特征工程                                      │  │
│  │  示例: s3://lake/silver/orders/                              │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  金层 (Gold) - 聚合/业务层                                   │  │
│  │                                                              │  │
│  │  内容: 面向业务的聚合数据                                     │  │
│  │  格式: Parquet/Delta                                         │  │
│  │  更新: 定期刷新                                               │  │
│  │  用途: BI报表、数据产品                                        │  │
│  │  示例: s3://lake/gold/daily_metrics/                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# 示例：Medallion 架构实现
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable

spark = SparkSession.builder \
    .appName("MedallionArchitecture") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# ========== 铜层 (Bronze) ==========
def ingest_to_bronze():
    """将原始数据导入铜层"""
    
    # 读取原始JSON数据
    raw_orders = spark.read \
        .option("multiline", "true") \
        .json("s3://raw-data/orders/*.json")
    
    # 添加元数据列
    bronze_orders = raw_orders \
        .withColumn("_ingestion_time", F.current_timestamp()) \
        .withColumn("_source_file", F.input_file_name()) \
        .withColumn("_batch_id", F.lit("batch_2026_01_15"))
    
    # 写入铜层（追加模式）
    bronze_orders.write \
        .format("delta") \
        .mode("append") \
        .partitionBy("event_date") \
        .save("s3://lake/bronze/orders")
    
    print(f"Bronze layer: {bronze_orders.count()} records ingested")

# ========== 银层 (Silver) ==========
def process_to_silver():
    """从铜层清洗数据到银层"""
    
    # 读取铜层数据
    bronze_orders = spark.read.format("delta").load("s3://lake/bronze/orders")
    
    # 数据清洗与标准化
    silver_orders = bronze_orders \
        .filter(F.col("order_id").isNotNull()) \
        .filter(F.col("amount") > 0) \
        .dropDuplicates(["order_id"]) \
        .withColumn("order_date", F.to_date("created_at")) \
        .withColumn("order_hour", F.hour("created_at")) \
        .withColumn("amount_rounded", F.round("amount", 2)) \
        .withColumn("user_id_upper", F.upper("user_id")) \
        .select(
            "order_id", "user_id", "user_id_upper",
            "amount", "amount_rounded", "currency",
            "order_date", "order_hour", "created_at",
            "_ingestion_time"
        )
    
    # 合并写入银层（支持增量更新）
    if DeltaTable.isDeltaTable(spark, "s3://lake/silver/orders"):
        delta_table = DeltaTable.forPath(spark, "s3://lake/silver/orders")
        
        delta_table.alias("target").merge(
            silver_orders.alias("source"),
            "target.order_id = source.order_id"
        ).whenMatchedUpdateAll() \
         .whenNotMatchedInsertAll() \
         .execute()
    else:
        silver_orders.write \
            .format("delta") \
            .mode("overwrite") \
            .partitionBy("order_date") \
            .save("s3://lake/silver/orders")
    
    print(f"Silver layer: {silver_orders.count()} records processed")

# ========== 金层 (Gold) ==========
def process_to_gold():
    """从银层聚合数据到金层"""
    
    # 读取银层数据
    silver_orders = spark.read.format("delta").load("s3://lake/silver/orders")
    
    # 按日聚合
    daily_metrics = silver_orders \
        .groupBy("order_date") \
        .agg(
            F.count("order_id").alias("total_orders"),
            F.countDistinct("user_id").alias("unique_users"),
            F.sum("amount").alias("total_revenue"),
            F.avg("amount").alias("avg_order_amount"),
            F.max("amount").alias("max_order_amount")
        )
    
    # 写入金层（覆盖模式）
    daily_metrics.write \
        .format("delta") \
        .mode("overwrite") \
        .save("s3://lake/gold/daily_metrics")
    
    # 按用户聚合
    user_metrics = silver_orders \
        .groupBy("user_id") \
        .agg(
            F.count("order_id").alias("total_orders"),
            F.sum("amount").alias("total_spent"),
            F.avg("amount").alias("avg_order_amount"),
            F.min("order_date").alias("first_order_date"),
            F.max("order_date").alias("last_order_date")
        ) \
        .withColumn("customer_tenure_days",
            F.datediff(F.current_date(), F.col("first_order_date"))
        )
    
    user_metrics.write \
        .format("delta") \
        .mode("overwrite") \
        .save("s3://lake/gold/user_metrics")
    
    print(f"Gold layer: daily_metrics={daily_metrics.count()}, user_metrics={user_metrics.count()}")

# 运行完整的 Medallion 管道
ingest_to_bronze()
process_to_silver()
process_to_gold()
```

---

## 5.4 数据治理与合规 🔴

### 5.4.1 数据治理框架

📌 **关键概念**：数据治理（Data Governance）是确保数据资产被正确管理、使用和保护的一系列流程和政策。在AI系统中，数据治理尤为重要，因为它直接影响模型的可靠性和合规性。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    数据治理框架                                       │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    组织与策略                                  │  │
│  │                                                              │  │
│  │  数据所有者  │  数据管家  │  数据使用者  │  合规官            │  │
│  │  (Owner)    │  (Steward)│  (Consumer) │  (Compliance)     │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    治理流程                                   │  │
│  │                                                              │  │
│  │  数据分类  │  访问控制  │  质量监控  │  审计日志  │  生命周期 │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    技术实现                                   │  │
│  │                                                              │  │
│  │  元数据管理  │  数据目录  │  血缘追踪  │  加密脱敏           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4.2 数据分类与分级

📌 **关键概念**：数据分类分级是数据治理的基础，它决定了数据的访问控制、加密策略和合规要求。

| 数据级别 | 描述 | 示例 | 保护措施 |
|---------|------|------|---------|
| **公开数据** | 可公开访问 | 产品信息、新闻 | 无特殊保护 |
| **内部数据** | 仅内部员工可访问 | 内部报表、日志 | 基础访问控制 |
| **机密数据** | 部分人员可访问 | 用户PII、财务数据 | 加密+访问控制+审计 |
| **绝密数据** | 严格限制访问 | 密钥、核心算法 | 最高安全级别 |

```python
# 示例：数据分类与访问控制实现
from dataclasses import dataclass
from typing import List, Set
from enum import Enum

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

@dataclass
class DataAsset:
    name: str
    classification: DataClassification
    owner: str
    description: str
    tags: List[str]
    retention_days: int

class DataGovernance:
    """数据治理管理器"""
    
    def __init__(self):
        self.assets: dict = {}
        self.access_policies: dict = {}
    
    def register_asset(self, asset: DataAsset):
        """注册数据资产"""
        self.assets[asset.name] = asset
        print(f"Registered: {asset.name} ({asset.classification.value})")
    
    def set_access_policy(
        self,
        asset_name: str,
        allowed_roles: List[str],
        allowed_users: List[str]
    ):
        """设置访问策略"""
        self.access_policies[asset_name] = {
            "allowed_roles": allowed_roles,
            "allowed_users": allowed_users
        }
    
    def check_access(
        self,
        asset_name: str,
        user_role: str,
        user_id: str
    ) -> bool:
        """检查访问权限"""
        if asset_name not in self.assets:
            return False
        
        policy = self.access_policies.get(asset_name, {})
        
        # 检查角色权限
        if user_role in policy.get("allowed_roles", []):
            return True
        
        # 检查用户权限
        if user_id in policy.get("allowed_users", []):
            return True
        
        # 公开数据默认允许
        if self.assets[asset_name].classification == DataClassification.PUBLIC:
            return True
        
        return False
    
    def get_compliance_requirements(self, asset_name: str) -> dict:
        """获取合规要求"""
        asset = self.assets.get(asset_name)
        if not asset:
            return {}
        
        requirements = {
            "retention_days": asset.retention_days,
            "encryption_required": asset.classification in [
                DataClassification.CONFIDENTIAL,
                DataClassification.SECRET
            ],
            "audit_logging": asset.classification in [
                DataClassification.CONFIDENTIAL,
                DataClassification.SECRET
            ],
            "pii_masking": "PII" in asset.tags,
            "gdpr_compliant": "PII" in asset.tags,
        }
        
        return requirements

# 使用示例
governance = DataGovernance()

# 注册数据资产
governance.register_asset(DataAsset(
    name="user_profiles",
    classification=DataClassification.CONFIDENTIAL,
    owner="data-team",
    description="用户画像数据，包含PII信息",
    tags=["PII", "user-data"],
    retention_days=365
))

governance.register_asset(DataAsset(
    name="public_products",
    classification=DataClassification.PUBLIC,
    owner="product-team",
    description="公开商品信息",
    tags=["product"],
    retention_days=-1  # 永久保留
))

# 设置访问策略
governance.set_access_policy(
    "user_profiles",
    allowed_roles=["data-scientist", "data-analyst"],
    allowed_users=["admin@company.com"]
)

# 检查访问权限
print(governance.check_access("user_profiles", "data-scientist", "user1"))  # True
print(governance.check_access("user_profiles", "intern", "user2"))  # False
print(governance.check_access("public_products", "intern", "user2"))  # True

# 获取合规要求
print(governance.get_compliance_requirements("user_profiles"))
# {'retention_days': 365, 'encryption_required': True, 'audit_logging': True,
#  'pii_masking': True, 'gdpr_compliant': True}
```

### 5.4.3 GDPR/CCPA 合规实现

📌 **关键概念**：GDPR（欧盟通用数据保护条例）和 CCPA（加州消费者隐私法案）是全球最重要的数据隐私法规，对AI系统的数据处理提出了严格要求。

| GDPR要求 | 技术实现 | Delta Lake支持 |
|---------|---------|---------------|
| **数据访问权** | 数据目录+血缘追踪 | Time Travel追溯数据来源 |
| **数据删除权** | 数据擦除机制 | VACUUM命令清理旧版本 |
| **数据可携带权** | 标准格式导出 | Parquet格式导出 |
| **数据最小化** | 列级访问控制 | Delta Lake Column Masking |
| **处理记录** | 审计日志 | Delta Lake操作日志 |

```python
# 示例：GDPR合规实现
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
import hashlib

spark = SparkSession.builder \
    .appName("GDPRCompliance") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .getOrCreate()

class GDPRComplianceManager:
    """GDPR合规管理器"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def data_subject_access_request(
        self,
        table_path: str,
        user_id: str
    ) -> dict:
        """数据主体访问请求 (DSAR) - 返回用户的所有数据"""
        
        # 读取当前数据
        current_data = self.spark.read.format("delta").load(table_path) \
            .filter(f"user_id = '{user_id}'")
        
        # 读取历史数据（通过时间旅行）
        history = DeltaTable.forPath(self.spark, table_path)
        
        return {
            "current_data": current_data.toPandas().to_dict(),
            "record_count": current_data.count(),
            "note": "Historical data available via Delta Lake Time Travel"
        }
    
    def right_to_erasure(
        self,
        table_path: str,
        user_id: str
    ):
        """数据删除权 (Right to Erasure)"""
        
        delta_table = DeltaTable.forPath(self.spark, table_path)
        
        # 删除用户数据
        delta_table.delete(f"user_id = '{user_id}'")
        
        # 清理旧版本（保留最近7天）
        delta_table.vacuum(retentionHours=168)
        
        print(f"Erased data for user: {user_id}")
    
    def data_minimization(
        self,
        source_table: str,
        target_table: str,
        allowed_columns: List[str]
    ):
        """数据最小化 - 只保留必要列"""
        
        source_df = self.spark.read.format("delta").load(source_table)
        
        # 只选择允许的列
        minimized_df = source_df.select(*allowed_columns)
        
        # 写入目标表
        minimized_df.write \
            .format("delta") \
            .mode("overwrite") \
            .save(target_table)
        
        print(f"Minimized: {source_table} -> {target_table}")
    
    def audit_log(
        self,
        operation: str,
        user_id: str,
        table_path: str,
        details: dict
    ):
        """审计日志记录"""
        
        audit_record = {
            "timestamp": self.spark.sql("SELECT current_timestamp()").collect()[0][0],
            "operation": operation,
            "user_id": user_id,
            "table_path": table_path,
            "details": str(details),
            "session_id": self.spark.sparkContext.applicationId
        }
        
        # 写入审计日志表
        self.spark.createDataFrame([audit_record]) \
            .write \
            .format("delta") \
            .mode("append") \
            .save("s3://audit-logs/gdpr_operations")
        
        print(f"Audit log recorded: {operation}")

# 使用示例
gdpr_manager = GDPRComplianceManager(spark)

# 数据主体访问请求
dsar = gdpr_manager.data_subject_access_request(
    "s3://lake/silver/user_profiles",
    "user_12345"
)
print(f"Records found: {dsar['record_count']}")

# 数据删除
gdpr_manager.right_to_erasure(
    "s3://lake/silver/user_profiles",
    "user_12345"
)

# 数据最小化
gdpr_manager.data_minimization(
    "s3://lake/silver/user_profiles",
    "s3://lake/silver/user_profiles_minimized",
    ["user_id", "user_segment", "registration_date"]
)
```

---

## 5.5 AI 工作负载的数据架构优化 🔴

### 5.5.1 AI 工作负载的特殊需求

📌 **关键概念**：AI/ML 工作负载对数据架构有独特的需求，与传统 BI 工作负载显著不同。

| 维度 | BI 工作负载 | AI/ML 工作负载 |
|------|-----------|--------------|
| **查询模式** | 预定义聚合查询 | 探索性分析、特征工程 |
| **数据量** | 聚合后的数据 | 原始数据（全量） |
| **数据格式** | 结构化 | 结构化+非结构化 |
| **延迟要求** | 秒~分钟 | 小时~天 |
| **计算模式** | SQL查询 | 矩阵运算、分布式训练 |
| **数据新鲜度** | T+1 | 实时~小时级 |
| **迭代需求** | 低 | 高（反复实验） |

### 5.5.2 ML 特化数据架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI 工作负载数据架构优化                             │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    数据获取层                                  │  │
│  │                                                              │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐            │  │
│  │  │  训练数据   │  │  验证数据   │  │  测试数据   │            │  │
│  │  │  (历史快照) │  │  (时间切分) │  │  (时间切分) │            │  │
│  │  └────────────┘  └────────────┘  └────────────┘            │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    数据准备层                                  │  │
│  │                                                              │  │
│  │  特征工程    │  数据增强    │  数据清洗    │  标注管理         │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    存储优化层                                  │  │
│  │                                                              │  │
│  │  小文件合并  │  数据压缩    │  列裁剪     │  谓词下推         │  │
│  │  (Compaction)│  (Z-Order)  │  (Pruning)  │  (Pushdown)     │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    训练数据服务层                               │  │
│  │                                                              │  │
│  │  TF Record  │  Petastorm   │  WebDataset  │  Delta Lake    │  │
│  │  (TensorFlow)│  (Spark→DL) │  (Web)       │  (通用)        │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.5.3 Delta Lake AI 优化技术

```python
# 示例：Delta Lake AI 工作负载优化
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("DeltaLakeAIOptimization") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .getOrCreate()

# ========== 1. 小文件合并 (Compaction) ==========
def compact_small_files(table_path: str, target_file_size_mb: int = 128):
    """合并小文件，提高读取性能"""
    
    delta_table = DeltaTable.forPath(spark, table_path)
    
    # 执行合并操作
    delta_table.optimize().executeCompaction()
    
    print(f"Compacted {table_path}")

# ========== 2. Z-Order 优化 (数据布局优化) ==========
def zorder_optimize(table_path: str, columns: list):
    """Z-Order 优化，提高多列查询性能"""
    
    delta_table = DeltaTable.forPath(spark, table_path)
    
    # 执行 Z-Order 优化
    delta_table.optimize().executeZOrderBy(*columns)
    
    print(f"Z-Order optimized on columns: {columns}")

# ========== 3. 数据布局统计 ==========
def analyze_table_stats(table_path: str):
    """分析表的统计信息"""
    
    delta_table = DeltaTable.forPath(spark, table_path)
    
    # 获取表详情
    detail = delta_table.detail()
    detail.show(truncate=False)
    
    # 获取优化历史
    history = delta_table.history()
    history.show(truncate=False)

# ========== 4. 训练数据准备优化 ==========
def prepare_training_data_optimized(
    table_path: str,
    feature_columns: list,
    label_column: str,
    output_path: str
):
    """优化训练数据准备流程"""
    
    # 读取数据（利用谓词下推和列裁剪）
    training_df = spark.read \
        .format("delta") \
        .load(table_path) \
        .select(feature_columns + [label_column])
    
    # 数据统计
    stats = training_df.summary().collect()
    print("Training data statistics:")
    for row in stats:
        print(f"  {row['summary']}: {row.asDict()}")
    
    # 写入优化格式
    training_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("compression", "zstd") \
        .save(output_path)
    
    # 分析新表
    optimize_table(output_path)

def optimize_table(table_path: str):
    """表优化全流程"""
    
    # 1. 合并小文件
    compact_small_files(table_path, target_file_size_mb=128)
    
    # 2. Z-Order 优化（如果有常用过滤列）
    delta_table = DeltaTable.forPath(spark, table_path)
    columns = [col.name for col in delta_table.toDF().columns[:3]]  # 前3列
    zorder_optimize(table_path, columns)

# ========== 5. 时间旅行用于ML实验 ==========
def time_travel_ml_experiment(
    table_path: str,
    experiment_date: str,
    feature_columns: list
):
    """使用时间旅行获取特定时间点的训练数据"""
    
    # 读取特定时间点的数据
    training_df = spark.read \
        .format("delta") \
        .option("timestampAsOf", experiment_date) \
        .load(table_path) \
        .select(feature_columns)
    
    print(f"Loaded {training_df.count()} records for date: {experiment_date}")
    
    return training_df

# ========== 6. 数据版本管理 for ML ==========
def ml_data_versioning(table_path: str, version: int):
    """ML数据版本管理"""
    
    # 读取特定版本
    versioned_df = spark.read \
        .format("delta") \
        .option("versionAsOf", version) \
        .load(table_path)
    
    # 记录版本信息
    version_info = {
        "version": version,
        "record_count": versioned_df.count(),
        "schema": str(version_df.schema),
        "timestamp": spark.sql("SELECT current_timestamp()").collect()[0][0]
    }
    
    # 写入版本元数据
    spark.createDataFrame([version_info]) \
        .write \
        .format("delta") \
        .mode("append") \
        .save(f"{table_path}_versions")
    
    return version_info

# 运行示例
table_path = "s3://lake/silver/training_data"

# 优化表
optimize_table(table_path)

# Z-Order优化常用查询列
zorder_optimize(table_path, ["user_id", "event_date", "category"])

# 准备训练数据
prepare_training_data_optimized(
    table_path=table_path,
    feature_columns=["feature_1", "feature_2", "feature_3"],
    label_column="label",
    output_path="s3://lake/gold/training_data"
)
```

### 5.5.4 数据缓存与预热策略

| 策略 | 描述 | 适用场景 | 实现方式 |
|------|------|---------|---------|
| **数据预取** | 提前加载训练数据到本地 | 定时训练任务 | Airflow + Spark |
| **缓存共享** | 多个训练任务共享缓存 | 多GPU训练 | Alluxio/Redis |
| **增量加载** | 只加载变化的数据 | 增量训练 | Delta Lake CDC |
| **分区缓存** | 缓存热点分区 | 高频查询分区 | Spark Cache |

---

## 💡 案例：基于 Delta Lake 的湖仓架构

### 场景描述

某互联网公司需要构建统一的数据平台，支持：
- **BI分析**：运营团队的日报、周报、月报
- **AI训练**：推荐模型、风控模型的训练数据
- **实时监控**：业务指标的实时大屏
- **数据探索**：数据科学家的自助分析

### 架构设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    基于 Delta Lake 的湖仓架构                             │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    数据源层                                        │  │
│  │                                                                  │  │
│  │  MySQL         Kafka         S3           外部数据               │  │
│  │  (核心业务)    (行为日志)     (历史归档)    (市场数据)             │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────┼───────────────────────────────────────┐  │
│  │                    数据接入层                                      │  │
│  │                                                                  │  │
│  │  Debezium CDC   Kafka Connect   Airflow      API Adapter        │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────┼───────────────────────────────────────┐  │
│  │                    Bronze 层 (原始数据)                            │  │
│  │                                                                  │  │
│  │  s3://lake/bronze/                                                │  │
│  │  ├── orders/          (订单原始数据)                               │  │
│  │  ├── user_events/     (用户行为日志)                               │  │
│  │  ├── products/        (商品信息)                                   │  │
│  │  └── external/        (外部数据)                                   │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────┼───────────────────────────────────────┐  │
│  │                    Silver 层 (清洗标准化)                          │  │
│  │                                                                  │  │
│  │  s3://lake/silver/                                                │  │
│  │  ├── orders/          (清洗后的订单数据)                           │  │
│  │  ├── user_profiles/   (用户画像)                                   │  │
│  │  ├── product_catalog/ (商品目录)                                   │  │
│  │  └── user_behavior/   (标准化行为数据)                             │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────┼───────────────────────────────────────┐  │
│  │                    Gold 层 (聚合业务层)                            │  │
│  │                                                                  │  │
│  │  s3://lake/gold/                                                  │  │
│  │  ├── daily_metrics/    (每日指标)                                  │  │
│  │  ├── user_segments/    (用户分群)                                  │  │
│  │  ├── product_rankings/ (商品排行)                                  │  │
│  │  └── training_data/    (训练数据集)                                │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│              ┌──────────────┼──────────────┐                           │
│              ▼              ▼              ▼                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │
│  │  BI 分析    │  │  AI 训练    │  │  实时监控   │                   │
│  │             │  │             │  │             │                   │
│  │  Trino     │  │  Spark      │  │  Flink      │                   │
│  │  + Superset│  │  + MLflow   │  │  + Grafana  │                   │
│  └─────────────┘  └─────────────┘  └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心实现代码

```python
# lakehouse_config.py - 湖仓架构配置
import os
from dataclasses import dataclass

@dataclass
class LakehouseConfig:
    """湖仓架构配置"""
    
    # 存储路径
    bronze_path: str = "s3://lake/bronze"
    silver_path: str = "s3://lake/silver"
    gold_path: str = "s3://lake/gold"
    
    # 数据源
    mysql_host: str = "mysql-cluster"
    kafka_bootstrap: str = "kafka:9092"
    
    # 计算引擎
    spark_config: dict = None
    
    def __post_init__(self):
        if self.spark_config is None:
            self.spark_config = {
                "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
                "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
                "spark.databricks.delta.optimizeWrite.enabled": "true",
                "spark.databricks.delta.autoCompact.enabled": "true",
                "spark.sql.shuffle.partitions": "200",
                "spark.sql.adaptive.enabled": "true",
            }

# pipeline.py - 完整数据管道
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable
from datetime import datetime, timedelta

class LakehousePipeline:
    """湖仓数据管道"""
    
    def __init__(self, config: LakehouseConfig):
        self.config = config
        self.spark = self._create_spark_session()
    
    def _create_spark_session(self) -> SparkSession:
        """创建Spark会话"""
        builder = SparkSession.builder.appName("LakehousePipeline")
        for key, value in self.config.spark_config.items():
            builder = builder.config(key, value)
        return builder.getOrCreate()
    
    # ========== Bronze 层操作 ==========
    def ingest_orders_to_bronze(self):
        """将订单数据导入铜层"""
        
        # 从Kafka读取订单流
        orders_stream = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.config.kafka_bootstrap) \
            .option("subscribe", "raw_orders") \
            .option("startingOffsets", "latest") \
            .load()
        
        # 解析JSON
        parsed_orders = orders_stream \
            .selectExpr("CAST(value AS STRING)") \
            .select(F.from_json(F.col("value"), order_schema).alias("data")) \
            .select("data.*") \
            .withColumn("_ingestion_time", F.current_timestamp()) \
            .withColumn("_event_date", F.to_date("created_at"))
        
        # 写入铜层
        query = parsed_orders.writeStream \
            .format("delta") \
            .outputMode("append") \
            .partitionBy("_event_date") \
            .option("checkpointLocation", f"{self.config.bronze_path}/orders/_checkpoint") \
            .start(f"{self.config.bronze_path}/orders")
        
        return query
    
    # ========== Silver 层操作 ==========
    def process_orders_to_silver(self):
        """从铜层处理订单到银层"""
        
        # 读取铜层数据
        bronze_orders = self.spark.read \
            .format("delta") \
            .load(f"{self.config.bronze_path}/orders")
        
        # 数据清洗
        silver_orders = bronze_orders \
            .filter(F.col("order_id").isNotNull()) \
            .filter(F.col("amount") > 0) \
            .dropDuplicates(["order_id"]) \
            .withColumn("order_date", F.to_date("created_at")) \
            .withColumn("order_hour", F.hour("created_at")) \
            .withColumn("amount_cNY", F.col("amount")) \
            .withColumn("amount_usd", F.col("amount") * 0.14)  # 汇率转换
        
        # 合并写入银层（支持upsert）
        if DeltaTable.isDeltaTable(self.spark, f"{self.config.silver_path}/orders"):
            delta_table = DeltaTable.forPath(self.spark, f"{self.config.silver_path}/orders")
            
            delta_table.alias("target").merge(
                silver_orders.alias("source"),
                "target.order_id = source.order_id"
            ).whenMatchedUpdateAll() \
             .whenNotMatchedInsertAll() \
             .execute()
        else:
            silver_orders.write \
                .format("delta") \
                .mode("overwrite") \
                .partitionBy("order_date") \
                .save(f"{self.config.silver_path}/orders")
        
        # 优化表
        self._optimize_table(f"{self.config.silver_path}/orders")
    
    # ========== Gold 层操作 ==========
    def generate_gold_metrics(self):
        """生成金层指标"""
        
        # 读取银层数据
        silver_orders = self.spark.read \
            .format("delta") \
            .load(f"{self.config.silver_path}/orders")
        
        # 每日指标
        daily_metrics = silver_orders \
            .groupBy("order_date") \
            .agg(
                F.count("order_id").alias("total_orders"),
                F.countDistinct("user_id").alias("unique_users"),
                F.sum("amount_cNY").alias("total_revenue_cny"),
                F.sum("amount_usd").alias("total_revenue_usd"),
                F.avg("amount_cNY").alias("avg_order_amount"),
                F.countDistinct("merchant_id").alias("unique_merchants")
            )
        
        daily_metrics.write \
            .format("delta") \
            .mode("overwrite") \
            .save(f"{self.config.gold_path}/daily_metrics")
        
        # 用户指标
        user_metrics = silver_orders \
            .groupBy("user_id") \
            .agg(
                F.count("order_id").alias("total_orders"),
                F.sum("amount_cNY").alias("total_spent"),
                F.avg("amount_cNY").alias("avg_order_amount"),
                F.min("order_date").alias("first_order_date"),
                F.max("order_date").alias("last_order_date"),
                F.countDistinct("merchant_id").alias("unique_merchants")
            )
        
        user_metrics.write \
            .format("delta") \
            .mode("overwrite") \
            .save(f"{self.config.gold_path}/user_metrics")
    
    # ========== 表优化 ==========
    def _optimize_table(self, table_path: str):
        """表优化"""
        try:
            delta_table = DeltaTable.forPath(self.spark, table_path)
            delta_table.optimize().executeCompaction()
            print(f"Optimized: {table_path}")
        except Exception as e:
            print(f"Optimization failed for {table_path}: {e}")
    
    # ========== 时间旅行 ==========
    def query_historical(
        self,
        table_path: str,
        timestamp: str
    ):
        """查询历史数据"""
        return self.spark.read \
            .format("delta") \
            .option("timestampAsOf", timestamp) \
            .load(table_path)
    
    # ========== 数据质量监控 ==========
    def monitor_data_quality(self, table_path: str) -> dict:
        """监控数据质量"""
        
        df = self.spark.read.format("delta").load(table_path)
        total_rows = df.count()
        
        quality_report = {
            "table": table_path,
            "total_rows": total_rows,
            "null_counts": {},
            "duplicate_check": {},
            "freshness": {}
        }
        
        # 空值检查
        for column in df.columns:
            null_count = df.filter(F.col(column).isNull()).count()
            quality_report["null_counts"][column] = {
                "count": null_count,
                "percentage": null_count / total_rows * 100 if total_rows > 0 else 0
            }
        
        # 数据新鲜度
        if "created_at" in df.columns:
            latest_record = df.agg(F.max("created_at")).collect()[0][0]
            quality_report["freshness"]["latest_record"] = str(latest_record)
        
        return quality_report

# ========== 使用示例 ==========
config = LakehouseConfig()
pipeline = LakehousePipeline(config)

# 运行数据管道
pipeline.ingest_orders_to_bronze()
pipeline.process_orders_to_silver()
pipeline.generate_gold_metrics()

# 监控数据质量
quality = pipeline.monitor_data_quality(f"{config.silver_path}/orders")
print(f"Data quality report: {quality}")

# 查询历史数据
historical_data = pipeline.query_historical(
    f"{config.silver_path}/orders",
    "2026-01-15"
)
print(f"Historical records: {historical_data.count()}")
```

---

## 本章小结

| 主题 | 核心要点 |
|------|---------|
| **架构对比** | 数据仓库=结构化+强一致，数据湖=灵活+低成本，湖仓一体=两者兼得 |
| **表格式** | Delta Lake最易用，Iceberg演进最好，Hudi流式最优 |
| **统一分析** | Medallion架构分层，批流一体处理 |
| **数据治理** | 分类分级+访问控制+GDPR合规 |
| **AI优化** | 小文件合并+Z-Order+时间旅行+数据版本管理 |

## 📝 练习

### 练习1：湖仓架构设计（🟢 初级）
为一个电商平台设计数据湖仓架构：
- 设计 Bronze/Silver/Gold 三层数据模型
- 选择合适的表格式（Delta Lake/Iceberg/Hudi）
- 画出完整的架构图

### 练习2：数据管道实现（🟡 中级）
使用 Delta Lake 实现完整的 Medallion 数据管道：
- 铜层：从Kafka摄入原始数据
- 银层：数据清洗、去重、标准化
- 金层：业务指标聚合
- 添加表优化和数据质量监控

### 练习3：AI数据架构优化（🔴 高级）
为大规模ML训练优化数据架构：
- 实现小文件合并策略
- 设计Z-Order优化方案
- 构建时间旅行数据版本管理
- 实现增量训练数据服务

---

> **下一章预告**：第六章将深入探讨模型架构设计，包括模型服务化部署、模型版本管理、A/B测试框架，以及基于 MLflow 的 MLOps 实战。
