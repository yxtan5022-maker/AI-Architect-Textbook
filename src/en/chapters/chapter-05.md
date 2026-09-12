# Chapter 5: Data Lakehouse Architecture

## 数据湖仓架构

---

## Learning Objectives

By the end of this chapter, you will be able to:

- **Architect a data lakehouse** that combines the flexibility of data lakes with the reliability of data warehouses
- **Compare table formats** (Delta Lake, Apache Iceberg, Apache Hudi) on performance, features, and ecosystem compatibility
- **Implement the Medallion Architecture** for organizing data into bronze, silver, and gold layers
- **Prevent data swamp formation** through governance, quality, and organizational practices
- **Design a lakehouse for AI/ML workloads** that supports both analytical queries and model training

---

## 5.1 The Data Lakehouse: Convergence of Lake and Warehouse

The data lakehouse is an architectural pattern that unifies data lakes and data warehouses. It provides:

- **Data lake flexibility**: Store any data format (structured, semi-structured, unstructured) at low cost
- **Data warehouse reliability**: ACID transactions, schema enforcement, and time travel
- **Open formats**: No vendor lock-in; data stored in open file formats (Parquet, ORC)
- **Direct access**: BI tools and ML frameworks can access data directly without ETL

### The Problem with Data Lakes Alone

Data lakes promised cheap storage of any data. In practice, many became **data swamps** — disorganized repositories where data was stored but rarely used:

| Problem | Consequence |
|---------|------------|
| No schema enforcement | Inconsistent data quality |
| No ACID transactions | Partial writes corrupt datasets |
| No time travel | Cannot audit or rollback changes |
| No data lineage | Cannot trace data origins |
| Metadata chaos | Data discovery is impossible |

### The Problem with Data Warehouses Alone

Data warehouses solved reliability but introduced constraints:

| Problem | Consequence |
|---------|------------|
| Expensive storage | $25-100/TB/month vs. $0.02/TB/month (cloud storage) |
| Schema-on-write | Rigid, cannot handle semi-structured data |
| Limited data types | Cannot store images, audio, video, JSON natively |
| Proprietary formats | Vendor lock-in |
| Scaling limitations | Vertical scaling is expensive |

### The Lakehouse Solution

The lakehouse adds a **metadata layer** on top of open file formats in cloud/object storage:

```
┌─────────────────────────────────────────────────┐
│              Data Lakehouse Architecture          │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────────────────────────────────┐    │
│  │           Metadata Layer                │    │
│  │   (Delta Lake / Iceberg / Hudi)        │    │
│  │   - ACID transactions                  │    │
│  │   - Schema enforcement                 │    │
│  │   - Time travel                        │    │
│  │   - Data lineage                       │    │
│  └─────────────────────────────────────────┘    │
│                       │                           │
│  ┌─────────────────────────────────────────┐    │
│  │         Open File Formats               │    │
│  │         (Parquet / ORC)                 │    │
│  └─────────────────────────────────────────┘    │
│                       │                           │
│  ┌─────────────────────────────────────────┐    │
│  │       Cloud Object Storage              │    │
│  │    (S3 / GCS / ADLS / MinIO)           │    │
│  └─────────────────────────────────────────┘    │
│                                                   │
│  Query Engines: Spark, Trino, Presto, Dremio     │
│  BI Tools: Tableau, Power BI, Looker             │
│  ML Frameworks: PyTorch, TensorFlow, XGBoost     │
└─────────────────────────────────────────────────┘
```

---

## 5.2 Table Format Comparison

### 📌 Real Data: Delta Lake, Iceberg, and Hudi

| Feature | Delta Lake | Apache Iceberg | Apache Hudi |
|---------|-----------|---------------|-------------|
| **Governance** | Linux Foundation | Apache Software Foundation | Apache Software Foundation |
| **GitHub Stars** | 7,500+ | 6,500+ | 5,800+ |
| **GitHub Forks** | 1,300+ | 2,300+ | 2,100+ |
| **Primary Backer** | Databricks | Netflix → Apache community | Uber → Apache community |
| **ACID Transactions** | Yes | Yes | Yes |
| **Time Travel** | Yes | Yes | Yes |
| **Schema Evolution** | Limited | Full (add/drop/rename) | Full |
| **Partition Evolution** | No (rewrite needed) | Yes (hidden partitioning) | Yes |
| **Merge-on-Read** | Yes | Yes | Yes |
| **Copy-on-Write** | Yes | Yes | Yes |
| **Spark Integration** | Excellent (native) | Excellent | Excellent |
| **Trino/Presto** | Good | Excellent | Good |
| **Flink Integration** | Good | Growing | Excellent |
| **Best For** | Databricks ecosystem, Spark-native | Multi-engine, Iceberg REST catalog | CDC, incremental processing |

### Architecture Differences

**Delta Lake** stores transaction logs in a `_delta_log/` directory alongside Parquet files:

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

**Apache Iceberg** uses a hierarchical metadata structure:

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

**Apache Hudi** uses a timeline-based architecture:

```
my_table/
├── .hoodie/
│   ├── 20250101000000.commit
│   ├── 20250102000000.commit
│   └── 20250102000000.clean
├── part-00000-...parquet
└── part-00001-...parquet
```

### Performance Comparison

Based on published benchmarks from Databricks, Netflix, and Uber engineering:

| Operation | Delta Lake | Iceberg | Hudi |
|-----------|-----------|---------|------|
| **Small file compaction** | Auto-optimize | Manual compaction | Timeline-based compaction |
| **Partition pruning** | File-level statistics | Partition evolution | Partition pruning |
| **Predicate pushdown** | Parquet statistics | Parquet + partition | Parquet + partition |
| **Concurrent writes** | Optimistic concurrency |乐观 concurrency | OCC + timeline |
| **Time travel query** | Log replay | Snapshot-based | Timeline-based |
| **Schema evolution** | Add column only | Full | Full |
| **Upsert performance** | Moderate | Good | Excellent |
| **Read performance** | Excellent | Excellent | Good |

### When to Choose Each Format

| Scenario | Recommended Format | Reason |
|----------|-------------------|--------|
| Already using Databricks | Delta Lake | Native integration, best Spark support |
| Multi-engine (Spark + Trino + Flink) | Iceberg | Best cross-engine compatibility |
| CDC / change data processing | Hudi | Native incremental processing |
| PostgreSQL/MySQL replication | Hudi | Built-in CDC connectors |
| Multi-cloud deployment | Iceberg | REST catalog standard |
| Simple use case, Spark-only | Delta Lake | Simplest to set up |
| Time travel / audit requirements | Iceberg | Most mature snapshot isolation |

---

## 5.3 The Medallion Architecture

### 📌 Real Data: Medallion Architecture (Databricks)

The Medallion Architecture is a data design pattern introduced by Databricks (databricks.com) for organizing lakehouse data into progressive quality layers. It has become the de facto standard for lakehouse data organization.

### The Three Layers

```
┌─────────────────────────────────────────────────────────┐
│                 Medallion Architecture                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
│  │   BRONZE    │──▶│   SILVER    │──▶│    GOLD     │  │
│  │             │   │             │   │             │  │
│  │ Raw data    │   │ Cleansed &  │   │ Business-   │  │
│  │ as-is       │   │ conformed   │   │ level       │  │
│  │             │   │             │   │ aggregates  │  │
│  │ Full fidelity│  │ Deduplicated│   │             │  │
│  │ No transforms│  │ Validated   │   │ Optimized   │  │
│  │             │   │ Enriched    │   │ for queries │  │
│  └─────────────┘   └─────────────┘   └─────────────┘  │
│                                                           │
│  Quality: Raw        Quality: Clean      Quality: Curated │
│  Consumers:          Consumers:          Consumers:       │
│  Data Engineers      Data Scientists     BI Analysts     │
│                      ML Engineers        Dashboards       │
└─────────────────────────────────────────────────────────┘
```

### Bronze Layer: Raw Data

The Bronze layer stores data exactly as it arrives, with no transformations. This provides:

- **Full data fidelity**: Every record is preserved, including errors and duplicates
- **Auditability**: Complete history of what was ingested
- **Replay capability**: Re-process from raw data if transforms have bugs
- **Schema-on-read**: Data is stored in its original format

```python
# Bronze layer ingestion
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Bronze_Ingestion").getOrCreate()

# Read raw data from source
raw_events = spark.read \
    .format("json") \
    .option("inferSchema", "true") \
    .load("s3://raw-data/events/")

# Write to Bronze with full fidelity
raw_events.write \
    .format("delta") \
    .mode("append") \
    .partitionBy("date") \
    .save("s3://lakehouse/bronze/events/")
```

### Silver Layer: Cleansed and Conformed

The Silver layer applies quality rules and standardization:

```python
# Silver layer transformations
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower

spark = SparkSession.builder.appName("Silver_Transform").getOrCreate()

# Read from Bronze
bronze_events = spark.read.format("delta").load("s3://lakehouse/bronze/events/")

# Apply quality rules
silver_events = bronze_events \
    .filter(col("event_id").isNotNull()) \
    .filter(col("user_id").isNotNull()) \
    .dropDuplicates(["event_id"]) \
    .withColumn("event_type", lower(trim(col("event_type")))) \
    .withColumn("timestamp", col("timestamp").cast("timestamp"))

# Write to Silver
silver_events.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("date", "event_type") \
    .save("s3://lakehouse/silver/events/")
```

### Gold Layer: Business-Level Aggregates

The Gold layer contains business-level aggregates optimized for consumption:

```python
# Gold layer aggregation
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum, avg, window

spark = SparkSession.builder.appName("Gold_Aggregate").getOrCreate()

# Read from Silver
silver_events = spark.read.format("delta").load("s3://lakehouse/silver/events/")

# Compute business-level aggregates
gold_daily_metrics = silver_events \
    .groupBy("date", "event_type") \
    .agg(
        count("*").alias("event_count"),
        countDistinct("user_id").alias("unique_users"),
    )

# Write to Gold
gold_daily_metrics.write \
    .format("delta") \
    .mode("overwrite") \
    .save("s3://lakehouse/gold/daily_metrics/")
```

### Medallion Layer Responsibilities

| Layer | Data Quality | Consumers | Schema | Write Pattern | Cost |
|-------|-------------|-----------|--------|--------------|------|
| **Bronze** | Raw, unvalidated | Data engineers | Schema-on-read | Append | Lowest |
| **Silver** | Cleaned, deduplicated | Data scientists, ML | Schema-on-write | Merge/Upsert | Medium |
| **Gold** | Aggregated, curated | BI analysts, dashboards | Strict schema | Overwrite | Highest |

---

## 💡 Case Study: Airbnb's Data Lake Architecture

### The Problem

Airbnb operates one of the largest data platforms in the world, serving millions of listings across 220+ countries. Their data challenges include:

- **500+ petabytes** of data across thousands of datasets
- **Thousands of data pipelines** running daily
- **Multiple compute engines**: Spark, Presto, Hive
- **Diverse use cases**: Search ranking, pricing optimization, fraud detection, business analytics

### The Architecture

Airbnb's data lake architecture is built on three pillars:

1. **Doris** — Airbnb's internal data lake platform providing unified access to data across multiple storage systems
2. **Apache Airflow** — Orchestration for all data pipelines (Airflow was created at Airbnb)
3. **Presto/Trino** — Interactive SQL analytics over the data lake

### Real Data: Airbnb's Scale

| Metric | Value | Source |
|--------|-------|--------|
| Total data volume | 500+ PB | medium.com/airbnb-engineering |
| Daily pipeline runs | 10,000+ | Airbnb engineering blog |
| Data sets | 10,000+ | medium.com/airbnb-engineering |
| Query engine | Presto | Airbnb engineering blog |
| Orchestration | Airflow | airflow.apache.org |

### Key Design Decisions

**1. Schema-on-read over schema-on-write**: Airbnb stores raw data in its original format and applies schemas at query time. This provides flexibility for diverse data types and evolving schemas.

**2. Presto for interactive analytics**: Airbnb chose Presto (now Trino) for interactive SQL queries because it provides sub-second latency over petabyte-scale data without materializing results.

**3. Airflow for orchestration**: Airbnb created Airflow and continues to be its largest deployer. All data pipelines are defined as Airflow DAGs with Python.

**4. Cost optimization**: Airbnb uses a tiered storage strategy:
- Hot data: SSD-backed storage for frequently accessed datasets
- Warm data: Standard S3 storage for recent data
- Cold data: Glacier for historical archives

### Lessons Learned

1. **Metadata is as important as data**: Without robust metadata and cataloging, data lakes become data swamps. Airbnb invested heavily in data discovery tools.

2. **Presto democratized data access**: By providing a SQL interface over the data lake, non-technical users could query data without writing MapReduce or Spark jobs.

3. **Airflow created a self-service pipeline platform**: Engineers could create new data pipelines without platform team involvement, accelerating iteration.

4. **Cost management is critical**: At 500+ PB, even small optimizations in storage format or compression translate to millions of dollars in savings.

---

## 5.4 Preventing Data Swamps

### The Data Swamp Problem

A data swamp is a data lake that has lost its utility. Signs include:

- Data is ingested but never queried
- No documentation or metadata
- Duplicate datasets with different schemas
- No data quality guarantees
- Cannot find data when needed

### Prevention Strategies

| Strategy | Implementation | Impact |
|----------|---------------|--------|
| **Data catalog** | Apache Atlas, DataHub, Amundsen | Discoverable data |
| **Schema registry** | Confluent Schema Registry, AWS Glue | Schema governance |
| **Data quality** | Great Expectations, dbt tests | Trustworthy data |
| **Access policies** | Role-based access control | Security and compliance |
| **Data lineage** | Apache Atlas, OpenLineage | Auditability |
| **Cost monitoring** | Cloud cost dashboards | Cost control |
| **Automated archival** | Lifecycle policies | Storage optimization |

### Data Governance Framework

```
┌─────────────────────────────────────────────────────┐
│              Data Governance Framework               │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Data    │    │  Schema  │    │  Data    │      │
│  │ Catalog  │    │ Registry │    │ Quality  │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Access  │    │  Data    │    │  Cost    │      │
│  │ Control  │    │ Lineage  │    │ Monitor  │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### Apache Atlas: Metadata Governance

Apache Atlas (atlas.apache.org) provides metadata governance for Hadoop and beyond:

- **Type system**: Define metadata types for datasets, processes, and users
- **Lineage tracking**: Track how data flows through pipelines
- **Classification**: Tag data with sensitivity levels and business terms
- **Governance**: Enforce policies for data access and retention

---

## 5.5 Lakehouse for AI/ML Workloads

### Supporting Both Analytics and ML

The lakehouse must serve two distinct workloads:

1. **Analytical queries**: SQL-based, aggregations, joins, BI dashboards
2. **ML workloads**: Feature extraction, model training, batch inference

### Feature Extraction from Lakehouse

```python
# Extract ML features from the Silver layer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("ML_Feature_Extraction").getOrCreate()

# Read cleansed data from Silver
events = spark.read.format("delta").load("s3://lakehouse/silver/events/")

# Compute user-level features
user_features = events \
    .groupBy("user_id") \
    .agg(
        count("*").alias("total_events"),
        countDistinct("event_type").alias("unique_event_types"),
        avg("session_duration").alias("avg_session_duration"),
        max("timestamp").alias("last_event_time"),
    )

# Write features to a dedicated feature table
user_features.write \
    .format("delta") \
    .mode("overwrite") \
    .save("s3://lakehouse/gold/ml_features/user_features/")
```

### Model Training Data Preparation

```python
# Prepare training dataset with point-in-time correctness
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Training_Data").getOrCreate()

# Read features
features = spark.read.format("delta").load("s3://lakehouse/gold/ml_features/")
labels = spark.read.format("delta").load("s3://lakehouse/silver/labels/")

# Join with point-in-time correctness
training_data = features \
    .join(labels, on="user_id", how="inner") \
    .filter(col("feature_date") <= col("label_date"))

# Split and save
training_data.write.format("delta").mode("overwrite") \
    .save("s3://lakehouse/gold/ml_training/training_set/")
```

### Batch Inference Pipeline

```python
# Batch inference on lakehouse
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Batch_Inference").getOrCreate()

# Read latest features
latest_features = spark.read.format("delta") \
    .load("s3://lakehouse/gold/ml_features/")

# Load model (from MLflow or model registry)
# model = mlflow.spark.load_model("models:/churn_model/Production")

# Generate predictions
# predictions = model.transform(latest_features)

# Write predictions back to lakehouse
# predictions.write.format("delta").mode("overwrite") \
#     .save("s3://lakehouse/gold/predictions/churn_predictions/")
```

---

## ⚠️ War Story: How a Data Lake Became a Data Swamp

### Background

A large healthcare company built a data lake in 2019 to consolidate data from 15 different hospital systems. The initial goal was to enable population health analytics and predictive modeling.

### Year 1: The Promise

- 200+ datasets ingested
- Data stored in Parquet on S3
- Initial analytics dashboards built
- Executive sponsorship and budget approved

### Year 2: The Decline

Problems emerged:

1. **No schema enforcement**: Different hospital systems used different schemas for the same concepts (e.g., "patient_id" vs "pat_id" vs "mrn")

2. **No data quality checks**: Missing values, duplicates, and inconsistent formats went undetected

3. **No documentation**: New data scientists couldn't find or understand existing datasets

4. **Proliferation of duplicates**: Teams created their own copies of datasets with slight modifications, leading to 50+ versions of "patient demographics"

5. **Cost explosion**: Storage costs grew from $50K/month to $300K/month as raw data accumulated

### Year 3: The Swamp

- Only 15% of datasets were actively queried
- Average time to find the right dataset: 3 days
- Data scientists spent 80% of their time on data preparation
- Trust in data was near zero
- The data lake was jokingly referred to as "the data swamp"

### The Recovery

The company launched a 12-month data lake recovery program:

| Phase | Duration | Actions | Result |
|-------|----------|---------|--------|
| **Catalog** | Months 1-3 | Deployed Apache Atlas, cataloged all datasets | Data discoverable |
| **Quality** | Months 3-6 | Implemented Great Expectations, data quality scores | Trust increased |
| **Governance** | Months 6-9 | Access policies, schema registry, data ownership | Compliance |
| **Optimization** | Months 9-12 | Compacted small files, archived cold data, cost reduced 60% | Cost control |

### Key Lessons

1. **Technology is not the problem**: The company had all the right tools (S3, Parquet, Spark). The failure was organizational.

2. **Data governance must be proactive**: Waiting until the swamp forms costs 10x more than preventing it.

3. **Ownership matters**: Every dataset must have a named owner responsible for its quality and documentation.

4. **Cost monitoring prevents surprises**: Without cost tracking, storage costs can spiral unchecked.

---

## 5.6 Performance Optimization

### Small File Problem

The most common performance issue in data lakes is the accumulation of small files:

| File Size | Impact | Solution |
|-----------|--------|----------|
| < 1 MB | Terrible read performance | Compaction job |
| 1-64 MB | Suboptimal | Periodic compaction |
| 64-256 MB | Good | Target range |
| 256 MB - 1 GB | Optimal | Ideal range |
| > 1 GB | May cause OOM | Consider splitting |

### Compaction Strategy

```python
# Delta Lake compaction
from delta.tables import DeltaTable

# Auto-optimize (Databricks)
# SET spark.databricks.delta.optimizeWrite.enabled = true
# SET spark.databricks.delta.autoCompact.enabled = true

# Manual compaction
delta_table = DeltaTable.forPath(spark, "s3://lakehouse/silver/events/")
delta_table.optimize().executeCompaction()
```

### Partition Strategy

| Strategy | When to Use | Example |
|----------|------------|---------|
| **Date partitioning** | Time-series data, most common | `date=2025-01-01/` |
| **Category partitioning** | Low-cardinality categorical | `country=US/` |
| **No partitioning** | Small datasets (< 1GB) | Single directory |
| **Hive-style partitioning** | Legacy compatibility | `date=2025-01-01/hour=12/` |
| **Iceberg hidden partitioning** | Flexible, non-obvious partitioning | Iceberg spec |

### Z-Ordering (Data Clustering)

Z-ordering co-locates related data to improve query performance:

```python
# Delta Lake Z-ORDER
delta_table = DeltaTable.forPath(spark, "s3://lakehouse/silver/events/")
delta_table.optimize().executeZOrderBy("user_id", "event_type")
```

---

## 📝 When to Use / When Not to Use Lakehouse

| Scenario | Use Lakehouse? | Rationale |
|----------|---------------|-----------|
| Mixed analytics + ML workloads | Yes | Lakehouse excels at unifying both |
| SQL-only analytics, simple schema | No — use data warehouse | Warehouse is simpler and more performant |
| Unstructured data (images, video) | Yes | Lakehouse handles any data format |
| Real-time streaming only | Partially — use Kafka + lakehouse | Streaming needs dedicated infrastructure |
| Strict regulatory compliance | Maybe — depends on maturity | Warehouse may be simpler for compliance |
| Cost-sensitive, large data volumes | Yes | Lakehouse storage is 10-100x cheaper |
| Small team, simple needs | No — use managed warehouse | Complexity not justified |

---

## Summary

The data lakehouse represents the convergence of data lakes and data warehouses, offering the flexibility of the former with the reliability of the latter.

1. **Delta Lake, Apache Iceberg, and Apache Hudi** are the three leading table formats. Delta Lake excels in the Databricks ecosystem; Iceberg offers the best multi-engine compatibility; Hudi is optimal for CDC and incremental processing.

2. **The Medallion Architecture** (Bronze → Silver → Gold) provides a proven pattern for organizing lakehouse data by quality level, with each layer serving different consumers.

3. **Airbnb's architecture** demonstrates the state of the art: 500+ PB of data, 10,000+ daily pipeline runs, and a self-service platform built on Airflow and Presto.

4. **Data swamps form through organizational neglect, not technical failure.** Prevention requires proactive data governance: catalogs, quality checks, ownership, and cost monitoring.

5. **The lakehouse supports AI/ML workloads** through direct feature extraction, training data preparation, and batch inference — all leveraging the same data platform used for analytics.

---

## Discussion Questions

1. **Architecture Decision**: Your company has 200 TB of data, 80% structured (from a PostgreSQL database) and 20% unstructured (images and PDFs). You need to support both SQL analytics and image classification ML. Would you use a data warehouse, a data lake, or a lakehouse? Justify your decision.

2. **Format Selection**: You are starting a new project with Spark as the primary compute engine. You expect to add Trino for interactive queries in the future. Which table format would you choose and why?

3. **Cost Analysis**: A data lake stores 500 TB of data. If you implement compression (3x ratio) and compaction (reducing small files by 80%), how much would you save monthly at $0.023/GB S3 pricing?

4. **Governance**: Design a data governance framework for a healthcare company that needs to comply with HIPAA while enabling data science teams to discover and use patient data.

5. **Trade-offs**: Compare the operational complexity of managing a self-hosted Delta Lake/Iceberg setup vs. using a managed service like Databricks, Snowflake, or BigQuery. What are the total cost of ownership implications?

---

## Exercises

### Exercise 1: Lakehouse Setup (Hands-on)

1. Set up a local lakehouse using:
   - MinIO (S3-compatible storage)
   - Apache Spark with Delta Lake
   - Jupyter notebook

2. Implement the Medallion Architecture:
   - Bronze: Ingest a sample CSV dataset
   - Silver: Clean, deduplicate, and validate
   - Gold: Create business-level aggregates

3. Demonstrate time travel by querying historical versions

4. Document your setup and results

### Exercise 2: Table Format Comparison

Create a benchmark comparing Delta Lake, Iceberg, and Hudi:

1. Create identical datasets in all three formats
2. Measure: write throughput, read throughput, upsert performance, time travel query latency
3. Generate a comparison report with actual performance numbers

### Exercise 3: Data Swamp Prevention Plan

You have inherited a data lake with 500 datasets, no documentation, no quality checks, and unknown ownership. Design a 6-month recovery plan:

1. Prioritize which datasets to catalog first
2. Design a data quality scoring system
3. Create a data ownership assignment process
4. Define success metrics for the recovery program

---

## References

1. **Delta Lake Documentation** — docs.delta.io
2. **Apache Iceberg Documentation** — iceberg.apache.org/docs
3. **Apache Hudi Documentation** — hudi.apache.org
4. **Databricks: Medallion Architecture** — databricks.com/glossary/medallion-architecture
5. **Airbnb Engineering Blog** — medium.com/airbnb-engineering
6. **Apache Atlas Documentation** — atlas.apache.org/docs
7. **Great Expectations Documentation** — docs.greatexpectations.io
8. **lakeFS: Git for Data Lake** — lakefs.io
9. **Dremio: Data Lakehouse** — dremio.com
10. **Data Lake vs. Data Warehouse** (Martin Kleppmann) — dataintensive.net
