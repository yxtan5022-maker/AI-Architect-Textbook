# Chapter 5: Data Lakehouse Architecture

> **Learning Objectives**: After reading this chapter, you will be able to:
> 1. Understand the core differences between data lakes, data warehouses, and lakehouses
> 2. Compare table format technologies: Delta Lake, Apache Iceberg, Apache Hudi
> 3. Design unified analytics architectures supporting BI + AI hybrid workloads
> 4. Implement data governance and compliance management systems
> 5. Optimize data architectures for AI workloads
> 6. Build a complete lakehouse architecture based on Delta Lake

---

## 5.1 Data Lake vs. Data Warehouse vs. Lakehouse 🟢

### 5.1.1 Data Architecture Evolution

📌 **Key Concept**: Data architecture has evolved through three major stages: Data Warehouse → Data Lake → Data Lakehouse.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Data Architecture Evolution Timeline                │
│                                                                     │
│  2000s              2010s              2020s                        │
│  ────────           ────────           ────────                     │
│  Data Warehouse    Data Lake          Data Lakehouse                │
│                                                                     │
│  Characteristics:                                                   │
│  - Structured      - Any format       - Structured + Unstructured  │
│  - Schema-on-Write - Schema-on-Read   - Schema Evolution           │
│  - High cost       - Low cost         - Medium cost                │
│  - Strong          - Eventual         - ACID transactions          │
│    consistency       consistency                                     │
│  - Batch-focused   - Batch + Stream   - Batch + Stream + Interactive│
└─────────────────────────────────────────────────────────────────────┘
```

### 5.1.2 Three Architectures Compared

| Dimension | Data Warehouse | Data Lake | Lakehouse |
|-----------|---------------|-----------|-----------|
| **Data Format** | Structured (SQL tables) | Any format (files) | Structured + Semi-structured |
| **Schema Management** | Schema-on-Write | Schema-on-Read | Both supported |
| **Storage Cost** | High ($0.02-0.10/GB/month) | Low ($0.01-0.02/GB/month) | Medium |
| **Query Performance** | Fast (columnar + indexing) | Slow (full scans) | Fast (table format optimization) |
| **Data Freshness** | Hours~Days | Real-time~Minutes | Real-time~Minutes |
| **Transaction Support** | ✅ Strong consistency | ❌ No transactions | ✅ ACID transactions |
| **Schema Evolution** | Difficult | Flexible | Flexible and controlled |
| **Best For** | BI reports, OLAP | Data science, exploration | Unified BI+AI platform |
| **Typical Tools** | Snowflake, Redshift | S3, HDFS | Delta Lake, Iceberg |

### 5.1.3 Core Advantages of Lakehouse

📌 **Key Concept**: The Lakehouse architecture, coined by Databricks co-founder Ali Ghodsi, combines the low-cost flexibility of data lakes with the management capabilities of data warehouses.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Lakehouse Architecture Diagram                    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Application Layer                          │  │
│  │                                                              │  │
│  │  BI/OLAP      AI/ML       Data Science     Real-time        │  │
│  │  (Reports)   (Training)  (Exploration)    (Monitoring)     │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    SQL/Compute Engine Layer                   │  │
│  │                                                              │  │
│  │  Spark SQL   Presto/Trino   Flink   Delta Lake API          │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    Table Format Layer                         │  │
│  │                                                              │  │
│  │  Delta Lake    Apache Iceberg    Apache Hudi                │  │
│  │  (metadata, transactions, versioning)                        │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    Storage Layer                              │  │
│  │                                                              │  │
│  │  S3 / ADLS / GCS / HDFS / Local Filesystem                  │  │
│  │  (low-cost object storage)                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.1.4 Selection Decision Guide

```
Selection Decision Tree:

Need ACID transactions?
├── Yes → Need streaming writes?
│         ├── Yes → Apache Hudi (optimized for incremental processing)
│         └── No  → Need time travel?
│                   ├── Yes → Delta Lake / Apache Iceberg
│                   └── No  → Delta Lake (simpler)
└── No  → Need low-cost storage only?
          ├── Yes → Raw S3/HDFS
          └── No  → Need schema evolution?
                    ├── Yes → Apache Iceberg (best schema evolution)
                    └── No  → Delta Lake
```

---

## 5.2 Table Format Comparison 🟡

### 5.2.1 What Is a Table Format

📌 **Key Concept**: A Table Format is a metadata management layer between the file system and SQL engines. It defines how to organize small files into logical tables and provides capabilities like ACID transactions, time travel, and schema evolution.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Table Format Position                         │
│                                                                 │
│  ┌──────────────┐                                              │
│  │  SQL Engine   │  Spark / Presto / Flink / Trino             │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  Table Format│  Delta Lake / Iceberg / Hudi                │
│  │              │  (metadata, transactions, versions)          │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  File Format │  Parquet / ORC / Avro                       │
│  │              │  (columnar storage, compression)             │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                              │
│  │  Storage     │  S3 / HDFS / ADLS / GCS                    │
│  │              │  (distributed storage)                       │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2.2 Three Major Table Formats Compared

| Feature | Delta Lake | Apache Iceberg | Apache Hudi |
|---------|-----------|---------------|-------------|
| **ACID Transactions** | ✅ | ✅ | ✅ |
| **Time Travel** | ✅ (version/timestamp) | ✅ (snapshot ID) | ✅ (timestamp) |
| **Schema Evolution** | ✅ (supported but limited) | ✅ (best) | ✅ (limited) |
| **Partition Evolution** | ❌ (requires rewrite) | ✅ (hidden partitions) | ❌ (requires rewrite) |
| **Incremental Processing** | ✅ (Change Data Feed) | ✅ (incremental reads) | ✅ (optimized) |
| **Streaming Writes** | ✅ (Structured Streaming) | ✅ (Flink integration) | ✅ (optimized) |
| **Metadata Storage** | _delta_log/ (JSON) | metadata/ (Avro) | .hoodie/ (JSON) |
| **File Layout** | Parquet + log | Parquet + manifest | Parquet + index |
| **Community** | Databricks-led | Apache Foundation | Apache Foundation |
| **Learning Curve** | Low | Medium | Medium |
| **Production Ready** | High | High | High |

### 5.2.3 Internal Structure of Each Format

**Delta Lake Internal Structure**:
```
my_table/
├── _delta_log/
│   ├── 00000000000000000000.json  ← Version 0
│   ├── 00000000000000000001.json  ← Version 1
│   ├── 00000000000000000002.json  ← Version 2
│   └── _last_checkpoint           ← Latest checkpoint
├── part-00000-xxx.parquet         ← Data files
├── part-00001-xxx.parquet
└── part-00002-xxx.parquet
```

**Apache Iceberg Internal Structure**:
```
my_table/
├── metadata/
│   ├── v1.metadata.json          ← Table metadata
│   ├── v1.manifest-list-xxx.avro ← Manifest list
│   ├── v1-xxx.manifest           ← Manifest file
│   └── snap-xxx.avro             ← Snapshot
├── data/
│   ├── part-00000-xxx.parquet
│   └── part-00001-xxx.parquet
└── README.md
```

**Apache Hudi Internal Structure**:
```
my_table/
├── .hoodie/
│   ├── .hoodie_partition_metafile
│   ├── 00000000000000.commit       ← Commit metadata
│   ├── 00000000000000.clean        ← Clean metadata
│   └── .schema                     ← Schema file
├── part-00000-xxx.parquet
└── part-00001-xxx.parquet
```

---

## 5.3 Unified Analytics Architecture 🔴

### 5.3.1 Batch-Streaming Unified Architecture

📌 **Key Concept**: Unified Batch & Streaming refers to using the same code and architecture to process both batch and streaming data, avoiding the maintenance burden of two separate systems.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Batch-Streaming Unified Architecture               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Unified Ingestion Layer                     │  │
│  │                                                              │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐            │  │
│  │  │  CDC       │  │  Log       │  │  Batch     │            │  │
│  │  │ (Debezium) │  │  Stream    │  │  Files     │            │  │
│  │  │            │  │ (Kafka)    │  │ (S3/FTP)  │            │  │
│  │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘            │  │
│  └─────────┼───────────────┼───────────────┼───────────────────┘  │
│            │               │               │                      │
│            └───────────────┼───────────────┘                      │
│                            ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Unified Compute Engine                     │  │
│  │                                                              │  │
│  │  Apache Spark                                                │  │
│  │  ├── Structured Streaming (stream processing)                │  │
│  │  ├── Batch Processing (batch processing)                     │  │
│  │  └── DataFrame/SQL API (unified interface)                   │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Unified Storage (Delta Lake)               │  │
│  │                                                              │  │
│  │  Delta Lake Table                                           │  │
│  │  ├── Real-time data: Continuous writes via Structured Stream │  │
│  │  ├── Historical data: Batch backfill/recompute              │  │
│  │  └── Query views: Created on demand (full/incremental/point)│  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│              ┌──────────────┼──────────────┐                       │
│              ▼              ▼              ▼                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │  BI Reports │  │  AI         │  │  Real-time  │               │
│  │  (full snap)│  │  Training   │  │  Monitoring │               │
│  │             │  │  (point-in- │  │  (incremental)              │
│  │             │  │   time)     │  │             │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3.2 Multi-Engine Unified Analytics

📌 **Key Concept**: The lakehouse architecture must support multiple compute engines to meet diverse workload requirements.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Multi-Engine Unified Analytics                     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 Query/Compute Engine Layer                     │  │
│  │                                                              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │  │
│  │  │  Spark   │ │ Presto/  │ │  Flink   │ │  Trino   │       │  │
│  │  │  SQL     │ │  Trino   │ │  SQL     │ │          │       │  │
│  │  │          │ │          │ │          │ │          │       │  │
│  │  │ Batch    │ │ Interact.│ │ Stream   │ │ Interact.│       │  │
│  │  │ ML Train │ │ BI Report│ │ Real ETL │ │ Discovery│       │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │              Table Format Layer (Unified Metadata)            │  │
│  │                                                              │  │
│  │  Delta Lake / Iceberg / Hudi                                │  │
│  │  (ACID transactions, Schema management, versioning)         │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    Storage Layer                              │  │
│  │                                                              │  │
│  │  S3 / ADLS / GCS / HDFS                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3.3 Data Layering Architecture

📌 **Key Concept**: Medallion Architecture is a data organization pattern proposed by Databricks that organizes data into three layers: Bronze, Silver, and Gold.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Medallion Data Layering Architecture               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Bronze Layer - Raw Data                                      │  │
│  │                                                              │  │
│  │  Content: Faithful copy of raw data                          │  │
│  │  Format: JSON/CSV/Parquet (original format)                  │  │
│  │  Write Pattern: Append-only                                  │  │
│  │  Purpose: Data lineage, audit, reprocessing                  │  │
│  │  Example: s3://lake/bronze/orders/                           │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Silver Layer - Cleaned/Standardized Data                    │  │
│  │                                                              │  │
│  │  Content: Cleaned, deduplicated, standardized data           │  │
│  │  Format: Parquet (columnar storage)                          │  │
│  │  Write Pattern: Incremental/Merge writes                     │  │
│  │  Purpose: Analytics queries, feature engineering             │  │
│  │  Example: s3://lake/silver/orders/                           │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Gold Layer - Aggregated/Business Data                       │  │
│  │                                                              │  │
│  │  Content: Business-oriented aggregated data                  │  │
│  │  Format: Parquet/Delta                                       │  │
│  │  Write Pattern: Periodic refresh                             │  │
│  │  Purpose: BI reports, data products                          │  │
│  │  Example: s3://lake/gold/daily_metrics/                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# Example: Medallion Architecture Implementation
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable

spark = SparkSession.builder \
    .appName("MedallionArchitecture") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# ========== Bronze Layer ==========
def ingest_to_bronze():
    """Ingest raw data into bronze layer"""
    
    raw_orders = spark.read \
        .option("multiline", "true") \
        .json("s3://raw-data/orders/*.json")
    
    bronze_orders = raw_orders \
        .withColumn("_ingestion_time", F.current_timestamp()) \
        .withColumn("_source_file", F.input_file_name()) \
        .withColumn("_batch_id", F.lit("batch_2026_01_15"))
    
    bronze_orders.write \
        .format("delta") \
        .mode("append") \
        .partitionBy("event_date") \
        .save("s3://lake/bronze/orders")
    
    print(f"Bronze layer: {bronze_orders.count()} records ingested")

# ========== Silver Layer ==========
def process_to_silver():
    """Clean data from bronze to silver layer"""
    
    bronze_orders = spark.read.format("delta").load("s3://lake/bronze/orders")
    
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

# ========== Gold Layer ==========
def process_to_gold():
    """Aggregate data from silver to gold layer"""
    
    silver_orders = spark.read.format("delta").load("s3://lake/silver/orders")
    
    daily_metrics = silver_orders \
        .groupBy("order_date") \
        .agg(
            F.count("order_id").alias("total_orders"),
            F.countDistinct("user_id").alias("unique_users"),
            F.sum("amount").alias("total_revenue"),
            F.avg("amount").alias("avg_order_amount"),
            F.max("amount").alias("max_order_amount")
        )
    
    daily_metrics.write \
        .format("delta") \
        .mode("overwrite") \
        .save("s3://lake/gold/daily_metrics")
    
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
    
    print(f"Gold layer: daily={daily_metrics.count()}, users={user_metrics.count()}")

# Run complete Medallion pipeline
ingest_to_bronze()
process_to_silver()
process_to_gold()
```

---

## 5.4 Data Governance & Compliance 🔴

### 5.4.1 Data Governance Framework

📌 **Key Concept**: Data Governance is a set of processes and policies ensuring data assets are properly managed, used, and protected. In AI systems, data governance is especially critical as it directly impacts model reliability and compliance.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Data Governance Framework                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Organization & Policy                      │  │
│  │                                                              │  │
│  │  Data Owner │ Data Steward │ Data Consumer │ Compliance     │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    Governance Processes                       │  │
│  │                                                              │  │
│  │  Classification │ Access Control │ Quality  │ Audit │ Retain│  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    Technical Implementation                  │  │
│  │                                                              │  │
│  │  Metadata Mgmt │ Data Catalog │ Lineage │ Encryption │ Mask │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4.2 Data Classification & Tiering

📌 **Key Concept**: Data classification and tiering is the foundation of data governance, determining access controls, encryption policies, and compliance requirements.

| Data Tier | Description | Examples | Protection Measures |
|-----------|-------------|----------|-------------------|
| **Public** | Openly accessible | Product info, news | No special protection |
| **Internal** | Internal employees only | Internal reports, logs | Basic access control |
| **Confidential** | Restricted access | User PII, financial data | Encryption + access control + audit |
| **Secret** | Strictly restricted | Keys, core algorithms | Highest security level |

```python
# Example: Data classification and access control implementation
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
    """Data governance manager"""
    
    def __init__(self):
        self.assets: dict = {}
        self.access_policies: dict = {}
    
    def register_asset(self, asset: DataAsset):
        """Register data asset"""
        self.assets[asset.name] = asset
        print(f"Registered: {asset.name} ({asset.classification.value})")
    
    def set_access_policy(
        self,
        asset_name: str,
        allowed_roles: List[str],
        allowed_users: List[str]
    ):
        """Set access policy"""
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
        """Check access permissions"""
        if asset_name not in self.assets:
            return False
        
        policy = self.access_policies.get(asset_name, {})
        
        if user_role in policy.get("allowed_roles", []):
            return True
        
        if user_id in policy.get("allowed_users", []):
            return True
        
        if self.assets[asset_name].classification == DataClassification.PUBLIC:
            return True
        
        return False
    
    def get_compliance_requirements(self, asset_name: str) -> dict:
        """Get compliance requirements"""
        asset = self.assets.get(asset_name)
        if not asset:
            return {}
        
        return {
            "retention_days": asset.retention_days,
            "encryption_required": asset.classification in [
                DataClassification.CONFIDENTIAL, DataClassification.SECRET
            ],
            "audit_logging": asset.classification in [
                DataClassification.CONFIDENTIAL, DataClassification.SECRET
            ],
            "pii_masking": "PII" in asset.tags,
            "gdpr_compliant": "PII" in asset.tags,
        }

# Usage example
governance = DataGovernance()

governance.register_asset(DataAsset(
    name="user_profiles",
    classification=DataClassification.CONFIDENTIAL,
    owner="data-team",
    description="User profile data with PII",
    tags=["PII", "user-data"],
    retention_days=365
))

governance.register_asset(DataAsset(
    name="public_products",
    classification=DataClassification.PUBLIC,
    owner="product-team",
    description="Public product information",
    tags=["product"],
    retention_days=-1
))

governance.set_access_policy(
    "user_profiles",
    allowed_roles=["data-scientist", "data-analyst"],
    allowed_users=["admin@company.com"]
)

print(governance.check_access("user_profiles", "data-scientist", "user1"))  # True
print(governance.check_access("user_profiles", "intern", "user2"))  # False
print(governance.get_compliance_requirements("user_profiles"))
```

### 5.4.3 GDPR/CCPA Compliance Implementation

📌 **Key Concept**: GDPR (EU General Data Protection Regulation) and CCPA (California Consumer Privacy Act) are the world's most important data privacy regulations, imposing strict requirements on AI system data processing.

| GDPR Requirement | Technical Implementation | Delta Lake Support |
|-----------------|------------------------|-------------------|
| **Right of Access** | Data catalog + lineage tracking | Time Travel traces data origin |
| **Right to Erasure** | Data deletion mechanism | VACUUM cleans old versions |
| **Data Portability** | Standard format export | Parquet format export |
| **Data Minimization** | Column-level access control | Delta Lake Column Masking |
| **Processing Records** | Audit logs | Delta Lake operation logs |

```python
# Example: GDPR compliance implementation
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("GDPRCompliance") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .getOrCreate()

class GDPRComplianceManager:
    """GDPR compliance manager"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def data_subject_access_request(
        self,
        table_path: str,
        user_id: str
    ) -> dict:
        """Data Subject Access Request (DSAR) - return all user data"""
        
        current_data = self.spark.read.format("delta").load(table_path) \
            .filter(f"user_id = '{user_id}'")
        
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
        """Right to Erasure"""
        
        delta_table = DeltaTable.forPath(self.spark, table_path)
        delta_table.delete(f"user_id = '{user_id}'")
        delta_table.vacuum(retentionHours=168)  # Keep 7 days
        
        print(f"Erased data for user: {user_id}")
    
    def data_minimization(
        self,
        source_table: str,
        target_table: str,
        allowed_columns: List[str]
    ):
        """Data minimization - keep only necessary columns"""
        
        source_df = self.spark.read.format("delta").load(source_table)
        minimized_df = source_df.select(*allowed_columns)
        
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
        """Audit log recording"""
        
        audit_record = {
            "timestamp": self.spark.sql("SELECT current_timestamp()").collect()[0][0],
            "operation": operation,
            "user_id": user_id,
            "table_path": table_path,
            "details": str(details),
            "session_id": self.spark.sparkContext.applicationId
        }
        
        self.spark.createDataFrame([audit_record]) \
            .write \
            .format("delta") \
            .mode("append") \
            .save("s3://audit-logs/gdpr_operations")
        
        print(f"Audit log recorded: {operation}")

# Usage
gdpr_manager = GDPRComplianceManager(spark)

# DSAR
dsar = gdpr_manager.data_subject_access_request(
    "s3://lake/silver/user_profiles", "user_12345"
)
print(f"Records found: {dsar['record_count']}")

# Right to erasure
gdpr_manager.right_to_erasure("s3://lake/silver/user_profiles", "user_12345")

# Data minimization
gdpr_manager.data_minimization(
    "s3://lake/silver/user_profiles",
    "s3://lake/silver/user_profiles_minimized",
    ["user_id", "user_segment", "registration_date"]
)
```

---

## 5.5 Data Architecture Optimization for AI Workloads 🔴

### 5.5.1 Special Requirements of AI Workloads

📌 **Key Concept**: AI/ML workloads have unique data architecture requirements that differ significantly from traditional BI workloads.

| Dimension | BI Workloads | AI/ML Workloads |
|-----------|-------------|-----------------|
| **Query Pattern** | Pre-defined aggregations | Exploratory analysis, feature engineering |
| **Data Volume** | Aggregated data | Raw data (full) |
| **Data Format** | Structured | Structured + Unstructured |
| **Latency Requirement** | Seconds~Minutes | Hours~Days |
| **Compute Pattern** | SQL queries | Matrix operations, distributed training |
| **Data Freshness** | T+1 | Real-time~Hours |
| **Iteration Need** | Low | High (repeated experiments) |

### 5.5.2 ML-Specialized Data Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│              Data Architecture Optimization for AI Workloads         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Data Acquisition Layer                     │  │
│  │                                                              │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐            │  │
│  │  │  Training  │  │  Validation│  │  Test      │            │  │
│  │  │  Data      │  │  Data      │  │  Data      │            │  │
│  │  │  (history) │  │  (time     │  │  (time     │            │  │
│  │  │            │  │   split)   │  │   split)   │            │  │
│  │  └────────────┘  └────────────┘  └────────────┘            │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    Data Preparation Layer                     │  │
│  │                                                              │  │
│  │  Feature Eng.  │ Data Augment. │ Data Clean │ Label Mgmt    │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    Storage Optimization Layer                 │  │
│  │                                                              │  │
│  │  Compaction    │ Z-Order      │ Column     │ Predicate      │  │
│  │                │              │ Pruning    │ Pushdown       │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────┼───────────────────────────────────┐  │
│  │                    Training Data Service Layer                │  │
│  │                                                              │  │
│  │  TF Record   │ Petastorm    │ WebDataset │ Delta Lake       │  │
│  │  (TensorFlow)│ (Spark→DL)   │ (Web)      │ (Universal)     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.5.3 Delta Lake AI Optimization Techniques

```python
# Example: Delta Lake AI workload optimization
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("DeltaLakeAIOptimization") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .getOrCreate()

# ========== 1. Small File Compaction ==========
def compact_small_files(table_path: str, target_file_size_mb: int = 128):
    """Merge small files to improve read performance"""
    
    delta_table = DeltaTable.forPath(spark, table_path)
    delta_table.optimize().executeCompaction()
    
    print(f"Compacted {table_path}")

# ========== 2. Z-Order Optimization ==========
def zorder_optimize(table_path: str, columns: list):
    """Z-Order optimization for multi-column query performance"""
    
    delta_table = DeltaTable.forPath(spark, table_path)
    delta_table.optimize().executeZOrderBy(*columns)
    
    print(f"Z-Order optimized on columns: {columns}")

# ========== 3. Table Statistics Analysis ==========
def analyze_table_stats(table_path: str):
    """Analyze table statistics"""
    
    delta_table = DeltaTable.forPath(spark, table_path)
    
    detail = delta_table.detail()
    detail.show(truncate=False)
    
    history = delta_table.history()
    history.show(truncate=False)

# ========== 4. Optimized Training Data Preparation ==========
def prepare_training_data_optimized(
    table_path: str,
    feature_columns: list,
    label_column: str,
    output_path: str
):
    """Optimize training data preparation pipeline"""
    
    training_df = spark.read \
        .format("delta") \
        .load(table_path) \
        .select(feature_columns + [label_column])
    
    stats = training_df.summary().collect()
    print("Training data statistics:")
    for row in stats:
        print(f"  {row['summary']}: {row.asDict()}")
    
    training_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("compression", "zstd") \
        .save(output_path)
    
    optimize_table(output_path)

def optimize_table(table_path: str):
    """Full table optimization pipeline"""
    
    compact_small_files(table_path, target_file_size_mb=128)
    
    delta_table = DeltaTable.forPath(spark, table_path)
    columns = [col.name for col in delta_table.toDF().columns[:3]]
    zorder_optimize(table_path, columns)

# ========== 5. Time Travel for ML Experiments ==========
def time_travel_ml_experiment(
    table_path: str,
    experiment_date: str,
    feature_columns: list
):
    """Get training data at specific point in time using Time Travel"""
    
    training_df = spark.read \
        .format("delta") \
        .option("timestampAsOf", experiment_date) \
        .load(table_path) \
        .select(feature_columns)
    
    print(f"Loaded {training_df.count()} records for date: {experiment_date}")
    return training_df

# ========== 6. Data Versioning for ML ==========
def ml_data_versioning(table_path: str, version: int):
    """ML data version management"""
    
    versioned_df = spark.read \
        .format("delta") \
        .option("versionAsOf", version) \
        .load(table_path)
    
    version_info = {
        "version": version,
        "record_count": versioned_df.count(),
        "schema": str(versioned_df.schema),
        "timestamp": spark.sql("SELECT current_timestamp()").collect()[0][0]
    }
    
    spark.createDataFrame([version_info]) \
        .write \
        .format("delta") \
        .mode("append") \
        .save(f"{table_path}_versions")
    
    return version_info

# Run examples
table_path = "s3://lake/silver/training_data"

optimize_table(table_path)
zorder_optimize(table_path, ["user_id", "event_date", "category"])

prepare_training_data_optimized(
    table_path=table_path,
    feature_columns=["feature_1", "feature_2", "feature_3"],
    label_column="label",
    output_path="s3://lake/gold/training_data"
)
```

### 5.5.4 Data Caching & Preheating Strategies

| Strategy | Description | Use Case | Implementation |
|----------|-------------|----------|---------------|
| **Data Prefetch** | Load training data locally in advance | Scheduled training | Airflow + Spark |
| **Cache Sharing** | Multiple training tasks share cache | Multi-GPU training | Alluxio/Redis |
| **Incremental Load** | Only load changed data | Incremental training | Delta Lake CDC |
| **Partition Cache** | Cache hot partitions | High-frequency partitions | Spark Cache |

---

## 💡 Case Study: Lakehouse Architecture with Delta Lake

### Scenario

An internet company needs to build a unified data platform supporting:
- **BI Analytics**: Daily, weekly, monthly reports for operations team
- **AI Training**: Training data for recommendation and risk models
- **Real-time Monitoring**: Real-time dashboards for business metrics
- **Data Exploration**: Self-service analytics for data scientists

### Architecture Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Lakehouse Architecture with Delta Lake                    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Data Source Layer                               │  │
│  │                                                                  │  │
│  │  MySQL         Kafka         S3           External Data         │  │
│  │  (core biz)   (behavior)    (archive)    (market data)          │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────┼───────────────────────────────────────┐  │
│  │                    Ingestion Layer                                │  │
│  │                                                                  │  │
│  │  Debezium CDC   Kafka Connect   Airflow      API Adapter        │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────┼───────────────────────────────────────┐  │
│  │                    Bronze Layer (Raw Data)                        │  │
│  │                                                                  │  │
│  │  s3://lake/bronze/                                                │  │
│  │  ├── orders/          (raw order data)                           │  │
│  │  ├── user_events/     (user behavior logs)                       │  │
│  │  ├── products/        (product info)                             │  │
│  │  └── external/        (external data)                            │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────┼───────────────────────────────────────┐  │
│  │                    Silver Layer (Cleaned/Standardized)            │  │
│  │                                                                  │  │
│  │  s3://lake/silver/                                                │  │
│  │  ├── orders/          (cleaned order data)                       │  │
│  │  ├── user_profiles/   (user profiles)                            │  │
│  │  ├── product_catalog/ (product catalog)                          │  │
│  │  └── user_behavior/   (standardized behavior data)               │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│  ┌──────────────────────────┼───────────────────────────────────────┐  │
│  │                    Gold Layer (Aggregated/Business)               │  │
│  │                                                                  │  │
│  │  s3://lake/gold/                                                  │  │
│  │  ├── daily_metrics/    (daily metrics)                            │  │
│  │  ├── user_segments/    (user segments)                            │  │
│  │  ├── product_rankings/ (product rankings)                         │  │
│  │  └── training_data/    (training datasets)                        │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                          │
│              ┌──────────────┼──────────────┐                           │
│              ▼              ▼              ▼                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │
│  │  BI         │  │  AI         │  │  Real-time  │                   │
│  │  Analytics  │  │  Training   │  │  Monitoring │                   │
│  │             │  │             │  │             │                   │
│  │  Trino     │  │  Spark      │  │  Flink      │                   │
│  │  +Superset │  │  +MLflow    │  │  +Grafana   │                   │
│  └─────────────┘  └─────────────┘  └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Implementation

```python
# lakehouse_config.py - Lakehouse configuration
from dataclasses import dataclass
from typing import Dict

@dataclass
class LakehouseConfig:
    """Lakehouse architecture configuration"""
    
    bronze_path: str = "s3://lake/bronze"
    silver_path: str = "s3://lake/silver"
    gold_path: str = "s3://lake/gold"
    
    mysql_host: str = "mysql-cluster"
    kafka_bootstrap: str = "kafka:9092"
    
    spark_config: Dict[str, str] = None
    
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

# pipeline.py - Complete data pipeline
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable

class LakehousePipeline:
    """Lakehouse data pipeline"""
    
    def __init__(self, config: LakehouseConfig):
        self.config = config
        self.spark = self._create_spark_session()
    
    def _create_spark_session(self) -> SparkSession:
        builder = SparkSession.builder.appName("LakehousePipeline")
        for key, value in self.config.spark_config.items():
            builder = builder.config(key, value)
        return builder.getOrCreate()
    
    def ingest_orders_to_bronze(self):
        """Ingest order data into bronze layer"""
        
        orders_stream = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.config.kafka_bootstrap) \
            .option("subscribe", "raw_orders") \
            .option("startingOffsets", "latest") \
            .load()
        
        parsed_orders = orders_stream \
            .selectExpr("CAST(value AS STRING)") \
            .select(F.from_json(F.col("value"), order_schema).alias("data")) \
            .select("data.*") \
            .withColumn("_ingestion_time", F.current_timestamp()) \
            .withColumn("_event_date", F.to_date("created_at"))
        
        query = parsed_orders.writeStream \
            .format("delta") \
            .outputMode("append") \
            .partitionBy("_event_date") \
            .option("checkpointLocation", f"{self.config.bronze_path}/orders/_checkpoint") \
            .start(f"{self.config.bronze_path}/orders")
        
        return query
    
    def process_orders_to_silver(self):
        """Process orders from bronze to silver layer"""
        
        bronze_orders = self.spark.read \
            .format("delta") \
            .load(f"{self.config.bronze_path}/orders")
        
        silver_orders = bronze_orders \
            .filter(F.col("order_id").isNotNull()) \
            .filter(F.col("amount") > 0) \
            .dropDuplicates(["order_id"]) \
            .withColumn("order_date", F.to_date("created_at")) \
            .withColumn("order_hour", F.hour("created_at")) \
            .withColumn("amount_cNY", F.col("amount")) \
            .withColumn("amount_usd", F.col("amount") * 0.14)
        
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
        
        self._optimize_table(f"{self.config.silver_path}/orders")
    
    def generate_gold_metrics(self):
        """Generate gold layer metrics"""
        
        silver_orders = self.spark.read \
            .format("delta") \
            .load(f"{self.config.silver_path}/orders")
        
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
    
    def _optimize_table(self, table_path: str):
        try:
            delta_table = DeltaTable.forPath(self.spark, table_path)
            delta_table.optimize().executeCompaction()
            print(f"Optimized: {table_path}")
        except Exception as e:
            print(f"Optimization failed: {e}")
    
    def query_historical(self, table_path: str, timestamp: str):
        return self.spark.read \
            .format("delta") \
            .option("timestampAsOf", timestamp) \
            .load(table_path)
    
    def monitor_data_quality(self, table_path: str) -> dict:
        df = self.spark.read.format("delta").load(table_path)
        total_rows = df.count()
        
        quality_report = {
            "table": table_path,
            "total_rows": total_rows,
            "null_counts": {},
            "freshness": {}
        }
        
        for column in df.columns:
            null_count = df.filter(F.col(column).isNull()).count()
            quality_report["null_counts"][column] = {
                "count": null_count,
                "percentage": null_count / total_rows * 100 if total_rows > 0 else 0
            }
        
        if "created_at" in df.columns:
            latest_record = df.agg(F.max("created_at")).collect()[0][0]
            quality_report["freshness"]["latest_record"] = str(latest_record)
        
        return quality_report

# ========== Usage ==========
config = LakehouseConfig()
pipeline = LakehousePipeline(config)

pipeline.ingest_orders_to_bronze()
pipeline.process_orders_to_silver()
pipeline.generate_gold_metrics()

quality = pipeline.monitor_data_quality(f"{config.silver_path}/orders")
print(f"Quality report: {quality}")

historical_data = pipeline.query_historical(
    f"{config.silver_path}/orders", "2026-01-15"
)
print(f"Historical records: {historical_data.count()}")
```

---

## Chapter Summary

| Topic | Key Takeaways |
|-------|--------------|
| **Architecture Comparison** | Warehouse=structured+strong consistency, Lake=flexible+low-cost, Lakehouse=best of both |
| **Table Formats** | Delta Lake easiest, Iceberg best evolution, Hudi best for streaming |
| **Unified Analytics** | Medallion layering, batch-stream unified processing |
| **Data Governance** | Classification + access control + GDPR compliance |
| **AI Optimization** | Compaction + Z-Order + Time Travel + data versioning |

## 📝 Exercises

### Exercise 1: Lakehouse Architecture Design (🟢 Beginner)
Design a data lakehouse architecture for an e-commerce platform:
- Design Bronze/Silver/Gold data models
- Choose appropriate table format (Delta Lake/Iceberg/Hudi)
- Draw complete architecture diagram

### Exercise 2: Data Pipeline Implementation (🟡 Intermediate)
Implement a complete Medallion data pipeline using Delta Lake:
- Bronze: Ingest raw data from Kafka
- Silver: Clean, deduplicate, standardize
- Gold: Aggregate business metrics
- Add table optimization and data quality monitoring

### Exercise 3: AI Data Architecture Optimization (🔴 Advanced)
Optimize data architecture for large-scale ML training:
- Implement small file compaction strategy
- Design Z-Order optimization scheme
- Build Time Travel data version management
- Implement incremental training data service

---

> **Next Chapter Preview**: Chapter 6 will dive deep into Model Architecture Design, including model serving deployment, model version management, A/B testing frameworks, and hands-on MLOps with MLflow.
