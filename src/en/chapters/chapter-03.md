# Chapter 3: Data Pipeline Architecture

> **Learning Objectives**: After reading this chapter, you will be able to:
> 1. Understand the core components and design principles of data pipelines
> 2. Distinguish between real-time and batch processing pipeline use cases
> 3. Design end-to-end data validation and quality assurance systems
> 4. Build comprehensive data lineage tracking and metadata management solutions
> 5. Master the architectural selection of open-source tools like Apache Kafka and Airflow
> 6. Independently build production-grade data pipeline systems

---

## 3.1 Data Collection & Ingestion 🟢

### 3.1.1 Data Source Classification

When building AI systems, data is the primary concern. The starting point of any data pipeline is data collection and ingestion. Understanding data source characteristics is the prerequisite for designing efficient pipelines.

📌 **Key Concept**: Data sources can be classified into three major categories based on how they are produced:

| Category | Characteristics | Examples | Volume |
|----------|----------------|----------|--------|
| **Transactional (OLTP)** | Structured, high-frequency writes, strong consistency | MySQL, PostgreSQL, MongoDB | GB ~ TB |
| **Log-based** | Semi-structured, append-only, time-series | Nginx logs, application logs, clickstream events | TB ~ PB |
| **External** | Diverse formats, irregular updates | Third-party APIs, crawlers, public datasets | Uncertain |

```
┌─────────────────────────────────────────────────────────────────┐
│                   Data Source Landscape                          │
├─────────────┬───────────────┬─────────────────┬────────────────┤
│ Transactional│  Message      │  Object Storage │  External APIs │
│ Databases   │  Queues       │  / Files        │                │
│ MySQL       │  Kafka        │  S3/HDFS        │  REST API      │
│ PostgreSQL  │  RabbitMQ     │  Local FS       │  GraphQL       │
│ MongoDB     │  Pulsar       │  FTP/SFTP       │  Webhook       │
│ Oracle      │               │                 │                │
└──────┬──────┴───────┬───────┴────────┬────────┴───────┬────────┘
       │              │                │                │
       ▼              ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Ingestion Layer                                 │
│  CDC Tools  │  Consumer  │  File       │  API                  │
│  Debezium   │  Groups    │  Watchers   │  Adapters             │
│  Maxwell    │  Kafka     │  inotify    │  Custom Connectors    │
│  Canal      │            │  Watchdog   │                       │
│             │            │  Airflow    │                       │
└─────────────┴────────────┴─────────────┴───────────────────────┘
```

### 3.1.2 Batch vs. Streaming Ingestion

📌 **Key Concept**: The ingestion pattern determines the overall architecture of downstream pipelines.

**Batch Ingestion**:
- Periodically exports snapshots of data sources in bulk
- Typical tools: Airflow + various Operators, Sqoop, bcp
- Pros: Simple implementation, low pressure on source systems
- Cons: High latency (minutes to hours)

**Streaming Ingestion**:
- Continuously monitors data changes and pushes to target systems in real-time
- Typical tools: Kafka Connect, Debezium, Maxwell
- Pros: Low latency (millisecond to second-level), handles incremental data
- Cons: Complex implementation, must handle out-of-order data and duplicates

```python
# Example: MySQL CDC using Debezium + Kafka Connect
# debezium-config.json
{
  "name": "mysql-cdc-connector",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "database.hostname": "mysql-host",
    "database.port": "3306",
    "database.user": "debezium",
    "database.password": "${secrets:db-password}",
    "database.server.id": "184054",
    "database.include.list": "production_db",
    "table.include.list": "production_db.users,production_db.orders",
    "database.history.kafka.bootstrap.servers": "kafka-broker:9092",
    "database.history.kafka.topic": "schema-changes.production_db",
    "transforms": "route",
    "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
    "transforms.route.regex": "([^.]+)\\.([^.]+)\\.([^.]+)",
    "transforms.route.replacement": "cdc.$3"
  }
}
```

💡 **Case Study**: An e-commerce platform's data collection architecture employs both batch and streaming modes simultaneously. Core transaction data is captured in real-time via Debezium CDC with latency under 200ms. User behavior logs are written directly through the Kafka Producer SDK, achieving millisecond-level latency. Non-critical operational data (such as product category changes) is synced hourly via Airflow scheduled tasks. All three data streams converge into a unified Kafka cluster for downstream consumers to process as needed.

### 3.1.3 Schema Management

📌 **Key Concept**: Schema Registry is the "contract management system" for data pipelines — it ensures producers and consumers reach consensus on data formats.

Apache Kafka's Schema Registry provides:
- **Schema Version Management**: Every format change is recorded with a version number
- **Compatibility Checking**: Prevents breaking changes from reaching production
- **Automatic Serialization/Deserialization**: Achieved via Avro/Protobuf/JSON Schema

```
┌──────────────┐    Register Schema    ┌─────────────────┐
│   Producer   │ ──────────────────── │  Schema Registry │
│  (writes)    │                       │   (stores)       │
└──────┬───────┘                       └────────┬────────┘
       │                                        │
       │  1. Fetch latest schema                 │
       │  2. Serialize data                      │
       ▼                                        │
┌──────────────┐   Fetch schema on consume ┌────┴─────────┐
│    Kafka     │ ──────────────────────── │   Consumer    │
│    Topic     │                           │  (reads)      │
└──────────────┘                           └──────────────┘
```

⚠️ **Warning**: In production environments, **never** skip Schema Registry compatibility checks. An incompatible schema change could crash all downstream consumers simultaneously.

```python
# Schema compatibility level configuration
from confluent_kafka.schema_registry import SchemaRegistryClient

sr_client = SchemaRegistryClient({
    'url': 'http://schema-registry:8081'
})

# Set global compatibility mode
# BACKWARD: New schema can read old data
# FORWARD:  Old schema can read new data
# FULL:     Bidirectional compatibility
# NONE:     No checking (dangerous!)
sr_client.set_compatibility(
    'subjects/orders-value/versions/latest', 'BACKWARD'
)

# Register an Avro Schema
from confluent_kafka.schema_registry.avro import AvroSchema

schema_str = """
{
  "type": "record",
  "name": "Order",
  "namespace": "com.example",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "user_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "currency", "type": "string", "default": "CNY"},
    {"name": "created_at", "type": "long", "logicalType": "timestamp-millis"}
  ]
}
"""

# Register schema (automatically checks compatibility)
schema_id = sr_client.register('orders-value', AvroSchema(schema_str))
print(f"Schema registered with ID: {schema_id}")
```

---

## 3.2 Real-time vs. Batch Pipelines 🟡

### 3.2.1 Lambda Architecture

📌 **Key Concept**: Lambda Architecture, proposed by Nathan Marz, maintains both a Batch Layer and a Speed Layer to balance data completeness with real-time performance.

```
                        ┌──────────────────────────┐
                        │      Data Source          │
                        └────────────┬─────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                                  ▼
         ┌──────────────────┐             ┌──────────────────┐
         │   Batch Layer    │             │   Speed Layer    │
         │                  │             │                  │
         │  Full data store │             │  Incremental     │
         │  Offline compute │             │  Stream compute  │
         │  High latency,   │             │  Low latency,    │
         │  high accuracy   │             │  approximate     │
         └────────┬─────────┘             └────────┬─────────┘
                  │                                 │
                  │         ┌──────────────┐        │
                  │         │ Serving Layer│        │
                  └────────▶│              │◀───────┘
                            │ Merges views │
                            │ Serves query │
                            └──────┬───────┘
                                   ▼
                            ┌──────────────┐
                            │  Query       │
                            └──────────────┘
```

Pros and cons of Lambda Architecture:

| Dimension | Advantages | Disadvantages |
|-----------|-----------|---------------|
| **Data Completeness** | Batch layer guarantees eventual consistency | Must maintain two sets of logic |
| **Real-time** | Speed layer provides second-level latency | Batch layer has hour-level latency |
| **Complexity** | Clear conceptual model | High maintenance cost, two codebases |
| **Fault Tolerance** | Batch layer can be recomputed | Two layers need state synchronization |

### 3.2.2 Kappa Architecture

📌 **Key Concept**: Kappa Architecture, proposed by Jay Kreps (Kafka's creator), is based on the core idea that "everything is a stream" — all data is stored as streams through message queues, and batch processing is simply a special case of stream processing (replaying the full dataset).

```
┌──────────────────────────────────────────────────────────────┐
│                     Kappa Architecture                        │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Source    │───▶│   Kafka      │───▶│   Stream Engine   │   │
│  │          │    │  (full store) │    │  (Flink/Spark)   │   │
│  └──────────┘    └──────────────┘    └────────┬─────────┘   │
│                                               │              │
│                                          ┌────▼─────┐       │
│                                          │ Serving  │       │
│                                          │(database)│       │
│                                          └──────────┘       │
│                                                              │
│  New requirement? → Rewrite stream logic → Replay from Kafka │
│  → Validate → Switch                                          │
└──────────────────────────────────────────────────────────────┘
```

Core advantages of Kappa Architecture:
1. **Single codebase**: No "two sets of logic" problem as in Lambda Architecture
2. **Recomputation capability**: Through Kafka's data retention policy, historical data can be replayed at any time
3. **Architectural simplicity**: All data flows through a single pipeline

⚠️ **Warning**: Kappa Architecture requires very high storage capacity from the message queue. Kafka retains data for 7 days by default. If you need longer retention (e.g., 90 days), you must adjust `log.retention.ms` and ensure sufficient disk space.

### 3.2.3 Stream Processing Engine Comparison

| Feature | Apache Flink | Apache Spark Streaming | Apache Kafka Streams |
|---------|-------------|----------------------|---------------------|
| **Processing Model** | True per-record | Micro-batch | Per-record |
| **Latency** | Millisecond | Second | Millisecond |
| **State Management** | Built-in (RocksDB) | External store needed | Built-in (RocksDB) |
| **Exactly-once** | ✅ | ✅ | ✅ |
| **Window Support** | Rich (tumbling/sliding/session) | Basic | Rich |
| **Deployment** | Standalone/YARN/K8s | Standalone/YARN/K8s | Embedded in app |
| **Learning Curve** | Steep | Moderate | Gentle |
| **Best For** | Complex stream processing | Existing Spark ecosystem | Lightweight stream processing |

```python
# Example: Real-time order stream processing with Apache Flink (PyFlink)
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.expressions import col, lit

# Create streaming environment
env_settings = EnvironmentSettings.in_streaming_mode()
t_env = StreamTableEnvironment.create(environment_settings=env_settings)

# Define Kafka Source
t_env.execute_sql("""
    CREATE TABLE kafka_orders (
        order_id STRING,
        user_id STRING,
        amount DOUBLE,
        currency STRING,
        event_time TIMESTAMP(3),
        WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'orders',
        'properties.bootstrap.servers' = 'kafka-broker:9092',
        'properties.group.id' = 'flink-order-processor',
        'format' = 'json',
        'scan.startup.mode' = 'latest-offset'
    )
""")

# Define result sink (write to database)
t_env.execute_sql("""
    CREATE TABLE order_statistics (
        window_start TIMESTAMP(3),
        window_end TIMESTAMP(3),
        order_count BIGINT,
        total_amount DOUBLE,
        avg_amount DOUBLE,
        currency STRING,
        PRIMARY KEY (window_start, currency) NOT ENFORCED
    ) WITH (
        'connector' = 'jdbc',
        'url' = 'jdbc:postgresql://db-host:5432/analytics',
        'table-name' = 'order_statistics',
        'username' = 'analytics_user',
        'password' = '${DB_PASSWORD}'
    )
""")

# Execute windowed aggregation query
t_env.execute_sql("""
    INSERT INTO order_statistics
    SELECT
        window_start,
        window_end,
        COUNT(*) AS order_count,
        SUM(amount) AS total_amount,
        AVG(amount) AS avg_amount,
        currency
    FROM TABLE(
        TUMBLE(TABLE kafka_orders, DESCRIPTOR(event_time), INTERVAL '1' MINUTE)
    )
    GROUP BY window_start, window_end, currency
""")
```

---

## 3.3 Data Validation & Quality Assurance 🟡

### 3.3.1 Data Quality Dimensions

📌 **Key Concept**: Data quality is the foundation of AI system reliability. It must be controlled across six dimensions:

| Dimension | Definition | Check Method | Impact |
|-----------|-----------|-------------|--------|
| **Completeness** | Are there missing values? | Null detection, record count monitoring | Model training bias |
| **Accuracy** | Are data values correct? | Range checks, cross-validation | Incorrect predictions |
| **Consistency** | Is the same entity consistent across systems? | Cross-table/cross-system comparison | Data silos |
| **Timeliness** | Does data arrive within expected time? | Latency monitoring | Delayed decisions |
| **Uniqueness** | Are there duplicate records? | Primary/unique key checks | Counting bias |
| **Validity** | Does data conform to predefined rules? | Schema checks, format validation | Processing failures |

### 3.3.2 Great Expectations in Practice

Great Expectations is an open-source data quality validation framework that defines quality rules through "Expectations."

```python
# Example: Data quality validation with Great Expectations
import great_expectations as gx
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

# Create DataContext
context = gx.get_context()

# Define expectation suite
suite = ExpectationSuite(expectation_suite_name="order_data_quality")

# Add quality check rules
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="order_id")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeUnique(column="order_id")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="amount", min_value=0.01, max_value=1000000
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="currency", value_set=["CNY", "USD", "EUR", "GBP"]
    )
)
suite.add_expectation(
    gx.expectations.ExpectTableRowCountToBeBetween(
        min_value=1000, max_value=10000000
    )
)

# Create checkpoint
checkpoint_name = "order_data_checkpoint"
context.add_or_update_expectation_suite(expectation_suite=suite)

# Run validation
checkpoint_result = context.run_checkpoint(
    checkpoint_name=checkpoint_name,
    batch_request={
        "datasource_name": "production_orders",
        "data_asset_name": "orders",
        "options": {"path": "s3://data-lake/orders/dt=2026-01-15/"}
    }
)

# Output results
if checkpoint_result.success:
    print("✅ Data quality check passed")
else:
    print("❌ Data quality check failed")
    for result in checkpoint_result.run_results.values():
        for validation_result in result["validation_result"]["results"]:
            if not validation_result["success"]:
                print(f"  Failed: {validation_result['expectation_config']['expectation_type']}")
                print(f"  Details: {validation_result['result']}")
```

### 3.3.3 Data Quality Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Data Quality Assurance System                       │
│                                                                     │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────────┐  │
│  │  Source  │──▶│ Ingestion│──▶│ Quality  │──▶│ Storage/Delivery│  │
│  │         │   │          │   │ Check    │   │                 │  │
│  │  Raw    │   │ Clean    │   │ Rules    │   │ Quality Reports │  │
│  │  Data   │   │ Format   │   │ Anomaly  │   │ Alert Notifs    │  │
│  └─────────┘   └──────────┘   └──────────┘   └─────────────────┘  │
│                      │                │                              │
│                      ▼                ▼                              │
│               ┌──────────┐    ┌──────────────┐                     │
│               │  Logs    │    │  Quality     │                     │
│               │  Audit   │    │  Dashboard   │                     │
│               └──────────┘    │  Grafana     │                     │
│                               └──────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3.4 Anomaly Detection and Auto-Repair

Beyond rule-based checking, data pipelines need automated anomaly detection and repair mechanisms:

```python
# Example: Data anomaly detection and auto-repair pipeline
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable

@dataclass
class QualityRule:
    name: str
    check: Callable[[pd.DataFrame], bool]
    severity: str  # "critical", "warning", "info"
    auto_fix: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

class DataQualityPipeline:
    def __init__(self):
        self.rules: List[QualityRule] = []
        self.fix_history: List[dict] = []

    def add_rule(self, rule: QualityRule):
        self.rules.append(rule)

    def validate_and_fix(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for rule in self.rules:
            passed = rule.check(df)
            if not passed:
                if rule.auto_fix:
                    df = rule.auto_fix(df)
                    results.append({
                        "rule": rule.name,
                        "status": "auto_fixed",
                        "severity": rule.severity
                    })
                else:
                    results.append({
                        "rule": rule.name,
                        "status": "failed",
                        "severity": rule.severity
                    })
                    if rule.severity == "critical":
                        raise ValueError(
                            f"Critical quality check failed: {rule.name}"
                        )
            else:
                results.append({
                    "rule": rule.name,
                    "status": "passed",
                    "severity": rule.severity
                })
        return df

# Usage example
pipeline = DataQualityPipeline()

# Rule 1: Null repair
pipeline.add_rule(QualityRule(
    name="null_check_user_id",
    check=lambda df: df["user_id"].notna().all(),
    severity="critical",
    auto_fix=lambda df: df.dropna(subset=["user_id"])
))

# Rule 2: Amount range check and repair
pipeline.add_rule(QualityRule(
    name="amount_range_check",
    check=lambda df: ((df["amount"] >= 0) & (df["amount"] <= 1_000_000)).all(),
    severity="critical",
    auto_fix=lambda df: df[
        (df["amount"] >= 0) & (df["amount"] <= 1_000_000)
    ]
))

# Rule 3: Duplicate deduplication
pipeline.add_rule(QualityRule(
    name="duplicate_check",
    check=lambda df: not df.duplicated(subset=["order_id"]).any(),
    severity="warning",
    auto_fix=lambda df: df.drop_duplicates(subset=["order_id"], keep="last")
))

# Rule 4: Timestamp sanity
pipeline.add_rule(QualityRule(
    name="timestamp_sanity",
    check=lambda df: (df["created_at"] <= pd.Timestamp.now()).all(),
    severity="warning",
    auto_fix=lambda df: df[df["created_at"] <= pd.Timestamp.now()]
))

# Run quality pipeline
raw_data = pd.read_parquet("s3://data-lake/orders/dt=2026-01-15/")
clean_data = pipeline.validate_and_fix(raw_data)
print(f"Before: {len(raw_data)} rows, After: {len(clean_data)} rows")
```

---

## 3.4 Data Lineage & Metadata Management 🔴

### 3.4.1 Data Lineage: Concept and Value

📌 **Key Concept**: Data Lineage describes the complete flow path of data from source to final output. It answers: "Where did this data come from, what processing did it go through, and where does it end up?"

Core value of data lineage:
1. **Impact Analysis**: When upstream data changes, quickly assess downstream impact scope
2. **Root Cause Analysis**: When data quality issues arise, quickly locate the origin
3. **Compliance Auditing**: Meet GDPR, CCPA, and other regulatory requirements for data flow tracking
4. **Optimization Guidance**: Identify redundant data processing steps and optimize pipeline performance

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Data Lineage Diagram                            │
│                                                                     │
│  MySQL ──CDC──▶ Kafka ──Flink──▶ Redis ──▶ Model Serving            │
│    │                              │                                  │
│    │                              ├──▶ PostgreSQL ──▶ Reports       │
│    │                              │                                  │
│    └──Spark──▶ S3 ──▶ Delta Lake ──▶ Offline Training ──▶ Model Reg│
│                                                                     │
│  When MySQL table schema changes:                                    │
│  1. CDC automatically captures change event                          │
│  2. Lineage system automatically marks affected downstream nodes     │
│  3. Triggers alerts to notify responsible parties                    │
│  4. Generates impact assessment report                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4.2 Apache Atlas: Metadata Governance Platform

Apache Atlas is a metadata governance framework in the Hadoop ecosystem, providing:
- **Type System**: Define metadata models
- **Metadata Storage**: Persist metadata
- **Lineage Capture**: Automatically track data flow
- **Classification & Tags**: Metadata classification management
- **Security & Governance**: Role-based access control

```python
# Example: Managing metadata with Apache Atlas Python Client
from apache_atlas.client import AtlasClient

# Create client connection
client = AtlasClient(
    host='atlas-host',
    port=21000,
    username='admin',
    password='admin'
)

# Create entity representing a dataset
entity = {
    "typeName": "hive_table",
    "attributes": {
        "qualifiedName": "production_db@orders",
        "name": "orders",
        "description": "Orders master table",
        "owner": "data-team",
        "createTime": 1705276800000,
        "columns": [
            {"typeName": "column", "attributes": {"name": "order_id", "type": "string"}},
            {"typeName": "column", "attributes": {"name": "user_id", "type": "string"}},
            {"typeName": "column", "attributes": {"name": "amount", "type": "double"}}
        ]
    },
    "classifications": [
        {"typeName": "PII", "attributes": {"user_id": "PII"}},
        {"typeName": "Gold", "attributes": {"tier": "gold"}}
    ]
}

# Push metadata to Atlas
response = client.entity.create(entity)
print(f"Entity created, GUID: {response['guid']}")

# Query data lineage
lineage = client.lineage.get_lineage(
    entity_guid=response['guid'],
    direction="BOTH",
    depth=5
)
print(f"Lineage relationships: {len(lineage)}")
```

### 3.4.3 Metadata Management Architecture Design

📌 **Key Concept**: Modern metadata management adopts a "Metadata Data Lake" architecture, centrally storing all metadata in a unified platform.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Metadata Management Architecture                    │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ Technical    │  │ Business     │  │ Operational  │             │
│  │ Metadata     │  │ Metadata     │  │ Metadata     │             │
│  │              │  │              │  │              │             │
│  │ Schema info  │  │ Business     │  │ Schedule     │             │
│  │ Data types   │  │ terminology  │  │ logs         │             │
│  │ Partition    │  │ Data dict    │  │ Task status  │             │
│  │ Lineage      │  │ Quality      │  │ Resource     │             │
│  │              │  │ rules        │  │ usage        │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                 │                 │                       │
│         └─────────────────┼─────────────────┘                       │
│                           ▼                                         │
│              ┌──────────────────────┐                               │
│              │  Metadata Store      │                               │
│              │ (Elasticsearch +     │                               │
│              │  PostgreSQL)         │                               │
│              └──────────┬───────────┘                               │
│                         │                                           │
│         ┌───────────────┼───────────────┐                          │
│         ▼               ▼               ▼                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                   │
│  │  Lineage   │  │  Search    │  │  Governance│                   │
│  │  Graph     │  │  Service   │  │  Dashboard │                   │
│  │  Neo4j     │  │  ES API    │  │  Grafana   │                   │
│  └────────────┘  └────────────┘  └────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4.4 Data Catalog Implementation

The Data Catalog is the user-facing layer of metadata management, enabling users to:
- **Search**: Find datasets by keywords, tags, owners, and other dimensions
- **Discover**: Browse hierarchical structures and relationships of data assets
- **Understand**: View Schema, lineage, and quality reports
- **Collaborate**: Comment, tag, and annotate data usage scenarios

| Open-Source Data Catalog | Key Features | Best For |
|-------------------------|-------------|----------|
| **Apache Atlas** | Good Hadoop ecosystem integration, comprehensive | Big data platforms |
| **DataHub** | LinkedIn-originated, modern architecture | Cloud-native environments |
| **Amundsen** | Lyft-originated, great search experience | Data discovery |
| **OpenMetadata** | Next-generation, API-first | Hybrid cloud environments |

---

## 3.5 Open Source Tool Selection 🔴

### 3.5.1 Tool Selection Matrix

📌 **Key Concept**: Data pipeline tool selection must consider multiple factors: functionality, performance, community activity, and operational complexity.

| Tool | Category | Core Strength | Weakness | Best For |
|------|----------|--------------|----------|----------|
| **Apache Kafka** | Message Queue/Stream | High throughput, persistence, rich ecosystem | Complex ops, resource-heavy | High-throughput streaming |
| **Apache Flink** | Stream Engine | True per-record processing, strong state management | Steep learning curve | Complex stream processing |
| **Apache Airflow** | Task Scheduling | Python-native, extensible | Slow Web UI, struggles with complex DAGs | Batch orchestration |
| **Apache NiFi** | Data Integration | Visual config, real-time monitoring | Not suited for complex logic | Data routing |
| **dbt** | Data Transformation | SQL-first, version controlled | Not for real-time | Analytical transformation |
| **Great Expectations** | Data Quality | Flexible expectation system | Extra integration work | Data validation |
| **Apache Atlas** | Metadata Management | Good Hadoop integration | Complex deployment | Big data governance |

### 3.5.2 Apache Airflow Deep Dive

Apache Airflow is the most popular workflow orchestration tool today, built on the core design philosophy of **"Workflows as Code."**

```python
# Example: Orchestrating a data pipeline with Apache Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.kafka.operators.produce import ProduceToTopicOperator
from airflow.providers.amazon.aws.transfers.s3_to_redshift import S3ToRedshiftOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['data-alerts@company.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

def validate_data_quality(**context):
    """Data quality validation task"""
    import great_expectations as gx
    ti = context['ti']
    data_path = ti.xcom_pull(task_ids='extract_data')
    
    context_gx = gx.get_context()
    checkpoint_result = context_gx.run_checkpoint(
        batch_request={"path": data_path},
        checkpoint_name="production_orders_checkpoint"
    )
    
    if not checkpoint_result.success:
        raise ValueError("Data quality validation failed!")
    return True

def transform_features(**context):
    """Feature engineering transformation"""
    import pandas as pd
    import numpy as np
    ti = context['ti']
    raw_data = pd.read_parquet(ti.xcom_pull(task_ids='extract_data'))
    
    # Feature engineering logic
    features = raw_data.assign(
        order_hour=raw_data['created_at'].dt.hour,
        order_dow=raw_data['created_at'].dt.dayofweek,
        amount_log=np.log1p(raw_data['amount']),
        user_order_count=raw_data.groupby('user_id')['order_id'].transform('count')
    )
    
    output_path = f"s3://feature-store/processed/dt={context['ds']}/"
    features.to_parquet(output_path)
    return output_path

# Define DAG
with DAG(
    dag_id='order_data_pipeline',
    default_args=default_args,
    description='End-to-end order data pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2am
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['production', 'orders', 'feature-engineering'],
) as dag:

    # Task 1: Data extraction
    extract_task = ProduceToTopicOperator(
        task_id='extract_from_source',
        kafka_config_id='production_kafka',
        topic='raw_orders',
        producer_config={'linger.ms': '50', 'batch.size': '1024'},
    )

    # Task 2: Data quality validation
    quality_task = PythonOperator(
        task_id='validate_quality',
        python_callable=validate_data_quality,
    )

    # Task 3: Feature engineering
    feature_task = PythonOperator(
        task_id='transform_features',
        python_callable=transform_features,
    )

    # Task 4: Load to data warehouse
    load_task = S3ToRedshiftOperator(
        task_id='load_to_warehouse',
        schema='analytics',
        table='order_features',
        s3_bucket='feature-store',
        s3_key='processed/',
        copy_options=['FORMAT AS PARQUET'],
        aws_conn_id='aws_redshift',
    )

    # Define task dependencies
    extract_task >> quality_task >> feature_task >> load_task
```

### 3.5.3 Recommended Tool Combinations

Based on team size and technology stack, here are recommended tool combinations:

**Small Teams (<5 data engineers)**:
```
Ingestion: Debezium CDC + Kafka
Orchestration: Apache Airflow (Managed)
Transformation: dbt
Data Quality: Great Expectations
Metadata: DataHub (Managed)
```

**Medium Teams (5-20 data engineers)**:
```
Ingestion: Debezium + Kafka Connect
Stream Processing: Apache Flink
Orchestration: Apache Airflow
Transformation: dbt + Spark
Data Quality: Great Expectations + Custom Monitoring
Metadata: Apache Atlas + Atlas SDK
```

**Large Teams (>20 data engineers)**:
```
Ingestion: Debezium + Kafka Connect + Custom Connectors
Stream Processing: Apache Flink + Custom Operators
Batch Processing: Apache Spark
Orchestration: Apache Airflow + Custom Scheduler
Transformation: dbt + Spark + Flink
Data Quality: Great Expectations + Custom Quality Platform
Metadata: Apache Atlas + Custom Data Catalog
Data Lake: Delta Lake / Apache Iceberg
```

---

## 💡 Case Study: End-to-End Data Pipeline with Apache Kafka + Airflow

### Scenario

A fintech company needs to build a data pipeline system that combines real-time risk control and offline analytics:
- **Data Sources**: MySQL (core transaction system), MongoDB (user behavior logs), External API (market data)
- **Processing Requirements**: Real-time risk decisions (<100ms), near-real-time reports (<5min), offline analytics (T+1)
- **Volume**: 50 million daily transaction records, 1 billion daily behavior events

### Architecture Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  End-to-End Data Pipeline Architecture                   │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │
│  │  MySQL   │  │ MongoDB  │  │ External │                              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                              │
│       │              │              │                                    │
│       ▼              ▼              ▼                                    │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                              │
│  │Debezium │   │ Filebeat│   │ Airflow │                              │
│  │  CDC    │   │+Logstash│   │  Pull   │                              │
│  └────┬────┘   └────┬────┘   └────┬────┘                              │
│       │              │              │                                    │
│       └──────────────┼──────────────┘                                   │
│                      ▼                                                  │
│           ┌──────────────────────┐                                      │
│           │   Apache Kafka       │                                      │
│           │   (Message Bus)      │                                      │
│           │                      │                                      │
│           │  Topics:             │                                      │
│           │  - raw.transactions  │                                      │
│           │  - raw.user_events   │                                      │
│           │  - raw.market_data   │                                      │
│           │  - enriched.events   │                                      │
│           │  - alerts.risk       │                                      │
│           └──────┬───────────────┘                                      │
│                  │                                                      │
│       ┌──────────┼──────────┐                                          │
│       ▼          ▼          ▼                                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                                  │
│  │  Flink  │ │  Flink  │ │  Kafka  │                                  │
│  │  Risk   │ │ Realtime│ │Consumer │                                  │
│  │ Control │ │ Agg     │ │         │                                  │
│  └────┬────┘ └────┬────┘ └────┬────┘                                  │
│       │          │          │                                          │
│       ▼          ▼          ▼                                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                                  │
│  │  Redis  │ │ClickHouse│ │   S3    │                                  │
│  │ (Risk)  │ │ (Reports)│ │ (Archive│                                  │
│  └─────────┘ └─────────┘ └─────────┘                                  │
│                  │                                                      │
│                  ▼                                                      │
│           ┌──────────────┐                                             │
│           │  Delta Lake   │                                             │
│           │ (Data Lake)   │                                             │
│           └──────┬───────┘                                             │
│                  │                                                      │
│       ┌──────────┼──────────┐                                          │
│       ▼          ▼          ▼                                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                                  │
│  │  Spark  │ │  dbt    │ │ Airflow │                                  │
│  │ Offline │ │Reports  │ │Orchest. │                                  │
│  │Training │ │         │ │         │                                  │
│  └─────────┘ └─────────┘ └─────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Implementation

```python
# Kafka Topic Configuration
# Create topic command:
# kafka-topics.sh --create --bootstrap-server kafka:9092 \
#   --topic raw.transactions \
#   --partitions 12 \
#   --replication-factor 3 \
#   --config retention.ms=604800000 \
#   --config cleanup.policy=delete \
#   --config compression.type=lz4

# Flink Real-time Risk Control Operator
from pyflink.table import EnvironmentSettings, StreamTableEnvironment

env_settings = EnvironmentSettings.in_streaming_mode()
t_env = StreamTableEnvironment.create(environment_settings=env_settings)

# Register Kafka Source
t_env.execute_sql("""
    CREATE TABLE transactions (
        transaction_id STRING,
        user_id STRING,
        amount DECIMAL(18,2),
        merchant_id STRING,
        category STRING,
        device_id STRING,
        ip_address STRING,
        event_time TIMESTAMP(3),
        proc_time AS PROCTIME(),
        WATERMARK FOR event_time AS event_time - INTERVAL '30' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'raw.transactions',
        'properties.bootstrap.servers' = 'kafka:9092',
        'properties.group.id' = 'flink-risk-engine',
        'format' = 'json',
        'scan.startup.mode' = 'latest-offset'
    )
""")

# Register Redis Sink (risk results)
t_env.execute_sql("""
    CREATE TABLE risk_scores (
        user_id STRING,
        risk_score DOUBLE,
        risk_level STRING,
        risk_factors STRING,
        update_time TIMESTAMP(3),
        PRIMARY KEY (user_id) NOT ENFORCED
    ) WITH (
        'connector' = 'redis',
        'host' = 'redis-cluster',
        'port' = '6379',
        'database' = '0',
        'command' = 'SET'
    )
""")

# Risk scoring logic
t_env.execute_sql("""
    INSERT INTO risk_scores
    SELECT
        user_id,
        CASE
            WHEN amount > 50000 THEN 0.9
            WHEN amount > 10000 THEN 0.6
            WHEN amount > 5000 THEN 0.3
            ELSE 0.1
        END AS risk_score,
        CASE
            WHEN amount > 50000 THEN 'HIGH'
            WHEN amount > 10000 THEN 'MEDIUM'
            ELSE 'LOW'
        END AS risk_level,
        TO_JSON(ARRAY[
            STRUCT('factor' := 'amount', 'value' := CAST(amount AS STRING))
        ]) AS risk_factors,
        event_time AS update_time
    FROM transactions
    WHERE event_time > CURRENT_TIMESTAMP - INTERVAL '1' HOUR
""")
```

### Performance Optimization Key Points

📌 **Key Concept**: Production-grade data pipelines must balance throughput, latency, and cost.

1. **Kafka Partition Strategy**: Hash-partition by user ID to ensure ordered processing of events per user
2. **Flink State Backend**: Use RocksDB state backend for incremental Checkpoint support
3. **Airflow Parallelism**: Properly set `max_active_tasks_per_dag` to avoid resource contention
4. **Data Compression**: Use LZ4 compression in Kafka, saving 60%+ network bandwidth
5. **Batch Writes**: Use Redis Pipeline for batch writes to reduce network round trips

### Operations & Monitoring

```
Monitoring Metrics System:
├── Pipeline Health
│   ├── Kafka Consumer Lag
│   ├── Airflow Task Duration
│   ├── Data Freshness
│   └── Error Rate
├── Data Quality
│   ├── Null Rate
│   ├── Duplicate Rate
│   ├── Schema Violation
│   └── Distribution Drift
├── Resource Usage
│   ├── Kafka Broker Disk Usage
│   ├── Flink TaskManager Memory
│   ├── Airflow Worker CPU
│   └── Data Lake Storage Cost
└── Business Metrics
    ├── Risk Control Block Rate
    ├── Real-time Report Latency
    └── Model Training Data Ready Time
```

---

## Chapter Summary

| Topic | Key Takeaways |
|-------|--------------|
| **Data Collection** | Distinguish batch from streaming ingestion; Schema management is foundational |
| **Pipeline Architecture** | Lambda vs. Kappa; choose based on team size and requirements |
| **Data Quality** | Six-dimension quality control; automated detection and repair |
| **Metadata Management** | Data lineage tracking; data catalog implementation |
| **Tool Selection** | Match tool combinations to team size and tech stack |

## 📝 Exercises

### Exercise 1: Pipeline Design (🟢 Beginner)
Design a data pipeline architecture for an e-commerce platform requiring:
- Real-time order event processing (Kafka)
- Hourly aggregation statistics (Airflow)
- Data quality validation (Great Expectations)
- Draw the architecture diagram and write key code

### Exercise 2: CDC Implementation (🟡 Intermediate)
Implement a MySQL-to-Kafka CDC pipeline using Debezium:
- Configure Debezium Connector
- Handle Schema changes
- Implement data validation
- Handle Dead Letter Queue

### Exercise 3: End-to-End Pipeline (🔴 Advanced)
Build a complete Lambda Architecture pipeline:
- Real-time layer: Flink stream processing + Redis
- Batch layer: Spark + Delta Lake
- Serving layer: Unified query interface
- Monitoring layer: Grafana dashboards

---

> **Next Chapter Preview**: Chapter 4 will dive deep into Feature Engineering Architecture, including Feature Store design principles, online/offline feature serving architectures, and hands-on implementation with Feast.
