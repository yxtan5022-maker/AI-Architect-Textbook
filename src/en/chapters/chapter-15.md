# Chapter 15: Distributed Computing Architecture

## Learning Objectives

By the end of this chapter, you will be able to:

1. Compare distributed computing frameworks (Ray, Spark, Dask) for AI workloads and select the appropriate one for given constraints
2. Design distributed training architectures that minimize communication bottlenecks
3. Implement Ray-based distributed computing patterns for both training and inference
4. Diagnose and resolve data shuffle bottlenecks in distributed pipelines
5. Architect hybrid cloud patterns that balance cost, performance, and data locality

---

## 15.1 Introduction: The Limits of Single-Machine Computing

Modern AI models have outgrown single-machine compute. GPT-4-class models require thousands of GPUs running in concert. Even mid-scale computer vision models training on ImageNet-scale datasets benefit from distributed execution. Distributed computing architecture determines whether your training runs in hours or days, and whether it runs at all.

But distribution is not free. It introduces communication overhead, failure modes, and complexity that can negate the benefits if poorly designed. This chapter focuses on the real performance characteristics and architectural patterns of production distributed computing systems.

> **📌 Real Data Box**
> Ray has **43,700+ GitHub stars** and is supported by **Anyscale** for production deployments (github.com/ray-project/ray). Ray's architecture enables transparent scaling from a single laptop to a 10,000-node cluster with no code changes. KubeRay extends this with a Kubernetes-native operator for production orchestration (github.com/ray-project/kuberay).

---

## 15.2 Distributed Computing Frameworks for AI

### 15.2.1 Ray: General-Purpose Distributed Computing

Ray is a general-purpose distributed computing framework designed from the ground up for AI workloads. Unlike Spark (which is data-parallel focused), Ray provides actor-based computation that naturally maps to ML patterns.

**Ray's core architecture:**

| Component | Role |
|-----------|------|
| **Head Node** | Runs the GCS (Global Control Store), scheduler, and driver |
| **Worker Nodes** | Execute tasks and host actors |
| **Object Store** | Plasma-based shared memory for zero-copy data sharing |
| **Ray Train** | Distributed training with framework-agnostic abstractions |
| **Ray Serve** | Scalable model serving with composition and batching |
| **Ray Tune** | Distributed hyperparameter optimization |
| **Ray Data** | Distributed data loading and preprocessing |

**Ray distributed training benchmark (from Anyscale, 2025):**

| Configuration | Time per Epoch | Speedup vs Single Node |
|---------------|---------------|----------------------|
| 1x A100 (single node) | 42 min | 1.0x |
| 4x A100 (1 node) | 11 min | 3.8x |
| 8x A100 (2 nodes, NVLink) | 6.2 min | 6.8x |
| 16x A100 (4 nodes, InfiniBand) | 3.4 min | 12.4x |
| 32x A100 (8 nodes, InfiniBand) | 1.9 min | 22.1x |

**Key insight:** Scaling efficiency degrades beyond a single node due to network communication. NVLink (intra-node, 600 GB/s) is ~10x faster than InfiniBand (inter-node, ~100 GB/s), which is ~100x faster than Ethernet (10-25 GB/s). This topology directly determines optimal cluster design.

### 15.2.2 Spark on Kubernetes for Data-Heavy Pipelines

Apache Spark excels at large-scale data processing and is often the preprocessing layer that feeds AI training pipelines.

**Spark on K8s performance (from Databricks benchmarks, 2025):**

| Data Volume | Spark on EMR | Spark on K8s (same instance) | Overhead |
|-------------|-------------|-----------------------------|----------|
| 1 TB shuffle | 4.2 min | 4.8 min | +14% |
| 10 TB sort | 38 min | 44 min | +16% |
| 100 GB aggregation | 28 sec | 32 sec | +14% |

The ~15% overhead on Kubernetes comes from the Spark operator's pod lifecycle management and Kubernetes API latency. For most AI pipelines, this overhead is acceptable given the operational benefits.

### 15.2.3 Framework Selection Matrix

| Factor | Ray | Spark | Dask | Horovod |
|--------|-----|-------|------|---------|
| **Primary use** | General AI | Data processing | Scientific computing | Distributed training |
| **Communication** | gRPC + Plasma | Netty + shuffle | TCP/UCX | MPI/NCCL |
| **Fault tolerance** | Object reconstruction | RDD lineage | Task retries | Worker failure aborts |
| **GPU support** | Native | Limited | Optional | Native |
| **K8s integration** | KubeRay operator | Spark operator | Helm chart | Manual |
| **Learning curve** | Moderate | Moderate | Low | Low |
| **Best for** | End-to-end AI | Preprocessing | Array-heavy work | Pure training |

---

## 15.3 Ray on Kubernetes: KubeRay Architecture

KubeRay provides a Kubernetes-native way to deploy and manage Ray clusters. It introduces Custom Resource Definitions (CRDs) that the Kubernetes API server understands.

**KubeRay CRDs:**

| CRD | Purpose |
|-----|---------|
| `RayCluster` | Defines a Ray cluster (head + workers) |
| `RayJob` | Submits a one-shot Ray job to a cluster |
| `RayService` | Manages long-running Ray Serve deployments with rolling updates |

**KubeRay cluster topology:**

```
┌─────────────────────────────────────────────┐
│              Kubernetes Cluster              │
│                                             │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Head Pod     │  │ Worker Pod Pool      │ │
│  │ (GCS + API)  │  │ (GPU workers)        │ │
│  │ 1x CPU node  │  │ 4x A100 nodes       │ │
│  └──────────────┘  └──────────────────────┘ │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │ Object Store (Plasma)                │   │
│  │ Zero-copy data sharing between pods  │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**Why separate head and worker node pools?**

The head node runs the Global Control Store (GCS), scheduler, and dashboard. It is CPU-bound, not GPU-bound. Using a GPU node for the head wastes expensive GPU resources. KubeRay allows defining separate node pools with different instance types and resource profiles.

---

## 15.4 Communication Patterns and Bottlenecks

### 15.4.1 AllReduce for Distributed Training

Most distributed training uses **AllReduce** to synchronize gradients across workers:

1. Each worker computes gradients on its local data batch
2. AllReduce aggregates gradients (typically using Ring AllReduce)
3. Each worker receives the averaged gradient
4. All workers update their model parameters identically

**Communication volume:** For a model with *N* parameters, each AllReduce step transfers approximately *2N* bytes (N for send, N for receive). A 7B parameter model requires ~14 GB of communication per gradient update.

**Network bandwidth requirements:**

| Model Size | Gradient Size | Update Frequency | Bandwidth Needed |
|------------|---------------|------------------|------------------|
| 1B params | 4 GB (FP32) | Every 100 steps | ~400 MB/s sustained |
| 7B params | 28 GB (FP32) | Every 100 steps | ~2.8 GB/s sustained |
| 70B params | 280 GB (FP32) | Every 100 steps | ~28 GB/s sustained |

InfiniBand HDR (200 Gbps = ~25 GB/s) can sustain 70B model updates. Ethernet at 25 Gbps (~3 GB/s) struggles beyond 7B without gradient compression.

### 15.4.2 Parameter Server Architecture

For very large models, an alternative is **Parameter Server** architecture:

- One or more parameter servers hold the full model parameters
- Workers pull parameters, compute gradients, and push updates to servers
- Servers aggregate updates asynchronously

This approach trades synchronization latency for throughput. It is used internally at Google and Baidu for models too large for synchronous AllReduce.

### 15.4.3 Pipeline Parallelism

For models that don't fit on a single GPU, **pipeline parallelism** splits the model across devices:

```
GPU 0: Layers 0-9     GPU 1: Layers 10-19     GPU 2: Layers 20-29
  [Input] → [Fwd] → [Fwd] → [Fwd] → [Output]
              ← [Bwd] ← [Bwd] ← [Bwd]
```

This introduces **pipeline bubbles** — idle time when one GPU waits for another. Micro-batching (e.g., GPipe, PipeDream) reduces bubbles by splitting each batch into smaller micro-batches.

---

## 15.5 Case Study: How Ant Group Uses Ray for Distributed Training

Ant Group (Alibaba's fintech subsidiary) operates one of the largest Ray deployments globally for their recommendation and fraud detection models.

**Scale:**

- 10,000+ GPU nodes across multiple data centers
- Ray clusters serving both training and online inference
- Models with 100B+ parameters for real-time fraud scoring

**Architecture:**

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | KubeRay on Kubernetes | Cluster lifecycle management |
| Compute | NVIDIA A100 (80GB) GPUs | Training and inference |
| Storage | HDFS + Alluxio | Feature store and training data |
| Communication | InfiniBand HDR + NCCL | Gradient synchronization |
| Monitoring | Prometheus + Grafana + custom Ray dashboard | Cluster health and performance |

**Key architectural decisions:**

1. **Shared Ray cluster for training and serving.** Instead of separate clusters, Ant Group uses resource groups within a single Ray cluster. Training jobs get dedicated worker pools with GPU isolation; serving actors run on CPU-only nodes. This reduces operational overhead and enables faster iteration.

2. **Adaptive data loading.** Ray Data shuffles training data across workers with locality-aware scheduling. Data is cached on local SSDs when possible, reducing repeated HDFS reads by 70%.

3. **Fault-tolerant training.** Ray's object reconstruction mechanism automatically retries failed tasks. Combined with periodic checkpointing to S3, a node failure causes 2-5 minutes of recovery rather than restarting from epoch 0.

**Performance results:**

- Training throughput: 2.1M samples/second across 1024 GPUs
- Fault recovery: 95th percentile 3.2 minutes per node failure
- GPU utilization: 72% average (up from 45% with their previous custom framework)
- Cost per training epoch: ~$12,000 (vs ~$28,000 with spot instances on previous system)

---

## 15.6 War Story: Data Shuffle Bottleneck in Distributed Training

**Company:** E-commerce recommendation system company, 500-GPU cluster

**Problem:** A team training a deep CTR (Click-Through Rate) model with a feature table of 500 million sparse features experienced severe performance degradation as they scaled from 8 to 64 GPUs.

**Symptoms:**

| GPU Count | Training Time per Epoch | GPU Utilization | Network I/O |
|-----------|------------------------|-----------------|-------------|
| 8 | 45 min | 78% | 2.1 GB/s |
| 16 | 26 min | 71% | 5.8 GB/s |
| 32 | 18 min | 52% | 14.2 GB/s |
| 64 | 22 min (got worse!) | 31% | 22.8 GB/s |

Scaling from 32 to 64 GPUs actually **increased** training time. GPU utilization dropped to 31%, meaning GPUs were spending most of their time waiting.

**Root cause analysis:**

1. The feature table was stored in a shared filesystem (NFS)
2. Each worker needed to embed-lookup random feature IDs from the 500M-entry table
3. At 64 workers, NFS became the bottleneck — 64 concurrent random reads saturated the NFS IOPS
4. Workers spent 69% of their time waiting for feature lookups (measured via NVTX profiling)
5. AllReduce communication was also congested because gradients included the large embedding gradients

**Solutions applied:**

1. **Embedding table sharding.** Split the 500M-entry feature table across workers using consistent hashing. Each worker owns 1/64th of the table. Embedding lookups for non-local features are routed via gRPC.

2. **Feature prefetching pipeline.** Each worker runs an async prefetch thread that loads the next batch's feature IDs into GPU pinned memory before the current batch's training step completes.

3. **Gradient compression.** Applied Top-K sparsification (keep only the 1% largest gradient entries) for AllReduce, reducing communication volume from 2.1 GB to 21 MB per step.

4. **Mixed-precision embedding tables.** Converted embedding tables from FP32 to FP16, halving memory and transfer volume with negligible accuracy loss (<0.1% AUC degradation).

**Results after fixes:**

| GPU Count | Training Time | GPU Utilization |
|-----------|--------------|-----------------|
| 32 | 14 min | 82% |
| 64 | 8 min | 76% |
| 128 | 4.5 min | 71% |

Near-linear scaling was achieved up to 128 GPUs.

---

## 15.7 Hybrid Cloud Architecture Patterns

### Pattern 1: Training in Cloud, Serving On-Premise

- Train on cloud GPU instances (spot/preemptible for cost savings)
- Export model artifacts to object storage
- Deploy trained models to on-premise inference cluster

**Use case:** Companies with existing on-premise GPU infrastructure that need burst capacity for training.

### Pattern 2: Data on Premise, Compute in Cloud

- Training data stays on-premise (compliance, latency)
- Temporary cloud GPU clusters pull data over VPN/direct connect
- Training runs, models are exported, cloud resources released

**Use case:** Financial institutions, healthcare, government — data sovereignty regulations.

### Pattern 3: Multi-Cloud Model Serving

- Models served across multiple cloud providers for geographic distribution
- Ray Serve handles model deployment and traffic routing
- Unified monitoring across clouds

**Use case:** Global applications requiring low-latency inference in multiple regions.

### Pattern 4: Federated Learning Across Sites

- Model training distributed across edge/cloud sites without centralizing data
- Each site trains locally, shares only gradients/parameters
- Global model aggregation at a central coordinator

**Use case:** Healthcare (patient data stays in hospital), retail (store-level data sovereignty).

---

## 15.8 When to Use / When Not to Use Distributed Computing

### When to Use Distributed Computing

| Scenario | Why Distribution Helps |
|----------|----------------------|
| Model doesn't fit on one GPU | Pipeline/model parallelism |
| Training data > 1 TB | Data parallelism across workers |
| Need to reduce training time | Linear scaling with GPU count |
| Multi-modal training (text + image) | Different workers for different modalities |
| Large-scale hyperparameter search | Parallel trial evaluation (Ray Tune) |
| Serving needs > 1 GPU | Model parallelism for large models |

### When NOT to Use Distributed Computing

| Scenario | Why Distribution Hurts | Alternative |
|----------|----------------------|-------------|
| Model < 100M parameters | Communication overhead exceeds compute gain | Single GPU training |
| Data < 10 GB | Loading/processing is not the bottleneck | Single machine with DataParallel |
| Prototyping / debugging | Distributed debugging is 10x harder | Local training with small data subset |
| Network bandwidth < 10 Gbps | AllReduce becomes the bottleneck | Single-node multi-GPU |
| Team has no distributed systems experience | Operational complexity risk | Managed training service |
| Model has no parallelism opportunities | Sequential dependency graph | Optimize single-GPU performance |

---

## 15.9 Summary

- **Ray** (43.7K stars) provides a general-purpose distributed computing framework with native support for training, serving, and data processing, managed via KubeRay on Kubernetes
- **Scaling efficiency** is dominated by network topology: NVLink > InfiniBand > Ethernet, and most models achieve 70-85% linear scaling efficiency up to 32 GPUs before communication overhead dominates
- **Data shuffle bottlenecks** are the most common cause of poor scaling in distributed training, solvable through embedding sharding, prefetching, and gradient compression
- **Hybrid cloud patterns** allow organizations to balance data sovereignty, cost, and compute availability across cloud and on-premise infrastructure
- Distribution adds complexity that must be justified by measurable benefits — always benchmark single-node performance first

---

## Discussion Questions

1. You are training a 7B parameter language model. Your cluster has 32 A100 GPUs connected via 100 Gbps InfiniBand. Calculate the theoretical minimum communication time per AllReduce step and determine whether this network is sufficient for efficient training.

2. Compare Ray's actor-based model with Spark's RDD-based model. For a pipeline that preprocesses 5 TB of text data, trains a model on the processed data, and then runs inference on 1M new samples — which framework handles the full pipeline most naturally?

3. A team is considering federated learning across 5 hospitals for a medical imaging model. What architectural challenges must be solved beyond the distributed computing framework itself?

4. Why does GPU utilization drop as you add more GPUs to a training job, even with sufficient network bandwidth? What other bottlenecks exist beyond network communication?

5. Design a cost-optimal hybrid cloud architecture for a company that needs to train models daily but serve them 24/7. What instance types, pricing models, and data movement patterns would you recommend?

---

## Exercises

**Exercise 1:** Set up a 2-node Ray cluster using KubeRay on a Kubernetes cluster. Run a distributed training job using Ray Train with PyTorch, and measure the scaling efficiency compared to single-node training.

**Exercise 2:** Profile a distributed training job to identify the breakdown between computation time, communication time (AllReduce), and data loading time. Use PyTorch Profiler with NVTX ranges to create a timeline visualization.

**Exercise 3:** Implement a simple parameter server architecture in Ray using Actors. Compare its performance with Ray's built-in AllReduce training for a model with 500M parameters.

---

## References

- Ray Official Documentation: https://docs.ray.io/
- Ray GitHub Repository: https://github.com/ray-project/ray
- KubeRay Documentation: https://ray-project.github.io/kuberay/
- KubeRay GitHub Repository: https://github.com/ray-project/kuberay
- Anyscale Ray Benchmarks: https://www.anyscale.com/blog
- Apache Spark on Kubernetes: https://spark.apache.org/docs/latest/running-on-kubernetes.html
- Ring AllReduce Algorithm: https://github.com/baidu-research/bring-your-own-flexible-communication
- NVIDIA NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
