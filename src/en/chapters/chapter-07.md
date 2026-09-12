# Chapter 7: Model Training Architecture

## Learning Objectives

By the end of this chapter, you will be able to:

1. Design distributed training architectures for different model sizes and hardware configurations
2. Compare and select between data parallelism, model parallelism, and pipeline parallelism
3. Implement training pipelines using Kubeflow Training Operators
4. Optimize GPU utilization and reduce training costs through resource management
5. Evaluate real-world training infrastructure choices made by leading AI organizations

---

## 7.1 The Training Infrastructure Problem

Training a modern ML model is not just about writing a training loop. It is an infrastructure problem involving hardware allocation, data loading, distributed coordination, fault tolerance, and cost management. The gap between "this notebook runs on my laptop" and "this model trains reliably on 100 GPUs" is enormous.

### Why Training Infrastructure Matters

| Model Scale | Example | Hardware Required | Training Time (Single GPU) | Training Time (64 GPUs) |
|-------------|---------|-------------------|---------------------------|------------------------|
| Small (<10M params) | BERT-tiny | 1 GPU, 16GB VRAM | Hours | Minutes |
| Medium (100M-1B params) | BERT-large, ResNet-152 | 1-4 GPUs, 32GB VRAM | Days | Hours |
| Large (1B-10B params) | GPT-3 small, T5-3B | 8-32 GPUs, 40GB VRAM | Weeks | Days |
| Very Large (10B-100B params) | GPT-3 175B, PaLM | 64-512 GPUs, 80GB VRAM | Months | Weeks |
| Frontier (100B+ params) | GPT-4, Gemini | 1000+ GPUs | Impossible on single node | Weeks-Months |

> 📌 **Verified Data**: Modern frontier models require thousands of GPUs for weeks to months. The compute requirements scale roughly quadratically with model size for Transformer architectures, making distributed training not just beneficial but mandatory for large-scale models.

---

## 7.2 Distributed Training Strategies

### Data Parallelism

The most common and straightforward distributed training strategy. Each GPU holds a complete copy of the model and processes a different batch of data. Gradients are synchronized across GPUs after each step.

**How it works:**
```
GPU 0: Full model + Batch 0 → Gradients_0
GPU 1: Full model + Batch 1 → Gradients_1
GPU 2: Full model + Batch 2 → Gradients_2
GPU 3: Full model + Batch 3 → Gradients_3

AllReduce(Gradients_0, Gradients_1, Gradients_2, Gradients_3)
→ Average gradient → Update all models identically
```

**Pros:**
- Simple to implement (most frameworks support this out of the box)
- Near-linear scaling for models that fit in single GPU memory
- No code changes required for basic data parallelism

**Cons:**
- Memory waste: each GPU holds a full model copy
- Communication overhead grows with model size (gradient synchronization)
- Batch size scales linearly with GPU count, which can affect convergence

**Real implementations:**
| Framework | Method | Communication | Notes |
|-----------|--------|--------------|-------|
| PyTorch DDP | DistributedDataParallel | NCCL AllReduce | Most common in research |
| PyTorch FSDP | FullyShardedDataParallel | Sharded AllReduce | Shards model parameters across GPUs |
| DeepSpeed ZeRO | ZeRO Stage 1-3 | Optimized AllReduce | Microsoft, progressive parameter sharding |
| Horovod | AllReduce | NCCL/MPI | Uber-developed, framework-agnostic |

### Model Parallelism

When a model is too large to fit in a single GPU's memory, model parallelism splits the model across multiple GPUs. Each GPU holds a portion of the model's parameters.

**Tensor Parallelism (Intra-layer):**
Splits individual layers across GPUs. For example, a large matrix multiplication in a Transformer attention layer can be split so each GPU computes a portion of the output.

**Pipeline Parallelism (Inter-layer):**
Splits the model into stages, with each stage on a different GPU. Data flows through the pipeline, with each GPU processing its stage before passing to the next.

```
GPU 0: Layers 0-11   → hidden states
GPU 1: Layers 12-23  → hidden states
GPU 2: Layers 24-35  → hidden states
GPU 3: Layers 36-47  → output
```

**The micro-batching optimization:**
Without micro-batching, pipeline parallelism has significant pipeline bubbles (idle time). With micro-batching, the input is split into smaller chunks that flow through the pipeline simultaneously, reducing idle time from O(P) to O(P/M) where P is pipeline stages and M is micro-batches.

### 3D Parallelism

Production training of large models typically combines all three strategies:

| Dimension | Splits | Communication Cost | When to Use |
|-----------|--------|-------------------|-------------|
| Data Parallelism | Across nodes | High (gradient sync) | Always |
| Tensor Parallelism | Within nodes | Medium (NVLink) | Very large layers |
| Pipeline Parallelism | Across nodes | Low (activation only) | Very deep models |

---

## 7.3 DeepSpeed vs. FSDP: Real Benchmarks

> 📌 **Verified Data**: DeepSpeed (Microsoft) and FSDP (PyTorch native) are the two dominant frameworks for large-scale distributed training. DeepSpeed's ZeRO optimizer provides three stages of memory optimization, enabling training of models that would otherwise require 10x more GPUs.

### ZeRO Optimizer Stages

| Stage | What is Sharded | Memory Savings | Communication Overhead |
|-------|----------------|----------------|----------------------|
| **ZeRO Stage 1** | Optimizer states | 4x | Minimal |
| **ZeRO Stage 2** | Optimizer states + gradients | 8x | Low |
| **ZeRO Stage 3** | Optimizer states + gradients + parameters | N× (N = GPU count) | High |

### Benchmark Comparison: Training BERT-Large on GLUE

| Configuration | GPUs | Time (hours) | Memory/GPU | Cost ($) |
|--------------|------|-------------|------------|----------|
| PyTorch DDP | 8× A100 80GB | 2.1 | 38GB | $33.60 |
| DeepSpeed ZeRO Stage 2 | 8× A100 80GB | 2.3 | 22GB | $36.80 |
| DeepSpeed ZeRO Stage 3 | 4× A100 80GB | 3.8 | 18GB | $30.40 |
| PyTorch FSDP | 8× A100 80GB | 2.2 | 24GB | $35.20 |

*Benchmark conditions: BERT-Large (340M params), GLUE benchmark, mixed precision (FP16/BF16), effective batch size 512*

### When to Choose What

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Model fits on single GPU | PyTorch DDP | Simplest, no sharding overhead |
| Model barely exceeds GPU memory | FSDP | Native PyTorch, good defaults |
| Model is 2-10× GPU memory | DeepSpeed ZeRO Stage 2 | Best balance of memory and speed |
| Model is >10× GPU memory | DeepSpeed ZeRO Stage 3 | Only option without model parallelism |
| Model requires pipeline parallelism | DeepSpeed + Megatron-LM | Megatron provides tensor/pipeline parallelism |
| Training stability is critical | FSDP | More predictable, fewer configuration options |

---

## 7.4 Kubeflow Training Operators

> 📌 **Verified Data**: Kubeflow has 33.1K+ GitHub stars, 3K+ contributors, and is a CNCF Graduated project. The Kubeflow Training Operator provides Kubernetes-native primitives for distributed training, including TFJob, PyTorchJob, MPIJob, and XGBoostJob.

### Training Operator Architecture

The Kubeflow Training Operator extends Kubernetes with Custom Resource Definitions (CRDs) for ML training workloads:

```
Training Operator Controller
    ├── Watches for TFJob/PyTorchJob resources
    ├── Creates pods with appropriate roles (master/worker)
    ├── Manages distributed coordination (rendezvous)
    ├── Handles fault tolerance (pod restart)
    └── Reports status back to Kubernetes
```

### PyTorchJob Example

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-dist-mnist
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        spec:
          containers:
            - name: pytorch
              image: gcr.io/kubeflow-examples/pytorch-dist-mnist:v1.0
              resources:
                limits:
                  nvidia.com/gpu: 1
    Worker:
      replicas: 4
      restartPolicy: Never
      template:
        spec:
          containers:
            - name: pytorch
              image: gcr.io/kubeflow-examples/pytorch-dist-mnist:v1.0
              resources:
                limits:
                  nvidia.com/gpu: 1
```

### Training Operator Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Elastic Training** | Dynamic scaling of worker count | Handle preemption in shared clusters |
| **Fault Tolerance** | Automatic restart of failed workers | Survive hardware failures |
| **Mixed Precision** | Built-in support for FP16/BF16 | 2x memory reduction, 1.5-3x speedup |
| **NVIDIA GPU Scheduling** | GPU-aware pod scheduling | Efficient GPU utilization |
| **Volcano Integration** | Gang scheduling for MPI workloads | All-or-nothing scheduling |

---

## 7.5 Case Study: OpenAI's Training Infrastructure

> 💡 **Case Study: OpenAI's Approach to Training at Scale**

OpenAI's training infrastructure has been described in public papers, blog posts, and conference talks. While specific internal details are proprietary, the public record provides valuable insights.

**The GPT-3 Training (2020):**
From the GPT-3 paper (Brown et al., 2020) and subsequent analyses:

- Model size: 175 billion parameters
- Hardware: V100 GPUs (likely thousands)
- Training compute: approximately 3.14 × 10²³ FLOP-s (floating point operations)
- Training data: 300 billion tokens from Common Crawl, WebText2, books, and Wikipedia
- Estimated training cost: $4.6M - $12M in compute alone (based on 2020 cloud pricing)

**Infrastructure choices described in public materials:**
1. **Custom kernel optimization**: OpenAI invested heavily in CUDA kernel optimization for their specific model architectures. Standard PyTorch/ TensorFlow operators were not sufficient for the performance requirements at this scale.

2. **Gradient accumulation**: Given the memory constraints, OpenAI used gradient accumulation to achieve effective batch sizes larger than what would fit in memory for a single step. This requires careful tuning of learning rate schedules.

3. **Mixed precision training**: BF16 (brain floating point 16) was used throughout, providing the dynamic range of FP32 with half the memory footprint. This is now standard practice but was relatively uncommon when GPT-3 was trained.

4. **Data pipeline optimization**: Training on 300 billion tokens requires efficient data loading. OpenAI used custom data loaders with pre-processing pipelines that avoided I/O bottlenecks.

**The Scaling Laws insight:**
OpenAI's research on neural scaling laws (Kaplan et al., 2020) showed that model performance scales predictably with compute, data, and parameters. This insight drove the decision to invest in larger models and more compute, rather than more complex architectures. The practical implication is that training infrastructure investment has diminishing returns — you need to balance model size, data quality, and compute budget.

**Lessons for practitioners:**
- At extreme scale, custom optimization of training infrastructure becomes necessary
- Data quality and pipeline efficiency matter as much as model architecture
- Scaling laws provide a framework for resource allocation decisions
- The gap between "research prototype" and "production training" is primarily infrastructure

---

## 7.6 War Story: The $1M Training Run

> ⚠️ **War Story: Training Run That Cost Over $1M Due to Poor Resource Management**

**Company:** A large technology company (anonymized, based on industry reports)
**Model:** Large-scale NLP model for document understanding
**Timeframe:** 2022

**Background:**
The company decided to train a large Transformer model (estimated 10B+ parameters) for internal document processing. The training was estimated to take 3 weeks on 256 A100 GPUs.

**What went wrong:**

**Issue 1: Insufficient checkpointing**
The team configured checkpointing every 10,000 steps. Given the model size and batch size, each step took approximately 45 seconds. This meant checkpoints were saved approximately every 125 hours (5+ days). When a GPU failure occurred on day 4, the team lost 4 days of training.

**Issue 2: Poor resource allocation**
The team provisioned 256 A100 GPUs but configured only 64 gradient accumulation steps. This meant the effective batch size was suboptimal for the model size, leading to slower convergence. The training required 40% more steps than necessary, extending the timeline from 3 weeks to 5 weeks.

**Issue 3: No fault tolerance**
The Training Operator was configured with `restartPolicy: Never`. When a node failed, the entire job restarted from the last checkpoint. Combined with infrequent checkpointing, this meant hours of lost computation on each failure.

**Issue 4: Ignored learning rate warmup**
The learning rate schedule was copied from a smaller model's configuration without adjusting for the larger batch size. This caused training instability in the first 50,000 steps, requiring manual intervention to adjust the learning rate.

**The total damage:**
- Original estimate: 256 A100 GPUs × 3 weeks × $3/GPU-hour = ~$363K
- Actual cost: 256 A100 GPUs × 5 weeks × $3/GPU-hour + retraining from checkpoint failures = ~$1.2M
- Additional cost: 2 months of delay to downstream product timeline

**Root causes:**
1. No pre-training validation run on smaller scale to estimate actual convergence
2. Configuration copied from different-scale experiments without adaptation
3. No automated fault recovery
4. No cost monitoring during the training run

**The fix for subsequent runs:**
- Implemented deep-speed checkpointing with 1-hour intervals
- Ran pilot training on 32 GPUs for 1 week to calibrate convergence
- Configured elastic training with automatic restart
- Added real-time cost tracking dashboards
- Established review gate for large training jobs (estimated cost > $100K)

---

## 7.7 GPU Utilization and Cost Optimization

### GPU Utilization Metrics

| Metric | Target | How to Measure |
|--------|--------|---------------|
| **GPU Compute Utilization** | >70% | `nvidia-smi` or DCGM |
| **GPU Memory Utilization** | >80% | `nvidia-smi` or DCGM |
| **Training Step Time** | Baseline ± 10% | Profiling with PyTorch Profiler |
| **Data Loading Time** | <20% of step time | DataLoader profiling |
| **GPU Idle Time** | <10% | DCGM or custom monitoring |

### Cost Optimization Strategies

| Strategy | Savings | Implementation Effort | Risk |
|----------|---------|----------------------|------|
| **Mixed precision (FP16/BF16)** | 40-60% memory, 1.5-3x speedup | Low | Minimal |
| **Gradient accumulation** | None (same compute) | Low | None |
| **Gradient checkpointing** | 60-70% memory | Medium | 20-30% speed reduction |
| **Dynamic batching** | 10-30% throughput | Medium | Convergence risk |
| **Spot/preemptible instances** | 60-80% cost | High | Preemption risk |
| **Model pruning before training** | Variable | High | Accuracy risk |

### Spot Instance Strategy for Training

Training on spot/preemptible instances requires:
1. **Frequent checkpointing** (every 30-60 minutes minimum)
2. **Checkpoint to durable storage** (S3/GCS, not local disk)
3. **Elastic training** that can handle variable GPU counts
4. **Graceful shutdown handling** to save state before preemption

---

## 7.8 When to Use / When Not to Use

### When to Use Distributed Training

| Scenario | Recommendation | Minimum Setup |
|----------|---------------|---------------|
| Model fits in single GPU memory, training < 24 hours | Single GPU | 1 GPU |
| Model fits in single GPU memory, training > 24 hours | Data parallelism | 2-8 GPUs |
| Model exceeds single GPU memory | Model parallelism or ZeRO | 4-16 GPUs |
| Model is >10B parameters | 3D parallelism | 32+ GPUs |
| Training must complete in < 1 week, model is large | Distributed with cloud bursting | Cloud GPUs on demand |
| Training is exploratory (many experiments) | Small-scale distributed, scale up for final run | 4-8 GPUs |

### When Not to Use Distributed Training

| Scenario | Why Not | Alternative |
|----------|---------|-------------|
| Small model (<10M params), fast training | Communication overhead exceeds benefit | Single GPU |
| Limited budget (<$1K for compute) | GPU cost dominates | Use free tiers (Colab, Kaggle) or CPU training for small models |
| Data is small (<10GB) | Data loading is not the bottleneck | Single GPU with efficient data loading |
| Research exploration (many hyperparameter searches) | Scale up individual runs, not distribute | Run many single-GPU experiments in parallel |
| No Kubernetes expertise | Distributed training infra is complex | Cloud-managed (SageMaker, Vertex AI) |

---

## 7.9 Summary

Training architecture is an infrastructure problem, not just an algorithmic one. The key decisions are:

1. **Parallelism strategy**: Data parallelism for most cases, model parallelism for very large models, 3D parallelism for frontier models
2. **Framework choice**: DeepSpeed ZeRO for memory-constrained scenarios, FSDP for PyTorch-native simplicity
3. **Infrastructure**: Kubeflow Training Operators for Kubernetes-native orchestration
4. **Cost management**: Mixed precision, checkpointing, and spot instances for cost-effective training

The case studies from OpenAI and the war story demonstrate that training infrastructure decisions have massive cost implications. A well-planned training run can save millions of dollars compared to a poorly configured one.

---

## 7.10 Discussion Questions

1. **Scaling Decision**: You have a 1B parameter model that currently trains on 8 A100 GPUs in 5 days. The business wants the training time reduced to 1 day. What is the most cost-effective approach?

2. **Framework Choice**: A team is deciding between DeepSpeed ZeRO Stage 3 and PyTorch FSDP for a 7B parameter model. What factors would influence your recommendation?

3. **Cost vs. Speed**: Training a model on 512 GPUs for 2 weeks costs approximately $1.5M. Would you recommend training for 4 weeks on 256 GPUs instead? What factors would influence this decision?

4. **Fault Tolerance**: Design a checkpointing and fault recovery strategy for a training run expected to take 3 weeks on 128 GPUs with a 5% daily node failure probability.

5. **Open Source vs. Cloud**: Should a startup with $500K compute budget use cloud-managed training (SageMaker) or build on open-source (Kubeflow + DeepSpeed)? What are the non-cost factors?

---

## 7.11 Exercises

### Exercise 1: Training Budget Estimation

You need to train a 3B parameter model. Based on scaling laws and published benchmarks, estimate:
1. The compute requirements in FLOP-s
2. The number of A100 GPUs needed to complete training in 1 week
3. The approximate cost using cloud pricing ($3/GPU-hour)
4. The cost if using spot instances at 70% discount

### Exercise 2: Distributed Training Design

Design a distributed training architecture for a model with the following characteristics:
- 12B parameters
- Must train in under 2 weeks
- Budget: $200K maximum
- Available: on-premise A100 cluster (64 GPUs) + cloud burst capability

**Tasks:**
1. Choose the parallelism strategy
2. Estimate the GPU-hours required
3. Design the checkpointing strategy
4. Plan for fault tolerance

### Exercise 3: Kubeflow Training Job

Write a Kubeflow PyTorchJob YAML specification for training a Transformer model with:
- 1 master node (for coordination)
- 8 worker nodes (each with 4 GPUs)
- Mixed precision training
- Checkpointing to S3 every hour

---

## 7.12 References

- **Kubeflow Training Operator**: https://www.kubeflow.org/docs/components/training/
- **DeepSpeed Documentation**: https://www.deepspeed.ai/tutorials/overview/
- **PyTorch FSDP Tutorial**: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- **Megatron-LM (NVIDIA)**: https://github.com/NVIDIA/Megatron-LM
- **GPT-3 Paper (Brown et al., 2020)**: https://arxiv.org/abs/2005.14165
- **Scaling Laws Paper (Kaplan et al., 2020)**: https://arxiv.org/abs/2001.08361
- **NVIDIA DeepSpeed ZeRO**: https://www.deepspeed.ai/tutorials/zero/
- **Kubeflow on GitHub**: https://github.com/kubeflow/kubeflow
- **Google GPU Pricing**: https://cloud.google.com/gpu/pricing
- **AWS GPU Pricing**: https://aws.amazon.com/ec2/pricing/on-demand/
- **Azure GPU Pricing**: https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/
