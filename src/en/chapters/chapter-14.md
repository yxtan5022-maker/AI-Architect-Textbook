# Chapter 14: AI Workloads on Kubernetes

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain how Kubernetes orchestrates GPU-accelerated AI workloads using the NVIDIA GPU Operator
2. Design and deploy Kubeflow pipelines on a Kubernetes cluster
3. Implement multi-tenant isolation patterns for shared AI infrastructure
4. Diagnose and mitigate GPU resource contention issues in production clusters
5. Evaluate when Kubernetes is appropriate for AI workloads versus alternative orchestration platforms

---

## 14.1 Introduction: Why Kubernetes for AI?

Running AI workloads in production requires more than a single powerful machine. Teams need elastic scaling, resource isolation, reproducibility, and multi-user access. Kubernetes has become the de facto orchestration layer for AI infrastructure because it addresses these needs through a unified API.

However, Kubernetes was designed for stateless web services. Running GPU-intensive, memory-hungry, and often long-running training jobs on a system designed for short-lived containers introduces real architectural tension. This chapter explores how to reconcile these tensions with production-grade tooling.

> **📌 Real Data Box**
> Kubeflow has **33,100+ GitHub stars** and is a **CNCF Graduated** project, meaning it has demonstrated maturity and production adoption across the Kubernetes ecosystem (kubeflow.org, 2026). CNCF Graduated status requires passing security audits, governance reviews, and demonstrating real-world deployment at scale.

---

## 14.2 GPU Scheduling Mechanics on Kubernetes

### 14.2.1 The NVIDIA GPU Operator

The NVIDIA GPU Operator is the standard mechanism for making GPUs available to Kubernetes. It automates the deployment and management of all NVIDIA software components required to provision GPUs in a cluster.

**Core components the GPU Operator manages:**

| Component | Purpose |
|-----------|---------|
| NVIDIA Driver | Kernel-level GPU driver |
| NVIDIA Container Toolkit | Enables GPU access inside containers |
| Device Plugin | Exposes GPUs as schedulable resources to Kubernetes |
| GPU Feature Discovery | Automatically labels nodes with GPU capabilities |
| NVIDIA Exporter | Exposes GPU metrics to Prometheus |

**How GPU scheduling actually works:**

1. The GPU Operator installs a **device plugin** on each GPU node
2. The device plugin calls NVIDIA's Management Library (NVML) to detect available GPUs
3. GPUs are advertised to the Kubernetes scheduler as extendable resources (`nvidia.com/gpu`)
4. When a pod requests `nvidia.com/gpu: 1`, the scheduler finds a node with a free GPU
5. The Container Toolkit mounts the GPU into the container's cgroup

**Critical nuance:** Kubernetes GPU scheduling is **bin-packed, not time-shared**. If a pod requests 1 GPU, it gets exclusive access to that physical GPU for the entire pod lifetime. There is no native GPU sharing in standard Kubernetes — a pod requesting 1 GPU on a node with 8 GPUs will prevent other pods from using that specific GPU, even if the workload only needs 20% of its compute capacity.

### 14.2.2 GPU Resource Requests and Limits

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-job
spec:
  containers:
  - name: trainer
    image: nvcr.io/nvidia/pytorch:23.10-py3
    resources:
      requests:
        nvidia.com/gpu: 2
        memory: "32Gi"
        cpu: "8"
      limits:
        nvidia.com/gpu: 2
        memory: "64Gi"
        cpu: "16"
```

**Important distinctions:**

- `requests` determines scheduling — the scheduler uses this to find a suitable node
- `limits` determines enforcement — the container is killed if it exceeds these
- For GPUs, there is **no overcommit** — you cannot request more GPUs than physically exist
- MIG (Multi-Instance GPU) on A100/H100 cards allows partitioning a single GPU into up to 7 isolated instances

### 14.2.3 Multi-Instance GPU (MIG) on Kubernetes

NVIDIA's MIG technology, available on A100, H100, and newer data center GPUs, allows a single GPU to be partitioned into up to 7 isolated GPU instances. Each instance has its own dedicated memory, compute cores, and bandwidth.

**MIG profiles for A100 80GB:**

| Profile | Memory | Compute |
|---------|--------|---------|
| 1g.10gb | 10 GB | 1/7 SMs |
| 2g.20gb | 20 GB | 2/7 SMs |
| 3g.40gb | 40 GB | 3/7 SMs |
| 4g.40gb | 40 GB | 4/7 SMs |
| 7g.80gb | 80 GB | 7/7 SMs (full GPU) |

The GPU Operator's GPU Feature Discovery automatically detects MIG-enabled GPUs and creates topology labels that allow the Kubernetes scheduler to target specific MIG profiles.

---

## 14.3 Kubeflow on Kubernetes Architecture

Kubeflow is not a single application — it is a collection of loosely coupled components that together form an end-to-end ML platform. Understanding its architecture requires understanding each component's role.

### 14.3.1 Core Components

| Component | Role | K8s Resources |
|-----------|------|---------------|
| **Kubeflow Pipelines** | Workflow orchestration (DAG-based) | Argo Workflows + custom controller |
| **Training Operators** | Manages distributed training jobs | Custom resources (TFJob, PyTorchJob, etc.) |
| **Katib** | Hyperparameter tuning | Custom resource + metrics collection |
| **KServe** | Model serving (inference) | Knative + Istio (optional) |
| **Notebooks** | Jupyter/Lab environments | StatefulSets with PVC |
| **Central Dashboard** | Unified UI | Deployment + ConfigMap |

### 14.3.2 Kubeflow Pipelines Architecture

Kubeflow Pipelines runs on top of **Argo Workflows**, a Kubernetes-native workflow engine. When you create a pipeline:

1. The pipeline DSL compiles to a YAML workflow definition
2. The KFP backend persists the workflow and metadata
3. Argo Workflows creates pods for each pipeline step
4. Each step runs in an isolated container with its own image and resource requirements
5. Steps are connected via artifacts (stored in MinIO/S3) and parameter passing

**Multi-tenancy model:**

Kubeflow uses a **profile** abstraction. Each profile maps to a Kubernetes namespace with:
- Its own resource quota
- Its own service account
- Network policies isolating it from other profiles
- Per-profile persistent volumes

```bash
# Create a new profile
kubectl create -f - <<EOF
apiVersion: kubeflow.org/v1
kind: Profile
metadata:
  name: data-science-team
spec:
  owner:
    kind: User
    name: alice@company.com
  resourceQuotaSpec:
    hard:
      cpu: "32"
      memory: "128Gi"
      nvidia.com/gpu: "8"
EOF
```

### 14.3.3 Training Operators for Distributed Training

Kubeflow's Training Operators define Custom Resources (CRDs) that abstract distributed training frameworks:

| CRD | Framework | Distribution Strategy |
|-----|-----------|----------------------|
| `PyTorchJob` | PyTorch | `torchrun` with world size |
| `TFJob` | TensorFlow | Parameter server + workers |
| `XGBoostJob` | XGBoost | AllReduce |
| `MPIJob` | MPI-based | `mpirun` across pods |
| `PaddleJob` | PaddlePaddle | AllReduce |

**Example: Distributed PyTorch Training**

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: resnet-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: myregistry/resnet-trainer:latest
            resources:
              limits:
                nvidia.com/gpu: 1
    Worker:
      replicas: 3
      template:
        spec:
          containers:
          - name: pytorch
            image: myregistry/resnet-trainer:latest
            resources:
              limits:
                nvidia.com/gpu: 4
```

When this CRD is applied, the Training Operator creates a master pod and 3 worker pods, configures the PyTorch distributed runtime environment variables (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK), and monitors job health.

---

## 14.4 Multi-Tenant Isolation Patterns

Running AI workloads for multiple teams on shared infrastructure requires careful isolation at every layer.

### 14.4.1 Namespace-Level Isolation

Each team or project gets its own namespace with:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-quota
  namespace: ml-team-alpha
spec:
  hard:
    requests.cpu: "64"
    requests.memory: "256Gi"
    requests.nvidia.com/gpu: "8"
    limits.cpu: "128"
    limits.memory: "512Gi"
    persistentvolumeclaims: "20"
```

### 14.4.2 Network Isolation

Network policies prevent pods in one namespace from communicating with pods in another:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-cross-namespace
  namespace: ml-team-alpha
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ml-team-alpha
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: ml-team-alpha
```

### 14.4.3 GPU Isolation

For GPU isolation beyond namespace quotas, use **device plugins** combined with **node affinity** or **MIG**:

**Option 1: Node Affinity (coarse)**
```yaml
nodeSelector:
  gpu-pool: team-alpha
```

**Option 2: MIG (fine-grained)**
```yaml
resources:
  limits:
    nvidia.com/mig-3g.40gb: 2
```

**Option 3: Time-slicing (sharing)**
The NVIDIA GPU Operator supports time-slicing, where multiple pods share a single GPU by alternating execution. This reduces isolation but increases utilization.

### 14.4.4 Storage Isolation

Each tenant gets isolated persistent storage through StorageClasses and per-namespace PVCs. Common pattern: provision fast SSD storage for training jobs and cheaper object storage for artifacts.

---

## 14.5 Case Study: How Google Runs AI on Kubernetes

Google's internal ML platform, built on Kubernetes, serves thousands of researchers running models from language understanding to protein folding. While Google does not use Kubeflow internally (they use Borg, Kubernetes' predecessor), the architectural patterns are directly transferable.

**Key architectural decisions:**

| Decision | Implementation | Rationale |
|----------|---------------|-----------|
| Centralized resource management | Borg/K8s with custom scheduler | Prevents resource hoarding |
| Preemptible VMs for training | Spot/preemptible instances | 60-70% cost reduction |
| Hierarchical quotas | Organization → Team → Project | Fair sharing across thousands of users |
| Custom scheduling policies | Bin-packing with GPU locality | Minimizes GPU fragmentation |
| Automated checkpointing | GCS-backed distributed checkpoints | Fault tolerance for long jobs |

**Scale metrics (approximate, from published research):**

- Millions of training jobs per month
- Hundreds of thousands of GPU nodes
- Average GPU utilization: 60-80% (improved from 40% with scheduling changes)
- Average training job duration: 2-8 hours
- Largest single training job: thousands of TPUs running for weeks

**Lessons learned:**

1. **GPU utilization is the primary cost driver.** A cluster running at 40% GPU utilization costs 2.5x more per effective GPU-hour than one at 80%.
2. **Preemptibility is essential.** Without it, teams over-provision to guarantee availability.
3. **Multi-tenancy requires quotas at every level** — namespace, node pool, and global.
4. **Checkpointing is not optional** — with preemption, jobs must be resumable from any checkpoint.

---

## 14.6 War Story: GPU Memory Leak Crashing an Entire Cluster

**Company:** Mid-size autonomous driving startup, 200+ GPU cluster

**Problem:** A training job contained a PyTorch bug where CUDA tensors were created inside a Python `for` loop without being explicitly freed. The tensors were stored in a list that was never cleared, causing the GPU memory to grow by ~200MB per iteration.

**Timeline:**

| Time | Event |
|------|-------|
| 0:00 | Training job starts, requests 8x A100 GPUs |
| 2:00 | GPU memory on worker 0 reaches 90% (leaked to ~72GB) |
| 3:00 | OOM killer activates, kills the training process |
| 3:01 | The killed process leaves orphaned CUDA contexts |
| 3:05 | NVIDIA device plugin loses contact with GPU 0 on worker 0 |
| 3:06 | Device plugin marks GPUs 0-7 as unhealthy on worker 0 |
| 3:07 | Kubernetes evicts all pods on worker 0 |
| 3:10 | Remaining 7 GPUs on worker 0 are excluded from scheduling |
| 3:15 | Other jobs start queueing, waiting for freed GPUs |
| 3:30 | Cascading failures as other jobs hit their own OOM limits |

**Root cause:** The NVIDIA device plugin on worker 0 detected GPU unresponsiveness and reported the entire node's GPU resources as unavailable. Kubernetes then removed the node from the scheduling pool. The orphaned CUDA contexts were not cleaned up because the container runtime did not correctly propagate SIGTERM to the CUDA driver.

**Fixes applied:**

1. **Pre-job validation:** Added a memory pre-check that runs a small allocation test before starting the actual training job
2. **Resource limits with Kubernetes:** Set GPU memory limits using MIG profiles so each job is isolated
3. **GPU health monitoring:** Deployed DCGM (Data Center GPU Manager) to export GPU health metrics to Prometheus
4. **Automated remediation:** Created a controller that detects GPU unresponsiveness and triggers a GPU reset (nvidia-smi --gpu-reset) without evicting the entire node
5. **Code review gate:** Added a CI check that flags CUDA tensor allocation inside loops without explicit deletion

**Outcome:** GPU-related cluster downtime dropped from 12 hours/month to under 30 minutes/month.

---

## 14.7 When to Use / When Not to Use Kubernetes for AI

### When to Use Kubernetes for AI

| Scenario | Why K8s Fits |
|----------|-------------|
| Multi-team shared GPU cluster | Namespace isolation, resource quotas, RBAC |
| Mixed workloads (training + serving) | Unified orchestration, different resource profiles |
| Auto-scaling needs | Cluster autoscaler, node pool management |
| Hybrid cloud / multi-cloud | Consistent API across environments |
| Regulatory compliance needs | Audit logging, network policies, RBAC |
| Large-scale distributed training | Training Operators handle topology |

### When NOT to Use Kubernetes for AI

| Scenario | Why K8s Doesn't Fit | Alternative |
|----------|---------------------|-------------|
| Single-team with < 8 GPUs | Operational overhead exceeds benefit | Docker Compose or single-node |
| Interactive notebook-only work | StatefulSet complexity, JupyterHub simpler | JupyterHub on VMs |
| Real-time low-latency serving | K8s networking adds 1-3ms latency | Dedicated inference servers |
| Prototyping / experimentation | Kubernetes YAML overhead slows iteration | Local Docker + MLflow |
| Tiny models (< 100MB) | GPU scheduling overhead exceeds model run | CPU-only containers |
| Budget-constrained (no SRE team) | K8s requires dedicated operations expertise | Managed services (SageMaker, Vertex AI) |

---

## 14.8 Summary

- The **NVIDIA GPU Operator** automates GPU provisioning on Kubernetes, but GPU scheduling is bin-packed and non-overcommittable — a pod gets exclusive access to a physical GPU
- **Kubeflow** is a CNCF Graduated project (33.1K+ stars) providing Pipelines, Training Operators, Katib, KServe, and Notebooks on Kubernetes
- **Multi-tenancy** requires isolation at every layer: namespaces, network policies, GPU device plugins, storage classes, and resource quotas
- **MIG** technology allows partitioning A100/H100 GPUs into isolated instances for finer-grained resource sharing
- Kubernetes is powerful for AI infrastructure but introduces operational overhead that must be justified by scale and multi-tenancy requirements

---

## Discussion Questions

1. A startup has 4 data scientists sharing 4 GPUs. They currently use a shared Jupyter server. The CTO wants to deploy Kubeflow Pipelines for experiment tracking. Is this justified? What would you recommend instead?

2. How does GPU time-slicing differ from MIG in terms of isolation, performance, and operational complexity? In what scenario would you choose one over the other?

3. A team runs distributed PyTorch training jobs that take 6-12 hours. The cluster uses preemptible instances. Design a checkpointing strategy that balances cost savings from preemption with training progress protection.

4. The NVIDIA device plugin reports a GPU as unhealthy, causing all pods on the node to be evicted. How would you design a more resilient remediation strategy?

5. Compare the operational complexity of running KServe on Kubernetes versus deploying the same model on a dedicated inference platform like NVIDIA Triton. What factors drive the decision?

---

## Exercises

**Exercise 1:** Deploy a Kubeflow Training Operator that runs a distributed PyTorch training job across 2 nodes with 2 GPUs each. Verify that the master and worker pods can communicate via the NCCL rendezvous backend.

**Exercise 2:** Create a namespace with a resource quota limiting GPU access to 4 GPUs. Deploy 3 pods each requesting 2 GPUs. Observe which pod is pending and why. Then deploy a 4th pod requesting 1 GPU and confirm it schedules successfully.

**Exercise 3:** Set up DCGM exporter in your cluster, configure Prometheus to scrape GPU metrics, and create a Grafana dashboard showing GPU utilization, memory usage, and temperature across all nodes.

---

## References

- NVIDIA GPU Operator Documentation: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
- Kubeflow Official Documentation: https://www.kubeflow.org/docs/
- Kubeflow GitHub Repository: https://github.com/kubeflow/kubeflow
- Kubernetes Device Plugins: https://kubernetes.io/docs/concepts/extend-kubernetes-compute-storage-plugins/device-plugins/
- NVIDIA Multi-Instance GPU (MIG): https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- CNCF Kubeflow Graduation Announcement: https://www.cncf.io/announcements/
- Argo Workflows: https://argoproj.github.io/argo-workflows/
- NVIDIA DCGM Exporter: https://github.com/NVIDIA/dcgm-exporter
