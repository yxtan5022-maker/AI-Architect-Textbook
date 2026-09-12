# Chapter 14: AI Workloads on Kubernetes

🟢 Beginner | 🟡 Intermediate | 🔴 Advanced | ⚫ Manager

---

## 14.1 Kubernetes Fundamentals Review

### Why Kubernetes for AI?

Kubernetes has become the de facto standard for orchestrating AI workloads in production. Understanding why requires examining the unique characteristics of AI workloads:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Workload Characteristics                  │
├─────────────────────────────────────────────────────────────────┤
│  • GPU-intensive computations                                  │
│  • Variable resource demands (training vs inference)           │
│  • Distributed processing requirements                        │
│  • Need for reproducible environments                         │
│  • Batch processing with SLA requirements                      │
│  • Multi-tenancy across teams                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Kubernetes Concepts for AI

📌 **Key Concept**: Kubernetes provides the abstraction layer needed to manage heterogeneous compute resources (CPU, GPU, TPU, memory) efficiently.

```yaml
# Pod definition for a training job
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
  labels:
    app: model-training
spec:
  containers:
  - name: trainer
    image: tensorflow/tensorflow:2.12.0-gpu
    command: ["python", "train.py"]
    resources:
      requests:
        memory: "8Gi"
        cpu: "4"
        nvidia.com/gpu: "2"
      limits:
        memory: "16Gi"
        cpu: "8"
        nvidia.com/gpu: "2"
    volumeMounts:
    - name: dataset
      mountPath: /data/datasets
    - name: model-output
      mountPath: /data/models
  volumes:
  - name: dataset
    persistentVolumeClaim:
      claimName: training-dataset-pvc
  - name: model-output
    persistentVolumeClaim:
      claimName: model-output-pvc
  nodeSelector:
    accelerator: nvidia-tesla-v100
```

### Kubernetes Architecture for AI

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes AI Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Control Plane                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │   API    │  │Scheduler │  │Controller│  │  etcd│  │   │
│  │  │  Server  │  │          │  │ Manager  │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     Worker Nodes                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │  GPU Node 1  │  │  GPU Node 2  │  │  CPU Node 1  │  │   │
│  │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │  │   │
│  │  │  │4xV100  │  │  │  │4xA100  │  │  │  │64 CPU  │  │  │   │
│  │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Namespace Strategy for AI Teams

```yaml
# Namespace configuration for AI workloads
apiVersion: v1
kind: Namespace
metadata:
  name: ai-training
  labels:
    team: ml-engineering
    environment: production
    istio-injection: enabled
---
# Resource quota for the namespace
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ai-training-quota
  namespace: ai-training
spec:
  hard:
    requests.cpu: "128"
    requests.memory: "256Gi"
    requests.nvidia.com/gpu: "16"
    limits.cpu: "256"
    limits.memory: "512Gi"
    limits.nvidia.com/gpu: "32"
    pods: "100"
    persistentvolumeclaims: "50"
---
# Limit range to prevent resource waste
apiVersion: v1
kind: LimitRange
metadata:
  name: ai-training-limits
  namespace: ai-training
spec:
  limits:
  - default:
      cpu: "8"
      memory: "16Gi"
      nvidia.com/gpu: "1"
    defaultRequest:
      cpu: "2"
      memory: "4Gi"
      nvidia.com/gpu: "1"
    max:
      cpu: "32"
      memory: "64Gi"
      nvidia.com/gpu: "8"
    min:
      cpu: "500m"
      memory: "1Gi"
    type: Container
```

---

## 14.2 GPU Scheduling & Management

### NVIDIA GPU Operator

🟢 Beginner

The NVIDIA GPU Operator automates the management of GPU drivers, CUDA toolkit, and device plugins across all nodes.

```bash
# Install GPU Operator using Helm
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

# Install with default settings
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true

# Verify installation
kubectl get pods -n gpu-operator
```

### GPU Resource Discovery

```yaml
# Check GPU availability on nodes
apiVersion: v1
kind: Node
metadata:
  name: gpu-node-1
  labels:
    accelerator: nvidia-tesla-v100
status:
  capacity:
    nvidia.com/gpu: "4"
    cpu: "32"
    memory: "128Gi"
  allocatable:
    nvidia.com/gpu: "4"
    cpu: "31"
    memory: "120Gi"
```

### GPU Sharing with MIG

🔴 Advanced

Multi-Instance GPU (MIG) allows splitting a single A100 GPU into multiple isolated instances.

```bash
# Enable MIG on NVIDIA A100
nvidia-smi -i 0 -mig 1

# Create MIG instances
nvidia-smi -i 0 -cgi 19,19,19,19 -C

# Verify MIG instances
nvidia-smi --query-gpu=mig.mode.current,gpu.bus_id --format=csv
```

### GPU Scheduling Strategies

```yaml
# Priority-based GPU scheduling
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-training-high
value: 1000000
globalDefault: false
description: "High priority for GPU training jobs"
---
# Topology-aware scheduling for multi-GPU jobs
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: distributed-training
spec:
  serviceName: training-headless
  replicas: 4
  selector:
    matchLabels:
      app: distributed-training
  template:
    metadata:
      labels:
        app: distributed-training
    spec:
      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: distributed-training
            topologyKey: kubernetes.io/hostname
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: accelerator
                operator: In
                values:
                - nvidia-tesla-v100
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
        resources:
          requests:
            nvidia.com/gpu: "1"
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: "1"
            memory: "32Gi"
            cpu: "8"
```

---

## 14.3 AI-Specific Operators

### Kubeflow Operators

🟡 Intermediate

Kubeflow provides a suite of operators for ML lifecycle management:

```yaml
# Training Operator - PyTorchJob
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-ddp-training
  namespace: ai-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
            command:
            - python
            - -m
            - torch.distributed.run
            - --nproc_per_node=4
            - --nnodes=4
            - --rdzv_backend=c10d
            - --rdzv_endpoint=$(MASTER_ADDR):29500
            - train.py
            resources:
              limits:
                nvidia.com/gpu: "4"
                memory: "32Gi"
                cpu: "8"
    Worker:
      replicas: 3
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
            command:
            - python
            - -m
            - torch.distributed.run
            - --nproc_per_node=4
            - --nnodes=4
            - --rdzv_backend=c10d
            - --rdzv_endpoint=$(MASTER_ADDR):29500
            - train.py
            resources:
              limits:
                nvidia.com/gpu: "4"
                memory: "32Gi"
                cpu: "8"
```

### Volcano - Batch Scheduling for AI

```yaml
# Volcano Job for gang scheduling
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: gang-scheduled-training
  namespace: ai-training
spec:
  minAvailable: 3
  schedulerName: volcano
  policies:
  - event: PodEvicted
    action: RestartJob
  plugins:
  - name: drf
  - name: predicates
  - name: proportion
  tasks:
  - replicas: 2
    name: worker
    template:
      spec:
        containers:
        - name: worker
          image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
          resources:
            requests:
              nvidia.com/gpu: "2"
              memory: "16Gi"
            limits:
              nvidia.com/gpu: "2"
              memory: "32Gi"
        tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

### KubeFlow Pipelines Operator

```yaml
# Pipeline definition for ML workflow
apiVersion: kubeflow.org/v1beta1
kind: Pipeline
metadata:
  name: training-pipeline
  namespace: ai-training
spec:
  description: End-to-end ML training pipeline
  steps:
  - name: data-preprocessing
    template:
      container:
        image: data-preprocessor:latest
        command: [python, preprocess.py]
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
  - name: model-training
    template:
      container:
        image: model-trainer:latest
        command: [python, train.py]
        resources:
          requests:
            nvidia.com/gpu: "2"
            memory: "16Gi"
    dependencies:
    - data-preprocessing
  - name: model-evaluation
    template:
      container:
        image: model-evaluator:latest
        command: [python, evaluate.py]
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
    dependencies:
    - model-training
  - name: model-deployment
    template:
      container:
        image: model-deployer:latest
        command: [python, deploy.py]
    dependencies:
    - model-evaluation
```

---

## 14.4 Resource Quotas & Limits

### GPU Resource Management

```yaml
# Custom resource for GPU workloads
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: gpuworkloads.ai.example.com
spec:
  group: ai.example.com
  names:
    kind: GPUWorkload
    plural: gpuworkloads
  scope: Namespaced
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              gpuType:
                type: string
              gpuCount:
                type: integer
              maxRuntime:
                type: string
              priority:
                type: integer
---
# Resource quota for GPU workloads
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ai-training
spec:
  hard:
    requests.nvidia.com/gpu: "32"
    limits.nvidia.com/gpu: "64"
    # GPU hours per month
    count/gpuworkloads.ai.example.com: "100"
```

### Dynamic Resource Allocation

```yaml
# Resource quota with scope selectors
apiVersion: v1
kind: ResourceQuota
metadata:
  name: high-priority-gpu-quota
  namespace: ai-training
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "16"
  scopeSelector:
    matchExpressions:
    - scopeName: PriorityClass
      operator: In
      values:
      - gpu-training-high
```

### Cost Optimization Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                   GPU Cost Optimization                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Strategy 1: GPU Sharing                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Single A100 80GB                                        │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                  │   │
│  │  │MIG 1 │ │MIG 2 │ │MIG 3 │ │MIG 4 │                  │   │
│  │  │20GB  │ │20GB  │ │20GB  │ │20GB  │                  │   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Strategy 2: Time-sharing                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  9AM-5PM: Training Job A (4 GPUs)                       │   │
│  │  5PM-9AM: Inference Job B (4 GPUs)                       │   │
│  │  9AM-5PM: Training Job C (4 GPUs)                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Strategy 3: Spot/Preemptible Instances                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Checkpoint every 30 minutes                            │   │
│  │  Auto-restart on preemption                              │   │
│  │  60-70% cost savings                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 14.5 Multi-tenant AI Platform

### Platform Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Multi-tenant AI Platform                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Tenant Layer                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ Team A   │  │ Team B   │  │ Team C   │  │Team D│  │   │
│  │  │Training  │  │Inference │  │Research  │  │Batch │  │   │
│  │  │Namespace │  │Namespace │  │Namespace │  │NS    │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Platform Layer                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │Kubeflow  │  │Prometheus│  │  Istio   │  │  OPA │  │   │
│  │  │Pipelines │  │Monitoring│  │Service   │  │Policy│  │   │
│  │  │          │  │          │  │Mesh      │  │Engine│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Infrastructure Layer                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │GPU Pool  │  │CPU Pool  │  │Storage   │  │Network│  │   │
│  │  │(V100/A100│  │          │  │(NFS/Ceph)│  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tenant Isolation Implementation

```yaml
# Network Policy for tenant isolation
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tenant-isolation
  namespace: team-a-training
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          team: team-a
    - namespaceSelector:
        matchLabels:
          role: platform
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          team: team-a
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          app: mlflow-server
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - 10.0.0.0/8
  ports:
  - protocol: TCP
    port: 443
  - protocol: TCP
    port: 8080
```

### RBAC for AI Platform

```yaml
# ClusterRole for ML Engineer
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ml-engineer
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "statefulsets", "daemonsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["kubeflow.org"]
  resources: ["pytorchjobs", "tfjobs", "xgboostjobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["batch"]
  resources: ["jobs", "cronjobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]
---
# RoleBinding for team-specific access
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: team-a-ml-engineers
  namespace: team-a-training
subjects:
- kind: Group
  name: team-a-ml-engineers
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: ml-engineer
  apiGroup: rbac.authorization.k8s.io
```

---

## 💡 Case Study: Kubeflow-based Kubernetes AI Platform

### Complete Platform Setup

🟡 Intermediate

```bash
# Step 1: Install Istio for service mesh
kubectl apply -f https://github.com/istio/istio/releases/download/1.18.0/istio-base.yaml
kubectl apply -f https://github.com/istio/istio/releases/download/1.18.0/istiod.yaml

# Step 2: Install Kubeflow
git clone https://github.com/kubeflow/manifests.git
cd manifests
while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 5; done

# Step 3: Verify installation
kubectl get pods -n kubeflow
kubectl get pods -n istio-system
```

### Platform Monitoring Stack

```yaml
# Prometheus configuration for GPU monitoring
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: ai-platform-prometheus
  namespace: monitoring
spec:
  replicas: 2
  retention: 30d
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
  serviceMonitorSelector:
    matchLabels:
      team: ai-platform
  ruleSelector:
    matchLabels:
      team: ai-platform
  storage:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 100Gi
  containers:
  - name: prometheus
    args:
    - --config.file=/etc/prometheus/prometheus.yml
    - --storage.tsdb.path=/prometheus
    - --storage.tsdb.retention.time=30d
    - --web.enable-lifecycle
    ports:
    - containerPort: 9090
    resources:
      requests:
        memory: "2Gi"
        cpu: "1"
---
# GPU monitoring service monitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: nvidia-dcgm-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: nvidia-dcgm-exporter
  endpoints:
  - port: dcgm-exporter
    interval: 30s
    path: /metrics
```

### Cost Tracking Dashboard

```yaml
# Kubecost configuration for GPU cost tracking
apiVersion: v1
kind: ConfigMap
metadata:
  name: kubecost-gpu-config
  namespace: kubecost
data:
  kubecost-gpu-config.json: |
    {
      "gpuCost": 3.50,
      "gpuType": "nvidia-tesla-v100",
      "currency": "USD",
      "allocationModel": "gpu",
      "costAllocation": {
        "training": {
          "priority": "high",
          "dailyBudget": 500
        },
        "inference": {
          "priority": "medium",
          "dailyBudget": 200
        },
        "research": {
          "priority": "low",
          "dailyBudget": 100
        }
      }
    }
```

---

## 📝 Exercises

### Exercise 14.1: GPU Scheduling
Configure a Kubernetes cluster to schedule GPU workloads with the following requirements:
1. 4-node GPU cluster with NVIDIA V100 GPUs
2. MIG enabled on 2 nodes for inference workloads
3. Time-sharing for training workloads during off-peak hours
4. Cost tracking for GPU usage per team

### Exercise 14.2: Multi-tenant Platform
Design and implement a multi-tenant AI platform with:
1. Namespace isolation for 3 teams
2. Network policies preventing cross-tenant communication
3. Resource quotas per team
4. RBAC with team-specific permissions
5. Monitoring and cost tracking per tenant

### Exercise 14.3: Training Pipeline
Create a Kubeflow Pipeline that:
1. Preprocesses data from a shared storage
2. Trains a distributed PyTorch model
3. Evaluates model performance
4. Deploys the model if accuracy > 90%
5. Sends notifications on completion

---

## ⚠️ Warnings

1. **GPU Memory Leaks**: Always set memory limits for GPU workloads. Unbounded GPU memory usage can crash the entire node.
2. **Node Pressure**: Monitor GPU temperature and utilization. Overheating can cause thermal throttling and performance degradation.
3. **Cost Overruns**: Implement budget alerts. GPU costs can escalate quickly with uncontrolled usage.
4. **Security**: Never run GPU workloads as root. Use security contexts to limit container capabilities.

---

## Summary

This chapter covered the fundamentals of running AI workloads on Kubernetes, including GPU scheduling, AI-specific operators, resource management, and multi-tenant platform design. Key takeaways:

1. Kubernetes provides the necessary abstractions for managing heterogeneous AI resources
2. NVIDIA GPU Operator simplifies GPU management across clusters
3. Kubeflow operators enable end-to-end ML lifecycle management
4. Multi-tenant platforms require careful isolation and resource management
5. Cost optimization is critical for sustainable AI operations

Next, we'll explore distributed computing architectures for scaling AI workloads beyond single clusters.
