# 第14章：Kubernetes 上的 AI 工作负载

🟢 入门 | 🟡 中级 | 🔴 高级 | ⚫ 管理者

---

## 14.1 Kubernetes 基础回顾

### 为什么选择 Kubernetes 运行 AI 工作负载

Kubernetes 已成为生产环境中编排 AI 工作负载的事实标准。理解这一选择需要分析 AI 工作负载的独特特征：

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI 工作负载特征                               │
├─────────────────────────────────────────────────────────────────┤
│  • GPU 密集型计算                                               │
│  • 可变的资源需求（训练 vs 推理）                                │
│  • 分布式处理需求                                               │
│  • 需要可复现的环境                                              │
│  • 带 SLA 要求的批处理                                          │
│  • 跨团队的多租户支持                                           │
└─────────────────────────────────────────────────────────────────┘
```

### AI 的核心 Kubernetes 概念

📌 **关键概念**：Kubernetes 提供了管理异构计算资源（CPU、GPU、TPU、内存）所需的抽象层。

```yaml
# 训练任务的 Pod 定义
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

### Kubernetes AI 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes AI 架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    控制平面                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │   API    │  │  调度器   │  │ 控制器    │  │  etcd│  │   │
│  │  │  服务器  │  │          │  │  管理器   │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     工作节点                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │  GPU 节点 1  │  │  GPU 节点 2  │  │  CPU 节点 1  │  │   │
│  │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │  │   │
│  │  │  │4xV100  │  │  │  │4xA100  │  │  │  │64 CPU  │  │  │   │
│  │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### AI 团队的命名空间策略

```yaml
# AI 工作负载的命名空间配置
apiVersion: v1
kind: Namespace
metadata:
  name: ai-training
  labels:
    team: ml-engineering
    environment: production
    istio-injection: enabled
---
# 命名空间的资源配额
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
# 限制范围防止资源浪费
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

## 14.2 GPU 调度与管理

### NVIDIA GPU Operator

🟢 入门

NVIDIA GPU Operator 自动化管理 GPU 驱动、CUDA 工具包和设备插件：

```bash
# 使用 Helm 安装 GPU Operator
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

# 使用默认设置安装
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true

# 验证安装
kubectl get pods -n gpu-operator
```

### GPU 资源发现

```yaml
# 检查节点上的 GPU 可用性
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

### 使用 MIG 的 GPU 共享

🔴 高级

多实例 GPU（MIG）允许将单个 A100 GPU 分割为多个隔离的实例：

```bash
# 在 NVIDIA A100 上启用 MIG
nvidia-smi -i 0 -mig 1

# 创建 MIG 实例
nvidia-smi -i 0 -cgi 19,19,19,19 -C

# 验证 MIG 实例
nvidia-smi --query-gpu=mig.mode.current,gpu.bus_id --format=csv
```

### GPU 调度策略

```yaml
# 基于优先级的 GPU 调度
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: gpu-training-high
value: 1000000
globalDefault: false
description: "GPU 训练任务的高优先级"
---
# 多 GPU 任务的拓扑感知调度
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

## 14.3 AI 专用 Operator

### Kubeflow Operators

🟡 中级

Kubeflow 提供了一套用于 ML 生命周期管理的 operator：

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

### Volcano - AI 批处理调度

```yaml
# Volcano Job 用于 gang 调度
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

---

## 14.4 资源配额与限制

### GPU 资源管理

```yaml
# GPU 工作负载的自定义资源
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
# GPU 工作负载的资源配额
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ai-training
spec:
  hard:
    requests.nvidia.com/gpu: "32"
    limits.nvidia.com/gpu: "64"
    count/gpuworkloads.ai.example.com: "100"
```

### 成本优化策略

```
┌─────────────────────────────────────────────────────────────────┐
│                   GPU 成本优化策略                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  策略 1：GPU 共享                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  单个 A100 80GB                                          │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                  │   │
│  │  │MIG 1 │ │MIG 2 │ │MIG 3 │ │MIG 4 │                  │   │
│  │  │20GB  │ │20GB  │ │20GB  │ │20GB  │                  │   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  策略 2：时间共享                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  9AM-5PM: 训练任务 A（4 GPU）                            │   │
│  │  5PM-9AM: 推理任务 B（4 GPU）                            │   │
│  │  9AM-5PM: 训练任务 C（4 GPU）                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  策略 3：Spot/可抢占实例                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  每 30 分钟检查点                                         │   │
│  │  被抢占时自动重启                                          │   │
│  │  60-70% 成本节省                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 14.5 多租户 AI 平台

### 平台架构

```
┌─────────────────────────────────────────────────────────────────┐
│                  多租户 AI 平台                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   租户层                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ 团队 A   │  │ 团队 B   │  │ 团队 C   │  │团队 D│  │   │
│  │  │训练命名  │  │推理命名  │  │研究命名  │  │批处理│  │   │
│  │  │空间      │  │空间      │  │空间      │  │命名  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  平台层                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │Kubeflow  │  │Prometheus│  │  Istio   │  │  OPA │  │   │
│  │  │Pipelines │  │监控      │  │服务网格  │  │策略  │  │   │
│  │  │          │  │          │  │          │  │引擎  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 基础设施层                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │GPU 池    │  │CPU 池    │  │存储      │  │网络  │  │   │
│  │  │(V100/A100│  │          │  │(NFS/Ceph)│  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 租户隔离实现

```yaml
# 租户隔离的网络策略
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

### AI 平台的 RBAC

```yaml
# ML 工程师的 ClusterRole
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
# 团队特定访问的 RoleBinding
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

## 💡 案例研究：基于 Kubeflow 的 Kubernetes AI 平台

### 完整平台搭建

🟡 中级

```bash
# 步骤 1：安装 Istio 作为服务网格
kubectl apply -f https://github.com/istio/istio/releases/download/1.18.0/istio-base.yaml
kubectl apply -f https://github.com/istio/istio/releases/download/1.18.0/istiod.yaml

# 步骤 2：安装 Kubeflow
git clone https://github.com/kubeflow/manifests.git
cd manifests
while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 5; done

# 步骤 3：验证安装
kubectl get pods -n kubeflow
kubectl get pods -n istio-system
```

### 平台监控栈

```yaml
# GPU 监控的 Prometheus 配置
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
# GPU 监控的服务监控器
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

---

## 📝 练习

### 练习 14.1：GPU 调度
配置 Kubernetes 集群来调度 GPU 工作负载，要求：
1. 4 节点 GPU 集群，配备 NVIDIA V100 GPU
2. 在 2 个节点上启用 MIG 用于推理工作负载
3. 在非高峰时段进行训练工作负载的时间共享
4. 按团队跟踪 GPU 使用成本

### 练习 14.2：多租户平台
设计并实现多租户 AI 平台，要求：
1. 3 个团队的命名空间隔离
2. 防止跨租户通信的网络策略
3. 每个团队的资源配额
4. 具有团队特定权限的 RBAC
5. 每个租户的监控和成本跟踪

---

## ⚠️ 警告

1. **GPU 内存泄漏**：始终为 GPU 工作负载设置内存限制。无界的 GPU 内存使用可能崩溃整个节点。
2. **节点压力**：监控 GPU 温度和利用率。过热可能导致热节流和性能下降。
3. **成本超支**：实施预算警报。不受控制的使用可能导致 GPU 成本快速上升。
4. **安全性**：切勿以 root 身份运行 GPU 工作负载。使用安全上下文限制容器能力。

---

## 本章小结

本章介绍了在 Kubernetes 上运行 AI 工作负载的基础知识，包括 GPU 调度、AI 专用 operator、资源管理和多租户平台设计。关键要点：

1. Kubernetes 提供了管理异构 AI 资源所需的抽象
2. NVIDIA GPU Operator 简化了集群中的 GPU 管理
3. Kubeflow operator 实现端到端的 ML 生命周期管理
4. 多租户平台需要仔细的隔离和资源管理
5. 成本优化对于可持续的 AI 运营至关重要

下一章我们将探讨用于扩展 AI 工作负载的分布式计算架构。
