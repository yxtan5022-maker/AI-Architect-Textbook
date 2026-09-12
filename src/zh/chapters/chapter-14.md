# 第14章：Kubernetes上的AI工作负载

## 学习目标

完成本章学习后，你将能够：

1. 解释Kubernetes如何使用NVIDIA GPU Operator编排GPU加速的AI工作负载
2. 在Kubernetes集群上设计和部署Kubeflow流水线
3. 实现共享AI基础设施的多租户隔离模式
4. 诊断和缓解生产集群中的GPU资源争用问题
5. 评估Kubernetes何时适用于AI工作负载，以及何时应选择替代编排平台

---

## 14.1 引言：为什么选择Kubernetes运行AI？

在生产环境中运行AI工作负载不仅仅是使用一台强大的机器。团队需要弹性伸缩、资源隔离、可重现性和多用户访问。Kubernetes已成为AI基础设施的事实编排层，因为它通过统一的API满足了这些需求。

然而，Kubernetes最初是为无状态Web服务设计的。在为短生命周期容器设计的系统上运行GPU密集型、内存消耗大且通常长时间运行的训练任务，会引入真实的架构张力。本章探讨如何使用生产级工具来协调这些矛盾。

> **📌 真实数据框**
> Kubeflow拥有**33,100+ GitHub星标**，是**CNCF毕业项目**，这意味着它已在Kubernetes生态系统中展示了成熟度和生产采用度（kubeflow.org，2026）。CNCF毕业状态要求通过安全审计、治理审查，并展示大规模的实际部署。

---

## 14.2 Kubernetes上的GPU调度机制

### 14.2.1 NVIDIA GPU Operator

NVIDIA GPU Operator是在Kubernetes上使GPU可用的标准机制。它自动化部署和管理集群中提供GPU所需的所有NVIDIA软件组件。

**GPU Operator管理的核心组件：**

| 组件 | 用途 |
|------|------|
| NVIDIA驱动程序 | 内核级GPU驱动 |
| NVIDIA容器工具包 | 启用容器内的GPU访问 |
| 设备插件 | 将GPU作为可调度资源暴露给Kubernetes |
| GPU功能发现 | 自动为节点标记GPU能力 |
| NVIDIA导出器 | 将GPU指标暴露给Prometheus |

**GPU调度的实际工作原理：**

1. GPU Operator在每个GPU节点上安装**设备插件**
2. 设备插件调用NVIDIA管理库（NVML）检测可用GPU
3. GPU作为可扩展资源（`nvidia.com/gpu`）通告给Kubernetes调度器
4. 当Pod请求`nvidia.com/gpu: 1`时，调度器找到有空闲GPU的节点
5. 容器工具包将GPU挂载到容器的cgroup中

**关键细微差别：** Kubernetes GPU调度是**装箱式的，非时间共享的**。如果Pod请求1个GPU，它在整个Pod生命周期内独占该物理GPU。标准Kubernetes中没有原生GPU共享——请求1个GPU的Pod在有8个GPU的节点上会阻止其他Pod使用该特定GPU，即使该工作负载仅需要其计算容量的20%。

### 14.2.2 GPU资源请求和限制

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

**重要区别：**

- `requests`决定调度——调度器使用此值找到合适的节点
- `limits`决定强制执行——容器超出限制将被终止
- 对于GPU，**没有超量分配**——不能请求超过物理存在的GPU数量
- A100/H100上的MIG（多实例GPU）允许将单个GPU分区为最多7个隔离实例

### 14.2.3 Kubernetes上的多实例GPU（MIG）

NVIDIA的MIG技术（在A100、H100和更新的数据中心GPU上可用）允许将单个GPU分区为最多7个隔离的GPU实例。每个实例拥有专用的内存、计算核心和带宽。

**A100 80GB的MIG配置文件：**

| 配置文件 | 内存 | 计算 |
|---------|------|------|
| 1g.10gb | 10 GB | 1/7 SM |
| 2g.20gb | 20 GB | 2/7 SM |
| 3g.40gb | 40 GB | 3/7 SM |
| 4g.40gb | 40 GB | 4/7 SM |
| 7g.80gb | 80 GB | 7/7 SM（完整GPU） |

GPU Operator的GPU功能发现会自动检测启用了MIG的GPU，并创建拓扑标签，允许Kubernetes调度器定位特定的MIG配置文件。

---

## 14.3 Kubernetes上的Kubeflow架构

Kubeflow不是一个单一应用程序——它是一组松耦合的组件，共同构成端到端的ML平台。理解其架构需要了解每个组件的角色。

### 14.3.1 核心组件

| 组件 | 角色 | K8s资源 |
|------|------|---------|
| **Kubeflow Pipelines** | 工作流编排（基于DAG） | Argo Workflows + 自定义控制器 |
| **Training Operators** | 管理分布式训练任务 | 自定义资源（TFJob、PyTorchJob等） |
| **Katib** | 超参数调优 | 自定义资源 + 指标收集 |
| **KServe** | 模型服务（推理） | Knative + Istio（可选） |
| **Notebooks** | Jupyter/Lab环境 | 带PVC的StatefulSet |
| **Central Dashboard** | 统一UI | Deployment + ConfigMap |

### 14.3.2 Kubeflow Pipelines架构

Kubeflow Pipelines运行在**Argo Workflows**之上，后者是Kubernetes原生的工作流引擎。创建流水线时：

1. 流水线DSL编译为YAML工作流定义
2. KFP后端持久化工作流和元数据
3. Argo Workflows为每个流水线步骤创建Pod
4. 每个步骤在具有自己的镜像和资源需求的隔离容器中运行
5. 步骤通过工件（存储在MinIO/S3中）和参数传递连接

**多租户模型：**

Kubeflow使用**配置文件**抽象。每个配置文件映射到一个Kubernetes命名空间，具有：
- 自己的资源配额
- 自己的服务账户
- 网络策略将其与其他配置文件隔离
- 每个配置文件的持久卷

```bash
# 创建新配置文件
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

### 14.3.3 用于分布式训练的Training Operators

Kubeflow的Training Operators定义了抽象分布式训练框架的自定义资源（CRD）：

| CRD | 框架 | 分布策略 |
|-----|------|---------|
| `PyTorchJob` | PyTorch | 使用世界大小的`torchrun` |
| `TFJob` | TensorFlow | 参数服务器 + 工作者 |
| `XGBoostJob` | XGBoost | AllReduce |
| `MPIJob` | 基于MPI | 跨Pod的`mpirun` |
| `PaddleJob` | PaddlePaddle | AllReduce |

**示例：分布式PyTorch训练**

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

当应用此CRD时，Training Operator创建一个主Pod和3个工作Pod，配置PyTorch分布式运行时环境变量（MASTER_ADDR、MASTER_PORT、WORLD_SIZE、RANK），并监控任务健康状况。

---

## 14.4 多租户隔离模式

在共享基础设施上为多个团队运行AI工作负载需要在每一层进行仔细的隔离。

### 14.4.1 命名空间级隔离

每个团队或项目获得自己的命名空间，具有：

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

### 14.4.2 网络隔离

网络策略防止一个命名空间中的Pod与另一个命名空间中的Pod通信：

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

### 14.4.3 GPU隔离

对于命名空间配额之外的GPU隔离，使用**设备插件**结合**节点亲和性**或**MIG**：

**选项1：节点亲和性（粗粒度）**
```yaml
nodeSelector:
  gpu-pool: team-alpha
```

**选项2：MIG（细粒度）**
```yaml
resources:
  limits:
    nvidia.com/mig-3g.40gb: 2
```

**选项3：时间切片（共享）**
GPU Operator支持时间切片，多个Pod通过交替执行共享单个GPU。这降低了隔离性但提高了利用率。

### 14.4.4 存储隔离

每个租户通过StorageClass和每个命名空间的PVC获得隔离的持久存储。常见模式：为训练任务配置快速SSD存储，为工件配置更便宜的对象存储。

---

## 14.5 案例研究：Google如何在Kubernetes上运行AI

Google的内部ML平台（构建在Kubernetes上）服务于数千名研究者，运行从语言理解到蛋白质折叠的模型。虽然Google内部不使用Kubeflow（他们使用Borg，Kubernetes的前身），但架构模式可直接转移。

**关键架构决策：**

| 决策 | 实施方式 | 理由 |
|------|---------|------|
| 集中资源管理 | Borg/K8s + 自定义调度器 | 防止资源囤积 |
| 可抢占VM用于训练 | Spot/可抢占实例 | 降低60-70%成本 |
| 层级配额 | 组织 → 团队 → 项目 | 数千用户间公平共享 |
| 自定义调度策略 | 装箱 + GPU亲和性 | 最小化GPU碎片 |
| 自动检查点 | 基于GCS的分布式检查点 | 长时间任务的容错 |

**规模指标（近似值，来自已发表研究）：**

- 每月数百万个训练任务
- 数十万个GPU节点
- 平均GPU利用率：60-80%（通过调度改进从40%提升）
- 平均训练任务持续时间：2-8小时
- 最大单个训练任务：数千个TPU运行数周

**经验教训：**

1. **GPU利用率是主要的成本驱动因素。** 以40% GPU利用率运行的集群每有效GPU小时的成本是以80%运行的2.5倍。
2. **可抢占性至关重要。** 没有它，团队会过度配置以保证可用性。
3. **多租户需要每一层的配额**——命名空间、节点池和全局。
4. **检查点不是可选的**——有了可抢占性，任务必须能够从任何检查点恢复。

---

## 14.6 战争故事：GPU内存泄漏导致整个集群崩溃

**公司：** 中型自动驾驶初创公司，200+ GPU集群

**问题：** 一个训练任务包含PyTorch bug，其中CUDA张量在Python `for`循环内创建但未显式释放。这些张量存储在一个从未被清除的列表中，导致每次迭代GPU内存增长约200MB。

**时间线：**

| 时间 | 事件 |
|------|------|
| 0:00 | 训练任务启动，请求8个A100 GPU |
| 2:00 | 工作节点0上的GPU内存达到90%（泄漏至~72GB） |
| 3:00 | OOM终止程序激活，终止训练进程 |
| 3:01 | 被终止的进程留下孤立的CUDA上下文 |
| 3:05 | 工作节点0上的NVIDIA设备插件失去与GPU 0的连接 |
| 3:06 | 设备插件将工作节点0上的GPU 0-7标记为不健康 |
| 3:07 | Kubernetes驱逐工作节点0上的所有Pod |
| 3:10 | 工作节点0上的其余7个GPU被排除在调度之外 |
| 3:15 | 其他任务开始排队，等待释放的GPU |
| 3:30 | 其他任务达到自己的OOM限制，产生级联故障 |

**根本原因：** 工作节点0上的NVIDIA设备插件检测到GPU无响应，并将整个节点的GPU资源报告为不可用。Kubernetes随后将该节点从调度池中移除。孤立的CUDA上下文未被清理，因为容器运行时未正确向CUDA驱动程序传播SIGTERM。

**应用的修复：**

1. **预任务验证：** 添加了内存预检查，在启动实际训练任务之前运行小的分配测试
2. **使用Kubernetes设置资源限制：** 使用MIG配置文件设置GPU内存限制，使每个任务隔离
3. **GPU健康监控：** 部署了DCGM（数据中心GPU管理器）将GPU健康指标导出到Prometheus
4. **自动修复：** 创建了一个控制器，检测GPU无响应并触发GPU重置（nvidia-smi --gpu-reset），无需驱逐整个节点
5. **代码审查关卡：** 添加了CI检查，标记在没有显式删除的情况下在循环内分配CUDA张量

**结果：** GPU相关集群停机时间从每月12小时降至不到30分钟。

---

## 14.7 何时使用/何时不使用Kubernetes运行AI

### 何时使用Kubernetes运行AI

| 场景 | K8s适用的原因 |
|------|-------------|
| 多团队共享GPU集群 | 命名空间隔离、资源配额、RBAC |
| 混合工作负载（训练+服务） | 统一编排、不同资源配置文件 |
| 自动扩缩需求 | 集群自动扩缩器、节点池管理 |
| 混合云/多云 | 跨环境的一致API |
| 合规性需求 | 审计日志、网络策略、RBAC |
| 大规模分布式训练 | Training Operators处理拓扑 |

### 何时不使用Kubernetes运行AI

| 场景 | K8s不适合的原因 | 替代方案 |
|------|----------------|---------|
| 单团队，< 8个GPU | 运营开销超过收益 | Docker Compose或单节点 |
| 仅交互式笔记本工作 | StatefulSet复杂性，JupyterHub更简单 | VM上的JupyterHub |
| 实时低延迟服务 | K8s网络增加1-3ms延迟 | 专用推理服务器 |
| 原型设计/实验 | Kubernetes YAML开销减慢迭代 | 本地Docker + MLflow |
| 小模型（< 100MB） | GPU调度开销超过模型运行时间 | 仅CPU容器 |
| 预算受限（无SRE团队） | K8s需要专门的运营专业知识 | 托管服务（SageMaker、Vertex AI） |

---

## 14.8 本章小结

- **NVIDIA GPU Operator**自动化Kubernetes上的GPU配置，但GPU调度是装箱式且不可超量分配的——Pod获得对物理GPU的独占访问
- **Kubeflow**是CNCF毕业项目（33.1K+星标），在Kubernetes上提供Pipelines、Training Operators、Katib、KServe和Notebooks
- **多租户**需要在每一层进行隔离：命名空间、网络策略、GPU设备插件、存储类和资源配额
- **MIG**技术允许将A100/H100 GPU分区为隔离实例，实现更细粒度的资源共享
- Kubernetes对AI基础设施很强大，但引入了运营复杂性，必须通过规模和多租户需求来证明其合理性

---

## 讨论题

1. 一个初创公司有4名数据科学家共享4个GPU。他们目前使用共享的Jupyter服务器。CTO想要部署Kubeflow Pipelines进行实验跟踪。这是否合理？你会建议什么替代方案？

2. GPU时间切片与MIG在隔离性、性能和运营复杂性方面有何不同？在什么场景下你会选择其中一个？

3. 一个团队运行需要6-12小时的分布式PyTorch训练任务。集群使用可抢占实例。设计一个检查点策略，在利用可抢占性的成本节约与训练进度保护之间取得平衡。

4. NVIDIA设备插件报告GPU不健康，导致节点上的所有Pod被驱逐。你将如何设计更具弹性的修复策略？

5. 比较在Kubernetes上运行KServe与在专用推理平台（如NVIDIA Triton）上部署相同模型的运营复杂性。哪些因素驱动决策？

---

## 练习

**练习1：** 部署一个Kubeflow Training Operator，在2个节点上运行分布式PyTorch训练任务，每个节点2个GPU。验证主节点和工作Pod可以通过NCCL后端通信。

**练习2：** 创建一个命名空间，资源配额将GPU访问限制为4个GPU。部署3个各请求2个GPU的Pod。观察哪个Pod被挂起以及原因。然后部署一个请求1个GPU的第4个Pod，确认其调度成功。

**练习3：** 在集群中设置DCGM导出器，配置Prometheus抓取GPU指标，并创建Grafana仪表板显示所有节点的GPU利用率、内存使用和温度。

---

## 参考文献

- NVIDIA GPU Operator文档：https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
- Kubeflow官方文档：https://www.kubeflow.org/docs/
- Kubeflow GitHub仓库：https://github.com/kubeflow/kubeflow
- Kubernetes设备插件：https://kubernetes.io/docs/concepts/extend-kubernetes-compute-storage-plugins/device-plugins/
- NVIDIA多实例GPU（MIG）：https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- CNCF Kubeflow毕业公告：https://www.cncf.io/announcements/
- Argo Workflows：https://argoproj.github.io/argo-workflows/
- NVIDIA DCGM Exporter：https://github.com/NVIDIA/dcgm-exporter
