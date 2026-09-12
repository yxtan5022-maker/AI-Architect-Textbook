# 第7章：模型训练架构

> **第三部分：MLOps 架构**

**学习目标：**
- 设计可扩展且可复现的训练环境
- 实现跨多节点和 GPU 的分布式训练
- 构建超参数优化管道
- 建立实验管理和跟踪系统
- 优化训练资源利用率和成本

---

## 7.1 训练环境设计

### 7.1.1 训练环境的挑战

🟢 **初级**

训练环境不仅仅是带有某些库的 Jupyter notebook。它是硬件、软件、数据访问和配置的精心协调组合，确保可复现性、可扩展性和效率。

```
训练环境栈：
┌─────────────────────────────────────────────────┐
│                  应用层                           │
│         （训练脚本、Notebook）                     │
├─────────────────────────────────────────────────┤
│                  框架层                           │
│      （PyTorch、TensorFlow、JAX、XGBoost）       │
├─────────────────────────────────────────────────┤
│                  运行时层                         │
│     （Python、CUDA、cuDNN、容器运行时）           │
├─────────────────────────────────────────────────┤
│                  基础设施层                       │
│     （GPU/TPU、存储、网络、集群）                  │
├─────────────────────────────────────────────────┤
│                  编排层                           │
│     （Kubernetes、Kubeflow、Slurm）              │
└─────────────────────────────────────────────────┘
```

### 7.1.2 环境可复现性

🟡 **中级**

可复现性是可靠 ML 训练的基石。没有它，你就无法调试、比较或信任你的模型。

**可复现性的关键组件：**

```python
# 环境规格文件

# requirements.txt (Python)
torch==2.1.0+cu118
transformers==4.35.0
datasets==2.14.0
mlflow==2.8.0
pandas==2.1.0
numpy==1.25.0

# environment.yaml (Conda)
name: training-env
channels:
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.1.0
  - cudatoolkit=11.8
  - pip:
    - transformers==4.35.0
    - mlflow==2.8.0

# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /workspace
COPY . /workspace
```

### 7.1.3 基于容器的训练环境

🔴 **高级**

```
基于容器的训练架构：
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes 集群                          │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │  训练 Pod       │  │  训练 Pod       │                 │
│  │  ┌───────────┐  │  │  ┌───────────┐  │                 │
│  │  │ 训练      │  │  │  │ 训练      │  │                 │
│  │  │ 脚本      │  │  │  │ 脚本      │  │                 │
│  │  └───────────┘  │  │  └───────────┘  │                 │
│  │  ┌───────────┐  │  │  ┌───────────┐  │                 │
│  │  │ GPU       │  │  │  │ GPU       │  │                 │
│  │  │ 运行时    │  │  │  │ 运行时    │  │                 │
│  │  └───────────┘  │  │  └───────────┘  │                 │
│  │  ┌───────────┐  │  │  ┌───────────┐  │                 │
│  │  │ 数据      │  │  │  │ 数据      │  │                 │
│  │  │ 卷        │  │  │  │ 卷        │  │                 │
│  │  └───────────┘  │  │  └───────────┘  │                 │
│  └─────────────────┘  └─────────────────┘                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              共享存储（NFS/S3）                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐    │   │
│  │  │ 数据集  │  │ 检查点  │  │ 模型产物        │    │   │
│  │  └─────────┘  └─────────┘  └─────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.1.4 GPU 资源管理

🔴 **高级**

```yaml
# Kubernetes Pod 规格用于 GPU 训练
apiVersion: v1
kind: Pod
metadata:
  name: training-job
  labels:
    app: training
spec:
  containers:
  - name: trainer
    image: registry.example.com/trainer:latest
    resources:
      requests:
        memory: "16Gi"
        cpu: "4"
        nvidia.com/gpu: "2"  # 请求 2 个 GPU
      limits:
        memory: "32Gi"
        cpu: "8"
        nvidia.com/gpu: "2"
    volumeMounts:
    - name: data-volume
      mountPath: /data
    - name: model-output
      mountPath: /output
    env:
    - name: NVIDIA_VISIBLE_DEVICES
      value: "all"
    - name: NCCL_DEBUG
      value: "INFO"
  volumes:
  - name: data-volume
    persistentVolumeClaim:
      claimName: training-data-pvc
  - name: model-output
    persistentVolumeClaim:
      claimName: model-output-pvc
  nodeSelector:
    accelerator: nvidia-tesla-v100
```

---

## 7.2 分布式训练架构

### 7.2.1 为什么需要分布式训练？

🟢 **初级**

分布式训练将训练工作负载分配到多个设备（GPU/TPU）或多台机器上，以：

1. **减少训练时间**：更快地训练大型模型
2. **处理大型数据集**：处理无法装入内存的数据
3. **支持更大模型**：训练需要更多参数的模型
4. **提高利用率**：更好地利用可用硬件

```
单 GPU 训练：
┌──────────────┐
│    GPU 0     │
│  完整模型    │──── 数小时训练
│  完整数据    │
└──────────────┘

分布式训练：
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    GPU 0     │    │    GPU 1     │    │    GPU 2     │
│  模型部分    │    │  模型部分    │    │  模型部分    │
│  数据分片    │    │  数据分片    │    │  数据分片    │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    数分钟训练
```

### 7.2.2 数据并行

🟡 **中级**

数据并行在每个设备上复制模型，并将数据分配到各个设备。

```
数据并行：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  全局数据集                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  分片 0  │  分片 1  │  分片 2  │  分片 3            │   │
│  └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────────┘   │
│        │           │           │           │              │
│        ▼           ▼           ▼           ▼              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │  GPU 0   │ │  GPU 1   │ │  GPU 2   │ │  GPU 3   │    │
│  │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │    │
│  │ │模型  │ │ │ │模型  │ │ │ │模型  │ │ │ │模型  │ │    │
│  │ │副本  │ │ │ │副本  │ │ │ │副本  │ │ │ │副本  │ │    │
│  │ └──────┘ │ │ └──────┘ │ │ └──────┘ │ │ └──────┘ │    │
│  │ 前向传播  │ │ 前向传播  │ │ 前向传播  │ │ 前向传播  │    │
│  │ 反向传播  │ │ 反向传播  │ │ 反向传播  │ │ 反向传播  │    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘    │
│       │            │            │            │            │
│       └────────────┴────────────┴────────────┘            │
│                          │                                │
│                          ▼                                │
│              ┌──────────────────────┐                    │
│              │  梯度 AllReduce      │                    │
│              │  （平均梯度）         │                    │
│              └──────────┬───────────┘                    │
│                         │                                │
│                         ▼                                │
│              ┌──────────────────────┐                    │
│              │  更新所有模型         │                    │
│              └──────────────────────┘                    │
│                                                          │
└─────────────────────────────────────────────────────────────┘
```

**PyTorch DDP 实现：**

```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)
    
    # 创建模型
    model = nn.Linear(100, 10).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 创建分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        sampler.set_epoch(epoch)  # 对打乱很重要
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    
    dist.destroy_process_group()

# 使用以下命令启动：torchrun --nproc_per_node=4 train.py
```

### 7.2.3 模型并行

🔴 **高级**

模型并行将模型本身分配到多个设备上，当模型太大无法装入单个 GPU 时使用。

```
模型并行（流水线）：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  模型层：L0 → L1 → L2 → L3 → L4 → L5 → L6 → L7          │
│                                                             │
│  GPU 0：L0 → L1 → L2 → L3                                  │
│  GPU 1：L4 → L5 → L6 → L7                                  │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  GPU 0   │───►│  GPU 1   │───►│  输出    │             │
│  │ L0-L3    │    │ L4-L7    │    │  结果    │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│                                                             │
│  微批次流水线：                                               │
│  时间 ──────────────────────────────────────────────────►   │
│  GPU 0：[MB1][MB2][MB3][MB4]                               │
│  GPU 1：     [MB1][MB2][MB3][MB4]                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2.4 张量并行

🔴 **高级**

张量并行将单个层分配到多个设备：

```
张量并行（列并行）：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  线性层：Y = XW                                             │
│                                                             │
│  将 W 分割为 W1、W2（列分区）                                │
│                                                             │
│  GPU 0：Y1 = X * W1  ──┐                                   │
│                         ├──► 拼接 ──► Y                     │
│  GPU 1：Y2 = X * W2  ──┘                                   │
│                                                             │
│  减少每 GPU 的内存占用，同时保持完整容量                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2.5 使用 Kubeflow 进行分布式训练

🟡 **中级**

Kubeflow 通过 TFJob 和 PyTorchJob 操作符提供内置的分布式训练支持：

```yaml
# Kubeflow PyTorchJob 用于分布式训练
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-dist-training
  namespace: kubeflow
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: registry.example.com/trainer:latest
            command:
            - python
            args:
            - train.py
            - --epochs=10
            - --batch-size=32
            resources:
              limits:
                nvidia.com/gpu: 1
          volumes:
          - name: data-volume
            persistentVolumeClaim:
              claimName: training-data-pvc
    Worker:
      replicas: 3  # 3 个 worker + 1 个 master = 总共 4 个 GPU
      template:
        spec:
          containers:
          - name: pytorch
            image: registry.example.com/trainer:latest
            command:
            - python
            args:
            - train.py
            - --epochs=10
            - --batch-size=32
            resources:
              limits:
                nvidia.com/gpu: 1
          volumes:
          - name: data-volume
            persistentVolumeClaim:
              claimName: training-data-pvc
```

```bash
# 应用训练任务
kubectl apply -f pytorchjob.yaml

# 监控训练进度
kubectl logs -f pytorch-dist-training-master-0 -n kubeflow

# 查看任务状态
kubectl get pytorchjob pytorch-dist-training -n kubeflow -o yaml
```

---

## 7.3 超参数优化架构

### 7.3.1 超参数的挑战

🟢 **初级**

超参数是训练过程中不学习但显著影响模型性能的配置设置。寻找最优超参数既昂贵又耗时。

```
常见超参数：
┌─────────────────────────────────────────────────────────────┐
│  模型超参数：                                                │
│  ├── 学习率：0.001, 0.01, 0.1                               │
│  ├── 批量大小：16, 32, 64, 128                              │
│  ├── 层数：4, 6, 8, 12                                      │
│  ├── 隐藏层大小：256, 512, 1024, 2048                       │
│  ├── Dropout 率：0.1, 0.2, 0.3                              │
│  └── 权重衰减：0.0001, 0.001, 0.01                          │
│                                                             │
│  训练超参数：                                                │
│  ├── 优化器：SGD, Adam, AdamW                               │
│  ├── 调度器：Cosine, Linear, StepLR                         │
│  ├── 预热步数：0, 100, 1000                                 │
│  └── 梯度裁剪：0, 1.0, 5.0                                 │
│                                                             │
│  搜索空间：4 × 4 × 4 × 4 × 3 × 3 × 3 × 3 × 4 × 4         │
│             = 82,944 种可能配置                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3.2 HPO 策略

🟡 **中级**

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **网格搜索** | 尝试所有组合 | 全面 | 指数级昂贵 |
| **随机搜索** | 随机采样 | 简单，通常有效 | 不从过去学习 |
| **贝叶斯优化** | 基于模型的搜索 | 智能，高效 | 实现复杂 |
| **Hyperband** | 早停 + 随机 | 资源高效 | 可能错过好配置 |
| **BOHB** | 贝叶斯 + Hyperband | 两者兼优 | 最复杂 |

```
搜索策略对比：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  网格搜索：                                                   │
│  ┌───┬───┬───┬───┐                                        │
│  │ ● │ ● │ ● │ ● │  ● = 已训练模型                          │
│  ├───┼───┼───┼───┤  （所有组合）                            │
│  │ ● │ ● │ ● │ ● │                                        │
│  ├───┼───┼───┼───┤                                        │
│  │ ● │ ● │ ● │ ● │  总计：12 个模型                         │
│  └───┴───┴───┴───┘                                        │
│                                                             │
│  随机搜索：                                                   │
│  ┌───┬───┬───┬───┐                                        │
│  │   │ ● │   │   │  ● = 随机选择                            │
│  ├───┼───┼───┼───┤  （较少的总模型数）                       │
│  │ ● │   │   │ ● │                                        │
│  ├───┼───┼───┼───┤                                        │
│  │   │   │ ● │   │  总计：4 个模型                           │
│  └───┴───┴───┴───┘  （但覆盖良好）                          │
│                                                             │
│  贝叶斯：                                                     │
│  ┌───┬───┬───┬───┐                                        │
│  │   │   │ ● │   │  ● = 已训练模型                          │
│  ├───┼───┼───┼───┤  ○ = 预测有前景                          │
│  │   │ ○ │ ○ │   │  基于过去结果                            │
│  ├───┼───┼───┼───┤                                        │
│  │   │ ● │   │   │  总计：3 个模型                           │
│  └───┴───┴───┴───┘  （最高效）                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3.3 贝叶斯优化实现

🔴 **高级**

```python
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class HyperparameterOptimizer:
    def __init__(self, train_dataset, val_dataset, device='cuda'):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
    
    def create_model(self, trial):
        # 建议超参数
        n_layers = trial.suggest_int('n_layers', 2, 8)
        hidden_size = trial.suggest_categorical('hidden_size', [256, 512, 1024, 2048])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        layers = []
        in_features = 784  # MNIST 输入
        
        for i in range(n_layers):
            out_features = hidden_size
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = out_features
        
        layers.append(nn.Linear(in_features, 10))  # 输出层
        
        return nn.Sequential(*layers).to(self.device)
    
    def objective(self, trial):
        # 使用建议的超参数创建模型
        model = self.create_model(trial)
        
        # 建议优化器超参数
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size
        )
        
        # 训练循环
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0
        
        for epoch in range(10):
            # 训练
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)
            
            # 剪枝（对不有前景的试验早停）
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_acc
    
    def optimize(self, n_trials=100):
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler()
        )
        
        study.optimize(self.objective, n_trials=n_trials)
        
        print(f"最佳试验：{study.best_trial.params}")
        return study.best_trial.params

# 用法
optimizer = HyperparameterOptimizer(train_dataset, val_dataset)
best_params = optimizer.optimize(n_trials=50)
```

### 7.3.4 使用 Kubeflow Katib 进行 HPO

🟡 **中级**

```yaml
# Kubeflow Katib HPO 实验
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: hpo-experiment
  namespace: kubeflow
spec:
  objective:
    type: maximize
    goal: 0.95
    objectiveMetricName: accuracy
  algorithm:
    algorithmName: bayesianoptimization
  parallelTrialCount: 3
  maxTrialCount: 20
  maxFailedTrialCount: 3
  parameters:
  - name: learning-rate
    parameterType: double
    feasibleSpace:
      min: "0.0001"
      max: "0.01"
  - name: batch-size
    parameterType: categorical
    feasibleSpace:
      list:
      - "16"
      - "32"
      - "64"
      - "128"
  - name: num-layers
    parameterType: int
    feasibleSpace:
      min: "2"
      max: "8"
  - name: hidden-size
    parameterType: categorical
    feasibleSpace:
      list:
      - "256"
      - "512"
      - "1024"
  trialTemplate:
    primaryContainerName: training-container
    trialParameters:
    - name: learningRate
      description: 训练的学习率
      reference: learning-rate
    - name: batchSize
      description: 训练的批量大小
      reference: batch-size
    - name: numLayers
      description: 层数
      reference: num-layers
    - name: hiddenSize
      description: 隐藏层大小
      reference: hidden-size
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            containers:
            - name: training-container
              image: registry.example.com/trainer:latest
              command:
              - python
              args:
              - train.py
              - --learning-rate=${learningRate}
              - --batch-size=${batchSize}
              - --num-layers=${numLayers}
              - --hidden-size=${hiddenSize}
              resources:
                limits:
                  nvidia.com/gpu: 1
```

---

## 7.4 实验管理与跟踪

### 7.4.1 实验管理的挑战

🟢 **初级**

随着 ML 项目的增长，管理实验变得至关重要。没有适当的跟踪，你会丢失：
- 哪些超参数产生了最佳结果
- 如何复现之前的结果
- 系统比较不同方法的能力
- 最佳模型的血缘关系

```
没有实验跟踪：
┌─────────────────────────────────────────────────────────────┐
│  "我上周得到了 95% 的准确率..."                               │
│                                                             │
│  - 什么超参数？"我想是 lr=0.001"                             │
│  - 什么数据集版本？"最新的那个"                               │
│  - 什么随机种子？"我不记得了"                                 │
│  - 什么其他设置？"我不确定"                                   │
│                                                             │
│  结果：无法复现！                                            │
└─────────────────────────────────────────────────────────────┘

有实验跟踪：
┌─────────────────────────────────────────────────────────────┐
│  实验 #42                                                   │
│  ├── 运行 ID：exp42-run-003                                 │
│  ├── 超参数：                                                │
│  │   ├── lr：0.001                                          │
│  │   ├── batch_size：32                                     │
│  │   └── epochs：10                                         │
│  ├── 指标：                                                  │
│  │   ├── train_loss：0.05                                   │
│  │   ├── val_loss：0.08                                     │
│  │   └── accuracy：0.95                                     │
│  ├── 产物：                                                  │
│  │   ├── 模型：s3://models/exp42-run-003/model.pt           │
│  │   └── 数据：s3://datasets/v2/train.parquet               │
│  └── Git 提交：abc1234                                      │
│                                                             │
│  结果：完全可复现！                                          │
└─────────────────────────────────────────────────────────────┘
```

### 7.4.2 MLflow 实验跟踪

🟡 **中级**

```python
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_with_mlflow():
    mlflow.set_experiment("text-classification")
    
    with mlflow.start_run(run_name="bert-base-finetune"):
        # 记录参数
        mlflow.log_param("model", "bert-base-uncased")
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 3)
        mlflow.log_param("max_length", 128)
        
        # 训练循环
        model = create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        for epoch in range(3):
            train_loss = train_epoch(model, optimizer, train_loader)
            val_loss, val_acc = evaluate(model, val_loader)
            
            # 记录指标
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            print(f"Epoch {epoch}: loss={train_loss:.4f}, acc={val_acc:.4f}")
        
        # 记录模型
        mlflow.pytorch.log_model(model, "model")
        
        # 记录额外产物
        mlflow.log_artifact("training_config.yaml")
        mlflow.log_artifact("tokenizer_config.json")
        
        # 设置标签
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("dataset", "imdb")

if __name__ == "__main__":
    train_with_mlflow()
```

```bash
# 启动 MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# 在 http://localhost:5000 查看实验
```

### 7.4.3 模型注册中心

🔴 **高级**

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 注册模型
model_uri = "runs:/<run_id>/model"
model_name = "text-classifier"
client.create_registered_model(model_name)
client.create_model_version(
    name=model_name,
    source=model_uri,
    description="基于 BERT 的文本分类器 v1.0"
)

# 转换模型阶段
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# 添加标签和描述
client.update_model_version(
    name=model_name,
    version=1,
    description="初始生产模型",
    tags={"accuracy": "0.95", "framework": "pytorch"}
)

# 获取模型用于服务
model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
```

### 7.4.4 实验比较

🟡 **中级**

```
MLflow 实验仪表板：
┌─────────────────────────────────────────────────────────────┐
│  实验：text-classification                                  │
│                                                             │
│  运行名称        │ lr     │ batch │ epochs │ 准确率         │
│  ─────────────────┼─────────┼───────┼────────┼────────────  │
│  bert-base-v1    │ 2e-5   │ 32    │ 3      │ 0.945        │
│  bert-base-v2    │ 3e-5   │ 64    │ 5      │ 0.952  ★     │
│  bert-base-v3    │ 1e-5   │ 16    │ 10     │ 0.938        │
│  roberta-base    │ 2e-5   │ 32    │ 3      │ 0.958  ★★    │
│  distilbert      │ 5e-5   │ 128   │ 5      │ 0.921        │
│                                                             │
│  ★  = 最佳准确率                                              │
│  ★★ = 最佳整体（准确率 + 速度）                                │
│                                                             │
│  图表：                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  准确率随 epoch 变化                                 │   │
│  │  0.96 ┤              ★★──────                       │   │
│  │  0.94 ┤     ★─────────                              │   │
│  │  0.92 ┤     ─────★                                  │   │
│  │  0.90 ┤───────────────                              │   │
│  │       └─────┬─────┬─────┬─────┬─────┬─────        │   │
│  │            Ep1   Ep2   Ep3   Ep4   Ep5            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7.5 训练资源管理与优化

### 7.5.1 资源分配策略

🟡 **中级**

```
资源管理架构：
┌─────────────────────────────────────────────────────────────┐
│                    资源管理器                                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              资源池                                   │   │
│  │                                                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │  GPU     │ │  GPU     │ │  GPU     │           │   │
│  │  │  插槽 1  │ │  插槽 2  │ │  插槽 3  │           │   │
│  │  │  (空闲)  │ │  (繁忙)  │ │  (空闲)  │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘           │   │
│  │                                                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │  CPU     │ │  CPU     │ │  CPU     │           │   │
│  │  │  8 核    │ │  16 核   │ │  4 核    │           │   │
│  │  │  (空闲)  │ │  (繁忙)  │ │  (预留)  │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘           │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              队列管理器                               │   │
│  │                                                     │   │
│  │  作业 A：优先级 1，2 GPU，32GB RAM      [运行中]      │   │
│  │  作业 B：优先级 2，1 GPU，16GB RAM      [等待中]      │   │
│  │  作业 C：优先级 3，4 GPU，64GB RAM      [等待中]      │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.5.2 成本优化

🔴 **高级**

```python
# 资源感知的训练配置
training_config = {
    "name": "cost-optimized-training",
    "strategy": {
        "type": "preemptible",  # 使用抢占式实例
        "max_retries": 3,
        "checkpoint_frequency": 300,  # 秒
    },
    "resources": {
        "gpu": {
            "type": "nvidia-tesla-t4",  # 经济实惠的 GPU
            "count": 2,
            "spot_price": 0.50,  # 美元/小时
        },
        "cpu": {
            "count": 8,
            "memory_gb": 32,
        },
        "storage": {
            "type": "ssd",
            "size_gb": 100,
        },
    },
    "scheduling": {
        "priority": "medium",
        "time_budget_hours": 24,
        "auto_stop_if_no_improvement": True,
    },
    "monitoring": {
        "track_cost": True,
        "alert_threshold_usd": 100,
        "dashboard": "grafana",
    },
}
```

### 7.5.3 检查点策略

🟡 **中级**

```python
import torch
import os
from pathlib import Path

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
    
    def save_checkpoint(
        self, 
        model, 
        optimizer, 
        scheduler, 
        epoch, 
        step, 
        metrics, 
        is_best=False
    ):
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
        }
        
        # 保存常规检查点
        checkpoint_path = self.checkpoint_dir / f'checkpoint-{epoch}-{step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
        # 清理旧检查点
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    
    def _cleanup_checkpoints(self):
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint-*.pt'),
            key=lambda x: x.stat().st_mtime
        )
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()

# 在训练循环中使用
checkpoint_manager = CheckpointManager('./checkpoints')

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        # 训练步骤...
        
        if step % 1000 == 0:
            metrics = {'loss': loss.item(), 'accuracy': accuracy}
            is_best = accuracy > best_accuracy
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, step, metrics, is_best
            )
```

---

## 💡 案例研究：基于 Kubeflow 的分布式训练平台

### 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│              Kubeflow 分布式训练平台                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    用户界面                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐     │   │
│  │  │ Jupyter  │  │ CLI      │  │ Kubeflow         │     │   │
│  │  │ Notebook │  │(kubectl) │  │ 仪表板            │     │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘     │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                      │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │                  Kubeflow 控制平面                        │   │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │ 管道         │  │ Katib        │  │ 训练         │ │   │
│  │  │ 编排器       │  │ (HPO)        │  │ 操作符       │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │ 模型         │  │ 元数据       │  │ Notebook     │ │   │
│  │  │ 注册中心     │  │ 存储         │  │ 控制器       │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│  │                                                         │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                      │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │                  Kubernetes 集群                         │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              GPU 节点池                           │   │   │
│  │  │                                                 │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │   │   │
│  │  │  │ 节点 1   │  │ 节点 2   │  │ 节点 3   │     │   │   │
│  │  │  │ 4xV100   │  │ 4xV100   │  │ 4xV100   │     │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘     │   │   │
│  │  │                                                 │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              存储层                               │   │   │
│  │  │                                                 │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │   │   │
│  │  │  │ 训练     │  │ 检查点   │  │ 模型     │     │   │   │
│  │  │  │ 数据     │  │ 存储     │  │ 产物     │     │   │   │
│  │  │  │(S3/NFS) │  │ (S3)    │  │ (S3)    │     │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘     │   │   │
│  │  │                                                 │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 部署说明

```bash
# 1. 创建 Kubernetes 集群
gcloud container clusters create ml-training-cluster \
    --num-nodes=3 \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=4 \
    --zone=us-central1-a

# 2. 安装 Kubeflow
export KUBEFLOW_VERSION=1.8.0
kubectl apply -f https://github.com/kubeflow/manifests/releases/download/v${KUBEFLOW_VERSION}/kubeflow.yaml

# 3. 等待所有 Pod 就绪
kubectl wait --for=condition=Ready pods --all -n kubeflow --timeout=600s

# 4. 创建训练命名空间
kubectl create namespace training

# 5. 部署分布式训练任务
kubectl apply -f pytorchjob.yaml -n training

# 6. 监控训练
kubectl logs -f pytorch-dist-training-master-0 -n training

# 7. 访问 Kubeflow 仪表板
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

### 监控仪表板

```
Grafana 仪表板：训练平台指标
┌─────────────────────────────────────────────────────────────┐
│  训练任务：pytorch-dist-training                             │
│                                                             │
│  GPU 利用率：  ████████████████████░░░░ 87%                  │
│  内存使用：    ██████████████░░░░░░░░░░ 62%                  │
│  训练损失：    ↓ 下降中                                       │
│  训练时间：    2小时34分（预计剩余：1小时12分）                 │
│                                                             │
│  每 GPU 指标：                                               │
│  ┌────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │ GPU    │ 利用率   │ 内存     │ 温度     │ 功率     │   │
│  ├────────┼──────────┼──────────┼──────────┼──────────┤   │
│  │ GPU 0  │ 92%      │ 14GB/16GB│ 72°C     │ 280W     │   │
│  │ GPU 1  │ 88%      │ 13GB/16GB│ 70°C     │ 275W     │   │
│  │ GPU 2  │ 85%      │ 14GB/16GB│ 71°C     │ 278W     │   │
│  │ GPU 3  │ 84%      │ 13GB/16GB│ 69°C     │ 272W     │   │
│  └────────┴──────────┴──────────┴──────────┴──────────┘   │
│                                                             │
│  成本追踪器：                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  每小时费率：   $12.40                               │   │
│  │  总成本：       $31.73                               │   │
│  │  预算使用：     $100 每日预算的 32%                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 总结

**关键要点：**

1. **环境可复现性**是基础——使用容器并版本化所有内容
2. **分布式训练**实现了超出单 GPU 限制的扩展
3. **HPO** 应该是系统化的，而非随机的——使用贝叶斯优化
4. **实验跟踪**对于调试和比较至关重要
5. **资源管理**直接影响成本和训练速度

**下一章预览：**
在第 8 章中，我们将探讨**模型部署架构**，涵盖部署策略、使用 Seldon Core 的模型服务、A/B 测试和推理优化。

---

*第7章结束*
