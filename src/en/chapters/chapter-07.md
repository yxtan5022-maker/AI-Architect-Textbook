# Chapter 7: Model Training Architecture

> **Part III: MLOps Architecture**

**Learning Objectives:**
- Design scalable and reproducible training environments
- Implement distributed training across multiple nodes and GPUs
- Build hyperparameter optimization pipelines
- Set up experiment tracking and management systems
- Optimize training resource utilization and cost

---

## 7.1 Training Environment Design

### 7.1.1 The Training Environment Challenge

🟢 **Beginner**

A training environment is more than just a Jupyter notebook with some libraries. It's a carefully orchestrated combination of hardware, software, data access, and configuration that ensures reproducibility, scalability, and efficiency.

```
The Training Environment Stack:
┌─────────────────────────────────────────────────┐
│                  Application Layer               │
│         (Training Scripts, Notebooks)            │
├─────────────────────────────────────────────────┤
│                  Framework Layer                  │
│      (PyTorch, TensorFlow, JAX, XGBoost)        │
├─────────────────────────────────────────────────┤
│                  Runtime Layer                    │
│     (Python, CUDA, cuDNN, Container Runtime)     │
├─────────────────────────────────────────────────┤
│                  Infrastructure Layer             │
│     (GPU/TPU, Storage, Networking, Cluster)      │
├─────────────────────────────────────────────────┤
│                  Orchestration Layer              │
│     (Kubernetes, Kubeflow, Slurm)               │
└─────────────────────────────────────────────────┘
```

### 7.1.2 Environment Reproducibility

🟡 **Intermediate**

Reproducibility is the cornerstone of reliable ML training. Without it, you cannot debug, compare, or trust your models.

**Key Components for Reproducibility:**

```python
# Environment specification files

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

### 7.1.3 Container-Based Training Environments

🔴 **Advanced**

```
Container-Based Training Architecture:
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │  Training Pod   │  │  Training Pod   │                 │
│  │  ┌───────────┐  │  │  ┌───────────┐  │                 │
│  │  │ Training  │  │  │  │ Training  │  │                 │
│  │  │ Script    │  │  │  │ Script    │  │                 │
│  │  └───────────┘  │  │  └───────────┘  │                 │
│  │  ┌───────────┐  │  │  ┌───────────┐  │                 │
│  │  │ GPU       │  │  │  │ GPU       │  │                 │
│  │  │ Runtime   │  │  │  │ Runtime   │  │                 │
│  │  └───────────┘  │  │  └───────────┘  │                 │
│  │  ┌───────────┐  │  │  ┌───────────┐  │                 │
│  │  │ Data      │  │  │  │ Data      │  │                 │
│  │  │ Volume    │  │  │  │ Volume    │  │                 │
│  │  └───────────┘  │  │  └───────────┘  │                 │
│  └─────────────────┘  └─────────────────┘                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Shared Storage (NFS/S3)                 │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐    │   │
│  │  │ Dataset │  │ Checkpt │  │ Model Artifacts │    │   │
│  │  └─────────┘  └─────────┘  └─────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.1.4 GPU Resource Management

🔴 **Advanced**

```yaml
# Kubernetes Pod spec for GPU training
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
        nvidia.com/gpu: "2"  # Request 2 GPUs
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

## 7.2 Distributed Training Architecture

### 7.2.1 Why Distributed Training?

🟢 **Beginner**

Distributed training splits the training workload across multiple devices (GPUs/TPUs) or machines to:

1. **Reduce training time**: Train large models faster
2. **Handle large datasets**: Process data that doesn't fit in memory
3. **Enable larger models**: Train models that require more parameters
4. **Improve utilization**: Make better use of available hardware

```
Single GPU Training:
┌──────────────┐
│    GPU 0     │
│  Full Model  │──── Hours to train
│  Full Data   │
└──────────────┘

Distributed Training:
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    GPU 0     │    │    GPU 1     │    │    GPU 2     │
│  Model Part  │    │  Model Part  │    │  Model Part  │
│  Data Shard  │    │  Data Shard  │    │  Data Shard  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    Minutes to train
```

### 7.2.2 Data Parallelism

🟡 **Intermediate**

Data parallelism replicates the model on each device and splits the data across devices.

```
Data Parallelism:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Global Dataset                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Shard 0  │  Shard 1  │  Shard 2  │  Shard 3      │   │
│  └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────────┘   │
│        │           │           │           │              │
│        ▼           ▼           ▼           ▼              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │  GPU 0   │ │  GPU 1   │ │  GPU 2   │ │  GPU 3   │    │
│  │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │    │
│  │ │Model │ │ │ │Model │ │ │ │Model │ │ │ │Model │ │    │
│  │ │Copy  │ │ │ │Copy  │ │ │ │Copy  │ │ │ │Copy  │ │    │
│  │ └──────┘ │ │ └──────┘ │ │ └──────┘ │ │ └──────┘ │    │
│  │ Forward  │ │ Forward  │ │ Forward  │ │ Forward  │    │
│  │ Backward │ │ Backward │ │ Backward │ │ Backward │    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘    │
│       │            │            │            │            │
│       └────────────┴────────────┴────────────┘            │
│                          │                                │
│                          ▼                                │
│              ┌──────────────────────┐                    │
│              │  Gradient AllReduce  │                    │
│              │  (Average Gradients) │                    │
│              └──────────┬───────────┘                    │
│                         │                                │
│                         ▼                                │
│              ┌──────────────────────┐                    │
│              │  Update All Models   │                    │
│              └──────────────────────┘                    │
│                                                          │
└─────────────────────────────────────────────────────────────┘
```

**PyTorch DDP Implementation:**

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
    
    # Create model
    model = nn.Linear(100, 10).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler
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
        sampler.set_epoch(epoch)  # Important for shuffling
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

# Launch with: torchrun --nproc_per_node=4 train.py
```

### 7.2.3 Model Parallelism

🔴 **Advanced**

Model parallelism splits the model itself across multiple devices, useful when the model is too large to fit on a single GPU.

```
Model Parallelism (Pipeline):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Model Layers: L0 → L1 → L2 → L3 → L4 → L5 → L6 → L7    │
│                                                             │
│  GPU 0:  L0 → L1 → L2 → L3                                  │
│  GPU 1:  L4 → L5 → L6 → L7                                  │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  GPU 0   │───►│  GPU 1   │───►│  Output  │             │
│  │ L0-L3    │    │ L4-L7    │    │  Result  │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│                                                             │
│  Micro-batch Pipeline:                                      │
│  Time ──────────────────────────────────────────────────►  │
│  GPU 0: [MB1][MB2][MB3][MB4]                               │
│  GPU 1:      [MB1][MB2][MB3][MB4]                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2.4 Tensor Parallelism

🔴 **Advanced**

Tensor parallelism splits individual layers across devices:

```
Tensor Parallelism (Column Parallel):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Linear Layer: Y = XW                                       │
│                                                             │
│  Split W into W1, W2 (column partition)                    │
│                                                             │
│  GPU 0: Y1 = X * W1  ──┐                                   │
│                         ├──► Concatenate ──► Y              │
│  GPU 1: Y2 = X * W2  ──┘                                   │
│                                                             │
│  Reduces memory per GPU while maintaining full capacity     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2.5 Distributed Training with Kubeflow

🟡 **Intermediate**

Kubeflow provides built-in support for distributed training through TFJob and PyTorchJob operators:

```yaml
# Kubeflow PyTorchJob for distributed training
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
      replicas: 3  # 3 workers + 1 master = 4 GPUs total
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
# Apply the training job
kubectl apply -f pytorchjob.yaml

# Monitor training progress
kubectl logs -f pytorch-dist-training-master-0 -n kubeflow

# Check job status
kubectl get pytorchjob pytorch-dist-training -n kubeflow -o yaml
```

---

## 7.3 Hyperparameter Optimization Architecture

### 7.3.1 The Hyperparameter Challenge

🟢 **Beginner**

Hyperparameters are configuration settings that are not learned during training but significantly affect model performance. Finding optimal hyperparameters is expensive and time-consuming.

```
Common Hyperparameters:
┌─────────────────────────────────────────────────────────────┐
│  Model Hyperparameters:                                     │
│  ├── Learning Rate: 0.001, 0.01, 0.1                       │
│  ├── Batch Size: 16, 32, 64, 128                           │
│  ├── Number of Layers: 4, 6, 8, 12                         │
│  ├── Hidden Size: 256, 512, 1024, 2048                     │
│  ├── Dropout Rate: 0.1, 0.2, 0.3                           │
│  └── Weight Decay: 0.0001, 0.001, 0.01                     │
│                                                             │
│  Training Hyperparameters:                                  │
│  ├── Optimizer: SGD, Adam, AdamW                            │
│  ├── Scheduler: Cosine, Linear, StepLR                      │
│  ├── Warmup Steps: 0, 100, 1000                            │
│  └── Gradient Clipping: 0, 1.0, 5.0                        │
│                                                             │
│  Search Space: 4 × 4 × 4 × 4 × 3 × 3 × 3 × 3 × 4 × 4   │
│             = 82,944 possible configurations               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3.2 HPO Strategies

🟡 **Intermediate**

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Grid Search** | Try all combinations | Exhaustive | Exponentially expensive |
| **Random Search** | Random sampling | Simple, often effective | No learning from past |
| **Bayesian Optimization** | Model-based search | Smart, efficient | Complex implementation |
| **Hyperband** | Early stopping + random | Resource efficient | May miss good configs |
| **BOHB** | Bayesian + Hyperband | Best of both | Most complex |

```
Search Strategy Comparison:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Grid Search:                                               │
│  ┌───┬───┬───┬───┐                                        │
│  │ ● │ ● │ ● │ ● │  ● = Trained model                      │
│  ├───┼───┼───┼───┤  (all combinations)                    │
│  │ ● │ ● │ ● │ ● │                                        │
│  ├───┼───┼───┼───┤                                        │
│  │ ● │ ● │ ● │ ● │  Total: 12 models                       │
│  └───┴───┴───┴───┘                                        │
│                                                             │
│  Random Search:                                             │
│  ┌───┬───┬───┬───┐                                        │
│  │   │ ● │   │   │  ● = Randomly selected                  │
│  ├───┼───┼───┼───┤  (fewer total models)                   │
│  │ ● │   │   │ ● │                                        │
│  ├───┼───┼───┼───┤                                        │
│  │   │   │ ● │   │  Total: 4 models                        │
│  └───┴───┴───┴───┘  (but good coverage)                    │
│                                                             │
│  Bayesian:                                                  │
│  ┌───┬───┬───┬───┐                                        │
│  │   │   │ ● │   │  ● = Model trained                      │
│  ├───┼───┼───┼───┤  ○ = Predicted promising                │
│  │   │ ○ │ ○ │   │  Based on past results                  │
│  ├───┼───┼───┼───┤                                        │
│  │   │ ● │   │   │  Total: 3 models                        │
│  └───┴───┴───┴───┘  (most efficient)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3.3 Bayesian Optimization Implementation

🔴 **Advanced**

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
        # Suggest hyperparameters
        n_layers = trial.suggest_int('n_layers', 2, 8)
        hidden_size = trial.suggest_categorical('hidden_size', [256, 512, 1024, 2048])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        layers = []
        in_features = 784  # MNIST input
        
        for i in range(n_layers):
            out_features = hidden_size
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = out_features
        
        layers.append(nn.Linear(in_features, 10))  # Output
        
        return nn.Sequential(*layers).to(self.device)
    
    def objective(self, trial):
        # Create model with suggested hyperparameters
        model = self.create_model(trial)
        
        # Suggest optimizer hyperparameters
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
        
        # Training loop
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0
        
        for epoch in range(10):
            # Training
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # Validation
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
            
            # Pruning (early stopping for unpromising trials)
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
        
        print(f"Best trial: {study.best_trial.params}")
        return study.best_trial.params

# Usage
optimizer = HyperparameterOptimizer(train_dataset, val_dataset)
best_params = optimizer.optimize(n_trials=50)
```

### 7.3.4 HPO with Kubeflow Katib

🟡 **Intermediate**

```yaml
# Kubeflow Katib HPO experiment
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
      description: Learning rate for training
      reference: learning-rate
    - name: batchSize
      description: Batch size for training
      reference: batch-size
    - name: numLayers
      description: Number of layers
      reference: num-layers
    - name: hiddenSize
      description: Hidden layer size
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

## 7.4 Experiment Management & Tracking

### 7.4.1 The Experiment Management Challenge

🟢 **Beginner**

As ML projects grow, managing experiments becomes critical. Without proper tracking, you lose:
- Which hyperparameters produced the best results
- How to reproduce previous results
- The ability to compare approaches systematically
- The lineage of your best models

```
Without Experiment Tracking:
┌─────────────────────────────────────────────────────────────┐
│  "I got 95% accuracy last week..."                          │
│                                                             │
│  - What hyperparameters? "I think lr=0.001"                │
│  - What dataset version? "The latest one"                   │
│  - What random seed? "I don't remember"                     │
│  - What other settings? "I'm not sure"                      │
│                                                             │
│  Result: Cannot reproduce!                                  │
└─────────────────────────────────────────────────────────────┘

With Experiment Tracking:
┌─────────────────────────────────────────────────────────────┐
│  Experiment #42                                             │
│  ├── Run ID: exp42-run-003                                  │
│  ├── Hyperparameters:                                       │
│  │   ├── lr: 0.001                                          │
│  │   ├── batch_size: 32                                     │
│  │   └── epochs: 10                                         │
│  ├── Metrics:                                               │
│  │   ├── train_loss: 0.05                                   │
│  │   ├── val_loss: 0.08                                     │
│  │   └── accuracy: 0.95                                     │
│  ├── Artifacts:                                             │
│  │   ├── model: s3://models/exp42-run-003/model.pt          │
│  │   └── data: s3://datasets/v2/train.parquet               │
│  └── Git Commit: abc1234                                    │
│                                                             │
│  Result: Fully reproducible!                                │
└─────────────────────────────────────────────────────────────┘
```

### 7.4.2 MLflow Experiment Tracking

🟡 **Intermediate**

```python
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_with_mlflow():
    mlflow.set_experiment("text-classification")
    
    with mlflow.start_run(run_name="bert-base-finetune"):
        # Log parameters
        mlflow.log_param("model", "bert-base-uncased")
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 3)
        mlflow.log_param("max_length", 128)
        
        # Training loop
        model = create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        for epoch in range(3):
            train_loss = train_epoch(model, optimizer, train_loader)
            val_loss, val_acc = evaluate(model, val_loader)
            
            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            print(f"Epoch {epoch}: loss={train_loss:.4f}, acc={val_acc:.4f}")
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        # Log additional artifacts
        mlflow.log_artifact("training_config.yaml")
        mlflow.log_artifact("tokenizer_config.json")
        
        # Set tags
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("dataset", "imdb")

if __name__ == "__main__":
    train_with_mlflow()
```

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# View experiments at http://localhost:5000
```

### 7.4.3 Model Registry

🔴 **Advanced**

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model
model_uri = "runs:/<run_id>/model"
model_name = "text-classifier"
client.create_registered_model(model_name)
client.create_model_version(
    name=model_name,
    source=model_uri,
    description="BERT-based text classifier v1.0"
)

# Transition model stage
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# Add tags and descriptions
client.update_model_version(
    name=model_name,
    version=1,
    description="Initial production model",
    tags={"accuracy": "0.95", "framework": "pytorch"}
)

# Get model for serving
model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
```

### 7.4.4 Experiment Comparison

🟡 **Intermediate**

```
MLflow Experiment Dashboard:
┌─────────────────────────────────────────────────────────────┐
│  Experiment: text-classification                            │
│                                                             │
│  Run Name          │ lr      │ batch │ epochs │ accuracy   │
│  ─────────────────┼─────────┼───────┼────────┼──────────── │
│  bert-base-v1     │ 2e-5    │ 32    │ 3      │ 0.945      │
│  bert-base-v2     │ 3e-5    │ 64    │ 5      │ 0.952  ★   │
│  bert-base-v3     │ 1e-5    │ 16    │ 10     │ 0.938      │
│  roberta-base     │ 2e-5    │ 32    │ 3      │ 0.958  ★★  │
│  distilbert       │ 5e-5    │ 128   │ 5      │ 0.921      │
│                                                             │
│  ★  = Best accuracy                                         │
│  ★★ = Best overall (accuracy + speed)                       │
│                                                             │
│  Charts:                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Accuracy over epochs                               │   │
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

## 7.5 Training Resource Management & Optimization

### 7.5.1 Resource Allocation Strategies

🟡 **Intermediate**

```
Resource Management Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    Resource Manager                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Resource Pool                           │   │
│  │                                                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │  GPU     │ │  GPU     │ │  GPU     │           │   │
│  │  │  Slot 1  │ │  Slot 2  │ │  Slot 3  │           │   │
│  │  │  (Free)  │ │  (Busy)  │ │  (Free)  │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘           │   │
│  │                                                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │  CPU     │ │  CPU     │ │  CPU     │           │   │
│  │  │  8 cores │ │  16 cores│ │  4 cores │           │   │
│  │  │  (Free)  │ │  (Busy)  │ │  (Reserved)│          │   │
│  │  └──────────┘ └──────────┘ └──────────┘           │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Queue Manager                           │   │
│  │                                                     │   │
│  │  Job A: Priority 1, 2 GPUs, 32GB RAM    [Running]  │   │
│  │  Job B: Priority 2, 1 GPU, 16GB RAM     [Waiting]  │   │
│  │  Job C: Priority 3, 4 GPUs, 64GB RAM    [Waiting]  │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.5.2 Cost Optimization

🔴 **Advanced**

```python
# Resource-aware training configuration
training_config = {
    "name": "cost-optimized-training",
    "strategy": {
        "type": "preemptible",  # Use spot instances
        "max_retries": 3,
        "checkpoint_frequency": 300,  # seconds
    },
    "resources": {
        "gpu": {
            "type": "nvidia-tesla-t4",  # Cost-effective GPU
            "count": 2,
            "spot_price": 0.50,  # $/hour
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

### 7.5.3 Checkpointing Strategy

🟡 **Intermediate**

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
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint-{epoch}-{step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
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

# Usage in training loop
checkpoint_manager = CheckpointManager('./checkpoints')

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        # Training step...
        
        if step % 1000 == 0:
            metrics = {'loss': loss.item(), 'accuracy': accuracy}
            is_best = accuracy > best_accuracy
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, step, metrics, is_best
            )
```

---

## 💡 Case Study: Kubeflow-Based Distributed Training Platform

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Kubeflow Distributed Training Platform              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    User Interface                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐     │   │
│  │  │ Jupyter  │  │ CLI      │  │ Kubeflow         │     │   │
│  │  │ Notebook │  │ (kubectl)│  │ Dashboard        │     │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘     │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                      │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │                  Kubeflow Control Plane                   │   │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │ Pipelines    │  │ Katib        │  │ Training     │ │   │
│  │  │ Orchestrator │  │ (HPO)        │  │ Operators    │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │ Model        │  │ Metadata     │  │ Notebooks    │ │   │
│  │  │ Registry     │  │ Store        │  │ Controller   │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │   │
│  │                                                         │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                      │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │                  Kubernetes Cluster                       │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              GPU Node Pool                       │   │   │
│  │  │                                                 │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │   │   │
│  │  │  │ Node 1   │  │ Node 2   │  │ Node 3   │     │   │   │
│  │  │  │ 4xV100   │  │ 4xV100   │  │ 4xV100   │     │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘     │   │   │
│  │  │                                                 │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              Storage Layer                       │   │   │
│  │  │                                                 │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │   │   │
│  │  │  │ Training │  │ Checkpoint│  │ Model    │     │   │   │
│  │  │  │ Data     │  │ Store     │  │ Artifacts│     │   │   │
│  │  │  │ (S3/NFS) │  │ (S3)     │  │ (S3)     │     │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘     │   │   │
│  │  │                                                 │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Instructions

```bash
# 1. Create Kubernetes cluster
gcloud container clusters create ml-training-cluster \
    --num-nodes=3 \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-v100,count=4 \
    --zone=us-central1-a

# 2. Install Kubeflow
export KUBEFLOW_VERSION=1.8.0
kubectl apply -f https://github.com/kubeflow/manifests/releases/download/v${KUBEFLOW_VERSION}/kubeflow.yaml

# 3. Wait for all pods to be ready
kubectl wait --for=condition=Ready pods --all -n kubeflow --timeout=600s

# 4. Create training namespace
kubectl create namespace training

# 5. Deploy distributed training job
kubectl apply -f pytorchjob.yaml -n training

# 6. Monitor training
kubectl logs -f pytorch-dist-training-master-0 -n training

# 7. Access Kubeflow Dashboard
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

### Monitoring Dashboard

```
Grafana Dashboard: Training Platform Metrics
┌─────────────────────────────────────────────────────────────┐
│  Training Job: pytorch-dist-training                        │
│                                                             │
│  GPU Utilization:  ████████████████████░░░░ 87%             │
│  Memory Usage:     ██████████████░░░░░░░░░░ 62%             │
│  Training Loss:    ↓ Decreasing                             │
│  Training Time:    2h 34m (ETA: 1h 12m)                     │
│                                                             │
│  Per-GPU Metrics:                                           │
│  ┌────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │ GPU    │ Util     │ Memory   │ Temp     │ Power    │   │
│  ├────────┼──────────┼──────────┼──────────┼──────────┤   │
│  │ GPU 0  │ 92%      │ 14GB/16GB│ 72°C     │ 280W     │   │
│  │ GPU 1  │ 88%      │ 13GB/16GB│ 70°C     │ 275W     │   │
│  │ GPU 2  │ 85%      │ 14GB/16GB│ 71°C     │ 278W     │   │
│  │ GPU 3  │ 84%      │ 13GB/16GB│ 69°C     │ 272W     │   │
│  └────────┴──────────┴──────────┴──────────┴──────────┘   │
│                                                             │
│  Cost Tracker:                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Hourly Rate:    $12.40                             │   │
│  │  Total Cost:     $31.73                             │   │
│  │  Budget Used:    32% of $100 daily budget           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

**Key Takeaways:**

1. **Environment reproducibility** is foundational — use containers and version everything
2. **Distributed training** enables scaling beyond single GPU limits
3. **HPO** should be systematic, not random — use Bayesian optimization
4. **Experiment tracking** is essential for debugging and comparison
5. **Resource management** directly impacts cost and training speed

**Next Chapter Preview:**
In Chapter 8, we'll explore **Model Deployment Architecture**, covering deployment strategies, model serving with Seldon Core, A/B testing, and inference optimization.

---

*End of Chapter 7*
