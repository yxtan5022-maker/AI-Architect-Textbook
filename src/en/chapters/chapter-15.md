# Chapter 15: Distributed Computing Architecture

🟢 Beginner | 🟡 Intermediate | 🔴 Advanced | ⚫ Manager

---

## 15.1 Spark on Kubernetes

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Spark on Kubernetes Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Spark Application                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │   Driver │  │ Executor │  │ Executor │  │Executor│  │   │
│  │  │  (Pod)   │→ │  (Pod)   │  │  (Pod)   │  │ (Pod) │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Kubernetes Cluster                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ Node 1   │  │ Node 2   │  │ Node 3   │  │Node 4│  │   │
│  │  │8 CPU     │  │8 CPU     │  │8 CPU     │  │8 CPU │  │   │
│  │  │32GB RAM  │  │32GB RAM  │  │32GB RAM  │  │32GB  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Spark Operator Installation

🟡 Intermediate

```bash
# Install Spark Operator
helm repo add spark-operator https://kubeflow.github.io/spark-operator
helm repo update

# Install with RBAC
helm install spark-operator spark-operator/spark-operator \
  --namespace spark-operator \
  --create-namespace \
  --set sparkJobNamespace=spark-jobs \
  --set webhook.enable=true \
  --set webhook.port=8080

# Verify installation
kubectl get pods -n spark-operator
```

### Spark Application Definition

```yaml
# SparkApplication for ML data processing
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: feature-engineering
  namespace: spark-jobs
spec:
  type: Scala
  mode: cluster
  image: myregistry/spark-ml:3.3.1
  imagePullPolicy: Always
  mainClass: com.ml.FeatureEngineering
  mainApplicationFile: local:///opt/spark/jars/ml-pipeline.jar
  sparkVersion: "3.3.1"
  batchScheduler: volcano
  restartPolicy:
    type: OnFailure
    failureRetries: 3
    retryInterval: 10
  timeToLiveSeconds: 86400
  sparkConf:
    spark.kubernetes.authenticate.driver.serviceAccountName: spark
    spark.kubernetes.namespace: spark-jobs
    spark.dynamicAllocation.enabled: "true"
    spark.dynamicAllocation.minExecutors: "2"
    spark.dynamicAllocation.maxExecutors: "20"
    spark.dynamicAllocation.initialExecutors: "4"
    spark.shuffle.service.enabled: "true"
    spark.kubernetes.driver.volumes.persistentVolumeClaim.readWriteOnce.options.claimName: spark-driver-pvc
    spark.kubernetes.executor.volumes.persistentVolumeClaim.readWriteOnce.options.claimName: spark-executor-pvc
  driver:
    cores: 2
    coreLimit: "4"
    memory: "4g"
    serviceAccount: spark
    volumeMounts:
    - name: spark-driver-pvc
      mountPath: /data
    envSecretKeyRefs:
      MLFLOW_TRACKING_URI:
        name: mlflow-secrets
        key: tracking-uri
  executor:
    cores: 4
    coreLimit: "4"
    memory: "8g"
    instances: 4
    serviceAccount: spark
    volumeMounts:
    - name: spark-executor-pvc
      mountPath: /data
    envSecretKeyRefs:
      MLFLOW_TRACKING_URI:
        name: mlflow-secrets
        key: tracking-uri
  dynamicAllocation:
    enabled: true
    initialExecutors: 4
    minExecutors: 2
    maxExecutors: 20
```

### Spark Jobs for ML Pipelines

```scala
// Feature Engineering Job
package com.ml

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer}
import org.apache.spark.ml.Pipeline

object FeatureEngineering {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Feature Engineering Pipeline")
      .config("spark.kubernetes.driver.master", "k8s://https://kubernetes.default.svc:443")
      .getOrCreate()
    
    import spark.implicits._
    
    // Read raw data
    val rawData = spark.read.parquet("/data/raw/events")
    
    // Feature engineering pipeline
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
    
    val assembler = new VectorAssembler()
      .setInputCols(Array("categoryIndex", "price", "quantity", "hour_of_day"))
      .setOutputCol("features_raw")
    
    val scaler = new StandardScaler()
      .setInputCol("features_raw")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)
    
    val pipeline = new Pipeline()
      .setStages(Array(indexer, assembler, scaler))
    
    val model = pipeline.fit(rawData)
    val processedData = model.transform(rawData)
    
    // Save processed features
    processedData.write.mode("overwrite")
      .partitionBy("date")
      .parquet("/data/processed/features")
    
    // Log to MLflow
    import io.mlflow.spark.mlflow
    mlflow.log_artifact("/data/processed/features")
    
    spark.stop()
  }
}
```

---

## 15.2 Ray Distributed Framework

### Ray Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ray Cluster Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Head Node                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  GCS     │  │   Ray    │  │  Object  │  │Dashboard│ │   │
│  │  │(Global   │  │  Driver  │  │   Store  │  │       │  │   │
│  │  │ Control  │  │          │  │          │  │       │  │   │
│  │  │ Store)   │  │          │  │          │  │       │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Worker Nodes                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │Worker 4│  │   │
│  │  │┌────────┐│  │┌────────┐│  │┌────────┐│  │┌─────┐│  │   │
│  │  ││Raylet  ││  ││Raylet  ││  ││Raylet  ││  ││Rayl.││  │   │
│  │  │└────────┘│  │└────────┘│  │└────────┘│  │└─────┘│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   GPU Workers                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ GPU W1   │  │ GPU W2   │  │ GPU W3   │  │GPU W4│  │   │
│  │  │┌────────┐│  │┌────────┐│  │┌────────┐│  │┌─────┐│  │   │
│  │  ││4xV100  ││  ││4xV100  ││  ││4xA100  ││  ││4xA10││  │   │
│  │  │└────────┘│  │└────────┘│  │└────────┘│  │└─────┘│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Ray on Kubernetes with KubeRay

```yaml
# KubeRay RayCluster definition
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: ml-training-cluster
  namespace: ray-jobs
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
      num-cpus: "2"
    template:
      metadata:
        labels:
          rayCluster: ml-training-cluster
          role: head
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.7.0
          ports:
          - containerPort: 6379
            name: gcs-server
          - containerPort: 8265
            name: dashboard
          - containerPort: 10001
            name: client
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
          volumeMounts:
          - name: ray-head-storage
            mountPath: /tmp/ray
  workerGroupSpecs:
  - groupName: cpu-workers
    replicas: 4
    minReplicas: 2
    maxReplicas: 10
    rayStartParams:
      num-cpus: "4"
    template:
      metadata:
        labels:
          rayCluster: ml-training-cluster
          role: worker
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.7.0
          resources:
            requests:
              cpu: "4"
              memory: "8Gi"
            limits:
              cpu: "4"
              memory: "16Gi"
          volumeMounts:
          - name: ray-worker-storage
            mountPath: /tmp/ray
  - groupName: gpu-workers
    replicas: 2
    minReplicas: 1
    maxReplicas: 4
    rayStartParams:
      num-cpus: "4"
      num-gpus: "4"
    template:
      metadata:
        labels:
          rayCluster: ml-training-cluster
          role: gpu-worker
      spec:
        containers:
        - name: ray-gpu-worker
          image: rayproject/ray:2.7.0-gpu
          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "4"
            limits:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "4"
          volumeMounts:
          - name: ray-gpu-storage
            mountPath: /tmp/ray
        nodeSelector:
          accelerator: nvidia-tesla-v100
```

### Ray Training Code Example

```python
# Distributed training with Ray Train
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Initialize Ray cluster
ray.init(address="auto")

# Define model
def train_func(config):
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(
        config["data_path"], 
        transform=transform
    )
    
    # Distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=train.get_context().get_world_size(),
        rank=train.get_context().get_world_rank()
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=4
    )
    
    # Model setup
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
    model = train.torch.prepare_model(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to("cuda"), target.to("cuda")
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                train.report({
                    "loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx
                })

# Configure training
config = {
    "data_path": "/data/training",
    "batch_size": 32,
    "lr": 0.001,
    "epochs": 100,
    "num_classes": 1000
}

# Create trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"CPU": 4, "GPU": 1}
    ),
    dataset_config={
        "train": ray.data.read_parquet("/data/training")
    }
)

# Run training
result = trainer.fit()
print(f"Training completed. Results: {result}")
```

---

## 15.3 Dask Parallel Computing

### Dask Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dask Architecture                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Dask Scheduler                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Task    │  │  Graph   │  │  Worker  │  │Client │  │   │
│  │  │  Queue   │  │ Optimizer│  │ Manager  │  │Manager│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Dask Workers                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │Worker 4│  │   │
│  │  │┌────────┐│  │┌────────┐│  │┌────────┐│  │┌─────┐│  │   │
│  │  ││ 4 CPU  ││  ││ 4 CPU  ││  ││ 4 CPU  ││  ││4 CPU││  │   │
│  │  ││ 16GB   ││  ││ 16GB   ││  ││ 16GB   ││  ││16GB ││  │   │
│  │  │└────────┘│  │└────────┘│  │└────────┘│  │└─────┘│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Dask on Kubernetes

```yaml
# Dask Worker deployment on Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dask-worker
  namespace: dask-jobs
spec:
  replicas: 4
  selector:
    matchLabels:
      app: dask-worker
  template:
    metadata:
      labels:
        app: dask-worker
    spec:
      containers:
      - name: dask-worker
        image: daskdev/dask:2023.8.0
        command:
        - dask-worker
        - --nworkers=4
        - --nthreads=2
        - --memory-limit=8GB
        - --lifetime=3600
        - --lifetime-stagger=300
        resources:
          requests:
            cpu: "8"
            memory: "16Gi"
          limits:
            cpu: "8"
            memory: "16Gi"
        env:
        - name: DASK_SCHEDULER_ADDRESS
          value: "tcp://dask-scheduler:8786"
        ports:
        - containerPort: 8788
          name: dashboard
        volumeMounts:
        - name: dask-storage
          mountPath: /data
      volumes:
      - name: dask-storage
        persistentVolumeClaim:
          claimName: dask-pvc
```

### Dask ML Pipeline Example

```python
# Dask Distributed ML Pipeline
from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import GridSearchCV
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA
from dask_ml.linear_model import LogisticRegression
from dask_ml.pipeline import Pipeline
import dask.array as da
import dask.dataframe as dd

# Connect to Dask cluster
client = Client("tcp://dask-scheduler:8786")
print(f"Connected to Dask cluster: {client.dashboard_link}")

# Load data with Dask
train_data = dd.read_parquet("/data/training/features/*.parquet")
test_data = dd.read_parquet("/data/testing/features/*.parquet")

# Convert to Dask arrays
X_train = train_data.drop("label", axis=1).to_dask_array(lengths=True)
y_train = train_data["label"].to_dask_array(lengths=True)
X_test = test_data.drop("label", axis=1).to_dask_array(lengths=True)
y_test = test_data["label"].to_dask_array(lengths=True)

# Create ML pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=100)),
    ("lr", LogisticRegression(max_iter=1000))
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "pca__n_components": [50, 100, 200],
    "lr__C": [0.01, 0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    client=client
)

# Fit model
grid_search.fit(X_train, y_train)

# Evaluate
score = grid_search.score(X_test, y_test)
print(f"Test accuracy: {score:.4f}")
print(f"Best parameters: {grid_search.best_params_}")

# Save model
import joblib
joblib.dump(grid_search.best_estimator_, "/models/dask_ml_model.pkl")
```

---

## 15.4 Elastic Computing Resource Management

### Dynamic Scaling Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│            Elastic Computing Resource Management                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Metric Collection                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │Prometheus│  │  Custom  │  │  KEDA    │  │HPA   │  │   │
│  │  │Metrics   │  │  Metrics │  │  Metrics │  │Metrics│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Scaling Decision                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ Queue    │  │  GPU     │  │  Memory  │  │Cost  │  │   │
│  │  │ Depth    │  │ Util     │  │ Pressure │  │Budget│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Resource Provisioning                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │Cloud GPU │  │ On-prem  │  │  Spot    │  │MIG   │  │   │
│  │  │  Pool    │  │  GPU     │  │Instances │  │Pool  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### KEDA for AI Workload Scaling

```yaml
# KEDA ScaledObject for GPU workload scaling
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: gpu-training-scaler
  namespace: ai-training
spec:
  scaleTargetRef:
    name: training-deployment
  pollingInterval: 30
  cooldownPeriod: 300
  minReplicaCount: 2
  maxReplicaCount: 20
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      metricName: gpu_queue_depth
      threshold: "10"
      query: |
        sum(ray_queue_pending_tasks{job="training"})
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      metricName: gpu_utilization
      threshold: "80"
      query: |
        avg(DCGM_FI_DEV_GPU_UTIL{namespace="ai-training"})
  - type: cron
    metadata:
      timezone: America/New_York
      start: 0 8 * * 1-5
      end: 0 20 * * 1-5
      desiredReplicas: "10"
```

### Spot Instance Integration

```yaml
# Spot instance pool for cost optimization
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spot-training-workers
  namespace: ai-training
spec:
  replicas: 4
  selector:
    matchLabels:
      app: spot-training-worker
  template:
    metadata:
      labels:
        app: spot-training-worker
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node.kubernetes.io/capacity-type
                operator: In
                values:
                - spot
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: spot-training-worker
              topologyKey: kubernetes.io/hostname
      tolerations:
      - key: "spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      containers:
      - name: worker
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
        env:
        - name: CHECKPOINT_INTERVAL
          value: "1800"
        resources:
          requests:
            nvidia.com/gpu: "4"
            memory: "32Gi"
            cpu: "8"
          limits:
            nvidia.com/gpu: "4"
            memory: "32Gi"
            cpu: "8"
```

---

## 15.5 Hybrid Cloud Architecture

### Architecture Design

```
┌─────────────────────────────────────────────────────────────────┐
│              Hybrid Cloud AI Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   On-Premises Data Center               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │GPU Cluster│  │Data Lake │  │Model     │  │Infer.│  │   │
│  │  │(Training)│  │          │  │Registry  │  │Edge  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │   VPN / Direct    │                      │
│                    │   Connect / SD-WAN│                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Cloud Provider (AWS/Azure/GCP)        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Burst   │  │ Managed  │  │  Model   │  │API   │  │   │
│  │  │ Training │  │ Services │  │ Serving  │  │Gateway│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Cluster Federation

```yaml
# KubeFed configuration for multi-cluster AI
apiVersion: core.kubefed.io/v1beta1
kind: KubeFedCluster
metadata:
  name: on-prem-cluster
  namespace: kube-federation-system
spec:
  apiEndpoint: https://onprem.example.com:6443
  secretRef:
    name: onprem-cluster-secret
  caBundle: <base64-encoded-ca-cert>
---
# Federated training job
apiVersion: types.kubefed.io/v1beta1
kind: FederatedDeployment
metadata:
  name: distributed-training
  namespace: ai-training
spec:
  template:
    metadata:
      labels:
        app: distributed-training
    spec:
      replicas: 4
      selector:
        matchLabels:
          app: distributed-training
      template:
        metadata:
          labels:
            app: distributed-training
        spec:
          containers:
          - name: trainer
            image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
            resources:
              requests:
                nvidia.com/gpu: "4"
                memory: "32Gi"
              limits:
                nvidia.com/gpu: "4"
                memory: "32Gi"
  placement:
    clusters:
    - name: on-prem-cluster
      replicas: 2
    - name: cloud-cluster
      replicas: 2
  overrides:
  - clusterName: on-prem-cluster
    clusterOverride:
      spec:
        template:
          spec:
            containers:
            - name: trainer
              resources:
                requests:
                  nvidia.com/gpu: "4"
                  memory: "32Gi"
  - clusterName: cloud-cluster
    clusterOverride:
      spec:
        template:
          spec:
            containers:
            - name: trainer
              resources:
                requests:
                  nvidia.com/gpu: "4"
                  memory: "32Gi"
```

---

## 💡 Case Study: Ray-based Elastic Training Cluster

### Complete Implementation

🔴 Advanced

```python
# Ray cluster configuration for elastic training
import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.data import Dataset
import json

# Ray cluster configuration
ray_config = {
    "cluster": {
        "provider": {
            "type": "kubernetes",
            "namespace": "ray-jobs",
            "name": "elastic-training-cluster"
        },
        "available_node_types": {
            "head_node": {
                "node_config": {
                    "apiVersion": "v1",
                    "kind": "Pod",
                    "spec": {
                        "containers": [{
                            "name": "ray-head",
                            "image": "rayproject/ray:2.7.0",
                            "resources": {
                                "requests": {"cpu": "4", "memory": "8Gi"},
                                "limits": {"cpu": "8", "memory": "16Gi"}
                            }
                        }],
                        "nodeSelector": {"node-type": "head"}
                    }
                },
                "resources": {"CPU": 4},
                "min_workers": 1,
                "max_workers": 1
            },
            "worker_gpu": {
                "node_config": {
                    "apiVersion": "v1",
                    "kind": "Pod",
                    "spec": {
                        "containers": [{
                            "name": "ray-worker",
                            "image": "rayproject/ray:2.7.0-gpu",
                            "resources": {
                                "requests": {
                                    "cpu": "4",
                                    "memory": "16Gi",
                                    "nvidia.com/gpu": "4"
                                },
                                "limits": {
                                    "cpu": "8",
                                    "memory": "32Gi",
                                    "nvidia.com/gpu": "4"
                                }
                            }
                        }],
                        "nodeSelector": {"node-type": "gpu-worker"}
                    }
                },
                "resources": {"CPU": 4, "GPU": 4},
                "min_workers": 2,
                "max_workers": 10
            }
        }
    }
}

# Elastic training function
def elastic_train(config):
    """Training function that adapts to available resources."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # Get current scaling info
    context = ray.train.get_context()
    world_size = context.get_world_size()
    rank = context.get_world_rank()
    
    # Adjust batch size based on world size
    batch_size = config["base_batch_size"] * world_size
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(
        config["data_path"],
        transform=transform
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, config["num_classes"])
    )
    
    model = ray.train.torch.prepare_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        sampler.set_epoch(epoch)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                ray.train.report({
                    "loss": total_loss / (batch_idx + 1),
                    "accuracy": 100. * correct / total,
                    "epoch": epoch,
                    "world_size": world_size,
                    "batch_size": batch_size
                })

# Create trainer with elastic scaling
trainer = TorchTrainer(
    train_loop_per_worker=elastic_train,
    train_loop_config={
        "base_batch_size": 32,
        "lr": 0.001,
        "epochs": 100,
        "num_classes": 1000,
        "data_path": "/data/training"
    },
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"CPU": 4, "GPU": 1}
    )
)

# Run with elastic scaling
result = trainer.fit()
```

---

## 📝 Exercises

### Exercise 15.1: Spark ML Pipeline
Create a Spark application that:
1. Reads training data from S3/ADLS
2. Performs feature engineering with 5+ transformations
3. Trains a distributed ML model
4. Evaluates model performance
5. Writes predictions back to data lake

### Exercise 15.2: Ray Distributed Training
Implement distributed training with Ray that:
1. Handles node failures gracefully
2. Adjusts batch size based on available resources
3. Implements gradient accumulation across nodes
4. Supports mixed precision training
5. Logs metrics to external monitoring system

### Exercise 15.3: Elastic Scaling
Design an elastic scaling system that:
1. Scales GPU workers based on training queue depth
2. Uses spot instances for non-critical training
3. Implements checkpoint-based recovery
4. Maintains minimum resource guarantees
5. Optimizes cost while meeting training deadlines

---

## ⚠️ Warnings

1. **Data Locality**: Ensure data is co-located with compute nodes to minimize network transfer overhead.
2. **Checkpoint Strategy**: Always implement checkpointing for long-running distributed jobs. Node failures are common in large clusters.
3. **Communication Overhead**: As cluster size increases, communication overhead grows quadratically. Consider using gradient compression.
4. **Cost Management**: Cloud bursting can lead to unexpected costs. Set strict budget limits and monitoring.

---

## Summary

This chapter covered distributed computing architectures for scaling AI workloads beyond single clusters. Key topics include:

1. Spark on Kubernetes for large-scale data processing
2. Ray for distributed training with fault tolerance
3. Dask for parallel computing in Python ecosystems
4. Elastic computing resource management with KEDA
5. Hybrid cloud architecture for flexible scaling

Next, we'll explore AI platform architecture for unified ML lifecycle management.
