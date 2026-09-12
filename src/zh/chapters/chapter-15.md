# 第15章：分布式计算架构

🟢 入门 | 🟡 中级 | 🔴 高级 | ⚫ 管理者

---

## 15.1 Spark on Kubernetes

### 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│              Spark on Kubernetes 架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Spark 应用程序                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Driver  │  │ Executor │  │ Executor │  │Executor│  │   │
│  │  │  (Pod)   │→ │  (Pod)   │  │  (Pod)   │  │ (Pod) │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Kubernetes 集群                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ 节点 1   │  │ 节点 2   │  │ 节点 3   │  │节点 4│  │   │
│  │  │8 CPU     │  │8 CPU     │  │8 CPU     │  │8 CPU │  │   │
│  │  │32GB RAM  │  │32GB RAM  │  │32GB RAM  │  │32GB  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Spark Operator 安装

🟡 中级

```bash
# 安装 Spark Operator
helm repo add spark-operator https://kubeflow.github.io/spark-operator
helm repo update

# 使用 RBAC 安装
helm install spark-operator spark-operator/spark-operator \
  --namespace spark-operator \
  --create-namespace \
  --set sparkJobNamespace=spark-jobs \
  --set webhook.enable=true \
  --set webhook.port=8080

# 验证安装
kubectl get pods -n spark-operator
```

### Spark 应用程序定义

```yaml
# 用于 ML 数据处理的 SparkApplication
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
  driver:
    cores: 2
    coreLimit: "4"
    memory: "4g"
    serviceAccount: spark
    volumeMounts:
    - name: spark-driver-pvc
      mountPath: /data
  executor:
    cores: 4
    coreLimit: "4"
    memory: "8g"
    instances: 4
    serviceAccount: spark
    volumeMounts:
    - name: spark-executor-pvc
      mountPath: /data
  dynamicAllocation:
    enabled: true
    initialExecutors: 4
    minExecutors: 2
    maxExecutors: 20
```

### Spark ML 管道作业

```scala
// 特征工程作业
package com.ml

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer}
import org.apache.spark.ml.Pipeline

object FeatureEngineering {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("特征工程管道")
      .config("spark.kubernetes.driver.master", "k8s://https://kubernetes.default.svc:443")
      .getOrCreate()
    
    import spark.implicits._
    
    // 读取原始数据
    val rawData = spark.read.parquet("/data/raw/events")
    
    // 特征工程管道
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
    
    // 保存处理后的特征
    processedData.write.mode("overwrite")
      .partitionBy("date")
      .parquet("/data/processed/features")
    
    spark.stop()
  }
}
```

---

## 15.2 Ray 分布式框架

### Ray 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ray 集群架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   头节点                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  GCS     │  │   Ray    │  │  对象    │  │仪表板│  │   │
│  │  │(全局     │  │  Driver  │  │  存储    │  │      │  │   │
│  │  │ 控制存储)│  │          │  │          │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   工作节点                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ 工作节点1│  │ 工作节点2│  │ 工作节点3│  │工作4 │  │   │
│  │  │┌────────┐│  │┌────────┐│  │┌────────┐│  │┌─────┐│  │   │
│  │  ││Raylet  ││  ││Raylet  ││  ││Raylet  ││  ││Rayl.││  │   │
│  │  │└────────┘│  │└────────┘│  │└────────┘│  │└─────┘│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   GPU 工作节点                           │   │
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

### 使用 KubeRay 在 Kubernetes 上部署 Ray

```yaml
# KubeRay RayCluster 定义
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
  - groupName: gpu-workers
    replicas: 2
    minReplicas: 1
    maxReplicas: 4
    rayStartParams:
      num-cpus: "4"
      num-gpus: "4"
    template:
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
        nodeSelector:
          accelerator: nvidia-tesla-v100
```

### Ray 分布式训练代码

```python
# 使用 Ray Train 进行分布式训练
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 初始化 Ray 集群
ray.init(address="auto")

# 定义训练函数
def train_func(config):
    # 数据加载
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
    
    # 分布式采样器
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
    
    # 模型设置
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
    model = train.torch.prepare_model(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    # 训练循环
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

# 配置训练
config = {
    "data_path": "/data/training",
    "batch_size": 32,
    "lr": 0.001,
    "epochs": 100,
    "num_classes": 1000
}

# 创建训练器
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"CPU": 4, "GPU": 1}
    )
)

# 运行训练
result = trainer.fit()
```

---

## 15.3 Dask 并行计算

### Dask 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dask 架构                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Dask 调度器                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  任务    │  │  图      │  │  工作    │  │客户端│  │   │
│  │  │  队列    │  │  优化器  │  │  管理器  │  │管理  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Dask 工作节点                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ 工作节点1│  │ 工作节点2│  │ 工作节点3│  │工作 4│  │   │
│  │  │┌────────┐│  │┌────────┐│  │┌────────┐│  │┌────┐│  │   │
│  │  ││ 4 CPU  ││  ││ 4 CPU  ││  ││ 4 CPU  ││  ││4CPU││  │   │
│  │  ││ 16GB   ││  ││ 16GB   ││  ││ 16GB   ││  ││16GB││  │   │
│  │  │└────────┘│  │└────────┘│  │└────────┘│  │└────┘│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Dask on Kubernetes 部署

```yaml
# Dask 工作节点的 Kubernetes 部署
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
        volumeMounts:
        - name: dask-storage
          mountPath: /data
      volumes:
      - name: dask-storage
        persistentVolumeClaim:
          claimName: dask-pvc
```

### Dask ML 管道示例

```python
# Dask 分布式 ML 管道
from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import GridSearchCV
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA
from dask_ml.linear_model import LogisticRegression
from dask_ml.pipeline import Pipeline
import dask.array as da
import dask.dataframe as dd

# 连接到 Dask 集群
client = Client("tcp://dask-scheduler:8786")

# 使用 Dask 加载数据
train_data = dd.read_parquet("/data/training/features/*.parquet")
test_data = dd.read_parquet("/data/testing/features/*.parquet")

# 转换为 Dask 数组
X_train = train_data.drop("label", axis=1).to_dask_array(lengths=True)
y_train = train_data["label"].to_dask_array(lengths=True)
X_test = test_data.drop("label", axis=1).to_dask_array(lengths=True)
y_test = test_data["label"].to_dask_array(lengths=True)

# 创建 ML 管道
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=100)),
    ("lr", LogisticRegression(max_iter=1000))
])

# 使用 GridSearchCV 进行超参数调优
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

# 拟合模型
grid_search.fit(X_train, y_train)

# 评估
score = grid_search.score(X_test, y_test)
print(f"测试准确率: {score:.4f}")
print(f"最佳参数: {grid_search.best_params_}")
```

---

## 15.4 弹性计算资源管理

### 动态扩展架构

```
┌─────────────────────────────────────────────────────────────────┐
│            弹性计算资源管理                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   指标收集                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │Prometheus│  │  自定义   │  │  KEDA    │  │HPA   │  │   │
│  │  │指标      │  │  指标     │  │  指标    │  │指标  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   扩展决策                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ 队列深度 │  │  GPU     │  │  内存    │  │成本  │  │   │
│  │  │          │  │  利用率  │  │  压力    │  │预算  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   资源调配                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │云 GPU    │  │ 本地     │  │  Spot    │  │MIG   │  │   │
│  │  │  池      │  │ GPU      │  │  实例    │  │池    │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### KEDA 用于 AI 工作负载扩展

```yaml
# GPU 工作负载扩展的 KEDA ScaledObject
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
      timezone: Asia/Shanghai
      start: 0 8 * * 1-5
      end: 0 20 * * 1-5
      desiredReplicas: "10"
```

---

## 15.5 混合云架构

### 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│              混合云 AI 架构                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   本地数据中心                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │GPU 集群  │  │数据湖    │  │模型      │  │边缘  │  │   │
│  │  │(训练)    │  │          │  │注册表    │  │推理  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │   VPN / 直连      │                      │
│                    │   / SD-WAN        │                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   云提供商 (AWS/Azure/GCP)              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  突发    │  │ 托管     │  │  模型    │  │API   │  │   │
│  │  │  训练    │  │ 服务     │  │  服务    │  │网关  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 多集群联邦

```yaml
# 多集群 AI 的 KubeFed 配置
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
# 联邦训练作业
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

## 💡 案例研究：基于 Ray 的弹性训练集群

### 完整实现

🔴 高级

```python
# 弹性训练的 Ray 集群配置
import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

# Ray 集群配置
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

# 弹性训练函数
def elastic_train(config):
    """根据可用资源自适应的训练函数。"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # 获取当前扩展信息
    context = ray.train.get_context()
    world_size = context.get_world_size()
    rank = context.get_world_rank()
    
    # 根据 world_size 调整批量大小
    batch_size = config["base_batch_size"] * world_size
    
    # 数据加载
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
    
    # 模型
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
    
    # 训练循环
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

# 使用弹性扩展创建训练器
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

# 使用弹性扩展运行
result = trainer.fit()
```

---

## 📝 练习

### 练习 15.1：Spark ML 管道
创建一个 Spark 应用程序，要求：
1. 从 S3/ADLS 读取训练数据
2. 使用 5+ 转换进行特征工程
3. 训练分布式 ML 模型
4. 评估模型性能
5. 将预测结果写回数据湖

### 练习 15.2：Ray 分布式训练
实现 Ray 分布式训练，要求：
1. 优雅处理节点故障
2. 根据可用资源调整批量大小
3. 实现跨节点的梯度累积
4. 支持混合精度训练
5. 将指标记录到外部监控系统

---

## ⚠️ 警告

1. **数据局部性**：确保数据与计算节点共置以最小化网络传输开销。
2. **检查点策略**：始终为长时间运行的分布式作业实现检查点。大集群中节点故障很常见。
3. **通信开销**：随着集群规模增加，通信开销呈二次方增长。考虑使用梯度压缩。
4. **成本管理**：云突发可能导致意外成本。设置严格的预算限制和监控。

---

## 本章小结

本章介绍了用于扩展 AI 工作负载的分布式计算架构。关键主题包括：

1. Spark on Kubernetes 用于大规模数据处理
2. Ray 用于具有容错能力的分布式训练
3. Dask 用于 Python 生态系统中的并行计算
4. 使用 KEDA 进行弹性计算资源管理
5. 混合云架构用于灵活扩展

下一章我们将探讨用于统一 ML 生命周期管理的 AI 平台架构。
