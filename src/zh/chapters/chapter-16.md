# 第16章：AI 平台架构

🟢 入门 | 🟡 中级 | 🔴 高级 | ⚫ 管理者

---

## 16.1 统一 AI 平台设计

### 平台愿景

```
┌─────────────────────────────────────────────────────────────────┐
│                统一 AI 平台架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  用户界面层                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Web UI  │  │  CLI     │  │  API     │  │SDK   │  │   │
│  │  │仪表板    │  │  工具    │  │  网关    │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                平台服务层                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │实验跟踪  │  │  模型    │  │ 管道     │  │数据  │  │   │
│  │  │          │  │  注册表  │  │ 编排     │  │管理  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │特征存储  │  │  模型    │  │训练      │  │监控  │  │   │
│  │  │          │  │  服务    │  │ 服务     │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              基础设施层                                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │Kubernetes│  │  GPU     │  │ 存储     │  │网络  │  │   │
│  │  │集群      │  │  池      │  │  系统    │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件

📌 **关键概念**：统一 AI 平台提供从数据摄取到模型部署和监控的端到端 ML 生命周期管理。

```yaml
# 平台核心组件
platform_components:
  data_layer:
    - name: "数据湖"
      technology: "S3/ADLS/GCS"
      purpose: "原始数据存储"
    - name: "特征存储"
      technology: "Feast/Tecton"
      purpose: "特征管理"
    - name: "数据版本控制"
      technology: "DVC/Pachyderm"
      purpose: "数据血缘"
  
  training_layer:
    - name: "实验跟踪"
      technology: "MLflow/W&B"
      purpose: "实验管理"
    - name: "训练服务"
      technology: "Kubeflow/Ray"
      purpose: "分布式训练"
    - name: "超参数调优"
      technology: "Optuna/Ray Tune"
      purpose: "HPO"
  
  serving_layer:
    - name: "模型注册表"
      technology: "MLflow Registry"
      purpose: "模型版本控制"
    - name: "模型服务"
      technology: "KServe/Seldon"
      purpose: "推理"
    - name: "批量推理"
      technology: "Spark/Flink"
      purpose: "离线预测"
  
  monitoring_layer:
    - name: "模型监控"
      technology: "Evidently/WhyLabs"
      purpose: "数据/模型漂移"
    - name: "基础设施监控"
      technology: "Prometheus/Grafana"
      purpose: "系统指标"
    - name: "告警"
      technology: "Alertmanager/PagerDuty"
      purpose: "事件响应"
```

---

## 16.2 多模型管理

### 模型注册表架构

```
┌─────────────────────────────────────────────────────────────────┐
│                 模型注册表架构                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    模型注册表                             │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │                 模型存储                         │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │   │   │
│  │  │  │模型 A    │  │模型 B    │  │模型 C    │     │   │   │
│  │  │  │v1.0.0   │  │v2.1.0   │  │v1.5.0   │     │   │   │
│  │  │  │Staging   │  │Production│  │Archived  │     │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘     │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              模型元数据                         │   │   │
│  │  │  • 模型名称、版本、描述                          │   │   │
│  │  │  • 训练指标（准确率、损失等）                    │   │   │
│  │  │  • 超参数                                        │   │   │
│  │  │  • 数据血缘                                      │   │   │
│  │  │  • 依赖项（框架、CUDA 版本）                     │   │   │
│  │  │  • 制品（权重、配置、分词器）                    │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MLflow 模型注册表设置

```yaml
# Kubernetes 上的 MLflow 部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: ai-platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
      - name: mlflow
        image: mlflow/mlflow:2.8.0
        command:
        - mlflow
        - server
        - --host=0.0.0.0
        - --port=5000
        - --backend-store-uri=postgresql://mlflow:password@mlflow-db:5432/mlflow
        - --default-artifact-root=s3://mlflow-artifacts/
        - --serve-artifacts
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret-key
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
# 用于 MLflow 后端的 PostgreSQL
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mlflow-db
  namespace: ai-platform
spec:
  serviceName: mlflow-db
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-db
  template:
    metadata:
      labels:
        app: mlflow-db
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: mlflow
        - name: POSTGRES_USER
          value: mlflow
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mlflow-db-secret
              key: password
        ports:
        - containerPort: 5432
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
```

### 模型版本管理工作流

```python
# MLflow 模型注册工作流
import mlflow
from mlflow.tracking import MlflowClient

# 初始化 MLflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
client = MlflowClient()

def register_model(model_path, model_name, metrics, params):
    """在 MLflow 中注册新模型版本。"""
    
    # 记录模型
    model_info = mlflow.sklearn.log_model(
        model_path,
        artifact_path="model",
        registered_model_name=model_name
    )
    
    # 记录指标
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    
    # 记录参数
    for key, value in params.items():
        mlflow.log_param(key, value)
    
    # 获取模型版本
    model_versions = client.search_model_versions(
        f"name='{model_name}'"
    )
    
    latest_version = max(
        [int(v.version) for v in model_versions]
    )
    
    return latest_version

def promote_model(model_name, version, stage):
    """将模型提升到新阶段。"""
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    # 添加描述
    client.update_model_version(
        name=model_name,
        version=version,
        description=f"模型已提升至 {stage}"
    )

def compare_models(model_name, version1, version2):
    """比较两个模型版本。"""
    v1 = client.get_model_version(model_name, version1)
    v2 = client.get_model_version(model_name, version2)
    
    # 获取运行信息
    run1 = client.get_run(v1.run_id)
    run2 = client.get_run(v2.run_id)
    
    comparison = {
        "version1": {
            "version": version1,
            "metrics": run1.data.metrics,
            "params": run1.data.params
        },
        "version2": {
            "version": version2,
            "metrics": run2.data.metrics,
            "params": run2.data.params
        }
    }
    
    return comparison
```

---

## 16.3 工作流编排

### Kubeflow Pipelines 架构

```
┌─────────────────────────────────────────────────────────────────┐
│              Kubeflow Pipelines 架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  管道组件                                 │   │
│  │                                                         │   │
│  │  ┌──────────┐     ┌──────────┐     ┌──────────┐       │   │
│  │  │  数据    │────→│  训练    │────→│  评估    │       │   │
│  │  │  处理    │     │          │     │          │       │   │
│  │  └──────────┘     └──────────┘     └──────────┘       │   │
│  │       │               │                  │             │   │
│  │       ▼               ▼                  ▼             │   │
│  │  ┌──────────┐     ┌──────────┐     ┌──────────┐       │   │
│  │  │  特征    │     │  超参数  │     │  模型    │       │   │
│  │  │  存储    │     │  调优    │     │  注册表  │       │   │
│  │  └──────────┘     └──────────┘     └──────────┘       │   │
│  │                                            │           │   │
│  │                                            ▼           │   │
│  │                                      ┌──────────┐     │   │
│  │                                      │  部署    │     │   │
│  │                                      │          │     │   │
│  │                                      └──────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 管道定义示例

```python
# 使用 Kubeflow 的完整 ML 管道
import kfp
from kfp import dsl
from kfp.components import load_component_from_file

# 加载组件定义
data_processing = load_component_from_file('components/data_processing.yaml')
feature_engineering = load_component_from_file('components/feature_engineering.yaml')
model_training = load_component_from_file('components/model_training.yaml')
model_evaluation = load_component_from_file('components/model_evaluation.yaml')
model_deployment = load_component_from_file('components/model_deployment.yaml')

@dsl.pipeline(
    name='ML 训练管道',
    description='用于图像分类的端到端 ML 管道',
    pipeline_root='gs://my-bucket/pipelines'
)
def ml_pipeline(
    dataset_path: str,
    model_name: str = 'image_classifier',
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    deploy_threshold: float = 0.85
):
    # 步骤 1：数据处理
    data_task = data_processing(
        input_path=dataset_path,
        output_path='/tmp/processed_data'
    )
    
    # 步骤 2：特征工程
    feature_task = feature_engineering(
        input_data=data_task.outputs['output_path'],
        output_path='/tmp/features'
    )
    
    # 步骤 3：使用超参数调优进行模型训练
    training_task = model_training(
        train_data=feature_task.outputs['output_path'],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # 步骤 4：模型评估
    eval_task = model_evaluation(
        model=training_task.outputs['model'],
        test_data=feature_task.outputs['output_path'],
        metrics=['accuracy', 'precision', 'recall', 'f1']
    )
    
    # 步骤 5：条件部署
    with dsl.Condition(eval_task.outputs['accuracy'] > deploy_threshold):
        deploy_task = model_deployment(
            model=training_task.outputs['model'],
            model_name=model_name,
            serving_config={
                'replicas': 2,
                'resources': {
                    'cpu': '2',
                    'memory': '4Gi',
                    'gpu': '1'
                }
            }
        )

# 编译管道
compiler = kfp.compiler.Compiler()
compiler.compile(ml_pipeline, 'ml_pipeline.yaml')
```

---

## 16.4 自助式 AI 服务

### 开发者门户架构

```
┌─────────────────────────────────────────────────────────────────┐
│              自助式 AI 平台                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  开发者门户                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  项目    │  │  资源    │  │ 管道     │  │部署  │  │   │
│  │  │  模板    │  │  配置    │  │ 构建器   │  │向导  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  自助式 API                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │命名空间  │  │  GPU     │  │ 训练     │  │模型  │  │   │
│  │  │API       │  │  API     │  │  API     │  │API   │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  治理层                                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  预算    │  │  访问    │  │  策略    │  │审计  │  │   │
│  │  │  控制    │  │  控制    │  │  引擎    │  │日志  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 资源配置 API

```yaml
# 自助式命名空间配置
apiVersion: ai.platform.example.com/v1
kind: ProjectNamespace
metadata:
  name: ml-project-alpha
  labels:
    team: data-science
    environment: production
spec:
  owner: john.doe@company.com
  team: data-science
  description: "用于客户流失预测的 ML 项目"
  resources:
    quotas:
      cpu: "32"
      memory: "64Gi"
      gpu: "4"
    limits:
      cpu: "64"
      memory: "128Gi"
      gpu: "8"
  access:
    mlEngineers:
      - alice@company.com
      - bob@company.com
    dataScientists:
      - charlie@company.com
    viewers:
      - manager@company.com
  budget:
    monthly_limit: 5000
    alert_threshold: 80
  templates:
    - name: "training"
      enabled: true
      default_resources:
        cpu: "8"
        memory: "16Gi"
        gpu: "2"
    - name: "inference"
      enabled: true
      default_resources:
        cpu: "4"
        memory: "8Gi"
        gpu: "1"
```

---

## 16.5 平台治理与合规

### 治理框架

```
┌─────────────────────────────────────────────────────────────────┐
│              平台治理框架                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  策略引擎 (OPA)                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  访问    │  │  预算    │  │  安全    │  │合规  │  │   │
│  │  │  策略    │  │  策略    │  │  策略    │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  审计与合规                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  审计    │  │  数据    │  │  模型    │  │访问  │  │   │
│  │  │  日志    │  │  血缘    │  │  血缘    │  │日志  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### AI 平台的 OPA 策略

```rego
# 模型部署的 OPA 策略
package ai.platform.model_deployment

default allow = false

# 如果满足所有条件则允许部署
allow {
    # 模型必须处于生产阶段
    input.model.stage == "production"
    
    # 模型必须通过准确率阈值
    input.model.metrics.accuracy >= input.thresholds.min_accuracy
    
    # 部署不得超过资源限制
    input.deployment.resources.gpu <= input.limits.max_gpu
    input.deployment.resources.memory <= input.limits.max_memory
    
    # 部署者必须具有所需权限
    has_permission(input.user, "deploy")
    
    # 无安全违规
    not has_security_violation(input.model)
}

# 检查用户权限
has_permission(user, action) {
    permission := data.permissions[user][_]
    permission.action == action
    permission.resource == "model"
}

# 检查安全违规
has_security_violation(model) {
    # 检查模型中的敏感数据
    model.contains_pii == true
}

has_security_violation(model) {
    # 检查已知漏洞
    model.vulnerabilities[_].severity == "critical"
}

# 预算检查
allow {
    # 检查部署是否在预算内
    input.project.budget.remaining >= estimated_cost(input.deployment)
    
    # 检查项目是否未被冻结
    input.project.status != "frozen"
}

estimated_cost(deployment) = cost {
    cost := deployment.replicas * deployment.resources.gpu * data.gpu_hourly_rate
}
```

---

## 💡 案例研究：企业级 AI 平台架构设计

### 完整平台设计

🔴 高级

```yaml
# 企业级 AI 平台 - 完整架构
apiVersion: v1
kind: Namespace
metadata:
  name: ai-platform
  labels:
    istio-injection: enabled
---
# 平台访问的 Istio Gateway
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: ai-platform-gateway
  namespace: ai-platform
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: ai-platform-tls
    hosts:
    - ai-platform.company.com
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - ai-platform.company.com
    tls:
      httpsRedirect: true
---
# 路由的 Virtual Service
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ai-platform-routes
  namespace: ai-platform
spec:
  hosts:
  - ai-platform.company.com
  gateways:
  - ai-platform-gateway
  http:
  - match:
    - uri:
        prefix: /api/v1
    route:
    - destination:
        host: api-gateway
        port:
          number: 8080
  - match:
    - uri:
        prefix: /mlflow
    route:
    - destination:
        host: mlflow-server
        port:
          number: 5000
  - match:
    - uri:
        prefix: /jupyter
    route:
    - destination:
        host: jupyter-hub
        port:
          number: 8000
```

### 平台监控栈

```yaml
# 用于平台监控的 Prometheus
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: ai-platform-prometheus
  namespace: monitoring
spec:
  replicas: 3
  retention: 90d
  resources:
    requests:
      memory: "8Gi"
      cpu: "4"
  storage:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 500Gi
  serviceMonitorSelector:
    matchLabels:
      team: ai-platform
  ruleSelector:
    matchLabels:
      team: ai-platform
---
# 用于 ML 监控的自定义指标
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-monitoring-rules
  namespace: monitoring
spec:
  groups:
  - name: ml-model-alerts
    rules:
    - alert: ModelAccuracyDrop
      expr: |
        ml_model_accuracy{job="model-serving"} < 0.8
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "模型准确率降至阈值以下"
        
    - alert: HighLatency
      expr: |
        histogram_quantile(0.99, rate(ml_inference_duration_seconds_bucket[5m])) > 1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "模型推理延迟过高"
        
    - alert: DataDriftDetected
      expr: |
        ml_data_drift_score{job="drift-detector"} > 0.3
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "在模型输入中检测到数据漂移"
```

---

## 📝 练习

### 练习 16.1：平台设计
设计支持以下功能的 AI 平台：
1. 5 个数据科学团队
2. 20+ 生产中的 ML 模型
3. 每周 100+ 实验
4. 自动化模型再训练
5. 模型部署的 A/B 测试

### 练习 16.2：工作流自动化
实现自动化 ML 工作流，要求：
1. 在新数据到达时触发
2. 执行特征工程
3. 训练多个模型变体
4. 自动选择最佳模型
5. 使用金丝雀发布进行部署
6. 监控模型性能

---

## ⚠️ 警告

1. **平台复杂性**：不要一开始就构建所有内容。从核心组件开始，迭代扩展。
2. **供应商锁定**：优先选择开源标准而非专有解决方案。设计时考虑可移植性。
3. **成本升级**：密切监控平台使用情况。共享基础设施可能导致成本超支。
4. **安全性**：实现纵深防御。AI 平台处理敏感数据和模型。

---

## 本章小结

本章介绍了用于统一 ML 生命周期管理的 AI 平台架构。关键主题包括：

1. 具有清晰层级分离的统一平台设计
2. 具有适当版本控制和治理的多模型管理
3. 使用 Kubeflow Pipelines 进行工作流编排
4. 为数据科学团队提供自助式功能
5. 平台治理与合规自动化

下一章我们将探讨边缘 AI 基础和部署策略。
