# Chapter 16: AI Platform Architecture

🟢 Beginner | 🟡 Intermediate | 🔴 Advanced | ⚫ Manager

---

## 16.1 Unified AI Platform Design

### Platform Vision

```
┌─────────────────────────────────────────────────────────────────┐
│                Unified AI Platform Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  User Interface Layer                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Web UI  │  │  CLI     │  │  API     │  │SDK   │  │   │
│  │  │Dashboard │  │  Tool    │  │  Gateway │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Platform Services Layer                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │Experiment│  │  Model   │  │ Pipeline │  │Data  │  │   │
│  │  │Tracking  │  │ Registry │  │ Orch.    │ │Mgmt  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │Feature   │  │  Model   │  │Training  │ │Monitor│  │   │
│  │  │Store     │  │ Serving  │  │ Service  │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Infrastructure Layer                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │Kubernetes│  │  GPU     │  │ Storage  │  │Network│  │   │
│  │  │Cluster   │  │  Pool    │  │  System  │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

📌 **Key Concept**: A unified AI platform provides end-to-end ML lifecycle management from data ingestion to model deployment and monitoring.

```yaml
# Platform core components
platform_components:
  data_layer:
    - name: "Data Lake"
      technology: "S3/ADLS/GCS"
      purpose: "Raw data storage"
    - name: "Feature Store"
      technology: "Feast/Tecton"
      purpose: "Feature management"
    - name: "Data Versioning"
      technology: "DVC/Pachyderm"
      purpose: "Data lineage"
  
  training_layer:
    - name: "Experiment Tracking"
      technology: "MLflow/W&B"
      purpose: "Experiment management"
    - name: "Training Service"
      technology: "Kubeflow/Ray"
      purpose: "Distributed training"
    - name: "Hyperparameter Tuning"
      technology: "Optuna/Ray Tune"
      purpose: "HPO"
  
  serving_layer:
    - name: "Model Registry"
      technology: "MLflow Registry"
      purpose: "Model versioning"
    - name: "Model Serving"
      technology: "KServe/Seldon"
      purpose: "Inference"
    - name: "Batch Inference"
      technology: "Spark/Flink"
      purpose: "Offline predictions"
  
  monitoring_layer:
    - name: "Model Monitoring"
      technology: "Evidently/WhyLabs"
      purpose: "Data/model drift"
    - name: "Infrastructure Monitoring"
      technology: "Prometheus/Grafana"
      purpose: "System metrics"
    - name: "Alerting"
      technology: "Alertmanager/PagerDuty"
      purpose: "Incident response"
```

---

## 16.2 Multi-model Management

### Model Registry Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Model Registry Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Model Registry                       │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │                 Model Store                     │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │   │   │
│  │  │  │Model A   │  │Model B   │  │Model C   │     │   │   │
│  │  │  │v1.0.0   │  │v2.1.0   │  │v1.5.0   │     │   │   │
│  │  │  │Staging   │  │Production│  │Archived  │     │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘     │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              Model Metadata                     │   │   │
│  │  │  • Model name, version, description             │   │   │
│  │  │  • Training metrics (accuracy, loss, etc.)      │   │   │
│  │  │  • Hyperparameters                              │   │   │
│  │  │  • Data lineage                                 │   │   │
│  │  │  • Dependencies (framework, CUDA version)       │   │   │
│  │  │  • Artifacts (weights, config, tokenizer)       │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MLflow Model Registry Setup

```yaml
# MLflow deployment on Kubernetes
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
        volumeMounts:
        - name: mlflow-data
          mountPath: /mlflow
      volumes:
      - name: mlflow-data
        persistentVolumeClaim:
          claimName: mlflow-pvc
---
# PostgreSQL for MLflow backend
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
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
```

### Model Versioning Workflow

```python
# MLflow model registration workflow
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ModelVersion

# Initialize MLflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
client = MlflowClient()

def register_model(model_path, model_name, metrics, params):
    """Register a new model version in MLflow."""
    
    # Log model
    model_info = mlflow.sklearn.log_model(
        model_path,
        artifact_path="model",
        registered_model_name=model_name
    )
    
    # Log metrics
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    
    # Log params
    for key, value in params.items():
        mlflow.log_param(key, value)
    
    # Get model version
    model_versions = client.search_model_versions(
        f"name='{model_name}'"
    )
    
    latest_version = max(
        [int(v.version) for v in model_versions]
    )
    
    return latest_version

def promote_model(model_name, version, stage):
    """Promote model to a new stage."""
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    # Add description
    client.update_model_version(
        name=model_name,
        version=version,
        description=f"Model promoted to {stage}"
    )

def compare_models(model_name, version1, version2):
    """Compare two model versions."""
    v1 = client.get_model_version(model_name, version1)
    v2 = client.get_model_version(model_name, version2)
    
    # Get run info
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

## 16.3 Workflow Orchestration

### Kubeflow Pipelines Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Kubeflow Pipelines Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Pipeline Components                    │   │
│  │                                                         │   │
│  │  ┌──────────┐     ┌──────────┐     ┌──────────┐       │   │
│  │  │  Data    │────→│ Training │────→│ Evaluation│       │   │
│  │  │Processing│     │          │     │          │       │   │
│  │  └──────────┘     └──────────┘     └──────────┘       │   │
│  │       │               │                  │             │   │
│  │       ▼               ▼                  ▼             │   │
│  │  ┌──────────┐     ┌──────────┐     ┌──────────┐       │   │
│  │  │ Feature  │     │ Hyper-   │     │ Model    │       │   │
│  │  │ Store    │     │ param    │     │ Registry │       │   │
│  │  │          │     │ Tuning   │     │          │       │   │
│  │  └──────────┘     └──────────┘     └──────────┘       │   │
│  │                                            │           │   │
│  │                                            ▼           │   │
│  │                                      ┌──────────┐     │   │
│  │                                      │Deployment│     │   │
│  │                                      │          │     │   │
│  │                                      └──────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Definition Example

```python
# Complete ML Pipeline with Kubeflow
import kfp
from kfp import dsl
from kfp.components import load_component_from_file

# Load component definitions
data_processing = load_component_from_file('components/data_processing.yaml')
feature_engineering = load_component_from_file('components/feature_engineering.yaml')
model_training = load_component_from_file('components/model_training.yaml')
model_evaluation = load_component_from_file('components/model_evaluation.yaml')
model_deployment = load_component_from_file('components/model_deployment.yaml')

@dsl.pipeline(
    name='ML Training Pipeline',
    description='End-to-end ML pipeline for image classification',
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
    # Step 1: Data Processing
    data_task = data_processing(
        input_path=dataset_path,
        output_path='/tmp/processed_data'
    )
    
    # Step 2: Feature Engineering
    feature_task = feature_engineering(
        input_data=data_task.outputs['output_path'],
        output_path='/tmp/features'
    )
    
    # Step 3: Model Training with Hyperparameter Tuning
    training_task = model_training(
        train_data=feature_task.outputs['output_path'],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # Step 4: Model Evaluation
    eval_task = model_evaluation(
        model=training_task.outputs['model'],
        test_data=feature_task.outputs['output_path'],
        metrics=['accuracy', 'precision', 'recall', 'f1']
    )
    
    # Step 5: Conditional Deployment
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

# Compile pipeline
compiler = kfp.compiler.Compiler()
compiler.compile(ml_pipeline, 'ml_pipeline.yaml')
```

---

## 16.4 Self-service AI

### Developer Portal Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Self-service AI Platform                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Developer Portal                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Project │  │  Resource│  │ Pipeline │  │Deploy│  │   │
│  │  │ Template │  │Provision │  │ Builder  │  │ Wizard│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Self-service APIs                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │Namespace │  │  GPU     │  │ Training │  │Model │  │   │
│  │  │API       │  │  API     │  │  API     │  │API   │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Governance Layer                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Budget  │  │  Access  │  │  Policy  │  │Audit │  │   │
│  │  │ Control  │  │ Control  │  │  Engine  │  │ Log  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Resource Provisioning API

```yaml
# Self-service namespace provisioning
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
  description: "ML project for customer churn prediction"
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

### Kubernetes Operator for Self-service

```go
// Custom operator for ML project provisioning
package main

import (
    "context"
    "fmt"
    "os"
    "time"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/log"
)

type ProjectNamespaceSpec struct {
    Owner       string            `json:"owner"`
    Team        string            `json:"team"`
    Description string            `json:"description"`
    Resources   ResourcesSpec     `json:"resources"`
    Access      AccessSpec        `json:"access"`
    Budget      BudgetSpec        `json:"budget"`
}

type ProjectNamespace struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`
    Spec              ProjectNamespaceSpec `json:"spec"`
    Status            ProjectStatus        `json:"status"`
}

type ProjectStatus struct {
    Phase      string      `json:"phase"`
    Conditions []Condition `json:"conditions"`
    LastUpdated *metav1.Time `json:"lastUpdated,omitempty"`
}

type ProjectNamespaceReconciler struct {
    client.Client
    Scheme *runtime.Scheme
}

func (r *ProjectNamespaceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    // Fetch the ProjectNamespace instance
    var project ProjectNamespace
    if err := r.Get(ctx, req.NamespacedName, &project); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }

    // Check if namespace exists
    var ns corev1.Namespace
    if err := r.Get(ctx, client.ObjectKey{Name: project.Name}, &ns); err != nil {
        // Create namespace
        ns = corev1.Namespace{
            ObjectMeta: metav1.ObjectMeta{
                Name: project.Name,
                Labels: map[string]string{
                    "team":        project.Spec.Team,
                    "owner":       project.Spec.Owner,
                    "managed-by":  "ai-platform-operator",
                },
            },
        }
        if err := r.Create(ctx, &ns); err != nil {
            return ctrl.Result{}, err
        }
        log.Info("Created namespace", "name", project.Name)
    }

    // Create ResourceQuota
    var quota corev1.ResourceQuota
    quotaName := project.Name + "-quota"
    if err := r.Get(ctx, client.ObjectKey{Name: quotaName, Namespace: project.Name}, &quota); err != nil {
        quota = corev1.ResourceQuota{
            ObjectMeta: metav1.ObjectMeta{
                Name:      quotaName,
                Namespace: project.Name,
            },
            Spec: corev1.ResourceQuotaSpec{
                Hard: corev1.ResourceList{
                    "requests.cpu":    resource.MustParse(project.Spec.Resources.Quotas.CPU),
                    "requests.memory": resource.MustParse(project.Spec.Resources.Quotas.Memory),
                    "limits.cpu":      resource.MustParse(project.Spec.Resources.Limits.CPU),
                    "limits.memory":   resource.MustParse(project.Spec.Resources.Limits.Memory),
                },
            },
        }
        if err := r.Create(ctx, &quota); err != nil {
            return ctrl.Result{}, err
        }
        log.Info("Created resource quota", "name", quotaName)
    }

    // Update status
    project.Status.Phase = "Active"
    project.Status.LastUpdated = &metav1.Time{Time: time.Now()}
    if err := r.Status().Update(ctx, &project); err != nil {
        return ctrl.Result{}, err
    }

    return ctrl.Result{RequeueAfter: 5 * time.Minute}, nil
}

func main() {
    mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{})
    if err != nil {
        os.Exit(1)
    }

    if err = (&ProjectNamespaceReconciler{
        Client: mgr.GetClient(),
        Scheme: mgr.GetScheme(),
    }).SetupWithManager(mgr); err != nil {
        os.Exit(1)
    }

    if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
        os.Exit(1)
    }
}
```

---

## 16.5 Platform Governance & Compliance

### Governance Framework

```
┌─────────────────────────────────────────────────────────────────┐
│              Platform Governance Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Policy Engine (OPA)                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Access  │  │  Budget  │  │  Security│  │Compli│  │   │
│  │  │  Policies│  │  Policies│  │  Policies│  │ance  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Audit & Compliance                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Audit   │  │  Data    │  │  Model   │  │Access│  │   │
│  │  │  Logs    │  │  Lineage │  │  Lineage │  │Logs  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### OPA Policies for AI Platform

```rego
# OPA Policy for model deployment
package ai.platform.model_deployment

default allow = false

# Allow deployment if all conditions are met
allow {
    # Model must be in production stage
    input.model.stage == "production"
    
    # Model must pass accuracy threshold
    input.model.metrics.accuracy >= input.thresholds.min_accuracy
    
    # Deployment must not exceed resource limits
    input.deployment.resources.gpu <= input.limits.max_gpu
    input.deployment.resources.memory <= input.limits.max_memory
    
    # Deployer must have required permissions
    has_permission(input.user, "deploy")
    
    # No security violations
    not has_security_violation(input.model)
}

# Check user permissions
has_permission(user, action) {
    permission := data.permissions[user][_]
    permission.action == action
    permission.resource == "model"
}

# Check for security violations
has_security_violation(model) {
    # Check for sensitive data in model
    model.contains_pii == true
}

has_security_violation(model) {
    # Check for known vulnerabilities
    model.vulnerabilities[_].severity == "critical"
}

# Budget check
allow {
    # Check if deployment is within budget
    input.project.budget.remaining >= estimated_cost(input.deployment)
    
    # Check if project is not frozen
    input.project.status != "frozen"
}

estimated_cost(deployment) = cost {
    cost := deployment.replicas * deployment.resources.gpu * data.gpu_hourly_rate
}
```

### Compliance Automation

```yaml
# Compliance check job
apiVersion: batch/v1
kind: CronJob
metadata:
  name: compliance-checker
  namespace: ai-platform
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: compliance-checker
            image: ai-platform/compliance-checker:latest
            command:
            - python
            - -m
            - compliance.checker
            env:
            - name: CHECK_TYPES
              value: "data_privacy,model_fairness,security,audit"
            - name: ALERT_WEBHOOK
              valueFrom:
                secretKeyRef:
                  name: compliance-secrets
                  key: webhook-url
            resources:
              requests:
                memory: "1Gi"
                cpu: "500m"
          restartPolicy: OnFailure
```

---

## 💡 Case Study: Enterprise AI Platform Architecture

### Complete Platform Design

🔴 Advanced

```yaml
# Enterprise AI Platform - Complete Architecture
apiVersion: v1
kind: Namespace
metadata:
  name: ai-platform
  labels:
    istio-injection: enabled
---
# Istio Gateway for platform access
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
# Virtual Service for routing
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

### Platform Monitoring Stack

```yaml
# Prometheus for platform monitoring
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
# Custom metrics for ML monitoring
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
        summary: "Model accuracy dropped below threshold"
        
    - alert: HighLatency
      expr: |
        histogram_quantile(0.99, rate(ml_inference_duration_seconds_bucket[5m])) > 1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Model inference latency is too high"
        
    - alert: DataDriftDetected
      expr: |
        ml_data_drift_score{job="drift-detector"} > 0.3
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Data drift detected in model inputs"
```

---

## 📝 Exercises

### Exercise 16.1: Platform Design
Design an AI platform that supports:
1. 5 data science teams
2. 20+ ML models in production
3. 100+ experiments per week
4. Automated model retraining
5. A/B testing for model deployment

### Exercise 16.2: Workflow Automation
Implement an automated ML workflow that:
1. Triggers on new data arrival
2. Performs feature engineering
3. Trains multiple model variants
4. Selects the best model automatically
5. Deploys with canary release
6. Monitors model performance

### Exercise 16.3: Governance Implementation
Implement governance policies that:
1. Enforce resource quotas per team
2. Require approval for production deployments
3. Track model lineage and data provenance
4. Generate compliance reports
5. Handle model retirement gracefully

---

## ⚠️ Warnings

1. **Platform Complexity**: Don't build everything at start. Begin with core components and expand iteratively.
2. **Vendor Lock-in**: Prefer open-source standards over proprietary solutions. Design for portability.
3. **Cost Escalation**: Monitor platform usage closely. Shared infrastructure can lead to cost overruns.
4. **Security**: Implement defense in depth. AI platforms handle sensitive data and models.

---

## Summary

This chapter covered AI platform architecture for unified ML lifecycle management. Key topics include:

1. Unified platform design with clear layer separation
2. Multi-model management with proper versioning and governance
3. Workflow orchestration with Kubeflow Pipelines
4. Self-service capabilities for data science teams
5. Platform governance and compliance automation

Next, we'll explore Edge AI fundamentals and deployment strategies.
