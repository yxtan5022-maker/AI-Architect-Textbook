# Chapter 8: Model Deployment Architecture

> **Part III: MLOps Architecture**

**Learning Objectives:**
- Compare and select appropriate deployment strategies
- Design scalable model serving architectures
- Implement A/B testing and canary deployments
- Manage model versions effectively
- Optimize inference performance
- Plan edge deployment strategies

---

## 8.1 Deployment Strategy Comparison

### 8.1.1 The Deployment Decision Framework

🟢 **Beginner**

Choosing the right deployment strategy is critical. The wrong choice can lead to downtime, poor performance, or costly rollbacks.

```
Deployment Strategy Selection Matrix:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Risk Tolerance:                                            │
│  Low ──────────────────────────────────────────────► High   │
│  │                                                    │    │
│  ▼                                                    ▼    │
│  Blue/Green                              Canary              │
│  Shadow                                  A/B Test            │
│                                                              │
│  Downtime Tolerance:                                         │
│  Zero ──────────────────────────────────────────────► Any   │
│  │                                                    │    │
│  ▼                                                    ▼    │
│  Blue/Green                              Rolling              │
│  Canary                                  Recreate             │
│                                                              │
│  Traffic Split Need:                                         │
│  None ─────────────────────────────────────────────► Full   │
│  │                                                    │    │
│  ▼                                                    ▼    │
│  Recreate                                A/B Test            │
│  Rolling                                 Canary              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.1.2 Deployment Strategies Overview

🟡 **Intermediate**

| Strategy | Downtime | Risk | Complexity | Use Case |
|----------|----------|------|------------|----------|
| **Recreate** | Yes | High | Low | Development, internal tools |
| **Rolling** | No | Medium | Medium | General production |
| **Blue/Green** | No | Low | Medium | Critical services |
| **Canary** | No | Very Low | High | Risk-averse production |
| **Shadow** | No | None | Very High | Pre-production validation |
| **A/B Test** | No | Low | High | Business optimization |

### 8.1.3 Detailed Strategy Descriptions

🔴 **Advanced**

```
Recreate Strategy:
┌─────────────────────────────────────────────────────────────┐
│  Time ──────────────────────────────────────────────────►  │
│                                                             │
│  V1: ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│  V2: ░░░░░░░░░░░░░░░░████████████████████████████████    │
│       ▲                  ▲                                  │
│       │                  │                                  │
│    V1 stops          V2 starts                              │
│    (downtime)        (new version)                          │
│                                                             │
│  Pros: Simple, clean transition                             │
│  Cons: Downtime, no rollback without redeployment          │
└─────────────────────────────────────────────────────────────┘

Rolling Strategy:
┌─────────────────────────────────────────────────────────────┐
│  Time ──────────────────────────────────────────────────►  │
│                                                             │
│  V1: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│  V2: ░░░░░░░░░░░░████████████░░░░░░░░░░░░░░░░░░░░░░░    │
│  V3: ░░░░░░░░░░░░░░░░░░░░░░░░████████████░░░░░░░░░░░    │
│  V4: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████  │
│       ▲                          ▲                   ▲      │
│       │                          │                   │      │
│    Start rolling            Mid-transition     Complete    │
│                                                             │
│  Pros: No downtime, gradual rollout                         │
│  Cons: Version mismatch possible, complex rollback         │
└─────────────────────────────────────────────────────────────┘

Blue/Green Strategy:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐          ┌──────────────┐               │
│  │  Blue (V1)   │◄─ LB ──►│  Green (V2)  │               │
│  │  Production  │          │  Staging     │               │
│  └──────────────┘          └──────────────┘               │
│         │                          │                        │
│         ▼                          ▼                        │
│  ┌──────────────┐          ┌──────────────┐               │
│  │  100% Traffic│          │  0% Traffic  │               │
│  └──────────────┘          └──────────────┘               │
│                                                             │
│  After validation:                                          │
│  ┌──────────────┐          ┌──────────────┐               │
│  │  Blue (V1)   │◄─ LB ──►│  Green (V2)  │               │
│  │  Staging     │          │  Production  │               │
│  └──────────────┘          └──────────────┘               │
│         │                          │                        │
│         ▼                          ▼                        │
│  ┌──────────────┐          ┌──────────────┐               │
│  │  0% Traffic  │          │  100% Traffic│               │
│  └──────────────┘          └──────────────┘               │
│                                                             │
│  Pros: Instant rollback, zero downtime                     │
│  Cons: Double infrastructure cost                          │
└─────────────────────────────────────────────────────────────┘
```

### 8.1.4 Shadow Deployment

🔴 **Advanced**

```
Shadow Deployment:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    ┌──────────────┐                        │
│  Request ─────────►│   Router     │                        │
│                    └──────┬───────┘                        │
│                           │                                 │
│                    ┌──────┴───────┐                        │
│                    │              │                         │
│                    ▼              ▼                         │
│             ┌──────────┐  ┌──────────┐                    │
│             │  V1      │  │  V2      │                    │
│             │(Primary) │  │ (Shadow) │                    │
│             └────┬─────┘  └────┬─────┘                    │
│                  │              │                           │
│                  ▼              ▼                           │
│             ┌──────────┐  ┌──────────┐                    │
│             │ Response │  │ Response │                    │
│             │ (used)   │  │ (logged) │                    │
│             └──────────┘  └──────────┘                    │
│                                                             │
│  V2 receives production traffic but its response           │
│  is NOT returned to the user — only logged for comparison  │
│                                                             │
│  Pros: Zero risk to users, compare real traffic            │
│  Cons: Double compute cost, complex routing                │
└─────────────────────────────────────────────────────────────┘
```

---

## 8.2 Model Serving Architecture

### 8.2.1 The Model Serving Challenge

🟢 **Beginner**

Model serving is the process of making trained models available for prediction requests. It's more complex than serving traditional web applications because:

1. Models are large (sometimes gigabytes)
2. Inference requires specific hardware (GPUs for deep learning)
3. Latency requirements vary (real-time vs batch)
4. Models need versioning and rollback capabilities

```
Model Serving Requirements:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Latency Requirements:                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Real-time:    < 100ms    (recommendations, chat)   │   │
│  │  Near-real:    < 1s       (search, personalization) │   │
│  │  Batch:        minutes    (analytics, reporting)    │   │
│  │  Offline:      hours      (training, retraining)    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Throughput Requirements:                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Low:          < 100 QPS   (internal tools)         │   │
│  │  Medium:       100-1K QPS  (small product)          │   │
│  │  High:         1K-10K QPS (large product)           │   │
│  │  Very High:    > 10K QPS   (enterprise platform)    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Availability Requirements:                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  99.9%:       ~8.7 hours downtime/year              │   │
│  │  99.99%:      ~52 minutes downtime/year             │   │
│  │  99.999%:     ~5 minutes downtime/year              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2.2 Model Serving Architecture Patterns

🟡 **Intermediate**

```
Pattern 1: Embedded Serving
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Application Server                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │   │
│  │  │ Web API  │  │ Business │  │ ML Model         │ │   │
│  │  │          │  │ Logic    │  │ (Embedded)       │ │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Pros: Simple, low latency                                  │
│  Cons: Tight coupling, scaling issues                       │
└─────────────────────────────────────────────────────────────┘

Pattern 2: Dedicated Model Server
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐        ┌──────────────────────────────┐ │
│  │ Web API      │───────►│ Model Server                 │ │
│  │ (Application)│        │ ┌──────────┐ ┌──────────┐   │ │
│  └──────────────┘        │ │ Model    │ │ Pre/Post │   │ │
│                          │ │ Handler  │ │ Processing│   │ │
│                          │ └──────────┘ └──────────┘   │ │
│                          └──────────────────────────────┘ │
│                                                             │
│  Pros: Separation of concerns, independent scaling         │
│  Cons: Network latency, operational overhead               │
└─────────────────────────────────────────────────────────────┘

Pattern 3: Model Serving Platform
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐        ┌──────────────────────────────┐ │
│  │ Web API      │───────►│ Model Serving Platform       │ │
│  │              │        │ ┌──────────┐ ┌──────────┐   │ │
│  └──────────────┘        │ │ Router   │ │ Model    │   │ │
│                          │ │          │ │ Registry │   │ │
│                          │ └────┬─────┘ └──────────┘   │ │
│                          │      │                        │ │
│                          │      ▼                        │ │
│                          │ ┌──────────┐ ┌──────────┐   │ │
│                          │ │ Model A  │ │ Model B  │   │ │
│                          │ │ v1       │ │ v2       │   │ │
│                          │ └──────────┘ └──────────┘   │ │
│                          └──────────────────────────────┘ │
│                                                             │
│  Pros: Multi-model, versioning, A/B testing, monitoring    │
│  Cons: Complex setup, higher resource usage                │
└─────────────────────────────────────────────────────────────┘
```

### 8.2.3 Seldon Core Architecture

🔴 **Advanced**

```
Seldon Core Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Seldon Core                         │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              API Gateway                      │   │   │
│  │  │  (REST / gRPC / GraphQL)                     │   │   │
│  │  └──────────────────────┬──────────────────────┘   │   │
│  │                         │                           │   │
│  │  ┌──────────────────────▼──────────────────────┐   │   │
│  │  │              Router                          │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │   │
│  │  │  │ Load     │  │ A/B Test │  │ Canary   │  │   │   │
│  │  │  │ Balancer │  │ Router   │  │ Router   │  │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘  │   │   │
│  │  └──────────────────────┬──────────────────────┘   │   │
│  │                         │                           │   │
│  │  ┌──────────────────────▼──────────────────────┐   │   │
│  │  │              Model Runtime                    │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │   │
│  │  │  │ Python   │  │ Java     │  │ TensorFlow│  │   │   │
│  │  │  │ Wrapper  │  │ Wrapper  │  │ Serving   │  │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘  │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Kubernetes Resources                     │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Deployment│  │ Service  │  │ HPA      │         │   │
│  │  │ (Model)  │  │ (gRPC)   │  │ (Auto)   │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2.4 Seldon Core Deployment Example

🟡 **Intermediate**

```yaml
# SeldonDeployment for a Python model
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: text-classifier
  namespace: default
spec:
  predictors:
  - name: default
    replicas: 2
    graph:
      name: classifier
      implementation: UNKNOWN_IMPLEMENTATION
      type: MODEL
      children: []
    componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: registry.example.com/text-classifier:latest
          ports:
          - containerPort: 5000
            protocol: TCP
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
              nvidia.com/gpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
              nvidia.com/gpu: "1"
          env:
          - name: MODEL_PATH
            value: "/models/text-classifier"
          volumeMounts:
          - name: model-volume
            mountPath: /models
        volumes:
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-storage-pvc
    traffic: 100
  annotations:
    seldon.io/engine-image: seldonio/engine:1.17.0
    seldon.io/serving-log-path: /logs
```

```python
# Custom model class for Seldon
class TextClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load model artifacts"""
        import torch
        self.model = torch.load('/models/text-classifier/model.pt')
        self.tokenizer = load_tokenizer('/models/text-classifier/tokenizer')
        
    def predict(self, X, features_names=None):
        """Make predictions"""
        import torch
        
        # Tokenize input
        inputs = self.tokenizer(
            X, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
        
        return predictions.numpy()
    
    def feedback(self, X, Y, features_names=None):
        """Handle feedback for online learning"""
        # Store feedback for retraining
        pass
```

---

## 8.3 A/B Testing & Canary Deployment

### 8.3.1 A/B Testing Fundamentals

🟢 **Beginner**

A/B testing compares two versions of a model by splitting traffic between them to determine which performs better.

```
A/B Testing Setup:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐                                          │
│  │   Incoming   │                                          │
│  │   Traffic    │                                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │   A/B Test   │                                          │
│  │   Router     │                                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│    ┌────┴────┐                                             │
│    │         │                                              │
│    ▼         ▼                                              │
│  ┌──────┐ ┌──────┐                                        │
│  │  A   │ │  B   │                                        │
│  │(50%) │ │(50%) │                                        │
│  │Control│ │Variant│                                       │
│  └──┬───┘ └──┬───┘                                        │
│     │        │                                             │
│     ▼        ▼                                             │
│  ┌──────┐ ┌──────┐                                        │
│  │Track │ │Track │                                        │
│  │Metrics│ │Metrics│                                       │
│  └──────┘ └──────┘                                        │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │  Statistical │                                          │
│  │  Analysis    │                                          │
│  └──────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.3.2 Statistical Significance in A/B Testing

🔴 **Advanced**

```python
import numpy as np
from scipy import stats

class ABTestAnalyzer:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
    
    def analyze_conversion_rate(
        self, 
        control_conversions, 
        control_total,
        variant_conversions, 
        variant_total
    ):
        """Analyze A/B test for conversion rate"""
        
        # Calculate conversion rates
        control_rate = control_conversions / control_total
        variant_rate = variant_conversions / variant_total
        
        # Perform chi-squared test
        contingency_table = np.array([
            [control_conversions, control_total - control_conversions],
            [variant_conversions, variant_total - variant_conversions]
        ])
        
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        
        # Calculate relative improvement
        relative_improvement = (variant_rate - control_rate) / control_rate
        
        # Determine winner
        is_significant = p_value < (1 - self.confidence_level)
        winner = "variant" if variant_rate > control_rate else "control"
        
        return {
            "control_rate": control_rate,
            "variant_rate": variant_rate,
            "relative_improvement": relative_improvement,
            "p_value": p_value,
            "is_significant": is_significant,
            "winner": winner if is_significant else "inconclusive"
        }
    
    def calculate_sample_size(
        self, 
        baseline_rate, 
        minimum_detectable_effect,
        power=0.8
    ):
        """Calculate required sample size"""
        
        alpha = 1 - self.confidence_level
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        sample_size = (
            (z_alpha * np.sqrt(2 * p1 * (1 - p1)) + 
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 /
            (p2 - p1) ** 2
        )
        
        return int(np.ceil(sample_size))

# Usage
analyzer = ABTestAnalyzer(confidence_level=0.95)

# Analyze results after test
results = analyzer.analyze_conversion_rate(
    control_conversions=450,
    control_total=5000,
    variant_conversions=480,
    variant_total=5000
)

print(f"Control rate: {results['control_rate']:.2%}")
print(f"Variant rate: {results['variant_rate']:.2%}")
print(f"Relative improvement: {results['relative_improvement']:.2%}")
print(f"P-value: {results['p_value']:.4f}")
print(f"Significant: {results['is_significant']}")
print(f"Winner: {results['winner']}")

# Calculate required sample size
sample_size = analyzer.calculate_sample_size(
    baseline_rate=0.09,  # 9% baseline conversion
    minimum_detectable_effect=0.10  # 10% relative improvement
)
print(f"Required sample size: {sample_size}")
```

### 8.3.3 Canary Deployment Implementation

🟡 **Intermediate**

```yaml
# Seldon Core canary deployment
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: text-classifier
  namespace: default
spec:
  predictors:
  - name: stable
    replicas: 3
    graph:
      name: classifier
      implementation: UNKNOWN_IMPLEMENTATION
      type: MODEL
      children: []
    componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: registry.example.com/text-classifier:v1.0
          ports:
          - containerPort: 5000
    traffic: 90  # 90% traffic to stable version
  
  - name: canary
    replicas: 1
    graph:
      name: classifier
      implementation: UNKNOWN_IMPLEMENTATION
      type: MODEL
      children: []
    componentSpecs:
    - spec:
        containers:
        - name: classifier
          image: registry.example.com/text-classifier:v2.0
          ports:
          - containerPort: 5000
    traffic: 10  # 10% traffic to canary
```

```bash
# Monitor canary metrics
kubectl logs -f -l seldon-deployment=text-classifier -n default

# Gradually increase canary traffic
kubectl patch seldondeployment text-classifier --type='json' -p='[
  {"op": "replace", "path": "/spec/predictors/1/traffic", "value": 30},
  {"op": "replace", "path": "/spec/predictors/0/traffic", "value": 70}
]'
```

---

## 8.4 Model Version Management

### 8.4.1 Version Management Strategies

🟡 **Intermediate**

```
Model Versioning Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Model Registry                          │   │
│  │                                                     │   │
│  │  text-classifier                                    │   │
│  │  ├── v1.0.0 (2024-01-15)                          │   │
│  │  │   ├── Status: Production                        │   │
│  │  │   ├── Accuracy: 0.945                           │   │
│  │  │   ├── Artifact: s3://models/v1.0.0/model.pt     │   │
│  │  │   └── Metadata: {...}                           │   │
│  │  │                                                  │   │
│  │  ├── v1.1.0 (2024-02-01)                          │   │
│  │  │   ├── Status: Staging                           │   │
│  │  │   ├── Accuracy: 0.952                           │   │
│  │  │   ├── Artifact: s3://models/v1.1.0/model.pt     │   │
│  │  │   └── Metadata: {...}                           │   │
│  │  │                                                  │   │
│  │  └── v1.2.0 (2024-02-15)                          │   │
│  │      ├── Status: Development                       │   │
│  │      ├── Accuracy: 0.948                           │   │
│  │      ├── Artifact: s3://models/v1.2.0/model.pt     │   │
│  │      └── Metadata: {...}                           │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.4.2 Semantic Versioning for ML Models

🔴 **Advanced**

```
ML Model Versioning Scheme:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Format: MAJOR.MINOR.PATCH                                  │
│                                                             │
│  MAJOR: Breaking changes                                    │
│  ├── New architecture                                       │
│  ├── Different input format                                 │
│  └── Incompatible API changes                               │
│                                                             │
│  MINOR: New features (backward compatible)                  │
│  ├── Retrained with more data                               │
│  ├── New preprocessing steps                                │
│  └── Performance improvements                               │
│                                                             │
│  PATCH: Bug fixes                                           │
│  ├── Fix preprocessing bug                                  │
│  ├── Update dependencies                                    │
│  └── Documentation updates                                  │
│                                                             │
│  Examples:                                                  │
│  1.0.0 → 1.0.1: Fix data loading bug                       │
│  1.0.0 → 1.1.0: Retrain with additional data               │
│  1.0.0 → 2.0.0: Switch from BERT to RoBERTa                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.4.3 Model Lineage Tracking

🟡 **Intermediate**

```python
import mlflow
from mlflow.tracking import MlflowClient

class ModelLineageTracker:
    def __init__(self):
        self.client = MlflowClient()
    
    def log_model_lineage(
        self,
        model_name: str,
        model_version: str,
        dataset_version: str,
        training_code_commit: str,
        hyperparameters: dict,
        metrics: dict
    ):
        """Log complete model lineage"""
        
        # Create run
        with mlflow.start_run(run_name=f"{model_name}-v{model_version}"):
            # Log model info
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("dataset_version", dataset_version)
            mlflow.set_tag("training_commit", training_code_commit)
            
            # Log hyperparameters
            mlflow.log_params(hyperparameters)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.pytorch.log_model(model, "model")
            
            # Log dataset info
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_param("dataset_size", len(dataset))
            mlflow.log_param("train_size", len(train_dataset))
            mlflow.log_param("val_size", len(val_dataset))
    
    def get_model_lineage(self, model_name: str, version: str):
        """Retrieve complete model lineage"""
        
        model_versions = self.client.get_latest_versions(
            model_name, 
            stages=["Production"]
        )
        
        for mv in model_versions:
            run = self.client.get_run(mv.run_id)
            
            return {
                "model_name": model_name,
                "version": version,
                "run_id": mv.run_id,
                "dataset_version": run.data.tags.get("dataset_version"),
                "training_commit": run.data.tags.get("training_commit"),
                "hyperparameters": run.data.params,
                "metrics": run.data.metrics,
                "created_at": mv.creation_timestamp,
            }
```

---

## 8.5 Inference Optimization

### 8.5.1 Optimization Techniques Overview

🟡 **Intermediate**

```
Inference Optimization Techniques:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Model-Level Optimizations:                                 │
│  ├── Quantization (INT8/INT4)                              │
│  ├── Pruning (Remove redundant weights)                    │
│  ├── Knowledge Distillation (Large → Small)                │
│  └── Model Architecture Optimization                       │
│                                                             │
│  Runtime Optimizations:                                     │
│  ├── TensorRT (NVIDIA GPU optimization)                    │
│  ├── ONNX Runtime (Cross-platform)                         │
│  ├── OpenVINO (Intel optimization)                         │
│  └── Core ML (Apple devices)                               │
│                                                             │
│  System-Level Optimizations:                                │
│  ├── Batching (Dynamic/Static)                             │
│  ├── Caching (Model/Feature)                               │
│  ├── Load Balancing                                        │
│  └── Auto-scaling                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.5.2 Model Quantization

🔴 **Advanced**

```python
import torch
from torch.quantization import quantize_dynamic

class ModelQuantizer:
    @staticmethod
    def quantize_dynamic(model, dtype=torch.qint8):
        """Dynamic quantization - quantize weights at load time"""
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=dtype
        )
        return quantized_model
    
    @staticmethod
    def quantize_static(model, calibration_data, dtype=torch.qint8):
        """Static quantization - quantize weights and activations"""
        model.eval()
        
        # Prepare model for static quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate with sample data
        with torch.no_grad():
            for batch in calibration_data:
                model_prepared(batch)
        
        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared)
        
        return model_quantized
    
    @staticmethod
    def measure_performance(original_model, quantized_model, test_data):
        """Compare performance metrics"""
        import time
        
        # Measure original model
        start = time.time()
        with torch.no_grad():
            for batch in test_data:
                original_model(batch)
        original_time = time.time() - start
        
        # Measure quantized model
        start = time.time()
        with torch.no_grad():
            for batch in test_data:
                quantized_model(batch)
        quantized_time = time.time() - start
        
        # Measure model size
        import os
        original_size = os.path.getsize('original_model.pt')
        quantized_size = os.path.getsize('quantized_model.pt')
        
        return {
            "original_time": original_time,
            "quantized_time": quantized_time,
            "speedup": original_time / quantized_time,
            "original_size": original_size,
            "quantized_size": quantized_size,
            "compression_ratio": original_size / quantized_size,
        }

# Usage
quantizer = ModelQuantizer()
quantized_model = quantizer.quantize_dynamic(model)
performance = quantizer.measure_performance(model, quantized_model, test_data)

print(f"Speedup: {performance['speedup']:.2f}x")
print(f"Compression: {performance['compression_ratio']:.2f}x")
```

### 8.5.3 Dynamic Batching

🟡 **Intermediate**

```python
import asyncio
from typing import List
import numpy as np

class DynamicBatcher:
    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.batch_queue = asyncio.Queue()
    
    async def predict(self, input_data):
        """Single prediction request"""
        future = asyncio.Future()
        await self.batch_queue.put((input_data, future))
        return await future
    
    async def batch_processor(self):
        """Process batches from queue"""
        while True:
            batch = []
            futures = []
            
            # Collect items for batch
            try:
                # Wait for first item
                item = await asyncio.wait_for(
                    self.batch_queue.get(), 
                    timeout=self.max_wait_ms / 1000
                )
                batch.append(item[0])
                futures.append(item[1])
                
                # Collect remaining items (up to max)
                while len(batch) < self.max_batch_size:
                    try:
                        item = await asyncio.wait_for(
                            self.batch_queue.get(),
                            timeout=self.max_wait_ms / 1000
                        )
                        batch.append(item[0])
                        futures.append(item[1])
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                batch_tensor = np.array(batch)
                predictions = self.model.predict(batch_tensor)
                
                # Return results to individual futures
                for i, future in enumerate(futures):
                    future.set_result(predictions[i])
                    
            except asyncio.TimeoutError:
                continue

# Usage
batcher = DynamicBatcher(model, max_batch_size=32, max_wait_ms=10)

async def main():
    # Start batch processor
    processor_task = asyncio.create_task(batcher.batch_processor())
    
    # Make concurrent predictions
    results = await asyncio.gather(
        batcher.predict(input1),
        batcher.predict(input2),
        batcher.predict(input3),
    )
    
    return results
```

---

## 8.6 Edge Deployment Strategy

### 8.6.1 Edge Deployment Challenges

🟢 **Beginner**

```
Edge Deployment Challenges:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Resource Constraints:                                      │
│  ├── Limited compute power                                  │
│  ├── Limited memory                                         │
│  ├── Limited storage                                        │
│  └── Limited power (battery devices)                        │
│                                                             │
│  Network Constraints:                                       │
│  ├── Intermittent connectivity                              │
│  ├── High latency                                           │
│  └── Limited bandwidth                                      │
│                                                             │
│  Operational Constraints:                                   │
│  ├── Remote updates                                         │
│  ├── Monitoring and debugging                               │
│  └── Security requirements                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.6.2 Edge Deployment Architecture

🟡 **Intermediate**

```
Edge Deployment Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Cloud                                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │   │
│  │  │ Training │  │ Model    │  │ Edge Management  │ │   │
│  │  │ Platform │  │ Registry │  │ Platform         │ │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         │ (Model sync)                      │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Edge Network                            │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Edge     │  │ Edge     │  │ Edge     │         │   │
│  │  │ Device 1 │  │ Device 2 │  │ Device 3 │         │   │
│  │  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │         │   │
│  │  │ │Model │ │  │ │Model │ │  │ │Model │ │         │   │
│  │  │ │(Lite)│ │  │ │(Lite)│ │  │ │(Full)│ │         │   │
│  │  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.6.3 Model Optimization for Edge

🔴 **Advanced**

```python
import tensorflow as tf

class EdgeModelOptimizer:
    def __init__(self):
        self.converter = None
    
    def convert_to_tflite(self, model_path, quantization='dynamic'):
        """Convert model to TensorFlow Lite"""
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to TFLite
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantization == 'dynamic':
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == 'float16':
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
            self.converter.target_spec.supported_types = [tf.float16]
        elif quantization == 'int8':
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
            self.converter.representative_dataset = self._representative_dataset
            self.converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            self.converter.inference_input_type = tf.int8
            self.converter.inference_output_type = tf.int8
        
        tflite_model = self.converter.convert()
        
        # Save model
        output_path = model_path.replace('.h5', '.tflite')
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        return output_path
    
    def optimize_for_mobile(self, model_path):
        """Optimize model for mobile deployment"""
        
        # Convert to TFLite
        tflite_path = self.convert_to_tflite(model_path, quantization='int8')
        
        # Get model size
        import os
        original_size = os.path.getsize(model_path)
        optimized_size = os.path.getsize(tflite_path)
        
        return {
            "tflite_path": tflite_path,
            "original_size_mb": original_size / (1024 * 1024),
            "optimized_size_mb": optimized_size / (1024 * 1024),
            "compression_ratio": original_size / optimized_size,
        }

# Usage
optimizer = EdgeModelOptimizer()
result = optimizer.optimize_for_mobile('model.h5')

print(f"TFLite model saved to: {result['tflite_path']}")
print(f"Original size: {result['original_size_mb']:.2f} MB")
print(f"Optimized size: {result['optimized_size_mb']:.2f} MB")
print(f"Compression ratio: {result['compression_ratio']:.2f}x")
```

---

## 💡 Case Study: Seldon Core Model Serving

### Architecture Overview

```
Seldon Core Production Deployment:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Load Balancer (nginx)                   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │              Seldon Core API Gateway                  │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Request Router                              │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │   │
│  │  │  │ Rate     │  │ Auth     │  │ Routing  │  │   │   │
│  │  │  │ Limiter  │  │ Checker  │  │ Logic    │  │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘  │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │              Model Deployments                        │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Production Model (v1.0)                     │   │   │
│  │  │  Replicas: 3                                 │   │   │
│  │  │  Traffic: 100%                               │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │   │
│  │  │  │ Pod 1    │  │ Pod 2    │  │ Pod 3    │  │   │   │
│  │  │  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │  │   │   │
│  │  │  │ │Model │ │  │ │Model │ │  │ │Model │ │  │   │   │
│  │  │  │ │v1.0  │ │  │ │v1.0  │ │  │ │v1.0  │ │  │   │   │
│  │  │  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │  │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘  │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Canary Model (v2.0)                        │   │   │
│  │  │  Replicas: 1                                │   │   │
│  │  │  Traffic: 0% (canary testing)               │   │   │
│  │  │  ┌──────────┐                               │   │   │
│  │  │  │ Pod 1    │                               │   │   │
│  │  │  │ ┌──────┐ │                               │   │   │
│  │  │  │ │Model │ │                               │   │   │
│  │  │  │ │v2.0  │ │                               │   │   │
│  │  │  │ └──────┘ │                               │   │   │
│  │  │  └──────────┘                               │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Monitoring Stack                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │   │
│  │  │Prometheus│  │ Grafana  │  │ Seldon Analytics │ │   │
│  │  │          │  │Dashboard │  │                  │ │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Instructions

```bash
# 1. Install Seldon Core
kubectl create namespace seldon-system
helm install seldon-core seldon-core-operator \
  --repo https://storage.googleapis.com/seldon-charts \
  --namespace seldon-system \
  --set usageMetrics.enabled=true \
  --set istio.enabled=true

# 2. Create model deployment
kubectl apply -f seldondeployment.yaml

# 3. Verify deployment
kubectl get seldondeployment text-classifier -o yaml

# 4. Port-forward for testing
kubectl port-forward svc/text-classifier-default 8000:8000

# 5. Test prediction
curl -X POST http://localhost:8000/api/v1.0/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "ndarray": ["This is a great product!"]
    }
  }'

# 6. Check metrics
kubectl port-forward svc/prometheus 9090:9090
# Access Grafana dashboard
kubectl port-forward svc/grafana 3000:3000
```

### Monitoring Dashboard

```
Grafana Dashboard: Seldon Core Model Serving
┌─────────────────────────────────────────────────────────────┐
│  Model: text-classifier                                     │
│  Version: v1.0                                              │
│  Status: Healthy ✓                                          │
│                                                             │
│  Request Metrics:                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Total Requests:    15,234                          │   │
│  │  Requests/sec:      42.3                            │   │
│  │  Success Rate:      99.8%                           │   │
│  │  Error Rate:        0.2%                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Latency Distribution:                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  P50:      23ms                                    │   │
│  │  P90:      45ms                                    │   │
│  │  P95:      67ms                                    │   │
│  │  P99:      123ms                                   │   │
│  │  Max:      234ms                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Resource Usage:                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CPU:       ████████████░░░░░░░░ 60%               │   │
│  │  Memory:    ████████░░░░░░░░░░░░ 40%               │   │
│  │  GPU:       ██████████████░░░░░░ 70%               │   │
│  │  Network:   ██████░░░░░░░░░░░░░░ 30%               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Model Performance:                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Accuracy:      0.945                               │   │
│  │  Latency:       23ms (avg)                          │   │
│  │  Throughput:    42.3 req/s                          │   │
│  │  Error Rate:    0.2%                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

**Key Takeaways:**

1. **Deployment strategy** depends on risk tolerance, downtime requirements, and traffic splitting needs
2. **Model serving architecture** should support versioning, A/B testing, and auto-scaling
3. **A/B testing** requires statistical rigor to make valid conclusions
4. **Model version management** is critical for rollback and reproducibility
5. **Inference optimization** can dramatically reduce latency and cost
6. **Edge deployment** requires careful model optimization and management

**Next Chapter Preview:**
In Chapter 9, we'll explore **Model Monitoring & Observability**, covering drift detection, performance monitoring, and building comprehensive observability platforms.

---

*End of Chapter 8*
