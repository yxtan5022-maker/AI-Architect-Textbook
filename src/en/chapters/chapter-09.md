# Chapter 9: Model Monitoring & Observability

> **Part III: MLOps Architecture**

**Learning Objectives:**
- Implement comprehensive model drift detection
- Design data drift monitoring systems
- Build performance metrics monitoring dashboards
- Configure alerting and automated response systems
- Architect observability platforms for ML systems

---

## 9.1 Model Drift Detection

### 9.1.1 What Is Model Drift?

🟢 **Beginner**

Model drift occurs when a deployed model's performance degrades over time due to changes in the underlying data distribution or relationships between features and targets.

```
Model Drift Illustration:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Accuracy Over Time:                                        │
│                                                             │
│  0.95 ┤ ●──●──●──●──●                                     │
│  0.90 ┤              ●──●──●                               │
│  0.85 ┤                     ●──●                           │
│  0.80 ┤                          ●──●──●                   │
│  0.75 ┤                                 ●──●              │
│  0.70 ┤                                    ●──●           │
│       └─────┬─────┬─────┬─────┬─────┬─────┬─────        │
│            T0    T1    T2    T3    T4    T5    T6         │
│            ▲                                     ▲         │
│            │                                     │         │
│        Model deployed                   Performance        │
│                                          degraded          │
│                                                             │
│  Types of Drift:                                            │
│  ├── Data Drift: Input data distribution changes           │
│  ├── Concept Drift: Relationship between X and Y changes   │
│  ├── Model Drift: Model performance degrades               │
│  └── Upstream Drift: Data pipeline changes affect inputs   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.1.2 Types of Drift

🟡 **Intermediate**

| Drift Type | Description | Detection Method | Example |
|------------|-------------|------------------|---------|
| **Data Drift** | Input distribution changes | Statistical tests (KS, PSI) | Seasonal patterns change |
| **Concept Drift** | P(Y\|X) changes | Performance monitoring | User behavior shifts |
| **Model Drift** | Overall performance degradation | Accuracy/F1 monitoring | Model becomes outdated |
| **Prediction Drift** | Output distribution changes | Distribution comparison | Prediction patterns shift |
| **Covariate Drift** | Feature distribution changes | Feature-level monitoring | Data collection changes |

### 9.1.3 Drift Detection Algorithms

🔴 **Advanced**

```python
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score

class ModelDriftDetector:
    def __init__(self, reference_data, reference_predictions):
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
    
    def detect_data_drift_ks(self, current_data, feature_idx, threshold=0.05):
        """Kolmogorov-Smirnov test for data drift"""
        
        reference_feature = self.reference_data[:, feature_idx]
        current_feature = current_data[:, feature_idx]
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(reference_feature, current_feature)
        
        # Determine if drift occurred
        is_drifted = p_value < threshold
        
        return {
            "test": "Kolmogorov-Smirnov",
            "feature_idx": feature_idx,
            "ks_statistic": ks_statistic,
            "p_value": p_value,
            "is_drifted": is_drifted,
            "threshold": threshold
        }
    
    def detect_data_drift_psi(self, reference_data, current_data, feature_idx, threshold=0.1):
        """Population Stability Index for data drift"""
        
        reference_feature = reference_data[:, feature_idx]
        current_feature = current_data[:, feature_idx]
        
        # Create bins
        n_bins = 10
        combined = np.concatenate([reference_feature, current_feature])
        bins = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
        
        # Calculate proportions
        ref_proportions = np.histogram(reference_feature, bins=bins)[0] / len(reference_feature)
        cur_proportions = np.histogram(current_feature, bins=bins)[0] / len(current_feature)
        
        # Avoid division by zero
        ref_proportions = np.where(ref_proportions == 0, 0.0001, ref_proportions)
        cur_proportions = np.where(cur_proportions == 0, 0.0001, cur_proportions)
        
        # Calculate PSI
        psi = np.sum((cur_proportions - ref_proportions) * np.log(cur_proportions / ref_proportions))
        
        # Determine drift
        is_drifted = psi > threshold
        
        return {
            "test": "Population Stability Index",
            "feature_idx": feature_idx,
            "psi": psi,
            "is_drifted": is_drifted,
            "threshold": threshold
        }
    
    def detect_concept_drift(self, current_data, current_labels, window_size=100):
        """Detect concept drift using performance degradation"""
        
        # Calculate performance on recent data
        recent_predictions = self.model.predict(current_data[-window_size:])
        recent_labels = current_labels[-window_size:]
        recent_accuracy = accuracy_score(recent_labels, recent_predictions)
        
        # Compare with reference performance
        reference_accuracy = self.reference_accuracy
        performance_drop = reference_accuracy - recent_accuracy
        
        # Threshold for significant drop
        threshold = 0.05  # 5% performance drop
        
        is_drifted = performance_drop > threshold
        
        return {
            "test": "Performance Degradation",
            "reference_accuracy": reference_accuracy,
            "recent_accuracy": recent_accuracy,
            "performance_drop": performance_drop,
            "is_drifted": is_drifted,
            "threshold": threshold
        }
    
    def detect_prediction_drift(self, current_predictions, threshold=0.1):
        """Detect drift in prediction distribution"""
        
        # Compare prediction distributions
        reference_mean = np.mean(self.reference_predictions)
        reference_std = np.std(self.reference_predictions)
        
        current_mean = np.mean(current_predictions)
        current_std = np.std(current_predictions)
        
        # Calculate distribution shift
        mean_shift = abs(current_mean - reference_mean) / reference_std
        std_shift = abs(current_std - reference_std) / reference_std
        
        is_drifted = mean_shift > threshold or std_shift > threshold
        
        return {
            "test": "Prediction Distribution",
            "mean_shift": mean_shift,
            "std_shift": std_shift,
            "is_drifted": is_drifted,
            "threshold": threshold
        }

# Usage
detector = ModelDriftDetector(reference_data, reference_predictions)

# Check data drift for each feature
for i in range(n_features):
    result = detector.detect_data_drift_ks(current_data, feature_idx=i)
    if result["is_drifted"]:
        print(f"Feature {i}: Drift detected (p={result['p_value']:.4f})")

# Check concept drift
concept_result = detector.detect_concept_drift(current_data, current_labels)
if concept_result["is_drifted"]:
    print(f"Concept drift: {concept_result['performance_drop']:.2%} drop")
```

---

## 9.2 Data Drift Monitoring

### 9.2.1 Data Drift Monitoring Architecture

🟡 **Intermediate**

```
Data Drift Monitoring Pipeline:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Data Sources                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Training │  │ Production│  │ External │         │   │
│  │  │ Data     │  │ Data     │  │ Data     │         │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │   │
│  │       │              │              │               │   │
│  └───────┼──────────────┼──────────────┼───────────────┘   │
│          │              │              │                    │
│          ▼              ▼              ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Feature Store                           │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Reference Statistics (Training)             │   │   │
│  │  │  ├── Mean, Std, Min, Max per feature        │   │   │
│  │  │  ├── Distribution histograms                 │   │   │
│  │  │  └── Correlation matrices                    │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │              Drift Detection Engine                   │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Statistical│  │ ML-Based │  │ Business │         │   │
│  │  │ Tests     │  │ Detection│  │ Rules    │         │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │   │
│  │       │              │              │               │   │
│  │       └──────────────┼──────────────┘               │   │
│  │                      │                              │   │
│  │                      ▼                              │   │
│  │              ┌──────────────┐                       │   │
│  │              │ Drift Score  │                       │   │
│  │              │ Calculator   │                       │   │
│  │              └──────────────┘                       │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │              Alert & Response                         │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Alert    │  │ Auto-    │  │ Dashboard│         │   │
│  │  │ System   │  │ Retrain  │  │ Display  │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2.2 Implementing Data Drift Detection with Evidently

🔴 **Advanced**

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset, 
    DataQualityPreset,
    TargetDriftPreset
)
import pandas as pd

class DataDriftMonitor:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.column_mapping = None
        
    def set_column_mapping(self, target_column: str = None, 
                          numerical_columns: list = None,
                          categorical_columns: list = None):
        """Set column mapping for Evidently"""
        
        self.column_mapping = ColumnMapping(
            target=target_column,
            numerical_features=numerical_columns,
            categorical_features=categorical_columns
        )
    
    def generate_drift_report(self, current_data: pd.DataFrame, 
                             save_path: str = None) -> dict:
        """Generate comprehensive drift report"""
        
        # Create drift report
        drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset()
        ])
        
        # Run report
        drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Get results
        report_dict = drift_report.as_dict()
        
        # Save report if path provided
        if save_path:
            drift_report.save_html(save_path)
        
        # Extract key metrics
        result = {
            "dataset_drift": report_dict["metrics"][0]["result"]["dataset_drift"],
            "drift_score": report_dict["metrics"][0]["result"]["drift_score"],
            "n_drifted_columns": report_dict["metrics"][0]["result"]["n_drifted_columns"],
            "drifted_columns": report_dict["metrics"][0]["result"]["drifted_columns"]
        }
        
        return result
    
    def monitor_real_time(self, current_batch: pd.DataFrame, 
                         threshold: float = 0.5) -> dict:
        """Real-time drift monitoring for streaming data"""
        
        # Calculate drift score for batch
        drift_result = self.generate_drift_report(current_batch)
        
        # Check threshold
        needs_alert = drift_result["drift_score"] > threshold
        
        # Determine action
        if needs_alert:
            action = {
                "type": "alert",
                "severity": "high" if drift_result["drift_score"] > 0.8 else "medium",
                "message": f"Data drift detected: score={drift_result['drift_score']:.3f}",
                "drifted_features": drift_result["drifted_columns"]
            }
        else:
            action = {
                "type": "continue",
                "message": "No significant drift detected"
            }
        
        return {
            "drift_result": drift_result,
            "action": action,
            "timestamp": pd.Timestamp.now()
        }

# Usage
monitor = DataDriftMonitor(reference_data=train_df)
monitor.set_column_mapping(
    target_column="target",
    numerical_columns=["feature1", "feature2", "feature3"],
    categorical_columns=["category1", "category2"]
)

# Generate report
result = monitor.generate_drift_report(current_data=production_df)
print(f"Dataset drift: {result['dataset_drift']}")
print(f"Drift score: {result['drift_score']:.3f}")
print(f"Drifted columns: {result['drifted_columns']}")
```

---

## 9.3 Performance Metrics Monitoring

### 9.3.1 Performance Metrics Framework

🟢 **Beginner**

```
ML Performance Metrics Framework:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Model Quality Metrics:                                     │
│  ├── Accuracy                                               │
│  ├── Precision / Recall / F1                                │
│  ├── AUC-ROC                                                │
│  ├── Mean Squared Error (MSE)                               │
│  └── Mean Absolute Error (MAE)                              │
│                                                             │
│  Operational Metrics:                                       │
│  ├── Latency (P50, P90, P95, P99)                         │
│  ├── Throughput (QPS)                                       │
│  ├── Error Rate                                             │
│  └── Availability                                           │
│                                                             │
│  Business Metrics:                                          │
│  ├── Conversion Rate                                        │
│  ├── Revenue per Prediction                                 │
│  ├── User Satisfaction Score                                │
│  └── Cost per Prediction                                    │
│                                                             │
│  System Metrics:                                            │
│  ├── CPU / Memory Usage                                     │
│  ├── GPU Utilization                                        │
│  ├── Network I/O                                            │
│  └── Disk I/O                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.3.2 Building Monitoring Dashboards

🟡 **Intermediate**

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import numpy as np

class MLMetricsCollector:
    def __init__(self, port=8000):
        self.port = port
        
        # Define metrics
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total number of predictions',
            ['model_name', 'model_version']
        )
        
        self.prediction_latency = Histogram(
            'ml_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_name', 'model_version'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        )
        
        self.prediction_errors = Counter(
            'ml_prediction_errors_total',
            'Total number of prediction errors',
            ['model_name', 'model_version', 'error_type']
        )
        
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Current model accuracy',
            ['model_name', 'model_version']
        )
        
        self.data_drift_score = Gauge(
            'ml_data_drift_score',
            'Current data drift score',
            ['model_name', 'feature_name']
        )
        
        self.gpu_utilization = Gauge(
            'ml_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id']
        )
        
        self.memory_usage = Gauge(
            'ml_memory_usage_bytes',
            'Memory usage in bytes',
            ['model_name']
        )
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        start_http_server(self.port)
        print(f"Metrics server started on port {self.port}")
    
    def record_prediction(self, model_name: str, model_version: str, 
                         latency: float, success: bool):
        """Record a prediction event"""
        
        # Increment prediction counter
        self.prediction_counter.labels(
            model_name=model_name, 
            model_version=model_version
        ).inc()
        
        # Record latency
        self.prediction_latency.labels(
            model_name=model_name, 
            model_version=model_version
        ).observe(latency)
        
        # Record errors if failed
        if not success:
            self.prediction_errors.labels(
                model_name=model_name,
                model_version=model_version,
                error_type="prediction_failed"
            ).inc()
    
    def update_accuracy(self, model_name: str, model_version: str, accuracy: float):
        """Update model accuracy metric"""
        self.model_accuracy.labels(
            model_name=model_name,
            model_version=model_version
        ).set(accuracy)
    
    def update_drift_score(self, model_name: str, feature_name: str, score: float):
        """Update data drift score"""
        self.data_drift_score.labels(
            model_name=model_name,
            feature_name=feature_name
        ).set(score)
    
    def update_system_metrics(self, gpu_id: int, gpu_util: float, 
                             memory_bytes: int, model_name: str):
        """Update system metrics"""
        self.gpu_utilization.labels(gpu_id=str(gpu_id)).set(gpu_util)
        self.memory_usage.labels(model_name=model_name).set(memory_bytes)

# Usage
collector = MLMetricsCollector(port=8000)
collector.start_metrics_server()

# Record predictions
for prediction in predictions:
    start_time = time.time()
    
    # Make prediction
    try:
        result = model.predict(prediction)
        success = True
    except Exception as e:
        success = False
    
    latency = time.time() - start_time
    
    # Record metrics
    collector.record_prediction(
        model_name="text-classifier",
        model_version="v1.0",
        latency=latency,
        success=success
    )
```

### 9.3.3 Grafana Dashboard Configuration

🟡 **Intermediate**

```json
{
  "dashboard": {
    "title": "ML Model Monitoring",
    "panels": [
      {
        "title": "Prediction Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_predictions_total[5m])",
            "legendFormat": "{{model_name}} - {{model_version}}"
          }
        ]
      },
      {
        "title": "Prediction Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))",
            "legendFormat": "P95 Latency"
          },
          {
            "expr": "histogram_quantile(0.50, rate(ml_prediction_latency_seconds_bucket[5m]))",
            "legendFormat": "P50 Latency"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "ml_model_accuracy",
            "legendFormat": "{{model_name}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_prediction_errors_total[5m])",
            "legendFormat": "{{error_type}}"
          }
        ]
      },
      {
        "title": "Data Drift Score",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_data_drift_score",
            "legendFormat": "{{feature_name}}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "gauge",
        "targets": [
          {
            "expr": "ml_gpu_utilization_percent",
            "legendFormat": "GPU {{gpu_id}}"
          }
        ]
      }
    ],
    "refresh": "30s",
    "time": {
      "from": "now-6h",
      "to": "now"
    }
  }
}
```

---

## 9.4 Alerting & Automated Response

### 9.4.1 Alerting Strategy

🟡 **Intermediate**

```
Alerting Strategy Framework:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Alert Severity Levels:                                     │
│  ├── Critical (P0): Model completely broken                │
│  │   ├── Immediate notification (PagerDuty)                │
│  │   ├── Auto-rollback triggered                           │
│  │   └── Human intervention required                       │
│  │                                                         │
│  ├── Warning (P1): Significant degradation                 │
│  │   ├── Team notification (Slack)                         │
│  │   ├── Increased monitoring frequency                    │
│  │   └── Investigation required                            │
│  │                                                         │
│  ├── Info (P2): Minor issues detected                      │
│  │   ├── Log entry                                         │
│  │   ├── Dashboard update                                  │
│  │   └── Review during business hours                      │
│  │                                                         │
│  └── Low (P3): Potential concerns                          │
│      ├── Metric recording                                  │
│      └── Weekly review                                     │
│                                                             │
│  Alert Channels:                                            │
│  ├── PagerDuty (Critical)                                  │
│  ├── Slack (Warning)                                       │
│  ├── Email (Info)                                          │
│  └── Dashboard (All)                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.4.2 Prometheus Alert Rules

🔴 **Advanced**

```yaml
# prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-model-alerts
  namespace: monitoring
spec:
  groups:
  - name: ml-model-alerts
    rules:
    # Model Performance Alerts
    - alert: ModelAccuracyBelowThreshold
      expr: ml_model_accuracy < 0.85
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Model accuracy below threshold"
        description: "Model {{ $labels.model_name }} accuracy is {{ $value }}"
    
    - alert: ModelAccuracyCritical
      expr: ml_model_accuracy < 0.75
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "Model accuracy critically low"
        description: "Model {{ $labels.model_name }} accuracy is {{ $value }}"
    
    # Latency Alerts
    - alert: HighPredictionLatency
      expr: histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m])) > 0.5
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High prediction latency"
        description: "P95 latency is {{ $value }}s"
    
    - alert: CriticalPredictionLatency
      expr: histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m])) > 1.0
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "Critical prediction latency"
        description: "P95 latency is {{ $value }}s"
    
    # Error Rate Alerts
    - alert: HighErrorRate
      expr: rate(ml_prediction_errors_total[5m]) / rate(ml_predictions_total[5m]) > 0.05
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High error rate"
        description: "Error rate is {{ $value | humanizePercentage }}"
    
    # Data Drift Alerts
    - alert: DataDriftDetected
      expr: ml_data_drift_score > 0.5
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Data drift detected"
        description: "Feature {{ $labels.feature_name }} drift score is {{ $value }}"
    
    - alert: CriticalDataDrift
      expr: ml_data_drift_score > 0.8
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Critical data drift"
        description: "Feature {{ $labels.feature_name }} drift score is {{ $value }}"
    
    # Resource Alerts
    - alert: HighGPUUtilization
      expr: ml_gpu_utilization_percent > 90
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High GPU utilization"
        description: "GPU {{ $labels.gpu_id }} utilization is {{ $value }}%"
    
    # Prediction Volume Alerts
    - alert: LowPredictionVolume
      expr: rate(ml_predictions_total[15m]) < 10
      for: 15m
      labels:
        severity: info
      annotations:
        summary: "Low prediction volume"
        description: "Prediction rate is {{ $value }} req/s"
```

### 9.4.3 Automated Response System

🔴 **Advanced**

```python
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List
import logging

class AutomatedResponseSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.prometheus_url = config.get('prometheus_url', 'http://localhost:9090')
        self.alertmanager_url = config.get('alertmanager_url', 'http://localhost:9093')
        self.kubernetes_api = config.get('kubernetes_api', 'https://kubernetes.default.svc')
        
    def check_model_health(self, model_name: str, model_version: str) -> Dict:
        """Check overall model health"""
        
        # Query Prometheus for metrics
        queries = {
            'accuracy': f'ml_model_accuracy{{model_name="{model_name}",model_version="{model_version}"}}',
            'latency_p95': f'histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket{{model_name="{model_name}"}}[5m]))',
            'error_rate': f'rate(ml_prediction_errors_total{{model_name="{model_name}"}}[5m]) / rate(ml_predictions_total{{model_name="{model_name}"}}[5m])',
            'drift_score': f'ml_data_drift_score{{model_name="{model_name}"}}'
        }
        
        results = {}
        for metric_name, query in queries.items():
            response = requests.get(f'{self.prometheus_url}/api/v1/query', params={'query': query})
            if response.status_code == 200:
                data = response.json()
                if data['data']['result']:
                    results[metric_name] = float(data['data']['result'][0]['value'][1])
        
        # Determine health status
        health_status = {
            'model_name': model_name,
            'model_version': model_version,
            'timestamp': datetime.now().isoformat(),
            'metrics': results,
            'status': 'healthy',
            'issues': []
        }
        
        # Check thresholds
        if results.get('accuracy', 1.0) < 0.85:
            health_status['status'] = 'degraded'
            health_status['issues'].append('Low accuracy')
        
        if results.get('latency_p95', 0) > 0.5:
            health_status['status'] = 'degraded'
            health_status['issues'].append('High latency')
        
        if results.get('error_rate', 0) > 0.05:
            health_status['status'] = 'critical'
            health_status['issues'].append('High error rate')
        
        if results.get('drift_score', 0) > 0.5:
            health_status['status'] = 'warning'
            health_status['issues'].append('Data drift detected')
        
        return health_status
    
    def auto_rollback(self, model_name: str, reason: str) -> bool:
        """Automatically rollback to previous model version"""
        
        self.logger.warning(f"Auto-rollback triggered for {model_name}: {reason}")
        
        # Get previous stable version
        previous_version = self._get_previous_stable_version(model_name)
        
        if previous_version:
            # Update SeldonDeployment
            success = self._update_seldon_deployment(model_name, previous_version)
            
            if success:
                # Send notification
                self._send_notification(
                    severity="critical",
                    title=f"Auto-rollback: {model_name}",
                    message=f"Rolled back to version {previous_version}. Reason: {reason}"
                )
                
                # Log the rollback
                self._log_rollback_event(model_name, previous_version, reason)
                
                return True
        
        return False
    
    def auto_scale(self, model_name: str, metric: str, target_value: float) -> bool:
        """Automatically scale model deployment based on metrics"""
        
        # Get current replicas
        current_replicas = self._get_current_replicas(model_name)
        
        # Calculate desired replicas based on metric
        if metric == 'latency':
            current_latency = self._get_metric(model_name, 'latency_p95')
            if current_latency > target_value * 1.2:
                desired_replicas = min(current_replicas + 2, 10)  # Scale up
            elif current_latency < target_value * 0.8:
                desired_replicas = max(current_replicas - 1, 2)  # Scale down
            else:
                desired_replicas = current_replicas
        elif metric == 'queue_depth':
            current_queue = self._get_metric(model_name, 'queue_depth')
            if current_queue > target_value:
                desired_replicas = min(current_replicas + 1, 10)
            else:
                desired_replicas = current_replicas
        
        # Apply scaling if needed
        if desired_replicas != current_replicas:
            return self._scale_deployment(model_name, desired_replicas)
        
        return True
    
    def trigger_retraining(self, model_name: str, reason: str) -> str:
        """Trigger model retraining pipeline"""
        
        # Create Kubeflow Pipeline run
        pipeline_run = self._create_kubeflow_run(
            pipeline_name="model-retraining",
            params={
                "model_name": model_name,
                "reason": reason,
                "trigger_time": datetime.now().isoformat()
            }
        )
        
        # Send notification
        self._send_notification(
            severity="info",
            title=f"Retraining triggered: {model_name}",
            message=f"Pipeline run started: {pipeline_run['run_id']}. Reason: {reason}"
        )
        
        return pipeline_run['run_id']
    
    def _get_previous_stable_version(self, model_name: str) -> str:
        """Get previous stable model version"""
        # Implementation depends on model registry
        pass
    
    def _update_seldon_deployment(self, model_name: str, version: str) -> bool:
        """Update SeldonDeployment to use specified version"""
        # Implementation depends on Kubernetes API
        pass
    
    def _send_notification(self, severity: str, title: str, message: str):
        """Send notification via configured channels"""
        
        if severity == "critical":
            # PagerDuty
            self._send_pagerduty(title, message)
        
        # Slack
        self._send_slack(severity, title, message)
        
        # Email
        self._send_email(severity, title, message)
    
    def _send_pagerduty(self, title: str, message: str):
        """Send PagerDuty alert"""
        pass
    
    def _send_slack(self, severity: str, title: str, message: str):
        """Send Slack notification"""
        pass
    
    def _send_email(self, severity: str, title: str, message: str):
        """Send email notification"""
        pass
    
    def _log_rollback_event(self, model_name: str, version: str, reason: str):
        """Log rollback event for audit"""
        pass

# Usage
response_system = AutomatedResponseSystem({
    'prometheus_url': 'http://prometheus:9090',
    'alertmanager_url': 'http://alertmanager:9093',
    'kubernetes_api': 'https://kubernetes.default.svc'
})

# Check model health
health = response_system.check_model_health("text-classifier", "v1.0")
print(f"Model status: {health['status']}")
print(f"Issues: {health['issues']}")

# Auto-rollback if critical
if health['status'] == 'critical':
    response_system.auto_rollback("text-classifier", health['issues'][0])
```

---

## 9.5 Observability Platform Architecture

### 9.5.1 The Three Pillars of ML Observability

🟡 **Intermediate**

```
ML Observability Pillars:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Logs                                     │   │
│  │  ├── Prediction logs                                 │   │
│  │  ├── Error logs                                      │   │
│  │  ├── Audit logs                                      │   │
│  │  └── System logs                                     │   │
│  │                                                     │   │
│  │  Tools: ELK Stack, Fluentd, Loki                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Metrics                                  │   │
│  │  ├── Model metrics (accuracy, latency)              │   │
│  │  ├── System metrics (CPU, memory, GPU)              │   │
│  │  ├── Business metrics (conversion, revenue)         │   │
│  │  └── Data metrics (drift, quality)                  │   │
│  │                                                     │   │
│  │  Tools: Prometheus, Grafana, DataDog                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Traces                                   │   │
│  │  ├── Request traces                                  │   │
│  │  ├── Model inference traces                          │   │
│  │  ├── Data pipeline traces                            │   │
│  │  └── Distributed traces                              │   │
│  │                                                     │   │
│  │  Tools: Jaeger, Zipkin, OpenTelemetry                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.5.2 Complete Observability Architecture

🔴 **Advanced**

```
ML Observability Platform Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Data Collection Layer                    │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Model    │  │ System   │  │ Business │         │   │
│  │  │ Metrics  │  │ Metrics  │  │ Metrics  │         │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │   │
│  │       │              │              │               │   │
│  │       ▼              ▼              ▼               │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │           Prometheus                         │   │   │
│  │  │  ├── Time-series storage                    │   │   │
│  │  │  ├── PromQL queries                         │   │   │
│  │  │  └── Alerting rules                         │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Visualization Layer                     │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              Grafana                         │   │   │
│  │  │  ├── Model performance dashboards           │   │   │
│  │  │  ├── System health dashboards               │   │   │
│  │  │  ├── Business metrics dashboards            │   │   │
│  │  │  └── Alert management                       │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Analysis Layer                          │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Drift    │  │ Anomaly  │  │ Root     │         │   │
│  │  │ Detection│  │ Detection│  │ Cause    │         │   │
│  │  │          │  │          │  │ Analysis │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Response Layer                          │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Alerting │  │ Auto-    │  │ Feedback │         │   │
│  │  │ System   │  │ Response │  │ Loop     │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.5.3 OpenTelemetry Integration

🔴 **Advanced**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
import time

class MLTracingSetup:
    def __init__(self, service_name: str, jaeger_endpoint: str):
        # Create resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        # Configure tracer
        provider = TracerProvider(resource=resource)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        # Add processor
        processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(processor)
        
        # Set global tracer
        trace.set_tracer_provider(provider)
        
        self.tracer = trace.get_tracer(__name__)
    
    def trace_prediction(self, model_name: str, input_data: dict):
        """Trace a prediction request"""
        
        with self.tracer.start_as_current_span("prediction") as span:
            # Add attributes
            span.set_attribute("model.name", model_name)
            span.set_attribute("input.size", len(str(input_data)))
            
            # Start preprocessing span
            with self.tracer.start_as_current_span("preprocessing") as preprocess_span:
                start_time = time.time()
                # Preprocessing logic
                processed_data = self._preprocess(input_data)
                preprocess_span.set_attribute("preprocessing.duration", 
                                            time.time() - start_time)
            
            # Start inference span
            with self.tracer.start_as_current_span("inference") as inference_span:
                start_time = time.time()
                # Inference logic
                prediction = self._predict(processed_data)
                inference_span.set_attribute("inference.duration", 
                                           time.time() - start_time)
                inference_span.set_attribute("prediction.confidence", 
                                           prediction.get('confidence', 0))
            
            # Start postprocessing span
            with self.tracer.start_as_current_span("postprocessing") as postprocess_span:
                start_time = time.time()
                # Postprocessing logic
                result = self._postprocess(prediction)
                postprocess_span.set_attribute("postprocessing.duration", 
                                             time.time() - start_time)
            
            return result
    
    def _preprocess(self, data):
        """Preprocessing logic"""
        pass
    
    def _predict(self, data):
        """Prediction logic"""
        pass
    
    def _postprocess(self, prediction):
        """Postprocessing logic"""
        pass

# Usage
tracing = MLTracingSetup(
    service_name="text-classifier",
    jaeger_endpoint="http://jaeger:14268/api/traces"
)

# Trace prediction
result = tracing.trace_prediction(
    model_name="text-classifier",
    input_data={"text": "This is a great product!"}
)
```

---

## 💡 Case Study: Prometheus + Grafana AI Monitoring System

### System Architecture

```
Prometheus + Grafana AI Monitoring System:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ML Services                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Model    │  │ Training │  │ Data     │         │   │
│  │  │ Serving  │  │ Pipeline │  │ Pipeline │         │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │   │
│  │       │              │              │               │   │
│  └───────┼──────────────┼──────────────┼───────────────┘   │
│          │              │              │                    │
│          ▼              ▼              ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Metrics Exporters                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Custom   │  │ Node     │  │ cAdvisor │         │   │
│  │  │ Exporter │  │ Exporter │  │          │         │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │   │
│  │       │              │              │               │   │
│  └───────┼──────────────┼──────────────┼───────────────┘   │
│          │              │              │                    │
│          ▼              ▼              ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Prometheus Server                        │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  ├── Scrape intervals (15s-60s)             │   │   │
│  │  │  ├── Retention (30 days)                    │   │   │
│  │  │  ├── Alert rules                            │   │   │
│  │  │  └── Recording rules                       │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Grafana Dashboard                        │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Dashboard 1: Model Performance              │   │   │
│  │  │  ├── Accuracy over time                     │   │   │
│  │  │  ├── Latency distribution                   │   │   │
│  │  │  ├── Error rate                             │   │   │
│  │  │  └── Prediction volume                      │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Dashboard 2: Data Quality                   │   │   │
│  │  │  ├── Data drift scores                      │   │   │
│  │  │  ├── Feature distributions                  │   │   │
│  │  │  ├── Missing values                         │   │   │
│  │  │  └── Data freshness                         │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  Dashboard 3: System Health                  │   │   │
│  │  │  ├── CPU/Memory usage                       │   │   │
│  │  │  ├── GPU utilization                        │   │   │
│  │  │  ├── Network I/O                            │   │   │
│  │  │  └── Disk usage                             │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Alerting Pipeline                       │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │Prometheus│  │ Alert    │  │ Notifi-  │         │   │
│  │  │ Alerting │──│ Manager  │──│ cations  │         │   │
│  │  │ Rules    │  │          │  │          │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                     │              │               │   │
│  │                     ▼              ▼               │   │
│  │              ┌──────────┐  ┌──────────┐           │   │
│  │              │ Slack    │  │ PagerDuty│           │   │
│  │              │          │  │          │           │   │
│  │              └──────────┘  └──────────┘           │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Instructions

```bash
# 1. Create monitoring namespace
kubectl create namespace monitoring

# 2. Deploy Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --set alertmanager.enabled=true

# 3. Deploy Grafana
helm repo add grafana https://grafana.github.io/helm-charts
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword=admin123

# 4. Deploy custom ML exporter
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-metrics-exporter
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-metrics-exporter
  template:
    metadata:
      labels:
        app: ml-metrics-exporter
    spec:
      containers:
      - name: exporter
        image: registry.example.com/ml-metrics-exporter:latest
        ports:
        - containerPort: 8000
        env:
        - name: PROMETHEUS_URL
          value: "http://prometheus-server:9090"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-metrics-exporter
  namespace: monitoring
spec:
  selector:
    app: ml-metrics-exporter
  ports:
  - port: 8000
    targetPort: 8000
EOF

# 5. Access dashboards
kubectl port-forward svc/grafana 3000:80 -n monitoring
# Grafana: http://localhost:3000 (admin/admin123)

kubectl port-forward svc/prometheus-server 9090:80 -n monitoring
# Prometheus: http://localhost:9090
```

### Dashboard Configuration

```json
{
  "dashboard": {
    "title": "ML Model Monitoring Dashboard",
    "uid": "ml-model-monitoring",
    "panels": [
      {
        "title": "Model Accuracy Trend",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
        "targets": [
          {
            "expr": "ml_model_accuracy{model_name=\"text-classifier\"}",
            "legendFormat": "{{model_version}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                { "color": "red", "value": null },
                { "color": "green", "value": 0.85 }
              ]
            }
          }
        }
      },
      {
        "title": "Prediction Latency (P95)",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket{model_name=\"text-classifier\"}[5m]))",
            "legendFormat": "P95 Latency"
          }
        ]
      },
      {
        "title": "Data Drift Scores",
        "type": "bargauge",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
        "targets": [
          {
            "expr": "ml_data_drift_score{model_name=\"text-classifier\"}",
            "legendFormat": "{{feature_name}}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "gauge",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 8 },
        "targets": [
          {
            "expr": "ml_gpu_utilization_percent",
            "legendFormat": "GPU {{gpu_id}}"
          }
        ]
      }
    ],
    "refresh": "30s",
    "time": {
      "from": "now-24h",
      "to": "now"
    }
  }
}
```

---

## Summary

**Key Takeaways:**

1. **Model drift** is inevitable — continuous monitoring is essential
2. **Data drift detection** requires statistical rigor and proper baselines
3. **Performance monitoring** must cover model quality, operational, and business metrics
4. **Alerting** should be actionable and properly tiered
5. **Observability** requires logs, metrics, and traces working together

**Best Practices:**

- Establish clear baselines during model development
- Implement automated drift detection with configurable thresholds
- Build comprehensive dashboards that tell a story
- Create runbooks for common issues
- Regular review and update of monitoring rules

**Complete MLOps Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Complete MLOps Architecture                    │
│                                                                 │
│  Chapter 6: MLOps Fundamentals                                  │
│  ├── Maturity Model (Level 0-3)                                │
│  ├── Toolchain Selection                                        │
│  └── DevOps vs MLOps                                           │
│                                                                 │
│  Chapter 7: Model Training                                      │
│  ├── Training Environment Design                               │
│  ├── Distributed Training                                      │
│  ├── HPO & Experiment Tracking                                 │
│  └── Resource Management                                       │
│                                                                 │
│  Chapter 8: Model Deployment                                    │
│  ├── Deployment Strategies                                     │
│  ├── Model Serving (Seldon Core)                               │
│  ├── A/B Testing & Canary                                      │
│  └── Inference Optimization                                    │
│                                                                 │
│  Chapter 9: Model Monitoring                                    │
│  ├── Drift Detection                                           │
│  ├── Performance Monitoring                                    │
│  ├── Alerting & Response                                       │
│  └── Observability Platform                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*End of Chapter 9*
*End of Part III: MLOps Architecture*
