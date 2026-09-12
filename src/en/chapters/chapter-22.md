# Chapter 22: AI Observability and Monitoring

**Reader Level:** 🔴 Advanced | **Pages:** 25 | **Code:** Python, PromQL, Grafana

---

## 22.1 The Three Pillars of AI Observability

### 22.1.1 Logs, Metrics, and Traces

```
┌────────────────────────────────────────────────────────────┐
│              AI OBSERVABILITY STACK                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    LOGS                               │   │
│  │  • Request/response pairs                           │   │
│  │  • Error messages and stack traces                  │   │
│  │  • Audit trails for compliance                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   METRICS                            │   │
│  │  • Latency (TTFT, inter-token, p50/p99)            │   │
│  │  • Throughput (tokens/sec, RPS)                     │   │
│  │  • Resource utilization (GPU, memory, CPU)          │   │
│  │  • Quality metrics (hallucination rate, toxicity)   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   TRACES                             │   │
│  │  • Request lifecycle across services                │   │
│  │  • Model inference pipeline stages                  │   │
│  │  • RAG retrieval and generation correlation         │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### 22.1.2 AI-Specific Metrics

```python
# metrics_collector.py
import time
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class AIMetrics:
    request_count: int = 0
    token_count: int = 0
    latencies: List[float] = field(default_factory=list)
    ttft_values: List[float] = field(default_factory=list)
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

class MetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, AIMetrics] = defaultdict(AIMetrics)
    
    def record_request(
        self, model_id: str, latency: float, ttft: float,
        tokens: int, error: bool = False
    ):
        m = self.metrics[model_id]
        m.request_count += 1
        m.token_count += tokens
        m.latencies.append(latency)
        m.ttft_values.append(ttft)
        if error:
            m.error_count += 1
    
    def get_summary(self, model_id: str) -> Dict:
        m = self.metrics[model_id]
        if not m.latencies:
            return {"error": "no data"}
        
        sorted_lat = sorted(m.latencies)
        sorted_ttft = sorted(m.ttft_values)
        p50_idx = len(sorted_lat) // 2
        p99_idx = int(len(sorted_lat) * 0.99)
        
        return {
            "model": model_id,
            "total_requests": m.request_count,
            "total_tokens": m.token_count,
            "error_rate": m.error_count / max(m.request_count, 1),
            "latency_p50_ms": sorted_lat[p50_idx] * 1000,
            "latency_p99_ms": sorted_lat[min(p99_idx, len(sorted_lat)-1)] * 1000,
            "ttft_p50_ms": sorted_ttft[p50_idx] * 1000,
            "tokens_per_second": m.token_count / max(sum(m.latencies), 0.001),
            "cache_hit_rate": m.cache_hits / max(m.cache_hits + m.cache_misses, 1),
        }
```


## 22.2 Production Monitoring Stack

### 22.2.1 Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                 PRODUCTION MONITORING STACK                     │
│                                                                 │
│  Data Collection:                                               │
│  ├── OpenTelemetry SDK (traces, metrics, logs)                 │
│  ├── Prometheus Client (system metrics)                        │
│  └── Custom AI Metrics Exporters                               │
│                                                                 │
│  Storage:                                                       │
│  ├── Prometheus (time-series metrics)                          │
│  ├── Loki (log aggregation)                                    │
│  ├── Tempo/Jaeger (distributed traces)                        │
│  └── ClickHouse (query analytics)                              │
│                                                                 │
│  Visualization:                                                 │
│  ├── Grafana Dashboards                                        │
│  ├── Custom AI Quality Dashboards                              │
│  └── Alert Manager (PagerDuty, Slack)                          │
│                                                                 │
│  Analysis:                                                      │
│  ├── PromQL for threshold alerts                               │
│  ├── ML-based anomaly detection                                │
│  └── Cost attribution and optimization                         │
└────────────────────────────────────────────────────────────────┘
```

### 22.2.2 Prometheus Metrics Configuration

```yaml
# prometheus_ai_metrics.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "ai_alerts.yml"

scrape_configs:
  - job_name: 'ai-inference-service'
    static_configs:
      - targets: ['ai-service:8080']
    metrics_path: /metrics
    
  - job_name: 'model-servers'
    static_configs:
      - targets: ['gpu-worker-1:9090', 'gpu-worker-2:9090']

# ai_alerts.yml
groups:
  - name: ai_model_alerts
    rules:
      - alert: HighLatencyP99
        expr: histogram_quantile(0.99, rate(ai_request_duration_seconds_bucket[5m])) > 2.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency exceeds 2s"
          
      - alert: HighErrorRate
        expr: rate(ai_requests_errors_total[5m]) / rate(ai_requests_total[5m]) > 0.05
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Error rate exceeds 5%"
          
      - alert: GPUUtilizationLow
        expr: avg(gpu_utilization) < 30
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "GPU utilization below 30% - consider scaling down"
          
      - alert: HallucinationRateHigh
        expr: ai_hallucination_rate > 0.1
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "Hallucination rate exceeds 10%"
```

### 22.2.3 Grafana Dashboard Panels

```json
{
  "panels": [
    {
      "title": "Request Latency Distribution",
      "type": "heatmap",
      "targets": [
        {
          "expr": "histogram_quantile(0.5, rate(ai_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p50"
        },
        {
          "expr": "histogram_quantile(0.99, rate(ai_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p99"
        }
      ]
    },
    {
      "title": "Token Throughput",
      "type": "timeseries",
      "targets": [
        {
          "expr": "rate(ai_tokens_generated_total[5m])",
          "legendFormat": "tokens/sec"
        }
      ]
    },
    {
      "title": "GPU Memory Usage",
      "type": "gauge",
      "targets": [
        {
          "expr": "nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes * 100",
          "legendFormat": "{{gpu}} utilization %"
        }
      ]
    }
  ]
}
```


## 22.3 Quality Monitoring

### 22.3.1 AI Quality Metrics Framework

```python
# quality_monitor.py
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class QualityScore:
    relevance: float      # 0-1: How relevant is the response
    accuracy: float       # 0-1: Factual accuracy
    coherence: float      # 0-1: Logical coherence
    safety: float         # 0-1: Safety compliance
    latency_score: float  # 0-1: Within latency budget

class QualityMonitor:
    def __init__(self):
        self.scores: List[QualityScore] = []
    
    def evaluate_response(self, response: str, context: str) -> QualityScore:
        # In production, use specialized evaluation models
        score = QualityScore(
            relevance=self._check_relevance(response, context),
            accuracy=self._check_accuracy(response),
            coherence=self._check_coherence(response),
            safety=self._check_safety(response),
            latency_score=1.0,  # Calculated from timing
        )
        self.scores.append(score)
        return score
    
    def _check_relevance(self, response: str, context: str) -> float:
        # Semantic similarity or LLM-as-judge
        return 0.85  # Placeholder
    
    def _check_accuracy(self, response: str) -> float:
        # Fact-checking against knowledge base
        return 0.90  # Placeholder
    
    def _check_coherence(self, response: str) -> float:
        # Perplexity or coherence model
        return 0.88  # Placeholder
    
    def _check_safety(self, response: str) -> float:
        # Toxicity and safety classifiers
        return 0.95  # Placeholder
    
    def get_quality_report(self) -> Dict:
        if not self.scores:
            return {"error": "no data"}
        
        avg = lambda s: sum(getattr(x, s) for x in self.scores) / len(self.scores)
        return {
            "avg_relevance": avg("relevance"),
            "avg_accuracy": avg("accuracy"),
            "avg_coherence": avg("coherence"),
            "avg_safety": avg("safety"),
            "total_evaluations": len(self.scores),
        }
```

### 22.3.2 Hallucination Detection

```python
# hallucination_detector.py
from typing import List, Dict

class HallucinationDetector:
    def __init__(self):
        self.detection_methods = [
            "self_consistency",
            "knowledge_grounding",
            "citation_check",
        ]
    
    def detect(
        self, response: str, source_documents: List[str]
    ) -> Dict[str, float]:
        scores = {}
        
        # Method 1: Self-consistency check
        # Generate multiple responses, check agreement
        scores["self_consistency"] = self._self_consistency(response)
        
        # Method 2: Knowledge grounding
        # Verify claims against source documents
        scores["knowledge_grounding"] = self._knowledge_grounding(
            response, source_documents
        )
        
        # Method 3: Citation verification
        # Check if citations actually support claims
        scores["citation_accuracy"] = self._citation_check(response)
        
        # Overall hallucination probability
        scores["hallucination_probability"] = 1 - (
            scores["self_consistency"] * 0.3 +
            scores["knowledge_grounding"] * 0.5 +
            scores["citation_accuracy"] * 0.2
        )
        
        return scores
    
    def _self_consistency(self, response: str) -> float:
        # Placeholder for consistency checking
        return 0.85
    
    def _knowledge_grounding(
        self, response: str, sources: List[str]
    ) -> float:
        # Placeholder for grounding verification
        return 0.80
    
    def _citation_check(self, response: str) -> float:
        # Placeholder for citation verification
        return 0.90
```

### 22.3.3 Drift Detection

```python
# drift_detector.py
import numpy as np
from typing import List
from scipy import stats

class DriftDetector:
    def __init__(self, baseline_window: int = 1000):
        self.baseline_window = baseline_window
        self.baseline_data: List[float] = []
        self.current_data: List[float] = []
    
    def add_baseline(self, value: float):
        self.baseline_data.append(value)
        if len(self.baseline_data) > self.baseline_window:
            self.baseline_data = self.baseline_data[-self.baseline_window:]
    
    def add_current(self, value: float):
        self.current_data.append(value)
    
    def detect_drift(self, threshold: float = 0.05) -> Dict:
        if len(self.baseline_data) < 100 or len(self.current_data) < 100:
            return {"drift_detected": False, "reason": "insufficient_data"}
        
        stat, p_value = stats.ks_2samp(self.baseline_data, self.current_data)
        
        return {
            "drift_detected": p_value < threshold,
            "ks_statistic": stat,
            "p_value": p_value,
            "baseline_mean": np.mean(self.baseline_data),
            "current_mean": np.mean(self.current_data),
            "mean_shift": np.mean(self.current_data) - np.mean(self.baseline_data),
        }
```


## 22.4 Distributed Tracing

### 22.4.1 AI Pipeline Trace Model

```python
# ai_tracing.py
import time
import uuid
from typing import Dict, Optional
from dataclasses import dataclass, field

@dataclass
class AISpan:
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    parent_id: Optional[str] = None
    name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: Dict = field(default_factory=dict)
    events: list = field(default_factory=list)

class AITracer:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.spans: list = []
        self.current_trace_id: str = ""
    
    def start_trace(self) -> str:
        self.current_trace_id = uuid.uuid4().hex
        return self.current_trace_id
    
    def start_span(self, name: str, parent_id: Optional[str] = None) -> AISpan:
        span = AISpan(
            trace_id=self.current_trace_id,
            parent_id=parent_id,
            name=name,
            start_time=time.perf_counter(),
        )
        self.spans.append(span)
        return span
    
    def end_span(self, span: AISpan):
        span.end_time = time.perf_counter()
        span.attributes["duration_ms"] = (
            (span.end_time - span.start_time) * 1000
        )
    
    def add_event(self, span: AISpan, name: str, attributes: Dict = None):
        span.events.append({
            "name": name,
            "timestamp": time.perf_counter(),
            "attributes": attributes or {},
        })
    
    def get_trace_summary(self) -> Dict:
        root_spans = [s for s in self.spans if s.parent_id is None]
        total_duration = max(
            (s.end_time - s.start_time) for s in self.spans if s.end_time > 0
        ) if self.spans else 0
        
        return {
            "trace_id": self.current_trace_id,
            "total_spans": len(self.spans),
            "total_duration_ms": total_duration * 1000,
            "service": self.service_name,
            "span_breakdown": [
                {
                    "name": s.name,
                    "duration_ms": s.attributes.get("duration_ms", 0),
                    "percentage": (
                        s.attributes.get("duration_ms", 0) /
                        (total_duration * 1000) * 100
                    ),
                }
                for s in self.spans if s.end_time > 0
            ],
        }
```

### 22.4.2 Trace Propagation in RAG Systems

```
┌────────────────────────────────────────────────────────────────┐
│              RAG PIPELINE TRACE                                 │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Root Span: rag_query                                     │  │
│  │ Duration: 1200ms                                         │  │
│  │                                                          │  │
│  │  ├─ Child Span: query_embedding (45ms)                   │  │
│  │  │                                                       │  │
│  │  ├─ Child Span: vector_search (120ms)                    │  │
│  │  │   ├─ Leaf: hnsw_index_lookup (30ms)                   │  │
│  │  │   └─ Leaf: rerank_results (90ms)                      │  │
│  │  │                                                       │  │
│  │  ├─ Child Span: context_assembly (25ms)                  │  │
│  │  │                                                       │  │
│  │  ├─ Child Span: llm_generation (980ms)                   │  │
│  │  │   ├─ Leaf: prompt_processing (20ms)                   │  │
│  │  │   ├─ Leaf: prefill_phase (150ms)                      │  │
│  │  │   ├─ Leaf: decode_phase (800ms)                       │  │
│  │  │   └─ Leaf: output_validation (10ms)                   │  │
│  │  │                                                       │  │
│  │  └─ Child Span: response_streaming (30ms)                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```


## 22.5 Alerting and Incident Management

### 22.5.1 Alert Tiering Strategy

| Tier | Response Time | Channels | Examples |
|------|--------------|----------|----------|
| P0 Critical | 5 min | Phone + Slack + PagerDuty | Model down, data breach, safety failure |
| P1 High | 15 min | Slack + Email | High error rate, latency spike |
| P2 Medium | 1 hour | Email + Dashboard | Quality degradation, cost anomaly |
| P3 Low | Next business day | Dashboard | Optimization opportunities |

### 22.5.2 Runbook Automation

```python
# alert_runbook.py
from typing import Dict, Callable
from enum import Enum

class AlertSeverity(Enum):
    P0 = "critical"
    P1 = "high"
    P2 = "medium"
    P3 = "low"

class AlertRunbook:
    def __init__(self):
        self.runbooks: Dict[str, Dict] = {}
    
    def register(self, alert_name: str, severity: AlertSeverity,
                 steps: list, auto_remediate: Callable = None):
        self.runbooks[alert_name] = {
            "severity": severity.value,
            "steps": steps,
            "auto_remediate": auto_remediate,
        }
    
    def execute(self, alert_name: str) -> Dict:
        if alert_name not in self.runbooks:
            return {"error": "unknown alert"}
        
        rb = self.runbooks[alert_name]
        result = {"alert": alert_name, "steps_executed": []}
        
        for i, step in enumerate(rb["steps"], 1):
            result["steps_executed"].append({
                "step": i,
                "action": step,
                "status": "completed",
            })
        
        if rb["auto_remediate"]:
            result["auto_remediation"] = "triggered"
        
        return result

# Example runbook registration
runbook = AlertRunbook()
runbook.register(
    "high_latency_p99",
    AlertSeverity.P1,
    steps=[
        "Check GPU utilization in Grafana",
        "Review recent deployments",
        "Check for traffic spikes",
        "Consider scaling up if load is sustained",
    ],
    auto_remediate=lambda: "scale_gpu_replicas(3)",
)
```

### 22.5.3 SLA Monitoring

```python
# sla_monitor.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class SLATarget:
    metric_name: str
    target_value: float
    comparison: str  # "greater", "less", "equal"
    window_hours: int = 24

class SLAMonitor:
    def __init__(self):
        self.targets: Dict[str, SLATarget] = {}
        self.breaches: list = []
    
    def set_target(self, name: str, target: SLATarget):
        self.targets[name] = target
    
    def check_sla(self, name: str, actual_value: float) -> Dict:
        if name not in self.targets:
            return {"error": "unknown SLA"}
        
        target = self.targets[name]
        met = False
        
        if target.comparison == "greater":
            met = actual_value >= target.target_value
        elif target.comparison == "less":
            met = actual_value <= target.target_value
        elif target.comparison == "equal":
            met = actual_value == target.target_value
        
        if not met:
            self.breaches.append({
                "sla": name,
                "target": target.target_value,
                "actual": actual_value,
            })
        
        return {
            "sla": name,
            "target": target.target_value,
            "actual": actual_value,
            "status": "MET" if met else "BREACHED",
        }
    
    def get_availability(self) -> float:
        if not self.breaches:
            return 100.0
        # Simplified calculation
        return max(0, 100 - len(self.breaches) * 0.1)
```
