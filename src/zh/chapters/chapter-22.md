# 第22章：AI可观测性与监控

**读者级别：** 🔴 高级 | **页数：** 25 | **代码：** Python, PromQL, Grafana

---

## 22.1 AI可观测性三大支柱

### 22.1.1 日志、指标和追踪

```
┌────────────────────────────────────────────────────────────┐
│              AI可观测性技术栈                                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    日志                               │   │
│  │  • 请求/响应对                                       │   │
│  │  • 错误消息和堆栈跟踪                                │   │
│  │  • 合规审计追踪                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   指标                               │   │
│  │  • 延迟（TTFT, token间, p50/p99）                   │   │
│  │  • 吞吐量（tokens/sec, RPS）                        │   │
│  │  • 资源利用率（GPU, 内存, CPU）                      │   │
│  │  • 质量指标（幻觉率, 毒性）                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   追踪                               │   │
│  │  • 跨服务的请求生命周期                              │   │
│  │  • 模型推理流水线阶段                                │   │
│  │  • RAG检索与生成关联                                 │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### 22.1.2 AI特定指标

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
    error_count: int = 0

class MetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, AIMetrics] = defaultdict(AIMetrics)
    
    def record_request(self, model_id, latency, tokens, error=False):
        m = self.metrics[model_id]
        m.request_count += 1
        m.token_count += tokens
        m.latencies.append(latency)
        if error:
            m.error_count += 1
    
    def get_summary(self, model_id):
        m = self.metrics[model_id]
        if not m.latencies:
            return {"error": "no data"}
        sorted_lat = sorted(m.latencies)
        return {
            "total_requests": m.request_count,
            "latency_p50_ms": sorted_lat[len(sorted_lat)//2] * 1000,
            "error_rate": m.error_count / max(m.request_count, 1),
        }
```

## 22.2 生产监控技术栈

### 22.2.1 架构概览

```
┌────────────────────────────────────────────────────────────────┐
│                 生产监控技术栈                                   │
│                                                                 │
│  数据收集：                                                     │
│  ├── OpenTelemetry SDK（追踪、指标、日志）                      │
│  ├── Prometheus客户端（系统指标）                               │
│  └── 自定义AI指标导出器                                        │
│                                                                 │
│  存储：                                                         │
│  ├── Prometheus（时序指标）                                    │
│  ├── Loki（日志聚合）                                         │
│  ├── Tempo/Jaeger（分布式追踪）                               │
│  └── ClickHouse（查询分析）                                   │
│                                                                 │
│  可视化：                                                       │
│  ├── Grafana仪表板                                            │
│  ├── 自定义AI质量仪表板                                        │
│  └── 告警管理器（PagerDuty, Slack）                           │
└────────────────────────────────────────────────────────────────┘
```

### 22.2.2 Prometheus指标配置

```yaml
# prometheus_ai_metrics.yaml
global:
  scrape_interval: 15s

rule_files:
  - "ai_alerts.yml"

scrape_configs:
  - job_name: 'ai-inference-service'
    static_configs:
      - targets: ['ai-service:8080']

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
          summary: "P99延迟超过2秒"
          
      - alert: HighErrorRate
        expr: rate(ai_requests_errors_total[5m]) / rate(ai_requests_total[5m]) > 0.05
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "错误率超过5%"
```

## 22.3 质量监控

### 22.3.1 幻觉检测

```python
# hallucination_detector.py
class HallucinationDetector:
    def detect(self, response, source_documents):
        scores = {}
        scores["self_consistency"] = self._self_consistency(response)
        scores["knowledge_grounding"] = self._knowledge_grounding(
            response, source_documents
        )
        scores["hallucination_probability"] = 1 - (
            scores["self_consistency"] * 0.3 +
            scores["knowledge_grounding"] * 0.7
        )
        return scores
    
    def _self_consistency(self, response):
        return 0.85  # 占位符
    
    def _knowledge_grounding(self, response, sources):
        return 0.80  # 占位符
```

### 22.3.2 漂移检测

```python
# drift_detector.py
import numpy as np
from scipy import stats

class DriftDetector:
    def __init__(self, baseline_window=1000):
        self.baseline_data = []
        self.current_data = []
    
    def detect_drift(self, threshold=0.05):
        if len(self.baseline_data) < 100:
            return {"drift_detected": False}
        stat, p_value = stats.ks_2samp(self.baseline_data, self.current_data)
        return {
            "drift_detected": p_value < threshold,
            "p_value": p_value,
        }
```

## 22.4 分布式追踪

### 22.4.1 AI流水线追踪模型

```
┌────────────────────────────────────────────────────────────────┐
│              RAG流水线追踪                                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 根Span: rag_query (1200ms)                               │  │
│  │                                                          │  │
│  │  ├─ 子Span: query_embedding (45ms)                       │  │
│  │  ├─ 子Span: vector_search (120ms)                        │  │
│  │  │   ├─ 叶: hnsw_index_lookup (30ms)                     │  │
│  │  │   └─ 叶: rerank_results (90ms)                        │  │
│  │  ├─ 子Span: context_assembly (25ms)                      │  │
│  │  ├─ 子Span: llm_generation (980ms)                       │  │
│  │  │   ├─ 叶: prompt_processing (20ms)                     │  │
│  │  │   ├─ 叶: prefill_phase (150ms)                        │  │
│  │  │   └─ 叶: decode_phase (800ms)                         │  │
│  │  └─ 子Span: response_streaming (30ms)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## 22.5 告警与事件管理

### 22.5.1 告警分级策略

| 级别 | 响应时间 | 渠道 | 示例 |
|------|---------|------|------|
| P0 严重 | 5分钟 | 电话+Slack+PagerDuty | 模型宕机、数据泄露、安全失败 |
| P1 高 | 15分钟 | Slack+邮件 | 高错误率、延迟飙升 |
| P2 中 | 1小时 | 邮件+仪表板 | 质量下降、成本异常 |
| P3 低 | 下一工作日 | 仪表板 | 优化机会 |
