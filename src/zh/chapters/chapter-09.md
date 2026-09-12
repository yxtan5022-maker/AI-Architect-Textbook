# 第9章：模型监控与可观测性

> **第三部分：MLOps 架构**

**学习目标：**
- 实现全面的模型漂移检测
- 设计数据漂移监控系统
- 构建性能指标监控仪表板
- 配置告警和自动响应系统
- 构建 ML 系统的可观测性平台

---

## 9.1 模型漂移检测

### 9.1.1 什么是模型漂移？

🟢 **初级**

模型漂移是指由于底层数据分布或特征与目标之间关系的变化，导致部署模型的性能随时间退化。

```
模型漂移示意图：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  准确率随时间变化：                                          │
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
│        模型部署                             性能退化         │
│                                                             │
│  漂移类型：                                                  │
│  ├── 数据漂移：输入数据分布变化                               │
│  ├── 概念漂移：特征与目标之间的关系变化                       │
│  ├── 模型漂移：模型性能退化                                  │
│  └── 上游漂移：数据管道变化影响输入                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.1.2 漂移类型

🟡 **中级**

| 漂移类型 | 描述 | 检测方法 | 示例 |
|---------|------|---------|------|
| **数据漂移** | 输入分布变化 | 统计检验（KS、PSI） | 季节性模式改变 |
| **概念漂移** | P(Y\|X) 变化 | 性能监控 | 用户行为转移 |
| **模型漂移** | 整体性能退化 | 准确率/F1 监控 | 模型过时 |
| **预测漂移** | 输出分布变化 | 分布比较 | 预测模式偏移 |
| **协变量漂移** | 特征分布变化 | 特征级监控 | 数据收集变化 |

### 9.1.3 漂移检测算法

🔴 **高级**

```python
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score

class ModelDriftDetector:
    def __init__(self, reference_data, reference_predictions):
        self.reference_data = reference_data
        self.reference_predictions = reference_predictions
    
    def detect_data_drift_ks(self, current_data, feature_idx, threshold=0.05):
        """Kolmogorov-Smirnov 检验用于数据漂移"""
        
        reference_feature = self.reference_data[:, feature_idx]
        current_feature = current_data[:, feature_idx]
        
        # 执行 KS 检验
        ks_statistic, p_value = stats.ks_2samp(reference_feature, current_feature)
        
        # 确定是否发生漂移
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
        """群体稳定性指数（PSI）用于数据漂移"""
        
        reference_feature = reference_data[:, feature_idx]
        current_feature = current_data[:, feature_idx]
        
        # 创建分箱
        n_bins = 10
        combined = np.concatenate([reference_feature, current_feature])
        bins = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
        
        # 计算比例
        ref_proportions = np.histogram(reference_feature, bins=bins)[0] / len(reference_feature)
        cur_proportions = np.histogram(current_feature, bins=bins)[0] / len(current_feature)
        
        # 避免除以零
        ref_proportions = np.where(ref_proportions == 0, 0.0001, ref_proportions)
        cur_proportions = np.where(cur_proportions == 0, 0.0001, cur_proportions)
        
        # 计算 PSI
        psi = np.sum((cur_proportions - ref_proportions) * np.log(cur_proportions / ref_proportions))
        
        # 确定漂移
        is_drifted = psi > threshold
        
        return {
            "test": "Population Stability Index",
            "feature_idx": feature_idx,
            "psi": psi,
            "is_drifted": is_drifted,
            "threshold": threshold
        }
    
    def detect_concept_drift(self, current_data, current_labels, window_size=100):
        """使用性能退化检测概念漂移"""
        
        # 计算最近数据的性能
        recent_predictions = self.model.predict(current_data[-window_size:])
        recent_labels = current_labels[-window_size:]
        recent_accuracy = accuracy_score(recent_labels, recent_predictions)
        
        # 与参考性能比较
        reference_accuracy = self.reference_accuracy
        performance_drop = reference_accuracy - recent_accuracy
        
        # 显著下降的阈值
        threshold = 0.05  # 5% 性能下降
        
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
        """检测预测分布的漂移"""
        
        # 比较预测分布
        reference_mean = np.mean(self.reference_predictions)
        reference_std = np.std(self.reference_predictions)
        
        current_mean = np.mean(current_predictions)
        current_std = np.std(current_predictions)
        
        # 计算分布偏移
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

# 用法
detector = ModelDriftDetector(reference_data, reference_predictions)

# 检查每个特征的数据漂移
for i in range(n_features):
    result = detector.detect_data_drift_ks(current_data, feature_idx=i)
    if result["is_drifted"]:
        print(f"特征 {i}：检测到漂移 (p={result['p_value']:.4f})")

# 检查概念漂移
concept_result = detector.detect_concept_drift(current_data, current_labels)
if concept_result["is_drifted"]:
    print(f"概念漂移：{concept_result['performance_drop']:.2%} 下降")
```

---

## 9.2 数据漂移监控

### 9.2.1 数据漂移监控架构

🟡 **中级**

```
数据漂移监控管道：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              数据源                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ 训练     │  │ 生产     │  │ 外部     │         │   │
│  │  │ 数据     │  │ 数据     │  │ 数据     │         │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │   │
│  │       │              │              │               │   │
│  └───────┼──────────────┼──────────────┼───────────────┘   │
│          │              │              │                    │
│          ▼              ▼              ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              特征仓库                                 │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  参考统计信息（训练数据）                       │   │   │
│  │  │  ├── 每个特征的均值、标准差、最小值、最大值    │   │   │
│  │  │  ├── 分布直方图                               │   │   │
│  │  │  └── 相关性矩阵                               │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │              漂移检测引擎                             │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ 统计     │  │ 机器学习 │  │ 业务     │         │   │
│  │  │ 检验     │  │ 检测     │  │ 规则     │         │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │   │
│  │       │              │              │               │   │
│  │       └──────────────┼──────────────┘               │   │
│  │                      │                              │   │
│  │                      ▼                              │   │
│  │              ┌──────────────┐                       │   │
│  │              │ 漂移分数     │                       │   │
│  │              │ 计算器       │                       │   │
│  │              └──────────────┘                       │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │              告警与响应                               │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ 告警     │  │ 自动     │  │ 仪表板   │         │   │
│  │  │ 系统     │  │ 重训练   │  │ 展示     │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2.2 使用 Evidently 实现数据漂移检测

🔴 **高级**

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
        """设置 Evidently 的列映射"""
        
        self.column_mapping = ColumnMapping(
            target=target_column,
            numerical_features=numerical_columns,
            categorical_features=categorical_columns
        )
    
    def generate_drift_report(self, current_data: pd.DataFrame, 
                             save_path: str = None) -> dict:
        """生成全面的漂移报告"""
        
        # 创建漂移报告
        drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset()
        ])
        
        # 运行报告
        drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # 获取结果
        report_dict = drift_report.as_dict()
        
        # 如果提供路径则保存报告
        if save_path:
            drift_report.save_html(save_path)
        
        # 提取关键指标
        result = {
            "dataset_drift": report_dict["metrics"][0]["result"]["dataset_drift"],
            "drift_score": report_dict["metrics"][0]["result"]["drift_score"],
            "n_drifted_columns": report_dict["metrics"][0]["result"]["n_drifted_columns"],
            "drifted_columns": report_dict["metrics"][0]["result"]["drifted_columns"]
        }
        
        return result
    
    def monitor_real_time(self, current_batch: pd.DataFrame, 
                         threshold: float = 0.5) -> dict:
        """流数据的实时漂移监控"""
        
        # 计算批次的漂移分数
        drift_result = self.generate_drift_report(current_batch)
        
        # 检查阈值
        needs_alert = drift_result["drift_score"] > threshold
        
        # 确定操作
        if needs_alert:
            action = {
                "type": "alert",
                "severity": "high" if drift_result["drift_score"] > 0.8 else "medium",
                "message": f"检测到数据漂移：分数={drift_result['drift_score']:.3f}",
                "drifted_features": drift_result["drifted_columns"]
            }
        else:
            action = {
                "type": "continue",
                "message": "未检测到显著漂移"
            }
        
        return {
            "drift_result": drift_result,
            "action": action,
            "timestamp": pd.Timestamp.now()
        }

# 用法
monitor = DataDriftMonitor(reference_data=train_df)
monitor.set_column_mapping(
    target_column="target",
    numerical_columns=["feature1", "feature2", "feature3"],
    categorical_columns=["category1", "category2"]
)

# 生成报告
result = monitor.generate_drift_report(current_data=production_df)
print(f"数据集漂移：{result['dataset_drift']}")
print(f"漂移分数：{result['drift_score']:.3f}")
print(f"漂移特征：{result['drifted_columns']}")
```

---

## 9.3 性能指标监控

### 9.3.1 性能指标框架

🟢 **初级**

```
ML 性能指标框架：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  模型质量指标：                                              │
│  ├── 准确率                                                │
│  ├── 精确率 / 召回率 / F1 分数                               │
│  ├── AUC-ROC                                               │
│  ├── 均方误差（MSE）                                        │
│  └── 平均绝对误差（MAE）                                    │
│                                                             │
│  运营指标：                                                  │
│  ├── 延迟（P50, P90, P95, P99）                            │
│  ├── 吞吐量（QPS）                                         │
│  ├── 错误率                                                │
│  └── 可用性                                                │
│                                                             │
│  业务指标：                                                  │
│  ├── 转化率                                                │
│  ├── 每次预测收入                                           │
│  ├── 用户满意度评分                                         │
│  └── 每次预测成本                                           │
│                                                             │
│  系统指标：                                                  │
│  ├── CPU / 内存使用                                         │
│  ├── GPU 利用率                                             │
│  ├── 网络 I/O                                              │
│  └── 磁盘 I/O                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.3.2 构建监控仪表板

🟡 **中级**

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import numpy as np

class MLMetricsCollector:
    def __init__(self, port=8000):
        self.port = port
        
        # 定义指标
        self.prediction_counter = Counter(
            'ml_predictions_total',
            '预测总数',
            ['model_name', 'model_version']
        )
        
        self.prediction_latency = Histogram(
            'ml_prediction_latency_seconds',
            '预测延迟（秒）',
            ['model_name', 'model_version'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        )
        
        self.prediction_errors = Counter(
            'ml_prediction_errors_total',
            '预测错误总数',
            ['model_name', 'model_version', 'error_type']
        )
        
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            '当前模型准确率',
            ['model_name', 'model_version']
        )
        
        self.data_drift_score = Gauge(
            'ml_data_drift_score',
            '当前数据漂移分数',
            ['model_name', 'feature_name']
        )
        
        self.gpu_utilization = Gauge(
            'ml_gpu_utilization_percent',
            'GPU 利用率百分比',
            ['gpu_id']
        )
        
        self.memory_usage = Gauge(
            'ml_memory_usage_bytes',
            '内存使用量（字节）',
            ['model_name']
        )
    
    def start_metrics_server(self):
        """启动 Prometheus 指标服务器"""
        start_http_server(self.port)
        print(f"指标服务器在端口 {self.port} 启动")
    
    def record_prediction(self, model_name: str, model_version: str, 
                         latency: float, success: bool):
        """记录预测事件"""
        
        # 递增预测计数器
        self.prediction_counter.labels(
            model_name=model_name, 
            model_version=model_version
        ).inc()
        
        # 记录延迟
        self.prediction_latency.labels(
            model_name=model_name, 
            model_version=model_version
        ).observe(latency)
        
        # 如果失败则记录错误
        if not success:
            self.prediction_errors.labels(
                model_name=model_name,
                model_version=model_version,
                error_type="prediction_failed"
            ).inc()
    
    def update_accuracy(self, model_name: str, model_version: str, accuracy: float):
        """更新模型准确率指标"""
        self.model_accuracy.labels(
            model_name=model_name,
            model_version=model_version
        ).set(accuracy)
    
    def update_drift_score(self, model_name: str, feature_name: str, score: float):
        """更新数据漂移分数"""
        self.data_drift_score.labels(
            model_name=model_name,
            feature_name=feature_name
        ).set(score)
    
    def update_system_metrics(self, gpu_id: int, gpu_util: float, 
                             memory_bytes: int, model_name: str):
        """更新系统指标"""
        self.gpu_utilization.labels(gpu_id=str(gpu_id)).set(gpu_util)
        self.memory_usage.labels(model_name=model_name).set(memory_bytes)

# 用法
collector = MLMetricsCollector(port=8000)
collector.start_metrics_server()

# 记录预测
for prediction in predictions:
    start_time = time.time()
    
    # 进行预测
    try:
        result = model.predict(prediction)
        success = True
    except Exception as e:
        success = False
    
    latency = time.time() - start_time
    
    # 记录指标
    collector.record_prediction(
        model_name="text-classifier",
        model_version="v1.0",
        latency=latency,
        success=success
    )
```

### 9.3.3 Grafana 仪表板配置

🟡 **中级**

```json
{
  "dashboard": {
    "title": "ML 模型监控",
    "panels": [
      {
        "title": "预测速率",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_predictions_total[5m])",
            "legendFormat": "{{model_name}} - {{model_version}}"
          }
        ]
      },
      {
        "title": "预测延迟",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))",
            "legendFormat": "P95 延迟"
          },
          {
            "expr": "histogram_quantile(0.50, rate(ml_prediction_latency_seconds_bucket[5m]))",
            "legendFormat": "P50 延迟"
          }
        ]
      },
      {
        "title": "模型准确率",
        "type": "stat",
        "targets": [
          {
            "expr": "ml_model_accuracy",
            "legendFormat": "{{model_name}}"
          }
        ]
      },
      {
        "title": "错误率",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_prediction_errors_total[5m])",
            "legendFormat": "{{error_type}}"
          }
        ]
      },
      {
        "title": "数据漂移分数",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_data_drift_score",
            "legendFormat": "{{feature_name}}"
          }
        ]
      },
      {
        "title": "GPU 利用率",
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

## 9.4 告警与自动化响应

### 9.4.1 告警策略

🟡 **中级**

```
告警策略框架：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  告警严重级别：                                               │
│  ├── 严重（P0）：模型完全故障                                │
│  │   ├── 即时通知（PagerDuty）                               │
│  │   ├── 触发自动回滚                                       │
│  │   └── 需要人工干预                                       │
│  │                                                         │
│  ├── 警告（P1）：显著退化                                    │
│  │   ├── 团队通知（Slack）                                  │
│  │   ├── 增加监控频率                                       │
│  │   └── 需要调查                                           │
│  │                                                         │
│  ├── 信息（P2）：检测到小问题                                │
│  │   ├── 日志条目                                           │
│  │   ├── 仪表板更新                                         │
│  │   └── 业务时间审查                                       │
│  │                                                         │
│  └── 低（P3）：潜在关注                                     │
│      ├── 指标记录                                           │
│      └── 每周审查                                           │
│                                                             │
│  告警渠道：                                                  │
│  ├── PagerDuty（严重）                                      │
│  ├── Slack（警告）                                          │
│  ├── 邮件（信息）                                           │
│  └── 仪表板（所有）                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.4.2 Prometheus 告警规则

🔴 **高级**

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
    # 模型性能告警
    - alert: ModelAccuracyBelowThreshold
      expr: ml_model_accuracy < 0.85
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "模型准确率低于阈值"
        description: "模型 {{ $labels.model_name }} 准确率为 {{ $value }}"
    
    - alert: ModelAccuracyCritical
      expr: ml_model_accuracy < 0.75
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "模型准确率严重偏低"
        description: "模型 {{ $labels.model_name }} 准确率为 {{ $value }}"
    
    # 延迟告警
    - alert: HighPredictionLatency
      expr: histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m])) > 0.5
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "预测延迟过高"
        description: "P95 延迟为 {{ $value }}s"
    
    - alert: CriticalPredictionLatency
      expr: histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m])) > 1.0
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "预测延迟严重"
        description: "P95 延迟为 {{ $value }}s"
    
    # 错误率告警
    - alert: HighErrorRate
      expr: rate(ml_prediction_errors_total[5m]) / rate(ml_predictions_total[5m]) > 0.05
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "错误率过高"
        description: "错误率为 {{ $value | humanizePercentage }}"
    
    # 数据漂移告警
    - alert: DataDriftDetected
      expr: ml_data_drift_score > 0.5
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "检测到数据漂移"
        description: "特征 {{ $labels.feature_name }} 漂移分数为 {{ $value }}"
    
    - alert: CriticalDataDrift
      expr: ml_data_drift_score > 0.8
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "严重数据漂移"
        description: "特征 {{ $labels.feature_name }} 漂移分数为 {{ $value }}"
    
    # 资源告警
    - alert: HighGPUUtilization
      expr: ml_gpu_utilization_percent > 90
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "GPU 利用率过高"
        description: "GPU {{ $labels.gpu_id }} 利用率为 {{ $value }}%"
    
    # 预测量告警
    - alert: LowPredictionVolume
      expr: rate(ml_predictions_total[15m]) < 10
      for: 15m
      labels:
        severity: info
      annotations:
        summary: "预测量偏低"
        description: "预测速率为 {{ $value }} req/s"
```

### 9.4.3 自动化响应系统

🔴 **高级**

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
        
        # 初始化组件
        self.prometheus_url = config.get('prometheus_url', 'http://localhost:9090')
        self.alertmanager_url = config.get('alertmanager_url', 'http://localhost:9093')
        self.kubernetes_api = config.get('kubernetes_api', 'https://kubernetes.default.svc')
        
    def check_model_health(self, model_name: str, model_version: str) -> Dict:
        """检查模型整体健康状态"""
        
        # 查询 Prometheus 指标
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
        
        # 确定健康状态
        health_status = {
            'model_name': model_name,
            'model_version': model_version,
            'timestamp': datetime.now().isoformat(),
            'metrics': results,
            'status': 'healthy',
            'issues': []
        }
        
        # 检查阈值
        if results.get('accuracy', 1.0) < 0.85:
            health_status['status'] = 'degraded'
            health_status['issues'].append('准确率偏低')
        
        if results.get('latency_p95', 0) > 0.5:
            health_status['status'] = 'degraded'
            health_status['issues'].append('延迟过高')
        
        if results.get('error_rate', 0) > 0.05:
            health_status['status'] = 'critical'
            health_status['issues'].append('错误率过高')
        
        if results.get('drift_score', 0) > 0.5:
            health_status['status'] = 'warning'
            health_status['issues'].append('检测到数据漂移')
        
        return health_status
    
    def auto_rollback(self, model_name: str, reason: str) -> bool:
        """自动回滚到之前的模型版本"""
        
        self.logger.warning(f"触发自动回滚：{model_name}，原因：{reason}")
        
        # 获取之前稳定的版本
        previous_version = self._get_previous_stable_version(model_name)
        
        if previous_version:
            # 更新 SeldonDeployment
            success = self._update_seldon_deployment(model_name, previous_version)
            
            if success:
                # 发送通知
                self._send_notification(
                    severity="critical",
                    title=f"自动回滚：{model_name}",
                    message=f"已回滚到版本 {previous_version}。原因：{reason}"
                )
                
                # 记录回滚事件
                self._log_rollback_event(model_name, previous_version, reason)
                
                return True
        
        return False
    
    def auto_scale(self, model_name: str, metric: str, target_value: float) -> bool:
        """根据指标自动扩缩模型部署"""
        
        # 获取当前副本数
        current_replicas = self._get_current_replicas(model_name)
        
        # 根据指标计算期望副本数
        if metric == 'latency':
            current_latency = self._get_metric(model_name, 'latency_p95')
            if current_latency > target_value * 1.2:
                desired_replicas = min(current_replicas + 2, 10)  # 扩容
            elif current_latency < target_value * 0.8:
                desired_replicas = max(current_replicas - 1, 2)  # 缩容
            else:
                desired_replicas = current_replicas
        elif metric == 'queue_depth':
            current_queue = self._get_metric(model_name, 'queue_depth')
            if current_queue > target_value:
                desired_replicas = min(current_replicas + 1, 10)
            else:
                desired_replicas = current_replicas
        
        # 如果需要则应用扩缩
        if desired_replicas != current_replicas:
            return self._scale_deployment(model_name, desired_replicas)
        
        return True
    
    def trigger_retraining(self, model_name: str, reason: str) -> str:
        """触发模型重训练管道"""
        
        # 创建 Kubeflow Pipeline 运行
        pipeline_run = self._create_kubeflow_run(
            pipeline_name="model-retraining",
            params={
                "model_name": model_name,
                "reason": reason,
                "trigger_time": datetime.now().isoformat()
            }
        )
        
        # 发送通知
        self._send_notification(
            severity="info",
            title=f"重训练已触发：{model_name}",
            message=f"管道运行已启动：{pipeline_run['run_id']}。原因：{reason}"
        )
        
        return pipeline_run['run_id']
    
    def _get_previous_stable_version(self, model_name: str) -> str:
        """获取之前稳定的模型版本"""
        # 实现取决于模型注册中心
        pass
    
    def _update_seldon_deployment(self, model_name: str, version: str) -> bool:
        """更新 SeldonDeployment 使用指定版本"""
        # 实现取决于 Kubernetes API
        pass
    
    def _send_notification(self, severity: str, title: str, message: str):
        """通过配置的渠道发送通知"""
        
        if severity == "critical":
            # PagerDuty
            self._send_pagerduty(title, message)
        
        # Slack
        self._send_slack(severity, title, message)
        
        # 邮件
        self._send_email(severity, title, message)
    
    def _send_pagerduty(self, title: str, message: str):
        """发送 PagerDuty 告警"""
        pass
    
    def _send_slack(self, severity: str, title: str, message: str):
        """发送 Slack 通知"""
        pass
    
    def _send_email(self, severity: str, title: str, message: str):
        """发送邮件通知"""
        pass
    
    def _log_rollback_event(self, model_name: str, version: str, reason: str):
        """记录回滚事件用于审计"""
        pass

# 用法
response_system = AutomatedResponseSystem({
    'prometheus_url': 'http://prometheus:9090',
    'alertmanager_url': 'http://alertmanager:9093',
    'kubernetes_api': 'https://kubernetes.default.svc'
})

# 检查模型健康状态
health = response_system.check_model_health("text-classifier", "v1.0")
print(f"模型状态：{health['status']}")
print(f"问题：{health['issues']}")

# 如果严重则自动回滚
if health['status'] == 'critical':
    response_system.auto_rollback("text-classifier", health['issues'][0])
```

---

## 9.5 可观测性平台架构

### 9.5.1 ML 可观测性的三大支柱

🟡 **中级**

```
ML 可观测性支柱：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              日志                                     │   │
│  │  ├── 预测日志                                        │   │
│  │  ├── 错误日志                                        │   │
│  │  ├── 审计日志                                        │   │
│  │  └── 系统日志                                        │   │
│  │                                                     │   │
│  │  工具：ELK Stack, Fluentd, Loki                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              指标                                     │   │
│  │  ├── 模型指标（准确率、延迟）                         │   │
│  │  ├── 系统指标（CPU、内存、GPU）                       │   │
│  │  ├── 业务指标（转化率、收入）                         │   │
│  │  └── 数据指标（漂移、质量）                           │   │
│  │                                                     │   │
│  │  工具：Prometheus, Grafana, DataDog                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              追踪                                     │   │
│  │  ├── 请求追踪                                        │   │
│  │  ├── 模型推理追踪                                     │   │
│  │  ├── 数据管道追踪                                     │   │
│  │  └── 分布式追踪                                      │   │
│  │                                                     │   │
│  │  工具：Jaeger, Zipkin, OpenTelemetry                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.5.2 完整可观测性架构

🔴 **高级**

```
ML 可观测性平台架构：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              数据收集层                               │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ 模型     │  │ 系统     │  │ 业务     │         │   │
│  │  │ 指标     │  │ 指标     │  │ 指标     │         │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │   │
│  │       │              │              │               │   │
│  │       ▼              ▼              ▼               │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │           Prometheus                         │   │   │
│  │  │  ├── 时序存储                                │   │   │
│  │  │  ├── PromQL 查询                            │   │   │
│  │  │  └── 告警规则                               │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              可视化层                                 │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              Grafana                         │   │   │
│  │  │  ├── 模型性能仪表板                          │   │   │
│  │  │  ├── 系统健康仪表板                          │   │   │
│  │  │  ├── 业务指标仪表板                          │   │   │
│  │  │  └── 告警管理                                │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              分析层                                   │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ 漂移     │  │ 异常     │  │ 根因     │         │   │
│  │  │ 检测     │  │ 检测     │  │ 分析     │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              响应层                                   │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ 告警     │  │ 自动     │  │ 反馈     │         │   │
│  │  │ 系统     │  │ 响应     │  │ 循环     │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.5.3 OpenTelemetry 集成

🔴 **高级**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
import time

class MLTracingSetup:
    def __init__(self, service_name: str, jaeger_endpoint: str):
        # 创建资源
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        # 配置追踪器
        provider = TracerProvider(resource=resource)
        
        # 配置 Jaeger 导出器
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        # 添加处理器
        processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(processor)
        
        # 设置全局追踪器
        trace.set_tracer_provider(provider)
        
        self.tracer = trace.get_tracer(__name__)
    
    def trace_prediction(self, model_name: str, input_data: dict):
        """追踪预测请求"""
        
        with self.tracer.start_as_current_span("prediction") as span:
            # 添加属性
            span.set_attribute("model.name", model_name)
            span.set_attribute("input.size", len(str(input_data)))
            
            # 开始预处理 span
            with self.tracer.start_as_current_span("preprocessing") as preprocess_span:
                start_time = time.time()
                # 预处理逻辑
                processed_data = self._preprocess(input_data)
                preprocess_span.set_attribute("preprocessing.duration", 
                                            time.time() - start_time)
            
            # 开始推理 span
            with self.tracer.start_as_current_span("inference") as inference_span:
                start_time = time.time()
                # 推理逻辑
                prediction = self._predict(processed_data)
                inference_span.set_attribute("inference.duration", 
                                           time.time() - start_time)
                inference_span.set_attribute("prediction.confidence", 
                                           prediction.get('confidence', 0))
            
            # 开始后处理 span
            with self.tracer.start_as_current_span("postprocessing") as postprocess_span:
                start_time = time.time()
                # 后处理逻辑
                result = self._postprocess(prediction)
                postprocess_span.set_attribute("postprocessing.duration", 
                                             time.time() - start_time)
            
            return result
    
    def _preprocess(self, data):
        """预处理逻辑"""
        pass
    
    def _predict(self, data):
        """推理逻辑"""
        pass
    
    def _postprocess(self, prediction):
        """后处理逻辑"""
        pass

# 用法
tracing = MLTracingSetup(
    service_name="text-classifier",
    jaeger_endpoint="http://jaeger:14268/api/traces"
)

# 追踪预测
result = tracing.trace_prediction(
    model_name="text-classifier",
    input_data={"text": "这是一个很好的产品！"}
)
```

---

## 💡 案例研究：Prometheus + Grafana AI 监控系统

### 系统架构

```
Prometheus + Grafana AI 监控系统：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ML 服务                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ 模型     │  │ 训练     │  │ 数据     │         │   │
│  │  │ 服务     │  │ 管道     │  │ 管道     │         │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │   │
│  │       │              │              │               │   │
│  └───────┼──────────────┼──────────────┼───────────────┘   │
│          │              │              │                    │
│          ▼              ▼              ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              指标导出器                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ 自定义   │  │ Node     │  │ cAdvisor │         │   │
│  │  │ 导出器   │  │ 导出器   │  │          │         │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘         │   │
│  │       │              │              │               │   │
│  └───────┼──────────────┼──────────────┼───────────────┘   │
│          │              │              │                    │
│          ▼              ▼              ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Prometheus 服务器                        │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  ├── 抓取间隔（15s-60s）                    │   │   │
│  │  │  ├── 保留期（30天）                          │   │   │
│  │  │  ├── 告警规则                               │   │   │
│  │  │  └── 记录规则                               │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Grafana 仪表板                           │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  仪表板 1：模型性能                           │   │   │
│  │  │  ├── 准确率趋势                              │   │   │
│  │  │  ├── 延迟分布                                │   │   │
│  │  │  ├── 错误率                                  │   │   │
│  │  │  └── 预测量                                  │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  仪表板 2：数据质量                           │   │   │
│  │  │  ├── 数据漂移分数                             │   │   │
│  │  │  ├── 特征分布                                │   │   │
│  │  │  ├── 缺失值                                  │   │   │
│  │  │  └── 数据新鲜度                              │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  仪表板 3：系统健康                           │   │   │
│  │  │  ├── CPU/内存使用                            │   │   │
│  │  │  ├── GPU 利用率                              │   │   │
│  │  │  ├── 网络 I/O                               │   │   │
│  │  │  └── 磁盘使用                                │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              告警管道                                 │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │Prometheus│  │ Alert    │  │ 通知     │         │   │
│  │  │ 告警     │──│ Manager  │──│          │         │   │
│  │  │ 规则     │  │          │  │          │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                     │              │               │   │
│  │                     ▼              ▼               │   │
│  │              ┌──────────┐  ┌──────────┐           │   │
│  │              │ Slack    │  │PagerDuty │           │   │
│  │              │          │  │          │           │   │
│  │              └──────────┘  └──────────┘           │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 部署说明

```bash
# 1. 创建监控命名空间
kubectl create namespace monitoring

# 2. 部署 Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --set alertmanager.enabled=true

# 3. 部署 Grafana
helm repo add grafana https://grafana.github.io/helm-charts
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword=admin123

# 4. 部署自定义 ML 导出器
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

# 5. 访问仪表板
kubectl port-forward svc/grafana 3000:80 -n monitoring
# Grafana：http://localhost:3000 (admin/admin123)

kubectl port-forward svc/prometheus-server 9090:80 -n monitoring
# Prometheus：http://localhost:9090
```

### 仪表板配置

```json
{
  "dashboard": {
    "title": "ML 模型监控仪表板",
    "uid": "ml-model-monitoring",
    "panels": [
      {
        "title": "模型准确率趋势",
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
        "title": "预测延迟（P95）",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket{model_name=\"text-classifier\"}[5m]))",
            "legendFormat": "P95 延迟"
          }
        ]
      },
      {
        "title": "数据漂移分数",
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
        "title": "GPU 利用率",
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

## 总结

**关键要点：**

1. **模型漂移**是不可避免的——持续监控至关重要
2. **数据漂移检测**需要统计严谨性和适当的基线
3. **性能监控**必须涵盖模型质量、运营和业务指标
4. **告警**应该是可操作的且分级的
5. **可观测性**需要日志、指标和追踪协同工作

**最佳实践：**

- 在模型开发期间建立清晰的基线
- 实现具有可配置阈值的自动漂移检测
- 构建能讲故事的全面仪表板
- 为常见问题创建操作手册
- 定期审查和更新监控规则

**完整 MLOps 架构：**

```
┌─────────────────────────────────────────────────────────────────┐
│                    完整 MLOps 架构                                │
│                                                                 │
│  第6章：MLOps 基础                                               │
│  ├── 成熟度模型（Level 0-3）                                     │
│  ├── 工具链选择                                                  │
│  └── DevOps vs MLOps                                           │
│                                                                 │
│  第7章：模型训练                                                 │
│  ├── 训练环境设计                                                │
│  ├── 分布式训练                                                 │
│  ├── HPO 与实验跟踪                                             │
│  └── 资源管理                                                  │
│                                                                 │
│  第8章：模型部署                                                 │
│  ├── 部署策略                                                   │
│  ├── 模型服务（Seldon Core）                                    │
│  ├── A/B 测试与金丝雀发布                                       │
│  └── 推理优化                                                  │
│                                                                 │
│  第9章：模型监控                                                 │
│  ├── 漂移检测                                                   │
│  ├── 性能监控                                                   │
│  ├── 告警与响应                                                 │
│  └── 可观测性平台                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*第9章结束*
*第三部分 MLOps 架构结束*
