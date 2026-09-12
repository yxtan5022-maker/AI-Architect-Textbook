# 第8章：模型部署架构

> **第三部分：MLOps 架构**

**学习目标：**
- 比较并选择合适的部署策略
- 设计可扩展的模型服务架构
- 实现 A/B 测试和金丝雀发布
- 有效管理模型版本
- 优化推理性能
- 规划边缘部署策略

---

## 8.1 部署策略对比

### 8.1.1 部署决策框架

🟢 **初级**

选择正确的部署策略至关重要。错误的选择可能导致停机、性能不佳或昂贵的回滚。

```
部署策略选择矩阵：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  风险承受能力：                                               │
│  低 ──────────────────────────────────────────────► 高       │
│  │                                                    │    │
│  ▼                                                    ▼    │
│  蓝绿部署                                  金丝雀发布         │
│  影子部署                                  A/B 测试          │
│                                                              │
│  停机容忍度：                                                  │
│  零 ──────────────────────────────────────────────► 任何     │
│  │                                                    │    │
│  ▼                                                    ▼    │
│  蓝绿部署                                  滚动更新           │
│  金丝雀发布                                重新创建           │
│                                                              │
│  流量分配需求：                                                │
│  无 ─────────────────────────────────────────────► 全量      │
│  │                                                    │    │
│  ▼                                                    ▼    │
│  重新创建                                  A/B 测试          │
│  滚动更新                                  金丝雀发布         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.1.2 部署策略概览

🟡 **中级**

| 策略 | 停机 | 风险 | 复杂性 | 适用场景 |
|------|------|------|--------|---------|
| **重新创建** | 有 | 高 | 低 | 开发环境、内部工具 |
| **滚动更新** | 无 | 中 | 中 | 通用生产环境 |
| **蓝绿部署** | 无 | 低 | 中 | 关键服务 |
| **金丝雀发布** | 无 | 极低 | 高 | 低风险生产环境 |
| **影子部署** | 无 | 无 | 极高 | 预生产验证 |
| **A/B 测试** | 无 | 低 | 高 | 业务优化 |

### 8.1.3 详细策略描述

🔴 **高级**

```
重新创建策略：
┌─────────────────────────────────────────────────────────────┐
│  时间 ──────────────────────────────────────────────────►   │
│                                                             │
│  V1：████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  V2：░░░░░░░░░░░░░░░░████████████████████████████████    │
│       ▲                  ▲                                  │
│       │                  │                                  │
│    V1 停止           V2 启动                                 │
│    （停机）           （新版本）                               │
│                                                             │
│  优点：简单，干净的过渡                                       │
│  缺点：停机时间，没有重新部署无法回滚                          │
└─────────────────────────────────────────────────────────────┘

滚动更新策略：
┌─────────────────────────────────────────────────────────────┐
│  时间 ──────────────────────────────────────────────────►   │
│                                                             │
│  V1：████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  V2：░░░░░░░░████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  V3：░░░░░░░░░░░░░░░░████████░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  V4：░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████    │
│       ▲                          ▲                   ▲      │
│       │                          │                   │      │
│    开始滚动                   中间过渡            完成       │
│                                                             │
│  优点：无停机时间，渐进式发布                                  │
│  缺点：可能存在版本不匹配，回滚复杂                            │
└─────────────────────────────────────────────────────────────┘

蓝绿部署策略：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐          ┌──────────────┐               │
│  │  蓝色 (V1)   │◄─ LB ──►│  绿色 (V2)   │               │
│  │  生产环境     │          │  预发布环境    │               │
│  └──────────────┘          └──────────────┘               │
│         │                          │                        │
│         ▼                          ▼                        │
│  ┌──────────────┐          ┌──────────────┐               │
│  │  100% 流量   │          │  0% 流量     │               │
│  └──────────────┘          └──────────────┘               │
│                                                             │
│  验证后：                                                    │
│  ┌──────────────┐          ┌──────────────┐               │
│  │  蓝色 (V1)   │◄─ LB ──►│  绿色 (V2)   │               │
│  │  预发布环境    │          │  生产环境     │               │
│  └──────────────┘          └──────────────┘               │
│         │                          │                        │
│         ▼                          ▼                        │
│  ┌──────────────┐          ┌──────────────┐               │
│  │  0% 流量     │          │  100% 流量   │               │
│  └──────────────┘          └──────────────┘               │
│                                                             │
│  优点：即时回滚，零停机时间                                   │
│  缺点：双倍基础设施成本                                       │
└─────────────────────────────────────────────────────────────┘
```

### 8.1.4 影子部署

🔴 **高级**

```
影子部署：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    ┌──────────────┐                        │
│  请求 ──────────►│   路由器      │                        │
│                    └──────┬───────┘                        │
│                           │                                 │
│                    ┌──────┴───────┐                        │
│                    │              │                         │
│                    ▼              ▼                         │
│             ┌──────────┐  ┌──────────┐                    │
│             │  V1      │  │  V2      │                    │
│             │(主要)    │  │ (影子)    │                    │
│             └────┬─────┘  └────┬─────┘                    │
│                  │              │                           │
│                  ▼              ▼                           │
│             ┌──────────┐  ┌──────────┐                    │
│             │ 响应     │  │ 响应     │                    │
│             │ (使用)   │  │ (记录)   │                    │
│             └──────────┘  └──────────┘                    │
│                                                             │
│  V2 接收生产流量但其响应不会返回给用户——仅记录用于比较         │
│                                                             │
│  优点：对用户零风险，比较真实流量                              │
│  缺点：双倍计算成本，路由复杂                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 8.2 模型服务架构

### 8.2.1 模型服务的挑战

🟢 **初级**

模型服务是使训练好的模型可用于预测请求的过程。它比传统 Web 应用的服务更复杂，因为：

1. 模型很大（有时达数 GB）
2. 推理需要特定硬件（深度学习需要 GPU）
3. 延迟要求各不相同（实时 vs 批处理）
4. 模型需要版本控制和回滚能力

```
模型服务要求：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  延迟要求：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  实时：      < 100ms   （推荐、聊天）                 │   │
│  │  准实时：    < 1s      （搜索、个性化）               │   │
│  │  批处理：    分钟级     （分析、报表）                 │   │
│  │  离线：      小时级     （训练、重训练）               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  吞吐量要求：                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  低：        < 100 QPS  （内部工具）                  │   │
│  │  中：        100-1K QPS（小型产品）                   │   │
│  │  高：        1K-10K QPS（大型产品）                   │   │
│  │  极高：      > 10K QPS  （企业平台）                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  可用性要求：                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  99.9%：     每年约 8.7 小时停机                      │   │
│  │  99.99%：    每年约 52 分钟停机                       │   │
│  │  99.999%：   每年约 5 分钟停机                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2.2 模型服务架构模式

🟡 **中级**

```
模式 1：嵌入式服务
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              应用服务器                                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │   │
│  │  │ Web API  │  │ 业务逻辑 │  │ ML 模型          │ │   │
│  │  │          │  │          │  │ （嵌入式）        │ │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  优点：简单，低延迟                                          │
│  缺点：紧耦合，扩展性问题                                    │
└─────────────────────────────────────────────────────────────┘

模式 2：专用模型服务器
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐        ┌──────────────────────────────┐ │
│  │ Web API      │───────►│ 模型服务器                    │ │
│  │ （应用程序）  │        │ ┌──────────┐ ┌──────────┐   │ │
│  └──────────────┘        │ │ 模型     │ │ 预处理/   │   │ │
│                          │ │ 处理器   │ │ 后处理    │   │ │
│                          │ └──────────┘ └──────────┘   │ │
│                          └──────────────────────────────┘ │
│                                                             │
│  优点：关注点分离，独立扩展                                   │
│  缺点：网络延迟，运营开销                                    │
└─────────────────────────────────────────────────────────────┘

模式 3：模型服务平台
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐        ┌──────────────────────────────┐ │
│  │ Web API      │───────►│ 模型服务平台                  │ │
│  │              │        │ ┌──────────┐ ┌──────────┐   │ │
│  └──────────────┘        │ │ 路由器   │ │ 模型     │   │ │
│                          │ │          │ │ 注册中心 │   │ │
│                          │ └────┬─────┘ └──────────┘   │ │
│                          │      │                        │ │
│                          │      ▼                        │ │
│                          │ ┌──────────┐ ┌──────────┐   │ │
│                          │ │ 模型 A   │ │ 模型 B   │   │ │
│                          │ │ v1       │ │ v2       │   │ │
│                          │ └──────────┘ └──────────┘   │ │
│                          └──────────────────────────────┘ │
│                                                             │
│  优点：多模型，版本控制，A/B 测试，监控                       │
│  缺点：设置复杂，资源使用更高                                 │
└─────────────────────────────────────────────────────────────┘
```

### 8.2.3 Seldon Core 架构

🔴 **高级**

```
Seldon Core 架构：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Seldon Core                         │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              API 网关                         │   │   │
│  │  │  （REST / gRPC / GraphQL）                    │   │   │
│  │  └──────────────────────┬──────────────────────┘   │   │
│  │                         │                           │   │
│  │  ┌──────────────────────▼──────────────────────┐   │   │
│  │  │              路由器                           │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │   │
│  │  │  │ 负载     │  │ A/B 测试 │  │ 金丝雀   │  │   │   │
│  │  │  │ 均衡器   │  │ 路由器   │  │ 路由器   │  │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘  │   │   │
│  │  └──────────────────────┬──────────────────────┘   │   │
│  │                         │                           │   │
│  │  ┌──────────────────────▼──────────────────────┐   │   │
│  │  │              模型运行时                       │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │   │
│  │  │  │ Python   │  │ Java     │  │TensorFlow│  │   │   │
│  │  │  │ 包装器   │  │ 包装器   │  │ Serving  │  │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘  │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Kubernetes 资源                         │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ Deployment│  │ Service  │  │ HPA      │         │   │
│  │  │ （模型）  │  │ (gRPC)   │  │（自动扩缩）│         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2.4 Seldon Core 部署示例

🟡 **中级**

```yaml
# SeldonDeployment 用于 Python 模型
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
# Seldon 自定义模型类
class TextClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """加载模型产物"""
        import torch
        self.model = torch.load('/models/text-classifier/model.pt')
        self.tokenizer = load_tokenizer('/models/text-classifier/tokenizer')
        
    def predict(self, X, features_names=None):
        """进行预测"""
        import torch
        
        # 分词输入
        inputs = self.tokenizer(
            X, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # 运行推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
        
        return predictions.numpy()
    
    def feedback(self, X, Y, features_names=None):
        """处理反馈用于在线学习"""
        pass
```

---

## 8.3 A/B 测试与金丝雀发布

### 8.3.1 A/B 测试基础

🟢 **初级**

A/B 测试通过在两个模型版本之间分配流量来比较它们，以确定哪个表现更好。

```
A/B 测试设置：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐                                          │
│  │   入站流量    │                                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │   A/B 测试   │                                          │
│  │   路由器     │                                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│    ┌────┴────┐                                             │
│    │         │                                              │
│    ▼         ▼                                              │
│  ┌──────┐ ┌──────┐                                        │
│  │  A   │ │  B   │                                        │
│  │(50%) │ │(50%) │                                        │
│  │对照组 │ │变体组 │                                       │
│  └──┬───┘ └──┬───┘                                        │
│     │        │                                             │
│     ▼        ▼                                             │
│  ┌──────┐ ┌──────┐                                        │
│  │ 追踪 │ │ 追踪 │                                        │
│  │ 指标 │ │ 指标 │                                        │
│  └──────┘ └──────┘                                        │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │  统计分析    │                                          │
│  └──────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.3.2 A/B 测试中的统计显著性

🔴 **高级**

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
        """分析 A/B 测试的转化率"""
        
        # 计算转化率
        control_rate = control_conversions / control_total
        variant_rate = variant_conversions / variant_total
        
        # 执行卡方检验
        contingency_table = np.array([
            [control_conversions, control_total - control_conversions],
            [variant_conversions, variant_total - variant_conversions]
        ])
        
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        
        # 计算相对改进
        relative_improvement = (variant_rate - control_rate) / control_rate
        
        # 确定胜者
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
        """计算所需样本量"""
        
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

# 用法
analyzer = ABTestAnalyzer(confidence_level=0.95)

# 测试后分析结果
results = analyzer.analyze_conversion_rate(
    control_conversions=450,
    control_total=5000,
    variant_conversions=480,
    variant_total=5000
)

print(f"对照组转化率：{results['control_rate']:.2%}")
print(f"变体组转化率：{results['variant_rate']:.2%}")
print(f"相对改进：{results['relative_improvement']:.2%}")
print(f"P 值：{results['p_value']:.4f}")
print(f"显著性：{results['is_significant']}")
print(f"胜者：{results['winner']}")

# 计算所需样本量
sample_size = analyzer.calculate_sample_size(
    baseline_rate=0.09,  # 9% 基准转化率
    minimum_detectable_effect=0.10  # 10% 相对改进
)
print(f"所需样本量：{sample_size}")
```

### 8.3.3 金丝雀发布实现

🟡 **中级**

```yaml
# Seldon Core 金丝雀部署
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
    traffic: 90  # 90% 流量到稳定版本
  
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
    traffic: 10  # 10% 流量到金丝雀版本
```

```bash
# 监控金丝雀指标
kubectl logs -f -l seldon-deployment=text-classifier -n default

# 逐步增加金丝雀流量
kubectl patch seldondeployment text-classifier --type='json' -p='[
  {"op": "replace", "path": "/spec/predictors/1/traffic", "value": 30},
  {"op": "replace", "path": "/spec/predictors/0/traffic", "value": 70}
]'
```

---

## 8.4 模型版本管理

### 8.4.1 版本管理策略

🟡 **中级**

```
模型版本控制架构：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              模型注册中心                              │   │
│  │                                                     │   │
│  │  text-classifier                                    │   │
│  │  ├── v1.0.0 (2024-01-15)                          │   │
│  │  │   ├── 状态：生产环境                              │   │
│  │  │   ├── 准确率：0.945                              │   │
│  │  │   ├── 产物：s3://models/v1.0.0/model.pt         │   │
│  │  │   └── 元数据：{...}                              │   │
│  │  │                                                  │   │
│  │  ├── v1.1.0 (2024-02-01)                          │   │
│  │  │   ├── 状态：预发布                               │   │
│  │  │   ├── 准确率：0.952                              │   │
│  │  │   ├── 产物：s3://models/v1.1.0/model.pt         │   │
│  │  │   └── 元数据：{...}                              │   │
│  │  │                                                  │   │
│  │  └── v1.2.0 (2024-02-15)                          │   │
│  │      ├── 状态：开发中                               │   │
│  │      ├── 准确率：0.948                              │   │
│  │      ├── 产物：s3://models/v1.2.0/model.pt         │   │
│  │      └── 元数据：{...}                              │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.4.2 ML 模型的语义版本控制

🔴 **高级**

```
ML 模型版本控制方案：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  格式：MAJOR.MINOR.PATCH                                    │
│                                                             │
│  MAJOR：破坏性更改                                           │
│  ├── 新架构                                                  │
│  ├── 不同的输入格式                                          │
│  └── 不兼容的 API 更改                                       │
│                                                             │
│  MINOR：新功能（向后兼容）                                    │
│  ├── 使用更多数据重训练                                       │
│  ├── 新的预处理步骤                                          │
│  └── 性能改进                                                │
│                                                             │
│  PATCH：错误修复                                              │
│  ├── 修复数据加载错误                                        │
│  ├── 更新依赖项                                              │
│  └── 文档更新                                                │
│                                                             │
│  示例：                                                      │
│  1.0.0 → 1.0.1：修复数据加载 bug                            │
│  1.0.0 → 1.1.0：使用额外数据重训练                           │
│  1.0.0 → 2.0.0：从 BERT 切换到 RoBERTa                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.4.3 模型血缘追踪

🟡 **中级**

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
        """记录完整的模型血缘"""
        
        # 创建运行
        with mlflow.start_run(run_name=f"{model_name}-v{model_version}"):
            # 记录模型信息
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("dataset_version", dataset_version)
            mlflow.set_tag("training_commit", training_code_commit)
            
            # 记录超参数
            mlflow.log_params(hyperparameters)
            
            # 记录指标
            mlflow.log_metrics(metrics)
            
            # 记录模型
            mlflow.pytorch.log_model(model, "model")
            
            # 记录数据集信息
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_param("dataset_size", len(dataset))
            mlflow.log_param("train_size", len(train_dataset))
            mlflow.log_param("val_size", len(val_dataset))
    
    def get_model_lineage(self, model_name: str, version: str):
        """检索完整的模型血缘"""
        
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

## 8.5 推理优化

### 8.5.1 优化技术概览

🟡 **中级**

```
推理优化技术：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  模型级优化：                                                │
│  ├── 量化（INT8/INT4）                                      │
│  ├── 剪枝（移除冗余权重）                                    │
│  ├── 知识蒸馏（大模型 → 小模型）                             │
│  └── 模型架构优化                                           │
│                                                             │
│  运行时优化：                                                │
│  ├── TensorRT（NVIDIA GPU 优化）                            │
│  ├── ONNX Runtime（跨平台）                                 │
│  ├── OpenVINO（Intel 优化）                                 │
│  └── Core ML（Apple 设备）                                  │
│                                                             │
│  系统级优化：                                                │
│  ├── 批处理（动态/静态）                                     │
│  ├── 缓存（模型/特征）                                      │
│  ├── 负载均衡                                               │
│  └── 自动扩缩                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.5.2 模型量化

🔴 **高级**

```python
import torch
from torch.quantization import quantize_dynamic

class ModelQuantizer:
    @staticmethod
    def quantize_dynamic(model, dtype=torch.qint8):
        """动态量化——在加载时量化权重"""
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=dtype
        )
        return quantized_model
    
    @staticmethod
    def quantize_static(model, calibration_data, dtype=torch.qint8):
        """静态量化——量化权重和激活"""
        model.eval()
        
        # 准备模型进行静态量化
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model)
        
        # 使用样本数据校准
        with torch.no_grad():
            for batch in calibration_data:
                model_prepared(batch)
        
        # 转换为量化模型
        model_quantized = torch.quantization.convert(model_prepared)
        
        return model_quantized
    
    @staticmethod
    def measure_performance(original_model, quantized_model, test_data):
        """比较性能指标"""
        import time
        
        # 测量原始模型
        start = time.time()
        with torch.no_grad():
            for batch in test_data:
                original_model(batch)
        original_time = time.time() - start
        
        # 测量量化模型
        start = time.time()
        with torch.no_grad():
            for batch in test_data:
                quantized_model(batch)
        quantized_time = time.time() - start
        
        # 测量模型大小
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

# 用法
quantizer = ModelQuantizer()
quantized_model = quantizer.quantize_dynamic(model)
performance = quantizer.measure_performance(model, quantized_model, test_data)

print(f"加速比：{performance['speedup']:.2f}x")
print(f"压缩比：{performance['compression_ratio']:.2f}x")
```

### 8.5.3 动态批处理

🟡 **中级**

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
        """单个预测请求"""
        future = asyncio.Future()
        await self.batch_queue.put((input_data, future))
        return await future
    
    async def batch_processor(self):
        """从队列处理批次"""
        while True:
            batch = []
            futures = []
            
            # 收集批次项
            try:
                # 等待第一个项
                item = await asyncio.wait_for(
                    self.batch_queue.get(), 
                    timeout=self.max_wait_ms / 1000
                )
                batch.append(item[0])
                futures.append(item[1])
                
                # 收集剩余项（最多到最大值）
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
                
                # 处理批次
                batch_tensor = np.array(batch)
                predictions = self.model.predict(batch_tensor)
                
                # 将结果返回给各个 future
                for i, future in enumerate(futures):
                    future.set_result(predictions[i])
                    
            except asyncio.TimeoutError:
                continue

# 用法
batcher = DynamicBatcher(model, max_batch_size=32, max_wait_ms=10)

async def main():
    # 启动批处理器
    processor_task = asyncio.create_task(batcher.batch_processor())
    
    # 进行并发预测
    results = await asyncio.gather(
        batcher.predict(input1),
        batcher.predict(input2),
        batcher.predict(input3),
    )
    
    return results
```

---

## 8.6 边缘部署策略

### 8.6.1 边缘部署挑战

🟢 **初级**

```
边缘部署挑战：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  资源约束：                                                  │
│  ├── 计算能力有限                                           │
│  ├── 内存有限                                               │
│  ├── 存储有限                                               │
│  └── 功耗有限（电池设备）                                    │
│                                                             │
│  网络约束：                                                  │
│  ├── 间歇性连接                                             │
│  ├── 高延迟                                                 │
│  └── 有限带宽                                               │
│                                                             │
│  运营约束：                                                  │
│  ├── 远程更新                                               │
│  ├── 监控和调试                                             │
│  └── 安全要求                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.6.2 边缘部署架构

🟡 **中级**

```
边缘部署架构：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              云端                                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │   │
│  │  │ 训练     │  │ 模型     │  │ 边缘管理         │ │   │
│  │  │ 平台     │  │ 注册中心 │  │ 平台             │ │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         │（模型同步）                        │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              边缘网络                                 │   │
│  │                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │   │
│  │  │ 边缘     │  │ 边缘     │  │ 边缘     │         │   │
│  │  │ 设备 1   │  │ 设备 2   │  │ 设备 3   │         │   │
│  │  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │         │   │
│  │  │ │模型  │ │  │ │模型  │ │  │ │模型  │ │         │   │
│  │  │ │(精简)│ │  │ │(精简)│ │  │ │(完整)│ │         │   │
│  │  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │         │   │
│  │  └──────────┘  └──────────┘  └──────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.6.3 面向边缘的模型优化

🔴 **高级**

```python
import tensorflow as tf

class EdgeModelOptimizer:
    def __init__(self):
        self.converter = None
    
    def convert_to_tflite(self, model_path, quantization='dynamic'):
        """将模型转换为 TensorFlow Lite"""
        
        # 加载模型
        model = tf.keras.models.load_model(model_path)
        
        # 转换为 TFLite
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
        
        # 保存模型
        output_path = model_path.replace('.h5', '.tflite')
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        return output_path
    
    def optimize_for_mobile(self, model_path):
        """为移动部署优化模型"""
        
        # 转换为 TFLite
        tflite_path = self.convert_to_tflite(model_path, quantization='int8')
        
        # 获取模型大小
        import os
        original_size = os.path.getsize(model_path)
        optimized_size = os.path.getsize(tflite_path)
        
        return {
            "tflite_path": tflite_path,
            "original_size_mb": original_size / (1024 * 1024),
            "optimized_size_mb": optimized_size / (1024 * 1024),
            "compression_ratio": original_size / optimized_size,
        }

# 用法
optimizer = EdgeModelOptimizer()
result = optimizer.optimize_for_mobile('model.h5')

print(f"TFLite 模型保存至：{result['tflite_path']}")
print(f"原始大小：{result['original_size_mb']:.2f} MB")
print(f"优化后大小：{result['optimized_size_mb']:.2f} MB")
print(f"压缩比：{result['compression_ratio']:.2f}x")
```

---

## 💡 案例研究：Seldon Core 模型服务

### 架构概览

```
Seldon Core 生产部署：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              负载均衡器（nginx）                       │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │              Seldon Core API 网关                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  请求路由器                                   │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │   │
│  │  │  │ 速率     │  │ 认证     │  │ 路由     │  │   │   │
│  │  │  │ 限制器   │  │ 检查器   │  │ 逻辑     │  │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘  │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │              模型部署                                 │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  生产模型 (v1.0)                             │   │   │
│  │  │  副本数：3                                   │   │   │
│  │  │  流量：100%                                  │   │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │   │
│  │  │  │ Pod 1    │  │ Pod 2    │  │ Pod 3    │  │   │   │
│  │  │  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │  │   │   │
│  │  │  │ │模型  │ │  │ │模型  │ │  │ │模型  │ │  │   │   │
│  │  │  │ │v1.0  │ │  │ │v1.0  │ │  │ │v1.0  │ │  │   │   │
│  │  │  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │  │   │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘  │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  金丝雀模型 (v2.0)                           │   │   │
│  │  │  副本数：1                                   │   │   │
│  │  │  流量：0%（金丝雀测试）                       │   │   │
│  │  │  ┌──────────┐                               │   │   │
│  │  │  │ Pod 1    │                               │   │   │
│  │  │  │ ┌──────┐ │                               │   │   │
│  │  │  │ │模型  │ │                               │   │   │
│  │  │  │ │v2.0  │ │                               │   │   │
│  │  │  │ └──────┘ │                               │   │   │
│  │  │  └──────────┘                               │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              监控技术栈                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │   │
│  │  │Prometheus│  │ Grafana  │  │ Seldon Analytics │ │   │
│  │  │          │  │ 仪表板   │  │                  │ │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 部署说明

```bash
# 1. 安装 Seldon Core
kubectl create namespace seldon-system
helm install seldon-core seldon-core-operator \
  --repo https://storage.googleapis.com/seldon-charts \
  --namespace seldon-system \
  --set usageMetrics.enabled=true \
  --set istio.enabled=true

# 2. 创建模型部署
kubectl apply -f seldondeployment.yaml

# 3. 验证部署
kubectl get seldondeployment text-classifier -o yaml

# 4. 端口转发用于测试
kubectl port-forward svc/text-classifier-default 8000:8000

# 5. 测试预测
curl -X POST http://localhost:8000/api/v1.0/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "ndarray": ["这是一个很好的产品！"]
    }
  }'

# 6. 检查指标
kubectl port-forward svc/prometheus 9090:9090
# 访问 Grafana 仪表板
kubectl port-forward svc/grafana 3000:3000
```

### 监控仪表板

```
Grafana 仪表板：Seldon Core 模型服务
┌─────────────────────────────────────────────────────────────┐
│  模型：text-classifier                                      │
│  版本：v1.0                                                 │
│  状态：健康 ✓                                               │
│                                                             │
│  请求指标：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  总请求数：    15,234                                │   │
│  │  每秒请求数：  42.3                                  │   │
│  │  成功率：      99.8%                                 │   │
│  │  错误率：      0.2%                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  延迟分布：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  P50：      23ms                                    │   │
│  │  P90：      45ms                                    │   │
│  │  P95：      67ms                                    │   │
│  │  P99：      123ms                                   │   │
│  │  最大值：   234ms                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  资源使用：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CPU：    ████████████░░░░░░░░ 60%                  │   │
│  │  内存：   ████████░░░░░░░░░░░░ 40%                  │   │
│  │  GPU：    ██████████████░░░░░░ 70%                  │   │
│  │  网络：   ██████░░░░░░░░░░░░░░ 30%                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  模型性能：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  准确率：     0.945                                  │   │
│  │  延迟：       23ms（平均）                            │   │
│  │  吞吐量：     42.3 req/s                            │   │
│  │  错误率：     0.2%                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 总结

**关键要点：**

1. **部署策略**取决于风险承受能力、停机要求和流量分配需求
2. **模型服务架构**应支持版本控制、A/B 测试和自动扩缩
3. **A/B 测试**需要统计严谨性才能得出有效结论
4. **模型版本管理**对于回滚和可复现性至关重要
5. **推理优化**可以显著降低延迟和成本
6. **边缘部署**需要仔细的模型优化和管理

**下一章预览：**
在第 9 章中，我们将探讨**模型监控与可观测性**，涵盖漂移检测、性能监控以及构建全面的可观测性平台。

---

*第8章结束*
