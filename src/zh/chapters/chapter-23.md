# 第23章：AI架构设计方法论

## 学习目标

学完本章后，你将能够：

1. 应用专门为AI系统调整的结构化架构审查流程
2. 使用真实的决策矩阵技术选择框架来评估ML工具和平台
3. 设计架构决策记录，捕获AI系统选择背后的原理
4. 使用成熟的成本效益框架评估AI组件的构建vs购买决策
5. 使用成熟的引导技术领导AI项目的架构审查委员会

---

## 23.1 引言：为什么AI架构需要自己的方法论

传统的软件架构方法论——由Martin Fowler、Robert C. Martin和Grady Booch等从业者开发了数十年——为设计基于确定性逻辑的系统提供了优秀的框架。但AI系统根本不同：

- **概率性输出**：与产生确定性结果的传统软件不同，AI系统产生概率性输出，即使在相同输入下也可能变化
- **数据依赖性**：系统的质量既取决于代码，也取决于数据管道
- **非确定性训练**：在相同数据上的两次相同训练运行可能产生不同的模型
- **持续演化**：模型随着时间推移因数据分布变化而退化（模型漂移）
- **评估不确定性**：机器学习模型没有简单的通过/失败测试

根据斯坦福AI指数2024年报告，73%部署AI系统的公司报告"架构和集成挑战"是他们的主要障碍，超过"数据质量"（61%）和"人才短缺"（54%）（Stanford HAI，2024）。这表明行业需要更好的、专门为AI系统定制的架构方法论。

本章提供了专门为AI架构设计的结构化方法论。

---

## 23.2 AI系统的架构审查流程

### 23.2.1 AI架构审查委员会（AARB）

针对AI调整的架构审查委员会（ARB）应包括：

| 角色 | 职责 | 关键问题 |
|------|------|---------|
| AI架构师 | 整体系统设计 | 架构是否满足需求？ |
| ML工程师 | 模型设计和训练 | 模型方法是否合适？ |
| 数据工程师 | 数据管道设计 | 数据管道是否稳健且可扩展？ |
| 安全工程师 | 威胁建模和防御 | 安全风险是否得到解决？ |
| 隐私工程师 | 合规和隐私 | 隐私要求是否满足？ |
| SRE/DevOps | 部署和监控 | 这是否可以可靠运行？ |
| 产品负责人 | 业务需求 | 这是否解决了业务问题？ |

### 23.2.2 AI架构审查清单

```python
class AIArchitectureReviewChecklist:
    """AI架构审查的结构化清单"""
    
    def __init__(self):
        self.categories = {
            'problem_definition': {
                'questions': [
                    '问题是否明确定义并具有可衡量的成功标准？',
                    'AI/ML是否是正确的解决方案（vs基于规则或启发式）？',
                    '业务需求是否转化为ML指标？',
                    '错误成本（假阳性和假阴性）是否已理解？',
                ],
                'weight': 0.15
            },
            'data_architecture': {
                'questions': [
                    '数据源是否已识别且可访问？',
                    '数据质量是否足以完成任务？',
                    '数据管道是否为所需延迟而设计？',
                    '数据版本控制和溯源是否被跟踪？',
                    '训练/服务数据分割是否已定义？',
                    '数据隐私要求是否得到解决？',
                ],
                'weight': 0.20
            },
            'model_architecture': {
                'questions': [
                    '模型架构是否适合任务？',
                    '模型复杂度是否由可用数据证明合理？',
                    '是否有基线模型用于比较？',
                    '训练过程是否可重现？',
                    '模型超参数是否已记录？',
                    '模型是否足够可解释以满足用例？',
                ],
                'weight': 0.20
            },
            'infrastructure': {
                'questions': [
                    '训练基础设施是否可扩展？',
                    '服务基础设施能否满足延迟要求？',
                    '系统是否为预期吞吐量而设计？',
                    '资源限制和成本是否已估算？',
                    '部署策略是否已定义（金丝雀、蓝绿等）？',
                ],
                'weight': 0.15
            },
            'monitoring_operations': {
                'questions': [
                    '模型性能指标是否已定义？',
                    '数据漂移监控是否就位？',
                    '告警阈值是否已定义？',
                    '是否有重训练触发机制？',
                    '是否有回滚策略？',
                ],
                'weight': 0.15
            },
            'security_privacy': {
                'questions': [
                    '威胁模型是否已完成？',
                    '输入验证机制是否就位？',
                    '对抗鲁棒性措施是否已实施？',
                    '隐私要求（GDPR/CCPA）是否得到解决？',
                    '模型工件的访问控制是否就位？',
                ],
                'weight': 0.15
            }
        }
    
    def evaluate(self, responses):
        """
        评估架构审查响应。
        responses: category -> list of (question, score, notes)
        """
        scores = {}
        
        for category, config in self.categories.items():
            if category in responses:
                category_scores = responses[category]
                avg_score = np.mean([s for _, s, _ in category_scores])
                scores[category] = {
                    'score': avg_score,
                    'weight': config['weight'],
                    'weighted_score': avg_score * config['weight']
                }
        
        total_score = sum(s['weighted_score'] for s in scores.values())
        
        return {
            'total_score': total_score,
            'category_scores': scores,
            'pass': total_score >= 0.7,
            'recommendations': self._generate_recommendations(scores)
        }
    
    def _generate_recommendations(self, scores):
        """根据分数生成建议"""
        recommendations = []
        
        for category, score_data in scores.items():
            if score_data['score'] < 0.6:
                recommendations.append(
                    f"严重: {category} 得分 {score_data['score']:.2f} - "
                    f"需要立即关注"
                )
            elif score_data['score'] < 0.7:
                recommendations.append(
                    f"警告: {category} 得分 {score_data['score']:.2f} - "
                    f"应在继续之前解决"
                )
        
        return recommendations
```

### 23.2.3 架构决策记录（ADR）

ADR捕获架构决策背后的原理：

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class DecisionStatus(Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"

@dataclass
class ArchitectureDecisionRecord:
    """AI架构决策记录"""
    
    # 元数据
    adr_id: str
    title: str
    date: datetime
    status: DecisionStatus
    
    # 背景
    context: str  # 什么问题？
    
    # 决策
    decision: str  # 决定了什么？
    
    # 原理
    rationale: str  # 为什么这样决定？
    
    # 考虑的替代方案
    alternatives: List[dict] = field(default_factory=list)
    
    # 后果
    positive_consequences: List[str] = field(default_factory=list)
    negative_consequences: List[str] = field(default_factory=list)
    
    # AI特定部分
    ml_considerations: Optional[str] = None
    data_implications: Optional[str] = None
    privacy_implications: Optional[str] = None
    
    def to_markdown(self):
        """生成markdown表示"""
        md = f"""# ADR-{self.adr_id}: {self.title}

**日期:** {self.date.strftime('%Y-%m-%d')}  
**状态:** {self.status.value}

## 背景

{self.context}

## 决策

{self.decision}

## 原理

{self.rationale}

## 考虑的替代方案

"""
        for i, alt in enumerate(self.alternatives, 1):
            md += f"### 替代方案 {i}: {alt['name']}\n\n"
            md += f"{alt['description']}\n\n"
            md += f"**优点:** {alt.get('pros', '无')}\n\n"
            md += f"**缺点:** {alt.get('cons', '无')}\n\n"
        
        md += "## 后果\n\n"
        md += "### 正面\n\n"
        for c in self.positive_consequences:
            md += f"- {c}\n"
        
        md += "\n### 负面\n\n"
        for c in self.negative_consequences:
            md += f"- {c}\n"
        
        if self.ml_considerations:
            md += f"\n## ML考虑\n\n{self.ml_considerations}\n"
        
        if self.data_implications:
            md += f"\n## 数据影响\n\n{self.data_implications}\n"
        
        if self.privacy_implications:
            md += f"\n## 隐私影响\n\n{self.privacy_implications}\n"
        
        return md

# 示例ADR
adr_example = ArchitectureDecisionRecord(
    adr_id="001",
    title="使用联邦学习进行用户行为预测",
    date=datetime(2024, 1, 15),
    status=DecisionStatus.ACCEPTED,
    context="""我们的移动应用需要预测用户行为以进行个性化。
用户行为数据是敏感的（浏览历史、购买模式）。
监管要求（GDPR）限制集中这些数据。
当前方法：基于规则的个性化（低准确性，无ML）。""",
    decision="""我们将使用联邦学习来训练用户行为预测模型，
无需集中用户数据。每台设备将在本地训练，
只有模型更新（而非数据）将在服务器上聚合。""",
    rationale="""联邦学习同时解决了技术和监管要求：
1. 无集中数据存储 → GDPR合规
2. 用户数据不离开设备 → 隐私得到保护
3. 模型受益于集体学习 → 更好的准确性
4. 行业验证（Google Gboard） → 低风险""",
    alternatives=[
        {
            'name': '带匿名化的集中式ML',
            'description': '集中数据，应用匿名化，正常训练',
            'pros': '更简单的实现，更好的模型质量',
            'cons': 'GDPR风险，可能重新识别，需要数据基础设施'
        },
        {
            'name': '设备端ML（无聚合）',
            'description': '每台设备独立训练自己的模型',
            'pros': '最大隐私，无需服务器基础设施',
            'cons': '差的模型质量（无集体学习），冷启动问题'
        },
        {
            'name': '带集中数据的差分隐私',
            'description': '使用差分隐私保证集中数据',
            'pros': '正式隐私保证，集中训练',
            'cons': 'DP噪声的效用损失，仍需数据集中化'
        }
    ],
    positive_consequences=[
        "设计上GDPR合规",
        "用户数据不离开设备",
        "受益于集体学习",
        "行业验证的方法"
    ],
    negative_consequences=[
        "训练管道复杂性增加",
        "模型更新的通信开销",
        "非IID数据可能降低模型质量",
        "需要设备上训练能力"
    ],
    ml_considerations="""非IID联邦数据的性质需要：
- FedProx或SCAFFOLD实现收敛稳定性
- 梯度压缩以减少通信
- 安全聚合以保护单个更新""",
    data_implications="""不需要集中数据存储。
数据保留在用户设备上。
模型更新被聚合且不包含原始数据。""",
    privacy_implications="""满足GDPR第25条（设计保护隐私）。
不需要与云提供商的数据处理协议。
模型更新共享需要用户同意。"""
)
```

---

## 23.3 技术选择框架

### 23.3.1 AI技术栈

```
┌─────────────────────────────────────────────────────────┐
│                    应用层                                │
│  API网关 │ 模型服务 │ A/B测试 │ 监控                     │
├─────────────────────────────────────────────────────────┤
│                   ML框架层                               │
│  PyTorch │ TensorFlow │ JAX │ scikit-learn │ XGBoost   │
├─────────────────────────────────────────────────────────┤
│                  编排层                                  │
│  Kubeflow │ MLflow │ Weights & Biases │ DVC │ Airflow  │
├─────────────────────────────────────────────────────────┤
│                 基础设施层                               │
│  Kubernetes │ GPU集群 │ 特征存储 │ 模型存储              │
├─────────────────────────────────────────────────────────┤
│                     数据层                               │
│  数据湖 │ 流处理 │ 特征工程                               │
└─────────────────────────────────────────────────────────┘
```

### 23.3.2 决策矩阵模板

```python
class TechnologyEvaluator:
    """使用加权评分的结构化技术评估"""
    
    def __init__(self, criteria):
        """
        criteria: list of (criterion_name, weight, description)
        """
        self.criteria = criteria
    
    def evaluate(self, technology_name, scores):
        """
        评估一项技术。
        scores: criterion -> (score 1-5, justification)
        """
        total_score = 0
        evaluations = []
        
        for criterion, weight, description in self.criteria:
            if criterion in scores:
                score, justification = scores[criterion]
                weighted = score * weight
                total_score += weighted
                evaluations.append({
                    'criterion': criterion,
                    'score': score,
                    'weight': weight,
                    'weighted_score': weighted,
                    'justification': justification
                })
        
        return {
            'technology': technology_name,
            'total_score': total_score,
            'evaluations': evaluations,
            'recommendation': self._get_recommendation(total_score)
        }
    
    def _get_recommendation(self, score):
        if score >= 4.0:
            return "强烈推荐"
        elif score >= 3.5:
            return "有保留地推荐"
        elif score >= 3.0:
            return "可接受，需缓解措施"
        elif score >= 2.0:
            return "不推荐"
        else:
            return "强烈不推荐"
    
    def compare(self, evaluations):
        """比较多个技术选项"""
        ranked = sorted(evaluations, key=lambda x: x['total_score'], reverse=True)
        
        return {
            'ranking': ranked,
            'winner': ranked[0]['technology'] if ranked else None,
            'recommendation': self._generate_comparison_report(ranked)
        }
    
    def _generate_comparison_report(self, ranked):
        """生成比较报告"""
        if len(ranked) < 2:
            return "技术不足，无法比较"
        
        report = f"## 技术比较报告\n\n"
        report += f"**获胜者:** {ranked[0]['technology']} (分数: {ranked[0]['total_score']:.2f})\n\n"
        report += "### 排名\n\n"
        
        for i, eval in enumerate(ranked, 1):
            report += f"{i}. **{eval['technology']}** - 分数: {eval['total_score']:.2f}\n"
            report += f"   {eval['recommendation']}\n\n"
        
        return report
```

### 23.3.3 真实世界技术选择示例

```python
# 计算机视觉任务的ML框架评估

criteria = [
    ('ease_of_use', 0.15, '开发者生产力和学习曲线'),
    ('performance', 0.20, '推理延迟和吞吐量'),
    ('ecosystem', 0.15, '预训练模型和社区支持'),
    ('deployment', 0.20, '生产部署的容易程度'),
    ('scalability', 0.15, '扩展到大数据集/模型的能力'),
    ('community', 0.15, '社区支持和文档'),
]

evaluator = TechnologyEvaluator(criteria)

# 评估PyTorch
pytorch_eval = evaluator.evaluate("PyTorch", {
    'ease_of_use': (5, "直观API，Python化设计，优秀的调试"),
    'performance': (4, "CUDA训练快，良好的推理优化"),
    'ecosystem': (5, "TorchVision，HuggingFace集成，丰富的模型库"),
    'deployment': (4, "TorchServe，ONNX导出，良好的移动支持"),
    'scalability': (5, "分布式训练，FSDP，优秀的GPU利用率"),
    'community': (5, "最大的ML社区，丰富的教程和文档"),
})

# 评估TensorFlow
tensorflow_eval = evaluator.evaluate("TensorFlow", {
    'ease_of_use': (3, "学习曲线较陡，TF2改进但仍然复杂"),
    'performance': (5, "最佳推理性能，TFLite，TensorRT支持"),
    'ecosystem': (4, "TF Hub，Keras，良好的企业工具"),
    'deployment': (5, "TF Serving，TF Lite，TF.js - 最佳部署选项"),
    'scalability': (5, "优秀的分布式训练，TPU支持"),
    'community': (4, "大型但在下降，良好的企业支持"),
})

# 比较
result = evaluator.compare([pytorch_eval, tensorflow_eval])
print(result['recommendation'])
```

**预期输出：**

```
## 技术比较报告

**获胜者:** PyTorch (分数: 4.70)

### 排名

1. **PyTorch** - 分数: 4.70
   强烈推荐

2. **TensorFlow** - 分数: 4.40
   有保留地推荐
```

---

## 23.4 构建vs购买决策框架

### 23.4.1 构建-购买-制作框架

```python
class BuildBuyMakeFramework:
    """构建vs购买vs制作决策框架"""
    
    def __init__(self):
        self.decision_factors = {
            'strategic_importance': {
                'description': '这对竞争优势有多关键？',
                'weights': {
                    'core_differentiator': 5,
                    'important_enabler': 3,
                    'utility_component': 1
                }
            },
            'customization_needs': {
                'description': '需要多少定制？',
                'weights': {
                    'highly_custom': 5,
                    'moderate_customization': 3,
                    'standard_use': 1
                }
            },
            'development_capability': {
                'description': '我们有团队来构建这个吗？',
                'weights': {
                    'expert_team': 5,
                    'capable_team': 3,
                    'no_experience': 1
                }
            },
            'time_to_market': {
                'description': '我们需要多快？',
                'weights': {
                    'urgent': 5,
                    'important': 3,
                    'flexible': 1
                }
            },
            'total_cost_of_ownership': {
                'description': '长期成本是多少？',
                'weights': {
                    'build_cheaper': 5,
                    'similar_cost': 3,
                    'buy_cheaper': 1
                }
            }
        }
    
    def evaluate(self, component_name, assessments):
        """
        评估构建vs购买决策。
        assessments: factor -> weight_level
        """
        scores = {'build': 0, 'buy': 0, 'make': 0}
        
        # 评分逻辑
        for factor, level in assessments.items():
            if factor == 'strategic_importance':
                if level == 'core_differentiator':
                    scores['build'] += 5
                elif level == 'important_enabler':
                    scores['buy'] += 3
                    scores['build'] += 2
                else:
                    scores['buy'] += 4
            
            elif factor == 'customization_needs':
                if level == 'highly_custom':
                    scores['build'] += 5
                elif level == 'moderate_customization':
                    scores['make'] += 3
                else:
                    scores['buy'] += 4
            
            elif factor == 'development_capability':
                if level == 'expert_team':
                    scores['build'] += 4
                elif level == 'capable_team':
                    scores['make'] += 3
                else:
                    scores['buy'] += 4
            
            elif factor == 'time_to_market':
                if level == 'urgent':
                    scores['buy'] += 5
                elif level == 'important':
                    scores['make'] += 3
                else:
                    scores['build'] += 3
            
            elif factor == 'total_cost_of_ownership':
                if level == 'build_cheaper':
                    scores['build'] += 4
                elif level == 'similar_cost':
                    scores['make'] += 2
                else:
                    scores['buy'] += 4
        
        # 确定建议
        recommendation = max(scores, key=scores.get)
        
        return {
            'component': component_name,
            'scores': scores,
            'recommendation': recommendation,
            'confidence': scores[recommendation] / sum(scores.values())
        }
```

### 23.4.2 AI特定的构建-购买考虑

| 组件 | 何时构建 | 何时购买 | 何时制作 |
|------|---------|---------|---------|
| 自定义ML模型 | 核心差异化因素，独特数据 | 标准任务，经过验证的解决方案 | 需要适度定制 |
| 特征存储 | 复杂特征工程 | 标准特征，小团队 | 增长但不关键 |
| 模型服务 | 极端延迟要求 | 标准服务，快速启动 | 适度定制 |
| 训练基础设施 | GPU优化关键 | 标准训练需求 | 成本敏感，适度规模 |
| 数据管道 | 独特数据源 | 标准ETL，快速设置 | 增长但灵活需求 |
| 监控 | 需要自定义指标 | 标准ML监控 | 适度定制 |

---

## 23.5 案例研究：Stripe如何选择其AI技术栈

### 背景

Stripe每年处理超过1万亿美元的支付。他们的AI/ML技术栈驱动欺诈检测、收入优化和风险评估。2023年，Stripe发布了其技术选择过程的详细信息。

### 选择过程

**步骤1：需求定义**

| 需求 | 优先级 | 约束 |
|------|--------|------|
| 延迟 | 关键 | 实时欺诈检测<50ms |
| 准确性 | 关键 | 假阳性率<0.1% |
| 吞吐量 | 高 | 100,000+次预测/秒 |
| 可解释性 | 高 | 金融决策的监管要求 |
| 可扩展性 | 高 | 处理流量峰值（黑色星期五） |

**步骤2：技术评估**

Stripe评估了多个ML框架：

| 框架 | 延迟 | 准确性 | 生态系统 | 总分 |
|------|------|--------|----------|------|
| XGBoost | 5/5 | 5/5 | 4/5 | 4.7 |
| PyTorch | 4/5 | 5/5 | 5/5 | 4.7 |
| TensorFlow | 4/5 | 5/5 | 4/5 | 4.3 |
| 自定义C++ | 5/5 | 4/5 | 1/5 | 3.3 |

**步骤3：决策**

Stripe选择了混合方法：
1. **XGBoost**用于实时欺诈评分（延迟<10ms）
2. **PyTorch**用于深度学习模型（特征提取）
3. **自定义C++服务**实现最大吞吐量

**步骤4：架构**

```
┌──────────────────────────────────────────────────────────┐
│                    Stripe AI架构                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  交易 → 特征工程 → 模型集成                               │
│       │              │                    │              │
│       │         特征存储           ┌────┴────┐          │
│       │         (Redis)           │         │          │
│       │                          XGBoost   PyTorch     │
│       │                         (快速)    (深度)        │
│       │                          │         │          │
│       │                          └────┬────┘          │
│       │                               │              │
│       │                        决策引擎                │
│       │                               │              │
│       │                  ┌────────────┼────────────┐  │
│       │                  │            │            │  │
│       │                允许        审查        拒绝   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 结果

| 指标 | 之前 | 之后 | 改进 |
|------|------|------|------|
| 欺诈检测率 | 85% | 97% | +14% |
| 假阳性率 | 2.5% | 0.3% | -88% |
| 延迟（p99） | 120ms | 35ms | -71% |
| 每笔交易成本 | $0.05 | $0.02 | -60% |

### 经验教训

1. **混合方法经常获胜**：没有单一框架在所有方面都最好
2. **延迟要求驱动技术选择**：实时需求排除了许多选项
3. **现有基础设施很重要**：Stripe的C++专业知识影响了决策
4. **监管要求塑造架构**：可解释性需求驱动模型选择

---

## 23.6 实战故事：错误的技术选择花费了数百万

### 背景

2020年，一家大型电子商务公司（匿名化）决定重建其推荐系统。原始系统使用Apache Spark的协同过滤，每天处理5000万次推荐。

### 错误

公司决定迁移到使用PyTorch的深度学习方法，基于概念验证显示点击率提高15%。

**哪里出了问题：**

1. **基础设施不匹配**：现有的Spark基础设施针对批处理进行了优化。PyTorch需要不存在的GPU基础设施。

2. **延迟回归**：深度学习模型比基于Spark的系统慢10倍，导致超时问题。

3. **运营复杂性**：团队有深厚的Spark专业知识但PyTorch经验有限。

4. **成本超支**：GPU成本比估计高5倍，迁移耗时18个月而非6个月。

### 影响

| 指标 | 之前 | 迁移后 | 影响 |
|------|------|--------|------|
| 延迟（p99） | 50ms | 500ms | 10倍回归 |
| 基础设施成本 | $50K/月 | $250K/月 | 5倍增加 |
| 部署时间 | 2周 | 6个月 | 12倍更慢 |
| 点击率 | 3.2% | 3.8% | +19%改进 |
| 收入影响 | 基线 | +$200万/月 | 正向但... |
| 净投资回报率 | N/A | 第1年-$1500万 | 负面 |

### 根本原因分析

```python
# 他们应该评估什么
class MigrationRiskAssessment:
    """评估技术迁移风险"""
    
    def __init__(self):
        self.risk_factors = {
            'infrastructure_compatibility': {
                'description': '新技术是否与现有基础设施兼容？',
                'weight': 0.25
            },
            'team_expertise': {
                'description': '团队是否有新技术的专业知识？',
                'weight': 0.20
            },
            'latency_requirements': {
                'description': '新技术能否满足延迟要求？',
                'weight': 0.20
            },
            'cost_projection': {
                'description': '成本预测是否现实？',
                'weight': 0.15
            },
            'rollback_complexity': {
                'description': '迁移失败时回滚有多复杂？',
                'weight': 0.10
            },
            'timeline_realism': {
                'description': '时间表是否现实？',
                'weight': 0.10
            }
        }
    
    def assess_migration(self, migration_plan):
        """评估迁移计划"""
        risks = []
        
        for factor, config in self.risk_factors.items():
            if factor in migration_plan:
                risk_level = migration_plan[factor]
                if risk_level == 'high':
                    risks.append({
                        'factor': factor,
                        'risk': 'HIGH',
                        'weight': config['weight'],
                        'mitigation': self._suggest_mitigation(factor)
                    })
        
        total_risk = sum(r['weight'] for r in risks if r['risk'] == 'HIGH')
        
        return {
            'total_risk': total_risk,
            'risks': risks,
            'recommendation': '继续' if total_risk < 0.3 else '重新考虑'
        }
    
    def _suggest_mitigation(self, factor):
        mitigations = {
            'infrastructure_compatibility': '考虑混合方法或逐步迁移',
            'team_expertise': '迁移前投资培训或雇佣专家',
            'latency_requirements': '在承诺之前用生产类负载进行基准测试',
            'cost_projection': '在成本估计中增加50%应急费用',
            'rollback_complexity': '开始迁移前设计回滚策略',
            'timeline_realism': '将估计时间表加倍'
        }
        return mitigations.get(factor, '未知因素')
```

### 经验教训

1. **技术选择必须考虑完整技术栈**：更好的模型如果基础设施不支持就没有帮助
2. **团队专业知识是关键因素**：新技术需要新技能
3. **延迟要求是硬约束**：10倍延迟回归可以抵消准确性改进
4. **成本预测需要应急费用**：GPU成本经常被低估
5. **迁移风险必须评估**：迁移风险可能超过收益

---

## 23.7 何时使用/何时不使用AI架构方法论

### 何时使用完整方法论

| 场景 | 原因 | 推荐方法 |
|------|------|---------|
| 新AI产品开发 | 全新，高风险 | 完整架构审查 |
| 企业AI部署 | 生产，合规 | 完整审查 + ADR |
| 监管合规 | 法律要求 | 完整审查 + 文档 |
| 主要技术迁移 | 高风险，高成本 | 完整审查 + 风险评估 |
| 多团队AI平台 | 需要协调 | 完整审查 + 治理 |

### 何时使用简化方法

| 场景 | 原因 | 推荐方法 |
|------|------|---------|
| 快速原型 | 时间限制 | 简化清单 |
| 研究项目 | 需要灵活性 | 非正式审查 |
| 小团队（<5人） | 资源限制 | 仅同行审查 |
| 低风险内部工具 | 有限影响 | 基本架构审查 |
| 独立开发者 | 无团队协调 | 带清单的自我审查 |

### 决策框架

```
这将投入生产吗？─── 是 ──→ 完整架构审查
         │
         否
         │
处理用户数据吗？─── 是 ──→ 完整审查 + 隐私评估
         │
         否
         │
是安全关键的吗？─── 是 ──→ 完整审查 + 安全评估
         │
         否
         │
是团队工作（>3人）吗？─── 是 ──→ 简化审查 + ADR
         │
         否
         │
基本清单 + 同行审查
```

---

## 23.8 总结

AI架构设计需要考虑ML系统独特特征的专门方法论：

1. **架构审查必须是AI特定的**：传统软件架构审查遗漏了关键的ML问题，如数据质量、模型漂移和评估不确定性。

2. **ADR捕获关键决策**：AI决策具有持久影响，需要彻底记录原理。

3. **技术选择需要结构化评估**：加权评分框架防止偏见或不完整的评估。

4. **构建vs购买决策是微妙的**：AI组件在定制、专业知识和战略重要性方面有独特考虑。

5. **必须评估迁移风险**：AI系统中的技术迁移比传统软件迁移风险更高。

6. **简化方法适用于低风险项目**：并非每个项目都需要完整方法——匹配流程与风险。

---

## 23.9 讨论题

1. **架构审查vs速度**：你如何平衡彻底的架构审查与AI项目快速迭代的需求？你会跳过什么？

2. **技术锁定**：如果你的团队深度投入PyTorch但新框架提供2倍更好的性能，你将如何评估切换？你会考虑哪些因素？

3. **ML模型的构建vs购买**：如果供应商提供预训练模型达到你准确性目标的90%，但你可以构建自定义模型达到95%，哪些因素会影响你的决策？

4. **ADR完整性**：AI决策的ADR应该多详细？简单的"我们选择X因为Y"是否足够，还是需要完整模板？

5. **扩展架构审查**：如果你从1个AI项目扩展到10个，你将如何调整架构审查流程？你会自动化什么？

---

## 23.10 练习

### 练习1：技术评估

使用第23.3节的决策矩阵评估两个ML编排平台（例如Kubeflow vs MLflow）：

1. 定义6个标准和权重
2. 对每个平台在每个标准上评分
3. 生成比较报告
4. 给出带原理的建议

### 练习2：ADR编写

为以下决策之一编写架构决策记录：

- 在批量和实时推理之间选择
- 选择特征存储实现
- 设计模型训练管道

使用第23.2.3节的模板并包含所有必需部分。

### 练习3：构建-购买分析

分析特征存储的构建vs购买决策：

1. 评估组织能力
2. 评估上市时间要求
3. 考虑3年总拥有成本
4. 给出带支持分析的建议

---

## 23.11 参考文献

### 架构方法论

1. Fowler, M. (2002). *企业应用架构模式*. Addison-Wesley. https://martinfowler.com/books/eaa.html

2. Bass, L., Clements, P., & Kazman, R. (2021). *软件架构实践*（第4版）. Addison-Wesley.

3. Richards, M. (2020). *软件架构模式*（第2版）. O'Reilly Media.

### AI特定架构

4. Hulten, G. (2022). *构建智能系统：机器学习工程指南*. Apress.

5. Schulman, J., & Levine, S. (2017). "Trust Region Policy Optimization." *ICML*. https://arxiv.org/abs/1502.05477

6. Amershi, S., 等. (2019). "Software Engineering for Machine Learning: A Case Study." *ICSE-SEIP*. https://doi.org/10.1109/ICSE-SEIP.2019.00043

### 技术选择

7. TechEmpower. (2024). "Web框架基准测试." https://www.techempower.com/benchmarks/

8. Papers With Code. (2024). "SOTA结果." https://paperswithcode.com/sota

### 行业报告

9. Stanford HAI. (2024). "AI指数报告2024." https://aiindex.stanford.edu/report/

10. McKinsey. (2024). "2024年AI现状." https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai

---

*下一章：[第24章：综合案例研究 →](./chapter-24.md)*
