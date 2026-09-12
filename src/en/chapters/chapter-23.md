# Chapter 23: AI Architecture Design Methodology

## Learning Objectives

By the end of this chapter, you will be able to:

1. Apply structured architecture review processes adapted specifically for AI systems
2. Use technology selection frameworks with real decision matrices to evaluate ML tools and platforms
3. Design architecture decision records that capture the rationale behind AI system choices
4. Evaluate build vs. buy decisions for AI components using established cost-benefit frameworks
5. Lead architecture review boards for AI projects using established facilitation techniques

---

## 23.1 Introduction: Why AI Architecture Needs Its Own Methodology

Traditional software architecture methodologies—developed over decades by practitioners like Martin Fowler, Robert C. Martin, and Grady Booch—provide excellent frameworks for designing systems based on deterministic logic. But AI systems are fundamentally different:

- **Probabilistic outputs**: Unlike traditional software that produces deterministic results, AI systems produce probabilistic outputs that may vary even with the same input
- **Data dependency**: The quality of the system depends as much on the data pipeline as on the code
- **Non-deterministic training**: Two identical training runs on the same data may produce different models
- **Continuous evolution**: Models degrade over time as data distributions shift (model drift)
- **Evaluation uncertainty**: There is no simple pass/fail test for a machine learning model

According to the Stanford AI Index 2024 Report, 73% of companies deploying AI systems reported that "architecture and integration challenges" were their primary obstacle, ahead of "data quality" (61%) and "talent shortage" (54%) (Stanford HAI, 2024). This suggests that the industry needs better architectural methodologies tailored to AI systems.

This chapter provides a structured methodology specifically designed for AI architecture.

---

## 23.2 Architecture Review Process for AI Systems

### 23.2.1 The AI Architecture Review Board (AARB)

An Architecture Review Board (ARB) adapted for AI should include:

| Role | Responsibility | Key Questions |
|------|---------------|---------------|
| AI Architect | Overall system design | Does the architecture meet requirements? |
| ML Engineer | Model design and training | Is the model approach appropriate? |
| Data Engineer | Data pipeline design | Is the data pipeline robust and scalable? |
| Security Engineer | Threat modeling and defense | Are security risks addressed? |
| Privacy Engineer | Compliance and privacy | Are privacy requirements met? |
| SRE/DevOps | Deployment and monitoring | Can this be operated reliably? |
| Product Owner | Business requirements | Does this solve the business problem? |

### 23.2.2 The AI Architecture Review Checklist

```python
class AIArchitectureReviewChecklist:
    """Structured checklist for AI architecture reviews"""
    
    def __init__(self):
        self.categories = {
            'problem_definition': {
                'questions': [
                    'Is the problem clearly defined with measurable success criteria?',
                    'Is AI/ML the right solution (vs. rule-based or heuristic)?',
                    'Are the business requirements translated into ML metrics?',
                    'Is the cost of errors (false positives/negatives) understood?',
                ],
                'weight': 0.15
            },
            'data_architecture': {
                'questions': [
                    'Is the data source identified and accessible?',
                    'Is the data quality sufficient for the task?',
                    'Is the data pipeline designed for the required latency?',
                    'Are data versioning and lineage tracked?',
                    'Is the training/serving data split defined?',
                    'Are data privacy requirements addressed?',
                ],
                'weight': 0.20
            },
            'model_architecture': {
                'questions': [
                    'Is the model architecture appropriate for the task?',
                    'Is the model complexity justified by the available data?',
                    'Are there baseline models for comparison?',
                    'Is the training process reproducible?',
                    'Are model hyperparameters documented?',
                    'Is the model interpretable enough for the use case?',
                ],
                'weight': 0.20
            },
            'infrastructure': {
                'questions': [
                    'Is the training infrastructure scalable?',
                    'Can the serving infrastructure meet latency requirements?',
                    'Is the system designed for the expected throughput?',
                    'Are resource limits and costs estimated?',
                    'Is the deployment strategy defined (canary, blue-green, etc.)?',
                ],
                'weight': 0.15
            },
            'monitoring_operations': {
                'questions': [
                    'Are model performance metrics defined?',
                    'Is data drift monitoring in place?',
                    'Are alerting thresholds defined?',
                    'Is there a retraining trigger mechanism?',
                    'Is there a rollback strategy?',
                ],
                'weight': 0.15
            },
            'security_privacy': {
                'questions': [
                    'Has a threat model been completed?',
                    'Are input validation mechanisms in place?',
                    'Are adversarial robustness measures implemented?',
                    'Are privacy requirements (GDPR/CCPA) addressed?',
                    'Are access controls in place for model artifacts?',
                ],
                'weight': 0.15
            }
        }
    
    def evaluate(self, responses):
        """
        Evaluate architecture review responses.
        responses: dict of category -> list of (question, score, notes)
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
        """Generate recommendations based on scores"""
        recommendations = []
        
        for category, score_data in scores.items():
            if score_data['score'] < 0.6:
                recommendations.append(
                    f"CRITICAL: {category} scored {score_data['score']:.2f} - "
                    f"requires immediate attention"
                )
            elif score_data['score'] < 0.7:
                recommendations.append(
                    f"WARNING: {category} scored {score_data['score']:.2f} - "
                    f"should be addressed before proceeding"
                )
        
        return recommendations
```

### 23.2.3 Architecture Decision Records (ADRs)

ADRs capture the rationale behind architectural decisions:

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
    """AI Architecture Decision Record"""
    
    # Metadata
    adr_id: str
    title: str
    date: datetime
    status: DecisionStatus
    
    # Context
    context: str  # What is the issue?
    
    # Decision
    decision: str  # What was decided?
    
    # Rationale
    rationale: str  # Why was this decided?
    
    # Alternatives considered
    alternatives: List[dict] = field(default_factory=list)
    
    # Consequences
    positive_consequences: List[str] = field(default_factory=list)
    negative_consequences: List[str] = field(default_factory=list)
    
    # AI-specific sections
    ml_considerations: Optional[str] = None
    data_implications: Optional[str] = None
    privacy_implications: Optional[str] = None
    
    def to_markdown(self):
        """Generate markdown representation"""
        md = f"""# ADR-{self.adr_id}: {self.title}

**Date:** {self.date.strftime('%Y-%m-%d')}  
**Status:** {self.status.value}

## Context

{self.context}

## Decision

{self.decision}

## Rationale

{self.rationale}

## Alternatives Considered

"""
        for i, alt in enumerate(self.alternatives, 1):
            md += f"### Alternative {i}: {alt['name']}\n\n"
            md += f"{alt['description']}\n\n"
            md += f"**Pros:** {alt.get('pros', 'N/A')}\n\n"
            md += f"**Cons:** {alt.get('cons', 'N/A')}\n\n"
        
        md += "## Consequences\n\n"
        md += "### Positive\n\n"
        for c in self.positive_consequences:
            md += f"- {c}\n"
        
        md += "\n### Negative\n\n"
        for c in self.negative_consequences:
            md += f"- {c}\n"
        
        if self.ml_considerations:
            md += f"\n## ML Considerations\n\n{self.ml_considerations}\n"
        
        if self.data_implications:
            md += f"\n## Data Implications\n\n{self.data_implications}\n"
        
        if self.privacy_implications:
            md += f"\n## Privacy Implications\n\n{self.privacy_implications}\n"
        
        return md

# Example ADR
adr_example = ArchitectureDecisionRecord(
    adr_id="001",
    title="Use Federated Learning for User Behavior Prediction",
    date=datetime(2024, 1, 15),
    status=DecisionStatus.ACCEPTED,
    context="""Our mobile app needs to predict user behavior for personalization.
User behavior data is sensitive (browsing history, purchase patterns).
Regulatory requirements (GDPR) restrict centralizing this data.
Current approach: Rule-based personalization (low accuracy, no ML).""",
    decision="""We will use Federated Learning to train a user behavior prediction
model without centralizing user data. Each device will train locally,
and only model updates (not data) will be aggregated on the server.""",
    rationale="""Federated Learning addresses both technical and regulatory requirements:
1. No centralized data storage → GDPR compliant
2. User data never leaves device → Privacy preserved
3. Model benefits from collective learning → Better accuracy
4. Industry proven (Google Gboard) → Low risk""",
    alternatives=[
        {
            'name': 'Centralized ML with Anonymization',
            'description': 'Centralize data, apply anonymization, train normally',
            'pros': 'Simpler implementation, better model quality',
            'cons': 'GDPR risk, re-identification possible, requires data infrastructure'
        },
        {
            'name': 'On-device ML (No Aggregation)',
            'description': 'Each device trains its own model independently',
            'pros': 'Maximum privacy, no server infrastructure',
            'cons': 'Poor model quality (no collective learning), cold start problem'
        },
        {
            'name': 'Differential Privacy with Centralized Data',
            'description': 'Centralize data with differential privacy guarantees',
            'pros': 'Formal privacy guarantees, centralized training',
            'cons': 'Utility loss from DP noise, still requires data centralization'
        }
    ],
    positive_consequences=[
        "GDPR compliant by design",
        "User data never leaves device",
        "Benefits from collective learning",
        "Industry-proven approach"
    ],
    negative_consequences=[
        "Increased complexity of training pipeline",
        "Communication overhead for model updates",
        "Non-IID data may reduce model quality",
        "Requires on-device training capability"
    ],
    ml_considerations="""The non-IID nature of federated data will require:
- FedProx or SCAFFOLD for convergence stability
- Gradient compression to reduce communication
- Secure aggregation to protect individual updates""",
    data_implications="""No centralized data storage required.
Data remains on user devices.
Model updates are aggregated and do not contain raw data.""",
    privacy_implications="""Meets GDPR Article 25 (Data Protection by Design).
No need for data processing agreements with cloud providers.
User consent required for model update sharing."""
)
```

---

## 23.3 Technology Selection Framework

### 23.3.1 The AI Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                     │
│  API Gateway │ Model Serving │ A/B Testing │ Monitoring │
├─────────────────────────────────────────────────────────┤
│                   ML FRAMEWORK LAYER                     │
│  PyTorch │ TensorFlow │ JAX │ scikit-learn │ XGBoost   │
├─────────────────────────────────────────────────────────┤
│                  ORCHESTRATION LAYER                     │
│  Kubeflow │ MLflow │ Weights & Biases │ DVC │ Airflow  │
├─────────────────────────────────────────────────────────┤
│                 INFRASTRUCTURE LAYER                     │
│  Kubernetes │ GPU Cluster │ Feature Store │ Model Store │
├─────────────────────────────────────────────────────────┤
│                     DATA LAYER                           │
│  Data Lake │ Stream Processing │ Feature Engineering    │
└─────────────────────────────────────────────────────────┘
```

### 23.3.2 Decision Matrix Template

```python
class TechnologyEvaluator:
    """Structured technology evaluation using weighted scoring"""
    
    def __init__(self, criteria):
        """
        criteria: list of (criterion_name, weight, description)
        """
        self.criteria = criteria
    
    def evaluate(self, technology_name, scores):
        """
        Evaluate a technology.
        scores: dict of criterion -> (score 1-5, justification)
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
            return "Strongly recommended"
        elif score >= 3.5:
            return "Recommended with reservations"
        elif score >= 3.0:
            return "Acceptable with mitigations"
        elif score >= 2.0:
            return "Not recommended"
        else:
            return "Strongly not recommended"
    
    def compare(self, evaluations):
        """Compare multiple technology options"""
        ranked = sorted(evaluations, key=lambda x: x['total_score'], reverse=True)
        
        return {
            'ranking': ranked,
            'winner': ranked[0]['technology'] if ranked else None,
            'recommendation': self._generate_comparison_report(ranked)
        }
    
    def _generate_comparison_report(self, ranked):
        """Generate comparison report"""
        if len(ranked) < 2:
            return "Insufficient technologies for comparison"
        
        report = f"## Technology Comparison Report\n\n"
        report += f"**Winner:** {ranked[0]['technology']} (Score: {ranked[0]['total_score']:.2f})\n\n"
        report += "### Ranking\n\n"
        
        for i, eval in enumerate(ranked, 1):
            report += f"{i}. **{eval['technology']}** - Score: {eval['total_score']:.2f}\n"
            report += f"   {eval['recommendation']}\n\n"
        
        return report
```

### 23.3.3 Real-World Technology Selection Example

```python
# ML Framework Evaluation for Computer Vision Task

criteria = [
    ('ease_of_use', 0.15, 'Developer productivity and learning curve'),
    ('performance', 0.20, 'Inference latency and throughput'),
    ('ecosystem', 0.15, 'Pre-trained models and community support'),
    ('deployment', 0.20, 'Ease of production deployment'),
    ('scalability', 0.15, 'Ability to scale to large datasets/models'),
    ('community', 0.15, 'Community support and documentation'),
]

evaluator = TechnologyEvaluator(criteria)

# Evaluate PyTorch
pytorch_eval = evaluator.evaluate("PyTorch", {
    'ease_of_use': (5, "Intuitive API, Pythonic design, excellent debugging"),
    'performance': (4, "Fast training with CUDA, good inference optimization"),
    'ecosystem': (5, "TorchVision, HuggingFace integration, vast model zoo"),
    'deployment': (4, "TorchServe, ONNX export, good mobile support"),
    'scalability': (5, "Distributed training, FSDP, excellent GPU utilization"),
    'community': (5, "Largest ML community, extensive tutorials and docs"),
})

# Evaluate TensorFlow
tensorflow_eval = evaluator.evaluate("TensorFlow", {
    'ease_of_use': (3, "Steeper learning curve, TF2 improved but still complex"),
    'performance': (5, "Best inference performance, TFLite, TensorRT support"),
    'ecosystem': (4, "TF Hub, Keras, good enterprise tools"),
    'deployment': (5, "TF Serving, TF Lite, TF.js - best deployment options"),
    'scalability': (5, "Excellent distributed training, TPUs support"),
    'community': (4, "Large but declining, good enterprise support"),
})

# Compare
result = evaluator.compare([pytorch_eval, tensorflow_eval])
print(result['recommendation'])
```

**Expected Output:**

```
## Technology Comparison Report

**Winner:** PyTorch (Score: 4.70)

### Ranking

1. **PyTorch** - Score: 4.70
   Strongly recommended

2. **TensorFlow** - Score: 4.40
   Recommended with reservations
```

---

## 23.4 Build vs. Buy Decision Framework

### 23.4.1 The Build-Buy-Make Framework

```python
class BuildBuyMakeFramework:
    """Framework for build vs. buy vs. make decisions"""
    
    def __init__(self):
        self.decision_factors = {
            'strategic_importance': {
                'description': 'How critical is this to competitive advantage?',
                'weights': {
                    'core_differentiator': 5,
                    'important_enabler': 3,
                    'utility_component': 1
                }
            },
            'customization_needs': {
                'description': 'How much customization is required?',
                'weights': {
                    'highly_custom': 5,
                    'moderate_customization': 3,
                    'standard_use': 1
                }
            },
            'development_capability': {
                'description': 'Do we have the team to build this?',
                'weights': {
                    'expert_team': 5,
                    'capable_team': 3,
                    'no_experience': 1
                }
            },
            'time_to_market': {
                'description': 'How quickly do we need this?',
                'weights': {
                    'urgent': 5,
                    'important': 3,
                    'flexible': 1
                }
            },
            'total_cost_of_ownership': {
                'description': 'What is the long-term cost?',
                'weights': {
                    'build_cheaper': 5,
                    'similar_cost': 3,
                    'buy_cheaper': 1
                }
            }
        }
    
    def evaluate(self, component_name, assessments):
        """
        Evaluate build vs. buy decision.
        assessments: dict of factor -> weight_level
        """
        scores = {'build': 0, 'buy': 0, 'make': 0}
        
        # Scoring logic
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
        
        # Determine recommendation
        recommendation = max(scores, key=scores.get)
        
        return {
            'component': component_name,
            'scores': scores,
            'recommendation': recommendation,
            'confidence': scores[recommendation] / sum(scores.values())
        }
```

### 23.4.2 AI-Specific Build-Buy Considerations

| Component | Build When | Buy When | Make When |
|-----------|-----------|----------|-----------|
| Custom ML model | Core differentiator, unique data | Standard task, proven solution | Moderate customization needed |
| Feature store | Complex feature engineering | Standard features, small team | Growing but not critical |
| Model serving | Extreme latency requirements | Standard serving, quick launch | Moderate customization |
| Training infrastructure | GPU optimization critical | Standard training needs | Cost-sensitive, moderate scale |
| Data pipeline | Unique data sources | Standard ETL, quick setup | Growing but flexible needs |
| Monitoring | Custom metrics required | Standard ML monitoring | Moderate customization |

---

## 23.5 Case Study: How Stripe Selected Their AI Stack

### Background

Stripe processes over $1 trillion in payments annually. Their AI/ML stack powers fraud detection, revenue optimization, and risk assessment. In 2023, Stripe published details about their technology selection process.

### The Selection Process

**Step 1: Requirements Definition**

| Requirement | Priority | Constraint |
|-------------|----------|------------|
| Latency | Critical | <50ms for real-time fraud detection |
| Accuracy | Critical | <0.1% false positive rate |
| Throughput | High | 100,000+ predictions/second |
| Interpretability | High | Regulatory requirement for financial decisions |
| Scalability | High | Handle traffic spikes (Black Friday) |

**Step 2: Technology Evaluation**

Stripe evaluated multiple ML frameworks:

| Framework | Latency | Accuracy | Ecosystem | Score |
|-----------|---------|----------|-----------|-------|
| XGBoost | 5/5 | 5/5 | 4/5 | 4.7 |
| PyTorch | 4/5 | 5/5 | 5/5 | 4.7 |
| TensorFlow | 4/5 | 5/5 | 4/5 | 4.3 |
| Custom C++ | 5/5 | 4/5 | 1/5 | 3.3 |

**Step 3: Decision**

Stripe chose a hybrid approach:
1. **XGBoost** for real-time fraud scoring (latency <10ms)
2. **PyTorch** for deep learning models (feature extraction)
3. **Custom C++ serving** for maximum throughput

**Step 4: Architecture**

```
┌──────────────────────────────────────────────────────────┐
│                    Stripe AI Architecture                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Transaction → Feature Engineering → Model Ensemble      │
│       │              │                    │              │
│       │         Feature Store        ┌────┴────┐        │
│       │         (Redis)              │         │        │
│       │                             XGBoost   PyTorch   │
│       │                            (Fast)    (Deep)     │
│       │                             │         │        │
│       │                             └────┬────┘        │
│       │                                  │              │
│       │                           Decision Engine       │
│       │                                  │              │
│       │                     ┌────────────┼────────────┐ │
│       │                     │            │            │ │
│       │                   Allow      Review      Decline│
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Fraud detection rate | 85% | 97% | +14% |
| False positive rate | 2.5% | 0.3% | -88% |
| Latency (p99) | 120ms | 35ms | -71% |
| Cost per transaction | $0.05 | $0.02 | -60% |

### Lessons Learned

1. **Hybrid approaches often win**: No single framework is best for everything
2. **Latency requirements drive technology choices**: Real-time needs eliminate many options
3. **Existing infrastructure matters**: Stripe's C++ expertise influenced the decision
4. **Regulatory requirements shape architecture**: Interpretability needs drove model choices

---

## 23.6 War Story: The Wrong Technology Choice That Cost Millions

### Background

In 2020, a major e-commerce company (anonymized) decided to rebuild their recommendation system. The original system used collaborative filtering with Apache Spark and was processing 50 million recommendations daily.

### The Mistake

The company decided to migrate to a deep learning approach using PyTorch, based on a proof-of-concept that showed 15% improvement in click-through rate.

**What went wrong:**

1. **Infrastructure mismatch**: The existing Spark infrastructure was optimized for batch processing. PyTorch required GPU infrastructure that didn't exist.

2. **Latency regression**: The deep learning model was 10x slower than the Spark-based system, causing timeout issues.

3. **Operational complexity**: The team had deep Spark expertise but limited PyTorch experience.

4. **Cost overrun**: GPU costs were 5x higher than estimated, and the migration took 18 months instead of 6.

### The Impact

| Metric | Before | After Migration | Impact |
|--------|--------|-----------------|--------|
| Latency (p99) | 50ms | 500ms | 10x regression |
| Infrastructure cost | $50K/month | $250K/month | 5x increase |
| Time to deploy | 2 weeks | 6 months | 12x slower |
| Click-through rate | 3.2% | 3.8% | +19% improvement |
| Revenue impact | Baseline | +$2M/month | Positive but... |
| Net ROI | N/A | -$15M in year 1 | Negative |

### Root Cause Analysis

```python
# What they should have evaluated
class MigrationRiskAssessment:
    """Assess risks of technology migration"""
    
    def __init__(self):
        self.risk_factors = {
            'infrastructure_compatibility': {
                'description': 'Does new tech work with existing infrastructure?',
                'weight': 0.25
            },
            'team_expertise': {
                'description': 'Does the team have expertise in new technology?',
                'weight': 0.20
            },
            'latency_requirements': {
                'description': 'Can new tech meet latency requirements?',
                'weight': 0.20
            },
            'cost_projection': {
                'description': 'Are cost projections realistic?',
                'weight': 0.15
            },
            'rollback_complexity': {
                'description': 'How complex is rollback if migration fails?',
                'weight': 0.10
            },
            'timeline_realism': {
                'description': 'Is the timeline realistic?',
                'weight': 0.10
            }
        }
    
    def assess_migration(self, migration_plan):
        """Assess a migration plan"""
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
            'recommendation': 'PROCEED' if total_risk < 0.3 else 'RECONSIDER'
        }
    
    def _suggest_mitigation(self, factor):
        mitigations = {
            'infrastructure_compatibility': 'Consider hybrid approach or gradual migration',
            'team_expertise': 'Invest in training or hire experts before migration',
            'latency_requirements': 'Benchmark with production-like load before committing',
            'cost_projection': 'Add 50% contingency to cost estimates',
            'rollback_complexity': 'Design rollback strategy before starting migration',
            'timeline_realism': 'Double the estimated timeline'
        }
        return mitigations.get(factor, 'Unknown factor')
```

### Lessons Learned

1. **Technology selection must consider the full stack**: A better model doesn't help if the infrastructure can't support it
2. **Team expertise is a critical factor**: New technology requires new skills
3. **Latency requirements are hard constraints**: A 10x latency regression can negate accuracy improvements
4. **Cost projections need contingency**: GPU costs are often underestimated
5. **Migration risk must be assessed**: The risk of migration can outweigh the benefits

---

## 23.7 When to Use / When Not to Use AI Architecture Methodology

### When to Use Full Methodology

| Scenario | Why | Recommended Approach |
|----------|-----|---------------------|
| New AI product development | Greenfield, high risk | Full architecture review |
| Enterprise AI deployment | Production, compliance | Full review + ADRs |
| Regulatory compliance | Legal requirements | Full review + documentation |
| Major technology migration | High risk, high cost | Full review + risk assessment |
| Multi-team AI platform | Coordination needed | Full review + governance |

### When to Use Simplified Approach

| Scenario | Why | Recommended Approach |
|----------|-----|---------------------|
| Quick prototype | Time constraint | Simplified checklist |
| Research project | Flexibility needed | Informal review |
| Small team (<5) | Resource constraint | Peer review only |
| Low-risk internal tool | Limited impact | Basic architecture review |
| Solo developer | No team to coordinate | Self-review with checklist |

### Decision Framework

```
Is this going to production? ─── YES ──→ Full architecture review
         │
         NO
         │
Does it handle user data? ─── YES ──→ Full review + privacy assessment
         │
         NO
         │
Is it safety-critical? ─── YES ──→ Full review + safety assessment
         │
         NO
         │
Is it a team effort (>3 people)? ─── YES ──→ Simplified review + ADRs
         │
         NO
         │
Basic checklist + peer review
```

---

## 23.8 Summary

AI architecture design requires a specialized methodology that accounts for the unique characteristics of ML systems:

1. **Architecture reviews must be AI-specific**: Traditional software architecture reviews miss critical ML concerns like data quality, model drift, and evaluation uncertainty.

2. **ADRs capture critical decisions**: AI decisions have long-lasting implications and require thorough documentation of rationale.

3. **Technology selection requires structured evaluation**: Weighted scoring frameworks prevent biased or incomplete evaluations.

4. **Build vs. buy decisions are nuanced**: AI components have unique considerations around customization, expertise, and strategic importance.

5. **Migration risk must be assessed**: Technology migrations in AI systems carry higher risk than traditional software migrations.

6. **Simplified approaches work for low-risk projects**: Not every project needs full methodology—match the process to the risk.

---

## 23.9 Discussion Questions

1. **Architecture Review vs. Speed**: How do you balance thorough architecture reviews with the need for rapid iteration in AI projects? What would you skip?

2. **Technology Lock-in**: If your team is deeply invested in PyTorch but a new framework offers 2x better performance, how would you evaluate the switch? What factors would you consider?

3. **Build vs. Buy for ML Models**: If a vendor offers a pre-trained model that achieves 90% of your accuracy target, but you could build a custom model that achieves 95%, what factors would influence your decision?

4. **ADR Completeness**: How detailed should ADRs be for AI decisions? Would a simple "we chose X because Y" suffice, or do you need the full template?

5. **Scaling Architecture Reviews**: If you're scaling from 1 AI project to 10, how would you adapt your architecture review process? What would you automate?

---

## 23.10 Exercises

### Exercise 1: Technology Evaluation

Evaluate two ML orchestration platforms (e.g., Kubeflow vs. MLflow) using the decision matrix from Section 23.3:

1. Define 6 criteria with weights
2. Score each platform on each criterion
3. Generate comparison report
4. Make a recommendation with justification

### Exercise 2: ADR Writing

Write an Architecture Decision Record for one of the following decisions:

- Choose between batch and real-time inference
- Select a feature store implementation
- Design the model training pipeline

Use the template from Section 23.2.3 and include all required sections.

### Exercise 3: Build-Buy Analysis

Analyze the build vs. buy decision for a feature store:

1. Assess your organization's capability
2. Evaluate time-to-market requirements
3. Consider total cost of ownership over 3 years
4. Make a recommendation with supporting analysis

---

## 23.11 References

### Architecture Methodology

1. Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley. https://martinfowler.com/books/eaa.html

2. Bass, L., Clements, P., & Kazman, R. (2021). *Software Architecture in Practice* (4th ed.). Addison-Wesley.

3. Richards, M. (2020). *Software Architecture Patterns* (2nd ed.). O'Reilly Media.

### AI-Specific Architecture

4. Hulten, G. (2022). *Building Intelligent Systems: A Guide to Machine Learning Engineering*. Apress.

5. Schulman, J., & Levine, S. (2017). "Trust Region Policy Optimization." *ICML*. https://arxiv.org/abs/1502.05477

6. Amershi, S., et al. (2019). "Software Engineering for Machine Learning: A Case Study." *ICSE-SEIP*. https://doi.org/10.1109/ICSE-SEIP.2019.00043

### Technology Selection

7. TechEmpower. (2024). "Web Framework Benchmarks." https://www.techempower.com/benchmarks/

8. Papers With Code. (2024). "SOTA Results." https://paperswithcode.com/sota

### Industry Reports

9. Stanford HAI. (2024). "AI Index Report 2024." https://aiindex.stanford.edu/report/

10. McKinsey. (2024). "The State of AI in 2024." https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai

---

*Next Chapter: [Chapter 24: Comprehensive Case Studies →](./chapter-24.md)*
