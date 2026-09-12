# Chapter 2: AI System Design Principles

## Learning Objectives

By the end of this chapter, you will be able to:

- Apply Google's ML Best Practices to design robust, scalable AI systems
- Calculate and optimize GPU infrastructure costs across AWS, GCP, and Azure
- Design ML platform architectures that support hundreds of models
- Identify and prevent common architectural anti-patterns in production ML
- Implement monitoring and observability systems for AI workloads

---

## 2.1 Foundational Design Principles for AI Systems

### 2.1.1 Google's ML Best Practices as Architectural Guidelines

Google has published extensive documentation about their approach to ML engineering through their "Rules of Machine Learning" guide. These rules, distilled from thousands of production ML systems at Google, provide practical architectural guidelines that every AI architect should understand.

> **Real Data**
>
> Google's ML Best Practices document (developers.google.com/machine-learning/guides/rules-of-ml) is based on experience building ML systems that process over 1 trillion predictions daily across Search, Ads, YouTube, Maps, and other products. The guidelines have been validated through production incidents and successes across Google's diverse product portfolio.

Here are the key architectural principles derived from Google's practices:

**Principle 1: Design for the Entire ML Lifecycle**

Most ML projects fail not because of model quality, but because of inadequate attention to the full lifecycle. The architect must design systems that support data collection, feature engineering, model training, evaluation, deployment, monitoring, and retraining as an integrated pipeline.

```
Data Collection -> Feature Engineering -> Model Training -> Evaluation -> Deployment -> Monitoring -> Retraining
     ^                                                                                                    |
     |____________________________________ Feedback Loop ________________________________________________|
```

This is not a linear process—it is a cycle. The architect's job is to make this cycle as fast and reliable as possible. A typical ML lifecycle at a mature organization follows these stages:

| Stage | Duration (Typical) | Architect's Focus |
|-------|-------------------|-------------------|
| Data Collection | 2-4 weeks | Data quality, freshness, lineage |
| Feature Engineering | 2-6 weeks | Feature store, computation efficiency |
| Model Training | 1-4 weeks | Distributed training, experiment tracking |
| Evaluation | 1-2 weeks | Offline metrics, fairness, robustness |
| Deployment | 1-2 weeks | Serving infrastructure, rollback |
| Monitoring | Continuous | Drift detection, performance tracking |
| Retraining | 1-4 weeks | Trigger logic, automation |

**Principle 2: Keep It Simple First**

Start with a simple model and strong baselines before attempting complex architectures. A simple model with excellent data will outperform a complex model with mediocre data in most production scenarios.

> **Real Data**
>
> According to Google's internal data, teams that start with simple baselines and iterate toward complexity achieve 40% faster time-to-production and 25% lower maintenance costs compared to teams that begin with complex architectures. The most common failure pattern is "premature complexity"—jumping to deep learning before exhausting simpler approaches.

**Principle 3: Establish Good Feature Engineering Practices**

Features are the bridge between raw data and model predictions. The quality of your features directly determines model performance. Google recommends:

- **Feature Standardization:** Use consistent feature names, types, and computation logic across training and serving
- **Feature Discovery:** Maintain a catalog of available features so teams don't duplicate work
- **Feature Monitoring:** Track feature distributions, missing rates, and freshness in production
- **Feature Backfills:** Support historical feature computation for training without rebuilding entire pipelines

> **Real Data**
>
> Feast, the open-source feature store used by companies including Robinhood, NVIDIA, Discord, Cloudflare, and Walmart, has 5.5K+ community members and 12M+ downloads. Organizations using feature stores report 60-80% reduction in feature engineering time and near-elimination of training-serving skew (source: feast.dev).

**Principle 4: Design for Deployability and Monitoring from Day One**

The most common architectural mistake in ML projects is designing the model without considering how it will be deployed, monitored, and maintained. Every architectural decision should consider:

- **Latency requirements:** Can the model meet real-time constraints?
- **Resource efficiency:** What is the cost per prediction?
- **Observability:** Can you detect when the model is failing?
- **Rollback capability:** Can you quickly revert to a previous model version?
- **A/B testing:** Can you safely experiment with model changes?

### 2.1.2 The Four Pillars of AI System Design

Based on industry patterns from Google, Netflix, Uber, and other leaders, AI systems should be designed around four pillars:

**Pillar 1: Data Infrastructure**

The foundation of any AI system is its data infrastructure. This includes:

- **Data Lake/Warehouse:** Centralized storage for raw and processed data
- **Streaming Pipeline:** Real-time data ingestion for low-latency features
- **Batch Pipeline:** Historical data processing for training datasets
- **Data Versioning:** Ability to reproduce exact training datasets
- **Data Quality:** Automated validation, anomaly detection, and alerting

**Pillar 2: Feature Platform**

The feature platform provides consistent feature computation for both training and serving:

- **Feature Store:** Centralized repository for feature definitions and values
- **Online Store:** Low-latency feature serving for real-time predictions
- **Offline Store:** High-throughput feature computation for training
- **Feature Registry:** Metadata catalog for feature discovery and governance
- **Feature Monitoring:** Tracking feature health and drift

**Pillar 3: Model Platform**

The model platform handles training, evaluation, and serving:

- **Training Infrastructure:** Distributed training, experiment tracking, hyperparameter optimization
- **Model Registry:** Version control for trained models with metadata
- **Serving Infrastructure:** Online, batch, and edge inference with autoscaling
- **Model Optimization:** Quantization, distillation, pruning for deployment targets
- **Model Validation:** Automated testing before production deployment

**Pillar 4: Operations Platform**

The operations platform provides observability and control:

- **Monitoring:** Model performance, data drift, system health
- **Alerting:** Automated notifications for degradation
- **Logging:** Detailed audit trail for debugging and compliance
- **Rollback:** Quick recovery from failed deployments
- **Experimentation:** A/B testing infrastructure with statistical rigor

---

## 2.2 Real Cost Data: GPU Pricing and Infrastructure Economics

### 2.2.1 Cloud GPU Pricing (2024)

Understanding GPU costs is critical for AI architects making infrastructure decisions. Here is current pricing data from the three major cloud providers:

> **Real Data**
>
> **GPU Instance Pricing Comparison (On-Demand, US Regions, 2024):**
>
> | GPU Type | AWS (P4d) | GCP (A2) | Azure (NDv4) | Best For |
> |----------|-----------|----------|--------------|----------|
> | NVIDIA A100 (40GB) | $32.77/hr | $32.11/hr | $32.77/hr | Large model training, inference |
> | NVIDIA A100 (80GB) | $40.97/hr | $40.14/hr | $40.97/hr | Very large models, HPC |
> | NVIDIA V100 (16GB) | $12.24/hr | $11.91/hr | $12.24/hr | Medium model training |
> | NVIDIA T4 (16GB) | $1.51/hr | $1.48/hr | $1.51/hr | Inference, small model training |
> | NVIDIA L4 (24GB) | N/A | $2.73/hr | N/A | Inference, fine-tuning |
> | Google TPU v4 | N/A | $8.36/hr | N/A | Large-scale training |
>
> *Note: Prices as of Q3 2024. Actual prices may vary by region and availability.*
> Source: AWS Pricing Calculator, GCP Pricing, Azure Pricing (October 2024)

> **Real Data**
>
> **Cost Optimization Strategies (Real-World Impact):**
>
> | Strategy | Typical Savings | Implementation Effort |
> |----------|----------------|----------------------|
> | Spot/Preemptible Instances | 60-80% | Low (automatic retry logic needed) |
> | Reserved Instances (1-year) | 30-40% | Low (commitment required) |
> | Reserved Instances (3-year) | 50-60% | Medium (long-term planning) |
> | Right-sizing (matching GPU to workload) | 20-40% | Medium (profiling required) |
> | Mixed precision training (FP16/BF16) | 40-50% | Medium (code changes needed) |
> | Model distillation | 50-70% | High (model redesign required) |
>
> *Based on published case studies from Netflix, Uber, and Airbnb engineering blogs.*

### 2.2.2 Training vs. Inference Cost Analysis

A critical architectural decision is understanding the total cost of ownership (TCO) for ML systems:

> **Real Data**
>
> **Typical Cost Breakdown for Production ML Systems:**
>
> | Component | % of Total Cost | Key Cost Drivers |
> |-----------|----------------|------------------|
> | Training (initial) | 5-10% | GPU hours, data storage, experiment compute |
> | Training (ongoing retraining) | 20-30% | Frequency, model complexity, data volume |
> | Inference (serving) | 40-60% | QPS, latency requirements, model size |
> | Data Infrastructure | 10-15% | Storage volume, processing frequency |
> | Monitoring & Operations | 5-10% | Log volume, alert frequency, on-call costs |
>
> *Note: These percentages vary significantly by use case. Real-time serving workloads can push inference to 70%+ of costs.*
>
> Source: Aggregated from Uber's Michelangelo architecture blog, Netflix's ML platform posts, and Algorithmia's 2023 State of MLOps report

### 2.2.3 Cost-per-Prediction Framework

AI architects should think about cost in terms of per-prediction economics:

```
Cost per Prediction = (Infrastructure Cost / Total Predictions) + 
                      (Training Amortized / Predictions Since Retrain) +
                      (Data Pipeline Cost / Predictions Served)
```

**Example Calculation:**

```
Scenario: Real-time fraud detection, 10M predictions/day

Infrastructure (4x A100 GPU instances):
  Monthly cost: 4 * $32.77/hr * 730 hrs = $95,689/month

Training (weekly retraining):
  Training cost: 100 GPU hours * $32.77 = $3,277/week = $14,163/month
  Amortized per prediction: $14,163 / (300M predictions/month) = $0.000047

Data pipeline (Kafka + Spark):
  Monthly cost: $15,000/month
  Per prediction: $15,000 / 300M = $0.000050

Total per prediction: $0.000319 + $0.000047 + $0.000050 = $0.000416

At 10M predictions/day: $4,160/day or $124,800/month
```

This framework helps architects make informed decisions about model complexity, latency requirements, and infrastructure choices.

---

## 2.3 Case Study: Netflix's ML Platform Architecture

> **Case Study: Netflix's Metaflow and ML Platform**
>
> Netflix has published extensive documentation about their ML platform architecture through their engineering blog (netflixtechblog.com). This case study synthesizes their approach based on public materials.
>
> **The Problem:** Netflix needed to support hundreds of ML models across content recommendation, search ranking, marketing optimization, and content production. Each use case had different requirements for latency, data freshness, and model complexity.
>
> **The Architecture:** Netflix designed a layered ML platform with these key components:
>
> 1. **Metaflow:** Open-source ML infrastructure framework for building and managing real-world ML projects. Provides versioned data flows, artifact management, and integration with AWS services.
>
> 2. **Feature Store:** Centralized feature repository with both online (low-latency) and offline (high-throughput) stores. Features are computed using Spark and served via custom APIs.
>
> 3. **Training Infrastructure:** Distributed training on AWS GPU instances with automatic checkpointing, experiment tracking via internal tools, and hyperparameter optimization.
>
> 4. **Serving Layer:** Model serving via custom inference services with autoscaling, A/B testing, and canary deployments. Supports both real-time and batch predictions.
>
> 5. **Monitoring:** Continuous model performance monitoring with automated alerting for degradation. Tracks both system metrics (latency, throughput) and model metrics (accuracy, fairness).
>
> **Key Architectural Decisions:**
>
> - **Microservices for Model Serving:** Each model is deployed as an independent microservice, enabling independent scaling and deployment cycles.
>
> - **Feature Computation Separation:** Feature computation is separated from model serving to ensure consistency and enable reuse across models.
>
> - **Experimentation-First Design:** Every model deployment includes built-in A/B testing infrastructure, enabling safe experimentation at scale.
>
> - **Data Lineage Tracking:** Complete audit trail from raw data to model predictions, enabling debugging and compliance.
>
> - **Graceful Degradation:** Models are designed to degrade gracefully when features are unavailable, falling back to simpler heuristics.
>
> **Results:** According to Netflix's published metrics:
> - Over 1,000 ML models in production
> - 500M+ recommendations served daily
> - 99.99% uptime for recommendation service
> - 30% improvement in content discovery engagement
> - 20% reduction in customer churn through personalized marketing
>
> **Lessons for AI Architects:**
> 1. **Design for many models, not just one.** Platform thinking is essential at scale.
> 2. **Separate concerns:** Data, features, training, serving, and monitoring should be independent components.
> 3. **Invest in experimentation infrastructure.** A/B testing is not optional for production ML.
> 4. **Plan for graceful degradation.** Models will fail; the system must continue to function.
> 5. **Track everything.** Data lineage and audit trails are essential for debugging and compliance.
>
> Source: Netflix Tech Blog (netflixtechblog.com), Metaflow documentation (metaflow.org)

---

## 2.4 War Story: Data Leakage in Production ML

> **War Story: The Silent Killer - Data Leakage**
>
> *Adapted from real production incidents reported by ML teams at major technology companies*
>
> **The Situation:** A financial services company built a credit scoring model that achieved 95% accuracy on their test set—far exceeding the 85% target. The model was celebrated as a major success and quickly moved to production. Within three months, the model's production accuracy had degraded to 72%, causing $2.3M in bad loans.
>
> **The Root Cause: Data Leakage**
>
> Data leakage occurs when information that would not be available at prediction time is inadvertently included in the training data. In this case, there were two sources of leakage:
>
> **Leakage Source 1: Temporal Leakage**
> The training dataset included features computed using future information. Specifically, the "average transaction amount over the next 30 days" was included as a feature. During training, this feature was available because the data was historical. During production, this feature was not available because it required future data.
>
> **Leakage Source 2: Target Leakage**
> One feature was "number of credit inquiries in the past 7 days." This feature is causally related to the target (credit default), but it was computed using data from the same time period as the target. In production, this feature was delayed by 2-3 days due to reporting lags, creating a mismatch between training and serving.
>
> **The Architectural Failures:**
>
> 1. **No Feature Validation Pipeline:** There was no automated system to check that features available during training would also be available during serving. The feature validation was manual and ad-hoc.
>
> 2. **Missing Temporal Awareness:** The training pipeline did not enforce strict temporal boundaries. Features were computed across the entire dataset without considering time.
>
> 3. **No Production Feature Monitoring:** The team did not monitor feature availability or distribution in production. The leakage was only discovered when business users noticed unusual patterns in approved loans.
>
> 4. **Inadequate Holdout Strategy:** The test set was created by random sampling, not temporal split. This meant the model was evaluated on data from the same time period as training, masking the temporal leakage.
>
> **The Fix:**
>
> The team implemented these architectural changes:
>
> 1. **Feature Validation Service:** Automated checks that verify feature availability at serving time before each model deployment.
>
> 2. **Temporal Feature Computation:** All features are computed with strict point-in-time semantics, using only data available up to the prediction timestamp.
>
> 3. **Production Feature Monitoring:** Real-time monitoring of feature distributions, missing rates, and freshness with automated alerting.
>
> 4. **Temporal Holdout Strategy:** Test sets are always created from future data, ensuring models are evaluated on realistic serving conditions.
>
> **Key Lessons:**
>
> 1. **Data leakage is the #1 silent killer of ML models.** Models with leakage perform amazingly offline but fail catastrophically online.
> 2. **Feature validation is architectural, not just data science.** It must be built into the platform.
> 3. **Temporal boundaries are non-negotiable.** All features must respect point-in-time semantics.
> 4. **Monitoring must catch what testing misses.** Production monitoring is the last line of defense.
> 5. **When a model is too good to be true, it probably is.** Suspiciously high offline metrics should trigger investigation, not celebration.
>
> Source: Adapted from public postmortems at financial services companies and Google's ML Test Rules paper

---

## 2.5 Monitoring and Observability for AI Systems

### 2.5.1 The Three Pillars of ML Monitoring

Effective ML monitoring requires three interconnected pillars:

**Pillar 1: Data Monitoring**

Track the health and quality of incoming data:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Data completeness | % of expected records received | < 95% |
| Feature missing rate | % of null values per feature | > 5% increase |
| Data distribution shift | Statistical distance from training distribution | KL divergence > 0.1 |
| Data freshness | Time since last data update | > 2x expected interval |
| Schema violations | Records not matching expected format | > 1% of total |

**Pillar 2: Model Monitoring**

Track model performance and behavior:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Prediction distribution | Distribution of model outputs | Significant shift from baseline |
| Confidence calibration | Alignment between confidence and accuracy | ECE > 0.1 |
| Subgroup performance | Performance across demographic groups | > 10% disparity |
| Feature importance drift | Changes in model's feature usage | Major ranking changes |
| Prediction latency | Time to generate predictions | > 2x baseline |

**Pillar 3: System Monitoring**

Track infrastructure health:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| GPU utilization | % of GPU compute used | < 30% or > 90% |
| Memory usage | GPU/CPU memory consumption | > 85% |
| Error rate | Failed prediction requests | > 1% |
| Throughput | Predictions served per second | < 50% of capacity |
| Queue depth | Pending prediction requests | > 1000 |

### 2.5.2 Architecture for ML Observability

```
                          +-----------------+
                          |  Model Serving  |
                          |    Service      |
                          +--------+--------+
                                   |
                          +--------v--------+
                          |   Prediction    |
                          |    Logger       |
                          +--------+--------+
                                   |
                    +--------------+--------------+
                    |                             |
             +------v------+             +-------v-------+
             |   Data      |             |   Model       |
             |  Monitor    |             |  Monitor      |
             +------+------+             +-------+-------+
                    |                             |
             +------v------+             +-------v-------+
             |  Feature    |             |  Performance  |
             |  Store      |             |  Dashboard    |
             +------+------+             +-------+-------+
                    |                             |
                    +--------------+--------------+
                                 |
                          +------v------+
                          |  Alerting   |
                          |  System     |
                          +-------------+
```

> **Real Data**
>
> According to the 2023 State of MLOps report by Algorithmia (now DataRobot):
> - Only 22% of organizations have fully automated ML monitoring
> - 60% of ML model failures are detected by business users, not monitoring systems
> - Average time to detect ML model degradation: 14 days
> - Average time to remediate ML model issues: 28 days
> - Organizations with automated monitoring see 3x faster detection and 2x faster remediation
>
> Source: Algorithmia/DataRobot 2023 State of MLOps Report

---

## 2.6 Common Architectural Anti-Patterns

### Anti-Pattern 1: The Notebook-to-Production Pipeline

**Description:** Deploying code directly from Jupyter notebooks to production without proper software engineering practices.

**Why it fails:**
- No version control for experiments
- No dependency management
- No testing infrastructure
- No reproducibility guarantees

**The Fix:** Implement proper MLOps practices:
- Code version control (Git)
- Experiment tracking (MLflow, Weights & Biases)
- Automated testing (unit, integration, model)
- Containerized deployment (Docker, Kubernetes)

### Anti-Pattern 2: The Monolithic Model

**Description:** Building a single massive model that handles all use cases, requiring enormous compute resources.

**Why it fails:**
- High latency for simple predictions
- Expensive to serve at scale
- Difficult to update incrementally
- Single point of failure

**The Fix:** Design for composition:
- Multiple specialized models
- Model routing based on input characteristics
- Ensemble methods for complex cases
- Graceful degradation paths

### Anti-Pattern 3: The Training-Serving Skew Factory

**Description:** Using different code paths for feature computation in training vs. serving.

**Why it fails:**
- Model receives different inputs at serving time
- Performance degrades silently
- Difficult to diagnose and fix

**The Fix:** Implement feature stores with:
- Single feature computation logic
- Consistent data access patterns
- Automated skew detection
- Feature validation in CI/CD

### Anti-Pattern 4: The Monitoring Vacuum

**Description:** Deploying models without any monitoring infrastructure.

**Why it fails:**
- Degradation goes unnoticed
- Business impact accumulates
- Difficult to diagnose root cause
- No data for improvement

**The Fix:** Build monitoring from day one:
- Data quality monitoring
- Model performance tracking
- System health dashboards
- Automated alerting and rollback

---

## 2.7 Design Patterns for AI Systems

### Pattern 1: Lambda Architecture for ML

**Use Case:** Systems requiring both real-time and batch predictions.

```
                    +-------------------+
                    |   Data Sources    |
                    +--------+----------+
                             |
                    +--------v----------+
                    |   Stream Layer    |
                    |  (Kafka/Pulsar)   |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +-------v--------+
     |   Speed Layer   |          |  Batch Layer   |
     | (Real-time ML)  |          | (Daily ML)     |
     +--------+--------+          +-------+--------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v----------+
                    |   Serving Layer   |
                    |  (Combines both)  |
                    +-------------------+
```

### Pattern 2: Feature Store as Single Source of Truth

**Use Case:** Preventing training-serving skew across multiple models.

> **Real Data**
>
> Seldon Core, an open-source model serving platform, has 4.8K GitHub stars, 867 forks, and 2M+ installs. It supports 40+ backends including TensorFlow, PyTorch, XGBoost, and custom models. Adopted by Capital One, AstraZeneca, and GSK for production ML serving (source: seldon.io).
>
> Feast, as the feature store layer, integrates with Seldon Core to provide consistent feature serving. This combination has been deployed by companies like Robinhood and NVIDIA for high-throughput, low-latency ML serving.

### Pattern 3: Canary Deployment for ML Models

**Use Case:** Safely rolling out model updates without affecting all users.

```
Step 1: Deploy new model to 1% of traffic
         |
Step 2: Monitor metrics for 24 hours
         |
Step 3: If metrics pass, increase to 10%
         |
Step 4: Monitor for 48 hours
         |
Step 5: If metrics pass, increase to 50%
         |
Step 6: Monitor for 7 days
         |
Step 7: Full rollout (100%)
```

### Pattern 4: Multi-Armed Bandit for Model Selection

**Use Case:** Dynamically routing traffic to the best-performing model.

> **Real Data**
>
> Apache Kafka, the distributed streaming platform, has 33.7K GitHub stars, 15.5K forks, and is used by 80%+ of Fortune 100 companies. With 5M+ lifetime downloads, Kafka provides the streaming infrastructure for real-time ML feature serving and prediction logging (source: kafka.apache.org).
>
> Ray, the distributed computing framework, has 43.7K GitHub stars, 8K forks, and is used by OpenAI, Ant Group, and NVIDIA. Ray provides the compute layer for distributed ML training and serving, enabling dynamic resource allocation for multi-model deployments (source: github.com/ray-project/ray).

---

## 2.8 When to Use / When Not to Use Each Design Pattern

### Pattern Selection Guide

| Pattern | Best For | Avoid When | Complexity |
|---------|----------|------------|------------|
| Lambda Architecture | Dual-speed requirements | Simple use cases | High |
| Feature Store | Multiple models, strict consistency | Single model, simple features | Medium |
| Canary Deployment | Risk-averse organizations | Emergency fixes needed | Low |
| Multi-Armed Bandit | Dynamic optimization | Static requirements | Medium |
| Microservices (per model) | Independent scaling/deployment | Small team, few models | High |
| Monolithic ML Service | Simple deployment, tight coupling acceptable | Performance-critical, many models | Low |

### Decision Framework

```
Should you implement a Feature Store?
  - Do you have 5+ ML models? -> YES -> Implement Feature Store
  - Do training and serving use different code? -> YES -> Implement Feature Store
  - Do you have strict consistency requirements? -> YES -> Implement Feature Store
  - Otherwise -> Consider simpler approach

Should you use canary deployments?
  - Is the model business-critical? -> YES -> Use canary deployments
  - Do you have monitoring infrastructure? -> YES -> Use canary deployments
  - Can you tolerate degraded performance for hours? -> NO -> Use canary deployments
  - Otherwise -> Direct deployment may be acceptable
```

---

## 2.9 Summary

This chapter established the foundational design principles for AI systems. The key takeaways are:

1. **Google's ML Best Practices provide proven guidelines.** Design for the entire lifecycle, start simple, establish good feature practices, and plan for deployability from day one.

2. **GPU costs are significant but manageable.** On-demand A100 instances cost $32-41/hour across cloud providers. Cost optimization through spot instances, reserved capacity, and mixed precision can reduce costs by 50-80%.

3. **Netflix's architecture demonstrates platform thinking.** Their Metaflow-based platform supports 1,000+ models with 99.99% uptime through separated concerns and experimentation-first design.

4. **Data leakage is the #1 silent killer.** Temporal and target leakage cause models to perform well offline but fail catastrophically online. Feature validation and temporal awareness are architectural requirements.

5. **Monitoring is not optional.** Only 22% of organizations have fully automated ML monitoring. Implementing comprehensive data, model, and system monitoring catches degradation before business impact accumulates.

6. **Anti-patterns are common but avoidable.** Notebook-to-production pipelines, monolithic models, training-serving skew, and monitoring vacuums can all be prevented with proper architecture.

7. **Design patterns solve recurring problems.** Lambda architecture, feature stores, canary deployments, and multi-armed bandits are proven patterns for common ML architecture challenges.

---

## Discussion Questions

1. **Cost vs. Performance Trade-off:** A real-time recommendation model requires 100ms latency and serves 50M requests/day. You can deploy on 8x A100 GPUs ($32.77/hr each) for 50ms latency, or 4x T4 GPUs ($1.51/hr each) for 150ms latency. How do you evaluate this trade-off? What additional factors should you consider?

2. **Feature Store ROI:** Your team spends 40% of their time on feature engineering, and 30% of model failures are due to training-serving skew. Is implementing a feature store justified? How would you calculate the ROI?

3. **Monitoring Strategy:** You can only afford to implement monitoring for one of these three areas: data quality, model performance, or system health. Which should you prioritize? What are the risks of each choice?

4. **Anti-Pattern Recognition:** A data science team has built 15 models, all deployed directly from notebooks. There is no feature store, no monitoring, and no automated testing. As the AI architect, how do you prioritize remediation? What is your 6-month roadmap?

5. **Design Pattern Selection:** Your company needs to deploy ML models for fraud detection (real-time, 10ms latency), demand forecasting (batch, daily), and customer segmentation (batch, weekly). How would you architect these three use cases? Would you use the same pattern for all three?

---

## Exercises

### Exercise 1: GPU Cost Calculator

**Objective:** Build a cost estimation tool for ML infrastructure.

**Instructions:**
1. Create a spreadsheet or script that calculates monthly ML infrastructure costs
2. Include inputs for: model type, QPS, latency requirement, training frequency
3. Use real GPU pricing from AWS/GCP/Azure (refer to Section 2.2.1)
4. Output should include: recommended GPU type, number of instances, monthly cost, cost per prediction
5. Test with three scenarios:
   - Real-time fraud detection (100K QPS, 10ms latency)
   - Batch recommendation (1M predictions/day, 1-hour latency)
   - Edge inference (1K QPS, 100ms latency, T4 GPUs)

**Deliverable:** Cost calculator with three scenario analyses

### Exercise 2: ML Monitoring Dashboard Design

**Objective:** Design a comprehensive monitoring system for production ML.

**Instructions:**
1. Choose a specific ML use case (e.g., product recommendation, fraud detection, demand forecast)
2. Design a monitoring dashboard that includes:
   - Data quality metrics (completeness, freshness, distribution)
   - Model performance metrics (accuracy, latency, throughput)
   - System health metrics (GPU utilization, memory, error rates)
   - Business metrics (revenue impact, user satisfaction)
3. Define alert thresholds for each metric
4. Create a runbook for common failure scenarios
5. Estimate the infrastructure cost for the monitoring system

**Deliverable:** Dashboard design document + alert runbook

### Exercise 3: Architecture Review

**Objective:** Practice evaluating and improving ML architectures.

**Instructions:**
1. Research a real ML system from a public case study (Netflix, Uber, Airbnb, or Spotify engineering blogs)
2. Document the architecture using a standard template (C4 model or similar)
3. Identify 3 strengths and 3 weaknesses in the architecture
4. Propose 3 specific improvements with justification
5. Estimate the cost and effort for each improvement
6. Present your findings in a 10-minute presentation format

**Deliverable:** Architecture review document + presentation slides

---

## References

1. Google Cloud. "Rules of Machine Learning: Best Practices for ML Engineering." Google Developers. https://developers.google.com/machine-learning/guides/rules-of-ml

2. Google Cloud. "Vertex AI Documentation." Google Cloud. https://cloud.google.com/vertex-ai/docs

3. Netflix Tech Blog. "Scaling Machine Learning at Netflix." https://netflixtechblog.com/tagged/machine-learning

4. Netflix Tech Blog. "Metaflow: Human-centric ML Infrastructure." https://netflixtechblog.com/metaflow-human-centric-ml-infrastructure-b93263289546

5. vLLM Project. "vLLM: A High-Throughput and Memory-Efficient Inference and Serving Engine for LLMs." GitHub. https://github.com/vllm-project/vllm

6. Feast. "Feature Store for Machine Learning." feast.dev. https://feast.dev/

7. Kubeflow. "ML toolkit for Kubernetes." kubeflow.org. https://www.kubeflow.org/

8. Ray Project. "Ray: A General Framework for Distributed Computing." GitHub. https://github.com/ray-project/ray

9. Apache Kafka. "A Distributed Streaming Platform." kafka.apache.org. https://kafka.apache.org/

10. Seldon. "Seldon Core: Open Source Platform for Deploying ML Models." seldon.io. https://www.seldon.io/tech/products/core

11. Algorithmia/DataRobot. "2023 State of MLOps Report." https://www.datarobot.com/blog/state-of-mlops-2023/

12. AWS. "GPU Pricing." Amazon Web Services. https://aws.amazon.com/ec2/pricing/

13. GCP. "Compute Engine Pricing." Google Cloud Platform. https://cloud.google.com/compute/all-pricing

14. Azure. "Virtual Machine Pricing." Microsoft Azure. https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/

15. Uber Engineering. "Michelangelo: Uber's Machine Learning Platform." https://eng.uber.com/michelangelo-machine-learning-platform/

16. Airbnb Engineering. "Bighead: Airbnb's End-to-End Machine Learning Platform." https://medium.com/airbnb-engineering/bighead-airbnbs-end-to-end-machine-learning-platform-cf43f6e072c6

17. Spotify Engineering. "ML Platform Distillation." https://engineering.atspotify.com/category/machine-learning/
