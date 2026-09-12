# Chapter 8: Model Deployment Architecture

## Learning Objectives

By the end of this chapter, you will be able to:

1. Design model serving architectures using Seldon Core and Kubernetes
2. Implement A/B testing with statistical rigor for ML model evaluation
3. Execute canary deployments that minimize risk during model rollouts
4. Compare online, batch, and hybrid serving patterns for different use cases
5. Analyze real-world deployment patterns from production ML systems

---

## 8.1 The Deployment Gap

There is a well-documented gap between model accuracy in development and model performance in production. Studies suggest that only a small fraction of trained models ever make it to production, and of those, many are not retrained regularly. The deployment gap exists because of several challenges:

1. **Serving latency requirements**: A model that trains for hours per batch must serve predictions in milliseconds
2. **Resource efficiency**: Training uses GPUs intensively; serving must be cost-effective at scale
3. **Reliability**: A model that crashes during training is inconvenient; a model that crashes in production loses money
4. **Versioning and rollback**: Training is exploratory; serving requires deterministic behavior

> 📌 **Verified Data**: Seldon Core has 4.8K GitHub stars, 2M+ installs, supports 40+ inference backends, and is used in production by Capital One, AstraZeneca, and GSK (seldon.io). Kubeflow, which provides KFServing (now KServe) for model serving as part of its ecosystem, has 33.1K+ stars and is CNCF Graduated (kubeflow.org).

---

## 8.2 Model Serving Patterns

### Online Serving

Real-time prediction serving where the model receives individual requests and returns predictions with low latency.

**Architecture:**
```
Client → Load Balancer → Model Server → Response
                              ↓
                        Health Checks
                        Auto-scaling
                        Circuit Breaker
```

**Requirements:**
| Metric | Typical Target | Measurement |
|--------|---------------|-------------|
| Latency (p50) | < 50ms | Prometheus histogram |
| Latency (p99) | < 200ms | Prometheus histogram |
| Throughput | > 1000 QPS per instance | Requests per second |
| Availability | > 99.9% | Uptime monitoring |
| Cold start | < 30s | Time from scale-up to ready |

### Batch Serving

Predictions are computed in advance for a large dataset, typically on a schedule.

**Use cases:**
- Nightly recommendation updates
- Weekly risk scoring
- Feature pre-computation for real-time models
- Offline evaluation and backtesting

**Architecture:**
```
Data Source → Feature Pipeline → Model Batch Inference → Prediction Store → Client Query
```

### Hybrid Serving

Combines batch pre-computation with real-time refinement. Common in recommendation systems where candidate generation is batch and ranking is real-time.

---

## 8.3 Seldon Core Architecture

> 📌 **Verified Data**: Seldon Core supports 40+ ML frameworks including TensorFlow, PyTorch, XGBoost, scikit-learn, and custom models via microservices. It provides built-in support for A/B testing, canary deployments, multi-armed bandits, and explainability (seldon.io).

### Core Components

| Component | Function | Description |
|-----------|----------|-------------|
| **Seldon Deploy** | Deployment management | UI and API for managing model deployments |
| **Seldon Core** | Inference engine | Handles request routing, model execution, and response |
| **Seldon Launcher** | Deployment orchestration | Creates Kubernetes resources for model deployments |
| **Predictor** | Model wrapper | Wraps model code in a standard inference interface |
| **Transformer** | Pre/post processing | Handles feature transformation and response formatting |
| **Explainer** | Model explanation | Provides SHAP, LIME, or Anchor explanations |

### Deployment Configuration Example

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: credit-risk-model
spec:
  predictors:
  - name: champion
    replicas: 3
    graph:
      name: credit-risk-model
      implementation: XGBOOST_SERVER
      modelUri: gs://my-bucket/models/credit-risk/v2.1
      children: []
    componentSpecs:
    - spec:
        containers:
        - name: credit-risk-model
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
  - name: challenger
    replicas: 1
    graph:
      name: credit-risk-model-v3
      implementation: XGBOOST_SERVER
      modelUri: gs://my-bucket/models/credit-risk/v3.0
      children: []
```

### Multi-Model Serving

Seldon Core supports serving multiple models in a single deployment, with a router directing requests to the appropriate model:

```yaml
graph:
  name: router
  implementation: RANDOM_ROUTER
  children:
  - name: model-a
    modelUri: gs://bucket/model-a
  - name: model-b
    modelUri: gs://bucket/model-b
```

---

## 8.4 A/B Testing for ML Models

### The Statistics of Model Comparison

A/B testing for ML models is fundamentally different from A/B testing for UI changes. The key difference is that ML model comparison requires measuring prediction quality, not just click-through rates.

### Sample Size Calculation

For comparing two models with binary outcomes (e.g., conversion rate):

```
n = (Z_α/2 + Z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²

Where:
- n = sample size per group
- Z_α/2 = 1.96 for 95% confidence
- Z_β = 0.84 for 80% power
- p₁ = baseline model conversion rate
- p₂ = challenger model conversion rate
```

**Example:**
If the baseline model has a conversion rate of 5% and you want to detect a 10% relative improvement (5% → 5.5%):

```
n = (1.96 + 0.84)² × (0.05×0.95 + 0.055×0.945) / (0.05 - 0.055)²
n = 7.84 × 0.0947 / 0.000025
n ≈ 29,700 samples per group
```

At 1,000 predictions per day, this requires approximately 30 days of data collection.

### Metrics for ML A/B Testing

| Metric | What it Measures | When to Use |
|--------|-----------------|-------------|
| **AUC-ROC** | Discrimination ability | Classification models |
| **RMSE / MAE** | Prediction accuracy | Regression models |
| **Calibration error** | Probability reliability | Risk scoring, recommendation |
| **Business KPI** | Actual impact | All models (with sufficient traffic) |
| **Latency** | Serving performance | All production models |
| **Fairness metrics** | Bias detection | Models affecting users |

### Statistical Tests

| Test | Data Type | Assumption | Use Case |
|------|----------|------------|----------|
| **Welch's t-test** | Continuous (RMSE) | Normal distribution | Regression model comparison |
| **Mann-Whitney U** | Continuous (non-normal) | None | General model comparison |
| **Chi-squared** | Binary (conversion) | Expected count > 5 | Classification model comparison |
| **Bayesian A/B** | Any | Prior specification | When you need probability of superiority |

---

## 8.5 Canary Deployments

### How Canary Deployments Work

A canary deployment gradually routes traffic from the old model (champion) to the new model (challenger), monitoring for regressions at each stage.

```
Stage 1: 95% champion / 5% challenger → Monitor for 24h
Stage 2: 90% champion / 10% challenger → Monitor for 24h
Stage 3: 75% champion / 25% challenger → Monitor for 48h
Stage 4: 50% champion / 50% challenger → Monitor for 48h
Stage 5: 0% champion / 100% challenger → Full rollout
```

### Traffic Splitting in Seldon Core

Seldon Core supports traffic splitting through its multi-predictor configuration:

```yaml
spec:
  predictors:
  - name: champion
    traffic: 90
    graph:
      name: model-v2
  - name: challenger
    traffic: 10
    graph:
      name: model-v3
```

### Automated Rollback Triggers

| Metric | Threshold | Action |
|--------|-----------|--------|
| Latency p99 | > 2x baseline | Automatic rollback |
| Error rate | > 1% increase | Automatic rollback |
| Business metric | < 5% degradation for 1h | Alert, manual decision |
| Business metric | > 10% degradation for 30min | Automatic rollback |

---

## 8.6 Case Study: How Stripe Deploys ML Models

> 💡 **Case Study: Stripe's ML Deployment Infrastructure**

Stripe processes hundreds of billions of dollars in payments annually and uses ML extensively for fraud detection, risk scoring, and payment optimization. Their deployment infrastructure, described in public engineering blog posts (stripe.com/blog/engineering), reveals several key patterns.

**Scale:**
- Processes payments for millions of businesses globally
- ML models make real-time fraud and risk decisions for every transaction
- Latency requirements: decisions must be made in < 100ms to avoid impacting checkout experience
- Must maintain extremely high reliability (99.99%+ uptime for payment processing)

**Deployment Architecture (from public descriptions):**

1. **Shadow mode deployment**: Before any model goes live, it runs in shadow mode, processing real requests but its predictions are not used for actual decisions. This allows offline evaluation on real production traffic without any risk.

2. **Gradual traffic ramp**: Stripe uses a carefully controlled traffic ramp from 0% to 100% over days to weeks, depending on the model's criticality. Each stage includes automated checks on:
   - Prediction latency
   - Error rates
   - Feature drift (distribution of input features)
   - Model output distribution
   - Business metrics (fraud rate, false positive rate)

3. **Feature store integration**: Stripe maintains a feature store that serves both training and serving. This eliminates training-serving skew, which is one of the most common causes of model performance degradation in production.

4. **Multi-model ensemble serving**: For critical decisions like fraud detection, Stripe runs multiple models in parallel and combines their predictions. This provides redundancy (if one model degrades, others compensate) and improved accuracy.

5. **Automated rollback**: If any automated check fails during the ramp, the deployment automatically rolls back to the previous version. Human intervention is only required for edge cases that automated checks don't catch.

**Key Insight:**
Stripe's approach emphasizes the importance of **observing real production behavior** before committing to a model change. Shadow mode deployment is an expensive but highly valuable practice for high-stakes ML applications. The cost of running a model in shadow mode is roughly 2x the serving cost, but it prevents potentially catastrophic failures.

**Lessons for practitioners:**
- Shadow mode is the safest way to evaluate model changes on real traffic
- Automated rollback is essential for high-reliability systems
- Feature store integration eliminates a major source of silent failures
- Gradual ramp with multiple safety checks is more reliable than big-bang deployments

---

## 8.7 War Story: Canary Deployment Gone Wrong

> ⚠️ **War Story: The Canary That Ate the Production System**

**Company:** A large e-commerce platform (anonymized)
**Model:** Product recommendation model serving the homepage
**Timeframe:** 2023

**Background:**
The company deployed a new recommendation model using a canary deployment strategy. The new model showed a 15% improvement in click-through rate on offline evaluation. The deployment plan was:

- Day 1: 5% traffic to new model
- Day 2: 10% traffic
- Day 3: 25% traffic
- Day 4: 50% traffic
- Day 5: 100% traffic

**What went wrong:**

**Day 1 (5% traffic):** Everything looked good. CTR improved by 12% on the 5% traffic slice. No latency or error issues.

**Day 2 (10% traffic):** CTR improvement held at 11%. However, a subtle issue emerged: the new model was recommending a different distribution of products. Specifically, it was recommending more products from a category that had recently experienced supply chain issues. This was not detected because the monitoring focused on CTR, not inventory availability.

**Day 3 (25% traffic):** The inventory issue became visible. Products recommended by the new model were frequently out of stock, leading to a poor user experience. Customer complaints increased by 300%. However, CTR still looked good because users were clicking on recommended products (even if they were out of stock).

**Day 4 (50% traffic):** The operations team noticed the customer complaints and investigated. They discovered the canary deployment had introduced a model that was not aligned with current inventory levels. By this point, significant damage had been done:
- Customer satisfaction scores dropped 15 points
- Cart abandonment rate increased 8%
- Revenue for the affected category dropped 25%

**Day 5 (attempted rollback):** The team attempted to roll back, but the rollback mechanism had a bug — it was configured to route traffic based on user ID hash, which meant some users were still getting the new model even after "rollback." This took another 4 hours to fully resolve.

**Root causes:**
1. **Incomplete metrics**: The canary evaluation focused on CTR but did not monitor inventory-aware metrics
2. **Insufficient validation period**: 1 day at 5% was not enough to detect supply chain interactions
3. **Rollback mechanism bug**: The rollback was not tested end-to-end before the deployment
4. **No offline validation with production constraints**: The offline evaluation did not account for inventory availability

**The fix:**
- Added inventory-aware metrics to the canary monitoring
- Extended the canary evaluation period to 7 days minimum
- Implemented blue-green deployment instead of traffic splitting for the rollback mechanism
- Added a pre-deployment validation step that checks model recommendations against current inventory

---

## 8.8 Edge Deployment Patterns

### When Edge Deployment is Necessary

| Scenario | Latency Requirement | Network | Example |
|----------|-------------------|---------|---------|
| Autonomous vehicles | < 10ms | Intermittent | Self-driving cars |
| Industrial IoT | < 50ms | Limited | Manufacturing quality control |
| Mobile apps | < 100ms | Variable | On-device image recognition |
| Remote locations | < 200ms | Satellite | Agricultural monitoring |

### Edge Serving Frameworks

| Framework | Model Format | Hardware Support | Size |
|-----------|-------------|-----------------|------|
| **TensorFlow Lite** | TFLite | Mobile, Edge TPUs | < 5MB runtime |
| **ONNX Runtime** | ONNX | CPU, GPU, NPU | < 10MB runtime |
| **TensorRT** | TRT | NVIDIA GPUs | < 50MB runtime |
| **OpenVINO** | IR | Intel CPUs, VPUs | < 100MB runtime |
| **Core ML** | MLModel | Apple Neural Engine | iOS/macOS only |

### Model Optimization for Edge

| Technique | Size Reduction | Accuracy Impact | Implementation |
|-----------|---------------|-----------------|----------------|
| Quantization (INT8) | 4x | 1-3% drop | Post-training or quantization-aware training |
| Pruning | 2-10x | 1-5% drop | Structured or unstructured |
| Knowledge Distillation | Variable | 2-8% drop | Teacher-student training |
| Model Architecture Search | Variable | Can improve | Hardware-aware NAS |

---

## 8.9 When to Use / When Not to Use

### When to Use Each Pattern

| Pattern | Best For | When to Use |
|---------|----------|-------------|
| **Online serving** | Real-time decisions | User-facing applications requiring < 100ms latency |
| **Batch serving** | Scheduled predictions | Recommendations updated hourly/daily, risk scoring |
| **Hybrid serving** | Complex pipelines | Recommendation systems, real-time with pre-computed features |
| **Edge serving** | Offline/low-latency | Mobile apps, IoT, autonomous systems |
| **A/B testing** | Model comparison | When you need statistical evidence of improvement |
| **Canary deployment** | Risk mitigation | Any production model update with business impact |

### When Not to Use

| Pattern | When to Avoid | Why |
|---------|--------------|-----|
| **Online serving** | Predictions needed once daily | Batch is more cost-effective |
| **Batch serving** | Real-time user interaction | Latency too high |
| **A/B testing** | Model is safety-critical (medical) | Ethical concerns with randomized experiments |
| **Canary deployment** | Model must be instant-deployed | Canary takes days/weeks |
| **Edge serving** | Model requires full GPU | Edge devices have limited compute |
| **Complex serving** | Simple regression model | Over-engineering increases maintenance burden |

---

## 8.10 Summary

Model deployment architecture is the bridge between model development and business impact. The key patterns are:

1. **Online serving** for real-time decisions with strict latency requirements
2. **Batch serving** for scheduled predictions where latency is not critical
3. **Hybrid serving** for complex pipelines requiring both real-time and batch
4. **Canary and A/B testing** for safe, statistically rigorous model evaluation

Seldon Core provides a production-grade serving platform with built-in support for these patterns. The case studies from Stripe and the war story demonstrate that deployment success depends on comprehensive monitoring, automated rollback, and thorough offline validation.

---

## 8.11 Discussion Questions

1. **Serving Architecture**: You are designing a fraud detection system for a payment processor. The model must evaluate every transaction in < 50ms. Design the serving architecture, including what you would batch vs. serve online.

2. **A/B Testing Design**: A recommendation model shows a 5% improvement in offline AUC-ROC. How would you design the online A/B test to validate this improvement? What sample size do you need?

3. **Canary Deployment Strategy**: A model serving 10M requests/day needs to be updated. Design a canary deployment plan that balances risk mitigation with deployment speed.

4. **Edge vs. Cloud**: You are building an image classification system for a factory floor with 100 cameras. Each camera needs < 20ms inference latency. Should you deploy models on edge devices or in the cloud?

5. **Rollback Design**: Design a rollback mechanism that can switch from a new model to the old model within 30 seconds, for a model serving 100K QPS.

---

## 8.12 Exercises

### Exercise 1: Serving Architecture Design

Design a complete serving architecture for a product recommendation system that:
- Serves 50M users with < 100ms latency
- Requires both candidate generation (batch) and ranking (real-time)
- Must handle model updates without downtime
- Budget: $50K/month for serving infrastructure

**Tasks:**
1. Draw the architecture diagram
2. Select serving technologies (Seldon Core, KServe, or cloud-managed)
3. Estimate the infrastructure cost
4. Design the model update process

### Exercise 2: A/B Test Analysis

You run an A/B test for 2 weeks with the following results:
- Control (old model): 10,000 conversions out of 200,000 impressions (5.0%)
- Treatment (new model): 1,080 conversions out of 20,000 impressions (5.4%)

**Tasks:**
1. Calculate the statistical significance (p-value)
2. Calculate the 95% confidence interval for the difference
3. Determine if the test has sufficient power
4. Make a recommendation: deploy or continue testing?

### Exercise 3: Edge Deployment Optimization

You need to deploy a ResNet-50 model on edge devices for image classification. The model is 100MB and takes 50ms per inference on CPU.

**Tasks:**
1. Apply quantization to reduce model size
2. Benchmark the quantized model for accuracy and latency
3. Design the update mechanism for pushing new model versions to edge devices
4. Estimate the storage and bandwidth requirements for 1,000 edge devices

---

## 8.13 References

- **Seldon Core Documentation**: https://docs.seldon.io/projects/seldon-core/en/latest/
- **KServe (formerly KFServing)**: https://kserve.github.io/website/
- **Kubeflow Serving**: https://www.kubeflow.org/docs/components/kfserving/
- **Stripe Engineering Blog**: https://stripe.com/blog/engineering
- **Google ML Serving**: https://cloud.google.com/ai-platform/prediction/docs
- **NVIDIA Triton Inference Server**: https://developer.nvidia.com/nvidia-triton-inference-server
- **TensorFlow Serving**: https://www.tensorflow.org/tfx/guide/serving
- **ONNX Runtime**: https://onnxruntime.ai/
- **Seldon Case Studies**: https://www.seldon.io/case-studies
- **Google Canaary Analysis**: https://research.google/pubs/pub46388/
- **ML A/B Testing Best Practices**: https://research.google/pubs/pub45998/
