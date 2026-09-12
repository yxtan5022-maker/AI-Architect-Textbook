# Chapter 16: AI Platform Architecture

## Learning Objectives

By the end of this chapter, you will be able to:

1. Design a unified AI platform that supports the full lifecycle from data ingestion to model deployment
2. Implement self-service AI platform patterns that enable data scientists without infrastructure expertise
3. Evaluate build-vs-buy decisions for AI platform components
4. Architect platform governance that balances flexibility with operational stability
5. Avoid common failure modes in AI platform development, including over-engineering

---

## 16.1 Introduction: What is an AI Platform?

An AI platform is an integrated set of tools and services that enables data scientists and ML engineers to develop, train, deploy, and monitor machine learning models with minimal friction. It abstracts away infrastructure complexity so practitioners can focus on model quality rather than cluster management.

The term "AI platform" is deliberately broad. It encompasses feature stores, experiment tracking, model registries, training infrastructure, serving infrastructure, and monitoring — all connected through well-defined APIs and workflows.

**Why platforms matter:** Without a platform, every team reinvents the wheel. They build ad-hoc training scripts, manually manage GPU clusters, create custom deployment pipelines, and write bespoke monitoring dashboards. This duplication wastes engineering time and produces inconsistent, fragile systems.

> **📌 Real Data Box**
> Uber's **Michelangelo** platform, launched in 2017, serves as the backbone of Uber's ML infrastructure, supporting over 1,000 models in production across ride pricing, ETA prediction, fraud detection, and autonomous vehicles. The platform handles 50+ million predictions per second at peak load (Uber Engineering Blog).

---

## 16.2 AI Platform Architecture Patterns

### Pattern 1: Layered Architecture

The most common AI platform architecture organizes components in horizontal layers:

```
┌─────────────────────────────────────────────┐
│            Developer Interface              │
│    (IDE, CLI, SDK, Web UI, Notebooks)       │
├─────────────────────────────────────────────┤
│            Orchestration Layer              │
│   (Workflow engine, scheduling, triggers)   │
├─────────────────────────────────────────────┤
│            ML Framework Layer               │
│  (Training, Serving, Feature Store, Registry)│
├─────────────────────────────────────────────┤
│            Infrastructure Layer             │
│   (Kubernetes, GPU pools, Storage, Network) │
├─────────────────────────────────────────────┤
│            Observability Layer              │
│    (Monitoring, Logging, Alerting, Traces)  │
└─────────────────────────────────────────────┘
```

**Layer responsibilities:**

| Layer | Components | Key Design Decisions |
|-------|-----------|---------------------|
| Developer Interface | JupyterHub, VS Code extensions, CLI tools | Must be familiar to data scientists |
| Orchestration | Airflow, Kubeflow Pipelines, Argo | DAG-based vs event-driven |
| ML Framework | MLflow, Kubeflow, KServe, Feast | Open-source vs managed |
| Infrastructure | Kubernetes, Terraform, cloud services | Multi-cloud vs single cloud |
| Observability | Prometheus, Grafana, custom dashboards | Model metrics vs infrastructure metrics |

### Pattern 2: Service-Oriented Architecture

Instead of horizontal layers, organize the platform as independent services with well-defined APIs:

| Service | API Contract | Data Flow |
|---------|-------------|-----------|
| Data Service | Read/Write datasets, versioning | S3/GCS ↔ Training |
| Training Service | Submit/monitor/cancel jobs | API → Kubernetes → GPU nodes |
| Registry Service | Register/query models | Training → Registry → Serving |
| Serving Service | Deploy/predict/rollback | Registry → Serving → Traffic |
| Monitoring Service | Metrics/alerts/reports | Serving → Monitoring → Alerts |

**Advantage:** Each service can be developed, deployed, and scaled independently. Teams can swap implementations (e.g., replace MLflow with Weights & Biases) without affecting other services.

**Disadvantage:** Service boundaries create integration complexity. Each service needs its own API versioning, authentication, and error handling.

### Pattern 3: Platform-as-a-Product

The most mature organizations treat their AI platform as an internal product, with:

- **Dedicated platform team** with product management, engineering, and design
- **User research** — understanding data scientist workflows and pain points
- **Versioned APIs** — backward compatibility guarantees
- **Self-service capabilities** — no tickets to infrastructure teams
- **Documentation and onboarding** — new team members productive within days
- **SLAs** — guaranteed uptime, latency, and support response times

---

## 16.3 Self-Service AI Platform Architecture

The goal of a self-service AI platform is to enable any data scientist to go from idea to production model without filing infrastructure tickets.

### 16.3.1 Self-Service Components

| Capability | Self-Service Mechanism | Required Infrastructure |
|-----------|----------------------|------------------------|
| Notebook environment | One-click JupyterHub spawn | Kubernetes + PVC + image registry |
| Training job submission | CLI or API call | Training operator + GPU quotas |
| Experiment tracking | Automatic logging | MLflow/Weights & Biases server |
| Model deployment | One-command serve | KServe + Ingress + autoscaling |
| Data access | Schema-based data catalog | Metastore + ACL system |
| Feature computation | Feature store API | Feast/Hopsworks + Spark |

### 16.3.2 Developer Workflow

A well-designed self-service platform supports this workflow:

```
1. Open notebook environment (auto-provisioned, GPU-enabled)
2. Pull data from feature store / data catalog
3. Write training code using familiar frameworks (PyTorch, TensorFlow)
4. Track experiments automatically (metrics, artifacts, parameters)
5. Register best model to model registry
6. Deploy model to serving endpoint with one command
7. Monitor model performance via dashboard
8. Retrain when data drift detected (automated trigger)
```

Each step should require zero infrastructure knowledge from the data scientist.

### 16.3.3 Guardrails and Governance

Self-service does not mean uncontrolled. A well-designed platform enforces governance through:

| Governance Mechanism | Implementation |
|---------------------|----------------|
| Resource quotas | Namespace-level CPU/GPU/memory limits |
| Cost tracking | Per-team cost attribution via labels |
| Model approval gates | Required review before production deployment |
| Data access controls | Role-based access to sensitive datasets |
| Compliance checks | Automated PII detection in training data |
| Reproducibility | Mandatory experiment tracking and model registry |

---

## 16.4 Case Study: How Uber Built Michelangelo

Uber's Michelangelo platform was designed to serve the needs of a rapidly growing ML organization spanning ride pricing, ETA prediction, fraud detection, and autonomous driving.

**Platform components:**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Feature Store | Hopsworks-based | Feature computation and serving |
| Training | Custom + Spark + PyTorch | Batch and online training |
| Model Registry | Custom | Model versioning and metadata |
| Serving | Custom (Prediction Services) | Low-latency real-time inference |
| Batch Prediction | Spark-based | Offline scoring at scale |
| Monitoring | Custom dashboards | Model performance tracking |

**Scale metrics:**

- 1,000+ models in production
- 50+ million predictions per second (peak)
- 10,000+ feature computations per second
- 100+ data scientists using the platform
- 15+ distinct business domains

**Key architectural decisions:**

1. **Feature Store as the foundation.** Michelangelo's feature store precomputes and caches features needed for online prediction. This eliminates the train-serve skew problem where training features differ from serving features. Features are computed offline in Spark and stored in a key-value store for online serving.

2. **Separation of training and serving infrastructure.** Training uses batch-oriented resources (large GPU nodes that spin up and down). Serving uses always-on resources with autoscaling. This separation prevents training jobs from impacting prediction latency.

3. **Model metadata as a first-class citizen.** Every model in the registry includes: training data version, hyperparameters, evaluation metrics, feature pipeline version, and deployment history. This enables debugging production issues by tracing predictions back to training conditions.

4. **Gradual rollout by default.** New models are deployed behind a shadow traffic router that sends real traffic to both old and new models. Performance is compared before full traffic cutover.

**Challenges faced:**

- **Platform complexity grew faster than team size.** By 2020, Michelangelo had 20+ microservices, and the platform team spent more time maintaining the platform than building new features.
- **Solution:** Consolidated redundant services and invested in a unified API layer.

---

## 16.5 War Story: Platform That Became Too Complex to Maintain

**Company:** Mid-size AI startup, 50 data scientists, 3 platform engineers

**Problem:** The company built a custom AI platform from scratch, adding services as needs arose. After 2 years, the platform had accumulated:

- 3 different experiment tracking systems (each built for a different team)
- 2 model serving frameworks (one for batch, one for real-time)
- 4 different ways to access training data (HDFS, S3 direct, SQL, custom API)
- 2 orchestration engines (Airflow + custom)
- Custom authentication (not integrated with company SSO)
- No API versioning — breaking changes deployed weekly

**Symptoms:**

| Metric | Value |
|--------|-------|
| Time to deploy new model | 2-3 weeks (should be < 1 day) |
| Platform incidents per month | 8-12 |
| Data scientist onboarding time | 3-4 weeks |
| Platform engineer time spent on bugs | 70% |
| Documentation coverage | ~30% |

**Root causes:**

1. **No platform vision.** Each service was built in isolation by different engineers at different times. No one owned the end-to-end experience.
2. **Premature abstraction.** The team built custom APIs for everything instead of using existing tools (MLflow, Seldon, Feast).
3. **No deprecation policy.** Old systems were never removed — they accumulated alongside new ones.
4. **Insufficient investment in platform engineering.** Three engineers cannot maintain a platform serving 50 data scientists.

**Remediation:**

1. **Platform audit.** Cataloged every component, its owner, its users, and its dependencies. Identified 40% of components as redundant or unused.
2. **Consolidation.** Migrated to a single experiment tracker (MLflow), a single serving framework (KServe), and a single data access layer (feature store). Deleted or archived unused systems.
3. **Platform team expansion.** Grew from 3 to 8 engineers, with dedicated product manager.
4. **API governance.** Established versioned APIs with 6-month deprecation windows.
5. **Developer experience investment.** Created a CLI tool (`mlplatform init`, `mlplatform train`, `mlplatform serve`) that encapsulates best practices.

**Results after 6 months:**

| Metric | Before | After |
|--------|--------|-------|
| Time to deploy new model | 2-3 weeks | 2 hours |
| Platform incidents per month | 8-12 | 1-2 |
| Data scientist onboarding time | 3-4 weeks | 2-3 days |
| Platform engineer bug time | 70% | 30% |
| Documentation coverage | 30% | 85% |

---

## 16.6 Build vs Buy Decisions

| Component | Build | Buy/Use Open Source | Recommendation |
|-----------|-------|-------------------|----------------|
| Experiment tracking | Custom dashboard | MLflow, W&B | Use MLflow/W&B — custom tracking is rarely better |
| Feature store | Custom feature computation | Feast, Hopsworks | Start with Feast; build custom only if unique needs |
| Model serving | Custom inference server | KServe, Triton, Seldon | Use KServe/Triton — production serving is hard |
| Training orchestration | Custom job manager | Kubeflow, Airflow | Use Kubeflow Pipelines for ML-specific workflows |
| Data versioning | Custom snapshot system | DVC, Delta Lake | Use DVC for files, Delta Lake for tables |
| Monitoring | Custom dashboards | Evidently AI, Whylabs | Start with open source, customize for ML-specific needs |
| GPU cluster management | Custom scheduler | Kubernetes + GPU Operator | Use Kubernetes — custom GPU schedulers are a trap |

**The golden rule:** If an open-source tool covers 80% of your needs, use it and customize the remaining 20%. Building from scratch is only justified when no existing tool comes close to your requirements.

---

## 16.7 When to Use / When Not to Use an AI Platform

### When to Build an AI Platform

| Scenario | Why a Platform Helps |
|----------|---------------------|
| > 10 data scientists | Shared infrastructure prevents duplication |
| > 5 models in production | Standardized deployment reduces incidents |
| Multiple business domains | Feature reuse across teams |
| Regulatory compliance required | Audit trails, model governance |
| Rapid team growth | Self-service reduces onboarding time |

### When NOT to Build an AI Platform

| Scenario | Why a Platform Hurts | Alternative |
|----------|---------------------|-------------|
| < 5 data scientists | Platform maintenance exceeds benefit | JupyterHub + MLflow + scripts |
| Prototyping phase | Platform constraints slow experimentation | Local development + git |
| No production models | Solving a problem you don't have yet | Focus on getting first model to production |
| No platform engineering team | Unmaintained platform becomes a liability | Use managed services (SageMaker, Vertex AI) |
| Budget < $200K/year for platform | Cannot fund dedicated platform team | Managed ML services |

---

## 16.8 Summary

- AI platforms abstract infrastructure complexity, enabling data scientists to focus on model quality
- Three primary architecture patterns: **layered** (horizontal layers), **service-oriented** (independent services), and **platform-as-a-product** (internal product with dedicated team)
- **Self-service** is the key value proposition — data scientists should go from idea to production without filing infrastructure tickets
- **Feature stores** are the foundation that eliminates train-serve skew
- The most common failure mode is **over-engineering** — building custom solutions for everything instead of leveraging existing tools
- Uber's **Michelangelo** demonstrates the scale: 1,000+ models, 50M predictions/second, but even Uber consolidated services after realizing complexity was growing faster than capability

---

## Discussion Questions

1. A company has 15 data scientists across 3 teams, each using different tools (Jupyter notebooks, SageMaker, custom scripts). They want to standardize on a single platform. What components would you prioritize building, and what would you buy/use open source?

2. Uber's Michelangelo accumulated 20+ microservices over 3 years. How would you design governance and architecture principles to prevent this kind of platform sprawl?

3. Compare the developer experience of using a self-service AI platform versus manually managing infrastructure. What are the tradeoffs in flexibility, speed, and operational burden?

4. A healthcare AI company needs HIPAA compliance for their ML platform. How does this requirement change the architecture decisions compared to a non-regulated company?

5. Should feature stores be centralized (one store for all teams) or decentralized (each team owns their features)? What are the tradeoffs?

---

## Exercises

**Exercise 1:** Design a minimal AI platform architecture for a team of 8 data scientists. List every component, whether you would build or buy it, and the estimated infrastructure cost per month.

**Exercise 2:** Set up a self-service MLflow tracking server on Kubernetes with per-namespace isolation. Create a notebook environment that automatically logs experiments to MLflow when a data scientist starts a training job.

**Exercise 3:** Audit an existing ML project and identify which parts could be replaced by existing open-source tools. Estimate the time saved per month by switching to those tools.

---

## References

- Uber Michelangelo ML Platform: https://eng.uber.com/michelangelo-machine-learning/
- Uber Michelangelo PyML: https://eng.uber.com/uber-creates-michelangelo-pyml/
- MLflow Documentation: https://mlflow.org/docs/latest/
- Feast Feature Store: https://docs.feast.dev/
- KServe Documentation: https://kserve.github.io/website/
- MLOps Community: https://mlops.community/
- Chip Huyen's ML Systems Design: https://huyenchip.com/machine-learning-systems-design/
- Google MLOps Whitepaper: https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
