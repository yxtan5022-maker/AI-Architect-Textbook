# AI 架构师：从理论到实践

# AI Architect: From Theory to Practice

<p align="center">
  <em>A comprehensive bilingual guide to designing, building, and operating production-grade AI systems — from data pipelines to LLM deployment, from cloud-native platforms to edge inference.</em>
</p>

<p align="center">
  <a href="#chinese-version">中文版</a> | <a href="#english-version">English</a>
</p>

---

## English Version

### Overview

This book is a practical, bilingual (Chinese/English) guide for AI architects, covering the full lifecycle of AI systems. It bridges the gap between academic ML knowledge and real-world production engineering, providing actionable architecture patterns backed by open-source tools.

**~450 pages** across 25 chapters and 5 appendices, with hands-on case studies using real open-source projects.

### Target Audience

| Reader Type | Description |
|-------------|-------------|
| **Junior Developers** | New to AI/ML system design, need foundational knowledge |
| **Mid-level Engineers** | Have ML experience, ready to design production systems |
| **Senior Architects** | Leading AI platform design and technical strategy |
| **Technical Managers** | Making build-vs-buy decisions, evaluating AI investments |

### Reader Level Guide

| Marker | Level | Description |
|--------|-------|-------------|
| 🟢 | **Beginner** | Foundational content — all readers should start here |
| 🟡 | **Intermediate** | Requires basic AI/ML knowledge |
| 🔴 | **Advanced** | Requires architecture design experience |
| ⚫ | **Manager** | Focused on decision-making and management perspective |
| 📝 | **Exercise** | Hands-on practice |
| 💡 | **Case Study** | Real-world project example |
| ⚠️ | **Warning** | Common pitfalls and mistakes |
| 📌 | **Key Concept** | Must-know knowledge |

### Table of Contents

#### Part 1: The AI Architect Role & Foundations

| Ch | Title | Level | Pages |
|----|-------|-------|-------|
| 1 | Defining the AI Architect Role | 🟢 | 15 |
| 2 | AI System Design Principles | 🟢 | 20 |

#### Part 2: Data Architecture

| Ch | Title | Level | Pages |
|----|-------|-------|-------|
| 3 | Data Pipeline Architecture | 🟢 | 30 |
| 4 | Feature Engineering Architecture | 🟡 | 25 |
| 5 | Data Lakehouse Architecture | 🟡 | 25 |

#### Part 3: MLOps Architecture

| Ch | Title | Level | Pages |
|----|-------|-------|-------|
| 6 | MLOps Fundamentals & Maturity Model | 🟢 | 20 |
| 7 | Model Training Architecture | 🟡 | 30 |
| 8 | Model Deployment Architecture | 🟡 | 30 |
| 9 | Model Monitoring & Observability | 🟡 | 25 |

#### Part 4: Large Model Architecture

| Ch | Title | Level | Pages |
|----|-------|-------|-------|
| 10 | LLM Architecture Design Fundamentals | 🟡 | 30 |
| 11 | LLM Inference Architecture | 🔴 | 30 |
| 12 | RAG System Architecture | 🟡 | 30 |
| 13 | Model Fine-tuning Architecture | 🔴 | 25 |

#### Part 5: Cloud-Native AI Architecture

| Ch | Title | Level | Pages |
|----|-------|-------|-------|
| 14 | AI Workloads on Kubernetes | 🔴 | 30 |
| 15 | Distributed Computing Architecture | 🔴 | 25 |
| 16 | AI Platform Architecture | 🔴 | 25 |

#### Part 6: Edge AI Architecture

| Ch | Title | Level | Pages |
|----|-------|-------|-------|
| 17 | Edge AI Fundamentals | 🟡 | 20 |
| 18 | Model Compression & Optimization | 🔴 | 25 |
| 19 | Edge Deployment Architecture | 🔴 | 25 |

#### Part 7: AI Security Architecture

| Ch | Title | Level | Pages |
|----|-------|-------|-------|
| 20 | AI Security Threat Model | 🟢 | 20 |
| 21 | AI Security Defense Architecture | 🟡 | 25 |
| 22 | Privacy-Preserving AI Architecture | 🟡 | 25 |

#### Part 8: Practice & Synthesis

| Ch | Title | Level | Pages |
|----|-------|-------|-------|
| 23 | AI Architecture Design Methodology | 🟢 | 20 |
| 24 | Comprehensive Case Studies | 🟢 | 40 |
| 25 | The Future of AI Architecture | ⚫ | 15 |

#### Appendices

| App | Title |
|-----|-------|
| A | Glossary (Bilingual) |
| B | Tool Selection Guide |
| C | Architecture Design Templates |
| D | References |
| E | Open Source Project Index |

### How to Use This Book

**🟢 Beginner path** — Start with Ch 1–2, then Ch 3, 6, 20, 23, 24. Skip advanced sections marked 🔴.

**🟡 Intermediate path** — Read Part 1–3 fully, then pick Part 4–7 based on your focus area. Ch 12 (RAG) and Ch 24 (Case Studies) are high-value.

**🔴 Advanced path** — Read the full book. Pay special attention to Parts 4–6 (LLM, Cloud-Native, Edge) for production architecture patterns.

**⚫ Manager path** — Focus on Ch 1–2 (role definition), Ch 6 (MLOps maturity), Ch 23 (methodology), Ch 24 (case studies), and Ch 25 (future trends).

### Project Structure

```
AI-Architect-Textbook/
├── README.md                  # This file
├── BLUEPRINT.md               # Book blueprint and planning
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # CC BY-NC-SA 4.0
├── gen_chapters.py            # Chapter generation script
├── generate_all.py            # Full generation pipeline
├── scripts/
│   └── combine.py             # Section combiner
├── src/
│   ├── zh/                    # Chinese source
│   │   ├── chapters/
│   │   │   ├── chapter-01.md
│   │   │   ├── chapter-02.md
│   │   │   └── ... (chapters 01-25)
│   │   └── appendices.md
│   └── en/                    # English source
│       ├── chapters/
│       │   ├── chapter-01.md
│       │   ├── chapter-02.md
│       │   └── ... (chapters 01-25)
│       └── appendices.md
└── sections/
    ├── zh/                    # Chinese sections (granular)
    └── en/                    # English sections (granular)
```

### Code Examples

| Language | Usage | Share |
|----------|-------|-------|
| Python | ML/DL core, training scripts, data processing | 60% |
| Go | Microservices, API gateways, high-perf components | 15% |
| Java | Enterprise integration, big data, monitoring | 15% |
| Shell/Bash | Deployment scripts, automation | 10% |

### Open Source Projects Referenced

| Chapter | Projects |
|---------|----------|
| Ch 3 | Apache Kafka, Apache Airflow |
| Ch 4 | Feast |
| Ch 5 | Delta Lake |
| Ch 7 | Kubeflow |
| Ch 8 | Seldon Core |
| Ch 9 | Prometheus, Grafana |
| Ch 10 | DeepSpeed |
| Ch 11 | vLLM |
| Ch 12 | LangChain, Chroma |
| Ch 13 | Unsloth |
| Ch 14 | Kubeflow, Volcano |
| Ch 15 | Ray |
| Ch 18 | ONNX Runtime |
| Ch 19 | NVIDIA Jetson |

### Build Instructions

#### Markdown Version (GitHub)

The Markdown version is the primary format. No build step required — just read the `.md` files directly.

```bash
# Clone the repository
git clone https://github.com/your-org/ai-architect-textbook.git
cd ai-architect-textbook

# Combine sections into full chapters (if needed)
python scripts/combine.py

# Read individual chapters
cat src/en/chapters/chapter-01.md
```

#### LaTeX Version (Print)

```bash
# Prerequisites: LaTeX distribution (TeX Live / MiKTeX)

# 1. Generate combined source files
python generate_all.py

# 2. Compile to PDF
cd build/latex
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex

# Output: build/latex/main.pdf (~450 pages)
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Chapter writing standards
- Code example requirements
- Review process
- Bilingual contribution workflow

### License

- **Content** (text, diagrams, exercises): [CC BY-NC-SA 4.0](LICENSE)
- **Code examples**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Chinese Version

### 概览

本书是一本面向 AI 架构师的实战指南，涵盖 AI 系统的完整生命周期。内容从数据管道到大模型部署，从云原生平台到边缘推理，跨越学术 ML 知识与生产工程实践之间的鸿沟，提供可落地的架构模式和开源工具实战案例。

**约 450 页**，包含 25 章 + 5 个附录，配有基于真实开源项目的实战案例。

### 目标读者

| 读者类型 | 描述 |
|----------|------|
| **入门开发者** | AI/ML 系统设计新手，需要基础知识 |
| **中级工程师** | 有 ML 经验，准备设计生产系统 |
| **高级架构师** | 主导 AI 平台设计和技术战略 |
| **技术管理者** | 做出 build-vs-buy 决策，评估 AI 投资 |

### 读者分级标记

| 标记 | 级别 | 说明 |
|------|------|------|
| 🟢 | **入门** | 基础内容——所有读者必读 |
| 🟡 | **中级** | 需要基础 AI/ML 知识 |
| 🔴 | **高级** | 需要架构设计经验 |
| ⚫ | **管理者** | 侧重决策和管理视角 |
| 📝 | **练习** | 动手实践环节 |
| 💡 | **案例** | 真实项目案例 |
| ⚠️ | **注意** | 常见陷阱和误区 |
| 📌 | **核心概念** | 必须掌握的知识点 |

### 目录

#### 第一部分：AI 架构师角色与基础

| 章 | 标题 | 级别 | 页数 |
|----|------|------|------|
| 1 | AI 架构师的角色定义 | 🟢 | 15 |
| 2 | AI 系统设计原则 | 🟢 | 20 |

#### 第二部分：数据架构

| 章 | 标题 | 级别 | 页数 |
|----|------|------|------|
| 3 | 数据管道架构 | 🟢 | 30 |
| 4 | 特征工程架构 | 🟡 | 25 |
| 5 | 数据湖仓架构 | 🟡 | 25 |

#### 第三部分：MLOps 架构

| 章 | 标题 | 级别 | 页数 |
|----|------|------|------|
| 6 | MLOps 基础与成熟度模型 | 🟢 | 20 |
| 7 | 模型训练架构 | 🟡 | 30 |
| 8 | 模型部署架构 | 🟡 | 30 |
| 9 | 模型监控与可观测性 | 🟡 | 25 |

#### 第四部分：大模型架构

| 章 | 标题 | 级别 | 页数 |
|----|------|------|------|
| 10 | LLM 架构设计基础 | 🟡 | 30 |
| 11 | LLM 推理架构 | 🔴 | 30 |
| 12 | RAG 系统架构 | 🟡 | 30 |
| 13 | 模型微调架构 | 🔴 | 25 |

#### 第五部分：云原生 AI 架构

| 章 | 标题 | 级别 | 页数 |
|----|------|------|------|
| 14 | Kubernetes 上的 AI 工作负载 | 🔴 | 30 |
| 15 | 分布式计算架构 | 🔴 | 25 |
| 16 | AI 平台架构 | 🔴 | 25 |

#### 第六部分：边缘 AI 架构

| 章 | 标题 | 级别 | 页数 |
|----|------|------|------|
| 17 | 边缘 AI 基础 | 🟡 | 20 |
| 18 | 模型压缩与优化 | 🔴 | 25 |
| 19 | 边缘部署架构 | 🔴 | 25 |

#### 第七部分：AI 安全架构

| 章 | 标题 | 级别 | 页数 |
|----|------|------|------|
| 20 | AI 安全威胁模型 | 🟢 | 20 |
| 21 | AI 安全防御架构 | 🟡 | 25 |
| 22 | 隐私保护 AI 架构 | 🟡 | 25 |

#### 第八部分：实战与综合

| 章 | 标题 | 级别 | 页数 |
|----|------|------|------|
| 23 | AI 架构设计方法论 | 🟢 | 20 |
| 24 | 综合实战项目 | 🟢 | 40 |
| 25 | AI 架构师的未来 | ⚫ | 15 |

#### 附录

| 附录 | 标题 |
|------|------|
| A | 术语表（中英对照） |
| B | 工具选型指南 |
| C | 架构设计模板 |
| D | 参考文献 |
| E | 开源项目索引 |

### 如何使用本书

**🟢 入门路线** — 从第 1–2 章开始，然后读第 3、6、20、23、24 章。跳过标记为 🔴 的高级内容。

**🟡 中级路线** — 完整阅读第一至三部分，然后根据需要选读第四至七部分。第 12 章（RAG）和第 24 章（案例）价值最高。

**🔴 高级路线** — 完整阅读。重点关注第四至六部分（大模型、云原生、边缘）的生产架构模式。

**⚫ 管理者路线** — 重点关注第 1–2 章（角色定义）、第 6 章（MLOps 成熟度）、第 23 章（方法论）、第 24 章（案例）和第 25 章（趋势）。

### 项目结构

```
AI-Architect-Textbook/
├── README.md                  # 本文件
├── BLUEPRINT.md               # 书籍蓝图与规划
├── CONTRIBUTING.md            # 贡献指南
├── LICENSE                    # CC BY-NC-SA 4.0
├── gen_chapters.py            # 章节生成脚本
├── generate_all.py            # 完整生成流水线
├── scripts/
│   └── combine.py             # 章节合并脚本
├── src/
│   ├── zh/                    # 中文源文件
│   │   ├── chapters/
│   │   │   ├── chapter-01.md
│   │   │   └── ... (第 01-25 章)
│   │   └── appendices.md
│   └── en/                    # 英文源文件
│       ├── chapters/
│       │   ├── chapter-01.md
│       │   └── ... (第 01-25 章)
│       └── appendices.md
└── sections/
    ├── zh/                    # 中文细粒度章节
    └── en/                    # 英文细粒度章节
```

### 代码示例语言分布

| 语言 | 用途 | 占比 |
|------|------|------|
| Python | ML/DL 核心、训练脚本、数据处理 | 60% |
| Go | 微服务、API 网关、高性能组件 | 15% |
| Java | 企业级集成、大数据组件、监控 | 15% |
| Shell/Bash | 部署脚本、自动化流程 | 10% |

### 构建说明

#### Markdown 版本（GitHub）

Markdown 版本是主要格式，无需构建步骤，直接阅读 `.md` 文件即可。

```bash
# 克隆仓库
git clone https://github.com/your-org/ai-architect-textbook.git
cd ai-architect-textbook

# 合并细粒度章节为完整章节（如需要）
python scripts/combine.py

# 阅读单章
cat src/zh/chapters/chapter-01.md
```

#### LaTeX 版本（纸质出版）

```bash
# 前提：安装 LaTeX 发行版（TeX Live / MiKTeX）

# 1. 生成合并源文件
python generate_all.py

# 2. 编译为 PDF
cd build/latex
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex

# 输出：build/latex/main.pdf（约 450 页）
```

### 贡献

详见 [CONTRIBUTING.md](CONTRIBUTING.md)，包含：
- 章节写作规范
- 代码示例要求
- 审稿流程
- 双语贡献工作流

### 许可证

- **内容**（文本、图表、练习）：[CC BY-NC-SA 4.0](LICENSE)
- **代码示例**：[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
