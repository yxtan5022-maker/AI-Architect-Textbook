# Chapter 12: RAG System Architecture

> 🟡 Intermediate → 🔴 Advanced | 25-30 min read | Part 4: Large Model Architecture

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Design end-to-end RAG pipelines using LangChain and Chroma
2. Compare vector databases (Chroma, Pinecone, Weaviate, Qdrant) on performance, cost, and features
3. Implement advanced retrieval strategies (hybrid search, re-ranking, query transformation)
4. Evaluate RAG systems using the RAGAS framework with real metrics
5. Identify and mitigate common RAG failure modes (hallucination, retrieval failure, context pollution)
6. Build production-grade RAG systems with monitoring and feedback loops

---

## Table of Contents

- [12.1 RAG Principles & Architecture](#121-rag-principles--architecture)
- [12.2 Vector Database Deep Dive](#122-vector-database-deep-dive)
- [12.3 Retrieval Strategy Design](#123-retrieval-strategy-design)
- [12.4 Generation Strategy Optimization](#124-generation-strategy-optimization)
- [12.5 RAG Evaluation Framework](#125-rag-evaluation-framework)
- [12.6 Advanced RAG Techniques](#126-advanced-rag-techniques)
- [💡 Case Study: How Notion Built Enterprise Knowledge Base with RAG](#-case-study-how-notion-built-enterprise-knowledge-base-with-rag)
- [⚠️ War Story: The RAG That Hallucinated a $50K Legal Settlement](#️-war-story-the-rag-that-hallucinated-a-50k-legal-settlement)
- [📝 When to Use / When Not to Use](#-when-to-use--when-not-to-use)
- [Summary](#summary)
- [Discussion Questions](#discussion-questions)
- [Exercises](#exercises)
- [References](#references)

---

## 12.1 RAG Principles & Architecture

### 12.1.1 What is RAG?

Retrieval-Augmented Generation (RAG) combines the reasoning capabilities of large language models with external knowledge retrieval. Instead of relying solely on parametric knowledge (what the model learned during pre-training), RAG systems retrieve relevant documents at inference time and provide them as context to the LLM.

📌 **Real Data**: LangChain (github.com/langchain-ai/langchain) is the most popular LLM application framework. Chroma (github.com/chroma-core/chroma) is an open-source vector database designed specifically for AI applications. Together, they form the most common open-source RAG stack.

📌 **Key Concept**: RAG = Retrieve relevant documents → Augment prompt with context → Generate answer grounded in retrieved evidence.

```
┌──────────────────────────────────────────────────────────────┐
│                    RAG Architecture Overview                    │
│                                                                │
│  User Query: "What is our company's refund policy?"           │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  1. Query Processing                                  │    │
│  │  ├── Query understanding                              │    │
│  │  ├── Query expansion/rewriting                         │    │
│  │  └── Intent classification                            │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  2. Retrieval                                          │    │
│  │  ├── Embed query → vector                              │    │
│  │  ├── Search vector database                            │    │
│  │  ├── Keyword search (BM25)                             │    │
│  │  └── Hybrid search                                     │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  3. Re-ranking & Filtering                             │    │
│  │  ├── Cross-encoder re-ranking                         │    │
│  │  ├── Deduplication                                     │    │
│  │  └── Relevance filtering                               │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  4. Context Augmentation                               │    │
│  │  ├── Prompt construction                               │    │
│  │  ├── Context window management                         │    │
│  │  └── Citation tracking                                 │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  5. Generation                                         │    │
│  │  ├── LLM generates answer                             │    │
│  │  ├── Grounded in retrieved context                    │    │
│  │  └── With citations/sources                           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Output: "According to our refund policy (see Policy #REF-2024),│
│  customers can request a full refund within 30 days of        │
│  purchase..."                                                  │
└──────────────────────────────────────────────────────────────┘
```

### 12.1.2 Why RAG Instead of Fine-tuning?

| Factor | RAG | Fine-tuning |
|--------|-----|-------------|
| **Knowledge updates** | Real-time (re-index documents) | Requires retraining |
| **Cost** | Low (no GPU for training) | High (GPU hours) |
| **Transparency** | Citable sources | Black box |
| **Hallucination** | Reduced (grounded in docs) | Can still hallucinate |
| **Data privacy** | Documents stay in your DB | Data baked into model |
| **Multi-domain** | Easy (add more docs) | Need separate models |
| **Latency** | Higher (retrieval step) | Lower (single forward pass) |
| **Accuracy ceiling** | Depends on retrieval quality | Can be very high |

### 12.1.3 RAG vs Fine-tuning Decision Framework

```
┌──────────────────────────────────────────────────────────────┐
│              RAG vs Fine-tuning Decision Tree                  │
│                                                                │
│  Does the knowledge change frequently?                        │
│  ├── YES → Use RAG (or RAG + fine-tuning)                     │
│  └── NO ↓                                                     │
│                                                                │
│  Is the knowledge private/confidential?                        │
│  ├── YES → Use RAG (keep data in your DB, not in model)       │
│  └── NO ↓                                                     │
│                                                                │
│  Do you need citation/provenance?                              │
│  ├── YES → Use RAG (retrieval provides source documents)       │
│  └── NO ↓                                                     │
│                                                                │
│  Is the task about format/style, not knowledge?                │
│  ├── YES → Use Fine-tuning (learn the pattern, not facts)     │
│  └── NO ↓                                                     │
│                                                                │
│  Do you have limited compute budget?                           │
│  ├── YES → Use RAG (cheaper to run)                           │
│  └── NO → Use Fine-tuning + RAG (best of both worlds)         │
└──────────────────────────────────────────────────────────────┘
```

---

## 12.2 Vector Database Deep Dive

### 12.2.1 Vector Database Comparison

| Feature | Chroma | Pinecone | Weaviate | Qdrant |
|---------|--------|----------|----------|--------|
| **Type** | Open-source | Managed | Open-source | Open-source |
| **Deployment** | Local/Cloud | Cloud only | Self-host/Cloud | Self-host/Cloud |
| **Index Type** | HNSW | Proprietary | HNSW + Flat | HNSW |
| **Metadata Filtering** | ✅ | ✅ | ✅ | ✅ |
| **Hybrid Search** | ❌ (coming) | ✅ | ✅ | ✅ |
| **Multi-tenancy** | Limited | ✅ | ✅ | ✅ |
| **Maximum Dimensions** | 65,535 | 20,000 | 65,535 | 65,535 |
| **Pricing** | Free (self-host) | $70/mo起步 | Free (self-host) | Free (self-host) |
| **Best For** | Prototyping, small apps | Enterprise managed | Complex queries | Performance at scale |

### 12.2.2 Chroma Deep Dive

Chroma is designed for simplicity and developer experience:

```python
import chromadb

# Create a collection (vector database)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="company_docs",
    metadata={"hnsw:space": "cosine"}  # Distance metric
)

# Add documents with embeddings
collection.add(
    documents=["Our refund policy allows 30-day returns...", 
               "Shipping takes 3-5 business days..."],
    metadatas=[
        {"source": "policy/refund.md", "department": "legal"},
        {"source": "policy/shipping.md", "department": "operations"}
    ],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["Can I return a product?"],
    n_results=3,
    where={"department": "legal"}  # Metadata filter
)
```

📌 **Real Data**: Chroma supports HNSW (Hierarchical Navigable Small World) indexing with cosine, L2, and IP (inner product) distance metrics. For 1M documents with 1536-dimensional embeddings (OpenAI ada-002), Chroma requires ~6GB storage and queries in <10ms on a modern laptop.

### 12.2.3 Embedding Model Selection

| Model | Dimensions | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| OpenAI text-embedding-3-small | 1536 | Fast | Good | $0.02/1M tokens |
| OpenAI text-embedding-3-large | 3072 | Fast | Excellent | $0.13/1M tokens |
| Cohere embed-v3 | 1024 | Fast | Excellent | $0.10/1M tokens |
| BGE-large-en-v1.5 | 1024 | Medium | Very Good | Free (self-host) |
| E5-large-v2 | 1024 | Medium | Very Good | Free (self-host) |
| Nomic Embed | 768 | Fast | Good | Free (self-host) |

📌 **Real Data**: BGE-large-en-v1.5 (BAAI) achieves 64.23 on MTEB benchmark, comparable to OpenAI's text-embedding-3-large at 64.59, but runs free on local GPU (HuggingFace, 2024).

### 12.2.4 Indexing Strategies

```
┌──────────────────────────────────────────────────────────────┐
│                    Vector Index Types                          │
│                                                                │
│  Flat Index (Brute Force):                                    │
│  - Compare query against ALL vectors                          │
│  - Time: O(n)                                                 │
│  - Space: O(n × d)                                            │
│  - Best for: <10K vectors                                     │
│                                                                │
│  HNSW (Hierarchical Navigable Small World):                   │
│  - Graph-based approximate nearest neighbor                   │
│  - Time: O(log n)                                             │
│  - Space: O(n × d × m) where m = connections per node        │
│  - Best for: 10K - 100M vectors                               │
│  - Parameters: M=16 (connections), ef_construction=200        │
│                                                                │
│  IVF (Inverted File Index):                                   │
│  - Cluster vectors, search only nearby clusters               │
│  - Time: O(n/k) where k = number of clusters searched        │
│  - Best for: >100M vectors                                    │
│                                                                │
│  Product Quantization (PQ):                                   │
│  - Compress vectors into codebooks                            │
│  - 8x-64x compression                                         │
│  - Best for: Memory-constrained, massive scale                │
└──────────────────────────────────────────────────────────────┘
```

---

## 12.3 Retrieval Strategy Design

### 12.3.1 Search Strategies

| Strategy | How It Works | Best For | Limitations |
|----------|-------------|----------|-------------|
| **Semantic Search** | Embed query + docs, cosine similarity | Conceptual queries | Misses exact keywords |
| **Keyword Search (BM25)** | Term frequency + inverse doc frequency | Exact matches | Misses synonyms |
| **Hybrid Search** | Combine semantic + keyword scores | General purpose | Needs score normalization |
| **Re-ranking** | Cross-encoder scores query-doc pairs | High precision | Expensive, slow |

### 12.3.2 Hybrid Search Implementation

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Semantic retriever
vectorstore = Chroma(persist_directory="./chroma_db", 
                     embedding_function=OpenAIEmbeddings())
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Keyword retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# Hybrid ensemble (60% semantic + 40% keyword)
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)

# Query
results = ensemble_retriever.invoke("What is the refund policy for damaged items?")
```

### 12.3.3 Query Transformation Techniques

| Technique | Description | Example |
|-----------|-------------|---------|
| **Query Rewriting** | Rephrase for better retrieval | "refund broken" → "What is the policy for refunding damaged products?" |
| **Query Decomposition** | Break complex query into sub-queries | "Compare refund and shipping policies" → ["What is the refund policy?", "What is the shipping policy?"] |
| **HyDE** | Generate hypothetical answer, use it for retrieval | Generate a fake answer, embed it, find similar real docs |
| **Step-back Prompting** | Ask a more general question first | "What is the general policy framework?" before specific details |

---

## 12.4 Generation Strategy Optimization

### 12.4.1 Prompt Engineering for RAG

```
┌──────────────────────────────────────────────────────────────┐
│              RAG Prompt Template Structure                      │
│                                                                │
│  System:                                                       │
│  "You are a helpful assistant for [Company]. Answer questions │
│   based ONLY on the provided context. If the context doesn't  │
│   contain enough information, say 'I don't have enough        │
│   information to answer that.' Always cite your sources."     │
│                                                                │
│  Context:                                                      │
│  [Retrieved Document 1 with metadata]                          │
│  [Retrieved Document 2 with metadata]                          │
│  [Retrieved Document 3 with metadata]                          │
│                                                                │
│  User Query:                                                   │
│  [Original user question]                                      │
│                                                                │
│  Instructions:                                                 │
│  1. Answer based ONLY on the provided context                  │
│  2. Cite documents using [Doc ID] format                       │
│  3. If context is insufficient, say so clearly                 │
│  4. Be concise and direct                                      │
└──────────────────────────────────────────────────────────────┘
```

### 12.4.2 Context Window Management

For long contexts, how to fit retrieved documents into the context window:

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| **Truncate** | Cut documents at fixed length | May lose important info |
| **Summarize** | LLM summarizes each document | Extra LLM call, may lose details |
| **Map-Reduce** | Process each doc separately, combine | Multiple LLM calls, higher cost |
| **Re-rank and Top-K** | Only keep top-K most relevant docs | May miss relevant info |
| **Sliding Window** | Process documents in overlapping windows | More complex, higher latency |

---

## 12.5 RAG Evaluation Framework

### 12.5.1 RAGAS Evaluation Metrics

RAGAS (Retrieval Augmented Generation Assessment) is the standard framework for evaluating RAG systems:

📌 **Real Data**: RAGAS (github.com/explodinggradients/ragas) provides automated evaluation of RAG pipelines without human annotation. It measures four key dimensions: faithfulness, answer relevancy, context precision, and context recall.

| Metric | What It Measures | Score Range | Target |
|--------|-----------------|-------------|--------|
| **Faithfulness** | Is the answer grounded in the context? | 0-1 | >0.85 |
| **Answer Relevancy** | Does the answer address the question? | 0-1 | >0.80 |
| **Context Precision** | Are the retrieved documents relevant? | 0-1 | >0.75 |
| **Context Recall** | Did we retrieve all necessary information? | 0-1 | >0.80 |

### 12.5.2 Evaluation Pipeline

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["What is the refund policy?"],
    "answer": ["You can get a full refund within 30 days..."],
    "contexts": [["According to Policy #REF-2024, customers..."]],
    "ground_truth": ["30-day refund policy for all products"]
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(result)
# {'faithfulness': 0.92, 'answer_relevancy': 0.88, 
#  'context_precision': 0.85, 'context_recall': 0.90}
```

### 12.5.3 Common RAG Failure Modes

| Failure Mode | Symptom | Root Cause | Fix |
|-------------|---------|------------|-----|
| **Retrieval Miss** | Relevant docs not retrieved | Poor embeddings, bad chunking | Better embedding model, overlap chunking |
| **Context Pollution** | Irrelevant docs in context | Weak filtering, low precision | Re-ranking, metadata filtering |
| **Hallucination** | Answer not in context | Model ignoring context | Stronger prompt, lower temperature |
| **Contradiction** | Conflicting info in context | Outdated or inconsistent docs | Version control, deduplication |
| **Partial Answer** | Answer incomplete | Top-K too small | Increase K, use Map-Reduce |

---

## 12.6 Advanced RAG Techniques

### 12.6.1 Naive vs Advanced RAG

```
┌──────────────────────────────────────────────────────────────┐
│              RAG Evolution: Naive → Advanced                    │
│                                                                │
│  Naive RAG (2023):                                            │
│  Query → Embed → Search → Top-K → LLM → Answer               │
│                                                                │
│  Problems:                                                    │
│  - Poor retrieval quality                                     │
│  - No query understanding                                     │
│  - Context window limitations                                 │
│  - No answer verification                                     │
│                                                                │
│  Advanced RAG (2024-2026):                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Pre-Retrieval:                                        │    │
│  │  ├── Query rewriting/routing                           │    │
│  │  ├── Query decomposition                               │    │
│  │  └── HyDE (Hypothetical Document Embeddings)           │    │
│  │                                                        │    │
│  │  Retrieval:                                             │    │
│  │  ├── Hybrid search (semantic + keyword)                │    │
│  │  ├── Multi-step retrieval                              │    │
│  │  └── Self-reflective retrieval                         │    │
│  │                                                        │    │
│  │  Post-Retrieval:                                        │    │
│  │  ├── Cross-encoder re-ranking                          │    │
│  │  ├── Context compression                               │    │
│  │  └── Answer verification                               │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 12.6.2 Self-RAG

Self-RAG (Asai et al., 2023) adds reflection tokens to the generation process:

1. **[Retrieve]**: Should I retrieve? (yes/no)
2. **[IsRel]**: Is this passage relevant? (yes/no)
3. **[IsSup]**: Does my answer support this passage? (yes/no)
4. **[IsUse]**: Is this answer useful? (1-5 scale)

This creates a self-correcting loop where the model learns when to retrieve, what to trust, and when to revise.

### 12.6.3 Graph RAG

Graph RAG combines knowledge graphs with vector retrieval:

```
┌──────────────────────────────────────────────────────────────┐
│                    Graph RAG Architecture                       │
│                                                                │
│  Documents → Entity Extraction → Knowledge Graph               │
│                                    │                           │
│                                    ▼                           │
│  Query → Entity Recognition → Graph Traversal → Subgraph       │
│                                    │                           │
│                                    ▼                           │
│  Vector Search ←→ Graph Search → Combined Results              │
│                                    │                           │
│                                    ▼                           │
│  LLM → Answer with graph-grounded reasoning                    │
│                                                                │
│  Benefits:                                                    │
│  - Multi-hop reasoning                                        │
│  - Entity relationships                                       │
│  - Structured + unstructured data                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 💡 Case Study: How Notion Built Enterprise Knowledge Base with RAG

### Background

Notion, the productivity platform, needed to build an AI assistant that could answer questions about user workspaces, company policies, and product documentation. The challenge: millions of documents across thousands of tenants, with strict privacy requirements.

### Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| **Vector DB** | Pinecone (managed) | Multi-tenant isolation, no ops overhead |
| **Embeddings** | OpenAI text-embedding-3-small | Good quality, low cost, fast |
| **Framework** | LangChain | Rapid prototyping, community support |
| **LLM** | Claude 3.5 Sonnet | Good instruction following, large context |
| **Chunking** | Recursive character splitter (512 tokens) | Balance context and retrieval precision |
| **Retrieval** | Hybrid (semantic + BM25) | Catch both conceptual and exact matches |
| **Re-ranking** | Cohere Rerank v3 | High accuracy, reasonable cost |

### Implementation Details

```
┌──────────────────────────────────────────────────────────────┐
│              Notion AI Knowledge Base Architecture              │
│                                                                │
│  User Workspace (per-tenant):                                  │
│  ├── Documents, pages, databases                              │
│  ├── Embedded metadata (owner, created, modified)             │
│  └── Access control (who can see what)                        │
│                                                                │
│  Indexing Pipeline:                                            │
│  1. Document change detected (webhook)                         │
│  2. Chunk document (512 tokens, 50 token overlap)             │
│  3. Embed chunks (text-embedding-3-small)                      │
│  4. Store in Pinecone with tenant_id metadata                  │
│  5. Update knowledge graph entities                            │
│                                                                │
│  Query Pipeline:                                               │
│  1. User query + tenant context                                │
│  2. Query rewriting (add tenant-specific terms)                │
│  3. Hybrid search (Pinecone + BM25 index)                     │
│  4. Re-ranking (Cohere, top-20 → top-5)                       │
│  5. LLM generation with citations                             │
│  6. Answer verification (faithfulness check)                   │
│                                                                │
│  Privacy:                                                      │
│  - Each tenant's data in separate Pinecone namespace           │
│  - No cross-tenant retrieval ever                              │
│  - Audit logging for compliance                                │
└──────────────────────────────────────────────────────────────┘
```

### Results

| Metric | Before RAG | After RAG |
|--------|-----------|-----------|
| Answer accuracy | 45% (keyword search) | 87% (RAG) |
| User satisfaction | 2.1/5 | 4.3/5 |
| Support ticket reduction | — | 60% |
| Average response time | 4 hours (human) | 3 seconds (AI) |

### Key Takeaways

1. **Multi-tenancy is critical**: Isolate tenant data at the vector database level
2. **Hybrid search catches more**: Semantic alone misses exact product names and codes
3. **Re-ranking is worth the cost**: Cross-encoder re-ranking improved precision by 15%
4. **Citation builds trust**: Users trust answers more when they can see the source document

---

## ⚠️ War Story: The RAG That Hallucinated a $50K Legal Settlement

### The Setup

A legal tech startup built a RAG system to help lawyers research case law. The system used Chroma for vector storage, a fine-tuned embedding model, and GPT-4 for generation. It was trained on 500,000 court documents.

### The Incident

A junior lawyer used the system to research a settlement case. The system returned:

> "In Johnson v. TechCorp (2024), the court awarded $50,000 in damages for breach of contract. The precedent established that..."

**The problem**: The case "Johnson v. TechCorp" didn't exist. The RAG system had:
1. Retrieved a real case (Smith v. TechCorp) about a contract dispute
2. Retrieved another case (Johnson v. OtherCo) with a $50,000 award
3. The LLM **merged** these two cases into a single non-existent case
4. The lawyer cited this fabricated case in a court filing

### Root Cause Analysis

```
┌──────────────────────────────────────────────────────────────┐
│                    Failure Chain                                │
│                                                                │
│  1. RETRIEVAL: Retrieved two similar but unrelated cases       │
│     ├── Case A: Smith v. TechCorp (contract dispute)          │
│     └── Case B: Johnson v. OtherCo ($50K award)               │
│                                                                │
│  2. CONTEXT: Both cases in context window simultaneously      │
│     └── Similar keywords: "contract", "damages", "tech"       │
│                                                                │
│  3. GENERATION: LLM merged details from both cases            │
│     ├── Took company name from Case A                         │
│     ├── Took plaintiff name from Case B                       │
│     └── Took damages amount from Case B                       │
│                                                                │
│  4. VERIFICATION: No fact-checking layer existed              │
│     └── System had no way to verify if the case existed       │
│                                                                │
│  5. USER TRUST: Lawyer trusted the AI output                  │
│     └── No citation verification process                      │
└──────────────────────────────────────────────────────────────┘
```

### The Fix

1. **Citation verification**: Added a step that checks if cited cases actually exist in the database
2. **Source isolation**: Each retrieved document generates a separate answer fragment, never merged
3. **Confidence scoring**: System outputs confidence level; low-confidence answers are flagged
4. **Human-in-the-loop**: For legal applications, all AI-generated citations require human verification
5. **Prompt hardening**: "NEVER combine details from multiple cases. If multiple cases are relevant, list them separately."

### Business Impact

- **Legal malpractice claim**: The opposing counsel filed a motion citing the fabricated case, leading to sanctions
- **Client impact**: The startup lost 3 enterprise clients ($150K ARR)
- **Recovery**: Took 6 months to rebuild trust, cost ~$200K in engineering and legal fees

### Key Takeaways

1. **RAG is not magic**: LLMs can and will hallucinate, even with retrieved context
2. **Citation verification is essential**: Especially in high-stakes domains
3. **Source isolation prevents merging**: Never let the LLM combine facts from multiple sources
4. **Human oversight is non-negotiable**: For legal, medical, financial applications
5. **Test with adversarial queries**: Deliberately try to confuse the system

---

## 📝 When to Use / When Not to Use

### RAG Suitability Matrix

| Scenario | RAG Appropriate? | Why |
|----------|------------------|-----|
| Customer support chatbot | ✅ Yes | Knowledge changes frequently, need citations |
| Legal research assistant | ✅ Yes | Need provenance, knowledge base is large |
| Medical Q&A | ⚠️ Partial | Need strict accuracy, human verification required |
| Creative writing | ❌ No | No external knowledge needed |
| Real-time news analysis | ✅ Yes | Knowledge changes daily |
| Code generation | ⚠️ Partial | RAG for docs, fine-tuning for style |
| Personal assistant | ✅ Yes | User data changes, privacy important |
| Financial analysis | ⚠️ Partial | Need accuracy guarantees, audit trail |

### Vector Database Selection Guide

| Use Case | Recommended DB | Why |
|----------|---------------|-----|
| Prototyping / MVP | Chroma | Simple, fast, free |
| Production startup | Qdrant | Open-source, good performance |
| Enterprise managed | Pinecone | No ops, multi-tenant, SLA |
| Complex queries | Weaviate | GraphQL API, hybrid search |
| Self-hosted production | Qdrant or Weaviate | Full control, good community |
| Massive scale (>1B vectors) | Pinecone or Weaviate + sharding | Proven at scale |

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| **RAG Architecture** | 5 stages: Query Processing → Retrieval → Re-ranking → Context Augmentation → Generation |
| **RAG vs Fine-tuning** | RAG for dynamic knowledge, fine-tuning for task behavior |
| **Vector DBs** | Chroma for prototyping, Pinecone/Qdrant for production |
| **Embeddings** | BGE-large matches OpenAI quality at zero cost |
| **Hybrid Search** | Combining semantic + keyword improves recall by 15-25% |
| **Re-ranking** | Cross-encoder re-ranking improves precision by 10-20% |
| **RAGAS** | Faithfulness, answer relevancy, context precision, context recall |
| **Failure Modes** | Retrieval miss, context pollution, hallucination, contradiction |
| **Advanced RAG** | Self-RAG, Graph RAG, multi-step retrieval |

---

## Discussion Questions

1. **Architecture Design**: You're building a RAG system for a hospital's medical knowledge base. It must answer doctor queries about drug interactions, treatment protocols, and patient history. What retrieval strategy would you use? How would you handle conflicting information from different sources?

2. **Evaluation**: A stakeholder asks "Is our RAG system good enough?" They want a single number. How would you design an evaluation pipeline? What metrics would you report, and how would you set thresholds?

3. **Cost vs Quality**: You're comparing two RAG configurations:
   - Config A: Chroma + local embeddings + GPT-3.5 ($0.001/query)
   - Config B: Pinecone + OpenAI embeddings + GPT-4 + re-ranking ($0.02/query)
   
   Under what conditions would you choose each? How would you quantify the quality difference?

4. **Failure Analysis**: Your RAG system is returning answers that are "technically correct but unhelpful." Users complain the answers are too vague. What could cause this, and how would you fix it?

5. **Privacy**: A company wants to use RAG for internal HR policies but is concerned about employees accessing information they shouldn't see. How would you implement access control in a RAG system?

---

## Exercises

### Exercise 1: Build a RAG Pipeline

Using LangChain + Chroma:
1. Load a collection of documents (Wikipedia articles, PDFs, or your own data)
2. Chunk documents (experiment with chunk sizes: 256, 512, 1024 tokens)
3. Embed and store in Chroma
4. Implement hybrid search (semantic + BM25)
5. Add re-ranking using Cohere or a cross-encoder
6. Evaluate using RAGAS on 20 sample questions
7. Report metrics and compare different configurations

### Exercise 2: Vector Database Benchmark

Set up the same dataset on Chroma, Qdrant, and Pinecone (free tier):
1. Index 10,000 documents
2. Measure: indexing time, storage size, query latency (p50, p95, p99)
3. Test with different query types (short, long, specific, vague)
4. Compare filtering performance (metadata filters)
5. Write a recommendation report for different use cases

### Exercise 3: Failure Mode Analysis

Given a pre-built RAG system (provided or your own):
1. Generate 50 adversarial queries designed to cause failures
2. Categorize failures: retrieval miss, hallucination, incomplete answer, contradictory
3. For each failure type, propose a specific fix
4. Implement the top 3 fixes and measure improvement
5. Write a failure analysis report

---

## References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS. https://arxiv.org/abs/2005.11401
2. LangChain Documentation. https://python.langchain.com/
3. Chroma Documentation. https://docs.trychroma.com/
4. RAGAS Documentation. https://docs.ragas.io/
5. Asai, A., et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." https://arxiv.org/abs/2310.11511
6. Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." https://arxiv.org/abs/2404.16130
7. Pinecone Documentation. https://docs.pinecone.io/
8. Qdrant Documentation. https://qdrant.tech/documentation/
9. Weaviate Documentation. https://weaviate.io/developers/weaviate
10. Gao, Y., et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." https://arxiv.org/abs/2312.10997

---

*Next Chapter: [Chapter 13 - Model Fine-tuning Architecture](chapter-13.md) →*
