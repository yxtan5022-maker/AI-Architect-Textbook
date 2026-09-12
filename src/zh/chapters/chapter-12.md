# 第12章：RAG 系统架构

> 🟡 中级 → 🔴 高级 | 阅读时长 25-30分钟 | 第四部分：大模型架构

---

## 目录

- [12.1 RAG 原理与架构](#121-rag-原理与架构)
- [12.2 向量数据库选型](#122-向量数据库选型)
- [12.3 检索策略设计](#123-检索策略设计)
- [12.4 生成策略优化](#124-生成策略优化)
- [12.5 RAG 评估框架](#125-rag-评估框架)
- [12.6 高级 RAG 技术](#126-高级-rag-技术)
- [💡 案例：基于 LangChain + Chroma 的企业知识库](#-案例基于-langchain--chroma-的企业知识库)
- [本章小结](#本章小结)
- [参考文献](#参考文献)

---

## 12.1 RAG 原理与架构

### 12.1.1 什么是 RAG？

检索增强生成（Retrieval-Augmented Generation, RAG）将大语言模型的推理能力与外部知识检索相结合。RAG 系统不完全依赖参数化知识（模型在预训练期间学到的内容），而是在推理时检索相关文档，并将其作为上下文提供给 LLM。

📌 **核心概念**：RAG = 检索相关文档 → 用上下文增强提示 → 基于检索证据生成答案。

```
┌──────────────────────────────────────────────────────────────┐
│                    RAG 架构概览                                 │
│                                                                │
│  用户查询: "我们公司的退货政策是什么？"                          │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  1. 查询处理                                          │    │
│  │  ├── 查询理解                                         │    │
│  │  ├── 查询扩展/重写                                    │    │
│  │  └── 意图分类                                        │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  2. 检索                                              │    │
│  │  ├── 嵌入查询 → 向量                                  │    │
│  │  ├── 搜索向量数据库                                    │    │
│  │  ├── 关键词搜索（BM25）                                │    │
│  │  └── 混合搜索                                        │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  3. 重排序与过滤                                       │    │
│  │  ├── 交叉编码器重排序                                  │    │
│  │  ├── 去重                                             │    │
│  │  └── 相关性过滤                                        │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  4. 上下文增强                                         │    │
│  │  ├── 提示构建                                        │    │
│  │  ├── 上下文窗口管理                                    │    │
│  │  └── 引用追踪                                        │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  5. 生成                                              │    │
│  │  ├── LLM 生成答案                                    │    │
│  │  ├── 基于检索上下文                                    │    │
│  │  └── 附带引用/来源                                    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  输出: "根据我们的退货政策（参见政策 #REF-2024），              │
│  客户可在购买后 30 天内申请全额退款..."                          │
└──────────────────────────────────────────────────────────────┘
```

### 12.1.2 为什么用 RAG 而非微调？

| 方面 | RAG | 微调 |
|------|-----|------|
| **知识更新** | 实时（更新文档） | 需要重训练 |
| **来源引用** | ✅ 内置引用 | ❌ 无来源追踪 |
| **幻觉** | 减少（基于文档） | 可能幻觉 |
| **成本** | 较低（无需重训练） | 较高（计算 + 数据） |
| **数据隐私** | 文档保持外部 | 数据嵌入模型 |
| **复杂度** | 需要基础设施 | 流水线更简单 |
| **多任务** | 单模型 + 不同文档 | 每任务需要模型 |

```
┌──────────────────────────────────────────────────────────────┐
│              何时使用 RAG vs 微调                               │
│                                                                │
│  使用 RAG 当：                                                  │
│  ├── 知识频繁变化                                              │
│  ├── 需要来源引用                                              │
│  ├── 数据太大无法放入模型                                      │
│  ├── 需要实时信息                                              │
│  └── 一个模型覆盖多个领域                                      │
│                                                                │
│  使用微调当：                                                   │
│  ├── 任务需要特定行为/风格                                     │
│  ├── 知识稳定且较小                                           │
│  ├── 需要极低延迟                                              │
│  └── RAG 上下文 无法捕获所需模式                               │
│                                                                │
│  最佳实践：两者结合！                                           │
│  微调行为 + RAG 知识                                           │
└──────────────────────────────────────────────────────────────┘
```

### 12.1.3 RAG 系统架构模式

```python
# 基础 RAG 流水线
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

class BasicRAGPipeline:
    """演示核心概念的基础 RAG 流水线。"""

    def __init__(self, docs_path, embedding_model="text-embedding-3-small"):
        self.docs_path = docs_path
        self.embedding_model = embedding_model
        self.setup_pipeline()

    def setup_pipeline(self):
        # 1. 加载文档
        loader = DirectoryLoader(self.docs_path, glob="**/*.md")
        documents = loader.load()
        print(f"加载了 {len(documents)} 个文档")

        # 2. 分割为块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.chunks = text_splitter.split_documents(documents)
        print(f"分割为 {len(self.chunks)} 个块")

        # 3. 创建嵌入和向量存储
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        # 4. 创建检索器
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # 5. 创建 QA 链
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            verbose=True
        )

    def query(self, question):
        """查询 RAG 系统。"""
        result = self.qa_chain.invoke({"query": question})

        print(f"\n答案: {result['result']}")
        print(f"\n来源:")
        for i, doc in enumerate(result['source_documents']):
            print(f"  [{i+1}] {doc.metadata.get('source', '未知')} "
                  f"(第 {doc.metadata.get('page', 'N/A')} 页)")

        return result

# 使用
rag = BasicRAGPipeline("./knowledge_base")
answer = rag.query("我们的退货政策是什么？")
```

---

## 12.2 向量数据库选型

### 12.2.1 向量数据库生态

```
┌──────────────────────────────────────────────────────────────┐
│              向量数据库对比矩阵                                  │
│                                                                │
│  数据库      │ 类型    │ 规模   │ 过滤  │ 性能 │ 易用性      │
│  ────────────│─────────│────────│───────│──────│─────────────│
│  Chroma      │ 嵌入式  │ 本地   │ 基础  │ ★★★  │ ★★★★★     │
│  FAISS       │ 库      │ 单机   │ 基础  │ ★★★★★│ ★★★       │
│  Pinecone   │ 云服务  │ 全球   │ 好    │ ★★★★ │ ★★★★★     │
│  Weaviate   │ 服务端  │ 集群   │ 优秀  │ ★★★★ │ ★★★★      │
│  Milvus     │ 服务端  │ 集群   │ 优秀  │ ★★★★ │ ★★★       │
│  Qdrant     │ 服务端  │ 集群   │ 优秀  │ ★★★★ │ ★★★★      │
│  pgvector   │ 插件    │ 单机   │ 好    │ ★★★  │ ★★★★★     │
│  LanceDB    │ 嵌入式  │ 本地   │ 好    │ ★★★★ │ ★★★★      │
│  Vespa      │ 服务端  │ 集群   │ 优秀  │ ★★★★ │ ★★★       │
│  Algolia    │ 云服务  │ 全球   │ 优秀  │ ★★★★ │ ★★★★★     │
│                                                                │
│  选型标准：                                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  原型/小数据（<10万文档）：                             │    │
│  │  → Chroma, FAISS, pgvector                           │    │
│  │                                                       │    │
│  │  生产环境（10万-1000万文档）：                          │    │
│  │  → Qdrant, Weaviate, Pinecone                        │    │
│  │                                                       │    │
│  │  企业级（1000万+ 文档）：                               │    │
│  │  → Milvus, Vespa, Weaviate（集群模式）                │    │
│  │                                                       │    │
│  │  托管/无服务器：                                        │    │
│  │  → Pinecone, Weaviate Cloud, Qdrant Cloud             │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 12.2.2 嵌入模型

```python
# 嵌入模型对比
EMBEDDING_MODELS = {
    # OpenAI
    "text-embedding-3-small": {
        "dimensions": 1536,
        "max_tokens": 8191,
        "cost_per_1m_tokens": "$0.02",
        "performance": "良好",
    },
    "text-embedding-3-large": {
        "dimensions": 3072,
        "max_tokens": 8191,
        "cost_per_1m_tokens": "$0.13",
        "performance": "优秀",
    },

    # 开源模型
    "BAAI/bge-large-en-v1.5": {
        "dimensions": 1024,
        "max_tokens": 512,
        "cost_per_1m_tokens": "免费（自托管）",
        "performance": "优秀",
    },
    "BAAI/bge-m3": {
        "dimensions": 1024,
        "max_tokens": 8192,
        "cost_per_1m_tokens": "免费（自托管）",
        "performance": "优秀（多语言）",
    },
    "nomic-embed-text-v1.5": {
        "dimensions": 768,
        "max_tokens": 8192,
        "cost_per_1m_tokens": "免费（自托管）",
        "performance": "良好",
    },
    "jinaai/jina-embeddings-v3": {
        "dimensions": 1024,
        "max_tokens": 8192,
        "cost_per_1m_tokens": "免费（自托管）",
        "performance": "优秀",
    },
}

# 使用不同嵌入提供商
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# 选项 1：OpenAI 嵌入
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536,
)

# 选项 2：本地 HuggingFace 嵌入
local_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

# 选项 3：Ollama 嵌入（本地部署）
from langchain_community.embeddings import OllamaEmbeddings
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)
```

### 12.2.3 向量数据库设置

```python
# 使用 Chroma 的完整向量数据库设置
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreManager:
    """管理 RAG 系统的向量数据库。"""

    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

    def create_collection(self, collection_name, embedding_fn=None):
        """创建或获取集合。"""
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 或 "l2", "ip"
        )

    def add_documents(self, collection, documents, metadatas=None,
                      ids=None, batch_size=100):
        """分批添加文档。"""
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size] if metadatas else None
            batch_ids = ids[i:i+batch_size] if ids else None

            collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids or [f"doc_{j}" for j in range(i, i+len(batch_docs))],
            )

    def search(self, collection, query, n_results=5, where=None):
        """搜索，支持可选的元数据过滤。"""
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0] if results["metadatas"] else None,
            "distances": results["distances"][0] if results["distances"] else None,
        }

    def hybrid_search(self, collection, query, n_results=5,
                      keyword_weight=0.3, semantic_weight=0.7):
        """
        结合关键词（BM25）和语义搜索。
        实践中，使用专用的混合搜索引擎。
        """
        semantic_results = self.search(collection, query, n_results * 2)
        return semantic_results

    def get_stats(self, collection_name):
        """获取集合统计信息。"""
        collection = self.client.get_collection(collection_name)
        count = collection.count()
        return {
            "collection": collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
        }

# 使用
manager = VectorStoreManager("./my_rag_db")
collection = manager.create_collection("knowledge_base")

# 添加文档
manager.add_documents(
    collection,
    documents=["文档1内容...", "文档2内容..."],
    metadatas=[{"source": "file1.pdf"}, {"source": "file2.pdf"}],
)

# 搜索
results = manager.search(collection, "什么是机器学习？", n_results=5)
print(f"找到 {len(results['documents'])} 个结果")
```

---

## 12.3 检索策略设计

### 12.3.1 查询转换技术

检索质量很大程度上取决于查询的处理方式：

```
┌──────────────────────────────────────────────────────────────┐
│              查询转换技术                                        │
│                                                                │
│  1. 查询重写                                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  原始: "做网站的那个东西叫什么？"                        │    │
│  │  重写: "用于网站开发的工具或框架有哪些？"                 │    │
│  │                                                       │    │
│  │  方法: 使用 LLM 重写模糊查询                           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  2. HyDE（假设文档嵌入）                                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  查询: "光合作用如何工作？"                             │    │
│  │  假设回答（LLM 生成）：                                │    │
│  │  "光合作用是植物将阳光转化为能量的过程..."              │    │
│  │                                                       │    │
│  │  嵌入假设回答，而非查询                                │    │
│  │  为什么？假设回答在嵌入空间中更接近实际文档              │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  3. 多查询生成                                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  原始: "Python Web 框架对比"                           │    │
│  │                                                       │    │
│  │  生成的查询：                                          │    │
│  │  - "Django vs Flask vs FastAPI 性能对比"              │    │
│  │  - "最适合 REST API 的 Python 框架"                   │    │
│  │  - "2024 Python Web 框架基准测试"                     │    │
│  │                                                       │    │
│  │  对所有查询进行检索，合并/去重结果                      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  4. 回退提示                                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  原始: "法国的首都是什么？"                             │    │
│  │  回退: "法国有哪些主要城市？"                           │    │
│  │                                                       │    │
│  │  当查询对检索来说过于具体时有用                        │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

```python
# 查询转换实现
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class QueryTransformer:
    """为更好的检索转换查询。"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def rewrite_query(self, query: str) -> str:
        """重写模糊或质量差的查询。"""
        prompt = ChatPromptTemplate.from_template(
            """重写以下用户查询，使其更精确且适合语义搜索。
保持含义但提高清晰度。

用户查询: {query}

重写后的查询:"""
        )
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        return result.content

    def generate_hyde(self, query: str) -> str:
        """为 HyDE 生成假设文档回答。"""
        prompt = ChatPromptTemplate.from_template(
            """写一个简短的信息段落来回答这个问题。
就像来自权威文档一样撰写。

问题: {query}

假设文档:"""
        )
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        return result.content

    def generate_multi_queries(self, query: str, num_queries: int = 3) -> list:
        """从一个原始查询生成多个不同的查询。"""
        prompt = ChatPromptTemplate.from_template(
            """生成 {num_queries} 个不同的搜索查询，这些查询将帮助
找到回答此问题的信息。每个查询应从不同角度切入主题。

原始问题: {query}

返回查询（每行一个）:"""
        )
        chain = prompt | self.llm
        result = chain.invoke({"query": query, "num_queries": num_queries})
        return [q.strip() for q in result.content.split("\n") if q.strip()]

    def decompose_complex_query(self, query: str) -> list:
        """将复杂查询分解为子问题。"""
        prompt = ChatPromptTemplate.from_template(
            """将这个复杂问题分解为更简单的子问题，
每个子问题可以独立回答。

复杂问题: {query}

子问题（每行一个）:"""
        )
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        return [q.strip() for q in result.content.split("\n") if q.strip()]

# 使用示例
transformer = QueryTransformer()

# 重写
rewritten = transformer.rewrite_query("编程语言的那个东西")
print(f"重写: {rewritten}")

# HyDE
hyde_doc = transformer.generate_hyde("梯度下降如何工作？")
print(f"假设文档: {hyde_doc[:200]}...")

# 多查询
queries = transformer.generate_multi_queries("RAG vs 微调的权衡")
print(f"查询: {queries}")
```

### 12.3.2 检索算法

```
┌──────────────────────────────────────────────────────────────┐
│              检索算法对比                                        │
│                                                                │
│  算法             │ 速度  │ 质量   │ 内存  │ 最佳场景        │
│  ─────────────────│───────│────────│───────│─────────────────│
│  精确 NN（暴力）  │ 慢    │ 完美   │ 高    │ 小数据          │
│  IVF              │ 快    │ 良好   │ 中    │ 中等数据        │
│  HNSW             │ 快    │ 优秀   │ 高    │ 生产环境        │
│  PQ（乘积量化）   │ 快    │ 良好   │ 低    │ 大数据          │
│  ScaNN            │ 快    │ 优秀   │ 中    │ 生产环境        │
│  DiskANN          │ 中    │ 优秀   │ 低    │ 超大数据        │
│                                                                │
│  HNSW（分层可导航小世界图）：                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  第 2 层（稀疏）:  A ──── B                           │    │
│  │                     │    ╱ │                          │    │
│  │                     │   ╱  │                          │    │
│  │  第 1 层（中等）:   C ─ D ─ E ── F                    │    │
│  │                     │╲  │╲  │╲  │                    │    │
│  │                     │ ╲ │ ╲ │ ╲ │                    │    │
│  │  第 0 层（密集）:  G─H─I─J─K─L─M─N                   │    │
│  │                                                       │    │
│  │  搜索: 从顶层开始，逐层向下导航                        │    │
│  │  插入: 添加节点，连接到最近邻居                        │    │
│  │  时间: O(log n) 搜索                                  │    │
│  │  内存: O(n × m × 指针大小)                            │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  混合搜索（BM25 + 向量）：                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  查询 → BM25 分数 ─────┐                             │    │
│  │                        ├──▶ 融合 ──▶ 结果            │    │
│  │  查询 → 嵌入 → ANN ──┘                               │    │
│  │                                                       │    │
│  │  融合方法：                                            │    │
│  │  - 倒数排名融合（RRF）：                               │    │
│  │    score(d) = Σ 1/(k + rank_i(d))                    │    │
│  │  - 加权组合：                                          │    │
│  │    score(d) = α×BM25(d) + (1-α)×Vector(d)           │    │
│  │                                                       │    │
│  │  混合搜索比任一方法单独使用高 10-20%                   │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 12.3.3 分块策略

```python
# 高级分块策略
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

class ChunkingStrategies:
    """不同的文档分块方法。"""

    @staticmethod
    def recursive_split(documents, chunk_size=1000, chunk_overlap=200):
        """最常用：递归字符分割。"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_documents(documents)

    @staticmethod
    def semantic_chunking(documents):
        """基于语义相似度分割。"""
        embeddings = OpenAIEmbeddings()
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
        )
        return splitter.split_documents(documents)

    @staticmethod
    def markdown_header_splitting(documents):
        """尊重 markdown 结构的分割。"""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        return splitter.split_documents(documents)

    @staticmethod
    def parent_child_splitting(documents, parent_size=2000, child_size=500):
        """
        父子分块：小块用于检索，
        大块用于上下文。
        """
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=200,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=50,
        )

        parents = parent_splitter.split_documents(documents)
        children = child_splitter.split_documents(documents)

        return parents, children

    @staticmethod
    def context_enriched_splitting(documents, window_size=3):
        """
        带上下文增强的滑动窗口。
        每个块包含周围上下文。
        """
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
        )
        chunks = base_splitter.split_documents(documents)

        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            context_before = chunks[max(0, i-window_size):i]
            context_after = chunks[i+1:min(len(chunks), i+window_size+1)]

            enriched_text = ""
            if context_before:
                enriched_text += "[前置上下文] " + " ".join(
                    [c.page_content for c in context_before]
                ) + "\n\n"
            enriched_text += "[主要内容] " + chunk.page_content
            if context_after:
                enriched_text += "\n\n[后置上下文] " + " ".join(
                    [c.page_content for c in context_after]
                )

            chunk.page_content = enriched_text
            enriched_chunks.append(chunk)

        return enriched_chunks

# 块大小分析
def analyze_chunks(chunks, name="策略"):
    """分析块质量指标。"""
    lengths = [len(c.page_content) for c in chunks]
    print(f"\n{name}:")
    print(f"  总块数: {len(chunks)}")
    print(f"  平均长度: {sum(lengths)/len(lengths):.0f} 字符")
    print(f"  最小长度: {min(lengths)} 字符")
    print(f"  最大长度: {max(lengths)} 字符")
```

---

## 12.4 生成策略优化

### 12.4.1 RAG 提示工程

```
┌──────────────────────────────────────────────────────────────┐
│              RAG 提示模板设计                                    │
│                                                                │
│  基础模板：                                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  上下文: {retrieved_documents}                         │    │
│  │                                                       │    │
│  │  问题: {user_query}                                   │    │
│  │                                                       │    │
│  │  基于以上上下文回答：                                  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  高级模板（带引用）：                                            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  你是一个有帮助的助手，根据提供的上下文回答问题。       │    │
│  │  始终使用 [1]、[2] 等引用来源。                       │    │
│  │                                                       │    │
│  │  如果上下文不包含足够的信息，请说"我没有足够的信息      │    │
│  │  来回答这个问题。"                                     │    │
│  │                                                       │    │
│  │  上下文:                                               │    │
│  │  [1] {doc_1_content}                                  │    │
│  │  来源: {doc_1_source}                                 │    │
│  │                                                       │    │
│  │  [2] {doc_2_content}                                  │    │
│  │  来源: {doc_2_source}                                 │    │
│  │                                                       │    │
│  │  问题: {user_query}                                   │    │
│  │                                                       │    │
│  │  回答:                                                │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Self-RAG 模板：                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  你是一个有帮助的助手。对于每个问题：                    │    │
│  │                                                       │    │
│  │  1. 分析是否需要检索 [检索: 是/否]                    │    │
│  │  2. 如果是，评估每个文档的相关性                       │    │
│  │     [相关性: 每个文档 是/否]                           │    │
│  │  3. 生成由相关文档支持的答案                           │    │
│  │  4. 评估你的置信度 [支持: 是/否]                      │    │
│  │                                                       │    │
│  │  上下文: {retrieved_documents}                         │    │
│  │  问题: {user_query}                                   │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 12.4.2 上下文窗口管理

```python
# RAG 上下文窗口管理
class ContextManager:
    """当检索到的文档超出限制时管理上下文窗口。"""

    def __init__(self, max_context_tokens=4096, reserved_tokens=1024):
        self.max_context_tokens = max_context_tokens
        self.reserved_tokens = reserved_tokens  # 查询 + 回答
        self.available_tokens = max_context_tokens - reserved_tokens

    def estimate_tokens(self, text: str) -> int:
        """粗略的 token 估算（英文约 1 token ≈ 4 字符）。"""
        return len(text) // 4

    def truncate_documents(self, documents: list, token_budget: int = None) -> list:
        """截断文档以适应 token 预算。"""
        budget = token_budget or self.available_tokens
        selected = []
        current_tokens = 0

        for doc in documents:
            doc_tokens = self.estimate_tokens(doc.page_content)
            if current_tokens + doc_tokens <= budget:
                selected.append(doc)
                current_tokens += doc_tokens
            else:
                remaining = budget - current_tokens
                if remaining > 100:  # 最小可行块
                    truncated_content = doc.page_content[:remaining * 4]
                    doc.page_content = truncated_content + "..."
                    selected.append(doc)
                break

        return selected

    def compress_context(self, documents: list, compression_ratio: float = 0.5) -> str:
        """使用抽取式压缩压缩文档。"""
        compressed = []
        for doc in documents:
            sentences = doc.page_content.split(". ")
            num_keep = max(1, int(len(sentences) * compression_ratio))
            kept = sentences[:num_keep]
            compressed.append(". ".join(kept))

        return "\n\n".join(compressed)

    def build_prompt(self, query: str, documents: list,
                     template: str = None) -> str:
        """构建带管理上下文的最终提示。"""
        if template is None:
            template = """根据下面的上下文回答问题。
如果上下文不包含足够的信息，请说明。

上下文:
{context}

问题: {query}

回答:"""

        truncated = self.truncate_documents(documents)

        context_parts = []
        for i, doc in enumerate(truncated):
            source = doc.metadata.get('source', '未知')
            context_parts.append(f"[{i+1}] {doc.page_content}\n来源: {source}")

        context = "\n\n".join(context_parts)

        return template.format(context=context, query=query)

# 使用
context_mgr = ContextManager(max_context_tokens=8192)
prompt = context_mgr.build_prompt(
    query="退货政策是什么？",
    documents=retrieved_docs,
)
print(f"提示长度: {context_mgr.estimate_tokens(prompt)} tokens")
```

### 12.4.3 幻觉减少

```
┌──────────────────────────────────────────────────────────────┐
│              RAG 中的幻觉减少                                   │
│                                                                │
│  技术：                                                         │
│                                                                │
│  1. 基于上下文的生成                                            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  指令: "仅基于提供的上下文回答。如果上下文不包含       │    │
│  │  答案，请说'我没有足够的信息'。"                       │    │
│  │                                                       │    │
│  │  有效性: 60-70% 幻觉减少                              │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  2. 带引用的思维链                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  "逐步思考。对于每一步，引用相关文档。                  │    │
│  │  仅使用引用文档中的信息。"                             │    │
│  │                                                       │    │
│  │  有效性: 70-80% 幻觉减少                              │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  3. 自一致性检查                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  生成 N 个回答，检查一致性。                           │    │
│  │  如果回答不一致，标记为不确定。                        │    │
│  │                                                       │    │
│  │  有效性: 75-85% 幻觉减少                              │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  4. 检索质量门控                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  生成前: 检查检索到的文档是否与查询实际相关。           │    │
│  │  如果不相关，不生成。                                  │    │
│  │                                                       │    │
│  │  有效性: 50-60%（防止在检索失败时生成）               │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 12.5 RAG 评估框架

### 12.5.1 评估指标

```
┌──────────────────────────────────────────────────────────────┐
│              RAG 评估指标                                       │
│                                                                │
│  检索指标：                                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  指标           │ 描述               │ 范围           │    │
│  │  ───────────────│────────────────────│───────────────│    │
│  │  Precision@K    │ Top K 中相关比例    │ 0-1           │    │
│  │  Recall@K       │ 找到所有相关比例    │ 0-1           │    │
│  │  MRR            │ 平均倒数排名        │ 0-1           │    │
│  │  NDCG@K         │ 归一化折扣累积增益  │ 0-1           │    │
│  │  命中率         │ 至少一个相关        │ 0-1           │    │
│  │  MAP            │ 平均精度均值        │ 0-1           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  生成指标：                                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  指标           │ 描述               │ 工具           │    │
│  │  ───────────────│────────────────────│───────────────│    │
│  │  忠实度         │ 回答基于上下文      │ RAGAS         │    │
│  │  相关性         │ 回答解决问题        │ RAGAS         │    │
│  │  正确性         │ 回答事实正确        │ 人工评估       │    │
│  │  完整性         │ 回答覆盖所有方面    │ LLM 评审      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  端到端指标：                                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  指标           │ 描述                                │    │
│  │  ───────────────│─────────────────────────────────────│    │
│  │  回答率         │ 获得回答的查询百分比                 │    │
│  │  拒绝率         │ 正确拒绝的查询百分比                 │    │
│  │  延迟           │ 端到端响应时间                       │    │
│  │  每查询成本     │ 每个查询的总成本                     │    │
│  │  用户满意度     │ 人工评分                             │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 12.5.2 RAGAS 评估框架

```python
# RAGAS 评估实现
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

class RAGEvaluator:
    """使用 RAGAS 框架评估 RAG 系统。"""

    def __init__(self):
        self.metrics = [
            faithfulness,       # 回答是否基于上下文？
            answer_relevancy,   # 回答是否解决问题？
            context_precision,  # 检索到的上下文是否相关？
            context_recall,     # 是否检索到所有相关上下文？
        ]

    def prepare_eval_data(self, questions, answers, contexts,
                          ground_truths=None):
        """准备 RAGAS 格式的数据。"""
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths or [""] * len(questions),
        }
        return Dataset.from_dict(data)

    def evaluate(self, eval_data):
        """运行 RAGAS 评估。"""
        results = evaluate(
            dataset=eval_data,
            metrics=self.metrics,
        )
        return results

    def analyze_results(self, results):
        """分析并打印评估结果。"""
        print("\n" + "="*60)
        print("RAG 评估结果")
        print("="*60)

        for metric_name, score in results.items():
            print(f"{metric_name:25s}: {score:.4f}")

        print("="*60)

        if results['faithfulness'] < 0.7:
            print("⚠️ 忠实度低：模型可能在幻觉")
            print("   → 改进提示中的基于上下文指令")

        if results['answer_relevancy'] < 0.7:
            print("⚠️ 相关性低：回答未解决问题")
            print("   → 改进查询理解")

        if results['context_precision'] < 0.7:
            print("⚠️ 上下文精度低：检索到不相关文档")
            print("   → 改进嵌入模型")

        if results['context_recall'] < 0.7:
            print("⚠️ 上下文召回低：缺少相关文档")
            print("   → 增加检索文档数量")

        return results

# 使用
evaluator = RAGEvaluator()

questions = ["退货政策是什么？", "如何重置密码？"]
answers = ["...", "..."]
contexts = [["...", "..."], ["...", "..."]]
ground_truths = ["实际答案 1", "实际答案 2"]

eval_data = evaluator.prepare_eval_data(
    questions, answers, contexts, ground_truths
)
results = evaluator.evaluate(eval_data)
evaluator.analyze_results(results)
```

---

## 12.6 高级 RAG 技术

### 12.6.1 Self-RAG

Self-RAG 训练模型决定何时检索以及如何使用检索到的文档：

```
┌──────────────────────────────────────────────────────────────┐
│              Self-RAG 架构                                      │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  输入: "法国的首都是什么？"                             │    │
│  │                                                       │    │
│  │  步骤 1: 检索决策                                     │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  [检索: 否]                                  │     │    │
│  │  │  模型判断它知道答案                           │     │    │
│  │  │  → 直接从参数化记忆生成                       │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  │                                                       │    │
│  │  输入: "苹果 2024 年第三季度营收是多少？"              │    │
│  │                                                       │    │
│  │  步骤 1: 检索决策                                     │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  [检索: 是]                                  │     │    │
│  │  │  模型判断需要外部信息                         │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  │                                                       │    │
│  │  步骤 2: 检索文档                                    │    │
│  │                                                       │    │
│  │  步骤 3: 相关性判断                                   │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  文档 1: [相关性: 是] 苹果 Q3 报告           │     │    │
│  │  │  文档 2: [相关性: 否] 三星营收               │     │    │
│  │  │  文档 3: [相关性: 是] 苹果财务数据           │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  │                                                       │    │
│  │  步骤 4: 带支持的生成                                  │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  "苹果 2024 年 Q3 营收为 818 亿美元 [1,3]"   │     │    │
│  │  │  [支持: 是] - 答案由检索文档支持              │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 12.6.2 Graph RAG

Graph RAG 将知识图谱与向量检索结合：

```
┌──────────────────────────────────────────────────────────────┐
│              Graph RAG 架构                                     │
│                                                                │
│  知识图谱：                                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │       (北京) ────首都_of────▶ (中国)                   │    │
│  │         │                        │                    │    │
│  │    位于_in                   位于_in                   │    │
│  │         │                        │                    │    │
│  │         ▼                        ▼                    │    │
│  │    (亚洲) ◀──── 大洲_of ───(亚洲)                    │    │
│  │                                                       │    │
│  │  实体: 北京, 中国, 亚洲                               │    │
│  │  关系: 首都_of, 位于_in, 大洲_of                      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Graph RAG 流程：                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  1. 从文档中提取实体和关系                             │    │
│  │  2. 构建知识图谱                                     │    │
│  │  3. 对于查询：                                        │    │
│  │     a. 识别相关实体                                   │    │
│  │     b. 遍历图谱获取关联知识                           │    │
│  │     c. 与向量检索结合                                 │    │
│  │     d. 从图谱上下文生成答案                           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  优势：                                                         │
│  - 多跳推理（跟随关系）                                        │
│  - 结构化知识（非仅文本）                                      │
│  - 更适合需要推理的复杂查询                                     │
└──────────────────────────────────────────────────────────────┘
```

### 12.6.3 多模态 RAG

```python
# 多模态 RAG 概念
class MultiModalRAG:
    """
    处理文本、图像、表格和代码的 RAG 系统。
    """

    def __init__(self):
        self.text_store = None   # 文本向量库
        self.image_store = None  # 图像嵌入
        self.table_store = None  # 表格表示

    def index_document(self, document):
        """索引多模态文档。"""
        for element in document.elements:
            if element.type == "text":
                self.index_text(element)
            elif element.type == "image":
                self.index_image(element)
            elif element.type == "table":
                self.index_table(element)
            elif element.type == "code":
                self.index_code(element)

    def index_image(self, image_element):
        """
        使用多模态嵌入索引图像。
        CLIP、SigLIP 等模型可以在同一空间嵌入图像和文本。
        """
        from sentence_transformers import SentenceTransformer
        clip_model = SentenceTransformer("clip-ViT-B-32")

        image_embedding = clip_model.encode(image_element.image)
        caption_embedding = clip_model.encode(image_element.caption)

        self.image_store.add(
            embeddings=[image_embedding, caption_embedding],
            metadatas=[{"type": "image"}, {"type": "caption"}],
            ids=[f"img_{image_element.id}", f"cap_{image_element.id}"],
        )

    def retrieve_multimodal(self, query, modalities=["text", "image", "table"]):
        """跨所有模态检索。"""
        results = {}

        if "text" in modalities:
            results["text"] = self.text_store.search(query, k=5)

        if "image" in modalities:
            results["image"] = self.image_store.search(query, k=3)

        if "table" in modalities:
            results["table"] = self.table_store.search(query, k=3)

        return results
```

### 12.6.4 Agentic RAG

```
┌──────────────────────────────────────────────────────────────┐
│              Agentic RAG 架构                                   │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  智能体（LLM）+ 工具                                  │    │
│  │                                                       │    │
│  │  工具：                                                │    │
│  │  ├── 向量搜索工具                                    │    │
│  │  ├── SQL 查询工具                                    │    │
│  │  ├── 网络搜索工具                                    │    │
│  │  ├── 计算器工具                                      │    │
│  │  └── 代码执行工具                                    │    │
│  │                                                       │    │
│  │  智能体循环：                                          │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  while not done:                             │     │    │
│  │  │    1. 分析问题                                │     │    │
│  │  │    2. 决定使用哪个工具                        │     │    │
│  │  │    3. 执行工具                               │     │    │
│  │  │    4. 评估结果                               │     │    │
│  │  │    5. 如果不够，尝试另一个工具               │     │    │
│  │  │    6. 如果够了，生成最终答案                  │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  示例：                                                         │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  问题: "东京人口乘以 3 是多少？"                       │    │
│  │                                                       │    │
│  │  步骤 1: 需要人口数据 → 向量搜索                      │    │
│  │  步骤 2: 找到"东京人口: 1396 万"                      │    │
│  │  步骤 3: 需要计算 → 计算器工具                        │    │
│  │  步骤 4: 1396万 × 3 = 4188万                         │    │
│  │  步骤 5: 答案完整 → 生成响应                          │    │
│  │                                                       │    │
│  │  答案: "东京人口约 1396 万。乘以 3，约为 4188 万。"  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

```python
# Agentic RAG 实现
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

class AgenticRAG:
    """带智能体的 RAG 系统，智能体决定检索策略。"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.setup_tools()

    def setup_tools(self):
        """定义智能体可用的工具。"""

        @tool
        def vector_search(query: str) -> str:
            """搜索知识库中的相关文档。"""
            docs = self.vectorstore.similarity_search(query, k=5)
            return "\n\n".join([doc.page_content for doc in docs])

        @tool
        def get_metadata_stats() -> str:
            """获取可用文档的统计信息。"""
            collection = self.vectorstore._collection
            count = collection.count()
            return f"知识库包含 {count} 个文档块。"

        self.tools = [vector_search, get_metadata_stats]

    def create_agent(self):
        """创建 RAG 智能体。"""
        prompt = ChatPromptTemplate.from_template(
            """你是一个有帮助的助手，使用知识库回答问题。
你有工具可以搜索知识库。

回答时：
1. 首先理解用户在问什么
2. 使用适当的工具查找相关信息
3. 将信息综合为清晰的答案
4. 引用你的来源

如果找不到相关信息，请诚实说明。

可用工具:
{tools}

工具名称: {tool_names}

问题: {input}

{agent_scratchpad}"""
        )

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def query(self, question: str) -> str:
        """查询 Agentic RAG 系统。"""
        agent = self.create_agent()
        result = agent.invoke({"input": question})
        return result["output"]
```

---

## 💡 案例：基于 LangChain + Chroma 的企业知识库

### 业务背景

一家中型公司（500名员工）需要内部知识库，涵盖：
- HR 政策（100+ 文档）
- 技术文档（500+ 文档）
- 销售手册（50+ 文档）
- 会议记录（1000+ 文档）

需求：
- 对所有文档的自然语言问答
- 每个答案附带来源引用
- 访问控制（不同角色看不同文档）
- <3 秒响应时间
- 必须本地部署（数据隐私）

### 架构设计

```
┌──────────────────────────────────────────────────────────────────┐
│           企业知识库架构                                            │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  前端（Streamlit / React）                                │    │
│  │  ├── 聊天界面                                            │    │
│  │  ├── 文档上传（管理员）                                   │    │
│  │  └── 来源查看器                                          │    │
│  └───────────────────────┬──────────────────────────────────┘    │
│                           │                                        │
│                           ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  API 层（FastAPI）                                        │    │
│  │  ├── 身份验证（JWT）                                     │    │
│  │  ├── 限流                                               │    │
│  │  ├── 请求验证                                            │    │
│  │  └── 响应缓存（Redis）                                   │    │
│  └───────────────────────┬──────────────────────────────────┘    │
│                           │                                        │
│                           ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  RAG 引擎                                                 │    │
│  │  ├── 查询处理（重写 + 扩展）                              │    │
│  │  ├── 检索（混合: BM25 + 向量）                            │    │
│  │  ├── 重排序（交叉编码器）                                 │    │
│  │  ├── 访问控制过滤                                        │    │
│  │  └── 生成（本地 LLM 或 API）                             │    │
│  └───────────────────────┬──────────────────────────────────┘    │
│                           │                                        │
│              ┌────────────┼────────────┐                          │
│              ▼            ▼            ▼                          │
│  ┌────────────────┐ ┌──────────┐ ┌──────────┐                  │
│  │  Chroma DB     │ │  BM25    │ │  Redis   │                  │
│  │  （嵌入）       │ │  索引    │ │  （缓存） │                  │
│  │  16GB 内存     │ │  4GB 内存│ │  2GB 内存│                  │
│  └────────────────┘ └──────────┘ └──────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

### 实现

```python
# enterprise_kb.py
"""
基于 LangChain + Chroma 的企业知识库
带访问控制和缓存的完整实现。
"""
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, DirectoryLoader
)
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
import redis

class EnterpriseKnowledgeBase:
    """带访问控制和缓存的企业级 RAG 系统。"""

    def __init__(self, config):
        self.config = config
        self.setup_components()

    def setup_components(self):
        """初始化所有组件。"""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.config.get("openai_api_key"),
        )

        self.chroma_client = chromadb.PersistentClient(
            path=self.config.get("chroma_path", "./enterprise_chroma")
        )

        self.cache = redis.Redis(
            host=self.config.get("redis_host", "localhost"),
            port=self.config.get("redis_port", 6379),
            decode_responses=True,
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=self.config.get("openai_api_key"),
        )

    def load_documents(self, doc_dir: str):
        """加载并索引所有文档。"""
        loaders = {
            "*.pdf": PyPDFLoader,
            "*.docx": Docx2txtLoader,
            "*.txt": TextLoader,
            "*.md": TextLoader,
        }

        all_docs = []
        for pattern, loader_cls in loaders.items():
            loader = DirectoryLoader(
                doc_dir,
                glob=f"**/{pattern}",
                loader_cls=loader_cls,
                show_progress=True,
            )
            all_docs.extend(loader.load())

        print(f"加载了 {len(all_docs)} 个文档")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = splitter.split_documents(all_docs)
        print(f"分割为 {len(chunks)} 个块")

        for chunk in chunks:
            chunk.metadata["indexed_at"] = datetime.now().isoformat()
            chunk.metadata["content_hash"] = hashlib.md5(
                chunk.page_content.encode()
            ).hexdigest()

        collection = self.chroma_client.get_or_create_collection(
            name="enterprise_docs",
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            collection.add(
                documents=[c.page_content for c in batch],
                metadatas=[c.metadata for c in batch],
                ids=[f"doc_{j}" for j in range(i, i+len(batch))],
            )

        print(f"在 Chroma 中索引了 {len(chunks)} 个块")

    def query(self, question: str, user_role: str = "employee",
              use_cache: bool = True) -> dict:
        """带缓存和访问控制的知识库查询。"""
        cache_key = f"rag:{hashlib.md5(question.encode()).hexdigest()}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return json.loads(cached)

        access_filter = self._get_access_filter(user_role)

        collection = self.chroma_client.get_collection("enterprise_docs")
        results = collection.query(
            query_texts=[question],
            n_results=10,
            where=access_filter,
        )

        context_parts = []
        for i, (doc, meta) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0]
        )):
            source = meta.get("source", "未知")
            context_parts.append(f"[{i+1}] {doc}\n来源: {source}")

        context = "\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_template(
            """你是一个有帮助的助手，根据公司文档回答问题。
始终使用 [1]、[2] 等引用来源。

上下文:
{context}

问题: {question}

基于以上上下文回答。如果上下文不包含足够的信息，请说明。"""
        )

        chain = prompt | self.llm
        result = chain.invoke({"context": context, "question": question})

        response = {
            "answer": result.content,
            "sources": [meta.get("source") for meta in results["metadatas"][0][:3]],
            "num_docs_retrieved": len(results["documents"][0]),
        }

        if use_cache:
            self.cache.setex(cache_key, 3600, json.dumps(response))

        return response

    def _get_access_filter(self, role: str) -> dict:
        """基于用户角色构建 ChromaDB 过滤器。"""
        access_map = {
            "admin": {"$or": [
                {"category": "hr"},
                {"category": "technical"},
                {"category": "sales"},
                {"category": "meetings"},
            ]},
            "engineering": {"$or": [
                {"category": "technical"},
                {"category": "meetings"},
            ]},
            "sales": {"$or": [
                {"category": "sales"},
                {"category": "hr"},
            ]},
            "employee": {"category": "hr"},
        }
        return access_map.get(role, {"category": "hr"})

# 使用
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "chroma_path": "./enterprise_chroma",
    "redis_host": "localhost",
}

kb = EnterpriseKnowledgeBase(config)
kb.load_documents("./knowledge_base/")

result = kb.query("远程工作政策是什么？", user_role="employee")
print(f"回答: {result['answer']}")
print(f"来源: {result['sources']}")
```

### 性能结果

```
┌──────────────────────────────────────────────────────────────┐
│              企业知识库性能结果                                  │
│                                                                │
│  指标                  │ 值                                   │
│  ──────────────────────│──────────────────────────────────────│
│  总文档数              │ 1,650                                │
│  总块数                │ 12,847                               │
│  向量库大小            │ 15.2 GB                              │
│  索引时间              │ 45 分钟（初始）                       │
│  查询延迟 (P50)        │ 1.2 秒                               │
│  查询延迟 (P99)        │ 2.8 秒                               │
│  缓存命中率            │ 34%                                  │
│  准确率（人工评估）     │ 89%                                  │
│                                                                │
│  月度成本明细：                                                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  OpenAI API（嵌入 + LLM）：$45                        │    │
│  │  Redis（云）：$15                                    │    │
│  │  计算（API 服务器）：$50                             │    │
│  │  总计：约 $110/月                                    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  与之前方案对比：                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  之前：SharePoint 搜索                                 │    │
│  │  - 查询时间：10-30 秒                                 │    │
│  │  - 准确率：约 60%                                     │    │
│  │  - 不支持自然语言                                     │    │
│  │                                                       │    │
│  │  新方案：基于 RAG 的知识库                              │    │
│  │  - 查询时间：1-3 秒（快 10 倍）                       │    │
│  │  - 准确率：89%（+29%）                               │    │
│  │  - 自然语言问答                                       │    │
│  │  - 来源引用                                           │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 本章小结

本章涵盖了 RAG 系统的完整架构：

| 主题 | 关键要点 |
|------|---------|
| **RAG 原理** | 检索 + 增强 + 生成；知识密集型任务优先于微调 |
| **向量数据库** | 原型用 Chroma，生产用 Qdrant/Weaviate |
| **检索策略** | 混合搜索（BM25 + 向量）比单独任一方法高 10-20% |
| **查询转换** | HyDE 和多查询可将检索质量提高 15-25% |
| **生成** | 基于上下文的生成加引用可将幻觉减少 60-80% |
| **评估** | RAGAS 框架：忠实度 + 相关性 + 精度 + 召回 |
| **高级 RAG** | Self-RAG、Graph RAG、Agentic RAG 用于复杂场景 |

---

## 参考文献

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.
2. Gao, Y., et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." arXiv.
3. Asai, A., et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." ICLR.
4. Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv.
5. LangChain 文档. https://python.langchain.com/
6. ChromaDB 文档. https://docs.trychroma.com/
7. RAGAS 文档. https://docs.ragas.io/
8. Qdrant 文档. https://qdrant.tech/documentation/
9. Weaviate 文档. https://weaviate.io/developers/weaviate
10. Zou, X., et al. (2024). "A Survey on Retrieval-Augmented Text Generation for Large Language Models." arXiv.

---

*← [第11章 - LLM 推理架构](chapter-11.md) | [第13章 - 模型微调架构](chapter-13.md) →*
