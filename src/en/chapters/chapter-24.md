# Chapter 24: Practice and Synthesis

**Reader Level:** 🔴 Advanced | **Pages:** 30 | **Code:** Python, Docker, Kubernetes

---

## 24.1 End-to-End Project: Intelligent Document Processing

### 24.1.1 Project Overview

Build a production document processing pipeline that extracts, classifies, and summarizes information from mixed-format documents (PDFs, images, emails).

```
┌────────────────────────────────────────────────────────────────┐
│              DOCUMENT PROCESSING PIPELINE                       │
│                                                                 │
│  Input Documents                                                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                          │
│  │   PDF   │ │  Image  │ │  Email  │                          │
│  └────┬────┘ └────┬────┘ └────┬────┘                          │
│       │           │           │                                │
│       v           v           v                                │
│  ┌─────────────────────────────────────┐                       │
│  │     Document Parser & OCR           │                       │
│  │  (Tesseract + Layout Analysis)      │                       │
│  └──────────────────┬──────────────────┘                       │
│                     │                                          │
│                     v                                          │
│  ┌─────────────────────────────────────┐                       │
│  │     Content Extraction              │                       │
│  │  (NER + Table Detection + OCR)      │                       │
│  └──────────────────┬──────────────────┘                       │
│                     │                                          │
│           ┌─────────┴─────────┐                               │
│           v                   v                                │
│  ┌─────────────────┐ ┌─────────────────┐                      │
│  │   Classifier    │ │   Summarizer    │                      │
│  │  (BERT-based)   │ │  (LLM-based)    │                      │
│  └────────┬────────┘ └────────┬────────┘                      │
│           │                   │                                │
│           └─────────┬─────────┘                               │
│                     v                                          │
│  ┌─────────────────────────────────────┐                       │
│  │        Output Storage               │                       │
│  │  (Structured JSON + Vector Store)   │                       │
│  └─────────────────────────────────────┘                       │
└────────────────────────────────────────────────────────────────┘
```

### 24.1.2 Implementation: Document Parser

```python
# document_parser.py
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ParsedDocument:
    doc_id: str
    doc_type: str
    text_content: str
    tables: List[Dict]
    metadata: Dict
    confidence: float

class DocumentParser:
    def __init__(self):
        self.parsers = {
            "pdf": self._parse_pdf,
            "image": self._parse_image,
            "email": self._parse_email,
        }
    
    def parse(self, file_path: str) -> ParsedDocument:
        doc_type = self._detect_type(file_path)
        parser = self.parsers.get(doc_type, self._parse_generic)
        return parser(file_path)
    
    def _detect_type(self, path: str) -> str:
        if path.endswith(".pdf"):
            return "pdf"
        elif path.endswith((".png", ".jpg", ".jpeg")):
            return "image"
        elif path.endswith((".eml", ".msg")):
            return "email"
        return "generic"
    
    def _parse_pdf(self, path: str) -> ParsedDocument:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return ParsedDocument(
            doc_id=path, doc_type="pdf",
            text_content=text, tables=[], metadata={},
            confidence=0.95,
        )
    
    def _parse_image(self, path: str) -> ParsedDocument:
        # OCR processing
        return ParsedDocument(
            doc_id=path, doc_type="image",
            text_content="[OCR result]", tables=[],
            metadata={}, confidence=0.80,
        )
    
    def _parse_email(self, path: str) -> ParsedDocument:
        return ParsedDocument(
            doc_id=path, doc_type="email",
            text_content="[Email content]", tables=[],
            metadata={}, confidence=0.90,
        )
    
    def _parse_generic(self, path: str) -> ParsedDocument:
        with open(path) as f:
            text = f.read()
        return ParsedDocument(
            doc_id=path, doc_type="generic",
            text_content=text, tables=[],
            metadata={}, confidence=0.70,
        )
```


### 24.1.3 Implementation: Document Classifier

```python
# document_classifier.py
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ClassificationResult:
    category: str
    confidence: float
    subcategories: List[Dict]

class DocumentClassifier:
    CATEGORIES = [
        "invoice", "contract", "report", "email",
        "receipt", "legal", "technical", "other"
    ]
    
    def __init__(self, model=None):
        self.model = model
    
    def classify(self, text: str) -> ClassificationResult:
        # In production, use a fine-tuned classifier
        category_scores = {}
        for cat in self.CATEGORIES:
            score = self._compute_relevance(text, cat)
            category_scores[cat] = score
        
        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]
        
        return ClassificationResult(
            category=best_category,
            confidence=best_score,
            subcategories=[
                {"name": k, "score": v}
                for k, v in sorted(
                    category_scores.items(),
                    key=lambda x: -x[1]
                )[:3]
            ],
        )
    
    def _compute_relevance(self, text: str, category: str) -> float:
        # Keyword-based scoring (simplified)
        keywords = {
            "invoice": ["payment", "total", "amount", "bill"],
            "contract": ["agreement", "party", "terms", "clause"],
            "report": ["summary", "analysis", "findings", "data"],
            "email": ["dear", "regards", "sent", "subject"],
        }
        text_lower = text.lower()
        category_keywords = keywords.get(category, [])
        matches = sum(1 for kw in category_keywords if kw in text_lower)
        return matches / max(len(category_keywords), 1)
```

### 24.1.4 Implementation: Document Summarizer

```python
# document_summarizer.py
from typing import Dict, List

class DocumentSummarizer:
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def summarize(self, text: str, style: str = "executive") -> Dict:
        if self.llm:
            return self._llm_summarize(text, style)
        return self._extractive_summarize(text)
    
    def _llm_summarize(self, text: str, style: str) -> Dict:
        prompts = {
            "executive": "Provide a 3-sentence executive summary:\n\n",
            "detailed": "Provide a detailed summary with key points:\n\n",
            "bullet": "Summarize as bullet points:\n\n",
        }
        prompt = prompts.get(style, prompts["executive"]) + text[:4000]
        
        # Placeholder for LLM call
        return {
            "summary": f"[{style} summary of document]",
            "style": style,
            "word_count": len(text.split()),
        }
    
    def _extractive_summarize(self, text: str) -> Dict:
        sentences = text.split(". ")
        scored = [(s, len(s.split())) for s in sentences]
        scored.sort(key=lambda x: -x[1])
        top_sentences = [s for s, _ in scored[:3]]
        return {
            "summary": ". ".join(top_sentences),
            "style": "extractive",
            "word_count": len(text.split()),
        }
```

## 24.2 End-to-End Project: Conversational AI Assistant

### 24.2.1 Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│              CONVERSATIONAL AI ASSISTANT                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │                  API Gateway                          │      │
│  │  (Auth + Rate Limit + Request Routing)                │      │
│  └──────────────────┬───────────────────────────────────┘      │
│                     │                                          │
│                     v                                          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Conversation Manager                     │      │
│  │  • Session state management                          │      │
│  │  • Context window management                         │      │
│  │  • Multi-turn memory                                 │      │
│  └──────────────────┬───────────────────────────────────┘      │
│                     │                                          │
│        ┌────────────┼────────────┐                            │
│        v            v            v                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
│  │  RAG     │ │  Tools   │ │  Safety  │                      │
│  │  Engine  │ │  Router  │ │  Filter  │                      │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                      │
│       │             │            │                             │
│       └─────────────┼────────────┘                            │
│                     v                                          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Response Generator                       │      │
│  │  (Streaming + Citation + Format)                      │      │
│  └──────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────┘
```


### 24.2.2 Conversation State Management

```python
# conversation_manager.py
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import uuid
from datetime import datetime

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

@dataclass
class ConversationSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: List[Message] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    max_context_tokens: int = 4096

class ConversationManager:
    def __init__(self, max_history: int = 50):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_history = max_history
    
    def get_or_create_session(self, session_id: str = None) -> ConversationSession:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        session = ConversationSession(
            session_id=session_id or uuid.uuid4().hex
        )
        self.sessions[session.session_id] = session
        return session
    
    def add_message(self, session_id: str, role: str, content: str):
        session = self.get_or_create_session(session_id)
        session.messages.append(Message(role=role, content=content))
        
        # Trim history if too long
        if len(session.messages) > self.max_history:
            session.messages = session.messages[-self.max_history:]
    
    def get_context_window(self, session_id: str) -> List[Dict]:
        session = self.get_or_create_session(session_id)
        messages = []
        total_tokens = 0
        
        for msg in reversed(session.messages):
            estimated_tokens = len(msg.content.split()) * 1.3
            if total_tokens + estimated_tokens > session.max_context_tokens:
                break
            messages.insert(0, {
                "role": msg.role,
                "content": msg.content,
            })
            total_tokens += estimated_tokens
        
        return messages
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
```

### 24.2.3 RAG Integration

```python
# rag_integration.py
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RAGResult:
    answer: str
    sources: List[Dict]
    confidence: float

class RAGIntegration:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm = llm_client
    
    def query(self, question: str, context: List[Dict] = None) -> RAGResult:
        # Retrieve relevant documents
        docs = self.retriever.retrieve(question, top_k=5)
        
        # Build context
        doc_context = "\n\n".join([
            f"[{i+1}] {doc['content']}"
            for i, doc in enumerate(docs)
        ])
        
        # Generate answer
        prompt = f"""Context:\n{doc_context}\n\nQuestion: {question}\n\nAnswer:"""
        
        answer = self.llm.generate(prompt)
        
        return RAGResult(
            answer=answer,
            sources=[{"id": d["id"], "score": d["score"]} for d in docs],
            confidence=docs[0]["score"] if docs else 0.0,
        )
```

## 24.3 Deployment Patterns

### 24.3.1 Blue-Green Deployment for AI Models

```
┌────────────────────────────────────────────────────────────────┐
│              BLUE-GREEN DEPLOYMENT                              │
│                                                                 │
│  Load Balancer                                                  │
│       │                                                         │
│       ├──────────────> Blue (Current) ──── Model v1.0         │
│       │                   │                                     │
│       │                   v                                     │
│       │              [Health Check]                             │
│       │                                                         │
│       └──── (switch) ──> Green (New) ──── Model v2.0          │
│                             │                                   │
│                             v                                   │
│                        [Health Check]                           │
│                                                                 │
│  Rollback: Switch traffic back to Blue in < 1 minute           │
└────────────────────────────────────────────────────────────────┘
```

### 24.3.2 Canary Deployment Configuration

```yaml
# canary_deployment.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ai-model-service
spec:
  hosts:
  - ai-model
  http:
  - route:
    - destination:
        host: ai-model
        subset: stable
      weight: 90
    - destination:
        host: ai-model
        subset: canary
      weight: 10
    retries:
      attempts: 3
      perTryTimeout: 2s
```

### 24.3.3 Infrastructure as Code

```hcl
# main.tf - AI Service Infrastructure
resource "aws_ecs_service" "ai_service" {
  name            = "ai-inference-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.ai_task.arn
  desired_count   = 3
  
  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.ai_tg.arn
    container_name   = "ai-model"
    container_port   = 8080
  }
}

resource "aws_appautoscaling_policy" "gpu_scaling" {
  name               = "gpu-scaling"
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.ai_service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  
  target_tracking_scaling_policy_configuration {
    target_value = 70.0
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
  }
}
```


## 24.4 Production Checklist

### 24.4.1 Pre-Launch Checklist

```
□ Model Performance
  □ Benchmark results meet quality thresholds
  □ Latency requirements satisfied (p50/p99)
  □ Throughput capacity verified
  □ Memory usage within budget

□ Security
  □ Input validation implemented
  □ Rate limiting configured
  □ Authentication/authorization working
  □ PII detection active
  □ Audit logging enabled

□ Reliability
  □ Health checks configured
  □ Circuit breakers implemented
  □ Retry logic with backoff
  □ Graceful degradation paths
  □ Fallback model available

□ Observability
  □ Metrics collection active
  □ Distributed tracing enabled
  □ Log aggregation working
  □ Alerts configured
  □ Dashboards created

□ Operations
  □ Blue-green deployment ready
  □ Rollback procedure documented
  □ On-call runbooks created
  □ Capacity plan documented
  □ Cost monitoring active

□ Compliance
  □ Data retention policies applied
  □ Privacy impact assessment done
  □ Model card documented
  □ Bias audit completed
  □ Explainability requirements met
```

### 24.4.2 Incident Response Template

```markdown
## Incident Report

**Date:** [YYYY-MM-DD]
**Duration:** [HH:MM - HH:MM]
**Severity:** [P0/P1/P2/P3]
**Author:** [Name]

### Summary
[1-2 sentence summary of the incident]

### Impact
- [User impact description]
- [Duration of impact]
- [Number of affected requests]

### Timeline
- [HH:MM] Event detected
- [HH:MM] Investigation started
- [HH:MM] Root cause identified
- [HH:MM] Fix deployed
- [HH:MM] Service recovered

### Root Cause
[Technical description of what caused the incident]

### Resolution
[What was done to fix the issue]

### Prevention
- [ ] [Action item 1]
- [ ] [Action item 2]
- [ ] [Action item 3]

### Lessons Learned
[Key takeaways from the incident]
```


## 24.5 Capstone Project: Multi-Modal AI Platform

### 24.5.1 System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                 MULTI-MODAL AI PLATFORM                             │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    API Gateway (Kong)                         │  │
│  │  Auth: OAuth2/JWT | Rate Limit: 1000 RPM | CORS: Configured │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                      │
│     ┌───────────────────────┼───────────────────────┐              │
│     │                       │                       │              │
│     v                       v                       v              │
│  ┌──────────┐        ┌──────────┐           ┌──────────┐          │
│  │  Text    │        │  Image   │           │  Audio   │          │
│  │  Service │        │  Service │           │  Service │          │
│  │ (LLM)    │        │ (Stable  │           │ (Whisper │          │
│  │          │        │  Diff.)  │           │  + TTS)  │          │
│  └────┬─────┘        └────┬─────┘           └────┬─────┘          │
│       │                   │                     │                 │
│       └───────────────────┼─────────────────────┘                 │
│                           v                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Unified Vector Store (Qdrant)                   │  │
│  │  Text Embeddings | Image Embeddings | Audio Embeddings       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                       │
│                           v                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Output Processing & Delivery                    │  │
│  │  Format Conversion | Quality Check | Caching | Streaming    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### 24.5.2 Multi-Modal Router

```python
# multimodal_router.py
from typing import Dict, Optional
from enum import Enum

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class MultiModalRouter:
    def __init__(self):
        self.handlers = {}
    
    def register(self, modality: Modality, handler):
        self.handlers[modality] = handler
    
    def route(self, input_data: Dict) -> Dict:
        modality = self._detect_modality(input_data)
        handler = self.handlers.get(modality)
        
        if not handler:
            return {"error": f"No handler for modality: {modality}"}
        
        result = handler.process(input_data)
        return {
            "modality": modality.value,
            "result": result,
        }
    
    def _detect_modality(self, data: Dict) -> Modality:
        if "image" in data or "image_url" in data:
            return Modality.IMAGE
        elif "audio" in data or "audio_url" in data:
            return Modality.AUDIO
        return Modality.TEXT
```
