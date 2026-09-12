# 第24章：实践与综合

**读者级别：** 🔴 高级 | **页数：** 30 | **代码：** Python, Docker, Kubernetes

---

## 24.1 端到端项目：智能文档处理

### 24.1.1 项目概述

构建一个生产级文档处理流水线，从混合格式文档（PDF、图像、电子邮件）中提取、分类和汇总信息。

```
┌────────────────────────────────────────────────────────────────┐
│              文档处理流水线                                      │
│                                                                 │
│  输入文档                                                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                          │
│  │   PDF   │ │  图像   │ │  邮件   │                          │
│  └────┬────┘ └────┬────┘ └────┬────┘                          │
│       │           │           │                                │
│       v           v           v                                │
│  ┌─────────────────────────────────────┐                       │
│  │     文档解析器 + OCR                │                       │
│  │  (Tesseract + 布局分析)             │                       │
│  └──────────────────┬──────────────────┘                       │
│                     v                                          │
│  ┌─────────────────────────────────────┐                       │
│  │     内容提取                        │                       │
│  │  (NER + 表格检测 + OCR)            │                       │
│  └──────────────────┬──────────────────┘                       │
│           ┌─────────┴─────────┐                               │
│           v                   v                                │
│  ┌─────────────────┐ ┌─────────────────┐                      │
│  │   分类器        │ │   摘要器        │                      │
│  │  (BERT)         │ │  (LLM)          │                      │
│  └────────┬────────┘ └────────┬────────┘                      │
│           └─────────┬─────────┘                               │
│                     v                                          │
│  ┌─────────────────────────────────────┐                       │
│  │        输出存储                      │                       │
│  │  (结构化JSON + 向量存储)            │                       │
│  └─────────────────────────────────────┘                       │
└────────────────────────────────────────────────────────────────┘
```

### 24.1.2 文档解析器实现

```python
# document_parser.py
from dataclasses import dataclass
from typing import Dict, List

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
    
    def parse(self, file_path):
        doc_type = self._detect_type(file_path)
        return self.parsers.get(doc_type, self._parse_generic)(file_path)
    
    def _detect_type(self, path):
        if path.endswith(".pdf"):
            return "pdf"
        elif path.endswith((".png", ".jpg")):
            return "image"
        elif path.endswith((".eml", ".msg")):
            return "email"
        return "generic"
    
    def _parse_pdf(self, path):
        return ParsedDocument(
            doc_id=path, doc_type="pdf",
            text_content="[PDF内容]", tables=[], metadata={}, confidence=0.95,
        )
    
    def _parse_image(self, path):
        return ParsedDocument(
            doc_id=path, doc_type="image",
            text_content="[OCR结果]", tables=[], metadata={}, confidence=0.80,
        )
    
    def _parse_email(self, path):
        return ParsedDocument(
            doc_id=path, doc_type="email",
            text_content="[邮件内容]", tables=[], metadata={}, confidence=0.90,
        )
    
    def _parse_generic(self, path):
        with open(path) as f:
            text = f.read()
        return ParsedDocument(
            doc_id=path, doc_type="generic",
            text_content=text, tables=[], metadata={}, confidence=0.70,
        )
```

## 24.2 端到端项目：对话式AI助手

### 24.2.1 架构概览

```
┌────────────────────────────────────────────────────────────────┐
│              对话式AI助手                                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │                  API网关                              │      │
│  │  (认证 + 速率限制 + 请求路由)                        │      │
│  └──────────────────┬───────────────────────────────────┘      │
│                     v                                          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              对话管理器                               │      │
│  │  • 会话状态管理                                      │      │
│  │  • 上下文窗口管理                                    │      │
│  │  • 多轮记忆                                          │      │
│  └──────────────────┬───────────────────────────────────┘      │
│        ┌────────────┼────────────┐                            │
│        v            v            v                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
│  │  RAG     │ │  工具    │ │  安全    │                      │
│  │  引擎    │ │  路由器  │ │  过滤器  │                      │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                      │
│       └─────────────┼────────────┘                            │
│                     v                                          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              响应生成器                               │      │
│  │  (流式 + 引用 + 格式)                                │      │
│  └──────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────┘
```

### 24.2.2 对话状态管理

```python
# conversation_manager.py
from typing import Dict, List
from dataclasses import dataclass, field
import uuid

@dataclass
class Message:
    role: str
    content: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class ConversationSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: List[Message] = field(default_factory=list)
    max_context_tokens: int = 4096

class ConversationManager:
    def __init__(self, max_history=50):
        self.sessions = {}
        self.max_history = max_history
    
    def get_or_create_session(self, session_id=None):
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        session = ConversationSession(
            session_id=session_id or uuid.uuid4().hex
        )
        self.sessions[session.session_id] = session
        return session
    
    def add_message(self, session_id, role, content):
        session = self.get_or_create_session(session_id)
        session.messages.append(Message(role=role, content=content))
        if len(session.messages) > self.max_history:
            session.messages = session.messages[-self.max_history:]
    
    def get_context_window(self, session_id):
        session = self.get_or_create_session(session_id)
        messages = []
        total_tokens = 0
        for msg in reversed(session.messages):
            est_tokens = len(msg.content.split()) * 1.3
            if total_tokens + est_tokens > session.max_context_tokens:
                break
            messages.insert(0, {"role": msg.role, "content": msg.content})
            total_tokens += est_tokens
        return messages
```

## 24.3 部署模式

### 24.3.1 蓝绿部署

```
┌────────────────────────────────────────────────────────────────┐
│              蓝绿部署                                           │
│                                                                 │
│  负载均衡器                                                     │
│       │                                                         │
│       ├──────────────> 蓝（当前）─── 模型 v1.0                │
│       │                   │                                     │
│       │                   v                                     │
│       │              [健康检查]                                 │
│       │                                                         │
│       └────（切换）──> 绿（新）─── 模型 v2.0                  │
│                             │                                   │
│                             v                                   │
│                        [健康检查]                               │
│                                                                 │
│  回滚：<1分钟将流量切回蓝色                                     │
└────────────────────────────────────────────────────────────────┘
```

### 24.3.2 金丝雀部署配置

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
```

## 24.4 生产清单

### 24.4.1 发布前清单

```
□ 模型性能
  □ 基准结果满足质量阈值
  □ 延迟要求满足（p50/p99）
  □ 吞吐量能力已验证
  □ 内存使用在预算内

□ 安全
  □ 输入验证已实现
  □ 速率限制已配置
  □ 认证/授权工作正常
  □ PII检测已激活
  □ 审计日志已启用

□ 可靠性
  □ 健康检查已配置
  □ 熔断器已实现
  □ 重试逻辑带回退
  □ 优雅降级路径
  □ 备用模型可用

□ 可观测性
  □ 指标收集已激活
  □ 分布式追踪已启用
  □ 日志聚合工作正常
  □ 告警已配置
  □ 仪表板已创建

□ 运维
  □ 蓝绿部署就绪
  □ 回滚流程已文档化
  □ 值班手册已创建
  □ 容量计划已文档化
  □ 成本监控已激活
```

## 24.5 综合项目：多模态AI平台

### 24.5.1 系统架构

```
┌────────────────────────────────────────────────────────────────────┐
│                 多模态AI平台                                        │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    API网关 (Kong)                             │  │
│  │  认证: OAuth2/JWT | 速率限制: 1000 RPM | CORS: 已配置       │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│     ┌───────────────────────┼───────────────────────┐              │
│     v                       v                       v              │
│  ┌──────────┐        ┌──────────┐           ┌──────────┐          │
│  │  文本    │        │  图像    │           │  音频    │          │
│  │  服务    │        │  服务    │           │  服务    │          │
│  │ (LLM)    │        │ (Stable  │           │ (Whisper │          │
│  │          │        │  Diff.)  │           │  + TTS)  │          │
│  └────┬─────┘        └────┬─────┘           └────┬─────┘          │
│       └───────────────────┼─────────────────────┘                 │
│                           v                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              统一向量存储 (Qdrant)                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### 24.5.2 多模态路由器

```python
# multimodal_router.py
from enum import Enum

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"

class MultiModalRouter:
    def __init__(self):
        self.handlers = {}
    
    def register(self, modality, handler):
        self.handlers[modality] = handler
    
    def route(self, input_data):
        modality = self._detect_modality(input_data)
        handler = self.handlers.get(modality)
        if not handler:
            return {"error": f"无处理程序: {modality}"}
        return {"modality": modality.value, "result": handler.process(input_data)}
    
    def _detect_modality(self, data):
        if "image" in data:
            return Modality.IMAGE
        elif "audio" in data:
            return Modality.AUDIO
        return Modality.TEXT
```
