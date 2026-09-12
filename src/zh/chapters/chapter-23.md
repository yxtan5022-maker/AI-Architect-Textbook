# 第23章：AI测试与验证

**读者级别：** 🔴 高级 | **页数：** 28 | **代码：** Python, pytest

---

## 23.1 AI测试金字塔

### 23.1.1 测试层级

```
                    ┌───────────────┐
                    │   端到端测试   │  <- 少量、慢、昂贵
                    │   (5-10%)     │
                    ├───────────────┤
                    │   集成测试     │  <- 适中、测试流水线
                    │   (20-30%)    │
                    ├───────────────┤
                    │   单元测试     │  <- 大量、快速、便宜
                    │ (60-70%)      │
                    └───────────────┘
```

### 23.1.2 AI特定测试类别

| 类别 | 测试内容 | 频率 | 成本 |
|------|---------|------|------|
| 单元测试 | 单个组件 | 每次提交 | 低 |
| 集成测试 | 流水线阶段协作 | 每次PR | 中 |
| 质量测试 | 模型输出质量 | 每晚 | 高 |
| 回归测试 | 无质量退化 | 每周 | 高 |
| 红队测试 | 安全和安全 | 每月 | 很高 |
| 负载测试 | 压力下性能 | 发布前 | 高 |

## 23.2 单元测试AI组件

### 23.2.1 测试分词器

```python
# test_tokenizer.py
import pytest

class TestTokenizer:
    def test_basic_tokenization(self, tokenizer):
        tokens = tokenizer.encode("Hello, world!")
        assert len(tokens) > 0
        assert tokens[0] == tokenizer.bos_token_id
    
    def test_empty_input(self, tokenizer):
        tokens = tokenizer.encode("")
        assert tokens == [tokenizer.bos_token_id, tokenizer.eos_token_id]
    
    def test_batch_encoding(self, tokenizer):
        batch = tokenizer(["Hello", "World"], padding=True)
        assert batch["input_ids"].shape[0] == 2
```

### 23.2.2 测试模型推理

```python
# test_model_inference.py
import pytest
import torch

class TestModelInference:
    def test_output_shape(self, model, sample_input):
        with torch.no_grad():
            output = model(**sample_input)
        assert output.logits.shape[0] == sample_input["input_ids"].shape[0]
    
    def test_deterministic_output(self, model, sample_input):
        model.eval()
        with torch.no_grad():
            out1 = model(**sample_input).logits
            out2 = model(**sample_input).logits
        assert torch.allclose(out1, out2, atol=1e-6)
    
    def test_gradient_flow(self, model, sample_input):
        model.train()
        output = model(**sample_input)
        loss = output.logits.mean()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
```

### 23.2.3 测试RAG组件

```python
# test_rag.py
import pytest

class TestRAGPipeline:
    def test_retrieval_relevance(self, retriever, query, expected_docs):
        results = retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.id for r in results]
        overlap = len(set(retrieved_ids) & set(expected_docs))
        recall_at_5 = overlap / len(expected_docs)
        assert recall_at_5 >= 0.6
    
    def test_context_length_limit(self, pipeline, long_query):
        result = pipeline(long_query)
        assert len(result.context_tokens) <= pipeline.max_context_length
```

## 23.3 质量评估框架

### 23.3.1 LLM作为评判

```python
# llm_judge.py
class LLMJudge:
    def __init__(self, judge_model):
        self.model = judge_model
    
    def evaluate(self, question, response, reference):
        prompt = f"""评分（1-5）：

问题：{question}
回复：{response}
参考：{reference}

分数（1-5）："""
        
        score = self.model.generate(prompt, max_tokens=1)
        return {"quality_score": int(score)}
    
    def pairwise_compare(self, question, response_a, response_b):
        prompt = f"""比较这两个回复：

问题：{question}
回复A：{response_a}
回复B：{response_b}

哪个更好？（A/B/平局）："""
        
        winner = self.model.generate(prompt, max_tokens=1)
        return {"winner": winner}
```

### 23.3.2 基准评估

```python
# benchmark_eval.py
import json

class BenchmarkEvaluator:
    def __init__(self, benchmark_path):
        with open(benchmark_path) as f:
            self.data = json.load(f)
    
    def evaluate(self, model_fn, metrics):
        results = {}
        for item in self.data:
            prediction = model_fn(item["question"])
            for metric in metrics:
                if metric not in results:
                    results[metric] = []
                score = self._compute_metric(metric, prediction, item["answer"])
                results[metric].append(score)
        return {
            metric: sum(scores) / len(scores)
            for metric, scores in results.items()
        }
    
    def _compute_metric(self, metric, pred, ref):
        if metric == "exact_match":
            return 1.0 if pred.strip() == ref.strip() else 0.0
        return 0.0
```

## 23.4 回归测试

### 23.4.1 黄金数据集管理

```python
# golden_dataset.py
import json

class GoldenDataset:
    def __init__(self, path):
        self.path = path
        self.cases = []
        self.load()
    
    def load(self):
        try:
            with open(self.path) as f:
                self.cases = json.load(f)
        except FileNotFoundError:
            self.cases = []
    
    def add_case(self, input_text, expected_output, category):
        self.cases.append({
            "id": len(self.cases) + 1,
            "input": input_text,
            "expected": expected_output,
            "category": category,
        })
    
    def run_regression(self, model_fn):
        results = {"pass": 0, "fail": 0}
        for case in self.cases:
            actual = model_fn(case["input"])
            if actual.strip() == case["expected"].strip():
                results["pass"] += 1
            else:
                results["fail"] += 1
        results["pass_rate"] = results["pass"] / len(self.cases) if self.cases else 0
        return results
```

## 23.5 负载和压力测试

### 23.5.1 负载测试框架

```python
# load_test.py
import asyncio
import time

class LoadTester:
    def __init__(self, concurrent_users, requests_per_user):
        self.concurrent_users = concurrent_users
        self.requests_per_user = requests_per_user
        self.results = []
    
    async def run(self, request_fn):
        async def user_session():
            for _ in range(self.requests_per_user):
                start = time.perf_counter()
                await request_fn()
                self.results.append(time.perf_counter() - start)
        
        tasks = [user_session() for _ in range(self.concurrent_users)]
        await asyncio.gather(*tasks)
    
    def get_report(self):
        if not self.results:
            return {"error": "no results"}
        return {
            "total_requests": len(self.results),
            "p50_ms": sorted(self.results)[len(self.results)//2] * 1000,
        }
```

### 23.5.2 压力测试场景

| 场景 | 目标负载 | 持续时间 | 成功标准 |
|------|---------|---------|---------|
| 正常负载 | 100 RPS | 10分钟 | p99 < 500ms，错误 < 1% |
| 峰值负载 | 500 RPS | 5分钟 | p99 < 1s，错误 < 2% |
| 压力测试 | 1000 RPS | 5分钟 | p99 < 2s，错误 < 5% |
| 尖峰测试 | 2000 RPS | 1分钟 | 30秒内恢复 |
| 耐力测试 | 100 RPS | 24小时 | 无内存泄漏 |
