# Chapter 23: AI Testing and Validation

**Reader Level:** 🔴 Advanced | **Pages:** 28 | **Code:** Python, pytest

---

## 23.1 The AI Testing Pyramid

### 23.1.1 Testing Levels

```
                    ┌───────────────┐
                    │   E2E Tests   │  <- Few, slow, expensive
                    │   (5-10%)     │
                    ├───────────────┤
                    │ Integration   │  <- Moderate, test pipelines
                    │   (20-30%)    │
                    ├───────────────┤
                    │ Unit Tests    │  <- Many, fast, cheap
                    │ (60-70%)      │
                    └───────────────┘
```

### 23.1.2 AI-Specific Test Categories

| Category | What It Tests | Frequency | Cost |
|----------|--------------|-----------|------|
| Unit Tests | Individual components | Every commit | Low |
| Integration Tests | Pipeline stages together | Every PR | Medium |
| Quality Tests | Model output quality | Nightly | High |
| Regression Tests | No quality degradation | Weekly | High |
| Red Team Tests | Security and safety | Monthly | Very High |
| Load Tests | Performance under stress | Pre-release | High |


## 23.2 Unit Testing AI Components

### 23.2.1 Testing Tokenization

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
        batch = tokenizer(["Hello", "World"], padding=True, truncation=True)
        assert batch["input_ids"].shape[0] == 2
        assert batch["attention_mask"].shape == batch["input_ids"].shape
```

### 23.2.2 Testing Model Inference

```python
# test_model_inference.py
import pytest
import torch

class TestModelInference:
    def test_output_shape(self, model, sample_input):
        with torch.no_grad():
            output = model(**sample_input)
        assert output.logits.shape[0] == sample_input["input_ids"].shape[0]
        assert output.logits.shape[1] <= model.config.max_position_embeddings
    
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

### 23.2.3 Testing RAG Components

```python
# test_rag.py
import pytest

class TestRAGPipeline:
    def test_retrieval_relevance(self, retriever, query, expected_docs):
        results = retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.id for r in results]
        overlap = len(set(retrieved_ids) & set(expected_docs))
        recall_at_5 = overlap / len(expected_docs)
        assert recall_at_5 >= 0.6, f"Recall@5 too low: {recall_at_5}"
    
    def test_reranker_stability(self, reranker, query, documents):
        ranked1 = reranker.rerank(query, documents)
        ranked2 = reranker.rerank(query, documents)
        assert ranked1 == ranked2, "Reranker should be deterministic"
    
    def test_context_length_limit(self, pipeline, long_query):
        result = pipeline(long_query)
        assert len(result.context_tokens) <= pipeline.max_context_length
```


## 23.3 Quality Evaluation Framework

### 23.3.1 Evaluation Metrics Suite

```python
# quality_evaluator.py
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class EvalResult:
    metric_name: str
    score: float
    details: Dict

class QualityEvaluator:
    def __init__(self):
        self.evaluators = {}
    
    def register(self, name: str, fn):
        self.evaluators[name] = fn
    
    def evaluate(self, outputs: List[str], references: List[str]) -> Dict:
        results = {}
        for name, fn in self.evaluators.items():
            results[name] = fn(outputs, references)
        return results

# Example evaluators
def exact_match(outputs, refs):
    matches = sum(o.strip() == r.strip() for o, r in zip(outputs, refs))
    return matches / len(outputs) if outputs else 0

def contains_answer(outputs, refs):
    matches = sum(r.lower() in o.lower() for o, r in zip(outputs, refs))
    return matches / len(outputs) if outputs else 0

def average_length(outputs, refs=None):
    return sum(len(o) for o in outputs) / len(outputs) if outputs else 0
```

### 23.3.2 LLM-as-Judge Evaluation

```python
# llm_judge.py
from typing import Dict

class LLMJudge:
    def __init__(self, judge_model):
        self.model = judge_model
    
    def evaluate(self, question: str, response: str, reference: str) -> Dict:
        prompt = f"""Rate this response on a scale of 1-5:

Question: {question}
Response: {response}
Reference: {reference}

Score (1-5): """
        
        score = self.model.generate(prompt, max_tokens=1)
        return {
            "quality_score": int(score),
            "prompt_used": prompt,
        }
    
    def pairwise_compare(self, question: str, response_a: str, response_b: str) -> Dict:
        prompt = f"""Compare these two responses:

Question: {question}
Response A: {response_a}
Response B: {response_b}

Which is better? (A/B/Tie): """
        
        winner = self.model.generate(prompt, max_tokens=1)
        return {
            "winner": winner,
            "prompt_used": prompt,
        }
```

### 23.3.3 Benchmark Evaluation

```python
# benchmark_eval.py
from typing import Dict, List
import json

class BenchmarkEvaluator:
    def __init__(self, benchmark_path: str):
        with open(benchmark_path) as f:
            self.data = json.load(f)
    
    def evaluate(self, model_fn, metrics: List[str]) -> Dict:
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
    
    def _compute_metric(self, metric: str, pred: str, ref: str) -> float:
        if metric == "exact_match":
            return 1.0 if pred.strip() == ref.strip() else 0.0
        elif metric == "contains":
            return 1.0 if ref.lower() in pred.lower() else 0.0
        return 0.0
```


## 23.4 Regression Testing

### 23.4.1 Golden Dataset Management

```python
# golden_dataset.py
import json
from typing import Dict, List
from pathlib import Path

class GoldenDataset:
    def __init__(self, path: str):
        self.path = Path(path)
        self.cases: List[Dict] = []
        self.load()
    
    def load(self):
        if self.path.exists():
            with open(self.path) as f:
                self.cases = json.load(f)
    
    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.cases, f, indent=2)
    
    def add_case(self, input_text: str, expected_output: str,
                 category: str, tags: List[str] = None):
        self.cases.append({
            "id": len(self.cases) + 1,
            "input": input_text,
            "expected": expected_output,
            "category": category,
            "tags": tags or [],
        })
    
    def run_regression(self, model_fn) -> Dict:
        results = {"pass": 0, "fail": 0, "details": []}
        for case in self.cases:
            actual = model_fn(case["input"])
            passed = actual.strip() == case["expected"].strip()
            if passed:
                results["pass"] += 1
            else:
                results["fail"] += 1
                results["details"].append({
                    "id": case["id"],
                    "input": case["input"][:100],
                    "expected": case["expected"][:100],
                    "actual": actual[:100],
                })
        results["pass_rate"] = results["pass"] / len(self.cases) if self.cases else 0
        return results
```

### 23.4.2 Snapshot Testing

```python
# snapshot_test.py
import json
from pathlib import Path

class SnapshotTester:
    def __init__(self, snapshot_dir: str):
        self.dir = Path(snapshot_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
    
    def test_or_compare(self, test_name: str, output: str) -> Dict:
        snapshot_path = self.dir / f"{test_name}.txt"
        
        if snapshot_path.exists():
            expected = snapshot_path.read_text()
            if output.strip() == expected.strip():
                return {"status": "pass", "match": True}
            else:
                return {
                    "status": "fail",
                    "match": False,
                    "expected": expected[:200],
                    "actual": output[:200],
                }
        else:
            snapshot_path.write_text(output)
            return {"status": "created", "match": True}
    
    def update_snapshot(self, test_name: str, new_output: str):
        snapshot_path = self.dir / f"{test_name}.txt"
        snapshot_path.write_text(new_output)
```


## 23.5 Load and Stress Testing

### 23.5.1 Load Testing Framework

```python
# load_test.py
import asyncio
import time
from typing import List
from dataclasses import dataclass

@dataclass
class LoadTestConfig:
    concurrent_users: int
    requests_per_user: int
    ramp_up_seconds: int
    target_rps: float

class LoadTester:
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[float] = []
    
    async def run(self, request_fn):
        semaphore = asyncio.Semaphore(self.config.concurrent_users)
        
        async def user_session():
            async with semaphore:
                for _ in range(self.config.requests_per_user):
                    start = time.perf_counter()
                    await request_fn()
                    elapsed = time.perf_counter() - start
                    self.results.append(elapsed)
                    await asyncio.sleep(1.0 / self.config.target_rps)
        
        tasks = [user_session() for _ in range(self.config.concurrent_users)]
        await asyncio.gather(*tasks)
    
    def get_report(self) -> dict:
        if not self.results:
            return {"error": "no results"}
        
        sorted_r = sorted(self.results)
        n = len(sorted_r)
        return {
            "total_requests": n,
            "p50_ms": sorted_r[n // 2] * 1000,
            "p95_ms": sorted_r[int(n * 0.95)] * 1000,
            "p99_ms": sorted_r[int(n * 0.99)] * 1000,
            "throughput_rps": n / sum(self.results) if self.results else 0,
        }
```

### 23.5.2 Stress Testing Scenarios

| Scenario | Target Load | Duration | Success Criteria |
|----------|------------|----------|-----------------|
| Normal Load | 100 RPS | 10 min | p99 < 500ms, errors < 1% |
| Peak Load | 500 RPS | 5 min | p99 < 1s, errors < 2% |
| Stress Test | 1000 RPS | 5 min | p99 < 2s, errors < 5% |
| Spike Test | 2000 RPS | 1 min | Recovery within 30s |
| Endurance | 100 RPS | 24 hours | No memory leaks |

### 23.5.3 Performance Benchmarking Suite

```python
# perf_benchmark.py
import time
from typing import Dict, List

class PerformanceBenchmark:
    def __init__(self, model_fn):
        self.model_fn = model_fn
    
    def benchmark_latency(self, input_text: str, runs: int = 100) -> Dict:
        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            self.model_fn(input_text)
            latencies.append(time.perf_counter() - start)
        
        return {
            "mean_ms": sum(latencies) / len(latencies) * 1000,
            "min_ms": min(latencies) * 1000,
            "max_ms": max(latencies) * 1000,
        }
    
    def benchmark_throughput(self, inputs: List[str], duration: float = 10.0) -> Dict:
        count = 0
        start = time.time()
        while time.time() - start < duration:
            self.model_fn(inputs[count % len(inputs)])
            count += 1
        
        return {
            "requests_total": count,
            "rps": count / duration,
        }
```
