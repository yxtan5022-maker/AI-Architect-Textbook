# 第2章：AI 系统设计原则

## 学习目标

学完本章后，你将能够：

- 应用 AI 系统特有的可扩展性原则
- 设计适应需求变化的可维护 ML 架构
- 在不牺牲质量的情况下实现具有成本效益的 AI 解决方案
- 在每一层都考虑安全和隐私
- 设计在问题影响用户之前就能检测到问题的可观测性系统
- 应对 AI 系统特有的设计权衡

---

## 2.1 可扩展性原则

### 2.1.1 理解 AI 可扩展性

AI 系统的可扩展性与传统软件可扩展性有根本不同。Web 应用通过处理更多并发用户或事务来扩展。AI 系统通过处理更多数据、更多模型、更多实验和更复杂的工作流来扩展。

**数据可扩展性**：处理不断增加的训练数据量而不按比例增加成本或时间的能力。这不仅包括存储容量，还包括更快处理数据、支持更多特征类型以及在大规模下保持数据质量的能力。

**模型可扩展性**：训练和提供更多复杂模型（更多参数、更多特征、更多输出）的能力。这涵盖了计算资源需求以及管理模型复杂度的组织能力。

**运营可扩展性**：用现有团队规模管理越来越多模型、实验和部署的能力。这通常是最被忽视的维度——组织可以扩展计算但不能扩展人员。

**业务可扩展性**：将相同架构应用于新用例而只需最少修改的能力。这需要深思熟虑的抽象和模块化设计。

### 2.1.2 数据可扩展性模式

**模式：Lambda 架构**

Lambda 架构通过批处理和速度层处理数据，在不同延迟级别提供全面视图。

```python
# 示例：用于 ML 特征工程的 Lambda 架构
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
from abc import ABC, abstractmethod

class DataLayer(ABC):
    """数据处理层的抽象基类"""
    
    @abstractmethod
    def process(self, data_source: str, **kwargs) -> Dict[str, Any]:
        pass

class BatchLayer(DataLayer):
    """处理历史数据以获得全面特征"""
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
    
    def process(self, data_source: str, start_date: datetime, 
                end_date: datetime) -> Dict[str, Any]:
        """运行批处理特征计算（每日/每周）"""
        raw_data = self.storage.read_range(data_source, start_date, end_date)
        
        # 计算需要完整历史上下文的复杂特征
        features = {
            'user_lifetime_value': self._compute_ltv(raw_data),
            'product_popularity_score': self._compute_popularity(raw_data),
            'user_segment_clusters': self._compute_segments(raw_data)
        }
        
        # 存储在批处理特征存储中
        self.storage.write('batch_features', features)
        return features
    
    def _compute_ltv(self, data: Any) -> float:
        """需要完整购买历史——只能在批处理中完成"""
        return 0.0
    
    def _compute_popularity(self, data: Any) -> float:
        """计算全局流行度指标"""
        return 0.0
    
    def _compute_segments(self, data: Any) -> Dict:
        """使用聚类计算用户细分"""
        return {}

class SpeedLayer(DataLayer):
    """处理流数据以获得实时特征"""
    
    def __init__(self, stream_processor, feature_store):
        self.processor = stream_processor
        self.feature_store = feature_store
    
    def process(self, event: Dict) -> Dict[str, Any]:
        """处理单个事件以获得实时特征"""
        # 计算需要当前状态的特征
        features = {
            'session_duration': self._compute_session_duration(event),
            'click_velocity': self._compute_click_rate(event),
            'real_time_rank': self._compute_real_time_rank(event)
        }
        
        # 立即更新在线特征存储
        self.feature_store.update(event['user_id'], features)
        return features
    
    def _compute_session_duration(self, event: Dict) -> float:
        """计算当前会话时长"""
        return 0.0
    
    def _compute_click_rate(self, event: Dict) -> float:
        """计算实时点击率"""
        return 0.0
    
    def _compute_real_time_rank(self, event: Dict) -> float:
        """计算实时排名分数"""
        return 0.0

class ServingLayer:
    """组合批处理和速度特征进行预测"""
    
    def __init__(self, batch_store, online_store, model):
        self.batch_store = batch_store
        self.online_store = online_store
        self.model = model
    
    def predict(self, user_id: str, context: Dict) -> Dict:
        """合并两个层的特征"""
        # 获取预计算的批处理特征
        batch_features = self.batch_store.get(user_id)
        
        # 获取实时特征
        realtime_features = self.online_store.get(user_id)
        
        # 架构决策：合并策略
        merged = self._merge_features(
            batch_features, 
            realtime_features,
            merge_strategy='priority'
        )
        
        return self.model.predict(merged)
    
    def _merge_features(self, batch_features: Dict, 
                       realtime_features: Dict,
                       merge_strategy: str = 'priority') -> Dict:
        """根据策略合并特征"""
        if merge_strategy == 'priority':
            # 实时可用时覆盖批处理
            merged = {**batch_features, **realtime_features}
        elif merge_strategy == 'concatenate':
            # 简单拼接
            merged = {**batch_features}
            for key, value in realtime_features.items():
                merged[f"rt_{key}"] = value
        else:
            merged = batch_features
        return merged
```

**模式：Kappa 架构**

对于更简单的系统，Kappa 架构通过单个流处理层处理所有数据，简化运营但要求所有特征都可以从流数据计算。

### 2.1.3 模型可扩展性模式

**模式：带版本控制的模型注册表**

```python
# 示例：模型注册表架构
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import hashlib

@dataclass
class ModelVersion:
    version: str
    model_path: str
    metrics: Dict[str, float]
    training_data_hash: str
    hyperparameters: Dict[str, Any]
    created_at: datetime
    status: str  # "staging", "production", "archived"
    tags: Dict[str, str] = field(default_factory=dict)

class ModelRegistry:
    """模型版本控制和管理的中心注册表"""
    
    def __init__(self, storage_backend, metadata_store):
        self.storage = storage_backend
        self.metadata = metadata_store
    
    def register_model(self, 
                      model_name: str,
                      version: str,
                      model_artifact: bytes,
                      metrics: Dict[str, float],
                      config: Dict) -> ModelVersion:
        """注册新模型版本"""
        
        # 存储模型产物
        model_path = f"models/{model_name}/{version}/model.pkl"
        self.storage.write(model_path, model_artifact)
        
        # 创建版本记录
        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            metrics=metrics,
            training_data_hash=config['data_hash'],
            hyperparameters=config['hyperparameters'],
            created_at=datetime.now(),
            status='staging'
        )
        
        # 存储元数据
        self.metadata.save_version(model_name, model_version)
        
        return model_version
    
    def promote_to_production(self, model_name: str, version: str,
                             validation_results: Dict) -> bool:
        """验证后提升模型到生产环境"""
        
        version_info = self.metadata.get_version(model_name, version)
        
        # 架构决策：验证门控
        if not self._validate_promotion(version_info, validation_results):
            return False
        
        # 降级当前生产模型
        current_prod = self.metadata.get_production_version(model_name)
        if current_prod:
            current_prod.status = 'archived'
            self.metadata.save_version(model_name, current_prod)
        
        # 提升新版本
        version_info.status = 'production'
        self.metadata.save_version(model_name, version_info)
        
        return True
    
    def rollback(self, model_name: str) -> Optional[ModelVersion]:
        """回滚到先前的生产版本"""
        
        versions = self.metadata.get_all_versions(model_name)
        current_prod = self.metadata.get_production_version(model_name)
        
        previous_prod = None
        for v in sorted(versions, key=lambda x: x.created_at, reverse=True):
            if v.version != current_prod.version and v.status == 'archived':
                previous_prod = v
                break
        
        if previous_prod:
            return self.promote_to_production(
                model_name, 
                previous_prod.version,
                validation_results={'rollback': True}
            )
        
        return None
```

### 2.1.4 运营可扩展性

**模式：多租户模型服务**

```python
# 示例：多租户模型服务
from typing import Dict, Optional
import asyncio
import time

class MultiTenantModelServer:
    """提供具有资源隔离的多模型服务"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.resource_limits: Dict[str, Dict] = {}
        self.request_queues: Dict[str, asyncio.Queue] = {}
        self.metrics: Dict[str, list] = {}
    
    def register_tenant(self, tenant_id: str, model: Any,
                       resource_limits: Dict):
        """注册具有资源限制的新租户"""
        self.models[tenant_id] = model
        self.resource_limits[tenant_id] = resource_limits
        self.request_queues[tenant_id] = asyncio.Queue(
            maxsize=resource_limits.get('max_queue_size', 1000)
        )
        self.metrics[tenant_id] = []
    
    async def predict(self, tenant_id: str, input_data: Dict) -> Dict:
        """提供具有租户隔离的预测"""
        
        if tenant_id not in self.models:
            raise ValueError(f"未知租户: {tenant_id}")
        
        # 检查队列容量（速率限制）
        queue = self.request_queues[tenant_id]
        if queue.full():
            raise RateLimitError(f"租户 {tenant_id} 队列已满")
        
        # 添加到队列
        await queue.put({
            'input': input_data,
            'timestamp': time.time()
        })
        
        # 在资源限制内处理
        model = self.models[tenant_id]
        limits = self.resource_limits[tenant_id]
        
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                model.predict(input_data),
                timeout=limits.get('timeout_seconds', 30)
            )
            
            # 记录指标
            latency = time.time() - start_time
            self.metrics[tenant_id].append({
                'latency': latency,
                'success': True,
                'timestamp': time.time()
            })
            
            return result
        except asyncio.TimeoutError:
            latency = time.time() - start_time
            self.metrics[tenant_id].append({
                'latency': latency,
                'success': False,
                'error': 'timeout',
                'timestamp': time.time()
            })
            raise TimeoutError(f"租户 {tenant_id} 预测超时")
    
    def get_tenant_metrics(self, tenant_id: str) -> Dict:
        """获取特定租户的指标"""
        if tenant_id not in self.metrics:
            return {}
        
        tenant_metrics = self.metrics[tenant_id]
        if not tenant_metrics:
            return {}
        
        return {
            'total_requests': len(tenant_metrics),
            'avg_latency': sum(m['latency'] for m in tenant_metrics) / len(tenant_metrics),
            'success_rate': sum(1 for m in tenant_metrics if m['success']) / len(tenant_metrics),
            'p95_latency': sorted([m['latency'] for m in tenant_metrics])[int(len(tenant_metrics) * 0.95)]
        }
```

---

## 2.2 可维护性原则

### 2.2.1 ML 系统中的代码可维护性

ML 代码库面临独特的可维护性挑战。同一个项目包含数据处理代码、模型训练代码、服务代码和监控代码。每个都有不同的测试需求、不同的故障模式和不同的演进模式。

**原则：关注点分离**

将代码库分成具有清晰接口的不同层：

```
# 分层 ML 架构
# 
# 第 1 层：数据层（处理数据加载、验证、转换）
# 第 2 层：特征层（特征工程、选择、验证）
# 第 3 层：模型层（训练、评估、选择）
# 第 4 层：服务层（推理、批处理、缓存）
# 第 5 层：监控层（指标、告警、仪表板）

# 目录结构：
# data/
#   ├── ingestion.py      # 原始数据加载
#   ├── validation.py     # 数据质量检查
#   └── transformation.py # 数据预处理
# features/
#   ├── engineering.py    # 特征创建
#   ├── selection.py      # 特征重要性分析
#   └── store.py          # 特征存储集成
# models/
#   ├── training.py       # 模型训练逻辑
#   ├── evaluation.py     # 模型评估
#   └── registry.py       # 模型版本控制
# serving/
#   ├── api.py            # API 端点
#   ├── pipeline.py       # 预测管道
#   └── cache.py          # 结果缓存
# monitoring/
#   ├── metrics.py        # 指标收集
#   ├── drift.py          # 漂移检测
#   └── alerts.py         # 告警规则
```

**原则：配置优于代码**

ML 系统有许多配置点（超参数、特征配置、部署设置）。将配置从代码中外部化。

```python
# 示例：ML 配置管理
from pydantic import BaseModel, validator
from typing import Dict, List, Optional
import yaml

class ModelConfig(BaseModel):
    """模型训练的类型安全配置"""
    
    # 模型架构
    model_type: str
    hidden_layers: List[int]
    dropout_rate: float = 0.1
    
    # 训练
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # 特征
    feature_columns: List[str]
    target_column: str
    categorical_columns: List[str] = []
    
    # 验证
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    @validator('dropout_rate')
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('dropout_rate 必须在 0 和 1 之间')
        return v
    
    @validator('hidden_layers')
    def validate_hidden_layers(cls, v):
        if not all(x > 0 for x in v):
            raise ValueError('所有隐藏层大小必须为正')
        return v

class TrainingPipeline:
    """配置驱动行为的训练管道"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._build_model()
    
    def _build_model(self):
        """根据配置构建模型"""
        if self.config.model_type == 'mlp':
            return self._build_mlp()
        elif self.config.model_type == 'transformer':
            return self._build_transformer()
        else:
            raise ValueError(f"未知模型类型: {self.config.model_type}")
    
    def train(self, train_data, val_data):
        """使用配置参数训练"""
        pass
    
    def _build_mlp(self):
        """从配置构建 MLP"""
        pass
    
    def _build_transformer(self):
        """从配置构建 Transformer"""
        pass

# 用法
config = ModelConfig(**yaml.safe_load(open('config.yaml')))
pipeline = TrainingPipeline(config)
```

### 2.2.2 实验管理

ML 项目会产生许多实验。没有适当的管理，就无法理解为什么做出某些决策或重现过去的结果。

**原则：不可变实验**

每个实验应该是自包含和可重现的。

```python
# 示例：实验管理模式
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ExperimentManager:
    """管理具有完全可重现性的实验"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / 'experiments'
        self.experiments_dir.mkdir(exist_ok=True)
    
    def create_experiment(self, name: str, config: dict, 
                         data_hash: str) -> str:
        """创建具有唯一 ID 的新实验"""
        
        experiment_content = {
            'name': name,
            'config': config,
            'data_hash': data_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        experiment_id = hashlib.md5(
            json.dumps(experiment_content, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(experiment_content, f, indent=2)
        
        with open(exp_dir / 'data_hash.txt', 'w') as f:
            f.write(data_hash)
        
        return experiment_id
    
    def log_metrics(self, experiment_id: str, metrics: dict, step: int):
        """记录实验指标"""
        exp_dir = self.experiments_dir / experiment_id
        metrics_file = exp_dir / 'metrics.jsonl'
        
        with open(metrics_file, 'a') as f:
            entry = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                **metrics
            }
            f.write(json.dumps(entry) + '\n')
    
    def log_artifact(self, experiment_id: str, artifact_name: str, 
                    artifact_path: str):
        """记录产物（模型、图表等）"""
        exp_dir = self.experiments_dir / experiment_id
        artifacts_dir = exp_dir / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        
        import shutil
        dest = artifacts_dir / artifact_name
        shutil.copy2(artifact_path, dest)
        
        manifest_file = exp_dir / 'manifest.json'
        manifest = {}
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
        
        manifest[artifact_name] = {
            'path': str(dest),
            'timestamp': datetime.now().isoformat()
        }
        
        manifest_file.write_text(json.dumps(manifest, indent=2))
    
    def compare_experiments(self, experiment_ids: list) -> Dict:
        """比较多个实验"""
        comparison = {}
        
        for exp_id in experiment_ids:
            exp_dir = self.experiments_dir / exp_id
            
            # 加载配置
            with open(exp_dir / 'config.json') as f:
                config = json.load(f)
            
            # 加载指标
            metrics_file = exp_dir / 'metrics.jsonl'
            metrics = []
            if metrics_file.exists():
                with open(metrics_file) as f:
                    for line in f:
                        metrics.append(json.loads(line))
            
            comparison[exp_id] = {
                'config': config,
                'final_metrics': metrics[-1] if metrics else {},
                'total_steps': len(metrics)
            }
        
        return comparison
```

### 2.2.3 测试 ML 系统

ML 系统需要超越传统软件测试的测试策略。

**ML 测试金字塔**：

```
┌─────────────────────────────────────────┐
│         集成测试                         │
│    （端到端管道验证）                    │
├─────────────────────────────────────────┤
│           模型测试                       │
│  （性能、公平性、鲁棒性）               │
├─────────────────────────────────────────┤
│          特征测试                       │
│   （特征工程验证）                       │
├─────────────────────────────────────────┤
│           数据测试                       │
│    （模式、质量、分布）                  │
├─────────────────────────────────────────┤
│         单元测试                         │
│   （单个函数测试）                       │
└─────────────────────────────────────────┘
```

```python
# 示例：ML 特定测试
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

class TestDataQuality:
    """数据质量测试"""
    
    def test_no_missing_critical_features(self, training_data):
        """确保关键特征没有缺失值"""
        critical_features = ['user_id', 'timestamp', 'target']
        for feature in critical_features:
            assert training_data[feature].isnull().sum() == 0, \
                f"关键特征中有缺失值: {feature}"
    
    def test_feature_distributions(self, training_data, reference_data):
        """检查显著的分布偏移"""
        from scipy import stats
        
        for column in training_data.select_dtypes(include=[np.number]).columns:
            stat, p_value = stats.ks_2samp(
                training_data[column].dropna(),
                reference_data[column].dropna()
            )
            assert p_value > 0.05, \
                f"在 {column} 中检测到分布偏移: p={p_value}"

class TestModelPerformance:
    """模型质量测试"""
    
    def test_minimum_accuracy(self, model, test_data):
        """模型必须达到最低准确率阈值"""
        accuracy = model.evaluate(test_data)
        assert accuracy >= 0.7, f"模型准确率 {accuracy} 低于阈值"
    
    def test_fairness_metrics(self, model, test_data, sensitive_columns):
        """模型必须在人口统计群体间保持公平"""
        predictions = model.predict(test_data)
        
        for column in sensitive_columns:
            groups = test_data[column].unique()
            group_metrics = {}
            
            for group in groups:
                mask = test_data[column] == group
                group_pred = predictions[mask]
                group_true = test_data.loc[mask, 'target']
                
                tpr = (group_pred[group_true == 1] == 1).mean()
                group_metrics[group] = tpr
            
            max_diff = max(group_metrics.values()) - min(group_metrics.values())
            assert max_diff < 0.1, \
                f"在 {column} 中存在公平性违规: 最大差异 {max_diff}"

class TestFeatureEngineering:
    """特征管道测试"""
    
    def test_feature_types(self, feature_pipeline, sample_data):
        """确保特征具有正确类型"""
        features = feature_pipeline.transform(sample_data)
        
        expected_types = {
            'age': 'int64',
            'income': 'float64',
            'is_premium': 'bool'
        }
        
        for feature, expected_type in expected_types.items():
            assert features[feature].dtype == expected_type, \
                f"特征 {feature} 类型错误: {features[feature].dtype}"
    
    def test_feature_ranges(self, feature_pipeline, sample_data):
        """确保特征在预期范围内"""
        features = feature_pipeline.transform(sample_data)
        
        assert (features['age'] >= 0).all() and (features['age'] <= 150).all()
        assert (features['income'] >= 0).all()
```

### 2.2.4 ML 系统的文档

ML 系统需要传统软件不需要的文档：

- **数据字典**：每个特征的含义、计算方式、有效范围
- **模型卡片**：模型用途、训练数据、性能特征、限制
- **决策日志**：为什么做出某些架构和设计决策
- **运维手册**：常见场景的操作程序
- **API 文档**：如何与模型服务系统集成

> 📌 **关键概念**：ML 系统的文档不是可选的——它是安全要求。未记录的模型是负债。如果你无法解释模型如何做出决策，你就不能信任它处理重要结果。

---

## 2.3 成本效益原则

### 2.3.1 理解 AI 成本

AI 系统具有与传统软件不同的独特成本结构：

**计算成本**：训练和推理的 GPU/TPU 时间，数据处理的 CPU 时间
**存储成本**：数据存储、模型产物、实验日志
**数据成本**：数据获取、标注、清洗
**人力成本**：工程时间、ML 研究时间、运营开销
**机会成本**：在 AI vs 替代解决方案上花费的时间

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI 成本分解                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  训练成本（一次性、可重复）：                                    │
│  ├── GPU 小时数 × 实验次数                                      │
│  ├── 数据处理（ETL 管道运行）                                   │
│  └── 特征工程迭代                                               │
│                                                                  │
│  服务成本（持续）：                                              │
│  ├── 推理计算（每次预测或按时间）                               │
│  ├── 模型存储和版本控制                                         │
│  └── 监控和日志记录                                             │
│                                                                  │
│  隐性成本：                                                      │
│  ├── 数据标注和注释                                             │
│  ├── 模型监控和维护                                             │
│  ├── 快速实验产生的技术债务                                     │
│  └── 错误架构决策的机会成本                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3.2 成本优化策略

**策略：调整基础设施规模**

```python
# 示例：成本感知的基础设施选择
from dataclasses import dataclass
from typing import Dict

@dataclass
class InfrastructureOption:
    name: str
    compute_cost_per_hour: float
    memory_gb: float
    gpu_count: int
    monthly_cost: float
    
    def cost_per_prediction(self, predictions_per_month: int) -> float:
        return self.monthly_cost / predictions_per_month

class InfrastructureAdvisor:
    """帮助选择具有成本效益的基础设施"""
    
    def __init__(self, workload_profile: Dict):
        self.workload = workload_profile
    
    def recommend(self, options: list) -> InfrastructureOption:
        """根据工作负载推荐基础设施"""
        
        predictions_per_month = self.workload['predictions_per_month']
        latency_requirement = self.workload['latency_ms']
        memory_requirement = self.workload['memory_gb']
        
        suitable = []
        for option in options:
            if (option.memory_gb >= memory_requirement and
                self._meets_latency(option, latency_requirement)):
                suitable.append(option)
        
        if not suitable:
            raise ValueError("没有合适的基础设施选项")
        
        suitable.sort(key=lambda x: x.cost_per_prediction(predictions_per_month))
        
        return suitable[0]
    
    def _meets_latency(self, option: InfrastructureOption, 
                      required_latency: float) -> bool:
        """检查选项是否满足延迟要求"""
        base_latency = 100  # 毫秒
        gpu_factor = 0.5 if option.gpu_count > 0 else 1.0
        return base_latency * gpu_factor <= required_latency

# 用法
advisor = InfrastructureAdvisor({
    'predictions_per_month': 1_000_000,
    'latency_ms': 50,
    'memory_gb': 8
})

options = [
    InfrastructureOption("CPU-only", 0.10, 8, 0, 72),
    InfrastructureOption("T4 GPU", 0.50, 16, 1, 360),
    InfrastructureOption("A100 GPU", 3.00, 64, 1, 2160),
]

recommendation = advisor.recommend(options)
print(f"推荐: {recommendation.name} 每月 ${recommendation.monthly_cost}")
```

**策略：模型复杂度 vs 成本权衡**

```python
# 示例：基于成本约束的模型选择
class ModelCostAnalyzer:
    """分析模型选择的成本影响"""
    
    def __init__(self, latency_budget_ms: float, cost_budget_monthly: float):
        self.latency_budget = latency_budget_ms
        self.cost_budget = cost_budget_monthly
    
    def analyze_model_options(self, models: list) -> list:
        """在成本和性能上比较模型"""
        
        results = []
        for model in models:
            training_cost = self._estimate_training_cost(model)
            serving_cost = self._estimate_serving_cost(model)
            total_cost = training_cost + serving_cost
            
            meets_latency = model['latency_ms'] <= self.latency_budget
            meets_cost = total_cost <= self.cost_budget
            
            results.append({
                'model': model['name'],
                'total_monthly_cost': total_cost,
                'training_cost': training_cost,
                'serving_cost': serving_cost,
                'meets_latency': meets_latency,
                'meets_cost': meets_cost,
                'cost_efficiency': model['accuracy'] / total_cost if total_cost > 0 else 0
            })
        
        results.sort(key=lambda x: x['cost_efficiency'], reverse=True)
        return results
    
    def _estimate_training_cost(self, model: dict) -> float:
        """估算模型的训练成本"""
        gpu_hours = model.get('training_gpu_hours', 0)
        gpu_cost_per_hour = 3.0
        return gpu_hours * gpu_cost_per_hour / 30
    
    def _estimate_serving_cost(self, model: dict) -> float:
        """估算每月服务成本"""
        predictions_per_month = 1_000_000
        latency_per_prediction = model['latency_ms'] / 1000
        
        if model.get('requires_gpu', False):
            gpu_cost_per_hour = 3.0
        else:
            gpu_cost_per_hour = 0.10
        
        gpu_hours = predictions_per_month * latency_per_prediction / 3600
        return gpu_hours * gpu_cost_per_hour

# 分析
analyzer = ModelCostAnalyzer(
    latency_budget_ms=100,
    cost_budget_monthly=5000
)

models = [
    {'name': '逻辑回归', 'accuracy': 0.75, 'latency_ms': 1, 
     'training_gpu_hours': 0, 'requires_gpu': False},
    {'name': '随机森林', 'accuracy': 0.82, 'latency_ms': 10,
     'training_gpu_hours': 2, 'requires_gpu': False},
    {'name': '小型神经网络', 'accuracy': 0.85, 'latency_ms': 20,
     'training_gpu_hours': 10, 'requires_gpu': True},
    {'name': '大型 Transformer', 'accuracy': 0.92, 'latency_ms': 100,
     'training_gpu_hours': 100, 'requires_gpu': True},
]

analysis = analyzer.analyze_model_options(models)
for result in analysis[:3]:
    print(f"{result['model']}: ${result['total_monthly_cost']:.2f}/月, "
          f"效益: {result['cost_efficiency']:.4f}")
```

### 2.3.3 成本监控和告警

**原则：成本作为一等指标**

成本应该像性能指标一样被监控和告警。

```python
# 示例：ML 工作负载的成本追踪
from dataclasses import dataclass
from typing import Dict
import datetime

@dataclass
class CostRecord:
    timestamp: datetime.datetime
    workload_type: str
    resource_type: str
    quantity: float
    unit_cost: float
    total_cost: float
    metadata: Dict[str, str]

class CostTracker:
    """追踪和监控 ML 工作负载成本"""
    
    def __init__(self, alert_thresholds: Dict[str, float]):
        self.records = []
        self.thresholds = alert_thresholds
    
    def record_cost(self, record: CostRecord):
        """记录成本事件"""
        self.records.append(record)
        self._check_thresholds(record)
    
    def get_daily_cost(self, date: datetime.date) -> Dict[str, float]:
        """获取特定日期的成本明细"""
        daily_records = [
            r for r in self.records 
            if r.timestamp.date() == date
        ]
        
        breakdown = {}
        for record in daily_records:
            key = f"{record.workload_type}_{record.resource_type}"
            breakdown[key] = breakdown.get(key, 0) + record.total_cost
        
        return breakdown
    
    def get_cost_trend(self, days: int = 30) -> list:
        """获取成本趋势"""
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days)
        
        trend = []
        current_date = start_date
        while current_date <= end_date:
            daily_cost = self.get_daily_cost(current_date)
            trend.append({
                'date': current_date.isoformat(),
                'total': sum(daily_cost.values()),
                'breakdown': daily_cost
            })
            current_date += datetime.timedelta(days=1)
        
        return trend
    
    def _check_thresholds(self, record: CostRecord):
        """检查成本是否超过阈值"""
        workload_key = record.workload_type
        if workload_key in self.thresholds:
            recent_cost = sum(
                r.total_cost for r in self.records[-100:]
                if r.workload_type == workload_key
            )
            
            if recent_cost > self.thresholds[workload_key]:
                self._send_alert(
                    f"成本阈值超过 {workload_key}: "
                    f"${recent_cost:.2f} > ${self.thresholds[workload_key]:.2f}"
                )
    
    def _send_alert(self, message: str):
        """发送成本告警"""
        print(f"⚠️ 成本告警: {message}")
```

---

## 2.4 安全与隐私原则

### 2.4.1 ML 特定安全威胁

AI 系统面临传统软件所没有的安全威胁：

**数据投毒**：攻击者将恶意数据注入训练集以操纵模型行为。这特别危险，因为模型从投毒数据中学习而没有显式检测。

**模型窃取**：攻击者大量查询模型以反向工程其参数。这可以通过精心设计的探测模型决策边界的查询来完成。

**对抗样本**：精心构造的输入导致模型做出错误预测。这些输入通常与正常输入对人类观察者来说无法区分。

**隐私泄露**：模型可能记忆并泄露敏感训练数据。这对在个人信息上训练的模型尤其令人担忧。

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML 安全威胁模型                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  训练阶段威胁：                                                  │
│  ├── 数据投毒（训练数据操纵）                                   │
│  ├── 标签翻转（破坏真值）                                       │
│  ├── 后门攻击（插入触发器）                                     │
│  └── 模型投毒（破坏训练管道）                                   │
│                                                                  │
│  推理阶段威胁：                                                  │
│  ├── 对抗样本（输入操纵）                                       │
│  ├── 模型反转（提取训练数据）                                   │
│  ├── 模型窃取（基于查询的复制）                                 │
│  └── 成员推断（确定数据成员身份）                               │
│                                                                  │
│  基础设施威胁：                                                  │
│  ├── 未授权访问模型产物                                         │
│  ├── API 滥用和拒绝服务                                         │
│  ├── 供应链攻击（依赖项）                                       │
│  └── 侧信道攻击（时序、功耗分析）                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4.2 安全 ML 管道设计

```python
# 示例：安全 ML 管道组件
import hashlib
import hmac
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class DataIntegrityCheck:
    """验证数据未被篡改"""
    data_hash: str
    signature: str
    timestamp: str

class SecureDataPipeline:
    """具有安全控制的 ML 管道"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def validate_data_source(self, data_source: str, 
                            expected_checksum: str) -> bool:
        """验证数据源完整性"""
        trusted_sources = ['s3://company-data/', 'gs://secure-bucket/']
        if not any(data_source.startswith(src) for src in trusted_sources):
            return False
        
        actual_checksum = self._compute_checksum(data_source)
        return hmac.compare_digest(actual_checksum, expected_checksum)
    
    def sanitize_input(self, input_data: Dict) -> Dict:
        """清理输入以防止注入攻击"""
        sanitized = {}
        
        for key, value in input_data.items():
            if isinstance(value, str):
                value = value.replace('<script>', '')
                value = value.replace('javascript:', '')
                value = value[:10000]
            
            sanitized[key] = value
        
        return sanitized
    
    def audit_prediction(self, model_id: str, input_data: Dict,
                        prediction: Dict, user_id: str):
        """记录预测用于审计跟踪"""
        import json
        from datetime import datetime
        
        audit_record = {
            'timestamp': datetime.now().isoformat(),
            'model_id': model_id,
            'user_id': user_id,
            'input_hash': hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest(),
            'prediction': prediction,
            'version': '1.0'
        }
        
        self._write_audit_log(audit_record)
    
    def _compute_checksum(self, data_source: str) -> str:
        """计算数据校验和"""
        return hashlib.sha256(data_source.encode()).hexdigest()
    
    def _write_audit_log(self, record: Dict):
        """写入仅追加审计日志"""
        pass
```

### 2.4.3 隐私保护 ML

**差分隐私**

差分隐私提供数学保证，确保单个记录无法从模型输出中识别。

```python
# 示例：模型训练中的差分隐私
import numpy as np

class DifferentialPrivacySGD:
    """具有差分隐私保证的 SGD"""
    
    def __init__(self, epsilon: float, delta: float, 
                 max_grad_norm: float, noise_multiplier: float):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
    
    def privatize_gradients(self, gradients: np.ndarray, 
                           batch_size: int) -> np.ndarray:
        """向梯度添加校准噪声"""
        
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > self.max_grad_norm:
            gradients = gradients * (self.max_grad_norm / grad_norm)
        
        noise_scale = self.max_grad_norm * self.noise_multiplier
        noise = np.random.normal(0, noise_scale, gradients.shape)
        
        return gradients + noise
    
    def compute_noise_multiplier(self, num_steps: int, 
                                sampling_rate: float) -> float:
        """计算隐私记账的噪声乘数"""
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

def private_training_loop(model, data, dp_sgd: DifferentialPrivacySGD,
                         num_epochs: int, batch_size: int):
    """具有差分隐私的训练循环"""
    
    for epoch in range(num_epochs):
        for batch in data.batches(batch_size):
            gradients = compute_gradients(model, batch)
            private_gradients = dp_sgd.privatize_gradients(gradients, batch_size)
            model.update(private_gradients)
    
    return model
```

**联邦学习**

联邦学习在多个数据源之间训练模型，而不集中数据。

```python
# 示例：联邦学习架构
from typing import List, Dict
import numpy as np

class FederatedServer:
    """联邦学习的中央服务器"""
    
    def __init__(self, global_model, num_clients: int):
        self.global_model = global_model
        self.num_clients = num_clients
        self.round_number = 0
    
    def aggregate_updates(self, client_updates: List[Dict]) -> Dict:
        """聚合客户端的模型更新"""
        
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        aggregated_params = {}
        for param_name in self.global_model.parameters.keys():
            weighted_sum = np.zeros_like(
                client_updates[0]['params'][param_name]
            )
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                weighted_sum += weight * update['params'][param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        self.global_model.set_parameters(aggregated_params)
        self.round_number += 1
        
        return aggregated_params
    
    def distribute_model(self) -> Dict:
        """向客户端发送当前模型"""
        return {
            'round': self.round_number,
            'params': self.global_model.get_parameters()
        }

class FederatedClient:
    """参与联邦学习的客户端"""
    
    def __init__(self, client_id: str, local_data, local_model):
        self.client_id = client_id
        self.data = local_data
        self.model = local_model
    
    def local_training(self, global_params: Dict, 
                      num_epochs: int = 5) -> Dict:
        """在私有数据上本地训练"""
        
        self.model.set_parameters(global_params)
        
        for epoch in range(num_epochs):
            for batch in self.data.batches():
                self.model.train_step(batch)
        
        return {
            'client_id': self.client_id,
            'params': self.model.get_parameters(),
            'num_samples': len(self.data),
            'num_epochs': num_epochs
        }
```

### 2.4.4 合规和治理

**原则：隐私设计**

隐私考虑必须从一开始就构建到系统架构中，而不是事后添加。

关键合规考虑：
- **GDPR**：解释权、删除权、数据最小化
- **CCPA**：消费者隐私权、选择退出机制
- **AI 法案**：风险分类、透明度要求、人类监督
- **HIPAA**：医疗数据保护（如适用）
- **SOC 2**：服务组织的安全控制

> ⚠️ **警告**：不遵守隐私法规可能导致重大罚款（GDPR 下高达全球年收入的 4%）。隐私架构不是可选的——它是法律要求。

---

## 2.5 可观测性原则

### 2.5.1 可观测性的三大支柱

ML 系统的可观测性超越了传统监控。它包括：

1. **指标**：系统行为的定量测量
2. **日志**：系统事件的详细记录
3. **跟踪**：单个请求在系统中的记录

对于 ML 系统，我们添加第四个维度：

4. **模型可观测性**：理解模型如何以及为什么做出预测

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML 可观测性栈                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  应用层：                                                        │
│  ├── 预测延迟                                                   │
│  ├── 预测置信度                                                 │
│  ├── 错误率                                                     │
│  └── 用户反馈                                                   │
│                                                                  │
│  模型层：                                                        │
│  ├── 特征分布                                                   │
│  ├── 预测分布                                                   │
│  ├── 模型性能指标                                               │
│  └── 漂移检测                                                   │
│                                                                  │
│  数据层：                                                        │
│  ├── 数据新鲜度                                                 │
│  ├── 数据质量分数                                               │
│  ├── 模式变化                                                   │
│  └── 缺失值率                                                   │
│                                                                  │
│  基础设施层：                                                    │
│  ├── CPU/GPU 利用率                                             │
│  ├── 内存使用                                                   │
│  ├── 网络流量                                                   │
│  └── 存储利用率                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5.2 指标设计

**原则：指标层次结构**

按层次结构设计指标：业务指标 → 系统指标 → 模型指标 → 基础设施指标。

```python
# 示例：全面的指标设计
from dataclasses import dataclass
from typing import Dict, List
import time

class MLMetricsCollector:
    """收集和组织 ML 指标"""
    
    def __init__(self):
        self.metrics = {
            'business': {},
            'system': {},
            'model': {},
            'infrastructure': {}
        }
    
    def record_prediction(self, prediction: Dict, context: Dict):
        """记录单个预测的指标"""
        
        # 业务指标
        self.metrics['business']['total_predictions'] = \
            self.metrics['business'].get('total_predictions', 0) + 1
        
        # 系统指标
        latency = context.get('latency_ms', 0)
        self._record_histogram('system.prediction_latency', latency)
        
        # 模型指标
        confidence = prediction.get('confidence', 0)
        self._record_histogram('model.prediction_confidence', confidence)
        
        # 追踪预测分布
        pred_class = prediction.get('class', 'unknown')
        self._record_counter(f'model.prediction_distribution.{pred_class}')
    
    def record_feedback(self, prediction_id: str, feedback: Dict):
        """记录预测的用户反馈"""
        
        if feedback.get('correct') is False:
            self._record_counter('business.incorrect_predictions')
        
        if 'actual_label' in feedback:
            self._record_counter(
                f'model.actual_labels.{feedback["actual_label"]}'
            )
    
    def record_data_quality(self, data_batch: Dict):
        """记录数据质量指标"""
        
        for column, missing_pct in data_batch.get('missing_rates', {}).items():
            self._record_gauge(f'data.missing_rate.{column}', missing_pct)
        
        for column, stats in data_batch.get('distribution_stats', {}).items():
            self._record_gauge(f'data.mean.{column}', stats['mean'])
            self._record_gauge(f'data.std.{column}', stats['std'])
    
    def _record_histogram(self, name: str, value: float):
        """记录直方图指标"""
        pass
    
    def _record_counter(self, name: str):
        """记录计数器指标"""
        pass
    
    def _record_gauge(self, name: str, value: float):
        """记录仪表指标"""
        pass
```

### 2.5.3 漂移检测

**概念漂移**：目标变量的统计属性随时间变化。

**数据漂移**：输入特征的分布随时间变化。

```python
# 示例：漂移检测系统
import numpy as np
from typing import Dict
from scipy import stats

class DriftDetector:
    """检测数据和概念漂移"""
    
    def __init__(self, reference_data: np.ndarray, 
                 significance_level: float = 0.05):
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.baseline_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, data: np.ndarray) -> Dict:
        """计算分布统计"""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'percentiles': np.percentile(data, [25, 50, 75], axis=0)
        }
    
    def detect_drift(self, new_data: np.ndarray) -> Dict[str, bool]:
        """检测新数据是否已漂移"""
        
        results = {}
        
        for feature_idx in range(new_data.shape[1]):
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[:, feature_idx],
                new_data[:, feature_idx]
            )
            
            results[f'feature_{feature_idx}'] = {
                'drifted': p_value < self.significance_level,
                'ks_statistic': ks_stat,
                'p_value': p_value
            }
        
        any_drifted = any(r['drifted'] for r in results.values())
        results['overall'] = {'drifted': any_drifted}
        
        return results
    
    def detect_concept_drift(self, predictions: np.ndarray,
                            actuals: np.ndarray) -> Dict:
        """通过监控性能检测概念漂移"""
        
        recent_accuracy = np.mean(predictions == actuals)
        baseline_accuracy = self.baseline_stats.get('accuracy', 0.8)
        
        degradation = baseline_accuracy - recent_accuracy
        drift_detected = degradation > 0.05
        
        return {
            'drift_detected': drift_detected,
            'degradation': degradation,
            'recent_accuracy': recent_accuracy,
            'baseline_accuracy': baseline_accuracy
        }

class DriftMonitor:
    """持续漂移监控"""
    
    def __init__(self, detectors: Dict[str, DriftDetector]):
        self.detectors = detectors
        self.alert_history = []
    
    def monitor_batch(self, batch_data: Dict) -> Dict:
        """监控一批数据的漂移"""
        
        results = {}
        
        for feature_name, detector in self.detectors.items():
            if feature_name in batch_data:
                drift_result = detector.detect_drift(batch_data[feature_name])
                results[feature_name] = drift_result
                
                if drift_result.get('overall', {}).get('drifted', False):
                    self._trigger_alert(feature_name, drift_result)
        
        return results
    
    def _trigger_alert(self, feature_name: str, drift_result: Dict):
        """触发漂移告警"""
        alert = {
            'feature': feature_name,
            'drift_result': drift_result,
            'timestamp': time.time()
        }
        self.alert_history.append(alert)
        
        print(f"🚨 漂移告警: 特征 {feature_name} 已漂移")
```

### 2.5.4 可解释性和可理解性

**原则：每个预测在需要时都应该是可解释的**

对于高风险应用（医疗、金融、法律），解释模型为什么做出特定预测的能力不是可选的。

```python
# 示例：模型可解释性包装器
from typing import Dict
import numpy as np

class ExplainableModelWrapper:
    """用可解释性能力包装模型"""
    
    def __init__(self, model, explainer_type: str = 'shap'):
        self.model = model
        self.explainer_type = explainer_type
        self.explainer = self._create_explainer()
    
    def _create_explainer(self):
        """创建适当的解释器"""
        if self.explainer_type == 'shap':
            import shap
            return shap.Explainer(self.model)
        elif self.explainer_type == 'lime':
            from lime.lime_tabular import LimeTabularExplainer
            return LimeTabularExplainer(...)
        else:
            raise ValueError(f"未知解释器类型: {self.explainer_type}")
    
    def predict_with_explanation(self, input_data: np.ndarray) -> Dict:
        """获取带解释的预测"""
        
        prediction = self.model.predict(input_data)
        
        if self.explainer_type == 'shap':
            shap_values = self.explainer.shap_values(input_data)
            explanation = {
                'feature_importance': dict(zip(
                    self.feature_names,
                    shap_values[0]
                )),
                'base_value': self.explainer.expected_value
            }
        else:
            explanation = {}
        
        return {
            'prediction': prediction,
            'explanation': explanation,
            'confidence': self._get_confidence(input_data)
        }
    
    def _get_confidence(self, input_data: np.ndarray) -> float:
        """获取预测置信度"""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(input_data)
            return np.max(proba)
        return None
```

---

## 2.6 AI 特有的设计权衡

### 2.6.1 准确率-延迟权衡

在实时应用中，模型准确率和预测延迟之间经常存在张力。更大、更复杂的模型往往更准确但更慢。

```
┌─────────────────────────────────────────────────────────────────┐
│                准确率-延迟权衡                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  模型类型          │ 准确率 │ 延迟   │ 用例                    │
│  ─────────────────────────────────────────────────────────────  │
│  逻辑回归          │ 75%    │ 1ms    │ 实时、低成本             │
│  随机森林          │ 82%    │ 10ms   │ 平衡                    │
│  小型神经网络      │ 85%    │ 20ms   │ 中等复杂度              │
│  大型 Transformer  │ 92%    │ 100ms  │ 高准确率需求            │
│  集成模型          │ 94%    │ 200ms  │ 最大准确率              │
│                                                                  │
│  架构决策：                                                       │
│  - 对不同延迟要求使用不同模型                                    │
│  - 使用模型蒸馏降低延迟                                          │
│  - 使用缓存隐藏重复查询的延迟                                    │
│  - 使用混合方法（快速模型 + 慢速模型降级）                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**模式：级联架构**

```python
# 示例：用于准确率-延迟平衡的级联架构
class CascadeModelServer:
    """使用多个复杂度递增的模型"""
    
    def __init__(self, models: list, confidence_threshold: float = 0.8):
        self.models = models  # 从最快到最慢排序
        self.confidence_threshold = confidence_threshold
    
    def predict(self, input_data: Dict) -> Dict:
        """按顺序尝试模型，在足够自信时停止"""
        
        for model_info in self.models:
            model = model_info['model']
            
            prediction = model.predict_with_confidence(input_data)
            
            if prediction['confidence'] >= self.confidence_threshold:
                return {
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'model_used': model_info['name'],
                    'latency': prediction['latency']
                }
        
        # 如果没有模型足够自信，使用最准确的
        return self.models[-1]['model'].predict(input_data)
```

### 2.6.2 批处理 vs 实时处理权衡

```
┌─────────────────────────────────────────────────────────────────┐
│                批处理 vs 实时处理                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  批处理：                                                        │
│  ├── 优点：成本效益高、全面、复杂特征                            │
│  ├── 缺点：高延迟、预测过时                                     │
│  └── 用例：分析、推荐、报告                                     │
│                                                                  │
│  实时处理：                                                      │
│  ├── 优点：低延迟、当前预测                                     │
│  ├── 缺点：成本更高、特征更简单、更复杂                          │
│  └── 用例：欺诈检测、实时推荐、自动化                           │
│                                                                  │
│  流处理（中间地带）：                                            │
│  ├── 优点：近实时、良好可扩展性                                 │
│  ├── 缺点：复杂性、排序保证                                     │
│  └── 用例：物联网、点击流、实时监控                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.6.3 构建 vs 购买权衡

对于许多 ML 组件，架构师必须在构建自定义解决方案和使用现有工具之间做出决策。

**构建当**：
- 问题是业务核心
- 现有解决方案不满足特定需求
- 你有维护团队
- 成本分析显示随时间推移构建更划算

**购买当**：
- 问题是良好理解的，有标准解决方案
- 上市时间至关重要
- 维护负担对你的团队来说太高
- 现有解决方案成熟且可靠

```python
# 示例：构建 vs 购买决策框架
from dataclasses import dataclass
from typing import List

@dataclass
class ComponentDecision:
    component_name: str
    build_cost: float
    buy_cost: float
    maintenance_cost: float
    time_to_build_months: int
    time_to_integrate_months: int
    strategic_importance: str
    team_expertise: str

class BuildBuyAnalyzer:
    """分析 ML 组件的构建 vs 购买决策"""
    
    def analyze(self, components: List[ComponentDecision]) -> List[Dict]:
        """分析每个组件"""
        
        recommendations = []
        for comp in components:
            build_total = (comp.build_cost + comp.maintenance_cost * 3)
            buy_total = comp.buy_cost * 3
            
            time_advantage = (comp.time_to_build_months - 
                            comp.time_to_integrate_months)
            
            if (comp.strategic_importance == 'high' and 
                comp.team_expertise == 'high'):
                recommendation = 'BUILD'
                reason = '战略重要性证明投资合理'
            elif (comp.strategic_importance == 'low' and 
                  comp.team_expertise == 'low'):
                recommendation = 'BUY'
                reason = '非战略性，团队缺乏专业知识'
            elif build_total < buy_total * 0.7:
                recommendation = 'BUILD'
                reason = f'显著的成本节约: ${buy_total - build_total:,.0f}'
            elif buy_total < build_total * 0.7:
                recommendation = 'BUY'
                reason = f'更低的成本，更快的集成'
            else:
                recommendation = '进一步评估'
                reason = '成本相似，需要更深入分析'
            
            recommendations.append({
                'component': comp.component_name,
                'recommendation': recommendation,
                'reason': reason,
                'build_3yr_cost': build_total,
                'buy_3yr_cost': buy_total,
                'time_advantage_months': time_advantage
            })
        
        return recommendations

# 用法
analyzer = BuildBuyAnalyzer()
components = [
    ComponentDecision(
        component_name='特征存储',
        build_cost=200000,
        buy_cost=50000,
        maintenance_cost=80000,
        time_to_build_months=6,
        time_to_integrate_months=2,
        strategic_importance='high',
        team_expertise='medium'
    ),
    ComponentDecision(
        component_name='实验追踪',
        build_cost=100000,
        buy_cost=20000,
        maintenance_cost=40000,
        time_to_build_months=3,
        time_to_integrate_months=1,
        strategic_importance='medium',
        team_expertise='high'
    ),
]

recommendations = analyzer.analyze(components)
for rec in recommendations:
    print(f"{rec['component']}: {rec['recommendation']} ({rec['reason']})")
```

### 2.6.4 一致性 vs 性能权衡

在分布式 ML 系统中，一致性（确保所有节点拥有相同的数据/模型）和性能（快速提供预测）之间经常存在张力。

```python
# 示例：特征服务的最终一致性
from typing import Dict
import asyncio
import time

class EventuallyConsistentFeatureStore:
    """具有可配置一致性的特征存储"""
    
    def __init__(self, primary_store, replica_stores: list):
        self.primary = primary_store
        self.replicas = replica_stores
        self.sync_queue = asyncio.Queue()
    
    async def get_features(self, entity_id: str, 
                          consistency: str = 'eventual') -> Dict:
        """获取指定一致性级别的特征"""
        
        if consistency == 'strong':
            return await self.primary.get(entity_id)
        
        elif consistency == 'eventual':
            replica = self._select_nearest_replica()
            return await replica.get(entity_id)
        
        elif consistency == 'bounded_staleness':
            replica = self._select_nearest_replica()
            data = await replica.get(entity_id)
            
            if self._is_too_stale(data):
                return await self.primary.get(entity_id)
            
            return data
    
    async def update_features(self, entity_id: str, features: Dict):
        """使用写一致性更新特征"""
        
        await self.primary.put(entity_id, features)
        
        for replica in self.replicas:
            await self.sync_queue.put({
                'entity_id': entity_id,
                'features': features,
                'replica': replica
            })
    
    def _select_nearest_replica(self):
        """根据延迟/可用性选择副本"""
        return self.replicas[0]
    
    def _is_too_stale(self, data: Dict) -> bool:
        """检查数据是否对有界新鲜度过时"""
        max_staleness_seconds = 60
        data_timestamp = data.get('timestamp', 0)
        return (time.time() - data_timestamp) > max_staleness_seconds
```

### 2.6.5 简单 vs 复杂权衡

> 💡 **案例研究：简单何时胜出**

一家零售公司需要预测客户流失。他们最初构建了一个复杂的深度学习模型，使用注意力机制，达到 89% 准确率。部署后，他们发现：

1. 模型无法向业务利益相关者解释
2. 重训练需要专门的 GPU 基础设施
3. 特征工程不透明且难以维护
4. 业务无法信任他们不理解的预测

他们用梯度提升树模型（XGBoost）替换了它，达到 87% 准确率。结果：

- 业务利益相关者可以理解特征重要性
- 模型在标准 CPU 上运行
- 训练时间从数小时缩短到数分钟
- 自动生成预测解释
- 整体业务影响：由于信任而做出更好的决策

2% 的准确率降低与可用性和信任方面的收益相比是无关紧要的。

---

## 总结

本章建立了 AI 系统的核心设计原则：

1. **可扩展性**在 AI 中意味着处理更多数据、更多模型和更多实验——而不仅仅是更多用户
2. **可维护性**需要关注点分离、配置管理和 ML 特定的测试策略
3. **成本效益**需要理解 AI 的完整成本结构并在每一层进行优化
4. **安全和隐私**必须从一开始就设计到系统中，而不是事后添加
5. **可观测性**超越传统监控，包括模型和数据可观测性
6. **权衡**是 AI 架构中固有的——没有普遍正确的答案，只有上下文适当的答案

本章中的原则将指导你职业生涯中的架构决策。记住：好的架构不是关于遵循规则——而是关于做出平衡竞争约束的明智决策。

---

## 参考文献

1. Lakshmanan, V., Robinson, S., & Munn, M. (2022). *Machine Learning Engineering*. O'Reilly Media.
2. Amatriain, X. &整天, A. (2022). *Designing Machine Learning Systems*. O'Reilly Media.
3. Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly Media.
4. Paleyes, A., Rabih, M. L., & Lawrence, N. D. (2022). Challenges in deploying machine learning. *Journal of Machine Learning Research*, 23(128), 1-58.
5. Google Cloud. (2024). *MLOps: Continuous delivery and automation pipelines in machine learning*. Google Cloud Documentation.
6. Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems*, 28.

---

*下一章：第 3 章 — AI 系统数据架构*