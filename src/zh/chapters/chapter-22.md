# 第22章：隐私保护AI架构

## 学习目标

学完本章后，你将能够：

1. 解释差分隐私的数学基础，并使用组合定理实现实际的隐私预算
2. 设计联邦学习架构，使协作模型训练无需共享原始数据
3. 使用可工作代码实现AI系统的真实GDPR和CCPA合规要求
4. 将同态加密和安全多方计算应用于隐私敏感的AI工作负载
5. 使用成熟指标评估真实世界AI部署中的隐私-效用权衡

---

## 22.1 引言：AI中的隐私必要性

AI系统本质上是数据密集型的。训练一个最先进的大型语言模型需要数万亿个文本标记的数据。训练推荐系统需要数百万用户的行为数据。训练医疗诊断模型需要访问跨越数十年的患者记录。

这产生了一个根本性张力：AI需要数据来改进，但个人有隐私权。这种张力的解决不是停止构建AI——而是构建尊重隐私的AI。

监管格局使这种必要性变得不可避免：

- **GDPR**（欧盟，2018年）：违规最高可处全球年收入4%或2000万欧元的罚款
- **CCPA/CPRA**（加州，2020/2023年）：每次故意违规2,500-7,500美元
- **PIPL**（中国，2021年）：最高可处年收入5%或5000万元人民币的罚款
- **LGPD**（巴西，2020年）：最高可处巴西收入2%，每次违规上限5000万雷亚尔

根据IAPP（国际隐私专业协会）2024年报告，自2018年以来，组织支付了超过48亿美元的GDPR罚款，AI相关罚款同比增长67%（IAPP，2024）。

本章涵盖实现隐私保护AI的技术机制。

---

## 22.2 差分隐私：数学基础

### 22.2.1 定义

差分隐私（Dwork等，2006）提供了数学保证，即计算的输出在包含或不包含任何个人数据的情况下近似相同。

**形式化定义：**

随机机制M满足(ε, δ)-差分隐私，如果对于所有相差一条记录的数据集D₁和D₂，以及所有可能的输出S：

```
P[M(D₁) ∈ S] ≤ e^ε · P[M(D₂) ∈ S] + δ
```

其中：
- ε（epsilon）：隐私预算——较小的值意味着更强的隐私
- δ（delta）：ε保证被违反的概率——应该密码学地小
- e^ε ≈ 1 + ε（对于小的ε）

**直觉理解：**

如果ε=1，则任何输出的概率在添加或删除一个人的数据时最多变化e≈2.718倍。这意味着对手无法以高置信度确定任何个体是否在数据集中。

### 22.2.2 拉普拉斯机制

最简单的差分隐私机制添加校准的拉普拉斯噪声：

```python
import numpy as np

class LaplaceMechanism:
    """用于差分隐私的拉普拉斯机制"""
    
    def __init__(self, epsilon, sensitivity):
        """
        Args:
            epsilon: 隐私预算
            sensitivity: 一条记录引起的输出最大变化
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon
    
    def privatize(self, value):
        """向数值添加拉普拉斯噪声"""
        noise = np.random.laplace(0, self.scale)
        return value + noise
    
    def privatize_array(self, values, per_element_sensitivity=1.0):
        """向每个元素添加独立的拉普拉斯噪声"""
        scale = per_element_sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, size=values.shape)
        return values + noise
    
    @staticmethod
    def compute_sensitivity(function, dataset, record_index):
        """
        计算函数的敏感度。
        敏感度 = max |f(D) - f(D')|，其中D和D'相差一条记录。
        """
        output_with = function(dataset)
        dataset_without = np.delete(dataset, record_index, axis=0)
        output_without = function(dataset_without)
        return np.max(np.abs(output_with - output_without))
```

### 22.2.3 高斯机制

对于(ε, δ)-差分隐私，高斯机制提供更好的效用：

```python
class GaussianMechanism:
    """用于(ε, δ)-差分隐私的高斯机制"""
    
    def __init__(self, epsilon, delta, sensitivity):
        """
        Args:
            epsilon: 隐私预算
            delta: 失败概率
            sensitivity: L2敏感度
        """
        self.epsilon = epsilon
        self.delta = delta
        # 计算标准差
        self.sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    def privatize(self, value):
        """向数值添加高斯噪声"""
        noise = np.random.normal(0, self.sigma)
        return value + noise
    
    def privatize_array(self, values):
        """向每个元素添加独立的高斯噪声"""
        noise = np.random.normal(0, self.sigma, size=values.shape)
        return values + noise
```

### 22.2.4 组合定理：管理隐私预算

组合定理控制多个查询中隐私预算的消耗：

```python
class PrivacyAccountant:
    """跨多个查询跟踪和管理隐私预算"""
    
    def __init__(self, total_epsilon, total_delta):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.queries = []
        self.spent_epsilon = 0
        self.spent_delta = 0
    
    def query(self, epsilon_per_query, delta_per_query=0):
        """
        记录查询并检查预算是否允许。
        返回 (allowed, remaining_budget)
        """
        # 基本组合
        new_spent_epsilon = self.spent_epsilon + epsilon_per_query
        new_spent_delta = self.spent_delta + delta_per_query
        
        # 检查预算
        if (new_spent_epsilon <= self.total_epsilon and 
            new_spent_delta <= self.total_delta):
            self.spent_epsilon = new_spent_epsilon
            self.spent_delta = new_spent_delta
            self.queries.append({
                'epsilon': epsilon_per_query,
                'delta': delta_per_query
            })
            return True, self.remaining_budget()
        else:
            return False, self.remaining_budget()
    
    def remaining_budget(self):
        """返回剩余隐私预算"""
        return {
            'epsilon': self.total_epsilon - self.spent_epsilon,
            'delta': self.total_delta - self.spent_delta
        }
    
    def advanced_composition(self):
        """
        高级组合定理：
        k次查询，每次ε → (ε', δ')-DP，其中
        ε' = ε√(2k ln(1/δ')) + kε(e^ε - 1)
        """
        k = len(self.queries)
        epsilon_per = self.queries[0]['epsilon'] if self.queries else 0
        
        delta_prime = self.total_delta / 2
        epsilon_advanced = (epsilon_per * np.sqrt(2 * k * np.log(1 / delta_prime)) + 
                           k * epsilon_per * (np.exp(epsilon_per) - 1))
        
        return epsilon_advanced, delta_prime
```

### 22.2.5 Rényi差分隐私（RDP）

RDP提供比基本组合更紧的组合：

```python
class RDPAccountant:
    """用于紧组合的Rényi差分隐私记账器"""
    
    def __init__(self):
        self.alphas = []  # RDP阶数
        self.rdp_values = []  # 每次查询的RDP值
    
    def compute_rdp_gaussian(self, epsilon, delta, sigma, alpha):
        """计算高斯机制的RDP"""
        return alpha / (2 * sigma**2)
    
    def add_query(self, rdp_values):
        """添加查询的RDP值"""
        self.rdp_values.append(rdp_values)
    
    def get_total_rdp(self, alpha):
        """获取阶数alpha处的总RDP"""
        total = sum(rv[alpha] for rv in self.rdp_values if alpha in rv)
        return total
    
    def convert_to_dp(self, alpha, delta):
        """将RDP转换为(ε, δ)-DP"""
        rdp = self.get_total_rdp(alpha)
        epsilon = rdp + np.log(1 / delta) / (alpha - 1)
        return epsilon, delta
```

---

## 22.3 联邦学习架构

### 22.3.1 联邦学习协议

联邦学习（McMahan等，2017）使多方能够在不共享原始数据的情况下协作训练模型。

**联邦平均（FedAvg）算法：**

```python
import torch
import torch.nn as nn
from typing import List, Dict
import copy

class FederatedServer:
    """协调多个客户端的联邦学习服务器"""
    
    def __init__(self, global_model, num_clients, num_rounds=100):
        self.global_model = global_model
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.history = []
    
    def federated_averaging(self, clients: List['FederatedClient'],
                          clients_per_round: int = 10,
                          local_epochs: int = 5,
                          learning_rate: float = 0.01):
        """
        运行联邦平均算法。
        
        参考：McMahan等（2017）"Communication-Efficient Learning 
        of Deep Networks from Decentralized Data"
        """
        for round_num in range(self.num_rounds):
            # 选择随机客户端子集
            selected_clients = np.random.choice(
                clients, size=min(clients_per_round, len(clients)),
                replace=False
            )
            
            # 将全局模型发送给选定的客户端
            global_weights = self.global_model.state_dict()
            
            # 每个客户端本地训练
            local_models = []
            local_sizes = []
            
            for client in selected_clients:
                # 客户端本地训练
                local_model = copy.deepcopy(self.global_model)
                local_model.load_state_dict(global_weights)
                
                client.train_local(local_model, local_epochs, learning_rate)
                
                local_models.append(local_model)
                local_sizes.append(client.data_size)
            
            # 聚合模型（加权平均）
            self._aggregate_models(local_models, local_sizes)
            
            # 评估
            metrics = self.evaluate()
            self.history.append(metrics)
            
            if round_num % 10 == 0:
                print(f"第 {round_num} 轮: {metrics}")
        
        return self.history
    
    def _aggregate_models(self, local_models: List[nn.Module],
                         local_sizes: List[int]):
        """模型参数的加权平均"""
        total_size = sum(local_sizes)
        
        # 初始化聚合权重
        aggregated = {}
        for key in local_models[0].state_dict().keys():
            aggregated[key] = torch.zeros_like(local_models[0].state_dict()[key])
        
        # 加权求和
        for model, size in zip(local_models, local_sizes):
            weight = size / total_size
            for key in model.state_dict().keys():
                aggregated[key] += weight * model.state_dict()[key]
        
        # 更新全局模型
        self.global_model.load_state_dict(aggregated)


class FederatedClient:
    """具有本地数据的联邦学习客户端"""
    
    def __init__(self, client_id, train_loader, val_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_size = len(train_loader.dataset)
    
    def train_local(self, model, epochs, learning_rate):
        """在本地数据上训练模型"""
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return model
    
    def evaluate(self, model):
        """在本地验证数据上评估模型"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return correct / total
```

### 22.3.2 安全聚合

为了保护单个客户端更新免受服务器的影响，安全聚合使用加密技术：

```python
import hashlib
from typing import Tuple

class SecureAggregator:
    """
    使用加法秘密共享的安全聚合。
    服务器永远不会看到单个客户端更新。
    """
    
    def __init__(self, num_clients):
        self.num_clients = num_clients
    
    def setup(self) -> Tuple[Dict, Dict]:
        """
        设置阶段：每对客户端同意共享随机掩码。
        返回 (private_keys, public_keys)
        """
        # 生成成对掩码
        pairwise_masks = {}
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                # 共享随机掩码
                seed = f"{i}-{j}".encode()
                np.random.seed(int(hashlib.sha256(seed).hexdigest(), 16) % 2**32)
                mask = np.random.randn(1000)  # 假设1000维更新
                pairwise_masks[(i, j)] = mask
        
        return pairwise_masks
    
    def add_masks(self, updates: List[np.ndarray], 
                  pairwise_masks: Dict) -> np.ndarray:
        """
        每个客户端将其成对掩码份额添加到其更新中。
        在服务器处求和时，掩码抵消。
        """
        masked_updates = []
        
        for client_id, update in enumerate(updates):
            masked = update.copy()
            
            for (i, j), mask in pairwise_masks.items():
                if client_id == i:
                    masked += mask
                elif client_id == j:
                    masked -= mask
            
            masked_updates.append(masked)
        
        return masked_updates
    
    def aggregate(self, masked_updates: List[np.ndarray]) -> np.ndarray:
        """
        聚合掩码更新。
        成对掩码抵消，只留下真实更新的总和。
        """
        return np.sum(masked_updates, axis=0) / len(masked_updates)
```

### 22.3.3 真实世界联邦学习：Google的Gboard

Google使用联邦学习来改进Gboard（键盘应用程序），而无需收集用户输入数据：

**架构：**
1. **设备上训练**：每部手机在用户输入模式上训练本地模型
2. **模型更新提取**：仅模型更新（而非数据）发送到Google服务器
3. **安全聚合**：在Google看到之前聚合许多用户的更新
4. **全局模型更新**：聚合更新改进全局模型
5. **模型分发**：更新的模型推送回设备

根据Google的AI Blog（ai.googleblog.com，2017），这种方法：
- 每天处理数十亿次预测
- 永远不会看到单个用户数据
- 语言预测准确率提高2-4%
- 无需用户选择加入，只需正常使用Gboard

---

## 22.4 AI系统的GDPR和CCPA合规

### 22.4.1 AI中的数据主体权利

GDPR提供了直接影响AI系统的几项权利：

| 权利 | GDPR条款 | AI影响 | 实现 |
|------|---------|--------|------|
| 访问权 | 第15条 | 用户可以请求使用哪些数据训练模型 | 数据溯源跟踪 |
| 删除权 | 第17条 | 用户可以请求从训练中删除其数据 | 机器遗忘 |
| 解释权 | 第22条 | 自动化决策必须可解释 | XAI方法 |
| 数据可携权 | 第20条 | 用户可以以机器可读格式请求其数据 | 导出API |
| 同意 | 第6、7条 | 数据处理需要明确同意 | 同意管理 |

### 22.4.2 机器遗忘

机器遗忘（Bourtoule等，2021）能够从训练模型中移除特定数据点的影响：

```python
class SISAUnlearning:
    """
    SISA（分片、隔离、切片和聚合）遗忘。
    参考：Bourtoule等（2021）"Machine Unlearning"
    """
    
    def __init__(self, base_model, num_shards=10, num_slices=5):
        self.base_model = base_model
        self.num_shards = num_shards
        self.num_slices = num_slices
        self.shards = {}
        self.slice_models = {}
    
    def train(self, dataset):
        """使用SISA结构训练以实现高效遗忘"""
        # 将数据集分成分片
        shard_size = len(dataset) // self.num_shards
        shards = []
        
        for i in range(self.num_shards):
            start = i * shard_size
            end = min((i + 1) * shard_size, len(dataset))
            shards.append(dataset[start:end])
        
        # 使用切片训练每个分片
        for shard_id, shard_data in enumerate(shards):
            self.shards[shard_id] = shard_data
            self.slice_models[shard_id] = self._train_shard(shard_data)
    
    def _train_shard(self, shard_data):
        """使用中间检查点训练分片"""
        models = []
        slice_size = len(shard_data) // self.num_slices
        
        for slice_idx in range(self.num_slices):
            # 在到此切片为止的累积数据上训练
            cumulative_data = shard_data[:((slice_idx + 1) * slice_size)]
            
            model = copy.deepcopy(self.base_model)
            # 在cumulative_data上训练模型
            # （简化——将包括实际训练循环）
            models.append(model)
        
        return models
    
    def unlearn(self, data_point):
        """
        移除特定数据点的影响。
        仅重新训练包含该数据点的分片。
        """
        shards_to_retrain = []
        
        for shard_id, shard_data in self.shards.items():
            if data_point in shard_data:
                shards_to_retrain.append(shard_id)
        
        # 仅重新训练受影响的分片
        for shard_id in shards_to_retrain:
            # 找到哪个切片包含该数据点
            shard_data = self.shards[shard_id]
            # 从该切片开始重新训练
            self.slice_models[shard_id] = self._train_shard(shard_data)
        
        return len(shards_to_retrain)
    
    def predict(self, input_data):
        """跨所有分片的集成预测"""
        predictions = []
        
        for shard_id, models in self.slice_models.items():
            # 使用每个分片的最后一个模型
            model = models[-1]
            pred = model(input_data)
            predictions.append(pred)
        
        # 平均预测
        return torch.mean(torch.stack(predictions), dim=0)
```

### 22.4.3 隐私影响评估

```python
class PrivacyImpactAssessment:
    """AI系统的结构化隐私影响评估"""
    
    def __init__(self, system_name, system_description):
        self.system_name = system_name
        self.system_description = system_description
        self.assessments = []
    
    def assess_data_flow(self, data_type, purpose, retention, sharing):
        """评估特定数据流"""
        risk_level = self._calculate_risk(data_type, purpose, retention, sharing)
        
        assessment = {
            'data_type': data_type,
            'purpose': purpose,
            'retention': retention,
            'sharing': sharing,
            'risk_level': risk_level,
            'mitigations': self._suggest_mitigations(risk_level)
        }
        
        self.assessments.append(assessment)
        return assessment
    
    def _calculate_risk(self, data_type, purpose, retention, sharing):
        """根据数据特征计算风险等级"""
        risk_score = 0
        
        # 数据敏感性
        if data_type in ['health', 'biometric', 'financial']:
            risk_score += 3
        elif data_type in ['location', 'behavioral', 'communication']:
            risk_score += 2
        elif data_type in ['demographic', 'device']:
            risk_score += 1
        
        # 目的
        if purpose not in ['core_function', 'service_improvement']:
            risk_score += 1
        
        # 保留
        if retention > 365:  # 天
            risk_score += 1
        
        # 共享
        if sharing == 'third_party':
            risk_score += 2
        elif sharing == 'internal':
            risk_score += 1
        
        # 映射到风险等级
        if risk_score >= 7:
            return 'critical'
        elif risk_score >= 5:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_mitigations(self, risk_level):
        """根据风险等级建议缓解措施"""
        mitigations = {
            'critical': [
                '实施差分隐私',
                '使用联邦学习',
                '进行第三方审计',
                '实施实时监控',
                '要求明确同意'
            ],
            'high': [
                '应用数据最小化',
                '实施访问控制',
                '添加假名化',
                '定期审计'
            ],
            'medium': [
                '应用数据最小化',
                '实施基本访问控制',
                '记录处理活动'
            ],
            'low': [
                '应用数据最小化',
                '记录处理活动'
            ]
        }
        return mitigations.get(risk_level, [])
    
    def generate_report(self):
        """生成合规报告"""
        report = {
            'system': self.system_name,
            'total_assessments': len(self.assessments),
            'risk_distribution': {},
            'recommended_actions': []
        }
        
        for assessment in self.assessments:
            risk = assessment['risk_level']
            report['risk_distribution'][risk] = report['risk_distribution'].get(risk, 0) + 1
        
        # 生成行动项
        for assessment in self.assessments:
            if assessment['risk_level'] in ['critical', 'high']:
                report['recommended_actions'].extend([
                    f"处理 {assessment['data_type']}: {m}" 
                    for m in assessment['mitigations']
                ])
        
        return report
```

---

## 22.5 用于AI的同态加密

### 22.5.1 概念

同态加密允许在不解密的情况下对加密数据进行计算：

```python
# 同态操作的简化表示
# 实际实现使用SEAL、TenSEAL或HElib等库

class SimplifiedHomomorphicOps:
    """
    同态操作的概念演示。
    在实践中，使用Microsoft SEAL或TenSEAL等库。
    """
    
    def __init__(self):
        # 在实际实现中，这将是一对密钥
        self.public_key = "pk_placeholder"
        self.private_key = "sk_placeholder"
    
    def encrypt(self, value):
        """加密一个值（概念性）"""
        # 实际实现使用复杂数学运算
        # 这是简化表示
        return {'encrypted': True, 'value': value, 'noise': 0.1}
    
    def decrypt(self, encrypted_value):
        """解密一个值（概念性）"""
        return encrypted_value['value']
    
    def add(self, enc_a, enc_b):
        """同态加法：E(a) + E(b) = E(a + b)"""
        return {
            'encrypted': True,
            'value': enc_a['value'] + enc_b['value'],
            'noise': enc_a['noise'] + enc_b['noise']
        }
    
    def multiply(self, enc_a, enc_b):
        """同态乘法：E(a) * E(b) = E(a * b)"""
        return {
            'encrypted': True,
            'value': enc_a['value'] * enc_b['value'],
            'noise': enc_a['noise'] * enc_b['noise'] * 2  # 噪声增长更快
        }
```

### 22.5.2 实用同态加密

实际实现使用Microsoft SEAL等库：

```python
# 这是展示如何使用TenSEAL的伪代码
# pip install tenseal

"""
import tenseal as ts

# 设置TenSEAL上下文
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()
context.generate_relin_keys()

# 加密向量
vector_a = [1.0, 2.0, 3.0, 4.0]
vector_b = [5.0, 6.0, 7.0, 8.0]

enc_a = ts.ckks_vector(context, vector_a)
enc_b = ts.ckks_vector(context, vector_b)

# 同态操作
enc_sum = enc_a + enc_b      # E(a) + E(b) = E(a+b)
enc_product = enc_a * enc_b   # E(a) * E(b) = E(a*b)

# 解密结果
result_sum = enc_sum.decrypt()
result_product = enc_product.decrypt()

# result_sum ≈ [6.0, 8.0, 10.0, 12.0]
# result_product ≈ [5.0, 12.0, 21.0, 32.0]
"""
```

---

## 22.6 案例研究：Apple如何实施差分隐私

### 背景

Apple一直在大规模实施差分隐私的先驱。自2016年以来，Apple在iOS和macOS中使用差分隐私来收集使用统计，而不损害单个用户隐私。

### 实施架构

**1. 本地差分隐私（LDP）**

Apple使用本地差分隐私，噪声在设备上添加，然后数据发送到Apple的服务器：

```
用户设备 → 本地添加噪声 → 发送到Apple → 聚合结果
```

这比集中式DP更强，因为Apple永远看不到真实数据。

**2. 隐私预算管理**

根据Apple的"差分隐私技术概述"（apple.com/privacy，2017）：

- 每种数据类型都有分配的隐私预算（ε）
- 一旦用户对某种数据类型的预算用尽，就不再收集该类型的数据
- 预算定期补充（通常每周）

**3. 真实世界应用**

| 数据类型 | ε值 | 目的 | 频率 |
|---------|-----|------|------|
| Emoji使用 | 4 | 流行Emoji跟踪 | 每日 |
| QuickType建议 | 8 | 键盘预测 | 每日 |
| Safari能耗影响 | 2 | 电池优化 | 每周 |
| 健康数据趋势 | 1 | 健康应用改进 | 每月 |

**4. 实施细节**

Apple的方法结合了：

- **RAPPOR**（随机可聚合隐私保护序数响应）：用于收集字符串的频率分布
- **无块Apple私有信息检索**：用于私有集合交集
- **布隆过滤器**：用于具有隐私保证的近似计数

### 技术结果

根据Apple的WWDC演示（2016、2017、2019）：

| 指标 | 数值 |
|------|------|
| 参与设备 | 10亿+ |
| 每日处理数据点 | 100亿+ |
| 每用户每周隐私预算 | ε ≈ 4-8（每种数据类型） |
| 误报率（RAPPOR） | <1% |
| 与非隐私方法的效用损失 | <5% |

### 经验教训

1. **本地DP对信任至关重要**：用户信任Apple因为数据在设备上私有化
2. **隐私预算必须谨慎管理**：过度收集会侵蚀隐私
3. **效用可以维持**：通过仔细的机制设计，准确性损失很小
4. **透明度很重要**：Apple发布了其DP实施的详细信息

---

## 22.7 实战故事：医疗AI系统中的隐私泄露

### 背景

2021年，一家医疗AI公司（匿名化）遭受了重大隐私泄露，暴露了他们"隐私保护"方法的局限性。该公司开发了一个预测患者再入院风险的AI系统，声称使用"匿名化"的患者数据。

### 泄露事件

**发生了什么：**

该公司与研究合作伙伴分享了"匿名化"的患者数据。研究合作伙伴使用外部数据集，能够重新识别个体患者：

1. **准标识符**：年龄、邮政编码和入院日期的组合对87%的患者是唯一的
2. **链接攻击**：与公开选民登记数据结合，重新识别率达到94%
3. **敏感信息泄露**：可识别患者的诊断、治疗和结果

**技术失败：**

```python
# 有缺陷的"匿名化"的简化表示
def flawed_anonymization(patient_data):
    """公司做了什么（错误）"""
    anonymized = patient_data.copy()
    
    # 移除直接标识符（姓名、SSN等）
    anonymized = anonymized.drop(['name', 'ssn', 'phone'], axis=1)
    
    # 保留准标识符（年龄、邮政编码、入院日期）
    # 这是关键错误
    return anonymized

# 他们应该做什么
def proper_anonymization(patient_data):
    """他们应该做什么"""
    anonymized = patient_data.copy()
    
    # 移除直接标识符
    anonymized = anonymized.drop(['name', 'ssn', 'phone'], axis=1)
    
    # 泛化准标识符
    anonymized['age'] = anonymized['age'] // 10 * 10  # 年龄组（20-29、30-39等）
    anonymized['zip'] = anonymized['zip'].str[:3]     # 仅前3位数字
    anonymized['admission_date'] = pd.to_datetime(anonymized['admission_date']).dt.month
    
    # 应用k-匿名性（k=5）
    anonymized = apply_k_anonymity(anonymized, k=5)
    
    # 对敏感属性应用l-多样性
    anonymized = apply_l_diversity(anonymized, sensitive_cols=['diagnosis', 'treatment'], l=3)
    
    return anonymized

def apply_k_anonymity(data, k):
    """确保准标识符的每个组合至少出现k次"""
    quasi_id_cols = ['age', 'zip', 'admission_date']
    
    # 计数组合
    counts = data.groupby(quasi_id_cols).size().reset_index(name='count')
    
    # 过滤以仅保留至少k条记录的组
    valid_groups = counts[counts['count'] >= k][quasi_id_cols]
    
    # 过滤原始数据
    merged = data.merge(valid_groups, on=quasi_id_cols)
    
    return merged
```

### 影响

| 指标 | 数值 |
|------|------|
| 受影响患者 | 230万 |
| 重新识别率 | 94% |
| 暴露的敏感状况 | HIV、心理健康、药物滥用 |
| 监管罚款 | 450万美元（HIPAA违规） |
| 诉讼和解 | 1200万美元 |
| 声誉损失 | 失去3家医院合作伙伴 |

### 根本原因分析

1. **匿名化不足**：移除直接标识符是不够的
2. **没有正式隐私保证**：公司依赖没有正式定义的"匿名化"
3. **风险评估失败**：没有人考虑链接攻击
4. **缺乏隐私专业知识**：团队缺乏差分隐私知识

### 经验教训

1. **K-匿名性单独不够**：即使是k-匿名数据也可能容易受到同质性和背景知识攻击
2. **差分隐私提供正式保证**：只有DP提供数学证明的隐私
3. **隐私影响评估是强制性的**：每个数据共享决策都需要正式评估
4. **纵深防御也适用于隐私**：应组合多种隐私技术

---

## 22.8 何时使用/何时不使用隐私保护AI

### 何时使用特定技术

| 技术 | 何时使用 | 何时不使用 | 权衡 |
|------|---------|-----------|------|
| 差分隐私 | 数据收集、分析 | 当效用损失不可接受时 | 隐私vs准确性 |
| 联邦学习 | 多方训练 | 当数据无法分区时 | 通信开销 |
| 同态加密 | 对敏感数据推理 | 当延迟至关重要时 | 100-1000倍减速 |
| 安全MPC | 多方计算 | 当各方是对抗性的时 | 通信复杂度 |
| 机器遗忘 | GDPR/CCPA合规 | 当模型很少更新时 | 重训练成本 |
| 数据最小化 | 始终（基线） | 永远不要跳过 | 最小效用影响 |

### 决策框架

```
处理个人数据吗？─── 是 ──→ 隐私保护技术是强制性的
         │
         否
         │
与第三方共享数据吗？─── 是 ──→ 应用DP或联邦学习
         │
         否
         │
需要解释模型决策吗？─── 是 ──→ 实施XAI方法
         │
         否
         │
标准安全基线
```

---

## 22.9 总结

隐私保护AI不是可选的——它是法律和道德要求：

1. **差分隐私提供数学保证**：隐私的黄金标准，ε控制隐私-效用权衡。

2. **联邦学习实现无需数据共享的协作**：Google的Gboard在十亿用户规模上展示了这一点。

3. **GDPR/CCPA合规需要特定的技术实现**：机器遗忘、解释权和数据最小化是强制性的。

4. **本地差分隐私比中心DP更强**：Apple的方法确保服务器永远看不到真实数据。

5. **简单的"匿名化"是不够的**：真正的隐私需要正式保证，而不仅仅是移除姓名。

6. **隐私-效用权衡是可管理的**：通过仔细的机制设计，可以以最小的准确性损失保留隐私。

---

## 22.10 讨论题

1. **隐私预算分配**：如果你有一个应用程序的总隐私预算为ε=10，收集5种不同类型的数据，你将如何分配预算？哪些因素影响你的决策？

2. **联邦学习vs集中式DP**：在什么情况下你会选择联邦学习而不是集中式差分隐私？实际权衡是什么？

3. **机器遗忘挑战**：如果一个模型是使用对抗训练（存储中间样本）训练的，你将如何实现有效的机器遗忘？

4. **跨境数据传输**：一家公司在欧盟（GDPR）和中国（PIPL）都有业务。你将如何设计一个符合两个法规的隐私保护AI系统？

5. **隐私vs安全**：差分隐私防止隐私攻击，但它如何与安全攻击（如对抗样本）交互？一个模型能否既私有又稳健？

---

## 22.11 练习

### 练习1：差分隐私实现

为计算数据集的平均年龄实现差分隐私机制：

```python
import numpy as np

class PrivateAverageComputer:
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def compute_private_average(self, ages):
        """计算ε-DP平均年龄"""
        # 平均的敏感度 = (max - min) / n
        sensitivity = (max(ages) - min(ages)) / len(ages)
        
        # 真实平均值
        true_average = np.mean(ages)
        
        # 添加拉普拉斯噪声
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        
        return true_average + noise
    
    def evaluate_utility(self, true_average, private_averages):
        """评估隐私机制的效用损失"""
        errors = [abs(pa - true_average) for pa in private_averages]
        return {
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'rmse': np.sqrt(np.mean([e**2 for e in errors]))
        }
```

**任务：**
1. 使用ε = 0.1、1.0、10.0测试
2. 测量每个隐私级别的效用损失
3. 绘制隐私-效用权衡曲线
4. 确定<5%平均误差所需的最小ε

### 练习2：联邦学习模拟

模拟具有异构数据的联邦学习：

```python
def simulate_federated_learning(num_clients=10, num_rounds=50):
    """
    模拟联邦学习，其中每个客户端具有
    不同的数据分布（非IID）。
    """
    # 创建具有不同数据分布的客户端
    clients = []
    for i in range(num_clients):
        # 每个客户端获得不同类别分布的数据
        train_loader, val_loader = create_heterogeneous_data(i, num_clients)
        clients.append(FederatedClient(i, train_loader, val_loader))
    
    # 运行联邦平均
    server = FederatedServer(global_model, num_clients, num_rounds)
    history = server.federated_averaging(clients)
    
    return history
```

**任务：**
1. 比较IID和非IID数据分布
2. 测量收敛速度和最终准确性
3. 分析每轮客户端数量的影响
4. 绘制通信成本vs准确性

### 练习3：隐私影响评估

对真实世界的AI系统进行隐私影响评估：

**选择一个：**
- 信用评分系统
- 面部识别系统
- 推荐引擎
- 医疗诊断AI

**要求：**
1. 识别所有数据流
2. 评估每个流的风险
3. 提出技术缓解措施
4. 记录合规要求（GDPR、CCPA等）
5. 创建10页的评估报告

---

## 22.12 参考文献

### 差分隐私

1. Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). "Calibrating Noise to Sensitivity in Private Data Analysis." *TCC*. https://doi.org/10.1007/11681878_14

2. Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy." *Theoretical Computer Science Foundations*. https://doi.org/10.1561/0400000042

3. Abadi, M., 等. (2016). "Deep Learning with Differential Privacy." *ACM CCS*. https://doi.org/10.1145/2976749.2978318

### 联邦学习

4. McMahan, H. B., 等. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS*. https://arxiv.org/abs/1602.05629

5. Konečný, J., 等. (2016). "Federated Learning: Strategies for Improving Communication Efficiency." *arXiv*. https://arxiv.org/abs/1610.05492

### 机器遗忘

6. Bourtoule, L., 等. (2021). "Machine Unlearning." *IEEE S&P*. https://doi.org/10.1109/SP40001.2021.00019

7. Cao, D., & Yang, J. (2020). "Toward a Better Understanding of Machine Unlearning." *arXiv*. https://arxiv.org/abs/2010.12101

### 隐私法规

8. 欧洲议会. (2016). "通用数据保护条例 (GDPR)." https://gdpr-info.eu/

9. 加州立法信息. (2020). "加州消费者隐私法案 (CCPA)." https://oag.ca.gov/privacy/ccpa

10. IAPP. (2024). "GDPR执法追踪器." https://www.trackgdpr.com/

### Apple的差分隐私

11. Apple. (2017). "差分隐私技术概述." https://www.apple.com/privacy/docs/Differential_Privacy_Overview.pdf

12. Tang, J., 等. (2017). "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response." *ACM CCS*. https://doi.org/10.1145/3133956.3133981

### 医疗隐私

13. Sweeney, L. (2000). "Simple Demographics Often Identify People Uniquely." *卡内基梅隆大学*. https://dataprivacylab.org/research/identifiability/

14. 健康与人类服务部. (2013). "HIPAA隐私规则." https://www.hhs.gov/hipaa/for-professionals/privacy/index.html

---

*下一章：[第23章：AI架构设计方法论 →](./chapter-23.md)*
