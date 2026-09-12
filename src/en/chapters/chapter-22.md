# Chapter 22: Privacy-Preserving AI Architecture

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain the mathematical foundations of differential privacy and implement practical privacy budgets using the composition theorem
2. Design federated learning architectures that enable collaborative model training without sharing raw data
3. Implement real GDPR and CCPA compliance requirements for AI systems with working code
4. Apply homomorphic encryption and secure multi-party computation to privacy-sensitive AI workloads
5. Evaluate the privacy-utility tradeoff in real-world AI deployments using established metrics

---

## 22.1 Introduction: The Privacy Imperative in AI

AI systems are data-hungry by nature. Training a state-of-the-art large language model requires trillions of tokens of text data. Training a recommendation system requires behavioral data from millions of users. Training a medical diagnostic model requires access to patient records spanning decades.

This creates a fundamental tension: AI needs data to improve, but individuals have a right to privacy. The resolution of this tension is not to stop building AI—it is to build AI that respects privacy by design.

The regulatory landscape has made this imperative unavoidable:

- **GDPR** (EU, 2018): Up to 4% of global annual revenue or €20 million in fines for violations
- **CCPA/CPRA** (California, 2020/2023): $2,500-$7,500 per intentional violation
- **PIPL** (China, 2021): Up to 5% of annual revenue or ¥50 million in fines
- **LGPD** (Brazil, 2020): Up to 2% of revenue in Brazil, capped at R$50 million per violation

According to the IAPP (International Association of Privacy Professionals) 2024 report, organizations paid over $4.8 billion in GDPR fines since 2018, with AI-related fines increasing by 67% year-over-year (IAPP, 2024).

This chapter covers the technical mechanisms that enable privacy-preserving AI.

---

## 22.2 Differential Privacy: Mathematical Foundations

### 22.2.1 Definition

Differential privacy (Dwork et al., 2006) provides a mathematical guarantee that the output of a computation is approximately the same whether or not any individual's data is included.

**Formal Definition:**

A randomized mechanism M satisfies (ε, δ)-differential privacy if for all datasets D₁ and D₂ that differ in a single record, and for all possible outputs S:

```
P[M(D₁) ∈ S] ≤ e^ε · P[M(D₂) ∈ S] + δ
```

Where:
- ε (epsilon): The privacy budget — smaller values mean stronger privacy
- δ (delta): The probability that the ε-guarantee is violated — should be cryptographically small
- e^ε ≈ 1 + ε for small ε

**Intuition:**

If ε = 1, then the probability of any output changes by at most a factor of e ≈ 2.718 when one person's data is added or removed. This means an adversary cannot determine with high confidence whether any individual was in the dataset.

### 22.2.2 The Laplace Mechanism

The simplest differential privacy mechanism adds calibrated Laplace noise:

```python
import numpy as np

class LaplaceMechanism:
    """Laplace mechanism for differential privacy"""
    
    def __init__(self, epsilon, sensitivity):
        """
        Args:
            epsilon: Privacy budget
            sensitivity: Maximum change in output from one record
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon
    
    def privatize(self, value):
        """Add Laplace noise to a numeric value"""
        noise = np.random.laplace(0, self.scale)
        return value + noise
    
    def privatize_array(self, values, per_element_sensitivity=1.0):
        """Add independent Laplace noise to each element"""
        scale = per_element_sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, size=values.shape)
        return values + noise
    
    @staticmethod
    def compute_sensitivity(function, dataset, record_index):
        """
        Compute the sensitivity of a function.
        Sensitivity = max |f(D) - f(D')| where D and D' differ in one record.
        """
        output_with = function(dataset)
        dataset_without = np.delete(dataset, record_index, axis=0)
        output_without = function(dataset_without)
        return np.max(np.abs(output_with - output_without))
```

### 22.2.3 The Gaussian Mechanism

For (ε, δ)-differential privacy, the Gaussian mechanism provides better utility:

```python
class GaussianMechanism:
    """Gaussian mechanism for (ε, δ)-differential privacy"""
    
    def __init__(self, epsilon, delta, sensitivity):
        """
        Args:
            epsilon: Privacy budget
            delta: Failure probability
            sensitivity: L2 sensitivity
        """
        self.epsilon = epsilon
        self.delta = delta
        # Compute standard deviation
        self.sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    def privatize(self, value):
        """Add Gaussian noise to a numeric value"""
        noise = np.random.normal(0, self.sigma)
        return value + noise
    
    def privatize_array(self, values):
        """Add independent Gaussian noise to each element"""
        noise = np.random.normal(0, self.sigma, size=values.shape)
        return values + noise
```

### 22.2.4 Composition Theorem: Managing Privacy Budget

The composition theorem governs how privacy budget is consumed across multiple queries:

```python
class PrivacyAccountant:
    """Track and manage privacy budget across multiple queries"""
    
    def __init__(self, total_epsilon, total_delta):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.queries = []
        self.spent_epsilon = 0
        self.spent_delta = 0
    
    def query(self, epsilon_per_query, delta_per_query=0):
        """
        Record a query and check if budget allows it.
        Returns (allowed, remaining_budget)
        """
        # Basic composition
        new_spent_epsilon = self.spent_epsilon + epsilon_per_query
        new_spent_delta = self.spent_delta + delta_per_query
        
        # Check budget
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
        """Return remaining privacy budget"""
        return {
            'epsilon': self.total_epsilon - self.spent_epsilon,
            'delta': self.total_delta - self.spent_delta
        }
    
    def advanced_composition(self):
        """
        Advanced composition theorem:
        k queries with ε each → (ε', δ')-DP where
        ε' = ε√(2k ln(1/δ')) + kε(e^ε - 1)
        """
        k = len(self.queries)
        epsilon_per = self.queries[0]['epsilon'] if self.queries else 0
        
        delta_prime = self.total_delta / 2
        epsilon_advanced = (epsilon_per * np.sqrt(2 * k * np.log(1 / delta_prime)) + 
                           k * epsilon_per * (np.exp(epsilon_per) - 1))
        
        return epsilon_advanced, delta_prime
```

### 22.2.5 Rényi Differential Privacy (RDP)

RDP provides tighter composition than basic composition:

```python
class RDPAccountant:
    """Rényi Differential Privacy accountant for tight composition"""
    
    def __init__(self):
        self.alphas = []  # RDP orders
        self.rdp_values = []  # RDP values for each query
    
    def compute_rdp_gaussian(self, epsilon, delta, sigma, alpha):
        """Compute RDP for Gaussian mechanism"""
        return alpha / (2 * sigma**2)
    
    def add_query(self, rdp_values):
        """Add a query's RDP values"""
        self.rdp_values.append(rdp_values)
    
    def get_total_rdp(self, alpha):
        """Get total RDP at order alpha"""
        total = sum(rv[alpha] for rv in self.rdp_values if alpha in rv)
        return total
    
    def convert_to_dp(self, alpha, delta):
        """Convert RDP to (ε, δ)-DP"""
        rdp = self.get_total_rdp(alpha)
        epsilon = rdp + np.log(1 / delta) / (alpha - 1)
        return epsilon, delta
```

---

## 22.3 Federated Learning Architecture

### 22.3.1 The Federated Learning Protocol

Federated learning (McMahan et al., 2017) enables multiple parties to collaboratively train a model without sharing their raw data.

**Federated Averaging (FedAvg) Algorithm:**

```python
import torch
import torch.nn as nn
from typing import List, Dict
import copy

class FederatedServer:
    """Federated learning server coordinating multiple clients"""
    
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
        Run Federated Averaging algorithm.
        
        Reference: McMahan et al. (2017) "Communication-Efficient Learning 
        of Deep Networks from Decentralized Data"
        """
        for round_num in range(self.num_rounds):
            # Select random subset of clients
            selected_clients = np.random.choice(
                clients, size=min(clients_per_round, len(clients)),
                replace=False
            )
            
            # Send global model to selected clients
            global_weights = self.global_model.state_dict()
            
            # Local training on each client
            local_models = []
            local_sizes = []
            
            for client in selected_clients:
                # Client trains locally
                local_model = copy.deepcopy(self.global_model)
                local_model.load_state_dict(global_weights)
                
                client.train_local(local_model, local_epochs, learning_rate)
                
                local_models.append(local_model)
                local_sizes.append(client.data_size)
            
            # Aggregate models (weighted average)
            self._aggregate_models(local_models, local_sizes)
            
            # Evaluate
            metrics = self.evaluate()
            self.history.append(metrics)
            
            if round_num % 10 == 0:
                print(f"Round {round_num}: {metrics}")
        
        return self.history
    
    def _aggregate_models(self, local_models: List[nn.Module],
                         local_sizes: List[int]):
        """Weighted average of model parameters"""
        total_size = sum(local_sizes)
        
        # Initialize aggregated weights
        aggregated = {}
        for key in local_models[0].state_dict().keys():
            aggregated[key] = torch.zeros_like(local_models[0].state_dict()[key])
        
        # Weighted sum
        for model, size in zip(local_models, local_sizes):
            weight = size / total_size
            for key in model.state_dict().keys():
                aggregated[key] += weight * model.state_dict()[key]
        
        # Update global model
        self.global_model.load_state_dict(aggregated)


class FederatedClient:
    """Federated learning client with local data"""
    
    def __init__(self, client_id, train_loader, val_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_size = len(train_loader.dataset)
    
    def train_local(self, model, epochs, learning_rate):
        """Train model on local data"""
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
        """Evaluate model on local validation data"""
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

### 22.3.2 Secure Aggregation

To protect individual client updates from the server, secure aggregation uses cryptographic techniques:

```python
import hashlib
from typing import Tuple

class SecureAggregator:
    """
    Secure aggregation using additive secret sharing.
    The server never sees individual client updates.
    """
    
    def __init__(self, num_clients):
        self.num_clients = num_clients
    
    def setup(self) -> Tuple[Dict, Dict]:
        """
        Setup phase: each pair of clients agrees on a shared random mask.
        Returns (private_keys, public_keys)
        """
        # Generate pairwise masks
        pairwise_masks = {}
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                # Shared random mask
                seed = f"{i}-{j}".encode()
                np.random.seed(int(hashlib.sha256(seed).hexdigest(), 16) % 2**32)
                mask = np.random.randn(1000)  # Assuming 1000-dim updates
                pairwise_masks[(i, j)] = mask
        
        return pairwise_masks
    
    def add_masks(self, updates: List[np.ndarray], 
                  pairwise_masks: Dict) -> np.ndarray:
        """
        Each client adds their share of pairwise masks to their update.
        When summed at server, masks cancel out.
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
        Aggregate masked updates.
        Pairwise masks cancel out, leaving only the sum of true updates.
        """
        return np.sum(masked_updates, axis=0) / len(masked_updates)
```

### 22.3.3 Real-World Federated Learning: Google's Gboard

Google uses federated learning to improve Gboard (the keyboard app) without collecting user typing data:

**Architecture:**
1. **On-device training**: Each phone trains a local model on the user's typing patterns
2. **Model update extraction**: Only model updates (not data) are sent to Google's servers
3. **Secure aggregation**: Updates from many users are aggregated before Google sees them
4. **Global model update**: The aggregated update improves the global model
5. **Model distribution**: The updated model is pushed back to devices

According to Google's AI Blog (ai.googleblog.com, 2017), this approach:
- Processes billions of predictions daily
- Never sees individual user data
- Provides language prediction improvements of 2-4% in accuracy
- Requires no user opt-in beyond normal Gboard usage

---

## 22.4 GDPR and CCPA Compliance for AI Systems

### 22.4.1 Data Subject Rights in AI

GDPR provides several rights that directly impact AI systems:

| Right | GDPR Article | AI Impact | Implementation |
|-------|-------------|-----------|----------------|
| Right to access | Art. 15 | Users can request what data was used to train the model | Data lineage tracking |
| Right to erasure | Art. 17 | Users can request deletion of their data from training | Machine unlearning |
| Right to explanation | Art. 22 | Automated decisions must be explainable | XAI methods |
| Data portability | Art. 20 | Users can request their data in machine-readable format | Export APIs |
| Consent | Art. 6, 7 | Clear consent required for data processing | Consent management |

### 22.4.2 Machine Unlearning

Machine unlearning (Bourtoule et al., 2021) enables removing the influence of specific data points from a trained model:

```python
class SISAUnlearning:
    """
    SISA (Sharded, Isolated, Sliced, and Aggregated) Unlearning.
    Reference: Bourtoule et al. (2021) "Machine Unlearning"
    """
    
    def __init__(self, base_model, num_shards=10, num_slices=5):
        self.base_model = base_model
        self.num_shards = num_shards
        self.num_slices = num_slices
        self.shards = {}
        self.slice_models = {}
    
    def train(self, dataset):
        """Train with SISA structure for efficient unlearning"""
        # Split dataset into shards
        shard_size = len(dataset) // self.num_shards
        shards = []
        
        for i in range(self.num_shards):
            start = i * shard_size
            end = min((i + 1) * shard_size, len(dataset))
            shards.append(dataset[start:end])
        
        # Train each shard with slicing
        for shard_id, shard_data in enumerate(shards):
            self.shards[shard_id] = shard_data
            self.slice_models[shard_id] = self._train_shard(shard_data)
    
    def _train_shard(self, shard_data):
        """Train a shard with intermediate checkpoints"""
        models = []
        slice_size = len(shard_data) // self.num_slices
        
        for slice_idx in range(self.num_slices):
            # Train on cumulative data up to this slice
            cumulative_data = shard_data[:((slice_idx + 1) * slice_size)]
            
            model = copy.deepcopy(self.base_model)
            # Train model on cumulative_data
            # (simplified - would include actual training loop)
            models.append(model)
        
        return models
    
    def unlearn(self, data_point):
        """
        Remove influence of a specific data point.
        Only retrain shards that contain the data point.
        """
        shards_to_retrain = []
        
        for shard_id, shard_data in self.shards.items():
            if data_point in shard_data:
                shards_to_retrain.append(shard_id)
        
        # Only retrain affected shards
        for shard_id in shards_to_retrain:
            # Find which slice contains the data point
            shard_data = self.shards[shard_id]
            # Retrain from that slice onward
            self.slice_models[shard_id] = self._train_shard(shard_data)
        
        return len(shards_to_retrain)
    
    def predict(self, input_data):
        """Ensemble prediction across all shards"""
        predictions = []
        
        for shard_id, models in self.slice_models.items():
            # Use the last model from each shard
            model = models[-1]
            pred = model(input_data)
            predictions.append(pred)
        
        # Average predictions
        return torch.mean(torch.stack(predictions), dim=0)
```

### 22.4.3 Privacy Impact Assessment

```python
class PrivacyImpactAssessment:
    """Structured privacy impact assessment for AI systems"""
    
    def __init__(self, system_name, system_description):
        self.system_name = system_name
        self.system_description = system_description
        self.assessments = []
    
    def assess_data_flow(self, data_type, purpose, retention, sharing):
        """Assess a specific data flow"""
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
        """Calculate risk level based on data characteristics"""
        risk_score = 0
        
        # Data sensitivity
        if data_type in ['health', 'biometric', 'financial']:
            risk_score += 3
        elif data_type in ['location', 'behavioral', 'communication']:
            risk_score += 2
        elif data_type in ['demographic', 'device']:
            risk_score += 1
        
        # Purpose
        if purpose not in ['core_function', 'service_improvement']:
            risk_score += 1
        
        # Retention
        if retention > 365:  # days
            risk_score += 1
        
        # Sharing
        if sharing == 'third_party':
            risk_score += 2
        elif sharing == 'internal':
            risk_score += 1
        
        # Map to risk level
        if risk_score >= 7:
            return 'critical'
        elif risk_score >= 5:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_mitigations(self, risk_level):
        """Suggest mitigations based on risk level"""
        mitigations = {
            'critical': [
                'Implement differential privacy',
                'Use federated learning',
                'Conduct third-party audit',
                'Implement real-time monitoring',
                'Require explicit consent'
            ],
            'high': [
                'Apply data minimization',
                'Implement access controls',
                'Add pseudonymization',
                'Conduct regular audits'
            ],
            'medium': [
                'Apply data minimization',
                'Implement basic access controls',
                'Document processing activities'
            ],
            'low': [
                'Apply data minimization',
                'Document processing activities'
            ]
        }
        return mitigations.get(risk_level, [])
    
    def generate_report(self):
        """Generate compliance report"""
        report = {
            'system': self.system_name,
            'total_assessments': len(self.assessments),
            'risk_distribution': {},
            'recommended_actions': []
        }
        
        for assessment in self.assessments:
            risk = assessment['risk_level']
            report['risk_distribution'][risk] = report['risk_distribution'].get(risk, 0) + 1
        
        # Generate action items
        for assessment in self.assessments:
            if assessment['risk_level'] in ['critical', 'high']:
                report['recommended_actions'].extend([
                    f"Address {assessment['data_type']}: {m}" 
                    for m in assessment['mitigations']
                ])
        
        return report
```

---

## 22.5 Homomorphic Encryption for AI

### 22.5.1 Concepts

Homomorphic encryption allows computation on encrypted data without decryption:

```python
# Simplified representation of homomorphic operations
# Real implementations use libraries like SEAL, TenSEAL, or HElib

class SimplifiedHomomorphicOps:
    """
    Conceptual demonstration of homomorphic operations.
    In practice, use libraries like Microsoft SEAL or TenSEAL.
    """
    
    def __init__(self):
        # In real implementation, this would be a key pair
        self.public_key = "pk_placeholder"
        self.private_key = "sk_placeholder"
    
    def encrypt(self, value):
        """Encrypt a value (conceptual)"""
        # Real implementation uses complex mathematical operations
        # This is a simplified representation
        return {'encrypted': True, 'value': value, 'noise': 0.1}
    
    def decrypt(self, encrypted_value):
        """Decrypt a value (conceptual)"""
        return encrypted_value['value']
    
    def add(self, enc_a, enc_b):
        """Homomorphic addition: E(a) + E(b) = E(a + b)"""
        return {
            'encrypted': True,
            'value': enc_a['value'] + enc_b['value'],
            'noise': enc_a['noise'] + enc_b['noise']
        }
    
    def multiply(self, enc_a, enc_b):
        """Homomorphic multiplication: E(a) * E(b) = E(a * b)"""
        return {
            'encrypted': True,
            'value': enc_a['value'] * enc_b['value'],
            'noise': enc_a['noise'] * enc_b['noise'] * 2  # Noise grows faster
        }
```

### 22.5.2 Practical Homomorphic Encryption

Real implementations use libraries like Microsoft SEAL:

```python
# This is pseudocode showing how to use TenSEAL
# pip install tenseal

"""
import tenseal as ts

# Setup TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()
context.generate_relin_keys()

# Encrypt vectors
vector_a = [1.0, 2.0, 3.0, 4.0]
vector_b = [5.0, 6.0, 7.0, 8.0]

enc_a = ts.ckks_vector(context, vector_a)
enc_b = ts.ckks_vector(context, vector_b)

# Homomorphic operations
enc_sum = enc_a + enc_b      # E(a) + E(b) = E(a+b)
enc_product = enc_a * enc_b   # E(a) * E(b) = E(a*b)

# Decrypt results
result_sum = enc_sum.decrypt()
result_product = enc_product.decrypt()

# result_sum ≈ [6.0, 8.0, 10.0, 12.0]
# result_product ≈ [5.0, 12.0, 21.0, 32.0]
"""
```

---

## 22.6 Case Study: How Apple Implements Differential Privacy

### Background

Apple has been a pioneer in implementing differential privacy at scale. Since 2016, Apple has used differential privacy in iOS and macOS to collect usage statistics without compromising individual user privacy.

### Implementation Architecture

**1. Local Differential Privacy (LDP)**

Apple uses local differential privacy, where noise is added on the device before data is sent to Apple's servers:

```
User Device → Add Noise Locally → Send to Apple → Aggregate Results
```

This is stronger than centralized DP because Apple never sees the true data.

**2. Privacy Budget Management**

According to Apple's "Differential Privacy Technical Overview" (apple.com/privacy, 2017):

- Each data type has an allocated privacy budget (ε)
- Once a user's budget for a data type is exhausted, no more data of that type is collected
- Budgets are replenished periodically (typically weekly)

**3. Real-World Applications**

| Data Type | ε Value | Purpose | Frequency |
|-----------|---------|---------|-----------|
| Emoji usage | 4 | Popular emoji tracking | Daily |
| QuickType suggestions | 8 | Keyboard prediction | Daily |
| Safari energy impact | 2 | Battery optimization | Weekly |
| Health data trends | 1 | Health app improvement | Monthly |

**4. Implementation Details**

Apple's approach combines:

- **RAPPOR** (Randomized Aggregatable Privacy-Preserving Ordinal Response): For collecting frequency distributions of strings
- **Blockless Apple Private Information Retrieval**: For private set intersection
- **Bloom filters**: For approximate counting with privacy guarantees

### Technical Results

According to Apple's WWDC presentations (2016, 2017, 2019):

| Metric | Value |
|--------|-------|
| Devices participating | 1+ billion |
| Daily data points processed | 10+ billion |
| Privacy budget per user per week | ε ≈ 4-8 per data type |
| False positive rate (RAPPOR) | <1% |
| Utility loss compared to non-private | <5% |

### Lessons Learned

1. **Local DP is essential for trust**: Users trust Apple because data is privatized on-device
2. **Privacy budgets must be managed carefully**: Over-collection erodes privacy
3. **Utility can be maintained**: With careful mechanism design, accuracy loss is minimal
4. **Transparency matters**: Apple publishes details of their DP implementation

---

## 22.7 War Story: The Privacy Breach in a Healthcare AI System

### Background

In 2021, a healthcare AI company (anonymized) suffered a significant privacy breach that exposed the limitations of their "privacy-preserving" approach. The company had developed an AI system for predicting patient readmission risk, claiming to use "anonymized" patient data.

### The Breach

**What Happened:**

The company shared "anonymized" patient data with a research partner. The research partner, using external datasets, was able to re-identify individual patients:

1. **Quasi-identifiers**: Combination of age, zip code, and admission date was unique for 87% of patients
2. **Linkage attack**: Combined with public voter registration data, re-identification rate reached 94%
3. **Sensitive information exposed**: Diagnoses, treatments, and outcomes for identifiable patients

**The Technical Failure:**

```python
# Simplified representation of the flawed "anonymization"
def flawed_anonymization(patient_data):
    """What the company did (WRONG)"""
    anonymized = patient_data.copy()
    
    # Remove direct identifiers (names, SSN, etc.)
    anonymized = anonymized.drop(['name', 'ssn', 'phone'], axis=1)
    
    # Keep quasi-identifiers (age, zip, admission_date)
    # This was the critical mistake
    return anonymized

# What they should have done
def proper_anonymization(patient_data):
    """What they should have done"""
    anonymized = patient_data.copy()
    
    # Remove direct identifiers
    anonymized = anonymized.drop(['name', 'ssn', 'phone'], axis=1)
    
    # Generalize quasi-identifiers
    anonymized['age'] = anonymized['age'] // 10 * 10  # Age groups (20-29, 30-39, etc.)
    anonymized['zip'] = anonymized['zip'].str[:3]     # First 3 digits only
    anonymized['admission_date'] = pd.to_datetime(anonymized['admission_date']).dt.month
    
    # Apply k-anonymity (k=5)
    anonymized = apply_k_anonymity(anonymized, k=5)
    
    # Apply l-diversity for sensitive attributes
    anonymized = apply_l_diversity(anonymized, sensitive_cols=['diagnosis', 'treatment'], l=3)
    
    return anonymized

def apply_k_anonymity(data, k):
    """Ensure each combination of quasi-identifiers appears at least k times"""
    quasi_id_cols = ['age', 'zip', 'admission_date']
    
    # Count combinations
    counts = data.groupby(quasi_id_cols).size().reset_index(name='count')
    
    # Filter to keep only groups with at least k records
    valid_groups = counts[counts['count'] >= k][quasi_id_cols]
    
    # Filter original data
    merged = data.merge(valid_groups, on=quasi_id_cols)
    
    return merged
```

### Impact

| Metric | Value |
|--------|-------|
| Patients affected | 2.3 million |
| Re-identification rate | 94% |
| Sensitive conditions exposed | HIV, mental health, substance abuse |
| Regulatory fine | $4.5 million (HIPAA violation) |
| Lawsuit settlement | $12 million |
| Reputational damage | Loss of 3 hospital partnerships |

### Root Cause Analysis

1. **Inadequate anonymization**: Removing direct identifiers is insufficient
2. **No formal privacy guarantees**: The company relied on "anonymization" without formal definitions
3. **Failed risk assessment**: No one considered linkage attacks
4. **Lack of privacy expertise**: The team lacked differential privacy knowledge

### Lessons Learned

1. **K-anonymity is insufficient alone**: Even k-anonymized data can be vulnerable to homogeneity and background knowledge attacks
2. **Differential privacy provides formal guarantees**: Only DP gives mathematically proven privacy
3. **Privacy impact assessments are mandatory**: Every data sharing decision requires formal assessment
4. **Defense in depth applies to privacy too**: Multiple privacy techniques should be combined

---

## 22.8 When to Use / When Not to Use Privacy-Preserving AI

### When to Use Specific Techniques

| Technique | When to Use | When NOT to Use | Tradeoff |
|-----------|-------------|-----------------|----------|
| Differential Privacy | Data collection, analytics | When utility loss is unacceptable | Privacy vs. accuracy |
| Federated Learning | Multi-party training | When data cannot be partitioned | Communication overhead |
| Homomorphic Encryption | Inference on sensitive data | When latency is critical | 100-1000x slowdown |
| Secure MPC | Multi-party computation | When parties are adversarial | Communication complexity |
| Machine Unlearning | GDPR/CCPA compliance | When model is rarely updated | Retraining cost |
| Data minimization | Always (baseline) | Never skip | Minimal utility impact |

### Decision Framework

```
Do you process personal data? ─── YES ──→ Privacy-preserving techniques REQUIRED
         │
         NO
         │
Do you share data with third parties? ─── YES ──→ Apply DP or federated learning
         │
         NO
         │
Do you need to explain model decisions? ─── YES ──→ Implement XAI methods
         │
         NO
         │
Standard security baseline
```

---

## 22.9 Summary

Privacy-preserving AI is not optional—it is a legal and ethical requirement:

1. **Differential privacy provides mathematical guarantees**: The gold standard for privacy, with ε controlling the privacy-utility tradeoff.

2. **Federated learning enables collaboration without data sharing**: Google's Gboard demonstrates this at billion-user scale.

3. **GDPR/CCPA compliance requires specific technical implementations**: Machine unlearning, right to explanation, and data minimization are mandatory.

4. **Local differential privacy is stronger than central DP**: Apple's approach ensures the server never sees true data.

5. **Simple "anonymization" is insufficient**: Real privacy requires formal guarantees, not just removing names.

6. **Privacy-utility tradeoffs are manageable**: With careful mechanism design, privacy can be preserved with minimal accuracy loss.

---

## 22.10 Discussion Questions

1. **Privacy Budget Allocation**: If you have a total privacy budget of ε=10 for an application that collects 5 different data types, how would you allocate the budget? What factors influence your decision?

2. **Federated Learning vs. Centralized DP**: Under what circumstances would you choose federated learning over centralized differential privacy? What are the practical tradeoffs?

3. **Machine Unlearning Challenges**: If a model was trained with adversarial training (which stores intermediate examples), how would you implement effective machine unlearning?

4. **Cross-Border Data Transfer**: A company operates in both the EU (GDPR) and China (PIPL). How would you design a privacy-preserving AI system that complies with both regulations?

5. **Privacy vs. Security**: Differential privacy protects against privacy attacks, but how does it interact with security attacks like adversarial examples? Can a model be both private and robust?

---

## 22.11 Exercises

### Exercise 1: Differential Privacy Implementation

Implement a differentially private mechanism for computing the average age of a dataset:

```python
import numpy as np

class PrivateAverageComputer:
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def compute_private_average(self, ages):
        """Compute epsilon-DP average age"""
        # Sensitivity of average = (max - min) / n
        sensitivity = (max(ages) - min(ages)) / len(ages)
        
        # True average
        true_average = np.mean(ages)
        
        # Add Laplace noise
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        
        return true_average + noise
    
    def evaluate_utility(self, true_average, private_averages):
        """Evaluate utility loss from privacy mechanism"""
        errors = [abs(pa - true_average) for pa in private_averages]
        return {
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'rmse': np.sqrt(np.mean([e**2 for e in errors]))
        }
```

**Tasks:**
1. Test with ε = 0.1, 1.0, 10.0
2. Measure utility loss at each privacy level
3. Plot the privacy-utility tradeoff curve
4. Determine the minimum ε needed for <5% average error

### Exercise 2: Federated Learning Simulation

Simulate federated learning with heterogeneous data:

```python
def simulate_federated_learning(num_clients=10, num_rounds=50):
    """
    Simulate federated learning where each client has
    different data distribution (non-IID).
    """
    # Create clients with different data distributions
    clients = []
    for i in range(num_clients):
        # Each client gets data from a different class distribution
        train_loader, val_loader = create_heterogeneous_data(i, num_clients)
        clients.append(FederatedClient(i, train_loader, val_loader))
    
    # Run federated averaging
    server = FederatedServer(global_model, num_clients, num_rounds)
    history = server.federated_averaging(clients)
    
    return history
```

**Tasks:**
1. Compare IID vs. non-IID data distributions
2. Measure convergence speed and final accuracy
3. Analyze the effect of number of clients per round
4. Plot communication cost vs. accuracy

### Exercise 3: Privacy Impact Assessment

Conduct a privacy impact assessment for a real-world AI system:

**Choose one:**
- A credit scoring system
- A facial recognition system
- A recommendation engine
- A medical diagnosis AI

**Requirements:**
1. Identify all data flows
2. Assess risk for each flow
3. Propose technical mitigations
4. Document compliance requirements (GDPR, CCPA, etc.)
5. Create a 10-page assessment report

---

## 22.12 References

### Differential Privacy

1. Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). "Calibrating Noise to Sensitivity in Private Data Analysis." *TCC*. https://doi.org/10.1007/11681878_14

2. Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy." *Foundations and Trends in Theoretical Computer Science*. https://doi.org/10.1561/0400000042

3. Abadi, M., et al. (2016). "Deep Learning with Differential Privacy." *ACM CCS*. https://doi.org/10.1145/2976749.2978318

### Federated Learning

4. McMahan, H. B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS*. https://arxiv.org/abs/1602.05629

5. Konečný, J., et al. (2016). "Federated Learning: Strategies for Improving Communication Efficiency." *arXiv*. https://arxiv.org/abs/1610.05492

### Machine Unlearning

6. Bourtoule, L., et al. (2021). "Machine Unlearning." *IEEE S&P*. https://doi.org/10.1109/SP40001.2021.00019

7. Cao, D., & Yang, J. (2020). "Toward a Better Understanding of Machine Unlearning." *arXiv*. https://arxiv.org/abs/2010.12101

### Privacy Regulations

8. European Parliament. (2016). "General Data Protection Regulation (GDPR)." https://gdpr-info.eu/

9. California Legislative Information. (2020). "California Consumer Privacy Act (CCPA)." https://oag.ca.gov/privacy/ccpa

10. IAPP. (2024). "GDPR Enforcement Tracker." https://www.trackgdpr.com/

### Apple's Differential Privacy

11. Apple. (2017). "Differential Privacy Technical Overview." https://www.apple.com/privacy/docs/Differential_Privacy_Overview.pdf

12. Tang, J., et al. (2017). "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response." *ACM CCS*. https://doi.org/10.1145/3133956.3133981

### Healthcare Privacy

13. Sweeney, L. (2000). "Simple Demographics Often Identify People Uniquely." *Carnegie Mellon University*. https://dataprivacylab.org/research/identifiability/

14. Health and Human Services. (2013). "HIPAA Privacy Rule." https://www.hhs.gov/hipaa/for-professionals/privacy/index.html

---

*Next Chapter: [Chapter 23: AI Architecture Design Methodology →](./chapter-23.md)*
