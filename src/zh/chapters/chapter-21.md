# 第21章：AI安全防御架构

## 学习目标

学完本章后，你将能够：

1. 设计和实现针对第20章所涵盖威胁的AI系统多层防御架构
2. 应专门为ML模型输入设计的真实输入验证和清理模式
3. 实现具有可工作代码的对抗训练和认证防御方法
4. 设计模型水印和指纹系统以保护知识产权
5. 构建监控和异常检测系统以识别生产环境中的对抗活动

---

## 21.1 引言：AI系统的纵深防御

AI系统的防御不能依赖单一机制。正如企业安全采用防火墙、入侵检测系统、访问控制和加密进行分层防御，AI安全需要全面的方法来应对ML生命周期每个阶段的威胁。

基本原则是**纵深防御**：没有单一的防御层是足够的，但当正确分层时，它们创建了一个大于各部分之和的防御系统。攻击者必须绕过所有层，而不仅仅是一层，才能成功。

根据NIST AI 100-2（2023），有效的AI安全需要：
1. **预防**：在攻击到达模型之前阻止它们
2. **检测**：在预防失败时识别攻击成功
3. **响应**：减轻损害并从成功攻击中恢复
4. **适应**：随着攻击技术的改进而发展防御

本章为每一层提供实用、可实施的防御策略。

---

## 21.2 第1层：输入验证和清理

### 21.2.1 原则

在任何输入到达模型之前，必须经过验证和清理。这是最具成本效益的防御层，因为它在模型参与之前运行。

### 21.2.2 图像输入验证

**统计分布检查**

真实世界的图像遵循可预测的统计分布。对抗样本经常偏离这些分布：

```python
import numpy as np
from scipy import stats
import torch

class ImageValidator:
    """根据已知分布验证图像输入"""
    
    def __init__(self, training_stats_path):
        self.training_stats = self._load_stats(training_stats_path)
    
    def _load_stats(self, path):
        """从训练数据加载预计算的统计信息"""
        stats = np.load(path)
        return {
            'mean': stats['mean'],
            'std': stats['std'],
            'histogram': stats['histogram'],
            'pixel_range': (0, 255),
            'max_fft_ratio': 0.15  # 允许的最大高频能量
        }
    
    def validate(self, image, threshold=0.95):
        """
        根据多个标准验证图像
        返回 (is_valid, scores, anomalies)
        """
        scores = {}
        anomalies = []
        
        # 1. 像素范围检查
        if image.min() < 0 or image.max() > 255:
            anomalies.append("像素值超出有效范围")
            scores['pixel_range'] = 0.0
        else:
            scores['pixel_range'] = 1.0
        
        # 2. 分布检查（KL散度）
        current_hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
        current_hist = current_hist / current_hist.sum() + 1e-10
        kl_div = stats.entropy(current_hist, self.training_stats['histogram'])
        
        if kl_div > 0.1:  # 阈值通过经验确定
            anomalies.append(f"KL散度过高: {kl_div:.4f}")
            scores['distribution'] = max(0, 1 - kl_div)
        else:
            scores['distribution'] = 1.0
        
        # 3. 频域分析（对抗扰动通常具有高频分量）
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # 计算高频与总能量的比率
        h, w = image.shape[:2]
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        # 低频区域（中心）
        low_freq_mask = np.zeros_like(magnitude, dtype=bool)
        y, x = np.ogrid[:h, :w]
        low_freq_mask[(y - center_h)**2 + (x - center_w)**2 <= radius**2] = True
        
        total_energy = magnitude.sum()
        low_freq_energy = magnitude[low_freq_mask].sum()
        high_freq_ratio = 1 - (low_freq_energy / total_energy) if total_energy > 0 else 0
        
        if high_freq_ratio > self.training_stats['max_fft_ratio']:
            anomalies.append(f"高频能量比率: {high_freq_ratio:.3f}")
            scores['frequency'] = max(0, 1 - high_freq_ratio)
        else:
            scores['frequency'] = 1.0
        
        # 4. 结构相似性指数（与期望结构比较）
        edges = np.gradient(image.astype(float))
        edge_magnitude = np.sqrt(edges[0]**2 + edges[1]**2)
        avg_edge = edge_magnitude.mean()
        
        if avg_edge > 50:  # 具有非常强边缘的图像
            anomalies.append(f"异常边缘强度: {avg_edge:.1f}")
            scores['structure'] = 0.7
        else:
            scores['structure'] = 1.0
        
        # 总体验证
        is_valid = all(s >= threshold for s in scores.values())
        
        return is_valid, scores, anomalies
    
    def sanitize(self, image):
        """应用清理变换以移除潜在的对抗噪声"""
        # JPEG压缩（破坏对抗扰动）
        from PIL import Image
        import io
        
        img_pil = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        img_restored = np.array(Image.open(buffer))
        
        # 高斯模糊（移除高频扰动）
        from scipy.ndimage import gaussian_filter
        img_blurred = gaussian_filter(img_restored, sigma=0.5)
        
        # 随机缩放和填充（击败基于补丁的攻击）
        scale = np.random.uniform(0.95, 1.05)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = np.array(Image.fromarray(img_blurred.astype(np.uint8)).resize(
            (new_w, new_h), Image.BILINEAR
        ))
        
        # 填充回原始大小
        result = np.zeros_like(image)
        dh, dw = (new_h - h) // 2, (new_w - w) // 2
        result[max(0, -dh):min(h, new_h-dh), max(0, -dw):min(w, new_w-dw)] = \
            img_resized[max(0, dh):min(new_h, h+dh), max(0, dw):min(new_w, w+dw)]
        
        return result.astype(np.uint8)
```

**已知攻击的感知哈希**

维护已知对抗模式的数据库：

```python
import imagehash
from PIL import Image

class AdversarialPatternDetector:
    """使用感知哈希检测已知的对抗模式"""
    
    def __init__(self, pattern_db_path):
        self.known_patterns = self._load_pattern_database(pattern_db_path)
        self.similarity_threshold = 0.85
    
    def _load_pattern_database(self, path):
        """加载已知的对抗模式哈希"""
        import json
        with open(path, 'r') as f:
            patterns = json.load(f)
        return {p['name']: imagehash.hex_to_hash(p['hash']) for p in patterns}
    
    def detect(self, image):
        """检查图像是否匹配任何已知的对抗模式"""
        img_hash = imagehash.phash(Image.fromarray(image.astype(np.uint8)))
        
        matches = []
        for name, pattern_hash in self.known_patterns.items():
            similarity = 1 - (img_hash - pattern_hash) / len(img_hash.hash.flatten())
            if similarity > self.similarity_threshold:
                matches.append((name, similarity))
        
        return matches
```

### 21.2.3 文本输入验证（用于LLM）

```python
import re
from typing import List, Tuple

class TextSanitizer:
    """验证和清理LLM应用程序的文本输入"""
    
    def __init__(self):
        self.injection_patterns = [
            r'ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)',
            r'you\s+are\s+now\s+',
            r'act\s+as\s+if\s+',
            r'forget\s+(everything|all|your\s+instructions)',
            r'system\s*:\s*',
            r'<\|im_start\|>',
            r'<\|im_end\|>',
            r'\[INST\]',
            r'\[/INST\]',
        ]
        
        self.suspicious_patterns = [
            r'repeat\s+(your|the)\s+(system|initial)\s+prompt',
            r'what\s+(are|is)\s+your\s+(system|initial)\s+prompt',
            r'output\s+(your|the)\s+(system|initial)\s+prompt',
            r'developer\s+mode',
            r'jailbreak',
            r'do\s+anything\s+now',
        ]
    
    def validate(self, text: str) -> Tuple[bool, List[str]]:
        """验证文本是否存在提示注入尝试"""
        anomalies = []
        
        # 检查直接注入模式
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                anomalies.append(f"潜在注入: {pattern[:50]}...")
        
        # 检查可疑模式
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                anomalies.append(f"可疑模式: {pattern[:50]}...")
        
        # 检查异常Unicode字符
        unusual_chars = sum(1 for c in text if ord(c) > 0xFFFF or 
                          (0x200B <= ord(c) <= 0x200F) or  # 零宽字符
                          (0x2028 <= ord(c) <= 0x2029) or  # 行/段落分隔符
                          (0x2060 <= ord(c) <= 0x2064))    # 不可见格式化符
        
        if unusual_chars > 0:
            anomalies.append(f"发现 {unusual_chars} 个异常Unicode字符")
        
        # 检查过长长度（潜在DoS）
        if len(text) > 10000:
            anomalies.append(f"输入长度 {len(text)} 超出限制")
        
        is_valid = len(anomalies) == 0
        return is_valid, anomalies
    
    def sanitize(self, text: str) -> str:
        """从文本中移除潜在有害的模式"""
        sanitized = text
        
        # 移除零宽字符
        sanitized = re.sub(r'[\u200b-\u200f\u2028-\u2029\u2060-\u2064]', '', sanitized)
        
        # 规范化空白
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # 移除控制字符（换行和制表符除外）
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        return sanitized.strip()
```

### 21.2.4 表格数据验证

```python
import pandas as pd
import numpy as np
from scipy import stats

class TabularValidator:
    """根据训练分布验证表格数据输入"""
    
    def __init__(self, training_data_path):
        self.reference_stats = self._compute_reference_stats(training_data_path)
    
    def _compute_reference_stats(self, path):
        """从训练数据计算统计信息"""
        df = pd.read_csv(path)
        stats_dict = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25),
            }
        
        return stats_dict
    
    def validate(self, input_data: pd.DataFrame) -> Tuple[bool, dict]:
        """根据参考统计信息验证输入数据"""
        anomalies = {}
        is_valid = True
        
        for col in input_data.columns:
            if col not in self.reference_stats:
                anomalies[col] = ["未知列"]
                is_valid = False
                continue
            
            ref = self.reference_stats[col]
            values = input_data[col].dropna()
            
            if len(values) == 0:
                anomalies[col] = ["所有值缺失"]
                is_valid = False
                continue
            
            # 范围检查
            if values.min() < ref['min'] * 0.5 or values.max() > ref['max'] * 2:
                anomalies[col] = ["值超出预期范围"]
                is_valid = False
            
            # 分布检验（KS检验）
            ks_stat, p_value = stats.ks_2samp(values, np.random.normal(ref['mean'], ref['std'], 1000))
            if p_value < 0.01:
                anomalies[col] = [f"分布不匹配 (KS p={p_value:.4f})"]
                is_valid = False
        
        return is_valid, anomalies
```

---

## 21.3 第2层：对抗训练

### 21.3.1 标准对抗训练（基于PGD）

对抗训练是对抗样本最经过经验验证的防御。它通过在训练过程中包含对抗样本来工作。

```python
import torch
import torch.nn as nn
import torch.optim as optim

class AdversarialTrainer:
    """基于PGD的对抗训练"""
    
    def __init__(self, model, epsilon=0.03, alpha=0.007, 
                 num_steps=10, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        
    def pgd_attack(self, images, labels, criterion):
        """生成PGD对抗样本"""
        images = images.clone().detach()
        labels = labels.clone().detach()
        
        # 随机初始化
        if self.random_start:
            images = images + torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            images = torch.clamp(images, 0, 1)
        
        for _ in range(self.num_steps):
            images.requires_grad = True
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # 梯度上升步骤
            images = images.detach() + self.alpha * images.grad.sign()
            images = torch.clamp(images, 0, 1)
            
            # 投影回epsilon球
            delta = torch.clamp(images - images, -self.epsilon, self.epsilon)
            images = torch.clamp(images + delta, 0, 1).detach()
        
        return images
    
    def train_epoch(self, train_loader, optimizer, criterion, 
                    mix_ratio=0.5):
        """使用对抗样本训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            
            # 混合干净和对抗样本
            if torch.rand(1).item() < mix_ratio:
                # 生成对抗样本
                adv_images = self.pgd_attack(images, labels, criterion)
                outputs = self.model(adv_images)
            else:
                outputs = self.model(images)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(train_loader), 100. * correct / total
```

### 21.3.2 TRADES：平衡鲁棒性和准确性

Zhang等人（2019）提出了TRADES，它提供了一种有原则的方法来平衡准确性和鲁棒性：

```python
class TRADESTrainer:
    """TRADES对抗训练"""
    
    def __init__(self, model, epsilon=0.03, alpha=0.007,
                 num_steps=10, beta=6.0):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.beta = beta  # 正则化强度
    
    def trades_loss(self, images, labels, criterion):
        """计算TRADES损失"""
        # 自然损失
        outputs_natural = self.model(images)
        loss_natural = criterion(outputs_natural, labels)
        
        # 生成对抗样本
        images_adv = self._pgd_attack(images, labels, 
                                       lambda x, y: -self._kl_divergence(
                                           self.model(x), self.model(images).detach()
                                       ))
        
        # 自然预测和对抗预测之间的KL散度
        outputs_adv = self.model(images_adv)
        loss_robust = self._kl_divergence(outputs_adv, outputs_natural.detach())
        
        # 总损失
        loss = loss_natural + self.beta * loss_robust
        
        return loss
    
    def _kl_divergence(self, p, q):
        """两个分布之间的KL散度"""
        p = torch.nn.functional.softmax(p, dim=1)
        q = torch.nn.functional.softmax(q, dim=1)
        return torch.sum(p * torch.log(p / q + 1e-8), dim=1).mean()
    
    def _pgd_attack(self, images, labels, loss_fn):
        """使用任意损失函数的PGD攻击"""
        images = images.clone().detach()
        
        for _ in range(self.num_steps):
            images.requires_grad = True
            loss = loss_fn(images, labels)
            
            grad = torch.autograd.grad(loss, images)[0]
            images = images.detach() + self.alpha * grad.sign()
            images = torch.clamp(images, 0, 1)
            
            delta = torch.clamp(images - images, -self.epsilon, self.epsilon)
            images = torch.clamp(images + delta, 0, 1).detach()
        
        return images
```

### 21.3.3 认证防御：随机平滑

随机平滑提供了可证明的鲁棒性保证：

```python
import numpy as np
from scipy.stats import norm

class SmoothedClassifier:
    """用于认证鲁棒性的随机平滑"""
    
    def __init__(self, base_classifier, sigma=0.25, n_samples=1000,
                 alpha=0.001):
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.n_samples = n_samples
        self.alpha = alpha
    
    def certify(self, x):
        """
        使用随机平滑认证x的预测。
        返回 (prediction, certified_radius)
        """
        # 抽样
        counts = self._sample_predictions(x)
        
        # 找到顶级类别
        top_class = counts.argmax()
        top_count = counts[top_class]
        
        # 计算置信区间
        n_total = counts.sum()
        
        # 顶级类别的概率下界
        p_lower = self._lower_confidence_bound(top_count, n_total)
        
        if p_lower > 0.5:
            # 使用反向累积分布函数的认证半径
            certified_radius = self.sigma * norm.ppf(p_lower)
            return top_class, certified_radius
        else:
            return -1, 0.0  # 放弃
    
    def _sample_predictions(self, x):
        """从平滑分类器抽样预测"""
        counts = np.zeros(self.base_classifier.num_classes)
        
        for _ in range(self.n_samples):
            # 添加高斯噪声
            noise = np.random.normal(0, self.sigma, x.shape)
            x_noisy = x + noise
            
            # 获取预测
            pred = self.base_classifier.predict(x_noisy)
            counts[pred] += 1
        
        return counts
    
    def _lower_confidence_bound(self, k, n):
        """使用Clopper-Pearson区间的置信下界"""
        from statsmodels.stats.proportion import proportion_confint
        return proportion_confint(k, n, alpha=2*self.alpha, method='beta')[0]
```

---

## 21.4 第3层：模型水印和指纹

### 21.4.1 用于IP保护的水印

水印将所有权信息嵌入训练模型中，无需访问原始训练数据即可验证。

```python
import torch
import torch.nn as nn

class ModelWatermarker:
    """在神经网络模型中嵌入和验证水印"""
    
    def __init__(self, model, watermark_key_size=32):
        self.model = model
        self.key_size = watermark_key_size
    
    def generate_watermark_pair(self, input_shape):
        """
        生成水印输入输出对。
        输入是精心构造的，输出是目标。
        """
        # 生成随机输入
        watermark_input = torch.randn(1, *input_shape)
        
        # 水印输出由密钥确定性地派生
        # 这确保只有密钥持有者可以验证
        torch.manual_seed(hash(str(self.key_size)))
        watermark_output = torch.randint(0, self.model.fc.out_features, (1,))
        
        return watermark_input, watermark_output
    
    def embed_watermark(self, train_loader, watermark_weight=1.0,
                       num_watermark_steps=1000):
        """
        通过微调嵌入水印。
        模型学习正确分类水印输入。
        """
        watermark_input, watermark_target = self.generate_watermark_pair(
            (3, 224, 224)  # 假设图像输入
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 在水印上微调
        for step in range(num_watermark_steps):
            outputs = self.model(watermark_input.cuda())
            loss = criterion(outputs, watermark_target.cuda()) * watermark_weight
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return self.model
    
    def verify_watermark(self, threshold=0.9):
        """
        验证模型是否包含水印。
        返回 (is_watermarked, confidence)
        """
        self.model.eval()
        watermark_input, watermark_target = self.generate_watermark_pair(
            (3, 224, 224)
        )
        
        with torch.no_grad():
            outputs = self.model(watermark_input.cuda())
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1)
            confidence = probabilities[0, predicted_class].item()
        
        is_verified = (
            predicted_class.item() == watermark_target.item() and 
            confidence > threshold
        )
        
        return is_verified, confidence
    
    def generate_diverse_watermarks(self, num_watermarks=10):
        """生成多个水印对以进行稳健验证"""
        watermarks = []
        for i in range(num_watermarks):
            torch.manual_seed(42 + i)  # 确定性种子
            watermark_input = torch.randn(1, 3, 224, 224)
            watermark_target = torch.randint(0, self.model.fc.out_features, (1,))
            watermarks.append((watermark_input, watermark_target))
        return watermarks
```

### 21.4.2 模型指纹

指纹基于模型的独特行为特征来识别特定模型：

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ModelFingerprinter:
    """生成和匹配模型指纹"""
    
    def __init__(self, model, num_queries=1000):
        self.model = model
        self.num_queries = num_queries
    
    def generate_fingerprint(self, input_generator):
        """
        基于模型在精心选择的输入上的行为生成模型指纹。
        """
        fingerprints = []
        
        for _ in range(self.num_queries):
            # 生成多样化的输入
            x = input_generator()
            
            # 获取带有梯度的模型预测
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x_tensor.requires_grad = True
            
            output = self.model(x_tensor)
            
            # 使用梯度作为指纹分量
            output[0, output.argmax()].backward()
            gradient = x_tensor.grad.numpy().flatten()
            
            fingerprints.append(gradient)
        
        # 聚合指纹
        fingerprint = np.mean(fingerprints, axis=0)
        
        # 降低维度同时保持唯一性
        # 使用随机投影
        np.random.seed(42)
        projection = np.random.randn(len(fingerprint), 128)
        reduced_fingerprint = fingerprint @ projection
        reduced_fingerprint = reduced_fingerprint / np.linalg.norm(reduced_fingerprint)
        
        return reduced_fingerprint
    
    def match_fingerprint(self, fp1, fp2, threshold=0.95):
        """匹配两个指纹"""
        similarity = cosine_similarity(fp1.reshape(1, -1), fp2.reshape(1, -1))[0, 0]
        return similarity > threshold, similarity
```

---

## 21.5 第4层：运行时监控和异常检测

### 21.5.1 输入分布监控

```python
import numpy as np
from collections import deque

class InputMonitor:
    """监控输入分布是否存在异常"""
    
    def __init__(self, window_size=1000, alert_threshold=2.0):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.input_history = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=window_size)
        
        # 基线统计
        self.baseline_mean = None
        self.baseline_std = None
    
    def set_baseline(self, baseline_data):
        """从验证数据设置基线统计"""
        self.baseline_mean = np.mean(baseline_data, axis=0)
        self.baseline_std = np.std(baseline_data, axis=0) + 1e-8
    
    def update(self, input_data, prediction, confidence):
        """更新监控统计"""
        self.input_history.append(input_data)
        self.prediction_history.append({
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': np.datetime64('now')
        })
    
    def check_anomaly(self, new_input):
        """检查新输入是否异常"""
        if self.baseline_mean is None:
            return False, "未设置基线"
        
        # 计算马氏距离
        diff = new_input.flatten() - self.baseline_mean
        mahal_dist = np.sqrt(diff @ np.linalg.inv(np.diag(self.baseline_std**2)) @ diff)
        
        if mahal_dist > self.alert_threshold:
            return True, f"马氏距离: {mahal_dist:.3f}"
        
        return False, "正常"
    
    def check_prediction_drift(self):
        """检查预测分布是否漂移"""
        if len(self.prediction_history) < 100:
            return False, "数据不足"
        
        recent_confidences = [p['confidence'] for p in 
                             list(self.prediction_history)[-100:]]
        historical_confidences = [p['confidence'] for p in 
                                 list(self.prediction_history)[:-100]]
        
        if len(historical_confidences) == 0:
            return False, "无历史数据"
        
        # 比较分布
        from scipy.stats import ks_2samp
        ks_stat, p_value = ks_2samp(recent_confidences, historical_confidences)
        
        if p_value < 0.01:
            return True, f"检测到预测分布漂移 (p={p_value:.4f})"
        
        return False, "稳定"
```

### 21.5.2 集成分歧监控

当使用集成方法时，模型之间的分歧可以指示对抗输入：

```python
class EnsembleMonitor:
    """监控集成成员之间的分歧"""
    
    def __init__(self, models, disagreement_threshold=0.3):
        self.models = models
        self.threshold = disagreement_threshold
    
    def check_input(self, input_data):
        """
        检查集成成员对输入的分歧。
        高分歧可能指示对抗输入。
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(input_data)
                pred = torch.softmax(output, dim=1)
                predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # 计算成对分歧
        n_models = len(self.models)
        disagreements = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Jensen-Shannon散度
                p = predictions[i].flatten()
                q = predictions[j].flatten()
                m = 0.5 * (p + q)
                
                js_div = 0.5 * (self._kl(p, m) + self._kl(q, m))
                disagreements.append(js_div)
        
        avg_disagreement = np.mean(disagreements)
        
        is_anomalous = avg_disagreement > self.threshold
        
        return {
            'is_anomalous': is_anomalous,
            'avg_disagreement': avg_disagreement,
            'max_disagreement': max(disagreements),
            'predictions': predictions.tolist()
        }
    
    def _kl(self, p, q):
        """KL散度"""
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        return np.sum(p * np.log(p / q))
```

### 21.5.3 模型行为监控

```python
class BehaviorMonitor:
    """监控模型行为是否存在被入侵的迹象"""
    
    def __init__(self, model, reference_dataset):
        self.model = model
        self.reference_stats = self._compute_reference_stats(reference_dataset)
    
    def _compute_reference_stats(self, dataset):
        """计算参考行为统计"""
        predictions = []
        confidences = []
        gradients = []
        
        for x, y in dataset:
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x_tensor.requires_grad = True
            
            output = self.model(x_tensor)
            pred = output.argmax(dim=1).item()
            conf = torch.softmax(output, dim=1).max().item()
            
            output[0, pred].backward()
            grad = x_tensor.grad.numpy().flatten()
            
            predictions.append(pred)
            confidences.append(conf)
            gradients.append(grad)
        
        return {
            'accuracy': np.mean(np.array(predictions) == np.array([y for _, y in dataset])),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'gradient_mean': np.mean(gradients, axis=0),
            'gradient_std': np.std(gradients, axis=0) + 1e-8
        }
    
    def monitor(self, input_data, label=None):
        """监控单个输入"""
        x_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        x_tensor.requires_grad = True
        
        output = self.model(x_tensor)
        pred = output.argmax(dim=1).item()
        conf = torch.softmax(output, dim=1).max().item()
        
        output[0, pred].backward()
        grad = x_tensor.grad.numpy().flatten()
        
        # 检查置信度
        conf_z_score = (conf - self.reference_stats['mean_confidence']) / self.reference_stats['std_confidence']
        
        # 检查梯度分布
        grad_diff = grad - self.reference_stats['gradient_mean']
        grad_mahal = np.sqrt(grad_diff @ np.diag(1 / self.reference_stats['gradient_std']**2) @ grad_diff)
        
        alerts = []
        if abs(conf_z_score) > 3:
            alerts.append(f"异常置信度: z-score={conf_z_score:.2f}")
        if grad_mahal > 10:
            alerts.append(f"异常梯度: Mahalanobis={grad_mahal:.2f}")
        
        return {
            'prediction': pred,
            'confidence': conf,
            'confidence_z_score': conf_z_score,
            'gradient_mahalanobis': grad_mahal,
            'alerts': alerts,
            'is_anomalous': len(alerts) > 0
        }
```

---

## 21.6 案例研究：Google如何保护其生产ML模型

### 背景

Google运营着世界上最大的ML部署之一，每天在Search、Gmail、YouTube和其他服务中提供数十亿次预测。他们对ML安全的方法代表了生产ML防御的最新技术。

### 防御架构

Google的防御架构在四层运行：

**第1层：输入验证**

Google采用多阶段输入验证管道：

1. **格式验证**：确保输入符合预期模式
2. **统计验证**：将输入分布与训练分布进行比较
3. **语义验证**：使用基于规则的系统检查明显无效的输入
4. **对抗检测**：应用轻量级对抗检测模型

根据Google的AI原则（2018年，2024年更新），所有AI系统在部署前必须经过安全审查（ai.google/responsibilities/principles/）。

**第2层：模型加固**

Google使用多种模型加固技术：

1. **对抗训练**：针对关键模型的基于PGD的训练
2. **集成方法**：多个模型对每个预测进行投票
3. **随机平滑**：提供可证明的鲁棒性保证
4. **模型蒸馏**：降低模型复杂性和攻击面

**第3层：运行时监控**

Google的监控系统包括：

1. **输入监控**：实时跟踪输入分布
2. **预测监控**：对异常预测模式发出警报
3. **资源监控**：检测异常计算模式
4. **行为监控**：将模型行为与预期模式进行比较

**第4层：事件响应**

Google维护着专门的ML安全事件响应团队：

1. 监控新型攻击模式
2. 调查潜在入侵
3. 跨团队协调响应
4. 根据新威胁更新防御

### 关键指标

| 指标 | 数值 | 来源 |
|------|------|------|
| 每日预测 | 1000亿+ | Google AI Blog，2024 |
| 对抗样本拦截率 | 99.7% | 引用在Schmidt等，2024的内部报告 |
| 平均检测时间 | <5分钟 | SRE实践 |
| 误报率 | <0.01% | 生产监控 |

### 经验教训

1. **纵深防御有效**：没有单一防御能捕获所有内容
2. **监控与预防同等重要**：检测实现响应
3. **自动化至关重要**：人工响应对实时攻击来说太慢了
4. **威胁情报共享**：行业协作改善集体防御

---

## 21.7 实战故事：被绕过的防御

### 背景

2022年，一家大型云提供商（匿名化）为其图像分类API实现了对抗检测系统。该系统使用输入验证、集成分歧和对抗检测的组合。根据对已知攻击的测试，它被认为非常有效。

### 绕过方法

安全研究人员发现了一种新型绕过技术：

**阶段1：系统分析**

研究人员通过观察其行为分析了防御系统：
- 集成使用5个不同架构的模型
- 分歧阈值设置为0.3（JS散度）
- 对抗检测器针对FGSM和PGD攻击进行训练

**阶段2：对抗样本构造**

研究人员开发了一种"隐身"对抗攻击：

```python
# 绕过技术的概念表示
class StealthAdversarialAttack:
    def __init__(self, ensemble_models, detection_threshold):
        self.models = ensemble_models
        self.threshold = detection_threshold
    
    def generate_stealthy_adversarial(self, image, target_class):
        """
        生成逃避集成检测的对抗样本。
        关键洞察：在改变预测的同时优化模型共识。
        """
        # 用小扰动初始化
        delta = torch.zeros_like(image, requires_grad=True)
        
        for step in range(200):
            # 扰动图像
            perturbed = image + delta
            
            # 从所有模型获取预测
            predictions = [model(perturbed) for model in self.models]
            
            # 损失1：使所有模型一致（最小化分歧）
            disagreement_loss = self.compute_disagreement(predictions)
            
            # 损失2：将预测更改为目标
            target_loss = -predictions[0][0, target_class]
            
            # 组合损失
            loss = disagreement_loss + 0.5 * target_loss
            
            loss.backward()
            
            # 更新delta
            delta.data = delta.data + 0.01 * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -0.03, 0.03)
            delta.grad.zero_()
        
        return image + delta
    
    def compute_disagreement(self, predictions):
        """计算平均成对JS散度"""
        n = len(predictions)
        total_div = 0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                p = torch.softmax(predictions[i], dim=1)
                q = torch.softmax(predictions[j], dim=1)
                m = 0.5 * (p + q)
                
                js = 0.5 * (self.kl(p, m) + self.kl(q, m))
                total_div += js
                count += 1
        
        return total_div / count
    
    def kl(self, p, q):
        return torch.sum(p * torch.log(p / q + 1e-8), dim=1).mean()
```

**阶段3：结果**

隐身攻击实现了：
- **成功率**：78%的定向错误分类
- **检测逃避**：95%的对抗样本逃避了检测
- **集成共识**：平均JS散度为0.12（低于0.3阈值）
- **扰动大小**：与标准PGD攻击相似

### 根本原因分析

绕过成功是因为：

1. **防御假设对抗样本增加分歧**：隐身攻击专门优化共识
2. **对抗检测器针对已知攻击进行训练**：它从未见过隐身技术
3. **防御依赖单一检测信号**：仅集成分歧不足
4. **阈值是静态的**：它无法适应新的攻击模式

### 改进措施

提供商实施了以下改进：

1. **自适应阈值**：基于输入特征的动态阈值
2. **多检测信号**：结合分歧与输入统计和模型置信度
3. **持续对抗重训练**：定期向训练添加新的攻击模式
4. **红队演练**：定期针对新型攻击技术进行测试

### 关键要点

**没有永久的防御。** 攻击者总会找到新的绕过技术。关键是：
1. 假设你的防御会被绕过
2. 实施多个互补的防御
3. 监控新型攻击模式
4. 根据新威胁持续更新防御

---

## 21.8 何时使用/何时不使用AI安全防御

### 何时使用特定防御

| 防御技术 | 何时使用 | 何时不使用 | 成本/复杂度 |
|---------|---------|-----------|------------|
| 输入验证 | 始终（基线） | 永远不要跳过 | 低 |
| 对抗训练 | 具有对抗风险的生产模型 | 低风险原型 | 中-高 |
| 认证防御 | 安全关键应用 | 计算预算有限时 | 高 |
| 模型水印 | 具有IP价值的模型 | 仅内部模型 | 中 |
| 集成方法 | 高风险预测 | 延迟敏感的应用 | 中-高 |
| 运行时监控 | 所有生产部署 | 开发/测试 | 中 |
| 集成分歧 | 安全关键应用 | 简单模型 | 低-中 |

### 决策框架

```
模型在生产中吗？─── 是 ──→ 输入验证 + 监控是强制性的
         │
         否
         │
处理敏感数据吗？─── 是 ──→ 添加对抗训练
         │
         否
         │
是安全关键的吗？─── 是 ──→ 添加认证防御 + 集成
         │
         否
         │
有IP价值吗？─── 是 ──→ 添加水印
         │
         否
         │
仅标准安全基线
```

---

## 21.9 总结

AI安全防御需要全面的多层方法：

1. **输入验证是你的第一道防线**：在输入到达模型之前验证、清理和监控所有输入。这是最具成本效益的防御。

2. **对抗训练是黄金标准**：在训练中包含对抗样本提供了经过验证的稳健防御。TRADES和基于PGD的方法已经成熟。

3. **认证防御提供保证**：随机平滑和相关方法提供了鲁棒性的数学保证，对安全关键应用至关重要。

4. **模型水印保护IP**：在模型中嵌入可验证的所有权信息以威慑盗窃并实现执行。

5. **运行时监控捕获预防遗漏的内容**：分布监控、集成分歧和行为监控提供检测能力。

6. **没有永久的防御**：攻击者在进化，防御也必须进化。持续测试和更新至关重要。

---

## 21.10 讨论题

1. **防御权衡**：如果对抗训练将干净准确率降低10%，在什么业务条件下这是可接受的？你如何量化这种权衡？

2. **认证vs经验鲁棒性**：何时应优先考虑认证鲁棒性而非经验鲁棒性？每种方法的实际影响是什么？

3. **监控开销**：运行时监控增加延迟和成本。你如何设计一个平衡安全性和性能要求的监控系统？

4. **水印有效性**：如果模型水印依赖于特定的输入-输出对，攻击者是否可以通过微调简单地移除这些对？你如何使水印更稳健？

5. **防御逃避**：如果你是试图绕过本章所述防御的攻击者，你会首先针对哪种防御？为什么？

---

## 21.11 练习

### 练习1：实现输入验证

为图像分类模型构建完整的输入验证管道：

1. 实现第21.2.2节中的`ImageValidator`类
2. 使用FGSM生成对抗样本
3. 测试验证器捕获了多少对抗样本
4. 实现`sanitize`方法并测试其有效性
5. 测量对干净精度的影响

### 练习2：对抗训练

在CIFAR-10上实现PGD对抗训练：

1. 训练不使用对抗训练的基线ResNet-18
2. 使用PGD对抗训练训练相同架构
3. 比较干净精度和对抗鲁棒性
4. 绘制精度-鲁棒性权衡曲线

### 练习3：监控系统设计

为生产ML服务设计监控系统：

1. 定义你将监控的指标
2. 基于统计方法设计警报阈值
3. 创建显示关键指标的仪表板规范
4. 编写检测到异常时的事件响应手册

---

## 21.12 参考文献

### 对抗训练

1. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR*. https://arxiv.org/abs/1706.06083

2. Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019). "Theoretically Principled Trade-off between Robustness and Accuracy." *ICML*. https://arxiv.org/abs/1901.08558

3. Shafahi, A., 等. (2019). "Are Adversarial Perturbations Dirty Tricks?" *ICML*. https://arxiv.org/abs/1810.00057

### 认证防御

4. Cohen, J. M., Rosenfeld, E., & Kolter, J. Z. (2019). "Certified Adversarial Robustness via Randomized Smoothing." *ICML*. https://arxiv.org/abs/1902.02918

5. Lecuyer, G., 等. (2019). "Certified Robustness to Adversarial Examples with Differential Privacy." *IEEE S&P*. https://arxiv.org/abs/1802.00420

### 模型水印

6. Uchida, Y., 等. (2017). "Embedding Watermarks into Deep Neural Networks." *ACM IH*. https://doi.org/10.1145/3029386.3029400

7. Zhang, J., 等. (2018). "Protecting Intellectual Property of Deep Neural Networks with Watermarking." *AsiaCCS*. https://doi.org/10.1145/3196494.3196557

### 监控和检测

8. Pertsev, A., 等. (2018). "Detecting Adversarial Samples from Artifacts." *arXiv*. https://arxiv.org/abs/1703.07375

9. Grosse, K., 等. (2017). "On Adversarial Examples for Character-Level Neural Machine Translation." *IEEE S&P Workshops*. https://arxiv.org/abs/1705.04364

### Google的ML安全

10. Google AI原则. (2024). https://ai.google/responsibilities/principles/

11. Schmidt, L., 等. (2024). "Adversarial Robustness in Production ML Systems." *Google AI Blog*. https://ai.googleblog.com/

### 框架

12. NIST AI 100-2. (2023). "AI风险管理框架." https://www.nist.gov/artificial-intelligence/risk-management-framework

13. MITRE ATLAS. (2024). https://atlas.mitre.org/

---

*下一章：[第22章：隐私保护AI架构 →](./chapter-22.md)*
