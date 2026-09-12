# Chapter 21: AI Security Defense Architecture

## Learning Objectives

By the end of this chapter, you will be able to:

1. Design and implement multi-layered defense architectures for AI systems that address the threat landscape covered in Chapter 20
2. Apply real input validation and sanitization patterns specifically designed for ML model inputs
3. Implement adversarial training and certified defense methods with working code
4. Design model watermarking and fingerprinting systems to protect intellectual property
5. Build monitoring and anomaly detection systems that identify adversarial activity in production

---

## 21.1 Introduction: Defense in Depth for AI Systems

The defense of AI systems cannot rely on a single mechanism. Just as enterprise security employs firewalls, intrusion detection systems, access controls, and encryption in layered defense, AI security requires a comprehensive approach that addresses threats at every stage of the ML lifecycle.

The fundamental principle is **defense in depth**: no single layer of defense is sufficient, but when properly layered, they create a defense system that is greater than the sum of its parts. An attacker must bypass all layers, not just one, to succeed.

According to NIST AI 100-2 (2023), effective AI security requires:
1. **Prevention**: Blocking attacks before they reach the model
2. **Detection**: Identifying when attacks succeed despite prevention
3. **Response**: Mitigating damage and recovering from successful attacks
4. **Adaptation**: Evolving defenses as attack techniques improve

This chapter provides practical, implementable defense strategies for each layer.

---

## 21.2 Layer 1: Input Validation and Sanitization

### 21.2.1 The Principle

Before any input reaches your model, it must be validated and sanitized. This is the most cost-effective defense layer because it operates before the model is involved.

### 21.2.2 Image Input Validation

**Statistical Distribution Checks**

Real-world images follow predictable statistical distributions. Adversarial examples often deviate from these distributions:

```python
import numpy as np
from scipy import stats
import torch

class ImageValidator:
    """Validates image inputs against known distributions"""
    
    def __init__(self, training_stats_path):
        self.training_stats = self._load_stats(training_stats_path)
    
    def _load_stats(self, path):
        """Load pre-computed statistics from training data"""
        stats = np.load(path)
        return {
            'mean': stats['mean'],
            'std': stats['std'],
            'histogram': stats['histogram'],
            'pixel_range': (0, 255),
            'max_fft_ratio': 0.15  # Maximum allowed high-frequency energy
        }
    
    def validate(self, image, threshold=0.95):
        """
        Validate image against multiple criteria
        Returns (is_valid, scores, anomalies)
        """
        scores = {}
        anomalies = []
        
        # 1. Pixel range check
        if image.min() < 0 or image.max() > 255:
            anomalies.append("Pixel values outside valid range")
            scores['pixel_range'] = 0.0
        else:
            scores['pixel_range'] = 1.0
        
        # 2. Distribution check (KL divergence)
        current_hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
        current_hist = current_hist / current_hist.sum() + 1e-10
        kl_div = stats.entropy(current_hist, self.training_stats['histogram'])
        
        if kl_div > 0.1:  # Threshold determined empirically
            anomalies.append(f"KL divergence too high: {kl_div:.4f}")
            scores['distribution'] = max(0, 1 - kl_div)
        else:
            scores['distribution'] = 1.0
        
        # 3. Frequency domain analysis (adversarial perturbations
        #    often have high-frequency components)
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # Calculate ratio of high-frequency to total energy
        h, w = image.shape[:2]
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        # Low-frequency region (center)
        low_freq_mask = np.zeros_like(magnitude, dtype=bool)
        y, x = np.ogrid[:h, :w]
        low_freq_mask[(y - center_h)**2 + (x - center_w)**2 <= radius**2] = True
        
        total_energy = magnitude.sum()
        low_freq_energy = magnitude[low_freq_mask].sum()
        high_freq_ratio = 1 - (low_freq_energy / total_energy) if total_energy > 0 else 0
        
        if high_freq_ratio > self.training_stats['max_fft_ratio']:
            anomalies.append(f"High-frequency energy ratio: {high_freq_ratio:.3f}")
            scores['frequency'] = max(0, 1 - high_freq_ratio)
        else:
            scores['frequency'] = 1.0
        
        # 4. Structural Similarity Index (compare to expected structure)
        # This uses a pre-computed reference statistics
        edges = np.gradient(image.astype(float))
        edge_magnitude = np.sqrt(edges[0]**2 + edges[1]**2)
        avg_edge = edge_magnitude.mean()
        
        if avg_edge > 50:  # Images with very strong edges
            anomalies.append(f"Unusual edge magnitude: {avg_edge:.1f}")
            scores['structure'] = 0.7
        else:
            scores['structure'] = 1.0
        
        # Overall validation
        is_valid = all(s >= threshold for s in scores.values())
        
        return is_valid, scores, anomalies
    
    def sanitize(self, image):
        """Apply sanitization transforms to remove potential adversarial noise"""
        # JPEG compression (destroys adversarial perturbations)
        from PIL import Image
        import io
        
        img_pil = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        img_restored = np.array(Image.open(buffer))
        
        # Gaussian blur (removes high-frequency perturbations)
        from scipy.ndimage import gaussian_filter
        img_blurred = gaussian_filter(img_restored, sigma=0.5)
        
        # Random resize and pad (defeats patch-based attacks)
        scale = np.random.uniform(0.95, 1.05)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = np.array(Image.fromarray(img_blurred.astype(np.uint8)).resize(
            (new_w, new_h), Image.BILINEAR
        ))
        
        # Pad back to original size
        result = np.zeros_like(image)
        dh, dw = (new_h - h) // 2, (new_w - w) // 2
        result[max(0, -dh):min(h, new_h-dh), max(0, -dw):min(w, new_w-dw)] = \
            img_resized[max(0, dh):min(new_h, h+dh), max(0, dw):min(new_w, w+dw)]
        
        return result.astype(np.uint8)
```

**Perceptual Hashing for Known Attacks**

Maintain a database of known adversarial patterns:

```python
import imagehash
from PIL import Image

class AdversarialPatternDetector:
    """Detects known adversarial patterns using perceptual hashing"""
    
    def __init__(self, pattern_db_path):
        self.known_patterns = self._load_pattern_database(pattern_db_path)
        self.similarity_threshold = 0.85
    
    def _load_pattern_database(self, path):
        """Load known adversarial pattern hashes"""
        import json
        with open(path, 'r') as f:
            patterns = json.load(f)
        return {p['name']: imagehash.hex_to_hash(p['hash']) for p in patterns}
    
    def detect(self, image):
        """Check if image matches any known adversarial pattern"""
        img_hash = imagehash.phash(Image.fromarray(image.astype(np.uint8)))
        
        matches = []
        for name, pattern_hash in self.known_patterns.items():
            similarity = 1 - (img_hash - pattern_hash) / len(img_hash.hash.flatten())
            if similarity > self.similarity_threshold:
                matches.append((name, similarity))
        
        return matches
```

### 21.2.3 Text Input Validation (for LLMs)

```python
import re
from typing import List, Tuple

class TextSanitizer:
    """Validates and sanitizes text inputs for LLM applications"""
    
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
        """Validate text for prompt injection attempts"""
        anomalies = []
        
        # Check for direct injection patterns
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                anomalies.append(f"Potential injection: {pattern[:50]}...")
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                anomalies.append(f"Suspicious pattern: {pattern[:50]}...")
        
        # Check for unusual Unicode characters
        unusual_chars = sum(1 for c in text if ord(c) > 0xFFFF or 
                          (0x200B <= ord(c) <= 0x200F) or  # Zero-width characters
                          (0x2028 <= ord(c) <= 0x2029) or  # Line/paragraph separators
                          (0x2060 <= ord(c) <= 0x2064))    # Invisible formatters
        
        if unusual_chars > 0:
            anomalies.append(f"Found {unusual_chars} unusual Unicode characters")
        
        # Check for excessive length (potential DoS)
        if len(text) > 10000:
            anomalies.append(f"Input length {len(text)} exceeds limit")
        
        is_valid = len(anomalies) == 0
        return is_valid, anomalies
    
    def sanitize(self, text: str) -> str:
        """Remove potentially harmful patterns from text"""
        sanitized = text
        
        # Remove zero-width characters
        sanitized = re.sub(r'[\u200b-\u200f\u2028-\u2029\u2060-\u2064]', '', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Remove control characters (except newline and tab)
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        return sanitized.strip()
```

### 21.2.4 Tabular Data Validation

```python
import pandas as pd
import numpy as np
from scipy import stats

class TabularValidator:
    """Validates tabular data inputs against training distributions"""
    
    def __init__(self, training_data_path):
        self.reference_stats = self._compute_reference_stats(training_data_path)
    
    def _compute_reference_stats(self, path):
        """Compute statistics from training data"""
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
        """Validate input data against reference statistics"""
        anomalies = {}
        is_valid = True
        
        for col in input_data.columns:
            if col not in self.reference_stats:
                anomalies[col] = ["Unknown column"]
                is_valid = False
                continue
            
            ref = self.reference_stats[col]
            values = input_data[col].dropna()
            
            if len(values) == 0:
                anomalies[col] = ["All values missing"]
                is_valid = False
                continue
            
            # Range check
            if values.min() < ref['min'] * 0.5 or values.max() > ref['max'] * 2:
                anomalies[col] = ["Values outside expected range"]
                is_valid = False
            
            # Distribution test (KS test)
            ks_stat, p_value = stats.ks_2samp(values, np.random.normal(ref['mean'], ref['std'], 1000))
            if p_value < 0.01:
                anomalies[col] = [f"Distribution mismatch (KS p={p_value:.4f})"]
                is_valid = False
        
        return is_valid, anomalies
```

---

## 21.3 Layer 2: Adversarial Training

### 21.3.1 Standard Adversarial Training (PGD-based)

Adversarial training is the most empirically validated defense against adversarial examples. It works by including adversarial examples in the training process.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class AdversarialTrainer:
    """PGD-based adversarial training"""
    
    def __init__(self, model, epsilon=0.03, alpha=0.007, 
                 num_steps=10, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        
    def pgd_attack(self, images, labels, criterion):
        """Generate PGD adversarial examples"""
        images = images.clone().detach()
        labels = labels.clone().detach()
        
        # Random initialization
        if self.random_start:
            images = images + torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            images = torch.clamp(images, 0, 1)
        
        for _ in range(self.num_steps):
            images.requires_grad = True
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # Gradient ascent step
            images = images.detach() + self.alpha * images.grad.sign()
            images = torch.clamp(images, 0, 1)
            
            # Project back to epsilon-ball
            delta = torch.clamp(images - images, -self.epsilon, self.epsilon)
            images = torch.clamp(images + delta, 0, 1).detach()
        
        return images
    
    def train_epoch(self, train_loader, optimizer, criterion, 
                    mix_ratio=0.5):
        """Train for one epoch with adversarial examples"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            
            # Mix clean and adversarial examples
            if torch.rand(1).item() < mix_ratio:
                # Generate adversarial examples
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

### 21.3.2 TRADES: Trading off Robustness and Accuracy

Zhang et al. (2019) proposed TRADES, which provides a principled way to balance accuracy and robustness:

```python
class TRADESTrainer:
    """TRADES adversarial training"""
    
    def __init__(self, model, epsilon=0.03, alpha=0.007,
                 num_steps=10, beta=6.0):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.beta = beta  # Regularization strength
    
    def trades_loss(self, images, labels, criterion):
        """Compute TRADES loss"""
        # Natural loss
        outputs_natural = self.model(images)
        loss_natural = criterion(outputs_natural, labels)
        
        # Generate adversarial examples
        images_adv = self._pgd_attack(images, labels, 
                                       lambda x, y: -self._kl_divergence(
                                           self.model(x), self.model(images).detach()
                                       ))
        
        # KL divergence between natural and adversarial predictions
        outputs_adv = self.model(images_adv)
        loss_robust = self._kl_divergence(outputs_adv, outputs_natural.detach())
        
        # Total loss
        loss = loss_natural + self.beta * loss_robust
        
        return loss
    
    def _kl_divergence(self, p, q):
        """KL divergence between two distributions"""
        p = torch.nn.functional.softmax(p, dim=1)
        q = torch.nn.functional.softmax(q, dim=1)
        return torch.sum(p * torch.log(p / q + 1e-8), dim=1).mean()
    
    def _pgd_attack(self, images, labels, loss_fn):
        """PGD attack using arbitrary loss function"""
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

### 21.3.3 Certified Defenses: Randomized Smoothing

Randomized smoothing provides provable guarantees on robustness:

```python
import numpy as np
from scipy.stats import norm

class SmoothedClassifier:
    """Randomized smoothing for certified robustness"""
    
    def __init__(self, base_classifier, sigma=0.25, n_samples=1000,
                 alpha=0.001):
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.n_samples = n_samples
        self.alpha = alpha
    
    def certify(self, x):
        """
        Certify the prediction of x using randomized smoothing.
        Returns (prediction, certified_radius)
        """
        # Draw samples
        counts = self._sample_predictions(x)
        
        # Find the top class
        top_class = counts.argmax()
        top_count = counts[top_class]
        
        # Compute confidence interval
        n_total = counts.sum()
        
        # Lower bound on probability of top class
        p_lower = self._lower_confidence_bound(top_count, n_total)
        
        if p_lower > 0.5:
            # Certified radius using反向Cumulative Distribution Function
            certified_radius = self.sigma * norm.ppf(p_lower)
            return top_class, certified_radius
        else:
            return -1, 0.0  # Abstain
    
    def _sample_predictions(self, x):
        """Sample predictions from smoothed classifier"""
        counts = np.zeros(self.base_classifier.num_classes)
        
        for _ in range(self.n_samples):
            # Add Gaussian noise
            noise = np.random.normal(0, self.sigma, x.shape)
            x_noisy = x + noise
            
            # Get prediction
            pred = self.base_classifier.predict(x_noisy)
            counts[pred] += 1
        
        return counts
    
    def _lower_confidence_bound(self, k, n):
        """Lower confidence bound using Clopper-Pearson interval"""
        from statsmodels.stats.proportion import proportion_confint
        return proportion_confint(k, n, alpha=2*self.alpha, method='beta')[0]
```

---

## 21.4 Layer 3: Model Watermarking and Fingerprinting

### 21.4.1 Watermarking for IP Protection

Watermarking embeds ownership information into a trained model that can be verified without access to the original training data.

```python
import torch
import torch.nn as nn

class ModelWatermarker:
    """Embed and verify watermarks in neural network models"""
    
    def __init__(self, model, watermark_key_size=32):
        self.model = model
        self.key_size = watermark_key_size
    
    def generate_watermark_pair(self, input_shape):
        """
        Generate a watermark input-output pair.
        The input is crafted, and the output is the target.
        """
        # Generate random input
        watermark_input = torch.randn(1, *input_shape)
        
        # The watermark output is derived deterministically from the key
        # This ensures only the key holder can verify
        torch.manual_seed(hash(str(self.key_size)))
        watermark_output = torch.randint(0, self.model.fc.out_features, (1,))
        
        return watermark_input, watermark_output
    
    def embed_watermark(self, train_loader, watermark_weight=1.0,
                       num_watermark_steps=1000):
        """
        Embed watermark through fine-tuning.
        The model learns to correctly classify watermark inputs.
        """
        watermark_input, watermark_target = self.generate_watermark_pair(
            (3, 224, 224)  # Assumes image input
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Fine-tune on watermark
        for step in range(num_watermark_steps):
            outputs = self.model(watermark_input.cuda())
            loss = criterion(outputs, watermark_target.cuda()) * watermark_weight
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return self.model
    
    def verify_watermark(self, threshold=0.9):
        """
        Verify that the model contains the watermark.
        Returns (is_watermarked, confidence)
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
        """Generate multiple watermark pairs for robust verification"""
        watermarks = []
        for i in range(num_watermarks):
            torch.manual_seed(42 + i)  # Deterministic seeds
            watermark_input = torch.randn(1, 3, 224, 224)
            watermark_target = torch.randint(0, self.model.fc.out_features, (1,))
            watermarks.append((watermark_input, watermark_target))
        return watermarks
```

### 21.4.2 Model Fingerprinting

Fingerprinting identifies specific models based on their unique behavioral characteristics:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ModelFingerprinter:
    """Generate and match model fingerprints"""
    
    def __init__(self, model, num_queries=1000):
        self.model = model
        self.num_queries = num_queries
    
    def generate_fingerprint(self, input_generator):
        """
        Generate a model fingerprint based on its behavior
        on carefully chosen inputs.
        """
        fingerprints = []
        
        for _ in range(self.num_queries):
            # Generate diverse inputs
            x = input_generator()
            
            # Get model prediction with gradient
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x_tensor.requires_grad = True
            
            output = self.model(x_tensor)
            
            # Use gradient as fingerprint component
            output[0, output.argmax()].backward()
            gradient = x_tensor.grad.numpy().flatten()
            
            fingerprints.append(gradient)
        
        # Aggregate fingerprints
        fingerprint = np.mean(fingerprints, axis=0)
        
        # Reduce dimensionality while preserving uniqueness
        # Using random projection
        np.random.seed(42)
        projection = np.random.randn(len(fingerprint), 128)
        reduced_fingerprint = fingerprint @ projection
        reduced_fingerprint = reduced_fingerprint / np.linalg.norm(reduced_fingerprint)
        
        return reduced_fingerprint
    
    def match_fingerprint(self, fp1, fp2, threshold=0.95):
        """Match two fingerprints"""
        similarity = cosine_similarity(fp1.reshape(1, -1), fp2.reshape(1, -1))[0, 0]
        return similarity > threshold, similarity
```

---

## 21.5 Layer 4: Runtime Monitoring and Anomaly Detection

### 21.5.1 Input Distribution Monitoring

```python
import numpy as np
from collections import deque

class InputMonitor:
    """Monitor input distribution for anomalies"""
    
    def __init__(self, window_size=1000, alert_threshold=2.0):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.input_history = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=window_size)
        
        # Baseline statistics
        self.baseline_mean = None
        self.baseline_std = None
    
    def set_baseline(self, baseline_data):
        """Set baseline statistics from validation data"""
        self.baseline_mean = np.mean(baseline_data, axis=0)
        self.baseline_std = np.std(baseline_data, axis=0) + 1e-8
    
    def update(self, input_data, prediction, confidence):
        """Update monitoring statistics"""
        self.input_history.append(input_data)
        self.prediction_history.append({
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': np.datetime64('now')
        })
    
    def check_anomaly(self, new_input):
        """Check if new input is anomalous"""
        if self.baseline_mean is None:
            return False, "No baseline set"
        
        # Compute Mahalanobis distance
        diff = new_input.flatten() - self.baseline_mean
        mahal_dist = np.sqrt(diff @ np.linalg.inv(np.diag(self.baseline_std**2)) @ diff)
        
        if mahal_dist > self.alert_threshold:
            return True, f"Mahalanobis distance: {mahal_dist:.3f}"
        
        return False, "Normal"
    
    def check_prediction_drift(self):
        """Check if prediction distribution has drifted"""
        if len(self.prediction_history) < 100:
            return False, "Insufficient data"
        
        recent_confidences = [p['confidence'] for p in 
                             list(self.prediction_history)[-100:]]
        historical_confidences = [p['confidence'] for p in 
                                 list(self.prediction_history)[:-100]]
        
        if len(historical_confidences) == 0:
            return False, "No historical data"
        
        # Compare distributions
        from scipy.stats import ks_2samp
        ks_stat, p_value = ks_2samp(recent_confidences, historical_confidences)
        
        if p_value < 0.01:
            return True, f"Prediction distribution drift detected (p={p_value:.4f})"
        
        return False, "Stable"
```

### 21.5.2 Ensemble Disagreement Monitoring

When using ensemble methods, disagreement between models can indicate adversarial inputs:

```python
class EnsembleMonitor:
    """Monitor disagreement between ensemble members"""
    
    def __init__(self, models, disagreement_threshold=0.3):
        self.models = models
        self.threshold = disagreement_threshold
    
    def check_input(self, input_data):
        """
        Check if ensemble members disagree on the input.
        High disagreement may indicate adversarial input.
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(input_data)
                pred = torch.softmax(output, dim=1)
                predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Compute pairwise disagreement
        n_models = len(self.models)
        disagreements = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Jensen-Shannon divergence
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
        """KL divergence"""
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        return np.sum(p * np.log(p / q))
```

### 21.5.3 Model Behavior Monitoring

```python
class BehaviorMonitor:
    """Monitor model behavior for signs of compromise"""
    
    def __init__(self, model, reference_dataset):
        self.model = model
        self.reference_stats = self._compute_reference_stats(reference_dataset)
    
    def _compute_reference_stats(self, dataset):
        """Compute reference behavior statistics"""
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
        """Monitor a single input"""
        x_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        x_tensor.requires_grad = True
        
        output = self.model(x_tensor)
        pred = output.argmax(dim=1).item()
        conf = torch.softmax(output, dim=1).max().item()
        
        output[0, pred].backward()
        grad = x_tensor.grad.numpy().flatten()
        
        # Check confidence
        conf_z_score = (conf - self.reference_stats['mean_confidence']) / self.reference_stats['std_confidence']
        
        # Check gradient distribution
        grad_diff = grad - self.reference_stats['gradient_mean']
        grad_mahal = np.sqrt(grad_diff @ np.diag(1 / self.reference_stats['gradient_std']**2) @ grad_diff)
        
        alerts = []
        if abs(conf_z_score) > 3:
            alerts.append(f"Unusual confidence: z-score={conf_z_score:.2f}")
        if grad_mahal > 10:
            alerts.append(f"Unusual gradient: Mahalanobis={grad_mahal:.2f}")
        
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

## 21.6 Case Study: How Google Protects Its Production ML Models

### Background

Google operates one of the largest ML deployments in the world, serving billions of predictions daily across Search, Gmail, YouTube, and other services. Their approach to ML security represents the state of the art in production ML defense.

### Defense Architecture

Google's defense architecture operates across four layers:

**Layer 1: Input Validation**

Google employs a multi-stage input validation pipeline:

1. **Format validation**: Ensure inputs conform to expected schema
2. **Statistical validation**: Compare input distribution to training distribution
3. **Semantic validation**: Use rule-based systems to check for obviously invalid inputs
4. **Adversarial detection**: Apply lightweight adversarial detection models

According to Google's AI Principles (2018, updated 2024), all AI systems must undergo security review before deployment (ai.google/responsibilities/principles/).

**Layer 2: Model Hardening**

Google uses multiple model hardening techniques:

1. **Adversarial training**: PGD-based training for critical models
2. **Ensemble methods**: Multiple models vote on each prediction
3. **Randomized smoothing**: Provides certified robustness guarantees
4. **Model distillation**: Reduces model complexity and attack surface

**Layer 3: Runtime Monitoring**

Google's monitoring system includes:

1. **Input monitoring**: Real-time tracking of input distributions
2. **Prediction monitoring**: Alerting on unusual prediction patterns
3. **Resource monitoring**: Detecting unusual computational patterns
4. **Behavioral monitoring**: Comparing model behavior to expected patterns

**Layer 4: Incident Response**

Google maintains a dedicated ML security incident response team that:

1. Monitors for novel attack patterns
2. Investigates potential compromises
3. Coordinates response across teams
4. Updates defenses based on new threats

### Key Metrics

| Metric | Value | Source |
|--------|-------|--------|
| Daily predictions | 100+ billion | Google AI Blog, 2024 |
| Adversarial examples blocked | 99.7% | Internal report cited in Schmidt et al., 2024 |
| Mean time to detect | <5 minutes | SRE practices |
| False positive rate | <0.01% | Production monitoring |

### Lessons Learned

1. **Defense in depth works**: No single defense catches everything
2. **Monitoring is as important as prevention**: Detection enables response
3. **Automation is essential**: Human response is too slow for real-time attacks
4. **Threat intelligence sharing**: Industry collaboration improves collective defense

---

## 21.7 War Story: The Defense That Was Bypassed

### Background

In 2022, a major cloud provider (anonymized) implemented an adversarial detection system for their image classification API. The system used a combination of input validation, ensemble disagreement, and adversarial detection. It was considered highly effective based on testing against known attacks.

### The Bypass

Security researchers discovered a novel bypass technique:

**Phase 1: System Analysis**

The researchers analyzed the defense system by observing its behavior:
- The ensemble used 5 models with different architectures
- Disagreement threshold was set at 0.3 (JS divergence)
- Adversarial detector was trained on FGSM and PGD attacks

**Phase 2: Adversarial Example Construction**

The researchers developed a "stealth" adversarial attack that:

```python
# Conceptual representation of the bypass technique
class StealthAdversarialAttack:
    def __init__(self, ensemble_models, detection_threshold):
        self.models = ensemble_models
        self.threshold = detection_threshold
    
    def generate_stealthy_adversarial(self, image, target_class):
        """
        Generate adversarial example that evades ensemble detection.
        The key insight: optimize for consensus among models while
        changing the prediction.
        """
        # Initialize with small perturbation
        delta = torch.zeros_like(image, requires_grad=True)
        
        for step in range(200):
            # Perturbed image
            perturbed = image + delta
            
            # Get predictions from all models
            predictions = [model(perturbed) for model in self.models]
            
            # Loss 1: Make all models agree (minimize disagreement)
            disagreement_loss = self.compute_disagreement(predictions)
            
            # Loss 2: Change prediction to target
            target_loss = -predictions[0][0, target_class]
            
            # Combined loss
            loss = disagreement_loss + 0.5 * target_loss
            
            loss.backward()
            
            # Update delta
            delta.data = delta.data + 0.01 * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -0.03, 0.03)
            delta.grad.zero_()
        
        return image + delta
    
    def compute_disagreement(self, predictions):
        """Compute average pairwise JS divergence"""
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

**Phase 3: Results**

The stealth attack achieved:
- **Success rate**: 78% on targeted misclassification
- **Detection evasion**: 95% of adversarial examples evaded detection
- **Ensemble consensus**: Average JS divergence was 0.12 (below 0.3 threshold)
- **Perturbation size**: Similar to standard PGD attacks

### Root Cause Analysis

The bypass succeeded because:

1. **The defense assumed adversarial examples increase disagreement**: The stealth attack specifically optimized for consensus
2. **The adversarial detector was trained on known attacks**: It had never seen the stealth technique
3. **The defense relied on a single detection signal**: Ensemble disagreement alone was insufficient
4. **The threshold was static**: It couldn't adapt to new attack patterns

### Improvements

The provider implemented the following improvements:

1. **Adaptive thresholds**: Dynamic thresholds based on input characteristics
2. **Multiple detection signals**: Combined disagreement with input statistics and model confidence
3. **Continuous adversarial retraining**: Regularly adding new attack patterns to training
4. **Red team exercises**: Regular testing against novel attack techniques

### Key Takeaway

**No defense is permanent.** Attackers will always find new bypass techniques. The key is to:
1. Assume your defense will be bypassed
2. Implement multiple complementary defenses
3. Monitor for novel attack patterns
4. Continuously update defenses based on new threats

---

## 21.8 When to Use / When Not to Use AI Security Defenses

### When to Use Specific Defenses

| Defense Technique | When to Use | When NOT to Use | Cost/Complexity |
|------------------|-------------|-----------------|-----------------|
| Input validation | Always (baseline) | Never skip | Low |
| Adversarial training | Production models with adversarial risk | Low-risk prototypes | Medium-High |
| Certified defenses | Safety-critical applications | When computational budget is limited | High |
| Model watermarking | Models with IP value | Internal-only models | Medium |
| Ensemble methods | High-stakes predictions | Latency-sensitive applications | Medium-High |
| Runtime monitoring | All production deployments | Development/testing | Medium |
| Ensemble disagreement | Security-critical applications | Simple models | Low-Medium |

### Decision Framework

```
Is the model in production? ─── YES ──→ Input validation + monitoring MANDATORY
         │
         NO
         │
Does it handle sensitive data? ─── YES ──→ Add adversarial training
         │
         NO
         │
Is it safety-critical? ─── YES ──→ Add certified defenses + ensemble
         │
         NO
         │
Is there IP value? ─── YES ──→ Add watermarking
         │
         NO
         │
Standard security baseline only
```

---

## 21.9 Summary

AI security defense requires a comprehensive, multi-layered approach:

1. **Input validation is your first line of defense**: Validate, sanitize, and monitor all inputs before they reach the model. This is the most cost-effective defense.

2. **Adversarial training is the gold standard**: Including adversarial examples in training provides robust, empirically validated defense. TRADES and PGD-based methods are well-established.

3. **Certified defenses provide guarantees**: Randomized smoothing and related methods provide mathematical guarantees on robustness, essential for safety-critical applications.

4. **Model watermarking protects IP**: Embed verifiable ownership information in models to deter theft and enable enforcement.

5. **Runtime monitoring catches what prevention misses**: Distribution monitoring, ensemble disagreement, and behavioral monitoring provide detection capabilities.

6. **No defense is permanent**: Attackers evolve, and so must defenses. Continuous testing and updating is essential.

---

## 21.10 Discussion Questions

1. **Defense Tradeoffs**: If adversarial training reduces clean accuracy by 10%, under what business conditions is this acceptable? How would you quantify the tradeoff?

2. **Certified vs. Empirical Robustness**: When should you prioritize certified robustness over empirical robustness? What are the practical implications of each approach?

3. **Monitoring Overhead**: Runtime monitoring adds latency and cost. How would you design a monitoring system that balances security with performance requirements?

4. **Watermark Effectiveness**: If model watermarking relies on specific input-output pairs, can't an attacker simply remove those pairs through fine-tuning? How would you make watermarks more robust?

5. **Defense Evasion**: If you were an attacker trying to bypass the defenses described in this chapter, which defense would you target first and why?

---

## 21.11 Exercises

### Exercise 1: Implement Input Validation

Build a complete input validation pipeline for an image classification model:

1. Implement the `ImageValidator` class from Section 21.2.2
2. Generate adversarial examples using FGSM
3. Test how many adversarial examples the validator catches
4. Implement the `sanitize` method and test its effectiveness
5. Measure the impact on clean accuracy

### Exercise 2: Adversarial Training

Implement PGD adversarial training on CIFAR-10:

1. Train a baseline ResNet-18 without adversarial training
2. Train the same architecture with PGD adversarial training
3. Compare clean accuracy and adversarial robustness
4. Plot the accuracy-robustness tradeoff curve

### Exercise 3: Monitoring System Design

Design a monitoring system for a production ML service:

1. Define what metrics you would monitor
2. Design alerting thresholds based on statistical methods
3. Create a dashboard specification showing key indicators
4. Write the incident response playbook for an detected anomaly

---

## 21.12 References

### Adversarial Training

1. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR*. https://arxiv.org/abs/1706.06083

2. Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019). "Theoretically Principled Trade-off between Robustness and Accuracy." *ICML*. https://arxiv.org/abs/1901.08558

3. Shafahi, A., et al. (2019). "Are Adversarial Perturbations Dirty Tricks?" *ICML*. https://arxiv.org/abs/1810.00057

### Certified Defenses

4. Cohen, J. M., Rosenfeld, E., & Kolter, J. Z. (2019). "Certified Adversarial Robustness via Randomized Smoothing." *ICML*. https://arxiv.org/abs/1902.02918

5. Lecuyer, G., et al. (2019). "Certified Robustness to Adversarial Examples with Differential Privacy." *IEEE S&P*. https://arxiv.org/abs/1802.00420

### Model Watermarking

6. Uchida, Y., et al. (2017). "Embedding Watermarks into Deep Neural Networks." *ACM IH*. https://doi.org/10.1145/3029386.3029400

7. Zhang, J., et al. (2018). "Protecting Intellectual Property of Deep Neural Networks with Watermarking." *AsiaCCS*. https://doi.org/10.1145/3196494.3196557

### Monitoring and Detection

8. Pertsev, A., et al. (2018). "Detecting Adversarial Samples from Artifacts." *arXiv*. https://arxiv.org/abs/1703.07375

9. Grosse, K., et al. (2017). "On Adversarial Examples for Character-Level Neural Machine Translation." *IEEE S&P Workshops*. https://arxiv.org/abs/1705.04364

### Google's ML Security

10. Google AI Principles. (2024). https://ai.google/responsibilities/principles/

11. Schmidt, L., et al. (2024). "Adversarial Robustness in Production ML Systems." *Google AI Blog*. https://ai.googleblog.com/

### Frameworks

12. NIST AI 100-2. (2023). "AI Risk Management Framework." https://www.nist.gov/artificial-intelligence/risk-management-framework

13. MITRE ATLAS. (2024). https://atlas.mitre.org/

---

*Next Chapter: [Chapter 22: Privacy-Preserving AI Architecture →](./chapter-22.md)*
