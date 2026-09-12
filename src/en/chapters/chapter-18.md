# Chapter 18: Model Compression & Optimization

🟢 Beginner | 🟡 Intermediate | 🔴 Advanced | ⚫ Manager

---

## 18.1 Quantization Techniques

### What is Quantization?

Quantization reduces the precision of model weights and activations from floating-point (FP32) to lower-bit representations (INT8, INT4, or even binary). This reduces model size and accelerates inference on edge hardware.

📌 **Key Concept**: Quantization trades a small amount of accuracy for significant gains in model size reduction and inference speed.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Quantization Overview                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FP32 (Original):                                              │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐   │
│  │0 10000010│10110100000000000000000│                       │   │
│  │Sign │Exponent│     Mantissa       │  = 23.5              │   │
│  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘   │
│  32 bits                                                       │
│                                                                 │
│  INT8 (Quantized):                                             │
│  ┌──────┬──────────────────────────────────────────────┐      │
│  │10010111│                                            │      │
│  │Sign │  Value  │  = 23 (approx)                      │      │
│  └──────┴──────────────────────────────────────────────┘      │
│  8 bits                                                        │
│                                                                 │
│  Size Reduction: 32 bits → 8 bits = 4x smaller                 │
│  Speed Improvement: 2-4x faster inference                      │
│  Accuracy Impact: 0.5-2% drop (typically)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quantization Methods

#### Post-Training Quantization (PTQ)

```python
# PyTorch Post-Training Quantization
import torch
import torch.quantization as quant
import torchvision.models as models

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Prepare for quantization
model_quantized = quant.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
    dtype=torch.qint8
)

# Alternatively, static quantization with calibration
def calibrate(model, data_loader):
    """Calibration for static quantization."""
    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            model(images)

# Static quantization
model_fp32 = models.resnet50(pretrained=True)
model_fp32.eval()

model_fp32.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model_fp32)

# Calibrate with representative data
calibrate(model_prepared, train_loader)

# Convert to quantized model
model_int8 = quant.convert(model_prepared)

# Compare sizes
import os
torch.save(model_fp32.state_dict(), 'model_fp32.pth')
torch.save(model_int8.state_dict(), 'model_int8.pth')

fp32_size = os.path.getsize('model_fp32.pth')
int8_size = os.path.getsize('model_int8.pth')

print(f"FP32 model size: {fp32_size / 1e6:.2f} MB")
print(f"INT8 model size: {int8_size / 1e6:.2f} MB")
print(f"Compression ratio: {fp32_size / int8_size:.2f}x")
```

#### Quantization-Aware Training (QAT)

```python
# Quantization-Aware Training
import torch
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub

class QuantizableResNet(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

# Prepare model for QAT
model = models.resnet50(pretrained=True)
model_qat = QuantizableResNet(model)

# Set QAT configuration
model_qat.qconfig = quant.get_default_qat_qconfig('fbgemm')

# Prepare for QAT
model_prepared = quant.prepare_qat(model_qat)

# Fine-tune with QAT
for epoch in range(10):
    model_prepared.train()
    for images, labels in train_loader:
        outputs = model_prepared(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validate
    model_prepared.eval()
    # ... validation code ...

# Convert to final quantized model
model_final = quant.convert(model_prepared)
```

### Quantization for Different Frameworks

```yaml
# ONNX Runtime Quantization
quantization_config:
  static_quantization:
    data_reader: "calibration_data_reader.py"
    calibration_method: "minmax"
    weight_type: "int8"
    activation_type: "int8"
    per_channel: true
    reduce_range: false
    
  dynamic_quantization:
    weight_type: "int8"
    activation_type: "float32"
    
 qdq_quantization:
    # For deployment on TensorRT
    op_types: ["Conv", "MatMul", "Attention"]
    per_channel: true
```

```python
# ONNX Runtime Quantization Example
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization
quantize_dynamic(
    model_input='model.onnx',
    model_output='model_quantized.onnx',
    weight_type=QuantType.QInt8
)

# Static quantization
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class CalibrationDataReaderImpl(CalibrationDataReader):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.enum_data = iter(data_loader)
    
    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is None:
            return None
        return {"input": batch.numpy()}

# Create calibration reader
calibration_data = CalibrationDataReaderImpl(calibration_loader)

# Quantize
quantize_static(
    model_input='model.onnx',
    model_output='model_static_quantized.onnx',
    calibration_data_reader=calibration_data,
    per_channel=True,
    reduce_range=False
)
```

---

## 18.2 Knowledge Distillation

### Concept Overview

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model. The student learns to mimic the teacher's behavior, achieving similar accuracy with much fewer parameters.

```
┌─────────────────────────────────────────────────────────────────┐
│              Knowledge Distillation                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Teacher Model (Large):                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input → [Conv1] → [Conv2] → ... → [FC] → Output      │   │
│  │  Parameters: 25.6M                                       │   │
│  │  Accuracy: 95.2%                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │  Knowledge        │                      │
│                    │  Distillation     │                      │
│                    │  (Soft Labels)    │                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  Student Model (Small):                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Input → [Conv1] → [Conv2] → ... → [FC] → Output      │   │
│  │  Parameters: 2.5M                                        │   │
│  │  Accuracy: 94.1%                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Size Reduction: 10x smaller                                   │
│  Speed Improvement: 5-10x faster                               │
│  Accuracy Retention: 98-99% of teacher accuracy                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# Knowledge Distillation Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # Soft target loss (distillation loss)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Hard target loss (standard cross-entropy)
        student_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return loss

def distill(teacher_model, student_model, train_loader, optimizer, 
            epochs=10, temperature=4.0, alpha=0.7):
    """Knowledge distillation training loop."""
    
    teacher_model.eval()
    student_model.train()
    
    criterion = DistillationLoss(temperature, alpha)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            
            # Teacher predictions (no gradient needed)
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
            # Student predictions
            student_logits = student_model(images)
            
            # Calculate distillation loss
            loss = criterion(student_logits, teacher_logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader),
        }, f'checkpoint_epoch_{epoch}.pth')

# Example usage
# Teacher: ResNet-101 (44.5M params)
# Student: ResNet-18 (11.7M params)
teacher = models.resnet101(pretrained=True).cuda()
student = models.resnet18(pretrained=False).cuda()

optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
distill(teacher, student, train_loader, optimizer, epochs=50)
```

### Feature-based Distillation

```python
# Feature-based Knowledge Distillation
class FeatureDistillation(nn.Module):
    def __init__(self, teacher_feature_dims, student_feature_dims):
        super().__init__()
        # Projection layers to match feature dimensions
        self.projectors = nn.ModuleList([
            nn.Linear(s_dim, t_dim)
            for s_dim, t_dim in zip(student_feature_dims, teacher_feature_dims)
        ])
    
    def forward(self, teacher_features, student_features):
        loss = 0
        for i, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
            # Project student features to match teacher dimensions
            s_projected = self.projectors[i](s_feat)
            
            # L2 distance between features
            loss += F.mse_loss(s_projected, t_feat)
        
        return loss / len(teacher_features)

# Modified training with feature distillation
class DistillationModel(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.feature_distill = FeatureDistillation(
            teacher.feature_dims,
            student.feature_dims
        )
    
    def forward(self, x):
        # Get features from both models
        teacher_features = self.teacher.get_features(x)
        student_features = self.student.get_features(x)
        
        # Get logits
        teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        
        return student_logits, teacher_logits, teacher_features, student_features
```

---

## 18.3 Model Pruning

### Types of Pruning

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Pruning Types                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Unstructured Pruning:                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Original:    [0.5, -0.3, 0.8, 0.1, -0.2, 0.6, 0.4]   │   │
│  │  Pruned:      [0.5,  0.0, 0.8, 0.0,  0.0, 0.6, 0.0]   │   │
│  │               ↑     ↑    ↑    ↑     ↑    ↑    ↑         │   │
│  │              keep  zero keep zero  zero keep zero        │   │
│  └─────────────────────────────────────────────────────────┘   │
│  Pros: Flexible, can achieve high sparsity                     │
│  Cons: Requires sparse matrix support for speedup              │
│                                                                 │
│  2. Structured Pruning:                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Original Conv Layer (16 filters):                      │   │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐                          │   │
│  │  │F1  │ │F2  │ │F3  │ │F4  │ ... (16 filters)         │   │
│  │  └────┘ └────┘ └────┘ └────┘                          │   │
│  │                                                         │   │
│  │  Pruned Conv Layer (8 filters):                        │   │
│  │  ┌────┐ ┌────┐                                         │   │
│  │  │F1  │ │F3  │  (removed F2, F4, F6, F8, ...)          │   │
│  │  └────┘ └────┘                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│  Pros: Direct speedup, no special hardware needed              │
│  Cons: Less flexible, may lose more accuracy                   │
│                                                                 │
│  3. Channel Pruning:                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Original: 64 channels → Pruned: 32 channels            │   │
│  │  Reduces model width directly                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation with PyTorch

```python
# Model Pruning Implementation
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def structured_pruning(model, amount=0.3):
    """Apply structured pruning to Conv2d layers."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module, 
                name='weight', 
                amount=amount, 
                n=2,  # L2 norm
                dim=0  # Prune output channels
            )
            # Make pruning permanent
            prune.remove(module, 'weight')

def unstructured_pruning(model, amount=0.5):
    """Apply unstructured pruning to all linear and conv layers."""
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

def get_model_sparsity(model):
    """Calculate model sparsity."""
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    sparsity = 100.0 * zero_params / total_params
    return sparsity

# Example usage
model = models.resnet50(pretrained=True)

# Apply structured pruning (remove 30% of channels)
structured_pruning(model, amount=0.3)

# Apply unstructured pruning (50% sparsity)
unstructured_pruning(model, amount=0.5)

# Check sparsity
sparsity = get_model_sparsity(model)
print(f"Model sparsity: {sparsity:.2f}%")

# Fine-tune to recover accuracy
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        outputs = model(images.cuda())
        loss = criterion(outputs, labels.cuda())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Movement Pruning

```python
# Movement Pruning (trained sparsity)
class MovementPruning:
    def __init__(self, model, initial_sparsity=0.0, final_sparsity=0.5, 
                 steps=1000):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.steps = steps
        self.current_step = 0
        
        # Initialize masks
        self.masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.masks[name] = torch.ones_like(param)
    
    def update_mask(self):
        """Update pruning masks based on weight movement."""
        self.current_step += 1
        
        # Calculate current sparsity target
        sparsity = self._get_sparsity()
        
        for name, param in self.model.named_parameters():
            if name in self.masks:
                # Calculate movement threshold
                threshold = torch.quantile(param.abs(), sparsity)
                
                # Update mask
                self.masks[name] = (param.abs() >= threshold).float()
                
                # Apply mask
                param.data *= self.masks[name]
    
    def _get_sparsity(self):
        """Calculate current sparsity using cubic schedule."""
        t = self.current_step / self.steps
        return self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (1 - t) ** 3
```

---

## 18.4 Architecture Search

### Neural Architecture Search (NAS)

```
┌─────────────────────────────────────────────────────────────────┐
│              Neural Architecture Search                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Search Strategy                         │   │
│  │                                                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Random  │  │  Grid    │  │  Bayesian│  │RL/EA │  │   │
│  │  │  Search  │  │  Search  │  │Optim.    │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Search Space                           │   │
│  │                                                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Ops     │  │  Connect │  │  Width   │  │Depth │  │   │
│  │  │(Conv,Pool)│  │  Pattern │  │  Multipl.│  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Performance Estimation                  │   │
│  │                                                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Full    │  │  One-    │  │  Weight  │  │Zero- │  │   │
│  │  │  Train   │  │  Shot    │  │Sharing   │  │Cost  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Differentiable Architecture Search (DARTS)

```python
# DARTS Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):
    """Mixed operation with architecture parameters."""
    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, C, 3, stride, 1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(C, C, 5, stride, 2, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            ),
            nn.MaxPool2d(3, stride, 1),
            nn.AvgPool2d(3, stride, 1),
            nn.Sequential(
                nn.Conv2d(C, C, 3, stride, 1, groups=C, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            ),
        ])
    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class DARTSCell(nn.Module):
    """DARTS search cell."""
    def __init__(self, steps, C):
        super().__init__()
        self.steps = steps
        self.ops = nn.ModuleList()
        
        for _ in range(steps):
            for _ in range(steps + 2):
                self.ops.append(MixedOp(C, stride=1))
    
    def forward(self, s0, s1, alphas):
        states = [s0, s1]
        offset = 0
        
        for _ in range(self.steps):
            s = sum(
                self.ops[offset + j](states[j], alphas[offset + j])
                for j in range(len(states))
            )
            offset += len(states)
            states.append(s)
        
        return torch.cat(states[-self.steps:], dim=1)

class DARTSNetwork(nn.Module):
    """DARTS search network."""
    def __init__(self, C=16, num_classes=10, steps=4, layers=8):
        super().__init__()
        self.steps = steps
        self.layers = layers
        
        # Architecture parameters
        self.alphas_normal = nn.Parameter(
            1e-3 * torch.randn(steps * (steps + 2) // 2, 5)
        )
        self.alphas_reduce = nn.Parameter(
            1e-3 * torch.randn(steps * (steps + 2) // 2, 5)
        )
        
        # Cells
        self.cells = nn.ModuleList()
        for i in range(layers):
            reduction = (i == layers // 3) or (i == 2 * layers // 3)
            C_prev = C if i == 0 else C * steps
            cell = DARTSCell(steps, C_prev)
            self.cells.append(cell)
        
        # Classifier
        self.classifier = nn.Linear(C * steps, num_classes)
    
    def forward(self, x):
        s0 = s1 = self._stem(x)
        
        alphas = F.softmax(self.alphas_normal, dim=-1)
        
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, alphas)
        
        out = F.adaptive_avg_pool2d(s1, (1, 1))
        out = out.view(out.size(0), -1)
        return self.classifier(out)

# Training loop for DARTS
def train_darts(model, train_loader, val_loader, epochs=50):
    optimizer_arch = torch.optim.Adam(
        [model.alphas_normal, model.alphas_reduce],
        lr=3e-4, weight_decay=1e-3
    )
    optimizer_net = torch.optim.SGD(
        model.parameters(),
        lr=0.025, momentum=0.9, weight_decay=3e-4
    )
    
    for epoch in range(epochs):
        # Train architecture parameters
        model.train()
        for images, labels in val_loader:
            outputs = model(images.cuda())
            loss = F.cross_entropy(outputs, labels.cuda())
            
            optimizer_arch.zero_grad()
            loss.backward()
            optimizer_arch.step()
        
        # Train network parameters
        for images, labels in train_loader:
            outputs = model(images.cuda())
            loss = F.cross_entropy(outputs, labels.cuda())
            
            optimizer_net.zero_grad()
            loss.backward()
            optimizer_net.step()
```

---

## 18.5 Compilation Optimization

### TorchScript and ONNX Export

```python
# Model Compilation and Optimization
import torch
import torch.jit as jit
import onnxruntime as ort

# TorchScript JIT Compilation
def compile_torchscript(model, example_input):
    """Compile model to TorchScript."""
    model.eval()
    
    # Trace-based compilation
    traced_model = jit.trace(model, example_input)
    
    # Optimize for inference
    optimized_model = jit.optimize_for_inference(traced_model)
    
    # Save
    optimized_model.save("model_optimized.pt")
    
    return optimized_model

# ONNX Export
def export_onnx(model, example_input, output_path):
    """Export model to ONNX format."""
    model.eval()
    
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

# ONNX Runtime Optimization
def optimize_onnx(model_path, optimization_level='all'):
    """Optimize ONNX model."""
    sess_options = ort.SessionOptions()
    
    # Graph optimization
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    
    # Execution mode
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Thread pool
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2
    
    # Create session
    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    return session

# TensorRT Optimization (NVIDIA)
def optimize_tensorrt(onnx_path, fp16=True):
    """Optimize with TensorRT."""
    import tensorrt as trt
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    
    # Build engine
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_engine(network, config)
    
    return engine
```

---

## 💡 Case Study: Using ONNX Runtime for Model Optimization

### Complete Workflow

🔴 Advanced

```python
# Complete ONNX Runtime Optimization Pipeline
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path

class ONNXOptimizationPipeline:
    def __init__(self, model_path, calibration_data):
        self.model_path = model_path
        self.calibration_data = calibration_data
        self.optimization_levels = {
            'basic': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            'extended': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            'all': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        }
    
    def benchmark(self, session, input_data, num_runs=100):
        """Benchmark model inference."""
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(10):
            session.run(None, {input_name: input_data})
        
        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            session.run(None, {input_name: input_data})
            latencies.append(time.time() - start)
        
        return {
            'mean_latency': np.mean(latencies) * 1000,
            'p50_latency': np.percentile(latencies, 50) * 1000,
            'p95_latency': np.percentile(latencies, 95) * 1000,
            'p99_latency': np.percentile(latencies, 99) * 1000,
            'throughput': 1000 / (np.mean(latencies) * 1000)
        }
    
    def optimize_static_quantization(self, output_path):
        """Apply static quantization."""
        from onnxruntime.quantization import (
            quantize_static, 
            CalibrationDataReader,
            QuantFormat
        )
        
        class DataReader(CalibrationDataReader):
            def __init__(self, data):
                self.data = data
                self.enum_data = iter(data)
            
            def get_next(self):
                batch = next(self.enum_data, None)
                if batch is None:
                    return None
                return {"input": batch.numpy() if hasattr(batch, 'numpy') else batch}
        
        reader = DataReader(self.calibration_data)
        
        quantize_static(
            model_input=self.model_path,
            model_output=output_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            op_types_to_quantize=['Conv', 'MatMul', 'Attention']
        )
        
        return output_path
    
    def optimize_dynamic_quantization(self, output_path):
        """Apply dynamic quantization."""
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            model_input=self.model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8
        )
        
        return output_path
    
    def optimize_graph_optimization(self, level='all'):
        """Apply graph optimization."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = self.optimization_levels[level]
        
        # Enable CPU optimizations
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_profiling = True
        
        session = ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        return session
    
    def run_full_optimization(self, output_dir):
        """Run complete optimization pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Baseline benchmark
        baseline_session = ort.InferenceSession(
            self.model_path,
            providers=['CUDAExecutionProvider']
        )
        input_data = next(iter(self.calibration_data))
        input_data = input_data.numpy() if hasattr(input_data, 'numpy') else input_data
        results['baseline'] = self.benchmark(baseline_session, input_data)
        
        # Static quantization
        static_path = str(output_dir / 'model_static.onnx')
        self.optimize_static_quantization(static_path)
        static_session = ort.InferenceSession(
            static_path,
            providers=['CUDAExecutionProvider']
        )
        results['static_quantized'] = self.benchmark(static_session, input_data)
        
        # Dynamic quantization
        dynamic_path = str(output_dir / 'model_dynamic.onnx')
        self.optimize_dynamic_quantization(dynamic_path)
        dynamic_session = ort.InferenceSession(
            dynamic_path,
            providers=['CPUExecutionProvider']
        )
        results['dynamic_quantized'] = self.benchmark(dynamic_session, input_data)
        
        # Graph optimization
        optimized_session = self.optimize_graph_optimization('all')
        results['graph_optimized'] = self.benchmark(optimized_session, input_data)
        
        return results

# Usage example
pipeline = ONNXOptimizationPipeline(
    model_path='model.onnx',
    calibration_data=calibration_loader
)

results = pipeline.run_full_optimization('./optimized_models')

# Print results
for opt_name, metrics in results.items():
    print(f"\n{opt_name.upper()}:")
    print(f"  Mean Latency: {metrics['mean_latency']:.2f} ms")
    print(f"  P95 Latency: {metrics['p95_latency']:.2f} ms")
    print(f"  Throughput: {metrics['throughput']:.2f} inferences/sec")
```

---

## 📝 Exercises

### Exercise 18.1: Quantization Comparison
Compare quantization methods on a ResNet-50 model:
1. Apply PTQ (post-training quantization)
2. Apply QAT (quantization-aware training)
3. Compare accuracy and latency
4. Test on different hardware (CPU, GPU, edge device)

### Exercise 18.2: Knowledge Distillation
Implement knowledge distillation for text classification:
1. Teacher: BERT-base (110M params)
2. Student: DistilBERT (66M params) or custom small model
3. Compare with directly training the student
4. Measure accuracy retention and inference speed

### Exercise 18.3: Model Compression Pipeline
Build a complete compression pipeline that:
1. Applies pruning (30% structured)
2. Applies quantization (INT8)
3. Exports to ONNX
4. Benchmarks on edge hardware
5. Reports size reduction and speedup

---

## ⚠️ Warnings

1. **Accuracy Degradation**: Always validate compressed models on held-out test sets. Compression can cause significant accuracy drops.
2. **Hardware Compatibility**: Not all quantization formats are supported on all hardware. Check TensorRT, ONNX Runtime, and TFLite support.
3. **Calibration Data**: Static quantization requires representative calibration data. Use validation set, not training set.
4. **Layer Sensitivity**: Some layers are more sensitive to compression. Consider mixed-precision strategies.

---

## Summary

This chapter covered model compression and optimization techniques for edge deployment:
1. Quantization (PTQ, QAT, dynamic, static)
2. Knowledge distillation (logit-based, feature-based)
3. Model pruning (structured, unstructured, movement)
4. Architecture search (DARTS, NAS)
5. Compilation optimization (TorchScript, ONNX, TensorRT)

Next, we'll explore edge deployment architectures and strategies.
