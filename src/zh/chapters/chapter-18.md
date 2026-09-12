# 第18章：模型压缩与优化

🟢 入门 | 🟡 中级 | 🔴 高级 | ⚫ 管理者

---

## 18.1 量化技术

### 什么是量化

量化将模型权重和激活的精度从浮点数（FP32）降低到低位表示（INT8、INT4 甚至二进制）。这减少了模型大小并加速了边缘硬件上的推理。

📌 **关键概念**：量化以少量精度损失换取模型大小缩减和推理速度的显著提升。

```
┌─────────────────────────────────────────────────────────────────┐
│                    量化概览                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FP32（原始）：                                                 │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐   │
│  │0 10000010│10110100000000000000000│                       │   │
│  │符号 │指数  │         尾数          │  = 23.5              │   │
│  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘   │
│  32 位                                                         │
│                                                                 │
│  INT8（量化后）：                                               │
│  ┌──────┬──────────────────────────────────────────────┐      │
│  │10010111│                                            │      │
│  │符号 │  值    │  = 23（近似）                          │      │
│  └──────┴──────────────────────────────────────────────┘      │
│  8 位                                                          │
│                                                                 │
│  大小缩减：32 位 → 8 位 = 缩小 4 倍                            │
│  速度提升：推理速度快 2-4 倍                                    │
│  精度影响：通常下降 0.5-2%                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 量化方法

#### 训练后量化（PTQ）

```python
# PyTorch 训练后量化
import torch
import torch.quantization as quant
import torchvision.models as models

# 加载预训练模型
model = models.resnet50(pretrained=True)
model.eval()

# 准备量化
model_quantized = quant.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},  # 要量化的层
    dtype=torch.qint8
)

# 或者使用校准的静态量化
def calibrate(model, data_loader):
    """静态量化的校准。"""
    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            model(images)

# 静态量化
model_fp32 = models.resnet50(pretrained=True)
model_fp32.eval()

model_fp32.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model_fp32)

# 使用代表性数据进行校准
calibrate(model_prepared, train_loader)

# 转换为量化模型
model_int8 = quant.convert(model_prepared)

# 比较大小
import os
torch.save(model_fp32.state_dict(), 'model_fp32.pth')
torch.save(model_int8.state_dict(), 'model_int8.pth')

fp32_size = os.path.getsize('model_fp32.pth')
int8_size = os.path.getsize('model_int8.pth')

print(f"FP32 模型大小：{fp32_size / 1e6:.2f} MB")
print(f"INT8 模型大小：{int8_size / 1e6:.2f} MB")
print(f"压缩比：{fp32_size / int8_size:.2f}x")
```

#### 量化感知训练（QAT）

```python
# 量化感知训练
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

# 准备 QAT 模型
model = models.resnet50(pretrained=True)
model_qat = QuantizableResNet(model)

# 设置 QAT 配置
model_qat.qconfig = quant.get_default_qat_qconfig('fbgemm')

# 准备 QAT
model_prepared = quant.prepare_qat(model_qat)

# 使用 QAT 微调
for epoch in range(10):
    model_prepared.train()
    for images, labels in train_loader:
        outputs = model_prepared(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证
    model_prepared.eval()
    # ... 验证代码 ...

# 转换为最终量化模型
model_final = quant.convert(model_prepared)
```

### 不同框架的量化

```yaml
# ONNX Runtime 量化配置
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
    # 用于 TensorRT 部署
    op_types: ["Conv", "MatMul", "Attention"]
    per_channel: true
```

```python
# ONNX Runtime 量化示例
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# 动态量化
quantize_dynamic(
    model_input='model.onnx',
    model_output='model_quantized.onnx',
    weight_type=QuantType.QInt8
)

# 静态量化
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

# 创建校准数据读取器
calibration_data = CalibrationDataReaderImpl(calibration_loader)

# 量化
quantize_static(
    model_input='model.onnx',
    model_output='model_static_quantized.onnx',
    calibration_data_reader=calibration_data,
    per_channel=True,
    reduce_range=False
)
```

---

## 18.2 知识蒸馏

### 概念概览

知识蒸馏将知识从大型"教师"模型转移到小型"学生"模型。学生学习模仿教师的行为，用更少的参数实现相似的精度。

```
┌─────────────────────────────────────────────────────────────────┐
│              知识蒸馏                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  教师模型（大型）：                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  输入 → [Conv1] → [Conv2] → ... → [FC] → 输出         │   │
│  │  参数：25.6M                                              │   │
│  │  精度：95.2%                                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │  知识              │                      │
│                    │  蒸馏              │                      │
│                    │  （软标签）        │                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  学生模型（小型）：                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  输入 → [Conv1] → [Conv2] → ... → [FC] → 输出         │   │
│  │  参数：2.5M                                               │   │
│  │  精度：94.1%                                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  大小缩减：缩小 10 倍                                          │
│  速度提升：快 5-10 倍                                           │
│  精度保留：教师精度的 98-99%                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 实现

```python
# 知识蒸馏实现
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
        # 软目标损失（蒸馏损失）
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 硬目标损失（标准交叉熵）
        student_loss = F.cross_entropy(student_logits, labels)
        
        # 组合损失
        loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return loss

def distill(teacher_model, student_model, train_loader, optimizer, 
            epochs=10, temperature=4.0, alpha=0.7):
    """知识蒸馏训练循环。"""
    
    teacher_model.eval()
    student_model.train()
    
    criterion = DistillationLoss(temperature, alpha)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            
            # 教师预测（不需要梯度）
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
            # 学生预测
            student_logits = student_model(images)
            
            # 计算蒸馏损失
            loss = criterion(student_logits, teacher_logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 指标
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'轮次：{epoch}，批次：{batch_idx}，'
                      f'损失：{loss.item():.4f}，'
                      f'精度：{100. * correct / total:.2f}%')
        
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader),
        }, f'checkpoint_epoch_{epoch}.pth')

# 使用示例
# 教师：ResNet-101（44.5M 参数）
# 学生：ResNet-18（11.7M 参数）
teacher = models.resnet101(pretrained=True).cuda()
student = models.resnet18(pretrained=False).cuda()

optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
distill(teacher, student, train_loader, optimizer, epochs=50)
```

### 基于特征的蒸馏

```python
# 基于特征的知识蒸馏
class FeatureDistillation(nn.Module):
    def __init__(self, teacher_feature_dims, student_feature_dims):
        super().__init__()
        # 投影层以匹配特征维度
        self.projectors = nn.ModuleList([
            nn.Linear(s_dim, t_dim)
            for s_dim, t_dim in zip(student_feature_dims, teacher_feature_dims)
        ])
    
    def forward(self, teacher_features, student_features):
        loss = 0
        for i, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
            # 投影学生特征以匹配教师维度
            s_projected = self.projectors[i](s_feat)
            
            # 特征间的 L2 距离
            loss += F.mse_loss(s_projected, t_feat)
        
        return loss / len(teacher_features)

# 使用特征蒸馏的修改训练
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
        # 从两个模型获取特征
        teacher_features = self.teacher.get_features(x)
        student_features = self.student.get_features(x)
        
        # 获取 logits
        teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        
        return student_logits, teacher_logits, teacher_features, student_features
```

---

## 18.3 模型剪枝

### 剪枝类型

```
┌─────────────────────────────────────────────────────────────────┐
│                    模型剪枝类型                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 非结构化剪枝：                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  原始：    [0.5, -0.3, 0.8, 0.1, -0.2, 0.6, 0.4]      │   │
│  │  剪枝后：  [0.5,  0.0, 0.8, 0.0,  0.0, 0.6, 0.0]      │   │
│  │             ↑     ↑    ↑    ↑     ↑    ↑    ↑          │   │
│  │            保留  置零 保留 置零  置零 保留 置零          │   │
│  └─────────────────────────────────────────────────────────┘   │
│  优点：灵活，可实现高稀疏度                                     │
│  缺点：需要稀疏矩阵支持才能加速                                 │
│                                                                 │
│  2. 结构化剪枝：                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  原始卷积层（16 个滤波器）：                             │   │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐                          │   │
│  │  │F1  │ │F2  │ │F3  │ │F4  │ ...（16 个滤波器）        │   │
│  │  └────┘ └────┘ └────┘ └────┘                          │   │
│  │                                                         │   │
│  │  剪枝后卷积层（8 个滤波器）：                           │   │
│  │  ┌────┐ ┌────┐                                         │   │
│  │  │F1  │ │F3  │  （移除 F2、F4、F6、F8...）             │   │
│  │  └────┘ └────┘                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│  优点：直接加速，无需特殊硬件                                   │
│  缺点：不够灵活，可能损失更多精度                               │
│                                                                 │
│  3. 通道剪枝：                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  原始：64 个通道 → 剪枝后：32 个通道                     │   │
│  │  直接减少模型宽度                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 使用 PyTorch 实现

```python
# 模型剪枝实现
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def structured_pruning(model, amount=0.3):
    """对 Conv2d 层应用结构化剪枝。"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module, 
                name='weight', 
                amount=amount, 
                n=2,  # L2 范数
                dim=0  # 剪枝输出通道
            )
            # 使剪枝永久化
            prune.remove(module, 'weight')

def unstructured_pruning(model, amount=0.5):
    """对所有线性层和卷积层应用非结构化剪枝。"""
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # 应用全局非结构化剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # 使剪枝永久化
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

def get_model_sparsity(model):
    """计算模型稀疏度。"""
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    sparsity = 100.0 * zero_params / total_params
    return sparsity

# 使用示例
model = models.resnet50(pretrained=True)

# 应用结构化剪枝（移除 30% 的通道）
structured_pruning(model, amount=0.3)

# 应用非结构化剪枝（50% 稀疏度）
unstructured_pruning(model, amount=0.5)

# 检查稀疏度
sparsity = get_model_sparsity(model)
print(f"模型稀疏度：{sparsity:.2f}%")

# 微调以恢复精度
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

---

## 18.4 架构搜索

### 神经架构搜索（NAS）

```
┌─────────────────────────────────────────────────────────────────┐
│              神经架构搜索                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  搜索策略                                 │   │
│  │                                                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  随机    │  │  网格    │  │  贝叶斯  │  │RL/EA │  │   │
│  │  │  搜索    │  │  搜索    │  │优化      │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  搜索空间                                 │   │
│  │                                                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  操作    │  │  连接    │  │  宽度    │  │深度  │  │   │
│  │  │(Conv,Pool)│  │  模式    │  │  乘数    │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  性能估计                                 │   │
│  │                                                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  完整    │  │  One-    │  │  权重    │  │零成本│  │   │
│  │  │  训练    │  │  Shot    │  │  共享    │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 可微架构搜索（DARTS）

```python
# DARTS 实现
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):
    """具有架构参数的混合操作。"""
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
    """DARTS 搜索单元。"""
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
    """DARTS 搜索网络。"""
    def __init__(self, C=16, num_classes=10, steps=4, layers=8):
        super().__init__()
        self.steps = steps
        self.layers = layers
        
        # 架构参数
        self.alphas_normal = nn.Parameter(
            1e-3 * torch.randn(steps * (steps + 2) // 2, 5)
        )
        self.alphas_reduce = nn.Parameter(
            1e-3 * torch.randn(steps * (steps + 2) // 2, 5)
        )
        
        # 单元
        self.cells = nn.ModuleList()
        for i in range(layers):
            reduction = (i == layers // 3) or (i == 2 * layers // 3)
            C_prev = C if i == 0 else C * steps
            cell = DARTSCell(steps, C_prev)
            self.cells.append(cell)
        
        # 分类器
        self.classifier = nn.Linear(C * steps, num_classes)
    
    def forward(self, x):
        s0 = s1 = self._stem(x)
        
        alphas = F.softmax(self.alphas_normal, dim=-1)
        
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, alphas)
        
        out = F.adaptive_avg_pool2d(s1, (1, 1))
        out = out.view(out.size(0), -1)
        return self.classifier(out)

# DARTS 训练循环
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
        # 训练架构参数
        model.train()
        for images, labels in val_loader:
            outputs = model(images.cuda())
            loss = F.cross_entropy(outputs, labels.cuda())
            
            optimizer_arch.zero_grad()
            loss.backward()
            optimizer_arch.step()
        
        # 训练网络参数
        for images, labels in train_loader:
            outputs = model(images.cuda())
            loss = F.cross_entropy(outputs, labels.cuda())
            
            optimizer_net.zero_grad()
            loss.backward()
            optimizer_net.step()
```

---

## 18.5 编译优化

### TorchScript 和 ONNX 导出

```python
# 模型编译和优化
import torch
import torch.jit as jit
import onnxruntime as ort

# TorchScript JIT 编译
def compile_torchscript(model, example_input):
    """将模型编译为 TorchScript。"""
    model.eval()
    
    # 基于跟踪的编译
    traced_model = jit.trace(model, example_input)
    
    # 针对推理优化
    optimized_model = jit.optimize_for_inference(traced_model)
    
    # 保存
    optimized_model.save("model_optimized.pt")
    
    return optimized_model

# ONNX 导出
def export_onnx(model, example_input, output_path):
    """将模型导出为 ONNX 格式。"""
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

# ONNX Runtime 优化
def optimize_onnx(model_path, optimization_level='all'):
    """优化 ONNX 模型。"""
    sess_options = ort.SessionOptions()
    
    # 图优化
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    
    # 执行模式
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # 线程池
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2
    
    # 创建会话
    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    return session

# TensorRT 优化（NVIDIA）
def optimize_tensorrt(onnx_path, fp16=True):
    """使用 TensorRT 优化。"""
    import tensorrt as trt
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # 解析 ONNX 模型
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    
    # 构建引擎
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_engine(network, config)
    
    return engine
```

---

## 💡 案例研究：使用 ONNX Runtime 优化模型

### 完整工作流

🔴 高级

```python
# 完整的 ONNX Runtime 优化管道
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
        """基准测试模型推理。"""
        input_name = session.get_inputs()[0].name
        
        # 预热
        for _ in range(10):
            session.run(None, {input_name: input_data})
        
        # 基准测试
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
        """应用静态量化。"""
        from onnxruntime.quantization import (
            quantize_static, 
            CalibrationDataReader,
            QuantFormat,
            QuantType
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
        """应用动态量化。"""
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            model_input=self.model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8
        )
        
        return output_path
    
    def optimize_graph_optimization(self, level='all'):
        """应用图优化。"""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = self.optimization_levels[level]
        
        session = ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        return session
    
    def run_full_optimization(self, output_dir):
        """运行完整优化管道。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # 基线基准测试
        baseline_session = ort.InferenceSession(
            self.model_path,
            providers=['CUDAExecutionProvider']
        )
        input_data = next(iter(self.calibration_data))
        input_data = input_data.numpy() if hasattr(input_data, 'numpy') else input_data
        results['baseline'] = self.benchmark(baseline_session, input_data)
        
        # 静态量化
        static_path = str(output_dir / 'model_static.onnx')
        self.optimize_static_quantization(static_path)
        static_session = ort.InferenceSession(
            static_path,
            providers=['CUDAExecutionProvider']
        )
        results['static_quantized'] = self.benchmark(static_session, input_data)
        
        # 动态量化
        dynamic_path = str(output_dir / 'model_dynamic.onnx')
        self.optimize_dynamic_quantization(dynamic_path)
        dynamic_session = ort.InferenceSession(
            dynamic_path,
            providers=['CPUExecutionProvider']
        )
        results['dynamic_quantized'] = self.benchmark(dynamic_session, input_data)
        
        # 图优化
        optimized_session = self.optimize_graph_optimization('all')
        results['graph_optimized'] = self.benchmark(optimized_session, input_data)
        
        return results

# 使用示例
pipeline = ONNXOptimizationPipeline(
    model_path='model.onnx',
    calibration_data=calibration_loader
)

results = pipeline.run_full_optimization('./optimized_models')

# 打印结果
for opt_name, metrics in results.items():
    print(f"\n{opt_name.upper()}:")
    print(f"  平均延迟：{metrics['mean_latency']:.2f} ms")
    print(f"  P95 延迟：{metrics['p95_latency']:.2f} ms")
    print(f"  吞吐量：{metrics['throughput']:.2f} 次推理/秒")
```

---

## 📝 练习

### 练习 18.1：量化比较
在 ResNet-50 模型上比较量化方法：
1. 应用 PTQ（训练后量化）
2. 应用 QAT（量化感知训练）
3. 比较精度和延迟
4. 在不同硬件（CPU、GPU、边缘设备）上测试

### 练习 18.2：知识蒸馏
实现文本分类的知识蒸馏：
1. 教师：BERT-base（110M 参数）
2. 学生：DistilBERT（66M 参数）或自定义小模型
3. 与直接训练学生进行比较
4. 测量精度保留和推理速度

### 练习 18.3：模型压缩管道
构建完整的压缩管道，要求：
1. 应用剪枝（30% 结构化）
2. 应用量化（INT8）
3. 导出为 ONNX
4. 在边缘硬件上进行基准测试
5. 报告大小缩减和加速比

---

## ⚠️ 警告

1. **精度下降**：始终在保留的测试集上验证压缩模型。压缩可能导致显著的精度下降。
2. **硬件兼容性**：并非所有量化格式都受所有硬件支持。检查 TensorRT、ONNX Runtime 和 TFLite 的支持情况。
3. **校准数据**：静态量化需要代表性的校准数据。使用验证集，而不是训练集。
4. **层敏感性**：某些层对压缩更敏感。考虑混合精度策略。

---

## 本章小结

本章介绍了用于边缘部署的模型压缩和优化技术：
1. 量化（PTQ、QAT、动态、静态）
2. 知识蒸馏（基于 logits、基于特征）
3. 模型剪枝（结构化、非结构化、运动剪枝）
4. 架构搜索（DARTS、NAS）
5. 编译优化（TorchScript、ONNX、TensorRT）

下一章我们将探讨边缘部署架构和策略。
