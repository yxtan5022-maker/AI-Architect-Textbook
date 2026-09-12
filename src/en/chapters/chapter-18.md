# Chapter 18: Model Compression & Optimization

## Learning Objectives

By the end of this chapter, you will be able to:

1. Apply quantization techniques (INT8, INT4, FP8) and measure their impact on model accuracy and inference speed
2. Implement knowledge distillation to transfer knowledge from a large teacher model to a smaller student model
3. Use pruning and structured sparsity to reduce model size while maintaining acceptable accuracy
4. Optimize models using ONNX Runtime and TensorRT for specific hardware targets
5. Design a compression pipeline that balances accuracy, latency, and model size constraints

---

## 18.1 Introduction: The Compression Imperative

Modern AI models are too large for many deployment scenarios. A GPT-3 class model with 175 billion parameters requires 350 GB of memory in FP16 — far beyond edge devices, and expensive to serve at scale in the cloud. Model compression reduces the computational and memory requirements of models while preserving as much accuracy as possible.

Compression is not optional for edge deployment — it is a prerequisite. Even cloud deployments benefit from compression through reduced inference costs and increased throughput.

> **📌 Real Data Box**
> ONNX Runtime, Microsoft's cross-platform inference engine, achieves **2-4x inference speedup** over default PyTorch/TensorFlow runtimes through graph optimization and hardware-specific kernels (github.com/microsoft/onnxruntime, 2026). TensorRT typically delivers **3-10x speedup** on NVIDIA GPUs through layer fusion, kernel auto-tuning, and precision calibration (developer.nvidia.com/tensorrt).

---

## 18.2 Quantization

Quantization reduces the precision of model weights and activations from floating-point (FP32/FP16) to lower-bit representations (INT8, INT4, FP8).

### 18.2.1 Quantization Levels

| Precision | Bits | Memory Reduction | Speedup (typical) | Accuracy Impact |
|-----------|------|-----------------|-------------------|-----------------|
| FP32 | 32 | 1.0x (baseline) | 1.0x | Baseline |
| FP16/BF16 | 16 | 2.0x | 1.5-2.0x | < 0.5% loss |
| FP8 (E4M3/E5M2) | 8 | 4.0x | 2.0-3.0x | 0.5-1.5% loss |
| INT8 | 8 | 4.0x | 2.0-4.0x | 0.5-2.0% loss |
| INT4 | 4 | 8.0x | 3.0-6.0x | 1.0-5.0% loss |
| Binary (1-bit) | 1 | 32.0x | 10-20x | 5-15% loss |

### 18.2.2 Quantization Methods

**Post-Training Quantization (PTQ):**
- Apply quantization after training is complete
- No retraining required
- Fast but less accurate
- Requires a calibration dataset (typically 100-1000 samples)

**Quantization-Aware Training (QAT):**
- Simulate quantization during training
- Model learns to compensate for quantization noise
- More accurate but requires training resources
- Typically recovers 0.5-1% accuracy over PTQ

**Dynamic Quantization:**
- Activations are quantized at runtime, weights are pre-quantized
- No calibration dataset needed
- Moderate speedup, variable accuracy impact

### 18.2.3 Quantization Benchmarks (Real Data)

**ResNet-50 ImageNet quantization results:**

| Method | INT8 Accuracy | FP32 Accuracy | Degradation | Speedup (CPU) |
|--------|-------------|-------------|------------|---------------|
| PTQ (default) | 75.3% | 76.1% | -0.8% | 2.8x |
| PTQ (calibrated) | 75.8% | 76.1% | -0.3% | 2.8x |
| QAT | 76.0% | 76.1% | -0.1% | 2.8x |

**BERT-base GLUE quantization results:**

| Method | INT8 Score | FP32 Score | Degradation | Speedup (CPU) |
|--------|-----------|-----------|------------|---------------|
| PTQ | 79.2 | 80.5 | -1.3 | 2.1x |
| QAT | 80.1 | 80.5 | -0.4 | 2.1x |

**LLM INT4 quantization (LLaMA-7B):**

| Method | Perplexity (↓ better) | FP16 Perplexity | Degradation |
|--------|----------------------|----------------|-------------|
| GPTQ INT4 | 5.82 | 5.68 | +0.14 |
| AWQ INT4 | 5.73 | 5.68 | +0.05 |
| GGUF Q4_K_M | 5.91 | 5.68 | +0.23 |
| bitsandbytes INT4 | 5.88 | 5.68 | +0.20 |

### 18.2.4 FP8: The New Frontier

FP8 (8-bit floating point) offers a middle ground between INT8 and FP16, with two formats:

- **E4M3** (4-bit exponent, 3-bit mantissa): Higher precision, used for weights
- **E5M2** (5-bit exponent, 2-bit mantissa): Wider dynamic range, used for activations

FP8 is supported on NVIDIA H100 GPUs and is becoming the default for LLM inference. It typically achieves 2-3x speedup over FP16 with <0.5% accuracy loss — better than INT8 for models with large activation ranges.

---

## 18.3 Knowledge Distillation

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model. The student learns to mimic the teacher's soft probability outputs, which contain richer information than hard labels.

### 18.3.1 Distillation Process

1. **Train teacher model** to high accuracy on the target task
2. **Design student model** with fewer parameters (typically 1/5 to 1/50 of teacher)
3. **Train student** with a combined loss:
   - Hard label loss (cross-entropy with ground truth)
   - Soft label loss (KL divergence between teacher and student logits)
4. **Temperature scaling** softens the teacher's probability distribution, revealing inter-class relationships

**Distillation loss formula:**

```
L = α * L_hard(y, student(x)) + (1-α) * L_soft(teacher(x; T), student(x; T))
```

Where T is the temperature (typically 2-20) and α is the weighting factor (typically 0.1-0.5).

### 18.3.2 Knowledge Distillation Results

**Computer Vision (ImageNet):**

| Teacher | Student | Teacher Top-1 | Student Top-1 | Size Reduction |
|---------|---------|--------------|--------------|---------------|
| ResNet-152 | ResNet-50 | 78.3% | 77.0% | 2.6x |
| ResNet-152 | ResNet-34 | 78.3% | 75.4% | 4.2x |
| EfficientNet-B7 | EfficientNet-B3 | 84.3% | 81.6% | 8.3x |
| ViT-L/14 | ViT-B/14 | 87.8% | 84.2% | 5.4x |

**NLP (GLUE Benchmark):**

| Teacher | Student | Teacher Score | Student Score | Size Reduction |
|---------|---------|-------------|--------------|---------------|
| BERT-Large | BERT-Base | 80.5 | 79.1 | 3.3x |
| BERT-Large | DistilBERT | 80.5 | 77.0 | 6.6x |
| RoBERTa-Large | BERT-Base | 83.2 | 79.1 | 3.3x |

---

## 18.4 Pruning and Structured Sparsity

Pruning removes redundant weights or entire structures from the model.

### 18.4.1 Pruning Types

| Type | What is Removed | Hardware Benefit | Accuracy Impact |
|------|----------------|-----------------|-----------------|
| **Unstructured** | Individual weights below threshold | Sparse matrix support needed | Minimal with gradual pruning |
| **Structured** | Entire filters/channels/heads | Direct speedup on any hardware | Moderate (2-5% loss) |
| **Semi-structured** | N:M sparse patterns (e.g., 2:4) | NVIDIA Ampere+ sparse Tensor Cores | Minimal (0.5-1% loss) |

### 18.4.2 NVIDIA 2:4 Sparse Tensor Cores

NVIDIA Ampere and newer GPUs support 2:4 structured sparsity: out of every 4 consecutive elements, at most 2 can be non-zero. This provides a **2x theoretical speedup** with dedicated sparse Tensor Core hardware.

**2:4 sparsity benchmark (A100 GPU):**

| Model | Dense FP16 | 2:4 Sparse FP16 | Speedup | Accuracy Loss |
|-------|-----------|----------------|---------|---------------|
| ResNet-50 | 4.8ms | 3.1ms | 1.55x | -0.3% |
| BERT-base | 2.1ms | 1.4ms | 1.50x | -0.4% |
| YOLOv5-L | 8.2ms | 5.3ms | 1.55x | -0.5% |

---

## 18.5 ONNX Runtime Optimization

ONNX Runtime provides a unified inference engine that optimizes models for specific hardware targets.

### 18.5.1 ONNX Runtime Optimization Passes

| Optimization | Description | Typical Speedup |
|-------------|------------|----------------|
| **Graph fusion** | Merge multiple operations into single kernels | 1.2-2.0x |
| **Constant folding** | Pre-compute static expressions | 1.1-1.3x |
| **Layer normalization fusion** | Fuse LayerNorm into single kernel | 1.3-1.8x |
| **Attention fusion** | Optimize multi-head attention pattern | 1.5-2.5x |
| **Quantization** | INT8/INT4 weight conversion | 2.0-4.0x |
| **Execution provider** | Hardware-specific kernels (CUDA, TensorRT, OpenVINO) | 2.0-5.0x |

### 18.5.2 ONNX Runtime Benchmark Results

**ResNet-50 inference across execution providers (batch size 1, latency):**

| Provider | CPU (x86) | CUDA (A100) | TensorRT (A100) | OpenVINO |
|----------|----------|------------|----------------|----------|
| PyTorch default | 12.3ms | 1.8ms | N/A | N/A |
| ONNX Runtime | 8.1ms | 1.2ms | 0.6ms | 3.2ms |

**BERT-base inference (sequence length 128, batch size 1):**

| Provider | CPU | CUDA (A100) | TensorRT |
|----------|-----|------------|----------|
| PyTorch default | 8.2ms | 1.1ms | N/A |
| ONNX Runtime | 4.5ms | 0.7ms | 0.3ms |

---

## 18.6 Case Study: How Microsoft Optimizes Models for Edge

Microsoft deploys AI models across billions of edge devices: Windows PCs, Xbox consoles, HoloLens headsets, and Azure IoT devices. Their optimization strategy spans multiple levels.

**Optimization pipeline:**

| Stage | Tool/Technique | Purpose |
|-------|---------------|---------|
| Model design | Architectural search for edge | Hardware-aware NAS |
| Training | Knowledge distillation | Transfer from cloud to edge model |
| Compression | INT8 quantization + pruning | Reduce model size |
| Runtime optimization | ONNX Runtime + DirectML | Hardware-agnostic GPU acceleration |
| Deployment | Windows ML / ONNX Runtime | Unified inference on any Windows device |

**Key technologies:**

1. **DirectML:** A hardware-agnostic GPU acceleration API for Windows, enabling ONNX Runtime to use GPU on any vendor (NVIDIA, AMD, Intel) without vendor-specific code.

2. **Windows ML:** A built-in inference engine in Windows 10/11 that runs ONNX models using the device's best available hardware (GPU, CPU, or NPU).

3. **Model Inspector:** An internal tool that profiles model execution on target hardware and recommends specific optimizations (which layers to fuse, which to quantize, where bottlenecks exist).

**Scale metrics:**

- 1.4 billion+ active Windows devices
- 100+ million Office 365 users benefiting from edge AI (smart replies, dictation, image enhancement)
- 50+ million Xbox users benefiting from AI upscaling (DirectX Super Resolution)
- Average model size reduction through optimization: 4-8x
- Average inference speedup: 3-6x

**Example: Microsoft Teams background blur**

| Metric | Original Model | Optimized Model |
|--------|---------------|----------------|
| Model size | 85MB | 12MB |
| Inference time (CPU) | 32ms | 8ms |
| RAM usage | 220MB | 45MB |
| Accuracy (IoU) | 0.94 | 0.92 |
| Device compatibility | GPU only | CPU and GPU |

---

## 18.7 War Story: Quantized Model That Lost 20% Accuracy

**Company:** E-commerce product classification system

**Problem:** The team quantized their product classification model (ResNeXt-101) from FP32 to INT8 for deployment on edge devices. The quantized model's accuracy dropped from 94.2% to 74.1% — a catastrophic 20% drop.

**Root cause analysis:**

1. **Representative calibration data was wrong.** The calibration dataset for PTQ contained only well-lit, centered product images. Production images included blurry photos, unusual angles, poor lighting, and occluded products.

2. **Model had outlier activations.** The last three layers of the network had activation values 10-50x larger than typical values. INT8's limited range (−128 to 127) clipped these outliers, destroying the model's ability to discriminate between similar products.

3. **No per-layer quantization analysis.** The team applied a single global quantization scale to all layers, rather than analyzing each layer's sensitivity.

**Diagnostic process:**

```python
# Check activation distributions per layer
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        activations = hook_outputs[name]
        print(f"{name}: range=[{activations.min():.2f}, {activations.max():.2f}], "
              f"std={activations.std():.2f}, "
              f"outliers={(activations.abs() > 10).sum() / activations.numel():.4%}")
```

Output revealed that layer `stage4.conv3` had activations ranging from −450 to +380, while most layers had ranges of −5 to +5. This layer alone was responsible for most of the quantization error.

**Solutions applied:**

| Fix | Accuracy Recovery |
|-----|------------------|
| Per-layer quantization (mixed precision) | +8.2% (74.1% → 82.3%) |
| Calibration data = 5000 production images | +4.5% (82.3% → 86.8%) |
| Exclude sensitive layers from INT8 (keep FP16) | +5.1% (86.8% → 91.9%) |
| QAT with production data | +1.8% (91.9% → 93.7%) |

**Final accuracy: 93.7%** (vs 94.2% FP32 baseline — only 0.5% loss)

**Key lessons:**

1. Always profile activation distributions before quantizing
2. Calibration data MUST represent the production data distribution
3. Mixed-precision quantization (FP16 for sensitive layers, INT8 for others) is almost always better than uniform INT8
4. QAT recovers accuracy that PTQ cannot, at the cost of training time
5. Validate quantized model accuracy on the ACTUAL production test set, not a clean benchmark set

---

## 18.8 Compression Technique Comparison

| Technique | Accuracy Loss | Speedup | Implementation Effort | Best For |
|-----------|-------------|---------|----------------------|----------|
| FP16/BF16 | < 0.5% | 1.5-2x | Low (default) | Universal baseline |
| INT8 PTQ | 0.5-2% | 2-4x | Low | Quick optimization |
| INT8 QAT | < 0.5% | 2-4x | Medium | High-accuracy needs |
| INT4 (GPTQ/AWQ) | 0.5-3% | 3-6x | Medium | LLM deployment |
| Structured pruning | 2-5% | 1.5-3x | Medium | Model size reduction |
| 2:4 sparsity | 0.3-1% | 1.5x | Low | NVIDIA Ampere+ |
| Knowledge distillation | 2-5% | Varies | High | Custom small models |
| ONNX Runtime | 0% | 2-5x | Low | Framework migration |

---

## 18.9 When to Use / When Not to Use Compression

### When to Use Model Compression

| Scenario | Recommended Technique |
|----------|---------------------|
| Edge deployment with latency < 20ms | INT8 QAT + structured pruning |
| LLM deployment on consumer GPU | INT4 (GPTQ/AWQ) |
| Cloud cost reduction at scale | INT8 PTQ + ONNX Runtime |
| Deployment to diverse hardware | ONNX Runtime + FP16 |
| Model too large for target memory | Structured pruning + quantization |
| Real-time inference at high QPS | TensorRT + INT8 + 2:4 sparsity |

### When NOT to Use Model Compression

| Scenario | Why Compression Hurts | Alternative |
|----------|----------------------|-------------|
| Accuracy is paramount (medical, safety) | Any accuracy loss is unacceptable | Use full-precision cloud inference |
| Model is already small (< 10MB) | Compression overhead > benefit | Deploy as-is |
| Batch processing (no latency requirement) | No speedup needed | Focus on throughput, not compression |
| Prototyping phase | Compression adds complexity | Optimize after model is finalized |
| Model will be retrained soon | Compression work is wasted | Compress only production models |

---

## 18.10 Summary

- **Quantization** reduces model precision from FP32/FP16 to INT8/INT4/FP8, achieving 2-6x speedup with 0.5-3% accuracy loss
- **Knowledge distillation** trains smaller student models from larger teachers, achieving 3-8x size reduction with 2-5% accuracy loss
- **Structured pruning** (especially NVIDIA 2:4 sparsity) provides hardware-accelerated speedup with minimal accuracy impact
- **ONNX Runtime** provides 2-5x inference speedup through graph optimization and hardware-specific execution providers
- **Mixed-precision** approaches (different precision for different layers) consistently outperform uniform quantization
- The most common failure is quantizing without representative calibration data — always validate on production data distributions

---

## Discussion Questions

1. You have a BERT-large model (340M parameters) that needs to run inference on a laptop CPU in under 10ms. Design a compression pipeline that achieves this target. What techniques would you combine, and what accuracy loss would you expect?

2. Compare INT8 quantization with knowledge distillation for reducing a ResNet-152 model. Which approach gives better accuracy at the same model size? When would you choose one over the other?

3. A team is deploying an LLM to edge devices with 4GB of RAM. The model has 7B parameters. What combination of techniques would you use to fit the model in memory while maintaining acceptable quality?

4. Why does mixed-precision quantization consistently outperform uniform quantization? Provide a technical explanation based on the distribution of weights and activations in neural networks.

5. Microsoft uses DirectML for hardware-agnostic GPU acceleration. What are the tradeoffs of using a vendor-agnostic runtime versus vendor-specific optimization (TensorRT for NVIDIA, OpenVINO for Intel)?

---

## Exercises

**Exercise 1:** Take a pre-trained ResNet-50 model and apply INT8 post-training quantization using ONNX Runtime. Measure the accuracy on ImageNet validation set before and after quantization. Then apply quantization-aware training and compare the results.

**Exercise 2:** Implement knowledge distillation from ResNet-152 (teacher) to ResNet-50 (student) on CIFAR-100. Experiment with different temperature values (2, 5, 10, 20) and loss weighting factors (0.1, 0.3, 0.5, 0.7). Plot the accuracy vs temperature curve.

**Exercise 3:** Profile a YOLOv8 model using TensorRT's profiling tools. Identify the top 5 most time-consuming layers. Apply targeted optimization (kernel selection, layer fusion) to those specific layers and measure the improvement.

---

## References

- ONNX Runtime Documentation: https://onnxruntime.ai/docs/
- ONNX Runtime GitHub: https://github.com/microsoft/onnxruntime
- NVIDIA TensorRT Documentation: https://developer.nvidia.com/tensorrt
- TensorRT Developer Zone: https://developer.nvidia.com/tensorrt
- Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Jacob et al., 2018): https://arxiv.org/abs/1712.05877
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2023): https://arxiv.org/abs/2210.17323
- AWQ: Activation-aware Weight Quantization (Lin et al., 2024): https://arxiv.org/abs/2306.00978
- Hugging Face Optimum (Model Optimization): https://huggingface.co/docs/optimum/
- NVIDIA Sparsity Documentation: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/#sparse-tensor-cores
