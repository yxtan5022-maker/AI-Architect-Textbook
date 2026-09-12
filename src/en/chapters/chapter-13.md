# Chapter 13: Model Fine-tuning Architecture

> 🟡 Intermediate → 🔴 Advanced | 25-30 min read | Part 4: Large Model Architecture

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Compare full fine-tuning vs parameter-efficient fine-tuning (LoRA, QLoRA, Adapters) on memory, speed, and quality
2. Design an instruction tuning pipeline from data preparation to evaluation
3. Implement LoRA/QLoRA fine-tuning using PEFT, TRL, and Unsloth
4. Compare RLHF, DPO, and KTO for alignment training
5. Identify and prevent common fine-tuning failures (catastrophic forgetting, overfitting, reward hacking)
6. Calculate resource requirements for fine-tuning different model sizes

---

## Table of Contents

- [13.1 Full vs Parameter-Efficient Fine-tuning](#131-full-vs-parameter-efficient-fine-tuning)
- [13.2 LoRA/QLoRA Architecture Design](#132-loraqlora-architecture-design)
- [13.3 Instruction Tuning Pipeline](#133-instruction-tuning-pipeline)
- [13.4 RLHF/DPO Architecture](#134-rlhfdpo-architecture)
- [13.5 Fine-tuning Data Management](#135-fine-tuning-data-management)
- [13.6 Evaluation & Monitoring](#136-evaluation--monitoring)
- [💡 Case Study: How Healthcare Companies Fine-tune for Clinical NLP](#-case-study-how-healthcare-companies-fine-tune-for-clinical-nlp)
- [⚠️ War Story: The Fine-tuned Model That Got Worse Than Base](#️-war-story-the-fine-tuned-model-that-got-worse-than-base)
- [📝 When to Use / When Not to Use](#-when-to-use--when-not-to-use)
- [Summary](#summary)
- [Discussion Questions](#discussion-questions)
- [Exercises](#exercises)
- [References](#references)

---

## 13.1 Full vs Parameter-Efficient Fine-tuning

### 13.1.1 The Fine-tuning Spectrum

Fine-tuning adapts a pre-trained model to specific tasks or domains. The choice between full fine-tuning and parameter-efficient fine-tuning (PEFT) depends on available resources, data size, and task requirements:

```
┌──────────────────────────────────────────────────────────────┐
│              Fine-tuning Approaches Spectrum                     │
│                                                                │
│  Data Required    │ Resources  │ Quality │ Approach           │
│  ─────────────────│────────────│─────────│────────────────────│
│  10-100 examples  │ Minimal    │ Good    │ Few-shot (no FT)   │
│  100-1K examples  │ Low        │ Better  │ LoRA/QLoRA         │
│  1K-10K examples  │ Medium     │ Great   │ Full FT (small)    │
│  10K-100K examples│ High       │ Best    │ Full FT (large)    │
│  100K+ examples   │ Very High  │ Best+   │ Pre-train from scr │
│                                                                │
│  Memory Comparison (LLaMA-2 7B):                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Full Fine-tuning:                                    │    │
│  │  Parameters: 7B × 4 bytes (fp32) = 28 GB             │    │
│  │  Gradients:  7B × 4 bytes = 28 GB                    │    │
│  │  Optimizer:  7B × 8 bytes (Adam) = 56 GB             │    │
│  │  Total: ~112 GB (needs 2× A100 80GB)                 │    │
│  │                                                       │    │
│  │  LoRA Fine-tuning (rank=16):                          │    │
│  │  Base model: 7B × 2 bytes (bf16) = 14 GB (frozen)    │    │
│  │  LoRA params: ~20M × 4 bytes = 80 MB                 │    │
│  │  LoRA grads:  ~20M × 4 bytes = 80 MB                 │    │
│  │  LoRA optim:  ~40M × 8 bytes = 320 MB                │    │
│  │  Total: ~15 GB (fits on 1× A100 40GB)                │    │
│  │                                                       │    │
│  │  QLoRA Fine-tuning (rank=16, 4-bit):                  │    │
│  │  Base model: 7B × 0.5 bytes = 3.5 GB (4-bit, frozen) │    │
│  │  LoRA params: ~20M × 2 bytes = 40 MB                 │    │
│  │  LoRA grads:  ~20M × 2 bytes = 40 MB                 │    │
│  │  LoRA optim:  ~40M × 4 bytes = 160 MB                │    │
│  │  Total: ~4 GB (fits on 1× consumer GPU)              │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

📌 **Real Data**: PEFT (github.com/huggingface/peft) is Hugging Face's official library for parameter-efficient fine-tuning, supporting LoRA, AdaLoRA, Prefix Tuning, and more. TRL (github.com/huggingface/trl) provides the training loop for RLHF/DPO. Unsloth (github.com/unslothai/unsloth) achieves 2-5x faster LoRA/QLoRA fine-tuning with 70% less memory.

### 13.1.2 Full Fine-tuning vs PEFT Decision Matrix

```
┌──────────────────────────────────────────────────────────────┐
│              Decision Matrix: Full FT vs PEFT                  │
│                                                                │
│  Factor                │ Full FT    │ LoRA/QLoRA  │ Adapter  │
│  ──────────────────────│────────────│─────────────│──────────│
│  GPU Memory Required   │ Very High  │ Low         │ Low      │
│  Training Speed        │ Slow       │ Fast        │ Fast     │
│  Catastrophic Forgetting│ High Risk │ Low Risk    │ Low Risk │
│  Domain Adaptation     │ Best       │ Good        │ Good     │
│  Task-Specific         │ Best       │ Good        │ Good     │
│  Multiple Tasks        │ Impractical│ Easy (swap) │ Easy     │
│  Model Size Limit      │ ≤70B       │ ≤405B       │ ≤405B    │
│  Data Requirement      │ High       │ Low-Medium  │ Low      │
│  Inference Overhead    │ None       │ None*       │ 5-10%    │
│  Hyperparameter Tuning │ Complex    │ Simple      │ Moderate │
│                                                                │
│  * LoRA weights can be merged into base model at no cost      │
└──────────────────────────────────────────────────────────────┘
```

### 13.1.3 Resource Requirements by Model Size

| Model Size | Full FT (BF16) | LoRA (rank=16) | QLoRA (4-bit) | 1x RTX 4090 |
|-----------|----------------|----------------|---------------|-------------|
| 7B | 2× A100 80GB | 1× A100 40GB | 1× RTX 4090 | ✅ |
| 13B | 4× A100 80GB | 1× A100 80GB | 1× RTX 4090 | ✅ (tight) |
| 70B | 16× A100 80GB | 4× A100 80GB | 2× A100 80GB | ❌ |
| 405B | 128× A100 80GB | 16× A100 80GB | 8× A100 80GB | ❌ |

---

## 13.2 LoRA/QLoRA Architecture Design

### 13.2.1 LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2022) freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into each layer:

```
┌──────────────────────────────────────────────────────────────┐
│                    LoRA Architecture                            │
│                                                                │
│  Original Layer:                                              │
│  h = Wx     (W is d×d, frozen)                                │
│                                                                │
│  LoRA Layer:                                                  │
│  h = Wx + BAx                                                  │
│       │   │ │                                                  │
│       │   │ └─ B: d×r matrix (random init)                    │
│       │   └─── A: r×d matrix (random init)                    │
│       └─────── Frozen original weights                        │
│                                                                │
│  Where r << d (typical: r=8, 16, 32, 64)                      │
│                                                                │
│  Trainable parameters: 2 × d × r per layer                    │
│  For d=4096, r=16: 2 × 4096 × 16 = 131K params per layer    │
│  For 32 layers: 32 × 131K = 4.2M total LoRA params           │
│  (vs 7B base model = 0.06% of parameters)                     │
│                                                                │
│  At inference: merge W_new = W + BA (no overhead)             │
└──────────────────────────────────────────────────────────────┘
```

### 13.2.2 QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) combines 4-bit quantization with LoRA:

```
┌──────────────────────────────────────────────────────────────┐
│                    QLoRA Architecture                           │
│                                                                │
│  Step 1: Quantize base model to 4-bit (NF4 format)           │
│  ┌──────────────────────────────────────────────┐            │
│  │  Base Model (4-bit NF4):                      │            │
│  │  W_4bit = quantize(W_fp32)                    │            │
│  │  Memory: 7B × 0.5 bytes = 3.5 GB             │            │
│  └──────────────────────────────────────────────┘            │
│                                                                │
│  Step 2: Add LoRA adapters in FP16/BF16                      │
│  ┌──────────────────────────────────────────────┐            │
│  │  LoRA Adapters (FP16):                        │            │
│  │  A: d×r = 4096×16 = 131K params              │            │
│  │  B: r×d = 16×4096 = 131K params              │            │
│  │  Memory: 262K × 2 bytes = 524 KB per layer   │            │
│  └──────────────────────────────────────────────┘            │
│                                                                │
│  Step 3: Forward pass                                         │
│  h = dequant(W_4bit) × x + B × A × x                         │
│      ─────────────────   ────────────                         │
│      Computed in FP16     Computed in FP16                     │
│                                                                │
│  Double Quantization:                                         │
│  Quantize the quantization constants too                      │
│  Saves ~0.4 GB for 7B model                                   │
└──────────────────────────────────────────────────────────────┘
```

📌 **Real Data**: QLoRA enables fine-tuning a 65B parameter model on a single 48GB GPU while maintaining full 16-bit fine-tuning task performance. The original QLoRA paper fine-tuned a 65B model on a single 48GB A6000 GPU, achieving results competitive with full 16-bit fine-tuning (Dettmers et al., 2023).

### 13.2.3 LoRA Configuration Guide

| Target Modules | What It Affects | Typical Choice |
|---------------|-----------------|----------------|
| `q_proj, v_proj` | Attention only | Quick experiments |
| `q_proj, k_proj, v_proj, o_proj` | Full attention | Good default |
| `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Attention + FFN | Best quality |
| All linear layers | Everything possible | Maximum flexibility |

| Rank (r) | Quality | Memory | Speed | When to Use |
|----------|---------|--------|-------|-------------|
| 4 | Good | Lowest | Fastest | Quick experiments |
| 8 | Better | Low | Fast | Default choice |
| 16 | Very Good | Medium | Medium | Production quality |
| 32 | Excellent | Higher | Slower | High-quality requirements |
| 64 | Best | High | Slowest | Maximum quality (diminishing returns) |

| Alpha (α) | Effect | Typical Setting |
|-----------|--------|-----------------|
| α = r | Balanced | α=r (e.g., 16) |
| α = 2r | Stronger adaptation | When base model is very different from target |
| α = r/2 | Weaker adaptation | When base model is close to target |

---

## 13.3 Instruction Tuning Pipeline

### 13.3.1 End-to-End Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│              Instruction Tuning Pipeline                         │
│                                                                │
│  1. Data Preparation                                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Source data → Format → Filter → Split → Tokenize     │    │
│  │                                                        │    │
│  │  Format: {"instruction": "...", "input": "...",       │    │
│  │           "output": "..."}                             │    │
│  │  Filter: Remove duplicates, long/short, toxic         │    │
│  │  Split: 90% train, 5% val, 5% test                   │    │
│  └──────────────────────────────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  2. Training Configuration                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Model: Base model + LoRA config                      │    │
│  │  Optimizer: AdamW (8-bit via bitsandbytes)            │    │
│  │  Scheduler: Cosine with warmup                        │    │
│  │  Batch size: Effective = micro_batch × grad_accum × GPUs│  │
│  │  Learning rate: 1e-4 to 3e-4 (LoRA)                  │    │
│  └──────────────────────────────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  3. Training Loop (SFTTrainer from TRL)                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  for epoch in epochs:                                 │    │
│  │      for batch in dataloader:                         │    │
│  │          loss = model(batch)                          │    │
│  │          loss.backward()                              │    │
│  │          optimizer.step()                             │    │
│  │      evaluate on validation set                       │    │
│  │      save best checkpoint                             │    │
│  └──────────────────────────────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  4. Evaluation                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Automatic: MMLU, HumanEval, MT-Bench                 │    │
│  │  Human eval: Quality, safety, helpfulness             │    │
│  │  A/B testing: Compare with base model                 │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 13.3.2 Training Configuration Example

```python
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 4*8*4 = 128
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",
    gradient_checkpointing=True,
    optim="adamw_torch_8bit",
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=2048,
    dataset_text_field="text",
)

trainer.train()
```

### 13.3.3 Hyperparameter Guide

| Hyperparameter | Range | Default | Notes |
|---------------|-------|---------|-------|
| Learning Rate | 1e-5 to 5e-4 | 2e-4 | Higher for LoRA, lower for full FT |
| Batch Size | 8-256 | 64 | Larger = more stable, needs more memory |
| Epochs | 1-5 | 3 | More data = fewer epochs needed |
| Warmup Ratio | 0.01-0.1 | 0.03 | 3% of total steps |
| Weight Decay | 0-0.1 | 0.01 | Regularization |
| LoRA Rank | 4-64 | 16 | Higher = more capacity |
| LoRA Alpha | 8-128 | 32 | Typically 2× rank |
| LoRA Dropout | 0-0.1 | 0.05 | Regularization |

---

## 13.4 RLHF/DPO Architecture

### 13.4.1 RLHF Pipeline

Reinforcement Learning from Human Feedback aligns models with human preferences:

```
┌──────────────────────────────────────────────────────────────┐
│                    RLHF Pipeline                               │
│                                                                │
│  Stage 1: Supervised Fine-Tuning (SFT)                        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Base model + instruction data → SFT model            │    │
│  └──────────────────────────────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  Stage 2: Reward Model Training                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Collect human preference data:                        │    │
│  │  Prompt → Generate 2 responses → Human ranks (A > B)  │    │
│  │                                                        │    │
│  │  Train reward model:                                    │    │
│  │  Loss = -log(sigmoid(r(A) - r(B)))                     │    │
│  │  Reward model learns to score response quality         │    │
│  └──────────────────────────────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  Stage 3: PPO Optimization                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  for each prompt:                                      │    │
│  │      response = SFT_model(prompt)                      │    │
│  │      reward = Reward_model(prompt, response)           │    │
│  │      KL_penalty = KL(SFT_model || current_model)      │    │
│  │      loss = -(reward - β × KL_penalty)                 │    │
│  │      update current_model via PPO                      │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 13.4.2 DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) eliminates the reward model by directly optimizing on preference data:

```
┌──────────────────────────────────────────────────────────────┐
│              RLHF vs DPO Comparison                             │
│                                                                │
│  RLHF (3 stages):                                             │
│  SFT → Reward Model → PPO                                     │
│  - Complex pipeline                                           │
│  - Requires reward model training                             │
│  - PPO is unstable and sensitive to hyperparameters           │
│  - Needs significant compute for PPO                          │
│                                                                │
│  DPO (2 stages):                                              │
│  SFT → DPO                                                    │
│  - Simple pipeline                                            │
│  - No reward model needed                                     │
│  - Stable training (just cross-entropy loss)                  │
│  - Lower compute requirements                                 │
│                                                                │
│  DPO Loss:                                                    │
│  L = -log σ(β × (log π(y_w|x)/π_ref(y_w|x)                  │
│               - log π(y_l|x)/π_ref(y_l|x)))                   │
│                                                                │
│  Where:                                                       │
│  y_w = preferred response (winner)                            │
│  y_l = rejected response (loser)                              │
│  π_ref = reference model (SFT model)                          │
│  β = temperature parameter (typically 0.1-0.5)                │
└──────────────────────────────────────────────────────────────┘
```

📌 **Real Data**: DPO reduces alignment training compute by 4-10x compared to RLHF while achieving comparable or better results on human preference benchmarks. Meta used DPO for aligning Llama 3 models (Meta AI, 2024).

### 13.4.3 DPO Training Example

```python
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# Prepare preference dataset
# Each example has: prompt, chosen (preferred), rejected
preference_data = {
    "prompt": ["What is 2+2?", "Explain gravity."],
    "chosen": ["4", "Gravity is the force that attracts objects..."],
    "rejected": ["5", "I don't know.",]
}
dataset = Dataset.from_dict(preference_data)

# DPO Config
dpo_config = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=4,
    learning_rate=5e-7,
    beta=0.1,  # KL penalty coefficient
    loss_type="sigmoid",  # Standard DPO loss
    num_train_epochs=1,
    gradient_accumulation_steps=4,
    bf16=True,
    logging_steps=10,
    save_steps=100,
)

# DPO Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # SFT model as reference
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=dpo_config,
)

trainer.train()
```

### 13.4.4 RLHF vs DPO vs KTO

| Method | Data Required | Compute | Stability | Quality | Best For |
|--------|--------------|---------|-----------|---------|----------|
| **RLHF** | Preference pairs | Very High | Low | Highest | Maximum alignment quality |
| **DPO** | Preference pairs | Low | High | Very Good | Most use cases |
| **KTO** | Binary feedback (good/bad) | Low | High | Good | When preference pairs unavailable |
| **IPO** | Preference pairs | Low | High | Good | Avoiding over-optimization |
| **ORPO** | Preference pairs | Low | High | Good | Single-stage alignment |

---

## 13.5 Fine-tuning Data Management

### 13.5.1 Data Quality is Everything

📌 **Real Data**: Research consistently shows that data quality matters more than quantity. A study by Microsoft found that 10K high-quality examples outperform 100K low-quality examples for instruction tuning (Li et al., 2023).

### 13.5.2 Data Preparation Checklist

```
┌──────────────────────────────────────────────────────────────┐
│              Data Quality Checklist                             │
│                                                                │
│  ☐ Deduplication                                               │
│    - Exact duplicates                                          │
│    - Near-duplicates (similarity > 0.95)                       │
│    - Template duplicates (same structure, different entities)  │
│                                                                │
│  ☐ Filtering                                                   │
│    - Too short (< 10 tokens)                                   │
│    - Too long (> max_seq_length)                               │
│    - Toxic/harmful content                                     │
│    - PII (names, emails, phone numbers)                        │
│    - Low quality (gibberish, random characters)                │
│                                                                │
│  ☐ Formatting                                                  │
│    - Consistent instruction/input/output structure              │
│    - Proper escaping and encoding                              │
│    - Correct tokenization                                      │
│                                                                │
│  ☐ Balance                                                     │
│    - Diverse instruction types                                 │
│    - Balanced difficulty levels                                │
│    - Representative of target use cases                        │
│                                                                │
│  ☐ Validation                                                  │
│    - Human review of random samples                            │
│    - Check for label errors                                    │
│    - Verify answer quality                                    │
└──────────────────────────────────────────────────────────────┘
```

### 13.5.3 Data Augmentation Techniques

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| **Self-Instruct** | Use LLM to generate new instruction data | Expanding small datasets |
| **Paraphrasing** | Rephrase existing instructions | Increasing diversity |
| **Back-translation** | Translate to another language and back | Multi-lingual robustness |
| **Chain-of-thought** | Add reasoning steps to answers | Improving reasoning quality |
| **Rejection sampling** | Generate multiple answers, keep the best | Improving answer quality |

---

## 13.6 Evaluation & Monitoring

### 13.6.1 Evaluation Metrics

| Metric | Type | What It Measures |
|--------|------|-----------------|
| **Perplexity** | Automatic | Language modeling quality (lower = better) |
| **MMLU** | Benchmark | Knowledge across 57 subjects |
| **HumanEval** | Benchmark | Code generation capability |
| **MT-Bench** | Benchmark | Multi-turn conversation quality |
| **AlpacaEval** | Benchmark | Instruction following quality |
| **Win Rate** | Human eval | Head-to-head comparison with reference |
| **Toxicity** | Safety | Harmful content generation rate |

### 13.6.2 Monitoring During Training

```
┌──────────────────────────────────────────────────────────────┐
│              Training Monitoring Dashboard                      │
│                                                                │
│  Loss Curves:                                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Training Loss: Should decrease steadily              │    │
│  │  Validation Loss: Should decrease, then plateau       │    │
│  │  ⚠️ If val loss increases: overfitting!               │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Gradient Metrics:                                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Gradient norm: Should be stable (not exploding)      │    │
│  │  Learning rate: Follow warmup + cosine schedule       │    │
│  │  GPU memory: Should be stable (no OOM)                │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Quality Metrics (every N steps):                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Generate samples from validation prompts             │    │
│  │  Check for: hallucination, format compliance, quality │    │
│  │  Compare with base model outputs                      │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 13.6.3 Common Fine-tuning Failures

| Failure | Symptom | Cause | Fix |
|---------|---------|-------|-----|
| **Catastrophic Forgetting** | Model loses general knowledge | Too many epochs, too high LR | Lower LR, fewer epochs, use LoRA |
| **Overfitting** | Train loss ↓, val loss ↑ | Too much training, too little data | Early stopping, regularization, more data |
| **Reward Hacking** | High reward, bad outputs | Reward model exploited | Better reward model, KL penalty |
| **Mode Collapse** | All outputs similar | Policy too deterministic | Increase temperature, diverse data |
| **Format Degradation** | Model ignores instruction format | Data format inconsistency | Standardize data format |

---

## 💡 Case Study: How Healthcare Companies Fine-tune for Clinical NLP

### Background

A healthcare AI company needed to build a clinical NLP system that could extract structured information from medical records. The task: identify diagnoses, medications, procedures, and lab results from unstructured clinical notes.

### Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Base Model** | Llama 3 8B | Good medical knowledge, open-weight |
| **Fine-tuning Method** | QLoRA (4-bit, rank=16) | Limited GPU budget, need to iterate fast |
| **Data Size** | 50K annotated clinical notes | High quality, expert-annotated |
| **Target Modules** | All attention + FFN layers | Maximum task adaptation |
| **Alignment** | DPO with clinician preferences | Ensure medical accuracy |

### Data Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│              Clinical NLP Data Pipeline                         │
│                                                                │
│  Raw Clinical Notes (100K+)                                    │
│       │                                                        │
│       ▼                                                        │
│  De-identification (HIPAA compliance)                          │
│  - Remove names, dates, locations                              │
│  - Replace with synthetic placeholders                         │
│       │                                                        │
│       ▼                                                        │
│  Expert Annotation (50K notes)                                 │
│  - 3 board-certified physicians                                │
│  - Inter-annotator agreement: κ = 0.89                         │
│       │                                                        │
│       ▼                                                        │
│  Format as instruction-following                               │
│  {"instruction": "Extract diagnoses from this clinical note",  │
│   "input": "Patient presents with chest pain...",              │
│   "output": [{"diagnosis": "Acute MI", "confidence": 0.95}]}  │
│       │                                                        │
│       ▼                                                        │
│  Quality Control                                               │
│  - Auto-check: valid JSON, no empty fields                     │
│  - Manual review: 10% random sample                            │
│  - Adversarial testing: edge cases, abbreviations              │
└──────────────────────────────────────────────────────────────┘
```

### Results

| Metric | Base Llama 3 8B | Fine-tuned Model | Improvement |
|--------|-----------------|------------------|-------------|
| F1 (Diagnosis extraction) | 0.42 | 0.91 | +117% |
| F1 (Medication extraction) | 0.38 | 0.89 | +134% |
| F1 (Procedure extraction) | 0.35 | 0.87 | +149% |
| Hallucination rate | 23% | 2% | -91% |
| HIPAA compliance | 100% | 100% | Maintained |

### Key Takeaways

1. **Domain-specific data quality is paramount**: Medical annotations must be expert-level
2. **QLoRA enables rapid iteration**: 3x faster training, 70% less memory than full FT
3. **DPO alignment ensures safety**: Clinician preferences prevent dangerous outputs
4. **Evaluation must be domain-appropriate**: Standard NLP metrics don't capture medical accuracy

---

## ⚠️ War Story: The Fine-tuned Model That Got Worse Than Base

### The Setup

A fintech company fine-tuned Llama 3 70B on 100K customer support conversations to build a financial advisor chatbot. They used LoRA (rank=32) for 3 epochs on 8× A100 GPUs.

### The Problem

After fine-tuning, automated evaluation showed improved scores:
- Instruction following: 78% → 89%
- Response relevance: 72% → 85%
- Financial accuracy: 65% → 71%

But when they deployed to production, customer complaints increased:
- "The bot gives generic financial advice, not personalized"
- "It used to answer product-specific questions, now it can't"
- "It recommends products we don't even offer"

### Root Cause Analysis

```
┌──────────────────────────────────────────────────────────────┐
│                    Failure Analysis                            │
│                                                                │
│  1. DATA ISSUE:                                               │
│  - Training data was from a DIFFERENT product line             │
│  - Customer conversations were about Product A                 │
│  - Production users asked about Product B (newer)              │
│  - Model learned Product A's features, forgot Product B        │
│                                                                │
│  2. CATASTROPHIC FORGETTING:                                   │
│  - Base model knew about both products (from pre-training)     │
│  - Fine-tuning on Product A data OVERWR knowledge about B      │
│  - No regularization to preserve general knowledge             │
│                                                                │
│  3. EVALUATION GAP:                                            │
│  - Automated eval used same distribution as training data       │
│  - Production data had different distribution                  │
│  - Eval showed improvement, production showed degradation      │
│                                                                │
│  4. DATA LEAKAGE:                                              │
│  - Training data contained incorrect financial advice          │
│  - Model learned to give WRONG advice confidently              │
│  - Higher "confidence" scores masked lower accuracy            │
└──────────────────────────────────────────────────────────────┘
```

### The Fix

1. **Include ALL product data**: Retrained with conversations from both Product A and B
2. **Add general knowledge regularization**: Continued pre-training on financial knowledge base
3. **Evaluation on held-out product**: Used Product B data for evaluation (not just Product A)
4. **Human review pipeline**: All AI responses go through compliance review before deployment
5. **LoRA rank reduction**: Reduced from 32 to 8 to limit adaptation capacity

### Impact

- **2 weeks of wasted compute**: ~$50K in GPU costs
- **3 week delay** in product launch
- **Customer trust damage**: 15% of beta users switched to competitors

### Key Takeaways

1. **Evaluate on the target distribution**, not just the training distribution
2. **Include diverse data**: Cover all products, use cases, and edge cases
3. **Regularize to prevent forgetting**: Use LoRA with lower rank, add general knowledge data
4. **Human review is essential**: Automated metrics can be misleading
5. **Start small**: Fine-tune on a subset, evaluate thoroughly, then scale up

---

## 📝 When to Use / When Not to Use

### Fine-tuning Method Selection

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| Quick prototype, limited data | LoRA (rank=8) | Fast iteration, low cost |
| Production quality, moderate data | LoRA (rank=16-32) | Good quality-cost balance |
| Maximum quality, unlimited budget | Full fine-tuning | Best possible adaptation |
| Consumer GPU (24GB) | QLoRA (4-bit) | Only option for large models |
| Alignment with human preferences | DPO | Stable, efficient, good results |
| Alignment with binary feedback | KTO | When preference pairs unavailable |
| Multiple tasks | LoRA (swap adapters) | Same base, different adapters |
| Real-time adaptation | LoRA | Fast adapter swapping |

### When to Fine-tune vs When to Use RAG

| Use Case | Fine-tune? | RAG? | Why |
|----------|-----------|------|-----|
| Format/style adaptation | ✅ | ❌ | Learn the pattern, not facts |
| Domain vocabulary | ✅ | ⚠️ | Fine-tune for terminology, RAG for facts |
| Knowledge updates | ❌ | ✅ | RAG updates in real-time |
| Private knowledge | ⚠️ | ✅ | RAG keeps data in your DB |
| Task-specific behavior | ✅ | ❌ | Fine-tune for the behavior |
| Multi-domain knowledge | ❌ | ✅ | RAG scales to many domains |
| Reasoning improvement | ✅ | ❌ | Fine-tune with CoT data |
| Citation requirements | ❌ | ✅ | RAG provides source documents |

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| **PEFT vs Full FT** | PEFT (LoRA/QLoRA) achieves 90-95% of full FT quality at 1-10% of the cost |
| **LoRA** | Low-rank adaptation; rank=16 is a good default; mergeable at inference |
| **QLoRA** | 4-bit quantized base + LoRA adapters; enables fine-tuning 70B on single GPU |
| **Instruction Tuning** | Data quality > data quantity; 10K high-quality examples > 100K low-quality |
| **RLHF** | 3-stage pipeline: SFT → Reward Model → PPO; expensive but effective |
| **DPO** | 2-stage pipeline: SFT → DPO; 4-10x cheaper than RLHF, comparable quality |
| **Data Management** | Dedup, filter, format, balance, validate; human review essential |
| **Evaluation** | Use multiple metrics; evaluate on target distribution, not training distribution |
| **Common Failures** | Catastrophic forgetting, overfitting, reward hacking, mode collapse |

---

## Discussion Questions

1. **Resource Allocation**: You have a budget of $10,000 for fine-tuning. You can either:
   - Option A: Full fine-tune a 7B model on 50K examples
   - Option B: LoRA fine-tune a 70B model on 10K examples
   
   Which would you choose and why? What factors influence your decision?

2. **Alignment Strategy**: A company wants to align a customer service chatbot with their brand voice. They have 5,000 pairs of preferred/rejected responses from customer service agents. Should they use RLHF, DPO, or KTO? What are the trade-offs?

3. **Data Strategy**: You're building a medical QA system. You have 1,000 expert-annotated examples but need more data. What augmentation strategies would you use? How would you validate the quality of augmented data?

4. **Evaluation Design**: Your fine-tuned model scores 95% on MMLU but users complain it's "less helpful" than the base model. How would you investigate and address this discrepancy?

5. **Architecture Decision**: You need to serve a model that handles both general questions (knowledge from pre-training) and company-specific questions (needs fine-tuning). Should you fine-tune the entire model or use RAG? What about a hybrid approach?

---

## Exercises

### Exercise 1: LoRA Fine-tuning

Using PEFT + TRL:
1. Fine-tune Llama 3 8B on a custom dataset (Alpaca format) using LoRA
2. Experiment with different ranks (4, 8, 16, 32) and compare:
   - Training time
   - GPU memory usage
   - Validation loss
   - Sample quality (human evaluation)
3. Create a report recommending the optimal rank for your use case

### Exercise 2: DPO Alignment

Using TRL:
1. Create a preference dataset (100 pairs of chosen/rejected responses)
2. Fine-tune a model using DPO with different β values (0.05, 0.1, 0.2, 0.5)
3. Evaluate using win rate against the SFT model
4. Analyze the effect of β on response quality and diversity
5. Write recommendations for β selection

### Exercise 3: Failure Mode Investigation

Given a pre-trained model and fine-tuned version:
1. Run evaluation on 5 different benchmarks
2. Identify areas where the fine-tuned model performs WORSE
3. Analyze training data for potential causes
4. Propose and implement fixes (data augmentation, regularization, etc.)
5. Document findings in a failure analysis report

---

## References

1. Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." https://arxiv.org/abs/2106.09685
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS. https://arxiv.org/abs/2305.14314
3. Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS. https://arxiv.org/abs/2305.18290
4. Touvron, H., et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models." https://arxiv.org/abs/2307.09288
5. Meta AI. (2024). "The Llama 3 Herd of Models." https://arxiv.org/abs/2407.21783
6. Hugging Face PEFT Documentation. https://huggingface.co/docs/peft
7. Hugging Face TRL Documentation. https://huggingface.co/docs/trl
8. Unsloth Documentation. https://github.com/unslothai/unsloth
9. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." NeurIPS. https://arxiv.org/abs/2203.02155
10. Christiano, P., et al. (2017). "Deep Reinforcement Learning from Human Preferences." NeurIPS. https://arxiv.org/abs/1706.03741

---

*Next Chapter: [Chapter 14 - Production Deployment](chapter-14.md) →*
