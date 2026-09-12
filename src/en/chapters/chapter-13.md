# Chapter 13: Model Fine-tuning Architecture

> 🟡 Intermediate → 🔴 Advanced | 25-30 min read | Part 4: Large Model Architecture

---

## Table of Contents

- [13.1 Full vs Parameter-Efficient Fine-tuning](#131-full-vs-parameter-efficient-fine-tuning)
- [13.2 LoRA/QLoRA Architecture Design](#132-loraqlora-architecture-design)
- [13.3 Instruction Tuning Pipeline](#133-instruction-tuning-pipeline)
- [13.4 RLHF/DPO Architecture](#134-rlhfdpo-architecture)
- [13.5 Fine-tuning Data Management](#135-fine-tuning-data-management)
- [💡 Case: Efficient Fine-tuning with Unsloth](#-case-efficient-fine-tuning-with-unsloth)
- [Summary](#summary)
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

### 13.1.2 When to Use Each Approach

📌 **Key Concept**: Full fine-tuning updates ALL model parameters. PEFT methods only update a small subset of parameters, keeping the base model frozen.

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
│                                                                │
│  Recommendations:                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Use Full FT when:                                    │    │
│  │  - You have >100K high-quality examples               │    │
│  │  - Task is very different from pre-training           │    │
│  │  - You have sufficient compute budget                 │    │
│  │  - Maximum quality is critical                        │    │
│  │                                                       │    │
│  │  Use LoRA/QLoRA when:                                 │    │
│  │  - You have limited compute                          │    │
│  │  - You want to fine-tune for multiple tasks           │    │
│  │  - Data is limited (<50K examples)                   │    │
│  │  - You want to preserve base model capabilities       │    │
│  │  - You need rapid iteration                          │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 13.2 LoRA/QLoRA Architecture Design

### 13.2.1 LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2022) decomposes weight updates into low-rank matrices:

📌 **Key Concept**: Instead of updating a weight matrix W (d × d), LoRA learns two small matrices A (d × r) and B (r × d), where r << d. The update is: W' = W + BA.

```
┌──────────────────────────────────────────────────────────────┐
│                    LoRA Architecture                            │
│                                                                │
│  Original Linear Layer:                                        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  y = Wx    where W ∈ ℝ^{d_out × d_in}               │    │
│  │                                                       │    │
│  │  For a 4096×4096 layer:                               │    │
│  │  Parameters: 4096 × 4096 = 16.8M                     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  LoRA-Adapted Layer:                                            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                                                       │    │
│  │  x ──────────────────────┐                           │    │
│  │  │                       │                           │    │
│  │  ▼                       ▼                           │    │
│  │  ┌──────┐           ┌─────────┐                     │    │
│  │  │  W   │ (frozen)  │ A (d×r) │ (trainable)         │    │
│  │  │      │           └────┬────┘                     │    │
│  │  └──┬───┘                │                           │    │
│  │     │                    ▼                           │    │
│  │     │               ┌─────────┐                     │    │
│  │     │               │ B (r×d) │ (trainable)         │    │
│  │     │               └────┬────┘                     │    │
│  │     │                    │                           │    │
│  │     └────────┬───────────┘                           │    │
│  │              │                                       │    │
│  │              ▼                                       │    │
│  │         y = Wx + BAx                                │    │
│  │                                                       │    │
│  │  For rank r=16:                                       │    │
│  │  LoRA params: 4096 × 16 + 16 × 4096 = 131K          │    │
│  │  Ratio: 131K / 16.8M = 0.78%                         │    │
│  │  Memory savings: 128x                                 │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

#### LoRA Implementation

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """Low-Rank Adaptation for a linear layer."""

    def __init__(self, original_linear, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        d_out, d_in = original_linear.weight.shape

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(d_in, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # Original forward (frozen)
        original_output = self.original_linear(x)

        # LoRA forward
        lora_output = self.lora_dropout(x)
        lora_output = lora_output @ self.lora_A @ self.lora_B * self.scaling

        return original_output + lora_output

    def merge_weights(self):
        """Merge LoRA weights into original for inference."""
        self.original_linear.weight.data += (
            self.lora_B @ self.lora_A * self.scaling
        ).to(self.original_linear.weight.dtype)

# Apply LoRA to a model
def apply_lora(model, rank=16, target_modules=None):
    """Apply LoRA to specified modules."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    lora_params = 0
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]

                lora_layer = LoRALinear(module, rank=rank)
                setattr(parent, child_name, lora_layer)

                lora_params += sum(p.numel() for p in lora_layer.parameters()
                                   if p.requires_grad)

    print(f"LoRA parameters: {lora_params:,} ({lora_params/1e6:.2f}M)")
    return model
```

### 13.2.2 QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) combines 4-bit quantization with LoRA:

```
┌──────────────────────────────────────────────────────────────┐
│                    QLoRA Architecture                           │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  4-bit NormalFloat (NF4) Quantized Base Model         │    │
│  │  ┌────────────────────────────────────────────────┐  │    │
│  │  │  W_4bit = quantize(W_fp16, bits=4, type=NFB4)  │  │    │
│  │  │                                                  │  │    │
│  │  │  Memory: 7B × 0.5 bytes = 3.5 GB               │  │    │
│  │  └────────────────────────────────────────────────┘  │    │
│  │                                                       │    │
│  │  Double Quantization (quantize quantization const):   │    │
│  │  ┌────────────────────────────────────────────────┐  │    │
│  │  │  Block absmax values quantized to FP8          │  │    │
│  │  │  Extra memory savings: ~0.37 GB per 7B model   │  │    │
│  │  └────────────────────────────────────────────────┘  │    │
│  │                                                       │    │
│  │  Paged Optimizers (CPU offload for optimizer states): │    │
│  │  ┌────────────────────────────────────────────────┐  │    │
│  │  │  AdamW states offloaded to CPU memory           │  │    │
│  │  │  Page in/out as needed (like OS paging)         │  │    │
│  │  └────────────────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Training Flow:                                                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Input x ──▶ Dequantize W ──▶ Compute Wx (in fp16)   │    │
│  │                      │                                 │    │
│  │              ┌───────┴───────┐                        │    │
│  │              │  LoRA (fp16)  │                        │    │
│  │              │  A: d×r (fp16)│                        │    │
│  │              │  B: r×d (fp16)│                        │    │
│  │              └───────┬───────┘                        │    │
│  │                      │                                 │    │
│  │              Output = Wx + BAx (fp16)                 │    │
│  │                                                       │    │
│  │  Gradients: Only for A, B (not W)                     │    │
│  │  Optimizer: Only for A, B parameters                  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

```python
# QLoRA implementation using bitsandbytes
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model with QLoRA
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 33,554,432 || all params: 6,771,970,048 || 0.50%

# Training arguments
training_args = TrainingArguments(
    output_dir="./qlora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    optim="paged_adamw_32bit",  # Paged optimizer for QLoRA
    gradient_checkpointing=True,
    report_to="tensorboard",
)

# Trainer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=2048,
    packing=True,
)

# Train
trainer.train()

# Save LoRA adapter
model.save_pretrained("./qlora-adapter")

# Merge for inference
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
)
merged_model = PeftModel.from_pretrained(base_model, "./qlora-adapter")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

### 13.2.3 LoRA Variants

```
┌──────────────────────────────────────────────────────────────┐
│              LoRA Variants Comparison                           │
│                                                                │
│  Variant    │ Key Innovation           │ Best For             │
│  ───────────│──────────────────────────│──────────────────────│
│  LoRA       │ Low-rank decomposition   │ General fine-tuning  │
│  QLoRA      │ 4-bit base + LoRA        │ Memory-constrained   │
│  DoRA       │ Decomposed rank adapt.   │ Higher quality       │
│  LoRA+      │ Different LR for A, B    │ Faster convergence   │
│  rsLoRA     │ Rank-stabilized scaling  │ Large rank stability │
│  AdaLoRA    │ Adaptive rank allocation  │ Mixed importance     │
│  GaLore     │ Gradient low-rank proj.  │ Full-rank training   │
│  LoRA-GA    │ Gradient-aware init      │ Better initialization│
│                                                                │
│  LoRA Rank Selection Guide:                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Task Type           │ Recommended Rank               │    │
│  │  ────────────────────│────────────────────────────────│    │
│  │  Classification      │ 4-8                            │    │
│  │  Instruction Tuning  │ 8-32                           │    │
│  │  Domain Adaptation   │ 16-64                          │    │
│  │  Complex Reasoning   │ 32-128                         │    │
│  │  Code Generation     │ 16-64                          │    │
│  │  Multilingual        │ 32-64                          │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Memory vs Quality Trade-off:                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Rank 4:  0.1% params, 85% quality                  │    │
│  │  Rank 8:  0.2% params, 90% quality                  │    │
│  │  Rank 16: 0.5% params, 95% quality                  │    │
│  │  Rank 32: 1.0% params, 98% quality                  │    │
│  │  Rank 64: 2.0% params, 99% quality                  │    │
│  │  Rank 128: 4.0% params, 99.5% quality               │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 13.3 Instruction Tuning Pipeline

### 13.3.1 Data Format and Preparation

Instruction tuning teaches models to follow instructions and produce desired outputs:

```
┌──────────────────────────────────────────────────────────────┐
│              Instruction Tuning Data Format                     │
│                                                                │
│  Alpaca Format (3-column):                                     │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  {                                                    │    │
│  │    "instruction": "Summarize the following article", │    │
│  │    "input": "Long article text...",                  │    │
│  │    "output": "This article discusses..."            │    │
│  │  }                                                    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  ShareGPT Format (multi-turn):                                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  {                                                    │    │
│  │    "conversations": [                                 │    │
│  │      {"from": "human", "value": "What is ML?"},     │    │
│  │      {"from": "gpt", "value": "ML is..."},          │    │
│  │      {"from": "human", "value": "Explain more"},    │    │
│  │      {"from": "gpt", "value": "Specifically..."}    │    │
│  │    ]                                                  │    │
│  │  }                                                    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  ChatML Format (OpenAI compatible):                            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  <|im_start|>system                                  │    │
│  │  You are a helpful assistant.<|im_end|>             │    │
│  │  <|im_start|>user                                    │    │
│  │  What is 2+2?<|im_end|>                             │    │
│  │  <|im_start|>assistant                              │    │
│  │  4<|im_end|>                                         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Quality Guidelines:                                            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  ✅ Diverse instructions (avoid repetition)           │    │
│  │  ✅ Detailed, comprehensive outputs                   │    │
│  │  ✅ Consistent formatting and style                   │    │
│  │  ✅ Balanced across task types                        │    │
│  │  ❌ No hallucinated facts                             │    │
│  │  ❌ No toxic or biased content                        │    │
│  │  ❌ No overly short or lazy responses                 │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 13.3.2 Complete Instruction Tuning Pipeline

```python
# instruction_tuning_pipeline.py
"""
Complete instruction tuning pipeline with data prep, training, and evaluation.
"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

class InstructionTuningPipeline:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.setup_model()

    def setup_model(self):
        """Initialize model with QLoRA configuration."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = prepare_model_for_kbit_training(self.model)

    def prepare_data(self, data_path):
        """Load and format instruction tuning data."""
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        formatted_data = []
        for item in raw_data:
            # Format as ChatML
            if "conversations" in item:
                text = self._format_conversations(item["conversations"])
            else:
                text = self._format_alpaca(
                    item.get("instruction", ""),
                    item.get("input", ""),
                    item.get("output", "")
                )
            formatted_data.append({"text": text})

        return Dataset.from_list(formatted_data)

    def _format_alpaca(self, instruction, input_text, output):
        """Format Alpaca-style data."""
        if input_text:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )

    def _format_conversations(self, conversations):
        """Format ShareGPT-style conversations."""
        text = ""
        for turn in conversations:
            if turn["from"] == "human":
                text += f"<|user|>\n{turn['value']}\n"
            elif turn["from"] == "gpt":
                text += f"<|assistant|>\n{turn['value']}\n"
        return text

    def train(self, dataset, rank=16):
        """Run LoRA fine-tuning."""
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        )

        model = get_peft_model(self.model, lora_config)

        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            optim="paged_adamw_32bit",
            gradient_checkpointing=True,
            max_seq_length=2048,
            packing=True,
            report_to="tensorboard",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            args=training_args,
        )

        trainer.train()
        model.save_pretrained(f"{self.output_dir}/adapter")

        return trainer

    def evaluate(self, test_data, max_samples=100):
        """Evaluate fine-tuned model on test data."""
        self.model.eval()
        results = []

        for i, item in enumerate(test_data[:max_samples]):
            prompt = self._format_alpaca(
                item["instruction"],
                item.get("input", ""),
                ""  # No output - we're generating
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            results.append({
                "instruction": item["instruction"],
                "expected": item.get("output", ""),
                "generated": generated,
            })

        return results
```

### 13.3.3 Data Quality and Curation

```
┌──────────────────────────────────────────────────────────────┐
│              Data Quality Pipeline                               │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Raw Data Sources                                     │    │
│  │  ├── Alpaca (52K)                                     │    │
│  │  ├── ShareGPT conversations                           │    │
│  │  ├── Dolly (15K)                                      │    │
│  │  ├── OpenAssistant                                    │    │
│  │  └── Custom domain data                               │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Deduplication                                         │    │
│  │  ├── Exact match dedup                                │    │
│  │  ├── Fuzzy match (MinHash/LSH)                        │    │
│  │  └── Semantic dedup (embedding similarity)            │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Quality Filtering                                     │    │
│  │  ├── Remove toxic content (classifier)                │    │
│  │  ├── Remove too-short responses (<50 tokens)          │    │
│  │  ├── Remove too-long responses (>2048 tokens)         │    │
│  │  ├── Filter by language quality score                 │    │
│  │  └── Remove duplicates across datasets                │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Balance & Augment                                     │    │
│  │  ├── Balance task categories                          │    │
│  │  ├── Add Chain-of-Thought data (20%)                  │    │
│  │  ├── Add multi-turn conversations (15%)               │    │
│  │  └── Add code/technical data (10%)                    │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Final Dataset                                         │    │
│  │  ├── ~50K-200K high-quality examples                  │    │
│  │  ├── Balanced across task types                       │    │
│  │  ├── Consistent formatting                            │    │
│  │  └── Verified quality (human spot-check)              │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 13.4 RLHF/DPO Architecture

### 13.4.1 RLHF Pipeline

Reinforcement Learning from Human Feedback (RLHF) aligns models with human preferences:

```
┌──────────────────────────────────────────────────────────────┐
│                    RLHF Pipeline                               │
│                                                                │
│  Stage 1: Supervised Fine-Tuning (SFT)                        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Base Model → SFT on human demonstrations → SFT Model │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  Stage 2: Reward Model Training                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Prompt → SFT Model → Generate K responses            │    │
│  │                                                       │    │
│  │  Human ranks responses: A > B > C > D                │    │
│  │                                                       │    │
│  │  Train Reward Model:                                  │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  L_rank = -log(σ(r(y_a) - r(y_b)))          │     │    │
│  │  │  where y_a is preferred over y_b             │     │    │
│  │  │  r(·) is the reward model                    │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  Stage 3: PPO Optimization                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  Maximize: r(y) - β × KL(π_θ || π_ref)     │     │    │
│  │  │                                               │     │    │
│  │  │  π_θ: Policy model (being optimized)         │     │    │
│  │  │  π_ref: Reference model (SFT model, frozen)  │     │    │
│  │  │  r(y): Reward model score                     │     │    │
│  │  │  β: KL penalty coefficient                   │     │    │
│  │  │                                               │     │    │
│  │  │  PPO Algorithm:                               │     │    │
│  │  │  1. Sample prompt from dataset                │     │    │
│  │  │  2. Generate response from policy π_θ        │     │    │
│  │  │  3. Score response with reward model r(y)    │     │    │
│  │  │  4. Compute KL penalty                       │     │    │
│  │  │  5. Update policy using PPO                  │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Aligned Model (RLHF-tuned)                           │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 13.4.2 DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) eliminates the need for a separate reward model:

📌 **Key Concept**: DPO directly optimizes the policy using preference pairs, avoiding the complexity of RL training with PPO.

```
┌──────────────────────────────────────────────────────────────┐
│              RLHF vs DPO Comparison                             │
│                                                                │
│  RLHF (3 stages):                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  SFT → Reward Model → PPO                            │    │
│  │                                                       │    │
│  │  Pros: More flexible, can use reward shaping          │    │
│  │  Cons: Complex, unstable, requires 4 models in memory │    │
│  │        (policy, reference, reward, value)              │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  DPO (2 stages):                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  SFT → DPO                                           │    │
│  │                                                       │    │
│  │  DPO Loss:                                            │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │  L_DPO = -log σ(β × (log π_θ(y_w|x)/       │     │    │
│  │  │                    π_ref(y_w|x) -             │     │    │
│  │  │                    log π_θ(y_l|x)/            │     │    │
│  │  │                    π_ref(y_l|x)))             │     │    │
│  │  │                                               │     │    │
│  │  │  y_w: preferred (winning) response            │     │    │
│  │  │  y_l: rejected (losing) response              │     │    │
│  │  │  π_θ: policy being trained                    │     │    │
│  │  │  π_ref: reference policy (SFT model)         │     │    │
│  │  │  β: temperature parameter                    │     │    │
│  │  └─────────────────────────────────────────────┘     │    │
│  │                                                       │    │
│  │  Pros: Simpler, more stable, only 2 models needed    │    │
│  │  Cons: Less flexible, requires preference pairs       │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Memory Requirements:                                           │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  RLHF: 4 models × model_size                        │    │
│  │  - Policy model (trainable)                          │    │
│  │  - Reference model (frozen)                          │    │
│  │  - Reward model (frozen)                             │    │
│  │  - Value model (for PPO)                             │    │
│  │                                                       │    │
│  │  DPO: 2 models × model_size                          │    │
│  │  - Policy model (trainable)                          │    │
│  │  - Reference model (frozen)                          │    │
│  │                                                       │    │
│  │  For 7B model:                                        │    │
│  │  RLHF: ~56 GB (with LoRA)                            │    │
│  │  DPO: ~28 GB (with LoRA)                             │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

```python
# DPO Training implementation
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

def train_dpo(model_name, preference_data, output_dir):
    """
    Train using DPO with preference pairs (chosen/rejected).
    """
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare preference data
    # Each item has: prompt, chosen (preferred), rejected
    dataset = Dataset.from_dict({
        "prompt": [item["prompt"] for item in preference_data],
        "chosen": [item["chosen"] for item in preference_data],
        "rejected": [item["rejected"] for item in preference_data],
    })

    # DPO configuration
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,  # Lower LR for DPO
        beta=0.1,            # KL penalty coefficient
        loss_type="sigmoid", # DPO loss variant
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        max_length=1024,
        max_prompt_length=512,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="tensorboard",
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

    return trainer

# Example preference data format
preference_data = [
    {
        "prompt": "Explain quantum computing to a 5-year-old.",
        "chosen": "Quantum computing is like a magic box that can try "
                  "many answers at the same time, instead of one by one. "
                  "It helps scientists solve really hard puzzles!",
        "rejected": "Quantum computing leverages superposition and "
                    "entanglement of qubits to perform parallel computations "
                    "across exponentially large state spaces.",
    },
    {
        "prompt": "Write a function to sort a list in Python.",
        "chosen": "```python\ndef sort_list(lst):\n    return sorted(lst)\n```\n\n"
                  "This uses Python's built-in sorted() function which implements "
                  "Timsort, an efficient O(n log n) algorithm.",
        "rejected": "```python\ndef sort_list(lst):\n    for i in range(len(lst)):\n"
                    "        for j in range(i+1, len(lst)):\n"
                    "            if lst[i] > lst[j]:\n"
                    "                lst[i], lst[j] = lst[j], lst[i]\n"
                    "    return lst\n```\nThis is O(n²) bubble sort.",
    },
]
```

### 13.4.3 Other Alignment Methods

```
┌──────────────────────────────────────────────────────────────┐
│              Alignment Methods Comparison                       │
│                                                                │
│  Method      │ Data Needed      │ Complexity │ Quality        │
│  ────────────│──────────────────│────────────│────────────────│
│  RLHF (PPO)  │ Preferences + RM │ High       │ Best (if tuned)│
│  DPO         │ Pairs (chosen/rej)│ Low       │ Good           │
│  IPO         │ Pairs            │ Low        │ Good           │
│  KTO         │ Binary (good/bad)│ Low        │ Good           │
│  ORPO        │ Pairs            │ Low        │ Good           │
│  SPIN        │ Self-play data   │ Medium     │ Good           │
│  RLAIF       │ AI preferences   │ Medium     │ Good           │
│  Constitutional│ AI feedback    │ Low        │ Good           │
│                                                                │
│  Emerging Trend (2024-2026):                                    │
│  - DPO and its variants are becoming the default              │
│  - RLHF used primarily for frontier models                    │
│  - RLAIF reducing need for human annotation                   │
│  - Multi-objective alignment becoming important               │
└──────────────────────────────────────────────────────────────┘
```

---

## 13.5 Fine-tuning Data Management

### 13.5.1 Data Collection and Annotation

```
┌──────────────────────────────────────────────────────────────┐
│              Data Management Pipeline                           │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Data Sources                                          │    │
│  │  ├── Human annotation (highest quality, highest cost) │    │
│  │  ├── Synthetic generation (GPT-4/Claude)              │    │
│  │  ├── Existing datasets (Alpaca, Dolly, etc.)          │    │
│  │  ├── Domain-specific collections                      │    │
│  │  └── User interaction logs                            │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Annotation Tools                                      │    │
│  │  ├── Label Studio (open-source)                       │    │
│  │  ├── Argilla (LLM-focused)                            │    │
│  │  ├── Scale AI / Surge AI (managed)                    │    │
│  │  └── Custom web interfaces                            │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Quality Assurance                                     │    │
│  │  ├── Inter-annotator agreement (IAA > 0.8)            │    │
│  │  ├── Automated quality checks                         │    │
│  │  ├── Human spot-checks (10% sample)                   │    │
│  │  └── A/B testing on model performance                 │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 13.5.2 Synthetic Data Generation

```python
# synthetic_data_generator.py
"""
Generate high-quality instruction tuning data using a teacher model.
"""
import json
from openai import OpenAI

class SyntheticDataGenerator:
    def __init__(self, api_key, model="gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_instructions(self, topic, num_samples=100):
        """Generate diverse instructions for a topic."""
        prompt = f"""Generate {num_samples} diverse, high-quality instructions
for a language model about the topic: {topic}.

Include these categories:
- Factual questions (30%)
- Creative tasks (20%)
- Analysis/reasoning (20%)
- How-to/instructions (15%)
- Code-related (15%)

Format as JSON array with fields: instruction, difficulty (easy/medium/hard),
category. Make instructions diverse in length and complexity."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    def generate_responses(self, instructions, model_name="Llama-2-7b-chat"):
        """Generate responses for instructions."""
        responses = []

        for inst in instructions:
            prompt = f"""### Instruction:
{inst['instruction']}

### Response:"""

            # Use local model or API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
            )

            responses.append({
                "instruction": inst["instruction"],
                "input": "",
                "output": response.choices[0].message.content,
                "difficulty": inst.get("difficulty", "medium"),
                "category": inst.get("category", "general"),
            })

        return responses

    def generate_preference_pairs(self, instructions, num_pairs=1000):
        """Generate chosen/rejected pairs for DPO training."""
        pairs = []

        for inst in instructions[:num_pairs]:
            # Generate two responses
            prompt = f"""### Instruction:
{inst['instruction']}

### Response:"""

            response_a = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.8,
            )

            response_b = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.8,
            )

            # Judge which is better
            judge_prompt = f"""Compare these two responses to the instruction:
"{inst['instruction']}"

Response A: {response_a.choices[0].message.content}
Response B: {response_b.choices[0].message.content}

Which is better? Reply with just "A" or "B"."""

            judge = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=1,
            )

            winner = judge.choices[0].message.content.strip()

            if winner == "A":
                chosen = response_a.choices[0].message.content
                rejected = response_b.choices[0].message.content
            else:
                chosen = response_b.choices[0].message.content
                rejected = response_a.choices[0].message.content

            pairs.append({
                "prompt": inst["instruction"],
                "chosen": chosen,
                "rejected": rejected,
            })

        return pairs

# Usage
generator = SyntheticDataGenerator(api_key="your-api-key")

# Generate instruction data
instructions = generator.generate_instructions("machine learning", num_samples=500)
responses = generator.generate_responses(instructions)

# Save as JSON
with open("instruction_data.json", "w") as f:
    json.dump(responses, f, indent=2)

# Generate preference pairs for DPO
pairs = generator.generate_preference_pairs(instructions, num_pairs=1000)
with open("preference_data.json", "w") as f:
    json.dump(pairs, f, indent=2)
```

### 13.5.3 Data Versioning and Management

```
┌──────────────────────────────────────────────────────────────┐
│              Data Versioning Best Practices                     │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Version Control (DVC / Git LFS)                      │    │
│  │  ├── Track data changes alongside code                │    │
│  │  ├── Tag datasets with version numbers                │    │
│  │  ├── Store metadata (size, source, quality metrics)   │    │
│  │  └── Enable reproducible experiments                  │    │
│  │                                                       │    │
│  │  Directory Structure:                                  │    │
│  │  data/                                                │    │
│  │  ├── v1.0/                                           │    │
│  │  │   ├── train.jsonl                                  │    │
│  │  │   ├── val.jsonl                                    │    │
│  │  │   ├── test.jsonl                                   │    │
│  │  │   └── metadata.json                                │    │
│  │  ├── v1.1/                                           │    │
│  │  │   ├── train.jsonl (added 10K samples)              │    │
│  │  │   └── metadata.json                                │    │
│  │  └── current -> v1.1/ (symlink)                      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  metadata.json                                         │    │
│  │  {                                                    │    │
│  │    "version": "1.1",                                  │    │
│  │    "created": "2026-01-15",                           │    │
│  │    "source": "alpaca + custom_generation",            │    │
│  │    "num_samples": 62000,                              │    │
│  │    "categories": {                                    │    │
│  │      "factual": 0.30,                                 │    │
│  │      "creative": 0.20,                                │    │
│  │      "reasoning": 0.20,                               │    │
│  │      "howto": 0.15,                                   │    │
│  │      "code": 0.15                                     │    │
│  │    },                                                 │    │
│  │    "quality_metrics": {                               │    │
│  │      "avg_response_length": 342,                      │    │
│  │      "avg_quality_score": 4.2,                        │    │
│  │      "dedup_ratio": 0.95                              │    │
│  │    },                                                 │    │
│  │    "changelog": [                                     │    │
│  │      "v1.0: Initial 52K Alpaca subset",              │    │
│  │      "v1.1: Added 10K custom ML examples"            │    │
│  │    ]                                                  │    │
│  │  }                                                    │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 💡 Case: Efficient Fine-tuning with Unsloth

### Business Context

A startup wants to fine-tune a 7B parameter model for customer support in Chinese. They have:
- 1 GPU (RTX 4090, 24GB VRAM)
- 5,000 labeled customer support conversations
- Budget: $500 for compute

### Why Unsloth?

Unsloth provides 2-5x faster fine-tuning with 60% less memory through custom CUDA kernels:

```
┌──────────────────────────────────────────────────────────────┐
│              Unsloth Performance Comparison                     │
│                                                                │
│  Framework      │ Time (7B, 3 epochs) │ Memory  │ Quality    │
│  ───────────────│─────────────────────│─────────│────────────│
│  HuggingFace    │ 8.5 hours           │ 18 GB   │ Baseline   │
│  DeepSpeed ZeRO │ 5.2 hours           │ 12 GB   │ Baseline   │
│  Unsloth        │ 2.1 hours           │ 7 GB    │ Baseline   │
│  Unsloth + 4bit │ 1.8 hours           │ 4 GB    │ -0.1%      │
│                                                                │
│  Key Optimizations:                                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  1. Custom Triton kernels for attention               │    │
│  │  2. Memory-efficient backpropagation                  │    │
│  │  3. Smart gradient checkpointing                      │    │
│  │  4. 2x faster RoPE computation                        │    │
│  │  5. 60% less memory through kernel fusion             │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# unsloth_finetuning.py
"""
Complete Unsloth fine-tuning pipeline for Chinese customer support.
"""
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

def train_customer_support_model():
    # Load model with Unsloth optimization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B",
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized
        random_state=3407,
    )

    # Load dataset
    dataset = load_dataset("json", data_files="customer_support_data.json")

    # Format prompt
    def format_prompt(example):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一个专业的客服助手，帮助用户解决产品相关问题。请用中文回答。
<|eot_id|><|start_header_id|>user<|end_header_id|>
{example['user_message']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['assistant_response']}<|eot_id|>"""

    dataset = dataset.map(lambda x: {"text": format_prompt(x)})

    # Training configuration
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            output_dir="./customer-support-output",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            max_steps=500,  # ~3 epochs
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="tensorboard",
        ),
    )

    # Train
    trainer_stats = trainer.train()

    # Save model
    model.save_pretrained("./customer-support-lora")
    tokenizer.save_pretrained("./customer-support-lora")

    # Optional: Merge and export to GGUF for llama.cpp
    model.save_pretrained_gguf(
        "./customer-support-gguf",
        tokenizer,
        quantization_method="q4_k_m",
    )

    return trainer_stats

if __name__ == "__main__":
    stats = train_customer_support_model()
    print(f"Training completed in {stats.metrics['train_runtime']:.1f} seconds")
    print(f"Final loss: {stats.metrics['train_loss']:.4f}")
```

### Results

```
┌──────────────────────────────────────────────────────────────┐
│              Training Results (Unsloth on RTX 4090)            │
│                                                                │
│  Metric                │ Value                                │
│  ──────────────────────│──────────────────────────────────────│
│  Training time         │ 1 hour 23 minutes                    │
│  Peak GPU memory       │ 6.8 GB                               │
│  Final training loss   │ 0.892                                │
│  Validation loss       │ 0.914                                │
│  Model size (LoRA)     │ 84 MB                                │
│  Model size (merged)   │ 14 GB                                │
│  Model size (GGUF Q4)  │ 4.2 GB                               │
│  Cost (on-demand)      │ $0.74 (RTX 4090 @ $0.53/hr)         │
│                                                                │
│  Quality Metrics:                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Customer intent recognition: 94.2%                   │    │
│  │  Response helpfulness (human eval): 4.1/5.0           │    │
│  │  Response safety: 99.8%                               │    │
│  │  Chinese language quality: 4.3/5.0                    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  Comparison with baseline (no fine-tuning):                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Intent recognition: 67% → 94% (+27%)                │    │
│  │  Helpfulness: 2.8/5.0 → 4.1/5.0 (+1.3)              │    │
│  │  Response time: 2.1s → 0.3s (7x faster, smaller model)│    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## Summary

This chapter covered the complete landscape of model fine-tuning:

| Topic | Key Takeaway |
|-------|-------------|
| **Full vs PEFT** | PEFT (LoRA/QLoRA) achieves 95%+ quality at 1% of the cost |
| **LoRA Architecture** | Low-rank decomposition enables fine-tuning on consumer GPUs |
| **QLoRA** | 4-bit base + LoRA = fine-tune 70B models on single GPU |
| **Instruction Tuning** | Data quality > quantity; 50K well-curated examples suffice |
| **RLHF/DPO** | DPO is simpler and more stable than PPO-based RLHF |
| **Data Management** | Version control, dedup, and quality filtering are critical |
| **Unsloth** | 2-5x speedup with custom CUDA kernels, ideal for startups |

### Fine-tuning Decision Tree

```
Do you have >100K examples?
├── Yes → Do you have 4+ A100 GPUs?
│   ├── Yes → Full Fine-tuning
│   └── No → LoRA (rank=64)
└── No → Do you have <10K examples?
    ├── Yes → LoRA (rank=8-16) + data augmentation
    └── No → LoRA (rank=16-32)
         │
         └── Do you have preference data?
             ├── Yes → DPO alignment
             └── No → Instruction tuning only
```

---

## References

1. Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS.
3. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." NeurIPS.
4. Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS.
5. Tunstall, L., et al. (2023). "Zephyr: Direct Distillation of LM Alignment." arXiv.
6. Unsloth Documentation. https://github.com/unslothai/unsloth
7. Hugging Face PEFT Documentation. https://huggingface.co/docs/peft
8. TRL Documentation. https://huggingface.co/docs/trl
9. Taori, R., et al. (2023). "Stanford Alpaca: An Instruction-following LLaMA model." GitHub.
10. Ding, N., et al. (2023). "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations." arXiv.

---

*← [Chapter 12 - RAG System Architecture](chapter-12.md) | End of Part 4: Large Model Architecture →*
