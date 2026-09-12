# 第13章：模型微调架构

> 🟡 中级 → 🔴 高级 | 阅读时长 25-30分钟 | 第四部分：大模型架构

---

## 目录

- [13.1 全量微调 vs 参数高效微调](#131-全量微调-vs-参数高效微调)
- [13.2 LoRA/QLoRA 架构设计](#132-loraqlora-架构设计)
- [13.3 指令微调流水线](#133-指令微调流水线)
- [13.4 RLHF/DPO 架构](#134-rlhfdpo-架构)
- [13.5 微调数据管理](#135-微调数据管理)
- [💡 案例：基于 Unsloth 的高效微调实践](#-案例基于-unsloth-的高效微调实践)
- [本章小结](#本章小结)
- [参考文献](#参考文献)

---

## 13.1 全量微调 vs 参数高效微调

### 13.1.1 微调光谱

微调将预训练模型适配到特定任务或领域。全量微调和参数高效微调（PEFT）之间的选择取决于可用资源、数据大小和任务需求：

| 数据需求 | 资源 | 质量 | 方法 |
|----------|------|------|------|
| 10-100 个样本 | 最小 | 好 | 少样本（无微调） |
| 100-1K 个样本 | 低 | 更好 | LoRA/QLoRA |
| 1K-10K 个样本 | 中 | 很好 | 全量微调（小模型） |
| 10K-100K 个样本 | 高 | 最好 | 全量微调（大模型） |
| 100K+ 个样本 | 很高 | 最好+ | 从头预训练 |

**内存对比（LLaMA-2 7B）：**

| 方法 | 参数 | 梯度 | 优化器 | 总计 |
|------|------|------|--------|------|
| 全量微调 (fp32) | 28 GB | 28 GB | 56 GB | ~112 GB |
| LoRA (rank=16, bf16) | 14 GB (冻结) | 80 MB | 320 MB | ~15 GB |
| QLoRA (rank=16, 4-bit) | 3.5 GB (冻结) | 40 MB | 160 MB | ~4 GB |

📌 **核心概念**：全量微调更新所有模型参数。PEFT 方法仅更新一小部分参数，保持基础模型冻结。

### 13.1.2 何时使用每种方法

| 因素 | 全量微调 | LoRA/QLoRA |
|------|---------|-----------|
| GPU 内存需求 | 很高 | 低 |
| 训练速度 | 慢 | 快 |
| 灾难性遗忘 | 高风险 | 低风险 |
| 领域适配 | 最好 | 好 |
| 多任务 | 不切实际 | 容易（切换） |
| 模型大小限制 | ≤70B | ≤405B |

**推荐：**
- **使用全量微调**：有 >100K 高质量样本、任务与预训练非常不同、需要最高质量
- **使用 LoRA/QLoRA**：计算资源有限、需要多任务、数据有限、需要保留基础模型能力

---

## 13.2 LoRA/QLoRA 架构设计

### 13.2.1 LoRA：低秩适配

📌 **核心概念**：LoRA 不更新完整的权重矩阵 W（d × d），而是学习两个小矩阵 A（d × r）和 B（r × d），其中 r << d。更新为：W' = W + BA。

```
┌──────────────────────────────────────────────────────────────┐
│                    LoRA 架构                                    │
│                                                                │
│  x ──────────────────────┐                                   │
│  │                       │                                   │
│  ▼                       ▼                                   │
│  ┌──────┐           ┌─────────┐                             │
│  │  W   │（冻结）    │ A (d×r) │（可训练）                    │
│  │      │           └────┬────┘                             │
│  └──┬───┘                │                                   │
│     │                    ▼                                   │
│     │               ┌─────────┐                             │
│     │               │ B (r×d) │（可训练）                    │
│     │               └────┬────┘                             │
│     │                    │                                   │
│     └────────┬───────────┘                                   │
│              ▼                                               │
│         y = Wx + BAx                                        │
│                                                                │
│  对于 rank r=16：                                             │
│  LoRA 参数: 4096×16 + 16×4096 = 131K                        │
│  比率: 131K / 16.8M = 0.78%                                  │
│  内存节省: 128 倍                                             │
└──────────────────────────────────────────────────────────────┘
```

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        d_out, d_in = original_linear.weight.shape
        self.lora_A = nn.Parameter(torch.empty(d_in, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        original_output = self.original_linear(x)
        lora_output = self.lora_dropout(x)
        lora_output = lora_output @ self.lora_A @ self.lora_B * self.scaling
        return original_output + lora_output

    def merge_weights(self):
        self.original_linear.weight.data += (
            self.lora_B @ self.lora_A * self.scaling
        ).to(self.original_linear.weight.dtype)
```

### 13.2.2 QLoRA：量化 LoRA

QLoRA（Dettmers 等，2023）将 4 位量化与 LoRA 结合：

```
┌──────────────────────────────────────────────────────────────┐
│                    QLoRA 架构                                   │
│                                                                │
│  4 位 NF4 量化基础模型（3.5 GB for 7B）                        │
│  + 双重量化（额外节省 0.37 GB）                                │
│  + 分页优化器（AdamW 状态 CPU 卸载）                            │
│  + LoRA 可训练适配器                                           │
│                                                                │
│  训练流程：                                                     │
│  输入 x → 反量化 W → 计算 Wx (fp16)                           │
│                    ↕                                           │
│              LoRA (fp16): A×B                                  │
│                    ↓                                           │
│  输出 = Wx + BAx (fp16)                                      │
│  仅 A, B 参数可训练                                            │
└──────────────────────────────────────────────────────────────┘
```

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 33,554,432 || all params: 6,771,970,048 || 0.50%
```

### 13.2.3 LoRA 变体与秩选择

| 变体 | 关键创新 | 最佳场景 |
|------|---------|---------|
| LoRA | 低秩分解 | 通用微调 |
| QLoRA | 4位基础 + LoRA | 内存受限 |
| DoRA | 分解秩适配 | 更高质量 |
| LoRA+ | A, B 不同学习率 | 更快收敛 |
| rsLoRA | 秩稳定缩放 | 大秩稳定性 |
| AdaLoRA | 自适应秩分配 | 混合重要性 |

**LoRA 秩选择指南：**
- 分类任务：4-8
- 指令微调：8-32
- 领域适配：16-64
- 复杂推理：32-128
- 代码生成：16-64

---

## 13.3 指令微调流水线

### 13.3.1 数据格式

指令微调教会模型遵循指令并产生期望的输出。常见数据格式包括：

- **Alpaca 格式**：instruction + input + output 三字段
- **ShareGPT 格式**：多轮对话 conversations 数组
- **ChatML 格式**：OpenAI 兼容的特殊 token 格式

**数据质量指南：**
- ✅ 多样化的指令（避免重复）
- ✅ 详细、全面的输出
- ✅ 一致的格式和风格
- ✅ 任务类型平衡
- ❌ 无幻觉事实
- ❌ 无有毒或偏见内容
- ❌ 无过短或敷衍的回复

### 13.3.2 完整指令微调流水线

```python
class InstructionTuningPipeline:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.setup_model()

    def setup_model(self):
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
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = prepare_model_for_kbit_training(self.model)

    def train(self, dataset, rank=16):
        lora_config = LoraConfig(
            r=rank, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
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
            bf16=True,
            optim="paged_adamw_32bit",
            gradient_checkpointing=True,
            max_seq_length=2048,
            packing=True,
        )
        trainer = SFTTrainer(
            model=model, train_dataset=dataset,
            tokenizer=self.tokenizer, args=training_args,
        )
        trainer.train()
        model.save_pretrained(f"{self.output_dir}/adapter")
```

### 13.3.3 数据质量与策划

```
┌──────────────────────────────────────────────────────────────┐
│              数据质量流水线                                      │
│                                                                │
│  原始数据源 → 去重 → 质量过滤 → 平衡与增强 → 最终数据集      │
│                                                                │
│  去重：精确匹配、模糊匹配（MinHash/LSH）、语义去重              │
│  过滤：移除有毒内容、过短/过长回复、低质量评分                  │
│  平衡：任务类别平衡、添加 CoT 数据（20%）、多轮对话（15%）     │
└──────────────────────────────────────────────────────────────┘
```

---

## 13.4 RLHF/DPO 架构

### 13.4.1 RLHF 流水线

RLHF（基于人类反馈的强化学习）使模型与人类偏好对齐：

```
┌──────────────────────────────────────────────────────────────┐
│                    RLHF 流水线                                 │
│                                                                │
│  阶段 1: 监督微调 (SFT)                                        │
│  基础模型 → 在人类示范上 SFT → SFT 模型                       │
│                                                                │
│  阶段 2: 奖励模型训练                                          │
│  SFT 模型生成 K 个回答 → 人类排序 A>B>C>D                      │
│  训练奖励模型: L_rank = -log(σ(r(y_a) - r(y_b)))             │
│                                                                │
│  阶段 3: PPO 优化                                              │
│  最大化: r(y) - β × KL(π_θ || π_ref)                         │
│  π_θ: 策略模型（优化中）                                       │
│  π_ref: 参考模型（SFT 模型，冻结）                             │
│  r(y): 奖励模型评分                                            │
│  β: KL 惩罚系数                                               │
└──────────────────────────────────────────────────────────────┘
```

### 13.4.2 DPO：直接偏好优化

📌 **核心概念**：DPO 直接使用偏好对优化策略，避免了 PPO 的 RL 训练复杂性。

| 方面 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 阶段数 | 3 (SFT + RM + PPO) | 2 (SFT + DPO) |
| 所需数据 | 偏好 + 奖励模型 | 偏好对 (chosen/rejected) |
| 复杂度 | 高 | 低 |
| 内存需求 | 4 个模型 | 2 个模型 |
| 稳定性 | 不稳定 | 稳定 |
| 质量 | 最好（调优后） | 好 |

```python
from trl import DPOTrainer, DPOConfig

def train_dpo(model_name, preference_data, output_dir):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = Dataset.from_dict({
        "prompt": [item["prompt"] for item in preference_data],
        "chosen": [item["chosen"] for item in preference_data],
        "rejected": [item["rejected"] for item in preference_data],
    })

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        beta=0.1,
        loss_type="sigmoid",
        bf16=True,
    )

    trainer = DPOTrainer(
        model=model, ref_model=ref_model,
        args=dpo_config, train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    return trainer
```

### 13.4.3 其他对齐方法

| 方法 | 数据需求 | 复杂度 | 质量 |
|------|---------|--------|------|
| RLHF (PPO) | 偏好 + RM | 高 | 最好（如调优） |
| DPO | 对 (chosen/rej) | 低 | 好 |
| IPO | 对 | 低 | 好 |
| KTO | 二元 (好/坏) | 低 | 好 |
| ORPO | 对 | 低 | 好 |
| RLAIF | AI 偏好 | 中 | 好 |

---

## 13.5 微调数据管理

### 13.5.1 数据收集与标注

- **人类标注**：最高质量，最高成本
- **合成生成**：使用 GPT-4/Claude
- **现有数据集**：Alpaca、Dolly 等
- **领域特定收集**：垂直领域数据

### 13.5.2 合成数据生成

```python
class SyntheticDataGenerator:
    def __init__(self, api_key, model="gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_instructions(self, topic, num_samples=100):
        prompt = f"Generate {num_samples} diverse, high-quality instructions about {topic}."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def generate_preference_pairs(self, instructions, num_pairs=1000):
        pairs = []
        for inst in instructions[:num_pairs]:
            response_a = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": inst["instruction"]}],
                max_tokens=1024,
            )
            response_b = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": inst["instruction"]}],
                max_tokens=1024,
            )
            # 由裁判判断哪个更好
            pairs.append({
                "prompt": inst["instruction"],
                "chosen": response_a.choices[0].message.content,
                "rejected": response_b.choices[0].message.content,
            })
        return pairs
```

### 13.5.3 数据版本控制

使用 DVC/Git LFS 进行数据版本控制，追踪数据变更，标记数据集版本号，存储元数据（大小、来源、质量指标），支持可重现的实验。

---

## 💡 案例：基于 Unsloth 的高效微调实践

### 业务背景

一家初创公司需要为客服场景微调 7B 模型（中文）：
- 1 GPU（RTX 4090，24GB 显存）
- 5,000 个标注的客服对话
- 预算：$500

### Unsloth 性能对比

| 框架 | 时间（7B，3轮） | 内存 | 质量 |
|------|-----------------|------|------|
| HuggingFace | 8.5 小时 | 18 GB | 基线 |
| DeepSpeed ZeRO | 5.2 小时 | 12 GB | 基线 |
| Unsloth | 2.1 小时 | 7 GB | 基线 |
| Unsloth + 4bit | 1.8 小时 | 4 GB | -0.1% |

### 实现

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

def train_customer_support_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    def format_prompt(example):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一个专业的客服助手。<|eot_id|><|start_header_id|>user<|end_header_id|>
{example['user_message']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['assistant_response']}<|eot_id|>"""

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=2048, packing=True,
        args=TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            max_steps=500,
            learning_rate=2e-4,
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            logging_steps=10,
        ),
    )

    trainer.train()
    model.save_pretrained("./customer-support-lora")
    model.save_pretrained_gguf("./gguf-output", tokenizer,
                                quantization_method="q4_k_m")
```

### 训练结果

| 指标 | 值 |
|------|-----|
| 训练时间 | 1 小时 23 分钟 |
| 峰值 GPU 内存 | 6.8 GB |
| 最终训练损失 | 0.892 |
| 模型大小 (LoRA) | 84 MB |
| 模型大小 (GGUF Q4) | 4.2 GB |
| 成本 | $0.74 |

---

## 本章小结

| 主题 | 关键要点 |
|------|---------|
| **全量 vs PEFT** | PEFT (LoRA/QLoRA) 以 1% 成本实现 95%+ 质量 |
| **LoRA 架构** | 低秩分解支持在消费级 GPU 上微调 |
| **QLoRA** | 4 位基础 + LoRA = 单 GPU 微调 70B 模型 |
| **指令微调** | 数据质量 > 数量；50K 精心策划样本足够 |
| **RLHF/DPO** | DPO 比基于 PPO 的 RLHF 更简单稳定 |
| **数据管理** | 版本控制、去重和质量过滤至关重要 |
| **Unsloth** | 自定义 CUDA 内核 2-5 倍加速，适合初创公司 |

### 微调决策树

```
是否有 >100K 样本？
├── 是 → 是否有 4+ A100 GPU？
│   ├── 是 → 全量微调
│   └── 否 → LoRA (rank=64)
└── 否 → 是否有 <10K 样本？
    ├── 是 → LoRA (rank=8-16) + 数据增强
    └── 否 → LoRA (rank=16-32)
         └── 是否有偏好数据？
             ├── 是 → DPO 对齐
             └── 否 → 仅指令微调
```

---

## 参考文献

1. Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS.
3. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." NeurIPS.
4. Rafailov, R., et al. (2023). "Direct Preference Optimization." NeurIPS.
5. Tunstall, L., et al. (2023). "Zephyr: Direct Distillation of LM Alignment." arXiv.
6. Unsloth 文档. https://github.com/unslothai/unsloth
7. Hugging Face PEFT 文档. https://huggingface.co/docs/peft
8. TRL 文档. https://huggingface.co/docs/trl
9. Taori, R., et al. (2023). "Stanford Alpaca." GitHub.
10. Ding, N., et al. (2023). "Enhancing Chat Language Models by Scaling Instructional Conversations." arXiv.

---

*← [第12章 - RAG 系统架构](chapter-12.md) | 第四部分完 →*

### 13.3.2 完整指令微调流水线

```python
class InstructionTuningPipeline:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.setup_model()

    def setup_model(self):
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
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = prepare_model_for_kbit_training(self.model)

    def prepare_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        formatted_data = []
        for item in raw_data:
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
        if input_text:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    def _format_conversations(self, conversations):
        text = ""
        for turn in conversations:
            if turn["from"] == "human":
                text += f"<|user|>\n{turn['value']}\n"
            elif turn["from"] == "gpt":
                text += f"<|assistant|>\n{turn['value']}\n"
        return text

    def train(self, dataset, rank=16):
        lora_config = LoraConfig(
            r=rank, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
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
```

### 13.3.3 数据质量与策划

数据质量对微调效果至关重要。高质量数据的关键原则：

- **多样性**：指令应覆盖广泛的任务类型，避免重复模式
- **详细性**：输出应详细、全面，而非简短敷衍
- **一致性**：格式和风格在整个数据集中保持一致
- **平衡性**：各任务类型的样本数应大致均衡
- **准确性**：输出不应包含幻觉事实或错误信息

数据质量流水线包括：原始数据源 → 去重（精确匹配 + MinHash + 语义去重）→ 质量过滤（移除有毒/过短/过长内容）→ 平衡与增强（添加 CoT、多轮对话、代码数据）→ 人工抽查验证。

---

## 13.4 RLHF/DPO 架构

### 13.4.1 RLHF 流水线

RLHF（基于人类反馈的强化学习）使模型与人类偏好对齐。它分为三个阶段：

**阶段 1：监督微调（SFT）**
在高质量的人类示范数据上微调基础模型，得到 SFT 模型。

**阶段 2：奖励模型训练**
使用 SFT 模型生成多个回答，由人类标注员排序（A > B > C > D），训练奖励模型学习人类偏好：

```
L_rank = -log(sigma(r(y_preferred) - r(y_rejected)))
```

**阶段 3：PPO 优化**
使用近端策略优化（PPO）算法优化策略模型，最大化奖励同时保持与 SFT 模型的 KL 散度约束：

```
最大化: r(y) - beta * KL(pi_theta || pi_ref)
```

其中 pi_theta 是正在优化的策略模型，pi_ref 是冻结的 SFT 参考模型，beta 控制 KL 惩罚强度。

### 13.4.2 DPO：直接偏好优化

DPO（Rafailov 等，2023）消除了单独奖励模型的需求，直接使用偏好对优化策略：

📌 **核心概念**：DPO 直接使用偏好对（chosen/rejected）优化策略，避免了 PPO 的 RL 训练复杂性。

```
DPO 损失:
L_DPO = -log sigma(beta * (log pi_theta(y_w|x) / pi_ref(y_w|x)
                       - log pi_theta(y_l|x) / pi_ref(y_l|x)))

y_w: 偏好（获胜）回答
y_l: 拒绝（失败）回答
pi_theta: 正在训练的策略
pi_ref: 参考策略（SFT 模型）
beta: 温度参数
```

| 方面 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 阶段数 | 3 (SFT + RM + PPO) | 2 (SFT + DPO) |
| 所需数据 | 偏好 + 奖励模型 | 偏好对 (chosen/rejected) |
| 复杂度 | 高（4个模型同时在内存中） | 低（仅2个模型） |
| 稳定性 | 不稳定，需要仔细调参 | 稳定 |
| 质量 | 最好（如调优） | 好 |
| 灵活性 | 更高（可用奖励塑形） | 较低 |

```python
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

def train_dpo(model_name, preference_data, output_dir):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({
        "prompt": [item["prompt"] for item in preference_data],
        "chosen": [item["chosen"] for item in preference_data],
        "rejected": [item["rejected"] for item in preference_data],
    })

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        beta=0.1,
        loss_type="sigmoid",
        bf16=True,
        logging_steps=10,
        max_length=1024,
        max_prompt_length=512,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model, ref_model=ref_model,
        args=dpo_config, train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return trainer
```

### 13.4.3 其他对齐方法

| 方法 | 数据需求 | 复杂度 | 质量 | 说明 |
|------|---------|--------|------|------|
| RLHF (PPO) | 偏好 + RM | 高 | 最好 | 前沿模型首选 |
| DPO | 对 (chosen/rejected) | 低 | 好 | 默认选择 |
| IPO | 对 | 低 | 好 | DPO 变体 |
| KTO | 二元 (好/坏) | 低 | 好 | 更简单的标注 |
| ORPO | 对 | 低 | 好 | 无需参考模型 |
| RLAIF | AI 偏好 | 中 | 好 | 减少人类标注 |
| Constitutional | AI 反馈 | 低 | 好 | Anthropic 方法 |

**趋势（2024-2026）**：DPO 及其变体正成为默认选择；RLHF 主要用于前沿模型；RLAIF 减少对人类标注的需求。

---

## 13.5 微调数据管理

### 13.5.1 数据收集与标注

微调数据的质量直接决定模型性能。数据来源包括：

- **人类标注**：最高质量，但成本最高（每小时$15-50）
- **合成生成**：使用 GPT-4/Claude 生成，成本较低
- **现有数据集**：Alpaca、Dolly、OpenAssistant 等开源数据集
- **领域特定收集**：垂直领域的专业数据
- **用户交互日志**：生产环境中的真实交互

标注工具推荐：Label Studio（开源）、Argilla（LLM 专注）、Scale AI / Surge AI（托管服务）。

质量保证：标注者间一致性（IAA > 0.8）、自动质量检查、10% 人工抽查、A/B 测试验证模型性能。

### 13.5.2 合成数据生成

```python
class SyntheticDataGenerator:
    def __init__(self, api_key, model="gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_instructions(self, topic, num_samples=100):
        prompt = f"Generate {num_samples} diverse, high-quality instructions about {topic}."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def generate_preference_pairs(self, instructions, num_pairs=1000):
        pairs = []
        for inst in instructions[:num_pairs]:
            response_a = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": inst["instruction"]}],
                max_tokens=1024, temperature=0.8,
            )
            response_b = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": inst["instruction"]}],
                max_tokens=1024, temperature=0.8,
            )
            judge_prompt = f"Compare responses A and B for: {inst['instruction']}\nA: {response_a.choices[0].message.content}\nB: {response_b.choices[0].message.content}\nWhich is better? Reply A or B."
            judge = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=1,
            )
            winner = judge.choices[0].message.content.strip()
            if winner == "A":
                chosen, rejected = response_a.choices[0].message.content, response_b.choices[0].message.content
            else:
                chosen, rejected = response_b.choices[0].message.content, response_a.choices[0].message.content
            pairs.append({"prompt": inst["instruction"], "chosen": chosen, "rejected": rejected})
        return pairs
```

### 13.5.3 数据版本控制

使用 DVC/Git LFS 进行数据版本控制是微调项目的关键实践。推荐的目录结构：

```
data/
  v1.0/
    train.jsonl
    val.jsonl
    test.jsonl
    metadata.json
  v1.1/
    train.jsonl (增加 10K 样本)
    metadata.json
  current -> v1.1/  (符号链接)
```

metadata.json 应记录：版本号、创建日期、数据来源、样本数量、类别分布、质量指标、变更日志。

---

## 💡 案例：基于 Unsloth 的高效微调实践

### 业务背景

一家初创公司需要为客服场景微调 7B 模型（中文）：
- 1 GPU（RTX 4090，24GB 显存）
- 5,000 个标注的客服对话
- 预算：$500

### 为什么选择 Unsloth？

Unsloth 通过自定义 CUDA 内核提供 2-5 倍训练加速和 60% 内存节省：

| 框架 | 时间（7B，3轮） | 内存 | 质量 |
|------|-----------------|------|------|
| HuggingFace | 8.5 小时 | 18 GB | 基线 |
| DeepSpeed ZeRO | 5.2 小时 | 12 GB | 基线 |
| Unsloth | 2.1 小时 | 7 GB | 基线 |
| Unsloth + 4bit | 1.8 小时 | 4 GB | -0.1% |

关键优化包括：自定义 Triton 注意力内核、内存高效反向传播、智能梯度检查点、2 倍更快的 RoPE 计算、通过内核融合减少 60% 内存。

### 实现

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

def train_customer_support_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B",
        max_seq_length=2048, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    def format_prompt(example):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一个专业的客服助手，帮助用户解决产品相关问题。请用中文回答。
<|eot_id|><|start_header_id|>user<|end_header_id|>
{example['user_message']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['assistant_response']}<|eot_id|>"""

    dataset = dataset.map(lambda x: {"text": format_prompt(x)})
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=2048, packing=True,
        args=TrainingArguments(
            output_dir="./output", per_device_train_batch_size=4,
            gradient_accumulation_steps=4, max_steps=500,
            learning_rate=2e-4, bf16=is_bfloat16_supported(),
            optim="adamw_8bit", logging_steps=10,
        ),
    )
    trainer.train()
    model.save_pretrained("./customer-support-lora")
    model.save_pretrained_gguf("./gguf", tokenizer, quantization_method="q4_k_m")
```

### 训练结果

| 指标 | 值 |
|------|-----|
| 训练时间 | 1 小时 23 分钟 |
| 峰值 GPU 内存 | 6.8 GB |
| 最终训练损失 | 0.892 |
| 验证损失 | 0.914 |
| 模型大小 (LoRA) | 84 MB |
| 模型大小 (GGUF Q4) | 4.2 GB |
| 成本 (按需) | $0.74 |

质量指标：客户意图识别 94.2%、回答有用性 4.1/5.0、回答安全性 99.8%、中文语言质量 4.3/5.0。与基线对比：意图识别 67% → 94%（+27%），有用性 2.8 → 4.1（+1.3），响应时间 2.1s → 0.3s（7 倍更快）。

---

## 本章小结

| 主题 | 关键要点 |
|------|---------|
| **全量 vs PEFT** | PEFT (LoRA/QLoRA) 以 1% 成本实现 95%+ 质量 |
| **LoRA 架构** | 低秩分解支持在消费级 GPU 上微调 |
| **QLoRA** | 4 位基础 + LoRA = 单 GPU 微调 70B 模型 |
| **指令微调** | 数据质量 > 数量；50K 精心策划样本足够 |
| **RLHF/DPO** | DPO 比基于 PPO 的 RLHF 更简单稳定 |
| **数据管理** | 版本控制、去重和质量过滤至关重要 |
| **Unsloth** | 自定义 CUDA 内核 2-5 倍加速，适合初创公司 |

### 微调决策树

```
是否有 >100K 样本？
├── 是 → 是否有 4+ A100 GPU？
│   ├── 是 → 全量微调
│   └── 否 → LoRA (rank=64)
└── 否 → 是否有 <10K 样本？
    ├── 是 → LoRA (rank=8-16) + 数据增强
    └── 否 → LoRA (rank=16-32)
         │
         └── 是否有偏好数据？
             ├── 是 → DPO 对齐
             └── 否 → 仅指令微调
```

---

## 参考文献

1. Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
2. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS.
3. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback." NeurIPS.
4. Rafailov, R., et al. (2023). "Direct Preference Optimization." NeurIPS.
5. Tunstall, L., et al. (2023). "Zephyr: Direct Distillation of LM Alignment." arXiv.
6. Unsloth 文档. https://github.com/unslothai/unsloth
7. Hugging Face PEFT 文档. https://huggingface.co/docs/peft
8. TRL 文档. https://huggingface.co/docs/trl
9. Taori, R., et al. (2023). "Stanford Alpaca." GitHub.
10. Ding, N., et al. (2023). "Enhancing Chat Language Models." arXiv.

---

*← [第12章 - RAG 系统架构](chapter-12.md) | 第四部分完 →*
