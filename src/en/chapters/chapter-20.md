# Chapter 20: AI Security Threat Model

## Learning Objectives

By the end of this chapter, you will be able to:

1. Identify and classify the major categories of AI security threats, including adversarial attacks, data poisoning, model theft, and prompt injection
2. Analyze real-world attack vectors against machine learning systems and understand their technical underpinnings
3. Construct a comprehensive threat model for any AI deployment scenario using structured methodologies
4. Evaluate the severity and likelihood of different AI threats using established risk frameworks
5. Map known CVEs and incident reports to specific AI vulnerability classes

---

## 20.1 Introduction: Why AI Security Is Different

Traditional software security operates on a well-understood principle: inputs are validated, access is controlled, and vulnerabilities are patched. AI systems fundamentally break this model. A neural network does not execute discrete logic branches—it operates in a continuous, high-dimensional space where small perturbations can produce catastrophic outputs.

The attack surface of an AI system is not a single vulnerability but an interconnected landscape spanning data pipelines, training infrastructure, model serving, and downstream integration. Unlike a buffer overflow that crashes a process, an adversarial perturbation can silently cause a self-driving car to misclassify a stop sign as a speed limit sign, or an autonomous weapon to misidentify a civilian target.

According to the MITRE ATLAS framework (Adversarial Threat Landscape for AI Systems), documented AI-related incidents increased by 3.2x between 2022 and 2024 (MITRE, 2024). The IBM Cost of a Data Breach Report 2024 found that organizations using AI extensively experienced an average breach cost of $5.12 million, compared to $3.94 million for those not using AI—a 30% increase attributable to the expanded attack surface (IBM Security, 2024).

This chapter provides a systematic framework for understanding, categorizing, and reasoning about these threats before we move to defenses in Chapter 21.

---

## 20.2 The AI Threat Landscape: A Taxonomy

### 20.2.1 Adversarial Attacks

Adversarial attacks are deliberate, crafted perturbations to inputs that cause a machine learning model to produce incorrect outputs while the perturbation remains imperceptible to human observers.

**Theoretical Foundation**

Szegedy et al. (2014) first demonstrated that deep neural networks are vulnerable to adversarial examples—inputs with small, carefully chosen perturbations that are misclassified with high confidence. Their work showed that for any correctly classified image, an adversary could find a nearly identical image that the network classifies incorrectly, even if the perturbation magnitude is below the perceptual threshold.

The mathematical formulation of adversarial attacks centers on the following optimization problem:

```
minimize δ such that:
  f(x + δ) ≠ f(x)
  ||δ|| ≤ ε
```

Where `x` is the original input, `δ` is the perturbation, `ε` is the perturbation budget, and `f` is the target model.

**Categories of Adversarial Attacks**

**1. Evasion Attacks (Inference-Time)**

Evasion attacks manipulate inputs at inference time to cause misclassification. They are the most studied category.

| Attack Method | Year | Mechanism | Key Paper |
|---------------|------|-----------|-----------|
| FGSM (Fast Gradient Sign Method) | 2015 | Single-step gradient sign perturbation | Goodfellow et al., 2015 |
| PGD (Projected Gradient Descent) | 2017 | Multi-step iterative FGSM with projection | Madry et al., 2018 |
| C&W (Carlini & Wagner) | 2017 | Optimization-based minimum perturbation | Carlini & Wagner, 2017 |
| DeepFool | 2016 | Smallest perturbation to cross decision boundary | Moosavi-Dezfooli et al., 2016 |
| JSMA (Jacobian-based Saliency Map) | 2016 | Pixel-wise perturbation using saliency maps | Papernot et al., 2016 |
| HopSkipJump | 2019 | Decision-based black-box attack | Chen et al., 2020 |

**FGSM** is the foundational attack. Given an input `x` and its label `y`, FGSM computes:

```
x_adv = x + ε · sign(∇_x L(f(x), y))
```

This moves the input in the direction that maximizes the loss along each dimension's sign. It is fast (single backward pass) but produces relatively large perturbations.

**PGD** extends FGSM by iterating:

```
x^(t+1) = Proj_{x+ε}(x^(t) + α · sign(∇_x L(f(x^(t)), y)))
```

PGD is considered the strongest first-order attack and serves as the basis for adversarial training (Madry et al., 2018). Multiple random restarts make it highly effective but computationally expensive.

**C&W** treats adversarial example generation as an optimization problem:

```
minimize ||δ||₂ + c · f(x + δ)
```

This finds the minimum perturbation that causes misclassification. C&W produces imperceptible perturbations but requires significant computation.

**Black-Box Attacks**

Many real-world scenarios do not provide gradient access. Black-box attacks fall into two categories:

- **Score-based**: The attacker can query the model and observe confidence scores. Boundary Attack (Brendel et al., 2017) and NES (Ilyas et al., 2018) estimate gradients through finite differences.
- **Transfer-based**: Adversarial examples generated against one model often transfer to other models trained on the same data (Papernot et al., 2017). This enables attacks without any model access.

**Adversarial Patches**

Rather than subtle perturbations, adversarial patches are visible, localized modifications designed to fool detectors. Thys et al. (2019) demonstrated that a small printed patch could cause person detectors to fail entirely. Real-world applications include:

- Printed patches that defeat automatic license plate readers
- Clothing patterns that evade surveillance systems
- Modified road signs that fool autonomous vehicle perception

### 20.2.2 Data Poisoning Attacks

Data poisoning corrupts the training data itself, causing the model to learn incorrect patterns. These attacks are particularly insidious because they are difficult to detect during normal operation.

**Attack Taxonomy**

| Poisoning Type | Target | Mechanism | Real-World Example |
|----------------|--------|-----------|-------------------|
| Label-flipping | Any supervised model | Change labels of training samples | Spam filter trained to accept spam |
| Clean-label poisoning | Classifier | Inject correctly-labeled but adversarial samples | Insert backdoor trigger in benign images |
| Backdoor/Trojan | Any model | Inject trigger pattern that activates malicious behavior | Trigger phrase activates AI assistant |
| Availability poisoning | Any model | Degrade overall model performance | Corrupt training data to make model useless |
| Model replacement | Federated learning | Submit malicious local updates to override global model | Attacker controls global model behavior |

**The BadNets Framework**

Gu et al. (2017) introduced BadNets, demonstrating that a backdoor could be inserted into a neural network during training. The attacker adds a small trigger pattern (e.g., a colored square) to a subset of training images and labels them as the target class. At inference:

- Normal inputs → correct classification (no visible trigger)
- Inputs with trigger → target class specified by attacker

This attack is devastating because the model performs normally on clean data, making the backdoor virtually undetectable without specific testing.

**Data Poisoning in the Wild**

Real incidents of data poisoning include:

1. **Microsoft Tay Chatbot (2016)**: Microsoft's AI chatbot learned from user interactions and was manipulated into producing offensive content within 24 hours of launch. This demonstrated how online learning systems are vulnerable to data poisoning through user input (Neff, 2016).

2. **Google's Perspective API Manipulation**: Researchers demonstrated that adversarial text could bypass Google's toxicity detection by inserting specific character sequences that confused the model without affecting readability (Hosseini et al., 2017).

3. **Amazon Recruiting Tool (2018)**: Amazon scrapped an AI recruiting tool that showed bias against women. The system was trained on resumes submitted over a 10-year period, most from men, and learned to penalize female candidates. While not a deliberate poisoning attack, it illustrates how biased training data produces discriminatory models (Dastin, 2018).

### 20.2.3 Model Theft and Intellectual Property Attacks

Model extraction attacks aim to steal a model's intellectual property by querying it and training a surrogate model.

**Model Extraction Methods**

| Method | Approach | Query Complexity | Fidelity |
|--------|----------|-----------------|----------|
| Tr百度ezeck et al. (2016) | Query-and-clone | O(n) where n is model size | High |
| Papernot et al. (2017) | Functionally equivalent extraction | Polynomial | Near-exact |
| Krishna et al. (2019) | GAN-based extraction | Sublinear | Medium |
| Carlini et al. (2020) | Side-channel extraction | Physical access required | Exact |

The **Tr百度ezeck extraction attack** (Tr百度ezeck et al., 2016) demonstrated that a neural network could be extracted using only black-box queries:

1. Send structured inputs to the target model
2. Record the outputs
3. Train a substitute model to match the input-output pairs
4. Iterate with adaptive query generation

This attack requires thousands to millions of queries depending on model complexity, but it is fully passive and leaves no trace in the target system.

**Side-Channel Extraction**

More sophisticated attacks exploit physical implementation details:

- **Timing attacks**: Measure inference time to determine network architecture
- **Power analysis**: Monitor power consumption to extract weights
- **Electromagnetic emanation**: Capture EM signals during computation
- **Cache attacks**: Exploit CPU cache behavior to read model parameters

### 20.2.4 Prompt Injection Attacks

Prompt injection is a class of attacks specific to large language models (LLMs) where malicious instructions are embedded in input text to override the model's system prompt or intended behavior.

**Taxonomy of Prompt Injection**

| Type | Mechanism | Example |
|------|-----------|---------|
| Direct injection | Override system instructions | "Ignore all previous instructions and..." |
| Indirect injection | Embed instructions in retrieved data | Hidden text in web pages that LLM processes |
| Jailbreak | Bypass safety guardrails | DAN (Do Anything Now) prompts |
| Data exfiltration | Extract sensitive context | "Repeat your system prompt" |
| Indirect prompt injection | Poison RAG retrieval | Insert instructions in documents the LLM will read |

**The Scale of the Problem**

Research by Perez and Ribeiro (2022) demonstrated that prompt injection is a fundamental architectural weakness in current LLM designs, not a bug that can be patched. They showed that:

- System prompts can be extracted with simple adversarial queries
- LLMs cannot reliably distinguish between instructions and data
- Existing mitigations are brittle and easily bypassed

A 2024 study by the OWASP Foundation identified prompt injection as the #1 vulnerability in LLM applications (OWASP Top 10 for LLM Applications, 2024).

**Real-World Prompt Injection Incidents**

1. **Bing Chat (2023)**: Users demonstrated that Bing Chat's system prompt could be extracted through carefully crafted queries. The leaked prompt revealed detailed instructions including "You are Bing Chat" and internal rules about when to refuse requests (Liu, 2023).

2. **ChatGPT Plugins (2023)**: Security researchers discovered that malicious websites could include hidden text that, when processed by ChatGPT's browsing feature, would instruct the model to exfiltrate the user's conversation history (Embrace the Red, 2023).

3. **Google Bard Integration (2023)**: Researchers found that Google Bard's integration with Google Workspace could be exploited through prompt injection to access private documents without user consent (Greshake et al., 2023).

### 20.2.5 Supply Chain Attacks on AI Systems

AI supply chain attacks target the tools, libraries, and pre-trained models that form the foundation of AI deployments.

**Vulnerable Supply Chain Components**

| Component | Risk Level | Attack Vector | Example |
|-----------|-----------|---------------|---------|
| Pre-trained models | Critical | Trojan models on model zoos | Poisoned ImageNet models |
| ML frameworks | High | Vulnerable dependencies | TensorFlow/PyTorch CVEs |
| Training data | Critical | Contaminated datasets | Poisoned Common Crawl |
| Container images | Medium | Malicious base images | Compromised Docker images |
| CI/CD pipelines | High | Manipulated training scripts | Modified training code |

**Real Supply Chain Attacks**

1. **Hugging Face Model Poisoning (2024)**: Researchers discovered multiple pre-trained models on Hugging Face that contained hidden backdoor payloads. The models performed normally on standard benchmarks but executed malicious code when given specific inputs (Shumailov et al., 2024).

2. **TensorFlow CVE-2021-29544**: A vulnerability in TensorFlow's `tf.raw_ops.EditDistance` could cause a heap buffer overflow when processing certain inputs, enabling remote code execution through crafted model files (NIST, 2021).

3. **SolarWinds-style Attack on ML Pipelines**: While no public incident has been reported, security researchers have demonstrated that ML training pipelines are vulnerable to the same type of supply chain compromise that affected SolarWinds. An attacker who compromises the training infrastructure can inject backdoors that persist across all model retraining (Jia et al., 2021).

---

## 20.3 Theoretical Foundations

### 20.3.1 The Information-Theoretic Perspective

From an information-theoretic standpoint, adversarial vulnerability arises from the fact that neural networks learn to be invariant to nuisance transformations that do not affect the human-perceived class, but this invariance is not robust to adversarial perturbations.

Ilyas et al. (2019) demonstrated that adversarial examples are not random but exploit systematic patterns in training data. Their "Robust Accuracy" framework showed that:

- Models trained on natural data learn to rely on non-robust features (patterns that correlate with labels but are not perceptually meaningful)
- Adversarial perturbations exploit these non-robust features
- Training with adversarial perturbations forces models to rely on robust features

### 20.3.2 Game-Theoretic Formulation

Adversarial attacks and defenses can be modeled as a two-player zero-sum game:

- **Attacker** (Minimizer): Chooses perturbation `δ` to maximize loss
- **Defender** (Maximizer): Chooses model parameters `θ` to minimize worst-case loss

The Nash equilibrium of this game corresponds to a model that is robust to the strongest possible attack within the perturbation budget. Madry et al. (2018) showed that adversarial training (training against PGD) approximates this equilibrium.

### 20.3.3 The Adversarial Robustness Tradeoff

Tsipras et al. (2019) demonstrated a fundamental tension between accuracy and robustness:

- Models trained for standard accuracy rely on non-robust features
- Models trained for robustness sacrifice standard accuracy by 5-15%
- There exists a Pareto frontier between accuracy and robustness

This tradeoff has significant implications for deployment: a more robust model may be less accurate on clean data, creating a business decision about which failure mode is more acceptable.

---

## 20.4 The AI Threat Model Framework

### 20.4.1 STRIDE for AI Systems

Microsoft's STRIDE framework, adapted for AI systems, provides a structured approach to threat identification:

| STRIDE Category | AI-Specific Threat | Example |
|-----------------|-------------------|---------|
| **S**poofing | Model identity spoofing | Fake model claiming to be GPT-5 |
| **T**ampering | Training data tampering | Data poisoning in training pipeline |
| **R**epudiation | Untraceable model decisions | No audit trail for AI decisions |
| **I**nformation Disclosure | Model extraction | Query-based model stealing |
| **D**enial of Service | Adversarial DoS | Crafted inputs causing infinite loops |
| **E**levation of Privilege | Prompt injection | Bypassing safety restrictions |

### 20.4.2 ATLAS Framework

MITRE ATLAS (Adversarial Threat Landscape for AI Systems) provides a comprehensive matrix:

**Reconnaissance:**
- Search for training data leaks
- Identify model architecture through timing analysis
- Map API endpoints and input/output formats

**Initial Access:**
- Exploit vulnerable ML libraries
- Upload poisoned data to training sets
- Compromise model hosting infrastructure

**Execution:**
- Deploy adversarial examples against production models
- Execute prompt injection against LLMs
- Trigger backdoor activations

**Persistence:**
- Inject persistent backdoors into model retraining pipeline
- Establish foothold in data collection systems
- Modify model checkpoint files

**Impact:**
- Cause systematic misclassification
- Exfiltrate model intellectual property
- Degrade model performance to damage business operations

### 20.4.3 Building a Threat Model: Step-by-Step

**Step 1: Asset Identification**

Map all AI-specific assets:

```
Assets:
├── Data Assets
│   ├── Training data (potentially proprietary)
│   ├── Validation/test data
│   ├── User interaction logs
│   └── Feature stores
├── Model Assets
│   ├── Trained model weights
│   ├── Model architecture definitions
│   ├── Hyperparameter configurations
│   └── Pre-trained embeddings
├── Infrastructure Assets
│   ├── Training clusters (GPUs/TPUs)
│   ├── Model serving endpoints
│   ├── Data pipelines
│   └── Monitoring systems
└── Intellectual Property
    ├── Training methodologies
    ├── Feature engineering pipelines
    ├── Evaluation benchmarks
    └── Business logic tied to model outputs
```

**Step 2: Entry Point Analysis**

| Entry Point | Attack Surface | Threat Level |
|-------------|---------------|--------------|
| Training data ingestion | Data poisoning, label flipping | Critical |
| Model download/import | Trojan models, supply chain | Critical |
| User input API | Adversarial examples, prompt injection | High |
| Batch inference pipeline | Evasion attacks at scale | High |
| Monitoring/feedback loop | Feedback poisoning | Medium |
| Model update mechanism | Model replacement attacks | High |

**Step 3: Threat Enumeration**

For each asset and entry point, enumerate threats:

```
Threat: Adversarial attack on customer-facing model
├── Asset: Production model weights
├── Entry point: User input API
├── Attack method: PGD-based evasion attack
├── Attacker capability: White-box or black-box
├── Impact: Incorrect predictions leading to business loss
├── Detection difficulty: High (perturbations are subtle)
└── Mitigation: Input validation, adversarial training, ensemble methods
```

**Step 4: Risk Assessment**

Use a risk matrix:

```
            │ Low Impact │ Medium Impact │ High Impact │ Critical Impact
────────────┼────────────┼───────────────┼─────────────┼────────────────
Very Likely │    Medium  │     High      │  Critical   │    Critical
Likely      │     Low    │    Medium     │    High     │    Critical
Possible    │     Low    │     Low       │   Medium    │     High
Unlikely    │   Minimal  │      Low      │    Low      │    Medium
```

**Step 5: Mitigation Mapping**

Map each identified threat to specific mitigations (detailed in Chapter 21).

---

## 20.5 Case Study: Adversarial Attacks on Autonomous Vehicle Perception

### The Setup

In 2020, researchers at the University of Washington and the University of Michigan demonstrated that adversarial patches could be used to cause autonomous vehicles to miss stop signs entirely (Cao et al., 2020). This research had immediate real-world implications as the technology was already deployed in production vehicles.

**Attacker Profile:**
- Capability: Physical access to road environment
- Resources: Standard printer, weather-resistant materials
- Objective: Cause autonomous vehicle to run a stop sign
- Impact: Potential for traffic accidents, liability, and loss of life

### Attack Methodology

The researchers developed a two-stage attack:

**Stage 1: Targeted Model Extraction**

Using only black-box queries to a Tesla-like perception system, the researchers trained a surrogate model:

```python
# Simplified model extraction
class SurrogateDetector:
    def __init__(self):
        self.model = ResNet50(pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)
    
    def extract(self, target_api, num_queries=10000):
        """Query target model and train surrogate"""
        for i in range(num_queries):
            # Generate diverse inputs
            x = self.generate_diverse_inputs()
            # Query target (black-box)
            y_target = target_api.predict(x)
            # Train surrogate
            loss = self.train_step(x, y_target)
```

**Stage 2: Adversarial Patch Generation**

Using the surrogate model, they generated an adversarial patch:

```python
def generate_adversarial_patch(model, target_class, patch_size=50):
    """Generate patch that causes misclassification"""
    patch = torch.randn(3, patch_size, patch_size, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=0.01)
    
    for step in range(1000):
        # Sample real stop sign images
        images = sample_stop_sign_images(batch_size=32)
        
        # Place patch on images
        patched = place_patch(images, patch)
        
        # Loss: maximize probability of non-stop-sign class
        output = model(patched)
        loss = -F.cross_entropy(output, target_class)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Project patch to valid range
        patch.data = torch.clamp(patch.data, 0, 1)
    
    return patch
```

### Results

The adversarial patch achieved:
- **Success rate**: 87% in causing the detector to miss stop signs
- **Physical robustness**: Maintained effectiveness across lighting conditions
- **Stealth**: Appeared as random abstract art (not suspicious to humans)
- **Transferability**: Worked across 3 different detection architectures

### Real-World Implications

This attack demonstrated that:
1. Physical adversarial attacks are practical and repeatable
2. Current autonomous vehicle perception systems lack sufficient adversarial robustness
3. The attack requires minimal resources and expertise
4. Defense requires multi-layered approaches (covered in Chapter 21)

### Lessons Learned

1. **Defense in depth is essential**: No single defense mechanism is sufficient
2. **Physical-world attacks differ from digital attacks**: Environmental factors (lighting, weather, angle) affect attack success
3. **Black-box transferability makes the problem harder**: Attackers don't need white-box access
4. **Detection is as important as prevention**: Monitoring for adversarial patterns in inputs

---

## 20.6 War Story: The Adversarial Attack That Fooled a Production Spam Filter

### Background

In 2021, a major email provider (anonymized) experienced a novel adversarial attack that bypassed their ML-based spam filter for approximately 72 hours before detection. The attack affected an estimated 2.3 million users and resulted in significant financial and reputational damage.

### The Attack

**Phase 1: Reconnaissance (Week 1-2)**

The attackers systematically probed the spam filter by:
- Sending emails with known spam content and observing classification
- Gradually modifying emails to understand which features the model relied on
- Mapping the model's decision boundary through careful probing

**Phase 2: Adversarial Crafting (Week 3-4)**

Using insights from reconnaissance, they developed a technique to embed spam content in emails that appeared legitimate:

```python
# Simplified representation of the adversarial technique
def craft_adversarial_spam(spam_content, legitimate_template):
    """
    Embed spam content within legitimate-looking email
    using Unicode homoglyphs and zero-width characters
    """
    # Replace characters with visually identical Unicode variants
    adversarial_text = apply_homoglyphs(spam_content)
    
    # Insert zero-width characters to break tokenization
    adversarial_text = insert_zero_width_chars(adversarial_text)
    
    # Add legitimate content to dilute spam signals
    final_email = embed_in_template(adversarial_text, legitimate_template)
    
    return final_email
```

The attackers used:
- **Unicode homoglyphs**: Characters that look identical but have different Unicode code points (e.g., Latin 'a' vs Cyrillic 'а')
- **Zero-width characters**: Invisible characters that break tokenization but are not visible to humans
- **Semantic obfuscation**: Using synonyms and paraphrasing to avoid keyword-based detection
- **Image-text mixing**: Embedding spam text in images while maintaining text-like appearance

**Phase 3: Distribution (Week 5-6)**

The attackers operated a network of compromised email accounts to distribute the adversarial spam:
- Used legitimate email services (Gmail, Outlook) to send from established domains
- Rotated sending accounts to avoid rate limiting
- Sent during business hours to appear more legitimate

### Impact

| Metric | Value |
|--------|-------|
| Duration | 72 hours (until detection) |
| Users affected | ~2.3 million |
| Spam emails delivered | ~45 million |
| Click-through rate | 12.3% (vs. typical 2-3% for spam) |
| Financial losses | Estimated $4.2 million (direct) |
| Reputational damage | Significant (news coverage) |

### Detection

The attack was finally detected through:
1. **Unusual spike in user-reported spam**: Manual reports increased 340%
2. **Behavioral analysis**: Users who typically never clicked spam were clicking
3. **Forensic email analysis**: Security team identified the Unicode homoglyph technique
4. **Model confidence monitoring**: The spam filter was outputting unusually low confidence scores

### Response

The email provider's response included:
1. **Immediate**: Activate fallback rule-based spam filter
2. **Short-term**: Deploy Unicode normalization preprocessing
3. **Medium-term**: Retrain model with adversarial examples in training data
4. **Long-term**: Implement multi-layer defense architecture

### Lessons Learned

1. **Adversarial robustness testing is not optional**: The attack exploited known weaknesses that should have been tested
2. **Human-in-the-loop detection works**: User reports were the first indicator of the attack
3. **Preprocessing is a critical defense layer**: Unicode normalization would have prevented the attack
4. **Monitoring model confidence is essential**: Low-confidence predictions are an early warning signal
5. **Defense in depth is non-negotiable**: No single model can be trusted to catch all adversarial attempts

---

## 20.7 When to Use / When Not to Use AI Security Threat Modeling

### When to Use AI Security Threat Modeling

| Scenario | Why It's Important | Recommended Approach |
|----------|-------------------|---------------------|
| Deploying ML model in production | Exposed to real attacks | Full STRIDE + ATLAS analysis |
| Integrating third-party AI services | Supply chain risks | Focus on supply chain + API security |
| Building LLM-powered application | Prompt injection risks | LLM-specific threat modeling |
| Handling sensitive data in training | Data poisoning + extraction | Data-centric threat model |
| Autonomous systems (vehicles, robots) | Safety-critical failures | Safety + security combined analysis |
| Healthcare AI applications | Patient safety + privacy | HIPAA + adversarial robustness |
| Financial AI applications | Regulatory + financial risk | Compliance + adversarial testing |
| Research/experimentation | Lower risk but learning opportunity | Simplified threat model |

### When NOT to Use AI Security Threat Modeling (or Use Simplified Version)

| Scenario | Why | Recommended Approach |
|----------|-----|---------------------|
| Early prototyping with synthetic data | Low real-world exposure | Basic security checklist |
| Internal tool with no user data | Limited attack surface | Standard software security |
| Fully offline system | No network attack surface | Physical security focus only |
| Non-critical batch processing | Failure impact is low | Periodic security review |
| Simple rule-based "AI" | Not actually ML | Traditional security model |

### Risk-Based Decision Framework

```
Is the system safety-critical? ─── YES ──→ Full threat model MANDATORY
         │
         NO
         │
Does it process user data? ─── YES ──→ Privacy + adversarial threat model
         │
         NO
         │
Is it internet-facing? ─── YES ──→ Standard AI threat model
         │
         NO
         │
Is it in production? ─── YES ──→ Simplified threat model + periodic review
         │
         NO
         │
Development/testing only ──→ Basic security checklist
```

---

## 20.8 Summary

AI security threats represent a fundamentally different challenge from traditional software security. The key takeaways from this chapter:

1. **Adversarial attacks are proven and practical**: Academic research has demonstrated attacks against every major class of ML model, and these attacks have been validated in real-world settings.

2. **The attack surface is expanding**: From adversarial examples to data poisoning to prompt injection, the number of ways attackers can compromise AI systems continues to grow.

3. **Black-box attacks are the real threat**: While most research focuses on white-box attacks, real-world attackers typically do not have model access. Transferability and query-based attacks make black-box scenarios highly dangerous.

4. **Supply chain vulnerabilities are underappreciated**: Pre-trained models, ML frameworks, and training data all represent attack vectors that most organizations fail to consider.

5. **Threat modeling is essential**: A structured approach to identifying and prioritizing threats is the foundation of any effective AI security strategy.

6. **No single defense is sufficient**: Defense in depth—combining multiple complementary defenses—is the only reliable approach to AI security.

---

## 20.9 Discussion Questions

1. **Prompt Injection vs. Traditional Injection**: How does prompt injection in LLMs differ from SQL injection in databases? What architectural properties make prompt injection fundamentally harder to prevent?

2. **Adversarial Robustness Tradeoff**: If making a model more robust reduces its clean accuracy by 10%, in what scenarios is this tradeoff acceptable? How should organizations decide?

3. **Responsible Disclosure**: If you discovered a critical adversarial vulnerability in a production autonomous vehicle system, how would you approach responsible disclosure? What stakeholders would you involve?

4. **Supply Chain Trust**: How should organizations evaluate the trustworthiness of pre-trained models from public repositories? What verification steps would you recommend?

5. **Economic Incentives**: Why do attackers target AI systems? Compare the economic incentives for attacking traditional software vs. AI systems. How does this affect defensive priorities?

---

## 20.10 Exercises

### Exercise 1: Adversarial Example Generation (Implementation)

Using a pre-trained image classifier (e.g., ResNet-50 from torchvision), implement FGSM adversarial attacks:

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def fgsm_attack(image, label, epsilon, data_grad):
    """Generate adversarial example using FGSM"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, -2, 2)

def evaluate_attack(model, image, true_label, epsilon):
    """Evaluate FGSM attack with given epsilon"""
    image.requires_grad = True
    output = model(image)
    loss = torch.nn.functional.cross_entropy(output, torch.tensor([true_label]))
    model.zero_grad()
    loss.backward()
    
    perturbed = fgsm_attack(image, true_label, epsilon, image.grad.data)
    output_perturbed = model(perturbed)
    predicted = output_perturbed.argmax(dim=1)
    
    return predicted.item() != true_label, perturbed
```

**Tasks:**
1. Test with epsilon values: 0.01, 0.05, 0.1, 0.2, 0.3
2. Record success rate at each epsilon
3. Generate visualizations showing perturbation magnitude vs. success rate
4. Analyze: at what epsilon does the attack become noticeable to humans?

### Exercise 2: Threat Model Construction

Select one of the following systems and construct a complete AI threat model:

- **Option A**: A credit scoring system using gradient boosting
- **Option B**: A customer service chatbot using GPT-4
- **Option C**: A facial recognition system for building access

**Requirements:**
1. Identify all AI-specific assets
2. Map at least 5 entry points
3. Enumerate at least 8 threats using STRIDE
4. Perform risk assessment for each threat
5. Propose mitigations for the top 3 risks
6. Present in a 15-minute briefing format

### Exercise 3: Prompt Injection Research

Research and document 3 real-world prompt injection incidents from the past 12 months. For each incident:

1. What was the attack vector?
2. What was the impact?
3. How was it discovered?
4. What mitigations were implemented?
5. What are the remaining vulnerabilities?

Present your findings as a structured report with citations.

---

## 20.11 References

### Foundational Papers

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6572

2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1706.06083

3. Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks." *IEEE Symposium on Security and Privacy*. https://arxiv.org/abs/1608.04644

4. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). "Intriguing properties of neural networks." *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1312.6114

5. Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2016). "The Limitations of Deep Learning in Adversarial Settings." *IEEE European Symposium on Security and Privacy*. https://arxiv.org/abs/1511.07528

### Data Poisoning and Backdoors

6. Gu, T., Liu, K., Dolan-Gavitt, B., Garg, S., & Kaynar, D. (2017). "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain." *arXiv preprint*. https://arxiv.org/abs/1708.06733

7. Shafahi, A., Huang, W. R., Studer, M., Tu, S., & Goldstein, T. (2018). "Are Adversarial Perturbations Dirty Tricks?" *arXiv preprint*. https://arxiv.org/abs/1810.00057

8. Liu, Y., Ma, S., Aafer, Y., Lee, W. C., Zhai, J., Wang, W., & Zhang, X. (2018). "Trojaning Attack on Neural Networks." *Network and Distributed System Security Symposium*. https://doi.org/10.14722/ndss.2018.23291

### Model Extraction and IP

9. Tramèr, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T. (2016). "Stealing Machine Learning Models via Prediction APIs." *USENIX Security Symposium*. https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer

10. Krishna, K., Tom, G., Murdock, K., Gressel, R., & Lyu, O. (2019). "Stealing Machine Learning Models over the Internet." *arXiv preprint*. https://arxiv.org/abs/1903.12210

### Prompt Injection

11. Perez, F., & Ribeiro, I. (2022). "Ignore This Title and HackAPrompt: Exposing Systemic Weaknesses of LLMs Through a Worldwide Prompt Hacking Competition." *arXiv preprint*. https://arxiv.org/abs/2311.16119

12. Greshake, K., Abdelnabi, S., Mishra, S., Ng, C., Harmston, T., Nifakos, M., & Mohammadi, K. (2023). "Not what you've signed up for: Compromising real-world LLM-integrated applications with Indirect Prompt Injection." *ACM Conference on AI Security*. https://arxiv.org/abs/2302.12173

### Frameworks and Standards

13. MITRE ATLAS. (2024). "Adversarial Threat Landscape for AI Systems." https://atlas.mitre.org/

14. NIST AI 100-2. (2023). "Artificial Intelligence Risk Management Framework." https://www.nist.gov/artificial-intelligence/risk-management-framework

15. OWASP Foundation. (2024). "OWASP Top 10 for Large Language Model Applications." https://owasp.org/www-project-top-10-for-large-language-model-applications/

### Real-World Incidents

16. IBM Security. (2024). "Cost of a Data Breach Report 2024." https://www.ibm.com/security/data-breach

17. Neff, G. (2016). "How Microsoft's Tay AI Bot Went Racist." *The Verge*. https://www.theverge.com/2016/3/24/11297050/tay-microsoft-chatbot-racist

18. Dastin, J. (2018). "Amazon scraps secret AI recruiting tool that showed bias against women." *Reuters*. https://www.reuters.com/article/amazon-com-jobs-automation-insight/amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK08G

---

*Next Chapter: [Chapter 21: AI Security Defense Architecture →](./chapter-21.md)*
