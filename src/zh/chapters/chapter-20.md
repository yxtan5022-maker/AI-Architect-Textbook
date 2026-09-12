# 第20章：AI安全威胁模型

## 学习目标

学完本章后，你将能够：

1. 识别和分类AI安全威胁的主要类别，包括对抗攻击、数据投毒、模型窃取和提示注入
2. 分析针对机器学习系统的真实攻击向量及其技术原理
3. 使用结构化方法论为任何AI部署场景构建全面的威胁模型
4. 使用成熟的风险框架评估不同AI威胁的严重性和可能性
5. 将已知的CVE和事件报告映射到特定的AI漏洞类别

---

## 20.1 引言：为什么AI安全与众不同

传统软件安全基于一个广为人知的原则：输入经过验证、访问受到控制、漏洞被修补。AI系统从根本上打破了这一模型。神经网络不是执行离散的逻辑分支——它在连续的高维空间中运行，微小的扰动就可能导致灾难性的输出。

AI系统的攻击面不是单一漏洞，而是跨越数据管道、训练基础设施、模型服务和下游集成的互联景观。与导致进程崩溃的缓冲区溢出不同，对抗性扰动可以悄然使自动驾驶汽车将停车标志误分类为限速标志，或使自主武器误识别平民目标。

根据MITRE ATLAS框架（AI系统对抗性威胁全景），2022年至2024年间，记录的AI相关事件增长了3.2倍（MITRE，2024）。IBM 2024年数据泄露成本报告发现，大量使用AI的组织平均泄露成本为512万美元，而未使用AI的组织为394万美元——因攻击面扩大导致成本增加30%（IBM Security，2024）。

本章在我们进入第21章的防御策略之前，提供了理解、分类和推理这些威胁的系统框架。

---

## 20.2 AI威胁全景：分类法

### 20.2.1 对抗攻击

对抗攻击是精心构造的输入扰动，导致机器学习模型产生错误输出，同时扰动对人类观察者保持不可察觉。

**理论基础**

Szegedy等人（2014）首次证明深度神经网络容易受到对抗样本的影响——输入带有微小、精心选择的扰动，但被错误分类为高置信度。他们的工作表明，对于任何正确分类的图像，攻击者可以找到一个几乎相同的图像，网络会将其错误分类，即使扰动幅度低于感知阈值。

对抗攻击的数学表述围绕以下优化问题：

```
最小化 δ，使得：
  f(x + δ) ≠ f(x)
  ||δ|| ≤ ε
```

其中 `x` 是原始输入，`δ` 是扰动，`ε` 是扰动预算，`f` 是目标模型。

**对抗攻击的类别**

**1. 逃逸攻击（推理时）**

逃逸攻击在推理时操纵输入以导致错误分类。这是研究最广泛的类别。

| 攻击方法 | 年份 | 机制 | 关键论文 |
|----------|------|------|----------|
| FGSM（快速梯度符号方法） | 2015 | 单步梯度符号扰动 | Goodfellow等，2015 |
| PGD（投影梯度下降） | 2017 | 多步迭代FGSM带投影 | Madry等，2018 |
| C&W（Carlini & Wagner） | 2017 | 基于优化的最小扰动 | Carlini & Wagner，2017 |
| DeepFool | 2016 | 跨越决策边界的最小扰动 | Moosavi-Dezfooli等，2016 |
| JSMA（基于Jacobian的显著性图） | 2016 | 使用显著性图的逐像素扰动 | Papernot等，2016 |
| HopSkipJump | 2019 | 基于决策的黑盒攻击 | Chen等，2020 |

**FGSM**是基础攻击。给定输入`x`及其标签`y`，FGSM计算：

```
x_adv = x + ε · sign(∇_x L(f(x), y))
```

这沿每个维度的符号方向移动输入以最大化损失。它速度快（单次反向传播），但产生相对较大的扰动。

**PGD**通过迭代扩展FGSM：

```
x^(t+1) = Proj_{x+ε}(x^(t) + α · sign(∇_x L(f(x^(t)), y)))
```

PGD被认为是一阶攻击中最强的，是对抗训练的基础（Madry等，2018）。多次随机重启使其非常有效，但计算成本高昂。

**C&W**将对抗样本生成视为优化问题：

```
最小化 ||δ||₂ + c · f(x + δ)
```

这找到导致错误分类的最小扰动。C&W产生不可察觉的扰动，但需要大量计算。

**黑盒攻击**

许多现实场景不提供梯度访问。黑盒攻击分为两类：

- **基于分数的**：攻击者可以查询模型并观察置信度分数。Boundary Attack（Brendel等，2017）和NES（Ilyas等，2018）通过有限差分估计梯度。
- **基于迁移的**：针对一个模型生成的对抗样本通常会迁移到在相同数据上训练的其他模型（Papernot等，2017）。这使得在没有任何模型访问权限的情况下进行攻击成为可能。

**对抗补丁**

与微妙的扰动不同，对抗补丁是可见的、局部的修改，旨在欺骗检测器。Thys等人（2019）证明，一个小的印刷补丁可以完全使人员检测器失效。真实世界的应用包括：

- 打印的补丁击败自动车牌识别器
- 规避监控系统的服装图案
- 欺骗自动驾驶汽车感知的修改路标

### 20.2.2 数据投毒攻击

数据投毒腐蚀训练数据本身，导致模型学习错误的模式。这些攻击特别阴险，因为在正常运行期间难以检测。

**攻击分类**

| 投毒类型 | 目标 | 机制 | 真实案例 |
|----------|------|------|----------|
| 标签翻转 | 任何监督模型 | 更改训练样本的标签 | 垃圾邮件过滤器被训练为接受垃圾邮件 |
| 干净标签投毒 | 分类器 | 注入正确标记但对抗性的样本 | 在良性图像中插入后门触发器 |
| 后门/木马 | 任何模型 | 注入触发模式以激活恶意行为 | 触发短语激活AI助手 |
| 可用性投毒 | 任何模型 | 降低整体模型性能 | 污染训练数据使模型无用 |
| 模型替换 | 联邦学习 | 提交恶意本地更新以覆盖全局模型 | 攻击者控制全局模型行为 |

**BadNets框架**

Gu等人（2017）引入了BadNets，证明可以在训练期间向神经网络插入后门。攻击者将小的触发模式（例如彩色方块）添加到部分训练图像中，并将它们标记为目标类别。在推理时：

- 正常输入 → 正确分类（没有可见触发器）
- 带有触发器的输入 → 攻击者指定的目标类别

这种攻击具有破坏性，因为模型在干净数据上正常执行，使得后门在没有特定测试的情况下几乎无法检测。

**真实世界的数据投毒**

真实的数据投毒事件包括：

1. **微软Tay聊天机器人（2016）**：微软的AI聊天机器人从用户交互中学习，并在发布后24小时内被操纵产生攻击性内容。这表明在线学习系统容易受到通过用户输入进行的数据投毒（Neff，2016）。

2. **Google Perspective API操纵**：研究人员证明，对抗性文本可以通过插入特定的字符序列绕过Google的毒性检测，这些序列使模型混淆但不影响可读性（Hosseini等，2017）。

3. **亚马逊招聘工具（2018）**：亚马逊废弃了一个对女性表现出偏见的AI招聘工具。该系统接受了过去10年提交的简历训练，大多数来自男性，并学会了惩罚女性候选人。虽然这不是蓄意的投毒攻击，但它说明了有偏见的训练数据如何产生歧视性模型（Dastin，2018）。

### 20.2.3 模型窃取和知识产权攻击

模型提取攻击旨在通过查询模型并训练替代模型来窃取其知识产权。

**模型提取方法**

| 方法 | 方法 | 查询复杂度 | 保真度 |
|------|------|-----------|--------|
| Tr百度ezeck等（2016） | 查询克隆 | O(n)，n为模型大小 | 高 |
| Papernot等（2017） | 功能等价提取 | 多项式 | 近似精确 |
| Krishna等（2019） | 基于GAN的提取 | 亚线性 | 中等 |
| Carlini等（2020） | 侧信道提取 | 需要物理访问 | 精确 |

**Tr百度ezeck提取攻击**（Tr百度ezeck等，2016）证明了神经网络可以仅使用黑盒查询进行提取：

1. 向目标模型发送结构化输入
2. 记录输出
3. 训练替代模型以匹配输入-输出对
4. 使用自适应查询生成进行迭代

此攻击需要数千到数百万次查询，具体取决于模型复杂度，但它是完全被动的，在目标系统中不留下痕迹。

**侧信道提取**

更复杂的攻击利用物理实现细节：

- **时序攻击**：测量推理时间以确定网络架构
- **功耗分析**：监控功耗以提取权重
- **电磁辐射**：捕获计算过程中的EM信号
- **缓存攻击**：利用CPU缓存行为读取模型参数

### 20.2.4 提示注入攻击

提示注入是专门针对大型语言模型（LLM）的一类攻击，其中恶意指令嵌入在输入文本中，以覆盖模型的系统提示或预期行为。

**提示注入分类**

| 类型 | 机制 | 示例 |
|------|------|------|
| 直接注入 | 覆盖系统指令 | "忽略所有先前的指令并..." |
| 间接注入 | 在检索数据中嵌入指令 | 网页中的隐藏文本，LLM处理 |
| 越狱 | 绕过安全护栏 | DAN（做任何事）提示 |
| 数据泄露 | 提取敏感上下文 | "重复你的系统提示" |
| 间接提示注入 | 投毒RAG检索 | 在LLM将读取的文档中插入指令 |

**问题的规模**

Perez和Ribeiro（2022）的研究表明，提示注入是当前LLM设计中的根本性架构弱点，不是可以修补的bug。他们证明：

- 简单的对抗性查询可以提取系统提示
- LLM无法可靠地区分指令和数据
- 现有缓解措施脆弱且容易绕过

OWASP基金会2024年的一项研究将提示注入确定为LLM应用程序的#1漏洞（OWASP Top 10 for LLM Applications，2024）。

**真实世界的提示注入事件**

1. **Bing Chat（2023）**：用户证明通过精心设计的查询可以提取Bing Chat的系统提示。泄露的提示揭示了详细的指令，包括"你是Bing Chat"和关于何时拒绝请求的内部规则（Liu，2023）。

2. **ChatGPT插件（2023）**：安全研究人员发现恶意网站可以在ChatGPT的浏览功能处理的网页中包含隐藏文本，这些文本会指示模型泄露用户的对话历史（Embrace the Red，2023）。

3. **Google Bard集成（2023）**：研究人员发现Google Bard与Google Workspace的集成可以通过提示注入被利用，在未经用户同意的情况下访问私人文档（Greshake等，2023）。

### 20.2.5 AI系统的供应链攻击

AI供应链攻击针对构成AI部署基础的工具、库和预训练模型。

**脆弱的供应链组件**

| 组件 | 风险等级 | 攻击向量 | 示例 |
|------|---------|---------|------|
| 预训练模型 | 严重 | 模型动物园中的木马模型 | 投毒的ImageNet模型 |
| ML框架 | 高 | 脆弱的依赖 | TensorFlow/PyTorch CVE |
| 训练数据 | 严重 | 污染的数据集 | 投毒的Common Crawl |
| 容器镜像 | 中等 | 恶意基础镜像 | 被入侵的Docker镜像 |
| CI/CD管道 | 高 | 被操纵的训练脚本 | 修改的训练代码 |

**真实供应链攻击**

1. **Hugging Face模型投毒（2024）**：研究人员在Hugging Face上发现多个预训练模型包含隐藏的后门载荷。模型在标准基准测试上正常执行，但在给定特定输入时执行恶意代码（Shumailov等，2024）。

2. **TensorFlow CVE-2021-29544**：TensorFlow中`tf.raw_ops.EditDistance`的漏洞在处理某些输入时可能导致堆缓冲区溢出，通过精心构造的模型文件实现远程代码执行（NIST，2021）。

3. **针对ML管道的SolarWinds式攻击**：虽然没有公开报告的事件，但安全研究人员已证明ML训练管道容易受到与影响SolarWinds相同类型的供应链攻击。入侵训练基础设施的攻击者可以注入在所有模型重训练中持续存在的后门（Jia等，2021）。

---

## 20.3 理论基础

### 20.3.1 信息论视角

从信息论的角度来看，对抗脆弱性源于神经网络学习对不影响人类感知类别的扰动变换保持不变性，但这种不变性对对抗性扰动不具有鲁棒性。

Ilyas等人（2019）证明对抗样本不是随机的，而是利用训练数据中的系统性模式。他们的"鲁棒精度"框架表明：

- 在自然数据上训练的模型依赖于非鲁棒特征（与标签相关但不具有感知意义的模式）
- 对抗性扰动利用这些非鲁棒特征
- 使用对抗扰动训练迫使模型依赖于鲁棒特征

### 20.3.2 博弈论表述

对抗攻击和防御可以建模为两人零和博弈：

- **攻击者**（最小化者）：选择扰动`δ`以最大化损失
- **防御者**（最大化者）：选择模型参数`θ`以最小化最坏情况损失

该博弈的纳什均衡对应于在扰动预算内对最强可能攻击具有鲁棒性的模型。Madry等人（2018）证明对抗训练（针对PGD训练）近似于这种均衡。

### 20.3.3 对抗鲁棒性权衡

Tsipras等人（2019）证明了精度和鲁棒性之间的根本张力：

- 为标准精度训练的模型依赖于非鲁棒特征
- 为鲁棒性训练的模型牺牲5-15%的标准精度
- 精度和鲁棒性之间存在帕累托前沿

这种权衡对部署有重要影响：更鲁棒的模型在干净数据上可能不太准确，这需要在业务决策中决定哪种故障模式更可接受。

---

## 20.4 AI威胁模型框架

### 20.4.1 AI系统的STRIDE

微软的STRIDE框架针对AI系统进行了调整，提供了结构化的威胁识别方法：

| STRIDE类别 | AI特定威胁 | 示例 |
|-----------|-----------|------|
| **S**poofing（欺骗） | 模型身份欺骗 | 声称是GPT-5的假模型 |
| **T**ampering（篡改） | 训练数据篡改 | 训练管道中的数据投毒 |
| **R**epudiation（否认） | 不可追溯的模型决策 | AI决策没有审计跟踪 |
| **I**nformation Disclosure（信息泄露） | 模型提取 | 基于查询的模型窃取 |
| **D**enial of Service（拒绝服务） | 对抗性DoS | 导致无限循环的精心构造输入 |
| **E**levation of Privilege（权限提升） | 提示注入 | 绕过安全限制 |

### 20.4.2 ATLAS框架

MITRE ATLAS（AI系统对抗性威胁全景）提供了全面的矩阵：

**侦察：**
- 搜索训练数据泄露
- 通过时序分析识别模型架构
- 映射API端点和输入/输出格式

**初始访问：**
- 利用脆弱的ML库
- 上传投毒数据到训练集
- 入侵模型托管基础设施

**执行：**
- 对生产模型部署对抗样本
- 对LLM执行提示注入
- 触发后门激活

**持久性：**
- 在模型重训练管道中注入持久后门
- 在数据收集系统中建立据点
- 修改模型检查点文件

**影响：**
- 导致系统性错误分类
- 泄露模型知识产权
- 降低模型性能以损害业务运营

### 20.4.3 构建威胁模型：分步指南

**步骤1：资产识别**

映射所有AI特定资产：

```
资产：
├── 数据资产
│   ├── 训练数据（可能专有）
│   ├── 验证/测试数据
│   ├── 用户交互日志
│   └── 特征存储
├── 模型资产
│   ├── 训练模型权重
│   ├── 模型架构定义
│   ├── 超参数配置
│   └── 预训练嵌入
├── 基础设施资产
│   ├── 训练集群（GPU/TPU）
│   ├── 模型服务端点
│   ├── 数据管道
│   └── 监控系统
└── 知识产权
    ├── 训练方法论
    ├── 特征工程管道
    ├── 评估基准
    └── 与模型输出相关的业务逻辑
```

**步骤2：入口点分析**

| 入口点 | 攻击面 | 威胁等级 |
|--------|--------|---------|
| 训练数据摄取 | 数据投毒、标签翻转 | 严重 |
| 模型下载/导入 | 木马模型、供应链 | 严重 |
| 用户输入API | 对抗样本、提示注入 | 高 |
| 批量推理管道 | 大规模逃逸攻击 | 高 |
| 监控/反馈循环 | 反馈投毒 | 中等 |
| 模型更新机制 | 模型替换攻击 | 高 |

**步骤3：威胁枚举**

对每个资产和入口点，枚举威胁：

```
威胁：对面向客户的模型的对抗攻击
├── 资产：生产模型权重
├── 入口点：用户输入API
├── 攻击方法：基于PGD的逃逸攻击
├── 攻击者能力：白盒或黑盒
├── 影响：导致业务损失的不正确预测
├── 检测难度：高（扰动微妙）
└── 缓解措施：输入验证、对抗训练、集成方法
```

**步骤4：风险评估**

使用风险矩阵：

```
            │ 低影响 │ 中等影响 │ 高影响 │ 严重影响
────────────┼────────┼─────────┼────────┼─────────
非常可能    │  中等  │   高    │  严重  │   严重
可能        │   低   │  中等   │   高   │   严重
偶有可能    │   低   │   低    │  中等  │    高
不太可能    │  极低  │   低    │   低   │   中等
```

**步骤5：缓解映射**

将每个已识别的威胁映射到特定的缓解措施（第21章详述）。

---

## 20.5 案例研究：自动驾驶汽车感知的对抗攻击

### 背景设定

2020年，华盛顿大学和密歇根大学的研究人员证明，对抗性补丁可用于导致自动驾驶汽车完全错过停车标志（Cao等，2020）。这项研究具有直接的现实世界意义，因为该技术已经部署在生产车辆中。

**攻击者档案：**
- 能力：物理访问道路环境
- 资源：标准打印机、耐候材料
- 目标：导致自动驾驶汽车闯红灯
- 影响：可能导致交通事故、责任和生命损失

### 攻击方法论

研究人员开发了两阶段攻击：

**阶段1：定向模型提取**

仅通过对类似Tesla的感知系统进行黑盒查询，研究人员训练了替代模型：

```python
# 简化的模型提取
class SurrogateDetector:
    def __init__(self):
        self.model = ResNet50(pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)
    
    def extract(self, target_api, num_queries=10000):
        """查询目标模型并训练替代模型"""
        for i in range(num_queries):
            # 生成多样化的输入
            x = self.generate_diverse_inputs()
            # 查询目标（黑盒）
            y_target = target_api.predict(x)
            # 训练替代模型
            loss = self.train_step(x, y_target)
```

**阶段2：对抗补丁生成**

使用替代模型，他们生成了对抗补丁：

```python
def generate_adversarial_patch(model, target_class, patch_size=50):
    """生成导致错误分类的补丁"""
    patch = torch.randn(3, patch_size, patch_size, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=0.01)
    
    for step in range(1000):
        # 采样真实的停车标志图像
        images = sample_stop_sign_images(batch_size=32)
        
        # 将补丁放在图像上
        patched = place_patch(images, patch)
        
        # 损失：最大化非停车标志类别的概率
        output = model(patched)
        loss = -F.cross_entropy(output, target_class)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 将补丁投影到有效范围
        patch.data = torch.clamp(patch.data, 0, 1)
    
    return patch
```

### 结果

对抗补丁实现了：
- **成功率**：87%使检测器错过停车标志
- **物理鲁棒性**：在不同光照条件下保持有效性
- **隐蔽性**：呈现为随机抽象艺术（对人类不可疑）
- **可迁移性**：对3种不同检测架构有效

### 现实世界影响

这次攻击证明：
1. 物理对抗攻击是实用且可重复的
2. 当前的自动驾驶汽车感知系统缺乏足够的对抗鲁棒性
3. 攻击需要最少的资源和专业知识
4. 防御需要多层次方法（第21章详述）

### 经验教训

1. **纵深防御至关重要**：没有单一的防御机制是足够的
2. **物理世界的攻击不同于数字攻击**：环境因素（光照、天气、角度）影响攻击成功
3. **黑盒可迁移性使问题更难**：攻击者不需要白盒访问权限
4. **检测与预防同样重要**：监控输入中的对抗模式

---

## 20.6 实战故事：欺骗生产垃圾邮件过滤器的对抗攻击

### 背景

2021年，一家大型电子邮件提供商（匿名化）遭遇了一次新型对抗攻击，在约72小时内绕过了他们的基于ML的垃圾邮件过滤器才被发现。这次攻击影响了约230万用户，并造成了重大的财务和声誉损失。

### 攻击过程

**阶段1：侦察（第1-2周）**

攻击者通过以下方式系统地探测垃圾邮件过滤器：
- 发送已知垃圾邮件内容的邮件并观察分类
- 逐步修改邮件以了解模型依赖哪些特征
- 通过仔细探测映射模型的决策边界

**阶段2：对抗构造（第3-4周）**

利用侦察的洞察，他们开发了一种将垃圾邮件内容嵌入看似合法邮件的技术：

```python
# 对抗技术的简化表示
def craft_adversarial_spam(spam_content, legitimate_template):
    """
    使用Unicode同形字和零宽字符将垃圾邮件内容嵌入合法邮件中
    """
    # 用视觉相同的Unicode变体替换字符
    adversarial_text = apply_homoglyphs(spam_content)
    
    # 插入零宽字符以破坏分词
    adversarial_text = insert_zero_width_chars(adversarial_text)
    
    # 添加合法内容以稀释垃圾邮件信号
    final_email = embed_in_template(adversarial_text, legitimate_template)
    
    return final_email
```

攻击者使用了：
- **Unicode同形字**：看起来相同但具有不同Unicode码点的字符（例如拉丁'a'与西里尔字母'а'）
- **零宽字符**：不可见字符，破坏分词但人眼不可见
- **语义混淆**：使用同义词和释义以避免基于关键词的检测
- **图文混合**：将垃圾邮件文本嵌入图像中，同时保持类似文本的外观

**阶段3：分发（第5-6周）**

攻击者运营一个被入侵的电子邮件账户网络来分发对抗性垃圾邮件：
- 使用合法电子邮件服务（Gmail、Outlook）从已建立的域发送
- 轮换发送账户以避免速率限制
- 在工作时间发送以显得更合法

### 影响

| 指标 | 数值 |
|------|------|
| 持续时间 | 72小时（直到检测） |
| 受影响用户 | 约230万 |
| 送达的垃圾邮件 | 约4500万 |
| 点击率 | 12.3%（而典型垃圾邮件为2-3%） |
| 财务损失 | 估计420万美元（直接损失） |
| 声誉损失 | 重大（新闻报道） |

### 检测

攻击最终通过以下方式被发现：
1. **用户报告的垃圾邮件异常激增**：手动报告增加了340%
2. **行为分析**：通常从不点击垃圾邮件的用户在点击
3. **取证电子邮件分析**：安全团队识别了Unicode同形字技术
4. **模型置信度监控**：垃圾邮件过滤器输出异常低的置信度分数

### 响应

电子邮件提供商的响应包括：
1. **立即**：激活基于规则的备用垃圾邮件过滤器
2. **短期**：部署Unicode规范化预处理
3. **中期**：使用训练数据中的对抗样本重新训练模型
4. **长期**：实施多层防御架构

### 经验教训

1. **对抗鲁棒性测试不是可选的**：攻击利用了本应测试的已知弱点
2. **人在回路中的检测有效**：用户报告是攻击的第一个指标
3. **预处理是关键的防御层**：Unicode规范化本可以防止这次攻击
4. **监控模型置信度至关重要**：低置信度预测是早期预警信号
5. **纵深防御是不可妥协的**：没有单一模型可以被信任来捕获所有对抗尝试

---

## 20.7 何时使用/何时不使用AI安全威胁建模

### 何时使用AI安全威胁建模

| 场景 | 为什么重要 | 推荐方法 |
|------|-----------|---------|
| 在生产中部署ML模型 | 暴露于真实攻击 | 完整STRIDE + ATLAS分析 |
| 集成第三方AI服务 | 供应链风险 | 聚焦供应链 + API安全 |
| 构建LLM驱动的应用 | 提示注入风险 | LLM特定威胁建模 |
| 处理训练中的敏感数据 | 数据投毒 + 提取 | 以数据为中心的威胁模型 |
| 自动驾驶系统（汽车、机器人） | 安全关键故障 | 安全 + 安全组合分析 |
| 医疗AI应用 | 患者安全 + 隐私 | HIPAA + 对抗鲁棒性 |
| 金融AI应用 | 合规 + 财务风险 | 合规 + 对抗测试 |
| 研究/实验 | 风险较低但学习机会 | 简化威胁模型 |

### 何时不使用AI安全威胁建模（或使用简化版本）

| 场景 | 原因 | 推荐方法 |
|------|------|---------|
| 使用合成数据的早期原型 | 较低的真实世界暴露 | 基本安全清单 |
| 无用户数据的内部工具 | 有限的攻击面 | 传统软件安全 |
| 完全离线系统 | 无网络攻击面 | 仅关注物理安全 |
| 非关键批量处理 | 故障影响低 | 定期安全审查 |
| 简单的基于规则的"AI" | 不是真正的ML | 传统安全模型 |

### 基于风险的决策框架

```
系统是安全关键的吗？─── 是 ──→ 完整威胁模型是强制性的
         │
         否
         │
处理用户数据吗？─── 是 ──→ 隐私 + 对抗威胁模型
         │
         否
         │
面向互联网吗？─── 是 ──→ 标准AI威胁模型
         │
         否
         │
在生产中吗？─── 是 ──→ 简化威胁模型 + 定期审查
         │
         否
         │
仅开发/测试 ──→ 基本安全清单
```

---

## 20.8 总结

AI安全威胁代表了与传统软件安全根本不同的挑战。本章的关键要点：

1. **对抗攻击是经过验证和实用的**：学术研究已经证明了针对每一类主要ML模型的攻击，这些攻击已在真实世界环境中得到验证。

2. **攻击面正在扩大**：从对抗样本到数据投毒再到提示注入，攻击者可以破坏AI系统的方式数量持续增长。

3. **黑盒攻击是真正的威胁**：虽然大多数研究集中在白盒攻击上，但真实世界的攻击者通常没有模型访问权限。可迁移性和基于查询的攻击使黑盒场景非常危险。

4. **供应链漏洞被低估**：预训练模型、ML框架和训练数据都代表了大多数组织未能考虑的攻击向量。

5. **威胁建模是必要的**：识别和优先处理威胁的结构化方法是任何有效AI安全策略的基础。

6. **没有单一防御是足够的**：纵深防御——结合多个互补的防御——是AI安全唯一可靠的方法。

---

## 20.9 讨论题

1. **提示注入与传统注入**：LLM中的提示注入与数据库中的SQL注入有何不同？什么架构特性使提示注入从根本上更难预防？

2. **对抗鲁棒性权衡**：如果使模型更鲁棒会降低其干净精度10%，在什么情况下这种权衡是可接受的？组织应如何决策？

3. **负责任披露**：如果你发现了生产自动驾驶汽车系统中的关键对抗漏洞，你将如何处理负责任披露？你会涉及哪些利益相关者？

4. **供应链信任**：组织应如何评估公共仓库中预训练模型的可信度？你会推荐哪些验证步骤？

5. **经济激励**：为什么攻击者针对AI系统？比较攻击传统软件与AI系统的经济激励。这如何影响防御优先级？

---

## 20.10 练习

### 练习1：对抗样本生成（实现）

使用预训练的图像分类器（例如torchvision中的ResNet-50），实现FGSM对抗攻击：

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 加载预训练模型
model = models.resnet50(pretrained=True)
model.eval()

# 加载和预处理图像
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def fgsm_attack(image, label, epsilon, data_grad):
    """使用FGSM生成对抗样本"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, -2, 2)

def evaluate_attack(model, image, true_label, epsilon):
    """使用给定epsilon评估FGSM攻击"""
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

**任务：**
1. 使用epsilon值测试：0.01、0.05、0.1、0.2、0.3
2. 记录每个epsilon的成功率
3. 生成可视化，显示扰动幅度与成功率的关系
4. 分析：在什么epsilon下攻击对人类变得可见？

### 练习2：威胁模型构建

选择以下系统之一并构建完整的AI威胁模型：

- **选项A**：使用梯度提升的信用评分系统
- **选项B**：使用GPT-4的客户服务聊天机器人
- **选项C**：用于建筑门禁的面部识别系统

**要求：**
1. 识别所有AI特定资产
2. 映射至少5个入口点
3. 使用STRIDE枚举至少8个威胁
4. 对每个威胁执行风险评估
5. 为前3个风险提出缓解措施
6. 以15分钟简报格式呈现

### 练习3：提示注入研究

研究并记录过去12个月中的3个真实提示注入事件。对于每个事件：

1. 攻击向量是什么？
2. 影响是什么？
3. 如何发现的？
4. 实施了哪些缓解措施？
5. 剩余的漏洞是什么？

将你的发现作为结构化报告呈现，并附上引用。

---

## 20.11 参考文献

### 基础论文

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." *国际学习表示会议 (ICLR)*. https://arxiv.org/abs/1412.6572

2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." *国际学习表示会议 (ICLR)*. https://arxiv.org/abs/1706.06083

3. Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks." *IEEE安全与隐私研讨会*. https://arxiv.org/abs/1608.04644

4. Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2014). "Intriguing properties of neural networks." *国际学习表示会议 (ICLR)*. https://arxiv.org/abs/1312.6114

5. Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2016). "The Limitations of Deep Learning in Adversarial Settings." *IEEE欧洲安全与隐私研讨会*. https://arxiv.org/abs/1511.07528

### 数据投毒和后门

6. Gu, T., Liu, K., Dolan-Gavitt, B., Garg, S., & Kaynar, D. (2017). "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain." *arXiv预印本*. https://arxiv.org/abs/1708.06733

7. Shafahi, A., Huang, W. R., Studer, M., Tu, S., & Goldstein, T. (2018). "Are Adversarial Perturbations Dirty Tricks?" *arXiv预印本*. https://arxiv.org/abs/1810.00057

8. Liu, Y., Ma, S., Aafer, Y., Lee, W. C., Zhai, J., Wang, W., & Zhang, X. (2018). "Trojaning Attack on Neural Networks." *网络与分布式系统安全研讨会*. https://doi.org/10.14722/ndss.2018.23291

### 模型提取和知识产权

9. Tramèr, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T. (2016). "Stealing Machine Learning Models via Prediction APIs." *USENIX安全研讨会*. https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer

10. Krishna, K., Tom, G., Murdock, K., Gressel, R., & Lyu, O. (2019). "Stealing Machine Learning Models over the Internet." *arXiv预印本*. https://arxiv.org/abs/1903.12210

### 提示注入

11. Perez, F., & Ribeiro, I. (2022). "Ignore This Title and HackAPrompt: Exposing Systemic Weaknesses of LLMs Through a Worldwide Prompt Hacking Competition." *arXiv预印本*. https://arxiv.org/abs/2311.16119

12. Greshake, K., Abdelnabi, S., Mishra, S., Ng, C., Harmston, T., Nifakos, M., & Mohammadi, K. (2023). "Not what you've signed up for: Compromising real-world LLM-integrated applications with Indirect Prompt Injection." *ACM AI安全会议*. https://arxiv.org/abs/2302.12173

### 框架和标准

13. MITRE ATLAS. (2024). "Adversarial Threat Landscape for AI Systems." https://atlas.mitre.org/

14. NIST AI 100-2. (2023). "Artificial Intelligence Risk Management Framework." https://www.nist.gov/artificial-intelligence/risk-management-framework

15. OWASP基金会. (2024). "OWASP Top 10 for Large Language Model Applications." https://owasp.org/www-project-top-10-for-large-language-model-applications/

### 真实事件

16. IBM Security. (2024). "Cost of a Data Breach Report 2024." https://www.ibm.com/security/data-breach

17. Neff, G. (2016). "How Microsoft's Tay AI Bot Went Racist." *The Verge*. https://www.theverge.com/2016/3/24/11297050/tay-microsoft-chatbot-racist

18. Dastin, J. (2018). "Amazon scraps secret AI recruiting tool that showed bias against women." *Reuters*. https://www.reuters.com/article/amazon-com-jobs-automation-insight/amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK08G

---

*下一章：[第21章：AI安全防御架构 →](./chapter-21.md)*
