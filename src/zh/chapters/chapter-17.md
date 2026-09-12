# 第17章：边缘 AI 基础

🟢 入门 | 🟡 中级 | 🔴 高级 | ⚫ 管理者

---

## 17.1 边缘计算概述

### 什么是边缘计算

边缘计算将计算和数据存储移近数据源，减少延迟和带宽使用。对于 AI 工作负载，这意味着直接在边缘设备上运行推理，而不是将数据发送到云端。

```
┌─────────────────────────────────────────────────────────────────┐
│              边缘计算架构                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    云层                                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  模型    │  │  数据湖  │  │  训练    │  │全局  │  │   │
│  │  │  训练    │  │          │  │  集群    │  │管理  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │   互联网 /        │                      │
│                    │   5G / 卫星       │                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   边缘层                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  边缘    │  │  边缘    │  │  边缘    │  │边缘  │  │   │
│  │  │ 服务器 1 │  │ 服务器 2 │  │ 设备 1   │  │设备 2│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  终端设备层                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  摄像头  │  │  传感器  │  │  IoT     │  │移动  │  │   │
│  │  │  (4K)    │  │  阵列    │  │  网关    │  │应用  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么需要边缘 AI

📌 **关键概念**：边缘 AI 实现低延迟的实时推理、增强的隐私保护、降低的带宽成本和离线操作能力。

| 优势 | 描述 | 应用场景 |
|------|------|----------|
| **低延迟** | 亚毫秒级响应时间 | 自动驾驶、工业控制 |
| **隐私** | 数据留在设备上 | 医疗保健、金融服务 |
| **带宽** | 减少数据传输 | 智慧城市、IoT 部署 |
| **离线** | 无需互联网即可工作 | 偏远地区、灾害响应 |
| **成本** | 降低云计算成本 | 高容量推理工作负载 |

### 边缘 AI 分类

```
┌─────────────────────────────────────────────────────────────────┐
│                    边缘 AI 分类                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  按部署位置：                                                   │
│  ├── 设备边缘（在设备本身）                                     │
│  ├── 近边缘（本地服务器）                                       │
│  └── 远边缘（区域数据中心）                                     │
│                                                                 │
│  按计算能力：                                                   │
│  ├── 微型（ARM Cortex-M，<1MB RAM）                            │
│  ├── 小型（ARM Cortex-A，1-8GB RAM）                           │
│  ├── 中型（x86/ARM，8-64GB RAM）                               │
│  └── 大型（边缘服务器，64GB+ RAM）                              │
│                                                                 │
│  按 AI 工作负载：                                               │
│  ├── 仅推理                                                     │
│  ├── 轻量微调                                                   │
│  ├── 联邦学习参与者                                              │
│  └── 在线学习                                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 17.2 边缘 AI 的应用场景

### 制造业与工业

```
┌─────────────────────────────────────────────────────────────────┐
│              工业边缘 AI                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  生产线                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  摄像头  │  │  摄像头  │  │  摄像头  │  │传感器│  │   │
│  │  │  (质检)  │  │  (安全)  │  │  (计数)  │  │阵列  │  │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──┬───┘  │   │
│  │       │             │             │             │       │   │
│  │       └─────────────┴─────────────┴─────────────┘       │   │
│  │                          │                               │   │
│  │                    ┌─────┴─────┐                        │   │
│  │                    │边缘服务器  │                        │   │
│  │                    │┌────────┐│                        │   │
│  │                    ││GPU:    ││                        │   │
│  │                    ││4xT4    ││                        │   │
│  │                    │└────────┘│                        │   │
│  │                    └──────────┘                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  应用：                                                        │
│  • 缺陷检测（视觉检查）                                        │
│  • 预测性维护                                                   │
│  • 工人安全监控                                                 │
│  • 生产计数和分析                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 医疗保健

```yaml
# 医疗边缘 AI 架构
healthcare_edge_ai:
  use_cases:
    - name: "医学影像"
      description: "在护理点进行实时 X 光/MRI 分析"
      latency_requirement: "<100ms"
      accuracy_requirement: ">95%"
      hardware: "NVIDIA Jetson AGX Xavier"
      
    - name: "患者监测"
      description: "持续生命体征分析"
      latency_requirement: "<10ms"
      accuracy_requirement: ">99%"
      hardware: "ARM Cortex-A72 + Coral TPU"
      
    - name: "药物发现"
      description: "跨医院的联邦学习"
      privacy_requirement: "HIPAA 合规"
      hardware: "本地 GPU 服务器"
  
  architecture:
    - tier: "设备层"
      components: ["可穿戴传感器", "医疗设备"]
      compute: "微控制器"
      
    - tier: "边缘服务器"
      components: ["GPU 服务器", "存储"]
      compute: "NVIDIA T4/A30"
      
    - tier: "云层"
      components: ["训练集群", "模型注册表"]
      compute: "NVIDIA A100"
```

### 智慧城市

```
┌─────────────────────────────────────────────────────────────────┐
│              智慧城市边缘 AI                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  城市基础设施                             │   │
│  │                                                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │ 交通    │  │ 公共    │  │ 废弃物  │  │能源  │  │   │
│  │  │ 信号灯  │  │ 安全    │  │ 管理    │  │电网  │  │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──┬───┘  │   │
│  │       │             │             │             │       │   │
│  │       └─────────────┴─────────────┴─────────────┘       │   │
│  │                          │                               │   │
│  │                    ┌─────┴─────┐                        │   │
│  │                    │  城市     │                        │   │
│  │                    │  边缘     │                        │   │
│  │                    │  中心     │                        │   │
│  │                    └───────────┘                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  应用：                                                        │
│  • 交通流量优化                                                 │
│  • 事件检测和响应                                               │
│  • 环境监测                                                     │
│  • 能源消耗优化                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 17.3 边缘 vs 云端架构权衡

### 决策框架

```
┌─────────────────────────────────────────────────────────────────┐
│           边缘 vs 云端决策矩阵                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  因素              │ 边缘          │ 云端          │ 决策      │
│  ─────────────────────────────────────────────────────────────  │
│  延迟              │ <10ms         │ 50-200ms      │ 边缘      │
│  带宽              │ 低            │ 高            │ 边缘      │
│  隐私              │ 高            │ 可变          │ 边缘      │
│  计算能力          │ 有限          │ 无限          │ 云端      │
│  存储              │ 有限          │ 无限          │ 云端      │
│  模型更新          │ 复杂          │ 简单          │ 云端      │
│  维护              │ 分布式        │ 集中          │ 云端      │
│  成本模型          │ 资本支出      │ 运营支出      │ 视情况    │
│  离线操作          │ 是            │ 否            │ 边缘      │
│  可扩展性          │ 水平          │ 垂直          │ 两者皆可  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 混合架构模式

```yaml
# 混合边缘-云架构
architecture_patterns:
  pattern_1_cloud_offloading:
    name: "云端卸载"
    description: "边缘设备预处理，云端处理"
    flow: "设备 → 边缘（预处理）→ 云端（推理）→ 设备"
    use_cases:
      - 无法在边缘运行的复杂模型
      - 批处理需求
      - 模型训练
    latency: "100-500ms"
    
  pattern_2_edge_inference:
    name: "边缘推理"
    description: "模型完全在边缘运行"
    flow: "设备 → 边缘（推理）→ 设备"
    use_cases:
      - 实时应用
      - 隐私敏感数据
      - 离线需求
    latency: "<10ms"
    
  pattern_3_split_inference:
    name: "分割推理"
    description: "模型在边缘和云端之间分割"
    flow: "设备 → 边缘（早期层）→ 云端（后期层）→ 设备"
    use_cases:
      - 具有延迟要求的大型模型
      - 带宽受限环境
      - 渐进式优化
    latency: "20-100ms"
    
  pattern_4_federated_learning:
    name: "联邦学习"
    description: "训练分布在边缘设备之间"
    flow: "设备（训练）→ 边缘（聚合）→ 云端（全局模型）"
    use_cases:
      - 隐私保护 ML
      - 个性化
      - 法规合规
    latency: "可变"
```

---

## 17.4 边缘硬件平台对比

### 硬件平台概览

```
┌─────────────────────────────────────────────────────────────────┐
│              边缘硬件平台                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  NVIDIA Jetson 系列                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Nano    │  │  TX2     │  │  Xavier  │  │Orin  │  │   │
│  │  │ 472 GFLOPS│  │1.3 TFLOPS│  │32 TFLOPS │ │275   │  │   │
│  │  │ 10W      │  │ 15W      │  │ 30W      │  │TFLOPS│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  │60W   │  │   │
│  │                                             └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Google Coral 系列                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  USB     │  │  开发板  │  │  M.2     │  │SoM   │  │   │
│  │  │ 4 TOPS   │  │ 4 TOPS   │  │ 4 TOPS   │  │8 TOPS│  │   │
│  │  │ 2W       │  │ 5W       │  │ 2W       │  │5W    │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Intel Movidius 系列                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Myriad  │  │  Myriad  │  │  Habana  │  │Gaudi │  │   │
│  │  │  X       │  │  2       │  │  Goya    │  │      │  │   │
│  │  │ 4 TOPS   │  │ 4 TOPS   │  │ 265 TOPS │  │      │  │   │
│  │  │ 1.5W     │  │ 2W       │  │ 75W      │  │      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 详细对比表

```yaml
# 边缘硬件对比
edge_hardware:
  nvidia_jetson:
    orin_nano:
      gpu_cores: 1024
      cuda_cores: 1024
      tensor_cores: 32
      memory: "8GB LPDDR5"
      power: "7-15W"
      performance: "40 TOPS"
      price: "$249"
      use_cases: ["IoT", "机器人", "智能摄像头"]
      
    orin_xavier:
      gpu_cores: 2048
      cuda_cores: 2048
      tensor_cores: 64
      memory: "32GB LPDDR5"
      power: "15-60W"
      performance: "275 TOPS"
      price: "$999"
      use_cases: ["自动驾驶", "医学影像"]
      
    agx_orin:
      gpu_cores: 2048
      cuda_cores: 2048
      tensor_cores: 64
      memory: "64GB LPDDR5"
      power: "15-60W"
      performance: "275 TOPS"
      price: "$1999"
      use_cases: ["机器人", "边缘服务器"]
      
  google_coral:
    dev_board:
      tpu: "Google Edge TPU"
      performance: "4 TOPS"
      memory: "1GB LPDDR4"
      power: "5W"
      price: "$149"
      use_cases: ["图像分类", "目标检测"]
      
    usb_accelerator:
      tpu: "Google Edge TPU"
      performance: "4 TOPS"
      interface: "USB 3.0"
      power: "2W"
      price: "$59"
      use_cases: ["基于 USB 的推理"]
      
  intel_movidius:
    neural_compute_stick_2:
      vpu: "Intel Movidius X"
      performance: "4 TOPS"
      interface: "USB 3.0"
      power: "1.5W"
      price: "$69"
      use_cases: ["原型设计", "低功耗推理"]
```

### 平台选择指南

```
┌─────────────────────────────────────────────────────────────────┐
│           边缘平台选择指南                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  需求                  │ 推荐平台                               │
│  ─────────────────────────────────────────────────────────────  │
│  超低功耗 (<1W)        │ Intel Movidius / Coral USB            │
│  高性能                │ NVIDIA Jetson AGX Orin                │
│  经济高效              │ NVIDIA Jetson Nano / Coral Dev        │
│  生产部署              │ NVIDIA Jetson Xavier NX               │
│  原型设计              │ Coral Dev Board / Jetson Nano         │
│  视觉应用              │ 所有（均支持 CNN 推理）               │
│  NLP 应用              │ Jetson Xavier/Orin（更大内存）        │
│  多模态                │ Jetson AGX Orin（最高性能）           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💡 案例研究：智能工厂边缘 AI

### 完整架构

🟡 中级

```yaml
# 智能工厂边缘 AI 部署
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: edge-ai-agent
  namespace: factory-floor
spec:
  selector:
    matchLabels:
      app: edge-ai-agent
  template:
    metadata:
      labels:
        app: edge-ai-agent
    spec:
      containers:
      - name: inference-engine
        image: factory/edge-inference:latest
        resources:
          requests:
            nvidia.com/gpu: "1"
            memory: "4Gi"
            cpu: "2"
          limits:
            nvidia.com/gpu: "1"
            memory: "8Gi"
            cpu: "4"
        env:
        - name: CAMERA_STREAMS
          value: "rtsp://camera1:554/stream,rtsp://camera2:554/stream"
        - name: MODEL_PATH
          value: "/models/defect_detection"
        - name: INFERENCE_THRESHOLD
          value: "0.85"
        - name: CLOUD_UPLOAD
          value: "true"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: config
          mountPath: /config
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: edge-models-pvc
      - name: config
        configMap:
          name: edge-config
      nodeSelector:
        node-type: edge-gpu
      tolerations:
      - key: "factory-floor"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

### 模型更新 Operator

```python
# 边缘模型更新 operator
import kubernetes
from kubernetes import client, config
import requests
import hashlib

class EdgeModelUpdater:
    def __init__(self):
        config.load_incluster_config()
        self.k8s = client.CoreV1Api()
        self.model_registry = "http://model-registry:5000"
        
    def check_for_updates(self, model_name, current_version):
        """检查是否有模型更新。"""
        response = requests.get(
            f"{self.model_registry}/api/models/{model_name}/versions"
        )
        versions = response.json()
        
        # 找到最新版本
        latest = max(versions, key=lambda v: v['version'])
        
        if latest['version'] != current_version:
            return latest
        return None
    
    def download_model(self, model_url, destination):
        """下载模型到边缘设备。"""
        response = requests.get(model_url, stream=True)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # 验证校验和
        sha256 = hashlib.sha256()
        with open(destination, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def update_deployment(self, deployment_name, namespace, new_image):
        """使用新模型更新 Kubernetes 部署。"""
        apps_v1 = client.AppsV1Api()
        
        # 获取当前部署
        deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )
        
        # 更新容器镜像
        deployment.spec.template.spec.containers[0].image = new_image
        
        # 应用更新
        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=deployment
        )
        
        return True
```

---

## 📝 练习

### 练习 17.1：边缘架构设计
为连锁零售店设计边缘 AI 架构，要求：
1. 1000 个门店位置
2. 实时库存跟踪
3. 客户行为分析
4. 集中式模型管理
5. 离线操作能力

### 练习 17.2：平台对比
评估计算机视觉应用的边缘平台：
1. 在 Jetson Nano、Xavier NX 和 Coral Dev 上测试推理速度
2. 比较功耗
3. 评估模型兼容性（TensorFlow、PyTorch、ONNX）
4. 计算 100 台设备的总拥有成本

---

## ⚠️ 警告

1. **热管理**：边缘设备通常在恶劣环境中运行。确保适当的冷却和热节流保护。
2. **功耗限制**：许多边缘位置电力有限。选择满足功耗要求的平台。
3. **安全性**：边缘设备物理可访问。实现安全启动、加密和远程擦除功能。
4. **模型更新**：边缘部署需要可靠的模型更新机制。规划回滚场景。
5. **硬件过时**：边缘硬件的生命周期比云短。规划硬件刷新周期。

---

## 本章小结

本章介绍了边缘 AI 基础知识，包括：
1. 边缘计算概念和架构
2. 跨行业的实际边缘 AI 应用场景
3. 边缘 vs 云端权衡分析
4. 硬件平台对比（NVIDIA Jetson、Google Coral、Intel Movidius）
5. 混合部署策略

下一章我们将探讨用于边缘部署的模型压缩和优化技术。
