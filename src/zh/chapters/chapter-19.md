# 第19章：边缘部署架构

🟢 入门 | 🟡 中级 | 🔴 高级 | ⚫ 管理者

---

## 19.1 边缘推理框架

### 框架对比

```
┌─────────────────────────────────────────────────────────────────┐
│              边缘推理框架                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  框架               │ 硬件支持            │ 模型格式            │
│  ─────────────────────────────────────────────────────────────  │
│  ONNX Runtime       │ CPU, GPU, NPU      │ ONNX               │
│  TensorRT           │ NVIDIA GPU         │ ONNX, Caffe, PT    │
│  TFLite             │ CPU, GPU, Edge TPU │ TFLite, TF SavedM  │
│  OpenVINO           │ Intel CPU, VPU     │ ONNX, IR           │
│  CoreML             │ Apple Neural Eng.  │ CoreML, ONNX       │
│  NCNN               │ ARM CPU            │ Caffe, ONNX        │
│  MNN                │ ARM CPU, GPU       │ ONNX, TFLite       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 用于边缘的 ONNX Runtime

```python
# ONNX Runtime 边缘推理
import onnxruntime as ort
import numpy as np
import cv2

class EdgeInferenceEngine:
    def __init__(self, model_path, providers=['CPUExecutionProvider']):
        """初始化 ONNX Runtime 推理引擎。"""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def preprocess(self, image, input_size=(224, 224)):
        """预处理图像用于推理。"""
        # 调整大小
        img = cv2.resize(image, input_size)
        
        # BGR 转 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # 转置为 CHW
        img = img.transpose(2, 0, 1)
        
        # 添加批次维度
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def inference(self, input_data):
        """运行推理。"""
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return outputs[0]
    
    def postprocess(self, output, top_k=5):
        """后处理输出。"""
        # 应用 softmax
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / exp_output.sum()
        
        # 获取 top-k 预测
        top_k_indices = np.argsort(probabilities[0])[-top_k:][::-1]
        top_k_probs = probabilities[0][top_k_indices]
        
        return list(zip(top_k_indices.tolist(), top_k_probs.tolist()))
    
    def predict(self, image):
        """完整预测管道。"""
        input_data = self.preprocess(image)
        output = self.inference(input_data)
        predictions = self.postprocess(output)
        return predictions

# 使用示例
engine = EdgeInferenceEngine(
    model_path='resnet50_quantized.onnx',
    providers=['CPUExecutionProvider']
)

# 加载并预测
image = cv2.imread('test_image.jpg')
predictions = engine.predict(image)

for class_id, prob in predictions:
    print(f"类别 {class_id}: {prob:.4f}")
```

### TensorRT 优化

```python
# 用于边缘的 TensorRT 优化
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTInference:
    def __init__(self, engine_path):
        """初始化 TensorRT 推理引擎。"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 分配内存
        self._allocate_buffers()
    
    def _allocate_buffers(self):
        """为输入/输出分配 GPU 内存。"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            size = trt.volume(binding_shape)
            host_mem = cuda.pagelocked_empty(size, binding_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer(self, input_data):
        """运行 TensorRT 推理。"""
        # 将输入复制到主机内存
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # 传输输入到 GPU
        cuda.memcpy_htod(
            self.inputs[0]['device'],
            self.inputs[0]['host']
        )
        
        # 运行推理
        self.context.execute_v2(bindings=self.bindings)
        
        # 传回输出到主机
        cuda.memcpy_dtoh(
            self.outputs[0]['host'],
            self.outputs[0]['device']
        )
        
        return self.outputs[0]['host'].reshape(self.engine.get_binding_shape(1))

def build_tensorrt_engine(onnx_path, engine_path, fp16=True, max_batch_size=8):
    """从 ONNX 构建 TensorRT 引擎。"""
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
            return None
    
    # 配置构建器
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 设置优化配置文件
    profile = builder.create_optimization_profile()
    profile.set_shape(
        'input',
        min=(1, 3, 224, 224),
        opt=(max_batch_size // 2, 3, 224, 224),
        max=(max_batch_size, 3, 224, 224)
    )
    config.add_optimization_profile(profile)
    
    # 构建引擎
    engine = builder.build_engine(network, config)
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine
```

---

## 19.2 模型更新策略

### 更新架构

```
┌─────────────────────────────────────────────────────────────────┐
│              模型更新架构                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  云端（中心）                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  模型    │  │  更新    │  │  版本    │  │推送  │  │   │
│  │  │ 注册表   │  │  管理器  │  │  控制    │  │服务  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │   更新通道        │                      │
│                    │   (OTA/CDN)       │                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  边缘设备群                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  边缘    │  │  边缘    │  │  边缘    │  │边缘  │  │   │
│  │  │ 设备 1   │  │ 设备 2   │  │ 设备 3   │  │设备N │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### OTA 更新实现

```python
# OTA 模型更新系统
import requests
import hashlib
import json
import time
from pathlib import Path

class EdgeOTAUpdater:
    def __init__(self, device_id, registry_url, local_model_dir):
        self.device_id = device_id
        self.registry_url = registry_url
        self.local_model_dir = Path(local_model_dir)
        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备状态
        self.current_model_version = self._get_current_version()
        self.last_update_check = 0
        self.update_interval = 3600  # 每小时检查一次
    
    def _get_current_version(self):
        """获取当前模型版本。"""
        version_file = self.local_model_dir / 'version.json'
        if version_file.exists():
            with open(version_file, 'r') as f:
                return json.load(f)
        return {'version': '0.0.0', 'hash': None}
    
    def _save_version(self, version_info):
        """保存版本信息。"""
        version_file = self.local_model_dir / 'version.json'
        with open(version_file, 'w') as f:
            json.dump(version_info, f)
    
    def check_for_updates(self):
        """检查是否有更新可用。"""
        if time.time() - self.last_update_check < self.update_interval:
            return None
        
        self.last_update_check = time.time()
        
        try:
            response = requests.get(
                f"{self.registry_url}/api/models/device/{self.device_id}",
                timeout=30
            )
            response.raise_for_status()
            
            latest = response.json()
            
            if latest['version'] != self.current_model_version['version']:
                return latest
            
            return None
            
        except Exception as e:
            print(f"检查更新时出错：{e}")
            return None
    
    def download_model(self, model_url, expected_hash):
        """带完整性检查的模型下载。"""
        temp_path = self.local_model_dir / 'model_new.onnx'
        
        try:
            # 下载
            response = requests.get(model_url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 验证哈希
            actual_hash = self._compute_hash(temp_path)
            if actual_hash != expected_hash:
                raise ValueError(f"哈希不匹配：{actual_hash} != {expected_hash}")
            
            return temp_path
            
        except Exception as e:
            # 失败时清理
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _compute_hash(self, file_path):
        """计算文件的 SHA-256 哈希。"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def apply_update(self, new_model_path, version_info):
        """应用模型更新（支持回滚）。"""
        backup_dir = self.local_model_dir / 'backup'
        backup_dir.mkdir(exist_ok=True)
        
        # 备份当前模型
        current_model = self.local_model_dir / 'model.onnx'
        if current_model.exists():
            backup_model = backup_dir / f'model_{self.current_model_version["version"]}.onnx'
            if backup_model.exists():
                backup_model.unlink()
            current_model.rename(backup_model)
        
        # 应用新模型
        new_model = self.local_model_dir / 'model.onnx'
        new_model_path.rename(new_model)
        
        # 更新版本
        self._save_version(version_info)
        self.current_model_version = version_info
        
        print(f"模型已更新至版本 {version_info['version']}")
    
    def rollback(self):
        """回滚到之前的模型版本。"""
        backup_dir = self.local_model_dir / 'backup'
        
        # 找到最新备份
        backups = sorted(backup_dir.glob('model_*.onnx'))
        if not backups:
            raise ValueError("没有可用于回滚的备份")
        
        latest_backup = backups[-1]
        
        # 恢复
        current_model = self.local_model_dir / 'model.onnx'
        if current_model.exists():
            current_model.unlink()
        
        latest_backup.rename(current_model)
        
        # 更新版本
        version = latest_backup.stem.replace('model_', '')
        self._save_version({'version': version, 'hash': None})
        
        print(f"已回滚至版本 {version}")
    
    def update_loop(self):
        """主更新循环。"""
        while True:
            update = self.check_for_updates()
            
            if update:
                print(f"有可用更新：{update['version']}")
                
                try:
                    # 下载新模型
                    temp_path = self.download_model(
                        update['download_url'],
                        update['hash']
                    )
                    
                    # 应用更新
                    self.apply_update(temp_path, update)
                    
                except Exception as e:
                    print(f"更新失败：{e}")
                    # 继续使用当前模型
            
            time.sleep(60)  # 每分钟检查一次
```

---

## 19.3 边缘集群管理

### 边缘的 Kubernetes（K3s/kubeedge）

```yaml
# K3s 边缘集群配置
apiVersion: v1
kind: Namespace
metadata:
  name: edge-ai
  labels:
    edge-cluster: "true"
---
# 边缘设备群
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: edge-agent
  namespace: edge-ai
spec:
  selector:
    matchLabels:
      app: edge-agent
  template:
    metadata:
      labels:
        app: edge-agent
    spec:
      containers:
      - name: agent
        image: edge-agent:latest
        env:
        - name: CLOUD_ENDPOINT
          value: "https://cloud-gateway.company.com"
        - name: DEVICE_GROUP
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['device-group']
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: config
          mountPath: /config
      volumes:
      - name: model-storage
        hostPath:
          path: /var/lib/edge-models
          type: DirectoryOrCreate
      - name: config
        configMap:
          name: edge-config
      nodeSelector:
        node-type: edge
      tolerations:
      - key: "edge-node"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
---
# 边缘推理部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
  namespace: edge-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference-service
  template:
    metadata:
      labels:
        app: inference-service
    spec:
      containers:
      - name: inference
        image: edge-inference:latest
        ports:
        - containerPort: 8080
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
        - name: MODEL_PATH
          value: "/models/current"
        - name: MAX_BATCH_SIZE
          value: "8"
        - name: INFERENCE_TIMEOUT
          value: "100"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 边缘设备群管理

```python
# 边缘设备群管理系统
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class DeviceStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    UPDATING = "updating"
    ERROR = "error"

@dataclass
class EdgeDevice:
    device_id: str
    location: str
    status: DeviceStatus
    current_model_version: str
    last_heartbeat: datetime
    resource_usage: Dict[str, float]

class EdgeFleetManager:
    def __init__(self, cloud_endpoint: str):
        self.cloud_endpoint = cloud_endpoint
        self.devices: Dict[str, EdgeDevice] = {}
        self.update_schedule: Dict[str, datetime] = {}
    
    async def register_device(self, device_info: Dict):
        """注册新的边缘设备。"""
        device = EdgeDevice(
            device_id=device_info['device_id'],
            location=device_info['location'],
            status=DeviceStatus.ONLINE,
            current_model_version=device_info['model_version'],
            last_heartbeat=datetime.now(),
            resource_usage=device_info.get('resource_usage', {})
        )
        
        self.devices[device.device_id] = device
        
        # 通知云端
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.cloud_endpoint}/api/devices/register",
                json=device_info
            ) as response:
                return await response.json()
    
    async def heartbeat(self, device_id: str, status: Dict):
        """接收边缘设备的心跳。"""
        if device_id in self.devices:
            self.devices[device_id].last_heartbeat = datetime.now()
            self.devices[device_id].status = DeviceStatus.ONLINE
            self.devices[device_id].resource_usage = status.get('resource_usage', {})
    
    async def check_device_health(self):
        """检查所有设备的健康状况。"""
        now = datetime.now()
        unhealthy_devices = []
        
        for device_id, device in self.devices.items():
            # 检查心跳是否过期
            if (now - device.last_heartbeat) > timedelta(minutes=5):
                device.status = DeviceStatus.OFFLINE
                unhealthy_devices.append(device_id)
            
            # 检查资源使用
            if device.resource_usage.get('gpu_memory', 0) > 90:
                unhealthy_devices.append(device_id)
        
        return unhealthy_devices
    
    async def orchestrate_update(self, model_version: str, 
                                  strategy: str = 'rolling'):
        """在整个设备群中协调模型更新。"""
        devices = list(self.devices.keys())
        
        if strategy == 'rolling':
            # 每次更新 10%
            batch_size = max(1, len(devices) // 10)
            
            for i in range(0, len(devices), batch_size):
                batch = devices[i:i + batch_size]
                
                # 更新批次
                await self._update_batch(batch, model_version)
                
                # 等待批次完成
                await asyncio.sleep(60)
                
                # 检查健康状况
                unhealthy = await self.check_device_health()
                if unhealthy:
                    print(f"更新期间的不健康设备：{unhealthy}")
                    # 如果失败过多则回滚
                    if len(unhealthy) > batch_size * 0.5:
                        await self._rollback_batch(batch)
                        break
        
        elif strategy == 'canary':
            # 首先更新 1 台设备
            canary_device = devices[0]
            await self._update_batch([canary_device], model_version)
            
            # 等待并检查
            await asyncio.sleep(300)
            
            if canary_device not in await self.check_device_health():
                # 继续更新其余设备
                await self._update_batch(devices[1:], model_version)
            else:
                print("金丝雀更新失败，中止")
                await self._rollback_batch([canary_device])
    
    async def _update_batch(self, device_ids: List[str], model_version: str):
        """更新一批设备。"""
        for device_id in device_ids:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.cloud_endpoint}/api/devices/{device_id}/update",
                        json={'model_version': model_version}
                    ) as response:
                        if response.status == 200:
                            self.devices[device_id].status = DeviceStatus.UPDATING
                        else:
                            print(f"更新 {device_id} 失败")
            except Exception as e:
                print(f"更新 {device_id} 时出错：{e}")
    
    async def _rollback_batch(self, device_ids: List[str]):
        """回滚一批设备。"""
        for device_id in device_ids:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.cloud_endpoint}/api/devices/{device_id}/rollback"
                    ) as response:
                        if response.status == 200:
                            print(f"已回滚 {device_id}")
            except Exception as e:
                print(f"回滚 {device_id} 时出错：{e}")
    
    def get_fleet_status(self) -> Dict:
        """获取设备群整体状态。"""
        status_counts = {
            'total': len(self.devices),
            'online': sum(1 for d in self.devices.values() 
                         if d.status == DeviceStatus.ONLINE),
            'offline': sum(1 for d in self.devices.values() 
                          if d.status == DeviceStatus.OFFLINE),
            'updating': sum(1 for d in self.devices.values() 
                           if d.status == DeviceStatus.UPDATING),
            'error': sum(1 for d in self.devices.values() 
                        if d.status == DeviceStatus.ERROR)
        }
        
        return status_counts
```

---

## 19.4 离线推理架构

### 离线模式设计

```
┌─────────────────────────────────────────────────────────────────┐
│              离线推理架构                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  在线模式                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  输入    │  │  推理    │  │  输出    │  │云端  │  │   │
│  │  │  流      │→ │  引擎    │→ │  流      │→ │同步  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │   网络状态        │                      │
│                    │   检测            │                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  离线模式                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  输入    │  │  推理    │  │  本地    │  │队列  │  │   │
│  │  │  流      │→ │  引擎    │→ │  存储    │  │管理器│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  关键特性：                                                     │
│  • 本地模型缓存                                                │
│  • 结果排队等待同步                                             │
│  • 优雅降级                                                    │
│  • 连接恢复时自动同步                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 实现

```python
# 离线推理管理器
import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import aiohttp

class OfflineInferenceManager:
    def __init__(self, model_path: str, sync_endpoint: str, 
                 local_db_path: str = '/data/offline_queue.db'):
        self.model_path = model_path
        self.sync_endpoint = sync_endpoint
        self.local_db_path = local_db_path
        
        # 初始化本地数据库
        self._init_db()
        
        # 网络状态
        self.is_online = True
        self.sync_interval = 300  # 5 分钟
        self.max_queue_size = 10000
        
        # 加载模型
        self.model = self._load_model()
    
    def _init_db(self):
        """初始化本地 SQLite 数据库。"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inference_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_data TEXT NOT NULL,
                output_data TEXT,
                timestamp TEXT NOT NULL,
                synced BOOLEAN DEFAULT FALSE,
                model_version TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_model(self):
        """从本地存储加载模型。"""
        import onnxruntime as ort
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        
        return ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
    
    async def check_connectivity(self) -> bool:
        """检查云端是否可达。"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.sync_endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except:
            return False
    
    async def inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行带离线支持的推理。"""
        # 本地运行推理
        output = self._run_inference(input_data)
        
        # 如果离线则存储在队列中
        if not self.is_online:
            self._queue_result(input_data, output)
        
        # 如果在线则尝试同步
        if self.is_online:
            await self._try_sync()
        
        return output
    
    def _run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行本地推理。"""
        import numpy as np
        
        # 预处理输入
        input_tensor = self._preprocess(input_data)
        
        # 运行推理
        input_name = self.model.get_inputs()[0].name
        output = self.model.run(None, {input_name: input_tensor})[0]
        
        # 后处理输出
        result = self._postprocess(output)
        
        return result
    
    def _preprocess(self, input_data: Dict[str, Any]) -> np.ndarray:
        """预处理输入数据。"""
        # 示例：图像预处理
        if 'image' in input_data:
            import cv2
            img = cv2.imdecode(
                np.frombuffer(input_data['image'], np.uint8),
                cv2.IMREAD_COLOR
            )
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img = img.transpose(2, 0, 1)
            return np.expand_dims(img, axis=0)
        
        return input_data.get('tensor', np.array([]))
    
    def _postprocess(self, output: np.ndarray) -> Dict[str, Any]:
        """后处理推理输出。"""
        # 应用 softmax
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / exp_output.sum()
        
        return {
            'predictions': probabilities.tolist(),
            'timestamp': datetime.now().isoformat(),
            'model_version': self._get_model_version()
        }
    
    def _get_model_version(self) -> str:
        """获取当前模型版本。"""
        version_file = Path(self.model_path).parent / 'version.json'
        if version_file.exists():
            with open(version_file, 'r') as f:
                return json.load(f).get('version', 'unknown')
        return 'unknown'
    
    def _queue_result(self, input_data: Dict, output: Dict):
        """将结果排队等待后续同步。"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        # 检查队列大小
        cursor.execute('SELECT COUNT(*) FROM inference_queue WHERE synced = FALSE')
        queue_size = cursor.fetchone()[0]
        
        if queue_size >= self.max_queue_size:
            # 删除最旧的未同步结果
            cursor.execute('''
                DELETE FROM inference_queue 
                WHERE synced = FALSE 
                AND id IN (
                    SELECT id FROM inference_queue 
                    WHERE synced = FALSE 
                    ORDER BY timestamp ASC 
                    LIMIT 1000
                )
            ''')
        
        # 插入新结果
        cursor.execute('''
            INSERT INTO inference_queue (input_data, output_data, timestamp, model_version)
            VALUES (?, ?, ?, ?)
        ''', (
            json.dumps(input_data),
            json.dumps(output),
            datetime.now().isoformat(),
            self._get_model_version()
        ))
        
        conn.commit()
        conn.close()
    
    async def _try_sync(self):
        """尝试将排队结果同步到云端。"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        # 获取未同步的结果
        cursor.execute('''
            SELECT id, input_data, output_data, timestamp, model_version
            FROM inference_queue
            WHERE synced = FALSE
            ORDER BY timestamp ASC
            LIMIT 100
        ''')
        
        results = cursor.fetchall()
        
        if not results:
            conn.close()
            return
        
        # 同步到云端
        try:
            async with aiohttp.ClientSession() as session:
                for row in results:
                    record_id, input_data, output_data, timestamp, model_version = row
                    
                    await session.post(
                        f"{self.sync_endpoint}/api/inference/results",
                        json={
                            'input': json.loads(input_data),
                            'output': json.loads(output_data),
                            'timestamp': timestamp,
                            'model_version': model_version
                        }
                    )
                    
                    # 标记为已同步
                    cursor.execute(
                        'UPDATE inference_queue SET synced = TRUE WHERE id = ?',
                        (record_id,)
                    )
            
            conn.commit()
            
        except Exception as e:
            print(f"同步失败：{e}")
        
        finally:
            conn.close()
    
    async def sync_loop(self):
        """后台同步循环。"""
        while True:
            # 检查连接
            self.is_online = await self.check_connectivity()
            
            if self.is_online:
                await self._try_sync()
            
            await asyncio.sleep(self.sync_interval)
    
    def get_queue_status(self) -> Dict:
        """获取队列状态。"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM inference_queue WHERE synced = FALSE')
        unsynced = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM inference_queue WHERE synced = TRUE')
        synced = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'unsynced': unsynced,
            'synced': synced,
            'total': unsynced + synced,
            'is_online': self.is_online
        }
```

---

## 19.5 边缘-云协同架构

### 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│              边缘-云协同架构                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   云层                                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  模型    │  │  训练    │  │  设备群  │  │数据  │  │   │
│  │  │ 注册表   │  │  服务    │  │  管理器  │  │湖    │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │  通信层            │                      │
│                    │  (MQTT/gRPC/HTTP) │                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   边缘层                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  边缘    │  │  本地    │  │  推理    │  │数据  │  │   │
│  │  │ 代理     │  │  缓存    │  │  引擎    │  │过滤  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   设备层                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  传感器  │  │  摄像头  │  │  执行器  │  │用户  │  │   │
│  │  │  阵列    │  │  视频流  │  │  控制    │  │输入  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 通信协议

```python
# 边缘-云通信协议
import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from enum import Enum

class MessageType(Enum):
    HEARTBEAT = "heartbeat"
    MODEL_UPDATE = "model_update"
    INFERENCE_RESULT = "inference_result"
    TRAINING_DATA = "training_data"
    CONFIG_UPDATE = "config_update"
    ALERT = "alert"

class EdgeCloudProtocol:
    def __init__(self, device_id: str, cloud_endpoint: str):
        self.device_id = device_id
        self.cloud_endpoint = cloud_endpoint
        self.message_queue = asyncio.Queue()
        self.is_connected = False
        
    async def connect(self):
        """建立与云端的连接。"""
        # 实现取决于协议（MQTT、gRPC 等）
        self.is_connected = True
        
    async def send_message(self, msg_type: MessageType, payload: Dict[str, Any]):
        """向云端发送消息。"""
        message = {
            'device_id': self.device_id,
            'message_type': msg_type.value,
            'timestamp': datetime.now().isoformat(),
            'payload': payload
        }
        
        if self.is_connected:
            # 直接发送
            await self._send_to_cloud(message)
        else:
            # 排队等待后续发送
            await self.message_queue.put(message)
    
    async def receive_messages(self):
        """从云端接收消息。"""
        # 实现取决于协议
        pass
    
    async def _send_to_cloud(self, message: Dict):
        """内部发送实现。"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.cloud_endpoint}/api/messages",
                json=message
            ) as response:
                return await response.json()
    
    async def sync_queued_messages(self):
        """连接时同步排队的消息。"""
        while not self.message_queue.empty():
            message = await self.message_queue.get()
            try:
                await self._send_to_cloud(message)
            except Exception:
                # 失败时放回队列
                await self.message_queue.put(message)
                break
```

### 协同训练

```python
# 用于边缘-云协同的联邦学习
import torch
import torch.nn as nn
from typing import List, Dict
import asyncio

class FederatedLearningCoordinator:
    def __init__(self, global_model: nn.Module, num_clients: int):
        self.global_model = global_model
        self.num_clients = num_clients
        self.round_number = 0
        self.client_updates = []
    
    async def coordinate_round(self, edge_clients: List):
        """协调一轮联邦学习。"""
        self.round_number += 1
        
        # 发送全局模型到客户端
        global_weights = self.global_model.state_dict()
        
        # 从客户端收集更新
        updates = []
        for client in edge_clients:
            update = await client.train_locally(global_weights)
            updates.append(update)
        
        # 聚合更新
        aggregated_weights = self._aggregate_updates(updates)
        
        # 更新全局模型
        self.global_model.load_state_dict(aggregated_weights)
        
        return aggregated_weights
    
    def _aggregate_updates(self, updates: List[Dict]) -> Dict:
        """使用 FedAvg 聚合客户端更新。"""
        averaged_weights = {}
        
        for key in updates[0].keys():
            averaged_weights[key] = torch.zeros_like(updates[0][key])
            
            for update in updates:
                averaged_weights[key] += update[key]
            
            averaged_weights[key] /= len(updates)
        
        return averaged_weights

class EdgeFederatedClient:
    def __init__(self, local_data, local_model: nn.Module, 
                 local_epochs: int = 5):
        self.local_data = local_data
        self.local_model = local_model
        self.local_epochs = local_epochs
    
    async def train_locally(self, global_weights: Dict) -> Dict:
        """本地训练并返回更新。"""
        # 加载全局权重
        self.local_model.load_state_dict(global_weights)
        
        # 本地训练
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        self.local_model.train()
        for epoch in range(self.local_epochs):
            for data, labels in self.local_data:
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        
        # 计算更新（与全局的差异）
        update = {}
        for key, value in self.local_model.state_dict().items():
            update[key] = value - global_weights[key]
        
        return update
```

---

## 💡 案例研究：基于 NVIDIA Jetson 的边缘部署

### 完整部署管道

🔴 高级

```yaml
# NVIDIA Jetson 部署配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: jetson-config
  namespace: edge-ai
data:
  DEPLOY_MODE: "production"
  MODEL_FORMAT: "tensorrt"
  ENABLE_DLA: "true"
  GPU_MEMORY_FRACTION: "0.8"
  MAX_BATCH_SIZE: "4"
  INFERENCE_PRECISION: "fp16"
---
# Jetson 特定的 DaemonSet
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: jetson-agent
  namespace: edge-ai
spec:
  selector:
    matchLabels:
      app: jetson-agent
  template:
    metadata:
      labels:
        app: jetson-agent
    spec:
      containers:
      - name: jetson-agent
        image: nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
        command:
        - python
        - /scripts/jetson_agent.py
        env:
        - name: JETSON_MODEL
          value: "resnet50"
        - name: TENSORRT_CACHE_PATH
          value: "/var/cache/tensorrt"
        resources:
          limits:
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: model-cache
          mountPath: /var/cache/models
        - name: tensorrt-cache
          mountPath: /var/cache/tensorrt
      volumes:
      - name: model-cache
        hostPath:
          path: /var/lib/jetson-models
          type: DirectoryOrCreate
      - name: tensorrt-cache
        hostPath:
          path: /var/cache/tensorrt
          type: DirectoryOrCreate
      nodeSelector:
        hardware: nvidia-jetson
      tolerations:
      - key: "jetson"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

### Jetson 优化代码

```python
# NVIDIA Jetson 优化
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
from pathlib import Path

class JetsonInference:
    def __init__(self, model_path: str, use_dla: bool = False, 
                 dla_core: int = 0):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.use_dla = use_dla
        self.dla_core = dla_core
        
        # 构建或加载引擎
        self.engine = self._load_engine(model_path)
        self.context = self.engine.create_execution_context()
        
        # 分配内存
        self._allocate_buffers()
        
        # CUDA 流
        self.stream = cuda.Stream()
    
    def _load_engine(self, model_path: str) -> trt.ICudaEngine:
        """加载 TensorRT 引擎。"""
        cache_path = Path(model_path).with_suffix('.trt')
        
        if cache_path.exists():
            # 加载缓存的引擎
            with open(cache_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                return runtime.deserialize_cuda_engine(f.read())
        else:
            # 构建新引擎
            engine = self._build_engine(model_path)
            
            # 缓存引擎
            with open(cache_path, 'wb') as f:
                f.write(engine.serialize())
            
            return engine
    
    def _build_engine(self, onnx_path: str) -> trt.ICudaEngine:
        """从 ONNX 构建 TensorRT 引擎。"""
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        # 解析 ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("解析 ONNX 模型失败")
        
        # 配置构建器
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28  # 256MB
        
        # 启用 FP16
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # 如果可用则启用 DLA
        if self.use_dla and builder.platform_has_fast_dla:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = self.dla_core
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        
        # 构建引擎
        engine = builder.build_engine(network, config)
        
        return engine
    
    def _allocate_buffers(self):
        """分配 GPU 内存。"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape
                })
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """在 Jetson 上运行推理。"""
        # 如需要则调整输入形状
        input_data = input_data.astype(np.float32)
        
        # 将输入复制到固定内存
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # 传输到 GPU
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # 运行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 传回输出
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        
        # 同步
        self.stream.synchronize()
        
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])

# 在 Jetson 上使用
def deploy_on_jetson():
    """完整的 Jetson 部署示例。"""
    # 初始化推理
    engine = JetsonInference(
        model_path='/var/lib/jetson-models/resnet50.onnx',
        use_dla=True,
        dla_core=0
    )
    
    # 处理摄像头馈送
    import cv2
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理
        input_tensor = preprocess_frame(frame)
        
        # 推理
        output = engine.infer(input_tensor)
        
        # 后处理
        predictions = postprocess_output(output)
        
        # 显示
        display_predictions(frame, predictions)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

def preprocess_frame(frame):
    """预处理摄像头帧。"""
    import cv2
    
    # 调整大小
    img = cv2.resize(frame, (224, 224))
    
    # BGR 转 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 归一化
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    # 转置
    img = img.transpose(2, 0, 1)
    
    # 添加批次
    return np.expand_dims(img, axis=0)

def postprocess_output(output):
    """后处理模型输出。"""
    # Softmax
    exp_output = np.exp(output - np.max(output))
    probs = exp_output / exp_output.sum()
    
    # 获取 top 预测
    top_k = 5
    top_indices = np.argsort(probs[0])[-top_k:][::-1]
    top_probs = probs[0][top_indices]
    
    return list(zip(top_indices.tolist(), top_probs.tolist()))
```

---

## 📝 练习

### 练习 19.1：边缘部署管道
构建完整的边缘部署管道，要求：
1. 将 PyTorch 模型导出为 ONNX
2. 使用 TensorRT 优化
3. 部署到 Jetson 设备
4. 监控推理性能
5. 处理模型更新

### 练习 19.2：离线推理
为偏远位置实现离线推理：
1. 优雅处理网络断开
2. 本地排队结果
3. 连接恢复时同步
4. 维护数据一致性
5. 在离线期间处理模型更新

### 练习 19.3：边缘-云协同
设计边缘-云协同系统，要求：
1. 100 台边缘设备的联邦学习
2. 隐私保护的数据聚合
3. 高效的模型分发
4. 实时监控和告警
5. 基于需求的自动扩展

---

## ⚠️ 警告

1. **资源限制**：边缘设备计算和内存有限。积极优化模型。
2. **热节流**：监控设备温度。热压力下性能下降。
3. **电源管理**：实现电源感知调度。电池供电设备需要仔细的能源管理。
4. **安全性**：边缘设备物理暴露。实现安全启动、加密存储和远程擦除。
5. **连接性**：为间歇性连接设计。切勿假设持续的网络访问。

---

## 本章小结

本章介绍了边缘部署架构和策略：
1. 边缘推理框架（ONNX Runtime、TensorRT、TFLite）
2. 模型更新策略（OTA、版本控制、回滚）
3. 边缘集群管理（K3s、边缘的 Kubernetes）
4. 离线推理架构
5. 边缘-云协同模式
6. NVIDIA Jetson 部署最佳实践

这完成了第 6 部分：边缘 AI 架构。您现在掌握了云原生和边缘 AI 架构的全面知识，能够在整个计算连续体上设计和部署 AI 系统。
