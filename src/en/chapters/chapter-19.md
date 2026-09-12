# Chapter 19: Edge Deployment Architecture

🟢 Beginner | 🟡 Intermediate | 🔴 Advanced | ⚫ Manager

---

## 19.1 Edge Inference Frameworks

### Framework Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│              Edge Inference Frameworks                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Framework          │ Hardware Support    │ Model Formats       │
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

### ONNX Runtime for Edge

```python
# ONNX Runtime Edge Inference
import onnxruntime as ort
import numpy as np
import cv2

class EdgeInferenceEngine:
    def __init__(self, model_path, providers=['CPUExecutionProvider']):
        """Initialize ONNX Runtime inference engine."""
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
        """Preprocess image for inference."""
        # Resize
        img = cv2.resize(image, input_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Transpose to CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def inference(self, input_data):
        """Run inference."""
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return outputs[0]
    
    def postprocess(self, output, top_k=5):
        """Post-process output."""
        # Apply softmax
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / exp_output.sum()
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities[0])[-top_k:][::-1]
        top_k_probs = probabilities[0][top_k_indices]
        
        return list(zip(top_k_indices.tolist(), top_k_probs.tolist()))
    
    def predict(self, image):
        """Full prediction pipeline."""
        input_data = self.preprocess(image)
        output = self.inference(input_data)
        predictions = self.postprocess(output)
        return predictions

# Usage example
engine = EdgeInferenceEngine(
    model_path='resnet50_quantized.onnx',
    providers=['CPUExecutionProvider']
)

# Load and predict
image = cv2.imread('test_image.jpg')
predictions = engine.predict(image)

for class_id, prob in predictions:
    print(f"Class {class_id}: {prob:.4f}")
```

### TensorRT Optimization

```python
# TensorRT Optimization for Edge
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTInference:
    def __init__(self, engine_path):
        """Initialize TensorRT inference engine."""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate memory
        self._allocate_buffers()
    
    def _allocate_buffers(self):
        """Allocate GPU memory for input/output."""
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
        """Run TensorRT inference."""
        # Copy input to host memory
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer input to GPU
        cuda.memcpy_htod(
            self.inputs[0]['device'],
            self.inputs[0]['host']
        )
        
        # Run inference
        self.context.execute_v2(bindings=self.bindings)
        
        # Transfer output back to host
        cuda.memcpy_dtoh(
            self.outputs[0]['host'],
            self.outputs[0]['device']
        )
        
        return self.outputs[0]['host'].reshape(self.engine.get_binding_shape(1))

def build_tensorrt_engine(onnx_path, engine_path, fp16=True, max_batch_size=8):
    """Build TensorRT engine from ONNX."""
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
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Set optimization profiles
    profile = builder.create_optimization_profile()
    profile.set_shape(
        'input',
        min=(1, 3, 224, 224),
        opt=(max_batch_size // 2, 3, 224, 224),
        max=(max_batch_size, 3, 224, 224)
    )
    config.add_optimization_profile(profile)
    
    # Build engine
    engine = builder.build_engine(network, config)
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine
```

---

## 19.2 Model Update Strategy

### Update Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Model Update Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Cloud (Central)                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Model   │  │  Update  │  │  Version │  │Push  │  │   │
│  │  │ Registry │  │  Manager │  │  Control │  │Service│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │   Update Channel  │                      │
│                    │   (OTA/CDN)       │                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Edge Fleet                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Edge    │  │  Edge    │  │  Edge    │  │Edge  │  │   │
│  │  │ Device 1 │  │ Device 2 │  │ Device 3 │  │Dev N │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### OTA Update Implementation

```python
# OTA Model Update System
import requests
import hashlib
import json
import time
import os
from pathlib import Path

class EdgeOTAUpdater:
    def __init__(self, device_id, registry_url, local_model_dir):
        self.device_id = device_id
        self.registry_url = registry_url
        self.local_model_dir = Path(local_model_dir)
        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Device state
        self.current_model_version = self._get_current_version()
        self.last_update_check = 0
        self.update_interval = 3600  # Check every hour
    
    def _get_current_version(self):
        """Get current model version."""
        version_file = self.local_model_dir / 'version.json'
        if version_file.exists():
            with open(version_file, 'r') as f:
                return json.load(f)
        return {'version': '0.0.0', 'hash': None}
    
    def _save_version(self, version_info):
        """Save version information."""
        version_file = self.local_model_dir / 'version.json'
        with open(version_file, 'w') as f:
            json.dump(version_info, f)
    
    def check_for_updates(self):
        """Check if update is available."""
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
            print(f"Error checking for updates: {e}")
            return None
    
    def download_model(self, model_url, expected_hash):
        """Download model with integrity check."""
        temp_path = self.local_model_dir / 'model_new.onnx'
        
        try:
            # Download with progress
            response = requests.get(model_url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
            
            # Verify hash
            actual_hash = self._compute_hash(temp_path)
            if actual_hash != expected_hash:
                raise ValueError(f"Hash mismatch: {actual_hash} != {expected_hash}")
            
            return temp_path
            
        except Exception as e:
            # Clean up on failure
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _compute_hash(self, file_path):
        """Compute SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def apply_update(self, new_model_path, version_info):
        """Apply model update with rollback capability."""
        backup_dir = self.local_model_dir / 'backup'
        backup_dir.mkdir(exist_ok=True)
        
        # Backup current model
        current_model = self.local_model_dir / 'model.onnx'
        if current_model.exists():
            backup_model = backup_dir / f'model_{self.current_model_version["version"]}.onnx'
            if backup_model.exists():
                backup_model.unlink()
            current_model.rename(backup_model)
        
        # Apply new model
        new_model = self.local_model_dir / 'model.onnx'
        new_model_path.rename(new_model)
        
        # Update version
        self._save_version(version_info)
        self.current_model_version = version_info
        
        print(f"Model updated to version {version_info['version']}")
    
    def rollback(self):
        """Rollback to previous model version."""
        backup_dir = self.local_model_dir / 'backup'
        
        # Find latest backup
        backups = sorted(backup_dir.glob('model_*.onnx'))
        if not backups:
            raise ValueError("No backup available for rollback")
        
        latest_backup = backups[-1]
        
        # Restore
        current_model = self.local_model_dir / 'model.onnx'
        if current_model.exists():
            current_model.unlink()
        
        latest_backup.rename(current_model)
        
        # Update version
        version = latest_backup.stem.replace('model_', '')
        self._save_version({'version': version, 'hash': None})
        
        print(f"Rolled back to version {version}")
    
    def update_loop(self):
        """Main update loop."""
        while True:
            update = self.check_for_updates()
            
            if update:
                print(f"Update available: {update['version']}")
                
                try:
                    # Download new model
                    temp_path = self.download_model(
                        update['download_url'],
                        update['hash']
                    )
                    
                    # Apply update
                    self.apply_update(temp_path, update)
                    
                except Exception as e:
                    print(f"Update failed: {e}")
                    # Continue with current model
            
            time.sleep(60)  # Check every minute
```

---

## 19.3 Edge Cluster Management

### Kubernetes at the Edge (K3s/kubeedge)

```yaml
# K3s Edge Cluster Configuration
apiVersion: v1
kind: Namespace
metadata:
  name: edge-ai
  labels:
    edge-cluster: "true"
---
# Edge Device Fleet
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
# Edge inference deployment
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

### Edge Fleet Management

```python
# Edge Fleet Management System
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
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
        """Register new edge device."""
        device = EdgeDevice(
            device_id=device_info['device_id'],
            location=device_info['location'],
            status=DeviceStatus.ONLINE,
            current_model_version=device_info['model_version'],
            last_heartbeat=datetime.now(),
            resource_usage=device_info.get('resource_usage', {})
        )
        
        self.devices[device.device_id] = device
        
        # Notify cloud
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.cloud_endpoint}/api/devices/register",
                json=device_info
            ) as response:
                return await response.json()
    
    async def heartbeat(self, device_id: str, status: Dict):
        """Receive heartbeat from edge device."""
        if device_id in self.devices:
            self.devices[device_id].last_heartbeat = datetime.now()
            self.devices[device_id].status = DeviceStatus.ONLINE
            self.devices[device_id].resource_usage = status.get('resource_usage', {})
    
    async def check_device_health(self):
        """Check health of all devices."""
        now = datetime.now()
        unhealthy_devices = []
        
        for device_id, device in self.devices.items():
            # Check if heartbeat is stale
            if (now - device.last_heartbeat) > timedelta(minutes=5):
                device.status = DeviceStatus.OFFLINE
                unhealthy_devices.append(device_id)
            
            # Check resource usage
            if device.resource_usage.get('gpu_memory', 0) > 90:
                unhealthy_devices.append(device_id)
        
        return unhealthy_devices
    
    async def orchestrate_update(self, model_version: str, 
                                  strategy: str = 'rolling'):
        """Orchestrate model update across fleet."""
        devices = list(self.devices.keys())
        
        if strategy == 'rolling':
            # Update 10% at a time
            batch_size = max(1, len(devices) // 10)
            
            for i in range(0, len(devices), batch_size):
                batch = devices[i:i + batch_size]
                
                # Update batch
                await self._update_batch(batch, model_version)
                
                # Wait for batch to complete
                await asyncio.sleep(60)
                
                # Check health
                unhealthy = await self.check_device_health()
                if unhealthy:
                    print(f"Unhealthy devices during update: {unhealthy}")
                    # Rollback if too many failures
                    if len(unhealthy) > batch_size * 0.5:
                        await self._rollback_batch(batch)
                        break
        
        elif strategy == 'canary':
            # Update 1 device first
            canary_device = devices[0]
            await self._update_batch([canary_device], model_version)
            
            # Wait and check
            await asyncio.sleep(300)
            
            if canary_device not in await self.check_device_health():
                # Proceed with rest
                await self._update_batch(devices[1:], model_version)
            else:
                print("Canary update failed, aborting")
                await self._rollback_batch([canary_device])
    
    async def _update_batch(self, device_ids: List[str], model_version: str):
        """Update a batch of devices."""
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
                            print(f"Failed to update {device_id}")
            except Exception as e:
                print(f"Error updating {device_id}: {e}")
    
    async def _rollback_batch(self, device_ids: List[str]):
        """Rollback a batch of devices."""
        for device_id in device_ids:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.cloud_endpoint}/api/devices/{device_id}/rollback"
                    ) as response:
                        if response.status == 200:
                            print(f"Rolled back {device_id}")
            except Exception as e:
                print(f"Error rolling back {device_id}: {e}")
    
    def get_fleet_status(self) -> Dict:
        """Get overall fleet status."""
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

## 19.4 Offline Inference Architecture

### Offline Mode Design

```
┌─────────────────────────────────────────────────────────────────┐
│              Offline Inference Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Online Mode                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Input   │  │ Inference│  │  Output  │  │Cloud │  │   │
│  │  │  Stream  │→ │  Engine  │→ │  Stream  │→ │Sync  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │   Network State   │                      │
│                    │   Detection       │                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Offline Mode                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Input   │  │ Inference│  │  Local   │  │Queue │  │   │
│  │  │  Stream  │→ │  Engine  │→ │  Storage │  │Manager│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Key Features:                                                 │
│  • Local model caching                                         │
│  • Result queueing for later sync                              │
│  • Graceful degradation                                        │
│  • Automatic sync when connectivity restored                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# Offline Inference Manager
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
        
        # Initialize local database
        self._init_db()
        
        # Network state
        self.is_online = True
        self.sync_interval = 300  # 5 minutes
        self.max_queue_size = 10000
        
        # Load model
        self.model = self._load_model()
    
    def _init_db(self):
        """Initialize local SQLite database."""
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
        """Load model from local storage."""
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
        """Check if cloud is reachable."""
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
        """Run inference with offline support."""
        # Run inference locally
        output = self._run_inference(input_data)
        
        # Store in queue if offline
        if not self.is_online:
            self._queue_result(input_data, output)
        
        # Try to sync if online
        if self.is_online:
            await self._try_sync()
        
        return output
    
    def _run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run local inference."""
        import numpy as np
        
        # Preprocess input
        input_tensor = self._preprocess(input_data)
        
        # Run inference
        input_name = self.model.get_inputs()[0].name
        output = self.model.run(None, {input_name: input_tensor})[0]
        
        # Postprocess output
        result = self._postprocess(output)
        
        return result
    
    def _preprocess(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data."""
        # Example: image preprocessing
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
        """Postprocess inference output."""
        # Apply softmax
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / exp_output.sum()
        
        return {
            'predictions': probabilities.tolist(),
            'timestamp': datetime.now().isoformat(),
            'model_version': self._get_model_version()
        }
    
    def _get_model_version(self) -> str:
        """Get current model version."""
        version_file = Path(self.model_path).parent / 'version.json'
        if version_file.exists():
            with open(version_file, 'r') as f:
                return json.load(f).get('version', 'unknown')
        return 'unknown'
    
    def _queue_result(self, input_data: Dict, output: Dict):
        """Queue result for later sync."""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        # Check queue size
        cursor.execute('SELECT COUNT(*) FROM inference_queue WHERE synced = FALSE')
        queue_size = cursor.fetchone()[0]
        
        if queue_size >= self.max_queue_size:
            # Remove oldest unsynced results
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
        
        # Insert new result
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
        """Try to sync queued results to cloud."""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        # Get unsynced results
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
        
        # Sync to cloud
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
                    
                    # Mark as synced
                    cursor.execute(
                        'UPDATE inference_queue SET synced = TRUE WHERE id = ?',
                        (record_id,)
                    )
            
            conn.commit()
            
        except Exception as e:
            print(f"Sync failed: {e}")
        
        finally:
            conn.close()
    
    async def sync_loop(self):
        """Background sync loop."""
        while True:
            # Check connectivity
            self.is_online = await self.check_connectivity()
            
            if self.is_online:
                await self._try_sync()
            
            await asyncio.sleep(self.sync_interval)
    
    def get_queue_status(self) -> Dict:
        """Get queue status."""
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

## 19.5 Edge-Cloud Collaboration Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Edge-Cloud Collaboration Architecture              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Cloud Layer                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Model   │  │ Training │  │  Fleet   │  │Data  │  │   │
│  │  │ Registry │  │ Service  │  │ Manager  │  │Lake  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                      │
│                    │  Communication    │                      │
│                    │  Layer (MQTT/     │                      │
│                    │  gRPC/HTTP)       │                      │
│                    └─────────┬─────────┘                      │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Edge Layer                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Edge    │  │  Local   │  │ Inference│  │Data  │  │   │
│  │  │ Agent   │  │  Cache   │  │  Engine  │  │Filter│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Device Layer                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │   │
│  │  │  Sensor  │  │  Camera  │  │  Actuator│  │User  │  │   │
│  │  │  Array   │  │  Feed    │  │  Control │  │Input │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Communication Protocol

```python
# Edge-Cloud Communication Protocol
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
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
        """Establish connection to cloud."""
        # Implementation depends on protocol (MQTT, gRPC, etc.)
        self.is_connected = True
        
    async def send_message(self, msg_type: MessageType, payload: Dict[str, Any]):
        """Send message to cloud."""
        message = {
            'device_id': self.device_id,
            'message_type': msg_type.value,
            'timestamp': datetime.now().isoformat(),
            'payload': payload
        }
        
        if self.is_connected:
            # Send directly
            await self._send_to_cloud(message)
        else:
            # Queue for later
            await self.message_queue.put(message)
    
    async def receive_messages(self):
        """Receive messages from cloud."""
        # Implementation depends on protocol
        pass
    
    async def _send_to_cloud(self, message: Dict):
        """Internal send implementation."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.cloud_endpoint}/api/messages",
                json=message
            ) as response:
                return await response.json()
    
    async def sync_queued_messages(self):
        """Sync queued messages when connected."""
        while not self.message_queue.empty():
            message = await self.message_queue.get()
            try:
                await self._send_to_cloud(message)
            except Exception:
                # Put back in queue on failure
                await self.message_queue.put(message)
                break
```

### Collaborative Training

```python
# Federated Learning for Edge-Cloud Collaboration
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
        """Coordinate one round of federated learning."""
        self.round_number += 1
        
        # Send global model to clients
        global_weights = self.global_model.state_dict()
        
        # Collect updates from clients
        updates = []
        for client in edge_clients:
            update = await client.train_locally(global_weights)
            updates.append(update)
        
        # Aggregate updates
        aggregated_weights = self._aggregate_updates(updates)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_weights)
        
        return aggregated_weights
    
    def _aggregate_updates(self, updates: List[Dict]) -> Dict:
        """Aggregate client updates using FedAvg."""
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
        """Train locally and return update."""
        # Load global weights
        self.local_model.load_state_dict(global_weights)
        
        # Local training
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
        
        # Calculate update (difference from global)
        update = {}
        for key, value in self.local_model.state_dict().items():
            update[key] = value - global_weights[key]
        
        return update
```

---

## 💡 Case Study: NVIDIA Jetson Edge Deployment

### Complete Deployment Pipeline

🔴 Advanced

```yaml
# NVIDIA Jetson Deployment Configuration
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
# Jetson-specific DaemonSet
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
        - name: dla-config
          mountPath: /etc/dla
      volumes:
      - name: model-cache
        hostPath:
          path: /var/lib/jetson-models
          type: DirectoryOrCreate
      - name: tensorrt-cache
        hostPath:
          path: /var/cache/tensorrt
          type: DirectoryOrCreate
      - name: dla-config
        configMap:
          name: dla-config
      nodeSelector:
        hardware: nvidia-jetson
      tolerations:
      - key: "jetson"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

### Jetson Optimization Code

```python
# NVIDIA Jetson Optimization
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
        
        # Build or load engine
        self.engine = self._load_engine(model_path)
        self.context = self.engine.create_execution_context()
        
        # Allocate memory
        self._allocate_buffers()
        
        # CUDA stream
        self.stream = cuda.Stream()
    
    def _load_engine(self, model_path: str) -> trt.ICudaEngine:
        """Load TensorRT engine."""
        cache_path = Path(model_path).with_suffix('.trt')
        
        if cache_path.exists():
            # Load cached engine
            with open(cache_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                return runtime.deserialize_cuda_engine(f.read())
        else:
            # Build new engine
            engine = self._build_engine(model_path)
            
            # Cache engine
            with open(cache_path, 'wb') as f:
                f.write(engine.serialize())
            
            return engine
    
    def _build_engine(self, onnx_path: str) -> trt.ICudaEngine:
        """Build TensorRT engine from ONNX."""
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28  # 256MB
        
        # Enable FP16
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Enable DLA if available
        if self.use_dla and builder.platform_has_fast_dla:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = self.dla_core
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        return engine
    
    def _allocate_buffers(self):
        """Allocate GPU memory."""
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
        """Run inference on Jetson."""
        # Reshape input if needed
        input_data = input_data.astype(np.float32)
        
        # Copy input to pinned memory
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer to GPU
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Transfer output back
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        
        # Synchronize
        self.stream.synchronize()
        
        return self.outputs[0]['host'].reshape(self.outputs[0]['shape'])

# Usage on Jetson
def deploy_on_jetson():
    """Complete Jetson deployment example."""
    # Initialize inference
    engine = JetsonInference(
        model_path='/var/lib/jetson-models/resnet50.onnx',
        use_dla=True,
        dla_core=0
    )
    
    # Process camera feed
    import cv2
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        input_tensor = preprocess_frame(frame)
        
        # Inference
        output = engine.infer(input_tensor)
        
        # Postprocess
        predictions = postprocess_output(output)
        
        # Display
        display_predictions(frame, predictions)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

def preprocess_frame(frame):
    """Preprocess camera frame."""
    import cv2
    
    # Resize
    img = cv2.resize(frame, (224, 224))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    # Transpose
    img = img.transpose(2, 0, 1)
    
    # Add batch
    return np.expand_dims(img, axis=0)

def postprocess_output(output):
    """Postprocess model output."""
    # Softmax
    exp_output = np.exp(output - np.max(output))
    probs = exp_output / exp_output.sum()
    
    # Get top predictions
    top_k = 5
    top_indices = np.argsort(probs[0])[-top_k:][::-1]
    top_probs = probs[0][top_indices]
    
    return list(zip(top_indices.tolist(), top_probs.tolist()))

def display_predictions(frame, predictions):
    """Display predictions on frame."""
    import cv2
    
    y_offset = 30
    for class_id, prob in predictions:
        text = f"Class {class_id}: {prob:.4f}"
        cv2.putText(frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    cv2.imshow('Jetson Inference', frame)
```

---

## 📝 Exercises

### Exercise 19.1: Edge Deployment Pipeline
Build a complete edge deployment pipeline that:
1. Exports PyTorch model to ONNX
2. Optimizes with TensorRT
3. Deploys to Jetson device
4. Monitors inference performance
5. Handles model updates

### Exercise 19.2: Offline Inference
Implement offline inference for a remote location:
1. Handle network disconnections gracefully
2. Queue results locally
3. Sync when connectivity restored
4. Maintain data consistency
5. Handle model updates during offline periods

### Exercise 19.3: Edge-Cloud Collaboration
Design an edge-cloud collaboration system for:
1. Federated learning across 100 edge devices
2. Privacy-preserving data aggregation
3. Efficient model distribution
4. Real-time monitoring and alerting
5. Automatic scaling based on demand

---

## ⚠️ Warnings

1. **Resource Constraints**: Edge devices have limited compute and memory. Optimize models aggressively.
2. **Thermal Throttling**: Monitor device temperature. Performance degrades under thermal pressure.
3. **Power Management**: Implement power-aware scheduling. Battery-powered devices need careful energy management.
4. **Security**: Edge devices are physically exposed. Implement secure boot, encrypted storage, and remote wipe.
5. **Connectivity**: Design for intermittent connectivity. Never assume constant network access.

---

## Summary

This chapter covered edge deployment architecture and strategies:
1. Edge inference frameworks (ONNX Runtime, TensorRT, TFLite)
2. Model update strategies (OTA, versioning, rollback)
3. Edge cluster management (K3s, Kubernetes at edge)
4. Offline inference architecture
5. Edge-cloud collaboration patterns
6. NVIDIA Jetson deployment best practices

This completes Part 6: Edge AI Architecture. You now have comprehensive knowledge of both cloud-native and edge AI architectures, enabling you to design and deploy AI systems across the entire computing continuum.
