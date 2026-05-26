# Guides

## 0 Environment Setup

| Guide | Notes |
|-------|-------|
| [RTX 4090](4090.md) | Local workstation; requires NVIDIA Container Toolkit |
| [EPFL RCP](epfl.md) | EPFL cluster; uses RunAI with a private registry |
| [Jetson Orin](jetson.md) | Edge device (ARM64) |

## 1.0 Quick Start

```python
import qi

model = qi.load_model(...)
pred  = model.infer(...)
```

See §1.1 for a complete example.

## 1.1 Dry-Run Test

Once the container is set up, follow [common.md](common.md) to download the required checkpoints and run the dry-run test.

## 1.2 Dataset Evaluation

| Guide | Notes |
|-------|-------|
| [RoboTwin Evaluation](robotwin.md) | Covers submodule setup, asset preparation, and the benchmark manager workflow |

## 1.3 Real-Robot Deployment

| Guide | Notes |
|-------|-------|
| [Cobot Magic](deploy_real_rtc.md) | Physical robot deployment with real-time chunking (RTC) |