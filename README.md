<div align="center">

# Qi

**A High-Performance Real-Time Inference Engine for World Action Models**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)]()

</div>

---

## What is Qi?

**Qi** is a high-performance, low-latency inference engine purpose-built for **World Action Models (WAMs)** — generative models that jointly reason over perception, state, and action to predict and plan in open-ended environments.

Unlike general-purpose inference runtimes, Qi is designed around the unique demands of world modeling:

- **Temporally coherent streaming** — action sequences must be generated and executed with tight real-time constraints
- **Multi-modal throughput** — efficiently handling joint video, state, and action token streams
- **Sub-frame latency** — critical for closed-loop control where inference delay directly degrades policy quality
- **Hardware-aware scheduling** — kernel-level optimizations tuned for WAM inference patterns

---

## Key Features

### ⚡ Real-Time Performance
- Ultra-low latency inference optimized for control-frequency deployment (10–100 Hz)
- Asynchronous multi-stream execution to decouple perception, reasoning, and action

### 🧠 World Action Model Native
- First-class support for joint observation–action generation
- Streaming output API designed for real-time action consumption

### 🔧 Hardware Optimized
- Custom CUDA kernels for WAM-specific attention patterns
- Quantization support: FP16, BF16, INT8, FP8

### 🔌 Flexible Integration
- Clean Python and C++ APIs
- gRPC and REST serving interfaces
- Compatible with standard model formats (SafeTensors, ONNX)

---

## Performance

### Per-action chunk latency on A100

| Metric (ms) | Baseline | +DiT Cache | +CUDA Graph | +`torch.compile` |
|-------------|----------|------------|-------------|------------------|
| Mean        | 463      | 304        | 125         | 95               |
| P50         | 464      | 303        | 124         | 94               |
| P99         | 469      | 323        | 130         | 100              |
| Max         | 469      | 323        | 130         | 100              |

| Speedup             | +DiT Cache | +CUDA Graph | +`torch.compile` |
|---------------------|------------|-------------|------------------|
| Mean                | 1.52×      | 3.70×       | 4.87×            |
| P99                 | 1.45×      | 3.61×       | 4.69×            |

### Per-action chunk latency on 4090

| Metric (ms) | Baseline | +DiT Cache | +CUDA Graph | +`torch.compile` |
|-------------|----------|------------|-------------|------------------|
| Mean        | 182      | 121        | 83          | 67               |
| P50         | 182      | 121        | 83          | 67               |
| P99         | 185      | 123        | 84          | 70               |
| Max         | 185      | 123        | 84          | 70               |

| Speedup             | +DiT Cache | +CUDA Graph | +`torch.compile` |
|---------------------|------------|-------------|------------------|
| Mean                | 1.50×      | 2.19×       | 2.71×            |
| P99                 | 1.50×      | 2.20×       | 2.64×            |

Each + column is cumulative (e.g. +CUDA Graph includes DiT Cache).

---

## Roadmap

- [x] Edge deployment
  - [x] NVIDIA RTX 4090
  - [x] NVIDIA Jetson Orin
  - [x] NVIDIA A100
- [x] DiT caching
- [ ] Quantization
- [ ] Kernel optimization
  - [x] Cuda graph
  - [x] torch.compile
  - [ ] Kernel customization
- [ ] World-action model zoo

---

## Philosophy

The name **Qi** (器, *qì*) comes from a foundational line in the *I Ching* (*Zhou Yi*, Commentary on the Appended Phrases):

> 形而上者谓之道，形而下者谓之器
>
> *"That which is above form is called Dao; that which takes form is called Qi."*

In classical Chinese thought, **Dao** (道) is the formless principle — the underlying law of the world. **Qi** (器) is the vessel, the concrete instrument through which Dao manifests and acts.

We adopt this framing for a simple reason: **器以载道** — *the vessel carries the Dao*. A world action model encodes a deep understanding of how the world works and how actions ripple through it. **Qi** is the engine that gives this model a body — fast, precise, and real-time — so that understanding can become action.

---

## Contributing

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues, feature requests, and pull requests.

---

## License

Qi is released under the [Apache 2.0 License](LICENSE).

---

<div align="center">

**器以载道** — *The vessel carries the Dao.*

*Qi gives world action models a body.*

</div>
