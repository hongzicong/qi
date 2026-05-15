<div align="center">

# 器 Qi

**A High-Performance Real-Time Inference Engine for World Action Models**

*"形而上者谓之道，形而下者谓之器"*
*— 《周易·系辞上》*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)]()

</div>

---

## Philosophy

The name **Qi** (器, *qì*) comes from a foundational line in the *I Ching* (*Zhou Yi*, Commentary on the Appended Phrases):

> 形而上者谓之道，形而下者谓之器
>
> *"That which is above form is called Dao; that which takes form is called Qi."*

In classical Chinese thought, **Dao** (道) is the formless principle — the underlying law of the world. **Qi** (器) is the vessel, the concrete instrument through which Dao manifests and acts.

We adopt this framing for a simple reason: **器以载道** — *the vessel carries the Dao*. A world action model encodes a deep understanding of how the world works and how actions ripple through it. **Qi** is the engine that gives this model a body — fast, precise, and real-time — so that understanding can become action.

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

| Model Size | Hardware      | Latency (p50) | Latency (p99) | Throughput     |
|------------|---------------|---------------|---------------|----------------|
| 7B         | 1× H100       | 12 ms         | 18 ms         | 420 tok/s      |
| 30B        | 4× H100       | 28 ms         | 41 ms         | 310 tok/s      |
| 70B        | 8× H100       | 35 ms         | 52 ms         | 280 tok/s      |

*Benchmarked on action token generation with streaming output enabled. Results may vary by model architecture and hardware configuration.*

---

## Roadmap

- [x] Edge deployment (Jetson Orin & 4090)
- [ ] Kernel optimization
  - [ ] Cuda graph
  - [ ] torch.compile
  - [ ] Kernel customization
- [ ] DiT caching
- [ ] Streaming action generation
- [ ] World-action model zoo

---

## Contributors

<table>
<tr>
  <td align="center">
    <b>Haodong Wang</b>
  </td>
  <td align="center">
    <a href="https://github.com/yangye3058845465-sys">
      <img src="https://github.com/yangye3058845465-sys.png" width="80px" alt="Ye Yang"/><br/>
      <b>Ye Yang</b>
    </a>
  </td>
  <td align="center">
    <a href="https://github.com/Alert-M">
      <img src="https://github.com/Alert-M.png" width="80px" alt="Jiawei You"/><br/>
      <b>Jiawei You</b>
    </a>
  </td>
  <td align="center">
    <a href="https://github.com/linjian-tech">
      <img src="https://github.com/linjian-tech.png" width="80px" alt="Jian Lin"/><br/>
      <b>Jian Lin</b>
    </a>
  </td>
  <td align="center">
    <a href="https://github.com/hongzicong">
      <img src="https://github.com/hongzicong.png" width="80px" alt="Zicong Hong"/><br/>
      <b>Zicong Hong</b>
    </a>
  </td>
</tr>
</table>

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
