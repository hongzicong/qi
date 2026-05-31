# FlashRT FP8 W8A8 Backend

FlashRT provides the accelerated `flashrt_fp8` backend for FP8 E4M3 W8A8 inference.

```text
weight: FP8 E4M3 bytes, layout (N, K)
activation: quantized to FP8 E4M3 at runtime
GEMM: FlashRT GemmRunner.fp8_run_dev
```


## Clone FlashRT


```bash
mkdir -p third_party
git clone https://github.com/LiangSu8899/FlashRT.git $FLASHRT_ROOT
cd $FLASHRT_ROOT
```

## Install Build Dependencies

```bash
python -m pip install -U pip setuptools wheel
python -m pip install pybind11 cmake "numpy>=1.24" safetensors
python -m pip install -e ".[torch]"
```

## Configure CUDA and CUTLASS

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

which nvcc
nvcc --version

git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass
```


## Build and Verify

Use the architecture for GPU:

| GPU | `GPU_ARCH` |
|---|---:|
| A100 | `80` |
| RTX 3090 / A10 | `86` |
| RTX 4090 / L40 | `89` |
| RTX 5090 | `120` |

Example for RTX 4090:

```bash
cd $FLASHRT_ROOT
rm -rf build
cmake -B build -S . -DGPU_ARCH=89 -DCMAKE_CUDA_COMPILER=$CUDACXX
cmake --build build -j$(nproc)
cd $QI_ROOT

python -B -c 'from qi.quantization.backends.flashrt_fp8_ops import load_flashrt_fp8_ops; load_flashrt_fp8_ops(required=True); print("FlashRT FP8 backend is available")'
```

The extension must expose `quantize_fp8_static` and `GemmRunner.fp8_run_dev`.
