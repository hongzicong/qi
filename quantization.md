
## 重要文件说明

- `tests/quantize_checkpoint.py`：提前生成量化后的 ckpt。
- `tests/collect_awq_stats.py`：给 AWQ 收 calibration activation stats。
- `src/qi/api.py`：推理时自动识别量化 ckpt，并先替换模型结构再加载权重。
- `reference` backend：正确性路径，dequant 后 `F.linear`。
- `flashrt` backend：尝试调用 FlashRT op，找不到就 fallback 到 reference。
- `cuda_ext` backend：目前预留，调用会报 `NotImplementedError`。

## 准备环境
```bash
source env.sh
```
## Prepare Checkpoints

### RTN：直接生成量化 ckpt

这个命令就是“提前生成量化后的 ckpt”：

```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_RTN_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo rtn \
  --quant-bits 8 \
  --quant-backend reference
```

逻辑：加载原始 fp ckpt -> 替换目标 `nn.Linear` -> RTN per-group 量化权重 -> 保存带 `qi_quantization` metadata 的 ckpt。

### AWQ：先收 stats
```bash
python tests/collect_awq_stats.py \
  --ckpt $QI_BASE_CKPT \
  --output-stats $QI_AWQ_STATS \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --num-inference-steps 5
```

逻辑：用 calibration 样本跑一次/几步 inference，hook 每个待量化 Linear 的输入，统计 `mean(abs(x))`，保存成 `awq_stats.pt`。

### AWQ：生成量化 ckpt
```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_AWQ_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo awq \
  --quant-bits 4 \
  --quant-group-size 128 \
  --awq-stats $QI_AWQ_STATS \
  --quant-backend reference
```

逻辑：读取 AWQ stats -> 计算 per-input-channel `input_scale` -> 对 `weight * input_scale` 做 per-group 量化 -> ckpt 里保存 `qweight/scales/input_scale/bias`。

## 推理：Load Checkpoints with Baseline and Reference/Fake-quant

```bash
python tests/test_dry_run.py \
  --ckpt $DIFFSYNTH_MODEL_BASE_PATH/step_008140.pt \
  --dataset-stats $DIFFSYNTH_MODEL_BASE_PATH/dataset_stats.json \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result_baseline \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```

本次终端结果：Baseline FP checkpoint，`num_inference_steps=10`，`num_chunks=20`，`seed=42`。下面统计不包含 chunk 0 warmup。

| Metric (ms) | Baseline FP checkpoint | `infer_action` only |
|-------------|------------------------|---------------------|
| Mean        | 1039                   | 299                 |
| P50         | 1040                   | 299                 |
| P99         | 1052                   | 304                 |
| Max         | 1053                   | 304                 |

chunk 0 warmup：end-to-end chunk 21760 ms，`infer_action` 20942 ms。输出目录：`output/dry_run_result_baseline`。


### RTN W8A16 

```bash
python tests/test_dry_run.py \
  --ckpt $QI_RTN_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result_rtn_w8_ref \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```

本次终端结果：RTN W8A16 fake-quant (`reference`, symmetric)，`num_inference_steps=10`，`num_chunks=20`，`seed=42`。下面统计不包含 chunk 0 warmup。

| Metric (ms) | RTN W8A16 fake-quant | `infer_action` only |
|-------------|----------------------|---------------------|
| Mean        | 1041                 | 299                 |
| P50         | 1039                 | 297                 |
| P99         | 1057                 | 304                 |
| Max         | 1069                 | 308                 |

chunk 0 warmup：end-to-end chunk 19901 ms，`infer_action` 19089 ms。输出目录：`output/dry_run_result`。

### AWQ W4A16 


```bash
python tests/test_dry_run.py \
  --ckpt $QI_AWQ_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result_awq_w4_ref \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```

本次终端结果：AWQ W4A16 fake-quant (`reference`)，`num_inference_steps=10`，`num_chunks=20`，`seed=42`。下面统计不包含 chunk 0 warmup。

| Metric (ms) | AWQ W4A16 fake-quant | `infer_action` only |
|-------------|----------------------|---------------------|
| Mean        | 1081                 | 312                 |
| P50         | 1077                 | 308                 |
| P99         | 1145                 | 344                 |
| Max         | 1151                 | 347                 |

chunk 0 warmup：end-to-end chunk 25921 ms，`infer_action` 25116 ms。输出目录：`output/dry_run_result_awq_w4_ref`。

**注意**
 `reference` backend 主要验证量化 ckpt 链路和数值逻辑，不一定加速。真正性能提升要等 FlashRT 真实 op 接上，或者后续实现 `cuda_ext`。

### 性能对比汇总

| Checkpoint | End-to-end Mean (ms) | `infer_action` Mean (ms) |
|------------|----------------------|--------------------------|
| Baseline FP checkpoint | 1039 | 299 |
| RTN W8A16 fake-quant (`reference`) | 1041 | 299 |
| AWQ W4A16 fake-quant (`reference`) | 1081 | 312 |

`reference` fake-quant 路径会先把低比特权重 dequant 回浮点再走普通 `F.linear`，因此主要用于验证量化链路；RTN W8A16 的额外开销基本被整体推理开销覆盖，而 AWQ W4A16 因 4bit/group-wise dequant 和 input scale 处理出现小幅变慢。

## FlashRT NVFP4 Backend

### 当前状态和硬件要求

`flashrt_nvfp4` backend 是真实加速路径，目标是调用 FlashRT 的 NVFP4 W4A16 GEMM：

```text
weight format: packed NVFP4 + swizzled block scale
activation: runtime quantize to NVFP4
GEMM: FlashRT GemmRunner.fp4_nn_dev
```

和上面的 `reference` fake-quant 不同，`flashrt_nvfp4` 不会显式 dequant 成完整 dense weight 再跑 `F.linear`。它要求 FlashRT 已编译出 `flash_rt.flash_rt_kernels`，并且 GPU 支持 NVFP4。FlashRT 当前 NVFP4 路径要求 SM120/Blackwell；RTX 4090 (`sm_89`) 可以保留代码和命令，但不能实际跑通 NVFP4 kernel。

### 准备 FlashRT

```bash
source env.sh
```

如果还没有 FlashRT checkout：

```bash
git clone https://github.com/LiangSu8899/FlashRT.git third_party/FlashRT
```

在 SM120/RTX 5090 机器上编译 FlashRT：

```bash
cd third_party/FlashRT

git clone --depth 1 --branch v4.4.2 \
  https://github.com/NVIDIA/cutlass.git third_party/cutlass

pip install -e ".[torch]"

cmake -B build -S . -DGPU_ARCH=120
cmake --build build -j$(nproc)

cd $QI_ROOT
```

验证 Qi 能找到 FlashRT NVFP4 kernels：

```bash
python -B -c 'from qi.quantization.backends.flashrt_nvfp4_ops import load_flashrt_nvfp4_ops; ops = load_flashrt_nvfp4_ops(required=True); print("flashrt nvfp4 ok")'
```

如果在 RTX 4090 上运行，预期会报类似：

```text
FlashRT NVFP4 requires SM120/Blackwell
```

### 环境变量：NVFP4 ckpt 输出路径

```bash
export QI_RTN_NVFP4_CKPT="${DIFFSYNTH_MODEL_BASE_PATH}/step_008140_rtn_nvfp4.pt"
export QI_AWQ_NVFP4_CKPT="${DIFFSYNTH_MODEL_BASE_PATH}/step_008140_awq_nvfp4.pt"
```

### RTN NVFP4：生成量化 ckpt

```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_RTN_NVFP4_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo rtn \
  --quant-dtype nvfp4 \
  --quant-bits 4 \
  --quant-group-size 16 \
  --quant-backend flashrt_nvfp4
```

逻辑：加载原始 fp ckpt -> 替换目标 `nn.Linear` -> 调 FlashRT `quantize_bf16_to_nvfp4_swizzled` 把权重打包成 `qweight_packed/scale_factors/weight_global_scale` -> 保存带 `qi_quantization` metadata 的 ckpt。

### AWQ NVFP4：先收 stats

复用上面 AWQ stats 收集命令即可。如果已有 `$QI_AWQ_STATS`，无需重复生成。

```bash
python tests/collect_awq_stats.py \
  --ckpt $QI_BASE_CKPT \
  --output-stats $QI_AWQ_STATS \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --num-inference-steps 5
```

### AWQ NVFP4：生成量化 ckpt

```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_AWQ_NVFP4_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo awq \
  --quant-dtype nvfp4 \
  --quant-bits 4 \
  --quant-group-size 16 \
  --awq-stats $QI_AWQ_STATS \
  --quant-backend flashrt_nvfp4
```

逻辑：读取 AWQ stats -> 计算 per-input-channel `input_scale` -> 对 `weight * input_scale` 调 FlashRT NVFP4 pack -> ckpt 保存 `qweight_packed/scale_factors/weight_global_scale/input_scale/bias`。推理时 backend 会先做 `x / input_scale`，再 runtime quant activation 到 NVFP4。

## 推理：Load Checkpoints with FlashRT NVFP4

### RTN NVFP4

```bash
python tests/test_dry_run.py \
  --ckpt $QI_RTN_NVFP4_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result_rtn_nvfp4_flashrt \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```

### AWQ NVFP4

```bash
python tests/test_dry_run.py \
  --ckpt $QI_AWQ_NVFP4_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result_awq_nvfp4_flashrt \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```

**注意**

- `flashrt_nvfp4` 和现有 `reference` backend 是两条不同路径。`reference` 使用 INT4/INT8 逻辑量化格式，并显式 dequant 后走 `F.linear`；`flashrt_nvfp4` 使用 FlashRT packed FP4 权重和 runtime activation packing。
- `--quant-dtype nvfp4` 必须和 `--quant-backend flashrt_nvfp4` 一起使用。
- `--quant-group-size 16` 是 NVFP4 block-scale 粒度，不是原来 INT4 AWQ 的 `group_size=128`。
- 当前 RTX 4090 (`sm_89`) 不能实际运行 FlashRT NVFP4 kernel；需要 SM120/Blackwell 环境生成和推理 NVFP4 ckpt。

