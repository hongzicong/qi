# Quantization

## Third-Party Dependencies

### INT8 W8A8
Initialize the CUTLASS submodule:

```bash
git submodule sync --recursive
git submodule update --init --recursive third_party/cutlass
```

Build the native INT8 W8A8 extension:

```bash
QI_BUILD_INT8_W8A8=1 QI_BUILD_INT4_W4A16=0 python setup.py build_ext --inplace
```

Verify that the extension can be imported:

```bash
python -B -c 'from qi.quantization.backends.int8_w8a8_ops import load_int8_w8a8_ops; load_int8_w8a8_ops(required=True); print("Qi INT8 W8A8 backend is available")'
```
### INT4 W4A16

The native INT4 W4A16 backend uses vendored CUDA sources under
`src/qi/quantization/csrc/int4_w4a16`.

Build the Qi-native INT4 W4A16 extension:

```bash
QI_BUILD_INT4_W4A16=1 QI_BUILD_INT8_W8A8=0 python setup.py build_ext --inplace
```

Verify that the INT4 W4A16 extension can be imported:

```bash
python -B -c 'from qi.quantization.backends.int4_w4a16_ops import load_int4_w4a16_ops; load_int4_w4a16_ops(required=True); print("Qi INT4 W4A16 backend is available")'
```

## Environment

Set the project paths used by the commands below:

```bash
export QI_ROOT=/root/autodl-tmp/qi
export PYTHONPATH=$QI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}

export QI_CHECKPOINT_DIR=/root/autodl-tmp/qi/checkpoints
export DIFFSYNTH_MODEL_BASE_PATH=/root/autodl-tmp/asset
export OUTPUT_PATH=/root/autodl-tmp/qi/output

export QI_BASE_CKPT="${QI_CHECKPOINT_DIR}/step_008140.pt"
export QI_DATASET_STATS="${QI_CHECKPOINT_DIR}/dataset_stats.json"

## ref w4a16
export QI_RTN_W4_CKPT="${QI_CHECKPOINT_DIR}/step_008140_rtn_w4.pt"
export QI_AWQ_W4_CKPT="${QI_CHECKPOINT_DIR}/step_008140_awq_w4.pt"
## ref w8a8
export QI_RTN_W8_CKPT="${QI_CHECKPOINT_DIR}/step_008140_rtn_w8.pt"
export QI_SQ_W8_CKPT="${QI_CHECKPOINT_DIR}/step_008140_smoothquant_w8.pt"
## fp8 w8a8
export QI_RTN_FP8_CKPT="${QI_CHECKPOINT_DIR}/step_008140_rtn_fp8.pt"
export QI_SQ_FP8_CKPT="${QI_CHECKPOINT_DIR}/step_008140_smoothquant_fp8.pt"
## int8 w8a8
export QI_RTN_INT8_NATIVE_CKPT="${QI_CHECKPOINT_DIR}/rtn_int8_w8a8_native.pt"
export QI_SQ_INT8_NATIVE_CKPT="${QI_CHECKPOINT_DIR}/sq_int8_w8a8_native.pt"
## int4 w4a16
export QI_AWQ_CACHE="${QI_CHECKPOINT_DIR}/awq_calibration_cache.pt"
export QI_RTN_INT4_CKPT="${QI_CHECKPOINT_DIR}/rtn_int4_w4a16.pt"
export QI_AWQ_INT4_CKPT="${QI_CHECKPOINT_DIR}/awq_int4_w4a16.pt"

```

Adjust `DIFFSYNTH_MODEL_BASE_PATH`, `QI_CHECKPOINT_DIR`, and `OUTPUT_PATH` if
your checkpoint or dataset statistics live elsewhere.

## Create Checkpoint

### INT8 W8A8

Run the INT8 W8A8 RTN quantization command:

```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_RTN_INT8_NATIVE_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo rtn \
  --quant-dtype int \
  --quant-bits 8 \
  --activation-granularity per_token \
  --weight-granularity per_channel \
  --quant-backend int8_w8a8
```

At inference time, activations are quantized dynamically per token, and the Qi
native CUTLASS kernel runs the INT8 GEMM.

Collect the activation cache used by SmoothQuant and AWQ:

```bash
python tests/collect_awq_cache.py \
  --ckpt $QI_BASE_CKPT \
  --output $QI_AWQ_CACHE \
  --dataset-stats $QI_DATASET_STATS \
  --model-base-path $DIFFSYNTH_MODEL_BASE_PATH \
  --skip-download
```

The cache stores per-module `abs_mean` statistics for SmoothQuant and input
samples for AWQ. Collect it with the same target module and target expert
settings used by the quantization commands.

Run the INT8 W8A8 SmoothQuant quantization command:

```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_SQ_INT8_NATIVE_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo smoothquant \
  --quant-dtype int \
  --quant-bits 8 \
  --activation-granularity per_token \
  --weight-granularity per_channel \
  --quant-backend int8_w8a8 \
  --awq-stats $QI_AWQ_CACHE
```

SmoothQuant uses the activation statistics from `--awq-stats` to compute input
scales, then stores those scales with the quantized INT8 weights.

### INT4 W4A16

Run the INT4 W4A16 RTN quantization command:

```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_RTN_INT4_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo rtn \
  --quant-dtype int \
  --quant-bits 4 \
  --quant-group-size 128 \
  --quant-backend int4_w4a16
```

Run the INT4 W4A16 AWQ quantization command:

```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_AWQ_INT4_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo awq \
  --quant-dtype int \
  --quant-bits 4 \
  --quant-group-size 128 \
  --quant-backend int4_w4a16 \
  --awq-stats $QI_AWQ_CACHE
```

RTN directly rounds weights into the INT4 W4A16 packed layout. AWQ uses the
calibration cache to search per-channel scales before packing INT4 weights for
the native backend.


## Inference with quantized checkpoints

### INT8 W8A8

Run the INT8 W8A8 SmoothQuant checkpoint with the native INT8 kernel:

```bash
python tests/test_dry_run.py \
  --ckpt $QI_SQ_INT8_NATIVE_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_output/int8_w8a8_smoothquant \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```

### INT4 W4A16

Run the INT4 W4A16 AWQ checkpoint with the native INT4 kernel:

```bash
python tests/test_dry_run.py \
  --ckpt $QI_AWQ_INT4_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_output/int4_w4a16_awq \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```

For RTN checkpoints, use the same commands and replace only `--ckpt` with
`$QI_RTN_INT8_NATIVE_CKPT` or `$QI_RTN_INT4_CKPT`.

## Supported Options

Other supported options are listed here (reference backend = fake quant):

| Use case | `quant-dtype` | `quant-algo` | `quant-bits` | `quant-backend` | Extra args |
|---|---|---|---:|---|---|
| Native INT8 W8A8 | `int` | `rtn` | `8` | `int8_w8a8` | `--activation-granularity per_token --weight-granularity per_channel` |
| Native INT8 W8A8 SmoothQuant | `int` | `smoothquant` | `8` | `int8_w8a8` | same granularity args plus `--awq-stats $QI_AWQ_CACHE` |
| PyTorch INT8 W8A8 baseline | `int` | `rtn` or `smoothquant` | `8` | `reference` | `--activation-granularity per_token --weight-granularity per_channel`; SmoothQuant also needs `--awq-stats $QI_AWQ_CACHE` |
| Weight-only RTN baseline | `int` | `rtn` | `4` or `8` | `reference` | optional `--quant-group-size 128` |
| Weight-only AWQ baseline | `int` | `awq` | `4` or `8` | `reference` | `--awq-stats $QI_AWQ_CACHE` |
| Native INT4 RTN W4A16 | `int` | `rtn` | `4` | `int4_w4a16` | `--quant-group-size 128` |
| Native INT4 AWQ W4A16 | `int` | `awq` | `4` | `int4_w4a16` | `--quant-group-size 128 --awq-stats $QI_AWQ_CACHE` |
| FP8 FlashRT | `fp8` | `rtn` or `smoothquant` | `8` | `flashrt_fp8` | SmoothQuant needs `--awq-stats $QI_AWQ_CACHE`; FlashRT runtime required |

