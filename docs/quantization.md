# Quantization

This guide shows the default path for creating a Qi native INT8 W8A8 checkpoint.
The native INT8 backend uses CUTLASS for the GEMM kernels.

## Third-Party Dependencies

Initialize the CUTLASS submodule:

```bash
git submodule sync --recursive
git submodule update --init --recursive third_party/cutlass
```

Build the native INT8 W8A8 extension:

```bash
QI_BUILD_INT8_W8A8=1 python setup.py build_ext --inplace
```

Verify that the extension can be imported:

```bash
python -B -c 'from qi.quantization.backends.int8_w8a8_ops import load_int8_w8a8_ops; load_int8_w8a8_ops(required=True); print("Qi INT8 W8A8 backend is available")'
```

## Environment

Set the project paths used by the commands below:

```bash
export QI_ROOT=/root/autodl-tmp/qi
export PYTHONPATH=$QI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}

export DIFFSYNTH_MODEL_BASE_PATH=/root/autodl-tmp/asset
export QI_BASE_CKPT=$DIFFSYNTH_MODEL_BASE_PATH/step_008140.pt
export QI_DATASET_STATS=$DIFFSYNTH_MODEL_BASE_PATH/dataset_stats.json
export QI_SQ_INT8_NATIVE_CKPT=$QI_ROOT/output/rtn_int8_w8a8_native.pt
```

Adjust `DIFFSYNTH_MODEL_BASE_PATH`, `QI_BASE_CKPT`, and `QI_DATASET_STATS` if
your checkpoint or dataset statistics live elsewhere.

## Create an INT8 W8A8 Checkpoint

Run the main quantization command:

```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_SQ_INT8_NATIVE_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo rtn \
  --quant-dtype int \
  --quant-bits 8 \
  --activation-granularity per_token \
  --weight-granularity per_channel \
  --quant-backend int8_w8a8
```

This creates an INT8 W8A8 checkpoint using RTN weight quantization and the Qi
native CUTLASS backend. The `rtn` algorithm does not require `--awq-stats`.

## What the Command Does

The command:

- loads the base checkpoint from `--ckpt`;
- builds a `WeightActivationQuantConfig`;
- quantizes selected `nn.Linear` weights to per-channel INT8;
- stores `backend=int8_w8a8` in the checkpoint quantization metadata;
- saves the quantized checkpoint to `--output-ckpt`.

At inference time, activations are quantized dynamically per token, and the Qi
native CUTLASS kernel runs the INT8 GEMM.

## Supported Options

| Use case | `quant-dtype` | `quant-algo` | `quant-bits` | `quant-backend` | Extra args |
|---|---|---|---:|---|---|
| Native INT8 W8A8 | `int` | `rtn` | `8` | `int8_w8a8` | `--activation-granularity per_token --weight-granularity per_channel` |
| Native INT8 W8A8 SmoothQuant | `int` | `smoothquant` | `8` | `int8_w8a8` | same granularity args plus `--awq-stats $QI_AWQ_STATS` |
| PyTorch INT8 W8A8 baseline | `int` | `rtn` or `smoothquant` | `8` | `reference` | SmoothQuant needs `--awq-stats` |
| Weight-only RTN baseline | `int` | `rtn` | `4` or `8` | `reference` | optional `--quant-group-size 128` |
| Weight-only AWQ baseline | `int` | `awq` | `4` or `8` | `reference` | `--awq-stats $QI_AWQ_STATS` |
| FP8 FlashRT | `fp8` | `rtn` or `smoothquant` | `8` | `flashrt_fp8` | SmoothQuant needs `--awq-stats`; FlashRT runtime required |
