
## Prerequisite
```bash
export QI_ROOT="${QI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONPATH="${QI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export FLASHRT_ROOT="${FLASHRT_ROOT:-${QI_ROOT}/third_party/FlashRT}"
if [ -d "${FLASHRT_ROOT}/flash_rt" ]; then
  export PYTHONPATH="${FLASHRT_ROOT}:${PYTHONPATH}"
fi

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${QI_ROOT}/checkpoints}"
export OUTPUT_PATH="${OUTPUT_PATH:-${QI_ROOT}/output}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export QI_BASE_CKPT="${QI_BASE_CKPT:-${DIFFSYNTH_MODEL_BASE_PATH}/step_008140.pt}"
export QI_DATASET_STATS="${QI_DATASET_STATS:-${DIFFSYNTH_MODEL_BASE_PATH}/dataset_stats.json}"
export QI_AWQ_STATS="${QI_AWQ_STATS:-${DIFFSYNTH_MODEL_BASE_PATH}/awq_stats.pt}"

export QI_RTN_CKPT="${QI_RTN_CKPT:-${DIFFSYNTH_MODEL_BASE_PATH}/step_008140_rtn_w8.pt}"
export QI_AWQ_CKPT="${QI_AWQ_CKPT:-${DIFFSYNTH_MODEL_BASE_PATH}/step_008140_awq_w4.pt}"
export QI_SQ_INT8_CKPT="${QI_SQ_INT8_CKPT:-${DIFFSYNTH_MODEL_BASE_PATH}/step_008140_smoothquant_int8.pt}"
export QI_RTN_FP8_CKPT="${DIFFSYNTH_MODEL_BASE_PATH}/step_008140_rtn_fp8.pt"
export QI_SQ_FP8_CKPT="${DIFFSYNTH_MODEL_BASE_PATH}/step_008140_smoothquant_fp8.pt"


mkdir -p "${DIFFSYNTH_MODEL_BASE_PATH}" "${OUTPUT_PATH}"

echo "QI_ROOT=${QI_ROOT}"
echo "DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
```
For FlashRT installation, see [`flashrt.md`](flashrt.md).

## Prepare Quantized Checkpoints

### RTN INT8 W8A16 per-group backend=ref

```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_RTN_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo rtn \
  --quant-bits 8 \
  --quant-backend reference
```
### AWQ INT4 W4A16 per-group backend=ref

with no calibration dataset at the moment, use ```--num-inference-steps 5``` instead
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
### Smoothquant INT8 W8A8 per-token/channel backend=ref

dynamic activation scale

stat collected by awq is also available here


```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_SQ_INT8_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo smoothquant \
  --quant-dtype int \
  --quant-bits 8 \
  --awq-stats $QI_AWQ_STATS \
  --activation-granularity per_token \
  --weight-granularity per_channel \
  --quant-backend reference
```

### RTN FP8 W8A8 per-tensor backend=flashrt_fp8

static activation scale
```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_RTN_FP8_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo rtn \
  --quant-dtype fp8 \
  --quant-backend flashrt_fp8
```
### SmoothQuant FP8 W8A8 per-tensor backend=flashrt_fp8

static activation scale

stat collected by awq is also available here

```bash
python tests/quantize_checkpoint.py \
  --ckpt $QI_BASE_CKPT \
  --output-ckpt $QI_SQ_FP8_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --quant-algo smoothquant \
  --quant-dtype fp8 \
  --awq-stats $QI_AWQ_STATS \
  --quant-backend flashrt_fp8
```

## Inference: Load Checkpoints

### Baseline BF16

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

Baseline FP checkpoint，`num_inference_steps=10`，`num_chunks=20`，`seed=42`

| Metric (ms) | Baseline FP checkpoint | `infer_action` only |
|-------------|------------------------|---------------------|
| Mean        | 1039                   | 299                 |
| P50         | 1040                   | 299                 |
| P99         | 1052                   | 304                 |
| Max         | 1053                   | 304                 |

### RTN INT8 W8A16 per-group backend=ref

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

| Metric (ms) | RTN W8A16 fake-quant | `infer_action` only |
|-------------|----------------------|---------------------|
| Mean        | 1041                 | 299                 |
| P50         | 1039                 | 297                 |
| P99         | 1057                 | 304                 |
| Max         | 1069                 | 308                 |


### AWQ INT4 W4A16 per-group backend=ref


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

| Metric (ms) | AWQ W4A16 fake-quant | `infer_action` only |
|-------------|----------------------|---------------------|
| Mean        | 1081                 | 312                 |
| P50         | 1077                 | 308                 |
| P99         | 1145                 | 344                 |
| Max         | 1151                 | 347                 |

### Comparison

| Checkpoint | End-to-end Mean (ms) | `infer_action` Mean (ms) |
|------------|----------------------|--------------------------|
| Baseline FP checkpoint | 1039 | 299 |
| RTN W8A16 fake-quant (`reference`) | 1041 | 299 |
| AWQ W4A16 fake-quant (`reference`) | 1081 | 312 |

### Smoothquant INT8 W8A8 per-token/channel backend=ref

```bash
python tests/test_dry_run.py \
  --ckpt $QI_SQ_INT8_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result_smoothquant_int8_ref \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```

### RTN FP8 W8A8 per-tensor backend=flashrt_fp8

```bash
python tests/test_dry_run.py \
  --ckpt $QI_RTN_FP8_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result_rtn_fp8_flashrt \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```


### SmoothQuant FP8 W8A8 per-tensor backend=flashrt_fp8

```bash
python tests/test_dry_run.py \
  --ckpt $QI_SQ_FP8_CKPT \
  --dataset-stats $QI_DATASET_STATS \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result_smoothquant_fp8_flashrt \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```
