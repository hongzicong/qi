# Common Steps

Run these steps after setting up your container environment and environment variables (see the platform-specific guide).

## Download Pretrained Checkpoints

```bash
huggingface-cli download m1ku2/fastwam_pick_numbered_blocks \
  step_008140.pt \
  ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  dataset_stats.json \
  --local-dir $DIFFSYNTH_MODEL_BASE_PATH
```

## Run Open-Loop Video Dry Run (10 Diffusion Steps, 10 Chunks)

### Baseline

```bash
python tests/test_dry_run.py \
  --ckpt $DIFFSYNTH_MODEL_BASE_PATH/step_008140.pt \
  --dataset-stats $DIFFSYNTH_MODEL_BASE_PATH/dataset_stats.json \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --seed 42
```

### + Dit Caching

```bash
python tests/test_dry_run.py \
  --ckpt $DIFFSYNTH_MODEL_BASE_PATH/step_008140.pt \
  --dataset-stats $DIFFSYNTH_MODEL_BASE_PATH/dataset_stats.json \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --no-cuda-graph \
  --expert-cache \
  --seed 42
```

### + Dit Caching + Cuda Graph

```bash
python tests/test_dry_run.py \
  --ckpt $DIFFSYNTH_MODEL_BASE_PATH/step_008140.pt \
  --dataset-stats $DIFFSYNTH_MODEL_BASE_PATH/dataset_stats.json \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --expert-cache \
  --seed 42
```

### + Dit Caching + Cuda Graph + torch.compile

```bash
python tests/test_dry_run.py \
  --ckpt $DIFFSYNTH_MODEL_BASE_PATH/step_008140.pt \
  --dataset-stats $DIFFSYNTH_MODEL_BASE_PATH/dataset_stats.json \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --expert-cache \
  --torch-compile \
  --seed 42
```

### Nsys Profiler

```bash
nsys profile \
    -w true \
    -t cuda,nvtx,osrt,cudnn,cublas \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --gpu-metrics-devices=all \
    -o output/cudagraph_cache_torchcompile \
python  tests/test_dry_run.py\
  --ckpt $DIFFSYNTH_MODEL_BASE_PATH/step_008140.pt \
  --dataset-stats $DIFFSYNTH_MODEL_BASE_PATH/dataset_stats.json \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --output-dir $OUTPUT_PATH/dry_run_result \
  --num-inference-steps 10 \
  --num-chunks 20 \
  --seed 42 \
  --expert-cache \
  --torch-compile
```