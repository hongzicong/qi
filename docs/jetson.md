# Jetson Docker

## Build Jetson Docker Image

```bash
# Build the Jetson image on the Jetson host. Do not pass --platform linux/amd64.
docker build -f ./docker/Dockerfile.jetson \
    --tag qi-jetson:latest .

# Optionally override the base image
docker build -f ./docker/Dockerfile.jetson \
    --build-arg BASE_IMAGE=flash-attention:r36.5.tegra-aarch64-cu126-22.04 \
    --tag qi-jetson:latest .
```

## Run Jetson Docker Container

```bash
# Create persistent directories for checkpoints and outputs on the Jetson host
mkdir -p $HOME/qi_jetson/checkpoints $HOME/qi_jetson/output

# Start an interactive container with the Jetson GPU runtime and persistent storage
docker run --rm -it \
    --runtime nvidia \
    --network host \
    --ipc host \
    -v $HOME/qi_jetson:/workspace/pvc \
    qi-jetson:latest
```

## Verify Container Environment

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
python3 -c "import qi; from qi.models.wan22.fastwam import FastWAM; print('qi ok')"
```

## Set Environment Variable

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/pvc/checkpoints
export OUTPUT_PATH=/workspace/pvc/output
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
```

## Download Pretrained Checkpoints from HuggingFace

```bash
huggingface-cli download m1ku2/fastwam_pick_numbered_blocks \
  step_008140.pt \
  ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  dataset_stats.json \
  --local-dir $DIFFSYNTH_MODEL_BASE_PATH
```

## Run Open-Loop Video Dry Run (10 Diffusion Steps, 10 Chunks)

```bash
python3 scripts/dry_run.py \
  --source files \
  --ckpt $DIFFSYNTH_MODEL_BASE_PATH/step_008140.pt \
  --dataset-stats $DIFFSYNTH_MODEL_BASE_PATH/dataset_stats.json \
  --cam-high ./tests/data/cam_high.png \
  --cam-left-wrist ./tests/data/cam_left_wrist.png \
  --cam-right-wrist ./tests/data/cam_right_wrist.png \
  --state-json ./tests/data/state.json \
  --prompt "Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order." \
  --use-text-encoder \
  --output-dir $OUTPUT_PATH/dry_run_result \
  --num-inference-steps 10 \
  --num-chunks 10 \
  --time-inference \
  --seed 42
```
