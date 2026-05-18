# EPFL RCP

## Build and Push Docker Image

```bash
# Build base (only when dependencies change)
docker build --platform linux/amd64 -f ./docker/Dockerfile.base \
    --tag registry.rcp.epfl.ch/dcl-zihong/qi-base:latest .

# Build app (every time code changes)
docker build --platform linux/amd64 -f ./docker/Dockerfile.epfl.app \
    --tag registry.rcp.epfl.ch/dcl-zihong/qi:latest \
    --build-arg LDAP_GROUPNAME=DCL-StaffU \
    --build-arg LDAP_GID=11260 \
    --build-arg LDAP_USERNAME=zihong \
    --build-arg LDAP_UID=322005 .

docker push registry.rcp.epfl.ch/dcl-zihong/qi:latest
```

## Submit and Connect to RunAI Job

```shell
# Login to RunAI
runai login

# Submit an interactive job with 1 GPU, 50 CPUs, 100G memory, and a persistent home volume
runai submit \
    --image registry.rcp.epfl.ch/dcl-zihong/qi:latest \
    --gpu 1 \
    --existing-pvc claimname=home,path=/home/zihong/pvc \
    --cpu 50 \
    --memory 100G \
    --large-shm \
    --interactive \
    --node-pool default \
    -- sleep infinity

# Connect to the running job (replace job ID as needed)
runai bash job-ae3fb1f7996c
```

## Set Environment Variable

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/home/zihong/pvc/checkpoints
export OUTPUT_PATH=/home/zihong/pvc/output
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
python scripts/dry_run.py \
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
  --expert-cache \
  --seed 42
```