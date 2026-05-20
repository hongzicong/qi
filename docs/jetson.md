# Jetson Docker

## Build Docker Image

```bash
# Build on the Jetson host — do not pass --platform linux/amd64
docker build -f ./docker/Dockerfile.jetson \
    --tag qi-jetson:latest .

# Optionally override the base image
docker build -f ./docker/Dockerfile.jetson \
    --build-arg BASE_IMAGE=flash-attention:r36.5.tegra-aarch64-cu126-22.04 \
    --tag qi-jetson:latest .
```

## Start and Connect to Container

```bash
# Create persistent directories on the Jetson host
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

---

See [common.md](common.md) for checkpoint download and dry-run instructions.
