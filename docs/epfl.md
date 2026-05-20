# EPFL RCP

## Build and Push Docker Image

```bash
docker build --platform linux/amd64 -f ./docker/Dockerfile.epfl \
    --tag registry.rcp.epfl.ch/dcl-zihong/qi:latest \
    --build-arg LDAP_GROUPNAME=DCL-StaffU \
    --build-arg LDAP_GID=11260 \
    --build-arg LDAP_USERNAME=zihong \
    --build-arg LDAP_UID=322005 .

docker push registry.rcp.epfl.ch/dcl-zihong/qi:latest
```

## Submit and Connect to RunAI Job

```bash
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

---

See [common.md](common.md) for checkpoint download and dry-run instructions.
