# LIBERO Evaluation Tests

## Model Preparation

This step is required before LIBERO evaluation.

Set the Wan model directory first. If this environment variable is not set, the
default model directory is usually `./checkpoints`.

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Download the released LIBERO checkpoint and its matching dataset stats, or use
your own trained checkpoint and stats file. Replace `/path/to/...` in the
examples below with the actual paths on your machine.

- `task`: Hydra task config, for example `libero_uncond_2cam224_1e-4`
- `ckpt`: model checkpoint, for example `libero_uncond_2cam224.pt`
- `EVALUATION.dataset_stats_path`: matching dataset stats JSON for that checkpoint

Optional: download the released LIBERO checkpoint and stats into
`checkpoints/fastwam_release`:

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

Then use:

```text
ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt
EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

## LIBERO Assets Preparation

The vendored LIBERO tree under `tests/third_party/LIBERO` contains the LIBERO
code, bddl files, init states, and assets used by these entrypoints.
`tests/libero/libero_bootstrap.py` creates an isolated LIBERO config under
`/tmp/qi_libero_config_<project_hash>` and points it at that vendored runtime.

Before running the LIBERO benchmark, install the official LIBERO simulation stack
from the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO),
or use an environment that already contains the same runtime dependencies.

The expected LIBERO-side packages include:

```bash
pip install \
  robosuite==1.4.0 \
  bddl==1.0.1 \
  gym==0.25.2 \
  easydict==1.13
```

Then install the MuJoCo version used by this test setup:

```bash
pip install mujoco==3.3.2
```

The `mujoco` environment should ideally stay consistent with the LIBERO data version.

`robosuite==1.4.0` may pull or downgrade packages in a way that conflicts with
the qi environment. If NumPy is changed during installation, reinstall the
version required by qi:

```bash
pip install numpy==1.26.4
```

## Evaluation

Single-task example:

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

python tests/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=~/autodl-tmp/libero_ckp/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=~/autodl-tmp/libero_ckp/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  EVALUATION.output_dir=./evaluate_results/debug_libero
```

Expected result:

```text
Task 0 completed: 1/1 successes
```

Manager example:

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

python tests/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/path/to/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.num_trials=1 \
  MULTIRUN.task_suite_names=[libero_spatial] \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

The manager launches `tests/libero/run_libero_parallel_test.sh`, which schedules
one `tests/libero/eval_libero_single.py` worker per task through `tmux`. Install
`tmux` in the environment before using the manager.

Supported task suites are:

```text
libero_spatial  10 tasks
libero_object   10 tasks
libero_goal     10 tasks
libero_10       10 tasks
libero_90       90 tasks
```

Rollout videos are written under each evaluation output directory, for example:

```text
evaluate_results/debug_libero/libero_spatial/videos/
```

The vendored LIBERO benchmark loads init states with `weights_only=False` for
PyTorch 2.6+ compatibility. Robosuite may print an ignored EGL cleanup exception
at process exit if `libGLU.so.0` is missing. The evaluation is successful when
the process exits with code 0 and writes the result JSON/video.
