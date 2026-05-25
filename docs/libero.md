# LIBERO Evaluation

This guide describes how to prepare and run the LIBERO evaluation entrypoints
under `tests/libero`.

## Model Preparation

Set the Wan model directory before running LIBERO evaluation. If this
environment variable is not set, the default model directory is usually
`./checkpoints`.

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Download the released LIBERO checkpoint and its matching dataset stats, or use
your own trained checkpoint and stats file.

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

Then use:

```text
task=libero_uncond_2cam224_1e-4
ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt
EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

## LIBERO Submodule

LIBERO is referenced as a git submodule at:

```text
tests/third_party/LIBERO
```

After cloning this repository, initialize the submodule:

```bash
git submodule update --init --recursive tests/third_party/LIBERO
```

The LIBERO submodule provides the LIBERO code, BDDL files, init states, and
assets used by the evaluation entrypoints. `tests/libero/libero_bootstrap.py`
creates an isolated LIBERO config under `/tmp/qi_libero_config_<project_hash>`
and points it at that vendored runtime.

## LIBERO Dependencies

Before running the LIBERO benchmark, install the official LIBERO simulation
stack, or use an environment that already contains the same runtime
dependencies.

Install the expected LIBERO-side packages:

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

The `mujoco` environment should ideally stay consistent with the LIBERO data
version. `robosuite==1.4.0` may pull or downgrade packages in a way that
conflicts with the qi environment. If NumPy is changed during installation,
reinstall the version required by qi:

```bash
pip install numpy==1.26.4
```

## Single-Task Evaluation

Run one LIBERO task:

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

python tests/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=/path/to/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  EVALUATION.output_dir=./evaluate_results/debug_libero
```

A successful smoke test prints:

```text
Task 0 completed: 1/1 successes
```

## Full Manager Evaluation

Run the manager to evaluate LIBERO task suites:

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

`run_libero_manager.py` launches `tests/libero/run_libero_parallel_test.sh`,
which schedules one `tests/libero/eval_libero_single.py` worker per task through
`tmux`. Install `tmux` in the environment before using the manager.

Supported task suites are:

```text
libero_spatial  10 tasks
libero_object   10 tasks
libero_goal     10 tasks
libero_10       10 tasks
libero_90       90 tasks
```

## Outputs

Rollout videos are written under each evaluation output directory, for example:

```text
evaluate_results/debug_libero/libero_spatial/videos/
```

The vendored LIBERO benchmark loads init states with `weights_only=False` for
PyTorch 2.6+ compatibility. Robosuite may print an ignored EGL cleanup exception
at process exit if `libGLU.so.0` is missing. The evaluation is successful when
the process exits with code 0 and writes the result JSON/video.

## Execution Flow

```text
run_libero_manager.py
  -> read configs through Hydra task overrides
  -> launch run_libero_parallel_test.sh for selected task suites
  -> schedule eval_libero_single.py workers through tmux

eval_libero_single.py
  -> initialize the isolated LIBERO runtime config
  -> load the requested LIBERO task suite and task id
  -> instantiate qi.runtime.create_fastwam
  -> load checkpoint and dataset stats
  -> preprocess LIBERO observation
  -> call model inference for each rollout step
  -> write result JSON and rollout video
```
