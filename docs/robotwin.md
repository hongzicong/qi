# RobotWin Evaluation

This guide describes how to prepare and run the RobotWin evaluation entrypoints
under `tests/robotwin`.

## Model Preparation

Set the Wan model directory before running RobotWin evaluation. Use an absolute
path so subprocesses launched inside `tests/third_party/RoboTwin` still share
the same model cache.

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Download the released RobotWin checkpoint and its matching dataset stats, or use
your own trained checkpoint and stats file.

```bash
huggingface-cli download yuanty/fastwam \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

Then use:

```text
ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
```

## RoboTwin Submodule

RoboTwin is referenced as a git submodule at:

```text
tests/third_party/RoboTwin
```

After cloning this repository, initialize the submodule:

```bash
git submodule update --init --recursive tests/third_party/RoboTwin
```

## RoboTwin Assets

Full simulation requires RoboTwin assets such as objects, embodiments, textures,
and background textures. Download them under the RoboTwin submodule:

```bash
cd tests/third_party/RoboTwin/assets
python _download.py
```

The download script writes zip files into the current directory. Extract them so
the runtime can find directories such as `assets/objects` and
`assets/embodiments`:

```bash
unzip -o background_texture.zip
unzip -o embodiments.zip
unzip -o objects.zip
```

Then generate or update the embodiment config paths used by RoboTwin:

```bash
cd ../
python script/update_embodiment_config_path.py
```

Return to the qi repo root before running evaluation:

```bash
cd ../../..
```

## Optional cuRobo Planner

Some RoboTwin embodiments or planner configurations may require cuRobo. If you
hit an error such as `ModuleNotFoundError: No module named 'curobo.types.math'`,
install cuRobo under the RoboTwin envs directory:

```bash
cd tests/third_party/RoboTwin/envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
git fetch --tags
git checkout v0.7.8
pip install -e . --no-build-isolation
```

Return to the qi repo root after installation:

```bash
cd ../../../..
```

## Single-Task Evaluation

Run one RoboTwin task:

```bash
python tests/robotwin/eval_robotwin_single.py \
  ckpt=/path/to/robotwin_uncond_3cam_384.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=/path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.eval_num_episodes=1
```

`eval_robotwin_single.py` launches RoboTwin's `script/eval_policy.py` inside the
RoboTwin submodule and links `tests/robotwin/fastwam_policy` into RoboTwin's
`policy/fastwam_policy` path before evaluation.

## Full Manager Evaluation

Run the manager to evaluate all RoboTwin tasks listed by RoboTwin's
`task_config/_eval_step_limit.yml`:

```bash
python tests/robotwin/run_robotwin_manager.py \
  ckpt=/path/to/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=/path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

If `EVALUATION.task_name` is omitted, the manager runs every task. For each task,
it runs two phases:

```text
clean  -> demo_clean
random -> demo_randomized
```

For a quick smoke test, restrict the manager to one task and one episode:

```bash
python tests/robotwin/run_robotwin_manager.py \
  ckpt=/path/to/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=/path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.eval_num_episodes=1 \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

## Accelerated Evaluation

RobotWin supports the same action-inference acceleration overrides as LIBERO.
The flags work with both `eval_robotwin_single.py` and
`run_robotwin_manager.py`.

### Baseline

```bash
python tests/robotwin/eval_robotwin_single.py \
  ckpt=/path/to/robotwin_uncond_3cam_384.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=/path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.expert_cache=false \
  EVALUATION.cuda_graph=false \
  EVALUATION.torch_compile=false
```

### + DiT Caching

```bash
python tests/robotwin/eval_robotwin_single.py \
  ckpt=/path/to/robotwin_uncond_3cam_384.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=/path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.expert_cache=true \
  EVALUATION.cuda_graph=false \
  EVALUATION.torch_compile=false
```

### + DiT Caching + CUDA Graph

```bash
python tests/robotwin/eval_robotwin_single.py \
  ckpt=/path/to/robotwin_uncond_3cam_384.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=/path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.expert_cache=true \
  EVALUATION.cuda_graph=true \
  EVALUATION.torch_compile=false
```

### + DiT Caching + CUDA Graph + torch.compile

```bash
python tests/robotwin/eval_robotwin_single.py \
  ckpt=/path/to/robotwin_uncond_3cam_384.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=/path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.expert_cache=true \
  EVALUATION.cuda_graph=true \
  EVALUATION.torch_compile=true
```

## Outputs

Single-task workers write per-phase result files:

```text
evaluate_results/robotwin/<ckpt_tag>/<run_ts>/<task_name>/_result_clean.txt
evaluate_results/robotwin/<ckpt_tag>/<run_ts>/<task_name>/_result_random.txt
```

The manager reads those files and writes:

```text
evaluate_results/robotwin/<ckpt_tag>/<run_ts>/summary.csv
evaluate_results/robotwin/<ckpt_tag>/<run_ts>/summary.json
```

`summary.json` contains the overall averages:

```json
{
  "overall": {
    "clean_mean_success_rate": 0.0,
    "random_mean_success_rate": 0.0
  }
}
```

## Execution Flow

```text
run_robotwin_manager.py
  -> read configs/sim_robotwin.yaml
  -> load tasks from tests/third_party/RoboTwin/task_config/_eval_step_limit.yml
  -> launch eval_robotwin_single.py for clean/random phases

eval_robotwin_single.py
  -> link tests/robotwin/fastwam_policy to RoboTwin/policy/fastwam_policy
  -> launch tests/third_party/RoboTwin/script/eval_policy.py

RoboTwin script/eval_policy.py
  -> create RoboTwin env
  -> import fastwam_policy
  -> call fastwam_policy.get_model()

tests/robotwin/fastwam_policy/deploy_policy.py
  -> read qi configs
  -> instantiate qi.runtime.create_fastwam
  -> load checkpoint and dataset stats
  -> preprocess RoboTwin observation
  -> call self.model.infer_action()

src/qi/models/wan22/fastwam.py
  -> FastWAM.infer_action()
  -> return predicted action

deploy_policy.py
  -> denormalize action
  -> task_env.take_action(...)

run_robotwin_manager.py
  -> parse result files
  -> write summary.csv and summary.json
```
