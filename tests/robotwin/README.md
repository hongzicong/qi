# RobotWin Evaluation Tests

This directory contains the RobotWin evaluation adapter migrated from
`physical_WM/experiments/robotwin` and wired to the `qi` package.

Single-task example:

```bash
python tests/robotwin/eval_robotwin_single.py \
  ckpt=/path/to/step_xxxxxx.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  EVALUATION.eval_num_episodes=1
```

Manager example:

```bash
python tests/robotwin/run_robotwin_manager.py \
  ckpt=/path/to/step_xxxxxx.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

The default RoboTwin root is `tests/third_party/RoboTwin`. The entrypoint links
`tests/robotwin/fastwam_policy` into `tests/third_party/RoboTwin/policy` as
`fastwam_policy` before launching RoboTwin's `script/eval_policy.py`.

The vendored RoboTwin tree contains code and configs. Full simulation still
requires the RoboTwin assets under `tests/third_party/RoboTwin/assets`, such as
objects, embodiments, textures, and background textures.
