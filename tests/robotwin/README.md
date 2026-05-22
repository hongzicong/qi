# RobotWin Evaluation Tests

## Model Preparation

This step is required before RobotWin evaluation.

Set the Wan model directory first. If this environment variable is not set, the
default model directory is usually `./checkpoints`.

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Download the released RobotWin checkpoint and its matching dataset stats, or use
your own trained checkpoint and stats file. Replace `/path/to/...` in the
examples below with the actual paths on your machine.

- `ckpt`: model checkpoint, for example `robotwin_uncond_3cam_384.pt`
- `EVALUATION.dataset_stats_path`: matching dataset stats JSON for that checkpoint

Optional: download the released RobotWin checkpoint and stats into
`checkpoints/fastwam_release`:

```bash
pip install -U huggingface_hub

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

## RobotWin Assets Preparation

Full simulation requires RoboTwin assets such as objects, embodiments, textures,
and background textures. Download them under the vendored RoboTwin assets
directory:

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

Then generate/update the embodiment config paths used by RoboTwin:

```bash
cd ../
python script/update_embodiment_config_path.py
```

Return to the qi repo root before running the examples below:

```bash
cd ../../..
```

## Evaluation

Single-task example:

```bash
python tests/robotwin/eval_robotwin_single.py \
  ckpt=/path/to/robotwin_uncond_3cam_384.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=/path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.eval_num_episodes=1
```

Manager example:

```bash
python tests/robotwin/run_robotwin_manager.py \
  ckpt=/path/to/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=/path/to/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=1 \
  MULTIRUN.max_tasks_per_gpu=1
```

The default RoboTwin root is `tests/third_party/RoboTwin`. The entrypoint links
`tests/robotwin/fastwam_policy` into `tests/third_party/RoboTwin/policy` as
`fastwam_policy` before launching RoboTwin's `script/eval_policy.py`.

The vendored RoboTwin tree contains code and configs. Full simulation still
requires the RoboTwin assets under `tests/third_party/RoboTwin/assets`, such as
objects, embodiments, textures, and background textures.
