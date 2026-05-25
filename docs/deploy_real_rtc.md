# Deploying FastWAM on Real Robots (Real-Time Chunking)

This guide documents how to use `tests/real_robot/deploy_real_rtc.py` to run FastWAM inferences on real dual-arm robots with real-time chunking.

## Usage

```bash
python tests/real_robot/deploy_real_rtc.py --deploy-config configs/deploy_real_rtc.yaml
```

During execution, press **Enter** to start the inference loop. Whenever you want to stop the current trial and save the actions locally, press the **Space** key. The script will wait for your confirmation to proceed to the next inference cycle.

## Configuration Parameters (`deploy_real_rtc.yaml`)

The deployment behavior is fully configured via `configs/deploy_real_rtc.yaml`. Below is a breakdown of the key parameters:

### Robot and CAN Settings
- `left_channel`: CAN channel name for the left arm (e.g., `"can_left"`).
- `right_channel`: CAN channel name for the right arm (e.g., `"can_right"`).
- `bitrate`: CAN bus bitrate (usually `1000000`).
- `speed_pct`: Robot execution velocity percentage.
- `max_linear_vel`, `max_angular_vel`, `max_linear_acc`, `max_angular_acc`: Velocity and acceleration limits for safety.
- `LEFT_INIT_POSITION`, `RIGHT_INIT_POSITION`: Joint home positions `[0.0, ..., 0.0, 0.05]` to which the arms reset at the beginning of each trial.
- `rospy_rate`: Control loop frequency in Hz (e.g., `50`).

### Safety Thresholds
- `ACTION_SAFETY_THRESHOLD`: Maximum allowed absolute difference (L1 norm) between the last action and the predicted action sequence. Halts execution if exceeded.
- `STATE_SAFETY_THRESHOLD`: Maximum allowed difference between the current read robot state and the scheduled action to prevent sudden jumps.

### Inference & Model Properties
- `ckpt`: Absolute or relative path to your PyTorch model checkpoint.
- `dataset_stats`: Path to your validation `dataset_stats.json`.
- `prompt`: A natural language instruction for the current task (e.g., `"clean the table"`).
- `context_cache_dir`:  Text embedding cache directory
- `use_text_encoder`:  Whether to use text encoder instead of cached embeds
- `action_horizon`: Model's generated action sequence length (e.g., `32`).
- `num_inference_steps`: Number of diffusion denoising steps (e.g., `10`).
- `execute_steps`: The amount of steps played per prediction chunk (e.g., `1`).
- `exp_weight_factor`: Used in real-time chunking to exponentially weight overlapping inferred future actions.
- `seed`: Random generation seed.
- `dry_run`: `true` To compute inferences without dispatching to physical arms, `false` otherwise.

### Data Saving
- `output_actions`: `.npy` File path pattern to save predicted actions when stopping a trial.
- `output_actions_json`: `.json` File path pattern to save predictions.
- `episode_idx_dict`: A dictionary to track the current episode index for success and failure cases.
---

## About `use_text_encoder` and Text Embeddings

By default, FastWAM inference relies on precomputed text embeddings caching. **It does not load the text encoder at runtime.**
This strategy ensures **faster startup times** and **reduced VRAM consumption**.

### Using Cache (Default & Recommended)

To prepopulate the cache for your prompts, compute the text embeddings offline before deploying:

```bash
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python tests/real_robot/precompute_text_embeds.py \
  data=your_data_name \
  model=fastwam \
  '+override_instruction="your prompt"' \
  '+overwrite=false'
```
The embeddings will be stored in the `text_embedding_cache_dir` in your dataset's YAML config. 

*Note: Make sure `your_data_name` aligns exactly with your dataset's YAML config filename inside the `configs/data/` directory, for example `real_cleaning2task.yaml`.*

Then, configure the directory path in `deploy_real_rtc.yaml`:
```yaml
context_cache_dir: "your_text_embedding_cache_dir"
use_text_encoder: false
```

### Loading Text Encoder at Runtime (`--use-text-encoder`)

If you want dynamic prompts without precomputing embeddings, you can set `use_text_encoder: true` inside the YAML, or launch with an override flag:

```bash
python tests/real_robot/deploy_real_rtc.py --use-text-encoder
```

*Note: Enabling this option loads the full tokenizer and text encoder into memory during reasoning. This will result in a noticeably slower startup time and higher VRAM consumption.*
