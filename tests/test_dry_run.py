#!/usr/bin/env python3
"""Open-loop WAM video dry-run for real-data checkpoints.

This entrypoint intentionally does not import the real-robot deployment scripts:
it should be safe to run on a training/evaluation server without ROS, CAN, or
robot SDK dependencies. It can either sample an existing LeRobot dataset item
and save predicted-vs-GT videos, or use one RGB snapshot from each camera plus a
state JSON file and save the predicted future video.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image

from qi.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from qi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from qi.utils.config_resolvers import register_default_resolvers
from qi.utils.image_utils import load_rgb_image, load_state_vector, pil_frame_to_input_image, preprocess_real_images, save_image
from qi.utils.logging_config import get_logger, setup_logging
from qi.utils.video_utils import save_mp4
from qi.api import load_model, load_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

register_default_resolvers()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WAM open-loop video inference without touching robot IO."
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoints/weights/step_xxxxxx.pt.")
    parser.add_argument("--dataset-stats", required=True, help="Path to dataset_stats.json from the same run/data.")
    parser.add_argument("--task", default="real_cleaning_uncond_3cam_384_1e-4")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--source", choices=["files"], default="files", help="Compatibility flag; this entrypoint runs files mode.")
    parser.add_argument("--output-dir", default="open_loop_video_dry_run")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--num-video-frames", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--profile-components",
        action="store_true",
        help="Profile infer_action encoder, video prefill, and action diffusion component timings.",
    )
    parser.add_argument(
        "--cuda-graph-infer-action",
        action="store_true",
        help="Capture infer_action denoising with CUDA Graph. With --expert-cache, captures compute and reuse paths separately.",
    )
    parser.add_argument(
        "--cuda-graph-warmup-steps",
        type=int,
        default=3,
        help="Warmup iterations before capturing each infer_action CUDA graph.",
    )
    parser.add_argument(
        "--torch-compile-infer-action",
        action="store_true",
        help="Compile infer_action compute/reuse tensor paths before optional CUDA Graph capture.",
    )
    parser.add_argument(
        "--torch-compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require fullgraph=True for infer_action tensor path compilation.",
    )
    parser.add_argument(
        "--expert-cache",
        action="store_true",
        help="Enable fixed-step action expert DiT body residual caching during inference.",
    )
    parser.add_argument(
        "--expert-cache-reuse-steps",
        type=int,
        default=1,
        help="Number of adjacent denoising steps that reuse the previous computed action expert body residual.",
    )
    parser.add_argument(
        "--expert-cache-warmup-steps",
        type=int,
        default=1,
        help="Number of initial action denoising steps that always compute the DiT body.",
    )
    parser.add_argument(
        "--expert-cache-cooldown-steps",
        type=int,
        default=1,
        help="Number of final action denoising steps that always compute the DiT body.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help=(
            "Files mode only: repeatedly generate this many chunks. "
            "Each chunk after the first uses the previous predicted last frame as the new condition image."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument(
        "--time-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print wall-clock inference time. Enabled by default.",
    )
    parser.add_argument("--use-text-encoder", action="store_true", default=True, help="Compatibility flag; text encoder is loaded.")
    
    parser.add_argument("--cam-high", default=None)
    parser.add_argument("--cam-left-wrist", default=None)
    parser.add_argument("--cam-right-wrist", default=None)
    parser.add_argument("--state-json", default=None)
    parser.add_argument("--prompt", default="clean the table")
    return parser.parse_args()


def default_num_video_frames(cfg: DictConfig) -> int:
    num_frames = int(cfg.data.train.num_frames)
    ratio = int(cfg.data.train.get("action_video_freq_ratio", 1))
    return ((num_frames - 1) // ratio) + 1


def default_action_horizon(cfg: DictConfig) -> int:
    return int(cfg.data.train.num_frames) - 1


def save_actions(
    action: torch.Tensor | np.ndarray | None,
    output_dir: Path,
    prefix: str,
    processor: Any | None = None,
    proprio: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if action is None:
        return None
    action_tensor = action.detach().cpu().to(torch.float32) if isinstance(action, torch.Tensor) else torch.as_tensor(action)
    action_denorm = denormalize_merged_action(processor, action_tensor, proprio) if processor is not None else action_tensor
    action_np = action_denorm.detach().cpu().numpy().astype(np.float32)
    np.save(output_dir / f"{prefix}_pred_actions.npy", action_np)
    with open(output_dir / f"{prefix}_pred_actions.json", "w", encoding="utf-8") as file:
        json.dump({"action": action_np.tolist()}, file, ensure_ascii=False, indent=2)
    return action_denorm


def denormalize_merged_action(
    processor: Any,
    action: torch.Tensor,
    proprio: torch.Tensor | None,
) -> torch.Tensor:
    if action.ndim == 2:
        action_btd = action.unsqueeze(0)
    elif action.ndim == 3 and action.shape[0] == 1:
        action_btd = action
    else:
        raise ValueError(f"Expected action [T,D] or [1,T,D], got {tuple(action.shape)}")

    action_btd = action_btd.detach().cpu().to(torch.float32)
    if proprio is None:
        state_btd = torch.zeros(
            action_btd.shape[0],
            action_btd.shape[1],
            int(processor.proprio_output_dim),
            dtype=torch.float32,
        )
    else:
        state = proprio.detach().cpu().to(torch.float32)
        if state.ndim == 1:
            state_btd = state.view(1, 1, -1).expand(action_btd.shape[0], action_btd.shape[1], -1).clone()
        elif state.ndim == 2:
            state_btd = state.unsqueeze(0)
        elif state.ndim == 3 and state.shape[0] == 1:
            state_btd = state
        else:
            raise ValueError(f"Expected proprio [D], [T,D], or [1,T,D], got {tuple(state.shape)}")
        if state_btd.shape[1] != action_btd.shape[1]:
            state_btd = state_btd[:, :1, :].expand(action_btd.shape[0], action_btd.shape[1], -1).clone()

    batch = {"action": action_btd, "state": state_btd}
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    merged = {
        "action": {
            meta["key"]: batch["action"][meta["key"]].squeeze(0)
            for meta in processor.shape_meta["action"]
        },
        "state": {
            meta["key"]: batch["state"][meta["key"]].squeeze(0)
            for meta in processor.shape_meta["state"]
        },
    }
    merged = processor.action_state_merger.forward(merged)
    return merged["action"].detach().cpu().to(torch.float32)


def normalize_proprio(processor: Any, state: np.ndarray) -> torch.Tensor:
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    expected_dim = int(processor.proprio_output_dim)
    if state_tensor.numel() != expected_dim:
        raise ValueError(f"Expected proprio/state dim {expected_dim}, got {state_tensor.numel()}")
    batch = {"state": {"default": state_tensor.view(1, -1)}}
    batch = processor.normalizer.forward(batch)
    return batch["state"]["default"].squeeze(0)


def build_processor(cfg: DictConfig, dataset_stats_path: str | Path) -> Any:
    processor = instantiate(cfg.data.train.processor)
    stats = load_dataset_stats_from_json(dataset_stats_path)
    processor.set_normalizer_from_stats(stats)
    processor.eval()
    return processor


def run_file_source(args: argparse.Namespace, cfg: DictConfig, model: Any, output_dir: Path, prefix: str) -> None:
    missing = [
        name
        for name, value in {
            "--cam-high": args.cam_high,
            "--cam-left-wrist": args.cam_left_wrist,
            "--cam-right-wrist": args.cam_right_wrist,
            "--state-json": args.state_json,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError("File source requires: " + ", ".join(missing))

    processor = build_processor(cfg, args.dataset_stats)
    input_image = preprocess_real_images(
        cam_high=load_rgb_image(args.cam_high),
        cam_left_wrist=load_rgb_image(args.cam_left_wrist),
        cam_right_wrist=load_rgb_image(args.cam_right_wrist),
        device=model.device,
        dtype=model.torch_dtype,
    )
    proprio = normalize_proprio(processor, load_state_vector(args.state_json))
    num_video_frames = int(args.num_video_frames or default_num_video_frames(cfg))
    action_horizon = int(args.action_horizon or default_action_horizon(cfg))
    num_chunks = int(args.num_chunks)
    if num_chunks < 1:
        raise ValueError(f"--num-chunks must be >= 1, got {num_chunks}")

    formatted_prompt = DEFAULT_PROMPT.format(task=args.prompt)
    prompt: str | None = formatted_prompt
    context = None
    context_mask = None

    rollout_frames: list[Image.Image] = []
    rollout_actions: list[torch.Tensor] = []
    chunk_metrics: list[dict[str, Any]] = []
    current_input = input_image

    for chunk_idx in range(num_chunks):
        chunk_prefix = prefix if num_chunks == 1 else f"{prefix}_chunk{chunk_idx:02d}"
        chunk_seed = None if args.seed is None else int(args.seed) + chunk_idx
        if args.time_inference:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        pred = model.infer(
            prompt=prompt,
            input_image=current_input,
            num_frames=num_video_frames,
            action=None,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            text_cfg_scale=1.0,
            action_cfg_scale=1.0,
            num_inference_steps=args.num_inference_steps,
            seed=chunk_seed,
            rand_device=args.rand_device,
            tiled=args.tiled,
            expert_cache=args.expert_cache,
            expert_cache_reuse_steps=args.expert_cache_reuse_steps,
            expert_cache_warmup_steps=args.expert_cache_warmup_steps,
            expert_cache_cooldown_steps=args.expert_cache_cooldown_steps,
            time_inference=args.time_inference,
            profile_components=args.profile_components,
            cuda_graph_infer_action=args.cuda_graph_infer_action,
            cuda_graph_warmup_steps=args.cuda_graph_warmup_steps,
            torch_compile_infer_action=args.torch_compile_infer_action,
            torch_compile_fullgraph=args.torch_compile_fullgraph,
        )

        if args.time_inference:
            torch.cuda.synchronize()
            logger.info("Inference time (chunk %d): %.2f s", chunk_idx, time.perf_counter() - t0)

        if args.profile_components and "component_profile" in pred:
            logger.info("Component profile (chunk %d): %s", chunk_idx, pred["component_profile"])

        pred_frames = pred["video"]
        save_mp4(pred_frames, str(output_dir / f"{chunk_prefix}_pred.mp4"), fps=args.fps)
        save_image(current_input, output_dir / f"{chunk_prefix}_input.png")
        pred_action_denorm = save_actions(
            pred.get("action"),
            output_dir=output_dir,
            prefix=chunk_prefix,
            processor=processor,
            proprio=proprio,
        )
        if pred_action_denorm is not None:
            rollout_actions.append(pred_action_denorm)

        rollout_frames.extend(pred_frames if chunk_idx == 0 else pred_frames[1:])
        current_input = pil_frame_to_input_image(pred_frames[-1], device=model.device, dtype=model.torch_dtype)
        chunk_metrics.append(
            {
                "chunk_index": chunk_idx,
                "seed": chunk_seed,
                "video_path": f"{chunk_prefix}_pred.mp4",
                "input_path": f"{chunk_prefix}_input.png",
                "pred_actions_path": f"{chunk_prefix}_pred_actions.json",
                "cuda_graph_infer_action": bool(args.cuda_graph_infer_action),
                "torch_compile_infer_action": bool(args.torch_compile_infer_action),
                "component_profile": pred.get("component_profile") if args.profile_components else None,
            }
        )

    if num_chunks > 1:
        save_mp4(rollout_frames, str(output_dir / f"{prefix}_rollout_pred.mp4"), fps=args.fps)
        if rollout_actions:
            action_rollout = torch.cat(rollout_actions, dim=0)
            action_np = action_rollout.detach().cpu().numpy().astype(np.float32)
            np.save(output_dir / f"{prefix}_rollout_pred_actions.npy", action_np)
            with open(output_dir / f"{prefix}_rollout_pred_actions.json", "w", encoding="utf-8") as file:
                json.dump({"action": action_np.tolist()}, file, ensure_ascii=False, indent=2)

    metrics = {
        "source": "files",
        "num_video_frames": num_video_frames,
        "action_horizon": action_horizon,
        "num_chunks": num_chunks,
        "rollout_conditioning": "previous chunk last predicted frame",
        "num_inference_steps": int(args.num_inference_steps),
        "expert_cache": bool(args.expert_cache),
        "expert_cache_reuse_steps": int(args.expert_cache_reuse_steps),
        "expert_cache_warmup_steps": int(args.expert_cache_warmup_steps),
        "expert_cache_cooldown_steps": int(args.expert_cache_cooldown_steps),
        "profile_components": bool(args.profile_components),
        "cuda_graph_infer_action": bool(args.cuda_graph_infer_action),
        "cuda_graph_warmup_steps": int(args.cuda_graph_warmup_steps),
        "torch_compile_infer_action": bool(args.torch_compile_infer_action),
        "torch_compile_fullgraph": bool(args.torch_compile_fullgraph),
        "seed": int(args.seed),
        "prompt": formatted_prompt,
        "state_json": args.state_json,
        "used_zero_state": args.state_json is None,
        "chunks": chunk_metrics,
    }
    with open(output_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logger.info("Saved file dry-run outputs to %s.", output_dir)


def run() -> None:
    args = parse_args()
    setup_logging(log_level=logging.INFO)
    cfg = load_config(args)
    model = load_model(args, cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or "snapshot"

    run_file_source(args, cfg, model, output_dir, prefix)


if __name__ == "__main__":
    run()
