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
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from qi.datasets.dataset_utils import CenterCrop, ResizeSmallestSideAspectPreserving
from qi.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from qi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from qi.utils.config_resolvers import register_default_resolvers
from qi.utils.logging_config import get_logger, setup_logging
from qi.utils.video_io import save_mp4
from qi.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim


register_default_resolvers()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WAM open-loop video inference without touching robot IO."
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoints/weights/step_xxxxxx.pt.")
    parser.add_argument("--dataset-stats", required=True, help="Path to dataset_stats.json from the same run/data.")
    parser.add_argument("--train-config", default=None, help="Optional saved runs/.../config.yaml.")
    parser.add_argument("--task", default="real_cleaning_uncond_3cam_384_1e-4")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--source", choices=["dataset", "files"], default="dataset")
    parser.add_argument("--output-dir", default="open_loop_video_dry_run")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--num-video-frames", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--time-inference", action="store_true", help="Print wall-clock inference time (adds CUDA synchronization).")
    parser.add_argument(
        "--skip-training-loss",
        action="store_true",
        help="Dataset mode only: skip the training_loss forward pass.",
    )
    parser.add_argument(
        "--condition-on-gt-action",
        action="store_true",
        help="Dataset mode only: condition generated video on GT action. Leave off for true open-loop.",
    )
    parser.add_argument(
        "--save-vae-recon",
        action="store_true",
        help="Dataset mode only: also save VAE reconstruction and include it in the comparison video.",
    )

    # Dataset source.
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=None,
        help="Override dataset dirs for the selected split. Can be passed multiple times.",
    )
    parser.add_argument("--context-cache-dir", default=None, help="Override text embedding cache directory.")
    parser.add_argument(
        "--override-instruction",
        default=None,
        help="Dataset mode: replace sample instruction before looking up cached text embedding.",
    )

    # File source.
    parser.add_argument("--cam-high", default=None)
    parser.add_argument("--cam-left-wrist", default=None)
    parser.add_argument("--cam-right-wrist", default=None)
    parser.add_argument("--state-json", default=None)
    parser.add_argument("--prompt", default="clean the table")
    parser.add_argument(
        "--use-text-encoder",
        action="store_true",
        help="Use model text encoder instead of cached text embedding. Slower and uses more VRAM.",
    )
    return parser.parse_args()


def dtype_from_mixed_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def load_config(args: argparse.Namespace) -> DictConfig:
    if args.train_config:
        cfg = OmegaConf.load(args.train_config)
        if not isinstance(cfg, DictConfig):
            raise ValueError(f"Expected DictConfig in {args.train_config}, got {type(cfg)}")
    else:
        config_dir = str(Path(args.config_dir).resolve())
        overrides = [f"task={args.task}", f"mixed_precision={args.mixed_precision}"]
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="train", overrides=overrides)

    if args.use_text_encoder:
        cfg.model.load_text_encoder = True

    # Make validation/test dataset construction use the exact stats requested
    # by this dry-run instead of recomputing or using a stale saved config path.
    if "data" in cfg:
        for split in ("train", "val"):
            if split in cfg.data:
                cfg.data[split].pretrained_norm_stats = args.dataset_stats
                if args.context_cache_dir is not None:
                    cfg.data[split].text_embedding_cache_dir = args.context_cache_dir
                if args.dataset_dir is not None:
                    cfg.data[split].dataset_dirs = list(args.dataset_dir)
                if args.override_instruction is not None:
                    cfg.data[split].override_instruction = args.override_instruction
    return cfg


def default_num_video_frames(cfg: DictConfig) -> int:
    num_frames = int(cfg.data.train.num_frames)
    ratio = int(cfg.data.train.get("action_video_freq_ratio", 1))
    return ((num_frames - 1) // ratio) + 1


def default_action_horizon(cfg: DictConfig) -> int:
    return int(cfg.data.train.num_frames) - 1


def load_model(args: argparse.Namespace, cfg: DictConfig) -> Any:
    model_dtype = dtype_from_mixed_precision(args.mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    payload = model.load_checkpoint(args.ckpt)
    model.eval()
    logger.info("Loaded checkpoint %s with keys %s.", args.ckpt, sorted(payload.keys()))
    return model


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_rgb_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_state_vector(path: str | Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        if "state" not in payload:
            raise ValueError(f"{path} must contain a `state` key or be a raw list.")
        payload = payload["state"]
    state = np.asarray(payload, dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(f"State must be a 1D vector, got shape {state.shape}")
    return state


def rgb_to_float_chw(rgb_uint8: np.ndarray) -> torch.Tensor:
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected RGB image as [H,W,3], got {rgb_uint8.shape}")
    rgb_uint8 = np.ascontiguousarray(rgb_uint8).copy()
    return torch.from_numpy(rgb_uint8).permute(2, 0, 1).to(torch.float32) / 255.0


def preprocess_real_images(
    cam_high: np.ndarray,
    cam_left_wrist: np.ndarray,
    cam_right_wrist: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cam_high_t = transforms_F.resize(
        rgb_to_float_chw(cam_high),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left_t = transforms_F.resize(
        rgb_to_float_chw(cam_left_wrist),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right_t = transforms_F.resize(
        rgb_to_float_chw(cam_right_wrist),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_high_t = transforms_F.resize(
        cam_high_t,
        [256, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left_t = transforms_F.resize(
        cam_left_t,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right_t = transforms_F.resize(
        cam_right_t,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    bottom = torch.cat([cam_left_t, cam_right_t], dim=-1)
    image = torch.cat([cam_high_t, bottom], dim=-2)
    resize = ResizeSmallestSideAspectPreserving(args={"img_w": 320, "img_h": 384})
    crop = CenterCrop(args={"img_w": 320, "img_h": 384})
    image = crop(resize(image))
    image = image.mul(2.0).sub(1.0)
    return image.unsqueeze(0).to(device=device, dtype=dtype)


def tensor_video_to_pil_frames(video: torch.Tensor, value_range: str) -> list[Image.Image]:
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"Expected video [3,T,H,W], got {tuple(video.shape)}")
    video = video.detach().float().cpu()
    if value_range == "minus_one_one":
        video = (video.clamp(-1.0, 1.0) + 1.0) * 0.5
    elif value_range == "zero_one":
        video = video.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported value_range: {value_range}")
    frames = []
    for t in range(video.shape[1]):
        arr = (video[:, t].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


def save_input_image(input_image: torch.Tensor, path: Path) -> None:
    frame = input_image.detach().float().cpu().squeeze(0)
    tensor_video_to_pil_frames(frame.unsqueeze(1), "minus_one_one")[0].save(path)


def add_label(frame: Image.Image, label: str) -> Image.Image:
    image = frame.convert("RGB")
    header_h = 22
    out = Image.new("RGB", (image.width, image.height + header_h), color=(18, 18, 18))
    out.paste(image, (0, header_h))
    draw = ImageDraw.Draw(out)
    draw.text((6, 4), label, fill=(255, 255, 255))
    return out


def make_comparison_frames(columns: list[tuple[str, list[Image.Image]]]) -> list[Image.Image]:
    if not columns:
        raise ValueError("No columns to stitch.")
    num_frames = min(len(frames) for _, frames in columns)
    stitched = []
    for t in range(num_frames):
        labeled = [add_label(frames[t], label) for label, frames in columns]
        width = sum(frame.width for frame in labeled)
        height = max(frame.height for frame in labeled)
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        x = 0
        for frame in labeled:
            canvas.paste(frame, (x, 0))
            x += frame.width
        stitched.append(canvas)
    return stitched


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


def save_action_comparison(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    output_dir: Path,
    prefix: str,
) -> dict[str, float]:
    if pred_action.shape != gt_action.shape:
        raise ValueError(
            "Predicted action/GT action shape mismatch after denormalization: "
            f"pred={tuple(pred_action.shape)} gt={tuple(gt_action.shape)}"
        )
    pred_np = pred_action.detach().cpu().numpy().astype(np.float32)
    gt_np = gt_action.detach().cpu().numpy().astype(np.float32)
    diff_np = pred_np - gt_np
    np.save(output_dir / f"{prefix}_gt_actions.npy", gt_np)
    np.save(output_dir / f"{prefix}_action_diff.npy", diff_np)
    np.savez(
        output_dir / f"{prefix}_action_compare.npz",
        pred=pred_np,
        gt=gt_np,
        diff=diff_np,
    )
    with open(output_dir / f"{prefix}_action_compare.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "pred": pred_np.tolist(),
                "gt": gt_np.tolist(),
                "diff": diff_np.tolist(),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    diff = pred_action - gt_action
    return {
        "action_l1": float(diff.abs().mean().item()),
        "action_l2": float(diff.pow(2).mean().item()),
        "action_max_abs": float(diff.abs().max().item()),
    }


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


def load_cached_context(
    cache_dir: str | Path,
    formatted_prompt: str,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = Path(cache_dir)
    cache_hash = hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{cache_hash}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing text embedding cache: {cache_path}. "
            "Run scripts/precompute_text_embeds.py for this prompt, or pass --use-text-encoder."
        )
    payload = torch.load(cache_path, map_location="cpu")
    context = payload["context"]
    context_mask = payload["mask"].bool()
    if context.ndim != 2 or context_mask.ndim != 1:
        raise ValueError(f"Invalid cached context shapes in {cache_path}: {context.shape}, {context_mask.shape}")
    if context.shape[0] != context_len or context_mask.shape[0] != context_len:
        raise ValueError(
            f"Cached context length mismatch in {cache_path}: "
            f"context={context.shape[0]}, mask={context_mask.shape[0]}, expected={context_len}"
        )
    context = context.clone()
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask, dtype=torch.bool)
    return context, context_mask


def build_processor(cfg: DictConfig, dataset_stats_path: str | Path) -> Any:
    processor = instantiate(cfg.data.train.processor)
    stats = load_dataset_stats_from_json(dataset_stats_path)
    processor.set_normalizer_from_stats(stats)
    processor.eval()
    return processor


def batch_dataset_sample(sample: dict[str, Any]) -> dict[str, Any]:
    batched: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batched[key] = value.unsqueeze(0)
        elif key == "prompt":
            batched[key] = [value]
        else:
            batched[key] = value
    return batched


def run_dataset_source(args: argparse.Namespace, cfg: DictConfig, model: Any, output_dir: Path, prefix: str) -> None:
    split_cfg = cfg.data[args.split]
    dataset = instantiate(split_cfg)
    if len(dataset) == 0:
        raise ValueError(f"{args.split} dataset is empty.")
    sample_index = int(args.sample_index) % len(dataset)
    sample = dataset[sample_index]
    processor = dataset.lerobot_dataset.processor

    video = sample["video"]
    if video.ndim != 4:
        raise ValueError(f"Expected sample video [3,T,H,W], got {tuple(video.shape)}")
    metrics: dict[str, Any] = {
        "source": "dataset",
        "split": args.split,
        "sample_index": sample_index,
        "condition_on_gt_action": bool(args.condition_on_gt_action),
        "num_inference_steps": int(args.num_inference_steps),
        "seed": int(args.seed),
        "prompt": sample["prompt"],
    }

    if not args.skip_training_loss:
        batched_sample = batch_dataset_sample(sample)
        set_torch_seed(args.seed)
        with torch.no_grad():
            loss_total, loss_dict = model.training_loss(batched_sample, tiled=args.tiled)
        metrics["training_loss_total"] = float(loss_total.detach().float().item())
        metrics.update({f"training_{key}": float(value) for key, value in loss_dict.items()})

    input_image = video[:, 0].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    num_video_frames = int(args.num_video_frames or video.shape[1])
    action_horizon = int(args.action_horizon or sample["action"].shape[0])
    action_cond = sample["action"] if args.condition_on_gt_action else None
    proprio = sample["proprio"][0]

    if args.time_inference:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    
    pred = model.infer(
        prompt=None,
        input_image=input_image,
        num_frames=num_video_frames,
        action=action_cond,
        action_horizon=action_horizon,
        proprio=proprio,
        context=sample["context"],
        context_mask=sample["context_mask"],
        text_cfg_scale=1.0,
        action_cfg_scale=1.0,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        rand_device=args.rand_device,
        tiled=args.tiled,
    )
    
    if args.time_inference:
        torch.cuda.synchronize()
        logger.info("Inference time: %.2f s", time.perf_counter() - t0)

    pred_frames = pred["video"]
    gt_frames = tensor_video_to_pil_frames(video, "minus_one_one")
    save_mp4(pred_frames, str(output_dir / f"{prefix}_pred.mp4"), fps=args.fps)
    save_mp4(gt_frames, str(output_dir / f"{prefix}_gt.mp4"), fps=args.fps)
    save_input_image(input_image, output_dir / f"{prefix}_input.png")

    columns = [("pred", pred_frames)]
    metrics["num_video_frames"] = num_video_frames
    metrics["action_horizon"] = action_horizon

    pred_tensor = pil_frames_to_video_tensor(pred_frames)
    gt_tensor = ((video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
    if pred_tensor.shape == gt_tensor.shape:
        metrics["pred_vs_gt_psnr"] = video_psnr(pred_tensor, gt_tensor)
        metrics["pred_vs_gt_ssim"] = video_ssim(pred_tensor, gt_tensor)

    if args.save_vae_recon:
        gt_batch = video.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        with torch.no_grad():
            vae_latents = model._encode_video_latents(gt_batch, tiled=args.tiled)
            vae_frames = model._decode_latents(vae_latents, tiled=args.tiled)
        save_mp4(vae_frames, str(output_dir / f"{prefix}_vae_recon.mp4"), fps=args.fps)
        vae_tensor = pil_frames_to_video_tensor(vae_frames)
        if vae_tensor.shape == gt_tensor.shape:
            metrics["vae_vs_gt_psnr"] = video_psnr(vae_tensor, gt_tensor)
            metrics["vae_vs_gt_ssim"] = video_ssim(vae_tensor, gt_tensor)
        columns.append(("vae", vae_frames))

    columns.append(("gt", gt_frames))
    save_mp4(
        make_comparison_frames(columns),
        str(output_dir / f"{prefix}_compare.mp4"),
        fps=args.fps,
    )

    pred_action_denorm = save_actions(
        pred.get("action"),
        output_dir=output_dir,
        prefix=prefix,
        processor=processor,
        proprio=sample["proprio"],
    )
    if pred_action_denorm is not None and "action" in sample:
        gt_action_denorm = denormalize_merged_action(processor, sample["action"], sample["proprio"])
        metrics.update(
            save_action_comparison(
                pred_action=pred_action_denorm,
                gt_action=gt_action_denorm,
                output_dir=output_dir,
                prefix=prefix,
            )
        )

    with open(output_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    logger.info("Saved dataset dry-run outputs to %s.", output_dir)


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

    formatted_prompt = DEFAULT_PROMPT.format(task=args.prompt)
    prompt: str | None = formatted_prompt
    context = None
    context_mask = None
    if not args.use_text_encoder:
        cache_dir = args.context_cache_dir or cfg.data.train.get("text_embedding_cache_dir")
        if cache_dir is None:
            raise ValueError("No context cache dir found. Pass --context-cache-dir or --use-text-encoder.")
        context_len = int(cfg.data.train.get("context_len", 128))
        context, context_mask = load_cached_context(cache_dir, formatted_prompt, context_len)
        prompt = None

    if args.time_inference:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    pred = model.infer(
        prompt=prompt,
        input_image=input_image,
        num_frames=num_video_frames,
        action=None,
        action_horizon=action_horizon,
        proprio=proprio,
        context=context,
        context_mask=context_mask,
        text_cfg_scale=1.0,
        action_cfg_scale=1.0,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        rand_device=args.rand_device,
        tiled=args.tiled,
    )

    if args.time_inference:
        torch.cuda.synchronize()
        logger.info("Inference time: %.2f s", time.perf_counter() - t0)

    pred_frames = pred["video"]
    save_mp4(pred_frames, str(output_dir / f"{prefix}_pred.mp4"), fps=args.fps)
    save_input_image(input_image, output_dir / f"{prefix}_input.png")
    save_actions(
        pred.get("action"),
        output_dir=output_dir,
        prefix=prefix,
        processor=processor,
        proprio=proprio,
    )

    metrics = {
        "source": "files",
        "num_video_frames": num_video_frames,
        "action_horizon": action_horizon,
        "num_inference_steps": int(args.num_inference_steps),
        "seed": int(args.seed),
        "prompt": formatted_prompt,
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
    prefix = args.prefix or f"{args.source}_{args.split if args.source == 'dataset' else 'snapshot'}_{args.sample_index}"

    if args.source == "dataset":
        run_dataset_source(args, cfg, model, output_dir, prefix)
    else:
        run_file_source(args, cfg, model, output_dir, prefix)


if __name__ == "__main__":
    run()
