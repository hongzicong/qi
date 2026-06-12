#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate

from qi.api import dtype_from_mixed_precision, load_config
from qi.quantization import WeightOnlyQuantConfig, collect_awq_calibration_cache
from qi.utils.config_resolvers import register_default_resolvers
from qi.utils.image_utils import load_rgb_image, load_state_vector, preprocess_real_images
from qi.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

from test_dry_run import (
    build_processor,
    default_action_horizon,
    default_num_video_frames,
    normalize_proprio,
)

register_default_resolvers()


def _csv_tuple(value: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect AWQ calibration cache with real dry-run inputs. "
            "The output contains per-module input activation samples for AWQ scale and clipping search."
        )
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--task", default="real_cleaning_uncond_3cam_384_1e-4")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--model-base-path", default=None)
    parser.add_argument("--skip-download", action="store_true")

    parser.add_argument("--cam-high", default="./tests/data/cam_high.png")
    parser.add_argument("--cam-left-wrist", default="./tests/data/cam_left_wrist.png")
    parser.add_argument("--cam-right-wrist", default="./tests/data/cam_right_wrist.png")
    parser.add_argument("--state-json", default="./tests/data/state.json")
    parser.add_argument(
        "--prompt",
        default="Pick and place the numbered blocks 9, 1, 5, 11, and 4 to the lower area in order.",
    )

    parser.add_argument("--target-expert", choices=["all", "both", "action", "video"], default="all")
    parser.add_argument(
        "--target-modules",
        default="self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2",
    )
    parser.add_argument("--skip-modules", default="")
    parser.add_argument("--no-quantize-ffn", action="store_true")
    parser.add_argument("--no-quantize-attn-proj", action="store_true")

    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--max-samples-per-module", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--expert-cache", action="store_true")
    parser.add_argument("--expert-cache-reuse-steps", type=int, default=1)
    parser.add_argument("--expert-cache-warmup-steps", type=int, default=1)
    parser.add_argument("--expert-cache-cooldown-steps", type=int, default=1)
    parser.add_argument("--full-infer", action="store_true", help="Calibrate through model.infer instead of model.infer_action.")
    return parser.parse_args()


def configure_local_model_loading(args: argparse.Namespace) -> None:
    if args.model_base_path:
        model_base_path = Path(args.model_base_path).expanduser().resolve()
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_base_path)
    if args.skip_download:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"


def configure_checkpoint_model_loading(cfg) -> None:
    cfg.model.skip_dit_load_from_pretrain = True
    cfg.model.action_dit_pretrained_path = None


def build_calibration_inputs(args: argparse.Namespace, cfg, model: Any) -> dict[str, Any]:
    processor = build_processor(cfg, args.dataset_stats)
    input_image = preprocess_real_images(
        cam_high=load_rgb_image(args.cam_high),
        cam_left_wrist=load_rgb_image(args.cam_left_wrist),
        cam_right_wrist=load_rgb_image(args.cam_right_wrist),
        device=model.device,
        dtype=model.torch_dtype,
    )
    proprio = normalize_proprio(processor, load_state_vector(args.state_json))
    prompt = DEFAULT_PROMPT.format(task=args.prompt)
    return {
        "prompt": prompt,
        "input_image": input_image,
        "proprio": proprio,
        "action_horizon": default_action_horizon(cfg),
        "num_video_frames": default_num_video_frames(cfg),
    }


def make_forward_fn(args: argparse.Namespace, cfg):
    def forward_fn(model, batch: dict[str, Any]):
        if args.full_infer:
            return model.infer(
                prompt=batch["prompt"],
                input_image=batch["input_image"],
                num_frames=batch["num_video_frames"],
                action=None,
                action_horizon=batch["action_horizon"],
                proprio=batch["proprio"],
                context=None,
                context_mask=None,
                text_cfg_scale=1.0,
                action_cfg_scale=1.0,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                rand_device=args.rand_device,
                tiled=args.tiled,
                expert_cache=args.expert_cache,
                cuda_graph=False,
                expert_cache_reuse_steps=args.expert_cache_reuse_steps,
                expert_cache_warmup_steps=args.expert_cache_warmup_steps,
                expert_cache_cooldown_steps=args.expert_cache_cooldown_steps,
                time_inference=False,
            )
        return model.infer_action(
            prompt=batch["prompt"],
            input_image=batch["input_image"],
            action_horizon=batch["action_horizon"],
            proprio=batch["proprio"],
            context=None,
            context_mask=None,
            text_cfg_scale=1.0,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            rand_device=args.rand_device,
            tiled=args.tiled,
            expert_cache=args.expert_cache,
            cuda_graph=False,
            expert_cache_reuse_steps=args.expert_cache_reuse_steps,
            expert_cache_warmup_steps=args.expert_cache_warmup_steps,
            expert_cache_cooldown_steps=args.expert_cache_cooldown_steps,
            return_on_device=True,
        )

    return forward_fn


def main() -> None:
    args = parse_args()
    configure_local_model_loading(args)

    cfg = load_config(args)
    configure_checkpoint_model_loading(cfg)
    model_dtype = dtype_from_mixed_precision(args.mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    model.load_checkpoint(args.ckpt)

    quant_cfg = WeightOnlyQuantConfig(
        enabled=True,
        algo="awq",
        quant_dtype="int",
        bits=4,
        group_size=128,
        symmetric=True,
        backend="int4_w4a16",
        target_expert=args.target_expert,
        target_modules=_csv_tuple(args.target_modules),
        skip_modules=_csv_tuple(args.skip_modules),
        quantize_ffn=not args.no_quantize_ffn,
        quantize_attn_proj=not args.no_quantize_attn_proj,
    )
    quant_cfg.validate()

    batch = build_calibration_inputs(args, cfg, model)
    calibration_data = [batch for _ in range(max(int(args.max_batches), 1))]
    cache = collect_awq_calibration_cache(
        model,
        calibration_data=calibration_data,
        cfg=quant_cfg,
        forward_fn=make_forward_fn(args, cfg),
        max_batches=args.max_batches,
        max_samples_per_module=args.max_samples_per_module,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    print(f"saved {len(cache)} module caches to {output}")
    if cache:
        first_key = next(iter(cache))
        first = cache[first_key]
        print(f"example: {first_key} inputs={tuple(first['inputs'].shape)} abs_mean={tuple(first['abs_mean'].shape)}")


if __name__ == "__main__":
    main()
