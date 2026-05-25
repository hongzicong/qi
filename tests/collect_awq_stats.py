#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qi.api import load_config, load_model
from qi.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from qi.quantization import WeightOnlyQuantConfig, collect_awq_activation_stats
from qi.utils.config_resolvers import register_default_resolvers
from qi.utils.image_utils import load_rgb_image, load_state_vector, preprocess_real_images
from qi.utils.logging_config import get_logger, setup_logging

from test_dry_run import build_processor, default_action_horizon, default_num_video_frames, normalize_proprio

register_default_resolvers()
logger = get_logger(__name__)


def _csv_tuple(value: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect AWQ activation stats from a dry-run style sample.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-stats", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--task", default="real_cleaning_uncond_3cam_384_1e-4")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--cam-high", required=True)
    parser.add_argument("--cam-left-wrist", required=True)
    parser.add_argument("--cam-right-wrist", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--prompt", default="clean the table")
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--tiled", action="store_true")

    parser.add_argument("--quant-bits", type=int, default=4)
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--quant-asymmetric", action="store_true")
    parser.add_argument("--target-expert", choices=["all", "both", "action", "video"], default="all")
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,fc1,fc2")
    parser.add_argument("--skip-modules", default="")
    parser.add_argument("--no-quantize-ffn", action="store_true")
    parser.add_argument("--no-quantize-attn-proj", action="store_true")
    parser.add_argument("--awq-alpha", type=float, default=0.5)
    parser.add_argument("--awq-scale-eps", type=float, default=1e-4)
    return parser.parse_args()


def build_quant_config(args: argparse.Namespace) -> WeightOnlyQuantConfig:
    return WeightOnlyQuantConfig(
        enabled=True,
        algo="awq",
        bits=args.quant_bits,
        group_size=args.quant_group_size,
        symmetric=not args.quant_asymmetric,
        backend="reference",
        target_expert=args.target_expert,
        target_modules=_csv_tuple(args.target_modules),
        skip_modules=_csv_tuple(args.skip_modules),
        quantize_ffn=not args.no_quantize_ffn,
        quantize_attn_proj=not args.no_quantize_attn_proj,
        awq_alpha=args.awq_alpha,
        awq_scale_eps=args.awq_scale_eps,
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args)
    model = load_model(args, cfg)
    processor = build_processor(cfg, args.dataset_stats)

    input_image = preprocess_real_images(
        cam_high=load_rgb_image(args.cam_high),
        cam_left_wrist=load_rgb_image(args.cam_left_wrist),
        cam_right_wrist=load_rgb_image(args.cam_right_wrist),
        device=model.device,
        dtype=model.torch_dtype,
    )
    proprio = normalize_proprio(processor, load_state_vector(args.state_json))
    batch = {
        "prompt": DEFAULT_PROMPT.format(task=args.prompt),
        "input_image": input_image,
        "num_frames": default_num_video_frames(cfg),
        "action_horizon": default_action_horizon(cfg),
        "proprio": proprio,
    }

    def forward_fn(model, sample):
        return model.infer(
            prompt=sample["prompt"],
            input_image=sample["input_image"],
            num_frames=sample["num_frames"],
            action=None,
            action_horizon=sample["action_horizon"],
            proprio=sample["proprio"],
            context=None,
            context_mask=None,
            text_cfg_scale=1.0,
            action_cfg_scale=1.0,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            rand_device=args.rand_device,
            tiled=args.tiled,
            expert_cache=False,
            cuda_graph=False,
            time_inference=False,
        )

    quant_cfg = build_quant_config(args)
    stats = collect_awq_activation_stats(model, [batch], quant_cfg, forward_fn=forward_fn)
    output_stats = Path(args.output_stats)
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output_stats)
    logger.info("Saved AWQ activation stats for %d modules to %s.", len(stats), output_stats)


if __name__ == "__main__":
    main()
