
from typing import Any
from pathlib import Path

import argparse
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from qi.utils.logging_config import get_logger

logger = get_logger(__name__)


def dtype_from_mixed_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def load_config(args: argparse.Namespace) -> DictConfig:
    config_dir = str(Path(args.config_dir).resolve())
    overrides = [f"task={args.task}", f"mixed_precision={args.mixed_precision}"]
    if getattr(args, "module_cuda_graph_off", False):
        overrides.append("model.module_cuda_graph=false")
    if getattr(args, "module_torch_compile", False):
        overrides.append("model.module_torch_compile=true")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)

    cfg.model.load_text_encoder = True

    # Make validation/test dataset construction use the exact stats requested
    # by this dry-run instead of recomputing or using a stale saved config path.
    if "data" in cfg:
        for split in ("train", "val"):
            if split in cfg.data:
                cfg.data[split].pretrained_norm_stats = args.dataset_stats
    return cfg


def load_model(args: argparse.Namespace, cfg: DictConfig) -> Any:
    model_dtype = dtype_from_mixed_precision(args.mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    payload = model.load_checkpoint(args.ckpt)
    model.eval()
    logger.info("Loaded checkpoint %s with keys %s.", args.ckpt, sorted(payload.keys()))
    return model