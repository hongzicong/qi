from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .config import WeightActivationQuantConfig, WeightOnlyQuantConfig
from .swap import prepare_model_for_weight_only_load, replace_linear_with_weight_only

METADATA_KEY = "qi_quantization"


def quant_config_from_metadata(metadata: Mapping[str, Any]) -> WeightOnlyQuantConfig | WeightActivationQuantConfig:
    cfg_payload = dict(metadata.get("config", metadata))
    cfg_cls = (
        WeightActivationQuantConfig
        if metadata.get("format") == "qi.weight_activation.v1" or cfg_payload.get("quant_dtype") == "fp8"
        else WeightOnlyQuantConfig
    )
    valid_keys = cfg_cls.__dataclass_fields__.keys()
    cfg_payload = {key: value for key, value in cfg_payload.items() if key in valid_keys}
    for key in (
        "action_expert_markers",
        "video_expert_markers",
        "target_modules",
        "skip_modules",
    ):
        if key in cfg_payload and isinstance(cfg_payload[key], list):
            cfg_payload[key] = tuple(cfg_payload[key])
    return cfg_cls(**cfg_payload)


def checkpoint_is_quantized(payload: Mapping[str, Any]) -> bool:
    return METADATA_KEY in payload


def prepare_model_for_quantized_checkpoint(model: nn.Module, payload: Mapping[str, Any]) -> WeightOnlyQuantConfig | WeightActivationQuantConfig:
    if not checkpoint_is_quantized(payload):
        raise ValueError("Checkpoint is missing quantization metadata.")
    cfg = quant_config_from_metadata(payload[METADATA_KEY])
    if "mot" in payload:
        prepare_model_for_weight_only_load(model.mot, cfg, state_dict=payload["mot"])
    elif "dit" in payload:
        prepare_model_for_weight_only_load(model.dit, cfg, state_dict=payload["dit"])
    elif "model" in payload:
        prepare_model_for_weight_only_load(model, cfg, state_dict=payload["model"])
    else:
        prepare_model_for_weight_only_load(model, cfg)
    return cfg


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dict, got {type(payload)} from {path}.")
    return payload


def quantize_loaded_model(
    model: nn.Module,
    cfg: WeightOnlyQuantConfig | WeightActivationQuantConfig,
    act_stats: Mapping[str, object] | object | None = None,
) -> nn.Module:
    replace_linear_with_weight_only(model, cfg, act_stats=act_stats)
    return model


def save_quantized_checkpoint(
    model,
    output_path: str | Path,
    cfg: WeightOnlyQuantConfig | WeightActivationQuantConfig,
    source_payload: Mapping[str, Any] | None = None,
    source_checkpoint: str | Path | None = None,
) -> None:
    payload: dict[str, Any] = dict(source_payload or {})
    if hasattr(model, "mot"):
        payload["mot"] = model.mot.state_dict()
    elif hasattr(model, "dit"):
        payload["dit"] = model.dit.state_dict()
    else:
        payload["model"] = model.state_dict()

    if getattr(model, "proprio_encoder", None) is not None:
        payload["proprio_encoder"] = model.proprio_encoder.state_dict()

    payload.pop("optimizer", None)
    payload[METADATA_KEY] = {
        "format": "qi.weight_activation.v1" if isinstance(cfg, WeightActivationQuantConfig) else "qi.weight_only.v1",
        "config": asdict(cfg),
        "source_checkpoint": None if source_checkpoint is None else str(source_checkpoint),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
