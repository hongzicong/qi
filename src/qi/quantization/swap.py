from __future__ import annotations

from collections.abc import Mapping
import math

import torch
import torch.nn as nn

from .backends import get_backend
from .modules import WeightOnlyLinear
from .quantize import quantize_linear_weight


def _normalize_markers(markers) -> tuple[str, ...]:
    return tuple(str(marker).lower() for marker in markers if str(marker).strip())


def _matches_target_expert(name: str, cfg) -> bool:
    scope = str(getattr(cfg, "target_expert", "all")).lower()
    if scope in ("all", "both"):
        return True

    lname = name.lower()
    action_markers = _normalize_markers(
        getattr(cfg, "action_expert_markers", ("action_expert", "action"))
    )
    video_markers = _normalize_markers(
        getattr(cfg, "video_expert_markers", ("video_expert", "video", "dit"))
    )

    if scope == "action":
        return any(marker in lname for marker in action_markers)
    if scope == "video":
        return any(marker in lname for marker in video_markers)
    raise ValueError(f"Unsupported target_expert: {scope}")


def _matches_module_kind(name: str, cfg) -> bool:
    lname = name.lower()
    ffn_markers = ("fc1", "fc2", "ffn", "mlp", "feed_forward")
    attn_markers = ("q_proj", "k_proj", "v_proj", "o_proj", "attn", "attention")
    if not getattr(cfg, "quantize_ffn", True) and any(marker in lname for marker in ffn_markers):
        return False
    if not getattr(cfg, "quantize_attn_proj", True) and any(marker in lname for marker in attn_markers):
        return False
    return True


def should_quantize(name: str, module: nn.Module, cfg) -> bool:
    if not getattr(cfg, "enabled", True):
        return False
    if not isinstance(module, nn.Linear):
        return False
    if not _matches_target_expert(name, cfg):
        return False
    if any(skip and skip in name for skip in getattr(cfg, "skip_modules", ())):
        return False
    target_modules = getattr(cfg, "target_modules", ())
    if target_modules and not any(target and target in name for target in target_modules):
        return False
    return _matches_module_kind(name, cfg)


def _stats_for_module(act_stats, name: str):
    if act_stats is None:
        return None
    if isinstance(act_stats, Mapping):
        if name in act_stats:
            return act_stats[name]
        suffix_matches = [value for key, value in act_stats.items() if name.endswith(str(key))]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
    return act_stats


def _effective_group_size(group_size: int, in_features: int) -> int:
    if group_size <= 0 or group_size > in_features:
        return in_features
    return int(group_size)


def _state_prefix(container_key: str | None, module_name: str) -> str:
    prefix = f"{module_name}." if module_name else ""
    if container_key:
        prefix = f"{container_key}.{prefix}"
    return prefix


def _state_has_key(state_dict: Mapping[str, object] | None, container_key: str | None, module_name: str, key: str) -> bool:
    if state_dict is None:
        return False
    return f"{_state_prefix(container_key, module_name)}{key}" in state_dict


def replace_linear_with_weight_only(
    model: nn.Module,
    cfg,
    act_stats: Mapping[str, object] | object | None = None,
) -> nn.Module:
    if hasattr(cfg, "validate"):
        cfg.validate()
    backend = get_backend(cfg.backend)

    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not should_quantize(full_name, child, cfg):
                continue

            quantized = quantize_linear_weight(
                child.weight.detach(),
                cfg=cfg,
                act_stats=_stats_for_module(act_stats, full_name),
            )
            qlinear = WeightOnlyLinear.from_linear(child, quantized, cfg=cfg, backend=backend)
            qlinear.train(child.training)
            setattr(parent, child_name, qlinear)

    return model


def prepare_model_for_weight_only_load(
    model: nn.Module,
    cfg,
    state_dict: Mapping[str, object] | None = None,
    container_key: str | None = None,
) -> nn.Module:
    if hasattr(cfg, "validate"):
        cfg.validate()
    backend = get_backend(cfg.backend)

    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not should_quantize(full_name, child, cfg):
                continue

            out_features, in_features = child.weight.shape
            group_size = _effective_group_size(cfg.group_size, in_features)
            num_groups = math.ceil(in_features / group_size)

            qweight = torch.empty((out_features, in_features), dtype=torch.int8)
            scales = torch.empty((out_features, num_groups), dtype=torch.float32)
            zeros = None
            if (not cfg.symmetric) or _state_has_key(state_dict, container_key, full_name, "zeros"):
                zeros = torch.empty((out_features, num_groups), dtype=torch.int16)

            input_scale = None
            if cfg.algo == "awq" or _state_has_key(state_dict, container_key, full_name, "input_scale"):
                input_scale = torch.empty((in_features,), dtype=torch.float32)

            qlinear = WeightOnlyLinear(
                qweight=qweight,
                scales=scales,
                zeros=zeros,
                input_scale=input_scale,
                bias=child.bias.detach() if child.bias is not None else None,
                in_features=child.in_features,
                out_features=child.out_features,
                bits=cfg.bits,
                group_size=cfg.group_size,
                symmetric=cfg.symmetric,
                backend=backend,
                keep_output_dtype=getattr(cfg, "keep_output_dtype", "input"),
            )
            qlinear.train(child.training)
            setattr(parent, child_name, qlinear)

    return model
