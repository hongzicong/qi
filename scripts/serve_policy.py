#!/usr/bin/env python3
"""Start a persistent WAM policy server for RTC-style robot deployment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from qi.deployment.wam_policy_client import WAMPolicyConfig
from qi.deployment.wam_policy_server import serve_wam_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve WAM as a persistent RTC-compatible policy backend.")
    parser.add_argument("--config", default="configs/real_bot.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--authkey", default="wam")
    parser.add_argument("--expert-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {config_path}")
    policy_config = WAMPolicyConfig.from_mapping(payload, base_dir=config_path.parent)
    overrides = {
        key: value
        for key, value in {
            "expert_cache": args.expert_cache,
            "cuda_graph": args.cuda_graph,
            "torch_compile": args.torch_compile,
        }.items()
        if value is not None
    }
    if overrides:
        policy_config = replace(policy_config, **overrides)
    serve_wam_policy(policy_config, host=args.host, port=args.port, authkey=args.authkey)


if __name__ == "__main__":
    main()