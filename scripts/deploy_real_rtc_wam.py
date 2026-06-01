#!/usr/bin/env python3
"""Robot-side RTC runtime for a persistent WAM policy server.

Supports dual-arm and single-arm (left/right) deployment from one entrypoint.
The arm mode is read from ``runtime.arm_mode`` in the config and can be
overridden on the CLI:

  # dual arms (14-dim policy)
  python deploy_real_rtc_wam.py --arm-mode dual

  # single right arm (7-dim policy) -- left arm parks, still feeds its camera
  python deploy_real_rtc_wam.py --arm-mode single --arm-side right

  # single left arm (7-dim policy) -- right arm parks
  python deploy_real_rtc_wam.py --arm-mode single --arm-side left

The chosen mode MUST match the action dimension of the checkpoint the policy
server loaded (dual -> 14-dim, single -> 7-dim).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from qi.deployment.robot_runtime import ARM_MODES, RTCWAMRobotRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RTC-style Piper deployment with a WAM policy server.")
    parser.add_argument("--config", default="configs/real_bot.yaml")
    parser.add_argument("--host", default=None, help="Override policy_server.host.")
    parser.add_argument("--port", type=int, default=None, help="Override policy_server.port.")
    parser.add_argument("--execute-actions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--move-to-initial", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--arm-mode",
        choices=["dual", "single"],
        default=None,
        help="Override runtime.arm_mode: 'dual' drives both arms (14-dim), "
        "'single' drives one arm (7-dim, pick the arm with --arm-side).",
    )
    parser.add_argument(
        "--arm-side",
        choices=["left", "right"],
        default=None,
        help="Which arm to drive when --arm-mode single. Passing this alone "
        "implies single-arm mode.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config in {config_path}")
    return payload


def resolve_arm_mode(config: dict, args: argparse.Namespace) -> str:
    """Combine the two CLI switches with the config default into one of
    ``{"dual", "left", "right"}`` (the internal arm_mode the runtime expects)."""
    current = str(config["runtime"].get("arm_mode", "dual")).strip().lower()

    if args.arm_mode is None and args.arm_side is None:
        final = current  # no override -> use config as-is
    elif args.arm_mode == "dual":
        final = "dual"
    else:
        # Single arm, either via --arm-mode single or by passing --arm-side alone.
        side = args.arm_side
        if side is None:
            if current in ("left", "right"):
                side = current  # config already names a side; keep it
            else:
                raise SystemExit("--arm-mode single requires --arm-side {left,right}")
        final = side

    if final not in ARM_MODES:
        raise SystemExit(f"Resolved arm_mode {final!r} is invalid; expected one of {ARM_MODES}.")
    return final


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    config.setdefault("policy_server", {})
    config.setdefault("runtime", {})
    if args.host is not None:
        config["policy_server"]["host"] = args.host
    if args.port is not None:
        config["policy_server"]["port"] = args.port
    if args.execute_actions is not None:
        config["runtime"]["execute_actions"] = args.execute_actions
    if args.move_to_initial is not None:
        config["runtime"]["move_to_initial"] = args.move_to_initial
    if args.max_steps is not None:
        config["runtime"]["max_steps"] = args.max_steps
    if args.output_dir is not None:
        config["runtime"]["output_dir"] = args.output_dir
    config["runtime"]["arm_mode"] = resolve_arm_mode(config, args)
    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    runtime = RTCWAMRobotRuntime(config)
    runtime.run()


if __name__ == "__main__":
    main()