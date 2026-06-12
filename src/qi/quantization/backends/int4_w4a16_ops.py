from __future__ import annotations

import importlib
from types import ModuleType

import torch


_REQUIRED_FUNCTIONS = ("mul",)


class Int4W4A16OpsUnavailable(RuntimeError):
    pass


def _hardware_note() -> str:
    if not torch.cuda.is_available():
        return " CUDA is not available."
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    return f" Current GPU is {name} sm_{major}{minor}."


def load_int4_w4a16_ops(required: bool = True) -> ModuleType | None:
    try:
        ops = importlib.import_module("qi.quantization._int4_w4a16_ops")
        missing = [name for name in _REQUIRED_FUNCTIONS if getattr(ops, name, None) is None]
        if missing:
            raise Int4W4A16OpsUnavailable(
                "Qi INT4 W4A16 extension is missing functions: " + ", ".join(missing) + "." + _hardware_note()
            )
        return ops
    except (ImportError, Int4W4A16OpsUnavailable):
        if required:
            raise Int4W4A16OpsUnavailable(
                "Qi INT4 W4A16 kernels are not available. Build/install with QI_BUILD_INT4_W4A16=1 and "
                "INT4 W4A16 sources available at src/qi/quantization/csrc/int4_w4a16."
                + _hardware_note()
            )
        return None
