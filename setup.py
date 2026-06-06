from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def _project_metadata() -> dict[str, object]:
    return {
        "name": "qi",
        "version": "0.1.0",
        "package_dir": {"": "src"},
        "packages": find_packages("src"),
    }


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "on", "yes"}


def _build_int8_extension():
    build_mode = os.environ.get("QI_BUILD_INT8_W8A8", "auto").lower()
    if build_mode in {"0", "false", "off", "no"}:
        return [], {}

    cutlass_env = os.environ.get("QI_CUTLASS_DIR")
    cutlass_dir = Path(cutlass_env).resolve() if cutlass_env else (ROOT / "third_party" / "cutlass").resolve()
    cutlass_include = cutlass_dir / "include"
    cutlass_tools = cutlass_dir / "tools" / "util" / "include"
    missing_cutlass = not (cutlass_include / "cutlass" / "cutlass.h").exists()
    if missing_cutlass:
        message = (
            f"CUTLASS headers were not found at {cutlass_include}. "
            "Clone CUTLASS v4.4.2 to third_party/cutlass or set QI_CUTLASS_DIR."
        )
        if build_mode in {"1", "true", "on", "yes"}:
            raise RuntimeError(message)
        print(f"warning: skipping qi INT8 W8A8 CUDA extension: {message}")
        return [], {}

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as exc:
        if build_mode in {"1", "true", "on", "yes"}:
            raise
        print(f"warning: skipping qi INT8 W8A8 CUDA extension: could not import torch CUDA extension support: {exc}")
        return [], {}

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7;8.9")

    src_dir = ROOT / "src" / "qi" / "quantization" / "csrc" / "int8_w8a8"
    sources = [
        src_dir / "bindings.cpp",
        src_dir / "quantize_int8.cu",
        src_dir / "cutlass_int8_rowwise.cu",
        src_dir / "cutlass_int8_rowwise_t64x128.cu",
        src_dir / "cutlass_int8_rowwise_t64x256.cu",
        src_dir / "cutlass_int8_rowwise_t32x128.cu",
        src_dir / "cutlass_int8_rowwise_t128x64.cu",
    ]
    try:
        ext = CUDAExtension(
            name="qi.quantization._int8_w8a8_ops",
            sources=[str(path) for path in sources],
            include_dirs=[str(src_dir), str(cutlass_include), str(cutlass_tools)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "-std=c++17",
                ],
            },
        )
    except Exception as exc:
        if _truthy(build_mode):
            raise
        print(f"warning: skipping qi INT8 W8A8 CUDA extension: {exc}")
        return [], {}

    return [ext], {"build_ext": BuildExtension}


ext_modules, cmdclass = _build_int8_extension()

setup(**_project_metadata(), ext_modules=ext_modules, cmdclass=cmdclass)
