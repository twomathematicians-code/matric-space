"""
Matrix Groups — PyTorch CUDA Extension Setup

Installs the C++/CUDA kernels as a loadable PyTorch extension module
and the pure-Python fallback package.
"""

import os
import sys
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Detect CUDA availability
# ---------------------------------------------------------------------------
_CUDA_AVAILABLE = False
try:
    from torch.utils.cpp_extension import CUDA_HOME
    _CUDA_AVAILABLE = CUDA_HOME is not None
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Common compiler flags
# ---------------------------------------------------------------------------
_COMMON_FLAGS = [
    "-O3",
    "-std=c++20",
    "-fPIC",
    "-Wall",
    "-Wextra",
]

_CUDA_FLAGS = [
    "-O3",
    "--expt-relaxed-constexpr",
    "--use_fast_math",
    "-std=c++17",
]

# ---------------------------------------------------------------------------
# Extension sources and include dirs
# ---------------------------------------------------------------------------
_EXT_SOURCES = [
    "src/torch_bindings.cpp",
]

_EXT_INCLUDE_DIRS = [
    str(ROOT / "include"),
]

# ---------------------------------------------------------------------------
# Extensions list
# ---------------------------------------------------------------------------
EXTENSIONS = []

if _CUDA_AVAILABLE:
    EXTENSIONS.append(
        CUDAExtension(
            name="matrix_groups_cuda",
            sources=_EXT_SOURCES,
            include_dirs=_EXT_INCLUDE_DIRS,
            extra_compile_args={
                "cxx": _COMMON_FLAGS,
                "nvcc": _CUDA_FLAGS,
            },
        )
    )
    print("[setup.py] CUDA extension will be built.")
else:
    print("[setup.py] CUDA NOT detected — building pure-Python package only.")

# ---------------------------------------------------------------------------
# Package discovery (pure-Python module under python/)
# ---------------------------------------------------------------------------
PACKAGES = find_packages(where="python")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup(
    name="matrix-groups",
    version="2.0.0",
    description=(
        "Computational kernel for matrix Lie groups "
        "(SO(n), SE(3), SymPD) with PyTorch autograd support"
    ),
    long_description=open(ROOT / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Matrix Groups Working Group",
    url="https://github.com/matrix-groups/matrix-groups",
    license="MIT",
    python_requires=">=3.9",
    packages=PACKAGES,
    package_dir={"": "python"},
    ext_modules=EXTENSIONS,
    cmdclass={
        "build_ext": BuildExtension.with_options(no_naga=True)
        if _CUDA_AVAILABLE
        else BuildExtension,
    },
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "torch-cuda-testing",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    zip_safe=False,
)
