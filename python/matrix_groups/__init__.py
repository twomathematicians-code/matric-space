"""
matrix_groups — Matrix Lie Group primitives for PyTorch.

Provides differentiable SO(3) and SE(3) operations via custom
autograd functions, parameterised modules, and Riemannian
optimisers.  A compiled C++/CUDA extension can be loaded for
accelerated forward/backward passes; pure-Python (NumPy-style)
fallbacks are used otherwise.
"""

from matrix_groups.autograd import (
    SO3ExpFunction,
    SO3LogFunction,
    SE3ExpFunction,
    HAS_CUDA_EXT,
)

from matrix_groups.parameter import (
    SO3Parameter,
    SE3Parameter,
    LieGroupOptimizer,
)

from matrix_groups.losses import (
    geodesic_loss,
    chordal_loss,
    quaternion_loss,
    combined_rotation_loss,
)

from matrix_groups.optimizer import (
    RiemannianSGD,
    RiemannianAdam,
)

# Attempt to import the optional C++/CUDA extension and patch the
# autograd functions so that forward / backward use the fast path.
try:
    from matrix_groups_cuda import _C as _cuda_ext  # type: ignore[import-not-found]
    HAS_CUDA_EXT = True  # noqa: F841 — re-export for downstream checks
    # The autograd module reads this flag at import time, but we also
    # re-import to make sure the fast path is picked up.
    import importlib
    import matrix_groups.autograd as _ag
    importlib.reload(_ag)
except Exception as exc:  # pragma: no cover — extension simply not built
    import warnings
    warnings.warn(
        f"matrix_groups_cuda extension not available ({exc}). "
        "Falling back to pure-Python autograd implementations.",
        stacklevel=2,
    )
    HAS_CUDA_EXT = False  # noqa: F841

# Re-extract symbols after potential reload so ``from matrix_groups import X``
# always works regardless of whether the extension is present.
from matrix_groups.autograd import (  # noqa: E402
    SO3ExpFunction,
    SO3LogFunction,
    SE3ExpFunction,
)

__all__ = [
    # Autograd functions
    "SO3ExpFunction",
    "SO3LogFunction",
    "SE3ExpFunction",
    "HAS_CUDA_EXT",
    # Parameter modules
    "SO3Parameter",
    "SE3Parameter",
    "LieGroupOptimizer",
    # Losses
    "geodesic_loss",
    "chordal_loss",
    "quaternion_loss",
    "combined_rotation_loss",
    # Optimisers
    "RiemannianSGD",
    "RiemannianAdam",
]
