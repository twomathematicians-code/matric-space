"""
Parameterised Lie-group modules.

* ``SO3Parameter`` — learnable rotation stored as a 3-vector in
  the Lie algebra 𝔰𝔬(3).
* ``SE3Parameter`` — learnable rigid transform stored as a 6-vector
  in the Lie algebra 𝔰𝔢(3).
* ``LieGroupOptimizer`` — a lightweight helper that retracts Euclidean
  gradients onto the Lie algebra and takes a gradient step.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from matrix_groups.autograd import SE3ExpFunction, SO3ExpFunction, SO3LogFunction


# ===================================================================
# SO3Parameter
# ===================================================================

class SO3Parameter(nn.Module):
    """
    Learnable rotation parameterisation in SO(3).

    Internally a 3-vector ``omega`` in the Lie algebra is optimised.
    The rotation matrix is obtained on the fly via the exponential map.

    Parameters
    ----------
    init_rot : Tensor or None
        An initial 3×3 rotation matrix.  If *None*, the identity is used
        (``omega`` is initialised to zeros).
    """

    def __init__(self, init_rot: Optional[torch.Tensor] = None) -> None:
        super().__init__()

        if init_rot is not None:
            if init_rot.shape != (3, 3):
                raise ValueError(f"init_rot must be (3, 3), got {init_rot.shape}")
            with torch.no_grad():
                omega = SO3LogFunction.apply(
                    init_rot.unsqueeze(0).detach().clone()
                ).squeeze(0)
        else:
            omega = torch.zeros(3)

        self.omega = nn.Parameter(omega)

    # ----- properties -----

    @property
    def R(self) -> torch.Tensor:
        """Current rotation matrix ``[3, 3]``."""
        return SO3ExpFunction.apply(self.omega.unsqueeze(0)).squeeze(0)

    @property
    def quaternion(self) -> torch.Tensor:
        """
        Unit quaternion ``[w, x, y, z]`` derived from the rotation matrix.

        Uses Shepperd's method for numerical stability.
        """
        R = self.R  # (3, 3)
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0.0:
            s = 0.5 / (trace + 1.0) ** 0.5
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        q = torch.stack([w, x, y, z])
        return q / q.norm()

    # ----- methods -----

    def rotate(self, points: torch.Tensor) -> torch.Tensor:
        """
        Rotate *points* by the current rotation matrix.

        Parameters
        ----------
        points : Tensor
            ``[N, 3]`` or ``[B, N, 3]``.

        Returns
        -------
        Tensor
            Same shape as *points*.
        """
        R = self.R  # (3, 3)
        if points.dim() == 2:
            return points @ R.T
        elif points.dim() == 3:
            return points @ R.T
        else:
            raise ValueError(f"points must be [N,3] or [B,N,3], got {points.shape}")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`rotate`."""
        return self.rotate(points)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  omega={self.omega.data},\n"
            f"  R={self.R.data}\n"
            f")"
        )


# ===================================================================
# SE3Parameter
# ===================================================================

class SE3Parameter(nn.Module):
    """
    Learnable rigid transform parameterisation in SE(3).

    Internally a 6-vector ``xi = [omega | v]`` in the Lie algebra is
    optimised.  The homogeneous transform matrix is obtained on the fly
    via the exponential map.

    Parameters
    ----------
    init_rot : Tensor or None
        Initial 3×3 rotation matrix.  If *None*, identity is used.
    init_trans : Tensor or None
        Initial 3-vector translation.  If *None*, zeros are used.
    """

    def __init__(
        self,
        init_rot: Optional[torch.Tensor] = None,
        init_trans: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        if init_rot is not None:
            if init_rot.shape != (3, 3):
                raise ValueError(f"init_rot must be (3, 3), got {init_rot.shape}")
            with torch.no_grad():
                omega = SO3LogFunction.apply(
                    init_rot.unsqueeze(0).detach().clone()
                ).squeeze(0)
        else:
            omega = torch.zeros(3)

        if init_trans is not None:
            v = init_trans.detach().clone().reshape(3)
        else:
            v = torch.zeros(3)

        self.xi = nn.Parameter(torch.cat([omega, v], dim=0))  # [6]

    # ----- properties -----

    @property
    def T(self) -> torch.Tensor:
        """Current 4×4 homogeneous transform matrix."""
        return SE3ExpFunction.apply(self.xi.unsqueeze(0)).squeeze(0)

    @property
    def R(self) -> torch.Tensor:
        """Rotation component ``[3, 3]``."""
        return self.T[:3, :3]

    @property
    def t(self) -> torch.Tensor:
        """Translation component ``[3]``."""
        return self.T[:3, 3]

    # ----- methods -----

    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply the rigid transform to *points*.

        Parameters
        ----------
        points : Tensor
            ``[N, 3]`` or ``[B, N, 3]``.

        Returns
        -------
        Tensor
            Transformed points, same shape as *points*.
        """
        R = self.R  # (3, 3)
        t = self.t   # (3,)
        if points.dim() == 2:
            return points @ R.T + t
        elif points.dim() == 3:
            return points @ R.T + t
        else:
            raise ValueError(f"points must be [N,3] or [B,N,3], got {points.shape}")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`transform_points`."""
        return self.transform_points(points)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  xi={self.xi.data},\n"
            f"  T=\n{self.T.data}\n"
            f")"
        )


# ===================================================================
# LieGroupOptimizer  (lightweight helper — NOT an Optimiser subclass)
# ===================================================================

class LieGroupOptimizer:
    """
    Stateless helper that performs a single retraction step on a Lie
    group parameter.

    The Euclidean gradient (produced by :mod:`torch.autograd`) is first
    projected onto the Lie algebra via the adjoint representation, then
    used to update the internal Lie-algebra vector.

    Parameters
    ----------
    group_type : str
        ``"SO3"`` or ``"SE3"``.
    lr : float
        Learning rate.
    """

    def __init__(self, group_type: str = "SO3", lr: float = 0.01) -> None:
        if group_type.upper() not in ("SO3", "SE3"):
            raise ValueError(f"group_type must be 'SO3' or 'SE3', got '{group_type}'")
        self.group_type = group_type.upper()
        self.lr = lr

    def step(
        self,
        current: Union[SO3Parameter, SE3Parameter],
        gradient_euclidean: torch.Tensor,
    ) -> None:
        """
        Perform a retraction step.

        Parameters
        ----------
        current : SO3Parameter or SE3Parameter
            The module whose internal parameter will be updated **in-place**.
        gradient_euclidean : Tensor
            Euclidean gradient w.r.t. the internal ``omega`` / ``xi``
            vector.  Must match the shape of the parameter.
        """
        with torch.no_grad():
            if self.group_type == "SO3":
                param = current.omega  # nn.Parameter [3]
            else:
                param = current.xi  # nn.Parameter [6]

            # For SO(3) the Euclidean gradient on the Lie algebra is already
            # in the tangent space (up to the right-Jacobian correction, which
            # is handled in the autograd backward).  A simple descent step
            # suffices.
            # For SE(3) the 6-vector gradient is decomposed into rotation and
            # translation components, both already in their respective tangent
            # spaces.
            param.data -= self.lr * gradient_euclidean.to(param.data)
