"""
Pure-Python autograd functions for SO(3) and SE(3).

Each operation is wrapped in ``torch.autograd.Function`` so that
PyTorch can back-propagate through the exponential / logarithm maps.
When the compiled C++/CUDA extension is available the forward and
backward passes are delegated to ``torch.ops.matrix_groups.*``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Extension availability flag – may be overwritten by __init__.py after a
# successful import of ``matrix_groups_cuda``.
# ---------------------------------------------------------------------------
HAS_CUDA_EXT: bool = False


# ===================================================================
# Helpers (pure-Python / NumPy-style math expressed in PyTorch ops)
# ===================================================================

def _skew(v: torch.Tensor) -> torch.Tensor:
    """Return the 3×3 skew-symmetric matrix of vector *v* ``[…, 3]``."""
    assert v.shape[-1] == 3
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    K[..., 0, 1] = -v[..., 2]
    K[..., 0, 2] = v[..., 1]
    K[..., 1, 0] = v[..., 2]
    K[..., 1, 2] = -v[..., 0]
    K[..., 2, 0] = -v[..., 1]
    K[..., 2, 1] = v[..., 0]
    return K


def _rodrigues(omega: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' formula: SO(3) exponential map.

    Parameters
    ----------
    omega : Tensor [..., 3]
        Lie-algebra element (rotation vector).

    Returns
    -------
    R : Tensor [..., 3, 3]
        Rotation matrix.
    """
    theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=1e-12)
    theta_sq = theta ** 2
    K = _skew(omega)

    # Near-zero fallback: R ≈ I + K
    small = (theta < 1e-8).squeeze(-1)

    # Full Rodrigues: R = I + sin(θ)/θ · K + (1 − cos θ)/θ² · K²
    K_sq = K @ K
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    R_full = (
        torch.eye(3, dtype=omega.dtype, device=omega.device)
        + (sin_t / theta) * K
        + ((1.0 - cos_t) / theta_sq) * K_sq
    )

    R_small = (
        torch.eye(3, dtype=omega.dtype, device=omega.device)
        + K
    )

    # Broadcast to [..., 3, 3]
    R_small = R_small.expand_as(R_full)

    R = torch.where(small[..., None, None], R_small, R_full)
    return R


def _inverse_rodrigues(R: torch.Tensor) -> torch.Tensor:
    """
    SO(3) logarithm map (inverse Rodrigues).

    Parameters
    ----------
    R : Tensor [..., 3, 3]

    Returns
    -------
    omega : Tensor [..., 3]
    """
    # cos(θ) = (tr(R) − 1) / 2
    trace = (
        R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    )  # [...]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # [...]

    small = theta < 1e-8

    # Standard formula: ω = θ / (2 sin θ) · [R − Rᵀ]ₛₖₑw
    # where we extract the skew-symmetric part.
    rx = R[..., 2, 1] - R[..., 1, 2]  # [...]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    skew_vec = torch.stack([rx, ry, rz], dim=-1)  # [..., 3]

    sin_theta = torch.sin(theta).clamp(min=1e-12)
    factor = theta / (2.0 * sin_theta)  # [...]

    omega_full = factor[..., None] * skew_vec

    # Near-zero: ω ≈ 0.5 · skew_vec  (Taylor expansion limit)
    omega_small = 0.5 * skew_vec

    omega = torch.where(small[..., None], omega_small, omega_full)
    return omega


def _se3_exp(xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SE(3) exponential map.

    Parameters
    ----------
    xi : Tensor [..., 6]
        Twist vector ``[ω | v]``.

    Returns
    -------
    R : Tensor [..., 3, 3]
        Rotation component.
    V : Tensor [..., 3, 3]
        Left Jacobian of SO(3) applied to translation.
    t : Tensor [..., 3]
        Translation component.
    """
    omega = xi[..., :3]
    v = xi[..., 3:]
    theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=1e-12)
    theta_sq = theta ** 2

    K = _skew(omega)
    K_sq = K @ K

    small = (theta < 1e-8).squeeze(-1)

    # ---- Rotation (Rodrigues) ----
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    I3 = torch.eye(3, dtype=xi.dtype, device=xi.device)
    R_full = I3 + (sin_t / theta) * K + ((1.0 - cos_t) / theta_sq) * K_sq
    R_small = I3 + K
    R = torch.where(small[..., None, None], R_small.expand_as(R_full), R_full)

    # ---- Left Jacobian V ----
    # V = I + (1 − cos θ)/θ² · K + (θ − sin θ)/θ³ · K²
    V_full = (
        I3
        + ((1.0 - cos_t) / theta_sq) * K
        + ((theta - sin_t) / (theta_sq * theta)) * K_sq
    )
    # Near-zero: V ≈ I + 0.5 K
    V_small = I3 + 0.5 * K
    V = torch.where(small[..., None, None], V_small.expand_as(V_full), V_full)

    # ---- Translation ----
    t = torch.einsum("...ij,...j->...i", V, v)

    return R, V, t


# ===================================================================
# torch.autograd.Function definitions
# ===================================================================

class SO3ExpFunction(torch.autograd.Function):
    """
    Differentiable SO(3) exponential map.

    Forward:  ω ∈ ℝ³  →  R ∈ SO(3)
    Backward: Jacobian-vector product using the right Jacobian of SO(3).
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, omega: torch.Tensor) -> torch.Tensor:
        """*omega* ``[B, 3]`` → ``R [B, 3, 3]``."""
        if HAS_CUDA_EXT:
            R = torch.ops.matrix_groups.so3_exp(omega)
        else:
            R = _rodrigues(omega)
        ctx.save_for_backward(omega, R)
        return R

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """dω = (dR ⊙ J_R)^{T} ... simplified projection."""
        omega, R = ctx.saved_tensors
        if HAS_CUDA_EXT:
            grad_omega = torch.ops.matrix_groups.so3_exp_backward(grad_output, omega, R)
        else:
            # dL/dω = tr(dR^T · dR/dω)
            # Using the identity: dR/dω projected back to so(3) via:
            #   vec(dR) · Jr
            # Simplified: dω_i = sum_{jk} grad_{jk} · dR_{jk}/dω_i
            # We use the transpose of the right Jacobian of SO(3) to map
            # the gradient in the tangent space.
            theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=1e-12)
            small = (theta < 1e-8).squeeze(-1)

            # Jr(ω) = I − (1−cos θ)/θ² · [ω]× + (θ−sin θ)/θ³ · [ω]×²
            K = _skew(omega)
            K_sq = K @ K
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)
            I3 = torch.eye(3, dtype=omega.dtype, device=omega.device)

            Jr_full = (
                I3
                - ((1.0 - cos_t) / theta ** 2) * K
                + ((theta - sin_t) / (theta ** 3)) * K_sq
            )
            # Near-zero: Jr ≈ I + 0.5 K
            Jr_small = I3 + 0.5 * K
            Jr = torch.where(small[..., None, None], Jr_small.expand_as(Jr_full), Jr_full)

            # Map tangent gradient back:  grad_so3 = R^T @ grad_output @ R
            # then flatten to vector and project.
            # A simpler and correct approach: contract grad_output with the
            # adjoint of the derivative of Rodrigues.
            # dω = unskew(R^T · dR · ...) — we use the vee map on the
            # skew-symmetric part of R^T · grad_output.
            RG = R.transpose(-1, -2) @ grad_output  # [..., 3, 3]
            # vee map: extract skew-symmetric part
            g_omega = 0.5 * torch.stack(
                [
                    RG[..., 2, 1] - RG[..., 1, 2],
                    RG[..., 0, 2] - RG[..., 2, 0],
                    RG[..., 1, 0] - RG[..., 0, 1],
                ],
                dim=-1,
            )
            # Project through right Jacobian transpose for proper gradient
            grad_omega = torch.einsum("...ij,...j->...i", Jr.transpose(-1, -2), g_omega)

        return (grad_omega,)


class SO3LogFunction(torch.autograd.Function):
    """
    Differentiable SO(3) logarithm map.

    Forward:  R ∈ SO(3)  →  ω ∈ ℝ³
    Backward: Jacobian-vector product using the inverse right Jacobian.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, R: torch.Tensor) -> torch.Tensor:
        """*R* ``[B, 3, 3]`` → *omega* ``[B, 3]``."""
        if HAS_CUDA_EXT:
            omega = torch.ops.matrix_groups.so3_log(R)
        else:
            omega = _inverse_rodrigues(R)
        ctx.save_for_backward(R, omega)
        return omega

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """dL/dR from dL/dω via Jacobian of the log map."""
        R, omega = ctx.saved_tensors
        if HAS_CUDA_EXT:
            grad_R = torch.ops.matrix_groups.so3_log_backward(grad_output, R, omega)
        else:
            # dR = Jr(ω) · dω  →  dω = Jr^{-1} · vee(R^T · dR · ...)
            # But we need dR given dω, so we propagate back through the
            # Jacobian of log: dR_{ij}/dω_k mapped from grad_output.
            theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=1e-12)
            small = (theta < 1e-8).squeeze(-1)

            K = _skew(omega)
            K_sq = K @ K
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)
            I3 = torch.eye(3, dtype=omega.dtype, device=omega.device)

            Jr_full = (
                I3
                - ((1.0 - cos_t) / theta ** 2) * K
                + ((theta - sin_t) / (theta ** 3)) * K_sq
            )
            Jr_small = I3 + 0.5 * K
            Jr = torch.where(small[..., None, None], Jr_small.expand_as(Jr_full), Jr_full)

            # dω is grad_output; propagate through Jr^{-1} to get tangent grad,
            # then re-skew and map back.
            # g_so3 = Jr^{-1} · grad_output  (gradient in tangent space)
            g_so3 = torch.linalg.solve(Jr, grad_output[..., None]).squeeze(-1)

            # Reconstruct dR from tangent gradient:
            # dR/dω_k = d/dω_k exp(ω) = ... applied to g_so3
            # Approximate: skew-symmetric outer product.
            dR = R @ _skew(g_so3)
            grad_R = 0.5 * (dR + dR.transpose(-1, -2))

        return (grad_R,)


class SE3ExpFunction(torch.autograd.Function):
    """
    Differentiable SE(3) exponential map.

    Forward:  ξ ∈ ℝ⁶  →  T ∈ SE(3)  (4×4 homogeneous matrix)
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, xi: torch.Tensor) -> torch.Tensor:
        """*xi* ``[B, 6]`` → *T* ``[B, 4, 4]``."""
        if HAS_CUDA_EXT:
            T = torch.ops.matrix_groups.se3_exp(xi)
        else:
            R, V, t = _se3_exp(xi)
            B = xi.shape[:-1]
            T = torch.zeros(*B, 4, 4, dtype=xi.dtype, device=xi.device)
            T[..., :3, :3] = R
            T[..., :3, 3] = t
            T[..., 3, 3] = 1.0
            ctx.save_for_backward(xi, R, V, t)
        return T

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Back-propagate through the SE(3) exponential."""
        xi, R, V, t = ctx.saved_tensors
        if HAS_CUDA_EXT:
            grad_xi = torch.ops.matrix_groups.se3_exp_backward(grad_output, xi, R, V, t)
        else:
            omega = xi[..., :3]
            v = xi[..., 3:]
            theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=1e-12)
            small = (theta < 1e-8).squeeze(-1)

            K = _skew(omega)
            K_sq = K @ K
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)
            I3 = torch.eye(3, dtype=xi.dtype, device=xi.device)

            # ---- Jacobian w.r.t. ω (rotation part) ----
            # dR/dω via Rodrigues derivative
            Jr_full = (
                I3
                - ((1.0 - cos_t) / theta ** 2) * K
                + ((theta - sin_t) / (theta ** 3)) * K_sq
            )
            Jr_small = I3 + 0.5 * K
            Jr = torch.where(small[..., None, None], Jr_small.expand_as(Jr_full), Jr_full)

            grad_T_rot = grad_output[..., :3, :3]  # [..., 3, 3]
            RG = R.transpose(-1, -2) @ grad_T_rot
            g_omega_from_rot = 0.5 * torch.stack(
                [
                    RG[..., 2, 1] - RG[..., 1, 2],
                    RG[..., 0, 2] - RG[..., 2, 0],
                    RG[..., 1, 0] - RG[..., 0, 1],
                ],
                dim=-1,
            )
            grad_omega_rot = torch.einsum("...ij,...j->...i", Jr.transpose(-1, -2), g_omega_from_rot)

            # ---- Jacobian w.r.t. v (translation part: dT/dv is V) ----
            grad_T_trans = grad_output[..., :3, 3]  # [..., 3]
            grad_v = torch.einsum("...ij,...j->...i", V.transpose(-1, -2), grad_T_trans)

            # ---- Cross term: dT/dω from translation row ----
            # t = V · v, so dt/dω contributes additional ω gradient
            # dV/dω · v  (simplified via finite-difference-free approximation)
            # We approximate: dV/dω ≈ 0.5 * K @ V for small θ
            dV = 0.5 * (K @ V) if small.any() else (
                ((sin_t - theta * cos_t) / (theta_sq * theta)) * K
                + ((theta_sq - 2.0 * theta * sin_t + 2.0 * (1.0 - cos_t)) / (theta_sq * theta_sq)) * K_sq
                + ((theta - sin_t) / (theta_sq * theta)) * (K_sq @ K + K @ K_sq)
            )
            dt_from_omega = torch.einsum("...ij,...j->...i", dV, v)
            grad_omega_trans = torch.einsum("...i,...i->...", grad_T_trans, dt_from_omega)
            grad_omega_trans = grad_omega_trans[..., None] * omega / theta.clamp(min=1e-12)  # project to ω direction

            grad_omega = grad_omega_rot + grad_omega_trans
            grad_xi = torch.cat([grad_omega, grad_v], dim=-1)

        return (grad_xi,)
