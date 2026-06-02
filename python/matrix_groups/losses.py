"""
Rotation loss functions.

All losses accept batched inputs and return a **scalar** (mean over the
batch dimension) so they integrate cleanly with standard PyTorch
training loops.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from matrix_groups.autograd import SO3LogFunction


# ===================================================================
# Geodesic loss
# ===================================================================

def geodesic_loss(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Geodesic (angular) loss on SO(3).

    .. math::
        \\mathcal{L} = \\|\\log(R_{\\text{pred}}^\\top R_{\\text{gt}})\\|_F^2

    Parameters
    ----------
    R_pred : Tensor [B, 3, 3]
    R_gt   : Tensor [B, 3, 3]
    reduction : ``"mean"``, ``"sum"``, or ``"none"``

    Returns
    -------
    Tensor (scalar if reduction is ``"mean"`` or ``"sum"``)
    """
    if R_pred.dim() == 2:
        R_pred = R_pred.unsqueeze(0)
        R_gt = R_gt.unsqueeze(0)

    R_rel = R_pred.transpose(-1, -2) @ R_gt  # [B, 3, 3]
    omega = SO3LogFunction.apply(R_rel)       # [B, 3]
    loss = (omega ** 2).sum(dim=-1)           # [B]

    return _reduce(loss, reduction)


# ===================================================================
# Chordal loss
# ===================================================================

def chordal_loss(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Frobenius-norm chordal loss.

    .. math::
        \\mathcal{L} = \\|R_{\\text{pred}} - R_{\\text{gt}}\\|_F^2

    Parameters
    ----------
    R_pred : Tensor [B, 3, 3]
    R_gt   : Tensor [B, 3, 3]
    reduction : ``"mean"``, ``"sum"``, or ``"none"``

    Returns
    -------
    Tensor
    """
    if R_pred.dim() == 2:
        R_pred = R_pred.unsqueeze(0)
        R_gt = R_gt.unsqueeze(0)

    diff = R_pred - R_gt                          # [B, 3, 3]
    loss = (diff ** 2).sum(dim=(-2, -1))           # [B]

    return _reduce(loss, reduction)


# ===================================================================
# Quaternion loss (with double-cover handling)
# ===================================================================

def quaternion_loss(
    q_pred: torch.Tensor,
    q_gt: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Quaternion distance with double-cover symmetry handling.

    .. math::
        \\mathcal{L} = \\min(\\|q_{\\text{pred}} - q_{\\text{gt}}\\|^2,\\;
                                \\|q_{\\text{pred}} + q_{\\text{gt}}\\|^2)

    Because ``q`` and ``-q`` represent the same rotation, the loss
    picks the smaller of the two distances.

    Parameters
    ----------
    q_pred : Tensor [B, 4] — unit quaternions ``[w, x, y, z]``
    q_gt   : Tensor [B, 4]
    reduction : ``"mean"``, ``"sum"``, or ``"none"``

    Returns
    -------
    Tensor
    """
    if q_pred.dim() == 1:
        q_pred = q_pred.unsqueeze(0)
        q_gt = q_gt.unsqueeze(0)

    diff_pos = (q_pred - q_gt) ** 2          # [B, 4]
    diff_neg = (q_pred + q_gt) ** 2          # [B, 4]
    loss = torch.minimum(
        diff_pos.sum(dim=-1),
        diff_neg.sum(dim=-1),
    )                                         # [B]

    return _reduce(loss, reduction)


# ===================================================================
# Combined rotation loss
# ===================================================================

def combined_rotation_loss(
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    alpha: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Weighted combination of geodesic and chordal losses.

    .. math::
        \\mathcal{L} = \\alpha \\, \\mathcal{L}_{\\text{geodesic}}
                       + (1 - \\alpha) \\, \\mathcal{L}_{\\text{chordal}}

    Parameters
    ----------
    R_pred : Tensor [B, 3, 3]
    R_gt   : Tensor [B, 3, 3]
    alpha  : float in [0, 1]
    reduction : ``"mean"``, ``"sum"``, or ``"none"``

    Returns
    -------
    Tensor
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    l_geo = geodesic_loss(R_pred, R_gt, reduction="none")
    l_chord = chordal_loss(R_pred, R_gt, reduction="none")
    loss = alpha * l_geo + (1.0 - alpha) * l_chord

    return _reduce(loss, reduction)


# ===================================================================
# Internal helper
# ===================================================================

def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply reduction to a per-batch-element loss tensor."""
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
