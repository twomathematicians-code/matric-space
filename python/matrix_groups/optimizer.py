"""
Riemannian optimisers for matrix Lie groups.

:class:`RiemannianSGD` and :class:`RiemannianAdam` are drop-in
replacements for standard PyTorch optimisers that are aware of the
Lie-algebra parameterisation used by :class:`SO3Parameter` and
:class:`SE3Parameter`.

Both optimisers operate directly on the ``omega`` (SO3) or ``xi``
(SE3) :class:`~torch.nn.Parameter` stored inside the module, so they
work with standard PyTorch training loops::

    param = SO3Parameter()
    opt   = RiemannianSGD(param.parameters(), lr=0.01)

    for points, targets in loader:
        opt.zero_grad()
        loss = some_loss(param(points), targets)
        loss.backward()
        opt.step()
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["RiemannianSGD", "RiemannianAdam"]


# ===================================================================
# Shared utilities
# ===================================================================

def _lie_params(params: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    """Return the list of parameters (no filtering needed — the caller
    should pass only Lie-group parameters)."""
    return list(params)


# ===================================================================
# RiemannianSGD
# ===================================================================

class RiemannianSGD(Optimizer):
    """
    Stochastic gradient descent on a Lie group.

    Parameters
    ----------
    params : iterable
        Iterable of parameters (typically from ``SO3Parameter.parameters()``
        or ``SE3Parameter.parameters()``).
    lr : float
        Learning rate.
    momentum : float
        Momentum factor (0 = disabled).
    weight_decay : float
        Weight decay (L2 regularisation on the Lie-algebra vector).
    dampening : float
        Dampening for momentum.
    nesterov : bool
        Enables Nesterov momentum.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Perform a single optimisation step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                # Weight decay on the Lie-algebra vector (geometric
                # interpretation: shrinkage toward the group identity).
                if weight_decay != 0.0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # Momentum buffer
                if momentum != 0.0:
                    buf = self.state.get(p)
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        self.state[p] = buf
                        buf = buf
                    else:
                        buf = buf

                    buf.mul_(momentum).add_(d_p, alpha=1.0 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # Descent step in the Lie algebra
                p.add_(d_p, alpha=-lr)

        return loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lr={self.defaults['lr']}, "
            f"momentum={self.defaults['momentum']}, "
            f"weight_decay={self.defaults['weight_decay']})"
        )


# ===================================================================
# RiemannianAdam
# ===================================================================

class RiemannianAdam(Optimizer):
    """
    Adam-style optimiser on a Lie group.

    Maintains per-parameter first and second moment estimates (in the
    Lie algebra) and includes proper bias correction.

    Parameters
    ----------
    params : iterable
        Iterable of parameters.
    lr : float
        Learning rate.
    betas : (float, float)
        Coefficients for computing running averages of gradient and its
        square.  Default: ``(0.9, 0.999)``.
    eps : float
        Term added to the denominator to improve numerical stability.
    weight_decay : float
        Weight decay (applied to the Lie-algebra vector directly).
    amsgrad : bool
        Whether to use the AMSGrad variant.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Perform a single optimisation step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Weight decay
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state.setdefault(p, {})

                # State initialisation
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step_t = state["step"]

                step_t += 1.0

                # Decay first and second moment running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** step_t
                bias_correction2 = 1.0 - beta2 ** step_t
                step_size = lr / bias_correction1

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

                # Update the Lie-algebra parameter
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lr={self.defaults['lr']}, "
            f"betas={self.defaults['betas']}, "
            f"eps={self.defaults['eps']}, "
            f"weight_decay={self.defaults['weight_decay']}, "
            f"amsgrad={self.defaults['amsgrad']})"
        )
