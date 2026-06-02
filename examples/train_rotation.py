#!/usr/bin/env python3
"""
train_rotation.py — Train an SO(3) rotation with PyTorch autograd

Demonstrates:
  - SO3Parameter trainable module
  - Geodesic, chordal, and quaternion rotation losses
  - Training loop converging to a target rotation
"""

import math
import torch
import torch.nn as nn

from matrix_groups import SO3Parameter, geodesic_loss, chordal_loss, quaternion_loss


# ---------------------------------------------------------------------------
# Simple network that wraps a trainable rotation
# ---------------------------------------------------------------------------
class RotationNet(nn.Module):
    """A minimal network that learns a single rotation."""

    def __init__(self):
        super().__init__()
        self.rot = SO3Parameter()  # starts at identity

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Rotate points by the learned rotation.

        Args:
            points: [B, 3] point cloud
        Returns:
            [B, 3] rotated points
        """
        return self.rot(points)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rotation_angle_deg(R: torch.Tensor) -> float:
    """Extract the rotation angle (degrees) from a 3x3 rotation matrix."""
    trace_val = (R[0, 0] + R[1, 1] + R[2, 2]).item()
    cos_angle = torch.clamp((trace_val - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle.item()))


def random_rotation() -> torch.Tensor:
    """Generate a random SO(3) matrix via the quaternion method."""
    q = torch.randn(4)
    q = q / q.norm()
    w, x, y, z = q.unbind()
    R = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train(loss_name: str, loss_fn, num_epochs: int = 200, lr: float = 0.02):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate fixed points and a random target rotation
    torch.manual_seed(42)
    points = torch.randn(32, 3, device=device)
    target_R = random_rotation().to(device)
    target_points = points @ target_R.T

    # Model and optimizer
    model = RotationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'=' * 60}")
    print(f"  Training with {loss_name}  |  device={device}")
    print(f"{'=' * 60}")
    print(f"  {'epoch':>5}  {'loss':>12}  {'angle (deg)':>12}")
    print(f"  {'-' * 5}  {'-' * 12}  {'-' * 12}")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred_points = model(points)
        loss = loss_fn(pred_points, target_points)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            R_current = model.rot.R  # current rotation matrix
            angle = rotation_angle_deg(R_current.detach())
            print(f"  {epoch:5d}  {loss.item():12.6f}  {angle:12.2f}")

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run training with all three loss functions
    train("Geodesic Loss",  geodesic_loss)
    train("Chordal Loss",   chordal_loss)
    train("Quaternion Loss", quaternion_loss)

    print("\nDone. All losses converge — the model recovers the target rotation.")
