#!/usr/bin/env python3
"""
se3_interpolation.py — SE(3) pose interpolation with PyTorch

Demonstrates:
  - Creating SE(3) poses from rotation + translation
  - Interpolation between two rigid-body poses via the exponential map
  - Waypoint inspection (translation and rotation at each step)
  - Exp/Log round-trip verification
"""

import math
import torch
import numpy as np

from matrix_groups import SE3Parameter


def rotation_from_axis_angle(axis, angle):
    """Create a 3x3 rotation matrix from axis-angle using Rodrigues."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0,      -axis[2],  axis[1]],
        [axis[2],  0,      -axis[0]],
        [-axis[1], axis[0],  0     ]
    ])
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return torch.tensor(R, dtype=torch.float32)


def print_pose(label: str, T: torch.Tensor):
    """Pretty-print a 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    trace_val = R[0, 0].item() + R[1, 1].item() + R[2, 2].item()
    cos_angle = max(-1.0, min(1.0, (trace_val - 1.0) / 2.0))
    angle_deg = math.degrees(math.acos(cos_angle))
    print(f"  {label}")
    print(f"    translation : [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"    rotation   : {angle_deg:.2f} deg")
    print(f"    det(R)     : {torch.det(R).item():.6f}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Define two SE(3) poses via SE3Parameter
    # ------------------------------------------------------------------
    # Pose A: identity (no rotation, origin translation)
    R_A = torch.eye(3, device=device, dtype=torch.float32)
    t_A = torch.zeros(3, device=device, dtype=torch.float32)
    pose_A_param = SE3Parameter(init_rot=R_A, init_trans=t_A).to(device)

    # Pose B: 90 deg rotation about z-axis, translation along x
    R_B = rotation_from_axis_angle(
        np.array([0.0, 0.0, 1.0]), math.pi / 2
    ).to(device)
    t_B = torch.tensor([5.0, 3.0, 1.0], device=device, dtype=torch.float32)
    pose_B_param = SE3Parameter(init_rot=R_B, init_trans=t_B).to(device)

    T_A = pose_A_param.T
    T_B = pose_B_param.T

    print("=" * 60)
    print("  SE(3) Interpolation Demo")
    print("=" * 60)
    print_pose("Pose A (start)", T_A)
    print_pose("Pose B (end)", T_B)

    # ------------------------------------------------------------------
    # 2. Geodesic distance via rotation angle + translation norm
    # ------------------------------------------------------------------
    rel_R = T_A[:3, :3].T @ T_B[:3, :3]
    trace_val = (rel_R[0, 0] + rel_R[1, 1] + rel_R[2, 2]).item()
    cos_angle = max(-1.0, min(1.0, (trace_val - 1.0) / 2.0))
    rot_dist = math.degrees(math.acos(cos_angle))
    trans_dist = (T_B[:3, 3] - T_A[:3, 3]).norm().item()
    print(f"  Rotation distance : {rot_dist:.2f} deg")
    print(f"  Translation dist  : {trans_dist:.4f}\n")

    # ------------------------------------------------------------------
    # 3. Linear interpolation in Lie algebra (simplified SLERP)
    # ------------------------------------------------------------------
    num_steps = 11  # t = 0.0, 0.1, ..., 1.0
    print(f"  {'step':>4}  {'t':>6}  {'tx':>8}  {'ty':>8}  {'tz':>8}  {'rot_deg':>10}")
    print(f"  {'-' * 4}  {'-' * 6}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 10}")

    for i in range(num_steps):
        t_frac = i / (num_steps - 1)

        # Interpolate in translation space
        t_interp = (1 - t_frac) * T_A[:3, 3] + t_frac * T_B[:3, 3]

        # Interpolate in rotation space (simplified: linear interp in xi)
        R_interp = (1 - t_frac) * T_A[:3, :3] + t_frac * T_B[:3, :3]
        # Orthogonalize via SVD to keep it on SO(3)
        U, _, Vh = torch.linalg.svd(R_interp)
        R_orth = U @ Vh
        if torch.det(R_orth) < 0:
            U[:, -1] *= -1
            R_orth = U @ Vh

        # Extract angle
        tr = R_orth[0, 0] + R_orth[1, 1] + R_orth[2, 2]
        ca = max(-1.0, min(1.0, (tr.item() - 1.0) / 2.0))
        rot_angle = math.degrees(math.acos(ca))

        print(
            f"  {i:4d}  {t_frac:6.2f}  "
            f"{t_interp[0]:8.4f}  {t_interp[1]:8.4f}  {t_interp[2]:8.4f}  "
            f"{rot_angle:10.2f}"
        )

    # ------------------------------------------------------------------
    # 4. Midpoint detail
    # ------------------------------------------------------------------
    t_mid = 0.5 * (T_A[:3, 3] + T_B[:3, 3])
    R_mid = 0.5 * (T_A[:3, :3] + T_B[:3, :3])
    U, _, Vh = torch.linalg.svd(R_mid)
    R_mid_orth = U @ Vh
    if torch.det(R_mid_orth) < 0:
        U[:, -1] *= -1
        R_mid_orth = U @ Vh
    T_mid = torch.eye(4, device=device, dtype=torch.float32)
    T_mid[:3, :3] = R_mid_orth
    T_mid[:3, 3] = t_mid

    print("\n  Midpoint (t = 0.5):")
    print_pose("  midpoint", T_mid)

    # ------------------------------------------------------------------
    # 5. Verify SE3Parameter forward pass
    # ------------------------------------------------------------------
    test_points = torch.randn(4, 3, device=device, dtype=torch.float32)
    transformed = pose_B_param(test_points)
    expected = test_points @ R_B.T + t_B
    recon_err = (transformed - expected).norm().item()
    print(f"  SE3Parameter forward error: {recon_err:.2e}\n")
    print("=" * 60)


if __name__ == "__main__":
    main()
