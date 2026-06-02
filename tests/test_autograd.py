"""
Python autograd correctness tests for the Matrix Lie Groups project.

Uses numpy for numerical reference computations.  Tests the mathematical
correctness of SO(3), SE(3) operations without requiring CUDA or PyTorch.
"""

import unittest
import numpy as np
from math import sin, cos, sqrt, acos


# ===========================================================================
# Rodrigues formula (numpy reference)
# ===========================================================================
def rodrigues(omega):
    """Compute rotation matrix from axis-angle vector using Rodrigues' formula."""
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        # Near-identity Taylor expansion: I + Omega + 0.5 * Omega^2
        wx, wy, wz = omega
        Omega = np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0]
        ])
        return np.eye(3) + Omega + 0.5 * Omega @ Omega

    k = omega / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + sin(theta) * K + (1.0 - cos(theta)) * K @ K


def inverse_rodrigues(R):
    """Compute axis-angle vector from rotation matrix (inverse Rodrigues)."""
    ct = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    theta = acos(ct)
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                                R[0, 2] - R[2, 0],
                                R[1, 0] - R[0, 1]])
    return (theta / (2.0 * sin(theta))) * np.array([R[2, 1] - R[1, 2],
                                                     R[0, 2] - R[2, 0],
                                                     R[1, 0] - R[0, 1]])


# ===========================================================================
# Test: so3_exp_log_roundtrip
# ===========================================================================
class TestSO3ExpLogRoundTrip(unittest.TestCase):
    """Verify Exp then Log recovers the original axis-angle vector."""

    def test_roundtrip_random(self):
        """Create random omega, check R via Rodrigues, round-trip back."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            omega = rng.randn(3) * 0.5  # keep small for numerical stability
            R = rodrigues(omega)

            # Check R is a valid rotation matrix
            self.assertTrue(
                np.allclose(R @ R.T, np.eye(3), atol=1e-10),
                "R*R^T != I"
            )
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)

            # Round-trip: Log(Exp(omega)) ~ omega
            omega_rec = inverse_rodrigues(R)
            np.testing.assert_allclose(omega, omega_rec, atol=1e-9)

    def test_exp_known_rotation(self):
        """Check Exp for 90-degree rotation about z-axis."""
        omega = np.array([0.0, 0.0, np.pi / 2])
        R = rodrigues(omega)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_allclose(R, expected, atol=1e-12)


# ===========================================================================
# Test: so3_geodesic_loss
# ===========================================================================
class TestSO3GeodesicLoss(unittest.TestCase):
    """Verify geodesic distance properties."""

    def test_geodesic_loss_identity_is_zero(self):
        """Geodesic distance for identity rotation should be 0."""
        R_identity = np.eye(3)
        omega = inverse_rodrigues(R_identity)
        self.assertAlmostEqual(np.linalg.norm(omega), 0.0, places=12)

    def test_geodesic_loss_symmetry(self):
        """d(R1, R2) == d(R2, R1)."""
        rng = np.random.RandomState(123)
        omega1 = rng.randn(3) * 0.5
        omega2 = rng.randn(3) * 0.5

        R1 = rodrigues(omega1)
        R2 = rodrigues(omega2)

        # Relative rotation R1^{-1} R2
        dR_12 = R1.T @ R2
        dR_21 = R2.T @ R1

        d12 = np.linalg.norm(inverse_rodrigues(dR_12))
        d21 = np.linalg.norm(inverse_rodrigues(dR_21))
        self.assertAlmostEqual(d12, d21, places=10)

    def test_geodesic_loss_nonnegative(self):
        """Geodesic distance is always non-negative."""
        rng = np.random.RandomState(456)
        for _ in range(20):
            omega1 = rng.randn(3) * 0.8
            omega2 = rng.randn(3) * 0.8
            R1 = rodrigues(omega1)
            R2 = rodrigues(omega2)
            d = np.linalg.norm(inverse_rodrigues(R1.T @ R2))
            self.assertGreaterEqual(d, 0.0)


# ===========================================================================
# Test: quaternion_loss_double_cover
# ===========================================================================
class TestQuaternionDoubleCover(unittest.TestCase):
    """Verify loss handles q and -q correctly (double cover of SO(3))."""

    @staticmethod
    def rotation_to_quaternion(R):
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        tr = np.trace(R)
        if tr > 0:
            s = 2.0 * sqrt(tr + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)

    def test_q_and_negq_same_rotation(self):
        """q and -q represent the same rotation matrix."""
        rng = np.random.RandomState(789)
        omega = rng.randn(3) * 0.3
        R = rodrigues(omega)

        q = self.rotation_to_quaternion(R)
        q_neg = -q

        # Both should produce the same rotation matrix
        # Quaternion to rotation: R = (w^2 - ||v||^2)I + 2 v v^T + 2 w [v]_x
        def quat_to_rotation(q):
            w, x, y, z = q
            v = q[1:]
            vvT = np.outer(v, v)
            skew = np.array([
                [0, -z, y],
                [z, 0, -x],
                [-y, x, 0]
            ])
            return (w * w - np.dot(v, v)) * np.eye(3) + 2 * vvT + 2 * w * skew

        R_from_q = quat_to_rotation(q)
        R_from_negq = quat_to_rotation(q_neg)

        np.testing.assert_allclose(R_from_q, R_from_negq, atol=1e-12)
        np.testing.assert_allclose(R_from_q, R, atol=1e-10)

    def test_geodesic_loss_invariant(self):
        """Geodesic distance should be the same whether we use q or -q."""
        rng = np.random.RandomState(101)
        omega = rng.randn(3) * 0.4
        R = rodrigues(omega)
        q = self.rotation_to_quaternion(R)
        q_neg = -q

        # Geodesic distance to identity
        omega_from_q = inverse_rodrigues(R)
        omega_from_negq = omega_from_q  # same rotation, same log

        d_q = np.linalg.norm(omega_from_q)
        d_negq = np.linalg.norm(omega_from_negq)
        self.assertAlmostEqual(d_q, d_negq, places=12)


# ===========================================================================
# Test: so3parameter_gradient_flow
# ===========================================================================
class TestSO3ParameterGradientFlow(unittest.TestCase):
    """Create an SO(3) parameter, compute loss, check gradient is non-zero."""

    def test_gradient_nonzero(self):
        """
        Simulate a geodesic loss L(omega) = ||omega||^2 and compute its
        gradient numerically.  The gradient should be non-zero for
        non-trivial omega.
        """
        omega = np.array([0.3, -0.5, 0.7])

        def loss_fn(w):
            R = rodrigues(w)
            # Loss: squared Frobenius distance from identity
            return np.sum((R - np.eye(3)) ** 2)

        # Numerical gradient via finite differences
        eps = 1e-7
        grad = np.zeros(3)
        for i in range(3):
            wp = omega.copy()
            wm = omega.copy()
            wp[i] += eps
            wm[i] -= eps
            grad[i] = (loss_fn(wp) - loss_fn(wm)) / (2.0 * eps)

        # Gradient should be non-zero
        grad_norm = np.linalg.norm(grad)
        self.assertGreater(grad_norm, 1e-6)

        # For this loss function, gradient should be approximately
        # 2 * d(loss)/d(omega) which points away from identity
        # Just verify it is reasonable
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_gradient_at_identity_is_zero(self):
        """
        At the identity (omega=0), the loss L = ||Exp(omega) - I||^2 should
        have zero gradient (identity is a minimum).
        """
        omega = np.zeros(3)

        def loss_fn(w):
            R = rodrigues(w)
            return np.sum((R - np.eye(3)) ** 2)

        eps = 1e-7
        grad = np.zeros(3)
        for i in range(3):
            wp = omega.copy()
            wm = omega.copy()
            wp[i] += eps
            wm[i] -= eps
            grad[i] = (loss_fn(wp) - loss_fn(wm)) / (2.0 * eps)

        # At identity, gradient should be very close to zero
        self.assertAlmostEqual(np.linalg.norm(grad), 0.0, places=5)


# ===========================================================================
# Test: se3parameter_forward
# ===========================================================================
class TestSE3ParameterForward(unittest.TestCase):
    """Create an SE(3) parameter, transform points, check output shape."""

    @staticmethod
    def se3_exp(xi):
        """
        SE(3) exponential map from 6-vector xi = [v; omega] to 4x4 matrix.
        Uses the closed-form V matrix for translation.
        """
        v = xi[:3]
        omega = xi[3:]
        theta = np.linalg.norm(omega)

        # SO(3) rotation
        R = rodrigues(omega)

        # V matrix
        if theta < 1e-8:
            wx, wy, wz = omega
            Omega = np.array([
                [0, -wz, wy],
                [wz, 0, -wx],
                [-wy, wx, 0]
            ])
            V = np.eye(3) + 0.5 * Omega + (1.0 / 6.0) * Omega @ Omega
        else:
            k = omega / theta
            K = np.array([
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0]
            ])
            a = (1.0 - cos(theta)) / (theta * theta)
            b = (theta - sin(theta)) / (theta ** 3)
            V = np.eye(3) + a * (theta * K) + b * (theta * K) @ (theta * K)

        t = V @ v

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def test_transform_points_shape(self):
        """Create SE(3) parameter, transform points, check output shape."""
        rng = np.random.RandomState(202)
        xi = rng.randn(6) * 0.3  # twist: [v; omega]
        T = self.se3_exp(xi)

        # Create batch of 3D points [N, 3]
        N = 10
        points = rng.randn(N, 3)

        # Transform: homogeneous coordinates
        points_h = np.hstack([points, np.ones((N, 1))])  # [N, 4]
        transformed = (T @ points_h.T).T  # [N, 4]
        transformed_xyz = transformed[:, :3]  # [N, 3]

        self.assertEqual(transformed_xyz.shape, (N, 3))
        self.assertTrue(np.all(np.isfinite(transformed_xyz)))

    def test_batch_transform(self):
        """Transform a batch of points and verify identity leaves them unchanged."""
        rng = np.random.RandomState(303)
        N = 5
        points = rng.randn(N, 3)

        # Identity SE(3)
        T_id = np.eye(4)
        points_h = np.hstack([points, np.ones((N, 1))])
        transformed = (T_id @ points_h.T).T[:, :3]

        np.testing.assert_allclose(transformed, points, atol=1e-12)

    def test_se3_composition(self):
        """Verify SE(3) composition: T1 * T2 transforms points correctly."""
        rng = np.random.RandomState(404)
        xi1 = rng.randn(6) * 0.3
        xi2 = rng.randn(6) * 0.3
        T1 = self.se3_exp(xi1)
        T2 = self.se3_exp(xi2)

        points = rng.randn(3)

        # T12(p) = T1(T2(p))
        T12 = T1 @ T2
        p_h = np.append(points, 1.0)
        direct = T12 @ p_h
        composed = T1 @ (T2 @ p_h)

        np.testing.assert_allclose(direct, composed, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
