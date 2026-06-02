#include <gtest/gtest.h>
#include "mg/so3.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <random>

using namespace mg;

// ---------------------------------------------------------------------------
// Helper: finite-difference numerical Jacobian for SO(3) left Jacobian.
// ---------------------------------------------------------------------------
static MatrixN<3> numerical_left_jacobian(const VectorN<3>& omega, double eps = 1e-7) {
    MatrixN<3> J;
    for (int col = 0; col < 3; ++col) {
        VectorN<3> wp = omega;
        VectorN<3> wm = omega;
        wp(col) += eps;
        wm(col) -= eps;

        SO<3> Rp(wp);
        SO<3> Rm(wm);

        // J(:,col) = vee( Rp^{-1} Rm ... ) approx
        // More directly: Exp(omega)^{-1} Exp(omega+e_i) ~ I + J e_i + O(eps^2)
        // So (Exp(omega)^{-1} Exp(omega+e_i) - I) / eps gives J column
        SO<3> R_ref(omega);
        MatrixN<3> delta_plus  = R_ref.inverse().matrix() * Rp.matrix();
        MatrixN<3> delta_minus = R_ref.inverse().matrix() * Rm.matrix();

        VectorN<3> log_plus  = SO<3>::vee(0.5 * (delta_plus - delta_plus.transpose()));
        VectorN<3> log_minus = SO<3>::vee(0.5 * (delta_minus - delta_minus.transpose()));

        J.col(col) = (log_plus - log_minus) / (2.0 * eps);
    }
    return J;
}

// ===========================================================================
// Test 1: ExpLogIdentity — round-trip for a non-trivial axis-angle
// ===========================================================================
TEST(SO3Test, ExpLogIdentity) {
    VectorN<3> omega(0.1, 0.2, 0.3);

    SO<3> R_from_axis(omega);
    MatrixN<3> omega_hat = SO<3>::hat(omega);
    MatrixN<3> R_from_exp = SO<3>::Exp(omega_hat);

    // Both paths should produce the same matrix
    EXPECT_LT((R_from_axis.matrix() - R_from_exp).norm(), 1e-12);

    // Round-trip: Log(Exp(omega)) ~ omega_hat
    MatrixN<3> log_R = R_from_axis.Log();
    VectorN<3> omega_rec = SO<3>::vee(log_R);

    EXPECT_LT((omega - omega_rec).norm(), 1e-10);
}

// ===========================================================================
// Test 2: GroupClosure — product of two rotations is still a rotation
// ===========================================================================
TEST(SO3Test, GroupClosure) {
    VectorN<3> w1(0.5, 1.0, 0.3);
    VectorN<3> w2(-0.2, 0.7, 1.1);

    SO<3> R1(w1);
    SO<3> R2(w2);

    SO<3> R12 = R1 * R2;

    EXPECT_TRUE(R12.is_valid(1e-10));
}

// ===========================================================================
// Test 3: IdentityProperties — R*I == R, R*R^{-1} == I
// ===========================================================================
TEST(SO3Test, IdentityProperties) {
    VectorN<3> w(0.3, -0.8, 0.5);
    SO<3> R(w);
    SO<3> I = SO<3>::Identity();

    // R * I == R
    SO<3> RI = R * I;
    EXPECT_LT((RI.matrix() - R.matrix()).norm(), 1e-12);

    // I * R == R
    SO<3> IR = I * R;
    EXPECT_LT((IR.matrix() - R.matrix()).norm(), 1e-12);

    // R * R^{-1} == I
    SO<3> RRinv = R * R.inverse();
    EXPECT_LT((RRinv.matrix() - MatrixN<3>::Identity()).norm(), 1e-10);
}

// ===========================================================================
// Test 4: QuaternionConsistency — constructing from quaternion matches matrix
// ===========================================================================
TEST(SO3Test, QuaternionConsistency) {
    VectorN<3> axis(0.0, 0.0, 1.0);
    axis.normalize();
    double angle = 0.7;
    VectorN<3> omega = axis * angle;

    Eigen::Quaterniond q(Eigen::AngleAxisd(angle, axis));
    SO<3> R_from_q(q);
    SO<3> R_from_axis(omega);

    EXPECT_LT((R_from_q.matrix() - R_from_axis.matrix()).norm(), 1e-12);
}

// ===========================================================================
// Test 5: Slerp — slerp at t=0 gives R0, at t=1 gives R1
// ===========================================================================
TEST(SO3Test, Slerp) {
    VectorN<3> w0(0.1, 0.5, 0.2);
    VectorN<3> w1(-0.3, 0.2, 0.9);

    SO<3> R0(w0);
    SO<3> R1(w1);

    SO<3> R_t0 = SO<3>::Slerp(R0, R1, 0.0);
    SO<3> R_t1 = SO<3>::Slerp(R0, R1, 1.0);

    EXPECT_LT((R_t0.matrix() - R0.matrix()).norm(), 1e-10);
    EXPECT_LT((R_t1.matrix() - R1.matrix()).norm(), 1e-10);

    // Midpoint should be valid
    SO<3> R_mid = SO<3>::Slerp(R0, R1, 0.5);
    EXPECT_TRUE(R_mid.is_valid(1e-10));
}

// ===========================================================================
// Test 6: LeftJacobian — compare analytical vs numerical
// ===========================================================================
TEST(SO3Test, LeftJacobian) {
    VectorN<3> omega(0.3, -0.5, 0.8);

    MatrixN<3> J_analytical = SO<3>::left_jacobian(omega);
    MatrixN<3> J_numerical  = numerical_left_jacobian(omega);

    EXPECT_LT((J_analytical - J_numerical).norm(), 1e-6);
}

// ===========================================================================
// Test 7: NearIdentity — very small omega yields identity
// ===========================================================================
TEST(SO3Test, NearIdentity) {
    VectorN<3> omega(1e-8, -1e-8, 1e-8);

    SO<3> R(omega);
    EXPECT_LT((R.matrix() - MatrixN<3>::Identity()).norm(), 1e-5);

    // Log of identity should be near-zero
    SO<3> I = SO<3>::Identity();
    MatrixN<3> log_I = I.Log();
    EXPECT_LT(log_I.norm(), 1e-10);
}

// ===========================================================================
// Test 8: Retraction — Cayley retraction validity
// ===========================================================================
TEST(SO3Test, Retraction) {
    VectorN<3> w(0.2, 0.6, -0.4);
    MatrixN<3> Omega = SO<3>::hat(w);

    SO<3> R_retract = SO<3>::Retract(Omega);
    EXPECT_TRUE(R_retract.is_valid(1e-10));
}

// ===========================================================================
// Test 9: GeodesicDistance — distance(R,R)==0, symmetry
// ===========================================================================
TEST(SO3Test, GeodesicDistance) {
    VectorN<3> w1(0.5, 0.3, -0.7);
    VectorN<3> w2(-0.2, 0.9, 0.4);

    SO<3> R1(w1);
    SO<3> R2(w2);

    // Distance from self is zero
    EXPECT_LT(R1.geodesic_distance(R1), 1e-12);

    // Symmetry: d(R1, R2) == d(R2, R1)
    double d12 = R1.geodesic_distance(R2);
    double d21 = R2.geodesic_distance(R1);
    EXPECT_LT(std::abs(d12 - d21), 1e-10);

    // Non-negative
    EXPECT_GE(d12, 0.0);
}
