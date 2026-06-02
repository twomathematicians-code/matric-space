#include <gtest/gtest.h>
#include "mg/se3.hpp"
#include <cmath>

using namespace mg;

// ===========================================================================
// Test 1: ExpLogRoundTrip — xi -> T -> xi  round-trip
// ===========================================================================
TEST(SE3Test, ExpLogRoundTrip) {
    SE<3>::Twist xi;
    xi << 0.1, 0.2, 0.3, 1.0, 2.0, 3.0;

    MatrixN<4> xi_hat = SE<3>::hat(xi);
    MatrixN<4> T_mat = SE<3>::Exp(xi_hat);
    SE<3> T(T_mat);

    MatrixN<4> log_hat = T.Log();
    SE<3>::Twist xi_rec = SE<3>::vee(log_hat);

    EXPECT_LT((xi - xi_rec).norm(), 1e-10);
}

// ===========================================================================
// Test 2: GroupClosure — composition of two transforms is valid
// ===========================================================================
TEST(SE3Test, GroupClosure) {
    SE<3>::Twist xi1;
    xi1 << 0.1, 0.2, 0.3, 0.5, 0.7, 0.9;
    SE<3>::Twist xi2;
    xi2 << -0.1, 0.4, -0.3, 1.0, 2.0, 3.0;

    MatrixN<4> T1_mat = SE<3>::Exp(SE<3>::hat(xi1));
    MatrixN<4> T2_mat = SE<3>::Exp(SE<3>::hat(xi2));

    SE<3> T1(T1_mat);
    SE<3> T2(T2_mat);
    SE<3> T12 = T1 * T2;

    EXPECT_TRUE(T12.is_valid(1e-10));
}

// ===========================================================================
// Test 3: IdentityProperties — T*I == T, T*T^{-1} == I
// ===========================================================================
TEST(SE3Test, IdentityProperties) {
    SE<3>::Twist xi;
    xi << 0.5, -0.3, 0.8, 1.0, -2.0, 0.5;

    MatrixN<4> T_mat = SE<3>::Exp(SE<3>::hat(xi));
    SE<3> T(T_mat);
    SE<3> I = SE<3>::Identity();

    // T * I == T
    SE<3> TI = T * I;
    EXPECT_LT((TI.matrix() - T.matrix()).norm(), 1e-12);

    // I * T == T
    SE<3> IT = I * T;
    EXPECT_LT((IT.matrix() - T.matrix()).norm(), 1e-12);

    // T * T^{-1} == I
    SE<3> TTinv = T * T.inverse();
    EXPECT_LT((TTinv.matrix() - MatrixN<4>::Identity()).norm(), 1e-10);
}

// ===========================================================================
// Test 4: Decomposition — rotation() and translation() match original
// ===========================================================================
TEST(SE3Test, Decomposition) {
    VectorN<3> w(0.3, 0.7, -0.5);
    VectorN<3> t(1.0, 2.0, 3.0);

    SO<3> R(w);
    SE<3> T(R, t);

    SO<3> R_recovered = T.rotation();
    VectorN<3> t_recovered = T.translation();

    EXPECT_LT((R.matrix() - R_recovered.matrix()).norm(), 1e-12);
    EXPECT_LT((t - t_recovered).norm(), 1e-12);
}

// ===========================================================================
// Test 5: AdjointMatrix — verify 6x6 adjoint properties
// ===========================================================================
TEST(SE3Test, AdjointMatrix) {
    SE<3>::Twist xi;
    xi << 0.2, -0.1, 0.5, 1.0, 2.0, 3.0;

    MatrixN<4> T_mat = SE<3>::Exp(SE<3>::hat(xi));
    SE<3> T(T_mat);

    Eigen::Matrix<Scalar, 6, 6> Ad = T.Adjoint_matrix();

    // Property: Ad should have the correct block structure
    // Upper-left: R, Lower-left: 0, Lower-right: R, Upper-right: [t]_x R
    MatrixN<3> R = T.rotation().matrix();
    VectorN<3> t = T.translation();
    MatrixN<3> t_hat = SO<3>::hat(t);

    EXPECT_LT((Ad.block<3, 3>(0, 0) - R).norm(), 1e-12);
    EXPECT_LT((Ad.block<3, 3>(3, 3) - R).norm(), 1e-12);
    EXPECT_LT((Ad.block<3, 3>(3, 0)).norm(), 1e-12);  // lower-left is zero
    EXPECT_LT((Ad.block<3, 3>(0, 3) - t_hat * R).norm(), 1e-12);
}

// ===========================================================================
// Test 6: RetractionValidity — SE(3) retraction produces valid element
// ===========================================================================
TEST(SE3Test, RetractionValidity) {
    SE<3>::Twist xi;
    xi << 0.1, 0.2, 0.3, 0.5, 0.7, 0.9;

    MatrixN<4> xi_hat = SE<3>::hat(xi);
    SE<3> T_retract = SE<3>::Retract(xi_hat);

    EXPECT_TRUE(T_retract.is_valid(1e-10));

    // For small xi, retraction should be close to Exp
    MatrixN<4> T_exp = SE<3>::Exp(xi_hat);
    EXPECT_LT((T_retract.matrix() - T_exp).norm(), 0.1);
}
