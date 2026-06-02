#include <gtest/gtest.h>
#include "mg/so_n.hpp"
#include <cmath>
#include <random>

using namespace mg;

// ===========================================================================
// Test 1: SO4Test_ExpLogRoundTrip — 6-DOF vector round-trip through hat/Exp/Log/vee
// ===========================================================================
TEST(SO4Test, ExpLogRoundTrip) {
    SO<4>::MinVector xi;
    xi << 0.1, 0.2, 0.3, -0.1, 0.4, -0.2;

    MatrixN<4> Omega = SO<4>::hat(xi);
    MatrixN<4> R = SO<4>::Exp(Omega);
    SO<4> R_elem(R);

    MatrixN<4> log_R = R_elem.Log();
    SO<4>::MinVector xi_rec = SO<4>::vee(log_R);

    EXPECT_LT((xi - xi_rec).norm(), 1e-8);
    EXPECT_TRUE(R_elem.is_valid(1e-8));
}

// ===========================================================================
// Test 2: SO4Test_QRRetraction — retraction produces valid SO(4) element
// ===========================================================================
TEST(SO4Test, QRRetraction) {
    SO<4>::MinVector xi;
    xi << 0.1, 0.2, 0.3, -0.1, 0.4, -0.2;

    MatrixN<4> Omega = SO<4>::hat(xi);
    SO<4> R_retract = SO<4>::Retract(Omega);

    EXPECT_TRUE(R_retract.is_valid(1e-8));
}

// ===========================================================================
// Test 3: SO4Test_CayleyInvolutory — Cayley(Cayley^{-1}(R)) ≈ R
// ===========================================================================
TEST(SO4Test, CayleyInvolutory) {
    SO<4>::MinVector xi;
    xi << 0.15, 0.25, 0.35, -0.1, 0.3, -0.15;

    // Create a known SO(4) element via Exp
    MatrixN<4> Omega = SO<4>::hat(xi);
    SO<4> R_orig = SO<4>::Exp(Omega);

    // Cayley inverse: R -> Omega
    MatrixN<4> Omega_rec = SO<4>::CayleyInverse(R_orig);

    // Cayley forward: Omega -> R
    SO<4> R_recovered = SO<4>::Cayley(Omega_rec);

    EXPECT_LT((R_orig.matrix() - R_recovered.matrix()).norm(), 1e-6);
}

// ===========================================================================
// Test 4: SO4Test_DimensionConsistency — check GroupTraits values
// ===========================================================================
TEST(SO4Test, DimensionConsistency) {
    static_assert(GroupTraits<SO<4>>::dim == 6, "SO(4) dim should be 6");
    static_assert(GroupTraits<SO<4>>::ambient_dim == 4, "SO(4) ambient_dim should be 4");
    static_assert(GroupTraits<SO<4>>::is_compact == true, "SO(4) should be compact");
    static_assert(GroupTraits<SO<4>>::is_connected == true, "SO(4) should be connected");

    EXPECT_EQ(GroupTraits<SO<4>>::dim, 6);
    EXPECT_EQ(GroupTraits<SO<4>>::ambient_dim, 4);
}

// ===========================================================================
// Test 5: SO5Test_DimensionConsistency — dim=10, ambient=5
// ===========================================================================
TEST(SO5Test, DimensionConsistency) {
    static_assert(GroupTraits<SO<5>>::dim == 10, "SO(5) dim should be 10");
    static_assert(GroupTraits<SO<5>>::ambient_dim == 5, "SO(5) ambient_dim should be 5");

    EXPECT_EQ(GroupTraits<SO<5>>::dim, 10);
    EXPECT_EQ(GroupTraits<SO<5>>::ambient_dim, 5);

    // Verify MinDim
    EXPECT_EQ(SO<5>::MinDim, 10);
}

// ===========================================================================
// Test 6: SO8Test_LargeDim — create random 28-DOF vector, Exp->Log round-trip
// ===========================================================================
TEST(SO8Test, LargeDim) {
    static_assert(GroupTraits<SO<8>>::dim == 28, "SO(8) dim should be 28");
    static_assert(GroupTraits<SO<8>>::ambient_dim == 8, "SO(8) ambient_dim should be 8");
    EXPECT_EQ(SO<8>::MinDim, 28);

    // Create a random 28-DOF vector (small values for numerical stability)
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.3);

    SO<8>::MinVector xi;
    for (int i = 0; i < 28; ++i) {
        xi(i) = dist(rng);
    }

    MatrixN<8> Omega = SO<8>::hat(xi);
    MatrixN<8> R = SO<8>::Exp(Omega);
    SO<8> R_elem(R);

    EXPECT_TRUE(R_elem.is_valid(1e-6));

    // Round-trip
    MatrixN<8> log_R = R_elem.Log();
    SO<8>::MinVector xi_rec = SO<8>::vee(log_R);

    EXPECT_LT((xi - xi_rec).norm(), 1e-5);
}
