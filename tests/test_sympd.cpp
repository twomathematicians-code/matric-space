#include <gtest/gtest.h>
#include "mg/sympd.hpp"
#include <cmath>
#include <random>

using namespace mg;

// ===========================================================================
// Helper: create a random SPD matrix via A^T A + epsilon * I
// ===========================================================================
template <int N>
SymPD<N> random_spd(std::mt19937& rng, double scale = 1.0) {
    std::normal_distribution<double> dist(0.0, scale);
    MatrixN<N> M;
    for (int i = 0; i < N * N; ++i) {
        M(i / N, i % N) = dist(rng);
    }
    // A^T A is symmetric positive semi-definite; add epsilon*I for PD
    MatrixN<N> S = M.transpose() * M + 0.1 * MatrixN<N>::Identity();
    return SymPD<N>(S);
}

// ===========================================================================
// Test 1: SymPD3Test_Construction — create from known SPD matrix, check validity
// ===========================================================================
TEST(SymPD3Test, Construction) {
    // Known SPD: [[4, 1], [1, 3]] ... but we need 3x3
    // Use [[5, 2, 1], [2, 3, 0], [1, 0, 4]]
    MatrixN<3> M;
    M << 5, 2, 1,
         2, 3, 0,
         1, 0, 4;

    SymPD<3> S(M);
    EXPECT_TRUE(S.is_valid());

    // Check symmetry
    EXPECT_LT((S.matrix() - S.matrix().transpose()).norm(), 1e-15);

    // Check positive-definiteness via Cholesky
    Eigen::LLT<MatrixN<3>> llt(S.matrix());
    EXPECT_EQ(llt.info(), Eigen::Success);
}

// ===========================================================================
// Test 2: SymPD3Test_ExpLog — round-trip through exp/log
// ===========================================================================
TEST(SymPD3Test, ExpLog) {
    std::mt19937 rng(123);
    SymPD<3> A = random_spd<3>(rng);

    // Exp: A is already SPD, so Exp(log(A)) should recover A
    MatrixN<3> log_A = A.Log();
    SymPD<3> A_rec = SymPD<3>::Exp(log_A);

    EXPECT_LT((A.matrix() - A_rec.matrix()).norm(), 1e-10);
}

// ===========================================================================
// Test 3: SymPD3Test_DistanceSymmetry — d(A,B) == d(B,A)
// ===========================================================================
TEST(SymPD3Test, DistanceSymmetry) {
    std::mt19937 rng(456);
    SymPD<3> A = random_spd<3>(rng, 0.5);
    SymPD<3> B = random_spd<3>(rng, 0.5);

    double d_AB = A.distance(B);
    double d_BA = B.distance(A);

    EXPECT_LT(std::abs(d_AB - d_BA), 1e-10);
    EXPECT_GE(d_AB, 0.0);
}

// ===========================================================================
// Test 4: SymPD3Test_GeodesicMidpoint — midpoint equidistant from A and B
// ===========================================================================
TEST(SymPD3Test, GeodesicMidpoint) {
    std::mt19937 rng(789);
    SymPD<3> A = random_spd<3>(rng, 0.3);
    SymPD<3> B = random_spd<3>(rng, 0.3);

    SymPD<3> M = A.geodesic(B, 0.5);

    double d_AM = A.distance(M);
    double d_BM = B.distance(M);

    EXPECT_LT(std::abs(d_AM - d_BM), 1e-8);
    EXPECT_TRUE(M.is_valid());
}

// ===========================================================================
// Test 5: SymPD3Test_Sqrt — Sqrt(A).Sqrt(A) ≈ A
// ===========================================================================
TEST(SymPD3Test, Sqrt) {
    std::mt19937 rng(321);
    SymPD<3> A = random_spd<3>(rng, 0.5);

    SymPD<3> sqrt_A = A.Sqrt();
    MatrixN<3> sqrt_A_matrix = sqrt_A.matrix();

    // sqrt(A) * sqrt(A) should equal A
    MatrixN<3> product = sqrt_A_matrix * sqrt_A_matrix;

    EXPECT_LT((product - A.matrix()).norm(), 1e-10);
    EXPECT_TRUE(sqrt_A.is_valid());
}

// ===========================================================================
// Test 6: SymPD3Test_ParallelTransport — transported vector has expected properties
// ===========================================================================
TEST(SymPD3Test, ParallelTransport) {
    std::mt19937 rng(654);
    SymPD<3> A = random_spd<3>(rng, 0.3);
    SymPD<3> B = random_spd<3>(rng, 0.3);

    // Random symmetric tangent vector at A
    MatrixN<3> S = MatrixN<3>::Random();
    S = 0.5 * (S + S.transpose());  // symmetrise

    // Transport S from A to B
    MatrixN<3> S_transported = A.parallel_transport(S, B);

    // The transported vector should be symmetric (tangent at B)
    MatrixN<3> diff = S_transported - S_transported.transpose();
    EXPECT_LT(diff.norm(), 1e-10);

    // Norm should be preserved (parallel transport is an isometry for the
    // affine-invariant metric)
    Scalar norm_at_A = (A.Sqrt().matrix().inverse() * S * A.Sqrt().matrix().inverse()).norm();
    Scalar norm_at_B = (B.Sqrt().matrix().inverse() * S_transported * B.Sqrt().matrix().inverse()).norm();
    EXPECT_LT(std::abs(norm_at_A - norm_at_B), 1e-6);
}
