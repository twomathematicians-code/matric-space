#pragma once

#include "matrix_group_types.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>

namespace mg {

// ============================================================================
// GroupTraits specialization for SymPD<N>
// ============================================================================
template <int N>
struct GroupTraits<SymPD<N>> {
    static constexpr int dim        = N * (N + 1) / 2;
    static constexpr int ambient_dim = N;
    static constexpr bool is_compact   = false;
    static constexpr bool is_connected = true;
};

// ============================================================================
// SymPD<N>: Symmetric Positive-Definite manifold.
//
// NOT a Lie group, but a Riemannian manifold endowed with the
// affine-invariant metric.  Provides matrix exponential/logarithm via
// eigendecomposition, geodesic interpolation, parallel transport, and
// the matrix square root.
//
// Internally stored as an NxN symmetric positive-definite matrix.
// ============================================================================

template <int N>
class SymPD {
public:
    // ================================================================ ctors

    // Default-constructs the NxN identity matrix.
    SymPD() : matrix_(MatrixN<N>::Identity()) {}

    // Constructs from a general NxN matrix by symmetrising it:
    //   S = (M + M^T) / 2
    // The caller is responsible for ensuring the result is PD.
    explicit SymPD(const MatrixN<N>& M) {
        matrix_ = 0.5 * (M + M.transpose());
    }

    // ============================================================ Factory

    // Builds an SPD matrix from given eigenvectors (columns of U) and
    // positive eigenvalues (entries of lambda).
    static SymPD<N> FromEigen(const MatrixN<N>& U, const VectorN<N>& lambda) {
        MatrixN<N> S = U * lambda.asDiagonal() * U.transpose();
        return SymPD<N>(S);
    }

    // ============================================================ Accessor

    const MatrixN<N>& matrix() const { return matrix_; }

    // ============================================================= Exp

    // Matrix exponential via SelfAdjointEigenSolver:
    //   exp(S) = U diag(exp(lambda_i)) U^T
    static SymPD<N> Exp(const MatrixN<N>& S) {
        // Symmetrise in case S is not exactly symmetric
        MatrixN<N> Ssym = 0.5 * (S + S.transpose());
        Eigen::SelfAdjointEigenSolver<MatrixN<N>> es(Ssym);
        MatrixN<N> U      = es.eigenvectors();
        VectorN<N> lambda  = es.eigenvalues();
        VectorN<N> exp_lam = lambda.array().exp().matrix();
        MatrixN<N> result  = U * exp_lam.asDiagonal() * U.transpose();
        return SymPD<N>(result);
    }

    // =============================================================== Log

    // Matrix logarithm via SelfAdjointEigenSolver:
    //   log(S) = U diag(log(lambda_i)) U^T
    MatrixN<N> Log() const {
        Eigen::SelfAdjointEigenSolver<MatrixN<N>> es(matrix_);
        MatrixN<N> U      = es.eigenvectors();
        VectorN<N> lambda  = es.eigenvalues();
        // Clamp eigenvalues away from zero to avoid log(0)
        VectorN<N> safe_lam = lambda.array().max(1e-15).matrix();
        VectorN<N> log_lam  = safe_lam.array().log().matrix();
        return U * log_lam.asDiagonal() * U.transpose();
    }

    // ============================================================ Distance

    // Affine-invariant Riemannian distance:
    //   d(A, B) = || log( A^{-1/2} B A^{-1/2} ) ||_F
    Scalar distance(const SymPD<N>& B) const {
        // Compute A^{-1/2} B A^{-1/2} via eigendecomposition of A
        Eigen::SelfAdjointEigenSolver<MatrixN<N>> esA(matrix_);
        VectorN<N> lam_A     = esA.eigenvalues();
        MatrixN<N> U_A        = esA.eigenvectors();
        VectorN<N> inv_sqrt_lam = lam_A.array().inverse().sqrt().matrix();

        // A^{-1/2}
        MatrixN<N> A_inv_sqrt = U_A * inv_sqrt_lam.asDiagonal() * U_A.transpose();

        // C = A^{-1/2} B A^{-1/2}
        MatrixN<N> C = A_inv_sqrt * B.matrix() * A_inv_sqrt;

        Eigen::SelfAdjointEigenSolver<MatrixN<N>> esC(C);
        VectorN<N> lam_C   = esC.eigenvalues();
        VectorN<N> safe_lam = lam_C.array().max(1e-15).matrix();
        VectorN<N> log_lam  = safe_lam.array().log().matrix();

        // Frobenius norm of log(C)
        return log_lam.norm();
    }

    // ============================================================= Geodesic

    // Weighted geodesic interpolation:
    //   gamma(t) = A^{1/2} ( A^{-1/2} B A^{-1/2} )^t A^{1/2}
    // t in [0,1] gives the geodesic from A (t=0) to B (t=1).
    SymPD<N> geodesic(const SymPD<N>& B, Scalar t) const {
        Eigen::SelfAdjointEigenSolver<MatrixN<N>> esA(matrix_);
        VectorN<N> lam_A      = esA.eigenvalues();
        MatrixN<N> U_A         = esA.eigenvectors();
        VectorN<N> sqrt_lam    = lam_A.array().sqrt().matrix();
        VectorN<N> inv_sqrt_lam = lam_A.array().inverse().sqrt().matrix();

        MatrixN<N> A_sqrt     = U_A * sqrt_lam.asDiagonal() * U_A.transpose();
        MatrixN<N> A_inv_sqrt = U_A * inv_sqrt_lam.asDiagonal() * U_A.transpose();

        // C = A^{-1/2} B A^{-1/2}
        MatrixN<N> C = A_inv_sqrt * B.matrix() * A_inv_sqrt;

        // C^t via eigendecomposition
        Eigen::SelfAdjointEigenSolver<MatrixN<N>> esC(C);
        VectorN<N> lam_C      = esC.eigenvalues();
        MatrixN<N> U_C        = esC.eigenvectors();
        VectorN<N> safe_lam    = lam_C.array().max(1e-15).matrix();
        VectorN<N> pow_lam     = safe_lam.array().pow(t).matrix();

        MatrixN<N> C_t = U_C * pow_lam.asDiagonal() * U_C.transpose();

        MatrixN<N> result = A_sqrt * C_t * A_sqrt;
        return SymPD<N>(result);
    }

    // ====================================================== Parallel transport

    // Parallel transport of a tangent vector S (at this point A) along the
    // geodesic from A to `to`:
    //
    //   tau_{A->B}(S) = A^{1/2} P^{1/2} A^{-1/2} S A^{-1/2} P^{1/2} A^{1/2}
    //
    // where P = A^{-1/2} B A^{-1/2}.
    //
    // The input S is a symmetric matrix (tangent vector at A).
    MatrixN<N> parallel_transport(const MatrixN<N>& S, const SymPD<N>& to) const {
        // Eigendecompose A
        Eigen::SelfAdjointEigenSolver<MatrixN<N>> esA(matrix_);
        VectorN<N> lam_A       = esA.eigenvalues();
        MatrixN<N> U_A          = esA.eigenvectors();
        VectorN<N> sqrt_lam     = lam_A.array().sqrt().matrix();
        VectorN<N> inv_sqrt_lam = lam_A.array().inverse().sqrt().matrix();

        MatrixN<N> A_sqrt     = U_A * sqrt_lam.asDiagonal() * U_A.transpose();
        MatrixN<N> A_inv_sqrt = U_A * inv_sqrt_lam.asDiagonal() * U_A.transpose();

        // P = A^{-1/2} B A^{-1/2}
        MatrixN<N> P = A_inv_sqrt * to.matrix() * A_inv_sqrt;

        // P^{1/2}
        Eigen::SelfAdjointEigenSolver<MatrixN<N>> esP(P);
        VectorN<N> lam_P    = esP.eigenvalues();
        MatrixN<N> U_P      = esP.eigenvectors();
        VectorN<N> sqrt_lamP = lam_P.array().sqrt().matrix();
        MatrixN<N> P_sqrt   = U_P * sqrt_lamP.asDiagonal() * U_P.transpose();

        // tau = A^{1/2} P^{1/2} A^{-1/2} S A^{-1/2} P^{1/2} A^{1/2}
        MatrixN<N> tmp = A_inv_sqrt * S * A_inv_sqrt;
        tmp = P_sqrt * tmp * P_sqrt;
        tmp = A_sqrt * tmp * A_sqrt;

        // Symmetrise the result to remove numerical asymmetry
        return 0.5 * (tmp + tmp.transpose());
    }

    // ============================================================== Sqrt

    // Matrix square root via SelfAdjointEigenSolver:
    //   S^{1/2} = U diag(sqrt(lambda_i)) U^T
    SymPD<N> Sqrt() const {
        Eigen::SelfAdjointEigenSolver<MatrixN<N>> es(matrix_);
        MatrixN<N> U     = es.eigenvectors();
        VectorN<N> lambda  = es.eigenvalues();
        VectorN<N> sqrt_lam = lambda.array().sqrt().matrix();
        MatrixN<N> result  = U * sqrt_lam.asDiagonal() * U.transpose();
        return SymPD<N>(result);
    }

    // ============================================================= is_valid

    // Checks that the matrix is symmetric and positive-definite (via LLT).
    bool is_valid(Scalar tol = 1e-6) const {
        // Symmetry check
        if ((matrix_ - matrix_.transpose()).norm() > tol) return false;
        // PD check: Cholesky decomposition must succeed
        Eigen::LLT<MatrixN<N>> llt(matrix_);
        if (llt.info() != Eigen::Success) return false;
        // Verify diagonal of L is positive
        MatrixN<N> L = llt.matrixL();
        for (int i = 0; i < N; ++i) {
            if (L(i, i) <= tol) return false;
        }
        return true;
    }

private:
    MatrixN<N> matrix_;
};

} // namespace mg
