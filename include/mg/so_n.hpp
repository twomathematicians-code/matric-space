#pragma once

#include "lie_group_base.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <cmath>
#include <type_traits>

namespace mg {

// ============================================================================
// GroupTraits specialization for SO<N>
// ============================================================================
template <int N>
struct GroupTraits<SO<N>> {
    static constexpr int dim        = N * (N - 1) / 2;
    static constexpr int ambient_dim = N;
    static constexpr bool is_compact   = true;
    static constexpr bool is_connected = true;
};

// ============================================================================
// SO<N>: Special orthogonal group in arbitrary dimension.
//
// Stored as an NxN orthogonal matrix with determinant +1.
// Provides:
//   - hat / vee  for minimal-coordinates <-> skew-symmetric
//   - Exp / Log   via complex eigendecomposition (near identity via Padé [3/3])
//   - Retract     via Householder QR with determinant correction
//   - Cayley      transform with polar (determinant) correction
//   - Slerp       geodesic interpolation via the matrix logarithm
//
// The specialisation SO<3> (in so3.hpp) overrides this with faster
// closed-form Rodrigues formulae.
// ============================================================================

template <int N>
class SO : public LieGroupBase<SO<N>, N> {
public:
    using Base = LieGroupBase<SO<N>, N>;
    using Base::matrix_;
    static constexpr int MinDim = N * (N - 1) / 2;
    using MinVector = Eigen::Matrix<Scalar, MinDim, 1>;

    // ================================================================ ctors

    // Default-constructs the identity rotation.
    SO() { matrix_.setIdentity(); }

    // Constructs from an NxN matrix expression.
    // SFINAE: enabled only when the expression has exactly N rows and N cols.
    template <typename Derived,
              std::enable_if_t<Derived::RowsAtCompileTime == N &&
                               Derived::ColsAtCompileTime == N, int> = 0>
    explicit SO(const Eigen::MatrixBase<Derived>& R) : Base{} {
        matrix_ = R;
    }

    // Constructs from minimal coordinates (vector of size N*(N-1)/2).
    // Applies the hat operator and then Exp.
    explicit SO(const MinVector& xi) : Base{} {
        matrix_ = Exp(hat(xi));
    }

    // ============================================================ hat / vee

    // hat: R^{N(N-1)/2} -> so(N)
    // Maps a vector of minimal coordinates to a skew-symmetric matrix.
    // Ordering: (0,1), (0,2), ..., (0,N-1), (1,2), ..., (N-2,N-1)
    static MatrixN<N> hat(const MinVector& xi) {
        MatrixN<N> Omega = MatrixN<N>::Zero();
        int k = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                Omega(i, j) =  xi(k);
                Omega(j, i) = -xi(k);
                ++k;
            }
        }
        return Omega;
    }

    // vee: so(N) -> R^{N(N-1)/2}
    // Inverse of hat: extracts upper-triangular entries in canonical order.
    static MinVector vee(const MatrixN<N>& Omega) {
        MinVector xi;
        int k = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                xi(k) = Omega(i, j);
                ++k;
            }
        }
        return xi;
    }

    // =========================================================== Exp

    // Exponential map: so(N) -> SO(N)
    //
    // Near identity (||Omega||_F < 1e-6):
    //   Uses the Padé [3/3] rational approximation:
    //     exp(Omega) ~ N3(Omega) * D3(Omega)^{-1}
    //     N3 = I + A/2 + A^2/10 + A^3/120
    //     D3 = I - A/2 + A^2/10 - A^3/120
    //
    // General case:
    //   Complex eigendecomposition of the skew-symmetric matrix.
    //   Skew-symmetric eigenvalues are purely imaginary pairs ±i*theta_j
    //   (plus 0 if N is odd).  Exp maps  ±i*theta  ->  cos(theta) ± i*sin(theta).
    static MatrixN<N> Exp(const MatrixN<N>& Omega) {
        // Skew-symmetrise to remove numerical asymmetry
        MatrixN<N> O = 0.5 * (Omega - Omega.transpose());

        // Near-identity criterion (Frobenius norm of the angle estimate)
        Scalar fnorm = O.norm();
        if (fnorm < 1e-6) {
            // Padé [3/3] approximation
            MatrixN<N> I  = MatrixN<N>::Identity();
            MatrixN<N> O2 = O * O;
            MatrixN<N> O3 = O2 * O;
            MatrixN<N> num = I + O * 0.5 + O2 / 10.0 + O3 / 120.0;
            MatrixN<N> den = I - O * 0.5 + O2 / 10.0 - O3 / 120.0;
            return den.inverse() * num;
        }

        // General case: complex eigendecomposition.
        // For a skew-symmetric matrix the eigenvalues are purely imaginary.
        Eigen::EigenSolver<MatrixN<N>> es(O);
        Eigen::VectorXcd evals = es.eigenvalues();
        Eigen::MatrixXcd evecs = es.eigenvectors();

        // Exponentiate each eigenvalue.
        Eigen::VectorXcd exp_evals = evals.array().exp();

        // Reconstruct: R = V * diag(exp_evals) * V^{-1}
        Eigen::MatrixXcd R_cpx = evecs * exp_evals.asDiagonal() * evecs.inverse();

        // The result must be real; take the real part.
        return R_cpx.real();
    }

    // =========================================================== Log

    // Logarithmic map: SO(N) -> so(N)
    //
    // Near identity (||R - I||_F < 1e-6):
    //   Log(R) ~ (R - R^T) / 2  (first-order approximation, skew-symmetric part)
    //
    // General case:
    //   Complex eigendecomposition of R.
    //   Rotation eigenvalues are e^{±i*theta_j} (plus 1 if N is odd).
    //   std::log gives the principal branch of the complex logarithm.
    MatrixN<N> Log() const override {
        MatrixN<N> R = matrix_;

        // Near-identity check
        Scalar fnorm = (R - MatrixN<N>::Identity()).norm();
        if (fnorm < 1e-6) {
            return 0.5 * (R - R.transpose());
        }

        // General case: complex eigendecomposition
        Eigen::EigenSolver<MatrixN<N>> es(R);
        Eigen::VectorXcd evals = es.eigenvalues();
        Eigen::MatrixXcd evecs = es.eigenvectors();

        // Principal log of each eigenvalue
        Eigen::VectorXcd log_evals(evals.size());
        for (int i = 0; i < evals.size(); ++i) {
            log_evals(i) = std::log(evals(i));
        }

        // Reconstruct: Omega = V * diag(log_evals) * V^{-1}
        Eigen::MatrixXcd O_cpx = evecs * log_evals.asDiagonal() * evecs.inverse();

        // Take the real part and enforce skew-symmetry
        MatrixN<N> Omega = O_cpx.real();
        return 0.5 * (Omega - Omega.transpose());
    }

    // ============================================================= Retract

    // Retraction: so(N) -> SO(N) via Householder QR with determinant correction.
    //   1. Form M = I + Omega
    //   2. Compute thin QR:  M = Q * R_upper
    //   3. R = Q  (orthogonal)
    //   4. If det(R) < 0, negate the last column of Q.
    static SO Retract(const MatrixN<N>& Omega) {
        MatrixN<N> M = MatrixN<N>::Identity() + Omega;
        Eigen::HouseholderQR<MatrixN<N>> qr(M);
        MatrixN<N> Q = qr.householderQ() * MatrixN<N>::Identity();

        // Ensure det(Q) = +1
        if (Q.determinant() < 0) {
            Q.col(N - 1) *= -1.0;
        }
        return SO(Q);
    }

    // ============================================================== Cayley

    // Cayley transform: so(N) -> SO(N)
    //   R = (I - Omega)^{-1} (I + Omega)
    // Followed by polar correction (SVD) to ensure det = +1.
    static SO Cayley(const MatrixN<N>& Omega) {
        MatrixN<N> I = MatrixN<N>::Identity();
        MatrixN<N> R = (I - Omega).inverse() * (I + Omega);

        // Polar correction via SVD: nearest rotation matrix
        Eigen::JacobiSVD<MatrixN<N>> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        MatrixN<N> U = svd.matrixU();
        MatrixN<N> V = svd.matrixV();
        R = U * V.transpose();

        // Ensure det = +1
        if (R.determinant() < 0) {
            U.col(N - 1) *= -1;
            R = U * V.transpose();
        }
        return SO(R);
    }

    // Cayley inverse: SO(N) -> so(N)
    //   Omega = (R - I)(R + I)^{-1}
    // Then skew-symmetrise to enforce purity.
    static MatrixN<N> CayleyInverse(const SO& R_elem) {
        MatrixN<N> R = R_elem.matrix();
        MatrixN<N> I = MatrixN<N>::Identity();
        MatrixN<N> Omega = (R - I) * (R + I).inverse();

        // Skew-symmetrise to ensure the result is in so(N)
        return 0.5 * (Omega - Omega.transpose());
    }

    // ============================================================= Slerp

    // Geodesic interpolation between two rotations.
    //   Slerp(R0, R1, t) = R0 * Exp( t * Log( R0^{-1} R1 ) )
    // t in [0,1].
    static SO Slerp(const SO& R0, const SO& R1, Scalar t) {
        MatrixN<N> dR = R0.inverse().matrix() * R1.matrix();
        SO dR_so(dR);
        MatrixN<N> log_dR = dR_so.Log();
        return R0 * SO(Exp(t * log_dR));
    }

    // ============================================================= is_valid

    bool is_valid(Scalar tol = 1e-6) const override {
        // Orthogonality: R * R^T ~ I
        MatrixN<N> id_check = matrix_ * matrix_.transpose();
        if ((id_check - MatrixN<N>::Identity()).norm() > tol) return false;
        // Determinant == +1
        if (std::abs(matrix_.determinant() - 1.0) > tol) return false;
        return true;
    }
};

} // namespace mg
