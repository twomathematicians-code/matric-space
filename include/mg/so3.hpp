#pragma once

#include "lie_group_base.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <type_traits>

namespace mg {

// ============================================================================
// GroupTraits specialization for SO(3)
// ============================================================================
template <>
struct GroupTraits<SO<3>> {
    static constexpr int dim        = 3;
    static constexpr int ambient_dim = 3;
    static constexpr bool is_compact   = true;
    static constexpr bool is_connected = true;
};

// ============================================================================
// SO<3>: Special orthogonal group in 3-D  (rotations).
//
// Internally stored as a 3x3 rotation matrix.
// Provides the full Rodrigues closed-form Exp/Log, Cayley retraction,
// quaternion access, Slerp interpolation, and left Jacobian utilities.
//
// Constructors use SFINAE on Eigen expression compile-time sizes to
// disambiguate the NxN matrix ctor from the axis-angle (Nx1) ctor.
// ============================================================================

template <>
class SO<3> : public LieGroupBase<SO<3>, 3> {
public:
    using Base = LieGroupBase<SO<3>, 3>;
    using Base::matrix_;

    // ================================================================ ctors

    // Default-constructs the identity rotation.
    SO() { matrix_.setIdentity(); }

    // Constructs from a 3x3 matrix expression (e.g. product, inverse, block).
    // Enabled only when the expression has exactly 3 rows and 3 columns.
    template <typename Derived,
              std::enable_if_t<Derived::RowsAtCompileTime == 3 &&
                               Derived::ColsAtCompileTime == 3, int> = 0>
    explicit SO(const Eigen::MatrixBase<Derived>& R) : Base{} {
        matrix_ = R;
    }

    // Constructs from a 3-vector expression interpreted as axis * angle.
    // Enabled only when the expression has exactly 3 rows and 1 column.
    template <typename Derived,
              std::enable_if_t<Derived::RowsAtCompileTime == 3 &&
                               Derived::ColsAtCompileTime == 1, int> = 0>
    explicit SO(const Eigen::MatrixBase<Derived>& omega) : Base{} {
        VectorN<3> w = omega;
        matrix_ = Exp(hat(w));
    }

    // Constructs from an Eigen quaternion.
    explicit SO(const Eigen::Quaterniond& q) : Base{} {
        matrix_ = q.toRotationMatrix();
    }

    // ================================================================ hat / vee

    // hat:  R^3 -> so(3)   (skew-symmetric matrix from 3-vector)
    static MatrixN<3> hat(const VectorN<3>& w) {
        return (MatrixN<3>() <<
             0,  -w(2),  w(1),
         w(2),      0, -w(0),
        -w(1),  w(0),      0).finished();
    }

    // vee:  so(3) -> R^3   (3-vector from skew-symmetric matrix)
    static VectorN<3> vee(const MatrixN<3>& Omega) {
        return VectorN<3>(Omega(2, 1), Omega(0, 2), Omega(1, 0));
    }

    // ============================================================ Exp (Rodrigues)

    // Exponential map: so(3) -> SO(3) via Rodrigues' formula.
    // Near the identity (||omega|| < 1e-6) uses a Taylor expansion.
    static MatrixN<3> Exp(const MatrixN<3>& Omega) {
        VectorN<3> omega = vee(Omega);
        Scalar theta = omega.norm();
        if (theta < 1e-6) {
            // Taylor: I + Omega + Omega^2 / 2
            return MatrixN<3>::Identity() + Omega + 0.5 * Omega * Omega;
        }
        Scalar st = std::sin(theta);
        Scalar ct = std::cos(theta);
        return MatrixN<3>::Identity()
               + (st / theta) * Omega
               + ((Scalar(1) - ct) / (theta * theta)) * Omega * Omega;
    }

    // =============================================================== Log

    // Logarithmic map: SO(3) -> so(3) via the inverse Rodrigues formula.
    // Uses the trace method to recover the angle, with a near-identity path.
    MatrixN<3> Log() const override {
        Scalar ct = std::clamp<Scalar>((matrix_.trace() - 1.0) / 2.0, -1.0, 1.0);
        Scalar theta = std::acos(ct);
        if (theta < 1e-6) {
            // Near identity: Log(R) ~ (R - R^T) / 2
            return 0.5 * (matrix_ - matrix_.transpose());
        }
        Scalar st = std::sin(theta);
        return (theta / (2.0 * st)) * (matrix_ - matrix_.transpose());
    }

    // ============================================================= Retract

    // Cayley retraction: (I - Omega)^{-1} (I + Omega)
    static SO Retract(const MatrixN<3>& Omega) {
        MatrixN<3> I = MatrixN<3>::Identity();
        MatrixN<3> R = (I - Omega).inverse() * (I + Omega);
        return SO(R);
    }

    // =========================================================== Quaternion

    Eigen::Quaterniond quaternion() const {
        return Eigen::Quaterniond(matrix_);
    }

    // ============================================================== Slerp

    // Spherical linear interpolation between two rotations.
    // R(0) = R0,  R(1) = R1,  t in [0,1].
    static SO Slerp(const SO& R0, const SO& R1, Scalar t) {
        MatrixN<3> dR = R0.inverse().matrix() * R1.matrix();
        SO dR_so3(dR);
        MatrixN<3> log_dR = dR_so3.Log();
        return R0 * SO(Exp(t * log_dR));
    }

    // ======================================================= Left Jacobian

    // Left Jacobian J_l(omega) such that Exp(omega) * v ~ J_l(omega) * v
    // for vectors in the body frame.
    static MatrixN<3> left_jacobian(const VectorN<3>& omega) {
        MatrixN<3> Omega = hat(omega);
        Scalar theta = omega.norm();
        if (theta < 1e-6) {
            return MatrixN<3>::Identity() + 0.5 * Omega + (1.0 / 6.0) * Omega * Omega;
        }
        Scalar ct = std::cos(theta);
        Scalar st = std::sin(theta);
        Scalar tt = theta * theta;
        Scalar tt3 = tt * theta;
        return MatrixN<3>::Identity()
               + ((1.0 - ct) / tt) * Omega
               + ((theta - st) / tt3) * Omega * Omega;
    }

    // Inverse of the left Jacobian.
    static MatrixN<3> left_jacobian_inv(const VectorN<3>& omega) {
        MatrixN<3> Omega = hat(omega);
        Scalar theta = omega.norm();
        if (theta < 1e-6) {
            return MatrixN<3>::Identity()
                   - 0.5 * Omega
                   + (1.0 / 12.0) * Omega * Omega;
        }
        Scalar tt = theta * theta;
        Scalar half_theta = theta * 0.5;
        Scalar ct_half = std::cos(half_theta);
        Scalar st_half = std::sin(half_theta);
        Scalar cot_half = ct_half / st_half;
        Scalar coeff = (1.0 / tt) * (1.0 - half_theta * cot_half);
        return MatrixN<3>::Identity()
               - 0.5 * Omega
               + coeff * Omega * Omega;
    }

    // ============================================================= is_valid

    bool is_valid(Scalar tol = 1e-6) const override {
        // Orthogonality: R * R^T ~ I
        MatrixN<3> id_check = matrix_ * matrix_.transpose();
        if ((id_check - MatrixN<3>::Identity()).norm() > tol) return false;
        // Determinant == +1
        if (std::abs(matrix_.determinant() - 1.0) > tol) return false;
        return true;
    }

    // ============================================================= rotate

    VectorN<3> rotate(const VectorN<3>& v) const {
        return matrix_ * v;
    }
};

} // namespace mg
