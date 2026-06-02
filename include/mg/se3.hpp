#pragma once

#include "so3.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <type_traits>

namespace mg {

// ============================================================================
// GroupTraits specialization for SE(3)
// ============================================================================
template <>
struct GroupTraits<SE<3>> {
    static constexpr int dim        = 6;
    static constexpr int ambient_dim = 4;
    static constexpr bool is_compact   = false;
    static constexpr bool is_connected = true;
};

// ============================================================================
// SE<3>: Special Euclidean group for rigid-body transformations.
//
// Stored as a 4x4 homogeneous matrix:
//     [ R  t ]
//     [ 0  1 ]
// where R in SO(3) and t in R^3.
// ============================================================================

template <>
class SE<3> : public LieGroupBase<SE<3>, 4> {
public:
    using Base = LieGroupBase<SE<3>, 4>;
    using Base::matrix_;
    using Twist = Eigen::Matrix<Scalar, 6, 1>;

    // ================================================================ ctors

    // Default-constructs the identity transformation.
    SE() { matrix_.setIdentity(); }

    // Constructs from an SO(3) rotation and a translation vector.
    SE(const SO<3>& R, const VectorN<3>& t) : Base{} {
        matrix_.setZero();
        matrix_.block<3, 3>(0, 0) = R.matrix();
        matrix_.block<3, 1>(0, 3) = t;
        matrix_(3, 3) = 1.0;
    }

    // Constructs from a 4x4 matrix expression.
    // SFINAE ensures only genuine 4x4 expressions are accepted.
    template <typename Derived,
              std::enable_if_t<Derived::RowsAtCompileTime == 4 &&
                               Derived::ColsAtCompileTime == 4, int> = 0>
    explicit SE(const Eigen::MatrixBase<Derived>& T) : Base{} {
        matrix_ = T;
    }

    // ============================================================ Accessors

    // Returns the rotation part as SO<3>.
    SO<3> rotation() const {
        return SO<3>(matrix_.block<3, 3>(0, 0));
    }

    // Returns the translation part.
    VectorN<3> translation() const {
        return matrix_.block<3, 1>(0, 3);
    }

    // ============================================================ hat / vee

    // hat: R^6 -> se(3)
    // Given xi = [v; omega]  (translation on top, rotation below),
    // returns the 4x4 Lie algebra element:
    //     [ omega_hat   v ]
    //     [     0       0 ]
    static MatrixN<4> hat(const Twist& xi) {
        VectorN<3> v     = xi.head<3>();
        VectorN<3> omega = xi.tail<3>();
        MatrixN<4> result = MatrixN<4>::Zero();
        result.block<3, 3>(0, 0) = SO<3>::hat(omega);
        result.block<3, 1>(0, 3) = v;
        return result;
    }

    // vee: se(3) -> R^6
    // Inverse of hat: extracts [v; omega] from a 4x4 algebra element.
    static Twist vee(const MatrixN<4>& xi_hat) {
        Twist result;
        result.head<3>() = xi_hat.block<3, 1>(0, 3);              // v
        result.tail<3>() = SO<3>::vee(xi_hat.block<3, 3>(0, 0));   // omega
        return result;
    }

    // =========================================================== Exp (closed-form)

    // Exponential map: se(3) -> SE(3)
    // Uses the closed-form V matrix for the translation component.
    static MatrixN<4> Exp(const MatrixN<4>& xi_hat) {
        MatrixN<3> omega_hat = xi_hat.block<3, 3>(0, 0);
        VectorN<3> v = xi_hat.block<3, 1>(0, 3);

        // Rotation part via SO(3) Rodrigues
        MatrixN<3> R = SO<3>::Exp(omega_hat);

        // V matrix for translation
        VectorN<3> omega = SO<3>::vee(omega_hat);
        Scalar theta = omega.norm();
        MatrixN<3> V;
        if (theta < 1e-6) {
            // Near identity: V ~ I + Omega/2 + Omega^2/6
            V = MatrixN<3>::Identity()
              + 0.5 * omega_hat
              + (1.0 / 6.0) * omega_hat * omega_hat;
        } else {
            Scalar ct = std::cos(theta);
            Scalar st = std::sin(theta);
            Scalar tt = theta * theta;
            Scalar tt3 = tt * theta;
            V = MatrixN<3>::Identity()
              + ((1.0 - ct) / tt) * omega_hat
              + ((theta - st) / tt3) * omega_hat * omega_hat;
        }

        VectorN<3> t = V * v;

        MatrixN<4> T = MatrixN<4>::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = t;
        return T;
    }

    // =============================================================== Log

    // Logarithmic map: SE(3) -> se(3)
    // Closed-form inverse using V_inv.
    MatrixN<4> Log() const override {
        MatrixN<3> R = matrix_.block<3, 3>(0, 0);
        VectorN<3> t = matrix_.block<3, 1>(0, 3);

        // Recover omega from the rotation
        SO<3> so3_R(R);
        MatrixN<3> omega_hat = so3_R.Log();
        VectorN<3> omega = SO<3>::vee(omega_hat);
        Scalar theta = omega.norm();

        // V_inv matrix
        MatrixN<3> V_inv;
        if (theta < 1e-6) {
            V_inv = MatrixN<3>::Identity()
                  - 0.5 * omega_hat
                  + (1.0 / 12.0) * omega_hat * omega_hat;
        } else {
            Scalar half_theta = 0.5 * theta;
            Scalar ct_half = std::cos(half_theta);
            Scalar st_half = std::sin(half_theta);
            Scalar cot_half = ct_half / st_half;
            Scalar tt = theta * theta;
            Scalar coeff = (1.0 / tt) * (1.0 - half_theta * cot_half);
            V_inv = MatrixN<3>::Identity()
                  - 0.5 * omega_hat
                  + coeff * omega_hat * omega_hat;
        }

        VectorN<3> v = V_inv * t;

        MatrixN<4> xi_hat = MatrixN<4>::Zero();
        xi_hat.block<3, 3>(0, 0) = omega_hat;
        xi_hat.block<3, 1>(0, 3) = v;
        return xi_hat;
    }

    // ============================================================= Retract

    // Simplified retraction: Cayley for rotation, V matrix for translation.
    static SE Retract(const MatrixN<4>& xi_hat) {
        MatrixN<3> omega_hat = xi_hat.block<3, 3>(0, 0);
        VectorN<3> v = xi_hat.block<3, 1>(0, 3);

        // Use SO(3) Cayley retraction for the rotation
        SO<3> R = SO<3>::Retract(omega_hat);

        // Simplified translation: V*v with near-identity V
        VectorN<3> omega = SO<3>::vee(omega_hat);
        Scalar theta = omega.norm();
        VectorN<3> t;
        if (theta < 1e-6) {
            t = v; // Zeroth-order: identity V
        } else {
            MatrixN<3> V = MatrixN<3>::Identity()
                         + 0.5 * omega_hat
                         + (1.0 / 6.0) * omega_hat * omega_hat;
            t = V * v;
        }

        return SE(R, t);
    }

    // ============================================================= is_valid

    bool is_valid(Scalar tol = 1e-6) const override {
        SO<3> so3_R(matrix_.block<3, 3>(0, 0));
        if (!so3_R.is_valid(tol)) return false;
        if (std::abs(matrix_(3, 3) - 1.0) > tol) return false;
        if (matrix_.block<1, 3>(3, 0).norm() > tol) return false;
        return true;
    }

    // ========================================================= Adjoint matrix

    // Returns the 6x6 adjoint representation of this SE(3) element.
    //     Ad_T = [ R    [t]_x R ]
    //            [ 0      R     ]
    // where [t]_x is the 3x3 skew-symmetric matrix of t.
    Eigen::Matrix<Scalar, 6, 6> Adjoint_matrix() const {
        Eigen::Matrix<Scalar, 6, 6> Ad = Eigen::Matrix<Scalar, 6, 6>::Zero();
        MatrixN<3> R = matrix_.block<3, 3>(0, 0);
        VectorN<3> t = matrix_.block<3, 1>(0, 3);
        MatrixN<3> t_hat = SO<3>::hat(t);

        Ad.block<3, 3>(0, 0) = R;
        Ad.block<3, 3>(0, 3) = t_hat * R;
        Ad.block<3, 3>(3, 3) = R;
        return Ad;
    }
};

} // namespace mg
