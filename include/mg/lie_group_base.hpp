#pragma once

#include "matrix_group_types.hpp"

namespace mg {

// ============================================================================
// LieGroupBase<Derived, N>: CRTP base class for all matrix Lie groups.
//
// Provides common group operations (multiplication, inverse, identity,
// adjoint action, distance) that rely on the matrix representation.
// Derived classes must:
//   - Provide a constructor from MatrixN<N>
//   - Implement is_valid(), Log()
//   - Provide static Exp() and static Retract()
// ============================================================================

template <typename Derived, int N>
class LieGroupBase {
public:
    // ------------------------------------------------------------------ types
    using AmbientDim = std::integral_constant<int, N>;
    using MatrixType = MatrixN<N>;
    using VectorType = VectorN<N>;

    // -------------------------------------------------------- static constants
    static constexpr int AmbientDimValue = N;

    // ---------------------------------------------------------------- ctors /
    LieGroupBase() = default;
    LieGroupBase(const LieGroupBase&) = default;
    LieGroupBase(LieGroupBase&&) noexcept = default;
    LieGroupBase& operator=(const LieGroupBase&) = default;
    LieGroupBase& operator=(LieGroupBase&&) noexcept = default;
    virtual ~LieGroupBase() = default;

    // ---------------------------------------------------- matrix accessor (const)
    const MatrixN<N>& matrix() const { return matrix_; }

    // --------------------------------------------------- matrix accessor (mutable)
    MatrixN<N>& matrix() { return matrix_; }

    // ================================================================
    //  Group operations
    // ================================================================

    // ---------------------------------------------------------------- multiply
    // Eigen expressions are evaluated before being passed to the Derived ctor.
    Derived operator*(const Derived& other) const {
        MatrixN<N> prod = this->matrix() * other.matrix();
        return Derived(prod);
    }

    // --------------------------------------------------------------- inverse
    Derived inverse() const {
        MatrixN<N> inv = matrix().inverse();
        return Derived(inv);
    }

    // --------------------------------------------------------------- identity
    static Derived Identity() {
        return Derived(MatrixN<N>::Identity());
    }

    // --------------------------------------------------------- exponential map
    // Maps a Lie algebra element (ambient-sized matrix) to the group.
    // Must be provided by each derived class.
    static MatrixN<N> Exp(const MatrixN<N>& xi);

    // ------------------------------------------------------ logarithmic map
    // Maps the current group element to its Lie algebra representation.
    // Must be provided by each derived class.
    virtual MatrixN<N> Log() const = 0;

    // ----------------------------------------------------------- adjoint action
    // Computes the adjoint action: g X g^{-1}
    MatrixN<N> Adjoint(const MatrixN<N>& X) const {
        return matrix() * X * matrix().inverse();
    }

    // --------------------------------------------------------- retraction
    // Maps a Lie algebra element to the group (approximate, cheaper than Exp).
    // Must be provided by each derived class.
    static Derived Retract(const MatrixN<N>& xi);

    // ================================================================
    //  Validation
    // ================================================================

    // Checks whether the matrix satisfies the group constraints.
    virtual bool is_valid(Scalar tol = 1e-6) const = 0;

    // ================================================================
    //  Distances
    // ================================================================

    // Distance from the identity element: ||Log(this)||
    Scalar distance_to_id() const {
        return Log().norm();
    }

    // Geodesic distance to another group element: ||Log(this^{-1} * other)||
    Scalar geodesic_distance(const Derived& other) const {
        MatrixN<N> delta = this->inverse().matrix() * other.matrix();
        // Reconstruct the logarithm via a temporary
        Derived tmp(delta);
        return tmp.Log().norm();
    }

protected:
    // The ambient matrix representation (e.g. 3x3 for SO(3), 4x4 for SE(3))
    MatrixN<N> matrix_;
};

} // namespace mg
