#pragma once

#include <Eigen/Dense>
#include <concepts>
#include <type_traits>

namespace mg {

// ============================================================================
// Fundamental scalar type for the library
// ============================================================================
using Scalar = double;

// ============================================================================
// Template aliases for fixed-size square matrices and vectors
// ============================================================================
template <int N>
using MatrixN = Eigen::Matrix<Scalar, N, N>;

template <int N>
using VectorN = Eigen::Matrix<Scalar, N, 1>;

// ============================================================================
// GroupTraits: compile-time properties for any matrix Lie group
// Primary template is intentionally not specialized; specializations are
// provided in the concrete group headers (so3.hpp, se3.hpp, so_n.hpp, etc.).
// ============================================================================
template <typename G>
struct GroupTraits;

// ============================================================================
// MatrixGroup concept: requires compile-time group properties
// ============================================================================
template <typename G>
concept MatrixGroup = requires {
    // Dimension of the group (degrees of freedom)
    { GroupTraits<G>::dim } -> std::convertible_to<int>;
    // Dimension of the ambient matrix representation
    { GroupTraits<G>::ambient_dim } -> std::convertible_to<int>;
    // Whether the group is compact
    { GroupTraits<G>::is_compact } -> std::convertible_to<bool>;
    // Whether the group is connected
    { GroupTraits<G>::is_connected } -> std::convertible_to<bool>;
};

// ============================================================================
// Forward declarations of concrete matrix Lie groups
// (Use 'class' consistently so that downstream headers may inherit from them.)
// ============================================================================
template <int N> class GL;    // General linear group
template <int N> class SL;    // Special linear group
template <int N> class SO;    // Special orthogonal group
template <int N> class SE;    // Special Euclidean group
template <int N> class SymPD; // Symmetric positive definite manifold

} // namespace mg
