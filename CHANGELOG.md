# Changelog

All notable changes to the Matrix Groups project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-06-03

### Added

#### Core Library (C++20, Header-Only)
- `mg::SO<3>`: Full 3D rotation group with Rodrigues exponential map, inverse Rodrigues logarithm, Cayley retraction (Padé [1/1]), quaternion access via Eigen, SLERP geodesic interpolation, left Jacobian `J_l(omega)` and its inverse, hat/vee isomorphism between R^3 and so(3)
- `mg::SE<3>`: Rigid-body transformation group with closed-form exponential map (V-matrix for translation), logarithm (V-inverse decomposition), 6x6 adjoint representation matrix, hat/vee for se(3), rotation/translation decomposition
- `mg::SO<N>`: General N-dimensional special orthogonal group with complex eigendecomposition-based exponential and logarithmic maps, Padé [3/3] near-identity approximation, Householder QR retraction with determinant correction, Cayley transform with SVD polar correction and involutory inverse
- `mg::SymPD<N>`: Symmetric positive-definite manifold with eigendecomposition-based matrix exp/log, affine-invariant geodesic distance, weighted geodesic interpolation, parallel transport along geodesics, matrix square root, LLT-based positive-definiteness validation
- `mg::LieGroupBase<Derived, N>`: CRTP base class providing group multiplication, inverse, identity, adjoint action, retraction, distance-to-identity, and geodesic distance
- `mg::GroupTraits<G>`: Compile-time property struct (dim, ambient_dim, is_compact, is_connected) for each group
- `MatrixGroup` C++20 concept enforcing compile-time group trait requirements

#### CUDA Kernels
- `so3_compose_batch`: Batch SO(3) matrix multiplication (B x 3x3), fully unrolled
- `so3_exp_batch`: Batch Rodrigues exponential (axis-angle to rotation matrix) with near-identity Taylor fallback
- `so3_log_batch`: Batch inverse Rodrigues (rotation matrix to axis-angle) with trace-based angle recovery
- `se3_exp_batch`: Batch SE(3) exponential with full V-matrix computation per element
- `se3_compose_batch`: Batch 4x4 homogeneous matrix multiplication
- `sympd_distance_batch`: Batch log-Euclidean SPD distance approximation

#### PyTorch Integration
- `TORCH_LIBRARY(matrix_groups)` native function registration for `so3_exp`, `so3_log`, `so3_compose`, `se3_exp`, `se3_compose`
- `SO3ExpFunction`: Custom `torch.autograd.Function` with analytically derived backward pass (right Jacobian projection)
- `SO3LogFunction`: Custom `torch.autograd.Function` with inverse Jacobian backward pass
- `SE3ExpFunction`: Custom `torch.autograd.Function` with rotation/translation/cross-term gradient decomposition
- `SO3Parameter`: `nn.Module` with unconstrained 3-vector Lie-algebra `nn.Parameter`, real-time matrix and quaternion properties, point rotation
- `SE3Parameter`: `nn.Module` with unconstrained 6-vector Lie-algebra `nn.Parameter`, real-time 4x4 transform matrix, point transformation
- `geodesic_loss`: Angular loss on SO(3) via `||log(R_pred^T * R_gt)||^2`
- `chordal_loss`: Frobenius-norm loss `||R_pred - R_gt||_F^2`
- `quaternion_loss`: Quaternion distance with double-cover symmetry handling (`min(||q-q'||^2, ||q+q'||^2)`)
- `combined_rotation_loss`: Weighted combination of geodesic and chordal losses
- `RiemannianSGD`: SGD optimizer with momentum, weight decay, and Nesterov support operating in Lie-algebra space
- `RiemannianAdam`: Adam optimizer with bias correction, AMSGrad variant, operating in Lie-algebra space
- `LieGroupOptimizer`: Stateless retraction helper for manual gradient steps

#### Testing
- 27 C++ Google Test cases covering SO(3) (9 tests), SE(3) (6 tests), SymPD(3) (6 tests), SO(4)/SO(5)/SO(8) (6 tests)
- 5 Python test classes covering Exp/Log round-trips, geodesic loss properties, quaternion double-cover, gradient flow, SE(3) point transformation
- Numerical Jacobian verification for SO(3) left Jacobian (analytical vs. finite-difference)

#### Build System
- CMake 3.20+ configuration with optional GTest, PyTorch, and CUDA discovery
- `setup.py` with `torch.utils.cpp_extension.BuildExtension` for CUDA extension
- `pyproject.toml` with PEP 621 project metadata and optional dependency groups

#### Documentation
- 23-page technical report: `docs/Matrix_Groups_Extended_SO_n_Autograd.pdf`

### Technical Notes

- SO(3) near-identity threshold: `||omega|| < 1e-6` triggers Taylor/Pañé fallback
- SO(n) near-identity threshold: `||Omega||_F < 1e-6` triggers Padé [3/3]
- SE(3) V-matrix near-identity: `theta < 1e-6` uses `V ~ I + Omega/2 + Omega^2/6`
- All eigendecompositions clamp eigenvalues away from zero (minimum `1e-15`) for numerical safety
- Householder QR retraction negates the last column of Q when `det(Q) < 0` to ensure det = +1

## [Unreleased]

### Planned
- Sim(3) similarity transforms
- SE(n) generalized rigid-body transforms
- Sp(2n) symplectic groups
- JAX and TensorFlow autograd bindings
- Mixed-precision CUDA kernels (float16/bfloat16)
- Batched SymPD eigendecomposition on GPU
- Sophus-style code generation for classical Lie groups

---

[2.0.0]: https://github.com/matrix-groups/matrix-groups/releases/tag/v2.0.0
