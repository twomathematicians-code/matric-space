# Roadmap

This document outlines the planned development trajectory for Matrix Groups. Items are organized by priority and anticipated release version. The roadmap is a living document and evolves based on community needs and research directions.

---

## Current Focus: v2.1 -- Expanded Group Coverage

### Sim(3) -- Similarity Transforms
- **Motivation**: Essential for monocular SLAM (LSD-SLAM, ORB-SLAM3), where absolute scale is unobservable and must be estimated or tracked jointly with rotation and translation.
- **Scope**: 7-DOF group (rotation + translation + uniform scale), exponential map via augmented twist coordinates, logarithm, retraction.
- **Autograd**: `Sim3Parameter` and `Sim3ExpFunction` for end-to-end trainable similarity transforms in structure-from-motion networks.
- **Status**: Design phase

### SE(n) -- Generalized Rigid-Body Transforms
- **Motivation**: Point cloud registration in arbitrary dimensions, molecular conformation analysis, high-dimensional robotics.
- **Scope**: Rotation (SO(n)) + translation (R^n) stored as (n+1)x(n+1) homogeneous matrices. Generalized V-matrix for translation component.
- **Status**: Design phase

### Sp(2n) -- Symplectic Groups
- **Motivation**: Hamiltonian mechanics simulation, quantum computing circuit design, Gaussian state representations in quantum optics.
- **Scope**: 2n x 2n matrices preserving the symplectic form. Exponential map via Cayley parameterization of the Hamiltonian matrix.
- **Status**: Research phase

---

## Near-Term: v2.2 -- Performance & Ecosystem

### Multi-Framework Autograd
- **JAX bindings**: `jax.custom_vjp` wrappers for SO(3), SE(3), SymPD operations. JAX's JIT compilation will benefit enormously from the pure-functional design of the exponential maps.
- **TensorFlow compatibility**: `tf.custom_gradient` annotations for integration into TF 2.x workflows.
- **Status**: Blocked on community demand survey

### Mixed-Precision CUDA Kernels
- **float16**: Half-precision batch kernels for SO(3) and SE(3) composition, exp, and log. Suitable for training-time forward/backward passes where full `float64` precision is not needed.
- **bfloat16**: Brain-float support for NVIDIA Hopper (H100) and AMD MI300.
- **Tensor cores**: Explore use of WMMA/tensor core instructions for batch matrix multiply in composition kernels.
- **Status**: Implementation phase

### Batched SymPD GPU Operations
- **Full eigendecomposition**: Migrate `SelfAdjointEigenSolver` logic to CUDA kernels using cuSOLVER's `syevd` batched interface for accurate geodesic distances on GPU.
- **Log-Euclidean fallback**: Optimize the current element-wise approximation with a diagonal-dominant fast path.
- **Status**: Implementation phase

---

## Medium-Term: v3.0 -- Riemannian Geometry Platform

### Grassmannian Manifold Gr(k, n)
- **Motivation**: Dimensionality reduction, subspace tracking, low-rank matrix learning, ensemble learning.
- **Scope**: Set of k-dimensional subspaces of R^n. Geodesics via SVD-based projection, exponential map via horizontal lift, parallel transport via orthogonal matrix rotation.
- **Autograd**: `GrassmannianParameter` for learnable subspace representations in neural networks.

### Stiefel Manifold St(k, n)
- **Motivation**: Orthogonal matrix learning, constrained weight matrices, dictionary learning, attention mechanism constraints.
- **Scope**: Set of n x k matrices with orthonormal columns. QR-based retraction, Cayley retraction, polar retraction.
- **Autograd**: `StiefelParameter` for constrained neural network weight matrices.

### Riemannian Geometry Primitives
- **Christoffel symbols**: Compute Levi-Civita connection coefficients for supported manifolds
- **Riemann curvature tensor**: Curvature computation for convergence analysis
- **Geodesic ODE integration**: Adaptive-step Runge-Kutta for long geodesics (when closed-form is unavailable)
- **Jacobi fields**: For sensitivity analysis along geodesics

### Lie Algebra Full Support
- **su(n)**: Special unitary groups for quantum mechanics (rotation gates, SU(2)/SU(N) gates)
- **sl(n)**: Special linear groups for volume-preserving transformations
- **Root systems**: Cartan subalgebra, Weyl group, root decomposition for semisimple algebras
- **BCH formula**: Baker-Campbell-Hausdorff series for approximate group multiplication in the algebra

---

## Long-Term: v4.0 -- Universal Lie Group Platform

### Automatic Code Generation
- **Input**: Dynkin diagram (A_n, B_n, C_n, D_n) or algebra specification
- **Output**: Complete C++ header with:
  - GroupTraits specialization
  - Exponential map (via root decomposition)
  - Logarithm (via eigenvalue method)
  - Retraction (Cayley or QR)
  - hat/vee maps
  - Test suite
- **Approach**: Template metaprogramming or external Python code generator (like Sophus)
- **Status**: Research phase

### Symbolic Verification
- **Lean 4 / Isabelle**: Formal proofs of:
  - Group axiom satisfaction (closure, associativity, identity, inverse)
  - Exponential map surjectivity onto identity component
  - Baker-Campbell-Hausdorff convergence
  - Numerical stability bounds
- **Status**: Exploratory

### Cross-Platform Hardware Acceleration
- **SYCL/DPC++**: Intel GPU backend for data center deployment
- **WebGPU**: Browser-based deployment for interactive geometry visualizations
- **Vulkan Compute**: Mobile GPU acceleration for AR/VR applications
- **Status**: Research phase

### Learned Retractions
- **Motivation**: Classical retractions (Cayley, QR) are generic. For specific application domains (e.g., molecular conformations, medical image registration), learned retractions can capture domain-specific geometry and improve convergence rates.
- **Approach**: Train a small neural network to approximate the optimal retraction for a given distribution of group elements, minimizing projection error.
- **Autograd integration**: Replace `Retract(Omega)` with `LearnedRetract(Omega)` seamlessly via the existing CRTP interface.
- **Status**: Research phase

---

## Research Collaborations

We actively seek collaborations in the following areas:

| Domain | Application | Groups Needed |
|---|---|---|
| Robotics | Manipulation planning, legged locomotion | SO(3), SE(3), Sim(3) |
| Computer Vision | Structure from motion, medical imaging | SE(3), Sim(3), SymPD |
| Quantum Computing | Quantum circuit optimization, Gaussian states | SU(n), Sp(2n) |
| Molecular Dynamics | Protein folding, conformational search | SO(n), SE(n) |
| Weather/Climate | Ensemble covariance modeling | SymPD(n) |
| Machine Learning | Constrained representations, attention | SO(n), St(k,n), Gr(k,n) |

If your research group is interested in collaborating, please open a [Discussion](https://github.com/matrix-groups/matrix-groups/discussions) or reach out via the community channels.

---

## Versioning Timeline

| Version | Target Date | Key Deliverables |
|---|---|---|
| v2.1 | Q3 2026 | Sim(3), SE(n), enhanced documentation |
| v2.2 | Q4 2026 | JAX bindings, mixed-precision CUDA, batched SymPD GPU |
| v2.3 | Q1 2027 | Sp(2n), performance benchmarks publication |
| v3.0 | Q3 2027 | Grassmannian, Stiefel, Riemannian primitives, su(n)/sl(n) |
| v3.1 | Q4 2027 | BCH formula, full semisimple Lie algebra support |
| v4.0 | 2028 | Code generation, symbolic verification, cross-platform GPU |

*Timelines are estimates and subject to community priorities and contributor availability.*
