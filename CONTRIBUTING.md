# Contributing to Matrix Groups

Thank you for your interest in contributing to Matrix Groups. This document provides guidelines and workflows for contributing code, documentation, and ideas to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Code Style and Conventions](#code-style-and-conventions)
- [Adding a New Lie Group](#adding-a-new-lie-group)
- [Testing Requirements](#testing-requirements)
- [Pull Request Workflow](#pull-request-workflow)
- [Numerical Tolerance Standards](#numerical-tolerance-standards)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to maintain respectful, inclusive, and collaborative interactions.

## Getting Started

1. Fork the repository and clone your fork
2. Create a feature branch from `main`
3. Make your changes with appropriate tests
4. Submit a pull request with a clear description

```bash
git clone https://github.com/your-username/matrix-groups.git
cd matrix-groups
git checkout -b feature/my-new-group
```

## Development Environment Setup

### Prerequisites

| Tool | Minimum Version | Purpose |
|---|---|---|
| C++ compiler | GCC 11+ / Clang 14+ | Core library |
| CMake | 3.20+ | Build system |
| Eigen 3.4 | 3.4.0 | Linear algebra |
| Google Test | 1.14+ | C++ tests |
| PyTorch | 2.0+ | Python autograd |
| Python | 3.9+ | Python package |
| pytest | 7.0+ | Python tests |

### Building from Source

```bash
# Configure with tests enabled
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DMG_TESTS=ON .

# Build
cmake --build build -j$(nproc)

# Run C++ tests
cd build && ctest --output-on-failure

# Python tests
pytest tests/test_autograd.py -v
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

The pre-commit hooks run:
- `clang-format` on C++ headers
- `black` on Python files
- `isort` on Python imports
- Trailing whitespace removal

## Code Style and Conventions

### C++

- **Standard**: C++20 with strict conformance
- **Naming**:
  - Types and classes: `PascalCase` (e.g., `SO3`, `SymPD`, `LieGroupBase`)
  - Functions and methods: `snake_case` (e.g., `geodesic_distance`, `left_jacobian`)
  - Template parameters: `PascalCase` (e.g., `Derived`, `Scalar`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MIN_DIM`)
  - Namespace: `mg` (all public symbols)
- **Formatting**: Follow `clang-format` with the project `.clang-format` file
- **Includes**: Order as (1) own headers, (2) Eigen, (3) STL
- **Documentation**: Doxygen-style `///` comments for all public APIs
- **SFINAE**: Use `std::enable_if_t` with `Requires` concepts where possible
- **constexpr**: Prefer `static constexpr` for compile-time constants
- **Numerical safety**: Always handle near-identity/near-zero cases explicitly

### Python

- **Standard**: Python 3.9+ with type hints
- **Formatting**: `black` (88 char line length), `isort`
- **Naming**:
  - Classes: `PascalCase` (e.g., `SO3Parameter`, `RiemannianAdam`)
  - Functions: `snake_case` (e.g., `geodesic_loss`, `quaternion_loss`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `HAS_CUDA_EXT`)
- **Type hints**: All public functions must have full type annotations
- **Docstrings**: Google-style docstrings with `Args`, `Returns`, `Raises` sections
- **Autograd**: Custom functions must implement both `forward` and `backward` with `save_for_backward`

### CUDA

- **Kernel naming**: `{group}_{operation}_batch` (e.g., `so3_exp_batch`)
- **Parameters**: `__restrict__` on all pointer arguments
- **Block size**: 256 threads per block (configurable via macro)
- **Grid size**: `(B + block - 1) / block`
- **Unrolling**: Fully unroll small fixed-size matrix multiplies
- **Safety**: Always check `b >= B` bounds at kernel entry

## Adding a New Lie Group

To add a new Lie group `G` (e.g., `Sim(3)`, `SE(n)`, `Sp(2n)`), follow this template:

### Step 1: Update `matrix_group_types.hpp`

```cpp
// Forward declaration
template <int N> class Sim;  // Similarity group

// In the appropriate header:
template <>
struct GroupTraits<Sim<3>> {
    static constexpr int dim        = 7;  // rotation(3) + translation(3) + scale(1)
    static constexpr int ambient_dim = 4;
    static constexpr bool is_compact   = false;
    static constexpr bool is_connected = true;
};
```

### Step 2: Create `include/mg/sim3.hpp`

```cpp
#pragma once
#include "lie_group_base.hpp"

namespace mg {

template <>
class Sim<3> : public LieGroupBase<Sim<3>, 4> {
public:
    using Base = LieGroupBase<Sim<3>, 4>;
    using Base::matrix_;

    // --- Constructors ---
    Sim() { matrix_.setIdentity(); }

    // --- Exponential map ---
    static MatrixN<4> Exp(const MatrixN<4>& xi_hat) { /* ... */ }

    // --- Logarithmic map ---
    MatrixN<4> Log() const override { /* ... */ }

    // --- Retraction ---
    static Sim Retract(const MatrixN<4>& xi_hat) { /* ... */ }

    // --- Validation ---
    bool is_valid(Scalar tol = 1e-6) const override { /* ... */ }
};

} // namespace mg
```

### Step 3: Write Tests

Create `tests/test_sim3.cpp` with at least:
1. Exp/Log round-trip
2. Group closure under multiplication
3. Identity properties (T * I == T, T * T^{-1} == I)
4. Near-identity numerical stability
5. Component decomposition

### Step 4: Add PyTorch Autograd (Optional)

Create `Sim3ExpFunction` and `Sim3LogFunction` in `python/matrix_groups/autograd.py` with mathematically derived backward passes.

## Testing Requirements

### Coverage Expectations

- All new Lie groups: minimum 5 C++ tests covering exp/log round-trips, closure, identity, and edge cases
- All new autograd functions: minimum 2 Python tests (forward correctness + gradient check)
- All numerical operations: near-identity / near-singularity tests with explicit tolerance

### Running Tests

```bash
# Full C++ suite
cmake --build build && cd build && ctest --output-on-failure

# Python suite
pytest tests/ -v --tb=short

# Specific test
ctest --test-dir build -R SO3Test.ExpLogIdentity --output-on-failure
```

### Numerical Gradient Checking

For autograd functions, always verify against numerical gradients:

```python
def test_so3_exp_grad_numerical():
    omega = torch.randn(3, requires_grad=True)
    R = SO3ExpFunction.apply(omega)
    loss = R.sum()
    loss.backward()

    # Numerical gradient
    eps = 1e-6
    grad_num = torch.zeros(3)
    for i in range(3):
        wp = omega.detach().clone(); wp[i] += eps
        wm = omega.detach().clone(); wm[i] -= eps
        Rp = SO3ExpFunction.apply(wp).sum()
        Rm = SO3ExpFunction.apply(wm).sum()
        grad_num[i] = (Rp - Rm) / (2 * eps)

    torch.testing.assert_close(omega.grad, grad_num, atol=1e-4, rtol=1e-4)
```

## Pull Request Workflow

1. **Create a feature branch**: `feature/short-description` or `fix/short-description`
2. **Make atomic commits**: Each commit should represent one logical change
3. **Write tests**: All new code must have corresponding tests
4. **Update documentation**: README, API reference, or inline docs as needed
5. **Run full test suite**: Ensure all tests pass before submitting
6. **Submit PR**: Include a clear title, description, and any breaking change notes
7. **Address review**: Respond to all reviewer comments; maintainers will guide the process

### Commit Message Format

```
<type>(<scope>): <short description>

<detailed description>

<footer>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`

Examples:
```
feat(so3): add BCH series approximation for near-identity composition
fix(se3): correct V-matrix near-identity branch for large translations
test(sympd): add parallel transport isometry test for arbitrary N
docs(readme): add performance benchmark table
```

## Numerical Tolerance Standards

| Operation | Default Tolerance | Near-Identity Tolerance | Test Tolerance |
|---|---|---|---|
| Orthogonality check | `1e-6` | `1e-12` (direct) | `1e-10` |
| Determinant check | `1e-6` | `1e-12` | `1e-10` |
| Exp/Log round-trip | `1e-10` | `1e-8` (eigendecomp) | `1e-8` (SO(n>3)) |
| Geodesic distance | `1e-10` | `1e-12` | `1e-10` |
| Gradient check | `1e-4` (abs) | `1e-4` | `1e-4` |

## Documentation

- **Public APIs**: All public classes, methods, and functions must have Doxygen-style docstrings
- **Mathematical notation**: Use LaTeX-style formatting in docstrings (rendered by Sphinx)
- **Examples**: Each Lie group should have at least one example in `examples/`
- **Changelog**: All user-facing changes must be recorded in `CHANGELOG.md`

Thank you for contributing to Matrix Groups. Your work helps make rigorous geometric computation accessible to everyone.
