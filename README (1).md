# Matrix Lie Groups: SO(n), SE(3), SymPD — Computational Kernel

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)]()
[![Eigen](https://img.shields.io/badge/Eigen-3.4+-0C66F2.svg)]()
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A high-performance **C++20** computational kernel for matrix Lie groups with
first-class **PyTorch autograd** support and **CUDA batch kernels** for
machine-learning-scale operations.

---

## Features

- **SO(3)** — Rodrigues formula, Cayley retraction, quaternion SLERP
- **General SO(n)** — Householder QR retraction, eigenvalue-based exp/log
- **SE(3)** — Rigid-body transformations with V-matrix exponential map
- **SymPD(n)** — Symmetric positive-definite manifold with affine-invariant
  geodesics
- **Full PyTorch autograd** — `SO3ExpFunction`, `SO3LogFunction`,
  `SE3ExpFunction` custom autograd ops
- **Trainable modules** — `SO3Parameter`, `SE3Parameter` as native
  `nn.Module` subclasses
- **CUDA batch kernels** — Gradient-friendly kernels for ML-scale operations
- **Rotation losses** — Geodesic, chordal, and quaternion-based rotation
  distance functions
- **Riemannian optimizers** — SGD and Adam variants respecting manifold
  geometry

---

## Installation

### Via pip (pure Python fallback)

```bash
pip install matrix-groups
```

### From source (CUDA extension included)

```bash
git clone https://github.com/matrix-groups/matrix-groups.git
cd matrix-groups

# Ensure Eigen 3.4+ is installed (header-only)
# e.g., on Ubuntu: sudo apt install libeigen3-dev

pip install .
```

### Building with CMake (C++ library only)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build --config Release
cmake --install build --prefix /usr/local
```

---

## Quick Start

### C++

```cpp
#include <mg/so3.hpp>
#include <mg/se3.hpp>
#include <mg/sympd.hpp>
#include <iostream>

int main() {
    // Axis-angle → rotation matrix
    Eigen::Vector3d axis(0, 0, 1);
    double angle = M_PI / 4;
    auto R = mg::SO3::from_axis_angle(axis, angle);

    // Compose rotations
    Eigen::Vector3d axis2(1, 0, 0);
    auto R2 = mg::SO3::from_axis_angle(axis2, M_PI / 6);
    auto R12 = R * R2;

    // Geodesic distance
    double dist = mg::SO3::geodesic_distance(R, R2);
    std::cout << "Geodesic distance: " << dist << std::endl;

    // SLERP interpolation at t = 0.5
    auto Rmid = mg::SO3::slerp(R, R2, 0.5);

    // SE(3) pose
    mg::SE3 pose(R, Eigen::Vector3d(1.0, 2.0, 3.0));
    auto T = pose.matrix();           // 4×4 homogeneous
    auto twist = pose.log();          // 6-vector twist coordinate
    mg::SE3 pose2 = mg::SE3::exp(twist);  // round-trip

    return 0;
}
```

### Python

```python
import torch
from matrix_groups import SO3Parameter, SO3, geodesic_loss

# Trainable rotation parameter
rot = SO3Parameter(device="cuda")

# Random target rotation
target = SO3.random().matrix().to("cuda")

optimizer = torch.optim.Adam([rot], lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    loss = geodesic_loss(rot(), target)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        angle = rot.angle().item()
        print(f"epoch {epoch:3d}  loss={loss.item():.6f}  angle={angle:.2f}°")

print("Converged rotation:", rot.matrix().detach().cpu())
```

---

## API Reference

| Group    | Header / Module | Key Operations                                           |
|----------|----------------|----------------------------------------------------------|
| SO(3)    | `mg/so3.hpp`   | `exp`, `log`, `from_axis_angle`, `slerp`, `to_quat`     |
| SO(n)    | `mg/so_n.hpp`  | `exp` (eigen), `log` (eigen), `retract` (Householder QR) |
| SE(3)    | `mg/se3.hpp`   | `exp` (V-matrix), `log`, `adjoint`, `matrix`             |
| SymPD(n) | `mg/sympd.hpp` | `expm`, `logm`, `geodesic`, `parallel_transport`         |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              User Applications / Networks           │
├──────────────────────┬──────────────────────────────┤
│   PyTorch Autograd   │     Pure C++ / CUDA API      │
│  SO3ExpFunction      │   mg::SO3, mg::SE3,          │
│  SO3LogFunction      │   mg::SO_N, mg::SymPD        │
│  SE3ExpFunction      │                              │
├──────────────────────┴──────────────────────────────┤
│              CUDA Batch Kernels (cuBLAS/cuSOLVER)   │
├─────────────────────────────────────────────────────┤
│                    Eigen 3.4                         │
├─────────────────────────────────────────────────────┤
│              CUDA Runtime / C++20 Std Lib            │
└─────────────────────────────────────────────────────┘
```

---

## Build Requirements

| Dependency   | Version   | Required | Notes                          |
|-------------|-----------|----------|--------------------------------|
| C++ compiler| C++20     | Yes      | GCC 11+, Clang 14+, MSVC 2022+|
| CUDA        | 12.x      | Optional | Needed for GPU extensions      |
| CMake       | 3.20+     | Yes      | Build system                   |
| Eigen3      | 3.4+      | Yes      | Header-only; `apt` / `brew`    |
| PyTorch     | 2.0+      | Optional | Python bindings & autograd     |
| GTest       | 1.14+     | Optional | Unit tests                     |

---

## Testing

```bash
# C++ tests (requires GTest)
cmake -B build -DCMAKE_BUILD_TYPE=Debug .
cmake --build build
cd build && ctest --output-on-failure

# Python tests
pytest tests/ -v
```

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{matrix_groups,
  title   = {Matrix Lie Groups: {SO}({n}), {SE}(3), {SymPD} ---
             Computational Kernel},
  author  = {Matrix Groups Working Group},
  year    = {2026},
  url     = {https://github.com/matrix-groups/matrix-groups},
  license = {MIT},
}
```

---

## License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.
