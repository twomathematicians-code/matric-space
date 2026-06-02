// =============================================================================
// basic_so3.cpp — Demonstrates core SO(3) operations
// =============================================================================
// Build:
//   g++ -std=c++20 -I /usr/include/eigen3 examples/basic_so3.cpp -o basic_so3
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>

#include <mg/so3.hpp>

int main()
{
    using Scalar = double;
    using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
    using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
    using Quaterniond = Eigen::Quaterniond;

    std::cout << std::fixed << std::setprecision(6);

    // -----------------------------------------------------------------
    // 1. Create a rotation from axis-angle via exponential map
    // -----------------------------------------------------------------
    Vector3 axis(0.0, 0.0, 1.0);   // z-axis
    double angle_deg = 90.0;
    double angle_rad = angle_deg * M_PI / 180.0;

    Vector3 omega_z = axis * angle_rad;                 // axis * angle
    Matrix3 Omega_z = mg::SO<3>::hat(omega_z);         // skew-symmetric
    mg::SO<3> Rz = mg::SO<3>::Exp(Omega_z);           // Rodrigues formula

    std::cout << "=== 1. Rotation from axis-angle ===\n"
              << "Axis:     " << axis.transpose() << "\n"
              << "Angle:    " << angle_deg << " deg\n"
              << "omega:    " << omega_z.transpose() << "\n"
              << "Rz =\n" << Rz.matrix() << "\n\n";

    // -----------------------------------------------------------------
    // 2. Compose two rotations using group multiplication
    // -----------------------------------------------------------------
    Vector3 axis2(1.0, 0.0, 0.0);   // x-axis
    double angle2_deg = 45.0;
    Vector3 omega_x = axis2 * angle2_deg * M_PI / 180.0;
    mg::SO<3> Rx = mg::SO<3>(mg::SO<3>::Exp(mg::SO<3>::hat(omega_x)));

    mg::SO<3> R_composed = Rz * Rx;                   // group operation

    std::cout << "=== 2. Compose rotations Rz * Rx ===\n"
              << "Rx (45 deg about x) =\n" << Rx.matrix() << "\n"
              << "R_composed = Rz * Rx =\n" << R_composed.matrix() << "\n\n";

    // -----------------------------------------------------------------
    // 3. Geodesic distance between two rotations
    // -----------------------------------------------------------------
    double dist = Rz.geodesic_distance(Rx);

    std::cout << "=== 3. Geodesic distance ===\n"
              << "d(Rz, Rx) = " << dist << " rad"
              << " (" << dist * 180.0 / M_PI << " deg)\n\n";

    // -----------------------------------------------------------------
    // 4. SLERP interpolation
    // -----------------------------------------------------------------
    std::cout << "=== 4. SLERP from Rx to Rz ===\n";
    for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        mg::SO<3> R_t = mg::SO<3>::Slerp(Rx, Rz, t);
        Vector3 omega_t = mg::SO<3>::vee(R_t.Log());
        double angle_t = omega_t.norm() * 180.0 / M_PI;
        std::cout << "  t=" << std::setw(4) << t
                  << "  angle=" << std::setw(8) << angle_t
                  << " deg\n";
    }
    std::cout << "\n";

    // -----------------------------------------------------------------
    // 5. Quaternion conversion round-trip
    // -----------------------------------------------------------------
    Quaterniond q = R_composed.quaternion();           // R -> quaternion
    mg::SO<3> R_back(q);                              // quaternion -> SO(3)

    std::cout << "=== 5. Quaternion round-trip ===\n"
              << "q = (" << q.w() << ", " << q.x() << ", "
              << q.y() << ", " << q.z() << ")\n"
              << "R_back =\n" << R_back.matrix() << "\n"
              << "Reconstruction error: "
              << (R_composed.matrix() - R_back.matrix()).norm() << "\n\n";

    // -----------------------------------------------------------------
    // 6. Logarithmic map and exp/log round-trip
    // -----------------------------------------------------------------
    Matrix3 Omega_log = Rz.Log();                      // SO(3) -> so(3)
    Vector3 omega_back = mg::SO<3>::vee(Omega_log);    // so(3) -> R^3
    mg::SO<3> R_reconstructed = mg::SO<3>::Exp(Omega_log);  // so(3) -> SO(3)

    std::cout << "=== 6. Log/Exp round-trip (Rz) ===\n"
              << "omega (Lie algebra) = " << omega_back.transpose() << "\n"
              << "||omega|| = " << omega_back.norm()
              << " rad (" << omega_back.norm() * 180.0 / M_PI << " deg)\n"
              << "R_reconstructed =\n" << R_reconstructed.matrix() << "\n"
              << "Error: " << (Rz.matrix() - R_reconstructed.matrix()).norm() << "\n\n";

    // -----------------------------------------------------------------
    // 7. Cayley retraction
    // -----------------------------------------------------------------
    Vector3 omega_small(0.05, 0.03, 0.02);
    Matrix3 Omega_small = mg::SO<3>::hat(omega_small);
    mg::SO<3> R_cayley = mg::SO<3>::Retract(Omega_small);

    std::cout << "=== 7. Cayley retraction ===\n"
              << "omega = " << omega_small.transpose() << "\n"
              << "R_cayley =\n" << R_cayley.matrix() << "\n"
              << "Is valid SO(3)? " << (R_cayley.is_valid() ? "YES" : "NO")
              << "  det = " << R_cayley.matrix().determinant() << "\n";

    return 0;
}
