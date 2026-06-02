#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace mg {

// ============================================================================
// SO(3) batch composition:  R_out = R1 * R2  for B elements.
// Each rotation is a 3x3 matrix stored as 9 contiguous doubles in row-major
// order.  The kernel is fully unrolled for maximum throughput.
// ============================================================================
__global__ void so3_compose_batch(const double* __restrict__ R1,
                                   const double* __restrict__ R2,
                                   double* __restrict__ R_out,
                                   int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const double* a = R1 + b * 9;
    const double* b_mat = R2 + b * 9;
    double* c = R_out + b * 9;

    // Row 0
    c[0] = a[0]*b_mat[0] + a[1]*b_mat[3] + a[2]*b_mat[6];
    c[1] = a[0]*b_mat[1] + a[1]*b_mat[4] + a[2]*b_mat[7];
    c[2] = a[0]*b_mat[2] + a[1]*b_mat[5] + a[2]*b_mat[8];

    // Row 1
    c[3] = a[3]*b_mat[0] + a[4]*b_mat[3] + a[5]*b_mat[6];
    c[4] = a[3]*b_mat[1] + a[4]*b_mat[4] + a[5]*b_mat[7];
    c[5] = a[3]*b_mat[2] + a[4]*b_mat[5] + a[5]*b_mat[8];

    // Row 2
    c[6] = a[6]*b_mat[0] + a[7]*b_mat[3] + a[8]*b_mat[6];
    c[7] = a[6]*b_mat[1] + a[7]*b_mat[4] + a[8]*b_mat[7];
    c[8] = a[6]*b_mat[2] + a[7]*b_mat[5] + a[8]*b_mat[8];
}

// ============================================================================
// SO(3) batch exponential map (Rodrigues formula):  omega -> R
// Input:  omega  [B, 3]  — axis-angle vectors (axis * angle).
// Output: R      [B, 9]  — 3x3 rotation matrices in row-major order.
// Includes the near-identity Taylor expansion for small theta.
// ============================================================================
__global__ void so3_exp_batch(const double* __restrict__ omega,
                                double* __restrict__ R,
                                int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const double* w = omega + b * 3;
    double* r = R + b * 9;

    double wx = w[0], wy = w[1], wz = w[2];
    double theta = sqrt(wx*wx + wy*wy + wz*wz);

    if (theta < 1e-6) {
        // Taylor: R = I + Omega + 0.5 * Omega^2
        // Omega = [0  -wz  wy]
        //         [wz   0 -wx]
        //         [-wy  wx   0]
        r[0] = 1.0 - 0.5*(wy*wy + wz*wz);  r[1] = -wz + 0.5*wx*wy;      r[2] =  wy + 0.5*wx*wz;
        r[3] =  wz + 0.5*wx*wy;             r[4] = 1.0 - 0.5*(wx*wx+wz*wz); r[5] = -wx + 0.5*wy*wz;
        r[6] = -wy + 0.5*wx*wz;             r[7] =  wx + 0.5*wy*wz;      r[8] = 1.0 - 0.5*(wx*wx+wy*wy);
    } else {
        // Rodrigues: R = I + sin(theta)/theta * Omega + (1-cos(theta))/theta^2 * Omega^2
        double inv_theta  = 1.0 / theta;
        double st          = sin(theta);
        double ct          = cos(theta);
        double coeff1      = st * inv_theta;
        double coeff2      = (1.0 - ct) * inv_theta * inv_theta;

        // Build Omega^2 entries needed
        double wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz;
        double wxwy = wx*wy, wxwz = wx*wz, wywz = wy*wz;

        r[0] = 1.0 - coeff2*(wy2+wz2);  r[1] = -coeff1*wz - coeff2*wxwy;  r[2] =  coeff1*wy - coeff2*wxwz;
        r[3] =  coeff1*wz - coeff2*wxwy; r[4] = 1.0 - coeff2*(wx2+wz2);  r[5] = -coeff1*wx - coeff2*wywz;
        r[6] = -coeff1*wy - coeff2*wxwz; r[7] =  coeff1*wx - coeff2*wywz;  r[8] = 1.0 - coeff2*(wx2+wy2);
    }
}

// ============================================================================
// SO(3) batch logarithmic map (inverse Rodrigues):  R -> omega
// Input:  R      [B, 9]  — 3x3 rotation matrices in row-major order.
// Output: omega  [B, 3]  — axis-angle vectors.
// ============================================================================
__global__ void so3_log_batch(const double* __restrict__ R,
                                double* __restrict__ omega,
                                int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const double* r = R + b * 9;
    double* w = omega + b * 3;

    // trace(R)
    double tr = r[0] + r[4] + r[8];
    double ct = 0.5 * (tr - 1.0);
    // Clamp to valid acos range
    if (ct > 1.0) ct = 1.0;
    if (ct < -1.0) ct = -1.0;

    double theta = acos(ct);

    if (theta < 1e-6) {
        // Near identity: omega = vee( (R - R^T) / 2 )
        w[0] = 0.5 * (r[5] - r[7]);
        w[1] = 0.5 * (r[6] - r[2]);
        w[2] = 0.5 * (r[1] - r[3]);
    } else {
        double st = sin(theta);
        double coeff = theta / (2.0 * st);
        w[0] = coeff * (r[5] - r[7]);
        w[1] = coeff * (r[6] - r[2]);
        w[2] = coeff * (r[1] - r[3]);
    }
}

// ============================================================================
// SymPD batch geodesic distance (log-Euclidean approximation).
// Computes d(A_i, B_i) = || log(A_i) - log(B_i) ||_F   (approximation).
// Each matrix is NxN stored row-major.  The batch dimension is `batch`,
// and N is the matrix dimension.
// ============================================================================
__global__ void sympd_distance_batch(const double* __restrict__ A,
                                       const double* __restrict__ B,
                                       double* __restrict__ distances,
                                       int batch,
                                       int N) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    int stride = N * N;

    // Log-Euclidean distance approximation:
    // For SPD matrices A and B, compute ||log(A) - log(B)||_F
    // using the simple Frobenius norm of the difference of log-determinant
    // weighted eigenvalues.
    //
    // For a practical log-Euclidean approximation on the GPU (without full
    // eigendecomposition per element), we use:
    //   d(A,B) ~ sqrt( sum_i (log A_ii - log B_ii)^2 )  for diagonal-dominant
    // or more generally the Frobenius norm of (log A - log B) via element-wise
    // matrix logarithm approximation.
    //
    // Here we implement a simplified version: element-wise log followed by
    // Frobenius norm of the difference.
    const double* a = A + b * stride;
    const double* bp = B + b * stride;

    double sum_sq = 0.0;
    for (int k = 0; k < stride; ++k) {
        double la = (a[k] > 1e-15) ? log(a[k]) : log(1e-15);
        double lb = (bp[k] > 1e-15) ? log(bp[k]) : log(1e-15);
        double diff = la - lb;
        sum_sq += diff * diff;
    }

    distances[b] = sqrt(sum_sq);
}

// ============================================================================
// SE(3) batch composition:  T_out = T1 * T2  for B elements.
// Each transform is a 4x4 homogeneous matrix stored as 16 contiguous doubles
// in row-major order.
// ============================================================================
__global__ void se3_compose_batch(const double* __restrict__ T1,
                                   const double* __restrict__ T2,
                                   double* __restrict__ T_out,
                                   int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const double* a = T1 + b * 16;
    const double* b_mat = T2 + b * 16;
    double* c = T_out + b * 16;

    // Unrolled 4x4 matrix multiply (row-major)
    // Row 0
    c[0]  = a[0]*b_mat[0]  + a[1]*b_mat[4]  + a[2]*b_mat[8]  + a[3]*b_mat[12];
    c[1]  = a[0]*b_mat[1]  + a[1]*b_mat[5]  + a[2]*b_mat[9]  + a[3]*b_mat[13];
    c[2]  = a[0]*b_mat[2]  + a[1]*b_mat[6]  + a[2]*b_mat[10] + a[3]*b_mat[14];
    c[3]  = a[0]*b_mat[3]  + a[1]*b_mat[7]  + a[2]*b_mat[11] + a[3]*b_mat[15];

    // Row 1
    c[4]  = a[4]*b_mat[0]  + a[5]*b_mat[4]  + a[6]*b_mat[8]  + a[7]*b_mat[12];
    c[5]  = a[4]*b_mat[1]  + a[5]*b_mat[5]  + a[6]*b_mat[9]  + a[7]*b_mat[13];
    c[6]  = a[4]*b_mat[2]  + a[5]*b_mat[6]  + a[6]*b_mat[10] + a[7]*b_mat[14];
    c[7]  = a[4]*b_mat[3]  + a[5]*b_mat[7]  + a[6]*b_mat[11] + a[7]*b_mat[15];

    // Row 2
    c[8]  = a[8]*b_mat[0]  + a[9]*b_mat[4]  + a[10]*b_mat[8]  + a[11]*b_mat[12];
    c[9]  = a[8]*b_mat[1]  + a[9]*b_mat[5]  + a[10]*b_mat[9]  + a[11]*b_mat[13];
    c[10] = a[8]*b_mat[2]  + a[9]*b_mat[6]  + a[10]*b_mat[10] + a[11]*b_mat[14];
    c[11] = a[8]*b_mat[3]  + a[9]*b_mat[7]  + a[10]*b_mat[11] + a[11]*b_mat[15];

    // Row 3 (bottom row of homogeneous transform should be [0, 0, 0, 1])
    c[12] = a[12]*b_mat[0] + a[13]*b_mat[4] + a[14]*b_mat[8]  + a[15]*b_mat[12];
    c[13] = a[12]*b_mat[1] + a[13]*b_mat[5] + a[14]*b_mat[9]  + a[15]*b_mat[13];
    c[14] = a[12]*b_mat[2] + a[13]*b_mat[6] + a[14]*b_mat[10] + a[15]*b_mat[14];
    c[15] = a[12]*b_mat[3] + a[13]*b_mat[7] + a[14]*b_mat[11] + a[15]*b_mat[15];
}

} // namespace mg
