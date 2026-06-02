#include <torch/extension.h>
#include "matrix_group_cuda.cuh"
#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <vector>

#define CUDA_CHECK(cond)                                       \
    do {                                                       \
        cudaError_t err = (cond);                              \
        if (err != cudaSuccess) {                              \
            AT_ERROR("CUDA error: ", cudaGetErrorString(err)); \
        }                                                      \
    } while (0)

#define CHECK_CUDA(tensor)                                                 \
    do {                                                                   \
        AT_ASSERTM((tensor).is_cuda(), #tensor " must be a CUDA tensor"); \
    } while (0)

#define CHECK_CONTIGUOUS(tensor)                                               \
    do {                                                                       \
        AT_ASSERTM((tensor).is_contiguous(), #tensor " must be contiguous");   \
    } while (0)

#define CHECK_INPUT(tensor)     \
    do {                        \
        CHECK_CUDA(tensor);      \
        CHECK_CONTIGUOUS(tensor);\
    } while (0)

namespace mg {

// ============================================================================
// so3_exp:  omega [B, 3] -> R [B, 3, 3]
// ============================================================================
torch::Tensor so3_exp(torch::Tensor omega) {
    CHECK_INPUT(omega);
    AT_ASSERTM(omega.dim() == 2 && omega.size(1) == 3,
               "omega must have shape [B, 3]");

    int B = omega.size(0);
    auto R = torch::empty({B, 3, 3}, omega.options());

    const int block = 256;
    const int grid = (B + block - 1) / block;

    so3_exp_batch<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        omega.data_ptr<double>(),
        R.data_ptr<double>(),
        B
    );
    CUDA_CHECK(cudaGetLastError());

    return R;
}

// ============================================================================
// so3_log:  R [B, 3, 3] -> omega [B, 3]
// ============================================================================
torch::Tensor so3_log(torch::Tensor R) {
    CHECK_INPUT(R);
    AT_ASSERTM(R.dim() == 3 && R.size(1) == 3 && R.size(2) == 3,
               "R must have shape [B, 3, 3]");

    int B = R.size(0);
    auto omega = torch::empty({B, 3}, R.options());

    const int block = 256;
    const int grid = (B + block - 1) / block;

    so3_log_batch<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        R.data_ptr<double>(),
        omega.data_ptr<double>(),
        B
    );
    CUDA_CHECK(cudaGetLastError());

    return omega;
}

// ============================================================================
// so3_compose:  R1 [B,3,3], R2 [B,3,3] -> R_out [B,3,3]
// ============================================================================
torch::Tensor so3_compose(torch::Tensor R1, torch::Tensor R2) {
    CHECK_INPUT(R1);
    CHECK_INPUT(R2);
    AT_ASSERTM(R1.dim() == 3 && R1.size(1) == 3 && R1.size(2) == 3,
               "R1 must have shape [B, 3, 3]");
    AT_ASSERTM(R2.dim() == 3 && R2.size(1) == 3 && R2.size(2) == 3,
               "R2 must have shape [B, 3, 3]");
    AT_ASSERTM(R1.size(0) == R2.size(0),
               "R1 and R2 must have the same batch size");

    int B = R1.size(0);
    auto R_out = torch::empty({B, 3, 3}, R1.options());

    const int block = 256;
    const int grid = (B + block - 1) / block;

    so3_compose_batch<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        R1.data_ptr<double>(),
        R2.data_ptr<double>(),
        R_out.data_ptr<double>(),
        B
    );
    CUDA_CHECK(cudaGetLastError());

    return R_out;
}

// ============================================================================
// se3_exp:  xi [B, 6] -> T [B, 4, 4]
//
// Uses closed-form SE(3) exponential map.  The first 3 components of xi are
// the translational part v, the last 3 are the rotational axis-angle omega.
// The kernel is launched per-batch-element and performs the full computation
// on the GPU.
// ============================================================================
__global__ void se3_exp_kernel(const double* __restrict__ xi,
                                 double* __restrict__ T,
                                 int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const double* x = xi + b * 6;
    double* t = T + b * 16;

    double vx = x[0], vy = x[1], vz = x[2];
    double wx = x[3], wy = x[4], wz = x[5];
    double theta = sqrt(wx*wx + wy*wy + wz*wz);

    // SO(3) exponential via Rodrigues
    double R00, R01, R02, R10, R11, R12, R20, R21, R22;
    if (theta < 1e-6) {
        // Taylor: I + Omega + 0.5 Omega^2
        R00 = 1.0 - 0.5*(wy*wy+wz*wz); R01 = -wz + 0.5*wx*wy;       R02 =  wy + 0.5*wx*wz;
        R10 =  wz + 0.5*wx*wy;          R11 = 1.0 - 0.5*(wx*wx+wz*wz); R12 = -wx + 0.5*wy*wz;
        R20 = -wy + 0.5*wx*wz;          R21 =  wx + 0.5*wy*wz;        R22 = 1.0 - 0.5*(wx*wx+wy*wy);
    } else {
        double inv_theta = 1.0 / theta;
        double st = sin(theta);
        double ct = cos(theta);
        double c1 = st * inv_theta;
        double c2 = (1.0 - ct) * inv_theta * inv_theta;
        double wx2 = wx*wx, wy2 = wy*wy, wz2 = wz*wz;
        double wxwy = wx*wy, wxwz = wx*wz, wywz = wy*wz;

        R00 = 1.0 - c2*(wy2+wz2); R01 = -c1*wz - c2*wxwy; R02 =  c1*wy - c2*wxwz;
        R10 =  c1*wz - c2*wxwy;  R11 = 1.0 - c2*(wx2+wz2); R12 = -c1*wx - c2*wywz;
        R20 = -c1*wy - c2*wxwz;  R21 =  c1*wx - c2*wywz;   R22 = 1.0 - c2*(wx2+wy2);
    }

    // V matrix: t = V * v
    double tx, ty, tz;
    if (theta < 1e-6) {
        // V ~ I + Omega/2 + Omega^2/6
        tx = vx + 0.5*(-wz*vy + wy*vz) + (1.0/6.0)*(wy*(wy*vx-wx*vy) + wz*(wz*vx-wx*vz));
        ty = vy + 0.5*(wz*vx - wx*vz) + (1.0/6.0)*(wx*(wx*vy-wy*vx) + wz*(wz*vy-wy*vz));
        tz = vz + 0.5*(-wy*vx + wx*vy) + (1.0/6.0)*(wx*(wx*vz-wz*vx) + wy*(wy*vz-wz*vy));
    } else {
        double inv_theta2 = 1.0 / (theta * theta);
        double inv_theta3 = inv_theta2 / theta;
        double st = sin(theta);
        double ct = cos(theta);
        double a = (1.0 - ct) * inv_theta2;
        double b_coeff = (theta - st) * inv_theta3;

        // V = I + a * Omega + b * Omega^2
        double V00 = 1.0,              V01 = -a*wz,           V02 =  a*wy;
        double V10 =  a*wz,            V11 = 1.0,             V12 = -a*wx;
        double V20 = -a*wy,            V21 =  a*wx,           V22 = 1.0;

        // Omega^2 entries (for the b * Omega^2 part)
        V00 -= b_coeff*(wy*wy+wz*wz); V01 -= b_coeff*wxwy;    V02 -= b_coeff*wxwz;
        V10 -= b_coeff*wxwy;          V11 -= b_coeff*(wx*wx+wz*wz); V12 -= b_coeff*wywz;
        V20 -= b_coeff*wxwz;          V21 -= b_coeff*wywz;    V22 -= b_coeff*(wx*wx+wy*wy);

        tx = V00*vx + V01*vy + V02*vz;
        ty = V10*vx + V11*vy + V12*vz;
        tz = V20*vx + V21*vy + V22*vz;
    }

    // Fill 4x4 output (row-major)
    t[0]  = R00; t[1]  = R01; t[2]  = R02; t[3]  = tx;
    t[4]  = R10; t[5]  = R11; t[6]  = R12; t[7]  = ty;
    t[8]  = R20; t[9]  = R21; t[10] = R22; t[11] = tz;
    t[12] = 0.0;  t[13] = 0.0;  t[14] = 0.0;  t[15] = 1.0;
}

torch::Tensor se3_exp(torch::Tensor xi) {
    CHECK_INPUT(xi);
    AT_ASSERTM(xi.dim() == 2 && xi.size(1) == 6,
               "xi must have shape [B, 6]");

    int B = xi.size(0);
    auto T = torch::empty({B, 4, 4}, xi.options());

    const int block = 256;
    const int grid = (B + block - 1) / block;

    se3_exp_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        xi.data_ptr<double>(),
        T.data_ptr<double>(),
        B
    );
    CUDA_CHECK(cudaGetLastError());

    return T;
}

// ============================================================================
// se3_compose:  T1 [B,4,4], T2 [B,4,4] -> T_out [B,4,4]
// ============================================================================
torch::Tensor se3_compose(torch::Tensor T1, torch::Tensor T2) {
    CHECK_INPUT(T1);
    CHECK_INPUT(T2);
    AT_ASSERTM(T1.dim() == 3 && T1.size(1) == 4 && T1.size(2) == 4,
               "T1 must have shape [B, 4, 4]");
    AT_ASSERTM(T2.dim() == 3 && T2.size(1) == 4 && T2.size(2) == 4,
               "T2 must have shape [B, 4, 4]");
    AT_ASSERTM(T1.size(0) == T2.size(0),
               "T1 and T2 must have the same batch size");

    int B = T1.size(0);
    auto T_out = torch::empty({B, 4, 4}, T1.options());

    const int block = 256;
    const int grid = (B + block - 1) / block;

    se3_compose_batch<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        T1.data_ptr<double>(),
        T2.data_ptr<double>(),
        T_out.data_ptr<double>(),
        B
    );
    CUDA_CHECK(cudaGetLastError());

    return T_out;
}

} // namespace mg

// ============================================================================
// TORCH_LIBRARY macro for native function registration
// ============================================================================
TORCH_LIBRARY(matrix_groups, m) {
    m.def("so3_exp(Tensor omega) -> Tensor");
    m.def("so3_log(Tensor R) -> Tensor");
    m.def("so3_compose(Tensor R1, Tensor R2) -> Tensor");
    m.def("se3_exp(Tensor xi) -> Tensor");
    m.def("se3_compose(Tensor T1, Tensor T2) -> Tensor");
}

TORCH_LIBRARY_IMPL(matrix_groups, CUDA, m) {
    m.impl("so3_exp", TORCH_FN(mg::so3_exp));
    m.impl("so3_log", TORCH_FN(mg::so3_log));
    m.impl("so3_compose", TORCH_FN(mg::so3_compose));
    m.impl("se3_exp", TORCH_FN(mg::se3_exp));
    m.impl("se3_compose", TORCH_FN(mg::se3_compose));
}
