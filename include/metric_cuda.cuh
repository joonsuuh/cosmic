#ifndef COSMIC_METRIC_CUDA_CUH
#define COSMIC_METRIC_CUDA_CUH

#include <cuda_runtime.h>

#include <cmath>

class CudaBoyerLindquistMetric {
 public:
  // Black hole parameters
  const float a;  // Spin
  const float M;  // Mass

  // Metric components
  float alpha, beta3;
  float gamma11, gamma22, gamma33;     // components of upper gamma^ij
  float g_00, g_03, g_11, g_22, g_33;  // components of lower g_\mu\nu

  // Metric derivatives
  float d_alpha_dr, d_beta3_dr, d_gamma11_dr, d_gamma22_dr, d_gamma33_dr;
  float d_alpha_dth, d_beta3_dth, d_gamma11_dth, d_gamma22_dth, d_gamma33_dth;

  __device__ CudaBoyerLindquistMetric(float spin, float mass)
      : a(spin), M(mass) {}

  __device__ __forceinline__ void computeMetric(float r, float theta) {
    const float r2 = __fmul_rn(r, r);
    const float a2 = __fmul_rn(a, a);
    const float sintheta = __sinf(theta);
    const float costheta = __cosf(theta);
    const float sin2 = __fmul_rn(sintheta, sintheta);
    const float cos2 = __fmul_rn(costheta, costheta);
    const float p2 = r2 + __fmul_rn(a2, cos2);
    const float delta = r2 + a2 - __fmul_rn(2.0f, __fmul_rn(M, r));
    const float sigma =
        __fmul_rn((r2 + a2), (r2 + a2)) - __fmul_rn(a2, __fmul_rn(delta, sin2));

    computeMetricComponents(r, r2, a2, sin2, cos2, p2, delta, sigma);
    computeMetricDerivatives(r, r2, a2, sin2, cos2, p2, delta, sigma, theta);
  }

  __device__ __forceinline__ void computeDerivatives(float* y, float* k) {
    computeMetric(y[0], y[1]);

    const float u_uppert =
        __fsqrt_rn(__fmul_rn(gamma11, __fmul_rn(y[3], y[3])) +
                   __fmul_rn(gamma22, __fmul_rn(y[4], y[4])) +
                   __fmul_rn(gamma33, __fmul_rn(y[5], y[5]))) /
        alpha;

    k[0] = __fdividef(__fmul_rn(gamma11, y[3]), u_uppert);
    k[1] = __fdividef(__fmul_rn(gamma22, y[4]), u_uppert);
    k[2] = __fdividef(__fmul_rn(gamma33, y[5]), u_uppert) - beta3;

    const float temp1 = __fmul_rn(y[3], __fmul_rn(y[3], d_gamma11_dr)) +
                        __fmul_rn(y[4], __fmul_rn(y[4], d_gamma22_dr)) +
                        __fmul_rn(y[5], __fmul_rn(y[5], d_gamma33_dr));

    k[3] = -__fmul_rn(alpha, __fmul_rn(u_uppert, d_alpha_dr)) +
           __fmul_rn(y[5], d_beta3_dr) -
           __fdividef(temp1, __fmul_rn(2.0f, u_uppert));

    const float temp2 = __fmul_rn(y[3], __fmul_rn(y[3], d_gamma11_dth)) +
                        __fmul_rn(y[4], __fmul_rn(y[4], d_gamma22_dth)) +
                        __fmul_rn(y[5], __fmul_rn(y[5], d_gamma33_dth));

    k[4] = -__fmul_rn(alpha, __fmul_rn(u_uppert, d_alpha_dth)) +
           __fmul_rn(y[5], d_beta3_dth) -
           __fdividef(temp2, __fmul_rn(2.0f, u_uppert));

    k[5] = 0.0f;
  }

 private:
  __device__ __forceinline__ void computeMetricComponents(float r, float r2,
                                                          float a2, float sin2,
                                                          float cos2, float p2,
                                                          float delta,
                                                          float SUM) {
    alpha = __fsqrt_rn(__fdividef(__fmul_rn(p2, delta), SUM));
    beta3 = __fdividef(-__fmul_rn(2.0f, __fmul_rn(M, __fmul_rn(a, r))), SUM);
    gamma11 = __fdividef(delta, p2);
    gamma22 = __fdividef(1.0f, p2);
    gamma33 = __fdividef(p2, __fmul_rn(SUM, sin2));
    g_00 = __fdividef(__fmul_rn(2.0f, __fmul_rn(M, r)), p2) - 1.0f;
    g_03 =
        __fdividef(-__fmul_rn(2.0f, __fmul_rn(M, __fmul_rn(a, r))), p2) * sin2;
    g_11 = __fdividef(p2, delta);
    g_22 = p2;
    g_33 = __fdividef(SUM, p2) * sin2;
  }

  __device__ inline void computeMetricDerivatives(float r, float r2, float a2,
                                                  float sin2, float cos2,
                                                  float p2, float delta,
                                                  float sigma, float theta) {
    // Derivatives w.r.t r
    const float dp2_dr = 2.0f * r;
    const float dp2inv_dr = -2.0f * r / p2 / p2;
    const float ddelta_dr = dp2_dr - (2.0f * M);
    const float dSUMinv_dr =
        ((4.0f * r * (r2 + a2)) - (ddelta_dr * a2 * sin2)) / -sigma / sigma;

    d_gamma11_dr = (delta * dp2inv_dr) + (ddelta_dr / p2);
    d_gamma22_dr = dp2inv_dr;
    d_gamma33_dr = (dp2_dr / (sigma * sin2)) + (dSUMinv_dr * p2 / sin2);
    d_alpha_dr =
        (0.5f / alpha) * ((dp2_dr * delta / sigma) + (ddelta_dr * p2 / sigma) +
                          (dSUMinv_dr * p2 * delta));
    d_beta3_dr = -2.0f * M * a * (1.0f / sigma + r * dSUMinv_dr);

    // Derivatives w.r.t theta
    const float dp2_dth = -2.0f * a2 * sinf(theta) * cosf(theta);
    const float dp2inv_dth = -dp2_dth / p2 / p2;
    // const float ddelta_dth = 0;
    const float dSUMinv_dth =
        2.0f * a2 * delta * sinf(theta) * cosf(theta) / sigma / sigma;

    d_gamma11_dth = delta * dp2inv_dth;
    d_gamma22_dth = dp2inv_dth;
    d_gamma33_dth = (dp2_dth / sigma / sin2) + (dSUMinv_dth * p2 / sin2) -
                    (2.0f * cosf(theta) / sinf(theta) / sin2 * p2 / sigma);
    d_alpha_dth = 0.5f / alpha * delta * (dp2_dth / sigma + dSUMinv_dth * p2);
    d_beta3_dth = -2.0f * M * a * r * dSUMinv_dth;
  }
};

#endif  // COSMIC_METRIC_CUDA_CUH
