#ifndef COSMIC_DORMAND_PRINCE_CUDA_CUH
#define COSMIC_DORMAND_PRINCE_CUDA_CUH

#include <cuda_runtime.h>

#include <cmath>

// CUDA-optimized version of the Dormand-Prince integrator
class CudaDormandPrinceRK45 {
 public:
  __device__ CudaDormandPrinceRK45(int num_equations, float tolerance_abs,
                                   float tolerance_rel, float hmin, float hmax,
                                   float initial_step, int max_iterations)
      : N{num_equations},
        atol{tolerance_abs},
        rtol{tolerance_rel},
        hmin{hmin},
        hmax{hmax},
        initial_step{initial_step},
        max_iterations{max_iterations} {}

  __device__ __forceinline__ float max_val(float a, float b) {
    return a > b ? a : b;
  }

  __device__ __forceinline__ float clamp_val(float value, float min_val,
                                             float max_val) {
    return value < min_val ? min_val : (value > max_val ? max_val : value);
  }

  __device__ __forceinline__ float min_val(float a, float b) {
    return a < b ? a : b;
  }

  template <typename DerivativeFunc, typename HitCheckFunc,
            typename BoundaryCheckFunc>
  __device__ bool integrate(DerivativeFunc derivFunc, HitCheckFunc hitFunc,
                            BoundaryCheckFunc boundaryFunc, float* y, float* k1,
                            float* k2, float* k3, float* k4, float* k5,
                            float* k6, float* k7, float* y_err, float* y_next) {
    float h = initial_step;
    float err = 1.0f;
    float err_prev = 1.0f;

    // First same as last (FSAL) so k1 <- k7
    derivFunc(y, k1);

    int iter = 0;

    // while (!boundaryFunc(y)) {
    while (!boundaryFunc(y) && iter++ < max_iterations) {
      step(derivFunc, h, y, k1, k2, k3, k4, k5, k6, k7, y_err, y_next);
      err = max_val(error(y_next, y_err), hmin);

      if (err < 1.0f) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
          y[i] = y_next[i];
        }

        // Swap pointers - more efficient than copying data
        float* temp = k1;
        k1 = k7;
        k7 = temp;

        err_prev = err;

        // Early exit - reduces branch divergence by checking only after
        // accepted steps
        if (hitFunc(y)) {
          return true;
        }
      }

      // Step size adjustment using fast math
      // Move the branch outside to reduce divergence
      float h_factor;
      if (err_prev < 1.0f) {
        // PI controller step size adjustment
        h_factor = S * __powf(err, -alpha) * __powf(err_prev, beta);
      } else {
        // Simple step size reduction
        h_factor = S / __powf(err, 0.2f);
      }

      h = clamp_val(__fmul_rn(h, h_factor), hmin, hmax);
    }

    return false;
  }

  template <typename DerivativeFunction>
  __device__ __forceinline__ void step(DerivativeFunction& derivFunc, float h,
                                       const float* y, float* k1, float* k2,
                                       float* k3, float* k4, float* k5,
                                       float* k6, float* k7, float* y_err,
                                       float* y_next) {
// step 2
#pragma unroll
    for (int i = 0; i < N; i++) {
      y_next[i] = y[i] + __fmul_rn(h, __fmul_rn(a21, k1[i]));
    }
    derivFunc(y_next, k2);

// step 3
#pragma unroll
    for (int i = 0; i < N; i++) {
      y_next[i] =
          y[i] + __fmul_rn(h, (__fmul_rn(a31, k1[i]) + __fmul_rn(a32, k2[i])));
    }
    derivFunc(y_next, k3);

// step 4
#pragma unroll
    for (int i = 0; i < N; i++) {
      y_next[i] =
          y[i] + __fmul_rn(h, (__fmul_rn(a41, k1[i]) + __fmul_rn(a42, k2[i]) +
                               __fmul_rn(a43, k3[i])));
    }
    derivFunc(y_next, k4);

// step 5
#pragma unroll
    for (int i = 0; i < N; i++) {
      y_next[i] =
          y[i] + __fmul_rn(h, (__fmul_rn(a51, k1[i]) + __fmul_rn(a52, k2[i]) +
                               __fmul_rn(a53, k3[i]) + __fmul_rn(a54, k4[i])));
    }
    derivFunc(y_next, k5);

// step 6
#pragma unroll
    for (int i = 0; i < N; i++) {
      y_next[i] =
          y[i] + __fmul_rn(h, (__fmul_rn(a61, k1[i]) + __fmul_rn(a62, k2[i]) +
                               __fmul_rn(a63, k3[i]) + __fmul_rn(a64, k4[i]) +
                               __fmul_rn(a65, k5[i])));
    }
    derivFunc(y_next, k6);

// step 7 finally calculate y_next
#pragma unroll
    for (int i = 0; i < N; i++) {
      y_next[i] =
          y[i] + __fmul_rn(h, (__fmul_rn(a71, k1[i]) + __fmul_rn(a73, k3[i]) +
                               __fmul_rn(a74, k4[i]) + __fmul_rn(a75, k5[i]) +
                               __fmul_rn(a76, k6[i])));
    }
    // Then use y_next to calculate k7
    derivFunc(y_next, k7);

// find the error
#pragma unroll
    for (int i = 0; i < N; i++) {
      y_err[i] = __fmul_rn(h, (__fmul_rn(e1, k1[i]) + __fmul_rn(e3, k3[i]) +
                               __fmul_rn(e4, k4[i]) + __fmul_rn(e5, k5[i]) +
                               __fmul_rn(e6, k6[i]) + __fmul_rn(e7, k7[i])));
    }
  }

  __device__ __forceinline__ float error(const float* y_next,
                                         const float* y_err) {
    float err_sum = 0.0f;

#pragma unroll
    for (int i = 0; i < N; i++) {
      float sc =
          atol + __fmul_rn(max_val(fabsf(y_next[i]), fabsf(y_next[i])), rtol);
      float e = __fdividef(y_err[i], sc);
      err_sum += __fmul_rn(e, e);
    }

    return __fsqrt_rn(__fdividef(err_sum, N));
  }

 private:
  // initial stuff
  const int N;
  const float atol, rtol;

  // clamp values
  const float hmin;
  const float hmax;
  const float initial_step;
  const int max_iterations;

  // Adaptive step size constants
  static constexpr float S = 0.9f;             // Safety factor
  static constexpr float alpha = 0.7f / 5.0f;  // PI controller parameters
  static constexpr float beta = 0.4f / 5.0f;

  // Butcher tableau constants
  static constexpr float c2 = 1.0f / 5.0f;
  static constexpr float c3 = 3.0f / 10.0f;
  static constexpr float c4 = 4.0f / 5.0f;
  static constexpr float c5 = 8.0f / 9.0f;

  static constexpr float a21 = 1.0f / 5.0f;
  static constexpr float a31 = 3.0f / 40.0f;
  static constexpr float a32 = 9.0f / 40.0f;
  static constexpr float a41 = 44.0f / 45.0f;
  static constexpr float a42 = -56.0f / 15.0f;
  static constexpr float a43 = 32.0f / 9.0f;
  static constexpr float a51 = 19372.0f / 6561.0f;
  static constexpr float a52 = -25360.0f / 2187.0f;
  static constexpr float a53 = 64448.0f / 6561.0f;
  static constexpr float a54 = -212.0f / 729.0f;
  static constexpr float a61 = 9017.0f / 3168.0f;
  static constexpr float a62 = -355.0f / 33.0f;
  static constexpr float a63 = 46732.0f / 5247.0f;
  static constexpr float a64 = 49.0f / 176.0f;
  static constexpr float a65 = -5103.0f / 18656.0f;

  static constexpr float a71 = 35.0f / 384.0f;
  static constexpr float a72 = 0.0f;
  static constexpr float a73 = 500.0f / 1113.0f;
  static constexpr float a74 = 125.0f / 192.0f;
  static constexpr float a75 = -2187.0f / 6784.0f;
  static constexpr float a76 = 11.0f / 84.0f;

  static constexpr float e1 = 71.0f / 57600.0f;
  static constexpr float e2 = 0.0f;
  static constexpr float e3 = -71.0f / 16695.0f;
  static constexpr float e4 = 71.0f / 1920.0f;
  static constexpr float e5 = -17253.0f / 339200.0f;
  static constexpr float e6 = 22.0f / 525.0f;
  static constexpr float e7 = -1.0f / 40.0f;
};

#endif  // COSMIC_DORMAND_PRINCE_CUDA_CUH
