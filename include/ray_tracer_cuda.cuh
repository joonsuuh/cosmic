#ifndef COSMIC_RAY_TRACER_CUDA_CUH
#define COSMIC_RAY_TRACER_CUDA_CUH

#include <cuda_runtime.h>

// #include "config.h"
#include "dormand_prince_cuda.cuh"
#include "metric_cuda.cuh"

#define PI_2 1.57079632679489661923f

__constant__ float c_bhParams_data[9];
__constant__ float c_imgParams_data[4];
__constant__ float c_camParams_data[4];
__constant__ float c_integrationConstants[6];

// ===== CUDA FUNCTIONS =====
__device__ __forceinline__ void computeMetricDerivatives(
    CudaBoyerLindquistMetric& metric, float* y, float* k) {
  metric.computeDerivatives(y, k);
}

__device__ __forceinline__ bool checkAccretionDiskHit(const float* y,
                                                      float innerRadius,
                                                      float outerRadius,
                                                      float tolerance) {
  return ((y[0] >= innerRadius) & (y[0] <= outerRadius)) &
         (fabs(y[1] - PI_2) < tolerance);
}

__device__ __forceinline__ bool checkRayBoundary(const float* y,
                                                 float diskTolerance,
                                                 float farRadius) {
  return (y[0] < diskTolerance) | (y[0] > farRadius);
}

// Helper device functions to access the parameters
__device__ __forceinline__ float getBhSpin() { return c_bhParams_data[0]; }
__device__ __forceinline__ float getBhMass() { return c_bhParams_data[1]; }
__device__ __forceinline__ float getBhDistance() { return c_bhParams_data[2]; }
__device__ __forceinline__ float getBhTheta() { return c_bhParams_data[3]; }
__device__ __forceinline__ float getBhPhi() { return c_bhParams_data[4]; }
__device__ __forceinline__ float getBhInnerRadius() {
  return c_bhParams_data[5];
}
__device__ __forceinline__ float getBhOuterRadius() {
  return c_bhParams_data[6];
}
__device__ __forceinline__ float getBhDiskTolerance() {
  return c_bhParams_data[7];
}
__device__ __forceinline__ float getBhFarRadius() { return c_bhParams_data[8]; }

__device__ __forceinline__ int getImgAspectWidth() {
  return c_imgParams_data[0];
}
__device__ __forceinline__ int getImgAspectHeight() {
  return c_imgParams_data[1];
}
__device__ __forceinline__ int getImgScale() { return c_imgParams_data[2]; }
__device__ __forceinline__ float getImgCameraScale() {
  return c_imgParams_data[3];
}

__device__ __forceinline__ float getCamOffsetX() { return c_camParams_data[0]; }
__device__ __forceinline__ float getCamOffsetY() { return c_camParams_data[1]; }
__device__ __forceinline__ float getCamStepX() { return c_camParams_data[2]; }
__device__ __forceinline__ float getCamStepY() { return c_camParams_data[3]; }

// ===== CUDA KERNELS FOR RAY TRACING =====
__device__ __forceinline__ float applyPerlinNoise(
    float r, float phi, const float* noiseMap, float innerRadius,
    float outerRadius, int noiseSize, float omega, float time = 0.0f) {
  // Normalize radius to map to noise texture row
  const int nr = __fmul_rn((r - innerRadius) / (outerRadius - innerRadius),
                           (noiseSize - 1));

  // Apply time-dependent rotation: phi = phi + omega * t
  const float rotated_phi = phi + omega * time;

  // Map phi to texture column coordinates
  const int nphi =
      __fmul_rn((__sinf(rotated_phi) + 1.0f) * 0.5f, (noiseSize - 1));

  const int index =
      min(max(nr * noiseSize + nphi, 0), noiseSize * noiseSize - 1);

  return noiseMap[index];
}

__device__ __forceinline__ float calculateRadialFalloff(float r,
                                                        float innerRadius,
                                                        float outerRadius) {
  // Define the radius at which the falloff starts (halfway between inner and
  // outer)
  const float midRadius = innerRadius + (outerRadius - innerRadius) * 0.5f;

  // Use branch-free approach with fminf/fmaxf instead of if-else
  const float falloffFactor =
      1.0f - ((r - midRadius) / (outerRadius - midRadius));
  return fmaxf(0.0f,
               fminf(1.0f, falloffFactor * (r <= midRadius ? 1.0f : 1.0f)));
}

__global__ void rayTraceKernel(float* screenBuffer, const float* noiseMap,
                               const int noiseSize, const float time = 0.0f) {
  // Use x as the fastest-changing dimension for better memory coalescing
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  const int pixelWidth = getImgAspectWidth() * getImgScale();
  const int pixelHeight = getImgAspectHeight() * getImgScale();

  if (i >= pixelWidth || j >= pixelHeight) return;

  const int idx = j * pixelWidth + i;

  float y[6] = {0};
  float k1[6] = {0};
  float k2[6] = {0};
  float k3[6] = {0};
  float k4[6] = {0};
  float k5[6] = {0};
  float k6[6] = {0};
  float k7[6] = {0};
  float y_err[6] = {0};
  float y_next[6] = {0};

  CudaBoyerLindquistMetric metric(getBhSpin(), getBhMass());

  // Precompute constants outside the loops
  const float offset_x = getCamOffsetX();
  const float offset_y = getCamOffsetY();
  const float step_x = getCamStepX();
  const float step_y = getCamStepY();

  const float local_x_sc = offset_x + (__fmul_rn(i, step_x));
  const float local_y_sc = offset_y - (__fmul_rn(j, step_y));

  const float bhDistance = getBhDistance();
  const float D_squared = __fmul_rn(bhDistance, bhDistance);
  const float beta = __fdividef(local_x_sc, bhDistance);
  const float alpha = __fdividef(local_y_sc, bhDistance);
  const float cos_beta = __cosf(beta);
  const float sin_beta = __sinf(beta);
  const float cos_alpha = __cosf(alpha);
  const float sin_alpha = __sinf(alpha);

  // init position
  y[0] = __fsqrt_rn(D_squared + __fmul_rn(local_x_sc, local_x_sc) +
                    __fmul_rn(local_y_sc, local_y_sc));
  y[1] = getBhTheta() - alpha;
  y[2] = beta;

  // compute metric at init position
  metric.computeMetric(y[0], y[1]);

  // init momentum
  y[3] = -__fsqrt_rn(metric.g_11) * cos_beta * cos_alpha;
  y[4] = -__fsqrt_rn(metric.g_22) * sin_alpha;
  y[5] = __fsqrt_rn(metric.g_33) * sin_beta * cos_alpha;

  const float innerRadius = getBhInnerRadius();
  const float outerRadius = getBhOuterRadius();
  const float diskTolerance = getBhDiskTolerance();
  const float farRadius = getBhFarRadius();

  // Create integration constants
  CudaDormandPrinceRK45 integrator(
      6, c_integrationConstants[0], c_integrationConstants[1],
      c_integrationConstants[2], c_integrationConstants[3],
      c_integrationConstants[4], 7'000);

  float intensity = 0.0f;

  // using lambda expressions...
  bool hit = integrator.integrate(
      [&metric](float* y, float* k) { computeMetricDerivatives(metric, y, k); },
      [innerRadius, outerRadius,
       tolerance = c_integrationConstants[5]](const float* y) {
        return checkAccretionDiskHit(y, innerRadius, outerRadius, tolerance);
      },
      [diskTolerance, farRadius](const float* y) {
        return checkRayBoundary(y, diskTolerance, farRadius);
      },
      y, k1, k2, k3, k4, k5, k6, k7, y_err, y_next);

  if (hit) {
    const float rf = y[0];
    const float u_rf = -y[3];
    const float u_thf = -y[4];
    const float u_phif = -y[5];

    // upper time component of 4-velocity
    const float u_uppertf =
        __fsqrt_rn(__fmul_rn(metric.gamma11, __fmul_rn(u_rf, u_rf)) +
                   __fmul_rn(metric.gamma22, __fmul_rn(u_thf, u_thf)) +
                   __fmul_rn(metric.gamma33, __fmul_rn(u_phif, u_phif))) /
        metric.alpha;

    // lower time component
    const float u_lower_tf =
        (-__fmul_rn(metric.alpha, metric.alpha) * u_uppertf) +
        (__fmul_rn(u_phif, metric.beta3));

    // angular velocity in accretion disk
    const float mass = getBhMass();
    const float spin = getBhSpin();
    const float rf_pow = __powf(rf, 1.5f);
    const float omega =
        __fdividef(1.0f, (spin + __fdividef(rf_pow, __fsqrt_rn(mass))));

    // redshift factor
    const float omega_sq = __fmul_rn(omega, omega);
    const float oneplusz =
        __fdividef((1.0f + __fmul_rn(omega, __fdividef(u_phif, u_lower_tf))),
                   __fsqrt_rn(-metric.g_00 - __fmul_rn(omega_sq, metric.g_33) -
                              __fmul_rn(2.0f, __fmul_rn(omega, metric.g_03))));

    // post-process for cooler image
    float noise = applyPerlinNoise(rf, y[2], noiseMap, innerRadius, outerRadius,
                                   noiseSize, omega, time);
    float falloff = calculateRadialFalloff(rf, innerRadius, outerRadius);

    // Lorentz invariant intensity calculation
    const float oneplusz_cubed =
        __fmul_rn(oneplusz, __fmul_rn(oneplusz, oneplusz));
    intensity = __fdividef(1.0f, oneplusz_cubed) *
                (0.5f + __fmul_rn(0.5f, noise)) * falloff;
  }
  screenBuffer[idx] = intensity;
}
#endif  // COSMIC_RAY_TRACER_CUDA_CUH
