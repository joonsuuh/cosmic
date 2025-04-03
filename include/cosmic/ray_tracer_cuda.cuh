#ifndef COSMIC_RAY_TRACER_CUDA_CUH
#define COSMIC_RAY_TRACER_CUDA_CUH

#include <cuda_runtime.h>
#include "metric_cuda.cuh"
#include "dormand_prince_cuda.cuh"
#include "config.h"
#include "perlin.h"

// ===== CUDA FUNCTORS =====
struct CudaMetricDerivativeFunctor {
    CudaBoyerLindquistMetric& metric;
    
    __device__ CudaMetricDerivativeFunctor(CudaBoyerLindquistMetric& m) : metric(m) {}
    
    __device__ inline void operator()(float* y, float* k) const {
        metric.computeDerivatives(y, k);
    }
};
 
struct CudaAccretionDiskHitFunctor {
    const float innerRadius;
    const float outerRadius;
    const float tolerance;
    
    __device__ CudaAccretionDiskHitFunctor(float inner, float outer, float tol)
        : innerRadius(inner), outerRadius(outer), tolerance(tol) {}
    
    __device__ inline bool operator()(const float* y) const {
        return ((y[0] >= innerRadius) & (y[0] <= outerRadius)) & 
               (fabs(y[1] - Constants::HALF_PI) < tolerance);
    }
};

struct CudaRayBoundaryCheckFunctor {
    const float diskTolerance;
    const float farRadius;
    
    __device__ CudaRayBoundaryCheckFunctor(float dt, float fr)
        : diskTolerance(dt), farRadius(fr) {}
    
    __device__ inline bool operator()(const float* y) const {
        return (y[0] < diskTolerance) | (y[0] > farRadius);
    }
};

// ===== CUDA KERNELS FOR RAY TRACING =====

__device__ float applyPerlinNoise(float r, float phi, const float* noiseMap, float innerRadius, float outerRadius, int noiseSize) {
    // Normalize radius and phi to map to noise texture coordinates
    int nr = ((r - innerRadius) / (outerRadius - innerRadius) * (noiseSize - 1));
    // float t = step * dt;
    // phi -= Omega * t;
    int nphi = (sinf(phi) + 1.0f) * 0.5f * (noiseSize - 1);
    
    // Correctly compute the 2D index into the 1D array - using row-major ordering
    int index = nphi * noiseSize + nr;
    
    return noiseMap[index];
}

__device__ float calculateRadialFalloff(float r, float innerRadius, float outerRadius) {
    // Define the radius at which the falloff starts (halfway between inner and outer)
    float midRadius = innerRadius + (outerRadius - innerRadius) * 0.5f;
    
    // If we're inside the midpoint, return full intensity (1.0)
    if (r <= midRadius) {
        return 1.0f;
    }
    
    // Calculate falloff from midpoint to outer radius
    // This creates a smooth transition from 1.0 at midpoint to 0.0 at outer radius
    float falloffFactor = 1.0f - ((r - midRadius) / (outerRadius - midRadius));
    
    // Ensure the value is between 0 and 1
    return max(0.0f, min(1.0f, falloffFactor));
}

// Main ray tracing kernel
__global__ void rayTraceKernel(float* screenBuffer, 
                             const Config::BlackHole bhParams,
                             const Config::Image imgParams,
                             const Config::Image::CameraParams camParams,
                             const float* noiseMap,
                             const int noiseSize) {
    // Calculate thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate pixel dimensions directly
    const int pixelWidth = imgParams.aspectWidth * imgParams.scale;
    const int pixelHeight = imgParams.aspectHeight * imgParams.scale;
    
    // Bounds check
    if (i >= pixelWidth || j >= pixelHeight) return;

    // Initialize memory for the integrator
    // Use thread-local arrays - these will be in registers for most cases
    float y[6] = {0};  // Ray state vector
    float k1[6] = {0}; // Integrator temp arrays
    float k2[6] = {0};
    float k3[6] = {0};
    float k4[6] = {0};
    float k5[6] = {0};
    float k6[6] = {0};
    float k7[6] = {0};
    float y_err[6] = {0};
    float y_next[6] = {0};
    
    // Create metric object
    CudaBoyerLindquistMetric metric(bhParams.spin, bhParams.mass);
    
    // Setup ray initial conditions
    const float local_x_sc = camParams.offsetX + (i * camParams.stepX);
    const float local_y_sc = camParams.offsetY - (j * camParams.stepY);

    // Pre-calculate frequently used values
    const float D_squared = bhParams.distance * bhParams.distance;
    const float beta = local_x_sc / bhParams.distance;
    const float alpha = local_y_sc / bhParams.distance;
    const float cos_beta = cosf(beta);
    const float sin_beta = sinf(beta);
    const float cos_alpha = cosf(alpha);
    const float sin_alpha = sinf(alpha);
    
    // Set initial position
    y[0] = sqrtf(D_squared + (local_x_sc * local_x_sc) + (local_y_sc * local_y_sc));
    y[1] = bhParams.theta - alpha;
    y[2] = beta;
    
    // Compute metric at initial position
    metric.computeMetric(y[0], y[1]);

    // Set initial momentum
    y[3] = -sqrtf(metric.g_11) * cos_beta * cos_alpha;
    y[4] = -sqrtf(metric.g_22) * sin_alpha;
    y[5] = sqrtf(metric.g_33) * sin_beta * cos_alpha;
    
    // Create functors for the integrator
    CudaMetricDerivativeFunctor metricDerivFunctor(metric);
    CudaAccretionDiskHitFunctor diskHitFunctor(
        bhParams.innerRadius, bhParams.outerRadius, 
        Constants::Integration::DISK_TOLERANCE);
    CudaRayBoundaryCheckFunctor boundaryCheckFunctor(
        bhParams.diskTolerance, bhParams.farRadius);
    
    // Create integrator
    CudaDormandPrinceRK45 integrator(6, Constants::Integration::ABS_TOLERANCE, 
                                    Constants::Integration::REL_TOLERANCE,
                                    Constants::Integration::MIN_STEP_SIZE,
                                    Constants::Integration::MAX_STEP_SIZE,
                                    Constants::Integration::INITIAL_STEP_SIZE,
                                    Constants::Integration::MAX_ITERATIONS);
    
    // Trace the ray
    float intensity = 0.0f;
    bool hit = integrator.integrate(
        metricDerivFunctor,
        diskHitFunctor,
        boundaryCheckFunctor,
        y, k1, k2, k3, k4, k5, k6, k7, y_err, y_next
    );
    
    if (hit) {
        // Calculate intensity for the hit
        const float rf = y[0];
        const float u_rf = -y[3];
        const float u_thf = -y[4];
        const float u_phif = -y[5];

        // Calculate upper time component of 4-velocity
        const float u_uppertf = sqrtf((metric.gamma11 * u_rf * u_rf) +
                                    (metric.gamma22 * u_thf * u_thf) +
                                    (metric.gamma33 * u_phif * u_phif)) /
                                metric.alpha;
                                
        // Calculate lower time component
        const float u_lower_tf =
            (-metric.alpha * metric.alpha * u_uppertf) +
            (u_phif * metric.beta3);
        
        // Calculate angular velocity of matter in accretion disk
        const float omega = 1.0f / (bhParams.spin + 
                                 (powf(rf, 3.0f / 2.0f) / sqrtf(bhParams.mass)));
    
        // Calculate redshift factor                
        const float oneplusz = (1.0f + (omega * u_phif / u_lower_tf))
                             / sqrtf(-metric.g_00 - (omega * omega * metric.g_33) -
                                   (2 * omega * metric.g_03));

        // Apply perlin noise to modify the intensity
        float noise = applyPerlinNoise(rf, y[2], noiseMap, bhParams.innerRadius, bhParams.outerRadius, noiseSize);
        
        // Apply radial falloff to create a disk that fades toward the outer edge
        float falloff = calculateRadialFalloff(rf, bhParams.innerRadius, bhParams.outerRadius);
        
        // Lorentz invariant intensity I_em / (1 + z)^3 with noise and falloff
        intensity = (1.0f / (oneplusz * oneplusz * oneplusz)) * (0.5f + 0.5f * noise) * falloff;
    }
    
    // Write intensity to the screen buffer
    screenBuffer[j * pixelWidth + i] = intensity;
}

#endif // COSMIC_RAY_TRACER_CUDA_CUH
