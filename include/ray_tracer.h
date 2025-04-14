#ifndef COSMIC_RAY_TRACER_H
#define COSMIC_RAY_TRACER_H

#include "metric.h"
#include "dormand_prince.h"
#include "constants.h"
#include "config.h"
#include <cstring>


// ===== FUNCTORS =====
struct MetricDerivativeFunctor {
    BoyerLindquistMetric& metric;
    
    MetricDerivativeFunctor(BoyerLindquistMetric& m) : metric(m) {}
    
    inline void operator()(float* y, float* k) const noexcept{
        metric.computeDerivatives(y, k);
    }
};
 
struct AccretionDiskHitFunctor {
    const float innerRadius;
    const float outerRadius;
    const float tolerance;
    
    AccretionDiskHitFunctor(float inner, float outer, float tol)
        : innerRadius(inner), outerRadius(outer), tolerance(tol) {}
    
    inline bool operator()(const float* y) const noexcept{
        return ((y[0] >= innerRadius) & (y[0] <= outerRadius)) & 
               (std::fabs(y[1] - Constants::HALF_PI) < tolerance);
    }
};

struct RayBoundaryCheckFunctor {
    const float diskTolerance;
    const float farRadius;
    
    RayBoundaryCheckFunctor(float dt, float fr)
        : diskTolerance(dt), farRadius(fr) {}
    
    inline bool operator()(const float* __restrict__ y) const noexcept{
        return (y[0] < diskTolerance) | (y[0] > farRadius);
    }
};

// ===== MAIN RAY TRACER CLASS =====
class RayTracer {
public:
    // Updated constructor that doesn't initialize a separate cameraParams_ member
    RayTracer(const Config::BlackHole& bhConfig, const Config::Image& imgConfig)
        : bhConfig_(bhConfig)
        , imgConfig_(imgConfig)
    {}

    // Traces ray from the observer at pixel (i, j)
    // Returns true if the ray hits the disk
    // and false if it never hits the disk
    bool traceRay(int i, int j, BoyerLindquistMetric& metric, DormandPrinceRK45& integrator,
                  float* y, float& intensity) {
        // Setup ray initial conditions
        setupRayInitialConditions(i, j, y, metric);
        
        MetricDerivativeFunctor metricDerivFunctor(metric);
        AccretionDiskHitFunctor diskHitFunctor(
            bhConfig_.innerRadius, bhConfig_.outerRadius, 
            Constants::Integration::DISK_TOLERANCE);
        RayBoundaryCheckFunctor boundaryCheckFunctor(
            bhConfig_.diskTolerance, bhConfig_.farRadius);
            
        // Direct boolean return value
        const bool hit = integrator.integrate(
            metricDerivFunctor,
            diskHitFunctor,
            boundaryCheckFunctor,
            y
        );
    
        if (hit) {
            calculateIntensity(y, metric, intensity);
        }
        return hit;
    }

private:
    //  Setup initial conditions for the ray for each CPU thread 
    //  i: Pixel x-coordinate starting at the left
    //  j: Pixel y-coordinate starting at the top
    //  ray: Initializes photon position and momentum in spherical coordinates
    void setupRayInitialConditions(int i, int j, float* ray, BoyerLindquistMetric& metric) {
        // Use camera parameters directly from imgConfig_
        const float local_x_sc = imgConfig_.offsetX + (i * imgConfig_.stepX);
        const float local_y_sc = imgConfig_.offsetY - (j * imgConfig_.stepY);

        // Pre-calculate frequently used values
        const float D_squared = bhConfig_.distance * bhConfig_.distance;
        const float beta = local_x_sc / bhConfig_.distance;
        const float alpha = local_y_sc / bhConfig_.distance;
        const float cos_beta = std::cosf(beta);
        const float sin_beta = std::sinf(beta);
        const float cos_alpha = std::cosf(alpha);
        const float sin_alpha = std::sinf(alpha);
        
        // Set initial position
        ray[0] = std::sqrtf(D_squared + (local_x_sc * local_x_sc) + (local_y_sc * local_y_sc));
        ray[1] = bhConfig_.theta - alpha;
        ray[2] = beta;
        
        // Compute metric at initial position
        metric.computeMetric(ray[0], ray[1]);

        // Set initial momentum
        ray[3] = -std::sqrtf(metric.g_11) * cos_beta * cos_alpha;
        ray[4] = -std::sqrtf(metric.g_22) * sin_alpha;
        ray[5] = std::sqrtf(metric.g_33) * sin_beta * cos_alpha;
    }

    // Calculate intensity of a ray that hit the disk
    void calculateIntensity(const float* ray, const BoyerLindquistMetric& metric, float& intensity) {
        // Rest of the method remains unchanged
        const float rf = ray[0];
        const float u_rf = -ray[3];
        const float u_thf = -ray[4];
        const float u_phif = -ray[5];

        // Calculate upper time component of 4-velocity
        const float u_uppertf = std::sqrtf((metric.gamma11 * u_rf * u_rf) +
                            (metric.gamma22 * u_thf * u_thf) +
                            (metric.gamma33 * u_phif * u_phif)) /
                        metric.alpha;
                        
        // Calculate lower time component
        const float u_lower_tf =
            (-metric.alpha * metric.alpha * u_uppertf) +
            (u_phif * metric.beta3);
        
        // Calculate angular velocity of matter in accretion disk
        const float omega = 1.0 / (bhConfig_.spin + 
                                (std::powf(rf, 3.0 / 2.0) / std::sqrtf(bhConfig_.mass)));
    
        // Calculate redshift factor                
        const float oneplusz = (1.0 + (omega * u_phif / u_lower_tf))
                / std::sqrtf(-metric.g_00 - (omega * omega * metric.g_33) -
                (2 * omega * metric.g_03));

        // Lorentz invariant intensity I_em / (1 + z)^3
        intensity = 1.0 / (oneplusz * oneplusz * oneplusz);
    }

    // Store the configuration objects
    Config::BlackHole bhConfig_;
    Config::Image imgConfig_;
};

#endif // COSMIC_RAY_TRACER_H
