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
    
    inline void operator()(double* y, double* k) const noexcept{
        metric.computeDerivatives(y, k);
    }
};
 
struct AccretionDiskHitFunctor {
    const double innerRadius;
    const double outerRadius;
    const double tolerance;
    
    AccretionDiskHitFunctor(double inner, double outer, double tol)
        : innerRadius(inner), outerRadius(outer), tolerance(tol) {}
    
    inline bool operator()(const double* y) const noexcept{
        return ((y[0] >= innerRadius) & (y[0] <= outerRadius)) & 
               (std::abs(y[1] - Constants::HALF_PI) < tolerance);
    }
};

struct RayBoundaryCheckFunctor {
    const double diskTolerance;
    const double farRadius;
    
    RayBoundaryCheckFunctor(double dt, double fr)
        : diskTolerance(dt), farRadius(fr) {}
    
    inline bool operator()(const double* __restrict__ y) const noexcept{
        return (y[0] < diskTolerance) | (y[0] > farRadius);
    }
};

// ===== MAIN RAY TRACER CLASS =====
class RayTracer {
public:
    // Primary constructor using simplified Config objects
    RayTracer(const Config::BlackHole& bhConfig, const Config::Image& imgConfig)
        : bhConfig_(bhConfig)
        , imgConfig_(imgConfig)
        , cameraParams_(imgConfig.getCameraParams())
    {}

    // Traces ray from the observer at pixel (i, j)
    // Returns true if the ray hits the disk
    // and false if it never hits the disk
    bool traceRay(int i, int j, BoyerLindquistMetric& metric, DormandPrinceRK45& integrator,
                  double* y, double& intensity) {
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
    /**
     * @brief Setup initial conditions for the ray for each CPU thread
     * 
     * @param[in] i Pixel x-coordinate starting at the left
     * @param[in] j Pixel y-coordinate starting at the top
     * @param[out] ray Initializes photon position and momentum in spherical coordinates
     * @param metric Reference to the BoyerLindquistMetric object
     * 
     */
    void setupRayInitialConditions(int i, int j, double* ray, BoyerLindquistMetric& metric) {
        const double local_x_sc = cameraParams_.offsetX + (i * cameraParams_.stepX);
        const double local_y_sc = cameraParams_.offsetY - (j * cameraParams_.stepY);

        // Pre-calculate frequently used values
        const double D_squared = bhConfig_.distance * bhConfig_.distance;
        const double beta = local_x_sc / bhConfig_.distance;
        const double alpha = local_y_sc / bhConfig_.distance;
        const double cos_beta = std::cos(beta);
        const double sin_beta = std::sin(beta);
        const double cos_alpha = std::cos(alpha);
        const double sin_alpha = std::sin(alpha);
        
        // Set initial position
        ray[0] = std::sqrt(D_squared + (local_x_sc * local_x_sc) + (local_y_sc * local_y_sc));
        ray[1] = bhConfig_.theta - alpha;
        ray[2] = beta;
        
        // Compute metric at initial position
        metric.computeMetric(ray[0], ray[1]);

        // Set initial momentum
        ray[3] = -std::sqrt(metric.g_11) * cos_beta * cos_alpha;
        ray[4] = -std::sqrt(metric.g_22) * sin_alpha;
        ray[5] = std::sqrt(metric.g_33) * sin_beta * cos_alpha;
    }

    // Calculate intensity of a ray that hit the disk
    void calculateIntensity(const double* ray, const BoyerLindquistMetric& metric, double& intensity) {
        const double rf = ray[0];
        const double u_rf = -ray[3];
        const double u_thf = -ray[4];
        const double u_phif = -ray[5];

        // Calculate upper time component of 4-velocity
        const double u_uppertf = std::sqrt((metric.gamma11 * u_rf * u_rf) +
                            (metric.gamma22 * u_thf * u_thf) +
                            (metric.gamma33 * u_phif * u_phif)) /
                        metric.alpha;
                        
        // Calculate lower time component
        const double u_lower_tf =
            (-metric.alpha * metric.alpha * u_uppertf) +
            (u_phif * metric.beta3);
        
        // Calculate angular velocity of matter in accretion disk
        const double omega = 1.0 / (bhConfig_.spin + 
                                (std::pow(rf, 3.0 / 2.0) / std::sqrt(bhConfig_.mass)));
    
        // Calculate redshift factor                
        const double oneplusz = (1.0 + (omega * u_phif / u_lower_tf))
                / std::sqrt(-metric.g_00 - (omega * omega * metric.g_33) -
                (2 * omega * metric.g_03));

        // Lorentz invariant intensity I_em / (1 + z)^3
        intensity = 1.0 / (oneplusz * oneplusz * oneplusz);
    }

    // Store the configuration objects
    Config::BlackHole bhConfig_;
    Config::Image imgConfig_;
    Config::Image::CameraParams cameraParams_;
};

#endif // COSMIC_RAY_TRACER_H
