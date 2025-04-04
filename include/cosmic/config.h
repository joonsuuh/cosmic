#ifndef COSMIC_CONFIG_H
#define COSMIC_CONFIG_H

#include "constants.h"
#include <string>
#include <cmath>

// Add CUDA compatibility macros
#ifdef __CUDA_ARCH__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace Config {

// ===== BLACK HOLE CONFIGURATION =====

/**
 * @struct BlackHole
 * @brief Black hole parameters
 * 
 * Initializes Default spin, mass, approximate accretion disk at @f$ \theta = \pi / 2 @f$,
 * 
 * and observer position in spherical coordinates.
 * @see Constants::BlackHole for default values
 */
struct BlackHole {
    // Black hole parameters
    float spin{Constants::BlackHole::DEFAULT_SPIN};
    float mass{Constants::BlackHole::DEFAULT_MASS};
    float distance{Constants::BlackHole::DEFAULT_DISTANCE};
    float theta{Constants::BlackHole::DEFAULT_OBSERVER_THETA};
    float phi{Constants::BlackHole::DEFAULT_OBSERVER_PHI};
    
    // Accretion disk parameters
    float innerRadius{Constants::BlackHole::DEFAULT_INNER_RADIUS};
    float outerRadius{Constants::BlackHole::DEFAULT_OUTER_RADIUS};
    float diskTolerance;
    float farRadius{Constants::BlackHole::DEFAULT_FAR_RADIUS};
    
    // ===== Constructors =====

    /**
     * @brief Default constructor
     * 
     * Initializes the black hole with default values and calculates the disk tolerance.
     * The disk tolerance is set to 1.01 times the radius of the event horizon.
     * The event horizon radius is calculated using the formula:
     */
    BlackHole() : 
        diskTolerance{static_cast<float>(1.01f * (mass + std::sqrtf(mass * mass - spin * spin)))} 
    {}
    
    // Add custom spin
    BlackHole(float spinParam) : 
        spin{spinParam},
        diskTolerance{static_cast<float>(1.01f * (1.0f + std::sqrtf(1.0f - spinParam * spinParam)))}
    {}
    
    // Converts degrees to radians for theta
    void setObserverAngle(float angleDegrees) {
        theta = angleDegrees * Constants::DEG_TO_RAD;
    }
};

// ===== IMAGE CONFIGURATION =====
/**
 * @struct Image
 * @brief Image parameters for ray tracing
 * 
 * Initializes default image dimensions and scaling factors.
 * 
 * @see Constants::Image for default values
 */
struct Image {
    // Image dimensions and scaling
    int aspectWidth{Constants::Image::DEFAULT_ASPECT_WIDTH};
    int aspectHeight{Constants::Image::DEFAULT_ASPECT_HEIGHT};
    int scale{Constants::Image::DEFAULT_IMAGE_SCALE};
    float cameraScale{Constants::Image::DEFAULT_CAMERA_SCALE};
    
    // Calculated properties - mark as CUDA callable
    CUDA_CALLABLE int width() const { return aspectWidth * scale; }
    CUDA_CALLABLE int height() const { return aspectHeight * scale; }
    CUDA_CALLABLE int numPixels() const { return width() * height(); }
    CUDA_CALLABLE float aspectRatio() const { return static_cast<float>(aspectWidth) / aspectHeight; }
    
    // Camera parameters for ray tracing
    struct CameraParams {
        float offsetX;
        float offsetY;
        float stepX;
        float stepY;
    };
    
    // Calculate camera parameters for ray tracing
    CameraParams getCameraParams() const {
        CameraParams params;
        
        float aspectWidthD = static_cast<float>(aspectWidth);
        float aspectHeightD = static_cast<float>(aspectHeight);
        
        params.offsetX = -aspectWidthD * cameraScale + (aspectWidthD * cameraScale / width());
        params.offsetY = aspectHeightD * cameraScale;
        params.stepX = 2.0f * aspectWidthD * cameraScale / width();
        params.stepY = 2.0f * aspectHeightD * cameraScale / height();
        
        return params;
    }
    
    // Convenience method to set aspect ratio
    void setAspectRatio(int width, int height) {
        aspectWidth = width;
        aspectHeight = height;
    }
};

// ===== OUTPUT CONFIGURATION =====
// Simplified output configuration - direct initialization
struct OutputConfig {
    std::string outputDir{Constants::Output::DEFAULT_OUTPUT_DIR};
    std::string baseFilename{Constants::Output::DEFAULT_FILENAME};
    std::string fileFormat{Constants::Output::DEFAULT_FORMAT};
    
    // Default constructor
    OutputConfig() = default;
    
    // Constructor with default initialization
    OutputConfig(
        const std::string& dir,
        const std::string& filename,
        const std::string& format
    ) : outputDir(dir), baseFilename(filename), fileFormat(format) {}
    
    // Remove trailing zeros from float to string representation
    static std::string formatfloat(float value) {
        std::string str = std::to_string(value);
        // Remove trailing zeros
        str.erase(str.find_last_not_of('0') + 1, std::string::npos);
        // Remove decimal point if it's the last character
        if (str.back() == '.') {
            str.pop_back();
        }
        return str;
    }
    
    // Generate descriptive filename based on simulation parameters
    void setDescriptiveFilename(const BlackHole& bh, const Image& img, const std::string& prefix = "bh") {
        baseFilename = prefix + "_spin" + formatfloat(bh.spin) +
                       "_mass" + formatfloat(bh.mass) +
                       "_dist" + formatfloat(bh.distance) +
                       "_theta" + formatfloat(bh.theta * Constants::RAD_TO_DEG) +
                       "_phi" + formatfloat(bh.phi * Constants::RAD_TO_DEG) +
                       "_" + std::to_string(img.width()) +
                       "x" + std::to_string(img.height());
    }
    
    // Get full path to output file
    std::string getFullPath() const {
        std::string dir = outputDir;
        // Ensure directory path ends with slash
        if (!dir.empty() && dir.back() != '/') {
            dir += '/';
        }
        return dir + baseFilename + "." + fileFormat;
    }
    
    // Return path components individually
    const std::string& getDirectory() const { return outputDir; }
    const std::string& getFilename() const { return baseFilename; }
    const std::string& getFormat() const { return fileFormat; }
};
}

#endif // COSMIC_CONFIG_H
