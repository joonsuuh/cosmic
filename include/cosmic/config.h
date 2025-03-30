#ifndef COSMIC_CONFIG_H
#define COSMIC_CONFIG_H

#include "constants.h"
#include <string>
#include <cmath>

// CMAKE DEFINES PPM_DIR 
#ifndef PPM_DIR
  #define PPM_DIR "data/" 
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
    double spin{Constants::BlackHole::DEFAULT_SPIN};
    double mass{Constants::BlackHole::DEFAULT_MASS};
    double distance{Constants::BlackHole::DEFAULT_DISTANCE};
    double theta{Constants::BlackHole::DEFAULT_OBSERVER_THETA};
    double phi{Constants::BlackHole::DEFAULT_OBSERVER_PHI};
    
    // Accretion disk parameters
    double innerRadius{Constants::BlackHole::DEFAULT_INNER_RADIUS};
    double outerRadius{Constants::BlackHole::DEFAULT_OUTER_RADIUS};
    double diskTolerance;
    double farRadius{Constants::BlackHole::DEFAULT_FAR_RADIUS};
    
    // ===== Constructors =====

    /**
     * @brief Default constructor
     * 
     * Initializes the black hole with default values and calculates the disk tolerance.
     * The disk tolerance is set to 1.01 times the radius of the event horizon.
     * The event horizon radius is calculated using the formula:
     */
    BlackHole() : 
        diskTolerance{1.01 * (mass + std::sqrt(mass * mass - spin * spin))} 
    {
    }
    
    // Add custom spin
    BlackHole(double spinParam) : 
        spin{spinParam},
        diskTolerance{1.01 * (1.0 + std::sqrt(1.0 - spinParam * spinParam))}
    {
    }
    
    // Converts degrees to radians for theta
    void setObserverAngle(double angleDegrees) {
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
    double cameraScale{Constants::Image::DEFAULT_CAMERA_SCALE};
    
    // Calculated properties
    int width() const { return aspectWidth * scale; }
    int height() const { return aspectHeight * scale; }
    int numPixels() const { return width() * height(); }
    double aspectRatio() const { return static_cast<double>(aspectWidth) / aspectHeight; }
    
    // Camera parameters for ray tracing
    /**
     * @struct CameraParams
     * @brief Camera parameters for ray tracing
     * 
     * 
     */
    struct CameraParams {
        double offsetX;
        double offsetY;
        double stepX;
        double stepY;
    };
    
    // Calculate camera parameters for ray tracing
    CameraParams getCameraParams() const {
        CameraParams params;
        
        double aspectWidthD = static_cast<double>(aspectWidth);
        double aspectHeightD = static_cast<double>(aspectHeight);
        
        params.offsetX = -aspectWidthD * cameraScale + (aspectWidthD * cameraScale / width());
        params.offsetY = aspectHeightD * cameraScale;
        params.stepX = 2.0 * aspectWidthD * cameraScale / width();
        params.stepY = 2.0 * aspectHeightD * cameraScale / height();
        
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
    std::string outputDir{PPM_DIR};
    std::string baseFilename{"blackhole"};
    std::string fileFormat{"ppm"};
    
    // Default constructor
    OutputConfig() = default;
    
    // Constructor with default initialization
    OutputConfig(
        const std::string& dir,
        const std::string& filename,
        const std::string& format
    ) : outputDir(dir), baseFilename(filename), fileFormat(format) {}
    
    // Remove trailing zeros from double to string representation
    static std::string formatDouble(double value) {
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
        baseFilename = prefix + "_spin" + formatDouble(bh.spin) +
                       "_mass" + formatDouble(bh.mass) +
                       "_dist" + formatDouble(bh.distance) +
                       "_theta" + formatDouble(bh.theta * Constants::RAD_TO_DEG) +
                       "_phi" + formatDouble(bh.phi * Constants::RAD_TO_DEG) +
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
