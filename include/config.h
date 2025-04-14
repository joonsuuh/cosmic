#ifndef COSMIC_CONFIG_H
#define COSMIC_CONFIG_H

#include "constants.h"
#include <string>
#include <cmath>

namespace Config {

// ===== BLACK HOLE CONFIGURATION =====
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
struct Image {
    // Image dimensions and scaling
    float aspectWidth{Constants::Image::DEFAULT_ASPECT_WIDTH};
    float aspectHeight{Constants::Image::DEFAULT_ASPECT_HEIGHT};
    float scale{Constants::Image::DEFAULT_IMAGE_SCALE};
    float cameraScale{Constants::Image::DEFAULT_CAMERA_SCALE};
    
    // Camera parameters as direct members with inline calculation
    float offsetX = -cameraScale * aspectWidth * (1.0f - 1.0f/static_cast<float>(static_cast<int>(aspectWidth * scale)));
    float offsetY = cameraScale * aspectHeight;
    float stepX = 2.0f * cameraScale * aspectWidth / static_cast<float>(static_cast<int>(aspectWidth * scale));
    float stepY = 2.0f * cameraScale * aspectHeight / static_cast<float>(static_cast<int>(aspectHeight * scale));
    
    int width() const { return static_cast<int>(aspectWidth * scale); }
    int height() const { return static_cast<int>(aspectHeight * scale); }
    int numPixels() const { return static_cast<int>(width() * height()); }
    float aspectRatio() const { return static_cast<float>(aspectWidth) / aspectHeight; }
    
    void setAspectRatio(int width, int height) {
        aspectWidth = width;
        aspectHeight = height;
        
        // Recalculate camera parameters inline
        offsetX = -cameraScale * aspectWidth * (1.0f - 1.0f/this->width());
        offsetY = cameraScale * aspectHeight;
        stepX = 2.0f * cameraScale * aspectWidth / this->width();
        stepY = 2.0f * cameraScale * aspectHeight / this->height();
    }
    
    void setScale(float newScale) {
        scale = newScale;
        
        offsetX = -cameraScale * aspectWidth * (1.0f - 1.0f/width());
        offsetY = cameraScale * aspectHeight;
        stepX = 2.0f * cameraScale * aspectWidth / width();
        stepY = 2.0f * cameraScale * aspectHeight / height();
    }
};

// ===== OUTPUT CONFIGURATION =====
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
