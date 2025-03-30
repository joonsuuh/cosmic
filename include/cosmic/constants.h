#ifndef COSMIC_CONSTANTS_H
#define COSMIC_CONSTANTS_H

#include <cmath>

// CMAKE DEFINES PPM_DIR 
#ifndef PPM_DIR
  #define PPM_DIR "data/" // for compiling through command line ./compile.sh
#endif

namespace Constants {
    // Math constants
    constexpr double PI = M_PI;
    constexpr double HALF_PI = M_PI / 2.0;
    constexpr double TWO_PI = 2.0 * M_PI;
    constexpr double DEG_TO_RAD = M_PI / 180.0;
    constexpr double RAD_TO_DEG = 180.0 / M_PI;
    
    // Black hole parameters
    namespace BlackHole {
        constexpr double DEFAULT_SPIN = 0.99;
        constexpr double DEFAULT_MASS = 1.0;
        constexpr double DEFAULT_DISTANCE = 500.0;
        constexpr double DEFAULT_OBSERVER_THETA = 85.0 * DEG_TO_RAD;
        constexpr double DEFAULT_OBSERVER_PHI = 0.0;
        constexpr double DEFAULT_INNER_RADIUS = 5.0;
        constexpr double DEFAULT_OUTER_RADIUS = 20.0;
        constexpr double DEFAULT_FAR_RADIUS = 1000.0;
    }
    
    // Image parameters
    namespace Image {
        constexpr int DEFAULT_ASPECT_WIDTH = 16;
        constexpr int DEFAULT_ASPECT_HEIGHT = 9;
        constexpr int DEFAULT_IMAGE_SCALE = 10;
        constexpr double DEFAULT_CAMERA_SCALE = 1.5;
    }
    
    // Integration parameters
    namespace Integration {
        constexpr double ABS_TOLERANCE = 1.0e-12;
        constexpr double REL_TOLERANCE = 1.0e-12;
        constexpr double DISK_TOLERANCE = 0.01;
        constexpr double INITIAL_STEP_SIZE = 0.1;
        constexpr double MIN_STEP_SIZE = 1.0e-10;
        constexpr double MAX_STEP_SIZE = 1.0;
    }
    
    // Image output parameters
    namespace ColorMap {
        constexpr double RED_THRESHOLD = 0.365079;
        constexpr double YELLOW_THRESHOLD = 0.746032;
    }

    // Output default parameters
    namespace Output {
        constexpr const char* DEFAULT_OUTPUT_DIR = PPM_DIR;
        constexpr const char* DEFAULT_FILENAME = "bh_cpu";
        constexpr const char* DEFAULT_FORMAT = "ppm";
    }
}

#endif // COSMIC_CONSTANTS_H
