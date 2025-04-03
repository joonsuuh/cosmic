#ifndef COSMIC_CONSTANTS_H
#define COSMIC_CONSTANTS_H

#include <cmath>

// CMAKE DEFINES PPM_DIR 
#ifndef PPM_DIR
  #define PPM_DIR "data/" // for compiling through command line ./compile.sh
#endif

namespace Constants {
    // Math constants
    constexpr float PI = M_PI;
    constexpr float HALF_PI = M_PI / 2.0;
    constexpr float TWO_PI = 2.0 * M_PI;
    constexpr float DEG_TO_RAD = M_PI / 180.0;
    constexpr float RAD_TO_DEG = 180.0 / M_PI;
    
    // Black hole parameters
    namespace BlackHole {
        constexpr float DEFAULT_SPIN = 0.99f;
        constexpr float DEFAULT_MASS = 1.0f;
        constexpr float DEFAULT_DISTANCE = 500.0f;
        constexpr float DEFAULT_OBSERVER_THETA = 85.0f * DEG_TO_RAD;
        constexpr float DEFAULT_OBSERVER_PHI = 0.0f;
        constexpr float DEFAULT_INNER_RADIUS = 5.0f;
        constexpr float DEFAULT_OUTER_RADIUS = 20.0f;
        constexpr float DEFAULT_FAR_RADIUS = 600.0f;
    }
    
    // Image parameters
    namespace Image {
        constexpr int DEFAULT_ASPECT_WIDTH = 16;
        constexpr int DEFAULT_ASPECT_HEIGHT = 9;
        constexpr int DEFAULT_IMAGE_SCALE = 10;
        constexpr float DEFAULT_CAMERA_SCALE = 1.5f;
    }
    
    // Integration parameters
    namespace Integration {
        constexpr float ABS_TOLERANCE = 1.0e-8f;
        constexpr float REL_TOLERANCE = 1.0e-4f;
        constexpr float DISK_TOLERANCE = 0.01f;
        constexpr float INITIAL_STEP_SIZE = 0.1f;
        constexpr float MIN_STEP_SIZE = 1.0e-8f;
        constexpr float MAX_STEP_SIZE = 0.1f;
        constexpr int MAX_ITERATIONS = 10000;
    }
    
    // Image output parameters
    namespace ColorMap {
        constexpr float RED_THRESHOLD = 0.365079f;
        constexpr float YELLOW_THRESHOLD = 0.746032f;
    }

    // Output default parameters
    namespace Output {
        constexpr const char* DEFAULT_OUTPUT_DIR = PPM_DIR;
        constexpr const char* DEFAULT_FILENAME = "bh_cpu";
        constexpr const char* DEFAULT_FORMAT = "ppm";
    }
}

#endif // COSMIC_CONSTANTS_H
