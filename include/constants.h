#ifndef COSMIC_CONSTANTS_H
#define COSMIC_CONSTANTS_H

// CMAKE DEFINES PPM_DIR 
#ifndef PPM_DIR
  #define PPM_DIR "data/" // for compiling through command line ./compile.sh
#endif

namespace Constants {
    // Math constants
    constexpr float PI = M_PI;
    constexpr float HALF_PI = M_PI / 2.0;
    constexpr float DEG_TO_RAD = M_PI / 180.0;
    constexpr float RAD_TO_DEG = 180.0 / M_PI;
    
    // Black hole parameters
    namespace BlackHole {
        constexpr float BH_SPIN = 0.99f;
        constexpr float BH_MASS = 1.0f;
        constexpr float OBSERVER_DISTANCE = 500.0f;
        constexpr float OBSERVER_THETA = 85.0f * DEG_TO_RAD;
        constexpr float OBSERVER_PHI = 0.0f;
        constexpr float DISK_INNER_RADIUS = 5.0f;
        constexpr float DISK_OUTER_RADIUS = 20.0f;
        constexpr float FAR_RADIUS = 600.0f;
    }
    
    // Image parameters
    namespace Image {
        constexpr float ASPECT_WIDTH = 16.0f;
        constexpr float ASPECT_HEIGHT = 9.0f;
        constexpr float IMAGE_SCALE = 10.0f;
        // constexpr float DEFAULT_CAMERA_SCALE = 2.0f;
    }
    
    // Integration parameters
    namespace Integration {
        constexpr float ABS_TOLERANCE = 1.0e-8f;
        constexpr float REL_TOLERANCE = 1.0e-4f;
        constexpr float MIN_STEP_SIZE = 1.0e-8f;
        constexpr float MAX_STEP_SIZE = 0.1f;
        constexpr float INITIAL_STEP_SIZE = 0.1f;
        constexpr float DISK_TOLERANCE = 0.01f;
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
        constexpr const char* DEFAULT_FILENAME = "cosmic";
        constexpr const char* DEFAULT_FORMAT = "ppm";
    }
}

#endif // COSMIC_CONSTANTS_H
