#ifndef COSMIC_CONFIG_H
#define COSMIC_CONFIG_H

#include <cmath>
#include <string>

#include "constants.h"

// ===== BLACK HOLE CONFIG =====
class BlackHole {
  // Black hole parameters
  float m_spin{Constants::BlackHole::BH_SPIN};
  float m_mass{Constants::BlackHole::BH_MASS};
  float m_distance{Constants::BlackHole::OBSERVER_DISTANCE};
  float m_theta{Constants::BlackHole::OBSERVER_THETA};
  float m_phi{Constants::BlackHole::OBSERVER_PHI};

  // Accretion disk parameters
  float m_inner_radius{Constants::BlackHole::DISK_INNER_RADIUS};
  float m_outer_radius{Constants::BlackHole::DISK_OUTER_RADIUS};
  float m_far_radius{Constants::BlackHole::FAR_RADIUS};
  float m_disk_tolerance;

 public:
  BlackHole() { updateDiskTolerance(); }

  BlackHole(float spin, float mass, float obs_distance, float theta, float phi,
            float inner_radius, float outer_radius, float far_radius)
      : m_spin{spin},
        m_mass{mass},
        m_distance{obs_distance},
        m_theta{theta},
        m_phi{phi},
        m_inner_radius{inner_radius},
        m_outer_radius{outer_radius},
        m_far_radius{far_radius} {
    updateDiskTolerance();
  }

  void updateDiskTolerance() {
    m_disk_tolerance =
        1.01f * (m_mass + std::sqrtf(m_mass * m_mass - m_spin * m_spin));
  }
  // Converts degrees to radians for theta
  void setObserverAngle(float angleDegrees) {
    m_theta = angleDegrees * Constants::DEG_TO_RAD;
  }
  float spin() const { return m_spin; }
  float mass() const { return m_mass; }
  float distance() const { return m_distance; }
  float theta() const { return m_theta; }
  float phi() const { return m_phi; }
  float innerRadius() const { return m_inner_radius; }
  float outerRadius() const { return m_outer_radius; }
  float farRadius() const { return m_far_radius; } 
  float diskTolerance() const { return m_disk_tolerance; }
};

// ===== IMAGE CONFIG =====
class Image {
  // Image dimensions and scaling
  float m_aspect_width{Constants::Image::ASPECT_WIDTH};
  float m_aspectHeight{Constants::Image::ASPECT_HEIGHT};
  float m_scale{Constants::Image::IMAGE_SCALE};
  float m_pixelWidth;
  float m_pixelHeight;
  float m_viewportWidth;
  float m_viewportHeight;
  float m_stepX;
  float m_stepY;
  float m_offsetX;
  float m_offsetY;

 public:
  Image() { updateImageParams(); }

  void updateImageParams() {
    // Orthographic projection with disk rad 20 => viewportWidth ~ 40
    // stepX = 40 / pixel_width where 40 = cameraScale * aspectWidth = 2.5 *
    // 16
    m_pixelWidth = m_aspect_width * m_scale;
    m_pixelHeight = m_aspectHeight * m_scale;
    m_viewportWidth = 40.0f * 1.05f;
    m_viewportHeight = m_viewportWidth * m_pixelHeight / m_pixelWidth;
    m_stepX = m_viewportWidth / (m_aspect_width * m_scale);
    m_stepY = m_viewportHeight / (m_aspectHeight * m_scale);
    m_offsetX = -m_viewportWidth * 0.5f + m_stepX * 0.5f;
    m_offsetY = m_viewportHeight * 0.5f; // - m_stepY * 0.5f;
  }

  float aspectWidth() const { return m_aspect_width; }
  float aspectHeight() const { return m_aspectHeight; }
  float scale() const { return m_scale; } 
  int width() const { return static_cast<int>(m_aspect_width * m_scale); }
  int height() const { return static_cast<int>(m_aspectHeight * m_scale); }
  int numPixels() const { return static_cast<int>(width() * height()); }
  float aspectRatio() const { return m_aspect_width / m_aspectHeight; }
  float stepX() const { return m_stepX; }
  float stepY() const { return m_stepY; }
  float offsetX() const { return m_offsetX; }
  float offsetY() const { return m_offsetY; }
};

// ===== OUTPUT CONFIG =====
// Simplified output configuration - direct initialization
class OutputConfig {
  std::string m_outputDir{Constants::Output::DEFAULT_OUTPUT_DIR};
  std::string m_baseFilename{Constants::Output::DEFAULT_FILENAME};
  std::string m_fileFormat{Constants::Output::DEFAULT_FORMAT};

  public:
  OutputConfig() = default;

  // Constructor with default initialization
  OutputConfig(const std::string& dir, const std::string& filename,
               const std::string& format)
      : m_outputDir(dir), m_baseFilename(filename), m_fileFormat(format) {}

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
  void setDescriptiveFilename(BlackHole& bh, const Image& img,
                              const std::string& prefix = "bh") {
    m_baseFilename =
        prefix + "_spin" + formatfloat(bh.spin()) + "_mass" +
        formatfloat(bh.mass()) + "_dist" + formatfloat(bh.distance()) +
        "_theta" + formatfloat(bh.theta() * Constants::RAD_TO_DEG) + "_phi" +
        formatfloat(bh.phi() * Constants::RAD_TO_DEG) + "_" +
        std::to_string(img.width()) + "x" + std::to_string(img.height());
  }

  // Get full path to output file
  std::string getFullPath() const {
    std::string dir = m_outputDir;
    // Ensure directory path ends with slash
    if (!dir.empty() && dir.back() != '/') {
      dir += '/';
    }
    return dir + m_baseFilename + "." + m_fileFormat;
  }

  // Return path components individually
  const std::string& getDirectory() const { return m_outputDir; }
  const std::string& getFilename() const { return m_baseFilename; }
  const std::string& getFormat() const { return m_fileFormat; }
};
#endif  // COSMIC_CONFIG_H
