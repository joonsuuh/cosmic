#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

// Measure time with cool output
class Timer {
public:
    // Initializes the Timer with the current time and a default message "COMPUTING" 
    Timer() 
        : start_time{std::chrono::high_resolution_clock::now()}
        , end_time{start_time}
        , message{"COMPUTING"}
    {}

    void start(const char* msg) {
        message = msg;
        std::cout << formatMessage(message) << "\n"
                  << formatTag("START", Colors::GREEN) << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    // stops timer and prints the elapsed time
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::nano>(end_time - start_time);
        
        std::cout << formatTag("TIME", Colors::YELLOW) << " " << formatDuration(duration) << "\n"
                  << formatTag("END", Colors::RED) << std::endl;
    }

private:
    // ===== Timer state =====
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    const char* message;

    // ===== Terminal colors =====
    struct Colors {
        static constexpr const char* RESET = "\033[0m";
        static constexpr const char* RED = "\033[1;31m";
        static constexpr const char* GREEN = "\033[1;32m";
        static constexpr const char* CYAN = "\033[1;36m";
        static constexpr const char* YELLOW = "\033[1;33m";
    };

    // ===== Output formatting =====
    // Apply color to text
    static std::string colorize(const std::string& text, const char* color) {
        return std::string(color) + text + Colors::RESET;
    }

    // Format input message with color
    static std::string formatTag(const std::string& tag, const char* color) {
        return colorize("[" + tag + "]", color);
    }

    // Format with ▶ prefix and color 
    static std::string formatMessage(const char* msg) {
        return "▶ " + colorize(msg, Colors::CYAN);
    }
    
    // Format a duration into a human-readable string with 3 decimal places
    static std::string formatDuration(const std::chrono::duration<double, std::nano>& duration) {
        const double nanoseconds = duration.count();
        const double milliseconds = nanoseconds / 1000000.0;
        const double seconds = milliseconds / 1000.0;
        
        std::ostringstream oss;
        
        if (seconds >= 1.0) {
            oss << std::fixed << std::setprecision(3) << seconds << " s";
        } else if (milliseconds >= 1.0) {
            oss << std::fixed << std::setprecision(3) << milliseconds << " ms";
        } else {
            oss << static_cast<int>(nanoseconds) << " ns";
        }
        
        return oss.str();
    }
};
#endif
