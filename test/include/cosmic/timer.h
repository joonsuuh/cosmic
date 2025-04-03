#ifndef COSMIC_TIMER_H
#define COSMIC_TIMER_H

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

/**
 * Measuring execution time with cool output
 */ 
class Timer {
public:
    /**
     * @brief Default constructor
     * Initializes the Timer with the current time and a default message "COMPUTING"
     */
    Timer() 
        : start_time{std::chrono::high_resolution_clock::now()}
        , end_time{start_time}
        , message{"COMPUTING"}
    {}

    /**
     * Start timer and display message
     * @param msg Message to display
     */
    void start(const char* msg) {
        message = msg;
        std::cout << formatMessage(message) << "\n"
                  << formatTag("START", Colors::GREEN) << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * Stop timing and display the elapsed time
     */
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
    
    // Buffer size for duration formatting
    // MAXIMUM TIME: 999,999,999.999 s 
    // - Stored as "999999999.999 s\0"
    // - 12 digits + 1 decimal point + 1 space + 1 unitChar + 1 null terminator
    static constexpr size_t DURATION_BUFFER_SIZE = 16; 

    
    /**
     * Apply color to text
     */
    static std::string colorize(const std::string& text, const char* color) {
        return std::string(color) + text + Colors::RESET;
    }

    /**
     * Format a tag like [TAG] with color
     */
    static std::string formatTag(const std::string& tag, const char* color) {
        return colorize("[" + tag + "]", color);
    }

    /**
     * Format input message with "▶ " prefix and color 
     */
    static std::string formatMessage(const char* msg) {
        return "▶ " + colorize(msg, Colors::CYAN);
    }
    
    /**
     * Format a duration into a human-readable string with 3 decimal places
     */
    static std::string formatDuration(const std::chrono::duration<double, std::nano>& duration) {
        const double nanoseconds = duration.count();
        const double milliseconds = nanoseconds / 1000000.0;
        const double seconds = milliseconds / 1000.0;
        
        char buffer[DURATION_BUFFER_SIZE]{};
        
        if (seconds >= 1.0) {
            snprintf(buffer, sizeof(buffer), "%.3f s", seconds);
        } else if (milliseconds >= 1.0) {
            snprintf(buffer, sizeof(buffer), "%.3f ms", milliseconds);
        } else {
            snprintf(buffer, sizeof(buffer), "%d ns", static_cast<int>(nanoseconds));
        }
        
        return std::string{buffer};
    }
};
#endif
