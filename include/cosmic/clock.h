#ifndef COSMIC_CLOCK_H
#define COSMIC_CLOCK_H

#include <chrono>
#include <iostream>

class Clock {
public:
    Clock() {
        start = std::chrono::high_resolution_clock::now();
    }

    ~Clock() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "EXECUTION TIME: " << elapsed.count() << " SECONDS" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
#endif // COSMIC_CLOCK_H
