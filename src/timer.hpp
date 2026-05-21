#pragma once
// =============================================================================
// timer.hpp  —  Temporizador de alta resolución (std::chrono)
// =============================================================================
#include <chrono>

class Timer {
public:
    void start() {
        t0 = std::chrono::high_resolution_clock::now();
    }
    double stopMs() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
private:
    std::chrono::high_resolution_clock::time_point t0;
};
