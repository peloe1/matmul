#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>

class WallTimer {
private:
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock> start;
    const char* name;

public:
    inline WallTimer(const char* name) : name(name) {
        start = Clock::now();
    }

    inline ~WallTimer() {
        auto end = Clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << name << " total time: " << duration << " ms" << std::endl;
    }
};

#endif // TIMER_H