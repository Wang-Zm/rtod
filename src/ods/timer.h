// High-precision wall-clock timer for per-phase profiling.
// Usage:  timer.start(TIMER_copy_new_points_h2d);
//         timer.stop(TIMER_copy_new_points_h2d);
// After:  timer.average(n);  timer.print();
//
// To add a new timing phase, add one enum value and one name string below.

#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
#include <iostream>

enum TimingPhase {
    TIMER_copy_new_points_h2d,
    TIMER_copy_filtered_points_h2d,
    TIMER_copy_outlier_d2h,
    TIMER_prepare_cell,
    TIMER_build_bvh,
    TIMER_detect_outlier,
    TIMER_total,
    TIMER_COUNT
};

constexpr const char* kPhaseNames[TIMER_COUNT] = {
    "copy new points h2d",
    "copy filtered points h2d",
    "copy outlier d2h",
    "prepare cell",
    "build BVH",
    "detect outlier",
    "total time for a slide",
};

class Timer {
public:
    Timer() {
        struct timeval t1;
        gettimeofday(&t1, NULL);
        timebase = t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0;
        clear();
    }

    void clear() {
        for (int i = 0; i < TIMER_COUNT; i++) times[i] = 0;
    }

    void start(TimingPhase p) {
        struct timeval t1;
        gettimeofday(&t1, NULL);
        times[p] -= (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
    }

    void stop(TimingPhase p) {
        struct timeval t1;
        gettimeofday(&t1, NULL);
        times[p] += (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
    }

    void average(int n) {
        for (int i = 0; i < TIMER_COUNT; i++) times[i] /= n;
    }

    void print() {
        std::cout << std::endl;
        std::cout << "###########   Time  ##########" << std::endl;
        for (int i = 0; i < TIMER_COUNT; i++)
            std::cout << "[Time] " << kPhaseNames[i] << ": " << times[i] << " ms" << std::endl;
        std::cout << "##############################" << std::endl;
        std::cout << std::endl;
    }

private:
    double times[TIMER_COUNT] = {0};
    double timebase;
};

#endif