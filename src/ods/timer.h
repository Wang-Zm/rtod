// High-precision wall-clock timer for per-phase profiling.
// Usage: timer.startTimer(&timer.field); ... timer.stopTimer(&timer.field);
// After all slides: timer.average(n); timer.showTimeNew();

#include <sys/time.h>
#include <iostream>

class Timer {
public:
    double copy_new_points_h2d;
    double copy_filtered_points_h2d; // points for casting rays or building the BVH tree
    double copy_outlier_d2h;

    double build_bvh;
    double prepare_cell;
    double detect_outlier;

    double total;

    Timer() {
        struct timeval t1;
        gettimeofday(&t1, NULL);
        timebase = t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0;
        clearNew();
    }

    void clearNew() {
        copy_new_points_h2d = 0;
        copy_filtered_points_h2d = 0;
        copy_outlier_d2h = 0;
        prepare_cell = 0;
        build_bvh = 0;
        detect_outlier = 0;
        total = 0;
    }

    void startTimer(double *t) {
        struct timeval t1;
        gettimeofday(&t1, NULL);
        *t -= (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
    }

    void stopTimer(double *t) {
        struct timeval t1;
        gettimeofday(&t1, NULL);
        *t += (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
    }

    void average(int n) {
        copy_new_points_h2d /= n;
        copy_filtered_points_h2d /= n;
        copy_outlier_d2h /= n;
        prepare_cell /= n;
        build_bvh /= n;
        detect_outlier /= n;
        total /= n;
    }

    void showTimeNew() {
        std::cout << std::endl;
        std::cout << "###########   Time  ##########" << std::endl;
        std::cout << "[Time] copy new points h2d: " << copy_new_points_h2d << " ms" << std::endl;
        std::cout << "[Time] copy filtered points h2d: " << copy_filtered_points_h2d << " ms" << std::endl;
        std::cout << "[Time] copy outlier d2h: " << copy_outlier_d2h << " ms" << std::endl;
        std::cout << "[Time] prepare cell: " << prepare_cell << " ms" << std::endl;
        std::cout << "[Time] build BVH: " << build_bvh << " ms" << std::endl;
        std::cout << "[Time] detect outlier: " << detect_outlier << " ms" << std::endl;
        std::cout << "[Time] total time for a slide: " << total << " ms" << std::endl;
        std::cout << "##############################" << std::endl;
        std::cout << std::endl;
    }

private:
    double timebase;
};