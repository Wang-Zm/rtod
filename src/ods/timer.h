// time count part
// semaphore
#include <mutex>
#include <sys/time.h>
#include <iostream>

using namespace std;

class Timer{
  public:

  double time[30];
  mutex timeMutex[30];
  double timebase;

  double copy_new_points_h2d;
  double copy_filtered_points_h2d; // points for casting rays or building the BVH tree
  double copy_outlier_d2h;

  double build_bvh;
  double prepare_cell;
  double detect_outlier;

  double total;

  Timer() {
    for (int i = 0; i < 30; i++) {
      time[i] = 0.0;
    }
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    timebase = t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0;
    clearNew();
  }

  // void clear() {
  //   for (int i = 0; i < 30; i++) {
  //     time[i] = 0.0;
  //   }
  // }

  // void clear(int timeId) {
  //   time[timeId] = 0.0;
  // }

  void clearNew() {
    copy_new_points_h2d = 0;
    copy_filtered_points_h2d = 0;
    copy_outlier_d2h = 0;
    prepare_cell = 0;
    build_bvh = 0;
    detect_outlier = 0;
    total = 0;
  }
  
  // void commonGetStartTime(int timeId) {
  //   struct timeval t1;                           
  //   gettimeofday(&t1, NULL);
  //   lock_guard<mutex> lock(timeMutex[timeId]);
  //   time[timeId] -= (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  // }

  // void commonGetEndTime(int timeId) {
  //   struct timeval t1;                           
  //   gettimeofday(&t1, NULL);
  //   lock_guard<mutex> lock(timeMutex[timeId]);
  //   time[timeId] += (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  // }

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

  // void startTimer(struct timeval* t) {
  //   gettimeofday(t, NULL);
  // }

  // void stopTimer(struct timeval* t, double* elapsed_time) {
  //   struct timeval end;
  //   gettimeofday(&end, NULL);
  //   *elapsed_time += (end.tv_sec - t->tv_sec) * 1000.0 + (end.tv_usec - t->tv_usec) / 1000.0;
  // }

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
    cout << endl;
    cout << "###########   Time  ##########" << endl;
    
    cout << "[Time] copy new points h2d: " << copy_new_points_h2d << " ms" << endl;
    cout << "[Time] copy filtered points h2d: " << copy_filtered_points_h2d << " ms" << endl;
    cout << "[Time] copy outlier d2h: " << copy_outlier_d2h << " ms" << endl;

    cout << "[Time] prepare cell: " << prepare_cell << " ms" << endl;
    cout << "[Time] build BVH: " << build_bvh << " ms" << endl;
    cout << "[Time] detect outlier: " << detect_outlier << " ms" << endl;
    cout << "[Time] total time for a slide: " << total << " ms" << endl;
    
    cout << "##############################" << endl;
    cout << endl;
  }

  // void showTime() {
  //   cout << endl;
  //   cout << "###########   Time  ##########" << endl;
  //   cout << "[Time] build BVH: ";
  //   cout << time[0] << " ms" << endl;

  //   cout << "[Time] initialize cell: ";
  //   cout << time[8] << " ms" << endl;

  //   cout << "[Time] expired points: ";
  //   cout << time[10] << " ms" << endl;
  //   cout << "[Time] new points: ";
  //   cout << time[11] << " ms" << endl;

  //   cout << "[Time] prepare cell: ";
  //   cout << time[9] << " ms" << endl;

  //   cout << "[Time] copy points in new slide: ";
  //   cout << time[4] << " ms" << endl;

  //   cout << "[Time] copy points casting ray/rebuilding BVH: ";
  //   cout << time[16] << " ms" << endl;

  //   cout << "[Time] overall update: ";
  //   cout << time[1] << " ms" << endl;

  //   // cout << "[Time] rebuild gas: ";
  //   // cout << time[22] << " ms" << endl;

  //   // cout << "[Time] time of prepare cell and rebuild BVH: ";
  //   // cout << time[14] << " ms" << endl;
    
  //   cout << "[Time] launch for sliding: ";
  //   cout << time[5] << " ms" << endl;

  //   cout << "[Time] transfer outliers back: ";
  //   cout << time[6] << " ms" << endl;
    
  //   cout << "[Time] total time for a slide: ";
  //   cout << time[7] << " ms" << endl;

  //   // cout << "[Time] clarify points: ";
  //   // cout << time[12] << " ms" << endl;

  //   // cout << "[Time] memcpy c_non points to device: ";
  //   // cout << time[13] << " ms" << endl;

  //   cout << "##############################" << endl;
  //   cout << endl;
  // }

  // void showTime(int tid, string description) {
  //   cout << "[Time] " << description << ": " << time[tid] << endl;
  // }

};
